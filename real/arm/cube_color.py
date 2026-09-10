"""Кубик по ЦВЕТУ в камере кисти (настоящие зелёный и серый кубы без меток).

Для гибрида «политика подводит — алгоритм центрирует и берёт» на кубах без
ArUco: цветное пятно в кадре кисти -> контур -> лучи точек контура на плоскость
центра кубика (z = CUBE_HALF) через калибровку камеры кисти (T_gripper_wristcam
и FK) -> след кубика на столе -> центр и азимут граней по minAreaRect.

Смещение: в кадре виден верх и ближний бок кубика, поэтому центр следа сдвинут
к камере примерно на четверть ребра; поправка вычитается вдоль горизонтальной
проекции оси взгляда.

  from cube_color import cube_by_wrist_color
  xy, yaw, dbg = cube_by_wrist_color(d, "green", Xw)     # None, если пятна нет
"""
import cv2
import numpy as np

import autocalib_real as A

# HSV-диапазоны (OpenCV: H 0..180). Стол — светлое дерево (S низкая, V высокая),
# рука оранжевая (H 5–22). Серый: низкая насыщенность и V заметно ниже стола.
# замер 10.09 (верхняя камера, дневной свет): зелёный H79 S245 V125; серый выглядит
# серо-голубым H100 S117 V136; стол H36–87 S8–14 V155–160; тень S23; мышь V17
RANGES = {
    "green": [((55, 100, 50), (95, 255, 255))],
    "gray": [((88, 45, 70), (122, 200, 210))],
}
MIN_AREA, MAX_AREA = 1500, 120000       # px в кадре 640x480 с высоты 3–12 см
EDGE_M = 2 * A.CUBE_HALF


def color_mask(img_bgr, color):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    m = np.zeros(hsv.shape[:2], np.uint8)
    for lo, hi in RANGES[color]:
        m |= cv2.inRange(hsv, lo, hi)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    return m


def best_blob(mask):
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = None
    for c in cnts:
        area = cv2.contourArea(c)
        if not MIN_AREA <= area <= MAX_AREA:
            continue
        (cx, cy), (w, h), ang = cv2.minAreaRect(c)
        if min(w, h) < 15 or max(w, h) / max(min(w, h), 1) > 2.2:   # куб компактный
            continue
        fill = area / max(w * h, 1)
        if fill < 0.55:
            continue
        score = area * fill
        if best is None or score > best[0]:
            best = (score, c)
    return None if best is None else best[1]


def cube_by_wrist_color(d, color, Xw, frames=3):
    """(xy в базе, yaw, dbg) по цветному пятну в камере кисти; None, если не найден."""
    C = d.fk_T(d.read()[0][:5]) @ Xw               # камера кисти в базе
    hits = []
    dbg = {}
    for _ in range(frames):
        img = A.grab(A.CAM_WR)
        m = color_mask(img, color)
        c = best_blob(m)
        if c is None:
            continue
        pts = A.undistort(c.reshape(-1, 2).astype(np.float32), A.K_WR, A.D_WR)
        # лучи контура на плоскость z = CUBE_HALF
        Kinv = np.linalg.inv(A.K_WR)
        rays = (Kinv @ np.hstack([pts, np.ones((len(pts), 1))]).T).T
        R, t = C[:3, :3], C[:3, 3]
        dirs = rays @ R.T
        ok = dirs[:, 2] < -1e-4
        if ok.sum() < 8:
            continue
        s = (A.CUBE_HALF - t[2]) / dirs[ok, 2]
        foot = t[None, :2] + s[:, None] * dirs[ok, :2]
        (fx, fy), (fw, fh), fang = cv2.minAreaRect(foot.astype(np.float32))
        hits.append((fx, fy, np.radians(fang), max(fw, fh)))
        dbg = {"area_px": float(cv2.contourArea(c)), "foot_w_mm": round(max(fw, fh) * 1000, 1)}
    if len(hits) < 2:
        return None
    h = np.median(np.array(hits), axis=0)
    xy = h[:2].copy()
    # поправка: виден верх и ближний бок -> след вытянут к камере; сдвиг на четверть ребра
    view = C[:3, :3] @ np.array([0, 0, 1.0])
    horiz = view[:2] / (np.linalg.norm(view[:2]) + 1e-9)
    xy -= horiz * (EDGE_M / 4)
    dbg["hits"] = len(hits)
    return xy, float(h[2]), dbg


def cubes_by_top_color(Xt, colors=("green", "gray"), frames=3):
    """{color: xy в базе} по цветным пятнам в верхней камере (лежащие кубы).
    Пятно -> центр -> луч на плоскость z = CUBE_HALF. Берётся самый похожий на
    куб контур (компактный, 400–4000 px в кадре 1920x1080)."""
    out = {}
    for color in colors:
        pts = []
        for _ in range(frames):
            img = A.grab(A.CAM_TOP)
            m = color_mask(img, color)
            cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            best = None
            for c in cnts:
                area = cv2.contourArea(c)
                if not 2000 <= area <= 40000:      # куб ~115 px в кадре 1920x1080
                    continue
                (cx, cy), (w, h), _ = cv2.minAreaRect(c)
                if max(w, h) / max(min(w, h), 1) > 1.8 or area / max(w * h, 1) < 0.6:
                    continue
                px = A.undistort(np.array([[cx, cy]], np.float32), A.K_TOP, A.D_TOP)[0]
                ray = np.linalg.inv(A.K_TOP) @ np.array([px[0], px[1], 1.0])
                dirs = Xt[:3, :3] @ ray
                t = Xt[:3, 3]
                # центр пятна — примерно центр верхней грани (z = 2·CUBE_HALF) с небольшой долей бока
                s = (2 * A.CUBE_HALF - t[2]) / dirs[2]
                xy = t[:2] + s * dirs[:2]
                if not (0.10 <= xy[0] <= 0.36 and abs(xy[1]) <= 0.25):     # только зона стола перед рукой
                    continue
                if best is None or area > best[0]:
                    best = (area, xy)
            if best is None:
                continue
            pts.append(best[1])
        if len(pts) >= 2:
            out[color] = np.median(np.array(pts), axis=0)
    return out
