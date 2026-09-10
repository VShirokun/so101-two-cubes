#!/usr/bin/env python3
"""Интринсики камеры по нашим ArUco-объектам — без шахматной доски.

Объекты с точно известной геометрией уже напечатаны: кубы 28 мм (метки на
всех гранях, углы в системе куба известны) и коробка с меткой 60 мм в дне.
Двигайте куб по рабочей зоне и переворачивайте на разные грани; скрипт сам
принимает стабильные, достаточно отличающиеся виды и после N видов решает
cv2.calibrateCamera (3D-точки куба требуют стартового приближения — берётся
из грубого fov, флаг USE_INTRINSIC_GUESS).

Запуск (Терминал): python intrinsics_from_markers.py [--dir /tmp/roboom_cam]
                   [--out c920] [--views 45]
Выход: real/calib/<out>_intrinsics.json (K, dist, RMS).
"""

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parents[2] / "mlsim"))
from cube_cv import CUBE_IDS, DICT, MARKER_SIZE, _marker_T_cube

BOX_ID = 16
BOX_MARKER = 0.060


def object_points():
    """id метки -> (объект, 4 угла TL,TR,BR,BL в системе объекта)."""
    s = MARKER_SIZE / 2
    local = np.array([[-s, s, 0], [s, s, 0], [s, -s, 0], [-s, -s, 0]])
    out = {}
    for color, ids in CUBE_IDS.items():
        for fi, mid in enumerate(ids):
            T = _marker_T_cube(fi)
            out[mid] = (color, (T[:3, :3] @ local.T).T + T[:3, 3])
    b = BOX_MARKER / 2
    out[BOX_ID] = ("box", np.array([[-b, b, 0], [b, b, 0],
                                    [b, -b, 0], [-b, -b, 0]]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="/tmp/roboom_cam")
    ap.add_argument("--out", default="c920")
    ap.add_argument("--views", type=int, default=60)
    ap.add_argument("--f-expect", type=float, default=1360.0,
                    help="паспортное фокусное в px (C920 1080p ~1360-1400)")
    args = ap.parse_args()
    src = Path(args.dir) / "latest.jpg"
    objmap = object_points()
    det = cv2.aruco.ArucoDetector(DICT, cv2.aruco.DetectorParameters())

    views_obj, views_img = [], []
    last_centers = {}
    prev = {}
    last_m = 0.0
    size = None
    COLS, ROWS = 6, 4
    cover = np.zeros((ROWS, COLS), int)

    def empty_cells():
        names = []
        for r in range(ROWS):
            for c in range(COLS):
                if cover[r, c] == 0:
                    v = ["top", "top", "bottom", "bottom"][r]
                    hn = ["left", "left", "center", "center", "right", "right"][c]
                    names.append(f"{v}-{hn}")
        seen = []
        for n in names:
            if n not in seen:
                seen.append(n)
        return seen

    print(f"двигайте куб по ВСЕМУ кадру (можно в руке, на весу, с наклонами); "
          f"цель: {args.views}+ видов и покрытие всей сетки")
    while len(views_obj) < args.views or (cover == 0).sum() > 3:
        if len(views_obj) >= 120:
            print("достигнут предел 120 видов — завершаю с текущим покрытием")
            break
        try:
            m = src.stat().st_mtime
        except FileNotFoundError:
            time.sleep(0.2)
            continue
        if m == last_m:
            time.sleep(0.03)
            continue
        last_m = m
        img = cv2.imread(str(src))
        if img is None:
            continue
        size = (img.shape[1], img.shape[0])
        corners, ids, _ = det.detectMarkers(
            cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
        if ids is None:
            continue
        per_obj = {}
        for quad, mid in zip(corners, ids.ravel()):
            if int(mid) in objmap:
                obj, pts = objmap[int(mid)]
                per_obj.setdefault(obj, []).append(
                    (int(mid), pts, quad.reshape(4, 2)))
        for obj, dets in per_obj.items():
            imgpts = np.concatenate([q for _, _, q in dets]).astype(np.float32)
            objpts = np.concatenate([p for _, p, _ in dets]).astype(np.float32)
            marker_centers = {m: q.mean(0) for m, _, q in dets}
            center = imgpts.mean(0)
            # стабильность (два кадра подряд почти совпали — не смазан)
            key = obj
            pm = prev.get(key)
            common_p = set(pm) & set(marker_centers) if pm else set()
            ok_stable = bool(common_p) and all(
                np.linalg.norm(marker_centers[m] - pm[m]) < 1.5
                for m in common_p)
            prev[key] = marker_centers
            if not ok_stable:
                continue
            # новизна вида: только РЕАЛЬНЫЙ сдвиг объекта, по совпадающим
            # МЕТКАМ (ID к ID). Средний центр и число меток критериями быть
            # не могут: у неподвижного куба грань на границе детекции
            # мерцает, меняя и число меток, и «средний центр» (дважды
            # ловилось как самонабор видов без касания куба)
            lc = last_centers.get(key)
            if lc is not None:
                common = set(lc) & set(marker_centers)
                if common and max(np.linalg.norm(marker_centers[m] - lc[m])
                                  for m in common) < 40:
                    continue
            last_centers[key] = marker_centers
            views_obj.append(objpts)
            views_img.append(imgpts)
            for u, v in imgpts:
                cc = min(COLS - 1, int(u / size[0] * COLS))
                rr = min(ROWS - 1, int(v / size[1] * ROWS))
                cover[rr, cc] += 1
            need = empty_cells()
            hint = ("OK, can finish" if not need
                    else "empty: " + ", ".join(need[:4]))
            print(f"видов: {len(views_obj)}/{args.views} "
                  f"(+{obj}, меток {len(dets)}) | {hint}")
    f0 = 0.9 * size[0]
    K0 = np.array([[f0, 0, size[0] / 2], [0, f0, size[1] / 2], [0, 0, 1]])
    # k3 при неидеальном покрытии фитит мусор и коррелирует с фокусом
    flags = cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_FIX_K3
    rms, K, dist, _, _ = cv2.calibrateCamera(
        views_obj, views_img, size, K0, np.zeros(5), flags=flags)
    print(f"\nRMS {rms:.3f} px")
    fx = K[0, 0]
    if args.f_expect <= 0:
        print(f"fx={fx:.0f} (ожидание не задано — проверить физикой: "
              f"дистанция = fx * 21мм / сторона_метки_px)")
    elif abs(fx - args.f_expect) > 0.2 * args.f_expect:
        print(f"!!! fx={fx:.0f} далеко от паспорта ~{args.f_expect:.0f} — "
              f"калибровке НЕ ВЕРИТЬ, пересобрать с лучшим покрытием")
    else:
        print(f"fx={fx:.0f} согласуется с паспортом ~{args.f_expect:.0f} — похоже на правду")
    print("K =", np.round(K, 1).tolist())
    print("dist =", np.round(dist.ravel(), 4).tolist())
    out = Path(__file__).parent / f"{args.out}_intrinsics.json"
    out.write_text(json.dumps({
        "image_size": size, "rms_px": rms, "views": len(views_obj),
        "K": K.tolist(), "dist": dist.ravel().tolist(),
        "date": time.strftime("%Y-%m-%d %H:%M"),
    }, indent=2))
    print(f"сохранено: {out}")
    np.savez(Path(__file__).parent / f"{args.out}_views.npz",
             obj=np.array(views_obj, dtype=object),
             img=np.array(views_img, dtype=object),
             allow_pickle=True)
    cov_img = np.full((size[1] // 2, size[0] // 2, 3), 30, np.uint8)
    for ip in views_img:
        for u, v in ip:
            cv2.circle(cov_img, (int(u / 2), int(v / 2)), 3, (60, 220, 60), -1)
    cv2.imwrite(str(Path(__file__).parent / f"{args.out}_coverage.png"), cov_img)
    print(f"карта покрытия: {args.out}_coverage.png")
    if rms > 1.5:
        print("ВНИМАНИЕ: RMS великоват — стоит пересобрать спокойнее")


if __name__ == "__main__":
    main()
