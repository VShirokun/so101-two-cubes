"""Постпроцессинг датасета: перекраска ч/б ArUco-кубов в целевые цвета.

Вход — кадр камеры политики и позы кубов на этом кадре (сборщик логирует их
при записи: в симе — истина симулятора, в реале — CV по верхней камере для
лежащего куба и привязка к FK гриппера для несомого). Выход — кадр, где кубы
целевого цвета, метки закрашены, а всё остальное не тронуто ни на бит.

Как это работает, по шагам на куб:
1. Грани куба проецируются в кадр по позе куба и камеры; видимые дают
   геометрическую маску куба.
2. Внутри маски пиксели окклюдера (жёлтая рука) отличаются от ч/б куба по
   насыщенности и остаются нетронутыми — пальцы поверх куба сохраняются.
3. Чёрные клетки меток известны геометрически (паттерн каждой грани);
   их яркость заменяется яркостью белого этой грани — метки исчезают.
4. Освещение берётся из самого кадра: итоговый цвет = целевой RGB, умноженный
   на попиксельную яркость, нормированную на «белый» самой светлой грани.
   Тени от пальцев, шейдинг граней и рёбра куба сохраняются.
"""

import cv2
import numpy as np

from cube_cv import CUBE_HALF, CUBE_IDS, DICT, FACES, marker_grid

N = 8                 # клеток на грань (6 маркер + 2 белое поле)
SAT_OCCLUDER = 100    # насыщенность выше — чужой объект (рука, S~220), не куб
V_OCCLUDER = 60       # ...только у нетёмных пикселей...
HUE_OCCLUDER = (3, 35)  # ...и только оранжевого оттенка (рука, губки): тёмные клетки
                      # меток в реальной камере бывают синеватыми с S~130
BLACK_DILATE_FRAC = 0.12   # расширение маски чёрных клеток, доля стороны грани
                           # (~клетка: чёрное меток не должно пробиваться при
                           # ошибке позы в пару пикселей)
DEBUG = {}                 # маски последнего куба (для разбора кадров)
DARK_FRAC = 0.4            # темнее этой доли «белого» куба (и не ярче 60) — тёмное:
                           # чужой предмет, если пятно выходит за силуэт куба
                           # (корпус камеры на кисти, провод), иначе — метка


def cube_faces_bits(color: str) -> list[np.ndarray]:
    """Битовые сетки 8x8 всех граней куба (True = чёрная клетка)."""
    return [marker_grid(DICT, mid) for mid in CUBE_IDS[color]]


def _face_corners_local(fi: int) -> np.ndarray:
    """Углы грани fi в системе куба, порядок согласован с клеточной сеткой:
    (0,0) клеточного квадрата -> угол (-right, -up)."""
    _, right, up, normal = FACES[fi]
    right, up, normal = (np.asarray(v, float) for v in (right, up, normal))
    c = normal * CUBE_HALF
    h = CUBE_HALF
    return np.array([c - right * h - up * h, c + right * h - up * h,
                     c + right * h + up * h, c - right * h + up * h])


def recolor_frame(img: np.ndarray, K: np.ndarray, T_wc: np.ndarray,
                  cubes: list[dict], v_ref: float | None = None) -> np.ndarray:
    """img RGB uint8; T_wc — камера в мире (4x4); cubes — список
    {"T": куб в мире 4x4, "bits": 6 сеток 8x8, "rgb": целевой цвет (r,g,b)}.
    v_ref — яркость белого куба в этой камере, если известна по другим кадрам
    (когда куб почти целиком закрыт, по одному кадру её не узнать)."""
    H, W = img.shape[:2]
    T_cw = np.linalg.inv(T_wc)
    cam_pos = T_wc[:3, 3]
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    v_chan = hsv[:, :, 2].astype(np.float32)
    out = img.copy()
    allowed = np.zeros((H, W), bool)   # где менять пиксели было позволено

    # Проход 1: геометрия всех кубов. Кубы одинаково ч/б, и когда один
    # проходит перед другим, насыщенность их не разделяет — поэтому взаимные
    # окклюзии решаются глубиной: пиксель достаётся ближнему кубу.
    infos = []
    for cube in cubes:
        T = cube["T"]
        cube_mask = np.zeros((H, W), np.uint8)
        faces_px = []      # (fi, полигон 4x2, площадь)
        for fi in range(6):
            corners_w = (T[:3, :3] @ _face_corners_local(fi).T).T + T[:3, 3]
            n_w = T[:3, :3] @ np.asarray(FACES[fi][3], float)
            center_w = corners_w.mean(0)
            if np.dot(n_w, cam_pos - center_w) <= 0:
                continue
            pc = (T_cw[:3, :3] @ corners_w.T).T + T_cw[:3, 3]
            if np.any(pc[:, 2] <= 1e-6):
                continue
            px = (K @ pc.T).T
            px = px[:, :2] / px[:, 2:3]
            if px[:, 0].max() < 0 or px[:, 0].min() > W or \
                    px[:, 1].max() < 0 or px[:, 1].min() > H:
                continue
            poly = px.astype(np.float32)
            area = float(cv2.contourArea(poly))
            if area < 4:
                continue
            cv2.fillConvexPoly(cube_mask, np.round(poly).astype(np.int32), 1)
            faces_px.append((fi, poly, area))
        if faces_px:
            infos.append({"cube": cube, "mask": cube_mask, "faces": faces_px,
                          "dist": float(np.linalg.norm(T[:3, 3] - cam_pos))})

    # Проход 2: покраска от дальнего к ближнему (painter): маска каждого куба
    # исключает сырую геометрию более близких, а их собственная краска ляжет
    # поверх и закроет кромку.
    infos.sort(key=lambda i: -i["dist"])
    for idx, info in enumerate(infos):
        cube, faces_px = info["cube"], info["faces"]
        nearer = np.zeros((H, W), bool)
        for other in infos[idx + 1:]:
            nearer |= other["mask"] > 0

        # хроматичный окклюдер (рука): насыщенный И светлый — у тёмных пикселей
        # реальной камеры насыщенность шумная; тонкие всплески хромы на
        # кромках меток убирает открытие
        sat_occ = ((hsv[:, :, 1] >= SAT_OCCLUDER) & (hsv[:, :, 2] >= V_OCCLUDER)
                   & (hsv[:, :, 0] >= HUE_OCCLUDER[0]) & (hsv[:, :, 0] <= HUE_OCCLUDER[1])).astype(np.uint8)
        sat_occ = cv2.morphologyEx(sat_occ, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8)) > 0
        cube_geom = cv2.dilate(info["mask"], np.ones((5, 5), np.uint8)) > 0
        base_px = cube_geom & ~sat_occ & ~nearer

        # Тёмный окклюдер (корпус камеры на кисти, провод) против чёрных клеток
        # меток: и то и другое тёмное и ненасыщенное, но клетки целиком лежат
        # внутри силуэта куба (вокруг метки белое поле), а чужой предмет
        # выходит за него. Связная тёмная область, выходящая за силуэт, — не куб.
        side = max(np.sqrt(a) for *_, a in faces_px)
        cell = max(2, int(side / N))
        v_pre = v_ref if v_ref is not None else (
            float(np.percentile(v_chan[base_px], 80)) if base_px.sum() >= 4 else 255.0)
        dark = (v_chan < DARK_FRAC * v_pre).astype(np.uint8)
        # (а) тёмное пятно, у которого заметная часть (>30 %) снаружи силуэта
        # с запасом в клетку, — провод, корпус; размытая метка выходит за
        # силуэт лишь на пару пикселей и остаётся кубом
        n_lab, lab = cv2.connectedComponents(dark, connectivity=8)
        halo = cv2.dilate(info["mask"], np.ones((2 * cell + 1,) * 2, np.uint8)) > 0
        dark_occ = np.zeros((H, W), bool)
        for k in np.unique(lab[cube_geom & (dark > 0)]):
            if k == 0:
                continue
            comp = lab == k
            if (comp & ~halo).sum() > 0.3 * comp.sum():
                dark_occ |= comp
        # (б) тёмное дальше двух клеток от любого белого пикселя куба — не
        # метка (у клеток метки белое рядом всегда), а чужой тёмный предмет,
        # даже если его пятно раздроблено бликами. Только тёмное: затенённая
        # грань без белого — всё равно куб
        white = base_px & (v_chan > 0.75 * v_pre) & (hsv[:, :, 1] < 90)
        far = cv2.distanceTransform((~white).astype(np.uint8), cv2.DIST_L2, 3) > 2 * cell
        dark_occ |= (dark > 0) & far
        cube_px = base_px & ~dark_occ
        allowed |= cube_px
        DEBUG.update(dark=dark > 0, white=white, far=far, dark_occ=dark_occ,
                     sat_occ=sat_occ, geom=cube_geom, cube_px=cube_px, v_pre=v_pre)

        # эталон белого куба: самая светлая грань; у мелкого куба белых
        # пикселей может не набраться — светлый квантиль всех его пикселей
        v_fallback = float(np.percentile(v_chan[cube_px], 80)) \
            if cube_px.sum() >= 4 else 255.0
        v_fill = v_chan.copy()
        whites = {}
        for fi, poly, area in faces_px:
            side_px = np.sqrt(area)
            black = np.zeros((H, W), np.uint8)
            hmg = cv2.getPerspectiveTransform(
                np.array([[0, 0], [N, 0], [N, N], [0, N]], np.float32), poly)
            ys, xs = np.nonzero(cube["bits"][fi])
            for gy, gx in zip(ys, xs):
                cell = np.array([[gx, gy], [gx + 1, gy],
                                 [gx + 1, gy + 1], [gx, gy + 1]], np.float32)
                cpx = cv2.perspectiveTransform(cell[None], hmg)[0]
                cv2.fillConvexPoly(black, np.round(cpx).astype(np.int32), 1)
            r = max(1, int(side_px * BLACK_DILATE_FRAC))
            black = cv2.dilate(black, np.ones((2 * r + 1, 2 * r + 1), np.uint8)) > 0
            face_mask = np.zeros((H, W), np.uint8)
            cv2.fillConvexPoly(face_mask, np.round(poly).astype(np.int32), 1)
            face_mask = face_mask > 0
            white_px = face_mask & cube_px & ~black & (dark == 0)
            v_white = float(np.median(v_chan[white_px])) \
                if white_px.sum() >= 10 else v_fallback
            whites[fi] = v_white
            # закрасить чёрные клетки по геометрии И всё заметно темнее белого
            # своей грани (метка при ошибке позы или размытии — она серая)
            fb = face_mask & cube_px & (black | (v_chan < 0.7 * v_white))
            v_fill[fb] = v_white
            k = 2 * r + 1
            if fb.any():
                sm = cv2.GaussianBlur(v_fill, (k, k), 0)
                v_fill[fb] = sm[fb]
        v_ref = max(whites.values()) if whites else v_fallback

        # целевой цвет с пиксельной яркостью; лёгкое сглаживание швов заливки
        v_sm = cv2.GaussianBlur(v_fill, (5, 5), 0)
        shade = np.clip(v_sm / max(v_ref, 1.0), 0.0, 1.4)
        colored = np.clip(np.asarray(cube["rgb"], np.float32)[None, None, :]
                          * shade[:, :, None], 0, 255)

        alpha = cv2.GaussianBlur(cube_px.astype(np.float32), (3, 3), 0)
        alpha *= cube_px      # менять можно только пиксели куба
        out = (out.astype(np.float32) * (1 - alpha[:, :, None])
               + colored * alpha[:, :, None]).astype(np.uint8)
    return out, allowed


def camera_K(fovy_deg: float, w: int, h: int) -> np.ndarray:
    f = h / (2 * np.tan(np.radians(fovy_deg) / 2))
    return np.array([[f, 0, (w - 1) / 2], [0, f, (h - 1) / 2], [0, 0, 1]])


def calibrate_K(obs: list[tuple[np.ndarray, np.ndarray]], w: int, h: int):
    """Интринсики по соответствиям (точка в системе камеры, пиксель) МНК:
    u = fx*x/z + cx, v = fy*y/z + cy. Возвращает (K, RMS-остаток в px).
    Так же откалибруем реальную камеру, только точки дадут метки на столе."""
    ax, bx, ay, by = [], [], [], []
    for pc, uv in obs:
        ax.append([pc[0] / pc[2], 1.0])
        bx.append(uv[0])
        ay.append([pc[1] / pc[2], 1.0])
        by.append(uv[1])
    (fx, cx), resx, *_ = np.linalg.lstsq(np.array(ax), np.array(bx), rcond=None)
    (fy, cy), resy, *_ = np.linalg.lstsq(np.array(ay), np.array(by), rcond=None)
    n = len(obs)
    rms = float(np.sqrt(((np.array(ax) @ [fx, cx] - bx) ** 2).mean()
                        + ((np.array(ay) @ [fy, cy] - by) ** 2).mean()))
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    return K, rms
