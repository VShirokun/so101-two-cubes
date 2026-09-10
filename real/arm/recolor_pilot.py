#!/usr/bin/env python3
"""Перекраска кубов в записанных эпизодах пилота (постобработка).

Кубы физически чёрно-белые с метками; здесь они становятся цветными в обеих
камерах по геометрии (mlsim/recolor.py): нужны позы кубов и камер на каждом
кадре. Лежащий куб — по верхней камере: x, y, yaw подгоняются по углам его
меток в первом (и последнем) кадре, куб лежит на столе. Несомый куб —
привязка к кисти в момент смыкания губок (уточняется по камере кисти на
подъёме), дальше едет с FK (поправки автокалибровки). Камера кисти в базе —
FK · X_w. Кадры сначала избавляются от дисторсии (recolor работает с pinhole
K); верхняя камера кропается 4:3 -> 640x480, кисть 640x480 как есть. Цвета —
случайная пара из палитры на эпизод, задача — «pick up the <цвет> cube and
drop it».

  recolor_pilot.py [--raw real/data/pilot] [--out real/data/pilot_recolor]
                   [--episodes N] [--seed 0]
Выход: out/ep_XXXX/{top,wrist}/*.jpg (640x480 перекрашенные), steps.jsonl,
meta.json (+colors, task); out/recolor_preview.mp4 (сырой | перекрашенный),
out/recolor_stills.png.
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as Rot

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parents[1] / "mlsim"))
import autocalib_real as A  # noqa: E402
from cube_cv import CUBE_HALF, FACES, _ID2FACE, _avg_rotations  # noqa: E402
from recolor import _face_corners_local, cube_faces_bits, recolor_frame  # noqa: E402

# без оранжевого: рука и губки оранжевые, куб того же цвета сливается с ними
PALETTE = {"red": (215, 45, 45), "green": (10, 140, 90), "blue": (45, 90, 225),
           "yellow": (240, 205, 40), "purple": (150, 60, 215), "cyan": (40, 200, 220),
           "gray": (82, 126, 151)}
# green/gray — под НАСТОЯЩИЕ кубы 10.09.2026: верхняя камера видит зелёный как BGR (83,125,5),
# серый как серо-голубой BGR (136,114,74) при белом куба ~230; альбедо = замер × 255/230
ALLOWED = None            # --colors: ограничение палитры (настоящие кубы зелёный и серый)
GRIP_MID = 0.8            # чтение губок ниже — сомкнуты (куб едет с кистью)
TOP_W, TOP_H = 1920, 1080
CW = TOP_H * 4 // 3       # кроп 4:3
X0 = (TOP_W - CW) // 2
SCALE = 640 / CW
K_TOP640 = A.K_TOP.copy()
K_TOP640[0, 2] -= X0
K_TOP640[:2] *= SCALE
BITS = {c: cube_faces_bits(c) for c in ("red", "green")}


def prep_top(img_bgr):
    """Без дисторсии -> кроп 4:3 -> 640x480 RGB (K_TOP640)."""
    u = cv2.undistort(img_bgr, A.K_TOP, A.D_TOP, None, A.K_TOP)
    return cv2.cvtColor(cv2.resize(u[:, X0:X0 + CW], (640, 480),
                                   interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)


def prep_wrist(img_bgr):
    return cv2.cvtColor(cv2.undistort(img_bgr, A.K_WR, A.D_WR, None, A.K_WR),
                        cv2.COLOR_BGR2RGB)


def top_corners(img_bgr):
    """Углы меток в кадре верхней камеры без дисторсии: {id: 4x2}."""
    return {m: A.undistort(v, A.K_TOP, A.D_TOP) for m, v in A.corners_raw(img_bgr).items()}


def fit_lying(px_all, T_init, color, T_cam_base):
    """x, y, yaw лежащего куба по углам его меток (грань вверх — из T_init).
    Нет меток — T_init как есть и rms None."""
    px = {m: v for m, v in px_all.items() if _ID2FACE[m][0] == color}
    R = T_init[:3, :3]
    fi = int(np.argmax([(R @ np.asarray(f[3], float))[2] for f in FACES]))
    R0 = Rot.align_vectors([[0, 0, 1.0]], [np.asarray(FACES[fi][3], float)])[0].as_matrix()
    Rc = R @ R0.T
    p0 = [T_init[0, 3], T_init[1, 3], float(np.arctan2(Rc[1, 0], Rc[0, 0]))]
    if not px:
        return A._cube_T(R0, *p0), None

    def resid(p):
        cb = A.corners_base(color, R0, *p)
        return np.concatenate([(A.project(A.K_TOP, T_cam_base, cb[m]) - v).ravel()
                               for m, v in px.items() if m in cb])

    sol = least_squares(resid, p0, method="lm")
    return A._cube_T(R0, *sol.x), float(np.sqrt(np.mean(sol.fun ** 2)))


def cubes_by_cam(img_bgr, K, D):
    """Позы кубов в системе камеры по их меткам в ЭТОМ кадре (PnP по всем
    видимым меткам). Для камеры кисти на 6-10 см это точнее FK: миллиметры
    калибровки там — десятки пикселей."""
    return A.cube_poses({m: A.undistort(v, K, D) for m, v in A.corners_raw(img_bgr).items()}, K)


def avg_T(Ts):
    T = np.eye(4)
    T[:3, :3] = _avg_rotations([t[:3, :3] for t in Ts])
    T[:3, 3] = np.median([t[:3, 3] for t in Ts], 0)
    return T


MODULE_R = 0.025          # радиус корпуса камеры на кисти вокруг её центра, м


def hide_wrist_module(out, raw, C, cubes, v_ref):
    """Корпус камеры на кисти в верхней камере: его место известно из
    кинематики (центр — оптический центр камеры кисти); если он ближе к
    верхней камере, чем куб, всё небелое в его кружке — корпус, не куб."""
    T_cb = np.linalg.inv(_XT)
    pc = T_cb[:3, :3] @ C[:3, 3] + T_cb[:3, 3]
    if pc[2] <= 0.05:
        return out
    if all(np.linalg.norm(T_cb[:3, :3] @ c["T"][:3, 3] + T_cb[:3, 3]) < np.linalg.norm(pc)
           for c in cubes):
        return out                         # кубы ближе корпуса — не заслоняет
    uv = K_TOP640 @ pc
    u, v = uv[:2] / uv[2]
    r = int(K_TOP640[0, 0] * MODULE_R / pc[2]) + 2
    disc = np.zeros(out.shape[:2], np.uint8)
    cv2.circle(disc, (int(u), int(v)), r, 1, -1)
    hsv = cv2.cvtColor(raw, cv2.COLOR_RGB2HSV)
    white = (hsv[:, :, 2] > 0.75 * (v_ref or 200)) & (hsv[:, :, 1] < 90)
    keep = (disc > 0) & ~white
    out = out.copy()
    out[keep] = raw[keep]
    return out


def process_episode(ep_dir, out_dir, Xt, Xw, rng, keep=None, colors=None):
    global _XT
    _XT = Xt
    """Позы кубов по кадрам, затем покраска.

    Верхняя камера: лежащий куб — подгонка x/y/yaw по его меткам в КАЖДОМ
    кадре (иначе куб, сдвинутый губками, рисовался бы сбоку); несомый — по
    камере кисти через FK. Камера кисти: поза по её же меткам в кадре; в
    держании без детекции — медиана держания (куб в губках неподвижен
    относительно камеры); на столе без детекции — FK с поправкой по ближайшему
    кадру с детекцией (ошибка FK на 6-10 см даёт десятки пикселей — куб
    «просвечивал» рядом с настоящим)."""
    meta = json.loads((ep_dir / "meta.json").read_text())
    color = meta["cube"]
    other = [c for c in ("red", "green") if c != color][0]
    recs = [json.loads(l) for l in open(ep_dir / "steps.jsonl")]
    n = len(recs)
    T_cam_base = np.linalg.inv(Xt)
    F = [A.fk_T(np.asarray(r["state"][:5])) for r in recs]
    C = [f @ Xw for f in F]                       # камера кисти в базе

    if colors is None:
        names = [c for c in PALETTE if ALLOWED is None or c in ALLOWED]
        pick = rng.choice(len(names), size=2, replace=False)
        colors = {color: names[pick[0]], other: names[pick[1]]}
    task = f"pick up the {colors[color]} cube and drop it"

    # --- проход 1: детекции обеих камер по всем кадрам ---------------------
    top_px, wr_det = [], []
    for r in recs:
        top_px.append(top_corners(cv2.imread(str(ep_dir / "top" / f"{r['i']:06d}.jpg"))))
        wr_det.append(cubes_by_cam(cv2.imread(str(ep_dir / "wrist" / f"{r['i']:06d}.jpg")),
                                   A.K_WR, A.D_WR))

    # яркость белого куба в каждой камере по эпизоду (в кадре, где куб почти
    # закрыт корпусом камеры, по одному кадру её не узнать)
    def v_ref_of(cam, K, poses):
        vals = []
        for i in range(0, n, max(1, n // 40)):
            if poses[i] is None:
                continue
            img = cv2.imread(str(ep_dir / cam / f"{recs[i]['i']:06d}.jpg"))
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            m = np.zeros(img.shape[:2], np.uint8)
            T = poses[i]
            for fi in range(6):
                cw = (T[:3, :3] @ _face_corners_local(fi).T).T + T[:3, 3]
                if cw[:, 2].min() <= 0.01:
                    continue
                px = (K @ cw.T).T
                cv2.fillConvexPoly(m, np.round(px[:, :2] / px[:, 2:3]).astype(np.int32), 1)
            sel = (m > 0) & (hsv[:, :, 1] < 90)
            if sel.sum() > 50:
                vals.append(np.percentile(hsv[:, :, 2][sel], 90))
        return float(np.median(vals)) if vals else None
    v_ref_wr = v_ref_of("wrist", A.K_WR, [d.get(color) for d in wr_det])
    top_poses = []
    for i in range(n):
        px = {m: v for m, v in top_px[i].items() if _ID2FACE[m][0] == color}
        top_poses.append(A.cube_poses(px, A.K_TOP)[color] if px else None)   # в системе камеры
    v_ref_top = v_ref_of("top", A.K_TOP, top_poses)

    grip = np.array([r["state"][5] for r in recs])
    phase = [r["phase"] for r in recs]
    i_close = next((i for i in range(n) if phase[i] in ("grasp", "lift") and grip[i] < GRIP_MID), n)
    i_open = next((i for i in range(i_close + 1, n)
                   if phase[i] in ("release", "retreat", "home") and grip[i] > GRIP_MID), n)
    seg = ["table0" if i < i_close else "held" if i < i_open else "table1" for i in range(n)]

    # --- верхняя камера: лежащий куб по кадрам ----------------------------
    T_tab = [None] * n
    T_prev, rms0 = fit_lying(top_px[0], np.array(meta["cube_start"]), color, T_cam_base)
    for i in range(i_close):
        T_i, rms = fit_lying(top_px[i], T_prev, color, T_cam_base)
        if rms is not None and rms < 4:
            T_prev = T_i
        T_tab[i] = T_prev
    if i_open < n:
        T_init = np.array(meta["cube_end"]) if meta.get("cube_end") is not None else T_prev
        T_prev, _ = fit_lying(top_px[n - 1], T_init, color, T_cam_base)
        for i in range(i_open, n):
            T_i, rms = fit_lying(top_px[i], T_prev, color, T_cam_base)
            if rms is not None and rms < 4:
                T_prev = T_i
            T_tab[i] = T_prev
    T_other = None
    if other in meta.get("others", {}):
        T_other, _ = fit_lying(top_px[0], np.array(meta["others"][other]), other, T_cam_base)

    # --- держание: куб в камере кисти неподвижен -> медиана детекций --------
    held_det = [wr_det[i][color] for i in range(i_close, i_open) if color in wr_det[i]]
    G0 = np.linalg.inv(C[i_close - 1]) @ T_tab[i_close - 1] if 0 < i_close < n else None
    G_cam = avg_T(held_det) if held_det else G0
    # насколько куб сдвинулся в губках при смыкании (привязка vs камера кисти)
    held_shift = float(np.linalg.norm((C[i_close] @ G_cam)[:3, 3] - (C[i_close] @ G0)[:3, 3]) * 1000) \
        if held_det and G0 is not None and i_close < n else None

    def wrist_pose(i, c, T_table):
        """Поза куба c в камере кисти на кадре i."""
        if c in wr_det[i]:
            return wr_det[i][c]
        if c == color and seg[i] == "held":
            return G_cam
        if T_table is None:
            return None
        fk = lambda k: np.linalg.inv(C[k]) @ T_table(k)
        same = [k for k in range(n) if c in wr_det[k] and (c != color or seg[k] == seg[i])]
        if not same:
            return fk(i)
        k = min(same, key=lambda k: abs(k - i))
        if abs(k - i) > 45:                  # дальше 1,5 с поправка уже не та
            return fk(i)
        delta = wr_det[k][c] @ np.linalg.inv(fk(k))
        return delta @ fk(i)

    def top_pose(i):
        if seg[i] == "held":
            return C[i] @ (wr_det[i][color] if color in wr_det[i] else G_cam)
        return T_tab[i]

    # --- проход 2: покраска ---------------------------------------------
    (out_dir / "top").mkdir(parents=True, exist_ok=True)
    (out_dir / "wrist").mkdir(exist_ok=True)
    shutil.copy(ep_dir / "steps.jsonl", out_dir / "steps.jsonl")
    pairs, n_det, n_fill = [], 0, 0
    for i, r in enumerate(recs):
        top = cv2.imread(str(ep_dir / "top" / f"{r['i']:06d}.jpg"))
        wr = cv2.imread(str(ep_dir / "wrist" / f"{r['i']:06d}.jpg"))
        if top is None or wr is None:
            continue
        cubes_top = []
        T_c = top_pose(i)
        if T_c is not None:
            cubes_top.append({"T": T_c, "bits": BITS[color], "rgb": PALETTE[colors[color]]})
        if T_other is not None:
            cubes_top.append({"T": T_other, "bits": BITS[other], "rgb": PALETTE[colors[other]]})
        cubes_wr = []
        Tw = wrist_pose(i, color, lambda k: T_tab[k])
        n_det += color in wr_det[i]
        n_fill += Tw is not None and color not in wr_det[i]
        if Tw is not None:
            cubes_wr.append({"T": C[i] @ Tw, "bits": BITS[color], "rgb": PALETTE[colors[color]]})
        Tw2 = wrist_pose(i, other, (lambda k: T_other) if T_other is not None else None)
        if Tw2 is not None:
            cubes_wr.append({"T": C[i] @ Tw2, "bits": BITS[other], "rgb": PALETTE[colors[other]]})
        top_rgb, wr_rgb = prep_top(top), prep_wrist(wr)
        top_out, _ = recolor_frame(top_rgb, K_TOP640, Xt, cubes_top, v_ref=v_ref_top)
        top_out = hide_wrist_module(top_out, top_rgb, C[i], cubes_top, v_ref_top)
        wr_out, _ = recolor_frame(wr_rgb, A.K_WR, C[i], cubes_wr, v_ref=v_ref_wr)
        cv2.imwrite(str(out_dir / "top" / f"{r['i']:06d}.jpg"),
                    cv2.cvtColor(top_out, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])
        cv2.imwrite(str(out_dir / "wrist" / f"{r['i']:06d}.jpg"),
                    cv2.cvtColor(wr_out, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])
        if keep is not None and i % keep == 0:
            pairs.append((top_rgb, top_out, wr_rgb, wr_out, r["phase"]))
    meta.update({"colors": colors, "task": task, "recolor": {
        "fit_rms_px_start": rms0, "i_close": int(i_close), "i_open": int(i_open),
        "wrist_frames_for_held_pose": len(held_det), "wrist_frames_with_detection": int(n_det),
        "wrist_frames_filled": int(n_fill), "held_shift_mm": held_shift,
        "v_ref": {"top": v_ref_top, "wrist": v_ref_wr},
        "palette": {k: list(v) for k, v in PALETTE.items()}}})
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    return meta, pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, default=A._ROOT / "real/data/pilot")
    ap.add_argument("--out", type=Path, default=A._ROOT / "real/data/pilot_recolor")
    ap.add_argument("--episodes", type=int, default=0, help="0 = все успешные")
    ap.add_argument("--only", nargs="*", help="только эти эпизоды (ep_0003 ...)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--colors", default="", help="ограничить палитру, например green,gray (настоящие кубы)")
    a = ap.parse_args()
    global ALLOWED
    ALLOWED = a.colors.split(",") if a.colors else None
    res = json.loads(A.OUT.read_text())
    Xt, Xw = np.array(res["T_base_topcam"]), np.array(res["T_gripper_wristcam"])
    rng = np.random.default_rng(a.seed)
    def _ok(m):      # подъём был: успех, или захват подтверждён, или куб доехал до цели
        return m.get("success") or m.get("held") or (m.get("landing_err_mm") is not None and m["landing_err_mm"] < 40)
    eps = [p for p in sorted(a.raw.glob("ep_*")) if (p / "meta.json").exists()
           and _ok(json.loads((p / "meta.json").read_text()))]
    if a.only:
        eps = [p for p in eps if p.name in a.only]
    if a.episodes:
        eps = eps[:a.episodes]
    a.out.mkdir(parents=True, exist_ok=True)
    vw = cv2.VideoWriter(str(a.out / "recolor_preview.mp4"),
                         cv2.VideoWriter_fourcc(*"mp4v"), 15, (1280, 960))
    stills = []
    for ep in eps:
        meta, pairs = process_episode(ep, a.out / ep.name, Xt, Xw, rng, keep=3)
        print(f"  {ep.name}: {meta['cube']} -> {meta['colors']}; {meta['task']}; "
              f"привязка: кадры {meta['recolor']['i_close']}..{meta['recolor']['i_open']}, "
              f"кисть видела куб в {meta['recolor']['wrist_frames_for_held_pose']} кадрах, "
              f"сдвиг в губках {meta['recolor']['held_shift_mm']} мм, "
              f"дорисовано без детекции {meta['recolor']['wrist_frames_filled']} кадров")
        for k, (t0, t1, w0, w1, ph) in enumerate(pairs):
            fr = np.vstack([np.hstack([t0, t1]), np.hstack([w0, w1])])
            fr = cv2.cvtColor(fr, cv2.COLOR_RGB2BGR)
            cv2.putText(fr, f"{ep.name}  {ph}  raw | recolored: {meta['task']}", (10, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (40, 220, 40), 2, cv2.LINE_AA)
            vw.write(fr)
            if k in (len(pairs) // 4, len(pairs) // 2, 3 * len(pairs) // 4) and len(stills) < 12:
                stills.append(cv2.resize(fr, (640, 480)))
    vw.release()
    if stills:
        rows = [np.hstack(stills[i:i + 3]) for i in range(0, len(stills) - len(stills) % 3, 3)]
        if rows:
            cv2.imwrite(str(a.out / "recolor_stills.png"), np.vstack(rows))
    print(f"готово: {len(eps)} эпизодов -> {a.out}; превью recolor_preview.mp4, recolor_stills.png")


if __name__ == "__main__":
    main()
