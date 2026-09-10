#!/usr/bin/env python3
"""Пилотный сбор датасета на реальном стенде (сырой формат).

Эпизод: с парковки верхняя камера находит кубы → рука: над кубом 9 см →
3 см над гранью → к центру → смыкание поперёк граней → подъём → камера кисти
подтверждает куб в губках → перенос в случайную точку зоны → опустить до
1 см над столом → отпустить → подъём → парковка. Обе камеры и суставы
пишутся КАЖДЫЙ такт 30 Гц как есть (JPEG с серверов камер без
перекодирования + steps.jsonl), чтобы потом перекрасить кубы и собрать
lerobot-датасет. Углы — в радианах модели (с поправками автокалибровки).

  collect_pilot.py --episodes 20 [--out real/data/pilot]
Стоп: touch /tmp/roboom_arm/stop — рука замирает, серия прерывается.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import autocalib_real as A  # noqa: E402
from arm_driver import ArmDriver, GuardError, STOP  # noqa: E402
from ik import ArmIK  # noqa: E402

ZONE_X, ZONE_Y, R_MAX = (0.18, 0.27), (-0.13, 0.13), 0.28
MIN_GAP, MIN_MOVE = 0.09, 0.05      # от другого куба; минимальный перенос
TRANSPORT_Z = 0.08                  # 9 см — граница досягаемости IK для дальних целей
DEFAULT_OUT = A._ROOT / "real" / "data" / "pilot"


class Episode:
    """Запись эпизода: кадры обеих камер как есть + состояние на каждый такт."""

    def __init__(self, root, idx, d):
        self.d, self.dir = d, root / f"ep_{idx:04d}"
        (self.dir / "top").mkdir(parents=True)
        (self.dir / "wrist").mkdir()
        self.f = open(self.dir / "steps.jsonl", "w")
        self.n, self.phase = 0, "start"

    def tick(self, q_goal, q_read, raw):
        rec = {"i": self.n, "t": time.time(), "phase": self.phase,
               "state": [*self.d.q_model(q_read).tolist(), float(q_read[5])],
               "action": [*self.d.q_model(q_goal).tolist(), float(q_goal[5])],
               "raw": [int(raw[k]) for k in range(1, 7)]}
        for cam, path in (("top", A.CAM_TOP), ("wrist", A.CAM_WR)):
            rec[cam + "_ts"] = path.stat().st_mtime
            (self.dir / cam / f"{self.n:06d}.jpg").write_bytes(path.read_bytes())
        self.f.write(json.dumps(rec) + "\n")
        self.n += 1

    def close(self, meta):
        self.f.close()
        (self.dir / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))


def top_frame_640(img_bgr):
    """Верхняя камера 1920x1080 -> центральный кроп 4:3 -> 640x480 (как в датасете)."""
    h, w = img_bgr.shape[:2]
    cw = h * 4 // 3
    x0 = (w - cw) // 2
    return cv2.resize(img_bgr[:, x0:x0 + cw], (640, 480), interpolation=cv2.INTER_AREA)


def look(Xt):
    """Кубы верхней камерой (рука на парковке): {color: T база<-куб}, глубина
    снята лучом на плоскость «куб лежит»."""
    _, top = A.stable_view(A.CAM_TOP)
    out = {}
    for color, T_c in A.cube_poses({m: A.undistort(px, A.K_TOP, A.D_TOP)
                                    for m, px in top.items()}, A.K_TOP).items():
        T = Xt @ T_c
        c, ray = Xt[:3, 3], T[:3, 3] - Xt[:3, 3]
        T[:3, 3] = c + ray * (A.CUBE_HALF - c[2]) / ray[2]
        out[color] = T
    return out


def plan(ik, target, q_init, yaw):
    """IK: позиция + подход сверху + азимут губок; у предела — наклон подхода
    или без азимута. -> q | None."""
    az = float(np.arctan2(target[1], target[0]))
    inits = (np.asarray(q_init, float), np.array([az, -0.6, 0.9, 0.9, 0.0]),
             np.array([az, -0.3, 0.3, 1.2, 0.0]))     # DLS застревает из далёкого старта
    for oy, tilt in ((yaw, 0), (yaw, 15), (None, 0), (None, 15)):
        t = np.radians(tilt)
        appr = [np.cos(az) * np.sin(t), np.sin(az) * np.sin(t), -np.cos(t)]
        for q0 in inits:
            A._DATA.qpos[:] = 0
            A._DATA.qpos[:5] = q0
            q, err = ik.solve(A._DATA, target, approach=appr, q_init=q0, opening_yaw=oy)
            if err < 0.003:
                return q
    return None


def waypoints(xy, yaw=None):
    """Точки над кубом/целью: перенос, 3 см над гранью, центр, сброс с 1 см."""
    return {"above": [*xy, TRANSPORT_Z], "hover": [*xy, A.CUBE_HALF + 0.030],
            "low": [*xy, A.CUBE_HALF + 0.006], "drop": [*xy, A.CUBE_HALF + 0.012]}


def feasible(ik, xy, yaw):
    """Все точки над xy решаются IK (старт — из парковки)."""
    q = np.array(A.PARK)
    for k, pt in waypoints(xy).items():
        q = plan(ik, pt, q, yaw)
        if q is None:
            return False
    return True


def move(d, ik, target, grip, seconds, q_init, yaw, mode="transport", tick=None):
    """IK и движение; при блокировке на старте — подъём и повтор."""
    q = plan(ik, target, q_init, yaw)
    if q is None:
        raise GuardError(f"IK не встал для {np.round(target, 3)}")
    cmd = np.concatenate([d.q_cmd(q), [grip]])
    try:
        d.goto(cmd, seconds, mode=mode, tick=tick)
    except GuardError as e:
        if "на пути" not in str(e):
            raise
        print("  guard:", e, "-> lift и повтор")
        d.lift()
        d.goto(cmd, seconds, mode=mode, tick=tick)
    return q


def cube_by_wrist(d, color, Xw):
    """Доводка: куб камерой кисти, когда рука зависла над ним (3 см над
    гранью). Центр — луч на плоскость «куб лежит». -> (xy, yaw) | None."""
    try:
        _, wr = A.stable_view(A.CAM_WR, n=3, tries=2)
    except RuntimeError:
        return None
    est = A.cube_poses({m: A.undistort(px, A.K_WR, A.D_WR) for m, px in wr.items()
                        if A._ID2FACE[m][0] == color}, A.K_WR)
    if color not in est:
        return None
    C = d.fk_T(d.read()[0][:5]) @ Xw          # камера кисти в базе (с поправками)
    T = C @ est[color]
    c, ray = C[:3, 3], T[:3, 3] - C[:3, 3]
    if ray[2] > -1e-3:
        return None
    T[:3, 3] = c + ray * (A.CUBE_HALF - c[2]) / ray[2]
    return T[:2, 3].copy(), A.cube_yaw(T)


def cube_in_gripper(color):
    """Камера кисти: метка куба ближе 10 см — куб в губках."""
    try:
        _, wr = A.stable_view(A.CAM_WR, n=3, tries=2)
    except RuntimeError:
        return None
    est = A.cube_poses({m: A.undistort(px, A.K_WR, A.D_WR) for m, px in wr.items()
                        if A._ID2FACE[m][0] == color}, A.K_WR)
    return color in est and float(est[color][2, 3]) < 0.10


def reachable(T):
    """Брать можно любой куб, до которого достаёт IK (в зону обязана попадать
    только точка сброса — так оба куба по очереди оказываются в зоне)."""
    x, y = T[:2, 3]
    return np.hypot(x, y) <= 0.31 and 0.14 <= x <= 0.32 and abs(y) <= 0.21


def sample_target(rng, ik, cubes, color):
    P0, yaw = cubes[color][:2, 3], A.cube_yaw(cubes[color])
    others = [T[:2, 3] for c, T in cubes.items() if c != color]
    for _ in range(300):
        t = rng.uniform([ZONE_X[0], ZONE_Y[0]], [ZONE_X[1], ZONE_Y[1]])
        if np.hypot(*t) > R_MAX or np.linalg.norm(t - P0) < MIN_MOVE:
            continue
        if any(np.linalg.norm(t - o) < MIN_GAP for o in others):
            continue
        if feasible(ik, t, yaw):
            return t
    return None


def run_episode(d, ik, ep, T_bc, color, target, Xw):
    P0, yaw = T_bc[:3, 3].copy(), A.cube_yaw(T_bc)
    w, wt = waypoints(P0[:2]), waypoints(target)
    above, hov, low = w["above"], w["hover"], w["low"]
    t_above, t_drop = wt["above"], wt["drop"]
    tk = ep.tick
    q = d.q_model(d.read()[0])
    ep.phase = "approach"
    q = move(d, ik, above, A.GRIP_WIDE, 3.0, q, yaw, tick=tk)
    ep.phase = "descend"
    q = move(d, ik, hov, A.GRIP_WIDE, 1.5, q, yaw, tick=tk)
    # доводка по камере кисти: верхняя камера ошибается до 3 см (метка сбоку,
    # тень) — с 3 см над гранью куб виден крупно, центр и yaw точнее
    d.settle(2.0, 0.5, tick=tk)
    fix = cube_by_wrist(d, color, Xw)
    ep.servo_mm = None
    if fix is not None:
        xy2, yaw2 = fix
        ep.servo_mm = float(np.linalg.norm(xy2 - P0[:2]) * 1000)
        if ep.servo_mm < 40:      # больше — не верим (не тот куб), идём по верхней
            w2 = waypoints(xy2)
            hov, low, yaw = w2["hover"], w2["low"], yaw2
            q = move(d, ik, hov, A.GRIP_WIDE, 1.0, q, yaw, tick=tk)
    q = move(d, ik, low, A.GRIP_WIDE, 1.5, q, yaw, mode="grasp", tick=tk)
    ep.phase = "grasp"
    d.goto(np.concatenate([d.q_cmd(q), [A.GRIP_CLOSED]]), 0.7, mode="grasp", tick=tk)
    d.settle(0.5, 0.3, tick=tk)
    ep.phase = "lift"
    q = move(d, ik, above, A.GRIP_CLOSED, 1.5, q, yaw, mode="grasp", tick=tk)
    d.settle(tick=tk)                       # 2 с — замер камерой кисти
    held = cube_in_gripper(color)
    ep.phase = "transport"
    q = move(d, ik, t_above, A.GRIP_CLOSED, 2.5, q, yaw, tick=tk)
    ep.phase = "lower"
    q = move(d, ik, t_drop, A.GRIP_CLOSED, 1.5, q, yaw, mode="grasp", tick=tk)
    ep.phase = "release"
    d.goto(np.concatenate([d.q_cmd(q), [A.GRIP_WIDE]]), 0.7, mode="grasp", tick=tk)
    d.settle(0.5, 0.3, tick=tk)
    ep.phase = "retreat"
    q = move(d, ik, t_above, A.GRIP_WIDE, 1.5, q, yaw, mode="grasp", tick=tk)
    ep.phase = "home"
    d.goto(A.PARK + [A.GRIP_WIDE], 3.0, tick=tk)
    return held


def preview(root, path, every=3):
    """Ускоренное превью всех эпизодов серии: верхняя + кисть, фаза, номер."""
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 15, (1120, 360))
    for ep_dir in sorted(root.glob("ep_*")):
        meta = json.loads((ep_dir / "meta.json").read_text()) if (ep_dir / "meta.json").exists() else {}
        for k, line in enumerate(open(ep_dir / "steps.jsonl")):
            if k % every:
                continue
            rec = json.loads(line)
            top = cv2.imread(str(ep_dir / "top" / f"{rec['i']:06d}.jpg"))
            wr = cv2.imread(str(ep_dir / "wrist" / f"{rec['i']:06d}.jpg"))
            if top is None or wr is None:
                continue
            frame = np.hstack([cv2.resize(top, (640, 360)), cv2.resize(wr, (480, 360))])
            cv2.putText(frame, f"{ep_dir.name}  {rec['phase']}  "
                        f"{'OK' if meta.get('success') else ('fail' if meta else '')}",
                        (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (40, 220, 40), 2, cv2.LINE_AA)
            w.write(frame)
    w.release()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--preview-only", action="store_true")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    if a.preview_only:
        preview(a.out, a.out / "preview.mp4")
        return
    res = json.loads(A.OUT.read_text())
    Xt, Xw = np.array(res["T_base_topcam"]), np.array(res["T_gripper_wristcam"])
    ik = ArmIK(A._MODEL)
    rng = np.random.default_rng(int(time.time()))
    # следующий номер — за максимальным, а не по числу папок (удалённые эпизоды)
    start = max((int(p.name[3:]) for p in a.out.glob("ep_*")), default=-1) + 1
    d = ArmDriver()
    d.torque(True)
    d.hold()
    summary, last_color, no_cube = [], None, 0
    prev = a.out / "summary.jsonl"          # продолжение серии: чередовать с прошлым кубом
    if prev.exists() and prev.read_text().strip():
        last_color = json.loads(prev.read_text().strip().splitlines()[-1]).get("cube")
    try:
        d.goto(A.PARK + [A.GRIP_WIDE], 4.0, mode="grasp")
        d.settle()
        for ep_i in range(start, start + a.episodes):
            if STOP.exists():
                print("стоп-файл — серия прервана")
                break
            cubes = look(Xt)
            ok_colors = [c for c, T in cubes.items()
                         if reachable(T) and feasible(ik, T[:2, 3], A.cube_yaw(T))]
            if not ok_colors:
                no_cube += 1
                print(f"нет куба в зоне/досягаемости (видны: "
                      f"{ {c: np.round(T[:2, 3], 3).tolist() for c, T in cubes.items()} }) — "
                      "переставьте кубы")
                if no_cube >= 2:
                    break
                time.sleep(10)
                continue
            no_cube = 0
            color = next((c for c in ok_colors if c != last_color), ok_colors[0])
            if last_color is not None and color == last_color and len(cubes) > 1:
                print(f"  {[c for c in cubes if c != last_color]} недосягаем — снова {color}")
            target = sample_target(rng, ik, cubes, color)
            if target is None:
                print(f"{color}: не нашлось цели переноса")
                break
            ep = Episode(a.out, ep_i, d)
            meta = {"episode": ep_i, "cube": color, "t0": time.time(),
                    "cube_start": cubes[color].tolist(), "target_xy": target.tolist(),
                    "others": {c: T.tolist() for c, T in cubes.items() if c != color},
                    "calib": str(A.OUT.name), "task_hint": "pick up the cube and drop it"}
            print(f"эпизод {ep_i}: {color} в {np.round(cubes[color][:2, 3], 3).tolist()} "
                  f"-> цель {np.round(target, 3).tolist()}")
            try:
                held = run_episode(d, ik, ep, cubes[color], color, target, Xw)
                meta["held"] = held
                meta["servo_corr_mm"] = None if ep.servo_mm is None else round(ep.servo_mm, 1)
                d.settle()
                after = look(Xt)
                meta["cube_end"] = after[color].tolist() if color in after else None
                if color in after:
                    dist = float(np.linalg.norm(after[color][:2, 3] - target)) * 1000
                    meta["landing_err_mm"] = round(dist, 1)
                    meta["success"] = bool(held) and dist < 40
                else:
                    meta["success"] = False
                print(f"  доводка {meta['servo_corr_mm']} мм; захват {'подтверждён' if held else 'НЕ подтверждён'}, "
                      f"куб после: {None if color not in after else np.round(after[color][:2, 3], 3).tolist()}, "
                      f"до цели {meta.get('landing_err_mm')} мм -> "
                      f"{'OK' if meta['success'] else 'FAIL'}, кадров {ep.n}")
            except GuardError as e:
                meta.update({"success": False, "error": str(e)})
                print(f"  guard: {e} — эпизод помечен failed, восстановление")
                try:
                    q_now = d.hold()
                    if q_now[5] < 0.6:       # губки сомкнуты — куб может быть в них
                        F = d.fk_T(q_now[:5])
                        try:
                            q_low = plan(ik, [*F[:2, 3], A.CUBE_HALF + 0.012],
                                         d.q_model(q_now), None)
                            if q_low is not None:
                                d.goto(np.concatenate([d.q_cmd(q_low), [A.GRIP_CLOSED]]),
                                       2.0, mode="grasp")
                        except GuardError as e3:
                            print("  опустить не удалось:", e3)
                        d.goto(np.concatenate([d.read()[0][:5], [A.GRIP_WIDE]]), 0.7,
                               mode="grasp")
                        d.settle(0.5, 0.3)
                    d.lift()
                    d.goto(A.PARK + [A.GRIP_WIDE], 4.0, mode="grasp")
                except GuardError as e2:
                    print("  восстановление не удалось:", e2)
                    ep.close(meta)
                    break
            meta["frames"], meta["t1"] = ep.n, time.time()
            ep.close(meta)
            summary.append(meta)
            last_color = color
    finally:
        d.close()
    n_ok = sum(1 for m in summary if m.get("success"))
    print(f"серия: {len(summary)} эпизодов, успешных {n_ok}, кадров "
          f"{sum(m.get('frames', 0) for m in summary)}; данные в {a.out}")
    (a.out / "summary.jsonl").open("a").write(
        "".join(json.dumps(m, ensure_ascii=False) + "\n" for m in summary))
    preview(a.out, a.out / "preview.mp4")
    print(f"превью: {a.out / 'preview.mp4'}")


if __name__ == "__main__":
    main()
