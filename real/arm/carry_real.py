#!/usr/bin/env python3
"""Перенос кубика в коробку на реальном стенде (алгоритм, вторая половина
связки «политика поднимает — алгоритм несёт»; сим-версия — mlsim/carry.py).

Поза коробки: метка ArUco ID 16 (60 мм) в дне печатной коробки
(real/box/out) видна верхней камере, пока коробка пуста. PnP по метке ->
T_base_box через калибровку T_base_topcam. Снимается один раз командой
`box` и сохраняется в real/calib/box_pose.json (с кубиком внутри метка
закрыта).

Перенос из состояния «кубик в губках» (любая поза руки):
  дожать губки -> подняться над текущей точкой до TRANSPORT_Z -> над центр
  коробки -> опуститься до LOWER_Z (низ кубика над бортом 44 мм) -> раскрыть
  -> отойти вверх -> парковка. IK и движения — те же, что у сборщика
  (collect_pilot.move / plan), floor-guard и самостолкновения — в драйвере.

  carry_real.py box                 # найти коробку верхней камерой и сохранить позу
  carry_real.py test [--attempts 3] # захват сборщиком (без нейросети) + перенос в коробку
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
from arm_driver import ArmDriver, GuardError, NAMES  # noqa: E402
from collect_pilot import cube_by_wrist, look, move, plan, reachable, waypoints  # noqa: E402
from ik import ArmIK  # noqa: E402

BOX_ID, BOX_MARKER_M = 16, 0.060
BOX_POSE = A.CALIB / "box_pose.json"
TRANSPORT_Z = 0.08          # предел вертикального подхода IK (как у сборщика)
LOWER_Z = 0.072             # tcp: низ кубика (tcp − 14 мм) над бортом 44 мм с запасом 14 мм
RIM_HALF = 0.055


def find_box(Xt):
    """T_base_box (центр дна коробки, z = 0) по метке в дне, или None."""
    img, dets = A.stable_view(A.CAM_TOP)
    if BOX_ID not in dets:
        return None, img
    px = A.undistort(np.asarray(dets[BOX_ID], np.float32), A.K_TOP, A.D_TOP).astype(np.float32)
    s = BOX_MARKER_M / 2
    obj = np.array([[-s, s, 0], [s, s, 0], [s, -s, 0], [-s, -s, 0]], np.float32)
    ok, rvec, tvec = cv2.solvePnP(obj, px, A.K_TOP, None, flags=cv2.SOLVEPNP_IPPE_SQUARE)
    if not ok:
        return None, img
    T_cm = np.eye(4)
    T_cm[:3, :3] = cv2.Rodrigues(rvec)[0]
    T_cm[:3, 3] = tvec.ravel()
    T = Xt @ T_cm
    # глубина PnP по метке в 0,4 м шумит: центр берём лучом камеры на плоскость дна (z = 8 мм)
    c, ray = Xt[:3, 3], T[:3, 3] - Xt[:3, 3]
    T[:3, 3] = c + ray * (0.008 - c[2]) / ray[2]
    T[2, 3] = 0.0
    return T, img


def load_box():
    if not BOX_POSE.exists():
        raise SystemExit(f"нет {BOX_POSE}: снимите позу коробки командой `carry_real.py box` (коробка пустая)")
    b = json.loads(BOX_POSE.read_text())
    return np.array(b["xy"]), b


def carry_to_box(d, ik, box_xy, grip_hold=A.GRIP_CLOSED, tick=None):
    """Из позы с кубиком в губках — в коробку. -> dict с фазами и ошибками."""
    log = {}
    q_now, _ = d.read()
    q = d.q_model(q_now)
    # дожать губки на месте (политика смыкает слабее сборщика)
    d.goto(np.concatenate([q_now[:5], [min(float(q_now[5]), grip_hold)]]), 0.4, mode="grasp", tick=tick)
    F = d.fk_T(d.read()[0][:5])
    xy0 = F[:2, 3]
    z0 = float(F[2, 3])
    yaw = None
    if z0 < TRANSPORT_Z - 0.01:
        q = move(d, ik, [*xy0, TRANSPORT_Z], grip_hold, 1.5, q, yaw, mode="grasp", tick=tick)
    log["up"] = round(float(d.fk_T(d.read()[0][:5])[2, 3]), 3)
    q = move(d, ik, [*box_xy, TRANSPORT_Z], grip_hold, 2.5, q, yaw, mode="transport", tick=tick)
    q = move(d, ik, [*box_xy, LOWER_Z], grip_hold, 1.2, q, yaw, mode="grasp", tick=tick)
    d.settle(0.5, 0.3, tick=tick)
    F = d.fk_T(d.read()[0][:5])
    log["over_box_err_mm"] = round(float(np.linalg.norm(F[:2, 3] - box_xy)) * 1000, 1)
    d.goto(np.concatenate([d.q_cmd(q), [A.GRIP_WIDE]]), 0.6, mode="grasp", tick=tick)
    d.settle(0.6, 0.3, tick=tick)
    q = move(d, ik, [*box_xy, TRANSPORT_Z + 0.01], A.GRIP_WIDE, 1.2, q, yaw, mode="grasp", tick=tick)
    d.goto(A.PARK + [A.GRIP_WIDE], 3.5, mode="transport", tick=tick)
    d.settle(1.0, 0.5, tick=tick)
    return log


def cmd_box(a):
    res = json.loads(A.OUT.read_text())
    Xt = np.array(res["T_base_topcam"])
    T, img = find_box(Xt)
    if T is None:
        raise SystemExit(f"метка коробки ID {BOX_ID} не видна верхней камере (коробка должна быть пустой и в кадре)")
    xy = T[:2, 3]
    r = float(np.hypot(*xy))
    info = {"xy": xy.tolist(), "r": round(r, 3), "T_base_box": T.tolist(), "ts": time.time(),
            "date": time.strftime("%Y-%m-%d %H:%M"), "marker_id": BOX_ID}
    BOX_POSE.write_text(json.dumps(info, indent=2))
    ok_reach = 0.15 <= r <= 0.29
    print(f"коробка: центр {np.round(xy, 3).tolist()} м, r={r:.3f} м -> {BOX_POSE}"
          + ("" if ok_reach else "  ВНИМАНИЕ: вне удобной досягаемости 15–29 см, переставьте"))


def cmd_test(a):
    """Захват сборщиком (верхняя камера + доводка по кисти) и перенос в коробку."""
    box_xy, _ = load_box()
    res = json.loads(A.OUT.read_text())
    Xt, Xw = np.array(res["T_base_topcam"]), np.array(res["T_gripper_wristcam"])
    ik = ArmIK(A._MODEL)
    d = ArmDriver()
    d.torque(True)
    d.hold()
    results = []
    try:
        for k in range(a.attempts):
            d.goto(A.PARK + [A.GRIP_WIDE], 4.0, mode="grasp")
            d.settle()
            cubes = look(Xt)
            ok = [c for c, T in cubes.items() if reachable(T) and np.linalg.norm(T[:2, 3] - box_xy) > 0.09]
            if not ok:
                print("нет досягаемого куба вне коробки:", {c: np.round(T[:2, 3], 3).tolist() for c, T in cubes.items()})
                break
            color = ok[0]
            P0, yaw = cubes[color][:3, 3].copy(), A.cube_yaw(cubes[color])
            w = waypoints(P0[:2])
            q = d.q_model(d.read()[0])
            try:
                q = move(d, ik, w["above"], A.GRIP_WIDE, 3.0, q, yaw)
                q = move(d, ik, w["hover"], A.GRIP_WIDE, 1.5, q, yaw)
                d.settle(1.5, 0.5)
                fix = cube_by_wrist(d, color, Xw)
                hov, low = w["hover"], w["low"]
                if fix is not None and np.linalg.norm(fix[0] - P0[:2]) < 0.04:
                    w2 = waypoints(fix[0])
                    hov, low, yaw = w2["hover"], w2["low"], fix[1]
                    q = move(d, ik, hov, A.GRIP_WIDE, 1.0, q, yaw)
                q = move(d, ik, low, A.GRIP_WIDE, 1.5, q, yaw, mode="grasp")
                d.goto(np.concatenate([d.q_cmd(q), [A.GRIP_CLOSED]]), 0.7, mode="grasp")
                d.settle(0.5, 0.3)
                q = move(d, ik, w["above"], A.GRIP_CLOSED, 1.5, q, yaw, mode="grasp")
                d.settle(1.0, 0.5)
                t0 = time.time()
                log = carry_to_box(d, ik, box_xy)
                after = look(Xt)
                in_box = color in after and np.linalg.norm(after[color][:2, 3] - box_xy) < RIM_HALF
                hidden = color not in after      # кубик в коробке закрывает метку дна, а сам виден сверху
                r = {"attempt": k + 1, "cube": color, "carry": log, "seconds": round(time.time() - t0, 1),
                     "cube_after": None if color not in after else np.round(after[color][:2, 3], 3).tolist(),
                     "in_box": bool(in_box), "cube_not_seen": hidden}
            except GuardError as e:
                r = {"attempt": k + 1, "cube": color, "error": str(e)}
                print("  guard:", e)
                try:
                    d.goto(np.concatenate([d.read()[0][:5], [A.GRIP_WIDE]]), 0.7, mode="grasp")
                    d.lift()
                    d.goto(A.PARK + [A.GRIP_WIDE], 4.0, mode="grasp")
                except GuardError as e2:
                    print("  восстановление:", e2)
                    results.append(r)
                    break
            results.append(r)
            print(f"попытка {k+1}: {'В КОРОБКЕ' if r.get('in_box') else 'мимо'} {r}", flush=True)
    finally:
        d.close()
    out = A._ROOT / "real/data/policy_runs" / f"carry-test-{time.strftime('%m%d-%H%M')}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, ensure_ascii=False, indent=1))
    print(f"ИТОГ: {sum(1 for r in results if r.get('in_box'))}/{len(results)} в коробке; {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["box", "test"])
    ap.add_argument("--attempts", type=int, default=3)
    a = ap.parse_args()
    {"box": cmd_box, "test": cmd_test}[a.cmd](a)


if __name__ == "__main__":
    main()
