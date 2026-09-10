#!/usr/bin/env python3
"""Eye-in-hand для камеры ГРИППЕРА: кубы ЛЕЖАТ на столе, рука смотрит
на них кистью с разных ракурсов, решается X = гриппер->wrist-камера.

Идея Владимира: неподвижный куб на столе — мишень eye-in-hand
калибровки wrist-камеры. После неё цель захвата видна прямо в системе
кисти, и доводка не зависит от люфтов глобальной кинематики.
"""

import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

_ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(_ROOT / "real" / "arm"))
sys.path.insert(0, str(_ROOT / "mlsim"))
from arm_driver import ArmDriver, GuardError
from cube_cv import estimate_cubes_cam

CAM_JPG = Path("/tmp/roboom_wrist/latest.jpg")
N_TARGET = 40
# позы «кисть смотрит вниз на зону»: наклон кисти к столу обязателен
POSE_LO = np.array([-0.8, -0.9, 0.4, 0.4, -1.1])
POSE_HI = np.array([0.8, -0.1, 1.4, 1.5, 1.1])
OUT = _ROOT / "real" / "calib" / "handeye_wrist.json"

calib = json.loads((_ROOT / "real" / "calib" / "wrist_intrinsics.json").read_text())
K = np.array(calib["K"])
DIST = np.array(calib["dist"])


def grab_frame():
    for _ in range(3):
        img = cv2.imread(str(CAM_JPG))
        if img is not None:
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        time.sleep(0.1)
    return None


def see_cube():
    img = grab_frame()
    if img is None:
        return None
    est = estimate_cubes_cam(img, K, DIST)
    best = None
    for color, e in est.items():
        if e["reproj_px"] > 2.0:
            continue
        cos = max(-float(np.dot(m[:3, 2], m[:3, 3]))
                  / max(np.linalg.norm(m[:3, 3]), 1e-9)
                  for m in e["markers_cam"])
        if cos < 0.45:
            continue
        if best is None or e["reproj_px"] < best[1]["reproj_px"]:
            best = (color, e)
    return best


def collect(d):
    rng = np.random.default_rng(17)
    pairs, segments = [], [[]]
    tried = 0
    guard_streak = 0
    while len(pairs) < N_TARGET and tried < 140:
        tried += 1
        q5 = rng.uniform(POSE_LO, POSE_HI)
        q_now, _ = d.read()
        q_target = np.concatenate([q5, [q_now[5]]])
        try:
            dq = float(np.max(np.abs(q5 - q_now[:5])))
            d.goto(q_target, max(1.2, 2.4 * dq))
            guard_streak = 0
        except GuardError as e:
            guard_streak += 1
            print(f"[{tried}] guard: {e}")
            if guard_streak >= 3:
                print("  восстановление: еду в среднюю позу")
                try:
                    d.hold()
                    time.sleep(0.4)
                    d.goto([0.0, -0.35, 0.6, 0.4, 0.0, d.read()[0][5]], 3.0,
                           mode="grasp")
                    guard_streak = 0
                except GuardError as e2:
                    print("  восстановление не удалось:", e2)
            continue
        d.settle()                      # 2 с осадки + стабилизация чтений
        got = see_cube()
        for droll in (0.8, -1.6, 2.4):
            if got is not None:
                break
            q_now2, _ = d.read()
            q_try = q_now2.copy()
            q_try[4] = float(np.clip(q_try[4] + droll, -1.5, 1.5))
            try:
                d.goto(q_try, 0.9)
            except GuardError:
                break
            d.settle()
            got = see_cube()
        if got is None:
            print(f"[{tried}] куб не виден")
            continue
        time.sleep(0.5)
        got2 = see_cube()
        if got2 is None or got2[0] != got[0]:
            continue
        dmm = np.linalg.norm(got2[1]["T_cam"][:3, 3] - got[1]["T_cam"][:3, 3]) * 1000
        if dmm > 3.5:
            print(f"[{tried}] нестационарно ({dmm:.1f} мм)")
            continue
        q_read, _ = d.read()
        pairs.append((q_read[:5].copy(), got2[1]["T_cam"]))
        segments[-1].append(pairs[-1])
        print(f"пар: {len(pairs)}/{N_TARGET} (поза {tried}, {got2[0]}, "
              f"reproj {got2[1]['reproj_px']:.2f})")
    return [s for s in segments if len(s) >= 2]


def solve(segments, d):
    As, Bs = [], []
    for seg in segments:
        Fs = [d.fk_T(q5) for q5, _ in seg]
        Cs = [T for _, T in seg]
        for i in range(len(seg) - 1):
            for j in (i + 1, i + 2):
                if j >= len(seg):
                    continue
                # eye-in-hand: F_i X C_i = F_j X C_j -> inv(F_j) F_i X = X C_j inv(C_i)
                A = np.linalg.inv(Fs[j]) @ Fs[i]
                B = Cs[j] @ np.linalg.inv(Cs[i])
                a = np.linalg.norm(cv2.Rodrigues(A[:3, :3])[0])
                b = np.linalg.norm(cv2.Rodrigues(B[:3, :3])[0])
                if abs(a - b) > np.radians(2.5) or a < np.radians(3):
                    continue
                As.append(A)
                Bs.append(B)
    print(f"относительных движений: {len(As)}")
    if len(As) < 6:
        raise SystemExit("мало движений для решения")

    def park(idx):
        M = np.zeros((3, 3))
        for k in idx:
            a = cv2.Rodrigues(As[k][:3, :3])[0].ravel()
            b = cv2.Rodrigues(Bs[k][:3, :3])[0].ravel()
            M += np.outer(b, a)
        w, V = np.linalg.eigh(M.T @ M)
        Rx = V @ np.diag(1 / np.sqrt(np.maximum(w, 1e-12))) @ V.T @ M.T
        lhs = [As[k][:3, :3] - np.eye(3) for k in idx]
        rhs = [Rx @ Bs[k][:3, 3] - As[k][:3, 3] for k in idx]
        t = np.linalg.lstsq(np.vstack(lhs), np.concatenate(rhs), rcond=None)[0]
        X = np.eye(4)
        X[:3, :3] = Rx
        X[:3, 3] = t
        return X

    idx = list(range(len(As)))
    X = park(idx)
    for _ in range(2):
        res = sorted(
            ((float(np.linalg.norm((As[k] @ X - X @ Bs[k])[:3, 3])), k)
             for k in idx))
        idx = [k for _, k in res[:max(3, int(len(res) * 0.75))]]
        X = park(idx)
    resid = float(np.median([np.linalg.norm((As[k] @ X - X @ Bs[k])[:3, 3])
                             for k in idx])) * 1000
    return X, resid


def main():
    d = ArmDriver()
    d.torque(True)
    d.hold()
    try:
        segments = collect(d)
        pairs = [p for s in segments for p in s]
        print(f"сегменты: {[len(s) for s in segments]}")
        import os
        n = 2
        while OUT.with_name(f"handeye_pairs{n}.json").exists():
            n += 1
        raw_path = OUT.with_name(f"handeye_pairs{n}.json")
        raw_path.write_text(json.dumps(
            [[[q.tolist() for q, _ in seg], [T.tolist() for _, T in seg]]
             for seg in segments]))
        print(f"сырые пары: {raw_path}")
        X, resid = solve(segments, d)
        # качество сквозняком: хват постоянен, значит G = fk^-1 X C обязан
        # совпадать по всем парам; разброс G — честная сквозная ошибка
        Gs = [np.linalg.inv(d.fk_T(q5)) @ X @ C for q5, C in pairs]
        g0 = Gs[len(Gs) // 2]
        spread = sorted(float(np.linalg.norm(G[:3, 3] - g0[:3, 3])) * 1000
                        for G in Gs)
        print(f"разброс хвата G: медиана "
              f"{spread[len(spread) // 2]:.1f} мм, p90 "
              f"{spread[int(len(spread) * 0.9)]:.1f} мм")
        cam_pos = X[:3, 3]
        print(f"X (гриппер->wrist-камера): смещение {np.round(cam_pos, 3)}, "
              f"медианная невязка {resid:.1f} мм")
        OUT.write_text(json.dumps({
            "T_gripper2wristcam": X.tolist(), "pairs": len(pairs),
            "median_residual_mm": resid,
            "date": time.strftime("%Y-%m-%d %H:%M")}, indent=2))
        print(f"сохранено: {OUT}")
        print("возвращаюсь в среднюю позу")
        d.goto([0.0, -0.35, 0.6, 0.4, 0.0, d.read()[0][5]], 4.0)
    finally:
        d.close()


if __name__ == "__main__":
    main()
