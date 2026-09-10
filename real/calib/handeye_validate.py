#!/usr/bin/env python3
"""Валидация уточнённой калибровки на СВЕЖИХ позах: куб всё ещё в губках."""
import json, sys, time
from pathlib import Path
import cv2
import numpy as np

_ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(_ROOT / "real" / "arm"))
sys.path.insert(0, str(_ROOT / "mlsim"))
from arm_driver import ArmDriver, GuardError
from cube_cv import estimate_cubes_cam
import mujoco

ref = json.loads((Path(__file__).parent / "handeye_refined.json").read_text())
scale = np.array(ref["joint_scale"])
off = np.array(ref["joint_offset"])
X = np.array(ref["T_cam2base"])
G = np.array(ref["G_gripper_cube"])
calib = json.loads((Path(__file__).parent / "c920_intrinsics.json").read_text())
K, DIST = np.array(calib["K"]), np.array(calib["dist"])
print("G (куб отн. кисти):", np.round(G[:3, 3], 3))

POSES = [
    [0.2, -0.6, 1.05, 0.1, 0.6], [-0.35, -0.4, 0.7, 0.5, -0.4],
    [0.55, -0.45, 0.85, 0.4, 0.9], [-0.15, -0.65, 1.15, -0.1, -0.8],
    [0.35, -0.3, 0.6, 0.7, 0.2], [-0.5, -0.5, 0.95, 0.25, 0.7],
    [0.05, -0.35, 0.65, 0.55, -1.0], [0.65, -0.55, 1.1, 0.0, 0.4],
    [-0.2, -0.25, 0.5, 0.75, 1.1], [0.4, -0.7, 1.25, 0.3, -0.5],
]
d = ArmDriver()
d.torque(True); d.hold()
errs = []
try:
    for k, q5 in enumerate(POSES):
        q_now, _ = d.read()
        try:
            d.goto(list(q5) + [float(q_now[5])], 2.5)
        except GuardError as e:
            print(f"поза {k}: guard {e}")
            continue
        d.settle()
        img = cv2.imread("/tmp/roboom_cam/latest.jpg")
        est = estimate_cubes_cam(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), K, DIST)
        got = est.get("red") or est.get("green")
        if got is None or got["reproj_px"] > 2.0:
            print(f"поза {k}: куб не виден")
            continue
        q_read, _ = d.read()
        qq = scale * q_read[:5] + off
        d.scratch.qpos[:] = 0
        d.scratch.qpos[:5] = qq
        mujoco.mj_kinematics(d.model, d.scratch)
        T_fk = np.eye(4)
        T_fk[:3, :3] = d.scratch.site_xmat[d.sid].reshape(3, 3)
        T_fk[:3, 3] = d.scratch.site_xpos[d.sid]
        p_arm = (T_fk @ G)[:3, 3]
        p_cam = (X @ got["T_cam"])[:3, 3]
        e = float(np.linalg.norm(p_arm - p_cam) * 1000)
        errs.append(e)
        print(f"поза {k}: рука-vs-камера {e:.1f} мм")
    if errs:
        print(f"\nВАЛИДАЦИЯ: медиана {np.median(errs):.1f} мм, "
              f"max {max(errs):.1f} мм на {len(errs)} свежих позах")
    d.goto([0.0, -0.35, 0.6, 0.4, 0.0, d.read()[0][5]], 3.0)
finally:
    d.close()
