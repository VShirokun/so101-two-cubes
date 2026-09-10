#!/usr/bin/env python3
"""Уточнение hand-eye + кинематики по сырым парам тура (оффлайн).

Дешёвая рука: реальный ход суставов не равен модельному, и линейная
привязка упоров даёт мультипликативную ошибку углов. Оптимизируем разом:
масштаб и смещение пяти суставов, X (камера->база) и G (куб в гриппере),
минимизируя непостоянство G по всем позам тура. FK — модель MJCF.
"""

import json
import sys
import time
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as Rot

_ROOT = Path(__file__).parents[2]
PAIRS = Path(__file__).parent / "handeye_pairs.json"
HE = Path(__file__).parent / "handeye_c920.json"

model = mujoco.MjModel.from_xml_path(
    str(_ROOT / "mlsim" / "models" / "so101" / "pick_place.xml"))
data = mujoco.MjData(model)
sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "gripperframe")

Qs, Cs = [], []
for f in sorted(Path(__file__).parent.glob("handeye_pairs*.json")):
    segs = json.loads(f.read_text())
    Qs += [q for qs, _ in segs for q in qs]
    Cs += [T for _, Ts in segs for T in Ts]
    print(f"{f.name}: +{sum(len(qs) for qs, _ in segs)} поз")
Q, C = np.array(Qs), np.array(Cs)
print(f"поз всего: {len(Q)}")


def fk(q5):
    data.qpos[:] = 0
    data.qpos[:5] = q5
    mujoco.mj_kinematics(model, data)
    T = np.eye(4)
    T[:3, :3] = data.site_xmat[sid].reshape(3, 3)
    T[:3, 3] = data.site_xpos[sid]
    return T


def unpack(p):
    scale = p[0:5]
    off = p[5:10]
    Xr, Xt = Rot.from_rotvec(p[10:13]).as_matrix(), p[13:16]
    Gr, Gt = Rot.from_rotvec(p[16:19]).as_matrix(), p[19:22]
    X = np.eye(4); X[:3, :3] = Xr; X[:3, 3] = Xt
    G = np.eye(4); G[:3, :3] = Gr; G[:3, 3] = Gt
    return scale, off, X, G


def resid(p):
    scale, off, X, G = unpack(p)
    out = []
    for q5, Ci in zip(Q, C):
        qq = scale * q5 + off
        T_pred = fk(qq) @ G          # куб в базе через руку
        T_cv = X @ Ci                # куб в базе через камеру
        out.append((T_pred[:3, 3] - T_cv[:3, 3]) * 10)         # позиции (см)
        dR = T_pred[:3, :3].T @ T_cv[:3, :3]
        out.append(Rot.from_matrix(dR).as_rotvec())            # углы (рад)
    # регуляризация против переобучения: масштабы у единицы, смещения малые
    out.append((scale - 1.0) * 20)
    out.append(off * 6)
    return np.concatenate(out)


X0 = np.array(json.loads(HE.read_text())["T_cam2base"])
p0 = np.concatenate([np.ones(5), np.zeros(5),
                     Rot.from_matrix(X0[:3, :3]).as_rotvec(), X0[:3, 3],
                     np.zeros(3), [0, 0, -0.02]])
t0 = time.time()
r0 = resid(p0)[:len(Q) * 6].reshape(-1, 3)
print(f"старт: RMS позиций {np.linalg.norm(r0[::2])*10/np.sqrt(len(Q)):.1f} мм")
sol = least_squares(resid, p0, method="lm", max_nfev=4000)
scale, off, X, G = unpack(sol.x)
r = resid(sol.x)[:len(Q) * 6].reshape(-1, 3)
pos_err = np.linalg.norm(r[::2], axis=1) * 10   # мм
ang_err = np.degrees(np.linalg.norm(r[1::2], axis=1))
print(f"после уточнения ({time.time()-t0:.0f} c): "
      f"позиция куба медиана {np.median(pos_err):.1f} мм, p90 {np.percentile(pos_err, 90):.1f} мм; "
      f"угол медиана {np.median(ang_err):.2f}°")
print("масштабы суставов:", np.round(scale, 4))
print("смещения, °:", np.round(np.degrees(off), 2))
print("камера в базе:", np.round(X[:3, 3], 3))
out = {
    "joint_scale": scale.tolist(), "joint_offset": off.tolist(),
    "T_cam2base": X.tolist(), "G_gripper_cube": G.tolist(),
    "pos_err_mm_median": float(np.median(pos_err)),
    "pos_err_mm_p90": float(np.percentile(pos_err, 90)),
    "date": time.strftime("%Y-%m-%d %H:%M"),
}
(Path(__file__).parent / "handeye_refined.json").write_text(
    json.dumps(out, indent=2))
print("сохранено: handeye_refined.json")
