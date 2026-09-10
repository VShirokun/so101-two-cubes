#!/usr/bin/env python3
"""Автокалибровка реального стенда по двум камерам и двум кубам на столе.

Перенос mlsim/test_autocalib.py на живую руку. Кубы ЛЕЖАТ на столе, их видят
верхняя камера (C920) и камера гриппера. Рука объезжает позы, целясь камерой
кисти на кубы (прицел — по грубой старой hand-eye и номинальному креплению
камеры из MJCF); в каждой позе после осадки 2 с снимаются углы меток с обеих
камер. Решение — bundle adjustment по пикселям углов: неизвестны поза верхней
камеры в базе робота, поза камеры гриппера в кисти, поправки суставов
(масштаб/смещение, с приором) и x, y, yaw каждого куба (куб лежит: высота и
грань вверх известны).

  autocalib_real.py tour [--layout 1] [--poses 24]  наблюдения -> autocalib_obs.json
  autocalib_real.py solve                            BA -> autocalib_real.json + mp4
  autocalib_real.py selftest                         решатель на синтетике, без руки

Стоп руки: touch /tmp/roboom_arm/stop.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as Rot

_ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(_ROOT / "mlsim"))
from cube_cv import (CUBE_HALF, DICT, FACES, _ID2FACE, _OBJ_PTS,  # noqa: E402
                     _avg_rotations, _marker_T_cube)

CALIB = _ROOT / "real" / "calib"
OBS = CALIB / "autocalib_obs.json"
OUT = CALIB / "autocalib_real.json"
FRAMES = Path("/tmp/roboom_autocalib")
CAM_TOP, CAM_WR = Path("/tmp/roboom_cam/latest.jpg"), Path("/tmp/roboom_wrist/latest.jpg")
MID = [0.0, -0.35, 0.6, 0.4, 0.0]
GRIP = 1.0
POSE_LO = np.array([-0.9, -1.0, 0.3, 0.2, -1.2])
POSE_HI = np.array([0.9, 0.0, 1.5, 1.5, 1.2])
W_PRIOR = 3.0        # px на сигму приора поправок суставов (3 % масштаба, 2°)


def _cam(name):
    c = json.loads((CALIB / name).read_text())
    return np.array(c["K"]), np.array(c["dist"]), tuple(c["image_size"])


K_TOP, D_TOP, SZ_TOP = _cam("c920_intrinsics.json")
K_WR, D_WR, SZ_WR = _cam("wrist_intrinsics.json")
X_TOP0 = np.array(json.loads((CALIB / "handeye_refined.json").read_text())["T_cam2base"])
PREV = json.loads(OUT.read_text()) if OUT.exists() else None   # прошлое решение — старт
if PREV:
    X_TOP0 = np.array(PREV["T_base_topcam"])

_MODEL = mujoco.MjModel.from_xml_path(str(_ROOT / "mlsim/models/so101/pick_place.xml"))
_DATA = mujoco.MjData(_MODEL)
_SID = mujoco.mj_name2id(_MODEL, mujoco.mjtObj.mjOBJ_SITE, "gripperframe")
_CID = mujoco.mj_name2id(_MODEL, mujoco.mjtObj.mjOBJ_CAMERA, "wrist_cam")
_PARAMS = cv2.aruco.DetectorParameters()
_PARAMS.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX   # без него углы дрожат на 2-3 px
DET = cv2.aruco.ArucoDetector(DICT, _PARAMS)
_OBJ_H = np.hstack([_OBJ_PTS, np.ones((4, 1))])


def fk_T(q5):
    """T база<-gripperframe по углам (как в драйвере)."""
    _DATA.qpos[:] = 0
    _DATA.qpos[:5] = q5
    mujoco.mj_kinematics(_MODEL, _DATA)
    T = np.eye(4)
    T[:3, :3] = _DATA.site_xmat[_SID].reshape(3, 3)
    T[:3, 3] = _DATA.site_xpos[_SID]
    return T


def _nominal_xw():
    """Номинальное крепление wrist_cam из MJCF: кисть<-камера (оси OpenCV)."""
    _DATA.qpos[:] = 0
    mujoco.mj_forward(_MODEL, _DATA)
    C = np.eye(4)
    C[:3, :3] = _DATA.cam_xmat[_CID].reshape(3, 3) @ np.diag([1, -1, -1])
    C[:3, 3] = _DATA.cam_xpos[_CID]
    return np.linalg.inv(fk_T(np.zeros(5))) @ C


X_W0 = np.array(PREV["T_gripper_wristcam"]) if PREV else _nominal_xw()
S0 = np.array(PREV["joint_scale"]) if PREV else np.ones(5)
O0 = np.radians(PREV["joint_offset_deg"]) if PREV else np.zeros(5)


# --- кадры и метки ---------------------------------------------------------

def grab(path):
    for _ in range(50):
        if time.time() - path.stat().st_mtime < 1.0:
            img = cv2.imread(str(path))
            if img is not None:
                return img
        time.sleep(0.1)
    raise RuntimeError(f"{path}: камера не отдаёт свежих кадров")


def corners_raw(img_bgr):
    """{id: 4x2 сырых пикселей углов} — только метки кубов."""
    c, ids, _ = DET.detectMarkers(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY))
    if ids is None:
        return {}
    return {int(m): q.reshape(4, 2) for q, m in zip(c, ids.ravel())
            if int(m) in _ID2FACE}


def undistort(px, K, dist):
    return cv2.undistortPoints(np.asarray(px, np.float32).reshape(-1, 1, 2),
                               K, dist, P=K).reshape(-1, 2).astype(float)


def stable_view(path, n=5, tol_px=1.5, tries=4):
    """n кадров подряд: метка засчитывается, если видна хотя бы в 3 и её углы
    дрожат меньше tol (рука стоит, кадр живой). -> (кадр, {id: медиана})."""
    for _ in range(tries):
        imgs, dets = [], []
        for _ in range(n):
            imgs.append(grab(path))
            dets.append(corners_raw(imgs[-1]))
            time.sleep(0.1)
        out = {}
        for m in set.union(*map(set, dets)):
            v = [d[m] for d in dets if m in d]
            if len(v) >= 3 and max(np.abs(a - v[0]).max() for a in v) < tol_px:
                out[m] = np.median(v, 0)
        if out or not any(dets):
            return imgs[-1], out
    return imgs[-1], {}


def cube_poses(corners, K, dist=None):
    """{color: T камера<-куб} по PnP каждой метки, среднее по меткам куба."""
    per = {}
    for mid, px in corners.items():
        color, fi = _ID2FACE[mid]
        ok, rvec, tvec = cv2.solvePnP(_OBJ_PTS, np.asarray(px, np.float32), K,
                                      dist, flags=cv2.SOLVEPNP_ITERATIVE)
        if ok:
            T = np.eye(4)
            T[:3, :3] = cv2.Rodrigues(rvec)[0]
            T[:3, 3] = tvec.ravel()
            per.setdefault(color, []).append(T @ np.linalg.inv(_marker_T_cube(fi)))
    out = {}
    for color, Ts in per.items():
        T = np.eye(4)
        T[:3, :3] = _avg_rotations([t[:3, :3] for t in Ts])
        T[:3, 3] = np.mean([t[:3, 3] for t in Ts], 0)
        out[color] = T
    return out


def prepare(recs):
    """Сырые пиксели -> без дисторсии (pinhole K); сырые остаются для рисования."""
    for r in recs:
        for cam, K, D in (("wrist", K_WR, D_WR), ("top", K_TOP, D_TOP)):
            raw = {int(m): np.asarray(v, float) for m, v in r[cam].items()}
            r[cam + "_raw"] = raw
            r[cam] = {m: undistort(v, K, D) for m, v in raw.items()}
    return recs


# --- геометрия решателя ----------------------------------------------------

def T_of(v):
    M = np.eye(4)
    M[:3, :3] = Rot.from_rotvec(v[:3]).as_matrix()
    M[:3, 3] = v[3:6]
    return M


def v_of(T):
    return np.concatenate([Rot.from_matrix(T[:3, :3]).as_rotvec(), T[:3, 3]])


def project(K, T_cam_base, pts):
    pc = (T_cam_base[:3, :3] @ pts.T).T + T_cam_base[:3, 3]
    uv = (K @ pc.T).T
    return uv[:, :2] / uv[:, 2:3]


def _cube_T(R0, x, y, psi):
    """Поза лежащего куба в базе: Rz(psi) R0, z = полуребро."""
    T = np.eye(4)
    T[:3, :3] = Rot.from_rotvec([0, 0, psi]).as_matrix() @ R0
    T[:3, 3] = [x, y, CUBE_HALF]
    return T


def cube_corners(T, color):
    return {mid: (T @ _marker_T_cube(fi) @ _OBJ_H.T).T[:, :3]
            for mid, (c, fi) in _ID2FACE.items() if c == color}


def corners_base(color, R0, x, y, psi):
    """Углы всех меток лежащего куба в базе."""
    return cube_corners(_cube_T(R0, x, y, psi), color)


def pose_err(A, B):
    dR = A[:3, :3].T @ B[:3, :3]
    return (float(np.linalg.norm(A[:3, 3] - B[:3, 3]) * 1000),
            float(np.degrees(np.linalg.norm(Rot.from_matrix(dR).as_rotvec()))))


def aim_ok(q5, targets, close=False):
    """Сколько кубов попадёт в кадр камеры кисти в позе q5 (номинальная
    камера, грубые позиции кубов): ось смотрит вниз, куб в центре кадра,
    дистанция 9-30 см; close — как при захвате: круто сверху, 8-18 см."""
    C = fk_T(q5) @ X_W0
    if C[2, 2] > (-0.8 if close else -0.5):
        return 0
    T_cb = np.linalg.inv(C)
    seen = 0
    for P in targets:
        pc = T_cb[:3, :3] @ P + T_cb[:3, 3]
        if not (0.08 < pc[2] < 0.18 if close else 0.09 < pc[2] < 0.30):
            continue
        u, v = (K_WR @ pc)[:2] / pc[2]
        if 80 < u < SZ_WR[0] - 80 and 60 < v < SZ_WR[1] - 60:
            seen += 1
    return seen


class Problem:
    """BA: p = [X_w(6), X_top(6), scale(5), offset(5), (x, y, yaw) x кубов]."""

    def __init__(self, recs, X_top0=X_TOP0, X_w0=X_W0, w_prior=W_PRIOR):
        self.recs = [dict(r, wrist=dict(r["wrist"])) for r in recs]  # prune не трогает исходные
        self.layouts = sorted({r["layout"] for r in recs})
        self.top = {}
        for L in self.layouts:
            acc = {}
            for r in recs:
                if r["layout"] == L:
                    for m, px in r["top"].items():
                        acc.setdefault(m, []).append(px)
            self.top[L] = {m: np.median(v, 0) for m, v in acc.items()}
        self.objs, init = [], []
        for L in self.layouts:
            for color, T_c in cube_poses(self.top[L], K_TOP).items():
                T_b = X_top0 @ T_c
                fi = int(np.argmax([(T_b[:3, :3] @ np.asarray(FACES[f][3], float))[2]
                                    for f in range(6)]))
                R0 = Rot.align_vectors([[0, 0, 1.0]],
                                       [np.asarray(FACES[fi][3], float)])[0].as_matrix()
                Rc = T_b[:3, :3] @ R0.T
                self.objs.append((L, color, R0))
                init += [T_b[0, 3], T_b[1, 3], float(np.arctan2(Rc[1, 0], Rc[0, 0]))]
        self.w_prior = w_prior
        self.p0 = np.concatenate([v_of(X_w0), v_of(X_top0), S0, O0, init])

    def unpack(self, p):
        return (T_of(p[0:6]), T_of(p[6:12]), p[12:17], p[17:22],
                p[22:].reshape(-1, 3))

    def cubes_base(self, p):
        return {(L, c): corners_base(c, R0, *xyp)
                for (L, c, R0), xyp in zip(self.objs, self.unpack(p)[4])}

    def terms(self, p, recs=None, top=True):
        """Невязки по меткам: (вид, ключ, id, 4x2 px)."""
        Xw, Xt, s, o, _ = self.unpack(p)
        cb = self.cubes_base(p)
        T_tb = np.linalg.inv(Xt)
        if top:
            for L in self.layouts:
                for m, px in self.top[L].items():
                    key = (L, _ID2FACE[m][0])
                    if key in cb:
                        yield "top", L, m, project(K_TOP, T_tb, cb[key][m]) - px
        for i, r in enumerate(self.recs if recs is None else recs):
            if not r["wrist"]:
                continue
            T_wb = np.linalg.inv(fk_T(np.asarray(r["q"]) * s + o) @ Xw)
            for m, px in r["wrist"].items():
                key = (r["layout"], _ID2FACE[m][0])
                if key in cb:
                    yield "wrist", i, m, project(K_WR, T_wb, cb[key][m]) - px

    def resid(self, p):
        s, o = p[12:17], p[17:22]
        out = [t[3].ravel() for t in self.terms(p)]
        out += [(s - 1) / 0.03 * self.w_prior, o / np.radians(2) * self.w_prior]
        return np.concatenate(out)

    def solve(self):
        best = None
        for k in range(4):   # поворот камеры кисти вокруг оптической оси
            p0 = self.p0.copy()
            Xw = T_of(p0[:6])
            Xw[:3, :3] = Xw[:3, :3] @ Rot.from_rotvec([0, 0, np.pi / 2 * k]).as_matrix()
            p0[:6] = v_of(Xw)
            sol = least_squares(self.resid, p0, method="lm", max_nfev=4000)
            if best is None or sol.cost < best.cost:
                best = sol
        return best

    def prune(self, p, thr_px=3.0):
        """Выбросить метки с невязкой выше порога (блик, ложная детекция);
        порог относительный, чтобы плохая модель не выбрасывала всё подряд."""
        bad = 0
        errs = [(kind, key, m, float(np.sqrt((e ** 2).mean())))
                for kind, key, m, e in self.terms(p)]
        thr = max(thr_px, 4 * float(np.median([e for *_, e in errs])))
        for kind, key, m, e in errs:
            if e > thr:
                (self.top[key] if kind == "top" else self.recs[key]["wrist"]).pop(m)
                bad += 1
        return bad

    def wrist_rms(self, p, recs):
        e = [t[3].ravel() for t in self.terms(p, recs, top=False)]
        return float(np.sqrt(np.mean(np.concatenate(e) ** 2))) if e else float("nan")

    def cube_err_mm(self, p, recs):
        """Куб по PnP камеры кисти -> в базу через F X_w; расстояние до
        куба из решения — ошибка «куда ехать за кубом», мм."""
        Xw, _, s, o, cubes = self.unpack(p)
        pos = {(L, c): np.array([x, y, CUBE_HALF])
               for (L, c, _), (x, y, _) in zip(self.objs, cubes)}
        errs = []
        for r in recs:
            if not r["wrist"]:
                continue
            F = fk_T(np.asarray(r["q"]) * s + o)
            for c, T_c in cube_poses(r["wrist"], K_WR).items():
                if (r["layout"], c) in pos:
                    errs.append(np.linalg.norm((F @ Xw @ T_c)[:3, 3]
                                               - pos[(r["layout"], c)]) * 1000)
        return np.array(errs)


def fit(recs, w_prior=W_PRIOR, verbose=True):
    prob = Problem(recs, w_prior=w_prior)
    sol = prob.solve()
    bad = prob.prune(sol.x)
    if bad:
        if verbose:
            print(f"  выброшено меток с невязкой > 3 px: {bad}")
        sol = prob.solve()
    return prob, sol


# --- тур по позам ----------------------------------------------------------

def tour(layout, n_poses):
    sys.path.insert(0, str(Path(__file__).parent))
    from arm_driver import STOP, ArmDriver, GuardError
    FRAMES.mkdir(exist_ok=True)
    records = json.loads(OBS.read_text()) if OBS.exists() else []
    d = ArmDriver()
    d.torque(True)
    d.hold()
    try:
        d.lift()
        d.goto(MID + [GRIP], 4.0)
        d.settle()
        img, top = stable_view(CAM_TOP)
        poses = cube_poses(top, K_TOP, D_TOP)
        if len(poses) < 2:
            raise SystemExit(f"верхняя камера видит {len(poses)} куб(ов), нужны "
                             f"оба; метки: {sorted(top)}")
        targets = {c: (X_TOP0 @ T)[:3, 3] for c, T in poses.items()}
        print("кубы в базе (грубо, по старой hand-eye):",
              {c: np.round(p, 3).tolist() for c, p in targets.items()})
        k = sum(r["layout"] == layout for r in records)
        rng = np.random.default_rng(int(time.time()))
        got, tried, streak = 0, 0, 0
        while got < n_poses and tried < 6 * n_poses and not STOP.exists():
            tried += 1
            # чередуем: близко/далеко и какой куб в прицеле (кубы могут
            # лежать далеко друг от друга — оба в кадр попадают редко)
            cands = rng.uniform(POSE_LO, POSE_HI, size=(400, 5))
            aim = [list(targets.values())[(got // 2) % len(targets)]]
            scores = [aim_ok(q, aim, close=got % 2 == 1) for q in cands]
            if max(scores) == 0:
                continue
            q5 = cands[int(np.argmax(scores))]
            q_now, _ = d.read()
            try:
                d.goto(np.concatenate([q5, [GRIP]]),
                       max(1.5, 2.5 * float(np.max(np.abs(q5 - q_now[:5])))))
                streak = 0
            except GuardError as e:
                streak += 1
                print(f"[{tried}] guard: {e}")
                if streak >= 3:
                    print("  восстановление: в среднюю позу")
                    try:
                        d.goto(MID + [GRIP], 4.0)
                        streak = 0
                    except GuardError as e2:
                        print("  не удалось:", e2)
                continue
            d.settle()
            wr_img, wr = stable_view(CAM_WR)
            if not wr:
                print(f"[{tried}] камера кисти не видит меток")
                continue
            top_img, top = stable_view(CAM_TOP)
            q_read, _ = d.read()
            k += 1
            paths = {n: str(FRAMES / f"L{layout}_{k:02d}_{n}.jpg")
                     for n in ("wrist", "top")}
            cv2.imwrite(paths["wrist"], wr_img)
            cv2.imwrite(paths["top"], top_img)
            records.append({"layout": layout, "q": q_read[:5].tolist(),
                            "wrist": {m: v.tolist() for m, v in wr.items()},
                            "top": {m: v.tolist() for m, v in top.items()},
                            "wrist_jpg": paths["wrist"], "top_jpg": paths["top"],
                            "ts": time.time()})
            OBS.write_text(json.dumps(records))
            got += 1
            print(f"поза {got}/{n_poses}: кисть видит {sorted(wr)}, верх {sorted(top)}")
        print("возврат в среднюю позу")
        d.goto(MID + [GRIP], 4.0)
    finally:
        d.close()
    print(f"наблюдений всего: {len(records)} -> {OBS}")


# --- решение, отчёт, видео -------------------------------------------------

def _draw(img, K, dist, T_cam_base, cb_all, layout, raw):
    """Детекции (зелёные) и предсказания решения (красные) на сыром кадре."""
    for m, px in raw.items():
        cv2.polylines(img, [np.round(px).astype(np.int32)], True, (0, 220, 0), 2)
    for (L, c), cb in cb_all.items():
        if L != layout:
            continue
        for m, pts in cb.items():
            pc = (T_cam_base[:3, :3] @ pts.T).T + T_cam_base[:3, 3]
            n = -np.cross(pc[1] - pc[0], pc[3] - pc[0])
            if pc[:, 2].min() < 0.03 or np.dot(n, pc.mean(0)) >= 0:
                continue
            px, _ = cv2.projectPoints(pc, np.zeros(3), np.zeros(3), K, dist)
            px = px.reshape(4, 2)
            if (px < 0).any() or px[:, 0].max() > img.shape[1] or px[:, 1].max() > img.shape[0]:
                continue
            cv2.polylines(img, [np.round(px).astype(np.int32)], True, (0, 0, 255), 1)


def make_video(recs, prob, p, res, path):
    Xw, Xt, s, o, _ = prob.unpack(p)
    cb_all = prob.cubes_base(p)
    T_tb = np.linalg.inv(Xt)
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 15, (1280, 900))

    def emit(top, wr, lines, n):
        top = cv2.resize(top, (1280, 720))
        cv2.putText(top, "top camera C920: detected (green) vs predicted by solution (red)",
                    (14, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (40, 220, 40), 2, cv2.LINE_AA)
        wr = cv2.resize(wr, (240, 180))
        cv2.putText(wr, "wrist camera", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (40, 220, 40), 2, cv2.LINE_AA)
        panel = np.full((180, 1040, 3), 25, np.uint8)
        for i, ln in enumerate(lines[:5]):
            cv2.putText(panel, ln, (12, 32 + i * 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.68, (230, 230, 230), 2, cv2.LINE_AA)
        frame = np.vstack([top, np.hstack([wr, panel])])
        for _ in range(n):
            w.write(frame)

    shown = [r for r in recs if r["wrist"] and Path(r.get("wrist_jpg", "")).is_file()]
    for i, r in enumerate(shown):
        top, wr = cv2.imread(r["top_jpg"]), cv2.imread(r["wrist_jpg"])
        _draw(top, K_TOP, D_TOP, T_tb, cb_all, r["layout"], r["top_raw"])
        T_wb = np.linalg.inv(fk_T(np.asarray(r["q"]) * s + o) @ Xw)
        _draw(wr, K_WR, D_WR, T_wb, cb_all, r["layout"], r["wrist_raw"])
        err = prob.cube_err_mm(p, [r])
        emit(top, wr, [f"pose {i + 1}/{len(shown)}, layout {r['layout']}",
                       f"wrist reprojection RMS {prob.wrist_rms(p, [r]):.2f} px",
                       f"cube position from wrist camera vs solution: "
                       f"{np.mean(err):.1f} mm" if len(err) else ""], 12)
    if shown:
        top, wr = cv2.imread(shown[-1]["top_jpg"]), cv2.imread(shown[-1]["wrist_jpg"])
        _draw(top, K_TOP, D_TOP, T_tb, cb_all, shown[-1]["layout"], shown[-1]["top_raw"])
        emit(top, wr, ["AUTO-CALIBRATION RESULT (real rig):",
                       f"reprojection RMS {res['rms_px']:.2f} px over {res['n_poses']} poses",
                       f"held-out poses: wrist RMS {res['holdout']['wrist_rms_px']:.2f} px, "
                       f"cube position error median {res['holdout']['cube_err_mm_median']:.1f} mm, "
                       f"p90 {res['holdout']['cube_err_mm_p90']:.1f} mm",
                       f"top camera in robot base: {np.round(Xt[:3, 3], 3).tolist()} m; "
                       f"shift vs old hand-eye {res['vs_old_handeye_mm']:.0f} mm"], 60)
    w.release()


def solve(layouts=None):
    recs = prepare([r for r in json.loads(OBS.read_text())
                    if not layouts or r["layout"] in layouts])
    n_wr = sum(len(r["wrist"]) for r in recs)
    print(f"наблюдений: {len(recs)} поз, {n_wr} меток в камере кисти, "
          f"раскладок {sorted({r['layout'] for r in recs})}")
    prob, sol = fit(recs)
    p = sol.x
    Xw, Xt, s, o, cubes = prob.unpack(p)
    rms = float(np.sqrt(np.mean(np.concatenate(
        [t[3].ravel() for t in prob.terms(p)]) ** 2)))
    # контроль: те же данные с замороженными суставами (без поправок)
    _, sol_fix = fit(recs, w_prior=1e4, verbose=False)
    prob_fix = Problem(recs, w_prior=1e4)
    rms_fix = float(np.sqrt(np.mean(np.concatenate(
        [t[3].ravel() for t in prob_fix.terms(sol_fix.x)]) ** 2)))
    # hold-out: каждая 4-я поза не участвует в решении
    train = [r for i, r in enumerate(recs) if i % 4 != 3]
    held = [r for i, r in enumerate(recs) if i % 4 == 3 and r["wrist"]]
    prob_tr, sol_tr = fit(train, verbose=False)
    err_h = prob_tr.cube_err_mm(sol_tr.x, held)
    err_all = prob.cube_err_mm(p, recs)
    dp, da = pose_err(Xt, X_TOP0)
    res = {
        "T_base_topcam": Xt.tolist(), "T_gripper_wristcam": Xw.tolist(),
        "joint_scale": np.round(s, 4).tolist(),
        "joint_offset_deg": np.round(np.degrees(o), 2).tolist(),
        "rms_px": round(rms, 3), "rms_px_joints_fixed": round(rms_fix, 3),
        "n_poses": len(recs), "n_wrist_markers": n_wr,
        "holdout": {"n_poses": len(held),
                    "wrist_rms_px": round(prob_tr.wrist_rms(sol_tr.x, held), 3),
                    "cube_err_mm_median": round(float(np.median(err_h)), 2) if len(err_h) else None,
                    "cube_err_mm_p90": round(float(np.percentile(err_h, 90)), 2) if len(err_h) else None},
        "cube_err_mm_all": {"median": round(float(np.median(err_all)), 2),
                            "p90": round(float(np.percentile(err_all, 90)), 2)},
        "cubes_base": {f"L{L}_{c}": [round(float(x), 4), round(float(y), 4),
                                      round(float(np.degrees(psi)), 1)]
                       for (L, c, _), (x, y, psi) in zip(prob.objs, cubes)},
        "T_base_cube": {f"L{L}_{c}": _cube_T(R0, x, y, psi).tolist()
                        for (L, c, R0), (x, y, psi) in zip(prob.objs, cubes)},
        "top_cam_in_base": np.round(Xt[:3, 3], 4).tolist(),
        "vs_old_handeye_mm": round(dp, 1), "vs_old_handeye_deg": round(da, 2),
        "date": time.strftime("%Y-%m-%d %H:%M"),
    }
    OUT.write_text(json.dumps(res, indent=2))
    print(f"BA: RMS {rms:.2f} px (суставы заморожены: {rms_fix:.2f} px)")
    print(f"поправки суставов: масштаб {res['joint_scale']}, смещение {res['joint_offset_deg']}°")
    print(f"верхняя камера в базе: {res['top_cam_in_base']} м; от старой hand-eye "
          f"{dp:.0f} мм / {da:.1f}°")
    print(f"камера кисти в кисти: {np.round(Xw[:3, 3], 4).tolist()} м")
    print(f"ошибка положения куба (камера кисти vs решение): все позы медиана "
          f"{res['cube_err_mm_all']['median']} мм, p90 {res['cube_err_mm_all']['p90']} мм; "
          f"hold-out ({len(held)} поз): медиана {res['holdout']['cube_err_mm_median']} мм, "
          f"p90 {res['holdout']['cube_err_mm_p90']} мм, wrist RMS "
          f"{res['holdout']['wrist_rms_px']} px")
    print(f"кубы в базе: {res['cubes_base']}")
    print(f"сохранено: {OUT}")
    make_video(recs, prob, p, res, OUT.with_suffix(".mp4"))
    print(f"видео: {OUT.with_suffix('.mp4')}")


# --- проверка движением: зависнуть над кубом --------------------------------

def plan_hover(ik, T_bc, q_model_now, height):
    """IK зависания над кубом (позиция + ось подхода, азимут губок свободен:
    с ним у предела досягаемости IK не сходится); кубы дальше ~28 см
    вертикальным подходом не достать — ось подхода наклоняется вперёд.
    Конфигурация только «локоть как в туре». -> (q, h, tilt, err) | None."""
    az = float(np.arctan2(T_bc[1, 3], T_bc[0, 3]))
    inits = (q_model_now, np.array([az, -0.6, 0.9, 0.9, 0.0]),
             np.array([az, -0.3, 0.3, 1.2, 0.0]))
    for h in (height, 0.04):
        for tilt in (0, 15, 30, 45):
            t = np.radians(tilt)
            appr = [np.cos(az) * np.sin(t), np.sin(az) * np.sin(t), -np.cos(t)]
            for q0 in inits:
                _DATA.qpos[:] = 0
                _DATA.qpos[:5] = q0
                q, err = ik.solve(_DATA, T_bc[:3, 3] + [0, 0, h], approach=appr,
                                  q_init=q0)
                if err < 0.003 and q[2] > 0.05 and q[1] < 0.35:
                    return q, h, tilt, err
    return None


def measure_hover(d, T_bc, color, Xw, Xt, s, o, h):
    """Замер в текущей позе: куб по камере кисти vs точка захвата по FK.
    -> (плитка кадров с оверлеями, строка результата)."""
    from ik import TCP_OFFSET
    wr_img, wr = stable_view(CAM_WR)
    top_img, top = stable_view(CAM_TOP)
    q_read, _ = d.read()
    F = fk_T(q_read[:5] * s + o)
    tcp = F[:3, 3] + F[:3, :3] @ TCP_OFFSET
    cb = {(0, color): cube_corners(T_bc, color)}
    T_wb = np.linalg.inv(F @ Xw)
    _draw(wr_img, K_WR, D_WR, T_wb, cb, 0, wr)
    _draw(top_img, K_TOP, D_TOP, np.linalg.inv(Xt), cb, 0, top)
    est = cube_poses({m: undistort(px, K_WR, D_WR) for m, px in wr.items()
                      if _ID2FACE[m][0] == color}, K_WR)
    if color in est:
        P = (F @ Xw @ est[color])[:3, 3]
        px_err = np.mean([np.linalg.norm(project(K_WR, T_wb, cb[(0, color)][m])
                                         - undistort(px, K_WR, D_WR), axis=1).mean()
                          for m, px in wr.items() if _ID2FACE[m][0] == color])
        line = (f"{color}: cube seen by wrist camera at "
                f"{np.hypot(*(P[:2] - tcp[:2])) * 1000:.1f} mm lateral from grasp point, "
                f"{(tcp[2] - P[2]) * 1000:.0f} mm below it (target {h * 1000:.0f}); "
                f"marker reprojection {px_err:.1f} px")
    else:
        line = f"{color}: wrist camera sees no marker of this cube"
    cv2.putText(wr_img, f"{color}: predicted (red) vs detected (green)", (10, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (40, 220, 40), 2, cv2.LINE_AA)
    return np.vstack([cv2.resize(top_img, (640, 360)), wr_img]), line


def hover(height=0.06):
    """По решению рука зависает над каждым кубом (точка захвата на height над
    центром, губки поперёк граней); камера кисти с 10 см измеряет, где куб
    на самом деле относительно точки захвата. Кадры -> hover_check.png."""
    sys.path.insert(0, str(Path(__file__).parent))
    from arm_driver import ArmDriver, GuardError
    from ik import ArmIK, TCP_OFFSET
    res = json.loads(OUT.read_text())
    Xw, Xt = np.array(res["T_gripper_wristcam"]), np.array(res["T_base_topcam"])
    s, o = np.array(res["joint_scale"]), np.radians(res["joint_offset_deg"])
    ik = ArmIK(_MODEL)
    d = ArmDriver()
    d.torque(True)
    d.hold()
    tiles, lines = [], []
    latest = max(k.split("_")[0] for k in res["T_base_cube"])   # кубы лежат по последней раскладке
    try:
        for name, T_bc in res["T_base_cube"].items():
            if not name.startswith(latest + "_"):
                continue
            T_bc, color = np.array(T_bc), name.split("_")[1]
            q_now, _ = d.read()
            plan = plan_hover(ik, T_bc, q_now[:5] * s + o, height)
            if plan is None:
                print(f"{name}: IK не встал (r={np.hypot(*T_bc[:2, 3]):.3f} м; "
                      f"вертикальный подход достаёт до ~0.28 м)")
                continue
            q_ik, h, tilt, err = plan
            print(f"{name}: высота {h * 100:.0f} см, наклон подхода {tilt}°, "
                  f"IK {err * 1000:.1f} мм")
            try:
                d.goto(np.concatenate([(q_ik - o) / s, [GRIP]]), 4.0)
            except GuardError as e:
                print(f"{name}: guard: {e}")
                continue
            d.settle()
            tile, line = measure_hover(d, T_bc, color, Xw, Xt, s, o, h)
            print(line)
            lines.append(line)
            tiles.append(tile)
        print("возврат в среднюю позу")
        try:
            d.goto(MID + [GRIP], 4.0)
        except GuardError as e:          # низкий путь: с полом захвата
            print("  transport:", e, "-> режим grasp")
            d.goto(MID + [GRIP], 4.0, mode="grasp")
    finally:
        d.close()
    if tiles:
        band = np.full((40 + 30 * len(lines), 640 * len(tiles), 3), 25, np.uint8)
        cv2.putText(band, "hover check: arm sent by the calibration above each cube",
                    (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (230, 230, 230), 2, cv2.LINE_AA)
        for i, ln in enumerate(lines):
            cv2.putText(band, ln, (12, 58 + 30 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (230, 230, 230), 1, cv2.LINE_AA)
        out = CALIB / "hover_check.png"
        cv2.imwrite(str(out), np.vstack([np.hstack(tiles), band]))
        print(f"кадры: {out}")


# --- первый автономный захват по калибровке ----------------------------------

GRIP_WIDE, GRIP_CLOSED = 1.5, 0.1
# парковка для взгляда верхней камерой: рука сложена сбоку и не заслоняет
# кубы (в средней позе губки закрывали красный куб с точки зрения камеры)
PARK = [0.75, -0.851, 0.414, 1.462, 0.0]


def cube_yaw(T_bc):
    """Азимут нормали боковой грани — вдоль него должна раскрываться губка."""
    n = next(v for v in (T_bc[:3, :3] @ np.asarray(f[3], float) for f in FACES)
             if abs(v[2]) < 0.5)
    return float(np.arctan2(n[1], n[0]))


def grasp(color="red"):
    """Как у сим-эксперта: над кубом 9 см -> зависание 3 см над гранью ->
    медленный спуск к центру -> смыкание поперёк граней -> подъём -> проверка
    верхней камерой, что куб ушёл со стола -> вернуть на место с 1 см ->
    отпустить -> средняя поза. Поза куба — верхней камерой по калибровке."""
    sys.path.insert(0, str(Path(__file__).parent))
    from arm_driver import ArmDriver, GuardError
    from ik import ArmIK
    res = json.loads(OUT.read_text())
    Xt = np.array(res["T_base_topcam"])
    ik = ArmIK(_MODEL)
    d = ArmDriver()
    d.torque(True)
    d.hold()

    def cube_in_base():
        """Куб верхней камерой; глубина PnP по мелкой метке шумит — центр
        снимается лучом камеры на плоскость «куб лежит» (z = полуребро)."""
        _, top = stable_view(CAM_TOP)
        est = cube_poses({m: undistort(px, K_TOP, D_TOP) for m, px in top.items()
                          if _ID2FACE[m][0] == color}, K_TOP)
        if color not in est:
            return None
        T = Xt @ est[color]
        c, ray = Xt[:3, 3], T[:3, 3] - Xt[:3, 3]
        T[:3, 3] = c + ray * (CUBE_HALF - c[2]) / ray[2]
        return T

    def move(target, grip, seconds, q_init, yaw, mode="transport"):
        az = float(np.arctan2(target[1], target[0]))
        for oy, tilt in ((yaw, 0), (yaw, 15), (None, 0), (None, 15)):
            t = np.radians(tilt)      # у предела досягаемости — подход чуть вперёд
            appr = [np.cos(az) * np.sin(t), np.sin(az) * np.sin(t), -np.cos(t)]
            _DATA.qpos[:] = 0
            _DATA.qpos[:5] = q_init
            q, err = ik.solve(_DATA, target, approach=appr, q_init=q_init, opening_yaw=oy)
            if err < 0.003:
                break
        else:
            raise GuardError(f"IK не встал для {np.round(target, 3)} ({err * 1000:.1f} мм)")
        try:
            d.goto(np.concatenate([d.q_cmd(q), [grip]]), seconds, mode=mode)
        except GuardError as e:          # старт ниже пола (просела) — поднять и повторить
            print("  guard:", e, "-> lift и повтор")
            d.lift()
            d.goto(np.concatenate([d.q_cmd(q), [grip]]), seconds, mode=mode)
        d.settle()
        return q

    def look(grip):
        """С парковки (рука не заслоняет кубы) — куб верхней камерой."""
        d.goto(PARK + [grip], 3.5, mode="grasp")
        d.settle()
        return cube_in_base()

    log = []
    try:
        T0 = look(GRIP_WIDE)
        if T0 is None:
            raise SystemExit(f"верхняя камера не видит куб {color}")
        P0, yaw = T0[:3, 3].copy(), cube_yaw(T0)
        ba = {k: v for k, v in res["T_base_cube"].items() if k.endswith(color)}
        P_ba = np.array(list(ba.values())[-1])[:3, 3] if ba else None
        log.append(f"{color}: куб в базе по верхней камере {np.round(P0, 3).tolist()}, "
                   f"r={np.hypot(*P0[:2]):.3f} м, грани под {np.degrees(yaw):.0f}°"
                   + (f"; от решения BA {np.linalg.norm(P0 - P_ba) * 1000:.1f} мм"
                      if P_ba is not None else ""))
        print(log[-1])
        xy = P0[:2]
        above, hov, low = [*xy, 0.09], [*xy, CUBE_HALF + 0.030], [*xy, CUBE_HALF + 0.006]
        q = d.q_model(d.read()[0])
        q = move(above, GRIP_WIDE, 4.0, q, yaw)
        q = move(hov, GRIP_WIDE, 2.0, q, yaw)
        q = move(low, GRIP_WIDE, 3.0, q, yaw, mode="grasp")
        d.goto(np.concatenate([d.q_cmd(q), [GRIP_CLOSED]]), 1.0, mode="grasp")
        d.settle()
        q = move(above, GRIP_CLOSED, 2.5, q, yaw, mode="grasp")
        T1 = look(GRIP_CLOSED)
        still = T1 is not None and np.linalg.norm(T1[:2, 3] - xy) < 0.03 and T1[2, 3] < 0.04
        log.append("подъём: " + ("куб остался на месте — захват не удался" if still
                                 else "куб ушёл со стола (поднят)"))
        print(log[-1])
        if still:
            d.goto(PARK + [GRIP_WIDE], 1.0, mode="grasp")
        else:
            q = d.q_model(d.read()[0])
            q = move(above, GRIP_CLOSED, 3.5, q, yaw)
            q = move([*xy, CUBE_HALF + 0.012], GRIP_CLOSED, 3.0, q, yaw, mode="grasp")  # вернуть с 1 см
            d.goto(np.concatenate([d.q_cmd(q), [GRIP_WIDE]]), 1.0, mode="grasp")
            d.settle()
            q = move(above, GRIP_WIDE, 2.5, q, yaw, mode="grasp")
            T2 = look(GRIP_WIDE)
            if T2 is not None:
                log.append(f"после отпускания куб в {np.round(T2[:3, 3], 3).tolist()}, "
                           f"сдвиг от исходного {np.linalg.norm(T2[:2, 3] - xy) * 1000:.0f} мм")
            else:
                log.append("после отпускания верхняя камера куб не видит")
            print(log[-1])
    finally:
        d.close()
    (CALIB / f"grasp_{color}.txt").write_text("\n".join(log) + "\n")
    return log


# --- самотест на синтетике -------------------------------------------------

def selftest():
    """Синтетические наблюдения по модели с известной правдой (возмущённые
    камеры, поправки суставов, кубы) -> решатель обязан её восстановить."""
    rng = np.random.default_rng(1)
    X_top_true = X_TOP0 @ T_of(np.concatenate([rng.normal(size=3) * 0.03,
                                               rng.normal(size=3) * 0.02]))
    X_w_true = X_W0 @ T_of(np.concatenate([rng.normal(size=3) * 0.05,
                                           rng.normal(size=3) * 0.01]))
    s_true = 1 + rng.normal(size=5) * 0.02
    o_true = np.radians(rng.normal(size=5) * 1.5)
    o_true[[0, 4]] = 0      # смещения pan и roll неотличимы от поворота камер

    def observe(cb, K, dist, T_cb, sz):
        out = {}
        for c in cb:
            for m, pts in cb[c].items():
                pc = (T_cb[:3, :3] @ pts.T).T + T_cb[:3, 3]
                n = -np.cross(pc[1] - pc[0], pc[3] - pc[0])
                if pc[:, 2].min() < 0.05 or np.dot(n, pc.mean(0)) >= 0:
                    continue
                px, _ = cv2.projectPoints(pc, np.zeros(3), np.zeros(3), K, dist)
                px = px.reshape(4, 2)
                if (px > 5).all() and (px[:, 0] < sz[0] - 5).all() and (px[:, 1] < sz[1] - 5).all():
                    out[m] = (px + rng.normal(size=(4, 2)) * 0.3).tolist()
        return out

    recs, truth = [], {}
    for L, xy in ((1, ((0.22, 0.05), (0.27, -0.08))), (2, ((0.18, -0.06), (0.30, 0.09)))):
        cubes = {}
        for color, (x, y) in zip(("red", "green"), xy):
            fi = int(rng.integers(6))
            R0 = Rot.align_vectors([[0, 0, 1.0]], [np.asarray(FACES[fi][3], float)])[0].as_matrix()
            cubes[color] = (R0, x, y, float(rng.uniform(-np.pi, np.pi)))
            truth[(L, color)] = (x, y)
        cb = {c: corners_base(c, *v) for c, v in cubes.items()}
        top_px = observe(cb, K_TOP, D_TOP, np.linalg.inv(X_top_true), SZ_TOP)
        targets = [np.array([x, y, CUBE_HALF]) for (_, x, y, _) in cubes.values()]
        n = 0
        while n < 14:
            q = rng.uniform(POSE_LO, POSE_HI)
            if aim_ok(q, targets) < 1:
                continue
            wr = observe(cb, K_WR, D_WR, np.linalg.inv(fk_T(q) @ X_w_true), SZ_WR)
            if not wr:
                continue
            recs.append({"layout": L, "q": ((q - o_true) / s_true).tolist(),
                         "wrist": wr, "top": top_px})
            n += 1
    prob, sol = fit(prepare(recs))
    Xw, Xt, s, o, cubes = prob.unpack(sol.x)
    e_top, e_w = pose_err(Xt, X_top_true), pose_err(Xw, X_w_true)
    e_cube = max(np.hypot(x - truth[(L, c)][0], y - truth[(L, c)][1]) * 1000
                 for (L, c, _), (x, y, _) in zip(prob.objs, cubes))
    e_s, e_o = np.abs(s - s_true).max(), np.degrees(np.abs(o - o_true).max())
    rms = float(np.sqrt(np.mean(np.concatenate([t[3].ravel() for t in prob.terms(sol.x)]) ** 2)))
    print(f"selftest: {len(recs)} поз, RMS {rms:.2f} px (шум 0.3 px); ошибки: "
          f"верхняя камера {e_top[0]:.2f} мм / {e_top[1]:.3f}°, камера кисти "
          f"{e_w[0]:.2f} мм / {e_w[1]:.3f}°, кубы {e_cube:.2f} мм, масштаб "
          f"{e_s:.4f}, смещение {e_o:.3f}°")
    assert e_top[0] < 2 and e_top[1] < 0.15 and e_w[0] < 2 and e_w[1] < 0.3 \
        and e_cube < 1.0 and e_s < 0.01 and e_o < 0.3, "решатель не восстановил правду"
    print("selftest OK")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["tour", "solve", "hover", "grasp", "selftest"])
    ap.add_argument("--cube", default="red")
    ap.add_argument("--layout", type=int, default=1)
    ap.add_argument("--layouts", type=int, nargs="*", help="solve: только эти раскладки")
    ap.add_argument("--poses", type=int, default=24)
    a = ap.parse_args()
    {"tour": lambda: tour(a.layout, a.poses), "solve": lambda: solve(a.layouts),
     "hover": hover, "grasp": lambda: grasp(a.cube), "selftest": selftest}[a.cmd]()


if __name__ == "__main__":
    main()
