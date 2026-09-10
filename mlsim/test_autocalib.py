#!/usr/bin/env python3
"""Автокалибровка положения робота по двум камерам и двум кубам (сим).

Схема Владимира: кубы ЛЕЖАТ на столе, их видят и верхняя камера, и камера
гриппера. Рука объезжает позы «кистью вниз», и из наблюдений автоматически
решаются две неизвестные матрицы: поза камеры гриппера относительно кисти
(eye-in-hand) и поза робота относительно верхней камеры. Куб в губки класть
не нужно — мишень неподвижна, ёрзать нечему.

Геометрия — как в реальном эксперименте: верхняя камера ставится в матрицу,
полученную hand-eye'ем на живой руке (real/calib/handeye_refined.json),
интринсики fx=830 при 1920x1080 как у реальной C920-образной камеры.
Режим с «люфтами» добавляет шум в отсчёты суставов, чтобы показать
устойчивость. Выход: reports/autocalib_sim.json + autocalib_sim.mp4.
"""

import json
import sys
import time
from pathlib import Path

import cv2
import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as Rot

import cube_cv
from cube_cv import (CUBE_HALF, build_scene, estimate_cubes_cam, _ID2FACE,
                     _marker_T_cube, _OBJ_PTS, FACES)
from expert import CUBES, GRIPPER_OPEN
from test_cube_cv import set_cube, random_pose, sample_two, REPORTS

ROOT = Path(__file__).parents[1]
TOP_W, TOP_H, TOP_F = 1920, 1080, 830.0
WR_W, WR_H = 640, 480
K_TOP = np.array([[TOP_F, 0, (TOP_W - 1) / 2], [0, TOP_F, (TOP_H - 1) / 2], [0, 0, 1]])
K_WR = np.array([[400.0, 0, (WR_W - 1) / 2], [0, 533.3, (WR_H - 1) / 2], [0, 0, 1]])
N_POSES = 30
POSE_LO = np.array([-0.8, -0.9, 0.4, 0.4, -1.1])
POSE_HI = np.array([0.8, -0.1, 1.4, 1.5, 1.1])


def detect_px(img_rgb):
    """{id: 4x2 пикселей углов} по кадру."""
    det = cv2.aruco.ArucoDetector(cube_cv.DICT, cv2.aruco.DetectorParameters())
    c, ids, _ = det.detectMarkers(cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY))
    if ids is None:
        return {}
    return {int(m): q.reshape(4, 2) for q, m in zip(c, ids.ravel()) if int(m) in _ID2FACE}


def real_camera_spec():
    """Верхняя камера ровно там, где реальная (hand-eye на живой руке)."""
    X = np.array(json.loads((ROOT / "real/calib/handeye_refined.json")
                            .read_text())["T_cam2base"])
    R_mj = X[:3, :3] @ np.diag([1, -1, -1])   # OpenCV -> MuJoCo оси камеры
    fovy = np.degrees(2 * np.arctan(TOP_H / 2 / TOP_F))
    return {"name": "cv_top",
            "pos": " ".join(f"{v:.4f}" for v in X[:3, 3]),
            "xyaxes": " ".join(f"{v:.5f}" for v in np.concatenate(
                [R_mj[:, 0], R_mj[:, 1]])),
            "fovy": f"{fovy:.2f}"}, X


def cam_T(model, data, name):
    cid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, name)
    T = np.eye(4)
    T[:3, :3] = data.cam_xmat[cid].reshape(3, 3) @ np.diag([1, -1, -1])
    T[:3, 3] = data.cam_xpos[cid]
    return T


def site_T(model, data, name):
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
    T = np.eye(4)
    T[:3, :3] = data.site_xmat[sid].reshape(3, 3)
    T[:3, 3] = data.site_xpos[sid]
    return T


def pose_err(T_est, T_true):
    dpos = float(np.linalg.norm(T_est[:3, 3] - T_true[:3, 3]) * 1000)
    dR = T_true[:3, :3].T @ T_est[:3, :3]
    return dpos, float(np.degrees(np.linalg.norm(Rot.from_matrix(dR).as_rotvec())))


def park_martin(As, Bs):
    M = np.zeros((3, 3))
    for A, B in zip(As, Bs):
        a = Rot.from_matrix(A[:3, :3]).as_rotvec()
        b = Rot.from_matrix(B[:3, :3]).as_rotvec()
        M += np.outer(b, a)
    U, _, Vt = np.linalg.svd(M.T)           # проекция на SO(3), det = +1
    Rx = U @ np.diag([1, 1, np.linalg.det(U @ Vt)]) @ Vt
    lhs = [A[:3, :3] - np.eye(3) for A in As]
    rhs = [Rx @ B[:3, 3] - A[:3, 3] for A, B in zip(As, Bs)]
    t = np.linalg.lstsq(np.vstack(lhs), np.concatenate(rhs), rcond=None)[0]
    X = np.eye(4)
    X[:3, :3] = Rx
    X[:3, 3] = t
    return X


class Sim:
    def __init__(self):
        spec, self.X_top_real = real_camera_spec()
        xml = build_scene(cv_cam=spec, out_name="pick_place_autocalib.xml")
        self.model = mujoco.MjModel.from_xml_path(str(xml))
        self.data = mujoco.MjData(self.model)
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        self.r_top = mujoco.Renderer(self.model, TOP_H, TOP_W)
        self.r_wr = mujoco.Renderer(self.model, WR_H, WR_W)
        self.r_show = mujoco.Renderer(self.model, 360, 640)
        self.sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE,
                                     "gripperframe")

    def set_q(self, q5, grip=GRIPPER_OPEN):
        self.data.qpos[:5] = q5
        self.data.qpos[5] = grip
        mujoco.mj_forward(self.model, self.data)

    def fk(self, q5):
        """FK для решателя — как в драйвере: только по углам."""
        q_save = self.data.qpos[:6].copy()
        self.set_q(q5)
        T = site_T(self.model, self.data, "gripperframe")
        self.data.qpos[:6] = q_save
        mujoco.mj_forward(self.model, self.data)
        return T

    def render(self, cam):
        r = {"cv_top": self.r_top, "wrist_cam": self.r_wr, "show": self.r_show}[cam]
        r.update_scene(self.data, camera=cam)
        return r.render()

    def truths(self):
        X_top = cam_T(self.model, self.data, "cv_top")          # база<-top
        X_w = np.linalg.inv(site_T(self.model, self.data, "gripperframe")) \
            @ cam_T(self.model, self.data, "wrist_cam")          # кисть<-wrist
        return X_top, X_w


class Video:
    def __init__(self, path):
        self.w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"),
                                 15, (1280, 900))

    def frame(self, top_rgb, wrist_rgb, show_rgb, lines, top_est=None):
        top = cv2.cvtColor(top_rgb, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(top, cv2.COLOR_BGR2GRAY)
        det = cv2.aruco.ArucoDetector(cube_cv.DICT, cv2.aruco.DetectorParameters())
        c, ids, _ = det.detectMarkers(gray)
        if ids is not None:
            cv2.aruco.drawDetectedMarkers(top, c, ids)
        top = cv2.resize(top, (1280, 720))
        cv2.putText(top, "top camera (as real C920, fx=830)", (14, 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (40, 220, 40), 2, cv2.LINE_AA)
        wr = cv2.cvtColor(wrist_rgb, cv2.COLOR_RGB2BGR)
        c2, ids2, _ = det.detectMarkers(cv2.cvtColor(wr, cv2.COLOR_BGR2GRAY))
        if ids2 is not None:
            cv2.aruco.drawDetectedMarkers(wr, c2, ids2)
        wr = cv2.resize(wr, (240, 180))
        cv2.putText(wr, "wrist", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (40, 220, 40), 2, cv2.LINE_AA)
        sh = cv2.resize(cv2.cvtColor(show_rgb, cv2.COLOR_RGB2BGR), (320, 180))
        panel = np.full((180, 720, 3), 25, np.uint8)
        for i, ln in enumerate(lines[:5]):
            cv2.putText(panel, ln, (12, 32 + i * 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.68, (230, 230, 230), 2, cv2.LINE_AA)
        bottom = np.hstack([sh, wr, panel])
        self.w.write(np.vstack([top, bottom]))

    def close(self):
        self.w.release()


def run(sim, noise_deg, video=None, seed=5, layouts=2):
    """Тур по нескольким раскладкам кубов; решение — bundle adjustment по
    пикселям углов меток обеих камер с ограничением «куб лежит на столе»
    (поза куба = x, y, yaw; грань вверх известна по ID верхней метки)."""
    rng = np.random.default_rng(seed)
    sim.set_q(np.array([0.0, -0.35, 0.6, 0.4, 0.0]))
    X_top_true, X_w_true = sim.truths()
    obs, top_data, tried_total = [], [], 0
    for L in range(layouts):
        xy = dict(zip(("red", "green"), sample_two(rng)))
        for color in CUBES:
            set_cube(sim.model, sim.data, color, (*xy[color], CUBE_HALF),
                     random_pose(rng))
        mujoco.mj_forward(sim.model, sim.data)
        top_img = sim.render("cv_top")
        top_px = detect_px(top_img)
        top_obs = estimate_cubes_cam(top_img, K_TOP)
        assert len(top_obs) == 2, "верхняя камера должна видеть оба куба"
        top_data.append((top_obs, top_px, top_img))
        n_layout, tried = 0, 0
        q_prev = sim.data.qpos[:5].copy()
        while n_layout < N_POSES // layouts and tried < 600:
            tried += 1
            q5 = rng.uniform(POSE_LO, POSE_HI)
            p = sim.fk(q5)[:3, 3]
            if not (0.10 < p[2] < 0.30 and 0.15 < np.hypot(p[0], p[1]) < 0.34):
                continue
            sim.set_q(q5)
            if cam_T(sim.model, sim.data, "wrist_cam")[:3, 2][2] > -0.25:
                continue
            wr_img = sim.render("wrist_cam")
            wr_px = detect_px(wr_img)
            if not wr_px:
                continue
            wr_pos = {c: e["T_cam"][:3, 3] for c, e in
                      estimate_cubes_cam(wr_img, K_WR).items()}
            if video is not None:
                for a in np.linspace(0, 1, 10):
                    sim.set_q(q_prev + a * (q5 - q_prev))
                    video.frame(sim.render("cv_top"), sim.render("wrist_cam"),
                                sim.render("show"),
                                [f"layout {L + 1}/{layouts}, joint backlash {noise_deg:.1f} deg",
                                 f"poses: {len(obs)}/{N_POSES}",
                                 "arm looks at the cubes on the table with its wrist camera"])
                sim.set_q(q5)
            q_meas = q5 + np.radians(noise_deg) * rng.standard_normal(5)
            obs.append((L, q_meas, wr_px, wr_pos))
            n_layout += 1
            q_prev = q5
        tried_total += tried
    print(f"поз: {len(obs)} (перебрано {tried_total}), раскладок {layouts}")
    Fs = [sim.fk(q) for _, q, _, _ in obs]

    # объекты BA: (раскладка, куб) -> x, y, yaw; грань вверх по ID top-метки
    objs = []
    for L, (top_obs, top_px, _) in enumerate(top_data):
        for color in CUBES:
            ups = [mid for mid in top_px if _ID2FACE[mid][0] == color]
            if not ups:
                continue
            fi = _ID2FACE[ups[0]][1]
            n = np.asarray(FACES[fi][3], float)
            R0 = Rot.align_vectors([[0, 0, 1.0]], [n])[0].as_matrix()
            objs.append((L, color, R0))
    obj_h = np.hstack([_OBJ_PTS, np.ones((4, 1))])

    def corners_base(color, R0, x, y, psi):
        T = np.eye(4)
        T[:3, :3] = Rot.from_rotvec([0, 0, psi]).as_matrix() @ R0
        T[:3, 3] = [x, y, CUBE_HALF]
        return {mid: (T @ _marker_T_cube(fi) @ obj_h.T).T[:, :3]
                for mid, (c, fi) in _ID2FACE.items() if c == color}

    def project(K, T_cam_base, pts):
        pc = (T_cam_base[:3, :3] @ pts.T).T + T_cam_base[:3, 3]
        uv = (K @ pc.T).T
        return uv[:, :2] / uv[:, 2:3]

    def T_of(v):
        M = np.eye(4)
        M[:3, :3] = Rot.from_rotvec(v[:3]).as_matrix()
        M[:3, 3] = v[3:6]
        return M

    def resid(p):
        Xw, Xt = T_of(p[0:6]), T_of(p[6:12])
        T_top_cam = np.linalg.inv(Xt)
        out = []
        for i, (L, color, R0) in enumerate(objs):
            x, y, psi = p[12 + 3 * i:15 + 3 * i]
            cb = corners_base(color, R0, x, y, psi)
            for mid, px in top_data[L][1].items():
                if mid in cb:
                    out.append((project(K_TOP, T_top_cam, cb[mid]) - px).ravel())
            for F, (Lo, _, wpx, _) in zip(Fs, obs):
                if Lo != L:
                    continue
                T_w_cam = np.linalg.inv(F @ Xw)
                for mid, px in wpx.items():
                    if mid in cb:
                        out.append((project(K_WR, T_w_cam, cb[mid]) - px).ravel())
        return np.concatenate(out)

    # --- инициализация без случайностей -----------------------------------
    # (i) X_w из позиций кубов в wrist: F_i X_w c_i = P_obj (одна точка на
    #     объект), нелинейный LSQ с мультистартом только по 6 dof X_w
    obj_index = {(L, color): i for i, (L, color, _) in enumerate(objs)}

    def resid_w(p):
        Xw = T_of(p[:6])
        out = []
        for F, (L, _, _, wpos) in zip(Fs, obs):
            for color, c in wpos.items():
                if (L, color) in obj_index:
                    i = obj_index[(L, color)]
                    P = p[6 + 3 * i:9 + 3 * i]
                    out.append(((F @ Xw @ np.append(c, 1))[:3] - P) * 10)
        return np.concatenate(out)

    rng_ms = np.random.default_rng(0)
    best_w = None
    for _ in range(10):
        p0 = np.concatenate([rng_ms.normal(size=3) * 1.5, [0, 0.05, -0.05]]
                            + [[0.25, 0.0, CUBE_HALF]] * len(objs))
        try:
            sol = least_squares(resid_w, p0, method="lm", max_nfev=2000)
        except Exception:
            continue
        if best_w is None or sol.cost < best_w.cost:
            best_w = sol
    X_w0 = T_of(best_w.x[:6])
    P_base = {objs[i][:2]: best_w.x[6 + 3 * i:9 + 3 * i] for i in range(len(objs))}
    # (ii) X_top по Прокрусту: позиции объектов в top (PnP) <-> в базе
    A = np.array([top_data[L][0][color]["T_cam"][:3, 3] for L, color, _ in objs])
    Bp = np.array([P_base[(L, color)] for L, color, _ in objs])
    ca, cb = A.mean(0), Bp.mean(0)
    U, _, Vt = np.linalg.svd((A - ca).T @ (Bp - cb))
    Rk = (U @ np.diag([1, 1, np.linalg.det(U @ Vt)]) @ Vt).T
    X_top0 = np.eye(4)
    X_top0[:3, :3] = Rk
    X_top0[:3, 3] = cb - Rk @ ca
    # (iii) yaw кубов из top PnP, затем BA с этой точки (+ малые рестарты)
    p_init = [Rot.from_matrix(X_w0[:3, :3]).as_rotvec(), X_w0[:3, 3],
              Rot.from_matrix(X_top0[:3, :3]).as_rotvec(), X_top0[:3, 3]]
    for L, color, R0 in objs:
        Rc = (X_top0 @ top_data[L][0][color]["T_cam"])[:3, :3] @ R0.T
        psi = float(np.arctan2(Rc[1, 0], Rc[0, 0]))
        P = P_base[(L, color)]
        p_init.append([P[0], P[1], psi])
    p_init = np.concatenate([np.asarray(v, float).ravel() for v in p_init])
    best = None
    for k in range(4):
        p0 = p_init.copy()
        if k:
            p0[:12] += rng_ms.normal(size=12) * 0.02
            p0[14::3] += rng_ms.normal(size=len(objs)) * 0.1
        try:
            sol = least_squares(resid, p0, method="lm", max_nfev=3000)
        except Exception:
            continue
        if best is None or sol.cost < best.cost:
            best = sol
    X_w_r, X_top_r = T_of(best.x[0:6]), T_of(best.x[6:12])
    rms_px = float(np.sqrt(np.mean(best.fun ** 2)))
    print(f"  BA: RMS репроекции {rms_px:.2f} px")
    top_img = top_data[-1][2]
    As = []
    res = {}
    for name, est, true in (("robot_vs_top", X_top_r, X_top_true),
                            ("wrist_vs_gripper", X_w_r, X_w_true)):
        dp, da = pose_err(est, true)
        res[name] = {"pos_mm": round(dp, 2), "ang_deg": round(da, 3)}
    base_in_cam = np.linalg.inv(X_top_r)
    res["robot_in_top_camera_est"] = np.round(base_in_cam[:3, 3], 4).tolist()
    res["robot_in_top_camera_true"] = np.round(np.linalg.inv(X_top_true)[:3, 3], 4).tolist()
    res["poses"] = len(obs)
    res["relative_motions"] = len(As)
    print(f"люфты {noise_deg}°: робот↔верхняя камера "
          f"{res['robot_vs_top']['pos_mm']} мм / {res['robot_vs_top']['ang_deg']}°; "
          f"wrist↔кисть {res['wrist_vs_gripper']['pos_mm']} мм / "
          f"{res['wrist_vs_gripper']['ang_deg']}°")
    if video is not None:
        for _ in range(45):
            video.frame(top_img, sim.render("wrist_cam"), sim.render("show"), [
                "AUTO-CALIBRATION RESULT:",
                f"robot vs top camera: {res['robot_vs_top']['pos_mm']:.1f} mm, "
                f"{res['robot_vs_top']['ang_deg']:.2f} deg",
                f"wrist camera vs gripper: {res['wrist_vs_gripper']['pos_mm']:.1f} mm, "
                f"{res['wrist_vs_gripper']['ang_deg']:.2f} deg",
                f"poses: {len(obs)}, joint backlash {noise_deg:.1f} deg"])
    return res


def main():
    quick = "--quick" in sys.argv          # только идеальный режим с видео
    REPORTS.mkdir(exist_ok=True)
    sim = Sim()
    video = Video(REPORTS / "autocalib_sim.mp4")
    report = {"camera_pose_from_real_handeye": sim.X_top_real[:3, 3].tolist()}
    report["ideal"] = run(sim, 0.0, video=video)
    if not quick:
        report["backlash_0.5deg"] = run(sim, 0.5, seed=6)
        report["backlash_1.0deg"] = run(sim, 1.0, seed=7)
    video.close()
    (REPORTS / "autocalib_sim.json").write_text(json.dumps(report, indent=2))
    print(f"отчёт: {REPORTS / 'autocalib_sim.json'}")


if __name__ == "__main__":
    main()
