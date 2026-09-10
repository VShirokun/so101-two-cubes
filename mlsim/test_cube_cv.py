#!/usr/bin/env python3
"""Тест алгоритма авто-сбора реального датасета — в симуляции, end-to-end.

Проверяется вся цепочка будущего реального стенда: кубы с ArUco на гранях ->
верхняя CV-камера -> детекция и 6-DoF поза куба ИЗ ПИКСЕЛЕЙ (не из ground
truth) -> IK-захват по CV-позе -> подъём -> CV-верификация успеха -> перенос
в случайную точку -> отпускание с 1 см -> куб ложится новой раскладкой ->
следующий цикл. Сим даёт то, чего не даст реальный стенд: точный ground truth
для каждой ступени.

Ступени (--stage):
  faces    соответствие кубических текстур MuJoCo граням и манифесту меток:
           куб в 6 известных ориентациях, детекция верхней грани, сверка ID и
           позы; при расхождении печатает правильный TEX_LAYOUT.
  pose     статистика точности CV-позы на случайных раскладках (позиция/угол
           против ground truth, полнота детекции).
  collect  полный цикл сборщика на CV-позах: grasp-rate, точность
           CV-верификации против ground truth, видео.
  all      всё подряд (по умолчанию).

Запуск: /opt/anaconda3/envs/lerobot/bin/python test_cube_cv.py [--stage ...]
"""

import argparse
import json
import time
from pathlib import Path

import cv2
import mujoco
import numpy as np

import cube_cv
from cube_cv import (CUBE_HALF, CUBE_IDS, CV_CAM, FACES, build_scene,
                     camera_matrix, camera_T_world, estimate_cubes, pose_errors)
from expert import (CONTROL_HZ, CUBES, GRIPPER_APPROACH, GRIPPER_CLOSED,
                    GRIPPER_OPEN, TRANSPORT_Z, ZONE_X, ZONE_Y, MIN_CUBE_GAP,
                    PickPlaceExpert)

REPORTS = Path(__file__).parent / "reports"
PARK = {"red": (0.14, -0.20), "green": (0.14, 0.20)}  # углы зоны для stage faces

# Парковка руки для обзора: в home рука висит над зоной и закрывает кубы от
# CV-камеры (проверено кадром). Перед каждой детекцией — увести базу в сторону.
# Тот же приём понадобится на реальном стенде.
Q_PARK = np.array([-1.5, -1.2, 1.2, 0.6, 0.0, 0.9])
# Зеркальная обзорная поза: из одной парковки рука закрывает полосу стола со
# своей стороны — если куб не виден из первой, посмотреть из второй.
Q_PARK_B = np.array([1.5, -1.2, 1.2, 0.6, 0.0, 0.9])

# Базовые ориентации «грань F смотрит вверх»: (ось, угол)
_FACE_UP = {
    "+Z": (None, 0.0), "-Z": ((1, 0, 0), np.pi),
    "+X": ((0, 1, 0), -np.pi / 2), "-X": ((0, 1, 0), np.pi / 2),
    "+Y": ((1, 0, 0), np.pi / 2), "-Y": ((1, 0, 0), -np.pi / 2),
}


def quat_mul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def axis_quat(axis, angle):
    if axis is None:
        return np.array([1.0, 0, 0, 0])
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    return np.concatenate([[np.cos(angle / 2)], axis * np.sin(angle / 2)])


def set_cube(model, data, color, pos, quat):
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, CUBES[color]["joint"])
    adr = model.jnt_qposadr[jid]
    data.qpos[adr:adr + 3] = pos
    data.qpos[adr + 3:adr + 7] = quat
    data.qvel[model.jnt_dofadr[jid]:model.jnt_dofadr[jid] + 6] = 0.0


def cube_gt(model, data, color):
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, CUBES[color]["joint"])
    adr = model.jnt_qposadr[jid]
    return data.qpos[adr:adr + 3].copy(), data.qpos[adr + 3:adr + 7].copy()


class Rig:
    def __init__(self, tex_layout=None):
        xml = build_scene(tex_layout)
        self.model = mujoco.MjModel.from_xml_path(str(xml))
        self.data = mujoco.MjData(self.model)
        self.cv_renderer = mujoco.Renderer(self.model, cube_cv.CV_H, cube_cv.CV_W)
        self.show_renderer = mujoco.Renderer(self.model, 480, 640)
        self.K = camera_matrix()
        self.reset()

    def reset(self):
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        mujoco.mj_forward(self.model, self.data)
        self.T_wc = camera_T_world(self.model, self.data)
        self.q_home = self.data.ctrl[:6].copy()

    def teleport_park(self):
        """Мгновенно поставить руку в обзорную позу (для статичных стейджей)."""
        self.data.qpos[:6] = Q_PARK
        self.data.qvel[:6] = 0.0
        self.data.ctrl[:6] = Q_PARK
        mujoco.mj_forward(self.model, self.data)

    def cv_frame(self):
        self.cv_renderer.update_scene(self.data, camera=CV_CAM["name"])
        return self.cv_renderer.render()

    def show_frame(self):
        self.show_renderer.update_scene(self.data, camera="show")
        return self.show_renderer.render()

    def estimate(self, frame=None, lying=False):
        frame = self.cv_frame() if frame is None else frame
        return estimate_cubes(frame, self.T_wc, self.K, lying=lying)


# --- stage faces ----------------------------------------------------------

def stage_faces() -> bool:
    """Куб каждой гранью вверх: детекция верхней метки, сверка ID и полной
    позы куба. Если раскладка текстур неверна — вычислить и напечатать верную."""
    rig = Rig()
    rig.teleport_park()
    set_cube(rig.model, rig.data, "green", (*PARK["green"], CUBE_HALF),
             (1, 0, 0, 0))
    ok, fixes = True, {}
    for face, axis_angle in _FACE_UP.items():
        q = axis_quat(*axis_angle)
        set_cube(rig.model, rig.data, "red", (0.25, 0.0, CUBE_HALF), q)
        mujoco.mj_forward(rig.model, rig.data)
        raw = rig.estimate().get("red")
        est = rig.estimate(lying=True)
        got = est.get("red")
        pos_gt, quat_gt = cube_gt(rig.model, rig.data, "red")
        face_id = dict(zip([f[0] for f in FACES], CUBE_IDS["red"]))[face]
        if got is None:
            print(f"[faces] {face} вверх: красный куб не детектирован ВООБЩЕ")
            ok = False
            continue
        dpos, dang = pose_errors(raw["T"], pos_gt, quat_gt)
        dpos_s, dang_s = pose_errors(got["T"], pos_gt, quat_gt)
        status = "OK" if dpos_s < 2 and dang_s < 2 else "FAIL"
        print(f"[faces] {face} вверх (id {face_id}): меток {got['n_markers']}, "
              f"сырая поза {dpos:.1f} мм/{dang:.1f}°, "
              f"лёжа-оценка {dpos_s:.2f} мм/{dang_s:.2f}° -> {status}")
        if status == "FAIL":
            ok = False
            fixes[face] = (dpos, dang)
    if not ok and fixes:
        _diagnose_layout()
    return ok


def _diagnose_layout():
    """Диагностическая раскладка: слоту i — метка CUBE_IDS['red'][i] без
    поворота; по детекции верхней грани восстанавливается, какой слот лёг на
    какую грань box-геома и с каким поворотом."""
    slots = list(cube_cv.TEX_LAYOUT)
    diag = {slot: (FACES[i][0], 0) for i, slot in enumerate(slots)}
    rig = Rig(tex_layout=diag)
    rig.teleport_park()
    set_cube(rig.model, rig.data, "green", (*PARK["green"], CUBE_HALF), (1, 0, 0, 0))
    slot_of_id = {CUBE_IDS["red"][i]: slot for i, slot in enumerate(slots)}
    print("[faces] диагностика: слот -> фактическая грань (поворот)")
    layout = {}
    for face, axis_angle in _FACE_UP.items():
        q = axis_quat(*axis_angle)
        set_cube(rig.model, rig.data, "red", (0.25, 0.0, CUBE_HALF), q)
        mujoco.mj_forward(rig.model, rig.data)
        frame = rig.cv_frame()
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        det = cv2.aruco.ArucoDetector(cube_cv.DICT, cv2.aruco.DetectorParameters())
        corners, ids, _ = det.detectMarkers(gray)
        top = None
        for quad, mid in zip(corners, ids.ravel() if ids is not None else []):
            okp, rvec, _ = cv2.solvePnP(cube_cv._OBJ_PTS, quad, rig.K, None,
                                        flags=cv2.SOLVEPNP_ITERATIVE)
            if not okp:
                continue
            n_world = rig.T_wc[:3, :3] @ cv2.Rodrigues(rvec)[0][:, 2]
            if n_world[2] > 0.9:
                top = (int(mid), cv2.Rodrigues(rvec)[0])
        if top is None:
            print(f"  {face}: верхняя метка не найдена")
            continue
        mid, R_cm = top
        # up метки в мире против ожидаемого up грани face при данной ориентации куба
        R_wm = rig.T_wc[:3, :3] @ R_cm
        R_cube = np.zeros(9)
        mujoco.mju_quat2Mat(R_cube, q)
        R_cube = R_cube.reshape(3, 3)
        exp_up = R_cube @ np.array(dict((f[0], f[2]) for f in FACES)[face])
        exp_right = R_cube @ np.array(dict((f[0], f[1]) for f in FACES)[face])
        cosu = float(np.dot(R_wm[:, 1], exp_up))
        cosr = float(np.dot(R_wm[:, 1], exp_right))
        turns = int(np.round(np.degrees(np.arctan2(cosr, cosu)) / 90)) % 4
        print(f"  грань {face}: слот {slot_of_id[mid]}, доворот {turns} четв.")
        layout[slot_of_id[mid]] = (face, turns)
    print(f"[faces] предложение TEX_LAYOUT = {layout}")


# --- stage pose -----------------------------------------------------------

def random_pose(rng):
    face = list(_FACE_UP)[rng.integers(6)]
    q = quat_mul(axis_quat((0, 0, 1), rng.uniform(0, 2 * np.pi)),
                 axis_quat(*_FACE_UP[face]))
    return q


def sample_two(rng):
    while True:
        a = np.array([rng.uniform(*ZONE_X), rng.uniform(*ZONE_Y)])
        b = np.array([rng.uniform(*ZONE_X), rng.uniform(*ZONE_Y)])
        if np.linalg.norm(a - b) >= MIN_CUBE_GAP:
            return a, b


def stage_pose(n=60, seed=7) -> dict:
    rig = Rig()
    rig.teleport_park()
    rng = np.random.default_rng(seed)
    errs = {"red": [], "green": []}
    missing = 0
    t0 = time.time()
    for _ in range(n):
        xy = dict(zip(("red", "green"), sample_two(rng)))
        for color in CUBES:
            set_cube(rig.model, rig.data, color, (*xy[color], CUBE_HALF),
                     random_pose(rng))
        mujoco.mj_forward(rig.model, rig.data)
        frame = rig.cv_frame()
        est_raw = rig.estimate(frame)
        est = rig.estimate(frame, lying=True)
        for color in CUBES:
            if color not in est:
                missing += 1
                continue
            pos_gt, quat_gt = cube_gt(rig.model, rig.data, color)
            raw = pose_errors(est_raw[color]["T"], pos_gt, quat_gt)
            snap = pose_errors(est[color]["T"], pos_gt, quat_gt)
            errs[color].append(raw + snap + (est[color]["n_markers"],))
    all_err = np.array(errs["red"] + errs["green"])
    rep = {
        "layouts": n, "cubes_expected": 2 * n,
        "missing": missing,
        "raw_pos_mm_median": float(np.median(all_err[:, 0])),
        "raw_ang_deg_median": float(np.median(all_err[:, 1])),
        "snap_pos_mm_median": float(np.median(all_err[:, 2])),
        "snap_pos_mm_p95": float(np.percentile(all_err[:, 2], 95)),
        "snap_ang_deg_median": float(np.median(all_err[:, 3])),
        "snap_ang_deg_p95": float(np.percentile(all_err[:, 3], 95)),
        "markers_mean": float(np.mean(all_err[:, 4])),
        "seconds": round(time.time() - t0, 1),
    }
    print(f"[pose] {rep}")
    assert missing == 0, f"CV не нашёл куб в {missing} случаях из {2 * n}"
    assert rep["snap_pos_mm_median"] < 2 and rep["snap_ang_deg_median"] < 2, rep
    return rep


# --- stage collect --------------------------------------------------------

class CVExpert(PickPlaceExpert):
    """План строится не по ground truth сима, а по позе от CV."""

    def plan_pick(self, cube_xyz, yaw):
        above = np.array([cube_xyz[0], cube_xyz[1], TRANSPORT_Z])
        hover = np.array([cube_xyz[0], cube_xyz[1], cube_xyz[2] + 0.030])
        grasp = np.array([cube_xyz[0], cube_xyz[1], cube_xyz[2] + 0.002])
        lift = np.array([cube_xyz[0], cube_xyz[1], TRANSPORT_Z])
        return [
            (above, GRIPPER_APPROACH, 0.8, yaw),
            (hover, GRIPPER_APPROACH, 0.5, yaw),
            (hover, GRIPPER_APPROACH, 0.2, yaw),
            (grasp, GRIPPER_APPROACH, 0.6, yaw),
            (grasp, GRIPPER_APPROACH, 0.25, yaw),
            (grasp, GRIPPER_CLOSED, 0.5, yaw),
            (lift, GRIPPER_CLOSED, 0.7, yaw),
        ]

    def plan_place(self, target_xy, yaw_from, yaw_to):
        """Перенос и отпускание: низ куба в 1 см над столом (центр куба на
        уровне tcp - 2 мм, значит tcp на 0.014 + 0.010 + 0.002).

        Ехать — с yaw захвата, доворачивать к yaw_to отдельной фазой НА
        МЕСТЕ: вращение запястья на ходу выбивало куб из губок (5 потерь на
        29 эпизодов), поворот на месте — нет."""
        drop_z = CUBE_HALF + 0.010 + 0.002
        over = np.array([target_xy[0], target_xy[1], TRANSPORT_Z])
        low = np.array([target_xy[0], target_xy[1], drop_z])
        # длительность переноса от дистанции: лимит скорости конца ~20 см/с,
        # быстрый фиксированный перенос ронял куб на дальних целях
        tcp = self.ik.tcp(self.data)[0][:2]
        carry_s = float(np.clip(np.linalg.norm(over[:2] - tcp) / 0.20, 1.0, 2.5))
        return [
            (over, GRIPPER_CLOSED, carry_s, yaw_from),
            (over, GRIPPER_CLOSED, 0.8, yaw_to),
            (low, GRIPPER_CLOSED, 0.6, yaw_to),
            (low, GRIPPER_OPEN, 0.35, yaw_to),
            (over, GRIPPER_OPEN, 0.5, yaw_to),
        ]


def cv_yaw(T):
    """Азимут самой горизонтальной оси куба (как cube_yaw, но из CV-позы)."""
    R = T[:3, :3]
    axis = R[:, int(np.argmin(np.abs(R[2, :])))]
    return float(np.arctan2(axis[1], axis[0]))


class Recorder:
    def __init__(self, path, fps=15):
        self.writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"),
                                      fps, (640, 480))

    def add(self, frame_rgb):
        self.writer.write(cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))

    def add_cv_checkpoint(self, cv_frame_rgb, est, rig, label, hold=8):
        img = cv_frame_rgb.copy()
        for color, e in est.items():
            T_cam = np.linalg.inv(rig.T_wc) @ e["T"]
            rvec = cv2.Rodrigues(T_cam[:3, :3])[0]
            cv2.drawFrameAxes(img, rig.K, None, rvec, T_cam[:3, 3], 0.03)
            px, _ = cv2.projectPoints(T_cam[:3, 3].reshape(1, 3), np.zeros(3),
                                      np.zeros(3), rig.K, None)
            x, y = px.ravel().astype(int)
            cv2.putText(img, color, (x + 15, y - 15), cv2.FONT_HERSHEY_SIMPLEX,
                        1.6, (255, 40, 40), 3, cv2.LINE_AA)
        cv2.putText(img, label, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 2.0,
                    (20, 20, 20), 4, cv2.LINE_AA)
        small = cv2.resize(img, (640, 480))
        for _ in range(hold):
            self.add(small)

    def close(self):
        self.writer.release()


def run_actions(rig, acts, rec=None):
    sub = int(round(1.0 / (CONTROL_HZ * rig.model.opt.timestep)))
    for i, a in enumerate(acts):
        rig.data.ctrl[:6] = a
        for _ in range(sub):
            mujoco.mj_step(rig.model, rig.data)
        if rec is not None and i % 2 == 0:
            rec.add(rig.show_frame())


def settle(rig, seconds, rec=None):
    acts = [rig.data.ctrl[:6].copy()] * max(2, int(seconds * CONTROL_HZ))
    run_actions(rig, acts, rec)


def joint_move(rig, q_target, seconds, rec=None):
    """Плавный переезд в суставную позу (парковка для обзора и т.п.)."""
    q0 = rig.data.ctrl[:6].copy()
    steps = max(2, int(seconds * CONTROL_HZ))
    acts = []
    for i in range(1, steps + 1):
        t_ = 0.5 - 0.5 * np.cos(np.pi * i / steps)
        acts.append(q0 + t_ * (np.asarray(q_target) - q0))
    run_actions(rig, acts, rec)


def stage_collect(episodes=30, seed=11, video=REPORTS / "cv_collector_sim.mp4"):
    rig = Rig()
    expert = CVExpert(rig.model, rig.data)
    rng = np.random.default_rng(seed)
    rec = Recorder(video)
    stats = {"episodes": 0, "grasp_ok": 0, "verify_match_gt": 0,
             "cv_missing_at_pick": 0, "ik_reject": 0, "human_reset": 0,
             "carry_lost": 0, "peek_used": 0, "drop_err_mm": [], "cv_pick_err_mm": []}
    t0 = time.time()
    for ep in range(episodes):
        color = ("red", "green")[ep % 2]
        joint_move(rig, Q_PARK, 0.8, rec)      # парковка: освободить обзор
        est = rig.estimate(lying=True)
        if len(est) < 2:
            joint_move(rig, Q_PARK_B, 1.0, rec)  # взгляд с другой стороны
            est = rig.estimate(lying=True)
        for c in CUBES:
            # куб, выбитый неудачным захватом за пределы обзора или зоны,
            # возвращает «человек» (в реале — оператор; счётчик учитывает)
            if c not in est:
                stats["cv_missing_at_pick"] += int(c == color)
                stats["human_reset"] += 1
                print(f"  ep{ep + 1}: {c} не найден с обеих парковок, "
                      f"gt={np.round(cube_gt(rig.model, rig.data, c)[0], 3)}")
                set_cube(rig.model, rig.data, c,
                         (0.25, 0.10 if c == "red" else -0.10, CUBE_HALF),
                         random_pose(rng))
                mujoco.mj_forward(rig.model, rig.data)
                est = rig.estimate(lying=True)
        e = est[color]
        pos_gt, quat_gt = cube_gt(rig.model, rig.data, color)
        stats["cv_pick_err_mm"].append(pose_errors(e["T"], pos_gt, quat_gt)[0])
        rec.add_cv_checkpoint(rig.cv_frame(), est, rig,
                              f"ep{ep + 1}: pick {color}")
        # захват начинается из home: как эпизод записи датасета, и IK
        # стартует из близкой конфигурации (из парковки не сходился)
        joint_move(rig, rig.q_home, 0.7, rec)

        # куб 90-градусно симметричен: если IK не встал с этим yaw губок,
        # эквивалентные yaw +-90/180 дают тот же захват другой конфигурацией
        acts = []
        for dyaw in (0, np.pi / 2, -np.pi / 2, np.pi):
            acts = expert.actions_for(
                expert.plan_pick(e["T"][:3, 3], cv_yaw(e["T"]) + dyaw))
            if acts:
                break
        if not acts:
            stats["ik_reject"] += 1
            stats["human_reset"] += 1
            print(f"  ep{ep + 1} {color}: pick-IK не встал, куб на "
                  f"{np.round(e['T'][:3, 3], 3)}")
            set_cube(rig.model, rig.data, color, (0.25, 0.0, CUBE_HALF),
                     random_pose(rng))
            mujoco.mj_forward(rig.model, rig.data)
            continue
        run_actions(rig, acts, rec)

        # GT-успех захвата фиксируется в момент подъёма (grasp-сегмент
        # датасета заканчивается на lift; потери при последующем переносе —
        # отдельный счётчик carry_lost).
        gt_ok = cube_gt(rig.model, rig.data, color)[0][2] > 0.05

        # CV-верификация подъёма. Куб в гриппере сверху НЕ виден (закрыт
        # запястьем), а куб, не взятый и оставшийся под рукой, — тоже не
        # виден, так что «не виден» сам по себе ничего не доказывает
        # (ловилось как ложноположительное). Второй взгляд: плавный сдвиг
        # гриппера на 10 см вбок по декартову — резкий суставной мах с кубом
        # в губках выбивает куб (проверено: терялись все).
        frame = rig.cv_frame()
        est_lift = rig.estimate(frame)          # куб на весу: без lying-снапа
        got = est_lift.get(color)
        peek_used = got is None
        stats["peek_used"] += int(peek_used)
        if got is not None:
            cv_ok = bool(got["T"][2, 3] > 0.05)
        else:
            side_y = e["T"][1, 3] + (0.10 if e["T"][1, 3] < 0 else -0.10)
            peek = [(np.array([e["T"][0, 3], side_y, TRANSPORT_Z]),
                     GRIPPER_CLOSED, 1.2, cv_yaw(e["T"]))]
            run_actions(rig, expert.actions_for(peek), rec)
            frame = rig.cv_frame()
            est_lift = rig.estimate(frame)
            g2 = est_lift.get(color)
            cv_ok = not (g2 is not None and g2["T"][2, 3] < 0.03)
        stats["episodes"] += 1
        stats["grasp_ok"] += int(gt_ok)
        stats["verify_match_gt"] += int(cv_ok == gt_ok)
        if not gt_ok or cv_ok != gt_ok:
            seen = ("не виден" if got is None
                    else f"z_cv={got['T'][2, 3]:.3f}")
            print(f"  ep{ep + 1} {color}: gt_ok={gt_ok}, cv_ok={cv_ok} "
                  f"({seen}), cv_pick_err={stats['cv_pick_err_mm'][-1]:.2f} мм")
        rec.add_cv_checkpoint(frame, est_lift, rig,
                              f"ep{ep + 1}: lift {'OK' if cv_ok else 'FAIL'} (CV)")

        if gt_ok:
            other = "green" if color == "red" else "red"
            other_xy = (est[other]["T"][:2, 3] if other in est
                        else np.array([0.25, 0.0]))
            # куб 90-градусно симметричен: доворот больше 45 градусов не
            # добавляет случайности ориентации, а трясёт куб в губках
            yaw_pick = cv_yaw(e["T"])
            acts_place = []
            for attempt in range(8):
                # цели дропа с запасом от границ зоны и досягаемости: куб
                # после отпускания откатывается на ~1.5-2 см, и положенный
                # впритык к границе куб на следующем цикле не берётся
                # (pick-IK вне зоны). Углы прямоугольной зоны с радиусом
                # >= 0.335 недостижимы вовсе (скан IK).
                target = np.array([rng.uniform(0.20, 0.30),
                                   rng.uniform(-0.13, 0.13)])
                if np.linalg.norm(target - other_xy) < MIN_CUBE_GAP or \
                        np.hypot(*target) > 0.31:
                    continue
                yaw_t = yaw_pick + rng.uniform(-np.pi / 4, np.pi / 4)
                acts_place = expert.actions_for(
                    expert.plan_place(target, yaw_pick, yaw_t))
                if not acts_place:      # цель ок по радиусу, но yaw не встал
                    acts_place = expert.actions_for(
                        expert.plan_place(target, yaw_pick, yaw_pick))
                if acts_place:
                    break
            if not acts_place:
                # гарантированный сброс: отпустить там, где стоим, — куб не
                # должен оставаться в губках ни при каком исходе планирования
                stats["ik_reject"] += 1
                target = expert.ik.tcp(rig.data)[0][:2].copy()
                yaw_t = yaw_pick
                acts_place = expert.actions_for(
                    expert.plan_place(target, yaw_pick, yaw_pick))
                print(f"  ep{ep + 1} {color}: place-IK не встал, сброс на месте "
                      f"{np.round(target, 3)}")
            run_actions(rig, acts_place, rec)
            settle(rig, 0.7, rec)
            joint_move(rig, Q_PARK, 0.8, rec)
            pos_after = cube_gt(rig.model, rig.data, color)[0]
            drop_err = float(np.linalg.norm(pos_after[:2] - target) * 1000)
            stats["drop_err_mm"].append(drop_err)
            if drop_err > 120:
                stats["carry_lost"] += 1
                print(f"  ep{ep + 1} {color}: перенос потерял куб — "
                      f"drop_err={drop_err:.0f} мм, peek={peek_used}, "
                      f"yaw_t={np.degrees(yaw_t):.0f}°, цель={np.round(target, 3)}")
            est_after = rig.estimate(lying=True)
            if color not in est_after or pos_after[2] > 0.03 or \
                    not (0.10 <= pos_after[0] <= 0.34 and abs(pos_after[1]) <= 0.24):
                stats["human_reset"] += 1
                set_cube(rig.model, rig.data, color, (0.25, 0.0, CUBE_HALF),
                         random_pose(rng))
        else:
            # неудачный захват: вернуть руку и переиграть раскладку куба
            run_actions(rig, expert.actions_for(
                [(np.array([0.25, 0.0, TRANSPORT_Z]), GRIPPER_OPEN, 0.8, None)]),
                rec)
            stats["human_reset"] += 1
            set_cube(rig.model, rig.data, color, (0.25, 0.0, CUBE_HALF),
                     random_pose(rng))
        mujoco.mj_forward(rig.model, rig.data)
    rec.close()
    rep = {
        "episodes": stats["episodes"],
        "grasp_rate": round(stats["grasp_ok"] / max(1, stats["episodes"]), 3),
        "cv_verify_match_gt": round(
            stats["verify_match_gt"] / max(1, stats["episodes"]), 3),
        "cv_pick_err_mm_median": float(np.median(stats["cv_pick_err_mm"])),
        "drop_err_mm_median": float(np.median(stats["drop_err_mm"]))
        if stats["drop_err_mm"] else None,
        "carry_lost": stats["carry_lost"],
        "peek_used": stats["peek_used"],
        "ik_reject": stats["ik_reject"],
        "cv_missing_at_pick": stats["cv_missing_at_pick"],
        "human_reset": stats["human_reset"],
        "seconds": round(time.time() - t0, 1),
        "video": str(video),
    }
    print(f"[collect] {rep}")
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all",
                    choices=["faces", "pose", "collect", "all"])
    ap.add_argument("--episodes", type=int, default=30)
    ap.add_argument("--layouts", type=int, default=60)
    args = ap.parse_args()
    REPORTS.mkdir(exist_ok=True)
    report = {}
    if args.stage in ("faces", "all"):
        assert stage_faces(), "stage faces: раскладка текстур не сходится"
        report["faces"] = "ok"
    if args.stage in ("pose", "all"):
        report["pose"] = stage_pose(n=args.layouts)
    if args.stage in ("collect", "all"):
        report["collect"] = stage_collect(episodes=args.episodes)
    out = REPORTS / "cv_collector_sim.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"отчёт: {out}")


if __name__ == "__main__":
    main()
