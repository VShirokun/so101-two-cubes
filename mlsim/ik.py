"""Обратная кинематика SO-101 для скриптового оператора.

Численный метод (демпфированные наименьшие квадраты по якобиану), а не
аналитика: у руки 5 степеней свободы, звенья заданы мешами со смещёнными
системами координат, и любая аналитическая формула здесь — источник тихих
ошибок при обновлении модели. Якобиан берётся из самой модели, поэтому
решение остаётся верным, даже если модель поменяется.

Кроме позиции задаётся ОСЬ ПОДХОДА. Без неё рука приходит в нужную точку в
произвольной ориентации и закрывает губки плашмя об стол: губки схвата
расходятся вдоль локальной оси Z сайта, а пальцы вытянуты вдоль локальной X.
Пять связей (3 позиция + 2 направление) ровно покрывают пять суставов;
вращение вокруг самой оси подхода остаётся свободным — кубик симметричен.
"""

import mujoco
import numpy as np

ARM_JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]

# Точка захвата относительно сайта gripperframe, в локальных осях сайта:
# сайт стоит у неподвижной губки, центр между губками смещён вдоль оси
# раскрытия (локальная Z) примерно на 11 мм.
TCP_OFFSET = np.array([0.0, 0.0, 0.011])
# Ось раскрытия губок в локальных осях сайта — вдоль неё пальцы расходятся.
OPEN_AXIS_LOCAL = np.array([0.0, 0.0, 1.0])

# Пальцы вытянуты вдоль локальной X сайта — это и есть ось подхода.
APPROACH_AXIS_LOCAL = np.array([1.0, 0.0, 0.0])


class ArmIK:
    def __init__(self, model: mujoco.MjModel, site: str = "gripperframe") -> None:
        self.model = model
        self.site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site)
        if self.site_id < 0:
            raise ValueError(f"нет сайта {site} в модели")
        self.body_id = model.site_bodyid[self.site_id]

        ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j) for j in ARM_JOINTS]
        self.dof = np.array([model.jnt_dofadr[i] for i in ids])
        self.qadr = np.array([model.jnt_qposadr[i] for i in ids])
        self.lo = np.array([model.jnt_range[i][0] for i in ids])
        self.hi = np.array([model.jnt_range[i][1] for i in ids])

    def tcp(self, data: mujoco.MjData) -> tuple[np.ndarray, np.ndarray]:
        """Мировые координаты точки захвата и текущей оси подхода."""
        R = data.site_xmat[self.site_id].reshape(3, 3)
        return data.site_xpos[self.site_id] + R @ TCP_OFFSET, R @ APPROACH_AXIS_LOCAL

    def solve(
        self,
        data: mujoco.MjData,
        target: np.ndarray,
        approach: np.ndarray | None = np.array([0.0, 0.0, -1.0]),
        q_init: np.ndarray | None = None,
        iters: int = 200,
        damping: float = 0.10,
        ori_weight: float = 0.35,
        opening_yaw: float | None = None,
    ) -> tuple[np.ndarray, float]:
        """Возвращает (углы 5 суставов, ошибка позиции в метрах).

        `approach` — куда должны смотреть пальцы в мировых осях; по умолчанию
        вниз, то есть захват сверху. None отключает контроль ориентации.

        `opening_yaw` — мировой азимут ОСИ РАСКРЫТИЯ губок. Задаёт, под каким
        углом схват встречает предмет: для кубика с поворотом yaw губки должны
        смыкаться перпендикулярно его граням, иначе кубик доворачивается под
        схват в момент смыкания. Кубик симметричен, поэтому берётся ближайший
        эквивалент по модулю 90°.
        """
        # Работаем на копии состояния: пробные позы IK не должны шевелить
        # настоящую сцену, иначе кубики поедут вслед за итерациями.
        d = mujoco.MjData(self.model)
        d.qpos[:] = data.qpos
        q = np.array(data.qpos[self.qadr] if q_init is None else q_init, dtype=float)

        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        target = np.asarray(target, dtype=float)
        a_des = None if approach is None else np.asarray(approach, float) / np.linalg.norm(approach)
        o_des = None
        if opening_yaw is not None:
            o_des = np.array([np.cos(opening_yaw), np.sin(opening_yaw), 0.0])

        err = np.inf
        for _ in range(iters):
            d.qpos[self.qadr] = q
            mujoco.mj_kinematics(self.model, d)
            mujoco.mj_comPos(self.model, d)

            point, a_cur = self.tcp(d)
            dp = target - point
            err = float(np.linalg.norm(dp))

            # Якобиан именно ТОЧКИ ЗАХВАТА, а не сайта: точка жёстко связана
            # с телом схвата, mj_jac считает для неё честный якобиан.
            mujoco.mj_jac(self.model, d, jacp, jacr, point, self.body_id)
            J = jacp[:, self.dof]
            e = dp

            if a_des is not None:
                # Ошибка направления как малый поворот, совмещающий оси.
                e_rot = np.cross(a_cur, a_des)
                if o_des is not None:
                    R = d.site_xmat[self.site_id].reshape(3, 3)
                    o_cur = R @ OPEN_AXIS_LOCAL
                    o_cur = o_cur - a_cur * float(o_cur @ a_cur)   # в плоскость, ⊥ подходу
                    n = np.linalg.norm(o_cur)
                    if n > 1e-6:
                        o_cur /= n
                        # кубик симметричен: из 4 эквивалентов берём ближайший
                        best = None
                        for k in range(4):
                            ang = np.deg2rad(90 * k)
                            c, s_ = np.cos(ang), np.sin(ang)
                            cand = np.array([o_des[0] * c - o_des[1] * s_,
                                             o_des[0] * s_ + o_des[1] * c, 0.0])
                            if best is None or float(cand @ o_cur) > float(best @ o_cur):
                                best = cand
                        e_rot = e_rot + np.cross(o_cur, best)
                if err < 1e-4 and np.linalg.norm(e_rot) < 1e-3:
                    break
                J = np.vstack([J, ori_weight * jacr[:, self.dof]])
                e = np.concatenate([dp, ori_weight * e_rot])
            elif err < 1e-4:
                break

            JT = J.T
            A = J @ JT + (damping ** 2) * np.eye(J.shape[0])
            q = np.clip(q + JT @ np.linalg.solve(A, e), self.lo, self.hi)

        return q, err
