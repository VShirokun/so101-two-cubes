"""Проверка самостолкновений руки SO-101 по модели MuJoCo.

Зачем. 10.09.2026 Владимир видел, как рука упёрлась корпусом камеры кисти в
собственное «тело». Ни один путь движения (goto, lift, политика) этого не
проверял: floor-guard смотрит только на пол и досягаемость.

Как. Берётся модель руки `mlsim/models/so101/so101.xml` (у звеньев, губок и
корпуса камеры кисти есть коробки столкновений), к базе добавляются две
коробки (корпус мотора плеча и основание — в исходной модели у базы только
визуальные меши). Для позы считается `mj_forward` с маржой геометрии
`margin`, и все контакты между НЕсоседними звеньями (соседние MuJoCo не
проверяет сам) возвращаются как список (звено1, звено2, зазор). Зазор < 0 —
проникновение, 0..margin — сближение.

Проверено по 1509 позам из 100 записанных эпизодов сборщика: ни одного
контакта при margin 0 — модель не даёт ложных срабатываний на нормальных
движениях.

  from selfcol import SelfCollision
  sc = SelfCollision(); hits = sc.check(q6, margin=0.02)   # q6 — радианы модели
  sc.clearance(q6) -> минимальный зазор (м) или None, если дальше margin
"""
from pathlib import Path

import mujoco
import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
SO101 = _ROOT / "mlsim/models/so101/so101.xml"
JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
# коробки базы в системе тела base (по габаритам визуальных мешей)
BASE_BOXES = [  # (центр, полуразмеры)
    ((0.013, 0.0, 0.052), (0.040, 0.025, 0.016)),   # корпус мотора плеча и держатель
    ((0.021, 0.0, 0.012), (0.043, 0.055, 0.012)),   # основание
]


class SelfCollision:
    def __init__(self, margin_default: float = 0.02):
        spec = mujoco.MjSpec.from_file(str(SO101))
        base = spec.body("base")
        for k, (pos, size) in enumerate(BASE_BOXES):
            g = base.add_geom()
            g.name = f"base_col{k}"
            g.type = mujoco.mjtGeom.mjGEOM_BOX
            g.pos = np.array(pos)
            g.size = np.array(size)
            g.contype, g.conaffinity = 1, 1
            g.group = 3
            g.rgba = np.array([1.0, 0.0, 0.0, 0.3])
        # пары, которые касаются по конструкции: мотор плеча сидит в базе (MuJoCo не
        # фильтрует base–shoulder, т.к. база приварена к миру), подвижная губка при
        # полном раскрытии в 14 мм от запястья
        for b1, b2 in (("base", "shoulder"), ("wrist", "moving_jaw_so101_v1")):
            spec.add_exclude(bodyname1=b1, bodyname2=b2)
        self.model = spec.compile()
        self.data = mujoco.MjData(self.model)
        self.qadr = [self.model.jnt_qposadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, j)]
                     for j in JOINTS]
        self.margin_default = margin_default
        self._names = {b: mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, b) for b in range(self.model.nbody)}

    def check(self, q6, margin=None):
        """Список (звено1, звено2, зазор м) для контактов и сближений ближе margin."""
        margin = self.margin_default if margin is None else margin
        self.model.geom_margin[:] = margin
        self.data.qpos[:] = 0
        for a, v in zip(self.qadr, np.asarray(q6, float)[:6]):
            self.data.qpos[a] = v
        mujoco.mj_forward(self.model, self.data)
        out = []
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            b1, b2 = self.model.geom_bodyid[c.geom1], self.model.geom_bodyid[c.geom2]
            out.append((self._names[b1], self._names[b2], float(c.dist)))
        return sorted(out, key=lambda t: t[2])

    def clearance(self, q6, margin=None):
        hits = self.check(q6, margin)
        return hits[0][2] if hits else None

    @staticmethod
    def describe(hits):
        if not hits:
            return ""
        b1, b2, d = hits[0]
        return f"{b1}–{b2} {'проникновение' if d < 0 else 'зазор'} {abs(d) * 1000:.0f} мм"
