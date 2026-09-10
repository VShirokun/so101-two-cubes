"""Алгоритмический перенос кубика в коробку из состояния «кубик в губках».

Разделение труда (решение Владимира, 09.09.2026): обученная политика только
поднимает кубик, дальше — детерминированный алгоритм, которому известна поза
коробки (сайт `bin_center` в модели; на реальном стенде — калибровка).

Фазы от ТЕКУЩЕЙ позы руки (интерполяция в суставах, как у скриптового
эксперта, `PickPlaceExpert._interp`):
  0. дожатие губок до GRIPPER_CLOSED на месте (политика смыкает слабее эксперта);
  1. подъём до TRANSPORT_Z над текущей точкой (если рука ниже);
  2. перенос над центр коробки на TRANSPORT_Z (борт 4,4 см, кубик проходит);
  3. спуск до LOWER_Z: низ губок над бортом, кубик уже внутри контура коробки;
  4. раскрытие губок — кубик опускается на дно;
  5. отход вверх на TRANSPORT_Z (чтобы не зацепить борт при следующем движении).
Схват держит min(текущая команда, GRIPPER_CLOSED) до самого раскрытия.

Проверка: `criteria.cube_in_bin` — единый критерий проекта.

  python carry.py --selftest --attempts 50     # эксперт берёт и поднимает, алгоритм несёт
"""
import argparse
import sys

import mujoco
import numpy as np

from criteria import cube_in_bin, cube_xyz
from expert import CONTROL_HZ, CUBES, GRIPPER_CLOSED, GRIPPER_OPEN, TRANSPORT_Z, PickPlaceExpert, sample_layout, set_layout
from ik import ArmIK

LOWER_Z = 0.070     # tcp: низ губок (tcp − 1,7 см) над бортом 4,4 см с запасом


class CarryToBin:
    """Перенос кубика из губок в коробку. Пользуется только позой руки и
    известной позой коробки — где лежит кубик, алгоритм не знает."""

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        self.model, self.data = model, data
        self.ik = ArmIK(model)
        self.bin_site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "bin_center")
        self.sub = int(round(1.0 / (CONTROL_HZ * model.opt.timestep)))

    def phases(self):
        tcp, _ = self.ik.tcp(self.data)
        grip = float(self.data.ctrl[5])
        bin_xy = self.data.site_xpos[self.bin_site][:2]
        up = np.array([tcp[0], tcp[1], max(tcp[2], TRANSPORT_Z)])
        over = np.array([bin_xy[0], bin_xy[1], TRANSPORT_Z])
        low = np.array([bin_xy[0], bin_xy[1], LOWER_Z])
        # дожать губки до GRIPPER_CLOSED: политика смыкает слабее эксперта, и
        # 3 из 79 поднятых кубиков выпадали при переносе (оценка 10.09.2026)
        hold = min(grip, GRIPPER_CLOSED)
        return [
            (tcp,  hold,         0.3, None),
            (up,   hold,         0.5, None),
            (over, hold,         1.2, None),
            (low,  hold,         0.5, None),
            (low,  GRIPPER_OPEN, 0.4, None),
            (over, GRIPPER_OPEN, 0.4, None),
        ]

    def actions(self):
        """Команды приводов (6 значений) на каждый такт; [] — IK не встал."""
        q = np.array(self.data.qpos[self.ik.qadr], dtype=float)
        grip = float(self.data.ctrl[5])
        out = []
        for target, grip_cmd, seconds, yaw in self.phases():
            q_goal, err = self.ik.solve(self.data, target, q_init=q, opening_yaw=yaw)
            if err > 0.02:
                return []
            steps = max(2, int(seconds * CONTROL_HZ))
            for i in range(1, steps + 1):
                s = 0.5 - 0.5 * np.cos(np.pi * i / steps)
                out.append(np.concatenate([q + s * (q_goal - q), [grip + s * (grip_cmd - grip)]]))
            q, grip = q_goal, grip_cmd
        return out

    def run(self, on_step=None) -> int:
        """Исполнить перенос физикой. -> число тактов (0 — IK не встал)."""
        acts = self.actions()
        for k, a in enumerate(acts):
            self.data.ctrl[:] = a
            for _ in range(self.sub):
                mujoco.mj_step(self.model, self.data)
            if on_step is not None:
                on_step(k)
        return len(acts)


def selftest(attempts: int, seed: int, scene: str) -> int:
    """Эксперт берёт и поднимает (фазы до `lift` включительно), алгоритм несёт."""
    model = mujoco.MjModel.from_xml_path(scene)
    data = mujoco.MjData(model)
    expert = PickPlaceExpert(model, data)
    carry = CarryToBin(model, data)
    rng = np.random.default_rng(seed)
    ok = lifted = ik_fail = 0
    for i in range(attempts):
        color = ["red", "green"][i % 2]
        mujoco.mj_resetDataKeyframe(model, data, 0)
        set_layout(model, data, sample_layout(rng, random_yaw=True))
        mujoco.mj_forward(model, data)
        plan = expert.plan(color)
        lift_phase = next(k for k, p in enumerate(plan) if np.allclose(p[0][2], TRANSPORT_Z) and k > 0)
        acts = expert.actions_for(plan[:lift_phase + 1])
        for a in acts:
            data.ctrl[:] = a
            for _ in range(carry.sub):
                mujoco.mj_step(model, data)
        if cube_xyz(model, data, color)[2] < 0.05:
            print(f"  {i+1}: эксперт не поднял ({color}), перенос не проверяется")
            continue
        lifted += 1
        n = carry.run()
        if n == 0:
            ik_fail += 1
        # дать кубику упасть и успокоиться
        for _ in range(int(0.5 * CONTROL_HZ) * carry.sub):
            mujoco.mj_step(model, data)
        got = cube_in_bin(model, data, color, carry.bin_site)
        ok += int(got)
        print(f"  {i+1}/{attempts} [{color}] {'✓' if got else '✗'} кубик {np.round(cube_xyz(model, data, color), 3).tolist()}")
    print(f"\nИТОГ переноса: {ok}/{lifted} в коробке (поднято экспертом {lifted}/{attempts}, IK не встал {ik_fail})")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--attempts", type=int, default=50)
    ap.add_argument("--seed", type=int, default=777)
    ap.add_argument("--scene", default="models/so101/pick_place.xml")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(selftest(a.attempts, a.seed, a.scene))
