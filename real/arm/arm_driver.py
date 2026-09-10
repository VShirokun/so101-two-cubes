#!/usr/bin/env python3
"""Драйвер SO-101: радианы MJCF-модели <-> тики приводов, floor-guard, движения.

Единицы: всё наружу — в радианах сим-модели (mlsim/models/so101), гриппер —
в радианах её же сустава. Внутри: тики STS3215 через нули lerobot-калибровки
(homing_offset) и знаки из real/arm/axes.json (поза-якорь).

Floor-guard (обязателен, см. память проекта): каждая команда проверяет ВЕСЬ
интерполированный путь по FK модели — нижняя точка губок/куба (tcp минус
17 мм) не опускается ниже пола режима: transport 20 мм, grasp -5 мм (само
касание разрешено только в фазе захвата). Плюс суставные лимиты: пересечение
механических (тики калибровки) и модельных (MJCF range).

Аварийная остановка: touch /tmp/roboom_arm/stop — рука замирает на месте.
"""

import json
import sys
import time
from pathlib import Path

import mujoco
import numpy as np

_ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(_ROOT / "host"))
from calib import ArmCalibration, find_calibration_file
from feetech import FeetechBus, find_ports
sys.path.insert(0, str(Path(__file__).parent))
from selfcol import SelfCollision  # noqa: E402

NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex",
         "wrist_flex", "wrist_roll", "gripper"]
TICK = 2 * np.pi / 4096
HZ = 30
STOP = Path("/tmp/roboom_arm/stop")
STATE = Path("/tmp/roboom_arm/state.json")
TIP_BELOW_TCP = 0.017          # низ губок/куба ниже точки tcp
Z_FLOOR = {"transport": 0.020, "grasp": -0.010}   # grasp: рука проседает на ~4 мм под нагрузкой
MJCF_RANGE = {"shoulder_pan": (-1.91986, 1.91986),
              "shoulder_lift": (-1.7453293, 1.7453293),
              "elbow_flex": (-1.69, 1.69),
              "wrist_flex": (-1.658063, 1.658063),
              "wrist_roll": (-2.7438473, 2.7438473),
              "gripper": (-0.174533, 1.7453292)}


class GuardError(RuntimeError):
    pass


class ArmDriver:
    def __init__(self):
        self.cal = ArmCalibration.load(find_calibration_file("robots/so_follower"))
        self.signs = {n: json.loads((Path(__file__).parent / "axes.json")
                                    .read_text())[n]["sign"] for n in NAMES}
        self.bus = FeetechBus(find_ports()[0])
        self.model = mujoco.MjModel.from_xml_path(
            str(_ROOT / "mlsim" / "models" / "so101" / "pick_place.xml"))
        self.scratch = mujoco.MjData(self.model)
        self.sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE,
                                     "gripperframe")
        # поправки суставов из автокалибровки (real/calib/autocalib_real.json):
        # модельный угол = scale * прочитанный + offset; без них FK врёт до 3 см
        self.js, self.jo = np.ones(5), np.zeros(5)
        cal = _ROOT / "real" / "calib" / "autocalib_real.json"
        if cal.exists():
            c = json.loads(cal.read_text())
            self.js, self.jo = np.array(c["joint_scale"]), np.radians(c["joint_offset_deg"])
        STOP.unlink(missing_ok=True)
        self.sc = SelfCollision()          # самостолкновения по модели (10.09.2026)

    def self_hits(self, q6, margin=0.0):
        """Контакты/сближения звеньев для команды q6 (конвенция драйвера)."""
        return self.sc.check(np.concatenate([self.q_model(q6[:5]), [float(q6[5])]]), margin)

    def q_model(self, q5):
        """Прочитанные (линейно отображённые) углы -> углы модели."""
        return np.asarray(q5[:5], float) * self.js + self.jo

    def q_cmd(self, q5_model):
        """Углы модели -> что командовать приводам."""
        return (np.asarray(q5_model, float) - self.jo) / self.js

    # --- конвертация: линейная привязка калибровочных упоров к диапазону
    # модели. Нули lerobot-конвенции не совпадают с нулями MJCF (у плеча
    # расхождение ~90 градусов ловилось как «нефизичные» чтения); механические
    # упоры, снятые калибровкой Владимира, и есть общий референс с моделью.
    def rad_of(self, name, raw):
        j = self.cal.joints[name]
        t = 2.0 * (raw - j["min"]) / (j["max"] - j["min"]) - 1.0
        if j["drive_mode"]:
            t = -t
        if self.signs[name] < 0:
            t = -t
        lo, hi = MJCF_RANGE[name]
        return (lo + hi) / 2 + t * (hi - lo) / 2

    def raw_of(self, name, rad):
        j = self.cal.joints[name]
        lo, hi = MJCF_RANGE[name]
        t = (rad - (lo + hi) / 2) / ((hi - lo) / 2)
        if j["drive_mode"]:
            t = -t
        if self.signs[name] < 0:
            t = -t
        raw = j["min"] + (t + 1.0) / 2.0 * (j["max"] - j["min"])
        return int(np.clip(round(raw), j["min"], j["max"]))

    def clamp_rad(self, q6):
        out = []
        for n, v in zip(NAMES, q6):
            lo, hi = MJCF_RANGE[n]
            v = float(np.clip(v, lo, hi))
            # и в тиках назад-вперёд: механика может быть уже модели
            v = self.rad_of(n, self.raw_of(n, v))
            out.append(v)
        return np.array(out)

    # --- состояние --------------------------------------------------------
    def read(self):
        raw = self.bus.read_positions(list(range(1, 7)))
        return np.array([self.rad_of(n, raw[i + 1])
                         for i, n in enumerate(NAMES)]), raw

    def publish(self, q6, raw):
        joints = {}
        for i, n in enumerate(NAMES):
            joints[n] = {"raw": raw[i + 1], "rad": round(float(q6[i]), 4),
                         "norm": self.cal.normalize(n, raw[i + 1])}
        tmp = STATE.with_suffix(".tmp")
        tmp.write_text(json.dumps({"ok": True, "ts": time.time(),
                                   "joints": joints, "driver": True}))
        tmp.replace(STATE)

    # --- геометрия --------------------------------------------------------
    def fk_T(self, q5):
        """T_база->gripperframe (4x4) по прочитанным углам (с поправками)."""
        self.scratch.qpos[:] = 0
        self.scratch.qpos[:5] = self.q_model(q5)
        mujoco.mj_kinematics(self.model, self.scratch)
        T = np.eye(4)
        T[:3, :3] = self.scratch.site_xmat[self.sid].reshape(3, 3)
        T[:3, 3] = self.scratch.site_xpos[self.sid]
        return T

    def tcp_z(self, q5):
        self.scratch.qpos[:] = 0
        self.scratch.qpos[:5] = self.q_model(q5)
        mujoco.mj_kinematics(self.model, self.scratch)
        return float(self.scratch.site_xpos[self.sid][2]), \
            self.scratch.site_xpos[self.sid].copy()

    def path_ok(self, q_from, q_to, mode):
        zmin = Z_FLOOR[mode] + TIP_BELOW_TCP
        for a in np.linspace(0.0, 1.0, 12):
            q = q_from + a * (q_to - q_from)
            z, p = self.tcp_z(q[:5])
            r = float(np.hypot(p[0], p[1]))
            if z < zmin:
                return False, f"tcp z={z:.3f} < {zmin:.3f} на пути (floor-guard)"
            if not (0.04 < r < 0.40):
                return False, f"tcp r={r:.3f} вне досягаемости"
            hits = self.self_hits(q)
            if hits:
                return False, f"самостолкновение на пути: {self.sc.describe(hits)}"
        return True, ""

    # --- движения ---------------------------------------------------------
    def torque(self, on):
        self.bus.set_torque(list(range(1, 7)), on)

    def hold(self):
        q, raw = self.read()
        goals = {self.cal.joints[n]["id"]: raw[i + 1]
                 for i, n in enumerate(NAMES)}
        self.bus.sync_write(42, 2, goals)   # goal position addr STS3215
        return q

    def goto(self, q6_target, seconds, mode="transport", tick=None):
        """tick(q_goal6, q_read6, raw) — вызывается КАЖДЫЙ такт 30 Гц (запись
        датасета); без него чтение каждый третий такт."""
        q6_target = self.clamp_rad(np.asarray(q6_target, float))
        q_now, raw = self.read()
        ok, why = self.path_ok(q_now, q6_target, mode)
        if not ok:
            raise GuardError(why)
        steps = max(2, int(seconds * HZ))
        t0 = time.time()
        for i in range(1, steps + 1):
            if STOP.exists():
                self.hold()
                raise GuardError("аварийный стоп-файл — рука остановлена")
            s = 0.5 - 0.5 * np.cos(np.pi * i / steps)
            q = q_now + s * (q6_target - q_now)
            goals = {self.cal.joints[n]["id"]: self.raw_of(n, q[j])
                     for j, n in enumerate(NAMES)}
            self.bus.sync_write(42, 2, goals)
            time.sleep(max(0.0, t0 + i / HZ - time.time()))
            if tick is not None or i % 3 == 0:
                qr, rr = self.read()
                if i % 3 == 0:
                    self.publish(qr, rr)
                if tick is not None:
                    tick(q, qr, rr)
                # слежение: рука обязана идти за командой; отставание =
                # упёрлась или неверный знак — замереть немедленно
                lag = float(np.max(np.abs(qr[:5] - q[:5])))
                if lag > 0.35:
                    self.hold()
                    raise GuardError(f"отставание от команды {lag:.2f} рад — стоп")
        qr, rr = self.read()
        self.publish(qr, rr)
        return qr

    def lift(self, step_deg=2.0):
        """Подъём из-под «пола» (рука лежит губками на столе, goto откажет):
        сустав и знак с наибольшим ростом tcp z по FK, маленькими шагами,
        пока низ губок не окажется выше пола transport с запасом 1 см."""
        zmin = Z_FLOOR["transport"] + TIP_BELOW_TCP + 0.01
        # Цель копится относительно ПРЕДЫДУЩЕЙ команды, а не прочитанной позы:
        # привод — П-регулятор, под нагрузкой вытянутой руки он не отрабатывает
        # шаг в 2° от текущей позы (ток 4, ошибка 25–50 тиков, 09.09.2026: рука
        # легла губками на стол и 80 шагов её не подняли). Опережение растёт,
        # пока рука не пойдёт; больше 0.3 рад — считаем, что упёрлась.
        q_cmd = None
        for _ in range(80):
            q, raw = self.read()
            self.publish(q, raw)
            z, _ = self.tcp_z(q[:5])
            if z >= zmin:
                return q
            if q_cmd is None:
                q_cmd = q.copy()
            best = None
            for j in (1, 2, 3):
                for sgn in (1, -1):
                    dq = np.zeros(6)
                    dq[j] = sgn * np.radians(step_deg)
                    dz = self.tcp_z((q + dq)[:5])[0] - z
                    if best is None or dz > best[0]:
                        best = (dz, dq)
            if best[0] < 0.0005:
                raise GuardError("ни один сустав не поднимает tcp — подъём невозможен")
            q_cmd = q_cmd + best[1]
            if self.self_hits(q_cmd):
                raise GuardError(f"подъём: самостолкновение {self.sc.describe(self.self_hits(q_cmd))}")
            if float(np.max(np.abs(q_cmd[:5] - q[:5]))) > 0.3:
                raise GuardError("подъём: рука не идёт за командой (опережение 0.3 рад) — упёрлась")
            goals = {self.cal.joints[n]["id"]: self.raw_of(n, q_cmd[j])
                     for j, n in enumerate(NAMES)}
            self.bus.sync_write(42, 2, goals)
            time.sleep(0.25)
        raise GuardError("подъём не вывел tcp над пол")

    def settle(self, min_seconds=2.0, max_extra=2.0, tol_rad=0.004, tick=None):
        """Осадка в конечном положении (требование Владимира): минимум 2 с,
        затем ждать, пока чтения суставов перестанут меняться. Любой замер
        (кадр, детекция, пара для калибровки) — только после этого.
        С tick — чтение и колбэк каждый такт 30 Гц (пауза пишется в датасет)."""
        t0 = time.time()
        prev, t_prev, goal = None, t0, None
        while True:
            time.sleep(1.0 / HZ if tick is not None else 0.15)
            q, raw = self.read()
            self.publish(q, raw)
            if goal is None:
                goal = q.copy()
            if tick is not None:
                tick(goal, q, raw)
            now = time.time()
            if now - t0 < min_seconds or now - t_prev < 0.15:
                continue
            if prev is not None and (float(np.max(np.abs(q[:5] - prev[:5]))) < tol_rad
                                     or now - t0 >= min_seconds + max_extra):
                return q
            prev, t_prev = q, now

    # --- проверка знаков приводами ---------------------------------------
    def verify_signs(self):
        """Микродвижения +-4 градуса по каждому суставу: командуем +delta в
        модельных радианах и проверяем, что фактическое чтение сдвинулось
        в ту же сторону. Порядок — от безопасных суставов к несущим."""
        order = ["wrist_roll", "gripper", "wrist_flex", "elbow_flex",
                 "shoulder_lift", "shoulder_pan"]
        delta = np.radians(4)
        self.torque(True)
        self.hold()
        time.sleep(0.3)
        results = {}
        for n in order:
            j = NAMES.index(n)
            q0, _ = self.read()
            q1 = q0.copy()
            q1[j] += delta
            self.goto(q1, 1.0)
            time.sleep(0.3)
            q2, _ = self.read()
            moved = float(q2[j] - q0[j])
            ok = moved > delta * 0.4
            results[n] = (round(np.degrees(moved), 2), ok)
            print(f"  {n:14s}: команда +4.0°, факт {np.degrees(moved):+5.2f}° "
                  f"{'OK' if ok else '!! ЗНАК/ПРИВОД'}")
            self.goto(q0, 1.0)
            if not ok:
                raise GuardError(f"сустав {n}: движение не совпало с командой")
        print("все знаки подтверждены приводами")
        return results

    def close(self):
        self.bus.close()


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify-signs", action="store_true")
    ap.add_argument("--hold", action="store_true",
                    help="включить момент и держать текущую позу")
    args = ap.parse_args()
    d = ArmDriver()
    try:
        if args.hold:
            d.torque(True)
            q = d.hold()
            print("держу позу:", np.round(q, 3))
        if args.verify_signs:
            d.verify_signs()
    finally:
        d.close()


if __name__ == "__main__":
    main()
