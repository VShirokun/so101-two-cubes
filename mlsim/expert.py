"""Скриптовый оператор: берёт названный кубик и кладёт в коробку.

Он заменяет человека-телеоператора, которого у нас нет без железа. Важно
понимать границу честности: это НЕ обученная политика, а источник
демонстраций — ровно та роль, которую на реальном роботе играет оператор.
Политика (ACT/SmolVLA) обучается на его записях и потом работает сама, уже
без доступа к координатам кубика: только камера и позы суставов.

Позиции кубиков рандомизируются в каждом эпизоде. Оба кубика всегда в кадре —
значит, единственный источник различия между задачами «возьми красный» и
«возьми зелёный» — это текст инструкции.
"""

import os

import mujoco
import numpy as np

from ik import ArmIK

CONTROL_HZ = 30
GRIPPER_OPEN = 1.35
GRIPPER_APPROACH = 1.60   # на подводе к кубику: шире, чтобы зазор был с ОБЕИХ сторон
GRIPPER_CLOSED = 0.02

# Потолок рабочей высоты: выше 9 см рука уже не держит вертикальный подход
# (проверено перебором целей — ошибка растёт с 7 мм до 27 мм). Перенос идёт
# на этой высоте: борт коробки 4,4 см, кубик проходит с запасом.
TRANSPORT_Z = 0.09

# Рабочая зона для случайной раскладки кубиков (метры, начало — основание руки)
ZONE_X = (0.19, 0.31)
ZONE_Y = (-0.15, 0.15)
MIN_CUBE_GAP = 0.09        # чтобы схват не задевал соседний кубик

# Язык инструкций. По умолчанию русский — как и было; переключается переменной
# окружения ROBOOM_TASK_LANG=en. Английский вариант нужен потому, что токенизатор
# базовой SmolVLA рвёт русский текст на байтовые огрызки: «красный» и «зелёный»
# расходятся в 28 токенах из 34, тогда как red/green — ровно в одном из двенадцати.
# Датасет и политику надо записывать/обучать/оценивать на ОДНОМ языке.
_TASK_TEXT = {
    "ru": {"red": "Возьми красный кубик и положи его в коробку.",
           "green": "Возьми зелёный кубик и положи его в коробку."},
    "en": {"red": "Pick up the red cube and put it in the box.",
           "green": "Pick up the green cube and put it in the box."},
}
TASK_LANG = os.environ.get("ROBOOM_TASK_LANG", "ru").lower()
if TASK_LANG not in _TASK_TEXT:
    raise SystemExit(f"ROBOOM_TASK_LANG должен быть ru или en, получено {TASK_LANG!r}")

CUBES = {
    "red": {"body": "red_cube", "joint": "red_free",
            "task": _TASK_TEXT[TASK_LANG]["red"]},
    "green": {"body": "green_cube", "joint": "green_free",
              "task": _TASK_TEXT[TASK_LANG]["green"]},
}


def set_layout(model: mujoco.MjModel, data: mujoco.MjData, layout: dict[str, np.ndarray]) -> None:
    """Расставить кубики. Адреса qpos берём у модели по имени сустава:
    захардкоженные индексы — прямой путь записать позицию кубика поверх
    сустава руки и потом искать причину в физике.

    Элемент layout — [x, y, z] или [x, y, z, yaw]: четвёртое число — поворот
    вокруг вертикали. Раньше все кубики стояли строго одинаково (кватернион
    единичный), и модель никогда не видела повёрнутых граней."""
    for color, pos in layout.items():
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, CUBES[color]["joint"])
        adr = model.jnt_qposadr[jid]
        data.qpos[adr:adr + 3] = pos[:3]
        yaw = float(pos[3]) if len(pos) > 3 else 0.0
        data.qpos[adr + 3:adr + 7] = (np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2))
        data.qvel[model.jnt_dofadr[jid]:model.jnt_dofadr[jid] + 6] = 0.0


# Расширенная зона для ЗАПИСИ данных (не для оценки). Разбор 195 промахов
# показал: 86% неудач — «кубик не поднят», и в 61% таких случаев кубик к тому
# моменту ВЫТОЛКНУТ за пределы номинальной зоны (медиана выноса 5.7 см). Модель
# в обучении такие раскладки не видела и заходила по кругу до конца лимита.
# Границы взяты по фактической достижимости IK (проверено перебором):
# при малых x рука берёт кубик до |y| ~ 0.24, при x > 0.31 не дотягивается.
WIDE_X = (0.12, 0.32)
WIDE_Y = (-0.22, 0.22)
BIN_KEEPOUT = 0.10        # радиус вокруг коробки, куда кубик не кладём


def sample_layout(rng: np.random.Generator, random_yaw: bool = False,
                  wide: bool = False) -> dict[str, np.ndarray]:
    """Две позиции в рабочей зоне, разнесённые не ближе MIN_CUBE_GAP.

    random_yaw добавляет каждому кубику случайный поворот вокруг вертикали.
    По умолчанию выключено, чтобы прежние стенды оценки воспроизводили
    исторические раскладки байт в байт.

    wide=True расширяет зону до достижимой руки (для ЗАПИСИ данных): так в
    обучении появляются кубики у краёв и в углах — там, где реально оказывается
    кубик, случайно выбитый политикой. Стенд оценки эту опцию НЕ использует:
    протокол сравнения остаётся прежним."""
    zx, zy = (WIDE_X, WIDE_Y) if wide else (ZONE_X, ZONE_Y)
    bin_xy = np.array([0.10, 0.26])
    while True:
        a = np.array([rng.uniform(*zx), rng.uniform(*zy), 0.014])
        b = np.array([rng.uniform(*zx), rng.uniform(*zy), 0.014])
        if wide and (np.linalg.norm(a[:2] - bin_xy) < BIN_KEEPOUT
                     or np.linalg.norm(b[:2] - bin_xy) < BIN_KEEPOUT):
            continue          # не подкладывать кубик вплотную к коробке
        if np.linalg.norm(a[:2] - b[:2]) >= MIN_CUBE_GAP:
            if random_yaw:
                a = np.append(a, rng.uniform(0, np.pi / 2))   # куб симметричен: 90° хватает
                b = np.append(b, rng.uniform(0, np.pi / 2))
            return {"red": a, "green": b}


class PickPlaceExpert:
    """Конечный автомат: подвестись → опуститься → взять → перенести → бросить."""

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        self.model = model
        self.data = data
        self.ik = ArmIK(model)
        self.bin_site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "bin_center")

    def cube_yaw(self, color: str) -> float:
        """Азимут горизонтальных граней кубика — угол, под которым губки должны
        встретить его в плоскости стола.

        Извлекать yaw из кватерниона напрямую нельзя: после падения кубик может
        улечься на ЛЮБУЮ грань, и разложение на углы Эйлера даст азимут не той
        оси. Вместо этого берём столбцы матрицы поворота (нормали граней) и
        находим самый горизонтальный — его направление в плоскости XY и есть
        искомый угол (эквивалентность mod 90° добирает ик-решатель)."""
        bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, CUBES[color]["body"])
        R = self.data.xmat[bid].reshape(3, 3)
        axis = R[:, int(np.argmin(np.abs(R[2, :])))]   # нормаль с наименьшей вертикальной частью
        return float(np.arctan2(axis[1], axis[0]))

    def body_pos(self, name: str) -> np.ndarray:
        bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
        return self.data.xpos[bid].copy()

    def plan(self, color: str) -> list[tuple[np.ndarray, float, float]]:
        """Список фаз: (цель схвата, команда схвата, длительность в секундах).

        Высоты подобраны под кубик 28 мм: подход на 12 см выше, захват на
        уровне центра кубика, подъём на 15 см — этого хватает, чтобы пронести
        кубик над бортом коробки.
        """
        cube = self.body_pos(CUBES[color]["body"])
        bin_xy = self.data.site_xpos[self.bin_site][:2]

        above = np.array([cube[0], cube[1], TRANSPORT_Z])
        # Промежуточная точка «зависание над кубиком»: 3 см над гранью, уже
        # точно по центру и под углом кубика. Разбор промахов (этап 24) показал
        # главный класс неудач: рука ЗАДЕВАЕТ кубик на спуске и выбивает его из
        # зоны (61% неудач «кубик не поднят»). Спуск разбит на две фазы —
        # быстрый до зависания и медленный, покадрово плотный, у самой грани:
        # в критической фазе у модели становится втрое больше кадров на
        # коррекцию прицела, а скорость касания при ошибке — втрое ниже.
        hover = np.array([cube[0], cube[1], cube[2] + 0.030])
        grasp = np.array([cube[0], cube[1], cube[2] + 0.002])
        lift = np.array([cube[0], cube[1], TRANSPORT_Z])
        over_bin = np.array([bin_xy[0], bin_xy[1], TRANSPORT_Z])

        # Азимут раскрытия губок ведём по ФАКТИЧЕСКОМУ повороту кубика: губки
        # обязаны встретить грани перпендикулярно, иначе кубик доворачивается
        # под схват при смыкании (замечание Владимира по превью датасета).
        yaw = self.cube_yaw(color)
        return [
            (above,    GRIPPER_APPROACH, 0.8,  yaw),
            (hover,    GRIPPER_APPROACH, 0.5,  yaw),   # быстрый спуск до 3 см над гранью
            (hover,    GRIPPER_APPROACH, 0.2,  yaw),   # замереть, показать прицел
            (grasp,    GRIPPER_APPROACH, 0.6,  yaw),   # МЕДЛЕННЫЙ спуск последних 3 см
            (grasp,    GRIPPER_APPROACH, 0.25, yaw),   # пауза: кубик между губками
            (grasp,    GRIPPER_CLOSED,   0.5,  yaw),   # смыкание перпендикулярно граням
            (lift,     GRIPPER_CLOSED,   0.7,  yaw),
            (over_bin, GRIPPER_CLOSED,   1.2,  None),
            (over_bin, GRIPPER_OPEN,     0.4,  None),  # отпустить над коробкой
            (over_bin, GRIPPER_OPEN,     0.3,  None),
        ]

    def actions_for(self, phases) -> list[np.ndarray]:
        """Исполнитель произвольного списка фаз от ТЕКУЩЕЙ позы (для повторов)."""
        return self._interp(phases)

    def actions(self, color: str) -> list[np.ndarray]:
        """Полная траектория как последовательность команд приводов (6 значений).

        Между фазами интерполируем в пространстве СУСТАВОВ: так движение
        плавное и повторяет то, что делает оператор, ведущий руку рукой.
        """
        return self._interp(self.plan(color))

    def _interp(self, phases) -> list[np.ndarray]:
        q = np.array(self.data.qpos[self.ik.qadr], dtype=float)
        grip = float(self.data.ctrl[5])
        out: list[np.ndarray] = []

        for phase in phases:
            target, grip_cmd, seconds = phase[0], phase[1], phase[2]
            yaw = phase[3] if len(phase) > 3 else None
            q_goal, err = self.ik.solve(self.data, target, q_init=q, opening_yaw=yaw)
            if err > 0.02:
                # Недостижимая точка — эпизод честно бракуется вызывающим кодом.
                return []
            steps = max(2, int(seconds * CONTROL_HZ))
            for i in range(1, steps + 1):
                a = i / steps
                # Сглаживание по косинусу: без рывков на старте и в конце.
                s = 0.5 - 0.5 * np.cos(np.pi * a)
                out.append(np.concatenate([q + s * (q_goal - q), [grip + s * (grip_cmd - grip)]]))
            q, grip = q_goal, grip_cmd

        return out
