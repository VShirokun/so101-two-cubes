"""Загрузка калибровок lerobot (формат v3: {joint: {id, drive_mode, homing_offset,
range_min, range_max}}) и нормализация позиций.

Тело руки: [-100, 100], gripper: [0, 100] — как в lerobot.
Homing offset уже записан в EEPROM сервоприводов при калибровке, поэтому
сырые Present_Position уже «выровнены» и попадают в [range_min, range_max].
"""
from __future__ import annotations

import glob
import json
import os

from consts import JOINTS

CAL_ROOT = os.environ.get(
    "HF_LEROBOT_CALIBRATION",
    os.path.expanduser("~/.cache/huggingface/lerobot/calibration"),
)


def find_calibration_file(kind: str, arm_id: str | None = None) -> str | None:
    """kind: 'leader' | 'follower'. Ищет JSON в стандартном кэше lerobot."""
    sub = "teleoperators" if kind == "leader" else "robots"
    # сначала кэш lerobot, затем копия в репозитории (переезд на другую машину)
    repo = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                        "real", "calib", "lerobot_calibration")
    cands = sorted(glob.glob(os.path.join(CAL_ROOT, sub, "*", "*.json"))) + \
        sorted(glob.glob(os.path.join(repo, "*", "*.json")))
    key = kind.split("/")[-1]
    cands = [c for c in cands if key in os.path.basename(c) or key in c]
    if arm_id:
        exact = [c for c in cands if os.path.splitext(os.path.basename(c))[0] == arm_id]
        if exact:
            return exact[0]
    return cands[0] if cands else None


class ArmCalibration:
    def __init__(self, data: dict, path: str = "<inline>"):
        self.path = path
        self.raw = data
        self.joints: dict[str, dict] = {}
        for name in JOINTS:
            j = data.get(name)
            if not j:
                raise ValueError(f"в калибровке {path} нет сустава {name}")
            self.joints[name] = {
                "id": int(j["id"]),
                "drive_mode": int(j.get("drive_mode", 0)),
                "min": int(j["range_min"]),
                "max": int(j["range_max"]),
            }

    @classmethod
    def load(cls, path: str) -> "ArmCalibration":
        with open(path) as f:
            return cls(json.load(f), path)

    def ids(self) -> list[int]:
        return [self.joints[n]["id"] for n in JOINTS]

    def normalize(self, name: str, raw: int) -> float:
        j = self.joints[name]
        span = j["max"] - j["min"]
        if not span:
            # Битая калибровка (min == max). Раньше здесь получался 0.0, то
            # есть −100 по шкале тела: ведомая уходила в упор. Для сустава,
            # про который мы ничего не знаем, безопасная поза — середина.
            return 0.0 if name != "gripper" else 0.0
        f = (raw - j["min"]) / span
        if j["drive_mode"]:
            f = 1.0 - f
        f = max(0.0, min(1.0, f))
        if name == "gripper":
            return round(f * 100.0, 2)
        return round(f * 200.0 - 100.0, 2)

    def denormalize(self, name: str, norm: float) -> int:
        j = self.joints[name]
        f = norm / 100.0 if name == "gripper" else (norm + 100.0) / 200.0
        f = max(0.0, min(1.0, f))
        if j["drive_mode"]:
            f = 1.0 - f
        return int(round(j["min"] + f * (j["max"] - j["min"])))
