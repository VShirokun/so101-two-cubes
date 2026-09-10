#!/usr/bin/env python3
"""Выверка знаков суставов по одной позе-якорю.

Отключает момент (руку ДЕРЖАТЬ РУКАМИ — она обмякнет!), ждёт, пока руку
выставят по образцу и отпустят на 3 секунды неподвижно, читает позиции и
вычисляет знак каждого сустава против MJCF-модели. Результат: axes.json.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[2] / "host"))
from calib import ArmCalibration, find_calibration_file
from feetech import FeetechBus, find_ports

NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex",
         "wrist_flex", "wrist_roll", "gripper"]
Q_ANCHOR = np.array([0.6, -0.9, 1.1, 0.7, 1.0, 0.8])
TICK = 2 * np.pi / 4096


def main():
    cal = ArmCalibration.load(find_calibration_file("robots/so_follower"))
    bus = FeetechBus(find_ports()[0])
    ids = list(range(1, 7))
    bus.set_torque(ids, False)
    print("МОМЕНТ ОТКЛЮЧЁН — придерживайте руку!")
    print("Выставьте руку по образцу (картинка) и отпустите на 3 секунды...")
    hist = []
    moved = False
    while True:
        raw = bus.read_positions(ids)
        hist.append((time.time(), raw))
        hist = [(t, r) for t, r in hist if t > time.time() - 3.0]
        if len(hist) > 5:
            spans = [max(r[i] for _, r in hist) - min(r[i] for _, r in hist)
                     for i in ids]
            if max(spans) > 60:
                moved = True
                print(".", end="", flush=True)
            elif moved and max(spans) < 12:
                break
        time.sleep(0.15)
    print("\nруку зафиксировали — читаю")
    raw = hist[-1][1]
    axes = {}
    all_ok = True
    homing = {n: cal.raw[n]["homing_offset"] for n in NAMES}
    for i, n in enumerate(NAMES):
        rad = (raw[i + 1] + homing[n] - 2048) * TICK
        q = Q_ANCHOR[i]
        sign = 1 if abs(rad - q) <= abs(rad + q) else -1
        resid = sign * rad - q
        ok = abs(resid) < 0.5
        all_ok &= ok
        axes[n] = {"sign": sign}
        print(f"  {n:14s} raw={raw[i + 1]:5d} rad={rad:+.2f} "
              f"ожидалось {q:+.2f} -> sign {sign:+d}, невязка {resid:+.2f} "
              f"{'OK' if ok else '!! ПРОВЕРИТЬ'}")
    out = Path(__file__).parent / "axes.json"
    out.write_text(json.dumps(axes, indent=1))
    print(f"сохранено: {out}")
    if not all_ok:
        print("Есть большие невязки: либо поза сильно отличалась от образца, "
              "либо ноль сустава не 2048+homing — повторите точнее")
    bus.close()
    print("момент так и ВЫКЛЮЧЕН — положите руку в устойчивое положение")


if __name__ == "__main__":
    main()
