#!/usr/bin/env python3
"""Безопасная проверка руки: пинг сервоприводов и чтение поз. БЕЗ движений."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2] / "host"))
from calib import ArmCalibration, find_calibration_file
from feetech import FeetechBus, find_ports

ports = find_ports()
print("порты:", ports)
assert ports, "адаптер руки не найден"
bus = FeetechBus(ports[0])
ids = bus.scan(range(1, 8))
print("сервоприводы на шине:", ids)
raw = bus.read_positions(ids)
print("сырые позиции:", raw)
cal_path = find_calibration_file("robots/so_follower")
if cal_path:
    cal = ArmCalibration.load(cal_path)
    print("калибровка:", cal_path)
    for i, name in enumerate(
            ["shoulder_pan", "shoulder_lift", "elbow_flex",
             "wrist_flex", "wrist_roll", "gripper"], start=1):
        if i in raw:
            try:
                print(f"  {name:14s} raw={raw[i]:5d} -> {cal.normalize(name, raw[i]):+.3f}")
            except Exception as e:
                print(f"  {name:14s} raw={raw[i]:5d} (normalize: {e})")
bus.close()
print("связь с рукой в порядке; движений не было")
