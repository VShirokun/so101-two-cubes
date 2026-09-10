#!/usr/bin/env python3
"""Паблишер телеметрии руки: позиции суставов -> /tmp/roboom_arm/state.json.

Единственный процесс на serial-порту (панель и прочие читают json, не шину).
Когда появится драйвер движений, он заменит паблишера в этой роли.
Остановка: touch /tmp/roboom_arm/stop.
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2] / "host"))
from calib import ArmCalibration, find_calibration_file
from feetech import FeetechBus, find_ports

OUT = Path("/tmp/roboom_arm")
NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex",
         "wrist_flex", "wrist_roll", "gripper"]


def publish(data):
    tmp = OUT / ".state.tmp"
    tmp.write_text(json.dumps(data))
    tmp.replace(OUT / "state.json")


def main():
    OUT.mkdir(exist_ok=True)
    stop = OUT / "stop"
    stop.unlink(missing_ok=True)
    cal = None
    cal_path = find_calibration_file("robots/so_follower")
    if cal_path:
        cal = ArmCalibration.load(cal_path)
    bus, ids = None, []
    while not stop.exists():
        try:
            if bus is None:
                ports = find_ports()
                if not ports:
                    publish({"ok": False, "err": "адаптер не найден",
                             "ts": time.time()})
                    time.sleep(2)
                    continue
                bus = FeetechBus(ports[0])
                ids = bus.scan(range(1, 7))
            raw = bus.read_positions(ids)
            joints = {}
            for i, name in enumerate(NAMES, start=1):
                if i in raw:
                    j = {"raw": raw[i]}
                    if cal:
                        try:
                            j["norm"] = round(cal.normalize(name, raw[i]), 2)
                        except Exception:
                            pass
                    joints[name] = j
            publish({"ok": True, "ts": time.time(), "joints": joints})
        except Exception as e:
            publish({"ok": False, "err": str(e)[:120], "ts": time.time()})
            try:
                bus.close()
            except Exception:
                pass
            bus = None
            time.sleep(1)
        time.sleep(0.1)
    if bus:
        bus.close()
    publish({"ok": False, "err": "остановлен", "ts": time.time()})


if __name__ == "__main__":
    main()
