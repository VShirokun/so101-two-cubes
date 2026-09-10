#!/usr/bin/env python3
"""Запись обеих камер стенда в один mp4: верхняя 960x540 + кисть 720x540, 15 fps.

  record.py out.mp4 [--seconds 120]      остановить раньше: touch /tmp/roboom_record_stop
"""

import sys
import time
from pathlib import Path

import cv2
import numpy as np

TOP, WR = Path("/tmp/roboom_cam/latest.jpg"), Path("/tmp/roboom_wrist/latest.jpg")
STOP = Path("/tmp/roboom_record_stop")


def main():
    out = sys.argv[1]
    secs = float(sys.argv[sys.argv.index("--seconds") + 1]) if "--seconds" in sys.argv else 120.0
    STOP.unlink(missing_ok=True)
    w = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), 15, (1680, 540))
    t0, n = time.time(), 0
    while time.time() - t0 < secs and not STOP.exists():
        top, wr = cv2.imread(str(TOP)), cv2.imread(str(WR))
        if top is None or wr is None:
            time.sleep(0.05)
            continue
        frame = np.hstack([cv2.resize(top, (960, 540)), cv2.resize(wr, (720, 540))])
        cv2.putText(frame, f"{time.strftime('%H:%M:%S')}  +{time.time() - t0:5.1f} s",
                    (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (40, 220, 40), 2, cv2.LINE_AA)
        w.write(frame)
        n += 1
        time.sleep(max(0.0, t0 + n / 15 - time.time()))
    w.release()
    print(f"{out}: {n} кадров, {time.time() - t0:.0f} с")


if __name__ == "__main__":
    main()
