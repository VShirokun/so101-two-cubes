#!/usr/bin/env python3
"""Живой просмотр камеры стенда: отдельное окно + CV-оверлей.

Камеру не занимает — читает кадры cam_server.py из /tmp/roboom_cam/latest.jpg
и рисует поверх то, что видит наш пайплайн: рамки и ID ArUco-меток
(DICT_4X4_50 — словарь кубов и коробки). Выход: q или Esc в окне.
"""

import argparse
import os
import time
from pathlib import Path

import cv2

ap = argparse.ArgumentParser()
ap.add_argument("--dir", default="/tmp/roboom_cam")
ap.add_argument("--title", default="RoboOM cam")
args = ap.parse_args()
SRC = Path(args.dir) / "latest.jpg"
DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
DET = cv2.aruco.ArucoDetector(DICT, cv2.aruco.DetectorParameters())

log = open(Path(args.dir) / "viewer.log", "w", buffering=1)
print("viewer стартует", file=log)
cv2.namedWindow(args.title, cv2.WINDOW_NORMAL)
cv2.resizeWindow(args.title, 1280, 720)
print("окно создано", file=log)
last_mtime = 0.0
frame = None
prev_small = None
frozen_since = None
upd_times = []
import numpy as np
import sys
hb = 0.0
while True:
    if time.time() - hb > 5:
        hb = time.time()
        print(f"alive {time.strftime('%H:%M:%S')} last_mtime={last_mtime}",
              file=log)
    try:
        m = SRC.stat().st_mtime
    except FileNotFoundError:
        m = 0.0
    if m != last_mtime:
        img = cv2.imread(str(SRC))
        if img is not None:
            last_mtime = m
            # критерий живости (идея Владимира): живая камера шумит, кадры
            # обязаны отличаться; одинаковые кадры = застывший захват
            small = img[::37, ::37].astype(int)
            noise = None if prev_small is None or prev_small.shape != small.shape                 else float(np.abs(small - prev_small).mean())
            prev_small = small
            if noise is None or noise > 0.05:
                frozen_since = None
            elif frozen_since is None:
                frozen_since = time.time()
            corners, ids, _ = DET.detectMarkers(
                cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
            if ids is not None:
                cv2.aruco.drawDetectedMarkers(img, corners, ids)
            n = 0 if ids is None else len(ids)
            h, w = img.shape[:2]
            if frozen_since is not None and time.time() - frozen_since > 2.0:
                cv2.putText(img, "FROZEN: identical frames", (20, 95),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3,
                            cv2.LINE_AA)
            g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            cy, cx = g.shape[0] // 2, g.shape[1] // 2
            roi = g[max(0, cy - 160):cy + 160, max(0, cx - 160):cx + 160]
            sharp = float(cv2.Laplacian(roi, cv2.CV_64F).var())
            cv2.rectangle(img, (cx - 160, cy - 160), (cx + 160, cy + 160),
                          (200, 200, 60), 2)
            cv2.putText(img, f"sharp: {sharp:.0f} (turn focus ring to maximize)",
                        (20, img.shape[0] - 25), cv2.FONT_HERSHEY_SIMPLEX,
                        1.0, (200, 200, 60), 2, cv2.LINE_AA)
            upd_times.append(time.time())
            while upd_times and upd_times[0] < time.time() - 2.0:
                upd_times.pop(0)
            live = "LIVE" if frozen_since is None else "?"
            cv2.putText(img, f"{w}x{h} {len(upd_times)/2.0:.0f}fps  "
                        f"markers: {n}  {live} "
                        f"noise={0.0 if noise is None else noise:.2f}",
                        (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                        (40, 220, 40), 3, cv2.LINE_AA)
            frame = img
    if frame is not None:
        show = frame.copy()
        if time.time() - last_mtime > 2.5:
            cv2.putText(show, "NO FRESH FRAMES (server stopped?)",
                        (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.1,
                        (0, 0, 255), 3, cv2.LINE_AA)
        cv2.imshow(args.title, show)
    key = cv2.waitKey(15) & 0xFF
    if key in (ord("q"), 27):
        break
print("viewer закрыт пользователем", file=log)
cv2.destroyAllWindows()
