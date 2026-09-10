#!/usr/bin/env python3
"""Единая панель стенда: обе камеры, телеметрия руки, статусы, подсказка.

Читает только файлы (кадры серверов камер, state.json телеметрии,
calib.log, /tmp/roboom_status.txt) — ни камер, ни шины не занимает.
Выход: q или Esc.
"""

import json
import re
import time
from pathlib import Path

import cv2
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parents[2] / "mlsim"))
from cube_cv import DICT

CAMS = [
    {"dir": Path("/tmp/roboom_cam"), "label": "C920 (table, CV)",
     "size": (1120, 630)},
    {"dir": Path("/tmp/roboom_wrist"), "label": "wrist (arm)",
     "size": (640, 480)},
]
ARM = Path("/tmp/roboom_arm/state.json")
STATUS = Path("/tmp/roboom_status.txt")

DET = cv2.aruco.ArucoDetector(DICT, cv2.aruco.DetectorParameters())
W, H = 1760, 810
GREEN, RED, YEL, GRAY = (60, 220, 60), (60, 60, 255), (60, 210, 240), (190, 190, 190)


def put(img, text, xy, scale=0.8, color=GRAY, thick=2):
    cv2.putText(img, text, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, color,
                thick, cv2.LINE_AA)


class CamView:
    def __init__(self, spec):
        self.spec = spec
        self.mtime = 0.0
        self.frame = None
        self.updates = []
        self.prev_small = None
        self.frozen_since = None
        self.n_markers = 0
        self.sharp = 0.0

    def poll(self):
        p = self.spec["dir"] / "latest.jpg"
        try:
            m = p.stat().st_mtime
        except FileNotFoundError:
            return
        if m == self.mtime:
            return
        img = cv2.imread(str(p))
        if img is None:
            return
        self.mtime = m
        self.updates.append(time.time())
        small = img[::31, ::31].astype(int)
        noise = None if self.prev_small is None or \
            self.prev_small.shape != small.shape \
            else float(np.abs(small - self.prev_small).mean())
        self.prev_small = small
        if noise is None or noise > 0.05:
            self.frozen_since = None
        elif self.frozen_since is None:
            self.frozen_since = time.time()
        corners, ids, _ = DET.detectMarkers(
            cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
        if ids is not None:
            cv2.aruco.drawDetectedMarkers(img, corners, ids)
        self.n_markers = 0 if ids is None else len(ids)
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cy, cx = g.shape[0] // 2, g.shape[1] // 2
        self.sharp = float(cv2.Laplacian(
            g[max(0, cy - 120):cy + 120, max(0, cx - 120):cx + 120],
            cv2.CV_64F).var())
        self.frame = img

    def render(self):
        w, h = self.spec["size"]
        canvas = np.full((h, w, 3), 25, np.uint8)
        self.updates = [t for t in self.updates if t > time.time() - 2]
        if self.frame is not None:
            canvas = cv2.resize(self.frame, (w, h))
        if True:
            hh, ww = canvas.shape[:2]
            for zone in calib_empty_zones(self.spec["dir"] / "calib.log"):
                try:
                    v, hname = zone.split("-")
                except ValueError:
                    continue
                r0 = 0 if v == "top" else hh // 2
                ci = {"left": 0, "center": 1, "right": 2}.get(hname, 1)
                x0 = ci * ww // 3
                ov = canvas[r0:r0 + hh // 2, x0:x0 + ww // 3]
                ov[:] = (ov * 0.6 + np.array([0, 0, 90])).clip(0, 255)
                put(canvas, "CUBE HERE", (x0 + 40, r0 + hh // 4), 1.0, RED, 2)
        stale = time.time() - self.mtime > 2.5 if self.mtime else True
        frozen = self.frozen_since is not None and \
            time.time() - self.frozen_since > 2.0
        status = "NO FRAMES" if stale else (
            "FROZEN" if frozen else "LIVE")
        color = RED if status != "LIVE" else GREEN
        if status != "LIVE":
            canvas[:] = (canvas * 0.35).astype(np.uint8)
            put(canvas, status + " - camera is not delivering frames",
                (canvas.shape[1] // 6, canvas.shape[0] // 2), 1.3, RED, 3)
        put(canvas, f"{self.spec['label']}  {status}  "
            f"{len(self.updates) / 2:.0f}fps  markers:{self.n_markers}  "
            f"sharp:{self.sharp:.0f}", (14, 30), 0.75, color, 2)
        return canvas


def arm_block(w=640, h=150):
    img = np.full((h, w, 3), 32, np.uint8)
    try:
        d = json.loads(ARM.read_text())
    except Exception:
        put(img, "ARM: no telemetry (arm_telemetry not running)",
            (14, 40), 0.7, RED)
        return img
    age = time.time() - d.get("ts", 0)
    if not d.get("ok"):
        put(img, f"ARM: {d.get('err', '?')}", (14, 40), 0.7, RED)
        return img
    color = GREEN if age < 1.5 else RED
    put(img, f"ARM: link OK (data age {age:.1f}s)", (14, 34), 0.75, color)
    j = d.get("joints", {})
    row1 = "  ".join(f"{k.split('_')[0][:5]}:{v.get('norm', v['raw']):+7.2f}"
                     for k, v in list(j.items())[:3])
    row2 = "  ".join(f"{k.split('_')[0][:5]}:{v.get('norm', v['raw']):+7.2f}"
                     for k, v in list(j.items())[3:])
    put(img, row1, (14, 78), 0.75, GRAY)
    put(img, row2, (14, 116), 0.75, GRAY)
    return img


def calib_empty_zones(log):
    """Имена пустых зон из последней empty-строки calib.log."""
    if not log.exists() or time.time() - log.stat().st_mtime > 600:
        return []
    lines = log.read_text().strip().splitlines()
    for line in reversed(lines):
        if "OK, can finish" in line:
            return []
        if "empty:" in line:
            return [z.strip() for z in line.split("empty:")[1].split(",")]
        if "сохранено" in line or "RMS" in line:
            return []
    return []


def status_block(w=W, h=180):
    img = np.full((h, w, 3), 18, np.uint8)
    lines = []
    if STATUS.exists():
        lines = STATUS.read_text().strip().splitlines()
    calib = ""
    for lg in (Path("/tmp/roboom_cam/calib.log"),
               Path("/tmp/roboom_wrist/calib.log")):
        if lg.exists() and time.time() - lg.stat().st_mtime < 600:
            txt = lg.read_text().strip().splitlines()
            prog = [l for l in txt
                    if "видов" in l or "RMS" in l or "сохранено" in l or "fx=" in l]
            if prog:
                calib = prog[-1]
    y = 48
    for line in lines[:2]:
        put(img, line, (18, y), 1.0, YEL, 2)
        y += 46
    if calib:
        put(img, f"calibration: {calib}", (18, y), 0.9, GREEN, 2)
    put(img, "q/Esc - close panel", (w - 320, h - 16), 0.6, GRAY, 1)
    return img


def main():
    cv2.namedWindow("RoboOM panel", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("RoboOM panel", 1500, 700)
    views = [CamView(c) for c in CAMS]
    while True:
        for v in views:
            v.poll()
        top = np.hstack([
            views[0].render(),
            np.vstack([views[1].render(), arm_block()]),
        ])
        canvas = np.vstack([top, status_block()])
        cv2.imshow("RoboOM panel", canvas)
        if cv2.waitKey(30) & 0xFF in (ord("q"), 27):
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
