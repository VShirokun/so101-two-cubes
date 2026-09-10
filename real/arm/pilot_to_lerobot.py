#!/usr/bin/env python3
"""Сырые эпизоды пилота (real/data/pilot/ep_*) -> lerobot-датасет.

Кадры: верхняя камера 1920x1080 -> центральный кроп 4:3 (1440x1080) -> 640x480
(как top в сим-датасетах); камера кисти 640x480 как есть. state/action —
радианы модели (6 суставов, как в симе). Берутся только успешные эпизоды
(meta.success), целиком. Перекраска кубов — отдельным проходом позже, здесь
кубы чёрно-белые, задача одна на все эпизоды.

  pilot_to_lerobot.py [--raw real/data/pilot] [--root real/data/lerobot/real-pilot] [--overwrite]
  pilot_to_lerobot.py --raw real/data/pilot_recolor --root real/data/lerobot/real-pilot-color
"""

import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).parents[2]
JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
FPS = 30
TASK = "pick up the cube and drop it"
REPO_ID = "roboom/real-pilot"


def top_frame(img_bgr):
    """1920x1080 -> кроп 1440x1080 по центру -> 640x480, RGB; перекрашенные
    эпизоды уже 640x480 (и без дисторсии) — как есть."""
    h, w = img_bgr.shape[:2]
    if (h, w) == (480, 640):
        return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    cw = h * 4 // 3
    x0 = (w - cw) // 2
    return cv2.cvtColor(cv2.resize(img_bgr[:, x0:x0 + cw], (640, 480),
                                   interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, default=ROOT / "real/data/pilot")
    ap.add_argument("--root", type=Path, default=ROOT / "real/data/lerobot/real-pilot")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--all", action="store_true", help="и неуспешные эпизоды тоже")
    a = ap.parse_args()
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    if a.root.exists():
        if not a.overwrite:
            raise SystemExit(f"уже есть {a.root}; --overwrite чтобы пересоздать")
        shutil.rmtree(a.root)
    features = {
        "observation.state": {"dtype": "float32", "shape": (6,), "names": JOINTS},
        "action": {"dtype": "float32", "shape": (6,), "names": JOINTS},
    }
    for cam in ("top", "wrist_cam"):
        features[f"observation.images.{cam}"] = {
            "dtype": "video", "shape": (480, 640, 3), "names": ["height", "width", "channels"]}
    ds = LeRobotDataset.create(repo_id=REPO_ID, fps=FPS, root=a.root, robot_type="so101",
                               features=features, use_videos=True)
    eps = sorted(p for p in a.raw.glob("ep_*") if (p / "meta.json").exists())
    saved, skipped = 0, 0
    for ep in eps:
        meta = json.loads((ep / "meta.json").read_text())
        if not (meta.get("success") or a.all):
            skipped += 1
            continue
        n = 0
        for line in open(ep / "steps.jsonl"):
            r = json.loads(line)
            top = cv2.imread(str(ep / "top" / f"{r['i']:06d}.jpg"))
            wr = cv2.imread(str(ep / "wrist" / f"{r['i']:06d}.jpg"))
            if top is None or wr is None:
                continue
            ds.add_frame({
                "observation.state": np.asarray(r["state"], np.float32),
                "action": np.asarray(r["action"], np.float32),
                "task": meta.get("task", TASK),   # перекрашенные: задача по цвету
                "observation.images.top": top_frame(top),
                "observation.images.wrist_cam": cv2.cvtColor(wr, cv2.COLOR_BGR2RGB),
            })
            n += 1
        ds.save_episode()
        saved += 1
        print(f"  {ep.name}: {n} кадров, куб {meta.get('cube')}, "
              f"посадка {meta.get('landing_err_mm')} мм")
    ds.finalize()
    print(f"готово: {saved} эпизодов (пропущено неуспешных {skipped}) -> {a.root}")


if __name__ == "__main__":
    main()
