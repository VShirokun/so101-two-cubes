#!/usr/bin/env python3
"""Сырые эпизоды сборщика (real/data/pilot/ep_*) -> датасет «взять и ПОДНЯТЬ»
в раскладке LeRobot v2.1, которую читает GR00T (как gr00t-*-v21 в симе).

Эпизод обрезается по концу фазы `lift` (рука подняла куб на 9 см и 2 с
стоит, пока камера кисти подтверждает захват); перенос, сброс и возврат
в датасет не входят. Берутся эпизоды с подтверждённым захватом (meta.held),
успешная посадка не требуется. Кадры: верхняя камера 1920x1080 -> кроп 4:3
по центру -> 640x480 (как top в симе), кисть 640x480 как есть; state/action —
радианы модели с поправками автокалибровки (6 суставов), как записано.
Кубы чёрно-белые с метками, задача одна: «Подними кубик.»

  pilot_to_lift_v21.py [--raw real/data/pilot] [--dst /srv/data/vshirokun/datasets/real-lift-v21] [--jobs 8]
Без lerobot: parquet через pandas, видео через ffmpeg из imageio (libx264, crf 20).
"""
import argparse
import json
import shutil
import subprocess
import sys
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
FPS = 30
TASK = "Подними кубик."
TASK_BY_COLOR = {"green": "Подними зелёный кубик.", "gray": "Подними серый кубик.", "red": "Подними красный кубик.",
                 "blue": "Подними синий кубик.", "yellow": "Подними жёлтый кубик.", "purple": "Подними фиолетовый кубик.",
                 "cyan": "Подними голубой кубик."}
CAMS = ("front", "wrist")
LAST_PHASE = "lift"


def top_frame(img):
    h, w = img.shape[:2]
    cw = h * 4 // 3
    x0 = (w - cw) // 2
    return cv2.resize(img[:, x0:x0 + cw], (640, 480), interpolation=cv2.INTER_AREA)


def write_video(frames_bgr, out, fps=FPS):
    import imageio_ffmpeg
    out.parent.mkdir(parents=True, exist_ok=True)
    h, w = frames_bgr[0].shape[:2]
    cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error",
           "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", str(fps), "-i", "-",
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", "-threads", "2", str(out)]
    p = subprocess.run(cmd, input=b"".join(np.ascontiguousarray(f).tobytes() for f in frames_bgr),
                       capture_output=True)
    if p.returncode != 0:
        raise RuntimeError(p.stderr.decode()[-800:])


IDLE_KEEP = 10          # тактов покоя, которые остаются от каждого статичного участка
IDLE_EPS = 0.004        # рад: цель не меняется и рука её достигла


def drop_idle(recs):
    """Убрать длинные простои (осадка 2 с на зависании и после подъёма): клон
    поведения на них «залипает» и смыкает губки на высоте зависания, не
    спускаясь (живой прогон 09.09.2026, 0/3). От каждого статичного участка
    остаются первые IDLE_KEEP тактов."""
    out, run = [], 0
    prev = None
    for r in recs:
        a, st = np.asarray(r["action"]), np.asarray(r["state"])
        static = prev is not None and np.abs(a - prev).max() < IDLE_EPS \
            and np.abs(a[:5] - st[:5]).max() < 0.02
        run = run + 1 if static else 0
        prev = a
        if run <= IDLE_KEEP:
            out.append(r)
    return out


def episode_task(ep_dir):
    """Перекрашенные эпизоды: задача по цвету целевого куба из meta (colors[cube])."""
    m = json.loads((ep_dir / "meta.json").read_text())
    colors = m.get("colors")
    if colors and m.get("cube") in colors:
        return TASK_BY_COLOR.get(colors[m["cube"]], TASK)
    return TASK


def convert(job):
    ep_dir, dst, new, no_idle = job
    recs = [json.loads(l) for l in open(ep_dir / "steps.jsonl")]
    last = max(r["i"] for r in recs if r["phase"] == LAST_PHASE)
    recs = [r for r in recs if r["i"] <= last]
    if no_idle:
        recs = drop_idle(recs)
    top, wr, st, ac = [], [], [], []
    for r in recs:
        t = cv2.imread(str(ep_dir / "top" / f"{r['i']:06d}.jpg"))
        w = cv2.imread(str(ep_dir / "wrist" / f"{r['i']:06d}.jpg"))
        if t is None or w is None:
            continue
        top.append(top_frame(t))
        wr.append(w)
        st.append(np.asarray(r["state"], np.float32))
        ac.append(np.asarray(r["action"], np.float32))
    n = len(st)
    write_video(top, dst / f"videos/chunk-000/observation.images.front/episode_{new:06d}.mp4")
    write_video(wr, dst / f"videos/chunk-000/observation.images.wrist/episode_{new:06d}.mp4")
    return new, n, st, ac


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, default=ROOT / "real/data/pilot")
    ap.add_argument("--dst", type=Path, default=Path("/srv/data/vshirokun/datasets/real-lift-v21"))
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--all", action="store_true", help="и эпизоды без подтверждённого захвата")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--drop-idle", action="store_true", help="убрать простои длиннее IDLE_KEEP тактов")
    a = ap.parse_args()
    eps = []
    for p in sorted(a.raw.glob("ep_*")):
        if not (p / "meta.json").exists():
            continue
        m = json.loads((p / "meta.json").read_text())
        # подъём был, если камера кисти подтвердила куб в губках ИЛИ куб доехал до цели
        landed = m.get("landing_err_mm") is not None and m["landing_err_mm"] < 40
        if m.get("held") or landed or a.all:
            eps.append(p)
    if a.dst.exists():
        if not a.overwrite:
            raise SystemExit(f"уже есть {a.dst}; --overwrite")
        shutil.rmtree(a.dst)
    (a.dst / "meta").mkdir(parents=True)
    (a.dst / "data/chunk-000").mkdir(parents=True)
    res = {}
    with Pool(a.jobs) as pool:
        for new, n, st, ac in pool.imap_unordered(convert, [(p, a.dst, i, a.drop_idle) for i, p in enumerate(eps)]):
            res[new] = (n, st, ac)
            print(f"  {eps[new].name} -> episode {new}: {n} кадров", flush=True)
    gidx, lines, lengths = 0, [], []
    tasks = {}
    for new in range(len(eps)):
        n, st, ac = res[new]
        task = episode_task(eps[new])
        ti = tasks.setdefault(task, len(tasks))
        pd.DataFrame({
            "observation.state": list(st), "action": list(ac),
            "timestamp": np.arange(n, dtype=np.float32) / FPS,
            "frame_index": np.arange(n), "episode_index": np.full(n, new),
            "index": np.arange(gidx, gidx + n), "task_index": np.full(n, ti, dtype=np.int64),
        }).to_parquet(a.dst / f"data/chunk-000/episode_{new:06d}.parquet")
        lines.append({"episode_index": new, "tasks": [task], "length": n})
        lengths.append(n)
        gidx += n
    with open(a.dst / "meta/episodes.jsonl", "w") as fh:
        for l in lines:
            fh.write(json.dumps(l, ensure_ascii=False) + "\n")
    (a.dst / "meta/tasks.jsonl").write_text("".join(json.dumps({"task_index": i, "task": t}, ensure_ascii=False) + "\n"
                                                   for t, i in tasks.items()))
    vid = {"video.fps": float(FPS), "video.codec": "h264", "video.pix_fmt": "yuv420p",
           "video.is_depth_map": False, "has_audio": False}
    feats = {
        "observation.state": {"dtype": "float32", "shape": [6], "names": JOINTS},
        "action": {"dtype": "float32", "shape": [6], "names": JOINTS},
        "timestamp": {"dtype": "float32", "shape": [1], "names": None},
        "frame_index": {"dtype": "int64", "shape": [1], "names": None},
        "episode_index": {"dtype": "int64", "shape": [1], "names": None},
        "index": {"dtype": "int64", "shape": [1], "names": None},
        "task_index": {"dtype": "int64", "shape": [1], "names": None},
    }
    for cam in CAMS:
        feats[f"observation.images.{cam}"] = {"dtype": "video", "shape": [480, 640, 3],
                                              "names": ["height", "width", "channels"], "info": vid}
    json.dump({"codebase_version": "v2.1", "robot_type": "so101", "total_episodes": len(eps),
               "total_frames": gidx, "total_tasks": len(tasks), "total_videos": 2 * len(eps),
               "total_chunks": 1, "chunks_size": 1000, "fps": FPS, "splits": {"train": f"0:{len(eps)}"},
               "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
               "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
               "features": feats}, open(a.dst / "meta/info.json", "w"), ensure_ascii=False, indent=2)
    json.dump({"state": {"single_arm": {"start": 0, "end": 5}, "gripper": {"start": 5, "end": 6}},
               "action": {"single_arm": {"start": 0, "end": 5}, "gripper": {"start": 5, "end": 6}},
               "video": {"front": {"original_key": "observation.images.front"},
                         "wrist": {"original_key": "observation.images.wrist"}},
               "annotation": {"human.task_description": {"original_key": "task_index"}}},
              open(a.dst / "meta/modality.json", "w"), indent=2)
    print(f"готово: {a.dst}: {len(eps)} эпизодов, {gidx} кадров, длина {min(lengths)}–{max(lengths)}, "
          f"медиана {int(np.median(lengths))}")


if __name__ == "__main__":
    sys.exit(main())
