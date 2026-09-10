#!/usr/bin/env python3
"""Датасет «взять и поднять» из датасета «взять и положить в коробку»
(LeRobot v2.1, раскладка GR00T): каждый эпизод обрезается в конце фазы
подъёма — там, где схват, сомкнувшись на кубике, поднял его и ещё не
двинулся к коробке.

Зачем. Задача политики для реального стенда сокращена до «поднять кубик
на высоту, достаточную, чтобы он не выпал» (дальше кубик несёт алгоритм).
Прежде чем учить на реальных данных, рецепт проверяется на симе: там
успех судит физика (eval_groot_lift.py).

Правило отсечки (по прямой кинематике MJCF-модели руки из observation.state):
  c — ПОСЛЕДНЕЕ смыкание схвата (переход state[5] >= 0.3 -> < 0.3), после
      которого схват остаётся сомкнутым и точка схвата поднимается > 4 см;
  i — первый такт после c, где точка схвата ушла в плоскости стола дальше
      2 см от места смыкания (начало переноса к коробке);
  эпизод = кадры [0, i). Эпизоды без такого смыкания пропускаются.

  make_lift_dataset.py --src <v21 датасет> --dst <куда> [--limit N] [--jobs 16]
Дальше: python gr00t/data/stats.py --dataset-path <dst> --embodiment-tag NEW_EMBODIMENT \
        --modality-config-path examples/SO100/so100_config.py
"""
import argparse
import json
import shutil
import subprocess
import sys
from multiprocessing import Pool
from pathlib import Path

import av
import mujoco
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SCENE = str(ROOT / "mlsim/models/so101/pick_place.xml")
JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
CAMS = ["front", "wrist"]
TASKS = {"красный": "Подними красный кубик.", "зелёный": "Подними зелёный кубик."}
GRIP_CLOSED, RISE_MIN, DEPART = 0.3, 0.04, 0.02

_M = _D = _QADR = _SID = None


def _fk_init():
    global _M, _D, _QADR, _SID
    _M = mujoco.MjModel.from_xml_path(SCENE)
    _D = mujoco.MjData(_M)
    _QADR = [_M.jnt_qposadr[mujoco.mj_name2id(_M, mujoco.mjtObj.mjOBJ_JOINT, j)] for j in JOINTS]
    _SID = mujoco.mj_name2id(_M, mujoco.mjtObj.mjOBJ_SITE, "gripperframe")


def tcp_path(states):
    if _M is None:
        _fk_init()
    out = []
    for s in states:
        _D.qpos[:] = 0
        for a, v in zip(_QADR, s):
            _D.qpos[a] = v
        mujoco.mj_kinematics(_M, _D)
        out.append(_D.site_xpos[_SID].copy())
    return np.array(out)


def lift_cut(states):
    """Индекс отсечки (эксклюзивно) или None."""
    g = states[:, 5]
    P = tcp_path(states)
    closed = g < GRIP_CLOSED
    starts = [k for k in range(1, len(g)) if closed[k] and not closed[k - 1]]
    for c in reversed(starts):
        dep = np.linalg.norm(P[c:, :2] - P[c, :2], axis=1) > DEPART
        i = c + int(np.argmax(dep)) if dep.any() else len(g)
        if closed[c:i].all() and P[c:i, 2].max() - P[c, 2] > RISE_MIN:
            return i
    return None


def read_frames(path, n):
    frames = []
    with av.open(str(path)) as cont:
        for fr in cont.decode(video=0):
            frames.append(fr.to_ndarray(format="rgb24"))
            if len(frames) >= n:
                break
    return frames


def write_video(frames, out, fps):
    import imageio_ffmpeg
    out.parent.mkdir(parents=True, exist_ok=True)
    h, w = frames[0].shape[:2]
    cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(fps), "-i", "-",
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", str(out)]
    p = subprocess.run(cmd, input=b"".join(np.ascontiguousarray(f).tobytes() for f in frames),
                       capture_output=True)
    if p.returncode != 0:
        raise RuntimeError(p.stderr.decode()[-800:])


def convert_episode(job):
    src, dst, ep, fps, task_map = job
    df = pd.read_parquet(src / f"data/chunk-000/episode_{ep:06d}.parquet")
    st = np.stack(df["observation.state"].values).astype(np.float32)
    cut = lift_cut(st)
    if cut is None or cut < 10:
        return ep, None
    for cam in CAMS:
        v = src / f"videos/chunk-000/observation.images.{cam}/episode_{ep:06d}.mp4"
        frames = read_frames(v, cut)
        if len(frames) < cut:
            return ep, None
        write_video(frames, dst / f"videos/chunk-000/observation.images.{cam}/episode_{ep:06d}.mp4", fps)
    old_task = int(df["task_index"].iloc[0])
    return ep, (df.iloc[:cut].copy(), task_map[old_task])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--dst", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--jobs", type=int, default=16)
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    info = json.loads((a.src / "meta/info.json").read_text())
    fps = info["fps"]
    old_tasks = [json.loads(l) for l in open(a.src / "meta/tasks.jsonl")]
    # старая задача -> новая по цвету в тексте
    new_tasks = {}
    task_map = {}
    for t in old_tasks:
        color = next(c for c in TASKS if c in t["task"])
        new_tasks.setdefault(TASKS[color], len(new_tasks))
        task_map[t["task_index"]] = new_tasks[TASKS[color]]
    if a.dst.exists():
        if not a.overwrite:
            raise SystemExit(f"уже есть {a.dst}; --overwrite")
        shutil.rmtree(a.dst)
    (a.dst / "meta").mkdir(parents=True)
    (a.dst / "data/chunk-000").mkdir(parents=True)
    n_src = a.limit or info["total_episodes"]
    jobs = [(a.src, a.dst, ep, fps, task_map) for ep in range(n_src)]
    results = {}
    with Pool(a.jobs, initializer=_fk_init) as pool:
        for k, (ep, res) in enumerate(pool.imap_unordered(convert_episode, jobs), 1):
            results[ep] = res
            if k % 50 == 0:
                print(f"  {k}/{n_src}", flush=True)
    # перенумеровать оставшиеся эпизоды подряд
    kept = [ep for ep in range(n_src) if results[ep] is not None]
    ep_lines, gidx, lengths = [], 0, []
    for new, ep in enumerate(kept):
        df, task_idx = results[ep]
        n = len(df)
        df["timestamp"] = np.arange(n, dtype=np.float32) / fps
        df["frame_index"] = np.arange(n)
        df["episode_index"] = new
        df["index"] = np.arange(gidx, gidx + n)
        df["task_index"] = task_idx
        df.to_parquet(a.dst / f"data/chunk-000/episode_{new:06d}.parquet")
        for cam in CAMS:
            d = a.dst / f"videos/chunk-000/observation.images.{cam}"
            if new != ep:
                (d / f"episode_{ep:06d}.mp4").rename(d / f"episode_{new:06d}.mp4")
        task_text = next(t for t, i in new_tasks.items() if i == task_idx)
        ep_lines.append({"episode_index": new, "tasks": [task_text], "length": n})
        lengths.append(n)
        gidx += n
    with open(a.dst / "meta/episodes.jsonl", "w") as fh:
        for l in ep_lines:
            fh.write(json.dumps(l, ensure_ascii=False) + "\n")
    with open(a.dst / "meta/tasks.jsonl", "w") as fh:
        for t, i in new_tasks.items():
            fh.write(json.dumps({"task_index": i, "task": t}, ensure_ascii=False) + "\n")
    info.update({"total_episodes": len(kept), "total_frames": gidx, "total_tasks": len(new_tasks),
                 "total_videos": len(kept) * len(CAMS), "splits": {"train": f"0:{len(kept)}"}})
    json.dump(info, open(a.dst / "meta/info.json", "w"), ensure_ascii=False, indent=2)
    shutil.copy(a.src / "meta/modality.json", a.dst / "meta/modality.json")
    print(f"готово: {a.dst}: {len(kept)} эпизодов из {n_src} (пропущено {n_src - len(kept)}), "
          f"{gidx} кадров, длина {np.min(lengths)}–{np.max(lengths)}, медиана {int(np.median(lengths))}")


if __name__ == "__main__":
    sys.exit(main())
