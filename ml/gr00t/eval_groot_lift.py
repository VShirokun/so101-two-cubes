#!/usr/bin/env python3
"""Оценка политики GR00T на задаче «взять и ПОДНЯТЬ кубик» в MuJoCo-стенде.

Тот же мост к модели, что в eval_groot.py (наблюдение, чанк из 16 действий,
раскладки по seed, повёрнутые кубики), другой критерий успеха:

  успех = названный кубик держится выше LIFT_Z (центр кубика; на столе 1,4 см)
          HOLD_STEPS тактов подряд (0,5 с), а чужой кубик при этом не поднят;
          после этого попытка заканчивается (ранний конец).

Задача короче, чем «в коробку», поэтому лимит по умолчанию 12 с.
Нарезка промахов — правило проекта — та же (--fail-reel).

  eval_groot_lift.py --policy <чекпоинт> --attempts 100 --seed 777 --fail-reel <mp4>
"""
import argparse
import json
import sys
import time
from pathlib import Path

import mujoco
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "mlsim"))
from carry import CarryToBin  # noqa: E402
from criteria import cube_in_bin, cube_xyz  # noqa: E402
from expert import CONTROL_HZ, CUBES, sample_layout, set_layout  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_groot import write_fail_reel  # noqa: E402

SCENE = str(ROOT / "mlsim/models/so101/pick_place.xml")
JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
CAMS = {"top": "front", "wrist_cam": "wrist"}
TASKS = {"red": "Подними красный кубик.", "green": "Подними зелёный кубик."}
LIFT_Z = 0.05          # центр кубика выше 5 см (на столе 1,4 см; эксперт поднимает до ~9 см)
OTHER_Z = 0.03         # чужой кубик считается тронутым, если поднят выше
HOLD_STEPS = 15        # 0,5 с подряд


def write_demo(path, clips, carry):
    """Ролик успешных попыток: титр + кадры камеры show (x3)."""
    import subprocess
    import tempfile

    import imageio_ffmpeg
    from PIL import Image, ImageDraw, ImageFont

    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = []
    for idx, (attempt, color, vid) in enumerate(clips, 1):
        for k, fr in enumerate(vid):
            im = Image.fromarray(fr)
            d = ImageDraw.Draw(im)
            d.rectangle([0, 0, 480, 30], fill=(14, 16, 20))
            d.text((8, 5), f"{idx}/{len(clips)} · попытка {attempt} [{color}] · политика поднимает"
                           f"{', алгоритм несёт в коробку' if carry else ''} · {k*3/30:.1f} с",
                   font=font, fill=(235, 235, 240))
            frames.append(np.asarray(im))
    with tempfile.TemporaryDirectory() as tmp:
        for i, f in enumerate(frames):
            Image.fromarray(f).save(Path(tmp) / f"f{i:05d}.png")
        r = subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-framerate", "30",
                            "-i", str(Path(tmp) / "f%05d.png"), "-c:v", "libx264",
                            "-pix_fmt", "yuv420p", "-crf", "24", str(path)], capture_output=True)
    print(f"демо: {path} ({len(clips)} попыток, {len(frames)/30:.0f} с)" if r.returncode == 0
          else r.stderr.decode()[-400:], flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True)
    ap.add_argument("--task", default="both", choices=["red", "green", "both"])
    ap.add_argument("--attempts", type=int, default=100)
    ap.add_argument("--seconds", type=float, default=12.0)
    ap.add_argument("--seed", type=int, default=777)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=str(ROOT / "ml/gr00t/reeval/lift"))
    ap.add_argument("--img-h", type=int, default=480)
    ap.add_argument("--img-w", type=int, default=640)
    ap.add_argument("--random-yaw", action="store_true", default=True)
    ap.add_argument("--fail-reel", default=None)
    ap.add_argument("--demo", default=None, help="mp4: первые --demo-clips УСПЕШНЫХ попыток подряд (для витрины)")
    ap.add_argument("--demo-clips", type=int, default=6)
    ap.add_argument("--carry", action="store_true",
                    help="после подъёма политикой кубик несёт в коробку алгоритм (mlsim/carry.py); "
                         "успех = кубик в коробке по criteria.cube_in_bin")
    args = ap.parse_args()

    from gr00t.policy import Gr00tPolicy
    policy = Gr00tPolicy(model_path=args.policy, embodiment_tag="NEW_EMBODIMENT", device=args.device)

    model = mujoco.MjModel.from_xml_path(SCENE)
    data = mujoco.MjData(model)
    renderers = {cam: mujoco.Renderer(model, args.img_h, args.img_w) for cam in CAMS}
    show_r = mujoco.Renderer(model, 360, 480) if (args.fail_reel or args.demo) else None
    carry = CarryToBin(model, data) if args.carry else None
    qadr = [model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j)] for j in JOINTS]
    sub = int(round(1.0 / (CONTROL_HZ * model.opt.timestep)))
    steps = int(args.seconds * CONTROL_HZ)
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    colors = ["red", "green"] if args.task == "both" else [args.task]

    def observe(task_text):
        state = np.array([data.qpos[a] for a in qadr], dtype=np.float32)
        obs = {"video": {}, "state": {}, "language": {}}
        for cam, name in CAMS.items():
            renderers[cam].update_scene(data, camera=cam)
            obs["video"][name] = renderers[cam].render().copy()[None, None]
        obs["state"]["single_arm"] = state[:5][None, None]
        obs["state"]["gripper"] = state[5:6][None, None]
        obs["language"]["annotation.human.task_description"] = [[task_text]]
        return obs

    trials, reel_clips, demo_clips = [], [], []
    for i in range(args.attempts):
        color = colors[i % len(colors)]
        other = "green" if color == "red" else "red"
        mujoco.mj_resetDataKeyframe(model, data, 0)
        set_layout(model, data, sample_layout(rng, random_yaw=args.random_yaw))
        mujoco.mj_forward(model, data)
        t0 = time.monotonic()
        vid, done, hold, success, other_lifted, zmax = [], 0, 0, False, False, 0.0
        while done < steps and not success:
            chunk, _ = policy.get_action(observe(TASKS[color]))
            arm = chunk.get("action.single_arm", chunk.get("single_arm"))
            grip = chunk.get("action.gripper", chunk.get("gripper"))
            for t in range(arm.shape[1]):
                if done >= steps:
                    break
                data.ctrl[:5] = np.asarray(arm[0][t], dtype=float)
                data.ctrl[5] = float(np.asarray(grip[0][t]).reshape(-1)[0])
                for _ in range(sub):
                    mujoco.mj_step(model, data)
                done += 1
                z = cube_xyz(model, data, color)[2]
                zmax = max(zmax, z)
                other_lifted = other_lifted or cube_xyz(model, data, other)[2] > OTHER_Z
                hold = hold + 1 if z > LIFT_Z else 0
                if show_r is not None and done % 3 == 0:
                    show_r.update_scene(data, camera="show")
                    vid.append(show_r.render().copy())
                if hold >= HOLD_STEPS and not other_lifted:
                    success = True
                    break
        in_bin = None
        if carry is not None and success:
            def _rec(k):
                if show_r is not None and k % 3 == 0:
                    show_r.update_scene(data, camera="show")
                    vid.append(show_r.render().copy())
            n_carry = carry.run(on_step=_rec)
            for _ in range(int(0.5 * CONTROL_HZ) * sub):
                mujoco.mj_step(model, data)
            in_bin = bool(cube_in_bin(model, data, color, carry.bin_site)) and n_carry > 0
            done += n_carry
        lifted = bool(success)
        if carry is not None:
            success = lifted and bool(in_bin)
        t = {"attempt": i + 1, "asked": color, "success": bool(success), "lifted": lifted, "in_bin": in_bin,
             "wrong_cube": bool(other_lifted and not success), "both": False, "ever_in_bin": False,
             "target_xyz": [round(v, 4) for v in cube_xyz(model, data, color)],
             "other_xyz": [round(v, 4) for v in cube_xyz(model, data, other)],
             "z_max": round(float(zmax), 4), "retries": 0, "finished_early": bool(success),
             "sim_steps": done, "seconds": round(time.monotonic() - t0, 1)}
        trials.append(t)
        print(f"  попытка {i+1}/{args.attempts} [{color}] {'✓' if success else '✗'}  "
              f"z_max={zmax*100:.1f} см{'' if in_bin is None else ('  в коробке' if in_bin else '  НЕ в коробке')}  {t['seconds']} с", flush=True)
        if args.fail_reel and not success and vid:
            reel_clips.append((i + 1, color, vid, t))
        if args.demo and success and vid and len(demo_clips) < args.demo_clips:
            demo_clips.append((i + 1, color, vid))

    if args.fail_reel and reel_clips:
        write_fail_reel(Path(args.fail_reel), reel_clips, args)
    if args.demo and demo_clips:
        write_demo(Path(args.demo), demo_clips, args.carry)
    ok = sum(t["success"] for t in trials)
    n_lift = sum(t["lifted"] for t in trials)
    wrong = sum(t["wrong_cube"] for t in trials)
    step_dir = next((p for p in Path(args.policy).resolve().parts if p.startswith("checkpoint-")), None)
    report = {"policy": args.policy, "type": "gr00t-n1.7", "task_kind": "lift",
              "attempts": args.attempts, "success": ok, "lifted": n_lift, "wrong_cube": wrong,
              "carry": bool(args.carry), "note": f"успех = названный кубик выше {LIFT_Z} м {HOLD_STEPS} тактов подряд, чужой не поднят",
              "seed": args.seed, "seconds_limit": args.seconds, "task": args.task,
              "task_texts": {c: TASKS[c] for c in colors}, "checkpoint_step": step_dir,
              "img_size": [args.img_h, args.img_w], "random_yaw": bool(args.random_yaw),
              "criteria": {"lift_z": LIFT_Z, "hold_steps": HOLD_STEPS, "other_z": OTHER_Z},
              "trials": trials}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    name = f"eval_lift{'-carry' if args.carry else ''}_{args.task}_{step_dir or 'ckpt'}_{args.attempts}_seed{args.seed}.json"
    (out / name).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nИТОГ: {ok}/{args.attempts} успешных (поднято политикой {n_lift}), поднят чужой кубик: {wrong}")
    print(f"отчёт: {out / name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
