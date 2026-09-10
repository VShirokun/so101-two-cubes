"""Мост: политика GR00T в нашем MuJoCo-стенде. Протокол оценки — тот же,
что у eval_sim.py, судивший все остальные модели: те же раскладки (seed),
тот же критерий успеха (criteria.py), тот же лимит времени, тот же формат
отчёта. Иначе сравнение с ACT/SmolVLA было бы нечестным.

Отличия только в интерфейсе модели:
  - наблюдение: вложенный словарь video/state/language, картинки uint8 (B,T,H,W,3),
    состояние разбито на single_arm (5) + gripper (1), камеры зовутся front/wrist;
  - выход: чанк действий (B,T,D) в ФИЗИЧЕСКИХ единицах — пайплайн Gr00tPolicy сам
    декодирует относительные действия в абсолютные по текущей позе (проверено по
    штатному примеру gr00t/eval/real_robot/SO100/eval_so100.py, который шлёт этот
    выход прямо в моторы);
  - перепланирование: исполняем весь чанк (execution horizon = длине чанка из
    конфига обучения, у нас 16 шагов = 0.53 с), затем спрашиваем модель снова.

Запуск — из groot-venv, с LD_LIBRARY_PATH на ffmpeg7-shim (см. run_groot_2task.sh).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import mujoco
import numpy as np
import torch

sys.path.insert(0, "/srv/data/vshirokun/RoboOM/RoboOM/mlsim")
import criteria  # noqa: E402
from criteria import cube_in_bin, cube_xyz  # noqa: E402
from expert import CONTROL_HZ, CUBES, TASK_LANG, sample_layout, set_layout  # noqa: E402

SCENE = "/srv/data/vshirokun/RoboOM/RoboOM/mlsim/models/so101/pick_place.xml"
JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
IMG_H, IMG_W = 240, 320
# наши камеры -> имена, под которыми GR00T обучался (modality.json конвертера)
CAMS = {"top": "front", "wrist_cam": "wrist"}


def write_fail_reel(path: Path, clips, args) -> None:
    """Нарезка промахов одним файлом: перед каждым — титр с диагнозом.

    Правило Владимира (docs/visual-evidence.md): нарезка неуспешных попыток
    делается ВСЕГДА при оценке — по ней промахи разбираются глазами, а не
    по одной цифре успеха.
    """
    import subprocess
    import tempfile

    import imageio_ffmpeg
    from PIL import Image, ImageDraw, ImageFont

    big = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 26)
    mid = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
    small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 15)
    W, H = 480, 360
    path.parent.mkdir(parents=True, exist_ok=True)

    def diagnose(t):
        """Куда делся кубик — по координатам из протокола попытки."""
        z = t["target_xyz"][2]
        if t["wrong_cube"]:
            return "взят ЧУЖОЙ кубик"
        if t.get("ever_in_bin"):
            return "кубик побывал в коробке и выпал"
        if z > 0.05:
            return f"кубик поднят (z={z*100:.0f} см), но не донесён"
        return f"кубик так и не поднят (z={z*100:.1f} см)"

    out_frames = []
    for idx, (attempt, color, frames, t) in enumerate(clips, 1):
        # титр 1.2 с
        card = Image.new("RGB", (W, H), (14, 16, 20))
        d = ImageDraw.Draw(card)
        d.text((22, 96), f"промах {idx} из {len(clips)}", font=big, fill=(235, 235, 240))
        d.text((22, 140), f"попытка {attempt} · {'красный' if color == 'red' else 'зелёный'} кубик",
               font=mid, fill=(180, 185, 195))
        d.text((22, 172), diagnose(t), font=mid, fill=(224, 166, 72))
        # 'retries' — базовый протокол, 'grasp_attempts' — конечный автомат
        n_try = t.get("retries", t.get("grasp_attempts", 0))
        d.text((22, 208), f"заходов: {n_try} · длительность: {t['sim_steps']/30:.1f} с "
                          f"из {args.seconds:.0f} с", font=small, fill=(150, 156, 166))
        d.text((22, 236), f"кубик в начале: x={t['target_xyz'][0]:.2f} y={t['target_xyz'][1]:.2f}",
               font=small, fill=(150, 156, 166))
        out_frames += [np.asarray(card)] * 36

        for k, fr in enumerate(frames):
            im = Image.fromarray(fr)
            d = ImageDraw.Draw(im)
            d.rectangle([0, 0, W, 30], fill=(14, 16, 20))
            d.text((8, 5), f"промах {idx}/{len(clips)} · попытка {attempt} [{color}] · "
                           f"{k*3/30:.1f} с", font=small, fill=(235, 235, 240))
            out_frames.append(np.asarray(im))

    with tempfile.TemporaryDirectory() as tmp:
        for i, f in enumerate(out_frames):
            Image.fromarray(f).save(Path(tmp) / f"f{i:05d}.png")
        r = subprocess.run([
            imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-framerate", "30",
            "-i", str(Path(tmp) / "f%05d.png"), "-c:v", "libx264",
            "-pix_fmt", "yuv420p", "-crf", "24", str(path),
        ], capture_output=True)
    if r.returncode == 0:
        print(f"нарезка промахов: {path} ({len(clips)} промахов, "
              f"{len(out_frames)/30:.0f} с)", flush=True)
    else:
        print(r.stderr.decode()[-400:], file=sys.stderr)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True, help="каталог чекпоинта GR00T")
    ap.add_argument("--task", default="both", choices=["red", "green", "both"])
    ap.add_argument("--attempts", type=int, default=100)
    ap.add_argument("--seconds", type=float, default=12.0)
    ap.add_argument("--seed", type=int, default=777)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="/srv/data/vshirokun/roboom-work/reeval/groot")
    ap.add_argument("--img-h", type=int, default=IMG_H,
                    help="разрешение рендера наблюдений. ОБЯЗАНО совпадать с тем, на "
                         "котором обучалась политика: HD-модель, получая кадры 240x320, "
                         "давала 1-4%% вместо ~90%%")
    ap.add_argument("--img-w", type=int, default=IMG_W)
    ap.add_argument("--wrist-size", type=int, default=0,
                    help="квадратное разрешение камеры запястья (0 = как у всех); "
                         "ОБЯЗАНО совпадать с записью датасета")
    ap.add_argument("--random-yaw", action="store_true",
                    help="случайный поворот кубиков — как при записи датасета")
    ap.add_argument("--early-stop", action="store_true",
                    help="закончить попытку через 0.3 с после того, как названный "
                         "кубик оказался в коробке (ускорение оценки)")
    ap.add_argument("--exec-horizon", type=int, default=0,
                    help="исполнять только первые N шагов чанка (0 = весь чанк из 16). "
                         "Меньше N — чаще обратная связь во время спуска к кубику")
    ap.add_argument("--retry-jitter", type=float, default=0.0,
                    help="случайное отклонение (рад) домашней позы при откате: "
                         "неудачи детерминированы раскладкой, и повтор из той же "
                         "позы повторяет ту же ошибку; сдвиг старта декоррелирует")
    ap.add_argument("--retry", action="store_true",
                    help="супервизор повторных попыток: если за 3.2 с кубик так и не "
                         "поднялся, рука ФИЗИКОЙ (приводами, не телепортом) возвращается "
                         "в домашнюю позу и модель заходит снова. Сцена после отката — "
                         "в обучающем распределении: все эпизоды стартуют из дома.")
    ap.add_argument("--video-fails", type=int, default=0,
                    help="записать гифы первых N НЕУСПЕШНЫХ попыток (обзорная "
                         "камера show, ускорение x3) — правило visual-evidence")
    ap.add_argument("--video-dir", default="/srv/data/vshirokun/roboom-work/report-html/assets")
    ap.add_argument("--fail-reel", default=None,
                    help="путь к mp4: НАРЕЗКА всех снятых промахов подряд, с "
                         "титром перед каждым (номер попытки, задача, куда "
                         "делся кубик, сколько заходов). Правило Владимира: "
                         "нарезка промахов делается ВСЕГДА при оценке")
    args = ap.parse_args()

    from gr00t.policy import Gr00tPolicy
    policy = Gr00tPolicy(model_path=args.policy,
                         embodiment_tag="NEW_EMBODIMENT",
                         device=args.device)

    model = mujoco.MjModel.from_xml_path(SCENE)
    data = mujoco.MjData(model)
    renderers = {}
    for cam in CAMS:
        if cam == "wrist_cam" and args.wrist_size > 0:
            renderers[cam] = mujoco.Renderer(model, args.wrist_size, args.wrist_size)
        else:
            renderers[cam] = mujoco.Renderer(model, args.img_h, args.img_w)
    want_video = args.video_fails > 0 or args.fail_reel
    show_r = mujoco.Renderer(model, 360, 480) if want_video else None
    saved_fail_gifs = 0
    reel_clips = []          # (номер попытки, цвет, кадры, сведения) для нарезки
    bin_site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "bin_center")
    qadr = [model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j)]
            for j in JOINTS]
    sub = int(round(1.0 / (CONTROL_HZ * model.opt.timestep)))
    steps = int(args.seconds * CONTROL_HZ)

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    colors = ["red", "green"] if args.task == "both" else [args.task]

    def observe(task_text: str) -> dict:
        state = np.array([data.qpos[a] for a in qadr], dtype=np.float32)
        obs = {"video": {}, "state": {}, "language": {}}
        for cam, name in CAMS.items():
            renderers[cam].update_scene(data, camera=cam)
            # (B=1, T=1, H, W, 3) uint8 — так требует Gr00tPolicy
            obs["video"][name] = renderers[cam].render().copy()[None, None]
        obs["state"]["single_arm"] = state[:5][None, None]
        obs["state"]["gripper"] = state[5:6][None, None]
        obs["language"]["annotation.human.task_description"] = [[task_text]]
        return obs

    home_q = np.array(model.key_qpos[0][:len(qadr)], dtype=float)  # keyframe 0
    home_ctrl = np.concatenate([home_q[:5], [1.35]])                # схват раскрыт

    trials = []
    for i in range(args.attempts):
        color = colors[i % len(colors)]
        other = "green" if color == "red" else "red"
        task_text = CUBES[color]["task"]
        mujoco.mj_resetDataKeyframe(model, data, 0)
        set_layout(model, data, sample_layout(rng, random_yaw=args.random_yaw))
        mujoco.mj_forward(model, data)

        t0 = time.monotonic()
        vid_frames = []
        ever_target = ever_other = False
        done_steps = 0
        stable = 0
        finished_early = False
        retries = 0
        last_reset = 0
        lifted_since_reset = False
        while done_steps < steps:
            chunk, _info = policy.get_action(observe(task_text))
            # ключи бывают с префиксом action. и без — принимаем оба варианта
            arm = chunk.get("action.single_arm", chunk.get("single_arm"))
            grip = chunk.get("action.gripper", chunk.get("gripper"))
            horizon = arm.shape[1]
            if args.exec_horizon > 0:
                horizon = min(horizon, args.exec_horizon)
            for t in range(horizon):
                if done_steps >= steps:
                    break
                data.ctrl[:5] = np.asarray(arm[0][t], dtype=float)
                data.ctrl[5] = float(np.asarray(grip[0][t]).reshape(-1)[0])
                for _ in range(sub):
                    mujoco.mj_step(model, data)
                ever_target = ever_target or cube_in_bin(model, data, color, bin_site)
                ever_other = ever_other or cube_in_bin(model, data, other, bin_site)
                if cube_xyz(model, data, color)[2] > 0.025:
                    lifted_since_reset = True
                done_steps += 1
                if show_r is not None and done_steps % 3 == 0:
                    show_r.update_scene(data, camera="show")
                    vid_frames.append(show_r.render().copy())
                if args.early_stop:
                    stable = stable + 1 if cube_in_bin(model, data, color, bin_site) else 0
                    if stable >= 9:            # 0.3 с подряд в коробке
                        finished_early = True
                        break
            if finished_early:
                break
            # СУПЕРВИЗОР: 3.2 с прошло, кубик не поднимался -> откат приводами домой
            if (args.retry and not lifted_since_reset and not ever_target
                    and done_steps - last_reset >= int(3.2 * CONTROL_HZ)
                    and steps - done_steps > int(1.5 * CONTROL_HZ)):
                cur = np.concatenate([[data.qpos[a] for a in qadr[:5]], [1.35]])
                tgt = home_ctrl.copy()
                if args.retry_jitter > 0:
                    tgt[:5] = tgt[:5] + rng.uniform(-args.retry_jitter, args.retry_jitter, 5)
                back = max(2, int(0.8 * CONTROL_HZ))
                for k in range(1, back + 1):
                    a = 0.5 - 0.5 * np.cos(np.pi * k / back)
                    data.ctrl[:] = cur + a * (tgt - cur)
                    for _ in range(sub):
                        mujoco.mj_step(model, data)
                    done_steps += 1
                retries += 1
                last_reset = done_steps
                lifted_since_reset = False

        got_target = cube_in_bin(model, data, color, bin_site)
        got_other = cube_in_bin(model, data, other, bin_site)
        trials.append({
            "attempt": i + 1, "asked": color,
            "success": got_target and not got_other,
            "wrong_cube": got_other and not got_target,
            "both": got_other and got_target,
            "ever_in_bin": ever_target, "ever_in_bin_other": ever_other,
            "target_xyz": [round(v, 4) for v in cube_xyz(model, data, color)],
            "other_xyz": [round(v, 4) for v in cube_xyz(model, data, other)],
            "retries": retries,
            "finished_early": finished_early,
            "sim_steps": done_steps,
            "seconds": round(time.monotonic() - t0, 1),
        })
        t = trials[-1]
        mark = "✓" if t["success"] else ("✗ чужой" if t["wrong_cube"] else "✗")
        print(f"  попытка {i+1}/{args.attempts} [{color}] {mark}  {t['seconds']} с", flush=True)

        if args.fail_reel and not t["success"] and vid_frames:
            reel_clips.append((i + 1, color, vid_frames, t))

        if (show_r is not None and saved_fail_gifs < args.video_fails
                and not t["success"] and vid_frames):
            from PIL import Image, ImageDraw, ImageFont
            font = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
            imgs = []
            for k, fr in enumerate(vid_frames):
                im = Image.fromarray(fr)
                dr = ImageDraw.Draw(im)
                dr.rectangle([0, 0, 480, 26], fill=(15, 15, 18))
                dr.text((6, 3), f"промах: попытка {i+1} [{color}] "
                        f"кадр {k*3}/{t['sim_steps']}", font=font, fill=(255, 255, 255))
                imgs.append(im)
            saved_fail_gifs += 1
            name = Path(args.policy).name
            out_gif = Path(args.video_dir) / f"fail-{name}-a{i+1}.gif"
            imgs[0].save(out_gif, save_all=True, append_images=imgs[1:],
                         duration=100, loop=0)
            print(f"  гиф промаха: {out_gif}", flush=True)

    if args.fail_reel and reel_clips:
        write_fail_reel(Path(args.fail_reel), reel_clips, args)

    ok = sum(t["success"] for t in trials)
    wrong = sum(t["wrong_cube"] for t in trials)
    step_dir = next((p for p in Path(args.policy).resolve().parts if p.startswith("checkpoint-")), None)
    report = {
        "policy": args.policy, "type": "gr00t-n1.7",
        "attempts": args.attempts, "success": ok, "wrong_cube": wrong,
        "note": "успех = названный кубик в коробке И чужой не тронут; "
                "протокол идентичен eval_sim.py",
        "seed": args.seed, "seconds_limit": args.seconds, "task": args.task,
        "task_lang": TASK_LANG,
        "task_texts": {c: CUBES[c]["task"] for c in colors},
        "checkpoint_step": step_dir,
        "retry_supervisor": bool(args.retry),
        "retry_jitter": args.retry_jitter,
        "exec_horizon": args.exec_horizon or None,
        "early_stop": bool(args.early_stop),
        "img_size": [args.img_h, args.img_w],
        "wrist_size": args.wrist_size or None,
        "random_yaw": bool(args.random_yaw),
        "criteria": {"xy_max": criteria.XY_MAX, "z_max": criteria.Z_MAX},
        "trials": trials,
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tag = "retry_" if args.retry else ""
    name = f"eval_groot_{tag}{args.task}_{step_dir or 'ckpt'}_{args.attempts}_seed{args.seed}.json"
    (out / name).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nИТОГ: {ok}/{args.attempts} успешных, взят чужой кубик: {wrong}")
    print(f"отчёт: {out / name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
