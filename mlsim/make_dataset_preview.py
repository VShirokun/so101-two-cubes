"""Ускоренное видео-превью датасета — обязательный артефакт после записи.

Правило проекта (docs/visual-evidence.md): после записи КАЖДОГО датасета
делается короткое ускоренное видео — не весь датасет, а равномерная выборка
эпизодов, ужатая до пары минут. Смотрится глазами до того, как потратить часы
GPU: перекошенные раскладки, брак рендера, неожиданно одинаковые ориентации
и прочие сюрпризы видно за минуту.

Кадры декодируются ИЗ САМОГО датасета (то, что реально увидит модель),
камеры кладутся рядом, сверху — номер эпизода и текст задачи.

    python3 make_dataset_preview.py --root <датасет> [--episodes 40] [--speed 4]
"""
import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import packaging.version  # noqa: F401
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from lerobot.datasets.lerobot_dataset import LeRobotDataset

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--episodes", type=int, default=40,
                    help="сколько эпизодов взять (равномерно по датасету)")
    ap.add_argument("--speed", type=int, default=4,
                    help="во сколько раз ускорить (берётся каждый N-й кадр)")
    ap.add_argument("--height", type=int, default=240, help="высота плитки камеры")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    root = Path(args.root)
    out = Path(args.out) if args.out else root / "preview.mp4"
    ds = LeRobotDataset("roboom/sim-cubes", root=str(root), video_backend="pyav")
    eps = pd.read_parquet(root / "meta/episodes")
    n = ds.num_episodes
    picks = sorted(set(np.linspace(0, n - 1, min(args.episodes, n)).astype(int)))
    cams = [k for k in ds[0] if k.startswith("observation.images.")]
    def _font(size):
        try:
            return ImageFont.truetype(FONT, size)
        except OSError:            # на Mac нет шрифта с 4090-машины
            return ImageFont.load_default()

    font, small = _font(18), _font(14)
    print(f"{root.name}: {n} эпизодов, берём {len(picks)}, камеры {cams}, ускорение x{args.speed}")

    frames = []
    for ep in picks:
        row = eps[eps.episode_index == ep]
        lo, hi = int(row.dataset_from_index.iloc[0]), int(row.dataset_to_index.iloc[0])
        task = ds[lo]["task"]
        for i in range(lo, hi, args.speed):
            s = ds[i]
            tiles = []
            for cam in cams:
                img = (s[cam].numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
                h, w = img.shape[:2]
                tiles.append(Image.fromarray(img).resize(
                    (int(w * args.height / h), args.height)))
            W = sum(t.width for t in tiles) + 6 * (len(tiles) - 1)
            canvas = Image.new("RGB", (W, args.height + 46), (18, 18, 20))
            x = 0
            for t in tiles:
                canvas.paste(t, (x, 46))
                x += t.width + 6
            d = ImageDraw.Draw(canvas)
            d.text((10, 4), task, font=font, fill=(255, 255, 255))
            d.text((10, 26), f"эпизод {ep + 1}/{n} · кадр {i - lo + 1}/{hi - lo}",
                   font=small, fill=(160, 160, 165))
            frames.append(np.asarray(canvas))

    print(f"кадров в превью: {len(frames)} (~{len(frames)/30:.0f} с при 30 fps)")
    import imageio_ffmpeg
    with tempfile.TemporaryDirectory() as tmp:
        for i, f in enumerate(frames):
            Image.fromarray(f).save(Path(tmp) / f"f{i:05d}.png")
        r = subprocess.run([
            imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-framerate", "30",
            "-i", str(Path(tmp) / "f%05d.png"), "-c:v", "libx264",
            "-pix_fmt", "yuv420p", "-crf", "23",
            "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", str(out),
        ], capture_output=True)
        if r.returncode != 0:
            print(r.stderr.decode()[-800:], file=sys.stderr)
            return 2
    print(f"записано: {out} ({out.stat().st_size // 1024} КБ)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
