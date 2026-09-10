#!/usr/bin/env python3
"""Минутный ролик для конкурса NVIDIA GTC Golden Ticket: монтаж из артефактов проекта.
Кадры 1280x720, 30 fps, титры внизу; выход docs/contest/two-cubes-reel.mp4."""
import subprocess, sys
from pathlib import Path
import cv2, numpy as np
from PIL import Image, ImageDraw, ImageFont
import imageio_ffmpeg

M = Path('/srv/data/vshirokun/Office-Rover/docs/article/media')
A = Path('/srv/data/vshirokun/Office-Rover/report-html/assets')
OUT = Path('/srv/data/vshirokun/Office-Rover/docs/contest/two-cubes-reel.mp4')
W, H, FPS = 1280, 720, 30
F_BIG = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 44)
F_MID = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 30)
F_SM = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 24)

def fit(img):
    h, w = img.shape[:2]; s = min(W / w, (H - 90) / h)
    im = cv2.resize(img, (int(w * s), int(h * s)))
    canvas = np.full((H, W, 3), 16, np.uint8)
    y0 = (H - 90 - im.shape[0]) // 2; x0 = (W - im.shape[1]) // 2
    canvas[y0:y0 + im.shape[0], x0:x0 + im.shape[1]] = im
    return canvas

def caption(frame, title, sub=None):
    im = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)); d = ImageDraw.Draw(im)
    d.rectangle([0, H - 90, W, H], fill=(16, 16, 18))
    d.text((28, H - 82), title, font=F_MID, fill=(240, 240, 240))
    if sub: d.text((28, H - 44), sub, font=F_SM, fill=(170, 175, 180))
    return cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR)

def card(lines, secs, big_first=True):
    im = Image.new('RGB', (W, H), (16, 16, 18)); d = ImageDraw.Draw(im)
    y = 200
    for i, l in enumerate(lines):
        f = F_BIG if (i == 0 and big_first) else F_MID
        d.text((80, y), l, font=f, fill=(240, 240, 240) if i == 0 else (180, 185, 190)); y += 70 if i == 0 else 48
    fr = cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR)
    return [fr] * int(secs * FPS)

def clip(path, start, secs, speed, title, sub, crop_top=0):
    cap = cv2.VideoCapture(str(path)); fps = cap.get(cv2.CAP_PROP_FPS) or 30
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(start * fps))
    need = int(secs * FPS); out = []; k = 0
    while len(out) < need:
        ok, fr = cap.read()
        if not ok: break
        k += 1
        if int(k * speed) % max(1, int(speed)) if speed >= 1 else False: pass
        if speed > 1 and (k % int(speed)) != 0: continue
        if crop_top: fr = fr[crop_top:]
        out.append(caption(fit(fr), title, sub))
        if fps < FPS:  # дублировать кадры до 30 fps
            for _ in range(int(round(FPS / fps)) - 1):
                if len(out) < need: out.append(out[-1])
    return out

def still(path, secs, title, sub):
    fr = caption(fit(cv2.imread(str(path))), title, sub)
    return [fr] * int(secs * FPS)

frames = []
frames += card(["Two cubes with markers", "A real SO-101 arm calibrates itself, records its own dataset,",
                "and learns to pick with NVIDIA Isaac GR00T N1.7 on one RTX 4090"], 4)
frames += still(M / '01-cubes-on-table.jpg', 3, "Two 3D-printed 28 mm cubes, ArUco on all six faces", "the object to grasp and the measuring instrument at once")
frames += clip(A / 'autocalib-real.mp4', 4, 8, 1, "Self-calibration from two cubes lying on the table", "48 poses, both cameras: 2.9 mm on held-out poses, joint corrections up to 14 %")
frames += clip(A / 'grasp-real-first.mp4', 22, 9, 2, "First autonomous grasp (2x)", "find with the top camera, grasp, lift, verify, put back")
frames += clip(A / 'real-pilot-preview.mp4', 12, 9, 1, "The robot records its own dataset (4x)", "200+ episodes, two cameras at 30 Hz, no teleoperation")
frames += clip(A / 'real-pilot-recolor-preview.mp4', 10, 7, 1, "Geometric recolouring: one recording, any cube colours", "markers replaced, fingers and shadows preserved")
frames += clip(A / 'demo-lift-carry-sim-10k.mp4', 1, 8, 1, "GR00T N1.7 fine-tuned on one RTX 4090: 86/100 in simulation", "policy lifts, a plain algorithm carries to the box", crop_top=32)
frames += clip(M / 'first-real-lift-by-policy.mp4', 0, 6.4, 1, "First real cube lifted by the fine-tuned GR00T", "top camera left, wrist camera right")
frames += still(A / 'real-color-cubes-lift-by-color.jpg', 3.5, "Real green and grey cubes, no markers", "wrist-camera colour servo; the policy trained on recoloured data + colour servo lifted the green one")
frames += card(["Everything is open source", "github.com/VShirokun/so101-two-cubes  (this project)",
                "github.com/VShirokun/gr00t-on-4090  (GR00T on one 24 GB GPU, datasets)", "#NVIDIAGTC"], 4)
print("frames", len(frames), "sec", len(frames) / FPS)
ff = imageio_ffmpeg.get_ffmpeg_exe()
p = subprocess.Popen([ff, '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{W}x{H}', '-r', str(FPS), '-i', '-',
                      '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '21', '-movflags', '+faststart', str(OUT)], stdin=subprocess.PIPE)
for fr in frames: p.stdin.write(np.ascontiguousarray(fr).tobytes())
p.stdin.close(); p.wait(); print(OUT, OUT.stat().st_size // 1024, 'KB')
