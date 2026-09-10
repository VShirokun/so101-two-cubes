#!/usr/bin/env python3
"""Сервер веб-камеры для Linux (V4L2): кадры выбранной ПО ИМЕНИ камеры ->
<dir>/latest.jpg. Тот же контракт, что у cam_server.py (macOS): атомарная
подмена latest.jpg, --fps-out, стоп-файл <dir>/stop, вотчдог с пересозданием
захвата, ожидание камеры после отвала по USB.

Камера ищется в /dev/v4l/by-id/*<name>*-video-index0 (C920 -> "C920",
камера кисти Sonix -> "USB2.0"). Формат — MJPG: у C920 несжатый канал
упирается в полосу USB, честные 30 fps только по MJPG (проверено 09.09.2026:
1920x1080 MJPG 30.0 fps, 640x480 MJPG 30.5 fps).

  cam_server_linux.py                                   # C920 1920x1080 -> /tmp/roboom_cam
  cam_server_linux.py --name USB2.0 --width 640 --height 480 --dir /tmp/roboom_wrist
"""

import argparse
import glob
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np


def find_device(sub):
    hits = sorted(p for p in glob.glob("/dev/v4l/by-id/*-video-index0")
                  if sub.lower() in os.path.basename(p).lower())
    names = [os.path.basename(p) for p in glob.glob("/dev/v4l/by-id/*-video-index0")]
    if not hits:
        raise RuntimeError(f"камера '{sub}' не найдена; есть: {names}")
    dev = os.path.realpath(hits[0])
    print(f"камеры: {names}\nвыбрана: {os.path.basename(hits[0])} -> {dev}", flush=True)
    return dev


def open_capture(dev, w, h, fps):
    cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
    if not cap.isOpened():
        raise RuntimeError(f"{dev}: не открывается")
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    cap.set(cv2.CAP_PROP_FPS, fps)
    # прогрев и проверка живыми кадрами (дешёвые модули объявляют то, чего не отдают)
    got = 0
    t0 = time.time()
    while time.time() - t0 < 4.0 and got < 6:
        ok, _ = cap.read()
        got += int(ok)
    if got < 6:
        cap.release()
        raise RuntimeError(f"{dev}: формат {w}x{h} MJPG молчит")
    aw, ah = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"формат работает: {aw}x{ah} MJPG @{cap.get(cv2.CAP_PROP_FPS):.0f} fps", flush=True)
    return cap


def run_passthrough(args, dev, out, stop):
    """MJPG-пакеты камеры пишутся в latest.jpg как есть (PyAV/v4l2): нулевая
    нагрузка на CPU. Без этого декодирование+кодирование 1080p в OpenCV под
    нагрузкой машины (обучение GR00T) проседало до 15 fps (09.09.2026).
    -> True, если поток шёл и остановлен штатно; False — пересоздать; None —
    пакеты не JPEG, нужен режим декодирования."""
    import av
    cont = av.open(dev, format="v4l2", options={"input_format": "mjpeg",
                                                 "video_size": f"{args.width}x{args.height}",
                                                 "framerate": str(int(args.fps))})
    st = cont.streams.video[0]
    period = 1.0 / args.fps_out
    tmp, latest = out / ".latest.tmp.jpg", out / "latest.jpg"
    got = kept = same = 0
    next_due = 0.0
    last_fp = None
    t_rep = last_frame = time.time()
    healthy = True
    checked = False
    try:
        for pkt in cont.demux(st):
            if stop.exists():
                break
            if pkt.size == 0:
                continue
            data = bytes(pkt)
            now = time.time()
            if not checked:
                if cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR) is None:
                    print("пакеты камеры не читаются как JPEG — режим декодирования", flush=True)
                    return None
                print(f"формат работает: {st.width}x{st.height} MJPG passthrough "
                      f"@{float(st.average_rate):.0f} fps\nзахват запущен", flush=True)
                checked = True
            got += 1
            last_frame = now
            if now - t_rep >= 3.0:
                print(f"захват: {got / (now - t_rep):.1f} fps, "
                      f"запись: {kept / (now - t_rep):.1f} fps", flush=True)
                got = kept = 0
                t_rep = now
            if now < next_due:
                continue
            next_due = max(next_due + period, now - period)
            fp = data[::997]
            same = same + 1 if fp == last_fp else 0
            last_fp = fp
            if same > 45:
                print("кадры идентичны (поток застыл) — пересоздаю захват", flush=True)
                healthy = False
                break
            tmp.write_bytes(data)
            os.replace(tmp, latest)
            kept += 1
    finally:
        cont.close()
    if healthy:
        print("остановлен", flush=True)
    return healthy


def run_once(args, out, stop):
    dev = find_device(args.name)
    if not args.decode:
        r = run_passthrough(args, dev, out, stop)
        if r is not None:
            return r
    cap = open_capture(dev, args.width, args.height, args.fps)
    period = 1.0 / args.fps_out
    tmp, latest = out / ".latest.tmp.jpg", out / "latest.jpg"
    got = kept = same = 0
    next_due = 0.0          # план записи по сетке, без потери каждого третьего кадра из-за джиттера
    last_fp = None
    t_rep = last_frame = time.time()
    healthy = True
    print("захват запущен", flush=True)
    try:
        while not stop.exists():
            ok, frame = cap.read()
            now = time.time()
            if not ok or frame is None:
                if now - last_frame > 5.0:
                    print("кадры перестали приходить — пересоздаю захват", flush=True)
                    healthy = False
                    break
                time.sleep(0.01)
                continue
            got += 1
            last_frame = now
            if now - t_rep >= 3.0:
                print(f"захват: {got / (now - t_rep):.1f} fps, "
                      f"запись: {kept / (now - t_rep):.1f} fps", flush=True)
                got = kept = 0
                t_rep = now
            if now < next_due:
                continue
            next_due = max(next_due + period, now - period)
            fp = frame[::89, ::89].tobytes()
            same = same + 1 if fp == last_fp else 0
            last_fp = fp
            if same > 45:
                print("кадры идентичны (поток застыл) — пересоздаю захват", flush=True)
                healthy = False
                break
            cv2.imwrite(str(tmp), frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            os.replace(tmp, latest)
            kept += 1
    finally:
        cap.release()
    if healthy:
        print("остановлен", flush=True)
    return healthy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="C920")
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--fps-out", type=float, default=30.0)
    ap.add_argument("--dir", default="/tmp/roboom_cam")
    ap.add_argument("--decode", action="store_true", help="старый путь: OpenCV декодирует и кодирует JPEG")
    args = ap.parse_args()
    out = Path(args.dir)
    out.mkdir(parents=True, exist_ok=True)
    stop = out / "stop"
    stop.unlink(missing_ok=True)
    while not stop.exists():
        try:
            if run_once(args, out, stop):
                break
        except (RuntimeError, OSError) as e:      # OSError: камера отвалилась по USB (av.error.OSError)
            print(f"захват прерван: {e} — жду камеру", flush=True)
        time.sleep(1.5)


if __name__ == "__main__":
    main()
