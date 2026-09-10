#!/usr/bin/env python3
"""Сервер веб-камеры: кадры выбранной ПО ИМЕНИ камеры -> /tmp/roboom_cam/latest.jpg.

Захват напрямую через AVFoundation (pyobjc) с ЯВНЫМ выбором формата
устройства. Причина: у C920 несжатый канал (yuvs) упирается в полосу USB
(720p максимум 10 fps, 1080p — 5), честные 30 fps камера отдаёт только по
сжатому каналу '420v' — а ffmpeg-демюксер avfoundation формат устройства не
переключает и сидел на 10 fps. Здесь берётся лучший 420v-формат с 30 fps
(1920x1080), AVFoundation сам декодирует в BGRA.

Камеру macOS даёт процессам Terminal.app — запускать оттуда
(start_cam.command). Остановка: Ctrl+C или `touch /tmp/roboom_cam/stop`.
"""

import argparse
import sys
import os
import time
from pathlib import Path

import cv2
import numpy as np
import objc
import AVFoundation as AV
import CoreMedia as CM
import Quartz.CoreVideo as CV
from Foundation import NSObject, NSDate, NSRunLoop
from libdispatch import dispatch_queue_create

kCVPixelFormatType_32BGRA = 0x42475241  # 'BGRA'


def pick_device(sub):
    devs = AV.AVCaptureDevice.devicesWithMediaType_(AV.AVMediaTypeVideo)
    names = [str(d.localizedName()) for d in devs]
    hits = [d for d in devs if sub.lower() in str(d.localizedName()).lower()]
    if not hits:
        raise SystemExit(f"камера '{sub}' не найдена; есть: {names}")
    print(f"камеры: {names}\nвыбрана: {hits[0].localizedName()}")
    return hits[0]


def format_candidates(dev, want_w, min_fps):
    """Кандидаты по убыванию желанности: сжатый канал 420v от want_w вниз,
    затем несжатые. Дешёвые модули объявляют форматы, которых не отдают
    (wrist-камера "объявляла" 1080p30 и молчала) — каждый кандидат
    проверяется живыми кадрами в main."""
    rows = []
    for f in dev.formats():
        desc = f.formatDescription()
        dims = CM.CMVideoFormatDescriptionGetDimensions(desc)
        code = CM.CMFormatDescriptionGetMediaSubType(desc)
        fourcc = "".join(chr((code >> sh) & 0xFF) for sh in (24, 16, 8, 0))
        fps = max(float(r.maxFrameRate())
                  for r in f.videoSupportedFrameRateRanges())
        if fps < min_fps or dims.width > want_w:
            continue
        rows.append((0 if fourcc == "420v" else 1, -dims.width,
                     (f, dims.width, dims.height, fourcc, fps)))
    if not rows:
        raise SystemExit("нет форматов с нужными параметрами")
    rows.sort(key=lambda r: (r[0], r[1]))
    return [r[2] for r in rows]


class Grabber(NSObject):
    def initWithDir_fpsOut_(self, out_dir, fps_out):
        self = objc.super(Grabber, self).init()
        self.out = Path(out_dir)
        self.period = 1.0 / fps_out
        self.last_save = 0.0
        self.got = 0
        self.kept = 0
        self.t_rep = time.time()
        self.last_frame_ts = time.time()
        self.last_fp = None
        self.same = 0
        return self

    def captureOutput_didOutputSampleBuffer_fromConnection_(self, output, sbuf,
                                                            conn):
        img = CM.CMSampleBufferGetImageBuffer(sbuf)
        if img is None:
            return
        self.got += 1
        now = time.time()
        self.last_frame_ts = now
        if now - self.t_rep >= 3.0:
            print(f"захват: {self.got / (now - self.t_rep):.1f} fps, "
                  f"запись: {self.kept / (now - self.t_rep):.1f} fps")
            self.got = 0
            self.kept = 0
            self.t_rep = now
        if now - self.last_save < self.period:
            return
        self.last_save = now
        CV.CVPixelBufferLockBaseAddress(img, 1)
        try:
            h = CV.CVPixelBufferGetHeight(img)
            w = CV.CVPixelBufferGetWidth(img)
            stride = CV.CVPixelBufferGetBytesPerRow(img)
            base = CV.CVPixelBufferGetBaseAddress(img)
            buf = base.as_buffer(stride * h)
            arr = np.frombuffer(buf, np.uint8).reshape(h, stride // 4, 4)
            frame = arr[:, :w, :3]          # BGRA -> BGR
            fp = bytes(frame[::89, ::89].tobytes())
            self.same = self.same + 1 if fp == self.last_fp else 0
            self.last_fp = fp
            tmp = self.out / ".latest.tmp.jpg"
            cv2.imwrite(str(tmp), np.ascontiguousarray(frame),
                        [cv2.IMWRITE_JPEG_QUALITY, 85])
            os.replace(tmp, self.out / "latest.jpg")
            self.kept += 1
        finally:
            CV.CVPixelBufferUnlockBaseAddress(img, 1)


def run_once(args, out, stop):
    dev = resolve_retry(args.name, stop)
    candidates = format_candidates(dev, args.width, args.min_fps)

    session = AV.AVCaptureSession.alloc().init()
    inp, err = AV.AVCaptureDeviceInput.deviceInputWithDevice_error_(dev, None)
    if inp is None:
        raise SystemExit(f"вход не создан: {err}")
    session.addInput_(inp)
    outp = AV.AVCaptureVideoDataOutput.alloc().init()
    outp.setVideoSettings_({"PixelFormatType": kCVPixelFormatType_32BGRA})
    outp.setAlwaysDiscardsLateVideoFrames_(True)
    grab = Grabber.alloc().initWithDir_fpsOut_(str(out), args.fps_out)
    outp.setSampleBufferDelegate_queue_(grab,
                                        dispatch_queue_create(b"cam", None))
    session.addOutput_(outp)

    session.startRunning()
    # на macOS startRunning сбрасывает формат устройства к preset'у сессии —
    # формат ставится ПОСЛЕ старта; каждый кандидат подтверждается живыми
    # кадрами, потому что дешёвые модули объявляют то, чего не отдают
    started = False
    for fmt, fw, fh, fcc, ffps in candidates:
        ok, lockerr = dev.lockForConfiguration_(None)
        if not ok:
            raise SystemExit(f"lockForConfiguration не удался: {lockerr}")
        dev.setActiveFormat_(fmt)
        # у формата свои точные длительности кадра (1000000/30000030 и т.п.)
        rng = max(fmt.videoSupportedFrameRateRanges(),
                  key=lambda r: float(r.maxFrameRate()))
        dev.setActiveVideoMinFrameDuration_(rng.minFrameDuration())
        dev.setActiveVideoMaxFrameDuration_(rng.minFrameDuration())
        dev.unlockForConfiguration()
        base = grab.got
        deadline = time.time() + 4.0
        rl = NSRunLoop.currentRunLoop()
        while time.time() < deadline and grab.got - base < 6:
            rl.runUntilDate_(NSDate.dateWithTimeIntervalSinceNow_(0.2))
        if grab.got - base >= 6:
            print(f"формат работает: {fw}x{fh} {fcc} @{ffps:.0f} fps")
            started = True
            break
        print(f"формат {fw}x{fh} {fcc} @{ffps:.0f} молчит — следующий")
    if not started:
        session.stopRunning()
        return False
    print("захват запущен")
    loop = NSRunLoop.currentRunLoop()
    grab.last_frame_ts = time.time()
    healthy = True
    while not stop.exists():
        loop.runUntilDate_(NSDate.dateWithTimeIntervalSinceNow_(0.25))
        # watchdog: сессия молча встаёт (кадры перестают приходить или
        # приходят идентичные) — пересоздать захват целиком
        if time.time() - grab.last_frame_ts > 5.0:
            print("кадры перестали приходить — пересоздаю захват")
            healthy = False
            break
        if grab.same > 45:
            print("кадры идентичны (поток застыл) — пересоздаю захват")
            healthy = False
            break
    session.stopRunning()
    if healthy:
        print("остановлен")
    return healthy


def resolve_retry(sub, stop, attempts=15):
    """Ждать появления камеры; после N неудач перезапустить ПРОЦЕСС:
    AVFoundation в долгоживущем процессе кеширует список устройств и не
    видит переткнутую камеру (ловилось с C920 после передёргивания USB)."""
    import os
    n = 0
    while not stop.exists():
        try:
            return pick_device(sub)
        except SystemExit as e:
            print(e)
            n += 1
            if n >= attempts:
                print("список устройств устарел — перезапускаю процесс")
                os.execv(sys.executable, [sys.executable] + sys.argv)
            time.sleep(1.5)
    raise SystemExit("остановлен")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="C920")
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--min-fps", type=float, default=30.0)
    ap.add_argument("--fps-out", type=float, default=30.0)
    ap.add_argument("--dir", default="/tmp/roboom_cam")
    args = ap.parse_args()
    out = Path(args.dir)
    out.mkdir(parents=True, exist_ok=True)
    stop = out / "stop"
    stop.unlink(missing_ok=True)
    while not stop.exists():
        try:
            done = run_once(args, out, stop)
            if done:
                break
        except SystemExit as e:
            print(e)
        time.sleep(1.5)


if __name__ == "__main__":
    main()
