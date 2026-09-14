#!/usr/bin/env python3
"""Веб-телеоперация SO-101 с телефона: обе камеры на экране, схват и рука.

Сервер aiohttp на 0.0.0.0:8080. Страница под телефон (вертикально):
  - верхняя камера и камера кисти как MJPEG-потоки (/cam/top.mjpg, /cam/wrist.mjpg,
    кадры из /tmp/roboom_cam|roboom_wrist/latest.jpg серверов камер, верхняя
    уменьшается до 640 px);
  - джойстик в осях КАДРА КАМЕРЫ КИСТИ: вверх на экране = вглубь кадра, вправо =
    вправо в кадре, независимо от поворота губок (оси считаются по FK и
    калибровке камеры кисти каждый такт); кнопки вверх/вниз (высота), поворот
    губок, ползунок схвата 0–100 %; рамка кадра кисти краснеет с той стороны,
    где стена зоны, тем сильнее, чем ближе (плавно, по направлению);
    кнопки «Парковка» и «СТОП»;
  - телеметрия: положение точки схвата в базе, высота, сообщения охраны.

Управление: WebSocket /ws, телефон шлёт {"vx","vy","vz","wyaw","grip"} ~20 Гц;
сервер интегрирует целевую точку схвата (м, рад), решает IK (подход сверху,
азимут губок = yaw), проверяет пол, досягаемость и самостолкновения драйвера
и пишет цель приводам на 30 Гц с ограничением шага. Нет команд 0,5 с —
рука стоит (deadman). Стоп-файл /tmp/roboom_arm/stop — как везде.

  web_teleop.py [--port 8080] [--speed 0.06] [--dry]
"""
import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from aiohttp import web

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "arm"))
import autocalib_real as A  # noqa: E402
from arm_driver import ArmDriver, GuardError, HZ, NAMES, STOP, TIP_BELOW_TCP, Z_FLOOR  # noqa: E402
from collect_pilot import plan  # noqa: E402
from ik import ArmIK, TCP_OFFSET  # noqa: E402

CAMS = {"top": Path("/tmp/roboom_cam/latest.jpg"), "wrist": Path("/tmp/roboom_wrist/latest.jpg")}
GRIP_MIN, GRIP_MAX = A.GRIP_CLOSED, A.GRIP_WIDE
Z_MIN, Z_MAX = 0.012, 0.10            # точка схвата над столом: чуть ниже центра кубика .. 10 см (предел подхода сверху)
HOME = [0.20, 0.0, 0.08]              # стартовая точка телеоперации (подход сверху достижим)
R_MIN, R_MAX = 0.12, 0.30
HTML = (_HERE / "web_teleop.html").read_text(encoding="utf-8")
XR_HTML = (_HERE / "xr.html").read_text(encoding="utf-8")
TILT_HTML = (_HERE / "tilt.html").read_text(encoding="utf-8")
KEY = ""      # ключ доступа (--key): страница и сокет отвечают только с ?k=KEY — для публичного туннеля


def authorized(request):
    return not KEY or request.query.get("k") == KEY or request.cookies.get("k") == KEY


class Recorder:
    """Демонстрации оператора в формате сборщика (real/data/teleop/<сессия>/ep_NNNN):
    обе камеры как есть + steps.jsonl (state/action в радианах модели) на каждом
    такте 30 Гц. Подходит в pilot_to_lift_v21.py без изменений."""

    def __init__(self, root):
        self.root = Path(root)
        self.ep = None
        self.n = 0
        self.t0 = 0.0
        self.task = ""

    def start(self, task):
        self.root.mkdir(parents=True, exist_ok=True)
        idx = max((int(p.name[3:]) for p in self.root.glob("ep_*")), default=-1) + 1
        self.ep = self.root / f"ep_{idx:04d}"
        (self.ep / "top").mkdir(parents=True)
        (self.ep / "wrist").mkdir()
        self.f = open(self.ep / "steps.jsonl", "w")
        self.n, self.t0, self.task = 0, time.time(), task
        return self.ep.name

    def tick(self, d, q_goal, q_read, raw, phase):
        if self.ep is None:
            return
        rec = {"i": self.n, "t": time.time(), "phase": phase,
               "state": [*d.q_model(q_read).tolist(), float(q_read[5])],
               "action": [*d.q_model(q_goal).tolist(), float(q_goal[5])],
               "raw": [int(raw[k]) for k in range(1, 7)]}
        for cam, path in CAMS.items():
            try:
                rec[cam + "_ts"] = path.stat().st_mtime
                (self.ep / cam / f"{self.n:06d}.jpg").write_bytes(path.read_bytes())
            except OSError:
                pass
        self.f.write(json.dumps(rec) + "\n")
        self.n += 1

    def stop(self, success=None):
        if self.ep is None:
            return None
        self.f.close()
        meta = {"episode": int(self.ep.name[3:]), "source": "teleop", "task_hint": self.task, "t0": self.t0,
                "t1": time.time(), "frames": self.n, "success": success, "held": success, "cube": None}
        (self.ep / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))
        name, self.ep = self.ep.name, None
        return name


class Teleop:
    def __init__(self, dry, speed, mirror=False):
        self.dry, self.speed, self.mirror = dry, speed, mirror
        self.Xw = np.array(json.loads(A.OUT.read_text())["T_gripper_wristcam"])
        self.ik = ArmIK(A._MODEL)
        self.d = ArmDriver()
        self.d.torque(True)
        self.d.hold()
        q, _ = self.d.read()
        self.q_cmd = q.copy()
        F = self.d.fk_T(q[:5])
        tcp = F[:3, 3] + F[:3, :3] @ TCP_OFFSET
        self.target = np.array([tcp[0], tcp[1], max(tcp[2], Z_MIN)])
        self.yaw = 0.0
        self.pitch = 0.0          # наклон оси подхода от базы наружу, рад (0 = строго сверху)
        self.grip = float(q[5])
        self.cmd = {"vx": 0.0, "vy": 0.0, "vz": 0.0, "wyaw": 0.0, "wpitch": 0.0, "grip": None}
        self.controller = None    # id сокета, который управляет; остальные — зрители
        self.controller_seen = 0.0
        self.tick_times = []      # для измерения реальной частоты такта
        self.rec = Recorder(A._ROOT / "real/data/teleop" / time.strftime("%Y%m%d"))
        self.phase = "idle"
        self.xr = None            # {"dx","dy","dz","dyaw"} в осях робота, когда телефон в режиме «следовать»
        self.xr_anchor = None     # (target, yaw) на момент включения «следовать»
        self.xr_scale = 1.0
        self.last_cmd = 0.0
        self.msg = ""
        self.q_read = q
        self.limits = {"x": False, "y": False, "z": False, "r": False, "ik": False}
        self.zmin = Z_FLOOR["grasp"] + TIP_BELOW_TCP
        self.home()

    def _sync_from_arm(self):
        q, _ = self.d.read()
        self.q_read, self.q_cmd = q, q.copy()
        F = self.d.fk_T(q[:5])
        self.target = F[:3, 3] + F[:3, :3] @ TCP_OFFSET
        self.grip = float(q[5])

    def home(self):
        """В стартовую точку телеоперации: подход сверху, губки раскрыты."""
        self.msg = "домой"
        try:
            q = plan(self.ik, HOME, self.d.q_model(self.d.read()[0]), 0.0)
            if q is None:
                raise GuardError("IK не встал для домашней точки")
            self.d.goto(np.concatenate([self.d.q_cmd(q), [A.GRIP_WIDE]]), 3.0, mode="grasp")
            self.d.settle(1.0, 0.5)
            self.msg = ""
        except GuardError as e:
            self.msg = f"домой: {e}"
        self._sync_from_arm()
        self.target = np.array(HOME, float)
        self.yaw, self.pitch = 0.0, 0.0

    def solve(self, t, yaw, pitch, q_init):
        """IK для точки t: подход наклонён на pitch от вертикали в сторону «вверх по кадру»,
        азимут губок yaw; несколько стартов, как в collect_pilot.plan. -> q5 | None"""
        # у 5-осевой SO-101 ось подхода наклоняется только в радиальной плоскости
        # (от базы наружу); наклон «вбок» потребовал бы шестой оси
        az = float(np.arctan2(t[1], t[0]))
        appr = np.array([np.cos(az) * np.sin(pitch), np.sin(az) * np.sin(pitch), -np.cos(pitch)])
        for q0 in (np.asarray(q_init, float), np.array([az, -0.6, 0.9, 0.9, 0.0]), np.array([az, -0.3, 0.3, 1.2, 0.0])):
            A._DATA.qpos[:] = 0
            A._DATA.qpos[:5] = q0
            q, err = self.ik.solve(A._DATA, t, approach=appr, q_init=q0, opening_yaw=yaw)
            if err < 0.003:
                return q
        return None

    def image_axes(self, q5):
        """Оси кадра камеры кисти в плоскости стола (базa): (вправо, вверх по кадру)."""
        C = self.d.fk_T(q5) @ self.Xw
        R = C[:3, :3]
        right, up = R @ np.array([1.0, 0, 0]), R @ np.array([0, -1.0, 0])
        ex = right[:2] / (np.linalg.norm(right[:2]) + 1e-9)
        ey = up[:2] / (np.linalg.norm(up[:2]) + 1e-9)
        if self.mirror:
            ex = -ex
        return ex, ey

    def blocks(self, t):
        """Стены зоны вокруг цели t: [{ang: угол в кадре кисти (0 = вправо, 90 = вверх), w: 0..1}]
        плюс близость к потолку/полу. w растёт линейно за RAMP до стены."""
        RAMP = 0.05
        ex, ey = self.image_axes(self.q_cmd[:5])
        out = []
        r = float(np.hypot(t[0], t[1]))
        u = t[:2] / max(r, 1e-6)
        cands = [(u, 1 - (R_MAX - r) / RAMP), (-u, 1 - (r - R_MIN) / RAMP),
                 (np.array([0, 1.0]), 1 - (0.22 - t[1]) / RAMP), (np.array([0, -1.0]), 1 - (t[1] + 0.22) / RAMP),
                 (np.array([-1.0, 0]), 1 - (t[0] - 0.08) / RAMP)]
        if self.limits.get("ik") and (abs(self.cmd["vx"]) + abs(self.cmd["vy"]) > 0.05):
            v = self.cmd["vx"] * ex + self.cmd["vy"] * ey
            cands.append((v / (np.linalg.norm(v) + 1e-9), 1.0))
        for d, w in cands:
            w = float(np.clip(w, 0, 1))
            if w > 0.02:
                out.append({"ang": round(float(np.degrees(np.arctan2(d @ ey, d @ ex)))), "w": round(w, 2)})
        up = float(np.clip(1 - (Z_MAX - t[2]) / 0.03, 0, 1))
        down = float(np.clip(1 - (t[2] - Z_MIN) / 0.02, 0, 1))
        return out, round(up, 2), round(down, 2)

    def status(self):
        F = self.d.fk_T(self.q_read[:5])
        tcp = F[:3, 3] + F[:3, :3] @ TCP_OFFSET
        b, up, down = self.blocks(self.target)
        return {"tcp": [round(float(v), 3) for v in tcp], "target": [round(float(v), 3) for v in self.target],
                "yaw_deg": round(float(np.degrees(self.yaw)), 0), "pitch_deg": round(float(np.degrees(self.pitch)), 0), "grip_pct": round(100 * (GRIP_MAX - self.grip) / (GRIP_MAX - GRIP_MIN)),
                "limits": self.limits, "blocks": b, "up": up, "down": down, "msg": self.msg,
                "axes": [[round(float(v), 2) for v in a] for a in self.image_axes(self.q_read[:5])],
                "hz": round(len(self.tick_times) / 2.0, 1),
                "rec": None if self.rec.ep is None else {"ep": self.rec.ep.name, "frames": self.rec.n,
                                                          "sec": round(time.time() - self.rec.t0, 1)},
                "alive": time.time() - self.last_cmd < 0.5, "stop": STOP.exists()}

    def park(self):
        self.msg = "парковка"
        try:
            self.d.goto(A.PARK + [A.GRIP_WIDE], 4.0, mode="grasp")
            self.d.settle(1.0, 0.5)
        except GuardError as e:
            self.msg = f"парковка: {e}"
        self._sync_from_arm()
        self.yaw, self.pitch = 0.0, 0.0

    def tick(self, dt):
        """Один такт 30 Гц: интегрировать команду, IK, охрана, запись приводам."""
        now = time.time()
        self.tick_times = [x for x in self.tick_times if now - x < 2.0] + [now]
        alive = time.time() - self.last_cmd < 0.5 and not STOP.exists()
        c = self.cmd if alive else {"vx": 0, "vy": 0, "vz": 0, "wyaw": 0, "grip": None}
        ex, ey = self.image_axes(self.q_cmd[:5])
        if alive and self.xr is not None:
            # AR: телефон «вперёд/влево/вверх» на момент захвата -> оси кадра кисти на момент захвата
            if self.xr_anchor is None:
                self.xr_anchor = (self.target.copy(), self.yaw, ex.copy(), ey.copy(), self.pitch)
            t0, yaw0, ax, ay, pitch0 = self.xr_anchor
            d = self.xr
            sc = d.get("scale") or self.xr_scale
            v_xy = (d["dy"] * (-ax) + d["dx"] * ay) * sc     # вперёд телефона = вверх по кадру, влево телефона = влево по кадру
            t = np.array([t0[0] + v_xy[0], t0[1] + v_xy[1], t0[2] + d["dz"] * sc])
            yaw_target = yaw0 + d["dyaw"]
            # телефон наклонили ниже (смотрит круче вниз) -> подход ближе к вертикали
            pitch_target = float(np.clip(pitch0 - d.get("dpitch", 0.0), 0.0, np.radians(60)))
        else:
            v_xy = c["vx"] * ex + c["vy"] * ey          # vx = вправо по кадру кисти, vy = вверх по кадру
            t = self.target + np.array([v_xy[0], v_xy[1], c["vz"]]) * self.speed * dt
            yaw_target, pitch_target = None, None
        self.limits = {"x": False, "y": False, "z": False, "r": False, "ik": False}
        r = float(np.hypot(t[0], t[1]))
        if r > R_MAX or r < R_MIN:
            s = np.clip(r, R_MIN, R_MAX) / max(r, 1e-6)
            t[0], t[1] = t[0] * s, t[1] * s
            self.limits["r"] = True
        if abs(t[1]) > 0.22:
            t[1] = np.clip(t[1], -0.22, 0.22); self.limits["y"] = True
        if t[0] < 0.08:
            t[0] = 0.08; self.limits["x"] = True
        if not Z_MIN <= t[2] <= Z_MAX:
            t[2] = float(np.clip(t[2], Z_MIN, Z_MAX)); self.limits["z"] = True
        yaw = yaw_target if yaw_target is not None else self.yaw + c["wyaw"] * 0.6 * dt      # 34°/с при полном отклонении
        pitch = pitch_target if pitch_target is not None else \
            float(np.clip(self.pitch + c.get("wpitch", 0.0) * 0.5 * dt, 0.0, np.radians(60)))
        if c["grip"] is not None:
            self.grip = float(GRIP_MAX - (GRIP_MAX - GRIP_MIN) * np.clip(c["grip"], 0, 1))
        q5 = self.solve(t, yaw, pitch, self.d.q_model(self.q_cmd[:5]))
        if q5 is None:
            self.limits["ik"] = True
            self.msg = "IK: точка недостижима"
            return
        cand = np.concatenate([self.d.q_cmd(q5), [self.grip]])
        cand = self.d.clamp_rad(cand)
        step = np.clip(cand - self.q_cmd, -0.05, 0.05)
        cand = self.q_cmd + step
        z, _ = self.d.tcp_z(cand[:5])
        if z < self.zmin:
            self.limits["z"] = True
            self.msg = "пол"
            return
        hits = self.d.self_hits(cand, margin=0.015)
        if hits and hits[0][2] < 0:
            self.limits["ik"] = True
            self.msg = "самостолкновение: " + self.d.sc.describe(hits)
            return
        self.target, self.yaw, self.pitch = t, yaw, pitch
        self.q_cmd = cand
        self.msg = "" if alive else ("стоп-файл" if STOP.exists() else "нет связи с телефоном — стою")
        if not self.dry:
            goals = {self.d.cal.joints[n]["id"]: self.d.raw_of(n, cand[j]) for j, n in enumerate(NAMES)}
            self.d.bus.sync_write(42, 2, goals)
        if self.rec.ep is not None:
            q, raw = self.d.read()
            self.q_read = q
            self.rec.tick(self.d, cand, q, raw, self.phase)


async def control_loop(app):
    t = app["teleop"]
    k = 0
    while True:
        t0 = time.time()
        try:
            t.tick(1.0 / HZ)
            k += 1
            if k % 3 == 0:
                q, raw = t.d.read()
                t.q_read = q
                t.d.publish(q, raw)
        except Exception as e:      # шина/чтение: не ронять сервер
            t.msg = f"ошибка: {e}"
        await asyncio.sleep(max(0.0, 1.0 / HZ - (time.time() - t0)))


try:
    from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack
    from av import VideoFrame
    HAVE_RTC = True
except Exception:          # aiortc нет в окружении — страница сама откатится на MJPEG
    HAVE_RTC = False


if HAVE_RTC:
    class FileCamTrack(VideoStreamTrack):
        """Кадры latest.jpg сервера камеры как видеотрек WebRTC (15 fps, верхняя камера 640x360)."""
        kind = "video"

        def __init__(self, cam):
            super().__init__()
            self.path = CAMS[cam]
            self.cam = cam
            self.last = 0
            self.frame = np.zeros((360 if cam == "top" else 480, 640, 3), np.uint8)

        async def recv(self):
            pts, time_base = await self.next_timestamp()
            try:
                m = self.path.stat().st_mtime
                if m != self.last:
                    img = cv2.imdecode(np.frombuffer(self.path.read_bytes(), np.uint8), cv2.IMREAD_COLOR)
                    if img is not None:
                        if self.cam == "top":
                            img = cv2.resize(img, (640, 360), interpolation=cv2.INTER_AREA)
                        self.frame = img
                        self.last = m
            except (OSError, ValueError):
                pass
            vf = VideoFrame.from_ndarray(self.frame, format="bgr24")
            vf.pts, vf.time_base = pts, time_base
            return vf

    PCS = set()

    async def rtc_offer(request):
        if not authorized(request):
            raise web.HTTPForbidden()
        params = await request.json()
        pc = RTCPeerConnection()
        PCS.add(pc)

        @pc.on("connectionstatechange")
        async def on_state():
            if pc.connectionState in ("failed", "closed"):
                await pc.close()
                PCS.discard(pc)
        pc.addTrack(FileCamTrack("wrist"))
        pc.addTrack(FileCamTrack("top"))
        await pc.setRemoteDescription(RTCSessionDescription(sdp=params["sdp"], type=params["type"]))
        await pc.setLocalDescription(await pc.createAnswer())
        return web.json_response({"sdp": pc.localDescription.sdp, "type": pc.localDescription.type})


async def mjpeg(request):
    if not authorized(request):
        raise web.HTTPForbidden()
    cam = request.match_info["cam"]
    path = CAMS[cam]
    resp = web.StreamResponse(headers={"Content-Type": "multipart/x-mixed-replace; boundary=frame",
                                       "Cache-Control": "no-cache"})
    await resp.prepare(request)
    last = 0
    try:
        while True:
            try:
                m = path.stat().st_mtime
            except FileNotFoundError:
                await asyncio.sleep(0.2); continue
            if m == last:
                await asyncio.sleep(0.02); continue
            last = m
            data = path.read_bytes()
            if cam == "top":
                img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
                if img is None:
                    continue
                img = cv2.resize(img, (640, 360), interpolation=cv2.INTER_AREA)
                data = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 70])[1].tobytes()
            await resp.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: %d\r\n\r\n" % len(data) + data + b"\r\n")
            await asyncio.sleep(1 / 12)
    except (ConnectionResetError, asyncio.CancelledError):
        pass
    return resp


async def ws(request):
    if not authorized(request):
        raise web.HTTPForbidden()
    sock = web.WebSocketResponse(heartbeat=5)
    await sock.prepare(request)
    t = request.app["teleop"]
    me = id(sock)
    if t.controller is None or time.time() - t.controller_seen > 30:     # свободно или оператор молчит 30 с
        t.controller, t.controller_seen = me, time.time()
    try:
      async for msg in sock:
        if msg.type != web.WSMsgType.TEXT:
            continue
        m = json.loads(msg.data)
        if m.get("cmd") == "take":
            t.controller, t.controller_seen = me, time.time()
            t.xr, t.xr_anchor = None, None
        role = "controller" if t.controller == me else "viewer"
        if role == "controller":
            t.controller_seen = time.time()
        if role == "viewer" and "cmd" not in m:
            st = t.status(); st["role"] = role
            await sock.send_str(json.dumps(st, ensure_ascii=False))
            continue
        if m.get("cmd") == "rec_start":
            t.msg = "запись " + t.rec.start(m.get("task", ""))
            t.phase = "teleop"
        elif m.get("cmd") == "rec_stop":
            name = t.rec.stop(m.get("success"))
            t.msg = f"записан {name}" if name else ""
        elif m.get("cmd") == "park":
            await asyncio.get_event_loop().run_in_executor(None, t.park)
        elif m.get("cmd") == "home":
            await asyncio.get_event_loop().run_in_executor(None, t.home)
        elif m.get("cmd") == "stop":
            STOP.touch(); t.msg = "СТОП"
        elif m.get("cmd") == "resume":
            STOP.unlink(missing_ok=True); t.msg = ""
        elif m.get("mode") == "xr":
            if m.get("engaged"):
                t.xr = {k: float(m.get(k, 0)) for k in ("dx", "dy", "dz", "dyaw", "dpitch")}
                t.xr["scale"] = float(m["scale"]) if m.get("scale") else None
            else:
                t.xr, t.xr_anchor = None, None
            t.cmd = {"vx": 0.0, "vy": 0.0, "vz": 0.0, "wyaw": 0.0, "grip": None if m.get("grip") is None else float(m["grip"])}
            t.last_cmd = time.time()
        else:
            t.xr, t.xr_anchor = None, None
            t.cmd = {"vx": float(m.get("vx", 0)), "vy": float(m.get("vy", 0)), "vz": float(m.get("vz", 0)),
                     "wyaw": float(m.get("wyaw", 0)), "wpitch": float(m.get("wpitch", 0)),
                     "grip": None if m.get("grip") is None else float(m["grip"])}
            t.last_cmd = time.time()
        st = t.status(); st["role"] = "controller" if t.controller == me else "viewer"
        await sock.send_str(json.dumps(st, ensure_ascii=False))
    finally:
      if t.controller == me:          # safe disconnect: управление отпущено, рука стоит, запись закрыта
          t.controller = None
          t.xr, t.xr_anchor = None, None
          t.cmd = {"vx": 0.0, "vy": 0.0, "vz": 0.0, "wyaw": 0.0, "wpitch": 0.0, "grip": None}
          if t.rec.ep is not None:
              t.rec.stop(None)
    return sock


async def xr_page(request):
    if not authorized(request):
        return web.Response(text="нужен ключ: откройте адрес с ?k=КЛЮЧ", status=403)
    page = TILT_HTML if request.path.startswith("/tilt") else XR_HTML
    resp = web.Response(text=page, content_type="text/html")
    if KEY:
        resp.set_cookie("k", KEY, max_age=86400 * 30, samesite="Lax")
    return resp


async def index(request):
    if not authorized(request):
        return web.Response(text="нужен ключ: откройте адрес с ?k=КЛЮЧ", status=403)
    resp = web.Response(text=HTML, content_type="text/html")
    if KEY:
        resp.set_cookie("k", KEY, max_age=86400 * 30, samesite="Lax")
    return resp


async def on_startup(app):
    app["loop_task"] = asyncio.create_task(control_loop(app))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--speed", type=float, default=0.06, help="м/с при полном отклонении джойстика")
    ap.add_argument("--dry", action="store_true", help="не писать приводам")
    ap.add_argument("--key", default="", help="ключ доступа для публичного туннеля (?k=...)")
    ap.add_argument("--mirror", action="store_true", help="зеркалить «вправо» джойстика, если на стенде перепутано")
    ap.add_argument("--xr-scale", type=float, default=1.0, help="масштаб смещения телефона -> схвата (1 = 1:1)")
    a = ap.parse_args()
    global KEY
    KEY = a.key
    app = web.Application()
    app["teleop"] = Teleop(a.dry, a.speed, a.mirror)
    app["teleop"].xr_scale = a.xr_scale
    app.add_routes([web.get("/", index), web.get("/xr", xr_page), web.get("/tilt", xr_page), web.get("/cam/{cam}.mjpg", mjpeg), web.get("/ws", ws)])
    if HAVE_RTC:
        app.add_routes([web.post("/rtc/offer", rtc_offer)])
    print(f"WebRTC: {'есть' if HAVE_RTC else 'нет (aiortc не установлен), только MJPEG'}", flush=True)
    app.on_startup.append(on_startup)
    print(f"телеоперация: http://<ip этой машины>:{a.port}/  (dry={a.dry})", flush=True)
    web.run_app(app, host="0.0.0.0", port=a.port, print=None)


if __name__ == "__main__":
    main()
