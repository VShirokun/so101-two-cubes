#!/usr/bin/env python3
"""Сторож калибровки: во время работы стенда, не прерывая её, проверяет, что
камеры, робот и объекты (коробка, опорные метки) не сдвинулись.

Читает только файлы (кадры серверов камер, телеметрию руки state.json, meta
новых эпизодов сборщика) — ни камер, ни шины не занимает. Раз в --period с
сравнивает текущее состояние с эталоном, снятым командой --snapshot на
заведомо исправном стенде (сразу после калибровки/проверки захватом).

Проверки и что они различают:
  top_bg   — сдвиг статичного фона в верхней камере (фазовая корреляция по
             нескольким областям стола вне зоны руки) -> ВЕРХНЯЯ КАМЕРА сдвинулась;
  anchors  — ArUco-метки, не принадлежащие кубам (коробка ID 16, любые метки,
             наклеенные на стол): сдвиг центров в px и мм по PnP -> при неподвижном
             фоне сдвинулся ОБЪЕКТ, при сдвиге фона — камера;
  arm_park — силуэт руки (оранжевая маска) в верхней камере, когда рука в
             парковке: при неподвижном фоне и камере сдвиг силуэта = сдвинулся
             РОБОТ или уехала калибровка приводов;
  wrist    — силуэт губок в камере кисти в парковке с раскрытым схватом ->
             КАМЕРА КИСТИ сдвинулась на креплении;
  servo    — доводка по камере кисти из meta новых эпизодов сборщика
             (расхождение верхней камеры и камеры кисти в мм) — сквозная проверка
             всей цепочки калибровки в каждом эпизоде.

Выход: /tmp/roboom_watch/status.json, первая строка /tmp/roboom_status.txt
(её показывает панель), журнал /tmp/roboom_watch/watch.log. При тревоге —
файл /tmp/roboom_watch/ALERT с причиной; с --stop-on-alert ещё и
/tmp/roboom_arm/stop (серия сборщика остановится на границе эпизода).

  calib_watch.py --snapshot            # эталон (рука в парковке, схват раскрыт)
  calib_watch.py [--period 1] [--stop-on-alert]
  calib_watch.py --once                # одна проверка, печать и выход
"""
import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "mlsim"))
from cube_cv import DICT  # noqa: E402

CAM_TOP, CAM_WR = Path("/tmp/roboom_cam/latest.jpg"), Path("/tmp/roboom_wrist/latest.jpg")
ARM_STATE = Path("/tmp/roboom_arm/state.json")
STOP = Path("/tmp/roboom_arm/stop")
STATUS_LINE = Path("/tmp/roboom_status.txt")
OUT = Path("/tmp/roboom_watch")
REF = ROOT / "real/calib/watch_ref.npz"
PILOT = ROOT / "real/data/pilot"
CUBE_IDS = {1, 2, 3, 4, 5, 6, 7, 10, 11, 13, 14, 15}
ANCHOR_SIZE_MM = {16: 60.0}          # коробка; прочие метки — --anchor-mm
PARK = np.array([0.75, -0.851, 0.414, 1.462, 0.0])
NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
# области фона верхней камеры (1920x1080) вне досягаемости руки: правая часть
# стола и передняя кромка; кубы малы и фазовую корреляцию не сбивают
BG_ROIS = [(1250, 300, 1900, 640), (1250, 660, 1900, 1000), (200, 900, 1000, 1070)]
ARM_ROI = (0, 0, 800, 420)            # рука в парковке (верхняя камера)
WRIST_ROI_FRAC = 0.55                 # губки — нижняя часть кадра кисти
THR = {"top_mm": (3.0, 6.0), "anchor_mm": (4.0, 8.0), "arm_px": (8.0, 16.0),
       "wrist_px": (4.0, 8.0), "servo_mm": (20.0, 30.0)}   # (warn, alert)
DET = cv2.aruco.ArucoDetector(DICT, cv2.aruco.DetectorParameters())


def load_json(p):
    try:
        return json.loads(Path(p).read_text())
    except Exception:
        return None


def intrinsics():
    c = load_json(ROOT / "real/calib/c920_intrinsics.json") or {}
    K = np.array(c.get("K", c.get("camera_matrix", [[1400, 0, 960], [0, 1400, 540], [0, 0, 1]])), float)
    D = np.array(c.get("dist", c.get("dist_coeffs", [0, 0, 0, 0, 0])), float).ravel()
    a = load_json(ROOT / "real/calib/autocalib_real.json") or {}
    z = float(np.array(a.get("top_cam_in_base", [0, 0, 0.40]))[2]) if "top_cam_in_base" in a else 0.40
    return K, D, z


def grab(path, max_age=2.0):
    try:
        if time.time() - path.stat().st_mtime > max_age:
            return None
        return cv2.imread(str(path))
    except Exception:
        return None


def arm_pose(max_age=3600.0):
    """Последняя опубликованная поза руки. Без драйвера рука стоит под моментом
    там, где её оставили, поэтому устаревшая телеметрия (до часа) годится."""
    s = load_json(ARM_STATE)
    if not s or time.time() - s.get("ts", 0) > max_age:
        return None
    return np.array([s["joints"][n]["rad"] for n in NAMES])


def at_park(q, tol_deg=3.0):
    return q is not None and np.degrees(np.abs(q[:5] - PARK)).max() < tol_deg and q[5] > 1.3


def prep(gray):
    g = cv2.GaussianBlur(gray, (0, 0), 1.2).astype(np.float32)
    g = (g - g.mean()) / (g.std() + 1e-6)
    return g


def bg_shift(ref_gray, cur_gray):
    """(медианный сдвиг px по областям, min отклик, список по областям)."""
    rows = []
    for x0, y0, x1, y1 in BG_ROIS:
        a, b = prep(ref_gray[y0:y1, x0:x1]), prep(cur_gray[y0:y1, x0:x1])
        win = cv2.createHanningWindow((x1 - x0, y1 - y0), cv2.CV_32F)
        (dx, dy), resp = cv2.phaseCorrelate(a, b, win)
        rows.append((float(dx), float(dy), float(resp)))
    good = [r for r in rows if r[2] > 0.15]
    if len(good) < 2:
        return None, rows
    pts = np.array([[r[0], r[1]] for r in good])
    # сдвиг камеры двигает ВСЕ области одинаково; если области разошлись больше
    # чем на 2 px — в кадре переставили предмет (ноутбук, мышь), а не камеру
    if np.ptp(pts, axis=0).max() > 2.0:
        return "inconsistent", rows
    d = np.median(pts, axis=0)
    return (float(d[0]), float(d[1])), rows


def orange_mask(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    m = cv2.inRange(hsv, (5, 120, 90), (22, 255, 255))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    return m


def blob(mask):
    n = int(mask.sum() // 255)
    if n < 200:
        return None
    ys, xs = np.nonzero(mask)
    return np.array([xs.mean(), ys.mean(), n], float)


def anchors(img, K, D, mm_default):
    corners, ids, _ = DET.detectMarkers(img)
    out = {}
    if ids is None:
        return out
    for c, i in zip(corners, ids.ravel()):
        i = int(i)
        if i in CUBE_IDS:
            continue
        px = c.reshape(4, 2)
        s = ANCHOR_SIZE_MM.get(i, mm_default) / 1000.0
        obj = np.array([[-s / 2, s / 2, 0], [s / 2, s / 2, 0], [s / 2, -s / 2, 0], [-s / 2, -s / 2, 0]], np.float32)
        ok, rvec, tvec = cv2.solvePnP(obj, px.astype(np.float32), K, D, flags=cv2.SOLVEPNP_IPPE_SQUARE)
        out[i] = {"center_px": px.mean(0).tolist(), "xyz_cam": tvec.ravel().tolist() if ok else None}
    return out


def snapshot(a):
    K, D, z = intrinsics()
    top, wr = grab(CAM_TOP, 5.0), grab(CAM_WR, 5.0)
    if top is None:
        raise SystemExit("нет свежего кадра верхней камеры")
    q = arm_pose()
    if q is not None and not at_park(q):
        raise SystemExit(f"рука не в парковке: {np.round(np.degrees(q[:5]), 1).tolist()} — эталон снимать только в PARK")
    if q is None and not a.assume_park:
        raise SystemExit("телеметрии руки нет: если рука точно в парковке со схватом раскрытым, добавьте --assume-park")
    gray = cv2.cvtColor(top, cv2.COLOR_BGR2GRAY)
    m = orange_mask(top[ARM_ROI[1]:ARM_ROI[3], ARM_ROI[0]:ARM_ROI[2]])
    arm = blob(m)
    ys, xs = np.nonzero(m)
    arm_box = np.array([xs.min(), ys.min(), xs.max(), ys.max()]) if len(xs) else np.array([])
    wrist_blob = None
    if wr is not None:
        h = wr.shape[0]
        wrist_blob = blob(orange_mask(wr[int(h * (1 - WRIST_ROI_FRAC)):]))
    anc = anchors(top, K, D, a.anchor_mm)
    REF.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(REF, top_gray=gray, arm=np.array([] if arm is None else arm),
                        wrist=np.array([] if wrist_blob is None else wrist_blob),
                        anchors=json.dumps(anc), ts=time.time(), cam_z=z,
                        q_ref=np.array([] if q is None else q), arm_box=arm_box)
    print(f"эталон записан: {REF}\n  фон: {len(BG_ROIS)} областей; силуэт руки: "
          f"{'есть' if arm is not None else 'НЕТ (проверьте ARM_ROI/освещение)'}; губки в камере кисти: "
          f"{'есть' if wrist_blob is not None else 'нет'}; опорные метки: {sorted(anc) or 'нет'}")
    if not anc:
        print("  подсказка: наклейте ArUco-метку DICT_4X4_50 (ID вне 1–15, например 20) на стол в поле зрения — "
              "она отличит сдвиг камеры от сдвига робота и объектов")


def level(v, thr):
    if v is None:
        return "n/a"
    return "ALERT" if v >= thr[1] else ("warn" if v >= thr[0] else "ok")


def check(ref, a, K, D, seen_eps):
    now = time.time()
    res = {"ts": now, "checks": {}, "verdict": "ok", "why": []}
    top, wr, q = grab(CAM_TOP), grab(CAM_WR), arm_pose()
    park = at_park(q)
    q_ref = ref["q_ref"] if "q_ref" in ref.files and len(ref["q_ref"]) else None
    same_pose = park and (q_ref is None or np.degrees(np.abs(q[:5] - q_ref[:5])).max() < 0.7)
    mm_per_px = float(ref["cam_z"]) / float(K[0, 0]) * 1000.0
    # 1. фон верхней камеры
    if top is None:
        res["checks"]["top_bg"] = {"state": "no frame"}
    else:
        gray = cv2.cvtColor(top, cv2.COLOR_BGR2GRAY)
        d, rows = bg_shift(ref["top_gray"], gray)
        if d is None:
            res["checks"]["top_bg"] = {"state": "occluded", "rois": rows}
        elif d == "inconsistent":
            res["checks"]["top_bg"] = {"state": "warn", "note": "области фона разошлись: в кадре переставлен предмет, не камера",
                                       "rois": [[round(x, 2) for x in r] for r in rows]}
        else:
            mm = float(np.hypot(*d)) * mm_per_px
            res["checks"]["top_bg"] = {"state": level(mm, THR["top_mm"]), "shift_px": [round(d[0], 2), round(d[1], 2)],
                                       "shift_mm": round(mm, 2), "rois": [[round(x, 2) for x in r] for r in rows]}
        # 2. опорные метки
        ref_anc = json.loads(str(ref["anchors"]))
        cur = anchors(top, K, D, a.anchor_mm)
        anc_out = {}
        for i, r in ref_anc.items():
            c = cur.get(int(i))
            if c is None:
                anc_out[i] = {"state": "not visible"}
                continue
            dpx = float(np.linalg.norm(np.subtract(c["center_px"], r["center_px"])))
            dmm = None
            if c["xyz_cam"] and r["xyz_cam"]:
                dmm = float(np.linalg.norm(np.subtract(c["xyz_cam"], r["xyz_cam"]))) * 1000
            anc_out[i] = {"state": level(dmm if dmm is not None else dpx * mm_per_px, THR["anchor_mm"]),
                          "shift_px": round(dpx, 2), "shift_mm": None if dmm is None else round(dmm, 2)}
        for i in cur:
            if str(i) not in ref_anc:
                anc_out[str(i)] = {"state": "new marker"}
        res["checks"]["anchors"] = anc_out
        # 3. силуэт руки в парковке
        if same_pose and len(ref["arm"]):
            roi = top[ARM_ROI[1]:ARM_ROI[3], ARM_ROI[0]:ARM_ROI[2]]
            m = orange_mask(roi)
            if "arm_box" in ref.files and len(ref["arm_box"]):
                # только окрестность силуэта из эталона (+40 px): тон кожи человека
                # рядом с роботом попадает в оранжевую маску и сдвигает центроид
                x0, y0, x1, y1 = ref["arm_box"]
                keep = np.zeros_like(m)
                keep[max(0, y0 - 40):y1 + 40, max(0, x0 - 40):x1 + 40] = 255
                m = cv2.bitwise_and(m, keep)
            b = blob(m)
            if b is None:
                res["checks"]["arm_park"] = {"state": "no silhouette"}
            else:
                dpx = float(np.linalg.norm(b[:2] - ref["arm"][:2]))
                ratio = float(b[2] / ref["arm"][2])
                st_ = "occluded" if not 0.8 < ratio < 1.25 else level(dpx, THR["arm_px"])
                res["checks"]["arm_park"] = {"state": st_, "shift_px": round(dpx, 2), "area_ratio": round(ratio, 3)}
        else:
            res["checks"]["arm_park"] = {"state": ("skip (поза отличается от эталонной)" if park else "skip (рука не в парковке)")
                                         if q is not None else "skip (нет телеметрии)"}
    # 4. губки в камере кисти
    # только в парковке: в других позах в нижнюю часть кадра кисти попадают другие
    # оранжевые звенья руки и маска «губок» уезжает (ложная тревога 75 px, 10.09)
    if wr is not None and same_pose and len(ref["wrist"]):
        h = wr.shape[0]
        b = blob(orange_mask(wr[int(h * (1 - WRIST_ROI_FRAC)):]))
        if b is None:
            res["checks"]["wrist"] = {"state": "no silhouette"}
        else:
            dpx = float(np.linalg.norm(b[:2] - ref["wrist"][:2]))
            res["checks"]["wrist"] = {"state": level(dpx, THR["wrist_px"]), "shift_px": round(dpx, 2)}
    else:
        res["checks"]["wrist"] = {"state": "skip"}
    # 5. доводка из новых эпизодов сборщика
    servo = None
    for ep in sorted(PILOT.glob("ep_*")):
        m = ep / "meta.json"
        if m.exists() and str(ep) not in seen_eps:
            seen_eps.add(str(ep))
            meta = load_json(m) or {}
            if meta.get("servo_corr_mm") is not None:
                servo = (ep.name, float(meta["servo_corr_mm"]))
    if servo:
        res["checks"]["servo"] = {"state": level(servo[1], THR["servo_mm"]), "episode": servo[0], "mm": servo[1]}
    # вердикт и объяснение
    st = {k: v.get("state", "") for k, v in res["checks"].items() if k != "anchors"}
    anc_states = {i: v.get("state", "") for i, v in res["checks"].get("anchors", {}).items()}
    lv = lambda s: 2 if s == "ALERT" else (1 if s == "warn" else 0)   # noqa: E731
    worst = max([lv(s) for s in st.values()] + [lv(s) for s in anc_states.values()] + [0])
    res["verdict"] = ["ok", "warn", "ALERT"][worst]
    top_moved = lv(st.get("top_bg", "")) > 0 and "shift_mm" in res["checks"].get("top_bg", {})
    if res["checks"].get("top_bg", {}).get("note"):
        res["why"].append(res["checks"]["top_bg"]["note"])
    if top_moved:
        res["why"].append(f"верхняя камера сдвинулась на {res['checks']['top_bg']['shift_mm']} мм по столу")
    for i, s in anc_states.items():
        if lv(s) > 0:
            res["why"].append(f"метка {i} ({'коробка' if int(i) == 16 else 'опора'}) сдвинулась на "
                              f"{res['checks']['anchors'][i].get('shift_mm') or res['checks']['anchors'][i]['shift_px']} "
                              f"{'мм' if res['checks']['anchors'][i].get('shift_mm') else 'px'}"
                              + (" — вместе с фоном: это камера" if top_moved else ""))
    if lv(st.get("arm_park", "")) > 0:
        res["why"].append("силуэт руки в парковке сместился на "
                          f"{res['checks']['arm_park']['shift_px']} px — сдвинулся робот или уехала калибровка приводов"
                          + (" (но и камера сдвинулась)" if top_moved else ""))
    if lv(st.get("wrist", "")) > 0:
        res["why"].append(f"губки в камере кисти сместились на {res['checks']['wrist']['shift_px']} px — камера кисти на креплении")
    if servo and lv(st.get("servo", "")) > 0:
        res["why"].append(f"доводка в {servo[0]}: верхняя камера и камера кисти расходятся на {servo[1]} мм")
    return res


def summary_line(res):
    parts = []
    for k in ("top_bg", "arm_park", "wrist", "servo"):
        c = res["checks"].get(k)
        if not c:
            continue
        v = c.get("shift_mm", c.get("shift_px", c.get("mm")))
        parts.append(f"{k}:{c['state']}" + (f"({v})" if v is not None else ""))
    anc = res["checks"].get("anchors", {})
    if anc:
        parts.append("anchors:" + ",".join(f"{i}={v['state']}" for i, v in anc.items()))
    return f"WATCH {res['verdict']} {time.strftime('%H:%M:%S')} " + " ".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", action="store_true")
    ap.add_argument("--assume-park", action="store_true")
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--period", type=float, default=1.0)
    ap.add_argument("--stop-on-alert", action="store_true")
    ap.add_argument("--anchor-mm", type=float, default=60.0, help="размер неизвестных опорных меток")
    a = ap.parse_args()
    if a.snapshot:
        snapshot(a)
        return
    if not REF.exists():
        raise SystemExit(f"нет эталона {REF}: снимите его командой --snapshot на исправном стенде")
    ref = np.load(REF, allow_pickle=False)
    K, D, _ = intrinsics()
    OUT.mkdir(parents=True, exist_ok=True)
    log = open(OUT / "watch.log", "a")
    seen = set(str(p) for p in PILOT.glob("ep_*") if (p / "meta.json").exists())
    last_verdict = None
    while True:
        res = check(ref, a, K, D, seen)
        (OUT / "status.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
        line = summary_line(res)
        try:
            STATUS_LINE.write_text(line + "\n" + "; ".join(res["why"]) + "\n")
        except Exception:
            pass
        if res["verdict"] != last_verdict or res["verdict"] != "ok":
            log.write(line + ("  | " + "; ".join(res["why"]) if res["why"] else "") + "\n")
            log.flush()
        if res["verdict"] == "ALERT":
            (OUT / "ALERT").write_text("\n".join(res["why"]) + "\n")
            if a.stop_on_alert and not STOP.exists():
                STOP.touch()
                log.write(f"{time.strftime('%H:%M:%S')} стоп-файл руки выставлен по тревоге\n")
                log.flush()
        elif (OUT / "ALERT").exists() and res["verdict"] == "ok":
            (OUT / "ALERT").unlink()
        last_verdict = res["verdict"]
        if a.once:
            print(line)
            for w in res["why"]:
                print("  ", w)
            print(json.dumps(res["checks"], ensure_ascii=False, indent=1))
            return
        time.sleep(a.period)


if __name__ == "__main__":
    main()
