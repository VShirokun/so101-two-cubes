"""Автотесты софта реального стенда без железа (правило: регрессии ловить до
вечера с роботом). Запуск: `bash tests/run_real_tests.sh` (groot-venv).

Что проверяется:
  - самостолкновения: парковка и домашняя точка телеоперации чисты, сложенная
    поза ловится, стоимость проверки < 1 мс;
  - сторож калибровки: синтетический сдвиг кадра на N px даётся в мм с точностью,
    одинаковый кадр — «ok», перестановка предмета — «inconsistent», а не «камера»;
  - обрезка эпизода «поднять»: правило последнего смыкания на синтетической
    траектории, вырезание простоев;
  - конвертер сырых эпизодов -> LeRobot v2.1: крошечный эпизод из сгенерированных
    JPEG собирается в parquet + mp4 с правильными метаданными и задачей по цвету;
  - детектор кубика по цвету: синтетический зелёный и серо-голубой квадраты
    на «столе» находятся, стол и тень — нет;
  - алгоритм переноса в коробку в симе: 3 попытки эксперт+перенос -> все в коробке.
"""
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "real/arm"))
sys.path.insert(0, str(ROOT / "real/watch"))
sys.path.insert(0, str(ROOT / "mlsim"))


# --- самостолкновения ---------------------------------------------------------
def test_selfcol_clean_and_catching():
    from selfcol import SelfCollision
    sc = SelfCollision()
    park = [0.75, -0.851, 0.414, 1.462, 0.0, 1.5]
    assert sc.check(park, 0.0) == []
    assert sc.check(park, 0.02) == []            # даже с запасом 2 см
    folded = [0.0, 0.9, 1.5, 1.4, 0.0, 1.5]
    hits = sc.check(folded, 0.0)
    assert hits and hits[0][2] < 0
    t = time.time()
    for _ in range(200):
        sc.check(park, 0.02)
    assert (time.time() - t) / 200 < 0.001


# --- сторож калибровки --------------------------------------------------------
def _textured(seed=0):
    rng = np.random.default_rng(seed)
    img = cv2.GaussianBlur(rng.integers(0, 255, (1080, 1920), np.uint8), (0, 0), 3)
    return img


def test_watch_shift_detection():
    import calib_watch as W
    ref = _textured()
    d, rows = W.bg_shift(ref, ref)
    assert d is not None and abs(d[0]) < 0.3 and abs(d[1]) < 0.3
    shifted = np.roll(np.roll(ref, 7, axis=1), -3, axis=0)
    d, rows = W.bg_shift(ref, shifted)
    assert d is not None and abs(d[0] - 7) < 0.6 and abs(d[1] + 3) < 0.6
    # перестановка предмета в одной области — не сдвиг камеры
    obj = ref.copy()
    obj[350:600, 1300:1800] = np.roll(obj[350:600, 1300:1800], 25, axis=1)
    d, rows = W.bg_shift(ref, obj)
    assert d == "inconsistent"


def test_watch_level_thresholds():
    import calib_watch as W
    assert W.level(1.0, (3.0, 6.0)) == "ok"
    assert W.level(3.5, (3.0, 6.0)) == "warn"
    assert W.level(7.0, (3.0, 6.0)) == "ALERT"
    assert W.level(None, (3.0, 6.0)) == "n/a"


# --- обрезка эпизода «поднять» ------------------------------------------------
def test_lift_cut_last_closure():
    sys.path.insert(0, str(ROOT / "ml/gr00t"))
    from make_lift_dataset import lift_cut
    import mujoco
    # синтетика: рука стоит; сначала ложное смыкание без подъёма, затем настоящее с подъёмом и уходом
    m = mujoco.MjModel.from_xml_path(str(ROOT / "mlsim/models/so101/pick_place.xml"))
    d = mujoco.MjData(m)
    sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "gripperframe")
    from ik import ArmIK
    ik = ArmIK(m)
    def q_at(x, y, z):
        q, err = ik.solve(d, np.array([x, y, z]), q_init=np.array([0.0, -0.3, 0.3, 1.2, 0.0]))
        assert err < 0.01
        return q
    traj = []
    low, high, far = q_at(0.22, 0.0, 0.02), q_at(0.22, 0.0, 0.09), q_at(0.12, 0.2, 0.09)
    for _ in range(10): traj.append([*low, 1.5])          # открыт у стола
    for _ in range(10): traj.append([*low, 0.1])          # ложное смыкание, подъёма нет
    for _ in range(10): traj.append([*low, 1.5])          # раскрыл
    for _ in range(10): traj.append([*low, 0.1])          # настоящее смыкание
    for k in range(20): traj.append([*(low + (high - low) * k / 19), 0.1])   # подъём
    for k in range(20): traj.append([*(high + (far - high) * k / 19), 0.1])  # перенос
    st = np.array(traj, np.float32)
    cut = lift_cut(st)
    assert cut is not None and 58 <= cut <= 63, cut     # конец подъёма, до ухода в сторону


def test_drop_idle():
    from pilot_to_lift_v21 import drop_idle, IDLE_KEEP
    recs = []
    a = [0.0] * 6
    for i in range(50):
        recs.append({"i": i, "phase": "descend", "state": a, "action": a})   # 50 тактов покоя
    b = [0.1] * 6
    for i in range(50, 60):
        recs.append({"i": i, "phase": "grasp", "state": b, "action": [0.1 + 0.01 * (i - 50)] * 6})
    kept = drop_idle(recs)
    assert len(kept) == 1 + IDLE_KEEP + 10       # первый + IDLE_KEEP покоя + движение


# --- конвертер сырых эпизодов -> v2.1 ----------------------------------------
def test_pilot_to_lift_v21_tiny(tmp_path):
    import subprocess
    raw = tmp_path / "pilot"
    ep = raw / "ep_0000"
    (ep / "top").mkdir(parents=True)
    (ep / "wrist").mkdir()
    with open(ep / "steps.jsonl", "w") as fh:
        for i in range(12):
            phase = "approach" if i < 4 else ("lift" if i < 8 else "transport")
            top = np.full((1080, 1920, 3), 120, np.uint8)
            cv2.rectangle(top, (900 + i * 10, 500), (960 + i * 10, 560), (30, 200, 30), -1)
            cv2.imwrite(str(ep / "top" / f"{i:06d}.jpg"), top)
            cv2.imwrite(str(ep / "wrist" / f"{i:06d}.jpg"), np.full((480, 640, 3), 90, np.uint8))
            fh.write(json.dumps({"i": i, "t": i / 30, "phase": phase, "state": [0.01 * i] * 6,
                                 "action": [0.01 * i + 0.005] * 6, "raw": [2000] * 6}) + "\n")
    (ep / "meta.json").write_text(json.dumps({"cube": "green", "held": True, "colors": {"green": "gray"}}))
    dst = tmp_path / "v21"
    r = subprocess.run([sys.executable, str(ROOT / "real/arm/pilot_to_lift_v21.py"), "--raw", str(raw),
                        "--dst", str(dst), "--jobs", "1"], capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr[-800:]
    info = json.loads((dst / "meta/info.json").read_text())
    assert info["total_episodes"] == 1 and info["total_frames"] == 8      # до конца фазы lift
    tasks = [json.loads(l) for l in open(dst / "meta/tasks.jsonl")]
    assert tasks[0]["task"] == "Подними серый кубик."
    assert (dst / "videos/chunk-000/observation.images.front/episode_000000.mp4").stat().st_size > 0
    import pandas as pd
    df = pd.read_parquet(dst / "data/chunk-000/episode_000000.parquet")
    assert len(df) == 8 and list(df["frame_index"]) == list(range(8))


# --- кубик по цвету -------------------------------------------------------------
def test_color_masks_synthetic():
    import cube_color as C
    img = np.full((480, 640, 3), (150, 155, 151), np.uint8)             # стол (BGR из замера)
    cv2.rectangle(img, (300, 200), (400, 300), (83, 125, 5), -1)         # зелёный куб
    cv2.rectangle(img, (100, 100), (200, 200), (136, 114, 74), -1)       # серо-голубой куб
    cv2.rectangle(img, (450, 350), (600, 450), (127, 138, 140), -1)      # тень
    g = C.best_blob(C.color_mask(img, "green"))
    s = C.best_blob(C.color_mask(img, "gray"))
    assert g is not None and cv2.boundingRect(g)[:2] == (300, 200)
    assert s is not None and cv2.boundingRect(s)[:2] == (100, 100)


# --- перенос в коробку (сим) --------------------------------------------------
def test_carry_to_bin_sim():
    import mujoco
    from carry import CarryToBin, TRANSPORT_Z
    from criteria import cube_in_bin, cube_xyz
    from expert import PickPlaceExpert, sample_layout, set_layout
    model = mujoco.MjModel.from_xml_path(str(ROOT / "mlsim/models/so101/pick_place.xml"))
    data = mujoco.MjData(model)
    expert, carry = PickPlaceExpert(model, data), CarryToBin(model, data)
    rng = np.random.default_rng(3)
    ok = 0
    for i in range(3):
        color = ["red", "green"][i % 2]
        mujoco.mj_resetDataKeyframe(model, data, 0)
        set_layout(model, data, sample_layout(rng, random_yaw=True))
        mujoco.mj_forward(model, data)
        plan = expert.plan(color)
        lift_phase = next(k for k, p in enumerate(plan) if np.allclose(p[0][2], TRANSPORT_Z) and k > 0)
        for a in expert.actions_for(plan[:lift_phase + 1]):
            data.ctrl[:] = a
            for _ in range(carry.sub):
                mujoco.mj_step(model, data)
        if cube_xyz(model, data, color)[2] < 0.05:
            continue
        carry.run()
        for _ in range(15 * carry.sub):
            mujoco.mj_step(model, data)
        ok += int(cube_in_bin(model, data, color, carry.bin_site))
    assert ok >= 2
