"""Демо Д-20 без железа: датасет демонстраций в формате LeRobot из симулятора.

Что здесь честно, а что нет — важно проговорить, потому что на этом строится
весь ML-трек:

  • Формат, состав наблюдений и действий — ровно те же, что на живой SO-101:
    кадры камер + позы шести суставов, действие = целевые углы приводов.
    Обученная на этом политика запускается на железе без переделок.
  • Демонстрации порождает скриптовый оператор, а не человек. На реальном
    роботе их пишет телеоператор; здесь рук нет, и это записано в витрине
    прямым текстом, а не спрятано.
  • Политика координат кубиков НЕ видит: только пиксели и суставы. Никакого
    «подсматривания» в состояние симулятора в наблюдениях нет.

Запуск:
    python3 mlsim/record_dataset.py --episodes 50
    python3 mlsim/record_dataset.py --episodes 4 --preview   # быстрая проверка
"""

import argparse
import shutil
from pathlib import Path

import mujoco
import numpy as np

from criteria import cube_in_bin
from expert import (CONTROL_HZ, CUBES, ZONE_X, ZONE_Y, PickPlaceExpert,
                    sample_layout, set_layout)

SCENE = Path(__file__).parent / "models/so101/pick_place.xml"
JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
IMG_H, IMG_W = 240, 320          # умолчание; для GR00T есть смысл писать крупнее (см. --img-h)
REPO_ID = "roboom/sim-cubes"


def success(model, data, expert, color: str) -> bool:
    """Кубик внутри коробки. Критерий общий с оценкой — см. `criteria.py`:
    приёмка датасета и подсчёт успеха обязаны считать одинаково."""
    return cube_in_bin(model, data, color, expert.bin_site)


def random_start_pose(model, data, expert, rng) -> bool:
    """Случайная стартовая поза руки вместо одинакового «дома».

    Зачем: при одинаковом старте единственный кадр, где цель определяется
    командой, — самый первый; уже со второго кадра цель читается из
    проприоцепции («продолжай начатое движение»), и модель может вообще не
    учить язык. Измерено на SmolVLA: ошибка предсказания на кадре 0 в 12 раз
    выше, чем на кадре 30, а различение команд — монетка.

    Поза берётся не случайными углами, а IK в случайную точку над рабочей
    зоной: гарантированно достижимо, без столкновений со столом и кубиками.
    """
    target = np.array([
        rng.uniform(0.16, 0.30),
        rng.uniform(-0.16, 0.16),
        rng.uniform(0.08, 0.14),
    ])
    q, err = expert.ik.solve(data, target)
    if err > 0.02:
        return False
    data.qpos[expert.ik.qadr] = q
    data.ctrl[:5] = q
    data.ctrl[5] = float(data.qpos[
        model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "gripper")]])
    mujoco.mj_forward(model, data)
    return True


CUBE_HALF = 0.014                    # полуразмер кубика из pick_place.xml
FINGER_TIPS = ["fixed_jaw_sph_tip1", "fixed_jaw_sph_tip2", "fixed_jaw_sph_tip3",
               "moving_jaw_sph_tip1", "moving_jaw_sph_tip2", "moving_jaw_sph_tip3"]
GRIP_BODIES = ("gripper", "moving_jaw_so101_v1", "camera_mount")
_CUBE_CONTACTS: dict[int, dict[int, tuple[int, int]]] = {}
_GRIP_CONTACTS: dict[int, dict[int, tuple[int, int]]] = {}


def _grip_contact_defaults(model) -> dict[int, tuple[int, int]]:
    """Коллизионные геомы губок (визуальные, с contype=0, не трогаем)."""
    if id(model) not in _GRIP_CONTACTS:
        out = {}
        for name in GRIP_BODIES:
            bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
            for g in range(model.ngeom):
                if model.geom_bodyid[g] == bid and (
                        model.geom_contype[g] or model.geom_conaffinity[g]):
                    out[g] = (int(model.geom_contype[g]), int(model.geom_conaffinity[g]))
        _GRIP_CONTACTS[id(model)] = out
    return _GRIP_CONTACTS[id(model)]


def _reset_grip_contacts(model) -> None:
    for g, (ct, ca) in _grip_contact_defaults(model).items():
        model.geom_contype[g] = ct
        model.geom_conaffinity[g] = ca


def _disable_grip_contacts(model) -> None:
    for g in _grip_contact_defaults(model):
        model.geom_contype[g] = 0
        model.geom_conaffinity[g] = 0


def _cube_contact_defaults(model) -> dict[int, tuple[int, int]]:
    """Исходные contype/conaffinity геомов кубиков — снимаются один раз ДО
    любого отключения, чтобы аварийно оборванный эпизод не оставил кубик
    бесплотным навсегда."""
    if id(model) not in _CUBE_CONTACTS:
        out = {}
        for name in ("red_geom", "green_geom"):
            g = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
            out[g] = (int(model.geom_contype[g]), int(model.geom_conaffinity[g]))
        _CUBE_CONTACTS[id(model)] = out
    return _CUBE_CONTACTS[id(model)]


def _reset_cube_contacts(model) -> None:
    for g, (ct, ca) in _cube_contact_defaults(model).items():
        model.geom_contype[g] = ct
        model.geom_conaffinity[g] = ca


def rollout(model, data, renderer, cams, rng, color, random_start=False,
            drop=False, miss=False, trim=False, random_yaw=False, wide=False):
    """Один эпизод. Возвращает (кадры, успех) — кадры пустые, если IK не решился.

    drop: захват всегда ЧИСТЫЙ, но на случайном шаге подъёма/переноса кубик
    «выпадает» из закрытого схвата: его геому выключаются контакты, он
    проваливается сквозь сомкнутые губки, и как только верхняя грань опускается
    ниже кончиков пальцев (а нижняя ещё над столом) — контакты возвращаются, и
    кубик естественной физикой падает на стол в случайной ориентации. Рука
    «замечает» пропажу с человеческой задержкой и берёт его заново.

    miss: «призрачный» ПЕРВЫЙ захват (идея Владимира): подвод и смыкание
    идеальны и неотличимы от успешных, но с паузы над кубиком у ГУБОК выключены
    коллизии — они смыкаются сквозь кубик, рука уходит вверх пустой (кубик так
    и стоит на столе), и когда кончики пальцев поднимаются выше его верхней
    грани, коллизии возвращаются; затем рука берёт кубик по-настоящему.
    Отключаются геомы губок, а не кубика: бесплотный кубик провалился бы и
    сквозь стол. Это модель самой ЧАСТОЙ ошибки политики (все 170 промахов
    из 600 попыток этапа 11 — несостоявшийся захват), в отличие от drop.
    trim: оборвать эпизод через ~0.4 с после попадания кубика в коробку.
    """
    _reset_cube_contacts(model)      # прошлый эпизод мог оборваться посреди «падения»
    _reset_grip_contacts(model)      # ... или посреди «призрачного» захвата
    mujoco.mj_resetDataKeyframe(model, data, 0)
    set_layout(model, data, sample_layout(rng, random_yaw=random_yaw, wide=wide))
    mujoco.mj_forward(model, data)

    expert = PickPlaceExpert(model, data)
    if random_start and not random_start_pose(model, data, expert, rng):
        return [], False

    sub = int(round(1.0 / (CONTROL_HZ * model.opt.timestep)))
    qadr = [model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j)] for j in JOINTS]
    frames = []
    tail = -1

    def step(act, cb=None):
        nonlocal tail
        _record_step(model, data, renderer, cams, qadr, sub, act, frames, cb)
        if trim:
            if tail < 0 and success(model, data, expert, color):
                tail = 12                       # ~0.4 с хвоста после успеха
            elif tail > 0:
                tail -= 1
                if tail == 0:
                    return True
        return False

    plan = expert.plan(color)
    acts = expert.actions_for(plan)
    if not acts:
        return [], False

    if miss:
        # --- эпизод с «призрачным» первым захватом ---
        cube_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, CUBES[color]["body"])
        tip_gids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, n) for n in FINGER_TIPS]
        lens = [max(2, int(p[2] * CONTROL_HZ)) for p in plan]
        # План (expert.plan): 0 подвод, 1 быстрый спуск, 2 зависание,
        # 3 медленный спуск, 4 пауза, 5 смыкание, 6 подъём, 7 перенос, 8-9 сброс.
        acts1 = acts[:sum(lens[:7])]         # до конца подъёма включительно
        k_off = sum(lens[:4])                # с паузы: губки уже вокруг кубика
        ghost = {"on": False}

        def ghost_cb():
            """Вернуть коллизии губок, когда кончики пальцев выше кубика."""
            if not ghost["on"]:
                return
            tips_z = min(data.geom_xpos[g][2] for g in tip_gids)
            if tips_z > data.xpos[cube_bid][2] + CUBE_HALF + 0.004:
                _reset_grip_contacts(model)
                ghost["on"] = False

        for k, act in enumerate(acts1):
            if k == k_off:
                _disable_grip_contacts(model)
                ghost["on"] = True
            if step(act, ghost_cb if ghost["on"] else None):
                return frames, True
        _reset_grip_contacts(model)          # страховка

        acts2 = expert.actions(color)        # настоящий захват: кубик там же
        if not acts2:
            return [], False
        for act in acts2:
            if step(act):
                return frames, True
        return frames, success(model, data, expert, color)

    if not drop:
        for act in acts:
            if step(act):
                return frames, True
        return frames, success(model, data, expert, color)

    # --- эпизод с выпадением кубика ---
    cube_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, CUBES[color]["body"])
    cube_gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"{color}_geom")
    dofadr = model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, CUBES[color]["joint"])]
    tip_gids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, n) for n in FINGER_TIPS]
    bin_xy = data.site_xpos[expert.bin_site][:2].copy()

    # Окно выпадения: вторая половина подъёма — первая треть переноса. Дальше
    # нельзя: кубик, уроненный на полпути к коробке, приземляется вне зоны
    # досягаемости, и повторный захват не решается ик-решателем (проверено
    # дымовым тестом: брак 5/11 против обычных ~1/60).
    lens = [max(2, int(p[2] * CONTROL_HZ)) for p in plan]
    lift_start = sum(lens[:6])            # начало подъёма (фаза 6)
    carry_end = sum(lens[:7]) + lens[7] // 3
    drop_from = int(rng.integers(lift_start + lens[6] // 2, carry_end))

    falling = {"on": False}

    def fall_cb():
        """Каждый подшаг физики: вернуть контакты, когда кубик миновал пальцы,
        но ещё не долетел до стола (за один шаг управления он пролетает больше
        сантиметра — на частоте 30 Гц окно было бы уже пропущено)."""
        if not falling["on"]:
            return
        z = data.xpos[cube_bid][2]
        tips_z = min(data.geom_xpos[g][2] for g in tip_gids)
        if z + CUBE_HALF < tips_z - 0.004 and z - CUBE_HALF > 0.004:
            _reset_cube_contacts(model)
            falling["on"] = False

    dropped_at = -1
    for k, act in enumerate(acts):
        if dropped_at < 0 and drop_from <= k <= carry_end:
            x, y, z = data.xpos[cube_bid]
            held = z > 0.05
            # Ронять только над рабочей зоной (с небольшим припуском) и не над
            # коробкой: место приземления обязано быть достижимо для повтора.
            in_zone = (ZONE_X[0] - 0.03 <= x <= ZONE_X[1] + 0.02
                       and ZONE_Y[0] - 0.03 <= y <= ZONE_Y[1] + 0.03)
            over_bin = np.linalg.norm((x - bin_xy[0], y - bin_xy[1])) <= 0.09
            if held and in_zone and not over_bin:
                model.geom_contype[cube_gid] = 0
                model.geom_conaffinity[cube_gid] = 0
                # Закрутка в полёте: кубик обязан упасть в ДРУГОЙ ориентации и
                # чуть в стороне — иначе модель могла бы выучить «вернись в ту
                # же точку тем же движением», а нужно «посмотри, как он лёг
                # ТЕПЕРЬ, и перестрой подход» (замечание Владимира).
                axis = rng.normal(size=3)
                axis /= np.linalg.norm(axis)
                data.qvel[dofadr + 3:dofadr + 6] = axis * rng.uniform(9.0, 18.0)
                data.qvel[dofadr:dofadr + 2] += rng.uniform(-0.12, 0.12, 2)
                falling["on"] = True
                dropped_at = k
        if step(act, fall_cb if dropped_at >= 0 else None):
            return frames, True
        if dropped_at >= 0 and k - dropped_at >= 12:
            break                        # ~0.4 с «замечает» пропажу и прерывает перенос
    _reset_cube_contacts(model)          # страховка: окно восстановления нельзя пропустить
    if dropped_at < 0:
        return frames, success(model, data, expert, color)

    # Дождаться, пока кубик уляжется (держим текущую позу): план повторного
    # захвата должен строиться по УЛЁГШЕМУСЯ кубику — позиция и грань финальны.
    hold = frames[-1][2]
    for _ in range(18):
        if np.linalg.norm(data.qvel[dofadr:dofadr + 3]) < 0.05:
            break
        if step(hold):
            return frames, True

    acts2 = expert.actions(color)        # обычный чистый план по новому положению и углу
    if not acts2:
        return [], False
    for act in acts2:
        if step(act):
            return frames, True
    return frames, success(model, data, expert, color)


def _record_step(model, data, renderer, cams, qadr, sub, act, frames, substep_cb=None):
    """Снять наблюдение, применить действие, шагнуть физику (вынесено из rollout)."""
    # Наблюдение снимаем ДО действия: политика учится «вижу это → делаю то».
    obs = {}
    for cam in cams:
        r = renderer[cam] if isinstance(renderer, dict) else renderer
        r.update_scene(data, camera=cam)
        obs[cam] = r.render().copy()
    state = np.array([data.qpos[a] for a in qadr], dtype=np.float32)
    frames.append((obs, state, act.astype(np.float32)))

    data.ctrl[:] = act
    for _ in range(sub):
        mujoco.mj_step(model, data)
        if substep_cb is not None:
            substep_cb()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=50, help="успешных эпизодов всего")
    ap.add_argument("--root", default=str(Path.home() / ".cache/huggingface/lerobot/roboom/sim-cubes"))
    ap.add_argument("--seed", type=int, default=20260810)
    ap.add_argument("--cams", default="top,wrist_cam")
    ap.add_argument("--img-h", type=int, default=IMG_H,
                    help="высота кадра. GR00T приводит вход к 256 по короткой стороне: "
                         "наши 240 РАСТЯГИВАЮТСЯ; запись 480x640 даёт настоящую детализацию")
    ap.add_argument("--img-w", type=int, default=IMG_W)
    ap.add_argument("--wrist-size", type=int, default=0,
                    help="квадратное разрешение КАМЕРЫ ЗАПЯСТЬЯ (0 = как у всех). "
                         "Камера жёстко закреплена на схвате, её кадр — система "
                         "координат схвата; квадрат 512 даёт честные 256x256 после "
                         "resize без кропа и без искажения геометрии")
    ap.add_argument("--colors", default="red,green",
                    help="какие языковые задачи писать; 'red' даёт однозадачный датасет")
    ap.add_argument("--random-start", action="store_true",
                    help="случайная стартовая поза руки вместо одинакового «дома»")
    ap.add_argument("--random-start-prob", type=float, default=1.0,
                    help="доля эпизодов со случайным стартом; остальные — из домашней "
                         "позы, как сбрасывает сцену стенд оценки. Полностью случайные "
                         "старты (1.0) убирают утечку цели через проприоцепцию, но "
                         "оставляют модель без домашней позы: измерено 5%% успеха из "
                         "дома против 48%% со случайного старта")
    ap.add_argument("--random-yaw", action="store_true",
                    help="случайный поворот кубиков вокруг вертикали (0..90°); раньше "
                         "все кубики стояли строго одинаково")
    ap.add_argument("--wide-zone", action="store_true",
                    help="расширенная зона раскладки (до достижимой руки: x 0.12-0.32, "
                         "|y| до 0.22) — кубики у краёв и в углах. Лечит главный класс "
                         "промахов: политика выбивает кубик за номинальную зону и не "
                         "знает, что с ним делать (61% неудач «кубик не поднят»). "
                         "Стенд оценки эту опцию не использует")
    ap.add_argument("--miss-prob", type=float, default=0.0,
                    help="доля эпизодов с «призрачным» первым захватом: подвод и "
                         "смыкание идеальны, но у губок на время смыкания и подъёма "
                         "выключены коллизии — рука поднимается пустой (кубик стоит "
                         "на месте) и берёт его повторно. Модель самой частой ошибки "
                         "политики: несостоявшийся захват")
    ap.add_argument("--drop-prob", type=float, default=0.0,
                    help="доля эпизодов, где кубик «выпадает» из закрытого схвата "
                         "посреди переноса (контакты геома выключаются, он проваливается "
                         "сквозь губки и падает на стол в случайной ориентации), после "
                         "чего рука берёт его заново. Захваты при этом ВСЕГДА чистые — "
                         "в отличие от прежнего --recovery-prob с нарочными промахами")
    ap.add_argument("--trim", action="store_true",
                    help="закончить эпизод через ~0.4 с после попадания кубика в коробку")
    ap.add_argument("--start-copies", type=int, default=0,
                    help="сколько укороченных копий начала эпизода дописывать")
    ap.add_argument("--start-len", type=int, default=20,
                    help="длина укороченной копии в кадрах")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--preview", action="store_true", help="не писать датасет, только прогнать")
    args = ap.parse_args()

    model = mujoco.MjModel.from_xml_path(str(SCENE))
    data = mujoco.MjData(model)
    cams = args.cams.split(",")
    renderers = {}
    for cam in cams:
        if cam == "wrist_cam" and args.wrist_size > 0:
            renderers[cam] = mujoco.Renderer(model, args.wrist_size, args.wrist_size)
        else:
            renderers[cam] = mujoco.Renderer(model, args.img_h, args.img_w)
    renderer = renderers  # дальше по коду словарь: рендер на камеру
    rng = np.random.default_rng(args.seed)

    dataset = None
    if not args.preview:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        root = Path(args.root)
        if root.exists():
            if not args.overwrite:
                print(f"датасет уже существует: {root}\nдобавьте --overwrite, если пересоздать")
                return 1
            shutil.rmtree(root)

        features = {
            "observation.state": {"dtype": "float32", "shape": (6,), "names": JOINTS},
            "action": {"dtype": "float32", "shape": (6,), "names": JOINTS},
        }
        for cam in cams:
            if cam == "wrist_cam" and args.wrist_size > 0:
                shape = (args.wrist_size, args.wrist_size, 3)
            else:
                shape = (args.img_h, args.img_w, 3)
            features[f"observation.images.{cam}"] = {
                "dtype": "video", "shape": shape,
                "names": ["height", "width", "channels"],
            }
        dataset = LeRobotDataset.create(
            repo_id=REPO_ID, fps=CONTROL_HZ, root=root,
            robot_type="so101", features=features, use_videos=True,
        )

    saved = rejected = 0
    colors = [c.strip() for c in args.colors.split(",") if c.strip()]

    def write_episode(frames, color):
        for obs, state, act in frames:
            # Языковая инструкция едет полем кадра — в этой версии LeRobot
            # add_frame принимает задачу внутри словаря, а не аргументом.
            frame = {
                "observation.state": state,
                "action": act,
                "task": CUBES[color]["task"],
            }
            for cam in cams:
                frame[f"observation.images.{cam}"] = obs[cam]
            dataset.add_frame(frame)
        dataset.save_episode()

    while saved < args.episodes:
        color = colors[saved % len(colors)]  # поровну всех заказанных задач
        want_random = args.random_start and rng.random() < args.random_start_prob
        r = rng.random()
        want_miss = r < args.miss_prob
        want_drop = (not want_miss) and r < args.miss_prob + args.drop_prob
        frames, ok = rollout(model, data, renderer, cams, rng, color,
                             random_start=want_random,
                             drop=want_drop, miss=want_miss, trim=args.trim,
                             random_yaw=args.random_yaw, wide=args.wide_zone)
        if not ok or not frames:
            rejected += 1
            print(f"  брак ({'IK' if not frames else 'кубик мимо коробки'}), всего брака: {rejected}")
            if rejected > args.episodes:
                print("!! брака больше, чем эпизодов — прекращаю, чините эксперта")
                return 2
            continue

        if dataset is not None:
            write_episode(frames, color)
            # Перевес решающих кадров: начало эпизода — единственное место, где
            # цель определяется командой, но это ~15 кадров из 138, и в среднем
            # loss их почти не видит. Дописываем укороченные копии начала как
            # отдельные эпизоды: недостающий хвост чанка помечается is_pad и
            # маскируется в loss (после починки опечатки actions_id_pad в
            # lerobot — см. DEVIATIONS.md).
            for _ in range(args.start_copies):
                write_episode(frames[:args.start_len], color)

        saved += 1
        extra = f" (+{args.start_copies}x{args.start_len})" if args.start_copies else ""
        print(f"  эпизод {saved}/{args.episodes} [{color}] кадров {len(frames)}{extra}")

    print(f"\nготово: {saved} эпизодов, брак {rejected}")
    if dataset is not None:
        print(f"датасет: {args.root}")
        print("проверить:  lerobot-dataset-viz --repo-id " + REPO_ID + " --root " + args.root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
