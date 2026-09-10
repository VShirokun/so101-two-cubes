#!/usr/bin/env python3
"""Политика GR00T на реальном стенде: задача «взять и поднять кубик».

Цикл попытки: парковка -> каждые 16 тактов (0,53 с) модель получает обе камеры
(верхняя 1920x1080 -> кроп 4:3 -> 640x480, кисть 640x480), позу суставов и
текст задачи, отдаёт чанк из 16 абсолютных целей (радианы модели), которые
исполняются на 30 Гц. Ограничения на исполнение: шаг сустава за такт не больше
--max-step рад, floor-guard по FK (низ губок не ниже пола режима grasp),
отставание от команды > 0.35 рад — стоп (как в arm_driver.goto).

Успех: схват сомкнут (< 0.6 рад) и точка схвата выше --lift-z 1 с подряд,
после чего камера кисти подтверждает куб в губках (метка ближе 10 см).
Дальше — алгоритмически: куб опускается на стол там, где рука стоит,
рука в парковку. Видео обеих камер каждой попытки -> --video-dir
(нарезка промахов — правило проекта).

  run_policy.py --policy <чекпоинт> --attempts 10 [--seconds 15] [--task "Подними кубик."]
Стоп: touch /tmp/roboom_arm/stop
"""
import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import autocalib_real as A  # noqa: E402
from arm_driver import ArmDriver, GuardError, HZ, NAMES, STOP, TIP_BELOW_TCP, Z_FLOOR  # noqa: E402
from collect_pilot import cube_by_wrist, cube_in_gripper, look, move, plan, top_frame_640, waypoints  # noqa: E402
from cube_color import color_mask, cube_by_wrist_color  # noqa: E402
from ik import ArmIK  # noqa: E402

TASK = "Подними кубик."


def observe(d, task):
    top = A.grab(A.CAM_TOP)
    wr = A.grab(A.CAM_WR)
    q, raw = d.read()
    state = np.concatenate([d.q_model(q), [float(q[5])]]).astype(np.float32)
    obs = {"video": {"front": cv2.cvtColor(top_frame_640(top), cv2.COLOR_BGR2RGB)[None, None],
                     "wrist": cv2.cvtColor(wr, cv2.COLOR_BGR2RGB)[None, None]},
           "state": {"single_arm": state[:5][None, None], "gripper": state[5:6][None, None]},
           "language": {"annotation.human.task_description": [[task]]}}
    return obs, q, raw, top, wr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True)
    ap.add_argument("--attempts", type=int, default=10)
    ap.add_argument("--seconds", type=float, default=15.0)
    ap.add_argument("--task", default=TASK)
    ap.add_argument("--lift-z", type=float, default=0.06, help="точка схвата выше — куб поднят")
    ap.add_argument("--max-step", type=float, default=0.06, help="рад за такт на сустав")
    ap.add_argument("--exec-horizon", type=int, default=16)
    ap.add_argument("--video-dir", type=Path, default=A._ROOT / "real/data/policy_runs")
    ap.add_argument("--dry", action="store_true", help="модель считает, рука не движется")
    ap.add_argument("--carry-to-box", action="store_true",
                    help="после подъёма кубик несёт в коробку алгоритм (real/arm/carry_real.py; поза коробки "
                         "из real/calib/box_pose.json, снять командой `carry_real.py box`)")
    ap.add_argument("--servo-color", default="",
                    help="настоящие кубы без меток: цвета через запятую (green,gray) — доводка гибрида и "
                         "подтверждение захвата по цветному пятну в камере кисти вместо ArUco")
    ap.add_argument("--servo-grasp", action="store_true",
                    help="гибрид: когда политика зависла над кубом (tcp 3.5–8 см, схват раскрыт, рука "
                         "почти стоит), доводку по камере кисти, спуск, смыкание и подъём делает алгоритм сборщика")
    ap.add_argument("--min-brightness", type=float, default=80.0,
                    help="средняя яркость верхней камеры, ниже которой запуск отклоняется "
                         "(датасет писался при ~130; ночью 10.09 было 27 — политика вне распределения)")
    a = ap.parse_args()
    bright = float(A.grab(A.CAM_TOP).mean())
    if bright < a.min_brightness:
        raise SystemExit(f"верхняя камера слишком тёмная: средняя яркость {bright:.0f} < {a.min_brightness:.0f} "
                         "— включите свет как при записи датасета")
    print(f"яркость верхней камеры {bright:.0f} (датасет ~130)", flush=True)

    from gr00t.policy import Gr00tPolicy
    policy = Gr00tPolicy(model_path=a.policy, embodiment_tag="NEW_EMBODIMENT", device="cuda")
    ik = ArmIK(A._MODEL)
    d = ArmDriver()
    d.torque(True)
    d.hold()
    a.video_dir.mkdir(parents=True, exist_ok=True)
    tag = time.strftime("%m%d-%H%M")
    zmin = Z_FLOOR["grasp"] + TIP_BELOW_TCP
    res_cal = json.loads(A.OUT.read_text())
    Xt, Xw = np.array(res_cal["T_base_topcam"]), np.array(res_cal["T_gripper_wristcam"])
    box_xy = None
    if a.carry_to_box:
        from carry_real import RIM_HALF, carry_to_box, load_box
        box_xy, _ = load_box()
        print(f"коробка: {np.round(box_xy, 3).tolist()}", flush=True)
    results = []
    try:
        for k in range(a.attempts):
            if STOP.exists():
                print("стоп-файл — прогон прерван", flush=True)
                break
            try:
                d.goto(A.PARK + [A.GRIP_WIDE], 4.0, mode="grasp")
                d.settle()
            except GuardError as e:
                print(f"парковка перед попыткой {k+1}: {e} — прогон прерван", flush=True)
                break
            try:
                cubes0 = {c: T[:2, 3].copy() for c, T in look(Xt).items()}   # где кубы до попытки
            except Exception:
                cubes0 = {}
            vw = cv2.VideoWriter(str(a.video_dir / f"run-{tag}-a{k+1:02d}.mp4"),
                                 cv2.VideoWriter_fourcc(*"mp4v"), 15, (1280, 480))
            t0, ticks, held_ticks, lifted, why = time.time(), 0, 0, False, ""
            n_selfcol, handover, q_hist = 0, False, []
            q_cmd_prev = None
            while time.time() - t0 < a.seconds and not lifted and not STOP.exists():
                obs, q, raw, top, wr = observe(d, a.task)
                if q_cmd_prev is None:
                    q_cmd_prev = q.copy()
                t_inf = time.time()
                chunk, _ = policy.get_action(obs)
                arm = np.asarray(chunk.get("action.single_arm", chunk.get("single_arm")))[0]
                grip = np.asarray(chunk.get("action.gripper", chunk.get("gripper")))[0].reshape(-1)
                dt_inf = time.time() - t_inf
                for t in range(min(a.exec_horizon, len(arm))):
                    target = np.concatenate([d.q_cmd(arm[t]), [float(grip[t])]])
                    target = d.clamp_rad(target)
                    step = np.clip(target - q_cmd_prev, -a.max_step, a.max_step)
                    target = q_cmd_prev + step
                    z, _ = d.tcp_z(target[:5])
                    if z < zmin:
                        target[:5] = q_cmd_prev[:5]          # floor-guard: не опускать
                    hits = d.self_hits(target, margin=0.02)
                    if hits and hits[0][2] < 0:              # самостолкновение — стоять
                        target[:5] = q_cmd_prev[:5]
                        n_selfcol += 1
                    elif hits:                               # сближение < 2 см — вдвое медленнее
                        target[:5] = q_cmd_prev[:5] + 0.5 * (target[:5] - q_cmd_prev[:5])
                    if not a.dry:
                        goals = {d.cal.joints[n]["id"]: d.raw_of(n, target[j]) for j, n in enumerate(NAMES)}
                        d.bus.sync_write(42, 2, goals)
                    q_cmd_prev = target
                    ticks += 1
                    time.sleep(max(0.0, 1.0 / HZ - 0.002))
                    if ticks % 3 == 0:
                        qr, rr = d.read()
                        d.publish(qr, rr)
                        lag = float(np.max(np.abs(qr[:5] - target[:5])))
                        if lag > 0.35:
                            d.hold()
                            raise GuardError(f"отставание {lag:.2f} рад — стоп")
                        zr, _ = d.tcp_z(qr[:5])
                        held_ticks = held_ticks + 3 if (qr[5] < 0.6 and zr > a.lift_z) else 0
                        fr = np.hstack([cv2.resize(top_frame_640(top), (640, 480)), wr])
                        cv2.putText(fr, f"a{k+1} t={time.time()-t0:4.1f}s inf={dt_inf*1000:.0f}ms "
                                        f"z={zr*100:.1f}cm grip={qr[5]:.2f}",
                                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (40, 220, 40), 2, cv2.LINE_AA)
                        vw.write(fr)
                        if held_ticks >= HZ:
                            lifted = True
                            break
                        # гибрид: политика довела руку до зависания над кубом — дальше алгоритм
                        q_hist.append(qr.copy())
                        q_hist = q_hist[-15:]
                        # передача алгоритму: кубик виден камере кисти вблизи (точка схвата ниже 9 см),
                        # губки ещё раскрыты; проверка не чаще раза в 10 тактов (детекция ~0,4 с)
                        if a.servo_grasp and not handover and qr[5] > 1.2 and 0.03 < zr < 0.09 \
                                and time.time() - t0 > 2.0 and ticks % 9 == 0:
                            fix = None
                            tcp_xy = d.fk_T(qr[:5])[:2, 3]
                            colors_try = a.servo_color.split(",") if a.servo_color else ("red", "green")
                            for color in colors_try:
                                if a.servo_color:
                                    r_ = cube_by_wrist_color(d, color, Xw)
                                    fix = None if r_ is None else (r_[0], r_[1])
                                else:
                                    fix = cube_by_wrist(d, color, Xw)
                                # куб под кистью: оценка не дальше 8 см от точки схвата и в зоне досягаемости
                                if fix is not None and (np.linalg.norm(fix[0] - tcp_xy) > 0.08
                                                        or not (0.14 <= fix[0][0] <= 0.32 and abs(fix[0][1]) <= 0.21)):
                                    print(f"  гибрид: оценка {color} {np.round(fix[0], 3).tolist()} отвергнута "
                                          f"(схват в {np.round(tcp_xy, 3).tolist()})", flush=True)
                                    fix = None
                                if fix is not None:
                                    break
                            if fix is not None:
                                handover = True
                                xy, yaw = fix
                                print(f"  гибрид: куб {color} по камере кисти в {np.round(xy, 3).tolist()} — "
                                      "доводка, спуск, смыкание, подъём алгоритмом", flush=True)
                                try:
                                    w = waypoints(xy)
                                    qm = d.q_model(qr)
                                    qm = move(d, ik, w["hover"], A.GRIP_WIDE, 1.0, qm, yaw, mode="grasp")
                                    qm = move(d, ik, w["low"], A.GRIP_WIDE, 1.5, qm, yaw, mode="grasp")
                                    d.goto(np.concatenate([d.q_cmd(qm), [A.GRIP_CLOSED]]), 0.7, mode="grasp")
                                    d.settle(0.5, 0.3)
                                    move(d, ik, w["above"], A.GRIP_CLOSED, 1.5, qm, yaw, mode="grasp")
                                    d.settle(1.0, 0.5)
                                    qr, _ = d.read()
                                    lifted = bool(qr[5] < 0.6 and d.tcp_z(qr[:5])[0] > a.lift_z)
                                except GuardError as e:
                                    print("  гибрид: guard:", e, flush=True)
                                    why = f"hybrid guard: {e}"
                                break
                if handover:
                    break
            vw.release()
            qf, _ = d.read()
            if a.servo_color:
                # цветной куб в губках: пятно занимает заметную часть низа кадра кисти
                wr_now = A.grab(A.CAM_WR)
                h_ = wr_now.shape[0]
                frac = max(float((color_mask(wr_now, c)[int(h_ * 0.45):] > 0).mean()) for c in a.servo_color.split(","))
                confirmed = bool(lifted and frac > 0.12)
            else:
                confirmed = bool(cube_in_gripper("red")) or bool(cube_in_gripper("green")) if lifted else False
            res = {"attempt": k + 1, "lifted_by_fk": lifted, "cube_in_gripper": confirmed,
                   "seconds": round(time.time() - t0, 1),
                   "ticks": ticks, "tcp_z": round(d.tcp_z(qf[:5])[0], 3), "grip": round(float(qf[5]), 2),
                   "selfcol_blocked_ticks": n_selfcol, "handover": handover, "why": why}
            # связка: поднятый кубик несёт в коробку алгоритм
            res["in_box"] = None
            if box_xy is not None and lifted:
                try:
                    res["carry"] = carry_to_box(d, ik, box_xy)
                    after = look(Xt)
                    seen = [c for c, T in after.items() if np.linalg.norm(T[:2, 3] - box_xy) < RIM_HALF]
                    gone = [c for c in cubes0 if c not in after]
                    res["in_box"] = bool(seen or gone)      # кубик виден в контуре коробки или пропал со стола
                    print(f"  перенос: над коробкой ошибка {res['carry'].get('over_box_err_mm')} мм, "
                          f"{'В КОРОБКЕ' if res['in_box'] else 'мимо'}", flush=True)
                except GuardError as e:
                    print("  перенос не удался:", e, flush=True)
                    res["carry_error"] = str(e)
                qf, _ = d.read()
            # вернуть куб на стол там, где рука, и раскрыть
            if qf[5] < 0.6:
                F = d.fk_T(qf[:5])
                q_low = plan(ik, [*F[:2, 3], A.CUBE_HALF + 0.012], d.q_model(qf), None)
                if q_low is not None:
                    try:
                        d.goto(np.concatenate([d.q_cmd(q_low), [A.GRIP_CLOSED]]), 2.0, mode="grasp")
                    except GuardError as e:
                        print("  опустить не удалось:", e)
                # раскрыть на месте прямой командой: goto здесь отказывает, если рука
                # уже у пола (z ровно на пороге floor-guard, 10.09)
                qn, _ = d.read()
                goals = {d.cal.joints[n]["id"]: d.raw_of(n, v) for n, v in zip(NAMES, [*qn[:5], A.GRIP_WIDE])}
                d.bus.sync_write(42, 2, goals)
                time.sleep(0.8)
            try:
                d.lift()
                d.goto(A.PARK + [A.GRIP_WIDE], 4.0, mode="grasp")
                d.settle(1.0, 0.5)
            except GuardError as e:
                print("  возврат в парковку:", e)
                break
            # второе подтверждение: куб, который несли, лежит теперь не там, где был
            # (метка у самого объектива камеры кисти читается ненадёжно)
            moved_mm = None
            try:
                cubes1 = {c: T[:2, 3] for c, T in look(Xt).items()}
                shifts = [float(np.linalg.norm(cubes1[c] - cubes0[c])) * 1000 for c in cubes0 if c in cubes1]
                moved_mm = round(max(shifts), 1) if shifts else None
            except Exception:
                pass
            res["cube_moved_mm"] = moved_mm
            res["success"] = bool(lifted and (confirmed or (moved_mm is not None and moved_mm > 40) or res.get("in_box")))
            results.append(res)
            print(f"попытка {k+1}/{a.attempts}: {'УСПЕХ' if res['success'] else 'промах'} "
                  f"(подъём {lifted}, куб в губках {confirmed}, куб переместился {moved_mm} мм"
                  f"{', гибрид' if handover else ''}, {res['seconds']} с)", flush=True)
    finally:
        d.close()
    n = sum(r["success"] for r in results)
    out = a.video_dir / f"run-{tag}.json"
    out.write_text(json.dumps({"policy": a.policy, "task": a.task, "seconds": a.seconds,
                               "success": n, "attempts": len(results), "trials": results},
                              ensure_ascii=False, indent=2))
    print(f"\nИТОГ: {n}/{len(results)} успешных; отчёт {out}; видео {a.video_dir}/run-{tag}-a*.mp4")


if __name__ == "__main__":
    main()
