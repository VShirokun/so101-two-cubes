#!/usr/bin/env python3
"""Коробка-приёмник для реального стенда (печать FDM) — двойник сим-коробки.

Геометрия повторяет bin из mlsim/models/so101/pick_place.xml ОДИН В ОДИН:
внешний след 110x110 мм, стенка 8 мм, высота борта 44 мм, дно 8 мм — эти
числа входят в критерий успеха (mlsim/criteria.py: верх борта 0.044, кубик
на дне z = 0.022), менять их значит менять протокол оценки.

В дно встроена крупная ArUco-метка (модуль 10 мм, маркер 60 мм) тем же
приёмом, что у кубов: карман глубиной 0.8 мм + цельный чёрный вкладыш
(ID со связной чёрной областью). По метке CV-стенд узнаёт позу коробки
той же камерой, что ведёт кубы, — это нужно алгоритмическому опусканию
кубика в коробку. MARKER = False даёт гладкое дно без кармана.

python3 make_box.py -> out/: STL по цветам, единый двухцветный 3MF
(тело — экструдер 1, метка — экструдер 2), превью, manifest.json + проверки.
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1] / "cubes"))
from make_cubes import (CLEAR, INLAY_T, OVERLAP, black_connected, box_tris,
                        detect_ok, marker_grid, mesh_volume,
                        write_3mf_2color, write_stl)

# --- габариты сим-коробки, мм (менять только вместе с pick_place.xml) ------
OUTER = 110.0
WALL = 8.0
HEIGHT = 44.0
FLOOR_T = 8.0
INNER = OUTER - 2 * WALL          # 94

MARKER = True
MODULE = 10.0                     # клетка метки; зона с полем 8 клеток = 80 мм
NCELL = 8
ZONE = MODULE * NCELL             # 80 < INNER=94

CUBE_IDS_USED = {1, 2, 3, 4, 5, 6, 7, 10, 11, 13, 14, 15}  # кубы A и B


def pick_box_id():
    dic = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    for i in range(dic.bytesList.shape[0]):
        if i in CUBE_IDS_USED:
            continue
        if black_connected(marker_grid(dic, i)):
            return dic, i
    raise SystemExit("нет свободного связного ID в DICT_4X4_50")


def cell_xy(gx, gy):
    x0 = -ZONE / 2 + gx * MODULE
    y0 = -ZONE / 2 + gy * MODULE
    return x0, x0 + MODULE, y0, y0 + MODULE


def body_tris(grid):
    """Тело коробки: дно (с клеточным верхним слоем и карманом под метку,
    если MARKER) + четыре борта. Начало координат — центр дна, z=0 — низ."""
    tris = []
    z_top = FLOOR_T
    if MARKER and grid is not None:
        base_t = FLOOR_T - INLAY_T
        tris += box_tris([-OUTER / 2, -OUTER / 2, 0],
                         [OUTER / 2, OUTER / 2, base_t])
        # сплошной верхний слой дна вне зоны метки (рамка из 4 боксов)
        z0, z1 = base_t - OVERLAP, FLOOR_T
        h = OUTER / 2
        zn = ZONE / 2
        tris += box_tris([-h, -h, z0], [h, -zn, z1])
        tris += box_tris([-h, zn, z0], [h, h, z1])
        tris += box_tris([-h, -zn, z0], [-zn, zn, z1])
        tris += box_tris([zn, -zn, z0], [h, zn, z1])
        # клеточный слой зоны метки: белые клетки, чёрные — карман
        for gy in range(NCELL):
            for gx in range(NCELL):
                if grid[gy, gx]:
                    continue
                x0, x1, y0, y1 = cell_xy(gx, gy)
                tris += box_tris([x0, y0, z0], [x1, y1, z1])
    else:
        tris += box_tris([-OUTER / 2, -OUTER / 2, 0],
                         [OUTER / 2, OUTER / 2, FLOOR_T])
    o, i, zt = OUTER / 2, INNER / 2, HEIGHT
    zb = FLOOR_T - OVERLAP
    tris += box_tris([-o, -o, zb], [o, -i, zt])
    tris += box_tris([-o, i, zb], [o, o, zt])
    tris += box_tris([-o, -i, zb], [-i, i, zt])
    tris += box_tris([i, -i, zb], [o, i, zt])
    return tris


def marker_rects(grid, clear):
    """Прямоугольники вкладыша в клетках зоны (офсет как у кубов)."""
    black = lambda gx, gy: 0 <= gx < NCELL and 0 <= gy < NCELL and grid[gy, gx]
    rects = []
    for gy in range(NCELL):
        for gx in range(NCELL):
            if not grid[gy, gx]:
                continue
            x0, x1, y0, y1 = cell_xy(gx, gy)
            x0 += clear if not black(gx - 1, gy) else -0.005
            x1 -= clear if not black(gx + 1, gy) else -0.005
            y0 += clear if not black(gx, gy - 1) else -0.005
            y1 -= clear if not black(gx, gy + 1) else -0.005
            rects.append((x0, x1, y0, y1))
    return rects


def inlay_tris_flat(grid):
    """Вкладыш отдельной деталью, лицом ВНИЗ (печатать как лежит)."""
    tris = []
    for x0, x1, y0, y1 in marker_rects(grid, CLEAR):
        tris += box_tris([-x1, y0, 0], [-x0, y1, INLAY_T])
    return tris


def inlay_tris_assembled(grid):
    """Чёрные клетки на своём месте в дне (для двухцветного 3MF)."""
    tris = []
    for x0, x1, y0, y1 in marker_rects(grid, 0.0):
        tris += box_tris([x0, y0, FLOOR_T - INLAY_T], [x1, y1, FLOOR_T])
    return tris


def zone_image(grid, rects=None, s=6):
    px = int(ZONE * s)
    img = np.full((px, px), 255, np.uint8)
    if rects is None:
        rects = [cell_xy(gx, gy) for gy in range(NCELL) for gx in range(NCELL)
                 if grid[gy, gx]]
    for x0, x1, y0, y1 in rects:
        c0 = int(round((x0 + ZONE / 2) * s))
        c1 = int(round((x1 + ZONE / 2) * s))
        r1 = int(round((ZONE / 2 - y0) * s))
        r0 = int(round((ZONE / 2 - y1) * s))
        img[r0:r1, c0:c1] = 0
    return img


def main():
    out = Path(__file__).parent / "out"
    out.mkdir(exist_ok=True)
    dic, mid = pick_box_id()
    grid = marker_grid(dic, mid)
    print(f"метка коробки: DICT_4X4_50 id {mid} (кубы заняли {sorted(CUBE_IDS_USED)})")

    assert ZONE < INNER, "зона метки не помещается в дно"
    assert detect_ok(zone_image(grid), dic, mid), "метка не детектится (номинал)"
    assert detect_ok(zone_image(grid, marker_rects(grid, CLEAR)), dic, mid), \
        f"метка не детектится с зазором {CLEAR}"

    body = body_tris(grid if MARKER else None)
    write_stl(out / "bin_body.stl", body)
    vol_body = mesh_volume(body)
    files = ["bin_body.stl"]
    if MARKER:
        flat = inlay_tris_flat(grid)
        write_stl(out / "bin_marker_inlay.stl", flat)
        asm = inlay_tris_assembled(grid)
        assert mesh_volume(flat) > 0 and mesh_volume(asm) > 0
        write_3mf_2color(out / "bin_2color.3mf", [("bin", body, asm)])
        files += ["bin_marker_inlay.stl", "bin_2color.3mf"]

    tile = cv2.cvtColor(zone_image(grid), cv2.COLOR_GRAY2BGR)
    pad = int((INNER - ZONE) / 2 * 6)
    tile = cv2.copyMakeBorder(tile, pad, pad, pad, pad, cv2.BORDER_CONSTANT,
                              value=(235, 235, 235))
    wall = int(WALL * 6)
    tile = cv2.copyMakeBorder(tile, wall, wall + 60, wall, wall,
                              cv2.BORDER_CONSTANT, value=(140, 140, 140))
    cv2.putText(tile, f"bin {OUTER:.0f}x{OUTER:.0f}x{HEIGHT:.0f}, wall {WALL:.0f}, "
                f"marker id {mid} ({MODULE * 6:.0f} mm)", (12, tile.shape[0] - 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (30, 30, 30), 1, cv2.LINE_AA)
    cv2.imwrite(str(out / "preview_bin.png"), tile)

    manifest = {
        "units": "mm",
        "outer": [OUTER, OUTER, HEIGHT], "wall": WALL, "floor": FLOOR_T,
        "inner": [INNER, INNER, HEIGHT - FLOOR_T],
        "sim_match": "mlsim/models/so101/pick_place.xml bin: след 0.110, "
                     "борт до z=0.044, дно до z=0.008 — критерий criteria.py "
                     "переносится без правок",
        "marker": None if not MARKER else {
            "dictionary": "DICT_4X4_50", "id": mid,
            "marker_size_mm": MODULE * 6, "module_mm": MODULE,
            "center_in_bin_mm": [0, 0, FLOOR_T],
            "normal": [0, 0, 1], "up": [0, 1, 0], "right": [1, 0, 0],
            "note": "up метки = +Y коробки; поза коробки в мире = "
                    "поза метки, сдвинутая на -floor по z",
        },
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2))
    print(f"тело ~{vol_body / 1000:.0f} см3 solid (при 15% заполнении ~"
          f"{vol_body / 1000 * 1.24 * 0.35:.0f} г PLA); файлы: {files}")
    print("самопроверка пройдена")


if __name__ == "__main__":
    main()
