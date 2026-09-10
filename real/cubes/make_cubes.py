#!/usr/bin/env python3
"""Двухцветные кубы с ArUco-метками для реального стенда (печать FDM).

Каждый куб — 7 деталей двух цветов:
  * белая: куб с фигурными карманами глубиной INLAY_T на всех 6 гранях
    (белое поле + белые биты маркера — единое тело);
  * чёрные: 6 плоских вкладышей-«ключей» (чёрная рамка маркера + чёрные биты
    данных одной связной деталью), вклеиваются в карманы заподлицо.

Чёрная область ArUco содержит замкнутую рамку, поэтому из словаря берутся
только ID, у которых все чёрные клетки связны по сторонам, — иначе вкладыш
развалился бы на части. Форма кармана уникальна для каждого ID и не имеет
поворотной симметрии (гарантия словаря), так что чужой или повёрнутый вкладыш
физически не встанет — сборку не перепутать.

python3 make_cubes.py  ->  out/: STL по цветам, cubes_2color.3mf, manifest.json
(позы меток для CV), preview_*.png (какая грань — какой ID) + самопроверка
детекцией; плюс cube_1color.3mf — тот же куб 28 мм одним телом без меток, для
печати цветными филаментами (реальные цветные кубы для проверки политики).
"""

import json
import struct
import zipfile
from pathlib import Path

import cv2
import numpy as np

CUBE = 28.0      # ребро куба, мм — как в симе (полуразмер 0.014 м в pick_place.xml)
N = 8            # клеток на грань: 6 (маркер 4x4 + рамка) + 2 (белое поле)
M = CUBE / N     # модуль, мм (3.5)
INLAY_T = 0.8    # толщина вкладыша = глубина кармана (4 слоя по 0.2)
CLEAR = 0.2      # зазор: наружные стороны вкладыша усаживаются на эту величину
OVERLAP = 0.4    # заглубление белых клеток в ядро (сварка тел в слайсере)
GAP = 6.0        # шаг раскладки вкладышей в общем STL
N_CUBES = 2
DICTS = ["DICT_4X4_50", "DICT_4X4_100", "DICT_4X4_250"]

# (имя, right, up, normal): right = up x normal, вид на грань снаружи.
# Боковые грани: «верх» маркера смотрит на верх куба.
FACES = [
    ("+Z", (1, 0, 0), (0, 1, 0), (0, 0, 1)),
    ("-Z", (1, 0, 0), (0, -1, 0), (0, 0, -1)),
    ("+X", (0, 1, 0), (0, 0, 1), (1, 0, 0)),
    ("-X", (0, -1, 0), (0, 0, 1), (-1, 0, 0)),
    ("+Y", (-1, 0, 0), (0, 0, 1), (0, 1, 0)),
    ("-Y", (1, 0, 0), (0, 0, 1), (0, -1, 0)),
]


def marker_grid(dic, mid):
    """8x8 карта грани, True = чёрная клетка. grid[gy, gx]: gx вдоль right
    (0 слева), gy вдоль up (0 снизу); внешний ряд — белое поле."""
    img = cv2.aruco.generateImageMarker(dic, mid, 6)  # 6x6, строка 0 — верх
    grid = np.zeros((N, N), bool)
    grid[1:7, 1:7] = (img == 0)[::-1, :]  # переворот: строки img идут сверху вниз
    return grid


def black_connected(grid):
    """Все чёрные клетки связны по сторонам (иначе вкладыш распадётся)."""
    ys, xs = np.nonzero(grid)
    todo = [(ys[0], xs[0])]
    seen = {todo[0]}
    while todo:
        y, x = todo.pop()
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            p = (y + dy, x + dx)
            if 0 <= p[0] < N and 0 <= p[1] < N and grid[p] and p not in seen:
                seen.add(p)
                todo.append(p)
    return len(seen) == len(ys)


def pick_ids(need):
    for name in DICTS:
        dic = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, name))
        ids = [i for i in range(dic.bytesList.shape[0])
               if black_connected(marker_grid(dic, i))]
        if len(ids) >= need:
            return name, dic, ids[:need]
    raise SystemExit(f"не нашлось {need} связных ID даже в {DICTS[-1]}")


# --- STL: боксы -> треугольники -> binary ---------------------------------

def box_tris(lo, hi, basis=np.eye(3)):
    """12 треугольников бокса [lo, hi] в осях basis (столбцы: right, up, normal)."""
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    corners = np.array([[x, y, z] for x in (lo[0], hi[0])
                        for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
    corners = corners @ np.asarray(basis, float).T
    # индексы: bit0 -> x, bit1 -> y, bit2 -> z (см. corners: x внешний цикл)
    idx = lambda x, y, z: x * 4 + y * 2 + z
    quads = [  # (четыре вершины CCW снаружи)
        (idx(0, 0, 0), idx(0, 0, 1), idx(0, 1, 1), idx(0, 1, 0)),  # -x
        (idx(1, 0, 0), idx(1, 1, 0), idx(1, 1, 1), idx(1, 0, 1)),  # +x
        (idx(0, 0, 0), idx(1, 0, 0), idx(1, 0, 1), idx(0, 0, 1)),  # -y
        (idx(0, 1, 0), idx(0, 1, 1), idx(1, 1, 1), idx(1, 1, 0)),  # +y
        (idx(0, 0, 0), idx(0, 1, 0), idx(1, 1, 0), idx(1, 0, 0)),  # -z
        (idx(0, 0, 1), idx(1, 0, 1), idx(1, 1, 1), idx(0, 1, 1)),  # +z
    ]
    tris = []
    for a, b, c, d in quads:
        tris.append((corners[a], corners[b], corners[c]))
        tris.append((corners[a], corners[c], corners[d]))
    return tris


def write_stl(path, tris):
    with open(path, "wb") as f:
        f.write(b"\0" * 80)
        f.write(struct.pack("<I", len(tris)))
        for a, b, c in tris:
            n = np.cross(b - a, c - a)
            n = n / (np.linalg.norm(n) or 1.0)
            f.write(struct.pack("<12fH", *n, *a, *b, *c, 0))


def mesh_volume(tris):
    return sum(np.dot(a, np.cross(b, c)) for a, b, c in tris) / 6.0


# --- геометрия деталей ----------------------------------------------------

def cell_bounds(gx, gy):
    u0 = -CUBE / 2 + gx * M
    v0 = -CUBE / 2 + gy * M
    return u0, u0 + M, v0, v0 + M


def white_body(grids):
    """Белая деталь: ядро + белые клетки всех граней (с нахлёстом в ядро)."""
    core = CUBE / 2 - INLAY_T
    tris = box_tris([-core] * 3, [core] * 3)
    for (_, right, up, normal), grid in zip(FACES, grids):
        basis = np.column_stack([right, up, normal]).astype(float)
        for gy in range(N):
            for gx in range(N):
                if grid[gy, gx]:
                    continue
                u0, u1, v0, v1 = cell_bounds(gx, gy)
                tris += box_tris([u0, v0, core - OVERLAP],
                                 [u1, v1, CUBE / 2], basis)
    return tris


def inlay_rects(grid, clear=CLEAR):
    """Прямоугольники вкладыша: точный офсет ортогональной области — сторона,
    граничащая с белым, усаживается на clear; между чёрными — стык внахлёст."""
    black = lambda gx, gy: 0 <= gx < N and 0 <= gy < N and grid[gy, gx]
    rects = []
    for gy in range(N):
        for gx in range(N):
            if not grid[gy, gx]:
                continue
            u0, u1, v0, v1 = cell_bounds(gx, gy)
            u0 += clear if not black(gx - 1, gy) else -0.005
            u1 -= clear if not black(gx + 1, gy) else -0.005
            v0 += clear if not black(gx, gy - 1) else -0.005
            v1 -= clear if not black(gx, gy + 1) else -0.005
            rects.append((u0, u1, v0, v1))
    return rects


def black_body(grids):
    """Чёрные области всех граней на своих местах (сборка, зазор не нужен)."""
    tris = []
    for (_, right, up, normal), grid in zip(FACES, grids):
        basis = np.column_stack([right, up, normal]).astype(float)
        for u0, u1, v0, v1 in inlay_rects(grid, clear=0.0):
            tris += box_tris([u0, v0, CUBE / 2 - INLAY_T],
                             [u1, v1, CUBE / 2], basis)
    return tris


def inlay_tris(grid, x_off):
    """Вкладыш лицом ВНИЗ (лицо на z=0 — печатать как есть, лицом к пластине);
    вид сверху в STL поэтому зеркален виду грани снаружи."""
    tris = []
    for u0, u1, v0, v1 in inlay_rects(grid):
        tris += box_tris([x_off - u1, v0, 0], [x_off - u0, v1, INLAY_T])
    return tris


# --- двухцветный 3MF для Bambu Studio (AMS) -------------------------------

_3MF_NS = "http://schemas.microsoft.com/3dmanufacturing/core/2015/02"


def weld(tris):
    """Треугольники -> (вершины, грани) со сваркой вершин."""
    verts, faces = {}, []
    for tri in tris:
        faces.append([verts.setdefault(tuple(round(c, 4) for c in p), len(verts))
                      for p in tri])
    return list(verts), faces


def _mesh_object_xml(oid, tris):
    v, f = weld(tris)
    out = [f'  <object id="{oid}" type="model">', "   <mesh>", "    <vertices>"]
    out += [f'     <vertex x="{x}" y="{y}" z="{z}"/>' for x, y, z in v]
    out += ["    </vertices>", "    <triangles>"]
    out += [f'     <triangle v1="{a}" v2="{b}" v3="{c}"/>' for a, b, c in f]
    out += ["    </triangles>", "   </mesh>", "  </object>"]
    return out


def write_3mf_2color(path, cubes):
    """cubes: [(имя, белые_треугольники, чёрные_треугольники)].
    Каждый куб — объект из двух тел; extruder 1 — белый, 2 — чёрный
    (Metadata/model_settings.config, как в проектах Bambu Studio). Пустой
    чёрный список — одноцветный объект: одно тело на extruder 1."""
    model = ['<?xml version="1.0" encoding="UTF-8"?>',
             f'<model unit="millimeter" xml:lang="en-US" xmlns="{_3MF_NS}">',
             ' <metadata name="Application">BambuStudio</metadata>',
             " <resources>"]
    cfg = ['<?xml version="1.0" encoding="UTF-8"?>', "<config>"]
    items = []
    step = CUBE + 10
    x0 = 128 - step * (len(cubes) - 1) / 2
    oid = 0
    for i, (name, white, black) in enumerate(cubes):
        parts = [(f"{name}_white", white, 1), (f"{name}_black", black, 2)] \
            if black else [(name, white, 1)]
        pids = [oid + 1 + k for k in range(len(parts))]
        aid = oid = pids[-1] + 1
        cfg += [f' <object id="{aid}">', f'  <metadata key="name" value="{name}"/>']
        for pid, (pname, tris, ext) in zip(pids, parts):
            model += _mesh_object_xml(pid, tris)
            cfg += [f'  <part id="{pid}" subtype="normal_part">',
                    f'   <metadata key="name" value="{pname}"/>',
                    f'   <metadata key="extruder" value="{ext}"/>', "  </part>"]
        cfg.append(" </object>")
        model += [f'  <object id="{aid}" type="model">', "   <components>"]
        model += [f'    <component objectid="{p}"/>' for p in pids]
        model += ["   </components>", "  </object>"]
        items.append(f'  <item objectid="{aid}" transform="1 0 0 0 1 0 0 0 1 '
                     f'{x0 + i * step} 128 {CUBE / 2}" printable="1"/>')
    model += [" </resources>", " <build>"] + items + [" </build>", "</model>"]
    cfg.append("</config>")

    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml",
                   '<?xml version="1.0" encoding="UTF-8"?>\n'
                   '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">\n'
                   ' <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>\n'
                   ' <Default Extension="model" ContentType="application/vnd.ms-package.3dmanufacturing-3dmodel+xml"/>\n'
                   "</Types>")
        z.writestr("_rels/.rels",
                   '<?xml version="1.0" encoding="UTF-8"?>\n'
                   '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">\n'
                   ' <Relationship Target="/3D/3dmodel.model" Id="rel0" '
                   'Type="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel"/>\n'
                   "</Relationships>")
        z.writestr("3D/3dmodel.model", "\n".join(model))
        z.writestr("Metadata/model_settings.config", "\n".join(cfg))


# --- превью и самопроверка ------------------------------------------------

S = 20  # px/мм

def face_image(grid, rects=None):
    """Растр грани (вид снаружи, up — вверх картинки). rects: рисовать
    фактические прямоугольники вкладыша вместо номинальных клеток."""
    px = int(CUBE * S)
    img = np.full((px, px), 255, np.uint8)
    to_px = lambda u, v: (int(round((u + CUBE / 2) * S)),
                          int(round((CUBE / 2 - v) * S)))  # v вверх -> строка вниз
    if rects is None:
        rects = [cell_bounds(gx, gy) for gy in range(N) for gx in range(N)
                 if grid[gy, gx]]
    for u0, u1, v0, v1 in rects:
        x0, y1 = to_px(u0, v0)
        x1, y0 = to_px(u1, v1)
        img[y0:y1, x0:x1] = 0
    return img


def detect_ok(img, dic, mid):
    det = cv2.aruco.ArucoDetector(dic, cv2.aruco.DetectorParameters())
    _, ids, _ = det.detectMarkers(img)
    return ids is not None and list(ids.ravel()) == [mid]


def preview_sheet(path, cube_name, dic, face_ids):
    tiles = []
    for (fname, *_), mid in zip(FACES, face_ids):
        tile = cv2.cvtColor(face_image(marker_grid(dic, mid)), cv2.COLOR_GRAY2BGR)
        tile = cv2.copyMakeBorder(tile, 6, 70, 6, 6, cv2.BORDER_CONSTANT,
                                  value=(190, 190, 190))
        cv2.putText(tile, f"{fname}  id {mid}", (14, tile.shape[0] - 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (30, 30, 30), 3, cv2.LINE_AA)
        tiles.append(tile)
    sheet = cv2.hconcat(tiles)
    head = np.full((90, sheet.shape[1], 3), 255, np.uint8)
    cv2.putText(head, f"cube {cube_name}: faces outside view; inlays in STL "
                "lie face-down in this order", (14, 58),
                cv2.FONT_HERSHEY_SIMPLEX, 1.6, (30, 30, 30), 3, cv2.LINE_AA)
    cv2.imwrite(str(path), cv2.vconcat([head, sheet]))


# --- сборка ---------------------------------------------------------------

def main():
    out = Path(__file__).parent / "out"
    out.mkdir(exist_ok=True)
    dict_name, dic, ids = pick_ids(N_CUBES * 6)
    print(f"словарь {dict_name}, связные ID: {ids}")

    assembled = []
    manifest = {
        "units": "mm",
        "cube_size_mm": CUBE, "module_mm": M, "marker_size_mm": 6 * M,
        "inlay_thickness_mm": INLAY_T, "clearance_mm": CLEAR,
        "dictionary": dict_name,
        "face_convention": "right = up x normal; вид на грань снаружи, "
                           "строка 0 generateImageMarker — сторона up",
        "cubes": {},
    }

    for ci in range(N_CUBES):
        name = chr(ord("A") + ci)
        face_ids = ids[ci * 6:ci * 6 + 6]
        grids = [marker_grid(dic, mid) for mid in face_ids]

        for (fname, *_), mid, grid in zip(FACES, face_ids, grids):
            assert black_connected(grid), f"{name}{fname}: id {mid} не связан"
            assert detect_ok(face_image(grid), dic, mid), \
                f"{name}{fname}: id {mid} не детектится (номинал)"
            assert detect_ok(face_image(grid, inlay_rects(grid)), dic, mid), \
                f"{name}{fname}: id {mid} не детектится (с зазором {CLEAR})"

        body = white_body(grids)
        write_stl(out / f"cube{name}_body_white.stl", body)
        inlays = []
        for k, grid in enumerate(grids):
            inlays += inlay_tris(grid, k * (CUBE + GAP))
        write_stl(out / f"cube{name}_inlays_black.stl", inlays)
        vol_b, vol_i = mesh_volume(body), mesh_volume(inlays)
        assert vol_b > 0 and vol_i > 0, "нормали смотрят внутрь"
        preview_sheet(out / f"preview_cube{name}.png", name, dic, face_ids)
        blacks = black_body(grids)
        assert mesh_volume(blacks) > 0
        assembled.append((f"cube{name}", body, blacks))

        manifest["cubes"][name] = {"faces": [
            {"face": f[0], "id": mid, "right": list(f[1]), "up": list(f[2]),
             "normal": list(f[3]),
             "center_mm": [CUBE / 2 * c for c in f[3]]}
            for f, mid in zip(FACES, face_ids)]}
        print(f"куб {name}: ID {face_ids}, белая деталь ~{vol_b / 1000:.1f} см3, "
              f"вкладыши ~{vol_i / 1000:.2f} см3, {len(body) + len(inlays)} треуг.")

    write_3mf_2color(out / "cubes_2color.3mf", assembled)
    print(f"cubes_2color.3mf: оба куба одной моделью, extruder 1 = белый, "
          f"2 = чёрный")
    write_3mf_2color(out / "cube_1color.3mf",
                     [("cube", box_tris([-CUBE / 2] * 3, [CUBE / 2] * 3), [])])
    print("cube_1color.3mf: тот же куб 28 мм одним телом без меток — печатать "
          "цветными филаментами")
    (out / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2))
    print(f"самопроверка пройдена; всё в {out}/")


if __name__ == "__main__":
    main()
