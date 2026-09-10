#!/usr/bin/env python3
"""Проверка cubes_2color.3mf: структура + «Bambu Studio его понимает».

Две ступени:
1) без Bambu: файл читается, в model_settings.config у каждого куба две части
   (extruder 1 и 2), id частей совпадают с компонентами сборок в 3dmodel.model,
   меши частей замкнуты по объёму;
2) с установленным Bambu Studio: скормить файл его CLI на реэкспорт
   (--export-3mf) и убедиться, что студия сохранила обе части каждого куба с
   теми же extruder — то есть прочитала назначения цветов.

Мультицветный СЛАЙС в headless-CLI не проверить: эта версия Bambu Studio в
CLI-режиме не генерирует смены филамента даже для фирменного мультицветного
калибровочного проекта (проверено на auto_pa_line_dual.3mf с профилем H2D),
так что печатные смены цвета видны только в GUI.

Аргументом можно дать другой файл: check_2color.py out/cube_1color.3mf —
одноцветный куб (одно тело, extruder 1) проходит те же две ступени.
"""

import re
import shutil
import sys
import subprocess
import tempfile
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

BAMBU = Path("/Applications/BambuStudio.app/Contents/MacOS/BambuStudio")
PROFILES = Path("/Applications/BambuStudio.app/Contents/Resources/profiles/BBL")
SRC = Path(sys.argv[1]) if len(sys.argv) > 1 else \
    Path(__file__).parent / "out" / "cubes_2color.3mf"
NS = {"m": "http://schemas.microsoft.com/3dmanufacturing/core/2015/02"}


def part_extruders(path_3mf):
    """{имя объекта: {имя части: extruder}} из model_settings.config."""
    with zipfile.ZipFile(path_3mf) as z:
        cfg = ET.fromstring(z.read("Metadata/model_settings.config"))
    out = {}
    for obj in cfg.findall("object"):
        meta = {m.get("key"): m.get("value") for m in obj.findall("metadata")}
        parts = {}
        for part in obj.findall("part"):
            pm = {m.get("key"): m.get("value") for m in part.findall("metadata")}
            # одночастный объект студия сохраняет с extruder на уровне объекта
            parts[pm.get("name", part.get("id"))] = (part.get("id"),
                                                     pm.get("extruder", meta.get("extruder")))
        out[meta.get("name", obj.get("id"))] = parts
    return out


def structural_check():
    with zipfile.ZipFile(SRC) as z:
        model = ET.fromstring(z.read("3D/3dmodel.model"))
    meshes, assemblies = {}, {}
    for obj in model.findall(".//m:resources/m:object", NS):
        comps = obj.findall("m:components/m:component", NS)
        if comps:
            assemblies[obj.get("id")] = [c.get("objectid") for c in comps]
        else:
            verts = [(float(v.get("x")), float(v.get("y")), float(v.get("z")))
                     for v in obj.findall(".//m:vertex", NS)]
            tris = [(int(t.get("v1")), int(t.get("v2")), int(t.get("v3")))
                    for t in obj.findall(".//m:triangle", NS)]
            vol = sum(
                (verts[a][0] * (verts[b][1] * verts[c][2] - verts[c][1] * verts[b][2])
                 - verts[b][0] * (verts[a][1] * verts[c][2] - verts[c][1] * verts[a][2])
                 + verts[c][0] * (verts[a][1] * verts[b][2] - verts[b][1] * verts[a][2]))
                for a, b, c in tris) / 6.0
            meshes[obj.get("id")] = vol
    cubes = part_extruders(SRC)
    assert len(assemblies) == len(cubes) >= 1, (assemblies, cubes)
    for cube, parts in cubes.items():
        ids = {pid for pid, _ in parts.values()}
        extruders = sorted(ext for _, ext in parts.values())
        [asm_ids] = [v for v in assemblies.values() if set(v) == ids]
        assert extruders in (["1"], ["1", "2"]), f"{cube}: extruder'ы {extruders}"
        for pid in asm_ids:
            assert meshes[pid] > 1000, f"часть {pid} куба {cube}: объём {meshes[pid]}"
        print(f"  {cube}: extruder'ы {extruders}, объёмы "
              f"{[round(meshes[p] / 1000, 2) for p in asm_ids]} см3")
    print(f"структура OK: {len(cubes)} объект(ов), части с назначенными "
          f"extruder'ами, объёмы мешей положительные")
    return cubes


def bambu_roundtrip(expected):
    machine = PROFILES / "machine" / "Bambu Lab A1 0.4 nozzle.json"
    process = PROFILES / "process" / "0.20mm Standard @BBL A1.json"
    filament = PROFILES / "filament" / "Generic PLA @BBL A1.json"
    work = Path(tempfile.mkdtemp(prefix="cubes2color-"))
    cmd = [str(BAMBU), "--arrange", "0", "--orient", "0",
           "--load-settings", f"{machine};{process}",
           "--load-filaments", f"{filament};{filament}",
           "--outputdir", str(work), "--export-3mf", "roundtrip.3mf", str(SRC)]
    res = subprocess.run(cmd, capture_output=True, text=True)
    exported = work / "roundtrip.3mf"
    assert exported.is_file(), \
        f"Bambu Studio не смог перечитать файл (exit {res.returncode})"
    got = part_extruders(exported)
    for cube, parts in expected.items():
        got_ext = {name: ext for name, (_, ext) in got[cube].items()}
        want_ext = {name: ext for name, (_, ext) in parts.items()}
        assert got_ext == want_ext, f"{cube}: Bambu вернул {got_ext}, ждали {want_ext}"
    print(f"Bambu Studio прочитал и сохранил назначения: "
          f"{ {c: {n: e for n, (_, e) in p.items()} for c, p in got.items()} }")
    shutil.rmtree(work)


def main():
    expected = structural_check()
    if BAMBU.is_file():
        bambu_roundtrip(expected)
    else:
        print(f"Bambu Studio не найден ({BAMBU}) — пропущен только roundtrip")
    print("OK")


if __name__ == "__main__":
    main()
