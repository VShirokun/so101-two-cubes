"""ArUco-кубы в симуляции: сцена, текстуры и оценка позы кубика по камере.

Симуляционный двойник реального стенда авто-сбора датасета: на грани кубиков
натянуты те же метки, что на напечатанных кубах (real/cubes: DICT_4X4_50,
куб A/red — ID 1-6, куб B/green — ID 7,10,11,13,14,15; маркер занимает 6/8
грани, белое поле — 1 клетку). Поза кубика восстанавливается ТОЛЬКО из
пикселей верхней CV-камеры: детекция меток -> PnP по углам -> перевод в мир
через известную позу камеры (в реале её даст калибровка по доске на столе).
"""

import hashlib
from pathlib import Path
import xml.etree.ElementTree as ET

import cv2
import mujoco
import numpy as np

CUBE_HALF = 0.014
MARKER_SIZE = 2 * CUBE_HALF * 6 / 8     # 21 мм: 6 клеток из 8 по грани
DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
CUBE_IDS = {"red": [1, 2, 3, 4, 5, 6], "green": [7, 10, 11, 13, 14, 15]}

# Фреймы меток на гранях — как в real/cubes/manifest.json (right = up x normal;
# вид на грань снаружи, у боковых граней «верх» метки смотрит на верх куба).
FACES = [
    ("+Z", (1, 0, 0), (0, 1, 0), (0, 0, 1)),
    ("-Z", (1, 0, 0), (0, -1, 0), (0, 0, -1)),
    ("+X", (0, 1, 0), (0, 0, 1), (1, 0, 0)),
    ("-X", (0, -1, 0), (0, 0, 1), (-1, 0, 0)),
    ("+Y", (-1, 0, 0), (0, 0, 1), (0, 1, 0)),
    ("-Y", (1, 0, 0), (0, 0, 1), (0, -1, 0)),
]

# Раскладка граней в кубическую текстуру MuJoCo: слот -> (грань куба, сколько
# четвертей повернуть картинку). Определена диагностикой test_cube_cv.py
# --stage faces (рендер куба в известных ориентациях, детекция, сверка с
# ground truth) и закреплена там же ассертами.
TEX_LAYOUT = {
    "front": ("+Z", 0, False), "back": ("-Z", 2, False),
    "right": ("+X", 1, False), "left": ("-X", 3, False),
    "up": ("+Y", 2, False), "down": ("-Y", 0, False),
}  # выверено перебором против ground truth: 0.00° на всех гранях

CV_CAM = {"name": "cv_top", "pos": "0.25 0 0.50",
          "xyaxes": "0 -1 0 1 0 0", "fovy": "45"}
CV_W, CV_H = 1600, 1200

_HERE = Path(__file__).parent
SCENE_DIR = _HERE / "models" / "so101"
ARUCO_XML = SCENE_DIR / "pick_place_aruco.xml"


def marker_grid(dic, mid: int) -> np.ndarray:
    """8x8 карта грани, True = чёрная клетка; gx вдоль right (0 слева),
    gy вдоль up (0 снизу); внешний ряд — белое поле."""
    img = dic.generateImageMarker(mid, 6) if hasattr(dic, "generateImageMarker") \
        else cv2.aruco.generateImageMarker(dic, mid, 6)
    grid = np.zeros((8, 8), bool)
    grid[1:7, 1:7] = (img == 0)[::-1, :]   # строки картинки идут сверху вниз
    return grid


def _face_texture(marker_id: int, quarter_turns: int,
                  mirror: bool = False) -> np.ndarray:
    """Грань 256x256: белое поле в 1 клетку + маркер 6 клеток (32 px/клетка).
    Повёрнута/зеркалирована под конвенцию кубической текстуры MuJoCo (порядок:
    сначала rot90, потом fliplr)."""
    tile = np.full((256, 256), 255, np.uint8)
    tile[32:224, 32:224] = cv2.aruco.generateImageMarker(DICT, marker_id, 192)
    tile = np.rot90(tile, quarter_turns)
    return np.fliplr(tile) if mirror else tile


def build_scene(tex_layout: dict | None = None, cv_cam: dict | None = None,
                out_name: str = "pick_place_aruco.xml") -> Path:
    """pick_place.xml -> сцена с кубиками в метках + камера cv_top.

    tex_layout позволяет диагностике подложить свою раскладку текстур;
    рабочая — TEX_LAYOUT. cv_cam переопределяет позу/fov верхней камеры
    (например, повторяя реальную установку C920)."""
    tex_layout = tex_layout or TEX_LAYOUT
    cv_cam = cv_cam or CV_CAM
    out_xml = SCENE_DIR / out_name
    tex_dir = SCENE_DIR / "aruco_tex"
    tex_dir.mkdir(exist_ok=True)
    for old in tex_dir.glob("*.png"):
        old.unlink()
    tree = ET.parse(SCENE_DIR / "pick_place.xml")
    root = tree.getroot()

    glob = root.find("visual/global")
    glob.set("offwidth", str(max(CV_W, 1920)))
    glob.set("offheight", str(max(CV_H, 1200)))

    asset = root.find("asset")
    for color in CUBE_IDS:
        face2id = dict(zip([f[0] for f in FACES], CUBE_IDS[color]))
        tex = ET.SubElement(asset, "texture", name=f"aruco_{color}_tex", type="cube")
        for slot, spec in tex_layout.items():
            face, turns, mirror = (*spec, False)[:3]
            tile = _face_texture(face2id[face], turns, mirror)
            # имя файла содержит хэш содержимого: питоновские биндинги MuJoCo
            # кэшируют текстуры по имени файла в рамках процесса и не замечают
            # перезаписи (проверено: рендер не менялся после смены PNG)
            digest = hashlib.md5(np.ascontiguousarray(tile)).hexdigest()[:8]
            png = tex_dir / f"{color}_{slot}_{digest}.png"
            cv2.imwrite(str(png), tile)
            tex.set(f"file{slot}", str(png.relative_to(SCENE_DIR)))
        ET.SubElement(asset, "material", name=f"aruco_{color}_mat",
                      texture=f"aruco_{color}_tex", rgba="1 1 1 1")
        geom = root.find(f".//geom[@name='{color}_geom']")
        geom.set("material", f"aruco_{color}_mat")

    wb = root.find("worldbody")
    ET.SubElement(wb, "camera", **cv_cam)
    # выключенная сварка «гриппер-куб»: тест hand-eye включает её как модель
    # реального серво-зажима (непрерывное усилие губок против сим-выскальзывания)
    eq = ET.SubElement(root, "equality")
    ET.SubElement(eq, "weld", name="grasp_weld", body1="gripper",
                  body2="red_cube", active="false")
    tree.write(out_xml)
    return out_xml


# --- оценка позы по кадру -------------------------------------------------

def camera_matrix() -> np.ndarray:
    f = CV_H / (2 * np.tan(np.radians(float(CV_CAM["fovy"])) / 2))
    return np.array([[f, 0, (CV_W - 1) / 2], [0, f, (CV_H - 1) / 2], [0, 0, 1]])


def camera_T_world(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    """4x4 мир->оптическая система OpenCV (z вперёд, y вниз). В реале эту
    матрицу даст калибровка камеры по доске на столе."""
    cid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, CV_CAM["name"])
    T = np.eye(4)
    T[:3, :3] = data.cam_xmat[cid].reshape(3, 3) @ np.diag([1, -1, -1])
    T[:3, 3] = data.cam_xpos[cid]
    return T


def _marker_T_cube(face_idx: int) -> np.ndarray:
    _, right, up, normal = FACES[face_idx]
    T = np.eye(4)
    T[:3, :3] = np.column_stack([right, up, normal]).astype(float)
    T[:3, 3] = np.asarray(normal, float) * CUBE_HALF
    return T


_OBJ_PTS = np.array([  # углы метки в её системе: TL, TR, BR, BL (канон ArUco)
    [-MARKER_SIZE / 2, MARKER_SIZE / 2, 0], [MARKER_SIZE / 2, MARKER_SIZE / 2, 0],
    [MARKER_SIZE / 2, -MARKER_SIZE / 2, 0], [-MARKER_SIZE / 2, -MARKER_SIZE / 2, 0],
], dtype=np.float32)

_ID2FACE = {mid: (color, fi) for color, ids in CUBE_IDS.items()
            for fi, mid in enumerate(ids)}


def _avg_rotations(mats: list[np.ndarray]) -> np.ndarray:
    u, _, vt = np.linalg.svd(np.sum(mats, axis=0))
    R = u @ vt
    if np.linalg.det(R) < 0:
        R = u @ np.diag([1, 1, -1]) @ vt
    return R


def estimate_cubes_cam(img_rgb: np.ndarray, K: np.ndarray,
                       dist=None) -> dict:
    """Позы кубов в СИСТЕМЕ КАМЕРЫ: {color: {"T_cam": 4x4, "n_markers": int,
    "spread_deg": float, "reproj_px": float, "markers_cam": [4x4 меток]}}.
    Не требует позы камеры — на этом строится hand-eye калибровка."""
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    det = cv2.aruco.ArucoDetector(DICT, cv2.aruco.DetectorParameters())
    corners, ids, _ = det.detectMarkers(gray)
    per_cube: dict[str, list] = {}
    if ids is None:
        return {}
    for quad, mid in zip(corners, ids.ravel()):
        if int(mid) not in _ID2FACE:
            continue
        color, fi = _ID2FACE[int(mid)]
        ok, rvec, tvec = cv2.solvePnP(_OBJ_PTS, quad, K, dist,
                                      flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok:
            continue
        proj, _ = cv2.projectPoints(_OBJ_PTS, rvec, tvec, K, dist)
        reproj = float(np.linalg.norm(proj.reshape(-1, 2) - quad.reshape(-1, 2),
                                      axis=1).mean())
        T_cm = np.eye(4)
        T_cm[:3, :3] = cv2.Rodrigues(rvec)[0]
        T_cm[:3, 3] = tvec.ravel()
        T_ccube = T_cm @ np.linalg.inv(_marker_T_cube(fi))
        per_cube.setdefault(color, []).append((T_ccube, reproj, T_cm))
    out = {}
    for color, obs in per_cube.items():
        R = _avg_rotations([T[:3, :3] for T, _, _ in obs])
        spread = max(np.degrees(np.arccos(np.clip(
            (np.trace(R.T @ Ti[:3, :3]) - 1) / 2, -1, 1))) for Ti, _, _ in obs)
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = np.mean([Ti[:3, 3] for Ti, _, _ in obs], axis=0)
        out[color] = {"T_cam": T, "n_markers": len(obs),
                      "spread_deg": float(spread),
                      "reproj_px": float(np.mean([r for _, r, _ in obs])),
                      "markers_cam": [m for _, _, m in obs]}
    return out


def estimate_cubes(img_rgb: np.ndarray, T_wc: np.ndarray, K: np.ndarray,
                   lying: bool = False) -> dict:
    """{color: {"T": 4x4 мир->куб, "n_markers": int, "spread_deg": float,
    "reproj_px": float}} по одному кадру. Кубы без видимых меток отсутствуют.

    Обёртка над estimate_cubes_cam: перевод в мир по известной позе камеры.
    lying=True — знание «куб лежит на столе»: позиция берётся пересечением
    луча камера->верхняя метка с плоскостью её высоты (2*CUBE_HALF), после
    чего вертикальная ось куба выправляется строго вверх. Это убирает
    систематику глубины PnP (мип-фильтрация рендера сжимает квад на ~2%) и
    её радиальный след на краях кадра. Для куба на весу — lying=False."""
    cam_pos = T_wc[:3, 3]
    out = {}
    for color, e in estimate_cubes_cam(img_rgb, K).items():
        T = T_wc @ e["T_cam"]
        if lying:
            top = [T_wc @ m for m in e["markers_cam"]]
            top = [wm for wm in top if wm[2, 2] > 0.7]     # нормаль вверх
            if top:
                p_m = top[0][:3, 3]
                ray = p_m - cam_pos
                p_snap = cam_pos + ray * (2 * CUBE_HALF - cam_pos[2]) / ray[2]
                T[:3, 3] = p_snap - [0, 0, CUBE_HALF]
                T[:3, :3] = _level_vertical(T[:3, :3])
        out[color] = {"T": T, "n_markers": e["n_markers"],
                      "spread_deg": e["spread_deg"], "reproj_px": e["reproj_px"]}
    return out

def _level_vertical(R: np.ndarray) -> np.ndarray:
    """Минимальный поворот, делающий почти вертикальную ось куба строго
    вертикальной (куб лежит на грани — наклон это шум оценки)."""
    k = int(np.argmax(np.abs(R[2, :])))
    a = R[:, k] * np.sign(R[2, k])
    v = np.cross(a, [0, 0, 1.0])
    s = np.linalg.norm(v)
    if s < 1e-9:
        return R
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return (np.eye(3) + vx + vx @ vx * ((1 - float(a[2])) / s ** 2)) @ R


def pose_errors(T_est: np.ndarray, pos_gt: np.ndarray, quat_gt: np.ndarray):
    """(ошибка позиции в мм, угловая ошибка в градусах) против ground truth."""
    R_gt = np.zeros(9)
    mujoco.mju_quat2Mat(R_gt, quat_gt)
    R_gt = R_gt.reshape(3, 3)
    dpos = float(np.linalg.norm(T_est[:3, 3] - pos_gt) * 1000)
    dR = R_gt.T @ T_est[:3, :3]
    dang = float(np.degrees(np.arccos(np.clip((np.trace(dR) - 1) / 2, -1, 1))))
    return dpos, dang

