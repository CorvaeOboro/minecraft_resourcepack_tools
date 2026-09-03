"""
Minecraft Model Symmetry Fixer UI

Load an existing Minecraft block-model `.json`, 
analyze D4 symmetry (4 rotations + 4 reflections in XZ around a chosen center), 
and generate an "IDEAL" model arrangement that is symmetric
while staying within Minecraft block-model constraints
(axis-aligned cuboids with optional *single-axis* element rotation).

This tool is intended to fix small hand-adjustment drift where elements that were
meant to be symmetric (rotated/mirrored around a center) end up slightly offset
or have slightly incorrect pivots/angles.

Expected inputs
- A Minecraft model JSON file containing `elements[]`

Outputs
- Visual symmetry error highlighting in the viewport
- An IDEAL model JSON 

TOOLSGROUP::MODEL
SORTGROUP::7
SORTPRIORITY::79
STATUS::active
VERSION::20260308
"""

from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal, Optional


def _try_import_pyside6():
    try:
        from PySide6 import QtCore, QtGui, QtWidgets

        return QtCore, QtGui, QtWidgets
    except Exception:
        return None


_imports = _try_import_pyside6()

if _imports is None:
    print("Missing UI dependency. Install PySide6 and retry.")
    print("Example: pip install PySide6")
    raise SystemExit(1)

QtCore, QtGui, QtWidgets = _imports

Axis = Literal["x", "y", "z"]

_ALLOWED_ROTATION_ANGLES = (-45.0, -22.5, 0.0, 22.5, 45.0)


def _r6(x: float) -> float:
    return round(float(x), 6)


def _snap_to_allowed(v: float, allowed: tuple[float, ...], *, eps: float = 1e-6) -> float:
    vv = float(v)
    best = min(allowed, key=lambda a: abs(float(a) - vv))
    if abs(float(best) - vv) <= float(eps):
        return float(best)
    return float(v)


def _apply_dark_theme(app: QtWidgets.QApplication) -> None:
    app.setStyle("Fusion")
    app.setStyleSheet(
        "QWidget { background: #0b0b0d; color: #e8e8ea; font-size: 12px; }"
        "QLabel { color: #e8e8ea; }"
        "QGroupBox { border: 1px solid #1f2024; margin-top: 8px; }"
        "QGroupBox::title { subcontrol-origin: margin; left: 8px; padding: 0 4px 0 4px; color: #cfcfd4; }"
        "QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox { background: #000000; color: #ffffff; border: 1px solid #2a2b30; padding: 4px; border-radius: 4px; }"
        "QPlainTextEdit, QTextEdit { background: #000000; color: #ffffff; border: 1px solid #2a2b30; padding: 6px; border-radius: 4px; }"
        "QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus { border: 1px solid #3a6ea5; }"
        "QComboBox QAbstractItemView { background: #000000; color: #ffffff; selection-background-color: #2b4a6b; }"
        "QCheckBox { spacing: 6px; }"
        "QCheckBox::indicator { width: 14px; height: 14px; }"
        "QPushButton { background: #15161a; color: #e8e8ea; border: 1px solid #2a2b30; padding: 6px 10px; border-radius: 6px; }"
        "QPushButton:hover { border: 1px solid #3a3b42; }"
        "QPushButton:disabled { color: #777780; border: 1px solid #1a1b1f; background: #101114; }"
        "QPushButton#btn_analyze { background: #1b3326; border: 1px solid #2d5b3f; }"
        "QPushButton#btn_analyze:hover { border: 1px solid #3d7a55; }"
        "QPushButton#btn_solve { background: #241a2c; border: 1px solid #4a2c63; }"
        "QPushButton#btn_solve:hover { border: 1px solid #6a3b90; }"
        "QSplitter::handle { background: #0b0b0d; }"
    )


@dataclass(frozen=True)
class Rotation:
    axis: Axis
    angle: float
    origin: tuple[float, float, float]
    rescale: bool = False


@dataclass(frozen=True)
class Cuboid:
    fr: tuple[float, float, float]
    to: tuple[float, float, float]
    rotation: Optional[Rotation]


@dataclass
class Element:
    idx: int
    raw: dict
    cuboid: Cuboid

    def center(self) -> tuple[float, float, float]:
        fx, fy, fz = self.cuboid.fr
        tx, ty, tz = self.cuboid.to
        return ((fx + tx) * 0.5, (fy + ty) * 0.5, (fz + tz) * 0.5)

    def size(self) -> tuple[float, float, float]:
        fx, fy, fz = self.cuboid.fr
        tx, ty, tz = self.cuboid.to
        return (abs(tx - fx), abs(ty - fy), abs(tz - fz))


def _parse_vec2(raw: str) -> tuple[float, float]:
    parts = [p.strip() for p in raw.split(",")]
    if len(parts) != 2:
        raise ValueError("Expected x,z")
    return float(parts[0]), float(parts[1])


def _det3(m: tuple[tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]]) -> int:
    (a, b, c), (d, e, f), (g, h, i) = m
    det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)
    return int(det)


def _mat_vec(m: tuple[tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]], v: tuple[float, float, float]) -> tuple[float, float, float]:
    x, y, z = v
    return (
        float(m[0][0]) * x + float(m[0][1]) * y + float(m[0][2]) * z,
        float(m[1][0]) * x + float(m[1][1]) * y + float(m[1][2]) * z,
        float(m[2][0]) * x + float(m[2][1]) * y + float(m[2][2]) * z,
    )


def _mat_t(m: tuple[tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]]) -> tuple[tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]]:
    return (
        (int(m[0][0]), int(m[1][0]), int(m[2][0])),
        (int(m[0][1]), int(m[1][1]), int(m[2][1])),
        (int(m[0][2]), int(m[1][2]), int(m[2][2])),
    )


@dataclass(frozen=True)
class SymOp:
    name: str
    m: tuple[tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]]
    det: int

    def inverse(self) -> "SymOp":
        mt = _mat_t(self.m)
        return SymOp(name=self.name + "^-1", m=mt, det=_det3(mt))


_D4_OPS: list[SymOp] = []


def _init_ops() -> list[SymOp]:
    ops = [
        SymOp("R0", ((1, 0, 0), (0, 1, 0), (0, 0, 1)), 1),
        SymOp("R90", ((0, 0, 1), (0, 1, 0), (-1, 0, 0)), 1),
        SymOp("R180", ((-1, 0, 0), (0, 1, 0), (0, 0, -1)), 1),
        SymOp("R270", ((0, 0, -1), (0, 1, 0), (1, 0, 0)), 1),
        SymOp("MX", ((-1, 0, 0), (0, 1, 0), (0, 0, 1)), -1),
        SymOp("MZ", ((1, 0, 0), (0, 1, 0), (0, 0, -1)), -1),
        SymOp("MD", ((0, 0, 1), (0, 1, 0), (1, 0, 0)), -1),
        SymOp("MAD", ((0, 0, -1), (0, 1, 0), (-1, 0, 0)), -1),
    ]
    return [SymOp(op.name, op.m, _det3(op.m)) for op in ops]


_D4_OPS = _init_ops()


def _mat_mul3(
    a: tuple[tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]],
    b: tuple[tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]],
) -> tuple[tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]]:
    out = []
    for r in range(3):
        row = []
        for c in range(3):
            row.append(
                int(a[r][0]) * int(b[0][c])
                + int(a[r][1]) * int(b[1][c])
                + int(a[r][2]) * int(b[2][c])
            )
        out.append(tuple(int(x) for x in row))
    return tuple(out)  # type: ignore[return-value]


def _find_op_by_name(name: str) -> SymOp:
    op = next((o for o in _D4_OPS if str(o.name) == str(name)), None)
    if op is None:
        raise KeyError(f"Unknown sym op: {name}")
    return op


def _find_op_by_matrix(m: tuple[tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]]) -> SymOp:
    op = next((o for o in _D4_OPS if o.m == m), None)
    if op is None:
        raise KeyError(f"Unknown sym op matrix: {m}")
    return op


def _variant_label_op_name_pairs(*, include_reflections: bool, mirror_gen_op_name: str) -> list[tuple[str, str]]:
    rot_ops = [_find_op_by_name(n) for n in ("R0", "R90", "R180", "R270")]
    out: list[tuple[str, str]] = [("R0", "R0"), ("R90", "R90"), ("R180", "R180"), ("R270", "R270")]
    if not include_reflections:
        return out

    mirror_gen = _find_op_by_name(mirror_gen_op_name)
    for r_deg, r_op in zip((0, 90, 180, 270), rot_ops):
        m = _mat_mul3(r_op.m, mirror_gen.m)
        op_name = _find_op_by_matrix(m).name
        out.append((f"M{r_deg}", str(op_name)))
    return out


def _ops_for_variant_set(*, include_reflections: bool, mirror_gen_op_name: str) -> list[SymOp]:
    pairs = _variant_label_op_name_pairs(include_reflections=include_reflections, mirror_gen_op_name=mirror_gen_op_name)
    seen: set[str] = set()
    out: list[SymOp] = []
    for _label, name in pairs:
        if str(name) in seen:
            continue
        seen.add(str(name))
        out.append(_find_op_by_name(str(name)))
    return out


def _sym_apply_point(*, op: SymOp, p: tuple[float, float, float], center_xz: tuple[float, float]) -> tuple[float, float, float]:
    cx, cz = center_xz
    x, y, z = p
    v = (x - cx, y, z - cz)
    x2, y2, z2 = _mat_vec(op.m, v)
    return (x2 + cx, y2, z2 + cz)


def _axis_unit(axis: Axis) -> tuple[float, float, float]:
    if axis == "x":
        return (1.0, 0.0, 0.0)
    if axis == "y":
        return (0.0, 1.0, 0.0)
    return (0.0, 0.0, 1.0)


def _vec_to_axis(v: tuple[float, float, float]) -> tuple[Axis, float]:
    x, y, z = (float(v[0]), float(v[1]), float(v[2]))
    ax = abs(x)
    ay = abs(y)
    az = abs(z)
    if ax >= ay and ax >= az:
        return "x", 1.0 if x >= 0.0 else -1.0
    if ay >= ax and ay >= az:
        return "y", 1.0 if y >= 0.0 else -1.0
    return "z", 1.0 if z >= 0.0 else -1.0


def _sym_apply_rotation(*, op: SymOp, rot: Rotation, center_xz: tuple[float, float]) -> Rotation:
    origin2 = _sym_apply_point(op=op, p=rot.origin, center_xz=center_xz)

    a = _axis_unit(rot.axis)
    a2 = _mat_vec(op.m, a)
    a2 = (float(op.det) * a2[0], float(op.det) * a2[1], float(op.det) * a2[2])

    axis2, sign = _vec_to_axis(a2)
    angle2 = float(rot.angle) * float(sign)

    return Rotation(axis=axis2, angle=angle2, origin=origin2, rescale=bool(rot.rescale))


def _sym_apply_faces(*, op: SymOp, faces: dict) -> dict:
    if not isinstance(faces, dict):
        return faces

    face_normals = {
        "north": (0.0, 0.0, -1.0),
        "south": (0.0, 0.0, 1.0),
        "east": (1.0, 0.0, 0.0),
        "west": (-1.0, 0.0, 0.0),
        "up": (0.0, 1.0, 0.0),
        "down": (0.0, -1.0, 0.0),
    }

    def vec_to_face(n: tuple[float, float, float]) -> str:
        axis, sign = _vec_to_axis(n)
        if axis == "x":
            return "east" if sign > 0.0 else "west"
        if axis == "y":
            return "up" if sign > 0.0 else "down"
        return "south" if sign > 0.0 else "north"

    out: dict = {}
    for fk, fv in faces.items():
        n = face_normals.get(str(fk))
        if n is None:
            out[fk] = fv
            continue
        n2 = _mat_vec(op.m, n)
        fk2 = vec_to_face(n2)
        out[fk2] = fv
    return out


def _aabb_corners(fr: tuple[float, float, float], to: tuple[float, float, float]) -> list[tuple[float, float, float]]:
    fx, fy, fz = fr
    tx, ty, tz = to
    return [
        (fx, fy, fz),
        (tx, fy, fz),
        (fx, ty, fz),
        (tx, ty, fz),
        (fx, fy, tz),
        (tx, fy, tz),
        (fx, ty, tz),
        (tx, ty, tz),
    ]


def _sym_apply_cuboid(*, op: SymOp, cub: Cuboid, center_xz: tuple[float, float]) -> Cuboid:
    pts = _aabb_corners(cub.fr, cub.to)
    pts2 = [_sym_apply_point(op=op, p=p, center_xz=center_xz) for p in pts]

    xs = [p[0] for p in pts2]
    ys = [p[1] for p in pts2]
    zs = [p[2] for p in pts2]

    fr2 = (min(xs), min(ys), min(zs))
    to2 = (max(xs), max(ys), max(zs))

    rot2 = None
    if cub.rotation is not None and float(cub.rotation.angle) != 0.0:
        rot2 = _sym_apply_rotation(op=op, rot=cub.rotation, center_xz=center_xz)

    return Cuboid(fr=fr2, to=to2, rotation=rot2)


def _sym_apply_element(*, op: SymOp, el: Element, center_xz: tuple[float, float]) -> Element:
    cub2 = _sym_apply_cuboid(op=op, cub=el.cuboid, center_xz=center_xz)
    raw2 = json.loads(json.dumps(el.raw))

    raw2["from"] = [cub2.fr[0], cub2.fr[1], cub2.fr[2]]
    raw2["to"] = [cub2.to[0], cub2.to[1], cub2.to[2]]

    if "rotation" in raw2:
        raw2.pop("rotation", None)

    if cub2.rotation is not None and float(cub2.rotation.angle) != 0.0:
        rot_d: dict = {
            "origin": [cub2.rotation.origin[0], cub2.rotation.origin[1], cub2.rotation.origin[2]],
            "axis": cub2.rotation.axis,
            "angle": cub2.rotation.angle,
        }
        if bool(cub2.rotation.rescale):
            rot_d["rescale"] = True
        raw2["rotation"] = rot_d

    if isinstance(raw2.get("faces"), dict):
        raw2["faces"] = _sym_apply_faces(op=op, faces=raw2["faces"])

    return Element(idx=int(el.idx), raw=raw2, cuboid=cub2)


def _parse_element(idx: int, raw: dict) -> Element:
    fr0 = raw.get("from")
    to0 = raw.get("to")
    if not (isinstance(fr0, list) and isinstance(to0, list) and len(fr0) == 3 and len(to0) == 3):
        raise ValueError(f"Invalid element from/to at index {idx}")

    fr = (float(fr0[0]), float(fr0[1]), float(fr0[2]))
    to = (float(to0[0]), float(to0[1]), float(to0[2]))

    fx, fy, fz = fr
    tx, ty, tz = to
    fr = (min(fx, tx), min(fy, ty), min(fz, tz))
    to = (max(fx, tx), max(fy, ty), max(fz, tz))

    rot = None
    r0 = raw.get("rotation")
    if isinstance(r0, dict):
        axis = r0.get("axis")
        ang = r0.get("angle")
        org = r0.get("origin")
        if axis in {"x", "y", "z"} and isinstance(ang, (int, float)) and isinstance(org, list) and len(org) == 3:
            rot = Rotation(
                axis=str(axis),
                angle=float(ang),
                origin=(float(org[0]), float(org[1]), float(org[2])),
                rescale=bool(r0.get("rescale", False)),
            )

    cub = Cuboid(fr=fr, to=to, rotation=rot)
    return Element(idx=int(idx), raw=dict(raw), cuboid=cub)


def format_minecraft_model_json(model: dict) -> str:
    def is_scalar(v) -> bool:
        return v is None or isinstance(v, (str, int, float, bool))

    def try_inline(obj) -> Optional[str]:
        if isinstance(obj, list):
            if all(is_scalar(x) for x in obj) and len(obj) <= 16:
                s = json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
                if len(s) <= 80:
                    return s
            return None

        if isinstance(obj, dict):
            if all(isinstance(k, str) for k in obj.keys()):
                ok = True
                for v in obj.values():
                    if is_scalar(v):
                        continue
                    if isinstance(v, list) and all(is_scalar(x) for x in v) and len(v) <= 16:
                        continue
                    if isinstance(v, dict) and all(isinstance(kk, str) for kk in v.keys()) and all(is_scalar(vv) for vv in v.values()):
                        continue
                    ok = False
                    break
                if ok:
                    s = json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
                    if len(s) <= 80:
                        return s
            return None

        if is_scalar(obj):
            return json.dumps(obj, ensure_ascii=False)

        return None

    def fmt(obj, level: int) -> str:
        inline = try_inline(obj)
        if inline is not None:
            return inline

        ind = "  " * level
        ind2 = "  " * (level + 1)

        if isinstance(obj, list):
            if not obj:
                return "[]"
            parts = []
            for item in obj:
                parts.append(f"{ind2}{fmt(item, level + 1)}")
            return "[\n" + ",\n".join(parts) + f"\n{ind}]"

        if isinstance(obj, dict):
            if not obj:
                return "{}"
            parts = []
            for k, v in obj.items():
                ks = json.dumps(k, ensure_ascii=False)
                vs = fmt(v, level + 1)
                if "\n" in vs:
                    vs = "\n".join([vs.split("\n", 1)[0]] + [ind2 + line for line in vs.split("\n")[1:]])
                parts.append(f"{ind2}{ks}: {vs}")
            return "{\n" + ",\n".join(parts) + f"\n{ind}}}"

        return json.dumps(obj, ensure_ascii=False)

    return fmt(model, 0)


class ModelViewport(QtWidgets.QWidget):
    def __init__(self, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)

        self.setMouseTracking(True)
        self.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)

        self._yaw = 45.0
        self._pitch = 25.0
        self._distance = 42.0
        self._center = (8.0, 8.0, 8.0)

        self._last_mouse_pos: Optional[QtCore.QPoint] = None
        self._cuboids: list[Cuboid] = []
        self._visible_mask: list[bool] = []
        self._colors: list[QtGui.QColor] = []

        self._draw_faces = True
        self._faces_translucent = False
        self._draw_wireframe = True

    def set_render_options(self, *, faces: bool, faces_translucent: bool, wireframe: bool) -> None:
        self._draw_faces = bool(faces)
        self._faces_translucent = bool(faces_translucent)
        self._draw_wireframe = bool(wireframe)
        self.update()

    def set_cuboids(self, cuboids: Iterable[Cuboid], *, colors: Optional[list[QtGui.QColor]] = None) -> None:
        self._cuboids = list(cuboids)
        self._visible_mask = [True] * len(self._cuboids)
        self._colors = list(colors) if colors is not None else []
        self.update()

    def set_visibility_mask(self, mask: list[bool]) -> None:
        if len(mask) != len(self._cuboids):
            return
        self._visible_mask = list(bool(x) for x in mask)
        self.update()

    def wheelEvent(self, event: QtGui.QWheelEvent) -> None:
        delta = event.angleDelta().y() / 120.0
        self._distance *= 0.9 ** delta
        self._distance = max(6.0, min(250.0, self._distance))
        self.update()

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            self._last_mouse_pos = event.position().toPoint()
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            self._last_mouse_pos = None
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        if self._last_mouse_pos is not None:
            p = event.position().toPoint()
            dx = p.x() - self._last_mouse_pos.x()
            dy = p.y() - self._last_mouse_pos.y()
            self._last_mouse_pos = p
            self._yaw += float(dx) * 0.35
            self._pitch += float(dy) * 0.35
            self._pitch = max(-89.0, min(89.0, self._pitch))
            self.update()
            event.accept()
            return
        super().mouseMoveEvent(event)

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)
        painter.fillRect(self.rect(), QtGui.QColor(6, 6, 8))

        w = max(1, self.width())
        h = max(1, self.height())

        def rot_y(x: float, y: float, z: float, a: float) -> tuple[float, float, float]:
            c = math.cos(a)
            s = math.sin(a)
            return (c * x + s * z, y, -s * x + c * z)

        def rot_x(x: float, y: float, z: float, a: float) -> tuple[float, float, float]:
            c = math.cos(a)
            s = math.sin(a)
            return (x, c * y - s * z, s * y + c * z)

        yaw = math.radians(self._yaw)
        pitch = math.radians(self._pitch)

        cx0, cy0, cz0 = self._center

        def to_camera(p: tuple[float, float, float]) -> tuple[float, float, float]:
            x, y, z = p
            x -= cx0
            y -= cy0
            z -= cz0
            x, y, z = rot_y(x, y, z, yaw)
            x, y, z = rot_x(x, y, z, pitch)
            z += self._distance
            return x, y, z

        f = min(w, h) * 0.9

        def project_cam(cam: tuple[float, float, float]) -> Optional[QtCore.QPointF]:
            x, y, z = cam
            if z <= 0.05:
                return None
            sx = (x * f) / z + (w * 0.5)
            sy = (-y * f) / z + (h * 0.5)
            return QtCore.QPointF(sx, sy)

        def poly_area2(p0: QtCore.QPointF, p1: QtCore.QPointF, p2: QtCore.QPointF) -> float:
            return (p1.x() - p0.x()) * (p2.y() - p0.y()) - (p1.y() - p0.y()) * (p2.x() - p0.x())

        def rotate_about_axis(*, axis: str, angle_deg: float, origin: tuple[float, float, float], p: tuple[float, float, float]) -> tuple[float, float, float]:
            if angle_deg == 0.0:
                return p

            ox, oy, oz = origin
            x, y, z = p
            x -= ox
            y -= oy
            z -= oz

            a = math.radians(angle_deg)
            c = math.cos(a)
            s = math.sin(a)

            if axis == "x":
                yy = c * y - s * z
                zz = s * y + c * z
                xx = x
            elif axis == "y":
                xx = c * x + s * z
                zz = -s * x + c * z
                yy = y
            else:
                xx = c * x - s * y
                yy = s * x + c * y
                zz = z

            xx += ox
            yy += oy
            zz += oz
            return xx, yy, zz

        segments: list[tuple[tuple[float, float, float], tuple[float, float, float], QtGui.QColor, float]] = []
        faces: list[tuple[float, QtGui.QPolygonF, QtGui.QColor]] = []

        def add_seg(a: tuple[float, float, float], b: tuple[float, float, float], col: QtGui.QColor) -> None:
            ca = to_camera(a)
            cb = to_camera(b)
            pa = project_cam(ca)
            pb = project_cam(cb)
            if pa is None or pb is None:
                return
            segments.append((a, b, col, (ca[2] + cb[2]) * 0.5))

        o = (0.0, 0.0, 0.0)
        add_seg(o, (16.0, 0.0, 0.0), QtGui.QColor(210, 70, 70, 220))
        add_seg(o, (0.0, 16.0, 0.0), QtGui.QColor(70, 210, 120, 220))
        add_seg(o, (0.0, 0.0, 16.0), QtGui.QColor(85, 140, 230, 220))

        cube_col = QtGui.QColor(220, 220, 230, 90)
        c0 = (0.0, 0.0, 0.0)
        c1 = (16.0, 0.0, 0.0)
        c2 = (16.0, 16.0, 0.0)
        c3 = (0.0, 16.0, 0.0)
        c4 = (0.0, 0.0, 16.0)
        c5 = (16.0, 0.0, 16.0)
        c6 = (16.0, 16.0, 16.0)
        c7 = (0.0, 16.0, 16.0)
        for a, b in (
            (c0, c1),
            (c1, c2),
            (c2, c3),
            (c3, c0),
            (c4, c5),
            (c5, c6),
            (c6, c7),
            (c7, c4),
            (c0, c4),
            (c1, c5),
            (c2, c6),
            (c3, c7),
        ):
            add_seg(a, b, cube_col)

        for i_c, c in enumerate(self._cuboids):
            if i_c < len(self._visible_mask) and not bool(self._visible_mask[i_c]):
                continue

            base_col = QtGui.QColor(95, 170, 255, 190)
            if 0 <= int(i_c) < len(self._colors):
                base_col = self._colors[int(i_c)]

            face_alpha = 200 if not self._faces_translucent else 70
            face_base = QtGui.QColor(base_col.red(), base_col.green(), base_col.blue(), face_alpha)
            line_col = QtGui.QColor(base_col.red(), base_col.green(), base_col.blue(), min(255, base_col.alpha() + 30))

            fx, fy, fz = c.fr
            tx, ty, tz = c.to

            p000 = (fx, fy, fz)
            p100 = (tx, fy, fz)
            p110 = (tx, ty, fz)
            p010 = (fx, ty, fz)
            p001 = (fx, fy, tz)
            p101 = (tx, fy, tz)
            p111 = (tx, ty, tz)
            p011 = (fx, ty, tz)

            pts = [p000, p100, p110, p010, p001, p101, p111, p011]

            if c.rotation is not None and float(c.rotation.angle) != 0.0:
                axis = str(c.rotation.axis)
                angle = float(c.rotation.angle)
                origin = (float(c.rotation.origin[0]), float(c.rotation.origin[1]), float(c.rotation.origin[2]))
                pts = [rotate_about_axis(axis=axis, angle_deg=angle, origin=origin, p=p) for p in pts]

            if self._draw_faces:
                cam_pts = [to_camera(p) for p in pts]
                face_idx = (
                    (0, 1, 2, 3),
                    (4, 5, 6, 7),
                    (0, 1, 5, 4),
                    (1, 2, 6, 5),
                    (2, 3, 7, 6),
                    (3, 0, 4, 7),
                )

                for i0, i1, i2, i3 in face_idx:
                    p0 = project_cam(cam_pts[i0])
                    p1 = project_cam(cam_pts[i1])
                    p2 = project_cam(cam_pts[i2])
                    p3 = project_cam(cam_pts[i3])
                    if p0 is None or p1 is None or p2 is None or p3 is None:
                        continue

                    a2 = poly_area2(p0, p1, p2)
                    if a2 >= 0.0:
                        continue

                    v0 = cam_pts[i0]
                    v1 = cam_pts[i1]
                    v2 = cam_pts[i2]

                    ax, ay, az = (v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2])
                    bx, by, bz = (v2[0] - v0[0], v2[1] - v0[1], v2[2] - v0[2])
                    nx = (ay * bz) - (az * by)
                    ny = (az * bx) - (ax * bz)
                    nz = (ax * by) - (ay * bx)

                    nlen = math.sqrt((nx * nx) + (ny * ny) + (nz * nz))
                    shade = 0.85
                    if nlen > 1e-9:
                        shade = 0.35 + (0.65 * max(0.0, min(1.0, (-nz) / nlen)))

                    col = QtGui.QColor(
                        int(face_base.red() * shade),
                        int(face_base.green() * shade),
                        int(face_base.blue() * shade),
                        face_base.alpha(),
                    )

                    poly = QtGui.QPolygonF([p0, p1, p2, p3])
                    depth = (cam_pts[i0][2] + cam_pts[i1][2] + cam_pts[i2][2] + cam_pts[i3][2]) / 4.0
                    faces.append((depth, poly, col))

            if self._draw_wireframe:
                edges = (
                    (0, 1),
                    (1, 2),
                    (2, 3),
                    (3, 0),
                    (4, 5),
                    (5, 6),
                    (6, 7),
                    (7, 4),
                    (0, 4),
                    (1, 5),
                    (2, 6),
                    (3, 7),
                )
                for ia, ib in edges:
                    add_seg(pts[ia], pts[ib], line_col)

        if self._draw_faces and faces:
            faces.sort(key=lambda it: it[0], reverse=True)
            painter.setPen(QtCore.Qt.PenStyle.NoPen)
            for _depth, poly, col in faces:
                painter.setBrush(QtGui.QBrush(col))
                painter.drawPolygon(poly)

        segments.sort(key=lambda s: s[3], reverse=True)
        for a, b, col, _z in segments:
            pa = project_cam(to_camera(a))
            pb = project_cam(to_camera(b))
            if pa is None or pb is None:
                continue
            pen = QtGui.QPen(col)
            pen.setWidthF(1.4)
            painter.setPen(pen)
            painter.drawLine(pa, pb)

        painter.end()


@dataclass
class GroupMatch:
    seed_idx: int
    matches: dict[str, int]


def _dist3(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    dx = float(a[0]) - float(b[0])
    dy = float(a[1]) - float(b[1])
    dz = float(a[2]) - float(b[2])
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def _element_error(a: Element, b: Element) -> float:
    ca = a.center()
    cb = b.center()
    sa = a.size()
    sb = b.size()

    err = _dist3(ca, cb)
    err += abs(sa[0] - sb[0]) * 0.25
    err += abs(sa[1] - sb[1]) * 0.25
    err += abs(sa[2] - sb[2]) * 0.25

    ra = a.cuboid.rotation
    rb = b.cuboid.rotation
    if (ra is None) != (rb is None):
        err += 10.0
    elif ra is not None and rb is not None:
        if str(ra.axis) != str(rb.axis):
            err += 10.0
        err += abs(float(ra.angle) - float(rb.angle)) * 0.05
        err += _dist3(ra.origin, rb.origin) * 0.5

    return float(err)


def _match_groups(
    *,
    elements: list[Element],
    center_xz: tuple[float, float],
    pos_eps: float,
    include_reflections: bool,
    mirror_gen_op_name: str,
) -> list[GroupMatch]:
    unassigned: set[int] = set(range(len(elements)))
    groups: list[GroupMatch] = []

    r90 = _find_op_by_name("R90")
    mirror_gen = _find_op_by_name(str(mirror_gen_op_name))
    variant_pairs = _variant_label_op_name_pairs(include_reflections=include_reflections, mirror_gen_op_name=str(mirror_gen_op_name))

    def best_match_for(expected: Element, candidate_idxs: Iterable[int]) -> tuple[Optional[int], float]:
        best_j: Optional[int] = None
        best_err = 1e18
        for j in candidate_idxs:
            err = _element_error(expected, elements[int(j)])
            if err < best_err:
                best_err = err
                best_j = int(j)
        return best_j, float(best_err)

    while unassigned:
        seed_i = min(unassigned)
        seed = elements[seed_i]

        matches: dict[str, int] = {"R0": int(seed_i)}
        unassigned.remove(int(seed_i))

        rot_chain: list[tuple[str, SymOp]] = [("R90", _find_op_by_name("R90")), ("R180", _find_op_by_name("R90")), ("R270", _find_op_by_name("R90"))]
        prev_idx = int(seed_i)
        for rot_label, op_step in rot_chain:
            expected = _sym_apply_element(op=op_step, el=elements[int(prev_idx)], center_xz=center_xz)
            in_group = list(matches.values())
            j0, e0 = best_match_for(expected, in_group)
            if j0 is not None and float(e0) <= float(pos_eps):
                matches[str(rot_label)] = int(j0)
                prev_idx = int(j0)
                continue

            j1, e1 = best_match_for(expected, list(unassigned))
            if j1 is not None and float(e1) <= float(pos_eps):
                matches[str(rot_label)] = int(j1)
                unassigned.remove(int(j1))
                prev_idx = int(j1)

        if include_reflections:
            expected_m0 = _sym_apply_element(op=mirror_gen, el=seed, center_xz=center_xz)
            in_group = list(matches.values())
            jm0, em0 = best_match_for(expected_m0, in_group)
            m0_idx: Optional[int] = None
            if jm0 is not None and float(em0) <= float(pos_eps):
                matches[str(mirror_gen.name)] = int(jm0)
                m0_idx = int(jm0)
            else:
                jm0, em0 = best_match_for(expected_m0, list(unassigned))
                if jm0 is not None and float(em0) <= float(pos_eps):
                    matches[str(mirror_gen.name)] = int(jm0)
                    unassigned.remove(int(jm0))
                    m0_idx = int(jm0)

            if m0_idx is not None:
                prev_m = int(m0_idx)
                rot_ops = [_find_op_by_name(n) for n in ("R0", "R90", "R180", "R270")]
                for rot_op in rot_ops[1:]:
                    expected = _sym_apply_element(op=r90, el=elements[int(prev_m)], center_xz=center_xz)
                    target_m = _mat_mul3(rot_op.m, mirror_gen.m)
                    target_op_name = str(_find_op_by_matrix(target_m).name)
                    in_group = list(matches.values())
                    j0, e0 = best_match_for(expected, in_group)
                    if j0 is not None and float(e0) <= float(pos_eps):
                        matches[target_op_name] = int(j0)
                        prev_m = int(j0)
                        continue

                    j1, e1 = best_match_for(expected, list(unassigned))
                    if j1 is not None and float(e1) <= float(pos_eps):
                        matches[target_op_name] = int(j1)
                        unassigned.remove(int(j1))
                        prev_m = int(j1)

        remapped: dict[str, int] = {}
        for _label, op_name in variant_pairs:
            if str(op_name) in matches:
                remapped[str(op_name)] = int(matches[str(op_name)])
        groups.append(GroupMatch(seed_idx=int(seed_i), matches=remapped))

    return groups


def _theta_snap_mode_label(mode: str) -> str:
    return {
        "none": "None",
        "cardinal": "Cardinal (0/90/180/270)",
        "diagonal": "Diagonal (45/135/...) ",
        "auto": "Auto (cardinal vs diagonal)",
    }.get(str(mode), str(mode))


def _snap_theta(theta_deg: float, mode: str) -> float:
    t = float(theta_deg)

    def norm(a: float) -> float:
        aa = float(a) % 360.0
        if aa < 0.0:
            aa += 360.0
        return aa

    def best_to_set(target: float, base: float, step: float) -> float:
        target = norm(target)
        best = None
        best_d = 1e18
        for k in range(int(360.0 / step)):
            a = norm(base + k * step)
            d = abs(((a - target + 180.0) % 360.0) - 180.0)
            if d < best_d:
                best_d = d
                best = a
        return float(best) if best is not None else float(target)

    if mode == "cardinal":
        return best_to_set(t, 0.0, 90.0)
    if mode == "diagonal":
        return best_to_set(t, 45.0, 90.0)
    if mode == "auto":
        c = best_to_set(t, 0.0, 90.0)
        d = best_to_set(t, 45.0, 90.0)
        dc = abs(((c - t + 180.0) % 360.0) - 180.0)
        dd = abs(((d - t + 180.0) % 360.0) - 180.0)
        return float(c) if dc <= dd else float(d)
    return float(t)


def _dedupe_elements(elements: list[dict]) -> list[dict]:
    seen: set[str] = set()
    out: list[dict] = []

    def key_for(el: dict) -> str:
        fr = el.get("from")
        to = el.get("to")
        rot = el.get("rotation")
        if not (isinstance(fr, list) and isinstance(to, list) and len(fr) == 3 and len(to) == 3):
            return json.dumps(el, sort_keys=True)
        frk = tuple(_r6(float(x)) for x in fr)
        tok = tuple(_r6(float(x)) for x in to)
        if isinstance(rot, dict):
            axis = rot.get("axis")
            ang = rot.get("angle")
            org = rot.get("origin")
            if axis in {"x", "y", "z"} and isinstance(ang, (int, float)) and isinstance(org, list) and len(org) == 3:
                orgk = tuple(_r6(float(x)) for x in org)
                return f"{frk}|{tok}|{axis}|{_r6(float(ang))}|{orgk}|{bool(rot.get('rescale', False))}"
        return f"{frk}|{tok}|-"

    for el in elements:
        k = key_for(el)
        if k in seen:
            continue
        seen.add(k)
        out.append(el)
    return out


def _solve_ideal(
    *,
    model: dict,
    elements: list[Element],
    groups: list[GroupMatch],
    center_xz: tuple[float, float],
    include_reflections: bool,
    mirror_gen_op_name: str,
    fill_missing: bool,
    base_only: bool,
    snap_theta_mode: str,
    snap_angles: bool,
    angle_eps: float,
    dedupe: bool,
) -> dict:
    ops = _ops_for_variant_set(include_reflections=include_reflections, mirror_gen_op_name=str(mirror_gen_op_name))

    used: set[int] = set()
    out_elements: list[dict] = []

    def avg(vals: list[float]) -> float:
        if not vals:
            return 0.0
        return float(sum(vals)) / float(len(vals))

    for g in groups:
        idxs = list(g.matches.values())
        for i in idxs:
            used.add(int(i))

        seed = elements[g.matches.get("R0", g.seed_idx)]

        base_samples: list[Element] = []
        for op_name, el_idx in g.matches.items():
            op = next((o for o in ops if o.name == op_name), None)
            if op is None:
                continue
            inv = op.inverse()
            base_samples.append(_sym_apply_element(op=inv, el=elements[el_idx], center_xz=center_xz))

        if not base_samples:
            continue

        ref0 = base_samples[0]
        rot0 = ref0.cuboid.rotation

        frx = [s.cuboid.fr[0] for s in base_samples]
        fry = [s.cuboid.fr[1] for s in base_samples]
        frz = [s.cuboid.fr[2] for s in base_samples]
        tox = [s.cuboid.to[0] for s in base_samples]
        toy = [s.cuboid.to[1] for s in base_samples]
        toz = [s.cuboid.to[2] for s in base_samples]

        fr = (avg(frx), avg(fry), avg(frz))
        to = (avg(tox), avg(toy), avg(toz))

        fx, fy, fz = fr
        tx, ty, tz = to
        fr = (min(fx, tx), min(fy, ty), min(fz, tz))
        to = (max(fx, tx), max(fy, ty), max(fz, tz))

        rot_out: Optional[Rotation] = None
        if rot0 is not None:
            if all(s.cuboid.rotation is not None and str(s.cuboid.rotation.axis) == str(rot0.axis) for s in base_samples):
                angs = [float(s.cuboid.rotation.angle) for s in base_samples if s.cuboid.rotation is not None]
                orgx = [float(s.cuboid.rotation.origin[0]) for s in base_samples if s.cuboid.rotation is not None]
                orgy = [float(s.cuboid.rotation.origin[1]) for s in base_samples if s.cuboid.rotation is not None]
                orgz = [float(s.cuboid.rotation.origin[2]) for s in base_samples if s.cuboid.rotation is not None]

                ang = avg(angs)
                if snap_angles:
                    ang2 = _snap_to_allowed(ang, _ALLOWED_ROTATION_ANGLES, eps=float(angle_eps))
                    ang = float(ang2)

                rot_out = Rotation(
                    axis=str(rot0.axis),
                    angle=float(ang),
                    origin=(avg(orgx), avg(orgy), avg(orgz)),
                    rescale=bool(rot0.rescale),
                )

        cub_base = Cuboid(fr=fr, to=to, rotation=rot_out)
        base_el = Element(idx=int(seed.idx), raw=dict(ref0.raw), cuboid=cub_base)

        cx, cz = center_xz
        c0 = base_el.center()
        vx = float(c0[0]) - float(cx)
        vz = float(c0[2]) - float(cz)
        r = math.sqrt(vx * vx + vz * vz)
        if r > 1e-9:
            theta = math.degrees(math.atan2(vz, vx))
            theta2 = _snap_theta(theta, str(snap_theta_mode))
            th = math.radians(theta2)
            x2 = float(cx) + r * math.cos(th)
            z2 = float(cz) + r * math.sin(th)
            dx = float(x2) - float(c0[0])
            dz = float(z2) - float(c0[2])

            fx, fy, fz = base_el.cuboid.fr
            tx, ty, tz = base_el.cuboid.to
            fr2 = (fx + dx, fy, fz + dz)
            to2 = (tx + dx, ty, tz + dz)

            rot2 = base_el.cuboid.rotation
            if rot2 is not None:
                ox, oy, oz = rot2.origin
                rot2 = Rotation(axis=rot2.axis, angle=rot2.angle, origin=(ox + dx, oy, oz + dz), rescale=rot2.rescale)

            base_el = Element(idx=int(base_el.idx), raw=dict(base_el.raw), cuboid=Cuboid(fr=fr2, to=to2, rotation=rot2))

        def make_dict_from(el: Element) -> dict:
            d = json.loads(json.dumps(el.raw))
            d["from"] = [_r6(el.cuboid.fr[0]), _r6(el.cuboid.fr[1]), _r6(el.cuboid.fr[2])]
            d["to"] = [_r6(el.cuboid.to[0]), _r6(el.cuboid.to[1]), _r6(el.cuboid.to[2])]

            if "rotation" in d:
                d.pop("rotation", None)

            if el.cuboid.rotation is not None and float(el.cuboid.rotation.angle) != 0.0:
                r0 = el.cuboid.rotation
                r_d: dict = {
                    "origin": [_r6(r0.origin[0]), _r6(r0.origin[1]), _r6(r0.origin[2])],
                    "axis": r0.axis,
                    "angle": _r6(r0.angle),
                }
                if bool(r0.rescale):
                    r_d["rescale"] = True
                d["rotation"] = r_d
            return d

        if base_only:
            out_elements.append(make_dict_from(base_el))
        else:
            if fill_missing:
                for op in ops:
                    el2 = _sym_apply_element(op=op, el=base_el, center_xz=center_xz)
                    out_elements.append(make_dict_from(el2))
            else:
                want = set(str(k) for k in g.matches.keys())
                for op in ops:
                    if str(op.name) not in want:
                        continue
                    el2 = _sym_apply_element(op=op, el=base_el, center_xz=center_xz)
                    out_elements.append(make_dict_from(el2))

    for el in elements:
        if int(el.idx) in used:
            continue
        d = json.loads(json.dumps(el.raw))
        d["from"] = [_r6(el.cuboid.fr[0]), _r6(el.cuboid.fr[1]), _r6(el.cuboid.fr[2])]
        d["to"] = [_r6(el.cuboid.to[0]), _r6(el.cuboid.to[1]), _r6(el.cuboid.to[2])]
        out_elements.append(d)

    if dedupe:
        out_elements = _dedupe_elements(out_elements)

    out = json.loads(json.dumps(model))
    out["elements"] = out_elements
    return out


class SymmetryFixerMainWindow(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Minecraft Model Symmetry Fixer")

        self._source_path: Optional[Path] = None
        self._model: Optional[dict] = None
        self._elements: list[Element] = []
        self._groups: list[GroupMatch] = []

        self._ideal_model: Optional[dict] = None
        self._ideal_elements: list[Element] = []

        self._viewport = ModelViewport()

        controls = self._build_controls()
        output = self._build_output()

        splitter = QtWidgets.QSplitter()
        splitter.setOrientation(QtCore.Qt.Orientation.Horizontal)
        splitter.addWidget(controls)
        splitter.addWidget(self._viewport)
        splitter.addWidget(output)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 0)
        splitter.setCollapsible(0, False)
        splitter.setCollapsible(2, False)

        self.setCentralWidget(splitter)

        self._sync_viewport_render_options()

    def _build_controls(self) -> QtWidgets.QWidget:
        w = QtWidgets.QWidget()
        w.setMinimumWidth(380)

        self._txt_path = QtWidgets.QLineEdit("")
        self._txt_path.setReadOnly(True)

        self._btn_load = QtWidgets.QPushButton("Load JSON")
        self._btn_load.clicked.connect(self._on_load)

        self._center_xz = QtWidgets.QLineEdit("8,8")

        self._include_reflections = QtWidgets.QCheckBox("Include reflections (D4)")
        self._include_reflections.setChecked(True)

        self._mirror_axis = QtWidgets.QComboBox()
        self._mirror_axis.addItem("Mirror: X (MX)", userData="MX")
        self._mirror_axis.addItem("Mirror: Z (MZ)", userData="MZ")
        self._mirror_axis.addItem("Mirror: Diagonal (MD)", userData="MD")
        self._mirror_axis.addItem("Mirror: Anti-diagonal (MAD)", userData="MAD")
        self._mirror_axis.setCurrentIndex(self._mirror_axis.findData("MX"))
        self._mirror_axis.currentIndexChanged.connect(self._sync_viewport_model)

        self._pos_eps = QtWidgets.QDoubleSpinBox()
        self._pos_eps.setRange(0.001, 64.0)
        self._pos_eps.setValue(0.25)
        self._pos_eps.setDecimals(4)

        self._angle_eps = QtWidgets.QDoubleSpinBox()
        self._angle_eps.setRange(1e-6, 90.0)
        self._angle_eps.setValue(2.0)
        self._angle_eps.setDecimals(4)

        self._snap_angles = QtWidgets.QCheckBox("Snap element angles to allowed")
        self._snap_angles.setChecked(True)

        self._snap_theta = QtWidgets.QComboBox()
        for mode in ("none", "auto", "cardinal", "diagonal"):
            self._snap_theta.addItem(_theta_snap_mode_label(mode), userData=mode)
        self._snap_theta.setCurrentIndex(self._snap_theta.findData("auto"))

        self._fill_missing = QtWidgets.QCheckBox("Fill missing symmetric orbit")
        self._fill_missing.setChecked(True)

        self._base_only = QtWidgets.QCheckBox("Base-only IDEAL elements")
        self._base_only.setChecked(False)

        self._dedupe = QtWidgets.QCheckBox("Dedupe identical elements")
        self._dedupe.setChecked(True)

        self._btn_analyze = QtWidgets.QPushButton("Analyze")
        self._btn_analyze.setObjectName("btn_analyze")
        self._btn_analyze.clicked.connect(self._on_analyze)
        self._btn_analyze.setEnabled(False)

        self._btn_solve = QtWidgets.QPushButton("Solve IDEAL")
        self._btn_solve.setObjectName("btn_solve")
        self._btn_solve.clicked.connect(self._on_solve)
        self._btn_solve.setEnabled(False)

        self._btn_save_ideal = QtWidgets.QPushButton("Save IDEAL JSON")
        self._btn_save_ideal.clicked.connect(self._on_save_ideal)
        self._btn_save_ideal.setEnabled(False)

        self._chk_show_ideal = QtWidgets.QCheckBox("Show IDEAL in viewport")
        self._chk_show_ideal.setChecked(False)
        self._chk_show_ideal.stateChanged.connect(self._sync_viewport_model)
        self._chk_show_ideal.setEnabled(False)

        self._vp_faces = QtWidgets.QCheckBox("Faces")
        self._vp_faces.setChecked(True)
        self._vp_faces.stateChanged.connect(self._sync_viewport_render_options)

        self._vp_translucent = QtWidgets.QCheckBox("Translucent faces")
        self._vp_translucent.setChecked(False)
        self._vp_translucent.stateChanged.connect(self._sync_viewport_render_options)

        self._vp_wireframe = QtWidgets.QCheckBox("Wireframe")
        self._vp_wireframe.setChecked(True)
        self._vp_wireframe.stateChanged.connect(self._sync_viewport_render_options)

        self._lbl_status = QtWidgets.QLabel("Ready")
        self._lbl_status.setWordWrap(True)

        file_box = QtWidgets.QGroupBox("Model")
        file_form = QtWidgets.QFormLayout(file_box)
        file_form.setContentsMargins(8, 6, 8, 6)
        file_form.setVerticalSpacing(4)
        file_form.addRow("Path", self._txt_path)
        file_form.addRow(self._btn_load)

        sym_box = QtWidgets.QGroupBox("Symmetry")
        sym_form = QtWidgets.QFormLayout(sym_box)
        sym_form.setContentsMargins(8, 6, 8, 6)
        sym_form.setVerticalSpacing(4)
        sym_form.addRow("Center XZ", self._center_xz)
        sym_form.addRow(self._include_reflections)
        sym_form.addRow(self._mirror_axis)
        sym_form.addRow("Match eps", self._pos_eps)

        solve_box = QtWidgets.QGroupBox("IDEAL solve")
        solve_form = QtWidgets.QFormLayout(solve_box)
        solve_form.setContentsMargins(8, 6, 8, 6)
        solve_form.setVerticalSpacing(4)
        solve_form.addRow(self._snap_angles)
        solve_form.addRow("Angle eps", self._angle_eps)
        solve_form.addRow("Snap center dir", self._snap_theta)
        solve_form.addRow(self._fill_missing)
        solve_form.addRow(self._base_only)
        solve_form.addRow(self._dedupe)
        solve_form.addRow(self._chk_show_ideal)

        self._chk_group_filter = QtWidgets.QCheckBox("Filter viewport to group")
        self._chk_group_filter.setChecked(False)
        self._chk_group_filter.setEnabled(False)
        self._chk_group_filter.stateChanged.connect(self._sync_viewport_model)

        self._spin_group_index = QtWidgets.QSpinBox()
        self._spin_group_index.setRange(0, 0)
        self._spin_group_index.setValue(0)
        self._spin_group_index.setEnabled(False)
        self._spin_group_index.valueChanged.connect(self._sync_viewport_model)

        self._variant_checks: dict[str, QtWidgets.QCheckBox] = {}
        for k in ("R0", "R90", "R180", "R270", "M0", "M90", "M180", "M270"):
            chk = QtWidgets.QCheckBox(k)
            chk.setChecked(True)
            chk.setEnabled(False)
            chk.stateChanged.connect(self._sync_viewport_model)
            self._variant_checks[str(k)] = chk

        gv_box = QtWidgets.QGroupBox("Group view")
        gv_grid = QtWidgets.QGridLayout(gv_box)
        gv_grid.setContentsMargins(8, 6, 8, 6)
        gv_grid.setHorizontalSpacing(8)
        gv_grid.setVerticalSpacing(4)
        gv_grid.addWidget(self._chk_group_filter, 0, 0, 1, 2)
        gv_grid.addWidget(QtWidgets.QLabel("Group"), 1, 0)
        gv_grid.addWidget(self._spin_group_index, 1, 1)
        keys = ("R0", "R90", "R180", "R270", "M0", "M90", "M180", "M270")
        for i, k in enumerate(keys):
            gv_grid.addWidget(self._variant_checks[str(k)], 2 + (i // 2), i % 2)

        viewport_box = QtWidgets.QGroupBox("Viewport")
        viewport_layout = QtWidgets.QVBoxLayout(viewport_box)
        viewport_layout.setContentsMargins(8, 6, 8, 6)
        viewport_layout.setSpacing(4)
        viewport_layout.addWidget(self._vp_faces)
        viewport_layout.addWidget(self._vp_translucent)
        viewport_layout.addWidget(self._vp_wireframe)

        btn_row = QtWidgets.QHBoxLayout()
        btn_row.addWidget(self._btn_analyze)
        btn_row.addWidget(self._btn_solve)

        layout = QtWidgets.QVBoxLayout(w)
        layout.addWidget(file_box)
        layout.addWidget(sym_box)
        layout.addWidget(solve_box)
        layout.addWidget(gv_box)
        layout.addWidget(viewport_box)
        layout.addLayout(btn_row)
        layout.addWidget(self._btn_save_ideal)
        layout.addWidget(self._lbl_status)
        layout.addStretch(1)
        return w

    def _build_output(self) -> QtWidgets.QWidget:
        w = QtWidgets.QWidget()
        w.setMinimumWidth(460)
        w.setSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred, QtWidgets.QSizePolicy.Policy.Expanding)

        self._txt_analysis = QtWidgets.QPlainTextEdit()
        self._txt_analysis.setReadOnly(True)
        self._txt_analysis.setMaximumBlockCount(20000)
        self._txt_analysis.setLineWrapMode(QtWidgets.QPlainTextEdit.LineWrapMode.NoWrap)
        self._txt_analysis.setFont(QtGui.QFont("Consolas", 10))

        self._txt_ideal_json = QtWidgets.QPlainTextEdit()
        self._txt_ideal_json.setReadOnly(True)
        self._txt_ideal_json.setMaximumBlockCount(20000)
        self._txt_ideal_json.setLineWrapMode(QtWidgets.QPlainTextEdit.LineWrapMode.NoWrap)
        self._txt_ideal_json.setFont(QtGui.QFont("Consolas", 10))

        tabs = QtWidgets.QTabWidget()
        tabs.addTab(self._txt_analysis, "Analysis")
        tabs.addTab(self._txt_ideal_json, "IDEAL JSON")

        output_box = QtWidgets.QGroupBox("Output")
        output_layout = QtWidgets.QVBoxLayout(output_box)
        output_layout.setContentsMargins(8, 6, 8, 6)
        output_layout.setSpacing(6)
        output_layout.addWidget(tabs)

        layout = QtWidgets.QVBoxLayout(w)
        layout.addWidget(output_box)
        layout.setContentsMargins(0, 0, 0, 0)
        return w

    def _sync_viewport_render_options(self) -> None:
        faces = bool(self._vp_faces.isChecked())
        self._vp_translucent.setEnabled(faces)
        self._viewport.set_render_options(
            faces=faces,
            faces_translucent=bool(self._vp_translucent.isChecked()),
            wireframe=bool(self._vp_wireframe.isChecked()),
        )

    def _sync_viewport_model(self) -> None:
        if self._chk_show_ideal.isChecked() and self._ideal_elements:
            cubs = [e.cuboid for e in self._ideal_elements]
            cols = [QtGui.QColor(85, 210, 120, 210)] * len(cubs)
            self._viewport.set_cuboids(cubs, colors=cols)
            return

        if self._elements:
            cubs = [e.cuboid for e in self._elements]
            if bool(self._chk_group_filter.isChecked()) and self._groups:
                include_reflections = bool(self._include_reflections.isChecked())
                mirror_gen_op_name = str(self._mirror_axis.currentData() or "MX")
                pairs = _variant_label_op_name_pairs(include_reflections=include_reflections, mirror_gen_op_name=mirror_gen_op_name)
                want_labels = set(k for k, chk in self._variant_checks.items() if bool(chk.isChecked()))
                sel = int(self._spin_group_index.value()) if self._groups else 0
                sel = max(0, min(sel, max(0, len(self._groups) - 1)))
                g = self._groups[sel] if self._groups else None

                visible = [False] * len(self._elements)
                cols = [QtGui.QColor(90, 90, 100, 85)] * len(self._elements)
                palette: dict[str, QtGui.QColor] = {
                    "R0": QtGui.QColor(95, 170, 255, 210),
                    "R90": QtGui.QColor(255, 165, 90, 210),
                    "R180": QtGui.QColor(255, 95, 125, 210),
                    "R270": QtGui.QColor(165, 130, 255, 210),
                    "M0": QtGui.QColor(85, 210, 120, 210),
                    "M90": QtGui.QColor(115, 220, 210, 210),
                    "M180": QtGui.QColor(220, 210, 110, 210),
                    "M270": QtGui.QColor(210, 120, 210, 210),
                }

                if g is not None:
                    for label, op_name in pairs:
                        if str(label) not in want_labels:
                            continue
                        idx = g.matches.get(str(op_name))
                        if idx is None:
                            continue
                        if 0 <= int(idx) < len(visible):
                            visible[int(idx)] = True
                            cols[int(idx)] = palette.get(str(label), QtGui.QColor(95, 170, 255, 210))

                self._viewport.set_cuboids(cubs, colors=cols)
                self._viewport.set_visibility_mask(visible)
            else:
                cols = self._compute_error_colors()
                self._viewport.set_cuboids(cubs, colors=cols)
            return

        self._viewport.set_cuboids([], colors=[])

    def _compute_error_colors(self) -> list[QtGui.QColor]:
        if not self._elements or not self._groups or self._model is None:
            return []

        center_xz = self._get_center_xz()
        include_reflections = bool(self._include_reflections.isChecked())
        mirror_gen_op_name = str(self._mirror_axis.currentData() or "MX")
        ops = _ops_for_variant_set(include_reflections=include_reflections, mirror_gen_op_name=mirror_gen_op_name)

        err: list[float] = [0.0] * len(self._elements)
        for g in self._groups:
            seed = self._elements[g.matches.get("R0", g.seed_idx)]
            for op in ops:
                j = g.matches.get(op.name)
                if j is None:
                    continue
                expected = _sym_apply_element(op=op, el=seed, center_xz=center_xz)
                err[int(j)] = _element_error(expected, self._elements[int(j)])

        max_e = max([float(e) for e in err] + [1e-9])

        cols: list[QtGui.QColor] = []
        for e in err:
            r = max(0.0, min(1.0, float(e) / max_e))
            rr = int(235 * r + 85 * (1.0 - r))
            gg = int(80 * r + 210 * (1.0 - r))
            bb = int(80 * r + 120 * (1.0 - r))
            cols.append(QtGui.QColor(rr, gg, bb, 210))
        return cols

    def _get_center_xz(self) -> tuple[float, float]:
        return _parse_vec2(self._center_xz.text().strip())

    def _on_load(self) -> None:
        path, _filter = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Open Minecraft model JSON",
            str(Path.cwd()),
            "JSON (*.json)",
        )
        if not path:
            return

        try:
            p = Path(path)
            model = json.loads(p.read_text(encoding="utf-8"))
            els = model.get("elements")
            if not isinstance(els, list):
                raise ValueError("JSON missing elements[]")

            elements = [_parse_element(i, el) for i, el in enumerate(els) if isinstance(el, dict)]

            self._source_path = p
            self._model = model
            self._elements = elements
            self._groups = []

            self._ideal_model = None
            self._ideal_elements = []
            self._btn_save_ideal.setEnabled(False)
            self._chk_show_ideal.setEnabled(False)
            self._chk_show_ideal.setChecked(False)

            self._txt_path.setText(str(p))
            self._btn_analyze.setEnabled(True)
            self._btn_solve.setEnabled(False)

            self._chk_group_filter.setChecked(False)
            self._chk_group_filter.setEnabled(False)
            self._spin_group_index.setEnabled(False)
            for chk in self._variant_checks.values():
                chk.setEnabled(False)

            self._txt_analysis.setPlainText(f"Loaded {len(elements)} elements")
            self._txt_ideal_json.setPlainText("")

            self._sync_viewport_model()
            self._lbl_status.setText(f"Loaded: {p.name} ({len(elements)} elements)")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Load failed", str(e))

    def _on_analyze(self) -> None:
        if self._model is None or not self._elements:
            return

        try:
            center_xz = self._get_center_xz()
            pos_eps = float(self._pos_eps.value())
            include_reflections = bool(self._include_reflections.isChecked())
            mirror_gen_op_name = str(self._mirror_axis.currentData() or "MX")

            groups = _match_groups(
                elements=self._elements,
                center_xz=center_xz,
                pos_eps=pos_eps,
                include_reflections=include_reflections,
                mirror_gen_op_name=mirror_gen_op_name,
            )
            self._groups = groups

            lines: list[str] = []
            lines.append(f"Elements: {len(self._elements)}")
            lines.append(f"Groups: {len(groups)}")
            lines.append(f"Center XZ: ({center_xz[0]:g}, {center_xz[1]:g})")
            lines.append(f"Include reflections: {include_reflections}")
            lines.append(f"Mirror generator: {mirror_gen_op_name}")
            lines.append(f"Match eps: {pos_eps:g}")
            lines.append("")

            label_pairs = _variant_label_op_name_pairs(include_reflections=include_reflections, mirror_gen_op_name=mirror_gen_op_name)

            for gi, g in enumerate(groups):
                lines.append(f"Group {gi:03d} seed={g.seed_idx}")
                for label, op_name in label_pairs:
                    idx = g.matches.get(str(op_name))
                    lines.append(f"  {label:>4s} ({str(op_name):>4s}): {('-' if idx is None else str(idx))}")

                def pair_line(a: str, b: str) -> Optional[str]:
                    ia = g.matches.get(str(a))
                    ib = g.matches.get(str(b))
                    if ia is None or ib is None:
                        return None
                    return f"    pair {a:>4s} -> {b:>4s}: {ia} -> {ib}"

                rot_ops = ["R0", "R90", "R180", "R270"]
                for aa, bb in zip(rot_ops, rot_ops[1:] + rot_ops[:1]):
                    pl = pair_line(aa, bb)
                    if pl is not None:
                        lines.append(pl)

                if include_reflections and len(label_pairs) >= 8:
                    m_names = [name for label, name in label_pairs if str(label).startswith("M")]
                    for rr, mm in zip(rot_ops, m_names):
                        pl = pair_line(rr, mm)
                        if pl is not None:
                            lines.append(pl)

            self._txt_analysis.setPlainText("\n".join(lines))
            self._btn_solve.setEnabled(True)

            self._spin_group_index.setRange(0, max(0, len(groups) - 1))
            self._spin_group_index.setValue(0)
            self._spin_group_index.setEnabled(bool(groups))
            self._chk_group_filter.setEnabled(bool(groups))
            for chk in self._variant_checks.values():
                chk.setEnabled(bool(groups))

            self._sync_viewport_model()
            self._lbl_status.setText(f"Analyze done: {len(groups)} groups")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Analyze failed", str(e))

    def _on_solve(self) -> None:
        if self._model is None or not self._elements or not self._groups:
            return

        try:
            center_xz = self._get_center_xz()
            include_reflections = bool(self._include_reflections.isChecked())
            mirror_gen_op_name = str(self._mirror_axis.currentData() or "MX")
            fill_missing = bool(self._fill_missing.isChecked())
            base_only = bool(self._base_only.isChecked())
            snap_theta_mode = str(self._snap_theta.currentData() or "none")
            snap_angles = bool(self._snap_angles.isChecked())
            angle_eps = float(self._angle_eps.value())
            dedupe = bool(self._dedupe.isChecked())

            ideal = _solve_ideal(
                model=self._model,
                elements=self._elements,
                groups=self._groups,
                center_xz=center_xz,
                include_reflections=include_reflections,
                mirror_gen_op_name=mirror_gen_op_name,
                fill_missing=fill_missing,
                base_only=base_only,
                snap_theta_mode=snap_theta_mode,
                snap_angles=snap_angles,
                angle_eps=angle_eps,
                dedupe=dedupe,
            )

            els = ideal.get("elements")
            ideal_elements: list[Element] = []
            if isinstance(els, list):
                for i, el in enumerate(els):
                    if isinstance(el, dict):
                        ideal_elements.append(_parse_element(i, el))

            self._ideal_model = ideal
            self._ideal_elements = ideal_elements

            self._txt_ideal_json.setPlainText(format_minecraft_model_json(ideal))

            self._btn_save_ideal.setEnabled(True)
            self._chk_show_ideal.setEnabled(True)

            self._lbl_status.setText(f"Solved IDEAL: {len(ideal_elements)} elements")
            self._sync_viewport_model()
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Solve failed", str(e))

    def _on_save_ideal(self) -> None:
        if self._ideal_model is None:
            return

        default = Path.cwd() / "model_ideal.json"
        if self._source_path is not None:
            default = self._source_path.with_name(self._source_path.stem + "_IDEAL.json")

        out_path, _filter = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save IDEAL model JSON",
            str(default),
            "JSON (*.json)",
        )
        if not out_path:
            return

        try:
            Path(out_path).write_text(format_minecraft_model_json(self._ideal_model) + "\n", encoding="utf-8", newline="\n")
            self._lbl_status.setText(f"Saved: {out_path}")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Save failed", str(e))


def main() -> None:
    app = QtWidgets.QApplication(sys.argv)
    _apply_dark_theme(app)
    w = SymmetryFixerMainWindow()
    w.resize(1280, 800)
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
