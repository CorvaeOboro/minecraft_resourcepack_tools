"""Minecraft Diamond/Pyramid Plane Solver (UI)

PySide6 UI for generating Minecraft block model JSON that uses *thin cuboids as
planes* and (optionally) an alpha cutout atlas to approximate pyramid/diamond
geometry.

This tool focuses on *plane-based* elements:
- Each plane is exported as a very thin cuboid with only one (or two) faces.
- A texture with alpha can cut away the unwanted parts of the plane.
- Optional atlas mode generates UV islands and can generate a cutout atlas PNG.

Presets
- Unilateral bipyramid ("diamond")
- 4-sided pyramid
- Mode: 2 stacked bipyramids (used for variation/assembly workflows)

TOOLSGROUP::MODEL
SORTGROUP::7
SORTPRIORITY::75
STATUS::active
VERSION::20260308

Expected inputs
- UI-driven parameters:
  - Center, join/base Y, half-base
  - Slope angles (mapped onto Minecraft-allowed rotations)
  - Plane thickness, rescale/double-sided/shade options
  - Texture ids for top/bottom
  - Optional atlas UV settings (tile size/gap/border)

Outputs
- Displays generated `elements` and full model JSON
- Optional generated cutout atlas preview
- "Save JSON" writes model JSON to disk
- "Save Atlas PNG" writes the generated atlas PNG (when enabled)

Usage
- Simple:
  python DEV/mc_diamond_pyramid_solver_ui.py

- Project example:
  - Use `arborea:*` texture ids (e.g. `arborea:blocks/glimmerfluidstatic`) in the UI.
  - Save JSON into:
    `arborea_1_19_2/src/main/resources/assets/arborea/models/`
"""

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
PresetId = Literal["diamond", "pyramid", "mode"]
PyramidStyle = Literal["short", "tall"]
PyramidOrientation = Literal["up", "down"]

_ALLOWED_ROTATION_ANGLES = (-45.0, -22.5, 22.5, 45.0)
_SLOPE_ANGLES = (22.5, 45.0, 67.5)


def _snap_to_allowed(v: float, allowed: tuple[float, ...], *, eps: float = 1e-6) -> float:
    vv = float(v)
    best = min(allowed, key=lambda a: abs(float(a) - vv))
    if abs(float(best) - vv) <= float(eps):
        return float(best)
    raise ValueError(f"Value {vv:g} not in allowed set: {allowed}")


def _slope_to_build_params(*, slope_angle_deg: float) -> tuple[float, PyramidStyle]:
    s = abs(float(slope_angle_deg))
    s = _snap_to_allowed(s, _SLOPE_ANGLES)
    if s > 45.0:
        rot = 90.0 - s
        rot = _snap_to_allowed(rot, tuple(abs(a) for a in _ALLOWED_ROTATION_ANGLES))
        return rot, "tall"
    return s, "short"

_FAVORED_SIDE_UV_ROTATIONS: dict[tuple[PresetId, int], int] = {
    ("pyramid", 1): 0,
    ("pyramid", 2): 180,
    ("pyramid", 3): 90,
    ("pyramid", 4): 270,
    ("diamond", 0): 180,
    ("diamond", 1): 0,
    ("diamond", 2): 270,
    ("diamond", 3): 90,
    ("diamond", 4): 180,
    ("diamond", 5): 180,
    ("diamond", 6): 180,
    ("diamond", 7): 180,
}

_TALL_PYRAMID_FACE_UV_ROTATION: dict[str, int] = {
    "north": 0,
    "south": 180,
    "east": 90,
    "west": 270,
}


def _sin_deg(v: float) -> float:
    return math.sin(math.radians(v))


def _deg_to_rad(deg: float) -> float:
    return deg * (math.pi / 180.0)


def _rot_inv_xyz(*, axis: Axis, angle_deg: float, x: float, y: float, z: float) -> tuple[float, float, float]:
    if angle_deg == 0.0:
        return x, y, z

    a = _deg_to_rad(angle_deg)
    c = math.cos(a)
    s = math.sin(a)

    if axis == "x":
        yy = (c * y) + (s * z)
        zz = (-s * y) + (c * z)
        return x, yy, zz

    if axis == "y":
        xx = (c * x) + (s * z)
        zz = (-s * x) + (c * z)
        return xx, y, zz

    if axis == "z":
        xx = (c * x) + (s * y)
        yy = (-s * x) + (c * y)
        return xx, yy, z

    raise ValueError(f"Invalid axis: {axis}")


def _fwd_rotate_point(*, c: "Cuboid", p: tuple[float, float, float]) -> tuple[float, float, float]:
    x, y, z = p
    if c.rotation is None or float(c.rotation.angle) == 0.0:
        return x, y, z

    ox, oy, oz = c.rotation.origin
    x -= ox
    y -= oy
    z -= oz

    if bool(c.rotation.rescale):
        a = abs(float(c.rotation.angle))
        cos_a = math.cos(math.radians(a))
        if cos_a > 1e-12:
            s = 1.0 / cos_a
            if c.rotation.axis == "x":
                y *= s
                z *= s
            elif c.rotation.axis == "y":
                x *= s
                z *= s
            else:
                x *= s
                y *= s

    x, y, z = _rot_fwd_xyz(axis=c.rotation.axis, angle_deg=c.rotation.angle, x=x, y=y, z=z)
    x += ox
    y += oy
    z += oz
    return x, y, z


def _point_on_face_unrotated(*, c: "Cuboid", face: str, u: float, v: float) -> tuple[float, float, float]:
    a_axis, b_axis, amin, amax, bmin, bmax = _face_axes_and_bounds(c=c, face=face)
    aa = amin + (amax - amin) * float(u)
    bb = bmin + (bmax - bmin) * float(v)

    fx, fy, fz = c.fr
    tx, ty, tz = c.to
    if face == "up":
        fixed = float(ty)
        if a_axis == "x":
            return (aa, fixed, bb)
        return (bb, fixed, aa)
    if face == "down":
        fixed = float(fy)
        if a_axis == "x":
            return (aa, fixed, bb)
        return (bb, fixed, aa)
    if face == "north":
        fixed = float(fz)
        return (aa, bb, fixed)
    if face == "south":
        fixed = float(tz)
        return (aa, bb, fixed)
    if face == "east":
        fixed = float(tx)
        return (fixed, bb, aa)
    if face == "west":
        fixed = float(fx)
        return (fixed, bb, aa)
    raise ValueError(f"Invalid face: {face}")


def _plane_from_face(*, c: "Cuboid", face: str) -> tuple[tuple[float, float, float], float]:
    fx, fy, fz = c.fr
    tx, ty, tz = c.to

    if face == "up":
        p0u = (fx, ty, fz)
        p1u = (tx, ty, fz)
        p2u = (fx, ty, tz)
    elif face == "down":
        p0u = (fx, fy, fz)
        p1u = (fx, fy, tz)
        p2u = (tx, fy, fz)
    elif face == "north":
        p0u = (fx, fy, fz)
        p1u = (tx, fy, fz)
        p2u = (fx, ty, fz)
    elif face == "south":
        p0u = (fx, fy, tz)
        p1u = (fx, ty, tz)
        p2u = (tx, fy, tz)
    elif face == "east":
        p0u = (tx, fy, fz)
        p1u = (tx, ty, fz)
        p2u = (tx, fy, tz)
    elif face == "west":
        p0u = (fx, fy, fz)
        p1u = (fx, fy, tz)
        p2u = (fx, ty, fz)
    else:
        raise ValueError(f"Invalid face: {face}")

    p0 = _fwd_rotate_point(c=c, p=p0u)
    p1 = _fwd_rotate_point(c=c, p=p1u)
    p2 = _fwd_rotate_point(c=c, p=p2u)

    ax, ay, az = (p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2])
    bx, by, bz = (p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2])
    nx = (ay * bz) - (az * by)
    ny = (az * bx) - (ax * bz)
    nz = (ax * by) - (ay * bx)

    nlen = math.sqrt((nx * nx) + (ny * ny) + (nz * nz))
    if nlen > 1e-12:
        inv = 1.0 / nlen
        nx *= inv
        ny *= inv
        nz *= inv

    d = -((nx * p0[0]) + (ny * p0[1]) + (nz * p0[2]))
    return (nx, ny, nz), float(d)


def _plane_side(n: tuple[float, float, float], d: float, p: tuple[float, float, float]) -> float:
    return (n[0] * p[0]) + (n[1] * p[1]) + (n[2] * p[2]) + float(d)


def _rot_fwd_xyz(*, axis: Axis, angle_deg: float, x: float, y: float, z: float) -> tuple[float, float, float]:
    return _rot_inv_xyz(axis=axis, angle_deg=-float(angle_deg), x=x, y=y, z=z)


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

    def contains_point(self, p: tuple[float, float, float]) -> bool:
        x, y, z = p

        if self.rotation is not None:
            ox, oy, oz = self.rotation.origin
            x -= ox
            y -= oy
            z -= oz

            x, y, z = _rot_inv_xyz(axis=self.rotation.axis, angle_deg=self.rotation.angle, x=x, y=y, z=z)

            if bool(self.rotation.rescale):
                a = abs(float(self.rotation.angle))
                cos_a = math.cos(math.radians(a))
                if cos_a > 1e-12:
                    if self.rotation.axis == "x":
                        y *= cos_a
                        z *= cos_a
                    elif self.rotation.axis == "y":
                        x *= cos_a
                        z *= cos_a
                    else:
                        x *= cos_a
                        y *= cos_a

            x += ox
            y += oy
            z += oz

        fx, fy, fz = self.fr
        tx, ty, tz = self.to
        return (fx <= x <= tx) and (fy <= y <= ty) and (fz <= z <= tz)


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
        "QPushButton#btn_generate { background: #1b3326; border: 1px solid #2d5b3f; }"
        "QPushButton#btn_generate:hover { border: 1px solid #3d7a55; }"
        "QPushButton#btn_clear { background: #241a2c; border: 1px solid #4a2c63; }"
        "QPushButton#btn_clear:hover { border: 1px solid #6a3b90; }"
        "QPushButton#btn_save { background: #241a2c; border: 1px solid #4a2c63; }"
        "QPushButton#btn_save:hover { border: 1px solid #6a3b90; }"
        "QSplitter::handle { background: #0b0b0d; }"
    )


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

        self._debug_face_colors = False

        self._target_enabled = False
        self._target_segments: list[tuple[tuple[float, float, float], tuple[float, float, float], QtGui.QColor]] = []

        self._draw_faces = True
        self._faces_translucent = False
        self._draw_wireframe = True

        self._uv_debug_enabled = False
        self._uv_debug_info: list[tuple[str, Optional[int], str]] = []

    def set_render_options(self, *, faces: bool, faces_translucent: bool, wireframe: bool) -> None:
        self._draw_faces = bool(faces)
        self._faces_translucent = bool(faces_translucent)
        self._draw_wireframe = bool(wireframe)
        self.update()

    def clear_cuboids(self) -> None:
        self._cuboids = []
        self._visible_mask = []
        self.update()

    def set_cuboids(self, cuboids: Iterable[Cuboid]) -> None:
        self._cuboids = list(cuboids)
        self._visible_mask = [True] * len(self._cuboids)
        self.update()

    def set_visibility_mask(self, mask: list[bool]) -> None:
        if len(mask) != len(self._cuboids):
            return
        self._visible_mask = list(bool(x) for x in mask)
        self.update()

    def set_debug_face_colors(self, enabled: bool) -> None:
        self._debug_face_colors = bool(enabled)
        self.update()

    def set_target_wireframe(
        self,
        *,
        enabled: bool,
        segments: list[tuple[tuple[float, float, float], tuple[float, float, float], QtGui.QColor]],
    ) -> None:
        self._target_enabled = bool(enabled)
        self._target_segments = list(segments)
        self.update()
        self.update()

    def set_uv_debug(self, *, enabled: bool, info: list[tuple[str, Optional[int], str]]) -> None:
        self._uv_debug_enabled = bool(enabled)
        self._uv_debug_info = list(info)
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

        def rotate_about_axis(
            *,
            axis: str,
            angle_deg: float,
            origin: tuple[float, float, float],
            p: tuple[float, float, float],
        ) -> tuple[float, float, float]:
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
        uv_overlays: list[tuple[float, QtCore.QPointF, QtCore.QPointF, str]] = []

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

        c0 = (0.0, 0.0, 0.0)
        c1 = (16.0, 0.0, 0.0)
        c2 = (16.0, 16.0, 0.0)
        c3 = (0.0, 16.0, 0.0)
        c4 = (0.0, 0.0, 16.0)
        c5 = (16.0, 0.0, 16.0)
        c6 = (16.0, 16.0, 16.0)
        c7 = (0.0, 16.0, 16.0)

        cube_col = QtGui.QColor(220, 220, 230, 90)
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

        cub_col = QtGui.QColor(95, 170, 255, 190)
        face_base = QtGui.QColor(95, 170, 255, 200 if not self._faces_translucent else 70)
        face_debug_cols = {
            "north": QtGui.QColor(230, 90, 90, 180),
            "south": QtGui.QColor(230, 150, 90, 180),
            "east": QtGui.QColor(90, 150, 230, 180),
            "west": QtGui.QColor(90, 230, 150, 180),
            "up": QtGui.QColor(230, 230, 110, 180),
            "down": QtGui.QColor(160, 160, 170, 180),
        }

        for i_c, c in enumerate(self._cuboids):
            if i_c < len(self._visible_mask) and not bool(self._visible_mask[i_c]):
                continue
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

                    if self._debug_face_colors:
                        face_name = "north"
                        if (i0, i1, i2, i3) == (4, 5, 6, 7):
                            face_name = "south"
                        elif (i0, i1, i2, i3) == (0, 1, 5, 4):
                            face_name = "down"
                        elif (i0, i1, i2, i3) == (1, 2, 6, 5):
                            face_name = "east"
                        elif (i0, i1, i2, i3) == (2, 3, 7, 6):
                            face_name = "up"
                        elif (i0, i1, i2, i3) == (3, 0, 4, 7):
                            face_name = "west"

                        col = face_debug_cols[face_name]

                    poly = QtGui.QPolygonF([p0, p1, p2, p3])
                    depth = (cam_pts[i0][2] + cam_pts[i1][2] + cam_pts[i2][2] + cam_pts[i3][2]) / 4.0
                    faces.append((depth, poly, col))

                    if self._uv_debug_enabled and i_c < len(self._uv_debug_info):
                        outward_face, face_rot, label = self._uv_debug_info[i_c]

                        face_name = "north"
                        if (i0, i1, i2, i3) == (4, 5, 6, 7):
                            face_name = "south"
                        elif (i0, i1, i2, i3) == (0, 1, 5, 4):
                            face_name = "down"
                        elif (i0, i1, i2, i3) == (1, 2, 6, 5):
                            face_name = "east"
                        elif (i0, i1, i2, i3) == (2, 3, 7, 6):
                            face_name = "up"
                        elif (i0, i1, i2, i3) == (3, 0, 4, 7):
                            face_name = "west"

                        if str(face_name) == str(outward_face):
                            # Compute face-center and the "texture up" arrow direction in world-space.
                            cxm = (fx + tx) * 0.5
                            cym = (fy + ty) * 0.5
                            czm = (fz + tz) * 0.5

                            def base_center_and_axes() -> tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]:
                                if face_name in ("north", "south"):
                                    zf = float(fz) if face_name == "north" else float(tz)
                                    center0 = (float(cxm), float(cym), zf)
                                    u0 = (1.0, 0.0, 0.0)
                                    v0 = (0.0, 1.0, 0.0)
                                    return center0, u0, v0
                                if face_name in ("east", "west"):
                                    xf = float(tx) if face_name == "east" else float(fx)
                                    center0 = (xf, float(cym), float(czm))
                                    u0 = (0.0, 0.0, 1.0)
                                    v0 = (0.0, 1.0, 0.0)
                                    return center0, u0, v0
                                if face_name in ("up", "down"):
                                    yf = float(ty) if face_name == "up" else float(fy)
                                    center0 = (float(cxm), yf, float(czm))
                                    u0 = (1.0, 0.0, 0.0)
                                    v0 = (0.0, 0.0, 1.0)
                                    return center0, u0, v0
                                return (float(cxm), float(cym), float(czm)), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)

                            center0, u0, v0 = base_center_and_axes()
                            cu = (center0[0] + u0[0], center0[1] + u0[1], center0[2] + u0[2])
                            cv = (center0[0] + v0[0], center0[1] + v0[1], center0[2] + v0[2])
                            center_w = _fwd_rotate_point(c=c, p=center0)
                            cu_w = _fwd_rotate_point(c=c, p=cu)
                            cv_w = _fwd_rotate_point(c=c, p=cv)
                            u_w = (cu_w[0] - center_w[0], cu_w[1] - center_w[1], cu_w[2] - center_w[2])
                            v_w = (cv_w[0] - center_w[0], cv_w[1] - center_w[1], cv_w[2] - center_w[2])

                            r = 0 if face_rot is None else int(face_rot)
                            # v points down in texture space. "Up arrow" is -v.
                            if r == 90:
                                uvx, uvy = (-1.0, 0.0)
                            elif r == 180:
                                uvx, uvy = (0.0, 1.0)
                            elif r == 270:
                                uvx, uvy = (1.0, 0.0)
                            else:
                                uvx, uvy = (0.0, -1.0)

                            aw = (
                                (u_w[0] * uvx) + (v_w[0] * uvy),
                                (u_w[1] * uvx) + (v_w[1] * uvy),
                                (u_w[2] * uvx) + (v_w[2] * uvy),
                            )
                            a_len = math.sqrt((aw[0] * aw[0]) + (aw[1] * aw[1]) + (aw[2] * aw[2]))
                            if a_len > 1e-9:
                                inv = 1.0 / a_len
                                aw = (aw[0] * inv, aw[1] * inv, aw[2] * inv)

                            p_center = project_cam(to_camera(center_w))
                            tip_w = (center_w[0] + (aw[0] * 2.6), center_w[1] + (aw[1] * 2.6), center_w[2] + (aw[2] * 2.6))
                            p_tip = project_cam(to_camera(tip_w))
                            if p_center is not None and p_tip is not None:
                                uv_overlays.append((float(depth), p_center, p_tip, str(label)))

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
                    add_seg(pts[ia], pts[ib], cub_col)

        if self._draw_faces and faces:
            faces.sort(key=lambda it: it[0], reverse=True)
            painter.setPen(QtCore.Qt.PenStyle.NoPen)
            for _depth, poly, col in faces:
                painter.setBrush(QtGui.QBrush(col))
                painter.drawPolygon(poly)

        if self._uv_debug_enabled and uv_overlays:
            uv_overlays.sort(key=lambda it: it[0], reverse=True)
            pen = QtGui.QPen(QtGui.QColor(255, 255, 255, 230))
            pen.setWidthF(2.2)
            painter.setPen(pen)
            painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)
            for _depth, p_center, p_tip, label in uv_overlays:
                painter.drawLine(p_center, p_tip)

                dx = p_tip.x() - p_center.x()
                dy = p_tip.y() - p_center.y()
                dlen = math.sqrt((dx * dx) + (dy * dy))
                if dlen > 1e-6:
                    ux = dx / dlen
                    uy = dy / dlen
                    px = -uy
                    py = ux
                    ah = 8.0
                    aw = 5.0
                    h0 = QtCore.QPointF(p_tip.x() - (ux * ah) + (px * aw), p_tip.y() - (uy * ah) + (py * aw))
                    h1 = QtCore.QPointF(p_tip.x() - (ux * ah) - (px * aw), p_tip.y() - (uy * ah) - (py * aw))
                    painter.drawLine(p_tip, h0)
                    painter.drawLine(p_tip, h1)

                font = painter.font()
                font.setBold(True)
                painter.setFont(font)

                rect = QtCore.QRectF(p_center.x() - 16.0, p_center.y() - 10.0, 32.0, 20.0)
                painter.fillRect(rect, QtGui.QColor(0, 0, 0, 120))
                painter.drawText(rect, QtCore.Qt.AlignmentFlag.AlignCenter, label)

        segments.sort(key=lambda s: s[3], reverse=True)

        if self._target_enabled and self._target_segments:
            for a, b, col in self._target_segments:
                ca = to_camera(a)
                cb = to_camera(b)
                pa = project_cam(ca)
                pb = project_cam(cb)
                if pa is None or pb is None:
                    continue
                segments.append((a, b, col, (ca[2] + cb[2]) * 0.5))

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


def _parse_vec2(raw: str) -> tuple[float, float]:
    parts = [p.strip() for p in raw.split(",")]
    if len(parts) != 2:
        raise ValueError("Expected x,z")
    return float(parts[0]), float(parts[1])


def _faces_for_plane(*, outward_face: str, double_sided: bool) -> list[str]:
    if not double_sided:
        return [outward_face]

    opp = {
        "north": "south",
        "south": "north",
        "east": "west",
        "west": "east",
        "up": "down",
        "down": "up",
    }[outward_face]

    return [outward_face, opp]


def _atlas_grid_count(n: int) -> int:
    n = max(1, int(n))
    return int(math.ceil(math.sqrt(float(n))))


def _atlas_layout_px(*, n: int, tile_px: int, gap_px: int, total_px: int, border_px: int = 0) -> tuple[int, int, int]:
    g = _atlas_grid_count(n)

    gap_px = max(0, int(gap_px))
    border_px = max(0, int(border_px))
    tile_px = max(1, int(tile_px))
    total_px = int(total_px)

    if total_px > 0:
        usable = total_px - (2 * border_px) - ((g - 1) * gap_px)
        tile_px = max(1, usable // g)

    atlas_px = (g * tile_px) + ((g - 1) * gap_px) + (2 * border_px)
    return g, tile_px, atlas_px


def _atlas_uv_rect(*, index: int, n: int, tile_px: int, gap_px: int, total_px: int, border_px: int = 0) -> list[float]:
    g, tile_px, atlas_px = _atlas_layout_px(n=n, tile_px=tile_px, gap_px=gap_px, total_px=total_px, border_px=border_px)

    tile_uv = 16.0 * (float(tile_px) / float(atlas_px))
    gap_uv = 16.0 * (float(max(0, int(gap_px))) / float(atlas_px))
    border_uv = 16.0 * (float(max(0, int(border_px))) / float(atlas_px))

    ix = int(index) % g
    iy = int(index) // g

    step = tile_uv + gap_uv
    u0 = border_uv + float(ix) * step
    v0 = border_uv + float(iy) * step
    return [u0, v0, u0 + tile_uv, v0 + tile_uv]


def _inv_rotate_point(*, c: Cuboid, p: tuple[float, float, float]) -> tuple[float, float, float]:
    x, y, z = p
    if c.rotation is None or float(c.rotation.angle) == 0.0:
        return x, y, z

    ox, oy, oz = c.rotation.origin
    x -= ox
    y -= oy
    z -= oz
    x, y, z = _rot_inv_xyz(axis=c.rotation.axis, angle_deg=c.rotation.angle, x=x, y=y, z=z)

    if bool(c.rotation.rescale):
        a = abs(float(c.rotation.angle))
        cos_a = math.cos(math.radians(a))
        if cos_a > 1e-12:
            if c.rotation.axis == "x":
                y *= cos_a
                z *= cos_a
            elif c.rotation.axis == "y":
                x *= cos_a
                z *= cos_a
            else:
                x *= cos_a
                y *= cos_a

    x += ox
    y += oy
    z += oz
    return x, y, z


def _face_axes_and_bounds(*, c: Cuboid, face: str) -> tuple[str, str, float, float, float, float]:
    fx, fy, fz = c.fr
    tx, ty, tz = c.to

    if face == "up" or face == "down":
        return "x", "z", float(fx), float(tx), float(fz), float(tz)
    if face == "north" or face == "south":
        return "x", "y", float(fx), float(tx), float(fy), float(ty)
    if face == "east" or face == "west":
        return "z", "y", float(fz), float(tz), float(fy), float(ty)
    raise ValueError(f"Invalid face: {face}")


def _get_axis_value(axis: str, p: tuple[float, float, float]) -> float:
    if axis == "x":
        return float(p[0])
    if axis == "y":
        return float(p[1])
    if axis == "z":
        return float(p[2])
    raise ValueError(f"Invalid axis: {axis}")


def _make_cutout_atlas_image(
    *,
    preset: PresetId,
    planes: list[tuple[Cuboid, str]],
    face_rots: Optional[list[Optional[int]]] = None,
    center_xz: tuple[float, float],
    join_y: float,
    half_base: float,
    top_angle: float,
    bottom_angle: float,
    tile_px: int,
    gap_px: int = 0,
    border_px: int = 0,
    debug: bool = False,
    pad_alpha_1px: bool = True,
    pyramid_style: PyramidStyle = "short",
    pyramid_orientation: PyramidOrientation = "up",
) -> QtGui.QImage:
    g = _atlas_grid_count(len(planes))
    tile_px = max(8, int(tile_px))
    gap_px = max(0, int(gap_px))
    border_px = max(0, int(border_px))
    w = (g * tile_px) + ((g - 1) * gap_px) + (2 * border_px)
    h = (g * tile_px) + ((g - 1) * gap_px) + (2 * border_px)

    img = QtGui.QImage(w, h, QtGui.QImage.Format.Format_ARGB32)
    img.fill(QtCore.Qt.GlobalColor.transparent)

    p = QtGui.QPainter(img)
    p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)
    p.setPen(QtCore.Qt.PenStyle.NoPen)
    p.setBrush(QtGui.QBrush(QtGui.QColor(255, 255, 255, 255)))

    def apex_height(half: float, slope_angle_deg: float) -> float:
        a = abs(float(slope_angle_deg))
        t = math.tan(math.radians(a))
        if t <= 1e-9:
            return 0.0
        return float(half) * t

    cx, cz = center_xz

    def _tex_from_face_uv(u: float, v: float, rot_deg: Optional[int]) -> tuple[float, float]:
        r = 0 if rot_deg is None else int(rot_deg)
        u = float(u)
        v = float(v)
        if r == 90:
            return v, 1.0 - u
        if r == 180:
            return 1.0 - u, 1.0 - v
        if r == 270:
            return 1.0 - v, u
        return u, v

    def _face_from_tex_uv(u: float, v: float, rot_deg: Optional[int]) -> tuple[float, float]:
        r = 0 if rot_deg is None else int(rot_deg)
        u = float(u)
        v = float(v)
        if r == 90:
            return 1.0 - v, u
        if r == 180:
            return 1.0 - u, 1.0 - v
        if r == 270:
            return v, 1.0 - u
        return u, v

    def pyramid_face_triangle(idx: int) -> Optional[tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]]:
        base_y = float(join_y)
        h0 = apex_height(half_base, top_angle)
        apex_y = (base_y + h0) if pyramid_orientation == "up" else (base_y - h0)
        apex = (cx, apex_y, cz)
        nw = (cx - half_base, base_y, cz - half_base)
        ne = (cx + half_base, base_y, cz - half_base)
        se = (cx + half_base, base_y, cz + half_base)
        sw = (cx - half_base, base_y, cz + half_base)

        # plane order: base, ramp_south, ramp_north, ramp_east, ramp_west
        if idx == 1:
            return (apex, se, sw)
        if idx == 2:
            return (apex, nw, ne)
        if idx == 3:
            return (apex, ne, se)
        if idx == 4:
            return (apex, sw, nw)
        return None

    def diamond_face_triangle(idx: int) -> tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]:
        jy = float(join_y)
        th = apex_height(half_base, top_angle)
        bh = apex_height(half_base, bottom_angle)
        top_apex = (cx, jy + th, cz)
        bot_apex = (cx, jy - bh, cz)
        nw = (cx - half_base, jy, cz - half_base)
        ne = (cx + half_base, jy, cz - half_base)
        se = (cx + half_base, jy, cz + half_base)
        sw = (cx - half_base, jy, cz + half_base)

        # plane order: top_north, top_south, top_west, top_east, bottom_north, bottom_south, bottom_west, bottom_east
        if idx == 0:
            return (top_apex, nw, ne)
        if idx == 1:
            return (top_apex, se, sw)
        if idx == 2:
            return (top_apex, sw, nw)
        if idx == 3:
            return (top_apex, ne, se)
        if idx == 4:
            return (bot_apex, ne, nw)
        if idx == 5:
            return (bot_apex, sw, se)
        if idx == 6:
            return (bot_apex, nw, sw)
        return (bot_apex, se, ne)

    def diamond_face_triangle_at(
        *,
        jy: float,
        top_slope: float,
        bottom_slope: float,
        idx: int,
    ) -> tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]:
        th = apex_height(half_base, top_slope)
        bh = apex_height(half_base, bottom_slope)
        top_apex = (cx, jy + th, cz)
        bot_apex = (cx, jy - bh, cz)
        nw = (cx - half_base, jy, cz - half_base)
        ne = (cx + half_base, jy, cz - half_base)
        se = (cx + half_base, jy, cz + half_base)
        sw = (cx - half_base, jy, cz + half_base)

        # plane order: top_north, top_south, top_west, top_east, bottom_north, bottom_south, bottom_west, bottom_east
        if idx == 0:
            return (top_apex, nw, ne)
        if idx == 1:
            return (top_apex, se, sw)
        if idx == 2:
            return (top_apex, sw, nw)
        if idx == 3:
            return (top_apex, ne, se)
        if idx == 4:
            return (bot_apex, ne, nw)
        if idx == 5:
            return (bot_apex, sw, se)
        if idx == 6:
            return (bot_apex, nw, sw)
        return (bot_apex, se, ne)

    def face_triangle_for_plane_index(
        idx: int,
    ) -> tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]:
        ii = int(idx)
        if preset != "mode":
            return diamond_face_triangle(ii)

        # Two stacked bipyramids:
        # A: top=67.5, bottom=45 at join_y
        # B: top=45, bottom=67.5 at join_y2 (tip-to-tip with A)
        top1_slope = 67.5
        bot1_slope = 45.0
        top2_slope = 45.0
        bot2_slope = 67.5

        join_y2 = float(join_y) - apex_height(half_base, bot1_slope) - apex_height(half_base, top2_slope)
        grp = 0 if ii < 8 else 1
        local = ii % 8
        if grp == 0:
            return diamond_face_triangle_at(jy=float(join_y), top_slope=top1_slope, bottom_slope=bot1_slope, idx=local)
        return diamond_face_triangle_at(jy=float(join_y2), top_slope=top2_slope, bottom_slope=bot2_slope, idx=local)

    pyr_planes: Optional[list[tuple[tuple[float, float, float], float]]] = None
    pyr_inside: Optional[tuple[float, float, float]] = None
    if preset == "pyramid" and len(planes) >= 5:
        base_y = float(join_y)
        h0 = apex_height(half_base, top_angle)
        if pyramid_orientation == "up":
            pyr_inside = (float(cx), float(base_y) + (float(h0) * 0.35), float(cz))
        else:
            pyr_inside = (float(cx), float(base_y) - (float(h0) * 0.35), float(cz))
        pyr_planes = []
        for j in range(1, 5):
            cj, facej = planes[j]
            n, d = _plane_from_face(c=cj, face=facej)
            if pyr_inside is not None and _plane_side(n, d, pyr_inside) > 0.0:
                n = (-n[0], -n[1], -n[2])
                d = -float(d)
            pyr_planes.append((n, float(d)))

    def _tile_dilate_mask_1px(mask0: list[bool], w0: int, h0: int) -> list[bool]:
        out0 = list(mask0)
        for yy in range(h0):
            for xx in range(w0):
                if mask0[yy * w0 + xx]:
                    continue
                for oy in (-1, 0, 1):
                    y2 = yy + oy
                    if y2 < 0 or y2 >= h0:
                        continue
                    row = y2 * w0
                    for ox in (-1, 0, 1):
                        x2 = xx + ox
                        if x2 < 0 or x2 >= w0:
                            continue
                        if mask0[row + x2]:
                            out0[yy * w0 + xx] = True
                            oy = 2
                            break
                    if oy == 2:
                        break
        return out0

    for i, (cub, outward_face) in enumerate(planes):
        ix = i % g
        iy = i // g
        x0 = border_px + ix * (tile_px + gap_px)
        y0 = border_px + iy * (tile_px + gap_px)

        rot_i: Optional[int] = None
        if face_rots is not None and 0 <= int(i) < len(face_rots):
            rot_i = face_rots[int(i)]

        if preset == "pyramid" and i == 0:
            # Base plane: full tile opaque.
            p.drawRect(QtCore.QRectF(x0, y0, tile_px, tile_px))
            continue

        if preset == "pyramid":
            if pyr_planes is None or pyr_inside is None:
                continue

            base_y = float(join_y)
            eps = 2e-3

            mask0: list[bool] = [False] * (tile_px * tile_px)
            edge_id: list[int] = [-1] * (tile_px * tile_px)
            self_j = i - 1

            for py in range(tile_px):
                v = (py + 0.5) / float(tile_px)
                for px in range(tile_px):
                    u = (px + 0.5) / float(tile_px)
                    uf, vf = _face_from_tex_uv(u, v, rot_i)
                    p0u = _point_on_face_unrotated(c=cub, face=outward_face, u=uf, v=vf)
                    pw = _fwd_rotate_point(c=cub, p=p0u)

                    idx = py * tile_px + px

                    if pyramid_orientation == "up":
                        if pw[1] + eps < base_y:
                            continue
                    else:
                        if pw[1] - eps > base_y:
                            continue

                    keep = True
                    for j, (n, d) in enumerate(pyr_planes):
                        if _plane_side(n, d, pw) > eps:
                            keep = False
                            break
                    if not keep:
                        continue

                    mask0[idx] = True

                    mb = pw[1] - base_y
                    best_m = float(mb)
                    best_id = 4
                    for j, (n, d) in enumerate(pyr_planes):
                        if j == self_j:
                            continue
                        sd = _plane_side(n, d, pw)
                        m = -float(sd)
                        if m < best_m:
                            best_m = m
                            best_id = int(j)
                    edge_id[idx] = int(best_id)

            mask = mask0 if not bool(pad_alpha_1px) else _tile_dilate_mask_1px(mask0, tile_px, tile_px)

            cols = [
                QtGui.QColor(255, 70, 70, 210),
                QtGui.QColor(70, 255, 70, 210),
                QtGui.QColor(70, 150, 255, 210),
                QtGui.QColor(255, 255, 70, 210),
                QtGui.QColor(255, 90, 220, 210),
            ]

            for py in range(tile_px):
                for px in range(tile_px):
                    idx = py * tile_px + px
                    if not mask[idx]:
                        continue

                    col = QtGui.QColor(255, 255, 255, 255)
                    if debug and mask0[idx]:
                        boundary = False
                        if px <= 0 or px >= tile_px - 1 or py <= 0 or py >= tile_px - 1:
                            boundary = True
                        else:
                            if not mask0[idx - 1] or not mask0[idx + 1] or not mask0[idx - tile_px] or not mask0[idx + tile_px]:
                                boundary = True
                        if boundary:
                            eid = edge_id[idx]
                            if 0 <= eid < len(cols):
                                col = cols[eid]
                    img.setPixelColor(x0 + px, y0 + py, col)
            continue

        tri_w: Optional[tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]] = face_triangle_for_plane_index(i)
        v0w, v1w, v2w = tri_w
        v0 = _inv_rotate_point(c=cub, p=v0w)
        v1 = _inv_rotate_point(c=cub, p=v1w)
        v2 = _inv_rotate_point(c=cub, p=v2w)

        a_axis, b_axis, amin, amax, bmin, bmax = _face_axes_and_bounds(c=cub, face=outward_face)
        if abs(amax - amin) > 1e-9:
            ax0 = (_get_axis_value(a_axis, v0) - amin) / (amax - amin)
            ax1 = (_get_axis_value(a_axis, v1) - amin) / (amax - amin)
            ax2 = (_get_axis_value(a_axis, v2) - amin) / (amax - amin)
        else:
            ax0 = ax1 = ax2 = 0.5

        if abs(bmax - bmin) > 1e-9:
            bx0 = (_get_axis_value(b_axis, v0) - bmin) / (bmax - bmin)
            bx1 = (_get_axis_value(b_axis, v1) - bmin) / (bmax - bmin)
            bx2 = (_get_axis_value(b_axis, v2) - bmin) / (bmax - bmin)
        else:
            bx0 = bx1 = bx2 = 0.5

        def clean(t: float) -> float:
            tt = float(t)
            return tt if math.isfinite(tt) else 0.5

        ax0 = clean(ax0)
        ax1 = clean(ax1)
        ax2 = clean(ax2)
        bx0 = clean(bx0)
        bx1 = clean(bx1)
        bx2 = clean(bx2)

        ax0, bx0 = _tex_from_face_uv(ax0, bx0, rot_i)
        ax1, bx1 = _tex_from_face_uv(ax1, bx1, rot_i)
        ax2, bx2 = _tex_from_face_uv(ax2, bx2, rot_i)

        poly = QtGui.QPolygonF(
            [
                QtCore.QPointF(x0 + (ax0 * tile_px), y0 + (bx0 * tile_px)),
                QtCore.QPointF(x0 + (ax1 * tile_px), y0 + (bx1 * tile_px)),
                QtCore.QPointF(x0 + (ax2 * tile_px), y0 + (bx2 * tile_px)),
            ]
        )
        p.drawPolygon(poly)

    p.end()
    return img


def _element_for_plane(
    *,
    cuboid: Cuboid,
    texture_key: str,
    outward_face: str,
    double_sided: bool,
    force_full_uv: bool,
    uv: Optional[list[float]] = None,
    face_rotation: Optional[int] = None,
    shade: bool,
) -> dict:
    faces: dict = {}
    for f in _faces_for_plane(outward_face=outward_face, double_sided=double_sided):
        d = {"texture": texture_key}
        if uv is not None:
            d["uv"] = list(uv)
        elif force_full_uv:
            d["uv"] = [0.0, 0.0, 16.0, 16.0]
        r = None if face_rotation is None else int(face_rotation)
        if r in (0, 90, 180, 270) and int(r) != 0:
            d["rotation"] = int(r)
        faces[f] = d

    el: dict = {
        "from": [cuboid.fr[0], cuboid.fr[1], cuboid.fr[2]],
        "to": [cuboid.to[0], cuboid.to[1], cuboid.to[2]],
        "faces": faces,
    }

    if not shade:
        el["shade"] = False

    if cuboid.rotation is not None and float(cuboid.rotation.angle) != 0.0:
        rot: dict = {
            "origin": [cuboid.rotation.origin[0], cuboid.rotation.origin[1], cuboid.rotation.origin[2]],
            "axis": cuboid.rotation.axis,
            "angle": cuboid.rotation.angle,
        }
        if bool(cuboid.rotation.rescale):
            rot["rescale"] = True
        el["rotation"] = rot

    return el


def export_minecraft_model(*, elements: list[dict], textures: dict, particle: Optional[str] = None) -> dict:
    if particle is None:
        particle = str(textures.get("0", "minecraft:block/oak_planks"))

    out = {
        "textures": dict(textures),
        "elements": list(elements),
    }
    if "particle" not in out["textures"]:
        out["textures"]["particle"] = particle
    return out


def build_mode_planes(
    *,
    center_xz: tuple[float, float],
    join_y: float,
    half_base: float,
    thickness: float,
    rescale: bool,
) -> list[tuple[Cuboid, str]]:
    def apex_height(half: float, slope_angle_deg: float) -> float:
        a = abs(float(slope_angle_deg))
        t = math.tan(math.radians(a))
        if t <= 1e-9:
            return 0.0
        return float(half) * t

    top1_slope = 67.5
    bot1_slope = 45.0
    top2_slope = 45.0
    bot2_slope = 67.5

    top1_rot, top1_style = _slope_to_build_params(slope_angle_deg=top1_slope)
    bot1_rot, bot1_style = _slope_to_build_params(slope_angle_deg=bot1_slope)
    top2_rot, top2_style = _slope_to_build_params(slope_angle_deg=top2_slope)
    bot2_rot, bot2_style = _slope_to_build_params(slope_angle_deg=bot2_slope)

    # Stack the second bipyramid below the first, tip-to-tip.
    join_y2 = float(join_y) - apex_height(half_base, bot1_slope) - apex_height(half_base, top2_slope)

    a = build_unilateral_octahedron_planes(
        center_xz=center_xz,
        join_y=float(join_y),
        half_base=half_base,
        top_angle_deg=top1_rot,
        bottom_angle_deg=bot1_rot,
        top_style=top1_style,
        bottom_style=bot1_style,
        thickness=thickness,
        rescale=rescale,
    )
    b = build_unilateral_octahedron_planes(
        center_xz=center_xz,
        join_y=float(join_y2),
        half_base=half_base,
        top_angle_deg=top2_rot,
        bottom_angle_deg=bot2_rot,
        top_style=top2_style,
        bottom_style=bot2_style,
        thickness=thickness,
        rescale=rescale,
    )
    return list(a) + list(b)


def build_pyramid_planes(
    *,
    center_xz: tuple[float, float],
    base_y: float,
    half_base: float,
    angle_deg: float,
    thickness: float,
    rescale: bool,
    style: PyramidStyle = "short",
    orientation: PyramidOrientation = "up",
    include_base: bool = True,
) -> list[tuple[Cuboid, str]]:
    cx, cz = center_xz

    angle_deg = _snap_to_allowed(float(angle_deg), tuple(abs(a) for a in _ALLOWED_ROTATION_ANGLES))

    a = abs(float(angle_deg))
    t = float(thickness)
    if t <= 0.0:
        raise ValueError("Thickness must be > 0")

    tan_a = math.tan(math.radians(a))
    if tan_a <= 1e-9:
        raise ValueError("Angle too small")

    height_short = float(half_base) * float(tan_a)
    height_tall = float(half_base) / float(tan_a)

    height = float(height_short) if style != "tall" else float(height_tall)
    apex_y = float(base_y) + float(height)

    cos_a = math.cos(math.radians(a))
    if cos_a <= 1e-9:
        raise ValueError("Angle too steep")

    run = float(half_base) / float(cos_a)

    x0 = cx - float(half_base)
    x1 = cx + float(half_base)
    z0 = cz - float(half_base)
    z1 = cz + float(half_base)

    out: list[tuple[Cuboid, str]] = []

    # Base plane
    base = Cuboid(
        fr=(x0, float(base_y) - t * 0.5, z0),
        to=(x1, float(base_y) + t * 0.5, z1),
        rotation=None,
    )
    out.append((base, "down"))

    if style == "tall":
        z_n = cz - float(half_base)
        z_s = cz + float(half_base)
        x_w = cx - float(half_base)
        x_e = cx + float(half_base)

        h_pre = float(half_base) / max(1e-9, math.sin(math.radians(a)))

        c_s = Cuboid(
            fr=(x0, float(base_y), z_s - t * 0.5),
            to=(x1, float(base_y) + h_pre, z_s + t * 0.5),
            rotation=Rotation(axis="x", angle=-a, origin=(cx, float(base_y), z_s), rescale=rescale),
        )
        out.append((c_s, "south"))

        c_n = Cuboid(
            fr=(x0, float(base_y), z_n - t * 0.5),
            to=(x1, float(base_y) + h_pre, z_n + t * 0.5),
            rotation=Rotation(axis="x", angle=a, origin=(cx, float(base_y), z_n), rescale=rescale),
        )
        out.append((c_n, "north"))

        c_e = Cuboid(
            fr=(x_e - t * 0.5, float(base_y), z0),
            to=(x_e + t * 0.5, float(base_y) + h_pre, z1),
            rotation=Rotation(axis="z", angle=a, origin=(x_e, float(base_y), cz), rescale=rescale),
        )
        out.append((c_e, "east"))

        c_w = Cuboid(
            fr=(x_w - t * 0.5, float(base_y), z0),
            to=(x_w + t * 0.5, float(base_y) + h_pre, z1),
            rotation=Rotation(axis="z", angle=-a, origin=(x_w, float(base_y), cz), rescale=rescale),
        )
        out.append((c_w, "west"))
    else:
        apex = (cx, apex_y, cz)

        c_s = Cuboid(
            fr=(x0, apex_y, cz - t * 0.5),
            to=(x1, apex_y + t, cz + run + t * 0.5),
            rotation=Rotation(axis="x", angle=a, origin=apex, rescale=rescale),
        )
        out.append((c_s, "up"))

        c_n = Cuboid(
            fr=(x0, apex_y, cz - run - t * 0.5),
            to=(x1, apex_y + t, cz + t * 0.5),
            rotation=Rotation(axis="x", angle=-a, origin=apex, rescale=rescale),
        )
        out.append((c_n, "up"))

        c_e = Cuboid(
            fr=(cx - t * 0.5, apex_y, z0),
            to=(cx + run + t * 0.5, apex_y + t, z1),
            rotation=Rotation(axis="z", angle=-a, origin=apex, rescale=rescale),
        )
        out.append((c_e, "up"))

        c_w = Cuboid(
            fr=(cx - run - t * 0.5, apex_y, z0),
            to=(cx + t * 0.5, apex_y + t, z1),
            rotation=Rotation(axis="z", angle=a, origin=apex, rescale=rescale),
        )
        out.append((c_w, "up"))

    if not bool(include_base):
        out = out[1:]

    if orientation == "down":
        def flip_face(f: str) -> str:
            if f == "up":
                return "down"
            if f == "down":
                return "up"
            return f

        def flip_cub(c: Cuboid) -> Cuboid:
            fx, fy, fz = c.fr
            tx, ty, tz = c.to
            fy2 = (2.0 * float(base_y)) - float(fy)
            ty2 = (2.0 * float(base_y)) - float(ty)
            fr2 = (float(fx), min(fy2, ty2), float(fz))
            to2 = (float(tx), max(fy2, ty2), float(tz))

            rot2: Optional[Rotation] = None
            if c.rotation is not None:
                ox, oy, oz = c.rotation.origin
                oy2 = (2.0 * float(base_y)) - float(oy)
                ang = float(c.rotation.angle)
                if c.rotation.axis in ("x", "z"):
                    ang = -ang
                rot2 = Rotation(axis=c.rotation.axis, angle=ang, origin=(float(ox), float(oy2), float(oz)), rescale=bool(c.rotation.rescale))

            return Cuboid(fr=fr2, to=to2, rotation=rot2)

        out = [(flip_cub(c), flip_face(f)) for (c, f) in out]

    return out


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
                    if isinstance(v, dict) and all(isinstance(kk, str) for kk in v.keys()) and all(
                        is_scalar(vv) for vv in v.values()
                    ):
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


def build_unilateral_octahedron_planes(
    *,
    center_xz: tuple[float, float],
    join_y: float,
    half_base: float,
    top_angle_deg: float,
    bottom_angle_deg: float,
    top_style: PyramidStyle = "short",
    bottom_style: PyramidStyle = "short",
    thickness: float,
    rescale: bool,
) -> list[tuple[Cuboid, str]]:
    top = build_pyramid_planes(
        center_xz=center_xz,
        base_y=join_y,
        half_base=half_base,
        angle_deg=top_angle_deg,
        thickness=thickness,
        rescale=rescale,
        style=top_style,
        orientation="up",
        include_base=False,
    )
    bot = build_pyramid_planes(
        center_xz=center_xz,
        base_y=join_y,
        half_base=half_base,
        angle_deg=bottom_angle_deg,
        thickness=thickness,
        rescale=rescale,
        style=bottom_style,
        orientation="down",
        include_base=False,
    )

    if len(top) != 4 or len(bot) != 4:
        raise ValueError("Internal error building bipyramid")

    # build_pyramid_planes ramp order: south, north, east, west
    # desired plane order: top_north, top_south, top_west, top_east, bottom_north, bottom_south, bottom_west, bottom_east
    out = [top[1], top[0], top[3], top[2], bot[1], bot[0], bot[3], bot[2]]
    return out


@dataclass
class UiState:
    model_json: Optional[dict] = None
    atlas_image: Optional[QtGui.QImage] = None


class DiamondPyramidMainWindow(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Minecraft Diamond Pyramid (Plane Solver)")

        self._ui_state = UiState()
        self._current_cuboids: list[Cuboid] = []
        self._current_outward_faces: list[str] = []
        self._current_plane_labels: list[str] = []
        self._vis_mask: list[bool] = []
        self._target_segments: list[tuple[tuple[float, float, float], tuple[float, float, float], QtGui.QColor]] = []
        self._last_target_params: Optional[
            tuple[PresetId, tuple[float, float], float, float, float, float, PyramidStyle, PyramidOrientation]
        ] = None
        self._preset: PresetId = "diamond"

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

    def _try_update_last_target_params_from_ui(self) -> None:
        try:
            center_xz = _parse_vec2(self._center_xz.text().strip())
            join_y = float(self._join_y.value())
            half_base = float(self._half_base.value())
            top_angle = float(self._top_angle.currentData())
            bottom_angle = float(self._bottom_angle.currentData())
            pyr_style: PyramidStyle = "short"
            pyr_orient: PyramidOrientation = "up"
            if self._preset == "pyramid":
                _rot, pyr_style = _slope_to_build_params(slope_angle_deg=top_angle)
                vo = self._pyr_orient.currentData()
                pyr_orient = str(vo) if vo in {"up", "down"} else "up"  # type: ignore[assignment]
        except Exception:
            return

        self._last_target_params = (self._preset, center_xz, join_y, half_base, top_angle, bottom_angle, pyr_style, pyr_orient)
        self._sync_target_wireframe()

    def _build_output(self) -> QtWidgets.QWidget:
        w = QtWidgets.QWidget()
        w.setMinimumWidth(420)
        w.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Preferred,
            QtWidgets.QSizePolicy.Policy.Expanding,
        )

        self._txt_elements = QtWidgets.QPlainTextEdit()
        self._txt_elements.setReadOnly(True)
        self._txt_elements.setMaximumBlockCount(10000)
        self._txt_elements.setLineWrapMode(QtWidgets.QPlainTextEdit.LineWrapMode.NoWrap)
        self._txt_elements.setFont(QtGui.QFont("Consolas", 10))

        self._txt_model_json = QtWidgets.QPlainTextEdit()
        self._txt_model_json.setReadOnly(True)
        self._txt_model_json.setMaximumBlockCount(10000)
        self._txt_model_json.setLineWrapMode(QtWidgets.QPlainTextEdit.LineWrapMode.NoWrap)
        self._txt_model_json.setFont(QtGui.QFont("Consolas", 10))

        self._atlas_preview = QtWidgets.QLabel()
        self._atlas_preview.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop | QtCore.Qt.AlignmentFlag.AlignLeft)
        self._atlas_preview.setMinimumSize(320, 320)
        self._atlas_preview.setText("(No atlas generated)")

        atlas_scroll = QtWidgets.QScrollArea()
        atlas_scroll.setWidgetResizable(True)
        atlas_scroll.setWidget(self._atlas_preview)

        atlas_tab = QtWidgets.QWidget()
        atlas_tab_layout = QtWidgets.QVBoxLayout(atlas_tab)
        atlas_tab_layout.setContentsMargins(10, 10, 10, 10)
        atlas_tab_layout.addWidget(atlas_scroll, 1)
        atlas_tab_layout.addWidget(self._diamond_uv_rot_box, 0)
        atlas_tab_layout.addStretch(0)

        output_tabs = QtWidgets.QTabWidget()
        output_tabs.addTab(self._txt_elements, "Elements")
        output_tabs.addTab(self._txt_model_json, "Model JSON")
        output_tabs.addTab(atlas_tab, "Atlas")

        output_box = QtWidgets.QGroupBox("Output")
        output_box.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Preferred,
            QtWidgets.QSizePolicy.Policy.Expanding,
        )
        output_layout = QtWidgets.QVBoxLayout(output_box)
        output_layout.setContentsMargins(8, 6, 8, 6)
        output_layout.setSpacing(6)
        output_layout.addWidget(output_tabs)

        layout = QtWidgets.QVBoxLayout(w)
        layout.addWidget(output_box)
        layout.setContentsMargins(0, 0, 0, 0)

        return w

    def _set_all_planes_visible(self, visible: bool) -> None:
        if self._lst_planes.count() <= 0:
            return
        self._lst_planes.blockSignals(True)
        try:
            for i in range(self._lst_planes.count()):
                it = self._lst_planes.item(i)
                if it is None:
                    continue
                it.setCheckState(QtCore.Qt.CheckState.Checked if visible else QtCore.Qt.CheckState.Unchecked)
        finally:
            self._lst_planes.blockSignals(False)
        self._sync_plane_visibility_from_ui()

    def _on_plane_item_changed(self, item: QtWidgets.QListWidgetItem) -> None:
        self._sync_plane_visibility_from_ui()

    def _sync_plane_visibility_from_ui(self) -> None:
        mask: list[bool] = []
        for i in range(self._lst_planes.count()):
            it = self._lst_planes.item(i)
            mask.append(bool(it is not None and it.checkState() == QtCore.Qt.CheckState.Checked))
        self._vis_mask = mask
        self._viewport.set_visibility_mask(mask)
        self._sync_target_wireframe()

    def _update_plane_list(self) -> None:
        if self._preset == "pyramid":
            names = ["base", "ramp_south", "ramp_north", "ramp_east", "ramp_west"]
        elif self._preset == "mode":
            names = [
                "A_top_north",
                "A_top_south",
                "A_top_west",
                "A_top_east",
                "A_bottom_north",
                "A_bottom_south",
                "A_bottom_west",
                "A_bottom_east",
                "B_top_north",
                "B_top_south",
                "B_top_west",
                "B_top_east",
                "B_bottom_north",
                "B_bottom_south",
                "B_bottom_west",
                "B_bottom_east",
            ]
        else:
            names = [
                "top_north",
                "top_south",
                "top_west",
                "top_east",
                "bottom_north",
                "bottom_south",
                "bottom_west",
                "bottom_east",
            ]

        self._lst_planes.blockSignals(True)
        try:
            self._lst_planes.clear()
            for i, _c in enumerate(self._current_cuboids):
                label = names[i] if i < len(names) else f"{i:02d}"
                it = QtWidgets.QListWidgetItem(label)
                it.setFlags(it.flags() | QtCore.Qt.ItemFlag.ItemIsUserCheckable)
                it.setCheckState(QtCore.Qt.CheckState.Checked)
                self._lst_planes.addItem(it)
        finally:
            self._lst_planes.blockSignals(False)
        self._vis_mask = [True] * len(self._current_cuboids)
        self._viewport.set_visibility_mask(self._vis_mask)

    def _visible_cuboids(self) -> list[Cuboid]:
        if not self._current_cuboids:
            return []
        if not self._vis_mask or len(self._vis_mask) != len(self._current_cuboids):
            return list(self._current_cuboids)
        return [c for c, v in zip(self._current_cuboids, self._vis_mask) if bool(v)]

    def _update_target_segments(self) -> None:
        params = self._last_target_params
        if params is None:
            self._target_segments = []
            return

        preset, center_xz, join_y, half_base, top_angle, bottom_angle, pyr_style, pyr_orient = params

        def apex_height(half: float, slope_angle_deg: float) -> float:
            a = abs(float(slope_angle_deg))
            t = math.tan(math.radians(a))
            if t <= 1e-9:
                return 0.0
            return float(half) * t

        cx, cz = center_xz

        if preset == "pyramid":
            base_y = float(join_y)
            h = apex_height(half_base, top_angle)
            apex = (cx, (base_y + h) if pyr_orient == "up" else (base_y - h), cz)

            nw = (cx - half_base, base_y, cz - half_base)
            ne = (cx + half_base, base_y, cz - half_base)
            se = (cx + half_base, base_y, cz + half_base)
            sw = (cx - half_base, base_y, cz + half_base)

            faces = [
                (apex, nw, ne),
                (apex, ne, se),
                (apex, se, sw),
                (apex, sw, nw),
                (nw, sw, se),
                (nw, se, ne),
            ]
        elif preset == "mode":
            def face_set_for(join_y0: float, top_slope0: float, bot_slope0: float) -> list[tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]]:
                top_h0 = apex_height(half_base, top_slope0)
                bot_h0 = apex_height(half_base, bot_slope0)
                top_apex0 = (cx, join_y0 + top_h0, cz)
                bot_apex0 = (cx, join_y0 - bot_h0, cz)
                nw0 = (cx - half_base, join_y0, cz - half_base)
                ne0 = (cx + half_base, join_y0, cz - half_base)
                se0 = (cx + half_base, join_y0, cz + half_base)
                sw0 = (cx - half_base, join_y0, cz + half_base)
                return [
                    (top_apex0, nw0, ne0),
                    (top_apex0, ne0, se0),
                    (top_apex0, se0, sw0),
                    (top_apex0, sw0, nw0),
                    (bot_apex0, ne0, nw0),
                    (bot_apex0, se0, ne0),
                    (bot_apex0, sw0, se0),
                    (bot_apex0, nw0, sw0),
                ]

            join_a = float(join_y)
            join_b = float(join_y) - apex_height(half_base, 45.0) - apex_height(half_base, 45.0)
            faces = []
            faces += face_set_for(join_a, 67.5, 45.0)
            faces += face_set_for(join_b, 45.0, 67.5)
        else:
            top_h = apex_height(half_base, top_angle)
            bot_h = apex_height(half_base, bottom_angle)

            top_apex = (cx, join_y + top_h, cz)
            bot_apex = (cx, join_y - bot_h, cz)

            nw = (cx - half_base, join_y, cz - half_base)
            ne = (cx + half_base, join_y, cz - half_base)
            se = (cx + half_base, join_y, cz + half_base)
            sw = (cx - half_base, join_y, cz + half_base)

            faces = [
                (top_apex, nw, ne),
                (top_apex, ne, se),
                (top_apex, se, sw),
                (top_apex, sw, nw),
                (bot_apex, ne, nw),
                (bot_apex, se, ne),
                (bot_apex, sw, se),
                (bot_apex, nw, sw),
            ]

        def lerp(a: float, b: float, t: float) -> float:
            return a + (b - a) * t

        def col_for_ratio(r: float) -> QtGui.QColor:
            rr = max(0.0, min(1.0, float(r)))
            r0, g0, b0 = (235, 80, 80)
            r1, g1, b1 = (85, 210, 120)
            return QtGui.QColor(int(lerp(r0, r1, rr)), int(lerp(g0, g1, rr)), int(lerp(b0, b1, rr)), 220)

        def triangle_samples(v0, v1, v2, n: int = 7) -> list[tuple[float, float, float]]:
            out: list[tuple[float, float, float]] = []
            n = max(2, int(n))
            for i in range(n):
                for j in range(n - i):
                    a = (i + 0.5) / n
                    b = (j + 0.5) / n
                    c = 1.0 - a - b
                    if c <= 0.0:
                        continue
                    x = (v0[0] * a) + (v1[0] * b) + (v2[0] * c)
                    y = (v0[1] * a) + (v1[1] * b) + (v2[1] * c)
                    z = (v0[2] * a) + (v1[2] * b) + (v2[2] * c)
                    out.append((x, y, z))
            return out

        visible = self._visible_cuboids()
        use_cov = bool(self._vp_target_coverage.isChecked())

        segs: list[tuple[tuple[float, float, float], tuple[float, float, float], QtGui.QColor]] = []
        for v0, v1, v2 in faces:
            col = QtGui.QColor(200, 200, 210, 210)
            if use_cov and visible:
                pts = triangle_samples(v0, v1, v2)
                if pts:
                    hit = 0
                    for p in pts:
                        if any(c.contains_point(p) for c in visible):
                            hit += 1
                    col = col_for_ratio(hit / max(1, len(pts)))
            segs.append((v0, v1, col))
            segs.append((v1, v2, col))
            segs.append((v2, v0, col))

        self._target_segments = segs

    def _sync_viewport_render_options(self) -> None:
        faces = bool(self._vp_faces.isChecked())
        self._vp_translucent.setEnabled(faces)
        self._viewport.set_render_options(
            faces=faces,
            faces_translucent=bool(self._vp_translucent.isChecked()),
            wireframe=bool(self._vp_wireframe.isChecked()),
        )

        self._viewport.set_debug_face_colors(bool(self._vp_face_debug_colors.isChecked()))
        self._sync_target_wireframe()

    def _sync_target_wireframe(self) -> None:
        enabled = bool(self._vp_target_wire.isChecked())
        if not enabled:
            self._viewport.set_target_wireframe(enabled=False, segments=[])
            return
        self._update_target_segments()
        self._viewport.set_target_wireframe(enabled=True, segments=self._target_segments)

    def _build_controls(self) -> QtWidgets.QWidget:
        w = QtWidgets.QWidget()
        w.setMinimumWidth(360)

        self._preset_combo = QtWidgets.QComboBox()
        self._preset_combo.addItem("Unilateral bipyramid (diamond)", userData="diamond")
        self._preset_combo.addItem("Mode (2 stacked bipyramids)", userData="mode")
        self._preset_combo.addItem("4-sided pyramid", userData="pyramid")
        self._preset_combo.currentIndexChanged.connect(self._on_preset_changed)

        self._center_xz = QtWidgets.QLineEdit("8,8")
        self._center_xz.editingFinished.connect(self._try_update_last_target_params_from_ui)

        self._join_y = QtWidgets.QDoubleSpinBox()
        self._join_y.setRange(-64.0, 64.0)
        self._join_y.setValue(8.0)
        self._join_y.setDecimals(3)
        self._join_y.valueChanged.connect(lambda _v: self._try_update_last_target_params_from_ui())

        self._half_base = QtWidgets.QDoubleSpinBox()
        self._half_base.setRange(0.05, 128.0)
        self._half_base.setValue(6.0)
        self._half_base.setDecimals(3)
        self._half_base.valueChanged.connect(lambda _v: self._try_update_last_target_params_from_ui())

        self._top_angle = QtWidgets.QComboBox()
        for a in _SLOPE_ANGLES:
            self._top_angle.addItem(f"{a:g}", userData=float(a))
        self._top_angle.setCurrentText("45")
        self._top_angle.currentIndexChanged.connect(lambda _i: self._try_update_last_target_params_from_ui())

        self._pyr_orient = QtWidgets.QComboBox()
        self._pyr_orient.addItem("Up", userData="up")
        self._pyr_orient.addItem("Down", userData="down")
        self._pyr_orient.currentIndexChanged.connect(lambda _i: self._try_update_last_target_params_from_ui())

        self._bottom_angle = QtWidgets.QComboBox()
        for a in _SLOPE_ANGLES:
            self._bottom_angle.addItem(f"{a:g}", userData=float(a))
        self._bottom_angle.setCurrentText("22.5")
        self._bottom_angle.currentIndexChanged.connect(lambda _i: self._try_update_last_target_params_from_ui())

        self._thickness = QtWidgets.QDoubleSpinBox()
        self._thickness.setRange(0.001, 2.0)
        self._thickness.setValue(0.02)
        self._thickness.setDecimals(4)

        self._rescale = QtWidgets.QCheckBox("Rotation rescale")
        self._rescale.setChecked(True)

        self._double_sided = QtWidgets.QCheckBox("Double-sided planes")
        self._double_sided.setChecked(False)

        self._force_uv = QtWidgets.QCheckBox("Force full UV (0-16)")
        self._force_uv.setChecked(True)

        self._shade = QtWidgets.QCheckBox("Shade")
        self._shade.setChecked(True)

        self._texture_top = QtWidgets.QLineEdit("minecraft:block/oak_planks")
        self._texture_bottom = QtWidgets.QLineEdit("minecraft:block/oak_planks")

        self._uv_atlas = QtWidgets.QCheckBox("Atlas UVs")
        self._uv_atlas.setChecked(False)

        self._atlas_tile_px = QtWidgets.QSpinBox()
        self._atlas_tile_px.setRange(8, 2048)
        self._atlas_tile_px.setValue(64)

        self._atlas_total_px = QtWidgets.QSpinBox()
        self._atlas_total_px.setRange(0, 8192)
        self._atlas_total_px.setValue(256)

        self._atlas_gap_px = QtWidgets.QSpinBox()
        self._atlas_gap_px.setRange(0, 128)
        self._atlas_gap_px.setValue(2)

        self._atlas_border_px = QtWidgets.QSpinBox()
        self._atlas_border_px.setRange(0, 256)
        self._atlas_border_px.setValue(2)

        self._atlas_make_png = QtWidgets.QCheckBox("Generate cutout atlas")
        self._atlas_make_png.setChecked(False)

        self._atlas_debug = QtWidgets.QCheckBox("Atlas debug coloring")
        self._atlas_debug.setChecked(False)

        self._atlas_pad_alpha = QtWidgets.QCheckBox("Pad atlas alpha 1px")
        self._atlas_pad_alpha.setChecked(True)

        self._uv_side_rotate = QtWidgets.QCheckBox("Rotate side UVs 'up' (hardcoded)")
        self._uv_side_rotate.setChecked(False)
        self._uv_side_rotate.stateChanged.connect(lambda _v: self._sync_uv_rotation_controls_enabled())
        self._uv_side_rotate.stateChanged.connect(lambda _v: self._sync_viewport_uv_debug())

        self._diamond_uv_rots: list[QtWidgets.QComboBox] = []
        self._diamond_uv_rot_box = QtWidgets.QGroupBox("Diamond UV rotations")
        diamond_rot_form = QtWidgets.QFormLayout(self._diamond_uv_rot_box)
        diamond_rot_form.setContentsMargins(8, 6, 8, 6)
        diamond_rot_form.setVerticalSpacing(4)

        diamond_names = [
            "top_north",
            "top_south",
            "top_west",
            "top_east",
            "bottom_north",
            "bottom_south",
            "bottom_west",
            "bottom_east",
        ]

        for i, name in enumerate(diamond_names):
            cb = QtWidgets.QComboBox()
            for r in (0, 90, 180, 270):
                cb.addItem(str(r), userData=int(r))
            dflt = _FAVORED_SIDE_UV_ROTATIONS.get(("diamond", int(i)), 0)
            idx = cb.findData(int(dflt))
            cb.setCurrentIndex(idx if idx >= 0 else 0)
            cb.currentIndexChanged.connect(lambda _i: self._sync_viewport_uv_debug())
            self._diamond_uv_rots.append(cb)
            diamond_rot_form.addRow(name, cb)

        self._btn_save_atlas = QtWidgets.QPushButton("Save Atlas PNG")
        self._btn_save_atlas.setEnabled(False)
        self._btn_save_atlas.clicked.connect(self._on_save_atlas_png)

        self._vp_faces = QtWidgets.QCheckBox("Faces")
        self._vp_faces.setChecked(True)
        self._vp_faces.stateChanged.connect(self._sync_viewport_render_options)

        self._vp_translucent = QtWidgets.QCheckBox("Translucent faces")
        self._vp_translucent.setChecked(False)
        self._vp_translucent.setEnabled(True)
        self._vp_translucent.stateChanged.connect(self._sync_viewport_render_options)

        self._vp_wireframe = QtWidgets.QCheckBox("Wireframe")
        self._vp_wireframe.setChecked(True)
        self._vp_wireframe.stateChanged.connect(self._sync_viewport_render_options)

        self._vp_face_debug_colors = QtWidgets.QCheckBox("Face debug colors")
        self._vp_face_debug_colors.setChecked(False)
        self._vp_face_debug_colors.stateChanged.connect(self._sync_viewport_render_options)

        self._vp_target_wire = QtWidgets.QCheckBox("Target wireframe")
        self._vp_target_wire.setChecked(True)
        self._vp_target_wire.stateChanged.connect(self._sync_viewport_render_options)

        self._vp_target_coverage = QtWidgets.QCheckBox("Target coverage colors")
        self._vp_target_coverage.setChecked(True)
        self._vp_target_coverage.stateChanged.connect(self._sync_viewport_render_options)

        self._lst_planes = QtWidgets.QListWidget()
        self._lst_planes.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection)
        self._lst_planes.itemChanged.connect(self._on_plane_item_changed)

        self._btn_show_all = QtWidgets.QPushButton("Show all")
        self._btn_show_all.clicked.connect(lambda: self._set_all_planes_visible(True))

        self._btn_hide_all = QtWidgets.QPushButton("Hide all")
        self._btn_hide_all.clicked.connect(lambda: self._set_all_planes_visible(False))

        self._btn_generate = QtWidgets.QPushButton("Generate")
        self._btn_generate.setObjectName("btn_generate")
        self._btn_generate.clicked.connect(self._on_generate)

        self._btn_clear = QtWidgets.QPushButton("Clear")
        self._btn_clear.setObjectName("btn_clear")
        self._btn_clear.clicked.connect(self._on_clear)

        self._btn_save = QtWidgets.QPushButton("Save JSON")
        self._btn_save.setObjectName("btn_save")
        self._btn_save.clicked.connect(self._on_save_json)

        self._lbl_status = QtWidgets.QLabel("Ready")
        self._lbl_status.setWordWrap(True)

        target_box = QtWidgets.QGroupBox("Diamond pyramid")
        target_form = QtWidgets.QFormLayout(target_box)
        target_form.setContentsMargins(8, 6, 8, 6)
        target_form.setVerticalSpacing(4)
        target_form.addRow("Preset", self._preset_combo)
        target_form.addRow("Center XZ", self._center_xz)
        self._lbl_join_y = QtWidgets.QLabel("Join Y")
        target_form.addRow(self._lbl_join_y, self._join_y)
        target_form.addRow("Half base", self._half_base)
        target_form.addRow("Top angle", self._top_angle)
        self._lbl_pyr_orient = QtWidgets.QLabel("Pyramid orientation")
        target_form.addRow(self._lbl_pyr_orient, self._pyr_orient)
        self._lbl_bottom_angle = QtWidgets.QLabel("Bottom angle")
        target_form.addRow(self._lbl_bottom_angle, self._bottom_angle)
        target_form.addRow("Thickness", self._thickness)
        target_form.addRow("Top texture", self._texture_top)
        target_form.addRow("Bottom texture", self._texture_bottom)

        opts_box = QtWidgets.QGroupBox("Options")
        opts_layout = QtWidgets.QVBoxLayout(opts_box)
        opts_layout.setContentsMargins(8, 6, 8, 6)
        opts_layout.setSpacing(4)
        opts_layout.addWidget(self._rescale)
        opts_layout.addWidget(self._double_sided)
        opts_layout.addWidget(self._force_uv)
        opts_layout.addWidget(self._shade)

        uv_box = QtWidgets.QGroupBox("UV / Texture")
        uv_form = QtWidgets.QFormLayout(uv_box)
        uv_form.setContentsMargins(8, 6, 8, 6)
        uv_form.setVerticalSpacing(4)
        uv_form.addRow(self._uv_atlas)
        uv_form.addRow(self._uv_side_rotate)
        uv_form.addRow("Atlas total px (0=tile px)", self._atlas_total_px)
        uv_form.addRow("UV island gap px", self._atlas_gap_px)
        uv_form.addRow("UV border pad px", self._atlas_border_px)
        uv_form.addRow("Atlas tile px", self._atlas_tile_px)
        uv_form.addRow(self._atlas_make_png)
        uv_form.addRow(self._atlas_debug)
        uv_form.addRow(self._atlas_pad_alpha)
        uv_form.addRow(self._btn_save_atlas)

        viewport_box = QtWidgets.QGroupBox("Viewport")
        viewport_layout = QtWidgets.QVBoxLayout(viewport_box)
        viewport_layout.setContentsMargins(8, 6, 8, 6)
        viewport_layout.setSpacing(4)
        viewport_layout.addWidget(self._vp_faces)
        viewport_layout.addWidget(self._vp_translucent)
        viewport_layout.addWidget(self._vp_wireframe)
        viewport_layout.addWidget(self._vp_face_debug_colors)
        viewport_layout.addWidget(self._vp_target_wire)
        viewport_layout.addWidget(self._vp_target_coverage)

        planes_box = QtWidgets.QGroupBox("Planes")
        planes_layout = QtWidgets.QVBoxLayout(planes_box)
        planes_layout.setContentsMargins(8, 6, 8, 6)
        planes_layout.setSpacing(6)
        planes_layout.addWidget(self._lst_planes)

        planes_btn_row = QtWidgets.QHBoxLayout()
        planes_btn_row.addWidget(self._btn_show_all)
        planes_btn_row.addWidget(self._btn_hide_all)
        planes_layout.addLayout(planes_btn_row)

        btn_row = QtWidgets.QHBoxLayout()
        btn_row.addWidget(self._btn_generate)
        btn_row.addWidget(self._btn_clear)

        btn_row2 = QtWidgets.QHBoxLayout()
        btn_row2.addWidget(self._btn_save)

        layout = QtWidgets.QVBoxLayout(w)
        layout.addWidget(target_box)
        layout.addWidget(opts_box)
        layout.addWidget(uv_box)
        layout.addWidget(planes_box)
        layout.addWidget(viewport_box)
        layout.addLayout(btn_row)
        layout.addLayout(btn_row2)
        layout.addWidget(self._lbl_status)
        layout.addStretch(1)

        self._sync_viewport_render_options()
        self._try_update_last_target_params_from_ui()

        self._sync_uv_rotation_controls_enabled()

        self._sync_viewport_uv_debug()

        return w

    def _sync_uv_rotation_controls_enabled(self) -> None:
        is_diamond = self._preset == "diamond"
        use_uv_rot = bool(self._uv_side_rotate.isChecked())
        self._diamond_uv_rot_box.setEnabled(is_diamond and use_uv_rot)

    def _sync_viewport_uv_debug(self) -> None:
        is_diamond = self._preset == "diamond"
        use_uv_rot = bool(self._uv_side_rotate.isChecked())
        if not is_diamond or not use_uv_rot or not self._current_cuboids:
            self._viewport.set_uv_debug(enabled=False, info=[])
            return

        info: list[tuple[str, Optional[int], str]] = []
        for i in range(len(self._current_cuboids)):
            outward = self._current_outward_faces[i] if i < len(self._current_outward_faces) else "up"
            label = self._current_plane_labels[i] if i < len(self._current_plane_labels) else f"{i}"
            rot: Optional[int] = None
            if 0 <= int(i) < len(self._diamond_uv_rots):
                v = self._diamond_uv_rots[int(i)].currentData()
                rot = None if v is None else int(v)
            info.append((str(outward), rot, str(label)))

        self._viewport.set_uv_debug(enabled=True, info=info)

    def _on_preset_changed(self) -> None:
        v = self._preset_combo.currentData()
        self._preset = str(v) if v in {"diamond", "mode", "pyramid"} else "diamond"  # type: ignore[assignment]

        is_pyr = self._preset == "pyramid"
        is_mode = self._preset == "mode"
        self._bottom_angle.setEnabled((not is_pyr) and (not is_mode))
        self._lbl_bottom_angle.setEnabled((not is_pyr) and (not is_mode))
        self._top_angle.setEnabled(not is_mode)
        self._texture_bottom.setEnabled(True)

        self._pyr_orient.setEnabled(is_pyr)
        self._lbl_pyr_orient.setEnabled(is_pyr)

        self._lbl_join_y.setText("Base Y" if is_pyr else ("Top Join Y" if is_mode else "Join Y"))

        self._sync_uv_rotation_controls_enabled()

        self._sync_viewport_uv_debug()

        self._try_update_last_target_params_from_ui()

    def _on_clear(self) -> None:
        self._current_cuboids = []
        self._current_outward_faces = []
        self._current_plane_labels = []
        self._vis_mask = []
        self._target_segments = []
        self._last_target_params = None
        self._ui_state.model_json = None
        self._ui_state.atlas_image = None
        self._btn_save_atlas.setEnabled(False)
        self._viewport.clear_cuboids()
        self._viewport.set_uv_debug(enabled=False, info=[])
        self._viewport.set_target_wireframe(enabled=False, segments=[])
        self._lst_planes.clear()
        self._txt_elements.setPlainText("")
        self._txt_model_json.setPlainText("")
        self._lbl_status.setText("Cleared")

    def _on_generate(self) -> None:
        try:
            center_xz = _parse_vec2(self._center_xz.text().strip())
            join_y = float(self._join_y.value())
            half_base = float(self._half_base.value())

            top_slope = float(self._top_angle.currentData())
            bottom_slope = float(self._bottom_angle.currentData())
            thickness = float(self._thickness.value())

            pyr_style: PyramidStyle = "short"
            pyr_orient: PyramidOrientation = "up"

            if self._preset == "pyramid":
                rot_angle, pyr_style = _slope_to_build_params(slope_angle_deg=top_slope)
                vo = self._pyr_orient.currentData()
                pyr_orient = str(vo) if vo in {"up", "down"} else "up"  # type: ignore[assignment]
                planes = build_pyramid_planes(
                    center_xz=center_xz,
                    base_y=join_y,
                    half_base=half_base,
                    angle_deg=rot_angle,
                    thickness=thickness,
                    rescale=bool(self._rescale.isChecked()),
                    style=pyr_style,
                    orientation=pyr_orient,
                    include_base=True,
                )
            elif self._preset == "mode":
                planes = build_mode_planes(
                    center_xz=center_xz,
                    join_y=join_y,
                    half_base=half_base,
                    thickness=thickness,
                    rescale=bool(self._rescale.isChecked()),
                )
            else:
                top_rot, _top_style = _slope_to_build_params(slope_angle_deg=top_slope)
                bot_rot, _bot_style = _slope_to_build_params(slope_angle_deg=bottom_slope)
                planes = build_unilateral_octahedron_planes(
                    center_xz=center_xz,
                    join_y=join_y,
                    half_base=half_base,
                    top_angle_deg=top_rot,
                    bottom_angle_deg=bot_rot,
                    top_style=_top_style,
                    bottom_style=_bot_style,
                    thickness=thickness,
                    rescale=bool(self._rescale.isChecked()),
                )

            shade = bool(self._shade.isChecked())
            double_sided = bool(self._double_sided.isChecked())
            force_uv = bool(self._force_uv.isChecked())
            atlas_uv = bool(self._uv_atlas.isChecked())

            atlas_gap_px = int(self._atlas_gap_px.value())
            atlas_total_px = int(self._atlas_total_px.value())
            atlas_tile_px = int(self._atlas_tile_px.value())
            atlas_border_px = int(self._atlas_border_px.value())
            g, atlas_tile_px_eff, atlas_px_eff = _atlas_layout_px(
                n=len(planes),
                tile_px=atlas_tile_px,
                gap_px=atlas_gap_px,
                total_px=atlas_total_px,
                border_px=atlas_border_px,
            )

            elements_json: list[dict] = []
            cuboids: list[Cuboid] = []
            outward_faces: list[str] = []
            plane_labels: list[str] = []
            plane_face_rots: list[Optional[int]] = []

            diamond_labels = [
                "NT",
                "ST",
                "WT",
                "ET",
                "NB",
                "SB",
                "WB",
                "EB",
            ]
            pyr_labels = ["B", "S", "N", "E", "W"]

            for i, (cub, outward) in enumerate(planes):
                cuboids.append(cub)
                outward_faces.append(str(outward))
                if self._preset == "diamond":
                    plane_labels.append(diamond_labels[i] if i < len(diamond_labels) else f"{i}")
                elif self._preset == "mode":
                    j = int(i) % 8
                    grp = "A" if int(i) < 8 else "B"
                    lab = diamond_labels[j] if j < len(diamond_labels) else f"{i}"
                    plane_labels.append(f"{lab}{grp}")
                else:
                    plane_labels.append(pyr_labels[i] if i < len(pyr_labels) else f"{i}")

                # Textures: for pyramid use bottom texture for base (index 0), top texture for ramps.
                if self._preset == "pyramid":
                    texture_key = "#1" if i == 0 else "#0"
                elif self._preset == "mode":
                    texture_key = "#0" if (int(i) % 8) < 4 else "#1"
                else:
                    texture_key = "#0" if i < 4 else "#1"

                uv = (
                    _atlas_uv_rect(
                        index=i,
                        n=len(planes),
                        tile_px=atlas_tile_px,
                        gap_px=atlas_gap_px,
                        total_px=atlas_total_px,
                        border_px=atlas_border_px,
                    )
                    if atlas_uv
                    else None
                )
                face_rot: Optional[int] = None
                if bool(self._uv_side_rotate.isChecked()):
                    if self._preset == "pyramid" and pyr_style == "tall":
                        face_rot = _TALL_PYRAMID_FACE_UV_ROTATION.get(str(outward))
                    elif self._preset == "diamond":
                        if 0 <= int(i) < len(self._diamond_uv_rots):
                            v = self._diamond_uv_rots[int(i)].currentData()
                            face_rot = None if v is None else int(v)
                    elif self._preset == "mode":
                        face_rot = _FAVORED_SIDE_UV_ROTATIONS.get(("diamond", int(i) % 8))
                    else:
                        face_rot = _FAVORED_SIDE_UV_ROTATIONS.get((self._preset, int(i)))

                plane_face_rots.append(face_rot)
                elements_json.append(
                    _element_for_plane(
                        cuboid=cub,
                        texture_key=texture_key,
                        outward_face=outward,
                        double_sided=double_sided,
                        force_full_uv=force_uv and (not atlas_uv),
                        uv=uv,
                        face_rotation=face_rot,
                        shade=shade,
                    )
                )

            model = export_minecraft_model(
                elements=elements_json,
                textures={
                    "0": self._texture_top.text().strip(),
                    "1": self._texture_bottom.text().strip(),
                },
                particle=self._texture_top.text().strip(),
            )

            self._current_cuboids = cuboids
            self._current_outward_faces = outward_faces
            self._current_plane_labels = plane_labels
            self._ui_state.model_json = model

            self._ui_state.atlas_image = None
            self._btn_save_atlas.setEnabled(False)
            if atlas_uv and bool(self._atlas_make_png.isChecked()):
                img = _make_cutout_atlas_image(
                    preset=self._preset,
                    planes=list(planes),
                    face_rots=plane_face_rots,
                    center_xz=center_xz,
                    join_y=join_y,
                    half_base=half_base,
                    top_angle=top_slope,
                    bottom_angle=bottom_slope,
                    tile_px=atlas_tile_px_eff,
                    gap_px=atlas_gap_px,
                    border_px=atlas_border_px,
                    debug=bool(self._atlas_debug.isChecked()),
                    pad_alpha_1px=bool(self._atlas_pad_alpha.isChecked()),
                    pyramid_style=pyr_style if self._preset == "pyramid" else "short",
                    pyramid_orientation=pyr_orient if self._preset == "pyramid" else "up",
                )
                self._ui_state.atlas_image = img
                self._btn_save_atlas.setEnabled(True)

                pm = QtGui.QPixmap.fromImage(img)
                max_w = 380
                if pm.width() > max_w:
                    pm = pm.scaledToWidth(max_w, QtCore.Qt.TransformationMode.SmoothTransformation)
                self._atlas_preview.setPixmap(pm)
            else:
                self._atlas_preview.setPixmap(QtGui.QPixmap())
                self._atlas_preview.setText("(No atlas generated)")

            self._viewport.set_cuboids(cuboids)
            self._sync_viewport_uv_debug()
            self._update_plane_list()

            lines: list[str] = []
            for idx, c in enumerate(cuboids):
                rot = "-"
                if c.rotation is not None and float(c.rotation.angle) != 0.0:
                    rot = f"{c.rotation.axis} {float(c.rotation.angle):g} @ ({c.rotation.origin[0]:g},{c.rotation.origin[1]:g},{c.rotation.origin[2]:g})"
                lines.append(
                    f"{idx:03d} from=({c.fr[0]:g},{c.fr[1]:g},{c.fr[2]:g}) to=({c.to[0]:g},{c.to[1]:g},{c.to[2]:g}) rot={rot}"
                )

            self._txt_elements.setPlainText("\n".join(lines))
            self._txt_model_json.setPlainText(format_minecraft_model_json(model))

            def apex_height(half: float, angle_deg: float) -> float:
                a = abs(float(angle_deg))
                t = math.tan(math.radians(a))
                if t <= 1e-9:
                    return 0.0
                return float(half) * t

            top_h = apex_height(half_base, top_slope)
            bot_h = apex_height(half_base, bottom_slope)
            if self._preset == "pyramid":
                self._last_target_params = (
                    self._preset,
                    center_xz,
                    join_y,
                    half_base,
                    top_slope,
                    bottom_slope,
                    pyr_style,
                    pyr_orient,
                )
                self._sync_target_wireframe()
                extra = ""
                if atlas_uv:
                    extra = f" | atlas={g}x{g} {atlas_px_eff}x{atlas_px_eff} gap={atlas_gap_px}px border={atlas_border_px}px" + (
                        " +png" if self._ui_state.atlas_image is not None else ""
                    )
                self._lbl_status.setText(f"Generated {len(cuboids)} planes | apex_height={top_h:g}{extra}")
            elif self._preset == "mode":
                h_a_top = apex_height(half_base, 67.5)
                h_a_bot = apex_height(half_base, 45.0)
                h_b_top = apex_height(half_base, 45.0)
                h_b_bot = apex_height(half_base, 67.5)
                self._last_target_params = (self._preset, center_xz, join_y, half_base, top_slope, bottom_slope, "short", "up")
                self._sync_target_wireframe()
                extra = ""
                if atlas_uv:
                    extra = f" | atlas={g}x{g} {atlas_px_eff}x{atlas_px_eff} gap={atlas_gap_px}px border={atlas_border_px}px" + (
                        " +png" if self._ui_state.atlas_image is not None else ""
                    )
                self._lbl_status.setText(
                    f"Generated {len(cuboids)} planes | A(top={h_a_top:g} bottom={h_a_bot:g}) B(top={h_b_top:g} bottom={h_b_bot:g}){extra}"
                )
            else:
                self._last_target_params = (self._preset, center_xz, join_y, half_base, top_slope, bottom_slope, "short", "up")
                self._sync_target_wireframe()
                extra = ""
                if atlas_uv:
                    extra = f" | atlas={g}x{g} {atlas_px_eff}x{atlas_px_eff} gap={atlas_gap_px}px border={atlas_border_px}px" + (
                        " +png" if self._ui_state.atlas_image is not None else ""
                    )
                self._lbl_status.setText(
                    f"Generated {len(cuboids)} planes | top_apex_height={top_h:g} bottom_apex_height={bot_h:g}{extra}"
                )
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Generate failed", str(e))

    def _on_save_atlas_png(self) -> None:
        img = self._ui_state.atlas_image
        if img is None:
            QtWidgets.QMessageBox.information(self, "No atlas", "Enable 'Atlas UVs' + 'Generate cutout atlas' then Generate.")
            return

        out_path, _filter = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save Atlas PNG",
            str(Path.cwd() / "plane_atlas.png"),
            "PNG (*.png)",
        )
        if not out_path:
            return

        ok = img.save(out_path, "PNG")
        if not ok:
            QtWidgets.QMessageBox.critical(self, "Save failed", f"Failed to save PNG: {out_path}")
            return
        self._lbl_status.setText(f"Saved atlas: {out_path}")

    def _on_save_json(self) -> None:
        model = self._ui_state.model_json
        if model is None:
            QtWidgets.QMessageBox.information(self, "Nothing to save", "Run Generate first.")
            return

        out_path, _filter = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save Minecraft model JSON",
            str(Path.cwd() / "diamond_pyramid.json"),
            "JSON (*.json)",
        )
        if not out_path:
            return

        Path(out_path).write_text(format_minecraft_model_json(model) + "\n", encoding="utf-8", newline="\n")
        self._lbl_status.setText(f"Saved: {out_path}")


def main() -> None:
    app = QtWidgets.QApplication(sys.argv)
    _apply_dark_theme(app)
    w = DiamondPyramidMainWindow()
    w.resize(1200, 780)
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
