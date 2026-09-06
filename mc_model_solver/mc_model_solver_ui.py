"""Minecraft Model JSON Solver (UI)

Approximates 3D shapes with Minecraft block-model JSON `elements`
(axis-aligned cuboids with optional single-axis rotation to maintain
minecraft 1.20 and below compatibility).

This is an iterative *design aid*: you choose a target primitive (sphere/cone/
pyramid/double-helix), tune solver settings, and the tool tries to cover the
shape with a limited number of Minecraft-valid elements.

TOOLSGROUP::MODEL
SORTGROUP::7
SORTPRIORITY::74
STATUS::active
VERSION::20260308

Expected inputs
  - Target parameters (e.g. radius/height)
  - Solver constraints (grid, max elements, rotation settings)
  - Texture id used for exported JSON (e.g. `arborea:blocks/glimmerfluidstatic`)

Outputs
- Displays elements + formatted model JSON in the right panel
- "Save JSON" writes the current model JSON to disk

- Project example  :
  - In the UI, set Texture to: `arborea:blocks/glimmerfluidstatic`
  - Then Save JSON and move it under:
    `arborea_1_19_2/src/main/resources/assets/arborea/models/`
"""

#region SETUP
# Module imports, PySide6 bootstrap, and core re-exports

import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


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


from mc_model_solver_core import (
    Axis,
    Cuboid,
    Rotation,
    Shape,
    SolveConfig,
    _build_shape,
    _shape_symmetry_centers,
    export_minecraft_model,
    format_minecraft_model_json,
    solve,
)
#endregion


#region THEME
# Dark Fusion stylesheet for the application window


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
        "QPushButton#btn_solve { background: #1b3326; border: 1px solid #2d5b3f; }"
        "QPushButton#btn_solve:hover { border: 1px solid #3d7a55; }"
        "QPushButton#btn_reseed { background: #1a2838; border: 1px solid #2d4664; }"
        "QPushButton#btn_reseed:hover { border: 1px solid #3a5f86; }"
        "QPushButton#btn_clear { background: #241a2c; border: 1px solid #4a2c63; }"
        "QPushButton#btn_clear:hover { border: 1px solid #6a3b90; }"
        "QPushButton#btn_save { background: #241a2c; border: 1px solid #4a2c63; }"
        "QPushButton#btn_save:hover { border: 1px solid #6a3b90; }"
        "QSplitter::handle { background: #0b0b0d; }"
    )
#endregion


#region VIEWPORT
# 3D wireframe/face viewport with mouse orbit, depth-sorted polygon fill,
# and an axis gizmo overlay


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
        self._cuboids = []

        self._draw_faces = False
        self._faces_translucent = True
        self._draw_wireframe = True

        self._draw_gizmo = True

    def set_render_options(self, *, faces: bool, faces_translucent: bool, wireframe: bool) -> None:
        self._draw_faces = bool(faces)
        self._faces_translucent = bool(faces_translucent)
        self._draw_wireframe = bool(wireframe)
        self.update()

    def set_gizmo_enabled(self, enabled: bool) -> None:
        self._draw_gizmo = bool(enabled)
        self.update()

    def clear_cuboids(self) -> None:
        self._cuboids = []
        self.update()

    def set_cuboids(self, cuboids) -> None:
        self._cuboids = list(cuboids)
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
            mouse_pt = event.position().toPoint()
            dx = mouse_pt.x() - self._last_mouse_pos.x()
            dy = mouse_pt.y() - self._last_mouse_pos.y()
            self._last_mouse_pos = mouse_pt
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

        def to_camera(pt: tuple[float, float, float]) -> tuple[float, float, float]:
            x, y, z = pt
            x -= cx0
            y -= cy0
            z -= cz0
            x, y, z = rot_y(x, y, z, yaw)
            x, y, z = rot_x(x, y, z, pitch)
            z += self._distance
            return x, y, z

        f = min(w, h) * 0.9

        def project(pt: tuple[float, float, float]) -> Optional[tuple[QtCore.QPointF, float]]:
            x, y, z = to_camera(pt)
            if z <= 0.05:
                return None
            sx = (x * f) / z + (w * 0.5)
            sy = (-y * f) / z + (h * 0.5)
            return QtCore.QPointF(sx, sy), z

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
            pt: tuple[float, float, float],
        ) -> tuple[float, float, float]:
            if angle_deg == 0.0:
                return pt

            ox, oy, oz = origin
            x, y, z = pt
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
            pa = project(a)
            pb = project(b)
            if pa is None or pb is None:
                return
            _p2a, za = pa
            _p2b, zb = pb
            segments.append((a, b, col, (za + zb) * 0.5))

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
        for cub in self._cuboids:
            fx, fy, fz = cub.fr
            tx, ty, tz = cub.to

            p000 = (fx, fy, fz)
            p100 = (tx, fy, fz)
            p110 = (tx, ty, fz)
            p010 = (fx, ty, fz)
            p001 = (fx, fy, tz)
            p101 = (tx, fy, tz)
            p111 = (tx, ty, tz)
            p011 = (fx, ty, tz)

            pts = [p000, p100, p110, p010, p001, p101, p111, p011]

            if cub.rotation is not None and float(cub.rotation.angle) != 0.0:
                axis = str(cub.rotation.axis)
                angle = float(cub.rotation.angle)
                origin = (float(cub.rotation.origin[0]), float(cub.rotation.origin[1]), float(cub.rotation.origin[2]))
                pts = [rotate_about_axis(axis=axis, angle_deg=angle, origin=origin, p=pt) for pt in pts]

            if self._draw_faces:
                cam_pts = [to_camera(pt) for pt in pts]

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
                    add_seg(pts[ia], pts[ib], cub_col)

        if self._draw_faces and faces:
            faces.sort(key=lambda it: it[0], reverse=True)
            painter.setPen(QtCore.Qt.PenStyle.NoPen)
            for _depth, poly, col in faces:
                painter.setBrush(QtGui.QBrush(col))
                painter.drawPolygon(poly)

        segments.sort(key=lambda s: s[3], reverse=True)

        for a, b, col, _z in segments:
            pa = project(a)
            pb = project(b)
            if pa is None or pb is None:
                continue
            p2a, _za = pa
            p2b, _zb = pb
            pen = QtGui.QPen(col)
            pen.setWidthF(1.4)
            painter.setPen(pen)
            painter.drawLine(p2a, p2b)

        if self._draw_gizmo:
            origin = QtCore.QPointF(56.0, float(h) - 56.0)
            L = 34.0

            def cam_vec(v: tuple[float, float, float]) -> tuple[float, float, float]:
                x, y, z = v
                x, y, z = rot_y(float(x), float(y), float(z), yaw)
                x, y, z = rot_x(float(x), float(y), float(z), pitch)
                return x, y, z

            def draw_arrow(end: QtCore.QPointF, col: QtGui.QColor, label: str) -> None:
                pen = QtGui.QPen(col)
                pen.setWidthF(2.4)
                painter.setPen(pen)
                painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)
                painter.drawLine(origin, end)

                dx = float(end.x() - origin.x())
                dy = float(end.y() - origin.y())
                d = math.hypot(dx, dy)
                if d > 1e-6:
                    ux = dx / d
                    uy = dy / d
                    px = -uy
                    py = ux
                    ah = 7.5
                    aw = 4.5
                    tip = end
                    base = QtCore.QPointF(float(end.x() - ux * ah), float(end.y() - uy * ah))
                    p1 = QtCore.QPointF(float(base.x() + px * aw), float(base.y() + py * aw))
                    p2 = QtCore.QPointF(float(base.x() - px * aw), float(base.y() - py * aw))
                    painter.setPen(QtCore.Qt.PenStyle.NoPen)
                    painter.setBrush(QtGui.QBrush(col))
                    painter.drawPolygon(QtGui.QPolygonF([tip, p1, p2]))

                painter.setPen(QtGui.QPen(col))
                painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)
                painter.setFont(QtGui.QFont("Consolas", 9))
                painter.drawText(end + QtCore.QPointF(4.0, -4.0), label)

            xcv = cam_vec((1.0, 0.0, 0.0))
            ycv = cam_vec((0.0, 1.0, 0.0))
            zcv = cam_vec((0.0, 0.0, 1.0))

            ex = origin + QtCore.QPointF(float(xcv[0]) * L, float(-xcv[1]) * L)
            ey = origin + QtCore.QPointF(float(ycv[0]) * L, float(-ycv[1]) * L)
            ez = origin + QtCore.QPointF(float(zcv[0]) * L, float(-zcv[1]) * L)

            draw_arrow(ex, QtGui.QColor(210, 70, 70, 230), "X")
            draw_arrow(ey, QtGui.QColor(70, 210, 120, 230), "Y")
            draw_arrow(ez, QtGui.QColor(85, 140, 230, 230), "Z")

        painter.end()
#endregion


#region WINDOW
# UiState dataclass, SolverWorker thread bridge, and SolverMainWindow:
# the main application window with target controls, solver settings,
# viewport, output tabs, and solve/clear/save handlers


@dataclass
class UiState:
    elements_json: Optional[dict] = None


class SolverWorker(QtCore.QObject):
    finished = QtCore.Signal(list)
    failed = QtCore.Signal(str)

    def __init__(self, *, shape: Shape, config: SolveConfig):
        super().__init__()
        self._shape = shape
        self._config = config

    @QtCore.Slot()
    def run(self) -> None:
        try:
            elements = solve(shape=self._shape, config=self._config)
            self.finished.emit(elements)
        except Exception as e:
            self.failed.emit(str(e))


class SolverMainWindow(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Minecraft Model JSON Solver")

        self._ui_state = UiState()
        self._current_elements = []

        self._prev_step_value: Optional[float] = None

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

        self._thread: Optional[QtCore.QThread] = None
        self._worker: Optional[SolverWorker] = None

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

        output_tabs = QtWidgets.QTabWidget()
        output_tabs.addTab(self._txt_elements, "Elements")
        output_tabs.addTab(self._txt_model_json, "Model JSON")

        output_box = QtWidgets.QGroupBox("Output")
        output_box.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Preferred,
            QtWidgets.QSizePolicy.Policy.Expanding,
        )
        output_layout = QtWidgets.QVBoxLayout(output_box)
        output_layout.setContentsMargins(10, 10, 10, 10)
        output_layout.addWidget(output_tabs)

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

    def _sync_symmetry_options(self) -> None:
        radial = bool(self._radial_symmetry.isChecked())
        self._sym_x.setEnabled(not radial)
        self._sym_y.setEnabled(not radial)
        self._sym_z.setEnabled(not radial)

    def _sync_precision_options(self) -> None:
        hp = bool(self._high_precision.isChecked())
        self._min_span_2d.setEnabled(hp)
        if hp:
            if self._prev_step_value is None:
                self._prev_step_value = float(self._step.value())
            self._step.setValue(0.01)
            self._step.setEnabled(False)
            return

        self._step.setEnabled(True)
        if self._prev_step_value is not None:
            self._step.setValue(float(self._prev_step_value))
        self._prev_step_value = None

    def _build_controls(self) -> QtWidgets.QWidget:
        w = QtWidgets.QWidget()
        w.setMinimumWidth(340)

        self._shape_combo = QtWidgets.QComboBox()
        self._shape_combo.addItem("Sphere", userData="sphere")
        self._shape_combo.addItem("Cone", userData="cone")
        self._shape_combo.addItem("Pyramid", userData="pyramid")
        self._shape_combo.addItem("Double Helix", userData="double_helix")
        self._shape_combo.currentIndexChanged.connect(self._on_target_changed)

        self._target_stack = QtWidgets.QStackedWidget()

        self._sphere_center = QtWidgets.QLineEdit("8,8,8")
        self._sphere_radius = QtWidgets.QDoubleSpinBox()
        self._sphere_radius.setRange(0.1, 128.0)
        self._sphere_radius.setValue(7.5)
        self._sphere_radius.setDecimals(2)

        sphere_page = QtWidgets.QWidget()
        sphere_form = QtWidgets.QFormLayout(sphere_page)
        sphere_form.setContentsMargins(0, 0, 0, 0)
        sphere_form.addRow("Center", self._sphere_center)
        sphere_form.addRow("Radius", self._sphere_radius)

        self._cone_center_xz = QtWidgets.QLineEdit("8,8")
        self._cone_base_y = QtWidgets.QDoubleSpinBox()
        self._cone_base_y.setRange(-64.0, 64.0)
        self._cone_base_y.setValue(0.0)
        self._cone_base_y.setDecimals(2)
        self._cone_height = QtWidgets.QDoubleSpinBox()
        self._cone_height.setRange(0.1, 128.0)
        self._cone_height.setValue(16.0)
        self._cone_height.setDecimals(2)
        self._cone_radius = QtWidgets.QDoubleSpinBox()
        self._cone_radius.setRange(0.1, 128.0)
        self._cone_radius.setValue(8.0)
        self._cone_radius.setDecimals(2)

        cone_page = QtWidgets.QWidget()
        cone_form = QtWidgets.QFormLayout(cone_page)
        cone_form.setContentsMargins(0, 0, 0, 0)
        cone_form.addRow("Center XZ", self._cone_center_xz)
        cone_form.addRow("Base Y", self._cone_base_y)
        cone_form.addRow("Height", self._cone_height)
        cone_form.addRow("Radius", self._cone_radius)

        self._pyramid_center_xz = QtWidgets.QLineEdit("8,8")
        self._pyramid_base_y = QtWidgets.QDoubleSpinBox()
        self._pyramid_base_y.setRange(-64.0, 64.0)
        self._pyramid_base_y.setValue(0.0)
        self._pyramid_base_y.setDecimals(2)
        self._pyramid_height = QtWidgets.QDoubleSpinBox()
        self._pyramid_height.setRange(0.1, 128.0)
        self._pyramid_height.setValue(16.0)
        self._pyramid_height.setDecimals(2)
        self._pyramid_half_base = QtWidgets.QDoubleSpinBox()
        self._pyramid_half_base.setRange(0.1, 128.0)
        self._pyramid_half_base.setValue(8.0)
        self._pyramid_half_base.setDecimals(2)

        pyramid_page = QtWidgets.QWidget()
        pyramid_form = QtWidgets.QFormLayout(pyramid_page)
        pyramid_form.setContentsMargins(0, 0, 0, 0)
        pyramid_form.addRow("Center XZ", self._pyramid_center_xz)
        pyramid_form.addRow("Base Y", self._pyramid_base_y)
        pyramid_form.addRow("Height", self._pyramid_height)
        pyramid_form.addRow("Half base", self._pyramid_half_base)

        self._helix_center_xz = QtWidgets.QLineEdit("8,8")
        self._helix_base_y = QtWidgets.QDoubleSpinBox()
        self._helix_base_y.setRange(-64.0, 64.0)
        self._helix_base_y.setValue(0.0)
        self._helix_base_y.setDecimals(2)
        self._helix_height = QtWidgets.QDoubleSpinBox()
        self._helix_height.setRange(0.1, 128.0)
        self._helix_height.setValue(16.0)
        self._helix_height.setDecimals(2)
        self._helix_turns = QtWidgets.QDoubleSpinBox()
        self._helix_turns.setRange(0.1, 32.0)
        self._helix_turns.setValue(2.0)
        self._helix_turns.setDecimals(2)
        self._helix_radius_start = QtWidgets.QDoubleSpinBox()
        self._helix_radius_start.setRange(0.1, 128.0)
        self._helix_radius_start.setValue(2.0)
        self._helix_radius_start.setDecimals(2)
        self._helix_radius_end = QtWidgets.QDoubleSpinBox()
        self._helix_radius_end.setRange(0.1, 128.0)
        self._helix_radius_end.setValue(6.0)
        self._helix_radius_end.setDecimals(2)
        self._helix_tube_radius = QtWidgets.QDoubleSpinBox()
        self._helix_tube_radius.setRange(0.1, 16.0)
        self._helix_tube_radius.setValue(1.25)
        self._helix_tube_radius.setDecimals(2)

        helix_page = QtWidgets.QWidget()
        helix_form = QtWidgets.QFormLayout(helix_page)
        helix_form.setContentsMargins(0, 0, 0, 0)
        helix_form.addRow("Center XZ", self._helix_center_xz)
        helix_form.addRow("Base Y", self._helix_base_y)
        helix_form.addRow("Height", self._helix_height)
        helix_form.addRow("Turns", self._helix_turns)
        helix_form.addRow("Radius start", self._helix_radius_start)
        helix_form.addRow("Radius end", self._helix_radius_end)
        helix_form.addRow("Tube radius", self._helix_tube_radius)

        self._target_stack.addWidget(sphere_page)
        self._target_stack.addWidget(cone_page)
        self._target_stack.addWidget(pyramid_page)
        self._target_stack.addWidget(helix_page)

        self._grid = QtWidgets.QSpinBox()
        self._grid.setRange(4, 80)
        self._grid.setValue(24)

        self._max_elements = QtWidgets.QSpinBox()
        self._max_elements.setRange(1, 500)
        self._max_elements.setValue(48)

        self._candidates = QtWidgets.QSpinBox()
        self._candidates.setRange(1, 5000)
        self._candidates.setValue(100)

        self._outside_sample = QtWidgets.QDoubleSpinBox()
        self._outside_sample.setRange(0.0, 1.0)
        self._outside_sample.setValue(0.25)
        self._outside_sample.setDecimals(3)

        self._spill_weight = QtWidgets.QDoubleSpinBox()
        self._spill_weight.setRange(0.0, 1000.0)
        self._spill_weight.setValue(2.0)
        self._spill_weight.setDecimals(3)

        self._volume_weight = QtWidgets.QDoubleSpinBox()
        self._volume_weight.setRange(0.0, 10.0)
        self._volume_weight.setValue(0.0)
        self._volume_weight.setDecimals(6)

        self._step = QtWidgets.QDoubleSpinBox()
        self._step.setRange(0.001, 2.0)
        self._step.setValue(0.25)
        self._step.setDecimals(3)

        self._min_span_2d = QtWidgets.QDoubleSpinBox()
        self._min_span_2d.setRange(0.0, 16.0)
        self._min_span_2d.setValue(0.5)
        self._min_span_2d.setDecimals(3)

        self._high_precision = QtWidgets.QCheckBox("High precision (0.01)")
        self._high_precision.setChecked(False)
        self._high_precision.stateChanged.connect(self._sync_precision_options)

        self._allow_rot = QtWidgets.QCheckBox("Allow rotations")
        self._allow_rot.setChecked(True)

        self._hollow = QtWidgets.QCheckBox("Hollow (surface)")
        self._hollow.setChecked(False)

        self._sym_x = QtWidgets.QCheckBox("Symmetry X")
        self._sym_x.setChecked(False)

        self._sym_y = QtWidgets.QCheckBox("Symmetry Y")
        self._sym_y.setChecked(False)

        self._sym_z = QtWidgets.QCheckBox("Symmetry Z")
        self._sym_z.setChecked(False)

        self._radial_symmetry = QtWidgets.QCheckBox("Radial symmetry (6 sides)")
        self._radial_symmetry.setChecked(False)
        self._radial_symmetry.stateChanged.connect(self._sync_symmetry_options)

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

        self._rot_axes = QtWidgets.QLineEdit("x,y,z")

        self._seed = QtWidgets.QSpinBox()
        self._seed.setRange(0, 2_000_000_000)
        self._seed.setValue(0)

        self._texture = QtWidgets.QLineEdit("minecraft:block/oak_planks")

        self._btn_solve = QtWidgets.QPushButton("Solve")
        self._btn_solve.setObjectName("btn_solve")
        self._btn_solve.clicked.connect(self._on_solve)

        self._btn_reseed = QtWidgets.QPushButton("Reseed")
        self._btn_reseed.setObjectName("btn_reseed")
        self._btn_reseed.clicked.connect(self._on_reseed)

        self._btn_clear = QtWidgets.QPushButton("Clear")
        self._btn_clear.setObjectName("btn_clear")
        self._btn_clear.clicked.connect(self._on_clear)

        self._btn_save = QtWidgets.QPushButton("Save JSON")
        self._btn_save.setObjectName("btn_save")
        self._btn_save.clicked.connect(self._on_save_json)

        self._lbl_status = QtWidgets.QLabel("Ready")
        self._lbl_status.setWordWrap(True)

        target_box = QtWidgets.QGroupBox("Target")
        target_layout = QtWidgets.QVBoxLayout(target_box)
        target_layout.setContentsMargins(10, 10, 10, 10)
        target_layout.addWidget(self._shape_combo)
        target_layout.addWidget(self._target_stack)

        settings_box = QtWidgets.QGroupBox("Solver")
        settings_form = QtWidgets.QFormLayout(settings_box)
        settings_form.setContentsMargins(10, 10, 10, 10)
        settings_form.addRow("Grid", self._grid)
        settings_form.addRow("Max elements", self._max_elements)
        settings_form.addRow("Candidates", self._candidates)
        settings_form.addRow("Outside sample", self._outside_sample)
        settings_form.addRow("Spill weight", self._spill_weight)
        settings_form.addRow("Volume weight", self._volume_weight)
        settings_form.addRow("Snap step", self._step)
        settings_form.addRow("", self._high_precision)
        settings_form.addRow("Min span (2 sides)", self._min_span_2d)
        settings_form.addRow("Rot axes", self._rot_axes)
        settings_form.addRow("Seed", self._seed)
        settings_form.addRow("Texture", self._texture)

        btn_row = QtWidgets.QHBoxLayout()
        btn_row.addWidget(self._btn_solve)
        btn_row.addWidget(self._btn_reseed)
        btn_row.addWidget(self._btn_clear)

        btn_row2 = QtWidgets.QHBoxLayout()
        btn_row2.addWidget(self._btn_save)

        viewport_box = QtWidgets.QGroupBox("Viewport")
        viewport_layout = QtWidgets.QVBoxLayout(viewport_box)
        viewport_layout.setContentsMargins(10, 10, 10, 10)
        viewport_layout.addWidget(self._vp_faces)
        viewport_layout.addWidget(self._vp_translucent)
        viewport_layout.addWidget(self._vp_wireframe)

        layout = QtWidgets.QVBoxLayout(w)
        layout.addWidget(target_box)
        layout.addWidget(settings_box)
        layout.addWidget(self._allow_rot)
        layout.addWidget(self._hollow)
        layout.addWidget(self._sym_x)
        layout.addWidget(self._sym_y)
        layout.addWidget(self._sym_z)
        layout.addWidget(self._radial_symmetry)
        layout.addWidget(viewport_box)
        layout.addLayout(btn_row)
        layout.addLayout(btn_row2)
        layout.addWidget(self._lbl_status)
        layout.addStretch(1)

        self._sync_viewport_render_options()
        self._sync_precision_options()
        self._sync_symmetry_options()

        return w

    def _parse_vec2(self, raw: str) -> tuple[float, float]:
        parts = [part.strip() for part in raw.split(",")]
        if len(parts) != 2:
            raise ValueError("Expected x,z")
        return float(parts[0]), float(parts[1])

    def _parse_center3(self) -> tuple[float, float, float]:
        raw = self._sphere_center.text().strip()
        parts = [part.strip() for part in raw.split(",")]
        if len(parts) != 3:
            raise ValueError("Center must be x,y,z")
        return float(parts[0]), float(parts[1]), float(parts[2])

    def _current_target(self) -> str:
        target_data = self._shape_combo.currentData()
        return str(target_data) if target_data is not None else "sphere"

    def _on_target_changed(self) -> None:
        target = self._current_target()
        if target == "sphere":
            self._target_stack.setCurrentIndex(0)
        elif target == "cone":
            self._target_stack.setCurrentIndex(1)
        elif target == "pyramid":
            self._target_stack.setCurrentIndex(2)
        else:
            self._target_stack.setCurrentIndex(3)

    def _parse_axes(self) -> list[Axis]:
        raw = self._rot_axes.text().strip()
        parts = [axis.strip() for axis in raw.split(",") if axis.strip()]
        if not parts:
            return ["x", "y", "z"]
        out: list[Axis] = []
        for axis in parts:
            if axis not in {"x", "y", "z"}:
                raise ValueError("Rot axes must be subset of x,y,z")
            out.append(axis)
        return out

    def _set_busy(self, busy: bool) -> None:
        self._btn_solve.setEnabled(not busy)
        self._btn_reseed.setEnabled(not busy)
        self._btn_save.setEnabled(not busy)

    def _on_reseed(self) -> None:
        self._seed.setValue(random.randint(0, 2_000_000_000))

    def _on_clear(self) -> None:
        self._current_elements = []
        self._ui_state.elements_json = None
        self._viewport.clear_cuboids()
        self._txt_elements.setPlainText("")
        self._txt_model_json.setPlainText("")
        self._lbl_status.setText("Cleared")

    def _on_solve(self) -> None:
        if self._thread is not None:
            return

        try:
            sphere_center = self._parse_center3()
            axes = self._parse_axes()
            cone_center_xz = self._parse_vec2(self._cone_center_xz.text().strip())
            pyramid_center_xz = self._parse_vec2(self._pyramid_center_xz.text().strip())
            helix_center_xz = self._parse_vec2(self._helix_center_xz.text().strip())
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Invalid input", str(e))
            return

        target = self._current_target()
        sphere_radius = float(self._sphere_radius.value())
        cone_base_y = float(self._cone_base_y.value())
        cone_height = float(self._cone_height.value())
        cone_radius = float(self._cone_radius.value())
        pyramid_base_y = float(self._pyramid_base_y.value())
        pyramid_height = float(self._pyramid_height.value())
        pyramid_half_base = float(self._pyramid_half_base.value())
        helix_base_y = float(self._helix_base_y.value())
        helix_height = float(self._helix_height.value())
        helix_turns = float(self._helix_turns.value())
        helix_radius_start = float(self._helix_radius_start.value())
        helix_radius_end = float(self._helix_radius_end.value())
        helix_tube_radius = float(self._helix_tube_radius.value())

        try:
            shape = _build_shape(
                target=target,
                sphere_center=sphere_center,
                sphere_radius=sphere_radius,
                cone_center_xz=cone_center_xz,
                cone_base_y=cone_base_y,
                cone_height=cone_height,
                cone_radius=cone_radius,
                pyramid_center_xz=pyramid_center_xz,
                pyramid_base_y=pyramid_base_y,
                pyramid_height=pyramid_height,
                pyramid_half_base=pyramid_half_base,
                helix_center_xz=helix_center_xz,
                helix_base_y=helix_base_y,
                helix_height=helix_height,
                helix_turns=helix_turns,
                helix_radius_start=helix_radius_start,
                helix_radius_end=helix_radius_end,
                helix_tube_radius=helix_tube_radius,
            )
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Invalid target", str(e))
            return

        sym_center_xz, sym_center = _shape_symmetry_centers(
            target=target,
            sphere_center=sphere_center,
            cone_center_xz=cone_center_xz,
            cone_base_y=cone_base_y,
            cone_height=cone_height,
            pyramid_center_xz=pyramid_center_xz,
            pyramid_base_y=pyramid_base_y,
            pyramid_height=pyramid_height,
            helix_center_xz=helix_center_xz,
            helix_base_y=helix_base_y,
            helix_height=helix_height,
        )

        high_precision = bool(self._high_precision.isChecked())
        cfg = SolveConfig(
            grid=int(self._grid.value()),
            max_elements=int(self._max_elements.value()),
            candidates_per_element=int(self._candidates.value()),
            outside_sample_fraction=float(self._outside_sample.value()),
            spill_weight=float(self._spill_weight.value()),
            volume_weight=float(self._volume_weight.value()),
            seed=int(self._seed.value()),
            allow_rotations=bool(self._allow_rot.isChecked()),
            rotation_axes=axes,
            grid_step=float(self._step.value()),
            high_precision=high_precision,
            min_span_2d=float(self._min_span_2d.value()),
            hollow=bool(self._hollow.isChecked()),
            radial_symmetry=bool(self._radial_symmetry.isChecked()),
            symmetry_center_xz=sym_center_xz,
            symmetry_x=bool(self._sym_x.isChecked()),
            symmetry_y=bool(self._sym_y.isChecked()),
            symmetry_z=bool(self._sym_z.isChecked()),
            symmetry_center=sym_center,
        )

        self._lbl_status.setText("Solving...")
        self._set_busy(True)

        self._thread = QtCore.QThread(self)
        self._worker = SolverWorker(shape=shape, config=cfg)
        self._worker.moveToThread(self._thread)

        self._thread.started.connect(self._worker.run)
        self._worker.finished.connect(self._on_solve_finished)
        self._worker.failed.connect(self._on_solve_failed)

        self._worker.finished.connect(self._thread.quit)
        self._worker.failed.connect(self._thread.quit)
        self._thread.finished.connect(self._cleanup_thread)

        self._thread.start()

    @QtCore.Slot(list)
    def _on_solve_finished(self, elements) -> None:
        self._current_elements = elements
        self._viewport.set_cuboids(elements)
        self._lbl_status.setText(f"Solved: {len(elements)} elements")

        lines: list[str] = []
        for i, cub in enumerate(elements):
            rot = "-"
            if cub.rotation is not None and float(cub.rotation.angle) != 0.0:
                rot = f"{cub.rotation.axis} {float(cub.rotation.angle):g} @ ({cub.rotation.origin[0]:g},{cub.rotation.origin[1]:g},{cub.rotation.origin[2]:g})"
            lines.append(
                f"{i:03d} from=({cub.fr[0]:g},{cub.fr[1]:g},{cub.fr[2]:g}) to=({cub.to[0]:g},{cub.to[1]:g},{cub.to[2]:g}) rot={rot}"
            )
        self._txt_elements.setPlainText("\n".join(lines))

        try:
            model = export_minecraft_model(elements=elements, texture=self._texture.text().strip())
            self._ui_state.elements_json = model
            self._txt_model_json.setPlainText(format_minecraft_model_json(model))
        except Exception:
            self._ui_state.elements_json = None
            self._txt_model_json.setPlainText("")

    @QtCore.Slot(str)
    def _on_solve_failed(self, msg: str) -> None:
        self._lbl_status.setText("Solve failed")
        QtWidgets.QMessageBox.critical(self, "Solve failed", msg)

    @QtCore.Slot()
    def _cleanup_thread(self) -> None:
        self._set_busy(False)

        if self._worker is not None:
            self._worker.deleteLater()
        self._worker = None

        if self._thread is not None:
            self._thread.deleteLater()
        self._thread = None

    def _on_save_json(self) -> None:
        model = self._ui_state.elements_json
        if model is None:
            QtWidgets.QMessageBox.information(self, "Nothing to save", "Run Solve first.")
            return

        out_path, _filter = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save Minecraft model JSON",
            str(Path.cwd() / "model.json"),
            "JSON (*.json)",
        )
        if not out_path:
            return

        Path(out_path).write_text(format_minecraft_model_json(model) + "\n", encoding="utf-8", newline="\n")
        self._lbl_status.setText(f"Saved: {out_path}")
#endregion


#region ENTRY
# Application bootstrap: create QApplication, apply theme, show main window


def main() -> None:
    app = QtWidgets.QApplication(sys.argv)
    _apply_dark_theme(app)
    w = SolverMainWindow()
    w.resize(1200, 780)
    w.show()
    sys.exit(app.exec())
#endregion


if __name__ == "__main__":
    main()
