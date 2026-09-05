"""
Minecraft Model Greedy Optimizer (UI)

PySide6 UI wrapper around `mc_model_greedy_optimizer_core.py`.

Load an existing Minecraft block-model `.json` and reduce the number of
`elements` (cuboids) by merging pairs that can be replaced by a single cuboid,
preserving the exact occupied volume. Works with rotated cuboids: elements
sharing the same rotation frame are merged in their local axis-aligned space.

See the core module docstring for the algorithm details.

TOOLSGROUP::MODEL
SORTGROUP::7
SORTPRIORITY::76
STATUS::active
VERSION::20260316
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Iterable, Optional

try:
    from mc_model_solver_ui import Cuboid, _apply_dark_theme
except Exception as e:  # pragma: no cover - import guard
    print(f"Missing dependency: mc_model_solver_ui.py ({e})")
    raise

from mc_model_optimizer_core import (
    Element,
    OptimizeResult,
    _optimize_model,
    _parse_element,
    format_blockbench_json,
)

import PySide6  # noqa: F401  (ensure import error surfaces clearly)
from PySide6 import QtCore, QtGui, QtWidgets


class ColorizeViewport(QtWidgets.QWidget):
    """3D viewport with priority-based wireframe rendering.

    When colorize mode is enabled, wireframes are drawn in priority order:
    priority 0 (red, removed) first, priority 1 (blue, untouched) next,
    priority 2 (green, new) last so green renders on top.
    """

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
        self._colors: list[QtGui.QColor] = []
        self._priorities: list[int] = []

        self._draw_faces = True
        self._faces_translucent = False
        self._draw_wireframe = True
        self._colorize_enabled = False

    def set_render_options(self, *, faces: bool, faces_translucent: bool, wireframe: bool) -> None:
        self._draw_faces = bool(faces)
        self._faces_translucent = bool(faces_translucent)
        self._draw_wireframe = bool(wireframe)
        self.update()

    def set_colorize_enabled(self, enabled: bool) -> None:
        self._colorize_enabled = bool(enabled)
        self.update()

    def set_cuboids(
        self,
        cuboids: Iterable[Cuboid],
        *,
        colors: Optional[list[QtGui.QColor]] = None,
        priorities: Optional[list[int]] = None,
    ) -> None:
        self._cuboids = list(cuboids)
        self._colors = list(colors) if colors is not None else []
        self._priorities = list(priorities) if priorities is not None else []
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
            *, axis: str, angle_deg: float, origin: tuple[float, float, float], p: tuple[float, float, float]
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

        # segments: (point_a, point_b, color, sort_key, priority)
        segments: list[tuple[tuple[float, float, float], tuple[float, float, float], QtGui.QColor, float, int]] = []
        faces: list[tuple[float, QtGui.QPolygonF, QtGui.QColor]] = []

        def add_seg(a: tuple[float, float, float], b: tuple[float, float, float], col: QtGui.QColor, prio: int) -> None:
            ca = to_camera(a)
            cb = to_camera(b)
            pa = project_cam(ca)
            pb = project_cam(cb)
            if pa is None or pb is None:
                return
            depth = (ca[2] + cb[2]) * 0.5
            segments.append((a, b, col, depth, prio))

        # world axes
        o = (0.0, 0.0, 0.0)
        add_seg(o, (16.0, 0.0, 0.0), QtGui.QColor(210, 70, 70, 220), 0)
        add_seg(o, (0.0, 16.0, 0.0), QtGui.QColor(70, 210, 120, 220), 0)
        add_seg(o, (0.0, 0.0, 16.0), QtGui.QColor(85, 140, 230, 220), 0)

        # unit block wireframe
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
            (c0, c1), (c1, c2), (c2, c3), (c3, c0),
            (c4, c5), (c5, c6), (c6, c7), (c7, c4),
            (c0, c4), (c1, c5), (c2, c6), (c3, c7),
        ):
            add_seg(a, b, cube_col, 0)

        for i_c, c in enumerate(self._cuboids):
            base_col = QtGui.QColor(95, 170, 255, 190)
            if 0 <= int(i_c) < len(self._colors):
                base_col = self._colors[int(i_c)]

            prio = 1
            if 0 <= int(i_c) < len(self._priorities):
                prio = int(self._priorities[int(i_c)])

            face_alpha = 200 if not self._faces_translucent else 70
            face_base = QtGui.QColor(base_col.red(), base_col.green(), base_col.blue(), face_alpha)
            line_col = QtGui.QColor(
                base_col.red(), base_col.green(), base_col.blue(), min(255, base_col.alpha() + 30)
            )

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
                origin = (
                    float(c.rotation.origin[0]),
                    float(c.rotation.origin[1]),
                    float(c.rotation.origin[2]),
                )
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
                    (0, 1), (1, 2), (2, 3), (3, 0),
                    (4, 5), (5, 6), (6, 7), (7, 4),
                    (0, 4), (1, 5), (2, 6), (3, 7),
                )
                for ia, ib in edges:
                    add_seg(pts[ia], pts[ib], line_col, prio)

        # draw faces sorted by depth (back to front)
        if self._draw_faces and faces:
            faces.sort(key=lambda it: it[0], reverse=True)
            painter.setPen(QtCore.Qt.PenStyle.NoPen)
            for _depth, poly, col in faces:
                painter.setBrush(QtGui.QBrush(col))
                painter.drawPolygon(poly)

        # draw wireframes: priority order when colorize is on, depth order otherwise
        if self._colorize_enabled:
            # priority ascending: red(0) first, blue(1) next, green(2) last (on top)
            segments.sort(key=lambda s: (s[4], s[3]), reverse=False)
        else:
            # depth sort: back to front
            segments.sort(key=lambda s: s[3], reverse=True)

        for a, b, col, _z, _p in segments:
            pa = project_cam(to_camera(a))
            pb = project_cam(to_camera(b))
            if pa is None or pb is None:
                continue
            pen = QtGui.QPen(col)
            pen.setWidthF(1.4)
            painter.setPen(pen)
            painter.drawLine(pa, pb)

        painter.end()


# ---------------------------------------------------------------------------
# Colors for change visualization
# ---------------------------------------------------------------------------

_COL_REMOVED = QtGui.QColor(235, 70, 70, 220)    # red
_COL_UNTOUCHED = QtGui.QColor(90, 140, 230, 220)  # blue
_COL_NEW = QtGui.QColor(85, 210, 120, 230)        # green
_COL_DEFAULT = QtGui.QColor(95, 170, 255, 200)

_PRIO_REMOVED = 0
_PRIO_UNTOUCHED = 1
_PRIO_NEW = 2


class GreedyOptimizerMainWindow(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Minecraft Model Greedy Optimizer")

        self._source_path: Optional[Path] = None
        self._model: Optional[dict] = None
        self._elements: list[Element] = []

        self._result: Optional[OptimizeResult] = None
        self._optimized_elements: list[Element] = []

        # view mode: "original" or "optimized"
        self._view_mode = "original"

        self._viewport = ColorizeViewport()

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
        w.setMinimumWidth(360)

        self._txt_path = QtWidgets.QLineEdit("")
        self._txt_path.setReadOnly(True)

        self._btn_load = QtWidgets.QPushButton("Load JSON")
        self._btn_load.clicked.connect(self._on_load)

        self._merge_eps = QtWidgets.QDoubleSpinBox()
        self._merge_eps.setRange(0.0, 1.0)
        self._merge_eps.setDecimals(6)
        self._merge_eps.setValue(0.001)
        self._merge_eps.setSingleStep(0.001)

        self._btn_optimize = QtWidgets.QPushButton("Optimize")
        self._btn_optimize.setObjectName("btn_solve")
        self._btn_optimize.clicked.connect(self._on_optimize)
        self._btn_optimize.setEnabled(False)

        self._btn_reduce_overlaps = QtWidgets.QPushButton("Reduce Overlaps")
        self._btn_reduce_overlaps.setObjectName("btn_overlap")
        self._btn_reduce_overlaps.setToolTip(
            "Trim cuboids so overlapping intersections are removed while\n"
            "preserving the external (union) volume, then merge adjacent boxes."
        )
        self._btn_reduce_overlaps.clicked.connect(self._on_reduce_overlaps)
        self._btn_reduce_overlaps.setEnabled(False)

        self._btn_save = QtWidgets.QPushButton("Save Optimized JSON")
        self._btn_save.setObjectName("btn_save")
        self._btn_save.clicked.connect(self._on_save)
        self._btn_save.setEnabled(False)

        # view toggle button
        self._btn_toggle_view = QtWidgets.QPushButton("Show: Original")
        self._btn_toggle_view.setObjectName("btn_toggle")
        self._btn_toggle_view.clicked.connect(self._on_toggle_view)
        self._btn_toggle_view.setEnabled(False)

        # colorize checkbox
        self._chk_colorize = QtWidgets.QCheckBox("Colorize changes (red=removed, blue=untouched, green=new)")
        self._chk_colorize.setChecked(False)
        self._chk_colorize.stateChanged.connect(self._on_colorize_changed)
        self._chk_colorize.setEnabled(False)

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

        opt_box = QtWidgets.QGroupBox("Optimize")
        opt_form = QtWidgets.QFormLayout(opt_box)
        opt_form.setContentsMargins(8, 6, 8, 6)
        opt_form.setVerticalSpacing(4)
        opt_form.addRow("Merge epsilon", self._merge_eps)
        opt_form.addRow(self._btn_optimize)
        opt_form.addRow(self._btn_reduce_overlaps)
        opt_form.addRow(self._btn_save)

        view_box = QtWidgets.QGroupBox("View")
        view_layout = QtWidgets.QVBoxLayout(view_box)
        view_layout.setContentsMargins(8, 6, 8, 6)
        view_layout.setSpacing(4)
        view_layout.addWidget(self._btn_toggle_view)
        view_layout.addWidget(self._chk_colorize)
        view_layout.addWidget(self._vp_faces)
        view_layout.addWidget(self._vp_translucent)
        view_layout.addWidget(self._vp_wireframe)

        layout = QtWidgets.QVBoxLayout(w)
        layout.addWidget(file_box)
        layout.addWidget(opt_box)
        layout.addWidget(view_box)
        layout.addWidget(self._lbl_status)
        layout.addStretch(1)
        return w

    def _build_output(self) -> QtWidgets.QWidget:
        w = QtWidgets.QWidget()
        w.setMinimumWidth(460)
        w.setSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred, QtWidgets.QSizePolicy.Policy.Expanding)

        self._txt_log = QtWidgets.QPlainTextEdit()
        self._txt_log.setReadOnly(True)
        self._txt_log.setMaximumBlockCount(20000)
        self._txt_log.setLineWrapMode(QtWidgets.QPlainTextEdit.LineWrapMode.NoWrap)
        self._txt_log.setFont(QtGui.QFont("Consolas", 10))

        self._txt_json = QtWidgets.QPlainTextEdit()
        self._txt_json.setReadOnly(True)
        self._txt_json.setMaximumBlockCount(20000)
        self._txt_json.setLineWrapMode(QtWidgets.QPlainTextEdit.LineWrapMode.NoWrap)
        self._txt_json.setFont(QtGui.QFont("Consolas", 10))

        tabs = QtWidgets.QTabWidget()
        tabs.addTab(self._txt_log, "Log")
        tabs.addTab(self._txt_json, "Optimized JSON")

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

    def _on_colorize_changed(self) -> None:
        self._viewport.set_colorize_enabled(bool(self._chk_colorize.isChecked()))
        self._sync_viewport_model()

    def _on_toggle_view(self) -> None:
        if self._view_mode == "original":
            self._view_mode = "optimized"
            self._btn_toggle_view.setText("Show: Optimized")
        else:
            self._view_mode = "original"
            self._btn_toggle_view.setText("Show: Original")
        self._sync_viewport_model()

    def _sync_viewport_model(self) -> None:
        colorize = bool(self._chk_colorize.isChecked())

        if self._view_mode == "optimized" and self._optimized_elements:
            cubs = [e.cuboid for e in self._optimized_elements]
            if colorize and self._result is not None:
                cols = []
                prios = []
                for status in self._result.output_status:
                    if status == "new":
                        cols.append(_COL_NEW)
                        prios.append(_PRIO_NEW)
                    else:
                        cols.append(_COL_UNTOUCHED)
                        prios.append(_PRIO_UNTOUCHED)
                self._viewport.set_cuboids(cubs, colors=cols, priorities=prios)
            else:
                cols = [_COL_DEFAULT] * len(cubs)
                self._viewport.set_cuboids(cubs, colors=cols)
            return

        if self._view_mode == "original" and self._elements:
            cubs = [e.cuboid for e in self._elements]
            if colorize and self._result is not None:
                cols = []
                prios = []
                for status in self._result.input_status:
                    if status == "removed":
                        cols.append(_COL_REMOVED)
                        prios.append(_PRIO_REMOVED)
                    else:
                        cols.append(_COL_UNTOUCHED)
                        prios.append(_PRIO_UNTOUCHED)
                self._viewport.set_cuboids(cubs, colors=cols, priorities=prios)
            else:
                cols = [_COL_DEFAULT] * len(cubs)
                self._viewport.set_cuboids(cubs, colors=cols)
            return

        self._viewport.set_cuboids([], colors=[])

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

            self._result = None
            self._optimized_elements = []
            self._btn_save.setEnabled(False)
            self._btn_toggle_view.setEnabled(False)
            self._chk_colorize.setEnabled(False)
            self._chk_colorize.setChecked(False)
            self._view_mode = "original"
            self._btn_toggle_view.setText("Show: Original")
            self._viewport.set_colorize_enabled(False)

            self._txt_path.setText(str(p))
            self._btn_optimize.setEnabled(True)
            self._btn_reduce_overlaps.setEnabled(True)

            self._txt_log.setPlainText(f"Loaded {len(elements)} elements from {p.name}")
            self._txt_json.setPlainText("")

            self._sync_viewport_model()
            self._lbl_status.setText(f"Loaded: {p.name} ({len(elements)} elements)")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Load failed", str(e))

    def _on_optimize(self) -> None:
        self._run_pass(reduce_overlaps=False)

    def _on_reduce_overlaps(self) -> None:
        self._run_pass(reduce_overlaps=True)

    def _run_pass(self, *, reduce_overlaps: bool) -> None:
        if self._model is None or not self._elements:
            return

        try:
            eps = float(self._merge_eps.value())

            result = _optimize_model(
                model=self._model,
                elements=self._elements,
                eps=eps,
                reduce_overlaps=reduce_overlaps,
            )
            self._result = result

            els = result.model.get("elements")
            opt_elements: list[Element] = []
            if isinstance(els, list):
                for i, el in enumerate(els):
                    if isinstance(el, dict):
                        opt_elements.append(_parse_element(i, el))
            self._optimized_elements = opt_elements

            self._txt_log.setPlainText("\n".join(result.log))
            self._txt_json.setPlainText(format_blockbench_json(result.model))

            self._btn_save.setEnabled(True)
            self._btn_toggle_view.setEnabled(True)
            self._chk_colorize.setEnabled(True)

            # switch to optimized view automatically
            self._view_mode = "optimized"
            self._btn_toggle_view.setText("Show: Optimized")

            label = "Reduced overlaps" if reduce_overlaps else "Optimized"
            self._lbl_status.setText(
                f"{label}: {result.input_count} -> {result.output_count} elements"
            )
            self._sync_viewport_model()
        except Exception as e:
            op = "Reduce overlaps" if reduce_overlaps else "Optimize"
            QtWidgets.QMessageBox.critical(self, f"{op} failed", str(e))

    def _on_save(self) -> None:
        if self._result is None:
            return

        default = Path.cwd() / "model_optimized.json"
        if self._source_path is not None:
            default = self._source_path.with_name(self._source_path.stem + "_OPT.json")

        out_path, _filter = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save optimized model JSON",
            str(default),
            "JSON (*.json)",
        )
        if not out_path:
            return

        try:
            Path(out_path).write_text(format_blockbench_json(self._result.model) + "\n", encoding="utf-8", newline="\n")
            self._lbl_status.setText(f"Saved: {out_path}")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Save failed", str(e))


def main() -> None:
    app = QtWidgets.QApplication(sys.argv)
    _apply_dark_theme(app)
    w = GreedyOptimizerMainWindow()
    w.resize(1280, 800)
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
