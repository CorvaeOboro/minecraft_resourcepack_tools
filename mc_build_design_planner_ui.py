"""Minecraft Build Design Planner (UI)

 UI for generating and visualizing 2D block designs (pixel shapes) useful
for planning Minecraft builds.

Current focus:
- Circle (default radius=12)
- Double spiral (two arms, 180 degrees apart) from the origin
- Curve thickness controls

TOOLSGROUP::MODEL
SORTGROUP::7
SORTPRIORITY::81
STATUS::active
VERSION::20260316
"""

from __future__ import annotations

import math
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


try:
    from mc_model_solver_ui import _apply_dark_theme
except Exception:

    def _apply_dark_theme(app: QtWidgets.QApplication) -> None:
        app.setStyle("Fusion")


@dataclass(frozen=True)
class ThicknessRule:
    metric: str
    radius: int


def _lerp(a: float, b: float, t: float) -> float:
    tt = 0.0 if t < 0.0 else 1.0 if t > 1.0 else float(t)
    return float(a) * (1.0 - tt) + float(b) * tt


def _bresenham(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    x0, y0 = int(a[0]), int(a[1])
    x1, y1 = int(b[0]), int(b[1])
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy
    out: list[tuple[int, int]] = []
    while True:
        out.append((int(x0), int(y0)))
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy
    return out


def _parse_vec2(raw: str) -> tuple[int, int]:
    parts = [p.strip() for p in raw.split(",")]
    if len(parts) != 2:
        raise ValueError("Expected x,z")
    return int(round(float(parts[0]))), int(round(float(parts[1])))


def _neighbors_within(rule: ThicknessRule) -> list[tuple[int, int]]:
    r = int(rule.radius)
    if r <= 0:
        return [(0, 0)]

    out: list[tuple[int, int]] = []
    for dz in range(-r, r + 1):
        for dx in range(-r, r + 1):
            adx = abs(int(dx))
            adz = abs(int(dz))
            if rule.metric == "manhattan":
                if (adx + adz) <= r:
                    out.append((dx, dz))
            elif rule.metric == "chebyshev":
                if max(adx, adz) <= r:
                    out.append((dx, dz))
            else:
                if (dx * dx + dz * dz) <= (r * r):
                    out.append((dx, dz))
    return out


def _dilate(points: set[tuple[int, int]], *, rule: ThicknessRule) -> set[tuple[int, int]]:
    offsets = _neighbors_within(rule)
    out: set[tuple[int, int]] = set()
    for x, z in points:
        for dx, dz in offsets:
            out.add((int(x) + int(dx), int(z) + int(dz)))
    return out


def _corner_fill(points: set[tuple[int, int]]) -> set[tuple[int, int]]:
    if not points:
        return set()

    out = set(points)
    diags = ((1, 1), (1, -1), (-1, 1), (-1, -1))
    for x, z in points:
        for dx, dz in diags:
            if (int(x) + int(dx), int(z) + int(dz)) in points:
                out.add((int(x) + int(dx), int(z)))
                out.add((int(x), int(z) + int(dz)))
    return out


def _gen_circle_points(
    *,
    center: tuple[int, int],
    radius: float,
    filled: bool,
) -> set[tuple[int, int]]:
    cx, cz = center
    r = float(radius)
    if r <= 0.0:
        return set()

    rr = int(math.ceil(r + 2.0))
    out: set[tuple[int, int]] = set()

    for z in range(int(cz) - rr, int(cz) + rr + 1):
        for x in range(int(cx) - rr, int(cx) + rr + 1):
            dx = float(x) - float(cx)
            dz = float(z) - float(cz)
            d = math.sqrt(dx * dx + dz * dz)
            if filled:
                if d <= (r + 0.5):
                    out.add((int(x), int(z)))
            else:
                if abs(d - r) <= 0.5:
                    out.add((int(x), int(z)))

    return out


def _gen_double_spiral_points(
    *,
    center: tuple[int, int],
    radius_limit: float,
    spacing: float,
    guide_rotation_deg: float,
    segments: int,
    max_chord_step: float,
) -> set[tuple[int, int]]:
    cx, cz = center
    rlim = float(radius_limit)
    if rlim <= 0.0:
        return set()

    sp = float(spacing)
    if sp <= 1e-9:
        sp = 1.0

    turns = float(rlim) / float(sp)
    turns = max(0.0, float(turns))

    segs = max(8, int(segments))
    max_step = float(max_chord_step)
    max_step = 0.0 if max_step < 0.0 else float(max_step)

    guide_rad = math.radians(float(guide_rotation_deg))

    def spiral_point(tt: float, phase: float) -> tuple[float, float]:
        a = (2.0 * math.pi) * float(turns) * float(tt) + float(phase) + float(guide_rad)
        r = _lerp(0.0, float(rlim), float(tt))
        return (float(cx) + r * math.cos(a), float(cz) + r * math.sin(a))

    t_breaks: list[float] = []
    for i in range(segs + 1):
        t_breaks.append(float(i) / float(segs))

    if max_step > 1e-9:
        refined: list[float] = [float(t_breaks[0])]
        for i in range(len(t_breaks) - 1):
            ta = float(t_breaks[i])
            tb = float(t_breaks[i + 1])
            ax, az = spiral_point(ta, 0.0)
            bx, bz = spiral_point(tb, 0.0)
            L = math.hypot(float(bx - ax), float(bz - az))
            n = int(max(1.0, math.ceil(float(L) / float(max_step))))
            for k in range(1, n + 1):
                refined.append(ta + (tb - ta) * (float(k) / float(n)))
        t_breaks = refined

    def rasterize_arm(phase: float) -> set[tuple[int, int]]:
        pts: set[tuple[int, int]] = set()
        prev: Optional[tuple[int, int]] = None
        for tt in t_breaks:
            x, z = spiral_point(float(tt), float(phase))
            cur = (int(round(x)), int(round(z)))
            if prev is None:
                pts.add(cur)
                prev = cur
                continue
            for p in _bresenham(prev, cur):
                pts.add(p)
            prev = cur
        return pts

    out = rasterize_arm(0.0) | rasterize_arm(math.pi)
    return out


def _points_to_ascii(
    points: set[tuple[int, int]],
    *,
    pad: int,
    on: str,
    off: str,
) -> str:
    if not points:
        return ""

    xs = [p[0] for p in points]
    zs = [p[1] for p in points]

    min_x = min(xs) - int(pad)
    max_x = max(xs) + int(pad)
    min_z = min(zs) - int(pad)
    max_z = max(zs) + int(pad)

    lines: list[str] = []
    for z in range(int(max_z), int(min_z) - 1, -1):
        row = []
        for x in range(int(min_x), int(max_x) + 1):
            row.append(on if (int(x), int(z)) in points else off)
        lines.append("".join(row))

    return "\n".join(lines)


class GridViewport2D(QtWidgets.QWidget):
    def __init__(self, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)

        self._points: dict[tuple[int, int], QtGui.QColor] = {}
        self._zoom = 24.0
        self._pan = QtCore.QPointF(0.0, 0.0)
        self._last_mouse: Optional[QtCore.QPoint] = None
        self._show_grid = True
        self._show_axes = True

    def set_show_grid(self, show: bool) -> None:
        self._show_grid = bool(show)
        self.update()

    def set_show_axes(self, show: bool) -> None:
        self._show_axes = bool(show)
        self.update()

    def clear(self) -> None:
        self._points = {}
        self.update()

    def set_points(self, pts: dict[tuple[int, int], QtGui.QColor]) -> None:
        self._points = dict(pts)
        self.update()

    def fit_to_points(self) -> None:
        if not self._points:
            self._pan = QtCore.QPointF(0.0, 0.0)
            self._zoom = 24.0
            self.update()
            return

        xs = [p[0] for p in self._points.keys()]
        zs = [p[1] for p in self._points.keys()]
        min_x, max_x = min(xs), max(xs)
        min_z, max_z = min(zs), max(zs)

        w = max(1, self.width())
        h = max(1, self.height())

        span_x = max(1.0, float(max_x - min_x + 3))
        span_z = max(1.0, float(max_z - min_z + 3))

        cell_x = float(w) / span_x
        cell_z = float(h) / span_z
        self._zoom = max(6.0, min(80.0, min(cell_x, cell_z)))

        cx = (float(min_x) + float(max_x)) * 0.5
        cz = (float(min_z) + float(max_z)) * 0.5
        self._pan = QtCore.QPointF(-cx * self._zoom, cz * self._zoom)
        self.update()

    def wheelEvent(self, event: QtGui.QWheelEvent) -> None:
        delta = event.angleDelta().y() / 120.0
        if delta == 0.0:
            return

        before = float(self._zoom)
        self._zoom *= 1.15 ** float(delta)
        self._zoom = max(4.0, min(140.0, self._zoom))
        after = float(self._zoom)

        if abs(after - before) > 1e-9:
            p = event.position()
            cx = float(self.width()) * 0.5
            cy = float(self.height()) * 0.5
            sx = float(p.x()) - cx
            sy = float(p.y()) - cy
            if before > 1e-9:
                scale = after / before
                self._pan = QtCore.QPointF(float(self._pan.x()) * scale + sx * (1.0 - scale), float(self._pan.y()) * scale + sy * (1.0 - scale))
        self.update()

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            self._last_mouse = event.position().toPoint()
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            self._last_mouse = None
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        if self._last_mouse is not None:
            p = event.position().toPoint()
            dx = p.x() - self._last_mouse.x()
            dy = p.y() - self._last_mouse.y()
            self._last_mouse = p
            self._pan = QtCore.QPointF(float(self._pan.x()) + float(dx), float(self._pan.y()) + float(dy))
            self.update()
            event.accept()
            return
        super().mouseMoveEvent(event)

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, False)
        painter.fillRect(self.rect(), QtGui.QColor(6, 6, 8))

        w = max(1, self.width())
        h = max(1, self.height())
        cx = float(w) * 0.5
        cy = float(h) * 0.5

        def world_to_screen(x: float, z: float) -> QtCore.QPointF:
            sx = cx + float(self._pan.x()) + (x * float(self._zoom))
            sy = cy + float(self._pan.y()) - (z * float(self._zoom))
            return QtCore.QPointF(float(sx), float(sy))

        def screen_to_world(p: QtCore.QPointF) -> tuple[float, float]:
            x = (float(p.x()) - cx - float(self._pan.x())) / float(self._zoom)
            z = -(float(p.y()) - cy - float(self._pan.y())) / float(self._zoom)
            return float(x), float(z)

        if self._show_grid and float(self._zoom) >= 10.0:
            tl = screen_to_world(QtCore.QPointF(0.0, 0.0))
            br = screen_to_world(QtCore.QPointF(float(w), float(h)))
            min_x = int(math.floor(min(tl[0], br[0]))) - 1
            max_x = int(math.ceil(max(tl[0], br[0]))) + 1
            min_z = int(math.floor(min(tl[1], br[1]))) - 1
            max_z = int(math.ceil(max(tl[1], br[1]))) + 1

            grid_col = QtGui.QColor(25, 26, 30, 255)
            axis_col = QtGui.QColor(42, 62, 96, 255)

            pen = QtGui.QPen(grid_col)
            pen.setWidthF(1.0)
            painter.setPen(pen)
            z0 = float(min_z) - 0.5
            z1 = float(max_z) + 0.5
            x0 = float(min_x) - 0.5
            x1 = float(max_x) + 0.5
            for x in range(min_x, max_x + 2):
                gx = float(x) + 0.5
                p0 = world_to_screen(float(gx), float(z0))
                p1 = world_to_screen(float(gx), float(z1))
                painter.drawLine(p0, p1)
            for z in range(min_z, max_z + 2):
                gz = float(z) + 0.5
                p0 = world_to_screen(float(x0), float(gz))
                p1 = world_to_screen(float(x1), float(gz))
                painter.drawLine(p0, p1)

            if self._show_axes:
                pen2 = QtGui.QPen(axis_col)
                pen2.setWidthF(1.6)
                painter.setPen(pen2)
                painter.drawLine(world_to_screen(float(min_x), 0.0), world_to_screen(float(max_x), 0.0))
                painter.drawLine(world_to_screen(0.0, float(min_z)), world_to_screen(0.0, float(max_z)))

        if self._points:
            painter.setPen(QtCore.Qt.PenStyle.NoPen)
            cell = float(self._zoom)
            pad = 0.9 if cell >= 10.0 else 0.0
            s = max(1.0, cell - pad)

            for (x, z), col in self._points.items():
                p = world_to_screen(float(x), float(z))
                rect = QtCore.QRectF(float(p.x() - s * 0.5), float(p.y() - s * 0.5), float(s), float(s))
                painter.setBrush(QtGui.QBrush(col))
                painter.drawRect(rect)

        painter.end()


class BuildDesignPlannerMainWindow(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Minecraft Build Design Planner")

        self._viewport = GridViewport2D()

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

        self._current_points: dict[tuple[int, int], QtGui.QColor] = {}

        self._sync_enable_sections()

    def _build_controls(self) -> QtWidgets.QWidget:
        w = QtWidgets.QWidget()
        w.setMinimumWidth(360)

        self._center_xz = QtWidgets.QLineEdit("0,0")
        self._y_level = QtWidgets.QSpinBox()
        self._y_level.setRange(-2048, 2048)
        self._y_level.setValue(0)

        self._show_grid = QtWidgets.QCheckBox("Grid")
        self._show_grid.setChecked(True)
        self._show_axes = QtWidgets.QCheckBox("Axes")
        self._show_axes.setChecked(True)

        self._show_grid.stateChanged.connect(lambda: self._viewport.set_show_grid(bool(self._show_grid.isChecked())))
        self._show_axes.stateChanged.connect(lambda: self._viewport.set_show_axes(bool(self._show_axes.isChecked())))

        self._btn_generate = QtWidgets.QPushButton("Generate")
        self._btn_generate.setObjectName("btn_solve")
        self._btn_generate.clicked.connect(self._on_generate)

        self._btn_clear = QtWidgets.QPushButton("Clear")
        self._btn_clear.setObjectName("btn_clear")
        self._btn_clear.clicked.connect(self._on_clear)

        self._btn_fit = QtWidgets.QPushButton("Fit")
        self._btn_fit.clicked.connect(self._viewport.fit_to_points)

        self._btn_save = QtWidgets.QPushButton("Save coords")
        self._btn_save.setObjectName("btn_save")
        self._btn_save.clicked.connect(self._on_save_coords)

        self._circle_enable = QtWidgets.QCheckBox("Enable circle")
        self._circle_enable.setChecked(True)
        self._circle_enable.stateChanged.connect(self._sync_enable_sections)

        self._circle_radius = QtWidgets.QSpinBox()
        self._circle_radius.setRange(1, 2048)
        self._circle_radius.setValue(12)

        self._circle_filled = QtWidgets.QCheckBox("Filled")
        self._circle_filled.setChecked(False)

        self._circle_thickness = QtWidgets.QSpinBox()
        self._circle_thickness.setRange(1, 512)
        self._circle_thickness.setValue(1)

        self._circle_metric = QtWidgets.QComboBox()
        self._circle_metric.addItem("Euclidean", userData="euclidean")
        self._circle_metric.addItem("Chebyshev (square)", userData="chebyshev")
        self._circle_metric.addItem("Manhattan (diamond)", userData="manhattan")
        self._circle_metric.setCurrentIndex(0)

        self._spiral_enable = QtWidgets.QCheckBox("Enable double spiral")
        self._spiral_enable.setChecked(True)
        self._spiral_enable.stateChanged.connect(self._sync_enable_sections)

        self._spiral_radius_limit = QtWidgets.QSpinBox()
        self._spiral_radius_limit.setRange(1, 4096)
        self._spiral_radius_limit.setValue(12)

        self._spiral_spacing = QtWidgets.QDoubleSpinBox()
        self._spiral_spacing.setRange(0.05, 128.0)
        self._spiral_spacing.setDecimals(3)
        self._spiral_spacing.setValue(8.0)

        self._spiral_guide_rotation = QtWidgets.QDoubleSpinBox()
        self._spiral_guide_rotation.setRange(-360.0, 360.0)
        self._spiral_guide_rotation.setDecimals(2)
        self._spiral_guide_rotation.setValue(0.0)

        self._spiral_segments = QtWidgets.QSpinBox()
        self._spiral_segments.setRange(8, 8192)
        self._spiral_segments.setValue(128)

        self._spiral_max_chord_step = QtWidgets.QDoubleSpinBox()
        self._spiral_max_chord_step.setRange(0.0, 64.0)
        self._spiral_max_chord_step.setDecimals(3)
        self._spiral_max_chord_step.setValue(0.75)

        self._spiral_thickness = QtWidgets.QSpinBox()
        self._spiral_thickness.setRange(1, 512)
        self._spiral_thickness.setValue(1)

        self._spiral_metric = QtWidgets.QComboBox()
        self._spiral_metric.addItem("Euclidean", userData="euclidean")
        self._spiral_metric.addItem("Chebyshev (square)", userData="chebyshev")
        self._spiral_metric.addItem("Manhattan (diamond)", userData="manhattan")
        self._spiral_metric.setCurrentIndex(0)

        self._lbl_status = QtWidgets.QLabel("Ready")
        self._lbl_status.setWordWrap(True)

        scene_box = QtWidgets.QGroupBox("Scene")
        scene_form = QtWidgets.QFormLayout(scene_box)
        scene_form.setContentsMargins(10, 10, 10, 10)
        scene_form.setVerticalSpacing(4)
        scene_form.addRow("Center XZ", self._center_xz)
        scene_form.addRow("Y level", self._y_level)

        vp_box = QtWidgets.QGroupBox("Viewport")
        vp_layout = QtWidgets.QVBoxLayout(vp_box)
        vp_layout.setContentsMargins(10, 10, 10, 10)
        vp_layout.setSpacing(4)
        vp_layout.addWidget(self._show_grid)
        vp_layout.addWidget(self._show_axes)
        vp_layout.addWidget(self._btn_fit)

        circle_box = QtWidgets.QGroupBox("Circle")
        circle_form = QtWidgets.QFormLayout(circle_box)
        circle_form.setContentsMargins(10, 10, 10, 10)
        circle_form.setVerticalSpacing(4)
        circle_form.addRow(self._circle_enable)
        circle_form.addRow("Radius", self._circle_radius)
        circle_form.addRow("Thickness", self._circle_thickness)
        circle_form.addRow("Thickness metric", self._circle_metric)
        circle_form.addRow("", self._circle_filled)

        spiral_box = QtWidgets.QGroupBox("Double spiral")
        spiral_form = QtWidgets.QFormLayout(spiral_box)
        spiral_form.setContentsMargins(10, 10, 10, 10)
        spiral_form.setVerticalSpacing(4)
        spiral_form.addRow(self._spiral_enable)
        spiral_form.addRow("Radius limit", self._spiral_radius_limit)
        spiral_form.addRow("Spacing", self._spiral_spacing)
        spiral_form.addRow("Rotation (deg)", self._spiral_guide_rotation)
        spiral_form.addRow("Segments", self._spiral_segments)
        spiral_form.addRow("Max chord step", self._spiral_max_chord_step)
        spiral_form.addRow("Thickness", self._spiral_thickness)
        spiral_form.addRow("Thickness metric", self._spiral_metric)

        btn_row = QtWidgets.QHBoxLayout()
        btn_row.addWidget(self._btn_generate)
        btn_row.addWidget(self._btn_clear)

        btn_row2 = QtWidgets.QHBoxLayout()
        btn_row2.addWidget(self._btn_save)

        layout = QtWidgets.QVBoxLayout(w)
        layout.addWidget(scene_box)
        layout.addWidget(circle_box)
        layout.addWidget(spiral_box)
        layout.addWidget(vp_box)
        layout.addLayout(btn_row)
        layout.addLayout(btn_row2)
        layout.addWidget(self._lbl_status)
        layout.addStretch(1)
        return w

    def _build_output(self) -> QtWidgets.QWidget:
        w = QtWidgets.QWidget()
        w.setMinimumWidth(460)
        w.setSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred, QtWidgets.QSizePolicy.Policy.Expanding)

        self._txt_coords = QtWidgets.QPlainTextEdit()
        self._txt_coords.setReadOnly(True)
        self._txt_coords.setMaximumBlockCount(20000)
        self._txt_coords.setLineWrapMode(QtWidgets.QPlainTextEdit.LineWrapMode.NoWrap)
        self._txt_coords.setFont(QtGui.QFont("Consolas", 10))

        self._txt_ascii = QtWidgets.QPlainTextEdit()
        self._txt_ascii.setReadOnly(True)
        self._txt_ascii.setMaximumBlockCount(20000)
        self._txt_ascii.setLineWrapMode(QtWidgets.QPlainTextEdit.LineWrapMode.NoWrap)
        self._txt_ascii.setFont(QtGui.QFont("Consolas", 10))

        tabs = QtWidgets.QTabWidget()
        tabs.addTab(self._txt_coords, "Coords")
        tabs.addTab(self._txt_ascii, "ASCII")

        box = QtWidgets.QGroupBox("Output")
        box_layout = QtWidgets.QVBoxLayout(box)
        box_layout.setContentsMargins(10, 10, 10, 10)
        box_layout.addWidget(tabs)

        layout = QtWidgets.QVBoxLayout(w)
        layout.addWidget(box)
        layout.setContentsMargins(0, 0, 0, 0)
        return w

    def _sync_enable_sections(self) -> None:
        circle_on = bool(self._circle_enable.isChecked())
        self._circle_radius.setEnabled(circle_on)
        self._circle_filled.setEnabled(circle_on)
        self._circle_thickness.setEnabled(circle_on)
        self._circle_metric.setEnabled(circle_on)

        spiral_on = bool(self._spiral_enable.isChecked())
        self._spiral_radius_limit.setEnabled(spiral_on)
        self._spiral_spacing.setEnabled(spiral_on)
        self._spiral_guide_rotation.setEnabled(spiral_on)
        self._spiral_segments.setEnabled(spiral_on)
        self._spiral_max_chord_step.setEnabled(spiral_on)
        self._spiral_thickness.setEnabled(spiral_on)
        self._spiral_metric.setEnabled(spiral_on)

    def _compose_points(self) -> tuple[dict[tuple[int, int], QtGui.QColor], set[tuple[int, int]]]:
        center = _parse_vec2(self._center_xz.text().strip())

        pts_colored: dict[tuple[int, int], QtGui.QColor] = {}
        pts_all: set[tuple[int, int]] = set()

        if bool(self._circle_enable.isChecked()):
            radius = float(self._circle_radius.value())
            filled = bool(self._circle_filled.isChecked())
            p = _gen_circle_points(center=center, radius=radius, filled=filled)

            thick = int(self._circle_thickness.value())
            metric = str(self._circle_metric.currentData() or "euclidean")
            if thick <= 1 and metric == "chebyshev":
                p = _corner_fill(p)
            if thick > 1:
                rule = ThicknessRule(metric=metric, radius=int(thick - 1))
                p = _dilate(p, rule=rule)

            for q in p:
                pts_colored[q] = QtGui.QColor(95, 170, 255, 215)
            pts_all |= p

        if bool(self._spiral_enable.isChecked()):
            rlim = float(self._spiral_radius_limit.value())
            spacing = float(self._spiral_spacing.value())
            guide_rotation = float(self._spiral_guide_rotation.value())
            segments = int(self._spiral_segments.value())
            max_chord_step = float(self._spiral_max_chord_step.value())
            p = _gen_double_spiral_points(
                center=center,
                radius_limit=rlim,
                spacing=spacing,
                guide_rotation_deg=guide_rotation,
                segments=segments,
                max_chord_step=max_chord_step,
            )

            thick = int(self._spiral_thickness.value())
            metric = str(self._spiral_metric.currentData() or "euclidean")
            if thick <= 1 and metric == "chebyshev":
                p = _corner_fill(p)
            if thick > 1:
                rule = ThicknessRule(metric=metric, radius=int(thick - 1))
                p = _dilate(p, rule=rule)

            for q in p:
                if q in pts_colored:
                    pts_colored[q] = QtGui.QColor(185, 210, 120, 225)
                else:
                    pts_colored[q] = QtGui.QColor(85, 210, 120, 215)
            pts_all |= p

        return pts_colored, pts_all

    def _on_generate(self) -> None:
        try:
            pts_colored, pts_all = self._compose_points()
            self._current_points = dict(pts_colored)

            self._viewport.set_points(self._current_points)

            y = int(self._y_level.value())
            coords = sorted(list(pts_all), key=lambda p: (int(p[1]), int(p[0])))

            lines = [f"{int(x)} {int(y)} {int(z)}" for x, z in coords]
            self._txt_coords.setPlainText("\n".join(lines))

            ascii_map = _points_to_ascii(pts_all, pad=1, on="#", off=".")
            self._txt_ascii.setPlainText(ascii_map)

            self._lbl_status.setText(f"Blocks: {len(coords)}")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Generate failed", str(e))

    def _on_clear(self) -> None:
        self._current_points = {}
        self._viewport.clear()
        self._txt_coords.setPlainText("")
        self._txt_ascii.setPlainText("")
        self._lbl_status.setText("Cleared")

    def _on_save_coords(self) -> None:
        if not self._txt_coords.toPlainText().strip():
            return

        default = (Path.cwd() / "build_coords.txt").resolve()
        out_path, _filter = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save coordinates",
            str(default),
            "Text (*.txt)",
        )
        if not out_path:
            return

        try:
            Path(out_path).write_text(self._txt_coords.toPlainText().rstrip() + "\n", encoding="utf-8", newline="\n")
            self._lbl_status.setText(f"Saved: {out_path}")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Save failed", str(e))


def main() -> None:
    app = QtWidgets.QApplication(sys.argv)
    _apply_dark_theme(app)
    w = BuildDesignPlannerMainWindow()
    w.resize(1280, 820)
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
