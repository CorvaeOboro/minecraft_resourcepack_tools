"""
Minecraft Build Design Planner (UI)

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


# region SETUP
# PySide6 bootstrap, optional dark-theme import from the solver UI module


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


# endregion SETUP
# region DATA
# ThicknessRule dataclass and generation-guard constants shared across
# the helper, shape, and UI layers


# Maximum structuring-element radius used by the thickness brush.  This caps
# dilation cost and prevents the UI from hanging when a very large thickness
# is requested.  The thickness spinboxes expose ``_MAX_THICKNESS_RADIUS + 1``
# so the largest selectable brush is a 64-block radius.
_MAX_THICKNESS_RADIUS = 63

# Upper bound on the number of points generated for a double spiral.  This
# keeps the tool responsive and avoids overflowing the coordinate output.
_MAX_SPIRAL_POINTS = 100_000


@dataclass(frozen=True)
class ThicknessRule:
    metric: str
    radius: int


# endregion DATA
# region HELPERS
# Math utilities: lerp, Bresenham line rasterization, vec2 parsing,
# metric structuring elements, Minkowski dilation, corner fill, and
# the unified thickness-brush dispatcher


def _lerp(value_start: float, value_end: float, t: float) -> float:
    t_clamped = 0.0 if t < 0.0 else 1.0 if t > 1.0 else float(t)
    return float(value_start) * (1.0 - t_clamped) + float(value_end) * t_clamped


def _bresenham(point_start: tuple[int, int], point_end: tuple[int, int]) -> list[tuple[int, int]]:
    x_start, y_start = int(point_start[0]), int(point_start[1])
    x_end, y_end = int(point_end[0]), int(point_end[1])
    delta_x = abs(x_end - x_start)
    delta_y = abs(y_end - y_start)
    step_x = 1 if x_start < x_end else -1
    step_y = 1 if y_start < y_end else -1
    error = delta_x - delta_y
    rasterized_points: list[tuple[int, int]] = []
    while True:
        rasterized_points.append((int(x_start), int(y_start)))
        if x_start == x_end and y_start == y_end:
            break
        error2 = 2 * error
        if error2 > -delta_y:
            error -= delta_y
            x_start += step_x
        if error2 < delta_x:
            error += delta_x
            y_start += step_y
    return rasterized_points


def _parse_vec2(raw_text: str) -> tuple[int, int]:
    """Parse a "x,z" string into integer block coordinates.

    Floating-point values are accepted only when they are numerically equal
    to an integer (e.g. "12.0,-3.0"); otherwise a ``ValueError`` is raised so
    the caller can notify the user instead of silently rounding.
    """
    parts_str = [p.strip() for p in raw_text.split(",")]
    if len(parts_str) != 2:
        raise ValueError("Expected x,z")

    parsed_values = []
    for part_str in parts_str:
        value_float = float(part_str)
        value_rounded = round(value_float)
        if not math.isclose(value_float, value_rounded, abs_tol=1e-6):
            raise ValueError(f"Center coordinate {part_str!r} is not an integer")
        parsed_values.append(int(value_rounded))
    return parsed_values[0], parsed_values[1]


def _neighbors_within(rule: ThicknessRule) -> list[tuple[int, int]]:
    radius = int(rule.radius)
    if radius <= 0:
        return [(0, 0)]

    offsets: list[tuple[int, int]] = []
    for dz in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            abs_dx = abs(int(dx))
            abs_dz = abs(int(dz))
            if rule.metric == "manhattan":
                if (abs_dx + abs_dz) <= radius:
                    offsets.append((dx, dz))
            elif rule.metric == "chebyshev":
                if max(abs_dx, abs_dz) <= radius:
                    offsets.append((dx, dz))
            else:
                if (dx * dx + dz * dz) <= (radius * radius):
                    offsets.append((dx, dz))
    return offsets


def _dilate(points: set[tuple[int, int]], *, rule: ThicknessRule) -> set[tuple[int, int]]:
    """Minkowski sum of ``points`` with the structuring element from ``rule``.

    The cost is ``|points| * |offsets|`` so very large radii are rejected to
    keep the tool responsive.
    """
    radius = int(rule.radius)
    if radius > _MAX_THICKNESS_RADIUS:
        raise ValueError(
            f"Thickness radius {radius} exceeds maximum {_MAX_THICKNESS_RADIUS}; "
            "reduce the thickness value."
        )

    offsets = _neighbors_within(rule)
    dilated_points: set[tuple[int, int]] = set()
    for x, z in points:
        for dx, dz in offsets:
            dilated_points.add((int(x) + int(dx), int(z) + int(dz)))
    return dilated_points


def _corner_fill(points: set[tuple[int, int]]) -> set[tuple[int, int]]:
    if not points:
        return set()

    filled_points = set(points)
    diagonal_offsets = ((1, 1), (1, -1), (-1, 1), (-1, -1))
    for x, z in points:
        for dx, dz in diagonal_offsets:
            if (int(x) + int(dx), int(z) + int(dz)) in points:
                filled_points.add((int(x) + int(dx), int(z)))
                filled_points.add((int(x), int(z) + int(dz)))
    return filled_points


def _apply_thickness(
    points: set[tuple[int, int]], *, metric: str, thick: int
) -> set[tuple[int, int]]:
    """Apply a metric brush of ``thick`` blocks to ``points``.

    The thickness value follows the convention ``radius = thick - 1`` so
    that a thickness of 1 leaves the rasterized line exactly one block wide.
    Chebyshev metric with thickness 1 also applies corner-filling so that
    diagonal steps of the rasterized curve are connected under the square
    (Chebyshev) metric.
    """
    brushed_points = set(points)
    if metric == "chebyshev" and thick == 1:
        brushed_points = _corner_fill(brushed_points)
    elif thick > 1:
        rule = ThicknessRule(metric=metric, radius=int(thick - 1))
        brushed_points = _dilate(brushed_points, rule=rule)
    return brushed_points


# endregion HELPERS
# region SHAPES
# 2D integer rasterizers: scanline circle fill and double Archimedean spiral


def _gen_circle_points(
    *,
    center: tuple[int, int],
    radius: float,
    filled: bool,
) -> set[tuple[int, int]]:
    """Generate pixelated integer circle points using a scanline fill.

    For ``filled=True`` every integer grid point whose distance from the
    centre is at most ``radius + 0.5`` is included.  For ``filled=False``
    only the ring satisfying ``abs(distance - radius) <= 0.5`` is included.

    The scanline approach walks only rows that can intersect the shape,
    avoiding the full square bounding-box scan used by an earlier version.
    """
    cx, cz = center
    radius_float = float(radius)
    if radius_float <= 0.0:
        return set()

    circle_points: set[tuple[int, int]] = set()
    scan_radius = int(math.ceil(radius_float + 0.5))

    radius_outer = radius_float + 0.5
    radius_outer_sq = radius_outer * radius_outer

    if filled:
        radius_inner_sq = -1.0
    else:
        radius_inner = max(0.0, radius_float - 0.5)
        radius_inner_sq = radius_inner * radius_inner

    for z in range(int(cz) - scan_radius, int(cz) + scan_radius + 1):
        dz = float(z) - float(cz)
        dz_sq = dz * dz
        if dz_sq > radius_outer_sq:
            continue

        half_width = math.sqrt(radius_outer_sq - dz_sq)
        x_start = int(math.ceil(float(cx) - half_width))
        x_end = int(math.floor(float(cx) + half_width))

        for x in range(x_start, x_end + 1):
            dx = float(x) - float(cx)
            dist_sq = dx * dx + dz_sq
            if radius_inner_sq <= dist_sq <= radius_outer_sq:
                circle_points.add((x, z))

    return circle_points


def _gen_double_spiral_points(
    *,
    center: tuple[int, int],
    radius_limit: float,
    spacing: float,
    guide_rotation_deg: float,
    segments: int,
    max_chord_step: float,
) -> set[tuple[int, int]]:
    """Generate a rasterized double Archimedean spiral.

    Two arms are drawn 180 degrees apart.  The spiral is refined until the
    chord between consecutive samples is no larger than ``max_chord_step``.
    Pathological settings (extremely small spacing or huge radius limit) are
    rejected before generation so the tool cannot hang.
    """
    cx, cz = center
    radius_limit_float = float(radius_limit)
    if radius_limit_float <= 0.0:
        return set()

    spacing_float = float(spacing)
    if spacing_float <= 1e-9:
        spacing_float = 1.0

    turns = float(radius_limit_float) / float(spacing_float)
    turns = max(0.0, float(turns))
    if turns > 1000.0:
        raise ValueError(
            f"Spiral too dense ({turns:.0f} turns). "
            "Increase the spacing or reduce the radius limit."
        )

    segment_count = max(8, int(segments))
    # Ensure the base sampling is fine enough relative to the number of turns
    # so that chord refinement does not explode.
    segment_count = max(segment_count, min(8192, int(math.ceil(turns * 4.0))))
    segment_count = min(8192, segment_count)

    max_chord_step_float = float(max_chord_step)
    max_chord_step_float = 0.0 if max_chord_step_float < 0.0 else float(max_chord_step_float)

    if max_chord_step_float > 1e-9:
        # Estimate total arc length as ~ pi * radius_limit * turns.  The
        # number of refined samples is roughly that divided by max_chord_step.
        estimated_points = int(math.ceil(math.pi * radius_limit_float * turns / max_chord_step_float))
        if estimated_points > _MAX_SPIRAL_POINTS:
            raise ValueError(
                f"Estimated spiral points ({estimated_points:,}) exceeds the "
                f"limit ({_MAX_SPIRAL_POINTS:,}). Increase the max chord step, "
                "increase the spacing, or reduce the radius limit."
            )

    guide_rotation_rad = math.radians(float(guide_rotation_deg))

    def spiral_point(t_normalized: float, phase: float) -> tuple[float, float]:
        angle_rad = (2.0 * math.pi) * float(turns) * float(t_normalized) + float(phase) + float(guide_rotation_rad)
        spiral_radius = _lerp(0.0, float(radius_limit_float), float(t_normalized))
        return (float(cx) + spiral_radius * math.cos(angle_rad), float(cz) + spiral_radius * math.sin(angle_rad))

    t_samples: list[float] = []
    for i in range(segment_count + 1):
        t_samples.append(float(i) / float(segment_count))

    if max_chord_step_float > 1e-9:
        refined_samples: list[float] = [float(t_samples[0])]
        for i in range(len(t_samples) - 1):
            t_start = float(t_samples[i])
            t_end = float(t_samples[i + 1])
            point_a_x, point_a_z = spiral_point(t_start, 0.0)
            point_b_x, point_b_z = spiral_point(t_end, 0.0)
            chord_length = math.hypot(float(point_b_x - point_a_x), float(point_b_z - point_a_z))
            subdivisions = int(max(1.0, math.ceil(float(chord_length) / float(max_chord_step_float))))
            for subdiv_index in range(1, subdivisions + 1):
                refined_samples.append(t_start + (t_end - t_start) * (float(subdiv_index) / float(subdivisions)))
        t_samples = refined_samples

    def rasterize_arm(phase: float) -> set[tuple[int, int]]:
        arm_points: set[tuple[int, int]] = set()
        first_t = float(t_samples[0])
        x, z = spiral_point(first_t, float(phase))
        prev_point: tuple[int, int] = (int(round(x)), int(round(z)))
        arm_points.add(prev_point)
        for t_normalized in t_samples[1:]:
            x, z = spiral_point(float(t_normalized), float(phase))
            current_point = (int(round(x)), int(round(z)))
            if current_point == prev_point:
                continue
            for line_point in _bresenham(prev_point, current_point):
                arm_points.add(line_point)
            prev_point = current_point
        return arm_points

    spiral_points = rasterize_arm(0.0) | rasterize_arm(math.pi)
    return spiral_points


# endregion SHAPES
# region ASCII
# Block-character grid renderer for the coordinate output tab


def _points_to_ascii(
    points: set[tuple[int, int]],
    *,
    pad: int,
    on: str,
    off: str,
) -> str:
    """Render ``points`` as a block-character grid.

    The output is oriented so that the positive Z axis points up and the
    positive X axis points right.  This matches the in-game Minecraft XZ plane
    where +Z is south and +X is east.
    """
    if not points:
        return ""

    x_coords = [p[0] for p in points]
    z_coords = [p[1] for p in points]

    min_x = min(x_coords) - int(pad)
    max_x = max(x_coords) + int(pad)
    min_z = min(z_coords) - int(pad)
    max_z = max(z_coords) + int(pad)

    lines: list[str] = []
    for z in range(int(max_z), int(min_z) - 1, -1):
        row = []
        for x in range(int(min_x), int(max_x) + 1):
            row.append(on if (int(x), int(z)) in points else off)
        lines.append("".join(row))

    return "\n".join(lines)


# endregion ASCII
# region VIEWPORT
# 2D pan/zoom grid viewport with world-to-screen projection, grid lines,
# axis overlays, and colored block-cell rendering


class GridViewport2D(QtWidgets.QWidget):
    def __init__(self, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)

        self._points_colored: dict[tuple[int, int], QtGui.QColor] = {}
        self._zoom_factor = 24.0
        self._pan_offset = QtCore.QPointF(0.0, 0.0)
        self._last_mouse_pos: Optional[QtCore.QPoint] = None
        self._show_grid = True
        self._show_axes = True

    def set_show_grid(self, show: bool) -> None:
        self._show_grid = bool(show)
        self.update()

    def set_show_axes(self, show: bool) -> None:
        self._show_axes = bool(show)
        self.update()

    def clear(self) -> None:
        self._points_colored = {}
        self.update()

    def set_points(self, pts: dict[tuple[int, int], QtGui.QColor]) -> None:
        self._points_colored = dict(pts)
        self.update()

    def fit_to_points(self) -> None:
        if not self._points_colored:
            self._pan_offset = QtCore.QPointF(0.0, 0.0)
            self._zoom_factor = 24.0
            self.update()
            return

        x_coords = [p[0] for p in self._points_colored.keys()]
        z_coords = [p[1] for p in self._points_colored.keys()]
        min_x, max_x = min(x_coords), max(x_coords)
        min_z, max_z = min(z_coords), max(z_coords)

        width_px = max(1, self.width())
        height_px = max(1, self.height())

        span_x = max(1.0, float(max_x - min_x + 3))
        span_z = max(1.0, float(max_z - min_z + 3))

        cell_size_x = float(width_px) / span_x
        cell_size_z = float(height_px) / span_z
        self._zoom_factor = max(6.0, min(80.0, min(cell_size_x, cell_size_z)))

        center_x = (float(min_x) + float(max_x)) * 0.5
        center_z = (float(min_z) + float(max_z)) * 0.5
        self._pan_offset = QtCore.QPointF(-center_x * self._zoom_factor, center_z * self._zoom_factor)
        self.update()

    def wheelEvent(self, event: QtGui.QWheelEvent) -> None:
        wheel_delta = event.angleDelta().y() / 120.0
        if wheel_delta == 0.0:
            return

        zoom_before = float(self._zoom_factor)
        self._zoom_factor *= 1.15 ** float(wheel_delta)
        self._zoom_factor = max(4.0, min(140.0, self._zoom_factor))
        zoom_after = float(self._zoom_factor)

        if abs(zoom_after - zoom_before) > 1e-9:
            mouse_pos = event.position()
            screen_center_x = float(self.width()) * 0.5
            screen_center_y = float(self.height()) * 0.5
            mouse_offset_x = float(mouse_pos.x()) - screen_center_x
            mouse_offset_y = float(mouse_pos.y()) - screen_center_y
            if zoom_before > 1e-9:
                zoom_scale = zoom_after / zoom_before
                self._pan_offset = QtCore.QPointF(
                    float(self._pan_offset.x()) * zoom_scale + mouse_offset_x * (1.0 - zoom_scale),
                    float(self._pan_offset.y()) * zoom_scale + mouse_offset_y * (1.0 - zoom_scale),
                )
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
            mouse_pos = event.position().toPoint()
            delta_x = mouse_pos.x() - self._last_mouse_pos.x()
            delta_y = mouse_pos.y() - self._last_mouse_pos.y()
            self._last_mouse_pos = mouse_pos
            self._pan_offset = QtCore.QPointF(
                float(self._pan_offset.x()) + float(delta_x),
                float(self._pan_offset.y()) + float(delta_y),
            )
            self.update()
            event.accept()
            return
        super().mouseMoveEvent(event)

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, False)
        painter.fillRect(self.rect(), QtGui.QColor(6, 6, 8))

        width_px = max(1, self.width())
        height_px = max(1, self.height())
        screen_center_x = float(width_px) * 0.5
        screen_center_y = float(height_px) * 0.5

        def world_to_screen(x: float, z: float) -> QtCore.QPointF:
            screen_x = screen_center_x + float(self._pan_offset.x()) + (x * float(self._zoom_factor))
            screen_y = screen_center_y + float(self._pan_offset.y()) - (z * float(self._zoom_factor))
            return QtCore.QPointF(float(screen_x), float(screen_y))

        def screen_to_world(screen_point: QtCore.QPointF) -> tuple[float, float]:
            x = (float(screen_point.x()) - screen_center_x - float(self._pan_offset.x())) / float(self._zoom_factor)
            z = -(float(screen_point.y()) - screen_center_y - float(self._pan_offset.y())) / float(self._zoom_factor)
            return float(x), float(z)

        if self._show_grid and float(self._zoom_factor) >= 10.0:
            top_left_world = screen_to_world(QtCore.QPointF(0.0, 0.0))
            bottom_right_world = screen_to_world(QtCore.QPointF(float(width_px), float(height_px)))
            min_x = int(math.floor(min(top_left_world[0], bottom_right_world[0]))) - 1
            max_x = int(math.ceil(max(top_left_world[0], bottom_right_world[0]))) + 1
            min_z = int(math.floor(min(top_left_world[1], bottom_right_world[1]))) - 1
            max_z = int(math.ceil(max(top_left_world[1], bottom_right_world[1]))) + 1

            grid_color = QtGui.QColor(25, 26, 30, 255)
            axis_color = QtGui.QColor(42, 62, 96, 255)

            grid_pen = QtGui.QPen(grid_color)
            grid_pen.setWidthF(1.0)
            painter.setPen(grid_pen)
            grid_z_min = float(min_z) - 0.5
            grid_z_max = float(max_z) + 0.5
            grid_x_min = float(min_x) - 0.5
            grid_x_max = float(max_x) + 0.5
            for x in range(min_x, max_x + 2):
                grid_x = float(x) + 0.5
                line_start = world_to_screen(float(grid_x), float(grid_z_min))
                line_end = world_to_screen(float(grid_x), float(grid_z_max))
                painter.drawLine(line_start, line_end)
            for z in range(min_z, max_z + 2):
                grid_z = float(z) + 0.5
                line_start = world_to_screen(float(grid_x_min), float(grid_z))
                line_end = world_to_screen(float(grid_x_max), float(grid_z))
                painter.drawLine(line_start, line_end)

            if self._show_axes:
                axis_pen = QtGui.QPen(axis_color)
                axis_pen.setWidthF(1.6)
                painter.setPen(axis_pen)
                painter.drawLine(world_to_screen(float(min_x), 0.0), world_to_screen(float(max_x), 0.0))
                painter.drawLine(world_to_screen(0.0, float(min_z)), world_to_screen(0.0, float(max_z)))

        if self._points_colored:
            painter.setPen(QtCore.Qt.PenStyle.NoPen)
            cell_size = float(self._zoom_factor)
            cell_pad = 0.9 if cell_size >= 10.0 else 0.0
            cell_size_drawn = max(1.0, cell_size - cell_pad)

            for (x, z), point_color in self._points_colored.items():
                screen_pos = world_to_screen(float(x), float(z))
                cell_rect = QtCore.QRectF(
                    float(screen_pos.x() - cell_size_drawn * 0.5),
                    float(screen_pos.y() - cell_size_drawn * 0.5),
                    float(cell_size_drawn),
                    float(cell_size_drawn),
                )
                painter.setBrush(QtGui.QBrush(point_color))
                painter.drawRect(cell_rect)

        painter.end()


# endregion VIEWPORT
# region MAINWIN
# BuildDesignPlannerMainWindow: scene/circle/spiral controls, output tabs,
# point composition, generate/clear/save handlers, and section enable-sync


class BuildDesignPlannerMainWindow(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Minecraft Build Design Planner")

        self._viewport = GridViewport2D()

        controls_widget = self._build_controls()
        output_widget = self._build_output()

        splitter = QtWidgets.QSplitter()
        splitter.setOrientation(QtCore.Qt.Orientation.Horizontal)
        splitter.addWidget(controls_widget)
        splitter.addWidget(self._viewport)
        splitter.addWidget(output_widget)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 0)
        splitter.setCollapsible(0, False)
        splitter.setCollapsible(2, False)

        self.setCentralWidget(splitter)

        self._current_points: dict[tuple[int, int], QtGui.QColor] = {}

        self._sync_enable_sections()

    def _build_controls(self) -> QtWidgets.QWidget:
        controls_widget = QtWidgets.QWidget()
        controls_widget.setMinimumWidth(360)

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
        self._circle_thickness.setRange(1, _MAX_THICKNESS_RADIUS + 1)
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
        self._spiral_thickness.setRange(1, _MAX_THICKNESS_RADIUS + 1)
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

        buttons_row = QtWidgets.QHBoxLayout()
        buttons_row.addWidget(self._btn_generate)
        buttons_row.addWidget(self._btn_clear)

        save_button_row = QtWidgets.QHBoxLayout()
        save_button_row.addWidget(self._btn_save)

        controls_layout = QtWidgets.QVBoxLayout(controls_widget)
        controls_layout.addWidget(scene_box)
        controls_layout.addWidget(circle_box)
        controls_layout.addWidget(spiral_box)
        controls_layout.addWidget(vp_box)
        controls_layout.addLayout(buttons_row)
        controls_layout.addLayout(save_button_row)
        controls_layout.addWidget(self._lbl_status)
        controls_layout.addStretch(1)
        return controls_widget

    def _build_output(self) -> QtWidgets.QWidget:
        output_widget = QtWidgets.QWidget()
        output_widget.setMinimumWidth(460)
        output_widget.setSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred, QtWidgets.QSizePolicy.Policy.Expanding)

        self._txt_coords = QtWidgets.QPlainTextEdit()
        self._txt_coords.setReadOnly(True)
        self._txt_coords.setMaximumBlockCount(100000)
        self._txt_coords.setLineWrapMode(QtWidgets.QPlainTextEdit.LineWrapMode.NoWrap)
        self._txt_coords.setFont(QtGui.QFont("Consolas", 10))

        self._txt_ascii = QtWidgets.QPlainTextEdit()
        self._txt_ascii.setReadOnly(True)
        self._txt_ascii.setMaximumBlockCount(100000)
        self._txt_ascii.setLineWrapMode(QtWidgets.QPlainTextEdit.LineWrapMode.NoWrap)
        self._txt_ascii.setFont(QtGui.QFont("Consolas", 10))

        output_tabs = QtWidgets.QTabWidget()
        output_tabs.addTab(self._txt_coords, "Coords")
        output_tabs.addTab(self._txt_ascii, "ASCII")

        output_box = QtWidgets.QGroupBox("Output")
        output_box_layout = QtWidgets.QVBoxLayout(output_box)
        output_box_layout.setContentsMargins(10, 10, 10, 10)
        output_box_layout.addWidget(output_tabs)

        output_layout = QtWidgets.QVBoxLayout(output_widget)
        output_layout.addWidget(output_box)
        output_layout.setContentsMargins(0, 0, 0, 0)
        return output_widget

    def _sync_enable_sections(self) -> None:
        circle_enabled = bool(self._circle_enable.isChecked())
        self._circle_radius.setEnabled(circle_enabled)
        self._circle_filled.setEnabled(circle_enabled)
        self._circle_thickness.setEnabled(circle_enabled)
        self._circle_metric.setEnabled(circle_enabled)

        spiral_enabled = bool(self._spiral_enable.isChecked())
        self._spiral_radius_limit.setEnabled(spiral_enabled)
        self._spiral_spacing.setEnabled(spiral_enabled)
        self._spiral_guide_rotation.setEnabled(spiral_enabled)
        self._spiral_segments.setEnabled(spiral_enabled)
        self._spiral_max_chord_step.setEnabled(spiral_enabled)
        self._spiral_thickness.setEnabled(spiral_enabled)
        self._spiral_metric.setEnabled(spiral_enabled)

    def _compose_points(self) -> tuple[dict[tuple[int, int], QtGui.QColor], set[tuple[int, int]]]:
        center = _parse_vec2(self._center_xz.text().strip())

        points_colored: dict[tuple[int, int], QtGui.QColor] = {}
        points_all: set[tuple[int, int]] = set()

        if bool(self._circle_enable.isChecked()):
            radius = float(self._circle_radius.value())
            filled = bool(self._circle_filled.isChecked())
            shape_points = _gen_circle_points(center=center, radius=radius, filled=filled)

            thickness = int(self._circle_thickness.value())
            metric_name = str(self._circle_metric.currentData() or "euclidean")
            shape_points = _apply_thickness(shape_points, metric=metric_name, thick=thickness)

            for point in shape_points:
                points_colored[point] = QtGui.QColor(95, 170, 255, 215)
            points_all |= shape_points

        if bool(self._spiral_enable.isChecked()):
            radius_limit = float(self._spiral_radius_limit.value())
            spacing = float(self._spiral_spacing.value())
            guide_rotation = float(self._spiral_guide_rotation.value())
            segments = int(self._spiral_segments.value())
            max_chord_step = float(self._spiral_max_chord_step.value())
            shape_points = _gen_double_spiral_points(
                center=center,
                radius_limit=radius_limit,
                spacing=spacing,
                guide_rotation_deg=guide_rotation,
                segments=segments,
                max_chord_step=max_chord_step,
            )

            thickness = int(self._spiral_thickness.value())
            metric_name = str(self._spiral_metric.currentData() or "euclidean")
            shape_points = _apply_thickness(shape_points, metric=metric_name, thick=thickness)

            for point in shape_points:
                if point in points_colored:
                    points_colored[point] = QtGui.QColor(185, 210, 120, 225)
                else:
                    points_colored[point] = QtGui.QColor(85, 210, 120, 215)
            points_all |= shape_points

        return points_colored, points_all

    def _on_generate(self) -> None:
        try:
            points_colored, points_all = self._compose_points()
            self._current_points = dict(points_colored)

            self._viewport.set_points(self._current_points)

            y_level = int(self._y_level.value())
            sorted_coords = sorted(list(points_all), key=lambda p: (int(p[1]), int(p[0])))

            coord_lines = [f"{int(x)} {int(y_level)} {int(z)}" for x, z in sorted_coords]
            self._txt_coords.setPlainText("\n".join(coord_lines))

            ascii_text = _points_to_ascii(points_all, pad=1, on="#", off=".")
            self._txt_ascii.setPlainText(ascii_text)

            self._lbl_status.setText(f"Blocks: {len(sorted_coords)}")
        except Exception as error:
            QtWidgets.QMessageBox.critical(self, "Generate failed", str(error))

    def _on_clear(self) -> None:
        self._current_points = {}
        self._viewport.clear()
        self._txt_coords.setPlainText("")
        self._txt_ascii.setPlainText("")
        self._lbl_status.setText("Cleared")

    def _on_save_coords(self) -> None:
        if not self._txt_coords.toPlainText().strip():
            return

        default_path = (Path.cwd() / "build_coords.txt").resolve()
        output_path, _selected_filter = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save coordinates",
            str(default_path),
            "Text (*.txt)",
        )
        if not output_path:
            return

        try:
            Path(output_path).write_text(self._txt_coords.toPlainText().rstrip() + "\n", encoding="utf-8", newline="\n")
            self._lbl_status.setText(f"Saved: {output_path}")
        except Exception as error:
            QtWidgets.QMessageBox.critical(self, "Save failed", str(error))


# endregion MAINWIN
# region ENTRY
# Application bootstrap: create QApplication, apply theme, show main window


def main() -> None:
    app = QtWidgets.QApplication(sys.argv)
    _apply_dark_theme(app)
    main_window = BuildDesignPlannerMainWindow()
    main_window.resize(1280, 820)
    main_window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()


# endregion ENTRY
