"""
Minecraft Model UV Packer (UI)

PySide6 UI layer for generating non-overlapping UVs for Minecraft block-model
cuboids (`elements[]`). The core logic (connected box-unwrap, island packing,
UV write-back, resnap, and preview generation) lives in
`mc_model_uv_packer_core`, which this module imports and re-exports.

This tool writes UVs in *normalized* (0..1) space with high precision floats.
(You can convert to vanilla block-model UV units later by multiplying by 16.)

TOOLSGROUP::MODEL
SORTGROUP::7
SORTPRIORITY::80
STATUS::active
VERSION::20260309

Expected inputs
- Minecraft model JSON file containing `elements[]`.

Outputs
- Updated model JSON where each element face has a packed `uv` rect.
- Optional per-face `rotation` (90-degree steps) when needed to keep box-net
  edges consistent.
- 2D atlas preview showing packed islands.

"""

#region SETUP
# Module imports, PySide6 bootstrap, and core re-exports

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Iterable, Optional

from mc_model_uv_packer_core import (
    Axis,
    Cuboid,
    Element,
    Face,
    FacePlacement,
    FaceRect,
    Island,
    PackedIsland,
    Rotation,
    _apply_uvs_to_model,
    _build_connected_box_island,
    _choose_central_face,
    _describe_texel_density,
    _detect_uv_islands,
    _detect_uv_units,
    _face_rects_px,
    _pack_islands_shelf,
    _parse_element,
    _pixels_to_uv_rect,
    _rects_touch,
    _resnap_model_uvs,
    _r12,
    _r3,
    _secondary_fill_pass,
    _snap_int,
    _uvs_to_packed_preview,
    _uv_rect_to_pixels,
    format_minecraft_model_json,
)


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
#endregion


#region UI-THEME
# Dark theme stylesheet and per-element color generation (golden-ratio hue stride)


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
        "QPushButton#btn_pack { background: #1b3326; border: 1px solid #2d5b3f; }"
        "QPushButton#btn_pack:hover { border: 1px solid #3d7a55; }"
        "QPushButton#btn_save { background: #241a2c; border: 1px solid #4a2c63; }"
        "QPushButton#btn_save:hover { border: 1px solid #6a3b90; }"
        "QSplitter::handle { background: #0b0b0d; }"
    )


def _color_for_index(i: int) -> QtGui.QColor:
    h = (float(int(i) * 0.61803398875) % 1.0) * 360.0
    c = QtGui.QColor()
    c.setHsv(int(h) % 360, 170, 235, 210)
    return c
#endregion


#region UI-ATLAS
# 2D atlas preview (QGraphicsView): pan/zoom, per-element colors, overlap
# highlighting, and seam visualization for the selected element's box-net


class AtlasPreview(QtWidgets.QGraphicsView):
    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self.setRenderHints(QtGui.QPainter.RenderHint.Antialiasing | QtGui.QPainter.RenderHint.TextAntialiasing)
        self.setDragMode(QtWidgets.QGraphicsView.DragMode.NoDrag)
        self.setTransformationAnchor(QtWidgets.QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QtWidgets.QGraphicsView.ViewportAnchor.AnchorUnderMouse)

        self._scene = QtWidgets.QGraphicsScene(self)
        self.setScene(self._scene)

        self._tex_w = 256
        self._tex_h = 256

        self._is_panning = False
        self._pan_last: Optional[QtCore.QPoint] = None
        self._user_view = False
        self._show_labels = True

    def set_show_labels(self, enabled: bool) -> None:
        self._show_labels = bool(enabled)
        self.update()

    def _set_user_view(self) -> None:
        self._user_view = True

    def _fake_left_event(self, event: QtGui.QMouseEvent, *, etype: QtCore.QEvent.Type) -> QtGui.QMouseEvent:
        # Drive QGraphicsView's built-in hand-drag logic using middle mouse.
        return QtGui.QMouseEvent(
            etype,
            event.position(),
            event.globalPosition(),
            QtCore.Qt.MouseButton.LeftButton,
            QtCore.Qt.MouseButton.LeftButton,
            event.modifiers(),
        )

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        if event.button() == QtCore.Qt.MouseButton.MiddleButton:
            self._set_user_view()
            self._is_panning = True
            self.setDragMode(QtWidgets.QGraphicsView.DragMode.ScrollHandDrag)
            super().mousePressEvent(self._fake_left_event(event, etype=QtCore.QEvent.Type.MouseButtonPress))
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:
        if event.button() == QtCore.Qt.MouseButton.MiddleButton:
            self._is_panning = False
            super().mouseReleaseEvent(self._fake_left_event(event, etype=QtCore.QEvent.Type.MouseButtonRelease))
            self.setDragMode(QtWidgets.QGraphicsView.DragMode.NoDrag)
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        if self._is_panning:
            super().mouseMoveEvent(self._fake_left_event(event, etype=QtCore.QEvent.Type.MouseMove))
            event.accept()
            return
        super().mouseMoveEvent(event)

    def wheelEvent(self, event: QtGui.QWheelEvent) -> None:
        delta = event.angleDelta().y() / 120.0
        scale = 1.15 ** delta
        self._set_user_view()
        self.scale(scale, scale)

    def set_preview(
        self,
        *,
        tex_w: int,
        tex_h: int,
        packed: list[PackedIsland],
        colors_by_element: Optional[dict[int, QtGui.QColor]] = None,
        selected_element=None,
    ) -> None:
        tex_w = int(tex_w)
        tex_h = int(tex_h)
        size_changed = tex_w != int(self._tex_w) or tex_h != int(self._tex_h)
        if size_changed:
            self._user_view = False

        self._tex_w = int(tex_w)
        self._tex_h = int(tex_h)

        if selected_element is None:
            sel_set: set[int] = set()
        elif isinstance(selected_element, int):
            sel_set = {int(selected_element)}
        else:
            sel_set = {int(x) for x in selected_element}

        self._scene.clear()
        self._scene.setSceneRect(0, 0, float(tex_w), float(tex_h))

        bg = QtGui.QColor(10, 10, 12)
        border_pen = QtGui.QPen(QtGui.QColor(130, 130, 140, 120))
        border_pen.setWidthF(1.0)
        self._scene.addRect(0, 0, float(tex_w), float(tex_h), border_pen, QtGui.QBrush(bg))

        face_outline = QtGui.QPen(QtGui.QColor(220, 220, 230, 130))
        face_outline.setWidthF(1.0)
        face_outline_over = QtGui.QPen(QtGui.QColor(255, 85, 85, 220))
        face_outline_over.setWidthF(2.0)
        face_outline_sel = QtGui.QPen(QtGui.QColor(255, 255, 255, 220))
        face_outline_sel.setWidthF(2.0)

        face_outline_overlap = QtGui.QPen(QtGui.QColor(255, 0, 255, 255))
        face_outline_overlap.setWidthF(2.5)

        text_pen = QtGui.QPen(QtGui.QColor(230, 230, 235, 180))

        def intersects(a: QtCore.QRectF, b: QtCore.QRectF) -> bool:
            x0 = max(float(a.left()), float(b.left()))
            y0 = max(float(a.top()), float(b.top()))
            x1 = min(float(a.right()), float(b.right()))
            y1 = min(float(a.bottom()), float(b.bottom()))
            return (x1 - x0) > 1e-6 and (y1 - y0) > 1e-6

        drawn_rects: list[tuple[int, QtCore.QRectF, bool]] = []
        placed_label_rects: list[QtCore.QRectF] = []

        for i, pisl in enumerate(packed):
            base = None
            if colors_by_element is not None:
                base = colors_by_element.get(int(pisl.island.element_idx))
            if base is None:
                base = _color_for_index(int(pisl.island.element_idx))

            alpha = 90
            if sel_set and int(pisl.island.element_idx) not in sel_set:
                alpha = 40
            col = QtGui.QColor(base.red(), base.green(), base.blue(), alpha)

            for pl in pisl.island.placements:
                x = float(pisl.x_px + pl.x_px)
                y = float(pisl.y_px + pl.y_px)
                w = float(pl.w_px)
                h = float(pl.h_px)
                r = QtCore.QRectF(x, y, w, h)
                overlap = False
                for _el_idx, rr, _ov in drawn_rects:
                    if intersects(r, rr):
                        overlap = True
                        break

                pen = face_outline
                if sel_set and int(pisl.island.element_idx) in sel_set:
                    pen = face_outline_sel
                if bool(pisl.overflow):
                    pen = face_outline_over
                if overlap:
                    pen = face_outline_overlap
                self._scene.addRect(x, y, w, h, pen, QtGui.QBrush(col))

                drawn_rects.append((int(pisl.island.element_idx), r, overlap))

                if self._show_labels:
                    lab = f"{pisl.island.element_idx}:{pl.face}"
                    if int(pl.rot) % 360 != 0:
                        lab += f" r{int(pl.rot)%360}"
                    rect_size = min(float(w), float(h))
                    font_pt = max(4, min(14, int(rect_size * 0.14)))
                    lab_font = QtGui.QFont("Consolas", font_pt)
                    t = self._scene.addText(lab, lab_font)
                    t.setDefaultTextColor(text_pen.color())
                    tw = t.boundingRect().width()
                    th = t.boundingRect().height()
                    candidates = [
                        (x + 2.0, y + 1.0),
                        (x + w - tw - 2.0, y + 1.0),
                        (x + 2.0, y + h - th - 1.0),
                        (x + w - tw - 2.0, y + h - th - 1.0),
                        (x + (w - tw) * 0.5, y + (h - th) * 0.5),
                    ]
                    best_pos = candidates[0]
                    for cx, cy in candidates:
                        lr = QtCore.QRectF(cx, cy, tw, th)
                        if not any(intersects(lr, pr) for pr in placed_label_rects):
                            best_pos = (cx, cy)
                            break
                    placed_label_rects.append(QtCore.QRectF(best_pos[0], best_pos[1], tw, th))
                    t.setPos(best_pos[0], best_pos[1])

        def stable_hash(s: str) -> int:
            h = 2166136261
            for ch in s:
                h ^= ord(ch)
                h = (h * 16777619) & 0xFFFFFFFF
            return int(h)

        def color_for_edge_key(a: str, b: str) -> QtGui.QColor:
            k0, k1 = (a, b) if str(a) <= str(b) else (b, a)
            hv = stable_hash(f"{k0}|{k1}")
            hue = int(hv % 360)
            c = QtGui.QColor()
            c.setHsv(hue, 220, 255, 255)
            return c

        def rects_touch(a: QtCore.QRectF, b: QtCore.QRectF, *, eps: float = 1e-6) -> bool:
            ax0, ay0, ax1, ay1 = float(a.left()), float(a.top()), float(a.right()), float(a.bottom())
            bx0, by0, bx1, by1 = float(b.left()), float(b.top()), float(b.right()), float(b.bottom())

            ovx = min(ax1, bx1) - max(ax0, bx0)
            ovy = min(ay1, by1) - max(ay0, by0)

            if ovy > eps and (abs(ax1 - bx0) <= eps or abs(bx1 - ax0) <= eps):
                return True
            if ovx > eps and (abs(ay1 - by0) <= eps or abs(by1 - ay0) <= eps):
                return True
            return False

        def rotate_local_side_for_uv(*, uv_side: str, rot_deg: int) -> str:
            r = int(rot_deg) % 360
            if r not in {0, 90, 180, 270}:
                r = 0

            order = ["top", "right", "bottom", "left"]
            if uv_side not in order:
                return str(uv_side)

            idx = order.index(uv_side)
            steps = (r // 90) % 4

            local = order[(idx - steps) % 4]
            return str(local)

        face_neighbors = {
            "north": {"top": "up", "bottom": "down", "left": "west", "right": "east"},
            "south": {"top": "up", "bottom": "down", "left": "east", "right": "west"},
            "east": {"top": "up", "bottom": "down", "left": "south", "right": "north"},
            "west": {"top": "up", "bottom": "down", "left": "north", "right": "south"},
            "up": {"top": "north", "bottom": "south", "left": "west", "right": "east"},
            "down": {"top": "south", "bottom": "north", "left": "west", "right": "east"},
        }

        for sel in sel_set:
            sel_item = next((pi for pi in packed if int(pi.island.element_idx) == sel), None)
            if sel_item is not None:
                by_face: dict[str, tuple[QtCore.QRectF, int]] = {}
                for pl in sel_item.island.placements:
                    x = float(sel_item.x_px + pl.x_px)
                    y = float(sel_item.y_px + pl.y_px)
                    r = QtCore.QRectF(x, y, float(pl.w_px), float(pl.h_px))
                    by_face[str(pl.face)] = (r, int(pl.rot) % 360)

                connected: set[tuple[str, str]] = set()
                faces = list(by_face.keys())
                for i in range(len(faces)):
                    for j in range(i + 1, len(faces)):
                        a = by_face[faces[i]][0]
                        b = by_face[faces[j]][0]
                        if rects_touch(a, b, eps=1e-6):
                            f0 = str(faces[i])
                            f1 = str(faces[j])
                            k = (f0, f1) if f0 <= f1 else (f1, f0)
                            connected.add(k)

                seam_pen_width = 3.0
                for face, (r, rot) in by_face.items():
                    neigh_map = face_neighbors.get(str(face), {})
                    for uv_side in ("top", "right", "bottom", "left"):
                        local_side = rotate_local_side_for_uv(uv_side=uv_side, rot_deg=rot)
                        neigh = str(neigh_map.get(str(local_side), ""))
                        if not neigh:
                            continue

                        k = (str(face), neigh) if str(face) <= neigh else (neigh, str(face))
                        if k in connected:
                            continue

                        col = color_for_edge_key(str(face), neigh)
                        pen = QtGui.QPen(col)
                        pen.setWidthF(seam_pen_width)
                        pen.setCosmetic(True)

                        if uv_side == "top":
                            self._scene.addLine(float(r.left()), float(r.top()), float(r.right()), float(r.top()), pen)
                        elif uv_side == "bottom":
                            self._scene.addLine(float(r.left()), float(r.bottom()), float(r.right()), float(r.bottom()), pen)
                        elif uv_side == "left":
                            self._scene.addLine(float(r.left()), float(r.top()), float(r.left()), float(r.bottom()), pen)
                        else:
                            self._scene.addLine(float(r.right()), float(r.top()), float(r.right()), float(r.bottom()), pen)

        if not self._user_view:
            self.resetTransform()
            if tex_w > 0 and tex_h > 0:
                self.fitInView(0, 0, float(tex_w), float(tex_h), QtCore.Qt.AspectRatioMode.KeepAspectRatio)
#endregion


#region UI-VIEW
# 3D model viewport (QWidget): hand-rolled perspective camera, painter's-algorithm
# depth sort, face shading, wireframe, and click-to-pick cuboid selection


class UVModelViewport(QtWidgets.QWidget):
    selectionChanged = QtCore.Signal(int)

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)

        self.setMouseTracking(True)
        self.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)

        self._yaw = 45.0
        self._pitch = 25.0
        self._distance = 42.0
        self._center = (8.0, 8.0, 8.0)

        self._last_mouse_pos: Optional[QtCore.QPoint] = None
        self._mouse_down: Optional[QtCore.QPoint] = None
        self._cuboids: list[Cuboid] = []
        self._colors: list[QtGui.QColor] = []
        self._central_faces: list[Optional[Face]] = []
        self._shade_only_central = False
        self._draw_faces = True
        self._draw_wireframe = True
        self._draw_face_label = True
        self._draw_element_id = False

        self._selected: set[int] = set()

    def set_render_options(self, *, faces: bool, wireframe: bool, shade_only_central: bool) -> None:
        self._draw_faces = bool(faces)
        self._draw_wireframe = bool(wireframe)
        self._shade_only_central = bool(shade_only_central)
        self.update()

    def set_draw_face_label(self, enabled: bool) -> None:
        self._draw_face_label = bool(enabled)
        self.update()

    def set_draw_element_id(self, enabled: bool) -> None:
        self._draw_element_id = bool(enabled)
        self.update()

    def set_cuboids(
        self,
        cuboids: Iterable[Cuboid],
        *,
        colors: Optional[list[QtGui.QColor]] = None,
        central_faces: Optional[list[Optional[Face]]] = None,
    ) -> None:
        self._cuboids = list(cuboids)
        self._colors = list(colors) if colors is not None else []
        self._central_faces = list(central_faces) if central_faces is not None else []
        self.update()

    def set_selected(self, idx) -> None:
        if idx is None:
            self._selected = set()
        elif isinstance(idx, int):
            self._selected = {int(idx)}
        else:
            self._selected = {int(x) for x in idx}
        self.update()

    def wheelEvent(self, event: QtGui.QWheelEvent) -> None:
        delta = event.angleDelta().y() / 120.0
        self._distance *= 0.9 ** delta
        self._distance = max(6.0, min(250.0, self._distance))
        self.update()

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            p = event.position().toPoint()
            self._last_mouse_pos = p
            self._mouse_down = p
            event.accept()
            return

        if event.button() == QtCore.Qt.MouseButton.RightButton:
            idx = self._pick_cuboid(event.position())
            if idx is not None:
                self._selected = {int(idx)}
                self.selectionChanged.emit(int(idx))
                self.update()
            event.accept()
            return

        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            if self._mouse_down is not None:
                p = event.position().toPoint()
                dx = int(p.x() - self._mouse_down.x())
                dy = int(p.y() - self._mouse_down.y())
                if (dx * dx + dy * dy) <= 9:
                    idx = self._pick_cuboid(event.position())
                    if idx is not None:
                        self._selected = {int(idx)}
                        self.selectionChanged.emit(int(idx))
                        self.update()
            self._last_mouse_pos = None
            self._mouse_down = None
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

    def _camera(self):
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

        return to_camera, project_cam

    def _cuboid_points(self, c: Cuboid) -> list[tuple[float, float, float]]:
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
            angle_deg = float(c.rotation.angle)
            origin = (float(c.rotation.origin[0]), float(c.rotation.origin[1]), float(c.rotation.origin[2]))

            def rotate(p: tuple[float, float, float]) -> tuple[float, float, float]:
                ox, oy, oz = origin
                x, y, z = p
                x -= ox
                y -= oy
                z -= oz

                a = math.radians(angle_deg)
                cc = math.cos(a)
                ss = math.sin(a)

                if axis == "x":
                    yy = cc * y - ss * z
                    zz = ss * y + cc * z
                    xx = x
                elif axis == "y":
                    xx = cc * x + ss * z
                    zz = -ss * x + cc * z
                    yy = y
                else:
                    xx = cc * x - ss * y
                    yy = ss * x + cc * y
                    zz = z

                return (xx + ox, yy + oy, zz + oz)

            pts = [rotate(p) for p in pts]

        return pts

    def _pick_cuboid(self, pos: QtCore.QPointF) -> Optional[int]:
        if not self._cuboids:
            return None

        to_camera, project_cam = self._camera()
        px = float(pos.x())
        py = float(pos.y())

        best_i: Optional[int] = None
        best_z = 1e18

        for i, c in enumerate(self._cuboids):
            pts = self._cuboid_points(c)
            cam_pts = [to_camera(p) for p in pts]
            proj = [project_cam(cp) for cp in cam_pts]
            proj2 = [p for p in proj if p is not None]
            if not proj2:
                continue

            xs = [float(p.x()) for p in proj2]
            ys = [float(p.y()) for p in proj2]
            x0, x1 = min(xs), max(xs)
            y0, y1 = min(ys), max(ys)

            if not (x0 <= px <= x1 and y0 <= py <= y1):
                continue

            z = min(float(cp[2]) for cp in cam_pts)
            if z < best_z:
                best_z = z
                best_i = int(i)

        return best_i

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)
        painter.fillRect(self.rect(), QtGui.QColor(12, 12, 16))

        to_camera, project_cam = self._camera()

        face_idx: list[tuple[tuple[int, int, int, int], Face]] = [
            ((0, 1, 2, 3), "north"),
            ((4, 5, 6, 7), "south"),
            ((0, 1, 5, 4), "down"),
            ((1, 2, 6, 5), "east"),
            ((2, 3, 7, 6), "up"),
            ((3, 0, 4, 7), "west"),
        ]

        faces: list[tuple[float, QtGui.QPolygonF, QtGui.QColor]] = []

        labels: list[tuple[QtCore.QPointF, str, float]] = []

        face_letter = {
            "north": "N",
            "south": "S",
            "east": "E",
            "west": "W",
            "up": "U",
            "down": "D",
        }

        if self._draw_faces:
            for i_c, c in enumerate(self._cuboids):
                pts = self._cuboid_points(c)
                cam_pts = [to_camera(p) for p in pts]

                base_col = _color_for_index(int(i_c))
                if 0 <= int(i_c) < len(self._colors):
                    base_col = self._colors[int(i_c)]

                face_base = QtGui.QColor(base_col.red(), base_col.green(), base_col.blue(), 135)
                central = None
                if 0 <= int(i_c) < len(self._central_faces):
                    central = self._central_faces[int(i_c)]

                for (i0, i1, i2, i3), fname in face_idx:
                    if self._shade_only_central and central is not None and fname != central:
                        continue

                    p0 = project_cam(cam_pts[i0])
                    p1 = project_cam(cam_pts[i1])
                    p2 = project_cam(cam_pts[i2])
                    p3 = project_cam(cam_pts[i3])
                    if p0 is None or p1 is None or p2 is None or p3 is None:
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
                        shade = 0.45 + (0.55 * max(0.0, min(1.0, abs(float(nz)) / nlen)))

                    col = QtGui.QColor(
                        int(face_base.red() * shade),
                        int(face_base.green() * shade),
                        int(face_base.blue() * shade),
                        face_base.alpha(),
                    )
                    poly = QtGui.QPolygonF([p0, p1, p2, p3])
                    depth = (cam_pts[i0][2] + cam_pts[i1][2] + cam_pts[i2][2] + cam_pts[i3][2]) / 4.0
                    faces.append((depth, poly, col))

                    if self._draw_face_label and int(i_c) in self._selected and central is not None and str(fname) == str(central):
                        xs = [float(p0.x()), float(p1.x()), float(p2.x()), float(p3.x())]
                        ys = [float(p0.y()), float(p1.y()), float(p2.y()), float(p3.y())]
                        cx = sum(xs) / 4.0
                        cy = sum(ys) / 4.0
                        face_w = max(xs) - min(xs)
                        face_h = max(ys) - min(ys)
                        face_size = max(face_w, face_h)
                        letter = face_letter.get(str(fname), str(fname)[:1].upper())
                        labels.append((QtCore.QPointF(cx, cy), str(letter), float(face_size)))

            faces.sort(key=lambda it: it[0], reverse=True)
            painter.setPen(QtCore.Qt.PenStyle.NoPen)
            for _depth, poly, col in faces:
                painter.setBrush(QtGui.QBrush(col))
                painter.drawPolygon(poly)

            for pos, letter, face_size in labels:
                font_pt = max(8, min(36, int(face_size * 0.22)))
                offset = max(1.0, font_pt * 0.12)
                painter.setPen(QtGui.QPen(QtGui.QColor(0, 0, 0, 220)))
                f = QtGui.QFont("Consolas", font_pt, QtGui.QFont.Weight.Bold)
                painter.setFont(f)
                painter.drawText(pos + QtCore.QPointF(offset, offset), str(letter))
                painter.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255, 240)))
                painter.drawText(pos, str(letter))

        if self._draw_wireframe:
            for i_c, c in enumerate(self._cuboids):
                pts = self._cuboid_points(c)

                base_col = _color_for_index(int(i_c))
                if 0 <= int(i_c) < len(self._colors):
                    base_col = self._colors[int(i_c)]

                line_col = QtGui.QColor(base_col.red(), base_col.green(), base_col.blue(), 235)

                width = 1.35
                if int(i_c) in self._selected:
                    width = 2.6
                    line_col = QtGui.QColor(255, 255, 255, 240)

                    cx = sum(p[0] for p in pts) / 8.0
                    cy = sum(p[1] for p in pts) / 8.0
                    cz = sum(p[2] for p in pts) / 8.0
                    pts = [(cx + (p[0] - cx) * 1.03, cy + (p[1] - cy) * 1.03, cz + (p[2] - cz) * 1.03) for p in pts]

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

                pen = QtGui.QPen(line_col)
                pen.setWidthF(float(width))
                painter.setPen(pen)
                for ia, ib in edges:
                    pa = project_cam(to_camera(pts[ia]))
                    pb = project_cam(to_camera(pts[ib]))
                    if pa is None or pb is None:
                        continue
                    painter.drawLine(pa, pb)

        if self._draw_element_id:
            for i_c, c in enumerate(self._cuboids):
                center = c.center()
                cp = to_camera(center)
                pp = project_cam(cp)
                if pp is None:
                    continue

                base_col = _color_for_index(int(i_c))
                if 0 <= int(i_c) < len(self._colors):
                    base_col = self._colors[int(i_c)]

                label_col = QtGui.QColor(base_col.red(), base_col.green(), base_col.blue(), 255)
                shadow_col = QtGui.QColor(0, 0, 0, 200)

                w = max(1, self.width())
                h = max(1, self.height())
                f = min(w, h) * 0.9
                z = float(cp[2])
                if z <= 0.05:
                    continue
                screen_size = (f / z) * 0.02
                font_pt = max(7, min(28, int(screen_size * 14)))

                text = str(int(i_c))
                offset = max(1.0, font_pt * 0.12)

                painter.setPen(QtGui.QPen(shadow_col))
                ef = QtGui.QFont("Consolas", font_pt, QtGui.QFont.Weight.Bold)
                painter.setFont(ef)
                painter.drawText(pp + QtCore.QPointF(offset, offset), text)
                painter.setPen(QtGui.QPen(label_col))
                painter.drawText(pp, text)

        painter.end()
#endregion


#region UI-MAIN
# Main window: 3-pane splitter (controls | 3D+atlas | output), slots for
# load/pack/save/resnap, element list with central-face overrides


class UVPackerMainWindow(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Minecraft Model UV Packer")

        self._source_path: Optional[Path] = None
        self._model: Optional[dict] = None
        self._elements: list[Element] = []

        self._central_overrides: dict[int, Face] = {}
        self._selected_element: Optional[int] = None
        self._selected_elements: list[int] = []
        self._element_colors: list[QtGui.QColor] = []
        self._last_central_faces: list[Optional[Face]] = []

        self._packed: list[PackedIsland] = []
        self._out_model: Optional[dict] = None

        self._viewport_3d = UVModelViewport()
        self._viewport_3d.selectionChanged.connect(self._on_viewport_selected)

        self._preview = AtlasPreview()

        self._mid_split = QtWidgets.QSplitter()
        self._mid_split.setOrientation(QtCore.Qt.Orientation.Horizontal)
        self._mid_split.addWidget(self._viewport_3d)
        self._mid_split.addWidget(self._preview)
        self._mid_split.setStretchFactor(0, 1)
        self._mid_split.setStretchFactor(1, 1)

        controls = self._build_controls()
        output = self._build_output()

        splitter = QtWidgets.QSplitter()
        splitter.setOrientation(QtCore.Qt.Orientation.Horizontal)
        splitter.addWidget(controls)
        splitter.addWidget(self._mid_split)
        splitter.addWidget(output)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 0)
        splitter.setCollapsible(0, False)
        splitter.setCollapsible(2, False)

        self.setCentralWidget(splitter)

    def _sync_views(self) -> None:
        colors_by_element = {i: self._element_colors[i] for i in range(len(self._element_colors))}
        self._preview.set_show_labels(bool(self._atlas_labels.isChecked()))
        self._preview.set_preview(
            tex_w=int(self._tex_w.value()),
            tex_h=int(self._tex_h.value()),
            packed=self._packed,
            colors_by_element=colors_by_element,
            selected_element=self._selected_elements,
        )

        self._viewport_3d.set_render_options(
            faces=bool(self._vp_faces.isChecked()),
            wireframe=bool(self._vp_wire.isChecked()),
            shade_only_central=bool(self._vp_shade_only_central.isChecked()),
        )
        self._viewport_3d.set_draw_face_label(bool(self._vp_face_label.isChecked()))
        self._viewport_3d.set_draw_element_id(bool(self._vp_element_id.isChecked()))
        self._viewport_3d.set_cuboids(
            [e.cuboid for e in self._elements],
            colors=self._element_colors,
            central_faces=self._last_central_faces,
        )
        self._viewport_3d.set_selected(self._selected_elements)

    def _on_viewport_selected(self, idx: int) -> None:
        if 0 <= int(idx) < self._lst_elements.count():
            self._lst_elements.setCurrentRow(int(idx))

    def _on_selection_changed(self) -> None:
        rows = sorted(i.row() for i in self._lst_elements.selectedItems())
        selected: list[int] = []
        for r in rows:
            if 0 <= r < len(self._elements):
                selected.append(int(self._elements[r].idx))
        self._selected_elements = selected
        self._selected_element = selected[0] if selected else None

        has_sel = len(selected) > 0
        any_override = any(int(s) in self._central_overrides for s in selected)
        self._btn_clear_override.setEnabled(has_sel and any_override)

        self._sync_override_face_display()

        self._sync_views()

    def _sync_override_face_display(self) -> None:
        selected = self._selected_elements
        if not selected:
            self._override_face.blockSignals(True)
            self._override_face.setCurrentIndex(self._override_face.findData("auto"))
            self._override_face.blockSignals(False)
        else:
            overrides = [self._central_overrides.get(int(s)) for s in selected]
            unique = set(overrides)
            if len(unique) == 1:
                ov = unique.pop()
                target = "auto" if ov is None else str(ov)
            else:
                target = "auto"
            self._override_face.blockSignals(True)
            self._override_face.setCurrentIndex(self._override_face.findData(target))
            self._override_face.blockSignals(False)

    def _on_override_face_changed(self) -> None:
        if not self._selected_elements:
            return
        v = str(self._override_face.currentData() or "auto")
        for s in self._selected_elements:
            if v == "auto":
                self._central_overrides.pop(int(s), None)
            else:
                self._central_overrides[int(s)] = v  # type: ignore[assignment]
        any_override = any(int(s) in self._central_overrides for s in self._selected_elements)
        self._btn_clear_override.setEnabled(any_override)
        self._on_pack()

    def _on_clear_override(self) -> None:
        if not self._selected_elements:
            return
        for s in self._selected_elements:
            self._central_overrides.pop(int(s), None)
        self._btn_clear_override.setEnabled(False)
        self._on_pack()

    def _build_controls(self) -> QtWidgets.QWidget:
        w = QtWidgets.QWidget()
        w.setMinimumWidth(410)

        self._txt_path = QtWidgets.QLineEdit("")
        self._txt_path.setReadOnly(True)

        self._btn_load = QtWidgets.QPushButton("Load JSON")
        self._btn_load.clicked.connect(self._on_load)

        self._tex_w = QtWidgets.QSpinBox()
        self._tex_w.setRange(1, 8192)
        self._tex_w.setValue(256)

        self._tex_h = QtWidgets.QSpinBox()
        self._tex_h.setRange(1, 8192)
        self._tex_h.setValue(256)

        self._density = QtWidgets.QDoubleSpinBox()
        self._density.setRange(0.01, 16.0)
        self._density.setDecimals(4)
        self._density.setValue(1.0)

        self._snap_px = QtWidgets.QSpinBox()
        self._snap_px.setRange(1, 256)
        self._snap_px.setValue(1)

        self._island_pad = QtWidgets.QSpinBox()
        self._island_pad.setRange(0, 256)
        self._island_pad.setValue(2)

        self._border_pad = QtWidgets.QSpinBox()
        self._border_pad.setRange(0, 256)
        self._border_pad.setValue(2)

        self._secondary_fill = QtWidgets.QCheckBox("Secondary fill pass")
        self._secondary_fill.setChecked(True)
        self._secondary_fill.setToolTip(
            "After initial shelf pack, relocate groups of similar islands into blank space to improve packing density."
        )

        self._prefer_horizontal = QtWidgets.QCheckBox("Prefer horizontal central face")
        self._prefer_horizontal.setChecked(True)

        self._central_mode = QtWidgets.QComboBox()
        self._central_mode.addItem("Auto", userData="auto")
        for f in ("north", "south", "east", "west", "up", "down"):
            self._central_mode.addItem(str(f), userData=str(f))
        self._central_mode.setCurrentIndex(self._central_mode.findData("up"))

        self._default_texture = QtWidgets.QLineEdit("#0")

        self._uv_units = QtWidgets.QComboBox()
        self._uv_units.addItem("Atlas pixels (0..W/H)", userData="px")
        self._uv_units.addItem("Normalized (0..1)", userData="norm")
        self._uv_units.addItem("Minecraft 0..16", userData="mc16")
        self._uv_units.setCurrentIndex(self._uv_units.findData("mc16"))
        self._uv_units.setEnabled(False)

        self._write_rotation = QtWidgets.QCheckBox("Write face rotation")
        self._write_rotation.setChecked(True)

        self._lst_elements = QtWidgets.QListWidget()
        self._lst_elements.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection)
        self._lst_elements.itemSelectionChanged.connect(self._on_selection_changed)

        self._override_face = QtWidgets.QComboBox()
        self._override_face.addItem("Auto", userData="auto")
        for f in ("north", "south", "east", "west", "up", "down"):
            self._override_face.addItem(str(f), userData=str(f))
        self._override_face.currentIndexChanged.connect(self._on_override_face_changed)

        self._btn_clear_override = QtWidgets.QPushButton("Clear override")
        self._btn_clear_override.clicked.connect(self._on_clear_override)
        self._btn_clear_override.setEnabled(False)

        self._vp_faces = QtWidgets.QCheckBox("3D faces")
        self._vp_faces.setChecked(True)
        self._vp_faces.stateChanged.connect(self._sync_views)

        self._vp_wire = QtWidgets.QCheckBox("3D wireframe")
        self._vp_wire.setChecked(True)
        self._vp_wire.stateChanged.connect(self._sync_views)

        self._vp_shade_only_central = QtWidgets.QCheckBox("Shade only central face")
        self._vp_shade_only_central.setChecked(False)
        self._vp_shade_only_central.stateChanged.connect(self._sync_views)

        self._vp_face_label = QtWidgets.QCheckBox("3D face label")
        self._vp_face_label.setChecked(True)
        self._vp_face_label.stateChanged.connect(self._sync_views)

        self._vp_element_id = QtWidgets.QCheckBox("3D element IDs")
        self._vp_element_id.setChecked(False)
        self._vp_element_id.stateChanged.connect(self._sync_views)

        self._atlas_labels = QtWidgets.QCheckBox("Atlas face labels")
        self._atlas_labels.setChecked(True)
        self._atlas_labels.stateChanged.connect(self._sync_views)

        self._btn_pack = QtWidgets.QPushButton("Generate + Pack")
        self._btn_pack.setObjectName("btn_pack")
        self._btn_pack.clicked.connect(self._on_pack)
        self._btn_pack.setEnabled(False)

        self._btn_save = QtWidgets.QPushButton("Save Updated JSON")
        self._btn_save.setObjectName("btn_save")
        self._btn_save.clicked.connect(self._on_save)
        self._btn_save.setEnabled(False)

        # --- Resnap controls ---
        self._resnap_min_face = QtWidgets.QSpinBox()
        self._resnap_min_face.setRange(1, 64)
        self._resnap_min_face.setValue(1)
        self._resnap_min_face.setToolTip("Minimum face size in pixels after snapping. Faces smaller than this are expanded.")

        self._resnap_bias_gt = QtWidgets.QSpinBox()
        self._resnap_bias_gt.setRange(1, 64)
        self._resnap_bias_gt.setValue(2)
        self._resnap_bias_gt.setToolTip("Bias face size towards at least this many pixels when there is room (keeps small cuboid faces from collapsing to 1px).")

        self._resnap_border_pad = QtWidgets.QSpinBox()
        self._resnap_border_pad.setRange(0, 256)
        self._resnap_border_pad.setValue(1)
        self._resnap_border_pad.setToolTip("Pixels of padding kept between every face and the texture border during resnap.")

        self._resnap_island_pad = QtWidgets.QSpinBox()
        self._resnap_island_pad.setRange(0, 256)
        self._resnap_island_pad.setValue(1)
        self._resnap_island_pad.setToolTip("Minimum pixel gap kept between faces during resnap overlap resolution.")

        self._resnap_uv_units = QtWidgets.QComboBox()
        self._resnap_uv_units.addItem("Auto", userData="auto")
        self._resnap_uv_units.addItem("Minecraft 0..16", userData="mc16")
        self._resnap_uv_units.addItem("Normalized 0..1", userData="norm")
        self._resnap_uv_units.addItem("Atlas pixels", userData="px")
        self._resnap_uv_units.setCurrentIndex(self._resnap_uv_units.findData("mc16"))
        self._resnap_uv_units.setToolTip("How to interpret existing UV values. 'Auto' guesses from the value range, but pick explicitly if ambiguous.")

        self._btn_resnap = QtWidgets.QPushButton("Resnap UVs to Texture")
        self._btn_resnap.setObjectName("btn_pack")
        self._btn_resnap.clicked.connect(self._on_resnap)
        self._btn_resnap.setEnabled(False)
        self._btn_resnap.setToolTip(
            "Re-snaps existing face UVs to the current Texture W/H so every UV vertex lands on an integer pixel.\n"
            "Biases faces to >= min px (preferably >= bias px), nudges away from borders / neighbors for padding,\n"
            "and keeps the result close to the original UV layout."
        )

        self._lbl_status = QtWidgets.QLabel("Ready")
        self._lbl_status.setWordWrap(True)

        model_box = QtWidgets.QGroupBox("Model")
        model_form = QtWidgets.QFormLayout(model_box)
        model_form.setContentsMargins(8, 6, 8, 6)
        model_form.setVerticalSpacing(4)
        model_form.addRow("Path", self._txt_path)
        model_form.addRow(self._btn_load)

        tex_box = QtWidgets.QGroupBox("Atlas")
        tex_form = QtWidgets.QFormLayout(tex_box)
        tex_form.setContentsMargins(8, 6, 8, 6)
        tex_form.setVerticalSpacing(4)
        tex_form.setHorizontalSpacing(8)
        tex_form.setFieldGrowthPolicy(QtWidgets.QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsFieldGrow)

        wh_row = QtWidgets.QHBoxLayout()
        wh_row.addWidget(QtWidgets.QLabel("W"))
        wh_row.addWidget(self._tex_w, 1)
        wh_row.addWidget(QtWidgets.QLabel("H"))
        wh_row.addWidget(self._tex_h, 1)
        tex_form.addRow("Texture", wh_row)

        pad_row = QtWidgets.QHBoxLayout()
        pad_row.addWidget(QtWidgets.QLabel("Island"))
        pad_row.addWidget(self._island_pad, 1)
        pad_row.addWidget(QtWidgets.QLabel("Border"))
        pad_row.addWidget(self._border_pad, 1)
        tex_form.addRow("Pad (px)", pad_row)

        snap_row = QtWidgets.QHBoxLayout()
        snap_row.addWidget(self._snap_px, 1)
        snap_row.addWidget(QtWidgets.QLabel("Density"))
        snap_row.addWidget(self._density, 1)
        tex_form.addRow("Snap (px)", snap_row)

        tex_form.addRow(self._secondary_fill)

        unwrap_box = QtWidgets.QGroupBox("Box unwrap")
        unwrap_form = QtWidgets.QFormLayout(unwrap_box)
        unwrap_form.setContentsMargins(8, 6, 8, 6)
        unwrap_form.setVerticalSpacing(4)
        unwrap_form.addRow(self._prefer_horizontal)
        unwrap_form.addRow("Central face", self._central_mode)
        unwrap_form.addRow("Default texture", self._default_texture)
        unwrap_form.addRow("UV units", self._uv_units)
        unwrap_form.addRow(self._write_rotation)

        el_box = QtWidgets.QGroupBox("Elements")
        el_layout = QtWidgets.QVBoxLayout(el_box)
        el_layout.setContentsMargins(8, 6, 8, 6)
        el_layout.setSpacing(6)
        el_layout.addWidget(self._lst_elements)

        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("Central"))
        row.addWidget(self._override_face)
        el_layout.addLayout(row)

        el_layout.addWidget(self._btn_clear_override)

        view_box = QtWidgets.QGroupBox("3D / Preview")
        view_layout = QtWidgets.QGridLayout(view_box)
        view_layout.setContentsMargins(8, 6, 8, 6)
        view_layout.setHorizontalSpacing(12)
        view_layout.setVerticalSpacing(4)
        view_layout.addWidget(self._vp_faces, 0, 0)
        view_layout.addWidget(self._vp_wire, 0, 1)
        view_layout.addWidget(self._vp_shade_only_central, 1, 0)
        view_layout.addWidget(self._vp_face_label, 1, 1)
        view_layout.addWidget(self._vp_element_id, 2, 0)
        view_layout.addWidget(self._atlas_labels, 2, 1)

        resnap_box = QtWidgets.QGroupBox("Resnap existing UVs")
        resnap_form = QtWidgets.QFormLayout(resnap_box)
        resnap_form.setContentsMargins(8, 6, 8, 6)
        resnap_form.setVerticalSpacing(4)
        resnap_form.setHorizontalSpacing(8)

        mf_row = QtWidgets.QHBoxLayout()
        mf_row.addWidget(QtWidgets.QLabel("Min"))
        mf_row.addWidget(self._resnap_min_face, 1)
        mf_row.addWidget(QtWidgets.QLabel("Bias"))
        mf_row.addWidget(self._resnap_bias_gt, 1)
        resnap_form.addRow("Face (px)", mf_row)

        rp_row = QtWidgets.QHBoxLayout()
        rp_row.addWidget(QtWidgets.QLabel("Border"))
        rp_row.addWidget(self._resnap_border_pad, 1)
        rp_row.addWidget(QtWidgets.QLabel("Island"))
        rp_row.addWidget(self._resnap_island_pad, 1)
        resnap_form.addRow("Pad (px)", rp_row)

        resnap_form.addRow("UV units", self._resnap_uv_units)
        resnap_form.addRow(self._btn_resnap)

        layout = QtWidgets.QVBoxLayout(w)
        layout.addWidget(model_box)
        layout.addWidget(tex_box)
        layout.addWidget(unwrap_box)
        layout.addWidget(el_box)
        layout.addWidget(view_box)
        layout.addWidget(self._btn_pack)
        layout.addWidget(self._btn_save)
        layout.addWidget(resnap_box)
        layout.addWidget(self._lbl_status)
        layout.addStretch(1)
        return w

    def _build_output(self) -> QtWidgets.QWidget:
        w = QtWidgets.QWidget()
        w.setMinimumWidth(520)

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
        tabs.addTab(self._txt_json, "Updated JSON")

        box = QtWidgets.QGroupBox("Output")
        box_layout = QtWidgets.QVBoxLayout(box)
        box_layout.setContentsMargins(8, 6, 8, 6)
        box_layout.setSpacing(6)
        box_layout.addWidget(tabs)

        layout = QtWidgets.QVBoxLayout(w)
        layout.addWidget(box)
        layout.setContentsMargins(0, 0, 0, 0)
        return w

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
            self._central_overrides = {}
            self._selected_element = None
            self._selected_elements = []
            self._element_colors = [_color_for_index(i) for i in range(len(elements))]
            self._last_central_faces = [None] * len(elements)
            self._packed = []
            self._out_model = None

            self._txt_path.setText(str(p))
            self._btn_pack.setEnabled(len(elements) > 0)
            self._btn_resnap.setEnabled(len(elements) > 0)
            self._btn_save.setEnabled(False)

            self._txt_log.setPlainText(f"Loaded {len(elements)} elements")
            self._txt_json.setPlainText("")
            self._preview.set_preview(
                tex_w=int(self._tex_w.value()),
                tex_h=int(self._tex_h.value()),
                packed=[],
                colors_by_element={i: self._element_colors[i] for i in range(len(self._element_colors))},
                selected_element=self._selected_elements,
            )

            self._lst_elements.clear()
            for el in elements:
                dx, dy, dz = el.cuboid.size()
                it = QtWidgets.QListWidgetItem(f"{el.idx:03d}  size=({dx:g},{dy:g},{dz:g})")
                col = self._element_colors[int(el.idx)] if 0 <= int(el.idx) < len(self._element_colors) else _color_for_index(int(el.idx))
                it.setBackground(QtGui.QBrush(QtGui.QColor(col.red(), col.green(), col.blue(), 55)))
                self._lst_elements.addItem(it)

            if self._lst_elements.count() > 0:
                self._lst_elements.setCurrentRow(0)

            self._sync_views()
            self._lbl_status.setText(f"Loaded: {p.name} ({len(elements)} elements)")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Load failed", str(e))

    def _on_pack(self) -> None:
        if self._model is None or not self._elements:
            return

        try:
            tex_w = int(self._tex_w.value())
            tex_h = int(self._tex_h.value())
            density = float(self._density.value())
            snap_px = int(self._snap_px.value())
            island_pad = int(self._island_pad.value())
            border_pad = int(self._border_pad.value())
            prefer_horizontal = bool(self._prefer_horizontal.isChecked())
            central_mode = str(self._central_mode.currentData() or "auto")
            default_texture = self._default_texture.text().strip() or "#0"
            write_rotation = bool(self._write_rotation.isChecked())

            islands: list[Island] = []
            lines: list[str] = []
            central_faces: list[Optional[Face]] = []
            lines.append(f"Texture: {tex_w}x{tex_h}")
            lines.append(f"Density: {density:g}")
            lines.append(f"Snap(px): {snap_px}")
            lines.append(f"Padding: island={island_pad}px border={border_pad}px")
            lines.append(f"Central mode: {central_mode}")
            lines.append(f"Prefer horizontal: {prefer_horizontal}")
            lines.append("")

            for el in self._elements:
                rects = _face_rects_px(c=el.cuboid, tex_w=tex_w, tex_h=tex_h, density=density, snap_px=snap_px)

                ov = self._central_overrides.get(int(el.idx))
                if ov is not None:
                    central = ov
                elif central_mode == "auto":
                    central = _choose_central_face(c=el.cuboid, prefer_horizontal=prefer_horizontal)
                else:
                    central = central_mode  # type: ignore[assignment]

                isl = _build_connected_box_island(element_idx=int(el.idx), face_rects=rects, central=central)  # type: ignore[arg-type]
                islands.append(isl)
                central_faces.append(str(central))

                w, h = isl.size()
                tag = " (override)" if ov is not None else ""
                lines.append(f"Element {el.idx:03d}: central={central}{tag} island={w}x{h}px")

            packed = _pack_islands_shelf(
                islands=islands,
                tex_w=tex_w,
                tex_h=tex_h,
                island_pad_px=island_pad,
                border_pad_px=border_pad,
                snap_px=snap_px,
            )

            if bool(self._secondary_fill.isChecked()):
                packed, relocated = _secondary_fill_pass(
                    packed=packed,
                    cuboids=[e.cuboid for e in self._elements],
                    tex_w=tex_w,
                    tex_h=tex_h,
                    island_pad_px=island_pad,
                    border_pad_px=border_pad,
                    snap_px=snap_px,
                )
                if relocated > 0:
                    lines.append(f"Secondary fill: relocated {relocated} island(s) into blank space.")

            lines.extend(_describe_texel_density(packed=packed, tex_w=tex_w, tex_h=tex_h))

            out_model = _apply_uvs_to_model(
                model=self._model,
                packed=packed,
                tex_w=tex_w,
                tex_h=tex_h,
                default_texture=default_texture,
                write_rotation=write_rotation,
                uv_units="mc16",
            )

            self._packed = packed
            self._out_model = out_model
            self._last_central_faces = list(central_faces)

            self._preview.set_preview(
                tex_w=tex_w,
                tex_h=tex_h,
                packed=packed,
                colors_by_element={i: self._element_colors[i] for i in range(len(self._element_colors))},
                selected_element=self._selected_elements,
            )
            if any(bool(p.overflow) for p in packed):
                lines.append("")
                lines.append("WARNING: Packing overflow (some islands outside texture bounds).")
                self._lbl_status.setText(f"Packed {len(packed)} islands (OVERFLOW)")

            self._txt_log.setPlainText("\n".join(lines))
            self._txt_json.setPlainText(format_minecraft_model_json(out_model))

            self._btn_save.setEnabled(True)
            if not any(bool(p.overflow) for p in packed):
                self._lbl_status.setText(f"Packed {len(packed)} islands")
            self._sync_views()
        except Exception as e:
            self._txt_log.setPlainText(f"Pack error: {e}")
            self._txt_json.setPlainText("")
            self._btn_save.setEnabled(False)
            self._lbl_status.setText("Pack error (see log)")

    def _on_save(self) -> None:
        if self._out_model is None:
            return

        default = Path.cwd() / "model_uv_packed.json"
        if self._source_path is not None:
            default = self._source_path.with_name(self._source_path.stem + "_UV_PACKED.json")

        out_path, _filter = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save updated model JSON",
            str(default),
            "JSON (*.json)",
        )
        if not out_path:
            return

        try:
            Path(out_path).write_text(format_minecraft_model_json(self._out_model) + "\n", encoding="utf-8", newline="\n")
            self._lbl_status.setText(f"Saved: {out_path}")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Save failed", str(e))

    def _on_resnap(self) -> None:
        if self._model is None or not self._elements:
            return

        try:
            tex_w = int(self._tex_w.value())
            tex_h = int(self._tex_h.value())
            min_face_px = int(self._resnap_min_face.value())
            bias_gt_px = int(self._resnap_bias_gt.value())
            border_pad_px = int(self._resnap_border_pad.value())
            island_pad_px = int(self._resnap_island_pad.value())
            uv_units_sel = str(self._resnap_uv_units.currentData() or "auto")
            uv_units_arg: Optional[str] = None if uv_units_sel == "auto" else uv_units_sel

            out_model, log, units = _resnap_model_uvs(
                model=self._model,
                tex_w=tex_w,
                tex_h=tex_h,
                min_face_px=min_face_px,
                bias_gt_px=bias_gt_px,
                border_pad_px=border_pad_px,
                island_pad_px=island_pad_px,
                uv_units=uv_units_arg,
            )

            self._out_model = out_model
            self._packed = _uvs_to_packed_preview(model=out_model, tex_w=tex_w, tex_h=tex_h, uv_units=units)

            log.extend(_describe_texel_density(packed=self._packed, tex_w=tex_w, tex_h=tex_h))

            self._preview.set_preview(
                tex_w=tex_w,
                tex_h=tex_h,
                packed=self._packed,
                colors_by_element={i: self._element_colors[i] for i in range(len(self._element_colors))},
                selected_element=self._selected_elements,
            )

            self._txt_log.setPlainText("\n".join(log))
            self._txt_json.setPlainText(format_minecraft_model_json(out_model))

            self._btn_save.setEnabled(True)
            self._lbl_status.setText(f"Resnapped {len(self._packed)} elements to {tex_w}x{tex_h} (units={units})")
            self._sync_views()
        except Exception as e:
            self._txt_log.setPlainText(f"Resnap error: {e}")
            self._txt_json.setPlainText("")
            self._btn_save.setEnabled(False)
            self._lbl_status.setText("Resnap error (see log)")
#endregion


#region ENTRY
# Application entry point


def main() -> None:
    app = QtWidgets.QApplication(sys.argv)
    _apply_dark_theme(app)

    w = UVPackerMainWindow()
    w.resize(1400, 820)
    w.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
#endregion
