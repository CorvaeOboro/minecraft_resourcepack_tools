"""Minecraft Procedural Generators (UI)

PySide6 UI tool that generates Minecraft *block model JSON* (`elements`) from
procedural generators and provides an interactive viewport to preview and
compose results.

TOOLSGROUP::MODEL
SORTGROUP::7
SORTPRIORITY::75
STATUS::active
VERSION::20260309

Structure
- UI + persistence: `mc_model_procedural_gen_ui.py` (this file)
- Spiral generators: `mc_model_procedural_gen_shape_spiral.py`
- Helix section generator: `mc_model_procedural_gen_shape_helix.py`
- Shared helpers + export: `mc_model_procedural_gen_common.py`
- Viewport + model-json formatting: `mc_model_solver_ui.py` (`ModelViewport`,
  `format_minecraft_model_json`)

Features
- Shapes
  - Spiral / Double Spiral
  - Double Helix (section)
- Arrangement
  - Radial arrangement (4-way) with per-direction plane + phase controls
  - Instances compositor mode (each instance has its own source + transform)
    - Source: current generated shape, model JSON file, or settings JSON file
    - Per-instance transform: scale + offset (plus global arrangement scale/offset)
- Arrangement projects
  - Save/Load arrangement project JSON (type: `mc_procedural_arrangement_project`)
  - Optional autosave of the loaded project
  - Instance file paths are stored relative to the project when possible
- Per-instance settings JSON
  - Settings JSON files can be either plain settings objects, or wrapped as
    type `mc_procedural_instance_settings` containing a `settings` object
- Post-processing
  - Post scale (about cuboid centroids)
  - Minimum volume culling
  - Depth stagger (small per-element offset along the plane-normal axis)
  - Optional greedy merge pass (merge adjacent aligned cuboids)
- Export
  - Save Base Model JSON or Arrangement Model JSON
  - Texture selection
  - Optional per-element `"shade": false`

"""

from __future__ import annotations

import math
import json
import sys
import traceback
from datetime import datetime
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
    from mc_model_solver_ui import ModelViewport, format_minecraft_model_json, _apply_dark_theme
    from mc_model_solver_ui import Cuboid, Rotation
except Exception as e:
    print(f"Missing dependency: mc_model_solver_ui.py ({e})")
    raise SystemExit(1)

from mc_model_procedural_gen_common import (
    Plane,
    _apply_depth_stagger,
    _cuboid_volume,
    _normalize_fr_to,
    _parse_vec3_str,
    _scale_cuboid_about_centroid,
    _transform_cuboid_global,
    export_minecraft_model_ext,
)
from mc_model_procedural_gen_shape_helix import HELIX_UI_SPECS, HelixSectionConfig, generate_double_helix_section, generate_radial_arrangement_4_helix
from mc_model_procedural_gen_shape_spiral import SPIRAL_UI_SPECS, SpiralConfig, generate_radial_arrangement_4, generate_shape

_INSTANCE_SETTINGS_JSON_TYPE = "mc_procedural_instance_settings"
_INSTANCE_SETTINGS_JSON_VERSION = 1


def _unwrap_instance_settings_json(obj: dict) -> dict:
    if not isinstance(obj, dict):
        raise ValueError("Settings JSON must be an object")
    if str(obj.get("type", "")) != _INSTANCE_SETTINGS_JSON_TYPE:
        return obj
    inner = obj.get("settings", None)
    if not isinstance(inner, dict):
        raise ValueError("Instance settings wrapper must contain a settings object")
    return inner


def _wrap_instance_settings_json(settings: dict) -> dict:
    if not isinstance(settings, dict):
        raise ValueError("Settings JSON must be an object")
    return {
        "type": _INSTANCE_SETTINGS_JSON_TYPE,
        "version": int(_INSTANCE_SETTINGS_JSON_VERSION),
        "settings": dict(settings),
    }


def _build_widget_from_spec(spec: dict) -> QtWidgets.QWidget:
    if not isinstance(spec, dict):
        raise ValueError("UI spec must be a dict")
    typ = str(spec.get("type", ""))
    if typ == "combo":
        w = QtWidgets.QComboBox()
        items = spec.get("items", [])
        if isinstance(items, list):
            for it in list(items):
                if isinstance(it, (tuple, list)) and len(it) == 2:
                    w.addItem(str(it[0]), userData=str(it[1]))
        cur = int(spec.get("current", 0))
        if w.count() > 0:
            cur = 0 if cur < 0 else (w.count() - 1) if cur >= w.count() else cur
            w.setCurrentIndex(int(cur))
        return w
    if typ == "spin":
        w = QtWidgets.QSpinBox()
        w.setRange(int(spec.get("min", 0)), int(spec.get("max", 999999)))
        w.setValue(int(spec.get("value", 0)))
        return w
    if typ == "double":
        w = QtWidgets.QDoubleSpinBox()
        w.setRange(float(spec.get("min", -1e9)), float(spec.get("max", 1e9)))
        w.setValue(float(spec.get("value", 0.0)))
        if "decimals" in spec:
            w.setDecimals(int(spec.get("decimals", 3)))
        return w
    if typ == "check":
        w = QtWidgets.QCheckBox(str(spec.get("text", "")))
        w.setChecked(bool(spec.get("checked", False)))
        return w
    raise ValueError(f"Unknown UI spec type: {typ}")


_UI_GROUP_MARGIN = 6
_UI_FORM_VSPACING = 3
_UI_LAYOUT_SPACING = 4


def _apply_button_scheme(btn: QtWidgets.QPushButton, scheme: str) -> None:
    s = str(scheme)
    if s == "green":
        bg = "#3f6d55"
        bg_h = "#4a7b60"
        bg_p = "#365f4a"
        bd = "#2f4f3e"
    elif s == "blue":
        bg = "#3a5f78"
        bg_h = "#436c88"
        bg_p = "#315165"
        bd = "#284252"
    else:
        bg = "#6a4b7a"
        bg_h = "#77548a"
        bg_p = "#5a3f67"
        bd = "#4a3454"

    btn.setStyleSheet(
        """
QPushButton {
  background-color: %s;
  color: #e9eef2;
  border: 1px solid %s;
  border-radius: 4px;
  padding: 4px 10px;
}
QPushButton:hover {
  background-color: %s;
}
QPushButton:pressed {
  background-color: %s;
}
QPushButton:disabled {
  background-color: #2a2a2a;
  color: #8a8a8a;
  border: 1px solid #3a3a3a;
}
"""
        % (bg, bd, bg_h, bg_p)
    )


def _parse_vec3_relaxed(text: str) -> tuple[float, float, float]:
    raw = str(text or "").strip()
    if not raw:
        return (0.0, 0.0, 0.0)
    if raw[0] in "([{" and raw[-1] in ")]}":
        raw = raw[1:-1].strip()
    try:
        return _parse_vec3_str(raw)
    except Exception:
        parts = [p for p in raw.replace("\t", " ").split(" ") if p.strip()]
        if len(parts) == 3:
            return (float(parts[0]), float(parts[1]), float(parts[2]))
        raise

def _generate_elements_from_settings_dict(settings: dict) -> tuple[list[Cuboid], float]:
    if not isinstance(settings, dict):
        raise ValueError("Settings JSON must be an object")

    settings = _unwrap_instance_settings_json(settings)

    shape = str(settings.get("shape", "spiral"))
    center = _parse_vec3_str(str(settings.get("center", "8,8,8")))

    use_rot = bool(settings.get("use_rotations", True))
    snap_step = float(settings.get("snap_step", 0.25))
    segments = int(settings.get("segments", 128))
    turns = float(settings.get("turns", 2.5))

    depth_plane: str
    base_raw: list[Cuboid]
    if shape == "double_helix_section":
        use_rot_h = bool(settings.get("helix_use_rotations", use_rot))
        snap_step_h = float(settings.get("helix_snap_step", snap_step))
        segments_h = int(settings.get("helix_segments", segments))
        turns_h = float(settings.get("helix_turns", turns))
        cfg_h = HelixSectionConfig(
            center=center,
            axis_plane=str(settings.get("helix_axis_plane", "XZ")),
            heading_deg=float(settings.get("helix_heading_deg", 0.0)),
            segments=int(segments_h),
            length=float(settings.get("helix_length", 16.0)),
            turns=float(turns_h),
            helix_radius=float(settings.get("helix_radius", 3.0)),
            phase_deg=float(settings.get("helix_phase_deg", 0.0)),
            segment_len=float(settings.get("helix_segment_len", 1.0)),
            strand_width=float(settings.get("helix_strand_width", 1.0)),
            strand_depth=float(settings.get("helix_strand_depth", 1.0)),
            scale_x=float(settings.get("helix_scale_x", 1.0)),
            scale_y=float(settings.get("helix_scale_y", 1.0)),
            scale_z=float(settings.get("helix_scale_z", 1.0)),
            snap_step=float(snap_step_h),
            use_rotations=bool(use_rot_h),
            chord_align_rotation=bool(settings.get("helix_chord_align_rotation", False)),
            chord_axis_only=bool(settings.get("helix_chord_axis_only", False)),
            rotation_axis_override=str(settings.get("helix_rotation_axis_override", "auto")),
            chord_align_rotation_offset_deg=float(settings.get("helix_chord_align_rotation_offset_deg", 0.0)),
            auto_merge=bool(settings.get("helix_auto_merge", True)),
            greedy_merge=bool(settings.get("helix_greedy_merge", False)),
            max_merge_len=float(settings.get("helix_max_merge_len", 6.0)),
            target_chord_len=float(settings.get("helix_target_chord_len", 0.75)),
            merge_max_dev=float(settings.get("helix_merge_max_dev", 0.15)),
        )
        base_raw = generate_double_helix_section(cfg_h, phase_offset_rad=0.0)
        depth_plane = str(cfg_h.axis_plane).split(":")[0]
        snap_step = float(cfg_h.snap_step)
    else:
        cfg = SpiralConfig(
            center=center,
            plane=str(settings.get("plane", "XZ")),
            segments=int(segments),
            turns=float(turns),
            guide_rotation_deg=float(settings.get("guide_rotation_deg", 0.0)),
            radius_outer=float(settings.get("radius_outer", 7.5)),
            radius_inner=float(settings.get("radius_inner", 0.75)),
            ramp_height=float(settings.get("ramp_height", 0.0)),
            width_outer=float(settings.get("width_outer", 1.0)),
            width_inner=float(settings.get("width_inner", 0.35)),
            tangent_len_outer=float(settings.get("tangent_len_outer", 2.0)),
            tangent_len_inner=float(settings.get("tangent_len_inner", 0.75)),
            axis_thickness=float(settings.get("axis_thickness", 1.0)),
            snap_step=float(snap_step),
            use_rotations=bool(use_rot),
            reverse_rotation=bool(settings.get("reverse_rotation", False)),
            chord_direction=bool(settings.get("chord_direction", True)),
            auto_merge=bool(settings.get("auto_merge", True)),
            greedy_merge=bool(settings.get("greedy_merge", False)),
            max_merge_len=float(settings.get("max_merge_len", 6.0)),
            max_chord_step=float(settings.get("max_chord_step", 0.75)),
            merge_max_dev=float(settings.get("merge_max_dev", 0.15)),
        )
        base_raw = generate_shape(cfg, shape, phase_offset_rad=0.0)
        depth_plane = str(cfg.plane)
        snap_step = float(cfg.snap_step)

    s = float(settings.get("post_scale", 1.0))
    min_vol = float(settings.get("min_volume", 0.0))
    post: list[Cuboid] = []
    for c in list(base_raw):
        cc = _scale_cuboid_about_centroid(c, s, origin_step=float(snap_step))
        if cc is None:
            continue
        if min_vol > 0.0 and _cuboid_volume(cc) < min_vol:
            continue
        post.append(cc)

    post = _apply_depth_stagger(list(post), plane=str(depth_plane), step=float(settings.get("depth_stagger_step", 0.0)))
    return post, float(snap_step)


def _elements_from_minecraft_model_json(model: dict) -> list[Cuboid]:
    if not isinstance(model, dict):
        raise ValueError("Model JSON must be an object")
    raw_els = model.get("elements", [])
    if raw_els is None:
        raw_els = []
    if not isinstance(raw_els, list):
        raise ValueError("Model JSON elements must be a list")

    out: list[Cuboid] = []
    for el in list(raw_els):
        if not isinstance(el, dict):
            continue
        fr0 = el.get("from", None)
        to0 = el.get("to", None)
        if not (isinstance(fr0, list) and isinstance(to0, list) and len(fr0) == 3 and len(to0) == 3):
            continue
        fr = (float(fr0[0]), float(fr0[1]), float(fr0[2]))
        to = (float(to0[0]), float(to0[1]), float(to0[2]))
        fr2, to2 = _normalize_fr_to(fr, to)

        rot = None
        r0 = el.get("rotation", None)
        if isinstance(r0, dict):
            axis = str(r0.get("axis", ""))
            angle = float(r0.get("angle", 0.0))
            origin0 = r0.get("origin", None)
            if axis in ("x", "y", "z") and isinstance(origin0, list) and len(origin0) == 3:
                origin = (float(origin0[0]), float(origin0[1]), float(origin0[2]))
                rot = Rotation(axis=str(axis), angle=float(angle), origin=origin)

        out.append(Cuboid(fr=fr2, to=to2, rotation=rot))
    return out


def _load_elements_from_model_json_file(path: Path) -> list[Cuboid]:
    raw = path.read_text(encoding="utf-8")
    model = json.loads(raw)
    return _elements_from_minecraft_model_json(model)


class ProceduralShapesMainWindow(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Minecraft Procedural Shapes")

        self._current_elements_base: list[Cuboid] = []
        self._current_model_base: Optional[dict] = None
        self._current_elements_arr: list[Cuboid] = []
        self._current_model_arr: Optional[dict] = None

        self._arr_project_path: Optional[Path] = None

        self._viewport = ModelViewport()

        controls = self._build_controls()
        arrangement = self._build_arrangement_panel()
        output = self._build_output()

        self._install_console_capture()

        controls_scroll = QtWidgets.QScrollArea()
        controls_scroll.setWidgetResizable(True)
        controls_scroll.setWidget(controls)
        controls_scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)

        right_splitter = QtWidgets.QSplitter()
        right_splitter.setOrientation(QtCore.Qt.Orientation.Vertical)
        right_splitter.addWidget(arrangement)
        right_splitter.addWidget(output)
        right_splitter.setStretchFactor(0, 0)
        right_splitter.setStretchFactor(1, 1)
        right_splitter.setCollapsible(0, False)
        right_splitter.setCollapsible(1, False)

        splitter = QtWidgets.QSplitter()
        splitter.setOrientation(QtCore.Qt.Orientation.Horizontal)
        splitter.addWidget(controls_scroll)
        splitter.addWidget(self._viewport)
        splitter.addWidget(right_splitter)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 0)
        splitter.setCollapsible(0, False)
        splitter.setCollapsible(2, False)

        top_bar = self._build_main_actions_bar()
        central = QtWidgets.QWidget()
        central_layout = QtWidgets.QVBoxLayout(central)
        central_layout.setContentsMargins(0, 0, 0, 0)
        central_layout.setSpacing(_UI_LAYOUT_SPACING)
        central_layout.addWidget(top_bar)
        central_layout.addWidget(splitter, 1)
        self.setCentralWidget(central)

        self._arr_instances: list[dict] = []
        self._arr_instances_prev_row = -1
        self._inst_editor_updating = False
        self._init_instances_model()

    def _default_instance_settings_filename(self, idx: int) -> str:
        i = int(idx)
        i = 0 if i < 0 else i
        return f"instance_{i + 1}_settings.json"

    def _default_instance_settings_path(self, idx: int) -> Optional[Path]:
        base = self._arr_project_dir()
        if base is None:
            return None
        return (base / self._default_instance_settings_filename(int(idx))).resolve()

    def _write_instance_settings_file(self, path: Path, settings: dict) -> None:
        data = _wrap_instance_settings_json(settings)
        path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")

    def _build_output(self) -> QtWidgets.QWidget:
        w = QtWidgets.QWidget()
        w.setMinimumWidth(460)

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

        self._txt_log = QtWidgets.QPlainTextEdit()
        self._txt_log.setReadOnly(True)
        self._txt_log.setMaximumBlockCount(20000)
        self._txt_log.setLineWrapMode(QtWidgets.QPlainTextEdit.LineWrapMode.NoWrap)
        self._txt_log.setFont(QtGui.QFont("Consolas", 10))

        self._btn_clear_log = QtWidgets.QPushButton("Clear Log")
        _apply_button_scheme(self._btn_clear_log, "purple")
        self._btn_clear_log.clicked.connect(lambda: self._txt_log.setPlainText(""))

        tabs = QtWidgets.QTabWidget()
        tabs.addTab(self._txt_elements, "Elements")
        tabs.addTab(self._txt_model_json, "Model JSON")
        tabs.addTab(self._txt_log, "Log")

        box = QtWidgets.QGroupBox("Output")
        box_layout = QtWidgets.QVBoxLayout(box)
        box_layout.setContentsMargins(_UI_GROUP_MARGIN, _UI_GROUP_MARGIN, _UI_GROUP_MARGIN, _UI_GROUP_MARGIN)
        box_layout.setSpacing(_UI_LAYOUT_SPACING)
        box_layout.addWidget(self._btn_clear_log)
        box_layout.addWidget(tabs)

        layout = QtWidgets.QVBoxLayout(w)
        layout.addWidget(box)
        layout.setSpacing(_UI_LAYOUT_SPACING)
        layout.setContentsMargins(0, 0, 0, 0)
        return w

    def _build_main_actions_bar(self) -> QtWidgets.QWidget:
        w = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(w)
        layout.setContentsMargins(_UI_GROUP_MARGIN, _UI_GROUP_MARGIN, _UI_GROUP_MARGIN, _UI_GROUP_MARGIN)
        layout.setSpacing(_UI_LAYOUT_SPACING)
        layout.addStretch(1)
        layout.addWidget(self._btn_generate)
        layout.addWidget(self._btn_clear)
        layout.addWidget(self._btn_save_base)
        layout.addWidget(self._btn_save_arr)
        layout.addWidget(self._btn_save_settings)
        layout.addWidget(self._btn_load_settings)
        layout.addStretch(1)
        return w

    def _sync_arr_mode_ui(self) -> None:
        mode = str(self._arr_mode.currentData() or "radial4")
        if hasattr(self, "_inst_box") and self._inst_box is not None:
            self._inst_box.setEnabled(bool(mode == "instances"))
        self._on_instance_source_changed()
        self._maybe_update_arr_preview()

    def _sync_shape_sections(self) -> None:
        shape = str(self._shape_combo.currentData() or "spiral")
        if hasattr(self, "_spiral_box") and self._spiral_box is not None:
            self._spiral_box.setChecked(bool(shape != "double_helix_section"))
            if hasattr(self, "_spiral_body") and self._spiral_body is not None:
                self._spiral_body.setVisible(bool(self._spiral_box.isChecked()))
        if hasattr(self, "_helix_box") and self._helix_box is not None:
            self._helix_box.setChecked(bool(shape == "double_helix_section"))
            if hasattr(self, "_helix_body") and self._helix_body is not None:
                self._helix_body.setVisible(bool(self._helix_box.isChecked()))

    def _on_view_mode_changed(self) -> None:
        view = str(self._view_mode.currentData() or "base")
        self._arr_preview.blockSignals(True)
        try:
            self._arr_preview.setChecked(bool(view == "arr"))
        finally:
            self._arr_preview.blockSignals(False)

        if view == "arr":
            self._arr_enable.setChecked(True)
        self._on_arr_settings_changed()

    def _on_arr_preview_toggled(self, state: int) -> None:
        want_arr = bool(state)
        self._view_mode.blockSignals(True)
        try:
            self._set_combo_data(self._view_mode, "arr" if want_arr else "base")
        finally:
            self._view_mode.blockSignals(False)
        if want_arr:
            self._arr_enable.setChecked(True)
        self._on_arr_settings_changed()

    def _default_instance(self, *, n: int) -> dict:
        return {
            "name": f"Instance {int(n)}",
            "enabled": True,
            "source": "current",
            "path": "",
            "scale": 1.0,
            "offset": "0,0,0",
        }

    def _init_instances_model(self) -> None:
        self._arr_instances = [self._default_instance(n=1)]
        self._refresh_instances_ui(select_row=0)

    def _refresh_instances_ui(self, *, select_row: int) -> None:
        self._inst_editor_updating = True
        try:
            self._inst_list.clear()
            for inst in list(self._arr_instances):
                name = str(inst.get("name", "Instance"))
                on = bool(inst.get("enabled", True))
                tag = "[on]" if on else "[off]"
                self._inst_list.addItem(f"{tag} {name}")

            if self._inst_list.count() <= 0:
                self._inst_list.setCurrentRow(-1)
                self._arr_instances_prev_row = -1
            else:
                r = int(select_row)
                r = 0 if r < 0 else (self._inst_list.count() - 1) if r >= self._inst_list.count() else r
                self._inst_list.setCurrentRow(r)
                self._arr_instances_prev_row = int(r)
                self._load_instance_to_editor(r)
        finally:
            self._inst_editor_updating = False
        self._sync_arr_mode_ui()

    def _selected_instance_index(self) -> int:
        return int(self._inst_list.currentRow())

    def _load_instance_to_editor(self, idx: int) -> None:
        if idx < 0 or idx >= len(self._arr_instances):
            return
        inst = dict(self._arr_instances[int(idx)])
        self._inst_name.setText(str(inst.get("name", "")))
        self._inst_enabled.setChecked(bool(inst.get("enabled", True)))
        self._set_combo_data(self._inst_source, str(inst.get("source", "current")))
        self._inst_path.setText(str(inst.get("path", "")))
        self._inst_scale.setValue(float(inst.get("scale", 1.0)))
        self._inst_offset.setText(str(inst.get("offset", "0,0,0")))
        self._on_instance_source_changed()

    def _store_editor_to_instance(self, idx: int) -> None:
        if idx < 0 or idx >= len(self._arr_instances):
            return
        inst = dict(self._arr_instances[int(idx)])
        inst["name"] = str(self._inst_name.text())
        inst["enabled"] = bool(self._inst_enabled.isChecked())
        inst["source"] = str(self._inst_source.currentData() or "current")
        inst["path"] = str(self._inst_path.text())
        inst["scale"] = float(self._inst_scale.value())
        inst["offset"] = str(self._inst_offset.text())
        self._arr_instances[int(idx)] = inst

    def _on_instance_selected(self, row: int) -> None:
        if bool(self._inst_editor_updating):
            return
        prev = int(self._arr_instances_prev_row)
        if prev >= 0:
            self._store_editor_to_instance(prev)
        self._arr_instances_prev_row = int(row)
        self._inst_editor_updating = True
        try:
            self._load_instance_to_editor(int(row))
        finally:
            self._inst_editor_updating = False
        self._refresh_instances_list_labels()
        self._maybe_autosave_arr_project()

    def _refresh_instances_list_labels(self) -> None:
        for i in range(self._inst_list.count()):
            if i < 0 or i >= len(self._arr_instances):
                continue
            inst = self._arr_instances[int(i)]
            name = str(inst.get("name", "Instance"))
            on = bool(inst.get("enabled", True))
            tag = "[on]" if on else "[off]"
            item = self._inst_list.item(int(i))
            if item is not None:
                item.setText(f"{tag} {name}")

    def _on_instance_add(self) -> None:
        n = len(self._arr_instances) + 1
        self._arr_instances.append(self._default_instance(n=n))
        self._refresh_instances_ui(select_row=len(self._arr_instances) - 1)
        self._maybe_autosave_arr_project()

    def _on_instance_remove(self) -> None:
        idx = self._selected_instance_index()
        if idx < 0 or idx >= len(self._arr_instances):
            return
        self._store_editor_to_instance(idx)
        del self._arr_instances[int(idx)]
        if len(self._arr_instances) <= 0:
            self._arr_instances = [self._default_instance(n=1)]
            idx = 0
        idx2 = idx
        if idx2 >= len(self._arr_instances):
            idx2 = len(self._arr_instances) - 1
        self._refresh_instances_ui(select_row=idx2)
        self._maybe_autosave_arr_project()

    def _on_instance_browse(self) -> None:
        src = str(self._inst_source.currentData() or "current")
        if src == "current":
            return
        start = self._inst_path.text().strip() or str(Path.cwd())
        path, _filter = QtWidgets.QFileDialog.getOpenFileName(self, "Select JSON file", start, "JSON (*.json)")
        if not path:
            return
        self._inst_path.setText(str(path))
        self._maybe_autosave_arr_project()

    def _on_instance_source_changed(self) -> None:
        if not hasattr(self, "_inst_path"):
            return
        src = str(self._inst_source.currentData() or "current")
        need_file = bool(src in ("model_json", "settings_json"))
        self._inst_path.setEnabled(need_file)
        self._btn_inst_browse.setEnabled(need_file)
        self._maybe_update_arr_preview()

    def _on_instance_editor_changed(self, *_args) -> None:
        if bool(self._inst_editor_updating):
            return
        idx = self._selected_instance_index()
        if idx < 0:
            return
        self._store_editor_to_instance(idx)
        self._refresh_instances_list_labels()
        self._maybe_update_arr_preview()
        self._maybe_autosave_arr_project()

    def _on_arr_settings_changed(self, *_args) -> None:
        self._maybe_update_arr_preview()
        self._maybe_autosave_arr_project()
        view = str(self._view_mode.currentData() or "base")
        if bool(self._arr_enable.isChecked()) and view == "arr":
            return
        self._refresh_viewport_from_mode()

    def _maybe_update_arr_preview(self) -> None:
        if not bool(self._arr_enable.isChecked()):
            return
        view = str(self._view_mode.currentData() or "base")
        if view != "arr":
            return
        try:
            shape = str(self._shape_combo.currentData() or "spiral")
            arr = self._compute_arrangement_elements(shape)
        except Exception:
            try:
                print("Arrangement preview failed:")
                print(traceback.format_exc())
                self._lbl_status.setText("Arrangement preview failed (see Log)")
            except Exception:
                return
            return

        self._current_elements_arr = list(arr)
        tex = self._texture.text().strip() or "minecraft:block/oak_planks"
        shade_false = bool(self._export_shade_false.isChecked())
        self._current_model_arr = export_minecraft_model_ext(elements=self._current_elements_arr, texture=tex, shade_false=shade_false) if len(self._current_elements_arr) > 0 else None
        self._btn_save_arr.setEnabled(self._current_model_arr is not None)
        self._lbl_status.setText(f"Generated: {len(self._current_elements_arr)} elements")
        self._refresh_viewport_from_mode()

    def _refresh_viewport_from_mode(self) -> None:
        view = str(self._view_mode.currentData() or "base")
        show_arr = bool(view == "arr") and bool(self._arr_enable.isChecked())

        show_elements = self._current_elements_arr if show_arr else self._current_elements_base
        self._viewport.set_cuboids(show_elements)

        lines: list[str] = []
        for i, c in enumerate(show_elements):
            lines.append(f"{i:03d} from=({c.fr[0]:g},{c.fr[1]:g},{c.fr[2]:g}) to=({c.to[0]:g},{c.to[1]:g},{c.to[2]:g})")
        self._txt_elements.setPlainText("\n".join(lines))

        show_model = self._current_model_arr if show_arr and self._current_model_arr is not None else self._current_model_base
        if show_model is not None:
            self._txt_model_json.setPlainText(format_minecraft_model_json(show_model))

    def _compute_arrangement_elements(self, shape: str) -> list[Cuboid]:
        mode = str(self._arr_mode.currentData() or "radial4")

        arr: list[Cuboid] = []
        if mode == "instances":
            pivot = (8.0, 0.0, 8.0)
            composed: list[Cuboid] = []
            prev = int(self._arr_instances_prev_row)
            if prev >= 0:
                self._store_editor_to_instance(prev)
                self._refresh_instances_list_labels()

            for inst in list(self._arr_instances):
                if not bool(inst.get("enabled", True)):
                    continue
                src = str(inst.get("source", "current"))
                els: list[Cuboid]
                if src == "current":
                    els = list(self._current_elements_base)
                elif src == "model_json":
                    p = Path(str(inst.get("path", "")))
                    if not p.exists():
                        raise ValueError(f"Missing model JSON: {p}")
                    els = _load_elements_from_model_json_file(p)
                elif src == "settings_json":
                    p = Path(str(inst.get("path", "")))
                    if not p.exists():
                        raise ValueError(f"Missing settings JSON: {p}")
                    raw = p.read_text(encoding="utf-8")
                    sdict = json.loads(raw)
                    els, _snap = _generate_elements_from_settings_dict(sdict)
                else:
                    continue

                iscale = float(inst.get("scale", 1.0))
                ioff = _parse_vec3_relaxed(str(inst.get("offset", "0,0,0")))
                for c in list(els):
                    cc = _transform_cuboid_global(c, scale=iscale, offset=ioff, pivot=pivot)
                    if cc is not None:
                        composed.append(cc)
            arr = composed
        else:
            planes = (
                str(self._arr_plane_pX.currentData() or "XZ"),
                str(self._arr_plane_pZ.currentData() or "YZ"),
                str(self._arr_plane_nX.currentData() or "XZ"),
                str(self._arr_plane_nZ.currentData() or "YZ"),
            )
            phases = (
                float(self._arr_phase_pX.value()),
                float(self._arr_phase_pZ.value()),
                float(self._arr_phase_nX.value()),
                float(self._arr_phase_nZ.value()),
            )
            if str(shape) == "double_helix_section":
                cfg_h2 = self._make_helix_config()
                heading_offsets = (0.0, 90.0, 180.0, 270.0) if bool(self._arr_helix_heading_by_side.isChecked()) else (0.0, 0.0, 0.0, 0.0)
                arr_raw = generate_radial_arrangement_4_helix(
                    cfg_h2,
                    radius=float(self._arr_radius.value()),
                    planes=planes,
                    phase_offsets_deg=phases,
                    heading_offsets_deg=heading_offsets,
                    depth_stagger_step=float(self._depth_stagger.value()),
                )
                arr = self._postprocess_elements(list(arr_raw), snap_step=float(cfg_h2.snap_step))
            else:
                cfg2 = self._make_spiral_config()
                arr_raw = generate_radial_arrangement_4(
                    cfg2,
                    str(shape),
                    radius=float(self._arr_radius.value()),
                    planes=planes,
                    phase_offsets_deg=phases,
                    depth_stagger_step=float(self._depth_stagger.value()),
                )
                arr = self._postprocess_elements(list(arr_raw), snap_step=float(cfg2.snap_step))

        gscale = float(self._arr_global_scale.value())
        goff = _parse_vec3_relaxed(self._arr_global_offset.text())
        pivot = (8.0, 0.0, 8.0)

        transformed: list[Cuboid] = []
        for c in list(arr):
            cc = _transform_cuboid_global(c, scale=gscale, offset=goff, pivot=pivot)
            if cc is not None:
                transformed.append(cc)
        return transformed

    def _append_log_text(self, text: str) -> None:
        if not hasattr(self, "_txt_log") or self._txt_log is None:
            return
        t = str(text)
        if not t:
            return
        if "\n" in t:
            parts = t.splitlines()
            for p in parts:
                if p != "":
                    self._txt_log.appendPlainText(p)
        else:
            if t != "":
                self._txt_log.appendPlainText(t)

    def _install_console_capture(self) -> None:
        class _QtStream(QtCore.QObject):
            text_emitted = QtCore.Signal(str)

            def write(self, s: str) -> None:
                self.text_emitted.emit(str(s))

            def flush(self) -> None:
                return

        self._orig_stdout = sys.stdout
        self._orig_stderr = sys.stderr

        self._qt_stdout = _QtStream(self)
        self._qt_stderr = _QtStream(self)
        self._qt_stdout.text_emitted.connect(self._append_log_text)
        self._qt_stderr.text_emitted.connect(self._append_log_text)

        sys.stdout = self._qt_stdout
        sys.stderr = self._qt_stderr

    def _build_arrangement_panel(self) -> QtWidgets.QWidget:
        w = QtWidgets.QWidget()
        w.setMinimumWidth(460)

        proj_box = QtWidgets.QGroupBox("Arrangement Project")
        proj_layout = QtWidgets.QVBoxLayout(proj_box)
        proj_layout.setContentsMargins(_UI_GROUP_MARGIN, _UI_GROUP_MARGIN, _UI_GROUP_MARGIN, _UI_GROUP_MARGIN)
        proj_layout.setSpacing(_UI_LAYOUT_SPACING)

        proj_layout.addWidget(self._lbl_arr_project)

        proj_btn_row = QtWidgets.QHBoxLayout()
        proj_btn_row.setSpacing(_UI_LAYOUT_SPACING)
        proj_btn_row.addWidget(self._btn_arr_project_save)
        proj_btn_row.addWidget(self._btn_arr_project_save_as)
        proj_btn_row.addWidget(self._btn_arr_project_load)
        proj_layout.addLayout(proj_btn_row)
        proj_layout.addWidget(self._arr_project_autosave)

        _apply_button_scheme(self._btn_arr_project_save, "green")
        _apply_button_scheme(self._btn_arr_project_save_as, "green")
        _apply_button_scheme(self._btn_arr_project_load, "blue")

        arr_box = QtWidgets.QGroupBox("Arrangement")
        arr_form = QtWidgets.QFormLayout(arr_box)
        arr_form.setContentsMargins(_UI_GROUP_MARGIN, _UI_GROUP_MARGIN, _UI_GROUP_MARGIN, _UI_GROUP_MARGIN)
        arr_form.setVerticalSpacing(_UI_FORM_VSPACING)
        arr_form.addRow("", self._arr_enable)
        arr_form.addRow("", self._arr_preview)
        arr_form.addRow("Mode", self._arr_mode)
        arr_form.addRow("Radius", self._arr_radius)
        arr_form.addRow("", self._arr_helix_heading_by_side)
        row_px = QtWidgets.QWidget(arr_box)
        row_px_l = QtWidgets.QHBoxLayout(row_px)
        row_px_l.setContentsMargins(0, 0, 0, 0)
        row_px_l.addWidget(self._arr_plane_pX)
        row_px_l.addWidget(self._arr_phase_pX)
        arr_form.addRow("+X", row_px)

        row_nx = QtWidgets.QWidget(arr_box)
        row_nx_l = QtWidgets.QHBoxLayout(row_nx)
        row_nx_l.setContentsMargins(0, 0, 0, 0)
        row_nx_l.addWidget(self._arr_plane_nX)
        row_nx_l.addWidget(self._arr_phase_nX)
        arr_form.addRow("-X", row_nx)

        row_pz = QtWidgets.QWidget(arr_box)
        row_pz_l = QtWidgets.QHBoxLayout(row_pz)
        row_pz_l.setContentsMargins(0, 0, 0, 0)
        row_pz_l.addWidget(self._arr_plane_pZ)
        row_pz_l.addWidget(self._arr_phase_pZ)
        arr_form.addRow("+Z", row_pz)

        row_nz = QtWidgets.QWidget(arr_box)
        row_nz_l = QtWidgets.QHBoxLayout(row_nz)
        row_nz_l.setContentsMargins(0, 0, 0, 0)
        row_nz_l.addWidget(self._arr_plane_nZ)
        row_nz_l.addWidget(self._arr_phase_nZ)
        arr_form.addRow("-Z", row_nz)
        arr_form.addRow("Global scale (pivot 8,0,8)", self._arr_global_scale)
        arr_form.addRow("Global offset", self._arr_global_offset)

        inst_box = QtWidgets.QGroupBox("Instances")
        self._inst_box = inst_box
        inst_layout = QtWidgets.QVBoxLayout(inst_box)
        inst_layout.setContentsMargins(_UI_GROUP_MARGIN, _UI_GROUP_MARGIN, _UI_GROUP_MARGIN, _UI_GROUP_MARGIN)
        inst_layout.setSpacing(_UI_LAYOUT_SPACING)

        inst_layout.addWidget(self._inst_list)

        inst_btn_row = QtWidgets.QHBoxLayout()
        inst_btn_row.setSpacing(_UI_LAYOUT_SPACING)
        inst_btn_row.addWidget(self._btn_inst_add)
        inst_btn_row.addWidget(self._btn_inst_remove)
        inst_btn_row.addStretch(1)
        inst_layout.addLayout(inst_btn_row)

        _apply_button_scheme(self._btn_inst_add, "green")
        _apply_button_scheme(self._btn_inst_remove, "purple")

        inst_form = QtWidgets.QFormLayout()
        inst_form.setVerticalSpacing(_UI_FORM_VSPACING)
        inst_form.addRow("Name", self._inst_name)
        inst_form.addRow("", self._inst_enabled)
        inst_form.addRow("Source", self._inst_source)

        row_path = QtWidgets.QWidget(inst_box)
        row_path_l = QtWidgets.QHBoxLayout(row_path)
        row_path_l.setContentsMargins(0, 0, 0, 0)
        row_path_l.setSpacing(_UI_LAYOUT_SPACING)
        row_path_l.addWidget(self._inst_path)
        row_path_l.addWidget(self._btn_inst_browse)
        inst_form.addRow("File", row_path)

        _apply_button_scheme(self._btn_inst_browse, "blue")

        inst_form.addRow("Scale", self._inst_scale)
        inst_form.addRow("Offset", self._inst_offset)
        inst_layout.addLayout(inst_form)

        layout = QtWidgets.QVBoxLayout(w)
        layout.addWidget(proj_box)
        layout.addWidget(arr_box)
        layout.addWidget(inst_box)
        layout.addStretch(1)
        layout.setSpacing(_UI_LAYOUT_SPACING)
        layout.setContentsMargins(0, 0, 0, 0)
        self._sync_arr_mode_ui()
        return w

    def _build_controls(self) -> QtWidgets.QWidget:
        w = QtWidgets.QWidget()
        w.setMinimumWidth(380)

        self._shape_combo = QtWidgets.QComboBox()
        self._shape_combo.addItem("Spiral", userData="spiral")
        self._shape_combo.addItem("Double Spiral", userData="double_spiral")
        self._shape_combo.addItem("Double Helix (section)", userData="double_helix_section")

        self._center = QtWidgets.QLineEdit("8,8,8")

        spiral_map = {
            "plane": "_plane",
            "segments": "_segments",
            "turns": "_turns",
            "guide_rotation": "_guide_rotation",
            "radius_outer": "_radius_outer",
            "radius_inner": "_radius_inner",
            "ramp_height": "_ramp_height",
            "width_outer": "_width_outer",
            "width_inner": "_width_inner",
            "tangent_outer": "_tangent_outer",
            "tangent_inner": "_tangent_inner",
            "axis_thickness": "_axis_thickness",
            "snap_step": "_snap_step",
            "use_rotations": "_use_rotations",
            "reverse_rotation": "_reverse_rotation",
            "chord_direction": "_chord_direction",
            "auto_merge": "_auto_merge",
            "greedy_merge": "_greedy_merge",
            "max_merge_len": "_max_merge_len",
            "max_chord_step": "_max_chord_step",
            "merge_max_dev": "_merge_max_dev",
        }
        for k, attr in spiral_map.items():
            setattr(self, str(attr), _build_widget_from_spec(dict(SPIRAL_UI_SPECS.get(str(k), {}))))

        helix_map = {
            "axis_plane": "_helix_axis_plane",
            "heading": "_helix_heading",
            "length": "_helix_length",
            "radius": "_helix_radius",
            "phase": "_helix_phase",
            "segment_len": "_helix_segment_len",
            "strand_width": "_helix_strand_width",
            "strand_depth": "_helix_strand_depth",
            "scale_x": "_helix_scale_x",
            "scale_y": "_helix_scale_y",
            "scale_z": "_helix_scale_z",
            "segments": "_helix_segments",
            "turns": "_helix_turns",
            "snap_step": "_helix_snap_step",
            "use_rotations": "_helix_use_rotations",
            "chord_align_rotation": "_helix_chord_align_rotation",
            "chord_axis_only": "_helix_chord_axis_only",
            "rotation_axis_override": "_helix_rotation_axis_override",
            "chord_align_rotation_offset": "_helix_chord_align_rotation_offset",
            "auto_merge": "_helix_auto_merge",
            "greedy_merge": "_helix_greedy_merge",
            "max_merge_len": "_helix_max_merge_len",
            "target_chord_len": "_helix_target_chord_len",
            "merge_max_dev": "_helix_merge_max_dev",
        }
        for k, attr in helix_map.items():
            setattr(self, str(attr), _build_widget_from_spec(dict(HELIX_UI_SPECS.get(str(k), {}))))

        self._texture = QtWidgets.QLineEdit("minecraft:block/oak_planks")

        self._export_shade_false = QtWidgets.QCheckBox("Emit shade=false")
        self._export_shade_false.setChecked(False)

        self._post_scale = QtWidgets.QDoubleSpinBox()
        self._post_scale.setRange(0.01, 2.0)
        self._post_scale.setValue(1.0)
        self._post_scale.setDecimals(4)

        self._min_volume = QtWidgets.QDoubleSpinBox()
        self._min_volume.setRange(0.0, 4096.0)
        self._min_volume.setValue(0.0)
        self._min_volume.setDecimals(6)

        self._depth_stagger = QtWidgets.QDoubleSpinBox()
        self._depth_stagger.setRange(-1.0, 1.0)
        self._depth_stagger.setValue(0.0)
        self._depth_stagger.setDecimals(6)

        self._arr_enable = QtWidgets.QCheckBox("Radial arrangement (4-way)")
        self._arr_enable.setChecked(False)

        self._arr_preview = QtWidgets.QCheckBox("Preview arrangement")
        self._arr_preview.setChecked(False)

        self._arr_mode = QtWidgets.QComboBox()
        self._arr_mode.addItem("Radial 4-way", userData="radial4")
        self._arr_mode.addItem("Instances", userData="instances")
        self._arr_mode.setCurrentIndex(0)

        self._arr_helix_heading_by_side = QtWidgets.QCheckBox("Helix: rotate heading by side")
        self._arr_helix_heading_by_side.setChecked(True)

        self._inst_list = QtWidgets.QListWidget()
        self._inst_list.setMinimumHeight(120)

        self._btn_inst_add = QtWidgets.QPushButton("Add")
        self._btn_inst_remove = QtWidgets.QPushButton("Remove")

        self._inst_name = QtWidgets.QLineEdit("")
        self._inst_enabled = QtWidgets.QCheckBox("Enabled")
        self._inst_enabled.setChecked(True)

        self._inst_source = QtWidgets.QComboBox()
        self._inst_source.addItem("Current shape", userData="current")
        self._inst_source.addItem("Model JSON file", userData="model_json")
        self._inst_source.addItem("Settings JSON file", userData="settings_json")
        self._inst_source.setCurrentIndex(0)

        self._inst_path = QtWidgets.QLineEdit("")
        self._btn_inst_browse = QtWidgets.QPushButton("Browse")

        self._inst_scale = QtWidgets.QDoubleSpinBox()
        self._inst_scale.setRange(0.01, 64.0)
        self._inst_scale.setValue(1.0)
        self._inst_scale.setDecimals(4)

        self._inst_offset = QtWidgets.QLineEdit("0,0,0")

        self._arr_global_scale = QtWidgets.QDoubleSpinBox()
        self._arr_global_scale.setRange(0.01, 64.0)
        self._arr_global_scale.setValue(1.0)
        self._arr_global_scale.setDecimals(4)

        self._arr_global_offset = QtWidgets.QLineEdit("0,0,0")

        self._arr_radius = QtWidgets.QDoubleSpinBox()
        self._arr_radius.setRange(0.0, 256.0)
        self._arr_radius.setValue(8.0)
        self._arr_radius.setDecimals(3)

        self._arr_plane_pX = QtWidgets.QComboBox()
        self._arr_plane_pX.addItem("XZ (around Y)", userData="XZ")
        self._arr_plane_pX.addItem("XY (around Z)", userData="XY")
        self._arr_plane_pX.addItem("YZ (around X)", userData="YZ")
        self._arr_plane_pX.setCurrentIndex(0)

        self._arr_plane_nX = QtWidgets.QComboBox()
        self._arr_plane_nX.addItem("XZ (around Y)", userData="XZ")
        self._arr_plane_nX.addItem("XY (around Z)", userData="XY")
        self._arr_plane_nX.addItem("YZ (around X)", userData="YZ")
        self._arr_plane_nX.setCurrentIndex(0)

        self._arr_plane_pZ = QtWidgets.QComboBox()
        self._arr_plane_pZ.addItem("XZ (around Y)", userData="XZ")
        self._arr_plane_pZ.addItem("XY (around Z)", userData="XY")
        self._arr_plane_pZ.addItem("YZ (around X)", userData="YZ")
        self._arr_plane_pZ.setCurrentIndex(2)

        self._arr_plane_nZ = QtWidgets.QComboBox()
        self._arr_plane_nZ.addItem("XZ (around Y)", userData="XZ")
        self._arr_plane_nZ.addItem("XY (around Z)", userData="XY")
        self._arr_plane_nZ.addItem("YZ (around X)", userData="YZ")
        self._arr_plane_nZ.setCurrentIndex(2)

        self._arr_phase_pX = QtWidgets.QDoubleSpinBox()
        self._arr_phase_pX.setRange(-360.0, 360.0)
        self._arr_phase_pX.setValue(0.0)
        self._arr_phase_pX.setDecimals(3)

        self._arr_phase_nX = QtWidgets.QDoubleSpinBox()
        self._arr_phase_nX.setRange(-360.0, 360.0)
        self._arr_phase_nX.setValue(180.0)
        self._arr_phase_nX.setDecimals(3)

        self._arr_phase_pZ = QtWidgets.QDoubleSpinBox()
        self._arr_phase_pZ.setRange(-360.0, 360.0)
        self._arr_phase_pZ.setValue(90.0)
        self._arr_phase_pZ.setDecimals(3)

        self._arr_phase_nZ = QtWidgets.QDoubleSpinBox()
        self._arr_phase_nZ.setRange(-360.0, 360.0)
        self._arr_phase_nZ.setValue(270.0)
        self._arr_phase_nZ.setDecimals(3)

        self._lbl_arr_project = QtWidgets.QLabel("Project: (none)")
        self._lbl_arr_project.setWordWrap(True)

        self._arr_project_autosave = QtWidgets.QCheckBox("Autosave project")
        self._arr_project_autosave.setChecked(True)

        self._btn_arr_project_save = QtWidgets.QPushButton("Save Project")
        self._btn_arr_project_save_as = QtWidgets.QPushButton("Save Project As")
        self._btn_arr_project_load = QtWidgets.QPushButton("Load Project")

        self._vp_faces = QtWidgets.QCheckBox("Faces")
        self._vp_faces.setChecked(True)
        self._vp_translucent = QtWidgets.QCheckBox("Translucent faces")
        self._vp_translucent.setChecked(False)
        self._vp_wireframe = QtWidgets.QCheckBox("Wireframe")
        self._vp_wireframe.setChecked(True)

        self._view_mode = QtWidgets.QComboBox()
        self._view_mode.addItem("Base", userData="base")
        self._view_mode.addItem("Arrangement", userData="arr")
        self._view_mode.setCurrentIndex(0)

        self._vp_faces.stateChanged.connect(self._sync_viewport_render_options)
        self._vp_translucent.stateChanged.connect(self._sync_viewport_render_options)
        self._vp_wireframe.stateChanged.connect(self._sync_viewport_render_options)

        self._view_mode.currentIndexChanged.connect(self._on_view_mode_changed)

        self._arr_mode.currentIndexChanged.connect(self._sync_arr_mode_ui)

        self._arr_enable.stateChanged.connect(self._on_arr_settings_changed)
        self._arr_preview.stateChanged.connect(self._on_arr_preview_toggled)
        self._arr_global_scale.valueChanged.connect(self._on_arr_settings_changed)
        self._arr_global_offset.textChanged.connect(self._on_arr_settings_changed)
        self._arr_radius.valueChanged.connect(self._on_arr_settings_changed)
        self._arr_plane_pX.currentIndexChanged.connect(self._on_arr_settings_changed)
        self._arr_plane_nX.currentIndexChanged.connect(self._on_arr_settings_changed)
        self._arr_plane_pZ.currentIndexChanged.connect(self._on_arr_settings_changed)
        self._arr_plane_nZ.currentIndexChanged.connect(self._on_arr_settings_changed)
        self._arr_phase_pX.valueChanged.connect(self._on_arr_settings_changed)
        self._arr_phase_nX.valueChanged.connect(self._on_arr_settings_changed)
        self._arr_phase_pZ.valueChanged.connect(self._on_arr_settings_changed)
        self._arr_phase_nZ.valueChanged.connect(self._on_arr_settings_changed)
        self._arr_helix_heading_by_side.stateChanged.connect(self._on_arr_settings_changed)

        self._inst_list.currentRowChanged.connect(self._on_instance_selected)
        self._btn_inst_add.clicked.connect(self._on_instance_add)
        self._btn_inst_remove.clicked.connect(self._on_instance_remove)
        self._btn_inst_browse.clicked.connect(self._on_instance_browse)

        self._inst_name.textChanged.connect(self._on_instance_editor_changed)
        self._inst_enabled.stateChanged.connect(self._on_instance_editor_changed)
        self._inst_source.currentIndexChanged.connect(self._on_instance_source_changed)
        self._inst_path.textChanged.connect(self._on_instance_editor_changed)
        self._inst_scale.valueChanged.connect(self._on_instance_editor_changed)
        self._inst_offset.textChanged.connect(self._on_instance_editor_changed)

        self._shape_combo.currentIndexChanged.connect(self._sync_shape_sections)

        self._btn_generate = QtWidgets.QPushButton("Generate")
        self._btn_generate.setObjectName("btn_generate")
        self._btn_generate.clicked.connect(self._on_generate)

        self._btn_clear = QtWidgets.QPushButton("Clear")
        self._btn_clear.clicked.connect(self._on_clear)

        self._btn_save_base = QtWidgets.QPushButton("Save Base Model JSON")
        self._btn_save_base.setObjectName("btn_save")
        self._btn_save_base.clicked.connect(self._on_save_base)

        self._btn_save_arr = QtWidgets.QPushButton("Save Arrangement Model JSON")
        self._btn_save_arr.clicked.connect(self._on_save_arr)

        self._btn_save_settings = QtWidgets.QPushButton("Save Settings")
        self._btn_save_settings.clicked.connect(self._on_save_settings)

        self._btn_load_settings = QtWidgets.QPushButton("Load Settings")
        self._btn_load_settings.clicked.connect(self._on_load_settings)

        self._btn_arr_project_save.clicked.connect(self._on_save_arr_project)
        self._btn_arr_project_save_as.clicked.connect(self._on_save_arr_project_as)
        self._btn_arr_project_load.clicked.connect(self._on_load_arr_project)

        self._lbl_status = QtWidgets.QLabel("Ready")
        self._lbl_status.setWordWrap(True)

        shape_box = QtWidgets.QGroupBox("Shape")
        shape_layout = QtWidgets.QVBoxLayout(shape_box)
        shape_layout.setContentsMargins(_UI_GROUP_MARGIN, _UI_GROUP_MARGIN, _UI_GROUP_MARGIN, _UI_GROUP_MARGIN)
        shape_layout.setSpacing(_UI_LAYOUT_SPACING)
        shape_layout.addWidget(self._shape_combo)

        common_box = QtWidgets.QGroupBox("Common")
        common_form = QtWidgets.QFormLayout(common_box)
        common_form.setContentsMargins(_UI_GROUP_MARGIN, _UI_GROUP_MARGIN, _UI_GROUP_MARGIN, _UI_GROUP_MARGIN)
        common_form.setVerticalSpacing(_UI_FORM_VSPACING)
        common_form.addRow("Center", self._center)
        common_form.addRow("Post scale", self._post_scale)
        common_form.addRow("Min volume", self._min_volume)
        common_form.addRow("Depth stagger", self._depth_stagger)

        spiral_box = QtWidgets.QGroupBox("Spiral")
        self._spiral_box = spiral_box
        spiral_box.setCheckable(True)
        spiral_box.setChecked(True)
        spiral_body = QtWidgets.QWidget(spiral_box)
        self._spiral_body = spiral_body
        spiral_body_l = QtWidgets.QVBoxLayout(spiral_box)
        spiral_body_l.setContentsMargins(_UI_GROUP_MARGIN, _UI_GROUP_MARGIN, _UI_GROUP_MARGIN, _UI_GROUP_MARGIN)
        spiral_body_l.setSpacing(_UI_LAYOUT_SPACING)
        spiral_body_l.addWidget(spiral_body)
        spiral_form = QtWidgets.QFormLayout(spiral_body)
        spiral_form.setContentsMargins(_UI_GROUP_MARGIN, _UI_GROUP_MARGIN, _UI_GROUP_MARGIN, _UI_GROUP_MARGIN)
        spiral_form.setVerticalSpacing(_UI_FORM_VSPACING)
        spiral_form.addRow("Plane", self._plane)
        spiral_form.addRow("Segments", self._segments)
        spiral_form.addRow("Turns", self._turns)
        spiral_form.addRow("Guide rotation", self._guide_rotation)
        spiral_form.addRow("Outer radius", self._radius_outer)
        spiral_form.addRow("Inner radius", self._radius_inner)
        spiral_form.addRow("Ramp height", self._ramp_height)
        spiral_form.addRow("Width outer", self._width_outer)
        spiral_form.addRow("Width inner", self._width_inner)
        spiral_form.addRow("Tangent outer", self._tangent_outer)
        spiral_form.addRow("Tangent inner", self._tangent_inner)
        spiral_form.addRow("Axis thickness", self._axis_thickness)
        spiral_form.addRow("Snap step", self._snap_step)
        spiral_form.addRow("", self._use_rotations)
        spiral_form.addRow("", self._reverse_rotation)
        spiral_form.addRow("", self._chord_direction)
        spiral_form.addRow("", self._auto_merge)
        spiral_form.addRow("", self._greedy_merge)
        spiral_form.addRow("Max merge len", self._max_merge_len)
        spiral_form.addRow("Max chord step", self._max_chord_step)
        spiral_form.addRow("Merge max dev", self._merge_max_dev)
        spiral_box.toggled.connect(spiral_body.setVisible)
        spiral_body.setVisible(bool(spiral_box.isChecked()))

        helix_box = QtWidgets.QGroupBox("Double Helix (section)")
        self._helix_box = helix_box
        helix_box.setCheckable(True)
        helix_box.setChecked(False)
        helix_body = QtWidgets.QWidget(helix_box)
        self._helix_body = helix_body
        helix_body_l = QtWidgets.QVBoxLayout(helix_box)
        helix_body_l.setContentsMargins(_UI_GROUP_MARGIN, _UI_GROUP_MARGIN, _UI_GROUP_MARGIN, _UI_GROUP_MARGIN)
        helix_body_l.setSpacing(_UI_LAYOUT_SPACING)
        helix_body_l.addWidget(helix_body)
        helix_form = QtWidgets.QFormLayout(helix_body)
        helix_form.setContentsMargins(_UI_GROUP_MARGIN, _UI_GROUP_MARGIN, _UI_GROUP_MARGIN, _UI_GROUP_MARGIN)
        helix_form.setVerticalSpacing(_UI_FORM_VSPACING)
        helix_form.addRow("Segments", self._helix_segments)
        helix_form.addRow("Turns", self._helix_turns)
        helix_form.addRow("Axis plane", self._helix_axis_plane)
        helix_form.addRow("Axis heading", self._helix_heading)
        helix_form.addRow("Length", self._helix_length)
        helix_form.addRow("Helix radius", self._helix_radius)
        helix_form.addRow("Phase", self._helix_phase)
        helix_form.addRow("Segment len", self._helix_segment_len)
        helix_form.addRow("Strand width", self._helix_strand_width)
        helix_form.addRow("Strand depth", self._helix_strand_depth)

        row_hscale = QtWidgets.QWidget(helix_box)
        row_hscale_l = QtWidgets.QHBoxLayout(row_hscale)
        row_hscale_l.setContentsMargins(0, 0, 0, 0)
        row_hscale_l.setSpacing(_UI_LAYOUT_SPACING)
        row_hscale_l.addWidget(self._helix_scale_x)
        row_hscale_l.addWidget(self._helix_scale_y)
        row_hscale_l.addWidget(self._helix_scale_z)
        helix_form.addRow("Scale X/Y/Z", row_hscale)

        helix_form.addRow("Snap step", self._helix_snap_step)
        helix_form.addRow("", self._helix_use_rotations)

        helix_form.addRow("", self._helix_chord_align_rotation)
        helix_form.addRow("", self._helix_chord_axis_only)
        helix_form.addRow("Rotation axis", self._helix_rotation_axis_override)
        helix_form.addRow("Chord rot offset", self._helix_chord_align_rotation_offset)
        helix_form.addRow("", self._helix_auto_merge)
        helix_form.addRow("", self._helix_greedy_merge)
        helix_form.addRow("Max merge len", self._helix_max_merge_len)
        helix_form.addRow("Target chord len", self._helix_target_chord_len)
        helix_form.addRow("Merge max dev", self._helix_merge_max_dev)
        helix_box.toggled.connect(helix_body.setVisible)
        helix_body.setVisible(bool(helix_box.isChecked()))

        export_box = QtWidgets.QGroupBox("Export")
        export_form = QtWidgets.QFormLayout(export_box)
        export_form.setContentsMargins(_UI_GROUP_MARGIN, _UI_GROUP_MARGIN, _UI_GROUP_MARGIN, _UI_GROUP_MARGIN)
        export_form.setVerticalSpacing(_UI_FORM_VSPACING)
        export_form.addRow("Texture", self._texture)
        export_form.addRow("", self._export_shade_false)

        viewport_box = QtWidgets.QGroupBox("Viewport")
        viewport_layout = QtWidgets.QVBoxLayout(viewport_box)
        viewport_layout.setContentsMargins(_UI_GROUP_MARGIN, _UI_GROUP_MARGIN, _UI_GROUP_MARGIN, _UI_GROUP_MARGIN)
        viewport_layout.setSpacing(_UI_LAYOUT_SPACING)
        viewport_layout.addWidget(self._view_mode)
        viewport_layout.addWidget(self._vp_faces)
        viewport_layout.addWidget(self._vp_translucent)
        viewport_layout.addWidget(self._vp_wireframe)

        _apply_button_scheme(self._btn_generate, "green")
        _apply_button_scheme(self._btn_clear, "purple")
        _apply_button_scheme(self._btn_save_base, "green")
        _apply_button_scheme(self._btn_save_arr, "green")
        _apply_button_scheme(self._btn_save_settings, "green")
        _apply_button_scheme(self._btn_load_settings, "blue")

        layout = QtWidgets.QVBoxLayout(w)
        layout.addWidget(shape_box)
        layout.addWidget(common_box)
        layout.addWidget(spiral_box)
        layout.addWidget(helix_box)
        layout.addWidget(export_box)
        layout.addWidget(viewport_box)
        layout.addWidget(self._lbl_status)
        layout.addStretch(1)
        layout.setSpacing(_UI_LAYOUT_SPACING)

        self._sync_viewport_render_options()
        self._btn_save_base.setEnabled(False)
        self._btn_save_arr.setEnabled(False)

        self._sync_shape_sections()

        return w

    def _set_arr_project_path(self, path: Optional[Path]) -> None:
        self._arr_project_path = path
        if self._arr_project_path is None:
            self._lbl_arr_project.setText("Project: (none)")
        else:
            self._lbl_arr_project.setText(f"Project: {self._arr_project_path}")

    def _arr_project_dir(self) -> Optional[Path]:
        if self._arr_project_path is None:
            return None
        return self._arr_project_path.parent

    def _relpath_for_project(self, p: str) -> str:
        raw = str(p or "").strip()
        if not raw:
            return ""
        base = self._arr_project_dir()
        if base is None:
            return raw
        try:
            pp = Path(raw)
            if pp.is_absolute():
                return str(pp.resolve().relative_to(base.resolve()))
        except Exception:
            return raw
        return raw

    def _abspath_from_project(self, p: str) -> str:
        raw = str(p or "").strip()
        if not raw:
            return ""
        base = self._arr_project_dir()
        if base is None:
            return raw
        pp = Path(raw)
        if pp.is_absolute():
            return str(pp)
        return str((base / pp).resolve())

    def _gather_arr_project(self) -> dict:
        prev = int(self._arr_instances_prev_row)
        if prev >= 0:
            self._store_editor_to_instance(prev)

        inst_out: list[dict] = []
        for inst in list(self._arr_instances):
            if not isinstance(inst, dict):
                continue
            inst_out.append(
                {
                    "name": str(inst.get("name", "Instance")),
                    "enabled": bool(inst.get("enabled", True)),
                    "source": str(inst.get("source", "current")),
                    "path": self._relpath_for_project(str(inst.get("path", ""))),
                    "scale": float(inst.get("scale", 1.0)),
                    "offset": str(inst.get("offset", "0,0,0")),
                }
            )

        return {
            "type": "mc_procedural_arrangement_project",
            "version": 1,
            "arr": {
                "view_mode": str(self._view_mode.currentData() or "base"),
                "arr_enable": bool(self._arr_enable.isChecked()),
                "arr_preview": bool(self._arr_preview.isChecked()),
                "arr_mode": str(self._arr_mode.currentData() or "radial4"),
                "arr_helix_heading_by_side": bool(self._arr_helix_heading_by_side.isChecked()),
                "arr_global_scale": float(self._arr_global_scale.value()),
                "arr_global_offset": str(self._arr_global_offset.text()),
                "arr_radius": float(self._arr_radius.value()),
                "arr_plane_pX": str(self._arr_plane_pX.currentData() or "XZ"),
                "arr_plane_nX": str(self._arr_plane_nX.currentData() or "XZ"),
                "arr_plane_pZ": str(self._arr_plane_pZ.currentData() or "YZ"),
                "arr_plane_nZ": str(self._arr_plane_nZ.currentData() or "YZ"),
                "arr_phase_pX": float(self._arr_phase_pX.value()),
                "arr_phase_nX": float(self._arr_phase_nX.value()),
                "arr_phase_pZ": float(self._arr_phase_pZ.value()),
                "arr_phase_nZ": float(self._arr_phase_nZ.value()),
                "vp_faces": bool(self._vp_faces.isChecked()),
                "vp_translucent": bool(self._vp_translucent.isChecked()),
                "vp_wireframe": bool(self._vp_wireframe.isChecked()),
            },
            "instances": inst_out,
            "instances_selected": int(self._selected_instance_index()),
        }

    def _apply_arr_project(self, project: dict, *, base_dir: Path) -> None:
        if not isinstance(project, dict):
            raise ValueError("Project JSON must be an object")
        if str(project.get("type", "")) != "mc_procedural_arrangement_project":
            raise ValueError("Not an arrangement project")

        arr = project.get("arr", {})
        if not isinstance(arr, dict):
            arr = {}

        self._set_combo_data(self._view_mode, str(arr.get("view_mode", str(self._view_mode.currentData() or "base"))))
        self._arr_enable.setChecked(bool(arr.get("arr_enable", bool(self._arr_enable.isChecked()))))
        self._arr_preview.setChecked(bool(arr.get("arr_preview", bool(self._arr_preview.isChecked()))))
        self._set_combo_data(self._arr_mode, str(arr.get("arr_mode", str(self._arr_mode.currentData() or "radial4"))))
        self._arr_helix_heading_by_side.setChecked(bool(arr.get("arr_helix_heading_by_side", bool(self._arr_helix_heading_by_side.isChecked()))))
        self._arr_global_scale.setValue(float(arr.get("arr_global_scale", float(self._arr_global_scale.value()))))
        self._arr_global_offset.setText(str(arr.get("arr_global_offset", str(self._arr_global_offset.text()))))
        self._arr_radius.setValue(float(arr.get("arr_radius", float(self._arr_radius.value()))))

        self._set_combo_data(self._arr_plane_pX, str(arr.get("arr_plane_pX", str(self._arr_plane_pX.currentData() or "XZ"))))
        self._set_combo_data(self._arr_plane_nX, str(arr.get("arr_plane_nX", str(self._arr_plane_nX.currentData() or "XZ"))))
        self._set_combo_data(self._arr_plane_pZ, str(arr.get("arr_plane_pZ", str(self._arr_plane_pZ.currentData() or "YZ"))))
        self._set_combo_data(self._arr_plane_nZ, str(arr.get("arr_plane_nZ", str(self._arr_plane_nZ.currentData() or "YZ"))))
        self._arr_phase_pX.setValue(float(arr.get("arr_phase_pX", float(self._arr_phase_pX.value()))))
        self._arr_phase_nX.setValue(float(arr.get("arr_phase_nX", float(self._arr_phase_nX.value()))))
        self._arr_phase_pZ.setValue(float(arr.get("arr_phase_pZ", float(self._arr_phase_pZ.value()))))
        self._arr_phase_nZ.setValue(float(arr.get("arr_phase_nZ", float(self._arr_phase_nZ.value()))))

        self._vp_faces.setChecked(bool(arr.get("vp_faces", bool(self._vp_faces.isChecked()))))
        self._vp_translucent.setChecked(bool(arr.get("vp_translucent", bool(self._vp_translucent.isChecked()))))
        self._vp_wireframe.setChecked(bool(arr.get("vp_wireframe", bool(self._vp_wireframe.isChecked()))))
        self._sync_viewport_render_options()

        raw_instances = project.get("instances", None)
        instances: list[dict] = []
        if isinstance(raw_instances, list):
            for inst in list(raw_instances):
                if not isinstance(inst, dict):
                    continue
                path_rel = str(inst.get("path", ""))
                path_abs = ""
                if path_rel.strip():
                    p = Path(path_rel)
                    path_abs = str(p if p.is_absolute() else (base_dir / p).resolve())
                instances.append(
                    {
                        "name": str(inst.get("name", "Instance")),
                        "enabled": bool(inst.get("enabled", True)),
                        "source": str(inst.get("source", "current")),
                        "path": str(path_abs),
                        "scale": float(inst.get("scale", 1.0)),
                        "offset": str(inst.get("offset", "0,0,0")),
                    }
                )
        if len(instances) <= 0:
            instances = [self._default_instance(n=1)]
        self._arr_instances = instances
        sel = int(project.get("instances_selected", 0))
        self._refresh_instances_ui(select_row=sel)
        self._maybe_update_arr_preview()
        self._refresh_viewport_from_mode()

    def _save_arr_project_to(self, path: Path) -> None:
        data = self._gather_arr_project()
        path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
        self._set_arr_project_path(path)

    def _on_save_arr_project_as(self) -> None:
        out_path, _filter = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save arrangement project",
            str(Path.cwd() / "arrangement_project.json"),
            "JSON (*.json)",
        )
        if not out_path:
            return
        self._save_arr_project_to(Path(out_path))
        self._lbl_status.setText(f"Saved project: {out_path}")

    def _on_save_arr_project(self) -> None:
        if self._arr_project_path is None:
            self._on_save_arr_project_as()
            return
        self._save_arr_project_to(Path(self._arr_project_path))
        self._lbl_status.setText(f"Saved project: {self._arr_project_path}")

    def _on_load_arr_project(self) -> None:
        in_path, _filter = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Load arrangement project",
            str(Path.cwd()),
            "JSON (*.json)",
        )
        if not in_path:
            return
        p = Path(in_path)
        raw = p.read_text(encoding="utf-8")
        data = json.loads(raw)
        self._set_arr_project_path(p)
        self._apply_arr_project(data, base_dir=p.parent)
        self._lbl_status.setText(f"Loaded project: {in_path}")

    def _maybe_autosave_arr_project(self) -> None:
        if not hasattr(self, "_arr_project_autosave"):
            return
        if not bool(self._arr_project_autosave.isChecked()):
            return
        if self._arr_project_path is None:
            return
        try:
            self._save_arr_project_to(Path(self._arr_project_path))
        except Exception:
            return

    def _gather_settings(self) -> dict:
        return {
            "shape": str(self._shape_combo.currentData() or "spiral"),
            "center": str(self._center.text()),
            "plane": str(self._plane.currentData() or "XZ"),
            "segments": int(self._segments.value()),
            "turns": float(self._turns.value()),
            "guide_rotation_deg": float(self._guide_rotation.value()),
            "radius_outer": float(self._radius_outer.value()),
            "radius_inner": float(self._radius_inner.value()),
            "ramp_height": float(self._ramp_height.value()),
            "width_outer": float(self._width_outer.value()),
            "width_inner": float(self._width_inner.value()),
            "tangent_len_outer": float(self._tangent_outer.value()),
            "tangent_len_inner": float(self._tangent_inner.value()),
            "axis_thickness": float(self._axis_thickness.value()),
            "snap_step": float(self._snap_step.value()),
            "use_rotations": bool(self._use_rotations.isChecked()),
            "reverse_rotation": bool(self._reverse_rotation.isChecked()),
            "chord_direction": bool(self._chord_direction.isChecked()),
            "auto_merge": bool(self._auto_merge.isChecked()),
            "greedy_merge": bool(self._greedy_merge.isChecked()),
            "max_merge_len": float(self._max_merge_len.value()),
            "max_chord_step": float(self._max_chord_step.value()),
            "merge_max_dev": float(self._merge_max_dev.value()),
            "post_scale": float(self._post_scale.value()),
            "min_volume": float(self._min_volume.value()),
            "depth_stagger_step": float(self._depth_stagger.value()),
            "helix_axis_plane": str(self._helix_axis_plane.currentData() or "XZ"),
            "helix_heading_deg": float(self._helix_heading.value()),
            "helix_segments": int(self._helix_segments.value()),
            "helix_turns": float(self._helix_turns.value()),
            "helix_length": float(self._helix_length.value()),
            "helix_radius": float(self._helix_radius.value()),
            "helix_phase_deg": float(self._helix_phase.value()),
            "helix_segment_len": float(self._helix_segment_len.value()),
            "helix_strand_width": float(self._helix_strand_width.value()),
            "helix_strand_depth": float(self._helix_strand_depth.value()),
            "helix_scale_x": float(self._helix_scale_x.value()),
            "helix_scale_y": float(self._helix_scale_y.value()),
            "helix_scale_z": float(self._helix_scale_z.value()),
            "helix_snap_step": float(self._helix_snap_step.value()),
            "helix_use_rotations": bool(self._helix_use_rotations.isChecked()),
            "helix_chord_align_rotation": bool(self._helix_chord_align_rotation.isChecked()),
            "helix_chord_axis_only": bool(self._helix_chord_axis_only.isChecked()),
            "helix_rotation_axis_override": str(self._helix_rotation_axis_override.currentData() or "auto"),
            "helix_chord_align_rotation_offset_deg": float(self._helix_chord_align_rotation_offset.value()),
            "helix_auto_merge": bool(self._helix_auto_merge.isChecked()),
            "helix_greedy_merge": bool(self._helix_greedy_merge.isChecked()),
            "helix_max_merge_len": float(self._helix_max_merge_len.value()),
            "helix_target_chord_len": float(self._helix_target_chord_len.value()),
            "helix_merge_max_dev": float(self._helix_merge_max_dev.value()),
            "texture": str(self._texture.text()),
            "export_shade_false": bool(self._export_shade_false.isChecked()),
            "view_mode": str(self._view_mode.currentData() or "base"),
            "arr_enable": bool(self._arr_enable.isChecked()),
            "arr_preview": bool(self._arr_preview.isChecked()),
            "arr_mode": str(self._arr_mode.currentData() or "radial4"),
            "arr_helix_heading_by_side": bool(self._arr_helix_heading_by_side.isChecked()),
            "arr_instances": list(self._arr_instances),
            "arr_instances_selected": int(self._selected_instance_index()),
            "arr_global_scale": float(self._arr_global_scale.value()),
            "arr_global_offset": str(self._arr_global_offset.text()),
            "arr_radius": float(self._arr_radius.value()),
            "arr_plane_pX": str(self._arr_plane_pX.currentData() or "XZ"),
            "arr_plane_nX": str(self._arr_plane_nX.currentData() or "XZ"),
            "arr_plane_pZ": str(self._arr_plane_pZ.currentData() or "YZ"),
            "arr_plane_nZ": str(self._arr_plane_nZ.currentData() or "YZ"),
            "arr_phase_pX": float(self._arr_phase_pX.value()),
            "arr_phase_nX": float(self._arr_phase_nX.value()),
            "arr_phase_pZ": float(self._arr_phase_pZ.value()),
            "arr_phase_nZ": float(self._arr_phase_nZ.value()),
            "vp_faces": bool(self._vp_faces.isChecked()),
            "vp_translucent": bool(self._vp_translucent.isChecked()),
            "vp_wireframe": bool(self._vp_wireframe.isChecked()),
        }

    def _set_combo_data(self, combo: QtWidgets.QComboBox, data: str) -> None:
        want = str(data)
        for i in range(int(combo.count())):
            if str(combo.itemData(i)) == want:
                combo.setCurrentIndex(i)
                return

    def _apply_settings(self, settings: dict) -> None:
        if not isinstance(settings, dict):
            raise ValueError("Settings JSON must be an object")

        self._set_combo_data(self._shape_combo, str(settings.get("shape", "spiral")))
        self._center.setText(str(settings.get("center", self._center.text())))
        self._set_combo_data(self._plane, str(settings.get("plane", "XZ")))

        self._segments.setValue(int(settings.get("segments", int(self._segments.value()))))
        self._turns.setValue(float(settings.get("turns", float(self._turns.value()))))
        self._guide_rotation.setValue(float(settings.get("guide_rotation_deg", float(self._guide_rotation.value()))))
        self._radius_outer.setValue(float(settings.get("radius_outer", float(self._radius_outer.value()))))
        self._radius_inner.setValue(float(settings.get("radius_inner", float(self._radius_inner.value()))))
        self._ramp_height.setValue(float(settings.get("ramp_height", float(self._ramp_height.value()))))
        self._width_outer.setValue(float(settings.get("width_outer", float(self._width_outer.value()))))
        self._width_inner.setValue(float(settings.get("width_inner", float(self._width_inner.value()))))
        self._tangent_outer.setValue(float(settings.get("tangent_len_outer", float(self._tangent_outer.value()))))
        self._tangent_inner.setValue(float(settings.get("tangent_len_inner", float(self._tangent_inner.value()))))
        self._axis_thickness.setValue(float(settings.get("axis_thickness", float(self._axis_thickness.value()))))
        self._snap_step.setValue(float(settings.get("snap_step", float(self._snap_step.value()))))

        self._use_rotations.setChecked(bool(settings.get("use_rotations", bool(self._use_rotations.isChecked()))))
        self._reverse_rotation.setChecked(bool(settings.get("reverse_rotation", bool(self._reverse_rotation.isChecked()))))
        self._chord_direction.setChecked(bool(settings.get("chord_direction", bool(self._chord_direction.isChecked()))))
        self._auto_merge.setChecked(bool(settings.get("auto_merge", bool(self._auto_merge.isChecked()))))
        self._greedy_merge.setChecked(bool(settings.get("greedy_merge", bool(self._greedy_merge.isChecked()))))
        self._max_merge_len.setValue(float(settings.get("max_merge_len", float(self._max_merge_len.value()))))
        self._max_chord_step.setValue(float(settings.get("max_chord_step", float(self._max_chord_step.value()))))
        self._merge_max_dev.setValue(float(settings.get("merge_max_dev", float(self._merge_max_dev.value()))))

        self._post_scale.setValue(float(settings.get("post_scale", float(self._post_scale.value()))))
        self._min_volume.setValue(float(settings.get("min_volume", float(self._min_volume.value()))))
        self._depth_stagger.setValue(float(settings.get("depth_stagger_step", float(self._depth_stagger.value()))))

        self._set_combo_data(self._helix_axis_plane, str(settings.get("helix_axis_plane", str(self._helix_axis_plane.currentData() or "XZ"))))
        self._helix_heading.setValue(float(settings.get("helix_heading_deg", float(self._helix_heading.value()))))
        self._helix_segments.setValue(int(settings.get("helix_segments", int(settings.get("segments", int(self._helix_segments.value()))))))
        self._helix_turns.setValue(float(settings.get("helix_turns", float(settings.get("turns", float(self._helix_turns.value()))))))
        self._helix_length.setValue(float(settings.get("helix_length", float(self._helix_length.value()))))
        self._helix_radius.setValue(float(settings.get("helix_radius", float(self._helix_radius.value()))))
        self._helix_phase.setValue(float(settings.get("helix_phase_deg", float(self._helix_phase.value()))))
        self._helix_segment_len.setValue(float(settings.get("helix_segment_len", float(self._helix_segment_len.value()))))
        self._helix_strand_width.setValue(float(settings.get("helix_strand_width", float(self._helix_strand_width.value()))))
        self._helix_strand_depth.setValue(float(settings.get("helix_strand_depth", float(self._helix_strand_depth.value()))))
        self._helix_scale_x.setValue(float(settings.get("helix_scale_x", float(self._helix_scale_x.value()))))
        self._helix_scale_y.setValue(float(settings.get("helix_scale_y", float(self._helix_scale_y.value()))))
        self._helix_scale_z.setValue(float(settings.get("helix_scale_z", float(self._helix_scale_z.value()))))
        self._helix_snap_step.setValue(float(settings.get("helix_snap_step", float(settings.get("snap_step", float(self._helix_snap_step.value()))))))
        self._helix_use_rotations.setChecked(bool(settings.get("helix_use_rotations", bool(settings.get("use_rotations", bool(self._helix_use_rotations.isChecked()))))))
        self._helix_chord_align_rotation.setChecked(bool(settings.get("helix_chord_align_rotation", bool(self._helix_chord_align_rotation.isChecked()))))
        self._helix_chord_axis_only.setChecked(bool(settings.get("helix_chord_axis_only", bool(self._helix_chord_axis_only.isChecked()))))
        self._set_combo_data(self._helix_rotation_axis_override, str(settings.get("helix_rotation_axis_override", str(self._helix_rotation_axis_override.currentData() or "auto"))))
        self._helix_chord_align_rotation_offset.setValue(float(settings.get("helix_chord_align_rotation_offset_deg", float(self._helix_chord_align_rotation_offset.value()))))
        self._helix_auto_merge.setChecked(bool(settings.get("helix_auto_merge", bool(self._helix_auto_merge.isChecked()))))
        self._helix_greedy_merge.setChecked(bool(settings.get("helix_greedy_merge", bool(self._helix_greedy_merge.isChecked()))))
        self._helix_max_merge_len.setValue(float(settings.get("helix_max_merge_len", float(self._helix_max_merge_len.value()))))
        self._helix_target_chord_len.setValue(float(settings.get("helix_target_chord_len", float(self._helix_target_chord_len.value()))))
        self._helix_merge_max_dev.setValue(float(settings.get("helix_merge_max_dev", float(self._helix_merge_max_dev.value()))))

        self._texture.setText(str(settings.get("texture", self._texture.text())))
        self._export_shade_false.setChecked(bool(settings.get("export_shade_false", bool(self._export_shade_false.isChecked()))))

        self._set_combo_data(self._view_mode, str(settings.get("view_mode", str(self._view_mode.currentData() or "base"))))

        self._arr_enable.setChecked(bool(settings.get("arr_enable", bool(self._arr_enable.isChecked()))))
        self._arr_preview.setChecked(bool(settings.get("arr_preview", bool(self._arr_preview.isChecked()))))
        self._set_combo_data(self._arr_mode, str(settings.get("arr_mode", str(self._arr_mode.currentData() or "radial4"))))
        self._arr_helix_heading_by_side.setChecked(bool(settings.get("arr_helix_heading_by_side", bool(self._arr_helix_heading_by_side.isChecked()))))
        self._arr_global_scale.setValue(float(settings.get("arr_global_scale", float(self._arr_global_scale.value()))))
        self._arr_global_offset.setText(str(settings.get("arr_global_offset", self._arr_global_offset.text())))
        self._arr_radius.setValue(float(settings.get("arr_radius", float(self._arr_radius.value()))))

        raw_instances = settings.get("arr_instances", None)
        if isinstance(raw_instances, list) and len(raw_instances) > 0:
            cleaned: list[dict] = []
            for inst in list(raw_instances):
                if not isinstance(inst, dict):
                    continue
                cleaned.append(
                    {
                        "name": str(inst.get("name", "Instance")),
                        "enabled": bool(inst.get("enabled", True)),
                        "source": str(inst.get("source", "current")),
                        "path": str(inst.get("path", "")),
                        "scale": float(inst.get("scale", 1.0)),
                        "offset": str(inst.get("offset", "0,0,0")),
                    }
                )
            if len(cleaned) > 0:
                self._arr_instances = cleaned
        if len(self._arr_instances) <= 0:
            self._arr_instances = [self._default_instance(n=1)]
        sel = int(settings.get("arr_instances_selected", 0))
        self._refresh_instances_ui(select_row=sel)

        self._set_combo_data(self._arr_plane_pX, str(settings.get("arr_plane_pX", str(self._arr_plane_pX.currentData() or "XZ"))))
        self._set_combo_data(self._arr_plane_nX, str(settings.get("arr_plane_nX", str(self._arr_plane_nX.currentData() or "XZ"))))
        self._set_combo_data(self._arr_plane_pZ, str(settings.get("arr_plane_pZ", str(self._arr_plane_pZ.currentData() or "YZ"))))
        self._set_combo_data(self._arr_plane_nZ, str(settings.get("arr_plane_nZ", str(self._arr_plane_nZ.currentData() or "YZ"))))
        self._arr_phase_pX.setValue(float(settings.get("arr_phase_pX", float(self._arr_phase_pX.value()))))
        self._arr_phase_nX.setValue(float(settings.get("arr_phase_nX", float(self._arr_phase_nX.value()))))
        self._arr_phase_pZ.setValue(float(settings.get("arr_phase_pZ", float(self._arr_phase_pZ.value()))))
        self._arr_phase_nZ.setValue(float(settings.get("arr_phase_nZ", float(self._arr_phase_nZ.value()))))

        self._vp_faces.setChecked(bool(settings.get("vp_faces", bool(self._vp_faces.isChecked()))))
        self._vp_translucent.setChecked(bool(settings.get("vp_translucent", bool(self._vp_translucent.isChecked()))))
        self._vp_wireframe.setChecked(bool(settings.get("vp_wireframe", bool(self._vp_wireframe.isChecked()))))
        self._sync_viewport_render_options()
        self._sync_shape_sections()
        self._maybe_update_arr_preview()
        self._refresh_viewport_from_mode()

    def _default_settings_name(self) -> str:
        return "mc_model_procedural_shapes_ui_settings.json"

    def _write_settings_file(self, path: Path) -> None:
        path.write_text(json.dumps(self._gather_settings(), indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")

    def _settings_snapshot_path_for_model(self, model_path: Path) -> Path:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = f"mc_model_procedural_shapes_ui_settings_{ts}.json"
        return model_path.parent / name

    def _sync_viewport_render_options(self) -> None:
        faces = bool(self._vp_faces.isChecked())
        self._vp_translucent.setEnabled(faces)
        self._viewport.set_render_options(
            faces=faces,
            faces_translucent=bool(self._vp_translucent.isChecked()),
            wireframe=bool(self._vp_wireframe.isChecked()),
        )

    def _parse_center3(self) -> tuple[float, float, float]:
        raw = self._center.text().strip()
        parts = [p.strip() for p in raw.split(",")]
        if len(parts) != 3:
            raise ValueError("Center must be x,y,z")
        return float(parts[0]), float(parts[1]), float(parts[2])

    def _postprocess_elements(self, elements: list[Cuboid], *, snap_step: float) -> list[Cuboid]:
        s = float(self._post_scale.value())
        min_vol = float(self._min_volume.value())
        post: list[Cuboid] = []
        for c in list(elements):
            cc = _scale_cuboid_about_centroid(c, s, origin_step=float(snap_step))
            if cc is None:
                continue
            if min_vol > 0.0 and _cuboid_volume(cc) < min_vol:
                continue
            post.append(cc)
        return post

    def _make_spiral_config(self) -> SpiralConfig:
        return SpiralConfig(
            center=self._parse_center3(),
            plane=str(self._plane.currentData() or "XZ"),
            segments=int(self._segments.value()),
            turns=float(self._turns.value()),
            guide_rotation_deg=float(self._guide_rotation.value()),
            radius_outer=float(self._radius_outer.value()),
            radius_inner=float(self._radius_inner.value()),
            ramp_height=float(self._ramp_height.value()),
            width_outer=float(self._width_outer.value()),
            width_inner=float(self._width_inner.value()),
            tangent_len_outer=float(self._tangent_outer.value()),
            tangent_len_inner=float(self._tangent_inner.value()),
            axis_thickness=float(self._axis_thickness.value()),
            snap_step=float(self._snap_step.value()),
            use_rotations=bool(self._use_rotations.isChecked()),
            reverse_rotation=bool(self._reverse_rotation.isChecked()),
            chord_direction=bool(self._chord_direction.isChecked()),
            auto_merge=bool(self._auto_merge.isChecked()),
            greedy_merge=bool(self._greedy_merge.isChecked()),
            max_merge_len=float(self._max_merge_len.value()),
            max_chord_step=float(self._max_chord_step.value()),
            merge_max_dev=float(self._merge_max_dev.value()),
        )

    def _make_helix_config(self) -> HelixSectionConfig:
        return HelixSectionConfig(
            center=self._parse_center3(),
            axis_plane=str(self._helix_axis_plane.currentData() or "XZ"),
            heading_deg=float(self._helix_heading.value()),
            segments=int(self._helix_segments.value()),
            length=float(self._helix_length.value()),
            turns=float(self._helix_turns.value()),
            helix_radius=float(self._helix_radius.value()),
            phase_deg=float(self._helix_phase.value()),
            segment_len=float(self._helix_segment_len.value()),
            strand_width=float(self._helix_strand_width.value()),
            strand_depth=float(self._helix_strand_depth.value()),
            scale_x=float(self._helix_scale_x.value()),
            scale_y=float(self._helix_scale_y.value()),
            scale_z=float(self._helix_scale_z.value()),
            snap_step=float(self._helix_snap_step.value()),
            use_rotations=bool(self._helix_use_rotations.isChecked()),
            chord_align_rotation=bool(self._helix_chord_align_rotation.isChecked()),
            chord_axis_only=bool(self._helix_chord_axis_only.isChecked()),
            rotation_axis_override=str(self._helix_rotation_axis_override.currentData() or "auto"),
            chord_align_rotation_offset_deg=float(self._helix_chord_align_rotation_offset.value()),
            auto_merge=bool(self._helix_auto_merge.isChecked()),
            greedy_merge=bool(self._helix_greedy_merge.isChecked()),
            max_merge_len=float(self._helix_max_merge_len.value()),
            target_chord_len=float(self._helix_target_chord_len.value()),
            merge_max_dev=float(self._helix_merge_max_dev.value()),
        )

    def _on_generate(self) -> None:
        try:
            shape = str(self._shape_combo.currentData() or "spiral")
            print(f"Generate: shape={shape}")
            if shape == "double_helix_section":
                cfg_h = self._make_helix_config()
                print(
                    f"Helix cfg: plane={cfg_h.axis_plane} heading={cfg_h.heading_deg} segs={cfg_h.segments} len={cfg_h.length} "
                    f"turns={cfg_h.turns} radius={cfg_h.helix_radius} chord_align={cfg_h.chord_align_rotation} "
                    f"axis_only={cfg_h.chord_axis_only} target_chord={cfg_h.target_chord_len} auto_merge={cfg_h.auto_merge} greedy={cfg_h.greedy_merge}"
                )
                base_raw = generate_double_helix_section(cfg_h, phase_offset_rad=0.0)
                snap_step = float(cfg_h.snap_step)
                depth_plane = str(cfg_h.axis_plane).split(":")[0]
            else:
                cfg = self._make_spiral_config()
                print(f"Spiral cfg: plane={cfg.plane} segs={cfg.segments} turns={cfg.turns} outer={cfg.radius_outer} inner={cfg.radius_inner} ramp={cfg.ramp_height}")
                base_raw = generate_shape(cfg, shape, phase_offset_rad=0.0)
                snap_step = float(cfg.snap_step)
                depth_plane = str(cfg.plane)
        except Exception as e:
            print("Generate failed:")
            print(traceback.format_exc())
            QtWidgets.QMessageBox.critical(self, "Generate failed", str(e))
            return

        base = self._postprocess_elements(list(base_raw), snap_step=snap_step)
        base = _apply_depth_stagger(list(base), plane=str(depth_plane), step=float(self._depth_stagger.value()))
        self._current_elements_base = list(base)
        print(f"Base elements: {len(self._current_elements_base)}")

        arr: list[Cuboid] = []
        if bool(self._arr_enable.isChecked()):
            try:
                arr = self._compute_arrangement_elements(shape)
            except Exception as e:
                print("Arrangement failed:")
                print(traceback.format_exc())
                QtWidgets.QMessageBox.critical(self, "Arrangement failed", str(e))
                arr = []

        self._current_elements_arr = list(arr)

        self._refresh_viewport_from_mode()

        tex = self._texture.text().strip() or "minecraft:block/oak_planks"
        shade_false = bool(self._export_shade_false.isChecked())
        self._current_model_base = export_minecraft_model_ext(elements=self._current_elements_base, texture=tex, shade_false=shade_false)
        self._current_model_arr = export_minecraft_model_ext(elements=self._current_elements_arr, texture=tex, shade_false=shade_false) if len(self._current_elements_arr) > 0 else None

        view = str(self._view_mode.currentData() or "base")
        show_arr = bool(view == "arr") and bool(self._arr_enable.isChecked())
        show_model = self._current_model_arr if show_arr and self._current_model_arr is not None else self._current_model_base
        if show_model is not None:
            self._txt_model_json.setPlainText(format_minecraft_model_json(show_model))

        self._btn_save_base.setEnabled(True)
        self._btn_save_arr.setEnabled(self._current_model_arr is not None)
        show_elements = self._current_elements_arr if show_arr else self._current_elements_base
        self._lbl_status.setText(f"Generated: {len(show_elements)} elements")

    def _on_clear(self) -> None:
        self._current_elements_base = []
        self._current_model_base = None
        self._current_elements_arr = []
        self._current_model_arr = None
        self._viewport.clear_cuboids()
        self._txt_elements.setPlainText("")
        self._txt_model_json.setPlainText("")
        self._btn_save_base.setEnabled(False)
        self._btn_save_arr.setEnabled(False)
        self._lbl_status.setText("Cleared")

    def _save_model_json_and_snapshot(self, model: dict, *, suggested_name: str) -> None:
        out_path, _filter = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save Minecraft model JSON",
            str(Path.cwd() / suggested_name),
            "JSON (*.json)",
        )
        if not out_path:
            return

        Path(out_path).write_text(format_minecraft_model_json(model) + "\n", encoding="utf-8", newline="\n")
        snap_path = self._settings_snapshot_path_for_model(Path(out_path))
        self._write_settings_file(snap_path)
        self._lbl_status.setText(f"Saved: {out_path}")

    def _on_save_base(self) -> None:
        if self._current_model_base is None:
            QtWidgets.QMessageBox.information(self, "Nothing to save", "Generate first.")
            return
        self._save_model_json_and_snapshot(self._current_model_base, suggested_name="model_procedural_base.json")

    def _on_save_arr(self) -> None:
        if self._current_model_arr is None:
            QtWidgets.QMessageBox.information(self, "Nothing to save", "Generate arrangement first.")
            return
        self._save_model_json_and_snapshot(self._current_model_arr, suggested_name="model_procedural_arrangement.json")

    def _on_save_settings(self) -> None:
        out_path, _filter = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save procedural shapes settings",
            str(Path.cwd() / self._default_settings_name()),
            "JSON (*.json)",
        )
        if not out_path:
            return
        self._write_settings_file(Path(out_path))
        self._lbl_status.setText(f"Saved settings: {out_path}")

    def _on_load_settings(self) -> None:
        in_path, _filter = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Load procedural shapes settings",
            str(Path.cwd()),
            "JSON (*.json)",
        )
        if not in_path:
            return
        raw = Path(in_path).read_text(encoding="utf-8")
        settings = json.loads(raw)
        self._apply_settings(settings)
        self._lbl_status.setText(f"Loaded settings: {in_path}")


def main() -> None:
    app = QtWidgets.QApplication(sys.argv)
    _apply_dark_theme(app)
    w = ProceduralShapesMainWindow()
    w.resize(1400, 820)
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
