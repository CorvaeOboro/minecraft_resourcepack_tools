"""MC Resourcepack Blender Dashboard (UI)

PySide6 UI that scans a resourcepack (folder containing `assets/`) and orchestrates
headless Blender jobs to:
- Generate missing cached `.blend` files from Minecraft `.json` models or Optifine CEM `.jem`
- Render PNGs from cached `.blend` files (one PNG per camera)

It delegates Blender-side work to:
- `mc_resourcepack_blender_generate_blends.py`
- `mc_resourcepack_blender_render_blends.py`

TOOLSGROUP::RENDER
SORTGROUP::7
SORTPRIORITY::76
STATUS::active
VERSION::20260308

Dependencies
- Python 3.x
- PySide6
- Blender installed (either on PATH as `blender` or selected via UI)

Expected inputs
- Resourcepack root (must contain `assets/`)
- Optional scan root (scan only within a subfolder)

Outputs
- Writes cached `.blend` files next to source models when generating
- Writes rendered PNGs into `<blend_dir>/<blend_stem>/` for each `.blend`
- Writes UI settings to `DEV/mc_resourcepack_blender_dashboard_settings.json`

Usage
- Simple:
  python DEV/mc_resourcepack_blender_dashboard.py

- Project example:
  - Set Resourcepack root to:
    `RESOURCEPACK/00_OBORO_20260217`
  - Click Scan
  - Click Generate Missing .blend
  - Click Render Selected / Render All
"""

import json
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path


_DARK_QSS = """
QWidget {
  background-color: #1a1a1a;
  color: #d0d0d0;
}

QLabel {
  background-color: transparent;
  color: #d0d0d0;
}

QLineEdit {
  background-color: #000000;
  color: #d0d0d0;
  border: 1px solid #3a3a3a;
  border-radius: 4px;
  padding: 6px;
}

QLineEdit:focus {
  border: 1px solid #5a5a5a;
}

QPlainTextEdit {
  background-color: #000000;
  color: #d0d0d0;
  border: 1px solid #3a3a3a;
  border-radius: 4px;
}

QTableWidget {
  background-color: #121212;
  color: #d0d0d0;
  gridline-color: #2a2a2a;
  border: 1px solid #3a3a3a;
  border-radius: 4px;
}

QHeaderView::section {
  background-color: #202020;
  color: #d0d0d0;
  border: 1px solid #2f2f2f;
  padding: 4px 6px;
}

QTableWidget::item:selected {
  background-color: #2a4a66;
  color: #f0f0f0;
}

QCheckBox {
  spacing: 6px;
}

QCheckBox::indicator {
  width: 14px;
  height: 14px;
}

QCheckBox::indicator:unchecked {
  background-color: #000000;
  border: 1px solid #3a3a3a;
  border-radius: 3px;
}

QCheckBox::indicator:checked {
  background-color: #2a4a66;
  border: 1px solid #3a3a3a;
  border-radius: 3px;
}

QPushButton {
  background-color: #262626;
  color: #d0d0d0;
  border: 1px solid #3a3a3a;
  border-radius: 6px;
  padding: 6px 10px;
}

QPushButton:hover {
  background-color: #2e2e2e;
}

QPushButton:pressed {
  background-color: #1f1f1f;
}

QPushButton:disabled {
  background-color: #1b1b1b;
  color: #7a7a7a;
  border: 1px solid #2a2a2a;
}

QPushButton[category="blue"] {
  background-color: #2b3a4a;
  border: 1px solid #3c4f66;
}

QPushButton[category="blue"]:hover {
  background-color: #32455a;
}

QPushButton[category="blue"]:pressed {
  background-color: #243546;
}

QPushButton[category="green"] {
  background-color: #2f4433;
  border: 1px solid #3f5a44;
}

QPushButton[category="green"]:hover {
  background-color: #37503b;
}

QPushButton[category="green"]:pressed {
  background-color: #273a2b;
}

QPushButton[category="purple"] {
  background-color: #3b2f44;
  border: 1px solid #53405e;
}

QPushButton[category="purple"]:hover {
  background-color: #453752;
}

QPushButton[category="purple"]:pressed {
  background-color: #2f2636;
}
"""


def _quote_arg(a: str) -> str:
    s = str(a)
    if not s:
        return '""'
    if any(ch.isspace() for ch in s) or any(ch in s for ch in ['"', "'"]):
        return '"' + s.replace('"', '\\"') + '"'
    return s


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


@dataclass(frozen=True)
class ModelItem:
    source_path: Path | None
    blend_path: Path
    resourcepack_root: Path
    namespace: str
    render_rel_dir: Path
    model_stem: str

    def cache_blend_path(self) -> Path:
        return self.blend_path

    def render_output_dir(self) -> Path:
        p = self.cache_blend_path()
        return p.parent / p.stem

    def render_output_prefix(self) -> str:
        return f"{self.cache_blend_path().stem}_RENDER_"

    def model_id(self) -> str:
        rel = (self.render_rel_dir / self.model_stem).as_posix().lstrip("/")
        return f"{self.namespace}:{rel}"


def _find_resourcepack_root(model_json_path: Path) -> Path:
    p = model_json_path.resolve()
    for parent in [p.parent] + list(p.parents):
        if parent.name.lower() == "assets":
            return parent.parent
    raise ValueError(f"Could not locate resourcepack root for: {model_json_path}")


def _strip_first_dir_if(rel_dir: Path, names: set[str]) -> Path:
    if not rel_dir.parts:
        return rel_dir
    if rel_dir.parts[0].lower() in names:
        return Path(*rel_dir.parts[1:]) if len(rel_dir.parts) > 1 else Path()
    return rel_dir


def _identify_item(path: Path, resourcepack_root: Path | None = None) -> ModelItem:
    p = Path(path).resolve()
    if resourcepack_root is None:
        resourcepack_root = _find_resourcepack_root(p)
    resourcepack_root = Path(resourcepack_root).resolve()

    parts = p.parts
    assets_idx = None
    for i, part in enumerate(parts):
        if part.lower() == "assets":
            assets_idx = i
            break
    if assets_idx is None:
        raise ValueError(f"Invalid path (no assets/): {p}")
    if assets_idx + 2 >= len(parts):
        raise ValueError(f"Invalid path: {p}")

    namespace = str(parts[assets_idx + 1])
    rel_after_ns = Path(*parts[assets_idx + 2 :])
    rel_dir = rel_after_ns.parent
    stem = rel_after_ns.stem

    suf = p.suffix.lower()
    if suf in {".json", ".jem"}:
        source_path = p
        blend_path = p.with_suffix(".blend")
    elif suf == ".blend":
        source_path = None
        blend_path = p
        src_json = p.with_suffix(".json")
        src_jem = p.with_suffix(".jem")
        if src_json.exists():
            source_path = src_json
        elif src_jem.exists():
            source_path = src_jem
    else:
        raise ValueError(f"Unsupported item suffix: {p}")

    render_rel_dir = _strip_first_dir_if(rel_dir, {"models", "textures"})

    return ModelItem(
        source_path=source_path,
        blend_path=blend_path,
        resourcepack_root=resourcepack_root,
        namespace=namespace,
        render_rel_dir=render_rel_dir,
        model_stem=str(stem),
    )


def _iter_model_sources(resourcepack_root: Path) -> list[Path]:
    root = Path(resourcepack_root).resolve()
    assets_dir = root / "assets"
    if not assets_dir.exists():
        return []
    out: list[Path] = []
    for ns_dir in assets_dir.iterdir():
        if not ns_dir.is_dir():
            continue
        models_dir = ns_dir / "models"
        if models_dir.exists():
            for p in models_dir.rglob("*.json"):
                if p.is_file():
                    out.append(p)

        cem_dir = ns_dir / "optifine" / "cem"
        if cem_dir.exists():
            for p in cem_dir.rglob("*.jem"):
                if p.is_file():
                    out.append(p)
    return out


def _iter_blends(resourcepack_root: Path) -> list[Path]:
    root = Path(resourcepack_root).resolve()
    assets_dir = root / "assets"
    if not assets_dir.exists():
        return []
    out: list[Path] = []
    for ns_dir in assets_dir.iterdir():
        if not ns_dir.is_dir():
            continue
        for p in ns_dir.rglob("*.blend"):
            if p.is_file():
                out.append(p)
    return out


def _iter_items_under(dir_path: Path) -> list[Path]:
    base = Path(dir_path).resolve()
    if not base.exists() or not base.is_dir():
        return []
    out: list[Path] = []
    for p in base.rglob("*"):
        if not p.is_file():
            continue
        suf = p.suffix.lower()
        if suf in {".json", ".jem", ".blend"}:
            out.append(p)
    return out


def _has_any_png(dir_path: Path) -> bool:
    try:
        if not dir_path.exists():
            return False
        for p in dir_path.iterdir():
            if p.is_file() and p.suffix.lower() == ".png":
                return True
        return False
    except Exception:
        return False


class _Jobs(QtCore.QObject):
    log_line = QtCore.Signal(str)
    state = QtCore.Signal(str)
    finished = QtCore.Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._queue: list[tuple[str, list[str], str]] = []
        self._proc = QtCore.QProcess(self)
        self._proc.setProcessChannelMode(QtCore.QProcess.SeparateChannels)
        self._proc.started.connect(self._on_started)
        self._proc.stateChanged.connect(self._on_state_changed)
        self._proc.errorOccurred.connect(self._on_error)
        self._proc.readyReadStandardOutput.connect(self._on_ready_out)
        self._proc.readyReadStandardError.connect(self._on_ready_err)
        self._proc.finished.connect(self._on_finished)
        self._running = False
        self._current_label = ""

    def enqueue(self, exe: str, args: list[str], label: str) -> None:
        self._queue.append((exe, args, label))
        if not self._running:
            self._start_next()

    def clear(self) -> None:
        if self._running:
            try:
                self._proc.kill()
            except Exception:
                pass
        self._queue.clear()

    def is_running(self) -> bool:
        return self._running

    def _start_next(self) -> None:
        if not self._queue:
            self._running = False
            self.state.emit("Idle")
            self.finished.emit()
            return

        exe, args, label = self._queue.pop(0)
        self._running = True
        self._current_label = label
        self.state.emit(label)

        if ("\\" in exe or "/" in exe) and not Path(exe).exists():
            self.log_line.emit(f"[warn] exe path does not exist: {exe}")
        if ("\\" not in exe and "/" not in exe) and shutil.which(exe) is None:
            self.log_line.emit(f"[warn] exe not found on PATH: {exe}")

        cmd = " ".join([_quote_arg(exe)] + [_quote_arg(a) for a in args])
        self.log_line.emit(f"$ {cmd}")
        self._proc.start(exe, args)

    def _on_started(self) -> None:
        try:
            pid = int(self._proc.processId())
        except Exception:
            pid = -1
        self.log_line.emit(f"[started] pid={pid}")

    def _on_state_changed(self, st) -> None:
        name = getattr(st, "name", None)
        if not name:
            if st == QtCore.QProcess.NotRunning:
                name = "NotRunning"
            elif st == QtCore.QProcess.Starting:
                name = "Starting"
            elif st == QtCore.QProcess.Running:
                name = "Running"
            else:
                name = str(st)
        self.log_line.emit(f"[state] {name}")

    def _on_error(self, err) -> None:
        name = getattr(err, "name", None)
        if not name:
            if err == QtCore.QProcess.FailedToStart:
                name = "FailedToStart"
            elif err == QtCore.QProcess.Crashed:
                name = "Crashed"
            elif err == QtCore.QProcess.Timedout:
                name = "Timedout"
            elif err == QtCore.QProcess.WriteError:
                name = "WriteError"
            elif err == QtCore.QProcess.ReadError:
                name = "ReadError"
            elif err == QtCore.QProcess.UnknownError:
                name = "UnknownError"
            else:
                name = str(err)
        self.log_line.emit(f"[error] {name}: {self._proc.errorString()}")
        if name == "FailedToStart":
            self.log_line.emit(f"[exit] code=? (failed to start) label={self._current_label}")
            self._start_next()

    def _on_ready_out(self) -> None:
        try:
            data = bytes(self._proc.readAllStandardOutput()).decode("utf-8", errors="replace")
        except Exception:
            return
        for line in data.splitlines():
            self.log_line.emit(line)

    def _on_ready_err(self) -> None:
        try:
            data = bytes(self._proc.readAllStandardError()).decode("utf-8", errors="replace")
        except Exception:
            return
        for line in data.splitlines():
            self.log_line.emit(f"[stderr] {line}")

    def _on_finished(self, exit_code: int, exit_status) -> None:
        self.log_line.emit(f"[exit] code={exit_code} label={self._current_label}")
        self._start_next()


class Dashboard(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MC Resourcepack Blender Dashboard")
        self.resize(1100, 750)

        self._items: list[ModelItem] = []
        self._items_by_blend: dict[str, ModelItem] = {}

        self._loading_settings = False
        self._settings_timer = QtCore.QTimer(self)
        self._settings_timer.setSingleShot(True)
        self._settings_timer.timeout.connect(self._save_settings_now)

        self._jobs = _Jobs(self)
        self._jobs.log_line.connect(self._append_log)
        self._jobs.state.connect(self._set_state)

        root = QtWidgets.QWidget()
        self.setCentralWidget(root)

        layout = QtWidgets.QVBoxLayout(root)

        top = QtWidgets.QHBoxLayout()
        layout.addLayout(top)

        self._rp_root = QtWidgets.QLineEdit()
        self._rp_root.setPlaceholderText("Resourcepack root (folder containing assets/)")
        top.addWidget(self._rp_root, 4)

        btn_pick_rp = QtWidgets.QPushButton("Browse RP")
        btn_pick_rp.setProperty("category", "blue")
        btn_pick_rp.clicked.connect(self._browse_rp)
        top.addWidget(btn_pick_rp, 0)

        self._blender_exe = QtWidgets.QLineEdit()
        self._blender_exe.setPlaceholderText("Blender executable (optional if blender is on PATH)")
        top.addWidget(self._blender_exe, 3)

        btn_pick_blender = QtWidgets.QPushButton("Browse Blender")
        btn_pick_blender.setProperty("category", "blue")
        btn_pick_blender.clicked.connect(self._browse_blender)
        top.addWidget(btn_pick_blender, 0)

        sub = QtWidgets.QHBoxLayout()
        layout.addLayout(sub)

        self._scan_root = QtWidgets.QLineEdit()
        self._scan_root.setPlaceholderText("Subfolder to scan (optional). Example: .../assets/<ns>/textures/entity")
        sub.addWidget(self._scan_root, 5)

        btn_pick_scan = QtWidgets.QPushButton("Browse Folder")
        btn_pick_scan.setProperty("category", "blue")
        btn_pick_scan.clicked.connect(self._browse_scan_root)
        sub.addWidget(btn_pick_scan, 0)

        filt = QtWidgets.QHBoxLayout()
        layout.addLayout(filt)

        self._filter_text = QtWidgets.QLineEdit()
        self._filter_text.setPlaceholderText("Filter (matches model id / path)")
        self._filter_text.textChanged.connect(self._apply_filter)
        filt.addWidget(self._filter_text, 1)

        self._only_missing_cache = QtWidgets.QCheckBox("Only missing .blend")
        self._only_missing_cache.stateChanged.connect(self._apply_filter)
        filt.addWidget(self._only_missing_cache, 0)

        self._only_missing_renders = QtWidgets.QCheckBox("Only missing renders")
        self._only_missing_renders.stateChanged.connect(self._apply_filter)
        filt.addWidget(self._only_missing_renders, 0)

        actions = QtWidgets.QHBoxLayout()
        layout.addLayout(actions)

        btn_scan = QtWidgets.QPushButton("Scan")
        btn_scan.setProperty("category", "blue")
        btn_scan.clicked.connect(self._scan)
        actions.addWidget(btn_scan)

        btn_gen_missing = QtWidgets.QPushButton("Generate Missing .blend")
        btn_gen_missing.setProperty("category", "green")
        btn_gen_missing.clicked.connect(self._generate_missing)
        actions.addWidget(btn_gen_missing)

        btn_regen_sel = QtWidgets.QPushButton("Regenerate Selected .blend")
        btn_regen_sel.setProperty("category", "green")
        btn_regen_sel.clicked.connect(self._regenerate_selected)
        actions.addWidget(btn_regen_sel)

        btn_render_sel = QtWidgets.QPushButton("Render Selected")
        btn_render_sel.setProperty("category", "purple")
        btn_render_sel.clicked.connect(self._render_selected)
        actions.addWidget(btn_render_sel)

        btn_render_all = QtWidgets.QPushButton("Render All")
        btn_render_all.setProperty("category", "purple")
        btn_render_all.clicked.connect(self._render_all)
        actions.addWidget(btn_render_all)

        btn_cancel = QtWidgets.QPushButton("Cancel")
        btn_cancel.setProperty("category", "purple")
        btn_cancel.clicked.connect(self._cancel_jobs)
        actions.addWidget(btn_cancel)

        self._state = QtWidgets.QLabel("Idle")
        actions.addWidget(self._state, 1)

        self._table = QtWidgets.QTableWidget()
        self._table.setColumnCount(6)
        self._table.setHorizontalHeaderLabels(["Sel", "Item", "Source", "Blend", "Has Blend", "Has Renders"])
        self._table.horizontalHeader().setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeToContents)
        self._table.horizontalHeader().setSectionResizeMode(1, QtWidgets.QHeaderView.Stretch)
        self._table.horizontalHeader().setSectionResizeMode(2, QtWidgets.QHeaderView.Stretch)
        self._table.horizontalHeader().setSectionResizeMode(3, QtWidgets.QHeaderView.Stretch)
        self._table.horizontalHeader().setSectionResizeMode(4, QtWidgets.QHeaderView.ResizeToContents)
        self._table.horizontalHeader().setSectionResizeMode(5, QtWidgets.QHeaderView.ResizeToContents)
        self._table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self._table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        layout.addWidget(self._table, 3)

        self._log = QtWidgets.QPlainTextEdit()
        self._log.setReadOnly(True)
        font = QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.FixedFont)
        self._log.setFont(font)
        layout.addWidget(self._log, 2)

        self._rp_root.textChanged.connect(self._schedule_save_settings)
        self._blender_exe.textChanged.connect(self._schedule_save_settings)
        self._scan_root.textChanged.connect(self._schedule_save_settings)
        self._filter_text.textChanged.connect(self._schedule_save_settings)
        self._only_missing_cache.stateChanged.connect(self._schedule_save_settings)
        self._only_missing_renders.stateChanged.connect(self._schedule_save_settings)

        self._load_settings()

    def _append_log(self, line: str) -> None:
        self._log.appendPlainText(line)

    def _settings_path(self) -> Path:
        here = Path(__file__).resolve().parent
        return here / "mc_resourcepack_blender_dashboard_settings.json"

    def _collect_settings(self) -> dict:
        return {
            "resourcepack_root": (self._rp_root.text() or "").strip(),
            "blender_exe": (self._blender_exe.text() or "").strip(),
            "scan_root": (self._scan_root.text() or "").strip(),
            "filter_text": (self._filter_text.text() or "").strip(),
            "only_missing_blend": bool(self._only_missing_cache.isChecked()),
            "only_missing_renders": bool(self._only_missing_renders.isChecked()),
        }

    def _apply_settings(self, s: dict) -> None:
        self._loading_settings = True
        try:
            self._rp_root.setText(str(s.get("resourcepack_root") or ""))
            self._blender_exe.setText(str(s.get("blender_exe") or ""))
            self._scan_root.setText(str(s.get("scan_root") or ""))
            self._filter_text.setText(str(s.get("filter_text") or ""))
            self._only_missing_cache.setChecked(bool(s.get("only_missing_blend") or False))
            self._only_missing_renders.setChecked(bool(s.get("only_missing_renders") or False))
        finally:
            self._loading_settings = False

    def _load_settings(self) -> None:
        p = self._settings_path()
        if not p.exists():
            return
        try:
            s = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return
        if not isinstance(s, dict):
            return
        self._apply_settings(s)

    def _schedule_save_settings(self) -> None:
        if self._loading_settings:
            return
        self._settings_timer.start(250)

    def _save_settings_now(self) -> None:
        if self._loading_settings:
            return
        p = self._settings_path()
        try:
            p.write_text(json.dumps(self._collect_settings(), indent=2), encoding="utf-8")
        except Exception:
            return

    def closeEvent(self, event):
        try:
            self._save_settings_now()
        except Exception:
            pass
        return super().closeEvent(event)

    def _log_settings(self, action: str) -> None:
        rp = (self._rp_root.text() or "").strip()
        scan_root = (self._scan_root.text() or "").strip()
        blender_raw = (self._blender_exe.text() or "").strip()
        blender_res = self._blender_cmd()
        self._append_log(f"[settings] action={action}")
        self._append_log(f"[settings] resourcepack_root={rp}")
        self._append_log(f"[settings] scan_root={scan_root}")
        self._append_log(f"[settings] blender_raw={blender_raw}")
        self._append_log(f"[settings] blender_resolved={blender_res}")

    def _set_state(self, s: str) -> None:
        self._state.setText(s)

    def _browse_rp(self) -> None:
        d = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Resourcepack Root")
        if d:
            self._rp_root.setText(d)

    def _browse_blender(self) -> None:
        f, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select Blender Executable")
        if f:
            self._blender_exe.setText(f)

    def _browse_scan_root(self) -> None:
        d = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Models Subfolder")
        if d:
            self._scan_root.setText(d)

    def _scan(self) -> None:
        rp = Path(self._rp_root.text().strip() or "")
        if not rp.exists():
            QtWidgets.QMessageBox.warning(self, "Missing", "Resourcepack root does not exist")
            return

        self._log_settings("scan")
        self._items.clear()
        self._items_by_blend.clear()
        scan_root_raw = (self._scan_root.text() or "").strip()
        found_paths: list[Path] = []
        if scan_root_raw:
            found_paths = _iter_items_under(Path(scan_root_raw))
        else:
            found_paths.extend(_iter_model_sources(rp))
            found_paths.extend(_iter_blends(rp))

        by_blend: dict[Path, ModelItem] = {}
        for p in found_paths:
            try:
                it = _identify_item(p, rp)
            except Exception:
                continue
            key = it.cache_blend_path().resolve()
            prev = by_blend.get(key)
            if prev is None:
                by_blend[key] = it
            else:
                source = prev.source_path or it.source_path
                by_blend[key] = ModelItem(
                    source_path=source,
                    blend_path=prev.blend_path,
                    resourcepack_root=prev.resourcepack_root,
                    namespace=prev.namespace,
                    render_rel_dir=prev.render_rel_dir,
                    model_stem=prev.model_stem,
                )

        self._items = list(by_blend.values())
        self._items.sort(key=lambda x: (x.namespace, str(x.render_rel_dir), x.model_stem))
        self._items_by_blend = {str(it.cache_blend_path().resolve()): it for it in self._items}

        self._populate_table(self._items)
        self._append_log(f"Scanned items: {len(self._items)}")

    def _populate_table(self, items: list[ModelItem]) -> None:
        self._table.setRowCount(0)
        self._table.setRowCount(len(items))

        for r, it in enumerate(items):
            chk = QtWidgets.QTableWidgetItem("")
            chk.setFlags(QtCore.Qt.ItemIsUserCheckable | QtCore.Qt.ItemIsEnabled)
            chk.setCheckState(QtCore.Qt.Unchecked)

            model = QtWidgets.QTableWidgetItem(it.model_id())
            src_txt = str(it.source_path) if it.source_path is not None else ""
            j = QtWidgets.QTableWidgetItem(src_txt)
            cpath = it.cache_blend_path()
            c = QtWidgets.QTableWidgetItem(str(cpath))

            has_cache = cpath.exists()
            has_renders = _has_any_png(it.render_output_dir())

            hc = QtWidgets.QTableWidgetItem("yes" if has_cache else "no")
            hr = QtWidgets.QTableWidgetItem("yes" if has_renders else "no")

            self._table.setItem(r, 0, chk)
            self._table.setItem(r, 1, model)
            self._table.setItem(r, 2, j)
            self._table.setItem(r, 3, c)
            self._table.setItem(r, 4, hc)
            self._table.setItem(r, 5, hr)

    def _apply_filter(self) -> None:
        q = (self._filter_text.text() or "").strip().lower()
        only_missing_cache = self._only_missing_cache.isChecked()
        only_missing_renders = self._only_missing_renders.isChecked()

        out: list[ModelItem] = []
        for it in self._items:
            if q:
                src = str(it.source_path) if it.source_path is not None else ""
                hay = (it.model_id() + "\n" + src + "\n" + str(it.cache_blend_path())).lower()
                if q not in hay:
                    continue
            if only_missing_cache and it.cache_blend_path().exists():
                continue
            if only_missing_renders and _has_any_png(it.render_output_dir()):
                continue
            out.append(it)

        self._populate_table(out)

    def _selected_items(self) -> list[ModelItem]:
        items: list[ModelItem] = []
        for r in range(self._table.rowCount()):
            chk = self._table.item(r, 0)
            if chk is None or chk.checkState() != QtCore.Qt.Checked:
                continue
            blend_path = Path(self._table.item(r, 3).text()).resolve()
            it = self._items_by_blend.get(str(blend_path))
            if it is not None:
                items.append(it)
        return items

    def _blender_cmd(self) -> str:
        raw = (self._blender_exe.text() or "").strip()
        if raw.startswith('"') and raw.endswith('"') and len(raw) >= 2:
            raw = raw[1:-1].strip()
        if raw.startswith("'") and raw.endswith("'") and len(raw) >= 2:
            raw = raw[1:-1].strip()
        if raw:
            try:
                if ("\\" in raw or "/" in raw) and not Path(raw).exists():
                    self._append_log(f"[warn] Blender exe path does not exist: {raw}")
            except Exception:
                pass
            return raw
        found = shutil.which("blender") or shutil.which("blender.exe")
        if found:
            return found
        return "blender"

    def _script_path(self, name: str) -> str:
        here = Path(__file__).resolve().parent
        return str((here / name).resolve())

    def _cancel_jobs(self) -> None:
        self._jobs.clear()
        self._append_log("Cancelled")

    def _generate_missing(self) -> None:
        rp = Path(self._rp_root.text().strip() or "")
        if not rp.exists():
            QtWidgets.QMessageBox.warning(self, "Missing", "Resourcepack root does not exist")
            return

        self._log_settings("generate_missing")
        blender = self._blender_cmd()
        script = self._script_path("mc_resourcepack_blender_generate_blends.py")
        self._append_log(f"[settings] script_generate={script}")

        count = 0
        for it in self._items:
            if it.cache_blend_path().exists():
                continue
            if it.source_path is None:
                continue
            self._append_log(f"[job] generate model={it.model_id()}")
            self._append_log(f"[job] source={it.source_path}")
            self._append_log(f"[job] cache={it.cache_blend_path()}")
            self._append_log(f"[job] overwrite=false ensure_cameras=true")
            args = [
                "-b",
                "-P",
                script,
                "--",
                "--json",
                str(it.source_path),
                "--resourcepack_root",
                str(rp),
                "--overwrite",
                "false",
                "--ensure_cameras",
                "true",
            ]
            self._jobs.enqueue(blender, args, f"Generate {it.model_id()}")
            count += 1

        self._append_log(f"Enqueued generate missing: {count}")

    def _regenerate_selected(self) -> None:
        rp = Path(self._rp_root.text().strip() or "")
        if not rp.exists():
            QtWidgets.QMessageBox.warning(self, "Missing", "Resourcepack root does not exist")
            return

        self._log_settings("regenerate_selected")
        sel = self._selected_items()
        if not sel:
            QtWidgets.QMessageBox.information(self, "None", "No selected items")
            return

        blender = self._blender_cmd()
        script = self._script_path("mc_resourcepack_blender_generate_blends.py")
        self._append_log(f"[settings] script_generate={script}")

        for it in sel:
            if it.source_path is None:
                self._append_log(f"Skip regenerate (no source): {it.model_id()}")
                continue
            self._append_log(f"[job] regenerate model={it.model_id()}")
            self._append_log(f"[job] source={it.source_path}")
            self._append_log(f"[job] cache={it.cache_blend_path()}")
            self._append_log(f"[job] overwrite=true ensure_cameras=true")
            args = [
                "-b",
                "-P",
                script,
                "--",
                "--json",
                str(it.source_path),
                "--resourcepack_root",
                str(rp),
                "--overwrite",
                "true",
                "--ensure_cameras",
                "true",
            ]
            self._jobs.enqueue(blender, args, f"Regenerate {it.model_id()}")

        self._append_log(f"Enqueued regenerate: {len(sel)}")

    def _render_selected(self) -> None:
        rp = Path(self._rp_root.text().strip() or "")
        if not rp.exists():
            QtWidgets.QMessageBox.warning(self, "Missing", "Resourcepack root does not exist")
            return

        self._log_settings("render_selected")
        sel = self._selected_items()
        if not sel:
            QtWidgets.QMessageBox.information(self, "None", "No selected items")
            return

        self._enqueue_render(sel)

    def _render_all(self) -> None:
        rp = Path(self._rp_root.text().strip() or "")
        if not rp.exists():
            QtWidgets.QMessageBox.warning(self, "Missing", "Resourcepack root does not exist")
            return

        self._log_settings("render_all")
        if not self._items:
            QtWidgets.QMessageBox.information(self, "None", "No scanned items")
            return

        self._enqueue_render(list(self._items))

    def _enqueue_render(self, items: list[ModelItem]) -> None:
        rp = Path(self._rp_root.text().strip() or "")
        blender = self._blender_cmd()
        script = self._script_path("mc_resourcepack_blender_render_blends.py")
        self._append_log(f"[settings] script_render={script}")

        enq = 0
        for it in items:
            cache = it.cache_blend_path()
            if not cache.exists():
                self._append_log(f"Skip render (no cache): {it.model_id()}")
                continue
            out_dir = it.render_output_dir()
            out_prefix = it.render_output_prefix()
            self._append_log(f"[job] render model={it.model_id()}")
            self._append_log(f"[job] blend={cache}")
            self._append_log(f"[job] output_dir={out_dir}")
            self._append_log(f"[job] output_prefix={out_prefix}")
            self._append_log(f"[job] ensure_cameras=false")
            args = [
                "-b",
                str(cache),
                "-P",
                script,
                "--",
                "--output_dir",
                str(out_dir),
                "--output_prefix",
                str(out_prefix),
                "--ensure_cameras",
                "false",
            ]
            self._jobs.enqueue(blender, args, f"Render {it.model_id()}")
            enq += 1

        self._append_log(f"Enqueued render: {enq}")


def main() -> None:
    app = QtWidgets.QApplication(sys.argv)
    try:
        app.setStyle("Fusion")
    except Exception:
        pass
    app.setStyleSheet(_DARK_QSS)
    w = Dashboard()
    w.show()
    raise SystemExit(app.exec())


if __name__ == "__main__":
    main()
