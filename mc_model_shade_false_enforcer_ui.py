from __future__ import annotations

import json
import re
import sys
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


def _apply_dark_theme(app: QtWidgets.QApplication) -> None:
    app.setStyle("Fusion")
    app.setStyleSheet(
        "QWidget { background: #0b0b0d; color: #e8e8ea; font-size: 12px; }"
        "QLabel { color: #e8e8ea; }"
        "QGroupBox { border: 1px solid #1f2024; margin-top: 8px; }"
        "QGroupBox::title { subcontrol-origin: margin; left: 8px; padding: 0 4px 0 4px; color: #cfcfd4; }"
        "QLineEdit, QComboBox { background: #000000; color: #ffffff; border: 1px solid #2a2b30; padding: 4px; border-radius: 4px; }"
        "QPlainTextEdit { background: #000000; color: #ffffff; border: 1px solid #2a2b30; padding: 6px; border-radius: 4px; }"
        "QLineEdit:focus, QComboBox:focus { border: 1px solid #3a6ea5; }"
        "QComboBox QAbstractItemView { background: #000000; color: #ffffff; selection-background-color: #2b4a6b; }"
        "QCheckBox { spacing: 6px; }"
        "QCheckBox::indicator { width: 14px; height: 14px; }"
        "QPushButton { background: #15161a; color: #e8e8ea; border: 1px solid #2a2b30; padding: 6px 10px; border-radius: 6px; }"
        "QPushButton:hover { border: 1px solid #3a3b42; }"
        "QPushButton:disabled { color: #777780; border: 1px solid #1a1b1f; background: #101114; }"
        "QPushButton#btn_apply { background: #1b3326; border: 1px solid #2d5b3f; }"
        "QPushButton#btn_apply:hover { border: 1px solid #3d7a55; }"
        "QPushButton#btn_save_as { background: #241a2c; border: 1px solid #4a2c63; }"
        "QPushButton#btn_save_as:hover { border: 1px solid #6a3b90; }"
    )


def _format_json(obj: dict) -> str:
    return json.dumps(obj, indent=2, ensure_ascii=False) + "\n"


def _skip_ws(s: str, i: int) -> int:
    n = len(s)
    while i < n and s[i] in " \t\r\n":
        i += 1
    return int(i)


def _skip_string(s: str, i: int) -> int:
    n = len(s)
    if i >= n or s[i] != '"':
        raise ValueError("Expected string")
    i += 1
    while i < n:
        ch = s[i]
        if ch == "\\":
            i += 2
            continue
        if ch == '"':
            return int(i + 1)
        i += 1
    raise ValueError("Unterminated string")


def _skip_number(s: str, i: int) -> int:
    n = len(s)
    if i < n and s[i] == "-":
        i += 1
    if i < n and s[i] == "0":
        i += 1
    else:
        while i < n and s[i].isdigit():
            i += 1
    if i < n and s[i] == ".":
        i += 1
        while i < n and s[i].isdigit():
            i += 1
    if i < n and s[i] in "eE":
        i += 1
        if i < n and s[i] in "+-":
            i += 1
        while i < n and s[i].isdigit():
            i += 1
    return int(i)


def _skip_value(s: str, i: int) -> int:
    i = _skip_ws(s, int(i))
    if i >= len(s):
        raise ValueError("Unexpected end")
    ch = s[i]
    if ch == '"':
        return _skip_string(s, i)
    if ch == "{":
        return _skip_braced(s, i, "{", "}")
    if ch == "[":
        return _skip_braced(s, i, "[", "]")
    if ch in "-0123456789":
        return _skip_number(s, i)
    if s.startswith("true", i):
        return int(i + 4)
    if s.startswith("false", i):
        return int(i + 5)
    if s.startswith("null", i):
        return int(i + 4)
    raise ValueError("Invalid value")


def _skip_braced(s: str, i: int, open_ch: str, close_ch: str) -> int:
    n = len(s)
    if i >= n or s[i] != open_ch:
        raise ValueError("Expected open")
    depth = 0
    in_str = False
    esc = False
    while i < n:
        ch = s[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
            elif ch == open_ch:
                depth += 1
            elif ch == close_ch:
                depth -= 1
                if depth == 0:
                    return int(i + 1)
        i += 1
    raise ValueError("Unterminated structure")


def _find_root_elements_array_span(s: str) -> tuple[int, int]:
    n = len(s)
    i = 0
    depth_obj = 0
    depth_arr = 0
    in_str = False
    esc = False
    last_str: Optional[str] = None
    last_str_end = -1

    while i < n:
        ch = s[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
                if last_str_end == -1:
                    last_str_end = int(i + 1)
        else:
            if ch == '"':
                j = _skip_string(s, i)
                last_str = json.loads(s[i:j])
                last_str_end = int(j)
                i = j - 1
            elif ch == "{":
                depth_obj += 1
            elif ch == "}":
                depth_obj -= 1
            elif ch == "[":
                depth_arr += 1
            elif ch == "]":
                depth_arr -= 1
            elif ch == ":":
                if last_str == "elements" and depth_obj == 1 and depth_arr == 0 and last_str_end != -1:
                    j = _skip_ws(s, i + 1)
                    if j < n and s[j] == "[":
                        end = _skip_braced(s, j, "[", "]")
                        return int(j), int(end)
        i += 1

    raise ValueError("Could not locate root elements[] array")


def _iter_array_object_spans(s: str, arr_start: int, arr_end: int) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    i = int(arr_start + 1)
    end = int(arr_end - 1)
    while i < end:
        i = _skip_ws(s, i)
        if i >= end:
            break
        if s[i] == ",":
            i += 1
            continue
        if s[i] == "{":
            j = _skip_braced(s, i, "{", "}")
            spans.append((int(i), int(j)))
            i = j
            continue
        _ = _skip_value(s, i)
        i = _
    return spans


def _scan_object_props(s: str, obj_start: int, obj_end: int) -> list[dict]:
    props: list[dict] = []
    i = _skip_ws(s, int(obj_start + 1))
    while i < obj_end:
        i = _skip_ws(s, i)
        if i >= obj_end:
            break
        if s[i] == "}":
            break
        if s[i] == ",":
            i += 1
            continue
        if s[i] != '"':
            raise ValueError("Expected key string")
        key_start = int(i)
        key_end = _skip_string(s, i)
        key = json.loads(s[key_start:key_end])
        j = _skip_ws(s, key_end)
        if j >= obj_end or s[j] != ":":
            raise ValueError("Expected ':'")
        val_start = int(j + 1)
        val_end = _skip_value(s, val_start)
        k = _skip_ws(s, val_end)
        has_comma = False
        prop_end = int(val_end)
        if k < obj_end and s[k] == ",":
            has_comma = True
            prop_end = int(k + 1)
            i = int(k + 1)
        else:
            i = int(val_end)
        props.append(
            {
                "key": str(key),
                "key_start": int(key_start),
                "key_end": int(key_end),
                "val_start": int(val_start),
                "val_end": int(val_end),
                "end": int(prop_end),
                "has_comma": bool(has_comma),
            }
        )
    return props


def _detect_newline(s: str) -> str:
    return "\r\n" if "\r\n" in s else "\n"


def _indent_for_prop_line(s: str, pos: int) -> str:
    nl = s.rfind("\n", 0, int(pos))
    if nl == -1:
        return ""
    i = nl + 1
    out = []
    while i < len(s) and s[i] in " \t":
        out.append(s[i])
        i += 1
    return "".join(out)


def _enforce_shade_false_text(raw_text: str) -> tuple[str, int, int, int]:
    _ = json.loads(raw_text)
    if not isinstance(_, dict):
        raise ValueError("JSON root must be an object")
    if not isinstance(_.get("elements"), list):
        raise ValueError("JSON missing elements[]")

    nl = _detect_newline(raw_text)
    arr_start, arr_end = _find_root_elements_array_span(raw_text)
    spans = _iter_array_object_spans(raw_text, arr_start, arr_end)
    out = str(raw_text)
    inserted = 0
    flipped = 0

    for obj_start, obj_end in reversed(spans):
        props = _scan_object_props(out, obj_start, obj_end)
        by_key = {p["key"]: p for p in props}

        shade = by_key.get("shade")
        if shade is not None:
            m = re.match(r"\s*(true|false)", out[shade["val_start"] : shade["val_end"]])
            if m is not None and str(m.group(1)) == "true":
                v0 = int(shade["val_start"]) + int(m.start(1))
                v1 = int(shade["val_start"]) + int(m.end(1))
                out = out[:v0] + "false" + out[v1:]
                flipped += 1
            continue

        anchor = None
        if "to" in by_key:
            anchor = by_key["to"]
        elif "from" in by_key:
            anchor = by_key["from"]

        if anchor is None:
            indent = _indent_for_prop_line(out, obj_start + 1)
            if "\n" in out[obj_start:obj_end]:
                insert_at = int(obj_start + 1)
                insert_txt = f"{nl}{indent}\"shade\": false,"
                out = out[:insert_at] + insert_txt + out[insert_at:]
                inserted += 1
            else:
                insert_at = int(obj_start + 1)
                out = out[:insert_at] + "\"shade\": false," + out[insert_at:]
                inserted += 1
            continue

        multiline = "\n" in out[obj_start:obj_end]
        insert_at = int(anchor["end"])
        needs_comma_before = not bool(anchor["has_comma"])

        if needs_comma_before:
            insert_at = int(anchor["val_end"])
            out = out[:insert_at] + "," + out[insert_at:]
            insert_at += 1

        if multiline:
            indent = _indent_for_prop_line(out, int(anchor["key_start"]))
            tail_props = [p for p in props if int(p["key_start"]) > int(anchor["key_start"])]
            comma = "," if len(tail_props) > 0 else ""
            insert_txt = f"{nl}{indent}\"shade\": false{comma}"
            out = out[:insert_at] + insert_txt + out[insert_at:]
            inserted += 1
        else:
            tail_props = [p for p in props if int(p["key_start"]) > int(anchor["key_start"])]
            comma = "," if len(tail_props) > 0 else ""
            insert_txt = f" \"shade\": false{comma}"
            out = out[:insert_at] + insert_txt + out[insert_at:]
            inserted += 1

    total = int(len(spans))
    return out, total, int(inserted), int(flipped)


def _suggest_output_path(src: Path) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = src.with_name(f"{src.stem}_SHADE_FALSE_{ts}{src.suffix}")
    if not base.exists():
        return base

    i = 2
    while True:
        p = src.with_name(f"{src.stem}_SHADE_FALSE_{ts}_{i}{src.suffix}")
        if not p.exists():
            return p
        i += 1


class ShadeFalseMainWindow(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Minecraft Model: Enforce shade=false")

        self._source_path: Optional[Path] = None
        self._model_in: Optional[dict] = None
        self._model_out: Optional[dict] = None
        self._raw_in: Optional[str] = None
        self._raw_out: Optional[str] = None

        self._txt_path = QtWidgets.QLineEdit("")
        self._txt_path.setReadOnly(True)

        self._btn_load = QtWidgets.QPushButton("Load JSON")
        self._btn_load.clicked.connect(self._on_load)

        self._chk_autosave = QtWidgets.QCheckBox("Auto-save next to source")
        self._chk_autosave.setChecked(True)

        self._btn_apply = QtWidgets.QPushButton("Enforce + Save")
        self._btn_apply.setObjectName("btn_apply")
        self._btn_apply.clicked.connect(self._on_apply)
        self._btn_apply.setEnabled(False)

        self._btn_save_as = QtWidgets.QPushButton("Save As...")
        self._btn_save_as.setObjectName("btn_save_as")
        self._btn_save_as.clicked.connect(self._on_save_as)
        self._btn_save_as.setEnabled(False)

        self._lbl_status = QtWidgets.QLabel("Ready")
        self._lbl_status.setWordWrap(True)

        self._txt_log = QtWidgets.QPlainTextEdit()
        self._txt_log.setReadOnly(True)
        self._txt_log.setMaximumBlockCount(20000)
        self._txt_log.setLineWrapMode(QtWidgets.QPlainTextEdit.LineWrapMode.NoWrap)
        self._txt_log.setFont(QtGui.QFont("Consolas", 10))

        self._txt_out = QtWidgets.QPlainTextEdit()
        self._txt_out.setReadOnly(True)
        self._txt_out.setMaximumBlockCount(20000)
        self._txt_out.setLineWrapMode(QtWidgets.QPlainTextEdit.LineWrapMode.NoWrap)
        self._txt_out.setFont(QtGui.QFont("Consolas", 10))

        model_box = QtWidgets.QGroupBox("Model")
        model_form = QtWidgets.QFormLayout(model_box)
        model_form.setContentsMargins(8, 6, 8, 6)
        model_form.setVerticalSpacing(4)
        model_form.addRow("Path", self._txt_path)
        model_form.addRow(self._btn_load)
        model_form.addRow("", self._chk_autosave)

        actions_box = QtWidgets.QGroupBox("Actions")
        actions_layout = QtWidgets.QVBoxLayout(actions_box)
        actions_layout.setContentsMargins(8, 6, 8, 6)
        actions_layout.setSpacing(6)
        actions_layout.addWidget(self._btn_apply)
        actions_layout.addWidget(self._btn_save_as)
        actions_layout.addWidget(self._lbl_status)

        output_tabs = QtWidgets.QTabWidget()
        output_tabs.addTab(self._txt_log, "Log")
        output_tabs.addTab(self._txt_out, "Output JSON")

        out_box = QtWidgets.QGroupBox("Output")
        out_layout = QtWidgets.QVBoxLayout(out_box)
        out_layout.setContentsMargins(8, 6, 8, 6)
        out_layout.setSpacing(6)
        out_layout.addWidget(output_tabs)

        left = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(6)
        left_layout.addWidget(model_box)
        left_layout.addWidget(actions_box)
        left_layout.addStretch(1)

        splitter = QtWidgets.QSplitter()
        splitter.setOrientation(QtCore.Qt.Orientation.Horizontal)
        splitter.addWidget(left)
        splitter.addWidget(out_box)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setCollapsible(0, False)

        self.setCentralWidget(splitter)

    def _append_log(self, s: str) -> None:
        t = str(s)
        if t:
            self._txt_log.appendPlainText(t)

    def _load_path(self, p: Path) -> None:
        raw = p.read_text(encoding="utf-8")
        model = json.loads(raw)
        if not isinstance(model, dict):
            raise ValueError("JSON root must be an object")
        if not isinstance(model.get("elements"), list):
            raise ValueError("JSON missing elements[]")

        self._source_path = p
        self._model_in = model
        self._model_out = None
        self._raw_in = str(raw)
        self._raw_out = None

        self._txt_path.setText(str(p))
        self._btn_apply.setEnabled(True)
        self._btn_save_as.setEnabled(False)
        self._lbl_status.setText(f"Loaded: {p.name}")
        self._txt_out.setPlainText("")

        self._txt_log.setPlainText("")
        self._append_log(f"Loaded: {p}")
        self._append_log(f"Elements: {len(list(model.get('elements') or []))}")

    def _save_out_model_to(self, path: Path) -> None:
        if self._raw_out is None:
            raise ValueError("No output to save")
        path.write_text(str(self._raw_out), encoding="utf-8", newline="")

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
            self._load_path(Path(path))
        except Exception as e:
            self._txt_log.setPlainText(f"Load failed: {e}")
            self._lbl_status.setText("Load failed")
            self._btn_apply.setEnabled(False)
            self._btn_save_as.setEnabled(False)

    def _on_apply(self) -> None:
        if self._raw_in is None:
            return

        try:
            out_text, total, inserted, flipped = _enforce_shade_false_text(self._raw_in)
            self._raw_out = str(out_text)
            self._txt_out.setPlainText(str(out_text))
            self._btn_save_as.setEnabled(True)

            self._txt_log.setPlainText("")
            self._append_log(f"Elements processed: {total}")
            self._append_log(f"Inserted shade:    {inserted}")
            self._append_log(f"Flipped true->false: {flipped}")

            if self._chk_autosave.isChecked() and self._source_path is not None:
                out_path = _suggest_output_path(self._source_path)
                self._save_out_model_to(out_path)
                self._append_log(f"Saved: {out_path}")
                self._lbl_status.setText(f"Saved: {out_path.name}")
            else:
                self._lbl_status.setText("Done (not saved)")

        except Exception as e:
            self._txt_log.setPlainText(f"Apply failed: {e}")
            self._lbl_status.setText("Apply failed")

    def _on_save_as(self) -> None:
        if self._raw_out is None:
            return

        default = Path.cwd() / "model_SHADE_FALSE.json"
        if self._source_path is not None:
            default = _suggest_output_path(self._source_path)

        out_path, _filter = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save updated model JSON",
            str(default),
            "JSON (*.json)",
        )
        if not out_path:
            return

        try:
            self._save_out_model_to(Path(out_path))
            self._append_log(f"Saved: {out_path}")
            self._lbl_status.setText(f"Saved: {Path(out_path).name}")
        except Exception as e:
            self._txt_log.setPlainText(f"Save failed: {e}")
            self._lbl_status.setText("Save failed")


def main() -> None:
    app = QtWidgets.QApplication(sys.argv)
    _apply_dark_theme(app)

    w = ShadeFalseMainWindow()
    w.resize(1100, 760)
    w.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
