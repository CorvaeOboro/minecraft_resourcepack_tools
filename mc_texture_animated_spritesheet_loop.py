"""Vertical Spritesheet Loop Generator

Generate an animated vertical spritesheet (stacked frames) by baking a brightness/contrast loop into an input PNG.
Supports an emissive mask for selective brightening and an optional scrolling/tiled layer for seamless texture motion.

TOOLSGROUP::RENDER
SORTGROUP::7
SORTPRIORITY::73
STATUS::active
VERSION::20260308

Core functions
- make_vertical_spritesheet_loop(...): Pure image processing. Used by both CLI and GUI.
- run_gui(): PySide6 UI with drag/drop, previews, and playback.

Key features
- Emissive mask
  - Mask is a same-size RGBA image.
  - RGB->luminance combined with alpha becomes intensity.
  - --emissive-mask-scale boosts mask strength.
  - --emissive-blend-mode=white blends toward white (screen-like) to avoid saturation spikes.
- Seamless scrolling layer
  - A separate RGBA texture is tiled to the base size and offset every frame.
  - Offsets wrap/tile, so the animation loops cleanly frame N -> frame 0.
- Settings persistence
  - Each successful generation writes a timestamped settings JSON next to the output PNG.
  - The latest settings are also written to mc_make_vertical_spritesheet_loop_settings.json next to this script.
  - GUI auto-loads that last-used settings file on startup.

GUI usage
- Launch: python DEV/mc_make_vertical_spritesheet_loop.py
- Drag/drop input/mask/scroll-layer PNGs directly onto the preview tiles.
- Set Frametime (ticks) to match Minecraft mcmeta style (10=slow, 2=fast).

CLI usage examples
- Basic loop:
  python DEV/mc_make_vertical_spritesheet_loop.py in.png --frames 8 --apex-frame 5

- Project example (Arborea animated texture base):
  python DEV/mc_make_vertical_spritesheet_loop.py \
    arborea_1_19_2/src/main/resources/assets/arborea/textures/blocks/glimmerfluidstatic.png \
    --out arborea_1_19_2/src/main/resources/assets/arborea/textures/blocks/glimmerfluidstatic_spritesheet.png \
    --frames 8 --apex-frame 5 --ramp-brightness 0.25

- Mana pylon energy distortion spritesheet (typical workflow)
  - in.png: base energy/distortion tile (RGBA)
  - em.png: emissive mask for the hot core
  - scroll.png: noise/warp texture to scroll and create motion

  python DEV/mc_make_vertical_spritesheet_loop.py in.png \
    --out mana_pylon_energy.png \
    --frames 8 --apex-frame 5 \
    --overall-brightness 1.0 --ramp-brightness 0.35 \
    --emissive-mask em.png --emissive-mask-scale 2.5 \
    --emissive-blend-mode white --emissive-brightness 0.85 \
    --scroll-layer scroll.png --scroll-layer-enabled \
    --scroll-layer-blend-mode screen --scroll-layer-opacity 0.65 \
    --scroll-layer-dy-pct-per-frame 0.25
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional


def _script_settings_path() -> Path:
    return Path(__file__).with_name("mc_make_vertical_spritesheet_loop_settings.json")


def _timestamp_suffix() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")


def _save_settings_files(*, out_png_path: Path, settings: dict) -> None:
    ts = _timestamp_suffix()
    stamp_path = out_png_path.with_name(out_png_path.stem + f"_settings_{ts}.json")
    _write_json(stamp_path, settings)
    _write_json(_script_settings_path(), settings)


def _triangle_weight(i: int, frames: int, apex_frame: int) -> float:
    if frames <= 1:
        return 0.0

    if apex_frame <= 1:
        return 0.0

    if apex_frame > frames:
        apex_frame = frames

    p = i / frames
    p_apex = (apex_frame - 1) / frames

    if p_apex <= 0.0:
        return 0.0

    if p <= p_apex:
        w = p / p_apex
    else:
        denom = (1.0 - p_apex)
        w = (1.0 - p) / denom if denom > 0 else 0.0

    if w < 0.0:
        return 0.0
    if w > 1.0:
        return 1.0
    return float(w)


def _enhance_rgb(rgb, *, brightness: float, contrast: float):
    from PIL import ImageEnhance

    out = rgb

    if brightness != 1.0:
        out = ImageEnhance.Brightness(out).enhance(brightness)

    if contrast != 1.0:
        out = ImageEnhance.Contrast(out).enhance(contrast)

    return out


def _scale_rgba(texture_rgba, *, scale: float):
    from PIL import Image

    scale = float(scale)
    if scale == 1.0:
        return texture_rgba
    if scale <= 0.0:
        return Image.new("RGBA", (1, 1), (0, 0, 0, 0))

    if texture_rgba.mode != "RGBA":
        texture_rgba = texture_rgba.convert("RGBA")

    tw = max(1, int(texture_rgba.width))
    th = max(1, int(texture_rgba.height))
    nw = max(1, int(round(tw * scale)))
    nh = max(1, int(round(th * scale)))
    resampling = getattr(Image, "Resampling", None)
    if resampling is not None:
        return texture_rgba.resize((nw, nh), resample=resampling.NEAREST)
    return texture_rgba.resize((nw, nh), resample=Image.NEAREST)


def _overlay_rgb(base_rgb, layer_rgb):
    from PIL import Image, ImageMath

    if base_rgb.mode != "RGB":
        base_rgb = base_rgb.convert("RGB")
    if layer_rgb.mode != "RGB":
        layer_rgb = layer_rgb.convert("RGB")

    br, bg, bb = base_rgb.split()
    lr, lg, lb = layer_rgb.split()

    def _overlay_band(b, l):
        return ImageMath.eval(
            "convert((b < 128) * (2*b*l/255) + (b >= 128) * (255 - 2*(255-b)*(255-l)/255), 'L')",
            b=b,
            l=l,
        )

    return Image.merge("RGB", (_overlay_band(br, lr), _overlay_band(bg, lg), _overlay_band(bb, lb)))


def _load_emissive_intensity(mask_path: Path, *, size: tuple[int, int], scale: float):
    from PIL import Image, ImageChops

    mask = Image.open(mask_path)
    mask = mask.convert("RGBA")

    if mask.size != size:
        raise SystemExit(f"Emissive mask size mismatch: expected {size[0]}x{size[1]}, got {mask.size[0]}x{mask.size[1]}")

    r, g, b, a = mask.split()
    lum = Image.merge("RGB", (r, g, b)).convert("L")
    intensity = ImageChops.multiply(lum, a)

    scale = float(scale)
    if scale != 1.0:
        if scale <= 0.0:
            intensity = intensity.point(lambda _p: 0)
        else:
            intensity = intensity.point(lambda p: 255 if int(p * scale) >= 255 else int(p * scale))
    return intensity


def _make_tiled_layer(base_size: tuple[int, int], texture_rgba, *, offset_x: float, offset_y: float):
    from PIL import Image

    bw, bh = base_size

    if texture_rgba.mode != "RGBA":
        texture_rgba = texture_rgba.convert("RGBA")

    tw = max(1, int(texture_rgba.width))
    th = max(1, int(texture_rgba.height))

    off_x = int(round(float(offset_x)))
    off_y = int(round(float(offset_y)))

    start_x = -((off_x % tw + tw) % tw)
    start_y = -((off_y % th + th) % th)

    out = Image.new("RGBA", (bw, bh), (0, 0, 0, 0))

    y = start_y
    while y < bh:
        x = start_x
        while x < bw:
            out.paste(texture_rgba, (x, y), texture_rgba)
            x += tw
        y += th

    return out


def _blend_rgb_with_layer(*, base_rgb, layer_rgba, opacity_0_1: float, blend_mode: str):
    from PIL import Image, ImageChops

    opacity_0_1 = max(0.0, min(1.0, float(opacity_0_1)))
    if opacity_0_1 <= 0.0:
        return base_rgb

    if base_rgb.mode != "RGB":
        base_rgb = base_rgb.convert("RGB")
    if layer_rgba.mode != "RGBA":
        layer_rgba = layer_rgba.convert("RGBA")

    r, g, b, a = layer_rgba.split()
    if opacity_0_1 != 1.0:
        a = a.point(lambda v: int(v * opacity_0_1))
    layer_rgb = Image.merge("RGB", (r, g, b))

    mode = (blend_mode or "normal").strip().lower()
    if mode == "normal":
        base_a = Image.new("L", base_rgb.size, 255)
        base_rgba = Image.merge("RGBA", (*base_rgb.split(), base_a))
        layer_adj = Image.merge("RGBA", (r, g, b, a))
        out = Image.alpha_composite(base_rgba, layer_adj)
        return out.convert("RGB")

    if mode == "screen":
        blended = ImageChops.screen(base_rgb, layer_rgb)
    elif mode == "overlay":
        blended = _overlay_rgb(base_rgb, layer_rgb)
    elif mode == "lighten":
        blended = ImageChops.lighter(base_rgb, layer_rgb)
    elif mode == "add":
        blended = ImageChops.add(base_rgb, layer_rgb)
    else:
        blended = ImageChops.screen(base_rgb, layer_rgb)

    return Image.composite(blended, base_rgb, a)


def _generate_frames(
    *,
    base_rgba,
    frames: int,
    apex_frame: int,
    overall_brightness: float,
    overall_contrast: float,
    ramp_brightness: float,
    ramp_contrast: float,
    emissive_intensity,
    emissive_brightness: float,
    emissive_contrast: float,
    emissive_blend_mode: str,
    scroll_layer_rgba,
    scroll_layer_enabled: bool,
    scroll_layer_opacity: float,
    scroll_layer_blend_mode: str,
    scroll_layer_scale: float,
    scroll_layer_dx_pct_per_frame: float,
    scroll_layer_dy_pct_per_frame: float,
    zero_rgb_where_alpha0: bool,
):
    from PIL import Image

    w, h = base_rgba.size
    base_rgb = base_rgba.convert("RGB")
    base_a = base_rgba.split()[3]

    out_frames = []

    scroll_src = scroll_layer_rgba
    if scroll_src is not None and bool(scroll_layer_enabled):
        scroll_src = _scale_rgba(scroll_src, scale=float(scroll_layer_scale))

    for fi in range(frames):
        weight = _triangle_weight(fi, frames, apex_frame)

        frame_rgb = base_rgb

        frame_rgb = _enhance_rgb(
            frame_rgb,
            brightness=overall_brightness,
            contrast=overall_contrast,
        )

        frame_rgb = _enhance_rgb(
            frame_rgb,
            brightness=1.0 + (weight * ramp_brightness),
            contrast=1.0 + (weight * ramp_contrast),
        )

        if emissive_intensity is not None and (emissive_brightness != 0.0 or emissive_contrast != 0.0):
            mode = str(emissive_blend_mode).strip().lower()
            if mode == "white":
                from PIL import ImageChops

                amt = float(weight) * float(emissive_brightness)
                if amt < 0.0:
                    amt = 0.0
                if amt > 1.0:
                    amt = 1.0

                if amt > 0.0:
                    mask = emissive_intensity
                    if amt != 1.0:
                        mask = mask.point(lambda p: int(p * amt))
                    mask_rgb = Image.merge("RGB", (mask, mask, mask))

                    white = Image.new("RGB", (w, h), (255, 255, 255))
                    delta = ImageChops.subtract(white, frame_rgb)
                    add = ImageChops.multiply(delta, mask_rgb)
                    frame_rgb = ImageChops.add(frame_rgb, add)
            else:
                emissive_rgb = _enhance_rgb(
                    frame_rgb,
                    brightness=1.0 + (weight * emissive_brightness),
                    contrast=1.0 + (weight * emissive_contrast),
                )
                frame_rgb = Image.composite(emissive_rgb, frame_rgb, emissive_intensity)

        if scroll_src is not None and bool(scroll_layer_enabled):
            dx_px = float(scroll_layer_dx_pct_per_frame) * float(w) * float(fi)
            dy_px = float(scroll_layer_dy_pct_per_frame) * float(h) * float(fi)
            tiled = _make_tiled_layer((w, h), scroll_src, offset_x=dx_px, offset_y=dy_px)
            frame_rgb = _blend_rgb_with_layer(
                base_rgb=frame_rgb,
                layer_rgba=tiled,
                opacity_0_1=float(scroll_layer_opacity),
                blend_mode=str(scroll_layer_blend_mode),
            )

        r, g, b = frame_rgb.split()
        frame = Image.merge("RGBA", (r, g, b, base_a))

        if zero_rgb_where_alpha0:
            mask0 = base_a.point(lambda a: 255 if a == 0 else 0)
            frame = Image.composite(Image.new("RGBA", (w, h), (0, 0, 0, 0)), frame, mask0)

        out_frames.append(frame)

    return out_frames


def make_vertical_spritesheet_loop(
    *,
    in_path: Path,
    out_path: Path,
    frames: int = 4,
    apex_frame: int = 3,
    overall_brightness: float = 1.0,
    overall_contrast: float = 1.0,
    ramp_brightness: float = 0.25,
    ramp_contrast: float = 0.0,
    emissive_mask_path: Optional[Path] = None,
    emissive_mask_scale: float = 2.0,
    emissive_brightness: float = 0.5,
    emissive_contrast: float = 0.0,
    emissive_blend_mode: str = "white",
    scroll_layer_path: Optional[Path] = None,
    scroll_layer_enabled: bool = False,
    scroll_layer_opacity: float = 1.0,
    scroll_layer_blend_mode: str = "normal",
    scroll_layer_scale: float = 1.0,
    scroll_layer_dx_pct_per_frame: float = 0.0,
    scroll_layer_dy_pct_per_frame: float = 0.0,
    zero_rgb_where_alpha0: bool = True,
) -> Path:
    if not in_path.exists():
        raise SystemExit(f"Input not found: {in_path}")

    frames = int(frames)
    if frames < 2:
        raise SystemExit("--frames must be >= 2")

    apex_frame = int(apex_frame)
    if apex_frame < 1 or apex_frame > frames:
        raise SystemExit(f"--apex-frame must be in [1, {frames}]")

    overall_brightness = float(overall_brightness)
    overall_contrast = float(overall_contrast)
    ramp_brightness = float(ramp_brightness)
    ramp_contrast = float(ramp_contrast)
    emissive_mask_scale = float(emissive_mask_scale)
    emissive_brightness = float(emissive_brightness)
    emissive_contrast = float(emissive_contrast)
    emissive_blend_mode = str(emissive_blend_mode).strip().lower()

    scroll_layer_enabled = bool(scroll_layer_enabled)
    scroll_layer_opacity = float(scroll_layer_opacity)
    scroll_layer_blend_mode = str(scroll_layer_blend_mode).strip().lower()
    scroll_layer_scale = float(scroll_layer_scale)
    scroll_layer_dx_pct_per_frame = float(scroll_layer_dx_pct_per_frame)
    scroll_layer_dy_pct_per_frame = float(scroll_layer_dy_pct_per_frame)

    if overall_brightness <= 0.0:
        raise SystemExit("--overall-brightness must be > 0")
    if overall_contrast <= 0.0:
        raise SystemExit("--overall-contrast must be > 0")
    if ramp_brightness < 0.0:
        raise SystemExit("--ramp-brightness must be >= 0")
    if ramp_contrast < 0.0:
        raise SystemExit("--ramp-contrast must be >= 0")
    if emissive_brightness < 0.0:
        raise SystemExit("--emissive-brightness must be >= 0")
    if emissive_contrast < 0.0:
        raise SystemExit("--emissive-contrast must be >= 0")
    if emissive_mask_scale < 0.0:
        raise SystemExit("--emissive-mask-scale must be >= 0")
    if emissive_blend_mode not in {"enhance", "white"}:
        raise SystemExit("--emissive-blend-mode must be 'enhance' or 'white'")
    if scroll_layer_opacity < 0.0 or scroll_layer_opacity > 1.0:
        raise SystemExit("--scroll-layer-opacity must be in [0, 1]")
    if scroll_layer_scale <= 0.0:
        raise SystemExit("--scroll-layer-scale must be > 0")
    if scroll_layer_blend_mode not in {"normal", "screen", "overlay", "lighten", "add"}:
        raise SystemExit("--scroll-layer-blend-mode must be one of: normal, screen, overlay, lighten, add")

    try:
        from PIL import Image
    except ModuleNotFoundError:
        raise SystemExit("Missing dependency: Pillow. Install with: python -m pip install Pillow")

    base = Image.open(in_path).convert("RGBA")
    w, h = base.size

    emissive_intensity = None
    if emissive_mask_path is not None:
        if not emissive_mask_path.exists():
            raise SystemExit(f"Emissive mask not found: {emissive_mask_path}")
        emissive_intensity = _load_emissive_intensity(emissive_mask_path, size=(w, h), scale=emissive_mask_scale)

    scroll_layer_rgba = None
    if scroll_layer_path is not None and scroll_layer_enabled:
        if not scroll_layer_path.exists():
            raise SystemExit(f"Scroll layer not found: {scroll_layer_path}")
        scroll_layer_rgba = Image.open(scroll_layer_path).convert("RGBA")

    sheet = Image.new("RGBA", (w, h * frames), (0, 0, 0, 0))

    rendered_frames = _generate_frames(
        base_rgba=base,
        frames=frames,
        apex_frame=apex_frame,
        overall_brightness=overall_brightness,
        overall_contrast=overall_contrast,
        ramp_brightness=ramp_brightness,
        ramp_contrast=ramp_contrast,
        emissive_intensity=emissive_intensity,
        emissive_brightness=emissive_brightness,
        emissive_contrast=emissive_contrast,
        emissive_blend_mode=emissive_blend_mode,
        scroll_layer_rgba=scroll_layer_rgba,
        scroll_layer_enabled=scroll_layer_enabled,
        scroll_layer_opacity=scroll_layer_opacity,
        scroll_layer_blend_mode=scroll_layer_blend_mode,
        scroll_layer_scale=scroll_layer_scale,
        scroll_layer_dx_pct_per_frame=scroll_layer_dx_pct_per_frame,
        scroll_layer_dy_pct_per_frame=scroll_layer_dy_pct_per_frame,
        zero_rgb_where_alpha0=zero_rgb_where_alpha0,
    )

    for fi, frame in enumerate(rendered_frames):
        sheet.paste(frame, (0, fi * h))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path, format="PNG")
    return out_path


def run_gui() -> None:
    try:
        from PySide6 import QtCore, QtGui, QtWidgets
    except ModuleNotFoundError:
        raise SystemExit("Missing dependency: PySide6. Install with: python -m pip install PySide6")

    try:
        from PIL import Image
        from PIL.ImageQt import ImageQt
    except ModuleNotFoundError:
        raise SystemExit("Missing dependency: Pillow. Install with: python -m pip install Pillow")

    def _apply_dark_theme(app: "QtWidgets.QApplication") -> None:
        app.setStyle("Fusion")
        app.setStyleSheet(
            "QWidget{background:#0b0b0d;color:#e8e8ea;}"
            "QGroupBox{border:1px solid #1f2024;margin-top:6px;padding:6px;}"
            "QGroupBox::title{subcontrol-origin:margin;left:8px;padding:0 4px 0 4px;color:#cfcfd4;}"
            "QLineEdit,QSpinBox,QDoubleSpinBox{background:#000000;color:#ffffff;border:1px solid #2a2b30;border-radius:4px;padding:3px;}"
            "QPushButton{background:#15161a;border:1px solid #2a2b30;border-radius:6px;padding:4px 8px;}"
            "QPushButton:hover{background:#1b1c21;}"
            "QPushButton:pressed{background:#0f1013;}"
            "QLabel{color:#e8e8ea;}"
        )

    class DropImageLabel(QtWidgets.QLabel):
        def __init__(self, title: str, on_drop, parent=None):
            super().__init__(parent)
            self._on_drop = on_drop
            self.setAcceptDrops(True)
            self.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            self.setMinimumSize(220, 220)
            self.setFrameShape(QtWidgets.QFrame.Shape.StyledPanel)
            self.setText(title)

        def dragEnterEvent(self, event: "QtGui.QDragEnterEvent") -> None:
            if event.mimeData().hasUrls():
                event.acceptProposedAction()
                return
            event.ignore()

        def dropEvent(self, event: "QtGui.QDropEvent") -> None:
            urls = event.mimeData().urls()
            if not urls:
                return
            p = urls[0].toLocalFile()
            if not p:
                return
            if not p.lower().endswith(".png"):
                return
            self._on_drop(p)

    class MainWindow(QtWidgets.QMainWindow):
        def __init__(self) -> None:
            super().__init__()
            self.setWindowTitle("Vertical Spritesheet Loop")
            self._input_path: Optional[Path] = None
            self._mask_path: Optional[Path] = None
            self._scroll_layer_path: Optional[Path] = None
            self._input_img: Optional[Image.Image] = None
            self._mask_img: Optional[Image.Image] = None
            self._scroll_layer_img: Optional[Image.Image] = None
            self._preview_frames = []
            self._preview_index = 0

            self._preview_rebuild_timer = QtCore.QTimer(self)
            self._preview_rebuild_timer.setSingleShot(True)
            self._preview_rebuild_timer.timeout.connect(self._rebuild_preview_frames)

            self._anim_timer = QtCore.QTimer(self)
            self._anim_timer.timeout.connect(self._advance_preview)

            central = QtWidgets.QWidget(self)
            self.setCentralWidget(central)
            layout = QtWidgets.QGridLayout(central)
            layout.setColumnStretch(0, 0)
            layout.setColumnStretch(1, 1)
            layout.setColumnStretch(2, 0)

            self._lbl_status = QtWidgets.QLabel("Ready", central)
            self._lbl_status.setAlignment(QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignVCenter)
            layout.addWidget(self._lbl_status, 0, 0, 1, 3)

            core = QtWidgets.QWidget(central)
            core_layout = QtWidgets.QVBoxLayout(core)
            core_layout.setContentsMargins(0, 0, 0, 0)
            core_layout.setSpacing(6)

            output_box = QtWidgets.QGroupBox("Output", core)
            output_form = QtWidgets.QFormLayout(output_box)
            output_form.setLabelAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
            output_form.setVerticalSpacing(4)

            self._txt_out = QtWidgets.QLineEdit(output_box)
            btn_out = QtWidgets.QPushButton("Browse...", output_box)
            btn_out.clicked.connect(self._browse_out)
            row_out = QtWidgets.QHBoxLayout()
            row_out.addWidget(self._txt_out, 1)
            row_out.addWidget(btn_out)
            output_form.addRow("Output PNG", row_out)
            core_layout.addWidget(output_box)

            anim_box = QtWidgets.QGroupBox("Animation", core)
            anim_form = QtWidgets.QFormLayout(anim_box)
            anim_form.setLabelAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
            anim_form.setVerticalSpacing(4)

            self._sp_frames = QtWidgets.QSpinBox(anim_box)
            self._sp_frames.setRange(2, 256)
            self._sp_frames.setValue(4)
            self._sp_frames.valueChanged.connect(self._schedule_preview_rebuild)
            anim_form.addRow("Frames", self._sp_frames)

            self._sp_apex = QtWidgets.QSpinBox(anim_box)
            self._sp_apex.setRange(1, 256)
            self._sp_apex.setValue(3)
            self._sp_apex.valueChanged.connect(self._schedule_preview_rebuild)
            anim_form.addRow("Apex frame", self._sp_apex)

            self._sp_overall_b = QtWidgets.QDoubleSpinBox(anim_box)
            self._sp_overall_b.setRange(0.01, 100.0)
            self._sp_overall_b.setDecimals(3)
            self._sp_overall_b.setSingleStep(0.05)
            self._sp_overall_b.setValue(1.0)
            self._sp_overall_b.valueChanged.connect(self._schedule_preview_rebuild)
            anim_form.addRow("Overall brightness", self._sp_overall_b)

            self._sp_overall_c = QtWidgets.QDoubleSpinBox(anim_box)
            self._sp_overall_c.setRange(0.01, 100.0)
            self._sp_overall_c.setDecimals(3)
            self._sp_overall_c.setSingleStep(0.05)
            self._sp_overall_c.setValue(1.0)
            self._sp_overall_c.valueChanged.connect(self._schedule_preview_rebuild)
            anim_form.addRow("Overall contrast", self._sp_overall_c)

            self._sp_ramp_b = QtWidgets.QDoubleSpinBox(anim_box)
            self._sp_ramp_b.setRange(0.0, 10.0)
            self._sp_ramp_b.setDecimals(3)
            self._sp_ramp_b.setSingleStep(0.05)
            self._sp_ramp_b.setValue(0.25)
            self._sp_ramp_b.valueChanged.connect(self._schedule_preview_rebuild)
            anim_form.addRow("Ramp brightness", self._sp_ramp_b)

            self._sp_ramp_c = QtWidgets.QDoubleSpinBox(anim_box)
            self._sp_ramp_c.setRange(0.0, 10.0)
            self._sp_ramp_c.setDecimals(3)
            self._sp_ramp_c.setSingleStep(0.05)
            self._sp_ramp_c.setValue(0.0)
            self._sp_ramp_c.valueChanged.connect(self._schedule_preview_rebuild)
            anim_form.addRow("Ramp contrast", self._sp_ramp_c)

            core_layout.addWidget(anim_box)

            opts_box = QtWidgets.QGroupBox("Options", core)
            opts_layout = QtWidgets.QVBoxLayout(opts_box)
            self._chk_zero_rgb = QtWidgets.QCheckBox("Zero RGB where alpha == 0", opts_box)
            self._chk_zero_rgb.setChecked(True)
            self._chk_zero_rgb.stateChanged.connect(self._schedule_preview_rebuild)
            opts_layout.addWidget(self._chk_zero_rgb)
            core_layout.addWidget(opts_box)

            preview_box = QtWidgets.QGroupBox("Preview", core)
            preview_form = QtWidgets.QFormLayout(preview_box)
            preview_form.setLabelAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
            preview_form.setVerticalSpacing(4)

            self._sp_frametime = QtWidgets.QSpinBox(preview_box)
            self._sp_frametime.setRange(1, 100)
            self._sp_frametime.setValue(10)
            self._sp_frametime.valueChanged.connect(self._update_anim_timer)
            preview_form.addRow("Frametime (ticks)", self._sp_frametime)

            self._btn_play = QtWidgets.QPushButton("Play", preview_box)
            self._btn_play.clicked.connect(self._toggle_play)
            preview_form.addRow("Animation", self._btn_play)
            core_layout.addWidget(preview_box)

            settings_row = QtWidgets.QHBoxLayout()
            self._btn_load_settings = QtWidgets.QPushButton("Load settings", core)
            self._btn_load_settings.clicked.connect(self._on_load_settings)
            self._btn_save_settings = QtWidgets.QPushButton("Save settings", core)
            self._btn_save_settings.clicked.connect(self._on_save_settings)
            settings_row.addWidget(self._btn_load_settings)
            settings_row.addWidget(self._btn_save_settings)
            core_layout.addLayout(settings_row)

            btn_row = QtWidgets.QHBoxLayout()
            self._btn_generate = QtWidgets.QPushButton("Generate", core)
            self._btn_generate.clicked.connect(self._on_generate)
            self._btn_update_preview = QtWidgets.QPushButton("Update preview", core)
            self._btn_update_preview.clicked.connect(self._rebuild_preview_frames)
            btn_close = QtWidgets.QPushButton("Close", core)
            btn_close.clicked.connect(self.close)
            btn_row.addWidget(self._btn_generate)
            btn_row.addWidget(self._btn_update_preview)
            btn_row.addWidget(btn_close)
            core_layout.addLayout(btn_row)
            core_layout.addStretch(1)

            center = QtWidgets.QWidget(central)
            center_layout = QtWidgets.QVBoxLayout(center)
            center_layout.setContentsMargins(0, 0, 0, 0)
            center_layout.setSpacing(6)

            self._lbl_anim = QtWidgets.QLabel("Animation preview", center)
            self._lbl_anim.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            self._lbl_anim.setMinimumHeight(520)
            self._lbl_anim.setFrameShape(QtWidgets.QFrame.Shape.StyledPanel)
            center_layout.addWidget(self._lbl_anim, 1)

            layers = QtWidgets.QWidget(central)
            layers_layout = QtWidgets.QVBoxLayout(layers)
            layers_layout.setContentsMargins(0, 0, 0, 0)
            layers_layout.setSpacing(6)

            input_box = QtWidgets.QGroupBox("Input", layers)
            input_grid = QtWidgets.QGridLayout(input_box)
            input_grid.setContentsMargins(6, 6, 6, 6)
            input_grid.setHorizontalSpacing(10)
            input_grid.setVerticalSpacing(6)

            self._lbl_drop_input = DropImageLabel("Drop Input PNG", self._set_input_path, input_box)
            self._lbl_drop_input.setFixedSize(140, 140)
            input_grid.addWidget(self._lbl_drop_input, 0, 0, 2, 1)
            input_grid.addWidget(QtWidgets.QLabel("Input PNG", input_box), 0, 1)
            self._txt_input = QtWidgets.QLineEdit(input_box)
            btn_input = QtWidgets.QPushButton("Browse...", input_box)
            btn_input.clicked.connect(self._browse_input)
            row_input = QtWidgets.QHBoxLayout()
            row_input.addWidget(self._txt_input, 1)
            row_input.addWidget(btn_input)
            input_grid.addLayout(row_input, 1, 1)
            layers_layout.addWidget(input_box)

            emissive_box = QtWidgets.QGroupBox("Emissive Mask", layers)
            emissive_grid = QtWidgets.QGridLayout(emissive_box)
            emissive_grid.setContentsMargins(6, 6, 6, 6)
            emissive_grid.setHorizontalSpacing(10)
            emissive_grid.setVerticalSpacing(6)

            self._lbl_drop_mask = DropImageLabel("Drop Emissive Mask PNG", self._set_mask_path, emissive_box)
            self._lbl_drop_mask.setFixedSize(140, 140)
            emissive_grid.addWidget(self._lbl_drop_mask, 0, 0, 2, 1)

            emissive_right = QtWidgets.QWidget(emissive_box)
            emissive_form = QtWidgets.QFormLayout(emissive_right)
            emissive_form.setLabelAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
            emissive_form.setVerticalSpacing(4)

            self._txt_mask = QtWidgets.QLineEdit(emissive_right)
            btn_mask = QtWidgets.QPushButton("Browse...", emissive_right)
            btn_mask.clicked.connect(self._browse_mask)
            row_mask = QtWidgets.QHBoxLayout()
            row_mask.addWidget(self._txt_mask, 1)
            row_mask.addWidget(btn_mask)
            emissive_form.addRow("Mask PNG", row_mask)

            self._sp_mask_scale = QtWidgets.QDoubleSpinBox(emissive_right)
            self._sp_mask_scale.setRange(0.0, 25.0)
            self._sp_mask_scale.setDecimals(3)
            self._sp_mask_scale.setSingleStep(0.25)
            self._sp_mask_scale.setValue(2.0)
            self._sp_mask_scale.valueChanged.connect(self._schedule_preview_rebuild)
            emissive_form.addRow("Mask scale", self._sp_mask_scale)

            self._sp_em_b = QtWidgets.QDoubleSpinBox(emissive_right)
            self._sp_em_b.setRange(0.0, 10.0)
            self._sp_em_b.setDecimals(3)
            self._sp_em_b.setSingleStep(0.05)
            self._sp_em_b.setValue(0.5)
            self._sp_em_b.valueChanged.connect(self._schedule_preview_rebuild)
            emissive_form.addRow("Emissive brightness", self._sp_em_b)

            self._sp_em_c = QtWidgets.QDoubleSpinBox(emissive_right)
            self._sp_em_c.setRange(0.0, 10.0)
            self._sp_em_c.setDecimals(3)
            self._sp_em_c.setSingleStep(0.05)
            self._sp_em_c.setValue(0.0)
            self._sp_em_c.valueChanged.connect(self._schedule_preview_rebuild)
            emissive_form.addRow("Emissive contrast", self._sp_em_c)

            self._chk_em_white = QtWidgets.QCheckBox("Blend emissive to white (screen)", emissive_right)
            self._chk_em_white.setChecked(True)
            self._chk_em_white.stateChanged.connect(self._schedule_preview_rebuild)
            emissive_form.addRow("Blend mode", self._chk_em_white)

            emissive_grid.addWidget(emissive_right, 0, 1, 2, 1)
            layers_layout.addWidget(emissive_box)

            scroll_box = QtWidgets.QGroupBox("Scroll Layer", layers)
            scroll_grid = QtWidgets.QGridLayout(scroll_box)
            scroll_grid.setContentsMargins(6, 6, 6, 6)
            scroll_grid.setHorizontalSpacing(10)
            scroll_grid.setVerticalSpacing(6)

            self._lbl_drop_scroll = DropImageLabel("Drop Scroll Layer PNG", self._set_scroll_layer_path, scroll_box)
            self._lbl_drop_scroll.setFixedSize(140, 140)
            scroll_grid.addWidget(self._lbl_drop_scroll, 0, 0, 2, 1)

            scroll_right = QtWidgets.QWidget(scroll_box)
            scroll_form = QtWidgets.QFormLayout(scroll_right)
            scroll_form.setLabelAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
            scroll_form.setVerticalSpacing(4)

            self._txt_scroll = QtWidgets.QLineEdit(scroll_right)
            btn_scroll = QtWidgets.QPushButton("Browse...", scroll_right)
            btn_scroll.clicked.connect(self._browse_scroll)
            row_scroll = QtWidgets.QHBoxLayout()
            row_scroll.addWidget(self._txt_scroll, 1)
            row_scroll.addWidget(btn_scroll)
            scroll_form.addRow("Scroll PNG", row_scroll)

            self._chk_scroll_enabled = QtWidgets.QCheckBox("Enabled", scroll_box)
            self._chk_scroll_enabled.setChecked(False)
            self._chk_scroll_enabled.stateChanged.connect(self._schedule_preview_rebuild)
            scroll_form.addRow("Scroll layer", self._chk_scroll_enabled)

            self._sp_scroll_opacity = QtWidgets.QDoubleSpinBox(scroll_box)
            self._sp_scroll_opacity.setRange(0.0, 1.0)
            self._sp_scroll_opacity.setDecimals(3)
            self._sp_scroll_opacity.setSingleStep(0.05)
            self._sp_scroll_opacity.setValue(1.0)
            self._sp_scroll_opacity.valueChanged.connect(self._schedule_preview_rebuild)
            scroll_form.addRow("Opacity", self._sp_scroll_opacity)

            self._cmb_scroll_blend = QtWidgets.QComboBox(scroll_box)
            self._cmb_scroll_blend.addItems(["normal", "screen", "overlay", "lighten", "add"])
            self._cmb_scroll_blend.setCurrentText("normal")
            self._cmb_scroll_blend.currentTextChanged.connect(lambda _t: self._schedule_preview_rebuild())
            scroll_form.addRow("Blend", self._cmb_scroll_blend)

            self._sp_scroll_scale = QtWidgets.QDoubleSpinBox(scroll_box)
            self._sp_scroll_scale.setRange(0.01, 100.0)
            self._sp_scroll_scale.setDecimals(3)
            self._sp_scroll_scale.setSingleStep(0.1)
            self._sp_scroll_scale.setValue(1.0)
            self._sp_scroll_scale.valueChanged.connect(self._on_scroll_scale_changed)
            scroll_form.addRow("Image scale", self._sp_scroll_scale)

            self._sp_scroll_dx = QtWidgets.QDoubleSpinBox(scroll_box)
            self._sp_scroll_dx.setRange(-10.0, 10.0)
            self._sp_scroll_dx.setDecimals(4)
            self._sp_scroll_dx.setSingleStep(0.01)
            self._sp_scroll_dx.setValue(0.0)
            self._sp_scroll_dx.valueChanged.connect(self._schedule_preview_rebuild)
            scroll_form.addRow("DX %/frame", self._sp_scroll_dx)

            self._sp_scroll_dy = QtWidgets.QDoubleSpinBox(scroll_box)
            self._sp_scroll_dy.setRange(-10.0, 10.0)
            self._sp_scroll_dy.setDecimals(4)
            self._sp_scroll_dy.setSingleStep(0.01)
            self._sp_scroll_dy.setValue(0.0)
            self._sp_scroll_dy.valueChanged.connect(self._schedule_preview_rebuild)
            scroll_form.addRow("DY %/frame", self._sp_scroll_dy)

            scroll_grid.addWidget(scroll_right, 0, 1, 2, 1)
            layers_layout.addWidget(scroll_box)
            layers_layout.addStretch(1)

            layout.addWidget(core, 1, 0)
            layout.addWidget(center, 1, 1)
            layout.addWidget(layers, 1, 2)

            self.resize(1100, 650)
            self._update_anim_timer()

            self._load_last_settings()

        def _refresh_scroll_tile_preview(self) -> None:
            if not hasattr(self, "_lbl_drop_scroll"):
                return
            if self._scroll_layer_img is None:
                return
            if self._input_img is not None:
                scaled = _scale_rgba(self._scroll_layer_img, scale=float(self._sp_scroll_scale.value()))
                tiled = _make_tiled_layer(self._input_img.size, scaled, offset_x=0, offset_y=0)
                self._update_image_preview(self._lbl_drop_scroll, tiled)
            else:
                self._update_image_preview(self._lbl_drop_scroll, self._scroll_layer_img)

        def _on_scroll_scale_changed(self) -> None:
            self._refresh_scroll_tile_preview()
            self._schedule_preview_rebuild()

        def _schedule_preview_rebuild(self) -> None:
            self._preview_rebuild_timer.start(150)

        def _browse_input(self) -> None:
            p, _filter = QtWidgets.QFileDialog.getOpenFileName(self, "Select input PNG", str(Path.cwd()), "PNG (*.png)")
            if not p:
                return
            self._set_input_path(p)

        def _browse_mask(self) -> None:
            p, _filter = QtWidgets.QFileDialog.getOpenFileName(self, "Select emissive mask PNG", str(Path.cwd()), "PNG (*.png)")
            if not p:
                return
            self._set_mask_path(p)

        def _browse_scroll(self) -> None:
            p, _filter = QtWidgets.QFileDialog.getOpenFileName(self, "Select scroll layer PNG", str(Path.cwd()), "PNG (*.png)")
            if not p:
                return
            self._set_scroll_layer_path(p)

        def _browse_out(self) -> None:
            initial = self._txt_out.text().strip() or "spritesheet.png"
            p, _filter = QtWidgets.QFileDialog.getSaveFileName(self, "Save spritesheet PNG", initial, "PNG (*.png)")
            if not p:
                return
            self._txt_out.setText(p)

        def _on_save_settings(self) -> None:
            suggested = "spritesheet_settings.json"
            out_txt = self._txt_out.text().strip()
            if out_txt:
                try:
                    outp = Path(out_txt)
                    suggested = str(outp.with_name(outp.stem + "_settings.json"))
                except Exception:
                    suggested = "spritesheet_settings.json"

            p, _filter = QtWidgets.QFileDialog.getSaveFileName(self, "Save settings JSON", suggested, "JSON (*.json)")
            if not p:
                return

            try:
                _write_json(Path(p), self._settings_dict())
                self._lbl_status.setText(f"Saved settings: {p}")
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, "Save failed", str(e))

        def _on_load_settings(self) -> None:
            p, _filter = QtWidgets.QFileDialog.getOpenFileName(self, "Load settings JSON", str(Path.cwd()), "JSON (*.json)")
            if not p:
                return
            try:
                s = json.loads(Path(p).read_text(encoding="utf-8"))
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, "Load failed", str(e))
                return

            self._apply_settings_dict(s)
            self._lbl_status.setText(f"Loaded settings: {p}")

        def _settings_dict(self) -> dict:
            return {
                "input_path": str(self._input_path) if self._input_path is not None else "",
                "output_path": self._txt_out.text().strip(),
                "emissive_mask_path": str(self._mask_path) if self._mask_path is not None else "",
                "scroll_layer_path": str(self._scroll_layer_path) if self._scroll_layer_path is not None else "",
                "scroll_layer_enabled": bool(self._chk_scroll_enabled.isChecked()) if hasattr(self, "_chk_scroll_enabled") else False,
                "scroll_layer_opacity": float(self._sp_scroll_opacity.value()) if hasattr(self, "_sp_scroll_opacity") else 1.0,
                "scroll_layer_blend_mode": str(self._cmb_scroll_blend.currentText()) if hasattr(self, "_cmb_scroll_blend") else "normal",
                "scroll_layer_scale": float(self._sp_scroll_scale.value()) if hasattr(self, "_sp_scroll_scale") else 1.0,
                "scroll_layer_dx_pct_per_frame": float(self._sp_scroll_dx.value()) if hasattr(self, "_sp_scroll_dx") else 0.0,
                "scroll_layer_dy_pct_per_frame": float(self._sp_scroll_dy.value()) if hasattr(self, "_sp_scroll_dy") else 0.0,
                "frames": int(self._sp_frames.value()),
                "apex_frame": int(self._sp_apex.value()),
                "overall_brightness": float(self._sp_overall_b.value()),
                "overall_contrast": float(self._sp_overall_c.value()),
                "ramp_brightness": float(self._sp_ramp_b.value()),
                "ramp_contrast": float(self._sp_ramp_c.value()),
                "emissive_mask_scale": float(self._sp_mask_scale.value()),
                "emissive_brightness": float(self._sp_em_b.value()),
                "emissive_contrast": float(self._sp_em_c.value()),
                "emissive_blend_mode": ("white" if self._chk_em_white.isChecked() else "enhance"),
                "zero_rgb_where_alpha0": bool(self._chk_zero_rgb.isChecked()),
                "preview_frametime_ticks": int(self._sp_frametime.value()),
            }

        def _apply_settings_dict(self, s: dict) -> None:
            if not isinstance(s, dict):
                return

            def _get_str(key: str) -> str:
                v = s.get(key, "")
                return str(v) if v is not None else ""

            def _get_int(key: str, default: int) -> int:
                try:
                    return int(s.get(key, default))
                except Exception:
                    return int(default)

            def _get_float(key: str, default: float) -> float:
                try:
                    return float(s.get(key, default))
                except Exception:
                    return float(default)

            in_path = _get_str("input_path")
            out_path = _get_str("output_path")
            mask_path = _get_str("emissive_mask_path")
            scroll_path = _get_str("scroll_layer_path")

            if out_path:
                self._txt_out.setText(out_path)

            self._sp_frames.setValue(max(2, _get_int("frames", self._sp_frames.value())))
            self._sp_apex.setValue(max(1, _get_int("apex_frame", self._sp_apex.value())))
            self._sp_overall_b.setValue(max(0.01, _get_float("overall_brightness", self._sp_overall_b.value())))
            self._sp_overall_c.setValue(max(0.01, _get_float("overall_contrast", self._sp_overall_c.value())))
            self._sp_ramp_b.setValue(max(0.0, _get_float("ramp_brightness", self._sp_ramp_b.value())))
            self._sp_ramp_c.setValue(max(0.0, _get_float("ramp_contrast", self._sp_ramp_c.value())))
            self._sp_mask_scale.setValue(max(0.0, _get_float("emissive_mask_scale", self._sp_mask_scale.value())))
            self._sp_em_b.setValue(max(0.0, _get_float("emissive_brightness", self._sp_em_b.value())))
            self._sp_em_c.setValue(max(0.0, _get_float("emissive_contrast", self._sp_em_c.value())))

            blend = _get_str("emissive_blend_mode").strip().lower()
            self._chk_em_white.setChecked(blend != "enhance")

            self._chk_zero_rgb.setChecked(bool(s.get("zero_rgb_where_alpha0", self._chk_zero_rgb.isChecked())))

            ft = _get_int("preview_frametime_ticks", self._sp_frametime.value())
            self._sp_frametime.setValue(max(1, ft))

            self._chk_scroll_enabled.setChecked(bool(s.get("scroll_layer_enabled", self._chk_scroll_enabled.isChecked())))
            self._sp_scroll_opacity.setValue(max(0.0, min(1.0, _get_float("scroll_layer_opacity", self._sp_scroll_opacity.value()))))
            self._cmb_scroll_blend.setCurrentText(_get_str("scroll_layer_blend_mode") or self._cmb_scroll_blend.currentText())
            self._sp_scroll_scale.setValue(max(0.01, _get_float("scroll_layer_scale", self._sp_scroll_scale.value())))
            self._sp_scroll_dx.setValue(_get_float("scroll_layer_dx_pct_per_frame", self._sp_scroll_dx.value()))
            self._sp_scroll_dy.setValue(_get_float("scroll_layer_dy_pct_per_frame", self._sp_scroll_dy.value()))

            if in_path and Path(in_path).exists():
                self._set_input_path(in_path)
            if mask_path and Path(mask_path).exists():
                self._set_mask_path(mask_path)
            if scroll_path and Path(scroll_path).exists():
                self._set_scroll_layer_path(scroll_path)

            self._schedule_preview_rebuild()

        def _load_last_settings(self) -> None:
            p = _script_settings_path()
            if not p.exists():
                return
            try:
                s = json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                return
            self._apply_settings_dict(s)

        def _set_input_path(self, p: str) -> None:
            self._input_path = Path(p)
            self._txt_input.setText(str(self._input_path))
            if not self._txt_out.text().strip():
                self._txt_out.setText(str(self._input_path.with_name(self._input_path.stem + "_spritesheet.png")))
            try:
                self._input_img = Image.open(self._input_path).convert("RGBA")
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, "Load failed", str(e))
                self._input_img = None
                return
            self._update_image_preview(self._lbl_drop_input, self._input_img)
            self._refresh_scroll_tile_preview()
            self._schedule_preview_rebuild()

        def _set_mask_path(self, p: str) -> None:
            self._mask_path = Path(p)
            self._txt_mask.setText(str(self._mask_path))
            try:
                self._mask_img = Image.open(self._mask_path).convert("RGBA")
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, "Load failed", str(e))
                self._mask_img = None
                return
            self._update_image_preview(self._lbl_drop_mask, self._mask_img)
            self._schedule_preview_rebuild()

        def _set_scroll_layer_path(self, p: str) -> None:
            self._scroll_layer_path = Path(p)
            self._txt_scroll.setText(str(self._scroll_layer_path))
            try:
                self._scroll_layer_img = Image.open(self._scroll_layer_path).convert("RGBA")
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, "Load failed", str(e))
                self._scroll_layer_img = None
                return
            self._refresh_scroll_tile_preview()
            self._schedule_preview_rebuild()

        def _update_image_preview(self, label: "QtWidgets.QLabel", img: "Image.Image") -> None:
            qimg = ImageQt(img)
            pm = QtGui.QPixmap.fromImage(qimg)
            pm = pm.scaled(label.size(), QtCore.Qt.AspectRatioMode.KeepAspectRatio, QtCore.Qt.TransformationMode.SmoothTransformation)
            label.setPixmap(pm)

        def resizeEvent(self, event: "QtGui.QResizeEvent") -> None:
            super().resizeEvent(event)
            if self._input_img is not None:
                self._update_image_preview(self._lbl_drop_input, self._input_img)
            if self._mask_img is not None:
                self._update_image_preview(self._lbl_drop_mask, self._mask_img)
            if self._scroll_layer_img is not None:
                self._refresh_scroll_tile_preview()
            self._render_current_preview_frame()

        def _rebuild_preview_frames(self) -> None:
            if self._input_img is None:
                self._preview_frames = []
                self._lbl_anim.setText("Animation preview")
                return

            frames = int(self._sp_frames.value())
            apex = int(self._sp_apex.value())
            if apex < 1:
                apex = 1
            if apex > frames:
                apex = frames
                self._sp_apex.setValue(apex)

            emissive_intensity = None
            if self._mask_img is not None:
                if self._mask_img.size != self._input_img.size:
                    self._lbl_status.setText("Emissive mask size mismatch")
                else:
                    try:
                        if self._mask_path is not None:
                            emissive_intensity = _load_emissive_intensity(
                                self._mask_path,
                                size=self._input_img.size,
                                scale=float(self._sp_mask_scale.value()),
                            )
                    except Exception as e:
                        self._lbl_status.setText(str(e))
                        emissive_intensity = None

            scroll_layer_rgba = None
            if self._scroll_layer_img is not None:
                scroll_layer_rgba = self._scroll_layer_img

            try:
                self._preview_frames = _generate_frames(
                    base_rgba=self._input_img,
                    frames=frames,
                    apex_frame=apex,
                    overall_brightness=float(self._sp_overall_b.value()),
                    overall_contrast=float(self._sp_overall_c.value()),
                    ramp_brightness=float(self._sp_ramp_b.value()),
                    ramp_contrast=float(self._sp_ramp_c.value()),
                    emissive_intensity=emissive_intensity,
                    emissive_brightness=float(self._sp_em_b.value()),
                    emissive_contrast=float(self._sp_em_c.value()),
                    emissive_blend_mode=("white" if self._chk_em_white.isChecked() else "enhance"),
                    scroll_layer_rgba=scroll_layer_rgba,
                    scroll_layer_enabled=bool(self._chk_scroll_enabled.isChecked()),
                    scroll_layer_opacity=float(self._sp_scroll_opacity.value()),
                    scroll_layer_blend_mode=str(self._cmb_scroll_blend.currentText()),
                    scroll_layer_scale=float(self._sp_scroll_scale.value()),
                    scroll_layer_dx_pct_per_frame=float(self._sp_scroll_dx.value()),
                    scroll_layer_dy_pct_per_frame=float(self._sp_scroll_dy.value()),
                    zero_rgb_where_alpha0=bool(self._chk_zero_rgb.isChecked()),
                )
                self._preview_index = 0
                self._render_current_preview_frame()
                self._lbl_status.setText(f"Preview ready ({len(self._preview_frames)} frames)")
            except Exception as e:
                self._preview_frames = []
                self._lbl_anim.setText("Preview failed")
                self._lbl_status.setText(str(e))

        def _render_current_preview_frame(self) -> None:
            if not self._preview_frames:
                return
            img = self._preview_frames[self._preview_index]
            qimg = ImageQt(img)
            pm = QtGui.QPixmap.fromImage(qimg)
            pm = pm.scaled(self._lbl_anim.size(), QtCore.Qt.AspectRatioMode.KeepAspectRatio, QtCore.Qt.TransformationMode.SmoothTransformation)
            self._lbl_anim.setPixmap(pm)

        def _advance_preview(self) -> None:
            if not self._preview_frames:
                return
            self._preview_index = (self._preview_index + 1) % len(self._preview_frames)
            self._render_current_preview_frame()

        def _update_anim_timer(self) -> None:
            frametime = int(self._sp_frametime.value())
            interval_ms = max(1, frametime) * 50
            self._anim_timer.setInterval(interval_ms)

        def _toggle_play(self) -> None:
            if self._anim_timer.isActive():
                self._anim_timer.stop()
                self._btn_play.setText("Play")
            else:
                if not self._preview_frames:
                    self._rebuild_preview_frames()
                self._anim_timer.start()
                self._btn_play.setText("Pause")

        def _on_generate(self) -> None:
            if self._input_path is None:
                QtWidgets.QMessageBox.information(self, "Missing input", "Set an input PNG.")
                return

            out_txt = self._txt_out.text().strip()
            if not out_txt:
                out_txt = str(self._input_path.with_name(self._input_path.stem + "_spritesheet.png"))
                self._txt_out.setText(out_txt)

            out_path = Path(out_txt)

            mask_path = None
            if self._mask_path is not None and self._txt_mask.text().strip():
                mask_path = self._mask_path

            scroll_layer_path = None
            if self._scroll_layer_path is not None and self._txt_scroll.text().strip():
                scroll_layer_path = self._scroll_layer_path

            try:
                outp = make_vertical_spritesheet_loop(
                    in_path=self._input_path,
                    out_path=out_path,
                    frames=int(self._sp_frames.value()),
                    apex_frame=int(self._sp_apex.value()),
                    overall_brightness=float(self._sp_overall_b.value()),
                    overall_contrast=float(self._sp_overall_c.value()),
                    ramp_brightness=float(self._sp_ramp_b.value()),
                    ramp_contrast=float(self._sp_ramp_c.value()),
                    emissive_mask_path=mask_path,
                    emissive_mask_scale=float(self._sp_mask_scale.value()),
                    emissive_brightness=float(self._sp_em_b.value()),
                    emissive_contrast=float(self._sp_em_c.value()),
                    emissive_blend_mode=("white" if self._chk_em_white.isChecked() else "enhance"),
                    scroll_layer_path=scroll_layer_path,
                    scroll_layer_enabled=bool(self._chk_scroll_enabled.isChecked()),
                    scroll_layer_opacity=float(self._sp_scroll_opacity.value()),
                    scroll_layer_blend_mode=str(self._cmb_scroll_blend.currentText()),
                    scroll_layer_scale=float(self._sp_scroll_scale.value()),
                    scroll_layer_dx_pct_per_frame=float(self._sp_scroll_dx.value()),
                    scroll_layer_dy_pct_per_frame=float(self._sp_scroll_dy.value()),
                    zero_rgb_where_alpha0=bool(self._chk_zero_rgb.isChecked()),
                )
                _save_settings_files(out_png_path=Path(outp), settings=self._settings_dict())
                self._lbl_status.setText(f"Wrote: {outp}")
            except SystemExit as e:
                QtWidgets.QMessageBox.critical(self, "Generate failed", str(e))
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, "Generate failed", str(e))

    app = QtWidgets.QApplication(sys.argv)
    _apply_dark_theme(app)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate an animated vertical spritesheet by baking a brightness/contrast loop into stacked frames"
    )
    parser.add_argument("--gui", action="store_true", help="Launch the GUI")
    parser.add_argument("input", nargs="?", default=None, help="Input PNG (RGBA recommended)")
    parser.add_argument("--out", dest="out_path", default=None, help="Output spritesheet PNG path")

    parser.add_argument("--frames", type=int, default=4, help="Number of frames (default: 4)")
    parser.add_argument(
        "--apex-frame",
        type=int,
        default=3,
        help="1-based frame index for the brightest apex (default: 3)",
    )

    parser.add_argument("--overall-brightness", type=float, default=1.0)
    parser.add_argument("--overall-contrast", type=float, default=1.0)

    parser.add_argument(
        "--ramp-brightness",
        type=float,
        default=0.25,
        help="Additional brightness at apex (e.g. 0.25 -> 1.25x at apex)",
    )
    parser.add_argument(
        "--ramp-contrast",
        type=float,
        default=0.0,
        help="Additional contrast at apex (e.g. 0.2 -> 1.2x at apex)",
    )

    parser.add_argument(
        "--emissive-mask",
        dest="emissive_mask",
        default=None,
        help="Optional emissive mask PNG (same size as input). RGB+alpha becomes an intensity mask.",
    )
    parser.add_argument(
        "--emissive-mask-scale",
        type=float,
        default=2.0,
        help="Scale factor applied to emissive mask intensity (default: 2.0)",
    )
    parser.add_argument(
        "--emissive-brightness",
        type=float,
        default=0.5,
        help="Additional brightness applied through emissive mask at apex (default: 0.5 -> 1.5x)",
    )
    parser.add_argument(
        "--emissive-contrast",
        type=float,
        default=0.0,
        help="Additional contrast applied through emissive mask at apex (default: 0.0)",
    )
    parser.add_argument(
        "--emissive-blend-mode",
        choices=["white", "enhance"],
        default="white",
        help="How emissive affects RGB: 'white' blends toward white (less saturation), 'enhance' boosts brightness/contrast (default: white)",
    )

    parser.add_argument("--scroll-layer", default=None, help="Optional scrolling layer texture (tiled/wrapped)")
    parser.add_argument("--scroll-layer-enabled", action="store_true", help="Enable scroll layer")
    parser.add_argument("--scroll-layer-opacity", type=float, default=1.0, help="Scroll layer opacity in [0,1] (default: 1.0)")
    parser.add_argument(
        "--scroll-layer-blend-mode",
        choices=["normal", "screen", "overlay", "lighten", "add"],
        default="normal",
        help="Scroll layer blend mode (default: normal)",
    )
    parser.add_argument(
        "--scroll-layer-scale",
        type=float,
        default=1.0,
        help="Scale applied to scroll layer image before tiling (>1 bigger tiles, <1 smaller tiles) (default: 1.0)",
    )
    parser.add_argument(
        "--scroll-layer-dx-pct-per-frame",
        type=float,
        default=0.0,
        help="Horizontal scroll per frame as fraction of base width (e.g. 0.25) (default: 0.0)",
    )
    parser.add_argument(
        "--scroll-layer-dy-pct-per-frame",
        type=float,
        default=0.0,
        help="Vertical scroll per frame as fraction of base height (e.g. 0.25) (default: 0.0)",
    )

    parser.add_argument(
        "--no-zero-rgb-where-alpha0",
        dest="zero_rgb_where_alpha0",
        action="store_false",
        help="Do not force RGB=(0,0,0) where alpha==0",
    )
    parser.set_defaults(zero_rgb_where_alpha0=True)

    args = parser.parse_args()

    if bool(args.gui) or args.input is None:
        run_gui()
        return

    in_path = Path(args.input)

    emissive_mask_path = Path(args.emissive_mask) if args.emissive_mask else None
    scroll_layer_path = Path(args.scroll_layer) if args.scroll_layer else None
    if args.out_path is None:
        out_path = in_path.with_name(in_path.stem + "_spritesheet.png")
    else:
        out_path = Path(args.out_path)

    outp = make_vertical_spritesheet_loop(
        in_path=in_path,
        out_path=out_path,
        frames=int(args.frames),
        apex_frame=int(args.apex_frame),
        overall_brightness=float(args.overall_brightness),
        overall_contrast=float(args.overall_contrast),
        ramp_brightness=float(args.ramp_brightness),
        ramp_contrast=float(args.ramp_contrast),
        emissive_mask_path=emissive_mask_path,
        emissive_mask_scale=float(args.emissive_mask_scale),
        emissive_brightness=float(args.emissive_brightness),
        emissive_contrast=float(args.emissive_contrast),
        emissive_blend_mode=str(args.emissive_blend_mode),
        scroll_layer_path=scroll_layer_path,
        scroll_layer_enabled=bool(args.scroll_layer_enabled),
        scroll_layer_opacity=float(args.scroll_layer_opacity),
        scroll_layer_blend_mode=str(args.scroll_layer_blend_mode),
        scroll_layer_scale=float(args.scroll_layer_scale),
        scroll_layer_dx_pct_per_frame=float(args.scroll_layer_dx_pct_per_frame),
        scroll_layer_dy_pct_per_frame=float(args.scroll_layer_dy_pct_per_frame),
        zero_rgb_where_alpha0=bool(args.zero_rgb_where_alpha0),
    )
    _save_settings_files(
        out_png_path=Path(outp),
        settings={
            "input_path": str(in_path),
            "output_path": str(out_path),
            "emissive_mask_path": str(emissive_mask_path) if emissive_mask_path is not None else "",
            "scroll_layer_path": str(scroll_layer_path) if scroll_layer_path is not None else "",
            "scroll_layer_enabled": bool(args.scroll_layer_enabled),
            "scroll_layer_opacity": float(args.scroll_layer_opacity),
            "scroll_layer_blend_mode": str(args.scroll_layer_blend_mode),
            "scroll_layer_scale": float(args.scroll_layer_scale),
            "scroll_layer_dx_pct_per_frame": float(args.scroll_layer_dx_pct_per_frame),
            "scroll_layer_dy_pct_per_frame": float(args.scroll_layer_dy_pct_per_frame),
            "frames": int(args.frames),
            "apex_frame": int(args.apex_frame),
            "overall_brightness": float(args.overall_brightness),
            "overall_contrast": float(args.overall_contrast),
            "ramp_brightness": float(args.ramp_brightness),
            "ramp_contrast": float(args.ramp_contrast),
            "emissive_mask_scale": float(args.emissive_mask_scale),
            "emissive_brightness": float(args.emissive_brightness),
            "emissive_contrast": float(args.emissive_contrast),
            "emissive_blend_mode": str(args.emissive_blend_mode),
            "zero_rgb_where_alpha0": bool(args.zero_rgb_where_alpha0),
        },
    )
    print(f"Wrote: {outp}")


if __name__ == "__main__":
    main()
