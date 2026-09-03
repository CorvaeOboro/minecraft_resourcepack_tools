"""Blender-side Renderer (cached `.blend` -> PNGs)

Run this script via Blender (`bpy` required) to open a cached `.blend` and render
one PNG per camera to an output directory.

Rendering is intentionally non-destructive:
- The `.blend` is opened and rendered, but not saved.
- Uses Workbench flat textured rendering so output matches texture colors.
- Post-pass flattens alpha over black and sets output alpha to 1.0.

Typically called by:
- `mc_resourcepack_blender_dashboard.py`

TOOLSGROUP::RENDER
SORTGROUP::7
SORTPRIORITY::78
STATUS::active
VERSION::20260308

Dependencies
- Blender (run with `blender -b ... -P ...`)
- Python provided by Blender (`bpy`)

Expected inputs (passed after `--`)
- `--blend <path>` (optional if the `.blend` is passed as the Blender file argument)
- `--output_dir <path>`
- Optional:
  - `--output_prefix <str>`
  - `--ensure_cameras true|false`
  - `--cameras cam1,cam2,...`
  - `--json <path>` + `--resourcepack_root <path>` (infer cache/output paths)

Outputs
- Writes PNGs into `output_dir` (one per camera)

Usage
- Simple:
  blender -b path/to/model.blend -P DEV/mc_resourcepack_blender_render_blends.py -- --output_dir out

- Project example:
  blender -b RESOURCEPACK/00_OBORO_20260217/assets/minecraft/optifine/cem/iron_golem.blend \
    -P DEV/mc_resourcepack_blender_render_blends.py -- --output_dir renders/iron_golem
"""

import math
import sys
from array import array
from dataclasses import dataclass
from pathlib import Path


def _require_bpy():
    try:
        import bpy  # type: ignore

        return bpy
    except Exception as e:
        raise RuntimeError("This script must be run by Blender (bpy not available)") from e


@dataclass(frozen=True)
class ModelIdentity:
    source_path: Path
    resourcepack_root: Path
    namespace: str
    model_rel_dir: Path
    model_stem: str

    def cache_blend_path(self) -> Path:
        return self.source_path.with_suffix(".blend")

    def render_output_dir(self) -> Path:
        return (
            self.resourcepack_root
            / "assets"
            / self.namespace
            / "textures"
            / "renders"
            / self.model_rel_dir
            / self.model_stem
        )


def _find_resourcepack_root(model_json_path: Path) -> Path:
    p = model_json_path.resolve()
    for parent in [p.parent] + list(p.parents):
        if parent.name.lower() == "assets":
            return parent.parent
    raise ValueError(f"Could not locate resourcepack root for: {model_json_path} (expected .../assets/<ns>/models/...)")


def _identify_model(model_json_path: Path, resourcepack_root: Path | None = None) -> ModelIdentity:
    model_json_path = model_json_path.resolve()
    if resourcepack_root is None:
        resourcepack_root = _find_resourcepack_root(model_json_path)
    resourcepack_root = Path(resourcepack_root).resolve()

    parts = model_json_path.parts
    assets_idx = None
    for i, part in enumerate(parts):
        if part.lower() == "assets":
            assets_idx = i
            break
    if assets_idx is None:
        raise ValueError(f"Invalid model json path (no assets/): {model_json_path}")

    if assets_idx + 3 >= len(parts):
        raise ValueError(f"Invalid model json path: {model_json_path}")

    namespace = parts[assets_idx + 1]

    p2 = parts[assets_idx + 2].lower()
    p3 = parts[assets_idx + 3].lower() if (assets_idx + 3) < len(parts) else ""

    if p2 == "models" and model_json_path.suffix.lower() == ".json":
        rel_after_models = Path(*parts[assets_idx + 3 :])
        model_stem = rel_after_models.stem
        model_rel_dir = rel_after_models.parent
    elif p2 == "optifine" and p3 == "cem" and model_json_path.suffix.lower() == ".jem":
        rel_after_cem = Path(*parts[assets_idx + 4 :])
        model_stem = rel_after_cem.stem
        model_rel_dir = Path("optifine") / "cem" / rel_after_cem.parent
    else:
        raise ValueError(f"Unsupported model path: {model_json_path}")

    return ModelIdentity(source_path=model_json_path, resourcepack_root=resourcepack_root, namespace=str(namespace), model_rel_dir=model_rel_dir, model_stem=str(model_stem))


def _parse_args(argv: list[str]) -> dict | None:
    if "--" not in argv:
        return None

    args = argv[argv.index("--") + 1 :]
    if not args:
        return None

    out: dict[str, str] = {}
    key = None
    for a in args:
        if a.startswith("--"):
            key = a[2:].strip()
            out[key] = "true"
        else:
            if key is None:
                continue
            out[key] = a
            key = None

    return out


def _setup_render_settings(bpy, scene):
    scene.display_settings.display_device = "sRGB"
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.exposure = 0.0
    scene.view_settings.gamma = 1.0
    scene.sequencer_colorspace_settings.name = "sRGB"
    scene.render.engine = "BLENDER_WORKBENCH"
    try:
        sh = scene.display.shading
        sh.light = "FLAT"
        sh.color_type = "TEXTURE"
        if hasattr(sh, "show_shadows"):
            sh.show_shadows = False
        if hasattr(sh, "show_cavity"):
            sh.show_cavity = False
        if hasattr(sh, "show_specular_highlight"):
            sh.show_specular_highlight = False
        if hasattr(sh, "show_backface_culling"):
            sh.show_backface_culling = False
        if hasattr(sh, "background_type"):
            sh.background_type = "VIEWPORT"
        if hasattr(sh, "background_color"):
            sh.background_color = (0.0, 0.0, 0.0)
    except Exception:
        pass
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.image_settings.color_depth = "8"
    scene.render.film_transparent = True


def _force_unlit_textured_materials(bpy) -> None:
    for mat in bpy.data.materials:
        if mat is None or not getattr(mat, "use_nodes", False):
            continue
        nt = getattr(mat, "node_tree", None)
        if nt is None:
            continue

        nodes = nt.nodes
        links = nt.links

        out = nodes.get("Material Output")
        if out is None:
            continue

        tex = None
        for n in nodes:
            if n.type == "TEX_IMAGE" and getattr(n, "image", None) is not None:
                tex = n
                break

        if tex is None:
            continue

        emis = nodes.get("__MC_Emission")
        if emis is None:
            emis = nodes.new(type="ShaderNodeEmission")
            emis.name = "__MC_Emission"
            emis.label = "__MC_Emission"
            emis.location = (tex.location.x + 250, tex.location.y)

        trans = nodes.get("__MC_Transparent")
        if trans is None:
            trans = nodes.new(type="ShaderNodeBsdfTransparent")
            trans.name = "__MC_Transparent"
            trans.label = "__MC_Transparent"
            trans.location = (tex.location.x + 250, tex.location.y - 160)

        mix = nodes.get("__MC_Mix")
        if mix is None:
            mix = nodes.new(type="ShaderNodeMixShader")
            mix.name = "__MC_Mix"
            mix.label = "__MC_Mix"
            mix.location = (tex.location.x + 520, tex.location.y - 40)

        try:
            emis.inputs["Strength"].default_value = 1.0
        except Exception:
            pass

        for l in list(links):
            try:
                to_node = l.to_node
                to_socket = l.to_socket
            except ReferenceError:
                continue

            remove = False
            if to_node == out and to_socket == out.inputs.get("Surface"):
                remove = True
            if to_node == emis:
                remove = True
            if to_node == mix:
                remove = True

            if remove:
                try:
                    links.remove(l)
                except ReferenceError:
                    pass

        links.new(tex.outputs.get("Color"), emis.inputs.get("Color"))
        links.new(trans.outputs.get("BSDF"), mix.inputs[1])
        links.new(emis.outputs.get("Emission"), mix.inputs[2])

        alpha_out = tex.outputs.get("Alpha")
        if alpha_out is not None:
            links.new(alpha_out, mix.inputs.get("Fac"))
            mat.blend_method = "BLEND"
            mat.shadow_method = "NONE"
        else:
            try:
                mix.inputs.get("Fac").default_value = 1.0
            except Exception:
                pass
            mat.blend_method = "OPAQUE"
            mat.shadow_method = "NONE"

        links.new(mix.outputs.get("Shader"), out.inputs.get("Surface"))


def _ensure_default_cameras(bpy) -> None:
    if any(obj.type == "CAMERA" for obj in bpy.data.objects):
        return

    target = bpy.data.objects.get("MC_Target")
    if target is None:
        target = bpy.data.objects.new("MC_Target", None)
        target.empty_display_type = "PLAIN_AXES"
        target.empty_display_size = 0.2
        bpy.context.scene.collection.objects.link(target)
        target.location = (0.0, 0.0, 0.0)

    dist = 2.2
    ortho = 1.25
    cams = {
        "north": (0.0, dist, 0.0),
        "south": (0.0, -dist, 0.0),
        "east": (dist, 0.0, 0.0),
        "west": (-dist, 0.0, 0.0),
        "up": (0.0, 0.0, dist),
        "down": (0.0, 0.0, -dist),
    }

    for name, loc in cams.items():
        cam_data = bpy.data.cameras.new(name)
        cam_data.type = "ORTHO"
        cam_data.ortho_scale = ortho
        cam_obj = bpy.data.objects.new(name, cam_data)
        bpy.context.scene.collection.objects.link(cam_obj)
        cam_obj.location = loc
        cam_obj.rotation_mode = "XYZ"

        con = cam_obj.constraints.new(type="TRACK_TO")
        con.target = target
        con.track_axis = "TRACK_NEGATIVE_Z"
        con.up_axis = "UP_Y"


def _flatten_png_alpha_to_black(bpy, png_path: Path) -> None:
    p = Path(png_path)
    if not p.exists() or p.suffix.lower() != ".png":
        return

    img = None
    try:
        img = bpy.data.images.load(str(p), check_existing=False)
        w = int(getattr(img, "size", [0, 0])[0] or 0)
        h = int(getattr(img, "size", [0, 0])[1] or 0)
        if w <= 0 or h <= 0:
            return

        n = w * h * 4
        buf = array("f", [0.0]) * n
        img.pixels.foreach_get(buf)

        for i in range(0, n, 4):
            a = float(buf[i + 3])
            buf[i + 0] = float(buf[i + 0]) * a
            buf[i + 1] = float(buf[i + 1]) * a
            buf[i + 2] = float(buf[i + 2]) * a
            buf[i + 3] = 1.0

        img.pixels.foreach_set(buf)
        img.filepath_raw = str(p)
        img.file_format = "PNG"
        img.save()
    except Exception:
        return
    finally:
        try:
            if img is not None:
                bpy.data.images.remove(img)
        except Exception:
            pass


def _render_all_cameras(bpy, *, output_dir: Path, camera_names: list[str] | None) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)

    scene = bpy.context.scene
    _setup_render_settings(bpy, scene)

    cameras = [obj for obj in bpy.data.objects if obj.type == "CAMERA"]
    if camera_names:
        keep = set(camera_names)
        cameras = [cam for cam in cameras if cam.name in keep]

    output_prefix = ""
    try:
        output_prefix = str(bpy.context.scene.get("__mc_output_prefix") or "")
    except Exception:
        output_prefix = ""

    rendered = 0
    for cam in cameras:
        scene.camera = cam
        out_path = output_dir / f"{output_prefix}{cam.name}.png"
        scene.render.filepath = str(out_path)
        bpy.ops.render.render(write_still=True)
        _flatten_png_alpha_to_black(bpy, out_path)
        rendered += 1
    return rendered


def main() -> int:
    bpy = _require_bpy()

    parsed = _parse_args(sys.argv)
    if not parsed:
        return 0

    json_path_raw = parsed.get("json") or parsed.get("model") or ""
    rp_root = Path(parsed["resourcepack_root"]) if parsed.get("resourcepack_root") else None

    if json_path_raw:
        identity = _identify_model(Path(json_path_raw), rp_root)
        out_dir = Path(parsed["output_dir"]) if parsed.get("output_dir") else identity.render_output_dir()
        cache_path = identity.cache_blend_path()
    else:
        out_dir = Path(parsed["output_dir"]) if parsed.get("output_dir") else None
        cache_path_raw = parsed.get("blend") or parsed.get("cache") or ""
        cache_path = Path(cache_path_raw) if cache_path_raw else Path(bpy.data.filepath)

    if not cache_path.exists():
        print(f"Missing cache blend: {cache_path}")
        return 2

    if str(cache_path) and Path(bpy.data.filepath).resolve() != cache_path.resolve():
        bpy.ops.wm.open_mainfile(filepath=str(cache_path))

    ensure_cameras = str(parsed.get("ensure_cameras", "false")).lower() in {"1", "true", "yes", "y"}
    if ensure_cameras:
        _ensure_default_cameras(bpy)

    output_prefix = str(parsed.get("output_prefix") or "")
    try:
        bpy.context.scene["__mc_output_prefix"] = output_prefix
    except Exception:
        pass

    cameras_csv = str(parsed.get("cameras", "") or "")
    camera_names = [c.strip() for c in cameras_csv.split(",") if c.strip()] if cameras_csv else []

    if out_dir is None:
        print("Missing --output_dir (and no --json provided to infer output dir)")
        return 2

    rendered = _render_all_cameras(bpy, output_dir=out_dir, camera_names=camera_names or None)
    print(f"Rendered cameras: {rendered}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
