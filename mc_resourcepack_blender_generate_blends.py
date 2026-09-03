"""Blender-side `.blend` Generator (Minecraft JSON + Optifine CEM)

Run this script via Blender (`bpy` required) to import:
- Minecraft model JSON (`assets/<ns>/models/**/*.json`)
- Optifine CEM entity model JSON (`assets/<ns>/optifine/cem/**/*.jem`)

and save a cached `.blend` file next to the source model.

This script is intended to be run headlessly from the dashboard:
- `mc_resourcepack_blender_dashboard.py`

TOOLSGROUP::RENDER
SORTGROUP::7
SORTPRIORITY::77
STATUS::active
VERSION::20260308

Dependencies
- Blender (run with `blender -b -P ...`)
- Python provided by Blender (`bpy`)

Expected inputs (passed after `--`)
- `--json <path>`: source `.json` or `.jem`
- `--resourcepack_root <path>`: resourcepack root containing `assets/` (optional; inferred if omitted)
- `--overwrite true|false`: overwrite existing cached `.blend`
- `--ensure_cameras true|false`: create default orthographic cameras if none exist

Outputs
- Writes `<source_dir>/<source_stem>.blend` by default

Usage
- Simple (run in Blender):
  blender -b -P DEV/mc_resourcepack_blender_generate_blends.py -- --json path/to/model.json

- Project example (resourcepack CEM model):
  blender -b -P DEV/mc_resourcepack_blender_generate_blends.py -- \
    --json RESOURCEPACK/00_OBORO_20260217/assets/minecraft/optifine/cem/iron_golem.jem \
    --resourcepack_root RESOURCEPACK/00_OBORO_20260217 \
    --overwrite false --ensure_cameras true
"""

import json
import math
import os
import sys
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


def _load_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _model_id_to_json_path(identity: ModelIdentity, model_id: str) -> Path | None:
    if not model_id:
        return None

    mid = str(model_id)
    if ":" in mid:
        ns, sub = mid.split(":", 1)
    else:
        ns, sub = identity.namespace, mid

    sub = sub.lstrip("/")
    p = identity.resourcepack_root / "assets" / ns / "models" / f"{sub}.json"
    if p.exists():
        return p
    return None


def _resolve_model_inheritance(identity: ModelIdentity, model_json_path: Path) -> dict:
    seen: set[Path] = set()

    def load_chain(path: Path) -> list[dict]:
        chain: list[dict] = []
        cur_path = Path(path).resolve()
        for _ in range(32):
            if cur_path in seen:
                break
            seen.add(cur_path)

            data = _load_json(cur_path)
            chain.append(data)
            parent_id = data.get("parent")
            if not parent_id:
                break
            parent_path = _model_id_to_json_path(identity, str(parent_id))
            if parent_path is None:
                break
            cur_path = parent_path
        return chain

    chain = load_chain(model_json_path)
    chain_rev = list(reversed(chain))

    merged_textures: dict = {}
    merged_elements = None

    for m in chain_rev:
        tex = m.get("textures")
        if isinstance(tex, dict):
            merged_textures.update(tex)
        if "elements" in m and isinstance(m.get("elements"), list):
            merged_elements = m.get("elements")

    if merged_elements is None:
        merged_elements = []

    out = {
        "textures": merged_textures,
        "elements": merged_elements,
    }
    return out


def _resolve_texture_ref(textures: dict, ref: str) -> str:
    cur = str(ref)
    for _ in range(16):
        if not cur.startswith("#"):
            return cur
        k = cur[1:]
        nxt = textures.get(k)
        if not nxt:
            return cur
        cur = str(nxt)
    return cur


def _texture_id_to_png_path(identity: ModelIdentity, texture_id: str) -> Path | None:
    if not texture_id:
        return None

    tid = str(texture_id)
    if tid.startswith("#"):
        return None

    if ":" in tid:
        ns, sub = tid.split(":", 1)
    else:
        ns, sub = "minecraft", tid

    sub = sub.lstrip("/")
    if sub.startswith("textures/"):
        sub = sub[len("textures/") :]
    p = identity.resourcepack_root / "assets" / ns / "textures" / f"{sub}.png"
    if p.exists():
        return p
    return None


def _cem_texture_to_texture_id(identity: ModelIdentity, raw: str) -> str:
    s = str(raw or "").strip()
    if not s:
        return ""

    if s.endswith(".png"):
        s = s[: -len(".png")]

    s = s.replace("\\", "/")
    if ":" in s:
        return s

    s = s.lstrip("/")
    if s.startswith("textures/"):
        s = s[len("textures/") :]

    if "/" in s:
        return f"{identity.namespace}:{s}"

    # Heuristic: common entity pattern
    return f"{identity.namespace}:entity/{s}/{s}"


def _uv_rect_tex(u1: float, v1: float, u2: float, v2: float, tw: float, th: float) -> list[tuple[float, float]]:
    if tw <= 0.0:
        tw = 64.0
    if th <= 0.0:
        th = 32.0

    u1n = u1 / tw
    u2n = u2 / tw
    v1n = 1.0 - (v1 / th)
    v2n = 1.0 - (v2 / th)
    tl = (u1n, v1n)
    tr = (u2n, v1n)
    br = (u2n, v2n)
    bl = (u1n, v2n)
    return [bl, br, tr, tl]


def _cem_to_blender_xyz(mc: tuple[float, float, float]) -> tuple[float, float, float]:
    # Blockbench exports Optifine CEM in Minecraft-like model coords (x right, y up, z forward) in pixels.
    # Blender uses Z-up, so we swizzle to (x, z, y) and scale to blocks.
    x, y, z = mc
    bx = x / 16.0
    by = -(z / 16.0)
    bz = y / 16.0
    return (bx, by, bz)


def _cem_translate_to_blender_xyz(mc: tuple[float, float, float]) -> tuple[float, float, float]:
    return _cem_to_blender_xyz(mc)


def _cem_invert_flags(s: str | None) -> tuple[bool, bool, bool]:
    v = str(s or "").lower()
    return ("x" in v, "y" in v, "z" in v)


def _cem_invert_translate(t: list[float], inv: tuple[bool, bool, bool]) -> list[float]:
    ix, iy, iz = inv
    return [(-t[0] if ix else t[0]), (-t[1] if iy else t[1]), (-t[2] if iz else t[2])]


def _cem_invert_box_start_and_size(
    x: float, y: float, z: float, dx: float, dy: float, dz: float, inv: tuple[bool, bool, bool]
) -> tuple[float, float, float, float, float, float]:
    ix, iy, iz = inv
    if ix:
        x = -(x + dx)
    if iy:
        y = -(y + dy)
    if iz:
        z = -(z + dz)
    return x, y, z, dx, dy, dz


def _import_optifine_cem_jem_into_scene(bpy, *, jem_path: Path, identity: ModelIdentity, ensure_cameras: bool) -> None:
    def _bb_box_name(base: str, idx: int) -> str:
        b = str(base or "").strip() or "box"
        if idx <= 0:
            return b
        return f"{b}.{idx:03d}"

    data = _load_json(jem_path)

    tex_size = data.get("textureSize")
    tw, th = 64.0, 32.0
    if isinstance(tex_size, list) and len(tex_size) == 2:
        try:
            tw, th = float(tex_size[0]), float(tex_size[1])
        except Exception:
            tw, th = 64.0, 32.0

    texture_raw = data.get("texture")
    texture_id = _cem_texture_to_texture_id(identity, str(texture_raw or ""))
    mat = _ensure_material_for_texture(bpy, identity, texture_id) if texture_id else None

    models = data.get("models")
    if not isinstance(models, list):
        models = []

    root = bpy.data.objects.new("CEM_Root", None)
    bpy.context.scene.collection.objects.link(root)
    root.location = (0.0, 0.0, 0.0)
    root.rotation_mode = "XYZ"
    root.rotation_euler[2] = 0.0

    for mi, m in enumerate(models):
        if not isinstance(m, dict):
            continue

        part_name = str(m.get("part") or f"part_{mi:02d}")

        inv = _cem_invert_flags(m.get("invertAxis"))

        t = m.get("translate")
        if not (isinstance(t, list) and len(t) == 3):
            t = [0.0, 0.0, 0.0]

        t = [float(t[0]), float(t[1]), float(t[2])]

        r = m.get("rotate")
        if not (isinstance(r, list) and len(r) == 3):
            r = [0.0, 0.0, 0.0]

        r = [float(r[0]), float(r[1]), float(r[2])]

        t_use = [-float(t[0]), -float(t[1]), -float(t[2])]
        pivot = _cem_translate_to_blender_xyz((float(t_use[0]), float(t_use[1]), float(t_use[2])))

        empty = bpy.data.objects.new(f"CEM_{part_name}", None)
        bpy.context.scene.collection.objects.link(empty)
        empty.location = pivot
        empty.rotation_mode = "XYZ"
        empty.rotation_euler[0] = math.radians(float(r[0]))
        empty.rotation_euler[2] = math.radians(-float(r[1]))
        empty.rotation_euler[1] = math.radians(-float(r[2]))
        empty.parent = root

        def _import_boxes(*, parent, box_list, pivot_world, base_object_name: str, name_start_index: int = 0) -> None:
            if not isinstance(box_list, list):
                return

            for bi, b in enumerate(box_list):
                x = y = z = 0.0
                dx = dy = dz = 0.0
                u = v = 0.0
                size_add = 0.0

                if isinstance(b, list) and len(b) >= 8:
                    x, y, z, dx, dy, dz, u, v = [float(b[i]) for i in range(8)]
                elif isinstance(b, dict):
                    coords = b.get("coordinates")
                    uv = b.get("uv")
                    tex_off = b.get("textureOffset")
                    if isinstance(tex_off, list) and len(tex_off) >= 2:
                        u, v = float(tex_off[0]), float(tex_off[1])
                    if isinstance(coords, list) and len(coords) >= 6:
                        x, y, z, dx, dy, dz = [float(coords[i]) for i in range(6)]
                    if isinstance(uv, list) and len(uv) >= 2:
                        u, v = float(uv[0]), float(uv[1])
                    try:
                        size_add = float(b.get("sizeAdd") or 0.0)
                    except Exception:
                        size_add = 0.0
                else:
                    continue

                if dx < 0.0:
                    x = x + dx
                    dx = -dx
                if dy < 0.0:
                    y = y + dy
                    dy = -dy
                if dz < 0.0:
                    z = z + dz
                    dz = -dz

                base_dx, base_dy, base_dz = dx, dy, dz
                if size_add:
                    x -= size_add
                    y -= size_add
                    z -= size_add
                    dx += 2.0 * size_add
                    dy += 2.0 * size_add
                    dz += 2.0 * size_add

                x0, x1 = x, x + dx
                y0, y1 = y, y + dy
                z0, z1 = z, z + dz

                mx0, mx1 = min(x0, x1), max(x0, x1)
                my0, my1 = min(y0, y1), max(y0, y1)
                mz0, mz1 = min(z0, z1), max(z0, z1)

                mc_verts = [
                    (mx0, my0, mz0),
                    (mx1, my0, mz0),
                    (mx1, my1, mz0),
                    (mx0, my1, mz0),
                    (mx0, my0, mz1),
                    (mx1, my0, mz1),
                    (mx1, my1, mz1),
                    (mx0, my1, mz1),
                ]

                verts_world = [
                    _cem_to_blender_xyz((mx0, my0, mz0)),
                    _cem_to_blender_xyz((mx1, my0, mz0)),
                    _cem_to_blender_xyz((mx1, my1, mz0)),
                    _cem_to_blender_xyz((mx0, my1, mz0)),
                    _cem_to_blender_xyz((mx0, my0, mz1)),
                    _cem_to_blender_xyz((mx1, my0, mz1)),
                    _cem_to_blender_xyz((mx1, my1, mz1)),
                    _cem_to_blender_xyz((mx0, my1, mz1)),
                ]

                verts = [
                    (vw[0] - pivot_world[0], vw[1] - pivot_world[1], vw[2] - pivot_world[2])
                    for vw in verts_world
                ]

                faces = [
                    (0, 1, 2, 3),
                    (5, 4, 7, 6),
                    (0, 3, 7, 4),
                    (1, 5, 6, 2),
                    (1, 0, 4, 5),
                    (6, 7, 3, 2),
                ]

                mesh = bpy.data.meshes.new(f"CEM_mesh_{mi:02d}_{bi:03d}")
                mesh.from_pydata(verts, [], faces)
                mesh.update()

                uv_layer = mesh.uv_layers.new(name="UVMap")
                uv_data = uv_layer.data

                obj_name = _bb_box_name(base_object_name, name_start_index + bi)
                obj = bpy.data.objects.new(obj_name, mesh)
                bpy.context.scene.collection.objects.link(obj)
                obj.parent = parent

                if mat is not None:
                    mesh.materials.append(mat)

                du, dv = float(base_dx), float(base_dy)
                dw = float(base_dz)

                uv_west = (u, v + dw, u + dw, v + dw + dv)
                uv_north = (u + dw, v + dw, u + dw + du, v + dw + dv)
                uv_east = (u + dw + du, v + dw, u + dw + du + dw, v + dw + dv)
                uv_south = (u + dw + du + dw, v + dw, u + dw + du + dw + du, v + dw + dv)
                uv_down = (u + dw, v, u + dw + du, v + dw)
                uv_up = (u + dw + du, v, u + dw + du + du, v + dw)

                per_face = [
                    _uv_rect_tex(uv_north[0], uv_north[1], uv_north[2], uv_north[3], tw, th),
                    _uv_rect_tex(uv_south[0], uv_south[1], uv_south[2], uv_south[3], tw, th),
                    _uv_rect_tex(uv_east[0], uv_east[1], uv_east[2], uv_east[3], tw, th),
                    _uv_rect_tex(uv_west[0], uv_west[1], uv_west[2], uv_west[3], tw, th),
                    _uv_rect_tex(uv_up[0], uv_up[1], uv_up[2], uv_up[3], tw, th),
                    _uv_rect_tex(uv_down[0], uv_down[1], uv_down[2], uv_down[3], tw, th),
                ]

                poly_by_verts = {frozenset(p.vertices): p for p in mesh.polygons}
                for face_idx, face_verts in enumerate(faces):
                    poly = poly_by_verts.get(frozenset(face_verts))
                    if poly is None:
                        continue

                    uvs = per_face[face_idx] if face_idx < len(per_face) else per_face[0]
                    v_to_loop = {vi: li for vi, li in zip(poly.vertices, poly.loop_indices)}

                    face_corner_perm = [
                        (1, 0, 3, 2),  # face_idx 0
                        (1, 0, 3, 2),  # face_idx 1
                        (0, 3, 2, 1),  # face_idx 2
                        (1, 0, 3, 2),  # face_idx 3
                        (0, 1, 2, 3),  # face_idx 4
                        (3, 2, 1, 0),  # face_idx 5
                    ]
                    perm = face_corner_perm[face_idx] if 0 <= face_idx < len(face_corner_perm) else (0, 1, 2, 3)

                    for corner_i, vi in enumerate(face_verts):
                        li = v_to_loop.get(vi)
                        if li is None:
                            continue
                        uv_data[li].uv = uvs[int(perm[corner_i])]

        boxes = m.get("boxes")
        _import_boxes(parent=empty, box_list=boxes, pivot_world=pivot, base_object_name=part_name, name_start_index=0)

        submodels = m.get("submodels")
        if isinstance(submodels, list):
            for si, sm in enumerate(submodels):
                if not isinstance(sm, dict):
                    continue

                sm_name = str(sm.get("id") or sm.get("part") or f"submodel_{si:02d}")

                sm_inv = _cem_invert_flags(sm.get("invertAxis"))

                sm_t = sm.get("translate")
                if not (isinstance(sm_t, list) and len(sm_t) == 3):
                    sm_t = [0.0, 0.0, 0.0]
                sm_t = [float(sm_t[0]), float(sm_t[1]), float(sm_t[2])]

                sm_r = sm.get("rotate")
                if not (isinstance(sm_r, list) and len(sm_r) == 3):
                    sm_r = [0.0, 0.0, 0.0]
                sm_r = [float(sm_r[0]), float(sm_r[1]), float(sm_r[2])]

                sm_t_use = [-float(sm_t[0]), -float(sm_t[1]), -float(sm_t[2])]
                sm_off = _cem_translate_to_blender_xyz((float(sm_t_use[0]), float(sm_t_use[1]), float(sm_t_use[2])))
                sm_pivot = (pivot[0] + sm_off[0], pivot[1] + sm_off[1], pivot[2] + sm_off[2])

                sm_empty = bpy.data.objects.new(f"CEM_{sm_name}", None)
                bpy.context.scene.collection.objects.link(sm_empty)
                sm_empty.parent = empty
                sm_empty.location = (sm_pivot[0] - pivot[0], sm_pivot[1] - pivot[1], sm_pivot[2] - pivot[2])
                sm_empty.rotation_mode = "XYZ"
                sm_empty.rotation_euler[0] = math.radians(float(sm_r[0]))
                sm_empty.rotation_euler[2] = math.radians(-float(sm_r[1]))
                sm_empty.rotation_euler[1] = math.radians(-float(sm_r[2]))

                _import_boxes(parent=sm_empty, box_list=sm.get("boxes"), pivot_world=sm_pivot, base_object_name=sm_name, name_start_index=0)

    if ensure_cameras:
        _ensure_default_cameras(bpy)


def _ensure_material_for_texture(bpy, identity: ModelIdentity, texture_id: str):
    safe_name = "MC_" + texture_id.replace(":", "__").replace("/", "__")
    mat = bpy.data.materials.get(safe_name)
    if mat is None:
        mat = bpy.data.materials.new(name=safe_name)
    mat.use_nodes = True

    nt = mat.node_tree
    nodes = nt.nodes
    links = nt.links

    bsdf = nodes.get("Principled BSDF")
    out = nodes.get("Material Output")
    if bsdf is None or out is None:
        nodes.clear()
        bsdf = nodes.new(type="ShaderNodeBsdfPrincipled")
        out = nodes.new(type="ShaderNodeOutputMaterial")
        out.location = (300, 0)
        bsdf.location = (0, 0)
        links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])

    tex_node = None
    for n in nodes:
        if n.type == "TEX_IMAGE" and n.name.startswith("MC_TEX"):
            tex_node = n
            break

    if tex_node is None:
        tex_node = nodes.new(type="ShaderNodeTexImage")
        tex_node.name = "MC_TEX"
        tex_node.location = (-300, 0)

    try:
        tex_node.interpolation = "Closest"
    except Exception:
        pass

    img_path = _texture_id_to_png_path(identity, texture_id)
    if img_path is not None:
        try:
            img = bpy.data.images.load(str(img_path), check_existing=True)
            tex_node.image = img
        except Exception:
            tex_node.image = None

    if not any(l.from_node == tex_node and l.to_node == bsdf for l in links):
        for l in list(links):
            if l.to_node == bsdf and l.to_socket and l.to_socket.name in {"Base Color", "Alpha"}:
                links.remove(l)
        links.new(tex_node.outputs.get("Color"), bsdf.inputs.get("Base Color"))
        if tex_node.outputs.get("Alpha") is not None:
            links.new(tex_node.outputs.get("Alpha"), bsdf.inputs.get("Alpha"))

    mat.blend_method = "CLIP"
    mat.shadow_method = "CLIP"
    bsdf.inputs["Alpha"].default_value = 1.0
    return mat


def _mc_to_blender_xyz(mc: tuple[float, float, float]) -> tuple[float, float, float]:
    mx, my, mz = mc
    bx = (mx - 8.0) / 16.0
    by = -((mz - 8.0) / 16.0)
    bz = (my - 8.0) / 16.0
    return (bx, by, bz)


def _mc_axis_to_blender(axis: str, angle_deg: float) -> tuple[int, float]:
    a = float(angle_deg)
    if axis == "x":
        return 0, a
    if axis == "y":
        return 2, a
    if axis == "z":
        return 1, -a
    return 2, a


def _uv_rect(u1: float, v1: float, u2: float, v2: float) -> list[tuple[float, float]]:
    u1n = u1 / 16.0
    u2n = u2 / 16.0
    v1n = 1.0 - (v1 / 16.0)
    v2n = 1.0 - (v2 / 16.0)
    tl = (u1n, v1n)
    tr = (u2n, v1n)
    br = (u2n, v2n)
    bl = (u1n, v2n)
    return [bl, br, tr, tl]


def _rotate_uvs(uvs: list[tuple[float, float]], rot_deg: int) -> list[tuple[float, float]]:
    r = int(rot_deg) % 360
    if r == 0:
        return uvs
    if r == 90:
        return [uvs[1], uvs[2], uvs[3], uvs[0]]
    if r == 180:
        return [uvs[2], uvs[3], uvs[0], uvs[1]]
    if r == 270:
        return [uvs[3], uvs[0], uvs[1], uvs[2]]
    return uvs


def _ensure_default_cameras(bpy) -> None:
    scene = bpy.context.scene
    scene.render.resolution_x = 1024
    scene.render.resolution_y = 1024
    scene.render.resolution_percentage = 100

    target = bpy.data.objects.get("MC_Target")
    if target is None:
        target = bpy.data.objects.new("MC_Target", None)
        target.empty_display_type = "PLAIN_AXES"
        target.empty_display_size = 0.2
        bpy.context.scene.collection.objects.link(target)

    min_x = min_y = min_z = float("inf")
    max_x = max_y = max_z = float("-inf")

    depsgraph = None
    try:
        depsgraph = bpy.context.evaluated_depsgraph_get()
    except Exception:
        depsgraph = None

    for obj in scene.objects:
        if obj.type != "MESH":
            continue

        eval_obj = obj
        if depsgraph is not None:
            try:
                eval_obj = obj.evaluated_get(depsgraph)
            except Exception:
                eval_obj = obj

        mesh = None
        try:
            mesh = eval_obj.to_mesh()
        except Exception:
            mesh = None

        if mesh is None:
            continue

        mw = eval_obj.matrix_world
        try:
            for v in mesh.vertices:
                co = mw @ v.co
                wx, wy, wz = float(co.x), float(co.y), float(co.z)
                min_x = min(min_x, wx)
                min_y = min(min_y, wy)
                min_z = min(min_z, wz)
                max_x = max(max_x, wx)
                max_y = max(max_y, wy)
                max_z = max(max_z, wz)
        finally:
            try:
                eval_obj.to_mesh_clear()
            except Exception:
                pass

    if not (math.isfinite(min_x) and math.isfinite(min_y) and math.isfinite(min_z)):
        min_x = min_y = min_z = -0.5
        max_x = max_y = max_z = 0.5

    cx = (min_x + max_x) * 0.5
    cy = (min_y + max_y) * 0.5
    cz = (min_z + max_z) * 0.5
    ex = max_x - min_x
    ey = max_y - min_y
    ez = max_z - min_z

    target.location = (cx, cy, cz)

    pad = 2.0
    margin = 1.02
    pad_scale = 1.03
    dist = (0.5 * max(ex, ey, ez) + pad) * pad_scale
    if dist < 0.5:
        dist = (0.5 + pad) * pad_scale

    ortho_front_back = max(ex, ez) * margin * pad_scale
    ortho_left_right = max(ey, ez) * margin * pad_scale

    if ortho_front_back < 0.01:
        ortho_front_back = 0.01
    if ortho_left_right < 0.01:
        ortho_left_right = 0.01

    name_map = {
        "north": "front",
        "south": "back",
        "east": "right",
        "west": "left",
    }

    for old_name in ["up", "down"]:
        obj = bpy.data.objects.get(old_name)
        if obj is not None and obj.type == "CAMERA":
            cam_data = obj.data
            bpy.data.objects.remove(obj, do_unlink=True)
            if cam_data and getattr(cam_data, "users", 0) <= 0:
                bpy.data.cameras.remove(cam_data)

    for old, new in name_map.items():
        obj = bpy.data.objects.get(old)
        if obj is not None and obj.type == "CAMERA":
            obj.name = new
            if obj.data is not None:
                obj.data.name = new

    cams = {
        "front": ((cx, cy + dist, cz), ortho_front_back),
        "back": ((cx, cy - dist, cz), ortho_front_back),
        "right": ((cx + dist, cy, cz), ortho_left_right),
        "left": ((cx - dist, cy, cz), ortho_left_right),
    }

    for name, (loc, ortho_scale) in cams.items():
        cam_obj = bpy.data.objects.get(name)
        if cam_obj is None or cam_obj.type != "CAMERA":
            cam_data = bpy.data.cameras.new(name)
            cam_obj = bpy.data.objects.new(name, cam_data)
            bpy.context.scene.collection.objects.link(cam_obj)
        cam_data = cam_obj.data
        cam_data.type = "ORTHO"
        cam_data.ortho_scale = float(ortho_scale)
        cam_data.clip_start = 0.01
        cam_data.clip_end = float(dist * 10.0 + 10.0)
        cam_obj.location = loc
        cam_obj.rotation_mode = "XYZ"

        con = None
        for c in cam_obj.constraints:
            if c.type == "TRACK_TO":
                con = c
                break
        if con is None:
            con = cam_obj.constraints.new(type="TRACK_TO")
        con.target = target
        con.track_axis = "TRACK_NEGATIVE_Z"
        con.up_axis = "UP_Y"


def _import_minecraft_model_json_into_scene(
    bpy,
    *,
    model_json_path: Path,
    identity: ModelIdentity,
    ensure_cameras: bool,
) -> None:
    model = _resolve_model_inheritance(identity, model_json_path)
    elements = list(model.get("elements") or [])
    textures = dict(model.get("textures") or {})

    for idx, el in enumerate(elements):
        fr = el.get("from") or [0.0, 0.0, 0.0]
        to = el.get("to") or [16.0, 16.0, 16.0]

        fx, fy, fz = float(fr[0]), float(fr[1]), float(fr[2])
        tx, ty, tz = float(to[0]), float(to[1]), float(to[2])

        mx0, mx1 = min(fx, tx), max(fx, tx)
        my0, my1 = min(fy, ty), max(fy, ty)
        mz0, mz1 = min(fz, tz), max(fz, tz)

        verts = [
            _mc_to_blender_xyz((mx0, my0, mz0)),
            _mc_to_blender_xyz((mx1, my0, mz0)),
            _mc_to_blender_xyz((mx1, my1, mz0)),
            _mc_to_blender_xyz((mx0, my1, mz0)),
            _mc_to_blender_xyz((mx0, my0, mz1)),
            _mc_to_blender_xyz((mx1, my0, mz1)),
            _mc_to_blender_xyz((mx1, my1, mz1)),
            _mc_to_blender_xyz((mx0, my1, mz1)),
        ]

        face_keys = ["north", "south", "west", "east", "down", "up"]
        faces = [
            (0, 1, 2, 3),
            (4, 5, 6, 7),
            (0, 3, 7, 4),
            (1, 5, 6, 2),
            (0, 4, 5, 1),
            (3, 2, 6, 7),
        ]

        mesh = bpy.data.meshes.new(f"MC_mesh_{idx:04d}")
        mesh.from_pydata(verts, [], faces)
        mesh.update()

        uv_layer = mesh.uv_layers.new(name="UVMap")
        uv_data = uv_layer.data

        obj = bpy.data.objects.new(f"MC_el_{idx:04d}", mesh)
        bpy.context.scene.collection.objects.link(obj)

        faces_dict = el.get("faces") or {}

        mat_slots: dict[str, int] = {}

        def _mat_index_for(tex_id: str) -> int:
            if tex_id not in mat_slots:
                mat = _ensure_material_for_texture(bpy, identity, tex_id)
                mesh.materials.append(mat)
                mat_slots[tex_id] = len(mesh.materials) - 1
            return int(mat_slots[tex_id])

        for poly_idx, poly in enumerate(mesh.polygons):
            fk = face_keys[poly_idx] if poly_idx < len(face_keys) else "north"
            fdef = faces_dict.get(fk) or {}

            tex_ref = str(fdef.get("texture") or "")
            tex_id = _resolve_texture_ref(textures, tex_ref) if tex_ref else ""
            if tex_id:
                poly.material_index = _mat_index_for(tex_id)

            uv = fdef.get("uv")
            if isinstance(uv, list) and len(uv) == 4:
                u1, v1, u2, v2 = float(uv[0]), float(uv[1]), float(uv[2]), float(uv[3])
            else:
                u1, v1, u2, v2 = 0.0, 0.0, 16.0, 16.0

            uvs = _uv_rect(u1, v1, u2, v2)
            rot = int(fdef.get("rotation") or 0)
            uvs = _rotate_uvs(uvs, rot)

            loops = list(poly.loop_indices)
            for li, uv_val in zip(loops, uvs):
                uv_data[li].uv = uv_val

        rot = el.get("rotation")
        if isinstance(rot, dict):
            axis = str(rot.get("axis") or "")
            angle = float(rot.get("angle") or 0.0)
            origin = rot.get("origin")
            if axis in {"x", "y", "z"} and origin and isinstance(origin, list) and len(origin) == 3:
                ox, oy, oz = _mc_to_blender_xyz((float(origin[0]), float(origin[1]), float(origin[2])))

                for v in mesh.vertices:
                    v.co.x -= ox
                    v.co.y -= oy
                    v.co.z -= oz

                obj.location = (ox, oy, oz)
                obj.rotation_mode = "XYZ"
                idx_axis, adj_angle = _mc_axis_to_blender(axis, angle)
                obj.rotation_euler[idx_axis] = math.radians(adj_angle)

    if ensure_cameras:
        _ensure_default_cameras(bpy)


def _ensure_dirs(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def main() -> int:
    bpy = _require_bpy()

    parsed = _parse_args(sys.argv)
    if not parsed:
        return 0

    json_path_raw = parsed.get("json") or parsed.get("model") or ""
    if not json_path_raw:
        print("Missing --json <path>")
        return 2

    model_path = Path(json_path_raw)
    rp_root = Path(parsed["resourcepack_root"]) if parsed.get("resourcepack_root") else None
    identity = _identify_model(model_path, rp_root)

    overwrite = str(parsed.get("overwrite", "false")).lower() in {"1", "true", "yes", "y"}
    ensure_cameras = str(parsed.get("ensure_cameras", "true")).lower() in {"1", "true", "yes", "y"}

    out_blend_raw = str(parsed.get("output_blend") or "").strip()
    out_blend = Path(out_blend_raw) if out_blend_raw else identity.cache_blend_path()
    if out_blend.exists() and not overwrite:
        print(f"SKIP blend exists: {out_blend}")
        return 0

    bpy.ops.wm.read_factory_settings(use_empty=True)
    if model_path.suffix.lower() == ".jem":
        _import_optifine_cem_jem_into_scene(bpy, jem_path=model_path, identity=identity, ensure_cameras=ensure_cameras)
    else:
        _import_minecraft_model_json_into_scene(bpy, model_json_path=model_path, identity=identity, ensure_cameras=ensure_cameras)

    _ensure_dirs(out_blend.parent)
    bpy.ops.wm.save_as_mainfile(filepath=str(out_blend))
    print(f"WROTE {out_blend}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
