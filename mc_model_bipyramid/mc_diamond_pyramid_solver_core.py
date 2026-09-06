"""Minecraft Diamond/Pyramid Plane Solver (Core)

Pure-Python core logic for generating Minecraft block model JSON that uses *thin
cuboids as planes* to approximate pyramid/diamond geometry. This module has no
PySide6/UI dependencies and can be imported and tested headlessly.

The UI layer lives in `mc_diamond_pyramid_solver_ui.py`.

Presets
- Unilateral bipyramid ("diamond")
- 4-sided pyramid
- Mode: 2 stacked bipyramids (used for variation/assembly workflows)

Expected inputs
- Parameters:
  - Center, join/base Y, half-base
  - Slope angles (mapped onto Minecraft-allowed rotations)
  - Plane thickness, rescale/double-sided/shade options
  - Texture ids for top/bottom
  - Optional atlas UV settings (tile size/gap/border)

Outputs
- `elements` dicts for Minecraft block-model JSON
- Full model JSON dict ready for formatted serialization

"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Literal, Optional


#region DATA
# Type aliases, rotation-angle constants, UV-rotation lookup tables,
# and the Cuboid/Rotation dataclasses that mirror Minecraft model JSON


Axis = Literal["x", "y", "z"]
PresetId = Literal["diamond", "pyramid", "mode"]
PyramidStyle = Literal["short", "tall"]
PyramidOrientation = Literal["up", "down"]

_ALLOWED_ROTATION_ANGLES = (-45.0, -22.5, 22.5, 45.0)
_SLOPE_ANGLES = (22.5, 45.0, 67.5)

_FAVORED_SIDE_UV_ROTATIONS: dict[tuple[PresetId, int], int] = {
    ("pyramid", 1): 0,
    ("pyramid", 2): 180,
    ("pyramid", 3): 90,
    ("pyramid", 4): 270,
    ("diamond", 0): 180,
    ("diamond", 1): 0,
    ("diamond", 2): 270,
    ("diamond", 3): 90,
    ("diamond", 4): 180,
    ("diamond", 5): 180,
    ("diamond", 6): 180,
    ("diamond", 7): 180,
}

_TALL_PYRAMID_FACE_UV_ROTATION: dict[str, int] = {
    "north": 0,
    "south": 180,
    "east": 90,
    "west": 270,
}


@dataclass(frozen=True)
class Rotation:
    axis: Axis
    angle: float
    origin: tuple[float, float, float]
    rescale: bool = False


@dataclass(frozen=True)
class Cuboid:
    fr: tuple[float, float, float]
    to: tuple[float, float, float]
    rotation: Optional[Rotation]

    def contains_point(self, p: tuple[float, float, float]) -> bool:
        x, y, z = p

        if self.rotation is not None:
            ox, oy, oz = self.rotation.origin
            x -= ox
            y -= oy
            z -= oz

            x, y, z = _rot_inv_xyz(axis=self.rotation.axis, angle_deg=self.rotation.angle, x=x, y=y, z=z)

            if bool(self.rotation.rescale):
                a = abs(float(self.rotation.angle))
                cos_a = math.cos(math.radians(a))
                if cos_a > 1e-12:
                    if self.rotation.axis == "x":
                        y *= cos_a
                        z *= cos_a
                    elif self.rotation.axis == "y":
                        x *= cos_a
                        z *= cos_a
                    else:
                        x *= cos_a
                        y *= cos_a

            x += ox
            y += oy
            z += oz

        fx, fy, fz = self.fr
        tx, ty, tz = self.to
        return (fx <= x <= tx) and (fy <= y <= ty) and (fz <= z <= tz)
#endregion


#region HELPERS
# Math utilities: angle snapping, slope-to-rotation mapping,
# degree/radian conversions, axis rotation transforms, vec2 parsing


def _snap_to_allowed(v: float, allowed: tuple[float, ...], *, eps: float = 1e-6) -> float:
    vv = float(v)
    best = min(allowed, key=lambda a: abs(float(a) - vv))
    if abs(float(best) - vv) <= float(eps):
        return float(best)
    raise ValueError(f"Value {vv:g} not in allowed set: {allowed}")


def _slope_to_build_params(*, slope_angle_deg: float) -> tuple[float, PyramidStyle]:
    s = abs(float(slope_angle_deg))
    s = _snap_to_allowed(s, _SLOPE_ANGLES)
    if s > 45.0:
        rot = 90.0 - s
        rot = _snap_to_allowed(rot, tuple(abs(a) for a in _ALLOWED_ROTATION_ANGLES))
        return rot, "tall"
    return s, "short"


def _apex_height(half_base: float, slope_angle_deg: float) -> float:
    """Height of a pyramid apex above its base for a given half-base and slope."""
    a = abs(float(slope_angle_deg))
    t = math.tan(math.radians(a))
    if t <= 1e-9:
        return 0.0
    return float(half_base) * t


def _deg_to_rad(deg: float) -> float:
    return deg * (math.pi / 180.0)


def _rot_inv_xyz(*, axis: Axis, angle_deg: float, x: float, y: float, z: float) -> tuple[float, float, float]:
    if angle_deg == 0.0:
        return x, y, z

    a = _deg_to_rad(angle_deg)
    c = math.cos(a)
    s = math.sin(a)

    if axis == "x":
        yy = (c * y) + (s * z)
        zz = (-s * y) + (c * z)
        return x, yy, zz

    if axis == "y":
        xx = (c * x) + (s * z)
        zz = (-s * x) + (c * z)
        return xx, y, zz

    if axis == "z":
        xx = (c * x) + (s * y)
        yy = (-s * x) + (c * y)
        return xx, yy, z

    raise ValueError(f"Invalid axis: {axis}")


def _rot_fwd_xyz(*, axis: Axis, angle_deg: float, x: float, y: float, z: float) -> tuple[float, float, float]:
    return _rot_inv_xyz(axis=axis, angle_deg=-float(angle_deg), x=x, y=y, z=z)


def _parse_vec2(raw: str) -> tuple[float, float]:
    parts = [p.strip() for p in raw.split(",")]
    if len(parts) != 2:
        raise ValueError("Expected x,z")
    return float(parts[0]), float(parts[1])
#endregion


#region BUILD
# Plane builders: the main API for generating thin-cuboid planes that
# approximate pyramid, diamond (unilateral octahedron), and mode (2 stacked
# bipyramids) shapes. Each returns a list of (Cuboid, outward_face) pairs.


def build_pyramid_planes(
    *,
    center_xz: tuple[float, float],
    base_y: float,
    half_base: float,
    angle_deg: float,
    thickness: float,
    rescale: bool,
    style: PyramidStyle = "short",
    orientation: PyramidOrientation = "up",
    include_base: bool = True,
) -> list[tuple[Cuboid, str]]:
    cx, cz = center_xz

    angle_deg = _snap_to_allowed(float(angle_deg), tuple(abs(a) for a in _ALLOWED_ROTATION_ANGLES))

    a = abs(float(angle_deg))
    t = float(thickness)
    if t <= 0.0:
        raise ValueError("Thickness must be > 0")

    tan_a = math.tan(math.radians(a))
    if tan_a <= 1e-9:
        raise ValueError("Angle too small")

    height_short = float(half_base) * float(tan_a)
    height_tall = float(half_base) / float(tan_a)

    height = float(height_short) if style != "tall" else float(height_tall)
    apex_y = float(base_y) + float(height)

    cos_a = math.cos(math.radians(a))
    if cos_a <= 1e-9:
        raise ValueError("Angle too steep")

    run = float(half_base) / float(cos_a)

    x0 = cx - float(half_base)
    x1 = cx + float(half_base)
    z0 = cz - float(half_base)
    z1 = cz + float(half_base)

    out: list[tuple[Cuboid, str]] = []

    # Base plane
    base = Cuboid(
        fr=(x0, float(base_y) - t * 0.5, z0),
        to=(x1, float(base_y) + t * 0.5, z1),
        rotation=None,
    )
    out.append((base, "down"))

    if style == "tall":
        z_n = cz - float(half_base)
        z_s = cz + float(half_base)
        x_w = cx - float(half_base)
        x_e = cx + float(half_base)

        h_pre = float(half_base) / max(1e-9, math.sin(math.radians(a)))

        c_s = Cuboid(
            fr=(x0, float(base_y), z_s - t * 0.5),
            to=(x1, float(base_y) + h_pre, z_s + t * 0.5),
            rotation=Rotation(axis="x", angle=-a, origin=(cx, float(base_y), z_s), rescale=rescale),
        )
        out.append((c_s, "south"))

        c_n = Cuboid(
            fr=(x0, float(base_y), z_n - t * 0.5),
            to=(x1, float(base_y) + h_pre, z_n + t * 0.5),
            rotation=Rotation(axis="x", angle=a, origin=(cx, float(base_y), z_n), rescale=rescale),
        )
        out.append((c_n, "north"))

        c_e = Cuboid(
            fr=(x_e - t * 0.5, float(base_y), z0),
            to=(x_e + t * 0.5, float(base_y) + h_pre, z1),
            rotation=Rotation(axis="z", angle=a, origin=(x_e, float(base_y), cz), rescale=rescale),
        )
        out.append((c_e, "east"))

        c_w = Cuboid(
            fr=(x_w - t * 0.5, float(base_y), z0),
            to=(x_w + t * 0.5, float(base_y) + h_pre, z1),
            rotation=Rotation(axis="z", angle=-a, origin=(x_w, float(base_y), cz), rescale=rescale),
        )
        out.append((c_w, "west"))
    else:
        apex = (cx, apex_y, cz)

        c_s = Cuboid(
            fr=(x0, apex_y, cz - t * 0.5),
            to=(x1, apex_y + t, cz + run + t * 0.5),
            rotation=Rotation(axis="x", angle=a, origin=apex, rescale=rescale),
        )
        out.append((c_s, "up"))

        c_n = Cuboid(
            fr=(x0, apex_y, cz - run - t * 0.5),
            to=(x1, apex_y + t, cz + t * 0.5),
            rotation=Rotation(axis="x", angle=-a, origin=apex, rescale=rescale),
        )
        out.append((c_n, "up"))

        c_e = Cuboid(
            fr=(cx - t * 0.5, apex_y, z0),
            to=(cx + run + t * 0.5, apex_y + t, z1),
            rotation=Rotation(axis="z", angle=-a, origin=apex, rescale=rescale),
        )
        out.append((c_e, "up"))

        c_w = Cuboid(
            fr=(cx - run - t * 0.5, apex_y, z0),
            to=(cx + t * 0.5, apex_y + t, z1),
            rotation=Rotation(axis="z", angle=a, origin=apex, rescale=rescale),
        )
        out.append((c_w, "up"))

    if not bool(include_base):
        out = out[1:]

    if orientation == "down":
        def flip_face(f: str) -> str:
            if f == "up":
                return "down"
            if f == "down":
                return "up"
            return f

        def flip_cub(c: Cuboid) -> Cuboid:
            fx, fy, fz = c.fr
            tx, ty, tz = c.to
            fy2 = (2.0 * float(base_y)) - float(fy)
            ty2 = (2.0 * float(base_y)) - float(ty)
            fr2 = (float(fx), min(fy2, ty2), float(fz))
            to2 = (float(tx), max(fy2, ty2), float(tz))

            rot2: Optional[Rotation] = None
            if c.rotation is not None:
                ox, oy, oz = c.rotation.origin
                oy2 = (2.0 * float(base_y)) - float(oy)
                ang = float(c.rotation.angle)
                if c.rotation.axis in ("x", "z"):
                    ang = -ang
                rot2 = Rotation(axis=c.rotation.axis, angle=ang, origin=(float(ox), float(oy2), float(oz)), rescale=bool(c.rotation.rescale))

            return Cuboid(fr=fr2, to=to2, rotation=rot2)

        out = [(flip_cub(c), flip_face(f)) for (c, f) in out]

    return out


def build_unilateral_octahedron_planes(
    *,
    center_xz: tuple[float, float],
    join_y: float,
    half_base: float,
    top_angle_deg: float,
    bottom_angle_deg: float,
    top_style: PyramidStyle = "short",
    bottom_style: PyramidStyle = "short",
    thickness: float,
    rescale: bool,
) -> list[tuple[Cuboid, str]]:
    top = build_pyramid_planes(
        center_xz=center_xz,
        base_y=join_y,
        half_base=half_base,
        angle_deg=top_angle_deg,
        thickness=thickness,
        rescale=rescale,
        style=top_style,
        orientation="up",
        include_base=False,
    )
    bot = build_pyramid_planes(
        center_xz=center_xz,
        base_y=join_y,
        half_base=half_base,
        angle_deg=bottom_angle_deg,
        thickness=thickness,
        rescale=rescale,
        style=bottom_style,
        orientation="down",
        include_base=False,
    )

    if len(top) != 4 or len(bot) != 4:
        raise ValueError("Internal error building bipyramid")

    # build_pyramid_planes ramp order: south, north, east, west
    # desired plane order: top_north, top_south, top_west, top_east, bottom_north, bottom_south, bottom_west, bottom_east
    out = [top[1], top[0], top[3], top[2], bot[1], bot[0], bot[3], bot[2]]
    return out


def build_mode_planes(
    *,
    center_xz: tuple[float, float],
    join_y: float,
    half_base: float,
    thickness: float,
    rescale: bool,
) -> list[tuple[Cuboid, str]]:
    top1_slope = 67.5
    bot1_slope = 45.0
    top2_slope = 45.0
    bot2_slope = 67.5

    top1_rot, top1_style = _slope_to_build_params(slope_angle_deg=top1_slope)
    bot1_rot, bot1_style = _slope_to_build_params(slope_angle_deg=bot1_slope)
    top2_rot, top2_style = _slope_to_build_params(slope_angle_deg=top2_slope)
    bot2_rot, bot2_style = _slope_to_build_params(slope_angle_deg=bot2_slope)

    # Stack the second bipyramid below the first, tip-to-tip.
    join_y2 = float(join_y) - _apex_height(half_base, bot1_slope) - _apex_height(half_base, top2_slope)

    a = build_unilateral_octahedron_planes(
        center_xz=center_xz,
        join_y=float(join_y),
        half_base=half_base,
        top_angle_deg=top1_rot,
        bottom_angle_deg=bot1_rot,
        top_style=top1_style,
        bottom_style=bot1_style,
        thickness=thickness,
        rescale=rescale,
    )
    b = build_unilateral_octahedron_planes(
        center_xz=center_xz,
        join_y=float(join_y2),
        half_base=half_base,
        top_angle_deg=top2_rot,
        bottom_angle_deg=bot2_rot,
        top_style=top2_style,
        bottom_style=bot2_style,
        thickness=thickness,
        rescale=rescale,
    )
    return list(a) + list(b)
#endregion


#region EXPORT
# Convert (Cuboid, face) pairs into Minecraft block-model JSON element dicts,
# assemble the full model dict, and pretty-print with inline short arrays


def _element_for_plane(
    *,
    cuboid: Cuboid,
    texture_key: str,
    outward_face: str,
    double_sided: bool,
    force_full_uv: bool,
    uv: Optional[list[float]] = None,
    face_rotation: Optional[int] = None,
    shade: bool,
) -> dict:
    faces: dict = {}
    for f in _faces_for_plane(outward_face=outward_face, double_sided=double_sided):
        d = {"texture": texture_key}
        if uv is not None:
            d["uv"] = list(uv)
        elif force_full_uv:
            d["uv"] = [0.0, 0.0, 16.0, 16.0]
        r = None if face_rotation is None else int(face_rotation)
        if r in (0, 90, 180, 270) and int(r) != 0:
            d["rotation"] = int(r)
        faces[f] = d

    el: dict = {
        "from": [cuboid.fr[0], cuboid.fr[1], cuboid.fr[2]],
        "to": [cuboid.to[0], cuboid.to[1], cuboid.to[2]],
        "faces": faces,
    }

    if not shade:
        el["shade"] = False

    if cuboid.rotation is not None and float(cuboid.rotation.angle) != 0.0:
        rot: dict = {
            "origin": [cuboid.rotation.origin[0], cuboid.rotation.origin[1], cuboid.rotation.origin[2]],
            "axis": cuboid.rotation.axis,
            "angle": cuboid.rotation.angle,
        }
        if bool(cuboid.rotation.rescale):
            rot["rescale"] = True
        el["rotation"] = rot

    return el


def export_minecraft_model(*, elements: list[dict], textures: dict, particle: Optional[str] = None) -> dict:
    if particle is None:
        particle = str(textures.get("0", "minecraft:block/oak_planks"))

    out = {
        "textures": dict(textures),
        "elements": list(elements),
    }
    if "particle" not in out["textures"]:
        out["textures"]["particle"] = particle
    return out


def format_minecraft_model_json(model: dict) -> str:
    def is_scalar(v) -> bool:
        return v is None or isinstance(v, (str, int, float, bool))

    def try_inline(obj) -> Optional[str]:
        if isinstance(obj, list):
            if all(is_scalar(x) for x in obj) and len(obj) <= 16:
                s = json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
                if len(s) <= 80:
                    return s
            return None

        if isinstance(obj, dict):
            if all(isinstance(k, str) for k in obj.keys()):
                ok = True
                for v in obj.values():
                    if is_scalar(v):
                        continue
                    if isinstance(v, list) and all(is_scalar(x) for x in v) and len(v) <= 16:
                        continue
                    if isinstance(v, dict) and all(isinstance(kk, str) for kk in v.keys()) and all(
                        is_scalar(vv) for vv in v.values()
                    ):
                        continue
                    ok = False
                    break
                if ok:
                    s = json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
                    if len(s) <= 80:
                        return s
            return None

        if is_scalar(obj):
            return json.dumps(obj, ensure_ascii=False)

        return None

    def fmt(obj, level: int) -> str:
        inline = try_inline(obj)
        if inline is not None:
            return inline

        ind = "  " * level
        ind2 = "  " * (level + 1)

        if isinstance(obj, list):
            if not obj:
                return "[]"
            parts = []
            for item in obj:
                parts.append(f"{ind2}{fmt(item, level + 1)}")
            return "[\n" + ",\n".join(parts) + f"\n{ind}]"

        if isinstance(obj, dict):
            if not obj:
                return "{}"
            parts = []
            for k, v in obj.items():
                ks = json.dumps(k, ensure_ascii=False)
                vs = fmt(v, level + 1)
                if "\n" in vs:
                    vs = "\n".join([vs.split("\n", 1)[0]] + [ind2 + line for line in vs.split("\n")[1:]])
                parts.append(f"{ind2}{ks}: {vs}")
            return "{\n" + ",\n".join(parts) + f"\n{ind}}}"

        return json.dumps(obj, ensure_ascii=False)

    return fmt(model, 0)
#endregion


#region ROTATE
# Forward/inverse rotation of points about a Cuboid's rotation origin.
# Used by the UI's atlas generation and viewport UV-debug overlays.


def _fwd_rotate_point(*, c: Cuboid, p: tuple[float, float, float]) -> tuple[float, float, float]:
    x, y, z = p
    if c.rotation is None or float(c.rotation.angle) == 0.0:
        return x, y, z

    ox, oy, oz = c.rotation.origin
    x -= ox
    y -= oy
    z -= oz

    if bool(c.rotation.rescale):
        a = abs(float(c.rotation.angle))
        cos_a = math.cos(math.radians(a))
        if cos_a > 1e-12:
            s = 1.0 / cos_a
            if c.rotation.axis == "x":
                y *= s
                z *= s
            elif c.rotation.axis == "y":
                x *= s
                z *= s
            else:
                x *= s
                y *= s

    x, y, z = _rot_fwd_xyz(axis=c.rotation.axis, angle_deg=c.rotation.angle, x=x, y=y, z=z)
    x += ox
    y += oy
    z += oz
    return x, y, z


def _inv_rotate_point(*, c: Cuboid, p: tuple[float, float, float]) -> tuple[float, float, float]:
    x, y, z = p
    if c.rotation is None or float(c.rotation.angle) == 0.0:
        return x, y, z

    ox, oy, oz = c.rotation.origin
    x -= ox
    y -= oy
    z -= oz
    x, y, z = _rot_inv_xyz(axis=c.rotation.axis, angle_deg=c.rotation.angle, x=x, y=y, z=z)

    if bool(c.rotation.rescale):
        a = abs(float(c.rotation.angle))
        cos_a = math.cos(math.radians(a))
        if cos_a > 1e-12:
            if c.rotation.axis == "x":
                y *= cos_a
                z *= cos_a
            elif c.rotation.axis == "y":
                x *= cos_a
                z *= cos_a
            else:
                x *= cos_a
                y *= cos_a

    x += ox
    y += oy
    z += oz
    return x, y, z
#endregion


#region FACE
# Face/plane geometry helpers: axis-bounds lookup, point-on-face sampling,
# plane normal extraction, plane-side testing, and double-sided face pairing.
# Used by the UI's cutout-atlas rasterization and viewport rendering.


def _face_axes_and_bounds(*, c: Cuboid, face: str) -> tuple[str, str, float, float, float, float]:
    fx, fy, fz = c.fr
    tx, ty, tz = c.to

    if face == "up" or face == "down":
        return "x", "z", float(fx), float(tx), float(fz), float(tz)
    if face == "north" or face == "south":
        return "x", "y", float(fx), float(tx), float(fy), float(ty)
    if face == "east" or face == "west":
        return "z", "y", float(fz), float(tz), float(fy), float(ty)
    raise ValueError(f"Invalid face: {face}")


def _get_axis_value(axis: str, p: tuple[float, float, float]) -> float:
    if axis == "x":
        return float(p[0])
    if axis == "y":
        return float(p[1])
    if axis == "z":
        return float(p[2])
    raise ValueError(f"Invalid axis: {axis}")


def _point_on_face_unrotated(*, c: Cuboid, face: str, u: float, v: float) -> tuple[float, float, float]:
    a_axis, b_axis, amin, amax, bmin, bmax = _face_axes_and_bounds(c=c, face=face)
    aa = amin + (amax - amin) * float(u)
    bb = bmin + (bmax - bmin) * float(v)

    fx, fy, fz = c.fr
    tx, ty, tz = c.to
    if face == "up":
        fixed = float(ty)
        if a_axis == "x":
            return (aa, fixed, bb)
        return (bb, fixed, aa)
    if face == "down":
        fixed = float(fy)
        if a_axis == "x":
            return (aa, fixed, bb)
        return (bb, fixed, aa)
    if face == "north":
        fixed = float(fz)
        return (aa, bb, fixed)
    if face == "south":
        fixed = float(tz)
        return (aa, bb, fixed)
    if face == "east":
        fixed = float(tx)
        return (fixed, bb, aa)
    if face == "west":
        fixed = float(fx)
        return (fixed, bb, aa)
    raise ValueError(f"Invalid face: {face}")


def _plane_from_face(*, c: Cuboid, face: str) -> tuple[tuple[float, float, float], float]:
    fx, fy, fz = c.fr
    tx, ty, tz = c.to

    if face == "up":
        p0u = (fx, ty, fz)
        p1u = (tx, ty, fz)
        p2u = (fx, ty, tz)
    elif face == "down":
        p0u = (fx, fy, fz)
        p1u = (fx, fy, tz)
        p2u = (tx, fy, fz)
    elif face == "north":
        p0u = (fx, fy, fz)
        p1u = (tx, fy, fz)
        p2u = (fx, ty, fz)
    elif face == "south":
        p0u = (fx, fy, tz)
        p1u = (fx, ty, tz)
        p2u = (tx, fy, tz)
    elif face == "east":
        p0u = (tx, fy, fz)
        p1u = (tx, ty, fz)
        p2u = (tx, fy, tz)
    elif face == "west":
        p0u = (fx, fy, fz)
        p1u = (fx, fy, tz)
        p2u = (fx, ty, fz)
    else:
        raise ValueError(f"Invalid face: {face}")

    p0 = _fwd_rotate_point(c=c, p=p0u)
    p1 = _fwd_rotate_point(c=c, p=p1u)
    p2 = _fwd_rotate_point(c=c, p=p2u)

    ax, ay, az = (p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2])
    bx, by, bz = (p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2])
    nx = (ay * bz) - (az * by)
    ny = (az * bx) - (ax * bz)
    nz = (ax * by) - (ay * bx)

    nlen = math.sqrt((nx * nx) + (ny * ny) + (nz * nz))
    if nlen > 1e-12:
        inv = 1.0 / nlen
        nx *= inv
        ny *= inv
        nz *= inv

    d = -((nx * p0[0]) + (ny * p0[1]) + (nz * p0[2]))
    return (nx, ny, nz), float(d)


def _plane_side(n: tuple[float, float, float], d: float, p: tuple[float, float, float]) -> float:
    return (n[0] * p[0]) + (n[1] * p[1]) + (n[2] * p[2]) + float(d)


def _faces_for_plane(*, outward_face: str, double_sided: bool) -> list[str]:
    if not double_sided:
        return [outward_face]

    opp = {
        "north": "south",
        "south": "north",
        "east": "west",
        "west": "east",
        "up": "down",
        "down": "up",
    }[outward_face]

    return [outward_face, opp]
#endregion


#region ATLAS
# Atlas UV layout math: grid sizing, pixel layout, and per-tile UV rects.
# No image generation here (that requires PySide6 and lives in the UI module).


def _atlas_grid_count(n: int) -> int:
    n = max(1, int(n))
    return int(math.ceil(math.sqrt(float(n))))


def _atlas_layout_px(*, n: int, tile_px: int, gap_px: int, total_px: int, border_px: int = 0) -> tuple[int, int, int]:
    g = _atlas_grid_count(n)

    gap_px = max(0, int(gap_px))
    border_px = max(0, int(border_px))
    tile_px = max(1, int(tile_px))
    total_px = int(total_px)

    if total_px > 0:
        usable = total_px - (2 * border_px) - ((g - 1) * gap_px)
        tile_px = max(1, usable // g)

    atlas_px = (g * tile_px) + ((g - 1) * gap_px) + (2 * border_px)
    return g, tile_px, atlas_px


def _atlas_uv_rect(*, index: int, n: int, tile_px: int, gap_px: int, total_px: int, border_px: int = 0) -> list[float]:
    g, tile_px, atlas_px = _atlas_layout_px(n=n, tile_px=tile_px, gap_px=gap_px, total_px=total_px, border_px=border_px)

    tile_uv = 16.0 * (float(tile_px) / float(atlas_px))
    gap_uv = 16.0 * (float(max(0, int(gap_px))) / float(atlas_px))
    border_uv = 16.0 * (float(max(0, int(border_px))) / float(atlas_px))

    ix = int(index) % g
    iy = int(index) // g

    step = tile_uv + gap_uv
    u0 = border_uv + float(ix) * step
    v0 = border_uv + float(iy) * step
    return [u0, v0, u0 + tile_uv, v0 + tile_uv]
#endregion
