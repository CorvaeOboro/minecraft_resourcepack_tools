"""Shared helpers for the procedural shape generators.

Provides snapping, rotation quantization, cuboid transforms, greedy merge,
depth stagger, and Minecraft model JSON export. This module is headless
(no PySide6 dependency) and can be imported and unit-tested independently
of the UI.
"""

from __future__ import annotations

import math
from typing import Literal, Optional

# region IMPORTS
try:
    from mc_model_solver_core import Cuboid, Rotation
except Exception as exc:
    print(f"Missing dependency: mc_model_solver_core.py ({exc})")
    raise
# endregion

Plane = Literal["XZ", "XY", "YZ"]

_ALLOWED_ANGLES = (-45.0, -22.5, 0.0, 22.5, 45.0)


# region UTILS
def _clamp(val: float, lo: float, hi: float) -> float:
    return lo if val < lo else hi if val > hi else val


def _round_to_step(val: float, step: float) -> float:
    step = float(step)
    if step <= 0.0:
        return float(val)
    return round(float(val) / step) * step


def _parse_vec3_str(text: str) -> tuple[float, float, float]:
    raw = str(text).strip()
    parts = [part.strip() for part in raw.split(",")]
    if len(parts) != 3:
        raise ValueError("Vector must be x,y,z")
    return float(parts[0]), float(parts[1]), float(parts[2])


def _snap_min(val: float, step: float) -> float:
    step = float(step)
    if step <= 0.0:
        return float(val)
    return math.floor(float(val) / step) * step


def _snap_max(val: float, step: float) -> float:
    step = float(step)
    if step <= 0.0:
        return float(val)
    return math.ceil(float(val) / step) * step


def _normalize_fr_to(fr: tuple[float, float, float], to: tuple[float, float, float]) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    fx, fy, fz = fr
    tx, ty, tz = to
    return (min(fx, tx), min(fy, ty), min(fz, tz)), (max(fx, tx), max(fy, ty), max(fz, tz))


def _lerp(val_a: float, val_b: float, t: float) -> float:
    return (float(val_a) * (1.0 - float(t))) + (float(val_b) * float(t))


# region ROTATE
def _wrap180(deg: float) -> float:
    deg_val = float(deg)
    deg_val = (deg_val + 180.0) % 360.0 - 180.0
    return float(deg_val)


def _quantize_angle(deg: float) -> float:
    deg_val = float(deg)
    best = float(_ALLOWED_ANGLES[0])
    best_err = 1e18
    for cand_angle in _ALLOWED_ANGLES:
        err = abs(float(cand_angle) - deg_val)
        if err < best_err:
            best_err = err
            best = float(cand_angle)
    return float(best)


def _reduce_to_limited_rotation(angle_deg: float) -> tuple[bool, float]:
    """Reduce an arbitrary heading angle to Minecraft's limited rotation range.

    Minecraft allows element rotations in [-45, 45] degrees. Angles outside
    that range can be represented by swapping the cuboid's long axis (a 90-degree
    equivalent) and using a reduced rotation. This function returns a tuple of
    (swap_axes, reduced_angle). The reduced angle is wrapped to [-180, 180] but
    is NOT guaranteed to fall within [-45, 45] — callers should pass it through
    ``_quantize_angle`` to snap to the nearest allowed angle.
    """
    a = _wrap180(float(angle_deg))
    if a > 45.0:
        return True, _wrap180(a - 90.0)
    if a < -45.0:
        return True, _wrap180(a + 90.0)
    return False, float(a)


def _canonical_direction_deg(deg: float) -> float:
    deg_val = _wrap180(float(deg))
    if deg_val > 90.0:
        deg_val -= 180.0
    elif deg_val < -90.0:
        deg_val += 180.0
    return float(deg_val)


# endregion
# region TRANSFORM
def _translate_cuboid(cub: Cuboid, dx: float, dy: float, dz: float) -> Cuboid:
    fr = (float(cub.fr[0]) + float(dx), float(cub.fr[1]) + float(dy), float(cub.fr[2]) + float(dz))
    to = (float(cub.to[0]) + float(dx), float(cub.to[1]) + float(dy), float(cub.to[2]) + float(dz))
    fr2, to2 = _normalize_fr_to(fr, to)
    rot = cub.rotation
    if rot is None:
        return Cuboid(fr=fr2, to=to2, rotation=None)
    rot_origin = rot.origin
    new_origin = (float(rot_origin[0]) + float(dx), float(rot_origin[1]) + float(dy), float(rot_origin[2]) + float(dz))
    return Cuboid(fr=fr2, to=to2, rotation=Rotation(axis=str(rot.axis), angle=float(rot.angle), origin=new_origin))


def _plane_depth_axis(plane: str) -> tuple[float, float, float]:
    plane_str = str(plane)
    if plane_str == "XZ":
        return (0.0, 1.0, 0.0)
    if plane_str == "XY":
        return (0.0, 0.0, 1.0)
    return (1.0, 0.0, 0.0)


def _apply_depth_stagger(elements: list[Cuboid], *, plane: str, step: float) -> list[Cuboid]:
    stagger_step = float(step)
    if abs(stagger_step) <= 1e-18:
        return list(elements)
    ax, ay, az = _plane_depth_axis(str(plane))
    out: list[Cuboid] = []
    for i, cub in enumerate(list(elements)):
        depth_off = float(i) * stagger_step
        out.append(_translate_cuboid(cub, float(ax) * depth_off, float(ay) * depth_off, float(az) * depth_off))
    return out


# endregion
# region EXPORT
def _cuboid_to_minecraft_element_ext(cub: Cuboid, texture_key: str, *, shade_false: bool) -> dict:
    elem: dict = {
        "from": [cub.fr[0], cub.fr[1], cub.fr[2]],
        "to": [cub.to[0], cub.to[1], cub.to[2]],
    }
    if bool(shade_false):
        elem["shade"] = False
    elem["faces"] = {
        "north": {"texture": texture_key},
        "east": {"texture": texture_key},
        "south": {"texture": texture_key},
        "west": {"texture": texture_key},
        "up": {"texture": texture_key},
        "down": {"texture": texture_key},
    }

    if cub.rotation is not None and cub.rotation.angle != 0.0:
        elem["rotation"] = {
            "origin": [cub.rotation.origin[0], cub.rotation.origin[1], cub.rotation.origin[2]],
            "axis": str(cub.rotation.axis),
            "angle": cub.rotation.angle,
        }

    return elem


def export_minecraft_model_ext(*, elements: list[Cuboid], texture: str, particle: Optional[str] = None, shade_false: bool = False) -> dict:
    if particle is None:
        particle = texture
    return {
        "textures": {
            "0": texture,
            "particle": particle,
        },
        "elements": [_cuboid_to_minecraft_element_ext(cub, "#0", shade_false=bool(shade_false)) for cub in elements],
    }


# endregion
# region MERGE
def _greedy_merge_aligned_adjacent(elements: list[Cuboid], *, origin_step: float) -> list[Cuboid]:
    """Merge adjacent aligned cuboids that share a face along one axis.

    Two cuboids are mergeable when they have the same rotation key (axis + angle
    or no rotation) and their extents match on two axes while being adjacent on
    the third. The merge is **single-pass and order-dependent**: the input list
    is scanned left-to-right, and once a cuboid fails to merge with the current
    accumulator the accumulator is flushed and a new group starts. Elements are
    NOT re-checked against earlier groups after a merge. For best results,
    callers should sort elements along the merge axis before calling this
    function. The output is always valid Minecraft geometry — the only risk is
    suboptimal element count when elements are not pre-sorted.
    """
    def rot_key(cub: Cuboid) -> tuple:
        rot = cub.rotation
        if rot is None or abs(float(rot.angle)) <= 1e-12:
            return (None, 0.0)
        return (str(rot.axis), float(rot.angle))

    def almost(val_a: float, val_b: float) -> bool:
        return abs(float(val_a) - float(val_b)) <= 1e-9

    def can_merge(cub_a: Cuboid, cub_b: Cuboid) -> Optional[int]:
        if rot_key(cub_a) != rot_key(cub_b):
            return None

        afr, ato = cub_a.fr, cub_a.to
        bfr, bto = cub_b.fr, cub_b.to

        axes = [0, 1, 2]
        for merge_ax in axes:
            other_axes = [ax for ax in axes if ax != merge_ax]
            if almost(afr[other_axes[0]], bfr[other_axes[0]]) and almost(ato[other_axes[0]], bto[other_axes[0]]) and almost(afr[other_axes[1]], bfr[other_axes[1]]) and almost(ato[other_axes[1]], bto[other_axes[1]]):
                if almost(ato[merge_ax], bfr[merge_ax]) or almost(bto[merge_ax], afr[merge_ax]):
                    return int(merge_ax)
        return None

    def merged(cub_a: Cuboid, cub_b: Cuboid) -> Cuboid:
        fr, to = _normalize_fr_to(
            (min(float(cub_a.fr[0]), float(cub_b.fr[0])), min(float(cub_a.fr[1]), float(cub_b.fr[1])), min(float(cub_a.fr[2]), float(cub_b.fr[2]))),
            (max(float(cub_a.to[0]), float(cub_b.to[0])), max(float(cub_a.to[1]), float(cub_b.to[1])), max(float(cub_a.to[2]), float(cub_b.to[2]))),
        )

        rot_k = rot_key(cub_a)
        if rot_k[0] is None:
            return Cuboid(fr=fr, to=to, rotation=None)

        cx = (float(fr[0]) + float(to[0])) * 0.5
        cy = (float(fr[1]) + float(to[1])) * 0.5
        cz = (float(fr[2]) + float(to[2])) * 0.5
        rot = Rotation(
            axis=str(rot_k[0]),
            angle=float(rot_k[1]),
            origin=(
                float(_round_to_step(cx, float(origin_step))),
                float(_round_to_step(cy, float(origin_step))),
                float(_round_to_step(cz, float(origin_step))),
            ),
        )
        return Cuboid(fr=fr, to=to, rotation=rot)

    src = list(elements)
    if len(src) < 2:
        return src

    out: list[Cuboid] = []
    i = 0
    while i < len(src):
        cur = src[i]
        j = i + 1
        while j < len(src):
            merge_ax = can_merge(cur, src[j])
            if merge_ax is None:
                break
            cur = merged(cur, src[j])
            j += 1
        out.append(cur)
        i = j

    return out


# endregion
# region SCALE
def _transform_cuboid_global(
    cub: Cuboid,
    *,
    scale: float,
    offset: tuple[float, float, float],
    pivot: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> Optional[Cuboid]:
    scale_val = float(scale)
    if scale_val <= 0.0:
        return None
    ox, oy, oz = (float(offset[0]), float(offset[1]), float(offset[2]))

    px, py, pz = (float(pivot[0]), float(pivot[1]), float(pivot[2]))

    fr = ((float(cub.fr[0]) - px) * scale_val + px + ox, (float(cub.fr[1]) - py) * scale_val + py + oy, (float(cub.fr[2]) - pz) * scale_val + pz + oz)
    to = ((float(cub.to[0]) - px) * scale_val + px + ox, (float(cub.to[1]) - py) * scale_val + py + oy, (float(cub.to[2]) - pz) * scale_val + pz + oz)
    fr2, to2 = _normalize_fr_to(fr, to)
    if (to2[0] - fr2[0]) <= 1e-6 or (to2[1] - fr2[1]) <= 1e-6 or (to2[2] - fr2[2]) <= 1e-6:
        return None

    rot2 = cub.rotation
    if rot2 is not None:
        ro = rot2.origin
        rot2 = Rotation(
            axis=str(rot2.axis),
            angle=float(rot2.angle),
            origin=((float(ro[0]) - px) * scale_val + px + ox, (float(ro[1]) - py) * scale_val + py + oy, (float(ro[2]) - pz) * scale_val + pz + oz),
        )
    return Cuboid(fr=fr2, to=to2, rotation=rot2)


def _cuboid_volume(cub: Cuboid) -> float:
    dx = float(cub.to[0] - cub.fr[0])
    dy = float(cub.to[1] - cub.fr[1])
    dz = float(cub.to[2] - cub.fr[2])
    return float(max(0.0, dx) * max(0.0, dy) * max(0.0, dz))


def _scale_cuboid_about_centroid(cub: Cuboid, scale: float, *, origin_step: float) -> Optional[Cuboid]:
    scale_val = float(scale)
    if abs(scale_val - 1.0) <= 1e-12:
        return cub
    if scale_val <= 0.0:
        return None

    fx, fy, fz = (float(cub.fr[0]), float(cub.fr[1]), float(cub.fr[2]))
    tx, ty, tz = (float(cub.to[0]), float(cub.to[1]), float(cub.to[2]))

    cx = (fx + tx) * 0.5
    cy = (fy + ty) * 0.5
    cz = (fz + tz) * 0.5

    ex = (tx - fx) * 0.5 * scale_val
    ey = (ty - fy) * 0.5 * scale_val
    ez = (tz - fz) * 0.5 * scale_val

    nfr = (cx - ex, cy - ey, cz - ez)
    nto = (cx + ex, cy + ey, cz + ez)

    fr2, to2 = _normalize_fr_to(nfr, nto)
    if (to2[0] - fr2[0]) <= 1e-6 or (to2[1] - fr2[1]) <= 1e-6 or (to2[2] - fr2[2]) <= 1e-6:
        return None

    rot2 = cub.rotation
    if rot2 is not None:
        ccx = (float(fr2[0]) + float(to2[0])) * 0.5
        ccy = (float(fr2[1]) + float(to2[1])) * 0.5
        ccz = (float(fr2[2]) + float(to2[2])) * 0.5
        rot2 = Rotation(
            axis=str(rot2.axis),
            angle=float(rot2.angle),
            origin=(
                float(_round_to_step(ccx, origin_step)),
                float(_round_to_step(ccy, origin_step)),
                float(_round_to_step(ccz, origin_step)),
            ),
        )

    return Cuboid(fr=fr2, to=to2, rotation=rot2)
# endregion
