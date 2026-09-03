from __future__ import annotations

import math
from typing import Literal, Optional

try:
    from mc_model_solver_ui import Cuboid, Rotation
except Exception as e:
    print(f"Missing dependency: mc_model_solver_ui.py ({e})")
    raise

Plane = Literal["XZ", "XY", "YZ"]

_ALLOWED_ANGLES = (-45.0, -22.5, 0.0, 22.5, 45.0)


def _clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else hi if v > hi else v


def _round_to_step(v: float, step: float) -> float:
    step = float(step)
    if step <= 0.0:
        return float(v)
    return round(float(v) / step) * step


def _parse_vec3_str(text: str) -> tuple[float, float, float]:
    raw = str(text).strip()
    parts = [p.strip() for p in raw.split(",")]
    if len(parts) != 3:
        raise ValueError("Vector must be x,y,z")
    return float(parts[0]), float(parts[1]), float(parts[2])


def _snap_min(v: float, step: float) -> float:
    step = float(step)
    if step <= 0.0:
        return float(v)
    return math.floor(float(v) / step) * step


def _snap_max(v: float, step: float) -> float:
    step = float(step)
    if step <= 0.0:
        return float(v)
    return math.ceil(float(v) / step) * step


def _normalize_fr_to(fr: tuple[float, float, float], to: tuple[float, float, float]) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    fx, fy, fz = fr
    tx, ty, tz = to
    return (min(fx, tx), min(fy, ty), min(fz, tz)), (max(fx, tx), max(fy, ty), max(fz, tz))


def _lerp(a: float, b: float, t: float) -> float:
    return (float(a) * (1.0 - float(t))) + (float(b) * float(t))


def _wrap180(deg: float) -> float:
    d = float(deg)
    d = (d + 180.0) % 360.0 - 180.0
    return float(d)


def _quantize_angle(deg: float) -> float:
    d = float(deg)
    best = float(_ALLOWED_ANGLES[0])
    best_err = 1e18
    for a in _ALLOWED_ANGLES:
        err = abs(float(a) - d)
        if err < best_err:
            best_err = err
            best = float(a)
    if abs(best) < 1e-9:
        return 0.0
    return float(best)


def _reduce_to_limited_rotation(angle_deg: float) -> tuple[bool, float]:
    a = _wrap180(float(angle_deg))
    if a > 45.0:
        return True, _wrap180(a - 90.0)
    if a < -45.0:
        return True, _wrap180(a + 90.0)
    return False, float(a)


def _canonical_direction_deg(deg: float) -> float:
    d = _wrap180(float(deg))
    if d > 90.0:
        d -= 180.0
    elif d < -90.0:
        d += 180.0
    return float(d)


def _translate_cuboid(c: Cuboid, dx: float, dy: float, dz: float) -> Cuboid:
    fr = (float(c.fr[0]) + float(dx), float(c.fr[1]) + float(dy), float(c.fr[2]) + float(dz))
    to = (float(c.to[0]) + float(dx), float(c.to[1]) + float(dy), float(c.to[2]) + float(dz))
    fr2, to2 = _normalize_fr_to(fr, to)
    r = c.rotation
    if r is None:
        return Cuboid(fr=fr2, to=to2, rotation=None)
    o = r.origin
    o2 = (float(o[0]) + float(dx), float(o[1]) + float(dy), float(o[2]) + float(dz))
    return Cuboid(fr=fr2, to=to2, rotation=Rotation(axis=str(r.axis), angle=float(r.angle), origin=o2))


def _plane_depth_axis(plane: str) -> tuple[float, float, float]:
    p = str(plane)
    if p == "XZ":
        return (0.0, 1.0, 0.0)
    if p == "XY":
        return (0.0, 0.0, 1.0)
    return (1.0, 0.0, 0.0)


def _apply_depth_stagger(elements: list[Cuboid], *, plane: str, step: float) -> list[Cuboid]:
    s = float(step)
    if abs(s) <= 1e-18:
        return list(elements)
    ax, ay, az = _plane_depth_axis(str(plane))
    out: list[Cuboid] = []
    for i, c in enumerate(list(elements)):
        d = float(i) * s
        out.append(_translate_cuboid(c, float(ax) * d, float(ay) * d, float(az) * d))
    return out


def _cuboid_to_minecraft_element_ext(c: Cuboid, texture_key: str, *, shade_false: bool) -> dict:
    el: dict = {
        "from": [c.fr[0], c.fr[1], c.fr[2]],
        "to": [c.to[0], c.to[1], c.to[2]],
    }
    if bool(shade_false):
        el["shade"] = False
    el["faces"] = {
        "north": {"texture": texture_key},
        "east": {"texture": texture_key},
        "south": {"texture": texture_key},
        "west": {"texture": texture_key},
        "up": {"texture": texture_key},
        "down": {"texture": texture_key},
    }

    if c.rotation is not None and c.rotation.angle != 0.0:
        el["rotation"] = {
            "origin": [c.rotation.origin[0], c.rotation.origin[1], c.rotation.origin[2]],
            "axis": str(c.rotation.axis),
            "angle": c.rotation.angle,
        }

    return el


def export_minecraft_model_ext(*, elements: list[Cuboid], texture: str, particle: Optional[str] = None, shade_false: bool = False) -> dict:
    if particle is None:
        particle = texture
    return {
        "textures": {
            "0": texture,
            "particle": particle,
        },
        "elements": [_cuboid_to_minecraft_element_ext(c, "#0", shade_false=bool(shade_false)) for c in elements],
    }


def _greedy_merge_aligned_adjacent(elements: list[Cuboid], *, origin_step: float) -> list[Cuboid]:
    def rot_key(c: Cuboid) -> tuple:
        r = c.rotation
        if r is None or abs(float(r.angle)) <= 1e-12:
            return (None, 0.0)
        return (str(r.axis), float(r.angle))

    def almost(a: float, b: float) -> bool:
        return abs(float(a) - float(b)) <= 1e-9

    def can_merge(a: Cuboid, b: Cuboid) -> Optional[int]:
        if rot_key(a) != rot_key(b):
            return None

        afr, ato = a.fr, a.to
        bfr, bto = b.fr, b.to

        axes = [0, 1, 2]
        for ax in axes:
            o1 = [x for x in axes if x != ax]
            if almost(afr[o1[0]], bfr[o1[0]]) and almost(ato[o1[0]], bto[o1[0]]) and almost(afr[o1[1]], bfr[o1[1]]) and almost(ato[o1[1]], bto[o1[1]]):
                if almost(ato[ax], bfr[ax]) or almost(bto[ax], afr[ax]):
                    return int(ax)
        return None

    def merged(a: Cuboid, b: Cuboid) -> Cuboid:
        fr, to = _normalize_fr_to(
            (min(float(a.fr[0]), float(b.fr[0])), min(float(a.fr[1]), float(b.fr[1])), min(float(a.fr[2]), float(b.fr[2]))),
            (max(float(a.to[0]), float(b.to[0])), max(float(a.to[1]), float(b.to[1])), max(float(a.to[2]), float(b.to[2]))),
        )

        rk = rot_key(a)
        if rk[0] is None:
            return Cuboid(fr=fr, to=to, rotation=None)

        cx = (float(fr[0]) + float(to[0])) * 0.5
        cy = (float(fr[1]) + float(to[1])) * 0.5
        cz = (float(fr[2]) + float(to[2])) * 0.5
        rot = Rotation(
            axis=str(rk[0]),
            angle=float(rk[1]),
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
            ax = can_merge(cur, src[j])
            if ax is None:
                break
            cur = merged(cur, src[j])
            j += 1
        out.append(cur)
        i = j

    return out


def _transform_cuboid_global(
    c: Cuboid,
    *,
    scale: float,
    offset: tuple[float, float, float],
    pivot: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> Optional[Cuboid]:
    s = float(scale)
    if s <= 0.0:
        return None
    ox, oy, oz = (float(offset[0]), float(offset[1]), float(offset[2]))

    px, py, pz = (float(pivot[0]), float(pivot[1]), float(pivot[2]))

    fr = ((float(c.fr[0]) - px) * s + px + ox, (float(c.fr[1]) - py) * s + py + oy, (float(c.fr[2]) - pz) * s + pz + oz)
    to = ((float(c.to[0]) - px) * s + px + ox, (float(c.to[1]) - py) * s + py + oy, (float(c.to[2]) - pz) * s + pz + oz)
    fr2, to2 = _normalize_fr_to(fr, to)
    if (to2[0] - fr2[0]) <= 1e-6 or (to2[1] - fr2[1]) <= 1e-6 or (to2[2] - fr2[2]) <= 1e-6:
        return None

    rot2 = c.rotation
    if rot2 is not None:
        ro = rot2.origin
        rot2 = Rotation(
            axis=str(rot2.axis),
            angle=float(rot2.angle),
            origin=((float(ro[0]) - px) * s + px + ox, (float(ro[1]) - py) * s + py + oy, (float(ro[2]) - pz) * s + pz + oz),
        )
    return Cuboid(fr=fr2, to=to2, rotation=rot2)


def _cuboid_volume(c: Cuboid) -> float:
    dx = float(c.to[0] - c.fr[0])
    dy = float(c.to[1] - c.fr[1])
    dz = float(c.to[2] - c.fr[2])
    return float(max(0.0, dx) * max(0.0, dy) * max(0.0, dz))


def _scale_cuboid_about_centroid(c: Cuboid, scale: float, *, origin_step: float) -> Optional[Cuboid]:
    s = float(scale)
    if abs(s - 1.0) <= 1e-12:
        return c
    if s <= 0.0:
        return None

    fx, fy, fz = (float(c.fr[0]), float(c.fr[1]), float(c.fr[2]))
    tx, ty, tz = (float(c.to[0]), float(c.to[1]), float(c.to[2]))

    cx = (fx + tx) * 0.5
    cy = (fy + ty) * 0.5
    cz = (fz + tz) * 0.5

    ex = (tx - fx) * 0.5 * s
    ey = (ty - fy) * 0.5 * s
    ez = (tz - fz) * 0.5 * s

    nfr = (cx - ex, cy - ey, cz - ez)
    nto = (cx + ex, cy + ey, cz + ez)

    fr2, to2 = _normalize_fr_to(nfr, nto)
    if (to2[0] - fr2[0]) <= 1e-6 or (to2[1] - fr2[1]) <= 1e-6 or (to2[2] - fr2[2]) <= 1e-6:
        return None

    rot2 = c.rotation
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
