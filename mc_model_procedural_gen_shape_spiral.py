from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from mc_model_procedural_gen_common import (
    Plane,
    _apply_depth_stagger,
    _canonical_direction_deg,
    _clamp,
    _greedy_merge_aligned_adjacent,
    _lerp,
    _normalize_fr_to,
    _quantize_angle,
    _reduce_to_limited_rotation,
    _round_to_step,
    _snap_max,
    _snap_min,
    _transform_cuboid_global,
)

try:
    from mc_model_solver_ui import Cuboid, Rotation
except Exception as e:
    print(f"Missing dependency: mc_model_solver_ui.py ({e})")
    raise


@dataclass(frozen=True)
class SpiralConfig:
    center: tuple[float, float, float]
    plane: Plane
    segments: int
    turns: float
    guide_rotation_deg: float
    radius_outer: float
    radius_inner: float
    ramp_height: float
    width_outer: float
    width_inner: float
    tangent_len_outer: float
    tangent_len_inner: float
    axis_thickness: float
    snap_step: float
    use_rotations: bool
    reverse_rotation: bool
    chord_direction: bool
    auto_merge: bool
    greedy_merge: bool
    max_merge_len: float
    max_chord_step: float
    merge_max_dev: float


SPIRAL_UI_SPECS: dict[str, dict] = {
    "plane": {
        "type": "combo",
        "items": [("XZ (around Y)", "XZ"), ("XY (around Z)", "XY"), ("YZ (around X)", "YZ")],
        "current": 0,
    },
    "segments": {"type": "spin", "min": 1, "max": 2048, "value": 128},
    "turns": {"type": "double", "min": 0.0, "max": 100.0, "value": 2.5, "decimals": 3},
    "guide_rotation": {"type": "double", "min": -360.0, "max": 360.0, "value": 0.0, "decimals": 3},
    "radius_outer": {"type": "double", "min": 0.0, "max": 64.0, "value": 7.5, "decimals": 3},
    "radius_inner": {"type": "double", "min": 0.0, "max": 64.0, "value": 0.75, "decimals": 3},
    "ramp_height": {"type": "double", "min": -64.0, "max": 64.0, "value": 0.0, "decimals": 3},
    "width_outer": {"type": "double", "min": 0.01, "max": 64.0, "value": 1.0, "decimals": 3},
    "width_inner": {"type": "double", "min": 0.01, "max": 64.0, "value": 0.35, "decimals": 3},
    "tangent_outer": {"type": "double", "min": 0.01, "max": 64.0, "value": 2.0, "decimals": 3},
    "tangent_inner": {"type": "double", "min": 0.01, "max": 64.0, "value": 0.75, "decimals": 3},
    "axis_thickness": {"type": "double", "min": 0.01, "max": 64.0, "value": 1.0, "decimals": 3},
    "snap_step": {"type": "double", "min": 0.0, "max": 2.0, "value": 0.25, "decimals": 3},
    "use_rotations": {"type": "check", "text": "Use limited rotations", "checked": True},
    "reverse_rotation": {"type": "check", "text": "Reverse rotation", "checked": False},
    "chord_direction": {"type": "check", "text": "Chord direction", "checked": True},
    "auto_merge": {"type": "check", "text": "Auto-merge straight sections", "checked": True},
    "greedy_merge": {"type": "check", "text": "Greedy merge pass", "checked": False},
    "max_merge_len": {"type": "double", "min": 0.0, "max": 64.0, "value": 6.0, "decimals": 3},
    "max_chord_step": {"type": "double", "min": 0.0, "max": 64.0, "value": 0.75, "decimals": 3},
    "merge_max_dev": {"type": "double", "min": 0.0, "max": 64.0, "value": 0.15, "decimals": 3},
}


def generate_shape(cfg: SpiralConfig, shape: str, *, phase_offset_rad: float = 0.0) -> list[Cuboid]:
    s = str(shape)
    if s == "double_spiral":
        return generate_double_spiral(cfg, phase_offset_rad=float(phase_offset_rad))
    return generate_spiral(cfg, phase_offset_rad=float(phase_offset_rad))


def generate_spiral(cfg: SpiralConfig, *, phase_offset_rad: float = 0.0) -> list[Cuboid]:
    segs = max(1, int(cfg.segments))

    cx, cy, cz = cfg.center
    plane = str(cfg.plane)

    out: list[Cuboid] = []

    def spiral_point(t: float) -> tuple[float, float, float, float, float]:
        tt = max(0.0, min(1.0, float(t)))
        a = ((2.0 * math.pi) * float(cfg.turns) * tt) + float(phase_offset_rad) + math.radians(float(cfg.guide_rotation_deg))
        r = _lerp(float(cfg.radius_outer), float(cfg.radius_inner), tt)
        r = max(0.0, float(r))

        ramp = float(cfg.ramp_height)
        along = (tt * ramp) - (ramp * 0.5)

        p0 = r * math.cos(a)
        p1 = r * math.sin(a)
        return tt, a, along, p0, p1

    def build_element(*, center: tuple[float, float, float], width: float, thick: float, tlen: float, rot_axis: str, rot_angle: float, swap: bool) -> Optional[Cuboid]:
        x, y, z = (float(center[0]), float(center[1]), float(center[2]))
        width = max(0.01, float(width))
        thick = max(0.01, float(thick))
        tlen = max(0.01, float(tlen))

        if plane == "XZ":
            sx, sy, sz = (width, thick, tlen) if swap else (tlen, thick, width)
        elif plane == "XY":
            sx, sy, sz = (width, tlen, thick) if swap else (tlen, width, thick)
        else:
            sx, sy, sz = (thick, width, tlen) if swap else (thick, tlen, width)

        step = float(cfg.snap_step)

        fx = _snap_min(x - sx * 0.5, step)
        fy = _snap_min(y - sy * 0.5, step)
        fz = _snap_min(z - sz * 0.5, step)
        tx = _snap_max(x + sx * 0.5, step)
        ty = _snap_max(y + sy * 0.5, step)
        tz = _snap_max(z + sz * 0.5, step)

        fx = _clamp(fx, 0.0, 16.0)
        fy = _clamp(fy, 0.0, 16.0)
        fz = _clamp(fz, 0.0, 16.0)
        tx = _clamp(tx, 0.0, 16.0)
        ty = _clamp(ty, 0.0, 16.0)
        tz = _clamp(tz, 0.0, 16.0)

        fr, to = _normalize_fr_to((fx, fy, fz), (tx, ty, tz))
        if (to[0] - fr[0]) <= 1e-6 or (to[1] - fr[1]) <= 1e-6 or (to[2] - fr[2]) <= 1e-6:
            return None

        rot = None
        if cfg.use_rotations and abs(float(rot_angle)) > 1e-9:
            ox = _round_to_step(float(x), step)
            oy = _round_to_step(float(y), step)
            oz = _round_to_step(float(z), step)
            rot = Rotation(axis=str(rot_axis), angle=float(rot_angle), origin=(float(ox), float(oy), float(oz)))
        return Cuboid(fr=fr, to=to, rotation=rot)

    def to_world(tt: float, along: float, p0: float, p1: float) -> tuple[float, float, float]:
        if plane == "XZ":
            return (cx + p0, cy + along, cz + p1)
        if plane == "XY":
            return (cx + p0, cy + p1, cz + along)
        return (cx + along, cy + p0, cz + p1)

    def to_plane(v: tuple[float, float, float]) -> tuple[float, float]:
        if plane == "XZ":
            return (float(v[0]), float(v[2]))
        if plane == "XY":
            return (float(v[0]), float(v[1]))
        return (float(v[1]), float(v[2]))

    def seg_len_plane(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
        ax, ay = to_plane(a)
        bx, by = to_plane(b)
        return math.hypot(float(bx - ax), float(by - ay))

    def point_seg_dist_plane(p: tuple[float, float, float], a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
        px, py = to_plane(p)
        ax, ay = to_plane(a)
        bx, by = to_plane(b)
        abx = float(bx - ax)
        aby = float(by - ay)
        apx = float(px - ax)
        apy = float(py - ay)
        denom = (abx * abx) + (aby * aby)
        if denom <= 1e-12:
            return math.hypot(float(px - ax), float(py - ay))
        t = (apx * abx + apy * aby) / denom
        t = 0.0 if t < 0.0 else 1.0 if t > 1.0 else float(t)
        cx2 = float(ax) + abx * t
        cy2 = float(ay) + aby * t
        return math.hypot(float(px - cx2), float(py - cy2))

    if cfg.chord_direction and segs > 1:
        max_len = float(cfg.max_merge_len)
        max_len = 0.0 if max_len < 0.0 else float(max_len)

        max_step = float(cfg.max_chord_step)
        max_step = 0.0 if max_step < 0.0 else float(max_step)

        merge_max_dev = float(cfg.merge_max_dev)
        merge_max_dev = 0.0 if merge_max_dev < 0.0 else float(merge_max_dev)

        group_start: Optional[tuple[float, float, float]] = None
        group_end: Optional[tuple[float, float, float]] = None
        group_t0: float = 0.0
        group_t1: float = 0.0
        group_width: float = 1.0
        group_thick: float = 1.0
        group_rot_axis: str = "y"
        group_rot_angle: float = 0.0
        group_swap: bool = False

        def flush_group() -> None:
            nonlocal group_start, group_end
            if group_start is None or group_end is None:
                return
            center = ((group_start[0] + group_end[0]) * 0.5, (group_start[1] + group_end[1]) * 0.5, (group_start[2] + group_end[2]) * 0.5)
            tlen2 = seg_len_plane(group_start, group_end)
            el = build_element(center=center, width=group_width, thick=group_thick, tlen=tlen2, rot_axis=group_rot_axis, rot_angle=group_rot_angle, swap=group_swap)
            if el is not None:
                out.append(el)
            group_start = None
            group_end = None

        t_breaks: list[float] = []
        for i in range(segs + 1):
            t_breaks.append(float(i) / float(segs))

        if max_step > 0.0:
            refined: list[float] = [float(t_breaks[0])]
            for i in range(len(t_breaks) - 1):
                ta = float(t_breaks[i])
                tb = float(t_breaks[i + 1])
                _tta, _aa, along_a, p0_a, p1_a = spiral_point(ta)
                _ttb, _ab, along_b, p0_b, p1_b = spiral_point(tb)
                wa = to_world(_tta, along_a, p0_a, p1_a)
                wb = to_world(_ttb, along_b, p0_b, p1_b)
                L = seg_len_plane(wa, wb)
                n = int(max(1.0, math.ceil(float(L) / float(max_step))))
                for k in range(1, n + 1):
                    refined.append(ta + (tb - ta) * (float(k) / float(n)))
            t_breaks = refined

        for i in range(len(t_breaks) - 1):
            t0 = float(t_breaks[i])
            t1 = float(t_breaks[i + 1])
            tm = (t0 + t1) * 0.5

            tt, _a, along, _p0, _p1 = spiral_point(tm)
            width = _lerp(float(cfg.width_outer), float(cfg.width_inner), tt)
            width = max(0.01, float(width))
            thick = max(0.01, float(cfg.axis_thickness))

            _tt0, _a0, along0, p00, p01 = spiral_point(t0)
            _tt1, _a1, along1, p10, p11 = spiral_point(t1)

            if plane == "XZ":
                s0 = (cx + p00, cy + along0, cz + p01)
                s1 = (cx + p10, cy + along1, cz + p11)
                dir_deg = _canonical_direction_deg(math.degrees(math.atan2(float(s1[2] - s0[2]), float(s1[0] - s0[0]))))
                rot_in = -float(dir_deg)
                if cfg.reverse_rotation:
                    rot_in = -float(rot_in)
                swap, reduced = _reduce_to_limited_rotation(rot_in)
                rot_axis = "y"
            elif plane == "XY":
                s0 = (cx + p00, cy + p01, cz + along0)
                s1 = (cx + p10, cy + p11, cz + along1)
                dir_deg = _canonical_direction_deg(math.degrees(math.atan2(float(s1[1] - s0[1]), float(s1[0] - s0[0]))))
                rot_in = float(dir_deg)
                if cfg.reverse_rotation:
                    rot_in = -float(rot_in)
                swap, reduced = _reduce_to_limited_rotation(rot_in)
                rot_axis = "z"
            else:
                s0 = (cx + along0, cy + p00, cz + p01)
                s1 = (cx + along1, cy + p10, cz + p11)
                dir_deg = _canonical_direction_deg(math.degrees(math.atan2(float(s1[2] - s0[2]), float(s1[1] - s0[1]))))
                rot_in = float(dir_deg)
                if cfg.reverse_rotation:
                    rot_in = -float(rot_in)
                swap, reduced = _reduce_to_limited_rotation(rot_in)
                rot_axis = "x"

            rot_angle = _quantize_angle(reduced) if cfg.use_rotations else 0.0

            can_merge = bool(cfg.auto_merge)
            if group_start is None or group_end is None:
                can_merge = False
            else:
                if rot_axis != group_rot_axis or abs(float(rot_angle) - float(group_rot_angle)) > 1e-9 or bool(swap) != bool(group_swap):
                    can_merge = False

            if can_merge and merge_max_dev > 0.0:
                tm2 = (float(group_t0) + float(t1)) * 0.5
                _ttm, _am, along_m, p0_m, p1_m = spiral_point(tm2)
                pm = to_world(_ttm, along_m, p0_m, p1_m)
                if point_seg_dist_plane(pm, group_start, s1) > float(merge_max_dev):
                    can_merge = False

            if can_merge and max_len > 0.0:
                if seg_len_plane(group_start, s1) > max_len:
                    can_merge = False

            if not can_merge:
                flush_group()
                group_start = (float(s0[0]), float(s0[1]), float(s0[2]))
                group_end = (float(s1[0]), float(s1[1]), float(s1[2]))
                group_t0 = float(t0)
                group_t1 = float(t1)
                group_width = float(width)
                group_thick = float(thick)
                group_rot_axis = str(rot_axis)
                group_rot_angle = float(rot_angle)
                group_swap = bool(swap)
            else:
                group_end = (float(s1[0]), float(s1[1]), float(s1[2]))
                group_t1 = float(t1)
                group_width = max(float(group_width), float(width))
                group_thick = max(float(group_thick), float(thick))

        flush_group()
    else:
        for i in range(segs):
            tm = 0.0 if segs <= 1 else float(i) / float(segs - 1)
            tt, a, along, p0, p1 = spiral_point(tm)

            width = _lerp(float(cfg.width_outer), float(cfg.width_inner), tt)
            width = max(0.01, float(width))
            tlen = _lerp(float(cfg.tangent_len_outer), float(cfg.tangent_len_inner), tt)
            tlen = max(0.01, float(tlen))
            thick = max(0.01, float(cfg.axis_thickness))

            tangent_rad = a + (math.pi * 0.5)
            dir_deg = _canonical_direction_deg(math.degrees(tangent_rad))

            if plane == "XZ":
                x, y, z = cx + p0, cy + along, cz + p1
                rot_in = -float(dir_deg)
                if cfg.reverse_rotation:
                    rot_in = -float(rot_in)
                swap, reduced = _reduce_to_limited_rotation(rot_in)
                rot_axis = "y"
            elif plane == "XY":
                x, y, z = cx + p0, cy + p1, cz + along
                rot_in = float(dir_deg)
                if cfg.reverse_rotation:
                    rot_in = -float(rot_in)
                swap, reduced = _reduce_to_limited_rotation(rot_in)
                rot_axis = "z"
            else:
                x, y, z = cx + along, cy + p0, cz + p1
                rot_in = float(dir_deg)
                if cfg.reverse_rotation:
                    rot_in = -float(rot_in)
                swap, reduced = _reduce_to_limited_rotation(rot_in)
                rot_axis = "x"

            rot_angle = _quantize_angle(reduced) if cfg.use_rotations else 0.0
            el = build_element(center=(x, y, z), width=width, thick=thick, tlen=tlen, rot_axis=rot_axis, rot_angle=rot_angle, swap=bool(swap))
            if el is not None:
                out.append(el)

    if bool(cfg.greedy_merge):
        out = _greedy_merge_aligned_adjacent(list(out), origin_step=float(cfg.snap_step))

    return out


def generate_double_spiral(cfg: SpiralConfig, *, phase_offset_rad: float = 0.0) -> list[Cuboid]:
    a = generate_spiral(cfg, phase_offset_rad=float(phase_offset_rad) + 0.0)
    b = generate_spiral(cfg, phase_offset_rad=float(phase_offset_rad) + math.pi)

    out: list[Cuboid] = []
    seen: set[tuple] = set()

    def key(c: Cuboid) -> tuple:
        r = c.rotation
        if r is None:
            return (c.fr, c.to, None)
        return (c.fr, c.to, str(r.axis), float(r.angle), r.origin)

    for c in list(a) + list(b):
        k = key(c)
        if k in seen:
            continue
        seen.add(k)
        out.append(c)

    return out


def generate_radial_arrangement_4(
    cfg: SpiralConfig,
    shape: str,
    *,
    radius: float,
    planes: tuple[Plane, Plane, Plane, Plane],
    phase_offsets_deg: tuple[float, float, float, float],
    depth_stagger_step: float = 0.0,
) -> list[Cuboid]:
    base_center = (float(cfg.center[0]), float(cfg.center[1]), float(cfg.center[2]))
    out: list[Cuboid] = []

    r = float(radius)
    positions: list[tuple[float, float, float]] = [
        (r, 0.0, 0.0),
        (0.0, 0.0, r),
        (-r, 0.0, 0.0),
        (0.0, 0.0, -r),
    ]

    for k in range(4):
        phase_rad = math.radians(float(phase_offsets_deg[k]))
        plane_k = str(planes[k])

        cfg2 = SpiralConfig(
            center=base_center,
            plane=plane_k,
            segments=int(cfg.segments),
            turns=float(cfg.turns),
            guide_rotation_deg=float(cfg.guide_rotation_deg),
            radius_outer=float(cfg.radius_outer),
            radius_inner=float(cfg.radius_inner),
            ramp_height=float(cfg.ramp_height),
            width_outer=float(cfg.width_outer),
            width_inner=float(cfg.width_inner),
            tangent_len_outer=float(cfg.tangent_len_outer),
            tangent_len_inner=float(cfg.tangent_len_inner),
            axis_thickness=float(cfg.axis_thickness),
            snap_step=float(cfg.snap_step),
            use_rotations=bool(cfg.use_rotations),
            reverse_rotation=bool(cfg.reverse_rotation),
            chord_direction=bool(cfg.chord_direction),
            auto_merge=bool(cfg.auto_merge),
            greedy_merge=bool(cfg.greedy_merge),
            max_merge_len=float(cfg.max_merge_len),
            max_chord_step=float(cfg.max_chord_step),
            merge_max_dev=float(cfg.merge_max_dev),
        )

        els = generate_shape(cfg2, str(shape), phase_offset_rad=phase_rad)
        els = _apply_depth_stagger(list(els), plane=str(plane_k), step=float(depth_stagger_step))

        delta = positions[int(k)]
        for c in list(els):
            cc = _transform_cuboid_global(c, scale=1.0, offset=delta)
            if cc is not None:
                out.append(cc)

    return out
