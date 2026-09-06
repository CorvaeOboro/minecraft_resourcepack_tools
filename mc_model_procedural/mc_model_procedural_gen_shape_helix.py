"""Double-helix section generator for procedural Minecraft models.

Generates two intertwined helical strands as a list of Cuboid elements,
with optional chord-aligned rotation, auto-merge of collinear segments,
and a 4-way radial arrangement mode. This module is headless (no PySide6
dependency).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

# region IMPORTS
from mc_model_procedural_gen_common import (
    Plane,
    _apply_depth_stagger,
    _canonical_direction_deg,
    _greedy_merge_aligned_adjacent,
    _normalize_fr_to,
    _quantize_angle,
    _reduce_to_limited_rotation,
    _round_to_step,
    _snap_max,
    _snap_min,
    _transform_cuboid_global,
)

try:
    from mc_model_solver_core import Cuboid, Rotation
except Exception as exc:
    print(f"Missing dependency: mc_model_solver_core.py ({exc})")
    raise
# endregion


# region DATA
@dataclass(frozen=True)
class HelixSectionConfig:
    center: tuple[float, float, float]
    axis_plane: str
    heading_deg: float
    segments: int
    length: float
    turns: float
    helix_radius: float
    phase_deg: float
    segment_len: float
    strand_width: float
    strand_depth: float
    scale_x: float
    scale_y: float
    scale_z: float
    snap_step: float
    use_rotations: bool
    chord_align_rotation: bool
    chord_axis_only: bool
    rotation_axis_override: str
    chord_align_rotation_offset_deg: float
    auto_merge: bool
    greedy_merge: bool
    max_merge_len: float
    target_chord_len: float
    merge_max_dev: float


HELIX_UI_SPECS: dict[str, dict] = {
    "axis_plane": {
        "type": "combo",
        "items": [
            ("XZ (rotate about Y)", "XZ"),
            ("XZ (rotate about Y) mirrored", "XZ:mirror"),
            ("XY (rotate about Z)", "XY"),
            ("XY (rotate about Z) mirrored", "XY:mirror"),
            ("YZ (rotate about X)", "YZ"),
            ("YZ (rotate about X) mirrored", "YZ:mirror"),
        ],
        "current": 0,
    },
    "heading": {"type": "double", "min": -180.0, "max": 180.0, "value": 0.0, "decimals": 3},
    "length": {"type": "double", "min": 0.01, "max": 256.0, "value": 16.0, "decimals": 3},
    "radius": {"type": "double", "min": 0.0, "max": 64.0, "value": 3.0, "decimals": 3},
    "phase": {"type": "double", "min": -360.0, "max": 360.0, "value": 0.0, "decimals": 3},
    "segment_len": {"type": "double", "min": 0.01, "max": 64.0, "value": 1.0, "decimals": 3},
    "strand_width": {"type": "double", "min": 0.01, "max": 64.0, "value": 1.0, "decimals": 3},
    "strand_depth": {"type": "double", "min": 0.01, "max": 64.0, "value": 1.0, "decimals": 3},
    "scale_x": {"type": "double", "min": 0.01, "max": 64.0, "value": 1.0, "decimals": 4},
    "scale_y": {"type": "double", "min": 0.01, "max": 64.0, "value": 1.0, "decimals": 4},
    "scale_z": {"type": "double", "min": 0.01, "max": 64.0, "value": 1.0, "decimals": 4},
    "segments": {"type": "spin", "min": 1, "max": 2048, "value": 128},
    "turns": {"type": "double", "min": 0.0, "max": 100.0, "value": 2.5, "decimals": 3},
    "snap_step": {"type": "double", "min": 0.0, "max": 2.0, "value": 0.25, "decimals": 3},
    "use_rotations": {"type": "check", "text": "Use limited rotations", "checked": True},
    "chord_align_rotation": {"type": "check", "text": "Chord-align rotation", "checked": False},
    "chord_axis_only": {"type": "check", "text": "Axis-only chord (prefer heading)", "checked": False},
    "rotation_axis_override": {
        "type": "combo",
        "items": [("Auto", "auto"), ("Rotate about X", "x"), ("Rotate about Y", "y"), ("Rotate about Z", "z")],
        "current": 0,
    },
    "chord_align_rotation_offset": {"type": "double", "min": -180.0, "max": 180.0, "value": 0.0, "decimals": 3},
    "auto_merge": {"type": "check", "text": "Auto-merge straight", "checked": True},
    "greedy_merge": {"type": "check", "text": "Greedy merge pass", "checked": False},
    "max_merge_len": {"type": "double", "min": 0.0, "max": 256.0, "value": 6.0, "decimals": 3},
    "target_chord_len": {"type": "double", "min": 0.0, "max": 256.0, "value": 0.75, "decimals": 3},
    "merge_max_dev": {"type": "double", "min": 0.0, "max": 64.0, "value": 0.15, "decimals": 3},
}
# endregion


# region RADIAL
def generate_radial_arrangement_4_helix(
    cfg: HelixSectionConfig,
    *,
    radius: float,
    planes: tuple[Plane, Plane, Plane, Plane],
    phase_offsets_deg: tuple[float, float, float, float],
    heading_offsets_deg: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0),
    depth_stagger_step: float = 0.0,
) -> list[Cuboid]:
    base_center = (float(cfg.center[0]), float(cfg.center[1]), float(cfg.center[2]))
    out: list[Cuboid] = []

    radius_val = float(radius)
    positions: list[tuple[float, float, float]] = [
        (radius_val, 0.0, 0.0),
        (0.0, 0.0, radius_val),
        (-radius_val, 0.0, 0.0),
        (0.0, 0.0, -radius_val),
    ]

    for side_idx in range(4):
        phase_rad = math.radians(float(phase_offsets_deg[side_idx]))
        plane_k = str(planes[side_idx])
        heading_k = float(cfg.heading_deg) + float(heading_offsets_deg[side_idx])

        cfg2 = HelixSectionConfig(
            center=base_center,
            axis_plane=plane_k,
            heading_deg=float(heading_k),
            segments=int(cfg.segments),
            length=float(cfg.length),
            turns=float(cfg.turns),
            helix_radius=float(cfg.helix_radius),
            phase_deg=float(cfg.phase_deg),
            segment_len=float(cfg.segment_len),
            strand_width=float(cfg.strand_width),
            strand_depth=float(cfg.strand_depth),
            scale_x=float(cfg.scale_x),
            scale_y=float(cfg.scale_y),
            scale_z=float(cfg.scale_z),
            snap_step=float(cfg.snap_step),
            use_rotations=bool(cfg.use_rotations),
            chord_align_rotation=bool(cfg.chord_align_rotation),
            chord_axis_only=bool(cfg.chord_axis_only),
            rotation_axis_override=str(cfg.rotation_axis_override),
            chord_align_rotation_offset_deg=float(cfg.chord_align_rotation_offset_deg),
            auto_merge=bool(cfg.auto_merge),
            greedy_merge=bool(cfg.greedy_merge),
            max_merge_len=float(cfg.max_merge_len),
            target_chord_len=float(cfg.target_chord_len),
            merge_max_dev=float(cfg.merge_max_dev),
        )

        els = generate_double_helix_section(cfg2, phase_offset_rad=phase_rad)
        els = _apply_depth_stagger(list(els), plane=str(plane_k), step=float(depth_stagger_step))

        delta = positions[int(side_idx)]
        for cub in list(els):
            cub_xf = _transform_cuboid_global(cub, scale=1.0, offset=delta)
            if cub_xf is not None:
                out.append(cub_xf)

    return out


# endregion
# region GENERATE
def generate_double_helix_section(cfg: HelixSectionConfig, *, phase_offset_rad: float = 0.0) -> list[Cuboid]:
    segs = max(1, int(cfg.segments))

    plane_raw = str(cfg.axis_plane)
    parts = [part.strip() for part in plane_raw.split(":") if part.strip()]
    plane = str(parts[0] if len(parts) > 0 else "XZ")
    if plane not in ("XZ", "XY", "YZ"):
        plane = "XZ"
    mirror = bool(any(part.lower() == "mirror" for part in parts[1:]))
    handed = -1.0 if mirror else 1.0

    rot_offset = float(cfg.chord_align_rotation_offset_deg)

    def _rot_axis_for_plane(plane_str: str) -> str:
        if plane_str == "XZ":
            return "y"
        if plane_str == "XY":
            return "z"
        return "x"

    def _rot_from_delta_axis(dx: float, dy: float, dz: float, *, axis: str) -> tuple[bool, float]:
        axis_str = str(axis)
        if axis_str == "y":
            dir_deg = _canonical_direction_deg(math.degrees(math.atan2(float(dz), float(dx))))
            return _reduce_to_limited_rotation(-float(dir_deg) + float(rot_offset))
        if axis_str == "z":
            dir_deg = _canonical_direction_deg(math.degrees(math.atan2(float(dy), float(dx))))
            return _reduce_to_limited_rotation(float(dir_deg) + float(rot_offset))
        dir_deg = _canonical_direction_deg(math.degrees(math.atan2(float(dz), float(dy))))
        return _reduce_to_limited_rotation(float(dir_deg) + float(rot_offset))

    def build_element(*, center: tuple[float, float, float], width: float, thick: float, tlen: float, rot_axis: str, rot_angle: float, swap: bool) -> Optional[Cuboid]:
        x, y, z = (float(center[0]), float(center[1]), float(center[2]))
        width = max(0.01, float(width))
        thick = max(0.01, float(thick))
        tlen = max(0.01, float(tlen))

        rot_axis_str = str(rot_axis).strip().lower()
        if rot_axis_str == "y":
            sx, sy, sz = (width, thick, tlen) if swap else (tlen, thick, width)
        elif rot_axis_str == "z":
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

    heading = float(cfg.heading_deg)
    if plane == "XZ":
        hrad = math.radians(heading)
        axis_dir = (math.cos(hrad), 0.0, math.sin(hrad))
        basis1 = (-math.sin(hrad), 0.0, math.cos(hrad))
        basis2 = (0.0, 1.0, 0.0)
    elif plane == "XY":
        hrad = math.radians(heading)
        axis_dir = (math.cos(hrad), math.sin(hrad), 0.0)
        basis1 = (-math.sin(hrad), math.cos(hrad), 0.0)
        basis2 = (0.0, 0.0, 1.0)
    else:
        hrad = math.radians(heading)
        axis_dir = (0.0, math.cos(hrad), math.sin(hrad))
        basis1 = (0.0, -math.sin(hrad), math.cos(hrad))
        basis2 = (1.0, 0.0, 0.0)

    rot_axis_default = _rot_axis_for_plane(str(plane))
    axis_override = str(cfg.rotation_axis_override or "auto").strip().lower()
    rot_axis = axis_override if axis_override in ("x", "y", "z") else str(rot_axis_default)

    rot_plane = "XZ" if str(rot_axis) == "y" else "XY" if str(rot_axis) == "z" else "YZ"

    swap_fixed, reduced_fixed = _rot_from_delta_axis(float(axis_dir[0]), float(axis_dir[1]), float(axis_dir[2]), axis=str(rot_axis))
    rot_angle_fixed = _quantize_angle(reduced_fixed) if cfg.use_rotations else 0.0

    cx, cy, cz = (float(cfg.center[0]), float(cfg.center[1]), float(cfg.center[2]))
    scx = float(cfg.scale_x)
    scy = float(cfg.scale_y)
    scz = float(cfg.scale_z)
    length = float(cfg.length)
    turns = float(cfg.turns)
    radius = float(cfg.helix_radius)
    phase0 = math.radians(float(cfg.phase_deg)) + float(phase_offset_rad)

    out: list[Cuboid] = []

    def point_at(t_param: float, angle_rad: float) -> tuple[float, float, float]:
        along = (float(t_param) * length) - (length * 0.5)
        ox = float(axis_dir[0]) * along
        oy = float(axis_dir[1]) * along
        oz = float(axis_dir[2]) * along
        rx = (float(basis1[0]) * math.cos(angle_rad) + float(basis2[0]) * math.sin(angle_rad)) * radius
        ry = (float(basis1[1]) * math.cos(angle_rad) + float(basis2[1]) * math.sin(angle_rad)) * radius
        rz = (float(basis1[2]) * math.cos(angle_rad) + float(basis2[2]) * math.sin(angle_rad)) * radius
        dx = float(ox + rx)
        dy = float(oy + ry)
        dz = float(oz + rz)
        return (cx + dx * scx, cy + dy * scy, cz + dz * scz)

    def rot_from_delta(dx: float, dy: float, dz: float) -> tuple[bool, float]:
        return _rot_from_delta_axis(float(dx), float(dy), float(dz), axis=str(rot_axis))

    use_chord = bool(cfg.chord_align_rotation)
    axis_only = bool(cfg.chord_axis_only)

    def to_plane(pt: tuple[float, float, float]) -> tuple[float, float]:
        if rot_plane == "XZ":
            return (float(pt[0]), float(pt[2]))
        if rot_plane == "XY":
            return (float(pt[0]), float(pt[1]))
        return (float(pt[1]), float(pt[2]))

    def seg_len_plane(pt_a: tuple[float, float, float], pt_b: tuple[float, float, float]) -> float:
        ax, ay = to_plane(pt_a)
        bx, by = to_plane(pt_b)
        return math.hypot(float(bx - ax), float(by - ay))

    def point_seg_dist_plane(pt: tuple[float, float, float], pt_a: tuple[float, float, float], pt_b: tuple[float, float, float]) -> float:
        px, py = to_plane(pt)
        ax, ay = to_plane(pt_a)
        bx, by = to_plane(pt_b)
        abx = float(bx - ax)
        aby = float(by - ay)
        apx = float(px - ax)
        apy = float(py - ay)
        denom = (abx * abx) + (aby * aby)
        if denom <= 1e-12:
            return math.hypot(float(px - ax), float(py - ay))
        t_proj = (apx * abx + apy * aby) / denom
        t_proj = 0.0 if t_proj < 0.0 else 1.0 if t_proj > 1.0 else float(t_proj)
        cx2 = float(ax) + abx * t_proj
        cy2 = float(ay) + aby * t_proj
        return math.hypot(float(px - cx2), float(py - cy2))

    def strand_point(t_param: float, *, strand_phase: float) -> tuple[float, float, float]:
        angle_rad = (2.0 * math.pi) * turns * float(t_param) * float(handed) + phase0 + float(strand_phase)
        return point_at(float(t_param), float(angle_rad))

    def axis_component_len(pt_a: tuple[float, float, float], pt_b: tuple[float, float, float]) -> float:
        dx = float(pt_b[0] - pt_a[0])
        dy = float(pt_b[1] - pt_a[1])
        dz = float(pt_b[2] - pt_a[2])
        adx = float(axis_dir[0]) * scx
        ady = float(axis_dir[1]) * scy
        adz = float(axis_dir[2]) * scz
        norm = math.sqrt(adx * adx + ady * ady + adz * adz)
        if norm <= 1e-12:
            dot_val = float(dx) * float(axis_dir[0]) + float(dy) * float(axis_dir[1]) + float(dz) * float(axis_dir[2])
            return abs(float(dot_val))
        ux = adx / norm
        uy = ady / norm
        uz = adz / norm
        dot_val = float(dx) * ux + float(dy) * uy + float(dz) * uz
        return abs(float(dot_val))

    if not use_chord:
        for i in range(segs):
            t0 = 0.0 if segs <= 1 else float(i) / float(segs - 1)
            a0 = (2.0 * math.pi) * turns * float(t0) + phase0
            pt1 = point_at(t0, a0)
            pt2 = point_at(t0, a0 + math.pi)

            elem1 = build_element(
                center=pt1,
                width=float(cfg.strand_width),
                thick=float(cfg.strand_depth),
                tlen=float(cfg.segment_len),
                rot_axis=str(rot_axis),
                rot_angle=float(rot_angle_fixed),
                swap=bool(swap_fixed),
            )
            if elem1 is not None:
                out.append(elem1)

            elem2 = build_element(
                center=pt2,
                width=float(cfg.strand_width),
                thick=float(cfg.strand_depth),
                tlen=float(cfg.segment_len),
                rot_axis=str(rot_axis),
                rot_angle=float(rot_angle_fixed),
                swap=bool(swap_fixed),
            )
            if elem2 is not None:
                out.append(elem2)

        if bool(cfg.greedy_merge):
            out = _greedy_merge_aligned_adjacent(list(out), origin_step=float(cfg.snap_step))
        return out

    target_len = float(cfg.target_chord_len)
    target_len = 0.0 if target_len < 0.0 else float(target_len)

    t_breaks: list[float] = []
    for i in range(segs + 1):
        t_breaks.append(float(i) / float(segs))

    if target_len > 0.0:
        refined: list[float] = [float(t_breaks[0])]
        for i in range(len(t_breaks) - 1):
            t_start = float(t_breaks[i])
            t_end = float(t_breaks[i + 1])
            pt_a = strand_point(t_start, strand_phase=0.0)
            pt_b = strand_point(t_end, strand_phase=0.0)
            chord_len = axis_component_len(pt_a, pt_b) if axis_only else seg_len_plane(pt_a, pt_b)
            sub_count = int(max(1.0, math.ceil(float(chord_len) / float(target_len))))
            sub_count = int(min(128, sub_count))
            for sub_idx in range(1, sub_count + 1):
                refined.append(t_start + (t_end - t_start) * (float(sub_idx) / float(sub_count)))
        t_breaks = refined

    max_len = float(cfg.max_merge_len)
    max_len = 0.0 if max_len < 0.0 else float(max_len)

    merge_max_dev = float(cfg.merge_max_dev)
    merge_max_dev = 0.0 if merge_max_dev < 0.0 else float(merge_max_dev)

    auto_merge = bool(cfg.auto_merge)

    def build_strand(*, strand_phase: float) -> list[Cuboid]:
        group_start: Optional[tuple[float, float, float]] = None
        group_end: Optional[tuple[float, float, float]] = None
        group_t0: float = 0.0
        group_t1: float = 0.0
        group_rot_angle: float = 0.0
        group_swap: bool = False

        strand_out: list[Cuboid] = []

        def flush_group() -> None:
            nonlocal group_start, group_end
            if group_start is None or group_end is None:
                return
            center = ((group_start[0] + group_end[0]) * 0.5, (group_start[1] + group_end[1]) * 0.5, (group_start[2] + group_end[2]) * 0.5)
            tlen2 = axis_component_len(group_start, group_end) if axis_only else seg_len_plane(group_start, group_end)
            elem = build_element(
                center=center,
                width=float(cfg.strand_width),
                thick=float(cfg.strand_depth),
                tlen=float(tlen2),
                rot_axis=str(rot_axis),
                rot_angle=float(group_rot_angle),
                swap=bool(group_swap),
            )
            if elem is not None:
                strand_out.append(elem)
            group_start = None
            group_end = None

        for i in range(len(t_breaks) - 1):
            t0 = float(t_breaks[i])
            t1 = float(t_breaks[i + 1])
            pt0 = strand_point(t0, strand_phase=float(strand_phase))
            pt1 = strand_point(t1, strand_phase=float(strand_phase))
            dx = float(pt1[0] - pt0[0])
            dy = float(pt1[1] - pt0[1])
            dz = float(pt1[2] - pt0[2])

            if axis_only:
                swap = bool(swap_fixed)
                rot_angle = float(rot_angle_fixed)
            else:
                swap, reduced = rot_from_delta(dx, dy, dz)
                rot_angle = _quantize_angle(reduced) if cfg.use_rotations else 0.0

            can_merge = bool(auto_merge)
            if group_start is None or group_end is None:
                can_merge = False
            else:
                if abs(float(rot_angle) - float(group_rot_angle)) > 1e-9 or bool(swap) != bool(group_swap):
                    can_merge = False

            if can_merge and (not axis_only) and merge_max_dev > 0.0:
                t_mid = (float(group_t0) + float(t1)) * 0.5
                pt_mid = strand_point(t_mid, strand_phase=float(strand_phase))
                if point_seg_dist_plane(pt_mid, group_start, pt1) > float(merge_max_dev):
                    can_merge = False

            if can_merge and max_len > 0.0:
                chord_len_merge = axis_component_len(group_start, pt1) if axis_only else seg_len_plane(group_start, pt1)
                if float(chord_len_merge) > float(max_len):
                    can_merge = False

            if not can_merge:
                flush_group()
                group_start = (float(pt0[0]), float(pt0[1]), float(pt0[2]))
                group_end = (float(pt1[0]), float(pt1[1]), float(pt1[2]))
                group_t0 = float(t0)
                group_t1 = float(t1)
                group_rot_angle = float(rot_angle)
                group_swap = bool(swap)
            else:
                group_end = (float(pt1[0]), float(pt1[1]), float(pt1[2]))
                group_t1 = float(t1)

        flush_group()

        if bool(cfg.greedy_merge):
            strand_out = _greedy_merge_aligned_adjacent(list(strand_out), origin_step=float(cfg.snap_step))
        return strand_out

    strand_a = build_strand(strand_phase=0.0)
    strand_b = build_strand(strand_phase=math.pi)

    out = []
    max_count = max(len(strand_a), len(strand_b))
    for i in range(max_count):
        if i < len(strand_a):
            out.append(strand_a[i])
        if i < len(strand_b):
            out.append(strand_b[i])
    return out
# endregion
