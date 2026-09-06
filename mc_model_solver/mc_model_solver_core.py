"""Minecraft Model JSON Solver (Core)

Approximates 3D shapes with Minecraft block-model JSON ``elements``
(axis-aligned cuboids with optional single-axis rotation to maintain
Minecraft 1.20 and below compatibility).

This is the headless, pure-Python core: shape primitives, the iterative
solver, symmetry helpers, and Minecraft model JSON export/formatting.
No PySide6 dependency. Importable and testable independently.

See ``mc_model_solver_ui.py`` for the PySide6 UI.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from typing import Iterable, Literal, Optional

#region DATA
# Type aliases, constants, and frozen dataclasses shared by the solver
# and the UI.


Axis = Literal["x", "y", "z"]

_ALLOWED_ANGLES = (-45.0, -22.5, 22.5, 45.0)


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


@dataclass(frozen=True)
class Rotation:
    axis: Axis
    angle: float
    origin: tuple[float, float, float]


@dataclass(frozen=True)
class Cuboid:
    fr: tuple[float, float, float]
    to: tuple[float, float, float]
    rotation: Optional[Rotation]

    def contains_point(self, pt: tuple[float, float, float]) -> bool:
        x, y, z = pt

        if self.rotation is not None:
            ox, oy, oz = self.rotation.origin
            x -= ox
            y -= oy
            z -= oz

            x, y, z = _rot_inv_xyz(axis=self.rotation.axis, angle_deg=self.rotation.angle, x=x, y=y, z=z)

            x += ox
            y += oy
            z += oz

        fx, fy, fz = self.fr
        tx, ty, tz = self.to
        return (fx <= x <= tx) and (fy <= y <= ty) and (fz <= z <= tz)

    def volume(self) -> float:
        fx, fy, fz = self.fr
        tx, ty, tz = self.to
        return max(0.0, tx - fx) * max(0.0, ty - fy) * max(0.0, tz - fz)


class Shape:
    def is_inside(self, pt: tuple[float, float, float]) -> bool:
        raise NotImplementedError


@dataclass(frozen=True)
class Sphere(Shape):
    center: tuple[float, float, float]
    radius: float

    def is_inside(self, pt: tuple[float, float, float]) -> bool:
        x, y, z = pt
        cx, cy, cz = self.center
        dx = x - cx
        dy = y - cy
        dz = z - cz
        return (dx * dx + dy * dy + dz * dz) <= (self.radius * self.radius)


@dataclass(frozen=True)
class Cone(Shape):
    center_xz: tuple[float, float]
    base_y: float
    height: float
    radius: float

    def is_inside(self, pt: tuple[float, float, float]) -> bool:
        x, y, z = pt
        cx, cz = self.center_xz
        if self.height <= 0.0:
            return False
        if y < self.base_y or y > (self.base_y + self.height):
            return False
        t = (y - self.base_y) / self.height
        r = self.radius * (1.0 - t)
        dx = x - cx
        dz = z - cz
        return (dx * dx + dz * dz) <= (r * r)


@dataclass(frozen=True)
class Pyramid(Shape):
    center_xz: tuple[float, float]
    base_y: float
    height: float
    half_base: float

    def is_inside(self, pt: tuple[float, float, float]) -> bool:
        x, y, z = pt
        cx, cz = self.center_xz
        if self.height <= 0.0:
            return False
        if y < self.base_y or y > (self.base_y + self.height):
            return False
        t = (y - self.base_y) / self.height
        half = self.half_base * (1.0 - t)
        return (abs(x - cx) <= half) and (abs(z - cz) <= half)


@dataclass(frozen=True)
class DoubleHelix(Shape):
    center_xz: tuple[float, float]
    base_y: float
    height: float
    turns: float
    radius_start: float
    radius_end: float
    tube_radius: float

    def is_inside(self, pt: tuple[float, float, float]) -> bool:
        x, y, z = pt
        cx, cz = self.center_xz
        if self.height <= 0.0:
            return False
        if y < self.base_y or y > (self.base_y + self.height):
            return False
        t = (y - self.base_y) / self.height
        r = (self.radius_start * (1.0 - t)) + (self.radius_end * t)
        a = (2.0 * math.pi) * self.turns * t

        x1 = cx + (r * math.cos(a))
        z1 = cz + (r * math.sin(a))
        x2 = cx + (r * math.cos(a + math.pi))
        z2 = cz + (r * math.sin(a + math.pi))

        dx1 = x - x1
        dz1 = z - z1
        dx2 = x - x2
        dz2 = z - z2

        d1 = (dx1 * dx1) + (dz1 * dz1)
        d2 = (dx2 * dx2) + (dz2 * dz2)

        tr = self.tube_radius
        return min(d1, d2) <= (tr * tr)
#endregion


#region HELPERS
# Grid sampling, numeric snapping, and candidate generation utilities.


def _linspace_centers(n: int, a: float, b: float) -> list[float]:
    if n <= 0:
        return []
    step = (b - a) / n
    return [a + (i + 0.5) * step for i in range(n)]


def _sample_points(*, grid: int, bounds_min: float = 0.0, bounds_max: float = 16.0) -> list[tuple[float, float, float]]:
    xs = _linspace_centers(grid, bounds_min, bounds_max)
    ys = _linspace_centers(grid, bounds_min, bounds_max)
    zs = _linspace_centers(grid, bounds_min, bounds_max)
    pts: list[tuple[float, float, float]] = []
    for y in ys:
        for z in zs:
            for x in xs:
                pts.append((x, y, z))
    return pts


def _clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else hi if v > hi else v


def _round_to_step(v: float, step: float) -> float:
    if step <= 0:
        return v
    return round(v / step) * step


def _normalize_fr_to(fr: tuple[float, float, float], to: tuple[float, float, float]) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    fx, fy, fz = fr
    tx, ty, tz = to
    return (min(fx, tx), min(fy, ty), min(fz, tz)), (max(fx, tx), max(fy, ty), max(fz, tz))


def _generate_candidate(
    *,
    rng: random.Random,
    anchor: tuple[float, float, float],
    size_choices: list[tuple[float, float, float]],
    allow_rotations: bool,
    rotations_axes: list[Axis],
    grid_step: float,
    floor_ceil_snap: bool,
) -> Cuboid:
    ax, ay, az = anchor
    sx, sy, sz = rng.choice(size_choices)

    cx = ax + rng.uniform(-sx * 0.25, sx * 0.25)
    cy = ay + rng.uniform(-sy * 0.25, sy * 0.25)
    cz = az + rng.uniform(-sz * 0.25, sz * 0.25)

    def snap_min(v: float) -> float:
        if grid_step <= 0.0:
            return v
        if not floor_ceil_snap:
            return _round_to_step(v, grid_step)
        return math.floor(v / grid_step) * grid_step

    def snap_max(v: float) -> float:
        if grid_step <= 0.0:
            return v
        if not floor_ceil_snap:
            return _round_to_step(v, grid_step)
        return math.ceil(v / grid_step) * grid_step

    fx = snap_min(cx - sx / 2.0)
    fy = snap_min(cy - sy / 2.0)
    fz = snap_min(cz - sz / 2.0)
    tx = snap_max(cx + sx / 2.0)
    ty = snap_max(cy + sy / 2.0)
    tz = snap_max(cz + sz / 2.0)

    fx = _clamp(fx, 0.0, 16.0)
    fy = _clamp(fy, 0.0, 16.0)
    fz = _clamp(fz, 0.0, 16.0)
    tx = _clamp(tx, 0.0, 16.0)
    ty = _clamp(ty, 0.0, 16.0)
    tz = _clamp(tz, 0.0, 16.0)

    fr, to = _normalize_fr_to((fx, fy, fz), (tx, ty, tz))

    rotation: Optional[Rotation] = None

    if allow_rotations and rng.random() < 0.55:
        axis = rng.choice(rotations_axes)
        angle = rng.choice(_ALLOWED_ANGLES)
        if floor_ceil_snap and grid_step > 0.0:
            ox = snap_min(cx)
            oy = snap_min(cy)
            oz = snap_min(cz)
        else:
            ox = _round_to_step(cx, grid_step)
            oy = _round_to_step(cy, grid_step)
            oz = _round_to_step(cz, grid_step)
        rotation = Rotation(axis=axis, angle=angle, origin=(ox, oy, oz))

    return Cuboid(fr=fr, to=to, rotation=rotation)


@dataclass
class SolveConfig:
    grid: int
    max_elements: int
    candidates_per_element: int
    outside_sample_fraction: float
    spill_weight: float
    volume_weight: float
    seed: int
    allow_rotations: bool
    rotation_axes: list[Axis]
    grid_step: float
    high_precision: bool = False
    min_span_2d: float = 0.5
    hollow: bool = False
    radial_symmetry: bool = False
    symmetry_center_xz: tuple[float, float] = (8.0, 8.0)
    symmetry_x: bool = False
    symmetry_y: bool = False
    symmetry_z: bool = False
    symmetry_center: tuple[float, float, float] = (8.0, 8.0, 8.0)
#endregion


#region SOLVE
# The iterative solver: samples a grid, scores candidates by coverage
# and spill, and optionally applies mirror/radial symmetry.


def _surface_indices(*, inside_mask: list[bool], grid: int) -> list[int]:
    if grid <= 1:
        return []

    surface: list[int] = []
    stride_z = grid
    stride_y = grid * grid

    for i, inside in enumerate(inside_mask):
        if not inside:
            continue

        ix = i % grid
        iz = (i // stride_z) % grid
        iy = i // stride_y

        is_surface = False
        if ix > 0 and not inside_mask[i - 1]:
            is_surface = True
        elif ix < grid - 1 and not inside_mask[i + 1]:
            is_surface = True
        elif iz > 0 and not inside_mask[i - stride_z]:
            is_surface = True
        elif iz < grid - 1 and not inside_mask[i + stride_z]:
            is_surface = True
        elif iy > 0 and not inside_mask[i - stride_y]:
            is_surface = True
        elif iy < grid - 1 and not inside_mask[i + stride_y]:
            is_surface = True

        if is_surface:
            surface.append(i)

    return surface


def solve(*, shape: Shape, config: SolveConfig) -> list[Cuboid]:
    rng = random.Random(config.seed)

    pts = _sample_points(grid=config.grid)

    inside_mask = [shape.is_inside(pt) for pt in pts]

    inside_idx: list[int] = [i for i, inside in enumerate(inside_mask) if inside]
    if not inside_idx:
        return []

    outside_idx: list[int] = []
    for i, _pt in enumerate(pts):
        if inside_mask[i]:
            continue
        if rng.random() < config.outside_sample_fraction:
            outside_idx.append(i)

    if config.hollow:
        target_idx = _surface_indices(inside_mask=inside_mask, grid=config.grid)
        if not target_idx:
            target_idx = inside_idx
    else:
        target_idx = inside_idx

    radial = bool(config.radial_symmetry)

    sym_x = bool(config.symmetry_x)
    sym_y = bool(config.symmetry_y)
    sym_z = bool(config.symmetry_z)

    center = tuple(config.symmetry_center)

    if radial:
        cx, cy, cz = center

        def in_sector(pt: tuple[float, float, float]) -> bool:
            dx = pt[0] - cx
            dy = pt[1] - cy
            dz = pt[2] - cz
            if dz < 0.0:
                return False
            if dz < abs(dx):
                return False
            if dz < abs(dy):
                return False
            return True

        inside_idx = [i for i in inside_idx if in_sector(pts[i])]
        outside_idx = [i for i in outside_idx if in_sector(pts[i])]
        target_idx = [i for i in target_idx if in_sector(pts[i])]

        if not target_idx:
            return []

    elif sym_x or sym_y or sym_z:
        sx, sy, sz = center

        def in_domain(pt: tuple[float, float, float]) -> bool:
            if sym_x and pt[0] < sx:
                return False
            if sym_y and pt[1] < sy:
                return False
            if sym_z and pt[2] < sz:
                return False
            return True

        inside_idx = [i for i in inside_idx if in_domain(pts[i])]
        outside_idx = [i for i in outside_idx if in_domain(pts[i])]
        target_idx = [i for i in target_idx if in_domain(pts[i])]

        if not target_idx:
            return []

    is_target = [False] * len(pts)
    for i in target_idx:
        is_target[i] = True

    covered_target = [False] * len(pts)

    elements: list[Cuboid] = []

    size_choices: list[tuple[float, float, float]] = []
    base = [16.0, 12.0, 10.0, 8.0, 6.0, 4.0, 3.0, 2.0, 1.5, 1.0]
    if config.high_precision:
        thin = [0.5, 0.25, 0.125, 0.06, 0.04, 0.02, 0.01]
        min_size = float(config.grid_step)

        if config.hollow:
            min_span = max(float(config.min_span_2d), float(min_size))
            spans = [16.0, 12.0, 10.0, 8.0, 6.0, 4.0, 3.0, 2.0, 1.5, 1.0, 0.5, 0.25, 0.125]
            spans = [s for s in spans if s >= min_span]
            thicks = [t for t in thin if t >= min_size]
            for t in thicks:
                for a in spans:
                    for b in spans:
                        size_choices.append((t, a, b))
                        size_choices.append((a, t, b))
                        size_choices.append((a, b, t))
        else:
            base2 = base + thin + thin + thin
            for sx in base2:
                for sy in base2:
                    for sz in base2:
                        if sx >= min_size and sy >= min_size and sz >= min_size:
                            size_choices.append((sx, sy, sz))
    else:
        min_size = 1.0
        for sx in base:
            for sy in base:
                for sz in base:
                    if sx >= min_size and sy >= min_size and sz >= min_size:
                        size_choices.append((sx, sy, sz))

    def score_candidate(cub: Cuboid) -> tuple[float, int, int]:
        gain = 0
        spill = 0
        interior_fill = 0

        fx, fy, fz = cub.fr
        tx, ty, tz = cub.to
        dx = max(0.0, tx - fx)
        dy = max(0.0, ty - fy)
        dz = max(0.0, tz - fz)

        if config.high_precision:
            min_span = float(config.min_span_2d)
            if min_span > 0.0:
                span_count = (1 if dx >= min_span else 0) + (1 if dy >= min_span else 0) + (1 if dz >= min_span else 0)
                if span_count < 2:
                    return -1e9, 0, 0

        for i in target_idx:
            if covered_target[i]:
                continue
            if cub.contains_point(pts[i]):
                gain += 1

        if gain == 0:
            return -1e9, gain, spill

        for i in outside_idx:
            if cub.contains_point(pts[i]):
                spill += 1

        if config.hollow:
            for i in inside_idx:
                if is_target[i]:
                    continue
                if cub.contains_point(pts[i]):
                    interior_fill += 1

        score = (
            float(gain)
            - (config.spill_weight * float(spill))
            - (config.volume_weight * cub.volume())
            - ((config.spill_weight * 0.35) * float(interior_fill) if config.hollow else 0.0)
        )

        if config.high_precision and config.hollow:
            thickness = min(dx, dy, dz)
            score -= (config.spill_weight * 2.0) * thickness

        return score, gain, spill

    def pick_anchor() -> tuple[float, float, float]:
        for _ in range(64):
            i = rng.choice(target_idx)
            if not covered_target[i]:
                return pts[i]
        for i in target_idx:
            if not covered_target[i]:
                return pts[i]
        return pts[target_idx[0]]

    for _ in range(config.max_elements):
        anchor = pick_anchor()

        best: Optional[Cuboid] = None
        best_score = -1e18
        best_gain = 0

        for _k in range(config.candidates_per_element):
            cub = _generate_candidate(
                rng=rng,
                anchor=anchor,
                size_choices=size_choices,
                allow_rotations=config.allow_rotations,
                rotations_axes=config.rotation_axes,
                grid_step=config.grid_step,
                floor_ceil_snap=bool(config.high_precision),
            )
            sc, gain, _spill = score_candidate(cub)
            if sc > best_score:
                best_score = sc
                best = cub
                best_gain = gain

        if best is None or best_score <= 0.0 or best_gain <= 0:
            break

        elements.append(best)

        for i in target_idx:
            if covered_target[i]:
                continue
            if best.contains_point(pts[i]):
                covered_target[i] = True

        if all(covered_target[i] for i in target_idx):
            break

    if radial:
        return _apply_radial_symmetry(elements=elements, center=center)

    if sym_x or sym_y or sym_z:
        return _apply_symmetry(elements=elements, center=center, sym_x=sym_x, sym_y=sym_y, sym_z=sym_z)

    return elements
#endregion


#region SYMMETRY
# Mirror and radial symmetry: duplicate elements across symmetry planes
# or 6-fold radial sectors, deduplicating by cuboid key.


def _mirror_cuboid(
    cub: Cuboid,
    *,
    center: tuple[float, float, float],
    mirror_x: bool,
    mirror_y: bool,
    mirror_z: bool,
) -> Cuboid:
    cx, cy, cz = center

    def mx(x: float) -> float:
        return (2.0 * cx) - x if mirror_x else x

    def my(y: float) -> float:
        return (2.0 * cy) - y if mirror_y else y

    def mz(z: float) -> float:
        return (2.0 * cz) - z if mirror_z else z

    frx, fry, frz = cub.fr
    tox, toy, toz = cub.to
    fr, to = _normalize_fr_to((mx(frx), my(fry), mz(frz)), (mx(tox), my(toy), mz(toz)))

    rot = cub.rotation
    if rot is None:
        return Cuboid(fr=fr, to=to, rotation=None)

    ox, oy, oz = rot.origin
    new_origin = (mx(float(ox)), my(float(oy)), mz(float(oz)))
    new_angle = float(rot.angle)

    if mirror_x and rot.axis != "x":
        new_angle = -new_angle
    if mirror_y and rot.axis != "y":
        new_angle = -new_angle
    if mirror_z and rot.axis != "z":
        new_angle = -new_angle
    if new_angle == -0.0:
        new_angle = 0.0

    return Cuboid(fr=fr, to=to, rotation=Rotation(axis=rot.axis, angle=new_angle, origin=new_origin))


def _apply_symmetry(
    *,
    elements: list[Cuboid],
    center: tuple[float, float, float],
    sym_x: bool,
    sym_y: bool,
    sym_z: bool,
) -> list[Cuboid]:
    out: list[Cuboid] = []
    seen: set[tuple] = set()

    def key(cub: Cuboid) -> tuple:
        rot = cub.rotation
        if rot is None:
            return (cub.fr, cub.to, None)
        return (cub.fr, cub.to, rot.axis, float(rot.angle), rot.origin)

    xs = (False, True) if sym_x else (False,)
    ys = (False, True) if sym_y else (False,)
    zs = (False, True) if sym_z else (False,)

    for elem in elements:
        for mirror_x in xs:
            for mirror_y in ys:
                for mirror_z in zs:
                    cub = _mirror_cuboid(elem, center=center, mirror_x=mirror_x, mirror_y=mirror_y, mirror_z=mirror_z)
                    cub_key = key(cub)
                    if cub_key in seen:
                        continue
                    seen.add(cub_key)
                    out.append(cub)

    return out


def _transform_point_rot_y90(pt: tuple[float, float, float], center: tuple[float, float, float]) -> tuple[float, float, float]:
    cx, cy, cz = center
    dx = pt[0] - cx
    dy = pt[1] - cy
    dz = pt[2] - cz
    return (cx + dz, cy + dy, cz - dx)


def _transform_point_rot_y180(pt: tuple[float, float, float], center: tuple[float, float, float]) -> tuple[float, float, float]:
    cx, cy, cz = center
    dx = pt[0] - cx
    dy = pt[1] - cy
    dz = pt[2] - cz
    return (cx - dx, cy + dy, cz - dz)


def _transform_point_rot_y270(pt: tuple[float, float, float], center: tuple[float, float, float]) -> tuple[float, float, float]:
    cx, cy, cz = center
    dx = pt[0] - cx
    dy = pt[1] - cy
    dz = pt[2] - cz
    return (cx - dz, cy + dy, cz + dx)


def _transform_point_rot_x90(pt: tuple[float, float, float], center: tuple[float, float, float]) -> tuple[float, float, float]:
    cx, cy, cz = center
    dx = pt[0] - cx
    dy = pt[1] - cy
    dz = pt[2] - cz
    return (cx + dx, cy - dz, cz + dy)


def _transform_point_rot_x270(pt: tuple[float, float, float], center: tuple[float, float, float]) -> tuple[float, float, float]:
    cx, cy, cz = center
    dx = pt[0] - cx
    dy = pt[1] - cy
    dz = pt[2] - cz
    return (cx + dx, cy + dz, cz - dy)


def _apply_radial_symmetry(*, elements: list[Cuboid], center: tuple[float, float, float]) -> list[Cuboid]:
    transforms: list[tuple] = [
        (
            lambda pt: pt,
            {"x": ("x", 1.0), "y": ("y", 1.0), "z": ("z", 1.0)},
        ),
        (
            lambda pt, c=center: _transform_point_rot_y90(pt, c),
            {"x": ("z", -1.0), "y": ("y", 1.0), "z": ("x", 1.0)},
        ),
        (
            lambda pt, c=center: _transform_point_rot_y180(pt, c),
            {"x": ("x", -1.0), "y": ("y", 1.0), "z": ("z", -1.0)},
        ),
        (
            lambda pt, c=center: _transform_point_rot_y270(pt, c),
            {"x": ("z", 1.0), "y": ("y", 1.0), "z": ("x", -1.0)},
        ),
        (
            lambda pt, c=center: _transform_point_rot_x90(pt, c),
            {"x": ("x", 1.0), "y": ("z", 1.0), "z": ("y", -1.0)},
        ),
        (
            lambda pt, c=center: _transform_point_rot_x270(pt, c),
            {"x": ("x", 1.0), "y": ("z", -1.0), "z": ("y", 1.0)},
        ),
    ]

    out: list[Cuboid] = []
    seen: set[tuple] = set()

    def key(cub: Cuboid) -> tuple:
        rot = cub.rotation
        if rot is None:
            return (cub.fr, cub.to, None)
        return (cub.fr, cub.to, rot.axis, float(rot.angle), rot.origin)

    for elem in elements:
        for point_fn, axis_map in transforms:
            fr_pt = point_fn(elem.fr)
            to_pt = point_fn(elem.to)
            fr, to = _normalize_fr_to(fr_pt, to_pt)

            rot = elem.rotation
            if rot is None:
                cub = Cuboid(fr=fr, to=to, rotation=None)
            else:
                new_axis, sign = axis_map[str(rot.axis)]
                new_angle = float(rot.angle) * float(sign)
                if new_angle == -0.0:
                    new_angle = 0.0
                new_origin = point_fn((float(rot.origin[0]), float(rot.origin[1]), float(rot.origin[2])))
                cub = Cuboid(fr=fr, to=to, rotation=Rotation(axis=str(new_axis), angle=new_angle, origin=new_origin))

            cub_key = key(cub)
            if cub_key in seen:
                continue
            seen.add(cub_key)
            out.append(cub)

    return out
#endregion


#region BUILD
# Construct Shape instances from UI-style parameter dicts and derive
# symmetry centers for each target type.


def _build_shape(
    *,
    target: str,
    sphere_center: tuple[float, float, float],
    sphere_radius: float,
    cone_center_xz: tuple[float, float],
    cone_base_y: float,
    cone_height: float,
    cone_radius: float,
    pyramid_center_xz: tuple[float, float],
    pyramid_base_y: float,
    pyramid_height: float,
    pyramid_half_base: float,
    helix_center_xz: tuple[float, float],
    helix_base_y: float,
    helix_height: float,
    helix_turns: float,
    helix_radius_start: float,
    helix_radius_end: float,
    helix_tube_radius: float,
) -> Shape:
    """Construct the target Shape from UI parameters."""
    if target == "sphere":
        return Sphere(center=sphere_center, radius=float(sphere_radius))
    if target == "cone":
        return Cone(
            center_xz=cone_center_xz,
            base_y=float(cone_base_y),
            height=float(cone_height),
            radius=float(cone_radius),
        )
    if target == "pyramid":
        return Pyramid(
            center_xz=pyramid_center_xz,
            base_y=float(pyramid_base_y),
            height=float(pyramid_height),
            half_base=float(pyramid_half_base),
        )
    if target == "double_helix":
        return DoubleHelix(
            center_xz=helix_center_xz,
            base_y=float(helix_base_y),
            height=float(helix_height),
            turns=float(helix_turns),
            radius_start=float(helix_radius_start),
            radius_end=float(helix_radius_end),
            tube_radius=float(helix_tube_radius),
        )
    raise ValueError(f"Unknown target: {target}")


def _shape_symmetry_centers(
    *,
    target: str,
    sphere_center: tuple[float, float, float],
    cone_center_xz: tuple[float, float],
    cone_base_y: float,
    cone_height: float,
    pyramid_center_xz: tuple[float, float],
    pyramid_base_y: float,
    pyramid_height: float,
    helix_center_xz: tuple[float, float],
    helix_base_y: float,
    helix_height: float,
) -> tuple[tuple[float, float], tuple[float, float, float]]:
    """Return (symmetry_center_xz, symmetry_center) for the given target."""
    if target == "sphere":
        cxz = (float(sphere_center[0]), float(sphere_center[2]))
        return cxz, (float(sphere_center[0]), float(sphere_center[1]), float(sphere_center[2]))
    if target == "cone":
        cxz = (float(cone_center_xz[0]), float(cone_center_xz[1]))
        return cxz, (float(cone_center_xz[0]), float(cone_base_y) + float(cone_height) * 0.5, float(cone_center_xz[1]))
    if target == "pyramid":
        cxz = (float(pyramid_center_xz[0]), float(pyramid_center_xz[1]))
        return cxz, (float(pyramid_center_xz[0]), float(pyramid_base_y) + float(pyramid_height) * 0.5, float(pyramid_center_xz[1]))
    if target == "double_helix":
        cxz = (float(helix_center_xz[0]), float(helix_center_xz[1]))
        return cxz, (float(helix_center_xz[0]), float(helix_base_y) + float(helix_height) * 0.5, float(helix_center_xz[1]))
    return (8.0, 8.0), (8.0, 8.0, 8.0)
#endregion


#region EXPORT
# Convert Cuboid elements into Minecraft block-model JSON element dicts,
# assemble the full model dict, and pretty-print with inline short arrays.


def _cuboid_to_minecraft_element(cub: Cuboid, texture_key: str) -> dict:
    el: dict = {
        "from": [cub.fr[0], cub.fr[1], cub.fr[2]],
        "to": [cub.to[0], cub.to[1], cub.to[2]],
        "faces": {
            "north": {"texture": texture_key},
            "east": {"texture": texture_key},
            "south": {"texture": texture_key},
            "west": {"texture": texture_key},
            "up": {"texture": texture_key},
            "down": {"texture": texture_key},
        },
    }

    if cub.rotation is not None and cub.rotation.angle != 0.0:
        el["rotation"] = {
            "origin": [cub.rotation.origin[0], cub.rotation.origin[1], cub.rotation.origin[2]],
            "axis": cub.rotation.axis,
            "angle": cub.rotation.angle,
        }

    return el


def export_minecraft_model(
    *,
    elements: Iterable[Cuboid],
    texture: str,
    particle: Optional[str] = None,
) -> dict:
    if particle is None:
        particle = texture

    return {
        "textures": {
            "0": texture,
            "particle": particle,
        },
        "elements": [_cuboid_to_minecraft_element(cub, "#0") for cub in elements],
    }


def format_minecraft_model_json(model: dict) -> str:
    def is_scalar(v) -> bool:
        return v is None or isinstance(v, (str, int, float, bool))

    def is_faces_dict(d: dict) -> bool:
        if not isinstance(d, dict):
            return False
        if not d:
            return False
        for k, v in d.items():
            if not isinstance(k, str):
                return False
            if not isinstance(v, dict):
                return False
            if set(v.keys()) != {"texture"}:
                return False
            if not is_scalar(v.get("texture")):
                return False
        return True

    def try_inline(obj) -> Optional[str]:
        if isinstance(obj, list):
            if all(is_scalar(x) for x in obj) and len(obj) <= 16:
                s = json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
                if len(s) <= 80:
                    return s
            return None

        if isinstance(obj, dict):
            if is_faces_dict(obj):
                s = json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
                if len(s) <= 80:
                    return s
                return None

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
                parts.append(f"{ind2}{ks}: {vs}")
            return "{\n" + ",\n".join(parts) + f"\n{ind}}}"

        return json.dumps(obj, ensure_ascii=False)

    return fmt(model, 0)
#endregion
