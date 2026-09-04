"""
Minecraft Model UV Packer (Core)

Pure-Python core logic for generating non-overlapping UVs for Minecraft
block-model cuboids. This module contains no PySide6/UI dependencies and
can be imported and tested headlessly.

It performs a connected box-unwrap per cuboid (a single island
containing all 6 faces where possible), then packs islands into a texture atlas
with configurable padding between islands and padding to the atlas border.

This tool writes UVs in *normalized* (0..1) space with high precision floats.
(You can convert to vanilla block-model UV units later by multiplying by 16.)

Expected inputs
- Minecraft model JSON file containing `elements[]`.

Outputs
- Updated model JSON where each element face has a packed `uv` rect.
- Optional per-face `rotation` (90-degree steps) when needed to keep box-net
  edges consistent.

"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Dict, Iterable, Literal, Optional


Face = Literal["north", "south", "east", "west", "up", "down"]
Axis = Literal["x", "y", "z"]


#region HELPERS
# Core math helpers: float rounding, snapping, clamping
# Used by both the pack/unwrap pipeline and the resnap subsystem

def _r12(x: float) -> float:
    return float(f"{float(x):.12f}")


def _r3(x: float) -> float:
    return float(f"{float(x):.3f}")


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def _snap_int(v: float, step: int) -> int:
    step_i = max(1, int(step))
    return int(round(float(v) / float(step_i))) * step_i
#endregion


#region DATA
# Model dataclasses (Rotation, Cuboid, Element) and JSON element parser
# These mirror the Minecraft model JSON structure: from/to/rotation per element


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

    def size(self) -> tuple[float, float, float]:
        fx, fy, fz = self.fr
        tx, ty, tz = self.to
        return (abs(tx - fx), abs(ty - fy), abs(tz - fz))

    def center(self) -> tuple[float, float, float]:
        fx, fy, fz = self.fr
        tx, ty, tz = self.to
        return ((fx + tx) * 0.5, (fy + ty) * 0.5, (fz + tz) * 0.5)


@dataclass
class Element:
    idx: int
    raw: dict
    cuboid: Cuboid


def _parse_element(idx: int, raw: dict) -> Element:
    fr0 = raw.get("from")
    to0 = raw.get("to")
    if not (isinstance(fr0, list) and isinstance(to0, list) and len(fr0) == 3 and len(to0) == 3):
        raise ValueError(f"Invalid element from/to at index {idx}")

    fr = (float(fr0[0]), float(fr0[1]), float(fr0[2]))
    to = (float(to0[0]), float(to0[1]), float(to0[2]))

    fx, fy, fz = fr
    tx, ty, tz = to
    fr = (min(fx, tx), min(fy, ty), min(fz, tz))
    to = (max(fx, tx), max(fy, ty), max(fz, tz))

    rot = None
    r0 = raw.get("rotation")
    if isinstance(r0, dict):
        axis = r0.get("axis")
        ang = r0.get("angle")
        org = r0.get("origin")
        if axis in {"x", "y", "z"} and isinstance(ang, (int, float)) and isinstance(org, list) and len(org) == 3:
            rot = Rotation(
                axis=str(axis),
                angle=float(ang),
                origin=(float(org[0]), float(org[1]), float(org[2])),
                rescale=bool(r0.get("rescale", False)),
            )

    return Element(idx=int(idx), raw=dict(raw), cuboid=Cuboid(fr=fr, to=to, rotation=rot))
#endregion


#region JSONFMT
# Pretty-printer for Minecraft model JSON with inline heuristic for small structures
# Produces readable output with 2-space indentation, inlining short lists/dicts


def format_minecraft_model_json(model: dict) -> str:
    def is_scalar(v) -> bool:
        return v is None or isinstance(v, (str, int, float, bool))

    def try_inline(obj) -> Optional[str]:
        if isinstance(obj, list):
            if all(is_scalar(x) for x in obj) and len(obj) <= 16:
                s = json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
                if len(s) <= 110:
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
                    if isinstance(v, dict) and all(isinstance(kk, str) for kk in v.keys()) and all(is_scalar(vv) for vv in v.values()):
                        continue
                    ok = False
                    break
                if ok:
                    s = json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
                    if len(s) <= 110:
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


#region UVDATA
# UV layout dataclasses: FaceRect (face pixel size), FacePlacement (position+rotation),
# Island (collection of placements with bbox/size helpers)


@dataclass(frozen=True)
class FaceRect:
    w_px: int
    h_px: int


@dataclass
class FacePlacement:
    face: Face
    x_px: int
    y_px: int
    w_px: int
    h_px: int
    rot: int  # 0, 90, 180, 270


@dataclass
class Island:
    element_idx: int
    placements: list[FacePlacement]

    def bbox(self) -> tuple[int, int, int, int]:
        if not self.placements:
            return (0, 0, 0, 0)
        xs = [p.x_px for p in self.placements]
        ys = [p.y_px for p in self.placements]
        x2 = [p.x_px + p.w_px for p in self.placements]
        y2 = [p.y_px + p.h_px for p in self.placements]
        return (min(xs), min(ys), max(x2), max(y2))

    def size(self) -> tuple[int, int]:
        x0, y0, x1, y1 = self.bbox()
        return (max(0, x1 - x0), max(0, y1 - y0))
#endregion


#region BOXGEOM
# Box face geometry: face centers, face normals, and axis-rotation math
# Used by the central-face selection algorithm to score which face is "most outward"


def _face_centers(c: Cuboid) -> Dict[Face, tuple[float, float, float]]:
    fx, fy, fz = c.fr
    tx, ty, tz = c.to
    cx = (fx + tx) * 0.5
    cy = (fy + ty) * 0.5
    cz = (fz + tz) * 0.5

    return {
        "north": (cx, cy, fz),
        "south": (cx, cy, tz),
        "west": (fx, cy, cz),
        "east": (tx, cy, cz),
        "down": (cx, fy, cz),
        "up": (cx, ty, cz),
    }


def _face_normals() -> Dict[Face, tuple[float, float, float]]:
    return {
        "north": (0.0, 0.0, -1.0),
        "south": (0.0, 0.0, 1.0),
        "west": (-1.0, 0.0, 0.0),
        "east": (1.0, 0.0, 0.0),
        "down": (0.0, -1.0, 0.0),
        "up": (0.0, 1.0, 0.0),
    }


def _rotate_about_axis(*, axis: Axis, angle_deg: float, origin: tuple[float, float, float], p: tuple[float, float, float]) -> tuple[float, float, float]:
    if float(angle_deg) == 0.0:
        return p

    ox, oy, oz = origin
    x, y, z = p
    x -= ox
    y -= oy
    z -= oz

    a = math.radians(float(angle_deg))
    c = math.cos(a)
    s = math.sin(a)

    if axis == "x":
        yy = c * y - s * z
        zz = s * y + c * z
        xx = x
    elif axis == "y":
        xx = c * x + s * z
        zz = -s * x + c * z
        yy = y
    else:
        xx = c * x - s * y
        yy = s * x + c * y
        zz = z

    return (xx + ox, yy + oy, zz + oz)


def _rotate_vec_about_axis(*, axis: Axis, angle_deg: float, v: tuple[float, float, float]) -> tuple[float, float, float]:
    if float(angle_deg) == 0.0:
        return v

    x, y, z = v
    a = math.radians(float(angle_deg))
    c = math.cos(a)
    s = math.sin(a)

    if axis == "x":
        return (x, c * y - s * z, s * y + c * z)
    if axis == "y":
        return (c * x + s * z, y, -s * x + c * z)
    return (c * x - s * y, s * x + c * y, z)
#endregion


#region CENTRAL
# Central-face selection: picks the "most outward" face of a cuboid to serve
# as the anchor of the box-net. Scoring is centered on the cuboid's own center
# so it is independent of world position. Optional +0.25 bias for horizontal faces.


def _choose_central_face(*, c: Cuboid, prefer_horizontal: bool) -> Face:
    centers = _face_centers(c)
    normals = _face_normals()
    cuboid_center = c.center()

    rot = c.rotation
    if rot is not None and float(rot.angle) != 0.0:
        for k, p in list(centers.items()):
            centers[k] = _rotate_about_axis(axis=rot.axis, angle_deg=rot.angle, origin=rot.origin, p=p)
        for k, n in list(normals.items()):
            normals[k] = _rotate_vec_about_axis(axis=rot.axis, angle_deg=rot.angle, v=n)
        cuboid_center = _rotate_about_axis(axis=rot.axis, angle_deg=rot.angle, origin=rot.origin, p=cuboid_center)

    ccx, ccy, ccz = cuboid_center

    horiz: tuple[Face, ...] = ("north", "south", "east", "west")
    candidates: Iterable[Face]
    if prefer_horizontal:
        candidates = horiz
    else:
        candidates = ("north", "south", "east", "west", "up", "down")

    best_face: Face = "north"
    best_score = -1e30
    for f in candidates:
        px, py, pz = centers[f]
        nx, ny, nz = normals[f]
        score = ((px - ccx) * nx) + ((py - ccy) * ny) + ((pz - ccz) * nz)
        if f in horiz:
            score += 0.25
        if score > best_score:
            best_score = score
            best_face = f

    return best_face
#endregion


#region BOXNET
# Box-net (connected island) construction:
# - _face_rects_px maps each face to a pixel rect based on cuboid dimensions
# - _rot_wh swaps w/h for 90/270 degree rotations
# - _build_connected_box_island attaches all 6 faces around a central face in a cross net


def _face_rects_px(*, c: Cuboid, tex_w: int, tex_h: int, density: float, snap_px: int) -> Dict[Face, FaceRect]:
    dx, dy, dz = c.size()
    px_per_unit_x = (float(tex_w) / 16.0) * float(density)
    px_per_unit_y = (float(tex_h) / 16.0) * float(density)

    def snap_w(v: float) -> int:
        return max(1, _snap_int(v, snap_px))

    def snap_h(v: float) -> int:
        return max(1, _snap_int(v, snap_px))

    dx_px = snap_w(dx * px_per_unit_x)
    dy_px = snap_h(dy * px_per_unit_y)
    dz_px = snap_w(dz * px_per_unit_x)

    return {
        "north": FaceRect(w_px=dx_px, h_px=dy_px),
        "south": FaceRect(w_px=dx_px, h_px=dy_px),
        "west": FaceRect(w_px=dz_px, h_px=dy_px),
        "east": FaceRect(w_px=dz_px, h_px=dy_px),
        "up": FaceRect(w_px=dx_px, h_px=dz_px),
        "down": FaceRect(w_px=dx_px, h_px=dz_px),
    }


def _rot_wh(*, w: int, h: int, rot: int) -> tuple[int, int]:
    r = int(rot) % 360
    if r in {90, 270}:
        return (h, w)
    return (w, h)


def _build_connected_box_island(*, element_idx: int, face_rects: Dict[Face, FaceRect], central: Face) -> Island:
    c = central

    def rect(f: Face) -> FaceRect:
        return face_rects[f]

    placements: dict[Face, FacePlacement] = {}

    c_w = rect(c).w_px
    c_h = rect(c).h_px
    placements[c] = FacePlacement(face=c, x_px=0, y_px=0, w_px=c_w, h_px=c_h, rot=0)

    net: dict[Face, tuple[Optional[Face], str]] = {}

    if c == "west":
        net = {
            "north": ("west", "left"),
            "south": ("west", "right"),
            "up": ("west", "top"),
            "down": ("west", "bottom"),
            "east": ("south", "right"),
        }
    elif c == "east":
        net = {
            "north": ("east", "right"),
            "south": ("east", "left"),
            "up": ("east", "top"),
            "down": ("east", "bottom"),
            "west": ("south", "left"),
        }
    elif c == "north":
        net = {
            "west": ("north", "left"),
            "east": ("north", "right"),
            "up": ("north", "top"),
            "down": ("north", "bottom"),
            "south": ("east", "right"),
        }
    elif c == "south":
        net = {
            "east": ("south", "left"),
            "west": ("south", "right"),
            "up": ("south", "top"),
            "down": ("south", "bottom"),
            "north": ("west", "right"),
        }
    elif c == "up":
        net = {
            "north": ("up", "top"),
            "south": ("up", "bottom"),
            "west": ("up", "left"),
            "east": ("up", "right"),
            "down": ("south", "bottom"),
        }
    else:
        net = {
            "north": ("down", "top"),
            "south": ("down", "bottom"),
            "west": ("down", "left"),
            "east": ("down", "right"),
            "up": ("south", "bottom"),
        }

    remaining: set[Face] = set(f for f in ("north", "south", "east", "west", "up", "down") if f != c)
    for _pass in range(8):
        progressed = False
        for f in list(remaining):
            parent, side = net.get(f, (None, ""))
            if parent is None or parent not in placements:
                continue

            p = placements[parent]
            w0 = rect(f).w_px
            h0 = rect(f).h_px

            want = p.w_px if side in {"top", "bottom"} else p.h_px

            def rot_for_side(*, side: str, want: int, w0: int, h0: int) -> int:
                # For a shared edge:
                # - attaching on top/bottom means the shared edge is horizontal -> match resulting width
                # - attaching on left/right means the shared edge is vertical   -> match resulting height
                side = str(side)
                want = int(want)
                w0 = int(w0)
                h0 = int(h0)

                if side in {"top", "bottom"}:
                    # width after rot0 is w0; width after rot90 is h0
                    if w0 == want:
                        return 0
                    if h0 == want:
                        return 90
                    return 0 if abs(w0 - want) <= abs(h0 - want) else 90

                # left/right
                # height after rot0 is h0; height after rot90 is w0
                if h0 == want:
                    return 0
                if w0 == want:
                    return 90
                return 0 if abs(h0 - want) <= abs(w0 - want) else 90

            rot = rot_for_side(side=side, want=want, w0=w0, h0=h0)
            w, h = _rot_wh(w=w0, h=h0, rot=rot)

            if side == "left":
                x = p.x_px - w
                y = p.y_px
            elif side == "right":
                x = p.x_px + p.w_px
                y = p.y_px
            elif side == "top":
                x = p.x_px
                y = p.y_px - h
            elif side == "bottom":
                x = p.x_px
                y = p.y_px + p.h_px
            else:
                x, y = 0, 0

            placements[f] = FacePlacement(face=f, x_px=int(x), y_px=int(y), w_px=int(w), h_px=int(h), rot=int(rot))
            remaining.remove(f)
            progressed = True

        if not progressed:
            break

    isl = Island(element_idx=int(element_idx), placements=list(placements.values()))
    x0, y0, x1, y1 = isl.bbox()

    for p in isl.placements:
        p.x_px -= x0
        p.y_px -= y0

    return isl
#endregion


#region PACKER
# Shelf packer: places islands left-to-right in rows, wrapping when a row fills.
# PackedIsland records the atlas position and overflow flag for each island.


@dataclass(frozen=True)
class PackedIsland:
    island: Island
    x_px: int
    y_px: int
    overflow: bool = False


def _pack_islands_shelf(
    *,
    islands: list[Island],
    tex_w: int,
    tex_h: int,
    island_pad_px: int,
    border_pad_px: int,
    snap_px: int,
) -> list[PackedIsland]:
    pad = max(0, int(island_pad_px))
    border = max(0, int(border_pad_px))
    snap = max(1, int(snap_px))

    items: list[tuple[int, int, Island]] = []
    for isl in islands:
        w, h = isl.size()
        w2 = w + pad * 2
        h2 = h + pad * 2
        w2 = int(_snap_int(w2, snap))
        h2 = int(_snap_int(h2, snap))
        items.append((w2, h2, isl))

    items.sort(key=lambda t: (t[1], t[0]), reverse=True)

    x = int(_snap_int(border, snap))
    y = int(_snap_int(border, snap))
    row_h = 0

    packed: list[PackedIsland] = []

    for w2, h2, isl in items:
        if x + w2 > (tex_w - border) and x > border:
            x = int(_snap_int(border, snap))
            y += int(_snap_int(row_h, snap))
            row_h = 0

        overflow = bool(y + h2 > (tex_h - border))
        packed.append(PackedIsland(island=isl, x_px=x + pad, y_px=y + pad, overflow=overflow))

        x += int(_snap_int(w2, snap))
        row_h = max(row_h, h2)

    return packed
#endregion


#region PACKER-FILL
# Secondary packing pass: detects blank space left by the shelf packer and
# relocates groups of similar (volume/proportion-matched) islands into the
# gaps. Prioritizes overflow islands first, then compacts by pulling
# bottom-row islands up into available space. Runs multiple passes so that
# positions vacated by earlier moves become available for later ones.


def _similarity_key(c: Cuboid) -> tuple[int, float, float]:
    """Quantized similarity key based on element volume and proportions."""
    dx, dy, dz = c.size()
    vol = dx * dy * dz
    vol_bucket = int(round(math.log2(max(vol, 0.0625))))
    total = dx + dy + dz
    if total > 1e-6:
        nx = round(dx / total * 4.0) / 4.0
        ny = round(dy / total * 4.0) / 4.0
    else:
        nx = ny = 0.0
    return (vol_bucket, nx, ny)


def _compute_free_rects(
    occupied: list[tuple[int, int, int, int]],
    lo: int,
    hi_x: int,
    hi_y: int,
    min_w: int,
    min_h: int,
) -> list[tuple[int, int, int, int]]:
    """Compute maximal free rectangles within [lo,hi_x] x [lo,hi_y] given occupied rects.

    Uses the standard split-and-prune approach: start with the full interior
    as the only free rect, then for each occupied rect, split every
    overlapping free rect into up to 4 sub-rects.

    NOTE: The split-and-prune algorithm can produce overlapping free rects at
    the corners of occupied rects.  Callers that place items into these rects
    must perform their own overlap checks against previously placed items.
    """
    free: list[tuple[int, int, int, int]] = [(lo, lo, hi_x, hi_y)]

    for (ox0, oy0, ox1, oy1) in occupied:
        ox0 = max(lo, ox0)
        oy0 = max(lo, oy0)
        ox1 = min(hi_x, ox1)
        oy1 = min(hi_y, oy1)
        if ox1 <= ox0 or oy1 <= oy0:
            continue

        new_free: list[tuple[int, int, int, int]] = []
        for (fx0, fy0, fx1, fy1) in free:
            if ox1 <= fx0 or ox0 >= fx1 or oy1 <= fy0 or oy0 >= fy1:
                new_free.append((fx0, fy0, fx1, fy1))
                continue
            if ox0 > fx0:
                new_free.append((fx0, fy0, ox0, fy1))
            if ox1 < fx1:
                new_free.append((ox1, fy0, fx1, fy1))
            if oy0 > fy0:
                new_free.append((fx0, fy0, fx1, oy0))
            if oy1 < fy1:
                new_free.append((fx0, oy1, fx1, fy1))
        free = new_free

    free = [(x0, y0, x1, y1) for (x0, y0, x1, y1) in free
            if (x1 - x0) >= min_w and (y1 - y0) >= min_h]
    return free


def _rects_overlap(
    a: tuple[int, int, int, int],
    b: tuple[int, int, int, int],
    *,
    eps: int = 0,
) -> bool:
    """True if two axis-aligned rects overlap (interior intersection)."""
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ox = min(ax1, bx1) - max(ax0, bx0)
    oy = min(ay1, by1) - max(ay0, by0)
    return ox > eps and oy > eps


def _secondary_fill_pass(
    *,
    packed: list[PackedIsland],
    cuboids: list[Cuboid],
    tex_w: int,
    tex_h: int,
    island_pad_px: int,
    border_pad_px: int,
    snap_px: int,
    max_passes: int = 3,
) -> tuple[list[PackedIsland], int]:
    """Secondary pass: relocate groups of similar islands into blank space.

    After the initial shelf pack, detects free rectangles within the atlas
    and relocates islands into them.  Prioritizes overflow islands, then
    compacts by moving bottom-row islands up.  Groups similar islands
    (by volume and proportions) so siblings stay together.

    Returns (updated_packed_list, total_relocated).
    """
    if not packed or len(packed) < 2:
        return packed, 0

    pad = max(0, int(island_pad_px))
    border = max(0, int(border_pad_px))
    snap = max(1, int(snap_px))

    lo = border
    hi_x = int(tex_w) - border
    hi_y = int(tex_h) - border

    pos: list[list[int]] = []
    for pi in packed:
        pos.append([int(pi.x_px), int(pi.y_px)])

    overflow_flags = [bool(pi.overflow) for pi in packed]
    sizes = [pi.island.size() for pi in packed]

    def sim_key(i: int) -> tuple:
        idx = int(packed[i].island.element_idx)
        if 0 <= idx < len(cuboids):
            return _similarity_key(cuboids[idx])
        return (0, 0.0, 0.0)

    total_relocated = 0

    for _pass in range(max_passes):
        occupied: list[tuple[int, int, int, int]] = []
        for i in range(len(packed)):
            w, h = sizes[i]
            ox0 = pos[i][0] - pad
            oy0 = pos[i][1] - pad
            ox1 = pos[i][0] + w + pad
            oy1 = pos[i][1] + h + pad
            occupied.append((ox0, oy0, ox1, oy1))

        min_fw = min((w + pad * 2 for w, h in sizes), default=hi_x - lo)
        min_fh = min((h + pad * 2 for w, h in sizes), default=hi_y - lo)

        free_rects = _compute_free_rects(occupied, lo, hi_x, hi_y, min_fw, min_fh)
        free_rects.sort(key=lambda r: (r[2] - r[0]) * (r[3] - r[1]), reverse=True)

        if not free_rects:
            break

        max_y = max((pos[i][1] + sizes[i][1] for i in range(len(packed))), default=0)

        moved_this_pass: set[int] = set()
        new_pos: dict[int, tuple[int, int]] = {}
        placed_footprints: list[tuple[int, int, int, int]] = []

        for fr in free_rects:
            fr_x0, fr_y0, fr_x1, fr_y1 = fr
            fr_w = fr_x1 - fr_x0
            fr_h = fr_y1 - fr_y0

            candidates: list[int] = []
            for i in range(len(packed)):
                if i in moved_this_pass:
                    continue
                if not overflow_flags[i]:
                    continue
                w, h = sizes[i]
                if w + pad * 2 <= fr_w and h + pad * 2 <= fr_h:
                    candidates.append(i)

            bottom_threshold = max_y * 0.75
            compaction_candidates: list[int] = []
            for i in range(len(packed)):
                if i in moved_this_pass:
                    continue
                if overflow_flags[i]:
                    continue
                y_bot = pos[i][1] + sizes[i][1]
                if y_bot >= bottom_threshold:
                    w, h = sizes[i]
                    if w + pad * 2 <= fr_w and h + pad * 2 <= fr_h:
                        compaction_candidates.append(i)
            candidates.extend(compaction_candidates)

            if not candidates:
                continue

            cand_groups: dict[tuple, list[int]] = {}
            for i in candidates:
                key = sim_key(i)
                cand_groups.setdefault(key, []).append(i)
            sorted_groups = sorted(cand_groups.values(), key=len, reverse=True)

            cx = fr_x0 + pad
            cy = fr_y0 + pad
            row_h = 0

            for group in sorted_groups:
                group_sorted = sorted(group, key=lambda i: sizes[i][1], reverse=True)
                for i in group_sorted:
                    if i in moved_this_pass:
                        continue
                    w, h = sizes[i]
                    fw = w + pad * 2
                    fh = h + pad * 2

                    if cx + fw > fr_x1 + 1 and cx > fr_x0 + pad:
                        cx = fr_x0 + pad
                        cy += int(_snap_int(row_h, snap))
                        row_h = 0

                    if cy + fh > fr_y1 + 1:
                        continue

                    new_y_bot = int(cy) + h
                    cur_y_bot = pos[i][1] + h
                    if not overflow_flags[i] and new_y_bot >= cur_y_bot:
                        continue

                    footprint = (int(cx) - pad, int(cy) - pad, int(cx) + w + pad, int(cy) + h + pad)
                    if any(_rects_overlap(footprint, occ) for occ in occupied):
                        continue
                    if any(_rects_overlap(footprint, pf) for pf in placed_footprints):
                        continue

                    new_pos[i] = (int(cx), int(cy))
                    moved_this_pass.add(i)
                    placed_footprints.append(footprint)
                    cx += int(_snap_int(fw, snap))
                    row_h = max(row_h, fh)

        if not moved_this_pass:
            break

        for i, (nx, ny) in new_pos.items():
            pos[i][0] = nx
            pos[i][1] = ny
            w, h = sizes[i]
            overflow_flags[i] = bool(nx + w > hi_x or ny + h > hi_y)

        total_relocated += len(moved_this_pass)

    if total_relocated == 0:
        return packed, 0

    result: list[PackedIsland] = []
    for i, pi in enumerate(packed):
        result.append(PackedIsland(
            island=pi.island,
            x_px=pos[i][0],
            y_px=pos[i][1],
            overflow=overflow_flags[i],
        ))

    return result, total_relocated
#endregion


#region TEXEL-DESC
# Natural-language texel density summary for the log panel.
# Describes the smallest and largest face UV footprints in pixel space,
# plus aggregate stats (average, total used area, atlas utilization).


def _describe_texel_density(
    *,
    packed: list[PackedIsland],
    tex_w: int,
    tex_h: int,
) -> list[str]:
    """Generate natural-language texel density description for the log.

    Examines every face placement in every packed island and reports:
    - Smallest face: element index, face name, pixel dimensions
    - Largest face: element index, face name, pixel dimensions
    - Average face size, total faces, atlas utilization percentage
    """
    if not packed:
        return ["No faces to analyze."]

    faces_info: list[tuple[int, str, int, int, int]] = []
    total_area = 0

    for pisl in packed:
        el_idx = int(pisl.island.element_idx)
        for pl in pisl.island.placements:
            w = max(1, int(pl.w_px))
            h = max(1, int(pl.h_px))
            area = w * h
            total_area += area
            faces_info.append((el_idx, str(pl.face), w, h, area))

    if not faces_info:
        return ["No face placements found."]

    faces_info.sort(key=lambda t: t[4])

    smallest = faces_info[0]
    largest = faces_info[-1]

    avg_area = total_area / len(faces_info)
    avg_side = math.sqrt(avg_area)

    atlas_area = int(tex_w) * int(tex_h)
    utilization = (total_area / atlas_area * 100.0) if atlas_area > 0 else 0.0

    lines: list[str] = []
    lines.append("")
    lines.append("Texel density summary:")
    lines.append(
        f"  Smallest face: element {smallest[0]:03d} {smallest[1]} = "
        f"{smallest[2]}x{smallest[3]} px ({smallest[4]} px^2)"
    )
    lines.append(
        f"  Largest face:  element {largest[0]:03d} {largest[1]} = "
        f"{largest[2]}x{largest[3]} px ({largest[4]} px^2)"
    )
    lines.append(
        f"  {len(faces_info)} faces total, average ~{avg_side:.1f}x{avg_side:.1f} px "
        f"({avg_area:.0f} px^2), atlas utilization {utilization:.1f}%"
    )

    tiny = [f for f in faces_info if f[2] <= 2 or f[3] <= 2]
    if tiny:
        names = ", ".join(f"el{f[0]:03d}:{f[1]}({f[2]}x{f[3]})" for f in tiny[:8])
        extra = f" ... +{len(tiny)-8} more" if len(tiny) > 8 else ""
        lines.append(
            f"  NOTE: {len(tiny)} face(s) are very small (<=2px on one axis): {names}{extra}"
        )

    return lines
#endregion


#region UVWRITE
# UV write-back: deep-copies the model, then writes packed UV rects into each
# face dict. Supports px / norm / mc16 output units. mc16 gets clamping + min-step.


def _ensure_face_dict(el: dict, face: Face, default_texture: str) -> dict:
    faces = el.get("faces")
    if not isinstance(faces, dict):
        faces = {}
        el["faces"] = faces

    f = faces.get(face)
    if not isinstance(f, dict):
        f = {"texture": default_texture}
        faces[face] = f

    if "texture" not in f:
        f["texture"] = default_texture

    return f


def _apply_uvs_to_model(
    *,
    model: dict,
    packed: list[PackedIsland],
    tex_w: int,
    tex_h: int,
    default_texture: str,
    write_rotation: bool,
    uv_units: str,
) -> dict:
    out = json.loads(json.dumps(model))
    els = out.get("elements")
    if not isinstance(els, list):
        raise ValueError("Model missing elements[]")

    placements_by_idx: dict[int, list[tuple[FacePlacement, int, int]]] = {}
    for pisl in packed:
        for pl in pisl.island.placements:
            placements_by_idx.setdefault(int(pisl.island.element_idx), []).append((pl, int(pisl.x_px), int(pisl.y_px)))

    for i, el in enumerate(els):
        if not isinstance(el, dict):
            continue
        pls = placements_by_idx.get(int(i))
        if not pls:
            continue

        for pl, ix, iy in pls:
            f = _ensure_face_dict(el, pl.face, default_texture)

            x0 = int(ix + int(pl.x_px))
            y0 = int(iy + int(pl.y_px))
            w_px = max(1, int(pl.w_px))
            h_px = max(1, int(pl.h_px))
            x1 = int(x0 + w_px)
            y1 = int(y0 + h_px)

            units = str(uv_units or "px")
            if units == "norm":
                u0 = float(x0) / float(tex_w)
                v0 = float(y0) / float(tex_h)
                u1 = float(x1) / float(tex_w)
                v1 = float(y1) / float(tex_h)
            elif units == "mc16":
                u0 = (float(x0) / float(tex_w)) * 16.0
                v0 = (float(y0) / float(tex_h)) * 16.0
                u1 = (float(x1) / float(tex_w)) * 16.0
                v1 = (float(y1) / float(tex_h)) * 16.0
            else:
                u0 = float(x0)
                v0 = float(y0)
                u1 = float(x1)
                v1 = float(y1)

            if units == "mc16":
                step = 0.001

                def clamp16(v: float) -> float:
                    return 0.0 if float(v) < 0.0 else 16.0 if float(v) > 16.0 else float(v)

                u0 = _r3(clamp16(u0))
                v0 = _r3(clamp16(v0))
                u1 = _r3(clamp16(u1))
                v1 = _r3(clamp16(v1))

                if float(u1) <= float(u0):
                    u1 = _r3(clamp16(float(u0) + step))
                    if float(u1) <= float(u0):
                        u0 = _r3(clamp16(float(u1) - step))

                if float(v1) <= float(v0):
                    v1 = _r3(clamp16(float(v0) + step))
                    if float(v1) <= float(v0):
                        v0 = _r3(clamp16(float(v1) - step))

                f["uv"] = [u0, v0, u1, v1]
            else:
                f["uv"] = [_r12(u0), _r12(v0), _r12(u1), _r12(v1)]

            if write_rotation:
                if int(pl.rot) % 360 != 0:
                    f["rotation"] = int(pl.rot) % 360
                else:
                    f.pop("rotation", None)

    return out
#endregion


#region RESNAP
# UV resnap subsystem: re-snaps existing face UVs to a new target texture W/H
# so every UV vertex lands on an integer pixel, while:
#   - keeping each face >= min_face_px (and biased towards >= bias_gt_px)
#   - nudging faces away from the atlas border and from each other to preserve
#     border_pad_px / island_pad_px padding
#   - staying as close as possible to the original UV (small transforms only)
#
# Sub-regions: C-RS-UNITS, C-RS-ISLD, C-RS-SNAP, C-RS-XFORM, C-RS-OVLAP, C-RS-ORCH


#region RS-UNITS
# UV unit detection and pixel<->UV conversion for norm / mc16 / px spaces


def _detect_uv_units(uv_values: Iterable[float]) -> str:
    """Heuristically detect whether UVs are normalized (0..1), mc16 (0..16) or px."""
    max_val = 0.0
    for v in uv_values:
        try:
            fv = float(v)
        except (TypeError, ValueError):
            continue
        if fv > max_val:
            max_val = fv
    if max_val <= 1.0 + 1e-6:
        return "norm"
    if max_val <= 16.0 + 1e-6:
        return "mc16"
    return "px"


def _uv_rect_to_pixels(u0: float, v0: float, u1: float, v1: float, units: str, tex_w: int, tex_h: int) -> tuple[float, float, float, float]:
    if units == "norm":
        return (u0 * float(tex_w), v0 * float(tex_h), u1 * float(tex_w), v1 * float(tex_h))
    if units == "mc16":
        return (u0 / 16.0 * float(tex_w), v0 / 16.0 * float(tex_h), u1 / 16.0 * float(tex_w), v1 / 16.0 * float(tex_h))
    return (u0, v0, u1, v1)


def _pixels_to_uv_rect(px0: float, py0: float, px1: float, py1: float, units: str, tex_w: int, tex_h: int) -> tuple[float, float, float, float]:
    if units == "norm":
        return (px0 / float(tex_w), py0 / float(tex_h), px1 / float(tex_w), py1 / float(tex_h))
    if units == "mc16":
        return (px0 / float(tex_w) * 16.0, py0 / float(tex_h) * 16.0, px1 / float(tex_w) * 16.0, py1 / float(tex_h) * 16.0)
    return (px0, py0, px1, py1)
#endregion


#region RS-ISLD
# Island detection: union-find on touching rects to group faces into islands


def _rects_touch(a: tuple[float, float, float, float], b: tuple[float, float, float, float], *, eps: float = 1e-3) -> bool:
    """True if two axis-aligned rects share an edge (touching but not overlapping)."""
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b

    # overlap ranges
    ovx = min(ax1, bx1) - max(ax0, bx0)
    ovy = min(ay1, by1) - max(ay0, by0)

    # touching: one axis has shared edge, other axis has overlap
    if ovy > eps and (abs(ax1 - bx0) <= eps or abs(bx1 - ax0) <= eps):
        return True
    if ovx > eps and (abs(ay1 - by0) <= eps or abs(by1 - ay0) <= eps):
        return True
    return False


def _detect_uv_islands(faces_data: list[dict]) -> list[list[int]]:
    """Group face indices into islands via union-find on touching rects.

    Faces whose pixel rects touch (share an edge) belong to the same island.
    """
    n = len(faces_data)
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i in range(n):
        ri = (faces_data[i]["cur"][0], faces_data[i]["cur"][1], faces_data[i]["cur"][2], faces_data[i]["cur"][3])
        for j in range(i + 1, n):
            rj = (faces_data[j]["cur"][0], faces_data[j]["cur"][1], faces_data[j]["cur"][2], faces_data[j]["cur"][3])
            if _rects_touch(ri, rj, eps=1e-3):
                union(i, j)

    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return list(groups.values())
#endregion


#region RS-SNAP
# Coordinate snapping: maps shared island coords to monotonically increasing
# integers, biases outer boundary outward, enforces min_face_px per face


def _snap_island_coordinates(
    *,
    island_face_indices: list[int],
    faces_data: list[dict],
    min_face_px: int,
    bias_gt_px: int,
) -> None:
    """Snap all unique x/y coordinates of an island to integers in-place.

    Uses a shared coordinate mapping so faces that originally shared an edge
    continue to share an edge after snapping.  The island's outer boundary is
    biased outward (floor min / ceil max) to nudge face sizes up; internal
    coordinates round to nearest.  Monotonicity is enforced so no two distinct
    coordinates collapse to the same integer.

    After applying the shared map, each face is checked individually: if its
    width or height is below ``min_face_px``, the right/bottom edge is expanded
    to meet the minimum.  This per-face override may break edge-sharing for
    that specific face, but ensures no face collapses below the user-specified
    minimum.  ``bias_gt_px`` is enforced as a soft target on the island's
    outer span only.
    """
    # Collect unique x and y coordinates from all faces in the island
    x_coords_set: set[float] = set()
    y_coords_set: set[float] = set()
    for fi in island_face_indices:
        fd = faces_data[fi]
        x_coords_set.add(float(fd["cur"][0]))
        x_coords_set.add(float(fd["cur"][2]))
        y_coords_set.add(float(fd["cur"][1]))
        y_coords_set.add(float(fd["cur"][3]))

    x_sorted = sorted(x_coords_set)
    y_sorted = sorted(y_coords_set)

    def snap_axis(coords: list[float]) -> dict[float, int]:
        """Snap sorted coordinates to monotonically increasing integers."""
        n = len(coords)
        result: dict[float, int] = {}
        last = -(10 ** 18)
        for i, c in enumerate(coords):
            if i == 0:
                s = int(math.floor(c))  # bias: expand left boundary
            elif i == n - 1:
                s = int(math.ceil(c))   # bias: expand right boundary
            else:
                s = int(math.floor(c + 0.5))  # round half up
            if s <= last:
                s = last + 1
            result[c] = s
            last = s

        # Enforce min_face_px / bias_gt_px on the outermost span
        total = result[coords[-1]] - result[coords[0]]
        target = max(int(min_face_px), int(bias_gt_px))
        if total < target and n >= 2:
            need = target - total
            result[coords[-1]] += need
        return result

    x_map = snap_axis(x_sorted)
    y_map = snap_axis(y_sorted)

    # Apply snapped coordinates to each face in the island
    for fi in island_face_indices:
        fd = faces_data[fi]
        px0, py0, px1, py1 = fd["cur"]
        nx0 = x_map[float(px0)]
        ny0 = y_map[float(py0)]
        nx1 = x_map[float(px1)]
        ny1 = y_map[float(py1)]
        if nx1 <= nx0:
            nx1 = nx0 + 1
        if ny1 <= ny0:
            ny1 = ny0 + 1
        fd["cur"] = [float(nx0), float(ny0), float(nx1), float(ny1)]

    # Per-face enforcement of min_face_px: expand right/bottom edge if needed
    min_face = max(1, int(min_face_px))
    for fi in island_face_indices:
        fd = faces_data[fi]
        x0 = int(fd["cur"][0])
        y0 = int(fd["cur"][1])
        x1 = int(fd["cur"][2])
        y1 = int(fd["cur"][3])
        if x1 - x0 < min_face:
            fd["cur"][2] = float(x0 + min_face)
        if y1 - y0 < min_face:
            fd["cur"][3] = float(y0 + min_face)
#endregion


#region RS-XFORM
# Island transforms: bbox, translate, clamp to texture bounds (with degenerate-face fixup)


def _island_bbox(faces_data: list[dict], indices: list[int]) -> tuple[int, int, int, int]:
    xs0 = [int(faces_data[fi]["cur"][0]) for fi in indices]
    ys0 = [int(faces_data[fi]["cur"][1]) for fi in indices]
    xs1 = [int(faces_data[fi]["cur"][2]) for fi in indices]
    ys1 = [int(faces_data[fi]["cur"][3]) for fi in indices]
    return (min(xs0), min(ys0), max(xs1), max(ys1))


def _translate_island(faces_data: list[dict], indices: list[int], dx: int, dy: int) -> None:
    for fi in indices:
        fd = faces_data[fi]
        fd["cur"] = [
            float(int(fd["cur"][0]) + dx),
            float(int(fd["cur"][1]) + dy),
            float(int(fd["cur"][2]) + dx),
            float(int(fd["cur"][3]) + dy),
        ]


def _clamp_island_to_bounds(
    *,
    faces_data: list[dict],
    indices: list[int],
    lo: int,
    hi_x: int,
    hi_y: int,
    tex_w: int,
    tex_h: int,
) -> None:
    x0, y0, x1, y1 = _island_bbox(faces_data, indices)
    dx = 0
    dy = 0
    if x0 < lo:
        dx = lo - x0
    if y0 < lo:
        dy = lo - y0
    if x1 + dx > hi_x:
        dx -= (x1 + dx) - hi_x
    if y1 + dy > hi_y:
        dy -= (y1 + dy) - hi_y
    if dx != 0 or dy != 0:
        _translate_island(faces_data, indices, dx, dy)

    # If island is wider/taller than texture, shrink from the outside
    x0, y0, x1, y1 = _island_bbox(faces_data, indices)
    if x1 - x0 > hi_x - lo:
        _translate_island(faces_data, indices, lo - x0, 0)
        x0, y0, x1, y1 = _island_bbox(faces_data, indices)
        overflow = x1 - hi_x
        if overflow > 0:
            for fi in indices:
                fd = faces_data[fi]
                if int(fd["cur"][2]) > hi_x:
                    fd["cur"][2] = float(hi_x)
                if int(fd["cur"][0]) >= hi_x:
                    fd["cur"][0] = float(hi_x - 1)
    if y1 - y0 > hi_y - lo:
        _translate_island(faces_data, indices, 0, lo - y0)
        x0, y0, x1, y1 = _island_bbox(faces_data, indices)
        overflow = y1 - hi_y
        if overflow > 0:
            for fi in indices:
                fd = faces_data[fi]
                if int(fd["cur"][3]) > hi_y:
                    fd["cur"][3] = float(hi_y)
                if int(fd["cur"][1]) >= hi_y:
                    fd["cur"][1] = float(hi_y - 1)

    # Ensure no face has inverted or zero-width/height after clamping
    for fi in indices:
        fd = faces_data[fi]
        if int(fd["cur"][2]) <= int(fd["cur"][0]):
            fd["cur"][2] = float(int(fd["cur"][0]) + 1)
        if int(fd["cur"][3]) <= int(fd["cur"][1]):
            fd["cur"][3] = float(int(fd["cur"][1]) + 1)
#endregion


#region RS-OVLAP
# Overlap resolution: iteratively translates whole islands apart so every pair
# has at least pad px gap. Only sets moved=True when a translation is actually applied.


def _resolve_island_overlaps(
    *,
    faces_data: list[dict],
    islands: list[list[int]],
    lo: int,
    hi_x: int,
    hi_y: int,
    pad: int,
    max_iter: int = 64,
) -> bool:
    """Nudge whole islands apart so every pair has at least ``pad`` px gap.

    Each island is translated as a unit (all faces move by the same delta),
    preserving internal connectivity.  Returns True if stable.
    """
    for _ in range(max_iter):
        moved = False
        bboxes = [_island_bbox(faces_data, idxs) for idxs in islands]

        for i in range(len(islands)):
            ax0, ay0, ax1, ay1 = bboxes[i]
            for j in range(i + 1, len(islands)):
                bx0, by0, bx1, by1 = bboxes[j]

                # Gap-based test: conflict only if gap < pad in BOTH axes.
                # gap > 0 means separated, gap == 0 means touching,
                # gap < 0 means overlapping.
                gap_x = max(ax0, bx0) - min(ax1, bx1)
                gap_y = max(ay0, by0) - min(ay1, by1)
                if gap_x >= pad or gap_y >= pad:
                    continue  # enough gap in at least one axis

                # Shift amount needed to achieve pad gap
                need_x = pad - gap_x
                need_y = pad - gap_y

                # separate along the axis with smaller needed shift
                if need_x <= need_y:
                    shift = need_x
                    a_left_room = ax0 - lo
                    b_right_room = hi_x - bx1
                    if a_left_room >= b_right_room and a_left_room >= shift:
                        _translate_island(faces_data, islands[i], -shift, 0)
                        moved = True
                    elif b_right_room >= shift:
                        _translate_island(faces_data, islands[j], shift, 0)
                        moved = True
                    else:
                        half = shift // 2
                        rest = shift - half
                        applied = False
                        if a_left_room >= half:
                            _translate_island(faces_data, islands[i], -half, 0)
                            applied = True
                        if b_right_room >= rest:
                            _translate_island(faces_data, islands[j], rest, 0)
                            applied = True
                        if applied:
                            moved = True
                else:
                    shift = need_y
                    a_top_room = ay0 - lo
                    b_bot_room = hi_y - by1
                    if a_top_room >= b_bot_room and a_top_room >= shift:
                        _translate_island(faces_data, islands[i], 0, -shift)
                        moved = True
                    elif b_bot_room >= shift:
                        _translate_island(faces_data, islands[j], 0, shift)
                        moved = True
                    else:
                        half = shift // 2
                        rest = shift - half
                        applied = False
                        if a_top_room >= half:
                            _translate_island(faces_data, islands[i], 0, -half)
                            applied = True
                        if b_bot_room >= rest:
                            _translate_island(faces_data, islands[j], 0, rest)
                            applied = True
                        if applied:
                            moved = True

                if moved:
                    bboxes[i] = _island_bbox(faces_data, islands[i])
                    bboxes[j] = _island_bbox(faces_data, islands[j])
                    ax0, ay0, ax1, ay1 = bboxes[i]
        if not moved:
            return True
    return False
#endregion


#region RS-ORCH
# Top-level resnap orchestrator: 5-step pipeline
# (detect islands -> snap coords -> clamp -> resolve overlaps -> final clamp)
# Returns (new_model, log_lines, detected_uv_units)


def _resnap_model_uvs(
    *,
    model: dict,
    tex_w: int,
    tex_h: int,
    min_face_px: int,
    bias_gt_px: int,
    border_pad_px: int,
    island_pad_px: int,
    uv_units: Optional[str] = None,
) -> tuple[dict, list[str], str]:
    """Re-snap existing face UVs to the target texture grid, preserving islands.

    Faces that share edges in the original UV layout are treated as a single
    island: their shared coordinates are snapped together so the island stays
    connected.  Overlap resolution and border padding operate on whole islands
    (uniform translation), not individual faces.

    Returns (new_model, log_lines, detected_uv_units).
    """
    out = json.loads(json.dumps(model))
    els = out.get("elements")
    if not isinstance(els, list):
        raise ValueError("Model missing elements[]")

    # Collect UV values to auto-detect units if not provided
    all_uv_vals: list[float] = []
    for el in els:
        if not isinstance(el, dict):
            continue
        faces = el.get("faces")
        if not isinstance(faces, dict):
            continue
        for f in faces.values():
            if not isinstance(f, dict):
                continue
            uv = f.get("uv")
            if isinstance(uv, list) and len(uv) >= 4:
                try:
                    all_uv_vals.extend(float(v) for v in uv[:4])
                except (TypeError, ValueError):
                    pass

    if uv_units is None:
        uv_units = _detect_uv_units(all_uv_vals) if all_uv_vals else "mc16"
    units = str(uv_units)

    log: list[str] = []
    log.append(f"Resnap target: {tex_w}x{tex_h}  units={units}")
    log.append(f"Min face: {min_face_px}px  bias>={bias_gt_px}px  border_pad={border_pad_px}px  island_pad={island_pad_px}px")

    # Collect face data with pixel-space rects
    faces_data: list[dict] = []
    for i, el in enumerate(els):
        if not isinstance(el, dict):
            continue
        faces = el.get("faces")
        if not isinstance(faces, dict):
            continue
        for fname, f in faces.items():
            if not isinstance(f, dict):
                continue
            uv = f.get("uv")
            if not isinstance(uv, list) or len(uv) < 4:
                continue
            try:
                u0, v0, u1, v1 = (float(uv[0]), float(uv[1]), float(uv[2]), float(uv[3]))
            except (TypeError, ValueError):
                continue
            px0, py0, px1, py1 = _uv_rect_to_pixels(u0, v0, u1, v1, units, tex_w, tex_h)
            if px1 < px0:
                px0, px1 = px1, px0
            if py1 < py0:
                py0, py1 = py1, py0
            faces_data.append({
                "el": int(i),
                "face": str(fname),
                "ref": f,
                "orig": (px0, py0, px1, py1),
                "cur": [float(px0), float(py0), float(px1), float(py1)],
            })

    if not faces_data:
        log.append("No face UVs found to resnap.")
        return out, log, units

    lo = max(0, int(border_pad_px))
    hi_x = int(tex_w) - lo
    hi_y = int(tex_h) - lo
    pad = max(0, int(island_pad_px))

    # Step 1: detect islands from the original (pre-snap) layout
    islands = _detect_uv_islands(faces_data)
    log.append(f"Detected {len(islands)} UV island(s) from {len(faces_data)} faces")

    # Step 2: snap each island's shared coordinates to integers
    for island_indices in islands:
        _snap_island_coordinates(
            island_face_indices=island_indices,
            faces_data=faces_data,
            min_face_px=min_face_px,
            bias_gt_px=bias_gt_px,
        )

    # Step 3: clamp each island to texture bounds (translate whole island)
    for island_indices in islands:
        _clamp_island_to_bounds(
            faces_data=faces_data,
            indices=island_indices,
            lo=lo,
            hi_x=hi_x,
            hi_y=hi_y,
            tex_w=int(tex_w),
            tex_h=int(tex_h),
        )

    # Step 4: resolve inter-island overlaps with whole-island translations
    stable = _resolve_island_overlaps(
        faces_data=faces_data,
        islands=islands,
        lo=lo,
        hi_x=hi_x,
        hi_y=hi_y,
        pad=pad,
    )

    # Step 5: final clamp after nudges (preserve border padding)
    for island_indices in islands:
        _clamp_island_to_bounds(
            faces_data=faces_data,
            indices=island_indices,
            lo=lo,
            hi_x=hi_x,
            hi_y=hi_y,
            tex_w=int(tex_w),
            tex_h=int(tex_h),
        )

    # Write back
    log.append("")
    max_disp = 0.0
    for fd in faces_data:
        x0, y0, x1, y1 = (int(v) for v in fd["cur"])
        u0, v0, u1, v1 = _pixels_to_uv_rect(float(x0), float(y0), float(x1), float(y1), units, int(tex_w), int(tex_h))
        if units == "mc16":
            fd["ref"]["uv"] = [_r3(u0), _r3(v0), _r3(u1), _r3(v1)]
        else:
            fd["ref"]["uv"] = [_r12(u0), _r12(v0), _r12(u1), _r12(v1)]
        ox0, oy0, ox1, oy1 = fd["orig"]
        disp = abs(x0 - ox0) + abs(x1 - ox1) + abs(y0 - oy0) + abs(y1 - oy1)
        if disp > max_disp:
            max_disp = disp
        log.append(f"  el{fd['el']:03d} {fd['face']:<5s}: ({x0},{y0},{x1},{y1}) {x1-x0}x{y1-y0}px  disp={disp:.1f}px")

    log.append("")
    log.append(f"Resnapped {len(faces_data)} faces in {len(islands)} islands.  Max displacement: {max_disp:.1f}px.  Overlap-stable: {stable}")
    return out, log, units
#endregion
#endregion


#region PREVIEW
# Build a PackedIsland list from existing model face UVs for atlas preview.
# Reuses island detection so the preview reflects the actual UV island structure.


def _uvs_to_packed_preview(*, model: dict, tex_w: int, tex_h: int, uv_units: str) -> list[PackedIsland]:
    """Build a PackedIsland list from existing model face UVs for atlas preview.

    Faces are grouped into islands using the same touching-rect detection as
    the resnap logic, so the preview reflects the actual UV island structure.
    """
    els = model.get("elements")
    if not isinstance(els, list):
        return []

    # Collect face data
    faces_data: list[dict] = []
    for i, el in enumerate(els):
        if not isinstance(el, dict):
            continue
        faces = el.get("faces")
        if not isinstance(faces, dict):
            continue
        for fname, f in faces.items():
            if not isinstance(f, dict):
                continue
            uv = f.get("uv")
            if not isinstance(uv, list) or len(uv) < 4:
                continue
            try:
                u0, v0, u1, v1 = (float(uv[0]), float(uv[1]), float(uv[2]), float(uv[3]))
            except (TypeError, ValueError):
                continue
            px0, py0, px1, py1 = _uv_rect_to_pixels(u0, v0, u1, v1, str(uv_units), int(tex_w), int(tex_h))
            if px1 < px0:
                px0, px1 = px1, px0
            if py1 < py0:
                py0, py1 = py1, py0
            faces_data.append({
                "el": int(i),
                "face": str(fname),
                "cur": [float(px0), float(py0), float(px1), float(py1)],
            })

    if not faces_data:
        return []

    islands = _detect_uv_islands(faces_data)
    packed: list[PackedIsland] = []
    for island_indices in islands:
        placements: list[FacePlacement] = []
        first_el = int(faces_data[island_indices[0]]["el"])
        for fi in island_indices:
            fd = faces_data[fi]
            x = max(0, int(round(fd["cur"][0])))
            y = max(0, int(round(fd["cur"][1])))
            w = max(1, int(round(fd["cur"][2] - fd["cur"][0])))
            h = max(1, int(round(fd["cur"][3] - fd["cur"][1])))
            placements.append(FacePlacement(face=str(fd["face"]), x_px=x, y_px=y, w_px=w, h_px=h, rot=0))
        if placements:
            isl = Island(element_idx=first_el, placements=placements)
            packed.append(PackedIsland(island=isl, x_px=0, y_px=0))
    return packed
#endregion
