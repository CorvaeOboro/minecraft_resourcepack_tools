"""
Minecraft Model UV Packer (UI)

PySide6 tool that generates non-overlapping UVs for Minecraft block-model cuboids
(`elements[]`). It performs a connected box-unwrap per cuboid (a single island
containing all 6 faces where possible), then packs islands into a texture atlas
with configurable padding between islands and padding to the atlas border.

This tool writes UVs in *normalized* (0..1) space with high precision floats.
(You can convert to vanilla block-model UV units later by multiplying by 16.)

TOOLSGROUP::MODEL
SORTGROUP::7
SORTPRIORITY::80
STATUS::active
VERSION::20260309

Expected inputs
- Minecraft model JSON file containing `elements[]`.

Outputs
- Updated model JSON where each element face has a packed `uv` rect.
- Optional per-face `rotation` (90-degree steps) when needed to keep box-net
  edges consistent.
- 2D atlas preview showing packed islands.

"""

#region SETUP
# Module imports, PySide6 bootstrap, and type aliases

from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Literal, Optional, Tuple


def _try_import_pyside6():
    try:
        from PySide6 import QtCore, QtGui, QtWidgets

        return QtCore, QtGui, QtWidgets
    except Exception:
        return None


_imports = _try_import_pyside6()
if _imports is None:
    print("Missing UI dependency. Install PySide6 and retry.")
    print("Example: pip install PySide6")
    raise SystemExit(1)

QtCore, QtGui, QtWidgets = _imports

Face = Literal["north", "south", "east", "west", "up", "down"]
Axis = Literal["x", "y", "z"]
#endregion


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

                # pad-inflated overlap test
                ix0 = max(ax0, bx0) - pad
                ix1 = min(ax1, bx1) + pad
                iy0 = max(ay0, by0) - pad
                iy1 = min(ay1, by1) + pad
                ovx = ix1 - ix0
                ovy = iy1 - iy0
                if ovx <= 0 or ovy <= 0:
                    continue  # no conflict

                # separate along the axis with smaller padded overlap
                if ovx <= ovy:
                    shift = ovx
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
                    shift = ovy
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


#region UI-THEME
# Dark theme stylesheet and per-element color generation (golden-ratio hue stride)


def _apply_dark_theme(app: QtWidgets.QApplication) -> None:
    app.setStyle("Fusion")
    app.setStyleSheet(
        "QWidget { background: #0b0b0d; color: #e8e8ea; font-size: 12px; }"
        "QLabel { color: #e8e8ea; }"
        "QGroupBox { border: 1px solid #1f2024; margin-top: 8px; }"
        "QGroupBox::title { subcontrol-origin: margin; left: 8px; padding: 0 4px 0 4px; color: #cfcfd4; }"
        "QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox { background: #000000; color: #ffffff; border: 1px solid #2a2b30; padding: 4px; border-radius: 4px; }"
        "QPlainTextEdit, QTextEdit { background: #000000; color: #ffffff; border: 1px solid #2a2b30; padding: 6px; border-radius: 4px; }"
        "QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus { border: 1px solid #3a6ea5; }"
        "QComboBox QAbstractItemView { background: #000000; color: #ffffff; selection-background-color: #2b4a6b; }"
        "QCheckBox { spacing: 6px; }"
        "QCheckBox::indicator { width: 14px; height: 14px; }"
        "QPushButton { background: #15161a; color: #e8e8ea; border: 1px solid #2a2b30; padding: 6px 10px; border-radius: 6px; }"
        "QPushButton:hover { border: 1px solid #3a3b42; }"
        "QPushButton:disabled { color: #777780; border: 1px solid #1a1b1f; background: #101114; }"
        "QPushButton#btn_pack { background: #1b3326; border: 1px solid #2d5b3f; }"
        "QPushButton#btn_pack:hover { border: 1px solid #3d7a55; }"
        "QPushButton#btn_save { background: #241a2c; border: 1px solid #4a2c63; }"
        "QPushButton#btn_save:hover { border: 1px solid #6a3b90; }"
        "QSplitter::handle { background: #0b0b0d; }"
    )


def _color_for_index(i: int) -> QtGui.QColor:
    h = (float(int(i) * 0.61803398875) % 1.0) * 360.0
    c = QtGui.QColor()
    c.setHsv(int(h) % 360, 170, 235, 210)
    return c
#endregion


#region UI-ATLAS
# 2D atlas preview (QGraphicsView): pan/zoom, per-element colors, overlap
# highlighting, and seam visualization for the selected element's box-net


class AtlasPreview(QtWidgets.QGraphicsView):
    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self.setRenderHints(QtGui.QPainter.RenderHint.Antialiasing | QtGui.QPainter.RenderHint.TextAntialiasing)
        self.setDragMode(QtWidgets.QGraphicsView.DragMode.NoDrag)
        self.setTransformationAnchor(QtWidgets.QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QtWidgets.QGraphicsView.ViewportAnchor.AnchorUnderMouse)

        self._scene = QtWidgets.QGraphicsScene(self)
        self.setScene(self._scene)

        self._tex_w = 256
        self._tex_h = 256

        self._is_panning = False
        self._pan_last: Optional[QtCore.QPoint] = None
        self._user_view = False

    def _set_user_view(self) -> None:
        self._user_view = True

    def _fake_left_event(self, event: QtGui.QMouseEvent, *, etype: QtCore.QEvent.Type) -> QtGui.QMouseEvent:
        # Drive QGraphicsView's built-in hand-drag logic using middle mouse.
        return QtGui.QMouseEvent(
            etype,
            event.position(),
            event.globalPosition(),
            QtCore.Qt.MouseButton.LeftButton,
            QtCore.Qt.MouseButton.LeftButton,
            event.modifiers(),
        )

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        if event.button() == QtCore.Qt.MouseButton.MiddleButton:
            self._set_user_view()
            self._is_panning = True
            self.setDragMode(QtWidgets.QGraphicsView.DragMode.ScrollHandDrag)
            super().mousePressEvent(self._fake_left_event(event, etype=QtCore.QEvent.Type.MouseButtonPress))
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:
        if event.button() == QtCore.Qt.MouseButton.MiddleButton:
            self._is_panning = False
            super().mouseReleaseEvent(self._fake_left_event(event, etype=QtCore.QEvent.Type.MouseButtonRelease))
            self.setDragMode(QtWidgets.QGraphicsView.DragMode.NoDrag)
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        if self._is_panning:
            super().mouseMoveEvent(self._fake_left_event(event, etype=QtCore.QEvent.Type.MouseMove))
            event.accept()
            return
        super().mouseMoveEvent(event)

    def wheelEvent(self, event: QtGui.QWheelEvent) -> None:
        delta = event.angleDelta().y() / 120.0
        scale = 1.15 ** delta
        self._set_user_view()
        self.scale(scale, scale)

    def set_preview(
        self,
        *,
        tex_w: int,
        tex_h: int,
        packed: list[PackedIsland],
        colors_by_element: Optional[dict[int, QtGui.QColor]] = None,
        selected_element: Optional[int] = None,
    ) -> None:
        tex_w = int(tex_w)
        tex_h = int(tex_h)
        size_changed = tex_w != int(self._tex_w) or tex_h != int(self._tex_h)
        if size_changed:
            self._user_view = False

        self._tex_w = int(tex_w)
        self._tex_h = int(tex_h)

        self._scene.clear()
        self._scene.setSceneRect(0, 0, float(tex_w), float(tex_h))

        bg = QtGui.QColor(10, 10, 12)
        border_pen = QtGui.QPen(QtGui.QColor(130, 130, 140, 120))
        border_pen.setWidthF(1.0)
        self._scene.addRect(0, 0, float(tex_w), float(tex_h), border_pen, QtGui.QBrush(bg))

        face_outline = QtGui.QPen(QtGui.QColor(220, 220, 230, 130))
        face_outline.setWidthF(1.0)
        face_outline_over = QtGui.QPen(QtGui.QColor(255, 85, 85, 220))
        face_outline_over.setWidthF(2.0)
        face_outline_sel = QtGui.QPen(QtGui.QColor(255, 255, 255, 220))
        face_outline_sel.setWidthF(2.0)

        face_outline_overlap = QtGui.QPen(QtGui.QColor(255, 0, 255, 255))
        face_outline_overlap.setWidthF(2.5)

        text_pen = QtGui.QPen(QtGui.QColor(230, 230, 235, 180))
        font = QtGui.QFont("Consolas", 8)

        def intersects(a: QtCore.QRectF, b: QtCore.QRectF) -> bool:
            x0 = max(float(a.left()), float(b.left()))
            y0 = max(float(a.top()), float(b.top()))
            x1 = min(float(a.right()), float(b.right()))
            y1 = min(float(a.bottom()), float(b.bottom()))
            return (x1 - x0) > 1e-6 and (y1 - y0) > 1e-6

        drawn_rects: list[tuple[int, QtCore.QRectF, bool]] = []

        for i, pisl in enumerate(packed):
            base = None
            if colors_by_element is not None:
                base = colors_by_element.get(int(pisl.island.element_idx))
            if base is None:
                base = _color_for_index(int(pisl.island.element_idx))

            alpha = 90
            if selected_element is not None and int(pisl.island.element_idx) != int(selected_element):
                alpha = 40
            col = QtGui.QColor(base.red(), base.green(), base.blue(), alpha)

            for pl in pisl.island.placements:
                x = float(pisl.x_px + pl.x_px)
                y = float(pisl.y_px + pl.y_px)
                w = float(pl.w_px)
                h = float(pl.h_px)
                r = QtCore.QRectF(x, y, w, h)
                overlap = False
                for _el_idx, rr, _ov in drawn_rects:
                    if intersects(r, rr):
                        overlap = True
                        break

                pen = face_outline
                if selected_element is not None and int(pisl.island.element_idx) == int(selected_element):
                    pen = face_outline_sel
                if bool(pisl.overflow):
                    pen = face_outline_over
                if overlap:
                    pen = face_outline_overlap
                self._scene.addRect(x, y, w, h, pen, QtGui.QBrush(col))

                drawn_rects.append((int(pisl.island.element_idx), r, overlap))

                lab = f"{pisl.island.element_idx}:{pl.face}"
                if int(pl.rot) % 360 != 0:
                    lab += f" r{int(pl.rot)%360}"
                t = self._scene.addText(lab, font)
                t.setDefaultTextColor(text_pen.color())
                t.setPos(x + 2.0, y + 1.0)

        def stable_hash(s: str) -> int:
            h = 2166136261
            for ch in s:
                h ^= ord(ch)
                h = (h * 16777619) & 0xFFFFFFFF
            return int(h)

        def color_for_edge_key(a: str, b: str) -> QtGui.QColor:
            k0, k1 = (a, b) if str(a) <= str(b) else (b, a)
            hv = stable_hash(f"{k0}|{k1}")
            hue = int(hv % 360)
            c = QtGui.QColor()
            c.setHsv(hue, 220, 255, 255)
            return c

        def rects_touch(a: QtCore.QRectF, b: QtCore.QRectF, *, eps: float = 1e-6) -> bool:
            ax0, ay0, ax1, ay1 = float(a.left()), float(a.top()), float(a.right()), float(a.bottom())
            bx0, by0, bx1, by1 = float(b.left()), float(b.top()), float(b.right()), float(b.bottom())

            ovx = min(ax1, bx1) - max(ax0, bx0)
            ovy = min(ay1, by1) - max(ay0, by0)

            if ovy > eps and (abs(ax1 - bx0) <= eps or abs(bx1 - ax0) <= eps):
                return True
            if ovx > eps and (abs(ay1 - by0) <= eps or abs(by1 - ay0) <= eps):
                return True
            return False

        def rotate_local_side_for_uv(*, uv_side: str, rot_deg: int) -> str:
            r = int(rot_deg) % 360
            if r not in {0, 90, 180, 270}:
                r = 0

            order = ["top", "right", "bottom", "left"]
            if uv_side not in order:
                return str(uv_side)

            idx = order.index(uv_side)
            steps = (r // 90) % 4

            local = order[(idx - steps) % 4]
            return str(local)

        face_neighbors = {
            "north": {"top": "up", "bottom": "down", "left": "west", "right": "east"},
            "south": {"top": "up", "bottom": "down", "left": "east", "right": "west"},
            "east": {"top": "up", "bottom": "down", "left": "south", "right": "north"},
            "west": {"top": "up", "bottom": "down", "left": "north", "right": "south"},
            "up": {"top": "north", "bottom": "south", "left": "west", "right": "east"},
            "down": {"top": "south", "bottom": "north", "left": "west", "right": "east"},
        }

        if selected_element is not None:
            sel = int(selected_element)
            sel_item = next((pi for pi in packed if int(pi.island.element_idx) == sel), None)
            if sel_item is not None:
                by_face: dict[str, tuple[QtCore.QRectF, int]] = {}
                for pl in sel_item.island.placements:
                    x = float(sel_item.x_px + pl.x_px)
                    y = float(sel_item.y_px + pl.y_px)
                    r = QtCore.QRectF(x, y, float(pl.w_px), float(pl.h_px))
                    by_face[str(pl.face)] = (r, int(pl.rot) % 360)

                connected: set[tuple[str, str]] = set()
                faces = list(by_face.keys())
                for i in range(len(faces)):
                    for j in range(i + 1, len(faces)):
                        a = by_face[faces[i]][0]
                        b = by_face[faces[j]][0]
                        if rects_touch(a, b, eps=1e-6):
                            f0 = str(faces[i])
                            f1 = str(faces[j])
                            k = (f0, f1) if f0 <= f1 else (f1, f0)
                            connected.add(k)

                seam_pen_width = 3.0
                for face, (r, rot) in by_face.items():
                    neigh_map = face_neighbors.get(str(face), {})
                    for uv_side in ("top", "right", "bottom", "left"):
                        local_side = rotate_local_side_for_uv(uv_side=uv_side, rot_deg=rot)
                        neigh = str(neigh_map.get(str(local_side), ""))
                        if not neigh:
                            continue

                        k = (str(face), neigh) if str(face) <= neigh else (neigh, str(face))
                        if k in connected:
                            continue

                        col = color_for_edge_key(str(face), neigh)
                        pen = QtGui.QPen(col)
                        pen.setWidthF(seam_pen_width)
                        pen.setCosmetic(True)

                        if uv_side == "top":
                            self._scene.addLine(float(r.left()), float(r.top()), float(r.right()), float(r.top()), pen)
                        elif uv_side == "bottom":
                            self._scene.addLine(float(r.left()), float(r.bottom()), float(r.right()), float(r.bottom()), pen)
                        elif uv_side == "left":
                            self._scene.addLine(float(r.left()), float(r.top()), float(r.left()), float(r.bottom()), pen)
                        else:
                            self._scene.addLine(float(r.right()), float(r.top()), float(r.right()), float(r.bottom()), pen)

        if not self._user_view:
            self.resetTransform()
            if tex_w > 0 and tex_h > 0:
                self.fitInView(0, 0, float(tex_w), float(tex_h), QtCore.Qt.AspectRatioMode.KeepAspectRatio)
#endregion


#region UI-VIEW
# 3D model viewport (QWidget): hand-rolled perspective camera, painter's-algorithm
# depth sort, face shading, wireframe, and click-to-pick cuboid selection


class UVModelViewport(QtWidgets.QWidget):
    selectionChanged = QtCore.Signal(int)

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)

        self.setMouseTracking(True)
        self.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)

        self._yaw = 45.0
        self._pitch = 25.0
        self._distance = 42.0
        self._center = (8.0, 8.0, 8.0)

        self._last_mouse_pos: Optional[QtCore.QPoint] = None
        self._mouse_down: Optional[QtCore.QPoint] = None
        self._cuboids: list[Cuboid] = []
        self._colors: list[QtGui.QColor] = []
        self._central_faces: list[Optional[Face]] = []
        self._shade_only_central = False
        self._draw_faces = True
        self._draw_wireframe = True

        self._selected: Optional[int] = None

    def set_render_options(self, *, faces: bool, wireframe: bool, shade_only_central: bool) -> None:
        self._draw_faces = bool(faces)
        self._draw_wireframe = bool(wireframe)
        self._shade_only_central = bool(shade_only_central)
        self.update()

    def set_cuboids(
        self,
        cuboids: Iterable[Cuboid],
        *,
        colors: Optional[list[QtGui.QColor]] = None,
        central_faces: Optional[list[Optional[Face]]] = None,
    ) -> None:
        self._cuboids = list(cuboids)
        self._colors = list(colors) if colors is not None else []
        self._central_faces = list(central_faces) if central_faces is not None else []
        self.update()

    def set_selected(self, idx: Optional[int]) -> None:
        self._selected = None if idx is None else int(idx)
        self.update()

    def wheelEvent(self, event: QtGui.QWheelEvent) -> None:
        delta = event.angleDelta().y() / 120.0
        self._distance *= 0.9 ** delta
        self._distance = max(6.0, min(250.0, self._distance))
        self.update()

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            p = event.position().toPoint()
            self._last_mouse_pos = p
            self._mouse_down = p
            event.accept()
            return

        if event.button() == QtCore.Qt.MouseButton.RightButton:
            idx = self._pick_cuboid(event.position())
            if idx is not None:
                self._selected = int(idx)
                self.selectionChanged.emit(int(idx))
                self.update()
            event.accept()
            return

        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            if self._mouse_down is not None:
                p = event.position().toPoint()
                dx = int(p.x() - self._mouse_down.x())
                dy = int(p.y() - self._mouse_down.y())
                if (dx * dx + dy * dy) <= 9:
                    idx = self._pick_cuboid(event.position())
                    if idx is not None:
                        self._selected = int(idx)
                        self.selectionChanged.emit(int(idx))
                        self.update()
            self._last_mouse_pos = None
            self._mouse_down = None
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        if self._last_mouse_pos is not None:
            p = event.position().toPoint()
            dx = p.x() - self._last_mouse_pos.x()
            dy = p.y() - self._last_mouse_pos.y()
            self._last_mouse_pos = p
            self._yaw += float(dx) * 0.35
            self._pitch += float(dy) * 0.35
            self._pitch = max(-89.0, min(89.0, self._pitch))
            self.update()
            event.accept()
            return
        super().mouseMoveEvent(event)

    def _camera(self):
        w = max(1, self.width())
        h = max(1, self.height())

        def rot_y(x: float, y: float, z: float, a: float) -> tuple[float, float, float]:
            c = math.cos(a)
            s = math.sin(a)
            return (c * x + s * z, y, -s * x + c * z)

        def rot_x(x: float, y: float, z: float, a: float) -> tuple[float, float, float]:
            c = math.cos(a)
            s = math.sin(a)
            return (x, c * y - s * z, s * y + c * z)

        yaw = math.radians(self._yaw)
        pitch = math.radians(self._pitch)

        cx0, cy0, cz0 = self._center

        def to_camera(p: tuple[float, float, float]) -> tuple[float, float, float]:
            x, y, z = p
            x -= cx0
            y -= cy0
            z -= cz0
            x, y, z = rot_y(x, y, z, yaw)
            x, y, z = rot_x(x, y, z, pitch)
            z += self._distance
            return x, y, z

        f = min(w, h) * 0.9

        def project_cam(cam: tuple[float, float, float]) -> Optional[QtCore.QPointF]:
            x, y, z = cam
            if z <= 0.05:
                return None
            sx = (x * f) / z + (w * 0.5)
            sy = (-y * f) / z + (h * 0.5)
            return QtCore.QPointF(sx, sy)

        return to_camera, project_cam

    def _cuboid_points(self, c: Cuboid) -> list[tuple[float, float, float]]:
        fx, fy, fz = c.fr
        tx, ty, tz = c.to

        p000 = (fx, fy, fz)
        p100 = (tx, fy, fz)
        p110 = (tx, ty, fz)
        p010 = (fx, ty, fz)
        p001 = (fx, fy, tz)
        p101 = (tx, fy, tz)
        p111 = (tx, ty, tz)
        p011 = (fx, ty, tz)

        pts = [p000, p100, p110, p010, p001, p101, p111, p011]

        if c.rotation is not None and float(c.rotation.angle) != 0.0:
            axis = str(c.rotation.axis)
            angle_deg = float(c.rotation.angle)
            origin = (float(c.rotation.origin[0]), float(c.rotation.origin[1]), float(c.rotation.origin[2]))

            def rotate(p: tuple[float, float, float]) -> tuple[float, float, float]:
                ox, oy, oz = origin
                x, y, z = p
                x -= ox
                y -= oy
                z -= oz

                a = math.radians(angle_deg)
                cc = math.cos(a)
                ss = math.sin(a)

                if axis == "x":
                    yy = cc * y - ss * z
                    zz = ss * y + cc * z
                    xx = x
                elif axis == "y":
                    xx = cc * x + ss * z
                    zz = -ss * x + cc * z
                    yy = y
                else:
                    xx = cc * x - ss * y
                    yy = ss * x + cc * y
                    zz = z

                return (xx + ox, yy + oy, zz + oz)

            pts = [rotate(p) for p in pts]

        return pts

    def _pick_cuboid(self, pos: QtCore.QPointF) -> Optional[int]:
        if not self._cuboids:
            return None

        to_camera, project_cam = self._camera()
        px = float(pos.x())
        py = float(pos.y())

        best_i: Optional[int] = None
        best_z = 1e18

        for i, c in enumerate(self._cuboids):
            pts = self._cuboid_points(c)
            cam_pts = [to_camera(p) for p in pts]
            proj = [project_cam(cp) for cp in cam_pts]
            proj2 = [p for p in proj if p is not None]
            if not proj2:
                continue

            xs = [float(p.x()) for p in proj2]
            ys = [float(p.y()) for p in proj2]
            x0, x1 = min(xs), max(xs)
            y0, y1 = min(ys), max(ys)

            if not (x0 <= px <= x1 and y0 <= py <= y1):
                continue

            z = min(float(cp[2]) for cp in cam_pts)
            if z < best_z:
                best_z = z
                best_i = int(i)

        return best_i

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)
        painter.fillRect(self.rect(), QtGui.QColor(12, 12, 16))

        to_camera, project_cam = self._camera()

        face_idx: list[tuple[tuple[int, int, int, int], Face]] = [
            ((0, 1, 2, 3), "north"),
            ((4, 5, 6, 7), "south"),
            ((0, 1, 5, 4), "down"),
            ((1, 2, 6, 5), "east"),
            ((2, 3, 7, 6), "up"),
            ((3, 0, 4, 7), "west"),
        ]

        faces: list[tuple[float, QtGui.QPolygonF, QtGui.QColor]] = []

        label: Optional[tuple[QtCore.QPointF, str]] = None

        face_letter = {
            "north": "N",
            "south": "S",
            "east": "E",
            "west": "W",
            "up": "U",
            "down": "D",
        }

        if self._draw_faces:
            for i_c, c in enumerate(self._cuboids):
                pts = self._cuboid_points(c)
                cam_pts = [to_camera(p) for p in pts]

                base_col = _color_for_index(int(i_c))
                if 0 <= int(i_c) < len(self._colors):
                    base_col = self._colors[int(i_c)]

                face_base = QtGui.QColor(base_col.red(), base_col.green(), base_col.blue(), 135)
                central = None
                if 0 <= int(i_c) < len(self._central_faces):
                    central = self._central_faces[int(i_c)]

                for (i0, i1, i2, i3), fname in face_idx:
                    if self._shade_only_central and central is not None and fname != central:
                        continue

                    p0 = project_cam(cam_pts[i0])
                    p1 = project_cam(cam_pts[i1])
                    p2 = project_cam(cam_pts[i2])
                    p3 = project_cam(cam_pts[i3])
                    if p0 is None or p1 is None or p2 is None or p3 is None:
                        continue

                    v0 = cam_pts[i0]
                    v1 = cam_pts[i1]
                    v2 = cam_pts[i2]
                    ax, ay, az = (v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2])
                    bx, by, bz = (v2[0] - v0[0], v2[1] - v0[1], v2[2] - v0[2])
                    nx = (ay * bz) - (az * by)
                    ny = (az * bx) - (ax * bz)
                    nz = (ax * by) - (ay * bx)
                    nlen = math.sqrt((nx * nx) + (ny * ny) + (nz * nz))
                    shade = 0.85
                    if nlen > 1e-9:
                        shade = 0.45 + (0.55 * max(0.0, min(1.0, abs(float(nz)) / nlen)))

                    col = QtGui.QColor(
                        int(face_base.red() * shade),
                        int(face_base.green() * shade),
                        int(face_base.blue() * shade),
                        face_base.alpha(),
                    )
                    poly = QtGui.QPolygonF([p0, p1, p2, p3])
                    depth = (cam_pts[i0][2] + cam_pts[i1][2] + cam_pts[i2][2] + cam_pts[i3][2]) / 4.0
                    faces.append((depth, poly, col))

                    if self._selected is not None and int(i_c) == int(self._selected) and central is not None and str(fname) == str(central):
                        cx = (float(p0.x()) + float(p1.x()) + float(p2.x()) + float(p3.x())) / 4.0
                        cy = (float(p0.y()) + float(p1.y()) + float(p2.y()) + float(p3.y())) / 4.0
                        letter = face_letter.get(str(fname), str(fname)[:1].upper())
                        label = (QtCore.QPointF(cx, cy), str(letter))

            faces.sort(key=lambda it: it[0], reverse=True)
            painter.setPen(QtCore.Qt.PenStyle.NoPen)
            for _depth, poly, col in faces:
                painter.setBrush(QtGui.QBrush(col))
                painter.drawPolygon(poly)

            if label is not None:
                pos, letter = label
                painter.setPen(QtGui.QPen(QtGui.QColor(0, 0, 0, 220)))
                f = QtGui.QFont("Consolas", 14, QtGui.QFont.Weight.Bold)
                painter.setFont(f)
                painter.drawText(pos + QtCore.QPointF(1.5, 1.5), str(letter))
                painter.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255, 240)))
                painter.drawText(pos, str(letter))

        if self._draw_wireframe:
            for i_c, c in enumerate(self._cuboids):
                pts = self._cuboid_points(c)

                base_col = _color_for_index(int(i_c))
                if 0 <= int(i_c) < len(self._colors):
                    base_col = self._colors[int(i_c)]

                line_col = QtGui.QColor(base_col.red(), base_col.green(), base_col.blue(), 235)

                width = 1.35
                if self._selected is not None and int(i_c) == int(self._selected):
                    width = 2.6
                    line_col = QtGui.QColor(255, 255, 255, 240)

                    cx = sum(p[0] for p in pts) / 8.0
                    cy = sum(p[1] for p in pts) / 8.0
                    cz = sum(p[2] for p in pts) / 8.0
                    pts = [(cx + (p[0] - cx) * 1.03, cy + (p[1] - cy) * 1.03, cz + (p[2] - cz) * 1.03) for p in pts]

                edges = (
                    (0, 1),
                    (1, 2),
                    (2, 3),
                    (3, 0),
                    (4, 5),
                    (5, 6),
                    (6, 7),
                    (7, 4),
                    (0, 4),
                    (1, 5),
                    (2, 6),
                    (3, 7),
                )

                pen = QtGui.QPen(line_col)
                pen.setWidthF(float(width))
                painter.setPen(pen)
                for ia, ib in edges:
                    pa = project_cam(to_camera(pts[ia]))
                    pb = project_cam(to_camera(pts[ib]))
                    if pa is None or pb is None:
                        continue
                    painter.drawLine(pa, pb)

        painter.end()
#endregion


#region UI-MAIN
# Main window: 3-pane splitter (controls | 3D+atlas | output), slots for
# load/pack/save/resnap, element list with central-face overrides


class UVPackerMainWindow(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Minecraft Model UV Packer")

        self._source_path: Optional[Path] = None
        self._model: Optional[dict] = None
        self._elements: list[Element] = []

        self._central_overrides: dict[int, Face] = {}
        self._selected_element: Optional[int] = None
        self._element_colors: list[QtGui.QColor] = []
        self._last_central_faces: list[Optional[Face]] = []

        self._packed: list[PackedIsland] = []
        self._out_model: Optional[dict] = None

        self._viewport_3d = UVModelViewport()
        self._viewport_3d.selectionChanged.connect(self._on_viewport_selected)

        self._preview = AtlasPreview()

        self._mid_split = QtWidgets.QSplitter()
        self._mid_split.setOrientation(QtCore.Qt.Orientation.Horizontal)
        self._mid_split.addWidget(self._viewport_3d)
        self._mid_split.addWidget(self._preview)
        self._mid_split.setStretchFactor(0, 1)
        self._mid_split.setStretchFactor(1, 1)

        controls = self._build_controls()
        output = self._build_output()

        splitter = QtWidgets.QSplitter()
        splitter.setOrientation(QtCore.Qt.Orientation.Horizontal)
        splitter.addWidget(controls)
        splitter.addWidget(self._mid_split)
        splitter.addWidget(output)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 0)
        splitter.setCollapsible(0, False)
        splitter.setCollapsible(2, False)

        self.setCentralWidget(splitter)

    def _sync_views(self) -> None:
        colors_by_element = {i: self._element_colors[i] for i in range(len(self._element_colors))}
        self._preview.set_preview(
            tex_w=int(self._tex_w.value()),
            tex_h=int(self._tex_h.value()),
            packed=self._packed,
            colors_by_element=colors_by_element,
            selected_element=self._selected_element,
        )

        self._viewport_3d.set_render_options(
            faces=bool(self._vp_faces.isChecked()),
            wireframe=bool(self._vp_wire.isChecked()),
            shade_only_central=bool(self._vp_shade_only_central.isChecked()),
        )
        self._viewport_3d.set_cuboids(
            [e.cuboid for e in self._elements],
            colors=self._element_colors,
            central_faces=self._last_central_faces,
        )
        self._viewport_3d.set_selected(self._selected_element)

    def _on_viewport_selected(self, idx: int) -> None:
        if 0 <= int(idx) < self._lst_elements.count():
            self._lst_elements.setCurrentRow(int(idx))

    def _on_select_element(self, row: int) -> None:
        if row < 0 or row >= len(self._elements):
            self._selected_element = None
        else:
            self._selected_element = int(self._elements[int(row)].idx)

        sel = self._selected_element
        self._btn_apply_override.setEnabled(sel is not None)
        self._btn_clear_override.setEnabled(sel is not None and int(sel) in self._central_overrides)

        if sel is None:
            self._override_face.setCurrentIndex(self._override_face.findData("auto"))
        else:
            ov = self._central_overrides.get(int(sel))
            if ov is None:
                self._override_face.setCurrentIndex(self._override_face.findData("auto"))
            else:
                self._override_face.setCurrentIndex(self._override_face.findData(str(ov)))

        self._sync_views()

    def _on_apply_override(self) -> None:
        if self._selected_element is None:
            return
        v = str(self._override_face.currentData() or "auto")
        if v == "auto":
            self._central_overrides.pop(int(self._selected_element), None)
        else:
            self._central_overrides[int(self._selected_element)] = v  # type: ignore[assignment]
        self._btn_clear_override.setEnabled(int(self._selected_element) in self._central_overrides)
        self._on_pack()

    def _on_clear_override(self) -> None:
        if self._selected_element is None:
            return
        self._central_overrides.pop(int(self._selected_element), None)
        self._btn_clear_override.setEnabled(False)
        self._on_pack()

    def _build_controls(self) -> QtWidgets.QWidget:
        w = QtWidgets.QWidget()
        w.setMinimumWidth(410)

        self._txt_path = QtWidgets.QLineEdit("")
        self._txt_path.setReadOnly(True)

        self._btn_load = QtWidgets.QPushButton("Load JSON")
        self._btn_load.clicked.connect(self._on_load)

        self._tex_w = QtWidgets.QSpinBox()
        self._tex_w.setRange(1, 8192)
        self._tex_w.setValue(256)

        self._tex_h = QtWidgets.QSpinBox()
        self._tex_h.setRange(1, 8192)
        self._tex_h.setValue(256)

        self._density = QtWidgets.QDoubleSpinBox()
        self._density.setRange(0.01, 16.0)
        self._density.setDecimals(4)
        self._density.setValue(1.0)

        self._snap_px = QtWidgets.QSpinBox()
        self._snap_px.setRange(1, 256)
        self._snap_px.setValue(1)

        self._island_pad = QtWidgets.QSpinBox()
        self._island_pad.setRange(0, 256)
        self._island_pad.setValue(2)

        self._border_pad = QtWidgets.QSpinBox()
        self._border_pad.setRange(0, 256)
        self._border_pad.setValue(2)

        self._prefer_horizontal = QtWidgets.QCheckBox("Prefer horizontal central face")
        self._prefer_horizontal.setChecked(True)

        self._central_mode = QtWidgets.QComboBox()
        self._central_mode.addItem("Auto", userData="auto")
        for f in ("north", "south", "east", "west", "up", "down"):
            self._central_mode.addItem(str(f), userData=str(f))

        self._default_texture = QtWidgets.QLineEdit("#0")

        self._uv_units = QtWidgets.QComboBox()
        self._uv_units.addItem("Atlas pixels (0..W/H)", userData="px")
        self._uv_units.addItem("Normalized (0..1)", userData="norm")
        self._uv_units.addItem("Minecraft 0..16", userData="mc16")
        self._uv_units.setCurrentIndex(self._uv_units.findData("mc16"))
        self._uv_units.setEnabled(False)

        self._write_rotation = QtWidgets.QCheckBox("Write face rotation")
        self._write_rotation.setChecked(True)

        self._lst_elements = QtWidgets.QListWidget()
        self._lst_elements.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        self._lst_elements.currentRowChanged.connect(self._on_select_element)

        self._override_face = QtWidgets.QComboBox()
        self._override_face.addItem("Auto", userData="auto")
        for f in ("north", "south", "east", "west", "up", "down"):
            self._override_face.addItem(str(f), userData=str(f))

        self._btn_apply_override = QtWidgets.QPushButton("Apply central face")
        self._btn_apply_override.clicked.connect(self._on_apply_override)
        self._btn_apply_override.setEnabled(False)

        self._btn_clear_override = QtWidgets.QPushButton("Clear override")
        self._btn_clear_override.clicked.connect(self._on_clear_override)
        self._btn_clear_override.setEnabled(False)

        self._vp_faces = QtWidgets.QCheckBox("3D faces")
        self._vp_faces.setChecked(True)
        self._vp_faces.stateChanged.connect(self._sync_views)

        self._vp_wire = QtWidgets.QCheckBox("3D wireframe")
        self._vp_wire.setChecked(True)
        self._vp_wire.stateChanged.connect(self._sync_views)

        self._vp_shade_only_central = QtWidgets.QCheckBox("Shade only central face")
        self._vp_shade_only_central.setChecked(False)
        self._vp_shade_only_central.stateChanged.connect(self._sync_views)

        self._btn_pack = QtWidgets.QPushButton("Generate + Pack")
        self._btn_pack.setObjectName("btn_pack")
        self._btn_pack.clicked.connect(self._on_pack)
        self._btn_pack.setEnabled(False)

        self._btn_save = QtWidgets.QPushButton("Save Updated JSON")
        self._btn_save.setObjectName("btn_save")
        self._btn_save.clicked.connect(self._on_save)
        self._btn_save.setEnabled(False)

        # --- Resnap controls ---
        self._resnap_min_face = QtWidgets.QSpinBox()
        self._resnap_min_face.setRange(1, 64)
        self._resnap_min_face.setValue(1)
        self._resnap_min_face.setToolTip("Minimum face size in pixels after snapping. Faces smaller than this are expanded.")

        self._resnap_bias_gt = QtWidgets.QSpinBox()
        self._resnap_bias_gt.setRange(1, 64)
        self._resnap_bias_gt.setValue(2)
        self._resnap_bias_gt.setToolTip("Bias face size towards at least this many pixels when there is room (keeps small cuboid faces from collapsing to 1px).")

        self._resnap_border_pad = QtWidgets.QSpinBox()
        self._resnap_border_pad.setRange(0, 256)
        self._resnap_border_pad.setValue(1)
        self._resnap_border_pad.setToolTip("Pixels of padding kept between every face and the texture border during resnap.")

        self._resnap_island_pad = QtWidgets.QSpinBox()
        self._resnap_island_pad.setRange(0, 256)
        self._resnap_island_pad.setValue(1)
        self._resnap_island_pad.setToolTip("Minimum pixel gap kept between faces during resnap overlap resolution.")

        self._resnap_uv_units = QtWidgets.QComboBox()
        self._resnap_uv_units.addItem("Auto", userData="auto")
        self._resnap_uv_units.addItem("Minecraft 0..16", userData="mc16")
        self._resnap_uv_units.addItem("Normalized 0..1", userData="norm")
        self._resnap_uv_units.addItem("Atlas pixels", userData="px")
        self._resnap_uv_units.setCurrentIndex(self._resnap_uv_units.findData("mc16"))
        self._resnap_uv_units.setToolTip("How to interpret existing UV values. 'Auto' guesses from the value range, but pick explicitly if ambiguous.")

        self._btn_resnap = QtWidgets.QPushButton("Resnap UVs to Texture")
        self._btn_resnap.setObjectName("btn_pack")
        self._btn_resnap.clicked.connect(self._on_resnap)
        self._btn_resnap.setEnabled(False)
        self._btn_resnap.setToolTip(
            "Re-snaps existing face UVs to the current Texture W/H so every UV vertex lands on an integer pixel.\n"
            "Biases faces to >= min px (preferably >= bias px), nudges away from borders / neighbors for padding,\n"
            "and keeps the result close to the original UV layout."
        )

        self._lbl_status = QtWidgets.QLabel("Ready")
        self._lbl_status.setWordWrap(True)

        model_box = QtWidgets.QGroupBox("Model")
        model_form = QtWidgets.QFormLayout(model_box)
        model_form.setContentsMargins(8, 6, 8, 6)
        model_form.setVerticalSpacing(4)
        model_form.addRow("Path", self._txt_path)
        model_form.addRow(self._btn_load)

        tex_box = QtWidgets.QGroupBox("Atlas")
        tex_form = QtWidgets.QFormLayout(tex_box)
        tex_form.setContentsMargins(8, 6, 8, 6)
        tex_form.setVerticalSpacing(4)
        tex_form.addRow("Texture W", self._tex_w)
        tex_form.addRow("Texture H", self._tex_h)
        tex_form.addRow("Density", self._density)
        tex_form.addRow("Snap (px)", self._snap_px)
        tex_form.addRow("Island pad (px)", self._island_pad)
        tex_form.addRow("Border pad (px)", self._border_pad)

        unwrap_box = QtWidgets.QGroupBox("Box unwrap")
        unwrap_form = QtWidgets.QFormLayout(unwrap_box)
        unwrap_form.setContentsMargins(8, 6, 8, 6)
        unwrap_form.setVerticalSpacing(4)
        unwrap_form.addRow(self._prefer_horizontal)
        unwrap_form.addRow("Central face", self._central_mode)
        unwrap_form.addRow("Default texture", self._default_texture)
        unwrap_form.addRow("UV units", self._uv_units)
        unwrap_form.addRow(self._write_rotation)

        el_box = QtWidgets.QGroupBox("Elements")
        el_layout = QtWidgets.QVBoxLayout(el_box)
        el_layout.setContentsMargins(8, 6, 8, 6)
        el_layout.setSpacing(6)
        el_layout.addWidget(self._lst_elements)

        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("Central"))
        row.addWidget(self._override_face)
        el_layout.addLayout(row)

        row_btn = QtWidgets.QHBoxLayout()
        row_btn.addWidget(self._btn_apply_override)
        row_btn.addWidget(self._btn_clear_override)
        el_layout.addLayout(row_btn)

        view_box = QtWidgets.QGroupBox("3D")
        view_layout = QtWidgets.QVBoxLayout(view_box)
        view_layout.setContentsMargins(8, 6, 8, 6)
        view_layout.setSpacing(4)
        view_layout.addWidget(self._vp_faces)
        view_layout.addWidget(self._vp_wire)
        view_layout.addWidget(self._vp_shade_only_central)

        resnap_box = QtWidgets.QGroupBox("Resnap existing UVs")
        resnap_form = QtWidgets.QFormLayout(resnap_box)
        resnap_form.setContentsMargins(8, 6, 8, 6)
        resnap_form.setVerticalSpacing(4)
        resnap_form.addRow("Min face (px)", self._resnap_min_face)
        resnap_form.addRow("Bias >= (px)", self._resnap_bias_gt)
        resnap_form.addRow("Border pad (px)", self._resnap_border_pad)
        resnap_form.addRow("Island pad (px)", self._resnap_island_pad)
        resnap_form.addRow("UV units", self._resnap_uv_units)
        resnap_form.addRow(self._btn_resnap)

        layout = QtWidgets.QVBoxLayout(w)
        layout.addWidget(model_box)
        layout.addWidget(tex_box)
        layout.addWidget(unwrap_box)
        layout.addWidget(el_box)
        layout.addWidget(view_box)
        layout.addWidget(self._btn_pack)
        layout.addWidget(self._btn_save)
        layout.addWidget(resnap_box)
        layout.addWidget(self._lbl_status)
        layout.addStretch(1)
        return w

    def _build_output(self) -> QtWidgets.QWidget:
        w = QtWidgets.QWidget()
        w.setMinimumWidth(520)

        self._txt_log = QtWidgets.QPlainTextEdit()
        self._txt_log.setReadOnly(True)
        self._txt_log.setMaximumBlockCount(20000)
        self._txt_log.setLineWrapMode(QtWidgets.QPlainTextEdit.LineWrapMode.NoWrap)
        self._txt_log.setFont(QtGui.QFont("Consolas", 10))

        self._txt_json = QtWidgets.QPlainTextEdit()
        self._txt_json.setReadOnly(True)
        self._txt_json.setMaximumBlockCount(20000)
        self._txt_json.setLineWrapMode(QtWidgets.QPlainTextEdit.LineWrapMode.NoWrap)
        self._txt_json.setFont(QtGui.QFont("Consolas", 10))

        tabs = QtWidgets.QTabWidget()
        tabs.addTab(self._txt_log, "Log")
        tabs.addTab(self._txt_json, "Updated JSON")

        box = QtWidgets.QGroupBox("Output")
        box_layout = QtWidgets.QVBoxLayout(box)
        box_layout.setContentsMargins(8, 6, 8, 6)
        box_layout.setSpacing(6)
        box_layout.addWidget(tabs)

        layout = QtWidgets.QVBoxLayout(w)
        layout.addWidget(box)
        layout.setContentsMargins(0, 0, 0, 0)
        return w

    def _on_load(self) -> None:
        path, _filter = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Open Minecraft model JSON",
            str(Path.cwd()),
            "JSON (*.json)",
        )
        if not path:
            return

        try:
            p = Path(path)
            model = json.loads(p.read_text(encoding="utf-8"))
            els = model.get("elements")
            if not isinstance(els, list):
                raise ValueError("JSON missing elements[]")

            elements = [_parse_element(i, el) for i, el in enumerate(els) if isinstance(el, dict)]

            self._source_path = p
            self._model = model
            self._elements = elements
            self._central_overrides = {}
            self._selected_element = None
            self._element_colors = [_color_for_index(i) for i in range(len(elements))]
            self._last_central_faces = [None] * len(elements)
            self._packed = []
            self._out_model = None

            self._txt_path.setText(str(p))
            self._btn_pack.setEnabled(len(elements) > 0)
            self._btn_resnap.setEnabled(len(elements) > 0)
            self._btn_save.setEnabled(False)

            self._txt_log.setPlainText(f"Loaded {len(elements)} elements")
            self._txt_json.setPlainText("")
            self._preview.set_preview(
                tex_w=int(self._tex_w.value()),
                tex_h=int(self._tex_h.value()),
                packed=[],
                colors_by_element={i: self._element_colors[i] for i in range(len(self._element_colors))},
                selected_element=self._selected_element,
            )

            self._lst_elements.clear()
            for el in elements:
                dx, dy, dz = el.cuboid.size()
                it = QtWidgets.QListWidgetItem(f"{el.idx:03d}  size=({dx:g},{dy:g},{dz:g})")
                col = self._element_colors[int(el.idx)] if 0 <= int(el.idx) < len(self._element_colors) else _color_for_index(int(el.idx))
                it.setBackground(QtGui.QBrush(QtGui.QColor(col.red(), col.green(), col.blue(), 55)))
                self._lst_elements.addItem(it)

            if self._lst_elements.count() > 0:
                self._lst_elements.setCurrentRow(0)

            self._sync_views()
            self._lbl_status.setText(f"Loaded: {p.name} ({len(elements)} elements)")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Load failed", str(e))

    def _on_pack(self) -> None:
        if self._model is None or not self._elements:
            return

        try:
            tex_w = int(self._tex_w.value())
            tex_h = int(self._tex_h.value())
            density = float(self._density.value())
            snap_px = int(self._snap_px.value())
            island_pad = int(self._island_pad.value())
            border_pad = int(self._border_pad.value())
            prefer_horizontal = bool(self._prefer_horizontal.isChecked())
            central_mode = str(self._central_mode.currentData() or "auto")
            default_texture = self._default_texture.text().strip() or "#0"
            write_rotation = bool(self._write_rotation.isChecked())

            islands: list[Island] = []
            lines: list[str] = []
            central_faces: list[Optional[Face]] = []
            lines.append(f"Texture: {tex_w}x{tex_h}")
            lines.append(f"Density: {density:g}")
            lines.append(f"Snap(px): {snap_px}")
            lines.append(f"Padding: island={island_pad}px border={border_pad}px")
            lines.append(f"Central mode: {central_mode}")
            lines.append(f"Prefer horizontal: {prefer_horizontal}")
            lines.append("")

            for el in self._elements:
                rects = _face_rects_px(c=el.cuboid, tex_w=tex_w, tex_h=tex_h, density=density, snap_px=snap_px)

                ov = self._central_overrides.get(int(el.idx))
                if ov is not None:
                    central = ov
                elif central_mode == "auto":
                    central = _choose_central_face(c=el.cuboid, prefer_horizontal=prefer_horizontal)
                else:
                    central = central_mode  # type: ignore[assignment]

                isl = _build_connected_box_island(element_idx=int(el.idx), face_rects=rects, central=central)  # type: ignore[arg-type]
                islands.append(isl)
                central_faces.append(str(central))

                w, h = isl.size()
                tag = " (override)" if ov is not None else ""
                lines.append(f"Element {el.idx:03d}: central={central}{tag} island={w}x{h}px")

            packed = _pack_islands_shelf(
                islands=islands,
                tex_w=tex_w,
                tex_h=tex_h,
                island_pad_px=island_pad,
                border_pad_px=border_pad,
                snap_px=snap_px,
            )

            out_model = _apply_uvs_to_model(
                model=self._model,
                packed=packed,
                tex_w=tex_w,
                tex_h=tex_h,
                default_texture=default_texture,
                write_rotation=write_rotation,
                uv_units="mc16",
            )

            self._packed = packed
            self._out_model = out_model
            self._last_central_faces = list(central_faces)

            self._preview.set_preview(
                tex_w=tex_w,
                tex_h=tex_h,
                packed=packed,
                colors_by_element={i: self._element_colors[i] for i in range(len(self._element_colors))},
                selected_element=self._selected_element,
            )
            if any(bool(p.overflow) for p in packed):
                lines.append("")
                lines.append("WARNING: Packing overflow (some islands outside texture bounds).")
                self._lbl_status.setText(f"Packed {len(packed)} islands (OVERFLOW)")

            self._txt_log.setPlainText("\n".join(lines))
            self._txt_json.setPlainText(format_minecraft_model_json(out_model))

            self._btn_save.setEnabled(True)
            if not any(bool(p.overflow) for p in packed):
                self._lbl_status.setText(f"Packed {len(packed)} islands")
            self._sync_views()
        except Exception as e:
            self._txt_log.setPlainText(f"Pack error: {e}")
            self._txt_json.setPlainText("")
            self._btn_save.setEnabled(False)
            self._lbl_status.setText("Pack error (see log)")

    def _on_save(self) -> None:
        if self._out_model is None:
            return

        default = Path.cwd() / "model_uv_packed.json"
        if self._source_path is not None:
            default = self._source_path.with_name(self._source_path.stem + "_UV_PACKED.json")

        out_path, _filter = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save updated model JSON",
            str(default),
            "JSON (*.json)",
        )
        if not out_path:
            return

        try:
            Path(out_path).write_text(format_minecraft_model_json(self._out_model) + "\n", encoding="utf-8", newline="\n")
            self._lbl_status.setText(f"Saved: {out_path}")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Save failed", str(e))

    def _on_resnap(self) -> None:
        if self._model is None or not self._elements:
            return

        try:
            tex_w = int(self._tex_w.value())
            tex_h = int(self._tex_h.value())
            min_face_px = int(self._resnap_min_face.value())
            bias_gt_px = int(self._resnap_bias_gt.value())
            border_pad_px = int(self._resnap_border_pad.value())
            island_pad_px = int(self._resnap_island_pad.value())
            uv_units_sel = str(self._resnap_uv_units.currentData() or "auto")
            uv_units_arg: Optional[str] = None if uv_units_sel == "auto" else uv_units_sel

            out_model, log, units = _resnap_model_uvs(
                model=self._model,
                tex_w=tex_w,
                tex_h=tex_h,
                min_face_px=min_face_px,
                bias_gt_px=bias_gt_px,
                border_pad_px=border_pad_px,
                island_pad_px=island_pad_px,
                uv_units=uv_units_arg,
            )

            self._out_model = out_model
            self._packed = _uvs_to_packed_preview(model=out_model, tex_w=tex_w, tex_h=tex_h, uv_units=units)

            self._preview.set_preview(
                tex_w=tex_w,
                tex_h=tex_h,
                packed=self._packed,
                colors_by_element={i: self._element_colors[i] for i in range(len(self._element_colors))},
                selected_element=self._selected_element,
            )

            self._txt_log.setPlainText("\n".join(log))
            self._txt_json.setPlainText(format_minecraft_model_json(out_model))

            self._btn_save.setEnabled(True)
            self._lbl_status.setText(f"Resnapped {len(self._packed)} elements to {tex_w}x{tex_h} (units={units})")
            self._sync_views()
        except Exception as e:
            self._txt_log.setPlainText(f"Resnap error: {e}")
            self._txt_json.setPlainText("")
            self._btn_save.setEnabled(False)
            self._lbl_status.setText("Resnap error (see log)")
#endregion


#region ENTRY
# Application entry point


def main() -> None:
    app = QtWidgets.QApplication(sys.argv)
    _apply_dark_theme(app)

    w = UVPackerMainWindow()
    w.resize(1400, 820)
    w.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
#endregion
