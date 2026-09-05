"""
Minecraft Model Greedy Optimizer 

Pure-logic module with no PySide6 dependency. Reduces the number of cuboid
`elements` in a Minecraft block-model by merging pairs that can be replaced
by a single cuboid, preserving the exact occupied volume.

Algorithm
1. Group cuboids by their rotation frame (axis, angle, origin). Cuboids in the
   same frame are axis-aligned in that shared local space. Unrotated cuboids
   form their own group.
2. Within each group, greedily merge pairs of axis-aligned boxes when their
   union is itself a single box. Two boxes satisfy this when:
   - One box contains the other (the inner box is redundant and is dropped), or
   - They have identical intervals on 2 axes and overlapping-or-touching
     intervals on the third (they combine into one larger box).
3. The merged box lives in the same local frame, so when exported with the
   group's rotation it is a single valid Minecraft cuboid element.
4. Volume is preserved by construction: every merge produces a box whose
   volume equals the union of the two inputs (no gaps are filled, no volume
   is added).

Rotated cuboids with different rotation frames cannot be merged and pass
through unchanged.

Output elements are geometry-only (from/to/rotation). Faces are irrelevant to
this optimization and are not emitted.

TOOLSGROUP::MODEL
SORTGROUP::7
SORTPRIORITY::76
STATUS::active
VERSION::20260316
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Optional

try:
    from mc_model_solver_ui import Cuboid, Rotation, format_minecraft_model_json
except Exception as e:  # pragma: no cover - import guard
    print(f"Missing dependency: mc_model_solver_ui.py ({e})")
    raise


def _r6(x: float) -> float:
    return round(float(x), 6)


@dataclass
class Element:
    idx: int
    raw: dict
    cuboid: Cuboid

    def center(self) -> tuple[float, float, float]:
        fx, fy, fz = self.cuboid.fr
        tx, ty, tz = self.cuboid.to
        return ((fx + tx) * 0.5, (fy + ty) * 0.5, (fz + tz) * 0.5)

    def size(self) -> tuple[float, float, float]:
        fx, fy, fz = self.cuboid.fr
        tx, ty, tz = self.cuboid.to
        return (abs(tx - fx), abs(ty - fy), abs(tz - fz))


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
            )

    cub = Cuboid(fr=fr, to=to, rotation=rot)
    return Element(idx=int(idx), raw=dict(raw), cuboid=cub)


# ---------------------------------------------------------------------------
# Rotation grouping
# ---------------------------------------------------------------------------

def _rotation_key(el: Element) -> Optional[tuple]:
    """Return a hashable key that identifies the rotation frame.

    None means no rotation (axis-aligned in world space).
    Cuboids sharing the same key are axis-aligned relative to each other.
    """
    r = el.cuboid.rotation
    if r is None or abs(float(r.angle)) < 1e-9:
        return None
    return (
        str(r.axis),
        _r6(float(r.angle)),
        _r6(float(r.origin[0])),
        _r6(float(r.origin[1])),
        _r6(float(r.origin[2])),
    )


def _rotation_raw(el: Element) -> Optional[dict]:
    """Return the raw rotation dict from the element, or None if unrotated."""
    r0 = el.raw.get("rotation")
    if not isinstance(r0, dict):
        return None
    ang = r0.get("angle", 0)
    if isinstance(ang, (int, float)) and abs(float(ang)) < 1e-9:
        return None
    return json.loads(json.dumps(r0))


# ---------------------------------------------------------------------------
# Box geometry helpers
# ---------------------------------------------------------------------------

def _aabb_volume(fr: tuple[float, float, float], to: tuple[float, float, float]) -> float:
    return abs((to[0] - fr[0]) * (to[1] - fr[1]) * (to[2] - fr[2]))


def _overlap_volume(
    a_fr: tuple[float, float, float],
    a_to: tuple[float, float, float],
    b_fr: tuple[float, float, float],
    b_to: tuple[float, float, float],
) -> float:
    dx = max(0.0, min(a_to[0], b_to[0]) - max(a_fr[0], b_fr[0]))
    dy = max(0.0, min(a_to[1], b_to[1]) - max(a_fr[1], b_fr[1]))
    dz = max(0.0, min(a_to[2], b_to[2]) - max(a_fr[2], b_fr[2]))
    return dx * dy * dz


def _contains(
    outer_fr: tuple[float, float, float],
    outer_to: tuple[float, float, float],
    inner_fr: tuple[float, float, float],
    inner_to: tuple[float, float, float],
    eps: float,
) -> bool:
    for ax in range(3):
        if inner_fr[ax] < outer_fr[ax] - eps:
            return False
        if inner_to[ax] > outer_to[ax] + eps:
            return False
    return True


def _intervals_identical(a_lo: float, a_hi: float, b_lo: float, b_hi: float, eps: float) -> bool:
    return abs(a_lo - b_lo) <= eps and abs(a_hi - b_hi) <= eps


def _intervals_overlap_or_touch(a_lo: float, a_hi: float, b_lo: float, b_hi: float, eps: float) -> bool:
    return a_lo <= b_hi + eps and b_lo <= a_hi + eps


# ---------------------------------------------------------------------------
# Merge logic
# ---------------------------------------------------------------------------

def _try_merge(
    a_fr: tuple[float, float, float],
    a_to: tuple[float, float, float],
    b_fr: tuple[float, float, float],
    b_to: tuple[float, float, float],
    eps: float,
) -> Optional[tuple[tuple[float, float, float], tuple[float, float, float]]]:
    """Try to merge two axis-aligned boxes into one.

    Returns the merged (fr, to) if possible, or None.

    Merge is possible when:
    - One box contains the other (return the larger box), or
    - The boxes share identical intervals on 2 axes and overlap-or-touch on
      the third (return the union box).

    The merged box always has volume equal to the union of the two inputs
    (no gaps are filled, no extra volume is added).
    """
    # containment: a contains b
    if _contains(a_fr, a_to, b_fr, b_to, eps):
        return a_fr, a_to
    # containment: b contains a
    if _contains(b_fr, b_to, a_fr, a_to, eps):
        return b_fr, b_to

    # adjacency merge: identical on 2 axes, overlap-or-touch on the third
    for merge_ax in range(3):
        other_axes = [i for i in range(3) if i != merge_ax]
        identical = all(
            _intervals_identical(a_fr[i], a_to[i], b_fr[i], b_to[i], eps)
            for i in other_axes
        )
        if not identical:
            continue
        if not _intervals_overlap_or_touch(a_fr[merge_ax], a_to[merge_ax], b_fr[merge_ax], b_to[merge_ax], eps):
            continue

        new_fr = list(a_fr)
        new_to = list(a_to)
        new_fr[merge_ax] = min(a_fr[merge_ax], b_fr[merge_ax])
        new_to[merge_ax] = max(a_to[merge_ax], b_to[merge_ax])
        return tuple(new_fr), tuple(new_to)

    return None


def _greedy_merge_group(
    boxes: list[tuple[tuple[float, float, float], tuple[float, float, float]]],
    eps: float,
) -> list[tuple[tuple[float, float, float], tuple[float, float, float]]]:
    """Greedily merge boxes within a single rotation group until stable."""
    current = list(boxes)

    changed = True
    while changed:
        changed = False
        n = len(current)
        for i in range(n):
            for j in range(i + 1, n):
                merged = _try_merge(current[i][0], current[i][1], current[j][0], current[j][1], eps)
                if merged is not None:
                    current.pop(j)
                    current.pop(i)
                    current.append(merged)
                    changed = True
                    break
            if changed:
                break

    return current


# ---------------------------------------------------------------------------
# Overlap reduction
# ---------------------------------------------------------------------------
#
# Two cuboids that overlap share volume that is rendered twice (z-fighting /
# wasted geometry). When box A can be trimmed to a single smaller box A' such
# that the removed portion of A is entirely covered by box B (A \\ A' ⊆ B),
# replacing A with A' preserves the external (union) volume exactly while
# eliminating the A∩B overlap.
#
# A \\ (A∩B) is a single box iff, on exactly one axis, A extends past the
# overlap interval on a single side, and on the other two axes A coincides
# with the overlap (i.e. B contains A on those two axes). If A coincides with
# the overlap on all three axes then A ⊆ B and A can be dropped entirely.

def _try_trim(
    a_fr: tuple[float, float, float],
    a_to: tuple[float, float, float],
    b_fr: tuple[float, float, float],
    b_to: tuple[float, float, float],
    eps: float,
) -> Optional[tuple[str, Optional[tuple[tuple[float, float, float], tuple[float, float, float]]]]]:
    """Try to trim box A so it no longer overlaps box B, preserving the union.

    Returns:
      ("remove", None)  -- A is fully inside B; A can be deleted.
      ("trim", (fr,to)) -- A can be replaced by the single box A' = A \\ (A∩B).
      None              -- no single-box trim preserves the union.
    """
    exts: list[tuple[int, str]] = []
    for ax in range(3):
        ov_lo = max(a_fr[ax], b_fr[ax])
        ov_hi = min(a_to[ax], b_to[ax])
        if ov_hi <= ov_lo + eps:
            # no overlap on this axis -> boxes are disjoint -> nothing to trim
            return None

        lo_ext = a_fr[ax] < ov_lo - eps
        hi_ext = a_to[ax] > ov_hi + eps
        if lo_ext and hi_ext:
            # A extends past the overlap on both sides -> A\\(A∩B) is two boxes
            return None
        if lo_ext:
            exts.append((ax, "lo"))
        elif hi_ext:
            exts.append((ax, "hi"))

    if len(exts) == 0:
        # A == A∩B  ->  A ⊆ B  ->  A is redundant
        return ("remove", None)
    if len(exts) != 1:
        # would need more than one box to represent A \\ (A∩B)
        return None

    ax, side = exts[0]
    ov_lo = max(a_fr[ax], b_fr[ax])
    ov_hi = min(a_to[ax], b_to[ax])
    new_fr = list(a_fr)
    new_to = list(a_to)
    if side == "lo":
        new_to[ax] = ov_lo  # A' = [a_lo, ov_lo] on this axis
    else:
        new_fr[ax] = ov_hi  # A' = [ov_hi, a_hi] on this axis
    return ("trim", (tuple(new_fr), tuple(new_to)))


def _total_overlap_volume(
    boxes: list[tuple[tuple[float, float, float], tuple[float, float, float]]],
) -> float:
    """Sum of pairwise overlap volumes (counts each shared region per pair)."""
    total = 0.0
    n = len(boxes)
    for i in range(n):
        for j in range(i + 1, n):
            total += _overlap_volume(boxes[i][0], boxes[i][1], boxes[j][0], boxes[j][1])
    return total


def _reduce_overlaps_group(
    boxes: list[tuple[tuple[float, float, float], tuple[float, float, float]]],
    eps: float,
) -> list[tuple[tuple[float, float, float], tuple[float, float, float]]]:
    """Greedily trim boxes against each other to remove overlapping volume.

    Every trim preserves the union volume of the group by construction
    (the trimmed-away part of A is always fully covered by the box B it was
    trimmed against). Iterates until no further single-box trim is possible.
    """
    current = list(boxes)

    changed = True
    while changed:
        changed = False
        n = len(current)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                res = _try_trim(current[i][0], current[i][1], current[j][0], current[j][1], eps)
                if res is None:
                    continue
                kind, payload = res
                if kind == "remove":
                    current.pop(i)
                else:
                    current[i] = payload  # type: ignore[index]
                changed = True
                break
            if changed:
                break
            # n may have shrunk; recompute for the outer loop guard
            n = len(current)

    return current


# ---------------------------------------------------------------------------
# Output element construction
# ---------------------------------------------------------------------------

def _dominant_element(
    box_fr: tuple[float, float, float],
    box_to: tuple[float, float, float],
    elements: list[Element],
) -> Optional[int]:
    """Return the index of the input element that overlaps the box the most."""
    best_i: Optional[int] = None
    best_ov = 0.0
    for i, el in enumerate(elements):
        ov = _overlap_volume(box_fr, box_to, el.cuboid.fr, el.cuboid.to)
        if ov > best_ov:
            best_ov = ov
            best_i = i
    return best_i


def _make_element_dict(
    fr: tuple[float, float, float],
    to: tuple[float, float, float],
    rotation_raw: Optional[dict],
    donor: Optional[Element],
) -> dict:
    """Build an output element dict, donating faces/name/shade from the donor."""
    d: dict = {}

    if donor is not None:
        name = donor.raw.get("name")
        if isinstance(name, str) and name:
            d["name"] = str(name)

    d["from"] = [_r6(fr[0]), _r6(fr[1]), _r6(fr[2])]
    d["to"] = [_r6(to[0]), _r6(to[1]), _r6(to[2])]

    if rotation_raw is not None:
        d["rotation"] = json.loads(json.dumps(rotation_raw))

    if donor is not None:
        if isinstance(donor.raw.get("shade"), bool):
            d["shade"] = bool(donor.raw["shade"])
        if isinstance(donor.raw.get("faces"), dict):
            d["faces"] = json.loads(json.dumps(donor.raw["faces"]))

    return d


# ---------------------------------------------------------------------------
# Tab-based JSON formatter (Blockbench style)
# ---------------------------------------------------------------------------

def format_blockbench_json(obj: dict) -> str:
    """Format a Minecraft model dict with tab indentation matching Blockbench output.

    Key order is preserved (Python dicts maintain insertion order).
    Small dicts and lists of scalars are inlined when they fit on one line.
    """
    def is_scalar(v) -> bool:
        return v is None or isinstance(v, (str, int, float, bool))

    def try_inline(obj) -> Optional[str]:
        if isinstance(obj, list):
            if all(is_scalar(x) for x in obj) and len(obj) <= 16:
                s = json.dumps(obj, ensure_ascii=False, separators=(", ", ": "))
                if len(s) <= 80:
                    return s
            return None

        if isinstance(obj, dict):
            if not obj:
                return "{}"
            if all(isinstance(k, str) for k in obj.keys()):
                # Don't inline dicts where all values are strings (like textures).
                # Blockbench keeps these multi-line. Dicts with at least one
                # non-string value (lists, ints, bools) are inlined (like face
                # entries, rotation dicts).
                all_string_values = all(isinstance(v, str) for v in obj.values())
                if all_string_values:
                    return None

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
                    s = json.dumps(obj, ensure_ascii=False, separators=(", ", ": "))
                    if len(s) <= 100:
                        return s
            return None

        if is_scalar(obj):
            return json.dumps(obj, ensure_ascii=False)

        return None

    def fmt(obj, level: int) -> str:
        inline = try_inline(obj)
        if inline is not None:
            return inline

        ind = "\t" * level
        ind2 = "\t" * (level + 1)

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

    return fmt(obj, 0)


# ---------------------------------------------------------------------------
# Main optimization
# ---------------------------------------------------------------------------

@dataclass
class GroupStats:
    rotation_key: Optional[tuple]
    input_count: int
    output_count: int
    input_volume: float
    output_volume: float


@dataclass
class OptimizeResult:
    model: dict
    input_count: int
    output_count: int
    group_count: int
    input_volume: float
    output_volume: float
    groups: list[GroupStats]
    log: list[str]
    # Per-input-element status: "removed" (consumed by a merge) or "untouched"
    input_status: list[str]
    # Per-output-element status: "new" (result of a merge) or "untouched"
    output_status: list[str]


def _element_signature(el: Element) -> tuple:
    """Return a hashable signature for exact geometry comparison."""
    fr = tuple(_r6(v) for v in el.cuboid.fr)
    to = tuple(_r6(v) for v in el.cuboid.to)
    rot = el.cuboid.rotation
    if rot is None or abs(float(rot.angle)) < 1e-9:
        rot_sig = None
    else:
        rot_sig = (
            str(rot.axis),
            _r6(float(rot.angle)),
            _r6(float(rot.origin[0])),
            _r6(float(rot.origin[1])),
            _r6(float(rot.origin[2])),
        )
    return (fr, to, rot_sig)


def _classify_changes(
    input_elements: list[Element],
    output_elements: list[Element],
) -> tuple[list[str], list[str]]:
    """Classify each input as 'removed' or 'untouched', each output as 'new' or 'untouched'.

    An output element is 'untouched' if its geometry signature exactly matches
    an input element's signature. An input element is 'untouched' if it matches
    an output element. Each input/output is matched at most once.
    """
    input_sigs = [_element_signature(el) for el in input_elements]
    output_sigs = [_element_signature(el) for el in output_elements]

    input_status = ["removed"] * len(input_elements)
    output_status = ["new"] * len(output_elements)

    # match outputs to inputs by signature, each used at most once
    used_inputs: set[int] = set()
    for oi, osig in enumerate(output_sigs):
        for ii, isig in enumerate(input_sigs):
            if ii in used_inputs:
                continue
            if osig == isig:
                output_status[oi] = "untouched"
                input_status[ii] = "untouched"
                used_inputs.add(ii)
                break

    return input_status, output_status


def _optimize_model(
    model: dict,
    elements: list[Element],
    *,
    eps: float,
    reduce_overlaps: bool = False,
) -> OptimizeResult:
    # Group by rotation frame
    groups: dict[Optional[tuple], dict] = {}
    group_order: list[Optional[tuple]] = []

    for el in elements:
        key = _rotation_key(el)
        if key not in groups:
            groups[key] = {
                "rotation_raw": _rotation_raw(el),
                "boxes": [],
                "elements": [],
            }
            group_order.append(key)
        groups[key]["boxes"].append((el.cuboid.fr, el.cuboid.to))
        groups[key]["elements"].append(el)

    out_elements: list[dict] = []
    group_stats: list[GroupStats] = []
    total_input_vol = 0.0
    total_output_vol = 0.0
    total_input_overlap = 0.0
    total_output_overlap = 0.0

    log: list[str] = []
    log.append(f"Input elements: {len(elements)}")
    log.append(f"Rotation groups: {len(groups)}")
    if reduce_overlaps:
        log.append("Mode: overlap reduction (union volume preserved)")
    log.append("")

    for gi, key in enumerate(group_order):
        group = groups[key]
        boxes = group["boxes"]
        rot_raw = group["rotation_raw"]

        input_vol = sum(_aabb_volume(fr, to) for fr, to in boxes)
        total_input_vol += input_vol
        input_overlap = _total_overlap_volume(boxes)
        total_input_overlap += input_overlap

        if reduce_overlaps:
            reduced = _reduce_overlaps_group(boxes, eps)
        else:
            reduced = boxes
        merged = _greedy_merge_group(reduced, eps)

        output_vol = sum(_aabb_volume(fr, to) for fr, to in merged)
        total_output_vol += output_vol
        output_overlap = _total_overlap_volume(merged)
        total_output_overlap += output_overlap

        for fr, to in merged:
            donor_i = _dominant_element(fr, to, group["elements"])
            donor = group["elements"][donor_i] if donor_i is not None else None
            out_elements.append(_make_element_dict(fr, to, rot_raw, donor))

        label = "axis-aligned" if key is None else f"rot {key[0]} {key[1]}deg"
        log.append(
            f"Group {gi} ({label}): {len(boxes)} -> {len(merged)} boxes, "
            f"volume {input_vol:.4f} -> {output_vol:.4f}"
        )
        if reduce_overlaps:
            log.append(
                f"  overlap {input_overlap:.6f} -> {output_overlap:.6f}"
            )

        group_stats.append(GroupStats(
            rotation_key=key,
            input_count=len(boxes),
            output_count=len(merged),
            input_volume=input_vol,
            output_volume=output_vol,
        ))

    out_model = json.loads(json.dumps(model))
    out_model["elements"] = out_elements

    log.append("")
    log.append(f"Total: {len(elements)} -> {len(out_elements)} elements")
    log.append(f"Total volume: {total_input_vol:.6f} -> {total_output_vol:.6f}")
    vol_delta = total_output_vol - total_input_vol
    if abs(vol_delta) < 1e-9:
        log.append("Volume preserved: exact")
    else:
        log.append(f"Volume delta: {vol_delta:+.6f}")

    if reduce_overlaps:
        log.append(
            f"Total overlap: {total_input_overlap:.6f} -> {total_output_overlap:.6f}"
        )
        ov_delta = total_input_overlap - total_output_overlap
        if total_input_overlap > 1e-9:
            ov_pct = 100.0 * ov_delta / total_input_overlap
            log.append(f"Overlap removed: {ov_delta:.6f} ({ov_pct:.1f}%)")
        elif abs(ov_delta) < 1e-9:
            log.append("Overlap removed: none (no overlaps present)")

    reduction = len(elements) - len(out_elements)
    if len(elements) > 0:
        pct = 100.0 * reduction / len(elements)
        log.append(f"Element reduction: {reduction} ({pct:.1f}%)")

    # classify changes for visualization
    out_els_parsed = [_parse_element(i, el) for i, el in enumerate(out_elements)]
    input_status, output_status = _classify_changes(elements, out_els_parsed)

    removed_count = sum(1 for s in input_status if s == "removed")
    new_count = sum(1 for s in output_status if s == "new")
    untouched_in = sum(1 for s in input_status if s == "untouched")
    untouched_out = sum(1 for s in output_status if s == "untouched")
    log.append("")
    log.append(f"Changes: {removed_count} removed, {new_count} new, {untouched_in} untouched")

    return OptimizeResult(
        model=out_model,
        input_count=len(elements),
        output_count=len(out_elements),
        group_count=len(groups),
        input_volume=total_input_vol,
        output_volume=total_output_vol,
        groups=group_stats,
        log=log,
        input_status=input_status,
        output_status=output_status,
    )
