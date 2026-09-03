"""OBJ Geometry Diff (CEM Debug Tool)

Compare two Wavefront `.obj` files and report per-object geometry differences.
This is primarily used to validate Optifine CEM / Blockbench exports and to debug
coordinate system, pivot, and rotation issues by comparing:
- Object presence (missing/extra)
- Bounding boxes, extents, and centroids
- Optional exact vertex-set equivalence checks (including translation/rotation detection)

TOOLSGROUP::CEM
SORTGROUP::7
SORTPRIORITY::71
STATUS::active
VERSION::20260308

Expected inputs
- `--a`: Path to OBJ A
- `--b`: Path to OBJ B

Outputs
- Prints a summary and per-object diff report to stdout

Usage
- Simple:
  python DEV/cem_obj_geom_diff.py --a a.obj --b b.obj --only-diff

- Project example (compare resourcepack CEM OBJ exports):
  python DEV/cem_obj_geom_diff.py \
    --a RESOURCEPACK/00_OBORO_20260217/assets/minecraft/optifine/cem/iron_golem.obj \
    --b RESOURCEPACK/00_OBORO_20260217/assets/minecraft/optifine/cem/iron_golem_fromblockbench.obj \
    --only-diff --vertex-exact
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def _r6(x: float) -> float:
    return round(float(x), 6)


@dataclass(frozen=True)
class ObjObject:
    name: str
    verts: List[Tuple[float, float, float]]

    def bbox(self) -> Tuple[float, float, float, float, float, float]:
        xs = [v[0] for v in self.verts]
        ys = [v[1] for v in self.verts]
        zs = [v[2] for v in self.verts]
        return (_r6(min(xs)), _r6(min(ys)), _r6(min(zs)), _r6(max(xs)), _r6(max(ys)), _r6(max(zs)))

    def centroid(self) -> Tuple[float, float, float]:
        xs = [v[0] for v in self.verts]
        ys = [v[1] for v in self.verts]
        zs = [v[2] for v in self.verts]
        n = float(len(self.verts))
        return (_r6(sum(xs) / n), _r6(sum(ys) / n), _r6(sum(zs) / n))

    def extents(self) -> Tuple[float, float, float]:
        b = self.bbox()
        return (_r6(b[3] - b[0]), _r6(b[4] - b[1]), _r6(b[5] - b[2]))


@dataclass(frozen=True)
class ObjData:
    objects: Dict[str, ObjObject]


def _parse_obj(path: Path) -> ObjData:
    txt = Path(path).read_text(encoding="utf-8", errors="replace").splitlines()

    objects: Dict[str, List[Tuple[float, float, float]]] = {}
    cur_name: Optional[str] = None

    for line in txt:
        if not line:
            continue
        if line.startswith("o "):
            cur_name = line[2:].strip()
            if cur_name not in objects:
                objects[cur_name] = []
            continue
        if line.startswith("v "):
            if cur_name is None:
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            try:
                x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
            except Exception:
                continue
            objects[cur_name].append((_r6(x), _r6(y), _r6(z)))

    out = {name: ObjObject(name=name, verts=vs) for name, vs in objects.items() if vs}
    return ObjData(objects=out)


def _fmt3(v: Tuple[float, float, float]) -> str:
    return f"({_r6(v[0])}, {_r6(v[1])}, {_r6(v[2])})"


def _fmt6(v: Tuple[float, float, float, float, float, float]) -> str:
    return f"({_r6(v[0])}, {_r6(v[1])}, {_r6(v[2])}, {_r6(v[3])}, {_r6(v[4])}, {_r6(v[5])})"


def _sub3(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> Tuple[float, float, float]:
    return (_r6(a[0] - b[0]), _r6(a[1] - b[1]), _r6(a[2] - b[2]))


def _add3(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> Tuple[float, float, float]:
    return (_r6(a[0] + b[0]), _r6(a[1] + b[1]), _r6(a[2] + b[2]))


def _vertex_set(vs: List[Tuple[float, float, float]]) -> set:
    return set(vs)


def _apply_rot(v: Tuple[float, float, float], rot: Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]) -> Tuple[float, float, float]:
    x, y, z = v
    src = (x, y, z)
    ax0, s0 = rot[0]
    ax1, s1 = rot[1]
    ax2, s2 = rot[2]
    return (_r6(src[ax0] * s0), _r6(src[ax1] * s1), _r6(src[ax2] * s2))


def _det3(rot: Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]) -> int:
    a0, s0 = rot[0]
    a1, s1 = rot[1]
    a2, s2 = rot[2]

    p = (a0, a1, a2)
    perm_sign = 0
    if p in {(0, 1, 2), (1, 2, 0), (2, 0, 1)}:
        perm_sign = 1
    elif p in {(0, 2, 1), (2, 1, 0), (1, 0, 2)}:
        perm_sign = -1
    else:
        perm_sign = 0
    return int(perm_sign * s0 * s1 * s2)


def _all_right_handed_rots() -> List[Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]]:
    axes = [0, 1, 2]
    signs = [-1, 1]
    out: List[Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]] = []
    for a0 in axes:
        for a1 in axes:
            if a1 == a0:
                continue
            for a2 in axes:
                if a2 == a0 or a2 == a1:
                    continue
                for s0 in signs:
                    for s1 in signs:
                        for s2 in signs:
                            rot = ((a0, s0), (a1, s1), (a2, s2))
                            if _det3(rot) == 1:
                                out.append(rot)
    uniq: Dict[Tuple, Tuple] = {}
    for r in out:
        uniq[tuple(r)] = r
    return list(uniq.values())


def _vertex_exact_report(
    *,
    oa: ObjObject,
    ob: ObjObject,
    eps: float,
) -> List[str]:
    lines: List[str] = []
    va = list(oa.verts)
    vb = list(ob.verts)
    sa = _vertex_set(va)
    sb = _vertex_set(vb)

    if sa == sb:
        lines.append("  vertices_exact: identical")
        return lines

    ca = oa.centroid()
    cb = ob.centroid()
    t = _sub3(ca, cb)
    vb_t = [_add3(v, t) for v in vb]
    if sa == _vertex_set(vb_t):
        lines.append(f"  vertices_exact: translation t={_fmt3(t)}")
        return lines

    ra = [(_r6(v[0] - ca[0]), _r6(v[1] - ca[1]), _r6(v[2] - ca[2])) for v in va]
    rb = [(_r6(v[0] - cb[0]), _r6(v[1] - cb[1]), _r6(v[2] - cb[2])) for v in vb]
    sra = _vertex_set(ra)

    best = None
    rots = _all_right_handed_rots()
    for rot in rots:
        rb_r = [_apply_rot(v, rot) for v in rb]
        if sra == _vertex_set(rb_r):
            best = rot
            break

    if best is not None:
        rb_r = [_apply_rot(v, best) for v in rb]
        vb_rt = [_add3(_add3(v, ca), (0.0, 0.0, 0.0)) for v in rb_r]
        vb_rt2 = [_add3(v, (0.0, 0.0, 0.0)) for v in vb_rt]
        _ = vb_rt2
        lines.append(f"  vertices_exact: rotation+translation around centroids rot={best} ca={_fmt3(ca)} cb={_fmt3(cb)}")
        return lines

    only_in_a = sa - sb
    only_in_b = sb - sa
    lines.append(f"  vertices_exact: mismatch a_only={len(only_in_a)} b_only={len(only_in_b)}")
    return lines


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, type=Path)
    ap.add_argument("--b", required=True, type=Path)
    ap.add_argument("--eps", type=float, default=1e-6)
    ap.add_argument("--only-diff", action="store_true")
    ap.add_argument("--vertex-exact", action="store_true")
    args = ap.parse_args()

    a = _parse_obj(args.a)
    b = _parse_obj(args.b)

    names_a = set(a.objects)
    names_b = set(b.objects)
    common = sorted(names_a & names_b)

    missing_a = sorted(names_b - names_a)
    missing_b = sorted(names_a - names_b)

    print(f"objects_a={len(names_a)} objects_b={len(names_b)} common={len(common)} missing_in_a={len(missing_a)} missing_in_b={len(missing_b)}")
    if missing_a:
        print("\nMissing in A")
        for n in missing_a:
            print(f"  {n}")
    if missing_b:
        print("\nMissing in B")
        for n in missing_b:
            print(f"  {n}")

    eps = float(args.eps)

    diffs = 0
    print("\nPer-object diffs")
    for n in common:
        oa = a.objects[n]
        ob = b.objects[n]
        ca = oa.centroid()
        cb = ob.centroid()
        ba = oa.bbox()
        bb = ob.bbox()
        ea = oa.extents()
        eb = ob.extents()

        dc = _sub3(ca, cb)
        de = _sub3(ea, eb)

        any_diff = (
            any(abs(x) > eps for x in dc)
            or any(abs(x) > eps for x in de)
            or any(abs(ba[i] - bb[i]) > eps for i in range(6))
        )

        if args.only_diff and not any_diff:
            continue

        print(f"\n[{n}]")
        print(f"  n_verts a={len(oa.verts)} b={len(ob.verts)}")
        print(f"  centroid a={_fmt3(ca)} b={_fmt3(cb)} d(a-b)={_fmt3(dc)}")
        print(f"  extents  a={_fmt3(ea)} b={_fmt3(eb)} d(a-b)={_fmt3(de)}")
        print(f"  bbox     a={_fmt6(ba)}")
        print(f"           b={_fmt6(bb)}")

        if args.vertex_exact:
            for line in _vertex_exact_report(oa=oa, ob=ob, eps=eps):
                print(line)

        if any_diff:
            diffs += 1

    print(f"\nsummary_different_objects={diffs}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
