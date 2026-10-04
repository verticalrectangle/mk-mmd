"""Automatic prop cards (docs/design.md: PMX props): what a card says about an imported prop, from its geometry. numpy only,
so the Blender side (mkmmd.blender.build.pmxprop) only gathers arrays and the tests run without Blender.

Inputs are plain arrays in the prop's frame (metres, Z up): triangles `tris` (M, 3, 3), rigid bodies as dicts, and per
bone vertex counts. Outputs are JSON-able dicts; collider shapes come back as geometry (the caller turns them into hidden
objects and card specs)."""
import math

import numpy as np

ORIGINS = ("floor_center", "center", "keep")
MIN_THICKNESS = 0.01          # a box collider is never thinner than this (a cloth or a card has no thickness)
TOP_SHARE = 0.4               # share of the occupied footprint a level must cover to be "the top face"
TOP_NEAR = (0.02, 0.12)       # the top face is within max(a, b * height) of the prop's highest point
GRID = 96                     # cells along the longer side of the footprint
BODY_STATIC = 0               # PMX rigid body modes: 0 follows its bone, 1 dynamic, 2 dynamic with bone position


def as_tris(tris):
    t = np.asarray(tris, float).reshape(-1, 3, 3)
    if not len(t):
        raise ValueError("the prop has no triangles")
    return t


def bounds(tris):
    """(lo, hi) of triangle corners."""
    t = as_tris(tris)
    return t.reshape(-1, 3).min(0), t.reshape(-1, 3).max(0)


def origin_shift(lo, hi, origin="floor_center"):
    """Translation that puts the origin where `origin` says: `floor_center` (centre of the footprint, lowest point at
    z = 0), `center` (centre of the bounds) or `keep` (the author's)."""
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    if origin == "keep":
        return np.zeros(3)
    c = (lo + hi) / 2.0
    if origin == "center":
        return -c
    if origin == "floor_center":
        return np.array([-c[0], -c[1], -lo[2]])
    raise ValueError(f"origin must be one of {ORIGINS}, got {origin!r}")


def sample_points(tris, spacing):
    """Points on the triangles: every triangle is cut by midpoints until its edges are at most `spacing`, then its
    corners and centroid are taken (deterministic, vectorised)."""
    t = as_tris(tris)
    out = []
    for _ in range(9):
        e = np.linalg.norm(t[:, [1, 2, 0]] - t, axis=2).max(1)
        big = e > spacing
        small = t[~big]
        if len(small):
            out.append(small.reshape(-1, 3))
            out.append(small.mean(1))
        if not big.any():
            break
        a, b, c = t[big][:, 0], t[big][:, 1], t[big][:, 2]
        ab, bc, ca = (a + b) / 2, (b + c) / 2, (c + a) / 2
        t = np.concatenate([np.stack([a, ab, ca], 1), np.stack([ab, b, bc], 1), np.stack([ca, bc, c], 1),
                            np.stack([ab, bc, ca], 1)])
    else:
        out.append(t.reshape(-1, 3))
    return np.concatenate(out)


def top_surface(tris, share=TOP_SHARE, near=TOP_NEAR, grid=GRID):
    """The prop's top face when it is roughly flat-topped, else None. A height map (highest z over each cell of the
    footprint) is cut into levels; the top face is the highest level that covers at least `share` of the occupied cells
    and lies within max(near[0], near[1] * height) of the prop's highest point. Returns {z, center [x, y], size [w, d],
    radius (round tops only), fill (share of the size rectangle it covers), share (of the occupied footprint)}."""
    t = as_tris(tris)
    lo, hi = t.reshape(-1, 3).min(0), t.reshape(-1, 3).max(0)
    span = np.maximum(hi[:2] - lo[:2], 1e-6)
    cell = float(np.clip(span.max() / grid, 0.003, 0.05))
    pts = sample_points(t, cell * 0.7)
    n = np.maximum(np.ceil(span / cell).astype(int), 1)
    ix = np.clip(((pts[:, 0] - lo[0]) / cell).astype(int), 0, n[0] - 1)
    iy = np.clip(((pts[:, 1] - lo[1]) / cell).astype(int), 0, n[1] - 1)
    H = np.full(tuple(n), -np.inf)
    np.maximum.at(H, (ix, iy), pts[:, 2])
    occ = np.isfinite(H)
    cells = int(occ.sum())
    height = float(hi[2] - lo[2])
    tol = 0.006 + 0.01 * height
    h = np.sort(H[occ])[::-1]
    count = np.searchsorted(-h, -(h - tol), side="right") - np.arange(cells)
    ok = np.nonzero(count >= share * cells)[0]
    if not len(ok):
        return None
    z_top = float(h[ok[0]])
    if hi[2] - z_top > max(near[0], near[1] * height) + 1e-9:
        return None
    mask = occ & (H >= z_top - tol)
    xs, ys = np.nonzero(mask)
    x0, x1 = lo[0] + xs.min() * cell, lo[0] + (xs.max() + 1) * cell
    y0, y1 = lo[1] + ys.min() * cell, lo[1] + (ys.max() + 1) * cell
    w, d = float(x1 - x0), float(y1 - y0)
    fill = float(mask.sum() * cell * cell / max(w * d, 1e-12))
    out = {"z": float(H[mask].mean()), "center": [float((x0 + x1) / 2), float((y0 + y1) / 2)],
           "size": [round(w, 4), round(d, 4)], "fill": round(min(fill, 1.0), 3),
           "share": round(float(mask.sum() / cells), 3)}
    if 0.7 <= fill <= 0.86 and abs(w - d) <= 0.1 * max(w, d):
        out["radius"] = round((w + d) / 4.0, 4)
    return out


def moving_parts(parents, counts, min_verts=8, min_share=0.01):
    """The bones that move part of the prop: bones carrying vertices (at least `min_verts` and `min_share` of all
    weighted vertices) other than the single base bone. `parents` {bone: parent bone or None}, `counts` {bone: number of
    vertices the bone moves most}. One carrying bone (or none) = a rigid prop = []; one top-most carrier = the base and
    every other carrier is a part; several top-most carriers (siblings under an unweighted root) are all parts."""
    total = sum(counts.values())
    carriers = {b for b, n in counts.items() if n >= min_verts and n >= min_share * total}
    if len(carriers) <= 1:
        return []

    def top(b):
        p = parents.get(b)
        while p is not None:
            if p in carriers:
                return False
            p = parents.get(p)
        return True
    tops = [b for b in carriers if top(b)]
    base = set(tops) if len(tops) == 1 else set()
    return sorted(carriers - base, key=lambda b: (-counts[b], b))


def shapes_from_bodies(bodies, static_only=True):
    """Collider shapes from PMX rigid bodies. A body = {shape: SPHERE | CAPSULE | BOX, mode (0 static, 1 dynamic, 2
    dynamic + bone), center [x, y, z], rot (3x3, columns = the body's axes in the prop frame), half [hx, hy, hz]
    (half extents of its bounds along those axes)}. Bodies that follow their bone (mode 0) are the collision proxies of
    the prop; dynamic ones are soft parts (cords, ears) and are skipped. Returns [{kind: sphere | capsule | box, ...}]
    with sphere (c, R), capsule (a, b, R), box (center, rot, half)."""
    out = []
    for b in bodies:
        if static_only and int(b.get("mode", 0)) != BODY_STATIC:
            continue
        c = np.asarray(b["center"], float)
        R = np.asarray(b.get("rot", np.eye(3)), float).reshape(3, 3)
        half = np.maximum(np.asarray(b["half"], float), 1e-4)
        shape = str(b["shape"]).upper()
        if shape == "SPHERE":
            out.append({"kind": "sphere", "c": c.tolist(), "R": float(half.max())})
        elif shape == "CAPSULE":
            ax = int(np.argmax(half))
            rad = float(np.delete(half, ax).max())
            e = R[:, ax] * max(float(half[ax]) - rad, 0.0)
            out.append({"kind": "capsule", "a": (c - e).tolist(), "b": (c + e).tolist(), "R": rad})
        else:
            out.append({"kind": "box", "center": c.tolist(), "rot": R.tolist(), "half": half.tolist()})
    return out


def box_shape(lo, hi):
    """One box over the bounds (never thinner than MIN_THICKNESS)."""
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    half = np.maximum((hi - lo) / 2.0, MIN_THICKNESS / 2.0)
    return {"kind": "box", "center": ((lo + hi) / 2.0).tolist(), "rot": np.eye(3).tolist(), "half": half.tolist()}


def _r(v, n=4):
    return [round(float(x), n) for x in v]


def make_card(name, source, tris, bodies=None, parts=None, armature=None, front="-Y", origin="floor_center"):
    """The automatic card of an imported prop, from its triangles in the prop's frame (already shifted to `origin`):
    size and bounds, a `look` point at the centre, a `rest` plane `top` when it is roughly flat-topped, collider shapes
    from its static rigid bodies (else one box over the bounds), the moving `parts` and the armature that drives them.
    Returns (card, shapes); the card's `colliders` is empty until the caller has made objects for the shapes."""
    t = as_tris(tris)
    lo, hi = bounds(t)
    use = {"look": [{"name": "center", "point": _r((lo + hi) / 2.0)}]}
    top = top_surface(t)
    stats = {"triangles": int(len(t))}
    if top is not None:
        plane = {"name": "top", "type": "plane", "center": [top["center"][0], top["center"][1], round(top["z"], 4)],
                 "normal": [0, 0, 1]}
        if "radius" in top:
            plane["radius"] = top["radius"]
        else:
            plane["size"] = top["size"]
        use["rest"] = [plane]
        stats["top"] = {k: top[k] for k in ("fill", "share")}
    shapes = shapes_from_bodies(bodies or [])
    stats["rigid_bodies"] = {"static": len(shapes), "skipped": len(bodies or []) - len(shapes)}
    source_of = "rigid bodies"
    if not shapes:
        shapes, source_of = [box_shape(lo, hi)], "bounds"
    stats["colliders_from"] = source_of
    card = {"schema": 1, "name": name, "kind": "pmx", "source": source, "size": _r(hi - lo), "origin": origin,
            "front": front, "bounds": {"min": _r(lo), "max": _r(hi)}, "slots": {}, "use": use, "colliders": [],
            "parts": list(parts or []), "armature": armature, "stats": stats}
    return card, shapes


def merge_extra(card, extra):
    """`extra` merged into a copy of `card`: tables merge key by key; lists of entries with a `name` merge by name (an
    entry with a known name replaces it, a new name is appended, an empty list clears); everything else replaces."""
    if not extra:
        return card

    def merge(a, b):
        if isinstance(a, dict) and isinstance(b, dict):
            out = _copy(a)
            for k, v in b.items():
                out[k] = merge(a[k], v) if k in a else _copy(v)
            return out
        if isinstance(a, list) and isinstance(b, list) and b and all(isinstance(x, dict) and "name" in x for x in b) \
                and all(isinstance(x, dict) and "name" in x for x in a):
            out = [_copy(x) for x in a]
            for item in b:
                for i, x in enumerate(out):
                    if x["name"] == item["name"]:
                        out[i] = _copy(item)
                        break
                else:
                    out.append(_copy(item))
            return out
        return _copy(b)

    return merge(card, extra)


def _copy(v):
    if isinstance(v, dict):
        return {k: _copy(x) for k, x in v.items()}
    if isinstance(v, list):
        return [_copy(x) for x in v]
    return v


def describe_shape(s):
    """One line for logs: 'capsule R 0.012' ..."""
    if s["kind"] == "sphere":
        return f"sphere R {s['R']:.3f}"
    if s["kind"] == "capsule":
        return f"capsule R {s['R']:.3f} length {math.dist(s['a'], s['b']):.3f}"
    return "box " + " x ".join(f"{2 * h:.3f}" for h in s["half"])
