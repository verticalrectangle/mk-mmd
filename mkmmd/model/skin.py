"""Analytic skin weights for part builders (bpy-free; numpy, plus scipy.spatial for the surface search).

Conventions (everything in this module follows them)
  * model space: metres, Z up, the character faces -Y, her LEFT is +X; verts are float64 (n, 3); faces are lists of
    vertex index lists (tris, quads, ngons; counter-clockwise seen from outside); an (F, k) int array is accepted too.
  * a *weights* object is a plain dict {bone name: float64 array (n,)} (bones use their PMX names, a missing bone is 0).
    Nothing here modifies its inputs, and results never alias them. Weights from the generators below are non-negative
    and (except `envelope_weights` beyond its radii) already sum to 1 per vertex; after you `add`/`scale`/`mix` them
    call `normalise` once (cap 4, floor, fallback) before putting them into `Mesh.weights`.
  * ramps are (n,) float arrays in 0..1 (0 = none, 1 = full); they are masks for `mix`/`scale`.

API at a glance
  weights algebra   normalise(weights, n, cap=4, floor=1e-4, fallback=None)   to_matrix(weights, n, names=None)
                    from_matrix(names, W)   add(*dicts)   scale(weights, s)   mix(a, b, t)   rigid(n, bone)
                    region_blend(base, over, t)   check(weights, n, bones=None, tol=1e-3)
  ramps             smoothstep(x)   ramp(x, x0, x1)   plane_ramp(verts, point, normal, width, offset=0.0)
                    sphere_ramp(verts, centre, r_in, r_out)   capsule_ramp(verts, a, b, r_in, r_out)
                    twist_ramp(verts, a, b, start=0.0, end=1.0)
  generators        chain_param(verts, joints) -> (s, d)   chain_weights(verts, bones, joints, blend=0.2)
                    envelope_weights(verts, segments, radius=None, power=4.0, k=4)
  transfer          triangulate(faces)   closest_on_mesh(points, verts, faces)
                    transfer(src_verts, src_faces, src_weights, dst_verts, max_dist=None, fallback=None, ...)
                    transfer_from_meshes(meshes, dst_verts, **kw)   laplacian_smooth(weights, faces, n, ...)

Typical use
    w = skin.chain_weights(v, ["左腕", "左ひじ", "左手首"], [p_arm, p_elbow, p_wrist, p_tip], blend=0.06)
    w = skin.mix(w, skin.rigid(len(v), "左肩"), skin.sphere_ramp(v, p_arm, 0.03, 0.07))     # a shoulder cap
    w = skin.normalise(w, len(v), fallback="上半身")
    dw = skin.transfer(body.verts, body.faces, body.weights, dress_v, max_dist=0.04, fallback="下半身",
                       smooth_faces=dress_f, smooth_iters=2)                               # dress follows the skin

Notes
  * Everything is vectorised; big temporaries are processed in chunks, so a 100k-vertex mesh against a chain, an
    envelope or a 50k-triangle surface takes well under a few seconds.
  * `closest_on_mesh` is exact (see its docstring) and needs scipy.spatial.cKDTree for anything but tiny inputs;
    without scipy it evaluates all point/triangle pairs in chunks with plain numpy (correct, but slow for big meshes).
"""
import itertools

import numpy as np

__all__ = [
    "normalise", "to_matrix", "from_matrix", "add", "scale", "mix", "rigid", "region_blend", "check",
    "smoothstep", "ramp", "plane_ramp", "sphere_ramp", "capsule_ramp", "twist_ramp",
    "chain_param", "chain_weights", "envelope_weights",
    "triangulate", "closest_on_mesh", "transfer", "transfer_from_meshes", "laplacian_smooth",
]

_PAIR_BUDGET = 300_000     # point/segment or point/triangle pairs per numpy chunk (bounds temporary memory)
_BRUTE_PAIRS = 2_000       # closest_on_mesh evaluates every point/triangle pair when there are at most this many
_POINT_CHUNK = 2048        # closest_on_mesh (tree search): query points per ball query
_MIN_COS = 0.2             # chain_weights: cos(bend / 2) is clamped here, so hairpin bends stay finite


# ------------------------------------------------------------------------------------------------ input helpers
def _ret(x):
    """0-d arrays become numpy scalars (scalar in, scalar out); everything else is returned unchanged."""
    return x[()] if isinstance(x, np.ndarray) and x.ndim == 0 else x


def _as_points(x, what="verts"):
    """Finite float64 (n, 3); a single 3-vector is taken as one point."""
    a = np.asarray(x, dtype=np.float64)
    if a.ndim == 1 and a.shape[0] == 3:
        a = a.reshape(1, 3)
    if a.ndim != 2 or a.shape[1] != 3:
        raise ValueError(f"{what} must be an (n, 3) array, got shape {a.shape}")
    if not np.isfinite(a).all():
        raise ValueError(f"{what} contains non-finite values")
    return a


def _vec3(x, what):
    a = np.asarray(x, dtype=np.float64)
    if a.shape != (3,) or not np.isfinite(a).all():
        raise ValueError(f"{what} must be a finite 3-vector, got shape {a.shape}")
    return a


def _weight_array(w, n, bone):
    a = np.asarray(w, dtype=np.float64)
    if a.shape != (n,):
        raise ValueError(f"weights for {bone!r} have shape {a.shape}, expected ({n},)")
    return a


def _common_length(*dicts):
    """Length of the arrays in the given weight dicts (all must agree); None when every dict is empty."""
    n = None
    for d in dicts:
        for bone, w in d.items():
            a = np.asarray(w)
            if a.ndim != 1:
                raise ValueError(f"weights for {bone!r} must be a 1-d array, got shape {a.shape}")
            if n is None:
                n = a.shape[0]
            elif a.shape[0] != n:
                raise ValueError(f"weights for {bone!r} have length {a.shape[0]}, expected {n}")
    return n


def _chunks(n, per_item, budget=None):
    """Slices over range(n) so that (slice length) * per_item stays within `budget` (default: _PAIR_BUDGET)."""
    step = max(1, int((_PAIR_BUDGET if budget is None else budget) // max(1, per_item)))
    for s in range(0, n, step):
        yield slice(s, min(n, s + step))


def _flatten_faces(faces):
    """(flat vertex indices int64, face sizes int64) of a face list (or an (F, k) array)."""
    if isinstance(faces, np.ndarray) and faces.ndim == 2:
        count, size = faces.shape
        if count and size < 3:
            raise ValueError(f"faces need at least 3 vertices, got {size}")
        return faces.astype(np.int64).ravel(), np.full(count, size, dtype=np.int64)
    if not isinstance(faces, (list, tuple)):
        faces = list(faces)
    lens = np.fromiter((len(f) for f in faces), dtype=np.int64, count=len(faces))
    if lens.size and lens.min() < 3:
        bad = int(np.argmax(lens < 3))
        raise ValueError(f"face {bad} has {int(lens[bad])} vertices, need at least 3")
    flat = np.fromiter(itertools.chain.from_iterable(faces), dtype=np.int64, count=int(lens.sum()))
    return flat, lens


def _kdtree():
    """scipy.spatial.cKDTree, or None when scipy is not installed."""
    try:
        from scipy.spatial import cKDTree
    except ImportError:  # pragma: no cover - scipy is a declared dependency
        return None
    return cKDTree


def _stack(weights, n, names=None, strict=True):
    """(names, W (n, k) float64) from a weights dict; `names` fixes the column order (missing bones are zero)."""
    n = int(n)
    if names is None:
        names = list(weights)
    else:
        names = list(names)
        if len(set(names)) != len(names):
            raise ValueError("duplicate bone names")
        if strict:
            known = set(names)
            extra = [b for b in weights if b not in known]
            if extra:
                raise ValueError(f"{len(extra)} bones of the weights are not in names: {extra[:5]}")
    W = np.zeros((n, len(names)), dtype=np.float64)
    for j, b in enumerate(names):
        if b in weights:
            W[:, j] = _weight_array(weights[b], n, b)
    return names, W


# ------------------------------------------------------------------------------------------------ weights algebra
def to_matrix(weights, n, names=None, strict=True):
    """Weights dict -> dense matrix.

    Args: weights {bone: (n,) array}; n vertex count; names optional column order (bones missing from the dict give a
    zero column; default = the dict's own order); strict: with `names` given, bones of the dict that are not in
    `names` raise ValueError (False drops them silently).
    Returns: (names, W) with names a new list of bone names and W a float64 (n, len(names)) array (a fresh copy)."""
    return _stack(weights, n, names, strict)


def from_matrix(names, W, keep_zero=False):
    """Dense matrix -> weights dict.

    Args: names list of k distinct bone names; W (n, k) array; keep_zero: keep all-zero columns too (default: only
    bones with at least one non-zero weight are returned).
    Returns: {name: float64 (n,)} in `names` order; every array is an independent contiguous copy."""
    names = list(names)
    W = np.asarray(W, dtype=np.float64)
    if W.ndim != 2 or W.shape[1] != len(names):
        raise ValueError(f"W must be (n, {len(names)}) for {len(names)} names, got shape {W.shape}")
    if len(set(names)) != len(names):
        raise ValueError("duplicate bone names")
    live = W.any(axis=0) if W.shape[0] else np.zeros(len(names), dtype=bool)
    return {b: np.array(W[:, j], dtype=np.float64) for j, b in enumerate(names) if keep_zero or live[j]}


def _normalise_matrix(names, W, cap, floor, fallback, who="normalise"):
    """normalise() on a (n, k) matrix that the caller owns (modified in place); returns the weights dict."""
    n, k = W.shape
    cap = int(cap)
    floor = float(floor)
    if cap < 1:
        raise ValueError(f"cap must be >= 1, got {cap}")
    if not floor >= 0.0:
        raise ValueError(f"floor must be >= 0, got {floor}")
    bad = ~np.isfinite(W)
    if bad.any():
        j = int(np.flatnonzero(bad.any(axis=0))[0])
        raise ValueError(f"weights for {names[j]!r} contain non-finite values")
    np.maximum(W, 0.0, out=W)
    if floor > 0.0:
        W[W < floor] = 0.0
    if k > cap:
        rows = np.flatnonzero(np.count_nonzero(W, axis=1) > cap)
        if rows.size:
            sub = W[rows]
            order = np.argsort(-sub, axis=1, kind="stable")     # largest first, ties: earlier bone first
            np.put_along_axis(sub, order[:, cap:], 0.0, axis=1)
            W[rows] = sub
    tot = W.sum(axis=1)
    live = tot > 0.0
    if floor > 0.0 and k > 1 and live.any():
        small = (W > 0.0) & (W < floor * tot[:, None])         # below `floor` once renormalised
        if small.any():
            small[np.arange(n), W.argmax(axis=1)] = False      # a vertex always keeps its largest weight
            W[small] = 0.0
            tot = W.sum(axis=1)
    if live.any():
        W[live] /= tot[live, None]
    names = list(names)
    missing = int((~live).sum())
    if missing:
        if fallback is None:
            raise ValueError(f"{who}: {missing} of {n} vertices have no weight (none >= floor {floor:g}) "
                             f"and no fallback bone was given")
        if fallback in names:
            j = names.index(fallback)
        else:
            names.append(fallback)
            W = np.concatenate([W, np.zeros((n, 1))], axis=1)
            j = len(names) - 1
        W[~live, j] = 1.0
    return from_matrix(names, W)


def normalise(weights, n, cap=4, floor=1e-4, fallback=None):
    """Clean weights for export: at most `cap` influences per vertex, none below `floor`, every vertex summing to 1.

    Per vertex: negative values count as 0; weights below `floor` are dropped (so a vertex whose weights are all below
    `floor` counts as unweighted); the `cap` largest are kept (ties favour the bone that comes first in the dict); the
    rest is renormalised to sum 1; weights that fall below `floor` after renormalising are dropped and the vertex is
    renormalised once more (its largest weight is never dropped). Vertices left without any weight get
    {fallback: 1.0}.

    Args: weights {bone: (n,) array}; n vertex count; cap max influences per vertex (>= 1); floor smallest weight kept
    (>= 0); fallback bone name given to unweighted vertices (None: unweighted vertices are an error).
    Returns: a new dict {bone: float64 (n,)} that only holds bones with at least one non-zero weight, in the input's
    order (the fallback bone last when the input lacked it). Raises ValueError when an array is not (n,), when a
    weight is NaN/inf, or when vertices are unweighted and there is no fallback (the message gives their count)."""
    names, W = _stack(weights, n)
    return _normalise_matrix(names, W, cap, floor, fallback)


def add(*dicts):
    """Sum of weights dicts per bone (bone union; a bone missing from a dict counts as 0). No normalising.

    Args: any number of {bone: (n,) array} dicts with the same n. Returns: a new dict ({} for no input)."""
    out = {}
    _common_length(*dicts)
    for d in dicts:
        for bone, w in d.items():
            a = np.asarray(w, dtype=np.float64)
            out[bone] = out[bone] + a if bone in out else a.copy()
    return out


def scale(weights, s):
    """Every weight array times `s`.

    Args: weights {bone: (n,) array}; s a scalar or an (n,) array (e.g. a ramp). Returns: a new dict."""
    s = np.asarray(s, dtype=np.float64)
    n = _common_length(weights)
    if s.ndim > 1 or (s.ndim == 1 and n is not None and s.shape[0] != n):
        raise ValueError(f"s must be a scalar or ({n},), got shape {s.shape}")
    return {b: np.asarray(w, dtype=np.float64) * s for b, w in weights.items()}


def mix(a, b, t):
    """Per-vertex linear blend of two weights dicts: (1 - t) * a + t * b.

    Args: a, b {bone: (n,) array} (bone union; a bone missing on one side counts as 0); t scalar or (n,) array
    (0 -> a, 1 -> b; not clipped, pass a ramp). Returns: a new dict with every bone of a and b. If a and b both sum
    to 1 per vertex, so does the result (no normalising is done here)."""
    t = np.asarray(t, dtype=np.float64)
    n = _common_length(a, b)
    if t.ndim > 1 or (t.ndim == 1 and n is not None and t.shape[0] != n):
        raise ValueError(f"t must be a scalar or ({n},), got shape {t.shape}")
    out = {}
    for bone in list(a) + [x for x in b if x not in a]:
        wa = a.get(bone)
        wb = b.get(bone)
        if wa is not None and wb is not None:
            out[bone] = (1.0 - t) * np.asarray(wa, dtype=np.float64) + t * np.asarray(wb, dtype=np.float64)
        elif wa is not None:
            out[bone] = (1.0 - t) * np.asarray(wa, dtype=np.float64)
        else:
            out[bone] = t * np.asarray(wb, dtype=np.float64)
    return out


def region_blend(base, over, t):
    """Replace `base` by `over` where the mask `t` is 1 (smoothly in between): the same as mix(base, over, t).

    Args: base, over weights dicts; t scalar or (n,) mask (a ramp: plane_ramp, sphere_ramp, ...). Returns: new dict."""
    return mix(base, over, t)


def rigid(n, bone):
    """Every vertex fully on one bone. Args: n vertex count; bone name. Returns: {bone: ones(n)}."""
    return {bone: np.ones(int(n), dtype=np.float64)}


def check(weights, n, bones=None, tol=1e-3, cap=None):
    """Statistics and sanity flags of a weights dict (nothing is modified).

    Args: weights {bone: (n,) array}; n vertex count; bones optional collection of the skeleton's bone names
    (bones of the dict outside it are reported); tol tolerance for 'unweighted' (positive weights summing to <= tol)
    and for the 'ok' flag (sums within 1 +- tol); cap optional influence limit that 'ok' must respect.
    Returns: dict with
      max_bones_per_vertex  most bones with a weight > 0 on one vertex (int)
      unweighted            number of vertices whose positive weights sum to <= tol
      sum_min, sum_max      smallest / largest per-vertex weight total (floats; 0.0 where unweighted; nan if n == 0)
      unknown_bones         bones of the dict that are not in `bones` (list; [] when `bones` is None)
      negative              number of weights < 0
      nonfinite             number of NaN/inf weights (they count as 0 in the other numbers)
      n_bones               bones with at least one weight > 0
      ok                    True when unweighted == negative == nonfinite == 0, nothing unknown, every vertex sum is
                            within 1 +- tol and (if `cap` is given) max_bones_per_vertex <= cap.
    Arrays that are not (n,) raise ValueError."""
    names, W = _stack(weights, n)
    finite = np.isfinite(W)
    nonfinite = int(W.size - np.count_nonzero(finite))
    if nonfinite:
        W = np.where(finite, W, 0.0)
    pos = np.maximum(W, 0.0)
    total = W.sum(axis=1)
    count = np.count_nonzero(pos, axis=1)
    n = int(n)
    known = None if bones is None else set(bones)
    unknown = [] if known is None else [b for b in names if b not in known]
    out = {
        "max_bones_per_vertex": int(count.max()) if n else 0,
        "unweighted": int(np.count_nonzero(pos.sum(axis=1) <= tol)),
        "sum_min": float(total.min()) if n else float("nan"),
        "sum_max": float(total.max()) if n else float("nan"),
        "unknown_bones": unknown,
        "negative": int(np.count_nonzero(W < 0.0)),
        "nonfinite": nonfinite,
        "n_bones": int(np.count_nonzero(pos.any(axis=0))) if n else 0,
    }
    sums_ok = n == 0 or (out["sum_min"] >= 1.0 - tol and out["sum_max"] <= 1.0 + tol)
    out["ok"] = bool(out["unweighted"] == 0 and out["negative"] == 0 and nonfinite == 0 and not unknown and sums_ok
                     and (cap is None or out["max_bones_per_vertex"] <= cap))
    return out


# ------------------------------------------------------------------------------------------------ ramps
def smoothstep(x):
    """Hermite smoothstep: x clipped to 0..1, then 3x^2 - 2x^3 (0 below 0, 1 above 1, zero slope at both ends).

    Args: x scalar or array. Returns: float64 of the same shape, in 0..1 (a numpy scalar for scalar x)."""
    t = np.clip(np.asarray(x, dtype=np.float64), 0.0, 1.0)
    return _ret(t * t * (3.0 - 2.0 * t))


def ramp(x, x0, x1, smooth=True):
    """0 at x0, 1 at x1, clipped outside, smoothstep (or linear) in between; works for x1 < x0 too (falling ramp).

    Args: x scalar or array; x0, x1 the values mapped to 0 and 1 (scalars or arrays broadcastable with x, so
    per-vertex thresholds work); smooth False gives a linear ramp. x0 == x1 is a hard step: 1 where x >= x0, else 0.
    Returns: float64 array of x's shape in 0..1 (numpy scalar for scalar input)."""
    x = np.asarray(x, dtype=np.float64)
    x0 = np.asarray(x0, dtype=np.float64)
    x1 = np.asarray(x1, dtype=np.float64)
    d = x1 - x0
    flat = d == 0.0
    t = np.clip((x - x0) / np.where(flat, 1.0, d), 0.0, 1.0)
    if flat.any():
        t = np.where(flat, (x >= x0).astype(np.float64), t)
    if smooth:
        t = t * t * (3.0 - 2.0 * t)
    return _ret(t)


def plane_ramp(verts, point, normal, width, offset=0.0, smooth=True):
    """0 behind a plane, 1 in front of it, with a smooth transition centred on the plane.

    The plane passes through `point` + `offset` * unit(normal) and faces along `normal`.
    Args: verts (n, 3); point (3,) a point of the plane (before the offset); normal (3,) any non-zero length (front
    = the side it points to); width total length of the transition in metres (>= 0; 0 = hard step, 1 on the plane);
    offset metres to shift the plane along the normal; smooth False for a linear transition.
    Returns: (n,) floats in 0..1 (0.5 on the shifted plane for width > 0; 0 at <= -width/2, 1 at >= +width/2)."""
    V = _as_points(verts)
    p = _vec3(point, "point")
    nrm = _vec3(normal, "normal")
    length = float(np.linalg.norm(nrm))
    if not length > 0.0:
        raise ValueError("normal has zero length")
    width = float(width)
    if not width >= 0.0:
        raise ValueError(f"width must be >= 0, got {width}")
    s = (V - p) @ (nrm / length) - float(offset)
    return ramp(s, -0.5 * width, 0.5 * width, smooth)


def _radial(d, r_in, r_out, smooth):
    r_in = float(r_in)
    r_out = float(r_out)
    if not 0.0 <= r_in <= r_out:
        raise ValueError(f"need 0 <= r_in <= r_out, got r_in={r_in}, r_out={r_out}")
    return 1.0 - ramp(d, r_in, r_out, smooth)


def sphere_ramp(verts, centre, r_in, r_out, smooth=True):
    """1 inside a sphere, 0 outside a larger one, smooth in between (a soft ball mask).

    Args: verts (n, 3); centre (3,); r_in radius up to which the value is 1; r_out radius from which it is 0
    (r_in <= r_out; equal = hard step, 1 strictly inside); smooth False for a linear fall-off.
    Returns: (n,) floats in 0..1 (0.5 half way between the radii)."""
    V = _as_points(verts)
    c = _vec3(centre, "centre")
    return _radial(np.linalg.norm(V - c, axis=1), r_in, r_out, smooth)


def _segment_t(V, A, B):
    """Parameter (n,) in [0, 1] of the closest point on segment A->B (0 for a zero-length segment)."""
    e = B - A
    length2 = float(e @ e)
    if length2 <= 0.0:
        return np.zeros(len(V))
    return np.clip(((V - A) @ e) / length2, 0.0, 1.0)


def capsule_ramp(verts, a, b, r_in, r_out, smooth=True):
    """1 near a segment, 0 far from it, smooth in between (a soft capsule mask around a bone).

    Args: verts (n, 3); a, b (3,) segment end points; r_in, r_out as in `sphere_ramp`, applied to the distance to the
    segment a-b (end caps are spherical; a == b gives a sphere); smooth False for a linear fall-off.
    Returns: (n,) floats in 0..1."""
    V = _as_points(verts)
    A = _vec3(a, "a")
    B = _vec3(b, "b")
    t = _segment_t(V, A, B)
    d = np.linalg.norm(V - A - t[:, None] * (B - A), axis=1)
    return _radial(d, r_in, r_out, smooth)


def twist_ramp(verts, a, b, start=0.0, end=1.0, smooth=True):
    """Parameter along the segment a -> b through ramp(start, end): splits a limb between a bone and its twist bone.

    The parameter is the projection of the vertex on the segment, clamped to 0 (at a) .. 1 (at b); it is then mapped
    through ramp(t, start, end, smooth). Use w_twist = twist_ramp(...) and w_limb = 1 - w_twist, e.g. mix(limb, twist,
    twist_ramp(v, shoulder, elbow, 0.2, 0.8)).
    Args: verts (n, 3); a, b (3,) distinct segment end points; start, end parameter values mapped to 0 and 1 (end <
    start reverses the ramp); smooth False for linear.
    Returns: (n,) floats in 0..1."""
    V = _as_points(verts)
    A = _vec3(a, "a")
    B = _vec3(b, "b")
    if not float((B - A) @ (B - A)) > 0.0:
        raise ValueError("twist_ramp: a and b coincide")
    return ramp(_segment_t(V, A, B), start, end, smooth)


# ------------------------------------------------------------------------------------------------ chains
def _seg_t_d2(V, A, B):
    """Closest points of V (c, 3) on the segments A->B (S, 3): (t (c, S) in [0, 1], squared distance (c, S))."""
    E = B - A
    L2 = np.einsum("sj,sj->s", E, E)
    W = V[:, None, :] - A[None, :, :]
    U = np.einsum("csj,sj->cs", W, E)
    t = np.clip(U / np.where(L2 > 0.0, L2, 1.0), 0.0, 1.0)
    t[:, L2 <= 0.0] = 0.0
    R = W - t[:, :, None] * E[None, :, :]
    return t, np.einsum("csj,csj->cs", R, R)


def _nearest_segment(V, A, B):
    """Nearest of the segments A->B (S, 3) for each vertex: (segment index (n,), t (n,) in [0, 1], distance (n,))."""
    n, S = len(V), len(A)
    seg = np.empty(n, dtype=np.int64)
    for sl in _chunks(n, S):
        seg[sl] = np.argmin(_seg_t_d2(V[sl], A, B)[1], axis=1)
    a = A[seg]
    e = B[seg] - a
    L2 = np.einsum("ij,ij->i", e, e)
    u = np.einsum("ij,ij->i", V - a, e)
    t = np.where(L2 > 0.0, np.clip(u / np.where(L2 > 0.0, L2, 1.0), 0.0, 1.0), 0.0)
    r = V - a - t[:, None] * e
    return seg, t, np.sqrt(np.einsum("ij,ij->i", r, r))


def chain_param(verts, joints):
    """Where each vertex sits along a polyline of joints: the nearest point of the chain, as a segment parameter.

    Args: verts (n, 3); joints (k + 1, 3) the polyline p0..pk (segment i runs p_i -> p_{i+1}).
    Returns: (s, d): s (n,) = segment index + fraction (0..1) of the nearest point on the polyline, so s is in [0, k],
    continuous along the chain (a joint is exactly an integer), clamped to 0 before p0 and to k beyond pk; d (n,) =
    distance to the polyline. Note that on the inside of a sharp bend the nearest point can jump between segments
    (that is inherent to nearest-point projection); `chain_weights` does not suffer from it."""
    V = _as_points(verts)
    P = _as_points(joints, "joints")
    if len(P) < 2:
        raise ValueError("joints needs at least 2 points")
    seg, t, d = _nearest_segment(V, P[:-1], P[1:])
    return seg + t, d


def _segment_dirs(seg, length):
    """Unit directions of the segments; zero-length ones copy the nearest proper segment."""
    good = length > 1e-12 * float(length.max())
    d = np.zeros_like(seg)
    d[good] = seg[good] / length[good, None]
    if not good.all():
        idx = np.flatnonzero(good)
        for i in np.flatnonzero(~good):
            d[i] = d[idx[np.argmin(np.abs(idx - i))]]
    return d


def chain_weights(verts, bones, joints, blend=0.2):
    """Skin weights along a chain of bones: each bone owns its stretch of the chain, with smooth cross-fades.

    Bone i spans joints[i] -> joints[i + 1]. At every inner joint the weight passes from bone i-1 to bone i through a
    smoothstep cross-fade of total length `blend` (metres) centred on the joint, measured along the chain: a vertex on
    the chain exactly at the joint gets 50/50, one `blend`/2 before it is bone i-1 only, one `blend`/2 after bone i
    only. Before joints[0] only bone 0 has weight, after joints[k] only bone k-1. The cross-fade of a joint is clamped
    to the length of both of its segments (so fades never overlap and the outer bones still own their middle).

    Off the chain (a skin vertex around a limb) the fade follows the plane through the joint that bisects the two
    segments, so the weights are smooth everywhere, also on the inside and outside of a bend, and they do not depend on
    the distance to the chain. Chains may bend and curl (tails, braids, spines): every vertex is only blended between
    the bones of its nearest segment and its two neighbours. Bends sharper than ~157 deg (hairpins) are clamped; below
    that the arc-length behaviour above is exact on the chain. A vertex's weights are non-negative and sum to 1.

    Args: verts (n, 3); bones the k bone names (distinct); joints (k + 1, 3) joint positions p0..pk; blend cross-fade
    length in metres, a scalar or a list of k - 1 values (one per inner joint, bone i-1 -> i at joints[i]).
    Returns: {bone: (n,) array} with only the bones that get weight somewhere. Raises ValueError for a wrong
    number of joints, negative blend, duplicate bone names or a chain of zero length."""
    V = _as_points(verts)
    P = _as_points(joints, "joints")
    bones = list(bones)
    k = len(bones)
    if k < 1:
        raise ValueError("chain_weights: no bones")
    if len(P) != k + 1:
        raise ValueError(f"chain_weights: {k} bones need {k + 1} joints, got {len(P)}")
    if len(set(bones)) != k:
        raise ValueError("chain_weights: duplicate bone names")
    n = len(V)
    bl = np.asarray(blend, dtype=np.float64)
    if bl.ndim == 0:
        bl = np.full(k - 1, float(bl))
    if bl.shape != (k - 1,) or not np.isfinite(bl).all() or (bl < 0.0).any():
        raise ValueError(f"blend must be a non-negative scalar or a list of {k - 1} values, got {np.shape(blend)}")
    seg = P[1:] - P[:-1]
    length = np.linalg.norm(seg, axis=1)
    if not length.max() > 0.0:
        raise ValueError("chain_weights: all joints coincide")
    if k == 1:
        return {bones[0]: np.ones(n)}
    dirs = _segment_dirs(seg, length)
    # plane through inner joint j (1..k-1) that bisects segments j-1 and j; `inv_c` rescales the signed distance to
    # that plane so that, on the chain itself, it equals the arc length from the joint
    bis = dirs[:-1] + dirs[1:]
    bis_len = np.linalg.norm(bis, axis=1)
    ok = bis_len > 1e-9
    normal = np.zeros((k + 1, 3))
    normal[1:k] = np.where(ok[:, None], bis / np.where(ok, bis_len, 1.0)[:, None], dirs[1:])
    inv_c = np.ones(k + 1)
    inv_c[1:k] = 1.0 / np.clip(np.where(ok, 0.5 * bis_len, 0.0), _MIN_COS, 1.0)
    width = np.zeros(k + 1)
    width[1:k] = np.minimum(bl, np.minimum(length[:-1], length[1:]))

    def passed(j):
        """How far each vertex is past joint j: 0 before its fade, 1 after it (joint 0 is always passed, joint k never)."""
        z = np.einsum("ij,ij->i", V - P[j], normal[j]) * inv_c[j]
        f = ramp(z, -0.5 * width[j], 0.5 * width[j])
        return np.where(j == 0, 1.0, np.where(j == k, 0.0, f))

    near = _nearest_segment(V, P[:-1], P[1:])[0]
    a = passed(near)             # past the start joint of the nearest segment
    b = passed(near + 1)         # past its end joint
    W = np.zeros((n, k))
    rows = np.arange(n)
    W[rows, near] = a * (1.0 - b)
    before = near >= 1
    W[rows[before], near[before] - 1] = (1.0 - a)[before]
    after = near + 1 <= k - 1
    W[rows[after], near[after] + 1] = (a * b)[after]
    return from_matrix(bones, W)


def envelope_weights(verts, segments, radius=None, power=4.0, k=4):
    """Rough inverse-distance skinning to bone segments: cheap weights for garments, hair clumps and accessories.

    Every vertex takes the `k` nearest segments (distance to the segment, end caps included); a segment at distance d
    contributes 1 / (d**power + 1e-9) and the contributions are normalised to sum 1. A segment farther than its radius
    contributes nothing (and does not use up one of the k slots).

    Args: verts (n, 3); segments {bone: (a, b)} head and tail positions (a == b is fine); radius None (no limit), a
    scalar, or a dict {bone: radius} (bones missing from it are unlimited) in metres; power exponent of the fall-off
    (higher = more rigid); k how many nearest segments are mixed (>= 1).
    Returns: {bone: (n,) array} in the order of `segments`, only bones with some weight. Vertices outside every
    radius have no weight at all: hand them to `normalise(..., fallback=bone)`. Raises ValueError without segments."""
    V = _as_points(verts)
    names = list(segments)
    S = len(names)
    if S == 0:
        raise ValueError("envelope_weights: no segments")
    A = np.empty((S, 3))
    B = np.empty((S, 3))
    for i, b in enumerate(names):
        ends = segments[b]
        A[i] = _vec3(ends[0], f"segments[{b!r}][0]")
        B[i] = _vec3(ends[1], f"segments[{b!r}][1]")
    if radius is None:
        r2 = None
    else:
        if hasattr(radius, "get"):
            r = np.array([float(radius.get(b, np.inf)) for b in names])
        else:
            r = np.full(S, float(radius))
        if (r < 0.0).any():
            raise ValueError("radius must be >= 0")
        r2 = r * r
    k = int(k)
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    kk = min(k, S)
    n = len(V)
    idx = np.empty((n, kk), dtype=np.int64)
    w = np.empty((n, kk))
    for sl in _chunks(n, S):
        d2 = _seg_t_d2(V[sl], A, B)[1]
        if r2 is not None:
            d2 = np.where(d2 > r2[None, :], np.inf, d2)
        sel = np.argpartition(d2, kk - 1, axis=1)[:, :kk] if kk < S else np.tile(np.arange(S), (len(d2), 1))
        dsel = np.take_along_axis(d2, sel, axis=1)
        idx[sl] = sel
        w[sl] = 1.0 / (np.sqrt(dsel) ** power + 1e-9)
    tot = w.sum(axis=1)
    live = tot > 0.0
    w[live] /= tot[live, None]
    w[~live] = 0.0
    keep = (w > 0.0).ravel()
    row = np.repeat(np.arange(n), kk)[keep]
    bone = idx.ravel()[keep]
    val = w.ravel()[keep]
    order = np.argsort(bone, kind="stable")
    row, bone, val = row[order], bone[order], val[order]
    bounds = np.searchsorted(bone, np.arange(S + 1))
    out = {}
    for i, b in enumerate(names):
        lo, hi = bounds[i], bounds[i + 1]
        if hi > lo:
            arr = np.zeros(n)
            arr[row[lo:hi]] = val[lo:hi]
            out[b] = arr
    return out


# ------------------------------------------------------------------------------------------------ surface search
def triangulate(faces):
    """Fan-split polygons into triangles, keeping the winding.

    A face [v0, v1, ..., vm] becomes (v0, v1, v2), (v0, v2, v3), ... (a quad is split along its v0-v2 diagonal).
    Triangles come in face order, each face's fan in order; this is the 'triangulation' that `closest_on_mesh` returns
    triangle indices for.
    Args: faces list of vertex index lists (>= 3 each) or an (F, k) int array.
    Returns: (tris (T, 3) int64, tri_face (T,) int64 = index of the face each triangle came from)."""
    flat, lens = _flatten_faces(faces)
    nf = len(lens)
    ntri = lens - 2
    total = int(ntri.sum())
    face = np.repeat(np.arange(nf, dtype=np.int64), ntri)
    local = np.arange(total, dtype=np.int64) - np.repeat(np.cumsum(ntri) - ntri, ntri)
    s = (np.cumsum(lens) - lens)[face]
    tris = np.stack([flat[s], flat[s + local + 1], flat[s + local + 2]], axis=1)
    return tris, face


def _dot(a, b):
    return np.einsum("ij,ij->i", a, b)


def _tri_closest(p, a, b, c):
    """Closest points of p (m, 3) on the triangles (a, b, c) (each (m, 3), pairwise): (q (m, 3), bary (m, 3), d2 (m,)).

    The plane projection is taken when it falls inside the triangle, otherwise the nearest of the three edges (also
    right for degenerate triangles, which have no interior)."""
    ab = b - a
    ac = c - a
    ap = p - a
    d00 = _dot(ab, ab)
    d01 = _dot(ab, ac)
    d11 = _dot(ac, ac)
    d20 = _dot(ap, ab)
    d21 = _dot(ap, ac)
    den = d00 * d11 - d01 * d01
    good = den > 1e-15 * d00 * d11
    safe = np.where(good, den, 1.0)
    v = (d11 * d20 - d01 * d21) / safe
    w = (d00 * d21 - d01 * d20) / safe
    u = 1.0 - v - w
    bary = np.stack([u, v, w], axis=1)
    out = np.flatnonzero(~(good & (u >= 0.0) & (v >= 0.0) & (w >= 0.0)))
    if out.size:
        pa, aa, bb = p[out], a[out], b[out]
        e1, e2, e3 = ab[out], ac[out], c[out] - bb

        def edge(origin, e):
            l2 = _dot(e, e)
            t = np.clip(_dot(pa - origin, e) / np.where(l2 > 0.0, l2, 1.0), 0.0, 1.0)
            r = pa - origin - t[:, None] * e
            return t, _dot(r, r)

        t1, f1 = edge(aa, e1)         # a -> b
        t2, f2 = edge(aa, e2)         # a -> c
        t3, f3 = edge(bb, e3)         # b -> c
        which = np.argmin(np.stack([f1, f2, f3]), axis=0)
        eb = np.zeros((out.size, 3))
        eb[:, 0] = np.where(which == 0, 1.0 - t1, np.where(which == 1, 1.0 - t2, 0.0))
        eb[:, 1] = np.where(which == 0, t1, np.where(which == 2, 1.0 - t3, 0.0))
        eb[:, 2] = np.where(which == 1, t2, np.where(which == 2, t3, 0.0))
        bary[out] = eb
    q = bary[:, 0:1] * a + bary[:, 1:2] * b + bary[:, 2:3] * c
    r = p - q
    return q, bary, _dot(r, r)


def _pair_d2(P, A, B, C, pi, ti):
    """Squared distance of points P[pi] to triangles ti (chunked)."""
    out = np.empty(len(pi))
    for s in range(0, len(pi), _PAIR_BUDGET):
        sl = slice(s, s + _PAIR_BUDGET)
        t = ti[sl]
        out[sl] = _tri_closest(P[pi[sl]], A[t], B[t], C[t])[2]
    return out


def _group_argmin(d2, counts):
    """Minimum of each non-empty group of a flat array whose groups are contiguous with sizes `counts`:
    (mask of the non-empty groups, flat index of the first minimum of each of them)."""
    nz = counts > 0
    cnt = counts[nz]
    start = np.cumsum(cnt) - cnt
    low = np.minimum.reduceat(d2, start)
    hit = np.flatnonzero(d2 == np.repeat(low, cnt))
    gid = np.repeat(np.arange(cnt.size), cnt)[hit]
    first = hit[np.concatenate([[True], gid[1:] != gid[:-1]])]
    return nz, first


def _nearest_tri_brute(P, A, B, C):
    """Index of the nearest triangle per point, evaluating every pair (chunked)."""
    m, T = len(P), len(A)
    best = np.empty(m, dtype=np.int64)
    for sl in _chunks(m, T):
        c = sl.stop - sl.start
        d2 = _tri_closest(np.repeat(P[sl], T, axis=0), np.tile(A, (c, 1)), np.tile(B, (c, 1)), np.tile(C, (c, 1)))[2]
        best[sl] = d2.reshape(c, T).argmin(axis=1)
    return best


def _nearest_tri_tree(P, A, B, C, cKDTree):
    """Index of the nearest triangle per point, exactly, with KD-trees over the triangle centroids.

    A triangle whose centroid c has circumradius rho (largest centroid-to-vertex distance) is at least |p - c| - rho
    away from p. So after a first guess with distance D (the nearest few centroids), only triangles with
    |p - c| <= D + rho can be closer. Triangles are grouped in size classes (rho within a factor 2) with one tree each,
    largest first, and every class is searched with radius D + rho_max(class): nothing can be missed, however thin,
    long or differently sized the triangles are, and the radius shrinks as D improves."""
    m, T = len(P), len(A)
    cen = (A + B + C) / 3.0
    rho = np.sqrt(np.maximum(_sq(A - cen), np.maximum(_sq(B - cen), _sq(C - cen))))
    ks = min(4, T)
    ii = np.asarray(cKDTree(cen).query(P, k=ks)[1], dtype=np.int64).reshape(m, ks)
    d2 = _pair_d2(P, A, B, C, np.repeat(np.arange(m), ks), ii.ravel()).reshape(m, ks)
    rows = np.arange(m)
    pick = d2.argmin(axis=1)
    best = ii[rows, pick]
    best_d2 = d2[rows, pick]
    positive = rho[rho > 0.0]
    ref = float(np.median(positive)) if positive.size else 1.0
    size_class = np.floor(np.log2(np.maximum(rho, ref * 2.0 ** -6) / ref)).astype(np.int64)
    for cls in np.unique(size_class)[::-1]:
        tid = np.flatnonzero(size_class == cls)
        tree = cKDTree(cen[tid])
        rho_max = float(rho[tid].max())
        for s in range(0, m, _POINT_CHUNK):
            sl = slice(s, min(m, s + _POINT_CHUNK))
            radius = (np.sqrt(best_d2[sl]) + rho_max) * (1.0 + 1e-9) + 1e-12
            lists = tree.query_ball_point(P[sl], radius, return_sorted=False)
            counts = np.fromiter(map(len, lists), dtype=np.int64, count=len(lists))
            total = int(counts.sum())
            if total == 0:
                continue
            cand = tid[np.fromiter(itertools.chain.from_iterable(lists), dtype=np.int64, count=total)]
            d2 = _pair_d2(P, A, B, C, np.repeat(np.arange(sl.start, sl.stop), counts), cand)
            has, first = _group_argmin(d2, counts)
            who = np.flatnonzero(has) + sl.start
            better = d2[first] < best_d2[who]
            best_d2[who[better]] = d2[first][better]
            best[who[better]] = cand[first][better]
    return best


def _sq(x):
    return np.einsum("ij,ij->i", x, x)


def _closest(P, V, tris, method="auto"):
    """closest_on_mesh on validated inputs: (q, triangle index, bary, squared distance)."""
    m = len(P)
    if m == 0:
        return np.zeros((0, 3)), np.zeros(0, dtype=np.int64), np.zeros((0, 3)), np.zeros(0)
    A, B, C = V[tris[:, 0]], V[tris[:, 1]], V[tris[:, 2]]
    if method not in ("auto", "tree", "brute"):
        raise ValueError(f"method must be 'auto', 'tree' or 'brute', got {method!r}")
    cKDTree = None if method == "brute" else _kdtree()
    if method == "tree" and cKDTree is None:
        raise ImportError("closest_on_mesh(method='tree') needs scipy")
    if cKDTree is None or (method == "auto" and m * len(tris) <= _BRUTE_PAIRS):
        best = _nearest_tri_brute(P, A, B, C)
    else:
        best = _nearest_tri_tree(P, A, B, C, cKDTree)
    q, bary, d2 = _tri_closest(P, A[best], B[best], C[best])
    return q, best, bary, d2


def _mesh_triangles(verts, faces):
    V = _as_points(verts)
    tris, tri_face = triangulate(faces)
    if len(tris) == 0:
        raise ValueError("the mesh has no faces")
    if tris.min() < 0 or tris.max() >= len(V):
        raise ValueError(f"face indices must be in 0..{len(V) - 1}, got {int(tris.min())}..{int(tris.max())}")
    return V, tris, tri_face


def closest_on_mesh(points, verts, faces, method="auto"):
    """Exact closest points of `points` on the surface of a polygon mesh.

    Faces are fan-split into triangles (see `triangulate`) and the true nearest point of the triangle surface is
    found (interior, edge or vertex; no vertex snapping, no smoothing). The candidate search is conservative, based
    on triangle bounding radii and KD-trees over the centroids, so thin, long or very different sized triangles
    cannot be missed; cost grows with (distance to the surface) / (triangle size) only. Ties between triangles that
    share the closest point are resolved arbitrarily (the points coincide anyway).

    Args: points (m, 3) query points; verts (n, 3) mesh vertices; faces list of index lists (tris/quads/ngons) or an
    (F, k) array; method 'auto' (all pairs when m * triangles is tiny or scipy is missing, else the tree search),
    'tree' or 'brute' (every point/triangle pair in chunks, pure numpy: only sensible for small meshes).
    Returns: (q, tri_index, bary, face_index):
      q          (m, 3) the closest points
      tri_index  (m,) int64 index into `triangulate(faces)[0]` of the triangle holding q
      bary       (m, 3) barycentric coordinates of q in that triangle's vertices (sum 1, all >= 0), so
                 q == bary[:, 0] * v[a] + bary[:, 1] * v[b] + bary[:, 2] * v[c] for tris[tri_index] = (a, b, c)
      face_index (m,) int64 index of the original face (before triangulation)."""
    P = _as_points(points, "points") if len(np.asarray(points)) else np.zeros((0, 3))
    V, tris, tri_face = _mesh_triangles(verts, faces)
    q, tri, bary, _ = _closest(P, V, tris, method)
    return q, tri, bary, tri_face[tri]


# ------------------------------------------------------------------------------------------------ transfer
def _edge_lists(faces, n):
    """Both directions (i, j) of every distinct polygon edge of a face list, as two int64 arrays."""
    flat, lens = _flatten_faces(faces)
    if flat.size and (flat.min() < 0 or flat.max() >= n):
        raise ValueError(f"face indices must be in 0..{n - 1}, got {int(flat.min())}..{int(flat.max())}")
    start = np.cumsum(lens) - lens
    nxt = np.arange(flat.size) + 1
    nxt[start + lens - 1] = start
    a, b = flat, flat[nxt]
    lo = np.minimum(a, b)
    hi = np.maximum(a, b)
    ok = lo != hi
    key = np.unique(lo[ok] * n + hi[ok])
    lo, hi = key // n, key % n
    return np.concatenate([lo, hi]), np.concatenate([hi, lo])


def _smooth_matrix(W, ei, ej, iterations, lam, pinned):
    """Jacobi Laplacian smoothing of the columns of W (n, k) over the directed edge lists (ei, ej)."""
    n, k = W.shape
    cols = np.ascontiguousarray(W.T)
    deg = np.bincount(ei, minlength=n).astype(np.float64)
    move = deg > 0.0
    if pinned is not None:
        move &= ~pinned
    inv = np.where(deg > 0.0, 1.0 / np.where(deg > 0.0, deg, 1.0), 0.0)
    live = np.flatnonzero(cols.any(axis=1))
    for _ in range(int(iterations)):
        for c in live:
            col = cols[c]
            mean = np.bincount(ei, weights=col[ej], minlength=n) * inv
            cols[c] = np.where(move, (1.0 - lam) * col + lam * mean, col)
    return np.ascontiguousarray(cols.T)


def _check_smoothing(iterations, lam):
    if int(iterations) < 0:
        raise ValueError(f"iterations must be >= 0, got {iterations}")
    if not 0.0 <= float(lam) <= 1.0:
        raise ValueError(f"lam must be in 0..1, got {lam}")


def laplacian_smooth(weights, faces, n, iterations=2, lam=0.5, pinned=None):
    """Smooth weights over the mesh: each iteration moves every vertex `lam` of the way to the mean of its neighbours.

    Neighbours are the vertices sharing a polygon edge (each distinct edge counts once). The update is
    w <- (1 - lam) * w + lam * mean(neighbours' w), applied to all vertices at once (Jacobi). It keeps the weight sum
    of a vertex at 1 if all inputs sum to 1, but it spreads influences to neighbours, so it can raise the number of
    bones per vertex: `normalise` afterwards. Vertices without edges keep their weights.

    Args: weights {bone: (n,) array}; faces face list of the mesh the weights live on; n vertex count; iterations
    number of passes (0 = copy); lam step in 0..1; pinned optional (n,) boolean mask (or index array) of vertices that
    keep their weights.
    Returns: a new dict with every bone that has weight somewhere afterwards."""
    _check_smoothing(iterations, lam)
    names, W = _stack(weights, n)
    n = int(n)
    keep = None
    if pinned is not None:
        pinned = np.asarray(pinned)
        if pinned.dtype == bool:
            if pinned.shape != (n,):
                raise ValueError(f"pinned mask must be ({n},), got {pinned.shape}")
            keep = pinned
        else:
            keep = np.zeros(n, dtype=bool)
            keep[pinned.astype(np.int64)] = True
    ei, ej = _edge_lists(faces, n)
    return from_matrix(names, _smooth_matrix(W, ei, ej, iterations, float(lam), keep))


def transfer(src_verts, src_faces, src_weights, dst_verts, max_dist=None, fallback=None,
             smooth_faces=None, smooth_iters=0, lam=0.5):
    """Skin weights of a surface copied to other vertices: a dress or a sock gets its body's weights.

    Every destination vertex takes the closest point of the SOURCE SURFACE (exact, see `closest_on_mesh`) and the
    source weights are interpolated barycentrically there (source weights are first scaled to sum 1 per source vertex;
    source vertices without any weight are ignored by the interpolation). Then:
      * max_dist (with `fallback`): the further a vertex is from the source surface, the more it is pulled to
        {fallback: 1}: unchanged up to max_dist, smoothstep blend over the next max_dist, pure fallback beyond
        2 * max_dist. Without `fallback` max_dist has no effect (the closest surface point is used however far it is).
      * smooth_iters > 0: `laplacian_smooth` over the destination mesh `smooth_faces` (needs them) with step lam.
      * the result goes through normalise(cap=4, fallback=fallback), so destination vertices that find no weight
        (their source vertices are unweighted) get the fallback bone, or ValueError when there is none.

    Args: src_verts (ns, 3), src_faces, src_weights {bone: (ns,)} the source surface and its weights; dst_verts (m, 3);
    max_dist metres or None; fallback bone name or None; smooth_faces destination face list (indices into dst_verts);
    smooth_iters, lam as in `laplacian_smooth`.
    Returns: {bone: (m,) array} normalised, at most 4 bones per vertex."""
    Vs, tris, _ = _mesh_triangles(src_verts, src_faces)
    Vd = _as_points(dst_verts, "dst_verts")
    m = len(Vd)
    ns = len(Vs)
    if max_dist is not None and not float(max_dist) > 0.0:
        raise ValueError(f"max_dist must be > 0, got {max_dist}")
    if smooth_iters and smooth_faces is None:
        raise ValueError("smooth_iters needs smooth_faces (the destination mesh)")
    _check_smoothing(smooth_iters, lam)
    names, Ws = _stack(src_weights, ns)
    if not np.isfinite(Ws).all():
        raise ValueError("src_weights contain non-finite values")
    np.maximum(Ws, 0.0, out=Ws)
    tot = Ws.sum(axis=1)
    has = tot > 0.0
    Ws[has] /= tot[has, None]
    _, tri, bary, d2 = _closest(Vd, Vs, tris)
    corner = tris[tri]
    Wd = bary[:, 0:1] * Ws[corner[:, 0]]
    Wd += bary[:, 1:2] * Ws[corner[:, 1]]
    Wd += bary[:, 2:3] * Ws[corner[:, 2]]
    tot = Wd.sum(axis=1)
    has = tot > 0.0
    Wd[has] /= tot[has, None]
    if fallback is not None and fallback not in names:
        names.append(fallback)
        Wd = np.concatenate([Wd, np.zeros((m, 1))], axis=1)
    if max_dist is not None and fallback is not None:
        keep = 1.0 - ramp(np.sqrt(d2), float(max_dist), 2.0 * float(max_dist))
        Wd *= keep[:, None]
        Wd[:, names.index(fallback)] += 1.0 - keep
    if smooth_iters:
        ei, ej = _edge_lists(smooth_faces, m)
        Wd = _smooth_matrix(Wd, ei, ej, smooth_iters, float(lam), None)
    return _normalise_matrix(names, Wd, 4, 1e-4, fallback, who="transfer")


def transfer_from_meshes(meshes, dst_verts, **kw):
    """`transfer` with `part.Mesh` objects as the source surface.

    The meshes' vertices, faces (offset, fan-triangulated) and weights are concatenated into one source surface; a
    weights dict that lacks a bone counts as 0 for that mesh. Meshes without faces contribute nothing.
    Args: meshes a list of part.Mesh (anything with verts, faces and weights) or a single one; dst_verts (m, 3);
    **kw the keyword arguments of `transfer` (max_dist, fallback, smooth_faces, smooth_iters, lam).
    Returns: {bone: (m,) array}, as `transfer`."""
    if hasattr(meshes, "verts"):
        meshes = [meshes]
    meshes = list(meshes)
    if not meshes:
        raise ValueError("transfer_from_meshes: no meshes")
    bones = list(dict.fromkeys(b for mesh in meshes for b in mesh.weights))
    verts, tris, offsets, total = [], [], [], 0
    for mesh in meshes:
        v = _as_points(mesh.verts, f"mesh {mesh.name!r} verts") if len(mesh.verts) else np.zeros((0, 3))
        for b, w in mesh.weights.items():
            _weight_array(w, len(v), f"{mesh.name}/{b}")
        verts.append(v)
        t, _ = triangulate(mesh.faces)
        tris.append(t + total)
        offsets.append(total)
        total += len(v)
    weights = {b: np.zeros(total) for b in bones}
    for mesh, off in zip(meshes, offsets):
        for b, w in mesh.weights.items():
            weights[b][off:off + len(mesh.verts)] = np.asarray(w, dtype=np.float64)
    return transfer(np.concatenate(verts), np.concatenate(tris), weights, dst_verts, **kw)
