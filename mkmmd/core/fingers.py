"""Finger shapes: how far each finger's first three joints bend toward the palm, as a preset name or a table; how far the
fingers fan apart; and a thumb solved per hand where a shape needs it to touch something.

`curls("relaxed")` and `curls({"index": [8, 10], "thumb": [0, 8]})` both give {finger: (a, b, c)} in degrees for every
finger of mkmmd.core.bonemap.FINGERS; `spreads({"spread": 8, ...})` gives {finger: degrees} each finger's base joint turns
away from the middle finger in the plane of the palm (the thumb away from the index). `joints` turns a shape into the
rotation of every joint of a hand from its rest bones (each finger curled about its own flexion axis, its first segment
x the palm normal, its base first turned about the palm normal by its spread), what the pose stage keys, and `pose` into
where its joints and tips go: the moves library places a gesture by where the posed hand's fingertips are
(mkmmd.core.moves). bpy-free, so it is tested without Blender.

A thumb can be solved instead of curled by angles (THUMBS). Curled about the flexion axis of the other fingers, a thumb
only swings out of the palm's plane: it can neither fold across the palm nor turn its pad to a fingertip, and every model
holds its thumb at rest its own way, so one table of angles cannot make it touch anything on every model.
- `"tuck"` (the `point` preset's): folded over the curled middle finger, as a fist holds it (`tuck_thumb`: turned across
  the palm and curled, searched per hand from the rest pose: its tip on the middle finger's curled middle and end
  segments, past the index, no point of it nearer to the middle and index fingers than they are thick).
- `"ring"`: its tip meets the index's tip, closing a ring of the two (an OK sign: a ring to look through).
- `"cross"`: its last segment lies across the index's last one on the thumb's side, the two tips just past each other
  (the finger heart: the two tips are its lobes, the crossing its point).
`ring` and `cross` move the thumb as a thumb moves (`opposed`): its base joint swings it out of the palm's plane and
across it and rolls it about its own length (opposition), its two other joints bend toward its pad (the side its rest
pose bends to); `oppose_thumb` solves those five angles per hand for the shape's index, within a thumb's range, its
other joints clear of the index and middle fingers, with the least turning."""
import numpy as np

from . import bonemap

SPREAD = {"index": 1.0, "middle": 0.0, "ring": 0.6, "little": 1.2, "thumb": 0.8}    # share of `spread` each finger fans out
PRESETS = {                    # curl per joint (deg) for (fingers 1-3); the thumb separately (THUMBS: solved per hand)
    "flat": ((0, 0, 0), (0, 0, 0)),
    "relaxed": ((14, 22, 14), (8, 10, 8)),
    "curled": ((35, 50, 35), (14, 18, 14)),
    "fist": ((80, 95, 65), (25, 35, 40)),
    "point": ((80, 95, 65), "tuck"),
}
THUMBS = ("tuck", "ring", "cross")   # thumbs solved per hand: folded in a fist, meeting the index's tip, across its tip
TUCK_GAP = 0.22                # the tucked thumb's tip rests this share of the middle finger's length from its bones
TUCK_CLEAR = 0.18             # ... the rest of it stays this share away from the middle and index fingers' bones
TUCK_SLACK = 0.04             # ... all to within this share; the least turned of those thumbs wins
TUCK_ACROSS = 0.25            # ... its tip at most this share of the index-middle knuckle spacing short of their midline
# an opposed thumb (`opposed`): its five angles (deg) and their range: the base swung out of the palm's plane (palmar
# abduction) and away from the index in it (radial abduction) and rolled about its length (opposition), its middle and
# last joints bent toward its pad
OPPOSE = ("palmar", "radial", "roll", "mcp", "ip")
OPPOSE_LO = np.array([-15.0, -40.0, -40.0, -10.0, -20.0])
OPPOSE_HI = np.array([70.0, 45.0, 50.0, 70.0, 85.0])
OPPOSE_COST = np.array([1.0, 1.0, 1.0, 0.6, 0.4])     # per degree: the base turns cost the most
OPPOSE_CLEAR = 0.14            # share of the middle finger's length the thumb's other joints keep from the fingers' bones
RADIUS = 0.05                  # a finger's skin radius as a share of the middle finger's length (without the skin's own)
CROSS_AT = (0.55, 0.45)        # the crossing: this share along the thumb's last segment and along the index's last one
CROSS_DEG = 75.0               # ... crossing at this angle, the thumb in the plane square to the index's side
RING_DOT = -0.3                # the ring: the cosine between the thumb's and the index's last segments (round, not a beak)
# a typical hand at rest (the left), in its own frame: x the way the wrist bone points (where a hand's `dir` points it),
# y its palm normal (where `palm` faces it) square to x, z = x cross y; shares of the wrist-to-middle-fingertip length.
# The moves library falls back on it for a member whose fingers it cannot read
HAND = {
    "thumb": ((0.156, 0.095, -0.147), (0.277, 0.108, -0.208), (0.396, 0.116, -0.201), (0.628, 0.151, -0.173)),
    "index": ((0.451, 0.025, -0.118), (0.6, 0.031, -0.115), (0.708, 0.034, -0.112), (0.922, 0.056, -0.103)),
    "middle": ((0.462, -0.011, 0.006), (0.633, 0.003, 0.008), (0.765, 0.018, 0.008), (1.0, 0.025, 0.011)),
    "ring": ((0.449, -0.003, 0.121), (0.602, 0.019, 0.121), (0.717, 0.037, 0.118), (0.932, 0.067, 0.119)),
    "little": ((0.413, 0.023, 0.228), (0.517, 0.021, 0.227), (0.618, 0.03, 0.227), (0.82, 0.046, 0.224)),
}


def thumb_mode(spec):
    """The solved thumb a shape asks for (THUMBS), or None for a thumb curled by angles."""
    if isinstance(spec, str):
        t = PRESETS.get(spec, (None, None))[1]
        return t if isinstance(t, str) else None
    t = spec.get("thumb") if isinstance(spec, dict) else None
    return t if isinstance(t, str) and t in THUMBS else None


def curls(spec):
    """{finger: (a, b, c)} degrees of the first three joints of every finger. `spec` is a preset name (PRESETS; "point"
    leaves the index straight) or a table {finger: [a, b, c], spread} where a finger left out, or a joint left off the end
    of its list, stays straight (`spread` is read by `spreads`). A solved thumb (THUMBS, `thumb_mode`) is (0, 0, 0) here:
    `joints` solves it per hand. Raises ValueError naming what is wrong."""
    if isinstance(spec, str):
        if spec not in PRESETS:
            raise ValueError(f"finger preset {spec!r} (have {sorted(PRESETS)}, or a table {{finger: [a, b, c]}})")
        fing, thumb = PRESETS[spec]
        thumb = (0.0, 0.0, 0.0) if isinstance(thumb, str) else thumb
        return {f: (thumb if f == "thumb" else (0.0, 0.0, 0.0) if spec == "point" and f == "index" else fing)
                for f in bonemap.FINGERS}
    if isinstance(spec, dict):
        bad = sorted(set(spec) - set(bonemap.FINGERS) - {"spread"})
        if bad:
            raise ValueError(f"fingers {bad}: unknown (have {list(bonemap.FINGERS)} and spread)")
        out = {}
        for f in bonemap.FINGERS:
            v = spec.get(f, ())
            if f == "thumb" and isinstance(v, str) and v in THUMBS:
                out[f] = (0.0, 0.0, 0.0)
                continue
            v = [v] if isinstance(v, (int, float)) else list(v) if isinstance(v, (list, tuple)) else [v]
            if len(v) > 3 or not all(isinstance(a, (int, float)) for a in v):
                what = "up to three angles in degrees" + (f" or one of {', '.join(THUMBS)}" if f == "thumb" else "")
                raise ValueError(f"fingers.{f}: give {what}, got {spec[f]!r}")
            out[f] = tuple(float(a) for a in v) + (0.0,) * (3 - len(v))
        return out
    raise ValueError(f"fingers: a preset name or a table {{finger: [a, b, c]}}, got {spec!r}")


def spreads(spec):
    """{finger: degrees} each finger's base joint turns away from the middle finger in the plane of the palm (the thumb
    away from the index): a table's `spread` (deg, the index's; the others fan by SPREAD's shares, the middle finger stays)
    or {finger: deg}; nothing for a preset name or a table without it. Negative closes the fingers together."""
    s = spec.get("spread") if isinstance(spec, dict) else None
    if s is None:
        return {f: 0.0 for f in bonemap.FINGERS}
    if isinstance(s, (int, float)):
        return {f: float(s) * SPREAD[f] for f in bonemap.FINGERS}
    if isinstance(s, dict):
        bad = sorted(set(s) - set(bonemap.FINGERS))
        if bad or not all(isinstance(v, (int, float)) for v in s.values()):
            raise ValueError(f"fingers.spread: degrees, or {{finger: degrees}} for {list(bonemap.FINGERS)}, got {s!r}")
        return {f: float(s.get(f, 0.0)) for f in bonemap.FINGERS}
    raise ValueError(f"fingers.spread: degrees, or {{finger: degrees}}, got {s!r}")


# ---------------------------------------------------------------------------------------------------------- rotations
def _turn(v, axis, deg):
    """Vectors v (..., 3) turned about the unit `axis` by `deg` (...,) degrees (Rodrigues; positive: right-handed)."""
    t = np.radians(np.asarray(deg, float))[..., None]
    a = np.broadcast_to(axis, v.shape)
    return v * np.cos(t) + np.cross(a, v) * np.sin(t) + a * (np.sum(a * v, -1, keepdims=True)) * (1.0 - np.cos(t))


def _unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


def _rot(axis, deg):
    """The 3x3 rotation about the unit `axis` by `deg` degrees (right-handed)."""
    a, t = np.asarray(axis, float), np.radians(float(deg))
    K = np.array([[0.0, -a[2], a[1]], [a[2], 0.0, -a[0]], [-a[1], a[0], 0.0]])
    return np.eye(3) + np.sin(t) * K + (1.0 - np.cos(t)) * (K @ K)


def _square(v, axis):
    """v with its component along the unit `axis` removed."""
    v = np.asarray(v, float)
    return v - (v @ axis) * axis


def chain(points, rotations):
    """(posed points (n, 3), posed frames [3x3 per joint]) of a finger chain: its rest points (heads, then the tip) with
    each joint turned by its rotation relative to the joint before it (a bone's pose relative to its parent's, in the
    rest frame of the points), as Blender poses the bones: frame_i = frame_(i-1) @ rotation_i."""
    P = np.asarray(points, float)
    D, out, frames = np.eye(3), [P[0]], []
    for i, R in enumerate(rotations[:len(P) - 1]):
        D = D @ R
        frames.append(D)
        out.append(out[-1] + D @ (P[i + 1] - P[i]))
    return np.array(out), frames


def curled(points, palm, angles, ref=None, spread=0.0):
    """[3x3 per joint] of a finger curled by `angles` (deg per joint, toward the palm) about its flexion axis (its first
    segment x the palm normal), its base first turned by `spread` (deg) about the palm normal away from the base point
    `ref` (the middle finger's; the index's for the thumb)."""
    P = np.asarray(points, float)
    d = _unit(P[1] - P[0])
    axis = _unit(np.cross(d, palm))
    turn = np.eye(3)
    if spread and ref is not None:
        away = _square(P[0] - np.asarray(ref, float), palm)
        if np.linalg.norm(away) > 1e-9:
            spin = np.cross(d, _unit(away))
            if np.linalg.norm(spin) > 1e-9:
                turn = _rot(_unit(spin), spread)
    return [(turn if i == 0 else np.eye(3)) @ _rot(axis, a) for i, a in enumerate(list(angles)[:len(P) - 1])]


def _tip(hand, f):
    """Where finger f's skin ends at rest (the hand's `tips`), else its chain's last point."""
    t = (hand.get("tips") or {}).get(f)
    return np.asarray(hand["points"][f], float)[-1] if t is None else np.asarray(t, float)


def finger_end(points, skin):
    """(the point where a finger's skin ends along its last segment, the skin's radius about that segment) from its rest
    chain `points` (its joints' heads, then its tip bone's) and the rest positions (m, 3) of the skin its last bone moves
    most; (None, None) with too little skin. A model's tip bone may stand well past the fingertip or short of it."""
    P, S = np.asarray(points, float), np.asarray(skin, float).reshape(-1, 3)
    if len(S) < 4:
        return None, None
    a, d = P[-2], _unit(P[-1] - P[-2])
    s = (S - a) @ d
    return a + d * float(s.max()), float(np.median(np.linalg.norm((S - a) - np.outer(s, d), axis=1)))


def joints(hand, spec, side):
    """{finger: [3x3 per joint]}: the rotation of each joint (relative to the joint before it, in the frame of the hand's
    points) that gives the hand `side` ("L" / "R") the shape `spec` (`curls`, `spreads`, a solved thumb: THUMBS). `hand`
    is the hand at rest: {"points": {finger: (n, 3)} each finger's joint heads, then its tip (a finger with fewer than
    two points is left out), "palm": its palm normal (out of the palm, the way the fingers curl), and optionally "tips":
    {finger: where its skin ends (`finger_end`)} and "radii": {finger: its last segment's skin radius} (a solved thumb
    touches with those)}. Raises ValueError when a solved thumb lacks the chains it is solved against."""
    angles, spread = curls(spec), spreads(spec)
    mode = thumb_mode(spec)
    P = {f: np.asarray(v, float) for f, v in hand["points"].items() if v is not None and len(v) >= 2}
    palm = _unit(hand["palm"])
    if mode == "tuck":
        if all(f in P and len(P[f]) >= 3 for f in ("thumb", "index", "middle")):
            spread["thumb"], angles["thumb"] = tuck_thumb(P["thumb"], P["index"], P["middle"], palm, angles["index"],
                                                          angles["middle"])
    out = {}
    for f in bonemap.FINGERS:
        if f not in P or (f == "thumb" and mode in ("ring", "cross")):
            continue
        ref = P.get("index" if f == "thumb" else "middle")
        out[f] = curled(P[f], palm, angles[f], None if ref is None or f == "middle" else ref[0], spread[f])
    if "thumb" in P and mode in ("ring", "cross"):
        if "index" not in out or len(P["thumb"]) < 3 or len(P["index"]) < 4 or "middle" not in P:
            raise ValueError(f"fingers: a {mode} thumb needs the hand's thumb (3 or more bones), index (3 bones and its "
                             f"tip) and middle finger")
        out["thumb"] = oppose_thumb(dict(hand, points=P, palm=palm), side, out["index"], out["middle"], mode)[1]
    return out


def pose(hand, spec, side):
    """{"points": {finger: (n, 3)}, "tips": {finger: (3,)}}: where the joints and tips (the bones'), and where the ends
    of the fingers' skin (`finger_end`, else the bones' tips), go when the hand `side` at rest (`joints`) takes the shape
    `spec`."""
    J = joints(hand, spec, side)
    out = {"points": {}, "tips": {}}
    for f, rots in J.items():
        P = np.asarray(hand["points"][f], float)
        out["points"][f], frames = chain(P, rots)
        out["tips"][f] = out["points"][f][-2] + frames[-1] @ (_tip(hand, f) - P[-2])
    return out


# ---------------------------------------------------------------------------------------------------------- the tuck
def posed(points, palm, angles, spread_axis=None, spread=0.0):
    """The rest points (n <= 4, 3) of a finger chain (its joints' heads, then its tip) with its joints curled by `angles`
    (deg, one per joint) about the finger's flexion axis (its first segment x the palm normal) and its base first turned
    by `spread` (deg) about `spread_axis`, as the pose stage turns the bones. Angles and spreads may be arrays (k,): k
    poses at once, (k, n, 3)."""
    P = np.asarray(points, float)
    axis = _unit(np.cross(_unit(P[1] - P[0]), palm))
    angles = [np.asarray(a, float) for a in angles]
    k = np.broadcast(*angles, np.asarray(spread, float)).shape
    out = [np.broadcast_to(P[0], k + (3,))]
    total = np.zeros(k)
    for i in range(len(P) - 1):
        total = total + angles[i]
        v = _turn(np.broadcast_to(P[i + 1] - P[i], k + (3,)).copy(), axis, total)    # one axis: the curls add up
        if spread_axis is not None:
            v = _turn(v, spread_axis, np.broadcast_to(spread, k))
        out.append(out[-1] + v)
    return np.stack(out, -2)


def _seg_dist(X, a, b):
    """Distances (...,) of points X (..., 3) to the segment a-b."""
    ab = b - a
    t = np.clip(np.sum((X - a) * ab, -1) / max(float(ab @ ab), 1e-12), 0.0, 1.0)
    return np.linalg.norm(X - (a + t[..., None] * ab), axis=-1)


def tuck_thumb(thumb, index, middle, palm, index_curls, middle_curls):
    """(spread deg, (a, b, c) deg) that fold the thumb over the curled middle finger, from the rest points (heads, then
    tip) of the thumb, index and middle fingers and the rest palm normal (the fingers curl toward it); the index and middle
    fingers posed by their curls. A grid search: the thumb's tip TUCK_GAP of the middle finger's length from its middle
    and end segments and across it (at most TUCK_ACROSS of the knuckles' spacing short of the line halfway between the
    index and middle knuckles; else it stands up beside the index), every other point of it at least TUCK_CLEAR away
    from both fingers' bones; then the least turning, the base joint costing most."""
    th, ix, md, palm = (np.asarray(x, float) for x in (thumb, index, middle, palm))
    hand = float(np.linalg.norm(np.diff(md, axis=0), axis=1).sum())                  # the middle finger's length
    m = posed(md, palm, middle_curls)
    x = posed(ix, palm, index_curls)
    d = _unit(th[1] - th[0])
    away = th[0] - ix[0]
    away = away - (away @ palm) * palm
    spin = _unit(np.cross(d, _unit(away)))
    S, A, B, C = np.meshgrid(np.arange(-80.0, 21.0, 5.0), np.arange(-20.0, 41.0, 10.0), np.arange(0.0, 91.0, 10.0),
                             np.arange(0.0, 91.0, 10.0), indexing="ij")
    S, A, B, C = S.ravel(), A.ravel(), B.ravel(), C.ravel()
    t = posed(th, palm, (A, B, C), spin, S)                                          # (k, n, 3)
    tip = t[:, -1]
    rest_of = np.concatenate([t[:, 1:-1], 0.5 * (t[:, :-1] + t[:, 1:])], 1)          # its joints and segment middles
    gap = np.minimum(_seg_dist(tip, m[1], m[2]), _seg_dist(tip, m[2], m[3]))
    near = np.min([_seg_dist(rest_of, a, b).min(1) for a, b in zip(np.r_[m[:-1], x[:-1]], np.r_[m[1:], x[1:]])], 0)
    miss = np.abs(gap - TUCK_GAP * hand) + np.maximum(0.0, TUCK_CLEAR * hand - near)    # m from resting, clear
    side = md[0] - ix[0]
    side = side - (side @ palm) * palm                                                # index -> middle, in the palm
    across = (tip - 0.5 * (ix[0] + md[0])) @ _unit(side) + TUCK_ACROSS * np.linalg.norm(side)
    effort = 2.0 * (np.abs(S) + np.abs(A)) + 0.3 * (B + C)     # the base joint and the swing across cost the most
    ok = (miss <= TUCK_SLACK * hand) & (across >= 0.0)
    k = int(np.argmin(np.where(ok, effort, np.inf))) if ok.any() else int(np.argmin(miss - np.minimum(across, 0.0)))
    return float(S[k]), (float(A[k]), float(B[k]), float(C[k]))


# ---------------------------------------------------------------------------------------------------------- opposition
def _radial(points, palm):
    """The hand's thumb side: from its little finger's base (the middle finger's without one) to its index's, in the
    plane of the palm."""
    other = points["little"] if "little" in points else points["middle"]
    return _unit(_square(np.asarray(points["index"], float)[0] - np.asarray(other, float)[0], _unit(palm)))


def thumb_axes(points, palm, side):
    """The axes an opposed thumb (`opposed`) turns about, from the hand's rest points {finger: (n, 3)} and palm normal:
    {"dir": the thumb's way (its second joint to its tip), "pad": the side it bends toward at its last joint (its pad:
    ulnar and toward the palm when the rest chain is straight), "palmar", "radial", "roll", "flex"}: + swings it out of
    the palm's plane toward the palm side, + away from the index in that plane, + rolls it the same way on either hand,
    + bends it toward its pad."""
    T = np.asarray(points["thumb"], float)
    palm = _unit(palm)
    radial = _radial(points, palm)
    d = _unit(T[-1] - T[1]) if len(T) > 3 else _unit(T[-1] - T[0])
    bend = _square(_unit(T[-1] - T[-2]) - _unit(T[-2] - T[-3]), d)
    if np.linalg.norm(bend) < 0.02:
        bend = _square(palm - radial, d)
    pad = _unit(bend)
    return {"dir": d, "pad": pad, "palmar": _unit(np.cross(d, palm)), "radial": _unit(np.cross(d, radial)),
            "roll": (1.0 if side == "R" else -1.0) * d, "flex": _unit(np.cross(d, pad))}


def opposed(axes, angles, n=3):
    """[3x3 per joint] of an opposed thumb of `n` joints: its base turned by its palmar and radial swings and its roll,
    its other joints bent toward its pad, `angles` (deg) in OPPOSE's order (`thumb_axes`)."""
    a = list(angles)
    base = _rot(axes["palmar"], a[0]) @ _rot(axes["radial"], a[1]) @ _rot(axes["roll"], a[2])
    return [base] + [_rot(axes["flex"], v) for v in a[3:3 + n - 1]]


def _least_squares(res, x0, lo, hi, iters=60):
    """x within [lo, hi] (box) that brings the residuals res(x) nearest zero from x0 (Levenberg-Marquardt, the Jacobian
    by central differences): (x, sum of the squared residuals)."""
    x = np.clip(np.asarray(x0, float), lo, hi)
    r = res(x)
    cost, lam = float(r @ r), 1e-2
    for _ in range(iters):
        J = np.empty((len(r), len(x)))
        for j in range(len(x)):
            e = np.zeros(len(x))
            e[j] = 0.25
            J[:, j] = (res(np.clip(x + e, lo, hi)) - res(np.clip(x - e, lo, hi))) / max(
                np.clip(x + e, lo, hi)[j] - np.clip(x - e, lo, hi)[j], 1e-9)
        A, g = J.T @ J, J.T @ r
        moved = False
        for _ in range(10):
            xn = np.clip(x - np.linalg.solve(A + lam * np.diag(np.diag(A) + 1e-6), g), lo, hi)
            rn = res(xn)
            if float(rn @ rn) < cost - 1e-12:
                step, x, r, cost, lam, moved = float(np.abs(xn - x).max()), xn, rn, float(rn @ rn), lam / 3.0, True
                break
            lam *= 4.0
        if not moved or step < 1e-3:
            break
    return x, cost


def oppose_thumb(hand, side, index_joints, middle_joints, mode):
    """(angles (5,) deg in OPPOSE's order, [3x3 per joint]) of the thumb of the hand `side` at rest (`joints`' `hand`)
    that meets its index as `mode` says (THUMBS: "ring", "cross"), the index and middle finger posed by their joints:
    solved from a few starts for the least squares of what it misses by (in hundredths of the middle finger's length),
    how far its other joints come nearer to the index and middle fingers than OPPOSE_CLEAR, and its turning (OPPOSE_COST
    per 20 degrees). The fingers end where their skin does (`finger_end`, the hand's `tips`) and touch skin on skin
    (its `radii`; else RADIUS of the middle finger's length). `ring`: the two fingertips meet, the last segments at
    RING_DOT (a round ring, not a beak). `cross`: the thumb's last segment lies on the index's last one, on its thumb's
    side, at CROSS_AT along each, crossing at CROSS_DEG in the plane square to the index's side."""
    P, palm = hand["points"], _unit(hand["palm"])
    T, ix, md = (np.asarray(P[f], float) for f in ("thumb", "index", "middle"))
    length = float(np.linalg.norm(np.diff(md, axis=0), axis=1).sum())               # the middle finger's
    unit_len = 0.01 * length
    radii = hand.get("radii") or {}
    touch = sum(radii.get(f) or RADIUS * length for f in ("thumb", "index"))
    axes = thumb_axes(P, palm, side)
    I, Fi = chain(ix, index_joints)
    M, _ = chain(md, middle_joints)
    I_tip = I[-2] + Fi[-1] @ (_tip(hand, "index") - ix[-2])
    T_tip = _tip(hand, "thumb") - T[-2]                                             # from its last joint, at rest
    n = len(T) - 1
    i_dir = _unit(I_tip - I[-2])
    lateral = Fi[-1] @ _unit(np.cross(_unit(ix[1] - ix[0]), palm))      # the index's flexion axis as it is posed ...
    lateral = lateral if lateral @ _radial(P, palm) > 0.0 else -lateral  # ... toward the thumb's side
    bones = [(I[0], I[1]), (I[1], I[2]), (M[0], M[1])]
    clear = OPPOSE_CLEAR * length

    def res(x):
        t, frames = chain(T, opposed(axes, x, n))
        tip = t[-2] + frames[-1] @ T_tip
        t_dir = _unit(tip - t[-2])
        if mode == "ring":
            miss = np.concatenate([(tip - I_tip) / unit_len, [10.0 * (t_dir @ i_dir - RING_DOT)]])
        else:
            at = t[-2] + CROSS_AT[0] * (tip - t[-2])
            want = I[-2] + CROSS_AT[1] * (I_tip - I[-2]) + touch * lateral
            miss = np.concatenate([(at - want) / unit_len, [10.0 * (t_dir @ lateral)],
                                   [10.0 * (t_dir @ i_dir - np.cos(np.radians(CROSS_DEG)))]])
        near = np.array([t[1], 0.5 * (t[0] + t[1]), 0.5 * (t[1] + t[2])])
        gaps = np.min([_seg_dist(near, a, b) for a, b in bones], 0)
        return np.concatenate([miss, np.maximum(0.0, clear - gaps) / unit_len, OPPOSE_COST * x / 20.0])

    best = None
    for start in ((0.0, 0.0, 0.0, 0.0, 0.0), (30.0, 0.0, 20.0, 20.0, 20.0), (20.0, -20.0, 30.0, 30.0, 40.0),
                  (45.0, 15.0, 10.0, 10.0, 30.0)):
        x, cost = _least_squares(res, start, OPPOSE_LO, OPPOSE_HI)
        if best is None or cost < best[1] - 1e-9:
            best = (x, cost)
    return best[0], opposed(axes, best[0], n)
