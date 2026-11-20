"""The shatter drawn (docs/design.md: Transitions and inserts), the post layer (numpy, OpenCV): the outgoing picture broken
like a thin sheet of plastic (or paper, or glass) over the incoming one. mkmmd.core.shatter checks it and works out the
shards and their flights; this draws a window frame of it.

Over the first frames (the cracks) the picture is whole and the cracks run out from the impact: each a pale line (the
stressed plastic) with a fine dark line beside it, the shards they have reached each a shade lighter or darker (no longer
quite in one plane). Then every shard that has come away is a piece of the picture on a thin slab, seen in perspective
(its homography, mkmmd.core.shatter.homography, warps the picture onto it): its face lit from the upper left with a glint
where it turns toward the lamp, its back pale once it turns over (glass shows the picture through it, mirrored), its
edge a bright rim where the slab's thickness shows. The shards still in place are the picture itself, their broken edges
lined; under them all the incoming picture, darkened by the soft shadows of the shards in the air (each falls away from
the lamp by its height and is the softer the higher it is). The shards are drawn from the farthest to the nearest, each
only over its own box, so a frame costs about what its shards cover."""
import math
from functools import lru_cache

import cv2
import numpy as np

from .core import shatter as SH
from .core import sheet3d as S3

AMBIENT = 0.55                  # the light a face turned from the lamp still gets
BACK_AMBIENT = 0.78             # and a shard's pale back's (it reads pale whichever way it faces)
SHADE = 0.05                    # a cracked shard's shade, +- (it is no longer quite in the frame's plane)
GLINT_POWER = 40.0              # the glint's tightness
BACK_THROUGH = {"plastic": 0.12, "paper": 0.05, "glass": 0.8}    # how much of the picture shows on a shard's back
EDGE = {"plastic": 1.0, "paper": 0.9, "glass": 1.0}             # the rim's brightness (times the palette's lightest)
CRACK = (0.8, 0.4)              # the pale line's and the dark line's strength (paper: half as much)
PUNCH = 0.02                    # the impact frame's punch of the picture toward the impact
SHADOW_STEP = 4
SHADOW_BANDS = (0.0, 0.04, 0.12, 0.3, 10.0)    # heights (short sides) whose shadows are blurred alike
SHADOW_BLUR = (0.003, 0.16)     # a shadow's blur: short sides plus this share of its height
SHADOW_ALPHA = 0.65             # a shard's own shadow's darkness, before `shadow` (many overlap)
SUB = 3                         # sub-pixel bits for the polygons (cv2's `shift`)


def frame(tr, frame, main, incoming, impact, fps):
    """The outgoing frame `main` (float RGB) broken at `frame` of the transition `tr`'s window over `incoming`, the
    impact at `impact` (px)."""
    h, w = main.shape[:2]
    size = (w, h)
    geo = _geometry(tr["material"], tr["count"], tr["seed"], size,
                    (round(float(impact[0]), 2), round(float(impact[1]), 2)))
    main = np.ascontiguousarray(main, np.float32)
    k = frame - tr["first"]
    nc = SH.crack_frames(tr, fps)
    S = float(min(size))
    clocks = [SH.clock(tr, f, k, fps) for f in geo["flights"]]
    if k < nc:                                                  # the cracks run out
        if k == 0:                                              # the impact punches the picture in toward it
            ix, iy = geo["impact"]
            m = 1.0 - PUNCH
            main = cv2.warpAffine(main, np.float32([[m, 0, (1 - m) * ix], [0, m, (1 - m) * iy]]), size,
                                  flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        reach = geo["reach"] * ((k + 1) / nc) ** 0.75
        out = _shaded(main, geo, geo["dist"] <= reach)
        return _cracks(out, geo, reach, np.zeros(len(clocks), bool), tr, S)
    still = np.array([s <= 0.0 for s in clocks])
    out = np.where(still[geo["label"]][..., None], _shaded(main, geo, np.ones(len(clocks), bool)),
                   np.ascontiguousarray(incoming, np.float32))
    poses = [_pose(geo, i, s, size, tr["material"]) for i, s in enumerate(clocks) if 0.0 < s < 1.0]
    if poses and tr["shadow"] > 0.0:
        sh = _shadow(poses, size)
        out += (np.asarray(tr["ink"], np.float32) - out) * (np.float32(tr["shadow"]) * sh)[..., None]
    out = _cracks(out, geo, geo["reach"] + 1.0, ~still, tr, S)
    mat = SH.MATERIALS[tr["material"]]
    light = np.asarray(tr["back"], np.float32)
    for p in sorted(poses, key=lambda p: p["z"]):
        _draw(out, main, p, tr["material"], mat, light)
    return np.clip(out, 0.0, 1.0)


@lru_cache(maxsize=8)
def _geometry(material, count, seed, size, impact):
    """The shards of a break, their flights, which shard each pixel holds (`label`), the cracks and how far each shard
    lies from the impact."""
    tr = {"material": material, "count": count, "seed": seed}
    shards = SH.cells(tr, size, impact)
    flights = SH.flights(tr, size, impact, shards)
    w, h = size
    label = np.zeros((h, w), np.uint16)
    for c in shards:
        pts = np.round(np.float64(c["poly"]) * (1 << SUB)).astype(np.int32)
        cv2.fillPoly(label, [pts], int(c["index"]), lineType=cv2.LINE_8, shift=SUB)
    ix, iy = impact
    rng = np.random.default_rng(seed)
    return {"shards": shards, "flights": flights, "label": label, "cracks": SH.cracks(shards), "impact": impact,
            "dist": np.float32([math.hypot(f["g"][0] - ix, f["g"][1] - iy) for f in flights]),
            "shade": np.float32(1.0 + SHADE * rng.uniform(-1.0, 1.0, len(shards))),
            "reach": max(math.hypot(x - ix, y - iy) for x in (-0.5, w - 0.5) for y in (-0.5, h - 0.5))}


def _shaded(main, geo, cracked):
    """The picture with each cracked shard a shade lighter or darker."""
    shade = np.where(cracked, geo["shade"], np.float32(1.0)).astype(np.float32)
    return np.clip(main * shade[geo["label"]][..., None], 0.0, 1.0)


def _within(a, b, c, r):
    """The part of the segment a-b inside the circle of radius r about c, or None."""
    dx, dy = b[0] - a[0], b[1] - a[1]
    fx, fy = a[0] - c[0], a[1] - c[1]
    A, B, C = dx * dx + dy * dy, 2.0 * (fx * dx + fy * dy), fx * fx + fy * fy - r * r
    if A <= 1e-12:
        return None
    disc = B * B - 4.0 * A * C
    if disc <= 0.0:
        return None
    sq = math.sqrt(disc)
    t0, t1 = max((-B - sq) / (2.0 * A), 0.0), min((-B + sq) / (2.0 * A), 1.0)
    if t1 <= t0:
        return None
    return (a[0] + dx * t0, a[1] + dy * t0), (a[0] + dx * t1, a[1] + dy * t1)


def _cracks(out, geo, reach, gone, tr, S):
    """The cracks within `reach` px of the impact (clipped to it) drawn on `out` in place: a pale line and a fine dark one
    beside it, toward the lower right. A crack between two shards that have both gone is not drawn; one that has lost a
    neighbour is the broken edge of the shard still there."""
    h, w = out.shape[:2]
    px = S / 1080.0
    pale = np.zeros((h, w), np.uint8)
    dark = np.zeros((h, w), np.uint8)
    lw = max(2, int(round(3.5 * px)))                           # 3 to 4 px at 1080, a core of half that
    o = int(round(max(1.0, 1.3 * px) * (1 << SUB)))
    for i, j, a, b in geo["cracks"]:
        if gone[i] and gone[j]:
            continue
        seg = _within(a, b, geo["impact"], reach)
        if seg is None:
            continue
        P = tuple(int(round(v * (1 << SUB))) for v in seg[0])
        Q = tuple(int(round(v * (1 << SUB))) for v in seg[1])
        cv2.line(pale, P, Q, 150, lw, cv2.LINE_AA, SUB)
        cv2.line(pale, P, Q, 255, max(1, lw // 2), cv2.LINE_AA, SUB)              # its bright core
        cv2.line(dark, (P[0] + o, P[1] + o), (Q[0] + o, Q[1] + o), 255, max(1, lw - 1), cv2.LINE_AA, SUB)
    k = 0.5 if tr["material"] == "paper" else 1.0
    d = (dark.astype(np.float32) * np.float32(k * CRACK[1] / 255.0))[..., None]
    out += (np.asarray(tr["ink"], np.float32) - out) * d
    a = (pale.astype(np.float32) * np.float32(k * CRACK[0] / 255.0))[..., None]
    out += (np.asarray(tr["back"], np.float32) - out) * a
    return out


def _pose(geo, i, s, size, material):
    """Shard i at its own time s: its outline seen (`poly`), its slab's silhouette (`slab`), its homography `H`, its
    height `z` (px, mean), its outline in 3D (`X`) and the normal of the face we see (`N`, `front`)."""
    f = geo["flights"][i]
    c = geo["shards"][i]
    dx, dy, z, R = SH.at(f, s)
    gx, gy = f["g"]
    src = np.float64(c["poly"])
    rel = src - (gx, gy)
    Rm = np.float64(R)
    X = rel @ Rm[:, :2].T + (gx + dx, gy + dy, z)
    N = Rm[:, 2]
    thick = SH.MATERIALS[material]["thick"] * min(size)
    Xb = X - N * thick
    seen = np.stack(S3.project(X[:, 0], X[:, 1], X[:, 2], size)[:2], axis=1)
    back = np.stack(S3.project(Xb[:, 0], Xb[:, 1], Xb[:, 2], size)[:2], axis=1)
    a_src, a_seen = SH.area(c["poly"]), SH.area([tuple(p) for p in seen])
    front = (a_seen > 0) == (a_src > 0)
    hull = cv2.convexHull(np.float32(np.concatenate([seen, back]))).reshape(-1, 2)
    return {"i": i, "poly": seen, "slab": hull, "H": np.float64(SH.homography(f, s, size)), "z": float(X[:, 2].mean()),
            "X": X, "N": N if front else -N, "front": front}


def _shadow(poses, size):
    """(h, w) 0..1: the soft shadows of the shards in the air, each moved away from the lamp by its height and blurred
    the more the higher it is, worked out at 1 / SHADOW_STEP of the size."""
    w, h = size
    S = float(min(size))
    st = SHADOW_STEP
    small = ((w + st - 1) // st, (h + st - 1) // st)
    keep = np.ones((small[1], small[0]), np.float32)
    run = np.float64(S3.SHADOW_RUN)
    for lo, hi in zip(SHADOW_BANDS[:-1], SHADOW_BANDS[1:]):
        band = [p for p in poses if lo * S <= p["z"] < hi * S]
        if not band:
            continue
        canvas = np.zeros((small[1], small[0]), np.uint8)
        for p in band:
            on = (p["X"][:, :2] + p["X"][:, 2:3] * run) / st
            pts = np.round(on * (1 << SUB)).astype(np.int32)
            cv2.fillPoly(canvas, [pts], 255, cv2.LINE_AA, SUB)
        zm = float(np.mean([p["z"] for p in band]))
        sigma = max((SHADOW_BLUR[0] * S + SHADOW_BLUR[1] * zm) / st, 0.5)
        keep *= 1.0 - cv2.GaussianBlur(canvas.astype(np.float32) * np.float32(SHADOW_ALPHA / 255.0), (0, 0), sigma)
    return np.clip(cv2.resize(1.0 - keep, (w, h), interpolation=cv2.INTER_LINEAR), 0.0, 1.0)


def _draw(out, main, p, material, mat, light):
    """Shard pose `p` laid on `out` in place: its slab's rim, then its face (the picture warped onto it, or its back)."""
    h, w = out.shape[:2]
    pts = np.concatenate([p["poly"], p["slab"]])
    x0, y0 = max(int(math.floor(pts[:, 0].min())) - 1, 0), max(int(math.floor(pts[:, 1].min())) - 1, 0)
    x1, y1 = min(int(math.ceil(pts[:, 0].max())) + 2, w), min(int(math.ceil(pts[:, 1].max())) + 2, h)
    if x1 <= x0 or y1 <= y0:
        return
    bw, bh = x1 - x0, y1 - y0

    def mask(poly):
        m = np.zeros((bh, bw), np.uint8)
        q = np.round((poly - (x0, y0)) * (1 << SUB)).astype(np.int32)
        cv2.fillPoly(m, [q], 255, cv2.LINE_AA, SUB)
        return (m.astype(np.float32) * np.float32(1.0 / 255.0))[..., None]

    region = out[y0:y1, x0:x1]
    N = p["N"]
    L, Hf = np.float64(S3.LIGHT), np.float64(S3.HALF)
    amb = AMBIENT if p["front"] else BACK_AMBIENT
    lam = amb + (1.0 - amb) * max(float(N @ L), 0.0) / L[2]
    glint = mat["glint"] * max(float(N @ Hf), 0.0) ** GLINT_POWER
    rim = light * np.float32(EDGE[material] * min(1.0, 0.85 + 0.3 * (1.0 - abs(float(N[2])))))
    me = mask(p["slab"])
    region += (rim - region) * me
    T = np.float64([[1, 0, -x0], [0, 1, -y0], [0, 0, 1]])
    pic = cv2.warpPerspective(main, T @ p["H"], (bw, bh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    if p["front"]:
        face = pic * np.float32(lam)
    else:
        k = np.float32(BACK_THROUGH[material])
        face = (light * (1.0 - k) + pic * k) * np.float32(lam)
    face += (light - face) * np.float32(min(glint, 0.9))
    mf = mask(p["poly"])
    region += (face - region) * mf
