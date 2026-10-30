"""Construction-paper cut-outs, numpy only (docs/design.md: Looks, vector: `cutout`): the vector look's flat colours cut
out of coloured paper and laid in layers like a paper puppet. compose_vector (mkmmd.core.shotstyle) hands over the flat
drawing it painted (no lines: the pieces' edges and shadows draw them), the line group of every pixel (a piece of paper
per group: the face, each hair part, the blazer...) and the depth, inside Blender and in mk's own renderer alike.

Every piece is cut with slightly rough scissors (its edge wanders a pixel or two, cut again `fps` times a second: a
stop-motion boil), coloured paper with a mottle and fibres of its own, grown a little and set off its place by up to
`lift` px (a puppet's pieces never sit exactly on each other), and laid far to near, each casting a soft shadow on the
pieces under it. The whole figure stands on a white paper rim and casts a drop shadow on the background's paper."""
import math

import numpy as np

from . import crayon as CR

RIM = {"color": "#FFFFFF", "width": 10.0}                                 # px at 1080: the white paper round the figure
SHADOW = {"color": "#000000", "offset": [0.005, 0.008], "blur": 5.0, "alpha": 0.35}
CUTOUT = {"rim": RIM, "shadow": SHADOW, "rough": 1.6, "grain": 0.6, "lift": 3.0, "fps": 8.0, "seed": 1}
PIECE_SHADOW = 0.45     # a piece's shadow on the pieces under it: this share of the figure's offset and blur, alpha 0.8
ROUGH_CELLS = ((7.0, 0.7), (1.8, 0.3))   # (cell px at 1080, weight) of a scissor cut's wander


class CutoutError(ValueError):
    """A bad `cutout` table."""


def _num(v, what, lo=None, hi=None):
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(float(v)):
        raise CutoutError(f"cutout: {what} = {v!r} is not a number")
    v = float(v)
    if (lo is not None and v < lo) or (hi is not None and v > hi):
        raise CutoutError(f"cutout: {what} = {v:g} is out of range ({lo if lo is not None else '-inf'} .. "
                          f"{hi if hi is not None else 'inf'})")
    return v


def normalize(v, colour):
    """`cutout = true | {rim, shadow, rough, grain, lift, fps, seed}` with CUTOUT's defaults, its colours turned into
    [r, g, b] by `colour(value, what)`; None without one. `rim = false` and `shadow = false` leave them out."""
    if v is None or v is False:
        return None
    v = {} if v is True else v
    if not isinstance(v, dict):
        raise CutoutError(f"cutout = {v!r}: expected true or a table {{{', '.join(CUTOUT)}}}")
    unknown = sorted(set(v) - set(CUTOUT))
    if unknown:
        raise CutoutError(f"cutout: unknown keys {unknown} (known: {', '.join(CUTOUT)})")
    out = {"rough": _num(v.get("rough", CUTOUT["rough"]), "rough", 0.0, 40.0),
           "grain": _num(v.get("grain", CUTOUT["grain"]), "grain", 0.0, 1.0),
           "lift": _num(v.get("lift", CUTOUT["lift"]), "lift", 0.0, 40.0),
           "fps": _num(v.get("fps", CUTOUT["fps"]), "fps", 0.0, 120.0)}
    seed = v.get("seed", CUTOUT["seed"])
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise CutoutError(f"cutout: seed = {seed!r}: a whole number, 0 or more")
    out["seed"] = seed
    rim = v.get("rim", True)
    if rim is False:
        out["rim"] = None
    else:
        rim = {} if rim is True else rim
        if not isinstance(rim, dict) or set(rim) - set(RIM):
            raise CutoutError(f"cutout: rim = {rim!r}: expected false or {{color, width}} (px at 1080)")
        out["rim"] = {"color": colour(rim.get("color", RIM["color"]), "cutout rim color"),
                      "width": _num(rim.get("width", RIM["width"]), "rim width", 0.0, 200.0)}
    sh = v.get("shadow", True)
    if sh is False:
        out["shadow"] = None
    else:
        sh = {} if sh is True else sh
        if not isinstance(sh, dict) or set(sh) - set(SHADOW):
            raise CutoutError(f"cutout: shadow = {sh!r}: expected false or {{color, offset, blur, alpha}}")
        off = sh.get("offset", SHADOW["offset"])
        if not isinstance(off, (list, tuple)) or len(off) != 2:
            raise CutoutError(f"cutout: shadow offset = {off!r}: expected [dx, dy] in frame heights")
        out["shadow"] = {"color": colour(sh.get("color", SHADOW["color"]), "cutout shadow color"),
                         "offset": [_num(off[0], "shadow offset"), _num(off[1], "shadow offset")],
                         "blur": _num(sh.get("blur", SHADOW["blur"]), "shadow blur", 0.0, 200.0),
                         "alpha": _num(sh.get("alpha", SHADOW["alpha"]), "shadow alpha", 0.0, 1.0)}
    return out


def boil_step(spec, t):
    """Which cut of the scissors a frame at clip second `t` shows: a new one `fps` times a second (0: one cut)."""
    if t is None or spec["fps"] <= 0:
        return 0
    return int(math.floor(float(t) * spec["fps"] + 1e-6))


def _warped(a, dy, dx):
    """`a` ((h, w) or (h, w, c)) as a wandering pair of scissors cut it: every pixel takes the value `dy`, `dx` away."""
    h, w = a.shape[:2]
    yy = np.clip(np.rint(np.arange(h, dtype=np.float32)[:, None] + dy), 0, h - 1).astype(np.intp)
    xx = np.clip(np.rint(np.arange(w, dtype=np.float32)[None, :] + dx), 0, w - 1).astype(np.intp)
    return a[yy, xx]


def _shifted(a, dx, dy):
    """`a` (h, w) moved by whole pixels (dx right, dy down), what comes in from outside empty."""
    out = np.zeros_like(a)
    h, w = a.shape[:2]
    dx, dy = int(dx), int(dy)
    if abs(dx) >= w or abs(dy) >= h:
        return out
    out[max(dy, 0):h + min(dy, 0), max(dx, 0):w + min(dx, 0)] = a[max(-dy, 0):h - max(dy, 0), max(-dx, 0):w - max(dx, 0)]
    return out


def _spread(colour, mask, inside, r):
    """`colour` (h, w, 3) carried from the pixels of `inside` out over the rest of `mask` up to `r` px: a piece grown past
    its cut keeps its own colours at the margin."""
    have = inside.copy()
    out = colour.copy()
    for _ in range(int(math.ceil(r))):
        if not (mask & ~have).any():
            break
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            src = _shifted(have, dx, dy)
            take = src & mask & ~have
            if take.any():
                out[take] = np.stack([_shifted(out[..., c], dx, dy) for c in range(3)], -1)[take]
                have = have | take
    return out


def _box(cover, margin):
    ys, xs = np.nonzero(cover)
    h, w = cover.shape
    return (max(int(ys.min()) - margin, 0), min(int(ys.max()) + margin + 1, h),
            max(int(xs.min()) - margin, 0), min(int(xs.max()) + margin + 1, w))


def compose(flat, pieces, depth, background, spec, t, dilate):
    """The cut-out picture, float32 (H, W, 3), from the vector look's flat drawing `flat` (H, W, 3: every pixel its
    fill's colour, no lines), its pieces (H, W int: the line group of every pixel, 0 the background), the depth (H, W
    metres) and the background colour, at the clip second `t` (the scissors' boil). `dilate(mask, r)` grows a mask by
    a disc (shotstyle.dilate_round)."""
    H, W = pieces.shape
    px = min(H, W) / 1080.0
    step = boil_step(spec, t)
    grain = np.float32(spec["grain"])
    sheet = CR.fibres((H, W), px, spec["seed"] * 7919 + 1, spec["grain"])
    out = np.asarray(background, np.float32)[None, None, :] * sheet[..., None]
    cover = pieces > 0
    if not cover.any():
        return out.astype(np.float32)
    rim = spec["rim"]["width"] * px if spec["rim"] else 0.0
    sh = spec["shadow"]
    reach = (abs(sh["offset"][0]) + abs(sh["offset"][1])) * H + 3.0 * sh["blur"] * px if sh else 0.0
    y0, y1, x0, x1 = _box(cover, int(math.ceil(rim + 2.0 * spec["rough"] * px + 2.0 * spec["lift"] * px + reach)) + 4)
    P0, F0, Z0 = pieces[y0:y1, x0:x1], flat[y0:y1, x0:x1], depth[y0:y1, x0:x1]
    h, w = P0.shape
    rng = np.random.default_rng([spec["seed"], step, 4242])
    rough = np.float32(spec["rough"] * px)
    dy = sum(np.float32(k) * CR.noise(rng, (h, w), c * px) for c, k in ROUGH_CELLS) * rough
    dx = sum(np.float32(k) * CR.noise(rng, (h, w), c * px) for c, k in ROUGH_CELLS) * rough
    P, F = _warped(P0, dy, dx), _warped(F0, dy, dx)
    pad = 64
    tex = CR.fibres((h + 2 * pad, w + 2 * pad), px, spec["seed"] * 104729 + 3, 1.0)
    pic = out[y0:y1, x0:x1].copy()

    def shadowed(sub, m, scale, alpha):
        """`sub` under the soft shadow `m` casts, set off by the shadow's offset times `scale`."""
        off = sh["offset"]
        s = CR.blur(_shifted(m.astype(np.float32), round(off[0] * H * scale), round(off[1] * H * scale)),
                    0.5 * sh["blur"] * px * scale)
        a = (np.float32(alpha) * s)[..., None]
        return sub * (1.0 - a) + np.asarray(sh["color"], np.float32) * a

    def paper(j, a0, a1, b0, b1):
        """The paper of piece `j` over rows a0..a1 and columns b0..b1 of the box: a sheet of its own."""
        r = np.random.default_rng([spec["seed"], int(j), 99])
        oy, ox = int(r.integers(0, 2 * pad)), int(r.integers(0, 2 * pad))
        return (np.float32(1.0) + grain * (tex[oy + a0:oy + a1, ox + b0:ox + b1] - np.float32(1.0)))[..., None]
    if spec["rim"] or sh:
        figure = dilate(P > 0, rim) if rim > 0.5 else P > 0
        if sh:
            pic = shadowed(pic, figure, 1.0, sh["alpha"])
        if spec["rim"]:
            pic[figure] = (np.asarray(spec["rim"]["color"], np.float32) * paper(0, 0, h, 0, w))[figure]
    ids = [int(g) for g in np.unique(P0) if g > 0]
    far = {g: float(np.median(Z0[P0 == g])) for g in ids}
    lift = spec["lift"] * px
    margin = int(math.ceil(2.0 * lift + (PIECE_SHADOW * (reach + 2.0 * sh["blur"] * px) if sh else 0.0))) + 2
    for g in sorted(ids, key=lambda g: (-far[g], g)):
        ys, xs = np.nonzero(P == g)
        if not len(ys):
            continue
        a0, a1 = max(int(ys.min()) - margin, 0), min(int(ys.max()) + margin + 1, h)
        b0, b1 = max(int(xs.min()) - margin, 0), min(int(xs.max()) + margin + 1, w)
        m, colour = P[a0:a1, b0:b1] == g, F[a0:a1, b0:b1]
        if lift >= 0.5:
            r = np.random.default_rng([spec["seed"], g, 31])
            turn = r.uniform(0.0, 2.0 * math.pi)
            ox, oy = int(round(lift * math.cos(turn))), int(round(lift * math.sin(turn)))
            grown = dilate(m, lift)
            colour = _spread(colour, grown, m, lift)
            m = _shifted(grown, ox, oy)
            colour = np.stack([_shifted(colour[..., c], ox, oy) for c in range(3)], -1)
        sub = pic[a0:a1, b0:b1]
        if sh:
            sub = shadowed(sub, m, PIECE_SHADOW, 0.8 * sh["alpha"])
        sub[m] = (colour * paper(g, a0, a1, b0, b1))[m]
        pic[a0:a1, b0:b1] = sub
    out[y0:y1, x0:x1] = pic
    return out.astype(np.float32)
