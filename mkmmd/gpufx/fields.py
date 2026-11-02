"""What the colour fields share on the GPU (mkr-post fx/fields.rs, shaders/fields.wgsl): mkmmd.fields' noise calls,
patterns, masks and blends as the CPU path works them out, every constant float32 as numpy makes it."""
import math

import numpy as np

from .. import fields as FL
from ..core import framefx as FX

BLENDS = {"over": 0, "screen": 1, "multiply": 2, "soft-light": 3}      # fields.wgsl's field_blend modes
PATTERNS = {"spiral": 0, "rings": 1, "sunburst": 2, "moire": 3, "checker": 4}   # a drift is the CPU's


def f32(x):
    """`x` rounded to float32, as a float (JSON)."""
    return float(np.float32(x))


def call(tau, seed):
    """One call of mkmmd.fields.noise at time `tau` (lattice slices) with `seed`: {k, seed, w}, its slice and the
    float32 Catmull-Rom weights of the four slices about it."""
    k = math.floor(tau)
    return {"k": int(k), "seed": int(seed), "w": [f32(w) for w in FL._catmull(tau - k)]}


def drawable(spec):
    """Whether mkr-post draws the pattern `spec` (PATTERNS: a drift, mkmmd.fields.drift, stays on the CPU)."""
    return spec["pattern"] in PATTERNS


def backs_drawable(fx):
    """Whether mkr-post draws every pattern among a collage's `back` entries."""
    return all(drawable(b) for b in fx["back"] if isinstance(b, dict))


def region(ffx, fx, frame):
    """The part of the picture a masked entry works on (FrameFx._masked), as mkr-post's Region {layer, ground}; None
    without a mask; False when mkr-post cannot read its figure as the CPU does (no paths, a layer of another size). A
    FileNotFoundError when the layer is missing, as the CPU path raises."""
    if fx.get("mask") is None:
        return None
    paths = ffx.paths
    if paths is None:
        return False
    layer = paths.figure(FX.figure_key(fx), frame)
    if not paths.fits(layer):
        return False
    return {"layer": str(layer), "ground": fx["mask"] == "ground"}


def pattern(spec, q0, centre, unit):
    """mkmmd.fields.pattern(spec, size, q0, centre, unit) as mkr-post's PatternSpec: the pattern's kind, its colours
    and their mean, its constants as pattern() makes them, its centre and unit (px)."""
    C = np.asarray(spec["colors"], np.float32)
    p, m = spec["pattern"], len(C)
    out = {"kind": PATTERNS[p], "colors": C.tolist(), "mean": C.mean(0).tolist(), "q0": f32(q0),
           "center": [f32(centre[0]), f32(centre[1])], "unit": f32(unit)}
    if p in ("spiral", "sunburst"):
        out["n"] = f32(m * spec["arms"])
        if p == "spiral":
            out["tw"] = f32(spec["twist"] / 0.5)
    elif p == "rings":
        out["size"] = f32(spec["size"])
        out["g"] = f32(1.0 / (spec["size"] * unit))
    elif p == "moire":
        out["size"] = f32(spec["size"])
        out["g"] = f32(1.0 / (spec["size"] * unit))
        if spec["moire"] == "lines":
            a = math.radians(spec["angle"])
            out.update(moire=1, ca=f32(math.cos(a)), sa=f32(math.sin(a)))
        else:
            out["o"] = f32(spec["offset"] * spec["size"] / 2.0)
    else:
        a = math.radians(spec["angle"])
        out.update(ca=f32(math.cos(a)), sa=f32(math.sin(a)), size=f32(spec["size"]),
                   warp=f32(FL.BULGE[0] * spec["warp"]), c2k=f32(2.0 / FL.BULGE[1] ** 2))
    return out


def painter(back, w, h):
    """A collage's pattern back (a Painter's spec and phase, `back` = (spec, q)) painted at (w, h): the PatternSpec of
    mkmmd.fields.Painter.__call__ there."""
    spec, q = back
    c = spec["center"]
    return pattern(spec, q, (c[0] * w - 0.5, c[1] * h - 0.5), min(w, h))
