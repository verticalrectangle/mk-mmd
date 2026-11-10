"""Haze, bpy-free (docs/design.md: Frame effects, Haze): the `[[fx]]` kinds that put air and light between the lens and
the picture, checked and laid out here; mkmmd/haze.py draws them.

    mist    the light of the picture's highlights spread round them (a bloom) and, with `diffuse`, the whole picture
            softened as through a diffusion filter: everything glowing and a little smeared
    fog     haze drifting across the picture in a colour or a few, thicker toward the ground with `floor`; with a `mask`
            it lies on the ground behind the figure and only a `veil` of it in front of her
    bokeh   out-of-focus lights: soft discs of colour drifting through the frame and twinkling, swelling on hits; with a
            `mask` they are behind the figure but for a share of the largest (`front`), which pass in front of her

A light of a bokeh is a pure function of the entry, its number and the clip time (`lights`): a frame draws the same
lights wherever a render starts. Lengths are short sides of the frame, so every output shows the same haze."""
import math

import numpy as np

KINDS = ("mist", "fog", "bokeh")
BLENDS = ("screen", "add")
FOG_BLENDS = ("over", "screen")
HITS = {"times", "lane", "min_db", "every"}
KEYS = {"mist": {"radius", "threshold", "knee", "tint", "diffuse", "blend", "mask", "objects"},
        "fog": {"colors", "density", "scale", "speed", "angle", "layers", "detail", "floor", "veil", "blend", "seed",
                "mask", "objects"},
        "bokeh": {"count", "size", "colors", "speed", "angle", "wander", "twinkle", "rate", "soft", "rim", "front",
                  "pulse", "decay", "blend", "seed", "mask", "objects"} | HITS}
DEFAULTS = {"mist": {"radius": 0.03, "threshold": 0.7, "knee": 0.15, "tint": None, "diffuse": 0.0, "blend": "screen",
                     "mask": None},
            "fog": {"colors": ["rose"], "density": 0.5, "scale": 0.6, "speed": 0.03, "angle": 0.0, "layers": 2,
                    "detail": 3, "floor": 0.0, "veil": 0.35, "blend": "over", "seed": 1, "mask": None},
            "bokeh": {"count": 40, "size": [0.015, 0.07], "colors": ["gold", "rose", "foam"], "speed": 0.02,
                      "angle": 90.0, "wander": 0.5, "twinkle": 0.5, "rate": 0.5, "soft": 0.25, "rim": 0.35,
                      "front": 0.15, "pulse": 0.5, "decay": 0.3, "blend": "screen", "seed": 1, "mask": None,
                      "min_db": None, "every": 1}}
MARGIN = 1.25                                   # a bokeh's lights live in the frame grown by their largest radius x this


def _fx():
    from . import framefx as FX          # framefx imports this module: its helpers are there by the time this runs
    return FX


def normalize(spec, out, palette, what):
    """A haze entry's own keys checked and filled in, in place (`out` has its kind's defaults under `spec`)."""
    {"mist": _mist, "fog": _fog, "bokeh": _bokeh}[out["kind"]](_fx(), spec, out, palette, what)


def _mist(FX, spec, out, palette, what):
    out["radius"] = FX._num(out["radius"], f"{what} radius", 0.002, 0.5)
    out["threshold"] = FX._num(out["threshold"], f"{what} threshold", 0.0, 1.0)
    out["knee"] = FX._num(out["knee"], f"{what} knee", 0.0, 1.0)
    out["diffuse"] = FX._num(out["diffuse"], f"{what} diffuse", 0.0, 1.0)
    if out["tint"] is not None:
        out["tint"] = FX._colour(out["tint"], palette, f"{what} tint")
    if out["blend"] not in BLENDS:
        raise FX.FxError(f"{what}: blend {out['blend']!r} is none of {', '.join(BLENDS)}")
    FX._mask(spec, out, what)


def _fog(FX, spec, out, palette, what):
    cols = [out["colors"]] if isinstance(out["colors"], str) else out["colors"]
    if not isinstance(cols, (list, tuple)) or not 1 <= len(cols) <= 4:
        raise FX.FxError(f"{what}: colors = {out['colors']!r}: one to four colours, the thin fog's first")
    out["colors"] = FX._colours(list(cols), palette, f"{what} colors")
    out["density"] = FX._num(out["density"], f"{what} density", 0.0, 1.0)
    out["scale"] = FX._num(out["scale"], f"{what} scale", 0.05, 8.0)
    out["speed"] = FX._num(out["speed"], f"{what} speed", 0.0, 4.0)
    out["angle"] = FX._num(out["angle"], f"{what} angle", -3600.0, 3600.0)
    out["layers"] = FX._num(out["layers"], f"{what} layers", 1, 4, integer=True)
    out["detail"] = FX._num(out["detail"], f"{what} detail", 1, 5, integer=True)
    out["floor"] = FX._num(out["floor"], f"{what} floor", 0.0, 1.0)
    out["veil"] = FX._num(out["veil"], f"{what} veil", 0.0, 1.0)
    out["seed"] = FX._num(out["seed"], f"{what} seed", lo=0, integer=True)
    if out["blend"] not in FOG_BLENDS:
        raise FX.FxError(f"{what}: blend {out['blend']!r} is none of {', '.join(FOG_BLENDS)}")
    FX._mask(spec, out, what)


def _bokeh(FX, spec, out, palette, what):
    out["count"] = FX._num(out["count"], f"{what} count", 0, 400, integer=True)
    size = out["size"]
    if isinstance(size, (int, float)) and not isinstance(size, bool):
        size = [size, size]
    if not isinstance(size, (list, tuple)) or len(size) != 2:
        raise FX.FxError(f"{what}: size = {out['size']!r}: a radius in short sides, or [smallest, largest]")
    lo, hi = (FX._num(v, f"{what} size", 0.002, 0.5) for v in size)
    if hi < lo:
        raise FX.FxError(f"{what}: size = {out['size']!r}: the smallest radius first")
    out["size"] = [lo, hi]
    cols = [out["colors"]] if isinstance(out["colors"], str) else out["colors"]
    if not isinstance(cols, (list, tuple)) or not cols:
        raise FX.FxError(f"{what}: colors = {out['colors']!r}: a colour or a list of them")
    out["colors"] = FX._colours(list(cols), palette, f"{what} colors")
    out["speed"] = FX._num(out["speed"], f"{what} speed", 0.0, 4.0)
    out["angle"] = FX._num(out["angle"], f"{what} angle", -3600.0, 3600.0)
    for k, hi_ in (("wander", 1.0), ("twinkle", 1.0), ("soft", 1.0), ("rim", 2.0), ("front", 1.0), ("pulse", 4.0)):
        out[k] = FX._num(out[k], f"{what} {k}", 0.0, hi_)
    out["rate"] = FX._num(out["rate"], f"{what} rate", 0.0, 12.0)
    out["decay"] = FX._num(out["decay"], f"{what} decay", 0.01, 10.0)
    out["seed"] = FX._num(out["seed"], f"{what} seed", lo=0, integer=True)
    if out["blend"] not in BLENDS:
        raise FX.FxError(f"{what}: blend {out['blend']!r} is none of {', '.join(BLENDS)}")
    FX._hits(spec, out, what, required=False)
    FX._mask(spec, out, what)


def smooth(x):
    """Smoothstep of x clipped to 0..1."""
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def pulse(t, times, decay):
    """How far a bokeh's lights are swollen at clip second `t` by the hits before it: 1 on a hit, dying as
    exp(-age / decay); the strongest counts."""
    ages = [t - h for h in times if h <= t + 1e-9]
    return math.exp(-min(ages) / decay) if ages else 0.0


def lights(fx, t, aspect):
    """The lights of a bokeh entry at clip second `t` in a frame `aspect` (width / height) wide: {"at": (n, 2) their
    centres in short sides from the frame's top left (x right, y down), "r": (n,) radii in short sides, "level": (n,)
    brightness 0..1 with its twinkle, "color": (n,) index into the entry's colours, "front": (n,) bool, the share
    `front` of them with the largest radii, drawn over the figure}, the largest last. Each drifts at `speed` short
    sides a second along `angle` (degrees counter-clockwise from the right: 90 is up), its own pace 0.5 to 1.5 of it
    and its heading turned by up to 60 degrees x `wander`, a slow sway across its way on top, and comes round again at
    the far edge of the frame grown by a margin, so a light never pops in or out inside the picture."""
    n = int(fx["count"])
    r = np.random.default_rng([int(fx["seed"]), int(fx.get("index", 0)), 977])
    u = r.random((n, 8))
    lo, hi = fx["size"]
    rad = lo + (hi - lo) * u[:, 0] ** 2.2                    # most of them small
    w, h = (aspect, 1.0) if aspect >= 1.0 else (1.0, 1.0 / aspect)
    m = MARGIN * hi
    span = np.array([w + 2.0 * m, h + 2.0 * m])
    head = np.radians(fx["angle"] + (u[:, 1] * 2.0 - 1.0) * 60.0 * fx["wander"])
    pace = fx["speed"] * (0.5 + u[:, 2])
    vel = np.c_[np.cos(head), -np.sin(head)] * pace[:, None]             # y runs down the picture
    sway = fx["wander"] * 0.04 * np.sin(2.0 * np.pi * (0.07 + 0.1 * u[:, 5]) * t + 2.0 * np.pi * u[:, 6])
    across = np.c_[np.sin(head), np.cos(head)]
    at = (u[:, 3:5] * span + vel * t + across * sway[:, None]) % span - m
    level = 1.0 - fx["twinkle"] * 0.5 * (1.0 + np.sin(2.0 * np.pi * (fx["rate"] * (0.6 + 0.8 * u[:, 5]) * t + u[:, 7])))
    color = np.minimum((u[:, 6] * len(fx["colors"])).astype(int), len(fx["colors"]) - 1)
    order = np.argsort(rad, kind="stable")
    front = np.zeros(n, bool)
    k = int(round(fx["front"] * n))
    if k:
        front[order[n - k:]] = True
    return {"at": at[order], "r": rad[order], "level": level[order], "color": color[order], "front": front[order]}
