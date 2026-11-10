"""Mitosis, bpy-free (docs/design.md: Frame effects, Cell division): the `[[fx]]` kind `mitosis`, a figure that divides
as a cell does, checked, timed and laid out here; mkmmd/mitosis.py draws it.

The figure (`objects`) sits in a soft cell of its own (a membrane round its outline). On each hit the cell divides: it
is pulled long, pinches in the middle and comes apart into two smaller copies of the figure, each in its own cell, that
wobble for a moment. Odd divisions pull apart sideways, even ones up and down, each a little off true; `max` divisions
make 2 ** max cells. The colony is kept inside the frame (`fit`).

Everything is a function of the entry, its hits and the clip second: a frame comes out the same wherever a render
starts. Lengths are short sides of the frame."""
import math

import numpy as np

KINDS = ("mitosis",)
HITS = {"times", "lane", "min_db", "every"}
KEYS = {"mitosis": {"objects", "split", "max", "gap", "tilt", "fit", "body", "cytoplasm", "membrane", "alpha", "back",
                    "wobble", "seed"} | HITS}
DEFAULTS = {"mitosis": {"split": 0.6, "max": 4, "gap": 0.06, "tilt": 10.0, "fit": 0.92, "body": 0.03,
                        "cytoplasm": "rose", "membrane": "love", "alpha": 0.5, "back": "picture", "wobble": 0.12,
                        "seed": 1, "min_db": None, "every": 1}}
PICTURE = "picture"             # a mitosis's back: the picture, the figure's place in it filled from what is round it
SHRINK = 0.68                   # a daughter cell is this size of its mother (two side by side are then about as wide)
PULL = 0.25                     # a dividing cell is pulled this much longer than it is, half way through
RING = (0.28, 3.0)              # a freed cell's wobble dies over this many seconds, at this many beats a second


def _fx():
    from . import framefx as FX          # framefx imports this module: its helpers are there by the time this runs
    return FX


def normalize(spec, out, palette, what):
    """A mitosis entry's own keys checked and filled in, in place (`out` has the kind's defaults under `spec`)."""
    FX = _fx()
    objs = spec.get("objects")
    objs = [objs] if isinstance(objs, str) else objs
    if not isinstance(objs, (list, tuple)) or not objs or not all(isinstance(p, str) and p.strip() for p in objs):
        raise FX.FxError(f"{what}: objects = {spec.get('objects')!r}: the figure that divides, as object patterns "
                         f"(`name*`, `@collection`, `prop:key`), a list or one")
    out["objects"] = list(objs)
    out["split"] = FX._num(out["split"], f"{what} split", 0.1, 10.0)
    out["max"] = FX._num(out["max"], f"{what} max", 0, 6, integer=True)
    out["gap"] = FX._num(out["gap"], f"{what} gap", 0.0, 2.0)
    out["tilt"] = FX._num(out["tilt"], f"{what} tilt", 0.0, 45.0)
    out["fit"] = FX._num(out["fit"], f"{what} fit", 0.1, 1.0)
    out["body"] = FX._num(out["body"], f"{what} body", 0.002, 0.2)
    out["alpha"] = FX._num(out["alpha"], f"{what} alpha", 0.0, 1.0)
    out["wobble"] = FX._num(out["wobble"], f"{what} wobble", 0.0, 0.5)
    out["seed"] = FX._num(out["seed"], f"{what} seed", lo=0, integer=True)
    out["cytoplasm"] = FX._colour(out["cytoplasm"], palette, f"{what} cytoplasm")
    out["membrane"] = FX._colour(out["membrane"], palette, f"{what} membrane")
    if out["back"] != PICTURE:
        out["back"] = FX._colour(out["back"], palette, f"{what} back")
    FX._hits(spec, out, what, required=False)


def divisions(times, split, most):
    """The clip seconds the divisions start on: each hit of `times` at least `split` after the division before it began
    (a cell does not divide while it divides), `most` of them at most."""
    out = []
    for t in sorted(times):
        if len(out) >= most:
            break
        if not out or t >= out[-1] + split - 1e-9:
            out.append(float(t))
    return out


def state(fx, t, times):
    """(divisions done, how far the next one is (0: none on its way .. 1), seconds since the last one was done) at clip
    second `t`."""
    starts = divisions(times, fx["split"], fx["max"])
    done = [s for s in starts if t >= s + fx["split"]]
    on = [s for s in starts if s <= t < s + fx["split"]]
    since = t - (done[-1] + fx["split"]) if done else math.inf
    return len(done), ((t - on[0]) / fx["split"] if on else 0.0), since


def cells(fx, gen, phase, since, box):
    """The cells after `gen` whole divisions and `phase` (0..1) of the next, `since` seconds after the last one was done:
    (at (n, 2), scale (n,), angle (n,), stretch (n,)). `at` is each cell's middle in short sides from the first cell's
    (x right, y down), `scale` its size against the figure's own, `angle` (radians, counter-clockwise on screen) the way
    it was pulled and `stretch` how long it is drawn (1 + stretch along that way, 1 - stretch / 2 across it). `box`
    (w, h) is the figure's own box in short sides: daughters end `gap` of it further apart than they are wide (odd
    divisions sideways, even ones up and down, each off true by up to `tilt` degrees), each SHRINK the size of its
    mother; half way through, a dividing cell is PULL longer; a freed cell wobbles (`wobble`, dying over RING)."""
    rng = np.random.default_rng([int(fx["seed"]), 4421])
    at, scale = np.zeros((1, 2)), np.ones(1)
    angle, stretch = np.zeros(1), np.zeros(1)
    total = gen + (1 if phase > 0.0 else 0)
    for k in range(1, total + 1):
        raw = 1.0 if k <= gen else float(phase)
        p = raw * raw * (3.0 - 2.0 * raw)
        flat = bool(k % 2)
        a = math.radians((0.0 if flat else 90.0) + (rng.random() * 2.0 - 1.0) * fx["tilt"])
        way = np.array([math.cos(a), -math.sin(a)])
        s2 = scale * (1.0 + (SHRINK - 1.0) * p)
        d = 0.5 * (box[0] if flat else box[1]) * s2 * (1.0 + fx["gap"]) * p
        at = np.concatenate([at + way * d[:, None], at - way * d[:, None]])
        scale = np.concatenate([s2, s2])
        angle = np.full(len(scale), a)
        if k > gen:
            pull = PULL * math.sin(math.pi * raw)
        elif k == gen and math.isfinite(since):
            pull = -fx["wobble"] * math.exp(-since / RING[0]) * math.cos(2.0 * math.pi * RING[1] * since)
        else:
            pull = 0.0
        stretch = np.full(len(scale), pull)
    return at, scale, angle, stretch


def fit(at, scale, stretch, box, centre, frame, share):
    """(k, shift): the colony of cells (`at`, `scale`, `stretch` from `cells`; `box` the figure's own box and `centre` its
    middle, `frame` (w, h), all in short sides) scaled by k about `centre` and moved by `shift` so that it lies inside
    the middle `share` of the frame: 1 and no shift while it does anyway."""
    grow = 1.0 + np.abs(stretch)
    half = 0.5 * np.asarray(box, float)[None, :] * (scale * grow)[:, None]
    lo, hi = (at - half).min(0), (at + half).max(0)
    room = share * np.asarray(frame, float)
    k = float(min(1.0, room[0] / max(hi[0] - lo[0], 1e-9), room[1] / max(hi[1] - lo[1], 1e-9)))
    c = np.asarray(centre, float)
    lo, hi = c + k * lo, c + k * hi
    edge = 0.5 * (np.asarray(frame, float) - room)
    shift = np.maximum(edge - lo, 0.0) - np.maximum(hi - (np.asarray(frame, float) - edge), 0.0)
    return k, shift
