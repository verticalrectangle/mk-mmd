"""The picture as flat shapes, bpy-free (docs/design.md: Frame effects, Shapes): `[[fx]] kind = "shapes"` checked and
timed. The picture work is mkmmd/shapes.py.

The frame is laid out in cells, `cell` short sides apart, in offset rows (`hex`) or a square `grid`, centred on the
frame's middle; in each cell one flat shape (a `circle`, `square`, `triangle` or `diamond`, dealt from `shape` by the
`seed`) in the colour of the picture under the cell, snapped to the nearest of `inks` when it gives them (or each
shape kind its own ink, or the inks dealt round), on the picture or on a flat colour (`back`). A cell whose colour is
the back's draws nothing (`skip`), so a figure on a plain ground comes apart into shapes and the ground stays empty.
The cell size moves with its keys (`[[t, v], ...]`, as a `strength`); with hits (`times`, a `lane`) it changes only on
them, the picture refining (or coarsening) on the beat, each new layout's shapes popping in (`pop`, staggered by
`stagger` cell by cell). Shapes may be turned at random (`turn`) and spin (`spin`). A `mask` keeps them to the figure
or the ground."""
import math

import numpy as np

KINDS = ("shapes",)
SHAPES = ("circle", "square", "triangle", "diamond")
LAYOUTS = ("hex", "grid")
PICKS = ("nearest", "shape", "deal")
SKIP = 0.06                                     # a cell this close to a flat back's colour (RGB distance) draws nothing
CELL = (0.004, 1.0)                             # short sides: a cell's size from 4 px at 1080 to the whole short side
OVERSHOOT = 2.16429                             # an ease-out-back's constant whose peak is 15 % past the shape's size
KEYS = {"shapes": {"cell", "layout", "shape", "size", "turn", "spin", "inks", "pick", "back", "skip", "pop", "stagger",
                   "seed", "times", "lane", "min_db", "every", "mask", "objects"}}
DEFAULTS = {"shapes": {"cell": 0.05, "layout": "hex", "shape": ["circle", "triangle", "square"], "size": 0.92,
                       "turn": 0.0, "spin": 0.0, "inks": None, "pick": None, "back": "picture", "skip": None,
                       "pop": 0.15, "stagger": 0.1, "seed": 1, "min_db": None, "every": 1, "mask": None}}


def _fx():
    from . import framefx as FX          # framefx imports this module: its helpers are there by the time this runs
    return FX


def normalize(spec, out, palette, what):
    """A shapes entry's own keys checked and filled in, in place (`out` has the kind's defaults under `spec`)."""
    FX = _fx()
    out["cell"] = FX._keys(out["cell"], f"{what} cell")
    for _, v in out["cell"]:
        FX._num(v, f"{what} cell", *CELL)
    if out["layout"] not in LAYOUTS:
        raise FX.FxError(f"{what}: layout {out['layout']!r} is none of {', '.join(LAYOUTS)}")
    shapes = [out["shape"]] if isinstance(out["shape"], str) else out["shape"]
    if not isinstance(shapes, (list, tuple)) or not shapes or any(s not in SHAPES for s in shapes):
        raise FX.FxError(f"{what}: shape = {out['shape']!r}: expected one of {', '.join(SHAPES)}, or a list of them")
    out["shape"] = list(shapes)
    out["size"] = FX._num(out["size"], f"{what} size", 0.05, 1.0)
    out["turn"] = FX._num(out["turn"], f"{what} turn", 0.0, 180.0)
    out["spin"] = FX._num(out["spin"], f"{what} spin", -3600.0, 3600.0)
    if out["inks"] is None:
        if out["pick"] is not None:
            raise FX.FxError(f"{what}: pick chooses among `inks`: give the inks")
    else:
        out["inks"] = FX._colours(out["inks"], palette, f"{what} inks")
        out["pick"] = "nearest" if out["pick"] is None else out["pick"]
        if out["pick"] not in PICKS:
            raise FX.FxError(f"{what}: pick {out['pick']!r} is none of {', '.join(PICKS)}")
    if out["back"] == "picture":
        out["back"] = None
        if out["skip"] is not None:
            raise FX.FxError(f"{what}: skip leaves the cells of a flat back's colour empty: give the `back` colour")
    else:
        out["back"] = FX._colour(out["back"], palette, f"{what} back")
        out["skip"] = FX._num(SKIP if out["skip"] is None else out["skip"], f"{what} skip", 0.0, 2.0)
    out["pop"] = FX._num(out["pop"], f"{what} pop", 0.0, 2.0)
    out["stagger"] = FX._num(out["stagger"], f"{what} stagger", 0.0, 2.0)
    out["seed"] = FX._num(out["seed"], f"{what} seed", lo=0, integer=True)
    FX._hits(spec, out, what, required=False)
    FX._mask(spec, out, what)


def cell_at(fx, t, times):
    """(cell, since, step) of a shapes entry at clip second `t`: its cell size in short sides, the clip second its
    layout was laid (its last hit at or before `t`, else the span's start: the shapes pop in from there) and how many
    hits came before (a new deal each). Without hits the cell follows its keys on every frame, laid once at the start."""
    FX = _fx()
    keys = {"strength": fx["cell"]}
    if not times:
        return FX.strength(keys, t), fx["from"], 0
    past = [h for h in times if h <= t + 1e-9]
    since = past[-1] if past else fx["from"]
    return FX.strength(keys, since), since, len(past)


def lattice(size, cell, layout):
    """(dx, dy) px between the cells of a frame of `size` (w, h): `cell` short sides across, the rows `dx` x sqrt 3 / 2
    apart in a hex layout (every other row moved half a cell along), `dx` in a grid; at least 2 px."""
    dx = max(float(cell) * min(size), 2.0)
    return dx, (dx * math.sqrt(3.0) / 2.0 if layout == "hex" else dx)


def pop(u):
    """The size of a shape popping in, u = 0 .. 1 of its pop: out of nothing, 15 % past its size, settling."""
    u = np.clip(np.asarray(u, np.float64), 0.0, 1.0)
    return 1.0 + (OVERSHOOT + 1.0) * (u - 1.0) ** 3 + OVERSHOOT * (u - 1.0) ** 2


def deal(q, r, step, seed, n):
    """`n` numbers in 0..1 for each cell (q, r) of layout `step` (arrays, broadcast together): (..., n), the same for the
    same cell, layout and seed on any frame (splitmix64 of them)."""
    q = np.asarray(q, np.int64).astype(np.uint64)
    r = np.asarray(r, np.int64).astype(np.uint64)
    base = (q * np.uint64(0x9E3779B97F4A7C15)) ^ (r * np.uint64(0xC2B2AE3D27D4EB4F)) ^ \
        np.uint64((int(step) * 0x165667B19E3779F9 + int(seed) * 0x27D4EB2F165667C5) & 0xFFFFFFFFFFFFFFFF)
    out = []
    with np.errstate(over="ignore"):
        for i in range(n):
            z = base + np.uint64((i + 1) * 0x9E3779B97F4A7C15 & 0xFFFFFFFFFFFFFFFF)
            z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
            z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
            z = z ^ (z >> np.uint64(31))
            out.append((z >> np.uint64(11)).astype(np.float64) * (1.0 / (1 << 53)))
    return np.stack(out, axis=-1)
