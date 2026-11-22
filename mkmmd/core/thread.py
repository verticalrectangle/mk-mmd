"""Threads, bpy-free (docs/design.md: Frame effects, Sticker sheets): the `[[fx]]` kind `thread` checked and its shape
worked out; mkmmd/thread.py draws it.

    thread  a cotton thread tied to a point of a sheet's live sticker (a point of the figure's own frame, or a world
            point seen through the camera the figure is read through: her crown, carried by the sticker's placement,
            keys and slap on every frame) and running out to its `end` past the frame's edge, streaming: the far part
            follows the tie late (`lag`) and a wave runs out along it from the tie (`wave`; its amplitude keyed down to
            0 pulls it taut)

Places are frame fractions (0, 0 the top left), lengths frame heights, times clip seconds. The shape is a function of the
clip second and of where the tie is seen at the seconds before it, so any frame comes out the same in any order."""
import math

import numpy as np

KINDS = ("thread",)
WAVE = {"amp": 0.02, "length": 0.32, "speed": 1.4}      # frame heights, frame heights, crests a second
SHADOW = {"color": "text", "offset": [0.002, 0.004], "blur": 1.5, "alpha": 0.3}   # offset frame heights, blur px at 1080
KEYS = {"thread": {"sheet", "at", "end", "color", "width", "shadow", "wave", "lag"}}
DEFAULTS = {"thread": {"sheet": None, "at": None, "end": [0.5, -0.2], "color": "surface", "width": 2.5, "shadow": {},
                       "wave": {}, "lag": 0.25}}
POINTS = 160           # points along a thread
LAGS = 13              # the seconds over its lag at which the tie is looked up (between them it is interpolated)
RISE = 0.5             # the wave's amplitude grows from nothing at the tie over this share of a wavelength


# ================================================================================================= the entry
def normalize(spec, out, palette, what):
    """A thread's own keys checked and filled in, in place (`out` has its defaults under `spec`)."""
    from . import framefx as FX          # framefx imports this module (through sheet): its helpers are there by now
    for k in ("sheet", "at"):
        if spec.get(k) is None:
            raise FX.FxError(f"{what}: `{k}` is required: the `name` of the sheet whose live sticker it is tied to "
                             f"(`sheet`) and the point it is tied at (`at`: [x, y] fractions of the figure's own frame, or "
                             f"{{point = \"<expression>\"}}, a world point seen through the camera the figure is read "
                             f"through)")
    if not isinstance(out["sheet"], str) or not out["sheet"].strip():
        raise FX.FxError(f"{what}: sheet = {out['sheet']!r}: the `name` of a sheet with a live sticker (`objects`)")
    out["at"] = FX._center(out["at"], f"{what} at")
    if out["at"]["mode"] == "point":
        out["at"].update(shot=None, back=None, first=None)      # its sheet's camera and the frames it reads: link()
    out["end"] = FX._pair(out["end"], f"{what} end")
    out["color"] = FX._colour(out["color"], palette, f"{what} color")
    out["width"] = FX._num(out["width"], f"{what} width", 0.2, 40.0)
    out["lag"] = FX._num(out["lag"], f"{what} lag", 0.0, 2.0)
    s = out["shadow"]
    if s is None or s is False:
        out["shadow"] = None
    else:
        s = {} if s is True else s
        if not isinstance(s, dict) or set(s) - set(SHADOW):
            raise FX.FxError(f"{what}: shadow = {s!r}: expected {{color, offset, blur, alpha}} (offset [dx, dy] frame "
                             f"heights, blur px at 1080) or false")
        s = {**SHADOW, **s}
        out["shadow"] = {"color": FX._colour(s["color"], palette, f"{what} shadow color"),
                         "offset": FX._pair(s["offset"], f"{what} shadow offset"),
                         "blur": FX._num(s["blur"], f"{what} shadow blur", 0.0, 40.0),
                         "alpha": FX._num(s["alpha"], f"{what} shadow alpha", 0.0, 1.0)}
    w = out["wave"]
    w = {} if w is None or w is True else w
    if not isinstance(w, dict) or set(w) - set(WAVE):
        raise FX.FxError(f"{what}: wave = {out['wave']!r}: expected {{amp, length, speed}} (frame heights, frame heights, "
                         f"crests a second)")
    w = {**WAVE, **w}
    out["wave"] = {"amp": FX._keys(w["amp"], f"{what} wave amp"),
                   "length": FX._num(w["length"], f"{what} wave length", 1e-3, 10.0),
                   "speed": FX._num(w["speed"], f"{what} wave speed", -20.0, 20.0)}
    if any(v < 0.0 for _, v in out["wave"]["amp"]):
        raise FX.FxError(f"{what}: wave amp = {w['amp']!r}: frame heights, 0 or more (0: taut)")


# ================================================================================================= what it reads
def tie_key(fx):
    return f"x{fx['index']}a"


def own_points(fx):
    """[(key, expression, shot)]: the world point a thread is tied at (its `at = {point}`), seen through the camera its
    sheet's figure is read through (point/x<i>a/, with its size there) on the frames from `at.first` to its span's last:
    that sheet's plate `shot`, None for the shot in the cut on each frame (link() gives both)."""
    a = fx["at"]
    return [(tie_key(fx), a["expr"], a["shot"])] if a["mode"] == "point" else []


def looks_back(fx, frame):
    """The frames before `frame` whose world point a thread tied to one reads on `frame`, over its lag: the oldest first,
    none before `at.first`."""
    a = fx["at"]
    return range(max(a["first"], frame - a["back"]), frame)


def link(fxs, fps, frame0):
    """Each thread tied to a world point (and its aspect.<output> entries) gets in its `at`: `shot`, the camera it is
    seen through, the plate of the sheet it is tied to (by its `name`; None: the shot in the cut); `back`, the frames its
    far part looks back over (as many as any of its entries' `lag` takes); `first`, the first frame its point is read
    on (its span's first less `back`, not before the clip's first, `frame0`). An FxError when no sheet of that name lays
    a live sticker of its own."""
    from . import framefx as FX
    for fx in fxs:
        if fx["kind"] not in KINDS or fx["at"]["mode"] != "point":
            continue
        sheet = next((s for s in fxs if s["kind"] == "sheet" and s.get("name") == fx["sheet"]), None)
        if sheet is None or sheet.get("objects") is None or sheet.get("clip") is not None:
            raise FX.FxError(f"fx {fx['index']} (thread): sheet = {fx['sheet']!r}: no sheet of that `name` lays a live "
                             f"sticker of its own (`objects` and no `clip`) for its world point to be seen with")
        entries = [fx, *fx["aspect"].values()]
        back = max(int(math.ceil(e["lag"] * fps - 1e-9)) for e in entries)
        for e in entries:
            if e["at"]["mode"] == "point":
                e["at"].update(shot=sheet.get("shot"), back=back, first=max(fx["f0"] - back, int(frame0)))


# ================================================================================================= the shape
def amp_at(wave, t):
    """The wave's amplitude (frame heights) at clip second `t`: its `amp` keys, linear between them, held outside."""
    keys = wave["amp"]
    if t <= keys[0][0]:
        return keys[0][1]
    for (t0, v0), (t1, v1) in zip(keys, keys[1:]):
        if t <= t1:
            return v0 + (v1 - v0) * (t - t0) / (t1 - t0)
    return keys[-1][1]


def shape(tie, end, t, wave, lag, height):
    """The thread's points (POINTS, 2), frame px, at clip second `t`, from its tie out to `end` (frame px): `tie(tau)` is
    where the tie is seen (frame px) at clip second tau, `height` the frame's height px. The point a share u of the way
    out lies u of the way from where the tie was `lag` * u seconds before to the end (the far part follows the tie's
    moves late), pushed across the line from the tie to the end by a wave that runs out from the tie at `speed` crests
    a second, `length` frame heights from crest to crest, its amplitude growing from nothing at the tie over RISE of a
    wavelength."""
    u = np.linspace(0.0, 1.0, POINTS)
    ties = np.array([tie(t - lag * k / (LAGS - 1)) for k in range(LAGS)], float)
    k = u * (LAGS - 1)
    i = np.minimum(np.floor(k).astype(int), LAGS - 2)
    f = (k - i)[:, None]
    late = ties[i] * (1.0 - f) + ties[i + 1] * f
    e = np.asarray(end, float)
    pts = late + (e - late) * u[:, None]
    d = e - ties[0]
    length = max(math.hypot(d[0], d[1]), 1e-6)
    across = np.array([-d[1], d[0]]) / length
    s = u * length / height                                  # frame heights from the tie, along it
    lam = wave["length"]
    rise = np.clip(s / (RISE * lam), 0.0, 1.0)
    rise = rise * rise * (3.0 - 2.0 * rise)
    y = amp_at(wave, t) * rise * np.sin(2.0 * math.pi * (s / lam - wave["speed"] * t))
    return pts + np.outer(y * height, across)
