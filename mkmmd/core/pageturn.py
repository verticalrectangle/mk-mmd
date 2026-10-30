"""The page turn, bpy-free and numpy-free (docs/design.md: Transitions and inserts): `[[transition]] kind = "page"`, a
picture-book page turn. The outgoing frame is a page: it lifts at its edge (a corner first, by `angle`), curls over a
roll of `radius` that travels across the frame, and lies turned over on itself, its paper back showing, until all of it
has left the frame; under it is the next shot. This module checks it and does its timing and geometry; mkmmd.pageturn
draws it.

    dir = "left"        the way the page turns over: "left" (lifted at the right edge and turned toward the left, as a
                        book's page), "right", "up" or "down"
    angle = 20          degrees the curl's line is turned off square to `dir`, clockwise positive: the page lifts at one
                        corner first (with "left", 20 lifts the bottom right corner first, -20 the top right)
    radius = 0.12       the curl's radius, a share of the frame's short side
    back                the paper of its back (default the palette's lightest)
    through = 0.18      how much of the picture shows through the paper from behind, 0..1
    shadow = 0.45       how dark the shadow it casts on what is under it, 0..1
The window is the `dur` seconds (default 0.6, eased `inout`) that end at the cut, as a slash's. At the clip's end (`at` =
`[project] duration`, where no shot starts) the turn is the loop's wrap: it turns the clip's last frames, as finished (their
frame effects and screen type on), over onto its first frame as finished, held still under the page, so the video loops
through it (mkmmd.core.transition.wrap_of, mkmmd.pageturn.wrap).

The geometry, in px along the curl's normal n (from the part still flat toward the lifted edge; x' of a pixel is its
place along n): with the curl's line at c, a point of the page x' = c + u past it (u > 0) is rolled onto the cylinder
of radius R that touches the page along the line: at the angle u / R up round it while u < pi R (seen from the front at
c + R sin(u / R)), and laid back flat, turned over, at c - (u - pi R) beyond that. So over the strip c .. c + R the frame
sees the roll's top (the page's back) where the page reaches round it, else the inside of the roll (its front); before c
the turned-over part (its back) where the page reaches that far, else the flat page; past c + R what is under it."""
import math

from . import tween as TW

DIRS = {"left": (1.0, 0.0), "right": (-1.0, 0.0), "up": (0.0, 1.0), "down": (0.0, -1.0)}   # n with angle 0 (y down)
KEYS = {"at", "kind", "dur", "ease", "dir", "angle", "radius", "back", "through", "shadow"}
DEFAULTS = {"dir": "left", "angle": 20.0, "radius": 0.12, "through": 0.18, "shadow": 0.45}
DUR, EASE = 0.6, "inout"
MAX_ANGLE = 60.0                # degrees: a steeper line would lift a long edge rather than a corner


def normalize(spec, out, palette, what):
    """A page turn's own keys with their defaults laid into `out` (the transition's common keys, normalised by
    mkmmd.core.transition.normalize_transition), colours resolved."""
    from . import transition as TR
    from .iris import _lightest
    d = spec.get("dir", DEFAULTS["dir"])
    if d not in DIRS:
        raise TR.TransitionError(f"{what}: dir = {d!r}, expected one of {tuple(DIRS)} (the way the page turns over)")
    out["dir"] = d
    for k, lo, hi in (("angle", -MAX_ANGLE, MAX_ANGLE), ("radius", 0.01, 0.5), ("through", 0.0, 1.0),
                      ("shadow", 0.0, 1.0)):
        v = spec.get(k, DEFAULTS[k])
        try:
            if isinstance(v, bool):
                raise TypeError
            out[k] = float(v)
        except (TypeError, ValueError):
            raise TR.TransitionError(f"{what}: {k} = {v!r}: expected a number") from None
        if not lo <= out[k] <= hi:
            raise TR.TransitionError(f"{what}: {k} = {out[k]}: expected {lo} .. {hi}")
    out["back"] = TR._colour(spec.get("back", _lightest(palette)), palette, f"{what}: back")
    return out


def normal(tr):
    """(nx, ny): the unit normal of the curl's line, from the page's flat part toward its lifted edge (x right, y down):
    `dir`'s, turned `angle` degrees clockwise on the screen."""
    bx, by = DIRS[tr["dir"]]
    a = math.radians(tr["angle"])
    return (bx * math.cos(a) - by * math.sin(a), bx * math.sin(a) + by * math.cos(a))


def roll(tr, size):
    """The curl's radius in px in a frame `size` = (w, h)."""
    return tr["radius"] * min(size)


def travel(tr, size):
    """(c0, c1): where the curl's line runs along the normal (px, x' = x nx + y ny of index coordinates) from the first
    window frame's side to the last's: c0 at the page's farthest corner (nothing lifted yet), c1 past its nearest corner
    by the roll and a pixel (all of it gone)."""
    nx, ny = normal(tr)
    w, h = size
    xs = [x * nx + y * ny for x in (-0.5, w - 0.5) for y in (-0.5, h - 0.5)]
    return max(xs), min(xs) - roll(tr, size) - 1.0


def progress(tr, k):
    """0..1 of the way the curl has travelled at window frame `k`: eased, never exactly 0 or 1 inside the window, so the
    page has lifted on the first window frame and is gone on the cut."""
    return TW.ease((k + 1) / (tr["n"] + 1), tr["ease"])


def line(tr, k, size):
    """Where the curl's line is at window frame `k` (px along the normal, as `travel`)."""
    c0, c1 = travel(tr, size)
    return TW.lerp(c0, c1, progress(tr, k))


def wrap(data, cut, cuts, fps, frame0):
    """The shot a page turn at frame `cut` turns onto when that is the clip's end (`[project] duration`, the frame after
    its last) and no shot starts there: the loop's wrap, onto the shot of the cut on its first frame (`frame0`); None
    otherwise."""
    from . import shotstyle as SS
    duration = (data.get("project") or {}).get("duration")
    if duration is None or cut != int(frame0) + int(round(float(duration) * fps)):
        return None
    if any(c["from"] == cut for c in cuts):
        return None
    return SS.shot_at(cuts, int(frame0))
