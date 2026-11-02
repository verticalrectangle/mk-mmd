"""Photographs as prints, bpy-free (docs/design.md: Frame effects, Prints): the `[[fx]]` kind `prints` checked, timed
and laid out. Real photographs (pictures of the asset registry: slugs, tags, files or globs) are dealt over the picture as
physical prints, one landing on each hit (`times` or a `lane`): it falls from above, lands on the hit and settles. Each
is the photograph whole on a cream card (a mounted print, its border deeper at the foot, or a stereocard: both halves of a
stereograph side by side under arched tops), held by tape strips across two corners or by pushpins pushed in as it lands,
casting a soft drop shadow, turned a little. A print shows its photograph whole: nothing of it is cut away or retouched.

    scatter  the prints spread over a region of the frame (best-candidate places: each as far from the others as it can be)
    ring     the prints on a ring round a point (a scene point projected on every frame), the ring turning slowly
    pin      a base print, one photograph laid large, and the prints pinned over a region of it: a scrapbook page

Print k shows photograph k of the deal (`order`: as given, or shuffled by `seed`, every one before any comes again) on
place k mod `count`, over what lies there. Places are worked out in short sides of the frame (its shorter side is 1, so
outputs of one aspect lay them out alike) and a pin's in fractions of its base photograph; the fall, the settle and the
ring's turn are functions of the clip second, so any frame comes out the same in any order. mkmmd/prints.py draws them."""
import math

import numpy as np

from . import tween as TW

KINDS = ("prints",)
MODES = ("scatter", "ring", "pin")
STYLES = ("print", "stereo")                  # a mounted print; a stereocard (both halves under arched tops)
FIXES = ("tape", "pins", "none")
ORDERS = ("given", "shuffle")
TONES = ("own", "albumen", "palette")
# the tone of Alfred A. Hart's albumen prints as the Library of Congress scanned them, dark to light: the median colour at
# nine evenly spaced levels of luma over twelve of his stereographs' halves (each print's own 0.5 % / 99.5 % luma its
# black and white). A photograph toned "albumen" takes its luma through it: a copy print of another sits with them
ALBUMEN = ["#231b1b", "#3c302a", "#57493f", "#6e6254", "#847c6c", "#999485", "#adb09e", "#c1cabd", "#cfdfd4"]
PALETTE_TONE = "gold"                         # the middle of a "palette" tone: its darkest neutral, this, its lightest
NEUTRALS = ("base", "text")                   # the palette's neutrals a "palette" tone runs between (dark to light)
SIZE = {"scatter": 0.3, "ring": 0.24, "pin": 0.16}           # a print's longer side: short sides of the frame
COUNT = {"scatter": 6, "ring": 8, "pin": 5}                  # places
REGION = {"scatter": [0.03, 0.04, 0.97, 0.96],               # where the prints lie, a print's half extent inside it:
          "pin": [0.04, 0.66, 0.96, 0.99]}                   # frame fractions, a pin's of its base photograph
FIX = {"scatter": "tape", "ring": "tape", "pin": "pins"}
CARD = {"color": "gold:#f8f2e4:0.85", "border": 0.06, "foot": 1.8, "arch": 0.3}
TAPE = {"colors": ["gold:#fffdf6:0.82"], "alpha": 0.5, "width": 0.14, "length": 0.34}
PINS = {"colors": ["love", "pine", "gold", "iris", "foam"], "size": 0.018}
BASE = {"photo": None, "at": [0.5, 0.5], "size": 0.95, "angle": -1.2, "t": None, "tone": None}
MODE_KEYS = {"ring": ("center", "radius", "turn", "phase"), "pin": ("base",), "scatter": ()}
MAX_COUNT = 24                                # places at most
MAX_SHOWN = 64                                # prints drawn at most: the latest (older ones lie under them)
TRIES = 32                                    # candidates a best-candidate place is picked from
LAND = 0.3                                    # the share of its slide and spin a print has left as it lands
BOUNCE = 0.06                                 # the hop it makes as it lands, a share of its lift
HOP = 0.45                                    # the share of the settle the hop takes
PIN_PUSH = 0.07                               # seconds a pin takes to go in, from 1 + PIN_POP times its size
PIN_POP = 0.6
CORNERS = ((0, 2), (1, 3), (0, 1))            # the corners a print's two tape strips cross: TL BR, TR BL, TL TR
CORNER_AT = (lambda w, h: (0.0, 0.0), lambda w, h: (w, 0.0), lambda w, h: (w, h), lambda w, h: (0.0, h))
CORNER_IN = ((1.0, 1.0), (-1.0, 1.0), (-1.0, -1.0), (1.0, -1.0))  # into the card along each corner's diagonal (y down)
CORNER_TAPE = (45.0, -45.0, 45.0, -45.0)      # a strip's length across each corner: square to its diagonal
TAPE_ON = 0.45                               # the share of a strip's length over the card
STEREO = 0.5                                  # a stereocard's height over its width (17.5 x 8.7 cm)
KEYS = {"prints": {"mode", "photos", "order", "count", "size", "style", "card", "fix", "tape", "pins", "shadow", "tilt",
                   "drop", "settle", "lift", "spin", "tone", "toning", "center", "radius", "turn", "phase", "region",
                   "base", "seed", "times", "lane", "min_db", "every"}}
DEFAULTS = {"prints": {"mode": "scatter", "order": "given", "count": None, "size": None, "style": "print", "card": None,
                       "fix": None, "tape": None, "pins": None, "shadow": {}, "tilt": 7.0, "drop": 0.16, "settle": 0.3,
                       "lift": 0.4, "spin": 14.0, "tone": "own", "toning": 1.0, "center": [0.5, 0.5], "radius": 0.32,
                       "turn": 6.0, "phase": 0.0, "region": None, "base": None, "seed": 1, "min_db": None,
                       "every": 1}}


# ================================================================================================= the entries
def normalize(spec, out, palette, what):
    """A prints entry's own keys checked and filled in, in place (`out` has the kind's defaults under `spec`)."""
    from . import framefx as FX          # framefx imports this module; its helpers are all there by the time this runs
    from . import mandala as MD
    mode = out["mode"]
    if mode not in MODES:
        raise FX.FxError(f"{what}: mode {mode!r} is none of {', '.join(MODES)}")
    for m, keys in MODE_KEYS.items():
        stray = sorted(k for k in keys if k in spec and m != mode)
        if stray:
            raise FX.FxError(f"{what}: {', '.join(f'`{k}`' for k in stray)} belong{'s' if len(stray) == 1 else ''} to "
                             f"mode = \"{m}\", not \"{mode}\"")
    if "photos" not in spec:
        raise FX.FxError(f"{what}: `photos` is required: registry slugs, files, globs or {{tag = ...}}, one or a list")
    out["photos"] = MD._refs(FX, spec["photos"], what, "photos")
    if out["order"] not in ORDERS:
        raise FX.FxError(f"{what}: order {out['order']!r} is none of {', '.join(ORDERS)}")
    out["count"] = FX._num(COUNT[mode] if out["count"] is None else out["count"], f"{what} count", 1, MAX_COUNT,
                           integer=True)
    out["size"] = FX._num(SIZE[mode] if out["size"] is None else out["size"], f"{what} size", 1e-3, 4.0)
    if out["style"] not in STYLES:
        raise FX.FxError(f"{what}: style {out['style']!r} is none of {', '.join(STYLES)}")
    out["card"] = _card(FX, out["card"], palette, what)
    out["fix"] = FIX[mode] if out["fix"] is None else out["fix"]
    if out["fix"] not in FIXES:
        raise FX.FxError(f"{what}: fix {out['fix']!r} is none of {', '.join(FIXES)}")
    out["tape"] = _table(FX, out["tape"], TAPE, palette, what, "tape")
    out["pins"] = _table(FX, out["pins"], PINS, palette, what, "pins")
    out["shadow"] = None if out["shadow"] is False else FX._shadow(out["shadow"], palette, what)
    for k in ("tilt", "drop", "settle", "lift", "spin"):
        out[k] = FX._num(out[k], f"{what} {k}", lo=0.0)
    out["tone"] = tone(FX, out["tone"], palette, what)
    out["toning"] = FX._num(out["toning"], f"{what} toning", 0.0, 1.0)
    out["seed"] = FX._num(out["seed"], f"{what} seed", lo=0, integer=True)
    FX._hits(spec, out, what, required=False)
    if mode == "ring":
        out["center"] = FX._center(out["center"], f"{what} center")
        out["radius"] = FX._num(out["radius"], f"{what} radius", lo=0.0)
        out["turn"] = FX._num(out["turn"], f"{what} turn")
        out["phase"] = FX._num(out["phase"], f"{what} phase")
    else:
        out["center"] = None
    if mode == "ring":
        if "region" in spec:
            raise FX.FxError(f"{what}: `region` is a scatter's or a pin's: a ring lays its prints round its `center`")
        out["region"] = None
    else:
        out["region"] = _region(FX, REGION[mode] if out["region"] is None else out["region"], what)
    if mode == "pin":
        if spec.get("base") is None:
            raise FX.FxError(f"{what}: mode = \"pin\" needs a `base`: {{photo = \"<slug or file>\", at, size, angle, t}}, "
                             f"the photograph the prints are pinned over")
        out["base"] = _base(FX, spec["base"], out["tone"], palette, what)


def _card(FX, v, palette, what):
    """`card`: {color, border, foot, arch} with CARD's defaults: the card's colour, its border (a share of the
    photograph's shorter side), the foot's (times the border) and a stereocard's arches (their corners' radius, a share
    of a window's width)."""
    c = _keys(FX, v, CARD, what, "card")
    return {"color": FX._colour(c["color"], palette, f"{what} card color"),
            "border": FX._num(c["border"], f"{what} card border", 0.0, 0.5),
            "foot": FX._num(c["foot"], f"{what} card foot", 0.0, 6.0),
            "arch": FX._num(c["arch"], f"{what} card arch", 0.0, 0.5)}


def _keys(FX, v, base, what, name):
    if v is None:
        return dict(base)
    if not isinstance(v, dict) or set(v) - set(base):
        raise FX.FxError(f"{what}: {name} = {v!r}: expected a table of {', '.join(sorted(base))}")
    return {**base, **v}


def _table(FX, v, base, palette, what, name):
    """`tape` {colors, alpha, width, length} (its strips' colours, round the prints, their opacity, width and length as
    shares of a print's shorter side) or `pins` {colors, size} (their heads' colours round the prints, a head's
    diameter in short sides of the frame)."""
    t = _keys(FX, v, base, what, name)
    out = {"colors": FX._colours(t["colors"], palette, f"{what} {name} colors")}
    for k in sorted(set(base) - {"colors"}):
        hi = 1.0 if k == "alpha" else 4.0
        out[k] = FX._num(t[k], f"{what} {name} {k}", 0.0 if k == "alpha" else 1e-4, hi)
    return out


def tone(FX, v, palette, what):
    """`tone`: "own" (the photograph as it is: None), "albumen" (its luma through ALBUMEN), "palette" (through the
    palette's darkest neutral, its PALETTE_TONE and its lightest neutral) or a list of colours, dark to light -> the
    colours of the map (display floats), or None."""
    if v == "own":
        return None
    if v == "albumen":
        return [FX._colour(c, palette, f"{what} tone") for c in ALBUMEN]
    if v == "palette":
        dark, light = sorted((FX._colour(s, palette, f"{what} tone") for s in NEUTRALS), key=_luma)
        return [dark, FX._colour(PALETTE_TONE, palette, f"{what} tone"), light]
    if isinstance(v, (list, tuple)) and len(v) >= 2 and not all(isinstance(x, (int, float)) for x in v):
        return FX._colours(v, palette, f"{what} tone")
    raise FX.FxError(f"{what}: tone = {v!r}: one of {', '.join(TONES)}, or a list of colours, dark to light")


def _luma(c):
    return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]


def _region(FX, v, what):
    if not (isinstance(v, (list, tuple)) and len(v) == 4):
        raise FX.FxError(f"{what}: region = {v!r}: expected [x0, y0, x1, y1] in 0..1")
    r = [FX._num(x, f"{what} region", 0.0, 1.0) for x in v]
    if not (r[0] <= r[2] and r[1] <= r[3]):
        raise FX.FxError(f"{what}: region = {v!r}: x0 <= x1 and y0 <= y1 (a point puts every print there)")
    return r


def _base(FX, v, entry_tone, palette, what):
    """A pin's `base`: {photo, at, size, angle, t, tone}: the photograph (a slug or a file), where its card's middle
    lies (frame fractions), its longer side (short sides of the frame), its turn (degrees counter-clockwise), the clip
    second it lands (none: it lies there from the span's start) and its own `tone` (none: the entry's, `entry_tone`)."""
    b = _keys(FX, v, BASE, what, "base")
    if not isinstance(b["photo"], str) or not b["photo"].strip():
        raise FX.FxError(f"{what}: base photo = {b['photo']!r}: a registry slug or a file")
    return {"photo": b["photo"].strip(), "at": FX._pair(b["at"], f"{what} base at"),
            "size": FX._num(b["size"], f"{what} base size", 1e-3, 4.0),
            "angle": FX._num(b["angle"], f"{what} base angle"),
            "t": None if b["t"] is None else FX._num(b["t"], f"{what} base t"),
            "tone": entry_tone if b["tone"] is None else tone(FX, b["tone"], palette, f"{what} base")}


# ================================================================================================= the deal
def landings(fx, times):
    """The clip second each print lands, in the order they are dealt: one on each hit (`times`, clip seconds) before the
    span's end; with no hits to deal on (`times` None: the entry gives neither `times` nor a `lane`) one on every place,
    lying there from the start (-inf)."""
    if times is None:
        return [-math.inf] * fx["count"]
    return [float(t) for t in sorted(times) if t < fx["to"]]


def shown(fx, t, lands):
    """The prints on the page or falling onto it at clip second `t`, in the order they are laid (the latest on top):
    the indices k of `lands` whose fall has begun, the latest MAX_SHOWN of them."""
    ks = [k for k, tl in enumerate(lands) if t >= tl - fx["drop"]]
    return ks[-MAX_SHOWN:]


def fall(fx, t, t_land):
    """(h, r) of a print landing at clip second `t_land`, at clip second `t`: its height (0 on the page, 1 at its lift)
    and the share of its slide and spin it has left (1 where it starts, 0 at rest), or None before its fall begins. It
    falls over `drop` seconds, gathering speed (h = 1 - s^2), lands on `t_land` with LAND of its slide left, hops
    BOUNCE of its lift and settles over `settle` seconds, its slide easing out past its rest a little and back."""
    if t_land == -math.inf:
        return 0.0, 0.0
    if t < t_land:
        d = fx["drop"]
        if d <= 0.0 or t < t_land - d:
            return None
        s = (t - (t_land - d)) / d
        return 1.0 - s * s, LAND + (1.0 - LAND) * (1.0 - s * s)
    v = (t - t_land) / fx["settle"] if fx["settle"] > 0.0 else 1.0
    if v >= 1.0:
        return 0.0, 0.0
    h = BOUNCE * math.sin(math.pi * min(v / HOP, 1.0)) if v < HOP else 0.0
    return h, LAND * (1.0 - TW.pop(v, 0.08))


def pin_push(t, t_land):
    """A pin's size, times its own, at clip second `t` for a print landing at `t_land`: none before it lands (None),
    then pushed in from 1 + PIN_POP over PIN_PUSH seconds."""
    if t < t_land:
        return None
    if t_land == -math.inf:
        return 1.0
    return 1.0 + PIN_POP * (1.0 - TW.ease((t - t_land) / PIN_PUSH, "out"))


def jitter(fx, k):
    """Print k's own: (its tilt, degrees counter-clockwise at rest; the spin it falls with, degrees; where its fall
    starts, (dx, dy) short sides from its place; the corners its tape crosses, CORNERS' index), seeded by `seed`, the
    entry and k."""
    rng = np.random.default_rng([fx["seed"], fx.get("index", 0), 7, k])
    tilt = float(rng.uniform(-1.0, 1.0)) * fx["tilt"]
    spin = float(rng.choice((-1.0, 1.0)) * rng.uniform(0.5, 1.0)) * fx["spin"]
    a = float(rng.uniform(0.0, 2.0 * math.pi))
    reach = 0.25 * fx["size"] * float(rng.uniform(0.5, 1.0))
    return tilt, spin, (reach * math.cos(a), reach * math.sin(a)), int(rng.integers(len(CORNERS)))


def photo_pick(fx, k, n):
    """Which of `n` photographs print k shows: as given round and round (`order` "given"), or shuffled anew each time
    round ("shuffle", seeded), every one before any comes again."""
    if fx["order"] == "given":
        return k % n
    perm = np.random.default_rng([fx["seed"], fx.get("index", 0), 11, k // n]).permutation(n)
    return int(perm[k % n])


def pick(values, k):
    """Entry k of a list of colours, round and round (print k's tape or pin)."""
    return values[k % len(values)]


# ================================================================================================= the places
def spread(n, w, h, rng, tries=TRIES):
    """`n` points in a box `w` x `h` placed one by one, each the candidate (of `tries` drawn by `rng`) farthest from
    those before it: an even scatter that still looks dealt by hand. The first is the candidate nearest the middle."""
    pts = []
    for i in range(n):
        cand = rng.uniform(0.0, 1.0, (tries, 2)) * (w, h)
        if not pts:
            best = int(np.argmin(np.hypot(cand[:, 0] - w / 2.0, cand[:, 1] - h / 2.0)))
        else:
            p = np.asarray(pts)
            d = np.hypot(cand[:, None, 0] - p[None, :, 0], cand[:, None, 1] - p[None, :, 1]).min(1)
            best = int(np.argmax(d))
        pts.append((float(cand[best, 0]), float(cand[best, 1])))
    return pts


def half_extent(fx):
    """A print's nominal half width and half height, short sides of the frame: half its `size` each way (a stereocard's
    height half its width). Places keep this far inside their region, whichever photograph lands there."""
    s = fx["size"] / 2.0
    return (s, s * STEREO) if fx["style"] == "stereo" else (s, s)


def inset(lo, hi, by):
    """The span lo .. hi moved in by `by` at both ends, or its middle when it is not twice as long."""
    return (lo + by, hi - by) if hi - lo > 2.0 * by else ((lo + hi) / 2.0,) * 2


def scatter_places(fx, aspect):
    """A scatter's `count` places, [(x, y)] short sides of a frame `aspect` (width over height) wide, from its top left:
    spread over its `region`, kept a print's half extent inside it."""
    unit_w, unit_h = (aspect, 1.0) if aspect >= 1.0 else (1.0, 1.0 / aspect)
    hw, hh = half_extent(fx)
    x0, x1 = inset(fx["region"][0] * unit_w, fx["region"][2] * unit_w, hw)
    y0, y1 = inset(fx["region"][1] * unit_h, fx["region"][3] * unit_h, hh)
    rng = np.random.default_rng([fx["seed"], fx.get("index", 0), 3])
    return [(x0 + x, y0 + y) for x, y in spread(fx["count"], x1 - x0, y1 - y0, rng)]


def pin_places(fx, photo_aspect, half=(0.0, 0.0)):
    """A pin's `count` places, [(u, v)] fractions of its base photograph (`photo_aspect` its width over its height):
    spread over its `region` of it, kept `half` (a print's half extent, fractions of the photograph's width and
    height) inside it."""
    u0, u1 = inset(fx["region"][0], fx["region"][2], half[0])
    v0, v1 = inset(fx["region"][1], fx["region"][3], half[1])
    rng = np.random.default_rng([fx["seed"], fx.get("index", 0), 5])
    return [(u0 + x / photo_aspect, v0 + y) for x, y in spread(fx["count"], (u1 - u0) * photo_aspect, v1 - v0, rng)]


def ring_offset(fx, j, t):
    """Place j of a ring at clip second `t`, (dx, dy) short sides from its centre (y down): `count` places evenly round
    it from `phase` degrees counter-clockwise from straight up, all turning `turn` degrees a second since the span's
    start."""
    a = math.radians(fx["phase"] + 360.0 * j / fx["count"] + fx["turn"] * (t - fx["from"]))
    return -fx["radius"] * math.sin(a), -fx["radius"] * math.cos(a)


# ================================================================================================= the cards
def card_layout(pw, ph, border, foot):
    """A mounted print's card for a photograph `pw` x `ph`: (width, height, (x, y) of the photograph's top left on it),
    the border `border` times the photograph's shorter side, the foot `foot` times the border."""
    b = border * min(pw, ph)
    return pw + 2.0 * b, ph + b + foot * b, (b, b)


def stereo_layout(width, aspects):
    """A stereocard `width` wide (STEREO of it tall) holding two windows side by side, the photographs' `aspects` (width
    over height, left and right): (height, [(x, y, w, h) of each window]). The windows share a height: 0.76 of the card's
    under a top margin of 0.07, or less where they would come nearer the sides than 0.045 of its width; a gap of 0.03 of
    its width between them, the pair in the middle."""
    height = width * STEREO
    gap, side = 0.03 * width, 0.045 * width
    wh = 0.76 * height
    if wh * sum(aspects) + gap > width - 2.0 * side:
        wh = (width - 2.0 * side - gap) / sum(aspects)
    ws = [wh * a for a in aspects]
    x = (width - sum(ws) - gap) / 2.0
    top = 0.07 * height
    out = []
    for w in ws:
        out.append((x, top, w, wh))
        x += w + gap
    return height, out


def tape_strips(w, h, corners, width, length):
    """The two tape strips of a card `w` x `h` crossing the corners `corners` (CORNERS' index; TL, TR, BR, BL are 0..3):
    [(middle (x, y), angle (degrees counter-clockwise of the strip's length; y down), width, length)], each square to
    its corner's diagonal with TAPE_ON of its length over the card and the rest stuck down beyond its two edges.
    `width` and `length` are shares of the card's shorter side."""
    s = min(w, h)
    sw, sl = width * s, length * s
    d = TAPE_ON * sl / (2.0 * math.sqrt(2.0))                      # the strip's middle that far in along the diagonal
    out = []
    for c in CORNERS[corners]:
        (px, py), (ix, iy) = CORNER_AT[c](w, h), CORNER_IN[c]
        out.append(((px + ix * d, py + iy * d), CORNER_TAPE[c], sw, sl))
    return out
