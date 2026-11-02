"""The purikura look of a photo booth, bpy-free (docs/design.md: Picture effects: pixel sort, film, camcorder, doodles,
photo booth): `[[fx]] kind = "booth"` with `look = "purikura"` and the look's own keys in a `purikura` table, checked,
timed and laid out here; mkmmd/purikura.py draws it into the strip mkmmd.picturefx.booth makes.

    beauty filter  each capture lifted toward its lights (no further than a light key, KEY: a light capture keeps its
                   picture), soft-focused and blooming in its whites (`lift`, `soft`, `bloom`, the bloom tinted `glow`)
    frames         a pastel band round each photo (`frames`, `frame`), the strip's paper pastel with polka dots (`dots`)
    stamps         the booth's `doodles` as stickers: doodle shapes (`stamps`) filled in the pen's inks with a white rim,
                   near the photo's edges; and `marks`, stamps set on every photo (ears on her head, blush on her cheeks)
    writing        a date or a short word written over each photo by the gel-pen hand (mkmmd.core.hand) at `speed`,
                   boiling with the booth's doodle pen (`write`)
    flash          no more than FLASH_RATE flashes start in any second: a capture past them is taken without one

Its colours are the palette's: `pastel` mixes the frames and the glow (and the booth's paper a little further, PAPER)
toward the palette's light neutral, the lighter of `base` and `text`, so the strip is pastel with a dark palette and a
light one. A frame stays FRAME_APART from the paper (less pastel, or deepened toward the dark neutral, where the mix
would melt into it), and the booth's caption reads on its paper (`readable`, mkmmd.core.picturefx); how far apart two
colours stand is measured on the palette's own values, whatever palette it is. The look sets the booth's own keys in
BOOTH that the entry leaves out (its paper, its back, its stamps a photo, ...)."""
import math

import numpy as np

from . import eralook as ER
from . import hand as HD

LOOKS = ("plain", "purikura")
BOOTH = {"paper": "rose", "back": "love", "doodles": 3, "border": 24.0, "color": "love"}   # the look's booth keys
DEFAULTS = {"frames": ["rose", "iris", "foam", "gold"], "pastel": 0.5, "frame": 16.0, "dots": 8.0,
            "stamps": ["heart", "sparkle", "twinkle", "bow", "star", "flower"], "inks": ["love", "iris", "foam", "gold"],
            "marks": [], "write": [], "letters": 0.13, "speed": 14.0, "lift": 0.45, "soft": 0.5, "bloom": 0.7,
            "glow": "rose"}
MARK = {"shape": None, "at": None, "size": 0.2, "color": None, "angle": 0.0}   # a mark's keys (None: required, or the
#                                                                                first ink)
PAPER = 0.3                      # the paper is mixed this share of the rest further toward the light neutral
PEN = {"width": 0.14, "outline": 0.1, "wobble": 0.025, "turn": 1.5, "grow": 0.015}   # the writing's pen, in caps
WRITE_AT = 0.83                  # the writing's line: its middle, photo heights down
WRITE_WIDTH = 0.84               # the writing at most this share of a photo's width
WRITE_LATE = 0.15                # s after a photo lands: the pen starts writing on it
POP_GAP = 0.09                   # s between a photo's stamps popping in
FLASH_RATE = ER.FLASH_RATE       # flashes starting in any second, at most
TEXT_MAX = 24                    # characters written on a photo
CLOSE = 0.08                     # a stroke that comes back this near its start (a share of its size) is closed
HELD = 0.8                       # a stroke this share of whose points lie inside a closed one is drawn across it
TINT = 0.6                       # a stamp's holes and the lines across it: its ink mixed this far to the light neutral
FRAME_APART = 15.0               # a photo's frame stands at least this far from the paper round it (CIELAB delta E)
CAPTION_CONTRAST = 3.0           # a booth's caption reads on its paper at least this well (a contrast ratio, WCAG's for
#                                  large type)
KEY = 0.6                        # the beauty filter lifts a capture's key (its mean luma) to this at most
STEPS = 100                      # a colour moved toward another until it stands apart: in this many steps


def _rng(*key):
    return np.random.default_rng([int(k) for k in key])


# ================================================================================================= checks
def preset(spec, out):
    """The look's own defaults (BOOTH) for the booth's keys the entry leaves out, in place, before they are checked."""
    if out.get("look") == "purikura":
        for k, v in BOOTH.items():
            if k not in spec:
                out[k] = v


def luma(c):
    return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]


def mix(a, b, t):
    """Colour `a` mixed `t` of the way to `b` (display-space floats, as "a:b:t" mixes them)."""
    return [round(float(x) * (1.0 - t) + float(y) * t, 6) for x, y in zip(a, b)]


def light(FX, palette, what):
    """The palette's light neutral, the lighter of `base` and `text`: `text` in a dark palette, `base` in a light one."""
    return max((FX._colour(s, palette, what) for s in ("base", "text")), key=luma)


def dark(FX, palette, what):
    """The palette's dark neutral, the darker of `base` and `text`: `base` in a dark palette, `text` in a light one."""
    return min((FX._colour(s, palette, what) for s in ("base", "text")), key=luma)


def _linear(c):
    return [x / 12.92 if x <= 0.04045 else ((x + 0.055) / 1.055) ** 2.4 for x in c]


def contrast(a, b):
    """The contrast ratio of two display-space colours, 1 (alike) .. 21 (black on white): their relative luminances
    plus 0.05, the lighter over the darker (WCAG's)."""
    la, lb = (0.2126 * r + 0.7152 * g + 0.0722 * bl + 0.05 for r, g, bl in (_linear(a), _linear(b)))
    return max(la, lb) / min(la, lb)


def lab(c):
    """CIELAB (D65) of a display-space colour: (L* 0..100, a*, b*)."""
    r, g, b = _linear(c)
    xyz = ((0.4124 * r + 0.3576 * g + 0.1805 * b) / 0.95047, 0.2126 * r + 0.7152 * g + 0.0722 * b,
           (0.0193 * r + 0.1192 * g + 0.9505 * b) / 1.08883)
    fx, fy, fz = (v ** (1.0 / 3.0) if v > 216.0 / 24389.0 else (24389.0 / 27.0 * v + 16.0) / 116.0 for v in xyz)
    return 116.0 * fy - 16.0, 500.0 * (fx - fy), 200.0 * (fy - fz)


def apart(a, b):
    """How far apart two display-space colours look: their CIELAB delta E (1976); about 2 is just told apart, 10
    plainly."""
    return math.dist(lab(a), lab(b))


def framing(c, light_, t, paper, dark_):
    """A photo's frame colour: `c` mixed `t` toward the light neutral `light_` (pastel), or as much less of the way as it
    takes to stand FRAME_APART from the `paper` round it; when even `c` does not, `c` mixed toward the dark neutral
    `dark_` as far as it takes (a frame in the paper's own colour stays a band of its own)."""
    f = None
    for k in range(STEPS + 1):
        f = mix(c, light_, t * (1.0 - k / STEPS))
        if apart(f, paper) >= FRAME_APART:
            return f
    for k in range(1, STEPS + 1):
        f = mix(c, dark_, k / STEPS)
        if apart(f, paper) >= FRAME_APART:
            return f
    return f


def readable(c, ground, dark_, light_, least=CAPTION_CONTRAST):
    """`c` as it reads on `ground` at a contrast of `least` at least: itself when it does, else mixed toward the palette's
    neutral that reads better there (`dark_` on a light ground, `light_` on a dark one) as little as it takes."""
    if contrast(c, ground) >= least:
        return c
    end = dark_ if contrast(dark_, ground) >= contrast(light_, ground) else light_
    m = c
    for k in range(1, STEPS + 1):
        m = mix(c, end, k / STEPS)
        if contrast(m, ground) >= least:
            return m
    return m


def _colour_list(FX, v, palette, what, most=8):
    v = [v] if isinstance(v, str) else v
    if not isinstance(v, (list, tuple)) or not 1 <= len(v) <= most:
        raise FX.FxError(f"{what} = {v!r}: expected a colour or a list of 1 to {most} (palette slots, mixes \"a:b:t\")")
    return [FX._colour(c, palette, what) for c in v]


def _shape(FX, v, what):
    from . import picturefx as PC
    if v not in PC.SHAPES:
        raise FX.FxError(f"{what}: shape {v!r} is none of {', '.join(PC.SHAPES)}")
    return v


def _texts(FX, v, what):
    v = [v] if isinstance(v, str) else v
    if not isinstance(v, (list, tuple)) or not all(isinstance(s, str) for s in v):
        raise FX.FxError(f"{what} = {v!r}: expected a short text or a list of them, one a photo (\"\" for none)")
    for s in v:
        if len(s) > TEXT_MAX or (s and not s.strip()):
            raise FX.FxError(f"{what}: {s!r}: expected a short text (1..{TEXT_MAX} characters, or \"\" for none)")
        try:
            HD.check(s, what)
        except HD.HandError as e:
            raise FX.FxError(str(e)) from None
    return list(v)


def _marks(FX, v, palette, ink, what):
    if not isinstance(v, (list, tuple)) or not all(isinstance(m, dict) for m in v):
        raise FX.FxError(f"{what}: marks = {v!r}: expected a list of {{shape, at, size, color, angle}}")
    out = []
    for j, m in enumerate(v):
        w = f"{what} mark {j}"
        extra = sorted(set(m) - set(MARK))
        if extra:
            raise FX.FxError(f"{w}: unknown keys {extra} (known: {sorted(MARK)})")
        if "at" not in m:
            raise FX.FxError(f"{w}: `at` is required: [x, y], fractions of the photo")
        at = FX._pair(m["at"], f"{w} at")
        if not all(0.0 <= x <= 1.0 for x in at):
            raise FX.FxError(f"{w}: at = {at}: [x, y] in 0..1 of the photo")
        out.append({"shape": _shape(FX, m.get("shape"), w), "at": at,
                    "size": FX._num(m.get("size", MARK["size"]), f"{w} size", 0.01, 2.0),
                    "color": ink if m.get("color") is None else FX._colour(m["color"], palette, f"{w} color"),
                    "angle": FX._num(m.get("angle", MARK["angle"]), f"{w} angle")})
    return out


def normalize(FX, spec, out, palette, what):
    """A booth's `look` checked and, with look = "purikura", its `purikura` table filled in with DEFAULTS and checked,
    in place: {pastel, light (the palette's light neutral), frames (each FRAME_APART from the paper: `framing`), glow
    (mixed pastel), inks, frame, dots, stamps, marks, write, letters, speed, lift, soft, bloom}; the booth's paper
    (checked before) mixed pastel too. Without the look `purikura` is None."""
    if out["look"] not in LOOKS:
        raise FX.FxError(f"{what}: look {out['look']!r} is none of {', '.join(LOOKS)}")
    v = out.get("purikura")
    if out["look"] != "purikura":
        if v is not None:
            raise FX.FxError(f"{what}: `purikura` holds the purikura look's keys: give look = \"purikura\" with it")
        out["purikura"] = None
        return
    v = {} if v is None else v
    if not isinstance(v, dict):
        raise FX.FxError(f"{what}: purikura = {v!r}: expected a table of the look's keys ({', '.join(sorted(DEFAULTS))})")
    extra = sorted(set(v) - set(DEFAULTS))
    if extra:
        raise FX.FxError(f"{what} purikura: unknown keys {extra} (known: {sorted(DEFAULTS)})")
    p, w = {**DEFAULTS, **v}, f"{what} purikura"
    lt = light(FX, palette, f"{w} (base, text)")
    dk = dark(FX, palette, f"{w} (base, text)")
    t = FX._num(p["pastel"], f"{w} pastel", 0.0, 1.0)
    out["paper"] = mix(out["paper"], lt, t + (1.0 - t) * PAPER)
    pk = {"pastel": t, "light": lt,
          "frames": [framing(c, lt, t, out["paper"], dk) for c in _colour_list(FX, p["frames"], palette, f"{w} frames")],
          "glow": mix(FX._colour(p["glow"], palette, f"{w} glow"), lt, t),
          "inks": _colour_list(FX, p["inks"], palette, f"{w} inks"),
          "frame": FX._num(p["frame"], f"{w} frame", 0.0, 200.0),
          "dots": FX._num(p["dots"], f"{w} dots", 0.0, 200.0)}
    stamps = [p["stamps"]] if isinstance(p["stamps"], str) else p["stamps"]
    if not isinstance(stamps, (list, tuple)) or not stamps:
        raise FX.FxError(f"{w}: stamps = {p['stamps']!r}: expected a doodle shape or a list of them")
    pk["stamps"] = [_shape(FX, s, f"{w} stamps") for s in stamps]
    pk["marks"] = _marks(FX, p["marks"], palette, pk["inks"][0], w)
    pk["write"] = _texts(FX, p["write"], f"{w} write")
    pk["letters"] = FX._num(p["letters"], f"{w} letters", 0.03, 0.5)
    pk["speed"] = FX._num(p["speed"], f"{w} speed", lo=0.5)
    pk["lift"] = FX._num(p["lift"], f"{w} lift", 0.0, 1.0)
    pk["soft"] = FX._num(p["soft"], f"{w} soft", 0.0, 1.0)
    pk["bloom"] = FX._num(p["bloom"], f"{w} bloom", 0.0, 2.0)
    out["purikura"] = pk


# ================================================================================================= timing and layout
def flashing(fx, times):
    """The capture seconds of `times` whose flash goes off: all of them in the plain look; in the purikura look a capture
    while FLASH_RATE flashes began in the second before it is taken without one (eralook.thin)."""
    return list(times) if fx.get("purikura") is None else ER.thin(times, FLASH_RATE)


def lift_power(lift, key):
    """The power g of the beauty filter's lift, x -> 1 - (1 - x) ** g, for a capture whose key (mean luma, 0..1) is
    `key`: 1 + 2 `lift`, or less where that would lift the key past KEY (to KEY, a capture that light already not at
    all: a light scene keeps its picture instead of blowing out to white)."""
    g, key = 1.0 + 2.0 * lift, min(max(float(key), 0.0), 1.0)
    top = max(key, KEY)
    if key <= 0.0 or 1.0 - (1.0 - key) ** g <= top:
        return g
    return max(1.0, math.log(1.0 - top) / math.log(1.0 - key))


def _inside(pt, poly):
    """Whether point `pt` lies inside the polygon `poly` (n, 2), even-odd."""
    x, y = float(pt[0]), float(pt[1])
    p, q = np.asarray(poly, float), np.roll(np.asarray(poly, float), -1, axis=0)
    cross = (p[:, 1] > y) != (q[:, 1] > y)
    with np.errstate(divide="ignore", invalid="ignore"):
        xs = p[:, 0] + (y - p[:, 1]) * (q[:, 0] - p[:, 0]) / (q[:, 1] - p[:, 1])
    return bool(np.count_nonzero(cross & (x < xs)) % 2)


def _area(poly):
    p = np.asarray(poly, float)
    return 0.5 * abs(float(np.dot(p[:, 0], np.roll(p[:, 1], -1)) - np.dot(p[:, 1], np.roll(p[:, 0], -1))))


def closed(p):
    """Whether a stroke is a closed outline: it comes back through its first point (within CLOSE of its size) in its
    second half, as a pen drawing a shape round does."""
    p = np.asarray(p, float)
    if len(p) < 4:
        return False
    size = float(np.ptp(p, axis=0).max())
    back = p[len(p) // 2:] - p[0]
    return size > 0 and float(np.sqrt((back * back).sum(axis=1)).min()) <= CLOSE * size


def stamp_parts(strokes):
    """How a stamp draws each of a doodle's `strokes` (PC.shape's): "fill" (a closed outline, filled in its ink), "hole"
    (a closed one inside a larger fill: filled in a light tint of the ink), "line" (an open stroke: drawn in the ink) or
    "mark" (an open one across a fill: drawn in the light tint). A stroke lies inside a closed one when HELD of its
    points do."""
    shut = [closed(p) for p in strokes]
    area = [_area(p) if c else 0.0 for p, c in zip(strokes, shut)]
    out = []
    for i, p in enumerate(strokes):
        held = False
        for k, q in enumerate(strokes):
            if k != i and shut[k] and area[k] > area[i]:
                held = held or np.mean([_inside(pt, q) for pt in np.asarray(p, float)]) >= HELD
        out.append(("hole" if held else "fill") if shut[i] else ("mark" if held else "line"))
    return out


def writing(fx, cell, photo, t_capture):
    """What the pen writes over photo `cell` (its number) of a purikura booth, None for nothing: {text, at (x, y) px in
    the photo, the middle of its line; cap (px, its capitals' height); angle (degrees counter-clockwise); colour; seed
    (its letters' hand); t (the clip second the pen starts, a moment after the photo lands); dur (seconds it writes
    for, its ink at `speed` caps a second)}: text `cell` of `write`, along the photo's foot, `letters` photo heights tall
    or as large as fits WRITE_WIDTH of it, a little turned (seeded)."""
    pk = fx["purikura"]
    if cell >= len(pk["write"]) or not pk["write"][cell]:
        return None
    text = pk["write"][cell]
    pw, ph = photo
    r = _rng(fx["seed"], fx["index"], 32, cell)
    seed = int(r.integers(0, 2 ** 31 - 1))
    cap = min(pk["letters"] * ph, WRITE_WIDTH * pw / max(HD.width(text), 1e-6))
    ink = float(sum(HD.lengths(HD.strokes(text, seed))))
    at = (pw * (0.5 + float(r.uniform(-0.04, 0.04))), ph * (WRITE_AT + float(r.uniform(-0.02, 0.02))))
    return {"text": text, "at": at, "cap": cap, "angle": float(r.uniform(-7.0, 7.0)),
            "colour": pk["inks"][cell % len(pk["inks"])], "seed": seed, "t": t_capture + fx["drop"] + WRITE_LATE,
            "dur": max(ink / pk["speed"], 1e-3)}


def stamps(fx, cell, photo, t_capture):
    """The stamps of photo `cell` (its number) of a purikura booth: [{shape, at (x, y) px in the photo, size px, colour,
    angle, t (clip seconds: each pops in a moment after the one before, once the photo has landed)}]: the booth's
    `doodles` of them near its edges and corners (the middle is the face), one in each of as many arcs round it (not
    along its foot when the pen writes there), shapes from `stamps` and colours from `inks`, seeded; then its `marks`
    where they are set."""
    pk = fx["purikura"]
    r = _rng(fx["seed"], fx["index"], 31, cell)
    pw, ph = photo
    n = fx["doodles"]
    if cell < len(pk["write"]) and pk["write"][cell]:
        start, span = math.radians(140.0), math.radians(260.0)          # round from the lower left, over the top
    else:
        start, span = float(r.uniform(0.0, 2.0 * math.pi)), 2.0 * math.pi
    t0 = t_capture + fx["drop"]
    out, last = [], None
    for i in range(n):
        a = start + span * (i + 0.5 + float(r.uniform(-0.3, 0.3))) / max(n, 1)
        rad = float(r.uniform(0.33, 0.43))
        choices = [s for s in pk["stamps"] if s != last] or pk["stamps"]
        shape = choices[int(r.integers(0, len(choices)))]
        out.append({"shape": shape, "at": (pw * (0.5 + rad * math.cos(a)), ph * (0.5 + rad * math.sin(a))),
                    "size": ph * float(r.uniform(0.17, 0.25)), "colour": pk["inks"][int(r.integers(0, len(pk["inks"])))],
                    "angle": float(r.uniform(-20.0, 20.0)), "t": t0 + POP_GAP * (i + 1)})
        last = shape
    for i, m in enumerate(pk["marks"]):
        out.append({"shape": m["shape"], "at": (m["at"][0] * pw, m["at"][1] * ph), "size": m["size"] * ph,
                    "colour": m["color"], "angle": m["angle"], "t": t0 + POP_GAP * (n + i + 1)})
    return out
