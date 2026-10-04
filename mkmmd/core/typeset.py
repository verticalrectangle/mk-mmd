"""Kinetic type, the numbers (docs/design.md: Text): the bpy-free half of the build's `text` stage.

Surface frames, alignment words, number formats, the strings a keyed number will ever show, the typewriter's keys and
the character count they stand for, blink and flicker keys, and the layout (size, fit, alignment) of a measured string
block on a panel. Blender only measures the ink of strings; everything else is decided here so it can be tested."""
import math
import re
from dataclasses import dataclass

import numpy as np

from .palette import mix


class TextError(ValueError):
    """A [[text]] entry that cannot be laid out."""


# ------------------------------------------------------------------------------------------------------ frames
def _unit(v):
    v = np.asarray(v, float)
    n = float(np.linalg.norm(v))
    if n < 1e-12:
        raise TextError("zero-length direction")
    return v / n


def surface_matrix(center, normal, up=(0.0, 0.0, 1.0), lift=0.0):
    """4x4 frame of a surface (the card's use.surface convention): x = right = up x normal (what a viewer in front of
    the surface sees to the right), y = up, z = normal (out of the surface), origin at `center` + `lift` along the
    normal. `up` need not be perpendicular to `normal`; when it is parallel to it (a panel facing straight up with
    the default hint) +Y stands in."""
    n = _unit(normal)
    u = np.asarray(up, float)
    u = u - n * float(np.dot(u, n))
    if np.linalg.norm(u) < 1e-6:
        alt = np.array([0.0, 1.0, 0.0]) if abs(n[2]) > 0.9 else np.array([0.0, 0.0, 1.0])
        u = alt - n * float(np.dot(alt, n))
    u = _unit(u)
    m = np.eye(4)
    m[:3, 0], m[:3, 1], m[:3, 2] = np.cross(u, n), u, n
    m[:3, 3] = np.asarray(center, float) + n * float(lift)
    return m


# ------------------------------------------------------------------------------------------------------ alignment
def parse_align(align=None, valign=None):
    """(horizontal, vertical) from `align` and `valign`. `align` may carry both words ("left top", "center middle",
    ["right", "bottom"], either order); a single word sets one axis. "center" is the horizontal centre, or the vertical
    one when the horizontal word is already given. Defaults: center, middle."""
    words = []
    for src in (align, valign):
        if src is None:
            continue
        words += re.split(r"[\s,/+]+", src.strip().lower()) if isinstance(src, str) else \
            [str(w).strip().lower() for w in src]
    h = v = None
    centres = 0
    for w in (w for w in words if w):
        if w in ("left", "right"):
            if h not in (None, w):
                raise TextError(f"align {align!r}: two horizontal words")
            h = w
        elif w in ("top", "middle", "bottom"):
            if v not in (None, w):
                raise TextError(f"align {align!r}: two vertical words")
            v = w
        elif w in ("center", "centre"):
            centres += 1
        else:
            raise TextError(f"align word {w!r}: use left | center | right and top | middle | bottom")
    for _ in range(centres):
        if h is None:
            h = "center"
        elif v is None:
            v = "middle"
    return h or "center", v or "middle"


# ------------------------------------------------------------------------------------------------------ numbers
@dataclass(frozen=True)
class NumberFormat:
    """A restricted Python format string the node tree can reproduce: literal prefix, one number with `decimals`
    digits after the point (padded to `width` with `fill`), literal suffix."""
    prefix: str = ""
    suffix: str = ""
    decimals: int = 0
    width: int = 0
    fill: str = " "

    def format(self, v):
        s = f"{float(v):.{self.decimals}f}"
        if s.startswith("-") and float(s) == 0.0:
            s = s[1:]
        if len(s) < self.width:
            if self.fill == "0" and s.startswith("-"):
                s = "-" + s[1:].rjust(self.width - 1, "0")
            else:
                s = s.rjust(self.width, self.fill)
        return self.prefix + s + self.suffix


_FMT = re.compile(r"^(?P<pre>[^{}]*)\{:(?P<zero>0?)(?P<width>[1-9]\d*)?(?:\.(?P<prec>\d+))?(?P<type>[fd])?\}"
                  r"(?P<suf>[^{}]*)$")


def parse_format(fmt="{:.0f}"):
    """A format such as "{:.0f}", "{:.1f} MPH", "{:03d}", "{:4.0f}": text, one `{:[0][width][.precision][f|d]}` field,
    text. Anything else (other types, signs, alignment, several fields) cannot be rebuilt from nodes and raises."""
    m = _FMT.match(str(fmt))
    if not m or not (m.group("width") or m.group("prec") is not None or m.group("type")):
        raise TextError(f"format {fmt!r}: use literal text around one field like {{:.0f}}, {{:.1f}}, {{:03d}} or "
                        f"{{:4.0f}}")
    d = m.group("type") == "d"
    prec = m.group("prec")
    if d and prec:
        raise TextError(f"format {fmt!r}: d takes no precision")
    return NumberFormat(m.group("pre"), m.group("suf"), 0 if d or prec is None else int(prec),
                        int(m.group("width") or 0), "0" if m.group("zero") else " ")


def _spread(items, n):
    """n items spread evenly over a list (all of them when it is shorter), first and last included."""
    if len(items) <= n:
        return list(items)
    return [items[i] for i in np.linspace(0, len(items) - 1, n).round().astype(int)]


def candidate_strings(lo, hi, nf, limit=48):
    """Every distinct string a number sweeping from lo to hi can show with format `nf`, to measure the widest: all of
    them when there are at most `limit`, else the longest ones and an even spread over the sweep."""
    lo, hi = float(min(lo, hi)), float(max(lo, hi))
    step = 10.0 ** -nf.decimals
    k0, k1 = math.ceil(lo / step - 1e-9), math.floor(hi / step + 1e-9)
    if k1 < k0:
        vals = np.array([lo, hi])
    elif k1 - k0 > 4000:
        vals = np.linspace(lo, hi, 4001)
    else:
        vals = np.arange(k0, k1 + 1) * step
    out = list(dict.fromkeys(nf.format(v) for v in np.concatenate([[lo, hi], vals])))
    if len(out) > limit:
        longest = max(len(s) for s in out)
        keep = set(_spread([s for s in out if len(s) == longest], limit // 2)) | set(_spread(out, limit - limit // 2))
        out = [s for s in out if s in keep]
    return out


def ghost_string(s):
    """The unlit-segments string of a display: every letter and digit becomes 8 (all segments), the rest stays."""
    return "".join("8" if c.isalnum() else c for c in s)


def ghost_colour(lit, muted, share=0.5):
    """Hex of the unlit segments of a display: the lit colour pulled toward the palette's muted grey, so the segments
    that are off are darker and less saturated than the ones that are on."""
    return mix(lit, muted, share)


DISPLAY_GAMMA = 2.2


def display_share(share):
    """Emission, as a share of a lit emission, that shows at `share` of the lit brightness once the view transform has
    encoded it (about a 2.2 gamma: segments with 8 % of the emission are not 8 % as bright to the eye, they are about
    30 %). 1 -> 1, 0.08 -> 0.004."""
    share = float(share)
    if not 0.0 <= share <= 1.0:
        raise TextError("a brightness share must be in [0, 1]")
    return share ** DISPLAY_GAMMA


# ------------------------------------------------------------------------------------------------------ typewriter
def reveal_keys(t_from, t_to, fps):
    """[(t, r)] for the reveal ramp (linear keys). r < 0: nothing is typed yet; r in [0, 1]: character k (from 0) of n
    appears at r = k / (n - 1), so the first one comes at `t_from` and the last at `t_to`."""
    if not t_to > t_from:
        raise TextError("reveal: `to` must be later than `from`")
    return [(t_from - 1.0 / fps, -1.0), (t_from, 0.0), (t_to, 1.0)]


def reveal_count(r, n):
    """Characters shown at ramp value r of n: what the node tree computes (floor(r (n - 1) + eps) + 1 once r >= 0)."""
    if r < -1e-6 or n <= 0:
        return 0
    return int(min(n, math.floor(max(r, 0.0) * (n - 1) + 1e-5) + 1))


# ------------------------------------------------------------------------------------------------------ blink, flicker
def _window(spec, lo, hi):
    a, b = float(spec.get("from", lo)), float(spec.get("to", hi))
    if not b > a:
        raise TextError("`to` must be later than `from`")
    return a, b


def blink_keys(spec, lo, hi):
    """Piecewise-constant gain [(t, gain)] of a blinking light: on for `duty` of every `period` seconds (from `phase`),
    `low` otherwise, only between `from` and `to` (default lo..hi, the clip); on outside."""
    period, duty = float(spec.get("period", 1.0)), float(spec.get("duty", 0.5))
    low, phase = float(spec.get("low", 0.0)), float(spec.get("phase", 0.0))
    if period <= 0 or not 0.0 < duty <= 1.0:
        raise TextError("blink: period > 0 and 0 < duty <= 1")
    a, b = _window(spec, lo, hi)
    keys = [(a - 1e-4, 1.0)]
    t = a + phase
    while t < b:
        keys.append((max(t, a), 1.0))
        if duty < 1.0 and t + duty * period < b:
            keys.append((t + duty * period, low))
        t += period
    keys.append((b, 1.0))
    return _dedupe(keys)


def flicker_keys(spec, lo, hi):
    """Piecewise-constant gain [(t, gain)] of a failing tube: every 1/`rate` s a new level, mostly near 1 with random
    dips of depth up to `amount` (a share `dips` of the steps), only between `from` and `to`; 1 outside."""
    amount, rate = float(spec.get("amount", 0.5)), float(spec.get("rate", 12.0))
    dips, seed = float(spec.get("dips", 0.25)), int(spec.get("seed", 1))
    if not 0.0 <= amount <= 1.0 or rate <= 0 or not 0.0 <= dips <= 1.0:
        raise TextError("flicker: 0 <= amount <= 1, rate > 0, 0 <= dips <= 1")
    a, b = _window(spec, lo, hi)
    rng = np.random.default_rng(seed)
    n = max(1, int(math.ceil((b - a) * rate)))
    dip = rng.random(n) < dips
    depth = 0.3 + 0.7 * rng.random(n)
    jitter = rng.random(n)
    gain = np.where(dip, 1.0 - amount * depth, 1.0 - 0.06 * amount * jitter)
    keys = [(a - 1e-4, 1.0)] + [(a + i / rate, float(g)) for i, g in enumerate(gain)] + [(b, 1.0)]
    return _dedupe(keys)


def _dedupe(keys):
    out = []
    for t, g in keys:
        if out and abs(out[-1][0] - t) < 1e-9:
            out[-1] = (t, g)
        elif out and abs(out[-1][1] - g) < 1e-9:
            continue
        else:
            out.append((t, g))
    return out


def product_keys(a, b):
    """Product of two piecewise-constant gains given as [(t, g)] (the value before the first key is its first)."""
    ts = sorted({t for t, _ in a} | {t for t, _ in b})

    def at(keys, t):
        v = keys[0][1]
        for k, g in keys:
            if k <= t + 1e-12:
                v = g
        return v

    return _dedupe([(t, at(a, t) * at(b, t)) for t in ts])


# ------------------------------------------------------------------------------------------------------ layout
@dataclass(frozen=True)
class Layout:
    em: float           # String to Curves size (the font's em), metres
    tx: float           # translation of the string block in the text frame, metres
    ty: float
    cap: float          # resulting cap height, metres
    ink: tuple          # (width, height) of the ink block of the reference string(s), metres
    usable: tuple       # (width, height) of the region text is aligned in (panel x fit), metres
    fits: bool          # the ink block lies inside the panel (when the panel is known)


def plan_layout(ref, cap_ratio, panel=None, size=None, fit=None, align="center", valign="middle", offset=(0.0, 0.0),
                default_fit=0.9):
    """Size and place a string block on a panel.

    ref        (x0, y0, x1, y1): ink box of the widest / tallest string the text ever shows, at em size 1 and with the
               String to Curves alignment `align`/`valign` already applied (so the box carries its own origin)
    cap_ratio  cap height of the font per em
    panel      (width, height) of the surface in metres, or None for free text anchored at the origin
    size       cap height in metres; fit: share of the panel the ink may fill (also the margin of the region text is
               aligned in). With both, `size` is the largest cap height `fit` allows; with neither, fit = default_fit
    The text frame has its origin at the panel's centre, x right, y up."""
    x0, y0, x1, y1 = (float(v) for v in ref)
    rw, rh = x1 - x0, y1 - y0
    if not (rw > 0 or rh > 0):
        raise TextError("nothing to draw (the text has no ink)")
    if cap_ratio <= 0:
        raise TextError("the font has no capital height")
    pw, ph = (float(panel[0]), float(panel[1])) if panel else (0.0, 0.0)
    f = float(fit) if fit is not None else default_fit
    if not 0.0 < f <= 1.0:
        raise TextError("fit must be in (0, 1]")
    aw, ah = pw * f, ph * f
    em_fit = math.inf
    if aw > 0 and rw > 0:
        em_fit = min(em_fit, aw / rw)
    if ah > 0 and rh > 0:
        em_fit = min(em_fit, ah / rh)
    if size is not None:
        if not float(size) > 0:
            raise TextError("size must be positive")
        em = float(size) / cap_ratio
        if fit is not None:
            em = min(em, em_fit)
    else:
        if math.isinf(em_fit):
            raise TextError("give `size` (cap height, m) or a panel to fit the text in (`on`, or `box`)")
        em = em_fit
    u, v = float(offset[0]), float(offset[1])
    ax = {"left": x0, "center": (x0 + x1) / 2, "right": x1}[align] * em
    ay = {"bottom": y0, "middle": (y0 + y1) / 2, "top": y1}[valign] * em
    tx_t = {"left": -aw / 2, "center": 0.0, "right": aw / 2}[align]
    ty_t = {"bottom": -ah / 2, "middle": 0.0, "top": ah / 2}[valign]
    ink = (rw * em, rh * em)
    fits = (pw <= 0 or ink[0] <= pw * (1 + 1e-9)) and (ph <= 0 or ink[1] <= ph * (1 + 1e-9))
    return Layout(em, tx_t - ax + u, ty_t - ay + v, cap_ratio * em, ink, (aw, ah), fits)
