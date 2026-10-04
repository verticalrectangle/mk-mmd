"""Cassette type, the numbers (docs/design.md: Text, Lyrics): the bpy-free helpers of the tape look of lyric words, used by
`mkmmd/core/wordtype.py`.

  slide    a strip fed into the deck: it comes in from one side, decelerating, tilted, and clunks home with a short squash of
           its length (`style = "slide"`)
  wow      tape wow and flutter while a note is held: a slow drift of the strip's speed (position and length) and a fast
           flutter, fading in and out with the note (`wow = {...}`)
  rewind   the line leaves like a tape being rewound: the words shoot back the way they came, smeared long, and fade
           (`leave = "rewind"`)
  rows     the words of a line broken into a given number of rows of nearly equal width (`layout = {kind = "flow", rows}`)

Everything is a function of time (seconds after the landing, or the note's own clock) returning channel values for the
`kinetic` keys, one frame at a time; nothing here knows about Blender or about the words' text."""
import math
from itertools import combinations

SLIDE_DEFAULTS = {"from": "alt", "dist": 0.5, "dur": 0.16, "tilt": 4.0, "clunk": 0.07, "alpha0": 1.0, "alpha_frames": 2.0}
SLIDE_SIDES = {"left": -1.0, "right": 1.0}
WOW_DEFAULTS = {"amount": 0.12, "rate": 1.4, "flutter": 0.03, "flutter_rate": 9.0, "pitch": 0.05, "tilt": 2.5,
                "min_hold": 0.30, "attack": 0.12, "tail": 0.15}
REWIND_DEFAULTS = {"life": 0.2, "dist": 0.6, "stretch": 3.0, "gap": 0.02, "order": "first_first", "min_run": 0.4}
REWIND_ORDERS = ("first_first", "last_first", "none")
CLUNK_TAU = 0.05            # s: time constant of the squash at the end of a slide


def _sstep(a, b, x):
    u = min(1.0, max(0.0, (x - a) / (b - a)))
    return u * u * (3.0 - 2.0 * u)


# ------------------------------------------------------------------------------------------------------ slide
def slide_side(mode, k):
    """-1 (in from the left) or +1 (from the right) for word `k` of a selection: `left`, `right` or `alt` (alternating, the
    first from the right)."""
    if mode == "alt":
        return 1.0 if k % 2 == 0 else -1.0
    if mode not in SLIDE_SIDES:
        raise ValueError(f"slide from is 'left', 'right' or 'alt', not {mode!r}")
    return SLIDE_SIDES[mode]


def slide_state(tau, p):
    """Channels `tau` s after a strip lands (`p`: the style's numbers plus `side`): in from `side` by `dist` shares of the panel
    over `dur` s with an ease-out cubic, tilted by `tilt` degrees that straighten as it arrives, and once home a damped squash
    of its length (`clunk`, time constant 50 ms) with the matching bulge in height. Returns only the channels it moves."""
    side = p["side"]
    u = min(1.0, max(tau / p["dur"], 0.0)) if p["dur"] > 0 else 1.0
    e = 1.0 - (1.0 - u) ** 3
    out = {"dxp": side * p["dist"] * (1.0 - e), "rot": side * p["tilt"] * (1.0 - e)}
    v = tau - p["dur"]
    if v > 0.0:
        k = p["clunk"] * math.exp(-v / CLUNK_TAU)
        out["sx"] = 1.0 - k
        out["sy"] = 1.0 + 0.6 * k
    return out


# ------------------------------------------------------------------------------------------------------ wow and flutter
def wow_held(t0, t1, p):
    """Whether a note held from `t0` to `t1` is long enough to wow (`min_hold` s)."""
    return t1 - t0 >= p["min_hold"]


def wow_at(t, t0, t1, p, phase=0.0):
    """Additive channel values at time `t` for a note held from `t0` to `t1`: dx (em), dy (em), rot (degrees) and sx (a factor
    to add to 1). The slow drift (`rate` Hz, `amount` em, `pitch` of the length, `tilt` degrees) and the flutter (`flutter_rate`
    Hz, `flutter` em) fade in over `attack` s after `t0` and out over `tail` s after `t1`; `phase` (radians) keeps words apart."""
    env = _sstep(0.0, p["attack"], t - t0) * (1.0 - _sstep(0.0, p["tail"], t - t1))
    if env <= 0.0:
        return {"dx": 0.0, "dy": 0.0, "rot": 0.0, "sx": 0.0}
    u = t - t0
    slow = math.sin(2.0 * math.pi * p["rate"] * u + phase)
    fr = p["flutter_rate"]
    fast = math.sin(2.0 * math.pi * fr * u + 2.3 * phase) + 0.5 * math.sin(2.0 * math.pi * 1.7 * fr * u + phase)
    return {"dx": env * (p["amount"] * slow + p["flutter"] * fast),
            "dy": env * 0.8 * p["flutter"] * math.sin(2.0 * math.pi * 1.3 * fr * u + 4.0 * phase),
            "rot": env * p["tilt"] * slow,
            "sx": env * p["pitch"] * slow}


# ------------------------------------------------------------------------------------------------------ rewind
def rewind_ranks(order, n):
    """How many gaps after the first a word's rewind starts: `first_first` (0, 1, 2, ...), `last_first` or `none` (all at once)."""
    if order == "first_first":
        return list(range(n))
    if order == "last_first":
        return [n - 1 - k for k in range(n)]
    if order == "none":
        return [0] * n
    raise ValueError(f"rewind order is one of {list(REWIND_ORDERS)}")


def rewind_state(u, p):
    """Channels of a word `u` (0..1) of the way through its rewind: it shoots back by `dist` shares of the panel with an
    ease-in, stretches to 1 + `stretch` times its length (a smear) and fades out over the second half. Returns dxp, sx, alpha."""
    u = min(1.0, max(u, 0.0))
    return {"dxp": -p["dist"] * u ** 2.2, "sx": 1.0 + p["stretch"] * u * u, "alpha": 1.0 - _sstep(0.5, 1.0, u)}


# ------------------------------------------------------------------------------------------------------ rows
def wrap_rows(widths, n_rows, gap=0.0):
    """Row index (0-based) of every word when `widths` are set in order in `n_rows` rows, with `gap` between words: the
    breaks that make the widest row as narrow as possible, then the rows as even as possible. Rows are never empty (at most
    one per word)."""
    n = len(widths)
    n_rows = max(1, min(int(n_rows), n))
    best, best_key = None, None
    for cuts in combinations(range(1, n), n_rows - 1):
        edges = (0,) + cuts + (n,)
        rows = [sum(widths[a:b]) + gap * (b - a - 1) for a, b in zip(edges, edges[1:])]
        key = (max(rows), sum(r * r for r in rows))
        if best_key is None or key < best_key:
            best, best_key = edges, key
    return [r for r, (a, b) in enumerate(zip(best, best[1:])) for _ in range(b - a)]
