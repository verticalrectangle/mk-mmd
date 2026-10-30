"""Ring type, the numbers (docs/design.md: Text, Lyrics, Rings): the bpy-free half of `layout = {kind = "ring", ...}`.

The words of a lyrics entry sit round circles about a point of the panel, reading round them: word k+1 follows word k
clockwise with the tops of the letters pointing away from the centre (`face = "out"`, the way type runs round the top of
a badge) or counter-clockwise with the tops pointing at it (`face = "in"`, the way type runs along a badge's bottom). The
letters follow the curve (`bend`; `bend = false` sets each word straight on its tangent). Each word takes an arc as long
as its ink box (with the spread headroom, the strip behind it and its outline) plus `gap`, so long words take more of the
ring than short ones. Every word stays up from its landing to the cut, so a phrase sung several times ACCUMULATES:

  rings = "line"   each lyric line is a ring of its own, one `pitch` further out than the one before: every repeat of the
                   phrase adds a ring round the last. A line that does not go all the way round is centred on `start`
                   (`even = true`: its words share the whole ring out between them)
  rings = "fill"   the words wind round a ring, a repeat taking up where the last one stopped, until the next word would
                   not fit; that ring is full (its free arc is shared out between its words, so it closes evenly) and the
                   words go on round the next ring out

Each ring turns about the centre at `turn` degrees a second (clockwise for a positive number), the next ring the other
way (a list gives every ring its own, signed); a ring turns rigidly from the moment its first word lands, so every word
lands where its place on the turning ring has got to. The common size comes from the panel as for the other layouts:
the largest em at which the rings fit `fit` of the room round the centre (and at most `size`), no line wraps into itself,
and the innermost ring keeps half a word's height clear of the centre.

Units: the layout works in em (the String to Curves size) and returns metres in the text frame (x right, y up from the
panel's centre). Angles in the public results: `angle` is degrees clockwise from the top (where the word sits on its ring
when the ring has not turned), `orient` is the word's own rotation, degrees counter-clockwise (the `rot` of a kinetic
text). Nothing here sees the words' text: a word is its ink box."""
import math

RING_KEYS = {"center", "radius", "pitch", "gap", "turn", "start", "rings", "face", "even", "bend"}
MODES = ("line", "fill")
FACES = ("out", "in")
DEFAULTS = {"center": [0.0, 0.0], "radius": 0.35, "pitch": None, "gap": 0.3, "turn": 8.0, "start": None,
            "rings": "line", "face": "out", "even": False, "bend": True}
CLEAR = 0.5                 # word heights the innermost ring's words keep clear of the centre
TWO_PI = 2.0 * math.pi


class RingError(ValueError):
    """A ring layout that cannot be made; messages carry numbers only."""


def _num(v, what):
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(float(v)):
        raise RingError(f"ring {what} must be a number")
    return float(v)


def is_target(c):
    """True when a ring `center` names a point of the scene (a target: a string, a table, or [x, y, z] in the world) rather
    than a point of the panel ([u, v])."""
    return isinstance(c, (str, dict)) or (isinstance(c, (list, tuple)) and len(c) == 3)


def options(lay):
    """The checked ring layout (`layout` table without `kind`) with its defaults. `center` stays as given when it is a
    target (see `is_target`)."""
    unknown = sorted(set(lay) - RING_KEYS)
    if unknown:
        raise RingError(f"layout 'ring' keys {unknown} are unknown (known: {sorted(RING_KEYS)})")
    o = {**DEFAULTS, **lay}
    if o["rings"] not in MODES:
        raise RingError("ring rings is 'line' (a ring per lyric line) or 'fill' (each ring filled before the next)")
    if o["face"] not in FACES:
        raise RingError("ring face is 'out' (the tops away from the centre, reading clockwise) or 'in' (the tops toward "
                        "it, reading counter-clockwise)")
    for key in ("even", "bend"):
        if not isinstance(o[key], bool):
            raise RingError(f"ring {key} is true or false")
    c = o["center"]
    if not is_target(c):
        if not (isinstance(c, (list, tuple)) and len(c) == 2):
            raise RingError("ring center is [u, v] (shares of the panel from its centre) or a target (a point of the scene)")
        o["center"] = [_num(c[0], "center"), _num(c[1], "center")]
    o["radius"] = _num(o["radius"], "radius")
    if not o["radius"] > 0:
        raise RingError("ring radius is a positive share of half the panel's shorter side")
    if o["pitch"] is not None:
        o["pitch"] = _num(o["pitch"], "pitch")
        if not o["pitch"] > 0:
            raise RingError("ring pitch is em between rings, positive")
    o["gap"] = _num(o["gap"], "gap")
    if o["gap"] < 0:
        raise RingError("ring gap is em between words, not negative")
    turn = o["turn"]
    if isinstance(turn, (list, tuple)):
        if not turn:
            raise RingError("ring turn is degrees a second, or a list of them (one per ring, cycled)")
        o["turn"] = [_num(t, "turn") for t in turn]
    else:
        o["turn"] = _num(turn, "turn")
    o["start"] = (180.0 if o["face"] == "in" else 0.0) if o["start"] is None else _num(o["start"], "start")
    return o


def ring_turn(turn, i):
    """Degrees a second ring `i` turns, clockwise positive: a number alternates its sign ring by ring, a list is cycled."""
    if isinstance(turn, (list, tuple)):
        return float(turn[i % len(turn)])
    return float(turn) * (1.0 if i % 2 == 0 else -1.0)


def side(face):
    """Which side of a bent word its ring's centre is on: +1 below it (`out`), -1 above it (`in`)."""
    return 1 if face == "out" else -1


# ------------------------------------------------------------------------------------------------------ extents
def _extent(box, pad, rho, bend, gap):
    """(angular share, inner reach, outer reach) of a word on a ring of radius `rho` (em): its ink box `box` = (w, y0, y1)
    em grown by `pad` em (its outline) all round, the ring line through the box's middle, plus `gap` em of clear ring.
    The share is measured at the word's inner edge, where neighbours come closest."""
    w, y0, y1 = box
    hw, hh = 0.5 * w + pad, 0.5 * (y1 - y0) + pad
    inner = rho - hh                                # radius of the word's inner edge
    if inner <= 1e-9:
        return math.inf, hh, hh
    if bend:                                        # the letters follow the ring: the ink is w long on the ring line
        return w / rho + (2.0 * pad + gap) / inner, hh, hh
    # a straight word on its tangent: the inner corners spread widest, the outer corners reach out past the ring line
    return 2.0 * math.atan(hw / inner) + gap / inner, hh, math.hypot(rho + hh, hw) - rho


# ------------------------------------------------------------------------------------------------------ arrangement
def _arrange(boxes, lines, pads, rho0, o, gap, pitch):
    """Rings in em, the first at radius `rho0` em: (ring of each word, [ring radii], [(share, inner, outer) of each word]),
    or None when a line of `rings = "line"` would wrap into itself or a word cannot sit on its ring at all."""
    n = len(boxes)
    ring_of, shares = [0] * n, [None] * n
    if o["rings"] == "line":
        order = []
        for ln in lines:
            if ln not in order:
                order.append(ln)
        used = [0.0] * len(order)
        for k in range(n):
            i = order.index(lines[k])
            ring_of[k], shares[k] = i, _extent(boxes[k], pads[k], rho0 + i * pitch, o["bend"], gap)
            used[i] += shares[k][0]
        if max(used) > TWO_PI * (1.0 + 1e-9):
            return None
        n_rings = len(order)
    else:
        i, used = 0, 0.0
        for k in range(n):
            s = _extent(boxes[k], pads[k], rho0 + i * pitch, o["bend"], gap)
            if used > 0.0 and used + s[0] > TWO_PI * (1.0 + 1e-9):     # this ring is full: on round the next one out
                i, used = i + 1, 0.0
                s = _extent(boxes[k], pads[k], rho0 + i * pitch, o["bend"], gap)
            if s[0] > TWO_PI * (1.0 + 1e-9):
                return None
            ring_of[k], shares[k] = i, s
            used += s[0]
        n_rings = i + 1
    return ring_of, [rho0 + i * pitch for i in range(n_rings)], shares


def _auto_pitch(boxes, pads, rho0, o, gap):
    """Em between ring lines when `pitch` is not given: the furthest reach outward plus the furthest reach inward plus
    `gap`, so the words of neighbouring rings never touch however the rings turn (straight words reach furthest on the
    innermost ring)."""
    reach = [_extent(b, p, rho0, o["bend"], gap) for b, p in zip(boxes, pads)]
    return max(r[2] for r in reach) + max(r[1] for r in reach) + gap


def layout_ring(boxes, lines, panel, fit, size, cap, opts, pads=None, gap_scale=1.0):
    """Where every word goes round its ring.

    boxes  per word (w, y0, y1): its ink box in em with the baseline at 0 (widths include the spread headroom and the strip
           behind it), in reading order
    lines  per word, its lyric line (`rings = "line"`: a ring each, in the order they come)
    panel  (w, h) metres of the panel; fit its share the rings may fill (default 0.9); size the largest cap height (m)
    cap    cap height per em of the font; opts `options(...)`; pads per word, em its outline grows it by all round;
           gap_scale the entry's `word_spacing`
    Returns {"em", "center": (x, y) m, "rings": [{"radius" m, "turn" deg/s, "words": [k, ...], "full"}], "words":
    [{"ring", "angle" deg clockwise from the top, "offset" (x, y) m from the panel's centre, "orient" deg ccw}]}."""
    if not panel or not (float(panel[0]) > 0 and float(panel[1]) > 0):
        raise RingError("layout 'ring' needs a panel (`on`, `box` or `screen`): the rings are sized from it")
    n = len(boxes)
    if n == 0:
        raise RingError("layout 'ring' has no words to set")
    pads = [0.0] * n if pads is None else [float(p or 0.0) for p in pads]
    pw, ph = float(panel[0]), float(panel[1])
    f = 0.9 if fit is None else float(fit)
    if not 0.0 < f <= 1.0:
        raise RingError("fit must be in (0, 1]")
    target = is_target(opts["center"])
    cx, cy = (0.0, 0.0) if target else (opts["center"][0] * pw, opts["center"][1] * ph)
    room = f * min(0.5 * pw - abs(cx), 0.5 * ph - abs(cy))
    r0 = opts["radius"] * 0.5 * min(pw, ph)                     # metres: the first ring
    if room <= r0:
        raise RingError(f"ring radius {opts['radius']:g} leaves no room for the words inside {f:g} of the panel round "
                        f"its centre (the first ring is {r0:.4g}, the room {max(room, 0.0):.4g})")
    gap = opts["gap"] * float(gap_scale)
    hh = max(0.5 * (b[2] - b[1]) + p for b, p in zip(boxes, pads))      # the tallest half word, em
    em_hi = r0 / (hh + CLEAR * 2.0 * hh)                        # the innermost words clear of the centre
    if size is not None:
        if not float(size) > 0:
            raise RingError("size must be positive")
        em_hi = min(em_hi, float(size) / cap)

    def attempt(em):
        rho0 = r0 / em
        pitch = opts["pitch"] if opts["pitch"] is not None else _auto_pitch(boxes, pads, rho0, opts, gap)
        got = _arrange(boxes, lines, pads, rho0, opts, gap, pitch)
        if got is None:
            return None
        ring_of, radii, shares = got
        outer = max(radii[ring_of[k]] + shares[k][2] for k in range(n)) * em
        return got if outer <= room * (1.0 + 1e-9) else None

    em, best = em_hi, attempt(em_hi)
    if best is None:                                # the largest em that still fits: fitting only gets harder as em grows
        lo, hi = 0.0, em_hi
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            got = attempt(mid)
            if got is not None:
                lo, best = mid, got
            else:
                hi = mid
        if best is None:
            raise RingError("the words cannot be set round the rings at any size")
        em = lo
    ring_of, radii, shares = best
    return _place(ring_of, radii, shares, em, (cx, cy), opts)


def _place(ring_of, radii, shares, em, center, o):
    """Angles, offsets and orientations of the words once the rings are known (see layout_ring)."""
    n = len(ring_of)
    d = -1.0 if o["face"] == "in" else 1.0          # reading direction: clockwise, or counter-clockwise for `in`
    start = math.radians(o["start"])
    rings = [{"radius": rho * em, "turn": ring_turn(o["turn"], i), "words": [k for k in range(n) if ring_of[k] == i],
              "full": o["rings"] == "fill" and i < len(radii) - 1} for i, rho in enumerate(radii)]
    words = [None] * n
    for i, r in enumerate(rings):
        ks = r["words"]
        if not ks:
            continue
        a = [shares[k][0] for k in ks]
        slack = (TWO_PI - sum(a)) / len(ks) if (r["full"] or o["even"]) else 0.0
        a = [x + slack for x in a]
        if o["rings"] == "line" and not o["even"]:  # a line that does not go round: centred on `start`
            phi = start - d * (0.5 * sum(a) - 0.5 * a[0])
        else:                                       # the ring reads round from `start`, its first word centred there
            phi = start
        for j, k in enumerate(ks):
            if j:
                phi += d * 0.5 * (a[j - 1] + a[j])
            deg = math.degrees(phi)
            orient = -deg if o["face"] == "out" else 180.0 - deg
            words[k] = {"ring": i, "angle": deg % 360.0, "orient": (orient + 180.0) % 360.0 - 180.0,
                        "offset": (center[0] + r["radius"] * math.sin(phi), center[1] + r["radius"] * math.cos(phi))}
    return {"em": em, "center": tuple(center), "rings": rings, "words": words}


# ------------------------------------------------------------------------------------------------------ the turn
def orbit_keys(turn, t_ref, t_on, t_off):
    """Linear keys of a ring word's turn about the centre, degrees counter-clockwise (the kinetic `orbit`): the ring turns
    `turn` degrees a second clockwise from `t_ref` (when its first word lands), so the key at t is -turn (t - t_ref)."""
    return [[t_on, -turn * (t_on - t_ref)], [t_off, -turn * (t_off - t_ref)]]


def position(word, ring, t, t_ref, center=(0.0, 0.0)):
    """(x, y) metres of a word's centre at clip time t: its place on the ring turned on by `ring["turn"]` since t_ref (what
    the text stage's node tree does with the `orbit` keys)."""
    phi = math.radians(word["angle"] + ring["turn"] * (t - t_ref))
    return center[0] + ring["radius"] * math.sin(phi), center[1] + ring["radius"] * math.cos(phi)
