"""Lyric type, the numbers (docs/design.md: Text, `lyrics`): the bpy-free half of `mkmmd/blender/build/wordtype.py`.

A `[[text]]` entry with `lyrics = {...}` becomes one ordinary text per sung word. This module turns a timeline's words
into those texts: which words, when each lands, how it arrives, spreads while its note is held, leaves, and where it
goes. Everything keyed over time is returned as `[[t, value], ...]` lists (clip seconds, one key per frame) for the text
stage's `kinetic` key; Blender only measures glyph boxes (the `measure` callback) and builds nodes from the keys.

The words' text lives in exactly one field, `Word.text`, which is never printed: reprs, error messages and the summary
only carry `(line, word)` numbers (1-based) and counts. Timeline problems are reported without the offending value.

Timing (measured from the frames of a delivered cut): a word LANDS on clip frame
L = floor(onset * fps + 0.4), never earlier than the first frame of its shot (`from`), and is on screen from L (its
arrival animation plays after L). While its note is held (`start` .. `voiced_end`) the letters spread (tracking grows
with an ease-out cubic, then relaxes with a 0.13 s time constant) and a louder `vocal_db` makes the weight bolder; the
word's own level sets the base, the voice breathes it while the note is held. When the line ends before the cut and a
tempo tick falls between its last word and `to - 0.4 s`, the letters drip away on that tick (stretch downward from
their tops and fall like drops, the first one leaving on the tick's frame); otherwise the line holds to the cut, i.e.
the text is hidden from the first frame of the next shot, `round(frame0 + to * fps)` as the shots stage cuts."""
import math
from dataclasses import dataclass, field

import numpy as np

LYRICS_KEYS = {"timeline", "line", "lines", "words", "case", "clean", "style", "arrive", "from", "to", "leave", "drip",
               "spread", "weight", "colors", "color_by", "sizes", "offsets", "layout", "tilt", "recycle", "recolor",
               "backing"}
CASES = ("keep", "lower", "upper", "title")
STYLES = ("pop", "drop", "rise", "slap", "type", "none")
LAYOUTS = ("same", "flow", "stack", "slots")

DB_FLOOR, DB_RANGE, DB_GAMMA = -34.0, 29.0, 0.9     # vocal_db (dB re the vocal peak) mapped to a 0..1 weight level
SMOOTH_SIGMA, SMOOTH_HALF = 1.6, 5                  # frames: the Gaussian that smooths vocal_db for the breathing
BREATH = 0.65                                       # share of the voice's level the weight follows while the note is held
HOLD_TAIL = 0.10                                    # s: the breathing lingers this long after the note ends
MIN_RUN = 0.40                                      # s: a drip needs at least this much of the shot left
DRIP_LEAD = 0.03                                    # s: a tick this close before the last word's end still counts
DRIP_SPREAD = 0.10                                  # s: the letters of a word leave within this window
DRIP_TAU = 0.17                                     # s: time constant of the stretch
DRIP_FALL = 150.0                                   # the gravity constants are in px/s^2 at a 150 px em (the original)


class LyricsError(ValueError):
    """A lyrics entry that cannot be expanded. Messages carry (line, word) numbers and counts, never lyric text."""


@dataclass(frozen=True)
class Word:
    """A selected word. `text` is the only field holding lyric text and is kept out of every repr."""
    line: int                      # 1-based line of the timeline
    index: int                     # 1-based word within the line
    k: int                         # 0-based position in the selection
    start: float                   # s
    end: float
    voiced_end: float
    db: float                      # mean vocal_db over the note (mid-range when the timeline has no vocal track)
    text: str = field(repr=False, compare=False, default="")

    def __repr__(self):
        return f"Word({self.line},{self.index})"

    __str__ = __repr__

    @property
    def tag(self):
        return f"l{self.line}w{self.index}"


# ------------------------------------------------------------------------------------------------------ small maths
def sstep(a, b, x):
    u = min(1.0, max(0.0, (x - a) / (b - a)))
    return u * u * (3.0 - 2.0 * u)


def eo2(x):
    x = min(1.0, max(0.0, x))
    return 1.0 - (1.0 - x) ** 2


def eo3(x):
    x = min(1.0, max(0.0, x))
    return 1.0 - (1.0 - x) ** 3


def h01(*ints):
    """Deterministic hash of integers -> [0, 1)."""
    m = 0xFFFFFFFFFFFFFFFF
    x = 0x9E3779B97F4A7C15
    for v in ints:
        x ^= (int(v) + 0x7F4A7C15 + ((x << 6) & m) + (x >> 2)) & m
        x = (x * 0xBF58476D1CE4E5B9) & m
        x ^= x >> 27
    return (x & 0xFFFFFF) / float(1 << 24)


def cut_frame(t, fps, frame0=0):
    """Clip frame of clip time t, rounded the way the shots stage rounds its cuts: round(frame0 + t * fps) - frame0."""
    return int(round(frame0 + t * fps)) - int(frame0)


def landing_frame(onset, fps, first=None):
    """Clip frame a word lands on: floor(onset * fps + 0.4), not earlier than `first` (the shot's first frame)."""
    f = int(math.floor(onset * fps + 0.4))
    return f if first is None else max(f, int(first))


# ------------------------------------------------------------------------------------------------------ timeline
def ticks_of(tl):
    """Tempo ticks (s) of a timeline: `ticks`, `tempo.ticks`, else the beats."""
    t = tl.get("ticks") or (tl.get("tempo") or {}).get("ticks") or tl.get("beats") or (tl.get("tempo") or {}).get("beats")
    return sorted(float(x) for x in (t or []))


def smooth_vocal(vdb):
    """Per-frame vocal level smoothed with the breathing kernel (edge padded)."""
    v = np.asarray(vdb, float)
    if v.size == 0:
        return v
    k = np.exp(-0.5 * (np.arange(-SMOOTH_HALF, SMOOTH_HALF + 1) / SMOOTH_SIGMA) ** 2)
    k /= k.sum()
    return np.convolve(np.pad(v, (SMOOTH_HALF, SMOOTH_HALF), mode="edge"), k, mode="valid")


def vocal_at(vs, t, fps):
    """Linear interpolation of the smoothed vocal level at clip time t."""
    if len(vs) == 0:
        return DB_FLOOR + 0.5 * DB_RANGE
    x = min(max(t * fps, 0.0), len(vs) - 1.001)
    i = int(x)
    return float(vs[i] + (vs[min(i + 1, len(vs) - 1)] - vs[i]) * (x - i))


def level(db):
    """0..1 weight level of a vocal level in dB."""
    return min(1.0, max(0.0, (db - DB_FLOOR) / DB_RANGE)) ** DB_GAMMA


def word_db(vdb, start, voiced_end, fps):
    """Mean vocal_db over the note (frames int(start * fps) .. round(voiced_end * fps), at least one)."""
    v = np.asarray(vdb, float)
    fa = int(start * fps)
    fb = max(fa + 1, int(round(voiced_end * fps)))
    seg = v[fa:fb]
    return float(seg.mean()) if seg.size else DB_FLOOR + 0.5 * DB_RANGE


def _number(x, what):
    if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(float(x)):
        raise LyricsError(f"timeline: {what} is not a number")
    return float(x)


def _int(x, what):
    if isinstance(x, bool) or not isinstance(x, int):
        raise LyricsError(f"lyrics: {what} must be a whole number")
    return int(x)


def spell(text, case="keep", clean=False):
    """A word as it is set: `clean` keeps letters, digits and apostrophes (a curly one turns straight; a word with
    nothing else left stays as it was), `case` is keep | lower | upper | title."""
    t = text.strip()
    if clean:
        c = "".join(ch for ch in t.replace("\u2019", "'") if ch.isalnum() or ch == "'")
        t = c or t
    if case == "lower":
        return t.lower()
    if case == "upper":
        return t.upper()
    if case == "title":
        return t[:1].upper() + t[1:].lower()
    return t


def select(tl, line=None, lines=None, words=None, fps=30.0, case="keep", clean=False):
    """The selected words in song order. `line = N` or `lines = [a, b]` (1-based, inclusive); `words = [i, j]` (1-based,
    inclusive within every selected line; one number selects one word); `case` and `clean` as in `spell`. Errors name
    line and word numbers only."""
    if case not in CASES:
        raise LyricsError(f"lyrics: case is one of {list(CASES)}")
    rows = tl.get("lines") if isinstance(tl, dict) else None
    if not isinstance(rows, list) or not rows:
        raise LyricsError("the timeline has no lines")
    if (line is None) == (lines is None):
        raise LyricsError("lyrics: give exactly one of `line` and `lines`")
    if line is not None:
        a = b = _int(line, "line")
    else:
        if not isinstance(lines, (list, tuple)) or len(lines) != 2:
            raise LyricsError("lyrics: `lines` is [first, last] (1-based, inclusive)")
        a, b = _int(lines[0], "lines"), _int(lines[1], "lines")
    if not 1 <= a <= b <= len(rows):
        raise LyricsError(f"lyrics: lines {a}..{b} are out of range (the timeline has {len(rows)} lines)")
    if words is not None:
        if isinstance(words, int) and not isinstance(words, bool):
            words = [words, words]
        if not isinstance(words, (list, tuple)) or len(words) != 2:
            raise LyricsError("lyrics: `words` is [first, last] (1-based, inclusive within each line)")
        wi0, wi1 = _int(words[0], "words"), _int(words[1], "words")
    vdb = tl.get("vocal_db")
    out = []
    for ln in range(a, b + 1):
        row = rows[ln - 1]
        ws = row.get("words") if isinstance(row, dict) else None
        if not isinstance(ws, list) or not ws:
            raise LyricsError(f"timeline: line {ln} has no words")
        i0, i1 = (1, len(ws)) if words is None else (wi0, wi1)
        if not 1 <= i0 <= i1 <= len(ws):
            raise LyricsError(f"lyrics: words {i0}..{i1} are out of range (line {ln} has {len(ws)} words)")
        for wi in range(i0, i1 + 1):
            w = ws[wi - 1]
            if not isinstance(w, dict) or not isinstance(w.get("text"), str) or not w["text"].strip():
                raise LyricsError(f"timeline: word ({ln}, {wi}) has no text")
            t0 = _number(w.get("start"), f"start of word ({ln}, {wi})")
            te = _number(w.get("end", t0), f"end of word ({ln}, {wi})")
            t1 = _number(w.get("voiced_end", te), f"voiced_end of word ({ln}, {wi})")
            db = word_db(vdb, t0, t1, fps) if isinstance(vdb, list) and vdb else DB_FLOOR + 0.5 * DB_RANGE
            out.append(Word(ln, wi, len(out), t0, te, t1, db, spell(w["text"], case, bool(clean))))
    return out


# ------------------------------------------------------------------------------------------------------ motion
def spread_max(t0, t1):
    """Largest letter spread (fraction of the word's own layout) of a note held from t0 to t1."""
    return 0.07 + 0.27 * sstep(0.12, 0.85, max(t1 - t0, 0.06))


def spread_amount(t0, t1, t):
    """Extra letter spacing while the note is held (ease-out cubic to the maximum), relaxing after it."""
    hold = max(t1 - t0, 0.06)
    amp = spread_max(t0, t1)
    u = t - t0
    if u < 0.0:
        return 0.0
    if t <= t1:
        return amp * eo3(u / hold)
    return amp * math.exp(-(t - t1) / 0.13)


def weight_level(w, vs, t, fps, breath=BREATH):
    """0..1 weight level of word w at clip time t: its own level, breathing with the voice while the note is held."""
    base = level(w.db)
    if w.start <= t <= w.voiced_end + HOLD_TAIL:
        env = sstep(0.0, 0.05, t - w.start) * (1.0 - sstep(0.0, 0.10, t - w.voiced_end))
        return base + breath * env * (level(vocal_at(vs, t, fps)) - base)
    return base


def pop_scale(tau, a=0.30, td=0.075, tp=0.22):
    """Damped overshoot spring: 1 + a at the landing, settling to 1."""
    if tau < 0.0:
        return 1.0
    return 1.0 + a * math.exp(-tau / td) * math.cos(2.0 * math.pi * tau / tp)


def drop_state(tau):
    """A word dropping into its cell: (offset in cell heights, up is positive, y squash, x stretch)."""
    t1 = 0.115
    if tau < t1:
        x = tau / t1
        return 0.50 * (1.0 - x * x), 1.0 + 0.05 * x, 1.0 - 0.02 * x
    u = tau - t1
    return (0.045 * abs(math.sin(math.pi * u / 0.16)) * math.exp(-u / 0.13),
            1.0 - 0.12 * math.exp(-u / 0.045), 1.0 + 0.05 * math.exp(-u / 0.05))


def steam_progress(tau, span):
    """Share of the way from the rim to its slot a steam word has risen `tau` s after landing (it arrives after `span`)."""
    x = tau / span if span > 1e-9 else 1.0
    x = min(1.0, max(0.0, x))
    return 0.85 * eo2(x) + 0.15 * x if span > 1e-9 else 0.0


def exit_tick(ticks, t_end, t_cut, fps, min_run=MIN_RUN):
    """Start time (s) of the drip exit: one frame ahead of the first tempo tick from `t_end - 0.03` to `t_cut -
    min_run` (so the first displaced frame is the tick's own frame), or None when the line has to hold to the cut."""
    for tk in ticks:
        if t_end - DRIP_LEAD <= tk <= t_cut - min_run:
            return (math.floor(tk * fps + 0.4) - 1) / fps
    return None


def drip_letter(u, i, em_rel, g, stretch, life):
    """One letter `u` s into its exit (u <= 0: nothing yet): fall (in em), stretch about its top, sideways wobble (in
    em), opacity. The node tree builds exactly these expressions. `em_rel` scales the wobble (the original's 0.03 em)."""
    u = max(u, 0.0)
    s = 1.0 - math.exp(-u / DRIP_TAU)
    sy = 1.0 + stretch * s
    return {"dy": 0.5 * g / DRIP_FALL * u * u, "sy": sy, "sx": 1.0 / math.sqrt(sy),
            "dx": 0.03 * em_rel * math.sin(9.0 * u + i) * s, "alpha": 1.0 - sstep(0.14 * life, life, u)}


def drip_end(t_start, delay, life, spread=DRIP_SPREAD):
    """Time (s) after which every letter of a dripping word is gone."""
    return t_start + delay + spread + life


# ------------------------------------------------------------------------------------------------------ arrival styles
def arrival(style, tau, p):
    """Channels of an arrival `tau` s after landing: scale, dx, dy (em; up and right positive), dxp, dyp (fractions of
    the panel), sx, sy, rot (deg), alpha. `p` holds the style's numbers (see STYLE_DEFAULTS)."""
    c = {"scale": 1.0, "dx": 0.0, "dy": 0.0, "dxp": 0.0, "dyp": 0.0, "sx": 1.0, "sy": 1.0, "rot": 0.0, "alpha": 1.0}
    if p["alpha0"] < 1.0:
        c["alpha"] = p["alpha0"] + (1.0 - p["alpha0"]) * sstep(0.0, p["alpha_frames"] / p["fps"], tau)
    if style in ("pop", "slap"):
        c["scale"] = pop_scale(tau, p["a"], p["td"], p["tp"])
        c["dy"] = -p["rise"] * (1.0 - eo3(tau / p["rise_s"])) if p["rise"] else 0.0
        c["dx"] = p["jitter"] * math.exp(-tau / 0.05)
        if style == "slap":
            c["rot"] = p["wobble"] * math.exp(-tau / 0.07) * math.cos(2.0 * math.pi * tau / 0.16)
    elif style == "drop":
        dy, sy, sx = drop_state(tau)
        c["dyp"], c["sy"], c["sx"] = dy * p["drop"], sy, sx
    elif style == "rise":
        g = steam_progress(tau, p["span"])
        c["dxp"] = p["dxp0"] * (1.0 - g)
        c["dyp"] = p["dyp0"] * (1.0 - g)
        a = min(tau, p["span"])
        c["rot"] = p["sway_rot"] * sstep(0.0, 0.5, a) * math.cos(2.0 * math.pi * 0.95 * a + p["phi"])
        c["dx"] = p["sway"] * sstep(0.0, 0.5, a) * math.sin(2.0 * math.pi * 0.95 * a + p["phi"])
        c["dy"] = 0.033 * sstep(0.0, 0.4, a) * math.sin(2.0 * math.pi * 1.6 * a + 1.7 * p["phi"])
        if tau <= p["span"] + 0.2:
            pop = pop_scale(tau, p["a"], p["td"], p["tp"])
        else:
            pop = 1.0
        c["scale"] = pop
    return c


STYLE_DEFAULTS = {
    "pop": {"a": 0.20, "td": 0.06, "tp": 0.18, "rise": 0.0, "rise_s": 0.2, "jitter": 0.0, "alpha0": 1.0,
            "alpha_frames": 2.0, "wobble": 0.0, "drop": 1.0},
    "slap": {"a": 0.17, "td": 0.045, "tp": 0.13, "rise": 0.0, "rise_s": 0.2, "jitter": 0.0, "alpha0": 1.0,
             "alpha_frames": 2.0, "wobble": 2.8, "drop": 1.0},
    "drop": {"alpha0": 1.0, "alpha_frames": 2.0, "drop": 1.0},
    "rise": {"a": 0.22, "td": 0.06, "tp": 0.20, "alpha0": 1.0, "alpha_frames": 2.0, "sway": 0.13, "sway_rot": 7.0,
             "base": None},
    "type": {"alpha0": 1.0, "alpha_frames": 2.0},
    "none": {"alpha0": 1.0, "alpha_frames": 2.0},
}
STYLE_SPAN = {"pop": 0.6, "slap": 0.6, "drop": 1.0, "rise": 0.0, "type": 0.0, "none": 0.0}      # s the arrival can last


def style_params(style, over, fps):
    if style not in STYLE_DEFAULTS:
        raise LyricsError(f"lyrics: style {style!r} must be one of {list(STYLES)}")
    p = dict(STYLE_DEFAULTS[style])
    unknown = sorted(set(over or {}) - set(p) - {"fps"})
    if unknown:
        raise LyricsError(f"lyrics: arrive keys {unknown} do not belong to style {style!r} (known: {sorted(p)})")
    p.update(over or {})
    p["fps"] = float(fps)
    return p


# ------------------------------------------------------------------------------------------------------ key lists
def compress(keys, tol=1e-5):
    """Drop the keys of a [[t, v]] list that lie on the straight line between their neighbours (LINEAR keys stay
    exact on every frame) and a constant tail (the last value holds); a constant list shrinks to its first key."""
    keys = [(float(t), float(v)) for t, v in keys]
    if len(keys) > 2:
        out = [keys[0]]
        for i in range(1, len(keys) - 1):
            (t0, v0), (t1, v1), (t2, v2) = out[-1], keys[i], keys[i + 1]
            lin = v0 + (v2 - v0) * (t1 - t0) / (t2 - t0)
            if abs(v1 - lin) > tol * max(1.0, abs(v1)):
                out.append(keys[i])
        out.append(keys[-1])
        keys = out
    while len(keys) > 1 and abs(keys[-1][1] - keys[-2][1]) <= tol * max(1.0, abs(keys[-1][1])):
        keys.pop()
    return [list(k) for k in keys]


def _is_const(keys, value, tol=1e-9):
    return all(abs(v - value) <= tol for _, v in keys)


# ------------------------------------------------------------------------------------------------------ layouts
def fit_em(max_w, block_h, panel, fit, size, cap):
    """Em (m) for a block `max_w` x `block_h` (em units at size 1) on a panel: the largest that fits `fit` of it, at most
    `size` / cap when a cap height is given."""
    f = 0.9 if fit is None else float(fit)
    if not 0.0 < f <= 1.0:
        raise LyricsError("lyrics: fit must be in (0, 1]")
    em = math.inf
    if panel:
        pw, ph = float(panel[0]) * f, float(panel[1]) * f
        if max_w > 0:
            em = min(em, pw / max_w)
        if block_h > 0:
            em = min(em, ph / block_h)
    if size is not None:
        if not float(size) > 0:
            raise LyricsError("lyrics: size must be positive")
        em = min(em, float(size) / cap)
    if math.isinf(em):
        raise LyricsError("lyrics: this layout needs a panel (`on`, or `box`) or a `size`")
    return em


def _cycle(seq, k, default=None):
    if seq is None or len(seq) == 0:
        return default
    return seq[k % len(seq)]


def layout_flow(boxes, rows, panel, fit, size, cap, gap=0.28, leading=1.5):
    """Words side by side along rows (`rows` = row of each word, one row per lyric line), the block centred on the panel.
    `boxes` = per word (w, y0, y1): the ink box in em with the baseline at 0 (widths include the spread headroom); `gap`
    in em. Returns (em, [(u, v)]): metres from the panel's centre to each word's ink-box centre."""
    n_rows = max(rows) + 1
    count = [rows.count(r) for r in range(n_rows)]
    row_w = [sum(b[0] for b, r in zip(boxes, rows) if r == i) + gap * max(count[i] - 1, 0) for i in range(n_rows)]
    top = max(b[2] for b in boxes)
    bot = min(b[1] for b in boxes)
    pitch = leading * cap
    block_h = (top - bot) + (n_rows - 1) * pitch
    em = fit_em(max(row_w), block_h, panel, fit, size, cap)
    used = [0.0] * n_rows
    out = []
    for (w, y0, y1), r in zip(boxes, rows):
        x = -row_w[r] / 2 + used[r] + w / 2
        used[r] += w + gap
        base = 0.5 * block_h - top - r * pitch                  # baseline of row r from the block's centre line
        out.append((x * em, (base + 0.5 * (y0 + y1)) * em))
    return em, out


def layout_stack(boxes, n_rows, panel, fit, size, cap, pitch=1.3, direction="down", align=None, shift=None):
    """One word per row, word k on row k mod `n_rows`; rows `pitch` em apart (baseline to baseline), the first at the top
    (direction "down") or at the bottom ("up"). `align` cycles left / center / right inside the margin and `shift` cycles
    extra x shifts (shares of the usable width) per row. Returns (em, [(u, v)], [row of each word], [align of each])."""
    if direction not in ("down", "up"):
        raise LyricsError("lyrics: stack dir is 'down' (first word on top) or 'up' (first word at the bottom)")
    rows = [k % n_rows for k in range(len(boxes))]
    top = max(b[2] for b in boxes)
    bot = min(b[1] for b in boxes)
    block_h = (top - bot) + (n_rows - 1) * pitch
    em = fit_em(max(b[0] for b in boxes), block_h, panel, fit, size, cap)
    f = 0.9 if fit is None else float(fit)
    aw = float(panel[0]) * f if panel else max(b[0] for b in boxes) * em
    out, aligns = [], []
    for (w, y0, y1), r in zip(boxes, rows):
        a = _cycle(align, r, "center")
        if a not in ("left", "center", "right"):
            raise LyricsError("lyrics: stack align is left, center or right")
        x = {"left": -aw / 2 + w * em / 2, "center": 0.0, "right": aw / 2 - w * em / 2}[a]
        x += (_cycle(shift, r, 0.0) or 0.0) * aw
        line = r if direction == "down" else n_rows - 1 - r
        base = 0.5 * block_h - top - line * pitch
        out.append((x, (base + 0.5 * (y0 + y1)) * em))
        aligns.append(a)
    return em, out, rows, aligns


def recycle_hide(lands, slots, gap=0):
    """Frame each word has to be gone by: the landing of the next word that uses the same slot, minus `gap` frames
    (None: nobody takes the slot over)."""
    out = []
    for k, s in enumerate(slots):
        nxt = next((lands[j] for j in range(k + 1, len(slots)) if slots[j] == s), None)
        out.append(None if nxt is None else nxt - int(gap))
    return out


# ------------------------------------------------------------------------------------------------------ expand
FLOW_KEYS, STACK_KEYS, SLOT_KEYS = {"gap"}, {"rows", "pitch", "dir", "align", "shift", "tilt"}, {"slots", "assign"}
DRIP_KEYS = {"g", "stretch", "life", "gap", "min_run", "order"}
RECYCLE_KEYS = {"gap", "fade", "assign"}
RECOLOR_KEYS = {"color", "from", "over", "ease", "words"}
WEIGHT_KEYS = {"lo", "hi", "breath"}
SPREAD_KEYS = {"scale"}


def _table(lyr, key, allowed):
    """lyrics[key] as a dict (True = defaults) after checking its keys, or None when absent / false."""
    v = lyr.get(key)
    if v is None or v is False:
        return None
    v = {} if v is True else v
    if not isinstance(v, dict):
        raise LyricsError(f"lyrics: `{key}` must be a table")
    unknown = sorted(set(v) - allowed)
    if unknown:
        raise LyricsError(f"lyrics: {key} keys {unknown} are unknown (known: {sorted(allowed)})")
    return v


def _seconds(x, what):
    if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(float(x)):
        raise LyricsError(f"lyrics: {what} must be a number of seconds")
    return float(x)


def _accent(w, lyr, entry):
    colors = lyr.get("colors")
    if not colors:
        return entry.get("color")
    colors = [colors] if isinstance(colors, str) else list(colors)
    by = lyr.get("color_by", "line")
    if by not in ("line", "word"):
        raise LyricsError("lyrics: color_by is 'line' (cycle with the line's number) or 'word' (with the word's place "
                          "in the selection)")
    return colors[((w.line - 1) if by == "line" else w.k) % len(colors)]


def _word_boxes(words, measure, what):
    if measure is None:
        raise LyricsError(f"lyrics: layout {what!r} measures the words; the build supplies that")
    boxes, cap = [], None
    for w in words:
        m = measure(w.text)
        cap = m["cap"] if cap is None else cap
        boxes.append((m["w"] * (1.0 + spread_max(w.start, w.voiced_end)), m["y0"], m["y1"]))
    return boxes, cap


def _places(words, entry, lyr, panel, measure):
    """Where every word goes: (places, slot of each word, common cap-height size or None). A place is {"spec": keys for
    the word's text, "offset": (u, v) metres from the panel's centre or None, "tilt": degrees or None}."""
    n = len(words)
    layout = lyr.get("layout", "same")
    lay = dict(layout) if isinstance(layout, dict) else {"kind": layout}
    kind = lay.pop("kind", "same")
    if kind not in LAYOUTS:
        raise LyricsError(f"lyrics: layout kind {kind!r} must be one of {list(LAYOUTS)}")
    allowed = {"same": set(), "flow": FLOW_KEYS, "stack": STACK_KEYS, "slots": SLOT_KEYS}[kind]
    if set(lay) - allowed:
        raise LyricsError(f"lyrics: layout {kind!r} keys {sorted(set(lay) - allowed)} are unknown "
                          f"(known: {sorted(allowed)})")
    places = [{"spec": {}, "offset": None, "tilt": None} for _ in range(n)]
    on = entry.get("on")
    surfaces = len(on) if isinstance(on, (list, tuple)) else 1
    slot_of = [k % surfaces for k in range(n)]
    size = None
    if kind == "same":
        return kind, places, slot_of, size
    fit, sz = entry.get("fit"), entry.get("size")
    boxes, cap = _word_boxes(words, measure, kind)
    if kind == "flow":
        lines = sorted({w.line for w in words})
        rows = [lines.index(w.line) for w in words]
        gap = float(lay.get("gap", 0.28)) * float(entry.get("word_spacing", 1.0))
        em, off = layout_flow(boxes, rows, panel, fit, sz, cap, gap, float(entry.get("leading", 1.5)))
        slot_of = list(range(n))
        aligns = ["center"] * n
        tilts = [None] * n
    elif kind == "stack":
        rows_n = int(lay.get("rows", n))
        if rows_n < 1:
            raise LyricsError("lyrics: stack rows must be at least 1")
        em, off, slot_of, aligns = layout_stack(boxes, min(rows_n, n) if rows_n > n else rows_n, panel, fit, sz, cap,
                                                float(lay.get("pitch", 1.3)), lay.get("dir", "down"), lay.get("align"),
                                                lay.get("shift"))
        tilts = [None if _cycle(lay.get("tilt"), r) is None else float(_cycle(lay.get("tilt"), r)) for r in slot_of]
    else:
        slots = lay.get("slots")
        if not slots or not isinstance(slots, list):
            raise LyricsError("lyrics: layout 'slots' needs slots = [{at = [u, v], box = [w, h], align, tilt}, ...]")
        if not panel:
            raise LyricsError("lyrics: layout 'slots' needs a panel (`on`, or `box`)")
        assign = lay.get("assign")
        slot_of = [int(_cycle(assign, k, k % len(slots))) % len(slots) for k in range(n)]
        for k in range(n):
            s = slots[slot_of[k]]
            if not isinstance(s, dict) or "at" not in s:
                raise LyricsError("lyrics: a slot is a table with `at = [u, v]` (shares of the panel, 0 = its centre)")
            places[k]["offset"] = (float(s["at"][0]) * panel[0], float(s["at"][1]) * panel[1])
            if s.get("box") is not None:
                places[k]["spec"]["box"] = [float(s["box"][0]) * panel[0], float(s["box"][1]) * panel[1]]
            if s.get("size") is not None:
                places[k]["spec"]["size"] = float(s["size"])
            if s.get("align") is not None:
                places[k]["spec"]["align"] = s["align"]
            if s.get("tilt") is not None:
                places[k]["tilt"] = float(s["tilt"])
        return kind, places, slot_of, size
    for k in range(n):
        places[k]["offset"] = (off[k][0], off[k][1])
        places[k]["spec"]["align"] = "center"
        places[k]["spec"]["valign"] = "middle"
        places[k]["tilt"] = tilts[k]
    size = em * cap
    return kind, places, slot_of, size


def _with_offset(places, off0):
    for pl in places:
        if pl["offset"] is not None:
            pl["spec"]["offset"] = [off0[0] + pl["offset"][0], off0[1] + pl["offset"][1]]


def expand(entry, tl, *, fps, frame0=0, panel=None, measure=None):
    """Turn a `[[text]]` entry with `lyrics = {...}` into (specs, summary): one ordinary text spec per selected word that
    is ever on screen, plus numbers only (no lyric text) about what was built.

    entry    the [[text]] table: `name`, the placement and look keys every word inherits (`on` may be a list of surfaces:
             word k goes to surface k mod len), `lyrics`
    tl       the loaded timeline (dict)
    fps, frame0   the project's; `from` / `to` are rounded to frames exactly as the shots stage rounds its cuts
    panel    (w, h) metres of the entry's surface or `box`, for the stack / flow / slots layouts and panel-sized motion
    measure  callable(string) -> {"w", "y0", "y1", "cap"}: the ink box of `string` at em size 1 (centred, baseline at 0)
             and the cap height per em; only layouts that put words next to each other need it
    Each spec has `name` `<name>_l<line>w<word>`, `text`, `lyric = [line, word]` and a `kinetic` table (docs/design.md:
    Text, lyrics) of keys the text stage turns into node inputs."""
    name = entry.get("name")
    if not name:
        raise LyricsError("[[text]] with lyrics needs a name")
    lyr = entry.get("lyrics")
    if not isinstance(lyr, dict):
        raise LyricsError(f"text {name!r}: lyrics must be a table")
    unknown = sorted(set(lyr) - LYRICS_KEYS)
    if unknown:
        raise LyricsError(f"text {name!r}: lyrics keys {unknown} are unknown (known: {sorted(LYRICS_KEYS)})")
    for bad in ("text", "value", "kinetic", "reveal", "lyric"):
        if bad in entry:
            raise LyricsError(f"text {name!r}: a lyrics entry takes its words from the timeline: drop `{bad}`")
    try:
        return _expand(entry, name, lyr, tl, float(fps), int(frame0), panel, measure)
    except LyricsError as e:
        msg = str(e)
        raise LyricsError(msg if msg.startswith("text ") else f"text {name!r}: {msg}") from None


def _expand(entry, name, lyr, tl, fps, frame0, panel, measure):
    words = select(tl, lyr.get("line"), lyr.get("lines"), lyr.get("words"), fps, lyr.get("case", "keep"),
                   lyr.get("clean", False))
    n = len(words)
    style = lyr.get("style", "pop")
    p = style_params(style, lyr.get("arrive"), fps)
    first = cut_frame(_seconds(lyr["from"], "from"), fps, frame0) if lyr.get("from") is not None else None
    t_to = _seconds(lyr["to"], "to") if lyr.get("to") is not None else max(w.voiced_end for w in words)
    cut = cut_frame(t_to, fps, frame0)
    lands = [landing_frame(w.start, fps, first) for w in words]

    kind, places, slot_of, size = _places(words, entry, lyr, panel, measure)
    _with_offset(places, entry.get("offset") or (0.0, 0.0))
    tilts = lyr.get("tilt")
    tilt_rnd = 0.0
    if isinstance(tilts, dict):
        if set(tilts) - {"random"}:
            raise LyricsError("lyrics: tilt is a number, a list of numbers or {random = degrees}")
        tilt_rnd, tilts = float(tilts.get("random", 0.0)), None
    tilts = [tilts] if isinstance(tilts, (int, float)) and not isinstance(tilts, bool) else tilts

    # when each word is gone: the cut, its drip, the next word taking its slot
    hide = [cut] * n
    leave = lyr.get("leave", "cut")
    if leave not in ("cut", "drip"):
        raise LyricsError("lyrics: leave is 'cut' (hold to the cut) or 'drip' (drip away on a tempo tick when one fits)")
    drip = _table(lyr, "drip", DRIP_KEYS) or {}
    t_drip = None
    drip_t0 = [None] * n
    if leave == "drip":
        t_drip = exit_tick(ticks_of(tl), max(w.voiced_end for w in words), t_to, fps, float(drip.get("min_run", MIN_RUN)))
    if t_drip is not None:
        order = drip.get("order", "none")
        if order not in ("none", "bottom_first", "last_first"):
            raise LyricsError("lyrics: drip order is 'none', 'bottom_first' (rows low to high) or 'last_first'")
        ranks = _drip_ranks(order, places, n)
        for k in range(n):
            drip_t0[k] = t_drip + float(drip.get("gap", 0.0)) * ranks[k]
            hide[k] = min(hide[k], int(math.ceil(drip_end(drip_t0[k], 0.0, float(drip.get("life", 0.55))) * fps)) + 1)
    rec = _table(lyr, "recycle", RECYCLE_KEYS)
    group = slot_of
    if rec is not None:
        if rec.get("assign") is not None:        # which words take each other's place, apart from where they stand
            if not isinstance(rec["assign"], (list, tuple)) or not rec["assign"]:
                raise LyricsError("lyrics: recycle assign is a list of group numbers, cycled over the words")
            group = [int(_cycle(rec["assign"], k)) for k in range(n)]
        for k, h in enumerate(recycle_hide(lands, group, rec.get("gap", 0))):
            if h is not None:
                hide[k] = min(hide[k], h)

    keep = [k for k in range(n) if hide[k] > lands[k]]              # a word that is never on screen drops out
    skipped = [[words[k].line, words[k].index] for k in range(n) if k not in keep]
    t_last = lands[keep[-1]] / fps if keep else 0.0                 # the last word's landing (steam words stop then)
    spread = lyr.get("spread", True)
    spread_k = 0.0 if spread is False else float((_table(lyr, "spread", SPREAD_KEYS) or {}).get("scale", 1.0))
    wt = _table(lyr, "weight", WEIGHT_KEYS)
    rc = _table(lyr, "recolor", RECOLOR_KEYS)
    rc_words = None
    if rc is not None:
        if "color" not in rc:
            raise LyricsError("lyrics: recolor needs a `color`")
        if rc.get("words") is not None:           # only these words of the selection (1-based, inclusive)
            w = rc["words"]
            if not (isinstance(w, (list, tuple)) and len(w) == 2 and all(isinstance(x, int) for x in w)
                    and 1 <= w[0] <= w[1]):
                raise LyricsError("lyrics: recolor words is [first, last] (1-based, inclusive, in the selection)")
            rc_words = (int(w[0]), int(w[1]))
    vs = smooth_vocal(tl.get("vocal_db") or [])
    base_off = None
    if style == "rise":
        if not panel or places[0]["offset"] is None:
            raise LyricsError("lyrics: style 'rise' needs a layout (stack or slots) and a panel: the words rise to "
                              "their slots")
        base = p.get("base")
        if base:
            base_off = (float(base[0]) * panel[0], float(base[1]) * panel[1])
        elif keep:
            base_off = places[keep[-1]]["offset"]

    sizes = lyr.get("sizes")
    if sizes is not None:
        sizes = [sizes] if isinstance(sizes, (int, float)) and not isinstance(sizes, bool) else sizes
        if not (isinstance(sizes, (list, tuple)) and sizes
                and all(isinstance(x, (int, float)) and x > 0 for x in sizes)):
            raise LyricsError("lyrics: sizes is a list of cap heights in metres (cycled over the words)")
    offsets = lyr.get("offsets")
    if offsets is not None and (not isinstance(offsets, (list, tuple)) or not offsets or not all(
            isinstance(o, (list, tuple)) and len(o) == 2 and all(isinstance(x, (int, float)) for x in o) for o in offsets)):
        raise LyricsError("lyrics: offsets is a list of [u, v] metres (cycled over the words)")
    backing = lyr.get("backing")
    if backing is not None and not isinstance(backing, (dict, list, tuple, bool)):
        raise LyricsError("lyrics: backing is a table or a list of tables (cycled over the words)")
    inherit = {k: v for k, v in entry.items() if k not in ("lyrics", "name", "on")}
    surfaces = entry.get("on")
    specs = []
    for k in keep:
        w = words[k]
        sub = dict(inherit)
        sub.update(places[k]["spec"])
        sub["name"] = f"{name}_{w.tag}"
        sub["text"] = w.text
        sub["lyric"] = [w.line, w.index]
        if surfaces is not None:
            sub["on"] = surfaces[k % len(surfaces)] if isinstance(surfaces, (list, tuple)) else surfaces
        col = _accent(w, lyr, entry)
        if col is not None:
            sub["color"] = col
        if offsets is not None:                                         # a nudge per word on top of where it stands
            du, dv = _cycle(offsets, k)
            cur = sub.get("offset") or (0.0, 0.0)
            sub["offset"] = [float(cur[0]) + float(du), float(cur[1]) + float(dv)]
        if sizes is not None and "size" not in places[k]["spec"]:      # a slot's own size wins over the list
            sub["size"] = float(_cycle(sizes, k))
        if size is not None:
            sub.pop("fit", None)
            sub["size"] = size
        if places[k]["tilt"] is not None:
            tilt = places[k]["tilt"]
        elif tilt_rnd:
            tilt = tilt_rnd * (2.0 * h01(w.line, w.index, 31) - 1.0)      # the same turn for the same word, every build
        else:
            tilt = _cycle(tilts, k, 0.0)
        if backing is not None:
            sub["backing"] = _cycle(backing, k) if isinstance(backing, (list, tuple)) else backing
        pw = dict(p)
        if style == "rise":
            pw["span"] = max(t_last - lands[k] / fps, 0.0)
            off = places[k]["offset"]
            pw["dxp0"] = (base_off[0] - off[0]) / panel[0] if pw["span"] > 0 else 0.0
            pw["dyp0"] = (base_off[1] - off[1]) / panel[1] if pw["span"] > 0 else 0.0
            pw["phi"] = 2.0 * math.pi * h01(w.line, w.index, 3)
        rcw = rc if rc is not None and (rc_words is None or rc_words[0] <= k + 1 <= rc_words[1]) else None
        sub["kinetic"] = _kinetic(w, k, n, lands, hide[k], style, pw, fps, spread_k, wt, rcw, vs, tilt, drip, drip_t0[k],
                                  rec, group, t_last)
        if style == "type":
            t_from = lands[k] / fps
            sub["reveal"] = {"from": t_from, "to": max(w.voiced_end, t_from + 1.0 / fps)}
        specs.append(sub)
    summary = {"words": len(specs), "lines": sorted({w.line for w in words}), "style": style, "layout": kind,
               "skipped": skipped, "first_frame": min(lands[k] for k in keep) if keep else None,
               "last_frame": max(lands[k] for k in keep) if keep else None, "cut_frame": cut,
               "drip_from": None if t_drip is None else round(t_drip, 4)}
    return specs, summary


def _drip_ranks(order, places, n):
    """How many gaps after the first a word's drip starts: 0 for all, last word first, or the lowest row first."""
    if order == "last_first":
        return [n - 1 - k for k in range(n)]
    if order == "bottom_first":
        vs = [pl["offset"][1] if pl["offset"] is not None else 0.0 for pl in places]
        low = sorted(set(round(v, 6) for v in vs))
        return [low.index(round(v, 6)) for v in vs]
    return [0] * n


def _arrive_keys(style, land, hide, p, fps):
    """Per-frame arrival channels of one word, from its landing to the end of the style's span."""
    span = STYLE_SPAN.get(style, 0.0)
    if style == "rise":
        span = p["span"] + 0.25
    if p.get("alpha0", 1.0) < 1.0:
        span = max(span, p["alpha_frames"] / fps)
    last = min(land + int(math.ceil(span * fps)) + 1, hide - 1)
    rows = {k: [] for k in ("scale", "dx", "dy", "dxp", "dyp", "sx", "sy", "rot", "alpha")}
    for c in range(land, max(last, land) + 1):
        ch = arrival(style, (c - land) / fps, p)
        for k in rows:
            rows[k].append((c / fps, ch[k]))
    return rows


def _kinetic(w, k, n, lands, hide, style, p, fps, spread_k, wt, rc, vs, tilt, drip, drip_t0, rec, slot_of, t_last):
    """The `kinetic` table of one word (the keys are listed in mkmmd/core/typefx.py)."""
    land = lands[k]
    last = hide - 1
    kin = {"show": [land / fps, hide / fps]}
    arr = _arrive_keys(style, land, hide, p, fps)
    if p.get("jitter"):
        arr["dx"] = [(t, v * (2.0 * h01(w.line, w.index, 21) - 1.0)) for t, v in arr["dx"]]
    if p.get("wobble") and k % 2 == 0:
        arr["rot"] = [(t, -v) for t, v in arr["rot"]]
    for ch in ("scale", "sx", "sy", "dx", "dy", "dxp", "dyp", "rot"):
        keys = arr[ch]
        if ch == "rot" and tilt:
            keys = [(t, v + tilt) for t, v in keys]
        if not _is_const(keys, 1.0 if ch in ("scale", "sx", "sy") else 0.0) or (ch == "rot" and tilt):
            kin[ch] = compress(keys)
    if style == "drop":
        kin["pivot"] = "bottom"
    # opacity: the arrival ramp times the fade a recycled slot gets before the next word takes it over
    op = list(arr["alpha"])
    if rec is not None and float(rec.get("fade", 0)) > 0:
        fr = float(rec["fade"])
        nxt = next((lands[j] for j in range(k + 1, n) if slot_of[j] == slot_of[k]), None)
        if nxt is not None:
            t_exp = (nxt - float(rec.get("gap", 0))) / fps
            have = {round(t * fps) for t, _ in op}
            fade = lambda t: 1.0 - sstep(t_exp - fr / fps, t_exp - 0.5 / fps, t)       # noqa: E731
            op = [(t, v * fade(t)) for t, v in op]
            op += [(c / fps, fade(c / fps)) for c in range(max(int(math.floor(t_exp * fps - fr)), land), last + 1)
                   if c not in have]
            op.sort()
    if not _is_const(op, 1.0):
        kin["opacity"] = compress(op)
    if spread_k > 0.0:                      # tracking grows while the note is held and relaxes after it
        c1 = min(last, int(math.ceil((w.voiced_end + 8 * 0.13) * fps)))
        keys = [(c / fps, 1.0 + spread_k * spread_amount(w.start, w.voiced_end, c / fps))
                for c in range(land, max(c1, land) + 1)]
        if not _is_const(keys, 1.0):
            kin["tracking"] = compress(keys)
        kin["headroom"] = spread_k * spread_max(w.start, w.voiced_end)
    if wt is not None:                      # weight: a stroke radius (em) from the word's level, breathing with the voice
        lo, hi, br = float(wt.get("lo", 0.0)), float(wt.get("hi", 0.03)), float(wt.get("breath", BREATH))
        c1 = min(last, int(math.ceil((w.voiced_end + HOLD_TAIL) * fps)))
        keys = [(c / fps, lo + (hi - lo) * weight_level(w, vs, c / fps, fps, br)) for c in range(land, max(c1, land) + 1)]
        kin["weight"] = compress(keys)
    if rc is not None:                      # recolour (frost, ash): a keyed mix toward another palette colour
        t0 = t_last if rc.get("from", "last") == "last" else _seconds(rc["from"], "recolor from")
        over = float(rc.get("over", 0.5))
        ease = rc.get("ease", "smooth")
        if ease not in ("smooth", "linear"):
            raise LyricsError("lyrics: recolor ease is 'smooth' or 'linear'")
        c_end = min(max(int(math.ceil((t0 + over) * fps)), land), max(last, land))
        keys = []
        for c in range(land, c_end + 1):        # 0 until t0, then the ramp: a word that is gone by t0 never changes
            x = (c / fps - t0) / over if over > 0 else 1.0
            keys.append((c / fps, sstep(0.0, 1.0, x) if ease == "smooth" else min(1.0, max(0.0, x))))
        kin["tint"] = {"color": rc["color"], "keys": compress(keys)}
    if drip_t0 is not None:
        kin["drip"] = {"t": drip_t0, "g": float(drip.get("g", 2600.0)), "stretch": float(drip.get("stretch", 3.2)),
                       "life": float(drip.get("life", 0.55)), "seed": 1 + (1000 * w.line + w.index) % 997}
    return kin
