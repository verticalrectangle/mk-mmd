"""Googly eyes on sprites, bpy-free (docs/design.md: Frame effects, Mandalas and trails, Googly eyes): a sprite source's
`eyes` and `mouth` keys checked (`face`), where the eyes sit on a sprite (`anchors`: its registry entry's `extra.eyes`,
the rest found from its alpha by `auto_face`) and how the pupils swing (`swing`): each one a damped spring inside its
eye, resting where gravity drops it (or straight out, or toward a point it looks at), shaken by the sprite's own motion
and kicked on the hits. mkmmd/eyes.py draws them.

A face's anchors are fractions of the sprite's box (trimmed to its alpha): `at` the two eyes' centres [x, y] (x across,
y down), `r` an eye's radius as a share of the box's longer side, `mouth` the smile's middle [x, y] (false: no smile
fits that sprite). A pupil's offset is in units of its eye's free radius (how far the pupil's centre can stray from the
eye's), screen axes, x right and y down.

The swing is a function of the clip second alone: the spring is run from rest over SETTLE decay times before it, on a
grid of RATE samples a second aligned to clip second 0, the sprite's motion straight between the samples (each change of
its velocity a kick the other way: a pupil keeps going when its eye stops), each hit a kick in a direction hashed from
`seed`, the copy's key and the hit, so any frame comes out the same in any order."""
import math

import numpy as np

KEYS = ("eyes", "mouth")                 # what a sprite source may say about its face
EYES = {"size": 1.0, "wobble": 1.0, "look": "down", "seed": 1}
HITS = ("times", "lane", "min_db", "every")
WOBBLE = {"amount": 1.0, "hz": 3.0, "damping": 0.15}
LOOKS = ("down", "viewer", "center")     # gravity, straight out at the viewer, the owner's centre (or [x, y])
SAG = 0.72                               # a resting pupil this far down its free radius (gravity)
GAZE = 0.8                               # a pupil looking at a point: this far toward it
RATE = 60.0                              # motion samples a second
SETTLE = 4.0                             # the window run before a frame: this many of the spring's decay times
MAX_WINDOW = 3.0                         # seconds, at most
KICK = 1.8                               # a hit's kick: the swing it starts, in free radii (up to the rim and back)
SPREAD = 1.0                             # radians either side of straight up a hit kicks a resting pupil
JITTER = 0.3                             # radians either side of its sprite's way an eye's kick goes, and +-15 % strength
BOUNCE = 0.35                            # restitution where a pupil meets its eye's rim
VMAX = 8.0                               # fastest pupil, free radii a second per radian a second of the spring
# the auto face: an eye's radius as shares of the widest part's width, bounded by shares of the box's longer side
EYE_R = (0.19, 0.055, 0.12)
EYE_GAP = 0.3                            # the gap between the eyes, eye radii
WIDE = 0.9                               # the widest part: rows at least this share of the widest row's width
UPPER = 0.3                              # the eyes this far down the widest part (its upper middle)
MOUTH_DROP = 1.75                        # the smile's middle below the eyes' line, eye radii


def _fx():
    from . import framefx as FX          # framefx imports mandala, which imports this module: import it when used
    return FX


# ================================================================================================= the keys
def face(src, what):
    """A sprite source's face (`src`, its table: `eyes` true or {size, wobble, look, seed, times or lane (min_db, every)},
    `mouth` true or false) checked and filled in: None without eyes or a mouth, else {eyes, mouth, size, amount, hz,
    damping, look, seed, times, lane, min_db, every}. `look`: "down", "viewer", "center" or [x, y] (frame fractions);
    `times` None and `lane` None: the owner's hits (FrameFx gives them, mkmmd.eyes.hit_times picks)."""
    FX = _fx()
    e, m = src.get("eyes", False), src.get("mouth", False)
    if not isinstance(m, bool):
        raise FX.FxError(f"{what}: mouth = {m!r}: true (a small doodled smile under the eyes) or false")
    if e is True:
        e = {}
    elif e is False or e is None:
        e = None
    elif not isinstance(e, dict):
        raise FX.FxError(f"{what}: eyes = {e!r}: true or a table {{size, wobble, look, seed, lane, times}}")
    has_eyes = e is not None
    if not has_eyes and not m:
        return None
    e = e or {}
    known = set(EYES) | set(HITS)
    extra = sorted(set(e) - known)
    if extra:
        raise FX.FxError(f"{what}: eyes: unknown keys {extra} (known: {sorted(known)})")
    out = {"eyes": has_eyes, "mouth": m,
           "size": FX._num(e.get("size", EYES["size"]), f"{what} eyes size", 0.1, 4.0),
           "seed": FX._num(e.get("seed", EYES["seed"]), f"{what} eyes seed", lo=0, integer=True)}
    out.update(_wobble(FX, e.get("wobble", EYES["wobble"]), what))
    out["look"] = _look(FX, e.get("look", EYES["look"]), what)
    hits = {"times": None, "lane": None, "min_db": None, "every": 1}
    if any(k in e for k in HITS):
        FX._hits(e, hits, f"{what} eyes")
    out.update(times=hits.get("times"), lane=hits.get("lane"), min_db=hits.get("min_db"), every=hits.get("every", 1))
    return out


def _wobble(FX, v, what):
    """`wobble`: a number (`amount`: 0 still pupils, 1 the default swing) or {amount, hz, damping}."""
    if not isinstance(v, dict):
        v = {"amount": v}
    extra = sorted(set(v) - set(WOBBLE))
    if extra:
        raise FX.FxError(f"{what}: eyes wobble: unknown keys {extra} (known: {sorted(WOBBLE)})")
    w = {**WOBBLE, **v}
    return {"amount": FX._num(w["amount"], f"{what} eyes wobble amount", 0.0, 10.0),
            "hz": FX._num(w["hz"], f"{what} eyes wobble hz", 0.3, 12.0),
            "damping": FX._num(w["damping"], f"{what} eyes wobble damping", 0.02, 0.9)}


def _look(FX, v, what):
    if isinstance(v, str):
        if v not in LOOKS:
            raise FX.FxError(f"{what}: eyes look {v!r}: one of {', '.join(LOOKS)}, or [x, y] (frame fractions)")
        return v
    return FX._pair(v, f"{what} eyes look")


# ================================================================================================= where they sit
def _pool(alpha, side=256):
    """`alpha` averaged down in k x k blocks (padded with nothing to whole blocks) so its longer side is at most `side`
    px, and k."""
    a = np.asarray(alpha, np.float64)
    k = max(1, int(math.ceil(max(a.shape) / float(side))))
    if k == 1:
        return a, 1
    h, w = -(-a.shape[0] // k) * k, -(-a.shape[1] // k) * k
    a = np.pad(a, ((0, h - a.shape[0]), (0, w - a.shape[1])))
    return a.reshape(h // k, k, w // k, k).mean(axis=(1, 3)), k


def _runs(m):
    """(width, middle) of the longest run of covered pixels in each row of mask `m` (0, nan in an empty row)."""
    h, w = m.shape
    width, mid = np.zeros(h), np.full(h, np.nan)
    d = np.diff(np.pad(m.astype(np.int8), ((0, 0), (1, 1))), axis=1)
    for y in range(h):
        s, e = np.flatnonzero(d[y] == 1), np.flatnonzero(d[y] == -1)
        if len(s):
            i = int(np.argmax(e - s))
            width[y], mid[y] = e[i] - s[i], 0.5 * (s[i] + e[i]) - 0.5
    return width, mid


def _cover(m, cx, cy, r):
    """The share of the disc (cx, cy, r) (px of mask `m`) that `m` covers (outside the mask: not covered)."""
    g = np.linspace(-1.0, 1.0, 9)
    x, y = np.meshgrid(g, g)
    inside = x * x + y * y <= 1.0
    px, py = np.rint(cx + r * x[inside]).astype(int), np.rint(cy + r * y[inside]).astype(int)
    h, w = m.shape
    ok = (px >= 0) & (px < w) & (py >= 0) & (py < h)
    return float(np.sum(m[py[ok], px[ok]])) / float(inside.sum())


def auto_face(alpha, thresh=0.5):
    """A face for a sprite with coverage `alpha` ((h, w), its box trimmed to it): {at, r, mouth} (fractions, as the
    module says). The eyes sit side by side on the upper middle of the sprite's widest part (the rows at least WIDE of
    its widest row's width; their longest runs), their radius EYE_R of its width within bounds of the box's longer side,
    moved down or up the part to where both lie on the sprite; the smile under them where the sprite still is."""
    a, k = _pool(alpha)
    H, W = np.asarray(alpha).shape
    m = a >= thresh
    if not m.any():
        m = a > 0
    h, w = m.shape
    width, mid = _runs(m)
    win = max(1, int(round(0.05 * h)) | 1)
    smooth = np.convolve(width, np.ones(win) / win, mode="same")
    rows = np.flatnonzero(smooth >= WIDE * smooth.max())
    top, bottom = int(rows.min()), int(rows.max())
    part = np.zeros(h, bool)
    part[rows] = True
    part &= width > 0
    cx = float(np.nanmedian(mid[part])) if part.any() else 0.5 * (w - 1)
    long_side = max(H, W) / float(k)
    r = float(np.clip(EYE_R[0] * float(width.max()), EYE_R[1] * long_side, EYE_R[2] * long_side))
    want = top + UPPER * (bottom - top)
    dx = r * (1.0 + 0.5 * EYE_GAP)
    best = None
    for y in np.arange(want - r, want + 2.0 * r + 1e-9, 0.25 * r):
        if not 0.0 <= y <= h - 1:
            continue
        row = mid[int(round(y))]
        x0 = 0.5 * ((row if np.isfinite(row) else cx) + cx)
        c = min(_cover(m, x0 - dx, y, r), _cover(m, x0 + dx, y, r))
        score = (1, -abs(y - want)) if c >= 0.95 else (0, c - 0.02 * abs(y - want) / r)
        if best is None or score > best[0]:
            best = (score, x0, y)
    _, x0, y = best if best is not None else (None, cx, want)
    mouth = [x0, min(y + MOUTH_DROP * r, h - 1.0)]
    for drop in (MOUTH_DROP, 1.45, 1.2):
        if y + drop * r <= h - 1 and _cover(m, x0, y + drop * r, 0.45 * r) >= 0.9:
            mouth = [x0, y + drop * r]
            break

    def frac(x, n):                      # a pooled pixel's middle as a share of the box's side of n px
        return float((x + 0.5) * k / n)
    return {"at": [[frac(x0 - dx, W), frac(y, H)], [frac(x0 + dx, W), frac(y, H)]], "r": r / long_side,
            "mouth": [frac(mouth[0], W), frac(mouth[1], H)]}


def anchors(extra, alpha, what="sprite"):
    """A sprite's face anchors: its registry entry's `extra.eyes` ({at, r, mouth}, any of them; None: none) over the auto
    face of its coverage `alpha` (auto_face, worked out only when a key is missing). A malformed one is an FxError."""
    FX = _fx()
    if extra is None:
        return auto_face(alpha)
    if not isinstance(extra, dict) or set(extra) - {"at", "r", "mouth"}:
        raise FX.FxError(f"{what}: extra.eyes = {extra!r}: expected {{at = [[x, y], [x, y]], r, mouth = [x, y] or "
                         f"false}} (fractions of the sprite's box)")
    out = {} if {"at", "r", "mouth"} <= set(extra) else auto_face(alpha)
    if "at" in extra:
        at = extra["at"]
        if not (isinstance(at, (list, tuple)) and len(at) == 2):
            raise FX.FxError(f"{what}: extra.eyes at = {at!r}: two [x, y] (fractions of the sprite's box)")
        out["at"] = [FX._pair(p, f"{what} extra.eyes at") for p in at]
        if any(not -0.5 <= v <= 1.5 for p in out["at"] for v in p):
            raise FX.FxError(f"{what}: extra.eyes at = {at!r}: fractions of the sprite's box (0..1, a little past it)")
    if "r" in extra:
        out["r"] = FX._num(extra["r"], f"{what} extra.eyes r", 0.005, 0.5)
    if "mouth" in extra:
        mo = extra["mouth"]
        out["mouth"] = None if mo is False or mo is None else FX._pair(mo, f"{what} extra.eyes mouth")
    return out


# ================================================================================================= how they swing
def window(face, t):
    """The clip seconds the swing is run over for clip second `t`: the grid of RATE a second (aligned to 0) from SETTLE
    of the spring's decay times before `t` (at most MAX_WINDOW s) up to `t`, which comes last."""
    decay = 1.0 / (2.0 * math.pi * face["hz"] * face["damping"])
    n = int(math.ceil(min(SETTLE * decay, MAX_WINDOW) * RATE))
    g1 = int(math.floor(t * RATE + 1e-9))
    ts = np.arange(g1 - n, g1 + 1, dtype=np.float64) / RATE
    if t - ts[-1] > 1e-9:
        ts = np.append(ts, t)
    else:
        ts[-1] = t
    return ts


def rest(face, centres, look=None):
    """Where the pupils of eyes at `centres` ((k, 2) px) rest, (k, 2) offsets: `look` None, gravity (SAG down); "viewer"
    straight out (0, 0); a point (x, y) px, GAZE toward it."""
    c = np.asarray(centres, np.float64)
    if look is None:
        return np.tile([0.0, SAG], (len(c), 1))
    if isinstance(look, str):
        return np.zeros((len(c), 2))
    d = np.asarray(look, np.float64)[None, :] - c
    n = np.linalg.norm(d, axis=1, keepdims=True)
    return np.where(n > 1e-6, GAZE * d / np.maximum(n, 1e-6), 0.0)


_G = np.uint64(0x9E3779B97F4A7C15)
_M1, _M2 = np.uint64(0xBF58476D1CE4E5B9), np.uint64(0x94D049BB133111EB)


def hash01(*parts):
    """Uniform floats in [0, 1) hashed from whole numbers (arrays broadcast together; splitmix64): the same numbers,
    the same floats, on any machine and in any order."""
    arrs = [np.asarray(p, np.int64) for p in parts]
    x = np.zeros(np.broadcast(*arrs).shape, np.uint64)
    with np.errstate(over="ignore"):
        for p in arrs:
            x = x ^ p.astype(np.uint64)
            x = x + _G
            x = (x ^ (x >> np.uint64(30))) * _M1
            x = (x ^ (x >> np.uint64(27))) * _M2
            x = x ^ (x >> np.uint64(31))
    return (x >> np.uint64(11)).astype(np.float64) * (1.0 / 9007199254740992.0)


def kicks(face, h, keys, look=None):
    """The velocity kicks (k, 2) (free radii a second) a hit at clip second `h` gives the pupils of the eyes `keys`
    ((k, K) whole numbers, the last the eye of its sprite): each sprite kicked one way (up, within SPREAD of straight
    up, when its pupils rest under gravity; any way when they look), each eye within JITTER of it."""
    keys = np.asarray(keys, np.int64).reshape(len(keys), -1)
    hit = int(round(h * 1000.0))
    sprite = keys[:, :-1] if keys.shape[1] > 1 else np.zeros((len(keys), 1), np.int64)
    u = hash01(face["seed"], hit, 1, *sprite.T)
    a = -0.5 * math.pi + (2.0 * u - 1.0) * SPREAD if look is None else 2.0 * math.pi * u
    e1, e2 = hash01(face["seed"], hit, 2, *keys.T), hash01(face["seed"], hit, 3, *keys.T)
    a = a + (2.0 * e1 - 1.0) * JITTER
    s = face["amount"] * KICK * 2.0 * math.pi * face["hz"] * (0.85 + 0.3 * e2)
    return np.stack([np.cos(a) * s, np.sin(a) * s], axis=1)


def swing(face, ts, centres, free, hits=(), keys=None, look=None):
    """The pupils' offsets (k, 2) at clip second ts[-1] of eyes whose centres are at `centres` ((n, k, 2) px) and whose
    free radii are `free` ((n, k) px) at the clip seconds `ts` (n, ascending: `window`), kicked on the `hits` (clip
    seconds) after ts[0], each eye seeded by its row of `keys` ((k, K) whole numbers, the last its eye; default its
    index), resting as `rest` says for `look`. Between events the spring runs exactly (an underdamped oscillator about
    the rest); a pupil that reaches its eye's rim bounces back off it."""
    ts = np.asarray(ts, np.float64)
    c = np.asarray(centres, np.float64)
    rho = np.maximum(np.asarray(free, np.float64), 1e-3)
    n, k = c.shape[:2]
    keys = np.arange(k)[:, None] if keys is None else np.asarray(keys, np.int64).reshape(k, -1)
    w = 2.0 * math.pi * face["hz"]
    z = face["damping"]
    al, wd = z * w, w * math.sqrt(1.0 - z * z)
    amount, vmax = face["amount"], VMAX * w
    p = rest(face, c[0], look)
    if amount <= 0.0 or n < 2:
        return rest(face, c[-1], look)
    v = np.zeros((k, 2))
    hit_list = sorted(float(h) for h in hits if ts[0] < h <= ts[-1])
    hi, u_prev = 0, None

    def run(p, v, p0, dt):
        if dt <= 0.0:
            return p, v
        e, cs, sn = math.exp(-al * dt), math.cos(wd * dt), math.sin(wd * dt)
        y = p - p0
        y1 = e * (y * cs + (v + al * y) / wd * sn)
        v1 = e * (v * cs - (w * w * y + al * v) / wd * sn)
        return _rim(p0 + y1, v1)

    for i in range(1, n):
        t0, t1 = ts[i - 1], ts[i]
        dt = t1 - t0
        if dt <= 1e-12:
            continue
        u = (c[i] - c[i - 1]) / dt
        if u_prev is not None:
            v = v - amount * (u - u_prev) / rho[i - 1][:, None]
        u_prev = u
        p0 = rest(face, c[i - 1], look)
        at = t0
        while hi < len(hit_list) and hit_list[hi] <= t1:
            p, v = run(p, v, p0, hit_list[hi] - at)
            v = v + kicks(face, hit_list[hi], keys, look)
            at = hit_list[hi]
            hi += 1
        s = np.linalg.norm(v, axis=1, keepdims=True)
        v = np.where(s > vmax, v * (vmax / np.maximum(s, 1e-9)), v)
        p, v = run(p, v, p0, t1 - at)
    return p


def _rim(p, v):
    """Pupils past their eye's rim (|p| > 1) put back on it, their outward speed turned back (BOUNCE)."""
    r = np.linalg.norm(p, axis=1, keepdims=True)
    out = r > 1.0
    if not out.any():
        return p, v
    nrm = p / np.maximum(r, 1e-9)
    vr = np.sum(v * nrm, axis=1, keepdims=True)
    p = np.where(out, nrm, p)
    v = np.where(out & (vr > 0.0), v - (1.0 + BOUNCE) * vr * nrm, v)
    return p, v
