"""Floating hearts (pure numpy; bpy-free, importable from tests): the outline and the puffy mesh of one heart, and the
maths of a field of them that rise and sway around a point. The Blender side is the library prop `hearts`
(mkmmd/blender/library/props/hearts.py), which turns these numbers into objects and DRIVERS on the scene frame: a render
runs no Python, so every channel of every heart is one Blender "simple expression" (no `%`, `**`, `sign`, `copysign`; at most
255 characters, the length Blender stores) that reads the field's parameters from the prop root's custom properties.
`expressions()` writes them; the numpy functions below are the same maths, readable, and what the tests compare the
expressions with.

The heart. `heart_outline` is the classic curve x = 16 sin^3 t, y = 13 cos t - 5 cos 2t - 2 cos 3t - cos 4t, resampled by
arc length and low-passed in the Fourier domain (a Gaussian exp(-(k / rounding)^2) on harmonic k) so the cleft opens
into a round notch and the tip rounds off, scaled to unit width and centred on its area centroid. It is closed,
counter-clockwise seen from the front (x right, z up), starts at the tip, mirror symmetric and star-shaped about the origin
(every ray from the origin meets it once). `heart_mesh` inflates it into a pillow: ring k is the outline scaled by
s = sin(phi_k) (near the poles blended toward a circle) at depth y = -T cos(phi_k), phi from 0 to pi, so the rim at the
equator is the outline itself, the two lobes bulge and a valley runs from the centre down into the notch. The broad
face looks along -Y (the way a character faces), x is right, z up.

The field. Time is seconds (`frame / fps`). The hearts form two plumes, one each side of the column, born at even intervals
and climbing at one speed so that neighbours keep their distance (see `Layout`). Heart i lives a cycle of
`life_i = height / (rise v_i)` seconds (`v_i` a speed factor, 0.97 on the left, 1.03 on the right):
`age_i = fmod(t + (o_i + 1000) life_i, life_i)` climbs from 0, at its birth at the bottom of the column (z = -height / 2; the
prop root is the middle), to `life_i` at the top, z = height (age_i / life_i - 1 / 2). It is born with the `pop` (a back-out
ease over `pop` seconds that overshoots 10 %), shrinks to nothing over the last `fade` seconds, and is reborn the instant it
is gone: the jump back to the bottom happens at scale 0. It drifts sideways by `sway`, turns about the vertical axis by +-`spin`
degrees, rolls by +-`tilt` and nods by +-`tilt` / 2 on slow sines of its own, so it keeps its face toward a camera
in front (the -Y side) and reads as a heart. Sideways it stays in |x| >= clear + room: its centre is
`side (cl + m + d (spread - cl - m) (1 - fan + fan u))` with `u = age / life` the share of the climb done, `cl` the `clear`
column at its height (`clear` up to `clear_top`, closing to 0 over the next 0.3 m) and
`m = hw size + sway sa` (half the heart's width at its widest turn and pop, plus its sway), so no part of it ever enters
the column |x| < `clear`, where a figure stands, below `clear_top`; `fan` lets the plumes open as they rise.

| root property | unit | meaning |
|---|---|---|
| `rise` | m/s | climb speed of a heart of speed factor 1 |
| `height` | m | the column the hearts climb, centred on the prop root |
| `spread` | m | farthest |x| of a heart's centre (from the axis) |
| `fan` | 0..1 | 0: the hearts climb in parallel; 1: they leave the edge of the clear column and open out to `spread` at the top |
| `clear` | m | half-width of the column kept empty (a figure; hearts stay outside it, up to `clear_top`) |
| `clear_top` | m | height above the root where the column ends; above it the column closes over 0.3 m and hearts arch over a head (10: never) |
| `depth` | m | half the depth (y) the hearts are spread over |
| `sway` | m | amplitude of the slow sideways drift |
| `spin`, `tilt` | deg | amplitude of the yaw swing; of the roll (the nod is half of it) |
| `pop`, `fade` | s | birth ease (overshooting) and shrink at the top |
| `amount` | 0..1 | overall scale of the field (key it to make hearts swell in or out) |
| `size_min`, `size_max` | m | heart widths, spread over the hearts |
"""
import hashlib
import math

import numpy as np

# ===================================================================================================================
# the heart
# ===================================================================================================================
ROUNDING = 9.0               # harmonics kept by the Gaussian low-pass (smaller: rounder, blobbier)
PUFF = 0.42                  # thickness of the pillow as a share of the heart's width


def _resample_closed(P, n):
    """n points evenly spaced by arc length on the closed polygon P (m, 2), starting at P[0]."""
    Q = np.vstack([P, P[:1]])
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(Q, axis=0), axis=1))])
    u = np.linspace(0.0, s[-1], n, endpoint=False)
    return np.stack([np.interp(u, s, Q[:, k]) for k in range(2)], 1)


def signed_area(P):
    """Shoelace area of the closed polygon P (m, 2): positive counter-clockwise."""
    x, z = P[:, 0], P[:, 1]
    return 0.5 * float((x * np.roll(z, -1) - np.roll(x, -1) * z).sum())


def area_centroid(P):
    x, z = P[:, 0], P[:, 1]
    x1, z1 = np.roll(x, -1), np.roll(z, -1)
    cr = x * z1 - x1 * z
    return np.array([((x + x1) * cr).sum(), ((z + z1) * cr).sum()]) / (3.0 * cr.sum())


def heart_outline(n=96, rounding=ROUNDING):
    """The heart's outline: (n, 2) points (x, z), unit width, centred on its area centroid (the origin), counter-clockwise
    seen from the front starting at the tip, evenly spaced by arc length and exactly mirror symmetric (`n` is even: point
    n / 2 is the bottom of the notch and point n - j mirrors point j)."""
    if n % 2 or n < 16:
        raise ValueError(f"heart_outline: n must be even and at least 16, got {n}")
    t = np.linspace(0.0, 2.0 * math.pi, 4096, endpoint=False)
    x = 16.0 * np.sin(t) ** 3
    z = 13.0 * np.cos(t) - 5.0 * np.cos(2 * t) - 2.0 * np.cos(3 * t) - np.cos(4 * t)
    P = _resample_closed(np.stack([-x, z], 1), 1024)          # from the cleft, down the left side: counter-clockwise
    F = np.fft.fft(P[:, 0] + 1j * P[:, 1])
    k = np.fft.fftfreq(len(F), 1.0 / len(F))
    F = np.fft.ifft(F * np.exp(-(k / float(rounding)) ** 2))
    P = np.stack([F.real, F.imag], 1)
    P = np.roll(P, -int(np.argmin(P[:, 1])), axis=0)          # start at the tip; counter-clockwise goes right and up
    Q = np.vstack([P, P[:1]])
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(Q, axis=0), axis=1))])
    u = np.linspace(0.0, 0.5 * s[-1], n // 2 + 1)             # by symmetry the notch is half the perimeter from the tip
    R = np.stack([np.interp(u, s, Q[:, 0]), np.interp(u, s, Q[:, 1])], 1)          # the right half, tip .. notch
    R[0, 0] = R[-1, 0] = 0.0
    out = np.vstack([R, R[-2:0:-1] * (-1.0, 1.0)])
    out /= 2.0 * out[:, 0].max()                              # unit width
    out[:, 1] -= area_centroid(out)[1]                        # (the x of the centroid is 0 by symmetry)
    return out


def _smoothstep(a, b, x):
    t = np.clip((np.asarray(x, float) - a) / (b - a), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def heart_mesh(puff=PUFF, n=96, rings=20, rounding=ROUNDING):
    """The heart as a closed pillow (`core.shell.Mesh`, outward normals, smooth shading meant): unit width, thickness `puff`
    (along y, the front pole at y = -puff / 2 faces -Y), centred on the origin. `n` points round, `rings` (even) steps from
    pole to pole; the rim is the outline at the equator."""
    from . import shell as S
    if rings % 2 or rings < 6:
        raise ValueError(f"heart_mesh: rings must be even and at least 6, got {rings}")
    O = heart_outline(n, rounding)
    r = np.linalg.norm(O, axis=1)
    U = O / r[:, None]
    rbar = float(r.mean())
    T = 0.5 * float(puff)
    V, ring_id = [], {}
    for k in range(1, rings):
        phi = math.pi * k / rings
        s = math.sin(phi)
        w = float(_smoothstep(0.0, 0.6, s))                   # near the poles the rings are circles, not small hearts
        P = s * (w * O + (1.0 - w) * rbar * U)
        ring_id[k] = len(V) + np.arange(n)
        V.extend(np.column_stack([P[:, 0], np.full(n, -T * math.cos(phi)), P[:, 1]]))
    front, back = len(V), len(V) + 1
    V.extend([(0.0, -T, 0.0), (0.0, T, 0.0)])
    j = np.arange(n)
    j2 = (j + 1) % n
    Q = [np.stack([ring_id[k][j], ring_id[k + 1][j], ring_id[k + 1][j2], ring_id[k][j2]], 1) for k in range(1, rings - 1)]
    Tri = [np.stack([np.full(n, front), ring_id[1][j], ring_id[1][j2]], 1),
           np.stack([np.full(n, back), ring_id[rings - 1][j2], ring_id[rings - 1][j]], 1)]
    m = S.Mesh(np.array(V), np.concatenate(Q), np.concatenate(Tri))
    return m.flip() if m.volume() < 0 else m


def halfwidth(spin=25.0, tilt=10.0, puff=PUFF, pop_overshoot=1.1, mesh=None):
    """Half the widest sideways extent (x) of a unit heart over every turn the field gives it (yaw +-spin, roll +-tilt, pitch
    +-tilt / 2, degrees) at the peak of its pop: what `hw` in the field's room is. Measured on the mesh."""
    from . import shell as S
    m = mesh if mesh is not None else heart_mesh(puff)
    best = 0.0
    for yaw in np.linspace(-spin, spin, 5):
        for roll in np.linspace(-tilt, tilt, 5):
            for pitch in np.linspace(-0.5 * tilt, 0.5 * tilt, 3):
                R = S.rot_matrix(pitch, roll, yaw)
                best = max(best, float(np.abs(m.V @ R.T)[:, 0].max()))
    return best * pop_overshoot


# ===================================================================================================================
# the field
# ===================================================================================================================
PARAMS = {                   # root custom property: (default, unit, doc)
    "rise": (0.25, "m/s", "climb speed of a heart of speed factor 1"),
    "height": (1.6, "m", "the column the hearts climb, centred on the prop root"),
    "spread": (0.62, "m", "farthest |x| of a heart's centre"),
    "fan": (0.5, "", "0: parallel climb; 1: the plumes open out from the clear column's edge to `spread` at the top"),
    "clear": (0.0, "m", "half-width of the column kept empty (a figure stands there)"),
    "clear_top": (10.0, "m", "height above the root where the clear column ends (it opens over 0.3 m: hearts arch over a head)"),
    "depth": (0.25, "m", "half the depth (y) the hearts are spread over"),
    "sway": (0.06, "m", "amplitude of the slow sideways drift"),
    "spin": (25.0, "deg", "amplitude of the yaw swing"),
    "tilt": (10.0, "deg", "amplitude of the roll (the nod is half of it)"),
    "pop": (0.4, "s", "birth ease (overshoots 10 %)"),
    "fade": (0.6, "s", "shrink to nothing at the top"),
    "amount": (1.0, "", "overall scale of the field (0 hides it)"),
    "size_min": (0.06, "m", "smallest heart width"),
    "size_max": (0.16, "m", "largest heart width"),
}
DEFAULTS = {k: v[0] for k, v in PARAMS.items()}
OVERSHOOT = 1.1              # the pop's back-out ease peaks 10 % over
BACK = 1.70158               # its constant: ease(a) = 1 + (BACK + 1) (a - 1)^3 + BACK (a - 1)^2
WRAP = 1000.0                # whole cycles added to the clock so that fmod never sees a negative time
MIN_SCALE = 0.0005           # a heart is never exactly scale 0 (a singular matrix upsets everything that inverts one)
CLEAR_SOFT = 0.3             # metres over which the clear column closes above `clear_top`
MAX_EXPR = 255               # what Blender stores of a driver expression
JITTER = 0.24                # births within a plume are jittered by +-JITTER / 2 of their even spacing
NDIGITS = 5                  # decimals of every constant in an expression (and of a layout: the maths is the same)


def _num(slots, key, default):
    try:
        return float(slots.get(key, default))
    except (TypeError, ValueError):
        raise ValueError(f"hearts: {key} = {slots[key]!r} is not a number")


def options(slots):
    """(count, seed, params {root property: value}, puff, colour spec, glow) from the `slots` of the [[prop]], checked: a value
    that is not a number or is out of range is a ValueError that says what to change."""
    slots = slots or {}
    P = {k: _num(slots, k, d) for k, d in DEFAULTS.items()}
    size = slots.get("size")
    if size is not None:
        if not isinstance(size, (list, tuple)) or len(size) != 2:
            raise ValueError(f"hearts: size = {size!r}: give [smallest, largest] width in metres")
        P["size_min"], P["size_max"] = float(size[0]), float(size[1])
    count, seed = int(_num(slots, "count", 16)), int(_num(slots, "seed", 1))
    if count < 1:
        raise ValueError("hearts: count must be at least 1")
    if not 0.0 < P["size_min"] <= P["size_max"]:
        raise ValueError(f"hearts: size must satisfy 0 < smallest <= largest, got {[P['size_min'], P['size_max']]}")
    for k in ("rise", "height", "spread", "pop", "fade"):
        if P[k] <= 0:
            raise ValueError(f"hearts: {k} must be positive, got {P[k]}")
    for k in ("clear", "depth", "sway", "spin", "tilt", "amount", "fan"):
        if P[k] < 0:
            raise ValueError(f"hearts: {k} must not be negative, got {P[k]}")
    if P["fan"] > 1.0:
        raise ValueError(f"hearts: fan is 0..1, got {P['fan']}")
    puff = _num(slots, "puff", PUFF)
    if not 0.1 <= puff <= 1.0:
        raise ValueError(f"hearts: puff is the thickness as a share of the width, 0.1 .. 1, got {puff}")
    return count, seed, P, puff, str(slots.get("heart", "love")), _num(slots, "glow", 0.3)


def _alphas(d):
    """Kronecker constants of the d-dimensional generalised golden ratio: a sequence frac(a + i alpha) covers the unit cube
    evenly, so a small field has no clumps and no gaps."""
    x = 2.0
    for _ in range(80):
        x = (1.0 + x) ** (1.0 / (d + 1.0))
    return [x ** -(k + 1) for k in range(d)]


def _unit(seed, *key):
    """A deterministic number in [0, 1) from a seed and a key (the same on every platform and numpy version)."""
    h = hashlib.blake2b(("/".join(str(k) for k in (seed,) + key)).encode(), digest_size=8).digest()
    return int.from_bytes(h, "little") / 2.0 ** 64


class Layout:
    """The per-heart constants of a field (arrays of length `count`), fixed by `count` and `seed`:
    o phase offset (cycles), v speed factor, side (+1 / -1), d lateral anchor 0..1, e depth anchor -1..1, r size 0..1, sa sway
    share, and the slow sines of the sway (x, y), yaw, roll, pitch: angular frequency w and phase ph, amplitude share a.

    The hearts form two plumes, one each side of the column (heart i is on the right when i is even). Within a plume they are
    born at even intervals of the climb (a phase step of 1 / its number of hearts, jittered by 12 %) and climb at ONE speed,
    so the gap between neighbours never changes; their lateral anchors and sizes step by golden-ratio increments, so
    neighbours differ in both. That is what keeps them apart (a gap of the climb divided by the hearts of a plume, never
    under about 0.75 of it, and always a lateral step as well): a flat silhouette does not merge them into blobs. The two
    plumes differ a little in speed (+-3 %) and in phase, and never meet, the column between them being kept clear."""
    FIELDS = ("o", "v", "side", "d", "e", "r", "sa", "wx", "px", "wy", "py", "wyaw", "pyaw", "ayaw", "wroll", "proll", "broll",
              "wpitch", "ppitch", "cpitch")

    def __init__(self, count, seed=0):
        if count < 1:
            raise ValueError("a field needs at least one heart")
        n = self.count = int(count)
        self.seed = seed
        self.side = np.where(np.arange(n) % 2 == 0, 1.0, -1.0)
        k = np.arange(n) // 2                                                # a heart's number within its plume
        per = np.where(self.side > 0, (n + 1) // 2, n // 2).astype(float)    # the hearts of its plume
        al = _alphas(3)
        tag = lambda name, i: _unit(seed, name, int(self.side[i]))             # noqa: E731
        self.o = np.array([((k[i] + (_unit(seed, "oj", i) - 0.5) * JITTER) / per[i] + tag("o0", i)) % 1.0 for i in range(n)])
        self.d = np.array([(tag("d0", i) + k[i] * al[0]) % 1.0 for i in range(n)])
        self.e = np.array([2.0 * ((tag("e0", i) + k[i] * al[1]) % 1.0) - 1.0 for i in range(n)])
        self.r = np.array([(tag("r0", i) + k[i] * al[2]) % 1.0 for i in range(n)])
        self.v = 1.0 + 0.03 * self.side
        u = lambda tag, lo, hi: np.array([lo + (hi - lo) * _unit(seed, tag, i) for i in range(n)])   # noqa: E731
        two_pi = 2.0 * math.pi
        self.sa = u("sa", 0.6, 1.0)
        self.wx, self.px = u("wx", 0.20, 0.42) * two_pi, u("px", 0, 1) * two_pi
        self.wy, self.py = u("wy", 0.18, 0.36) * two_pi, u("py", 0, 1) * two_pi
        self.wyaw, self.pyaw, self.ayaw = u("wyaw", 0.22, 0.45) * two_pi, u("pyaw", 0, 1) * two_pi, u("ayaw", 0.6, 1.0)
        self.wroll, self.proll, self.broll = u("wroll", 0.2, 0.4) * two_pi, u("proll", 0, 1) * two_pi, u("broll", 0.6, 1.0)
        self.wpitch, self.ppitch, self.cpitch = u("wpitch", 0.2, 0.4) * two_pi, u("ppitch", 0, 1) * two_pi, u("cpitch", 0.6, 1.0)
        for f in self.FIELDS:                                  # the constants as the expressions print them (5 decimals)
            setattr(self, f, np.round(getattr(self, f), NDIGITS))


def life(P, L):
    """Seconds one climb takes, per heart."""
    return P["height"] / (np.maximum(P["rise"], 1e-3) * L.v)


def age(t, P, L):
    """Seconds since each heart's birth at time(s) t: (T, N) for an array t of shape (T,)."""
    lf = life(P, L)
    return np.fmod(np.asarray(t, float)[..., None] + (L.o + WRAP) * lf, lf)


def size(P, L):
    return P["size_min"] + (P["size_max"] - P["size_min"]) * L.r


def pop_in(a):
    """The birth ease: 0 -> 1 with a 10 % overshoot, for a = age / pop clamped to 0..1."""
    c = np.minimum(np.asarray(a, float), 1.0) - 1.0
    return 1.0 + c * c * (BACK + (BACK + 1.0) * c)


def room(P, L, hw):
    """The distance from the axis a heart's centre keeps beyond `clear`: half its widest extent plus its sway."""
    return hw * size(P, L) + P["sway"] * L.sa


def pose(t, P, L, hw, fps=None):
    """Where every heart is at time(s) t (seconds): a dict of arrays (T, N): x, y, z (m, in the prop root's frame),
    rx, ry, rz (radians, the euler angles of the object) and s (the uniform scale)."""
    hw = round(float(hw), NDIGITS)
    t = np.asarray(t, float)[..., None]
    a = age(t[..., 0], P, L)
    lf = life(P, L)
    m = room(P, L, hw)
    u = a / lf
    z = P["height"] * (u - 0.5)
    cl = P["clear"] * (1.0 - _smoothstep(P["clear_top"], P["clear_top"] + CLEAR_SOFT, z))
    x = L.side * (cl + m + L.d * np.maximum(P["spread"] - cl - m, 0.0) * (1.0 - P["fan"] + P["fan"] * u)) \
        + P["sway"] * L.sa * np.sin(L.wx * t + L.px)
    y = P["depth"] * L.e + 0.5 * P["sway"] * L.sa * np.sin(L.wy * t + L.py)
    rz = math.radians(1.0) * P["spin"] * L.ayaw * np.sin(L.wyaw * t + L.pyaw)
    ry = math.radians(1.0) * P["tilt"] * L.broll * np.sin(L.wroll * t + L.proll)
    rx = 0.5 * math.radians(1.0) * P["tilt"] * L.cpitch * np.sin(L.wpitch * t + L.ppitch)
    grow = pop_in(a / max(P["pop"], 1e-3)) * _smoothstep(0.0, 1.0, (lf - a) / max(P["fade"], 1e-3))
    s = np.maximum(size(P, L) * P["amount"] * grow, MIN_SCALE)
    return {"x": x, "y": y, "z": z, "rx": rx, "ry": ry, "rz": rz, "s": s, "age": a, "u": u, "cl": cl}


# ===================================================================================================================
# the driver expressions
# ===================================================================================================================
def _n(x):
    """A number as short plain decimal text (no exponent: Blender's expression parser reads plain decimals)."""
    s = f"{float(x):.{NDIGITS}f}".rstrip("0").rstrip(".")
    return "0" if s in ("-0", "") else s


CHANNELS = ("age", "u", "cl", "pin", "x", "y", "z", "rx", "ry", "rz", "s")   # in dependency order: the helpers first
HELPERS = ("age", "u", "cl", "pin")                                 # custom properties of the heart object, driven too
FUNCS = frozenset({"fmod", "min", "max", "clamp", "smoothstep", "sin", "cos", "abs", "radians", "sqrt"})


def expressions(L, i, fps, hw):
    """The driver expressions of heart i: {channel: text} for CHANNELS. They read the root's custom properties by their
    names in PARAMS, `frame`, and the heart's own helper properties `age` (seconds), `u` (share of the climb), `cl` (the clear
    column's half-width at its height) and `pin` (the birth ease): `age` is read by `u`, `pin` and `s`; `u` by `cl`, `x`, `z`;
    `cl` by `x`; `pin` by `s`."""
    hw = round(float(hw), NDIGITS)
    f, c = _n(fps), lambda k: _n(getattr(L, k)[i])                       # noqa: E731
    sz = f"(size_min+(size_max-size_min)*{c('r')})"
    lf = f"(height/(max(rise,0.001)*{c('v')}))"
    m = f"({_n(hw)}*{sz}+sway*{c('sa')})"
    e = {
        "age": f"fmod(frame/{f}+({c('o')}+{_n(WRAP)})*{lf},{lf})",
        "pin": f"1+C*C*({_n(BACK)}+{_n(BACK + 1)}*C)".replace("C", "(min(age/max(pop,0.001),1)-1)"),
        "u": f"age*max(rise,0.001)*{c('v')}/max(height,0.001)",
        "cl": f"clear*(1-smoothstep(clear_top,clear_top+{_n(CLEAR_SOFT)},height*(u-0.5)))",
        "x": f"{'' if L.side[i] > 0 else '-'}(cl+{m}+{c('d')}*max(spread-cl-{m},0)*(1-fan+fan*u))"
             f"+sway*{c('sa')}*sin({c('wx')}*frame/{f}+{c('px')})",
        "y": f"depth*{c('e')}+0.5*sway*{c('sa')}*sin({c('wy')}*frame/{f}+{c('py')})",
        "z": "height*(u-0.5)",
        "rz": f"radians(spin)*{c('ayaw')}*sin({c('wyaw')}*frame/{f}+{c('pyaw')})",
        "ry": f"radians(tilt)*{c('broll')}*sin({c('wroll')}*frame/{f}+{c('proll')})",
        "rx": f"0.5*radians(tilt)*{c('cpitch')}*sin({c('wpitch')}*frame/{f}+{c('ppitch')})",
        "s": f"max({sz}*amount*pin*smoothstep(0,1,({lf}-age)/max(fade,0.001)),{_n(MIN_SCALE)})",
    }
    return {k: e[k] for k in CHANNELS}


# ============================================================================================ library:heart (one pounding heart)
def beat_phase(per_beats=None):
    """The song clock's phase, 0..1 over one beat (or over `per_beats` beats, the name of a root property), as a driver
    expression of the frame and the heart root's properties f0 (frame of clip second 0), fps, start (a clip second on a beat)
    and bpm."""
    div = f"/{per_beats}" if per_beats else ""
    return f"fmod(((frame-f0)/fps-start)*bpm/60{div}+1000,1)"


def heartbeat_expressions(i, n, turn):
    """Driver expressions of library:heart: the big heart's scale (`big`: a bump on the beat and one 60 % of it a quarter
    beat later, by `beat`), and little heart `i` of `n` in the burst: x, z (it flies out along its spoke, `turn` degrees
    round, easing out over `every` beats to `reach`) and its scale (`bk`: it pops in, then shrinks as it flies)."""
    P, B = beat_phase(), beat_phase("every")
    out = {"big": f"size*(1+beat*(max(0,1-abs({P}-0.03)/0.09)+0.6*max(0,1-abs({P}-0.27)/0.09)))"}
    if n:
        a = math.radians(turn) + 2.0 * math.pi * i / n
        cx, cz = round(math.sin(a), 5), round(math.cos(a), 5)
        out["x"] = f"{cx}*reach*(1-pow(1-{B},3))"
        out["z"] = f"{cz}*reach*(1-pow(1-{B},3))"
        out["s"] = f"max({MIN_SCALE},bk*min(1,{B}*12)*(1-{B}))"
    return out
