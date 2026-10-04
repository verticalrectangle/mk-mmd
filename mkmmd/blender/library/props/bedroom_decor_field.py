"""Field expressions for procedurally printed surfaces (pure Python + numpy, no bpy: importable from tests).

A *field* is a scalar function of the point (x, y) of a flat design: `X`, `Y` are its inputs, `+ - * /`, `abs`,
`fract`, `remap`, `ramp` ... build expression trees, and colour expressions (`col`, `cmix`, `cramp`, ...) paint with
them. The same tree has two back ends: `Evaluator` (numpy, used by the tests and for previews) and the shader-node
compiler of `bedroom_decor_nodes` (Blender), which write the same arithmetic as Math / Map Range / Mapping / Color Ramp
nodes. Expressions are interned (structurally equal ones are the same object), so shared sub-expressions become shared
nodes. Every colour is a recipe of palette slots (`col(gold=.6, base=.4, k=1.1)`), resolved against the project's
palette by `Palette`; `paper` and `ink` are two more names for the brightest and the darkest neutral slot, so a design
keeps its contrast on a light palette.

Operations (the semantics are Blender's):
    scalar    const add sub mul div(x/0 = 0) min max mod(floored) pow sqrt abs sin cos fract floor pingpong atan2
              remap(value, from0, from1, to0, to1, smooth) clamped Map Range, ramp(value in 0..1, stops) Color Ramp
              used as a lookup, xf(ex, ey, ...) = a translated / rotated / scaled point, len2, dot2
    colour    col cmix(a, b, t) cadd(a, b, t) cscreen(a, b, t) cmul(a, b, t) cscale(a, s) cramp(t, stops)
"""
import math

import numpy as np

from .cafe_colors import Colors

_TABLE = {}                      # interning table: key -> Ex


class Ex:
    """One interned expression node: kind 's' (scalar field) or 'c' (colour), op, argument expressions, constant
    attributes and the output index of a multi-output op (xf: 0 = x, 1 = y)."""
    __slots__ = ("kind", "op", "args", "attrs", "out")

    def __init__(self, kind, op, args, attrs, out):
        self.kind, self.op, self.args, self.attrs, self.out = kind, op, args, attrs, out

    def __repr__(self):
        return f"Ex({self.kind}:{self.op}{self.attrs if self.attrs else ''})"

    def __add__(self, o):
        return add(self, o)

    def __radd__(self, o):
        return add(o, self)

    def __sub__(self, o):
        return sub(self, o)

    def __rsub__(self, o):
        return sub(o, self)

    def __mul__(self, o):
        return mul(self, o)

    def __rmul__(self, o):
        return mul(o, self)

    def __truediv__(self, o):
        return div(self, o)

    def __rtruediv__(self, o):
        return div(o, self)

    def __neg__(self):
        return mul(self, -1.0)

    def __abs__(self):
        return fabs(self)


def _mk(kind, op, args=(), attrs=(), out=0):
    key = (kind, op, tuple(id(a) for a in args), attrs, out)
    e = _TABLE.get(key)
    if e is None:
        e = _TABLE[key] = Ex(kind, op, tuple(args), attrs, out)
    return e


def clear():
    """Forget every interned expression (tests)."""
    _TABLE.clear()


def c(v):
    """Scalar constant."""
    return _mk("s", "const", (), (round(float(v), 9),))


def _w(a):
    return a if isinstance(a, Ex) else c(a)


def is_const(e):
    return e.kind == "s" and e.op == "const"


def cval(e):
    return e.attrs[0]


X = _mk("s", "x")
Y = _mk("s", "y")


# ------------------------------------------------------------------ scalar operations
def _bin(op, a, b, fold, commutative=False):
    a, b = _w(a), _w(b)
    if is_const(a) and is_const(b):
        return c(fold(cval(a), cval(b)))
    if commutative and is_const(a):
        a, b = b, a
    return _mk("s", op, (a, b))


def add(a, b):
    a, b = _w(a), _w(b)
    if is_const(b) and cval(b) == 0.0:
        return a
    if is_const(a) and cval(a) == 0.0:
        return b
    return _bin("add", a, b, lambda p, q: p + q, True)


def sub(a, b):
    a, b = _w(a), _w(b)
    if is_const(b) and cval(b) == 0.0:
        return a
    return _bin("sub", a, b, lambda p, q: p - q)


def mul(a, b):
    a, b = _w(a), _w(b)
    for p, q in ((a, b), (b, a)):
        if is_const(p) and cval(p) == 1.0:
            return q
        if is_const(p) and cval(p) == 0.0:
            return c(0.0)
    return _bin("mul", a, b, lambda p, q: p * q, True)


def div(a, b):
    a, b = _w(a), _w(b)
    if is_const(b) and cval(b) == 1.0:
        return a
    return _bin("div", a, b, lambda p, q: p / q if q else 0.0)


def fmin(*xs):
    out = _w(xs[0])
    for x in xs[1:]:
        out = _bin("min", out, x, min, True)
    return out


def fmax(*xs):
    out = _w(xs[0])
    for x in xs[1:]:
        out = _bin("max", out, x, max, True)
    return out


def _tree(fn, xs):
    """Balanced reduction (keeps expression depth logarithmic)."""
    xs = list(xs)
    while len(xs) > 1:
        xs = [fn(xs[i], xs[i + 1]) if i + 1 < len(xs) else xs[i] for i in range(0, len(xs), 2)]
    return xs[0]


def union(*ms):
    """Largest of several coverage masks (balanced)."""
    return _tree(lambda p, q: fmax(p, q), ms)


def fmod(a, b):
    return _bin("mod", a, b, lambda p, q: math.fmod(p, q) % q if q else 0.0)


def fpow(a, b):
    return _bin("pow", a, b, lambda p, q: p ** q if p >= 0 else 0.0)


def _un(op, a, fold):
    a = _w(a)
    if is_const(a):
        return c(fold(cval(a)))
    return _mk("s", op, (a,))


def fabs(a):
    return _un("abs", a, abs)


def sqrt(a):
    return _un("sqrt", a, lambda v: math.sqrt(max(v, 0.0)))


def sin(a):
    return _un("sin", a, math.sin)


def cos(a):
    return _un("cos", a, math.cos)


def fract(a):
    return _un("fract", a, lambda v: v - math.floor(v))


def floor(a):
    return _un("floor", a, math.floor)


def pingpong(a, scale):
    """Triangle wave in [0, scale] with period 2 * scale."""
    a = _w(a)
    return _mk("s", "pingpong", (a,), (round(float(scale), 9),))


def atan2(y, x):
    return _bin("atan2", y, x, math.atan2)


def remap(a, f0, f1, t0=0.0, t1=1.0, smooth=False):
    """Map Range with clamping: `f0 -> t0`, `f1 -> t1` (f0 > f1 allowed); smoothstep when `smooth`."""
    return _mk("s", "remap", (_w(a),), tuple(round(float(v), 9) for v in (f0, f1, t0, t1)) + (bool(smooth),))


def ramp(a, stops, interp="LINEAR"):
    """Colour Ramp used as a lookup table: `a` (clamped to 0..1) -> the value of `stops` [(pos, value), ...]; interp
    LINEAR | CONSTANT | EASE. (Blender samples a ramp at 257 points: keep sharp structure out of it.)"""
    st = tuple((round(float(p), 9), round(float(v), 9)) for p, v in stops)
    return _mk("s", "ramp", (_w(a),), (st, interp))


def xf(ex, ey, loc=(0.0, 0.0), rot=0.0, scale=(1.0, 1.0)):
    """The point (ex, ey) in the frame placed at `loc`, turned by `rot` (rad, anticlockwise) and scaled by `scale`
    (texture-mapping convention): ((R^T (p - loc)) / scale). Returns (x, y)."""
    if not isinstance(scale, (tuple, list)):
        scale = (scale, scale)
    at = (round(float(loc[0]), 9), round(float(loc[1]), 9), round(float(rot), 9), round(float(scale[0]), 9),
          round(float(scale[1]), 9))
    ex, ey = _w(ex), _w(ey)
    return _mk("s", "xf", (ex, ey), at, 0), _mk("s", "xf", (ex, ey), at, 1)


def len2(ex, ey):
    """sqrt(ex^2 + ey^2)."""
    return _mk("s", "len2", (_w(ex), _w(ey)))


def dot2(ex, ey, nx, ny):
    """ex * nx + ey * ny with constant (nx, ny)."""
    return _mk("s", "dot2", (_w(ex), _w(ey)), (round(float(nx), 9), round(float(ny), 9)))


# ------------------------------------------------------------------ colour expressions
def col(k=1.0, hue=0.0, chroma=1.0, **w):
    """Colour recipe: a blend of palette slots (and the roles `paper`, `ink`), as `Colors.blend`."""
    return _mk("c", "const", (), (tuple(sorted((s, float(v)) for s, v in w.items())), float(k), float(hue),
                                  float(chroma)))


def _wc(a):
    return a if isinstance(a, Ex) else col(**{a: 1.0})


def cmix(a, b, t):
    """a * (1 - t) + b * t."""
    t = _w(t)
    if is_const(t) and cval(t) <= 0.0:
        return _wc(a)
    if is_const(t) and cval(t) >= 1.0:
        return _wc(b)
    return _mk("c", "mix", (_wc(a), _wc(b), t))


def cadd(a, b, t=1.0):
    """a + b * t."""
    return _mk("c", "add", (_wc(a), _wc(b), _w(t)))


def cscreen(a, b, t=1.0):
    """Screen blend of b over a with factor t."""
    return _mk("c", "screen", (_wc(a), _wc(b), _w(t)))


def cmul(a, b, t=1.0):
    """Multiply blend of b over a with factor t."""
    return _mk("c", "mul", (_wc(a), _wc(b), _w(t)))


def cscale(a, s):
    """Colour times a scalar field."""
    return _mk("c", "scale", (_wc(a), _w(s)))


def cramp(t, stops, interp="LINEAR"):
    """Colour Ramp: t (clamped to 0..1) -> colour recipes `stops` [(pos, col(...)), ...]."""
    st = tuple((round(float(p), 9), _wc(v).attrs) for p, v in stops)
    return _mk("c", "ramp", (_w(t),), (st, interp))


def over(canvas, fill, mask):
    """Paint `fill` (colour or recipe) through coverage `mask` over `canvas`."""
    return cmix(canvas, fill, mask)


# ------------------------------------------------------------------ dependency order
def topo(roots):
    """Every expression the roots depend on, dependencies first (iterative: no recursion limit)."""
    order, seen = [], set()
    stack = [(r, False) for r in reversed(list(roots))]
    while stack:
        e, done = stack.pop()
        if done:
            order.append(e)
            continue
        if id(e) in seen:
            continue
        seen.add(id(e))
        stack.append((e, True))
        for a in reversed(e.args):
            if id(a) not in seen:
                stack.append((a, False))
    return order


# ------------------------------------------------------------------ palette roles
NEUTRAL_LIGHT = ("text", "base", "surface", "overlay")
NEUTRAL_DARK = ("base", "surface", "overlay", "text")


def _lum(hexstr):
    h = hexstr.lstrip("#")
    lin = [(v / 12.92 if v <= 0.04045 else ((v + 0.055) / 1.055) ** 2.4)
           for v in (int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4))]
    return 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]


class Palette:
    """Resolves colour recipes to linear RGB. `slots` is {slot: '#hex'} (the project palette; slots it lacks come
    from the Rose Pine Dawn defaults of the cafe kit), or pass `colors=` an existing `Colors` (a Kit's, so that the
    card lists the slots used). `paper` is the brightest and `ink` the darkest of the neutral slots, whatever the
    palette."""

    def __init__(self, slots=None, colors=None):
        self.colors = colors or Colors(slots)
        p = self.colors.p
        self.roles = {"paper": max(NEUTRAL_LIGHT, key=lambda s: _lum(p[s])),
                      "ink": min(NEUTRAL_DARK, key=lambda s: _lum(p[s]))}

    def rgb(self, attrs):
        """attrs of a `col` expression -> (r, g, b) linear."""
        weights, k, hue, chroma = attrs
        w = {}
        for s, v in weights:
            s = self.roles.get(s, s)
            w[s] = w.get(s, 0.0) + v
        return tuple(self.colors.blend(k=k, hue=hue, chroma=chroma, **w)[:3])

    def blend(self, k=1.0, hue=0.0, chroma=1.0, **w):
        """Like `Colors.blend` (linear RGBA), with the roles `paper` and `ink` as weights."""
        rgb = self.rgb((tuple(sorted(w.items())), float(k), float(hue), float(chroma)))
        return (*rgb, 1.0)

    def resolved(self):
        return self.colors.resolved()


# ------------------------------------------------------------------ numpy evaluation
def _np_ramp_scalar(v, st, interp):
    pos = np.array([p for p, _ in st], dtype=np.float64)
    val = np.array([q for _, q in st], dtype=np.float64)
    return _np_ramp(v, pos, val[:, None], interp)[..., 0]


def _np_ramp(v, pos, vals, interp):
    """v: any shape; pos (n,), vals (n, ch) -> (*v.shape, ch)."""
    v = np.clip(v, 0.0, 1.0)
    if interp == "CONSTANT":
        v = np.floor(v * 256.0) / 256.0                  # Blender looks a constant ramp up in a 257-entry table
        idx = np.clip(np.searchsorted(pos, v, side="right") - 1, 0, len(pos) - 1)
        return vals[idx]
    i1 = np.clip(np.searchsorted(pos, v, side="right"), 1, len(pos) - 1)
    i0 = i1 - 1
    span = np.maximum(pos[i1] - pos[i0], 1e-12)
    t = np.clip((v - pos[i0]) / span, 0.0, 1.0)
    if interp == "EASE":
        t = t * t * (3.0 - 2.0 * t)
    return vals[i0] * (1.0 - t)[..., None] + vals[i1] * t[..., None]


class Evaluator:
    """Evaluates expressions over arrays of points (float32)."""

    def __init__(self, x, y, palette):
        self.x = np.asarray(x, dtype=np.float32)
        self.y = np.asarray(y, dtype=np.float32)
        self.pal = palette
        self.memo = {}

    def run(self, roots):
        out = []
        for e in topo(roots):
            if id(e) not in self.memo:
                self.memo[id(e)] = self._node(e)
        for r in roots:
            out.append(self.memo[id(r)])
        return out

    def _node(self, e):
        m = self.memo
        a = [m[id(x)] for x in e.args]
        op = e.op
        f32 = np.float32
        with np.errstate(all="ignore"):
            if e.kind == "c":
                return self._colour(e, a)
            if op == "const":
                return np.full_like(self.x, e.attrs[0])
            if op == "x":
                return self.x
            if op == "y":
                return self.y
            if op == "add":
                return a[0] + a[1]
            if op == "sub":
                return a[0] - a[1]
            if op == "mul":
                return a[0] * a[1]
            if op == "div":
                return np.where(a[1] != 0, a[0] / np.where(a[1] != 0, a[1], 1), 0).astype(f32)
            if op == "min":
                return np.minimum(a[0], a[1])
            if op == "max":
                return np.maximum(a[0], a[1])
            if op == "mod":
                return np.where(a[1] != 0, np.mod(a[0], np.where(a[1] != 0, a[1], 1)), 0).astype(f32)
            if op == "pow":
                return np.power(np.maximum(a[0], 0), a[1]).astype(f32)
            if op == "abs":
                return np.abs(a[0])
            if op == "sqrt":
                return np.sqrt(np.maximum(a[0], 0))
            if op == "sin":
                return np.sin(a[0])
            if op == "cos":
                return np.cos(a[0])
            if op == "fract":
                return a[0] - np.floor(a[0])
            if op == "floor":
                return np.floor(a[0])
            if op == "atan2":
                return np.arctan2(a[0], a[1])
            if op == "pingpong":
                s = e.attrs[0]
                t = (a[0] - s) / (2 * s)
                return np.abs((t - np.floor(t)) * 2 * s - s)
            if op == "remap":
                f0, f1, t0, t1, smooth = e.attrs
                if f1 == f0:
                    return np.zeros_like(a[0])
                t = np.clip((a[0] - f0) / (f1 - f0), 0.0, 1.0)
                if smooth:
                    t = t * t * (3.0 - 2.0 * t)
                return (t0 + t * (t1 - t0)).astype(f32)
            if op == "ramp":
                return _np_ramp_scalar(a[0], *e.attrs).astype(f32)
            if op == "xf":
                cx, cy, rot, sx, sy = e.attrs
                dx, dy = a[0] - cx, a[1] - cy
                cr, sr = math.cos(rot), math.sin(rot)
                if e.out == 0:
                    return ((cr * dx + sr * dy) / sx if sx else dx * 0).astype(f32)
                return ((-sr * dx + cr * dy) / sy if sy else dy * 0).astype(f32)
            if op == "len2":
                return np.sqrt(a[0] * a[0] + a[1] * a[1])
            if op == "dot2":
                return a[0] * e.attrs[0] + a[1] * e.attrs[1]
        raise ValueError(f"unknown scalar op {op!r}")

    def _colour(self, e, a):
        op = e.op
        if op == "const":
            rgb = np.array(self.pal.rgb(e.attrs), dtype=np.float32)
            return np.broadcast_to(rgb, self.x.shape + (3,)).copy()
        if op == "mix":
            t = a[2][..., None]
            return a[0] * (1 - t) + a[1] * t
        if op == "add":
            return a[0] + a[1] * a[2][..., None]
        if op == "mul":
            t = a[2][..., None]
            return a[0] * (1 - t) + a[0] * a[1] * t
        if op == "screen":
            t = a[2][..., None]
            return a[0] * (1 - t) + (1 - (1 - a[0]) * (1 - a[1])) * t
        if op == "scale":
            return a[0] * a[1][..., None]
        if op == "ramp":
            st, interp = e.attrs
            pos = np.array([p for p, _ in st], dtype=np.float64)
            vals = np.array([self.pal.rgb(att) for _, att in st], dtype=np.float64)
            return _np_ramp(a[0], pos, vals, interp).astype(np.float32)
        raise ValueError(f"unknown colour op {op!r}")


def design_grid(aspect, h, w=None):
    """Pixel-centre design coordinates of an image h rows high: x in [-aspect/2, aspect/2], y in [-0.5, 0.5], y up.
    Returns (x, y) arrays of shape (h, w) with w = round(h * aspect) by default."""
    w = w or max(1, int(round(h * aspect)))
    xs = ((np.arange(w) + 0.5) / w - 0.5) * aspect
    ys = 0.5 - (np.arange(h) + 0.5) / h
    return np.meshgrid(xs.astype(np.float32), ys.astype(np.float32))


def render(roots, aspect, h, palette, rows=96):
    """Evaluate `roots` (expressions) on an image grid in bands of `rows` rows (bounded memory) -> list of arrays
    (h, w) or (h, w, 3)."""
    xs, ys = design_grid(aspect, h)
    outs = [[] for _ in roots]
    for r0 in range(0, h, rows):
        ev = Evaluator(xs[r0:r0 + rows], ys[r0:r0 + rows], palette)
        for i, v in enumerate(ev.run(roots)):
            outs[i].append(v)
    return [np.concatenate(o, axis=0) for o in outs]


def srgb8(lin):
    """Linear RGB array -> uint8 sRGB (clipped), for previews."""
    v = np.clip(lin, 0.0, 1.0)
    s = np.where(v <= 0.0031308, v * 12.92, 1.055 * np.power(v, 1 / 2.4) - 0.055)
    return (s * 255 + 0.5).astype(np.uint8)
