"""A leaf lying on the head (part `leaf`, numpy only): one broad leaf on the top of the head, rigid on the head bone.

Its blade is `length` long from its base to its tip and `width` across at its widest, `widest` of the way from the
base: a rounded base, a pointed tip; a short stem `stem` long comes out of its base, `stem_thick` thick, rising off what
it lies on. A round `[leaf.bite]` is taken out of one edge: `side` "left" or "right" (looking from the stem to the
tip), its middle `at` that share of the length from the base, `radius`, cutting `depth` metres into the blade.

It lies (`[leaf.lay]`) with its middle `at` [x, y] metres from the head's centre, its tip toward `toward` degrees round
from her front toward her left (-35: her front right; the stem points the other way), over what is under it: the head's
skin shell and every part built before the leaf but the body (list it after the hair and it lies on the hair and
between her ears; without hair, on the bald head). A leaf bridges the dips between locks: what it follows is that
surface closed by a round `span` metres in radius (morphological closing), and it lies `lift` above it; nowhere is its
underside nearer than `clear` to what is under it. Its tip curls up `curl` metres over the last `curl_from` of its
length, its halves rise `fold` at their edges (a soft V along the midrib). Then, so it reads as a leaf from in front of
her, it is turned `tilt` degrees toward her front (its face from up toward her front, about her left-right axis
through its middle) and set down again, `lift` over what is under it where it comes nearest: on a flat crown its stem
end stands off the hair. With `over` it slides along its tip's way from `at` until its tip reaches `over` metres past
the front edge of what it lies on (where that, closed, first falls away more steeply than EDGE degrees: over the
fringe). It is `thick` thick, its edge rolled round over `roll` metres, a closed solid seen from both sides.

Picture (`[colors.leaf]`): base green, shadow toward the edges and in the bite, light along the midrib, the midrib and
`veins` pairs of veins drawn in `vein`; the underside paler (light over base). Its outline is a darker shade of its
shadow. Material 葉っぱ. `info` holds its measures: the blade's and the stem's length, the width, its middle, the tip's
direction, how far its underside stands off what is under it (least, median, most), the tip's curl, how far the tip
reaches past the front edge (`past_edge`, the edge's point `edge`), the face's turn to the front; landmark `leaf`."""
import colorsys

import numpy as np
from PIL import Image, ImageDraw

from .. import spec as SP
from ..build import builder
from ..part import Part
from . import outfit_geo as G
from . import outfit_wear as WR
from .mob_cap import under

DEFAULTS = {"length": 0.105, "width": 0.060, "widest": 0.42, "stem": 0.010, "stem_thick": 0.0016,
            "thick": 0.0012, "roll": 0.0025, "veins": 4,
            "bite": {"side": "left", "at": 0.58, "radius": 0.0065, "depth": 0.0065},
            "lay": {"at": [0.0, -0.004], "toward": -35.0, "tilt": 0.0, "lift": 0.003, "clear": 0.002, "span": 0.03,
                    "curl": 0.0065, "curl_from": 0.3, "fold": 0.002},
            "rows": 64, "cols": 25, "bone": "頭"}
COLORS = {"base": "#4fae3a", "shadow": "#2f7d2a", "light": "#8fd45a", "vein": "#2a6b2a"}
MAT = "葉っぱ"
STEP = 0.0015                   # metres between the samples of the surface under the leaf
EDGE = 35.0                     # degrees: where what it lies on falls away more steeply, its front edge is


# ---------------------------------------------------------------- the outline
def half_widths(u, c):
    """Half widths (left, right) (len(u),) of the blade at shares `u` of its length from the base: an ovate leaf
    widest at `widest`, its base round, its tip pointed, the bite taken out of one side."""
    w = float(c["widest"])
    p, q = 0.62, 0.62 * (1.0 - w) / w                  # widest where u^p (1 - u)^q peaks
    f = np.clip(u, 0.0, 1.0) ** p * np.clip(1.0 - u, 0.0, 1.0) ** q
    h = 0.5 * float(c["width"]) * f / (w ** p * (1.0 - w) ** q)
    b = c["bite"]
    L, rad, depth = float(c["length"]), float(b["radius"]), float(b["depth"])
    hb = float(np.interp(float(b["at"]), u, h)) if len(u) > 1 else float(h[0])
    cv = hb + rad - depth                              # the bite's circle: its centre outside the edge
    dx = (u - float(b["at"])) * L
    inside = np.abs(dx) < rad
    cut = np.where(inside, cv - np.sqrt(np.maximum(rad * rad - dx * dx, 0.0)), np.inf)
    bitten = np.maximum(np.minimum(h, cut), 0.0)
    return (bitten, h) if str(b["side"]).lower() == "left" else (h, bitten)


# ---------------------------------------------------------------- what it lies on
def heights(surf, x0, y0, n, step):
    """The highest z of the surface `surf` (outfit_fit.Skin) over each cell of an n x n grid (x0 + i step, y0 + j step):
    (n, n) array (i along x), -inf where nothing is under it. Its triangles are sampled finely enough for the grid."""
    H = np.full((n, n), -np.inf)
    T = surf.verts[surf.tris]
    lo, hi = np.array([x0, y0]) - step, np.array([x0, y0]) + (n + 1) * step
    keep = ((T[:, :, :2].max(1) >= lo) & (T[:, :, :2].min(1) <= hi)).all(1)
    T = T[keep]
    if not len(T):
        return H
    edge = np.max(np.linalg.norm(T[:, :, :2] - np.roll(T[:, :, :2], 1, axis=1), axis=2), axis=1)
    k_of = np.clip(np.ceil(edge / (0.5 * step)).astype(int), 1, 64)
    for k in np.unique(k_of):
        tri = T[k_of == k]
        a, b = np.meshgrid(np.arange(k + 1), np.arange(k + 1), indexing="ij")
        ok = a + b <= k
        w1, w2 = a[ok] / k, b[ok] / k
        P = (tri[:, None, 0] * (1.0 - w1 - w2)[None, :, None] + tri[:, None, 1] * w1[None, :, None]
             + tri[:, None, 2] * w2[None, :, None]).reshape(-1, 3)
        i = np.floor((P[:, 0] - x0) / step + 0.5).astype(int)
        j = np.floor((P[:, 1] - y0) / step + 0.5).astype(int)
        m = (i >= 0) & (i < n) & (j >= 0) & (j < n)
        np.maximum.at(H, (i[m], j[m]), P[m, 2])
    return H


def closing(H, step, span):
    """The morphological closing of the height field H by a paraboloid of curvature radius `span`: the dips narrower
    than about `span` filled (a stiff sheet bridging them), never below H."""
    k = max(int(span / step), 1)
    off = np.arange(-k, k + 1)
    pad = np.pad(np.where(np.isfinite(H), H, np.nan), k, constant_values=np.nan)
    n0, n1 = H.shape

    def sweep(src, sign):
        out = np.full(H.shape, -np.inf if sign > 0 else np.inf)
        for di in off:
            for dj in off:
                d2 = (di * di + dj * dj) * step * step
                if d2 > (k * step) ** 2:
                    continue
                sh = src[k + di:k + di + n0, k + dj:k + dj + n1] - sign * d2 / (2.0 * span)
                sh = np.where(np.isnan(sh), -np.inf if sign > 0 else np.inf, sh)
                out = np.maximum(out, sh) if sign > 0 else np.minimum(out, sh)
        return out
    D = sweep(pad, 1.0)
    E = sweep(np.pad(np.where(np.isfinite(D), D, np.nan), k, constant_values=np.nan), -1.0)
    return np.where(np.isfinite(E), np.maximum(E, H), H)


class Ground:
    """The surface the leaf follows: heights over a grid round its place (`closing` of what is under it), with
    bilinear lookups of the height and its gradient."""

    def __init__(self, surf, centre, reach, step, span):
        self.n = int(2.0 * reach / step) + 1
        self.x0, self.y0, self.step = centre[0] - reach, centre[1] - reach, step
        self.raw = heights(surf, self.x0, self.y0, self.n, step)
        if not np.isfinite(self.raw).any():
            raise ValueError("leaf: nothing under the leaf to lie on")
        fill = np.where(np.isfinite(self.raw), self.raw, np.nanmin(np.where(np.isfinite(self.raw), self.raw, np.nan)))
        self.H = closing(fill, step, span)
        g = self.H.copy()
        for _ in range(3):                              # a soft surface: the closing's creases eased
            g[1:-1, 1:-1] = (4.0 * g[1:-1, 1:-1] + g[2:, 1:-1] + g[:-2, 1:-1] + g[1:-1, 2:] + g[1:-1, :-2]) / 8.0
        self.H = np.maximum(g, self.H)

    def z(self, xy, field=None):
        F = self.H if field is None else field
        fi = np.clip((xy[..., 0] - self.x0) / self.step, 0.0, self.n - 1.001)
        fj = np.clip((xy[..., 1] - self.y0) / self.step, 0.0, self.n - 1.001)
        i, j = np.floor(fi).astype(int), np.floor(fj).astype(int)
        a, b = fi - i, fj - j
        return (F[i, j] * (1 - a) * (1 - b) + F[i + 1, j] * a * (1 - b) + F[i, j + 1] * (1 - a) * b
                + F[i + 1, j + 1] * a * b)

    def normal(self, xy, eps=0.002):
        dx = (self.z(xy + [eps, 0.0]) - self.z(xy - [eps, 0.0])) / (2.0 * eps)
        dy = (self.z(xy + [0.0, eps]) - self.z(xy - [0.0, eps])) / (2.0 * eps)
        return G.unit(np.stack([-dx, -dy, np.ones_like(dx)], -1))


def arc_positions(ground, origin, direction, s, step=0.0005):
    """Horizontal points along `direction` from `origin` whose distance measured over the ground is `s` (signed)."""
    s = np.asarray(s, float)
    reach = float(np.abs(s).max()) * 1.2 + 0.002
    t = np.arange(-reach, reach + step, step)
    xy = origin[None] + t[:, None] * direction[None]
    z = ground.z(xy)
    seg = np.hypot(np.diff(t), np.diff(z))
    arc = np.concatenate([[0.0], np.cumsum(seg)])
    arc -= np.interp(0.0, t, arc)
    return origin[None] + np.interp(s, arc, t)[:, None] * direction[None]


# ---------------------------------------------------------------- the solid
def blade(ground, c, centre, m, left):
    """The blade's top and bottom surfaces (rows, cols, 3), its rows from the base to the tip and its columns from its
    right edge to its left, and their texture coordinates (rows, cols, 2) in the leaf's own plane."""
    lay = c["lay"]
    L, K, M = float(c["length"]), int(c["rows"]), int(c["cols"]) | 1
    u = 0.5 - 0.5 * np.cos(np.linspace(0.0, np.pi, K))          # rows closer at the base and the tip
    hl, hr = half_widths(u, c)
    s = np.linspace(-1.0, 1.0, M)
    mid = arc_positions(ground, centre, m, (u - 0.5) * L)
    xy = np.empty((K, M, 2))
    across = np.where(s[None] >= 0.0, s[None] * hl[:, None], s[None] * hr[:, None])        # metres toward its left
    for k in range(K):
        xy[k] = arc_positions(ground, mid[k], left, across[k])
    base = ground.z(xy) + float(lay["lift"])
    base = base + float(lay["fold"]) * np.abs(across) / max(0.5 * float(c["width"]), 1e-6)
    cf = float(lay["curl_from"])
    t = np.clip((u - (1.0 - cf)) / max(cf, 1e-6), 0.0, 1.0)
    base = base + float(lay["curl"]) * (t * t)[:, None]
    P = np.concatenate([xy, base[..., None]], -1)
    n = ground.normal(xy)
    edge = np.minimum(np.where(s[None] >= 0, hl[:, None], hr[:, None]) * (1.0 - np.abs(s[None])),
                      np.minimum(u, 1.0 - u)[:, None] * L)
    x = np.clip(edge / max(float(c["roll"]), 1e-6), 0.0, 1.0)
    half = 0.5 * float(c["thick"]) * np.sqrt(1.0 - (1.0 - x) ** 2)
    top, bottom = P + half[..., None] * n, P - half[..., None] * n
    uv = np.stack([np.broadcast_to(u[:, None], (K, M)), 0.5 + across / float(c["width"])], -1)
    return top, bottom, uv


def lift_clear(top, bottom, surf, clear):
    """Raise the blade where its underside comes nearer than `clear` to what is under it (a smooth raise, the shape
    kept), again until it holds (straight up gains less than the raise where that slopes); returns (top, bottom). The
    blade lies over the closed surface of what is under it, so the nearest distance needs no sign (locks of hair are
    sheets whose sides a sign cannot tell)."""
    up = np.array([0.0, 0.0, 1.0])
    for _ in range(4):
        d = np.abs(surf.closest(bottom.reshape(-1, 3))[0]).reshape(bottom.shape[:2])
        need = np.maximum(clear - d, 0.0)
        if need.max() <= 1e-6:
            break
        for _ in range(6):                              # spread the raise so the blade stays smooth
            need[1:-1, 1:-1] = np.maximum(need[1:-1, 1:-1], 0.25 * (need[2:, 1:-1] + need[:-2, 1:-1]
                                                                       + need[1:-1, 2:] + need[1:-1, :-2]))
        top, bottom = top + 1.5 * need[..., None] * up, bottom + 1.5 * need[..., None] * up
    return top, bottom


def front_edge(ground, start, way, reach, step=0.001, run=0.005):
    """How far along the level unit `way` from `start` (x, y) the front edge of what the leaf lies on is: where the
    closed surface first falls away more steeply than EDGE degrees over `run` metres (`reach` when it never does)."""
    s = np.arange(0.0, reach, step)
    z = ground.z(start[None] + s[:, None] * way[None])
    k = max(int(round(run / step)), 1)
    steep = np.flatnonzero(z[:-k] - z[k:] > np.tan(np.radians(EDGE)) * run)
    return float(s[steep[0]]) if len(steep) else float(reach)


def turned(top, bottom, ground, tilt, lift):
    """The blade turned `tilt` degrees toward her front (its face from up toward -y, about her left-right axis through
    its middle) and set down again, `lift` over what is under it where it comes nearest (straight down)."""
    t = np.radians(tilt)
    R = np.array([[1.0, 0.0, 0.0], [0.0, np.cos(t), -np.sin(t)], [0.0, np.sin(t), np.cos(t)]])
    pivot = 0.5 * (top + bottom)[top.shape[0] // 2, top.shape[1] // 2]
    top, bottom = (pivot + (X - pivot) @ R.T for X in (top, bottom))
    raw = np.where(np.isfinite(ground.raw), ground.raw, -1.0)
    down = float((bottom[..., 2] - ground.z(bottom[..., :2], raw)).min()) - lift
    return top - [0.0, 0.0, down], bottom - [0.0, 0.0, down]


def stem_path(ground, root, back, c, n=12):
    """The stem's centreline: from the blade's base straight back along `back`, rising off the ground as it goes."""
    L, lift = float(c["stem"]), float(c["lay"]["lift"])
    t = np.linspace(0.0, 1.0, n)
    xy = root[None, :2] + (t * L)[:, None] * back[None]
    z0 = root[2] - ground.z(root[None, :2])[0]
    z = ground.z(xy) + z0 + (lift + 0.4 * L) * t * t
    return np.concatenate([xy, z[:, None]], -1)


# ---------------------------------------------------------------- picture and material
def picture(col, c, w=1024, h=512):
    """The leaf's picture: u (x) along the blade from its base, v (y, up) across it: the top side in the upper half
    (v 0.5 .. 1 is the blade's width), the underside in the lower half, the same layout. Base green, in shadow toward
    the edges (the bite's too), lighter along the midrib; the midrib and the veins (curving toward the tip, ending short
    of the edge) drawn in `vein`, thinning as they go. The underside is paler and its lines fainter."""
    S = 2
    W, Hh = w * S, h // 2 * S
    width = float(c["width"])
    u = (np.arange(W) + 0.5) / W
    across = (0.5 - (np.arange(Hh) + 0.5) / Hh) * width                 # metres toward its left (row 0: its left edge)
    hl, hr = half_widths(u, c)
    hside = np.where(across[:, None] >= 0.0, hl[None], hr[None])
    e = 1.0 - np.abs(across[:, None]) / np.maximum(hside, 1e-6)          # 1 on the midrib, 0 on the edge
    shade = 0.5 * (1.0 - G.smoothstep(e / 0.45))
    light = 0.35 * (1.0 - G.smoothstep(np.abs(across[:, None]) / np.maximum(0.3 * np.maximum(hl, hr)[None], 1e-6)))
    full = half_widths(u, SP.merge(c, {"bite": {"depth": 0.0}}))[0]
    lines = Image.new("L", (W, Hh), 0)
    d = ImageDraw.Draw(lines)

    def px(uu, a):
        return (uu * W, (0.5 - a / width) * Hh)
    mm = Hh / width / 1000.0                                             # pixels per millimetre across
    for a0, a1 in zip(u[:-8:8], u[8::8]):                               # the midrib: thick at the base, thin at the tip
        d.line([px(a0, 0.0), px(a1, 0.0)], fill=255, width=max(1, int(round((1.8 - 1.3 * a0) * mm))))
    nv = int(c["veins"])
    for i in range(nv):                                                  # pairs of veins toward the edge and the tip
        a = 0.14 + 0.60 * i / max(nv - 1, 1)
        t = np.linspace(0.0, 1.0, 40)
        ua = a + (0.24 - 0.08 * i / max(nv - 1, 1)) * t ** 0.85
        for sgn in (1.0, -1.0):
            reach = np.interp(ua, u, full) * 0.86 * np.sin(0.5 * np.pi * t) ** 0.75
            pts = [px(x, sgn * r) for x, r in zip(ua, reach)]
            for p0, p1, tt in zip(pts[:-1], pts[1:], t[:-1]):
                d.line([p0, p1], fill=255, width=max(1, int(round((0.9 - 0.6 * tt) * mm))))
    ink_ = np.asarray(lines.resize((w, h // 2), Image.LANCZOS), float)[..., None] / 255.0
    small = lambda a: np.asarray(Image.fromarray((np.clip(a, 0.0, 1.0) * 255).astype(np.uint8)).resize(
        (w, h // 2), Image.LANCZOS), float)[..., None] / 255.0
    sh, li = small(shade), small(light)
    halves = []
    for top in (True, False):
        base = col["base"] if top else col["base"] + 0.45 * (col["light"] - col["base"])
        rgb = base[None, None] + (col["light"] - base)[None, None] * li * (1.0 if top else 0.4)
        rgb = rgb + (col["shadow"] - rgb) * sh * (1.0 if top else 0.6)
        vein = col["vein"] if top else 0.5 * (col["vein"] + base)
        halves.append(rgb + (vein[None, None] - rgb) * ink_)
    out = np.empty((h, w, 4), np.uint8)
    out[..., :3] = np.rint(np.clip(np.concatenate(halves, 0), 0.0, 1.0) * 255.0).astype(np.uint8)
    out[..., 3] = 255
    return out


def ink(rgb, k=0.55):
    """The outline: the shadow's hue, darker."""
    h, s, v = colorsys.rgb_to_hsv(*(float(x) for x in rgb))
    return colorsys.hsv_to_rgb(h, min(1.0, s * 1.1), v * k)


# ---------------------------------------------------------------- the builder
@builder("leaf", needs=("body", "head"))
def build(ctx):
    c = SP.merge(DEFAULTS, ctx.cfg)
    lay = c["lay"]
    col = WR.colours(ctx, "leaf", COLORS)
    head = ctx.need("head")
    centre = np.asarray(head.info["head_center"], float)
    top_z = float(np.asarray(head.info.get("skull_top", centre + [0.0, 0.0, 0.1]), float)[2])
    surf = under(ctx.parts, top_z - 0.08)
    th = np.radians(float(lay["toward"]))
    m = np.array([np.sin(th), -np.cos(th)])                         # toward the tip, horizontal
    left = np.array([-m[1], m[0]])                                  # the leaf's left, looking from its stem to its tip
    start = centre[:2] + np.asarray(lay["at"], float)
    over, tilt, lift = lay.get("over"), float(lay["tilt"]), float(lay["lift"])
    reach = 0.5 * float(c["length"]) + float(c["stem"]) + 0.03 + (float(c["length"]) if over is not None else 0.0)
    ground = Ground(surf, start, reach, STEP, float(lay["span"]))
    edge = start + front_edge(ground, start, m, reach) * m           # where what it lies on falls away ahead of it

    mid_xy = start
    for _ in range(4 if over is not None else 1):                    # slid along its tip's way: the tip `over` past it
        top, bottom, uv = blade(ground, c, mid_xy, m, left)
        K, M = top.shape[:2]
        tip = 0.5 * (top + bottom)[-1, M // 2]
        curl = float(tip[2] - ground.z(tip[None, :2])[0] - lift)    # as it lies, before it is turned
        if tilt:
            top, bottom = turned(top, bottom, ground, tilt, lift)
        past = float((0.5 * (top + bottom)[-1, M // 2, :2] - edge) @ m)
        if over is None or abs(past - float(over)) < 2e-4:
            break
        mid_xy = mid_xy + (float(over) - past) * m
    top, bottom = lift_clear(top, bottom, surf, float(lay["clear"]))
    up_uv = np.stack([uv[..., 0], 0.5 + 0.5 * uv[..., 1]], -1)       # the top side in the picture's upper half
    lo_uv = np.stack([uv[..., 0], 0.5 * uv[..., 1]], -1)
    p_top = G.loft(top, closed=False, u=up_uv[..., 0], v=up_uv[..., 1], tag="top")
    p_bot = G.loft(bottom, closed=False, u=lo_uv[..., 0], v=lo_uv[..., 1], flip=True, tag="bottom")
    mid = 0.5 * (top + bottom)
    for p, sgn in ((p_top, 1.0), (p_bot, -1.0)):
        fn = G.face_normals(p.v, p.f)
        if (fn[:, 2] * sgn).sum() < 0.0:
            G.flip_patch(p)

    root = mid[0, M // 2]                                            # the blade's base, on its midrib
    path = stem_path(ground, root - 0.0015 * np.append(m, 0.0), -m, c)
    stem = G.tube(path, np.linspace(0.5 * float(c["stem_thick"]), 0.35 * float(c["stem_thick"]), len(path)), n=8,
                  caps=(False, True), tag="stem")
    for p in stem:                                                   # the stem in the midrib's colour
        p.uv = np.tile([0.02, 0.75], (sum(len(f) for f in p.f), 1))
    bone = str(c["bone"])
    soup = G.Soup("leaf", [MAT])
    for p in [p_top, p_bot] + stem:
        p.w = {bone: np.ones(len(p.v))}
        soup.add(p, 0)
    mesh = soup.mesh(weld=2e-6)

    shadow = col["shadow"]
    mat = WR.material(ctx, MAT, "leaf", col["base"], picture(col, c),
                      shadow=tuple(float(x) for x in np.clip(shadow / np.maximum(col["base"], 1e-3), 0.0, 1.0)),
                      edge_size=0.5, double_sided=False)
    mat.edge_color = tuple(float(x) for x in ink(shadow)) + (1.0,)

    d = np.abs(surf.closest(bottom.reshape(-1, 3))[0]).reshape(K, M)
    raw = ground.z(bottom[..., :2], np.where(np.isfinite(ground.raw), ground.raw, -1.0))
    down = bottom[..., 2] - raw                                       # straight down to what is under it
    flat = uv[:, 0, 0] < 1.0 - float(lay["curl_from"])                 # the rows before the tip curls up
    tip = mid[-1, M // 2]
    across = np.linalg.norm(mid[:, 0] - mid[:, -1], axis=1)
    normal = G.unit(G.face_normals(p_top.v, p_top.f).sum(0))           # its upper face, on the whole
    info = {"length": float(np.linalg.norm(np.diff(mid[:, M // 2], axis=0), axis=1).sum()),
            "stem": float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum()),
            "width": float(across.max()), "middle": mid[K // 2, M // 2], "tip": tip, "base": root,
            "toward": float(np.degrees(np.arctan2(tip[0] - root[0], -(tip[1] - root[1])))),
            "off": {"least": float(d.min()), "median": float(np.median(d[flat])), "most": float(d[flat].max()),
                    "tip": float(d[~flat].max())},
            "over": {"least": float(down.min()), "median": float(np.median(down[flat])),
                     "most": float(down[flat].max())},
            "curl": curl, "dips": float((ground.z(bottom[..., :2]) - raw).max()),
            "faces_front": float(np.degrees(np.arctan2(-normal[1], normal[2]))), "past_edge": past,
            "edge": np.append(edge, ground.z(edge[None])[0]),
            "tip_over": float(down[-1, M // 2]), "landmarks": {"leaf": mid[K // 2, M // 2]}}
    o, v = info["off"], info["over"]
    ctx.log(f"leaf {info['length'] * 1000:.0f} mm long (+ stem {info['stem'] * 1000:.0f}), {info['width'] * 1000:.0f} mm "
            f"wide, tip toward {info['toward']:.0f} deg; its underside {o['least'] * 1000:.1f} .. {o['median'] * 1000:.1f}"
            f" .. {o['most'] * 1000:.1f} mm off what is under it, {v['least'] * 1000:.1f} .. {v['median'] * 1000:.1f} .. "
            f"{v['most'] * 1000:.1f} mm over it (bridging dips up to {info['dips'] * 1000:.1f}); tip curled up "
            f"{curl * 1000:.1f} mm, {info['tip_over'] * 1000:.0f} mm over what is below it and {past * 1000:.0f} mm past "
            f"the front edge of what it lies on; its face turned {info['faces_front']:.0f} deg to the front")
    return Part("leaf", meshes=[mesh], materials=[mat], info=info)
