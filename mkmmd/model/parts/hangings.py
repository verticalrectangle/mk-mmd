"""Things hanging on cords (part `hangings`, numpy only): a sake jug, an account book, each on a cord from a place on her
body, lying against what it hangs over and swinging on a chain of two bones from the lower body.

`[[hangings.items]]`, each: `kind` ("jug" or "book"); `at` [degrees round her from the front toward her left, height]:
where its cord starts, on the outermost surface built before it there (the smock), `gap` off it; the height is from the
floor, or from the hem of the part `[hangings] hem` names (its info `z_hem`: "smock" hangs them from the smock's hem
wherever it goes); `cord` {length, thick}: the cord runs `length` to the thing's top; `size`: the kind's sizes (below).
The cord, taut, and the thing hanging from its end come to rest as a weight on a cord does against a round skirt: each
turned in the upright plane through that place to the pose whose centre of mass lies lowest while both stay `gap` off
what they hang over (the parts built before, each garment's triangles turned to face away from her upright so that
outside them is in front of them whichever way they are wound, and `[hangings.stand_in]` while the garments they rest
on are not built: a surface of revolution round the upright through `axis` [x, y], its `profile` [[z, half width, half
depth], ...] from the top down), the thing leaning out `lean` degrees at most. A thing faces out from her: what is
drawn on it looks away from her side.

  jug   a round-bellied flask hung by its neck (`size`: `belly` [across, tall], `neck` [across, tall], `cork` [across,
        tall], `mark` the height of the brushed mark): white glaze ([colors.jug] glaze, glaze_shadow), the brushed
        character for sake on its belly (mark), a cork (cork), the cord tied round its neck (cord)
  book  a block of paper hung by its top (`size`: `page` [wide, tall, thick], `cover` the depth of the dark strip it is
        bound along its top under, `lines` how many brushed lines its top page has): [colors.ledger] paper,
        paper_shadow (the page edges), cover, ink (the lines: shapes, no words), cord

Each thing hangs on a chain of two bones from 下半身 (`bone`; 吊り飾り<A, B ...>1: the cord; 2: the thing), family
`accessory`: a thin capsule round the cord, a capsule (the jug) or a box (the book) round the thing, joints `swing` radians
each way; the cord's top stays on the lower body. With `[hangings] ride` naming a part built before them (a flared
skirt, whose own chains swing), they have no chain: each thing takes that part's weights where it lies on it (its
nearest vertices'), the cord going over to them from where it is tied, so it moves with the cloth it lies on and never
passes into it. Materials 吊り紐 (cords), 徳利 (glaze and mark), 徳利栓 (cork), 帳面 (paper
and lines), 帳面表紙 (the cover strip): each toon shade its shadow colour over its colour, each outline a darker shade
of its own hue. `info` holds the measures of each: its kind, sizes as built, where its cord starts, its top and bottom,
how far it swung out, its least gap to what it lies on; landmarks `hang.<i>` (each thing's middle)."""
import colorsys

import numpy as np
from PIL import Image, ImageDraw

from .. import spec as SP
from ..build import builder
from ..part import Bone, Joint, Part, RigidBody
from . import outfit_fit as OF
from . import outfit_geo as G
from . import outfit_wear as WR
from .hair_fit import raycast_hits
from .hair_rig import euler_for_axis
from .outfit_rig import euler_xyz

DEFAULTS = {"items": [], "gap": 0.002, "lean": 40.0, "hem": "", "bone": "下半身", "swing": 0.7, "stand_in": {},
            "ride": ""}
ITEM = {"kind": None, "at": [40.0, 0.845], "cord": {"length": 0.045, "thick": 0.002}, "size": {}}
SIZES = {"jug": {"belly": [0.062, 0.070], "neck": [0.022, 0.011], "cork": [0.017, 0.013], "mark": 0.030},
         "book": {"page": [0.070, 0.095, 0.012], "cover": 0.018, "lines": 4}}
STAND_IN = {"axis": [0.0, 0.0], "profile": []}
COLORS = {"jug": {"glaze": "#f3f1ec", "glaze_shadow": "#c5c2bd", "cork": "#c98f3d", "mark": "#3b2a2a",
                  "cord": "#8a6a4a"},
          "ledger": {"paper": "#f7f3ea", "paper_shadow": "#d6cfc0", "cover": "#3b2a2a", "ink": "#5b4a48",
                     "cord": "#8a6a4a"}}
TABLE = {"jug": "jug", "book": "ledger"}
MATS = ("吊り紐", "徳利", "徳利栓", "帳面", "帳面表紙")
CORD, JUG, CORK, PAPER, COVER = range(5)
MASS = {"cord": 0.05, "jug": 0.4, "book": 0.25}
UP = np.array([0.0, 0.0, 1.0])

# the brushed character for sake: strokes in a unit box (x right, y down), each [points, width at start, at end]
SAKE = [([(0.10, 0.10), (0.16, 0.15), (0.21, 0.21)], 0.10, 0.06),
        ([(0.06, 0.38), (0.13, 0.42), (0.19, 0.48)], 0.10, 0.06),
        ([(0.07, 0.92), (0.12, 0.80), (0.18, 0.68), (0.25, 0.56)], 0.10, 0.04),
        ([(0.33, 0.14), (0.64, 0.12), (0.95, 0.13)], 0.08, 0.06),
        ([(0.40, 0.30), (0.40, 0.62), (0.41, 0.95)], 0.08, 0.06),
        ([(0.40, 0.30), (0.66, 0.29), (0.88, 0.30), (0.89, 0.60), (0.89, 0.93), (0.85, 0.90)], 0.08, 0.06),
        ([(0.55, 0.14), (0.55, 0.30), (0.53, 0.45), (0.48, 0.57)], 0.07, 0.04),
        ([(0.72, 0.14), (0.72, 0.30), (0.72, 0.50), (0.78, 0.53), (0.83, 0.53)], 0.07, 0.045),
        ([(0.41, 0.68), (0.88, 0.68)], 0.065, 0.06),
        ([(0.41, 0.92), (0.88, 0.92)], 0.075, 0.065)]


# ---------------------------------------------------------------- what they hang over
def stand_in_mesh(axis, profile, n_lon=72, step=0.005):
    """A surface of revolution round the upright through `axis` (x, y): elliptic rings of [half width, half depth] at
    the heights of `profile` ([[z, half width, half depth], ...], top first), eased between them (Catmull-Rom),
    open at both ends, facing out: (verts, triangles). It stands in for a garment not built yet."""
    P = np.asarray(profile, float)
    if P.ndim != 2 or P.shape[1] != 3 or len(P) < 2 or (np.diff(P[:, 0]) >= 0.0).any():
        raise ValueError(f"[hangings.stand_in] profile = {profile!r}: [[z, half width, half depth], ...], two or more, "
                         "from the top down")
    rows = max(int((P[0, 0] - P[-1, 0]) / step), len(P))
    Q = G.catmull(P, rows)
    lon = 2.0 * np.pi * np.arange(n_lon) / n_lon
    V = np.stack([axis[0] + Q[:, 1:2] * np.sin(lon)[None], axis[1] - Q[:, 2:3] * np.cos(lon)[None],
                  np.broadcast_to(Q[:, :1], (len(Q), n_lon))], -1).reshape(-1, 3)
    q = G.grid_quads(len(Q), n_lon, True)                              # top ring first: these face out
    return V, np.vstack([q[:, [0, 1, 2]], q[:, [0, 2, 3]]])


def resting_surface(parts, stand_in, axis):
    """What the things hang over: every part built before them but the head (the body, the garments), and the stand-in
    when one is given (outfit_fit.Skin). A garment is a sheet that may be wound either way: each of its triangles is
    turned to face away from the upright through `axis` (x, y), so that in front of it is outside, as for the body."""
    V, T, off = [], [], 0
    for name, p in parts.items():
        if name == "head":
            continue
        for m in p.meshes:
            if len(m.verts) and len(m.faces):
                v = np.asarray(m.verts, float)
                t = G.tri_arrays(list(m.faces))[0]
                if name != "body":
                    X = v[t]
                    n = np.cross(X[:, 1] - X[:, 0], X[:, 2] - X[:, 0])
                    radial = X.mean(1) - [axis[0], axis[1], 0.0]
                    inward = n[:, 0] * radial[:, 0] + n[:, 1] * radial[:, 1] < 0.0
                    t[inward] = t[inward][:, [0, 2, 1]]
                V.append(v)
                T.append(t + off)
                off += len(v)
    if stand_in:
        s = SP.merge(STAND_IN, stand_in)
        v, t = stand_in_mesh(np.asarray(s["axis"], float), s["profile"])
        V.append(v)
        T.append(t + off)
    return OF.Skin(np.vstack(V), np.vstack(T), name="hangings_under")


def outermost(surf, axis, theta, z, reach=0.5):
    """The farthest point of `surf` from the vertical axis through `axis` (x, y) at azimuth `theta` (radians round from
    the front toward her left) and height z, and the horizontal outward direction there."""
    d = np.array([np.sin(theta), -np.cos(theta), 0.0])
    origin = np.array([axis[0], axis[1], z]) + d * reach
    t = raycast_hits(origin, -d[None], surf.verts, surf.tris)[0][0]
    if not np.isfinite(t):
        raise ValueError(f"hangings: nothing to hang from at {np.degrees(theta):.0f} deg, z {z:.3f}")
    return origin - d * t, d


def hem_edge(part, axis, theta, wedge=np.radians(6.0)):
    """The lowest point of the garment `part` within `wedge` of the azimuth `theta` round the upright through `axis`
    (x, y): the edge of its hem there, or the tip of its lace's nearest point. Returns (its azimuth, its height)."""
    V = np.vstack([np.asarray(m.verts, float) for m in part.meshes if len(m.verts)])
    az = np.arctan2(V[:, 0] - axis[0], -(V[:, 1] - axis[1]))
    near = np.flatnonzero(np.abs(np.angle(np.exp(1j * (az - theta)))) <= wedge)
    if not len(near):
        raise ValueError(f"[hangings] hem = {part.name!r}: nothing of it at {np.degrees(theta):.0f} deg")
    k = near[np.argmin(V[near, 2])]
    return float(az[k]), float(V[k, 2])


# ---------------------------------------------------------------- the things, in their own frame: top at the origin,
# hanging down -z, facing +y (out from her), +x across
def lathe(profile, m=40, mat=JUG, v=None, tag=""):
    """A surface of revolution about the z axis through the (r, z) `profile` (bottom to top), u round from -y (her side)
    through +x so that u 0.5 faces +y (out), v the profile's arc length share (or `v`); its points with r 0 close it."""
    P = np.asarray(profile, float)
    phi = 2.0 * np.pi * np.arange(m) / m
    rings = np.stack([np.sin(phi)[None] * P[:, :1], -np.cos(phi)[None] * P[:, :1], np.broadcast_to(P[:, 1:], (len(P), m))],
                     -1)
    if v is None:
        s = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(P[:, 0]), np.diff(P[:, 1])))])
        v = s / max(s[-1], 1e-9)
    patch = G.loft(rings[::-1], closed=True, u=np.linspace(0.0, 1.0, m + 1), v=np.asarray(v)[::-1], mat=mat, tag=tag)
    return G.orient_outward(patch, np.repeat(np.stack([np.zeros(len(P)), np.zeros(len(P)), P[:, 1]], -1)[::-1], m, 0))


def jug(sz):
    """The jug's patches, its height below its tie, the picture's layout and its capsule radius."""
    bw, bh = (float(x) for x in sz["belly"])
    nw, nh = (float(x) for x in sz["neck"])
    cw, ch = (float(x) for x in sz["cork"])
    R, rn = 0.5 * bw, 0.5 * nw
    t = np.linspace(0.0, 1.0, 26)
    # the belly: a flat foot rounding into a full belly widest a little under its middle, in at the shoulder to the neck
    hb = bh * t
    rb = R * np.sqrt(np.clip(np.sin(np.pi * np.clip(t, 0.0, 1.0) ** 0.85), 0.0, 1.0))
    rb = np.maximum(rb, rn * G.smoothstep((t - 0.6) / 0.4))
    rb[0] = 0.62 * R
    foot = np.array([[0.0, 0.0]])
    body = np.stack([rb, hb], 1)[1:]
    top = bh + nh + 0.0016
    cr = 0.5 * cw
    neck = np.array([[rn, bh + 0.15 * nh], [rn * 0.96, bh + 0.5 * nh], [rn, bh + 0.85 * nh],
                     [rn * 1.12, bh + nh], [rn * 1.12, top - 0.0004], [rn * 0.97, top], [cr * 0.98, top]])
    glaze = np.vstack([foot, [[0.62 * R, 0.0]], body, neck])               # its mouth closed under the cork
    cork = np.array([[0.0, top - 0.004], [cr * 0.92, top - 0.004], [cr, top + 0.4 * ch], [cr * 0.97, top + ch - 0.003],
                     [cr * 0.75, top + ch - 0.0004], [0.0, top + ch]])
    tie = bh + 0.5 * nh
    s = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(glaze[:, 0]), np.diff(glaze[:, 1])))])
    shift = np.array([0.0, -tie])
    p_glaze = lathe(glaze + shift, mat=JUG, v=s / s[-1], tag="jug")
    p_cork = lathe(cork + shift, mat=CORK, tag="cork")
    layout = {"arc": float(s[-1]), "belly_v": float(np.interp(0.45 * bh, glaze[1:, 1], s[1:]) / s[-1]),
              "girth": 2.0 * np.pi * R}
    return [p_glaze, p_cork], tie, layout, R


def rounded_rect(w, t, rc, m=64):
    """A rounded rectangle (m, 2) w wide (x) and t thick (y), corners rc, from the middle of its back (-y) round
    through +x, its front (+y) and -x; and each point's share round it."""
    rc = min(rc, 0.49 * t, 0.49 * w)
    a, b = 0.5 * w - rc, 0.5 * t - rc
    s = np.linspace(0.0, 1.0, m, endpoint=False)
    per = 4.0 * (a + b) + 2.0 * np.pi * rc
    pts = []
    for x in s * per:                                  # walk the outline from the back's middle toward +x
        segs = [(a, "b0"), (0.5 * np.pi * rc, "c0"), (2 * b, "s0"), (0.5 * np.pi * rc, "c1"), (2 * a, "f"),
                (0.5 * np.pi * rc, "c2"), (2 * b, "s1"), (0.5 * np.pi * rc, "c3"), (a, "b1")]
        for L, k in segs:
            if x <= L + 1e-12:
                break
            x -= L
        ang = x / rc if rc > 0 else 0.0
        pts.append({"b0": (x, -0.5 * t), "c0": (a + rc * np.sin(ang), -b - rc * np.cos(ang)),
                    "s0": (0.5 * w, -b + x), "c1": (a + rc * np.cos(ang), b + rc * np.sin(ang)),
                    "f": (a - x, 0.5 * t), "c2": (-a - rc * np.sin(ang), b + rc * np.cos(ang)),
                    "s1": (-0.5 * w, b - x), "c3": (-a - rc * np.cos(ang), -b - rc * np.sin(ang)),
                    "b1": (-a + x, -0.5 * t)}[k])
    return np.array(pts), s, (a + 0.5 * np.pi * rc + 2 * b + 0.5 * np.pi * rc) / per, 2 * a / per


def box_loft(w, t, z0, z1, rc, mat, m=64, rows=4, tag=""):
    """A rounded box w x t from z1 (top) down to z0, its sides lofted, its ends closed: patches, and where its front
    face starts and how much of the way round it is (for the picture)."""
    ring, u, f0, fw = rounded_rect(w, t, rc, m)
    zs = np.linspace(z1, z0, rows)
    rings = np.stack([np.concatenate([ring, np.full((m, 1), z)], 1) for z in zs])
    side = G.loft(rings, closed=True, u=np.append(u, 1.0), v=np.linspace(1.0, 0.0, rows), mat=mat, tag=tag)
    side = G.orient_outward(side, np.repeat(np.stack([np.zeros(rows), np.zeros(rows), zs], -1), m, 0))
    caps = []
    for z, up in ((z1, 1.0), (z0, -1.0)):
        cap = G.fan(np.array([0.0, 0.0, z]), rings[0 if up > 0 else -1], mat=mat, tag=tag, uv_apex=(0.5, 0.5))
        cap.uv = np.full((sum(len(f) for f in cap.f), 2), [0.5, 0.995 if up > 0 else 0.005])
        caps.append(G.orient_outward(cap, np.array([0.0, 0.0, z - up])))
    return [side] + caps, f0, fw


def book(sz):
    """The book's patches (the paper block, the cover strip over its top), its height and the picture's layout."""
    W, H, T = (float(x) for x in sz["page"])
    cov = float(sz["cover"])
    paper, f0, fw = box_loft(W, T, -H, -0.002, 0.0015, PAPER, tag="paper")
    strip, s0, sw = box_loft(W + 0.0014, T + 0.0014, -cov, 0.0, 0.0018, COVER, tag="cover")
    return paper + strip, H, {"front": (f0, fw), "height": H, "cover": cov}, 0.5 * T


# ---------------------------------------------------------------- pictures
def brush(draw, strokes, box, size, ink):
    """Strokes ([points], width at start, at end) of a unit box drawn with a round brush into `box` (x0, y0) of `size`
    pixels: smooth, thick where the brush lands, thinning as it lifts."""
    x0, y0 = box
    for pts, w0, w1 in strokes:
        P = np.asarray(pts, float)
        n = max(int(np.linalg.norm(np.diff(P, axis=0), axis=1).sum() * size / 1.5), 2)
        Q = G.catmull(P, n) if len(P) > 2 else np.linspace(P[0], P[-1], n)
        t = np.linspace(0.0, 1.0, len(Q))
        w = (w0 + (w1 - w0) * t) * (1.0 + 0.25 * np.exp(-((t - 0.04) / 0.06) ** 2))
        for (x, y), ww in zip(Q, w):
            r = 0.5 * ww * size
            draw.ellipse([x0 + x * size - r, y0 + y * size - r, x0 + x * size + r, y0 + y * size + r], fill=ink)


def jug_picture(col, lay, mark, w=512, h=512):
    """The jug's picture: u round it (0.5 its front, out from her), v up its profile. The glaze, in soft shade toward
    its foot, the brushed mark `mark` metres tall on the front of its belly."""
    v = (1.0 - (np.arange(h) + 0.5) / h)[:, None]
    shade = 0.45 * (1.0 - G.smoothstep(v / 0.25))
    rgb = col["glaze"][None, None] + (col["glaze_shadow"] - col["glaze"])[None, None] * shade[..., None]
    img = Image.fromarray(np.rint(np.broadcast_to(rgb, (h, w, 3)) * 255).astype(np.uint8))
    S = 4
    big = Image.new("L", (w * S, h * S), 0)
    d = ImageDraw.Draw(big)
    hv = mark / lay["arc"] * h * S                    # the mark's height in pixels of the picture (big)
    wu = mark / lay["girth"] * w * S                  # the same length round the belly
    cx, cy = 0.5 * w * S, (1.0 - lay["belly_v"]) * h * S
    sx = wu / hv
    tmp = Image.new("L", (int(hv) + 8, int(hv) + 8), 0)
    brush(ImageDraw.Draw(tmp), SAKE, (4, 4), hv, 255)
    tmp = tmp.resize((max(int(round((hv + 8) * sx)), 1), int(hv) + 8), Image.LANCZOS)
    big.paste(tmp, (int(cx - tmp.width / 2), int(cy - tmp.height / 2)))
    del d
    a = np.asarray(big.resize((w, h), Image.LANCZOS), float)[..., None] / 255.0
    out = np.empty((h, w, 4), np.uint8)
    base = np.asarray(img, float) / 255.0
    out[..., :3] = np.rint(np.clip(base + (col["mark"][None, None] - base) * a, 0, 1) * 255).astype(np.uint8)
    out[..., 3] = 255
    return out


def book_picture(col, lay, lines, seed, w=512, h=512):
    """The paper's picture: u round the block (its front `front` = (start, share)), v up it. The paper, its edges in
    thin page lines of its shadow, and on the front page `lines` brushed columns of shapes (no words), top to bottom
    under the cover strip."""
    rng = np.random.default_rng(seed)
    u = (np.arange(w) + 0.5) / w
    v = (1.0 - (np.arange(h) + 0.5) / h)
    f0, fw = lay["front"]
    front = (u >= f0) & (u <= f0 + fw)
    back = (u <= 0.5 * fw) | (u >= 1.0 - 0.5 * fw)                       # the back faces her, round u 0
    edge = ~front & ~back                                                 # the block's sides: the pages' edges
    pages = 0.5 + 0.5 * np.cos(u[None, :] * 2.0 * np.pi * 160.0)
    shade = np.where(edge[None, :], 0.45 + 0.35 * pages, 0.08 * (1.0 - v[:, None]))
    rgb = col["paper"][None, None] + (col["paper_shadow"] - col["paper"])[None, None] * shade[..., None]
    S = 4
    big = Image.new("L", (w * S, h * S), 0)
    d = ImageDraw.Draw(big)
    page_h = 1.0 - lay["cover"] / lay["height"]                           # the page below the strip, as a share of v
    for k in range(int(lines)):
        x = (f0 + fw * (0.78 - 0.56 * k / max(int(lines) - 1, 1))) * w * S
        y0, y1 = (1.0 - page_h * 0.92) * h * S, (1.0 - page_h * (0.12 + 0.25 * rng.random())) * h * S
        n = 9
        ys = np.linspace(y0, y1, n)
        xs = x + rng.normal(0.0, 0.010 * w * S, n) * np.sin(np.linspace(0, 3 * np.pi, n))
        for i in range(n - 1):
            d.line([(xs[i], ys[i]), (xs[i + 1], ys[i + 1])], fill=255, width=max(2, int(0.006 * w * S)))
    a = np.asarray(big.resize((w, h), Image.LANCZOS), float)[..., None] / 255.0
    out = np.empty((h, w, 4), np.uint8)
    out[..., :3] = np.rint(np.clip(rgb + (col["ink"][None, None] - rgb) * a, 0, 1) * 255).astype(np.uint8)
    out[..., 3] = 255
    return out


def ink(rgb, k=0.6):
    """An outline colour: the same hue, darker (a drawing's ink, never black)."""
    h, s, v = colorsys.rgb_to_hsv(*(float(x) for x in rgb))
    return colorsys.hsv_to_rgb(h, min(1.0, s * 1.1 + 0.05), max(v * k, 0.12))


def material(ctx, name, name_en, base, shadow, picture=None, edge=0.5, **kw):
    mult = tuple(float(x) for x in np.clip(shadow / np.maximum(base, 1e-3), 0.0, 1.0))
    mat = WR.material(ctx, name, name_en, base, picture, shadow=mult, edge_size=edge, double_sided=False, **kw)
    mat.edge_color = tuple(float(x) for x in ink(shadow)) + (1.0,)
    return mat


# ---------------------------------------------------------------- riding a cloth
def merged_weights(meshes):
    """{bone: (n,)} of the meshes' vertices stacked in their order (a bone a mesh lacks: 0 there)."""
    names = sorted({b for m in meshes for b in m.weights})
    return {b: np.concatenate([np.asarray(m.weights.get(b, np.zeros(len(m.verts))), float) for m in meshes])
            for b in names}


def riding_weights(V, W, p, k=8):
    """{bone: weight} of the cloth (vertices V (n, 3), weights W {bone: (n,)}) at the point p: its `k` nearest
    vertices' weights, the nearer counting more, normalised (the bones under 1e-4 left out)."""
    d = np.linalg.norm(np.asarray(V, float) - np.asarray(p, float), axis=1)
    idx = np.argpartition(d, min(k, len(d) - 1))[:k]
    near = 1.0 / np.maximum(d[idx], 1e-4)
    got = {b: float((w[idx] * near).sum() / near.sum()) for b, w in W.items()}
    got = {b: v for b, v in got.items() if v > 1e-4}
    total = sum(got.values())
    return {b: v / total for b, v in got.items()}


def blend(a, b, s):
    """Per-vertex weights (1 - s) a + s b: `a` {bone: (n,)}, `b` {bone: weight}, `s` (n,) 0..1."""
    out = {k: (1.0 - s) * np.asarray(v, float) for k, v in a.items()}
    for k, v in b.items():
        out[k] = out.get(k, 0.0) + s * v
    return out


# ---------------------------------------------------------------- hanging them
def frame(down, out):
    """Rotation (3, 3) from a thing's own frame (x across, y out, z up its axis) to model space, its axis along -`down`."""
    z = -G.unit(down)
    y = G.unit(out - (out @ z) * z)
    return np.stack([np.cross(y, z), y, z], 1)


def swung(b, t, angle):
    """Direction b turned about the unit axis t by `angle` (radians)."""
    t = G.unit(t)
    return b * np.cos(angle) + np.cross(t, b) * np.sin(angle) + t * (t @ b) * (1.0 - np.cos(angle))


def drape(surf, A, out, cord, clear, step=0.001):
    """Where a cord going down from A, taut, last rests on what is under it in the upright plane through `out`: the
    point K of that surface (`clear` off it) seen from A furthest out (a skirt bulging out under a hem, a lace's
    points), within the cord's reach; A itself when nothing below A stands out past it."""
    best, K = 0.0, A
    for k in range(1, int(cord / step) + 1):
        origin = A + out * 0.3 - [0.0, 0.0, k * step]
        hit = raycast_hits(origin, -out[None], surf.verts, surf.tris)[0][0]
        if not np.isfinite(hit):
            continue
        h = 0.3 - float(hit) + clear                     # how far out of A's upright it stands, with the clearance
        P = A + out * h - [0.0, 0.0, k * step]
        if np.arctan2(h, k * step) > best and np.linalg.norm(P - A) < cord:
            best, K = np.arctan2(h, k * step), P
    return K


def settle(surf, A, out, cord, local, com, gap, cord_r, lean=np.pi / 2):
    """Where the cord and the thing come to rest: the cord from A, draped over what bulges out under it to K (`drape`),
    then taut, the thing hanging from its end, each turned in the upright plane through `out` (positive: outward) to the
    pose whose centre of mass `com` (in the thing's own frame) lies lowest while the cord stays `gap` + `cord_r` and the
    thing `gap` off `surf` (as a weight on a cord comes to rest against a round skirt), the thing leaning out no more
    than `lean` (radians): the cord swings further instead. `local`: the thing's points in its own frame (top at the
    origin). Searched on a 10 degree grid, then finer round the best; then a little further out if any of its points
    comes nearer than `gap`. Returns (K, the cord's end B, the thing's rotation, the free cord's and the thing's angles
    from straight down)."""
    t = np.cross(out, UP)                                            # turning about this horizontal axis swings it out
    down = np.array([0.0, 0.0, -1.0])
    along = np.linspace(0.15, 1.0, 8)[:, None]
    reach = cord + float(np.linalg.norm(local, axis=1).max()) + 0.02
    surf = surf.where(tri_mask=(np.abs(surf.verts[surf.tris] - A) <= reach).all(2).any(1))
    pts = local[np.linspace(0, len(local) - 1, min(len(local), 160)).astype(int)]

    def search(K):
        free = cord - float(np.linalg.norm(K - A))
        seen = {}

        def energy(a1, a2, P=pts):
            key = (round(a1, 5), round(a2, 5), len(P))
            if key not in seen:
                d1 = swung(down, t, a1)
                B = K + free * d1
                R = frame(swung(down, t, a2), out)
                ok = (a2 <= lean + 1e-9 and surf.closest(K + along * free * d1)[0].min() >= gap + cord_r
                      and surf.closest(B + P @ R.T)[0].min() >= gap)
                seen[key] = float((B + R @ com)[2] + 0.1 * B[2]) if ok else np.inf
            return seen[key]
        best = min(((energy(a1, a2), a1, a2) for a1 in np.radians(np.arange(-20, 91, 10))
                    for a2 in np.radians(np.arange(-20, 121, 10))), key=lambda e: e[0])
        if not np.isfinite(best[0]):
            return None
        for step in (2.0, 0.5):
            d = np.radians(step * np.arange(-3, 4))
            best = min(((energy(best[1] + d1, best[2] + d2), best[1] + d1, best[2] + d2) for d1 in d for d2 in d),
                       key=lambda e: e[0])
        _, a1, a2 = best
        for _ in range(80):                                          # every point of it off what is under it
            if np.isfinite(energy(a1, a2, local)):
                break
            if a2 + np.radians(0.5) <= lean:
                a2 += np.radians(0.5)
            else:
                a1 += np.radians(0.5)
        return K, K + free * swung(down, t, a1), frame(swung(down, t, a2), out), a1, a2

    K = drape(surf, A, out, cord, gap + cord_r)
    got = search(K)
    lie = np.arctan2(float((K - A) @ out), float(A[2] - K[2])) if np.linalg.norm(K - A) > 1e-9 else np.inf
    if got is None or got[3] > lie + np.radians(1.0):                # it cannot rest so, or never touched: straight
        got = search(A)
    if got is None:
        raise ValueError(f"hangings: nothing hangs {gap * 1000:.1f} mm off what is under it from "
                         f"{np.round(A, 3).tolist()} leaning out {np.degrees(lean):.0f} degrees at most")
    return got


# ---------------------------------------------------------------- the builder
@builder("hangings", needs=("body",))
def build(ctx):
    c = SP.merge(DEFAULTS, ctx.cfg)
    items = [SP.merge(ITEM, it) for it in c["items"]]
    if not items:
        raise ValueError("[[hangings.items]] is empty: nothing hangs")
    bad = [it["kind"] for it in items if it["kind"] not in SIZES]
    if bad:
        raise ValueError(f"[[hangings.items]] kind = {bad[0]!r}: one of {', '.join(SIZES)}")
    cols = {k: WR.colours(ctx, t, COLORS[t]) for k, t in TABLE.items() if any(it["kind"] == k for it in items)}
    gap, bone = float(c["gap"]), str(c["bone"])
    hips = [ctx.land[k] for k in ("leg.L", "leg.R") if k in ctx.land]
    axis = np.mean(hips, 0) if hips else np.asarray(ctx.land["lower_body"], float)
    surf = resting_surface(ctx.parts, c["stand_in"], axis)
    hem = ctx.need(str(c["hem"])) if c["hem"] else None
    ride = ctx.need(str(c["ride"])) if c["ride"] else None
    ride_v = np.vstack([m.verts for m in ride.meshes]) if ride is not None else None
    ride_w = merged_weights(ride.meshes) if ride is not None else None
    anchor = ctx.find_body(bone)
    soup = G.Soup("hangings", list(MATS))
    bones, bodies, joints, info, land, pictures = [], [], [], [], {}, {}
    for i, it in enumerate(items):
        kind, sz = it["kind"], SP.merge(SIZES[it["kind"]], it["size"])
        cord_len, cord_r = float(it["cord"]["length"]), 0.5 * float(it["cord"]["thick"])
        theta, z = np.radians(float(it["at"][0])), float(it["at"][1])
        if hem is not None:                                          # tied just above its hem's lowest edge there
            theta, edge = hem_edge(hem, axis, theta)
            z += edge
        P0, out = outermost(surf, axis, theta, z)
        A = P0 + out * (gap + cord_r)
        patches, length, lay, radius = jug(sz) if kind == "jug" else book(sz)
        pictures[kind] = (lay, sz)
        local = np.vstack([p.v for p in patches])
        K, B, R, a1, a2 = settle(surf, A, out, cord_len, local, local.mean(0), gap, cord_r,
                                 np.radians(float(c["lean"])))
        lie = float(np.linalg.norm(K - A))
        pre = np.linspace(A, K, max(int(lie / 0.004), 1) + 1)[:-1] if lie > 1e-6 else np.zeros((0, 3))
        path = np.vstack([pre, np.linspace(K, B, 12)])
        tag = chr(ord("A") + i)
        names = [f"吊り飾り{tag}1", f"吊り飾り{tag}2"]
        C = B + R @ np.array([0.0, 0.0, -length])
        for p in patches:
            p.v = B + p.v @ R.T
        if ride is not None:                     # the cloth's weights where it lies on it: it moves with the cloth
            on = riding_weights(ride_v, ride_w, np.vstack([p.v for p in patches]).mean(0))
        for p in patches:
            p.w = {b: np.full(len(p.v), w) for b, w in on.items()} if ride is not None else {names[1]: np.ones(len(p.v))}
            soup.add(p)
        # the cord: on the lower body where it lies on what it hangs over (to K) and a little past it, the rest on the
        # cord's bone; tied round the thing's top
        cord = G.tube(path, cord_r, n=6, caps=(True, False), mat=CORD, tag="cord")
        arc = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))])
        for p in cord:
            near = np.argmin(np.linalg.norm(p.v[:, None] - path[None], axis=2), axis=1)
            hold = 1.0 - G.smoothstep((arc[near] - lie) / (0.25 * (arc[-1] - lie)))
            if ride is not None:                 # from where it is tied over to the cloth's weights at the thing
                s = G.smoothstep(arc[near] / max(arc[-1], 1e-9))
                p.w = blend({bone: np.ones(len(p.v))}, on, s)
            else:
                p.w = {bone: hold, names[0]: 1.0 - hold}
            p.uv = np.tile([0.5, 0.5], (sum(len(f) for f in p.f), 1))
            soup.add(p)
        knot = G.ellipsoid(A, [1.6 * cord_r] * 3, n_lat=5, n_lon=8, mat=CORD, tag="knot")
        knot.w = {bone: np.ones(len(knot.v))}
        knot.uv = np.tile([0.5, 0.5], (sum(len(f) for f in knot.f), 1))
        soup.add(knot)
        if kind == "jug":
            ring_r = 0.5 * float(sz["neck"][0]) + cord_r
            phi = 2.0 * np.pi * np.arange(25) / 24
            loop = B + (np.stack([ring_r * np.cos(phi), ring_r * np.sin(phi), np.zeros_like(phi)], 1)) @ R.T
            tie = G.tube(loop, cord_r, n=6, caps=(False, False), mat=CORD, tag="tie")
        else:
            arc = np.linspace(0.0, np.pi, 9)
            loop = B + np.stack([0.004 * np.cos(arc), np.zeros_like(arc), 0.0015 + 0.003 * np.sin(arc)], 1) @ R.T
            tie = G.tube(loop, cord_r, n=6, caps=(True, True), mat=CORD, tag="tie")
        for p in tie:
            p.w = {b: np.full(len(p.v), w) for b, w in on.items()} if ride is not None else {names[1]: np.ones(len(p.v))}
            p.uv = np.tile([0.5, 0.5], (sum(len(f) for f in p.f), 1))
            soup.add(p)
        # the chain: the cord's bone from where it leaves what it lies on, and the thing's (none when it rides a cloth)
        sw = float(c["swing"])
        if ride is None:
            for k, (h, tl) in enumerate(((K, B), (B, C))):
                bones.append(Bone(names[k], head=tuple(float(x) for x in h), tail=tuple(float(x) for x in tl),
                                  parent=bone if k == 0 else names[0], name_en=f"hanging_{tag}{k + 1}",
                                  tail_bone=names[1] if k == 0 else "", after_physics=True))
            rot1, span = euler_for_axis(B - K), float(np.linalg.norm(B - K))
            bodies.append(RigidBody(names[0], names[0], "capsule", (max(cord_r * 2.0, 0.003), max(span - 0.006, 0.002),
                                                                    0.0),
                                    tuple(float(x) for x in 0.5 * (K + B)), rot1, "dynamic", 8,
                                    (0, 4, 5, 6, 7, 8, 9, 10, 11), MASS["cord"], (0.9, 0.95), 0.4))
            if kind == "jug":
                shape, size, rot = "capsule", (radius, max(length - 2.0 * radius, 0.002), 0.0), euler_for_axis(C - B)
            else:
                W, H, T = (float(x) for x in sz["page"])
                shape, size, rot = "box", (0.5 * W, 0.5 * T, 0.5 * H), euler_xyz(R)
            bodies.append(RigidBody(names[1], names[1], shape, size, tuple(float(x) for x in 0.5 * (B + C)), rot,
                                    "dynamic", 7, (4, 5, 6, 7, 8, 9, 10, 11), MASS[kind], (0.9, 0.95), 0.5))
            for k, (a, b, at) in enumerate(((anchor.name if anchor else None, names[0], K), (names[0], names[1], B))):
                if a is None:
                    continue
                joints.append(Joint(f"J_{names[k]}", a, b, location=tuple(float(x) for x in at),
                                    rotation=euler_for_axis((B - K) if k == 0 else (C - B)),
                                    rot_lo=(-sw, -sw, -0.3), rot_hi=(sw, sw, 0.3), spring_rot=(3.0, 3.0, 1.5)))
        span = float(np.linalg.norm(B - K))
        V = np.vstack([p.v for p in patches])
        least = float(surf.closest(V)[0].min())
        info.append({"kind": kind, "cord": {"top": A, "lies_to": K, "end": B, "length": lie + span,
                                            "out_deg": float(np.degrees(a1))},
                     "top": float(V[:, 2].max()), "bottom": float(V[:, 2].min()), "middle": 0.5 * (B + C),
                     "across": float(np.ptp(V @ R[:, 0])), "deep": float(np.ptp(V @ R[:, 1])),
                     "tall": float(np.ptp(V @ R[:, 2])), "lean_deg": float(np.degrees(a2)), "gap": least,
                     "azimuth": float(np.degrees(theta))})
        land[f"hang.{i}"] = 0.5 * (B + C)
        ctx.log(f"{kind} at {np.degrees(theta):.0f} deg: cord from z {A[2]:.3f}, lying on what it hangs over "
                f"{lie * 1000:.0f} mm, to {B[2]:.3f} ({np.degrees(a1):.0f} deg out), {info[-1]['across'] * 1000:.0f} x "
                f"{info[-1]['deep'] * 1000:.0f} x {info[-1]['tall'] * 1000:.0f} mm, z {info[-1]['bottom']:.3f} .. "
                f"{info[-1]['top']:.3f}, leaning out {np.degrees(a2):.0f} deg, {least * 1000:.1f} mm off what it lies on")

    mats = []
    jc, lc = cols.get("jug"), cols.get("book")
    cord_col = (jc or lc)["cord"]
    mats.append(material(ctx, MATS[CORD], "cord", cord_col, cord_col * 0.75, None, 0.3))
    mats.append(material(ctx, MATS[JUG], "jug", jc["glaze"] if jc else np.ones(3), jc["glaze_shadow"] if jc else
                         np.full(3, 0.8), jug_picture(jc, *pictures["jug"][:1], float(pictures["jug"][1]["mark"]))
                         if jc else None, 0.5, specular=(0.35, 0.35, 0.33), shininess=40.0))
    mats.append(material(ctx, MATS[CORK], "cork", jc["cork"] if jc else np.ones(3),
                         (jc["cork"] if jc else np.ones(3)) * 0.72, None, 0.4))
    mats.append(material(ctx, MATS[PAPER], "ledger", lc["paper"] if lc else np.ones(3), lc["paper_shadow"] if lc else
                         np.full(3, 0.8), book_picture(lc, pictures["book"][0], int(pictures["book"][1]["lines"]),
                                                       int(ctx.rng.integers(0, 2 ** 31))) if lc else None, 0.45))
    mats.append(material(ctx, MATS[COVER], "ledger_cover", lc["cover"] if lc else np.ones(3),
                         (lc["cover"] if lc else np.ones(3)) * 0.7, None, 0.45))
    mesh = soup.mesh()
    used = set(mesh.mats)
    return Part("hangings", meshes=[mesh], materials=[m for m in mats if m.name in used], bones=bones, bodies=bodies,
                joints=joints, frames={"吊り飾り": [b.name for b in bones]},
                info={"items": info, "landmarks": land})
