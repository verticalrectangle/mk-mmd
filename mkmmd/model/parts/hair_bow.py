"""A ribbon bow tied on the side of the head (part `hair_bow`, numpy only): a knot, two puffy loops and two long tails
with V-cut ends that hang on chains and flutter.

The bow sits on the hair's outer surface where a ray from the head's centre at (`at`: azimuth from the front towards her
left, polar angle from the top, degrees) leaves the hair; it faces out of the head. Its pieces are the braid bows'
(hair_braids_bow: knot_block, wing_block, tail_block) at the size given here. The tails hang from the knot: each step
heads down (and `splay` apart, `out` away from the head) and is pushed `margin` off whatever is built before the bow (the
body, the garments, the hair), then a soft wave (`sway`, `turns`) runs along it; each tail is a chain of `bones` bones
(リボン横A1.., リボン横B1.., kind "ribbon": light, loose, the ribbon group) hanging from the head bone. Knot and loops are
rigid on the head.

Spec `[hair_bow]` (metres; DEFAULTS below) and `[colors.hair_bow]` ribbon. Material 横リボン."""
import numpy as np

from .. import spec as SP
from ..build import builder
from ..part import Part
from . import outfit_fit as OF
from . import outfit_wear as WR
from .hair_braids_bow import knot_block, tail_block, wing_block
from .hair_fit import dirs
from .hair_geo import MeshAccum, arclen, resample, smooth_polyline, unit
from .hair_rig import Rig

DEFAULTS = {"at": [-74.0, 50.0], "lift": 0.002, "span": 0.052, "loop_h": 0.030, "puff": 0.014, "curl": 0.18, "roll": -12.0,
            "knot": [0.010, 0.009, 0.008],
            "tails": {"length": 0.32, "width": 0.019, "thick": 0.0018, "notch": 0.016, "bones": 6, "radius": 0.006,
                      "splay": 0.22, "out": 0.03, "margin": 0.010, "step": 0.008, "sway": 0.018, "turns": 2.2,
                      "ratio": 0.80}}
COLORS = {"ribbon": "#76497b"}
MAT = "横リボン"


def _surface_point(skin, centre, d):
    """Where the ray from `centre` along unit `d` last leaves `skin` (the hair), or None."""
    up = np.array([0.0, 0.0, 1.0])
    v = unit(up - np.dot(up, d) * d) if abs(np.dot(up, d)) < 0.99 else unit(np.cross(d, [1.0, 0.0, 0.0]))
    r = skin.outline(centre, d, v, 64, "far", rmin=0.02, rmax=0.40, despike=0)
    return None if r is None else centre + d * float(r[0])


def hang(obstacles, hull, start, heading, out, length, step, margin, iters=3):
    """A ribbon's centreline from `start`: each step goes along `heading` plus `out(p)` (a pull away from the head), is
    moved out of the hair by `hull(p)` and pushed `margin` off the `obstacles` (outfit_fit.Skin: the body and the
    garments) the shortest way out. Returns (n, 3) points, `length` long."""
    pts = [hull(np.asarray(start, float))]
    h = unit(np.asarray(heading, float))
    s = 0.0
    while s < length:
        p = hull(pts[-1] + step * unit(h + out(pts[-1])))
        for _ in range(iters):
            d, q, tri, bary = obstacles.closest(p[None])
            if d[0] >= margin:
                break
            way = unit(p - q[0]) * np.sign(d[0]) if abs(d[0]) > 1e-5 else obstacles.outward(tri, bary)[0]
            p = p + way * (margin - d[0])
        s += float(np.linalg.norm(p - pts[-1]))
        pts.append(p)
    return np.array(pts)


def hair_hull(hair_skin, centre, margin):
    """p -> p moved `margin` past the hair's outer outline: along the ray from the head's `centre` for points above it,
    horizontally from the vertical axis through it below (the hair hangs: there it is a column round that axis)."""
    up = np.array([0.0, 0.0, 1.0])

    def push(p):
        o = np.array(centre, float)
        o[2] = min(o[2], p[2])                                    # below the centre: level, from the axis
        d = p - o
        r = float(np.linalg.norm(d))
        if r < 1e-6:
            return p
        u = d / r
        v = unit(up - np.dot(up, u) * u) if abs(np.dot(up, u)) < 0.99 else unit(np.cross(u, [1.0, 0.0, 0.0]))
        rr = hair_skin.outline(o, u, v, 16, "far", rmin=0.01, rmax=0.40, despike=0)
        if rr is None or r >= rr[0] + margin:
            return p
        return o + u * (rr[0] + margin)
    return push


def bow(ctx_parts, head_centre, cfg, acc, rig, anchor_body=None):
    """The bow into the MeshAccum `acc` (material 0) and its tails' chains into `rig`. `ctx_parts`: the parts built so
    far (the hair to tie it on: the tails hang outside its outline, `hair_hull`; and everything but the head, which
    they stay off). Returns info."""
    c = SP.merge(DEFAULTS, cfg)
    hair = [m for name, p in ctx_parts.items() if name == "hair" for m in p.meshes if len(m.verts)]
    rest = [m for name, p in ctx_parts.items() if name not in ("hair", "head") for m in p.meshes if len(m.verts)]
    if not hair:
        raise ValueError("hair_bow: no hair part to tie the bow on (list it after the hair in [model] parts)")
    hair_skin = OF.Skin.from_meshes(hair, "hair")
    obstacles = OF.Skin.from_meshes(rest, "body")
    th, ph = np.radians(c["at"][0]), np.radians(c["at"][1])
    d = unit(dirs(th, ph))
    hit = _surface_point(hair_skin, head_centre, d)
    if hit is None:
        raise ValueError(f"hair_bow: the ray at {c['at']} misses the hair")
    n = d
    t = unit(np.array([0.0, 0.0, -1.0]) - np.dot([0.0, 0.0, -1.0], n) * n)
    w = unit(np.cross(t, n))
    r = np.radians(c["roll"])
    t, w = unit(np.cos(r) * t + np.sin(r) * w), unit(-np.sin(r) * t + np.cos(r) * w)
    kt, kw, kd = c["knot"]
    C = hit + n * (kd + c["lift"])
    blocks = [knot_block(C, t, n, kt, kw, kd)]
    for sgn in (1.0, -1.0):
        blocks.append(wing_block(C + sgn * w * kw * 0.5, sgn * w, t, n, c["span"], c["loop_h"], c["puff"], curl=c["curl"]))
    for b in blocks:
        acc.add(b.verts, b.faces, b.uv, 0, weights={"頭": np.ones(len(b.verts))})
    tc = c["tails"]
    info = {"centre": C, "tails": {}}
    hull = hair_hull(hair_skin, head_centre, tc["margin"])

    def out(p):
        return tc["out"] * unit(np.array([p[0] - head_centre[0], p[1] - head_centre[1], 0.0]))
    w_h = unit(np.array([w[0], w[1], 0.0]))                    # across the bow, level: the tails splay along it
    for k, (sgn, tag) in enumerate(((1.0, "A"), (-1.0, "B"))):
        L = tc["length"] * (1.0 if k == 0 else tc["ratio"])
        start = C - t * kt * 0.6 + sgn * w * kw * 0.3
        heading = unit(np.array([0.0, 0.0, -1.0]) + sgn * tc["splay"] * w_h)
        P = hang(obstacles, hull, start, heading, out, L, tc["step"], tc["margin"])
        s = arclen(P)
        u = s / max(s[-1], 1e-9)
        P = P + (sgn * w_h)[None, :] * (tc["sway"] * np.sin(tc["turns"] * np.pi * u) * u)[:, None]
        P = smooth_polyline(resample(P, max(int(s[-1] / 0.006), 8)), 2)
        sp = arclen(P)                         # the V-cut pulls the end row's middle back by `notch`: no other row may
        P = P[(sp <= sp[-1] - 1.6 * tc["notch"]) | (sp == sp[-1])]       # lie within that, or the ribbon folds over
        nb = int(tc["bones"])
        pts = resample(P, nb + 1)
        ch = rig.chain(f"リボン横{tag}", pts, "頭", kind="ribbon", radius=tc["radius"], anchor_body=anchor_body,
                       name_en=f"side_ribbon_{tag}")
        face = unit(np.array([n[0], n[1], 0.0]) + 0.3 * sgn * w)
        tb = tail_block(P, face, tc["width"], tc["thick"], tc["notch"], w_end=1.15)
        acc.add(tb.verts, tb.faces, tb.uv, 0, weights=ch.weights(tb.verts))
        clear = float(obstacles.closest(P[3:])[0].min()) if len(P) > 3 else 0.0
        info["tails"][tag] = {"length": float(arclen(P)[-1]), "lowest_z": float(P[:, 2].min()), "clear": clear}
    return info


@builder("hair_bow", needs=("body", "head", "hair"))
def build(ctx):
    cfg = SP.merge(DEFAULTS, ctx.cfg)
    col = WR.colours(ctx, "hair_bow", COLORS)
    head = ctx.need("head")
    centre = np.asarray(head.info.get("head_center"), float)
    acc = MeshAccum("hair_bow", [MAT])
    rig = Rig()
    hb = ctx.find_body("頭")
    info = bow(ctx.parts, centre, cfg, acc, rig, hb.name if hb is not None else None)
    mat = WR.material(ctx, MAT, "ribbon", col["ribbon"], None, shadow=(0.70, 0.62, 0.78), specular=(0.18, 0.16, 0.20),
                      shininess=18.0, edge_size=0.6)
    ctx.log(f"bow at {np.round(info['centre'], 3).tolist()}, tails {info['tails']['A']['length']:.2f} m / "
            f"{info['tails']['B']['length']:.2f} m, down to {min(t['lowest_z'] for t in info['tails'].values()):.3f} m")
    return Part("hair_bow", meshes=[acc.to_mesh()], materials=[mat], bones=rig.bones, bodies=rig.bodies, joints=rig.joints,
                frames={"リボン": [b.name for b in rig.bones]}, info={"tails": info["tails"]})
