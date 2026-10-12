"""High-top sneakers (part `sneakers`, numpy only): a canvas upper closed over the instep and up past the ankle, a
padded rolled collar, a platform rubber sole, and a painted face: cream rubber foxing round the bottom and a big toe
cap, speckled, laces across the front, a stripe on each side running back from the cap and up the heel, fading between
two colours, tread marks round the sole.

The shoe is the Mary-Jane's machinery in outfit_legs fitted to the foot (Foot, build_upper, build_collar, build_sole) with
a sneaker's settings: no throat opening, the top line high round the ankle, no strap. Its face is one texture in the
upper's own grid coordinates: u round the foot from the heel (u = 0.5 at the toe), v from the welt (0) to the top line
(1), so the cap, laces and stripes stay where they belong on any foot.

Spec `[sneakers]`: `shoe` (outfit_legs DEFAULTS["shoe"] keys, merged over SHOE below) and the face: `foxing` (v),
`cap` (v at the toe, degrees round), `speckles` (count, seed), `laces` (count, v range, half_deg, width), `stripe`
(front / back: [degrees from the toe, v] of its middle line at each end, width in v, taper), `tread` (v range of the
ticks on the sole); `[colors.sneakers]` upper, cap, sole, lace, speckle, stripe_front, stripe_back, tread. Materials
スニーカー (upper and collar, painted) and スニーカー底 (sole)."""
import numpy as np

from .. import spec as SP
from ..build import builder
from ..part import Part
from . import outfit as OU
from . import outfit_geo as G
from . import outfit_legs as LG
from . import outfit_wear as WR

SHOE = {"collar": {"az": [0.0, 45.0, 90.0, 135.0, 180.0], "h": [0.044, 0.042, 0.040, 0.037, 0.040]},
        "collar_flare": 0.0035, "collar_r": 0.0045, "throat": {"w_start": 0.0, "w_end": 0.0},
        "ease": 0.0042, "min_gap": 0.0034}
DEFAULTS = {"shoe": {}, "foxing": 0.16, "cap": {"v": 0.46, "deg": 68.0},
            "speckles": {"count": 70, "seed": 7},
            "laces": {"count": 4, "v": [0.50, 0.92], "half_deg": 14.0, "width": 0.016},
            "stripe": {"front": [52.0, 0.20], "back": [172.0, 0.86], "width": 0.16, "taper": 0.35},
            "tread": {"v": [0.0, 0.45]}}
COLORS = {"upper": "#efdeeb", "cap": "#f6f6e8", "sole": "#f6f6e8", "lace": "#6e6276", "speckle": "#a9a397",
          "stripe_front": "#93b1f2", "stripe_back": "#c98af0", "tread": "#a8a294"}
MAT_UPPER = "スニーカー"
MAT_SOLE = "スニーカー底"


def face_texture(col, c, size=(1024, 256)):
    """The upper's face (h, w, 4) uint8 in its grid coordinates: x = u (0 the heel, 0.5 the toe, 1 the heel again),
    y = 1 - v (row 0 is the top line). A cream rubber foxing round the bottom (`foxing`, v) and a toe cap over the front
    (`cap`: v at the toe, degrees round), small speckles on the rubber, laces across the front, and on each side a
    stripe running from low at the front (its colour stripe_front) back and up the heel (stripe_back)."""
    W, H = size
    u = (np.arange(W) + 0.5) / W
    v = 1.0 - (np.arange(H) + 0.5) / H
    U, V = np.meshgrid(u, v)
    deg = (U - 0.5) * 360.0                                         # degrees from the toe, + towards her outer side
    img = np.broadcast_to(col["upper"], (H, W, 3)).copy()
    st = c["stripe"]
    (d0, v0), (d1, v1) = st["front"], st["back"]
    a = np.abs(deg)
    t = np.clip((a - d0) / (d1 - d0), 0.0, 1.0)
    mid = v0 + (v1 - v0) * t ** 1.6                                 # flat along the side, sweeping up the heel
    half = 0.5 * st["width"] * (1.0 - st["taper"] * (1.0 - t))
    inside = (a >= d0) & (a <= d1) & (np.abs(V - mid) <= half)
    grad = col["stripe_front"][None, None, :] * (1 - t[..., None]) + col["stripe_back"][None, None, :] * t[..., None]
    img = np.where(inside[..., None], grad, img)
    cp = c["cap"]
    cap_h = cp["v"] * np.sqrt(np.clip(1.0 - (deg / cp["deg"]) ** 2, 0.0, 1.0))
    rubber = (V <= np.maximum(cap_h, c["foxing"]))
    img = np.where(rubber[..., None], col["cap"], img)
    sp = c["speckles"]
    rng = np.random.default_rng(int(sp["seed"]))
    for _ in range(int(sp["count"])):
        dk = rng.uniform(-180.0, 180.0)
        lim = max(cp["v"] * np.sqrt(max(1.0 - (dk / cp["deg"]) ** 2, 0.0)), c["foxing"])
        vk = rng.uniform(0.25, 0.85) * lim
        r = np.hypot((deg - dk) / 0.6, (V - vk) / 0.006)
        img = np.where((r < 1.0)[..., None], col["speckle"], img)
    lc = c["laces"]
    for k in range(int(lc["count"])):
        vk = lc["v"][0] + (lc["v"][1] - lc["v"][0]) * (k + 0.5) / lc["count"]
        bar = (np.abs(deg) <= lc["half_deg"]) & (np.abs(V - vk - 0.012 * np.sin(deg / lc["half_deg"] * 1.6)) <= 0.5 * lc["width"])
        img = np.where(bar[..., None], col["lace"], img)
    return WR._rgba(img)


def sole_texture(col, c, size=(256, 64)):
    """The sole wall's face: cream rubber with short tread ticks in its lower part (x = u in ticks, y = 1 - v)."""
    W, H = size
    u = (np.arange(W) + 0.5) / W
    v = 1.0 - (np.arange(H) + 0.5) / H
    U, V = np.meshgrid(u, v)
    tr = c["tread"]
    tick = (np.abs(U - 0.5) < 0.12) & (V >= tr["v"][0] + 0.08) & (V <= tr["v"][1])
    img = np.where(tick[..., None], col["tread"], col["sole"])
    return WR._rgba(img)


def _grid_uv(theta, K1):
    """Per-column u (from the heel, the seam there) and the column order that makes it increase."""
    u = ((np.asarray(theta, float) + np.pi) % (2 * np.pi)) / (2 * np.pi)
    order = np.argsort(u)
    return u, order


def sneakers(fit, cfg, soup):
    """Both shoes into `soup` (material 0 the upper and collar, 1 the sole). Returns info per side and the closest
    distance of the canvas to the skin."""
    c = SP.merge(DEFAULTS, cfg)
    shoe = SP.merge(SP.merge(LG.DEFAULTS["shoe"], SHOE), c["shoe"])
    res = {}
    for side in ("L", "R"):
        F = LG.Foot(fit, side)
        U = LG.build_upper(F, shoe)
        u, order = _grid_uv(U["theta"], U["P"].shape[1])
        for key in ("P", "n"):
            U[key] = U[key][order]
        U["theta"] = U["theta"][order]
        P = U["P"]
        K1 = P.shape[1]
        uu = np.concatenate([u[order], [u[order][0] + 1.0]])
        vv = np.broadcast_to(np.linspace(0.0, 1.0, K1)[None, :], P.shape[:2])
        upper = LG.grid_patch(P, uu, vv, 0, "upper")
        upper = G.orient_outward(upper, np.repeat(np.array([F.c_a[0], F.c_a[1], U["z_p"]])[None], len(upper.v), 0))
        collar = LG.build_collar(F, shoe, U)
        collar.uv = np.column_stack([np.full(len(collar.uv), 0.25), np.full(len(collar.uv), 0.99)])   # plain canvas
        sole = LG.build_sole(F, shoe, U)
        wall = sole["patches"][0]
        z = np.array([wall.v[i][2] for f in wall.f for i in f])
        u_w = np.array([uv[0] for uv in wall.uv]) * (0.10 / 0.012)                             # a tick per 12 mm
        wall.uv = np.column_stack([u_w, (z - z.min()) / max(np.ptp(z), 1e-6)])
        for p in sole["patches"][1:]:
            p.uv = np.column_stack([np.full(len(p.uv), 0.05), np.full(len(p.uv), 0.95)])         # plain rubber
        donors = LG.foot_donors(F)
        bone = f"{F.name}足首"
        d, _, _, _ = F.srf.closest(np.concatenate([upper.v, collar.v]))
        for p in (upper, collar):
            F.patch_world(p)
            p.w = fit.weights_for(p.v, donors, default=bone)
            soup.add(p, 0)
        for p in sole["patches"]:
            F.patch_world(p)
            p.w = fit.weights_for(p.v, donors, default=bone)
            soup.add(p, 1)
        res[side] = {"min_dist": float(d.min()), "collar_z": float(P[:, -1, 2].max()), "sole_height": sole["height"],
                     "sole_min_z": sole["min_z"]}
    res["min_dist"] = min(res[s]["min_dist"] for s in ("L", "R"))
    return res


@builder("sneakers", needs=("body",))
def build(ctx):
    cfg = SP.merge(DEFAULTS, ctx.cfg)
    fit = OU.make_fit(ctx)
    col = WR.colours(ctx, "sneakers", COLORS)
    soup = G.Soup("sneakers", [MAT_UPPER, MAT_SOLE])
    info = sneakers(fit, cfg, soup)
    mats = [WR.material(ctx, MAT_UPPER, "upper", col["upper"], face_texture(col, cfg), shadow=(0.84, 0.80, 0.90),
                        double_sided=True),
            WR.material(ctx, MAT_SOLE, "sole", col["sole"], sole_texture(col, cfg), shadow=(0.84, 0.82, 0.80))]
    ctx.log(f"collars at {info['L']['collar_z']:.3f} m, soles {info['L']['sole_height'] * 1000:.0f} mm, canvas at least "
            f"{info['min_dist'] * 1000:.1f} mm off the skin")
    return Part("sneakers", meshes=[soup.mesh()], materials=mats, info=info)
