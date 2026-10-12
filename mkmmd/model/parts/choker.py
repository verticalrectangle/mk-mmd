"""A choker (part `choker`, numpy only): a thin band round the neck with a flower charm at the throat.

The band follows the neck's own cross-sections at its two edges (the body's neck and the head part's, which carries the
neck up to the jaw), stands `offset` off the skin and has rounded edges; the flower is `petals` flattened ellipsoids
round a domed centre, facing out of the neck at `at` degrees from the front towards her left. Everything takes the skin
weights of the body under it (the band bends with the neck, the flower moves with the point it sits on).

Spec `[choker]` (sizes in metres x S, S = head_tip.z / 1.7 with leg_extra left out): `z` where the band's middle sits
between the neck joint (landmark neck, 0) and the head joint (landmark head, 1), `width` 0.011, `offset` 0.0016,
`thick` 0.0014; `flower` {petals 7, length 0.0085, width 0.0052, centre 0.0030, at 14 (degrees), cup 18 (degrees the
petals tip out)}; `[colors.choker]` band, petal, centre. Materials チョーカー, チョーカー花, チョーカー花芯."""
import numpy as np

from .. import spec as SP
from ..build import builder
from ..part import Part
from . import outfit as OU
from . import outfit_fit as F
from . import outfit_geo as G
from . import outfit_wear as WR

DEFAULTS = {"z": 0.30, "width": 0.011, "offset": 0.0016, "thick": 0.0014, "segments": 48,
            "flower": {"petals": 7, "length": 0.0085, "width": 0.0052, "centre": 0.0030, "at": 14.0, "cup": 18.0}}
COLORS = {"band": "#1b171c", "petal": "#70d2d9", "centre": "#e0db76"}
MAT_BAND, MAT_PETAL, MAT_CENTRE = "チョーカー", "チョーカー花", "チョーカー花芯"


def _ring(skin, y, z, m):
    """The neck's cross-section at height z: centre (3,) and the outline points (m, 3), counter-clockwise from +x."""
    c0 = np.array([0.0, y, z])
    ex, ey = np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])
    c, r = skin.section(c0, ex, ey, m, "near", rmin=0.01, rmax=0.12)
    if c is None:
        raise ValueError(f"choker: no neck section at z = {z:.3f}")
    phi = G.ring_theta(m)
    return c, c + r[:, None] * (np.cos(phi)[:, None] * ex + np.sin(phi)[:, None] * ey)


def choker(fit, cfg, neck, soup):
    """The band (material 0) and the flower (1 petals, 2 centre) into `soup`; `neck` is a Skin of the body and the head
    part's neck. Returns info: the band's height, the flower's centre and the closest distance to the skin."""
    c = SP.merge(DEFAULTS, cfg)
    S, L = fit.S, fit.L
    z0, z1 = float(L["neck"][2]), float(L["head"][2])
    zc = z0 + c["z"] * (z1 - z0)
    y = float(L["neck"][1])
    w, off, th, m = c["width"] * S, c["offset"] * S, c["thick"] * S, int(c["segments"])
    rings = []
    for dz, out in ((-0.5 * w, off), (-0.5 * w + 0.3 * th, off + th), (0.5 * w - 0.3 * th, off + th), (0.5 * w, off)):
        cen, P = _ring(neck, y, zc + dz, m)
        d = P - cen
        d[:, 2] = 0.0
        n = G.unit(d)
        rings.append(P + n * out)
    rings = np.stack(rings[::-1])                                     # top to bottom: outward faces
    band = G.loft(rings, True, mat=0, tag="band")
    band = G.orient_outward(band, np.repeat(rings.mean((0, 1))[None], len(band.v), 0))
    band.w = fit.weights_for(band.v, None, default="首")
    soup.add(band, 0)
    # the flower: at `at` degrees from the front towards her left, facing out of the neck
    fl = c["flower"]
    ang = np.radians(-90.0 + fl["at"])
    cen, P = _ring(neck, y, zc, m)
    k = int(np.argmin(np.abs(((G.ring_theta(m) - ang) + np.pi) % (2 * np.pi) - np.pi)))
    nrm = G.unit(np.array([P[k, 0] - cen[0], P[k, 1] - cen[1], 0.0]))
    up = np.array([0.0, 0.0, 1.0])
    side = np.cross(up, nrm)
    base = P[k] + nrm * (off + th + 0.0008 * S)
    plen, pw, rc = fl["length"] * S, fl["width"] * S, fl["centre"] * S
    cup = np.radians(fl["cup"])
    pts = []
    for i in range(int(fl["petals"])):
        a = 2.0 * np.pi * i / fl["petals"] + np.pi / 2.0
        radial = np.cos(a) * side + np.sin(a) * up
        along = G.unit(radial * np.cos(cup) + nrm * np.sin(cup))
        across = G.unit(np.cross(nrm, radial))
        normal = G.unit(np.cross(along, across))
        ctr = base + along * (0.5 * plen + 0.3 * rc)
        petal = G.ellipsoid(ctr, (0.5 * plen, 0.5 * pw, 0.18 * pw), axes=np.stack([along, across, normal]), n_lat=4,
                            n_lon=10, mat=1, tag="petal")
        pts.append(petal)
    centre = G.ellipsoid(base + nrm * 0.25 * rc, (rc, rc, 0.55 * rc), axes=np.stack([side, up, nrm]), n_lat=5, n_lon=12,
                         mat=2, tag="centre")
    pts.append(centre)
    wk = fit.weights_for(base[None], None, default="首")
    for p in pts:
        p.w = {b: np.full(len(p.v), float(a[0])) for b, a in wk.items()}
        soup.add(p, p.mat)
    gap = float(neck.closest(band.v)[0].min())
    return {"z": zc, "flower": base, "min_gap": gap}


@builder("choker", needs=("body", "head"))
def build(ctx):
    cfg = SP.merge(DEFAULTS, ctx.cfg)
    fit = OU.make_fit(ctx)
    col = WR.colours(ctx, "choker", COLORS)
    soup = G.Soup("choker", [MAT_BAND, MAT_PETAL, MAT_CENTRE])
    meshes = [m for p in (ctx.need("body"), ctx.need("head")) for m in p.meshes[:1] if len(m.verts)]
    neck = F.Skin.from_meshes(meshes, "neck")
    info = choker(fit, cfg, neck, soup)
    mats = [WR.material(ctx, MAT_BAND, "band", col["band"], None, shadow=(0.70, 0.68, 0.76), specular=(0.15, 0.15, 0.17),
                        shininess=24.0, edge_size=0.5),
            WR.material(ctx, MAT_PETAL, "petal", col["petal"], None, shadow=(0.76, 0.84, 0.88), edge_size=0.5),
            WR.material(ctx, MAT_CENTRE, "centre", col["centre"], None, shadow=(0.86, 0.82, 0.70), edge_size=0.4)]
    ctx.log(f"band at {info['z']:.3f} m, {info['min_gap'] * 1000:.1f} mm off the skin")
    return Part("choker", meshes=[soup.mesh()], materials=mats, info=info)
