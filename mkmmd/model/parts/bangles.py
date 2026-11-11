"""Bangles (part `bangles`, numpy only): rings worn round a limb, a bracelet, an anklet, an armlet.

`[bangles] rings = [{on, toward, at, width, thick, offset, color}, ...]`: each ring sits round the limb that runs from
the landmark `on` toward the landmark `toward` (`ankle.L` toward `knee.L`, `wrist.R` toward `elbow.R` ...), `at` metres
from `on`. It follows the skin's own section there, a little rounder (a ring of metal does not dip into the hollows
beside a tendon), stands `offset` off it, is `width` along the limb and `thick`, with rounded edges, and takes the
weights of the skin under it. `color` names a key of `[colors.bangles]` (`gold`, `silver`, or one of your own); each
colour in use is a material, バングル<key>. Sizes are metres x S (S = head_tip.z / 1.7 with leg_extra left out).
Landmarks `bangle.<i>`: the rings' middles."""
import numpy as np

from .. import spec as SP
from ..build import builder
from ..part import Part
from . import outfit as OU
from . import outfit_fit as F
from . import outfit_geo as G
from . import outfit_wear as WR

DEFAULTS = {"rings": [], "segments": 40}
RING = {"on": None, "toward": None, "at": 0.03, "width": 0.008, "thick": 0.0022, "offset": 0.0015, "color": "gold"}
COLORS = {"gold": "#f0c050", "silver": "#dfe3ea"}
MAT = "バングル"


def ring(fit, skin, r, m):
    """Rings (4, m, 3) of one bangle, top to bottom along the limb, and its middle (3,)."""
    S, L = fit.S, fit.L
    for key in ("on", "toward"):
        if not r[key] or not L.has(r[key]):
            raise ValueError(f"[bangles] ring {key} = {r[key]!r}: no such landmark")
    a, b = np.asarray(L[r["on"]], float), np.asarray(L[r["toward"]], float)
    t = G.unit(b - a)
    ex = G.unit(np.cross(t, [0.0, 1.0, 0.0] if abs(t[1]) < 0.9 else [1.0, 0.0, 0.0]))
    ey = np.cross(t, ex)
    c0 = a + t * float(r["at"]) * S
    w, th, off = float(r["width"]) * S, float(r["thick"]) * S, float(r["offset"]) * S
    phi = G.ring_theta(m)
    rows = []
    for d, out in ((0.5 * w, off), (0.5 * w - 0.3 * th, off + th), (-0.5 * w + 0.3 * th, off + th), (-0.5 * w, off)):
        cen, rad = skin.section(c0 + t * d, ex, ey, m, "near", rmin=0.005 * S, rmax=0.12 * S)
        if cen is None:
            raise ValueError(f"[bangles] ring on {r['on']}: no limb to go round {float(r['at']):.3f} m toward {r['toward']}")
        rad = np.maximum(rad, G.smooth_circular(rad, 3, 2))
        rows.append(cen + (rad + out)[:, None] * (np.cos(phi)[:, None] * ex + np.sin(phi)[:, None] * ey))
    return np.stack(rows), c0


@builder("bangles", needs=("body",))
def build(ctx):
    cfg = SP.merge(DEFAULTS, ctx.cfg)
    rings = [SP.merge(RING, r) for r in cfg["rings"]]
    if not rings:
        raise ValueError("[bangles] rings = [{on, toward, at, ...}, ...] is empty: nothing to wear")
    fit = OU.make_fit(ctx)
    got = ctx.get("colors.bangles", {}) or {}
    keys = sorted({str(r["color"]) for r in rings})
    col = WR.colours(ctx, "bangles", {**COLORS, **{k: got[k] for k in got}})
    missing = [k for k in keys if k not in col]
    if missing:
        raise KeyError(f"[bangles] color {missing[0]!r}: no such key in [colors.bangles] (have {', '.join(col)})")
    soup = G.Soup("bangles", [MAT + k for k in keys])
    skin = F.Skin.from_meshes([m for m in ctx.need("body").meshes[:1] if len(m.verts)], "limbs")
    land, m = {}, int(cfg["segments"])
    for i, r in enumerate(rings):
        rows, mid = ring(fit, skin, r, m)
        p = G.loft(rows, True, mat=keys.index(str(r["color"])), tag=f"bangle{i}")
        p = G.orient_outward(p, np.repeat(rows.mean((0, 1))[None], len(p.v), 0))
        p.w = fit.weights_for(p.v, None, default="下半身")
        soup.add(p, p.mat)
        land[f"bangle.{i}"] = mid
    mats = [WR.material(ctx, MAT + k, f"bangle_{k}", col[k], None, shadow=(0.82, 0.70, 0.52), specular=(0.55, 0.5, 0.35),
                        shininess=40.0, edge_size=0.4) for k in keys]
    ctx.log(f"{len(rings)} rings: " + ", ".join(f"{r['on']} ({r['color']})" for r in rings))
    return Part("bangles", meshes=[soup.mesh()], materials=mats, info={"landmarks": land})
