"""Ribbed crew socks (part `socks`, numpy only): from the toes up the shin to a rolled cuff, over tucked-in pants. A
shell cut from the body's own skin (outfit_shell): the foot and the ankle snug (inside any shoe), standing further off the
skin above `snug` so pants tucked in stay under it. Where an artist's toes touch (the girl base's do) no offset can be
outside both, so the closest distance to the skin is measured above the ankle joints.

Spec `[socks]` (sizes in metres x S, S = head_tip.z / 1.7 with leg_extra left out):
  top     0.105   the cuff's edge above the ankle joints
  snug    0.0020  off the skin up to `rise` 0.052 above the ankle joints (inside a shoe), then `loose` 0.0050 (over pants)
  roll    0.0045  the cuff's edge closed back towards the skin; `relax` 6, `gap` 0.0015
`[colors.socks]` knit (cream). Material 靴下 (ribbed), double sided, outlined dark."""
import numpy as np

from .. import spec as SP
from ..build import builder
from ..part import Part
from . import outfit as OU
from . import outfit_geo as G
from . import outfit_shell as SH
from . import outfit_wear as WR

DEFAULTS = {"top": 0.105, "snug": 0.0020, "rise": 0.052, "loose": 0.0050, "roll": 0.0045, "relax": 6, "gap": 0.0015}
COLORS = {"knit": "#e1dfd7"}
MAT = "靴下"


def socks(fit, body_part, cfg, soup, mat=0):
    """Build both socks into `soup`. Returns info: the cuff height, the vertex count, the closest distance to the skin
    above the ankle joints (`min_gap`)."""
    c = SP.merge(DEFAULTS, cfg)
    S, L = fit.S, fit.L
    z_ank = 0.5 * (float(L["ankle.L"][2]) + float(L["ankle.R"][2]))
    z_top = z_ank + c["top"] * S
    sh = SH.Shell.from_part(body_part)
    sh.fields.update(z=sh.verts[:, 2].copy(), arm=sh.share(SH.ARM))
    sh = sh.clip(0.5 - sh.fields["arm"])
    sh = sh.clip(z_top - sh.fields["z"])
    t = G.smoothstep(np.clip((sh.fields["z"] - z_ank - c["rise"] * S) / (0.02 * S), 0.0, 1.0))
    dist = (c["snug"] + (c["loose"] - c["snug"]) * t) * S
    P = sh.offset(dist, relax=c["relax"], fit=fit, gap=c["gap"] * S)
    u = SH.limb_angle(sh.verts, L, SH.LEG_CHAIN, ref=(0.0, -1.0, 0.0))
    tops = [lp for lp in sh.loops() if float(np.mean(sh.fields["z"][lp])) > z_top - 0.01 * S]
    soup.add(sh.patch(P, u, -sh.verts[:, 2], u_rep=round(0.30 / WR.TILE), v_scale=1.0 / WR.TILE, tag="socks", rims=tops,
                      rim_depth=c["roll"] * S), mat)
    leg = sh.fields["z"] > z_ank
    return {"z_top": z_top, "verts": len(P), "cuffs": len(tops), "min_gap": float(fit.skin.closest(P[leg])[0].min())}


@builder("socks", needs=("body",))
def build(ctx):
    cfg = SP.merge(DEFAULTS, ctx.cfg)
    fit = OU.make_fit(ctx)
    col = WR.colours(ctx, "socks", COLORS)
    soup = G.Soup("socks", [MAT])
    info = socks(fit, ctx.need("body"), cfg, soup)
    mat = WR.material(ctx, MAT, "knit", col["knit"], WR.rib(col["knit"], ribs=10, depth=0.10), shadow=(0.80, 0.78, 0.86))
    ctx.log(f"cuffs at {info['z_top']:.3f} m ({info['cuffs']}), above the ankles at least {info['min_gap'] * 1000:.1f} mm "
            f"off the skin")
    return Part("socks", meshes=[soup.mesh()], materials=[mat], info=info)
