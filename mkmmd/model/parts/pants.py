"""Skinny pants (part `pants`, numpy only): smooth woven cloth from the waist to the lower shin, tight on the skin, meant
to be tucked into socks or boots. A shell cut from the body's own skin (outfit_shell), so it keeps the skin's weights and
bends with her hips and knees; smoothed so the cloth bridges the skin's small dips (the navel, the crease under the
buttocks) instead of following them.

Spec `[pants]` (sizes in metres x S, S = head_tip.z / 1.7 with leg_extra left out):
  top        0.080   the waist edge above the hip joints (landmark leg.L): under a sweater's waistband
  hem        0.080   the leg ends this far above the ankle joints (tucked into a sock of a higher top)
  offset     0.0025  off the skin; `gap` 0.0018 the least, `relax` 4 smoothing passes
  roll       0.0020  the waist edge closed back towards the skin (reads as a waistband seam)
`[colors.pants]` cloth (charcoal). Material ズボン, double sided, outlined dark."""
import numpy as np

from .. import spec as SP
from ..build import builder
from ..part import Part
from . import outfit as OU
from . import outfit_geo as G
from . import outfit_shell as SH
from . import outfit_wear as WR

DEFAULTS = {"top": 0.080, "hem": 0.080, "offset": 0.0025, "gap": 0.0018, "relax": 4, "roll": 0.0020}
COLORS = {"cloth": "#312d33"}
MAT = "ズボン"


def pants(fit, body_part, cfg, soup, mat=0):
    """Build the pants into `soup`. Returns info: the waist and hem heights, the vertex count, the closest distance of
    the cloth to the skin."""
    c = SP.merge(DEFAULTS, cfg)
    S, L = fit.S, fit.L
    sh = SH.Shell.from_part(body_part)
    z_top = float(L["leg.L"][2]) + c["top"] * S
    z_hem = 0.5 * (float(L["ankle.L"][2]) + float(L["ankle.R"][2])) + c["hem"] * S
    sh.fields.update(z=sh.verts[:, 2].copy(), arm=sh.share(SH.ARM), leg=sh.share(SH.LEG))
    sh = sh.clip(0.5 - sh.fields["arm"])                                         # no hands hanging by the hips
    sh = sh.band(sh.fields["z"], z_hem, z_top)
    P = sh.offset(c["offset"] * S, relax=c["relax"], fit=fit, gap=c["gap"] * S)
    on_leg = sh.fields["leg"] >= 0.5
    u = np.where(on_leg, SH.limb_angle(sh.verts, L, SH.LEG_CHAIN, ref=(0.0, -1.0, 0.0)), SH.azimuth(sh.verts))
    v = -sh.verts[:, 2]
    leg_t = on_leg[sh.tris].sum(1) >= 2
    top = sorted(sh.loops(), key=lambda lp: -float(np.mean(sh.fields["z"][lp])))[:1]
    reps = (round(0.80 / WR.TILE), round(0.40 / WR.TILE))
    soup.add(sh.patch(P, u, v, ~leg_t, u_rep=reps[0], v_scale=1.0 / WR.TILE, tag="waist", rims=top,
                      rim_depth=c["roll"] * S), mat)
    soup.add(sh.patch(P, u, v, leg_t, u_rep=reps[1], v_scale=1.0 / WR.TILE, tag="legs"), mat)
    return {"z_top": z_top, "z_hem": z_hem, "verts": len(P), "min_gap": sh.gap(P, fit)}


@builder("pants", needs=("body",))
def build(ctx):
    cfg = SP.merge(DEFAULTS, ctx.cfg)
    fit = OU.make_fit(ctx)
    col = WR.colours(ctx, "pants", COLORS)
    soup = G.Soup("pants", [MAT])
    info = pants(fit, ctx.need("body"), cfg, soup)
    mat = WR.material(ctx, MAT, "cloth", col["cloth"], WR.twill(col["cloth"]), shadow=(0.70, 0.66, 0.78))
    ctx.log(f"waist {info['z_top']:.3f} m, hem {info['z_hem']:.3f} m, closest to the skin {info['min_gap'] * 1000:.1f} mm")
    return Part("pants", meshes=[soup.mesh()], materials=[mat], info=info)
