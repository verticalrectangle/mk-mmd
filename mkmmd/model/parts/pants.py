"""Skinny pants (part `pants`, numpy only): smooth woven cloth from the waist to the lower shin, tight on the skin, meant
to be tucked into socks or boots. A shell cut from the body's own skin (outfit_shell), so it keeps the skin's weights and
bends with her hips and knees; smoothed so the cloth bridges the skin's small dips (the navel, the crease under the
buttocks) instead of following them.

Spec `[pants]` (sizes in metres x S, S = head_tip.z / 1.7 with leg_extra left out):
  top        0.080   the waist edge above the hip joints (landmark leg.L): under a sweater's waistband
  hem        0.080   the leg ends this far above the ankle joints (tucked into a sock of a higher top)
  offset     0.0025  off the skin; `gap` 0.0018 the least, `relax` 4 smoothing passes
  roll       0.0020  the waist edge closed back towards the skin (reads as a waistband seam)
  ink        drawn lines (outfit_ink), `width` 0.0026 wide: the groin from the crotch up each thigh's front (where the
             leg's weights take over from the hips'), a fold across the lower belly, creases behind each knee and one
             beside the knee cap (`enabled` true, `seed`)
`[colors.pants]` cloth (charcoal), ink (the lines). Materials ズボン (double sided, outlined dark) and ズボン線 (ink)."""
import numpy as np

from .. import spec as SP
from ..build import builder
from ..part import Part
from . import outfit as OU
from . import outfit_geo as G
from . import outfit_ink as INK
from . import outfit_shell as SH
from . import outfit_wear as WR

DEFAULTS = {"top": 0.080, "hem": 0.080, "offset": 0.0025, "gap": 0.0018, "relax": 4, "roll": 0.0020,
            "ink": {"enabled": True, "width": 0.0026, "seed": 3}}
COLORS = {"cloth": "#312d33", "ink": "#0f0c12"}
MAT = "ズボン"
MAT_INK = "ズボン線"


def _near(a, centre, half):
    """|a - centre| <= half for turn fractions a (wrapping)."""
    return np.abs((np.asarray(a, float) - centre + 0.5) % 1.0 - 0.5) <= half


def ink(sh, P, fit, z_top, rng):
    """The pants' drawn lines (outfit_ink lines, widths as shares of [pants.ink] width): the groin, a fold across the
    lower belly, the creases behind the knees and a line beside each knee cap."""
    S, L = fit.S, fit.L
    sh.fields.update(az=SH.azimuth(sh.verts), la=SH.limb_angle(sh.verts, L, SH.LEG_CHAIN, ref=(0.0, -1.0, 0.0)),
                     s=SH.both_sides(sh.verts, L, SH.LEG_CHAIN))
    lines, widths = [], []

    def add(ls, w):
        lines.extend(ls)
        widths.extend([w] * len(ls))
    hip = float(L["leg.L"][2])
    # the groin: where the leg takes over from the hips, on the front, from the crotch up 7 cm x S
    add(INK.arcs(sh, P, sh.fields["leg"], 0.6, lambda fl: _near(fl["az"], 0.0, 0.14) & (fl["z"] > hip - 0.09 * S)
                 & (fl["z"] < hip - 0.02 * S), fields=("az", "z")), 1.0)
    # a fold across the lower belly, under the sweater's waistband
    zf, af = 0.5 * (hip + z_top) - rng.uniform(0.0, 0.01) * S, rng.uniform(-0.02, 0.02)
    add(INK.arcs(sh, P, sh.fields["z"], zf, lambda fl: _near(fl["az"], af, 0.05), fields=("az",)), 0.9)
    for side, sg in (("L", 1.0), ("R", -1.0)):
        knee = np.asarray(L[f"knee.{side}"], float)
        s_knee = float(np.linalg.norm(knee - np.asarray(L[f"leg.{side}"], float)))
        on = (sh.fields["leg"][sh.tris] > 0.5).all(1) & (sg * sh.verts[sh.tris][:, :, 0] > 0.0).all(1)
        for _ in range(3):                                      # behind the knee
            c, half = 0.5 + rng.uniform(-0.05, 0.05), rng.uniform(0.06, 0.11)
            add(INK.arcs(sh, P, sh.fields["s"], s_knee + rng.uniform(-0.012, 0.018) * S,
                         lambda fl, c=c, half=half: _near(fl["la"], c, half), fields=("la",), tris=on), 0.8)
        c = float(SH.limb_angle((knee + np.array([0.035 * sg, -0.035, 0.0]))[None], L, SH.LEG_CHAIN,
                                ref=(0.0, -1.0, 0.0))[0])        # beside the knee cap, towards her outer side
        add(INK.arcs(sh, P, sh.fields["s"], s_knee - rng.uniform(0.005, 0.015) * S,
                     lambda fl, c=c: _near(fl["la"], c, 0.05), fields=("la",), tris=on), 0.8)
    return lines, widths


def pants(fit, body_part, cfg, soup):
    """Build the pants into `soup` (material 0 the cloth, 1 the ink). Returns info: the waist and hem heights, the
    vertex count, the closest distance of the cloth to the skin, the number of drawn lines."""
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
                      rim_depth=c["roll"] * S), 0)
    soup.add(sh.patch(P, u, v, leg_t, u_rep=reps[1], v_scale=1.0 / WR.TILE, tag="legs"), 0)
    lines = []
    if c["ink"]["enabled"]:
        lines, widths = ink(sh, P, fit, z_top, np.random.default_rng(int(c["ink"]["seed"])))
        soup.add(INK.strokes(sh.bones, lines, np.asarray(widths) * c["ink"]["width"] * S, lift=0.0004 * S), 1)
    return {"z_top": z_top, "z_hem": z_hem, "verts": len(P), "min_gap": sh.gap(P, fit), "lines": len(lines)}


@builder("pants", needs=("body",))
def build(ctx):
    cfg = SP.merge(DEFAULTS, ctx.cfg)
    fit = OU.make_fit(ctx)
    col = WR.colours(ctx, "pants", COLORS)
    soup = G.Soup("pants", [MAT, MAT_INK])
    info = pants(fit, ctx.need("body"), cfg, soup)
    mats = [WR.material(ctx, MAT, "cloth", col["cloth"], WR.twill(col["cloth"]), shadow=(0.70, 0.66, 0.78)),
            WR.ink_material(ctx, MAT_INK, "ink", col["ink"])]
    mesh = soup.mesh()
    ctx.log(f"waist {info['z_top']:.3f} m, hem {info['z_hem']:.3f} m, closest to the skin {info['min_gap'] * 1000:.1f} mm, "
            f"{info['lines']} drawn lines")
    return Part("pants", meshes=[mesh], materials=[m for m in mats if m.name in mesh.mats], info=info)
