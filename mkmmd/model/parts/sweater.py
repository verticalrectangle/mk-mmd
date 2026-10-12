"""The off-shoulder sweater (part `sweater`, numpy only): a wide folded rib band round the chest and the upper arms
leaving the shoulders bare, a loose knit body and long sleeves that blouse out over a ribbed waistband and long ribbed
cuffs. Every piece is a shell cut from the body's own skin (outfit_shell): it keeps the skin's weights, so the sweater
moves with her in any pose; the band is where her arm joins the chest too, so it stretches between them as cloth does.

Where the pieces are, as fields on the skin (sizes in metres x S, S = head_tip.z / 1.7 with leg_extra left out):
  band      the soft max of two descents, `chest` below the shoulder joints (landmark arm.L) on the torso and `arm` along
            the upper arm from its joint (outfit_shell.limb_coord): one level line runs across the chest and the back and
            round each upper arm; the band is `height` deep below it, `offset` off the skin, its top edge rolled back to
            the skin (`roll`) so it reads as a thick fold
  body      everything below the band down to the waistband and along the arms to the cuffs: `offset` off the skin at the
            band growing to `blouse` above the waistband and the cuffs (`sleeve` for the arms), tucked over `tuck` to
            `under` inside them (their tops roll down onto it); smoothed (`relax`) so the knit bridges the skin's small
            dips and kept `gap` off it (vertices and triangle centres)
  waistband `above_hip` (its top) over the hip joints (landmark leg.L), `height` deep, snug (`offset`), its hem rolled
  cuffs     `length` up the forearm from the wrist, `past` beyond it, snug, rolled at the hand

Spec: `[sweater]` (DEFAULTS below, tables merge key by key) and `[colors.sweater]` knit (the body and sleeves) and rib (the
band, waistband and cuffs). Materials セーター (knit) and セーターリブ (ribs), double sided, outlined dark."""
import numpy as np

from .. import spec as SP
from ..build import builder
from ..part import Part
from . import outfit as OU
from . import outfit_geo as G
from . import outfit_shell as SH
from . import outfit_wear as WR

DEFAULTS = {
    "band": {"chest": 0.075, "arm": 0.030, "corner": 0.010, "height": 0.066, "offset": 0.010, "roll": 0.008,
             "relax": 8},
    "body": {"offset": 0.008, "blouse": 0.030, "sleeve": 0.018, "tuck": 0.015, "under": 0.0022, "relax": 24, "gap": 0.003},
    "waistband": {"above_hip": 0.095, "height": 0.034, "offset": 0.006, "relax": 4},
    "cuff": {"length": 0.062, "past": 0.004, "offset": 0.006, "relax": 4},
}
COLORS = {"knit": "#e7a7eb", "rib": "#9f75b0"}
MAT_KNIT = "セーター"
MAT_RIB = "セーターリブ"


def fields(sh, fit, c):
    """Per-vertex fields on the skin shell `sh` (dict of (n,)): z, s (arm coordinate), arm (the arm bones' share),
    band (depth below the band's top line), and the sizes they are cut at."""
    S, L = fit.S, fit.L
    V = sh.verts
    s = SH.both_sides(V, L)
    b = c["band"]
    z_top = float(L["arm.L"][2]) - b["chest"] * S
    band = SH.soft_max(z_top - V[:, 2], s - b["arm"] * S, b["corner"] * S)
    arm_len = 0.5 * (SH.limb_length(L, "L") + SH.limb_length(L, "R"))
    z_wb = float(L["leg.L"][2]) + c["waistband"]["above_hip"] * S
    return {"z": V[:, 2].copy(), "s": s, "arm": sh.share(SH.ARM), "band": band, "arm_len": arm_len, "z_wb": z_wb,
            "S": S}


def _uv(sh, fit, centre_y):
    """Texture coordinates per vertex: round the torso (azimuth) or round the upper arm, a turn = 1; v = metres down."""
    az = SH.azimuth(sh.verts, (0.0, centre_y))
    aa = SH.limb_angle(sh.verts, fit.L)
    on_arm = sh.fields["arm"] >= 0.5
    u = np.where(on_arm, aa, az)
    v = np.where(on_arm, sh.fields["s"], -sh.verts[:, 2])
    return u, v, on_arm


def _charts(sh, on_arm):
    """Triangles of the torso chart and of the arm chart (by the majority of their corners)."""
    arm_t = on_arm[sh.tris].sum(1) >= 2
    return ~arm_t, arm_t


def _add(soup, sh, P, fit, centre_y, mat, reps, rims=(), depths=0.0, tag=""):
    """The shell as two patches (torso and arm charts) with tile repeats `reps` = (round the torso, round an arm)."""
    u, v, on_arm = _uv(sh, fit, centre_y)
    t_torso, t_arm = _charts(sh, on_arm)
    vs = 1.0 / WR.TILE
    first = True
    for sel, rep in ((t_torso, reps[0]), (t_arm, reps[1])):
        if not sel.any():
            continue
        soup.add(sh.patch(P, u, v, sel, u_rep=rep, v_scale=vs, tag=tag, rims=rims if first else (),
                          rim_depth=depths if first else 0.0), mat)
        first = False


def _loops_by(sh, key, top=True):
    """The shell's boundary loops ordered by the mean of field `key` (top first: the smallest band depth)."""
    loops = sh.loops()
    return sorted(loops, key=lambda lp: float(np.mean(sh.fields[key][lp])) * (1 if top else -1))


def sweater(fit, body_part, cfg, soup_knit_mat=0, soup_rib_mat=1, soup=None):
    """Build the sweater into `soup` (material 0 knit, 1 rib). Returns info: the pieces' vertex counts, the band's top
    height on the chest, the waistband's top and the closest distance of any piece to the skin."""
    c = SP.merge(DEFAULTS, cfg)
    S = fit.S
    base = SH.Shell.from_part(body_part)
    f = fields(base, fit, c)
    base.fields.update(z=f["z"], s=f["s"], arm=f["arm"], band=f["band"])
    near = np.abs(base.verts[:, 2] - f["z_wb"]) < 0.05 * S
    centre_y = float(np.median(base.verts[near, 1])) if near.any() else 0.0
    upper = base.clip(base.fields["z"] - (f["z_wb"] - c["waistband"]["height"] * S))          # nothing below the waistband
    upper = upper.clip(f["arm_len"] + c["cuff"]["past"] * S - upper.fields["s"])               # nothing past the cuffs
    torso_reps = (round(0.85 / WR.TILE), round(0.20 / WR.TILE))
    info, gaps = {}, []
    # ---- the band
    b = c["band"]
    h = b["height"] * S
    band = upper.band(upper.fields["band"], 0.0, h)
    Pb = band.offset(b["offset"] * S, relax=b["relax"], fit=fit, gap=b["offset"] * S * 0.8)
    loops = _loops_by(band, "band")                                         # the top rolled to the skin, the bottom
    rims = loops[:2]                                                        # onto the knit under it
    _add(soup, band, Pb, fit, centre_y, soup_rib_mat, torso_reps, rims,
         [b["roll"] * S, (b["offset"] - c["body"]["offset"]) * S][:len(rims)], "band")
    info["band_top_z"] = float(band.verts[band.fields["band"] < 1e-4 * S][:, 2].min()) if len(band.verts) else None
    gaps.append(band.gap(Pb, fit))
    # ---- body and sleeves
    bd = c["body"]
    wb, cf = c["waistband"], c["cuff"]
    z_wb, arm_len = f["z_wb"], f["arm_len"]
    tuck = bd["tuck"] * S
    body = upper.clip(upper.fields["band"] - (h - 0.012 * S))
    body = body.clip(body.fields["z"] - (z_wb - 0.010 * S))
    body = body.clip((arm_len - cf["length"] * S + 0.010 * S) - body.fields["s"])
    a = body.fields["arm"]
    # torso: band -> blouse at the waistband's top, tucked in over `tuck`; sleeve: the same towards the cuff
    z_top = float(fit.L["arm.L"][2]) - b["chest"] * S
    t_t = np.clip((body.fields["band"] - h) / max(z_top - z_wb - h, 1e-3), 0.0, 1.0)
    end_s = arm_len - cf["length"] * S
    t_s = np.clip((body.fields["s"] - b["arm"] * S - h) / max(end_s - b["arm"] * S - h, 1e-3), 0.0, 1.0)
    grow_t = bd["offset"] + (bd["blouse"] - bd["offset"]) * G.smoothstep(t_t)
    grow_s = bd["offset"] + (bd["sleeve"] - bd["offset"]) * G.smoothstep(t_s)
    tuck_t = G.smoothstep(np.clip((body.fields["z"] - z_wb) / tuck, 0.0, 1.0))
    tuck_s = G.smoothstep(np.clip((end_s - body.fields["s"]) / tuck, 0.0, 1.0))
    in_wb, in_cf = wb["offset"] - bd["under"], cf["offset"] - bd["under"]           # the knit goes in under the ribs
    off_t = in_wb + (grow_t - in_wb) * tuck_t
    off_s = in_cf + (grow_s - in_cf) * tuck_s
    dist = (a * off_s + (1.0 - a) * off_t) * S
    Pk = body.offset(dist, relax=bd["relax"], fit=fit, gap=bd["gap"] * S)
    _add(soup, body, Pk, fit, centre_y, soup_knit_mat, torso_reps, (), 0.0, "body")
    gaps.append(body.gap(Pk, fit))
    # ---- waistband
    ring = upper.clip(0.5 - upper.fields["arm"])
    ring = ring.band(ring.fields["z"], z_wb - wb["height"] * S, z_wb)
    Pw = ring.offset(wb["offset"] * S, relax=wb["relax"], fit=fit, gap=wb["offset"] * S * 0.8)
    lw = _loops_by(ring, "z")                                              # the hem rolled, the top over the knit
    _add(soup, ring, Pw, fit, centre_y, soup_rib_mat, torso_reps, lw[:2],
         [(wb["offset"] - 0.001) * S, bd["under"] * S][:len(lw[:2])], "waistband")
    gaps.append(ring.gap(Pw, fit))
    # ---- cuffs
    cuff = upper.clip(upper.fields["arm"] - 0.5)
    cuff = cuff.band(cuff.fields["s"], arm_len - cf["length"] * S, arm_len + cf["past"] * S)
    Pc = cuff.offset(cf["offset"] * S, relax=cf["relax"], fit=fit, gap=cf["offset"] * S * 0.8)
    lc = _loops_by(cuff, "s", top=False)                                  # the hand ends rolled, the tops over the knit
    _add(soup, cuff, Pc, fit, centre_y, soup_rib_mat, torso_reps, lc[:4],
         [(cf["offset"] - 0.001) * S] * 2 + [bd["under"] * S] * 2, "cuff")
    gaps.append(cuff.gap(Pc, fit))
    info.update(z_waistband=z_wb, min_gap=float(min(gaps)),
                verts={"band": len(Pb), "body": len(Pk), "waistband": len(Pw), "cuffs": len(Pc)})
    return info


@builder("sweater", needs=("body",))
def build(ctx):
    cfg = SP.merge(DEFAULTS, ctx.cfg)
    fit = OU.make_fit(ctx)
    col = WR.colours(ctx, "sweater", COLORS)
    soup = G.Soup("sweater", [MAT_KNIT, MAT_RIB])
    info = sweater(fit, ctx.need("body"), cfg, 0, 1, soup)
    mats = [WR.material(ctx, MAT_KNIT, "knit", col["knit"], WR.knit(col["knit"]), shadow=(0.82, 0.74, 0.88)),
            WR.material(ctx, MAT_RIB, "rib", col["rib"], WR.rib(col["rib"]), shadow=(0.78, 0.72, 0.86))]
    mesh = soup.mesh()
    ctx.log(f"band top {info['band_top_z']:.3f} m on the chest, waistband top {info['z_waistband']:.3f} m, closest to "
            f"the skin {info['min_gap'] * 1000:.1f} mm")
    return Part("sweater", meshes=[mesh], materials=[m for m in mats if m.name in mesh.mats], info=info)
