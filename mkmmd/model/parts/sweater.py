"""The off-shoulder sweater (part `sweater`, numpy only): a wide folded rib band wrapped round the top of her body (the
chest, the upper arms and the back as one piece, leaving the shoulders bare), a loose knit body and long sleeves that
hang from the bust and the upper arms, one piece of knit (the sleeves run into the body under the arms, and a web of
knit joins each sleeve's underside to her side below the armpit), and blouse out over a snug ribbed waistband and long
snug ribbed cuffs. Every piece but the webs is a shell cut from the body's own skin (outfit_shell): it keeps the skin's
weights, so the sweater moves with her in any pose; the band is where her arm joins the chest too, so it stretches
between them as cloth does.

Where the pieces are, as fields on the skin (sizes in metres x S, S = head_tip.z / 1.7 with leg_extra left out):
  band      one level line `chest` below the shoulder joints (landmark arm.L) were her arms hanging straight down
            (`hung_z`): with her arms down it runs round her chest, her upper arms and her back at one height, a whole
            wrap that never stops at a shoulder, just above her armpits (lower, the knit under the arm would rise past
            it); plus `under` metres under each arm (fully within a third of `under_span` turns of straight under it,
            fading out at `under_span`, and only within `under_width` of the joint's depth: a slit right under the arm,
            so across the front and the back of the armpit the band runs on, one piece from her chest over her arm),
            so with her arms out it goes over each upper arm and never rings one (under the arm the knit runs from the
            body into the sleeve, as an off-shoulder sweater's does); it is `height` deep below that line, `offset` off
            the skin, its top edge rolled back to the skin (`roll`) so it reads as a thick fold; by her arms it spans the
            skin's hollows with a ball of radius `bridge` (Skin.bridge; wider than the knit's, which it keeps across her
            chest, so over the fold where her arm meets her chest it stays one roll instead of going down into it; 0: off)
  body      everything below the band down to the waistband and along the arms to the cuffs (`hang`): under the band
            `offset` off the skin, then leaving it over `drop` for loose cloth; on the breasts (within `cling` of each
            one's front, `breasts`) it lies on them, so they show; elsewhere it is `outfit_shell.drape`d round a vertical
            axis (the body: from the band's edge and below the breasts it follows her figure in towards the waist,
            loosely: `ease` off the skin, coming in no faster than `taper` at the front and `taper_back` at the back, so
            it bridges the small of her back and never reads as a barrel) and round each arm's bone line
            (`sleeve_taper`, `sleeve_ease`, sagging `sleeve_sag` below the arm and `underarm_sag` more under the armpit,
            fading out over the first 60 % of the upper arm); the cloth it has to spare gathers in
            `folds` / `sleeve_folds` soft folds up to `fold_depth` deep, into a `blouse` over the waistband and a `puff`
            over each cuff; over its last `tuck` it goes in `under` the waistband and the cuffs (their tops roll down
            onto it); smoothed (`relax`) and kept `gap` off the skin (vertices and triangle centres); where it goes in
            under the band it stays `tucked` behind it (`tucked`)
  bridge    the knit spans the skin's hollows as cloth does (Skin.bridge: where a ball of radius `bridge` rolled over her
            cannot reach, the armpits and the cleavage, it lies on it instead of going in; before its smoothing, so it
            evens the spans, and again after it below the band, which the smoothing pulls back into the armpits; its
            part under the band stays behind the band; 0: off)
  web       an option, off by default (`side` 0): under each arm a wedge of knit (`web`) spans from where the sleeve
            meets her side down `side` metres along her side and `sleeve` along the sleeve's underside, its free edge
            hanging `sag` of `side` below the line between those ends, `depth` to the front and the back of the plane
            through the shoulder joint at the armpit, closing to the free edge; its weights go from her side's to the
            sleeve's, so with the arm raised it spans the gap and with it lowered it folds away between them (a dolman
            sleeve's underarm; `side` 0.075 meets her side about 4 cm below the armpit with her hand on her hip)
  waistband `above_hip` (its top) over the hip joints (landmark leg.L), `height` deep, snug (`offset`), its hem rolled
  cuffs     `length` up the forearm from the wrist, `past` beyond it, snug, rolled at the hand
  ink       drawn lines (outfit_ink, `width` wide): ribs across the band, the waistband and the cuffs every `band` /
            `waistband` / `cuff` metres, a fold line along the band, and with `folds` the knit's lines: folds over the
            waistband, inside the elbows and over the cuffs (`enabled` true); nothing outlines the breasts
  seed      the folds' and the lines' random draw

Spec: `[sweater]` (DEFAULTS below, tables merge key by key) and `[colors.sweater]` knit (the body and sleeves), rib (the
band, waistband and cuffs) and ink (the lines). Materials セーター (knit) and セーターリブ (ribs), double sided, outlined
dark, and セーター線 (the ink)."""
import numpy as np

from .. import skin as SK
from .. import spec as SP
from ..build import builder
from ..part import Part
from . import outfit as OU
from . import outfit_fit as OF
from . import outfit_geo as G
from . import outfit_ink as INK
from . import outfit_shell as SH
from . import outfit_wear as WR

DEFAULTS = {
    "band": {"chest": 0.022, "under": 0.14, "under_span": 0.14, "under_width": 0.020, "height": 0.050, "offset": 0.010,
             "roll": 0.008, "relax": 8, "bridge": 0.10},
    "body": {"offset": 0.008, "ease": 0.014, "taper": 0.5, "taper_back": 0.5, "cling": 0.070, "blouse": 0.010,
             "folds": 11, "fold_depth": 0.016, "sleeve_ease": 0.009, "sleeve_taper": 0.10, "sleeve_sag": 0.008,
             "sleeve_folds": 7, "puff": 0.012, "tuck": 0.040, "under": 0.0022, "drop": 0.040, "relax": 8, "gap": 0.003,
             "underarm_sag": 0.035, "tucked": 0.0015},
    "waistband": {"above_hip": 0.095, "height": 0.034, "offset": 0.006, "relax": 4},
    "cuff": {"length": 0.062, "past": 0.004, "offset": 0.006, "relax": 4},
    "ink": {"enabled": True, "width": 0.0026, "band": 0.040, "waistband": 0.040, "cuff": 0.024, "folds": True},
    "web": {"side": 0.0, "sleeve": 0.09, "sag": 0.35, "depth": 0.03},
    "bridge": 0.045,
    "seed": 7,
}
COLORS = {"knit": "#e7a7eb", "rib": "#9f75b0", "ink": "#2e2234"}
MAT_KNIT = "セーター"
MAT_RIB = "セーターリブ"
MAT_INK = "セーター線"


def hung_z(V, share, fit):
    """(n,) the height of each point of the skin were her arms hanging straight down: on each side the point is turned
    about the shoulder joint (landmark arm.L/R), in the plane across her, by the angle from where the upper arm points
    (to the elbow) to straight down, as much as the arm bones' `share` of it, so the shoulder hands over smoothly."""
    L = fit.L
    out = V[:, 2].copy()
    for side, sg in (("L", 1.0), ("R", -1.0)):
        j, e = np.asarray(L[f"arm.{side}"], float), np.asarray(L[f"elbow.{side}"], float)
        t = np.arctan2(sg * (e[0] - j[0]), j[2] - e[2])                    # the upper arm out from straight down
        on = sg * V[:, 0] > 0.0
        dx, dz = sg * (V[on, 0] - j[0]), V[on, 2] - j[2]
        out[on] = V[on, 2] + share[on] * (dz * np.cos(t) - dx * np.sin(t) - dz)
    return out


def fields(sh, fit, c):
    """Per-vertex fields on the skin shell `sh` (dict of (n,)): z, s (arm coordinate), arm (the arm bones' share),
    band (depth below the band's top line), and the sizes they are cut at."""
    S, L = fit.S, fit.L
    V = sh.verts
    s = SH.both_sides(V, L)
    arm = sh.share(SH.ARM)
    b = c["band"]
    # one level line `chest` below the shoulder joints were her arms hanging down: round her chest, her back and her
    # upper arms alike, so the band wraps the top of her body as one piece and never stops at a shoulder
    band = float(L["arm.L"][2]) - b["chest"] * S - hung_z(V, arm, fit)
    # under each arm the band's depth grows by `under` metres, so the knit is one piece and the band goes over the arm,
    # never round it; only right under the arm (`under_span` turns of straight under it, within `under_width` of the
    # joint's depth): a slit the arm hides, so across the front and the back of the armpit the band runs on from the
    # chest over the arm in one piece
    x_arm = 0.5 * (abs(float(L["arm.L"][0])) + abs(float(L["arm.R"][0])))
    y_arm = 0.5 * (float(L["arm.L"][1]) + float(L["arm.R"][1]))
    off = np.abs(_wrap(SH.limb_angle(V, L) - 0.5))                                   # turns from straight under the arm
    span = b["under_span"]
    under = G.smoothstep(np.clip((span - off) / (2.0 / 3.0 * span), 0.0, 1.0))
    by_arm = G.smoothstep(np.clip((np.abs(V[:, 0]) - (x_arm - 0.05 * S)) / (0.03 * S), 0.0, 1.0))
    side = G.smoothstep(np.clip((b["under_width"] * S - np.abs(V[:, 1] - y_arm)) / (0.015 * S), 0.0, 1.0))
    band = band + b["under"] * S * under * by_arm * side
    arm_len = 0.5 * (SH.limb_length(L, "L") + SH.limb_length(L, "R"))
    z_wb = float(L["leg.L"][2]) + c["waistband"]["above_hip"] * S
    return {"z": V[:, 2].copy(), "s": s, "arm": arm, "band": band, "arm_len": arm_len, "z_wb": z_wb, "S": S}


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


def _around(P, q, d):
    """Angle (fraction of a turn, 0 = up), distance and offset (n, 3) of points P from their local axis (points q, unit
    directions d)."""
    v = P - q
    v = v - np.einsum("ij,ij->i", v, d)[:, None] * d
    up = np.array([0.0, 0.0, 1.0])[None, :] - d[:, 2:3] * d
    up = up / np.maximum(np.linalg.norm(up, axis=1, keepdims=True), 1e-12)
    fw = np.cross(d, up)
    ang = np.arctan2(np.einsum("ij,ij->i", v, fw), np.einsum("ij,ij->i", v, up)) / (2 * np.pi)
    return ang % 1.0, np.linalg.norm(v, axis=1), v


def _folds(turn, count, rng):
    """(k,) irregular folds round a ring, -1..1: about `count` waves per turn at `turn` (fractions of a turn), a few
    neighbouring wave counts mixed with random strengths and phases so no two folds are alike."""
    ks = np.arange(max(int(count) - 3, 1), int(count) + 4)
    amp = rng.uniform(0.4, 1.0, len(ks)) * np.exp(-0.5 * ((ks - count) / 2.0) ** 2)
    ph = rng.uniform(0.0, 2 * np.pi, len(ks))

    def wave(t):
        return (amp[None, :] * np.sin(2 * np.pi * np.asarray(t, float)[:, None] * ks[None, :] + ph[None, :])).sum(1)
    return wave(turn) / np.abs(wave(np.linspace(0.0, 1.0, 721))).max()


def breasts(skin, fit):
    """The front of each breast on the skin Shell `skin` (with field arm): the torso's most forward point on each side
    within 3 cm x S of the bust_front landmark's height; [] when the body has no such landmark."""
    L, S = fit.L, fit.S
    if not L.has("bust_front"):
        return []
    K, zb, out = skin.verts, float(L["bust_front"][2]), []
    for sg in (1.0, -1.0):
        near = (skin.fields["arm"] < 0.3) & (np.abs(K[:, 2] - zb) < 0.03 * S) & (sg * K[:, 0] > 0.01 * S)
        if near.any():
            out.append(K[near][np.argmin(K[near][:, 1])])
    return out


def _from_fronts(P, fronts):
    """(n,) distance of points P to the nearest breast front (inf without any)."""
    return np.min([np.linalg.norm(P - f, axis=1) for f in fronts], axis=0) if fronts else np.full(len(P), np.inf)


def tucked(P, ids, sh, Q, gap):
    """Cloth at positions P (n, 3) whose points `ids` lie under an outer layer (the knit's top under the band) kept at
    least `gap` behind it: the layer is the Shell `sh` at positions Q, and a point closer to it than that, or in front
    of it, goes back along the layer's own surface normal at its nearest point (the smoothed knit can come out through
    the band's bottom edge in the cleavage, where the two dip differently). Returns new positions."""
    P = np.array(P, float)
    ids = np.asarray(ids)
    if not len(ids) or not len(Q):
        return P
    T = sh.tris
    fn = np.cross(Q[T[:, 1]] - Q[T[:, 0]], Q[T[:, 2]] - Q[T[:, 0]])
    vn = np.zeros_like(Q)
    for k in range(3):
        np.add.at(vn, T[:, k], fn)
    vn /= np.maximum(np.linalg.norm(vn, axis=1, keepdims=True), 1e-12)
    vn *= np.sign(np.einsum("ij,ij->i", vn, sh.normals).sum())          # outward, as the skin's it was cut from
    q, tri, bary, _ = SK.closest_on_mesh(P[ids], Q, T)
    n = np.einsum("ij,ijk->ik", bary, vn[T[tri]])
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    d = np.einsum("ij,ij->i", P[ids] - q, n)
    fix = d > -gap
    P[ids[fix]] -= (d[fix] + gap)[:, None] * n[fix]
    return P


def hang(knit, skin, fit, c, centre_y, h, z_wb, end_s, fronts, rng):
    """Where the loose knit (the Shell `knit`, cut from the skin Shell `skin`) hangs, (n, 3): the body round a vertical
    axis through the waist's middle, the sleeves round each arm's bone line, both `outfit_shell.drape`d over the skin
    (from the band's bottom edge and from the upper arm down, coming in no faster than `taper` at the front,
    `taper_back` at the back, `sleeve_taper` on the arms) `ease` / `sleeve_ease` off it; blended by the arm bones'
    share. Within `cling` of the breasts' `fronts` it lies on them (`offset` off the skin, the cleavage bridged) so they
    show, and it hangs from the skin below them. The cloth it has to spare gathers: `folds` waves round the body
    growing to `fold_depth` down to a `blouse` over the waistband, `sleeve_folds` waves round each sleeve over its last
    10 cm to a `puff` over the cuff, and the sleeves sag `sleeve_sag` below the arm (they rest on it and hang). Under the
    band it lies `offset` off the skin and leaves it over `drop`; over its last `tuck` it goes in `under` the waistband
    and the cuffs."""
    S, L = fit.S, fit.L
    bd, wb, cf = c["body"], c["waistband"], c["cuff"]
    V, N, a = knit.verts, knit.normals, knit.fields["arm"]
    z0 = float(L["arm.L"][2]) - c["band"]["chest"] * S - h                 # the band's bottom on the chest
    tuck = bd["tuck"] * S
    K = skin.verts

    def cyl(P):
        return z0 - P[:, 2], SH.azimuth(P, (0.0, centre_y)), np.hypot(P[:, 0], P[:, 1] - centre_y)
    # it hangs from the band's bottom edge: what lies under the band (the bust's front, the shoulder blades) holds the band
    trunk = (skin.fields["arm"] < 0.3) & (K[:, 2] > z_wb - 0.02 * S) & (skin.fields["band"] > h - 0.005 * S)
    sk, ak, rk = cyl(K[trunk])
    sc, ac, _ = cyl(V)
    turn = (np.arange(72) + 0.5) / 72
    taper = bd["taper"] + (bd["taper_back"] - bd["taper"]) * 0.5 * (1.0 - np.cos(2 * np.pi * turn))
    cl = bd["cling"] * S
    held = _from_fronts(K[trunk], fronts) < cl
    R = SH.drape(sk[~held], ak[~held], rk[~held], sc, ac, na=72, taper=taper) + bd["ease"] * S
    hug = G.smoothstep(np.clip((cl - _from_fronts(V, fronts)) / (0.012 * S), 0.0, 1.0))
    if fronts:
        R = hug * (SH.drape(sk, ak, rk, sc, ac, na=72, taper=10.0) + bd["offset"] * S) + (1.0 - hug) * R
    down = G.smoothstep(np.clip(sc / max(z0 - z_wb, 1e-6), 0.0, 1.0))                # 0 under the band, 1 at the hem
    R = R + bd["fold_depth"] * S * (1.0 - hug) * down ** 1.5 * _folds(ac, bd["folds"], rng)
    R = R + bd["blouse"] * S * np.exp(-((V[:, 2] - z_wb - tuck) / (0.025 * S)) ** 2)
    T = np.stack([R * np.sin(2 * np.pi * ac), centre_y - R * np.cos(2 * np.pi * ac), V[:, 2]], 1)
    Ts = T.copy()
    for side, sg in (("L", 1.0), ("R", -1.0)):
        on = sg * V[:, 0] >= 0.0
        arm = (skin.fields["arm"] > 0.5) & (sg * K[:, 0] > 0.0)
        s_k, q_k, d_k = SH.limb_axis(K[arm], L, side)
        a_k, r_k, _ = _around(K[arm], q_k, d_k)
        s_c, q_c, d_c = SH.limb_axis(V[on], L, side)
        a_c, _, v_c = _around(V[on], q_c, d_c)
        Rs = SH.drape(s_k, a_k, r_k, s_c, a_c, taper=bd["sleeve_taper"]) + bd["sleeve_ease"] * S
        Rs = Rs + bd["sleeve_sag"] * S * 0.5 * (1.0 - np.cos(2 * np.pi * a_c))     # a_c 0 = up: nothing on top
        la = float(np.linalg.norm(np.asarray(L[f"elbow.{side}"], float) - np.asarray(L[f"arm.{side}"], float)))
        near_pit = 1.0 - G.smoothstep(np.clip(s_c / (0.6 * la), 0.0, 1.0))                # most at the armpit
        Rs = Rs + bd["underarm_sag"] * S * 0.5 * (1.0 - np.cos(2 * np.pi * a_c)) * near_pit
        gather = np.clip((s_c - end_s + 0.10 * S) / (0.10 * S), 0.0, 1.0)
        Rs = Rs + bd["fold_depth"] * S * gather ** 1.5 * _folds(a_c, bd["sleeve_folds"], rng)
        Rs = Rs + bd["puff"] * S * np.exp(-((s_c - end_s + tuck) / (0.02 * S)) ** 2)
        Ts[on] = q_c + v_c / np.maximum(np.linalg.norm(v_c, axis=1, keepdims=True), 1e-12) * Rs[:, None]
    w = G.smoothstep(np.clip((a - 0.3) / 0.4, 0.0, 1.0))[:, None]
    T = w * Ts + (1.0 - w) * T
    top = G.smoothstep(np.clip((knit.fields["band"] - h + 0.012 * S) / (bd["drop"] * S), 0.0, 1.0))[:, None]
    T = (V + N * bd["offset"] * S) * (1.0 - top) + T * top
    t_in = w * G.smoothstep(np.clip((end_s - knit.fields["s"]) / tuck, 0.0, 1.0))[:, None] \
        + (1.0 - w) * G.smoothstep(np.clip((V[:, 2] - z_wb) / tuck, 0.0, 1.0))[:, None]
    inside = V + N * ((w[:, 0] * cf["offset"] + (1.0 - w[:, 0]) * wb["offset"] - bd["under"]) * S)[:, None]
    return inside * (1.0 - t_in) + T * t_in


def web(knit, P, fit, c, centre_y, reps):
    """The cloth under each arm (`[sweater.web]`, `side` > 0), as G.Patches of knit. In her frontal plane through the
    shoulder joint it runs from where the sleeve meets her side (the corner of the hung knit `knit` at positions P, in
    that plane) down `side` metres along her side and `sleeve` metres along the sleeve's underside, its free edge hanging
    `sag` of `side` below the line between those two ends; across that plane it is a wedge, `depth` metres to each side
    of it at the corner, closing to the free edge (a front and a back sheet that meet there, so it shades as the
    underside of the cloth, not as a flat panel), its rims (the edges along her side and the sleeve, and the corner) 4 mm
    x S inside the knit they run along, so only the cloth across the gap shows. Its weights go from the side's to the
    sleeve's (the knit's own at its two edges), so with her arm lowered it folds between them and with it raised it
    spans the gap; its texture runs on from the knit's at both edges. Returns ([patch], {side: (the corner, the end on
    her side, the end on the sleeve), (3,) each})."""
    S, L = fit.S, fit.L
    wc = c["web"]
    D = wc["side"] * S
    out, ends = [], {}
    if D <= 0.0:
        return out, ends
    arm, eps = knit.fields["arm"], 0.004 * S
    cloth = OF.Skin(P, knit.tris)
    ku, kv, k_arm = _uv(knit, fit, centre_y)
    k_uv = np.stack([ku * np.where(k_arm, reps[1], reps[0]), kv / WR.TILE], 1)
    for side, sg in (("L", 1.0), ("R", -1.0)):
        J, E = (np.asarray(L[f"{n}.{side}"], float) for n in ("arm", "elbow"))
        a2 = np.array([sg * (E - J)[0], (E - J)[2]])
        a2 /= np.linalg.norm(a2)                                       # the upper arm in her frontal plane (out, up)
        n2 = np.array([-a2[1], a2[0]]) * (1.0 if a2[0] > 0 else -1.0)  # across it, upwards
        j2 = np.array([sg * J[0], J[2]])
        p2 = np.stack([sg * P[:, 0], P[:, 2]], 1)
        slab = (np.abs(P[:, 1] - J[1]) < 0.012 * S) & (sg * P[:, 0] > 0.0)
        # her side: the outermost torso knit in the slab at each height under the arm
        tor = slab & (arm < 0.25) & (P[:, 2] < J[2] - 0.03 * S) & (P[:, 2] > J[2] - 0.22 * S)
        # the sleeve's underside: the lowest sleeve knit across the arm's line at each place along it
        rel = p2 - j2
        s2, h2 = rel @ a2, rel @ n2
        slv = slab & (arm > 0.75) & (s2 > 0.02 * S) & (s2 < 0.20 * S) & (h2 < 0.0)
        side_pts = _extreme(P[tor, 2], p2[tor, 0], 0.01 * S, max)
        arm_pts = _extreme(s2[slv], h2[slv], 0.01 * S, min)
        if len(side_pts) < 3 or len(arm_pts) < 3:
            continue
        beta, alpha = np.polyfit(side_pts[:, 0], side_pts[:, 1], 1)           # lateral = alpha + beta z
        delta, gamma = np.polyfit(arm_pts[:, 0], arm_pts[:, 1], 1)            # across = gamma + delta s
        den = beta * (a2[1] + delta * n2[1]) - (a2[0] + delta * n2[0])
        if abs(den) < 1e-9:
            continue
        s_k = (j2[0] + gamma * n2[0] - alpha - beta * (j2[1] + gamma * n2[1])) / den
        k2 = j2 + s_k * a2 + (gamma + delta * s_k) * n2                     # the corner: the side meets the sleeve
        e_side = np.array([-beta, -1.0]) / np.hypot(beta, 1.0)
        e_arm = (a2 + delta * n2) / np.linalg.norm(a2 + delta * n2)
        in_side = np.array([-1.0, beta]) / np.hypot(beta, 1.0)                 # into her body
        in_arm = np.array([-e_arm[1], e_arm[0]])
        in_arm *= np.sign(in_arm @ n2) or 1.0                                  # into the arm
        a_s, a_a = k2 + D * e_side, k2 + wc["sleeve"] * S * e_arm
        away = (a_s + a_a) / 2.0 - k2
        away /= np.linalg.norm(away)
        t = np.linspace(0.0, 1.0, 13)
        free = (1.0 - t)[:, None] * a_s + t[:, None] * a_a + (4.0 * t * (1.0 - t) * wc["sag"] * D)[:, None] * away
        free[0] += eps * in_side
        free[-1] += eps * in_arm
        apex = k2 - eps * away
        rho = np.linspace(0.0, 1.0, 9)[1:]
        mid = apex[None, None] + rho[None, :, None] * (free[:, None] - apex[None, None])      # (t, rho, 2)
        mid = np.stack([sg * mid[..., 0], np.full(mid.shape[:2], J[1]), mid[..., 1]], -1)      # (t, rho, 3)
        top = np.array([sg * apex[0], J[1], apex[1]])
        half = wc["depth"] * S * (1.0 - G.smoothstep(rho))                    # the wedge's half depth, none at the edge
        nt, nr = len(t), len(rho)
        b = G.smoothstep(t)
        verts, faces, W, uv = [], [], [], []
        free_ids = None
        for sy in (-1.0, 1.0):                                                 # the front sheet, then the back one
            Q = mid + sy * half[None, :, None] * np.array([0.0, 1.0, 0.0])
            corner = top + sy * wc["depth"] * S * np.array([0.0, 1.0, 0.0])
            rim = np.zeros((nt, nr), bool)
            rim[0, :] = rim[-1, :] = True
            pts = np.concatenate([corner[None], Q[rim]])
            _, q, tri, bary = cloth.closest(pts)
            pts = q - eps * cloth.outward(tri, bary)                            # just inside the knit they run along
            corner, Q[rim] = pts[0], pts[1:]
            base = len(verts)
            ids = np.full((nt, nr), -1)
            for i in range(nt):
                for j in range(nr):
                    if j == nr - 1 and free_ids is not None:                      # the free edge: shared with the front
                        ids[i, j] = free_ids[i]
                        continue
                    ids[i, j] = len(verts)
                    verts.append(Q[i, j])
            if free_ids is None:
                free_ids = ids[:, -1].copy()
            apex_id = len(verts)
            verts.append(corner)
            sheet = [(apex_id, ids[i, 0], ids[i + 1, 0]) for i in range(nt - 1)]
            sheet += [(ids[i, j], ids[i, j + 1], ids[i + 1, j + 1], ids[i + 1, j])
                      for i in range(nt - 1) for j in range(nr - 1)]
            V = np.asarray(verts, float)
            if G.face_normals(V, sheet).sum(0)[1] * sy < 0.0:              # the front sheet faces her front, the back one
                sheet = [f[::-1] for f in sheet]                              # her back
            faces += sheet
            new = np.arange(base, len(verts))
            # weights and texture: the knit's own at the sheet's two edges, graded across it
            tt = np.array([b[np.argwhere(ids == k)[0][0]] if k != apex_id else 0.5 for k in new])
            rr = np.array([np.argwhere(ids == k)[0][1] if k != apex_id else 0 for k in new])
            near_s = _nearest_ids(P, Q[0, rr])
            near_a = _nearest_ids(P, Q[-1, rr])
            W.append((1.0 - tt)[:, None] * knit.W[near_s] + tt[:, None] * knit.W[near_a])
            uv.append((1.0 - tt)[:, None] * k_uv[near_s] + tt[:, None] * k_uv[near_a])
        V3, Wm, UV = np.asarray(verts, float), np.concatenate(W), np.concatenate(uv)
        w = {bn: Wm[:, i] for i, bn in enumerate(knit.bones) if Wm[:, i].max() > 1e-6}
        out.append(G.Patch(V3, faces, np.array([UV[i] for f in faces for i in f]), w, 0, "web"))
        ends[side] = (np.array([sg * k2[0], J[1], k2[1]]), np.array([sg * a_s[0], J[1], a_s[1]]),
                      np.array([sg * a_a[0], J[1], a_a[1]]))
    return out, ends


def _extreme(key, val, step, pick):
    """(k, 2) rows (key, value): per bin of `step` along key, the value `pick` (max or min) chooses."""
    if not len(key):
        return np.zeros((0, 2))
    bins = np.floor(np.asarray(key) / step).astype(np.int64)
    rows = []
    for bn in np.unique(bins):
        m = bins == bn
        i = np.flatnonzero(m)[np.argmax(val[m]) if pick is max else np.argmin(val[m])]
        rows.append((key[i], val[i]))
    return np.asarray(rows, float)


def _nearest_ids(P, Q):
    """Index of the point of P nearest to each point of Q."""
    d2 = (Q * Q).sum(1)[:, None] - 2.0 * Q @ P.T + (P * P).sum(1)[None, :]
    return np.argmin(d2, axis=1)


def _wrap(a):
    """Turn fractions as signed offsets from 0 (-0.5 .. 0.5)."""
    return (np.asarray(a, float) + 0.5) % 1.0 - 0.5


def ink(fit, c, pieces, centre_y, z_wb, end_s, rng):
    """The drawn lines on the sweater (`[sweater.ink]`): (outfit_ink lines, their widths as shares of `width`). Ribs
    across the band, the waistband and the cuffs every `band` / `waistband` / `cuff` metres, a broken fold line along
    the band, and with `folds` the knit's own lines: where it blouses over the waistband, inside the elbows and where the
    sleeves gather over the cuffs. Nothing outlines her breasts: the knit lying on them shows them. `pieces`: {name:
    (Shell, positions)}."""
    S, L = fit.S, fit.L
    k, h, cf = c["ink"], c["band"]["height"] * S, c["cuff"]
    for sh, _ in pieces.values():
        sh.fields.update(az=SH.azimuth(sh.verts, (0.0, centre_y)), la=SH.limb_angle(sh.verts, L))
    lines, widths = [], []

    def add(ls, w):
        lines.extend(ls)
        widths.extend([w] * len(ls))

    def window(name, centre, half):
        return lambda fl: np.abs(_wrap(fl[name] - centre)) <= half

    sh, P = pieces["band"]
    torso_t, arm_t = _charts(sh, sh.fields["arm"] >= 0.5)
    for u, t in (("az", torso_t), ("la", arm_t)):
        add(INK.ribs(sh, P, u, k["band"] * S, "band", 0.0, h, rng, tris=t), 0.8)
    wob = 0.06 * h * np.sin(2 * np.pi * (sh.verts[:, 0] / (0.09 * S) + sh.verts[:, 1] / (0.13 * S)))
    fold = INK.isolines(sh, P, sh.fields["band"] + wob, 0.42 * h)
    add(INK.dashes(fold, rng, (0.05 * S, 0.16 * S), (0.008 * S, 0.03 * S)), 0.9)
    sh, P = pieces["waistband"]
    add(INK.ribs(sh, P, "az", k["waistband"] * S, "z", z_wb - c["waistband"]["height"] * S, z_wb, rng), 0.8)
    sh, P = pieces["cuffs"]
    add(INK.ribs(sh, P, "la", k["cuff"] * S, "s", end_s, end_s + (cf["length"] + cf["past"]) * S, rng), 0.8)
    if not k["folds"]:
        return lines, widths
    sh, P = pieces["body"]
    z, s = sh.fields["z"], sh.fields["s"]
    on_arm = sh.fields["arm"][sh.tris]
    trunk = (on_arm < 0.3).all(1)
    for _ in range(6):                                          # blousing over the waistband: slanted strokes
        ac, slope = rng.uniform(-0.32, 0.32), rng.uniform(-0.35, 0.35)
        g = z + slope * _wrap(sh.fields["az"] - ac) * 0.75 * S
        add(INK.arcs(sh, P, g, z_wb + rng.uniform(0.010, 0.040) * S, window("az", ac, rng.uniform(0.02, 0.045)),
                     fields=("az",), tris=trunk), 1.0)
    for side, sg in (("L", 1.0), ("R", -1.0)):
        el = np.asarray(L[f"elbow.{side}"], float)
        s_el = float(np.linalg.norm(el - np.asarray(L[f"arm.{side}"], float)))
        front = float(SH.limb_angle((el + np.array([0.0, -0.05, 0.0]))[None], L)[0])
        sleeve = (on_arm > 0.5).all(1) & (sg * sh.verts[sh.tris][:, :, 0] > 0.0).all(1)
        for _ in range(3):                                      # inside the elbow
            add(INK.arcs(sh, P, s, s_el + rng.uniform(-0.025, 0.02) * S,
                         window("la", front + rng.uniform(-0.06, 0.06), rng.uniform(0.05, 0.10)), fields=("la",),
                         tris=sleeve), 0.9)
        for _ in range(4):                                      # gathered over the cuff
            add(INK.arcs(sh, P, s, end_s - rng.uniform(0.004, 0.022) * S,
                         window("la", rng.uniform(0.0, 1.0), rng.uniform(0.06, 0.14)), fields=("la",), tris=sleeve), 0.9)
    return lines, widths


def sweater(fit, body_part, cfg, soup):
    """Build the sweater into `soup` (material 0 knit, 1 rib, 2 ink). Returns info: the pieces' vertex counts, the
    band's top height on the chest, the waistband's top, the closest distance of any piece to the skin and the number
    of drawn lines."""
    c = SP.merge(DEFAULTS, cfg)
    S = fit.S
    base = SH.Shell.from_part(body_part)
    f = fields(base, fit, c)
    base.fields.update(z=f["z"], s=f["s"], arm=f["arm"], band=f["band"])
    # the body's axis: midway between her front and back at the waistband (the trunk's own skin: her hands hang there)
    near = (np.abs(base.verts[:, 2] - f["z_wb"]) < 0.03 * S) & (base.fields["arm"] < 0.3) & (base.share(SH.LEG) < 0.3)
    centre_y = float(0.5 * (base.verts[near, 1].min() + base.verts[near, 1].max())) if near.any() else 0.0
    upper = base.clip(base.fields["z"] - (f["z_wb"] - c["waistband"]["height"] * S))          # nothing below the waistband
    upper = upper.clip(f["arm_len"] + c["cuff"]["past"] * S - upper.fields["s"])               # nothing past the cuffs
    torso_reps = (round(0.85 / WR.TILE), round(0.20 / WR.TILE))
    info, gaps = {}, []
    # ---- the band
    b = c["band"]
    h = b["height"] * S
    r = c["bridge"] * S                                                     # spanning the hollows (Skin.bridge)
    band = upper.band(upper.fields["band"], 0.0, h)
    Pb = band.offset(b["offset"] * S, relax=b["relax"], fit=fit, gap=b["offset"] * S * 0.8)
    g = b["offset"] * S * 0.8
    if r > 0.0 or b["bridge"] > 0.0:          # by her arms it spans the fold where the arm meets her chest with its own,
        x_arm = 0.5 * (abs(float(fit.L["arm.L"][0])) + abs(float(fit.L["arm.R"][0])))   # wider ball, one roll over it
        wide = G.smoothstep(np.clip((np.abs(band.verts[:, 0]) - 0.45 * x_arm) / (0.25 * x_arm), 0.0, 1.0))[:, None]
        p_knit = fit.skin.bridge(Pb, r, g) if r > 0.0 else Pb
        p_wide = fit.skin.bridge(Pb, b["bridge"] * S, g) if b["bridge"] > 0.0 else Pb
        Pb = band.clear(fit.push_out((1.0 - wide) * p_knit + wide * p_wide, g), fit, g)
    loops = _loops_by(band, "band")                                         # the top rolled to the skin, the bottom
    rims = loops[:2]                                                        # onto the knit under it
    _add(soup, band, Pb, fit, centre_y, 1, torso_reps, rims,
         [b["roll"] * S, (b["offset"] - c["body"]["offset"]) * S][:len(rims)], "band")
    top = (band.fields["band"] < 1e-4 * S) & (np.abs(band.verts[:, 0]) < 0.03 * S) & (band.verts[:, 1] < centre_y)
    info["band_top_z"] = float(band.verts[top, 2].mean()) if top.any() else None
    gaps.append(band.gap(Pb, fit))
    # ---- body and sleeves: loose knit, tight only where it goes into the ribs
    bd = c["body"]
    wb, cf = c["waistband"], c["cuff"]
    z_wb, arm_len = f["z_wb"], f["arm_len"]
    body = upper.clip(upper.fields["band"] - (h - 0.012 * S))
    body = body.clip(body.fields["z"] - (z_wb - 0.010 * S))
    body = body.clip((arm_len - cf["length"] * S + 0.010 * S) - body.fields["s"]).subdivide()   # fine enough to fold
    end_s = arm_len - cf["length"] * S
    rng = np.random.default_rng(int(c["seed"]))
    fronts = breasts(base, fit)
    Pk = hang(body, base, fit, c, centre_y, h, z_wb, end_s, fronts, rng)
    if r > 0.0:
        Pk = fit.skin.bridge(Pk, r, bd["gap"] * S)
    Pk = body.settle(Pk, relax=bd["relax"], fit=fit, gap=bd["gap"] * S)
    if r > 0.0:
        below = body.fields["band"] >= h
        Pk[below] = fit.skin.bridge(Pk[below], r, bd["gap"] * S)
        Pk = body.clear(fit.push_out(Pk, bd["gap"] * S), fit, bd["gap"] * S)
    Pk = tucked(Pk, np.flatnonzero(body.fields["band"] < h), band, Pb, bd["tucked"] * S)
    _add(soup, body, Pk, fit, centre_y, 0, torso_reps, (), 0.0, "body")
    webs, ends = web(body, Pk, fit, c, centre_y, torso_reps)
    info["web"] = {k: {"corner": v[0].round(4).tolist(), "side": v[1].round(4).tolist(), "sleeve": v[2].round(4).tolist()}
                   for k, v in ends.items()}
    for p in webs:
        soup.add(p, 0)
    gaps.append(body.gap(Pk, fit))
    info["knit_ease"] = float(np.median(fit.skin.closest(Pk[body.fields["arm"] < 0.3])[0]))
    # ---- waistband
    ring = upper.clip(0.5 - upper.fields["arm"])
    ring = ring.band(ring.fields["z"], z_wb - wb["height"] * S, z_wb)
    Pw = ring.offset(wb["offset"] * S, relax=wb["relax"], fit=fit, gap=wb["offset"] * S * 0.8)
    lw = _loops_by(ring, "z")                                              # the hem rolled, the top over the knit
    _add(soup, ring, Pw, fit, centre_y, 1, torso_reps, lw[:2],
         [(wb["offset"] - 0.001) * S, bd["under"] * S][:len(lw[:2])], "waistband")
    gaps.append(ring.gap(Pw, fit))
    # ---- cuffs
    cuff = upper.clip(upper.fields["arm"] - 0.5)
    cuff = cuff.band(cuff.fields["s"], arm_len - cf["length"] * S, arm_len + cf["past"] * S)
    Pc = cuff.offset(cf["offset"] * S, relax=cf["relax"], fit=fit, gap=cf["offset"] * S * 0.8)
    lc = _loops_by(cuff, "s", top=False)                                  # the hand ends rolled, the tops over the knit
    _add(soup, cuff, Pc, fit, centre_y, 1, torso_reps, lc[:4],
         [(cf["offset"] - 0.001) * S] * 2 + [bd["under"] * S] * 2, "cuff")
    gaps.append(cuff.gap(Pc, fit))
    lines = []
    if c["ink"]["enabled"]:
        pieces = {"band": (band, Pb), "body": (body, Pk), "waistband": (ring, Pw), "cuffs": (cuff, Pc)}
        lines, widths = ink(fit, c, pieces, centre_y, z_wb, end_s, rng)
        soup.add(INK.strokes(base.bones, lines, np.asarray(widths) * c["ink"]["width"] * S, lift=0.0004 * S), 2)
    info.update(z_waistband=z_wb, min_gap=float(min(gaps)), lines=len(lines),
                verts={"band": len(Pb), "body": len(Pk), "waistband": len(Pw), "cuffs": len(Pc)})
    return info


@builder("sweater", needs=("body",))
def build(ctx):
    cfg = SP.merge(DEFAULTS, ctx.cfg)
    fit = OU.make_fit(ctx)
    col = WR.colours(ctx, "sweater", COLORS)
    soup = G.Soup("sweater", [MAT_KNIT, MAT_RIB, MAT_INK])
    info = sweater(fit, ctx.need("body"), cfg, soup)
    mats = [WR.material(ctx, MAT_KNIT, "knit", col["knit"], WR.knit(col["knit"]), shadow=(0.82, 0.74, 0.88)),
            WR.material(ctx, MAT_RIB, "rib", col["rib"], WR.rib(col["rib"]), shadow=(0.78, 0.72, 0.86)),
            WR.ink_material(ctx, MAT_INK, "ink", col["ink"])]
    mesh = soup.mesh()
    ctx.log(f"band top {info['band_top_z']:.3f} m on the chest, waistband top {info['z_waistband']:.3f} m, closest to "
            f"the skin {info['min_gap'] * 1000:.1f} mm, {info['lines']} drawn lines")
    return Part("sweater", meshes=[mesh], materials=[m for m in mats if m.name in mesh.mats], info=info)
