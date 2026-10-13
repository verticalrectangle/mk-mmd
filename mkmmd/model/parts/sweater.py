"""The off-shoulder sweater (part `sweater`, numpy only): a wide folded rib band round the chest and the upper arms
leaving the shoulders bare, a loose knit body and long sleeves that hang from the bust and the upper arms and blouse out
over a snug ribbed waistband and long snug ribbed cuffs. Every piece is a shell cut from the body's own skin
(outfit_shell): it keeps the skin's weights, so the sweater moves with her in any pose; the band is where her arm joins
the chest too, so it stretches between them as cloth does.

Where the pieces are, as fields on the skin (sizes in metres x S, S = head_tip.z / 1.7 with leg_extra left out):
  band      the soft max of two descents, `chest` below the shoulder joints (landmark arm.L) on the torso, rising `rise`
            towards the arms (up to the shoulder joints' width), and `arm` along the upper arm from its joint
            (outfit_shell.limb_coord): one level line runs across the chest, over each upper arm and across the back, so
            the band is one piece from arm to arm; it is `height` deep below that line, `offset` off the skin, its top
            edge rolled back to the skin (`roll`) so it reads as a thick fold
  body      everything below the band down to the waistband and along the arms to the cuffs (`hang`): under the band
            `offset` off the skin, then leaving it over `drop` for loose cloth; on the breasts (within `cling` of each
            one's front, `breasts`) it lies on them, so they show; elsewhere it is `outfit_shell.drape`d round a vertical
            axis (the body: from the band's edge and below the breasts it falls almost straight, coming in no faster
            than `taper` at the front and `taper_back` at the back, `ease` off the skin) and round each arm's bone line
            (`sleeve_taper`, `sleeve_ease`, sagging `sleeve_sag` below the arm); the cloth it has to spare gathers in
            `folds` / `sleeve_folds` soft folds up to `fold_depth` deep, into a `blouse` over the waistband and a `puff`
            over each cuff; over its last `tuck` it goes in `under` the waistband and the cuffs (their tops roll down
            onto it); smoothed (`relax`) and kept `gap` off the skin (vertices and triangle centres); under each arm a
            sheet of the knit joins the sleeve to the body, `underarm` metres down her side and out along the sleeve,
            sagging `underarm_sag` of its width (`underarm`; 0 for none)
  waistband `above_hip` (its top) over the hip joints (landmark leg.L), `height` deep, snug (`offset`), its hem rolled
  cuffs     `length` up the forearm from the wrist, `past` beyond it, snug, rolled at the hand
  ink       drawn lines (outfit_ink, `width` wide): ribs across the band, the waistband and the cuffs every `band` /
            `waistband` / `cuff` metres, a fold line along the band, the cleavage on the skin above it, and with `folds`
            the knit's lines: under each breast where the knit leaves it, folds below the bust and over the waistband,
            inside the elbows and over the cuffs (`enabled` true)
  seed      the folds' and the lines' random draw

Spec: `[sweater]` (DEFAULTS below, tables merge key by key) and `[colors.sweater]` knit (the body and sleeves), rib (the
band, waistband and cuffs) and ink (the lines). Materials セーター (knit) and セーターリブ (ribs), double sided, outlined
dark, and セーター線 (the ink)."""
import numpy as np

from .. import spec as SP
from ..build import builder
from ..part import Part
from . import outfit as OU
from . import outfit_geo as G
from . import outfit_ink as INK
from . import outfit_shell as SH
from . import outfit_wear as WR

DEFAULTS = {
    "band": {"chest": 0.055, "rise": 0.030, "arm": 0.030, "corner": 0.010, "height": 0.058, "offset": 0.010,
             "roll": 0.008, "relax": 8},
    "body": {"offset": 0.008, "ease": 0.016, "taper": 0.03, "taper_back": 0.07, "cling": 0.070, "blouse": 0.010,
             "folds": 11, "fold_depth": 0.016, "sleeve_ease": 0.009, "sleeve_taper": 0.10, "sleeve_sag": 0.008,
             "sleeve_folds": 7, "puff": 0.012, "tuck": 0.040, "under": 0.0022, "drop": 0.040, "relax": 8, "gap": 0.003,
             "underarm": 0.08, "underarm_sag": 0.12},
    "waistband": {"above_hip": 0.095, "height": 0.034, "offset": 0.006, "relax": 4},
    "cuff": {"length": 0.062, "past": 0.004, "offset": 0.006, "relax": 4},
    "ink": {"enabled": True, "width": 0.0026, "band": 0.040, "waistband": 0.040, "cuff": 0.024, "folds": True},
    "seed": 7,
}
COLORS = {"knit": "#e7a7eb", "rib": "#9f75b0", "ink": "#2e2234"}
MAT_KNIT = "セーター"
MAT_RIB = "セーターリブ"
MAT_INK = "セーター線"


def fields(sh, fit, c):
    """Per-vertex fields on the skin shell `sh` (dict of (n,)): z, s (arm coordinate), arm (the arm bones' share),
    band (depth below the band's top line), and the sizes they are cut at."""
    S, L = fit.S, fit.L
    V = sh.verts
    s = SH.both_sides(V, L)
    b = c["band"]
    z_top = float(L["arm.L"][2]) - b["chest"] * S
    x_arm = 0.5 * (abs(float(L["arm.L"][0])) + abs(float(L["arm.R"][0])))
    rise = b["rise"] * S * np.clip(np.abs(V[:, 0]) / x_arm, 0.0, 1.0) ** 2           # up to the arms: one band, no notch
    band = SH.soft_max(z_top + rise - V[:, 2], s - b["arm"] * S, b["corner"] * S)
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


def underarm(knit, P, fit, c, rows=8, cols=9):
    """The cloth under each arm (`body.underarm` metres down her side and out along the sleeve, 0 for none): a sheet from
    the sleeve's underside to the body's side, hanging `body.underarm_sag` of its width between them, so the sleeve and
    the body are one garment, as a loose sweater's are, not a tube beside a tube. The band covers the armpit, where the
    knit `knit` (at positions P) of the arm and of the trunk meet; the sheet starts under it: each row runs from the
    sleeve's lowest point at a length along the arm (from where the sleeve starts) to the body's outermost point at a
    height (from the top of its side), its edges tucked a millimetre into the knit, its weights between the two.
    Returns outfit_geo Patches (the knit's material and tiles)."""
    S, bd = fit.S, c["body"]
    length = bd["underarm"] * S
    if length <= 0.0:
        return []
    V, a, s, z = knit.verts, knit.fields["arm"], knit.fields["s"], knit.fields["z"]
    tuck = P - knit.normals * 0.001 * S
    out = []
    for sg in (1.0, -1.0):
        sleeve, trunk = (sg * V[:, 0] > 0.0) & (a > 0.8), (sg * V[:, 0] > 0.0) & (a < 0.2)
        if not sleeve.any() or not trunk.any():
            continue
        s0, z0 = s[sleeve].min(), z[trunk].max()                        # where the sleeve and the side start, under the band
        ring_a, ring_b = [], []
        for t in np.linspace(0.0, 1.0, rows + 1):
            sel_a = np.flatnonzero(sleeve & (np.abs(s - s0 - t * length) < 0.006 * S))
            sel_b = np.flatnonzero(trunk & (np.abs(z - z0 + t * length) < 0.006 * S))
            if len(sel_a) and len(sel_b):
                ring_a.append(sel_a[np.argmin(z[sel_a])])                # the sleeve's underside there
                ring_b.append(sel_b[np.argmax(sg * V[sel_b, 0])])       # the body's side there
        if len(ring_a) < 2:
            continue
        u = np.linspace(0.0, 1.0, cols)
        pts, wts = [], []
        for ia, ib in zip(ring_a, ring_b):
            A_, B_ = tuck[ia], tuck[ib]
            sag = bd["underarm_sag"] * np.linalg.norm(B_ - A_) * 4.0 * u * (1.0 - u)
            pts += list((1.0 - u)[:, None] * A_ + u[:, None] * B_ - sag[:, None] * np.array([0.0, 0.0, 1.0]))
            wts += list((1.0 - u)[:, None] * knit.W[ia] + u[:, None] * knit.W[ib])
        faces = [(k * cols + j, (k + 1) * cols + j, (k + 1) * cols + j + 1, k * cols + j + 1)
                 for k in range(len(ring_a) - 1) for j in range(cols - 1)]
        pts = np.array(pts)
        uvs = [(pts[i, 0] / WR.TILE, -pts[i, 2] / WR.TILE) for f in faces for i in f]
        W = np.array(wts)
        patch = G.Patch(pts, faces, np.array(uvs), {b: W[:, i] for i, b in enumerate(knit.bones) if W[:, i].max() > 1e-6},
                        0, "underarm")
        out.append(G.orient_outward(patch, pts.mean(0) + np.array([0.0, 0.1, 0.0])))
    return out


def _wrap(a):
    """Turn fractions as signed offsets from 0 (-0.5 .. 0.5)."""
    return (np.asarray(a, float) + 0.5) % 1.0 - 0.5


def ink(fit, c, pieces, centre_y, z_wb, end_s, fronts, rng):
    """The drawn lines on the sweater (`[sweater.ink]`): (outfit_ink lines, their widths as shares of `width`). Ribs
    across the band, the waistband and the cuffs every `band` / `waistband` / `cuff` metres, a broken fold line along
    the band, the cleavage on the skin above it (each breast's outline through the band's top edge in the middle), and
    with `folds` the knit's own lines: under each breast where the knit leaves it (`cling` round the breasts' `fronts`)
    and a fold below each, where it blouses over the waistband, inside the elbows and where the sleeves gather over the
    cuffs. `pieces`: {name: (Shell, positions)}, "skin" the body's own."""
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
    if fronts:                                                  # the breasts' inner tops above the band (the cleavage):
        sh = pieces["skin"][0]                                  # each breast's outline through the band's top edge in
        chest = sh.clip(0.04 * S + sh.fields["band"])           # the middle, inside its front, on the skin up to 4 cm
        chest = chest.clip(0.004 * S - chest.fields["band"])    # x S above the band (made finer, so a line is smooth)
        chest = chest.clip(0.3 - chest.fields["arm"]).subdivide().subdivide()
        P = chest.verts
        mid = [p for ln in INK.isolines(chest, P, chest.fields["band"], 0.0) for p in ln[0] if p[1] < centre_y]
        if mid:
            c0 = min(mid, key=lambda p: abs(p[0]))
            chest.fields["x"] = chest.verts[:, 0]
            for f in fronts:
                add(INK.arcs(chest, P, np.linalg.norm(P - f, axis=1), float(np.linalg.norm(c0 - f)),
                             lambda fl, f=f: np.sign(f[0]) * fl["x"] < 0.7 * abs(float(f[0])), fields=("x",)), 0.9)
    sh, P = pieces["waistband"]
    add(INK.ribs(sh, P, "az", k["waistband"] * S, "z", z_wb - c["waistband"]["height"] * S, z_wb, rng), 0.8)
    sh, P = pieces["cuffs"]
    add(INK.ribs(sh, P, "la", k["cuff"] * S, "s", end_s, end_s + (cf["length"] + cf["past"]) * S, rng), 0.8)
    if not k["folds"]:
        return lines, widths
    sh, P = pieces["body"]
    z, s = sh.fields["z"], sh.fields["s"]
    sh.fields["x"] = sh.verts[:, 0]
    on_arm = sh.fields["arm"][sh.tris]
    trunk = (on_arm < 0.3).all(1)
    if fronts:                                                  # under each breast, where the knit leaves it
        sh.fields["fr"] = _from_fronts(sh.verts, fronts)
        cl, zf = c["body"]["cling"] * S, min(float(f[2]) for f in fronts)
        under = INK.arcs(sh, P, sh.fields["fr"], cl - 0.004 * S, lambda fl: fl["z"] < zf - 0.004 * S, fields=("z",),
                         tris=trunk)
        add(INK.dashes(under, rng, (0.03 * S, 0.08 * S), (0.003 * S, 0.010 * S)), 1.1)
        for f in fronts:                                        # and a fold below it, towards the middle
            xc, hw = 0.55 * float(f[0]), rng.uniform(0.012, 0.022) * S
            add(INK.arcs(sh, P, z, zf - cl - rng.uniform(0.006, 0.018) * S,
                         lambda fl, xc=xc, hw=hw: (np.abs(fl["x"] - xc) <= hw) & (np.abs(_wrap(fl["az"])) < 0.2),
                         fields=("x", "az"), tris=trunk), 1.0)
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
    band = upper.band(upper.fields["band"], 0.0, h)
    Pb = band.offset(b["offset"] * S, relax=b["relax"], fit=fit, gap=b["offset"] * S * 0.8)
    loops = _loops_by(band, "band")                                         # the top rolled to the skin, the bottom
    rims = loops[:2]                                                        # onto the knit under it
    _add(soup, band, Pb, fit, centre_y, 1, torso_reps, rims,
         [b["roll"] * S, (b["offset"] - c["body"]["offset"]) * S][:len(rims)], "band")
    info["band_top_z"] = float(band.verts[band.fields["band"] < 1e-4 * S][:, 2].min()) if len(band.verts) else None
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
    Pk = body.settle(hang(body, base, fit, c, centre_y, h, z_wb, end_s, fronts, rng), relax=bd["relax"], fit=fit,
                     gap=bd["gap"] * S)
    _add(soup, body, Pk, fit, centre_y, 0, torso_reps, (), 0.0, "body")
    webs = underarm(body, Pk, fit, c)
    for web in webs:
        soup.add(web, 0)
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
        pieces = {"band": (band, Pb), "body": (body, Pk), "waistband": (ring, Pw), "cuffs": (cuff, Pc),
                  "skin": (base, base.verts)}
        lines, widths = ink(fit, c, pieces, centre_y, z_wb, end_s, fronts, rng)
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
