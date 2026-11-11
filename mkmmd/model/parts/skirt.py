"""The skirt (part `skirt`, numpy only): a short A-line skirt of denim hung from her hips, a frill of lace sewn under
its hem, and chains that let it swing.

It is rings round a vertical axis through the middle of her hips, from `top` over the hip joints (landmark leg.L) down
`length`: each ring the convex section of her body without her arms (the cloth spans the gap between her thighs), `ease`
off it in the waistband (`waistband` deep) and `hip_ease` below, hanging from what is above it and coming in no faster
than `taper`, and standing `flare` further out at the hem (an A-line: the flare grows down the skirt as the share of the
way ** `flare_power`), with `waves` soft folds `wave_depth` deep at the hem. Its top edge goes in to her skin and its hem
is turned up inside, `thick` the cloth; nothing of it comes closer than `gap` to her skin.

  lace     (`enabled`) a frill of lace sewn inside the hem and hanging `height` under it, in soft pleats every `pleat`
           metres, `depth` deep, standing `flare` out at its hem (the sweater's lace)
  chains   `columns` chains of `rows` bones round her (スカートcc_r, family skirt; column 0 at the middle of her front,
           counted toward her left), hung from the lower body (下半身), their heads `first` of the way down the skirt.
           The skirt carries the weights of the skin under it, her hips' and her thighs'; below the chains' heads
           (letting go over HOLD above them) it gives the chains all of that but `follow`: at 0 a full skirt that only
           swings, at 0.5 (the default) a fitted one that a lifting thigh carries half the way while the chains swing
           it and her colliders push it. The first bone is `root` long (a chain's first capsule collides with nothing
           of her body, so it is short), the others share the rest of the way evenly, the last ending at the lace's
           hem; their capsules are `radius` thick, or as thin as the room to the body's colliders at rest asks
           (outfit_rig.Rig.ring), and each chain is tied sideways to its neighbours round the ring, where MMD's
           physics or mk's solver ([sim] families "skirt") moves them; `enabled` false leaves them out (the skirt then
           follows her skin alone)
  denim    the skirt is painted whole (`picture`): a twill of ribs `twill` metres apart and `depth` deep with the
           yarn's streaks down it, `fade` lighter where it wears (over her thighs, over her seat, along the hem), its
           seams and their topstitching (`stitch` metres a stitch) in the thread's colour: the waistband with its loops
           and button, the side seams, the fly, two front pockets, and behind a yoke and two patch pockets (`pockets`),
           two rows round the hem
  seed     the draw of the folds and of the yarn

Sizes are metres x S (S = head_tip.z / 1.7 with leg_extra left out). Spec: `[skirt]` (DEFAULTS below, tables merge key
by key) and `[colors.skirt]`: `denim`, `thread`, `lace`. Materials デニム and スカートレース, double sided, outlined dark."""
import numpy as np

from .. import spec as SP
from ..build import builder
from ..part import Part
from . import outfit as OU
from . import outfit_fit as OF
from . import outfit_geo as G
from . import outfit_rig as ORIG
from . import outfit_wear as WR
from .outfit_dress import frill_base, hang_frill
from .outfit_skirt import bone_names
from .sweater import LACE_TILE, circ_smooth, hull_radii, row_smooth, waves, with_chains

TAU = 2.0 * np.pi
DEFAULTS = {
    "top": 0.085, "length": 0.215, "waistband": 0.028, "ease": 0.004, "hip_ease": 0.007, "taper": 0.30, "flare": 0.030,
    "flare_power": 1.5, "waves": 5, "wave_depth": 0.004, "thick": 0.0022, "gap": 0.003, "per_column": 10, "step": 0.006,
    "lace": {"enabled": True, "height": 0.028, "pleat": 0.012, "depth": 0.0035, "flare": 0.004},
    "chains": {"enabled": True, "columns": 12, "rows": 3, "first": 0.30, "root": 0.02, "radius": 0.022, "follow": 0.5},
    "denim": {"twill": 0.0016, "depth": 0.10, "fade": 0.16, "stitch": 0.0032, "pockets": True},
    "seed": 11,
}
COLORS = {"denim": "#4a6ad8", "thread": "#f2b24c", "lace": "#fbf7f2"}
MAT = "デニム"
MAT_LACE = "スカートレース"
HOLD = 0.03                     # metres above its chains' heads over which her hips let go of the skirt
PX = (2048, 512)                # the skirt's picture: round her, down it
SEAM = (0.0007, 0.0022)         # a seam in the picture: half its groove, and how far beside it its edge is worn light
THREAD = 0.00055                # half the width of a row of topstitching


def shape(fit, c, rng):
    """(rings (K, M, 3) from the skirt's top down to its hem, the y of the vertical axis they go round, the rings'
    heights): module docstring."""
    S = fit.S
    ch = c["chains"]
    M = max(int(ch["columns"]), 3) * int(c["per_column"])
    z_top = float(fit.L["leg.L"][2]) + c["top"] * S
    zs = np.linspace(z_top, z_top - c["length"] * S, max(int(round(c["length"] / c["step"])) + 1, 6))
    core = fit.core.verts[np.unique(fit.core.tris)]
    hips = core[np.abs(core[:, 2] - float(fit.L["leg.L"][2])) < 0.03 * S]
    centre_y = float(0.5 * (hips[:, 1].min() + hips[:, 1].max())) if len(hips) else 0.0
    th = G.ring_theta(M)
    dirs = G.hdir(th)
    u, v = np.array([0.0, -1.0, 0.0]), np.array([1.0, 0.0, 0.0])
    H, last = [], np.full(M, 0.12 * S)
    for z in zs:
        r = fit.core.outline(np.array([0.0, centre_y, z]), u, v, M, "far", rmin=0.02 * S, rmax=0.35 * S)
        last = last if r is None else hull_radii(circ_smooth(r, 3), dirs[:, :2])
        H.append(last)
    H = row_smooth(np.array(H), 2)
    s = (z_top - zs) / max(z_top - zs[-1], 1e-9)                          # 0 at its top .. 1 at its hem
    under = G.smoothstep(np.clip((z_top - zs) / max(c["waistband"] * S, 1e-9) - 0.5, 0.0, 1.0))
    ease = (c["ease"] + (c["hip_ease"] - c["ease"]) * under) * S
    R = H + ease[:, None]
    for k in range(1, len(zs)):                  # it hangs from what is above it, coming in no faster than the taper
        R[k] = np.maximum(R[k], R[k - 1] - c["taper"] * (zs[k - 1] - zs[k]))
    R = circ_smooth(row_smooth(R, 2), 3)
    R = R + (c["flare"] * S * s ** c["flare_power"])[:, None]
    R = R + c["wave_depth"] * S * (s ** 2)[:, None] * waves(th, c["waves"], rng)[None, :]
    rings = np.array([np.array([0.0, centre_y, z]) + dirs * R[k][:, None] for k, z in enumerate(zs)])
    return fit.conform(rings, c["gap"] * S), centre_y, zs


def _along(U, V, pts):
    """(metres from the points (U, V) to the polyline `pts` (k, 2), metres along it at the nearest point, its length)."""
    best, at, run = np.full(U.shape, np.inf), np.zeros(U.shape), 0.0
    for a, b in zip(pts[:-1], pts[1:]):
        ab = np.asarray(b, float) - np.asarray(a, float)
        L = float(np.hypot(*ab))
        if L < 1e-9:
            continue
        t = np.clip(((U - a[0]) * ab[0] + (V - a[1]) * ab[1]) / (L * L), 0.0, 1.0)
        d = np.hypot(U - a[0] - t * ab[0], V - a[1] - t * ab[1])
        near = d < best
        best, at = np.where(near, d, best), np.where(near, run + t * L, at)
        run += L
    return best, at, run


def picture(col, around, length, wb, c, rng):
    """The skirt's picture (PX[1], PX[0], 4) uint8: across it the way round her from the middle of her front toward her
    left, once; down it from the skirt's top (row 0) to its hem. `around` and `length` are its metres, `wb` its
    waistband's height, `col` the colours (rgb 0..1) `denim` and `thread`. Module docstring: denim."""
    w, h = PX
    dn = c["denim"]
    x = (np.arange(w) + 0.5) / w * around
    X, Y = np.meshgrid(x, (np.arange(h) + 0.5) / h * length)

    def off(x0):                                          # metres round her from x0, the short way
        return (X - x0 + 0.5 * around) % around - 0.5 * around
    front, back, side = off(0.0), off(0.5 * around), 0.25 * around
    ribs = max(round(around / dn["twill"]), 1)            # whole ribs round her: the twill closes on itself
    shade = 1.0 - dn["depth"] * (0.5 + 0.5 * np.cos(TAU * (ribs * X / around + Y / dn["twill"])))
    k = np.arange(1, 120)
    amp, ph = rng.uniform(0.2, 1.0, len(k)) / np.sqrt(k), rng.uniform(0.0, TAU, len(k))
    slub = (amp[None, :] * np.sin(TAU * k[None, :] * x[:, None] / around + ph[None, :])).sum(1) / amp.sum()
    shade = shade * (1.0 + 0.30 * dn["depth"] * slub[None, :]) + 0.012 * rng.normal(size=(h, w))
    worn = (np.exp(-(front / (0.13 * around)) ** 2) + np.exp(-(back / (0.13 * around)) ** 2)) * \
        np.exp(-((Y / length - 0.55) / 0.35) ** 2) + 0.8 * np.clip(1.0 - (length - Y) / 0.0025, 0.0, 1.0)
    rgb = col["denim"][None, None, :] * shade[..., None]
    rgb = rgb + (1.0 - rgb) * (0.55 * dn["fade"] * np.clip(worn, 0.0, 1.0))[..., None]
    low = Y > wb                                          # under the waistband

    def seam(d, where=True):                              # a groove, and its edge worn light either side
        nonlocal rgb
        g = np.clip(1.0 - d / SEAM[0], 0.0, 1.0) * where
        e = np.exp(-((d - SEAM[1]) / (0.5 * SEAM[1])) ** 2) * where
        rgb = rgb * (1.0 - 0.45 * g)[..., None]
        rgb = rgb + (1.0 - rgb) * (0.5 * dn["fade"] * e)[..., None]

    def stitch(d, s, where=True):                         # a row of topstitching: dashes of thread along a line
        nonlocal rgb
        on = np.clip(1.5 - d / THREAD, 0.0, 1.0) * ((s % dn["stitch"]) < 0.72 * dn["stitch"]) * where
        rgb = rgb + (col["thread"][None, None, :] - rgb) * on[..., None]

    def line(pts, U, rows=(0.0,), where=True, groove=True):       # a sewn line: its seam and rows of stitching beside it
        d, s, total = _along(U, Y, np.asarray(pts, float))
        where = where & (s > 1e-6) & (s < total - 1e-6)           # it stops at its ends: no cap round them
        if groove:
            seam(d, where)
        for r in rows:
            stitch(np.abs(d - r), s, where)
    seam(np.abs(Y - wb))                                  # the waistband: its seam, a row along each of its edges
    for y0 in (0.003, wb - 0.003):
        stitch(np.abs(Y - y0), X)
    for y0 in (length - 0.006, length - 0.013):           # the hem: two rows
        stitch(np.abs(Y - y0), X)
    for x0, way in ((side, 1.0), (around - side, -1.0)):  # the side seams, felled toward her back
        seam(np.abs(off(x0)), low)
        stitch(np.abs(off(x0) - way * 0.0045), Y, low)
    below = length - wb                                   # what is sewn on it takes its size from the skirt's length
    seam(np.abs(front), low)                              # the fly: the front seam and its J of stitching
    fly = wb + 0.48 * below
    line([(0.026, wb), (0.026, fly - 0.027), (0.019, fly - 0.009), (0.0, fly)], front, (0.0, 0.005), low, False)
    t = np.linspace(0.0, 0.5 * np.pi, 9)
    for way in (1.0, -1.0):                               # the front pockets: a curve from the waistband to the side seam
        line(np.stack([way * side * (1.0 - 0.58 * np.cos(t)), wb + 0.34 * below * np.sin(t)], 1), front, (0.0, 0.005),
             low, False)
    if dn["pockets"]:
        line([(-side, wb + 0.12 * below), (0.0, wb + 0.24 * below), (side, wb + 0.12 * below)], back, (0.004,), low)
        tall = 0.44 * below                               # the yoke, and under it two patch pockets
        half = 0.42 * tall
        for way in (1.0, -1.0):
            pu, pv = back - way * 0.34 * side, Y - (wb + 0.30 * below)
            d, s, _ = _along(pu, pv, np.array([(-half, 0.0), (half, 0.0), (half, 0.7 * tall), (0.0, tall),
                                               (-half, 0.7 * tall), (-half, 0.0)]))
            seam(d)
            inner = (np.abs(pu) < half) & (pv > 0.0) & (pv < tall * (0.7 + 0.3 * (1.0 - np.abs(pu) / half)))
            stitch(np.abs(d - 0.004), s, inner)                   # two rows of stitching inside each
            stitch(np.abs(d - 0.009), s, inner)
    for x0 in (0.07, around - 0.07, side, around - side, 0.5 * around):     # belt loops, tacked at both ends
        d = np.abs(off(x0))
        loop = (d < 0.006) & (Y > 0.001) & (Y < wb + 0.004)
        edge = loop & ((d > 0.0048) | (Y < 0.0022) | (Y > wb + 0.0028))
        rgb = np.where(loop[..., None], rgb * 1.10, rgb)
        rgb = np.where(edge[..., None], rgb * 0.62, rgb)
        for y0 in (0.004, wb + 0.001):
            stitch(np.abs(Y - y0), X * 3.0, loop)
    r = np.hypot(off(0.012), Y - 0.5 * wb)                # the button
    brass = col["thread"][None, None, :] * (1.15 - 0.5 * np.clip(r / 0.0075, 0.0, 1.0) ** 2)[..., None]
    rgb = np.where((r < 0.0075)[..., None], brass, rgb)
    rgb = np.where(((r < 0.0075) & (r > 0.0062) | (r < 0.0012))[..., None], rgb * 0.55, rgb)
    return WR._rgba(rgb)


@builder("skirt", needs=("body",))
def build(ctx):
    c = SP.merge(DEFAULTS, ctx.cfg)
    fit = OU.make_fit(ctx)
    col = WR.colours(ctx, "skirt", COLORS)
    S = fit.S
    rng = np.random.default_rng(int(c["seed"]))
    lc, ch = c["lace"], c["chains"]
    laced = bool(lc["enabled"])
    soup = G.Soup("skirt", [MAT] + ([MAT_LACE] if laced else []))
    rings, centre_y, zs = shape(fit, c, rng)
    K, M = rings.shape[:2]
    out = np.broadcast_to(G.hdir(G.ring_theta(M)), rings.shape)
    thick, up = c["thick"] * S, np.array([0.0, 0.0, 1.0])
    edges = [rings[0] - out[0] * thick - up * 0.5 * thick,              # its top edge, in toward her skin
             rings[-1] - out[-1] * thick, rings[-1] - out[-1] * thick + up * 0.010 * S]        # its hem, turned up inside
    top, hem, turned = (fit.push_out(e, 0.4 * c["gap"] * S) for e in edges)
    grid = np.concatenate([[top], rings, [hem, turned]])
    down = (zs[0] - zs) / max(zs[0] - zs[-1], 1e-9)
    v = np.concatenate([[1.0], 1.0 - down, [0.0, 0.03]])
    axis = np.concatenate([np.zeros_like(grid[..., :1]), np.full_like(grid[..., :1], centre_y), grid[..., 2:3]], -1)
    patch = G.orient_outward(G.loft(grid, True, u=np.linspace(0.0, 1.0, M + 1), v=v, mat=0, tag="skirt"),
                             axis.reshape(-1, 3))
    donors = [b for b in fit.skin.weights if not any(b[1:].startswith(t) for t in OF.HAND_STEMS)]
    patch.w = fit.weights_for(patch.v, donors, default="下半身", faces=patch.f, smooth=2)
    lace = lace_rings = None
    if laced:                                             # a frill of lace sewn inside the hem, hanging under it
        centre = np.array([0.0, centre_y, float(zs[-1])])
        ring = rings[-1] - out[-1] * 0.7 * thick
        pleats = max(int(round(float(np.linalg.norm(np.roll(ring, -1, 0) - ring, axis=1).sum()) / (lc["pleat"] * S))), 8)
        base, away, _ = frill_base(ring, centre, pleats)
        prof = np.array([(0.0, 0.0), (0.3, 0.33), (0.7, 0.67), (1.0, 1.0)]) * [lc["flare"] * S, lc["height"] * S]
        lace, lace_rings = hang_frill(base, away, -up, prof, pleats, lc["depth"] * S, 1, "lace", centre,
                                      tile_u=LACE_TILE * S, seed=int(c["seed"]) + 3, fit=fit, margin=c["gap"] * S,
                                      hem_wobble=0.08 * lc["height"] * S)
        lace.w = fit.weights_for(lace.v, donors, default="下半身", faces=lace.f, smooth=2)
    rig = ORIG.Rig()
    info = {"z_top": float(zs[0]), "z_hem": float(zs[-1]), "min_gap": float(fit.skin.closest(rings.reshape(-1, 3))[0].min()),
            "chains": 0, "rows": 0, "radii": []}
    if ch["enabled"]:
        C, R = max(int(ch["columns"]), 3), max(int(ch["rows"]), 1)
        names = bone_names(C, R)
        z_end = float(lace_rings[-1][:, 2].min()) if laced else float(zs[-1])
        z_free = float(zs[0] - ch["first"] * (zs[0] - zs[-1]))                # the chains' heads
        hold = min(z_free + HOLD * S, float(zs[0]))                           # down to here her hips carry all of it
        nodes = ORIG.hung_heights(z_free, z_end, R, ch["root"] * S)
        at = np.clip((zs[0] - nodes) / max(zs[0] - zs[-1], 1e-9), 0.0, 1.0) * (K - 1)
        k0 = np.minimum(at.astype(int), K - 2)
        pts = np.zeros((C, R + 1, 3))
        for col_i in range(C):
            j = col_i * (M // C)
            pts[col_i, :R] = rings[k0, j] + (at - k0)[:, None] * (rings[k0 + 1, j] - rings[k0, j])
            end = lace_rings[-1] if laced else rings[-1]
            turn = (G.theta_of(end, centre_y) - col_i * TAU / C + np.pi) % TAU - np.pi
            pts[col_i, R] = end[int(np.argmin(np.abs(turn)))]
        body = ctx.find_body("下半身")
        statics = [rb for rb in ctx.need("body").bodies if rb.mode == "static"]
        rad = rig.ring(names, pts, "下半身", "skirt", ch["radius"] * S, statics, body.name if body is not None else None,
                       "Skirt")
        for p in (patch, lace):                               # below the chains' heads it goes over to them, but `follow`
            if p is not None:
                share = ORIG.ring_weights(G.theta_of(p.v, centre_y), p.v[:, 2], names, nodes, hold, {}, C)
                p.w = with_chains(p.w, {b: (1.0 - float(ch["follow"])) * w for b, w in share.items()})
        info.update(chains=C, rows=R, radii=[float(rad.min()), float(rad.max())], z_free=z_free, z_hold=hold)
    soup.add(patch)
    soup.add(lace)
    around = float(np.linalg.norm(np.roll(rings, -1, 1) - rings, axis=2).sum(1).mean())
    mats = [WR.material(ctx, MAT, "denim", col["denim"], picture(col, around, c["length"] * S, c["waistband"] * S, c, rng),
                        shadow=(0.70, 0.72, 0.90))]
    if laced:
        mats.append(WR.material(ctx, MAT_LACE, "lace", col["lace"], WR.lace(col["lace"]), shadow=(0.86, 0.84, 0.92),
                                edge_size=0.5))
    mesh = soup.mesh()
    front = rings[:, 0]
    info["landmarks"] = {"skirt_top": front[0], "skirt_hem": front[-1]}
    ctx.log(f"skirt from {info['z_top']:.3f} m down to {info['z_hem']:.3f} m, closest to the skin "
            f"{info['min_gap'] * 1000:.1f} mm, {info['chains']} chains of {info['rows']}"
            + (f", capsules {info['radii'][0] * 1000:.0f} to {info['radii'][1] * 1000:.0f} mm" if info["radii"] else ""))
    return Part("skirt", meshes=[mesh], materials=[m for m in mats if m.name in mesh.mats], bones=rig.bones,
                bodies=rig.bodies, joints=rig.joints, info=info)
