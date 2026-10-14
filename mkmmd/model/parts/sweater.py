"""The off-shoulder sweater (part `sweater`, numpy only), made as knitwear is made: separate pieces with clean, even
rows of stitches (lofted rings), each with the thickness of knit at its edges, and the bands ribbed in their shape, not
only in their colour:

  band       a thick folded rib round the top of her body: her chest, her upper arms and her back as one piece, the
             shoulders bare. Its top line is `chest` below the shoulder joints were her arms hanging straight down
             (round her chest and back that line is level, round each upper arm it is a ring across the arm; the two
             meet over the front and the back of the armpit; the arm bones' share of the skin decides, within `reach`
             of the upper arm's bone, fading over `fade`, so the shoulder girdle's share of her upper back does not bend
             it), plus `under` metres right under each arm (fully within a third of `under_span` turns of straight under
             it, fading out at `under_span`, and only within `under_width` of the joint's depth): a slit the arm hides.
             The band is `height` deep below that line. Its rows are pulled taut as an elastic rib is: round her chest
             and back in their own level planes (across the cleavage), on the skin round her arms, out of the skin's
             hollows (Skin.bridge, a ball of `bridge`), smoothed `smooth` times, and `turn_smooth` times more where it
             turns from her body onto an arm (so each row runs on over the fold there). It stands `offset` off them
             (along their normals, smoothed, so a tight crease does not fold its face over itself), slightly full in
             the middle, its top rolled over to the skin (a round of half its thickness, closing down onto her skin
             where the band spans a hollow), its bottom a rounded lip `edge` thick going back in under it; ribbed every
             `rib` metres, `rib_depth` deep, the ribs running down it
  body       the knit body from under the band down into the waistband, rings round a vertical axis through her
             waist's middle: each the convex section of her torso (the cloth spans her cleavage and the small of her
             back) `ease_*` off it (front, side, back), hanging from what is above it, coming in no faster than
             `taper_*` (so it falls from the bust and the shoulder blades instead of following her in at the waist),
             under each arm standing `pit_ease` more off her side from `pit_drop` below the shoulder joint up (an
             armhole sits below the armpit), a `blouse` over the waistband (its fullest a third of `blouse_height` up),
             gathered into it in `pleats` soft folds `pleat_depth` deep, and a few long folds from the bust down
             (`drape`); where it comes out from under the band, `gathers` soft pleats round it, `gather_depth` deep at
             the band's edge, fading over `gather_fade`
  sleeves    rings along each arm from where the band covers a whole ring by `start_depth` to the cuff: the convex
             section of the arm's own skin, `ease` off it (a table along the arm, 0 at the shoulder, 1 at the cuff),
             sagging `sag` under it at rest and `underarm` more by the armpit (gone 60 % of the way to the elbow), two
             soft folds `elbow_fold` deep in the crook of the elbow (`elbow_at` metres from it), a `puff` over the cuff
             and gathered into it in `pleats` folds `pleat_depth` deep over its last `gather` metres; gathered under the
             band as the body is (`gathers`, `gather_depth`, `gather_fade`)
  under the band  the knit lies at most `inside` x `offset` beyond the band's inner side (in along the skin's normal,
             so it never shows through the band) and carries the band's weights (it moves with the band), leaving it
             over half of `drop` below the band's edge; nothing of it lies above the band's top line (bare shoulders).
             Its vertices stay `gap` off the skin
  waistband  `above_hip` (its top) over the hip joints (landmark leg.L), `height` deep, `offset` off the convex section
             of her hips, `thick`, ribbed (`rib`, `rib_depth`), its hem rolled to the inside and its top a lip the knit
             goes in under
  cuffs      `length` up the forearm from the wrist, `past` beyond it, the same make as the waistband
  seed       the folds' draw

Sizes are metres x S (S = head_tip.z / 1.7 with leg_extra left out). Every piece carries the weights of the skin nearest
to it, smoothed round its rings over `spread` metres (band, body), so the sweater moves with her in any pose; on her
body away from her arms the band and the knit leave out the shoulder girdle's share, so they stay with her chest when
an arm moves. Spec: `[sweater]` (DEFAULTS below, tables merge key by key) and `[colors.sweater]` knit (the body and
sleeves) and rib (the band, waistband and cuffs). Materials セーター (knit) and セーターリブ (ribs), double sided,
outlined dark."""
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

TAU = 2.0 * np.pi

DEFAULTS = {
    "band": {"chest": 0.022, "under": 0.14, "under_span": 0.14, "under_width": 0.020, "height": 0.050,
             "reach": 0.050, "fade": 0.025, "bridge": 0.10, "smooth": 12, "offset": 0.012,
             "edge": 0.003, "rib": 0.009, "rib_depth": 0.0010, "spread": 0.03, "turn_smooth": 300},
    "body": {"ease_front": 0.008, "ease_side": 0.010, "ease_back": 0.008, "taper_front": 0.18, "taper_side": 0.40,
             "taper_back": 0.30, "blouse": 0.008, "blouse_height": 0.05, "pleats": 22, "pleat_depth": 0.004,
             "drape": 0.004, "inside": 0.3, "drop": 0.035, "gap": 0.003, "spread": 0.02, "gathers": 36,
             "gather_depth": 0.0025, "gather_fade": 0.03, "pit_drop": 0.06, "pit_ease": 0.012},
    "sleeve": {"ease": [[0.0, 0.009], [0.35, 0.010], [0.6, 0.012], [0.85, 0.015], [1.0, 0.016]], "sag": 0.008,
               "puff": 0.004, "pleats": 9, "pleat_depth": 0.0025, "gather": 0.06, "elbow_fold": 0.0025,
               "elbow_at": [-0.012, 0.012], "gathers": 11, "gather_depth": 0.003, "gather_fade": 0.04,
               "start_depth": 0.006, "underarm": 0.03},
    "waistband": {"above_hip": 0.095, "height": 0.034, "offset": 0.004, "thick": 0.005, "rib": 0.008,
                  "rib_depth": 0.0010},
    "cuff": {"length": 0.062, "past": 0.004, "offset": 0.004, "thick": 0.005, "rib": 0.007, "rib_depth": 0.0008},
    "seed": 7,
}
COLORS = {"knit": "#e7a7eb", "rib": "#9f75b0"}
MAT_KNIT = "セーター"
MAT_RIB = "セーターリブ"


# ---------------------------------------------------------------- plane helpers
def convex_hull(P):
    """Convex hull (h, 2) of 2d points, counter-clockwise (monotone chain)."""
    P = np.unique(np.round(np.asarray(P, float), 9), axis=0)
    if len(P) < 3:
        return P

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    lower, upper = [], []
    for p in P:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    for p in P[::-1]:
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return np.array(lower[:-1] + upper[:-1])


def ray_hull(H, d):
    """(m,) distance along unit rays d (m, 2) from the origin to the convex polygon H (h, 2) round it (-inf: missed)."""
    E = np.roll(H, -1, 0) - H

    def cr(a, b):
        return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]
    den = cr(d[:, None, :], E[None])
    with np.errstate(divide="ignore", invalid="ignore"):
        t = cr(H[None], E[None]) / den
        s = cr(H[None], d[:, None, :]) / den
    ok = (np.abs(den) > 1e-15) & (s >= -1e-9) & (s <= 1.0 + 1e-9) & (t > 0.0)
    return np.where(ok, t, -np.inf).max(1)


def hull_radii(r, dirs):
    """Radii (m,) of the convex hull of a star-shaped outline: points dirs (m, 2) x r (m,), at the same directions."""
    H = convex_hull(dirs * np.asarray(r, float)[:, None])
    return np.maximum(ray_hull(H, dirs), r)


def circ_smooth(r, passes, lam=0.5):
    """Laplacian smoothing round the last axis (periodic)."""
    r = np.array(r, float)
    for _ in range(passes):
        r = r + lam * (0.5 * (np.roll(r, 1, -1) + np.roll(r, -1, -1)) - r)
    return r


def row_smooth(R, passes, lam=0.5):
    """Laplacian smoothing along the first axis, its ends kept."""
    R = np.array(R, float)
    for _ in range(passes):
        R[1:-1] = R[1:-1] + lam * (0.5 * (R[:-2] + R[2:]) - R[1:-1])
    return R


def waves(theta, count, rng, spread=3):
    """(m,) smooth irregular waves round a ring (theta in radians), -1..1: a few neighbouring wave counts with random
    strengths and phases, so no two folds are alike."""
    ks = np.arange(max(int(count) - spread, 1), int(count) + spread + 1)
    amp = rng.uniform(0.5, 1.0, len(ks)) * np.exp(-0.5 * ((ks - count) / max(spread * 0.6, 1.0)) ** 2)
    ph = rng.uniform(0.0, TAU, len(ks))

    def w(t):
        return (amp[None, :] * np.sin(np.asarray(t, float)[:, None] * ks[None, :] + ph[None, :])).sum(1)
    return w(theta) / np.abs(w(np.linspace(0.0, TAU, 721))).max()


# ---------------------------------------------------------------- where the band is, on the skin
def band_depth(V, share, fit, c):
    """(depth (n,), arm blend (n,), near the arm (n,)) of the skin's points V: how far below the band's top line each
    lies (see the module docstring: level round the torso, a ring across each upper arm were it hanging down, blended by
    the arm bones' `share` near the arm, plus the slit under each arm), how much of the arm's way it took (0 torso,
    1 arm), and how near the upper arm's bone it is (1 within `reach`, 0 past `reach` + `fade`)."""
    S, L = fit.S, fit.L
    b = c["band"]
    V = np.asarray(V, float)
    z_line = float(L["arm.L"][2]) - b["chest"] * S
    near, hz = np.zeros(len(V)), V[:, 2].copy()
    for side, sg in (("L", 1.0), ("R", -1.0)):
        j, e = np.asarray(L[f"arm.{side}"], float), np.asarray(L[f"elbow.{side}"], float)
        t = np.arctan2(sg * (e[0] - j[0]), j[2] - e[2])                 # the upper arm out from straight down
        on = sg * V[:, 0] > 0.0
        dx, dz = sg * (V[on, 0] - j[0]), V[on, 2] - j[2]
        hz[on] = j[2] + dz * np.cos(t) - dx * np.sin(t)                  # turned with the arm to hang straight down
        a = G.unit(e - j)
        s = np.clip((V[on] - j) @ a, 0.0, float(np.linalg.norm(e - j)))
        d_cap = np.linalg.norm(V[on] - j - s[:, None] * a[None], axis=1)
        near[on] = G.smoothstep(np.clip((b["reach"] + b["fade"] - d_cap / S) / b["fade"], 0.0, 1.0))
    blend = share * near
    depth = z_line - ((1.0 - blend) * V[:, 2] + blend * hz)
    x_arm = 0.5 * (abs(float(L["arm.L"][0])) + abs(float(L["arm.R"][0])))
    y_arm = 0.5 * (float(L["arm.L"][1]) + float(L["arm.R"][1]))
    off = np.abs((SH.limb_angle(V, L) - 0.5 + 0.5) % 1.0 - 0.5)                  # turns from straight under the arm
    span = b["under_span"]
    under = G.smoothstep(np.clip((span - off) / (2.0 / 3.0 * span), 0.0, 1.0))
    by_arm = G.smoothstep(np.clip((np.abs(V[:, 0]) - (x_arm - 0.05 * S)) / (0.03 * S), 0.0, 1.0))
    across = G.smoothstep(np.clip((b["under_width"] * S - np.abs(V[:, 1] - y_arm)) / (0.015 * S), 0.0, 1.0))
    return depth + b["under"] * S * under * by_arm * across, blend, near


class Surface:
    """A triangle surface with per-vertex fields, asked about the points nearest to others."""

    def __init__(self, verts, tris, fields):
        self.verts, self.tris, self.fields = np.asarray(verts, float), np.asarray(tris, np.int64), fields

    def near(self, P):
        """(closest points (q, 3), {field: (q,) or (q, k) at them})."""
        q, tri, bary, _ = SK.closest_on_mesh(np.asarray(P, float).reshape(-1, 3), self.verts, self.tris)
        corners = self.tris[tri]
        out = {}
        for k, f in self.fields.items():
            f = np.asarray(f, float)
            out[k] = np.einsum("qc,qc...->q...", bary, f[corners])
        return q, out


# ---------------------------------------------------------------- the band
def _loop(sh, f, v):
    """The level line f == v on the shell `sh` (its longest piece) as a closed polyline (points, normals), counter-
    clockwise seen from above."""
    lines = INK.isolines(sh, sh.verts, f, v)
    P, N = (np.asarray(x, float) for x in max(lines, key=lambda x: len(x[0]))[:2])
    if np.allclose(P[0], P[-1]):
        P, N = P[:-1], N[:-1]
    if np.sum(P[:, 0] * np.roll(P[:, 1], -1) - np.roll(P[:, 0], -1) * P[:, 1]) < 0:
        P, N = P[::-1], N[::-1]
    return P, N


def _arc(P):
    seg = np.linalg.norm(np.roll(P, -1, 0) - P, axis=1)
    return np.concatenate([[0.0], np.cumsum(seg)]), float(seg.sum())


def _at(P, N, a):
    """Points and normals at arc lengths a along the closed polyline P."""
    s, tot = _arc(P)
    a = np.mod(a, tot)
    Pc, Nc = np.vstack([P, P[:1]]), np.vstack([N, N[:1]])
    i = np.clip(np.searchsorted(s, a, side="right") - 1, 0, len(P) - 1)
    t = ((a - s[i]) / np.maximum(s[i + 1] - s[i], 1e-12))[:, None]
    return Pc[i] * (1.0 - t) + Pc[i + 1] * t, G.unit(Nc[i] * (1.0 - t) + Nc[i + 1] * t)


def _runs(mask):
    """Maximal runs of True in a periodic bool array, as (first, last) index pairs (last may wrap below first)."""
    n = len(mask)
    if mask.all() or not mask.any():
        return [(0, n - 1)] if mask.all() else []
    start = int(np.flatnonzero(~mask)[0])
    out, i = [], None
    for k in range(1, n + 1):
        j = (start + k) % n
        if mask[j] and i is None:
            i = j
        if not mask[j] and i is not None:
            out.append((i, (j - 1) % n))
            i = None
    return out


ARM_TURN = 0.6      # round an arm the band's columns go this much by the turn round it, the rest by the line's length


def band_grid(loops, n, turn_of, w_of, arm_turn_of):
    """(points (K, n, 3), skin normals (K, n, 3), top line length): n columns on each of the closed lines `loops`
    ((points, skin normals), counter-clockwise from above, top first), even along the top line. Round her body (arm
    blend `w_of(points)` under 0.3 on the top line) each column is a constant turn round her were her arms
    hanging (`turn_of(points)`, radians, 0 her front's middle), so it runs straight down her chest and back as the ribs
    of a band do; round each arm, between the last of her body's columns on either side, each lies at the same share of
    the way round the arm (`arm_turn_of(points)`, turns, growing along the lines over the arm) as on the top line, so it
    runs along the arm."""
    rows, nrows, targets, top, arm, shares = [], [], None, None, None, None
    for P, N in loops:
        th = turn_of(P)
        i0 = int(np.argmin(np.abs((th + np.pi) % TAU - np.pi) + 10.0 * (P[:, 1] > np.median(P[:, 1]))))
        P, N, th = np.roll(P, -i0, 0), np.roll(N, -i0, 0), np.roll(th, -i0)
        s, tot = _arc(P)
        th = np.append(np.maximum.accumulate(np.unwrap(th)), th[0] + TAU)
        first = targets is None
        if first:
            top = tot
            targets = np.interp(np.arange(n) * tot / n, s, th)
            arm = _runs(w_of(_at(P, N, np.arange(n) * tot / n)[0]) > 0.3)
            shares = []
        par = np.interp(targets, th, s)
        for r, (i, j) in enumerate(arm):                       # round an arm: by the share of the way round it
            m = (j - i) % n + 1
            a, b = par[(i - 1) % n], par[(j + 1) % n]
            b = b + tot if b <= a else b
            ss = np.linspace(a, b, 200)
            turn = np.unwrap(arm_turn_of(_at(P, N, np.mod(ss, tot))[0]) * TAU) / TAU
            turn = np.maximum.accumulate(np.convolve(np.pad(turn, 15, mode="edge"), np.ones(31) / 31, "valid"))
            frac = ARM_TURN * (turn - turn[0]) / max(turn[-1] - turn[0], 1e-9) \
                + (1.0 - ARM_TURN) * np.linspace(0.0, 1.0, 200)
            if first:
                shares.append(np.interp(np.mod(np.arange(m) * tot / n + (i * tot / n) - a, tot) + a, ss, frac))
            par[(i + np.arange(m)) % n] = np.mod(np.interp(shares[r], frac, ss), tot)
        p, nn = _at(P, N, par)
        rows.append(p)
        nrows.append(nn)
    return np.array(rows), np.array(nrows), top


def taut_rows(B, w, axis_xy):
    """The rows of a (K, n, 3) grid whose torso part (arm blend `w` (K, n) near 0) is pulled taut in its own level
    plane, as an elastic rib round her chest and back is: each point moved out from the vertical axis `axis_xy` onto the
    convex hull of its row's torso points, as much as 1 - w."""
    out = np.array(B, float)
    ax = np.asarray(axis_xy, float)
    for k in range(len(out)):
        torso = w[k] < 0.05
        if torso.sum() < 3:
            continue
        r = np.linalg.norm(out[k, :, :2] - ax, axis=1)
        d = (out[k, :, :2] - ax) / np.maximum(r, 1e-12)[:, None]
        rh = ray_hull(convex_hull(out[k, torso, :2] - ax), d)
        out[k, :, :2] += d * (np.where(np.isfinite(rh) & (rh > r), rh - r, 0.0) * (1.0 - w[k]))[:, None]
    return out


def grid_smooth(B, along, across, lam=0.5):
    """A (K, n, 3) grid smoothed `along` times round its rows (periodic) and `across` times down its columns."""
    B = circ_smooth(np.moveaxis(np.asarray(B, float), 1, -1), along, lam)
    return row_smooth(np.moveaxis(B, -1, 1), across, lam)


def grid_normals(B):
    """Unit normals (K, n, 3) of a (K, n, 3) grid (rows down, columns round, periodic)."""
    return G.unit(np.cross(np.gradient(B, axis=0), np.roll(B, -1, 1) - np.roll(B, 1, 1)))


def grid_tris(K, n):
    """Triangles (2 (K-1) n, 3) of a (K, n) grid, periodic round its rows."""
    k, i = np.meshgrid(np.arange(K - 1), np.arange(n), indexing="ij")
    a, b = k * n + i, k * n + (i + 1) % n
    c, d = a + n, b + n
    return np.concatenate([np.stack([a, c, b], -1).reshape(-1, 3), np.stack([b, c, d], -1).reshape(-1, 3)])


BAND_ROLL, BAND_FACE = 8, 9         # the band's rows: its roll over the top, then its face
RIB_LIP, RIB_FACE = 2, 9            # a ribbed band's rows: its top lip, then its face


def face_rows(name, rings):
    """The rows (a slice) of a piece's rings that show from outside: the knit's all, a band's face."""
    if name == "band":
        return slice(BAND_ROLL, BAND_ROLL + BAND_FACE)
    if name.startswith(("waistband", "cuff")):
        return slice(RIB_LIP, RIB_LIP + RIB_FACE)
    return slice(0, len(rings))


def band(fit, sh, c, centre_y):
    """The folded rib band. Returns (rings (R, n, 3) from the roll's skin side over the top, down the face and back
    in under the lip, the texture's u (n+1,) and v (R,), points inside the fold (R, n, 3) for its facing, the band's
    inner side (a Surface), the top's height at her front's middle)."""
    S = fit.S
    b = c["band"]
    h = b["height"] * S
    levels = np.linspace(0.0, h, BAND_FACE)
    at = Surface(sh.verts, sh.tris, {"blend": sh.fields["blend"]})
    w_of = lambda P: at.near(P)[1]["blend"]

    def turn_of(P):
        P = np.asarray(P, float)
        return np.arctan2(P[:, 0], -(P[:, 1] - centre_y))
    arm_turn_of = lambda P: -SH.limb_angle(np.asarray(P, float), fit.L)    # grows over an arm along the lines
    loops = []
    for v in levels:      # each line pulled taut round her chest and back in its plane, out of the hollows, smoothed
        P, N = _loop(sh, sh.fields["band"], v)
        P = taut_rows(P[None], w_of(P)[None], (0.0, centre_y))[0]
        P = circ_smooth(fit.skin.bridge(P, b["bridge"] * S, 0.0).T, 8).T
        loops.append((fit.push_out(P, 0.0), N))
    _, _, top_len = band_grid(loops[:1], 64, turn_of, w_of, arm_turn_of)
    ribs = max(8, round(top_len / (b["rib"] * S)))
    n = 4 * ribs                                                         # a rib every 4 columns
    flat, Nk, _ = band_grid(loops, n, turn_of, w_of, arm_turn_of)
    w = w_of(flat.reshape(-1, 3)).reshape(flat.shape[:2])
    flat = grid_smooth(flat, int(b["smooth"]), 2)
    turn = (np.abs(w - 0.5) < 0.45).any(0)                             # where the band turns from her body onto an arm
    zone = np.clip(2.0 * np.convolve(np.concatenate([turn[-8:], turn, turn[:8]]).astype(float), np.ones(17) / 17,
                                     "same")[8:-8], 0.0, 1.0)[None, :, None]
    for it in range(int(b["turn_smooth"])):          # there each row runs on smoothly, the band spanning the fold
        flat = flat + 0.5 * zone * (0.5 * (np.roll(flat, 1, 1) + np.roll(flat, -1, 1)) - flat)
        if it % 25 == 24:
            flat = row_smooth(flat, 1) * zone + flat * (1.0 - zone)
            flat = fit.push_out(flat.reshape(-1, 3), 0.0005 * S).reshape(flat.shape)
    flat = fit.push_out(flat.reshape(-1, 3), 0.0005 * S).reshape(flat.shape)
    nrm = grid_normals(flat)                             # consistent over the whole grid: one sign, the skin's majority
    nrm = nrm * np.sign(np.einsum("kij,kij->", nrm, Nk) or 1.0)
    nrm = G.unit(grid_smooth(nrm, 10, 2))      # smoothed: offset along them, a tight crease (where her arm meets her
                                               # back) folded the face over itself
    along = G.unit(np.roll(flat, -1, 1) - np.roll(flat, 1, 1))           # round her, counter-clockwise from above
    down = G.unit(np.cross(along, nrm))                                  # down the band square to its rows, however its
                                                                         # columns lean
    off, edge, rd = b["offset"] * S, b["edge"] * S, b["rib_depth"] * S
    gap = 0.12 * off
    rib = -np.cos(TAU * np.arange(n) / 4.0)                              # wales where the rib texture lights them
    rows, vs = [], []
    rho = 0.5 * (off - gap)                                              # the roll: half round over the top
    _, q0, t0, b0 = fit.skin.closest(flat[0])
    seal = q0 + fit.skin.outward(t0, b0) * gap - (flat[0] + nrm[0] * gap)    # its skin side down onto her skin, where
    for phi in np.linspace(np.pi, 0.0, BAND_ROLL + 1)[:-1]:                  # the band spans a hollow (no opening at
        up = rho * np.sin(phi)                                               # its top for a lock of hair to slip into)
        rows.append(flat[0] + nrm[0] * (gap + rho + rho * np.cos(phi) + rd * rib * 0.5 * (1.0 - np.cos(phi)))[:, None]
                    - down[0] * up + G.smoothstep(np.clip(2.0 * phi / np.pi - 1.0, 0.0, 1.0)) * seal)
        vs.append(-(np.pi - phi) * rho)
    for k in range(len(levels)):                                         # the face, a little full in the middle
        full = 0.10 * off * np.sin(np.pi * levels[k] / h)
        rows.append(flat[k] + nrm[k] * (off + full + rd * rib)[:, None])
        vs.append(levels[k])
    for psi in np.linspace(0.0, np.pi / 2, 4)[1:]:                       # the lip, rounded
        rows.append(flat[-1] + nrm[-1] * (off - edge * (1.0 - np.cos(psi)) + rd * rib * np.cos(psi))[:, None]
                    + down[-1] * edge * np.sin(psi))
        vs.append(h + edge * psi)
    for nn in (0.55 * (off - edge), gap):                                # and back in under the band
        rows.append(flat[-1] + nrm[-1] * nn + down[-1] * edge)
        vs.append(vs[-1] + (off - edge - nn))
    rings = np.array(rows)
    inside = np.broadcast_to((flat[len(levels) // 2] + nrm[len(levels) // 2] * 0.5 * off)[None], rings.shape)
    inner = Surface(flat.reshape(-1, 3), grid_tris(len(levels), n), {})
    front = np.abs(flat[0, :, 0]) < 0.02 * S
    z_top = float(rings[:, front, 2].max()) if front.any() else float(rings[..., 2].max())
    return rings, np.linspace(0.0, ribs / 8.0, n + 1), np.array(vs) / WR.TILE, inside, inner, z_top


# ---------------------------------------------------------------- the knit under the band
def under_band(P, skin_field, fit, c):
    """Knit positions P (n, 3) under the band kept inside it: where the band covers the knit, moved in along the skin's
    normal to at most `inside` of the band's thickness beyond the band's inner side (`band_off` of `skin_field`: how
    far that stands off the skin there), right down to the band's edge, letting go over half of `drop` below the edge.
    Each point keeps its place over the skin, so neighbours stay neighbours where the band turns from her body onto an
    arm (taken to the band's nearest point, the knit tore across the armpit's corner). Returns the new positions."""
    S = fit.S
    bd = c["body"]
    h, drop = c["band"]["height"] * S, bd["drop"] * S
    P = np.array(P, float)
    d, _, tri, bary = fit.skin.closest(P)
    at = skin_field.near(P)[1]
    allow = at["band_off"] + bd["inside"] * c["band"]["offset"] * S
    t = 1.0 - G.smoothstep(np.clip((at["band"] - h) / (0.5 * drop), 0.0, 1.0))
    return P + (t * np.minimum(allow - d, 0.0))[:, None] * fit.skin.outward(tri, bary)


# ---------------------------------------------------------------- the knit body
def body(fit, c, centre_y, z_top, z_wb, wb_inner, rng, M):
    """(rings (K, M, 3), their outward directions (K, M, 3)) of the knit body round the vertical axis (0, centre_y),
    from z_top down into the waistband, whose inner side at its top is the ring `wb_inner` (M, 3); before the band."""
    S = fit.S
    bd = c["body"]
    th = G.ring_theta(M)
    dirs = G.hdir(th)
    d2 = dirs[:, :2]
    zs = np.arange(z_top, z_wb - 0.014 * S, -0.006 * S)
    front, back = np.clip(np.cos(th), 0, 1) ** 1.5, np.clip(-np.cos(th), 0, 1) ** 1.5
    ease = bd["ease_side"] + (bd["ease_front"] - bd["ease_side"]) * front + (bd["ease_back"] - bd["ease_side"]) * back
    taper = bd["taper_side"] + (bd["taper_front"] - bd["taper_side"]) * front \
        + (bd["taper_back"] - bd["taper_side"]) * back
    u, v = np.array([0.0, -1.0, 0.0]), np.array([1.0, 0.0, 0.0])
    H, last = [], None
    for z in zs:
        r = fit.torso.outline(np.array([0.0, centre_y, z]), u, v, M, "far", rmin=0.02 * S, rmax=0.30 * S)
        r = last if r is None else hull_radii(circ_smooth(r, 2), d2)
        r = np.full(M, 0.1 * S) if r is None else r
        H.append(r)
        last = r
    H = np.array(H)
    R = np.zeros_like(H)
    for k in range(len(zs)):                     # it hangs from what is above it, coming in no faster than the taper
        R[k] = H[k] + ease * S
        if k:
            R[k] = np.maximum(R[k], R[k - 1] - taper * (zs[k - 1] - zs[k]))
    R = circ_smooth(row_smooth(R, 3), 4)
    # under each arm the knit falls from lower down (a sweater's armhole sits below the armpit), standing off her side
    z_pit = float(fit.L["arm.L"][2]) - bd["pit_drop"] * S
    side_w = np.clip(np.abs(np.sin(th)), 0.0, 1.0) ** 4
    below_pit = G.smoothstep(np.clip((z_pit - zs) / (0.03 * S) + 1.0, 0.0, 1.0))
    R = R + bd["pit_ease"] * S * (1.0 - below_pit)[:, None] * side_w[None, :]
    wb_r = np.linalg.norm(wb_inner[:, :2] - np.array([0.0, centre_y]), axis=1)
    above = (zs - z_wb) / (bd["blouse_height"] * S)                       # 0 at the rib's top, 1 a blouse above
    R = R + bd["blouse"] * S * np.exp(-((above - 0.3) / 0.35) ** 2)[:, None]
    tuck = G.smoothstep(np.clip(1.0 - (zs - z_wb) / (0.010 * S), 0.0, 1.0))[:, None]
    R = (1.0 - tuck) * R + tuck * (wb_r[None, :] - 0.002 * S)               # in under the rib
    gather = np.clip(1.0 - above, 0.0, 1.0) ** 1.6 * (1.0 - tuck[:, 0])
    R = R + bd["pleat_depth"] * S * gather[:, None] * waves(th, bd["pleats"], rng)[None, :]
    z_bust = float(fit.L["bust_front"][2]) if fit.L.has("bust_front") else float(zs[len(zs) // 3])
    hang = np.clip((z_bust - zs) / max(z_bust - z_wb, 1e-6), 0.0, 1.0) ** 1.2 * (1.0 - tuck[:, 0])
    R = R + bd["drape"] * S * hang[:, None] * waves(th, 7, rng, spread=2)[None, :]
    rings = np.array([np.array([0.0, centre_y, z]) + dirs * R[k][:, None] for k, z in enumerate(zs)])
    return rings, np.broadcast_to(dirs, rings.shape).copy()


def edge_gathers(rings, dirs, skin_field, h, count, depth, fade, rng):
    """Rings (K, M, 3) gathered where they come out from under the band: `count` soft pleats round each ring, `depth`
    deep at the band's edge (its depth h on the skin), dying away over `fade` below it."""
    K, M = rings.shape[:2]
    d = skin_field.near(rings.reshape(-1, 3))[1]["band"].reshape(K, M) - h
    a = depth * G.smoothstep(np.clip(d / (0.25 * fade) + 1.0, 0.0, 1.0)) * np.exp(-np.maximum(d, 0.0) / fade)
    return rings + dirs * (a * waves(G.ring_theta(M), count, rng)[None, :])[..., None]


# ---------------------------------------------------------------- along an arm
def arm_frames(fit, side, ss):
    """(points, unit tangents, frames u (towards her front) and v, the arm's length shoulder -> wrist) along the arm's
    centre line at arc lengths ss from the shoulder joint."""
    L = fit.L
    a, e, w = (np.asarray(L[f"{k}.{side}"], float) for k in ("arm", "elbow", "wrist"))
    dense = G.catmull(np.array([a - (e - a) * 0.5, a, e, w, w + (w - e) * 0.6]), 400)
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(dense, axis=0), axis=1))])
    s = s - s[int(np.argmin(np.linalg.norm(dense - a, axis=1)))]
    pts = np.stack([np.interp(ss, s, dense[:, k]) for k in range(3)], 1)
    T = G.unit(np.stack([np.interp(ss, s, np.gradient(dense[:, k])) for k in range(3)], 1))
    front = np.array([0.0, -1.0, 0.0])
    U = G.unit(front[None] - T * (T @ front)[:, None])
    return pts, T, U, np.cross(T, U), float(np.linalg.norm(e - a) + np.linalg.norm(w - e))


def _arm_hull(arm_skin, pts, U, V, M, S, rmax=0.08):
    """(K, M) convex hull radii of the arm's own skin round the centre line (rays from it, angle from U towards V)."""
    th = G.ring_theta(M)
    d2 = np.stack([np.cos(th), np.sin(th)], 1)
    H = []
    for k in range(len(pts)):
        r = arm_skin.outline(pts[k], U[k], V[k], M, "far", rmin=0.003 * S, rmax=rmax * S)
        r = H[-1] if r is None and H else (np.full(M, 0.04 * S) if r is None else r)
        r = np.minimum(r, 1.25 * np.median(r))
        H.append(hull_radii(circ_smooth(r, 2), d2))
    return np.array(H)


def sleeve(fit, c, side, arm_skin, skin_field, rng, M):
    """(rings (K, M, 3), their outward directions) of one sleeve along the arm into the cuff, before the band: from the
    first ring that lies wholly under the band's top line (its top hidden under the band, its underside inside the
    body's knit), so no knit is above that line where her shoulders are bare."""
    S = fit.S
    sl, cf = c["sleeve"], c["cuff"]
    arm_len = arm_frames(fit, side, np.zeros(1))[4]
    s_end = arm_len - cf["length"] * S + 0.012 * S
    th = G.ring_theta(M)

    def rings_at(ss):
        pts, _, U, V, _ = arm_frames(fit, side, ss)
        dirs = np.cos(th)[None, :, None] * U[:, None, :] + np.sin(th)[None, :, None] * V[:, None, :]
        return pts, U, V, dirs
    probe = np.arange(0.0, 0.10 * S, 0.004 * S)                           # where the band's line crosses the arm
    pts, U, V, dirs = rings_at(probe)
    H0 = _arm_hull(arm_skin, pts, U, V, M, S)
    top = skin_field.near((pts[:, None, :] + dirs * H0[..., None]).reshape(-1, 3))[1]["band"].reshape(H0.shape).min(1)
    under = np.flatnonzero(top >= sl["start_depth"] * S)
    s0 = float(probe[under[0]]) if len(under) else float(probe[-1])
    ss = np.arange(s0, s_end, 0.008 * S)
    pts, U, V, dirs = rings_at(ss)
    H = _arm_hull(arm_skin, pts, U, V, M, S)
    t = ss / arm_len
    tab = np.array(sl["ease"], float)
    phi_down = np.arctan2(V[:, 2] * -1.0, U[:, 2] * -1.0)                  # where down is, round each ring
    hang = 0.5 * (1.0 + np.cos(th[None, :] - phi_down[:, None]))                # 1 straight under the arm
    sag = sl["sag"] * S * np.clip(t / 0.4, 0.0, 1.0)[:, None] * hang
    s_el = float(np.linalg.norm(np.asarray(fit.L[f"elbow.{side}"], float) - np.asarray(fit.L[f"arm.{side}"], float)))
    near_pit = 1.0 - G.smoothstep(np.clip(ss / (0.6 * s_el), 0.0, 1.0))     # most at the armpit, gone 60 % down the arm
    sag = sag + sl["underarm"] * S * near_pit[:, None] * hang ** 2
    R = row_smooth(H + np.interp(t, tab[:, 0], tab[:, 1])[:, None] * S + sag, 4)
    to_end = (s_end - ss) / (sl["gather"] * S)
    R = R + sl["puff"] * S * np.exp(-((to_end - 0.35) / 0.3) ** 2)[:, None]
    pleats = waves(th, sl["pleats"], rng)
    R = R + sl["pleat_depth"] * S * (np.clip(1.0 - to_end, 0.0, 1.0) ** 1.4)[:, None] * pleats[None, :]

    crook = np.clip(np.cos(th), 0.0, 1.0) ** 2                            # the front of the arm: where the elbow bends
    folds_at = np.array(sl["elbow_at"], float) * S
    bumps = np.exp(-0.5 * ((ss[:, None] - s_el - folds_at[None, :]) / (0.006 * S)) ** 2).sum(1)
    R = R + sl["elbow_fold"] * S * bumps[:, None] * crook[None, :]
    tuck = G.smoothstep(np.clip(1.0 - (s_end - ss) / (0.012 * S), 0.0, 1.0))[:, None]
    R = (1.0 - tuck) * R + tuck * (H + (cf["offset"] - 0.001) * S)          # in under the cuff
    return pts[:, None, :] + dirs * R[..., None], dirs


# ---------------------------------------------------------------- ribbed bands
def ribbed(face, out, thick, rib_depth):
    """Rings of a rib band from its outer face (K, M, 3) (top to bottom) and outward unit directions: a lip at the top
    `thick` deep back to what goes in under it, the face ribbed every 4 columns, the bottom edge rolled to the
    inside."""
    M = face.shape[1]
    rib = -np.cos(TAU * np.arange(M) / 4.0)[:, None]                     # wales where the rib texture lights them
    rows = [face[0] - out[0] * thick, face[0] - out[0] * 0.4 * thick][:RIB_LIP]
    rows += [face[k] + out[k] * rib_depth * rib for k in range(len(face))]
    along = G.unit(face[-1] - face[-2])
    rho = 0.5 * thick
    for phi in np.linspace(0.0, np.pi, 6)[1:]:
        rows.append(face[-1] + out[-1] * (rho * (np.cos(phi) - 1.0) + rib_depth * rib * 0.5 * (1.0 + np.cos(phi)))
                    + along * rho * np.sin(phi))
    rows.append(face[-1] - out[-1] * thick - along * thick)
    return np.array(rows)


def _rib_v(rings):
    d = np.linalg.norm(np.diff(rings, axis=0), axis=2).mean(1)
    return np.concatenate([[0.0], np.cumsum(d)]) / WR.TILE


def waistband(fit, c, centre_y):
    """(rings, the face's top ring, the top's height) of the waistband: as many columns as its ribs need (4 a rib)."""
    S = fit.S
    wb = c["waistband"]
    z_top = float(fit.L["leg.L"][2]) + wb["above_hip"] * S
    zs = np.linspace(z_top, z_top - wb["height"] * S, RIB_FACE)
    u, v = np.array([0.0, -1.0, 0.0]), np.array([1.0, 0.0, 0.0])

    def faces(M):
        dirs = G.hdir(G.ring_theta(M))
        H = np.array([hull_radii(circ_smooth(fit.core.outline(np.array([0.0, centre_y, z]), u, v, M, "far",
                                                              rmin=0.02 * S, rmax=0.35 * S), 3), dirs[:, :2])
                      for z in zs])
        R = row_smooth(H, 2) + (wb["offset"] + wb["thick"]) * S
        return np.array([np.array([0.0, centre_y, z]) + dirs * R[k][:, None] for k, z in enumerate(zs)]), dirs
    face, _ = faces(128)
    per = float(np.linalg.norm(np.roll(face[0], -1, 0) - face[0], axis=1).sum())
    face, dirs = faces(4 * max(8, round(per / (wb["rib"] * S))))
    return ribbed(face, np.broadcast_to(dirs, face.shape), wb["thick"] * S, wb["rib_depth"] * S), face[0], z_top


def cuff(fit, c, side, arm_skin, M):
    """Rings of one cuff."""
    S = fit.S
    cf = c["cuff"]
    arm_len = arm_frames(fit, side, np.zeros(1))[4]
    ss = np.linspace(arm_len - cf["length"] * S, arm_len + cf["past"] * S, RIB_FACE)
    pts, _, U, V, _ = arm_frames(fit, side, ss)
    th = G.ring_theta(M)
    H = row_smooth(_arm_hull(arm_skin, pts, U, V, M, S), 2)
    dirs = np.cos(th)[None, :, None] * U[:, None, :] + np.sin(th)[None, :, None] * V[:, None, :]
    face = pts[:, None, :] + dirs * (H + (cf["offset"] + cf["thick"]) * S)[..., None]
    return ribbed(face, dirs, cf["thick"] * S, cf["rib_depth"] * S)


# ---------------------------------------------------------------- assembly
GIRDLE = ("肩", "腕", "腕捩", "ひじ", "手捩", "手首")       # what a shoulder or an arm takes of her upper chest and back


def body_only(W, mix, default="上半身2"):
    """Weights {bone: (n,)} with the shoulder girdle's share (GIRDLE, either side) taken out as much as `mix` (n,) and
    given to the rest pro rata (to `default` where the rest has none): cloth round her chest and back that should stay
    with her body when an arm moves, while the skin under it follows the arm's share."""
    girdle = [b for b in W if b[:1] in ("左", "右") and b[1:] in GIRDLE]
    if not girdle or not np.any(mix > 0.0):
        return W
    g = sum(np.asarray(W[b], float) for b in girdle)
    rest = np.maximum(sum((np.asarray(w, float) for b, w in W.items() if b not in girdle), np.zeros_like(g)), 0.0)
    out = {b: np.asarray(w, float) * (1.0 - mix) if b in girdle else np.asarray(w, float) for b, w in W.items()}
    lost = g * mix
    scale = np.where(rest > 1e-9, 1.0 + lost / np.maximum(rest, 1e-9), 1.0)
    for b, wb in out.items():
        if b not in girdle:
            out[b] = wb * scale
    out[default] = out.get(default, np.zeros_like(g)) + np.where(rest > 1e-9, 0.0, lost)
    return out


def carry(W, P, rings, W_rings, t):
    """Weights {bone: (n,)} at points P blended, as much as t (n,), towards those of the closed piece `rings` (K, M, 3)
    at its nearest point (`W_rings` per ring point): cloth that lies under another moves with it."""
    T = grid_tris(*rings.shape[:2])
    _, tri, bary, _ = SK.closest_on_mesh(np.asarray(P, float), rings.reshape(-1, 3), T)
    corners = T[tri]
    zero = np.zeros(len(P))
    return {b: (1.0 - t) * np.asarray(W.get(b, zero), float)
            + t * ((np.asarray(W_rings[b], float)[corners] * bary).sum(1) if b in W_rings else zero)
            for b in set(W) | set(W_rings)}



def round_smooth(A, sigma):
    """(K, M, ...) values smoothed round closed rings (axis 1) with a Gaussian of `sigma` columns."""
    A = np.asarray(A, float)
    if sigma <= 0.0:
        return A
    k = np.arange(-int(np.ceil(3 * sigma)), int(np.ceil(3 * sigma)) + 1)
    if len(k) >= A.shape[1]:
        return np.broadcast_to(A.mean(1, keepdims=True), A.shape).copy()
    g = np.exp(-0.5 * (k / sigma) ** 2)
    return sum(gk * np.roll(A, int(kk), 1) for kk, gk in zip(k, g / g.sum()))


def _patch(rings, u, v, mat, tag, inside, fit, donors, default, mix=None, spread=0.0):
    """A lofted piece (closed rings) facing away from `inside` (per ring point), with the weights of the skin nearest to
    it, smoothed over the piece and round its rings over about `spread` metres (so the cloth turns smoothly where its
    skin hands over from one bone to another); with `mix` (per ring point) the shoulder girdle's share taken out that
    much, likewise smoothed."""
    p = G.orient_outward(G.loft(rings, True, u=u, v=v, mat=mat, tag=tag), np.asarray(inside, float).reshape(-1, 3))
    p.w = fit.weights_for(p.v, donors, default=default, faces=p.f, smooth=2)
    K, M = rings.shape[:2]
    col = float(np.linalg.norm(np.roll(rings, -1, 1) - rings, axis=2).mean())
    sigma = spread / max(col, 1e-9)
    if sigma > 0.0:
        p.w = {b: round_smooth(w.reshape(K, M), sigma).reshape(-1) for b, w in p.w.items()}
    if mix is not None:
        p.w = body_only(p.w, round_smooth(np.asarray(mix, float).reshape(K, M), sigma).reshape(-1))
    return p


def _circ_u(rings, per):
    """u (M+1,) round closed rings: whole tiles of `per` metres round their mean ring (seamless)."""
    per_ring = np.linalg.norm(np.roll(rings, -1, 1) - rings, axis=2).sum(1).mean()
    return np.linspace(0.0, max(1, round(per_ring / per)), rings.shape[1] + 1)


def _rib_u(rings):
    """u (M+1,) round a ribbed piece of M columns (4 a rib): the rib texture has 8 ribs a tile."""
    return np.linspace(0.0, rings.shape[1] / 32.0, rings.shape[1] + 1)


def _body_mix(P, skin_field, h, drop):
    """(n,) how much of the shoulder girdle's share cloth at P leaves out: all of it on her body away from her arms near
    the band, none near an arm (within the band's `reach` of its bone), none a drop below the band's edge."""
    near = skin_field.near(np.asarray(P, float).reshape(-1, 3))[1]
    return (1.0 - near["near"]) * G.smoothstep(np.clip((h + 2.0 * drop - near["band"]) / drop, 0.0, 1.0))


def _axis_points(rings, centre_y):
    """Points on the vertical axis (0, centre_y) at the height of each ring point."""
    return np.concatenate([np.zeros_like(rings[..., :1]), np.full_like(rings[..., :1], centre_y), rings[..., 2:3]], -1)


def sweater(fit, body_part, cfg, soup):
    """Build the sweater into `soup` (material 0 knit, 1 rib). Returns info: the band's top height at the middle of her
    front, the waistband's top, the closest each piece comes to the skin, and the pieces' vertex counts."""
    c = SP.merge(DEFAULTS, cfg)
    S = fit.S
    rng = np.random.default_rng(int(c["seed"]))
    sh = SH.Shell.from_part(body_part)
    share = sh.share(SH.ARM)
    depth, blend, near_arm = band_depth(sh.verts, share, fit, c)
    sh.fields.update(band=depth, blend=blend)
    h, drop = c["band"]["height"] * S, c["body"]["drop"] * S
    z_wb = float(fit.L["leg.L"][2]) + c["waistband"]["above_hip"] * S
    near = (np.abs(sh.verts[:, 2] - z_wb) < 0.03 * S) & (share < 0.3) & (sh.share(SH.LEG) < 0.3)
    centre_y = float(0.5 * (sh.verts[near, 1].min() + sh.verts[near, 1].max())) if near.any() else 0.0
    donors = [b for b in fit.skin.weights if not any(b[1:].startswith(t) for t in OF.HAND_STEMS)]
    arm_skin = fit.skin.where([n for n in fit.skin.weights if OF.is_arm(n)], 0.5, name="arm")
    info, pieces = {}, {}
    # ---- the band
    rings, u, v, inside, inner, info["band_top_z"] = band(fit, sh, c, centre_y)
    band_off = np.ones(len(sh.verts))                     # how far the band's inner side stands off the skin under it
    over = depth < h + drop
    band_off[over] = np.linalg.norm(inner.near(sh.verts[over])[0] - sh.verts[over], axis=1)
    skin_field = Surface(sh.verts, sh.tris, {"band": depth, "blend": blend, "near": near_arm, "band_off": band_off})
    band_patch = _patch(rings, u, v, 1, "band", inside, fit, donors, "上半身2", _body_mix(rings, skin_field, h, drop),
                        spread=c["band"]["spread"] * S)
    soup.add(band_patch)
    pieces["band"] = rings
    face = rings[face_rows("band", rings)]
    k0 = face_rows("band", rings).start * rings.shape[1]
    face_w = {b: w[k0:k0 + face.size // 3] for b, w in band_patch.w.items()}

    def under_the_band(p):                       # knit the band covers moves with the band, letting go below its edge
        t = 1.0 - G.smoothstep(np.clip((skin_field.near(p.v)[1]["band"] - (h - drop * 0.25)) / (drop * 0.5), 0.0, 1.0))
        p.w = carry(p.w, p.v, face, face_w, t)
        return p
    # ---- the waistband, then the knit body going in under it
    wb_rings, wb_top, info["z_waistband"] = waistband(fit, c, centre_y)
    soup.add(_patch(wb_rings, _rib_u(wb_rings), _rib_v(wb_rings), 1, "waistband",
                    _axis_points(wb_rings, centre_y), fit, donors, "下半身"))
    pieces["waistband"] = wb_rings
    Mb = 144
    wb_inner = G.resample_ring(wb_top, Mb)[0]
    out = G.unit(np.concatenate([wb_inner[:, :2] - np.array([0.0, centre_y]), np.zeros((Mb, 1))], 1))
    wb_inner = wb_inner - out * c["waistband"]["thick"] * S
    z_line = float(fit.L["arm.L"][2]) - c["band"]["chest"] * S
    kr, kd = body(fit, c, centre_y, z_line - 0.007 * S, z_wb, wb_inner, rng, Mb)
    bd = c["body"]
    kr = edge_gathers(kr, kd, skin_field, h, bd["gathers"], bd["gather_depth"] * S, bd["gather_fade"] * S, rng)
    kr = under_band(kr.reshape(-1, 3), skin_field, fit, c).reshape(kr.shape)
    kr = fit.conform(kr, bd["gap"] * S)
    soup.add(under_the_band(_patch(kr, _circ_u(kr, WR.TILE), _rib_v(kr), 0, "body", _axis_points(kr, centre_y), fit,
                                   donors, "上半身", _body_mix(kr, skin_field, h, drop), spread=c["body"]["spread"] * S)))
    pieces["body"] = kr
    # ---- sleeves and cuffs
    for side, pre in (("L", "左"), ("R", "右")):
        sr, sd = sleeve(fit, c, side, arm_skin, skin_field, rng, 64)
        sl = c["sleeve"]
        sr = edge_gathers(sr, sd, skin_field, h, sl["gathers"], sl["gather_depth"] * S, sl["gather_fade"] * S, rng)
        sr = under_band(sr.reshape(-1, 3), skin_field, fit, c).reshape(sr.shape)
        sr = fit.conform(sr, c["body"]["gap"] * S)
        soup.add(under_the_band(_patch(sr, _circ_u(sr, WR.TILE), _rib_v(sr), 0, f"sleeve_{side}", sr - sd * 0.02 * S, fit,
                                       donors, f"{pre}腕")))
        pieces[f"sleeve_{side}"] = sr
        per = float(np.linalg.norm(np.diff(sr[-1], axis=0), axis=1).sum())
        cr = cuff(fit, c, side, arm_skin, 4 * max(8, round(per / (c["cuff"]["rib"] * S))))
        soup.add(_patch(cr, _rib_u(cr), _rib_v(cr), 1, f"cuff_{side}",
                        np.broadcast_to(cr.mean(1, keepdims=True), cr.shape), fit, donors, f"{pre}手首"))
        pieces[f"cuff_{side}"] = cr
    gaps = {}
    for k, r in pieces.items():           # what shows: the knit below the band, the bands' faces; vertices and middles
        r = r[face_rows(k, r)]
        mid = 0.25 * (r[:-1] + r[1:] + np.roll(r[:-1], -1, 1) + np.roll(r[1:], -1, 1))
        P = np.concatenate([r.reshape(-1, 3), mid.reshape(-1, 3)])
        d = fit.skin.closest(P)[0]
        shown = skin_field.near(P)[1]["band"] > h if k in ("body", "sleeve_L", "sleeve_R") else np.ones(len(P), bool)
        gaps[k] = float(d[shown].min())
    info.update(min_gap=min(gaps.values()), gaps=gaps, verts={k: int(np.prod(r.shape[:2])) for k, r in pieces.items()})
    return info


@builder("sweater", needs=("body",))
def build(ctx):
    cfg = SP.merge(DEFAULTS, ctx.cfg)
    fit = OU.make_fit(ctx)
    col = WR.colours(ctx, "sweater", COLORS)
    soup = G.Soup("sweater", [MAT_KNIT, MAT_RIB])
    info = sweater(fit, ctx.need("body"), cfg, soup)
    mats = [WR.material(ctx, MAT_KNIT, "knit", col["knit"], WR.knit(col["knit"]), shadow=(0.82, 0.74, 0.88)),
            WR.material(ctx, MAT_RIB, "rib", col["rib"], WR.rib(col["rib"]), shadow=(0.78, 0.72, 0.86))]
    mesh = soup.mesh()
    ctx.log(f"band top {info['band_top_z']:.3f} m on the chest, waistband top {info['z_waistband']:.3f} m, closest to "
            f"the skin {info['min_gap'] * 1000:.1f} mm")
    return Part("sweater", meshes=[mesh], materials=[m for m in mats if m.name in mesh.mats], info=info)
