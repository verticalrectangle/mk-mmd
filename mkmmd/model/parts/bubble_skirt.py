"""The round skirt (part `bubble_skirt`, numpy only): printed cotton gathered into her waist, as a paper lantern
gathered into its hem too or as a flared skirt open at its hem, and loose bloomers under it whose frilled ends show
below its hem.

It is rings round a vertical axis through the middle of her hips (the smock's, `smock.axis_of`), from a band round
her waist down to the hem, then turned under:

  band      a band `height` high round her waist, its middle `at` over the hip joints, `ease` off her skin (under the
            smock: never seen); its top edge turned in
  body      `form` "lantern" or "flare". From the band the cloth fans out under the smock (when the smock is built
            before it, it stays `under` inside the smock, its lace and the line out to the lace's tips, and leaves them
            `under` inside the round of the tips, their flutes smoothed out, never outside a tip). A lantern then swells
            to its widest `widest` over the hip joints (negative: under them), `rx` from the axis at her sides and `ry`
            in front and behind (an ellipse round the axis), curving in above it as the share of the way ** `top`, and
            below it rounds in to the hem as a superellipse of power `low` (2: a quarter of an ellipse). A flare runs
            out to its hem, where it is widest, rounding out first and then running straight (the side line goes the
            share `1 - (1 - s) ** bell` of the way out at the share s of the way down: 1 a straight cone). Nothing of
            either nearer her skin than `gap`, and where it hangs free `clear` off the bloomers (the hem's roll inside
            included)
  hem       the hem `at` over the hip joints (negative: under them), `off` off her legs (the convex section of both
            legs there plus `off`), the cloth turned under `turn` deep and `thick` thick there (a rolled edge); a
            flare's hem is also at least the ellipse `rx` from the axis at her sides and `ry` in front and behind
  segments  `count` round her, one's crest at her front middle. A lantern's are panels, each a round crest between two
            sharp creases: `waist` from crest to crease under the band down to where it leaves the smock, `hem` at the
            hem, fading to `full` over the fullest part. A flare's are soft rolling folds, out and in by `waist` where
            it leaves the smock growing to `hem` at the hem. The creases (a flare's troughs) are painted as a soft
            darker line `width` wide, `crease` of the way to the shadow swatch where the folds are deepest
  tail      the cloth inside a tube round what leaves her back there (a tail) is cut away, `margin` round it: `path`
            and `radius` as the smock's [smock.tail] (model space), or, when they are empty, the smock's own
  print     its picture: big soft-edged motifs on the ground, round it (`motifs`: each {kind spiral | leaf | flame, at
            (the turn round from her front toward her left, 0..1), z (over the hip joints), size (across, metres x S),
            turn (degrees), mirror (bool), color (a key of [colors.skirt], default "print"; its outline and streaks in
            `<color>_shadow`, or `print_shadow` when that is not set)}): a spiral is a fat curl `stroke` wide with a
            tail and three small dots beside the tail, a leaf a broad blade with its veins, a flame a broad tongue
            rising from below its middle that curls over into an open hook, `streaks` (3; 0 none) up its lower half
            (set it low and it rises from the hem); `band` = {height (metres x S above the hem, 0: none), wave (its
            top rises and falls that share of it), waves (round her), color}: a band of a print colour along the hem,
            its top in waves, the turned-under hem in it too, under the motifs (waves breaking along the hem with the
            flames); edges `soft` metres soft; each row of the picture as long as the cloth is round her there, so a
            motif keeps its shape wherever it sits; the seam of the picture is at her back
  chains    `columns` chains of `rows` bones round her (スカート, family skirt) hung from the lower body (下半身),
            their heads where the skirt leaves the smock (`from` under its hem; without a smock `from` under the band),
            the first bone `root` long, capsules `radius` thick or as thin as the room to her colliders asks
            (outfit_rig.Rig.ring); below their heads the skirt goes over to them but `follow` of her hips' weights. A
            balloon bobs: its joints are `spring` stiff and swing at most `swing` radians
  bloomers  her skin from `top` over the hip joints down to the cuffs `cuff` over the hip joints (negative: under),
            `off` off it (loose), `snug` at the waist; at each cuff a band gathered `gather` off her leg and `rows` of
            frills `frill` long, pleated (`pleat` apart, `depth` deep), standing `flare` out at their hems, the hems in
            soft scallops rising and falling `scallop` of the frill's length (0.05 a gathered ruffle, 0.18 points).
            They take no cast shadow (white under the skirt's shade, shaded by their own toon step)

Sizes are metres x S (S = head_tip.z / 1.7 with leg_extra left out: 0.793 on the bases), the tail's tube model space.
Spec: `[bubble_skirt]` (DEFAULTS below, tables merge key by key), `[colors.skirt]` `base`, `shadow`, `light`, `print`,
`print_shadow`, `accent`, `accent_shadow` (the last two for motifs that name them) and `[colors.bloomers]` `base`,
`shadow`. Materials バルーンスカート and ブルマー; landmarks
`skirt_widest` (the widest line at her front middle) and `skirt_hem` (the hem there)."""
import numpy as np

from .. import spec as SP
from ..build import builder
from ..part import Part
from . import outfit as OU
from . import outfit_fit as OF
from . import outfit_geo as G
from . import outfit_rig as ORIG
from . import outfit_shell as SH
from . import outfit_wear as WR
from .gown import ink, shadow_of, tidy, weld_shell
from .outfit_dress import hang_frill
from .hair_tex import toon_ramp
from .outfit_skirt import bone_names
from .smock import axis_of, ring_at, tube, tube_field
from .sweater import circ_smooth, convex_hull, hull_radii, ray_hull, row_smooth, with_chains

TAU = 2.0 * np.pi
DEFAULTS = {
    "band": {"at": 0.129, "height": 0.024, "ease": 0.004},
    "body": {"form": "lantern", "widest": -0.179, "rx": 0.2396, "ry": 0.2244, "top": 1.4, "low": 2.0, "bell": 1.5,
             "under": 0.012, "gap": 0.004, "clear": 0.004},
    "hem": {"at": -0.343, "off": 0.0315, "turn": 0.012, "thick": 0.006, "rx": 0.30, "ry": 0.28},
    "segments": {"count": 16, "waist": 0.017, "hem": 0.017, "full": 0.004, "crease": 0.4, "width": 0.02},
    "tail": {"path": [], "radius": [], "margin": 0.006},
    "print": {"motifs": [], "band": {"height": 0.0, "wave": 0.0, "waves": 12, "color": "print"}, "stroke": 0.020,
              "soft": 0.0015, "line": 0.0015},
    "chains": {"enabled": True, "columns": 12, "rows": 3, "from": 0.004, "root": 0.02, "radius": 0.045,
               "follow": 0.0, "spring": 24.0, "swing": 0.55},
    "bloomers": {"enabled": True, "top": 0.10, "cuff": -0.366, "off": 0.0158, "snug": 0.004, "gather": 0.006,
                 "frill": 0.022, "pleat": 0.016, "depth": 0.003, "flare": 0.006, "rows": 2, "scallop": 0.18},
}
COLORS = {"skirt": {"base": "#a33f48", "shadow": "#6f2530", "light": "#bd5a5f", "print": "#c9a878",
                    "print_shadow": "#9a7a54", "accent": None, "accent_shadow": None},
          "bloomers": {"base": "#f3e9ec", "shadow": "#c9b9c6"}}
MAT_SKIRT = "バルーンスカート"
MAT_BLOOMERS = "ブルマー"
SKIRT, BLOOMERS = 0, 1
PX = 2048                      # the print's width (once round her at her widest)
HOLD = 0.03                    # metres above the chains' heads over which her hips let go of the skirt
UP = np.array([0.0, 0.0, 1.0])


# ---------------------------------------------------------------- the balloon's shape
def legs_hull(fit, axis_y, z, M):
    """(M,) radii round the axis (0, axis_y) of the convex hull of her legs' section at height z (G.ring_theta(M))."""
    S = fit.S
    dirs = G.hdir(G.ring_theta(M))
    r = fit.core.outline(np.array([0.0, axis_y, z]), np.array([0.0, -1.0, 0.0]), np.array([1.0, 0.0, 0.0]), M, "far",
                         rmin=0.0, rmax=0.35 * S)
    return hull_radii(circ_smooth(r, 2), dirs[:, :2]) if r is not None else np.full(M, 0.12 * S)


def panels(th, c):
    """(m,) the lantern's panels round a ring at the angles `th` (module docstring: segments): 0 on the creases
    between them, 1 on each panel's crest, `count` panels, one with its crest at her front middle."""
    n = max(int(c["segments"]["count"]), 3)
    f = ((np.asarray(th, float) - np.pi / n) * n / TAU) % 1.0
    return np.sin(np.pi * f) ** 0.6


def creases(c):
    """(n,) the turns round her (0 at her front, toward her left) of the creases between the panels."""
    n = max(int(c["segments"]["count"]), 3)
    return (np.pi / n + np.arange(n) * TAU / n) % TAU


def depth_at(zs, c, S, z_bb, z_t, z_w, z_h):
    """(K,) the panels' depth from crest to crease at the heights zs (module docstring: segments): `waist` from under
    the band down to where the skirt leaves the smock, `hem` at the hem, fading to `full` over the fullest part."""
    sc = c["segments"]
    up = G.smoothstep(np.clip((zs - z_w) / max(z_t - z_w, 1e-9), 0.0, 1.0))
    low = G.smoothstep(np.clip((z_w - zs) / max(z_w - z_h, 1e-9), 0.0, 1.0))
    d = sc["full"] + (sc["waist"] - sc["full"]) * up + (sc["hem"] - sc["full"]) * low
    return d * S * np.clip((z_bb - zs) / (0.015 * S), 0.0, 1.0)


def roll_depth(zs, c, S, z_bb, z_t, z_h):
    """(K,) a flared skirt's folds' depth, out and in (module docstring: segments), at the heights zs: `waist` where it
    leaves the smock growing to `hem` at the hem as the share of the way ** 1.3, none at the band."""
    sc = c["segments"]
    s = np.clip((z_t - zs) / max(z_t - z_h, 1e-9), 0.0, 1.0)
    d = sc["waist"] + (sc["hem"] - sc["waist"]) * s ** 1.3
    return d * S * np.clip((z_bb - zs) / (0.015 * S), 0.0, 1.0)


def inside_smock(info, axis_y, zs, th):
    """(K, M) the radius round the axis (0, axis_y) the skirt keeps inside at the heights zs and turns th: the smock's
    cloth down to its hem, then its lace, out to the lace's tips (`info`: the smock's); inf below the lace."""
    hang, tips = info["hang"], info["lace_tips"]
    d = G.hdir(th)
    z_sh = float(info["z_hem"])
    r_tip = np.interp(th, tips["theta"], tips["r"], period=TAU)
    z_tip = np.interp(th, tips["theta"], tips["z"], period=TAU)

    def cloth(z):
        Q = np.column_stack([np.zeros(len(th)), np.full(len(th), axis_y), np.full(len(th), z)]) + 0.1 * d
        return ring_at(hang["zs"], hang["R"], Q, axis_y)
    r_hem = cloth(z_sh)
    out = np.full((len(zs), len(th)), np.inf)
    for k, z in enumerate(zs):
        if z >= z_sh:
            out[k] = cloth(z)
        else:
            t = np.clip((z_sh - z) / np.maximum(z_sh - z_tip, 1e-6), 0.0, 1.0)
            out[k] = np.where(z >= z_tip, r_hem + (r_tip - r_hem) * t, np.inf)
    return out


def hull_floor(points, axis_y, zs, th, band):
    """(K, M) radii round the axis (0, axis_y) at the turns th of the convex hull of the points (n, 3) within `band`
    of each height zs (0 where there are none): what is under the skirt there."""
    d2 = G.hdir(th)[:, :2]
    out = np.zeros((len(zs), len(th)))
    for k, z in enumerate(zs):
        Q = points[np.abs(points[:, 2] - z) < band][:, :2] - np.array([0.0, axis_y])
        if len(Q) >= 3:
            out[k] = ray_hull(convex_hull(Q), d2)
    return out


def shape(fit, c, axis_y, smock, under=None, M=192):
    """(rings (K, M, 3) from the band's top down to the hem, their heights (K,), where the chains may start (the
    lowest height the smock still covers), the widest line's height, the panels' depth (K,)): module docstring: band,
    body, hem, segments. `under` (n, 3): what the skirt keeps `body.clear` off (the bloomers)."""
    S, L = fit.S, fit.L
    bc, bd, hm = c["band"], c["body"], c["hem"]
    z_hip = 0.5 * (float(L["leg.L"][2]) + float(L["leg.R"][2]))
    z_band = z_hip + bc["at"] * S
    z_top, z_bb = z_band + 0.5 * bc["height"] * S, z_band - 0.5 * bc["height"] * S
    z_w, z_h = z_hip + bd["widest"] * S, z_hip + hm["at"] * S
    th = G.ring_theta(M)
    dirs = G.hdir(th)
    step = 0.004 * S
    zs = np.concatenate([np.arange(z_top, z_h + 0.5 * step, -step), [z_h]])
    u, v = np.array([0.0, -1.0, 0.0]), np.array([1.0, 0.0, 0.0])
    H, last = [], np.full(M, 0.12 * S)
    for z in zs:
        r = fit.core.outline(np.array([0.0, axis_y, z]), u, v, M, "far", rmin=0.02 * S, rmax=0.35 * S)
        last = last if r is None else hull_radii(circ_smooth(r, 2), dirs[:, :2])
        H.append(last)
    H = row_smooth(np.array(H), 2)
    r_band = H[np.argmin(np.abs(zs - z_band))] + bc["ease"] * S
    r_w = 1.0 / np.sqrt((np.sin(th) / (bd["rx"] * S)) ** 2 + (np.cos(th) / (bd["ry"] * S)) ** 2)
    r_h = legs_hull(fit, axis_y, z_h, M) + hm["off"] * S
    keep_in = None
    if smock is not None and smock.info.get("lace_tips"):      # it leaves the smock just inside its lace's tips
        tips = smock.info["lace_tips"]
        z_t = float(np.min(tips["z"])) - 0.002 * S
        r_tip = np.interp(th, tips["theta"], tips["r"], period=TAU)
        f = np.fft.rfft(r_tip)
        f[5:] = 0.0                                             # the round of the tips, the smock's flutes left out
        r_t = np.minimum(np.fft.irfft(f, M) - bd["under"] * S, r_tip - 0.002 * S)
        keep_in = inside_smock(smock.info, axis_y, zs, th) - bd["under"] * S
    else:
        z_t = z_bb - 0.03 * S
        r_t = r_band + 0.02 * S
    form = bd["form"]
    if form not in ("lantern", "flare"):
        raise ValueError(f"[bubble_skirt.body] form = {form!r}: \"lantern\" or \"flare\"")
    if form == "flare":                          # widest at its open hem, at least the ellipse there
        r_h = np.maximum(r_h, 1.0 / np.sqrt((np.sin(th) / (hm["rx"] * S)) ** 2 + (np.cos(th) / (hm["ry"] * S)) ** 2))
        z_w = z_h
    R = np.zeros((len(zs), M))
    for k, z in enumerate(zs):
        if z >= z_bb:
            R[k] = r_band
        elif z >= z_t:                           # fanning out under the smock
            s = G.smoothstep((z_bb - z) / max(z_bb - z_t, 1e-9))
            R[k] = r_band + (r_t - r_band) * s
        elif form == "flare":                    # out to the hem, rounding out first, then running straight
            s = (z_t - z) / max(z_t - z_h, 1e-9)
            R[k] = r_t + (r_h - r_t) * (1.0 - (1.0 - s) ** bd["bell"])
        elif z >= z_w:                           # the lantern's top, curving out to its widest
            s = (z - z_w) / max(z_t - z_w, 1e-9)
            R[k] = r_w - (r_w - r_t) * s ** bd["top"]
        else:                                    # and round in again to the hem
            s = (z_w - z) / max(z_w - z_h, 1e-9)
            R[k] = r_h + (r_w - r_h) * np.clip(1.0 - s ** bd["low"], 0.0, 1.0) ** (1.0 / bd["low"])
    R = row_smooth(R, 3)
    if form == "flare":                          # soft rolling folds, a crest at her front middle
        depth = roll_depth(zs, c, S, z_bb, z_t, z_h)
        R = R + depth[:, None] * np.cos(max(int(c["segments"]["count"]), 3) * th)[None, :]
    else:
        depth = depth_at(zs, c, S, z_bb, z_t, z_w, z_h)
        R = R - depth[:, None] * (1.0 - panels(th, c))[None, :]
    if keep_in is not None:                      # under the smock and its lace: inside them
        R = np.minimum(R, keep_in)
    R = np.maximum(R, H + bd["gap"] * S)
    if under is not None:                        # where it hangs free off the bloomers, the hem's roll inside it included
        low = zs < (z_t if form == "flare" else z_w)
        floor = hull_floor(under, axis_y, zs[low], th, 0.01 * S)
        R[low] = np.maximum(R[low], np.where(floor > 0.0, floor + (bd["clear"] + 1.1 * hm["thick"]) * S, 0.0))
    rings = np.array([np.array([0.0, axis_y, z]) + dirs * R[k][:, None] for k, z in enumerate(zs)])
    rings = fit.conform(rings, bd["gap"] * S)
    return rings, zs, z_t, z_w, depth


def turned(rings, axis_y, c, S):
    """The rings with the band's top edge turned in and the hem turned under (module docstring: hem): (rings, the
    index of the hem's ring)."""
    hm = c["hem"]
    last = rings[-1]
    out = G.unit(np.column_stack([last[:, 0], last[:, 1] - axis_y, np.zeros(len(last))]))
    t, up = hm["thick"] * S, hm["turn"] * S                 # a soft roll: round under the hem and up inside it
    roll = [last - out * a * t - UP * b * t for a, b in ((0.15, 0.35), (0.5, 0.5), (0.85, 0.35), (1.0, 0.0))]
    under1 = roll[-1]
    under2 = last - out * 1.1 * t + UP * up
    first = rings[0]
    o0 = G.unit(np.column_stack([first[:, 0], first[:, 1] - axis_y, np.zeros(len(first))]))
    top_in = first - o0 * 0.6 * hm["thick"] * S - UP * 0.4 * hm["thick"] * S
    return np.concatenate([[top_in], rings, roll[:-1], [under1, under2]]), len(rings)


# ---------------------------------------------------------------- the print
def spiral_curve(size):
    """(points (k, 2) metres, stroke half widths (k,)) of a fat curl `size` across: from a round inner end one turn
    and a quarter outward, then a tail sweeping away; and the three dots beside the tail ((3, 2), radii (3,))."""
    phi = np.linspace(0.0, 2.35 * np.pi, 120)
    a = 0.06 * size
    b = 0.36 * size / (2.35 * np.pi)
    r = a + b * phi
    P = np.stack([r * np.cos(phi), r * np.sin(phi)], 1)
    end, ang = P[-1], phi[-1]
    tang = G.unit(np.array([np.cos(ang) * b - r[-1] * np.sin(ang), np.sin(ang) * b + r[-1] * np.cos(ang)]))
    t = np.linspace(0.0, 1.0, 40)[1:]
    bend = np.array([-tang[1], tang[0]])
    tail = end[None] + tang[None] * (0.42 * size * t)[:, None] - bend[None] * (0.10 * size * t ** 2)[:, None]
    P = np.concatenate([P, tail])
    w = np.concatenate([0.075 * size * (0.75 + 0.25 * np.clip(phi / (1.2 * np.pi), 0, 1)),
                        0.075 * size * (1.0 - 0.7 * t ** 1.3)])
    tip = tail[-1]
    side = G.unit(-bend)
    dots = np.array([tip + side * 0.12 * size - tang * k * 0.11 * size for k in (0.6, 1.6, 2.6)])
    return P, w, dots, np.full(3, 0.035 * size)


def leaf_shape(size):
    """(outline polygon (k, 2), vein polylines [(k, 2), ...]) of a broad leaf `size` long along +x, metres."""
    t = np.linspace(0.0, np.pi, 60)
    half = 0.30 * size * np.sin(t) ** 0.85 * (1.0 + 0.15 * np.cos(t))
    x = -0.5 * size * np.cos(t)
    top = np.stack([x, half], 1)
    bot = np.stack([x[::-1], -half[::-1]], 1)
    poly = np.concatenate([top, bot])
    veins = [np.array([[-0.62 * size, 0.0], [0.47 * size, 0.0]])]
    for f in (-0.25, 0.0, 0.25):
        x0 = f * size
        veins.append(np.array([[x0, 0.0], [x0 + 0.16 * size, 0.17 * size]]))
        veins.append(np.array([[x0, 0.0], [x0 + 0.16 * size, -0.17 * size]]))
    return poly, veins


def flame_curve(size, streaks=3):
    """(points (k, 2) metres, half widths (k,), streaks [(j, 2), ...]) of a flame `size` across, in its own frame (+y
    up): a broad tongue rising from 0.55 `size` under its middle, leaning a little to its right, that curls over to its
    left at the top into an open hook (a wave's crest); three streaks up its lower half (the flame-and-wave motif of a
    printed cotton)."""
    t = np.linspace(0.0, 1.0, 60)
    rise = np.stack([0.06 * size * (3.0 * t ** 2 - 2.0 * t ** 3), -0.55 * size + 0.55 * size * t], 1)
    r0, turns = 0.20 * size, 1.5 * np.pi
    phi = np.linspace(0.0, turns, 90)[1:]
    r = r0 * (1.0 - 0.5 * phi / turns)
    curl = rise[-1] - np.array([r0, 0.0]) + np.stack([r * np.cos(phi), r * np.sin(phi)], 1)
    P = np.concatenate([rise, curl])
    w = np.concatenate([0.10 * size + 0.16 * size * (1.0 - t) ** 1.3, 0.10 * size - 0.06 * size * phi / turns])
    tang = np.gradient(rise, axis=0)
    nrm = G.unit(np.stack([-tang[:, 1], tang[:, 0]], 1))
    low = t <= 0.6
    across = np.linspace(-0.5, 0.5, streaks) if streaks > 1 else np.zeros(max(streaks, 0))
    streaks = [rise[low] + nrm[low] * (w[:60][low] * f)[:, None] for f in across]
    return P, w, streaks


def seg_dist(Pq, A, B):
    """(n, m) distances from points Pq (n, 2) to the segments A (m, 2) -> B (m, 2), and the share along each."""
    AB = B - A
    L2 = np.maximum((AB ** 2).sum(1), 1e-18)
    t = np.clip(((Pq[:, None, :] - A[None]) * AB[None]).sum(2) / L2[None], 0.0, 1.0)
    D = Pq[:, None, :] - (A[None] + t[..., None] * AB[None])
    return np.sqrt((D ** 2).sum(2)), t


def stroke_sdf(loc, shape, A, B, wa, wb, reach, tile=32):
    """(n,) the signed distance (metres) of the points `loc` (n, 2), a grid of `shape` (rows, cols) row by row, to a
    stroke along the segments A -> B (m, 2) whose half width goes from `wa` to `wb` (m,) along each (0: the segments'
    own distance). Worked tile by tile against only the segments within `reach` of the tile: `reach` where none is."""
    h, w = shape
    grid = loc.reshape(h, w, 2)
    wa = np.broadcast_to(np.asarray(wa, float), (len(A),))
    wb = np.broadcast_to(np.asarray(wb, float), (len(A),))
    AB = B - A
    L2 = np.maximum((AB ** 2).sum(1), 1e-18)
    wide = np.maximum(wa, wb)
    out = np.full((h, w), float(reach))
    for i in range(0, h, tile):
        for j in range(0, w, tile):
            Q = grid[i:i + tile, j:j + tile].reshape(-1, 2)
            cen = Q.mean(0)
            rad = float(np.sqrt(((Q - cen) ** 2).sum(1).max()))
            t = np.clip(((cen - A) * AB).sum(1) / L2, 0.0, 1.0)
            near = np.sqrt(((cen - A - t[:, None] * AB) ** 2).sum(1)) - wide < rad + reach
            if not near.any():
                continue
            d, s = seg_dist(Q, A[near], B[near])
            sd = (d - (wa[near][None] + s * (wb[near] - wa[near])[None])).min(1)
            out[i:i + tile, j:j + tile] = np.minimum(sd, reach).reshape(out[i:i + tile, j:j + tile].shape)
    return out.reshape(-1)


def inside_poly(Pq, poly):
    """(n,) whether points are inside the closed polygon (even-odd)."""
    x, y = Pq[:, 0], Pq[:, 1]
    inside = np.zeros(len(Pq), bool)
    xa, ya = poly[:, 0], poly[:, 1]
    xb, yb = np.roll(xa, -1), np.roll(ya, -1)
    for i in range(len(poly)):
        cond = (ya[i] > y) != (yb[i] > y)
        xc = xa[i] + (y - ya[i]) * (xb[i] - xa[i]) / np.where(yb[i] != ya[i], yb[i] - ya[i], 1e-18)
        inside ^= cond & (x < xc)
    return inside


def print_colours(col, key, what):
    """(colour, its line colour) of the print's `key` of [colors.skirt]: the line `<key>_shadow`, else `print_shadow`."""
    if col.get(key) is None:
        raise ValueError(f"[bubble_skirt.print] {what} color {key!r}: [colors.skirt] {key} is not set")
    line = col.get(f"{key}_shadow")
    return col[key], col["print_shadow"] if line is None else line


def picture(col, c, circ, length, down_of, z_hip, S, folds=None, width_of=None, hem=None):
    """The skirt's picture (h, PX, 4) uint8: once round her at `circ` metres (u 0 at her back, 0.5 at her front),
    down it `length` metres along the cloth from the band's top (v 1; `down_of(z)`: metres down the cloth to the
    height z); the band along the hem (`hem`: metres down the cloth to it; module docstring: print), the motifs
    soft-edged on the ground, then the creases between the panels (`folds`: their u (n,) and how dark each row (h,)
    is: a soft line toward the shadow swatch, `segments.width` wide). `width_of(d)` (an array of metres down the cloth
    -> the cloth's length round her there; None: `circ` all the way) makes each row of the picture stand for the
    cloth round her at its own height, so motifs and lines keep their shape on a skirt that is narrower or wider than
    `circ` there. Returns (image, metres per pixel)."""
    pc = c["print"]
    mpp = circ / PX
    Hpx = int(np.ceil(length / mpp))
    ground = np.broadcast_to(col["base"], (Hpx, PX, 3)).copy()
    soft = max(pc["soft"] * S, mpp)
    k_row = np.ones(Hpx) if width_of is None else \
        np.maximum(np.asarray(width_of((np.arange(Hpx) + 0.5) * mpp), float) / circ, 1e-3)   # metres across per mpp
    band = pc.get("band") or {}
    if band.get("height", 0.0) > 0.0 and hem is not None:      # the band: waves along the hem, the turn-under too
        c_b, l_b = print_colours(col, band.get("color", "print"), "band")
        u = (np.arange(PX) + 0.5) / PX
        top = band["height"] * S * (1.0 + band.get("wave", 0.0) * np.sin(TAU * band.get("waves", 12) * u))
        above = (hem - (np.arange(Hpx) + 0.5) * mpp)[:, None]   # metres above the hem, each row
        sd = above - top[None, :]
        cover = np.clip(0.5 - sd / (2 * soft), 0.0, 1.0)[..., None]
        lw = pc["line"] * S
        o = np.clip(0.5 - (np.abs(sd + lw) - lw) / (2 * soft), 0.0, 1.0)[..., None]
        ground = ground * (1.0 - cover) + (c_b[None, None, :] * (1.0 - o) + l_b[None, None, :] * o) * cover
    for m in pc["motifs"]:
        kind = m["kind"]
        if kind not in ("spiral", "leaf", "flame"):
            raise ValueError(f"[bubble_skirt.print] motif kind {kind!r}: spiral, leaf or flame")
        c_m, c_line = print_colours(col, m.get("color", "print"), "motif")
        size = float(m.get("size", 0.15)) * S
        cu = (0.5 + float(m["at"])) % 1.0                  # the turn round from her front -> u
        cz = z_hip + float(m["z"]) * S
        cx_px, cy_px = cu * PX, down_of(cz) / mpp
        ang = np.radians(float(m.get("turn", 0.0)))
        half_y = int(np.ceil(0.85 * size / mpp)) + 4
        ys = np.arange(max(int(cy_px) - half_y, 0), min(int(cy_px) + half_y + 1, Hpx))
        if not len(ys):
            continue
        half_x = int(np.ceil(0.85 * size / (mpp * float(k_row[ys].min())))) + 4
        xs = np.arange(int(cx_px) - half_x, int(cx_px) + half_x + 1)
        X, Y = np.meshgrid(xs, ys)
        loc = np.stack([(X - cx_px) * mpp * k_row[Y], -(Y - cy_px) * mpp], -1).reshape(-1, 2)
        Rm = np.array([[np.cos(ang), np.sin(ang)], [-np.sin(ang), np.cos(ang)]])
        loc = loc @ Rm.T
        if m.get("mirror", False):
            loc[:, 0] = -loc[:, 0]
        reach = 4.0 * soft + 2.0 * pc["line"] * S
        if kind == "spiral":
            P, w, dots, dr = spiral_curve(size)
            sdf = stroke_sdf(loc, X.shape, P[:-1], P[1:], w[:-1], w[1:], reach)
            dd = np.sqrt(((loc[:, None, :] - dots[None]) ** 2).sum(2)) - dr[None]
            sdf = np.minimum(sdf, dd.min(1))
            cover = np.clip(0.5 - sdf / (2 * soft), 0.0, 1.0)
            col_m = np.broadcast_to(c_m, (len(loc), 3))
        elif kind == "flame":
            P, w, streaks = flame_curve(size, int(m.get("streaks", 3)))
            sdf = stroke_sdf(loc, X.shape, P[:-1], P[1:], w[:-1], w[1:], reach)
            cover = np.clip(0.5 - sdf / (2 * soft), 0.0, 1.0)
            col_m = np.broadcast_to(c_m, (len(loc), 3))
            if streaks:
                sd = np.min([seg_dist(loc, s[:-1], s[1:])[0].min(1) for s in streaks], axis=0) - 0.016 * size
                streak = np.clip(0.5 - sd / (2 * soft), 0.0, 1.0) * np.clip(-sdf / (3 * soft), 0.0, 1.0)
                col_m = c_m[None, :] * (1.0 - streak[:, None]) + c_line[None, :] * streak[:, None]
        else:
            poly, veins = leaf_shape(size)
            inside = inside_poly(loc, poly)
            d = stroke_sdf(loc, X.shape, poly, np.roll(poly, -1, 0), 0.0, 0.0, reach)
            sdf = np.where(inside, -d, d)
            stem = seg_dist(loc, np.array([[-0.5 * size, 0.0]]), np.array([[-0.66 * size, -0.03 * size]]))[0][:, 0] \
                - 0.018 * size
            sdf = np.minimum(sdf, stem)
            cover = np.clip(0.5 - sdf / (2 * soft), 0.0, 1.0)
            vd = np.min([seg_dist(loc, v[:-1], v[1:])[0].min(1) for v in veins], axis=0) - 0.012 * size
            vein = np.clip(0.5 - vd / (2 * soft), 0.0, 1.0) * np.clip(-sdf / (3 * soft), 0.0, 1.0)
            col_m = c_m[None, :] * (1.0 - vein[:, None]) + c_line[None, :] * vein[:, None]
        lw = pc["line"] * S                                  # a thin outline just inside its edge
        o = np.clip(0.5 - (np.abs(sdf + lw) - lw) / (2 * soft), 0.0, 1.0)[:, None]
        col_m = col_m * (1.0 - o) + c_line[None, :] * o
        k = cover.reshape(len(ys), len(xs), 1)
        cm = col_m.reshape(len(ys), len(xs), 3)
        xi = xs % PX
        patch = ground[ys[0]:ys[-1] + 1][:, xi]
        ground[ys[0]:ys[-1] + 1, xi] = patch * (1.0 - k) + cm * k
    if folds is not None:
        cu, dark = folds
        x = np.arange(PX) + 0.5
        gap = np.abs((x[:, None] - np.asarray(cu, float)[None, :] * PX + 0.5 * PX) % PX - 0.5 * PX).min(1)
        line = np.exp(-2.0 * (2.0 * gap[None, :] * mpp * k_row[:, None] / (c["segments"]["width"] * S)) ** 2)
        a = (np.asarray(dark, float)[:, None] * line)[..., None]
        ground = ground * (1.0 - a) + col["shadow"][None, None, :] * a
    return WR._rgba(ground), mpp


# ---------------------------------------------------------------- the bloomers
def bloomers(fit, c, body_part, soup, tb=None):
    """The bloomers (module docstring: bloomers) into `soup` (material BLOOMERS), cut round the tail's tube `tb` (as
    the skirt, `tail.margin` out from wherever they stand off her skin). Returns info (its "points": every point of
    them, for the skirt to keep off)."""
    S, L = fit.S, fit.L
    bc = c["bloomers"]
    z_hip = 0.5 * (float(L["leg.L"][2]) + float(L["leg.R"][2]))
    z_top, z_cuff = z_hip + bc["top"] * S, z_hip + bc["cuff"] * S
    sh = weld_shell(SH.Shell.from_part(body_part))
    for f in ("arm", "top", "cuff"):
        z = sh.verts[:, 2]
        sh = sh.clip({"arm": 0.5 - sh.share(OF.is_arm), "top": z_top - z, "cuff": z - z_cuff}[f])
    if tb is not None:                          # the skin under them cut wide enough for what stands off it
        sh = sh.clip(tube_field(sh.verts, tb) - c["tail"]["margin"] - bc["off"] * S)
    sh = sh.subdivide()
    z = sh.verts[:, 2]
    loose = G.smoothstep(np.clip((z_top - z) / (0.04 * S), 0.0, 1.0)) \
        * (1.0 - G.smoothstep(np.clip(1.0 - (z - z_cuff) / (0.025 * S), 0.0, 1.0)))
    dist = bc["snug"] * S + (bc["off"] - bc["snug"]) * S * loose
    near_cuff = G.smoothstep(np.clip(1.0 - (z - z_cuff) / (0.025 * S), 0.0, 1.0))
    dist = dist * (1.0 - near_cuff) + bc["gather"] * S * near_cuff
    P = sh.offset(dist, relax=8, fit=fit, gap=0.6 * bc["snug"] * S)
    u = SH.limb_angle(sh.verts, L, SH.LEG_CHAIN, ref=(0.0, -1.0, 0.0))
    loops = sh.loops()
    tops = [lp for lp in loops if float(np.mean(sh.verts[lp, 2])) > z_top - 0.01 * S]
    cuffs = [lp for lp in loops if float(np.mean(sh.verts[lp, 2])) < z_cuff + 0.01 * S]
    patch = sh.patch(P, u, -sh.verts[:, 2], u_rep=round(0.3 / WR.TILE), v_scale=1.0 / WR.TILE, mat=BLOOMERS,
                     tag="bloomers", rims=tops, rim_depth=0.6 * bc["snug"] * S)
    soup.add(patch)
    frills = []
    for lp in cuffs:
        ring = P[lp]
        side = "L" if ring[:, 0].mean() > 0 else "R"
        knee = np.asarray(L[f"knee.{side}"], float)
        centre = np.array([knee[0], knee[1], float(ring[:, 2].mean())])
        base = ring + G.unit(np.column_stack([ring[:, 0] - centre[0], ring[:, 1] - centre[1],
                                              np.zeros(len(ring))])) * 0.0
        length, flare = bc["frill"] * S, bc["flare"] * S
        t = np.linspace(0.0, 1.0, 6)
        donors = [b for b in fit.skin.weights if not OF.is_arm(b) and not OF.is_head(b)]
        out_r = G.unit(np.column_stack([ring[:, 0] - centre[0], ring[:, 1] - centre[1], np.zeros(len(ring))]))
        for row in range(int(bc["rows"])):             # rows of soft scallops, each over the one under it
            lift = row * 0.4 * length
            base = ring + UP * lift + out_r * row * 0.0025 * S
            prof = np.stack([t * flare * (1.0 + 0.3 * row), -0.004 * S + t * length * (1.0 - 0.15 * row)], 1)
            fp, frings, fbase = ring_frill_scalloped(base, centre + UP * lift, prof, bc["pleat"] * S,
                                                     bc["depth"] * S, f"bloomer_frill_{side}{row}",
                                                     (41 if side == "L" else 43) + 7 * row, bc["scallop"])
            wb = fit.weights_for(fbase, donors, default=f"{'左' if side == 'L' else '右'}ひざ")
            fp.w = {b: np.tile(np.asarray(w, float), len(frings)) for b, w in wb.items()}
            soup.add(fp)
            frills.append(frings)
    pts = np.concatenate([P] + [r.reshape(-1, 3) for f in frills for r in f])
    return {"z_top": z_top, "z_cuff": z_cuff, "gap": sh.gap(P, fit), "frills": frills, "points": pts,
            "frill_low": float(min(f[-1][:, 2].min() for f in frills)) if frills else None}


def ring_frill_scalloped(ring, centre, prof, pleat, depth, tag, seed, scallop=0.18):
    """A frill round the closed ring `ring` (round `centre`) hanging down: the profile `prof` (rows, 2) (out, down),
    soft pleats every `pleat` `depth` deep, its hem in soft scallops (two to a pleat) rising and falling `scallop` of
    its length (small: a soft gathered ruffle; large: pointed). Returns (patch (material BLOOMERS), rings, its base)."""
    around = float(np.linalg.norm(np.roll(ring, -1, 0) - ring, axis=1).sum())
    pleats = max(int(round(around / max(pleat, 1e-6))), 6)
    m = int(np.ceil(8 * pleats / 4.0) * 4)
    base = G.resample_ring(ring, m)[0]
    out = base - centre
    out[:, 2] = 0.0
    out = G.unit(out)
    along = np.broadcast_to(np.array([0.0, 0.0, -1.0]), base.shape)
    patch, rings = hang_frill(base, out, along, prof, pleats, depth, BLOOMERS, tag,
                              np.tile(base - out * 0.01, (len(prof), 1)), tile_u=0.05, seed=seed, wobble=0.2,
                              ridge=0.0, hem_wobble=scallop * float(np.abs(prof[:, 1]).max()))
    return patch, rings, base


# ---------------------------------------------------------------- the part
@builder("bubble_skirt", needs=("body",))
def build(ctx):
    c = SP.merge(DEFAULTS, ctx.cfg)
    fit = OU.make_fit(ctx)
    col = {"skirt": WR.colours(ctx, "skirt", COLORS["skirt"]), "bloomers": WR.colours(ctx, "bloomers",
                                                                                       COLORS["bloomers"])}
    S, L = fit.S, fit.L
    body_part = ctx.need("body")
    smock = ctx.parts.get("smock")
    axis_y = axis_of(fit)
    z_hip = 0.5 * (float(L["leg.L"][2]) + float(L["leg.R"][2]))
    soup = G.Soup("bubble_skirt", [MAT_SKIRT, MAT_BLOOMERS])
    tb = tube(c["tail"]["path"], c["tail"]["radius"])               # the tail's tube: the skirt's notch and the
    if tb is None and smock is not None and smock.info.get("tail"):    # bloomers' cut
        tb = (np.asarray(smock.info["tail"]["path"], float), np.asarray(smock.info["tail"]["radius"], float))
    info = {}
    under = None
    if c["bloomers"]["enabled"]:
        bl = bloomers(fit, c, body_part, soup, tb)
        under = bl["points"]
        info["bloomers"] = {k: v for k, v in bl.items() if k not in ("frills", "points")}
    rings, zs, z_t, z_w, depth = shape(fit, c, axis_y, smock, under)
    grid, k_hem = turned(rings, axis_y, c, S)
    K, M = grid.shape[:2]
    # texture coordinates: u round her (0 at her back), v down the cloth (metres from the band's top)
    th = G.ring_theta(M)
    seg = np.linalg.norm(np.diff(grid, axis=0), axis=2).mean(1)
    down = np.concatenate([[0.0], np.cumsum(seg)])
    j_seam = M // 2                                      # her back: the picture's seam
    order = np.roll(np.arange(M), -j_seam)
    grid_o = grid[:, order]
    uu = np.linspace(0.0, 1.0, M + 1)
    circ = TAU * float(np.mean(np.hypot(rings[np.argmin(np.abs(zs - z_w)), :, 0],
                                        rings[np.argmin(np.abs(zs - z_w)), :, 1] - axis_y)))
    length = float(down[-1])
    sc = c["segments"]
    deep = (depth - depth.min()) / max(float(np.ptp(depth)), 1e-9)          # the creases darker where it gathers
    dark = sc["crease"] * (0.55 + 0.45 * deep)
    folds = ((creases(c) - np.pi) / TAU % 1.0,
             np.interp(np.arange(int(np.ceil(length / (circ / PX)))) * circ / PX, down[1:k_hem + 1], dark))
    round_at = TAU * np.hypot(grid[:, :, 0], grid[:, :, 1] - axis_y).mean(1)   # the cloth's length round her, a ring
    pic, mpp = picture(col["skirt"], c, circ, length, lambda z: float(np.interp(-z, -grid[1:k_hem + 1, 0, 2],
                                                                                down[1:k_hem + 1])), z_hip, S, folds,
                       width_of=lambda d: np.interp(d, down[1:k_hem + 1], round_at[1:k_hem + 1]),
                       hem=float(down[k_hem]))
    vv = 1.0 - down / (pic.shape[0] * mpp)
    skirt = G.orient_outward(G.loft(grid_o, True, u=uu, v=vv, mat=SKIRT, tag="skirt"),
                             np.column_stack([np.zeros(K * M), np.full(K * M, axis_y), grid_o.reshape(-1, 3)[:, 2]]))
    donors = [b for b in fit.skin.weights if not (OF.is_arm(b) or OF.is_leg(b) or OF.is_head(b))]
    skirt.w = fit.weights_for(skirt.v, donors, default="下半身")
    rig = ORIG.Rig()
    info.update(z_top=float(zs[0]), z_hem=float(zs[-1]), z_widest=z_w, axis_y=axis_y)
    ch = c["chains"]
    if ch["enabled"]:
        C, R = max(int(ch["columns"]), 3), max(int(ch["rows"]), 1)
        names = bone_names(C, R)
        z_free = z_t - ch["from"] * S
        hold = z_free + HOLD * S
        end = grid[k_hem + 1]
        nodes = ORIG.hung_heights(z_free, float(end[:, 2].min()), R, ch["root"] * S)
        pts = np.zeros((C, R + 1, 3))
        for ci in range(C):
            j = int(round(ci * M / C)) % M
            col_ = grid[1:k_hem + 1, j]
            for r, z in enumerate(nodes):
                kk = int(np.clip(np.searchsorted(-col_[:, 2], -z), 1, len(col_) - 1))
                t = (col_[kk - 1, 2] - z) / max(col_[kk - 1, 2] - col_[kk, 2], 1e-9)
                pts[ci, r] = col_[kk - 1] + t * (col_[kk] - col_[kk - 1])
            pts[ci, R] = grid[k_hem, j] + 0.0
        body_rb = ctx.find_body("下半身")
        statics = [rb for rb in body_part.bodies if rb.mode == "static"]
        phys = {"spring": (float(ch["spring"]),) * 2 + (0.5 * float(ch["spring"]),),
                "swing": (float(ch["swing"]), 0.6 * float(ch["swing"]))}
        rad = rig.ring(names, pts, "下半身", "skirt", ch["radius"] * S, statics,
                       body_rb.name if body_rb is not None else None, "Skirt", physics=phys)
        share = ORIG.ring_weights(G.theta_of(skirt.v, axis_y), skirt.v[:, 2], names, nodes, hold, {}, C)
        skirt.w = with_chains(skirt.w, {b: (1.0 - float(ch["follow"])) * w for b, w in share.items()})
        info.update(chains=C, rows=R, radii=[float(rad.min()), float(rad.max())], z_free=float(z_free))
    if tb is not None:                               # cut round the tail
        T = SH.triangulate(skirt.f)
        bones = sorted(skirt.w)
        Wm = np.stack([skirt.w[b] for b in bones], 1)
        uv_corner = skirt.uv
        cut = SH.Shell(skirt.v, T, SH.vertex_normals(skirt.v, T), bones, Wm,
                       {"u": np.zeros(len(skirt.v)), "v": np.zeros(len(skirt.v))})
        uv_v = np.zeros((len(skirt.v), 2))
        k = 0
        for f in skirt.f:
            for vi in f:
                uv_v[vi] = uv_corner[k]
                k += 1
        cut.fields.update(u=uv_v[:, 0], v=uv_v[:, 1], tail=tube_field(skirt.v, tb) - c["tail"]["margin"])
        cut = cut.clip(cut.fields["tail"])
        skirt = cut.patch(cut.verts, cut.fields["u"], cut.fields["v"], mat=SKIRT, tag="skirt")
        info["notch"] = True
    soup.add(skirt)
    mesh = tidy(soup.mesh())
    mats = materials(ctx, col, pic)
    front = grid[k_hem, 0]
    info["landmarks"] = {"skirt_hem": front, "skirt_widest": rings[np.argmin(np.abs(zs - z_w)), 0]}
    ctx.log(f"round skirt from {zs[0]:.3f} m to its hem at {zs[-1]:.3f} m, widest at {z_w:.3f} m, "
            f"{info.get('chains', 0)} chains")
    return Part("bubble_skirt", meshes=[mesh], materials=[m for m in mats if m.name in mesh.mats],
                bones=rig.bones, bodies=rig.bodies, joints=rig.joints, info=info)


def materials(ctx, col, pic):
    """The skirt's and the bloomers' materials: matte cotton, a soft toon step, outlined in a dark shade of the
    shadow. The bloomers take no cast shadow: under the skirt's shade they would go grey; drawn, they stay white,
    shaded by their own toon step."""
    out = []
    for name, en, cc, tex in ((MAT_SKIRT, "skirt", col["skirt"], pic),
                              (MAT_BLOOMERS, "bloomers", col["bloomers"], WR.twill(col["bloomers"]["base"], lines=18,
                                                                                  depth=0.02))):
        sh = shadow_of({"c": cc["base"], "c_shadow": cc["shadow"]}, "c")
        m = WR.material(ctx, name, en, cc["base"], tex, shadow=sh, edge_size=0.6)
        m.toon = ctx.save_png(f"{en}_toon", toon_ramp(shadow=sh, edge=0.72, soft=0.18))
        m.edge_color = ink(cc["shadow"])
        if name == MAT_BLOOMERS:
            m.self_shadow = False
        out.append(m)
    return out
