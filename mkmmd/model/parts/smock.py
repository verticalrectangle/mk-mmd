"""The smock (part `smock`, numpy only): a sleeveless loose top of washed cotton to her hips, a row of lace points
along its hem, a soft flap on each shoulder edged with smaller lace points and a strap with a stud over each flap.

It is one piece of cloth cut from her skin and hung from it: the skin under it (welded, subdivided twice) is moved out
along its normals as the yoke lies on her shoulders and upper chest, then out from a vertical axis through her hips to
where the cloth hangs: below `hang.top` it falls from her bust and her shoulder blades, never following her in at the
waist. It is cut exactly along the level lines of fields on it (outfit_shell.Shell.clip), so its edges are clean
curves however coarse her skin is, and every cut edge is rolled back toward her (thickness where the cloth ends).

  neck      a round neck at the base of her neck: over the plane through `front` below the neck joint at her front
            middle and `back` below it at her back middle (level from side to side) the cloth on her neck is cut away;
            at her sides it keeps to `side` from her neck's axis (her shoulders' tops stay); the edge rolled `roll`
  armhole   each armhole a plane through the shoulder: `inside` the shoulder joint at the top of her shoulder, down to
            `bottom` below the shoulder joint at the side of the cloth (where it hangs there); rolled `roll`
  yoke      over her shoulders and upper chest the cloth lies on her: `ease` off her skin, relaxed `relax` times
  hang      from `top` below the shoulder joints (`blend` metres of hand-over) the cloth hangs round the vertical
            axis: each height the convex section of her body without her arms plus `ease_front`, `ease_side`,
            `ease_back`, coming in no faster than `taper` (it falls from her bust and shoulder blades and comes in a
            little under them), at least `waist` off her skin from under the bust to her hip joints; `folds` soft
            flutes round her (a crest either side of her front middle, mirrored, each a little different), growing
            from none at her bust to `fold_depth` from crest to hollow where the hem's flare starts
  hem       `above` the hip joints; over its last `flare` the cloth goes out (the share of the way ** `power`: an S
            with the fall above it) to `off` off her hips at the hem; there the flutes are `flute` deep and the hem
            line waves with them, `wave` from the crests' foot to the hollows'; its edge rolled `thick`
  arch      at the back of the hem a low arch round what leaves her there (a tail): `width` across where it meets the
            hem, rising `rise` over it in the middle, centred on the landmark `at` (x), its feet rounded `round`
            (width 0: none)
  tail      the cloth inside a tube is cut away: `path` (model space points, metres) and `radius` ([across, in its
            plane] per point, metres: what passes there, fur included) with `margin` (metres) round it, so the cut
            edge hugs what leaves through the cloth (an empty path: no cut). Published in `info["tail"]` for the
            garments built after it
  lace      a row of small points `width` wide and `height` long hanging point down from the hem, sewn inside it,
            `notch` under the hem between two points; each point has a small eyelet (`eyelet`, its radius over the
            point's width). It leaves the arch out
  flap      a soft piece on each shoulder like a small wing: from the neckline it lies on the cloth over the top of
            her shoulder out to `leave` past the shoulder joint, then goes on out `droop` degrees below level to `out`
            past the shoulder joint, `depth` from front to back, its outer edge a shallow point `point` deep, `lift`
            off what is under it, `thick`; the share of the arm's weights it takes grows from none at `follow_from`
            of the way out to `follow` at its outer edge (it tips with the arm). Its free edges are edged with lace
            points (`lace_width`, `lace_height`)
  strap     over each flap a strap `width` wide from the cloth's front `end` below the shoulder joint, over the
            shoulder `neck` from her midline where it crosses the neckline, to the same height at her back; `lift`
            over what is under it, `thick`. A diamond stud `stud` across on each, `stud_at` below the shoulder joint
            in front, `stud_thick` proud
  seed      the draw of the flutes' small differences

Sizes are metres x S (S = head_tip.z / 1.7 with leg_extra left out: 0.793 on the bases), but the tail's tube, which
is model space. Spec: `[smock]` (DEFAULTS below, tables merge key by key) and four colour tables: `[colors.smock]`
`base`, `shadow`, `light` (the cloth), `[colors.flaps]` `base`, `shadow`, `[colors.straps]` `base`, `shadow`, `stud`,
`[colors.trim]` `base`, `shadow` (the lace). Each cloth is matte: its toon shade is its shadow swatch over its colour,
its outline a dark shade of the shadow. Materials スモック, スモック裏 (the cloth's inside, no outline), 肩フラップ,
肩ストラップ, スモックレース. Landmarks `smock_arch` (the arch's top, on the cloth) and `smock_hem` (the hem at her
front middle); `info` holds the measured numbers."""
import numpy as np

from .. import spec as SP
from ..build import builder
from ..part import Part
from . import outfit as OU
from . import outfit_fit as OF
from . import outfit_geo as G
from . import outfit_shell as SH
from . import outfit_wear as WR
from .gown import ink, shadow_of, tidy, weld_shell
from .hair_tex import toon_ramp
from .sweater import circ_smooth, hull_radii, row_smooth, with_chains

TAU = 2.0 * np.pi
DEFAULTS = {
    "neck": {"front": 0.024, "back": 0.008, "side": 0.049, "roll": 0.0032},
    "armhole": {"inside": 0.0126, "bottom": 0.095, "roll": 0.0032},
    "yoke": {"ease": 0.0045, "relax": 10},
    "hang": {"top": 0.025, "blend": 0.05, "ease_front": 0.012, "ease_side": 0.012, "ease_back": 0.014,
             "taper": 0.07, "waist": 0.026, "folds": 8, "fold_depth": 0.0088},
    "hem": {"above": 0.111, "flare": 0.15, "power": 1.8, "off": 0.063, "flute": 0.0139, "wave": 0.008,
            "thick": 0.0032},
    "arch": {"width": 0.0, "rise": 0.076, "round": 0.010, "at": "tail_root"},
    "tail": {"path": [], "radius": [], "margin": 0.003},
    "lace": {"width": 0.0277, "height": 0.0328, "notch": 0.0025, "eyelet": 0.13},
    "flap": {"enabled": True, "out": 0.101, "depth": 0.145, "point": 0.03, "leave": 0.0164, "droop": 15.0,
             "lift": 0.0025, "thick": 0.0025, "follow_from": 0.4, "follow": 0.8, "lace_width": 0.0202,
             "lace_height": 0.024},
    "strap": {"enabled": True, "width": 0.0328, "end": 0.076, "neck": 0.057, "lift": 0.0018, "thick": 0.0018,
              "stud": 0.0202, "stud_at": 0.045, "stud_thick": 0.0035},
    "seed": 7,
}
COLORS = {"smock": {"base": "#c2a868", "shadow": "#917a48", "light": "#dcc78c"},
          "flaps": {"base": "#de9ea2", "shadow": "#b47a82"},
          "straps": {"base": "#b9a4d6", "shadow": "#8b78ab", "stud": "#e9def7"},
          "trim": {"base": "#fbf7f2", "shadow": "#d9cfc6"}}
MAT_CLOTH = "スモック"
MAT_FLAP = "肩フラップ"
MAT_STRAP = "肩ストラップ"
MAT_LACE = "スモックレース"
CLOTH, FLAP, STRAP, LACE = 0, 1, 2, 3                 # material indices of the smock's mesh
UP = np.array([0.0, 0.0, 1.0])


# ---------------------------------------------------------------- where things are on her
def neck_frame(fit, c, M=96):
    """The neckline (module docstring: neck): (ring (M, 3), a point of its plane (3,) on her neck's vertical axis, the
    plane's unit normal (3,) pointing up, the cylinder's radii (M,) round that axis at G.ring_theta(M), 0 at her front
    toward her left). The cloth over the plane through `front` below the neck joint at her front middle and `back`
    below it behind (level from side to side) is cut away where it is nearer the axis than the cylinder: `side` from
    the axis at her sides (her shoulders' tops, over the plane there, stay), and at her front and back wider than the
    cloth on her neck, so there the plane cuts it."""
    S, L = fit.S, fit.L
    nc = c["neck"]
    nk = np.asarray(L["neck"], float)
    hu, hv = np.array([0.0, -1.0, 0.0]), np.array([1.0, 0.0, 0.0])

    def section(z, need=True):
        r = fit.skin.outline(np.array([0.0, float(nk[1]), z]), hu, hv, M, "near", rmin=0.005 * S, rmax=0.12 * S)
        if r is None and need:
            raise ValueError(f"smock: no neck at z {z:.3f} for the neckline ([smock.neck])")
        return r
    zf, zb = float(nk[2]) - nc["front"] * S, float(nk[2]) - nc["back"] * S
    over = (c["yoke"]["ease"] + 0.004) * S
    rf = max(float(r[0]) for r in (section(z, False) for z in np.linspace(zf, zf + 0.02 * S, 5)) if r is not None) \
        + over
    rb = max(float(r[M // 2]) for r in (section(z, False) for z in np.linspace(zb, zb + 0.02 * S, 5))
             if r is not None) + over
    th = G.ring_theta(M)
    cf, cb = np.clip(np.cos(th), 0.0, 1.0) ** 2, np.clip(-np.cos(th), 0.0, 1.0) ** 2
    rs = nc["side"] * S
    r = circ_smooth(rs + (rf - rs) * cf + (rb - rs) * cb, 4)
    pf = np.array([0.0, float(nk[1]) - float(section(zf)[0]), zf])
    pb = np.array([0.0, float(nk[1]) + float(section(zb)[M // 2]), zb])
    n = G.unit(np.cross(np.array([1.0, 0.0, 0.0]), pf - pb))
    n = n if n[2] > 0 else -n
    axis = np.array([0.0, float(nk[1]), 0.0])
    axis[2] = pf[2] - ((axis - pf)[:2] @ n[:2]) / n[2]
    ring = axis[None] + G.hdir(th) * r[:, None]
    ring[:, 2] = pf[2] - ((ring - pf)[:, :2] @ n[:2]) / n[2]
    return ring, axis, n, r


def neck_field(P, axis, n, r):
    """(n,) >= 0 where the cloth at P is kept: under the neckline's plane (normal `n` through `axis`), or as far from
    the vertical line through `axis` as the cylinder `r` (round it, `neck_frame`)."""
    P = np.asarray(P, float)
    d = P[:, :2] - axis[:2]
    th = np.arctan2(d[:, 0], -d[:, 1]) % TAU
    rr = np.interp(th, G.ring_theta(len(r)), r, period=TAU)
    return np.maximum(np.hypot(d[:, 0], d[:, 1]) - rr, -((P - axis) @ n))


def top_of(skin, x, y, tol=0.004):
    """The highest point's z of the skin over (x, y) (its slice by the plane at x, within `tol` of y), or None."""
    seg = skin.slice(np.array([x, 0.0, 0.0]), np.array([1.0, 0.0, 0.0])).reshape(-1, 3)
    s = seg[np.abs(seg[:, 1] - y) < tol]
    return float(s[:, 2].max()) if len(s) else None


def armhole_planes(fit, c, x_side):
    """{side: (a point (3,), unit normal (3,) out toward the arm)} of the armholes (module docstring: armhole):
    through `inside` the shoulder joint at the top of her shoulder and `bottom` below it at the cloth's side
    (`x_side` {side: |x| of the cloth there})."""
    S, L = fit.S, fit.L
    ac = c["armhole"]
    out = {}
    for side, sg in (("L", 1.0), ("R", -1.0)):
        j = np.asarray(L[f"arm.{side}"], float)
        xa = abs(float(j[0])) - ac["inside"] * S
        za = top_of(fit.skin, sg * xa, float(j[1]))
        za = float(j[2]) + 0.02 * S if za is None else za
        zb = float(j[2]) - ac["bottom"] * S
        A = np.array([sg * xa, float(j[1]), za])
        B = np.array([sg * x_side[side], float(j[1]), zb])
        n = G.unit(np.cross(B - A, np.array([0.0, 1.0, 0.0])))
        n = n if n[0] * sg > 0 else -n
        out[side] = (A, n)
    return out


def axis_of(fit):
    """The y of the vertical axis the smock hangs round: the middle of her hips' depth at the hip joints."""
    S, L = fit.S, fit.L
    z_hip = 0.5 * (float(L["leg.L"][2]) + float(L["leg.R"][2]))
    core = fit.core.verts[np.unique(fit.core.tris)]
    hips = core[np.abs(core[:, 2] - z_hip) < 0.03 * S]
    return float(0.5 * (hips[:, 1].min() + hips[:, 1].max())) if len(hips) else 0.0


# ---------------------------------------------------------------- the hanging cloth
def flutes(th, c):
    """(m,) the cloth's soft flutes round her at the angles `th` (module docstring: hang): -1 in each hollow to 1 on
    each crest, `hang.folds` of them, a crest either side of her front middle, mirrored left and right, each a little
    deeper or shallower than the next (drawn by `seed`)."""
    th = np.asarray(th, float)
    a = np.random.default_rng(int(c["seed"])).uniform(-1.0, 1.0, 3)
    vary = 1.0 + 0.05 * (np.cos(th[..., None] * np.arange(1.0, 4.0)) @ a)
    return vary * np.cos(max(int(c["hang"]["folds"]), 1) * th - np.pi)


def hang_rings(fit, c, axis_y, z_top, z_hem, M=192):
    """(heights (K,) from z_top down below the hem, radii (K, M) round the axis (0, axis_y) at G.ring_theta(M), the
    convex sections (K, M)) where the cloth hangs (module docstring: hang, hem), its flutes laid on."""
    S, L = fit.S, fit.L
    hc, hm = c["hang"], c["hem"]
    th = G.ring_theta(M)
    dirs = G.hdir(th)
    step = 0.004 * S
    zs = np.arange(z_top, z_hem - 0.03 * S, -step)
    u, v = np.array([0.0, -1.0, 0.0]), np.array([1.0, 0.0, 0.0])
    H, last = [], None
    for z in zs:
        r = fit.core.outline(np.array([0.0, axis_y, z]), u, v, M, "far", rmin=0.02 * S, rmax=0.35 * S)
        if r is not None:
            last = hull_radii(circ_smooth(r, 2), dirs[:, :2])
        H.append(np.full(M, 0.1 * S) if last is None else last)
    H = row_smooth(np.array(H), 2)
    front, back = np.clip(np.cos(th), 0.0, 1.0) ** 1.5, np.clip(-np.cos(th), 0.0, 1.0) ** 1.5
    ease = (hc["ease_side"] + (hc["ease_front"] - hc["ease_side"]) * front
            + (hc["ease_back"] - hc["ease_side"]) * back) * S
    R = H + ease[None, :]
    for k in range(1, len(zs)):                  # it hangs from what is above it, coming in no faster than the taper
        R[k] = np.maximum(R[k], R[k - 1] - hc["taper"] * (zs[k - 1] - zs[k]))
    z_bust = float(L["bust_front"][2]) if L.has("bust_front") else float(L["upper_body2"][2]) + 0.05 * S
    z_hip = 0.5 * (float(L["leg.L"][2]) + float(L["leg.R"][2]))
    span = G.smoothstep(np.clip((z_bust - zs) / (0.03 * S), 0.0, 1.0)) \
        * G.smoothstep(np.clip((zs - z_hip) / (0.03 * S) + 1.0, 0.0, 1.0))
    R = np.maximum(R, H + hc["waist"] * S * span[:, None])                 # never following her in at the waist
    z_f = z_hem + hm["flare"] * S                # over the hem's last `flare` out to `off` off her hips at the hem
    kf = int(np.argmin(np.abs(zs - z_f)))
    Rh = H[int(np.argmin(np.abs(zs - z_hem)))] + hm["off"] * S
    s = np.clip((z_f - zs) / max(z_f - z_hem, 1e-6), 0.0, None) ** hm["power"]
    F = R[kf][None, :] + np.maximum(Rh - R[kf], 0.0)[None, :] * s[:, None]
    R = np.where((zs <= z_f)[:, None], np.maximum(R, F), R)
    R = circ_smooth(row_smooth(R, 4), 3)
    depth = hc["fold_depth"] * G.smoothstep(np.clip((z_bust - zs) / max(z_bust - z_f, 1e-6), 0.0, 1.0)) \
        + (hm["flute"] - hc["fold_depth"]) * G.smoothstep(np.clip((z_f - zs) / max(z_f - z_hem, 1e-6), 0.0, 1.0))
    R = R + 0.5 * S * depth[:, None] * flutes(th, c)[None, :]
    return zs, R, H


def ring_at(zs, R, P, axis_y):
    """(n,) the hanging radius at the points P (n, 3): bilinear in height and turn round the axis."""
    K, M = R.shape
    th = G.theta_of(P, axis_y)
    f = th / (TAU / M)
    j0 = np.floor(f).astype(int) % M
    j1 = (j0 + 1) % M
    a = f - np.floor(f)
    k = np.clip((zs[0] - P[:, 2]) / (zs[0] - zs[1]), 0.0, K - 1.000001)
    k0 = np.floor(k).astype(int)
    b = k - k0
    k1 = np.minimum(k0 + 1, K - 1)
    return ((1 - a) * (1 - b) * R[k0, j0] + a * (1 - b) * R[k0, j1] + (1 - a) * b * R[k1, j0] + a * b * R[k1, j1])


def hang_ring(zs, R, axis_y, th, z):
    """(n, 3) the hanging cloth (`hang_rings`) at the turns `th` (n,) round the axis (0, axis_y) and heights `z`."""
    d = G.hdir(th)
    Q = np.column_stack([np.zeros(len(d)), np.full(len(d), axis_y), np.broadcast_to(np.asarray(z, float), len(d))])
    return Q + d * ring_at(zs, R, Q + 0.1 * d, axis_y)[:, None]


def hung(P0, zs, R, axis_y, z_from, blend, soft):
    """Points P0 (n, 3) lying on her moved out from the axis (0, axis_y) to where the cloth hangs (`ring_at`): all of
    the way below z_from - blend, none above z_from, never in; `soft` rounds the corner where they meet."""
    P = np.array(P0, float)
    d = P[:, :2] - np.array([0.0, axis_y])
    r0 = np.maximum(np.linalg.norm(d, axis=1), 1e-9)
    w = G.smoothstep(np.clip((z_from - P[:, 2]) / max(blend, 1e-9), 0.0, 1.0))
    target = SH.soft_max(r0, ring_at(zs, R, P, axis_y), soft)
    r = r0 + w * (target - r0)
    P[:, :2] = np.array([0.0, axis_y]) + d * (r / r0)[:, None]
    return P, w


def arch_line(P, c, fit, z_hem, axis_y):
    """(n,) the height of the hem over the points P (n, 3): `z_hem`, the flutes' crests hanging `hem.wave` lower than
    their hollows, rising in the arch at her back (module docstring: arch) as a round arch whose feet are rounded."""
    S = fit.S
    ac = c["arch"]
    z = z_hem - 0.5 * c["hem"]["wave"] * S * flutes(G.theta_of(P, axis_y), c)
    hw = 0.5 * ac["width"] * S
    if hw <= 0.0:
        return z
    x0 = float(fit.L[ac["at"]][0]) if fit.L.has(ac["at"]) else 0.0
    rise, rnd = ac["rise"] * S, max(ac["round"] * S, 1e-6)
    t = np.abs(P[:, 0] - x0) / hw
    body = rise * np.sqrt(np.clip(1.0 - np.clip(t, 0.0, 1.0) ** 2, 0.0, 1.0))
    # the feet rounded: a soft max of the arch and the hem line with a corner `round` wide
    lift = SH.soft_max(body, np.zeros_like(body), 0.25 * rnd) - 0.25 * rnd * np.log(2.0)
    lift = np.where(t < 1.0 + 3.0 * rnd / hw, np.maximum(lift, 0.0), 0.0)
    back = G.smoothstep(np.clip((P[:, 1] - axis_y) / (0.03 * S), 0.0, 1.0))
    return z + lift * back


# ---------------------------------------------------------------- the tail's tube
def tube(path, radius):
    """(points (k, 3), radii (k, 2) [across, in its plane]) of a tube, or None when `path` is empty."""
    P = np.asarray(path, float).reshape(-1, 3)
    if len(P) < 2:
        return None
    R = np.asarray(radius, float).reshape(len(P), -1)
    return P, (np.repeat(R, 2, axis=1) if R.shape[1] == 1 else R[:, :2])


def tube_field(P, tb):
    """(n,) metres outside the tube `tb` (`tube`) of the points P (n, 3) (negative inside): measured across its path
    in the section's own ellipse (across: level, in plane: the rest), the ends cut square."""
    C, R = tb
    P = np.asarray(P, float)
    best = np.full(len(P), np.inf)
    for k in range(len(C) - 1):
        a, b = C[k], C[k + 1]
        ab = b - a
        t = (P - a) @ ab / max(float(ab @ ab), 1e-12)
        tc = np.clip(t, 0.0, 1.0)
        q = a + tc[:, None] * ab
        T = G.unit(ab)
        X = G.unit(np.cross(T, UP)) if abs(T[2]) < 0.99 else np.array([1.0, 0.0, 0.0])
        Y = np.cross(X, T)
        rr = R[k] + tc[:, None] * (R[k + 1] - R[k])
        d = P - q
        across = (np.hypot(d @ X / rr[:, 0], d @ Y / rr[:, 1]) - 1.0) * np.minimum(rr[:, 0], rr[:, 1])
        along = np.where((t < 0.0) | (t > 1.0), np.abs(d @ T), -np.inf)
        best = np.minimum(best, np.maximum(across, along))
    return best


# ---------------------------------------------------------------- the cloth
def cloth_shell(fit, c, body_part, axis_y, z_hem):
    """The smock's cloth as a Shell cut to its edges, its positions in the fields px, py, pz, `w` (how much of it
    hangs), and what the cuts used: (shell, info)."""
    S, L = fit.S, fit.L
    sh = weld_shell(SH.Shell.from_part(body_part))
    ring, axis, n, r = neck_frame(fit, c)
    j_z = 0.5 * (float(L["arm.L"][2]) + float(L["arm.R"][2]))
    z_from = j_z - c["hang"]["top"] * S
    zs, R, H = hang_rings(fit, c, axis_y, z_from + 0.01 * S, z_hem)
    z_b = j_z - c["armhole"]["bottom"] * S
    k_b = int(np.argmin(np.abs(zs - z_b)))
    M = R.shape[1]
    x_side = {"L": float(R[k_b, M // 4]), "R": float(R[k_b, 3 * M // 4])}
    arms = armhole_planes(fit, c, x_side)
    def arm_field(Q, side, pad=0.0):             # inside the armhole's plane, or under the armhole's bottom
        A, nn = arms[side]
        return np.maximum(-(Q - A) @ nn, z_b - 0.002 * S - Q[:, 2]) + pad
    # the region, roughly: below the neck, inside the armholes, above the hem, not the head, the arms or the hands
    V = sh.verts
    sh.fields.update(
        neck=neck_field(V, axis, n, r) + 0.02 * S,
        armL=arm_field(V, "L", 0.02 * S),
        armR=arm_field(V, "R", 0.02 * S),
        hem=V[:, 2] - (z_hem - 0.025 * S),
        head=0.5 - sh.share(OF.is_head),
        arm=0.5 - sh.share(OF.is_arm),
        hand=0.5 - sh.share(lambda b: any(b[1:].startswith(t) for t in OF.HAND_STEMS)))
    for k in ("hem", "neck", "armL", "armR", "head", "arm", "hand"):
        sh = sh.clip(sh.fields[k])
    sh.fields.clear()
    sh = sh.subdivide().subdivide()
    yc, hc = c["yoke"], c["hang"]
    P0 = sh.offset(yc["ease"] * S, relax=int(yc["relax"]), fit=fit, gap=0.6 * yc["ease"] * S)
    P, w = hung(P0, zs, R, axis_y, z_from, hc["blend"] * S, 0.004 * S)
    tb = tube(c["tail"]["path"], c["tail"]["radius"])
    sh.fields.update(px=P[:, 0], py=P[:, 1], pz=P[:, 2], w=w,
                     neck=neck_field(P, axis, n, r),
                     armL=arm_field(P, "L"),
                     armR=arm_field(P, "R"),
                     hem=P[:, 2] - arch_line(P, c, fit, z_hem, axis_y))
    cuts = ["neck", "armL", "armR", "hem"]
    if tb is not None:
        sh.fields["tail"] = tube_field(P, tb) - c["tail"]["margin"]
        cuts.append("tail")
    for k in cuts:
        sh = sh.clip(sh.fields[k])
    return sh, {"neck_ring": ring, "neck_centre": axis, "neck_normal": n, "arms": arms, "zs": zs, "R": R, "H": H,
                "z_from": z_from, "tube": tb}


def positions(sh):
    return np.stack([sh.fields["px"], sh.fields["py"], sh.fields["pz"]], 1)


def loops_by_kind(sh, P, z_hem, cen):
    """{"neck": loop, "L": loop, "R": loop, "hem": loop} of the cut cloth's boundary (vertex id lists)."""
    out = {}
    for lp in sh.loops():
        Q = P[lp]
        if Q[:, 2].min() < z_hem + 0.02:
            out["hem"] = lp
        elif np.linalg.norm(Q.mean(0)[:2] - cen[:2]) < 0.06 and Q[:, 2].mean() > cen[2] - 0.05:
            out["neck"] = lp
        else:
            out["L" if Q[:, 0].mean() > 0 else "R"] = lp
    return out


def rolled(P, N, loop, depth, mat, tag, u_of, v_of):
    """A strip from a cut edge (`loop`, positions P, the cloth's outward normals N) back in toward her by `depth`:
    the cloth's thickness where it ends."""
    lp = np.asarray(loop, np.int64)
    a, b = P[lp], P[lp] - N[lp] * depth
    n = len(lp)
    V = np.concatenate([a, b])
    faces, uv = [], []
    for k in range(n):
        k2 = (k + 1) % n
        faces.append((k2, k, n + k, n + k2))
        uv += [(u_of[lp[k2]], v_of[lp[k2]]), (u_of[lp[k]], v_of[lp[k]]), (u_of[lp[k]], v_of[lp[k]] - depth / WR.TILE),
               (u_of[lp[k2]], v_of[lp[k2]] - depth / WR.TILE)]
    uv = np.array(uv)
    du = uv[:, 0].reshape(-1, 4)
    wrap = du.max(1) - du.min(1) > 0.5
    du[wrap] = np.where(du[wrap] < 0.5 * (du[wrap].max(1, keepdims=True) + du[wrap].min(1, keepdims=True)),
                        du[wrap] + 1.0, du[wrap])
    uv[:, 0] = du.reshape(-1)
    return G.Patch(V, faces, uv, None, mat, tag), lp


def vertex_normals(P, T, centre_y):
    """Outward unit normals (n, 3) of the cloth at P (triangles T), turned away from the vertical axis (0, centre_y)
    where their winding says in."""
    N = SH.vertex_normals(P, T)
    d = P[:, :2] - np.array([0.0, centre_y])
    if np.sum(np.einsum("ij,ij->i", N[:, :2], d)) < 0:
        N = -N
    return N


# ---------------------------------------------------------------- lace points
def point_row(base, down, width, height, notch, mat, tag, lift=0.0, out=None):
    """A row of lace points hanging from the open polyline `base` (k, 3) (resampled to half a point's width), each
    `width` wide and `height` long from the line, pointing along `down` (k, 3), the notches between them `notch` under
    it; their top edge sewn on `lift` above the line and moved `-out` (k, 3: in, behind the cloth's edge). Returns the
    patch (u across each point 0..1, v 1 at the top to 0 at the tip)."""
    base = np.asarray(base, float)
    seg = np.linalg.norm(np.diff(base, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    n_pts = max(int(round(s[-1] / width)), 1)
    at = np.linspace(0.0, s[-1], 2 * n_pts + 1)
    B = np.stack([np.interp(at, s, base[:, k]) for k in range(3)], 1)
    D = G.unit(np.stack([np.interp(at, s, down[:, k]) for k in range(3)], 1))
    O = np.zeros_like(B) if out is None else np.stack([np.interp(at, s, out[:, k]) for k in range(3)], 1)
    tip = (np.arange(len(at)) % 2) == 1
    top = B - D * lift - O
    low = B + D * np.where(tip, height, notch)[:, None]
    V = np.concatenate([top, low])
    m = len(at)
    faces = [(k, k + 1, m + k + 1, m + k) for k in range(m - 1)]
    total = height + lift
    uv = []
    for k in range(m - 1):
        u0, u1 = (0.0, 0.5) if k % 2 == 0 else (0.5, 1.0)
        v0 = 1.0 - (lift + (height if tip[k] else notch)) / total
        v1 = 1.0 - (lift + (height if tip[k + 1] else notch)) / total
        uv += [(u0, 1.0), (u1, 1.0), (u1, v1), (u0, v0)]
    return G.Patch(V, faces, np.array(uv), None, mat, tag), B


def lace_picture(col, eyelet, size=64):
    """One lace point's picture (size, size, 4) uint8: the lace's colour, a little shade toward its tip, a small round
    eyelet in its middle (a shade of the lace) and a soft edge line along its two sides."""
    u = (np.arange(size) + 0.5) / size
    U, Vv = np.meshgrid(u, u[::-1])                           # V: 1 at the top row, 0 at the bottom
    base, shade = col["base"], col["shadow"]
    tipness = np.clip(1.0 - Vv / 0.85, 0.0, 1.0)
    rgb = base[None, None, :] * (1.0 - 0.25 * tipness[..., None]) + shade[None, None, :] * 0.25 * tipness[..., None]
    d = np.hypot(U - 0.5, (Vv - 0.52) * 0.85) / max(eyelet, 1e-6)
    ring = np.clip(1.0 - np.abs(d - 0.75) / 0.3, 0.0, 1.0)
    hole = np.clip(1.0 - d / 0.55, 0.0, 1.0)
    k = np.clip(0.8 * ring + 0.6 * hole, 0.0, 1.0)[..., None]
    rgb = rgb * (1.0 - k) + shade[None, None, :] * k
    return WR._rgba(rgb)


def hem_lace(fit, c, zs, R, axis_y, z_hem, gap_x, x_arch, weights_at):
    """The lace points along the hem (module docstring: lace) as a patch, leaving out the arch (`gap_x`: half its
    width, round `x_arch`), and where their tips hang all round her: {"theta" (n,), "r" (n,) from the axis, "z" (n,)}
    (the arch's left in)."""
    S = fit.S
    lc, hm = c["lace"], c["hem"]
    th = np.linspace(0.0, TAU, 2049)[:-1]
    d = G.hdir(th)
    zh = z_hem - 0.5 * hm["wave"] * S * flutes(th, c)

    ring, below = hang_ring(zs, R, axis_y, th, zh), hang_ring(zs, R, axis_y, th, zh - 0.01 * S)
    down = G.unit(G.unit(below - ring) + np.array([0.0, 0.0, -1.0]))
    tips = ring + down * lc["height"] * S
    back = (ring[:, 1] > axis_y) & (np.abs(ring[:, 0] - x_arch) < gap_x)
    if back.any():                                # start just past the arch's left foot, go round her front
        j0 = int(np.flatnonzero(back)[0])
        order = np.roll(np.arange(len(th)), -j0)
        keep = order[~back[order]]
    else:
        keep = np.concatenate([np.arange(len(th)), [0]])
    N_out = G.unit(np.column_stack([d[:, 0], d[:, 1], np.zeros(len(th))]))
    patch, B = point_row(ring[keep], down[keep], lc["width"] * S, lc["height"] * S, lc["notch"] * S, LACE,
                         "hem_lace", lift=2.5 * hm["thick"] * S, out=N_out[keep] * 1.4 * hm["thick"] * S)
    patch.w = weights_at(patch.v)
    return patch, {"theta": th, "r": np.hypot(tips[:, 0], tips[:, 1] - axis_y), "z": tips[:, 2]}


# ---------------------------------------------------------------- flaps, straps, studs
def surface(V, T, W=None):
    """An outfit_fit.Skin of triangles T over V (with weights W {bone: (n,)}) whose winding says outward as it is."""
    s = OF.Skin(V, np.asarray(T, np.int64), W)
    s._flip = False
    return s


def height_over(V, T, xy, z_max):
    """(n,) the highest z under z_max where the vertical line through each point xy (n, 2) meets the triangles T of
    V, and (n,) the triangle it meets (-1: none)."""
    A, B, C = V[T[:, 0]], V[T[:, 1]], V[T[:, 2]]
    xy = np.asarray(xy, float)
    z = np.full(len(xy), -np.inf)
    hit = np.full(len(xy), -1)
    det = (B[:, 1] - C[:, 1]) * (A[:, 0] - C[:, 0]) + (C[:, 0] - B[:, 0]) * (A[:, 1] - C[:, 1])
    ok = np.abs(det) > 1e-14
    A, B, C, det, idx = A[ok], B[ok], C[ok], det[ok], np.flatnonzero(ok)
    for s in range(0, len(xy), 64):
        p = xy[s:s + 64]
        l1 = ((B[None, :, 1] - C[None, :, 1]) * (p[:, None, 0] - C[None, :, 0])
              + (C[None, :, 0] - B[None, :, 0]) * (p[:, None, 1] - C[None, :, 1])) / det[None]
        l2 = ((C[None, :, 1] - A[None, :, 1]) * (p[:, None, 0] - C[None, :, 0])
              + (A[None, :, 0] - C[None, :, 0]) * (p[:, None, 1] - C[None, :, 1])) / det[None]
        l3 = 1.0 - l1 - l2
        inside = (l1 >= -1e-9) & (l2 >= -1e-9) & (l3 >= -1e-9)
        zz = l1 * A[None, :, 2] + l2 * B[None, :, 2] + l3 * C[None, :, 2]
        zz = np.where(inside & (zz < z_max), zz, -np.inf)
        k = np.argmax(zz, axis=1)
        z[s:s + 64] = zz[np.arange(len(p)), k]
        hit[s:s + 64] = np.where(np.isfinite(z[s:s + 64]), idx[k], -1)
    return z, hit


def lay_on(grid, sup, lift, passes=4):
    """A grid (rows, cols, 3) of cloth laid on the surface `sup` (an outfit_fit.Skin): smoothed `passes` times as
    cloth that bridges the dips under it, and kept `lift` off the surface (its edges too)."""
    G3 = np.array(grid, float)
    for _ in range(passes + 1):
        d, q, tri, bary = sup.closest(G3.reshape(-1, 3))
        n = sup.outward(tri, bary)
        flat = G3.reshape(-1, 3)
        low = d < lift
        flat[low] = q[low] + n[low] * lift
        G3 = flat.reshape(G3.shape)
        sm = G3.copy()
        sm[1:-1, 1:-1] = 0.5 * G3[1:-1, 1:-1] + 0.125 * (G3[:-2, 1:-1] + G3[2:, 1:-1] + G3[1:-1, :-2]
                                                         + G3[1:-1, 2:])
        G3 = sm
    d, q, tri, bary = sup.closest(G3.reshape(-1, 3))
    flat = G3.reshape(-1, 3)
    low = d < lift
    flat[low] = q[low] + sup.outward(tri[low], bary[low]) * lift
    return flat.reshape(G3.shape)


def kept_off(V, surf, margin, reach):
    """Points V (n, 3) within `reach` of the surface `surf` (an outfit_fit.Skin) moved out to `margin` off it where
    they are nearer than that or behind it."""
    V = np.array(V, float)
    d, q, tri, bary = surf.closest(V)
    low = (d < margin) & (np.abs(d) < reach)
    V[low] = q[low] + surf.outward(tri[low], bary[low]) * margin
    return V


def slab(top, thick, mat, tag, u, v):
    """A grid (rows, cols, 3) of cloth as a thin slab: its top surface, its underside `thick` under it and the
    walls round its edge (outward faces; `u`, `v` (rows, cols) its texture coordinates)."""
    R, Cn = top.shape[:2]
    n = G.unit(np.cross(np.gradient(top, axis=0), np.gradient(top, axis=1)))
    out = n if np.mean(n[..., 2]) > 0 else -n
    under = top - out * thick
    p_top = G.loft(top, closed=False, u=u[0], v=v[:, 0], mat=mat, tag=tag)
    p_top = G.orient_outward(p_top, (top - out * 0.01).reshape(-1, 3))
    p_bot = G.loft(under, closed=False, u=u[0], v=v[:, 0], mat=mat, tag=tag)
    p_bot = G.orient_outward(p_bot, (under + out * 0.01).reshape(-1, 3))
    ring = [(0, j) for j in range(Cn)] + [(i, Cn - 1) for i in range(1, R)] \
        + [(R - 1, j) for j in range(Cn - 2, -1, -1)] + [(i, 0) for i in range(R - 2, 0, -1)]
    edge = np.array([top[i, j] for i, j in ring])
    edge_u = np.array([u[i, j] for i, j in ring])
    wall = G.loft(np.stack([edge, np.array([under[i, j] for i, j in ring])]), closed=True,
                  u=np.concatenate([edge_u, edge_u[:1]]), v=[1.0, 1.0 - thick / WR.TILE], mat=mat, tag=tag)
    mid = 0.5 * (top + under).mean((0, 1))
    wall = G.orient_outward(wall, mid)
    return [p_top, p_bot, wall], ring, out


def flap(fit, c, side, neck_ring, sup):
    """One shoulder flap (module docstring: flap): (the grid (rows from the neckline out, cols from her front to her
    back, 3) of its top surface, the share of the way out of each row (rows,)). It lies on `sup` over the top of her
    shoulder, and from `leave` past the shoulder joint goes on out as a wing `droop` degrees below level, each line
    of it from her front to her back keeping the round it had where it left her."""
    S, L = fit.S, fit.L
    fc = c["flap"]
    sg = 1.0 if side == "L" else -1.0
    j = np.asarray(L[f"arm.{side}"], float)
    half = 0.5 * fc["depth"] * S
    x_out = abs(float(j[0])) + fc["out"] * S
    near = neck_ring[np.abs(neck_ring[:, 1] - float(j[1])) < 0.01 * S]
    x_in = float(np.abs(near[:, 0]).max()) if len(near) else abs(float(j[0])) * 0.45
    cols, rows = 17, 19
    b = np.linspace(-1.0, 1.0, cols)
    edge = x_out + fc["point"] * S * (1.0 - np.abs(b) ** 1.5) - 0.006 * S * (np.abs(b) ** 8)  # rounded corners
    a = np.linspace(0.0, 1.0, rows)
    X = x_in + (edge[None, :] - x_in) * a[:, None]
    Y = float(j[1]) + b[None, :] * half * (1.0 - 0.6 * a[:, None] ** 1.6)          # a soft kite toward its point
    Y = np.broadcast_to(Y, X.shape)
    V, T = sup.verts, sup.tris

    def over(x):                                # what lies under each point of the grid moved to x, or nan
        z, _ = height_over(V, T, np.stack([sg * x.ravel(), Y.ravel()], 1), float(j[2]) + 0.10 * S)
        z = z.reshape(X.shape)
        return np.where(np.isfinite(z) & (z > float(j[2]) - 0.08 * S), z, np.nan)
    x_w = abs(float(j[0])) + fc["leave"] * S
    Zw = over(np.full(X.shape, x_w))
    for i in range(rows):                       # past her shoulder's front and back: on from the nearest point
        k = np.flatnonzero(np.isfinite(Zw[i]))
        if len(k):
            near_k = k[np.argmin(np.abs(Y[i][:, None] - Y[i][k][None, :]), axis=1)]
            Zw[i] = np.where(np.isfinite(Zw[i]), Zw[i], Zw[i, near_k] - 0.35 * np.abs(Y[i] - Y[i, near_k]))
        else:
            Zw[i] = float(j[2])
    wing = Zw - np.tan(np.radians(fc["droop"])) * (X - x_w)
    Z = over(X)
    Z = np.where(X <= x_w, np.where(np.isfinite(Z), Z, Zw), np.fmax(Z, wing))
    grid = np.stack([sg * X, Y, Z + fc["lift"] * S], -1)
    return lay_on(grid, sup, fc["lift"] * S), a


def flap_weights(fit, c, side, P, a_share):
    """The weights of a flap's points P (n, 3): her skin's under them with the arm's share given to her shoulder, then
    `follow` of the arm's (her upper arm bone) growing from `follow_from` of the way out (`a_share` (n,)) to its edge."""
    fc = c["flap"]
    jp = "左" if side == "L" else "右"
    donors = [b for b in fit.skin.weights if not OF.is_head(b) and not any(b[1:].startswith(t) for t in OF.HAND_STEMS)]
    W = fit.weights_for(P, donors, default=f"{jp}肩")
    arm = sum((w for b, w in W.items() if OF.is_arm(b)), np.zeros(len(P)))
    W = {b: w for b, w in W.items() if not OF.is_arm(b)}
    W[f"{jp}肩"] = W.get(f"{jp}肩", 0.0) + arm
    t = G.smoothstep(np.clip((a_share - fc["follow_from"]) / max(1.0 - fc["follow_from"], 1e-6), 0.0, 1.0))
    return with_chains(W, {f"{jp}腕": fc["follow"] * t})


def strap_path(fit, c, side, over, z_end, n=160):
    """The centre line (k, 3) of one strap (module docstring: strap): the outline of `over` (an outfit_fit.Skin: the
    cloth and the flaps) in the plane `neck` from her midline, from her front at `z_end` over her shoulder to her back
    at `z_end`."""
    S, L = fit.S, fit.L
    sg = 1.0 if side == "L" else -1.0
    j = np.asarray(L[f"arm.{side}"], float)
    x = sg * c["strap"]["neck"] * S
    cen = np.array([x, float(j[1]), z_end + 0.01 * S])
    u, v = np.array([0.0, -1.0, 0.0]), np.array([0.0, 0.0, 1.0])
    r = over.outline(cen, u, v, 4 * n, "far", rmin=0.01 * S, rmax=0.25 * S)
    if r is None:
        raise ValueError("smock: no cloth where the strap goes ([smock.strap] neck)")
    phi = G.ring_theta(4 * n)
    P = cen[None] + (np.cos(phi)[:, None] * u + np.sin(phi)[:, None] * v) * r[:, None]
    order = np.argsort((phi + 0.5 * np.pi) % TAU)                      # from straight down at the front, round
    P = P[order]                                                         # over the top to straight down behind
    keep = P[:, 2] >= z_end
    first, last = int(np.argmax(keep)), len(keep) - 1 - int(np.argmax(keep[::-1]))
    P = P[first:last + 1]
    return P


def strap(fit, c, side, over, z_end, z_stud):
    """One strap and its stud: (patches (material STRAP), the stud's centre on the strap)."""
    S = fit.S
    sc = c["strap"]
    path = strap_path(fit, c, side, over, z_end)
    seg = np.linalg.norm(np.diff(path, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    at = np.linspace(0.0, s[-1], max(int(s[-1] / (0.004 * S)), 8))
    P = np.stack([np.interp(at, s, path[:, k]) for k in range(3)], 1)
    hw = 0.5 * sc["width"] * S
    across = np.array([1.0, 0.0, 0.0])
    cols = 5
    g = P[:, None, :] + across[None, None, :] * np.linspace(-hw, hw, cols)[None, :, None]
    g = lay_on(g, over, sc["lift"] * S, passes=2)
    u = np.broadcast_to(np.linspace(0.0, 1.0, cols)[None, :], g.shape[:2])
    v = np.full(g.shape[:2], 0.25)                                   # the strap's half of the picture, u across it
    parts, _, out = slab(g, sc["thick"] * S, STRAP, f"strap_{side}", u, v)
    mid = g[:, cols // 2]
    k = int(np.argmin(np.abs(mid[:, 2] - z_stud) + 10.0 * (mid[:, 1] > 0.0)))      # on her front
    along = G.unit(mid[min(k + 1, len(mid) - 1)] - mid[max(k - 1, 0)])
    return parts, mid[k], out[k, cols // 2], along


def stud(centre, nrm, along, size, proud, mat, tag):
    """A diamond stud `size` across with its points along the strap and across it, `proud` high in the middle: a low
    pyramid on a thin plate."""
    side = G.unit(np.cross(nrm, along))
    h = 0.5 * size
    base = np.array([centre + along * h, centre + side * h * 0.8, centre - along * h, centre - side * h * 0.8])
    lift = 0.35 * proud
    rim = base + nrm * lift
    apex = centre + nrm * proud
    V = np.concatenate([base, rim, [apex]])
    faces = [(0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7), (4, 5, 8), (5, 6, 8), (6, 7, 8), (7, 4, 8)]
    uv = np.array([(0.5, 0.0)] * 28)
    return G.orient_outward(G.Patch(V, faces, uv, None, mat, tag), centre - nrm * 0.01)


def flap_lace(fit, c, top, out, thick):
    """The lace points along a flap's free edges (front, outer, back) under its edge (`top`: its top surface's grid,
    `out` its normals, `thick` the flap): (patch, the base points)."""
    S = fit.S
    fc = c["flap"]
    R, Cn = top.shape[:2]
    free = [(i, 0) for i in range(R)] + [(R - 1, j) for j in range(1, Cn)] + [(i, Cn - 1) for i in range(R - 2, -1, -1)]
    E = np.array([top[i, j] - out[i, j] * thick for i, j in free])
    n_e = np.array([out[i, j] for i, j in free])
    away = G.unit(E - top[R // 2, Cn // 2])
    away = G.unit(away - n_e * np.einsum("ij,ij->i", away, n_e)[:, None])      # out of the flap, along it
    down = G.unit(0.6 * away + 0.8 * np.array([0.0, 0.0, -1.0]))
    return point_row(E, down, fc["lace_width"] * S, fc["lace_height"] * S, 0.0015 * S, LACE, "flap_lace",
                     lift=0.003 * S)


@builder("smock", needs=("body",))
def build(ctx):
    c = SP.merge(DEFAULTS, ctx.cfg)
    fit = OU.make_fit(ctx)
    col = {k: WR.colours(ctx, k, v) for k, v in COLORS.items()}
    S, L = fit.S, fit.L
    body_part = ctx.need("body")
    z_hip = 0.5 * (float(L["leg.L"][2]) + float(L["leg.R"][2]))
    z_hem = z_hip + c["hem"]["above"] * S
    axis_y = axis_of(fit)
    sh, cut = cloth_shell(fit, c, body_part, axis_y, z_hem)
    P = fit.push_out(sh.relax(positions(sh), 2), 0.6 * c["yoke"]["ease"] * S)      # relaxing pulls in at hollows
    N = vertex_normals(P, sh.tris, axis_y)
    trunk = [b for b in fit.skin.weights if not (OF.is_arm(b) or OF.is_leg(b) or OF.is_head(b))]
    hung_w = fit.weights_for(P, trunk, default="下半身", faces=[tuple(t) for t in sh.tris], smooth=2)
    own = sh.weights()
    w = sh.fields["w"]
    W = {b: (1.0 - w) * own.get(b, 0.0) + w * hung_w.get(b, 0.0) for b in set(own) | set(hung_w)}
    u = SH.azimuth(P, (0.0, axis_y))
    u_rep = round(0.9 / WR.TILE)
    v = P[:, 2] / WR.TILE
    soup = G.Soup("smock", [MAT_CLOTH, MAT_FLAP, MAT_STRAP, MAT_LACE])
    cloth = sh.patch(P, u, v, u_rep=u_rep, mat=CLOTH, tag="cloth")
    used = np.unique(sh.tris)
    cloth.w = {b: np.asarray(x, float)[used] for b, x in W.items()}
    soup.add(cloth)
    loops = loops_by_kind(sh, P, z_hem, cut["neck_centre"])
    for kind, lp in loops.items():
        depth = (c["hem"]["thick"] if kind == "hem" else c["neck"]["roll"] if kind == "neck"
                 else c["armhole"]["roll"]) * S
        rim, idx = rolled(P, N, lp, depth, CLOTH, f"rim_{kind}", u * u_rep, v)
        rim.v = fit.push_out(rim.v, 0.0005 * S)                     # rolled back toward her, never into her
        rim.w = {b: np.concatenate([np.asarray(x, float)[idx]] * 2) for b, x in W.items()}
        soup.add(rim)
    cloth_surf = surface(P, sh.tris, W)

    def weights_at(Q):
        return cloth_surf.transfer(Q, None, fallback="下半身")
    ac = c["arch"]
    x_arch = float(L[ac["at"]][0]) if ac["width"] > 0 and L.has(ac["at"]) else 0.0
    gap_x = 0.5 * ac["width"] * S + 0.6 * c["lace"]["width"] * S if ac["width"] > 0 else 0.0
    lace, tips = hem_lace(fit, c, cut["zs"], cut["R"], axis_y, z_hem, gap_x, x_arch, weights_at)
    soup.add(lace)
    info = {"z_hem": z_hem, "axis_y": axis_y, "loops": {k: len(x) for k, x in loops.items()},
            "hang": {"zs": cut["zs"], "R": cut["R"]}, "lace": c["lace"]["height"] * S, "lace_tips": tips,
            "tail": None if cut["tube"] is None else {"path": cut["tube"][0], "radius": cut["tube"][1]}}
    # ---- the flaps over the shoulders, the straps over the flaps
    sup = surface(np.concatenate([P, fit.skin.verts]), np.concatenate([sh.tris, fit.skin.tris + len(P)]))
    flap_tris, flap_V = [], []
    j_z = 0.5 * (float(L["arm.L"][2]) + float(L["arm.R"][2]))
    if c["flap"]["enabled"]:
        fc = c["flap"]
        for side in ("L", "R"):
            top, a = flap(fit, c, side, cut["neck_ring"], sup)
            R_, C_ = top.shape[:2]
            uu = np.broadcast_to(np.linspace(0.0, 1.5, C_)[None, :], (R_, C_))
            vv = np.broadcast_to(np.linspace(0.0, 1.2, R_)[:, None], (R_, C_))
            pieces, ring, out = slab(top, fc["thick"] * S, FLAP, f"flap_{side}", uu, vv)
            a_of = np.broadcast_to(a[:, None], (R_, C_))
            for p in pieces:
                _, _, tri, bary = surface(top.reshape(-1, 3), SW_grid(R_, C_)).closest(p.v)
                T3 = SW_grid(R_, C_)[tri]
                share = (a_of.reshape(-1)[T3] * bary).sum(1)
                p.w = flap_weights(fit, c, side, p.v, share)
                soup.add(p)
            lace, _ = flap_lace(fit, c, top, out, fc["thick"] * S)
            lace.v = kept_off(lace.v, cloth_surf, fc["lift"] * S, 0.015 * S)       # where it lies on her, on the cloth
            _, _, tri, bary = surface(top.reshape(-1, 3), SW_grid(R_, C_)).closest(lace.v)
            share = (a_of.reshape(-1)[SW_grid(R_, C_)[tri]] * bary).sum(1)
            lace.w = flap_weights(fit, c, side, lace.v, share)
            soup.add(lace)
            flap_V.append(top.reshape(-1, 3))
            flap_tris.append(SW_grid(R_, C_))
    if c["strap"]["enabled"]:
        sc = c["strap"]
        Vs, Ts, off = [P], [sh.tris], len(P)
        for Vf, Tf in zip(flap_V, flap_tris):
            Vs.append(Vf + 0.0)
            Ts.append(Tf + off)
            off += len(Vf)
        over = surface(np.concatenate(Vs), np.concatenate(Ts))
        z_end, z_stud = j_z - sc["end"] * S, j_z - sc["stud_at"] * S
        donors = [b for b in fit.skin.weights if not (OF.is_arm(b) or OF.is_head(b))]
        for side in ("L", "R"):
            parts, centre, nrm, along = strap(fit, c, side, over, z_end, z_stud)
            for p in parts:
                p.w = fit.weights_for(p.v, donors, default="上半身2")
                soup.add(p)
            st = stud(centre + nrm * sc["thick"] * S, nrm, along, sc["stud"] * S, sc["stud_thick"] * S, STRAP,
                      f"stud_{side}")
            st.uv = np.array([(0.5, 0.75)] * sum(len(f) for f in st.f))   # the stud's half of the picture
            st.w = fit.weights_for(st.v, donors, default="上半身2")
            soup.add(st)
    mesh = tidy(soup.mesh())
    mats = materials(ctx, c, col)
    front = P[np.argmin(np.abs(P[:, 0]) + 10.0 * np.maximum(P[:, 1] - axis_y, 0.0) + 5.0 * np.abs(P[:, 2] - z_hem))]
    info["landmarks"] = {"smock_hem": front}
    ctx.log(f"smock from its neckline at {cut['neck_ring'][:, 2].min():.3f} m down to its hem at {z_hem:.3f} m")
    return Part("smock", meshes=[mesh], materials=[m for m in mats if m.name in mesh.mats], info=info)


def SW_grid(R, C):
    """Triangles (2 (R - 1) (C - 1), 3) of an open (R, C) grid."""
    q = G.grid_quads(R, C, closed=False)
    return np.concatenate([q[:, [0, 1, 2]], q[:, [0, 2, 3]]])


def materials(ctx, c, col):
    """The smock's materials: matte cloth with a soft toon step, each outlined in a dark shade of its shadow."""
    def mat(name, en, cc, tex, edge):
        m = WR.material(ctx, name, en, cc["base"], tex, shadow=shadow_of({"c": cc["base"], "c_shadow": cc["shadow"]},
                                                                          "c"), edge_size=edge)
        m.toon = ctx.save_png(f"{en}_toon", toon_ramp(shadow=shadow_of({"c": cc["base"], "c_shadow": cc["shadow"]},
                                                                       "c"), edge=0.72, soft=0.18))
        m.edge_color = ink(cc["shadow"])
        return m
    sm, fl, st, tr = col["smock"], col["flaps"], col["straps"], col["trim"]
    strap_tex = np.broadcast_to(np.asarray(st["base"], float), (32, 16, 3)).copy()
    strap_tex[16:, :2] = strap_tex[16:, 14:] = st["shadow"]          # the strap, a darker line along each edge
    strap_tex[:16] = st["stud"]                                      # the stud
    return [mat(MAT_CLOTH, "smock", sm, WR.twill(sm["base"], lines=20, depth=0.02), 0.7),
            mat(MAT_FLAP, "flap", fl, WR.twill(fl["base"], lines=20, depth=0.02), 0.6),
            mat(MAT_STRAP, "strap", st, WR._rgba(strap_tex), 0.5),
            mat(MAT_LACE, "lace", tr, lace_picture(tr, c["lace"]["eyelet"]), 0.35)]
