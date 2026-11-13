"""The nightgown (part `gown`, numpy only): a long loose gown of striped cotton under a capelet, full sleeves gathered
into frilled wrist bands, ankle socks and flat shoes. It is sewn as such a gown is, from separate pieces of cloth with
the thickness of cloth at their edges, and composed from what the other garments have: the sweater's sleeves, their
chains, the colliders for their bags, the helper bones and the lining (sweater.py), the skirt's ring of tied chains
(outfit_rig.Rig.ring), the dress's frills (outfit_dress), the socks (socks.py) and the shoes (outfit_legs).

  neck      the neckline round the base of her neck: in a plane through `drop` below the neck joint at her sides,
            tilted `tilt` degrees down toward her front, `ease` off her skin. A frill `frill` tall stands up from it in
            soft pleats every `pleat` metres, `depth` deep, flaring `flare` out at its top
  yoke      her skin from the neckline down to a level seam `seam` above the bust's fullest point (landmark
            bust_front), her arms left out, `ease` off it and relaxed `relax` times; its edges turned in to the skin
            (at the neckline it meets her neck). Plain cloth (the picture's ground): the stripes start at its seam
  body      rings round a vertical axis through her hips from just inside the yoke's seam down to the hem: each the
            convex section of her body without her arms, `ease_*` off it (front, side, back), hanging from what is
            above it and coming in no faster than `taper` (it falls from her bust and her shoulder blades instead of
            following her in), at least `waist` off her skin from under the bust down to her hip joints, at most
            `front_max` off it across her front and her right hip from her hip joints up to the bust (a guitar rests
            there). Below the bust it falls out as an A-line (`hem`). Gathered under the seam into `folds` soft folds
            (fold_field) `fold_seam` deep from crest to hollow, easing to `fold_hip` over her hips, `run` of them
            running on to the hem and the rest fading out round her hips, and below the hips `waves` broad waves
            opening to `fold_hem` at the hem, so its hem line undulates; places, widths and depths of the folds vary by
            `vary`; rings every `step`, `gap` off her skin at least
  hem       the hem `above` the ankle joints, an ellipse `rx` by `ry` round the axis: from the bust (across her front
            from her hip joints) the gown goes out to it nearly straight (the share of the way ** `power`), its last
            `flare_height` flaring `flare` more; turned up inside, `thick` the cloth
  frill     the hem's frill, `height` long in soft pleats every `pleat`, `depth` deep, standing `flare` out at its edge
  stripes   the gown's picture (`picture`): `count` stripes round it from the yoke's seam down, each `share` of the
            pitch, straight down the folds to the hem (the body's columns keep their turn round the axis); one thin
            line `line_width` wide `line` above the frill; the folds shaded softly in it (`shade`)
  capelet   one piece of cloth from the neckline over her shoulders and the tops of her arms, a closed ring, drawn as
            it hangs with her arms lowered to `hang` degrees below horizontal (the lab's arms_down): in each half plane
            round the vertical through her neck the outer convex hull of the yoke, the gown and the sleeves (lowered
            with her arms) from the neckline down to a level edge `drop` below her shoulder joints, hanging straight
            down from its widest, `stand` off them (from `rim` at the neckline over its first `collar` metres),
            `columns` round by `rows` down, with `folds` soft radial folds (irregular) growing from none `fold_from`
            below the neckline to `fold_depth` from crest to hollow at its edge, the edge wavering `waver` up and
            down, its hem turned up inside, `thick`. It takes the weights of her skin under it there, smoothed over
            `smooth` metres round it, and goes back up onto her raised arms with them (inverse skinning): with her arms
            out it rides on them, the cloth between arm and chest a web, and when they come down it hangs as the cone
            it was drawn as. A frill `frill` long hangs from its edge in pleats every `pleat`, `depth` deep, flaring
            `flare`
  bow       a satin bow on the capelet at her collarbone, in the middle (outfit_geo.bow_ex): `width` across its two
            full loops, the ribbon `ribbon` wide, a knot `knot` across, two tails `tails` long lying on the capelet,
            `at` below the neckline's front, `lift` off the capelet
  sleeve    the sweater's sleeves (sweater.sleeve and its keys: `ease` a table along the arm, 0 at the shoulder, 1 at
            the wrist band; `sag`, `underarm`, `puff`, `gather`, `pleats`, `pleat_depth`, `elbow_fold`, `elbow_at`,
            `start_depth`), starting `start` below the shoulder joint were the arm hanging, under the capelet, with
            `folds` long soft folds down it, `fold_depth` deep from half way down the arm, gone in the band
  cuff      a band `band` wide ending `past` beyond the wrist, `offset` off the arm, `thick`, gathered every `rib`
            (`rib_depth`); from under it a frill `frill` long lies along her hand toward her fingers, opening `angle`
            degrees off the arm but never nearer her skin than `margin` (over the back of her hand and the base of her
            thumb it drapes instead of going in), in round soft pleats every `pleat`, `depth` deep (cuff_frill)
  chains    the sweater's chains under each sleeve (its keys: `upper`, `fore`, `radius`, `gap`, `peak`), family sleeve;
            none on the last `free` of the forearm (that much of it before the wrist goes with the arm alone, so the bag
            does not flap on a fast hand). Their capsules collide with the gown's colliders, not with her body's (hers
            stand off her arms into the bag and off her chest into her side)
  skirt     a ring of `columns` chains of `rows` bones round her (スカート, family skirt) hung from 下半身, their heads
            `from` above the hip joints; the first bone `root` long, the capsules `radius` thick or as thin as the room
            to her colliders at rest asks (outfit_rig.Rig.ring); below the heads (letting go over HOLD) the gown and its
            frill go over to the chains but `follow` of their trunk's weights
  colliders static capsules the size of the gown's body from the chains' heads up to the capelet's edge
            (sweater.knit_colliders: `step`, `inset`), for the sleeves' bags to lie on (group 12)
  helpers   the sweater's helper bones at each elbow and shoulder (its keys: `elbow`, `shoulder`, `peak`)
  lining    the skin the gown covers, `off` over it, from the neckline to the hip joints and along each arm to `cuff`
            short of the wrist band: her body wears the gown's picture, her arms the sleeves' cloth, so where the pieces
            part in a pose cloth shows, never skin (materials ガウン裏, ケープ裏, without outlines)
  socks     ankle socks (socks.socks and its keys: `top` above the ankle joints ...; no drawn lines)
  shoes     flat shoes (outfit_legs.shoes and its `shoe` keys): a low top line (`collar`), no strap, buckle or bow
            unless they are set; the sole's colour only under the foot, its sides the shoe's
  toon      the cloth's toon step: where it starts (`edge`, down the ramp) and how soft it is (`soft`)
  seed      the draw of the folds and the pleats

Sizes are metres x S (S = head_tip.z / 1.7 with leg_extra left out: 0.793 on the model bases, so 0.050 is 40 mm on
them). Spec: `[gown]` (DEFAULTS below, tables merge key by key) and `[colors.gown]`: `ground`, `stripe` (the gown,
with their `*_shadow` shades), `robe` (capelet and sleeves, `robe_shadow`), `trim` (frills, `trim_shadow`), `bow`
(`bow_shadow`), `sock` (`sock_shadow`), `shoe` (`shoe_shadow`) and `sole`. Each cloth is matte: its toon shade is its
shadow swatch over its colour, its outline a dark shade of the shadow. Materials ガウン (the striped gown), ケープ
(the capelet), ガウン袖 (sleeves and wrist bands), ガウンフリル (the frills), ガウンリボン (the bow), ガウン裏 and
ケープ裏 (the lining), ソックス, フラット靴 and フラット靴底."""
import numpy as np

from .. import spec as SP
from ..build import builder
from ..part import Part
from . import outfit as OU
from . import outfit_fit as OF
from . import outfit_geo as G
from . import outfit_legs as LG
from . import outfit_rig as ORIG
from . import outfit_shell as SH
from . import outfit_wear as WR
from . import socks as SO
from . import sweater as SW
from .hair_tex import toon_ramp
from .outfit_dress import copy_down, hang_frill
from .outfit_skirt import bone_names

TAU = 2.0 * np.pi
DEFAULTS = {
    "neck": {"drop": 0.013, "tilt": 15.0, "ease": 0.0028, "frill": 0.015, "pleat": 0.010, "depth": 0.0018,
             "flare": 0.0035},
    "yoke": {"seam": 0.050, "ease": 0.005, "relax": 10},
    "body": {"ease_front": 0.016, "ease_side": 0.012, "ease_back": 0.014, "taper": 0.10, "waist": 0.033,
             "front_max": 0.049, "folds": 18, "run": 0.5, "waves": 10, "vary": 0.35, "fold_seam": 0.009,
             "fold_hip": 0.004, "fold_hem": 0.015, "step": 0.010, "gap": 0.003},
    "hem": {"above": 0.0585, "rx": 0.378, "ry": 0.340, "power": 1.0, "flare": 0.019, "flare_height": 0.10,
            "thick": 0.0025},
    "frill": {"height": 0.050, "pleat": 0.022, "depth": 0.006, "flare": 0.012},
    "stripes": {"count": 28, "share": 0.30, "line": 0.010, "line_width": 0.004, "shade": 0.8},
    "capelet": {"drop": 0.113, "hang": 75.0, "columns": 160, "rows": 22, "stand": 0.010, "rim": 0.003,
                "collar": 0.025, "folds": 7, "fold_depth": 0.0032, "fold_from": 0.050, "waver": 0.0032,
                "thick": 0.003, "smooth": 0.035, "frill": 0.023, "pleat": 0.014, "depth": 0.003, "flare": 0.006},
    "bow": {"width": 0.088, "ribbon": 0.0176, "knot": 0.0126, "tails": 0.076, "at": 0.022, "lift": 0.003},
    "sleeve": {"ease": [[0.0, 0.008], [0.3, 0.012], [0.55, 0.022], [0.87, 0.038], [0.95, 0.030], [1.0, 0.016]],
               "sag": 0.010, "puff": 0.0, "pleats": 12, "pleat_depth": 0.0025, "gather": 0.05, "elbow_fold": 0.0,
               "elbow_at": [-0.012, 0.012], "start_depth": 0.006, "underarm": 0.020, "start": 0.0, "folds": 7,
               "fold_depth": 0.004},
    "cuff": {"band": 0.015, "past": 0.0, "offset": 0.004, "thick": 0.003, "rib": 0.006, "rib_depth": 0.0004,
             "frill": 0.028, "angle": 30.0, "pleat": 0.010, "depth": 0.0022, "margin": 0.002},
    "chains": {"enabled": True, "upper": 4, "fore": 3, "radius": 0.020, "gap": 0.03, "peak": 1.0, "free": 0.25},
    "skirt": {"enabled": True, "columns": 16, "rows": 5, "from": 0.0, "root": 0.02, "radius": 0.040, "follow": 0.0},
    "colliders": {"enabled": True, "step": 0.04, "inset": 0.003},
    "helpers": {"elbow": 0.5, "shoulder": 0.5, "peak": 0.8},
    "lining": {"enabled": True, "off": 0.001, "cuff": 0.004},
    "socks": {"top": 0.033, "snug": 0.0020, "rise": 0.052, "loose": 0.0030, "roll": 0.0035, "relax": 6, "gap": 0.0015},
    "shoes": {"shoe": {"strap": {"enabled": False}, "buckle": {"enabled": False}, "bow": {"enabled": False},
                       "collar": {"az": [0.0, 52.0, 70.0, 90.0, 110.0, 135.0, 180.0],
                                  "h": [-0.040, -0.040, -0.036, -0.032, -0.028, -0.024, -0.020]}}},
    "toon": {"edge": 0.72, "soft": 0.18},
    "seed": 5,
}
COLORS = {"ground": "#f6effa", "ground_shadow": "#cfc3de", "stripe": "#a886d6", "stripe_shadow": "#7d62ad",
          "robe": "#ead0ee", "robe_shadow": "#b99ccb", "trim": "#fbf7fd", "trim_shadow": "#d6cde2", "bow": "#d8344a",
          "bow_shadow": "#8f1d33", "sock": "#fbf7fd", "sock_shadow": "#d6cde2", "shoe": "#efc2d8",
          "shoe_shadow": "#b98aa6", "sole": "#8f6a80"}
MAT_GOWN = "ガウン"
MAT_ROBE = "ケープ"
MAT_SLEEVE = "ガウン袖"
MAT_TRIM = "ガウンフリル"
MAT_BOW = "ガウンリボン"
MAT_LINING = "ガウン裏"
MAT_LINING_ROBE = "ケープ裏"
MAT_SOCK = "ソックス"
MAT_SHOE = "フラット靴"
MAT_SOLE = "フラット靴底"
GOWN, ROBE, TRIM, BOW, SLEEVE = 0, 1, 2, 3, 4       # material indices of the gown's own mesh
PX = (2048, 1536)                                   # the gown's picture: round her (u), down it (v)
HOLD = 0.03                                         # metres above the chains' heads over which her hips let go
TRIM_TILE = 0.05                                    # metres of frill per texture tile


# ---------------------------------------------------------------- where things are on her skin
def neckline(fit, c, M=96):
    """(ring (M, 3), its centre (3,), the plane's unit normal (3,)) of the neckline: round her neck in a plane through
    `drop` below the neck joint at her sides, tilted `tilt` degrees down toward her front, `ease` off her skin (module
    docstring: neck). The ring starts at the middle of her front and goes toward her left."""
    S, L = fit.S, fit.L
    nc = c["neck"]
    t = np.radians(float(nc["tilt"]))
    n = np.array([0.0, -np.sin(t), np.cos(t)])
    u, v = np.array([0.0, -np.cos(t), -np.sin(t)]), np.array([1.0, 0.0, 0.0])
    c0 = np.array([0.0, float(L["neck"][1]), float(L["neck"][2]) - nc["drop"] * S])
    cen, r = fit.skin.section(c0, u, v, M, "near", rmin=0.005 * S, rmax=0.10 * S)
    if r is None:
        raise ValueError("gown: no neck where the neckline goes ([gown.neck] drop)")
    r = SW.circ_smooth(r, 6) + nc["ease"] * S
    phi = G.ring_theta(M)
    d = np.cos(phi)[:, None] * u[None] + np.sin(phi)[:, None] * v[None]
    return np.asarray(cen, float)[None] + d * r[:, None], np.asarray(cen, float), n


def start_depth(fit, sh, drop):
    """(n,) how far each skin vertex lies below the line `drop` under the shoulder joints were her arms hanging (the
    sweater band's line, sweater.band_depth): where the sleeves start."""
    band = {"band": {"chest": drop, "reach": 0.05, "fade": 0.025, "under": 0.0, "under_span": 0.14,
                     "under_width": 0.02}}
    return SW.band_depth(sh.verts, sh.share(SH.ARM), fit, band)[0]


def pic_v(z, z_hem, z_top):
    """The gown picture's v (up) at heights z: 0 at the hem, 1 at the top of the yoke."""
    return (np.asarray(z, float) - z_hem) / max(z_top - z_hem, 1e-9)


def weld_shell(sh, tol=1e-6):
    """The shell with its vertices at one place (to `tol` m) made one: an imported skin repeats its vertices along the
    seams of its texture, and a piece cut from it would open along them (edges, rims and frills where there is no edge).
    Normals, weights and fields of merged vertices are averaged."""
    key = np.round(sh.verts / tol).astype(np.int64)
    _, inv = np.unique(key, axis=0, return_inverse=True)
    inv = np.asarray(inv).reshape(-1)
    n = int(inv.max()) + 1
    cnt = np.bincount(inv, minlength=n).astype(float)[:, None]

    def mean(A):
        A = np.asarray(A, float).reshape(len(inv), -1)
        out = np.zeros((n, A.shape[1]))
        np.add.at(out, inv, A)
        return out / cnt
    T = inv[sh.tris]
    T = T[(T[:, 0] != T[:, 1]) & (T[:, 1] != T[:, 2]) & (T[:, 2] != T[:, 0])]
    return SH.Shell(mean(sh.verts), T, G.unit(mean(sh.normals)), sh.bones, mean(sh.W),
                    {k: mean(v)[:, 0] for k, v in sh.fields.items()})


def lift(surface, P, margin, iters=3):
    """Points P (n, 3) closer than `margin` (scalar or (n,)) to `surface` (an outfit_fit.Skin) or behind it, moved out
    to `margin` off it along its outward normal at their nearest points: cloth laid over other cloth."""
    P = np.array(P, float).reshape(-1, 3)
    m = np.broadcast_to(np.asarray(margin, float), (len(P),))
    for _ in range(iters):
        d, q, tri, bary = surface.closest(P)
        bad = d < m - 1e-6
        if not bad.any():
            break
        P[bad] = q[bad] + surface.outward(tri[bad], bary[bad]) * m[bad, None]
    return P


def surface_of(patches=(), verts=None, tris=None, weights=None):
    """An outfit_fit.Skin of the faces of several patches, or of `verts` and `tris` (with `weights`), whose winding
    says outward as it is (an open piece of cloth encloses no volume whose sign could tell)."""
    if verts is None:
        P = G.join([p for p in patches if p is not None and len(p.f)])
        verts, tris = P.v, SH.triangulate(P.f)
    s = OF.Skin(verts, np.asarray(tris, np.int64), weights)
    s._flip = False
    return s


# ---------------------------------------------------------------- her arms lowered
def arm_turns(fit, c, parents):
    """{side: (R, Rh, joint, the bones that turn with the arm, the shoulder helper's name)}: the turn about each
    shoulder joint that lowers the upper arm from her rest to `capelet.hang` degrees below horizontal (the lab's
    arms_down turns 腕 so), and the shoulder helper's share of it (`helpers.shoulder`, its rotation grant). The bones
    that turn with it are 腕's subtree in `parents` (bone -> parent): twist bones, elbow, hand, the sleeves' chains and
    the elbow helper."""
    out = {}
    for side, jp in SW.SIDES:
        sg = 1.0 if side == "L" else -1.0
        j, e = (np.asarray(fit.L[f"{n}.{side}"], float) for n in ("arm", "elbow"))
        turn = np.radians(float(c["capelet"]["hang"])) - np.arctan2(j[2] - e[2], abs(e[0] - j[0]))
        y = np.array([0.0, 1.0, 0.0])
        kids, grow = {f"{jp}腕"}, True
        while grow:
            new = {b for b, p in parents.items() if p in kids and b not in kids}
            kids |= new
            grow = bool(new)
        out[side] = (G.rot_axis(y, sg * turn), G.rot_axis(y, sg * turn * float(c["helpers"]["shoulder"])), j, kids,
                     f"{jp}腕補助")
    return out


def lowered(P, W, turns, inverse=False):
    """Points P (n, 3) with weights W {bone: (n,)} taken from her rest into the pose of `turns` (arm_turns: her arms
    lowered) by linear blend skinning as MMD does it, or from that pose back to her rest with `inverse`."""
    P = np.asarray(P, float).reshape(-1, 3)
    n = len(P)
    zero = np.zeros(n)
    tot = np.maximum(sum((np.asarray(w, float) for w in W.values()), zero), 1e-12)
    A = np.broadcast_to(np.eye(3), (n, 3, 3)).copy()
    t = np.zeros((n, 3))
    for R, Rh, j, kids, helper in turns.values():
        a = sum((np.asarray(W[b], float) for b in kids if b in W), zero) / tot
        h = np.asarray(W.get(helper, zero), float) / tot
        for wv, Mx in ((a, R), (h, Rh)):
            D = Mx - np.eye(3)
            A += wv[:, None, None] * D[None]
            t -= wv[:, None] * (D @ j)[None]
    if not inverse:
        return np.einsum("nij,nj->ni", A, P) + t
    return np.linalg.solve(A, (P - t)[..., None])[..., 0]


# ---------------------------------------------------------------- the yoke
def yoke(fit, sh, c, z_seam, axis_y, v_of):
    """The yoke's patch (material GOWN): her skin under the neckline and above the seam, her arms left out, `ease` off
    it, every edge turned in to her skin. Its u is the turn round the gown's axis (0, axis_y), v the picture's."""
    S = fit.S
    yc = c["yoke"]
    cut = SH.Shell(sh.verts, sh.tris, sh.normals, sh.bones, sh.W,
                   {"below": sh.fields["below"], "seam": sh.verts[:, 2] - z_seam, "arm": 0.5 - sh.fields["arm"]})
    for k in ("below", "seam", "arm"):
        cut = cut.clip(cut.fields[k])
    cut = cut.subdivide()
    ease = yc["ease"] * S
    P = cut.offset(ease, relax=int(yc["relax"]), fit=fit, gap=0.6 * ease)
    u = SH.azimuth(cut.verts, (0.0, axis_y))
    return cut.patch(P, u, v_of(cut.verts[:, 2]), mat=GOWN, tag="yoke", rims=cut.loops(), rim_depth=0.9 * ease)


# ---------------------------------------------------------------- the gown's body
def ellipse_r(theta, rx, ry):
    """(m,) radius of the ellipse rx (her side, x) by ry (her front and back, y) at the angles theta (0 her front)."""
    return 1.0 / np.sqrt((np.sin(theta) / rx) ** 2 + (np.cos(theta) / ry) ** 2)


def bumps(theta, centres, widths):
    """(n, m) soft bumps round a ring at the angles `theta` (m,): 1 at each of the n `centres`, falling smoothly to 0
    half its width (`widths`, radians) either side."""
    d = (np.asarray(theta, float)[None, :] - np.asarray(centres, float)[:, None] + np.pi) % TAU - np.pi
    return np.cos(np.pi * np.clip(d / np.asarray(widths, float)[:, None], -0.5, 0.5)) ** 2


def soft_folds(rng, count, vary, overlap=2.0):
    """(centres, widths, depths) of `count` soft folds round a ring drawn irregular: their places, widths and depths
    each `vary` (a share) about the even ones; `overlap` folds' widths to a place, so they run into one another."""
    n = max(int(count), 3)
    step = TAU / n
    centres = (np.arange(n) + rng.uniform(-0.3, 0.3, n)) * step + rng.uniform(0.0, TAU)
    return centres, overlap * step * (1.0 + rng.uniform(-vary, vary, n)), 1.0 + rng.uniform(-vary, vary, n)


def fold_field(c, rng, z_seam, z_hip, z_hem, S):
    """The gown's folds (module docstring: body): f(theta (m,), z (k,)) -> (k, m) metres its cloth stands out of
    (+) or in from (-) its rings. `folds` gathers start under the seam `fold_seam` deep and ease to `fold_hip` at the
    hips, some (`run` of them) running on to the hem and the rest fading out round the hips; below the hips `waves`
    broad waves open to `fold_hem` at the hem. Places, widths and depths vary by `vary`; round each height the field
    has no mean, so the stripes on the cloth keep their places."""
    bd = c["body"]
    vary = float(bd["vary"])
    gc, gw, gd = soft_folds(rng, bd["folds"], vary)
    runs = rng.uniform(0.0, 1.0, len(gc)) < float(bd["run"])
    ends = np.where(runs, z_hem - 1.0, z_hip + rng.uniform(-0.04, 0.10, len(gc)) * S)     # where each fades out
    wc, ww, wd = soft_folds(rng, bd["waves"], vary, overlap=1.4)

    def field(theta, z):
        z = np.asarray(z, float)[:, None]
        below = np.maximum(z_seam - z, 0.0)
        t = G.smoothstep(np.clip(below / max(z_seam - z_hip, 1e-6), 0.0, 1.0))
        g = (bd["fold_seam"] + (bd["fold_hip"] - bd["fold_seam"]) * t) * S \
            * G.smoothstep(np.clip(below / (0.008 * S), 0.0, 1.0))
        env = g * gd[None, :] * G.smoothstep(np.clip((z - ends[None, :]) / (0.12 * S) + 0.5, 0.0, 1.0))
        w = bd["fold_hem"] * S * G.smoothstep(np.clip((z_hip - z) / max(z_hip - z_hem, 1e-6), 0.0, 1.0)) ** 1.2
        D = env @ bumps(theta, gc, gw) + (w * wd[None, :]) @ bumps(theta, wc, ww)
        return D - D.mean(1, keepdims=True)
    return field


def body_rings(fit, c, axis_y, z_top, z_seam, z_hem, folds, M=192):
    """(rings (K, M, 3), outward directions (K, M, 3), heights (K,)) of the gown's body round the vertical axis
    (0, axis_y) from z_top (inside the yoke) down to the hem (module docstring: body, hem), its `folds` (fold_field)
    laid on."""
    S, L = fit.S, fit.L
    bd, hm = c["body"], c["hem"]
    th = G.ring_theta(M)
    dirs = G.hdir(th)
    step = bd["step"] * S
    zs = np.concatenate([np.arange(z_top, z_hem + 0.5 * step, -step), [z_hem]])
    u, v = np.array([0.0, -1.0, 0.0]), np.array([1.0, 0.0, 0.0])
    H, last = [], None
    for z in zs:
        r = fit.core.outline(np.array([0.0, axis_y, z]), u, v, M, "far", rmin=0.02 * S, rmax=0.35 * S)
        if r is not None:
            last = SW.hull_radii(SW.circ_smooth(r, 2), dirs[:, :2])
        H.append(np.full(M, 0.1 * S) if last is None else last)
    H = SW.row_smooth(np.array(H), 2)
    front, back = np.clip(np.cos(th), 0.0, 1.0) ** 1.5, np.clip(-np.cos(th), 0.0, 1.0) ** 1.5
    ease = (bd["ease_side"] + (bd["ease_front"] - bd["ease_side"]) * front
            + (bd["ease_back"] - bd["ease_side"]) * back) * S
    z_bust = float(L["bust_front"][2]) if L.has("bust_front") else z_seam - 0.04 * S
    z_hip = 0.5 * (float(L["leg.L"][2]) + float(L["leg.R"][2]))
    R = H + ease[None, :]
    for k in range(1, len(zs)):                  # it hangs from what is above it, coming in no faster than the taper
        R[k] = np.maximum(R[k], R[k - 1] - bd["taper"] * (zs[k - 1] - zs[k]))
    span = G.smoothstep(np.clip((z_bust - zs) / (0.03 * S), 0.0, 1.0)) \
        * G.smoothstep(np.clip((zs - z_hip) / (0.03 * S) + 1.0, 0.0, 1.0))
    R = np.maximum(R, H + bd["waist"] * S * span[:, None])                 # never following her in at the waist
    z_fall = z_bust
    kf = int(np.argmin(np.abs(zs - z_fall)))
    Rf = R[kf].copy()
    flare, fh = hm["flare"] * S, hm["flare_height"] * S
    Rh = ellipse_r(th, hm["rx"] * S, hm["ry"] * S) - flare
    s_side = np.clip((z_fall - zs) / max(z_fall - z_hem, 1e-6), 0.0, 1.0)
    s_front = np.clip((z_hip - zs) / max(z_hip - z_hem, 1e-6), 0.0, 1.0)
    fw = np.clip(np.cos(th), 0.0, 1.0) ** 2
    s = s_side[:, None] * (1.0 - fw[None, :]) + s_front[:, None] * fw[None, :]
    A = Rf[None, :] + np.maximum(Rh - Rf, 0.0)[None, :] * s ** hm["power"]
    A = A + flare * G.smoothstep(np.clip((z_hem + fh - zs) / fh, 0.0, 1.0))[:, None] ** 2     # its last flare
    R = np.where((zs <= z_fall)[:, None], np.maximum(R, A), R)              # the A-line
    guitar = G.smoothstep(np.clip((np.cos(th + 0.35) - 0.35) / 0.35, 0.0, 1.0))      # her front and right hip
    held = span[:, None] * guitar[None, :]
    R = R - held * np.maximum(R - (H + bd["front_max"] * S), 0.0)
    R = SW.circ_smooth(SW.row_smooth(R, 3), 3)
    tuck = G.smoothstep(np.clip((zs - (z_seam - 0.012 * S)) / (0.012 * S), 0.0, 1.0))     # in under the yoke
    R = (1.0 - tuck[:, None]) * R + tuck[:, None] * (H + 0.0015 * S)
    R = R + folds(th, zs)
    rings = np.array([np.array([0.0, axis_y, z]) + dirs * R[k][:, None] for k, z in enumerate(zs)])
    return rings, np.broadcast_to(dirs, rings.shape).copy(), zs


def gown_body(fit, c, rings, dirs, zs, axis_y, v_of):
    """The body's patch (material GOWN): the rings, its hem turned up inside. Returns (patch, the hem's ring (M, 3))."""
    S = fit.S
    thick = c["hem"]["thick"] * S
    up = np.array([0.0, 0.0, 1.0])
    hem = rings[-1] - dirs[-1] * thick
    turned = fit.push_out(np.concatenate([hem, hem + up * 0.010 * S]), 0.5 * c["body"]["gap"] * S).reshape(2, -1, 3)
    grid = np.concatenate([rings, turned])
    z = np.concatenate([zs, [zs[-1], zs[-1] + 0.010 * S]])
    M = rings.shape[1]
    axis = np.concatenate([np.zeros_like(grid[..., :1]), np.full_like(grid[..., :1], axis_y), grid[..., 2:3]], -1)
    p = G.orient_outward(G.loft(grid, True, u=np.linspace(0.0, 1.0, M + 1), v=v_of(z), mat=GOWN, tag="body"),
                         axis.reshape(-1, 3))
    return p, rings[-1]


# ---------------------------------------------------------------- frills
def arc_resample(P, n):
    """A closed polyline (k, 3) resampled to n points evenly spaced along it."""
    P = np.asarray(P, float)
    seg = np.linalg.norm(np.roll(P, -1, 0) - P, axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    at = np.arange(n) * s[-1] / n
    Pc = np.vstack([P, P[:1]])
    return np.stack([np.interp(at, s, Pc[:, k]) for k in range(3)], 1)


def ring_frill(ring, centre, along, prof, pleat, depth, tag, seed, fit=None, margin=0.0, wobble=0.3):
    """A frill round the closed ring `ring` (round `centre`), hanging `along` (3,): the profile `prof` (R, 2) of its
    rows (out from the centre, along), soft pleats every `pleat` `depth` deep, on the ring resampled evenly to six
    points a pleat, kept `margin` off her skin with `fit` (outfit_dress.hang_frill). Returns (patch (material TRIM),
    rings (R, m, 3), its base (m, 3))."""
    around = float(np.linalg.norm(np.roll(ring, -1, 0) - ring, axis=1).sum())
    pleats = max(int(round(around / max(pleat, 1e-6))), 6)
    m = int(np.ceil(6 * pleats / 4.0) * 4)
    base = arc_resample(ring, m)
    a = G.unit(np.asarray(along, float))
    out = np.asarray(base - centre, float)
    out = G.unit(out - np.outer(out @ a, a))
    prof = np.asarray(prof, float)
    patch, rings = hang_frill(base, out, np.broadcast_to(a, base.shape), prof, pleats, depth, TRIM, tag,
                              np.tile(base - out * 0.01, (len(prof), 1)), tile_u=TRIM_TILE, seed=seed, fit=fit,
                              margin=margin, wobble=wobble, ridge=0.05, hem_wobble=0.04 * float(np.abs(prof).max()))
    return patch, rings, base


def straight_frill(length, flare, rows=5):
    """The profile (rows, 2) (out, along) of a frill `length` long standing `flare` out at its edge, a straight cone."""
    t = np.linspace(0.0, 1.0, rows)
    return np.stack([t * flare, t * np.sqrt(max(length ** 2 - flare ** 2, 0.0))], 1)


# ---------------------------------------------------------------- the capelet, drawn with her arms lowered
def capelet_rings(fit, c, src, neck_ring, cy, z_edge, rng):
    """(rings (K, M, 3) from the neckline down to the edge, their outward normals (K, M, 3)) of the capelet as it
    hangs with her arms lowered (module docstring: capelet): `src` (an outfit_fit.Skin) is what lies under it in that
    pose, `neck_ring` the neckline, (0, cy) the vertical axis through her neck."""
    S = fit.S
    cp = c["capelet"]
    M, K = int(cp["columns"]), int(cp["rows"])
    th = G.ring_theta(M)
    dirs = G.hdir(th)
    an = G.theta_of(neck_ring, cy)
    o = np.argsort(an)
    rN = np.interp(th, an[o], np.hypot(neck_ring[o, 0], neck_ring[o, 1] - cy), period=TAU)
    zN = np.interp(th, an[o], neck_ring[o, 2], period=TAU)
    step = 0.003 * S
    zg = np.arange(z_edge, float(zN.max()) + step, step)
    u, v = np.array([0.0, -1.0, 0.0]), np.array([1.0, 0.0, 0.0])
    H = np.full((len(zg), M), np.nan)
    for i, z in enumerate(zg):                   # what lies under it, round each height: its convex outline
        r = src.outline(np.array([0.0, cy, z]), u, v, M, "far", rmin=0.01 * S, rmax=0.45 * S)
        if r is not None:
            H[i] = SW.hull_radii(SW.circ_smooth(r, 1), dirs[:, :2])
    t = np.linspace(0.0, 1.0, K)
    RZ = np.zeros((K, M, 2))
    arc = np.zeros((K, M))
    for j in range(M):                           # down each half plane: the outer convex hull, neckline to edge
        ok = np.isfinite(H[:, j]) & (zg < zN[j] - 0.002 * S)
        pts = np.stack([H[ok, j], zg[ok]], 1)
        Np = np.array([rN[j], zN[j]])
        Ep = np.array([max(float(pts[:, 0].max()) if len(pts) else rN[j], rN[j]), z_edge])   # straight down from
        hull = SW.convex_hull(np.vstack([Np, pts, Ep]))                                      # its widest
        iN = int(np.argmin(np.linalg.norm(hull - Np, axis=1)))
        iE = int(np.argmin(np.linalg.norm(hull - Ep, axis=1)))
        chain = hull[[(iE + k) % len(hull) for k in range((iN - iE) % len(hull) + 1)]][::-1]    # N .. E, outside
        s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(chain, axis=0), axis=1))])
        arc[:, j] = t * s[-1]
        RZ[:, j] = np.stack([np.interp(arc[:, j], s, chain[:, 0]), np.interp(arc[:, j], s, chain[:, 1])], 1)
    for k in range(1, K):                        # round each row smoothly (the neckline row stays)
        RZ[k] = SW.circ_smooth(RZ[k].T, 3).T
    T = np.gradient(RZ, axis=0)
    Nr = G.unit(np.stack([-T[..., 1], T[..., 0]], -1))                     # out of the meridian, away from her
    stand = cp["rim"] * S + (cp["stand"] - cp["rim"]) * S * G.smoothstep(np.clip(arc / (cp["collar"] * S), 0.0, 1.0))

    def ring_profile(count, overlap):            # irregular soft folds round it, no mean, 1 from crest to hollow
        fc, fw, fd = soft_folds(rng, count, 0.35, overlap)
        p = fd @ bumps(th, fc, fw)
        p = p - p.mean()
        return p / max(float(np.ptp(p)), 1e-9)
    span = np.maximum(arc[-1:] - cp["fold_from"] * S, 1e-6)
    grow = G.smoothstep(np.clip((arc - cp["fold_from"] * S) / span, 0.0, 1.0))      # none near the collar
    fold = cp["fold_depth"] * S * grow * ring_profile(cp["folds"], 1.6)[None, :]
    RZ = RZ + Nr * (stand + fold)[..., None]
    RZ[..., 1] += cp["waver"] * S * G.smoothstep(np.clip((t[:, None] - 0.6) / 0.4, 0.0, 1.0)) \
        * ring_profile(9, 1.5)[None, :]                                    # its edge wavers a little
    N3 = Nr[..., :1] * dirs[None] + Nr[..., 1:] * np.array([0.0, 0.0, 1.0])
    rings = np.array([0.0, cy, 0.0]) + RZ[..., :1] * dirs[None] + RZ[..., 1:] * np.array([0.0, 0.0, 1.0])
    rings = lift(src, rings.reshape(-1, 3), cp["rim"] * S).reshape(rings.shape)
    rings[0] = np.array([0.0, cy, 0.0]) + rN[:, None] * dirs + zN[:, None] * np.array([0.0, 0.0, 1.0])
    return rings, N3


def capelet(fit, c, src, skin_down, neck_ring, cy, z_edge, rng, donors):
    """The capelet with her arms lowered: (patch (material ROBE), its frill's patch, rings (K, M, 3), the edge's
    ring, its weights {bone: (K * M,)}) (module docstring: capelet)."""
    S = fit.S
    cp = c["capelet"]
    rings, N3 = capelet_rings(fit, c, src, neck_ring, cy, z_edge, rng)
    K, M = rings.shape[:2]
    thick = cp["thick"] * S
    up = np.array([0.0, 0.0, 1.0])
    hem = rings[-1] - N3[-1] * thick
    grid = np.concatenate([rings, [hem, hem + up * 3.0 * thick]])
    per = float(np.linalg.norm(np.roll(rings[-1], -1, 0) - rings[-1], axis=1).sum())
    v = -np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(grid, axis=0), axis=2).mean(1))]) / WR.TILE
    axis = np.concatenate([np.zeros_like(grid[..., :1]), np.full_like(grid[..., :1], cy), grid[..., 2:3]], -1)
    patch = G.orient_outward(G.loft(grid, True, u=np.linspace(0.0, max(round(per / WR.TILE), 1), M + 1), v=v,
                                    mat=ROBE, tag="capelet"), axis.reshape(-1, 3))
    W = skin_down.transfer(grid.reshape(-1, 3), set(donors), fallback="上半身2")
    col = per / M
    sigma = cp["smooth"] * S / max(col, 1e-9)
    Kg = len(grid)
    W = {b: SW.row_smooth(SW.round_smooth(w.reshape(Kg, M), sigma), 2).reshape(-1) for b, w in W.items()}
    patch.w = W
    fp, frings, base = ring_frill(hem + up * 1.5 * thick, np.array([0.0, cy, z_edge]), -up,
                                  straight_frill(cp["frill"] * S + 1.5 * thick, cp["flare"] * S), cp["pleat"] * S,
                                  cp["depth"] * S, "capelet_frill", 31)
    fp.v = lift(src, fp.v, 0.002 * S)
    near = np.argmin(((base[:, None, :] - rings[-1][None, :, :]) ** 2).sum(-1), axis=1)
    fp.w = copy_down({b: w.reshape(Kg, M)[K - 1][near] for b, w in W.items()}, len(frings))
    return patch, fp, rings, N3


def capelet_bow(fit, c, rings, N3, neck_ring, skin_down, donors):
    """The bow on the capelet at her collarbone (module docstring: bow): its patches (material BOW), rigid with the
    weights of the skin under its knot, its tails laid on the capelet, and its numbers."""
    S = fit.S
    bc = c["bow"]
    z_t = float(neck_ring[0, 2]) - bc["at"] * S
    col, nrm = rings[:, 0], N3[:, 0]                    # down the middle of her front
    k = int(np.clip(np.searchsorted(-col[:, 2], -z_t), 1, len(col) - 1))
    f = (col[k - 1, 2] - z_t) / max(col[k - 1, 2] - col[k, 2], 1e-9)
    P0 = col[k - 1] + f * (col[k] - col[k - 1])
    fwd = G.unit(nrm[k - 1] + f * (nrm[k] - nrm[k - 1]))
    right = np.array([1.0, 0.0, 0.0])
    up = G.unit(np.cross(fwd, right))
    size = 0.5 * bc["width"] * S                        # a loop from the knot's middle out to its end
    rib = bc["ribbon"] * S / (0.95 * size)              # outfit_geo.bow_ex's ribbon: this wide half way along a loop
    O = P0 + fwd * (bc["lift"] * S + 0.25 * size)
    bw = G.bow_ex(O, right, up, fwd, size=size, loop_len=1.0, loop_w=0.62, ribbon_w=rib, tail_len=bc["tails"] / bc[
        "width"] * 2.0, tail_splay=0.22, droop=0.10, knot=bc["knot"] / (1.8 * size / S), mat=BOW, tag="bow",
        tail_rows=10, notch=0.25, seed=5, cup=0.25, cols=5)
    K, M = rings.shape[:2]
    surf = surface_of(verts=rings.reshape(-1, 3), tris=SW.grid_tris(K, M))
    w = skin_down.transfer(O[None], set(donors), fallback="上半身2")
    tails = {id(p) for _, p in bw["tails"]}
    parts = []
    for p in bw["parts"]:
        if id(p) in tails:                              # the tails lie on the capelet
            _, q, tri, bary = surf.closest(p.v)
            p.v = q + surf.outward(tri, bary) * bc["lift"] * S
        p.v = lift(surf, p.v, bc["lift"] * S)
        p.w = {b: np.full(len(p.v), float(x[0])) for b, x in w.items()}
        parts.append(p)
    xs = np.concatenate([p.v for p in parts])
    loops = np.concatenate([p.v for p in bw["parts"][:2]])
    return parts, {"width": float(np.ptp(xs[:, 0])), "tails": float(O[2] - xs[:, 2].min()),
                   "loop": [float(np.ptp(loops[:, 0])) / 2.0, float(np.ptp(loops[:, 2]))],
                   "ribbon": float(rib * 0.95 * size)}


# ---------------------------------------------------------------- sleeves and cuffs
def sleeve_cfg(c):
    """The sweater-shaped table the sweater's sleeve, cuff, chain, collider and helper functions read."""
    cf = c["cuff"]
    return {"sleeve": c["sleeve"], "chains": c["chains"], "colliders": c["colliders"], "helpers": c["helpers"],
            "cuff": {"length": cf["band"], "past": cf["past"], "offset": cf["offset"], "thick": cf["thick"],
                     "rib": cf["rib"], "rib_depth": cf["rib_depth"]}}


def cuff_frill(fit, c, side, band, seed, rows=7):
    """The frill from under a wrist band `band` (its rings, sweater.cuff) (module docstring: cuff): rings round the
    arm's line from just inside the band's end along her hand, `frill` long, opening `angle` degrees off the arm, but
    never nearer her skin than `margin`: over the back of her hand and the base of her thumb it drapes instead of
    going in. Round soft pleats every `pleat`, `depth` deep, growing toward its edge. Returns (patch (material TRIM),
    rings (rows, m, 3))."""
    S = fit.S
    cf = c["cuff"]
    arm_len = SW.arm_frames(fit, side, np.zeros(1))[4]
    s0 = arm_len + cf["past"] * S - 0.003 * S
    a = np.radians(float(cf["angle"]))
    t = np.linspace(0.0, 1.0, rows)
    pts, _, U, V, _ = SW.arm_frames(fit, side, s0 + t * cf["frill"] * S * np.cos(a))
    face = band[SW.face_rows("cuff", band)][-1]
    d = face - pts[0]
    phi_b = np.arctan2(d @ V[0], d @ U[0]) % TAU
    o = np.argsort(phi_b)
    per = float(np.linalg.norm(np.roll(face, -1, 0) - face, axis=1).sum())
    pleats = max(int(round(per / (cf["pleat"] * S))), 8)
    m = 8 * pleats
    phi = G.ring_theta(m)
    r0 = 0.94 * np.interp(phi, phi_b[o], np.hypot(d @ U[0], d @ V[0])[o], period=TAU)
    rng = np.random.default_rng(seed)
    wave = np.cos(pleats * phi + 0.35 * G.periodic_noise(phi, int(rng.integers(1 << 30)), 4, 0.8))
    rings = []
    for k in range(rows):
        r = r0 + t[k] * cf["frill"] * S * np.sin(a)
        hand = fit.skin.outline(pts[k], U[k], V[k], m, "far", rmin=0.002 * S, rmax=0.08 * S)
        if hand is not None and k:
            r = np.maximum(r, SW.circ_smooth(hand, 3) + cf["margin"] * S)
        r = SW.circ_smooth(r, 2) + cf["depth"] * S * t[k] ** 1.2 * 0.5 * (1.0 + wave)
        rings.append(pts[k] + (np.cos(phi)[:, None] * U[k] + np.sin(phi)[:, None] * V[k]) * r[:, None])
    rings = np.array(rings)
    rings[1:] = fit.conform(rings, cf["margin"] * S, pin_first=True)[1:]
    u = np.linspace(0.0, max(round(per / TRIM_TILE), 1), m + 1)
    patch = G.loft(rings, True, u=u, v=np.linspace(1.0, 0.0, rows), mat=TRIM, tag=f"cuff_frill_{side}")
    return G.orient_outward(patch, np.repeat(pts, m, axis=0)), rings


def sleeves(fit, sh, c, soup, rig, find_body, rng, statics, donors):
    """Both sleeves, their wrist bands and frills into `soup`, the chains under them into `rig`. Returns info."""
    S, L = fit.S, fit.L
    sc = sleeve_cfg(c)
    cf = c["cuff"]
    arm_skin = fit.skin.where([n for n in fit.skin.weights if OF.is_arm(n)], 0.5, name="arm")
    start = SW.Surface(sh.verts, sh.tris, {"band": sh.fields["start"]})
    info = {}
    for side, pre in SW.SIDES:
        sr, sd, ss, H = SW.sleeve(fit, sc, side, arm_skin, start, rng, 64)
        arm_len = SW.arm_frames(fit, side, np.zeros(1))[4]
        sl = c["sleeve"]
        amp = sl["fold_depth"] * S * G.smoothstep(np.clip(ss / (0.5 * arm_len), 0.0, 1.0)) \
            * (1.0 - G.smoothstep(np.clip((ss - ss[-1] + 0.025 * S) / (0.015 * S), 0.0, 1.0)))
        sr = sr + sd * (amp[:, None] * SW.waves(G.ring_theta(sr.shape[1]), int(sl["folds"]), rng, spread=2)[None, :])[
            ..., None]
        sr = fit.conform(sr, c["body"]["gap"] * S)
        patch = SW._patch(sr, SW._circ_u(sr, WR.TILE), SW._rib_v(sr), SLEEVE, f"sleeve_{side}", sr - sd * 0.02 * S,
                          fit, donors, f"{pre}腕")
        fore = float(np.linalg.norm(np.asarray(L[f"wrist.{side}"], float) - np.asarray(L[f"elbow.{side}"], float)))
        s_cut = arm_len - c["chains"]["free"] * fore                  # the bag near the wrist goes with the arm alone
        if rig is not None:
            kc = int(np.searchsorted(ss, s_cut))
            share = SW.sleeve_chains(fit, sc, side, pre, sr[:kc], ss[:kc], H[:kc], rig, find_body, fit.bone_names,
                                     statics)
            full = {b: np.concatenate([np.asarray(w, float).reshape(kc, -1), np.zeros((len(ss) - kc, sr.shape[1]))])
                    for b, w in share.items()}
            patch.w = SW.with_chains(patch.w, full)
        soup.add(patch)
        per = float(np.linalg.norm(np.diff(sr[-1], axis=0), axis=1).sum())
        cr = SW.cuff(fit, sc, side, arm_skin, 4 * max(8, round(per / (cf["rib"] * S))))
        soup.add(SW._patch(cr, SW._rib_u(cr), SW._rib_v(cr), SLEEVE, f"cuff_{side}",
                           np.broadcast_to(cr.mean(1, keepdims=True), cr.shape), fit, donors, f"{pre}手首"))
        fp, fr = cuff_frill(fit, c, side, cr, 11 + (side == "R"))
        fp.w = fit.weights_for(fp.v, donors, default=f"{pre}手首", faces=fp.f, smooth=2)
        soup.add(fp)
        info[side] = {"rings": sr, "ss": ss, "H": H, "cuff": cr, "frill": fr, "arm_len": arm_len, "s_cut": s_cut}
    return info


# ---------------------------------------------------------------- the lining
def lining(fit, sh, c, axis_y, v_of, z_low):
    """The lining's patches (module docstring: lining): material 0 (the gown's picture) on her body, 1 (the sleeves'
    cloth) on her arms."""
    S, L, ln = fit.S, fit.L, c["lining"]
    V, share = sh.verts, sh.fields["arm"]
    reach = 0.5 * (SH.limb_length(L, "L") + SH.limb_length(L, "R")) - (c["cuff"]["band"] + ln["cuff"]) * S
    along = SH.both_sides(V, L)
    low = (1.0 - share) * (V[:, 2] - z_low) + share * (reach - along)
    cut = SH.Shell(V, sh.tris, sh.normals, sh.bones, sh.W, {
        "below": sh.fields["below"], "low": low, "leg": 0.5 - sh.share(SH.LEG), "arm": share,
        "u": SH.azimuth(V, (0.0, axis_y)), "v": v_of(V[:, 2]), "au": SH.limb_angle(V, L), "av": -along / WR.TILE})
    for k in ("below", "low", "leg"):
        cut = cut.clip(cut.fields[k])
    P = cut.offset(ln["off"] * S)
    arm = (cut.fields["arm"][cut.tris] >= 0.5).mean(1) > 0.5
    out = []
    if (~arm).any():
        out.append(cut.patch(P, cut.fields["u"], cut.fields["v"], tris=~arm, mat=0, tag="lining"))
    if arm.any():
        out.append(cut.patch(P, cut.fields["au"], cut.fields["av"], tris=arm, u_rep=round(0.15 * S / WR.TILE) or 1,
                             mat=1, tag="lining_arm"))
    return out


# ---------------------------------------------------------------- the rig
def skirt_chains(fit, c, rig, find_body, statics, rings, frill_rings, axis_y, patches):
    """The ring of chains the gown's skirt swings on (module docstring: skirt), and their share of the patches'
    weights. Returns info."""
    S, ch = fit.S, c["skirt"]
    C, R = max(int(ch["columns"]), 3), max(int(ch["rows"]), 1)
    names = bone_names(C, R)
    z_hip = 0.5 * (float(fit.L["leg.L"][2]) + float(fit.L["leg.R"][2]))
    z_free = z_hip + ch["from"] * S
    end = frill_rings[-1]
    nodes = ORIG.hung_heights(z_free, float(end[:, 2].min()), R, ch["root"] * S)
    hold = z_free + HOLD * S
    zs = rings[:, 0, 2]
    M = rings.shape[1]
    pts = np.zeros((C, R + 1, 3))
    th_e = G.theta_of(end, axis_y)
    for col in range(C):
        j = int(round(col * M / C)) % M
        for r, z in enumerate(nodes):
            k = int(np.clip(np.searchsorted(-zs, -z), 1, len(zs) - 1))
            t = (zs[k - 1] - z) / max(zs[k - 1] - zs[k], 1e-9)
            pts[col, r] = rings[k - 1, j] + t * (rings[k, j] - rings[k - 1, j])
        pts[col, R] = end[int(np.argmin(np.abs((th_e - col * TAU / C + np.pi) % TAU - np.pi)))]
    body = find_body("下半身") if find_body else None
    rad = rig.ring(names, pts, "下半身", "skirt", ch["radius"] * S, statics, body.name if body is not None else None,
                   "Skirt")
    keep = float(ch["follow"])
    for p in patches:
        share = ORIG.ring_weights(G.theta_of(p.v, axis_y), p.v[:, 2], names, nodes, hold, {}, C)
        p.w = SW.with_chains(p.w, {b: (1.0 - keep) * w for b, w in share.items()})
    return {"chains": C, "rows": R, "radii": [float(rad.min()), float(rad.max())], "z_free": float(z_free),
            "z_hold": float(hold), "names": names}


def overlaps(rig, statics):
    """[(gap m < 0, body, collider)]: the chains' dynamic bodies (not a chain's first, which collides with nothing of
    her body) that lie inside a static collider they collide with at rest."""
    out = []
    for rb in rig.bodies:
        if rb.mode != "dynamic" or rb.group == ORIG.ROOT_GROUP:
            continue
        hit = [s for s in statics if s.group not in rb.no_collide and rb.group not in s.no_collide]
        if not hit:
            continue
        r, h = float(rb.size[0]), float(rb.size[1])
        R = ORIG.euler_matrix(rb.rotation)
        a = np.asarray(rb.location, float) - R[:, 2] * 0.5 * h
        b = np.asarray(rb.location, float) + R[:, 2] * 0.5 * h
        gap = float(ORIG.chain_clearance([a, b], r, hit)[0])
        if gap < -1e-4:
            which = min(hit, key=lambda s: float(ORIG.union_sdf(np.array([a, b, 0.5 * (a + b)]), [s]).min()))
            out.append((gap, rb.name, which.name))
    return out


# ---------------------------------------------------------------- the pictures and materials
def picture(col, c, folds, z_hem, z_top, z_seam, S):
    """The gown's picture (PX[1], PX[0], 4) uint8: across it the way round her from the middle of her front toward her
    left, once (u); down it from the top of the yoke (row 0) to the hem (v 0). Above the seam plain `ground`; below,
    `count` stripes of `stripe`, each `share` of the pitch; the thin line `line` above the hem; the folds (`folds`:
    fold_field) shaded softly toward the shadow swatches in their hollows (`shade` at 4 mm x S deep) and a little
    lighter on their crests."""
    st = c["stripes"]
    Wd, Hd = PX
    u = (np.arange(Wd) + 0.5) / Wd
    z = z_top - (np.arange(Hd) + 0.5) / Hd * (z_top - z_hem)
    dz = (z_top - z_hem) / Hd
    n, share = int(st["count"]), float(st["share"])
    x = (u * n) % 1.0
    aa = 0.75 * n / Wd                                    # a pixel's width in stripe pitches
    inside = np.where(x < share, np.minimum(x, share - x), -np.minimum(x - share, 1.0 - x))
    m = np.clip(0.5 + inside / max(aa, 1e-9), 0.0, 1.0)[None, :] \
        * np.clip(0.5 + (z_seam - z) / dz, 0.0, 1.0)[:, None]                   # none on the yoke
    depth = folds(TAU * u, z) / (0.004 * S)
    hollow = (np.clip(-depth, 0.0, 1.0) ** 0.9 * float(st["shade"]))[..., None]
    crest = (np.clip(depth, 0.0, 1.0) ** 1.5 * 0.3 * float(st["shade"]))[..., None]
    ground = col["ground"] * (1.0 - hollow) + col["ground_shadow"] * hollow
    stripe = col["stripe"] * (1.0 - hollow) + col["stripe_shadow"] * hollow
    rgb = ground * (1.0 - m[..., None]) + stripe * m[..., None]
    rgb = rgb + (1.0 - rgb) * crest
    zl = z_hem + st["line"] * S + 0.5 * st["line_width"] * S
    line = np.clip(0.5 + (0.5 * st["line_width"] * S - np.abs(z - zl)) / dz, 0.0, 1.0)[:, None, None]
    rgb = rgb * (1.0 - line) + col["stripe"][None, None, :] * line
    return WR._rgba(rgb)


def satin(rgb, light, size=64):
    """A ribbon's picture: its colour with a soft sheen along it, a little off its middle (v across the ribbon)."""
    v = (np.arange(size) + 0.5) / size
    k = np.exp(-((v - 0.38) / 0.16) ** 2)[:, None, None]
    img = np.asarray(rgb, float)[None, None, :] * (1.0 - 0.35 * k) + np.asarray(light, float)[None, None, :] * 0.35 * k
    return WR._rgba(np.broadcast_to(img, (size, size, 3)))


def shadow_of(col, key):
    """The toon shade of a cloth: its shadow swatch over its colour, per channel."""
    return tuple(float(x) for x in np.clip(col[f"{key}_shadow"] / np.maximum(col[key], 1e-6), 0.3, 1.0))


def ink(rgb, k=0.62):
    """An outline colour: a dark shade of a cloth's shadow swatch (its hue, never black)."""
    return tuple(float(x) for x in np.clip(np.asarray(rgb, float) * k, 0.09, 1.0)) + (1.0,)


def tidy(mesh, h_min=2e-5):
    """The mesh without its slivers and unused vertices: a triangle whose height over its longest edge is under `h_min`
    metres is dropped, a quad that hides one becomes the triangle of its other three corners (or goes), as
    outfit_geo's Soup does for no area at all (Blender gives a sliver a normal of its own, against its winding)."""
    V = np.asarray(mesh.verts, float)
    lens = np.array([len(f) for f in mesh.faces])
    starts = np.concatenate([[0], np.cumsum(lens)[:-1]])
    Q = np.array([list(f) + [f[-1]] * (4 - len(f)) for f in mesh.faces])
    T = {3: (0, 1, 2), 2: (0, 1, 3), 1: (0, 2, 3), 0: (1, 2, 3)}       # corner k left out -> the triangle left

    def height(t):
        a, b, c = (V[Q[:, i]] for i in t)
        e = np.maximum.reduce([np.linalg.norm(b - a, axis=1), np.linalg.norm(c - b, axis=1),
                               np.linalg.norm(a - c, axis=1), np.full(len(a), 1e-12)])
        return np.linalg.norm(np.cross(b - a, c - a), axis=1) / e
    Hs = np.stack([height(T[k]) for k in range(4)], 1)                  # column k: corner k left out
    quad = lens == 4
    bad = np.where(quad, Hs.min(1) < h_min, Hs[:, 3] < h_min)
    best = np.argmax(Hs, 1)
    faces, uv, fm = list(mesh.faces), np.asarray(mesh.uv, float), np.asarray(mesh.face_mat, int)
    out_f, out_c, out_m = [], [], []
    for i in range(len(faces)):
        if not bad[i]:
            out_f.append(faces[i])
            out_c.append(np.arange(starts[i], starts[i] + lens[i]))
        elif quad[i] and Hs[i, best[i]] >= h_min:
            t = T[int(best[i])]
            out_f.append(tuple(faces[i][j] for j in t))
            out_c.append(starts[i] + np.asarray(t))
        else:
            continue
        out_m.append(fm[i])
    used = np.unique(np.concatenate([np.asarray(f) for f in out_f])) if out_f else np.zeros(0, np.int64)
    remap = np.full(len(V), -1, np.int64)
    remap[used] = np.arange(len(used))
    mesh.verts = V[used]
    mesh.faces = [tuple(int(x) for x in remap[np.asarray(f)]) for f in out_f]
    mesh.uv = uv[np.concatenate(out_c)] if out_c else np.zeros((0, 2))
    mesh.face_mat = np.asarray(out_m, int)
    mesh.weights = {b: np.asarray(w, float)[used] for b, w in mesh.weights.items()}
    mesh.morphs = {k_: np.asarray(d, float)[used] for k_, d in mesh.morphs.items()}
    return mesh


def materials(ctx, c, col, pic):
    """The gown's materials (outfit_wear.material, matte, with the soft toon step of `toon`)."""
    robe_tex = WR.twill(col["robe"], lines=18, depth=0.03)
    trim_tex = WR.twill(col["trim"], lines=18, depth=0.025)
    mats = [WR.material(ctx, MAT_GOWN, "gown", col["ground"], pic, shadow=shadow_of(col, "ground"), edge_size=0.7),
            WR.material(ctx, MAT_ROBE, "robe", col["robe"], robe_tex, shadow=shadow_of(col, "robe"), edge_size=0.7),
            WR.material(ctx, MAT_SLEEVE, "sleeve", col["robe"], robe_tex, shadow=shadow_of(col, "robe"), edge_size=0.7),
            WR.material(ctx, MAT_TRIM, "trim", col["trim"], trim_tex, shadow=shadow_of(col, "trim"), edge_size=0.45),
            WR.material(ctx, MAT_BOW, "bow", col["bow"], satin(col["bow"], np.clip(col["bow"] * 1.18, 0, 1)),
                        shadow=shadow_of(col, "bow"), edge_size=0.5),
            WR.material(ctx, MAT_LINING, "gown_lining", col["ground"], pic, shadow=shadow_of(col, "ground"),
                        edge_size=0.0),
            WR.material(ctx, MAT_LINING_ROBE, "robe_lining", col["robe"], robe_tex, shadow=shadow_of(col, "robe"),
                        edge_size=0.0),
            WR.material(ctx, MAT_SOCK, "sock", col["sock"], WR.rib(col["sock"], ribs=10, depth=0.06),
                        shadow=shadow_of(col, "sock"), edge_size=0.5),
            WR.material(ctx, MAT_SHOE, "shoe", col["shoe"], WR.twill(col["shoe"], lines=24, depth=0.015),
                        shadow=shadow_of(col, "shoe"), edge_size=0.6),
            WR.material(ctx, MAT_SOLE, "sole", col["sole"], WR.twill(col["sole"], lines=24, depth=0.02),
                        shadow=(0.82, 0.80, 0.88), edge_size=0.6)]
    shade = {MAT_GOWN: "ground", MAT_ROBE: "robe", MAT_SLEEVE: "robe", MAT_TRIM: "trim", MAT_BOW: "bow",
             MAT_LINING: "ground", MAT_LINING_ROBE: "robe", MAT_SOCK: "sock", MAT_SHOE: "shoe"}
    lines = {MAT_GOWN: "stripe_shadow", MAT_ROBE: "robe_shadow", MAT_SLEEVE: "robe_shadow", MAT_TRIM: "trim_shadow",
             MAT_BOW: "bow_shadow",
             MAT_SOCK: "sock_shadow", MAT_SHOE: "shoe_shadow", MAT_SOLE: "sole"}
    for m in mats:
        if m.name in shade:
            m.toon = ctx.save_png(f"{m.name_en}_toon", toon_ramp(shadow=shadow_of(col, shade[m.name]),
                                                               edge=float(c["toon"]["edge"]),
                                                               soft=float(c["toon"]["soft"])))
        if m.name in lines:
            m.edge_color = ink(col[lines[m.name]])
    return mats


# ---------------------------------------------------------------- the part
@builder("gown", needs=("body",))
def build(ctx):
    c = SP.merge(DEFAULTS, ctx.cfg)
    fit = OU.make_fit(ctx)
    col = WR.colours(ctx, "gown", COLORS)
    S, L = fit.S, fit.L
    rng = np.random.default_rng(int(c["seed"]))
    body_part = ctx.need("body")
    statics = [rb for rb in body_part.bodies if rb.mode == "static"]
    sh = weld_shell(SH.Shell.from_part(body_part))
    ring_n, neck_c, neck_n = neckline(fit, c)
    sh.fields["below"] = -(sh.verts - neck_c) @ neck_n
    sh.fields["arm"] = sh.share(SH.ARM)
    sh.fields["start"] = start_depth(fit, sh, c["sleeve"]["start"])
    trunk = [b for b in fit.skin.weights if not (OF.is_arm(b) or OF.is_leg(b) or OF.is_head(b))]
    donors = [b for b in fit.skin.weights if not any(b[1:].startswith(t) for t in OF.HAND_STEMS)]
    z_hip = 0.5 * (float(L["leg.L"][2]) + float(L["leg.R"][2]))
    z_ank = 0.5 * (float(L["ankle.L"][2]) + float(L["ankle.R"][2]))
    z_hem = z_ank + c["hem"]["above"] * S
    z_bust = float(L["bust_front"][2]) if L.has("bust_front") else float(L["upper_body2"][2]) + 0.06 * S
    z_seam = z_bust + c["yoke"]["seam"] * S
    z_top = float(ring_n[:, 2].max())
    z_edge = 0.5 * (float(L["arm.L"][2]) + float(L["arm.R"][2])) - c["capelet"]["drop"] * S
    core = fit.core.verts[np.unique(fit.core.tris)]
    hips = core[np.abs(core[:, 2] - z_hip) < 0.03 * S]
    axis_y = float(0.5 * (hips[:, 1].min() + hips[:, 1].max())) if len(hips) else 0.0

    def v_of(z):
        return pic_v(z, z_hem, z_top)
    soup = G.Soup("gown", [MAT_GOWN, MAT_ROBE, MAT_TRIM, MAT_BOW, MAT_SLEEVE])
    rig = ORIG.Rig()
    info = {}
    # ---- the yoke and the gown's body, its hem frill and the chains it swings on
    yoke_patch = soup.add(yoke(fit, sh, c, z_seam, axis_y, v_of))
    folds = fold_field(c, rng, z_seam, z_hip, z_hem, S)
    rings, dirs, zs = body_rings(fit, c, axis_y, z_seam + 0.010 * S, z_seam, z_hem, folds)
    rings = fit.conform(rings, c["body"]["gap"] * S)
    body_patch, hem_ring = gown_body(fit, c, rings, dirs, zs, axis_y, v_of)
    body_patch.w = fit.weights_for(body_patch.v, trunk, default="下半身", faces=body_patch.f, smooth=2)
    fr = c["frill"]
    hem_patch, hem_rings, _ = ring_frill(hem_ring - dirs[-1] * 0.6 * c["hem"]["thick"] * S,
                                         np.array([0.0, axis_y, float(hem_ring[:, 2].mean())]),
                                         np.array([0.0, 0.0, -1.0]),
                                         straight_frill(fr["height"] * S, fr["flare"] * S), fr["pleat"] * S,
                                         fr["depth"] * S, "hem_frill", 21, fit=fit, margin=c["body"]["gap"] * S)
    hem_patch.w = fit.weights_for(hem_patch.v, trunk, default="下半身")
    if c["skirt"]["enabled"]:
        info["skirt"] = skirt_chains(fit, c, rig, ctx.find_body, statics, rings, hem_rings, axis_y,
                                     [body_patch, hem_patch])
    shell = []
    if c["colliders"]["enabled"]:                # colliders the size of the gown for the sleeves' bags to lie on
        z_low = info["skirt"]["z_hold"] if "skirt" in info else z_hip
        shell = SW.knit_colliders(fit, sleeve_cfg(c), rings, z_low, z_edge, rig)
        for i, rb in enumerate(shell):
            rb.name = f"col_gown_{i + 1}"
    soup.add(body_patch)
    soup.add(hem_patch)
    # ---- sleeves, their bands and frills, and the chains under them
    info["sleeves"] = sleeves(fit, sh, c, soup, rig, ctx.find_body, rng, shell, donors)
    for rb in rig.bodies:                        # the bags lie on the gown's colliders; her own colliders stand off
        if rb.mode == "dynamic" and rb.group == ORIG.GROUPS["sleeve"]:     # her arms into the bag, her chest into
            rb.no_collide = tuple(sorted(set(rb.no_collide) | {0}))        # her side
    # ---- the capelet, drawn as it hangs with her arms lowered, over all of it
    parents = {b.name: b.parent for b in body_part.bones + rig.bones}
    turns = arm_turns(fit, c, parents)
    under = [p for p in soup.patches if p.tag == "yoke" or p.tag == "body" or p.tag.startswith("sleeve_")]
    V = np.concatenate([lowered(p.v, p.w, turns) for p in under] + [lowered(fit.skin.verts, fit.skin.weights, turns)])
    T = np.concatenate([SH.triangulate(p.f) + o for p, o in zip(under, np.cumsum([0] + [len(p.v) for p in under]))]
                       + [fit.skin.tris + sum(len(p.v) for p in under)])
    src = surface_of(verts=V, tris=T)
    skin_down = surface_of(verts=lowered(fit.skin.verts, fit.skin.weights, turns), tris=fit.skin.tris,
                           weights=fit.skin.weights)
    cape, cape_frill, cape_rings, cape_n = capelet(fit, c, src, skin_down, ring_n, float(neck_c[1]), z_edge, rng,
                                                   donors)
    bow_parts, info["bow"] = capelet_bow(fit, c, cape_rings, cape_n, ring_n, skin_down, trunk + ["首"])
    hung = [soup.add(cape), soup.add(cape_frill)] + [soup.add(p) for p in bow_parts]
    nf_ring = ring_n
    nc = c["neck"]
    neck_patch, neck_rings, neck_base = ring_frill(nf_ring, neck_c, neck_n,
                                                   straight_frill(nc["frill"] * S, nc["flare"] * S), nc["pleat"] * S,
                                                   nc["depth"] * S, "neck_frill", 41)
    neck_patch.w = copy_down(fit.weights_for(neck_base, donors, default="首"), len(neck_rings))
    soup.add(neck_patch)
    bones = SW.joint_helpers(fit, soup, sleeve_cfg(c), {b.name: b.parent for b in body_part.bones})
    parents.update({b.name: b.parent for b in bones})
    turns = arm_turns(fit, c, parents)
    rest = surface_of([yoke_patch, body_patch] + [p for p in soup.patches if p.tag.startswith("sleeve_")])
    for p in hung:                               # back onto her raised arms, as her skin goes
        p.v = lowered(p.v, p.w, turns, inverse=True)
        p.v = fit.push_out(lift(rest, p.v, c["capelet"]["rim"] * S), c["body"]["gap"] * S)
    meshes = [tidy(soup.mesh())]
    # ---- lining, socks and shoes
    if c["lining"]["enabled"]:
        lined = G.Soup("gown_lining", [MAT_LINING, MAT_LINING_ROBE])
        for p in lining(fit, sh, c, axis_y, v_of, z_hip):
            lined.add(p)
        meshes.append(tidy(lined.mesh()))
    feet = G.Soup("gown_socks", [MAT_SOCK])
    info["socks"] = SO.socks(fit, body_part, dict(c["socks"], ink={"enabled": False}), feet)
    shoes = G.Soup("gown_shoes", [MAT_SHOE, MAT_SOLE])
    info["shoes"] = {k: v for k, v in LG.shoes(fit, c["shoes"], shoes, G.Soup("gown_shoe_bows", [MAT_SHOE])).items()
                     if k in ("min_dist", "sole_min_z")}
    for p in [p for p in shoes.patches if p.mat == 1]:          # the sole's colour only under the foot
        n = G.face_normals(p.v, p.f)
        under = n[:, 2] < -0.5 * np.linalg.norm(n, axis=1)
        if (~under).any():
            side = G.Patch(p.v, [f for f, u in zip(p.f, under) if not u], None, p.w, 0, p.tag)
            k = np.repeat(~under, [len(f) for f in p.f])
            side.uv = p.uv[k]
            p.f, p.uv = [f for f, u in zip(p.f, under) if u], p.uv[~k]
            shoes.add(side)
    meshes += [tidy(feet.mesh()), tidy(shoes.mesh())]
    # ---- materials and checks
    pic = picture(col, c, folds, z_hem, z_top, z_seam, S)
    mats = materials(ctx, c, col, pic)
    for gap, bone, which in overlaps(rig, statics + shell):
        ctx.log(f"WARNING gown chain {bone} is {-gap * 1000:.0f} mm inside {which} at rest: MMD's physics will throw "
                f"it out ([gown.chains], [gown.skirt])")
    info.update(z_hem=z_hem, z_seam=z_seam, z_top=z_top, z_edge=z_edge, axis_y=axis_y, neck=ring_n, rings=rings,
                hem_rings=hem_rings, colliders=len(shell),
                chains=sum(1 for b in rig.bones if "袖下" in b.name and b.name.endswith("_1")))
    ctx.log(f"gown from {z_top:.3f} m (neckline) to its hem at {z_hem:.3f} m, capelet edge at {z_edge:.3f} m, "
            f"{info.get('skirt', {}).get('chains', 0)} chains round its skirt, {info['chains']} under the sleeves, "
            f"{len(shell)} colliders, bow {info['bow']['width'] * 1000:.0f} mm wide")
    used = {name for m in meshes for name in m.mats}
    return Part("gown", meshes=meshes, materials=[m for m in mats if m.name in used], bones=bones + rig.bones,
                bodies=rig.bodies, joints=rig.joints, info=info)
