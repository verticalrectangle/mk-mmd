"""Rin's two black cat ears (a slice of the hair part, called by hair.py as `build_ears(ctx, fit, cfg, rig, pal)`).

Each ear is a pointed triangular shell, cupped and leaning forward, standing on the scalp at `fit.cat[side]` and tilted
outward and a little forward; its base runs down through the hair volume and is clipped onto the skin (sunk 5 mm), so the
head hair parts around it. Outer shell and rim: black fur (`猫耳`), the recessed inner face: dark red -> pink (`猫耳内`)
inside the rim, a few pale tufts curling from the lower inner edge (`猫耳毛`).

Rig per side: a static twitch bone `猫耳左` / `猫耳右` (parent 頭, head at the base centre, pointing up the ear's axis, with a
tiny static body as the joint partner) and a stiff dynamic chain `猫耳左1..3` hanging from it (kind "ears"). mk's `twitch`
(family "ears") rotates exactly the twitch bones about their heads, so the ear pivots about its base; the base of the mesh is
rigid on the twitch bone and bends along the chain toward the tip.

Config `[hair.ears]` (DEFAULTS below; sizes scale with the head width published by the head part)."""
import numpy as np

from . import cat_common as CC
from . import cat_ear_geo as EG
from .cat_tex import ear_fur, ear_inner, tuft
from .hair_fit import angles
from .hair_geo import MeshAccum, Piece, unit
from .. import spec as SP
from ..part import Bone, Material, RigidBody

DEFAULTS = {
    "enabled": True,
    "width": None,              # m, base width where the ear leaves the hair (None: width_ratio x head width)
    "height": None,             # m, visible height above the hair (None: height_ratio x head width)
    "width_ratio": 0.50,        # x head width (0.095 m on the 0.19 m head of the reference proportions)
    "height_ratio": 0.46,       # x head width (0.087 m)
    "tilt_out_deg": 17.0,       # axis tilt from vertical, outward
    "tilt_fwd_deg": 8.0,        # axis tilt from vertical, forward
    "face_out_deg": 28.0,       # the inner face looks forward and this far outward
    "sink": 0.005,              # m, the base is sunk this far under the skin
    "hair_offset": -0.004,      # m, where the shell leaves the hair relative to the hair hull (skin + fit.vol)
    "thick": 0.18,              # x width, bulge of the back at the root
    "cup": 0.10,                # x width, depth of the inner bowl
    "lean": 0.20,               # x height, forward lean of the tip
    "curl": 0.26,               # edges curl forward by this x half width
    "apex_shift": 0.10,         # x width, the tip sits this far outward of the axis
    "lip": 0.72,                # bowl half width / ear half width
    "tufts": 4,                 # pale locks per ear (0 = none)
    "bones": 3,                 # chain bones per ear
    "radius": 0.0055,           # m, capsule radius of the chain bodies (x size scale)
    "hold": 0.5,                # share of the first chain bone that stays rigid with the twitch bone
    "seed": "hair.ears",
}
REF_W = 0.0739
SIDES = {"L": ("左", 1.0), "R": ("右", -1.0)}


def head_width(fit):
    """Skull width: twice the published x radius when there is one, else measured on the skin at the temples."""
    r = fit.info.get("head_radii")
    if r is not None:
        return 2.0 * float(r[0])
    from .hair_fit import dirs
    best = 0.0
    for ph in np.radians(np.arange(45.0, 85.0, 5.0)):
        d = dirs(np.array([np.pi / 2, -np.pi / 2]), np.array([ph, ph]))
        best = max(best, float((fit.skull.radius(d) * np.sin(ph)).sum()))
    return best


def _vol(fit):
    v = getattr(fit, "vol", None)
    if v is not None:
        return v
    return lambda th, ph: np.full(np.broadcast(th, ph).shape, 0.028)


def _frame(sg, cfg):
    """Axis u (up the ear), facing ey (the inner face looks along it), outward ex; `flip` when (ex, ey, u) is left-handed."""
    to, tf, fo = (np.radians(float(cfg[k])) for k in ("tilt_out_deg", "tilt_fwd_deg", "face_out_deg"))
    u = unit(np.array([sg * np.tan(to), -np.tan(tf), 1.0]))
    f0 = np.array([sg * np.sin(fo), -np.cos(fo), 0.0])
    ey = unit(f0 - (f0 @ u) * u)
    ex = np.cross(ey, u)
    if ex[0] * sg < 0:
        ex = -ex
    ex = unit(ex)
    flip = float(ex @ np.cross(ey, u)) < 0
    return u, ey, ex, flip


def _bisect(fn, p0, u, lo=-0.14, hi=0.14, iters=34):
    """z along u (per point of p0 (n, 3)) where fn(points) crosses zero upward."""
    n = len(p0)
    a, b = np.full(n, lo), np.full(n, hi)
    for _ in range(iters):
        m = 0.5 * (a + b)
        out = fn(p0 + m[:, None] * u[None]) > 0
        b = np.where(out, m, b)
        a = np.where(out, a, m)
    return 0.5 * (a + b)


def ear_frame(fit, side, cfg):
    """Everything about the ear's placement in model space (a dict): origin (skin), base centre O, axis u, facing ey,
    outward ex, flip, local functions `skin_z(x, y)` and `hull_z(x, y)` (local height of the skin / the hair hull)."""
    sg = SIDES[side][1]
    anchor = fit.cat[side]
    origin = np.asarray(anchor["origin"], float)
    u, ey, ex, flip = _frame(sg, cfg)
    O = origin - cfg["sink"] * u
    vol = _vol(fit)
    skull = fit.skull

    def pts(x, y):
        x, y = np.atleast_1d(np.asarray(x, float)), np.atleast_1d(np.asarray(y, float))
        return O[None] + x[:, None] * ex[None] + y[:, None] * ey[None]

    def skin_z(x, y):
        return _bisect(lambda P: skull.height_dist(P), pts(x, y), u) - cfg["sink"]

    def hull_z(x, y, off=0.0):
        def f(P):
            th, ph = angles(P - fit.center)
            return skull.height_dist(P) - vol(th, ph) - off
        return _bisect(f, pts(x, y), u)

    def world(L):
        L = np.atleast_2d(np.asarray(L, float))
        return O[None] + L[:, 0:1] * ex[None] + L[:, 1:2] * ey[None] + L[:, 2:3] * u[None]

    return dict(origin=origin, O=O, u=u, ey=ey, ex=ex, flip=flip, skin_z=skin_z, hull_z=hull_z, world=world, sg=sg,
                normal=np.asarray(anchor["normal"], float))


def _materials(ctx, pal, tex):
    """The three ear materials (draw order: fur, inner, tufts)."""
    toon_f = CC.toon(ctx, "toon_cat_fur", (0.55, 0.55, 0.62))
    toon_i = CC.toon(ctx, "toon_cat_inner", (0.60, 0.50, 0.58))
    toon_t = CC.toon(ctx, "toon_cat_tuft", (0.78, 0.70, 0.74))
    return [
        Material("猫耳", name_en="cat ear", diffuse=(1.0, 1.0, 1.0, 1.0), specular=(0.0, 0.0, 0.0), shininess=0.0,
                 ambient=(0.5, 0.5, 0.52), texture=tex["fur"], toon=toon_f, edge=True,
                 edge_color=CC.darker(pal["outer"], 0.45) + (1.0,), edge_size=0.8,
                 comment="cat ear fur: black with a cool sheen (generated)"),
        Material("猫耳内", name_en="cat ear inner", diffuse=(1.0, 1.0, 1.0, 1.0), specular=(0.0, 0.0, 0.0), shininess=0.0,
                 ambient=(0.55, 0.5, 0.52), texture=tex["inner"], toon=toon_i, edge=True,
                 edge_color=CC.darker(pal["inner_deep"], 0.55) + (1.0,), edge_size=0.5,
                 comment="inside of the cat ear: dark red to pink (generated)"),
        Material("猫耳毛", name_en="cat ear tufts", diffuse=(1.0, 1.0, 1.0, 1.0), specular=(0.0, 0.0, 0.0), shininess=0.0,
                 ambient=(0.6, 0.57, 0.57), texture=tex["tuft"], toon=toon_t, edge=True,
                 edge_color=CC.darker(pal["tuft_shadow"], 0.55) + (1.0,), edge_size=0.45,
                 comment="pale tufts at the lower inner edge (generated)"),
    ]


def build_ears(ctx, fit, cfg, rig, pal):
    cfg = SP.merge(DEFAULTS, cfg)
    if not cfg["enabled"]:
        return Piece(info={"enabled": False})
    rng = ctx.rng_for(cfg["seed"])
    col = CC.palette(ctx)
    hw = head_width(fit)
    W = float(cfg["width"]) if cfg["width"] else float(cfg["width_ratio"]) * hw
    H = float(cfg["height"]) if cfg["height"] else float(cfg["height_ratio"]) * hw
    k = W / REF_W
    tex = {"fur": ctx.save_png("cat_ear_fur", ear_fur(col, ctx.rng_for(cfg["seed"] + ".fur"))),
           "inner": ctx.save_png("cat_ear_inner", ear_inner(col, ctx.rng_for(cfg["seed"] + ".inner"))),
           "tuft": ctx.save_png("cat_ear_tuft", tuft(col, ctx.rng_for(cfg["seed"] + ".tuft")))}
    materials = _materials(ctx, col, tex)
    acc = MeshAccum("cat_ears", [m.name for m in materials])
    info = {"enabled": True, "size": {"width": W, "height": H, "scale": k, "head_width": hw}, "bones": [], "twitch": [],
            "sides": {}}
    frames = {"猫耳": []}
    head_body = ctx.find_body("頭")
    for side in [s for s in ("L", "R") if s in fit.cat]:
        jp, sg = SIDES[side]
        fr = ear_frame(fit, side, cfg)
        zh = float(fr["hull_z"](np.zeros(1), np.zeros(1), cfg["hair_offset"])[0])
        p = EG.EarParams(W=W, H=H, z_h=zh, thick=cfg["thick"] * W, rim_front=0.0025 * k, rim_back=0.0015 * k,
                         cup_depth=cfg["cup"] * W, lip=cfg["lip"], lean=cfg["lean"], curl=cfg["curl"],
                         apex_shift=cfg["apex_shift"])
        shell = EG.build_shell(p, fr["skin_z"])
        sh = shell["shape"]
        Vw = fr["world"](shell["verts"])
        faces = [list(f[::-1]) for f in shell["faces"]] if fr["flip"] else [list(f) for f in shell["faces"]]
        # ---- rig: twitch bone at the base centre, chain along the middle of the ear
        tw = f"猫耳{jp}"
        t_ch = np.linspace(-0.06, 1.0, int(cfg["bones"]) + 1)
        loc = []
        for t in t_ch:
            c, _ = sh.centre_half(t)
            yb = float(sh.back(0.0, t)) if t < 1.0 else float(sh.median(0.0, 1.0))
            yf = float(sh.front(0.0, t)) if t < 1.0 else float(sh.median(0.0, 1.0))
            loc.append([float(c), 0.5 * (yb + yf), float(sh.z_of(t))])
        pts = fr["world"](np.array(loc))
        rig.bones.append(Bone(name=tw, head=tuple(float(x) for x in fr["O"]), tail=tuple(float(x) for x in pts[0]),
                              parent="頭", name_en=f"cat ear {side}", tail_bone=f"{tw}1"))
        rig.bodies.append(RigidBody(name=tw, bone=tw, shape="sphere", size=(0.006 * k, 0.0, 0.0),
                                    location=tuple(float(x) for x in pts[0]), mode="static", group=15,
                                    no_collide=tuple(range(16)), mass=1.0))
        ch = rig.chain(tw, pts, anchor=tw, kind="ears", radius=cfg["radius"] * k, anchor_body=tw, hold=cfg["hold"],
                       name_en=f"cat ear {side} ")
        # ---- meshes: shell + tufts, weights from the chain
        w = ch.weights(Vw)
        mats = shell["face_mat"]
        acc.add(Vw, faces, shell["uv"], mat=mats, weights=w)
        n_tuft = int(cfg["tufts"])
        tuft_info = []
        if n_tuft > 0:
            specs = EG.tuft_specs(sh, n_tuft, W, ctx.rng_for(f"{cfg['seed']}.tufts.{side}"))
            for spec, st in zip(specs, EG.build_tufts(specs)):
                Vt = fr["world"](st["verts"])
                ft = [list(f[::-1]) for f in st["faces"]] if fr["flip"] else [list(f) for f in st["faces"]]
                acc.add(Vt, ft, st["uv"], mat=2, weights=ch.weights(Vt))
                tuft_info.append(fr["world"](spec["P"]))
        # ---- published frame
        base = shell["verts"][:shell["R"]]
        info["sides"][side] = {
            "twitch": tw, "bones": list(ch.names), "origin": fr["origin"], "base_centre": fr["O"], "axis": fr["u"],
            "facing": fr["ey"], "outward": fr["ex"], "tip": Vw[shell["apex"]], "chain": ch.pts.copy(),
            "z_hull": zh, "hull_ring": fr["world"](np.column_stack([base[:, 0], base[:, 1], np.full(len(base), zh)])),
            "footprint": Vw[:shell["R"]], "tufts": tuft_info,
        }
        info["bones"] += [tw] + list(ch.names)
        info["twitch"].append(tw)
        frames["猫耳"] += [tw] + list(ch.names)
    mesh = acc.to_mesh()
    return Piece(meshes=[mesh], materials=materials, info=info, frames=frames)
