"""Rin's outfit: the canonical dark green-black leaf-print dress with green frills (stand-up collar frill with a small black
bow, long slightly puffed sleeves with green frilled cuffs, thin black sash, full A-line skirt ending in a deep green
ruffle over a lighter inner frill), a black ribbon wound round the left calf with a bow near the ankle, and black
Mary-Jane shoes with a strap and a bow.

Everything is an offset shell of the body surface published by the body part (`outfit_fit`), built from rings
(`outfit_geo`): bodice, sleeves and ribbons take the body's own skin weights from the nearest surface point; the skirt is a
cone of chains (スカート) with dynamic bodies. Textures come from `outfit_tex`. Spec: `[outfit]` in outfit.toml; every key
has a default in DEFAULTS below (sizes are metres for a body 1.7 m tall, scaled by head_tip.z / 1.7).

Pieces and meshes: outfit_dress (bodice, sleeves, skirt: print fabric), outfit_frills (collar, cuff, hem ruffle: green;
inner frill: light green), outfit_ribbons (sash, bows, leg ribbon: black satin), outfit_shoes (leather, soles)."""
from ..build import builder
from ..part import Material, Part
from . import outfit_dress as D
from . import outfit_fit as F
from . import outfit_geo as G
from . import outfit_legs as LG
from . import outfit_rig as RG
from . import outfit_skirt as SK

DEFAULTS = {
    "bodice": {"segments": 64, "u_repeat": 2, "collar_h": 0.034, "shoulder_step": 0.003, "min_offset": 0.006,
               "shoulder_reach": 0.03},
    "sash": {"width": 0.0125},
    "bow": {"size": 0.034, "z_frac": -0.10},
    "sleeve": {"segments": 32, "rings": 30, "u_repeat": 1},
    "skirt": {"columns": 20, "rows": 4, "per_column": 10},
    "legs": {},
}

MAT_DRESS = "ドレス"
MAT_FRILL = "ドレスフリル"
MAT_FRILL_IN = "ドレス内フリル"
MAT_RUFFLE = "ドレス裾フリル"
MAT_SATIN = "黒リボン"
MAT_LEG = "脚リボン"
MAT_SHOE = "靴"
MAT_SOLE = "靴底"
# material order inside each mesh (patch.mat / Soup.add(mat=...) index these lists)
SATIN_MATS = [MAT_SATIN, MAT_LEG]
SHOE_MATS = [MAT_SHOE, MAT_SOLE]


def merge(base, over):
    out = dict(base)
    for k, v in (over or {}).items():
        out[k] = merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


MAT_KEYS = {"dress": MAT_DRESS, "frill": MAT_FRILL, "frill_inner": MAT_FRILL_IN, "ruffle": MAT_RUFFLE, "satin": MAT_SATIN,
            "leg": MAT_LEG, "shoe": MAT_SHOE, "sole": MAT_SOLE}


def apply_material_overrides(mats, over):
    """`[outfit.materials.<key>]` tables (keys: dress frill frill_inner ruffle satin leg shoe sole) set any Material field
    (specular, shininess, ambient, diffuse, sphere_mode, edge_size ...): look tuning without touching code."""
    by = {m.name: m for m in mats}
    for key, fields in (over or {}).items():
        if key not in MAT_KEYS:
            raise KeyError(f"[outfit.materials.{key}]: unknown material (have {', '.join(MAT_KEYS)})")
        m = by[MAT_KEYS[key]]
        for f, v in fields.items():
            if f == "name" or not hasattr(m, f):
                raise KeyError(f"[outfit.materials.{key}] {f}: not a Material field")
            setattr(m, f, tuple(v) if isinstance(v, (list, tuple)) else v)


def make_materials(tex, over=None):
    """Material declarations; `tex` maps texture keys to file names ({} -> flat colours); `over`: [outfit.materials]."""
    g = tex.get
    edge = (0.03, 0.06, 0.04, 1.0)
    mats = [
        Material(MAT_DRESS, "dress", diffuse=(1.0, 1.0, 1.0, 1.0), specular=(0.05, 0.06, 0.06), shininess=6.0,
                 ambient=(0.55, 0.60, 0.58), texture=g("dress", ""), toon=g("toon_dress", ""), double_sided=True,
                 edge=True, edge_color=edge, edge_size=0.8),
        Material(MAT_FRILL, "dress frill", diffuse=(1.0, 1.0, 1.0, 1.0), specular=(0.04, 0.05, 0.04), shininess=5.0,
                 ambient=(0.55, 0.62, 0.55), texture=g("frill", ""), toon=g("toon_frill", ""), double_sided=True,
                 edge=True, edge_color=(0.05, 0.12, 0.07, 1.0), edge_size=0.6),
        Material(MAT_RUFFLE, "dress hem ruffle", diffuse=(0.76, 0.86, 0.84, 1.0), specular=(0.04, 0.05, 0.04), shininess=5.0,
                 ambient=(0.55, 0.62, 0.55), texture=g("frill", ""), toon=g("toon_frill", ""), double_sided=True,
                 edge=True, edge_color=(0.04, 0.10, 0.06, 1.0), edge_size=0.6),
        Material(MAT_FRILL_IN, "dress inner frill", diffuse=(1.0, 1.0, 1.0, 1.0), specular=(0.04, 0.05, 0.04), shininess=5.0,
                 ambient=(0.62, 0.66, 0.58), texture=g("frill_inner", ""), toon=g("toon_frill", ""), double_sided=True,
                 edge=True, edge_color=(0.10, 0.18, 0.10, 1.0), edge_size=0.5),
        Material(MAT_LEG, "leg ribbon", diffuse=(1.0, 1.0, 1.0, 1.0), specular=(0.10, 0.10, 0.12), shininess=40.0,
                 ambient=(0.30, 0.30, 0.34), texture=g("satin_print", ""), toon=g("toon_satin", ""), sphere=g("sphere_satin", ""),
                 sphere_mode="add" if g("sphere_satin") else "none", double_sided=True, edge=True,
                 edge_color=(0.02, 0.02, 0.03, 1.0), edge_size=0.5),
        Material(MAT_SHOE, "shoes", diffuse=(1.0, 1.0, 1.0, 1.0), specular=(0.45, 0.45, 0.50), shininess=40.0,
                 ambient=(0.42, 0.42, 0.48), texture=g("leather", ""), toon=g("toon_leather", ""), sphere=g("sphere_leather", ""),
                 sphere_mode="add" if g("sphere_leather") else "none", double_sided=True, edge=True,
                 edge_color=(0.02, 0.02, 0.03, 1.0), edge_size=0.7),
        Material(MAT_SOLE, "soles", diffuse=(1.0, 1.0, 1.0, 1.0), specular=(0.10, 0.10, 0.10), shininess=8.0,
                 ambient=(0.45, 0.45, 0.48), texture=g("sole", ""), toon=g("toon_leather", ""), edge=True,
                 edge_color=(0.02, 0.02, 0.03, 1.0), edge_size=0.7),
        # no sphere map on the bows: mmd_tools adds it in linear light, so a flat sheet facing the camera (a bow loop) picks
        # up the map's value at its centre and turns mid-grey; black satin gets its sheen from the specular term instead
        Material(MAT_SATIN, "black ribbon", diffuse=(1.0, 1.0, 1.0, 1.0), specular=(0.20, 0.20, 0.23), shininess=20.0,
                 ambient=(0.30, 0.30, 0.34), texture=g("satin", ""), toon=g("toon_satin", ""), double_sided=True, edge=True,
                 edge_color=(0.02, 0.02, 0.03, 1.0), edge_size=0.6),
    ]
    apply_material_overrides(mats, over)
    return mats


def skin_from_body(ctx):
    body = ctx.need("body")
    meshes = [m for m in body.meshes if len(m.verts)]
    return F.Skin.from_meshes(meshes, "body")


def get_landmarks(ctx):
    land = dict(ctx.land)
    if "head_tip" in land and "neck" in land:
        return F.Land(land)
    merged = F.default_landmarks(1.0)
    merged.update(land)
    return F.Land(merged)


def textures(ctx):
    """Generated textures (file names by key); `[outfit] textures = false` skips them (flat material colours)."""
    if ctx.cfg.get("textures", True) is False:
        return {}
    from . import outfit_tex as T
    return T.build_textures(ctx, sizes=ctx.cfg.get("texture_sizes"))


def guided(ctx, cfg):
    """The measured outfit heights ([proportions.outfit_guides]) fill the keys the spec leaves open."""
    g = ctx.get("proportions.outfit_guides", {}) or {}
    out = merge(cfg, {})
    for sect, key, src in (("bodice", "waist_z", "sash_z"), ("bodice", "collar_top_z", "collar_top_z")):
        if src in g and not out.get(sect, {}).get(key):
            out.setdefault(sect, {})[key] = float(g[src])
    return out


@builder("outfit", needs=("body",))
def build(ctx):
    cfg = guided(ctx, merge(DEFAULTS, ctx.cfg))
    skin = skin_from_body(ctx)
    land = get_landmarks(ctx)
    fit = F.Fit(land, skin, log=ctx.log, bone_names=ctx.bones())
    tex = textures(ctx)
    mats = make_materials(tex, cfg.get("materials"))
    soup_dress = G.Soup("outfit_dress", [MAT_DRESS])
    soup_trim = G.Soup("outfit_frills", [MAT_FRILL, MAT_FRILL_IN, MAT_RUFFLE])
    soup_satin = G.Soup("outfit_ribbons", SATIN_MATS)
    soup_shoes = G.Soup("outfit_shoes", SHOE_MATS)
    rig = RG.Rig()
    info = D.bodice(fit, cfg["bodice"], soup_dress, soup_trim, rig, ctx.find_body)
    D.sash(fit, cfg["sash"], soup_satin, info)
    for side in ("L", "R"):
        D.sleeve(fit, side, cfg["sleeve"], soup_dress, soup_trim, info, rig, ctx.find_body)
    nb = ctx.find_body("首")
    D.throat_bow(fit, cfg["bow"], soup_satin, rig, info, "首", nb.name if nb else None)
    lb = ctx.find_body("下半身")
    sk = SK.skirt(fit, cfg["skirt"], info, soup_dress, soup_trim, rig, lb.name if lb else None)
    shin = ctx.find_body("左ひざ")
    statics = [rb for p in ctx.parts.values() for rb in p.bodies if rb.mode == "static"]
    LG.leg_ribbon(fit, cfg["legs"], soup_satin, rig, {"左ひざ": shin.name if shin else None}, statics)
    LG.shoes(fit, cfg["legs"], soup_shoes, soup_satin, rig)
    meshes = [m for m in (soup_dress.mesh(), soup_trim.mesh(), soup_satin.mesh(), soup_shoes.mesh())]
    used = {n for m in meshes for n in m.mats}
    part = Part("outfit", meshes=meshes, materials=[m for m in mats if m.name in used], bones=rig.bones,
                bodies=rig.bodies, joints=rig.joints,
                info={"scale": fit.S, "waist_z": info["z_waist"], "skirt": {"columns": sk["columns"], "rows": sk["rows"],
                                                                         "z_hem": sk["z_hem"], "z_nodes": sk["z_nodes"].tolist()}})
    return part
