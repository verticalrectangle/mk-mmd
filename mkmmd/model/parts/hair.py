"""Hair part: scalp cap, crown clumps, bangs, side locks and back hair (hair_head), twin braids with bows and tufts
(hair_braids), cat ears (ears) and cat tails (tails), with their bones, dynamic chains, rigid bodies and joints.

Spec: `[hair]` in hair.toml (see there; every key has a default in the module that uses it). Reads the head part's info
(skin shell, hairline, ears, cat-ear anchors, eyes ...) through `HeadFit` and the body part's landmarks; colours from
`[colors.hair]`. Chains: 前髪i_k (bangs), 横髪左/右1_k (side locks), 後髪i_k (back hair), 三つ編左/右k (braids), 猫耳左/右(k) (ears;
the ear base bone is the twitch bone), 尻尾1_k / 尻尾2_k (tails), リボン* (bow tails). Hair meshes are closed thin shells."""
import importlib

from ..build import builder
from ..part import Part
from . import hair_head
from . import hair_tex as TEX
from .hair_fit import HeadFit
from .hair_rig import Rig

# sub-builders: spec table -> (module, function); each takes (ctx, fit, cfg, rig, pal) and returns a Piece
SUB = (("braids", "hair_braids", "build_braids"), ("ears", "ears", "build_ears"), ("tails", "tails", "build_tails"))


@builder("hair", needs=("body", "head"))
def build(ctx):
    cfg = ctx.cfg
    head = ctx.need("head")
    fit = HeadFit(head.info, res_deg=float(cfg.get("fit_res_deg", 2.0)), meshes=head.meshes)
    colors = ctx.get("colors.hair") or {}
    pal = TEX.palette(colors)
    rig = Rig()
    pieces = [("head", hair_head.build_head_hair(ctx, fit, cfg, rig, pal))]
    for key, mod, fn in SUB:
        sub = cfg.get(key, {})
        if not sub.get("enabled", True):
            continue
        func = getattr(importlib.import_module(f"{__package__}.{mod}"), fn)
        pieces.append((key, func(ctx, fit, sub, rig, pal)))
    meshes, mats, info, frames = [], {}, {}, {}
    for key, pc in pieces:
        meshes += pc.meshes
        for m in pc.materials:
            mats.setdefault(m.name, m)
        info[key] = pc.info
        for fr, names in pc.frames.items():
            frames.setdefault(fr, []).extend(names)
    chains = {b: ch.names for b, ch in rig.chains.items()}
    info["chains"] = chains
    info["hair_top_z"] = float(pieces[0][1].info["top_z"])
    info["coverage_dirs"] = fit.hair_region_dirs()
    head_names = [n for k, v in pieces[0][1].info["chains"].items() for n in v]
    frames.setdefault("髪", []).extend(head_names)
    ctx.log(f"{sum(len(m.verts) for m in meshes)} verts, {len(rig.bones)} bones, {len(rig.bodies)} bodies, "
            f"{len(rig.joints)} joints, {len(chains)} chains")
    ordered = [m for m in mats.values() if not m.alpha_blend] + [m for m in mats.values() if m.alpha_blend]   # translucent last
    return Part("hair", meshes=meshes, materials=ordered, bones=rig.bones, bodies=rig.bodies,
                joints=rig.joints, frames=frames, info=info)
