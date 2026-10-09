"""Hair part: scalp cap, crown clumps, bangs, side locks and back hair (hair_head), twin braids with bows and tufts
(hair_braids), cat ears (ears) and cat tails (tails), with their bones, dynamic chains, rigid bodies and joints.

Spec: `[hair]` in hair.toml (see there; every key has a default in the module that uses it). Reads the head part's info
(skin shell, hairline, ears, cat-ear anchors, eyes ...) through `HeadFit` and the body part's landmarks; colours from
`[colors.hair]`. Chains: 前髪i_k (bangs), 横髪左/右1_k (side locks), 後髪i_k (back hair), 三つ編左/右k (braids), 猫耳左/右(k) (ears;
the ear base bone is the twitch bone), 尻尾1_k / 尻尾2_k (tails), リボン* (bow tails). Hair meshes are closed thin shells."""
import importlib

import numpy as np

from ..build import builder
from ..part import Part
from . import cat_common as CC
from . import hair_head
from . import hair_tex as TEX
from .hair_fit import HeadFit
from .hair_rig import Rig

# sub-builders: spec table -> (module, function); each takes (ctx, fit, cfg, rig, pal) and returns a Piece
SUB = (("braids", "hair_braids", "build_braids"), ("ears", "ears", "build_ears"), ("tails", "tails", "build_tails"))
# which table sets each chain family (by the chain's name), for the warnings
TABLE = (("前髪", "[hair.bangs]"), ("横髪", "[hair.side]"), ("後髪", "[hair.back] (delta moves it out, end_above_chin "
         "sets its length)"), ("三つ編", "[hair.braids]"), ("リボン", "[hair.braids]"), ("猫耳", "[hair.ears]"),
         ("尻尾", "[hair.tails]"))


def overlaps(rig, bodies):
    """[(gap m < 0, chain, bone, collider)]: the chains whose capsules overlap one of the static `bodies` at rest, the worst
    bone of each. A chain's first body ignores the static colliders (hair_rig) and is not checked."""
    statics = [rb for rb in bodies if rb.mode == "static"]
    dynamic = {rb.bone for rb in rig.bodies if rb.mode == "dynamic"}
    out = []
    for base, ch in rig.chains.items():
        worst = None
        for k in range(1, len(ch.names)):
            if ch.names[k] not in dynamic:
                continue
            p0, p1 = np.asarray(ch.pts[k], float), np.asarray(ch.pts[k + 1], float)
            length = float(np.linalg.norm(p1 - p0))
            r = min(float(ch.radius[k]), 0.45 * length)                  # the capsule hair_rig made for this bone
            e = (p1 - p0) / max(length, 1e-9) * 0.5 * max(length - 2 * r, 0.002)
            gap, which = CC.capsule_clearance(statics, 0.5 * (p0 + p1) - e, 0.5 * (p0 + p1) + e, r)
            if gap < 0.0 and (worst is None or gap < worst[0]):
                worst = (gap, base, ch.names[k], which)
        if worst is not None:
            out.append(worst)
    return out


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
    body = ctx.parts.get("body")
    for gap, base, bone, which in overlaps(rig, body.bodies if body is not None else []):
        table = next((t for stem, t in TABLE if base.startswith(stem)), "[hair]")
        ctx.log(f"WARNING hair chain {base.rstrip('_')}: {bone} is {-gap * 1000:.0f} mm inside {which} at rest; MMD's "
                f"physics will throw it out and mk's solver ignores that collider for it: move the chain clear ({table})")
    ordered = [m for m in mats.values() if not m.alpha_blend] + [m for m in mats.values() if m.alpha_blend]   # translucent last
    return Part("hair", meshes=meshes, materials=ordered, bones=rig.bones, bodies=rig.bodies,
                joints=rig.joints, frames=frames, info=info)
