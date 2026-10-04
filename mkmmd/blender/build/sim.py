"""sim: secondary motion (hair, ears, tails, skirts, ribbons) solved by mkmmd.solvers.strands and baked to keys.

[sim.<cast>] keys
  families = [...]            chain families to simulate (default: the hair families, ears and tails the rig has)
  params = {family: {...}}    solver parameters per family: sag = [root, tip] deg, drag, zeta, radius, radius_max,
                              friction, wind_drag (docs: mkmmd/solvers/strands.py)
  colliders = "set" | [...]   scene collider specs or a [colliders] set (docs/design.md: Colliders)
  props = true                add every prop card's colliders (seats, doors, dash, wheel...)
  fingers = true              finger and palm capsules of every cast member
  floor = 0.0                 ground height (false: no floor)
  wind = {direction, speed, exposure, gust, gust_period, turbulence, scale, seed, carrier = "car"}: with a carrier,
         the vehicle's velocity enters the air (exposure: the share of its motion the air does not follow)
  use_masks = false, anchor_free = 0.25, substeps = 10, settle_s = 1.5, engine = "auto"
The whole frame range is simulated (the pre-roll lets the hair settle while the character eases into its pose).

Branching chains (a strand that splits) are decomposed: the trunk follows the deepest subtree, every other subtree
becomes a chain anchored on the bone it grows from. Trunks and plain chains are solved first and keyed; then the
branches are solved riding the keyed trunks, level by level. The bodies of every simulated bone are left out of
the colliders (they move with the hair). Chain bones get LINEAR rotation keys on every frame."""
import json

import bpy
import numpy as np

from ...core import families as FAM
from ...solvers import geom
from .. import keys as K
from ..ops_sample import resolve_collider
from . import BuildError

DEFAULT_FAMILIES = sorted(FAM.HAIR_FAMILIES | {"ears", "tail"})


def _mat(m):
    return np.array([list(r) for r in m], float)


def _quats(R):
    """(F, S, 3, 3) rotations -> (F, S, 4) unit w x y z quaternions, sign-continuous along F."""
    F, S = R.shape[:2]
    q = geom.mat_to_quat(R.reshape(-1, 3, 3)).reshape(F, S, 4)
    for k in range(1, F):
        flip = np.einsum("sj,sj->s", q[k], q[k - 1]) < 0
        q[k, flip] *= -1.0
    return q


def sample_sources(sources, frames):
    """Posed world frames (unscaled) of shape sources on every frame: positions (F, S, 3), rotations (F, S, 3, 3)."""
    sc = bpy.context.scene
    F, S = len(frames), len(sources)
    pos, rot = np.zeros((F, S, 3)), np.tile(np.eye(3), (F, S, 1, 1))
    arms = {o.name: o for o in bpy.data.objects if o.type == "ARMATURE"}
    for i, f in enumerate(frames):
        sc.frame_set(int(f))
        dg = bpy.context.evaluated_depsgraph_get()
        evald = {}
        for j, (kind, owner, name) in enumerate(sources):
            if kind == "world":
                continue
            if kind == "bone":
                if owner not in evald:
                    evald[owner] = arms[owner].evaluated_get(dg)
                ae = evald[owner]
                M = geom.unscaled(_mat(ae.matrix_world @ ae.pose.bones[name].matrix))
            else:
                M = geom.unscaled(_mat(bpy.data.objects[name].evaluated_get(dg).matrix_world))
            pos[i, j], rot[i, j] = M[:3, 3], M[:3, :3]
    return pos, rot


def collider_items(ctx, spec, arm):
    specs = []
    cs = spec.get("colliders")
    if isinstance(cs, str):
        sets = ctx.data.get("colliders", {})
        if cs not in sets:
            raise BuildError(f"no collider set {cs!r}")
        specs += list(sets[cs])
    elif cs:
        specs += list(cs)
    if spec.get("props", True):
        for p in ctx.props.values():
            specs += p.colliders
    if spec.get("fingers", True):
        specs += [{"type": "fingers", "armature": c.arm.name, "tag": f"hand:{c.name}"} for c in ctx.cast.values()]
    floor = spec.get("floor", 0.0)
    if floor is not False:
        specs.append({"type": "floor", "z": float(floor)})
    return [it for s in specs for it in resolve_collider(s, arm.name)]


def solve_pass(ctx, m, spec, rig_pass, fams, items, tag):
    """Simulate the chains of rig_pass, key them, return (bones, penetration (F, 3), report)."""
    frames = ctx.frames
    arm = m.arm
    chains = geom.Chains(rig_pass, fams)
    shapes = geom.Shapes()
    for a in sorted(set(chains.anchor)):
        shapes.src("bone", arm.name, a)
    geom.model_shapes(shapes, rig_pass, arm.name, skip_bones=chains.chain_bones)
    for it in items:
        shapes.add_item(dict(it))
    sources = [list(s) for s in shapes.sources]
    bpy.context.scene.frame_set(int(frames[0]))
    A0 = geom.unscaled(_mat(arm.evaluated_get(bpy.context.evaluated_depsgraph_get()).matrix_world))
    B = arm.data.bones
    rest_R, rest_p = np.tile(np.eye(3), (len(sources), 1, 1)), np.zeros((len(sources), 3))
    for j, (kind, owner, bone) in enumerate(sources):
        if kind == "bone":
            M = geom.unscaled(A0 @ _mat(bpy.data.objects[owner].data.bones[bone].matrix_local))
            rest_R[j], rest_p[j] = M[:3, :3], M[:3, 3]
    pos, rot = sample_sources([tuple(s) for s in sources], frames)
    arrays = {"head": geom.apply(A0, chains.head), "end": geom.apply(A0, chains.end),
              "bone_rest_R": np.array([geom.unscaled(A0 @ _mat(B[b].matrix_local))[:3, :3] for b in chains.bones]),
              "src_pos": pos, "src_quat": _quats(rot), "rest_R": rest_R, "rest_p": rest_p}
    wind = dict(spec.get("wind") or {})
    carrier = wind.pop("carrier", None)
    if carrier:
        veh = getattr(ctx, "vehicles", {}).get(carrier)
        if veh is None:
            raise BuildError(f"sim.{m.name}: wind carrier {carrier!r} is not a vehicle")
        arrays["carrier_vel"] = np.gradient(np.asarray(veh["pos"], float), axis=0) * ctx.fps
    solver_spec = {"rig": rig_pass, "families": fams, "armature": arm.name,
                   "frames": [int(frames[0]), int(frames[-1])], "fps": ctx.fps,
                   "substeps": int(spec.get("substeps", 10)), "settle_s": float(spec.get("settle_s", 1.5)),
                   "sources": sources, "model_bodies": True, "colliders": items,
                   "params": spec.get("params") or {}, "wind": wind or None,
                   "use_masks": bool(spec.get("use_masks", False)),
                   "anchor_free": float(spec.get("anchor_free", geom.ANCHOR_FREE)),
                   "window": [ctx.frame0, int(frames[-1])], "engine": spec.get("engine", "auto")}
    res, rep = ctx.solve("mkmmd.solvers.strands", arrays, solver_spec, tag=tag)
    for i, b in enumerate(chains.bones):
        K.key_bone_quats(arm, b, frames, res["quats"][:, i], interp="LINEAR")
    return chains.bones, res["pen"], rep


def run(ctx):
    out = {}
    for name, m in ctx.cast.items():
        spec = ctx.section("sim", name)
        if not spec:
            continue
        if m.rig is None:
            raise BuildError(f"sim.{name}: the cast member has no rig.json (register it with mk assets add)")
        rig = m.rig
        have = {c["family"] for c in rig["chains"]}
        fams = [f for f in (spec.get("families") or DEFAULT_FAMILIES) if f in have]
        if not fams:
            ctx.log(f"sim {name}: no chains in families {spec.get('families') or DEFAULT_FAMILIES}")
            continue
        levels = {}
        for lv, piece in geom.split_chains(rig, fams):
            levels.setdefault(lv, []).append(piece)
        simulated = {b for pieces in levels.values() for p in pieces for b in p["bones"]}
        bodies = [b for b in rig["bodies"] if b.get("bone") not in simulated]
        items = collider_items(ctx, spec, m.arm)
        info = {"families": fams, "colliders": len(items), "passes": [], "bones": 0}
        pen_max = 0.0
        k0 = ctx.frame0 - int(ctx.frames[0])
        for lv in sorted(levels):
            rig_pass = dict(rig, chains=levels[lv], bodies=bodies)
            bones, pen, rep = solve_pass(ctx, m, spec, rig_pass, fams, items, tag=f"{name}-L{lv}")
            pen_max = max(pen_max, float(pen[k0:, 0].max()))
            info["passes"].append({"level": lv, "chains": len(levels[lv]), "bones": len(bones),
                                   "engine": rep.get("engine"), "seconds": (rep.get("seconds") or {}).get("total"),
                                   "report": rep.get("report")})
            info["bones"] += len(bones)
        info["penetration_mm_max"] = round(pen_max * 1000, 2)
        out[name] = info
        ctx.log("sim", name, json.dumps({k: v for k, v in info.items() if k != "passes"}, ensure_ascii=False),
                [(p["level"], p["chains"], p["bones"], p["seconds"]) for p in info["passes"]])
    return out
