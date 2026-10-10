"""moves: `[[move.<cast>]]` (docs/design.md: Moves) compiled by mkmmd.core.moves into the tables the pose and perform stages
read: the hand keys into `[pose.<cast>.hands.L/R] keys` (places as `{cast, point}` targets, the goals riding the chest so
leans, tilts and sways carry them), lean and tilt keys, twitches and expressions into `[perform.<cast>]`. Runs at the start
of the pose stage. The landmarks the places are built from come from the member's rest pose in its root's frame, the beats
from the project's timeline (audio/timeline.json), and the hands are kept off the member's own body, clothes and hair:
the distance the compiler measures is to every face of its meshes in the rest pose that belongs to neither arm."""
import json
import math
import os

import bpy
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree

from ...core import moves as MV
from .. import scene as S
from . import BuildError

MOUTH = (0.0, -0.06, 0.012)                 # the mouth from the head bone's head (an MMD head sits on the neck's top)
EYES = (0.0, -0.05, 0.07)                   # the eyes from there, for a rig without eye bones
ARM = ("arm", "arm_twist", "elbow", "wrist_twist", "wrist", "thumb", "index", "middle", "ring", "little")


def marks(m):
    """The landmarks of cast member `m` in its own frame (mkmmd.core.moves.MARKS) from its armature's rest pose."""
    inv = m.root.matrix_world.inverted()
    arm = m.arm

    def head(sem):
        b = S.resolve_bone(arm, sem)
        return inv @ (arm.matrix_world @ arm.data.bones[b].head_local)

    def tail(sem):
        b = S.resolve_bone(arm, sem)
        return inv @ (arm.matrix_world @ arm.data.bones[b].tail_local)

    out = {"arm.L": head("arm.L"), "arm.R": head("arm.R"), "chest": head("upper_body2")}
    h = head("head")
    out["mouth"] = h + Vector(MOUTH)
    try:
        out["eye"] = (head("eye.L") + head("eye.R")) / 2
    except (KeyError, ValueError):
        out["eye"] = h + Vector(EYES)
    out = {k: [round(float(c), 4) for c in v] for k, v in out.items()}
    mean = lambda f: round(sum(f(s) for s in "LR") / 2, 4)                                     # noqa: E731
    out["reach"] = mean(lambda s: (head(f"wrist.{s}") - head(f"arm.{s}")).length)
    out["upper"] = mean(lambda s: (head(f"elbow.{s}") - head(f"arm.{s}")).length)
    out["fore"] = mean(lambda s: (head(f"wrist.{s}") - head(f"elbow.{s}")).length)
    out["hand"] = mean(lambda s: (tail(f"middle3.{s}") - head(f"wrist.{s}")).length)
    return out


def body_clearance(m):
    """`clear(points)` for mkmmd.core.moves: the smallest distance (m) from points (n, 3) in the member's own frame to the
    faces of its meshes, in the pose they have now, whose vertices all belong to neither arm (by each vertex's heaviest
    bone weight; mmd_tools' non-bone groups such as `mmd_edge_scale` are not bones)."""
    arm = m.arm
    smap = S.semantic_map(arm)
    arm_bones = set()
    for sem, b in smap.items():
        stem, _, side = sem.partition(".")
        if side in ("L", "R") and stem.rstrip("0123456789").replace("_tip", "") in ARM:
            arm_bones.add(b)
    bones = set(arm.data.bones.keys())
    inv = np.array(m.root.matrix_world.inverted())
    dg = bpy.context.evaluated_depsgraph_get()
    verts, polys, off = [], [], 0
    for o in S.model_meshes(arm):
        names = [g.name for g in o.vertex_groups]
        on_arm = np.zeros(len(o.data.vertices), bool)
        for i, v in enumerate(o.data.vertices):
            gs = [g for g in v.groups if g.group < len(names) and names[g.group] in bones]
            if gs and names[max(gs, key=lambda g: g.weight).group] in arm_bones:
                on_arm[i] = True
        oe = o.evaluated_get(dg)
        me = oe.to_mesh()
        try:
            V = np.array([v.co for v in me.vertices], float).reshape(-1, 3)
            M = inv @ np.array(o.matrix_world)
            verts.append(V @ M[:3, :3].T + M[:3, 3])
            polys += [[i + off for i in p.vertices] for p in me.polygons if not on_arm[list(p.vertices)].any()]
            off += len(V)
        finally:
            oe.to_mesh_clear()
    if not polys:
        return None
    tree = BVHTree.FromPolygons([Vector(v) for v in np.concatenate(verts)], polys)

    def clear(points):
        best = math.inf
        for p in np.asarray(points, float).reshape(-1, 3):
            hit = tree.find_nearest(Vector(p))
            if hit[0] is not None:
                best = min(best, hit[3])
        return best
    return clear


def _beats(ctx):
    path = ctx.path("audio/timeline.json")
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as fh:
        return [float(b) for b in json.load(fh).get("beats", [])]


def apply(ctx):
    """Lay every member's compiled moves into ctx.data's pose and perform tables. Returns {cast: report}."""
    moves = ctx.data.get("move") or {}
    if not isinstance(moves, dict):
        raise BuildError("[[move.<cast>]]: expected entries named after cast members")
    out, beats = {}, None
    for name, entries in moves.items():
        m = ctx.cast.get(name)
        if m is None:
            raise BuildError(f"[[move.{name}]]: no cast member {name!r} (cast: {', '.join(sorted(ctx.cast))})")
        beats = _beats(ctx) if beats is None else beats
        yaw = math.degrees(m.root.matrix_world.to_euler().z)
        try:
            res = MV.compile(list(entries), marks(m), beats, yaw=yaw, clear=body_clearance(m))
        except MV.MoveError as e:
            raise BuildError(f"move.{name}: {e}") from None
        chest = "upper_body2" if "upper_body2" in S.semantic_map(m.arm) else "upper_body"
        pose = ctx.data.setdefault("pose", {}).setdefault(name, {})
        for side, keys in res["hands"].items():
            if not keys:
                continue
            hand = pose.setdefault("hands", {}).setdefault(side, {})
            if any(k in hand for k in ("at", "keys", "grip", "rest")):
                raise BuildError(f"[pose.{name}.hands.{side}] places that hand and [[move.{name}]] moves it")
            hand["keys"] = [dict(k, at={"cast": name, "point": k["at"]}) for k in keys]
            hand.setdefault("ride", f"cast:{name}.{chest}")
        perf = ctx.data.setdefault("perform", {}).setdefault(name, {})
        for key in ("lean", "tilt"):
            if res[key]:
                if perf.get(key):
                    raise BuildError(f"[perform.{name}] {key} and [[move.{name}]] both key it")
                perf[key] = res[key]
        for key in ("twitch", "expressions"):
            if res[key]:
                perf[key] = list(perf.get(key) or []) + res[key]
        out[name] = {"moves": len(entries), "hand_keys": {s: len(k) for s, k in res["hands"].items() if k},
                     "expressions": len(res["expressions"]), "twitches": len(res["twitch"]), "places": res["places"]}
        ctx.log("moves", name, json.dumps({k: v for k, v in out[name].items() if k != "places"}))
    return out
