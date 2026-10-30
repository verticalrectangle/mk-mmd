"""moves: `[[move.<cast>]]` (docs/design.md: Moves) compiled by mkmmd.core.moves into the tables the pose and perform stages
read: the hand keys into `[pose.<cast>.hands.L/R] keys` (places as `{cast, point}` targets, the goals riding the chest so
leans, tilts and sways carry them) with the `ik` keys of a hand a `free` move lets go, lean, tilt, turn and head keys,
twitches and expressions into `[perform.<cast>]`, and a gag move's (mkmmd.core.gags) legs (`bounce`, `rise`, `crouch`,
`air`, `walk`) and the spans it has the mouth (`mouth`) too. Runs at the start of the pose stage. The landmarks the places
are built from come from the member's rest pose in its root's frame, the beats and downbeats from the project's timeline
(audio/timeline.json), a gag's lanes (`hits` / `accents` = {lane, ...}) from the [audio] hits file, and the hands are kept
off the member's own body and clothes: the distance the compiler measures is to every face of its meshes in the rest pose
that belongs to neither arm, nor to a chain of hair, ears, a tail, a skirt or ribbons that the sim moves (those make way
for a hand, colliding with its fingers and palm, so long hair or a skirt at her sides does not push a hanging hand out).
Places that rest on a skirt (mkmmd.core.moves.CLOTH) measure to the skirt as well, simulated or not, so they rest on it.
The props a gag move holds are bone-parented at the end of the pose stage (`attach_held`), and the root properties it
fires (a carrot's `bite<k>`) are set here."""
import json
import math
import os

import bpy
import numpy as np
from mathutils import Matrix, Vector
from mathutils.bvhtree import BVHTree

from ...core import families as FAM
from ...core import lanes as LN
from ...core import moves as MV
from .. import keys as K
from .. import scene as S
from . import BuildError
from . import sim as SIM

MOUTH = (0.0, -0.06, 0.012)                 # the mouth from the head bone's head (an MMD head sits on the neck's top)
EYES = (0.0, -0.05, 0.07)                   # the eyes from there, for a rig without eye bones
ARM = ("arm", "arm_twist", "elbow", "wrist_twist", "wrist", "thumb", "index", "middle", "ring", "little")
YIELD = FAM.HAIR_FAMILIES | {"ears", "tail", "skirt", "ribbon"}   # chains that make way for a hand when the sim moves them


def marks(m):
    """The landmarks of cast member `m` in its own frame (mkmmd.core.moves.MARKS) from its armature's rest pose, with its
    arm at rest (`arm_rest`) and its hand (`hand_frame`)."""
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
        eyes = head("eye.L"), head("eye.R")
        out["eye"] = (eyes[0] + eyes[1]) / 2
    except (KeyError, ValueError):
        eyes, out["eye"] = None, h + Vector(EYES)
    out = {k: [round(float(c), 4) for c in v] for k, v in out.items()}
    mean = lambda f: round(sum(f(s) for s in "LR") / 2, 4)                                     # noqa: E731
    out["reach"] = mean(lambda s: (head(f"wrist.{s}") - head(f"arm.{s}")).length)
    out["upper"] = mean(lambda s: (head(f"elbow.{s}") - head(f"arm.{s}")).length)
    out["fore"] = mean(lambda s: (head(f"wrist.{s}") - head(f"elbow.{s}")).length)
    out["hand"] = mean(lambda s: (tail(f"middle3.{s}") - head(f"wrist.{s}")).length)
    if eyes is not None:
        out["eye_dx"] = round(abs(float((eyes[0] - eyes[1]).x)) / 2, 4)
        pupil = pupils(m)
        if pupil is not None:
            out["pupil"] = pupil
    rest = arm_rest(m)
    if rest is not None:
        out["arm_rest"] = rest
    hand = hand_frame(m)
    if hand is not None:
        out["fingers"] = hand
    return out


def pupils(m):
    """The landmark `pupil` of member `m` (mkmmd.core.moves.MARKS): its left pupil as it shows, in its own frame at rest,
    the right one mirrored onto it and the two averaged: the middle, by area, of the faces whose every vertex its eye
    bone moves most (an MMD eye's iris and pupil are a mesh of their own a few centimetres in front of the eye bone, and
    off its axis). None for a model whose eyes no face follows (eyes drawn on the face)."""
    arm = m.arm
    inv = np.array(m.root.matrix_world.inverted())
    names = {S.resolve_bone(arm, f"eye.{s}"): s for s in ("L", "R")}
    bones = set(arm.data.bones.keys())
    acc = {"L": [np.zeros(3), 0.0], "R": [np.zeros(3), 0.0]}
    for o in S.model_meshes(arm):
        gi = {g.index: g.name for g in o.vertex_groups}
        if not set(names) & set(gi.values()):
            continue
        top = []
        for v in o.data.vertices:
            gs = [g for g in v.groups if gi.get(g.group) in bones]
            top.append(gi[max(gs, key=lambda g: g.weight).group] if gs else None)
        M = inv @ np.array(o.matrix_world)
        for p in o.data.polygons:
            first = top[p.vertices[0]]
            if first in names and all(top[i] == first for i in p.vertices):
                a = acc[names[first]]
                a[0] += p.area * (M[:3, :3] @ np.array(p.center) + M[:3, 3])
                a[1] += p.area
    if not acc["L"][1] or not acc["R"][1]:
        return None
    left, right = (acc[s][0] / acc[s][1] for s in ("L", "R"))
    return [round(float(c), 4) for c in 0.5 * (left + right * [-1.0, 1.0, 1.0])]


def arm_rest(m):
    """The landmark `arm_rest` of member `m` (mkmmd.core.moves.arm_rest): its left arm at rest in its own frame, the
    upper arm's direction (arm head -> elbow head), the forearm's (elbow head -> wrist head), the way the wrist bone
    points and the palm's normal, with the right arm mirrored onto it and the two averaged; None without the bones."""
    from .pose import palm_normal
    arm = m.arm
    smap = S.semantic_map(arm)
    if any(f"{k}.{s}" not in smap for k in ("arm", "elbow", "wrist", "index1", "little1") for s in ("L", "R")):
        return None
    R = (m.root.matrix_world.inverted() @ arm.matrix_world).to_3x3().normalized()     # armature -> its own frame
    W = arm.matrix_world.to_3x3().normalized().inverted()                              # world -> armature
    out = {k: Vector() for k in ("upper", "fore", "dir", "palm")}
    for side, x in (("L", 1.0), ("R", -1.0)):
        head = lambda sem, side=side: arm.data.bones[smap[f"{sem}.{side}"]].head_local          # noqa: E731
        vs = {"upper": head("elbow") - head("arm"), "fore": head("wrist") - head("elbow"),
              "dir": W @ (K.rest_rot(arm, smap[f"wrist.{side}"]) @ Vector((0.0, 1.0, 0.0))),
              "palm": W @ palm_normal(arm, smap, side)}
        for k, v in vs.items():
            v = (R @ v).normalized()
            out[k] += Vector((x * v.x, v.y, v.z))
    return {k: [round(float(c), 5) for c in v.normalized()] for k, v in out.items()}


def hand_frame(m):
    """The landmark `fingers` of member `m` (mkmmd.core.moves.hand_rest): its left hand at rest in the hand's own frame (x
    the way the wrist bone points, y the palm normal square to it, z = x cross y; metres from the wrist's head), with the
    right hand mirrored onto it (its z turned over) and the two averaged: the chains of its fingers, the palm normal,
    where the skin of each finger ends and its radius (mkmmd.blender.build.pose.hand_rest). None for a member without
    hands the pose stage can curl."""
    from .pose import hand_rest
    arm = m.arm
    smap = S.semantic_map(arm)
    sides = []
    for side in ("L", "R"):
        h = hand_rest(arm, smap, side)
        if h is None or f"wrist.{side}" not in smap:
            return None
        wrist = arm.data.bones[smap[f"wrist.{side}"]]
        o = np.array(arm.matrix_world @ wrist.head_local)
        x = np.array(K.rest_rot(arm, wrist.name) @ Vector((0.0, 1.0, 0.0)))
        x = x / np.linalg.norm(x)
        y = h["palm"] - (h["palm"] @ x) * x
        y = y / np.linalg.norm(y)
        B = np.stack([x, y, np.cross(x, y)], 1) * (np.array([1.0, 1.0, -1.0]) if side == "R" else 1.0)
        sides.append({"points": {f: (np.asarray(p) - o) @ B for f, p in h["points"].items()}, "palm": h["palm"] @ B,
                      "tips": {f: (np.asarray(t) - o) @ B for f, t in h["tips"].items()}, "radii": dict(h["radii"])})

    def both(key):
        a, b = sides[0][key], sides[1][key]
        return {f: 0.5 * (np.asarray(a[f]) + np.asarray(b[f])) if f in b and np.shape(a[f]) == np.shape(b[f]) else a[f]
                for f in a}
    r5 = lambda v: np.round(np.asarray(v, float), 5).tolist()                                     # noqa: E731
    palm = sides[0]["palm"] + sides[1]["palm"]
    return {"points": {f: r5(v) for f, v in both("points").items()}, "palm": r5(palm / np.linalg.norm(palm)),
            "tips": {f: r5(v) for f, v in both("tips").items()},
            "radii": {f: round(float(v), 5) for f, v in both("radii").items()}}


def body_clearance(m, soft=()):
    """`clear(points)` for mkmmd.core.moves: the smallest distance (m) from points (n, 3) in the member's own frame to the
    faces of its meshes, in the pose they have now, whose vertices all belong to neither arm, leaving out the faces whose
    vertices all belong to `soft` bones (by each vertex's heaviest bone weight; mmd_tools' non-bone groups such as
    `mmd_edge_scale` are not bones)."""
    arm = m.arm
    smap = S.semantic_map(arm)
    arm_bones = set()
    for sem, b in smap.items():
        stem, _, side = sem.partition(".")
        if side in ("L", "R") and stem.rstrip("0123456789").replace("_tip", "") in ARM:
            arm_bones.add(b)
    bones, soft = set(arm.data.bones.keys()), set(soft)
    inv = np.array(m.root.matrix_world.inverted())
    dg = bpy.context.evaluated_depsgraph_get()
    verts, polys, off = [], [], 0
    for o in S.model_meshes(arm):
        names = [g.name for g in o.vertex_groups]
        on_arm = np.zeros(len(o.data.vertices), bool)
        yields = np.zeros(len(o.data.vertices), bool)
        for i, v in enumerate(o.data.vertices):
            gs = [g for g in v.groups if g.group < len(names) and names[g.group] in bones]
            if gs:
                top = names[max(gs, key=lambda g: g.weight).group]
                on_arm[i], yields[i] = top in arm_bones, top in soft
        oe = o.evaluated_get(dg)
        me = oe.to_mesh()
        try:
            V = np.array([v.co for v in me.vertices], float).reshape(-1, 3)
            M = inv @ np.array(o.matrix_world)
            verts.append(V @ M[:3, :3].T + M[:3, 3])
            polys += [[i + off for i in p.vertices] for p in me.polygons
                      if not on_arm[list(p.vertices)].any() and not yields[list(p.vertices)].all()]
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


def soft_bones(ctx, name, m, fams=YIELD):
    """The bones of member `name`'s loose chains the sim moves of the families `fams` (YIELD: hair, ears, a tail, a
    skirt, ribbons): they make way for a hand."""
    fams = set(SIM.families(ctx, name, m)) & set(fams)
    return {b for c in m.rig["chains"] if c["family"] in fams for b in c["bones"]} if fams else set()


def wink_sides(m):
    """{"L": the semantic wink morph that shuts member `m`'s left eye, "R": its right} read off the morphs' shape keys
    (where each moves the mesh, in the member's own frame), for mkmmd.core.moves.compile's `winks`: models name them from
    either side (some close the left eye with `ウィンク`, others the eye on the left of the picture). None when it cannot
    tell (a wink missing, not a shape key of the meshes, or both on one side)."""
    names = (m.rig or {}).get("morphs", {})
    inv = np.array(m.root.matrix_world.inverted())
    x = {}
    for sem in ("wink_l", "wink_r"):
        xs = []
        for o in S.model_meshes(m.arm):
            keys = o.data.shape_keys
            kb = keys.key_blocks.get(names.get(sem, sem)) if keys else None
            if kb is None or kb.relative_key is None or kb.relative_key == kb:
                continue
            n = len(kb.data)
            co, base = np.empty(n * 3), np.empty(n * 3)
            kb.data.foreach_get("co", co)
            kb.relative_key.data.foreach_get("co", base)
            co, base = co.reshape(-1, 3), base.reshape(-1, 3)
            moved = np.linalg.norm(co - base, axis=1) > 1e-5
            if moved.any():
                M = inv @ np.array(o.matrix_world)
                xs.append(base[moved] @ M[0, :3] + M[0, 3])            # x: the member's left
        if not xs:
            return None
        x[sem] = float(np.mean(np.concatenate(xs)))
    if x["wink_l"] * x["wink_r"] >= 0.0:
        return None
    return {"L": "wink_l", "R": "wink_r"} if x["wink_l"] > 0 else {"L": "wink_r", "R": "wink_l"}


def _beats(ctx):
    """(beats, downbeats) of the project's timeline (audio/timeline.json), clip seconds; none without one."""
    path = ctx.path("audio/timeline.json")
    if not os.path.exists(path):
        return [], []
    with open(path, encoding="utf-8") as fh:
        tl = json.load(fh)
    return [float(b) for b in tl.get("beats", [])], [float(b) for b in tl.get("downbeats", [])]


def _lanes(ctx, name, i, e):
    """The entry `e` with its `hits` and `accents` given as a lane of the [audio] hits file ({lane, min_db, every, from,
    to}: mkmmd.core.lanes; from and to default to the move's own span) read off it into clip seconds."""
    if not isinstance(e, dict) or not any(isinstance(e.get(k), dict) for k in ("hits", "accents")):
        return e
    where = f"move.{name}[{i}] ({e.get('name')!r})"
    hits_file = (ctx.data.get("audio") or {}).get("hits")
    if not hits_file:
        raise BuildError(f"{where}: a lane reads the [audio] hits file (write it with `mk timeline hits`)")
    t0 = float(e.get("t", 0.0))
    t1 = t0 + float(e.get("dur", 1.0))
    out = dict(e)
    for key in ("hits", "accents"):
        spec = e.get(key)
        if not isinstance(spec, dict):
            continue
        if set(spec) - set(LN.KEYS):
            raise BuildError(f"{where}: {key} = {{{', '.join(LN.KEYS)}}}, not {sorted(spec)}")
        try:
            c = LN.choice(spec, f"{where}.{key}")
            ts = LN.pick(LN.load(ctx.path(hits_file)), c["lane"], c["min_db"], c["every"],
                         t0 if c["t0"] is None else max(t0, float(c["t0"])), t1 if c["t1"] is None else min(t1, float(c["t1"])))
        except LN.LaneError as ex:
            raise BuildError(f"{where}.{key}: {ex}") from None
        if not ts and key == "hits":
            raise BuildError(f"{where}: lane {c['lane']!r} has no hit inside the move ({t0:g} .. {t1:g} s)")
        out[key] = ts
    return out


def _held_props(ctx, name, entries):
    """{prop: {"use", "bounds", "scale"}} of the props member `name`'s gag moves hold (`prop`), as mkmmd.core.moves.compile
    reads them; a held prop rides the member, so it is placed by no `attach`, `wear`, `anchor_to` or `place` of its own."""
    out = {}
    specs = {p.get("name"): p for p in ctx.data.get("prop", [])}
    for e in entries:
        prop = e.get("prop") if isinstance(e, dict) else None
        if not isinstance(prop, str) or prop in out:
            continue
        p = ctx.props.get(prop)
        if p is None:
            raise BuildError(f"move.{name}: prop {prop!r} is no [[prop]] (props: {', '.join(sorted(ctx.props)) or 'none'})")
        own = [k for k in ("attach", "wear", "anchor_to", "place") if (specs.get(prop) or {}).get(k)]
        if own:
            raise BuildError(f"move.{name}: prop {prop!r} is held by a move and has `{own[0]}` of its own: the move puts "
                             f"it in her hand (or on her chest)")
        out[prop] = {"use": p.card.get("use") or {}, "bounds": p.card.get("bounds"), "scale": float(p.root.scale[0])}
    return out


def hand_keys(name, keys):
    """A hand's compiled keys (mkmmd.core.moves) as the pose stage reads them: their places, and the elbow's pole of a
    place that holds the elbow its own way, are points in member `name`'s own frame, so targets of it."""
    return [dict(k, at={"cast": name, "point": k["at"]},
                 **({"pole": {"cast": name, "point": k["pole"]}} if k.get("pole") is not None else {})) for k in keys]


def apply(ctx):
    """Lay every member's compiled moves into ctx.data's pose and perform tables (a gag move's legs and mouth too), and set
    the root properties of the props they fire (a carrot's bites). Returns {cast: report}."""
    moves = ctx.data.get("move") or {}
    if not isinstance(moves, dict):
        raise BuildError("[[move.<cast>]]: expected entries named after cast members")
    out, clock = {}, None
    for name, entries in moves.items():
        m = ctx.cast.get(name)
        if m is None:
            raise BuildError(f"[[move.{name}]]: no cast member {name!r} (cast: {', '.join(sorted(ctx.cast))})")
        clock = _beats(ctx) if clock is None else clock
        beats, downbeats = clock
        yaw = math.degrees(m.root.matrix_world.to_euler().z)
        soft, rigid = soft_bones(ctx, name, m), soft_bones(ctx, name, m, YIELD - {"skirt"})
        clear = body_clearance(m, soft)
        entries = [_lanes(ctx, name, i, e) for i, e in enumerate(entries)]
        plan = {"entries": entries, "marks": marks(m), "beats": beats, "downbeats": downbeats, "yaw": yaw,
                "clear": clear, "cloth": clear if rigid == soft else body_clearance(m, rigid), "winks": wink_sides(m),
                "props": _held_props(ctx, name, entries)}
        try:
            res = MV.compile(plan["entries"], plan["marks"], beats, yaw=yaw, clear=plan["clear"], cloth=plan["cloth"],
                             winks=plan["winks"], downbeats=downbeats, props=plan["props"])
        except MV.MoveError as e:
            raise BuildError(f"move.{name}: {e}") from None
        plan["free"] = {s: [(float(e["t"]), float(e["t"]) + float(e.get("dur", 1.0))) for e in plan["entries"]
                            if isinstance(e, dict) and e.get("name") == "free" and e.get("hand", "R") == s]
                        for s in ("L", "R")}
        plan["held"] = res["held"]
        m.moves = plan                                   # pose.free_hands compiles them again with the strips' hands
        chest = "upper_body2" if "upper_body2" in S.semantic_map(m.arm) else "upper_body"
        pose = ctx.data.setdefault("pose", {}).setdefault(name, {})
        for side, keys in res["hands"].items():
            if not keys:
                continue
            hand = pose.setdefault("hands", {}).setdefault(side, {})
            if any(k in hand for k in ("at", "keys", "grip", "rest", "ik")):
                raise BuildError(f"[pose.{name}.hands.{side}] places that hand and [[move.{name}]] moves it")
            hand["keys"] = hand_keys(name, keys)
            hand.setdefault("ride", f"cast:{name}.{chest}")
            if side in res["ik"]:
                hand["ik"] = res["ik"][side]
        perf = ctx.data.setdefault("perform", {}).setdefault(name, {})
        for key in MV.BODY:
            if res[key]:
                if perf.get(key):
                    raise BuildError(f"[perform.{name}] {key} and [[move.{name}]] both key it")
                perf[key] = res[key]
        for key in ("twitch", "expressions"):
            if res[key]:
                perf[key] = list(perf.get(key) or []) + res[key]
        for key, v in res["perform"].items():            # a gag's legs (a bounce's pump and hops, a run) and its mouth
            if key in ("air", "mouth"):
                own = perf.get(key) or []
                perf[key] = (own if isinstance(own, list) else [own]) + v
            elif perf.get(key):
                raise BuildError(f"[perform.{name}] {key} and [[move.{name}]] both key it (a "
                                 f"{'run_in_place' if key == 'walk' else 'bounce'})")
            else:
                perf[key] = v
        for prop, vals in res["props"].items():          # a carrot's bites: the clip second each is bitten
            for key, v in vals.items():
                ctx.props[prop].root[key] = float(v)
        out[name] = {"moves": len(entries), "hand_keys": {s: len(k) for s, k in res["hands"].items() if k},
                     "free": {s: len(k) // 4 for s, k in res["ik"].items()},
                     "expressions": len(res["expressions"]), "twitches": len(res["twitch"]), "places": res["places"]}
        if res["perform"]:
            out[name]["perform"] = sorted(res["perform"])
        if res["held"]:
            out[name]["held"] = {p: h["bone"] for p, h in res["held"].items()}
        if res["props"]:
            out[name]["props"] = res["props"]
        ctx.log("moves", name, json.dumps({k: v for k, v in out[name].items() if k != "places"}))
    return out


def attach_held(ctx):
    """Bone-parent the props the members' gag moves hold (mkmmd.core.moves.compile's `held`): a held one to the wrist of
    its hand, placed in the fist as the moves have it in the hand's frame (its direction the wrist bone's, its palm the
    rest palm's normal square to it: how the pose stage turns the wrist to a goal), a hugged one to the chest bone as it
    stands in the settled pose (frame start + settle), where the hands' goals that ride the chest are read too. Returns
    {prop: "<cast>:<bone>"}."""
    from . import pose as POSE
    done = {}
    sc = bpy.context.scene
    for name, m in ctx.cast.items():
        plan = m.moves
        if not plan or not plan.get("held"):
            continue
        arm, smap = m.arm, S.semantic_map(m.arm)
        for prop, h in plan["held"].items():
            p = ctx.props[prop]
            k = Matrix.Scale(float(plan["props"][prop]["scale"]), 4)
            if h["bone"] == "wrist":
                bone = smap[f"wrist.{h['side']}"]
                W0 = arm.matrix_world @ arm.data.bones[bone].matrix_local          # the wrist's head frame at rest
                y = (W0.to_3x3() @ Vector((0.0, 1.0, 0.0))).normalized()
                n = POSE.palm_normal(arm, smap, h["side"])
                n = (n - n.dot(y) * y).normalized()
                H = Matrix((y, n, y.cross(n))).transposed()                     # the hand's frame at rest, as columns
                R, T = Matrix(h["rel"][0]), Vector(h["rel"][1])
                M = Matrix.Translation(W0.translation + H @ T) @ (H @ R).to_4x4() @ k
                rel = W0.inverted() @ M
            else:
                bone = "upper_body2" if "upper_body2" in smap else "upper_body"
                bone = smap[bone]
                keep = sc.frame_current
                sc.frame_set(ctx.start + ctx.settle)
                dg = bpy.context.evaluated_depsgraph_get()
                head = arm.matrix_world @ arm.evaluated_get(dg).pose.bones[bone].matrix
                R, origin = Matrix(h["pose"][0]), Vector(h["pose"][1])
                rel = head.inverted() @ m.root.matrix_world @ Matrix.Translation(origin) @ R.to_4x4() @ k
                sc.frame_set(keep)
            POSE.attach_to_bone(p.root, arm, bone, rel)
            done[prop] = f"{name}:{bone}"
    return done
