"""marionette: a cast member on the threads of a `library:marionette` prop (docs/design.md: Marionettes;
mkmmd.core.marionette has the maths).

The perform stage calls `prepare` before it samples the gaze (so she can look up at the cross) and `hands` after the
strum: `prepare` reads her settled pose, puts the cross over her head and the leg bar in front of her and keys them,
ties the threads' ends to her bones (the `<prop>_end_<part>` empties, bone-parented where the thread meets her) and
returns her channels for perform to add to its own; `hands` turns the worn guitar with the hand bar, swings a hand whose
thread is cut on its arm, and keys the hand bar to follow the picking hand (the thread does the strumming).

The build's `marionette` stage (`run`, after `stepped`) draws the threads from where the anchors on the bars and the
ends on her are on every frame, once everything is keyed: so a stepped member's threads step with her, and the bars are
held on the same steps. Each thread's own length is its span at `from`: straight while it is pulled, sagging when she
comes closer, dangling under the cross as it falls in, parted at a cut (the piece on the cross yanked up out of sight,
the piece on her falling limp over her and off onto the floor)."""
import json
import math

import bpy
import numpy as np
from mathutils import Euler, Matrix, Quaternion, Vector

from ...core import armtwist as AT
from ...core import marionette as MR
from ...core import marionette_world as MW
from ...core import perform as PF
from ...core import stepped as STP
from ...core import timeline as TL
from .. import keys as K
from .. import scope as SCOPE
from .. import scene as S
from . import BuildError, targets
from . import stepped as STEPPED

ENDS = {"head": ("head", (0.0, 0.0, 0.175)), "back": ("upper_body2", None), "wrist_L": ("wrist.L", (0.0, 0.0, 0.02)),
        "wrist_R": ("wrist.R", (0.0, 0.0, 0.02)), "knee_L": ("knee.L", (0.0, -0.045, 0.0)),
        "knee_R": ("knee.R", (0.0, -0.045, 0.0))}       # where each thread meets her: a bone and an offset (her frame)
BACK = (0.0, 0.075, -0.045)     # the back's end: from the middle of her shoulders, behind and a little lower
SIDE = {"head": 1.0, "back": 1.0, "wrist_L": 1.0, "wrist_R": -1.0, "knee_L": 1.0, "knee_R": -1.0}


def _frame(m):
    """Her frame: the cast root's place and turn (x her left, y behind her, z up), no scale."""
    loc, rot, _ = m.root.matrix_world.decompose()
    return Matrix.Translation(loc) @ rot.to_matrix().to_4x4(), rot


def _spec(ctx, name):
    v = ctx.section("perform", name).get("marionette")
    return v if isinstance(v, dict) and v else None


def plan(ctx, name):
    """(P, prop, card numbers) of a member's marionette table; None without one."""
    spec = _spec(ctx, name)
    if spec is None:
        return None
    where = f"perform.{name}.marionette"
    beats = []
    try:
        beats = TL.beats(ctx.timeline(spec, where))[0]
    except BuildError:
        if any(isinstance(spec.get(k), dict) for k in ("rock", "tilt", "turn", "sway", "knees")):
            raise
    try:
        P = MR.normalize(spec, beats)
    except MR.MarionetteError as e:
        raise BuildError(f"perform.{name}.{e}" if str(e).startswith("marionette") else f"{where}: {e}") from None
    prop = ctx.props.get(P["prop"])
    if prop is None or "marionette" not in prop.card:
        raise BuildError(f"{where}: prop = {P['prop']!r} is no library:marionette prop (props: {sorted(ctx.props)})")
    return P, prop, prop.card["marionette"]


def _key_hidden(ob, frames, hidden):
    K.set_fcurve(ob, "hide_render", 0, frames, np.asarray(hidden, float), interp="CONSTANT")


def prepare(ctx, name, m, ts, frames):
    """Key the cross and the leg bar over her, tie the threads to her and return her channels (mkmmd.core.marionette.
    body_plan, her frame = the armature's axes) with what `hands` needs; None without a `marionette` table, and for a
    member with no control (`tails` alone: the threads are tied to her, nothing of her is keyed)."""
    got = plan(ctx, name)
    if got is None:
        return None
    P, prop, card = got
    where = f"perform.{name}.marionette"
    sc = bpy.context.scene
    keep = sc.frame_current
    sc.frame_set(int(ctx.start + ctx.settle))
    arm = m.arm
    smap = S.semantic_map(arm)
    F, Frot = _frame(m)
    Fi = F.inverted()
    ends = _ends(m, smap, Fi, where)
    if not P["control"]:                                        # six tails, no control: tied on, drawn after stepped
        _tie(prop, arm, ends, F)
        sc.frame_set(keep)
        return None

    def her(b):
        return Fi @ (arm.matrix_world @ arm.pose.bones[smap[b]].head)

    a_head = Vector(card["anchors"]["head"]["point"])
    top = ends["head"][1]
    home = Vector((top.x - a_head.x, top.y - a_head.y, top.z + P["height"] - a_head.z))
    leg_home = home + Vector(card["leg_bar"]["home"])
    legs = {}
    for s in ("L", "R"):
        hip, knee, ankle = (arm.matrix_world @ arm.data.bones[smap[f"{b}.{s}"]].head_local for b in ("leg", "knee", "ankle"))
        toe = smap.get(f"toe.{s}") or smap.get(f"toe_ik.{s}")
        foot = (arm.matrix_world @ arm.data.bones[toe].tail_local - ankle).length if toe else 0.14
        legs[s] = {"thigh": (knee - hip).length, "shin": (ankle - knee).length, "foot": max(foot, 0.08)}
    cross = MR.cross_plan(ts, P, float(card["leg_bar"]["half"]))
    if P["world"]["on"]:                                        # she lies, flies, falls: her root moves (world_plan)
        _tie(prop, arm, ends, F)
        world = _prepare_world(ctx, name, m, P, prop, card, ts, frames, cross, legs, ends, F, Frot, smap, where)
        sc.frame_set(keep)
        body = world
        body["report"]["home"] = [round(float(v), 3) for v in world["cross"]["pos"][0]]
        return {"P": P, "prop": prop, "card": card, "body": body, "cross": cross, "F": F, "Frot": Frot,
                "shoulders": {s: her(f"arm.{s}") for s in ("L", "R")}, "world": world}
    body = MR.body_plan(ts, P, cross, legs)
    # the bars, keyed in the prop root's frame
    bars = {k: bpy.data.objects[v] for k, v in card["bars"].items()}
    Ri = prop.root.matrix_world.inverted()
    Rrot_i = prop.root.matrix_world.to_quaternion().inverted()
    locs, rots, llocs, lrots = [], [], [], []
    for i in range(len(ts)):
        q = Quaternion((0, 0, 1), float(cross["turn"][i])) @ Quaternion((0, 1, 0), float(cross["tilt"][i])) \
            @ Quaternion((1, 0, 0), float(cross["rock"][i]))
        locs.append(tuple(Ri @ (F @ (home + Vector(cross["pos"][i])))))
        rots.append(tuple(Rrot_i @ Frot @ q))
        ql = Quaternion((0, 0, 1), 0.5 * float(cross["turn"][i])) @ Quaternion((0, 1, 0), -float(cross["leg_roll"][i]))
        llocs.append(tuple(Ri @ (F @ (leg_home + Vector(cross["leg_pos"][i])))))
        lrots.append(tuple(Rrot_i @ Frot @ ql))
    for ob, L, Q in ((bars["cross"], locs, rots), (bars["leg"], llocs, lrots)):
        ob.rotation_mode = "QUATERNION"
        K.key_vec(ob, "location", frames, L, interp="LINEAR")
        K.key_vec(ob, "rotation_quaternion", frames, K.continuous(Q), interp="LINEAR")
    for ob in bars.values():
        _key_hidden(ob, frames, ~cross["shown"])
    _tie(prop, arm, ends, F)
    sc.frame_set(keep)
    body["report"]["home"] = [round(v, 3) for v in home]
    return {"P": P, "prop": prop, "card": card, "body": body, "cross": cross, "F": F, "Frot": Frot,
            "shoulders": {s: her(f"arm.{s}") for s in ("L", "R")}}


def _ends(m, smap, Fi, where):
    """{part: (bone, point in her frame)}: where each thread meets her at the current (settled) pose."""
    arm = m.arm
    need = [b for b, _ in ENDS.values()] + ["arm.L", "arm.R", "leg.L", "leg.R", "ankle.L", "ankle.R"]
    missing = [b for b in need if b not in smap]
    if missing:
        raise BuildError(f"{where}: the model has no {', '.join(missing)} bone (the threads are tied there)")

    def her(b):
        return Fi @ (arm.matrix_world @ arm.pose.bones[smap[b]].head)

    ends = {}
    for part, (bone, off) in ENDS.items():
        p = (her("arm.L") + her("arm.R")) / 2 + Vector(BACK) if off is None else her(bone) + Vector(off)
        ends[part] = (smap[bone], p)
    return ends


def _tie(prop, arm, ends, F):
    """The threads' ends, tied to her bones where they meet her at the settled pose."""
    for part, (bone, p) in ends.items():
        e = bpy.data.objects[f"{prop.name}_end_{part}"]
        e.parent = arm
        e.parent_type = "BONE"
        e.parent_bone = bone
        e.matrix_parent_inverse = Matrix.Identity(4)
        bpy.context.view_layer.update()
        e.matrix_world = Matrix.Translation(F @ p)


# ---- in the world: she lies, floats, is flown, falls, lands (mkmmd.core.marionette_world)


def _rest_frame(m):
    """Her rest frame (the cast's `at` and `yaw`, under its parent), no scale: what `_frame` reads before her root is
    keyed."""
    root = m.root
    Pw = root.parent.matrix_world @ root.matrix_parent_inverse if root.parent else Matrix.Identity(4)
    loc, rot, _ = (Pw @ Matrix.Translation(Vector(m.spec.get("at", (0.0, 0.0, 0.0))))
                   @ Euler((0.0, 0.0, math.radians(float(m.spec.get("yaw", 0.0))))).to_matrix().to_4x4()).decompose()
    return Matrix.Translation(loc) @ rot.to_matrix().to_4x4(), rot


def _outline(m, smap, Fi):
    """(k, 2) her outline seen from her side at the settled pose, y and z of her frame (mkmmd.core.marionette_world.
    lowest): the skin her head, torso and legs carry (not her arms, nor the sim's chains: hair, ears, skirt), the back
    of her head a hair's thickness further out (she lies on her hair)."""
    arm = m.arm
    skip = {b for c in (m.rig or {}).get("chains", []) for b in c["bones"]}
    for s in ("L", "R"):
        b = arm.data.bones.get(smap.get(f"arm.{s}", ""))
        if b is not None:
            skip |= {b.name} | {c.name for c in b.children_recursive}
    head = smap.get("head")
    head_set = {head} | ({c.name for c in arm.data.bones[head].children_recursive} - skip) if head else set()
    dg = bpy.context.evaluated_depsgraph_get()
    pts = []
    for ob in m.meshes:
        names = {g.index: g.name for g in ob.vertex_groups if g.name in arm.data.bones}   # mmd_tools adds groups that
        verts = ob.data.vertices                                                           # are no bone, weighted 1
        best = np.full(len(verts), -1)
        for v in verts:
            gs = [g for g in v.groups if g.group in names]
            if gs:
                best[v.index] = max(gs, key=lambda g: g.weight).group
        ev = ob.evaluated_get(dg)
        me = ev.to_mesh()
        co = np.empty(3 * len(me.vertices))
        me.vertices.foreach_get("co", co)
        ev.to_mesh_clear()
        if len(co) != 3 * len(verts):
            continue
        M = np.array(Fi @ ob.matrix_world)
        X = co.reshape(-1, 3) @ M[:3, :3].T + M[:3, 3]
        bone = np.array([names.get(int(g), "") for g in best])
        keep = np.array([b != "" and b not in skip and b in arm.data.bones for b in bone])
        hair = np.array([b in head_set for b in bone]) * MW_SKULL_HAIR
        X[:, 1] += hair
        pts.append(X[keep][:, 1:3])
    if not pts or not sum(len(p) for p in pts):
        raise BuildError(f"perform.{m.name}.marionette: no skin of her head, body and legs to lie on (meshes: "
                         f"{[o.name for o in m.meshes]})")
    return MW.hull2d(np.concatenate(pts))


MW_SKULL_HAIR = 0.012       # m of hair under the back of her head when she lies on it


def _rides(m, ref_objs):
    """The empty a flown prop rides on her (its `[[flight]]` pivot, parented to her bones or her root), or None."""
    for ob in ref_objs:
        for c in ob.constraints:
            pv = getattr(c, "target", None)
            if pv is not None and pv.parent in (m.arm, m.root):
                return pv
    return None


def _aims(ctx, m, P, frames, Fi):
    """{"reach": [(n, 3) or {"on", "rel"}], "land": (n, 3), "seat", "scissors"}: the targets of `reach` and `land` in her
    rest frame on every frame (a world point once). A target flown on her (a `[[flight]]` hung on her head) goes where
    she goes, so it is given as where it hangs on her (`on`, her rest frame) and its offset from there (`rel`). A
    marionette set free: her seat (`_seat_aim`) and where the scissors' grip lies (`_scissors_aim`)."""
    W = P["world"]
    refs = [("reach", r["at"]) for r in W["reach"]] + ([("land", W["land"]["at"])] if W["land"] else [])
    out = {"reach": []}
    if W.get("free"):
        out["seat"] = _seat_aim(ctx, W["seat"]["at"], Fi)
        out["scissors"] = _scissors_aim(ctx, W["scissors"]["prop"], Fi) if W.get("scissors") else None
    sc = bpy.context.scene
    keep = sc.frame_current
    for kind, ref in refs:
        if isinstance(ref, (list, tuple)) and len(ref) == 3:
            pts = np.tile(np.asarray(Fi @ Vector(ref)), (len(frames), 1))
        else:
            objs = targets.objects(ctx, ref)
            pv = _rides(m, objs) if kind == "reach" else None
            pts, on = np.empty((len(frames), 3)), np.empty((len(frames), 3))
            with SCOPE.only(objs + ([pv] if pv is not None else [])):
                for i, f in enumerate(frames):
                    sc.frame_set(int(f))
                    pts[i] = np.asarray(Fi @ targets.point(ctx, ref))
                    if pv is not None:
                        on[i] = np.asarray(Fi @ pv.matrix_world.translation)
            if pv is not None:
                pts = {"on": on, "rel": pts - on}
        if kind == "reach":
            out["reach"].append(pts)
        else:
            out["land"] = pts
    sc.frame_set(keep)
    return out


def _seat_aim(ctx, ref, Fi):
    """{hip, facing, floor} of a seat `"<prop>:<use>"` (a `use.sit` point: its hip point, the way it faces, its floor) in
    her rest frame."""
    pname, _, use = str(ref).partition(":")
    p = ctx.props.get(pname)
    if p is None:
        raise BuildError(f"marionette.seat = {ref!r}: no prop {pname!r} (props: {sorted(ctx.props)})")
    e = p.use("sit", use or None)
    hip = p.world(e["hip"])
    facing = p.world_dir(e.get("facing", (0.0, -1.0, 0.0)))
    floor = p.world((0.0, 0.0, float(e.get("floor_z", 0.0)))).z
    return {"hip": np.asarray(Fi @ hip), "facing": np.asarray(Fi.to_3x3() @ facing),
            "floor": float((Fi @ Vector((hip.x, hip.y, floor))).z)}


def _scissors_aim(ctx, pname, Fi):
    """{grip}: where the hand takes the scissors as they lie (their `use.grip` point, else their root), her rest frame."""
    p = ctx.props.get(pname)
    if p is None:
        raise BuildError(f"marionette.scissors.prop = {pname!r}: no such prop (props: {sorted(ctx.props)})")
    grips = p.card.get("use", {}).get("grip", [])
    g = next((u for u in grips if u.get("name") == "hold"), grips[0] if grips else None)
    at = p.world(g["point"]) if g and "point" in g else p.root.matrix_world.translation
    return {"grip": np.asarray(Fi @ at)}


def _prepare_world(ctx, name, m, P, prop, card, ts, frames, cross, legs, ends, F, Frot, smap, where):
    """The world plan (mkmmd.core.marionette_world.world_plan): her measures, her root keyed (a turn about her hips and
    a move), the bars, the surface she rests on and where a yank takes her thread. Returns the plan."""
    root, arm = m.root, m.arm
    if root is None or root == arm:
        raise BuildError(f"{where}: she lies, flies and falls by her root, and the member has none to move")
    if root.rotation_mode not in ("XYZ", "XZY", "YXZ", "YZX", "ZXY", "ZYX"):
        raise BuildError(f"{where}: the root {root.name!r} turns in {root.rotation_mode}, not Euler angles")
    spec = ctx.section("perform", name)
    if spec.get("walk") or spec.get("tumble") or spec.get("step_turn") or spec.get("spin"):
        raise BuildError(f"{where}: lie, float, release, land, yank and knot move her root, as walk, tumble, step_turn "
                         f"and spin do: give her one of them (a fall's tumble is `release.tumble`)")
    pose = ctx.section("pose", name)
    if pose.get("sit") or pose.get("feet") or pose.get("hips"):
        raise BuildError(f"{where}: a marionette that lies or flies stands where her threads put her: drop `sit`, "
                         f"`feet` and `hips` from [pose.{name}]")
    for k in ctx.data.get("key", []):
        if k.get("target") == root.name and k.get("prop") in ("location", "rotation_euler") and not k.get("relative"):
            raise BuildError(f"{where}: a [[key]] keys {root.name!r} {k['prop']} outright, which the marionette keys: "
                             f"give it `relative = true`")
    Fi = F.inverted()

    def her(b, tail=False):
        pb = arm.pose.bones[smap[b]]
        return np.asarray(Fi @ (arm.matrix_world @ (pb.tail if tail else pb.head)))

    def rest(b):
        return np.asarray(Fi.to_3x3() @ (arm.matrix_world.to_3x3() @ arm.data.bones[smap[b]].head_local))

    need = ("center", "elbow.L", "elbow.R", "wrist.L", "wrist.R", "index1.L", "little1.L", "middle1.L")
    for b in need + (("shoulder.L", "shoulder.R") if P["world"].get("free") else ()):
        if b not in smap:
            raise BuildError(f"{where}: the model has no {b} bone (her body and her hands move on it)")
    rig = {"pivot": her("center"), "crown": np.asarray(ends["head"][1]), "hang": np.asarray(ends["back"][1]),
           "hull": _outline(m, smap, Fi),
           "shoulders": {s: her(f"arm.{s}") for s in ("L", "R")}, "hips": {s: her(f"leg.{s}") for s in ("L", "R")},
           "ankles": {s: her(f"leg_ik.{s}") if f"leg_ik.{s}" in smap else her(f"ankle.{s}") for s in ("L", "R")},
           "arms": {s: {"upper": float(np.linalg.norm(rest(f"elbow.{s}") - rest(f"arm.{s}"))),
                        "fore": float(np.linalg.norm(rest(f"wrist.{s}") - rest(f"elbow.{s}")))} for s in ("L", "R")},
           "legs": legs, "anchors": {p: card["anchors"][p]["point"] for p in MR.PARTS},
           "leg_home": card["leg_bar"]["home"]}
    if P["world"].get("free"):                                    # a raised hand lifts its shoulder (marionette_free)
        rig["clavicles"] = {s: her(f"shoulder.{s}") for s in ("L", "R")}
    palm = np.cross(rest("index1.L") - rest("wrist.L"), rest("little1.L") - rest("wrist.L"))
    palm = AT.palm_down(palm, np.array([0.0, 0.0, 1.0]))           # MMD rests the palms down (her own frame)
    rig["rest"] = {"upper": rest("elbow.L") - rest("arm.L"), "fore": rest("wrist.L") - rest("elbow.L"),
                   "dir": rest("middle1.L") - rest("wrist.L"), "palm": palm / np.linalg.norm(palm)}
    world = MW.world_plan(ts, P, cross, rig, _aims(ctx, m, P, frames, Fi))
    # her root: a body point q goes to F (pos + R q), so the root's world matrix is F T(pos) R F^-1 M0
    M0 = root.matrix_world.copy()
    base = Fi @ M0
    Pw = root.parent.matrix_world @ root.matrix_parent_inverse if root.parent else Matrix.Identity(4)
    Pwi = Pw.inverted()
    locs, eul, prev = [], [], root.rotation_euler.copy()
    for i in range(len(ts)):
        X = Matrix([list(r) for r in world["root"]["rot"][i]]).to_4x4()
        X.translation = Vector(world["root"]["pos"][i])
        loc, rot, _ = (Pwi @ F @ X @ base).decompose()
        e = rot.to_euler(root.rotation_mode, prev)
        locs.append(tuple(loc))
        eul.append(tuple(e))
        prev = e
    K.key_vec(root, "location", frames, locs, interp="LINEAR")
    K.key_vec(root, "rotation_euler", frames, eul, interp="LINEAR")
    # the bars, the surface under her, where a yank takes the head thread: keyed in the prop root's frame
    Ri = prop.root.matrix_world.inverted()
    Rrot_i = prop.root.matrix_world.to_quaternion().inverted()
    bars = {k: bpy.data.objects[v] for k, v in card["bars"].items()}
    for key, ob in (("cross", bars["cross"]), ("leg", bars["leg"])):
        pl = world[key]
        ob.rotation_mode = "QUATERNION"
        K.key_vec(ob, "location", frames, [tuple(Ri @ (F @ Vector(p))) for p in pl["pos"]], interp="LINEAR")
        K.key_vec(ob, "rotation_quaternion", frames, K.continuous(
            [tuple(Rrot_i @ Frot @ Matrix([list(r) for r in R3]).to_quaternion()) for R3 in pl["rot"]]), interp="LINEAR")
        _key_hidden(ob, frames, ~pl["shown"])
    _key_hidden(bars["hand"], frames, ~world["cross"]["shown"])
    for kind, pts in (("rest", world["rest"]), ("grab", world["grab"])):
        ob = bpy.data.objects.get(f"{prop.name}_{kind}")
        if ob is None:
            ctx.log(f"perform {name}: the prop {prop.name!r} has no {kind} object (an older build of it)")
            continue
        ob.rotation_mode = "QUATERNION"
        K.key_vec(ob, "location", frames, [tuple(Ri @ (F @ Vector(p))) for p in pts], interp="LINEAR")
        K.key_vec(ob, "rotation_quaternion", frames, [tuple(Rrot_i @ Frot)] * len(frames), interp="LINEAR")
    return world


def _grip(m, side):
    """The grip of a hand's pose table ("guitar:strum", ...), "" without one."""
    rig = m.hands.get(side)
    return str(((rig or {}).get("h") or {}).get("grip") or "")


def _keyed(ob, path, n_comp, frames, still):
    """(F, n_comp) values of an object's keyed property on `frames` (its value where it has no curve)."""
    act = ob.animation_data.action if ob.animation_data else None
    out = np.empty((len(frames), n_comp))
    for c in range(n_comp):
        fc = act.fcurves.find(path, index=c) if act is not None else None
        out[:, c] = [fc.evaluate(f) for f in frames] if fc is not None else still[c]
    return out


def _guitar(ctx, m, mp, frames, gname):
    """Turn the worn guitar about its face normal through the point between her hands by the plan's `guitar` (+ its
    neck up): the hand bar's tilt lifts one hand and lowers the other. The hands ride it. Keys on top of the root's own
    turn (a strum's neck kick)."""
    ang = mp["body"]["guitar"]
    g = ctx.props.get(gname)
    if g is None or np.abs(ang).max() < 1e-4:
        return None
    entry = next((u for u in g.card.get("use", {}).get("grip", []) if u.get("normal")), None)
    if entry is None:
        return None
    root = g.root
    sc = bpy.context.scene
    keep = sc.frame_current
    sc.frame_set(int(ctx.start + ctx.settle))
    gi = root.matrix_world.inverted()
    hands = [gi @ m.ik[s][0].matrix_world.translation for s in ("L", "R") if s in m.ik]
    pivot = sum(hands, Vector()) / len(hands) if hands else Vector()
    nl = Vector(entry["normal"]).normalized()
    Rw = root.matrix_world.to_3x3().normalized()
    rise = (Rw @ (Quaternion(nl, 0.1) @ Vector((0.0, 0.0, 1.0)))).z - (Rw @ Vector((0.0, 0.0, 1.0))).z
    sign = 1.0 if rise > 0 else -1.0
    sc.frame_set(keep)
    if root.rotation_mode != "QUATERNION":
        q = root.matrix_basis.to_quaternion()
        root.rotation_mode = "QUATERNION"
        root.rotation_quaternion = q
    q0 = _keyed(root, "rotation_quaternion", 4, frames, tuple(root.rotation_quaternion))
    l0 = _keyed(root, "location", 3, frames, tuple(root.location))
    scale = root.matrix_basis.to_scale()
    S3 = Matrix.Diagonal(scale)
    rots, locs = [], []
    for i in range(len(frames)):
        qb = Quaternion(q0[i]).normalized()
        qn = Quaternion(nl, sign * float(ang[i]))
        rots.append(tuple(qb @ qn))
        locs.append(tuple(Vector(l0[i]) + qb.to_matrix() @ (S3 @ (pivot - qn @ pivot))))
    K.key_vec(root, "rotation_quaternion", frames, K.continuous(rots), interp="LINEAR")
    K.key_vec(root, "location", frames, locs, interp="LINEAR")
    return {"deg": round(float(np.degrees(np.abs(ang).max())), 1)}


def _hang(ctx, m, mp, ts, frames, side, cut):
    """A hand whose thread is cut drops off what it played and swings on its arm into its hang (core.marionette.
    hang_path), in the frame its goal rides, turning with the arm as it swings about the shoulder (so the forearm is
    not wrung); with `back` it goes back to its own keys over that window."""
    goal, _pole = m.ik[side]
    sc = bpy.context.scene
    keep = sc.frame_current
    sc.frame_set(int(ctx.start + ctx.settle))
    Pf = goal.matrix_world @ goal.matrix_basis.inverted()        # the frame the goal's keys are in
    shoulder = np.asarray(Pf.inverted() @ (mp["F"] @ mp["shoulders"][side]))
    to_goal = Pf.to_3x3().inverted() @ mp["Frot"].to_matrix()
    sc.frame_set(keep)
    goal.rotation_mode = "QUATERNION"
    loc = _keyed(goal, "location", 3, frames, tuple(goal.location))
    rot = _keyed(goal, "rotation_quaternion", 4, frames, tuple(goal.rotation_quaternion))
    i0 = min(int(np.searchsorted(ts, cut["t"])), len(ts) - 1)
    start = loc[i0]
    hang = np.asarray(start) + np.asarray(to_goal @ Vector(cut["hang"]))
    path = MR.hang_path(ts, cut["t"], start, shoulder, hang)
    w = MR.step(ts, cut["t"])
    if "back" in cut:
        w = w * (1.0 - PF.smooth((ts - cut["back"][0]) / (cut["back"][1] - cut["back"][0])))
    new = loc + w[:, None] * (path - loc)
    r0 = Vector(start - shoulder)
    q_cut = Quaternion(rot[i0]).normalized()
    quats = []
    for i in range(len(ts)):
        q = Quaternion(rot[i]).normalized()
        if w[i] > 0.0:
            swung = r0.rotation_difference(Vector(path[i] - shoulder)) @ q_cut
            q = q.slerp(swung, float(w[i]))
        quats.append(tuple(q))
    K.key_vec(goal, "location", frames, [tuple(v) for v in new], interp="LINEAR")
    K.key_vec(goal, "rotation_quaternion", frames, K.continuous(quats), interp="LINEAR")
    return {"t": cut["t"], "drop_mm": round(float(np.linalg.norm(hang - start)) * 1000, 1)}


def _hand_bar(ctx, m, mp, ts, frames, gname):
    """The hand bar follows the picking hand: its end over that hand goes up and down with the hand's strokes (the
    thread does the strumming) while that thread holds, the bar turning about the other hand's eye so that thread keeps
    its pull."""
    g = ctx.props.get(gname)
    side = next((s for s in ("R", "L") if s in m.ik and m.ik[s][0].parent is (g.root if g else None)
                 and _grip(m, s).endswith(":strum")), None) if g else None
    if side is None:
        return None
    bar = bpy.data.objects[mp["card"]["bars"]["hand"]]
    anchors = mp["card"]["anchors"]
    x_hand = float(anchors[f"wrist_{side}"]["point"][0])
    pivot = Vector(anchors[f"wrist_{'L' if side == 'R' else 'R'}"]["point"])
    pivot.z = 0.0
    lever = x_hand - pivot.x
    goal = m.ik[side][0]
    loc = _keyed(goal, "location", 3, frames, tuple(goal.location))
    Rw = g.root.matrix_world.to_3x3()
    dz = np.array([(Rw @ Vector(v)).z for v in loc - loc[0]])
    dz = dz - PF.lowpass(dz, 30)                                 # the strokes, not where the hand rests
    held = MR.held(ts, mp["P"], f"wrist_{side}")
    ang = np.arcsin(np.clip(dz * held / abs(lever), -0.3, 0.3)) * (-1.0 if lever < 0 else 1.0)
    at = Vector(mp["card"]["hand_bar"]["at"])
    qs = [Quaternion((0, 1, 0), -float(a)) for a in ang]
    bar.rotation_mode = "QUATERNION"
    K.key_vec(bar, "rotation_quaternion", frames, [tuple(q) for q in qs], interp="LINEAR")
    K.key_vec(bar, "location", frames, [tuple(at + pivot - q @ pivot) for q in qs], interp="LINEAR")
    return {"side": side, "deg": round(float(np.degrees(np.abs(ang).max())), 1)}


def hands(ctx, name, m, mp, ts, frames):
    """After the strum: the guitar turned with the hand bar, the hands whose threads are cut swung on their arms, the
    hand bar following the picking hand; in the world (`world_hands`) her hands where the plan has them. Returns
    report numbers."""
    if mp.get("world") is not None:
        return _world_hands(ctx, name, m, mp, frames)
    spec = ctx.section("perform", name)
    st = spec.get("strum")
    st = st[0] if isinstance(st, list) and st else st
    gname = mp["P"]["guitar"] or (st.get("prop") if isinstance(st, dict) else None)
    out = {}
    if gname:
        out["guitar"] = _guitar(ctx, m, mp, frames, gname)
        out["hand_bar"] = _hand_bar(ctx, m, mp, ts, frames, gname)
    for c in mp["P"]["cuts"]:
        for th in c["threads"]:
            if th.startswith("wrist_"):
                side = th[-1]
                grip = _grip(m, side)
                if side in m.ik and not grip.endswith(":neck"):      # a hand on a neck holds on to it
                    out[f"hang_{side}"] = _hang(ctx, m, mp, ts, frames, side, c)
    return out


def _world_hands(ctx, name, m, mp, frames):
    """Her hands where the world plan has them (mkmmd.core.marionette_world: held by their threads, lying on the water
    up by her face, reaching, swimming, cut free and swinging, treading the air, out for her balance): each arm IK's
    goal keyed on every frame from the wrist, the hand's direction and its palm, and its elbow's pole. A hand needs a
    pose table to have an arm IK (`at` anywhere: the threads take it from there)."""
    smap = S.semantic_map(m.arm)
    F, Frot = mp["F"], mp["Frot"].to_matrix()
    out = {}
    for side in ("L", "R"):
        if side not in m.ik:
            ctx.log("WARNING", f"perform.{name}.marionette: her {side} hand has no arm IK, so its thread cannot move "
                               f"it: give [pose.{name}.hands.{side}] an `at` (anywhere)")
            continue
        rig = m.hands[side]
        if rig.get("ride_bone") or (_grip(m, side) and _grip(m, side) != "rest"):
            raise BuildError(f"perform.{name}.marionette: her {side} hand rides or grips something, and her threads "
                             f"move it: give it an `at` alone")
        goal, pole = m.ik[side]
        H = mp["world"]["hands"][side]
        arm = m.arm
        wb = arm.data.bones[smap[f"wrist.{side}"]]
        Bl = wb.matrix_local.to_3x3().normalized()                # the wrist's rest turn, armature space
        head = {k: arm.data.bones[smap[f"{k}.{side}"]].head_local for k in ("wrist", "index1", "little1")}
        n_arm = (head["index1"] - head["wrist"]).cross(head["little1"] - head["wrist"]).normalized()
        n_arm = AT.palm_down(n_arm, Vector((0.0, 0.0, 1.0)))      # palms down at rest, in her own frame (the world's
        n_loc = Bl.inverted() @ n_arm                             # z is no guide once she lies on her back)
        y_loc = Vector((0.0, 1.0, 0.0))
        rest_basis = Matrix(np.array(AT.basis(np.array(y_loc), np.array(n_loc))).tolist())
        locs, rots, poles = [], [], []
        for i in range(len(frames)):
            d, p = Frot @ Vector(H["dir"][i]), Frot @ Vector(H["palm"][i])
            Rw = Matrix(np.array(AT.basis(np.array(d), np.array(p))).tolist()) @ rest_basis.transposed()
            M = Rw.to_4x4()
            M.translation = F @ Vector(H["wrist"][i])
            Ml = rig["par_inv"] @ (M @ Matrix.Translation((0.0, wb.length, 0.0)))
            locs.append(tuple(Ml.translation))
            rots.append(tuple(Ml.to_quaternion()))
            poles.append(tuple(rig["pole_inv"] @ (F @ Vector(H["pole"][i]))))
        goal.rotation_mode = "QUATERNION"
        K.key_vec(goal, "location", frames, locs, interp="LINEAR")
        K.key_vec(goal, "rotation_quaternion", frames, K.continuous(rots), interp="LINEAR")
        K.key_vec(pole, "location", frames, poles, interp="LINEAR")
        if "shrug" in H:                                          # set free: a raised hand lifts its shoulder
            Rra = m.root.matrix_world.to_3x3().normalized().inverted() @ arm.matrix_world.to_3x3().normalized()
            u = (Rra.inverted() @ Vector((0.0, 1.0, 0.0))).normalized()     # her front-to-back axis, armature axes
            sg = 1.0 if side == "L" else -1.0
            K.key_bone_arm(arm, smap[f"shoulder.{side}"], frames, [Quaternion(u, -sg * float(a)) for a in H["shrug"]],
                           interp="LINEAR")
        rest = mp["world"]["rig"]["rest"]
        R = mp["world"]["root"]["rot"]
        d, p = np.einsum("nji,nj->ni", R, H["dir"]), np.einsum("nji,nj->ni", R, H["palm"])
        S0 = H["shoulder"] if "shoulder" in H else np.tile(mp["world"]["rig"]["shoulders"][side], (len(frames), 1))
        reach = np.linalg.norm(H["body"] - S0, axis=1)
        tw = np.array([AT.twists(rest, side, S0[i], H["elbow"][i], H["body"][i], d[i], p[i])
                       for i in range(0, len(frames), 3)])
        out[f"hand_{side}"] = {"reach_mm": [round(float(reach.min()) * 1000, 1), round(float(reach.max()) * 1000, 1)],
                               "forearm_twist": [round(float(tw[:, 0].min()), 1), round(float(tw[:, 0].max()), 1)],
                               "arm_twist": [round(float(tw[:, 1].min()), 1), round(float(tw[:, 1].max()), 1)]}
        if "shrug" in H:
            out[f"hand_{side}"]["shrug_deg"] = round(math.degrees(float(np.max(H["shrug"]))), 1)
    return out


# ---- the stage: the threads


def _windows(ctx, name):
    entries = ctx.data.get("stepped", [])
    if not entries:
        return []
    try:
        return [w for w in STP.normalize(entries, list(ctx.cast), ctx.fps, ctx.frame0, ctx.end) if name in w["cast"]]
    except STP.StepError as e:
        raise BuildError(str(e)) from None


def _last(ctx, m):
    shows = m.spec.get("shows")
    if shows:
        return min(ctx.end, int(math.ceil(ctx.frame(max(float(b) for _a, b in shows)))))
    return ctx.end


WRAP = (("neck", 0.07, 0.0), ("head", 0.13, 0.06), ("crown", 0.06, -0.03), ("crown", 0.0, 0.015))
#   the back thread's way round her when she lies on her back: m behind and up from her neck's base, the base of her
#   skull and her crown (where the head thread is tied): under her nape, behind her skull, over the top of her head


def _resample(pts, n):
    """n points evenly along the polyline pts (m, 3)."""
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    u = np.linspace(0.0, cum[-1], n)
    return np.stack([np.interp(u, cum, pts[:, j]) for j in range(3)], 1)


def _wrapped(a, b, L, n, side, axes, at, w):
    """A thread from b (on her back) to a that wraps round under her nape, the back of her head and over her crown when
    she lies on her back (`w` 0..1 how far the straight line would go through her): b, the wrap points (`at` {neck,
    head, crown} world points, her axes: x left, y back, z up), then free to a, sagging as a thread of length L does
    over what is left of it."""
    back, up = axes[:, 1], axes[:, 2]
    way = [at[k] + back * dy + up * dz for k, dy, dz in WRAP]
    t = np.linspace(0.0, 1.0, len(way) + 2)[1:-1]
    way = [(1.0 - w) * (b + (a - b) * u) + w * p for u, p in zip(t, way)]
    held = np.array([b] + way)
    used = float(np.linalg.norm(np.diff(held, axis=0), axis=1).sum())
    free = MR.sag_points(held[-1], a, max(L - used, 1e-3), n, side)
    return _resample(np.concatenate([held, free]), n)


def _threads_world(ctx, m, P, prop, card, frames, ts, A, B, body):
    """The threads of a marionette in the world (mkmmd.core.marionette_world): each one's length the shortest span it
    has while it pulls her (so it is straight whenever it pulls and sags only when the control comes down to her), a
    twang when it snaps taut, the back thread round under her nape and her head while she lies on her back, a cut piece
    hanging from her (she is tied to it), the head thread streaming between her and the falling control and lying
    slack over the ground she lands on, taken by a hand at the grab point and pulled, every thread whole again from the
    knot. Keys the threads, the pieces and the snips; returns the report's `threads`."""
    n, npts = len(frames), int(card["thread"]["points"])
    F, Frot = _rest_frame(m)
    Fr = np.array(Frot.to_matrix())
    Ri = np.array(prop.root.matrix_world.inverted())
    radius = float(card["thread"]["radius"])
    cross = MR.cross_plan(ts, P, float(card["leg_bar"]["half"]))
    Rg, Tm = MW.regimes(ts, P)
    taut = MW.taut(ts, P, cross)
    ball = None
    if P["world"]["land"]:
        at = _aims(ctx, m, P, frames, F.inverted())["land"]
        c = np.array([tuple(F @ Vector(p)) for p in at]) - np.array([0.0, 0.0, P["world"]["land"]["radius"]])
        ball = (c, P["world"]["land"]["radius"] + radius + 0.002)
    rep = {}
    for j, part in enumerate(MR.PARTS):
        a_, b_ = A[:, j], B[:, j]
        span = np.linalg.norm(a_ - b_, axis=1)
        hold = MW.held(ts, P, part) > 0.5
        on = taut & hold
        i_on = int(np.clip(np.searchsorted(ts, P["from"]), 0, n - 1))
        L = float(span[on].min()) if on.any() else float(span[i_on])
        tc = MR.cut_time(P, part)
        cut = next((c for c in P["cuts"] if part in c["threads"]), None)
        side = Fr @ np.array([SIDE[part], 0.0, 0.0])
        # it snaps taut: a twang (how hard: how fast the slack was taken up)
        rate = np.gradient(span, ts) if n > 1 else np.zeros(n)
        slack = span < L - 0.015
        snaps = [(ts[i], min(abs(float(rate[i])) * 0.012, 0.02)) for i in range(1, n)
                 if slack[i - 1] and not slack[i] and hold[i]]
        main, low = np.empty((n, npts, 3)), np.empty((n, npts, 3))
        show_main, show_low = np.zeros(n, bool), np.zeros(n, bool)
        snip = np.empty((n, 3))
        C = None
        vel = np.gradient(b_, ts, axis=0) if n > 1 else np.zeros_like(b_)
        for i, t in enumerate(ts):
            a, b = a_[i], b_[i]
            low[i] = b
            if hold[i]:
                C = None
                if part == "head" and Rg["yanked"][i] and not Rg["carry"][i]:     # a hand pulls it at the grab
                    g = body["grab"][i]
                    main[i] = _resample(np.array([a, g, b]), npts)
                elif part == "back" and body["axes"][i][:, 1] @ (a - b) < 0.0:
                    w = float(np.clip(-(body["axes"][i][:, 1] @ _unitv(a - b)) / 0.25, 0.0, 1.0))
                    main[i] = _wrapped(a, b, L, npts, side, body["axes"][i],
                                       {"neck": body["neck"][i], "head": body["head"][i], "crown": B[i, 0]}, w)
                else:                                           # landed, it has paid out round the stirrer: slack
                    paid = max(L, 1.12 * span[i]) if Rg["landed"][i] and part == "head" else L
                    main[i] = MR.sag_points(a, b, paid, npts, side)
                if Rg["fall"][i] and part == "head":            # streaming in the air between her and the control
                    tau = t - Tm["rel"]
                    sweep = np.sin(np.pi * np.linspace(0.0, 1.0, npts))
                    main[i] += np.outer(0.04 * L * sweep * math.sin(9.0 * tau + 0.7), side)
                    main[i] += np.outer(0.02 * L * sweep ** 2 * math.sin(14.0 * tau), Fr @ np.array([0.0, 1.0, 0.0]))
                if ball is not None and Rg["landed"][i]:        # slack on the ground she landed on, not through it
                    d = main[i] - ball[0][i]
                    r = np.linalg.norm(d, axis=1)
                    inside = r < ball[1]
                    main[i][inside] = ball[0][i] + d[inside] / r[inside, None] * ball[1]
                for ts_, amp in snaps:
                    main[i] += np.outer(MR.twang(npts, t - ts_, amp), side)
                show_main[i] = True
            else:
                if C is None:
                    C = MR.point_at(main[i - 1] if i else MR.sag_points(a, b, L, npts, side), cut["at"] if cut else 0.5)
                    t_cut = t
                tau = t - t_cut
                main[i] = MR.cut_upper(a, C, tau, npts, side)
                show_main[i] = tau < MR.YANK + 0.08
                v = vel[i]
                hang = np.array([0.0, 0.0, -1.0]) - 0.12 * v / max(float(np.linalg.norm(v)), 1.0)
                low[i] = MR.cut_lower(b, C, tau, hang, -1e9, None, radius, npts)     # tied to her: it hangs from her
                show_low[i] = True
            snip[i] = body["grab"][i] if (part == "head" and Rg["yanked"][i]) else (
                MR.point_at(main[i], cut["at"] if cut else 0.5) if C is None else C)
        for kind, pts, shown in (("thread", main, show_main), ("cut", low, show_low)):
            ob = bpy.data.objects[f"{prop.name}_{kind}_{part}"]
            local = pts @ Ri[:3, :3].T + Ri[:3, 3]
            for q in range(npts):
                for c in range(3):
                    K.set_fcurve(ob.data, f"splines[0].points[{q}].co", c, frames, local[:, q, c], interp="LINEAR")
            _key_hidden(ob, frames, ~shown)
        sn = bpy.data.objects[f"{prop.name}_snip_{part}"]
        K.key_vec(sn, "location", frames, [tuple(v) for v in snip @ Ri[:3, :3].T + Ri[:3, 3]], interp="LINEAR")
        rep[part] = {"length_m": round(L, 3), "cut": None if tc == math.inf else round(tc, 3), "snaps": len(snaps)}
    return rep


def _key_width(P, prop, frames, ts):
    """`width`: every thread's and cut piece's radius over time (eased keys): a thread reads as thread in a close plate
    (a thin one while she lies under a lens a metre or two away) and as a line in a wide shot."""
    if not P.get("width"):
        return
    r = PF.eased_keys(ts, P["width"])
    for part in MR.PARTS:
        for kind in ("thread", "cut"):
            K.set_fcurve(bpy.data.objects[f"{prop.name}_{kind}_{part}"].data, "bevel_depth", 0, frames, r, interp="LINEAR")


def _unitv(v):
    v = np.asarray(v, float)
    return v / max(float(np.linalg.norm(v)), 1e-9)


def run(ctx):
    out = {}
    for name, m in ctx.cast.items():
        got = plan(ctx, name)
        if got is None or m.arm is None:
            continue
        P, prop, card = got
        bars = [bpy.data.objects[v] for v in card["bars"].values()]
        if P["world"]["on"]:                                      # the surface and the grab step with her too
            bars += [o for o in (bpy.data.objects.get(f"{prop.name}_{k}") for k in ("rest", "grab")) if o is not None]
        held = 0
        for w in _windows(ctx, name):                             # the bars step with her
            for ob in bars:
                ad = ob.animation_data
                for fc in (ad.action.fcurves if ad and ad.action else ()):
                    if len(fc.keyframe_points):
                        STEPPED.hold(fc, w)
                        held += 1
        if not P["control"] or P["world"].get("free"):            # set free, or tails alone (build/marionette_free.py)
            from . import marionette_free as MFREE
            out[name] = MFREE.run(ctx, name, m, P, prop, card, held)
            ctx.log("marionette", name, json.dumps(out[name]))
            continue
        f0 = max(ctx.start, int(math.floor(ctx.frame(P["from"] - P["drop"]))) - 1)
        f1 = _last(ctx, m)
        frames = np.arange(f0, f1 + 1)
        ts = np.array([ctx.time(f) for f in frames])
        anchors = [bpy.data.objects[card["anchors"][p]["object"]] for p in MR.PARTS]
        ends = [bpy.data.objects[f"{prop.name}_end_{p}"] for p in MR.PARTS]
        n, k = len(frames), len(MR.PARTS)
        A, B = np.empty((n, k, 3)), np.empty((n, k, 3))
        body = {"neck": np.empty((n, 3)), "head": np.empty((n, 3)), "axes": np.empty((n, 3, 3)), "grab": np.empty((n, 3))}
        smap = S.semantic_map(m.arm)
        grab = bpy.data.objects.get(f"{prop.name}_grab")
        sc = bpy.context.scene
        keep = sc.frame_current
        bpy.context.view_layer.update()                          # objects the sim removed leave stale entries
        with SCOPE.only(anchors + ends + bars + [m.arm, m.root, prop.root]):
            for i, f in enumerate(frames):
                sc.frame_set(int(f))
                dg = bpy.context.evaluated_depsgraph_get()
                for j in range(k):
                    A[i, j] = anchors[j].evaluated_get(dg).matrix_world.translation
                    B[i, j] = ends[j].evaluated_get(dg).matrix_world.translation
                if P["world"]["on"]:                              # what the back thread wraps round, and the yank
                    ae = m.arm.evaluated_get(dg)
                    body["axes"][i] = np.array(ae.matrix_world.to_3x3().normalized())
                    for key, b in (("neck", "neck"), ("head", "head")):
                        body[key][i] = ae.matrix_world @ ae.pose.bones[smap[b]].head
                    body["grab"][i] = grab.evaluated_get(dg).matrix_world.translation if grab else B[i, 0]
        sc.frame_set(keep)
        if P["world"]["on"]:
            rep = _threads_world(ctx, m, P, prop, card, frames, ts, A, B, body)
            _key_width(P, prop, frames, ts)
            out[name] = {"prop": prop.name, "frames": [int(frames[0]), int(frames[-1])], "threads": rep,
                         "bar_curves_stepped": held}
            ctx.log("marionette", name, json.dumps(out[name]))
            continue
        F, Frot = _frame(m)
        Fr = np.array(Frot.to_matrix())
        floor = float(F.translation.z)
        i_on = int(np.clip(np.searchsorted(ts, P["from"]), 0, n - 1))
        lengths = np.linalg.norm(A[i_on] - B[i_on], axis=1)
        Ri = np.array(prop.root.matrix_world.inverted())
        npts = int(card["thread"]["points"])
        rep = {}
        for j, part in enumerate(MR.PARTS):
            tc = MR.cut_time(P, part)
            cut = next((c for c in P["cuts"] if part in c["threads"]), None)
            side = Fr @ np.array([SIDE[part], 0.0, 0.0])
            drape = Fr @ np.asarray(MR.DRAPE[part], float)
            main = np.empty((n, npts, 3))
            low = np.empty((n, npts, 3))
            show_main, show_low = np.zeros(n, bool), np.zeros(n, bool)
            snip = np.empty((n, 3))
            L = float(lengths[j])
            C = B_slide = None
            ta = P["from"] - P["drop"]
            for i, t in enumerate(ts):
                a, b = A[i, j], B[i, j]
                low[i] = b
                if t < ta:
                    main[i] = MR.dangle(a, L, 0.0, npts, side)
                elif t < P["from"] - MR.ATTACH:
                    main[i] = MR.dangle(a, L, t - ta, npts, side)
                    show_main[i] = True
                elif t < P["from"]:
                    s = float(PF.smooth((t - P["from"] + MR.ATTACH) / MR.ATTACH))
                    e = (1.0 - s) * (a + np.array([0.0, 0.0, -L])) + s * b
                    main[i] = (1.0 - s) * MR.dangle(a, L, t - ta, npts, side) + s * MR.sag_points(a, e, L * 1.03, npts, side)
                    show_main[i] = True
                elif t < tc:
                    main[i] = MR.sag_points(a, b, L, npts, side)
                    main[i] += np.outer(MR.twang(npts, t - P["from"]), side)
                    show_main[i] = True
                else:
                    if C is None:
                        C = MR.point_at(main[i - 1] if i else MR.sag_points(a, b, L, npts, side), cut["at"])
                    tau = t - tc
                    main[i] = MR.cut_upper(a, C, tau, npts, side)
                    show_main[i] = tau < MR.YANK + 0.08
                    if tau >= MR.SLIDE and B_slide is None:
                        B_slide = b.copy()
                    low[i] = MR.cut_lower(b, C, tau, drape, floor, B_slide, float(card["thread"]["radius"]), npts)
                    show_low[i] = True
                snip[i] = MR.point_at(main[i], cut["at"] if cut else 0.5) if C is None else C
            for kind, pts, shown in (("thread", main, show_main), ("cut", low, show_low)):
                ob = bpy.data.objects[f"{prop.name}_{kind}_{part}"]
                local = pts @ Ri[:3, :3].T + Ri[:3, 3]
                cu = ob.data
                for q in range(npts):
                    for c in range(3):
                        K.set_fcurve(cu, f"splines[0].points[{q}].co", c, frames, local[:, q, c], interp="LINEAR")
                _key_hidden(ob, frames, ~shown)
            sn = bpy.data.objects[f"{prop.name}_snip_{part}"]
            K.key_vec(sn, "location", frames, [tuple(v) for v in snip @ Ri[:3, :3].T + Ri[:3, 3]], interp="LINEAR")
            rep[part] = {"length_m": round(L, 3), "cut": None if tc == math.inf else round(tc, 3)}
        _key_width(P, prop, frames, ts)
        out[name] = {"prop": prop.name, "frames": [int(frames[0]), int(frames[-1])], "threads": rep,
                     "bar_curves_stepped": held}
        ctx.log("marionette", name, json.dumps(out[name]))
    return out
