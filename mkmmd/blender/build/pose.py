"""pose: each cast member's base pose, eased in from the rest pose over the first `settle_frames` of the pre-roll.

[pose.<cast>] keys
  sit = "prop:seat" | {hip, facing, seat_z, floor_z, pelvis_deg, back_deg}   place the hips on a seat (root follows
        the prop); without sit the model stands where the cast stage put it
  feet = "prop:feet" | "floor" | {L = [x, y, z|nan], R = [...]}               ankle targets (leg IK); z omitted / nan =
        the model's own ankle height above the floor
  lean, turn (deg)            upper body forward lean / turn toward the model's left, on top of the seat's back angle
  head = {pitch, yaw, roll}   base head rotation (deg); perform's gaze adds on top
  [pose.<cast>.hands.<L|R>]   arm IK on the wrist through the twist bones:
     at = target ref, dir = [x, y, z] (hand direction), palm = [x, y, z] (palm normal), pole = target ref
     rest = "prop:edge" (+ along = 0..1, lift = m): the hand lies on an edge use point
     keys = [{t, at, dir, palm}]  moving targets (clip seconds, eased)
     fingers = "relaxed" | "curled" | "fist" | "flat" | "point"
  fingers = {L = preset, R = preset}  without arm IK
World-axis rotations are keyed with mkmmd.blender.keys (q_child = D_parent^-1 D_want)."""
import math

import bpy
from mathutils import Euler, Matrix, Quaternion, Vector

from .. import keys as K
from .. import scene as S
from . import BuildError, collection, targets

FINGER_PRESETS = {             # curl per joint (deg) for (finger 1, 2, 3); thumb separately
    "flat": ((0, 0, 0), (0, 0, 0)),
    "relaxed": ((14, 22, 14), (8, 10, 8)),
    "curled": ((35, 50, 35), (14, 18, 14)),
    "fist": ((80, 95, 65), (25, 35, 40)),
    "point": ((80, 95, 65), (25, 35, 40)),
}


def _q(axis, deg):
    return Quaternion(Vector(axis).normalized(), math.radians(deg))


def seat_frame(ctx, spec):
    """World seat: hip point, facing (unit, horizontal), floor z, pelvis tilt, back angle, prop (or None)."""
    if isinstance(spec, str):
        prop_name, _, use = spec.partition(":")
        if prop_name not in ctx.props:
            raise BuildError(f"sit {spec!r}: no prop {prop_name!r}")
        p = ctx.props[prop_name]
        u = p.use("sit", use or None)
        f = p.world_dir(u.get("facing", (0, -1, 0)))
        floor = (p.root.matrix_world @ Vector((0, 0, u.get("floor_z", 0.0)))).z
        return {"hip": p.world(u["hip"]), "facing": Vector((f.x, f.y, 0)).normalized(), "floor_z": floor,
                "pelvis": float(u.get("pelvis_deg", 6.0)), "back": float(u.get("back_deg", 0.0)), "prop": p,
                "use": u}
    f = Vector(spec.get("facing", (0, -1, 0)))
    return {"hip": Vector(spec["hip"]), "facing": Vector((f.x, f.y, 0)).normalized(),
            "floor_z": float(spec.get("floor_z", 0.0)), "pelvis": float(spec.get("pelvis_deg", 6.0)),
            "back": float(spec.get("back_deg", 0.0)), "prop": None, "use": spec}


def _place_root(m, seat):
    f = seat["facing"]
    yaw = math.atan2(f.x, -f.y)
    want = Matrix.Translation((seat["hip"].x, seat["hip"].y, seat["floor_z"])) @ Matrix.Rotation(yaw, 4, "Z")
    root = m.root
    if seat["prop"] is not None:
        root.parent = seat["prop"].root
        root.matrix_parent_inverse.identity()
        root.matrix_basis = seat["prop"].root.matrix_world.inverted() @ want
    else:
        root.matrix_world = want
    bpy.context.view_layer.update()


def _world_head(arm, bone):
    return arm.matrix_world @ arm.data.bones[bone].head_local


def _axes(m):
    """The model's world axes at rest: lateral (its left), back, up."""
    R = m.arm.matrix_world.to_3x3().normalized()
    return R @ Vector((1, 0, 0)), R @ Vector((0, 1, 0)), Vector((0, 0, 1))


def _spine(m, smap, lean, turn, head):
    """Base rotations of the spine and head in the ARMATURE's axes (the model faces -Y): {semantic: rotation the bone
    adds} and the composed chain (D2 chest, Dn neck, Dh head)."""
    lat, back, up = Vector((1, 0, 0)), Vector((0, 1, 0)), Vector((0, 0, 1))
    q_up1 = _q(up, turn * 0.6) @ _q(lat, lean * 0.6)
    has2 = "upper_body2" in smap
    q_up2 = _q(up, turn * 0.4) @ _q(lat, lean * 0.4) if has2 else Quaternion()
    D2 = q_up1 @ q_up2
    Dh = _q(back, head.get("roll", 0.0)) @ _q(up, head.get("yaw", 0.0)) @ _q(lat, head.get("pitch", 0.0))
    Dn = D2.slerp(Dh, 0.4)
    out = {"upper_body": q_up1, "neck": D2.inverted() @ Dn, "head": Dn.inverted() @ Dh}
    if has2:
        out["upper_body2"] = q_up2
    return out, {"D2": D2, "Dn": Dn, "Dh": Dh}


def _finger_quats(arm, smap, side, preset):
    """Bone -> local quaternion curling each finger toward the palm about its own flexion axis (finger direction x
    palm normal, from the rest pose)."""
    if preset not in FINGER_PRESETS:
        raise BuildError(f"finger preset {preset!r} (have {sorted(FINGER_PRESETS)})")
    from ...core import bonemap
    fing, thumb = FINGER_PRESETS[preset]
    if not all(f"{k}.{side}" in smap for k in ("wrist", "index1", "little1")):
        return {}
    palm = palm_normal(arm, smap, side)
    out = {}
    for name, sems in bonemap.FINGERS.items():
        angles = thumb if name == "thumb" else fing
        if preset == "point" and name == "index":
            angles = (0, 0, 0)
        bones = [smap[f"{s}.{side}"] for s in sems if f"{s}.{side}" in smap]
        if len(bones) < 2:
            continue
        d = (_world_head(arm, bones[1]) - _world_head(arm, bones[0])).normalized()
        axis = d.cross(palm).normalized()
        for b, deg in zip(bones[:3], angles):
            B = K.rest_rot(arm, b)
            out[b] = (B.inverted() @ _q(axis, deg).to_matrix() @ B).to_quaternion()
    return out


def _arm_ik(ctx, m, smap, side):
    """IK on the wrist through the twist bones (chain up to the upper arm). Returns (target, pole) empties."""
    arm = m.arm
    wrist, upper = smap[f"wrist.{side}"], smap[f"arm.{side}"]
    n, b = 1, arm.pose.bones[wrist]
    while b.parent is not None and b.name != upper:
        b = b.parent
        n += 1
    if b.name != upper:
        raise BuildError(f"{m.name}: {wrist} does not hang from {upper}")
    coll = collection("Rig")
    tgt = bpy.data.objects.new(f"{m.name}_hand.{side}", None)
    pole = bpy.data.objects.new(f"{m.name}_elbow.{side}", None)
    for o, size in ((tgt, 0.05), (pole, 0.04)):
        o.empty_display_size = size
        coll.objects.link(o)
    tgt.rotation_mode = "QUATERNION"
    c = arm.pose.bones[wrist].constraints.new("IK")
    c.name = "mk_arm_ik"
    c.target, c.pole_target = tgt, pole
    c.pole_angle = math.radians(-90)
    c.chain_count = n
    c.use_rotation = True
    c.orient_weight = 1.0
    c.iterations = 500
    for tw in (f"arm_twist.{side}", f"wrist_twist.{side}"):
        if tw in smap:
            tb = arm.pose.bones[smap[tw]]
            tb.lock_ik_x = tb.lock_ik_z = True
    return tgt, pole


def palm_normal(arm, smap, side):
    """The hand's palm normal in world at rest (MMD rest pose: palms face down)."""
    hw = _world_head(arm, smap[f"wrist.{side}"])
    n = (_world_head(arm, smap[f"index1.{side}"]) - hw).cross(_world_head(arm, smap[f"little1.{side}"]) - hw)
    n.normalize()
    return -n if n.z > 0 else n


def _hand_matrix(arm, smap, side, at, direction, palm=None):
    """IK goal for a wrist head at `at` with the hand pointing along `direction` and the palm facing `palm` (rolled
    about the hand axis; default: the least rotation from rest). IK goals act on the bone's tail, so the goal is the
    head frame moved along the bone by its length."""
    wrist = smap[f"wrist.{side}"]
    bone = arm.data.bones[wrist]
    B = K.rest_rot(arm, wrist)
    rest_dir = (B @ Vector((0, 1, 0))).normalized()
    q = rest_dir.rotation_difference(Vector(direction).normalized())
    R = q.to_matrix() @ B
    if palm is not None:
        n_local = B.inverted() @ palm_normal(arm, smap, side)
        y = (R @ Vector((0, 1, 0))).normalized()
        cur = (R @ n_local).normalized()
        want = Vector(palm) - Vector(palm).dot(y) * y
        cur = cur - cur.dot(y) * y
        if want.length > 1e-6 and cur.length > 1e-6:
            want.normalize()
            cur.normalize()
            ang = math.atan2(cur.cross(want).dot(y), cur.dot(want))
            R = Quaternion(y, ang).to_matrix() @ R
    M = R.to_4x4()
    M.translation = Vector(at)
    return M @ Matrix.Translation((0.0, bone.length, 0.0))


def run(ctx):
    out = {}
    f0, f1 = ctx.start, ctx.start + ctx.settle
    bpy.context.scene.frame_set(f0)                       # vehicles may have moved the props: one consistent frame
    for name, m in ctx.cast.items():
        spec = ctx.section("pose", name)
        if not spec:
            continue
        arm = m.arm
        smap = S.semantic_map(arm)
        info = {}
        seat = seat_frame(ctx, spec["sit"]) if spec.get("sit") else None
        if seat:
            _place_root(m, seat)
            m.seat = seat
        lat, back, up = _axes(m)
        # hips onto the seat
        if seat:
            hips = (_world_head(arm, smap["leg.L"]) + _world_head(arm, smap["leg.R"])) / 2
            off = seat["hip"] - hips
            K.key_bone_locs(arm, smap["center"], [f0, f1], [(0, 0, 0), tuple(off)])
            K.key_bone_arm(arm, smap["lower_body"], [f0, f1], [Quaternion(), _q((1, 0, 0), -seat["pelvis"])],
                           interp="BEZIER")
            info["hip_offset"] = [round(v, 4) for v in off]
        # spine and head
        lean = float(spec.get("lean", 0.0)) + (-seat["back"] if seat else 0.0)
        base, chain = _spine(m, smap, lean, float(spec.get("turn", 0.0)), spec.get("head", {}))
        for sem, q in base.items():
            K.key_bone_arm(arm, smap[sem], [f0, f1], [Quaternion(), q], interp="BEZIER")
        m.base = {"spine": base, "chain": chain}
        # feet on the floor / pedals (leg IK bones move to the ankle targets)
        feet = spec.get("feet", "seat" if seat else None)
        if feet:
            if feet == "seat" or feet == "floor":
                fd = None
                if seat and seat["prop"] is not None and feet == "seat":
                    for key in (seat["use"].get("name"), None):     # the seat's own feet, else the only feet
                        try:
                            fd = seat["prop"].use("feet", key)
                            break
                        except BuildError:
                            continue
                pts = {s: (seat["prop"].world(list(fd[s][:2]) + [0.0]) if fd else None) for s in ("L", "R")}
            elif isinstance(feet, str):
                prop_name, _, use = feet.partition(":")
                p = ctx.props[prop_name]
                fd = p.use("feet", use or None)
                pts = {s: p.world(fd[s][:2] + [0.0]) for s in ("L", "R")}
            else:
                pts = {s: Vector(feet[s][:2] + [0.0]) for s in ("L", "R")}
            for s in ("L", "R"):
                ik = smap.get(f"leg_ik.{s}")
                if not ik:
                    continue
                rest = _world_head(arm, ik)
                if pts[s] is None:                     # in front of the knee, legs as relaxed as the seat allows
                    knee_fwd = seat["hip"] - back * (rest - _world_head(arm, smap[f"leg.{s}"])).length * 0.55
                    side_off = lat * (0.1 if s == "L" else -0.1)
                    tgt = Vector((knee_fwd.x + side_off.x, knee_fwd.y + side_off.y, 0.0))
                else:
                    tgt = pts[s]
                z_rest = rest.z
                goal = Vector((tgt.x, tgt.y, z_rest))
                K.key_bone_locs(arm, ik, [f0, f1], [(0, 0, 0), tuple(goal - rest)])
        # arms
        hands = spec.get("hands", {})
        for side in ("L", "R"):
            h = hands.get(side)
            if not h:
                continue
            tgt, pole = _arm_ik(ctx, m, smap, side)
            m.ik[side] = (tgt, pole)
            wrist = smap[f"wrist.{side}"]
            sh = _world_head(arm, smap[f"arm.{side}"])
            out_dir = lat if side == "L" else -lat
            if h.get("pole") is not None:
                pole.location = targets.point(ctx, h["pole"])
            else:
                pole.location = sh + out_dir * 0.45 + back * 0.25 - up * 0.30
            if seat and seat["prop"] is not None:          # IK goals ride with the prop
                for o in (tgt, pole):
                    mw = o.matrix_world.copy()
                    o.parent = seat["prop"].root
                    o.matrix_world = mw
            if h.get("ride"):                              # ... or with one of its parts (a steering wheel)
                ob = bpy.data.objects.get(h["ride"])
                if ob is None:
                    raise BuildError(f"pose.{name}.hands.{side}: ride object {h['ride']!r} not found")
                mw = tgt.matrix_world.copy()
                tgt.parent = ob
                tgt.matrix_world = mw
            frames, mats = [], []
            if h.get("rest"):
                prop_name, _, use = h["rest"].partition(":")
                p = ctx.props[prop_name]
                u = p.use("rest", use)
                n_w = p.world_dir(u.get("normal", (0, 0, 1)))
                if u.get("type") == "plane" or "a" not in u:     # a plane: centre + offset in the prop's frame
                    at = p.world(Vector(u["center"]) + Vector(h.get("offset", (0.0, 0.0, 0.0))))
                else:                                             # an edge: a point along it
                    at = p.world(u["a"]).lerp(p.world(u["b"]), float(h.get("along", 0.5)))
                at = at + n_w * float(h.get("lift", 0.03))
                d = Vector(h.get("dir", tuple(-back + out_dir * 0.3)))
                mats.append((f1, _hand_matrix(arm, smap, side, at, d, h.get("palm", (0, 0, -1)))))
            if h.get("at") is not None:
                at = targets.point(ctx, h["at"])
                mats.append((f1, _hand_matrix(arm, smap, side, at, h.get("dir", tuple(-back)), h.get("palm"))))
            for k in h.get("keys", []):
                mats.append((ctx.frame(float(k["t"])), _hand_matrix(arm, smap, side, targets.point(ctx, k["at"]),
                                                                     k.get("dir", h.get("dir", tuple(-back))),
                                                                     k.get("palm", h.get("palm")))))
            if not mats:
                raise BuildError(f"pose.{name}.hands.{side}: give at, rest or keys")
            mats.sort(key=lambda fm: fm[0])
            par_inv = tgt.parent.matrix_world.inverted() if tgt.parent else Matrix()
            locs, rots = [], []
            for fr, M in mats:
                Ml = par_inv @ M
                frames.append(fr)
                locs.append(tuple(Ml.translation))
                rots.append(tuple(Ml.to_quaternion()))
            K.key_vec(tgt, "location", frames, locs, interp="BEZIER")
            K.key_vec(tgt, "rotation_quaternion", frames, K.continuous(rots), interp="BEZIER")
            c = arm.pose.bones[wrist].constraints["mk_arm_ik"]
            K.key_prop(arm, f'pose.bones["{wrist}"].constraints["mk_arm_ik"].influence', [f0, f1], [0.0, 1.0])
            c.influence = 1.0
            fp = h.get("fingers")
            if fp:
                for b, q in _finger_quats(arm, smap, side, fp).items():
                    K.key_bone_quats(arm, b, [f0, f1], [(1, 0, 0, 0), tuple(q)], interp="BEZIER")
        for side, fp in (spec.get("fingers") or {}).items():
            for b, q in _finger_quats(arm, smap, side, fp).items():
                K.key_bone_quats(arm, b, [f0, f1], [(1, 0, 0, 0), tuple(q)], interp="BEZIER")
        out[name] = info
        ctx.log("pose", name)
    out["attached"] = attach_props(ctx)
    bpy.context.view_layer.update()
    return out


def attach_to_bone(obj, arm, bone, rel):
    """Bone-parent obj with `rel` its matrix in the bone's HEAD frame (Blender bone parenting uses the tail)."""
    obj.parent = arm
    obj.parent_type = "BONE"
    obj.parent_bone = bone
    obj.matrix_parent_inverse.identity()
    obj.matrix_basis = Matrix.Translation((0.0, -arm.data.bones[bone].length, 0.0)) @ rel


def attach_props(ctx):
    """[[prop]] attach = "cast:bone" (semantic or Blender name), offset = [x, y, z] (m), attach_rot = [x, y, z] (deg,
    XYZ Euler) in the bone's head frame: the prop rides on the bone.
    [[prop]] anchor_to = "cast": the card's use.anchor entries that name a `bone` (semantic) and an `object` are
    bone-parented to that cast member keeping their world placement at the settled base pose (earbuds in the ears,
    a cord on the chest)."""
    done = {}
    sc = bpy.context.scene
    for spec in ctx.data.get("prop", []):
        if spec.get("attach"):
            cast_name, _, bone = spec["attach"].partition(":")
            m = ctx.cast.get(cast_name)
            if m is None:
                raise BuildError(f"prop {spec['name']!r}: attach to unknown cast member {cast_name!r}")
            b = S.resolve_bone(m.arm, bone)
            rel = Matrix.Translation(Vector(spec.get("offset", (0, 0, 0)))) @ \
                Euler([math.radians(a) for a in spec.get("attach_rot", (0, 0, 0))]).to_matrix().to_4x4()
            attach_to_bone(ctx.props[spec["name"]].root, m.arm, b, rel)
            done[spec["name"]] = f"{cast_name}:{b}"
        if spec.get("anchor_to"):
            m = ctx.cast.get(spec["anchor_to"])
            if m is None:
                raise BuildError(f"prop {spec['name']!r}: anchor_to unknown cast member {spec['anchor_to']!r}")
            sc.frame_set(ctx.start + ctx.settle)
            n = 0
            for u in ctx.props[spec["name"]].card.get("use", {}).get("anchor", []):
                if not (u.get("bone") and u.get("object")):
                    continue
                ob = bpy.data.objects[u["object"]]
                mw = ob.matrix_world.copy()
                ob.parent = m.arm
                ob.parent_type = "BONE"
                ob.parent_bone = S.resolve_bone(m.arm, u["bone"])
                ob.matrix_world = mw
                n += 1
            done[f"{spec['name']}.anchors"] = n
    return done
