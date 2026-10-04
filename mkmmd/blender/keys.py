"""Keying helpers: fast fcurve writes, world-space rotations to bone-local keys, morphs, object properties.

World-axis composition (docs/design.md: Posing): a bone's posed world delta is D_bone = D_parent . q where q is the
rotation it adds in WORLD axes; the bone-local quaternion to key is B^-1 q B with B the bone's rest rotation (armature
world rotation included). So a chain such as upper_body -> upper_body2 -> neck -> head is keyed exactly from desired
world deltas with q_child = D_parent^-1 . D_want."""
import numpy as np
from mathutils import Matrix, Quaternion, Vector

INTERP = {"CONSTANT": 0, "LINEAR": 1, "BEZIER": 2}


def ensure_action(id_data, name=None):
    ad = id_data.animation_data or id_data.animation_data_create()
    if ad.action is None:
        import bpy
        ad.action = bpy.data.actions.new(name or f"{id_data.name}_act")
    return ad.action


def set_fcurve(id_data, path, index, frames, values, interp="BEZIER", replace=True, group=None):
    """Write keys (frames, values) to one fcurve in one go. replace=False merges: existing keys inside the new frame
    range are removed first, keys outside it stay."""
    act = ensure_action(id_data)
    fc = act.fcurves.find(path, index=index)
    if fc is None:
        fc = act.fcurves.new(path, index=index, action_group=group) if group else act.fcurves.new(path, index=index)
    frames = np.asarray(frames, float)
    values = np.asarray(values, float)
    if not replace and len(fc.keyframe_points):
        old = np.empty(2 * len(fc.keyframe_points))
        fc.keyframe_points.foreach_get("co", old)
        of, ov = old[0::2], old[1::2]
        keep = (of < frames.min() - 1e-6) | (of > frames.max() + 1e-6)
        frames = np.concatenate([of[keep], frames])
        values = np.concatenate([ov[keep], values])
        order = np.argsort(frames, kind="stable")
        frames, values = frames[order], values[order]
    fc.keyframe_points.clear()
    fc.keyframe_points.add(len(frames))
    co = np.empty(2 * len(frames))
    co[0::2], co[1::2] = frames, values
    fc.keyframe_points.foreach_set("co", co)
    fc.keyframe_points.foreach_set("interpolation", [INTERP[interp]] * len(frames))
    if interp == "BEZIER":                               # RNA enum: FREE 0, AUTO 1, VECTOR 2, ALIGNED 3, AUTO_CLAMPED 4
        fc.keyframe_points.foreach_set("handle_left_type", [4] * len(frames))
        fc.keyframe_points.foreach_set("handle_right_type", [4] * len(frames))
    fc.update()
    return fc


def rest_rot(arm, bone):
    return (arm.matrix_world.to_3x3() @ arm.data.bones[bone].matrix_local.to_3x3()).normalized()


def local_q(arm, bone, q_world):
    B = rest_rot(arm, bone)
    return (B.inverted() @ q_world.to_matrix() @ B).to_quaternion()


def local_vec(arm, bone, v_world):
    return rest_rot(arm, bone).inverted() @ Vector(v_world)


def continuous(quats):
    """Flip signs so consecutive quaternions (n, 4) stay on one hemisphere (smooth interpolation)."""
    q = np.array(quats, float)
    for i in range(1, len(q)):
        if np.dot(q[i], q[i - 1]) < 0:
            q[i] = -q[i]
    return q


def key_bone_quats(arm, bone, frames, quats, interp="LINEAR", replace=True):
    """Bone-local rotation keys; quats (n, 4) w x y z."""
    pb = arm.pose.bones[bone]
    pb.rotation_mode = "QUATERNION"
    q = continuous(quats)
    path = f'pose.bones["{bone}"].rotation_quaternion'
    for c in range(4):
        set_fcurve(arm, path, c, frames, q[:, c], interp=interp, replace=replace, group=bone)


def key_bone_world(arm, bone, frames, q_worlds, interp="LINEAR", replace=True):
    """Keys from world-axis rotations q (mathutils Quaternions or (n, 4) arrays) added by the bone (see module doc)."""
    B = rest_rot(arm, bone)
    Bi = B.inverted()
    loc = []
    for q in q_worlds:
        q = q if isinstance(q, Quaternion) else Quaternion(q)
        loc.append(tuple((Bi @ q.to_matrix() @ B).to_quaternion()))
    key_bone_quats(arm, bone, frames, loc, interp=interp, replace=replace)


def key_bone_arm(arm, bone, frames, q_arms, interp="LINEAR", replace=True):
    """Keys from rotations q expressed in the ARMATURE's axes (independent of where the armature is placed or how
    it moves): local = R^-1 q R with R the bone's armature-space rest rotation."""
    R = arm.data.bones[bone].matrix_local.to_3x3().normalized()
    Ri = R.inverted()
    loc = []
    for q in q_arms:
        q = q if isinstance(q, Quaternion) else Quaternion(q)
        loc.append(tuple((Ri @ q.to_matrix() @ R).to_quaternion()))
    key_bone_quats(arm, bone, frames, loc, interp=interp, replace=replace)


def key_bone_locs(arm, bone, frames, world_offsets, interp="BEZIER", replace=True):
    """Location keys from WORLD offsets (converted to the bone's local axes)."""
    B = rest_rot(arm, bone).inverted()
    v = np.array([tuple(B @ Vector(w)) for w in world_offsets])
    path = f'pose.bones["{bone}"].location'
    for c in range(3):
        set_fcurve(arm, path, c, frames, v[:, c], interp=interp, replace=replace, group=bone)


def key_morph(meshes, name, frames, values, interp="BEZIER", replace=True):
    """Shape-key values on every mesh that has the key. Returns how many meshes were keyed."""
    n = 0
    for m in meshes:
        sk = m.data.shape_keys
        if sk and name in sk.key_blocks:
            set_fcurve(sk, f'key_blocks["{name}"].value', 0, frames, values, interp=interp, replace=replace)
            n += 1
    return n


def key_prop(id_data, path, frames, values, index=0, interp="BEZIER", replace=True):
    return set_fcurve(id_data, path, index, frames, values, interp=interp, replace=replace)


def key_vec(obj, path, frames, vecs, interp="BEZIER", replace=True):
    v = np.asarray(vecs, float)
    for c in range(v.shape[1]):
        set_fcurve(obj, path, c, frames, v[:, c], interp=interp, replace=replace)


def mat(m):
    """numpy (4, 4) -> mathutils Matrix."""
    return Matrix([list(r) for r in np.asarray(m, float)])
