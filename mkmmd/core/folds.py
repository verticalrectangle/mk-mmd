"""Fold shapes a joint angle drives: the rig description's `folds` layer (mkmmd.core.rig), evaluated on sampled bone
rotations. numpy only, Blender-safe.

A fold is a shape (a vertex morph a part builder sculpts: the crook of a sleeve bunching, a back pulled into diagonal
folds) and a driver that says how much of it a pose shows, from 0 to 1 as an angle of the pose runs from `from` to
`to` degrees (smoothstep; `from` may be the larger, for a weight that grows as the angle closes):

    angle  {"kind": "angle", "bone": B, "ref": A, "dir": [x, y, z], "from": deg, "to": deg}
           the angle between bone B's direction (head to tail) and the direction `dir` (armature space at rest)
           carried along by bone A: an elbow's bend (B the forearm, A the upper arm, dir the upper arm's direction),
           an arm raised from hanging (B the upper arm, A the chest, dir straight down), an arm reaching forward,
           a lean forward
    twist  {"kind": "twist", "bone": B, "ref": A, "axis": [x, y, z], "from": deg, "to": deg}
           how far bone B has turned relative to bone A about `axis` (armature space at rest, carried by A), signed
           (right hand: about +z positive turns her to her left): a waist twisting

    weights(folds, D, dirs, M=identity) -> {morph: (F,) weights}

`D` holds each driving bone's rotation delta per frame (posed rotation times the rest rotation's inverse, (F, 3, 3), in
the world or any one frame); `dirs` each bone's rest direction (armature space, unit); `M` turns armature directions into
the frame of `D` (the armature's world rotation; identity when `D` is in armature space)."""
import numpy as np

KINDS = ("angle", "twist")


def _unit(v):
    v = np.asarray(v, float)
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12)


def ramp(theta, a, b):
    """0 at `a` degrees, 1 at `b` (either way round), smoothstep between."""
    t = np.clip((np.asarray(theta, float) - a) / (b - a), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def angle(D_bone, D_ref, bone_dir, ref_dir):
    """(F,) degrees between the bone's posed direction and `ref_dir` carried by the reference bone (both unit (3,), in
    the frame of the deltas D (F, 3, 3))."""
    a = _unit(np.einsum("fij,j->fi", D_bone, bone_dir))
    b = _unit(np.einsum("fij,j->fi", D_ref, ref_dir))
    return np.degrees(np.arccos(np.clip(np.einsum("fi,fi->f", a, b), -1.0, 1.0)))


def twist(D_bone, D_ref, axis):
    """(F,) signed degrees the bone has turned relative to the reference bone about `axis` (unit (3,), carried by the
    reference): the twist of the swing-twist split of the relative rotation, 2 atan2(q_xyz . axis, q_w)."""
    R = np.einsum("fji,fjk->fik", D_ref, D_bone)                # D_ref^T D_bone: B relative to A, in the rest frame
    w2 = np.maximum(1.0 + R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2], 0.0)                    # 4 q_w^2
    v = np.stack([R[:, 2, 1] - R[:, 1, 2], R[:, 0, 2] - R[:, 2, 0], R[:, 1, 0] - R[:, 0, 1]], 1)   # 4 q_w q_xyz
    return np.degrees(2.0 * np.arctan2(v @ np.asarray(axis, float), w2))


def check(fold):
    """Raise ValueError for a malformed fold entry."""
    d = fold.get("driver") or {}
    if not fold.get("morph") or d.get("kind") not in KINDS or not d.get("bone") or not d.get("ref"):
        raise ValueError(f"fold {fold.get('morph')!r}: needs a morph and a driver of kind {KINDS} with bone and ref")
    key = "dir" if d["kind"] == "angle" else "axis"
    if np.asarray(d.get(key, ()), float).shape != (3,) or float(d.get("from", 0.0)) == float(d.get("to", 0.0)):
        raise ValueError(f"fold {fold['morph']!r}: driver needs a 3-vector `{key}` and from != to")


def weights(folds, D, dirs, M=None):
    """{morph: (F,) weight 0..1} of every fold for the bone rotation deltas `D` {bone: (F, 3, 3)} (module docstring).
    Folds sharing a morph add up (capped at 1)."""
    M = np.eye(3) if M is None else np.asarray(M, float)
    out = {}
    for f in folds:
        check(f)
        d = f["driver"]
        Db, Da = np.asarray(D[d["bone"]], float), np.asarray(D[d["ref"]], float)
        if d["kind"] == "angle":
            th = angle(Db, Da, _unit(M @ np.asarray(dirs[d["bone"]], float)), _unit(M @ np.asarray(d["dir"], float)))
        else:
            th = twist(Db, Da, _unit(M @ np.asarray(d["axis"], float)))
        out[f["morph"]] = np.minimum(out.get(f["morph"], 0.0) + ramp(th, float(d["from"]), float(d["to"])), 1.0)
    return out
