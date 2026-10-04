"""Static rigid bodies of the body part: the colliders mk's hair/skirt solver (and MMD users) run into. They follow their
bone, sit in group 0 and ignore group 0 (no body collides with another body), and hug the skin: radii come from the
sweep rings of the mesh, not from guesses.

Conventions of the contract: a capsule's height axis is +Z at rotation 0 (size = (radius, straight height, 0));
a box's size is its half extents along the model axes at rotation 0; rotation is an XYZ Euler in radians."""
import numpy as np

from ..part import RigidBody
from .body_geom import unit

SIDE_JP = {"L": "左", "R": "右"}


def euler_xyz(R):
    """XYZ Euler angles (rad) of a rotation matrix, Blender's convention (R = Rz Ry Rx)."""
    sy = -R[2, 0]
    if abs(sy) < 0.99999:
        ry = np.arcsin(sy)
        rx = np.arctan2(R[2, 1], R[2, 2])
        rz = np.arctan2(R[1, 0], R[0, 0])
    else:
        ry = np.copysign(np.pi / 2, sy)
        rx = np.arctan2(-R[1, 2], R[1, 1])
        rz = 0.0
    return (float(rx), float(ry), float(rz))


def frame_z_to(axis):
    """A rotation matrix whose +Z column is `axis` (the smallest turn from +Z)."""
    z = unit(axis)
    up = np.array([0.0, 0.0, 1.0])
    v = np.cross(up, z)
    c = float(up @ z)
    if np.linalg.norm(v) < 1e-9:
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + K + K @ K * (1.0 / (1.0 + c))


def capsule(name, bone, a, b, r, **kw):
    a, b = np.asarray(a, float), np.asarray(b, float)
    length = float(np.linalg.norm(b - a))
    h = max(length - 2.0 * r, 0.004)
    return RigidBody(name=name, bone=bone, shape="capsule", size=(float(r), float(h), 0.0),
                     location=tuple(float(x) for x in 0.5 * (a + b)), rotation=euler_xyz(frame_z_to(b - a)), **kw)


def sphere(name, bone, c, r, **kw):
    return RigidBody(name=name, bone=bone, shape="sphere", size=(float(r), 0.0, 0.0),
                     location=tuple(float(x) for x in c), **kw)


def box(name, bone, c, half, rot=(0.0, 0.0, 0.0), **kw):
    return RigidBody(name=name, bone=bone, shape="box", size=tuple(float(x) for x in half),
                     location=tuple(float(x) for x in c), rotation=tuple(float(x) for x in rot), **kw)


def _seg_dist(P, a, b):
    ab = b - a
    t = np.clip(((P - a) @ ab) / max(float(ab @ ab), 1e-12), 0.0, 1.0)
    return np.linalg.norm(P - (a + t[:, None] * ab), axis=1)


def ring_s(sh):
    """Arclength of every real ring of a tube shell (the first vertex of each ring carries it)."""
    return np.array([sh.s[sh.ring_index[i, 0]] for i in range(len(sh.ring_index))])


def limb_capsule(name, bone, sh, s0, s1, q=0.93, trim=0.05, **kw):
    """A capsule along the flesh of a tube shell between arclengths s0 and s1: its axis joins the centroids of the
    rings at the two ends, its radius is the `q` quantile of the skin's distance to that axis (end rings trimmed)."""
    ss = ring_s(sh)
    cen = np.array([sh.verts[sh.ring_index[i]].mean(axis=0) for i in range(len(ss))])
    ia, ib = int(np.argmin(np.abs(ss - s0))), int(np.argmin(np.abs(ss - s1)))
    a, b = cen[ia], cen[ib]
    lo, hi = s0 + trim * (s1 - s0), s1 - trim * (s1 - s0)
    pts = np.concatenate([sh.verts[sh.ring_index[i]] for i in range(len(ss)) if lo <= ss[i] <= hi])
    r = float(np.quantile(_seg_dist(pts, a, b), q))
    return capsule(name, bone, a, b, r, **kw)


def box_fit(name, bone, pts, axes, q=0.98, pad=0.0015, **kw):
    """A box around points `pts` in the frame whose columns are `axes` (rotation matrix): centre of the extents, half
    extents from the q / 1-q quantiles (so a few strays do not fatten it), plus `pad`."""
    q_ = pts @ axes
    lo, hi = np.quantile(q_, 1 - q, axis=0), np.quantile(q_, q, axis=0)
    c = axes @ (0.5 * (lo + hi))
    return box(name, bone, c, 0.5 * (hi - lo) + pad, rot=euler_xyz(axes), **kw)


def make_bodies(shape, shells, bone_names):
    """The list of RigidBody for the body part, fitted to the skin. `bone_names`: PMX names of the bones that exist.
    Left bodies are fitted on the left shells and mirrored for the right side (same size, mirrored placement)."""
    L = shape.land
    out = []
    byname = {sh.name: sh for sh in shells}

    def keep(rb):
        if rb.bone in bone_names:
            rb.group = 0
            rb.no_collide = (0,)
            rb.mass = 1.0
            out.append(rb)

    head, tip = L["head"], L["head_tip"]
    keep(sphere("col_head", "頭", head + np.array([0.0, -0.0103, 0.1007]), 0.094))
    nk = L["neck"]
    keep(capsule("col_neck", "首", nk + np.array([0.0, 0.004, -0.004]), head + np.array([0.0, 0.0, 0.012]), 0.027))
    # torso boxes (axis aligned) from the torso skin between the bones' heights
    V = byname["torso"].verts
    ub, ub2, ntop = L["upper_body"][2], L["upper_body2"][2], L["neck"][2]
    eye = np.eye(3)
    for name, bone, z0, z1 in (("col_lower_body", "下半身", V[:, 2].min() + 0.012, ub + 0.004),
                               ("col_upper_body", "上半身", ub + 0.004, ub2 + 0.004),
                               ("col_upper_body2", "上半身2", ub2 + 0.004, ntop + 0.002)):
        sel = (V[:, 2] >= z0) & (V[:, 2] <= z1)
        keep(box_fit(name, bone, V[sel], eye))
    # left side, then mirrored
    sh_, ar, el, wr = (L[f"{k}.L"] for k in ("shoulder", "arm", "elbow", "wrist"))
    arm, leg = byname["arm_L"], byname["leg_L"]
    la, lf = arm.info["la"], arm.info["lf"]
    a_, r_, n_ = shape.frame
    pts = [arm.verts[arm.s > la + lf]]
    for f in ("index", "middle", "ring", "little"):
        fs = byname[f"{f}_L"]
        pts.append(fs.verts[fs.s < fs.info["s"][2] + 0.002])
    fv = leg.verts[leg.verts[:, 2] < L["ankle.L"][2] - 0.012]
    left = [
        capsule("col_shoulder_L", "左肩", sh_ + np.array([0.0, 0.0, 0.012]), ar + np.array([0.0, 0.0, 0.004]), 0.034),
        limb_capsule("col_arm_L", "左腕", arm, 0.0, la),
        limb_capsule("col_forearm_L", "左ひじ", arm, la, la + lf),
        box_fit("col_hand_L", "左手首", np.concatenate(pts, 0), np.stack([a_, r_, -n_], 1)),
        limb_capsule("col_thigh_L", "左足", leg, 0.0, leg.info["sK"]),
        limb_capsule("col_shin_L", "左ひざ", leg, leg.info["sK"], leg.info["sA"]),
        box_fit("col_foot_L", "左足首", fv, np.eye(3)),
    ]
    for rb in left:
        keep(rb)
        keep(mirror_body(rb, rb.name[:-1] + "R", rb.bone.replace("左", "右")))
    return out


def mirror_body(rb, name=None, bone=None):
    """The x-mirror of a left rigid body (same size, mirrored location and rotation, `name`/`bone` replaced)."""
    from dataclasses import replace
    rx, ry, rz = rb.rotation
    return replace(rb, name=name or rb.name, bone=bone or rb.bone,
                   location=(-rb.location[0], rb.location[1], rb.location[2]), rotation=(rx, -ry, -rz))
