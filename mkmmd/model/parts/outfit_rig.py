"""Bones, dynamic rigid bodies and joints for the outfit's chains (numpy only).

Same conventions as the hair part (PMX collision groups 0..15): the body's static colliders are group 0 and ignore
themselves; dynamic chain bodies of the outfit live in groups 7 (ribbons), 9 (skirt), 10 (sleeves), 11 (collar), the
first body of every chain in group 8 which also ignores group 0 (a root never fights the body it hangs from, exactly the
pairs mk's strand solver skips near a chain root). All dynamic groups ignore each other."""
import numpy as np

from ..part import Bone, Joint, RigidBody
from . import outfit_geo as G

GROUPS = {"ribbon": 7, "skirt": 9, "sleeve": 10, "coat": 11}
ROOT_GROUP = 8
DYNAMIC = (4, 5, 6, 7, 8, 9, 10, 11)

# per kind: mass, damping (lin, ang), friction, swing limits about the two bend axes (rad), twist (rad), spring
PHYSICS = {
    "skirt": dict(mass=0.25, damp=(0.93, 0.95), friction=0.45, swing=(0.85, 0.30), twist=0.10, spring=(10.0, 10.0, 6.0)),
    "ribbon": dict(mass=0.12, damp=(0.90, 0.93), friction=0.30, swing=(0.70, 0.70), twist=0.40, spring=(2.0, 2.0, 1.0)),
    "sleeve": dict(mass=0.10, damp=(0.92, 0.94), friction=0.35, swing=(0.70, 0.40), twist=0.15, spring=(8.0, 8.0, 4.0)),
    "coat": dict(mass=0.10, damp=(0.92, 0.94), friction=0.35, swing=(0.45, 0.30), twist=0.10, spring=(10.0, 10.0, 5.0)),
}


def euler_xyz(R):
    """XYZ Euler angles (rad, Blender convention R = Rz Ry Rx) of a rotation matrix."""
    ry = -np.arcsin(np.clip(R[2, 0], -1.0, 1.0))
    if abs(R[2, 0]) < 0.999999:
        rx = np.arctan2(R[2, 1], R[2, 2])
        rz = np.arctan2(R[1, 0], R[0, 0])
    else:
        rx = np.arctan2(-R[1, 2], R[1, 1])
        rz = 0.0
    return (float(rx), float(ry), float(rz))


def euler_matrix(e):
    """Rotation matrix of XYZ Euler angles (rad, Blender convention R = Rz Ry Rx): the inverse of `euler_xyz`."""
    rx, ry, rz = (float(x) for x in e)
    cx, sx, cy, sy, cz, sz = np.cos(rx), np.sin(rx), np.cos(ry), np.sin(ry), np.cos(rz), np.sin(rz)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def body_sdf(points, rb):
    """Signed distance (negative inside) of points (n, 3) to the shape of a RigidBody: a sphere (r), a capsule (r, straight
    height) along its local Z, or a box (half extents), placed at `location` with the XYZ Euler `rotation` (model space)."""
    P = np.atleast_2d(np.asarray(points, float))
    L = (P - np.asarray(rb.location, float)) @ euler_matrix(rb.rotation)          # into the body's frame
    if rb.shape == "sphere":
        return np.linalg.norm(L, axis=1) - rb.size[0]
    if rb.shape == "capsule":
        z = np.clip(L[:, 2], -0.5 * rb.size[1], 0.5 * rb.size[1])
        return np.linalg.norm(L - np.stack([np.zeros_like(z), np.zeros_like(z), z], 1), axis=1) - rb.size[0]
    q = np.abs(L) - np.asarray(rb.size, float)
    return np.linalg.norm(np.maximum(q, 0.0), axis=1) + np.minimum(q.max(1), 0.0)


def union_sdf(points, bodies):
    """Smallest `body_sdf` over several bodies (+inf without any): the distance to the union of their shapes."""
    out = np.full(len(np.atleast_2d(points)), np.inf)
    for rb in bodies:
        out = np.minimum(out, body_sdf(points, rb))
    return out


def chain_clearance(pts, radius, bodies, samples=7):
    """Clearance (m) of every capsule of a chain through `pts` (k + 1 points, bone k = pts[k] -> pts[k + 1], capsule radius
    `radius`, scalar or per bone) from the union of `bodies`: the smallest axis-to-shape distance minus the radius, per bone
    (negative = overlap)."""
    pts = np.asarray(pts, float)
    rad = np.broadcast_to(np.asarray(radius, float), (len(pts) - 1,))
    t = np.linspace(0.0, 1.0, samples)[:, None]
    out = np.empty(len(pts) - 1)
    for k in range(len(pts) - 1):
        seg = pts[k] + t * (pts[k + 1] - pts[k])
        out[k] = union_sdf(seg, bodies).min() - rad[k]
    return out


def frame_z(axis, x_hint):
    """Rotation matrix with Z = axis and X as close to x_hint as possible (columns x, y, z)."""
    z = np.asarray(axis, float)
    z = z / max(np.linalg.norm(z), 1e-12)
    x = np.asarray(x_hint, float) - z * np.dot(x_hint, z)
    if np.linalg.norm(x) < 1e-6:
        x = np.cross(z, [0.0, 1.0, 0.0] if abs(z[1]) < 0.9 else [1.0, 0.0, 0.0])
    x = x / np.linalg.norm(x)
    y = np.cross(z, x)
    return np.stack([x, y, z], 1)


class Rig:
    """Accumulates bones, bodies and joints."""

    def __init__(self):
        self.bones, self.bodies, self.joints = [], [], []

    def chain(self, names, pts, anchor, kind, radius, anchor_body=None, x_hint=(1.0, 0.0, 0.0), name_en=None,
              physics=None, dynamic=True, layer=0):
        """A line of connected bones names[k]: pts[k] -> pts[k + 1] (len(pts) == len(names) + 1), the first one a child of
        `anchor`; a capsule body for every bone and a spring joint to its predecessor (the first to `anchor_body`)."""
        pts = np.asarray(pts, float)
        n = len(names)
        ph = dict(PHYSICS[kind])
        ph.update(physics or {})
        grp = GROUPS[kind]
        rad = np.broadcast_to(np.asarray(radius, float), (n,))
        prev = anchor_body
        for k in range(n):
            a, b = pts[k], pts[k + 1]
            self.bones.append(Bone(name=names[k], head=tuple(float(x) for x in a), tail=tuple(float(x) for x in b),
                                   parent=names[k - 1] if k else anchor, tail_bone=names[k + 1] if k + 1 < n else "",
                                   name_en=(name_en[k] if name_en else ""), layer=layer))
            if not dynamic:
                continue
            length = float(np.linalg.norm(b - a))
            r = float(min(rad[k], 0.45 * length))
            R = frame_z(b - a, x_hint)
            root = k == 0
            self.bodies.append(RigidBody(
                name=names[k], bone=names[k], shape="capsule", size=(r, max(length - 2 * r, 0.002), 0.0),
                location=tuple(float(x) for x in (a + b) / 2), rotation=euler_xyz(R), mode="dynamic",
                group=ROOT_GROUP if root else grp, no_collide=((0,) + DYNAMIC) if root else DYNAMIC,
                mass=ph["mass"] * (0.5 if root else 1.0), damping=ph["damp"], friction=ph["friction"]))
            if prev is not None:
                sx, sy = ph["swing"]
                tw = ph["twist"]
                kx, ky, kz = ph["spring"]
                self.joints.append(Joint(
                    name=f"J_{names[k]}", a=prev, b=names[k], location=tuple(float(x) for x in a), rotation=euler_xyz(R),
                    rot_lo=(-sx, -sy, -tw), rot_hi=(sx, sy, tw), spring_rot=(kx, ky, kz)))
            prev = names[k]
        return names


def ring_weights(theta, z, names, z_nodes, z_top, w_top, columns):
    """Skin weights for the vertices of a surface hung from chains arranged in columns around an axis: vertex i is at angle
    theta[i] and "height" z[i] (larger = nearer the root; any monotonic coordinate along the chains); names[r][c] are the
    chain bones (row r = 0.., column c); z_nodes[r] the height of row r's head and z_top the root height. Above the first
    node the body's own weights `w_top` blend (smoothstep) into row 1; between nodes linear; below the last node row R;
    between the two nearest columns smooth-linear (columns at theta_c = c * 2pi / columns)."""
    n = len(z)
    R = len(z_nodes)
    nodes = np.concatenate([[z_top], z_nodes])
    # continuous row index p: 0 at the seam, 1..R at the nodes (z decreasing)
    p = np.interp(-np.asarray(z), -nodes, np.arange(R + 1.0))
    i = np.minimum(np.floor(p).astype(int), R - 1)
    fr = p - i
    fr = np.where(p >= R, 1.0, fr)
    i = np.where(p >= R, R - 1, i)
    Rw = np.zeros((n, R + 1))                                  # column 0: the body, 1..R: rows
    t = np.where(i == 0, G.smoothstep(fr), fr)
    rows = np.arange(n)
    Rw[rows, i] += 1.0 - t
    Rw[rows, i + 1] += t
    u = (np.asarray(theta) % G.TAU) / (G.TAU / columns)
    c0 = np.floor(u).astype(int) % columns
    c1 = (c0 + 1) % columns
    f = u - np.floor(u)
    fw = 0.5 * (f + G.smoothstep(f))
    out = {}
    for b, a in w_top.items():
        out[b] = np.asarray(a, float) * Rw[:, 0]
    for r in range(R):
        wr = Rw[:, r + 1]
        if not wr.any():
            continue
        for c in range(columns):
            w = wr * ((1.0 - fw) * (c0 == c) + fw * (c1 == c))
            if w.any():
                out[names[r][c]] = out.get(names[r][c], 0.0) + w
    return out
