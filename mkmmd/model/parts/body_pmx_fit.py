"""Geometry of an imported body (numpy only; Blender-safe): what the other parts need to know about a skin surface that
arrives as arrays (vertices, triangles, weights by bone name) instead of being built from tables.

 - `bone_groups` / `vertex_groups` / `face_groups`: the region of every bone (torso, neck, head, shoulder, arm, hand, leg),
   the weight each vertex carries on each region, and the region that owns each triangle;
 - `hidden_faces`: the triangles an outfit covers in every pose (the skin that must not poke through a garment) from a few
   numbers: a collar line on the neck, how far the sleeves reach down the forearm, the height of the skirt on the legs;
 - `landmarks`: the standard bone heads by semantic name plus the points the garment, hair and tail builders ask for;
 - `raycast`, `slice_z`: surface queries (tail root on the lower back, the bust, sections of the torso and the neck);
 - `fit_bodies`: static colliders (`col_*`) that hug the skin, left side fitted on both sides mirrored so they are symmetric.

Model space: metres, Z up, the character faces -Y, her left is +X."""
import numpy as np

from ...core import bonemap
from ..part import RigidBody

SIDE_JP = {"L": "左", "R": "右"}
TORSO = ("center", "groove", "waist", "lower_body", "upper_body", "upper_body2", "upper_body3", "waist_cancel")
GROUP_STEMS = {
    "torso": TORSO,
    "neck": ("neck",),
    "head": ("head", "eyes", "eye", "eye_l", "eye_r"),
    "shoulder": ("shoulder", "shoulder_p", "shoulder_c"),
    "arm": ("arm", "arm_twist", "arm_twist1", "arm_twist2", "arm_twist3", "elbow", "wrist_twist", "wrist_twist1",
            "wrist_twist2", "wrist_twist3"),
    "hand": ("wrist", "thumb0", "thumb1", "thumb2", "index1", "index2", "index3", "middle1", "middle2", "middle3",
             "ring1", "ring2", "ring3", "little1", "little2", "little3"),
    "leg": ("leg", "knee", "ankle", "toe", "leg_d", "knee_d", "ankle_d", "toe_ex"),
}
GROUPS = tuple(GROUP_STEMS)
_STEM_GROUP = {s: g for g, stems in GROUP_STEMS.items() for s in stems}


def unit(v, eps=1e-12):
    v = np.asarray(v, float)
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), eps)


# ---------------------------------------------------------------- regions
def bone_groups(names):
    """{bone name: region} for the names that belong to one: standard bones by their semantic stem (腕捩2 -> arm), other
    names by what they say (a bone with 足 in it is a leg bone, 指 a finger, 捩 or 腕 an arm bone). Unknown bones are left
    out."""
    names = list(names)
    smap = bonemap.build_map({n: n for n in names})
    out = {}
    for sem, b in smap.items():
        stem = sem.split(".")[0]
        if stem in _STEM_GROUP:
            out[b] = _STEM_GROUP[stem]
    for n in names:
        if n in out:
            continue
        s = bonemap.norm(n)
        if "足" in s or "つま先" in s:
            out[n] = "leg"
        elif "指" in s:
            out[n] = "hand"
        elif "捩" in s or "腕" in s or "ひじ" in s:
            out[n] = "arm"
    return out


def vertex_groups(weights, groups, n):
    """(n, len(GROUPS)) weight of each vertex on each region (rows of vertices without a known bone stay 0)."""
    G = np.zeros((n, len(GROUPS)))
    for b, w in weights.items():
        g = groups.get(b)
        if g is not None:
            G[:, GROUPS.index(g)] += np.asarray(w, float)
    return G


def face_groups(tris, G):
    """Index into GROUPS of the region that owns each triangle (summed weight of its three vertices)."""
    t = np.asarray(tris, int)
    return np.argmax(G[t[:, 0]] + G[t[:, 1]] + G[t[:, 2]], axis=1)


def forearm_frame(land, side):
    """(elbow, wrist, unit axis elbow -> wrist, length) of one arm."""
    e, w = np.asarray(land[f"elbow.{side}"], float), np.asarray(land[f"wrist.{side}"], float)
    d = w - e
    ln = float(np.linalg.norm(d))
    return e, w, d / ln, ln


def hidden_faces(verts, tris, G, land, hide):
    """Boolean per triangle: True where an outfit covers the skin in every pose.

    `hide` (all optional): groups (default torso, shoulder, arm), neck_z (neck triangles whose centre is below it are
    covered: the line inside the collar), wrist_back (metres: the arm skin stops this far before the wrist head, inside the
    sleeve, so the cuff never shows a hole), leg_z (leg triangles above it are covered: under the skirt). Hand, head and the
    legs below leg_z stay visible."""
    V, T = np.asarray(verts, float), np.asarray(tris, int)
    fg = face_groups(T, G)
    c = V[T].mean(axis=1)
    covered = set(hide.get("groups", ("torso", "shoulder", "arm")))
    hidden = np.isin(fg, [GROUPS.index(g) for g in covered if g in GROUPS])
    if "arm" in covered:                                                # the forearm end stays, inside the sleeve
        back = float(hide.get("wrist_back", 0.035))
        for side, sx in (("L", 1.0), ("R", -1.0)):
            if f"elbow.{side}" not in land or f"wrist.{side}" not in land:
                continue
            e, w, u, ln = forearm_frame(land, side)
            s = (c - e) @ u
            near = (fg == GROUPS.index("arm")) & (c[:, 0] * sx > 0) & (s > ln - back)
            hidden &= ~near
    if hide.get("neck_z") is not None:
        neck = fg == GROUPS.index("neck")
        hidden |= neck & (c[:, 2] < float(hide["neck_z"]))
    if hide.get("leg_z") is not None:
        leg = fg == GROUPS.index("leg")
        hidden |= leg & (c[:, 2] > float(hide["leg_z"]))
    return hidden


# ---------------------------------------------------------------- landmarks
def tail_of(bone, by_name):
    if bone.tail is not None:
        return np.asarray(bone.tail, float)
    if bone.tail_bone and bone.tail_bone in by_name:
        return np.asarray(by_name[bone.tail_bone].head, float)
    return np.asarray(bone.head, float) + np.array([0.0, 0.0, 0.05])


def landmarks(bones, stems=None):
    """Semantic name -> bone head (m) for every standard bone of `bones` (part.Bone objects), plus `head_tip` (the tail of the
    head bone), `waist` (midway between lower_body and upper_body when the rig has none) and, for the semantic stems in
    `stems` ({"toe": "足つま先"}: stem -> the bone name without its side), the extra bones the standard map lacks: their
    head under the semantic name and `<name>_end.L` at their tail (not below the floor)."""
    by = {b.name: b for b in bones}
    smap = bonemap.build_map({b.name: b.name for b in bones})
    land = {sem: np.asarray(by[n].head, float) for sem, n in smap.items()}
    if "head" in smap:
        land["head_tip"] = tail_of(by[smap["head"]], by)
    if "waist" not in land and "lower_body" in land and "upper_body" in land:
        land["waist"] = 0.5 * (land["lower_body"] + land["upper_body"])
    for sem, stem in (stems or {}).items():
        for side, jp in SIDE_JP.items():
            for cand in (f"{jp}{stem}", f"{stem}{jp}"):
                if cand in by:
                    b = by[cand]
                    land[f"{sem}.{side}"] = np.asarray(b.head, float)
                    end = tail_of(b, by).copy()
                    end[2] = max(end[2], 0.0)
                    land[f"{sem}_end.{side}"] = end
                    break
    return land


# ---------------------------------------------------------------- surface queries
def raycast(verts, tris, origin, direction):
    """Nearest hit of the ray with the triangles: (t, triangle index, (u, v) barycentric weights of vertices 1 and 2) or
    None (Moller-Trumbore, all triangles at once)."""
    V = np.asarray(verts, float)
    T = np.asarray(tris, int)
    o, d = np.asarray(origin, float), unit(direction)
    a, b, c = V[T[:, 0]], V[T[:, 1]], V[T[:, 2]]
    e1, e2 = b - a, c - a
    p = np.cross(d, e2)
    det = np.einsum("ij,ij->i", e1, p)
    ok = np.abs(det) > 1e-14
    inv = np.where(ok, 1.0 / np.where(ok, det, 1.0), 0.0)
    s = o - a
    u = np.einsum("ij,ij->i", s, p) * inv
    q = np.cross(s, e1)
    v = (q @ d) * inv
    t = np.einsum("ij,ij->i", e2, q) * inv
    hit = ok & (u >= -1e-9) & (v >= -1e-9) & (u + v <= 1 + 1e-9) & (t > 1e-9)
    if not hit.any():
        return None
    k = int(np.argmin(np.where(hit, t, np.inf)))
    return float(t[k]), k, (float(u[k]), float(v[k]))


def hit_point_normal(verts, tris, normals, origin, direction):
    """(point, outward unit normal interpolated from the vertex normals) of the first surface hit, or (None, None)."""
    h = raycast(verts, tris, origin, direction)
    if h is None:
        return None, None
    t, k, (u, v) = h
    i0, i1, i2 = tris[k]
    n = normals[i0] * (1 - u - v) + normals[i1] * u + normals[i2] * v
    return np.asarray(origin, float) + unit(direction) * t, unit(n)


def slice_z(verts, tris, z):
    """Segments (S, 2, 3) where the plane height z cuts the triangles."""
    V, T = np.asarray(verts, float), np.asarray(tris, int)
    d = V[T, 2] - z
    segs = []
    for i, j in ((0, 1), (1, 2), (2, 0)):
        cross = (d[:, i] * d[:, j]) < 0
        t = np.where(cross, d[:, i] / np.where(cross, d[:, i] - d[:, j], 1.0), 0.0)
        segs.append((cross, V[T[:, i]] + (V[T[:, j]] - V[T[:, i]]) * t[:, None]))
    out = []
    for (ca, pa), (cb, pb), (cc, pc) in [(segs[0], segs[1], segs[2])]:
        for x, y, m in ((pa, pb, ca & cb), (pb, pc, cb & cc), (pc, pa, cc & ca)):
            out.append(np.stack([x[m], y[m]], 1))
    return np.concatenate(out, 0) if out else np.zeros((0, 2, 3))


def section_extent(verts, tris, z):
    """(x_min, x_max, y_min, y_max) of the section of a triangle set at height z, or None when it does not reach it."""
    s = slice_z(verts, tris, z)
    if not len(s):
        return None
    p = s.reshape(-1, 3)
    return float(p[:, 0].min()), float(p[:, 0].max()), float(p[:, 1].min()), float(p[:, 1].max())


def torso_profile(verts, tris, fg, z0, z1, n=24):
    """The cuts of the torso (triangles of the torso region only, arms excluded) as the dict the hair builders read:
    rows (z), half_width, y_front, y_back (front = -Y)."""
    keep = tris[fg == GROUPS.index("torso")]
    zs, hw, yf, yb = [], [], [], []
    for z in np.linspace(z0, z1, n):
        e = section_extent(verts, keep, z)
        if e is None:
            continue
        zs.append(float(z))
        hw.append(0.5 * (e[1] - e[0]))
        yf.append(e[2])
        yb.append(e[3])
    return {"rows": np.array(zs), "half_width": np.array(hw), "y_front": np.array(yf), "y_back": np.array(yb)}


def neck_ring(verts, tris, fg, z, n=32):
    """The neck at height z as the plain ellipse the head builders understand: {z, center, rx, ry, ring (n, 3) with point k at
    angle 2 pi k / n from the front (-Y) towards +X, n, start, dir}."""
    keep = tris[fg == GROUPS.index("neck")]
    e = section_extent(verts, keep, z)
    if e is None:
        return None
    cx, cy = 0.5 * (e[0] + e[1]), 0.5 * (e[2] + e[3])
    rx, ry = 0.5 * (e[1] - e[0]), 0.5 * (e[3] - e[2])
    th = 2 * np.pi * np.arange(n) / n
    ring = np.stack([cx + rx * np.sin(th), cy - ry * np.cos(th), np.full(n, z)], 1)
    return dict(z=float(z), center=(float(cx), float(cy)), rx=float(rx), ry=float(ry), ring=ring, n=n, start="front",
                dir="ccw_from_above")


def back_point(verts, tris, normals, z, x=0.0, y_far=0.6):
    """Skin point and outward normal on the BACK at height z and lateral position x (ray from behind, towards the front)."""
    return hit_point_normal(verts, tris, normals, (x, y_far, z), (0.0, -1.0, 0.0))


def front_point(verts, tris, normals, z, x=0.0, y_far=-0.6):
    return hit_point_normal(verts, tris, normals, (x, y_far, z), (0.0, 1.0, 0.0))


def extra_landmarks(verts, tris, normals, land, fg, tail_dz=0.040, bust_dz=0.058, root_x=0.022):
    """The points the hair, tail and garment builders ask the body for, taken from the real skin: `tail_root` (on the lower
    back, `tail_dz` below the lower-body bone) with `tail_root.L/R` at x = +-`root_x`, `bust_front` (the most forward point
    of the chest within a hand's breadth of `bust_dz` above upper_body2), and the outward unit normal of the tail root."""
    out = {}
    z_t = float(land["lower_body"][2]) - tail_dz
    p, n = back_point(verts, tris, normals, z_t)
    if p is None:
        raise ValueError(f"no skin behind the body at z = {z_t:.3f}: cannot place the tail root")
    if n[1] < 0:
        n = -n
    out["tail_root"] = p
    for side, sx in (("L", 1.0), ("R", -1.0)):
        q, _ = back_point(verts, tris, normals, z_t, x=sx * root_x)
        out[f"tail_root.{side}"] = q if q is not None else p + np.array([sx * root_x, 0.0, 0.0])
    z_b = float(land["upper_body2"][2]) + bust_dz
    best = None
    for x in (-0.05, -0.035, -0.02, 0.0, 0.02, 0.035, 0.05):
        for dz in (-0.02, 0.0, 0.02):
            q, _ = front_point(verts, tris, normals, z_b + dz, x=x)
            if q is not None and (best is None or q[1] < best[1]):
                best = q
    out["bust_front"] = np.array([0.0, best[1], z_b]) if best is not None else np.array([0.0, -0.1, z_b])
    return out, n


# ---------------------------------------------------------------- colliders
def _seg_dist(P, a, b):
    ab = b - a
    t = np.clip(((P - a) @ ab) / max(float(ab @ ab), 1e-12), 0.0, 1.0)
    return np.linalg.norm(P - (a + t[:, None] * ab), axis=1), t


def euler_xyz(R):
    """XYZ Euler angles (rad) of a rotation matrix, Blender's convention (R = Rz Ry Rx)."""
    sy = -R[2, 0]
    if abs(sy) < 0.99999:
        return (float(np.arctan2(R[2, 1], R[2, 2])), float(np.arcsin(sy)), float(np.arctan2(R[1, 0], R[0, 0])))
    return (float(np.arctan2(-R[1, 2], R[1, 1])), float(np.copysign(np.pi / 2, sy)), 0.0)


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


def capsule(name, bone, a, b, r):
    a, b = np.asarray(a, float), np.asarray(b, float)
    h = max(float(np.linalg.norm(b - a)) - 2.0 * r, 0.004)
    return RigidBody(name=name, bone=bone, shape="capsule", size=(float(r), float(h), 0.0),
                     location=tuple(float(x) for x in 0.5 * (a + b)), rotation=euler_xyz(frame_z_to(b - a)))


def sphere(name, bone, c, r):
    return RigidBody(name=name, bone=bone, shape="sphere", size=(float(r), 0.0, 0.0), location=tuple(float(x) for x in c))


def box(name, bone, c, half, rot=(0.0, 0.0, 0.0)):
    return RigidBody(name=name, bone=bone, shape="box", size=tuple(float(x) for x in half),
                     location=tuple(float(x) for x in c), rotation=tuple(float(x) for x in rot))


def mirror_body(rb, name, bone):
    """The x-mirror of a left rigid body (same size, mirrored location and rotation)."""
    from dataclasses import replace
    rx, ry, rz = rb.rotation
    return replace(rb, name=name, bone=bone, location=(-rb.location[0], rb.location[1], rb.location[2]),
                   rotation=(rx, -ry, -rz))


def sphere_fit(P):
    """Least-squares sphere (centre, radius) through points (n, 3)."""
    A = np.c_[2 * P, np.ones(len(P))]
    b = (P ** 2).sum(1)
    x, *_ = np.linalg.lstsq(A, b, rcond=None)
    c = x[:3]
    return c, float(np.sqrt(max(x[3] + c @ c, 1e-12)))


def nearest_segment(P, chain):
    """Index of the segment of the polyline `chain` (points) that each of the points P is nearest to: a vertex of the thigh
    belongs to the thigh even where it hangs lower than the knee's head, a heel to the foot, not to the shin."""
    d = np.stack([_seg_dist(P, a, b)[0] for a, b in zip(chain[:-1], chain[1:])], 1)
    return np.argmin(d, axis=1)


def capsule_fit(P, a, b, q=0.93, trim=0.05):
    """Radius of the capsule along a-b that holds the quantile q of the points within the segment (its ends trimmed by
    `trim` of the length, where joints blend into the next bone)."""
    d, t = _seg_dist(P, a, b)
    sel = (t >= trim) & (t <= 1 - trim) & (d < 0.5 * np.linalg.norm(b - a) + 0.1)
    if sel.sum() < 8:
        sel = np.ones(len(P), bool)
    return float(np.quantile(d[sel], q))


def centred_capsule(P, a, b, q=0.9):
    """(a', b', r): the capsule that follows the FLESH of a limb segment: a bone's straight line misses the calf's bulge, so
    the axis runs through the centroids of the points in the first and last parts of the segment (extended to the bone's own
    end points) and the radius holds the quantile q of the points' distances to it."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d, t = _seg_dist(P, a, b)
    ok = d < 0.5 * np.linalg.norm(b - a) + 0.1
    lo, hi = P[ok & (t >= 0.10) & (t <= 0.30)], P[ok & (t >= 0.70) & (t <= 0.90)]
    if len(lo) < 4 or len(hi) < 4:
        return a, b, capsule_fit(P, a, b, q)
    c0, c1 = lo.mean(axis=0), hi.mean(axis=0)
    step = (c1 - c0) / 0.6
    a2, b2 = c0 - step * 0.2, c1 + step * 0.2
    d2, t2 = _seg_dist(P, a2, b2)
    sel = (t2 >= 0.05) & (t2 <= 0.95) & ok
    return a2, b2, float(np.quantile(d2[sel], q)) if sel.sum() >= 8 else capsule_fit(P, a, b, q)


def _slab_centre(P, tt, a, ab, u, v, mask):
    """Centre (3,) and mean parameter of the points in `mask`: the middle of their bounding box in the plane perpendicular to
    the bone line (u, v), carried to the mean position along it. The box middle, unlike the centroid, does not drift towards
    the side where a sparse mesh happens to have more vertices."""
    Q = P[mask]
    x, y = (Q - a) @ u, (Q - a) @ v
    tm = float(tt[mask].mean())
    return a + ab * tm + u * 0.5 * (x.min() + x.max()) + v * 0.5 * (y.min() + y.max()), tm


def flesh_axis(P, a, b):
    """(c0, c1): the axis of the flesh of the limb segment a-b at its two ends (t = 0 and 1): the line through the cross-section
    centres (`_slab_centre`) of the first and the last third of the points' own extent along the segment. Few points: a-b."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    ab = b - a
    e = ab / max(float(np.linalg.norm(ab)), 1e-12)
    tt = ((P - a) @ ab) / max(float(ab @ ab), 1e-12)
    if len(P) < 12:
        return a, b
    t0, t1 = np.quantile(tt, [0.03, 0.97])
    span = max(t1 - t0, 1e-6)
    lo, hi = tt <= t0 + 0.35 * span, tt >= t1 - 0.35 * span
    if lo.sum() < 3 or hi.sum() < 3 or tt[hi].mean() - tt[lo].mean() < 0.2 * span:
        return a, b
    u = np.cross(e, np.array([0.0, 1.0, 0.0]) if abs(e[1]) < 0.9 else np.array([1.0, 0.0, 0.0]))
    u /= np.linalg.norm(u)
    v = np.cross(e, u)
    c0, m0 = _slab_centre(P, tt, a, ab, u, v, lo)
    c1, m1 = _slab_centre(P, tt, a, ab, u, v, hi)
    step = (c1 - c0) / (m1 - m0)                       # displacement per unit of the segment's own parameter
    return c0 - step * m0, c0 + step * (1.0 - m0)


def _stack_split(profile, pieces):
    """Split `profile` (the radius of each bin, in order) into `pieces` runs of consecutive bins so that the total air above
    the profile is least when every run takes the largest radius in it (a run is a capsule). Returns the run starts plus the
    end: [0, ..., len(profile)]."""
    n = len(profile)
    pieces = min(pieces, n)
    cost = np.full((n + 1, n + 1), np.inf)
    for i in range(n):
        top = -np.inf
        for j in range(i + 1, n + 1):
            top = max(top, profile[j - 1])
            cost[i, j] = float((top - profile[i:j]).sum())
    best = np.full((pieces + 1, n + 1), np.inf)
    back = np.zeros((pieces + 1, n + 1), int)
    best[0, 0] = 0.0
    for k in range(1, pieces + 1):
        for j in range(k, n + 1):
            c = best[k - 1, k - 1:j] + cost[k - 1:j, j]
            m = int(np.argmin(c))
            best[k, j], back[k, j] = c[m], m + k - 1
    cuts, j = [n], n
    for k in range(pieces, 0, -1):
        j = int(back[k, j])
        cuts.append(j)
    return cuts[::-1]


def tapered_chain(P, a, b, pieces=5, bins=30, q=0.95, margin=0.0015, floor=0.002, t_fit=(0.12, 0.92)):
    """`pieces` capsules for a limb segment a-b that tapers (calf -> shin -> ankle), as [(tip_a, tip_b, r)] for `capsule`,
    from the knee end down. The axis is the line through the flesh (`flesh_axis`); the skin's radius about it is measured
    in `bins` slabs (quantile q), and the slabs are cut into runs where a run's capsule, as wide as its widest slab plus
    `margin`, leaves the least air (`_stack_split`). Then every skin point that is still not at least `floor` inside the
    union gets its best-fitting capsule widened to take it (the steps between two capsules are where that happens). Points
    outside `t_fit` (the knee cap and the ankle, which the neighbouring colliders cover) do not count."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d, t = _seg_dist(P, a, b)
    ok = (d < 0.5 * np.linalg.norm(b - a) + 0.1) & (t >= t_fit[0]) & (t <= t_fit[1])
    ab = b - a
    c0, c1 = flesh_axis(P[ok], a + ab * t_fit[0], a + ab * t_fit[1])
    e = (c1 - c0) / max(float(np.linalg.norm(c1 - c0)), 1e-12)
    L = float(np.linalg.norm(c1 - c0))
    Q = P[ok]
    s = ((Q - c0) @ e) / L
    rad = np.linalg.norm((Q - c0) - np.outer((Q - c0) @ e, e), axis=1)
    edges = np.linspace(0.0, 1.0, bins + 1)
    prof = np.full(bins, np.nan)
    for k in range(bins):
        w = (s >= edges[max(k - 1, 0)]) & (s <= edges[min(k + 2, bins)])           # a window of three bins smooths the profile
        if w.sum() >= 4:
            prof[k] = np.quantile(rad[w], q)
    known = ~np.isnan(prof)
    prof = np.interp(np.arange(bins), np.flatnonzero(known), prof[known])
    cut = _stack_split(prof, pieces)
    caps = []
    for i0, i1 in zip(cut[:-1], cut[1:]):
        caps.append([c0 + (c1 - c0) * edges[i0], c0 + (c1 - c0) * edges[i1], float(prof[i0:i1].max()) + margin])
    caps[0][0] = c0 + (c1 - c0) * ((0.0 - t_fit[0]) / (t_fit[1] - t_fit[0]))           # the first reaches up to the joint (t = 0)
    for _ in range(3):
        depth = np.array([c[2] - _seg_dist(Q, c[0], c[1])[0] for c in caps])    # (capsules, points): > 0 is inside
        best = depth.argmax(axis=0)
        short = depth.max(axis=0) < floor
        if not short.any():
            break
        for i in range(len(caps)):
            mine = short & (best == i)
            if mine.any():
                caps[i][2] += float((floor - depth[i][mine]).max())
    # the fit above puts the SPHERE CENTRES on the axis; `capsule` spans tip to tip, so the tips lie one radius further out
    return [(x0 - e * r, x1 + e * r, r) for x0, x1, r in caps]


def box_fit(P, axes, q=0.98, pad=0.0015):
    """(centre, half extents) of the box with the frame `axes` (columns) around points P (quantiles q and 1-q, plus pad)."""
    Q = P @ axes
    lo, hi = np.quantile(Q, 1 - q, axis=0), np.quantile(Q, q, axis=0)
    return axes @ (0.5 * (lo + hi)), 0.5 * (hi - lo) + pad


LOWER_LEG = ("col_calf", "col_shin", "col_shin_mid", "col_shin_low", "col_ankle")      # from the knee down (`tapered_chain`)


def fit_bodies(verts, G, land, head_verts, bone_names, head_hint=None):
    """The static colliders for an imported body, named like the procedural body's (col_head, col_neck, col_upper_body,
    col_upper_body2, col_lower_body, col_shoulder_L/R, col_arm, col_forearm, col_hand, col_thigh, col_calf .. col_ankle (`LOWER_LEG`: the tapered lower leg, a stack of capsules on the knee bone), col_foot).

    verts, G   skin vertices (n, 3) and their region weights (`vertex_groups`); the dominant region owns a vertex
    land       landmarks (`landmarks`)
    head_verts the vertices of the head (face and eyes) for the sphere; `head_hint` (centre, radius) replaces the fit
    bone_names the bones that exist: a collider is only emitted when its bone does
    Left side bodies are fitted on the left vertices together with the mirrored right ones and mirrored back."""
    V = np.asarray(verts, float)
    dom = np.argmax(G, axis=1)
    has = G.sum(1) > 0.5
    own = lambda g: has & (dom == GROUPS.index(g))
    out = []

    def keep(rb):
        if rb.bone in bone_names:
            rb.group, rb.no_collide, rb.mass = 0, (0,), 1.0
            out.append(rb)

    # head and neck
    if head_hint is not None:
        c, r = np.asarray(head_hint[0], float), float(head_hint[1])
    else:
        c, r = sphere_fit(np.asarray(head_verts, float))
        d = np.linalg.norm(np.asarray(head_verts, float) - c, axis=1)
        r = float(np.quantile(d, 0.9))
    keep(sphere("col_head", "頭", c, r))
    nk, hd = land["neck"], land["head"]
    neck_v = V[own("neck")]
    r_neck = capsule_fit(neck_v, nk, hd, q=0.9) if len(neck_v) > 8 else 0.027
    keep(capsule("col_neck", "首", nk + np.array([0.0, 0.0, -0.004]), hd + np.array([0.0, 0.0, 0.012]), r_neck))
    # torso: axis aligned boxes of the torso region between the bones' heights
    T = V[own("torso")]
    ub, ub2 = float(land["upper_body"][2]), float(land["upper_body2"][2])
    for name, bone, z0, z1 in (("col_lower_body", "下半身", T[:, 2].min() + 0.012, ub + 0.004),
                               ("col_upper_body", "上半身", ub + 0.004, ub2 + 0.004),
                               ("col_upper_body2", "上半身2", ub2 + 0.004, float(land["neck"][2]) + 0.002)):
        sel = (T[:, 2] >= z0) & (T[:, 2] <= z1)
        if sel.sum() >= 12:
            c, half = box_fit(T[sel], np.eye(3))
            keep(box(name, bone, c, half))

    def left_points(group):
        P = V[own(group)]
        left, right = P[P[:, 0] > 0], P[P[:, 0] < 0] * np.array([-1.0, 1.0, 1.0])
        return np.concatenate([left, right], 0)

    def point(sem):
        p = np.asarray(land[f"{sem}.L"], float).copy()
        p[0] = 0.5 * (abs(land[f"{sem}.L"][0]) + abs(land[f"{sem}.R"][0]))
        return p

    sh, ar, el, wr = point("shoulder"), point("arm"), point("elbow"), point("wrist")
    leg, kn, an = point("leg"), point("knee"), point("ankle")
    left = []
    ps = left_points("shoulder")
    r_sh = capsule_fit(ps, sh, ar, q=0.85) if len(ps) > 8 else 0.034
    left.append(capsule("col_shoulder_L", "左肩", sh + np.array([0.0, 0.0, 0.012]), ar + np.array([0.0, 0.0, 0.004]), r_sh))
    pa = left_points("arm")
    ka = nearest_segment(pa, [ar, el, wr])
    left.append(capsule("col_arm_L", "左腕", *centred_capsule(pa[ka == 0], ar, el)))
    left.append(capsule("col_forearm_L", "左ひじ", *centred_capsule(pa[ka == 1], el, wr)))
    # hand: the palm in the frame of the hand (along: wrist -> middle knuckle, across: index -> little, palm normal)
    ph = left_points("hand")
    a_ = unit(point("middle1") - wr)
    across = unit(point("little1") - point("index1"))
    n_ = -unit(np.cross(a_, across))
    r_ = unit(np.cross(n_, a_))
    if r_ @ across < 0:
        r_ = -r_
    palm = ph[((ph - wr) @ a_ <= np.linalg.norm(point("middle1") - wr) + 0.004) & ((ph - wr) @ a_ >= -0.01)]
    if len(palm) >= 12:
        c, half = box_fit(palm, np.stack([a_, r_, -n_], 1))
        left.append(box("col_hand_L", "左手首", c, half, rot=euler_xyz(np.stack([a_, r_, -n_], 1))))
    pl = left_points("leg")
    toe = point("toe") if "toe.L" in land else an + np.array([0.0, -0.08, -0.05])
    kl = nearest_segment(pl, [leg, kn, an, toe])
    left.append(capsule("col_thigh_L", "左足", *centred_capsule(pl[kl == 0], leg, kn)))
    for name, cap in zip(LOWER_LEG, tapered_chain(pl[kl == 1], kn, an, pieces=len(LOWER_LEG))):
        left.append(capsule(name + "_L", "左ひざ", *cap))              # the shin tapers: a stack of capsules on the knee bone
    foot = pl[pl[:, 2] < an[2] - 0.012]
    if len(foot) >= 12:
        c, half = box_fit(foot, np.eye(3))
        left.append(box("col_foot_L", "左足首", c, half))
    for rb in left:
        keep(rb)
        keep(mirror_body(rb, rb.name[:-1] + "R", rb.bone.replace("左", "右")))
    return out
