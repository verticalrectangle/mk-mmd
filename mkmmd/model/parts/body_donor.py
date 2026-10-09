"""The body part in `mesh` mode (`[body] source = "mesh"`): the skin is an artist's whole body fitted to the character's
skeleton instead of lofted from the proportions' tables. The girl base's is Blender Studio's stylized female body (CC0);
bases/girl/make_body.py makes that asset from its source.

Spec (`[body]` in body.toml): `source = "mesh"` and `mesh = "base:girl/body.npz"` (a relative path is taken against the
spec's folder; a missing asset with a maker beside it is made on first use), plus the procedural body's keys that still
apply: skin, nails, neck_shadow, hand (the hands are the hand part's, welded on at the wrists), land, dims, skeleton.

The fit (`fit`) takes the asset's own space to the character's; k is her neck seam's height over the donor's neck cut,
[proportions] leg_extra left out (longer legs make no wider body):
  torso  a warp by height: the donor's hip joints, shoulder joints and neck cut go to the heights of her leg and arm joints
         and her neck seam (straight from the hips up to LEAD below the shoulder joints, a monotone bend from there to the
         neck); each height is scaled across (x) and front to back (y) by k, except where joints pin it: x at the hip and
         shoulder joints so they land on hers, both at the neck cut, onto her seam ellipse
  limbs  each segment (upper arm, forearm, thigh, shin) turned and stretched from the donor's joints onto hers, k across
         it; the foot (`foot_map`) keeps its flat stance: turned about the vertical onto her foot's line, sized to her
         `foot_length` and `foot_width`, its sole stood on her shoe's inner floor (`foot_inner_floor_z`) and its ankle on hers
  blend  every vertex takes the torso warp, its arm's and its leg's transforms by its share of each region (`membership`);
         a limb's segments hand over by smooth steps across their joints (HALF)
Then the neck's cut ring is laid onto the seam ellipse the head builds on (the skin below follows, fading over NECK_EASE),
and each forearm is cut at the asset's edge loop nearest before the hand's seam: a strip joins that loop to a ring of the
hand's seam size sampled from the fitted forearm at the seam, and the hand is welded to that ring as to the procedural
forearm's last ring (with the designed hand, the ring's section table comes from the fitted forearm too).

Weights: the torso's share by the procedural spine rules (body_weights.spine_weights), an arm's and a leg's by the
procedural chains along her limbs (arm_weights, leg_weights), the hands and nails as on the procedural body. The skin
texture: the asset's UVs fill the atlas's top half, painted with the procedural body's marks placed on the fitted skin
(knee caps, heels, elbows ...) and its neck shadow; the hands keep their tiles in the bottom half. Colliders are fitted to
the skin as for an imported body (body_pmx_fit.fit_bodies), the shoulders as wide as the shoulder's top (shoulder_tops).

The asset (an .npz):
  verts (n, 3), face_flat, face_sizes   the skin, faces counter-clockwise from outside, open only at the neck cut
  uv (corners, 2)                       per face corner, in the skin atlas's top half
  joint_names, joints (k, 3)            the donor's joints, left side: shoulder elbow wrist hip knee ankle, and the skin
                                        points toe_end (the front of the toes at the floor) and heel (its back)
  neck_z, waist_z, bust_z               heights of the neck cut, of the narrowest waist and of the deepest bust
  regions, membership (n, 5)            each vertex's share of torso, arm.L, arm.R, leg.L, leg.R (rows sum to one)
  wrist_loops (k, m)                    the left forearm's edge loops near the wrist, from the elbow's side to the wrist's
  mirror (n,)                           every vertex's mirror twin
  plain_uv (2,)                         a texel no mark reaches (the strip to the hand takes it)
  license, source                       recorded, not read

Published in `Part.info` as by the procedural body (body.py): landmarks (+ tail_root, bust_front taken from the skin, as
for an imported body), tail_normal, neck_top, skin, neck_shadow, torso_profile (cuts of the skin's torso region), regions
(vertex ids per region), hand, foot; plus source = "mesh"."""
from dataclasses import replace
from functools import lru_cache

import numpy as np

from .. import skeleton
from ..part import Mesh, Part
from ..spec import expand, make_missing
from . import body_geom as G
from . import body_hand as BH
from . import body_mesh as BM
from . import body_pmx_fit as F
from . import body_shape, body_tex
from . import body_weights as BW
from .body import neck_ring, skin_info, skin_look, skin_materials
from .body_geom import Path as Chain, Shell, Table, mirror_x, pchip, smoothstep, unit
from .body_hand_mesh import open_loop

REGIONS = ("torso", "arm.L", "arm.R", "leg.L", "leg.R")
JOINTS = ("shoulder", "elbow", "wrist", "hip", "knee", "ankle", "toe_end", "heel")
LEAD = 0.07                 # the torso's height map is straight up to this far (donor m) below the shoulder joints
HALF = dict(elbow=0.03, knee=0.05, ankle=0.03)    # half widths of the segment hand-overs (donor m)
NECK_EASE = 0.012           # the skin under the neck's cut ring follows it onto the seam ellipse, fading over this height
GAP = 0.004                 # the forearm is cut at an edge loop at least this far before the hand's seam
REACH = 0.06                # the forearm's section at the seam is the skin within this distance of the arm's centre line
REF = (0.0, -1.0, 0.0)      # a forearm ring's first point faces this way (body_mesh.arm_shell's convention)
SHARE = (0.10, 0.55)        # a limb's share of a vertex, for the weights, rises from 0 to 1 between these memberships: the
                            # fit wants the long hand-over over the shoulder, skinning a short one (the upper arm all
                            # 腕 from about 6 cm past the joint, as on the procedural arm)


# ---------------------------------------------------------------- the asset
@lru_cache(maxsize=2)
def _load(path, mtime):
    with np.load(path, allow_pickle=False) as z:
        a = {k: z[k] for k in z.files}
    need = ("verts", "face_flat", "face_sizes", "uv", "joint_names", "joints", "neck_z", "waist_z", "bust_z", "regions",
            "membership", "wrist_loops", "mirror", "plain_uv")
    missing = [k for k in need if k not in a]
    if missing:
        raise ValueError(f"body mesh {path}: missing {', '.join(missing)}")
    V = np.asarray(a["verts"], float)
    sizes = np.asarray(a["face_sizes"], int)
    cuts = np.cumsum(sizes)[:-1]
    faces = [[int(v) for v in f] for f in np.split(np.asarray(a["face_flat"], int), cuts)]
    J = {str(n): np.asarray(p, float) for n, p in zip(a["joint_names"], a["joints"])}
    lacking = [n for n in JOINTS if n not in J]
    if lacking:
        raise ValueError(f"body mesh {path}: joints missing: {', '.join(lacking)}")
    if tuple(str(r) for r in a["regions"]) != REGIONS:
        raise ValueError(f"body mesh {path}: regions must be {', '.join(REGIONS)}")
    member = np.asarray(a["membership"], float)
    if member.shape != (len(V), len(REGIONS)) or not np.allclose(member.sum(1), 1.0, atol=1e-4):
        raise ValueError(f"body mesh {path}: membership must be (verts, {len(REGIONS)}), every row summing to one")
    heights = [J["hip"][2], float(a["waist_z"]), float(a["bust_z"]), J["shoulder"][2], float(a["neck_z"])]
    if not all(p < q for p, q in zip(heights[:-1], heights[1:])):
        raise ValueError(f"body mesh {path}: hip joint, waist, bust, shoulder joint and neck cut must rise in that order")
    neck = open_loop(faces)
    if np.abs(V[neck, 2] - float(a["neck_z"])).max() > 1e-6:
        raise ValueError(f"body mesh {path}: its one open boundary must be the neck cut at neck_z")
    return dict(verts=V, faces=faces, uv=np.split(np.asarray(a["uv"], float), cuts), joints=J,
                neck_z=float(a["neck_z"]), waist_z=float(a["waist_z"]), bust_z=float(a["bust_z"]), member=member,
                loops=np.asarray(a["wrist_loops"], int), mirror=np.asarray(a["mirror"], int),
                plain_uv=np.asarray(a["plain_uv"], float), neck=neck)


def load(path):
    """The asset at `path` (shared, not to be changed); made by the maker beside it when missing."""
    from pathlib import Path
    p = Path(str(path)).expanduser()
    if not p.is_file() and not make_missing(p):
        raise ValueError(f"[body] mesh: {p} does not exist")
    return _load(str(p), p.stat().st_mtime)


# ---------------------------------------------------------------- the fit
def _segment_map(a_d, b_d, a_h, b_h, across):
    """p -> a_h + R S (p - a_d): S stretches along the donor's segment a_d-b_d to her length and scales by `across`
    across it, R turns it (the least turn) onto her segment a_h-b_h."""
    d = unit(b_d - a_d)
    s = float(np.linalg.norm(b_h - a_h) / np.linalg.norm(b_d - a_d))
    A = _turn(b_d - a_d, b_h - a_h) @ (across * np.eye(3) + (s - across) * np.outer(d, d))
    return lambda P: a_h + (P - a_d) @ A.T


def _turn(a, b):
    """The least rotation taking direction a onto direction b."""
    a, b = unit(a), unit(b)
    v, c = np.cross(a, b), float(a @ b)
    if np.linalg.norm(v) < 1e-12:
        return np.eye(3)
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + K + K @ K / (1.0 + c)


def _hand_over(P, a, b, c, half):
    """Share of segment b-c (against a-b) of points P: a smooth step across the plane through joint b that halves the
    angle between the segments, over +-half."""
    n = unit(unit(b - a) + unit(c - b))
    return smoothstep(((P - b) @ n) / (2.0 * half) + 0.5)


def _straight(xs, ys):
    """pchip through (xs, ys), continued straight beyond both ends."""
    f = pchip(xs, ys)
    s0 = (ys[1] - ys[0]) / (xs[1] - xs[0])
    s1 = (ys[-1] - ys[-2]) / (xs[-1] - xs[-2])

    def g(x):
        x = np.asarray(x, float)
        return np.where(x < xs[0], ys[0] + (x - xs[0]) * s0, np.where(x > xs[-1], ys[-1] + (x - xs[-1]) * s1, f(x)))
    return g


def height_map(donor, hers):
    """z -> z' through three heights (hip joint, shoulder joint, neck): straight from the first to LEAD below the second,
    a monotone bend from there to the third."""
    (h0, h1, h2), (g0, g1, g2) = donor, hers
    slope = (g1 - g0) / (h1 - h0)
    xs = [h0, 0.5 * (h0 + h1 - LEAD), h1 - LEAD, h1, h2]
    return _straight(np.array(xs), np.array([g0 + (x - h0) * slope for x in xs[:3]] + [g1, g2]))


def foot_map(V, on_foot, d, ankle, toe_end, foot):
    """The foot as it stands: turned about the vertical so the donor's heel-to-toe line `d` (heel, toe_end, ankle points)
    runs along her ankle-to-toe_end line, sized to her foot along and across that line (`foot_length`, `foot_width` of
    the dims), and stood on her shoe's inner floor: the donor's sole (the foot's lowest point) goes to
    `foot_inner_floor_z`, its ankle onto hers. `on_foot`: the donor's vertices of that foot (its width and its sole)."""
    a = unit((d["toe_end"] - d["heel"]) * np.array([1.0, 1.0, 0.0]))
    b = np.array([-a[1], a[0], 0.0])
    z = np.array([0.0, 0.0, 1.0])
    length = float(np.linalg.norm((d["toe_end"] - d["heel"])[:2]))
    width = float(np.ptp(V[on_foot] @ b))
    sole = float(V[on_foot, 2].min())
    S = (float(foot["foot_length"]) / length * np.outer(a, a) + float(foot["foot_width"]) / width * np.outer(b, b)
         + (ankle[2] - float(foot["foot_inner_floor_z"])) / (d["ankle"][2] - sole) * np.outer(z, z))
    A = _turn(a, (toe_end - ankle) * np.array([1.0, 1.0, 0.0])) @ S
    return lambda P: ankle + (P - d["ankle"]) @ A.T


def fit(A, shape, nt, extra=0.0):
    """(n, 3) the asset's vertices in the character's space (module docstring): `shape` her body_shape.Shape, `nt` her
    neck seam (body_mesh.neck_top), `extra` her [proportions] leg_extra."""
    J, L, V, M = A["joints"], shape.land, A["verts"], A["member"]
    ring = V[A["neck"]]
    rx_d, ry_d = 0.5 * float(np.ptp(ring[:, 0])), 0.5 * float(np.ptp(ring[:, 1]))
    yc_d = 0.5 * float(ring[:, 1].min() + ring[:, 1].max())
    k = (float(nt["z"]) - extra) / A["neck_z"]
    hip, sh, nz = J["hip"], J["shoulder"], A["neck_z"]
    leg, arm = np.asarray(L["leg.L"], float), np.asarray(L["arm.L"], float)
    zmap = height_map((hip[2], sh[2], nz), (leg[2], arm[2], float(nt["z"])))
    zs = [hip[2], A["waist_z"], A["bust_z"], sh[2], nz]
    sx = pchip(zs, [leg[0] / hip[0], k, k, arm[0] / sh[0], nt["rx"] / rx_d])
    sy = pchip(zs, [k, k, k, k, nt["ry"] / ry_d])
    yd = lambda z: hip[1] + (z - hip[2]) * (yc_d - hip[1]) / (nz - hip[2])
    yh = lambda z: leg[1] + (z - leg[2]) * (nt["center"][1] - leg[1]) / (nt["z"] - leg[2])
    z2 = zmap(V[:, 2])
    out = M[:, :1] * np.stack([sx(V[:, 2]) * V[:, 0], yh(z2) + sy(V[:, 2]) * (V[:, 1] - yd(V[:, 2])), z2], 1)
    floor = shape.dims["foot"]
    for side, sg in (("L", 1.0), ("R", -1.0)):
        d = {n: J[n] * np.array([sg, 1.0, 1.0]) for n in JOINTS}
        h = lambda n: np.asarray(L[f"{n}.{side}"], float)
        # arm
        up = _segment_map(d["shoulder"], d["elbow"], h("arm"), h("elbow"), k)
        fore = _segment_map(d["elbow"], d["wrist"], h("elbow"), h("wrist"), k)
        w = _hand_over(V, d["shoulder"], d["elbow"], d["wrist"], HALF["elbow"])[:, None]
        out += M[:, 1 if sg > 0 else 2][:, None] * ((1 - w) * up(V) + w * fore(V))
        # leg: thigh, shin, the foot as it stands
        col = 3 if sg > 0 else 4
        thigh = _segment_map(d["hip"], d["knee"], h("leg"), h("knee"), k)
        shin = _segment_map(d["knee"], d["ankle"], h("knee"), h("ankle"), k)
        on_foot = (M[:, col] > 0.5) & (V[:, 2] < d["ankle"][2] - 0.01)
        ft = foot_map(V, on_foot, d, h("ankle"), h("toe_end"), floor)
        fwd = d["ankle"] + unit((d["toe_end"] - d["heel"]) * np.array([1.0, 1.0, 0.0]))
        w1 = _hand_over(V, d["hip"], d["knee"], d["ankle"], HALF["knee"])[:, None]
        w2 = _hand_over(V, d["knee"], d["ankle"], fwd, HALF["ankle"])[:, None]
        out += M[:, col][:, None] * ((1 - w1) * thigh(V) + w1 * ((1 - w2) * shin(V) + w2 * ft(V)))
    return out


def onto_seam(V, loop, nt):
    """Lay the neck's cut ring `loop` onto the seam ellipse `nt` (each point at its own angle round the ellipse) and move
    the neck below with it, the move fading over NECK_EASE; changes V."""
    cx, cy = nt["center"]
    rx, ry, z = float(nt["rx"]), float(nt["ry"]), float(nt["z"])

    def angle(P):
        return np.arctan2((P[:, 0] - cx) / rx, -(P[:, 1] - cy) / ry)
    a = angle(V[loop])
    target = np.stack([cx + rx * np.sin(a), cy - ry * np.cos(a), np.full(len(a), z)], 1)
    move = target - V[loop]
    o = np.argsort(a)
    knots = np.concatenate([a[o][-1:] - 2 * np.pi, a[o], a[o][:1] + 2 * np.pi])
    rho = np.hypot((V[:, 0] - cx) / rx, (V[:, 1] - cy) / ry)
    near = np.flatnonzero((V[:, 2] > z - NECK_EASE) & (rho < 1.6))
    at = angle(V[near])
    fade = 1.0 - smoothstep((z - V[near, 2]) / NECK_EASE)
    for c in range(3):
        m = move[o, c]
        V[near, c] += fade * np.interp(at, knots, np.concatenate([m[-1:], m, m[:1]]))
    V[loop] = target


# ---------------------------------------------------------------- the forearm's end
def arm_path(shape):
    """(path, la, lf) of her left arm as body_mesh.arm_shell has it: shoulder, elbow, wrist, then on along the hand."""
    L = shape.land
    J, E, W = (np.asarray(L[n], float) for n in ("arm.L", "elbow.L", "wrist.L"))
    return Chain([J, E, W, W + shape.frame[0] * 0.04], blend=0.035), float(np.linalg.norm(E - J)), float(np.linalg.norm(W - E))


def _frame(path, s):
    c = path.point(np.array([s]))[0]
    t = unit(path.tangent(np.array([s]))[0])
    ef = unit(np.asarray(REF, float) - (t @ np.asarray(REF, float)) * t)
    return c, t, ef, np.cross(t, ef)


def _radii(V, tris, path, s, theta):
    """Distance from the arm's centre line, at arclength s, to the skin in the directions `theta` (the ring angles of
    body_geom.tube: 0 along REF, growing towards t x REF), on the section of the skin within REACH of the line."""
    c, t, ef, eb = _frame(path, s)
    R = np.stack([ef, eb, t])
    segs = F.slice_z((V - c) @ R.T, tris, 0.0)[..., :2]
    segs = segs[(np.linalg.norm(segs, axis=2) < REACH).all(1)]
    u = np.stack([np.cos(theta), np.sin(theta)], 1)
    A, E = segs[:, 0], segs[:, 1] - segs[:, 0]
    den = u[:, None, 0] * E[None, :, 1] - u[:, None, 1] * E[None, :, 0]
    ok = np.abs(den) > 1e-12
    den = np.where(ok, den, 1.0)
    r = (A[None, :, 0] * E[None, :, 1] - A[None, :, 1] * E[None, :, 0]) / den
    q = (A[None, :, 0] * u[:, None, 1] - A[None, :, 1] * u[:, None, 0]) / den
    r = np.where(ok & (q >= -1e-9) & (q <= 1 + 1e-9) & (r > 0), r, np.inf).min(1)
    if not np.isfinite(r).all():
        raise ValueError(f"the body mesh's forearm has no closed section at {s:.3f} m along the arm")
    return r, (c, ef, eb)


def forearm_end(shape, A, V, tris, M_ring):
    """The left forearm's cut: (loop: the asset's edge loop nearest before the hand's seam, ring (M_ring, 3) sampled from
    the skin at the seam, section table round the wrist for the designed hand, the path's numbers)."""
    path, la, lf = arm_path(shape)
    s_w = la + lf
    s_seam = s_w + BH.SEAM
    s_loops = np.array([path.project(V[lp]).max() for lp in A["loops"]])
    ok = np.flatnonzero(s_loops < s_seam - GAP)
    if not len(ok):
        raise ValueError("the body mesh's forearm loops all lie past the hand's seam: the arm is too short for them")
    loop = A["loops"][ok[np.argmax(s_loops[ok])]]
    th = 2 * np.pi * np.arange(M_ring) / M_ring
    r, (c, ef, eb) = _radii(V, tris, path, s_seam, th)
    ring = c + r[:, None] * (np.cos(th)[:, None] * ef + np.sin(th)[:, None] * eb)
    q4 = np.array([0.0, 0.5 * np.pi, np.pi, 1.5 * np.pi])
    rows = []
    for s in (s_seam - 0.01, s_seam, s_w + BH.WRIST_RING, s_w):
        r4, _ = _radii(V, tris, path, s, q4)
        rows.append((s, 0.5 * (r4[0] + r4[2]), 0.5 * (r4[1] + r4[3]), 0.5 * (r4[0] - r4[2]), 0.5 * (r4[1] - r4[3])))
    R_ = np.array(rows)
    tab = Table(R_[:, 0], rx=R_[:, 1], ry=R_[:, 2], ox=R_[:, 3], oy=R_[:, 4], n=np.full(len(R_), 2.0))
    return loop, ring, dict(path=path, ref=REF, tab=tab, la=la, lf=lf, s_w=s_w, s_seam=s_seam, M=M_ring)


def beyond(faces, loop, seed):
    """Indices of the faces reached from face `seed` without crossing an edge of the closed vertex loop `loop`."""
    wall = {(int(a), int(b)) for a, b in zip(loop, np.roll(loop, -1))}
    wall |= {(b, a) for a, b in wall}
    by_edge = {}
    for fi, f in enumerate(faces):
        for i in range(len(f)):
            by_edge.setdefault((min(f[i], f[(i + 1) % len(f)]), max(f[i], f[(i + 1) % len(f)])), []).append(fi)
    seen, todo = {seed}, [seed]
    while todo:
        f = faces[todo.pop()]
        for i in range(len(f)):
            e = (f[i], f[(i + 1) % len(f)])
            if e in wall:
                continue
            for g in by_edge[(min(e), max(e))]:
                if g not in seen:
                    seen.add(g)
                    todo.append(g)
    return seen


def zipper(a_ang, b_ang):
    """Triangles between two loops round one axis, as (i, j, k) with a loop A point as its index and a loop B point as
    len(A) + its index: both loops are walked once round (their angles increase), each step advancing the loop whose
    next point comes first. Facing outwards when B lies further along the axis than A."""
    n, m = len(a_ang), len(b_ang)
    a = np.unwrap(a_ang)
    b = np.unwrap(b_ang)
    a = a - 2 * np.pi * np.floor((a[0] - b[0]) / (2 * np.pi) + 0.5)
    i0 = int(np.argmin(np.abs(np.angle(np.exp(1j * (a - b[0]))))))
    order = np.roll(np.arange(n), -i0)
    a = np.unwrap(a[order])
    a -= 2 * np.pi * np.round((a[0] - b[0]) / (2 * np.pi))
    a, b = np.append(a, a[0] + 2 * np.pi), np.append(b, b[0] + 2 * np.pi)
    tris, i, j = [], 0, 0
    while i < n or j < m:
        if j == m or (i < n and a[i + 1] < b[j + 1]):
            tris.append((order[i % n], order[(i + 1) % n], n + j % m))
            i += 1
        else:
            tris.append((order[i % n], n + (j + 1) % m, n + j % m))
            j += 1
    return tris


def wrist_shell(name, V_loop, loop_ids, ring, info, plain_uv):
    """The strip from the forearm's cut loop (the donor's vertices `loop_ids`, welded: `ext`) to the ring the hand is
    welded to: a Shell with the ring as its own vertices and its one ring (ring_index) as body_mesh's arm has."""
    M = len(ring)
    c, t, ef, eb = _frame(info["path"], info["s_seam"])
    ang = lambda P: np.arctan2((P - c) @ eb, (P - c) @ ef)
    tri = zipper(ang(V_loop), ang(ring))
    n = len(loop_ids)
    sh = Shell(name)
    sh.verts = np.concatenate([V_loop, ring])
    sh.faces = [[int(i), int(j), int(k)] for i, j, k in tri]
    sh.uv = [np.tile(plain_uv, (3, 1)) for _ in tri]
    sh.s = np.concatenate([info["path"].project(V_loop), np.full(M, info["s_seam"])])
    sh.theta = np.full(len(sh.verts), np.nan)
    sh.ring = np.full(len(sh.verts), -1)
    sh.ring_index = (n + np.arange(M))[None]
    sh.rings = ring[None]
    sh.face_mat = [0] * len(tri)
    sh.ext = {i: ("donor", int(v)) for i, v in enumerate(loop_ids)}
    sh.info = dict(info, kind="wrist")
    return sh


# ---------------------------------------------------------------- weights
def _signed_s(path, P):
    """Arclength along `path` of points P, negative (along the first segment) before its start."""
    s = path.project(P)
    t0 = unit(path.pts[1] - path.pts[0])
    return np.where(s <= 1e-9, np.minimum((P - path.pts[0]) @ t0, 0.0), s)


def donor_weights(shape, V, member, bones):
    """{bone: (n,)} of the fitted donor skin: the torso's share by the spine rules, each arm's and leg's by the procedural
    chains along her limbs (their roots blending into the shoulder girdle and the lower body as on the procedural body).
    The shares are the regions' memberships sharpened by SHARE."""
    m = member.copy()
    m[:, 1:] = smoothstep((m[:, 1:] - SHARE[0]) / (SHARE[1] - SHARE[0]))
    m[:, 0] = np.clip(1.0 - m[:, 1:].sum(1), 0.0, 1.0)
    m /= m.sum(1, keepdims=True)
    W = {}

    def add(w, share):
        for b, x in w.items():
            W[b] = W.get(b, 0.0) + share * np.asarray(x)
    add(BW.spine_weights(shape, V[:, 2], V[:, 0], bones), m[:, 0])
    path, la, lf = arm_path(shape)
    lpath, _ = BM.leg_path(shape)
    L = shape.land
    sK, sA = lpath.project(np.array([L["knee.L"], L["ankle.L"]]))
    for side, sg, ca, cl in (("L", 1.0, 1, 3), ("R", -1.0, 2, 4)):
        P = V * np.array([sg, 1.0, 1.0])
        sh = Shell(f"arm_{side}")
        sh.verts, sh.s, sh.info = V, _signed_s(path, P), dict(la=la, lf=lf)
        add(BW.arm_weights(shape, sh, side), m[:, ca])
        sh = Shell(f"leg_{side}")
        sh.verts, sh.s, sh.info = V, _signed_s(lpath, P), dict(sK=float(sK), sA=float(sA))
        add(BW.leg_weights(shape, sh, side), m[:, cl])
    return W


# ---------------------------------------------------------------- texture
def _skin_point(V, mask, centre, axis, direction, slab=0.008):
    """The vertex of V[mask] within `slab` of the plane through `centre` across `axis` that lies furthest `direction`."""
    P = V[mask]
    for _ in range(4):
        near = np.abs((P - centre) @ axis) < slab
        if near.sum() >= 3:
            Q = P[near]
            return Q[np.argmax(Q @ np.asarray(direction, float))]
        slab *= 2
    raise ValueError("the body mesh has no skin where a texture mark goes")


def marks(shape, V, member):
    """[(centre, axis, sigma along, sigma across, strength, colour key)]: the procedural body's marks (body_tex.marks)
    placed on the fitted skin: knee caps and the backs of the knees, heels, the tops of the feet at the ball, toe tips,
    ankle bones, the points and the insides of the elbows."""
    L, out = shape.land, []
    for side, sg, ca, cl in (("L", 1.0, 1, 3), ("R", -1.0, 2, 4)):
        p = lambda n: np.asarray(L[f"{n}.{side}"], float)
        leg = member[:, cl] > 0.5
        a_sh, a_ft, a_to = unit(p("ankle") - p("knee")), unit(p("toe") - p("ankle")), unit(p("toe_end") - p("toe"))
        out += [(_skin_point(V, leg, p("knee"), a_sh, (0, -1, 0)), a_sh, 0.040, 0.050, 1.25, "blush_knee"),
                (_skin_point(V, leg, p("knee"), a_sh, (0, 1, 0)), a_sh, 0.020, 0.028, 0.55, "blush_knee"),
                (_skin_point(V, leg & (V[:, 2] < p("ankle")[2] - 0.02), p("ankle"), (1, 0, 0), (0, 1, 0), 0.03),
                 a_ft, 0.030, 0.040, 0.85, "blush_knee"),
                (_skin_point(V, leg, p("toe"), a_ft, (0, 0, 1)), a_ft, 0.030, 0.050, 0.55, "blush_knee"),
                (_skin_point(V, leg, p("toe") + 0.03 * a_to, a_to, (0, -1, 0), 0.02),
                 a_to, 0.045, 0.060, 0.80, "blush"),
                (_skin_point(V, leg, p("ankle"), a_sh, (sg, 0, 0)), a_sh, 0.018, 0.020, 0.55, "blush_knee"),
                (_skin_point(V, leg, p("ankle"), a_sh, (-sg, 0, 0)), a_sh, 0.018, 0.020, 0.55, "blush_knee")]
        arm = member[:, ca] > 0.5
        a_up, a_fo = unit(p("elbow") - p("arm")), unit(p("wrist") - p("elbow"))
        out += [(_skin_point(V, arm, p("elbow"), a_fo, (0, 1, 0)), a_fo, 0.011, 0.030, 1.00, "blush_knee"),
                (_skin_point(V, arm, p("elbow") - 0.02 * a_up, a_up, (0, 0, -1)), a_up, 0.020, 0.034, 0.30, "blush")]
    return out


def colour_of(mk, pal, base, nt, neck):
    """colour(P) of the donor skin: `base`, the marks `mk` mixed in (strength 1 mixes half the colour in at a mark's
    centre, never more than 0.85), the neck shadow under the seam (body_tex's band: darkest at the seam, `neck_weight`
    round the neck, fading over `height`; off the neck it fades out over the next neck's width)."""
    cx, cy = nt["center"]

    def colour(P):
        col = np.broadcast_to(base, P.shape).copy()
        for c, ax, sa, sc, k, key in mk:
            q = P - c
            a = q @ np.asarray(ax, float)
            r2 = np.maximum((q * q).sum(1) - a * a, 0.0)
            b = np.clip(k * 0.5 * np.exp(-(a / sa) ** 2 - r2 / sc ** 2), 0.0, 0.85)
            col = col * (1 - b[:, None]) + pal[key] * b[:, None]
        if neck["strength"] > 0:
            depth = float(nt["z"]) - P[:, 2]
            rho = np.hypot((P[:, 0] - cx) / nt["rx"], (P[:, 1] - cy) / nt["ry"])
            u = np.mod(np.arctan2(P[:, 0] - cx, -(P[:, 1] - cy)) / (2 * np.pi), 1.0)
            wv = np.where(depth >= -1e-6, 1.0 - smoothstep(depth / neck["height"]), 0.0)
            b = np.clip(neck["strength"] * wv * body_tex.neck_weight(u) * (1.0 - smoothstep(rho - 1.2)), 0.0, 0.9)
            col = col * (1 - b[:, None]) + pal["neck_shadow"] * b[:, None]
        return col
    return colour


def _normals(V, faces, nt):
    """Smooth vertex normals; on the neck within 5.5 mm of the seam horizontal and radial, as the head's seam ring."""
    n = G.vertex_normals(V, faces)
    cx, cy = nt["center"]
    rho = np.hypot((V[:, 0] - cx) / nt["rx"], (V[:, 1] - cy) / nt["ry"])
    top = (np.abs(V[:, 2] - nt["z"]) < 0.0055) & (rho < 1.6)
    rad = np.stack([(V[:, 0] - cx) / nt["rx"] ** 2, (V[:, 1] - cy) / nt["ry"] ** 2, np.zeros(len(V))], 1)
    n[top] = unit(rad[top])
    return n


# ---------------------------------------------------------------- the part
def build_donor(ctx):
    cfg = ctx.cfg or {}
    if not cfg.get("mesh"):
        raise ValueError('[body] source = "mesh" needs `mesh`, the body asset (e.g. "base:girl/body.npz")')
    prop = ctx.spec.get("proportions") or {}
    base = getattr(ctx.spec, "dir", None)
    shape = body_shape.resolve(prop, cfg, base=base)
    for n in shape.notes:
        ctx.log("WARNING " + n)
    A = load(expand(cfg["mesh"], base))
    nt = BM.neck_top(shape)
    nails = cfg.get("nails", True)

    # ---- the fitted skin, cut at the wrists
    V = fit(A, shape, nt, float(ctx.get("proportions.leg_extra") or 0.0))
    onto_seam(V, A["neck"], nt)
    faces, twin = A["faces"], A["mirror"]
    tris = G.triangulated(faces)
    hm = BM.hand_module(shape.hand)
    loop, ring, info = forearm_end(shape, A, V, tris, hm.seam_size(shape.hand))
    s_v = info["path"].project(V)
    tip = int(np.argmax(np.where(A["member"][:, 1] > 0.5, s_v, -np.inf)))
    left = beyond(faces, loop, next(fi for fi, f in enumerate(faces) if tip in f))
    by_set = {frozenset(f): fi for fi, f in enumerate(faces)}
    right = {by_set[frozenset(int(twin[v]) for v in faces[fi])] for fi in left}
    keep = np.ones(len(faces), bool)
    keep[list(left | right)] = False
    used = np.unique(np.concatenate([np.array(f) for f, k in zip(faces, keep) if k]))
    remap = -np.ones(len(V), int)
    remap[used] = np.arange(len(used))
    if (remap[twin[used]] < 0).any():
        raise ValueError("the body mesh is not symmetric where the forearms are cut")
    donor = Shell("donor")
    donor.verts = V[used]
    donor.faces = [[int(remap[v]) for v in f] for f, k in zip(faces, keep) if k]
    donor.uv = [u for u, k in zip(A["uv"], keep) if k]
    donor.s, donor.theta, donor.ring = np.zeros(len(used)), np.full(len(used), np.nan), np.full(len(used), -1)
    donor.face_mat = [0] * len(donor.faces)
    donor.info = dict(kind="donor")
    member, twin = A["member"][used], remap[twin[used]]
    loop = remap[loop]

    # ---- the wrists and the hands
    arm_l = wrist_shell("arm_L", donor.verts[loop], loop, ring, info, A["plain_uv"])
    hand_l, nails_l = hm.hand_shells(shape, arm_l, BM.ATLAS, nails=bool(nails))
    arm_r = mirror_x(arm_l, "arm_R")
    arm_r.ext = {i: ("donor", int(twin[v])) for i, (_, v) in arm_l.ext.items()}
    shells = [donor, arm_l, hand_l] + nails_l + [arm_r]
    for sh in [hand_l] + nails_l:
        m = mirror_x(sh, sh.name[:-2] + "_R")
        m.info = sh.info
        shells.append(m)
    verts, faces, uvs, face_mat, ranges, _, _, _, gidx = G.join(shells)
    nv = len(verts)

    # ---- skeleton and weights
    land = {k_: np.asarray(v, float) for k_, v in shape.land.items()}
    bones = skeleton.standard_bones(land, cfg.get("skeleton"))
    deform = {b.name for b in bones if b.deform}
    names = {b.name for b in bones}
    out = {}

    def put(sh, W):
        own = np.ones(len(sh), bool)
        own[list(sh.ext)] = False
        g = gidx[sh.name][own]
        for bone, w in W.items():
            if bone in deform:
                out.setdefault(bone, np.zeros(nv))[g] += np.broadcast_to(np.asarray(w, float), (len(sh),))[own]

    put(donor, donor_weights(shape, donor.verts, member, deform))
    for sh in shells[1:]:
        side = sh.name[-1]
        kind = sh.info["kind"]
        if kind == "wrist":
            st = Shell(sh.name)
            st.verts, st.s, st.info = sh.verts, sh.s, sh.info
            put(sh, BW.arm_weights(shape, st, side))
        elif kind == "hand":
            put(sh, BW.hand_weights(shape, sh, side))
        elif kind == "nail":
            put(sh, BW.nail_weights(sh, side))
    tot = sum(out.values())
    if (tot < 1e-9).any():
        raise ValueError(f"{int((tot < 1e-9).sum())} body vertices without weights")
    weights = BW.cap_weights({b: w / np.maximum(tot, 1e-12) for b, w in out.items()}, 4)

    # ---- materials and the skin texture
    look = skin_look(ctx, cfg)
    pal = look["pal"]
    tex = body_tex.skin_texture(body_tex.marks(shells), pal, flush=look["flush"]).astype(float) / 255.0
    T = np.array([[f[0], f[k_], f[k_ + 1]] for f in donor.faces for k_ in range(1, len(f) - 1)])
    UVt = np.array([[u[0], u[k_], u[k_ + 1]] for u in donor.uv for k_ in range(1, len(u) - 1)])
    col = colour_of(marks(shape, donor.verts, member), pal, body_tex.base_tone(pal, look["flush"]), nt, look["neck"])
    img, painted = body_tex.bake(UVt, donor.verts[T], col, size=tex.shape[0])
    tex[..., :3] = np.where(painted[..., None], img, tex[..., :3])
    tex_name = ctx.save_png("skin", tex)
    toon_name = ctx.save_png("skin_toon", body_tex.toon_ramp(pal, look["mult"]))
    mats, mat_names = skin_materials(ctx, cfg, look, tex_name, toon_name)
    normals = _normals(verts, faces, nt)
    mesh = Mesh("body", verts=verts, faces=faces, uv=np.concatenate(uvs, 0), face_mat=face_mat, mats=mat_names,
                weights=weights, normals=normals, smooth=True)

    # ---- colliders and published information
    groups = F.bone_groups(list(weights))
    Gr = F.vertex_groups(weights, groups, nv)
    tri = G.triangulated(faces)
    fg = F.face_groups(tri, Gr)
    head = land["head"]
    bodies = F.fit_bodies(verts, Gr, land, None, names, head_hint=(head + np.array([0.0, -0.0103, 0.1007]), 0.094))
    bodies = shoulder_tops(bodies, verts, land)
    extra, tail_n = F.extra_landmarks(verts, tri, normals, land, fg)
    landmarks = dict(shape.land)
    landmarks.update(extra)
    has = Gr.sum(1) > 0.5
    dom = np.argmax(Gr, axis=1)
    info = dict(
        landmarks=landmarks,
        tail_normal=tail_n,
        neck_top=dict(nt, ring=neck_ring(nt, 32), n=32, start="front", dir="ccw_from_above"),
        **skin_info(look, tex_name, toon_name, nt["z"]),
        torso_profile=F.torso_profile(verts, tri, fg, float(land["lower_body"][2]) - 0.02, float(land["neck"][2])),
        regions={g: np.flatnonzero(has & (dom == k_)) for k_, g in enumerate(F.GROUPS)},
        hand=dict(frame=tuple(np.asarray(x) for x in shape.frame), design=shape.hand),
        foot=dict(shape.dims["foot"]),
        mesh="body",
        source="mesh",
    )
    return Part("body", meshes=[mesh], materials=mats, bones=bones, bodies=bodies,
                frames=skeleton.standard_frames(bones), info=info)


def shoulder_tops(bodies, verts, land):
    """The shoulder colliders as wide as the top of the shoulder: from the bone line up to the skin over it (the median
    within 30 degrees of straight up), where hair and straps rest. fit_bodies sizes a shoulder by the spread of the skin
    its bones own; this body's shoulder bones own a stretch of upper chest and back, and their ball stood 4 cm above the
    skin, so hair hanging behind the neck sat inside it. Never wider than fitted; the same segment as fit_bodies'."""
    def point(sem):
        p = np.asarray(land[f"{sem}.L"], float).copy()
        p[0] = 0.5 * (abs(land[f"{sem}.L"][0]) + abs(land[f"{sem}.R"][0]))
        return p
    a, b = point("shoulder") + np.array([0.0, 0.0, 0.012]), point("arm") + np.array([0.0, 0.0, 0.004])
    V = np.asarray(verts, float)
    V = np.concatenate([V[V[:, 0] > 0], V[V[:, 0] < 0] * np.array([-1.0, 1.0, 1.0])])
    d, t = F._seg_dist(V, a, b)
    over = ((V - (a + t[:, None] * (b - a)))[:, 2] > np.cos(np.radians(30.0)) * d) & (t > 0.0) & (t < 1.0)
    if over.sum() < 4:
        return bodies
    span = float(np.linalg.norm(b - a))
    out = []
    for rb in bodies:
        if rb.name in ("col_shoulder_L", "col_shoulder_R"):
            r = min(float(rb.size[0]), float(np.median(d[over])))
            rb = replace(rb, size=(r, max(span - 2.0 * r, 0.004), 0.0))
        out.append(rb)
    return out
