"""Grips: where a hand's fingers go on a prop, solved on the model's own skin (numpy + scipy, no bpy).

`HandModel` is one hand at rest (the `.npz` the Blender op `hand_model` writes): the wrist and five finger chains, the
skinned vertices near the hand and their weights, and the anatomy the solvers work in (along / across / ventral axes,
finger flexion and spread axes, thumb axes, the vertex set of every phalanx surface). A pose is 21 joint angles
(`DOF`); `fk` + `skin` give the posed bones and skin by linear blend skinning, exactly what Blender does.

Styles, each `solve_<style>(hand, prop, **params) -> dict`:
  pen    lateral tripod on a pen: index pad on top near the nib, thumb pad on the side, middle finger underneath,
         the barrel crossing the thumb-index web; optionally the writing orientation of the whole hand
  wheel  power grip on a ring (torus): fingers wrap the tube, the thumb opposes
  pinch  thumb-index pad pinch of an object `width` thick
  rest   relaxed hand resting on a plane

Result: {"style", "side", "bones": {blender_bone: [w, x, y, z]}, "target_in_wrist": 4x4, "report": {...}, ...}.
`bones` are pose-bone `rotation_quaternion` values, bone-local and relative to rest (what Blender stores; a bone that
the solve leaves alone has the identity). `target_in_wrist` is the grip frame (defined per style, see the solvers)
in the wrist bone's rest frame at the wrist head: with the wrist posed, `frame_world = wrist_bone_world @
target_in_wrist`, so the build puts the wrist where `prop_grip_frame_world @ inv(target_in_wrist)` says. Units are
metres, angles degrees in reports and radians inside; all geometry is in the hand model's (world) frame."""
import hashlib
import json
import math
import multiprocessing
import os
import sys
import time

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as Rot

from ..core import bonemap, jsonx
from . import geom

VERSION = 1                                         # bump when a solver's results change (cache keys)
FINGERS = tuple(bonemap.FINGERS)                    # thumb index middle ring little; 4 chain entries each
FOUR = ("index", "middle", "ring", "little")
DOF = [f"{f}_{j}" for f in FOUR for j in ("mcp", "pip", "dip", "spr")] + ["t_palmar", "t_radial", "t_roll", "t_mcp",
                                                                          "t_ip"]
_THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")


# ---------------------------------------------------------------- small maths
def _unit(v):
    return v / np.linalg.norm(v)


def _rodrigues(axes, rad):
    """(m, 3) unit axes and (m,) angles -> (m, 3, 3) right-handed rotations."""
    c, s = np.cos(rad)[:, None, None], np.sin(rad)[:, None, None]
    return np.eye(3)[None] * c + geom.skew(axes) * s + (1.0 - c) * (axes[:, :, None] * axes[:, None, :])


def _rotm(axis, deg):
    return _rodrigues(_unit(np.asarray(axis, float))[None], np.array([math.radians(deg)]))[0]


def _softmin(v, tau):
    m = float(np.min(v))
    return m - tau * math.log(float(np.sum(np.exp(-(v - m) / tau))))


def _perp(v, axis):
    """v with its component along unit `axis` removed."""
    return v - (v @ axis) * axis


def _frame(R, p):
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, p
    return T


# ---------------------------------------------------------------- the hand
class HandModel:
    """One hand at rest. Chain order: wrist, then thumb / index / middle / ring / little with four entries each
    (thumb0 thumb1 thumb2 tip; index1 index2 index3 tip; ...); `F[finger]` are those chain indices. Names are Blender
    bone names ("" = a virtual tip the model lacks). Everything is in one frame (the world at export time)."""

    def __init__(self, side, names, parents, heads, tails, rest_rot, V, W, sem=None, arm_world=None, arm_chain=None):
        self.side = str(side).upper()
        if self.side not in ("L", "R"):
            raise ValueError(f"side must be L or R, got {side!r}")
        self.chir = 1.0 if self.side == "R" else -1.0
        self.names = [str(n) for n in names]
        self.sem = [str(s) for s in sem] if sem is not None else [""] * len(self.names)
        self.parents = np.asarray(parents, int)
        self.h = np.asarray(heads, float)
        self.tails = np.asarray(tails, float)
        self.rest_rot = np.asarray(rest_rot, float)
        self.V = np.asarray(V, float)
        self.W = np.asarray(W, float)
        self.nb = len(self.h)
        if self.nb != 1 + 4 * len(FINGERS) or self.W.shape != (len(self.V), self.nb + 1):
            raise ValueError(f"hand model needs {1 + 4 * len(FINGERS)} chain entries and weights (n, {self.nb + 1})")
        self.arm_world = np.eye(4) if arm_world is None else np.asarray(arm_world, float)
        self.arm_chain = np.full((3, 3), np.nan) if arm_chain is None else np.asarray(arm_chain, float)
        self.F = {f: [1 + 4 * k + j for j in range(4)] for k, f in enumerate(FINGERS)}
        self._anatomy()
        self._skinning()

    @classmethod
    def from_arrays(cls, a):
        """From the arrays the Blender op `hand_model` writes (a dict or an open .npz): side, names, parents, heads,
        tails, rest_rot, V, W and optionally sem, arm_world, arm_chain."""
        opt = {k: a[k] for k in ("sem", "arm_world", "arm_chain") if k in a}
        return cls(str(a["side"]), a["names"], a["parents"], a["heads"], a["tails"], a["rest_rot"], a["V"], a["W"],
                   **opt)

    @classmethod
    def from_npz(cls, path):
        with np.load(path, allow_pickle=False) as z:
            return cls.from_arrays(z)

    def digest(self):
        """Content hash (cache keys)."""
        h = hashlib.sha1(self.side.encode())
        for a in (self.parents, self.h, self.tails, self.rest_rot, self.V, self.W):
            h.update(np.ascontiguousarray(a).tobytes())
        h.update("\0".join(self.names).encode("utf-8"))
        return h.hexdigest()

    # ---- rest anatomy
    def _anatomy(self):
        h, F = self.h, self.F
        self.along = _unit(h[F["middle"][0]] - h[0])                    # wrist -> fingers
        self.across = _unit(h[F["little"][0]] - h[F["index"][0]])       # index -> little finger
        self.ventral = _unit(np.cross(self.along, self.across)) * self.chir     # out of the palm
        self.dorsal = -self.ventral
        self.radial = -self.across                                      # toward the thumb side
        self.palm_c = np.mean([h[F[f][0]] for f in FOUR] + [h[0]], axis=0)
        self.palm_length = float(np.linalg.norm(h[F["middle"][0]] - h[0]))
        self.bone_dir = np.array([_unit((h[i + 1] if i + 1 < self.nb and self.parents[i + 1] == i else self.tails[i])
                                        - h[i]) for i in range(self.nb)])
        self.dom = np.argmax(self.W, axis=1)                            # dominant bone per vertex (nb = not in chain)
        self.radius = np.array([self._seg_radius(i) for i in range(self.nb)])
        self.flex, self.spread = {}, {}
        for f in FOUR:
            d = self.bone_dir[F[f][0]]
            self.flex[f] = _unit(np.cross(d, self.ventral))             # + curls the finger toward the palm
            self.spread[f] = _unit(np.cross(d, self.radial))            # + swings it toward the thumb
        t = F["thumb"]
        self.t_dir = _unit(h[t[3]] - h[t[1]])
        # the gloved thumb has no nail, so its pad is the side the model's own rest thumb bends toward at the IP joint
        # (ulnar-ventral, as anatomy says); fall back to straight ulnar-ventral if the rest chain is nearly straight
        bend = _unit(h[t[3]] - h[t[2]]) - _unit(h[t[2]] - h[t[1]])
        bend = _perp(bend, self.t_dir)
        if np.linalg.norm(bend) < 0.02:
            bend = _perp(self.across + self.ventral, self.t_dir)
        self.pad_t = _unit(bend)
        self.flex["thumb"] = _unit(np.cross(self.t_dir, self.pad_t))   # + bends the thumb toward its pad
        self.t_palmar = _unit(np.cross(self.t_dir, self.ventral))      # swings the thumb out of the palm's plane
        self.t_radial = _unit(np.cross(self.t_dir, self.radial))       # swings it away from the index, in the plane
        ax = []
        for f in FOUR:
            ax += [self.flex[f], self.flex[f], self.flex[f], self.spread[f]]
        ax += [self.t_palmar, self.t_radial, self.chir * self.t_dir, self.flex["thumb"], self.flex["thumb"]]
        self._axes = np.array(ax)
        k = {n: i for i, n in enumerate(DOF)}
        self._dof = {j: np.array([k[f"{f}_{j}"] for f in FOUR]) for j in ("mcp", "pip", "dip", "spr")}
        self._finger_bone = {j: np.array([self.F[f][n] for f in FOUR]) for n, j in enumerate(("mcp", "pip", "dip"))}
        self._first = np.array([self.F[f][0] for f in FOUR])

    def _seg_radius(self, i):
        sel = self.dom == i
        if sel.sum() < 4:
            return 0.006
        a = self.h[i]
        b = self.h[i + 1] if i + 1 < self.nb and self.parents[i + 1] == i else self.tails[i]
        ab = b - a
        t = np.clip((self.V[sel] - a) @ ab / (ab @ ab), 0, 1)
        return float(np.median(np.linalg.norm(self.V[sel] - (a + np.outer(t, ab)), axis=1)))

    # ---- skinning and kinematics
    def _skinning(self):
        K = int(max(1, (self.W > 0).sum(1).max()))
        order = np.argsort(-self.W, axis=1)[:, :K]
        self._inf_idx = order
        self._inf_w = np.take_along_axis(self.W, order, 1)
        depth = np.zeros(self.nb, int)
        for i in range(self.nb):
            d, p = 0, self.parents[i]
            while p >= 0:
                d, p = d + 1, self.parents[p]
            depth[i] = d
        self._levels = [np.where(depth == d)[0] for d in range(depth.max() + 1)]

    def rotations(self, deg):
        """(nb, 3, 3) joint rotations in rest-world axes (child delta relative to its parent's delta) for the 21 joint
        angles `deg` in `DOF` order: per finger mcp pip dip spread (+ curls toward the palm, spread toward the thumb),
        then thumb palmar abduction, radial abduction, roll, mcp, ip."""
        R = _rodrigues(self._axes, np.radians(np.asarray(deg, float)))
        Q = np.broadcast_to(np.eye(3), (self.nb, 3, 3)).copy()
        Q[self._finger_bone["mcp"]] = R[self._dof["mcp"]] @ R[self._dof["spr"]]
        Q[self._finger_bone["pip"]] = R[self._dof["pip"]]
        Q[self._finger_bone["dip"]] = R[self._dof["dip"]]
        t = self.F["thumb"]
        Q[t[0]] = R[16] @ R[17] @ R[18]
        Q[t[1]] = R[19]
        Q[t[2]] = R[20]
        return Q

    def fk(self, Q):
        """Posed rotations D and bone heads H: D_child = D_parent @ Q_child, H_child = H_parent + D_parent (h_child -
        h_parent)."""
        D, H = np.empty((self.nb, 3, 3)), np.empty((self.nb, 3))
        for n, idx in enumerate(self._levels):
            if n == 0:
                D[idx], H[idx] = Q[idx], self.h[idx]
            else:
                p = self.parents[idx]
                D[idx] = D[p] @ Q[idx]
                H[idx] = H[p] + (D[p] @ (self.h[idx] - self.h[p])[:, :, None])[..., 0]
        return D, H

    def skin(self, D, H):
        """(nv, 3) posed skin vertices (linear blend skinning; the last weight column stays put)."""
        A = np.empty((self.nb + 1, 3, 4))
        A[:self.nb, :, :3] = D
        A[:self.nb, :, 3] = H - np.einsum("bij,bj->bi", D, self.h)
        A[self.nb, :, :3] = np.eye(3)
        A[self.nb, :, 3] = 0.0
        Ak = A[self._inf_idx]
        moved = np.matmul(Ak[..., :3], self.V[:, None, :, None])[..., 0] + Ak[..., 3]
        return np.einsum("vk,vki->vi", self._inf_w, moved)

    def pose(self, deg):
        Q = self.rotations(deg)
        D, H = self.fk(Q)
        return Q, D, H, self.skin(D, H)

    # ---- surfaces
    def surface(self, bone, side_dir, lo=0.0, hi=1.0, min_cos=0.3, bones=None):
        """Vertex indices of a phalanx's surface facing `side_dir` (a rest direction): vertices dominated by `bone`
        (or by any of `bones`) whose offset from the bone axis points that way (cosine > min_cos), between the
        fractions lo..hi of the vertex extent along the bone (a round fingertip touches anywhere on that half)."""
        idx = np.where(np.isin(self.dom, [bone] if bones is None else bones))[0]
        if not len(idx):
            raise ValueError(f"no skin is weighted mostly to {self.names[bone] or bone}")
        a = self.bone_dir[bone]
        rel = self.V[idx] - self.h[bone]
        s = rel @ a
        off = rel - np.outer(s, a)
        f = (s - s.min()) / max(s.max() - s.min(), 1e-12)
        keep = ((off @ side_dir) > min_cos * np.linalg.norm(off, axis=1)) & (f >= lo - 1e-9) & (f <= hi + 1e-9)
        return idx[keep]

    # ---- results in Blender terms
    def local_quats(self, Q):
        """{blender bone: [w, x, y, z]} pose-bone rotation_quaternion for the joint rotations Q: R^T Q R with R the
        bone's rest orientation (w >= 0). Every finger joint bone is listed (identity when unrotated); the wrist and
        the tip bones are not driven."""
        out = {}
        keys = [i for f in FINGERS for i in self.F[f][:3] if self.names[i]]
        R = self.rest_rot[keys]
        q = geom.mat_to_quat(np.swapaxes(R, 1, 2) @ Q[keys] @ R)
        q = np.where(q[:, :1] < 0, -q, q)
        for i, qi in zip(keys, q):
            out[self.names[i]] = [float(c) for c in qi]
        return out

    def rotations_from_local(self, bones):
        """Inverse of local_quats: joint rotations Q (nb, 3, 3) from {blender bone: [w, x, y, z]}."""
        Q = np.broadcast_to(np.eye(3), (self.nb, 3, 3)).copy()
        for i, n in enumerate(self.names):
            if n and n in bones:
                Q[i] = self.rest_rot[i] @ geom.quat_to_mat(np.asarray(bones[n], float)[None])[0] @ self.rest_rot[i].T
        return Q

    def frame_in_wrist(self, R, p):
        """4x4: the frame with axes R (columns) at p, in the wrist bone's rest frame at the wrist head."""
        Bw = self.rest_rot[0]
        return _frame(Bw.T @ R, Bw.T @ (np.asarray(p, float) - self.h[0]))


# ---------------------------------------------------------------- running seeds
def _run_seeds(fn, payloads, workers):
    """fn(payload) for every payload: in this process (workers 1) or in spawned processes with one BLAS thread each.
    Spawned workers re-import the caller's __main__, so scripts that pass workers > 1 need the usual
    `if __name__ == "__main__":` guard (the mk CLI has it)."""
    workers = min(len(payloads), (os.cpu_count() or 1) if workers is None else int(workers))
    if workers <= 1:
        return [fn(p) for p in payloads]
    saved = {k: os.environ.get(k) for k in _THREAD_VARS}
    for k in _THREAD_VARS:
        os.environ[k] = "1"
    try:
        with multiprocessing.get_context("spawn").Pool(workers) as pool:
            return pool.map(fn, payloads)
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _hand_args(hand):
    return dict(side=hand.side, names=hand.names, parents=hand.parents, heads=hand.h, tails=hand.tails,
                rest_rot=hand.rest_rot, V=hand.V, W=hand.W, sem=hand.sem, arm_world=hand.arm_world,
                arm_chain=hand.arm_chain)


# ---------------------------------------------------------------- shared pieces
CLASH_PAIRS = (("index", "middle"), ("middle", "ring"), ("ring", "little"), ("thumb", "index"), ("thumb", "middle"))


class _Clash:
    """Finger-inside-finger test: the skin of the first three bones of finger fa against the first three segments of
    finger fb, for each (fa, fb) in `pairs`; a vertex deeper than 0.92 of the segment's skin radius counts."""

    def __init__(self, hand, pairs=CLASH_PAIRS):
        v, b, r = [], [], []
        for fa, fb in pairs:
            va = np.where(np.isin(hand.dom, hand.F[fa][:3]))[0]
            bs = np.array(hand.F[fb][:3])
            v.append(va)
            b.append(np.tile(bs, (len(va), 1)))
            r.append(np.tile(0.92 * hand.radius[bs], (len(va), 1)))
        self.v, self.b, self.r = np.concatenate(v), np.concatenate(b), np.concatenate(r)

    def depth(self, P, H):
        """Depth (m, >= 0) of skin inside the next finger's segments, per (vertex, segment)."""
        a, b = H[self.b], H[self.b + 1]
        ab = b - a
        q = P[self.v][:, None, :] - a
        t = np.clip(np.einsum("mkj,mkj->mk", q, ab) / np.maximum(np.einsum("mkj,mkj->mk", ab, ab), 1e-12), 0.0, 1.0)
        return np.maximum(0.0, self.r - np.linalg.norm(q - t[..., None] * ab, axis=2)).ravel()


def _angles(deg):
    """{joint name: angle (deg, 1 decimal)} for the 21 joint angles in DOF order."""
    return {k: round(float(v), 1) for k, v in zip(DOF, deg)}


def _result(style, hand, Q, R, p, report, **extra):
    """The result every solver returns: local bone rotations for the joint rotations Q, the grip frame (axes R, origin
    p, in the hand model's frame) in the wrist's frame, the report and any extras."""
    return {"style": style, "side": hand.side, "bones": hand.local_quats(Q),
            "target_in_wrist": hand.frame_in_wrist(R, p), "report": report, **extra}


# ================================================================ pen: the lateral tripod
# Looking down the barrel from the cap: the index pad on top (~1 o'clock) 18-27 mm from the nib, the radial side of
# the middle finger's last phalanx underneath (~5 o'clock) 10-22 mm, the thumb pad on the radial side (~9 o'clock)
# 28-55 mm. Each contact is the pad half of that phalanx (smooth softmin over its vertices) against its slot on the
# barrel surface. The barrel crosses the thumb-index web on its dorsal side and rests on the web skin; the ring and
# little fingers curl; no hand vertex is inside the pen, no finger inside another; joint angles stay near measured
# tripod priors. "clock": direction around the barrel in the (top, thumb side) basis, 0 = top, +90 = thumb side,
# 180 = underneath. Slot: (clock deg, from, to along the barrel from the nib, m).
PEN_SLOTS = {"index": (-20.0, 0.018, 0.027), "thumb": (100.0, 0.028, 0.055), "middle": (-140.0, 0.010, 0.022)}
PEN_NAMES = ["i_mcp", "i_pip", "i_dip", "i_spr", "m_mcp", "m_pip", "m_dip", "m_spr", "r_curl", "l_curl",
             "t_palmar_abd", "t_radial_abd", "t_roll", "t_mcp", "t_ip", "px", "py", "pz", "ua", "ub", "gx", "gy", "gz"]
# priors: the dynamic tripod as measured on writers (index MCP/PIP/DIP ~45/35/5, middle tucked under it ~55/60/25,
# thumb swung away from the palm and rolled so its pad turns toward the fingers); bounds keep out hooks and fists
PEN_PRIOR = np.array([45, 35, 5, 2, 55, 60, 25, 0, 1.0, 1.0, 35, 0, -20, 10, 20], float)
PEN_SIGMA = np.array([12, 15, 10, 6, 12, 15, 12, 6, 0.15, 0.15, 15, 15, 25, 12, 15], float)
PEN_LO = np.array([15, 10, -20, -12, 25, 30, 0, -12, 0.6, 0.6, 0, -35, -80, -10, -20, -80, -80, -80, -80, -80,
                   -4, -4, -4], float)
PEN_HI = np.array([70, 70, 30, 15, 80, 90, 45, 12, 1.3, 1.3, 70, 35, 40, 45, 60, 80, 80, 80, 80, 80, 4, 4, 4], float)
RING_CURL = (55.0, 80.0, 40.0)                       # ring / little at curl 1: mcp pip dip (deg)
LITTLE_CURL = (60.0, 85.0, 42.0)
# writing posture (value, tolerance): pen 45-60 deg above the paper, pointing at or a little outside the shoulder (the
# lateral tripod crosses the hand obliquely, so rolling the hand outward swings the barrel outward), back of the hand
# rolled ~35 deg outward, wrist a little extended and ulnar-deviated. Azimuth and tilt count outward as positive.
PEN_TARGET = {"elevation": (52.0, 7.0), "azimuth": (12.0, 15.0), "tilt": (35.0, 6.0), "extension": (15.0, 12.0),
              "ulnar": (10.0, 10.0)}


def _radius_fn(radius, tip=None):
    """r(t): the barrel radius (m) at distance t (m) from the nib. A number is a cylinder with a conical tip
    (0.15 r at the nib, full radius at `tip`, default 3 r); a list of (t, r) pairs is a piecewise-linear profile
    (flat beyond its ends)."""
    if isinstance(radius, (int, float)):
        r = float(radius)
        cone = float(tip) if tip else 3.0 * r
        return lambda t: r * np.clip(np.asarray(t, float) / cone, 0.15, 1.0)
    prof = np.asarray(radius, float).reshape(-1, 2)
    prof = prof[np.argsort(prof[:, 0])]
    return lambda t: np.interp(np.asarray(t, float), prof[:, 0], prof[:, 1])


class _Pen:
    """The pen grip problem on one hand (see solve_pen); x = 15 finger parameters, the pen's position (mm along the
    hand's along / radial / ventral axes, so the problem does not depend on how the hand is placed or mirrored) and
    orientation (deg) relative to the design start, then the hand's world rotation vector (rad)."""

    def __init__(self, hand, length, radius, tip=None, slots=None, posture=None):
        self.hand, self.length = hand, float(length)
        self.rad = _radius_fn(radius, tip)
        self.slots = {k: tuple(v) for k, v in (slots or PEN_SLOTS).items()}
        hd, F = hand, hand.F
        h = hd.h
        self.pads = {"index": hd.surface(F["index"][2], hd.ventral, 0.15, 0.92),
                     "thumb": hd.surface(F["thumb"][2], hd.pad_t, 0.20, 0.92),
                     "middle": hd.surface(F["middle"][2], hd.radial, 0.0, 0.80)}
        # pen start, by design: the index arched along the top of the barrel, its pad ahead of and below the knuckle;
        # the barrel leaving over the thumb-index web
        pinch0 = h[F["index"][0]] + 0.028 * hd.along + 0.049 * hd.ventral + 0.002 * hd.radial
        wl = h[F["index"][0]] - h[F["thumb"][1]]
        web0 = h[F["thumb"][1]] + 0.45 * wl + 0.016 * _unit(_perp(hd.dorsal, _unit(wl)))
        self.U0 = _unit(web0 - pinch0)
        self.P0 = pinch0 - self.U0 * 0.022
        self.E1 = _unit(np.cross(self.U0, hd.dorsal))
        self.E2 = _unit(np.cross(self.U0, self.E1))
        # the first web space: skin of the index base, thumb base and palm within 16 mm of the line from the thumb's
        # MCP joint to the index knuckle (both stay put when the fingers bend). The barrel crosses that line between
        # the two joints, on its dorsal side (it leaves the hand between thumb and index, not over the palm or the
        # knuckles).
        self.web_a, self.web_b = h[F["thumb"][1]], h[F["index"][0]]
        ab = self.web_b - self.web_a
        s = np.clip((hd.V - self.web_a) @ ab / (ab @ ab), 0.0, 1.0)
        near = np.linalg.norm(hd.V - (self.web_a + np.outer(s, ab)), axis=1) < 0.016
        self.web_set = np.where(np.isin(hd.dom, [F["index"][0], F["thumb"][0], F["thumb"][1], 0]) & near)[0]
        self.web_up = _unit(_perp(hd.dorsal, _unit(ab)))
        self.clash = _Clash(hd, CLASH_PAIRS)
        self.hand_vertices = np.where(hd.dom < hd.nb)[0]
        self.posture = None if posture is None else _Posture(hand, posture)

    # ---- parameters
    def degrees(self, x):
        """The 21 joint angles (deg, DOF order) for the 15 finger parameters x[:15]."""
        deg = np.zeros(21)
        deg[0:4], deg[4:8] = x[0:4], x[4:8]
        deg[8:12] = [RING_CURL[0] * x[8], RING_CURL[1] * x[8], RING_CURL[2] * x[8], -3.0]
        deg[12:16] = [LITTLE_CURL[0] * x[9], LITTLE_CURL[1] * x[9], LITTLE_CURL[2] * x[9], -6.0]
        deg[16:21] = x[10:15]
        return deg

    def pen_pose(self, x):
        # E1 = U0 x dorsal turns like a cross product (mirrors with the same angle); E2 = U0 x E1 is a plain vector, so
        # a mirrored hand turns about it the other way
        E1, E2 = _rotm(self.E1, x[18]), _rotm(self.E2, self.hand.chir * x[19])
        hd = self.hand
        return self.P0 + 1e-3 * (x[15] * hd.along + x[16] * hd.radial + x[17] * hd.ventral), _unit(E1 @ E2 @ self.U0)

    def barrel_frame(self, u):
        top = _unit(_perp(self.hand.dorsal, u))                 # toward the back of the hand
        side = np.cross(u, top)
        if side @ self.hand.radial < 0:
            side = -side
        return top, side

    # ---- geometry against the pen
    def pen_gap(self, P, p, u):
        rel = P - p
        t = rel @ u
        tc = np.clip(t, 0.0, self.length)
        radial = rel - np.outer(tc, u)
        rho = np.maximum(np.linalg.norm(radial, axis=1), 1e-9)
        return t, radial / rho[:, None], rho - self.rad(tc)

    def pad_contact(self, idx, P, p, u, t, rdir, gap, top, side, slot):
        """A pad vertex set against its slot on the barrel surface: softmin distance to the slot's segment (one basin
        per finger, so each finger keeps its side of the barrel), plus where it actually touches (softmin over the
        gap)."""
        deg, t0, t1 = slot
        ct, cs = math.cos(math.radians(deg)), math.sin(math.radians(deg))
        ts = np.clip((P[idx] - p) @ u, t0, t1)
        target = p + np.outer(ts, u) + self.rad(ts)[:, None] * (ct * top + cs * side)
        d = np.linalg.norm(P[idx] - target, axis=1)
        g = gap[idx]
        w = np.exp(-(g - g.min()) / 0.0008)
        w /= w.sum()
        rk = _unit(w @ rdir[idx])
        return dict(dist=_softmin(d, 0.001), gap=_softmin(g, 0.0004), t=float(w @ t[idx]),
                    clock=math.degrees(math.atan2(rk @ side, rk @ top)))

    def web_cross(self, p, u):
        """Where the pen axis passes the web line: (t along the pen, s along thumb MCP -> index MCP, dorsal-ness of
        the offset from the web line to the axis)."""
        d = self.web_b - self.web_a
        r = p - self.web_a
        b, c, e, f = u @ d, d @ d, u @ r, d @ r
        den = c - b * b
        t = (b * f - c * e) / den
        s = (f - b * e) / den
        off = (p + t * u) - (self.web_a + s * d)
        return t, s, float(_unit(off) @ self.web_up)

    # ---- the whole state at x
    def analyse(self, x):
        hd = self.hand
        Q, D, H, P = hd.pose(self.degrees(x))
        p, u = self.pen_pose(x)
        top, side = self.barrel_frame(u)
        t, rdir, gap = self.pen_gap(P, p, u)
        con = {k: self.pad_contact(idx, P, p, u, t, rdir, gap, top, side, self.slots[k])
               for k, idx in self.pads.items()}
        tw = t[self.web_set]
        g_web = gap[self.web_set] + 2.0 * np.maximum(0.0, 0.055 - tw) + 2.0 * np.maximum(0.0, tw - 0.110)
        inside = (t > -0.002) & (t < self.length)
        A = dict(Q=Q, D=D, H=H, P=P, p=p, u=u, top=top, side=side, con=con, web=_softmin(g_web, 0.0005),
                 web_t=float(tw[np.argmin(g_web)]), web_x=self.web_cross(p, u),
                 pen=np.where(inside, np.maximum(0.0, 0.0008 - gap), 0.0), clash=self.clash.depth(P, H),
                 gap_min=float(np.min(np.where(inside, gap, np.inf))))
        if self.posture is not None:
            A["G"] = _rotvec(x[20:23])
            A["wr"] = self.posture.writing(A, A["G"], hd, self.hand_vertices)
        return A

    def residuals(self, x, stage="all"):
        """stage 'reach': pads on their slots + web + priors (fingers may pass through the barrel on the way);
        'grip': + no hand inside the pen, no finger inside another; 'all': + the writing posture through the hand's
        rotation G = x[20:23]."""
        A = self.analyse(x)
        c = A["con"]
        r = []
        for k in ("index", "thumb", "middle"):
            r += [c[k]["dist"] / 0.0006, (c[k]["gap"] - 0.0002) / 0.0006]
        wt, ws, wup = A["web_x"]
        r += [(A["web"] - 0.0010) / 0.0006, max(0.0, 0.2 - ws) / 0.08, max(0.0, ws - 0.9) / 0.08,
              max(0.0, 0.6 - wup) / 0.15, max(0.0, 0.055 - wt) / 0.005, max(0.0, wt - 0.110) / 0.005]
        r = np.concatenate([r, 0.5 * (x[:15] - PEN_PRIOR) / PEN_SIGMA])
        if stage == "reach":
            return r
        r = np.concatenate([r, A["pen"] / 0.0002, A["clash"] / 0.0004])
        if stage == "all" and self.posture is not None:
            r = np.concatenate([r, self.posture.residuals(A["wr"])])
        return r

    # ---- solving
    X_SCALE = np.r_[PEN_SIGMA, np.full(5, 20.0), np.full(3, 0.3)]

    def grip_seed(self, seed, nfev=(800, 800, 1500)):
        """Fingers onto the designed barrel (pen pinned: reach, then walls), then fingers + pen together."""
        rng = np.random.default_rng(seed)
        x0 = np.clip(np.r_[PEN_PRIOR + (rng.normal(0, 0.35, 15) * PEN_SIGMA if seed else 0.0), np.zeros(8)],
                     PEN_LO + 1e-6, PEN_HI - 1e-6)
        z = x0[:15]
        for stage, n in zip(("reach", "grip"), nfev[:2]):
            z = least_squares(lambda z_: self.residuals(np.r_[z_, np.zeros(8)], stage), z,
                              bounds=(PEN_LO[:15], PEN_HI[:15]), x_scale=self.X_SCALE[:15], max_nfev=n, xtol=1e-12).x
        r = least_squares(lambda z_: self.residuals(np.r_[z_, np.zeros(3)], "grip"), np.r_[z, np.zeros(5)],
                          bounds=(PEN_LO[:20], PEN_HI[:20]), x_scale=self.X_SCALE[:20], max_nfev=nfev[2], xtol=1e-12)
        return float(r.cost), r.x

    def solve_orientation(self, x, starts=32, max_nfev=2000):
        """The hand's world rotation G (rotation vector) that best meets the writing posture."""
        A = self.analyse(x)
        wp = self.posture

        def f(g):
            wr = wp.writing(A, _rotvec(g), self.hand, self.hand_vertices)
            return np.array([(wr[k] - m) / s for k, (m, s) in wp.target.items()] + [(wr["clear"] - 0.0008) / 0.0005]
                            + list(wr["floor"] / 0.004))

        best = None
        for k in range(starts):
            g0 = Rot.from_quat(np.random.default_rng(200 + k).normal(size=4)).as_rotvec()
            r = least_squares(f, g0, max_nfev=max_nfev)
            if best is None or r.cost < best.cost:
                best = r
        return Rot.from_rotvec(best.x).as_rotvec()               # canonical (|v| <= pi), inside the joint bounds

    def report(self, x, A):
        c = A["con"]
        rep = {"contacts": {k: {"gap_mm": round(v["gap"] * 1e3, 2), "from_nib_mm": round(v["t"] * 1e3, 1),
                                "clock_deg": round(v["clock"], 0)} for k, v in c.items()},
               "web": {"gap_mm": round(A["web"] * 1e3, 2), "from_nib_mm": round(A["web_t"] * 1e3, 1),
                       "crossing": {"from_nib_mm": round(A["web_x"][0] * 1e3, 1),
                                    "thumb_to_index": round(A["web_x"][1], 2), "dorsal": round(A["web_x"][2], 2)}},
               "penetration_mm": round(max(0.0, -A["gap_min"]) * 1e3, 3),
               "clearance_mm": 0.8, "margin_depth_mm": round(float(A["pen"].max()) * 1e3, 3),
               "margin_hits": int((A["pen"] > 0).sum()),
               "finger_clash_mm": round(float(A["clash"].max()) * 1e3, 2),
               "angles_deg": {k: round(float(v), 1) for k, v in zip(PEN_NAMES[:15], x[:15])}}
        if self.posture is not None:
            rep["writing"] = self.posture.report(A["wr"])
        return rep


def _rotvec(v):
    th = float(np.linalg.norm(v))
    return np.eye(3) if th < 1e-12 else _rodrigues((np.asarray(v, float) / th)[None], np.array([th]))[0]


class _Posture:
    """The writing arm and desk the pen's orientation is solved against. World coordinates (hand-model frame), `up`
    the vertical (default +Z), `facing` the character's horizontal heading (default -Y, so "outward" is -X for a right
    hand). Keys of `p`:
      nib          [x, y, z] where the nib writes; a list of points (a nib schedule) is averaged. Required.
      paper_z      height of the paper above the origin along `up` (default: the nib's height)
      shoulder     the arm's shoulder joint (a list of points is averaged; default: the model's arm bone head)
      pole         the elbow pole target, the elbow bends toward it (default: outward, back and down of the shoulder)
      upper, fore  arm segment lengths (default: the model's)
      table        {"z": top height, "center": [x, y], "radius": R}: the forearm stays `forearm_lift` above the top
                   wherever it is over the table (all of it when center/radius are omitted)
      forearm_lift (0.028), wrist_lift (0.032: wrist joint above the paper), reach (0.97 of the arm's length)
      target       {"elevation" | "azimuth" | "tilt" | "extension" | "ulnar": (value deg, tolerance deg)}
    """

    def __init__(self, hand, p):
        def point(key, default=None):
            v = p.get(key)
            if v is None:
                return default
            v = np.asarray(v, float)
            return v.mean(0) if v.ndim == 2 else v

        self.up = _unit(np.asarray(p.get("up", [0.0, 0.0, 1.0]), float))
        self.facing = _unit(_perp(np.asarray(p.get("facing", [0.0, -1.0, 0.0]), float), self.up))
        self.sign = hand.chir
        self.outward = np.cross(self.facing, self.up) * self.sign
        nib = point("nib")
        if nib is None:
            raise ValueError("a writing posture needs the nib position (posture['nib'])")
        self.paper_z = float(p["paper_z"]) if p.get("paper_z") is not None else float(nib @ self.up)
        self.nib_ref = nib + (self.paper_z - nib @ self.up) * self.up
        ac = hand.arm_chain
        self.shoulder = point("shoulder", ac[0])
        if np.isnan(self.shoulder).any():
            raise ValueError("a writing posture needs the shoulder (posture['shoulder']): the model has no arm bone")
        self.pole = point("pole", self.shoulder + 0.46 * self.outward - 0.21 * self.facing - 0.13 * self.up)
        self.l_up = float(p["upper"]) if p.get("upper") is not None else float(np.linalg.norm(ac[0] - ac[1]))
        self.l_fore = float(p["fore"]) if p.get("fore") is not None else float(np.linalg.norm(ac[1] - ac[2]))
        if not (self.l_up > 0 and self.l_fore > 0):
            raise ValueError("a writing posture needs the arm lengths (posture['upper'], ['fore'])")
        tb = p.get("table")
        self.table = None
        if tb is not None:
            c = tb.get("center")
            self.table = (float(tb["z"]), None if c is None else np.r_[np.asarray(c, float), 0.0][:3],
                          None if tb.get("radius") is None else float(tb["radius"]))
        self.forearm_lift = float(p.get("forearm_lift", 0.028))
        self.wrist_lift = float(p.get("wrist_lift", 0.032))
        self.reach_max = float(p.get("reach", 0.97))
        self.target = dict(PEN_TARGET)
        for k, v in (p.get("target") or {}).items():
            if k not in PEN_TARGET:
                raise ValueError(f"unknown posture target {k!r} (have {', '.join(PEN_TARGET)})")
            self.target[k] = (float(v[0]), float(v[1]))

    def elbow(self, wrist):
        d = wrist - self.shoulder
        L = float(np.linalg.norm(d))
        e = d / L
        Lc = min(L, 0.999 * (self.l_up + self.l_fore))
        a = (self.l_up ** 2 - self.l_fore ** 2 + Lc ** 2) / (2 * Lc)
        hh = math.sqrt(max(self.l_up ** 2 - a ** 2, 0.0))
        perp = _unit(_perp(self.pole - self.shoulder, e))
        return self.shoulder + a * e + hh * perp, L

    def writing(self, A, G, hand, hand_vertices):
        """Writing-posture quantities with the hand + pen rotated by G about the nib, nib at the paper."""
        up, p = self.up, A["p"]
        z = ((A["P"][hand_vertices] - p) @ G.T) @ up + (self.nib_ref @ up) - self.paper_z
        u = G @ A["u"]
        elev = math.degrees(math.asin(float(np.clip(u @ up, -1, 1))))
        az_t = _unit(_perp(self.shoulder - self.nib_ref, up))
        u_h = _perp(u, up)
        az = self.sign * math.degrees(math.atan2(float(np.cross(az_t, u_h) @ up), float(az_t @ u_h)))
        dw = G @ hand.dorsal
        tilt = math.degrees(math.atan2(float(dw @ self.outward), float(dw @ up)))
        Wp = self.nib_ref + G @ (hand.h[0] - p)
        E, reach = self.elbow(Wp)
        f = _unit(Wp - E)
        along = G @ hand.along
        d_f = _unit(_perp(dw, f))
        a_f = G @ hand.across
        a_f = _unit(a_f - (a_f @ f) * f - (a_f @ d_f) * d_f)
        ext = math.degrees(math.atan2(float(along @ d_f), float(along @ f)))
        uln = math.degrees(math.atan2(float(along @ a_f), float(along @ f)))
        over = E + np.linspace(0.0, 1.0, 9)[:, None] * (Wp - E)
        floor = np.zeros(10)
        if self.table is not None:
            top, center, radius = self.table
            lift = np.maximum(0.0, self.forearm_lift - (over @ up - top))
            if center is not None and radius is not None:
                off = over - center
                lift = np.where(np.linalg.norm(off - np.outer(off @ up, up), axis=1) < radius, lift, 0.0)
            floor[:9] = lift
        floor[9] = max(0.0, self.wrist_lift - (Wp @ up - self.paper_z))
        return dict(z=z, clear=_softmin(z, 0.0004), elevation=elev, azimuth=az, tilt=tilt, extension=ext, ulnar=uln,
                    reach=reach, wrist=Wp, elbow=E, floor=floor)

    def residuals(self, wr):
        return np.concatenate([[(wr["clear"] - 0.0008) / 0.0005,
                                max(0.0, wr["reach"] - self.reach_max * (self.l_up + self.l_fore)) / 0.005],
                               np.maximum(0.0, -wr["z"]) / 0.0003, wr["floor"] / 0.004,
                               [(wr[k] - m) / s for k, (m, s) in self.target.items()]])

    def report(self, wr):
        return {"hand_clearance_mm": round(wr["clear"] * 1e3, 2), "paper_hits": int((wr["z"] < -0.0002).sum()),
                **{k + "_deg": round(wr[k], 1) for k in self.target},
                "reach_pct": round(100 * wr["reach"] / (self.l_up + self.l_fore), 1),
                "wrist_above_paper_mm": round((wr["wrist"] @ self.up - self.paper_z) * 1e3, 1),
                "forearm_floor_miss_mm": round(float(wr["floor"].max()) * 1e3, 1)}


def _nib_offset(v):
    """The nib's position in the prop's frame: [x, y, z], a number (z along the pen) or None (the origin)."""
    if v is None:
        return np.zeros(3)
    v = np.asarray(v, float)
    return np.array([0.0, 0.0, float(v)]) if v.ndim == 0 else v.reshape(3)


def _pen_seed(job):
    hand_args, kw, seed, nfev = job
    return _Pen(HandModel(**hand_args), **kw).grip_seed(seed, nfev)


def solve_pen(hand, prop, posture=None, slots=None, seeds=12, workers=None, orient_starts=32,
              nfev=(800, 800, 1500, 1500)):
    """Lateral-tripod pen grip (see the section comment above); the pen is a cylinder with a conical tip.

    prop     {"length": m (nib to cap), "radius": m, or a list of (distance from the nib, radius) pairs,
              "tip": length of the conical tip when radius is a number (default 3 radius),
              "nib_offset": [x, y, z] the nib in the prop's own frame (default 0: the prop's origin is the nib)}
             The prop frame: +Z from the nib toward the cap, +X the side of the barrel facing the back of the hand
             (the clip side), Y = Z x X. `target_in_wrist` is this frame.
    posture  None: the grip in the hand only. Else the writing arm and desk (see _Posture: nib, shoulder, pole,
             table, target, ...): the hand's writing orientation is solved too, and the result gets "frame_world_quat",
             the prop frame's world rotation while writing, and report["writing"].
    slots    {"index" | "thumb" | "middle": (clock deg, from m, to m)} contact slots on the barrel (PEN_SLOTS)
    seeds    parallel starts of the grip solve (the best is kept); workers: processes (default: all cores, 1 = here)
    """
    if "length" not in prop or "radius" not in prop:
        raise ValueError("a pen prop needs 'length' and 'radius'")
    kw = dict(length=float(prop["length"]), radius=prop["radius"], tip=prop.get("tip"), slots=slots,
              posture=posture)
    pen = _Pen(hand, **kw)
    runs = _run_seeds(_pen_seed, [(_hand_args(hand), kw, s, tuple(nfev[:3])) for s in range(seeds)], workers)
    cost, z = min(runs, key=lambda c: c[0])
    x = np.r_[z, np.zeros(3)]
    if pen.posture is not None:
        x[20:] = pen.solve_orientation(x, orient_starts)
        r = least_squares(pen.residuals, x, bounds=(PEN_LO, PEN_HI), x_scale=pen.X_SCALE, max_nfev=nfev[3],
                          xtol=1e-12)
        x, cost = r.x, float(r.cost)
    A = pen.analyse(x)
    R = np.stack([A["top"], np.cross(A["u"], A["top"]), A["u"]], 1)       # x = barrel top, z = nib -> cap
    off = _nib_offset(prop.get("nib_offset"))
    out = _result("pen", hand, A["Q"], R, A["p"] - R @ off, pen.report(x, A),
                  solver={"version": VERSION, "cost": cost, "seeds": seeds, "x": x})
    if pen.posture is not None:
        out["frame_world_quat"] = geom.mat_to_quat((A["G"] @ R)[None])[0]
    return out


# ================================================================ wheel: the power grip on a ring
# A hand holding a ring (a torus: steering wheel, ring handle, hoop): the palm lies on the tube, the four fingers wrap
# round it with all three phalanges on its surface, the thumb comes round the other way: its pad rests on the tube on
# the side opposite to the one the fingers wrap toward ("over the top"), pointing along the tube toward the index side.
# Solved on the model's own skin: every phalanx's palmar surface, the palm and the thumb pad touch the tube (a smooth
# softmin over the gaps of their vertices), no hand vertex (palm and cuff included) is inside the tube, no finger inside
# another, and the joint angles stay near typical power-grip postures inside their bounds.
# GRIP FRAME: the torus has its ring centre at (-R, 0, 0), axis +z and tube radius r. The origin is on the tube's
# centreline at the ring angle of the palm centre (the palm centre has y = 0), x radially outward, y along the ring
# (counter-clockwise seen from +z), z the ring axis. "Section angle": around the centreline in the (x, z) plane, from +x
# toward +z. Unknowns: the 21 joint angles, and the hand's placement relative to a design (palm centre on the approach
# side of the tube, the palm facing the tube centre, the fingers along the wrap direction, the knuckle line along the
# ring): the palm centre's distance from the centreline and a rotation about the hand's own three axes. (Sliding the
# hand round the ring is a symmetry of the torus, removed by keeping the palm centre at y = 0; swinging it round the
# tube is nearly one, so the palm centre stays on the approach side and the rotation, mostly the pitch, slides the
# tube along the palm.)
# Start: the palm on the tube, every finger closed from its base outwards until each phalanx touches it, the thumb
# from a few starts (a short least squares on its five joints); then least squares on everything.
# priors: a cylinder grasp (index and middle MCP/PIP/DIP ~ 50-60/70-90/30-40, ring and little a little more flexed, the
# thumb abducted 30-40 with 10-30 of MCP and IP flexion). They are weak: the contacts decide, the priors pick among the
# ways of touching (a short-fingered hand or a thick tube simply flexes less); the bounds keep out hooks and
# hyperextension.
WHEEL_PRIOR = np.array([52, 78, 32, 0,  55, 82, 34, 0,  60, 85, 38, 0,  65, 88, 40, 0,   35, 25, -10, 20, 20], float)
WHEEL_SIGMA = np.array([14, 16, 12, 5,  14, 16, 12, 5,  14, 16, 12, 5,  14, 16, 12, 5,   15, 25, 35, 15, 15], float)
WHEEL_LO = np.array([0, 5, -5, -12] * 4 + [0, -30, -90, -5, -5], float)
WHEEL_HI = np.array([92, 105, 75, 12] * 4 + [70, 80, 50, 50, 60], float)
# placement: the palm centre's distance from the centreline beyond the design (mm), the hand's rotation (deg) about its
# along / across / ventral axes: roll, pitch (+ tips the fingers toward the back of the hand) and yaw (+ toward the
# little finger); the same numbers for either hand
WHEEL_PLACE = ["dist", "roll", "pitch", "yaw"]
WHEEL_PL_SIGMA = np.array([6.0, 15.0, 15.0, 20.0])
WHEEL_PL_LO = np.array([-8.0, -35.0, -40.0, -40.0])
WHEEL_PL_HI = np.array([25.0, 35.0, 40.0, 40.0])
WHEEL_G0 = 0.0003                       # touching: the smallest gap of a contact surface sits this far outside (m)
WHEEL_THUMB_SECTION = (25.0, 150.0)     # the thumb's contact lies this many degrees behind the palm, against the wrap
WHEEL_THUMB_TANGENT = (15.0, 90.0)      # ... and this many mm from the palm centre along the tube toward the index side
WHEEL_THUMB_ALIGN = 0.7                 # the thumb lies within acos(this) = 45 degrees of the tube's direction
WHEEL_THUMB_STARTS = np.array([[23, 50, -30, 15, 15], [45, 40, -40, 15, 15], [45, 70, 0, 15, 15], [65, 40, -40, 15, 15],
                               [30, 70, 0, 15, 15], [20, 20, 0, 15, 15], [35, 55, -20, 20, 20], [50, 25, -30, 10, 10]],
                              float)   # thumb: palmar, radial, roll, mcp, ip
WHEEL_PITCHES = (0.0, -10.0, -20.0, -5.0, -15.0, -25.0, -8.0, -18.0)   # seeds' start pitch (deg), the fingers' tilt
WHEEL_NFEV = (60, 20)                   # least-squares iterations: reach (loose softmin), polish (tight)


def _wheel_gap(Pg, R, r):
    """Signed distance (m) of grip-frame points (n, 3) to the torus: ring centre (-R, 0, 0), axis z, tube radius r."""
    return np.hypot(np.hypot(Pg[:, 0] + R, Pg[:, 1]) - R, Pg[:, 2]) - r


class _Wheel:
    """The wheel grip problem on one hand (see solve_wheel); x = 21 joint angles (deg, DOF order) and the 4 placement
    parameters (WHEEL_PLACE)."""

    def __init__(self, hand, radius, tube, approach, wrap, prior=WHEEL_PRIOR, thumb_section=WHEEL_THUMB_SECTION,
                 thumb_tangent=WHEEL_THUMB_TANGENT):
        self.hand = hand
        self.R, self.r = float(radius), float(tube)
        self.theta = math.radians(approach)
        self.wrap = float(wrap)
        self.prior = np.r_[prior, np.zeros(4)]
        self.behind = tuple(float(v) for v in thumb_section)
        self.tangent = tuple(float(v) * 1e-3 for v in thumb_tangent)
        hd, F = hand, hand.F
        self.across = hd.chir * np.cross(hd.ventral, hd.along)            # exactly orthogonal to along and ventral
        self.axes = np.stack([hd.along, self.across, hd.ventral], 1)       # the hand's axes (columns) in its frame
        self.fsets = {(f, j): hd.surface(F[f][j], hd.ventral, 0.0, 1.0, 0.3) for f in FOUR for j in range(3)}
        self.pad = hd.surface(F["thumb"][2], hd.pad_t, 0.20, 0.92)
        self.thumb_all = hd.surface(F["thumb"][2], hd.pad_t, 0.0, 1.0, 0.3)
        self.palm = np.where((hd.dom == 0) & ((hd.V - hd.palm_c) @ hd.ventral > 0.006))[0]
        sets = list(self.fsets.values()) + [self.palm, self.pad]
        if min(len(s) for s in sets) < 1:
            raise ValueError("the hand model lacks palmar skin on a phalanx, on the palm or on the thumb pad")
        self.cidx = np.concatenate(sets)                                   # all contact vertices, set after set
        self.clen = np.array([len(s) for s in sets])
        self.coff = np.r_[0, np.cumsum(self.clen)[:-1]]
        self.clash = _Clash(hd, CLASH_PAIRS)
        self.clash_thumb = _Clash(hd, (("thumb", "index"), ("thumb", "middle")))
        self.d0 = self.r + 0.012
        self.scale = np.r_[WHEEL_SIGMA, WHEEL_PL_SIGMA]
        self.lo, self.hi = np.r_[WHEEL_LO, WHEEL_PL_LO], np.r_[WHEEL_HI, WHEEL_PL_HI]

    # ---- the grip frame in the hand frame
    def frame(self, pl):
        """(R, p): the grip frame's axes (columns) and origin in the hand frame for the placement parameters pl: the
        palm centre is on the approach side of the tube, the hand turned by the three angles from the design."""
        dd, ra, rb, rc = pl
        u = np.array([math.cos(self.theta), 0.0, math.sin(self.theta)])   # tube centreline -> palm centre
        t = np.array([-math.sin(self.theta), 0.0, math.cos(self.theta)])  # toward increasing section angle
        vg, ag = -u, self.wrap * t                                         # palm normal, fingers' direction
        mine = np.stack([ag, self.hand.chir * np.cross(vg, ag), vg], 1)    # the hand's axes in the grip frame
        R = self.axes @ (mine @ _rotvec(np.radians([ra, rb, rc]))).T
        return R, self.hand.palm_c - R @ ((self.d0 + dd * 1e-3) * u)

    def analyse(self, x):
        Q, D, H, P = self.hand.pose(x[:21])
        R, p = self.frame(x[21:])
        Pg = (P - p) @ R
        return dict(Q=Q, D=D, H=H, P=P, R=R, p=p, Pg=Pg, gap=_wheel_gap(Pg, self.R, self.r))

    # ---- where things touch
    def softmins(self, gap, tau):
        """Softmin (m) of the gap over every contact set, in order: 12 phalanges (finger by finger), palm, thumb pad."""
        g = gap[self.cidx]
        m = np.minimum.reduceat(g, self.coff)
        return m - tau * np.log(np.add.reduceat(np.exp(-(g - np.repeat(m, self.clen)) / tau), self.coff))

    @staticmethod
    def contact_point(A, idx, tau=0.0008):
        """Where a vertex set touches: its vertices near the smallest gap, weighted."""
        g = A["gap"][idx]
        w = np.exp(-(g - g.min()) / tau)
        return (w / w.sum()) @ A["Pg"][idx]

    def section(self, c):
        """(section angle in deg [0, 360), ring angle in rad) of a grip-frame point."""
        ring = math.hypot(c[0] + self.R, c[1])
        return math.degrees(math.atan2(c[2], ring - self.R)) % 360.0, math.atan2(c[1], c[0] + self.R)

    def thumb_slot(self, A):
        """Where the thumb pad meets the tube: (degrees behind the palm centre against the wrap direction, tangent
        offset toward the index side in m, cosine between the thumb and the tube's direction toward the index side)."""
        hd = self.hand
        sec, ring = self.section(self.contact_point(A, self.pad))
        behind = -((sec - math.degrees(self.theta) + 180.0) % 360.0 - 180.0) * self.wrap
        toward = -hd.chir * self.wrap                                       # +y points toward the index when this is +1
        thumb = A["R"].T @ (A["D"][hd.F["thumb"][2]] @ hd.bone_dir[hd.F["thumb"][2]])
        along = toward * np.array([-math.sin(ring), math.cos(ring), 0.0])
        return behind, toward * ring * self.R, float(thumb @ along)

    def slot_terms(self, A):
        behind, tang, cosv = self.thumb_slot(A)
        return [max(0.0, self.behind[0] - behind) / 10.0, max(0.0, behind - self.behind[1]) / 10.0,
                max(0.0, self.tangent[0] - tang) / 0.008, max(0.0, tang - self.tangent[1]) / 0.008,
                max(0.0, WHEEL_THUMB_ALIGN - cosv) / 0.2]

    # ---- residuals
    def residuals(self, x, tau=0.0004):
        """Contacts (12 phalanges, palm, thumb pad at the touching gap), the thumb's place, joint and placement priors,
        walls (no hand vertex inside the tube) and finger clashes; tau is the softmin's sharpness (m)."""
        A = self.analyse(x)
        return np.concatenate([(self.softmins(A["gap"], tau) - WHEEL_G0) / 0.0005, self.slot_terms(A),
                               0.5 * (x - self.prior) / self.scale, np.maximum(0.0, WHEEL_G0 - A["gap"]) / 0.0001,
                               self.clash.depth(A["P"], A["H"]) / 0.0003])

    def thumb_residuals(self, z, x, tau=0.0004):
        """The thumb alone (joints z = x[16:21]): pad on the tube, its place, priors, walls, clashes with fingers."""
        A = self.analyse(np.r_[x[:16], z, x[21:]])
        return np.concatenate([(self.softmins(A["gap"], tau)[-1:] - WHEEL_G0) / 0.0005, self.slot_terms(A),
                               0.5 * (z - self.prior[16:21]) / WHEEL_SIGMA[16:21],
                               np.maximum(0.0, WHEEL_G0 - A["gap"]) / 0.0001,
                               self.clash_thumb.depth(A["P"], A["H"]) / 0.0003])

    def refine(self, x, nfev, tau):
        return least_squares(lambda z: self.residuals(z, tau), x, bounds=(self.lo, self.hi), x_scale=self.scale,
                             method="dogbox", max_nfev=nfev, xtol=1e-10).x

    # ---- starts
    def palm_touch(self, deg, pl):
        """The palm centre's distance (mm beyond the design) at which the palm just touches, fingers as in deg."""
        lo, hi = -25.0, 70.0
        for _ in range(28):
            mid = 0.5 * (lo + hi)
            if self.analyse(np.r_[deg, mid, pl[1:]])["gap"][self.palm].min() > WHEEL_G0:
                hi = mid
            else:
                lo = mid
        return 0.5 * (lo + hi)

    def close_fingers(self, deg, pl):
        """Every finger from its base outwards: open the joint, then close it until its phalanx's palmar surface
        touches the tube (a joint that never touches goes to its limit)."""
        deg = deg.copy()
        for fi, f in enumerate(FOUR):
            deg[fi * 4:fi * 4 + 4] = [WHEEL_LO[fi * 4], WHEEL_LO[fi * 4 + 1], WHEEL_LO[fi * 4 + 2], 0.0]
            for j in range(3):
                k, idx = fi * 4 + j, self.fsets[(f, j)]

                def gmin(a):
                    deg[k] = a
                    return self.analyse(np.r_[deg, pl])["gap"][idx].min()

                prev, hit = None, None
                for a in np.linspace(WHEEL_LO[k], WHEEL_HI[k], 19):
                    if gmin(a) <= WHEEL_G0:
                        hit = a
                        break
                    prev = a
                if hit is None or prev is None:
                    deg[k] = WHEEL_HI[k] if hit is None else WHEEL_LO[k]
                    continue
                lo, hi = prev, hit
                for _ in range(14):
                    mid = 0.5 * (lo + hi)
                    lo, hi = (mid, hi) if gmin(mid) > WHEEL_G0 else (lo, mid)
                deg[k] = hi
        return deg

    def place_thumb(self, x, starts, nfev=40):
        """The thumb's joints that put its pad on the tube in its place, the best of a few starts."""
        best = None
        for z0 in starts:
            res = least_squares(lambda z: self.thumb_residuals(z, x), np.clip(z0, WHEEL_LO[16:21], WHEEL_HI[16:21]),
                                bounds=(WHEEL_LO[16:21], WHEEL_HI[16:21]), x_scale=WHEEL_SIGMA[16:21], method="dogbox",
                                max_nfev=nfev)
            if best is None or res.cost < best.cost:
                best = res
        return np.r_[x[:16], best.x, x[21:]]

    def grip_seed(self, seed, nfev=WHEEL_NFEV):
        """One start: the design placement (seed 0) or a perturbed one, then close the fingers, place the thumb, refine.
        Returns (cost, x)."""
        rng = np.random.default_rng(seed)
        pl = np.zeros(4)
        if seed:
            pl[1:] = rng.normal(0.0, [3.0, 3.0, 6.0]) + [0.0, WHEEL_PITCHES[seed % len(WHEEL_PITCHES)], 0.0]
        deg = self.prior[:21].copy()
        pl[0] = self.palm_touch(deg, pl)
        pl = np.clip(pl, self.lo[21:], self.hi[21:])
        x = np.r_[self.close_fingers(deg, pl), pl]
        x = self.place_thumb(x, np.roll(WHEEL_THUMB_STARTS, -3 * seed, axis=0)[:4])
        x = self.refine(self.refine(x, nfev[0], 0.0004), nfev[1], 0.00015)
        return 0.5 * float(np.sum(self.residuals(x, 0.00015) ** 2)), x

    # ---- the report
    def report(self, x, A):
        hd, gap = self.hand, A["gap"]
        psi = math.degrees(self.theta) % 360.0
        contacts = {}
        for f in FOUR:
            g = [float(gap[self.fsets[(f, j)]].min()) * 1e3 for j in range(3)]
            tip = self.section(self.contact_point(A, self.fsets[(f, 2)]))[0]
            contacts[f] = {"gap_mm": round(min(g), 2), "prox_mm": round(g[0], 2), "mid_mm": round(g[1], 2),
                           "dist_mm": round(g[2], 2), "wrapped_deg": round(((tip - psi) * self.wrap) % 360.0, 0)}
        behind, tang, cosv = self.thumb_slot(A)
        contacts["thumb"] = {"gap_mm": round(float(gap[self.thumb_all].min()) * 1e3, 2),
                             "section_deg": round(self.section(self.contact_point(A, self.pad))[0], 1),
                             "behind_palm_deg": round(behind, 1), "tangent_mm": round(tang * 1e3, 1),
                             "along_tube": round(cosv, 2)}
        c = A["R"].T @ (hd.palm_c - A["p"])
        centre = math.hypot(math.hypot(c[0] + self.R, c[1]) - self.R, c[2])
        contacts["palm"] = {"gap_mm": round(float(gap[self.palm].min()) * 1e3, 2)}
        return {"contacts": contacts, "palm_centre_mm": round(centre * 1e3, 1),
                "penetration_mm": round(max(0.0, -float(gap.min())) * 1e3, 3),
                "finger_clash_mm": round(float(self.clash.depth(A["P"], A["H"]).max()) * 1e3, 2),
                "angles_deg": _angles(x[:21]),
                "placement": {k: round(float(v), 1) for k, v in zip(WHEEL_PLACE, x[21:])}}


def _wheel_seed(job):
    hand_args, kw, seed, nfev = job
    return _Wheel(HandModel(**hand_args), **kw).grip_seed(seed, nfev)


def _wheel_args(prop, approach, wrap, seeds, tuning):
    """Validated _Wheel keyword arguments and iteration limits for solve_wheel's inputs."""
    if not hasattr(prop, "get"):
        raise ValueError("a wheel prop is a dict with 'radius' (ring radius, m) and 'tube' (tube radius, m)")
    try:
        R, r, appr, wr = float(prop["radius"]), float(prop["tube"]), float(approach), float(wrap)
    except (KeyError, TypeError, ValueError):
        raise ValueError("a wheel prop needs numeric 'radius' (ring radius, m) and 'tube' (tube radius, m); approach "
                         "is degrees, wrap is +1 or -1") from None
    if not (math.isfinite(R) and math.isfinite(r) and 0.0 < r < R):
        raise ValueError(f"a wheel needs 0 < tube < radius (m), got radius={R}, tube={r}")
    if not math.isfinite(appr):
        raise ValueError(f"approach must be a finite angle in degrees, got {approach!r}")
    if wr not in (1.0, -1.0):
        raise ValueError(f"wrap must be +1 or -1, got {wrap!r}")
    try:
        valid = int(seeds) == seeds and seeds >= 1
    except (TypeError, ValueError):
        valid = False
    if not valid:
        raise ValueError(f"seeds must be a positive integer, got {seeds!r}")
    tuning = dict(tuning)
    kw = dict(radius=R, tube=r, approach=appr, wrap=wr, prior=WHEEL_PRIOR.copy())
    given = tuning.pop("prior", None) or {}
    if not hasattr(given, "items"):
        raise ValueError("prior is a {joint: degrees} dict")
    for k, v in given.items():
        if k not in DOF:
            raise ValueError(f"unknown joint {k!r} in prior (have {', '.join(DOF)})")
        kw["prior"][DOF.index(k)] = float(v)
    for key, default in (("thumb_section", WHEEL_THUMB_SECTION), ("thumb_tangent", WHEEL_THUMB_TANGENT)):
        v = tuning.pop(key, default)
        try:
            lo, hi = (float(c) for c in v)
        except (TypeError, ValueError):
            raise ValueError(f"{key} is a (min, max) pair, got {v!r}") from None
        if not lo <= hi:
            raise ValueError(f"{key} is a (min, max) pair, got {v!r}")
        kw[key] = (lo, hi)
    try:
        nfev = tuple(int(n) for n in tuning.pop("nfev", WHEEL_NFEV))
    except (TypeError, ValueError):
        nfev = ()
    if len(nfev) != 2 or min(nfev) < 1:
        raise ValueError("nfev is a pair of positive iteration limits (reach, polish)")
    if tuning:
        raise ValueError(f"unknown wheel parameters: {', '.join(sorted(tuning))}")
    return kw, nfev


def solve_wheel(hand, prop, approach=90.0, wrap=1, seeds=8, workers=None, **tuning):
    """Power grip on a ring (see the section comment above): palm on the tube, fingers wrapped round it, thumb opposing.

    prop      {"radius": m (the ring's), "tube": m (the tube's)}, other keys ignored. Meant for tubes of 8-25 mm and
              rings of 60-500 mm; a tube too big to close the fingers round gets the wrap the hand manages (the gaps in
              the report say how far the last phalanges are).
    The prop frame (`target_in_wrist` is this frame): the ring centre is at (-R, 0, 0), the ring axis +z. The origin is
    on the tube's centreline at the ring angle of the palm centre (the palm centre has y = 0), x points radially
    outward, y along the ring (counter-clockwise seen from +z), z = x cross y. A build puts that frame at any chosen
    angle of its ring.
    approach  degrees, where the palm is seen from the tube's centreline in the section plane (x, z): 0 = +x (outer
              side), 90 = +z (default), 180 = -x (inner side), 270 = -z. The palm centre lies exactly on that side; the
              hand is turned from the design (palm normal at the tube centre) by the solved roll / pitch / yaw.
    wrap      +1 / -1: the fingers go round the tube through increasing / decreasing section angle (measured from +x
              toward +z); the thumb comes from the other side.
    seeds     parallel starts, the lowest cost wins; workers: processes (default: all cores, 1 = this process)
    tuning    prior {joint: deg} (joints as in DOF), thumb_section (min, max) degrees the thumb contact lies behind the
              palm against the wrap direction, thumb_tangent (min, max) mm it lies from the palm centre along the tube
              toward the index side, nfev (reach, polish) least-squares iteration limits
    The report has contacts {finger: gap_mm (best of the three), prox_mm / mid_mm / dist_mm (palmar surface of each
    phalanx), wrapped_deg (round the tube from the palm centre to the distal contact)}, thumb {gap_mm, section_deg
    (contact angle round the tube), behind_palm_deg, tangent_mm (along the tube from the palm centre toward the index),
    along_tube (cosine between the thumb and the tube there)}, palm {gap_mm}; palm_centre_mm (palm centre to the tube's
    centreline), penetration_mm, finger_clash_mm, angles_deg, placement {dist (mm beyond the design), roll, pitch, yaw}.
    """
    kw, nfev = _wheel_args(prop, approach, wrap, seeds, tuning)
    wheel = _Wheel(hand, **kw)
    runs = _run_seeds(_wheel_seed, [(_hand_args(hand), kw, s, nfev) for s in range(int(seeds))], workers)
    cost, x = min(runs, key=lambda c: c[0])
    A = wheel.analyse(x)
    return _result("wheel", hand, A["Q"], A["R"], A["p"], wheel.report(x, A),
                   solver={"version": VERSION, "cost": cost, "seeds": int(seeds)})


# ================================================================ pinch: thumb-index pad pinch
# A thin object (a card, a strap, a coin edge, a key) held at its edge between the index pad and the thumb pad: the two
# pads touch opposite faces and face each other, the contacts overlap in the object's mid-plane, the middle, ring and
# little fingers curl (more and more) and stay clear of the object and of the index and thumb. The held part is a slab
# |z| <= width / 2, -edge <= x <= depth - edge, |y| <= length / 2 in the object's frame (origin midway between the two
# pad contacts, z from the index pad toward the thumb pad, x from the hand toward the held part, y = z cross x); the
# pads sit `edge` inside its near edge. The skin of each pad (softmin over its vertices) meets its face; no skin of the
# hand is inside the slab. The hand's pose relative to the frame is free (6 parameters) around a design start, with a
# soft prior that the wrist -> pinch point heading stays near +x; the roll of the hand about it is whatever lets the
# pads meet face to face (pad normal within PINCH_FACE of the squeeze axis). Joint angles stay near a relaxed pulp
# pinch.
PINCH_NAMES = ["i_mcp", "i_pip", "i_dip", "i_spr", "m_curl", "r_curl", "l_curl", "m_spr",
               "t_palmar", "t_radial", "t_roll", "t_mcp", "t_ip", "px", "py", "pz", "ua", "ub", "uc"]
# priors: index slightly flexed, thumb opposed (palmar and radial abduction, rolled so the pad turns toward the index,
# light MCP / IP flexion), middle / ring / little curled more and more (curl 1 = PINCH_CURL); bounds keep out
# hyperextension and fists
PINCH_PRIOR = np.array([32, 40, 20, 0, 1.0, 1.0, 1.0, 0, 45, 20, -60, 20, 20], float)
PINCH_SIGMA = np.array([12, 15, 10, 5, 0.3, 0.3, 0.3, 5, 12, 12, 30, 10, 10], float)
PINCH_LO = np.array([0, 0, 0, -12, 0.3, 0.3, 0.3, -12, 0, -20, -110, 0, 0, -80, -80, -80, -1.5, -1.5, -1.5], float)
PINCH_HI = np.array([70, 90, 45, 12, 1.6, 1.6, 1.6, 12, 80, 45, 50, 45, 60, 80, 80, 80, 1.5, 1.5, 1.5], float)
PINCH_CURL = {"middle": (45.0, 62.0, 32.0), "ring": (55.0, 72.0, 38.0), "little": (65.0, 82.0, 42.0)}   # mcp pip dip
PINCH_PRIOR_W = 0.75                                  # joint priors enter as PINCH_PRIOR_W * (x - prior) / sigma
PINCH_WIDTH = (0.0005, 0.040)                         # supported object thickness (m)
PINCH_TILT = 25.0                                     # the index pad faces this far toward the thumb side (deg)
PINCH_FACE = math.cos(math.radians(25.0))             # pad normal within this angle of the squeeze axis
PINCH_FACE_SCALE = 0.08                               # ... and this much cosine past it is one residual unit
PINCH_CLEAR = 0.010                                   # middle / ring / little skin stays this far from the object (m)
PINCH_STEP = 15.0                                     # each outer finger is curled this much (deg) more than the last
PINCH_NFEV = (400, 600, 300)                          # evaluations of the reach / grip / polish stages


def _pinch_prop(prop, edge):
    """(width, depth, length) in metres from the prop card, or ValueError."""
    if not hasattr(prop, "get"):
        raise ValueError("a pinch prop is a dict like {'width': m, 'depth': m, 'length': m}")

    def number(key, default=None):
        v = prop.get(key)
        v = default if v is None else v
        if v is None:
            raise ValueError(f"a pinch prop needs {key!r} (the object's thickness between the pads, m)")
        try:
            v = float(v)
        except (TypeError, ValueError):
            raise ValueError(f"pinch prop {key!r} must be a number, got {v!r}") from None
        if not math.isfinite(v):
            raise ValueError(f"pinch prop {key!r} must be finite, got {v!r}")
        return v

    w, depth, length = number("width"), number("depth", 0.15), number("length", 0.16)
    if not PINCH_WIDTH[0] <= w <= PINCH_WIDTH[1]:
        raise ValueError(f"pinch width {w * 1e3:.2f} mm is outside the supported {PINCH_WIDTH[0] * 1e3:g}-"
                         f"{PINCH_WIDTH[1] * 1e3:g} mm")
    if not (depth > edge and length > 0):
        raise ValueError(f"pinch depth must exceed the edge inset ({edge * 1e3:g} mm) and length must be positive")
    return w, depth, length


class _Pinch:
    """The pinch problem on one hand (see solve_pinch); x = 13 finger parameters (PINCH_NAMES), then the frame's
    translation (mm) and rotation vector (rad) relative to the design start."""

    X_SCALE = np.r_[PINCH_SIGMA, np.full(3, 1.0), np.full(3, 0.05)]

    def __init__(self, hand, width, depth=0.15, length=0.16, edge=0.004, heading=25.0, clear=PINCH_CLEAR):
        self.hand, self.w, self.depth = hand, float(width), float(depth)
        self.length, self.edge = float(length), float(edge)
        self.heading_tol, self.clear = float(heading), float(clear)
        hd, F = hand, hand.F
        self.pad_i = hd.surface(F["index"][2], hd.ventral, 0.0, 1.0, 0.3)       # the pads: palmar skin of the distal
        self.pad_t = hd.surface(F["thumb"][2], hd.pad_t, 0.0, 1.0, 0.3)         # phalanges, as check_posed measures
        if min(len(self.pad_i), len(self.pad_t)) < 3:
            raise ValueError("the hand model has too little skin on the index or thumb pad for a pinch")
        t = math.radians(PINCH_TILT)
        self.dir_i = _unit(math.cos(t) * hd.ventral + math.sin(t) * hd.radial)
        self.dir_t = hd.pad_t
        self.clash = _Clash(hd)
        self.centre = np.array([self.depth / 2 - self.edge, 0.0, 0.0])
        self.half = np.array([self.depth / 2, self.length / 2, self.w / 2])
        self.others = np.where(np.isin(hd.dom, [b for f in ("middle", "ring", "little") for b in F[f][:3]]))[0]
        self.R0 = self.p0 = None

    # ---- parameters
    def degrees(self, x):
        """The 21 joint angles (deg, DOF order) for the finger parameters x[:13]: the outer fingers follow PINCH_CURL
        scaled by their curl; the middle finger's spread is free, ring and little fan out a little (-3 / -6 deg)."""
        deg = np.zeros(21)
        deg[0:4] = x[0:4]
        for k, f in enumerate(("middle", "ring", "little")):
            c = PINCH_CURL[f]
            deg[4 + 4 * k:8 + 4 * k] = [c[0] * x[4 + k], c[1] * x[4 + k], c[2] * x[4 + k], x[7] if k == 0 else -3.0 * k]
        deg[16:21] = x[8:13]
        return deg

    def start_frame(self, deg):
        """Design start for a finger pose: origin between the pad centroids, z from the index pad toward the thumb pad,
        x the wrist -> pinch point heading made perpendicular to z."""
        hd = self.hand
        P = hd.pose(deg)[3]
        ci, ct = P[self.pad_i].mean(0), P[self.pad_t].mean(0)
        z = _unit(ct - ci)
        p = 0.5 * (ci + ct)
        x = _perp(_unit(p - hd.h[0]), z)
        if np.linalg.norm(x) < 1e-3:                                # heading along the squeeze axis: any x will do
            x = _perp(hd.ventral, z)
        x = _unit(x)
        return np.stack([x, np.cross(z, x), z], 1), p

    def frame(self, x):
        return _rotvec(x[16:19]) @ self.R0, self.p0 + x[13:16] * 1e-3

    # ---- geometry against the slab
    def box_sdf(self, L):
        """Signed distance (m, + outside) of points in the frame to the slab."""
        q = np.abs(L - self.centre) - self.half
        return np.linalg.norm(np.maximum(q, 0.0), axis=1) + np.minimum(q.max(axis=1), 0.0)

    @staticmethod
    def contact(gap, L, idx, tau):
        """A pad against its face: softmin gap, and the contact point (softmin-weighted mean of the pad's skin)."""
        g = gap[idx]
        w = np.exp(-(g - g.min()) / 0.001)
        return dict(gap=_softmin(g, tau), xyz=(w / w.sum()) @ L[idx])

    def analyse(self, x, stage="all"):
        hd = self.hand
        deg = self.degrees(x)
        Q, D, H, P = hd.pose(deg)
        R, p = self.frame(x)
        L = (P - p) @ R
        tau = 0.0001 if stage == "polish" else 0.0004
        ci = self.contact(-self.w / 2 - L[:, 2], L, self.pad_i, tau)             # distance outside the z = -w/2 face
        ct = self.contact(L[:, 2] - self.w / 2, L, self.pad_t, tau)
        A = dict(Q=Q, D=D, H=H, P=P, R=R, p=p, L=L, deg=deg, ci=ci, ct=ct)
        wr = R.T @ (p - hd.h[0])
        A["heading"] = math.degrees(math.acos(float(np.clip(wr[0] / np.linalg.norm(wr), -1.0, 1.0))))
        A["face_i"] = float((D[hd.F["index"][2]] @ self.dir_i) @ R[:, 2])
        A["face_t"] = -float((D[hd.F["thumb"][2]] @ self.dir_t) @ R[:, 2])
        if stage != "reach":
            A["sdf"] = self.box_sdf(L)
            A["clash"] = self.clash.depth(P, H)
            if stage == "polish":                                                # gap as measured: whole pad surface
                ci["gap"], ct["gap"] = _softmin(A["sdf"][self.pad_i], tau), _softmin(A["sdf"][self.pad_t], tau)
        return A

    def residuals(self, x, stage="all"):
        """stage 'reach': pads on their faces, facing each other, joint priors (skin may pass through the slab);
        'grip': + nothing of the hand inside the slab, no finger inside another, the outer fingers clear;
        'polish': the contact gaps on the measured pad surfaces (sharper softmin)."""
        A = self.analyse(x, stage)
        ci, ct = A["ci"], A["ct"]
        xy_i, xy_t = ci["xyz"][:2], ct["xyz"][:2]
        tot = A["deg"][[0, 4, 8, 12]] + A["deg"][[1, 5, 9, 13]] + A["deg"][[2, 6, 10, 14]]       # total flexion
        r = [(ci["gap"] - 0.0003) / 0.0006, (ct["gap"] - 0.0003) / 0.0006,
             *((xy_i + xy_t) / 2 / 0.0004), *((xy_i - xy_t) / 0.001),            # contacts centred, overlapping
             max(0.0, PINCH_FACE - A["face_i"]) / PINCH_FACE_SCALE,
             max(0.0, PINCH_FACE - A["face_t"]) / PINCH_FACE_SCALE,
             max(0.0, A["heading"] - self.heading_tol) / 10.0,
             *(np.maximum(0.0, tot[:3] + PINCH_STEP - tot[1:]) / 15.0)]
        r = np.concatenate([r, PINCH_PRIOR_W * (x[:13] - PINCH_PRIOR) / PINCH_SIGMA])
        if stage == "reach":
            return r
        return np.concatenate([r, np.maximum(0.0, 0.0002 - A["sdf"]) / 0.0002, A["clash"] / 0.0004,
                               np.maximum(0.0, self.clear - A["sdf"][self.others]) / 0.003])

    # ---- solving
    def seed(self, k, nfev=PINCH_NFEV):
        """Reach, grip, polish from the prior pose (k = 0) or a perturbed pose and frame (k > 0)."""
        rng = np.random.default_rng(k)
        f0 = np.clip(PINCH_PRIOR + (rng.normal(0, 0.5, 13) * PINCH_SIGMA if k else 0.0), PINCH_LO[:13] + 1e-6,
                     PINCH_HI[:13] - 1e-6)
        self.R0, self.p0 = self.start_frame(self.degrees(f0))
        if k:
            self.R0 = _rotvec(rng.normal(0, 0.3, 3)) @ self.R0
            self.p0 = self.p0 + rng.normal(0, 0.004, 3)
        z = np.r_[f0, np.zeros(6)]
        for stage, n, ftol in zip(("reach", "grip", "polish"), nfev, (1e-6, 1e-6, 1e-5)):
            r = least_squares(lambda z_: self.residuals(z_, stage), z, bounds=(PINCH_LO, PINCH_HI),
                              x_scale=self.X_SCALE, max_nfev=n, xtol=1e-12, ftol=ftol)
            z = r.x
        return float(r.cost), z, self.R0, self.p0

    def report(self, A):
        """Numbers measured on the final pose (hard minima over the skin, like check_posed does)."""
        sdf = A["sdf"]
        con = {}
        for name, pad, c, face in (("index", self.pad_i, A["ci"], A["face_i"]),
                                   ("thumb", self.pad_t, A["ct"], A["face_t"])):
            con[name] = {"gap_mm": round(float(sdf[pad].min()) * 1e3, 2), "x_mm": round(float(c["xyz"][0]) * 1e3, 2),
                         "y_mm": round(float(c["xyz"][1]) * 1e3, 2),
                         "facing_deg": round(math.degrees(math.acos(min(1.0, face))), 1)}
        return {"contacts": con, "width_mm": round(self.w * 1e3, 2),
                "pad_offset_mm": round(float(np.linalg.norm((A["ci"]["xyz"] - A["ct"]["xyz"])[:2])) * 1e3, 2),
                "squeeze_mm": round(con["index"]["gap_mm"] + con["thumb"]["gap_mm"], 2),
                "heading_deg": round(A["heading"], 1),
                "penetration_mm": round(max(0.0, -float(sdf.min())) * 1e3, 3),
                "finger_clash_mm": round(float(A["clash"].max()) * 1e3, 2),
                "outer_fingers_clear_mm": round(float(sdf[self.others].min()) * 1e3, 1),
                "angles_deg": _angles(A["deg"])}


def _pinch_seed(job):
    hand_args, kw, k, nfev = job
    return _Pinch(HandModel(**hand_args), **kw).seed(k, nfev)


def solve_pinch(hand, prop, edge=0.004, seeds=8, workers=None, **tuning):
    """Thumb-index pad pinch of a thin object (see the section comment above).

    prop    {"width": m, the object's thickness between the two pads (required, 0.5-40 mm: ValueError outside),
             "depth": m, the held extent along the frame's +x (default 0.15),
             "length": m, the held extent along the frame's y, centred (default 0.16)}; other keys are ignored.
            The frame (`target_in_wrist` is this frame): origin midway between the two pad contacts on the object's
            mid-plane, z the squeeze axis from the INDEX pad toward the THUMB pad, x from the hand toward the held part
            (away from the wrist), y = z cross x. The held part is the slab |z| <= width/2, -edge <= x <= depth - edge,
            |y| <= length/2: the pads sit `edge` inside its edge.
    edge    how far inside the object's edge the pads touch it (m, default 4 mm)
    seeds   parallel starts (the lowest cost wins); workers: processes (default: all cores, 1 = here)
    tuning  heading (deg, default 25: tolerance of the wrist -> pinch point direction around +x), clear (m, default 10
            mm: how far the middle, ring and little fingers stay from the object), nfev (evaluations of the reach /
            grip / polish stages)
    report  contacts {index, thumb: gap_mm (smallest distance from the phalanx's palmar skin to the slab, + outside),
            x_mm, y_mm (contact point in the frame: mean of the pad skin weighted by exp(-gap / 1 mm)), facing_deg (pad
            normal to squeeze axis)}, width_mm, pad_offset_mm (distance between the two contact points in the
            mid-plane), squeeze_mm (pads' distance along z minus width = the two gaps added), heading_deg (wrist ->
            frame origin against +x), penetration_mm (deepest skin vertex inside the slab), finger_clash_mm,
            outer_fingers_clear_mm (middle / ring / little skin to the slab), angles_deg."""
    unknown = set(tuning) - {"heading", "clear", "nfev"}
    if unknown:
        raise ValueError(f"unknown pinch tuning {sorted(unknown)} (have heading, clear, nfev)")
    edge = float(edge)
    if not (math.isfinite(edge) and edge >= 0.0):
        raise ValueError(f"pinch edge must be a non-negative number of metres, got {edge!r}")
    if int(seeds) < 1:
        raise ValueError("pinch needs at least one seed")
    w, depth, length = _pinch_prop(prop, edge)
    nfev = tuple(int(n) for n in tuning.get("nfev", PINCH_NFEV))
    kw = dict(width=w, depth=depth, length=length, edge=edge, heading=tuning.get("heading", 25.0),
              clear=tuning.get("clear", PINCH_CLEAR))
    pin = _Pinch(hand, **kw)
    runs = _run_seeds(_pinch_seed, [(_hand_args(hand), kw, s, nfev) for s in range(int(seeds))], workers)
    cost, z, pin.R0, pin.p0 = min(runs, key=lambda c: c[0])
    A = pin.analyse(z, "polish")
    return _result("pinch", hand, A["Q"], A["R"], A["p"], pin.report(A),
                   solver={"version": VERSION, "cost": cost, "seeds": int(seeds), "x": z})


# ================================================================ rest: a relaxed hand on a plane
# A hand left to itself on a table, a lap or a page: palm down (face "palm") or on its back (face "back"). Palm down,
# the heel of the palm and the finger pads touch the plane and the palm arches between them; the wrist is a little
# extended, the fingers curl gently (index MCP/PIP/DIP ~ 12/18/6, a little more toward the little finger, a slight
# fan), the thumb lies beside the index with its pad on the plane. On its back the dorsal palm, the backs of the
# proximal and middle phalanges and the thumb rest on the plane, the fingertips curl up and away. Each contact region
# is a smooth minimum of the heights of its skin vertices (a softmin) pulled to REST_GAP above the plane, every vertex
# of the hand's own skin stays above the plane, no finger is inside another, and the joint angles stay near relaxed
# priors (weak: the contacts win). The placement is three numbers: the palm centre's height, the pitch (wrist
# extension) and the roll; the hand's yaw and its place on the plane are free, so the grip frame is built from the
# solved hand. Regions: palm_heel = the wrist-weighted skin of the contact side over the proximal 45% of the palm
# (the whole dorsal palm on the back), index..little = the pad half of the distal phalanx (the back of the proximal
# and middle phalanges on the back), thumb = the pad of its distal phalanx (any skin of its two phalanges on the back).
# The static skin the model keeps near the wrist (a sleeve cuff) follows the forearm, not the hand: it is neither held
# off the plane nor counted in `penetration_mm` (see `cuff_penetration_mm`).
# Priors and bounds for x = the 21 joint angles in DOF order (deg), then the palm height (mm, free), pitch, roll (deg),
# and the prior widths. Palm down: the relaxed curl, the thumb beside the index, the wrist extended ~8. On its back the
# fingers lie nearly straight (MCP slightly extended: the arch of the knuckles holds them above the plane otherwise),
# PIP relaxed, the fingertips a little curled up, the thumb swung out and rolled.
REST_PRIOR = {
    "palm": np.array([12, 18, 6, 3, 14, 21, 8, 0, 17, 24, 10, -3, 20, 28, 12, -8, 8, 4, 0, 10, 8, 20, 8, 0], float),
    "back": np.array([-5, 15, 12, 3, -5, 15, 12, 0, -5, 15, 12, -3, -5, 15, 12, -8, -10, 4, 10, 0, 0, 20, 6, 0], float),
}
REST_SIGMA = {
    "palm": np.array([12, 14, 10, 6] * 4 + [15, 12, 20, 14, 14, 1e3, 8, 10], float),
    "back": np.array([10, 12, 10, 6] * 4 + [15, 12, 20, 14, 14, 1e3, 8, 10], float),
}
REST_LO = {
    "palm": np.array([-15, 0, 0, -15] * 4 + [-30, -25, -40, -10, -15, 0, -35, -45], float),
    "back": np.array([-30, 0, 0, -20] * 4 + [-45, -25, -40, -15, -15, 0, -35, -45], float),
}
REST_HI = {
    "palm": np.array([65, 85, 60, 15] * 4 + [50, 30, 50, 50, 60, 150, 35, 45], float),
    "back": np.array([65, 85, 60, 20] * 4 + [50, 30, 50, 50, 60, 150, 35, 45], float),
}
REST_GAP = 0.3                    # mm: the gap each region's softmin is pulled to (hard minima end 0.3-0.7)
# smoothing schedule (softmin width m, hinge margin m, ftol, max_nfev): the early stages see a blurred landscape, so
# every start reaches the same basin, the last one is sharp
REST_STAGES = ((0.002, 0.001, 1e-3, 100), (0.0006, 0.0004, 1e-4, 100), (0.00015, 0.00005, 1e-7, 150))
REST_STARTS = ((0.0, 0.0), (0.0, -9.0), (0.0, 9.0), (5.0, -4.0), (5.0, 4.0), (-5.0, 0.0))    # pitch, roll offsets (deg)
REST_TUNING = ("angles", "pitch", "roll", "gap_mm", "nfev")


class _Rest:
    """The rest problem on one hand (see solve_rest). x = the 21 joint angles (deg, DOF order), the palm centre's height
    above the plane (mm), pitch and roll (deg). The plane's normal n (out of the surface, toward the hand) in the
    hand's frame: palm down it starts as the dorsal axis, on the back as the ventral axis; pitch tilts it about the
    across axis (+ = wrist extension: the fingers rise from the plane palm down, dip into it on the back), roll about
    the along axis (+ raises the radial side)."""

    def __init__(self, hand, face, prior, sigma, gap, nfev):
        self.hand, self.face, self.prior, self.sigma, self.gap, self.nfev = hand, face, prior, sigma, gap, nfev
        self.lo, self.hi = REST_LO[face], REST_HI[face]
        hd, F = hand, hand.F
        down = hd.ventral if face == "palm" else hd.dorsal                 # the side of the hand that lies on the plane
        self.body = np.where(hd.dom < hd.nb)[0]                            # the hand's own skin
        self.cuff = np.where(hd.dom == hd.nb)[0]                           # static skin (a sleeve cuff)
        s = (hd.V - hd.h[0]) @ hd.along
        side = (hd.V - hd.palm_c) @ down
        reach = (0.45 if face == "palm" else 0.9) * hd.palm_length
        self.regions = {"palm_heel": np.where((hd.dom == 0) & (s < reach) & (side > 0))[0]}
        for f in FOUR:
            self.regions[f] = (hd.surface(F[f][2], hd.ventral, 0.0, 1.0, 0.3) if face == "palm" else
                               np.concatenate([hd.surface(F[f][j], hd.dorsal, 0.0, 1.0, 0.3) for j in (0, 1)]))
        self.regions["thumb"] = (hd.surface(F["thumb"][2], hd.pad_t, 0.0, 1.0, 0.3) if face == "palm" else
                                 np.where(np.isin(hd.dom, F["thumb"][1:3]))[0])
        for k, idx in self.regions.items():
            if not len(idx):
                raise ValueError(f"no skin found for the {k} contact of a {face} rest: is the hand model skinned "
                                 f"the usual way?")
        self.clash = _Clash(hd)

    def normal(self, pitch, roll):
        """Unit plane normal in the hand's frame for the pitch and roll (deg)."""
        hd = self.hand
        a, b = math.radians(pitch), math.radians(roll)
        if self.face == "palm":
            d = hd.dorsal * math.cos(a) + hd.along * math.sin(a)
        else:
            d = hd.ventral * math.cos(a) - hd.along * math.sin(a)
        return _unit(d * math.cos(b) + hd.radial * math.sin(b))

    def heights(self, P, x):
        """Height of every skin vertex above the plane (m), and the plane's normal."""
        n = self.normal(x[22], x[23])
        return (P - self.hand.palm_c) @ n + x[21] * 1e-3, n

    def residuals(self, x, tau, margin):
        """Contact gaps, vertices under the margin, finger clashes, priors; tau and margin in m."""
        Q, D, H, P = self.hand.pose(x[:21])
        z, n = self.heights(P, x)
        g = np.array([_softmin(z[idx], tau) for idx in self.regions.values()])
        return np.concatenate([(g - self.gap * 1e-3) / 0.0003, np.maximum(0.0, margin - z[self.body]) / 0.0001,
                               self.clash.depth(P, H) / 0.0004, 0.5 * (x - self.prior) / self.sigma])

    def start(self, seed):
        """Prior angles, the plane tilted by the seed's offsets, the hand lowered until its lowest vertex touches."""
        x = self.prior.copy()
        dp, dr = REST_STARTS[seed % len(REST_STARTS)]
        x[22:] += (dp, dr)
        if seed >= len(REST_STARTS):
            x[:21] += np.random.default_rng(seed).normal(0, 0.35, 21) * self.sigma[:21]
        x = np.clip(x, self.lo + 1e-6, self.hi - 1e-6)
        P = self.hand.pose(x[:21])[3]
        x[21] = np.clip(-((P[self.body] - self.hand.palm_c) @ self.normal(x[22], x[23])).min() * 1e3,
                        self.lo[21] + 1e-6, self.hi[21] - 1e-6)
        return x

    def solve_seed(self, seed):
        x = self.start(seed)
        for (tau, margin, ftol, _), n in zip(REST_STAGES, self.nfev):
            r = least_squares(lambda x_: self.residuals(x_, tau, margin), x, bounds=(self.lo, self.hi),
                              x_scale="jac", max_nfev=n, xtol=1e-9, ftol=ftol)
            x = r.x
        return float(r.cost), x

    def analyse(self, x):
        Q, D, H, P = self.hand.pose(x[:21])
        z, n = self.heights(P, x)
        return dict(Q=Q, H=H, P=P, z=z, n=n, gaps={k: float(z[idx].min()) for k, idx in self.regions.items()})


def _rest_tuning(face, tuning):
    """The validated solve settings (plain numbers, picklable) for the keyword tuning of solve_rest."""
    bad = sorted(set(tuning) - set(REST_TUNING))
    if bad:
        raise ValueError(f"unknown rest tuning {', '.join(bad)} (have {', '.join(REST_TUNING)})")
    prior, sigma = REST_PRIOR[face].copy(), REST_SIGMA[face].copy()
    angles = tuning.get("angles") or {}
    if not isinstance(angles, dict):
        raise ValueError(f"angles must be a {{joint name: degrees}} mapping, got {angles!r}")
    for k, v in angles.items():
        if k not in DOF:
            raise ValueError(f"angles: unknown joint {k!r} (have {', '.join(DOF)})")
        prior[DOF.index(k)] = float(v)
    for i, k in ((22, "pitch"), (23, "roll")):
        if tuning.get(k) is not None:
            try:
                prior[i], sigma[i] = (float(t) for t in tuning[k])
            except (TypeError, ValueError):
                raise ValueError(f"{k} must be (degrees, tolerance), got {tuning[k]!r}") from None
            if not sigma[i] > 0:
                raise ValueError(f"{k}: the tolerance must be positive")
    gap = float(REST_GAP if tuning.get("gap_mm") is None else tuning["gap_mm"])
    nfev = tuple(int(n) for n in (tuning.get("nfev") or [s[3] for s in REST_STAGES]))
    if gap < 0 or len(nfev) != len(REST_STAGES) or min(nfev) < 1:
        raise ValueError(f"gap_mm must be >= 0 and nfev {len(REST_STAGES)} positive iteration caps")
    return dict(prior=prior, sigma=sigma, gap=gap, nfev=nfev)


def _rest_seed(job):
    hand_args, face, seed, kw = job
    return _Rest(HandModel(**hand_args), face, **kw).solve_seed(seed)


def solve_rest(hand, prop, face="palm", seeds=6, workers=None, **tuning):
    """A relaxed hand lying on a plane (see the section comment above).

    prop     {"surface": "plane"} (other keys are ignored)
    face     "palm": palm down, palm heel and finger pads on the plane; "back": the back of the hand down, the fingers
             relaxed toward the palm side
    seeds    starts of the solve (pitch / roll offsets, then random finger perturbations); the best is kept
    workers  processes for the seeds (default: all cores, 1 = here)
    tuning   angles {joint name of DOF: deg}: replaces a joint's relaxed prior; pitch, roll (deg, tolerance): the
             prior on the wrist extension and the roll; gap_mm: the gap the contacts are pulled to (default 0.3);
             nfev: the iteration caps of the three smoothing stages

    The grip frame (target_in_wrist): origin on the plane directly below the palm centre (hand.palm_c), z the plane's
    normal pointing out of the surface toward the hand, x the hand's heading (hand.along, wrist -> fingers) projected
    onto the plane, y = z cross x. The hand's yaw and its place on the plane are free: lay the frame wherever the hand
    should be.

    report: contacts {palm_heel, index, middle, ring, little, thumb: {gap_mm}} (the smallest height of the region's
    skin above the plane), face, palm_height_mm and wrist_height_mm (the palm centre and the wrist head above the
    plane), pitch_deg (wrist extension: + lifts the fingers from the plane palm down, dips them into it on the back),
    roll_deg (+ lifts the thumb side), penetration_mm (the deepest vertex of the hand's own skin under the plane),
    cuff_penetration_mm (the same for the static skin), finger_clash_mm, angles_deg.
    """
    surface = prop.get("surface") if isinstance(prop, dict) else None
    if surface != "plane":
        raise ValueError(f"rest needs prop['surface'] == 'plane', got {surface!r}")
    if face not in ("palm", "back"):
        raise ValueError(f"face must be 'palm' or 'back', got {face!r}")
    if int(seeds) < 1:
        raise ValueError(f"seeds must be at least 1, got {seeds!r}")
    kw = _rest_tuning(face, tuning)
    rest = _Rest(hand, face, **kw)
    runs = _run_seeds(_rest_seed, [(_hand_args(hand), face, s, kw) for s in range(int(seeds))], workers)
    cost, x = min(runs, key=lambda c: c[0])
    A = rest.analyse(x)
    n, z = A["n"], A["z"]
    ex = _unit(_perp(hand.along, n))
    R = np.stack([ex, np.cross(n, ex), n], 1)
    report = {"contacts": {k: {"gap_mm": round(g * 1e3, 2)} for k, g in A["gaps"].items()}, "face": face,
              "palm_height_mm": round(float(x[21]), 1),
              "wrist_height_mm": round(float((hand.h[0] - hand.palm_c) @ n * 1e3 + x[21]), 1),
              "pitch_deg": round(float(x[22]), 1), "roll_deg": round(float(x[23]), 1),
              "penetration_mm": round(float(max(0.0, -z[rest.body].min())) * 1e3, 3),
              "cuff_penetration_mm": round(float(max(0.0, -z[rest.cuff].min())) * 1e3, 1) if len(rest.cuff) else 0.0,
              "finger_clash_mm": round(float(rest.clash.depth(A["P"], A["H"]).max()) * 1e3, 2),
              "angles_deg": _angles(x[:21])}
    return _result("rest", hand, A["Q"], R, hand.palm_c - x[21] * 1e-3 * n, report,
                   solver={"version": VERSION, "cost": cost, "seeds": int(seeds), "x": x})


STYLES = ("pen", "wheel", "pinch", "rest")


def solve(style, hand, prop, **params):
    """Dispatch to solve_<style> ("ring" is the wheel style)."""
    name = "wheel" if style == "ring" else style
    fn = globals().get(f"solve_{name}") if name in STYLES else None
    if fn is None:
        raise ValueError(f"unknown grip style {style!r} (have pen, wheel, pinch, rest)")
    return fn(hand, prop, **params)


# ---------------------------------------------------------------- command line (the build's solver interface)
def main(argv=None):
    """python -m mkmmd.solvers.grip IN.npz OUT.npz. Returns the exit code: 0 ok, 2 bad input.

    IN: the arrays of the Blender op `hand_model` (side, names, parents, heads, tails, rest_rot, V, W, ...) and `spec`,
    JSON text {"style": pen | wheel | pinch | rest, "prop": {...}, "params": {...}} (solve_<style>'s prop and keyword
    arguments; `workers` defaults to all cores).
    OUT: `bones` (n,) Blender bone names, `quats` (n, 4) w x y z pose-bone rotation_quaternion values, `target_in_wrist`
    (4, 4), `report` JSON text {"style", "side", "report", "solver", ...} (a pen with a writing posture adds
    "frame_world_quat")."""
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) != 2:
        print("usage: python -m mkmmd.solvers.grip IN.npz OUT.npz", file=sys.stderr)
        return 2
    t0 = time.perf_counter()
    try:
        with np.load(argv[0], allow_pickle=False) as z:
            a = {k: z[k] for k in z.files}
        spec = json.loads(str(a["spec"]))
        style, prop, params = spec["style"], dict(spec.get("prop") or {}), dict(spec.get("params") or {})
        hand = HandModel.from_arrays(a)
        res = solve(style, hand, prop, **params)
    except (KeyError, ValueError, TypeError, OSError, json.JSONDecodeError) as e:
        print(f"grip: bad input: {type(e).__name__}: {e}", file=sys.stderr)
        return 2
    names = list(res["bones"])
    seconds = round(time.perf_counter() - t0, 2)
    extra = {k: res[k] for k in ("frame_world_quat",) if k in res}
    report = {"style": res["style"], "side": res["side"], "report": res["report"],
              "solver": dict(res.get("solver", {}), seconds=seconds), **extra}
    np.savez(argv[1], bones=np.array(names), quats=np.array([res["bones"][n] for n in names], float).reshape(-1, 4),
             target_in_wrist=np.asarray(res["target_in_wrist"], float),
             report=np.array(json.dumps(jsonx.to_jsonable(report), ensure_ascii=False)))
    print(f"grip {res['style']} {res['side']}: {seconds}s", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
