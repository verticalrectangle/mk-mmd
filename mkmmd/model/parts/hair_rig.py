"""Chains for the hair part (bpy-free, numpy only): bones, dynamic rigid bodies, joints and skin weights.

A chain is a line of bones hanging from an anchor bone (the head, or a twitch bone): bone k runs from pts[k] to
pts[k + 1] (connected, so the last bone ends at the last point), every bone gets a dynamic capsule and a joint to the
previous body. mk's strand solver reads exactly this (rig.json chains = dynamic bodies grouped from each root that
hangs from a non-dynamic bone); MMD runs it with Bullet.

Body groups (PMX 0..15): the body part uses group 0 for its static colliders and ignores itself (no_collide = (0,)).
Chain bodies live in groups 4 (hair), 5 (ears), 6 (tails), 7 (ribbons) and ignore each other; the first body of a chain
sits in group 8 which also ignores group 0, so a root never fights the head it hangs from (the solver skips the same
pairs near a chain root)."""
from dataclasses import dataclass, field

import numpy as np

from ..part import Bone, Joint, RigidBody
from .hair_geo import arclen, smoothstep

GROUPS = {"hair": 4, "ears": 5, "tail": 6, "ribbon": 7}
ROOT_GROUP = 8
DYNAMIC = (4, 5, 6, 7, 8)

# per kind: mass (kg-ish), damping (lin, ang), friction, joint swing limit (rad), twist limit (rad), spring (N m / rad)
PHYSICS = {
    "hair": dict(mass=0.4, damp=(0.93, 0.95), friction=0.4, swing=0.55, twist=0.25, spring=6.0),
    "bangs": dict(mass=0.3, damp=(0.95, 0.96), friction=0.4, swing=0.35, twist=0.15, spring=14.0),
    "braid": dict(mass=0.6, damp=(0.92, 0.94), friction=0.5, swing=0.45, twist=0.2, spring=8.0),
    "ears": dict(mass=0.2, damp=(0.9, 0.95), friction=0.3, swing=0.35, twist=0.1, spring=22.0),
    "tail": dict(mass=0.9, damp=(0.9, 0.93), friction=0.5, swing=0.5, twist=0.3, spring=5.0),
    "ribbon": dict(mass=0.15, damp=(0.9, 0.93), friction=0.3, swing=0.7, twist=0.4, spring=2.0),
}


def euler_for_axis(d):
    """XYZ Euler (rad, Blender convention R = Rz Ry Rx) of the least rotation taking +Z onto direction d."""
    d = np.asarray(d, float)
    d = d / max(np.linalg.norm(d), 1e-12)
    c = float(np.clip(d[2], -1.0, 1.0))
    v = np.cross([0.0, 0.0, 1.0], d)
    s = np.linalg.norm(v)
    if s < 1e-9:
        R = np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    else:
        k = v / s
        K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
        R = np.eye(3) + s * K + (1 - c) * (K @ K)
    ry = -np.arcsin(np.clip(R[2, 0], -1.0, 1.0))
    if abs(R[2, 0]) < 0.999999:
        rx = np.arctan2(R[2, 1], R[2, 2])
        rz = np.arctan2(R[1, 0], R[0, 0])
    else:
        rx = np.arctan2(-R[1, 2], R[1, 1])
        rz = 0.0
    return (float(rx), float(ry), float(rz))


@dataclass
class Chain:
    """One built chain: bone names, the points they run through, and its weight function."""
    base: str
    names: list
    pts: np.ndarray                    # (n + 1, 3) bone heads then the tip
    anchor: str
    kind: str
    radius: np.ndarray                 # per bone (m)
    s: np.ndarray = None               # (n + 1,) cumulative arc length of pts
    hold: float = 0.35                 # the anchor bone fades out over 2 x hold x the first bone (0.7 of its length)
    extra: dict = field(default_factory=dict)

    def __post_init__(self):
        self.pts = np.asarray(self.pts, float)
        self.s = arclen(self.pts)

    @property
    def length(self):
        return float(self.s[-1])

    def project(self, P):
        """Arc coordinate (m) of the nearest point of the chain polyline for each point of P (m,3)."""
        P = np.asarray(P, float)
        a, b = self.pts[:-1], self.pts[1:]
        ab = b - a
        L2 = np.maximum((ab * ab).sum(1), 1e-18)
        t = np.clip(((P[:, None, :] - a[None]) * ab[None]).sum(2) / L2[None], 0.0, 1.0)
        q = a[None] + t[..., None] * ab[None]
        d2 = ((P[:, None, :] - q) ** 2).sum(2)
        k = d2.argmin(1)
        rows = np.arange(len(P))
        return self.s[:-1][k] + t[rows, k] * np.sqrt(L2[k])

    def weights(self, P=None, s=None, blend=0.6, hold=None):
        """{bone: (m,) weights} for points P (projected on the chain) or arc coordinates s; the anchor bone's weight
        is included under self.anchor. Smooth partition of unity: the anchor holds the root, each bone blends into the
        next across `blend` x the shorter neighbouring segment."""
        if s is None:
            s = self.project(P)
        s = np.asarray(s, float)
        n = len(self.names)
        seg = np.diff(self.s)
        hold = self.hold if hold is None else hold
        S = []
        c0, h0 = hold * seg[0], max(hold * seg[0], 1e-6)
        S.append(smoothstep((s - (c0 - h0)) / (2 * h0)))
        for k in range(1, n):
            h = max(blend * 0.5 * min(seg[k - 1], seg[k]), 1e-6)
            S.append(smoothstep((s - (self.s[k] - h)) / (2 * h)))
        out = {}
        rest = np.ones_like(s)
        out[self.anchor] = rest * (1 - S[0])
        rest = rest * S[0]
        for k in range(n):
            nxt = S[k + 1] if k + 1 < n else np.zeros_like(s)
            out[self.names[k]] = rest * (1 - nxt)
            rest = rest * nxt
        return out


class Rig:
    """Accumulates the bones, bodies and joints of the hair part's chains."""

    def __init__(self):
        self.bones, self.bodies, self.joints, self.chains = [], [], [], {}

    def chain(self, base, pts, anchor, *, kind="hair", radius=0.01, names=None, anchor_body=None, mass=None,
              group=None, dynamic=True, name_en="", first_group=ROOT_GROUP, physics=None, hold=0.35,
              layer=0):
        """Add a chain of len(pts) - 1 bones named base + 1, base + 2 ... (or `names`), hanging from `anchor`.
        radius: scalar or per bone. Returns the `Chain`."""
        pts = np.asarray(pts, float)
        n = len(pts) - 1
        names = names or [f"{base}{k + 1}" for k in range(n)]
        if len(names) != n:
            raise ValueError(f"{base}: {n} bones need {n} names")
        rad = np.broadcast_to(np.asarray(radius, float), (n,)).copy()
        ph = dict(PHYSICS[kind if kind in PHYSICS else "hair"])
        ph.update(physics or {})
        if mass is not None:
            ph["mass"] = mass
        grp = GROUPS.get("hair" if kind in ("bangs", "braid") else kind, 4) if group is None else group
        for k in range(n):
            self.bones.append(Bone(
                name=names[k], head=tuple(float(x) for x in pts[k]), tail=tuple(float(x) for x in pts[k + 1]),
                parent=names[k - 1] if k else anchor, tail_bone=names[k + 1] if k + 1 < n else "",
                name_en=f"{name_en}{k + 1}" if name_en else "", layer=layer,
                after_physics=bool(dynamic)))
        if dynamic:
            prev = anchor_body
            for k in range(n):
                a, b = pts[k], pts[k + 1]
                length = float(np.linalg.norm(b - a))
                r = float(min(rad[k], length * 0.45))
                mid = (a + b) / 2
                rot = euler_for_axis(b - a)
                root = k == 0
                self.bodies.append(RigidBody(
                    name=names[k], bone=names[k], shape="capsule", size=(r, max(length - 2 * r, 0.002), 0.0),
                    location=tuple(float(x) for x in mid), rotation=rot, mode="dynamic",
                    group=first_group if root else grp,
                    no_collide=((0,) + DYNAMIC) if root else DYNAMIC,
                    mass=ph["mass"] * (1.0 if not root else 0.5), damping=ph["damp"], friction=ph["friction"]))
                if prev is not None:
                    sw, tw, sp = ph["swing"], ph["twist"], ph["spring"]
                    self.joints.append(Joint(
                        name=f"J_{names[k]}", a=prev, b=names[k], location=tuple(float(x) for x in a), rotation=rot,
                        rot_lo=(-sw, -sw, -tw), rot_hi=(sw, sw, tw), spring_rot=(sp, sp, sp * 0.5)))
                prev = names[k]
        ch = Chain(base=base, names=list(names), pts=pts, anchor=anchor, kind=kind, radius=rad, hold=hold)
        self.chains[base] = ch
        return ch
