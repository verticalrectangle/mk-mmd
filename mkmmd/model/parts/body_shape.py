"""Proportions of the body part: every number that sets the body's size and shape is resolved here from the spec
(`proportions.toml`: landmarks, torso cuts, limb diameters, foot; then the `[body]` table of body.toml), over built-in
defaults (a measured slender figure), so the part also builds without a spec.

`resolve(prop, cfg)` returns `Shape`: `land` (semantic bone name -> head position, plus extra non-bone points such as
`toe_end.L`), `hand` (the hand's design, body_hand.DESIGN with `[body.hand]` over it), `dims` (torso cuts, limb sections,
foot) and the hand `frame`. The finger joints are not read from the landmarks: they are made from the hand's design at
the wrist (body_hand.joints), or taken from the hand mesh `[body.hand] mesh` names (body_hand_mesh.joints). Model
space: metres, Z up, facing -Y, her left is +X; only the left side is stored, the right side is its mirror."""
import copy
from dataclasses import dataclass, field

import numpy as np

from ..spec import expand
from . import body_hand, body_hand_mesh

# ---------------------------------------------------------------- defaults
LAND = {
    "root": (0.0, -0.0027, -0.0036),
    "center": (0.0, -0.0167, 0.8419),
    "groove": (0.0, -0.0167, 0.8564),
    "waist": (0.0, -0.0508, 0.8823),
    "lower_body": (0.0, -0.0508, 0.8766),
    "upper_body": (0.0, -0.0508, 0.8880),
    "upper_body2": (0.0, -0.0479, 0.9680),
    "neck": (0.0, 0.0020, 1.1563),
    "head": (0.0, -0.0045, 1.2041),
    "head_tip": (0.0001, -0.0133, 1.3483),
    "shoulder.L": (0.0438, 0.0084, 1.1265),
    "arm.L": (0.0868, 0.0057, 1.1206),
    "arm_twist.L": (0.1893, 0.0081, 1.0560),
    "elbow.L": (0.2539, 0.0132, 1.0140),
    "wrist_twist.L": (0.3395, 0.0089, 0.9702),
    "wrist.L": (0.3941, 0.0061, 0.9418),
    "leg.L": (0.0755, -0.0356, 0.7425),
    "knee.L": (0.0527, -0.0320, 0.4189),
    "ankle.L": (0.0602, -0.0008, 0.0936),
    "toe.L": (0.0569, -0.0796, 0.0390),
    "toe_end.L": (0.0538, -0.1566, 0.0000),
    "leg_ik.L": (0.0601, -0.0006, 0.0902),
    "toe_ik.L": (0.0538, -0.1566, 0.0000),
}

TORSO = dict(
    z=[0.7220, 0.7410, 0.7600, 0.7790, 0.7980, 0.8170, 0.8360, 0.8550, 0.8740, 0.8930, 0.9120, 0.9310, 0.9500, 0.9690,
       0.9880, 1.0070, 1.0260, 1.0450, 1.0640, 1.0830, 1.1020, 1.1210, 1.1400, 1.1590],
    width=[0.2110, 0.2336, 0.2458, 0.2408, 0.2322, 0.2215, 0.2027, 0.1864, 0.1708, 0.1555, 0.1426, 0.1337, 0.1359,
           0.1412, 0.1506, 0.1570, 0.1687, 0.1730, 0.1703, 0.1675, 0.1436, 0.1060, 0.0763, 0.0458],
    y_front=[-0.0805, -0.1012, -0.1064, -0.1081, -0.1088, -0.1063, -0.1059, -0.1071, -0.1073, -0.1054, -0.1050, -0.1045,
             -0.1043, -0.1036, -0.1017, -0.1025, -0.1095, -0.0866, -0.0825, -0.0724, -0.0558, -0.0406, -0.0238, -0.0246],
    y_back=[0.0462, 0.0547, 0.0584, 0.0587, 0.0564, 0.0495, 0.0390, 0.0243, 0.0154, 0.0080, 0.0056, 0.0066, 0.0141,
            0.0218, 0.0333, 0.0404, 0.0452, 0.0499, 0.0536, 0.0565, 0.0562, 0.0535, 0.0456, 0.0359],
)

# cross-section diameters [major, minor] (m) perpendicular to the bone
LIMB = dict(
    thigh_top=(0.1280, 0.1098), thigh_mid=(0.1030, 0.0881), thigh_low=(0.0768, 0.0694), knee=(0.0781, 0.0663),
    calf=(0.0770, 0.0772), shin_low=(0.0583, 0.0482), ankle=(0.0518, 0.0421),
    upper_arm_top=(0.0605, 0.0523), upper_arm_mid=(0.0557, 0.0505), elbow=(0.0529, 0.0483),
    forearm_top=(0.0566, 0.0508), forearm_mid=(0.0463, 0.0466), wrist=(0.0410, 0.0268),
)

FOOT = dict(foot_length=0.1696, foot_width=0.0700, foot_inner_floor_z=0.0229, shoe_y=(-0.1475, 0.0475))


@dataclass
class Shape:
    land: dict = field(default_factory=dict)         # semantic name -> (3,) array, incl. mirrored .R
    hand: dict = field(default_factory=dict)
    dims: dict = field(default_factory=dict)
    scale: float = 1.0
    frame: tuple = None                              # (a, r, n) of the left hand (body_hand.frame)
    notes: list = field(default_factory=list)


def mirror(p):
    q = np.array(p, float)
    q[0] = -q[0]
    return q


def _vec(v):
    return np.asarray(v, float).reshape(3)


def resolve(prop=None, cfg=None, base=None):
    """Landmarks, the hand's design and section tables from the `[proportions]` spec table `prop` (landmarks, sections)
    and the `[body]` table `cfg` (overrides: `land`, `hand`, `dims`). The finger joints come from the hand's design or
    its mesh (a relative `mesh` path is taken against `base`, the spec's folder); finger landmarks given in the spec are
    not used (a note says so)."""
    prop = prop or {}
    cfg = cfg or {}
    notes = []
    land = {k: _vec(v) for k, v in LAND.items()}
    given = prop.get("landmarks") or {}
    if given:
        for k, v in given.items():
            land[k] = _vec(v)
    else:
        notes.append("proportions.landmarks missing: built-in numbers used")
    for k, v in (cfg.get("land") or {}).items():
        land[k] = _vec(v)
    H = body_hand.resolve(cfg.get("hand"))
    if H["mesh"]:
        H["mesh"] = str(expand(H["mesh"], base))
        made, source = body_hand_mesh.joints(land, H), "its mesh ([body.hand] mesh)"
    else:
        made, source = body_hand.joints(land, H), "its design ([body.hand])"
    ignored = sorted(k for k in set(given) | set(cfg.get("land") or {}) if k in made)
    if ignored:
        notes.append(f"finger landmarks {', '.join(ignored)} not used: the hand is made from {source}")
    land.update(made)
    for k in list(land):
        if k.endswith(".L"):
            land[k[:-2] + ".R"] = mirror(land[k])
    dims = dict(torso=copy.deepcopy(TORSO), limb=dict(LIMB), foot=dict(FOOT))
    secs = prop.get("sections") or {}
    if secs.get("torso"):
        dims["torso"] = {k: list(v) for k, v in secs["torso"].items()}
    if secs.get("limb"):
        dims["limb"].update({k: tuple(v) for k, v in secs["limb"].items()})
    if secs.get("foot"):
        dims["foot"].update(secs["foot"])
    if secs.get("neck"):
        dims["neck"] = dict(secs["neck"])
    for k, v in (cfg.get("dims") or {}).items():
        if isinstance(v, dict) and isinstance(dims.get(k), dict):
            dims[k].update(v)
        else:
            dims[k] = v
    sh = Shape(land=land, hand=H, dims=dims, notes=notes)
    sh.frame = body_hand.frame(land, H)
    return sh
