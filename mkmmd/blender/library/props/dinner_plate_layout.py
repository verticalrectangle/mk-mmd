"""Layout of the dinner plate (`card = "library:dinner_plate"`; pure Python + numpy, no bpy: importable from tests): a glazed
ceramic dinner plate with stylised veggies on it, its meshes, the colour roles, the options and the card.

Frame (metres): origin floor_center (the foot ring's bottom at z = 0, on the plate's axis), front -Y. The plate is one
lathe of a filleted profile (`profile`): a foot ring 8 mm wide and 4.5 mm tall, the well's floor (its top at Z_WELL, flat
out to R_WELL), a cove up into the rim, the rim (the marly) flaring up and out about 19 degrees, and a rolled lip at R_EDGE
(0.27 m across). A coloured band runs round the rim's top (`band`, drawn by the material: BAND, LINE).

The veggies (option `veggies`, default true; `seed` jitters where they lie): two broccoli florets on their sides (a
stalk with a cut end flaring into a dome of floret clusters and buds), five carrot coins shingled in a fan (the core rings
are drawn by the material from the uv: u = the share of the coin's radius), a little heap of peas, a cherry tomato with a
green calyx star and a stalk. True colours by default; every role overridable in `slots`.

Colour roles (`slots`: a palette slot or a hex): plate (Dawn `surface`: the glaze), band (Dawn `foam`), floret, stalk,
skin, flesh, core (the coins), pea, tomato, calyx.

Card: `use.rest` `well` (a plane: the well's floor), `use.look` `centre`, `use.grip` `rim` (type "pinch": the rim at the
plate's back edge, +Y, where a hand in front of it holds it: thumb over, fingers under)."""
import math
import re

import numpy as np

from ....core import palette as PAL
from ....core import shell as S

R_EDGE = 0.135              # 0.27 across
Z_UNDER = 0.0045            # the underside inside the foot ring
FOOT = (0.0775, 0.0855)     # the foot ring's inner and outer radius at its bottom
Z_WELL = 0.0088             # the well's floor (top)
R_WELL = 0.08               # the flat of the well
LIP_C, LIP_R = (0.1326, 0.0215), 0.0024    # the rolled lip: a round bead
RIM_TOP_IN = (0.098, 0.0128)               # where the rim's top turns down into the well
RIM_UNDER_IN = (0.100, 0.0075)             # where the rim's underside starts out from the foot
BAND = (0.1135, 0.1235)     # the coloured band on the rim's top, radii
LINE = (0.1062, 0.0007)     # a thin line inside it: its radius and half-width
SEG = 128
GRIP_SPAN, GRIP_LENGTH = 0.03, 0.06        # the rim a pinch holds: how deep from the edge, how long along it
DIMPLE = 0.0008            # the tomato's dimple under its calyx

ROLES = {"plate": "surface", "band": "foam", "floret": "#2f6b2a", "stalk": "#9cc069", "skin": "#e0650f",
         "flesh": "#ee6f0a", "core": "#f39b22", "pea": "#69ad37", "tomato": "#d62a1e", "calyx": "#3e8b2e"}
MATS = ("plate", "floret", "stalk", "coin", "pea", "tomato", "calyx")   # material index -> material (`coin`: skin, flesh,
                                                                        # core by the uv)
SLOTS = tuple(PAL.PALETTES["rose-pine-dawn"])
HEX = re.compile(r"^#[0-9a-fA-F]{6}$")


# ================================================================================================= the plate
def profile():
    """(r, z) of the plate's section from the underside on the axis, out under the foot ring, up the underside of the rim,
    round the lip and back in over the rim and the well to the axis: filleted corners, the lip a round bead."""
    a = np.radians(np.linspace(-75.0, 115.0, 13))
    lip = np.stack([LIP_C[0] + LIP_R * np.cos(a), LIP_C[1] + LIP_R * np.sin(a)], 1)
    under = [(0.0, Z_UNDER), (FOOT[0] - 0.0005, Z_UNDER), (FOOT[0], 0.0), (FOOT[1], 0.0), (FOOT[1] + 0.0005, Z_UNDER),
             RIM_UNDER_IN, (0.129, 0.0185)]
    top = [RIM_TOP_IN, (0.086, Z_WELL), (0.0, Z_WELL)]
    pts = np.concatenate([np.array(under), lip, np.array(top)])
    rad = np.concatenate([[0.0, 0.002, 0.0012, 0.0012, 0.0025, 0.015, 0.002], np.zeros(len(lip)), [0.015, 0.008, 0.0]])
    return S.fillet(pts, rad, n=6)


def plate_mesh():
    """The plate (material `plate`): a lathe of `profile`, closed."""
    return S.lathe(profile(), seg=SEG, mat=MATS.index("plate"))


def _rim_at(r):
    """(z of the rim's top, z of its underside) at radius r on the rim."""
    P = profile()
    k = int(np.argmax(P[:, 0]))                                  # the lip's outermost point splits under from top
    under, top = P[:k + 1], P[k:][::-1]
    return float(np.interp(r, top[:, 0], top[:, 1])), float(np.interp(r, under[:, 0], under[:, 1]))


def rim_grip():
    """The `use.grip` entry `rim` (a pinch at the back edge, +Y): its centre in the middle of the rim's thickness GRIP_SPAN /
    2 in from the edge along the rim's slope, `normal` outward along the slope, `axis` along the edge, `width` the rim's
    thickness across it."""
    r0, r1 = R_EDGE - GRIP_SPAN, R_EDGE - 0.004
    t0, u0 = _rim_at(r0)
    t1, u1 = _rim_at(r1)
    slope = math.atan2(0.5 * (t1 + u1) - 0.5 * (t0 + u0), r1 - r0)
    rc = R_EDGE - 0.5 * GRIP_SPAN
    top, under = _rim_at(rc)
    n = [0.0, round(math.cos(slope), 5), round(math.sin(slope), 5)]
    return {"name": "rim", "type": "pinch", "width": round((top - under) * math.cos(slope), 5), "span": GRIP_SPAN,
            "length": GRIP_LENGTH, "center": [0.0, round(rc, 5), round(0.5 * (top + under), 5)], "axis": [1.0, 0.0, 0.0],
            "normal": n}


# ================================================================================================= veggies
def _sphere(r, seg=16, rings=9, mat=0):
    a = np.linspace(-0.5 * math.pi, 0.5 * math.pi, rings)
    return S.lathe(np.stack([r * np.cos(a), r * np.sin(a)], 1), seg=seg, mat=mat)


def _floret_head(radius, seed):
    """One floret about the origin, its crown along +Z: a puff with a flatter underside, budded on top (material `floret`;
    the underside, where its branch comes in, `stalk`)."""
    rng = np.random.RandomState(seed)
    a = np.radians(np.linspace(-90.0, 90.0, 12))
    prof = np.stack([radius * np.cos(a), radius * np.where(a < 0.0, 0.55, 0.9) * np.sin(a) + 0.05 * radius], 1)
    m = S.lathe(prof, seg=28, mat=MATS.index("floret"))
    U = m.V / np.maximum(np.linalg.norm(m.V, axis=1), 1e-9)[:, None]
    buds = rng.normal(size=(14, 3)) * [1.0, 1.0, 0.5] + [0.0, 0.0, 0.75]
    buds /= np.linalg.norm(buds, axis=1, keepdims=True)
    bump = 0.13 * np.exp(-(np.arccos(np.clip(U @ buds.T, -1.0, 1.0)) / 0.38) ** 2).max(axis=1)
    m.V = m.V * (1.0 + bump * np.clip(U[:, 2] + 0.3, 0.0, 1.0))[:, None]
    return m.assign(MATS.index("stalk"), lambda c, n: c[:, 2] < -0.3 * radius)


def floret_mesh(seed):
    """A broccoli floret standing on its cut end at the origin along +Z (about 0.06 tall): a trunk with a rounded cut end
    (material `stalk`) splitting into short branches, each ending in a budded floret (`floret`): one on top, the others in
    a ring round it, together a crown."""
    rng = np.random.RandomState(seed)
    trunk = [(0.0, 0.0), (0.0066, 0.0), (0.0076, 0.0008), (0.0078, 0.0016), (0.0064, 0.008), (0.0058, 0.016),
             (0.0068, 0.022), (0.0056, 0.0265), (0.0, 0.0275)]
    parts = [S.lathe(np.array(trunk), seg=24, mat=MATS.index("stalk"))]
    fork = np.array([0.0, 0.0, 0.022])
    n = rng.randint(5, 7)
    phase = rng.uniform(0.0, 2.0 * math.pi)
    for k in range(n + 1):
        if k == n:
            d, reach, rad = np.array([0.0, 0.0, 1.0]), 0.019, 0.0125
        else:
            el = math.radians(rng.uniform(34.0, 52.0))
            az = phase + 2.0 * math.pi * (k + rng.uniform(-0.12, 0.12)) / n
            d = np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)])
            reach, rad = rng.uniform(0.013, 0.016), rng.uniform(0.0105, 0.0125)
        tip = fork + np.array([0.0, 0.0, 0.006]) + d * reach
        u = np.linspace(0.0, 1.0, 6)[:, None]
        path = (1.0 - u) ** 2 * fork + 2.0 * u * (1.0 - u) * (fork + [0.0, 0.0, 0.008]) + u * u * tip   # up, then out
        parts.append(S.tube(path, np.linspace(0.0034, 0.0025, 6), sides=10, mat=MATS.index("stalk")))
        head = _floret_head(rad, seed * 31 + k)
        R = S.rotation_between(np.array([0.0, 0.0, 1.0]), d)
        parts.append(head.apply(R, tip + d * 0.35 * rad))
    return S.merge(parts)


def coin_mesh(radius, thick, seed):
    """A carrot coin lying flat, its bottom at z = 0 (material `coin`): a disc with a rounded edge, a little out of round;
    uv u = the share of its radius (the material draws the core rings from it), v = 1."""
    rng = np.random.RandomState(seed)
    f = 0.0012
    arc = lambda c, a0: np.stack([c[0] + f * np.cos(np.radians(np.linspace(a0, a0 + 90.0, 5)[1:])),       # noqa: E731
                                  c[1] + f * np.sin(np.radians(np.linspace(a0, a0 + 90.0, 5)[1:]))], 1)
    rho = np.linspace(0.0, radius - f, 6)
    prof = np.concatenate([np.stack([rho, np.zeros(6)], 1), arc((radius - f, f), -90.0),
                           arc((radius - f, thick - f), 0.0), np.stack([rho[::-1][1:], np.full(5, thick)], 1)])
    m = S.lathe(prof, seg=48, mat=MATS.index("coin"))
    r = np.hypot(m.V[:, 0], m.V[:, 1])
    th = np.arctan2(m.V[:, 1], m.V[:, 0])
    wob = 1.0 + 0.025 * np.cos(3 * th + rng.uniform(0, 6.3)) + 0.015 * np.cos(5 * th + rng.uniform(0, 6.3))
    m.V[:, :2] *= wob[:, None]
    m.UV = np.stack([np.clip(r / radius, 0.0, 1.0), np.ones(len(r))], 1)
    return m


def pea_heap(seed):
    """A little heap of peas (material `pea`), resting on z = 0 about the origin."""
    rng = np.random.RandomState(seed)
    r = 0.0046
    base = [(0.0, 0.0)] + [(2.05 * r * math.cos(t), 2.05 * r * math.sin(t)) for t in np.radians(np.arange(6) * 60.0 + 15.0)]
    base += [(4.0 * r, -0.6 * r), (-3.4 * r, 2.6 * r)]
    top = [(1.05 * r * math.cos(t), 1.05 * r * math.sin(t)) for t in np.radians([30.0, 150.0, 270.0])]
    out = []
    for (x, y), z in [(p, r) for p in base] + [(p, r + 1.55 * r) for p in top] + [((0.3 * r, 0.2 * r), r + 2.9 * r)]:
        rr = r * rng.uniform(0.9, 1.08)
        s = _sphere(rr, seg=16, rings=9, mat=MATS.index("pea"))
        s.V = s.V * [1.0, 1.0, 0.94] + [x + rng.uniform(-0.4, 0.4) * 0.001, y + rng.uniform(-0.4, 0.4) * 0.001,
                                        z - (r - rr) * 0.94]
        out.append(s)
    return S.merge(out)


def tomato_mesh(seed):
    """A cherry tomato resting on z = 0 (material `tomato`): a slightly oblate, faintly lobed ball with a dimple on top; its
    calyx star of five pointed sepals curling up at their tips and a short stalk (`calyx`)."""
    rng = np.random.RandomState(seed)
    R, flat = 0.0135, 0.86
    a = np.linspace(-0.5 * math.pi, 0.5 * math.pi, 15)
    m = S.lathe(np.stack([R * np.cos(a), R * flat * (1.0 + np.sin(a))], 1), seg=40, mat=MATS.index("tomato"))
    th = np.arctan2(m.V[:, 1], m.V[:, 0])
    side = np.hypot(m.V[:, 0], m.V[:, 1]) / R
    m.V[:, :2] *= (1.0 + 0.025 * np.cos(5 * th) * side ** 2)[:, None]
    top = 2.0 * R * flat
    m.V[:, 2] -= DIMPLE * np.exp(-(side / 0.28) ** 2) * (m.V[:, 2] > R * flat)         # the dimple under the calyx
    parts = [m]
    phase = rng.uniform(0.0, 2.0 * math.pi / 5.0)
    for k in range(5):
        az = phase + 2.0 * math.pi * k / 5.0
        u = np.linspace(0.0, 1.0, 14)
        ang = u * 0.62 + 0.02                                                # down the shoulder of the ball
        rr = R * np.sin(ang) * 1.02
        zz = (R * flat + R * flat * np.cos(ang) * 1.01 + 0.0035 * u ** 3                # the tip curls up off it
              - DIMPLE * np.exp(-((rr / R) / 0.28) ** 2))
        path = np.stack([rr * math.cos(az), rr * math.sin(az), zz], 1)
        w = 0.0024 * (1.0 - u) ** 0.8 * (0.55 + 0.45 * np.sin(math.pi * np.minimum(u * 1.4, 1.0))) + 0.00025
        sec = S.ellipse(1.0, 0.18, 12)
        parts.append(S.sweep(path, sec, scale=np.stack([w, np.full(len(u), 0.0035)], 1), up=(0.0, 0.0, 1.0),
                             mat=MATS.index("calyx")))
    stem = np.array([[0.0, 0.0, top - 0.003], [0.0, 0.0, top + 0.002], [0.0006, 0.0, top + 0.005], [0.0016, 0.0, top + 0.0068]])
    parts.append(S.tube(stem, np.array([0.0013, 0.0012, 0.0011, 0.001]), sides=10, mat=MATS.index("calyx")))
    return S.merge(parts)


def _rest(mesh, R, at):
    """`mesh` turned by R and put down with its lowest point on the well's floor at (x, y)."""
    m = mesh.apply(R)
    m.V += [at[0], at[1], Z_WELL - m.V[:, 2].min() - 0.0003]
    return m


def veggies_mesh(seed=1):
    """Every veggie on the plate, in the plate's frame, all inside the well: two florets on their sides at the back (their
    cut ends near each other, the crowns out to the left and right), the coins to the right, the peas front left, the
    tomato front right."""
    rng = np.random.RandomState(seed)
    j = lambda s: rng.uniform(-s, s)                                                # noqa: E731
    parts = []
    for (x, y), yaw, sd in (((0.002, 0.03), 162.0, 3), ((0.014, 0.026), 38.0, 5)):
        R = S.rot_matrix(0.0, 114.0 + j(4.0), yaw + j(8.0))                 # on its side: the crown and the cut end down
        parts.append(_rest(floret_mesh(seed * 10 + sd), R, (x + j(0.003), y + j(0.003))))
    row = np.array([math.cos(math.radians(15.0)), math.sin(math.radians(15.0)), 0.0])
    for i in range(5):                                                    # shingled: each leans on the one before
        rad, thick = 0.0112 + 0.0016 * rng.rand(), 0.0046 + 0.0008 * rng.rand()
        R = S.rot_matrix(0.0, 0.0, j(180.0))
        if i:
            R = S.axis_angle(np.cross([0.0, 0.0, 1.0], row), 12.0 + j(3.0)) @ R
        parts.append(_rest(coin_mesh(rad, thick, seed * 100 + i), R, row[:2] * 0.0105 * i + [0.018, -0.008]))
    parts.append(_rest(pea_heap(seed * 7 + 1), np.eye(3), (-0.04 + j(0.003), -0.028 + j(0.003))))
    parts.append(_rest(tomato_mesh(seed * 3 + 2), S.rot_matrix(j(6.0), j(6.0), j(180.0)), (0.03 + j(0.003), -0.036 + j(0.003))))
    return S.merge(parts)


# ================================================================================================= options and card
def options(slots):
    """{"veggies": bool, "seed": int, "roles": {role: slot or hex}} from the `slots` of the [[prop]] (keys this prop does
    not know are ignored), checked."""
    slots = dict(slots or {})
    veg = slots.get("veggies", True)
    if not isinstance(veg, bool):
        raise ValueError(f"dinner_plate: veggies is true (the veggies on it) or false (an empty plate), got {veg!r}")
    seed = slots.get("seed", 1)
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError(f"dinner_plate: seed is a whole number >= 0, got {seed!r}")
    roles = {}
    for role, default in ROLES.items():
        v = slots.get(role, default)
        if not isinstance(v, str) or not (v in SLOTS or HEX.match(v)):
            raise ValueError(f"dinner_plate: {role} is a palette slot ({', '.join(SLOTS)}) or a hex \"#rrggbb\", got {v!r}")
        roles[role] = v
    return {"veggies": veg, "seed": seed, "roles": roles}


def card(name, roles, size, veggies=True):
    """The prop card: `use.rest` well, `use.look` centre, `use.grip` rim."""
    objs = [f"{name}_plate"] + ([f"{name}_veggies"] if veggies else [])
    return {"size": [round(float(v), 4) for v in size], "origin": "floor_center", "front": "-Y", "slots": dict(roles),
            "use": {"rest": [{"name": "well", "type": "plane", "center": [0.0, 0.0, Z_WELL], "normal": [0.0, 0.0, 1.0],
                              "radius": R_WELL}],
                    "look": [{"name": "centre", "point": [0.0, 0.0, Z_WELL]}],
                    "grip": [rim_grip()]},
            "colliders": [], "params": {}, "objects": objs}
