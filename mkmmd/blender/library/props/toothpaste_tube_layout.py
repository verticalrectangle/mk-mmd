"""Layout of the giant toothpaste tube (pure Python + numpy, no bpy: importable from tests): its meshes, the squeeze, the
colour roles and the card.

Frame (metres): the tube stands on its crimped end along +Z, the crimp's edge at z = 0 and the nozzle's mouth at z =
LENGTH; the crimp is flat across X (its faces toward -Y and +Y). A 100 ml tube (38 mm across, the shoulder a short cone
into a threaded 12 mm neck, the crimp 58 mm wide) at about 3.4 times its size: 0.60 m long. The body is ONE cage of
rings from the crimp's edge up the body, over the shoulder and the threaded neck, over the lip and down the bore: each
ring an ellipse whose perimeter stays the tube's own (2 pi R) while its half thickness b grows from the sealed crimp
to R at the shoulder (a laminate tube flattens without stretching), the crimp ridged, the neck's thread a helix.

`squeeze` (a root property, 0 .. 1, for [[key]]; a [[ribbon]] `from` its nozzle keys it over the draw) drives the shape
key "squeeze": the body pressed flat round the hold (b down by SQUEEZE at its middle, a Gaussian along z), its
width growing as the perimeter stays.

Colour roles (`slots`: a palette slot or a hex): body (surface), shoulder (love: the shoulder and the neck), stripes
(five bands round the body, "love,gold,foam,iris,rose": the paste's own colours), paste (foam: what shows in the
mouth), cap (gold, with `capped = true`: a ribbed screw cap on the neck; the default leaves the mouth open). No logo or
lettering.

Card: `use.anchor` `nozzle` (the mouth, pointing out along +Z, with the radius of the paste it lets out), `use.grip`
`hold` (type "hold": the body a fist squeezes, along +Z toward the nozzle), `use.look` `nozzle` and `body`; params
`squeeze`."""
import math

import numpy as np

from ....core import shell as S

LENGTH = 0.60
R = 0.065                       # the round body's radius at the shoulder
CRIMP = 0.035                   # the crimp's height
SHOULDER = 0.50                 # where the body meets the shoulder
NECK_R, NECK = 0.030, (0.538, 0.585)    # the threaded neck's radius and span
BORE_R, BORE_DEPTH = 0.0225, 0.04       # the mouth's inner radius and how deep the bore shows
PASTE_R = 0.020                 # the paste the mouth lets out (the nozzle's `radius`)
THREAD = (0.0018, 0.008)        # the thread's height and pitch
SEAL = 0.004                    # the crimp's half thickness (and the body's least)
SQUEEZE = (0.5, 0.27, 0.11)     # at squeeze 1: b down by half at z 0.27, falling off over 0.11 m
HOLD = (0.06, 0.48)             # the span a fist can hold
STRIPES = (0.31, 0.39)          # the five bands' span
RING = 56
STRIPE_ROLES = ("stripe0", "stripe1", "stripe2", "stripe3", "stripe4")
ROLES = {"body": "surface", "shoulder": "love", "paste": "foam", "cap": "gold",
         "stripe0": "love", "stripe1": "gold", "stripe2": "foam", "stripe3": "iris", "stripe4": "rose"}
MATS = tuple(ROLES)             # material index -> role


def perimeter(a, b):
    """Ramanujan's perimeter of an ellipse with half axes a, b."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    return math.pi * (3.0 * (a + b) - np.sqrt((3.0 * a + b) * (a + 3.0 * b)))


def half_width(b, perim=2.0 * math.pi * R):
    """The half width a (>= b) of the ellipse of half thickness b with the tube's perimeter (bisection)."""
    b = np.asarray(b, float)
    lo, hi = b.copy(), np.full_like(b, perim / 4.0)
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        big = perimeter(mid, b) > perim
        hi, lo = np.where(big, mid, hi), np.where(big, lo, mid)
    return 0.5 * (lo + hi)


def body_b(z, squeeze=0.0):
    """The body's half thickness at heights z (n,) in the crimp and the body (z <= SHOULDER): the seal, ridged, then
    a rise to R, pressed by `squeeze`."""
    z = np.asarray(z, float)
    f = np.clip((z - CRIMP) / (SHOULDER - CRIMP), 0.0, 1.0)
    b = SEAL + (R - SEAL) * np.sin(0.5 * math.pi * f ** 0.6)
    ridge = 0.0012 * np.clip(np.sin(2.0 * math.pi * z / 0.007), 0.0, 1.0) * (z < CRIMP - 0.004)
    b = np.where(z < CRIMP, SEAL + ridge, b)
    dent, zc, sig = SQUEEZE
    press = 1.0 - dent * float(squeeze) * np.exp(-((z - zc) / sig) ** 2) * (z > CRIMP)
    return np.maximum(b * press, SEAL)


def _profile():
    """(z, r or None) stations above the body: None marks the elliptic body; the shoulder, neck, lip and bore are round."""
    zs = list(np.arange(0.0, SHOULDER, 0.005)) + [SHOULDER]
    rows = [(z, None) for z in zs]
    shoulder = S.pchip(np.array([SHOULDER, 0.512, 0.528, 0.535, NECK[0]]),
                       np.array([[R], [0.060], [0.044], [0.033], [NECK_R]]),
                       np.linspace(SHOULDER, NECK[0], 9)[1:])[:, 0]
    rows += [(z, r) for z, r in zip(np.linspace(SHOULDER, NECK[0], 9)[1:], shoulder)]
    rows += [(z, "thread") for z in np.arange(NECK[0] + 0.0016, NECK[1], 0.0016)]
    lip = [(NECK[1], NECK_R), (0.592, NECK_R - 0.0004), (0.597, NECK_R - 0.0018), (LENGTH, NECK_R - 0.0045),
           (LENGTH, BORE_R + 0.0018), (LENGTH - 0.002, BORE_R)]
    rows += lip + [(z, BORE_R) for z in np.linspace(LENGTH - 0.006, LENGTH - BORE_DEPTH, 6)]
    return rows


def rings(squeeze=0.0):
    """The cage's rings (ns, RING, 3), and per ring the role of the band above it."""
    t = 2.0 * math.pi * np.arange(RING) / RING
    out, roles = [], []
    for z, r in _profile():
        if r is None:
            b = float(body_b(np.array([z]), squeeze)[0])
            a = float(half_width(np.array([b]))[0])
            ring = np.stack([a * np.cos(t), b * np.sin(t), np.full(RING, z)], 1)
        elif r == "thread":
            rr = NECK_R + THREAD[0] * np.clip(np.sin(2.0 * math.pi * z / THREAD[1] - t), 0.0, 1.0) ** 0.7
            ring = np.stack([rr * np.cos(t), rr * np.sin(t), np.full(RING, z)], 1)
        else:
            ring = np.stack([r * np.cos(t), r * np.sin(t), np.full(RING, z)], 1)
        out.append(ring)
        if z >= SHOULDER:
            roles.append("shoulder")
        elif STRIPES[0] <= z < STRIPES[1]:
            roles.append(STRIPE_ROLES[min(4, int(5 * (z - STRIPES[0]) / (STRIPES[1] - STRIPES[0])))])
        else:
            roles.append("body")
    return np.array(out), roles


def body_mesh(squeeze=0.0):
    """The tube: one cage (its crimp edge closed by a thin fan, the bore's floor by a flat one), materials by band."""
    G, roles = rings(squeeze)
    m = S.cage(G, closed=True, caps=("fan", "fan"), cap_axes=(None, (0.0, 0.0, 1.0)), cap_bulge=(0.0005, 0.0),
               cap_rings=2)
    per_ring = np.array([MATS.index(r) for r in roles])
    band = np.repeat(per_ring[:-1], RING)                  # body quads: ring i to i + 1, RING of them
    m.Qm[:len(band)] = band
    m.Qm[len(band):] = MATS.index("body")                  # the crimp's edge
    m.Qm[-(len(m.Qm) - len(band)) // 2:] = MATS.index("paste")  # the bore's floor
    m.Tm[:] = MATS.index("body")
    m.Tm[len(m.Tm) // 2:] = MATS.index("paste")
    return m


def cap_mesh(seg=48):
    """A ribbed screw cap over the neck (with `capped`): a lathe whose wall carries 24 shallow ribs."""
    prof = S.fillet(np.array([(0.0, NECK[0] + 0.004), (NECK_R + 0.0065, NECK[0] + 0.004), (NECK_R + 0.0065, LENGTH + 0.012),
                              (0.0, LENGTH + 0.012)]), [0.0, 0.0025, 0.004, 0.0], 4)
    m = S.lathe(prof, seg=seg * 2, mat=MATS.index("cap"))
    ang = np.arctan2(m.V[:, 1], m.V[:, 0])
    r = np.hypot(m.V[:, 0], m.V[:, 1])
    rib = 1.0 + 0.04 * np.clip(np.cos(24 * ang), 0.0, 1.0) * (r > NECK_R + 0.006)
    m.V[:, 0] *= rib
    m.V[:, 1] *= rib
    return m


def squeeze_offsets(s=1.0):
    """How far each body vertex moves at squeeze `s` (the shape key's offsets, (n, 3))."""
    return body_mesh(s).V - body_mesh(0.0).V


def card(name, slots=None, capped=False):
    """The prop card (docs/design.md: Prop card): the nozzle, the hold, the look points; `squeeze` its live root
    property."""
    a0 = float(half_width(np.array([SEAL]))[0])
    hold = {"name": "hold", "type": "hold", "center": [0.0, 0.0, round(sum(HOLD) / 2, 4)], "axis": [0.0, 0.0, 1.0],
            "length": round(HOLD[1] - HOLD[0], 4), "radius": round(float(body_b(np.array([sum(HOLD) / 2]))[0]) * 0.75, 4)}
    objects = [f"{name}_body"] + ([f"{name}_cap"] if capped else [])
    return {"size": [round(2.0 * a0, 4), round(2.0 * R, 4), LENGTH + (0.012 if capped else 0.0)], "origin": "tail",
            "front": "-Y", "slots": dict(slots or {}), "params": {"squeeze": 0.0},
            "use": {"anchor": [{"name": "nozzle", "point": [0.0, 0.0, LENGTH], "dir": [0.0, 0.0, 1.0], "radius": PASTE_R}],
                    "grip": [hold],
                    "look": [{"name": "nozzle", "point": [0.0, 0.0, LENGTH]},
                             {"name": "body", "point": [0.0, 0.0, round(sum(HOLD) / 2, 4)]}]},
            "objects": objects}
