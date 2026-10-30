"""Layout of the giant toothbrush (pure Python + numpy, no bpy: importable from tests): its meshes, the colour roles and the
card.

Frame (metres): the brush stands on the end of its handle along +Z, the end at z = 0 and the tip of its head at z =
LENGTH; the bristles stand out of the head's front (-Y). A manual toothbrush of about 190 mm (handle 120, neck 35, head
28 x 11.5 mm, tufts 11 mm tall, the neck angled so the head sits forward of the handle's line) at about 3.9 times
its size, the handle kept thick enough for a fist (45 x 32 mm at the grip): 0.74 m long. The body (handle, neck and
head) is ONE cage of superellipse rings whose width, thickness, squareness and forward offset run smoothly along z
(monotone cubics through the STATIONS), its ends domed; the bristles are 36 lathed tufts in four columns of nine,
the middle two columns in the tuft colour, the tip row and the heel row a little longer. No logo or lettering.

Colour roles (`slots` of the prop: a palette slot or a hex): handle (iris), grip (rose: the soft panels on the front
and back of the handle), bristles (surface), tufts (foam: the two middle columns).

Card: `use.grip` `hold` (type "hold": the handle a fist can close round, from its end to the neck, along +Z toward the
head), `use.anchor` `brush` (the middle of the bristles' tips, facing -Y: what goes on the teeth), `use.look` `head`
and `grip`."""
import math

import numpy as np

from ....core import shell as S

LENGTH = 0.74
#            z      width  thick   y-centre  squareness
STATIONS = [(0.000, 0.024, 0.016, 0.0000, 2.2),
            (0.006, 0.036, 0.024, 0.0000, 2.3),
            (0.018, 0.042, 0.029, 0.0000, 2.4),
            (0.060, 0.044, 0.031, 0.0000, 2.5),
            (0.170, 0.045, 0.032, 0.0000, 2.5),
            (0.300, 0.042, 0.029, 0.0000, 2.5),
            (0.400, 0.034, 0.024, -0.0010, 2.4),
            (0.470, 0.024, 0.018, -0.0030, 2.3),
            (0.530, 0.019, 0.015, -0.0060, 2.2),
            (0.585, 0.020, 0.015, -0.0100, 2.2),
            (0.615, 0.040, 0.019, -0.0130, 2.6),
            (0.660, 0.047, 0.020, -0.0140, 2.8),
            (0.710, 0.046, 0.020, -0.0140, 2.8),
            (0.732, 0.036, 0.018, -0.0140, 2.6),
            (0.740, 0.020, 0.012, -0.0140, 2.4)]
RING = 40                                  # points round a ring
STEP = 0.006                               # ring spacing along the body (m)
GRIP = (0.07, 0.30)                        # the soft panels' z span
HOLD = (0.02, 0.50)                        # the span a fist can hold: from the end to the neck
TUFT_COLS = (-0.0165, -0.0055, 0.0055, 0.0165)
TUFT_ROWS = np.linspace(0.628, 0.722, 9)
TUFT_R = 0.0042                            # a tuft's radius at its root
TUFT_H = (0.036, 0.040)                    # outer and middle columns' heights; the tip and heel rows +4 mm
SINK = 0.003                               # tufts go this far into the head
ROLES = {"handle": "iris", "grip": "rose", "bristles": "surface", "tufts": "foam"}
MATS = tuple(ROLES)                        # material index -> role


def section(z):
    """(width, thickness, y-centre, squareness) of the body at heights z (n,)."""
    st = np.array(STATIONS)
    return S.pchip(st[:, 0], st[:, 1:], np.asarray(z, float)).T


def surface_y(x, z):
    """y of the body's front (-Y) surface at (x, z): its superellipse ring there."""
    w, t, yc, p = (float(v) for v in section(np.array([z])).ravel())
    u = min(abs(2.0 * x / w), 1.0)
    return yc - 0.5 * t * (1.0 - u ** p) ** (1.0 / p)


def body_mesh():
    """The handle, neck and head: one cage of superellipse rings (materials handle and grip)."""
    z = np.linspace(0.0, LENGTH, int(math.ceil(LENGTH / STEP)) + 1)
    w, t, yc, p = section(z)
    rings = []
    for i in range(len(z)):
        o = S.superellipse(w[i], t[i], p[i], RING)
        rings.append(np.stack([o[:, 0], o[:, 1] + yc[i], np.full(RING, z[i])], 1))
    m = S.cage(np.array(rings), closed=True, caps=("fan", "fan"), cap_bulge=(0.004, 0.004), cap_rings=2,
               mat=MATS.index("handle"))
    P = m.V[m.Q]
    ctr = P.mean(1)
    nrm = np.cross(P[:, 2] - P[:, 0], P[:, 3] - P[:, 1])
    nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-12)
    soft = (ctr[:, 2] > GRIP[0]) & (ctr[:, 2] < GRIP[1]) & (np.abs(nrm[:, 1]) > 0.55)
    m.Qm[soft] = MATS.index("grip")
    return m


def tuft_height(col, row):
    h = TUFT_H[1] if abs(TUFT_COLS[col]) < 0.01 else TUFT_H[0]
    return h + (0.004 if row in (0, len(TUFT_ROWS) - 1) else 0.0)


def tuft_profile(h):
    """(r, z) of one tuft along +Z from its root, flaring a little and rounded at the top."""
    pts = np.array([(0.0, 0.0), (TUFT_R, 0.0), (TUFT_R * 1.12, h - 0.0025), (TUFT_R * 0.55, h), (0.0, h)])
    return S.fillet(pts, [0.0, 0.0, 0.0022, 0.0018, 0.0], 3)


def tufts():
    """[(x, z, height, middle column?)] of every tuft."""
    return [(x, float(zr), tuft_height(c, r), abs(x) < 0.01) for c, x in enumerate(TUFT_COLS)
            for r, zr in enumerate(TUFT_ROWS)]


def bristle_meshes(seg=14):
    """The tufts (bristles, then the middle columns' tufts), each standing out of the head along -Y from SINK inside it."""
    R = S.rot_matrix(90.0, 0.0, 0.0)                         # +Z -> -Y
    out = {"bristles": [], "tufts": []}
    for x, z, h, middle in tufts():
        m = S.lathe(tuft_profile(h + SINK), seg=seg, mat=MATS.index("tufts" if middle else "bristles"))
        out["tufts" if middle else "bristles"].append(m.apply(R, (x, surface_y(x, z) + SINK, z)))
    return {k: S.merge(v) for k, v in out.items()}


def card(name, slots=None):
    """The prop card (docs/design.md: Prop card): the hold, the bristles' tips and the look points."""
    zm = float(np.mean(TUFT_ROWS))
    tip = surface_y(0.0, zm) - TUFT_H[1]
    w, t, _, _ = (float(v) for v in section(np.array([sum(HOLD) / 2])).ravel())
    hold = {"name": "hold", "type": "hold", "center": [0.0, 0.0, round(sum(HOLD) / 2, 4)], "axis": [0.0, 0.0, 1.0],
            "length": round(HOLD[1] - HOLD[0], 4), "radius": round(0.25 * (w + t), 4)}
    return {"size": [round(max(STATIONS, key=lambda s: s[1])[1], 4), round(-tip + 0.016, 4), LENGTH], "origin": "tail",
            "front": "-Y", "slots": dict(slots or {}),
            "use": {"grip": [hold],
                    "anchor": [{"name": "brush", "point": [0.0, round(tip, 4), round(zm, 4)], "dir": [0.0, -1.0, 0.0]}],
                    "look": [{"name": "head", "point": [0.0, round(surface_y(0.0, zm), 4), round(zm, 4)]},
                             {"name": "grip", "point": [0.0, 0.0, 0.17]}]},
            "objects": [f"{name}_body", f"{name}_bristles", f"{name}_tufts"]}
