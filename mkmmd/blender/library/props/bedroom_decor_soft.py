"""Soft, moulded forms of the bedroom decor props (pure numpy on `mkmmd.core.shell`, no bpy: importable from tests).

The `form` check calls a part a box when most of its surface faces along three orthogonal axes, bevelled or not, so the
furniture and the clock case are built from rolled forms instead of cuboids with small bevels:

    soft_slab    a plate or a block whose top and bottom rims are rolled over (radius up to ~ a quarter of the plan size,
                 a quarter circle in 4+ steps) over a plan outline with large corner radii, optionally crowned: the
                 carcass, the top slab, the shelf and the drawer front of the nightstand, the plinth of the clock
    clock_shell  the clock's case: its side profile (fillets of 3 / 9 / 11 / 5.5 mm) extruded across the width with both
                 ends rolled over (a pillow in the profile's shape, the ends shrink by `r_end` through a quarter circle)
    set_roles    material index per face from a function of the face centres and normals (the top face of a slab takes
                 another material than its rim)
"""
import math

import numpy as np

from ....core import shell as SH
from . import bedroom_decor_layout as LAY


def slab_profile(thickness, r_top, r_bot=None, n=4):
    """[(d, h)] for `offset_rings`: the outline moved d inward at height h from the bottom face's rim up a quarter-circle
    roll (radius r_bot), a straight wall, and over the top roll (radius r_top) to the top face. The radii shrink together
    when they do not fit in `thickness`."""
    r_bot = r_top if r_bot is None else r_bot
    s = min(1.0, 0.98 * thickness / max(r_top + r_bot, 1e-9))
    rt, rb = r_top * s, r_bot * s
    prof = [(rb * (1.0 - math.cos(p)), rb * (1.0 - math.sin(p))) for p in np.linspace(0.5 * math.pi, 0.0, n + 1)]
    prof += [(rt * (1.0 - math.cos(p)), thickness - rt + rt * math.sin(p)) for p in np.linspace(0.0, 0.5 * math.pi, n + 1)]
    out = [prof[0]]
    for q in prof[1:]:
        if abs(q[0] - out[-1][0]) > 1e-9 or abs(q[1] - out[-1][1]) > 1e-9:
            out.append(q)
    return out


def soft_slab(outline, thickness, r_top, r_bot=None, n=4, dome=0.0, cap_rings=2, mat=0):
    """A closed block with outward normals: the plan `outline` (m, 2) with both rims rolled over, bottom face at z = 0, top
    face at z = thickness (crowned up by `dome` in the middle)."""
    R = SH.offset_rings(outline, slab_profile(thickness, r_top, r_bot, n))
    return SH.cage(R, closed=True, caps=("fan", "fan"), cap_axes=((0, 0, -1), (0, 0, 1)), cap_bulge=(0.0, dome),
                   cap_rings=cap_rings, mat=mat)


def set_roles(mesh, fn):
    """Give every face the material index fn(centres (k, 3), unit normals (k, 3)) -> (k,) ints; returns the mesh."""
    V = mesh.V
    if len(mesh.Q):
        P = V[mesh.Q]
        n = np.cross(P[:, 2] - P[:, 0], P[:, 3] - P[:, 1])
        mesh.Qm = np.asarray(fn(P.mean(1), n / (np.linalg.norm(n, axis=1, keepdims=True) + 1e-12)), np.int32)
    if len(mesh.T):
        P = V[mesh.T]
        n = np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0])
        mesh.Tm = np.asarray(fn(P.mean(1), n / (np.linalg.norm(n, axis=1, keepdims=True) + 1e-12)), np.int32)
    return mesh


def clock_shell(r_end=None, n=5, seg=8):
    """The clock's case as a cage: rings along x, each the side profile (y, z) of `LAY.clock_profile()` scaled about its
    box centre so that its extreme points move in by d; the ends follow a quarter circle of radius `r_end`. Closed, outward
    normals; x from -CLOCK_W / 2 to +CLOCK_W / 2."""
    r = LAY.CLOCK_END_R if r_end is None else r_end
    prof = np.asarray(LAY.clock_profile(seg), float)
    lo, hi = prof.min(0), prof.max(0)
    c, half = (lo + hi) / 2.0, (hi - lo) / 2.0
    hw = LAY.CLOCK_W / 2.0
    ts = np.linspace(0.5 * math.pi, 0.0, n + 1)
    stations = [(-hw + r * (1.0 - math.sin(t)), r * (1.0 - math.cos(t))) for t in ts]
    stations += [(hw - r * (1.0 - math.sin(t)), r * (1.0 - math.cos(t))) for t in ts[::-1]]
    rings = []
    for x, d in stations:
        yz = c + (prof - c) * (1.0 - d / half)
        rings.append(np.column_stack([np.full(len(prof), x), yz]))
    return SH.cage(np.stack(rings), closed=True, caps=("fan", "fan"), cap_axes=((-1, 0, 0), (1, 0, 0)), cap_rings=2)
