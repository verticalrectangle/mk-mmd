"""Layout of the squeaky toy hammer (pure Python + numpy, no bpy: importable from tests): its meshes, the colour roles, the
squash and the card.

Frame (metres): the hammer stands on its handle's end along +Z, the end at z = 0 and the top of its head at z = LENGTH. The
head is a pleated bellows across the top, its axis along X, closed by a cap at each end (its two faces at x = +-HEAD_L / 2);
`head_mesh` is centred on the head (the builder puts it at HEAD_Z, so a squash scales it about its middle). Held in a fist
like the handheld mic (`attach` with `offset` and `attach_rot`: the handle across the palm, X along the fingers), a `bonk`
(docs/design.md: Moves) strikes with a face. The look points are `head` (the head's centre) and `grip` (where a fist
closes). A toy hammer in proportion: 320 mm long, the head 180 mm long and 96 mm across its caps, a handle of 23 to 26 mm.
No logo or lettering.

Colour roles (`slots` of the prop: a palette slot or a hex): head (love: the bellows), caps (gold), handle (gold).

The root property `squash` (0 .. 1, for [[key]]) shortens the head along its axis and swells it round (SQUASH): keyed up
on each hit and back, the toy squeaks in the picture."""
import math

import numpy as np

from ....core import shell as S

LENGTH = 0.32
HEAD_R, HEAD_L = 0.045, 0.18        # the bellows' radius; the head's length, cap to cap
CAP_R, CAP_T = 0.048, 0.016         # an end cap's radius and thickness
PLEATS, PLEAT_D = 6, 0.005          # the bellows' folds and how deep they go
HANDLE_R = (0.0115, 0.013)          # the handle's radius at its end and under the head
HEAD_Z = LENGTH - CAP_R             # the head's axis
GRIP_Z = 0.075                      # where a fist closes on the handle
SQUASH = (0.4, 0.18)                # at squash 1 the head is 40 % shorter and 18 % thicker
ROLES = {"head": "love", "caps": "gold", "handle": "gold"}
MATS = tuple(ROLES)                 # material index -> role


def _moved(m, f):
    """`m` with its vertices mapped by f ((n, 3) -> (n, 3), a rigid motion: the faces keep their winding)."""
    return S.Mesh(f(m.V), m.Q, m.T, m.Qm, m.Tm, m.UV, Ce=m.Ce, Cw=m.Cw)


def bellows_profile(per_pleat=12):
    """(r, z) of the bellows about its axis, axis to axis: folds of PLEAT_D between full rims, from just inside one cap
    to just inside the other."""
    z1 = HEAD_L / 2 - CAP_T + 0.002
    u = np.linspace(0.0, 1.0, PLEATS * per_pleat + 1)
    r = HEAD_R - 0.5 * PLEAT_D * (1.0 - np.cos(2.0 * math.pi * PLEATS * u))
    return np.vstack([[0.0, -z1], np.stack([r, -z1 + 2.0 * z1 * u], 1), [0.0, z1]])


def cap_profile(fillet=0.005):
    """(r, z) of an end cap, axis to axis: a disc CAP_T thick with rounded rims, from z = 0 up."""
    return S.fillet(np.array([(0.0, 0.0), (CAP_R, 0.0), (CAP_R, CAP_T), (0.0, CAP_T)]), [0.0, fillet, fillet, 0.0], 4)


def handle_profile(fillet=0.004):
    """(r, z) of the handle, axis to axis: a slight taper up from a rounded end into the head."""
    return S.fillet(np.array([(0.0, 0.0), (HANDLE_R[0], 0.0), (HANDLE_R[1], HEAD_Z), (0.0, HEAD_Z)]),
                    [0.0, fillet, 0.0, 0.0], 4)


def head_mesh(seg=48):
    """The head, centred on its middle with its axis along X: the bellows (material 0) and the two caps (1)."""
    bellows = S.lathe(bellows_profile(), seg=seg, mat=MATS.index("head"))
    cap = S.lathe(cap_profile(), seg=seg, mat=MATS.index("caps"))
    top = _moved(cap, lambda V: V + [0.0, 0.0, HEAD_L / 2 - CAP_T])
    bottom = _moved(cap, lambda V: V * [1.0, -1.0, -1.0] + [0.0, 0.0, -(HEAD_L / 2 - CAP_T)])  # a half turn about X
    along_z = S.merge([bellows, top, bottom])
    return _moved(along_z, lambda V: np.stack([V[:, 2], V[:, 1], -V[:, 0]], 1))      # a quarter turn about Y: Z -> X


def handle_mesh(seg=24):
    """The handle (material 2), from its end at the origin up into the head."""
    return S.lathe(handle_profile(), seg=seg, mat=MATS.index("handle"))


def squash(s):
    """(along, round): the head's scale along its axis and across it at squash `s`."""
    s = min(max(float(s), 0.0), 1.0)
    return 1.0 - SQUASH[0] * s, 1.0 + SQUASH[1] * s


def card(name, slots=None):
    """The prop card (docs/design.md: Prop card): the head and grip look points; `squash` its live root property."""
    looks = [{"name": "head", "point": [0.0, 0.0, round(HEAD_Z, 5)]}, {"name": "grip", "point": [0.0, 0.0, GRIP_Z]}]
    return {"size": [HEAD_L, round(2 * CAP_R, 4), LENGTH], "origin": "tail", "front": "+Z", "slots": dict(slots or {}),
            "use": {"look": looks}, "params": {"squash": 0.0}, "objects": [f"{name}_head", f"{name}_handle"]}
