"""Layout of the spike mallet (`card = "library:spike_mallet"`; pure Python + numpy, no bpy: importable from tests): its
meshes, the colour roles and the card.

Frame (metres): the mallet stands on the end of its handle along +Z, the end at z = 0; the head lies across the top, its
axis along X and its middle at (0, 0, HEAD_Z), 2 R_MID across, so the whole is LENGTH = HEAD_Z + R_MID tall. The +X end
(x = HEAD_L / 2) is the striking face; front -Y. A giant yokogine (the T-shaped pestle that pounds mochi in a mortar, the
moon rabbit's) that reads as a railroad spike maul too: a turned barrel of wood a little fuller at its middle than at its
ends (`barrel_r`), each face crowned CROWN and its rim rolled over (ROLL), an iron hoop shrunk on near each face (BAND_X,
BAND_W wide, standing BAND_T proud); a straight turned handle (`handle_r`) with a swell at its end (the knob the lower fist
rests on) and a slight flare into the head, passing through it: the tenon's end grain, split by its wedge, shows on the
top. No logo or lettering.

Colour roles (`slots` of the prop: a palette slot, `paper` / `ink`, a hex or a mix `a:b:t` of two such): wood (the head's
turned side), grain (the end grain of the faces and of the tenon), band (the iron hoops), handle.

Card: `use.grip` `hold` (type "hold": the handle a fist may close on, from just above the knob up to HOLD[1], `axis` +Z
toward the head, `radius` and the `profile` [[s, r], ...] from the head end), `use.anchor` `face` (the middle of the
striking face, `dir` +X: what meets a stake or the mochi), `use.look` `face`, `head` and `grip`."""
import numpy as np

from ....core import shell as S

HEAD_Z = 1.12                       # the head's axis
HEAD_L = 0.46                       # face to face (the crowns' middles)
R_MID, R_END = 0.120, 0.1125        # the barrel's radius at its middle and at its rims' plane
CROWN = 0.0025                      # a face's dome: its middle this far out of its rim's plane
ROLL = 0.014                        # the rolled rim's radius
BAND_X = 0.185                      # a hoop's middle, from the head's middle (at both ends)
BAND_W, BAND_T = 0.030, 0.0045      # a hoop's width and how far it stands proud of the wood
BAND_SINK = 0.002                   # its inside lies this far into the wood (no sliver of gap shows)
BAND_EDGE = 0.0016                  # the radius its outer edges are rounded to
LENGTH = HEAD_Z + R_MID             # the end of the handle to the top of the head: 1.24
SHAFT_R = 0.0208                    # the handle's radius along its straight shaft
SWELL = (0.0055, 0.026, 0.022)      # the knob at its end: a bump this high (m) round z, this wide (a gaussian)
FLARE = (0.0037, 0.78, 1.0)         # the flare into the head: this much wider, from z0 to z1 (a smoothstep)
END_H = 0.007                       # the handle's end is rounded over this height (an elliptic dome)
TENON_PROUD = 0.0006                # the tenon's end stands this far proud of the barrel's top
HOLD = (0.08, 0.72)                 # the span a fist may hold: from just above the knob up the handle (z)
PROFILE_SAMPLES = 9                 # (s, r) samples of the span in the hold's `profile`
SEG = 72                            # lathe segments round the head and the hoops
HANDLE_SEG = 40                     # ... round the handle
ROLES = {"wood": "gold:ink:0.45", "grain": "gold:ink:0.6", "band": "ink", "handle": "gold:ink:0.3"}
MATS = ("head", "band", "handle")   # material index -> material: the head's mixes wood and grain by the way its surface
                                    # faces, the handle's shows grain on the tenon's end
NDIGITS = 5


def _r(x):
    return round(float(x), NDIGITS)


def _unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


def barrel_r(x):
    """The head's radius at x along its axis (from its middle; up to the rims' plane at HEAD_L / 2 - CROWN): a barrel
    R_MID at the middle and R_END at the rims."""
    x = np.asarray(x, float)
    return R_MID - (R_MID - R_END) * (x / (HEAD_L / 2 - CROWN)) ** 2


def barrel_top(x, y):
    """z of the head's turned surface over (x, y) (|y| < its radius there): the top of the barrel."""
    return HEAD_Z + np.sqrt(np.maximum(barrel_r(x) ** 2 - np.asarray(y, float) ** 2, 0.0))


def handle_r(z):
    """The handle's radius at heights z (m, from its end): the straight shaft, the swell at the end, the flare into the
    head, the end rounded over END_H."""
    z = np.asarray(z, float)
    a, zc, w = SWELL
    fa, f0, f1 = FLARE
    u = np.clip((z - f0) / (f1 - f0), 0.0, 1.0)
    r = SHAFT_R + a * np.exp(-((z - zc) / w) ** 2) + fa * u * u * (3.0 - 2.0 * u)
    e = np.clip(z / END_H, 0.0, 1.0)
    return r * np.sqrt(1.0 - (1.0 - e) ** 2)


# ================================================================================================== the head
def _corner(K, d_in, d_out, radius, n):
    """The arc rounding the corner K between a line that comes in along the unit d_in and one that leaves along d_out."""
    return S.fillet(np.array([K - d_in, K, K + d_out]), [0.0, radius, 0.0], n)[1:-1]


def head_profile(n_face=10, n_roll=8, n_side=30):
    """(r, a) of the head about its axis (a along it), from the middle of the -X face to the middle of the +X face: the
    crowned face, its rolled rim, the barrel, the other rim and face."""
    A = HEAD_L / 2
    xr = A - CROWN
    K = np.array([R_END, -xr])                                         # the corner the roll rounds (-X end)
    d_in = _unit([1.0, 2.0 * CROWN / R_END])                           # the crown's tangent at its rim, outward
    d_out = _unit([2.0 * (R_MID - R_END) / xr, 1.0])                   # the barrel's tangent there, inward
    arc = _corner(K, d_in, d_out, ROLL, n_roll)
    rho = np.linspace(0.0, arc[0, 0], n_face + 1)[:-1]
    face = np.stack([rho, -A + CROWN * (rho / R_END) ** 2], 1)
    a = np.linspace(arc[-1, 1], 0.0, n_side + 1)[1:]
    side = np.stack([barrel_r(a), a], 1)
    half = np.vstack([face, arc, side])                                # the -X apex .. the middle
    return np.vstack([half, (half[::-1] * [1.0, -1.0])[1:]])


def _along_x(m, t=(0.0, 0.0, 0.0)):
    """A mesh turned on a lathe about Z laid along X (a quarter turn about Y: Z -> X), then moved by t."""
    return m.apply(S.rot_matrix(0.0, 90.0, 0.0), t)


def head_mesh(seg=SEG):
    """The head (material head), centred on its middle with its axis along X: the builder puts it at (0, 0, HEAD_Z)."""
    return _along_x(S.lathe(head_profile(), seg=seg, mat=MATS.index("head")))


def band_profile(c, n_edge=4):
    """(r, a) of the hoop round a = c on the head's axis, a closed loop: its inside BAND_SINK into the wood, its outside
    BAND_T over it (following the barrel), the outer edges rounded."""
    a = np.linspace(c - BAND_W / 2, c + BAND_W / 2, 9)
    outer = barrel_r(a) + BAND_T
    r_in = float(barrel_r(a).min()) - BAND_SINK
    loop = np.vstack([[[r_in, a[0]]], np.stack([outer, a], 1), [[r_in, a[-1]]]])
    radius = np.zeros(len(loop))
    radius[[1, -2]] = BAND_EDGE
    pts = S.fillet(loop, radius, n_edge, closed=True)
    return np.vstack([pts, pts[:1]])


def band_mesh(seg=SEG):
    """The two hoops (material band), in the head's frame like `head_mesh`."""
    hoops = [S.lathe(band_profile(c), seg=seg, mat=MATS.index("band")).weld() for c in (-BAND_X, BAND_X)]
    return _along_x(S.merge(hoops))


# ================================================================================================== the handle
def handle_profile():
    """(r, z) of the handle about +Z, from its end at the origin up through the head to the top: the rounded end, the
    swell, the shaft, the flare, the tenon (its end ring and middle are laid on the barrel's top by `handle_mesh`)."""
    z = np.concatenate([np.linspace(0.0, END_H, 8), np.linspace(END_H, 0.12, 30)[1:], np.arange(0.14, 0.78, 0.02),
                        np.linspace(0.78, 1.0, 12), [1.03, HEAD_Z + 0.06, HEAD_Z + 0.10]])
    r = handle_r(z)
    rt = float(handle_r(LENGTH))
    top = [(rt, LENGTH - 0.004), (0.5 * rt, LENGTH - 0.002), (0.0, LENGTH)]
    return np.vstack([np.stack([r, z], 1), top])


def handle_mesh(seg=HANDLE_SEG):
    """The handle (material handle), from its end at the origin through the head; the tenon's end follows the barrel's
    top, TENON_PROUD over it."""
    m = S.lathe(handle_profile(), seg=seg, mat=MATS.index("handle"))
    top = m.V[:, 2] > LENGTH - 0.005
    x, y = m.V[top, 0], m.V[top, 1]
    m.V[top, 2] = barrel_top(x, y) + TENON_PROUD
    return m


def meshes():
    """{part: (Mesh, location)}: the head and the hoops about the head's middle, the handle from the origin."""
    return {"head": (head_mesh(), (0.0, 0.0, HEAD_Z)), "bands": (band_mesh(), (0.0, 0.0, HEAD_Z)),
            "handle": (handle_mesh(), (0.0, 0.0, 0.0))}


# ================================================================================================== the card
def hold_profile():
    """[[s, r], ...]: PROFILE_SAMPLES samples of the handle's radius along the hold, s metres from its head end."""
    s = np.linspace(0.0, HOLD[1] - HOLD[0], PROFILE_SAMPLES)
    return [[_r(v), _r(handle_r(HOLD[1] - v))] for v in s]


def hold_grip():
    """The `use.grip` entry `hold` (type "hold"): the handle a fist closes round, its axis toward the head."""
    length = HOLD[1] - HOLD[0]
    centre = HOLD[0] + 0.5 * length
    return {"name": "hold", "type": "hold", "center": [0.0, 0.0, _r(centre)], "axis": [0.0, 0.0, 1.0],
            "length": _r(length), "radius": _r(handle_r(centre)), "profile": hold_profile()}


def face_point():
    """The middle of the striking face (the +X face's crown)."""
    return [_r(HEAD_L / 2), 0.0, _r(HEAD_Z)]


def card(name, roles):
    """The prop card (docs/design.md: Prop card): the hold, the striking face, the look points. `roles` {role: spec}."""
    hold = hold_grip()
    face = face_point()
    use = {"grip": [hold],
           "anchor": [{"name": "face", "point": face, "dir": [1.0, 0.0, 0.0]}],
           "look": [{"name": "face", "point": list(face)}, {"name": "head", "point": [0.0, 0.0, _r(HEAD_Z)]},
                    {"name": "grip", "point": list(hold["center"])}]}
    return {"size": [_r(HEAD_L), _r(2.0 * R_MID), _r(LENGTH)], "origin": "tail", "front": "-Y", "slots": dict(roles),
            "use": use, "colliders": [], "objects": [f"{name}_{part}" for part in ("head", "bands", "handle")]}
