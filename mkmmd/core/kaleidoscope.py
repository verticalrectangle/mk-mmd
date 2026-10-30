"""Layout of the cardboard kaleidoscope toy (`card = "library:kaleidoscope"`), bpy-free: its numbers, its meshes in the
prop's frame, the beads in its object chamber and the card. The Blender half is
mkmmd/blender/library/props/kaleidoscope*.py; the printed paper is mkmmd/core/kaleidoscope_print.py.

The toy: a cardboard tube wrapped in printed paper, LENGTH (230 mm) long and 49 mm across, a turned rim at each end. The
eyepiece end is a turned cap with a shallow dish, a brass eyelet round a small round eyehole (EYE_R) and a domed lens in
it; behind the lens a short dark tunnel ends on the view (a disc showing the mirrored beads). The far end is the object
chamber: a frosted glass disc in the far rim's lip, coloured beads and glass bits between it and a clear inner glass.

Frame (metres): the toy LIES ON ITS SIDE on its two rims (their beads are its widest part), its axis along +Y at
height AXIS_Z (the rims' reach), the eyepiece toward -Y (its front: `front` "-Y"), the far end toward +Y, the origin on the floor
under its middle (`origin` "floor_center"), so `at` on a table top or `place` lays it down there. Along the axis, `s`
runs from the eyepiece face (s = 0, y = -LENGTH / 2) to the far face (s = LENGTH). The parts are lathes about the axis,
built along +Z in the lathe frame (z = s) and turned into the prop frame by `to_prop` (lathe +Z -> +Y, lathe +Y -> -Z:
lathe angle 90 degrees is the bottom, where the paper's seam is and where the beads have settled).

Use points (the card): `look` `eyepiece` (the eyehole's centre, the lens's apex: where a camera aims), `chamber` (the
far face's centre) and `grip` (the tube's middle, on the axis: where a fist closes, for `attach`); `anchor` `eyehole` (the
eyepiece point, `dir` out of the eyepiece, `radius` EYE_OPEN, the eyelet's opening, `object` <name>_eyehole: an empty
there, so an insert's anchor is `obj("<name>_eyehole").loc`); `rest` `side` (an edge along the top of the tube between
the rims, normal +Z: a hand or a small prop rests on the toy lying down); `grip` `tube` (a cylinder wrap: centre, axis,
radius, height of the paper between the rims, as the mug's `body`). Root property `turn` (degrees, for [[key]]): the far
rim and the chamber turn about the axis by it, and the view in the eyehole with them."""
import math

import numpy as np

from . import shell as S

LENGTH = 0.230                  # the eyepiece face to the far face
TUBE_R = 0.0245                 # the paper-wrapped tube's outer radius (49 mm across)
WALL = 0.0018                   # the cardboard's wall
PAPER_T = 0.00012               # the paper's thickness: the step at the seam where its edge overlaps
RIM_R = 0.0268                  # the turned rims' widest bead (before its rounding: AXIS_Z is what they reach)
EYE_R = 0.0040                  # the eyehole's bore
EYE_OPEN = EYE_R - 0.0004       # the opening the eyelet leaves (what one looks through)
EYE_S = 0.0012                  # the dish of the eyepiece face: its depth at the eyehole (the lip is at s = 0)
PLATE = 0.0045                  # the eyepiece cap's thickness: the tube's end sits on it
COLLAR_EYE = 0.0160             # the eyepiece rim's length (its collar meets the paper there)
FAR_RIM = 0.0260                # the far rim's length
BORE_R = 0.0215                 # the far rim's bore: the frosted window's radius and the chamber's
FROST_S = LENGTH - 0.0015       # the frosted glass's outer face
FROST_T = 0.0016                # and its thickness
INNER_T = 0.0010                # the clear inner glass's thickness
CHAMBER = (LENGTH - 0.0115, FROST_S - FROST_T)   # s of the inner glass's outer face .. the frosted glass's inner face
LENS_R, LENS_SAG, LENS_T = 0.0037, 0.0008, 0.0010    # the eyepiece lens: radius, dome height (a bubble: it catches a
                                                      # glint from a wide spread of lights), edge thickness
LENS_S = EYE_S + 0.0003         # the lens's rim (its dome's apex is LENS_SAG nearer the face)
EYE_POINT_S = LENS_S - LENS_SAG     # the eyehole's centre: the lens's apex
VIEW_S = PLATE + 0.0105         # the view disc at the end of the tunnel
VIEW_R = 0.0058                 # the tunnel's radius and the view's
PAPER_S = (0.0140, LENGTH - 0.0240)   # the print's span along the axis (a little of it under each collar)
SEAM_DEG = 90.0                 # lathe angle of the paper's seam: the bottom of the toy lying down
SEG = 128                       # segments round the lathes (a 0.3 px chord sag at a frame-filling macro)
GRIP_S = 0.5 * (COLLAR_EYE + LENGTH - FAR_RIM)    # the middle of the paper: where a fist closes

ROLES = {"rim": "love", "face": "gold", "ferrule": "gold", "frost": "surface", "inside": "text"}
BEADS = ("love", "gold", "rose", "pine", "foam", "iris")         # the beads' and glass bits' palette slots
DEFAULTS = {"pattern": "roses", "seed": 1, "wear": 0.6, "beads": BEADS, "glow": 1.2, "turn": 0.0, "print": {}}
MATS = ("paper", "paper_edge", "card", "rim", "face", "ferrule", "lens", "tunnel", "view", "frost", "glass")
BEAD_MAT, BIT_MAT = 100, 200    # chamber pieces: material BEAD_MAT + i (opaque beads) or BIT_MAT + i (glass) of colour i
FACE_R = 0.0237                # the eyepiece cap's painted face: the dish inside the lip


def m(role):
    """The material index of a role (MATS)."""
    return MATS.index(role)


def options(slots, patterns=None):
    """The prop's non-colour options from its `slots`, defaults filled in and checked (ValueError naming the option):
    pattern (one of `patterns`, the print's), seed (an integer), wear (0 .. 1: how far the print is rubbed through where
    the paper wears), beads (palette slots or "#hex", a list or "a,b,c"), glow (the view's emission, >= 0), turn
    (degrees), print ({role: slot or "#hex"}: the print's own colour roles)."""
    o = {k: slots.get(k, v) for k, v in DEFAULTS.items()}
    if patterns is not None and o["pattern"] not in patterns:
        raise ValueError(f"kaleidoscope: pattern = {o['pattern']!r}, expected one of {tuple(patterns)}")
    try:
        o["seed"], o["wear"], o["glow"], o["turn"] = int(o["seed"]), float(o["wear"]), float(o["glow"]), float(o["turn"])
    except (TypeError, ValueError):
        raise ValueError("kaleidoscope: seed is an integer, wear, glow and turn are numbers") from None
    if not 0.0 <= o["wear"] <= 1.0 or o["glow"] < 0.0:
        raise ValueError(f"kaleidoscope: wear = {o['wear']:g} (0 .. 1), glow = {o['glow']:g} (>= 0)")
    beads = o["beads"].split(",") if isinstance(o["beads"], str) else list(o["beads"])
    o["beads"] = tuple(str(b).strip() for b in beads if str(b).strip())
    if not o["beads"]:
        raise ValueError("kaleidoscope: beads names no colour (palette slots or \"#hex\", a list or \"a,b,c\")")
    if not isinstance(o["print"], dict):
        raise ValueError(f"kaleidoscope: print = {o['print']!r}, expected {{role = slot or \"#hex\"}}")
    return o


# ================================================================================================= frames
ROT = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]])     # lathe -> prop: +Z -> +Y, +Y -> -Z
AXIS = ROT @ np.array([0.0, 0.0, 1.0])      # the axis's direction in the prop frame, eyepiece -> far end (+Y)


def to_prop(mesh):
    """A mesh built in the lathe frame (the axis +Z, z = s from the eyepiece face) in the prop's frame (lying on its side,
    the axis along +Y at AXIS_Z, the eyepiece at y = -LENGTH / 2)."""
    return mesh.apply(ROT, (0.0, -0.5 * LENGTH, AXIS_Z))


def point(s, r=0.0, deg=0.0):
    """The prop-frame point at `s` along the axis, `r` from it at lathe angle `deg` (90: straight down, -90: the top)."""
    a = math.radians(deg)
    return ROT @ np.array([r * math.cos(a), r * math.sin(a), s]) + np.array([0.0, -0.5 * LENGTH, AXIS_Z])


# ================================================================================================= profiles
def _loop(pts, fillets, n=3):
    """A closed (r, s) loop, corners filleted; walked counter-clockwise (r right, s up), `S.lathe` faces it outward."""
    return S.fillet(np.asarray(pts, float), fillets, n, closed=True)


def eyepiece_rim_profile():
    """The eyepiece cap and rim as one turned piece, (r, s) counter-clockwise: the face dishing in from its lip to the
    eyehole, the lip roll, a groove, the main bead, a step down to the collar that meets the paper; inside, the socket
    the tube sits in (r = TUBE_R down to the plate) and the plate's back to the eyehole's bore."""
    pts = [(EYE_R, EYE_S),                       # the eyehole's edge on the face
           (0.0200, 0.0005),                     # the dish rising to the lip
           (0.0238, 0.0000), (0.0263, 0.0000),   # the lip's flat
           (0.0265, 0.0028),                     # the lip roll
           (0.0255, 0.0044),                     # a groove
           (RIM_R, 0.0082),                      # the main bead
           (0.0263, 0.0118),
           (0.0253, 0.0136), (0.0251, COLLAR_EYE),   # the step down to the collar
           (TUBE_R, COLLAR_EYE),                 # the collar's top, at the paper
           (TUBE_R, PLATE),                      # the socket wall down to the plate
           (EYE_R, PLATE)]                       # the plate's back, to the bore
    fil = [0.0003, 0.002, 0.0008, 0.0010, 0.0012, 0.0010, 0.0035, 0.0012, 0.0008, 0.0004, 0.0, 0.0, 0.0003]
    return _loop(pts, fil)


def far_rim_profile():
    """The far rim, (r, s) counter-clockwise: the collar at the paper, a step up, the main bead, a groove, the lip roll,
    the lip's face rolling into the bore (the frosted window, BORE_R) down past the chamber to the socket the tube's end
    sits in."""
    s0 = LENGTH - FAR_RIM
    floor = CHAMBER[0] - INNER_T
    pts = [(TUBE_R, s0), (0.0251, s0),          # the collar's bottom, at the paper
           (0.0254, s0 + 0.0026),                # the step up
           (0.0263, s0 + 0.0062),
           (RIM_R, s0 + 0.0102),                 # the main bead
           (0.0262, LENGTH - 0.0110),
           (0.0255, LENGTH - 0.0082),            # a groove
           (0.0266, LENGTH - 0.0048),            # the lip roll
           (0.0262, LENGTH),                     # the lip's outer corner
           (0.0228, LENGTH),                     # the lip's face
           (BORE_R, LENGTH - 0.0010),            # rolling into the bore
           (BORE_R, floor),                      # the bore, down past the inner glass
           (TUBE_R, floor)]                      # the socket's floor (the tube's end)
    fil = [0.0, 0.0004, 0.0008, 0.0012, 0.0035, 0.0012, 0.0010, 0.0012, 0.0010, 0.0008, 0.0008, 0.0002, 0.0]
    return _loop(pts, fil)


AXIS_Z = round(float(max(eyepiece_rim_profile()[:, 0].max(), far_rim_profile()[:, 0].max())), 5)   # the rims' reach: the toy
                                # lies on them, so its axis is this high over the floor


def ferrule_profile():
    """The brass eyelet: a flange on the face round the eyehole and a barrel lining the bore, (r, s) counter-clockwise."""
    pts = [(EYE_OPEN, 0.0003),                   # the rolled inner lip, proud of the face
           (EYE_R + 0.0019, EYE_S - 0.0005),     # the flange's top
           (EYE_R + 0.0022, EYE_S + 0.0001),     # its edge, sunk into the face
           (EYE_R + 0.0001, EYE_S + 0.0001),
           (EYE_R + 0.0001, PLATE - 0.0004),     # the barrel lining the bore
           (EYE_OPEN, PLATE - 0.0004)]
    return _loop(pts, [0.0004, 0.0005, 0.0002, 0.0001, 0.0001, 0.0001])


def lens_profile(n=10):
    """The eyepiece lens, (r, s) from its apex on the axis out over its dome (facing out of the eyepiece), up its edge
    and in along its flat back: plano-convex."""
    r = np.linspace(0.0, LENS_R, n)
    dome = LENS_S - LENS_SAG * (1.0 - (r / LENS_R) ** 2)
    return np.vstack([np.stack([r, dome], 1), [(LENS_R, LENS_S + LENS_T), (0.0, LENS_S + LENS_T)]])


def disc_profile(r, s0, t, edge=0.0004):
    """A round disc `t` thick from s0 to s0 + t with rolled edges, (r, s) axis to axis."""
    return S.fillet(np.array([(0.0, s0), (r, s0), (r, s0 + t), (0.0, s0 + t)]), [0.0, edge, edge, 0.0], 3)


def tunnel_profile(t=0.0006):
    """The dark sleeve behind the eyehole (the mirrors' mouth), (r, s) counter-clockwise: a thin tube from the plate to
    the view disc that closes it."""
    return np.array([(VIEW_R, PLATE - 0.0001), (VIEW_R + t, PLATE - 0.0001), (VIEW_R + t, VIEW_S + t), (VIEW_R, VIEW_S + t)])


# ================================================================================================= meshes
def _lathe(profile, mat, seg=SEG, closed=True):
    P = np.asarray(profile, float)
    if closed:
        P = np.vstack([P, P[:1]])
    mesh = S.lathe(P, seg=seg, mat=mat)
    return mesh.weld(1e-9) if closed else mesh


def tube_mesh(seg=SEG, rings=12):
    """The cardboard tube with its paper, in the lathe frame: the paper's surface winds out by PAPER_T once round (a
    spiral), so where its edge overlaps the start there is a step, the seam (`paper_edge`), at SEAM_DEG; inside it the
    bare card. From the eyepiece cap's plate to the far rim's socket floor. UV, the print seen from outside as drawn:
    U = 1 - u (u round the tube from the seam, 0 .. 1), V = 1 at the print's eyepiece end down to 0 at its far end
    (PAPER_S; the parts under the collars run past 0 and 1), so an image whose top row is the eyepiece end reads
    unmirrored. The seam column is in it twice (u 0 and 1): the surface is closed by the two seam faces, not welded."""
    s0, s1 = PLATE, CHAMBER[0] - INNER_T
    ss = np.concatenate([[s0], np.linspace(PAPER_S[0], PAPER_S[1], rings), [s1]])
    u = np.arange(seg + 1) / seg
    a = math.radians(SEAM_DEG) + 2.0 * math.pi * u
    nr, nc = len(ss), seg + 1
    vv = 1.0 - (ss - PAPER_S[0]) / (PAPER_S[1] - PAPER_S[0])

    def grid(radius):
        V = np.stack([np.broadcast_to(radius * np.cos(a), (nr, nc)), np.broadcast_to(radius * np.sin(a), (nr, nc)),
                      np.broadcast_to(ss[:, None], (nr, nc))], -1).reshape(-1, 3)
        UV = np.stack([np.broadcast_to(1.0 - u, (nr, nc)), np.broadcast_to(vv[:, None], (nr, nc))], -1).reshape(-1, 2)
        return V, UV

    Vo, UVo = grid(TUBE_R - PAPER_T + PAPER_T * u)                  # the paper winds out to TUBE_R
    Vi, UVi = grid(np.full(nc, TUBE_R - WALL))
    n0 = nr * nc
    i, j = np.meshgrid(np.arange(nr - 1), np.arange(nc - 1), indexing="ij")
    k = (i * nc + j).ravel()
    outer = np.stack([k, k + 1, k + nc + 1, k + nc], 1)              # round counter-clockwise, then up: outward
    inner = (outer + n0)[:, ::-1]                                   # the inside faces the axis
    jj = np.arange(nc - 1)
    top = (nr - 1) * nc
    ends = [np.stack([jj + 1, jj, n0 + jj, n0 + jj + 1], 1),
            np.stack([top + jj, top + jj + 1, n0 + top + jj + 1, n0 + top + jj], 1)]
    ii = np.arange(nr - 1) * nc
    start = np.stack([ii, ii + nc, n0 + ii + nc, n0 + ii], 1)                           # where the wrap begins
    edge = np.stack([ii + nc + seg, ii + seg, n0 + ii + seg, n0 + ii + nc + seg], 1)    # the overlapping edge: the seam
    Q = np.concatenate([outer, inner] + ends + [start, edge])
    Qm = np.concatenate([np.full(len(outer), m("paper")),
                         np.full(len(inner) + 2 * len(jj) + len(ii), m("card")), np.full(len(ii), m("paper_edge"))])
    return S.Mesh(np.concatenate([Vo, Vi]), Q, None, Qm, None, np.concatenate([UVo, UVi]))


def tube_normals(mesh):
    """Per-corner normals of `tube_mesh` (the faces' corners in Q order, (len(Q) * 4, 3)): the analytic radial normal on
    the paper (outward) and the card's inside (inward), each end and seam face its own flat normal, so the step of the
    seam is a crisp line and the paper round it is smooth."""
    corners = mesh.V[mesh.Q]                                        # (q, 4, 3)
    radial = corners.copy()
    radial[..., 2] = 0.0
    radial /= np.maximum(np.linalg.norm(radial, axis=-1, keepdims=True), 1e-12)
    nq, _ = mesh.face_vectors()
    flat = nq / np.maximum(np.linalg.norm(nq, axis=1, keepdims=True), 1e-12)
    out = np.broadcast_to(flat[:, None, :], corners.shape).copy()
    paper = mesh.Qm == m("paper")
    card_in = (mesh.Qm == m("card")) & (np.einsum("ij,ij->i", flat, radial.mean(1)) < -0.9)
    out[paper] = radial[paper]
    out[card_in] = -radial[card_in]
    return out.reshape(-1, 3)


def eyepiece_meshes():
    """{part: mesh} of the eyepiece end in the lathe frame: the turned cap (its dish inside the lip, FACE_R, painted
    `face`), the eyelet, the lens, the tunnel and the view (uv: its picture centred on the axis)."""
    cap = _lathe(eyepiece_rim_profile(), m("rim"))
    cap.assign(m("face"), lambda c, n: (np.hypot(c[:, 0], c[:, 1]) < FACE_R) & (c[:, 2] < EYE_S + 0.0002) & (n[:, 2] < -0.5))
    view = S.lathe(disc_profile(VIEW_R + 0.0003, VIEW_S, 0.0006, edge=0.0001), seg=96, mat=m("view"))
    view.UV = view.V[:, :2] / (2.0 * VIEW_R) + 0.5
    return {"cap": cap, "ferrule": _lathe(ferrule_profile(), m("ferrule"), seg=96),
            "lens": S.lathe(lens_profile(), seg=96, mat=m("lens")), "tunnel": _lathe(tunnel_profile(), m("tunnel"), seg=64),
            "view": view}


def far_meshes():
    """{part: mesh} of the far end in the lathe frame: the turned rim, the window (frosted on its outer face, clear on its
    inner one, so it veils the chamber once), the clear inner glass and the mirrors' dark end behind it (what the beads
    are seen against through the window)."""
    frost = S.lathe(disc_profile(BORE_R + 0.0001, FROST_S - FROST_T, FROST_T), seg=SEG, mat=m("frost"))
    frost.assign(m("glass"), lambda c, n: n[:, 2] < -0.5)
    return {"rim": _lathe(far_rim_profile(), m("rim")), "frost": frost,
            "glass": S.lathe(disc_profile(BORE_R + 0.0001, CHAMBER[0] - INNER_T, INNER_T), seg=SEG, mat=m("glass")),
            "mirrors": S.lathe(disc_profile(TUBE_R - WALL, CHAMBER[0] - INNER_T - 0.0012, 0.0010), seg=SEG,
                               mat=m("tunnel"))}


# ================================================================================================= the chamber
KINDS = ("bead", "seed", "bit", "rod")
FILL = 0.62                     # the share of the chamber's height the settled pile reaches
ROD_R = 0.00055                 # a glass rod's radius


def _half_s(kind, size, thick, R):
    """How far a piece reaches along the axis (s) from its centre when turned by R (3 x 3)."""
    nz = abs(float(R[2, 2]))                                        # its own z (a disc's normal, a rod's length) along s
    side = math.sqrt(max(0.0, 1.0 - nz * nz))
    if kind == "bead":
        return size
    if kind == "seed":
        return thick + (size - thick) * side
    if kind == "rod":
        return (size - ROD_R) * nz + ROD_R
    return thick * nz + size * side


def chamber_layout(seed=1, counts=(("bit", 34), ("bead", 26), ("rod", 14), ("seed", 26)), strays=3):
    """The beads and glass bits in the chamber, settled to its bottom (the toy lying on its side): a list of
    {kind, centre (lathe frame: x, y with +y down, s), size (its reach across the axis), half (its reach along it), thick,
    turn (XYZ Euler degrees), ink (an integer: the colour is BEADS[ink % n])}. Deterministic for a seed. Kinds: round
    beads, seed beads (rings), flat glass bits and short glass rods, `counts` of each in a shuffled order: each is dropped
    at a random place across the chamber and comes to rest on the bore or on what lies under it, and is kept while the
    pile is under FILL of the chamber's height (one that finds no room in 60 drops is left out); then `strays` small
    pieces caught higher up. Pieces are kept apart by their bounding cylinders about the axis (radius `size`, half-height
    `half`), so flat ones lying at different depths overlap across the window as they do in a real chamber and none goes
    through another, the bore or either glass. Flat pieces lie nearly flat against the glass, as in a thin chamber."""
    rng = np.random.default_rng([int(seed), 47])
    lo, hi = CHAMBER[0] + 0.0002, CHAMBER[1] - 0.0002
    sizes = {"bead": (0.0014, 0.0022), "seed": (0.0011, 0.0015), "bit": (0.0019, 0.0033), "rod": (0.0020, 0.0028)}
    todo = [(k, float(rng.uniform(*sizes[k]))) for k, n in counts for _ in range(n)]
    todo = [todo[i] for i in rng.permutation(len(todo))]
    todo += [("seed", float(rng.uniform(*sizes["seed"]))) for _ in range(int(strays))]
    out, C, R, H = [], np.zeros((0, 3)), np.zeros(0), np.zeros(0)
    top = BORE_R - 2.0 * FILL * BORE_R                              # the pile's top (y, +y down)
    gap = 0.0002
    for n, (kind, size) in enumerate(todo):
        stray = n >= len(todo) - int(strays)
        thick = {"bead": size, "seed": 0.36 * size, "bit": 0.0007, "rod": ROD_R}[kind]
        tilt = 360.0 if kind in ("bead", "seed") else 18.0
        reach = BORE_R - size - 0.0003
        for _ in range(60):
            turn = [float(rng.uniform(-tilt, tilt)), float(rng.uniform(-tilt, tilt)), float(rng.uniform(0.0, 360.0))]
            if kind == "rod":
                turn[0] += 90.0                                     # a rod lies across the chamber, not along the axis
            half = _half_s(kind, size, thick, S.rot_matrix(*turn))
            if lo + half > hi - half:
                continue
            x, s = float(rng.uniform(-reach, reach)), float(rng.uniform(lo + half, hi - half))
            near = np.abs(C[:, 2] - s) < H + half + gap             # only pieces at its depth can stop it
            Cn, Rn = C[near], R[near]
            if stray:                                               # caught somewhere above the pile
                floor = math.sqrt(max(0.0, reach * reach - x * x))
                y = float(rng.uniform(-floor, max(-floor, min(top, floor))))
                if np.any(np.hypot(x - Cn[:, 0], y - Cn[:, 1]) < Rn + size + gap):
                    continue
            else:                                                   # falling from the top, then rolling down
                y = float(_rest(np.array([x]), -math.sqrt(max(0.0, reach * reach - x * x)), reach, Cn, Rn, size + gap)[0])
                for _ in range(4 if np.isfinite(y) else 0):
                    xs = x + size * np.linspace(-1.0, 1.0, 9)
                    ys = _rest(xs, y - 0.0002, reach, Cn, Rn, size + gap)
                    if not np.isfinite(ys).any() or np.nanmax(ys) <= y + 1e-5:
                        break
                    k = int(np.nanargmax(ys))
                    x, y = float(xs[k]), float(ys[k])
                if not np.isfinite(y) or y < top:
                    continue                                        # no room, or the pile is full there
            c = np.array([x, y, s])
            out.append({"kind": kind, "centre": c, "size": size, "half": float(half), "thick": float(thick),
                        "turn": turn, "ink": int(rng.integers(0, 1 << 16))})
            C, R, H = np.vstack([C, c]), np.append(R, size), np.append(H, half)
            break
    return out


def _rest(xs, y0, reach, C, R, clear):
    """Where a piece let go at (x, y0) stops falling (+y), for each x of `xs` (nan where it cannot be at y0): on the bore
    (its centre within `reach` of the axis) or where its centre first comes within `clear` + R of a piece of `C`
    (centres) / `R` (radii), the upper point of that circle over x."""
    xs = np.asarray(xs, float)
    floor = np.sqrt(np.maximum(0.0, reach * reach - xs * xs))
    dx = xs[:, None] - C[None, :, 0]
    rr = (R[None, :] + clear) ** 2 - dx * dx
    hit = rr > 0.0
    h = np.sqrt(np.where(hit, rr, 0.0))
    upper, lower = np.where(hit, C[None, :, 1] - h, np.inf), np.where(hit, C[None, :, 1] + h, -np.inf)
    inside = np.any((y0 > upper) & (y0 < lower), axis=1)             # let go inside a piece
    y = np.minimum(floor, np.where(upper >= y0, upper, np.inf).min(axis=1, initial=np.inf))
    return np.where(inside | (np.abs(xs) > reach) | (y0 < -floor) | (y < y0), np.nan, y)


def bit_outline(rng, size):
    """A glass bit's outline: an irregular convex polygon about its centre within `size`, counter-clockwise, (k, 2): the
    hull of 5 to 7 points at jittered even angles."""
    n = int(rng.integers(5, 8))
    a = 2.0 * math.pi * (np.arange(n) + rng.uniform(-0.3, 0.3, n)) / n
    r = size * rng.uniform(0.62, 1.0, n)
    pts = sorted(zip((r * np.cos(a)).tolist(), (r * np.sin(a)).tolist()))

    def chain(points):
        out = []
        for p in points:
            while len(out) >= 2 and ((out[-1][0] - out[-2][0]) * (p[1] - out[-2][1])
                                     - (out[-1][1] - out[-2][1]) * (p[0] - out[-2][0])) <= 0.0:
                out.pop()
            out.append(p)
        return out

    lower, upper = chain(pts), chain(pts[::-1])
    return np.array(lower[:-1] + upper[:-1], float)


def _capsule(half, r, mat, seg=10):
    """A capsule along z: a lathe from its bottom apex round to its top one (closed, outward)."""
    t = np.linspace(0.0, 0.5 * math.pi, 5)
    lower = np.stack([r * np.sin(t), -half - r * np.cos(t)], 1)
    upper = np.stack([r * np.cos(t), half + r * np.sin(t)], 1)
    return S.lathe(np.vstack([lower, upper]), seg=seg, mat=mat)


def piece_mesh(p, seed, mat):
    """One chamber piece as a closed mesh in the lathe frame (material `mat`): a sphere, a seed bead (a torus), a glass bit
    (a rounded slab of an irregular outline) or a rod (a capsule)."""
    rng = np.random.default_rng([int(seed), 91, int(p["ink"])])
    if p["kind"] == "bead":
        t = np.linspace(0.0, math.pi, 9)
        mesh = S.lathe(np.stack([p["size"] * np.sin(t), -p["size"] * np.cos(t)], 1), seg=16, mat=mat)
    elif p["kind"] == "seed":
        r = p["thick"]
        mesh = S.tube(S.torus_path(p["size"] - r, n=18), r, sides=10, closed=True, mat=mat).weld(1e-9)
    elif p["kind"] == "bit":
        out = S.fillet(bit_outline(rng, p["size"]), 0.0006, 3, closed=True)
        mesh = S.rounded_panel(out, 2.0 * p["thick"], r_edge=0.0003, n=2, r_back=0.0003, mat=mat)
        mesh = mesh.moved((0.0, 0.0, -p["thick"]))
    else:
        mesh = _capsule(p["size"] - ROD_R, ROD_R, mat)
    return mesh.rot(*p["turn"]).moved(p["centre"])


def chamber_mesh(seed=1, inks=len(BEADS)):
    """Every piece of `chamber_layout(seed)` in one mesh (lathe frame): beads and seed beads in material BEAD_MAT + i,
    glass bits and rods in BIT_MAT + i, i = ink % inks."""
    parts = []
    for p in chamber_layout(seed):
        base = BEAD_MAT if p["kind"] in ("bead", "seed") else BIT_MAT
        parts.append(piece_mesh(p, seed, base + p["ink"] % inks))
    return S.merge(parts)


# ================================================================================================= the card
def card(name, slots=None):
    """The prop card (docs/design.md: Prop card): use points in the prop frame (the module docstring), the colour roles
    and the root property `turn`."""
    def rnd(v):
        return [round(float(x), 5) + 0.0 for x in v]

    eye = point(EYE_POINT_S)
    paper = (COLLAR_EYE, LENGTH - FAR_RIM)
    top = TUBE_R + 0.0001
    use = {"look": [{"name": "eyepiece", "point": rnd(eye)}, {"name": "chamber", "point": rnd(point(LENGTH))},
                    {"name": "grip", "point": rnd(point(GRIP_S))}],
           "anchor": [{"name": "eyehole", "point": rnd(eye), "dir": rnd(-AXIS), "radius": round(EYE_OPEN, 5),
                       "object": f"{name}_eyehole"}],
           "rest": [{"name": "side", "type": "edge", "a": rnd(point(paper[0] + 0.01, top, -90.0)),
                     "b": rnd(point(paper[1] - 0.01, top, -90.0)), "normal": [0.0, 0.0, 1.0]}],
           "grip": [{"name": "tube", "type": "cylinder", "center": rnd(point(0.5 * (paper[0] + paper[1]))),
                     "axis": rnd(AXIS), "radius": TUBE_R, "height": round(paper[1] - paper[0], 5),
                     "object": f"{name}_tube"}]}
    return {"size": [round(2 * AXIS_Z, 5), LENGTH, round(2 * AXIS_Z, 5)], "origin": "floor_center", "front": "-Y",
            "slots": dict(slots or {}), "use": use, "params": {"turn": 0.0},
            "objects": [f"{name}_{p}" for p in ("tube", "cap", "ferrule", "lens", "tunnel", "view", "rim", "frost",
                                                "glass", "mirrors", "beads")]}
