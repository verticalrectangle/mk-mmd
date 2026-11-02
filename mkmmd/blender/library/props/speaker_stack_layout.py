"""Layout of the festival speaker stack (pure Python + numpy, no bpy: importable from tests): the cabinets, their meshes, the
cones' pump, the colour roles and the card.

Frame (metres): the stack stands on the floor, origin under the middle of its footprint, its fronts flush toward -Y (the
cones look -Y), +Z up. `stack` lists the cabinets from the floor up (default two subs under a top). Each cabinet stands on
four rubber feet on the one below.

- `sub`: a front-loaded bass reflex cabinet with two 18 inch woofers side by side, 1.20 x 0.75 x 0.56 m, a slot port at
  each end of the baffle.
- `top`: a mid / high cabinet, 0.80 x 0.52 x 0.40 m: two 10 inch mids either side of a constant-directivity horn.

A cabinet's shell is ONE closed cage of rings round its depth (rounded rectangles inset ring by ring): the back rolled into
the sides, the front edge rolled over into a rim and down a short wall into a cove onto the baffle, recessed RECESS behind
the front, with support rings beside every roll so the flat faces shade flat; the builder cuts the drivers' holes, the
ports' tunnels and the horn's mouth out of it (`cutters`, a Boolean, baked). The drivers are lathes about their axis (a
frame ring with eight bolt heads over the hole's edge, a half-roll surround, a flared paper cone, a domed dust cap; closed
behind, out of sight in the cabinet). A port is a deep stadium tunnel with a flared lip on the baffle;
the horn ONE closed cage of rounded-rectangle rings (a rolled lip on the baffle, its walls flaring from the throat, a thin
wall) with a phase plug in the throat; the grille a welded wire mesh just behind the front (vertical wires in front of the
horizontal ones, their ends in a perimeter wire); per side two rigging plates (front, back) with eyelets and a
quick-release pin, a bar handle between them; four lathed feet; a connector plate on the back. No logo, badge or
lettering.

The root property `pump` (0 .. 1, for [[key]]: a pulse on a lane's hits) drives the shape key `pump` of every cabinet's
`drivers`: each cone and its dust cap push out by the cabinet's PUSH (18 mm on a sub, 7 on a top), the surround stretching
from its fixed outer edge to the cone.

Colour roles (`slots` of the prop: a palette slot, `ink` / `paper`, a hex or an `a:b:t` mix): cabinet (base: the textured
coating), baffle (ink), grille (hl_med: powder-coated steel), cone (ink), surround (overlay:ink:0.5), cap (love: the dust
caps), horn (ink), hardware (subtle: plates, pins, bolts, frames, handles), rubber (ink: feet, sockets, the phase plugs),
port (ink: the port slots).

Card: `use.look` `front` (the middle of the stack's front), `top` (its top) and `cone<i>` (each cone's centre, bottom up,
left to right as you face the stack), `use.anchor` `cone<i>` (the same points, `dir` -Y), `use.rest` `top` (a plane on the
top cabinet), a box collider per cabinet, `params` {pump}, `form_max` FORM_MAX: a speaker cabinet is a box by nature, as a
box trailer is (the shells read 0.9 as boxes however well their edges roll; the stack measures about 0.7)."""
import math

import numpy as np

from ....core import shell as S
from . import showkit as SK

RIM = 0.030                      # the front rim's width round the baffle
RECESS = 0.050                   # how far the baffle sits behind the front
R_CORNER = 0.024                 # the section's corners (the edges along the depth)
R_EDGE = 0.018                   # the front and back edges' roll
R_IN = 0.008                     # the rim's inner roll into the opening
R_COVE = 0.006                   # the cove where the opening's wall meets the baffle
SUPPORT = 0.004                  # support rings this far beside a roll
R_MIN = 0.006                    # the least corner radius of an inset ring
HOLE = 0.93                      # a driver's hole radius, a share of its frame's (the frame ring covers the edge)
PORT_DEPTH = 0.30                # how deep a port's tunnel goes into the cabinet
FOOT = (0.032, 0.014)            # a rubber foot's radius and height
GRILLE = {"pitch": 0.045, "wire": 0.0022, "gap": 0.008}    # mesh pitch, wire radius, its plane behind the front
PORT_LIP = (0.012, 0.008)        # a port lip's width round the slot and its height on the baffle
HANDLE = (0.20, 0.045, 0.0095)   # a bar handle's span, how far it stands off the side, its bar's radius
HANDLE_OUT = HANDLE[1] + HANDLE[2]   # how far a handle reaches out of the side
BACK_OUT = 0.015                 # how far the connector plate and its sockets stand off the back
PUSH = {"sub": 0.018, "top": 0.007}                        # a cone's push out at pump 1 (clear of the grille)
CABINETS = {
    "sub": {"W": 1.20, "H": 0.56, "D": 0.75,
            "drivers": [(-0.275, 0.28, 0.229), (0.275, 0.28, 0.229)],      # (x, z, the frame's radius): 18 inch
            "ports": [(-0.540, 0.28, 0.036, 0.40), (0.540, 0.28, 0.036, 0.40)],   # x, z, slot width, height
            "horn": None},
    "top": {"W": 0.80, "H": 0.40, "D": 0.52,
            "drivers": [(-0.228, 0.20, 0.131), (0.228, 0.20, 0.131)],      # 10 inch
            "ports": [],
            "horn": {"x": 0.0, "z": 0.20, "mouth": (0.15, 0.30), "throat": (0.034, 0.05), "depth": 0.16}},
}
STACK = ["sub", "sub", "top"]
ROLES = {"cabinet": "base", "baffle": "ink", "grille": "hl_med", "cone": "ink", "surround": "overlay:ink:0.5",
         "cap": "love", "horn": "ink", "hardware": "subtle", "rubber": "ink", "port": "ink"}
MATS = tuple(ROLES)              # material index -> role
FORM_MAX = 0.8                   # the form limit of a stack of boxes by nature (as the semi's box trailer)
PARAMS = {"pump": (0.0, "", "the cones' push out, 0 .. 1 (key it on a lane's hits: a pulse)")}


def _m(role):
    return MATS.index(role)


def options(slots):
    """(stack [kind, ...] from the floor up, grille "mesh" | "none", params {pump}) from the `slots` of the [[prop]],
    checked: a bad value is a ValueError that says what to change."""
    slots = slots or {}
    stack = slots.get("stack", STACK)
    if isinstance(stack, str):
        stack = [stack]
    if not isinstance(stack, (list, tuple)) or not 1 <= len(stack) <= 6 or any(k not in CABINETS for k in stack):
        raise ValueError(f"speaker_stack: stack is a list of 1 .. 6 cabinets from the floor up, each "
                         f"{' or '.join(repr(k) for k in CABINETS)} (got {stack!r})")
    if list(stack) != sorted(stack, key=lambda k: k == "top"):
        raise ValueError(f"speaker_stack: the subs stand under the tops (got {stack!r})")
    grille = slots.get("grille", "mesh")
    if grille not in ("mesh", "none"):
        raise ValueError(f"speaker_stack: grille is \"mesh\" or \"none\" (got {grille!r})")
    return list(stack), grille, {"pump": SK.number(slots, "pump", 0.0, "speaker_stack", 0.0, 1.0)}


def placed(stack):
    """[{kind, k, z0 (the cabinet's underside), y0 (its middle), W, H, D}] of the cabinets, bottom up: each on its feet on
    the one below, the fronts flush at y = -D_max / 2."""
    dmax = max(CABINETS[k]["D"] for k in stack)
    out, z = [], 0.0
    for i, kind in enumerate(stack):
        c = CABINETS[kind]
        z += FOOT[1]
        out.append({"kind": kind, "k": i, "z0": z, "y0": -dmax / 2 + c["D"] / 2, "W": c["W"], "H": c["H"], "D": c["D"]})
        z += c["H"]
    return out


def _mapped(m, W):
    return S.Mesh(W, m.Q, m.T, m.Qm, m.Tm, Ce=m.Ce, Cw=m.Cw)


def to_front(m, x, z, y_face):
    """A mesh modelled about its own +Z (its face up, its +Y up the picture) stood on a front face: +Z -> -Y, +Y -> +Z,
    at (x, y_face, z). A rotation: the winding stays outward."""
    V = m.V
    return _mapped(m, np.stack([V[:, 0] + x, y_face - V[:, 2], V[:, 1] + z], 1))


def to_side(m, side, y, z, x_face):
    """A mesh modelled about its own +Z stood on a side face: +Z -> +X (side +1) or -X (side -1), its +Y -> +Z, at
    (x_face, y, z). A rotation."""
    V = m.V
    s = 1.0 if side > 0 else -1.0
    return _mapped(m, np.stack([x_face + s * V[:, 2], y + s * V[:, 0], z + V[:, 1]], 1))


def to_back(m, x, z, y_face):
    """A mesh modelled about its own +Z stood on the back face: +Z -> +Y, +Y -> +Z, at (x, y_face, z). A rotation."""
    V = m.V
    return _mapped(m, np.stack([x - V[:, 0], y_face + V[:, 2], V[:, 1] + z], 1))


# ------------------------------------------------------------------------------------------------------------- shell
def shell_profile(D, n=4):
    """(inset d, depth h) of the shell's rings from the back (h 0) to the baffle (h D - RECESS): a support ring on the back
    face, the back edge rolled into the sides, support rings on the sides, the front edge rolled over into the rim, the
    rim's inner roll down into the opening, its wall, the cove onto the baffle and a support ring on it."""
    a = np.linspace(0.0, 0.5 * math.pi, n + 1)
    back = [(R_EDGE * (1.0 - math.sin(t)), R_EDGE * (1.0 - math.cos(t))) for t in a]
    front = [(R_EDGE * (1.0 - math.cos(t)), D - R_EDGE + R_EDGE * math.sin(t)) for t in a]
    rim = [(RIM - R_IN + R_IN * math.sin(t), D - R_IN + R_IN * math.cos(t)) for t in a]
    cove = [(RIM + R_COVE * (1.0 - math.cos(t)), D - RECESS + R_COVE * (1.0 - math.sin(t))) for t in a]
    prof = ([(R_EDGE + SUPPORT, 0.0)] + back + [(0.0, R_EDGE + SUPPORT), (0.0, D - R_EDGE - SUPPORT)] + front + rim
            + [(RIM, D - RECESS + R_COVE + SUPPORT)] + cove + [(RIM + R_COVE + SUPPORT, D - RECESS)])
    return [p for i, p in enumerate(prof) if i == 0 or abs(p[0] - prof[i - 1][0]) + abs(p[1] - prof[i - 1][1]) > 1e-9]


def shell_mesh(kind, n=12):
    """A cabinet's shell in its own frame (x across, its front at y = -D / 2, its underside at z = 0): materials cabinet
    and, the face inside the cove, baffle. Each ring is the rounded rectangle inset by its d, its corner radius shrinking
    with the inset down to R_MIN (an exact offset would turn the corners inside out past the corner's own radius)."""
    c = CABINETS[kind]
    W, H, D = c["W"], c["H"], c["D"]
    R = np.stack([np.column_stack([S.rrect(W - 2 * d, H - 2 * d, max(R_CORNER - d, R_MIN), n), np.full(4 * (n + 1), h)])
                  for d, h in shell_profile(D)])                       # (rings, m, 3): (x, z - H / 2, h)
    m = S.cage(R, closed=True, caps=("fan", "fan"), cap_axes=((0.0, 0.0, -1.0), (0.0, 0.0, 1.0)), mat=_m("cabinet"))
    out = _mapped(m, np.stack([m.V[:, 0], D / 2 - m.V[:, 2], m.V[:, 1] + H / 2], 1))
    y_baffle = -D / 2 + RECESS
    out.assign(_m("baffle"), lambda ctr, nrm: (np.abs(ctr[:, 1] - y_baffle) < 1e-6) & (nrm[:, 1] < -0.99))
    return out


def opening(kind):
    """(half width, z0, z1) of a cabinet's opening inside the rim, in its own frame."""
    c = CABINETS[kind]
    return c["W"] / 2 - RIM, RIM, c["H"] - RIM


def _prism(outline, z0, z1):
    """A closed prism of a 2D outline from z0 to z1 about +Z (a cutter)."""
    return S.rounded_panel(outline, z1 - z0, r_edge=0.0005, n=1).moved((0.0, 0.0, z0))


def cutters(kind):
    """The solids cut out of a cabinet's shell (in its frame), all through the baffle into the cabinet: a round hole
    under each driver's frame (hidden by it), a deep stadium tunnel per port, the horn's mouth (hidden by its lip). The
    cut faces take the port's material."""
    c = CABINETS[kind]
    y_baffle = -c["D"] / 2 + RECESS
    out = []
    for x, z, Rf in c["drivers"]:
        hole = S.lathe(np.array([(0.0, -0.6 * Rf - 0.02), (HOLE * Rf, -0.6 * Rf - 0.02), (HOLE * Rf, 0.01), (0.0, 0.01)]),
                       seg=64)
        out.append(to_front(hole, x, z, y_baffle))
    for x, z, w, h in c["ports"]:
        out.append(to_front(_prism(S.stadium(w, h, 8), -PORT_DEPTH, 0.01), x, z, y_baffle))
    hc = c["horn"]
    if hc is not None:
        mw, mh = hc["mouth"]
        out.append(to_front(_prism(S.rrect(mw + 0.004, mh + 0.004, 0.028, 4), -hc["depth"] - 0.02, 0.01), hc["x"], hc["z"],
                            y_baffle))
    return out


# ----------------------------------------------------------------------------------------------------------- drivers
def driver_profile(Rf, n_roll=10, n_cone=12, n_cap=10):
    """(r, z) points of a woofer about its axis (z out of the baffle, the baffle at z = 0) and the pump weight of each
    (0 fixed, 1 moving with the cone): closed behind (the basket's back, out of sight), the frame ring 6 mm proud with a
    rolled outer edge, the half-roll surround (its weight going from 0 at the frame to 1 at the cone), the flared cone,
    the domed dust cap."""
    fw = 0.13 * Rf
    r_so, r_si = Rf - fw, Rf - 2.0 * fw
    roll = 0.062 * Rf
    r_dc, depth, bulge = 0.33 * Rf, 0.40 * Rf, 0.12 * Rf
    z_f, z_s = 0.006, 0.004
    pts = [(0.0, -0.55 * Rf, 0.0), (Rf, 0.0, 0.0), (Rf, z_f - 0.002, 0.0), (Rf - 0.002, z_f, 0.0),
           (r_so + 0.0015, z_f, 0.0), (r_so, z_s, 0.0)]
    rc, ra = 0.5 * (r_so + r_si), 0.5 * (r_so - r_si)
    for t in np.linspace(0.0, math.pi, n_roll + 1)[1:]:
        r = rc + ra * math.cos(t)
        pts.append((r, z_s + roll * math.sin(t), (r_so - r) / (r_so - r_si)))
    for u in np.linspace(0.0, 1.0, n_cone + 1)[1:]:
        pts.append((r_si + (r_dc - r_si) * u, z_s - depth * u ** 1.25, 1.0))
    z_dc = z_s - depth + 0.0015
    pts.append((r_dc - 0.002, z_dc, 1.0))                                # the cap's glued rim
    for t in np.linspace(0.0, 0.5 * math.pi, n_cap + 1)[1:]:
        pts.append(((r_dc - 0.002) * math.cos(t), z_dc + bulge * math.sin(t), 1.0))
    P = np.array(pts)
    P[-1, 0] = 0.0
    return P[:, :2], P[:, 2]


def driver_zones(Rf):
    """(r_so, r_si, r_dc): the surround's outer and inner radii and the dust cap's."""
    fw = 0.13 * Rf
    return Rf - fw, Rf - 2.0 * fw, 0.33 * Rf - 0.002


def driver_mesh(Rf, seg=None):
    """One woofer about its axis (+Z out of the baffle at z = 0): (Mesh, pump weight per vertex). The frame and the back
    are `hardware`, the surround `surround`, the cone `cone`, the dust cap `cap`; eight bolt heads on the frame ring."""
    seg = seg or (64 if Rf > 0.18 else 48)
    P, w = driver_profile(Rf)
    m = S.lathe(P, seg=seg, mat=_m("hardware"))
    wv = np.repeat(w, [1 if r < S.EPS else seg for r in P[:, 0]])
    r_so, r_si, r_dc = driver_zones(Rf)

    def front(lo, hi):
        return lambda c, n: (np.hypot(c[:, 0], c[:, 1]) > lo) & (np.hypot(c[:, 0], c[:, 1]) <= hi) & (n[:, 2] > -0.2)
    m.assign(_m("surround"), front(r_si, r_so + 1e-4))
    m.assign(_m("cone"), front(r_dc, r_si))
    m.assign(_m("cap"), front(-1.0, r_dc))
    rb = 0.0045 + 0.0087 * Rf                                            # a bolt head's radius
    bolt = S.lathe(S.fillet(np.array([(0.0, 0.0), (rb, 0.0), (rb, 0.0028), (0.0, 0.0042)]), [0.0, 0.0, 0.0012, 0.0], 3),
                   seg=12, mat=_m("hardware"))
    ring = 0.5 * (Rf + r_so)
    bolts = [bolt.moved((ring * math.cos(a), ring * math.sin(a), 0.005)) for a in
             np.linspace(0.0, 2.0 * math.pi, 9)[:-1] + math.pi / 8]
    out = S.merge([m] + bolts)
    return out, np.concatenate([wv, np.zeros(len(out.V) - len(wv))])


def drivers(kind):
    """(Mesh, pump displacement per vertex (n, 3) at pump 1) of a cabinet's woofers in its own frame, on the baffle."""
    c = CABINETS[kind]
    y_baffle = -c["D"] / 2 + RECESS
    parts, disp = [], []
    for x, z, Rf in c["drivers"]:
        m, w = driver_mesh(Rf)
        parts.append(to_front(m, x, z, y_baffle))
        d = np.zeros((len(w), 3))
        d[:, 1] = -PUSH[kind] * w
        disp.append(d)
    return S.merge(parts), np.concatenate(disp)


# -------------------------------------------------------------------------------------------------------- ports, horn
def ports_mesh(kind):
    """The ports' flared lips on the baffle (material port), each round the mouth of its tunnel (`cutters`): a ring with
    rolled rims, its inner wall flush with the tunnel's."""
    c = CABINETS[kind]
    y_baffle = -c["D"] / 2 + RECESS
    lip, h = PORT_LIP
    out = []
    for x, z, w, ht in c["ports"]:
        f = S.rounded_frame(S.stadium(w + 2 * lip, ht + 2 * lip, 8), S.stadium(w, ht, 8), h + 0.002, r_out=0.005,
                            r_in=0.004, n=3, mat=_m("port"), draft=0.003)
        out.append(to_front(f.moved((0.0, 0.0, -0.002)), x, z, y_baffle))
    return S.merge(out)


def horn_mesh(kind, rings=10, n=6):
    """The horn: a rolled lip on the baffle and its walls flaring from the throat, a thin closed wall (material horn),
    and the compression driver's phase plug in the throat (material rubber)."""
    hc = CABINETS[kind]["horn"]
    if hc is None:
        return S.Mesh()
    mw, mh = hc["mouth"]
    tw, th = hc["throat"]
    L = hc["depth"]
    wall, lip, z_lip = 0.004, 0.012, 0.007

    def flare(u, off=0.0):
        f = u ** 1.8                             # flaring faster toward the mouth
        return 0.5 * (tw + (mw - tw) * f) + off, 0.5 * (th + (mh - th) * f) + off, 0.006 + 0.02 * f + off
    secs = [(*flare(u), -L * (1.0 - u)) for u in np.linspace(0.0, 1.0, rings + 1)]          # up the inside
    for t in np.linspace(0.0, math.pi, n + 1)[1:-1]:                                        # over the rolled lip
        d = 0.5 * lip * (1.0 - math.cos(t))
        secs.append((0.5 * mw + d, 0.5 * mh + d, 0.026 + d, z_lip * math.sin(t)))
    secs.append((0.5 * mw + lip, 0.5 * mh + lip, 0.026 + lip, 0.0))
    secs += [(*flare(u, wall), -L * (1.0 - u)) for u in np.linspace(1.0, 0.0, rings + 1)]   # down the outside
    Rg = []
    for hw, hh, r, z in secs:
        P = S.rrect(2 * hw, 2 * hh, min(r, 0.99 * min(hw, hh)), 4)
        Rg.append(np.column_stack([P, np.full(len(P), z)]))
    m = S.cage(np.stack(Rg), closed=True, loop=True, mat=_m("horn"))
    pr = 0.9 * min(tw, th) / 2
    plug = S.lathe(S.fillet(np.array([(0.0, -L - 0.01), (pr, -L - 0.01), (pr, -L + 0.002), (0.0, -L + 0.009)]),
                            [0.0, 0.0, 0.003, 0.0], 3), seg=24, mat=_m("rubber"))
    c = CABINETS[kind]
    return to_front(S.merge([m, plug]), hc["x"], hc["z"], -c["D"] / 2 + RECESS)


# ------------------------------------------------------------------------------------------------------------ grille
def grille_mesh(kind):
    """The welded wire mesh across the opening (material grille): vertical wires in front, horizontal ones touching them
    from behind, the ends in a perimeter wire."""
    c = CABINETS[kind]
    hw, z0, z1 = opening(kind)
    pitch, r, gap = GRILLE["pitch"], GRILLE["wire"], GRILLE["gap"]
    yv = -c["D"] / 2 + gap                      # the vertical wires' axis
    yh = yv + 2.0 * r                           # the horizontal wires'
    inset = 0.007
    nx = int((2 * (hw - inset)) // pitch)
    xs = (np.arange(nx + 1) - nx / 2) * pitch
    nz = int((z1 - z0 - 2 * inset) // pitch)
    zs = 0.5 * (z0 + z1) + (np.arange(nz + 1) - nz / 2) * pitch
    up = (0.0, -1.0, 0.0)
    wires = [S.tube(np.array([[x, yv, z0 + inset], [x, yv, z1 - inset]]), r, sides=8, mat=_m("grille"), up=up)
             for x in xs]
    wires += [S.tube(np.array([[-(hw - inset), yh, z], [hw - inset, yh, z]]), r, sides=8, mat=_m("grille"), up=up)
              for z in zs]
    ring = S.rrect(2 * (hw - inset), (z1 - z0) - 2 * inset, 0.012, 4)
    path = np.column_stack([ring[:, 0], np.full(len(ring), 0.5 * (yv + yh)), ring[:, 1] + 0.5 * (z0 + z1)])
    frame = S.tube(path, 1.6 * r, sides=8, closed=True, caps=(False, False), mat=_m("grille"), up=up).weld(1e-9)
    return S.merge(wires + [frame if frame.volume() > 0 else frame.flip()])


# ---------------------------------------------------------------------------------------------------------- hardware
def _plate(w, h, t):
    return S.rounded_panel(S.rrect(w, h, 0.012, 4), t, r_edge=0.0025, n=3, mat=_m("hardware"))


def _eyelet():
    """A pressed steel eyelet (about +Z): a raised ring round a shallow hole, its foot sunk in the plate."""
    return S.lathe(S.fillet(np.array([(0.0, -0.002), (0.010, -0.002), (0.010, 0.0025), (0.006, 0.0025), (0.006, 0.0005),
                                      (0.0, 0.0005)]), [0.0, 0.0, 0.0012, 0.001, 0.0, 0.0], 3), seg=20,
                   mat=_m("hardware"))


def _pin():
    """A quick-release pin's button head (about +Z) with its pull ring hanging below (its own -Y)."""
    head = S.lathe(S.fillet(np.array([(0.0, 0.0), (0.0085, 0.0), (0.0085, 0.006), (0.004, 0.0085), (0.0, 0.0095)]),
                            [0.0, 0.0, 0.002, 0.0015, 0.0], 3), seg=24, mat=_m("hardware"))
    a = np.linspace(0.0, 2.0 * math.pi, 25)[:-1]
    ring = S.tube(np.stack([0.013 * np.cos(a), -0.013 + 0.013 * np.sin(a), np.full_like(a, 0.004)], 1), 0.0016, sides=8,
                  closed=True, caps=(False, False), mat=_m("hardware")).weld(1e-9)
    return S.merge([head, ring if ring.volume() > 0 else ring.flip()])


def _handle(span=HANDLE[0], stand=HANDLE[1], r=HANDLE[2]):
    """A bar handle (about +Z out of the face, the bar along its own X): a round bar on two legs bending into rosettes on
    the face."""
    k = 0.6 * stand
    pts = np.array([(-span / 2, 0.0, -0.004), (-span / 2, 0.0, stand - k), (-span / 2 + k, 0.0, stand),
                    (span / 2 - k, 0.0, stand), (span / 2, 0.0, stand - k), (span / 2, 0.0, -0.004)])
    P2 = S.fillet(pts[:, [0, 2]], [0.0, k, k, k, k, 0.0], 6)
    path = np.column_stack([P2[:, 0], np.zeros(len(P2)), P2[:, 1]])
    bar = S.tube(path, r, sides=14, mat=_m("hardware"), up=(0.0, 1.0, 0.0))
    rose = S.lathe(S.fillet(np.array([(0.0, -0.002), (0.022, -0.002), (0.022, 0.003), (0.0, 0.007)]),
                            [0.0, 0.0, 0.003, 0.0], 3), seg=24, mat=_m("rubber"))
    return S.merge([bar, rose.moved((-span / 2, 0.0, 0.0)), rose.moved((span / 2, 0.0, 0.0))])


def _foot():
    return S.lathe(S.fillet(np.array([(0.0, 0.0), (FOOT[0] - 0.006, 0.0), (FOOT[0], FOOT[1] - 0.002), (FOOT[0], FOOT[1]),
                                      (0.0, FOOT[1])]), [0.0, 0.003, 0.002, 0.0, 0.0], 3), seg=24, mat=_m("rubber"))


def _socket():
    """A round speaker socket (about +Z): a boss with a rolled rim round a recessed face."""
    return S.lathe(S.fillet(np.array([(0.0, -0.001), (0.017, -0.001), (0.017, 0.010), (0.010, 0.010), (0.010, 0.004),
                                      (0.0, 0.004)]), [0.0, 0.0, 0.002, 0.0015, 0.0, 0.0], 3), seg=24, mat=_m("rubber"))


def hardware_mesh(kind):
    """A cabinet's rigging and fittings in its own frame: per side two rigging plates (front, back) with two eyelets and a
    quick-release pin each, a bar handle between them; four feet under it; a connector plate with two sockets on the
    back."""
    c = CABINETS[kind]
    W, H, D = c["W"], c["H"], c["D"]
    out = []
    for side in (1.0, -1.0):
        xf = side * (W / 2 - 0.002)
        for y in (-D / 2 + 0.085, D / 2 - 0.085):
            out.append(to_side(_plate(0.11, 0.17, 0.006), side, y, H / 2, xf))
            for dz in (-0.055, 0.055):
                out.append(to_side(_eyelet(), side, y, H / 2 + dz, xf + side * 0.0042))
            out.append(to_side(_pin(), side, y, H / 2, xf + side * 0.0042))
        out.append(to_side(_handle(), side, 0.0, 0.62 * H, side * W / 2))
    for sx in (-1.0, 1.0):
        for sy in (-1.0, 1.0):
            out.append(_foot().moved((sx * (W / 2 - 0.09), sy * (D / 2 - 0.09), -FOOT[1])))
    panel = S.merge([_plate(0.15, 0.09, 0.005), _socket().moved((-0.035, 0.0, 0.004)),
                     _socket().moved((0.035, 0.0, 0.004))])
    out.append(to_back(panel, W / 2 - 0.16, 0.14, D / 2 - 0.001))
    return S.merge(out)


def collider_box(kind):
    """(centre, size) of a cabinet's collider box in its own frame."""
    c = CABINETS[kind]
    return (0.0, 0.0, c["H"] / 2), (c["W"], c["D"], c["H"])


# --------------------------------------------------------------------------------------------------------------- card
def parts(kind, grille="mesh"):
    """The objects of a cabinet (`<name>_cab<k>_<part>`), each a mesh in the cabinet's frame."""
    out = ["shell", "drivers", "hardware"]
    if CABINETS[kind]["ports"]:
        out.append("ports")
    if CABINETS[kind]["horn"]:
        out.append("horn")
    if grille == "mesh":
        out.append("grille")
    return out


def part_mesh(kind, part):
    """The mesh of one part of a cabinet (`drivers` without its pump: see `drivers`)."""
    return {"shell": shell_mesh, "drivers": lambda k: drivers(k)[0], "hardware": hardware_mesh, "ports": ports_mesh,
            "horn": horn_mesh, "grille": grille_mesh}[part](kind)


def cones(stack):
    """[(point, cabinet k)] of every cone's centre on its cabinet's baffle plane, bottom up, then left to right as you face
    the stack (from -X to +X)."""
    out = []
    for c in placed(stack):
        for x, z, _ in sorted(CABINETS[c["kind"]]["drivers"], key=lambda d: d[0]):
            y = c["y0"] - c["D"] / 2 + RECESS
            out.append(([round(x, 5), round(y, 5), round(c["z0"] + z, 5)], c["k"]))
    return out


def bounds(stack):
    """(lo, hi) of the stack in its frame: the widest cabinet with its handles, the deepest with its connector plates, the
    floor to the top."""
    P = placed(stack)
    W = max(c["W"] for c in P) + 2 * HANDLE_OUT
    dmax = max(c["D"] for c in P)
    return [-W / 2, -dmax / 2, 0.0], [W / 2, dmax / 2 + BACK_OUT, P[-1]["z0"] + P[-1]["H"]]


def layers(stack):
    """The card's height layers [(z0, z1, x0, y0, x1, y1), ...] (docs/design.md: Prop card): each cabinet's plan with its
    handles and back plate, from its feet up (what the placement rules would measure, without sampling every surface)."""
    return [[round(v, 5) for v in (c["z0"] - FOOT[1], c["z0"] + c["H"], -c["W"] / 2 - HANDLE_OUT, c["y0"] - c["D"] / 2,
                                   c["W"] / 2 + HANDLE_OUT, c["y0"] + c["D"] / 2 + BACK_OUT)] for c in placed(stack)]


def card(name, stack, grille="mesh", slots=None, params=None):
    """The prop card (docs/design.md: Prop card)."""
    P = placed(stack)
    lo, hi = bounds(stack)
    top = P[-1]
    zt = top["z0"] + top["H"]
    cs = cones(stack)
    looks = [{"name": "front", "point": [0.0, round(lo[1], 5), round(0.5 * zt, 5)]},
             {"name": "top", "point": [0.0, round(top["y0"], 5), round(zt, 5)]}]
    looks += [{"name": f"cone{i}", "point": p} for i, (p, _) in enumerate(cs)]
    anchors = [{"name": f"cone{i}", "point": p, "dir": [0.0, -1.0, 0.0]} for i, (p, _) in enumerate(cs)]
    rest = [{"name": "top", "type": "plane", "center": [0.0, round(top["y0"], 5), round(zt, 5)], "normal": [0.0, 0.0, 1.0],
             "size": [round(top["W"] - 0.08, 4), round(top["D"] - 0.08, 4)]}]
    return {"size": [round(h - l, 4) for l, h in zip(lo, hi)], "origin": "floor_center", "front": "-Y",
            "bounds": {"min": [round(v, 4) for v in lo], "max": [round(v, 4) for v in hi]}, "layers": layers(stack),
            "slots": dict(slots or {}), "use": {"look": looks, "anchor": anchors, "rest": rest},
            "colliders": [{"type": "box", "object": f"{name}_cab{c['k']}_col", "rnd": 0.02} for c in P],
            "params": dict(params or {"pump": 0.0}), "form_max": FORM_MAX, "stack": list(stack),
            "objects": [f"{name}_cab{c['k']}_{part}" for c in P for part in parts(c["kind"], grille)]}
