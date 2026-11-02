"""semi_truck_body_cab: the tractor's cab (pure numpy, no bpy): the hollow body, its doors, floor, cowl and drop visor, the
windshield and door glass with their rubber seals.

The cab body is a hollow ring stack (`semi_truck_body_kit.ring_shell`): an outer skin (flat sides at CAB_HALF_W, the front
wall at CAB_FRONT_Y stepping forward to the raked windshield and the header, rounded roof edges, back corners rounded
round the stacks) and an inner skin (sides at CAB_HALF_W - CAB_WALL_T, the firewall, the ceiling crowned to 2.78, the back
wall's face), bridged through the wall round four real openings: the two-pane windshield (a centre post between the panes),
the door apertures and the walk-through (L.WALKTHROUGH) into the sleeper. The doors are separate plates in the apertures (a
real shut-line gap all round), the floor a separate slab (top at CAB_FLOOR_Z), the cowl a loft from under the hood's rear edge
to the windshield's foot, the visor a thin curved blade over the glass. All of these are loose parts of the one "cab" cage.

Column j of the body's rings is one line up the cab (the centre post's edges, the windshield's edges and the supports that
keep its corners tight, the A-pillar's corner, the aperture's edges, the back corners, the walk-through's edges); `ROWS` are
the heights the openings and the roof roll need, each with its height on the outer and the inner skin.
"""
import math

import numpy as np

from ....core import shell as S
from . import semi_truck_body_kit as K
from . import semi_truck_layout as L

M = L.M

# ====================================================================================================== numbers
GAP = 0.005                          # the doors' shut-line gap
WS_K = (L.WS_TOP[0] - L.WS_BASE[0]) / (L.WS_TOP[1] - L.WS_BASE[1])    # dy/dz of the windshield's plane (its rake)
WS_PROUD = 0.013                     # the body's face round the windshield stands this far ahead of the glass (y)
WS_REVEAL = 0.045                    # the opening's inner rim behind the glass (y)
WS_OPEN_Z = (L.WS_BASE[1] + 0.015, L.WS_TOP[1] - 0.015)     # the opening's bottom and top (the seal laps the glass)
WS_OPEN_X = (L.WS_HALF_W[0] - 0.015, L.WS_HALF_W[1] - 0.015)  # the opening's half width at its bottom and top
POST_HW = 0.035                      # half width of the centre post between the panes (opening edge)
GLASS_T = 0.006
FRONT_R = (0.025, 0.04)              # the front corners' radius below the windshield and up the A-pillars
BACK_R = 0.135                       # the back corners' radius: clears the stacks' heat shields, its end is the walk-through
INNER_R = (0.015, 0.03)              # the inner skin's front and back corners
X_IN = L.CAB_HALF_W - L.CAB_WALL_T   # 0.94: the inner side walls
Y_FIREWALL = L.CAB_FRONT_Y + L.CAB_WALL_T          # -5.04: the firewall's inner face
Y_BACK_IN = L.BOC_Y - L.CAB_WALL_T                 # the back wall's inner face
ROOF_ROLL = 0.06                     # the roof's edge roll (inset of the top ring)
ROOF_CROWN = 0.035
CEILING_Z = L.CAB_ROOF_Z - 0.08      # 2.78 at the centre
DOOR_GLASS_X = L.CAB_HALF_W - 0.025  # outer face of the door glass
VENT_Y = -4.72                       # the vent window's divider (at the window's bottom)

# rows: name, z on the outer skin, z on the inner skin (the openings' edges, their corner supports, the roof roll)
ROWS = (("bot", 1.100, 1.100), ("ap_bot", L.DOOR_Z[0] - GAP, L.DOOR_Z[0] - GAP), ("ap_bsup", 1.170, 1.170),
        ("floor", L.WALKTHROUGH[1], L.WALKTHROUGH[1]), ("belt", L.DOOR_GLASS[0][1], L.DOOR_GLASS[0][1]),
        ("ledge", 2.040, 2.040), ("wsf", WS_OPEN_Z[0], WS_OPEN_Z[0]), ("ws_lo", WS_OPEN_Z[0] + 0.06, WS_OPEN_Z[0] + 0.06),
        ("wt_sup", L.WALKTHROUGH[2] - 0.06, L.WALKTHROUGH[2] - 0.06), ("wt_top", L.WALKTHROUGH[2], L.WALKTHROUGH[2]),
        ("ws_hi", WS_OPEN_Z[1] - 0.06, WS_OPEN_Z[1] - 0.06), ("wst", WS_OPEN_Z[1], WS_OPEN_Z[1]),
        ("ap_sup", L.DOOR_Z[1] - 0.05, 2.700), ("ap_top", L.DOOR_Z[1] + GAP, 2.735),
        ("roll0", L.CAB_ROOF_Z - ROOF_ROLL, 2.752), ("roof", L.CAB_ROOF_Z, 2.762))
ROW = {name: k for k, (name, _, _) in enumerate(ROWS)}
INNER_ROLL = {"roll0": 0.012, "roof": 0.04}


def ws_y(z):
    """y of the windshield's outer glass plane at height z."""
    return L.WS_BASE[0] + (z - L.WS_BASE[1]) * WS_K


def door_front_y(z):
    """y of the door's front edge at height z: upright to the belt, raked with the windshield above it."""
    return L.DOOR_Y[0] + max(z - L.DOOR_GLASS[0][1], 0.0) * WS_K


def window_front_y(z):
    """y of the door window's front edge (parallel to the door's front edge above the belt)."""
    return L.DOOR_GLASS[0][0] + max(z - L.DOOR_GLASS[0][1], 0.0) * WS_K


def ws_open_x(z):
    t = (z - WS_OPEN_Z[0]) / (WS_OPEN_Z[1] - WS_OPEN_Z[0])
    t = min(max(t, 0.0), 1.0)
    return WS_OPEN_X[0] + (WS_OPEN_X[1] - WS_OPEN_X[0]) * t


# ====================================================================================================== the body
def _front_y(z, inner):
    """The front face's y at height z: the front wall, a ledge to the windshield's foot, the glass's plane (proud of it
    outside, behind it inside) up through the header."""
    if z <= L.DOOR_GLASS[0][1] + 1e-9:
        return Y_FIREWALL if inner else L.CAB_FRONT_Y
    if z < WS_OPEN_Z[0] - 1e-9:
        return Y_FIREWALL - 0.03 if inner else L.CAB_FRONT_Y - 0.035
    return ws_y(z) + (WS_REVEAL if inner else -WS_PROUD)


def outline(z, inner=False, inset=0.0):
    """The plan outline (hw, yf, yb, rf, rb) of the body at height z."""
    rf = (INNER_R[0] if inner else (FRONT_R[0] if z < 2.0 else FRONT_R[1]))
    rb = INNER_R[1] if inner else BACK_R
    hw = X_IN if inner else L.CAB_HALF_W
    yb = Y_BACK_IN if inner else L.BOC_Y
    return (hw - inset, _front_y(z, inner) + inset, yb - inset, max(rf - inset, 0.008), max(rb - inset, 0.02))


def columns(z, inner=False, inset=0.0):
    """The column specs of the body's ring at height z (see the module docstring)."""
    hw, yf, yb, rf, rb = outline(z, inner, inset)
    xo = ws_open_x(z)
    front = [POST_HW, POST_HW + 0.06, 0.45, xo - 0.06, xo, hw - rf, ("CF", 0.5)]
    yaf = door_front_y(z) - GAP
    if inner:
        yaf = max(yaf, yf + rf + 0.01)
    yar = L.DOOR_Y[1] + GAP
    side = [("CF", 1.0), yaf, max(door_front_y(z) + 0.05, yaf + 0.02), L.DOOR_Y[1] - 0.05, yar]
    back = [hw - rb if not inner else L.WALKTHROUGH[0], L.WALKTHROUGH[0] - 0.06]
    if not inner:
        back[0] = L.WALKTHROUGH[0]
    return K.mirrored_columns(front, side, [0.45, 0.8], back)


def _ring(name, inner):
    _, zo, zi = ROWS[ROW[name]]
    z = zi if inner else zo
    zref = zo                           # the openings' columns follow the outer row's height
    if inner:
        inset = INNER_ROLL.get(name, 0.0)
    else:
        inset = {"bot": 0.018, "roof": ROOF_ROLL}.get(name, 0.0)
    P = outline(zref, inner, inset)
    return K.ring(P, columns(zref, inner, inset), z)


def holes():
    """Labels of the body's wall quads (rows of quads x columns): 1 the windshield's panes, 2 the door apertures, 3 the
    walk-through."""
    specs = columns(2.3)
    C = len(specs)
    H = np.zeros((len(ROWS) - 1, C), np.int64)
    idx = {(sp[0], round(sp[1], 4), sp[2]): j for j, sp in enumerate(specs)}

    def col(seg, v, s):
        return idx[(seg, round(v, 4), s)]

    xo = ws_open_x(2.3)
    for s in (1, -1):
        a, b = sorted((col("F", s * POST_HW, s), col("F", s * xo, s)))
        H[ROW["wsf"]:ROW["wst"], a:b] = 1
        a, b = sorted((col("S", door_front_y(2.3) - GAP, s), col("S", L.DOOR_Y[1] + GAP, s)))
        H[ROW["ap_bot"]:ROW["ap_top"], a:b] = 2
    a, b = sorted((col("B", L.WALKTHROUGH[0], 1), col("B", -L.WALKTHROUGH[0], -1)))
    H[ROW["floor"]:ROW["wt_top"], a:b] = 3
    return H


def body():
    """The hollow cab body: (Mesh, ids)."""
    names = [r[0] for r in ROWS]
    O = np.stack([_ring(n, False) for n in names])
    I = np.stack([_ring(n, True) for n in names])
    for k, n in enumerate(names):          # the openings' columns follow the outer rows: put each ring at its own height
        O[k, :, 2] = ROWS[k][1]
        I[k, :, 2] = ROWS[k][2]
    H = holes()
    paint, accent = M["paint"], M["paint_accent"]
    mat_out = np.full(H.shape, paint, np.int32)
    mat_out[ROW["roll0"]:] = accent                      # the roof (two-tone with the visor and the sleeper's cap)
    mat_in = np.full(H.shape, M["vinyl_dark"], np.int32)
    mat_in[ROW["roll0"]:] = M["headliner"]
    hw, yf, yb, rf, rb = outline(L.CAB_ROOF_Z, False, ROOF_ROLL)
    mesh, ids = K.ring_shell(O, I, H, top=K.Cap((0.94, 0.6), ROOF_CROWN, -(CEILING_Z - ROWS[-1][2])), bottom="rim",
                             mat_out=mat_out, mat_in=mat_in, mat_rim={1: M["rubber"], 2: M["seam"], 3: M["vinyl_dark"]},
                             mat_top=accent, mat_top_in=M["headliner"], mat_end=M["underbody"],
                             crease_out={1: 1.0, 2: 1.0, 3: 0.8}, crease_in={1: 1.0, 2: 1.0, 3: 0.7})
    return mesh, ids


# ====================================================================================================== doors
def door_grid(z_rows=None):
    """(rows z, columns y as functions of z) of a door: its edges, the window's edges and the supports by the corners."""
    zs = [L.DOOR_Z[0], L.DOOR_Z[0] + 0.05, L.DOOR_GLASS[0][1], L.DOOR_GLASS[0][1] + 0.05, L.DOOR_GLASS[1][1] - 0.05,
          L.DOOR_GLASS[1][1], L.DOOR_Z[1] - 0.05, L.DOOR_Z[1]]
    cols = [door_front_y, lambda z: door_front_y(z) + 0.05, window_front_y, lambda z: window_front_y(z) + 0.05,
            lambda z: L.DOOR_GLASS[1][0] - 0.05, lambda z: L.DOOR_GLASS[1][0], lambda z: L.DOOR_Y[1] - 0.05,
            lambda z: L.DOOR_Y[1]]
    return zs, cols


def door(sx=1.0):
    """One door (the +x one for sx = 1): a plate from the outer skin (x = CAB_HALF_W) to the inner (X_IN) with the window
    opening (DOOR_GLASS, its front edge raked) and the gap's faces; returns its Mesh."""
    zs, cols = door_grid()
    Y = np.array([[c(z) for c in cols] for z in zs])
    Z = np.repeat(np.array(zs)[:, None], len(cols), 1)
    O = np.stack([np.full_like(Y, L.CAB_HALF_W), Y, Z], -1)
    I = np.stack([np.full_like(Y, X_IN), Y, Z], -1)
    keep = np.ones((len(zs) - 1, len(cols) - 1), bool)
    keep[2:5, 2:5] = False
    mat_out = np.full(keep.shape, M["paint"], np.int32)
    m, _ = K.grid_slab(O, I, keep, mat_out=mat_out, mat_in=M["vinyl_dark"], mat_rim=M["seam"], crease_out=1.0,
                       crease_in=1.0)
    # the window's reveal is the glass channel (rubber), the plate's edge is the shut-line gap (seam)
    cq, _ = m.face_centres()
    nq, _ = m.face_vectors()
    rim = np.abs(nq[:, 0]) < 0.5 * np.linalg.norm(nq, axis=1)
    win = (cq[:, 2] > L.DOOR_GLASS[0][1] - 0.01) & (cq[:, 2] < L.DOOR_GLASS[1][1] + 0.01) \
        & (cq[:, 1] > window_front_y(2.3) - 0.40) & (cq[:, 1] < L.DOOR_GLASS[1][0] + 0.01)
    inside = np.array([window_front_y(z) - 0.01 < y for y, z in zip(cq[:, 1], cq[:, 2])])
    m.Qm[rim & win & inside] = M["rubber"]
    return m if sx > 0 else m.mirrored_x()


# ====================================================================================================== floor, cowl, visor
def floor():
    """The cab floor: a slab between the inner walls from the firewall to the back wall, its top at CAB_FLOOR_Z (the
    carpet lies on it). A cage box with creased edges."""
    x0, x1 = -X_IN, X_IN
    y0, y1 = Y_FIREWALL, Y_BACK_IN
    z0, z1 = L.DOOR_Z[0], L.CAB_FLOOR_Z
    V = np.array([[x, y, z] for z in (z0, z1) for y in (y0, y1) for x in (x0, x1)], float)
    Q = np.array([[0, 2, 3, 1], [4, 5, 7, 6], [0, 1, 5, 4], [2, 6, 7, 3], [0, 4, 6, 2], [1, 3, 7, 5]])
    m = S.Mesh(V, Q, mat=M["carpet"])
    if m.volume() < 0:
        m = m.flip()
    m.Qm[:] = M["underbody"]
    cq, _ = m.face_centres()
    m.Qm[cq[:, 2] > z1 - 1e-6] = M["carpet"]
    m.crease_where(lambda a, b: np.ones(len(a), bool), 1.0)
    return m


COWL_X = 0.70                       # half width (the air cleaners' brackets meet its sides)
COWL_Y = (-5.52, L.CAB_FRONT_Y + 0.01)
COWL_TOP = ((-5.52, L.HOOD_Z[1] - 0.006), (-5.30, L.HOOD_Z[1] + 0.008), (-5.12, L.WS_BASE[1] - 0.004))
COWL_BOTTOM = 1.60


def cowl():
    """The cowl panel between the hood's rear edge and the windshield's foot: a lofted block, flat sides where the air
    cleaners' brackets meet it, its top continuing the hood's line up to the glass, its hidden underside a vault."""
    ys = [COWL_Y[0], COWL_Y[0] + 0.02, -5.30, -5.15, COWL_Y[1]]
    ztop = np.interp(ys, [p[0] for p in COWL_TOP], [p[1] for p in COWL_TOP])
    st = []
    for i, (y, zt) in enumerate(zip(ys, ztop)):
        w = COWL_X - (0.012 if i == 0 else 0.0)
        zb = COWL_BOTTOM
        pts = [S.pt("keel", 0.0, zb + 0.10), S.pt("u1", 0.45 * w, zb + 0.06), S.pt("u2", w - 0.06, zb),
               S.pt("s0", w, zb + 0.06), S.pt("s1", w, zt - 0.05), S.pt("e1", w - 0.012, zt - 0.010),
               S.pt("e2", w - 0.05, zt), S.pt("t1", 0.4 * w, zt + 0.006), S.pt("tc", 0.0, zt + 0.008)]
        st.append(S.station(y, pts))
    lf = S.loft(st, spacing=0.12, cap_bulge=(0.02, 0.0), mat=M["paint"])
    lf.assign(M["underbody"], cols=("keel", "u2"))
    return lf.mesh


VISOR = dict(x=0.97, back=(-4.86, 2.845), front=(-5.24, 2.745), t=0.035)


def visor():
    """The drop visor over the windshield: a thin blade from the header (its back buried in the roof's front roll) out and
    down to a rounded front lip, slightly crowned across, its ends rounded in plan. A cage of sections along x."""
    (yb, zb), (yf, zf), t = VISOR["back"], VISOR["front"], VISOR["t"]
    hx = VISOR["x"]
    xs = np.array([-hx, -hx + 0.02, -hx + 0.08, -0.5, 0.0, 0.5, hx - 0.08, hx - 0.02, hx])
    rings = []
    for x in xs:
        e = abs(x) / hx
        sweep = 0.04 * e ** 4                                    # the ends swept back a little
        shrink = {1.0: 0.80}.get(round(e, 6), 1.0)
        crown = 0.012 * (1.0 - e * e)
        a = np.array([yb, zb + crown])
        b = np.array([yf + sweep, zf + crown])
        d = (b - a) / np.linalg.norm(b - a)
        n = np.array([-d[1], d[0]])                              # up off the blade (y, z)
        if n[1] < 0:
            n = -n
        span = np.linalg.norm(b - a) * shrink
        sec = []
        for s, h in ((0.0, 0.5), (0.25, 0.55), (0.6, 0.5), (0.88, 0.42), (0.97, 0.25), (1.0, 0.0), (0.97, -0.25),
                     (0.88, -0.42), (0.6, -0.5), (0.25, -0.5), (0.0, -0.5)):
            p = a + d * span * s + n * t * h
            sec.append([x, p[0], p[1]])
        rings.append(sec)
    R = np.array(rings)
    m = S.cage(R, closed=True, caps=("fan", "fan"), cap_rings=1, mat=M["paint_accent"])
    return m if m.volume() > 0 else m.flip()


def cab_cage():
    """Every loose part of the "cab" shell merged: the body, two doors, the floor, the cowl and the visor."""
    b, _ = body()
    return S.merge([b, door(1.0), door(-1.0), floor(), cowl(), visor()])


# ====================================================================================================== glass and seals
def ws_frame():
    """(origin, u, v, n): the windshield's plane frame, origin on the centre line at the glass's foot, u toward +x... seen
    from the front u points to the viewer's right (-x), v up the glass, n out of it (forward and up)."""
    up = np.array([0.0, L.WS_TOP[0] - L.WS_BASE[0], L.WS_TOP[1] - L.WS_BASE[1]])
    n = np.array([0.0, -up[2], up[1]])
    return K.frame_in_plane((0.0, L.WS_BASE[0], L.WS_BASE[1]), n, up)


def ws_length():
    return math.hypot(L.WS_TOP[0] - L.WS_BASE[0], L.WS_TOP[1] - L.WS_BASE[1])


def _pane_outline(u0, u1, w0, w1, h, r):
    """Closed CCW outline (u, v) of a pane from |u| = u0 (the post side) to u1 at the bottom (w1 at the top), v 0..h."""
    P = np.array([[u0, 0.0], [u1, 0.0], [w1, h], [w0, h]])
    if u1 < u0:
        P = P[::-1]
    return S.fillet(P, r, n=5, closed=True)


def windshield_glass():
    """The two flat panes in the windshield's plane (outer face on the WS_BASE -> WS_TOP line, GLASS_T thick)."""
    fr = ws_frame()
    o, u, v, n = fr
    h = ws_length()
    out = []
    for s in (1.0, -1.0):
        # u runs toward -x seen from the front: a pane on the +x side has negative u
        P = _pane_outline(-s * 0.02, -s * L.WS_HALF_W[0], -s * 0.02, -s * L.WS_HALF_W[1], h, [0.02, 0.06, 0.06, 0.02])
        pane = S.rounded_panel(P, GLASS_T, r_edge=0.0025, r_back=0.002, n=2, mat=M["glass"])
        out.append(S.place(pane, o - n * GLASS_T, n, v))
    return out


def ws_opening_outline(s, n_pts=64):
    """The windshield opening's edge on the body's face (the seal's path) for the pane on side s, as 3D points."""
    z0, z1 = WS_OPEN_Z
    corners = [(s * POST_HW, z0), (s * ws_open_x(z0), z0), (s * ws_open_x(z1), z1), (s * POST_HW, z1)]
    P = np.array([[x, ws_y(z) - WS_PROUD, z] for x, z in corners])
    # round the corners the way the subdivided opening does (radius ~ the support spacing)
    P2 = np.array([[p[0], p[2]] for p in P])
    if s < 0:
        P2 = P2[::-1]
    F = S.resample_loop(S.fillet(P2, 0.05, n=6, closed=True), n_pts)
    return np.array([[x, ws_y(z) - WS_PROUD, z] for x, z in F])


def windshield_seals():
    fr = ws_frame()
    n = fr[3]
    return [K.seal(ws_opening_outline(s), n, 0.022, 0.006, M["rubber"], lift=-0.002) for s in (1.0, -1.0)]


def door_window_outline(n_pts=72):
    """The +x door window's opening (y, z) as a closed CCW outline seen from +x, corners rounded as the subdivided
    door's opening rounds them."""
    (y0, z0), (y1, z1) = L.DOOR_GLASS
    P = np.array([[window_front_y(z0), z0], [y1, z0], [y1, z1], [window_front_y(z1), z1]])
    return S.resample_loop(S.fillet(P, 0.035, n=6, closed=True), n_pts)


def door_glass(sx=1.0):
    """The door's pane in the window (x = DOOR_GLASS_X outer face), lapping the opening by 1.5 cm."""
    P = door_window_outline(80)
    c = P.mean(0)
    P = c + (P - c) * (1.0 + 0.03 / np.maximum(np.abs(P - c).max(0), 1e-6))   # grown ~1.5 cm all round
    pane = S.rounded_panel(P, GLASS_T, r_edge=0.0025, r_back=0.002, n=2, mat=M["glass"])
    # the panel's (u, v) plane -> (y, z), its normal -> +x
    m = S.Mesh(np.column_stack([DOOR_GLASS_X - GLASS_T + pane.V[:, 2], pane.V[:, 0], pane.V[:, 1]]), pane.Q, pane.T,
               pane.Qm, pane.Tm)
    m = m if m.volume() > 0 else m.flip()
    return m if sx > 0 else m.mirrored_x()


def door_seal(sx=1.0):
    P = door_window_outline(90)
    out = np.column_stack([np.full(len(P), L.CAB_HALF_W), P[:, 0], P[:, 1]])
    m = K.seal(out, (1.0, 0.0, 0.0), 0.020, 0.005, M["rubber"], lift=-0.001)
    return m if sx > 0 else m.mirrored_x()


def vent_divider(sx=1.0):
    """The vent window's divider: a slim rubber-covered bar up the door window, raked like its front edge."""
    z0, z1 = L.DOOR_GLASS[0][1] - 0.01, L.DOOR_GLASS[1][1] + 0.01
    zz = np.linspace(z0, z1, 8)
    path = np.column_stack([np.full(8, DOOR_GLASS_X + 0.004), VENT_Y + (zz - z0) * WS_K * 0.55, zz])
    m = S.sweep(path, S.rrect(0.014, 0.022, 0.006, 3), up=(1.0, 0.0, 0.0), mat=M["rubber"])
    m = m if m.volume() > 0 else m.flip()
    return m if sx > 0 else m.mirrored_x()


def glass_parts():
    return windshield_glass() + [door_glass(1.0), door_glass(-1.0)]


def seal_parts():
    return windshield_seals() + [door_seal(1.0), door_seal(-1.0), vent_divider(1.0), vent_divider(-1.0)]
