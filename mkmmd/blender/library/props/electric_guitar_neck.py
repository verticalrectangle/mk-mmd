"""The guitar's neck, headstock and everything on them (pure numpy, no bpy).

The NECK is one loft (`core.shell.skin`) of ring sections along z from the heel (inside the body's pocket) over the nut and
through the headstock to its tip. Every ring is the lower half of a superellipse |2x/w|^p + (v/d)^p = 1 (v = depth below the
board's crown plane) cut at the board's underside (v = BOARD_T): along the neck that is EXACTLY the card's `section` (w, d from
`electric_guitar_layout.section_wd`, p = 2.6). Past the nut the ring's centre and width follow the headstock's outline and its
exponent grows to 6 (a rounded rectangle) while its depth stays that of the nut, so the back of the C flows into the
headstock's flat back and the headstock's face is the plane the board is glued on. The rings' top corners are rolled over a
little (a radius that grows toward the headstock).

    neck_mesh, board_mesh, frets_mesh, nut_mesh, inlay_mesh       the wood, the radiused rosewood slab, the wires, the bone
    tuners_mesh, truss_mesh, strings_mesh                         the machine heads and string retainers, truss rod cover, strings
"""
import math

import numpy as np

from ....core import shell as S
from . import electric_guitar_layout as L

M = L.MATS
Y_FACE = L.Y_BOARD + L.BOARD_T                    # the headstock's face (the neck wood's top, under the board)
Y_BACK = L.Y_BOARD + L.D_NUT                      # the headstock's back, flush with the back of the neck at the nut
Y_MID = 0.5 * (Y_FACE + Y_BACK)
Z_NUT_BACK = L.Z_NUT_BACK
FACE_UP = (0.0, 0.0, 1.0)
OUT = (0.0, -1.0, 0.0)                            # out of the face


def outward(mesh):
    return mesh if mesh.volume() > 0 else mesh.flip()


def on_face(mesh, x, y, z, normal=OUT, up=FACE_UP):
    """A part modelled with z out of the surface it stands on, x/y across it, put at (x, y, z) on a surface facing `normal`."""
    return S.place(mesh, (x, y, z), normal, up)


def head_extent(u):
    """(x_bass, x_treble) of the headstock's outline at distance u beyond the nut's front face (m), arrays."""
    u = np.clip(np.atleast_1d(np.asarray(u, float)), L.HEAD_BASS[0][0], L.HEAD_L)
    xb = S.pchip([p[0] for p in L.HEAD_BASS], np.array([[p[1]] for p in L.HEAD_BASS]), u)[:, 0]
    xt = S.pchip([p[0] for p in L.HEAD_TREBLE], np.array([[p[1]] for p in L.HEAD_TREBLE]), u)[:, 0]
    return xb, xt


# ===================================================================================================================
# the neck
# ===================================================================================================================
def ring_curve(xc, a, d, p, rc, v_top=L.BOARD_T, n_side=31, n_arc=3, n_face=9):
    """One section (m, 2) of (x, v): the lower half of the superellipse of half width `a`, depth `d` and exponent `p` about
    x = xc, cut at v = v_top, closed over the top by a flat face whose two corners are rolled over with radius rc. Points run
    from the +x wall around the back to the -x wall, then along the face from -x to +x (counter-clockwise with v up)."""
    rc = min(rc, 0.4 * a)
    vs = v_top + rc
    ts = math.asin(min((vs / d) ** (p / 2.0), 1.0))
    th = np.linspace(ts, math.pi - ts, n_side)
    c, s = np.cos(th), np.sin(th)
    x = xc + a * np.sign(c) * np.abs(c) ** (2.0 / p)
    v = d * np.abs(s) ** (2.0 / p)
    xl, xr = x[-1], x[0]
    phi = np.linspace(math.pi, 1.5 * math.pi, n_arc + 1)[1:]
    left = np.stack([xl + rc + rc * np.cos(phi), v_top + rc + rc * np.sin(phi)], 1)
    phi2 = np.linspace(1.5 * math.pi, 2.0 * math.pi, n_arc + 1)[:-1]
    right = np.stack([xr - rc + rc * np.cos(phi2), v_top + rc + rc * np.sin(phi2)], 1)
    face = np.stack([np.linspace(left[-1, 0], right[0, 0], n_face + 2)[1:-1], np.full(n_face, v_top)], 1)
    return np.concatenate([np.stack([x, v], 1), left, face, right])


def _smooth(k):
    k = min(max(k, 0.0), 1.0)
    return k * k * (3.0 - 2.0 * k)


def neck_stations():
    """[(z, xc, a, d, p, rc)] along the neck from the heel to the headstock's tip."""
    out = []
    for z in np.linspace(L.Z_HEEL, Z_NUT_BACK, 17):
        w, d = L.section_wd(z)
        out.append((float(z), 0.0, 0.5 * float(w), float(d), L.P_SECTION, 0.0008))
    us = np.array(list(np.arange(L.NUT_T + 0.004, 0.150, 0.0045)) + list(np.linspace(0.150, L.HEAD_L - 0.0012, 14)) + [L.HEAD_L])
    xb, xt = head_extent(us)
    for u, lo, hi in zip(us, xb, xt):
        k = _smooth((u - L.NUT_T) / 0.030)
        out.append((L.Z_NUT + float(u), 0.5 * float(lo + hi), 0.5 * float(hi - lo), L.D_NUT,
                    L.P_SECTION + (L.HEAD_P - L.P_SECTION) * k, 0.0008 + 0.0012 * k))
    return out


def neck_mesh():
    """The neck wood with its heel and headstock: one closed mesh, role `neck`."""
    rings = []
    for z, xc, a, d, p, rc in neck_stations():
        c = ring_curve(xc, a, d, p, rc)
        rings.append(np.column_stack([c[:, 0], L.Y_BOARD + c[:, 1], np.full(len(c), z)]))
    return outward(S.skin(np.stack(rings), caps=(True, True), closed=True, mat=M["neck"]))


# ===================================================================================================================
# the board, wires, nut, markers
# ===================================================================================================================
def board_ring(w, d, rb=0.0011, ru=0.0010, n_top=13, n_arc=3, n_wall=3):
    """The board's section (m, 2) of (x, v), counter-clockwise with v up: the underside (v = BOARD_T, +x to -x) with rolled
    corners, up the -x wall (which continues the neck's superellipse), the rolled -x edge, the radiused top (v = sag), the +x
    edge, down the +x wall."""
    edge = lambda v: float(L.back_half_width(v, w, d))                           # noqa: E731
    vc = float(L.board_sag(0.5 * w)) + rb
    for _ in range(3):
        xw = edge(vc)
        vc = float(L.board_sag(xw - rb)) + rb
    xw = edge(vc)
    x_arc = np.linspace(-(xw - rb), xw - rb, n_top)
    top = np.stack([x_arc, L.board_sag(x_arc)], 1)
    phi = np.linspace(1.5 * math.pi, 2.0 * math.pi, n_arc + 1)[1:]
    right = np.stack([xw - rb + rb * np.cos(phi), vc + rb * np.sin(phi)], 1)      # from the top down to the wall
    right[-1, 0] = xw
    v_u = L.BOARD_T - ru
    wv = np.linspace(vc, v_u, n_wall + 1)[1:]
    right_wall = np.stack([[edge(v) for v in wv], wv], 1)                        # down to where the underside rolls
    xu = edge(v_u)
    phi_u = np.linspace(0.0, 0.5 * math.pi, n_arc + 1)[1:]
    right_under = np.stack([xu - ru + ru * np.cos(phi_u), v_u + ru * np.sin(phi_u)], 1)    # round the corner onto the underside
    mirror = np.array([-1.0, 1.0])
    ring = np.concatenate([right_under[-1:], right_under[::-1] * mirror, right_wall[::-1] * mirror, right[::-1] * mirror,
                           top, right, right_wall, right_under[:-1]])
    return ring


def board_mesh():
    """The rosewood slab (closed, role `fretboard`) with the pearloid dots set into it (role `inlay`), in one mesh."""
    rings = []
    for z in np.linspace(L.Z_NUT, L.Z_HEEL, 12):
        w, d = (float(q) for q in L.section_wd(z))
        r = board_ring(w, d)
        rings.append(np.column_stack([r[:, 0], L.Y_BOARD + r[:, 1], np.full(len(r), z)]))
    slab = outward(S.skin(np.stack(rings), caps=(True, True), closed=True, mat=M["fretboard"]))
    return S.merge([slab, inlay_mesh()])


def inlay_mesh():
    prof = S.fillet(np.array([(0.0, 0.0), (0.0032, 0.0), (0.0032, 0.0002), (0.0, 0.0006)]), 0.0005, 3)
    dot = S.lathe(prof, seg=20, mat=M["inlay"])
    out = []
    for n in (3, 5, 7, 9, 12, 15, 17, 19, 21):
        z = 0.5 * float(L.fret_z(n - 1) + L.fret_z(n))
        for x in ((-0.0105, 0.0105) if n == 12 else (0.0,)):
            out.append(on_face(dot, x, L.Y_BOARD + float(L.board_sag(x)) + 0.0001, z))
    return S.merge(out)


def fret_section():
    """The wire's section in (u along the neck, v out of the board): a crown, 0.4 mm of it embedded."""
    hw, e = 0.5 * L.FRET_W, 0.0004
    b = L.FRET_H - e
    t = np.linspace(0.0, math.pi, 9)[1:-1]
    return np.concatenate([[(-hw, -e), (hw, -e), (hw, e)], np.stack([hw * np.cos(t), e + b * np.sin(t)], 1), [(-hw, e)]])


def frets_mesh():
    """The 22 fret wires: swept across the board along its radius, their ends rounded off. Role `frets`."""
    out = []
    sec = fret_section()
    n = 17
    taper = np.ones(n)
    taper[[0, -1]], taper[[1, -2]] = 0.55, 0.85
    for k in range(1, L.N_FRETS + 1):
        z = float(L.fret_z(k))
        w = float(L.section_wd(z)[0])
        xs = np.linspace(-(0.5 * w - 0.0004), 0.5 * w - 0.0004, n)
        path = np.stack([xs, L.Y_BOARD + L.board_sag(xs), np.full(n, z)], 1)
        out.append(outward(S.sweep(path, sec, scale=taper, up=OUT, mat=M["frets"])))
    return S.merge(out)


def nut_mesh():
    """The bone nut: a bar whose top follows the board's radius, rounded all round. Role `inlay`."""
    h = L.NUT_H + L.BOARD_T + 0.0002
    sec = S.rrect(L.NUT_T, h, 0.0008, 3)
    sec = sec + np.array([0.0, 0.5 * (L.NUT_H - L.BOARD_T - 0.0002)])
    n = 11
    half = 0.5 * L.W_NUT - 0.0002
    xs = np.linspace(-half, half, n)
    path = np.stack([xs, L.Y_BOARD + L.board_sag(xs), np.full(n, L.Z_NUT + 0.5 * L.NUT_T)], 1)
    return outward(S.sweep(path, sec, up=OUT, mat=M["inlay"]))


# ===================================================================================================================
# the headstock's fittings
# ===================================================================================================================
def _lathe(pts, r=0.0005, seg=20, mat=0, n=3):
    return S.lathe(S.fillet(np.array(pts, float), r, n), seg=seg, mat=mat)


def tuner_parts():
    """One machine head's pieces in its own frame (z out of the face): the bushing with its post (role hardware)."""
    return _lathe([(0.0, 0.0), (0.0063, 0.0), (0.0063, 0.0016), (0.0056, 0.0022), (L.POST_R + 0.0001, 0.0024), (L.POST_R, 0.0082),
                   (0.0027, 0.0090), (0.0, 0.0093)], 0.0004, 24, M["hardware"])


def key_dir():
    d = np.array(L.KEY_DIR, float)
    return d / np.linalg.norm(d)


def tuners_mesh():
    """Six machine heads on the bass edge (a post and bushing on the face, a collar and housing on the back, a shaft out of
    the headstock's edge, a paddle button), and two round string retainers. Role `hardware`."""
    hw = M["hardware"]
    post = tuner_parts()
    collar = _lathe([(0.0, 0.0), (0.0050, 0.0), (0.0050, 0.0010), (0.0046, 0.0013), (0.0, 0.0014)], 0.0003, 20, hw)
    housing = S.rounded_panel(S.stadium(0.0105, 0.0200, 6), 0.0030, r_edge=0.0009, n=2, dome=0.0003, mat=hw)
    button = S.rounded_panel(S.superellipse(0.0112, 0.0150, 3.2, 28), 0.0050, r_edge=0.0014, n=3, dome=0.0004, mat=hw)
    kd = key_dir()
    out = []
    for z, x in L.POSTS:
        out.append(on_face(post, x, Y_FACE, z))
        out.append(on_face(collar, x, Y_BACK, z, (0.0, 1.0, 0.0)))
        c = np.array([x, Y_BACK, z]) + kd * 0.0085
        out.append(S.place(housing, c, (0.0, 1.0, 0.0), kd))
        p0 = np.array([x, Y_MID, z])
        out.append(S.tube(np.stack([p0, p0 + kd * (L.KEY_LEN - 0.002)]), 0.0017, sides=10, up=OUT, mat=hw))
        out.append(S.place(button, p0 + kd * L.KEY_LEN + np.array([0.0, 0.5 * 0.0050, 0.0]), OUT, kd))
    stud = _lathe([(0.0, 0.0), (0.0021, 0.0), (0.0021, 0.0064), (0.0041, 0.0064), (0.0041, 0.0078), (0.0030, 0.0084),
                   (0.0, 0.0085)], 0.0004, 18, hw)
    for z, x in L.TREES:
        out.append(on_face(stud, x, Y_FACE, z))
    return S.merge(out)


def truss_mesh():
    """The truss rod's cover: a bullet-shaped plastic plate on the headstock face just past the nut. Role `knobs`."""
    w, h = L.TRUSS["size"]
    plate = S.rounded_panel(S.stadium(w, h, 6), 0.0012, r_edge=0.0004, n=2, dome=0.0003, mat=M["knobs"])
    x, z = L.TRUSS["center"]
    return on_face(plate, x, Y_FACE + 0.0001, z)


# ===================================================================================================================
# the strings
# ===================================================================================================================
def post_wraps(i, y_start, turns=2.5, n_per_turn=14):
    """The points (k, 3) where string i touches its post, from the tangent point of the straight run from the nut round the post
    counter-clockwise (seen from the front), sinking toward the headstock's face by a string diameter a turn."""
    r = L.STRING_R[i]
    pz, px = L.POSTS[i]
    rho = L.POST_R + r
    nx, nz = L.NUT_X[i], Z_NUT_BACK
    d = np.array([px - nx, pz - nz])
    dist = float(np.linalg.norm(d))
    off = math.acos(min(rho / dist, 1.0))
    for sgn in (1.0, -1.0):                            # the tangent point whose counter-clockwise tangent runs along the string
        th0 = math.atan2(d[1], d[0]) + math.pi + sgn * off
        t = np.array([px + rho * math.cos(th0) - nx, pz + rho * math.sin(th0) - nz])
        if t @ np.array([-math.sin(th0), math.cos(th0)]) > 0.0:
            break
    th = th0 + np.linspace(0.0, 2.0 * math.pi * turns, int(turns * n_per_turn) + 1)
    pitch = max(2.0 * r * 1.08, 0.0006)
    y = y_start + (th - th0) / (2.0 * math.pi) * pitch
    return np.stack([px + rho * np.cos(th), y, pz + rho * np.sin(th)], 1)


def strings_mesh():
    """Six strings as thin tubes (role `strings`): from the saddle over the nut to a post, which they are wound round 2.5 times,
    and down into the bridge plate behind the saddle on the other side. The part between the saddle and the nut's front edge
    lies EXACTLY on the card's nut and bridge points."""
    out = []
    for i in range(6):
        nut, sad = (np.array(p) for p in L.string_ends(i))
        back = np.array([nut[0], nut[1], Z_NUT_BACK])
        wraps = post_wraps(i, Y_FACE - 0.0062)
        tail = np.array([sad[0], -L.BRIDGE["t"] + 0.0002, L.BRIDGE_HOLE_Z])
        path = np.concatenate([wraps[::-1], [back, nut, sad, tail]])
        out.append(S.tube(path, L.STRING_R[i], sides=6, up=OUT, mat=M["strings"]))
    return S.merge(out)
