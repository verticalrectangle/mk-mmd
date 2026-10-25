"""The bass's neck, headstock and everything on them (pure numpy, no bpy), built as the guitar's are (electric_guitar_neck).

The NECK is one skin (`core.shell.skin`) of ring sections along z from the heel over the nut and through the headstock to its
tip. Every ring is the lower half of a superellipse |2x/w|^p + (v/d)^p = 1 (v = depth below the board's crown plane) cut at the
board's underside (v = BOARD_T): along the neck that is EXACTLY the card's `section` (w, d from `electric_bass_layout.section_wd`,
p = 2.6). Past the nut the ring's centre and width follow the Fender bass headstock's outline and its exponent grows to 6 (a
rounded rectangle) while its depth stays that of the nut, so the back of the C flows into the headstock's flat back and the
headstock's face is the plane the board is glued on.

    neck_mesh, board_mesh, frets_mesh, nut_mesh, inlay_mesh       the wood, the radiused rosewood slab, the wires, the bone
    tuners_mesh, tree_mesh, strings_mesh                          four bass machine heads with clover keys, the string
                                                                  retainer of the D and G strings, the strings
"""
import math

import numpy as np

from ....core import shell as S
from . import electric_bass_layout as L
from . import electric_guitar_neck as GN

M = L.MATS
Y_FACE = L.Y_BOARD + L.BOARD_T                    # the headstock's face (the neck wood's top, under the board)
Y_BACK = L.Y_BOARD + L.D_NUT                      # the headstock's back, flush with the back of the neck at the nut
Y_MID = 0.5 * (Y_FACE + Y_BACK)
Z_NUT_BACK = L.Z_NUT_BACK
FACE_UP = (0.0, 0.0, 1.0)
OUT = (0.0, -1.0, 0.0)                            # out of the face
U_END = L.HEAD_L - 0.0006                         # the last ring of the headstock, just short of its tip


def outward(mesh):
    return mesh if mesh.volume() > 0 else mesh.flip()


def on_face(mesh, x, y, z, normal=OUT, up=FACE_UP):
    """A part modelled with z out of the surface it stands on, x/y across it, put at (x, y, z) on a surface facing `normal`."""
    return S.place(mesh, (x, y, z), normal, up)


def head_extent(u):
    """(x_bass, x_treble) of the headstock's outline at distance u beyond the nut's front face (m), arrays."""
    bass, treble = L.head_outline()
    u = np.clip(np.atleast_1d(np.asarray(u, float)), 0.0, L.HEAD_L)
    xb = S.pchip(bass[:, 0], bass[:, 1:2], u)[:, 0]
    xt = S.pchip(treble[:, 0], treble[:, 1:2], u)[:, 0]
    return xb, xt


# ===================================================================================================================
# the neck
# ===================================================================================================================
def _smooth(k):
    k = min(max(k, 0.0), 1.0)
    return k * k * (3.0 - 2.0 * k)


def neck_stations():
    """[(z, xc, a, d, p, rc)] along the neck from the heel to the headstock's tip."""
    out = []
    for z in np.linspace(L.Z_HEEL, Z_NUT_BACK, 19):
        w, d = L.section_wd(z)
        out.append((float(z), 0.0, 0.5 * float(w), float(d), L.P_SECTION, 0.0008))
    us = np.array(list(np.arange(L.NUT_T + 0.004, 0.190, 0.0050)) + list(np.linspace(0.190, U_END - 0.0012, 16)) + [U_END])
    xb, xt = head_extent(us)
    for u, lo, hi in zip(us, xb, xt):
        k = _smooth((u - L.NUT_T) / 0.030)
        out.append((L.Z_NUT + float(u), 0.5 * float(lo + hi), 0.5 * float(hi - lo), L.D_NUT,
                    L.P_SECTION + (L.HEAD_P - L.P_SECTION) * k, 0.0008 + 0.0014 * k))
    return out


def neck_mesh():
    """The neck wood with its heel and headstock: one closed mesh, role `neck`."""
    rings = []
    for z, xc, a, d, p, rc in neck_stations():
        c = GN.ring_curve(xc, a, d, p, rc, v_top=L.BOARD_T)
        rings.append(np.column_stack([c[:, 0], L.Y_BOARD + c[:, 1], np.full(len(c), z)]))
    return outward(S.skin(np.stack(rings), caps=(True, True), closed=True, mat=M["neck"]))


# ===================================================================================================================
# the board, wires, nut, markers
# ===================================================================================================================
def board_ring(w, d, rb=0.0011, ru=0.0010, n_top=13, n_arc=3, n_wall=3):
    """The board's section (m, 2) of (x, v), counter-clockwise with v up: the underside (v = BOARD_T) with rolled corners, up
    the -x wall (which continues the neck's superellipse), the rolled -x edge, the radiused top (v = sag), the +x edge, down
    the +x wall (electric_guitar_neck.board_ring with this neck's radius and section)."""
    edge = lambda v: float(L.back_half_width(v, w, d))                           # noqa: E731
    vc = float(L.board_sag(0.5 * w)) + rb
    for _ in range(3):
        xw = edge(vc)
        vc = float(L.board_sag(xw - rb)) + rb
    xw = edge(vc)
    x_arc = np.linspace(-(xw - rb), xw - rb, n_top)
    top = np.stack([x_arc, L.board_sag(x_arc)], 1)
    phi = np.linspace(1.5 * math.pi, 2.0 * math.pi, n_arc + 1)[1:]
    right = np.stack([xw - rb + rb * np.cos(phi), vc + rb * np.sin(phi)], 1)
    right[-1, 0] = xw
    v_u = L.BOARD_T - ru
    wv = np.linspace(vc, v_u, n_wall + 1)[1:]
    right_wall = np.stack([[edge(v) for v in wv], wv], 1)
    xu = edge(v_u)
    phi_u = np.linspace(0.0, 0.5 * math.pi, n_arc + 1)[1:]
    right_under = np.stack([xu - ru + ru * np.cos(phi_u), v_u + ru * np.sin(phi_u)], 1)
    mirror = np.array([-1.0, 1.0])
    return np.concatenate([right_under[-1:], right_under[::-1] * mirror, right_wall[::-1] * mirror, right[::-1] * mirror,
                           top, right, right_wall, right_under[:-1]])


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
    prof = S.fillet(np.array([(0.0, 0.0), (0.0035, 0.0), (0.0035, 0.0002), (0.0, 0.0006)]), 0.0005, 3)
    dot = S.lathe(prof, seg=20, mat=M["inlay"])
    out = []
    for n in L.DOTS:
        z = 0.5 * float(L.fret_z(n - 1) + L.fret_z(n))
        for x in ((-0.0110, 0.0110) if n == 12 else (0.0,)):
            out.append(on_face(dot, x, L.Y_BOARD + float(L.board_sag(x)) + 0.0001, z))
    return S.merge(out)


def fret_section():
    """The wire's section in (u along the neck, v out of the board): a crown, 0.4 mm of it embedded."""
    hw, e = 0.5 * L.FRET_W, 0.0004
    b = L.FRET_H - e
    t = np.linspace(0.0, math.pi, 9)[1:-1]
    return np.concatenate([[(-hw, -e), (hw, -e), (hw, e)], np.stack([hw * np.cos(t), e + b * np.sin(t)], 1), [(-hw, e)]])


def frets_mesh():
    """The 19 fret wires: swept across the board along its radius, their ends rounded off. Role `frets`."""
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
    sec = S.rrect(L.NUT_T, h, 0.0009, 3)
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


def post_mesh():
    """A bass machine head's face side in its own frame (z out of the face): a wide bushing and the thick string post with
    a domed top (role hardware)."""
    return _lathe([(0.0, 0.0), (0.0072, 0.0), (0.0072, 0.0014), (0.0064, 0.0021), (L.POST_R + 0.0002, 0.0024),
                   (L.POST_R, 0.0150), (0.0036, 0.0162), (0.0, 0.0166)], 0.0005, 28, M["hardware"])


def clover_outline(n=72):
    """The clover key's outline (n, 2) in its own frame: three round lobes (out at -x, up and down), the narrow stem toward
    the shaft at +x."""
    t = 2.0 * math.pi * np.arange(n) / n
    r = 0.0124 + 0.0019 * np.cos(4.0 * t) - 0.0034 * (0.5 + 0.5 * np.cos(t)) ** 3
    return np.stack([r * np.cos(t), r * np.sin(t)], 1)


def key_point(i):
    """(gear centre on the back, key button centre) of tuner i (string 4 = E first): the worm runs from the gear toward the
    tip of the headstock by KEY_OFFSET and out of its bass edge along KEY_DIR."""
    pz, px = L.POSTS[i]
    zk = pz + L.KEY_OFFSET * L.HEAD_K
    gear = np.array([px - 0.0040, Y_BACK, pz + 0.0110])
    edge = float(head_extent(zk - L.Z_NUT)[0][0])
    key = np.array([min(px - L.KEY_LEN, edge - 0.0215), Y_BACK + 0.0040, zk])
    return gear, key


def tuners_mesh():
    """Four bass machine heads on the bass edge: a post and bushing on the face, a gear housing on the back, a worm shaft out
    of the headstock's bass edge, and a clover key button. Role `hardware`."""
    hw = M["hardware"]
    post = post_mesh()
    housing = S.rounded_panel(S.rrect(0.0230, 0.0270, 0.0045, 4), 0.0085, r_edge=0.0022, n=3, dome=0.0006, mat=hw)
    button = S.rounded_panel(clover_outline(), 0.0034, r_edge=0.0012, n=3, dome=0.0004, r_back=0.0010, mat=hw)
    hub = _lathe([(0.0, 0.0), (0.0046, 0.0), (0.0046, 0.0060), (0.0030, 0.0080), (0.0, 0.0080)], 0.0006, 18, hw)
    out = []
    kd = np.array(L.KEY_DIR, float)
    for i, (z, x) in enumerate(L.POSTS):
        out.append(on_face(post, x, Y_FACE, z))
        gear, key = key_point(i)
        out.append(S.place(housing, gear - np.array([0.0, 0.0002, 0.0]), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)))
        p0 = np.array([gear[0], key[1], key[2]])
        out.append(S.tube(np.stack([p0, key - kd * 0.0110]), 0.0019, sides=12, up=OUT, mat=hw))
        out.append(S.place(hub, key - kd * 0.0170, kd, OUT))
        c = S.place(button, (0.0, 0.0, 0.0), OUT, (0.0, 0.0, 1.0))           # its flat faces parallel to the headstock's
        out.append(c.moved(key - np.array([0.0, 0.5 * 0.0034, 0.0])))
    return S.merge(out)


def tree_mesh():
    """The round string retainer of the D and G strings on the headstock's face (role `hardware`)."""
    stud = _lathe([(0.0, 0.0), (0.0024, 0.0), (0.0024, 0.0070), (0.0050, 0.0070), (0.0050, 0.0086), (0.0036, 0.0093),
                   (0.0, 0.0094)], 0.0004, 20, M["hardware"])
    z, x = L.TREE
    return on_face(stud, x, Y_FACE, z)


# ===================================================================================================================
# the strings
# ===================================================================================================================
TREE_DX = 0.0034                                     # the D and G strings pass this far either side of the retainer's stud


def post_wraps(i, y_start, turns=1.5, n_per_turn=16):
    """The points (k, 3) where string i touches its post, from the tangent point of the straight run from the nut (or the
    string retainer) round the post counter-clockwise (seen from the front), sinking toward the headstock's face by a string
    diameter a turn (electric_guitar_neck.post_wraps)."""
    r = L.STRING_R[i]
    pz, px = L.POSTS[i]
    rho = L.POST_R + r
    sx, sz = guide(i)
    d = np.array([px - sx, pz - sz])
    dist = float(np.linalg.norm(d))
    off = math.acos(min(rho / dist, 1.0))
    for sgn in (1.0, -1.0):                            # the tangent point whose counter-clockwise tangent runs along the string
        th0 = math.atan2(d[1], d[0]) + math.pi + sgn * off
        t = np.array([px + rho * math.cos(th0) - sx, pz + rho * math.sin(th0) - sz])
        if t @ np.array([-math.sin(th0), math.cos(th0)]) > 0.0:
            break
    th = th0 + np.linspace(0.0, 2.0 * math.pi * turns, int(turns * n_per_turn) + 1)
    pitch = max(2.0 * r * 1.08, 0.0008)
    y = y_start + (th - th0) / (2.0 * math.pi) * pitch
    return np.stack([px + rho * np.cos(th), y, pz + rho * np.sin(th)], 1)


def guide(i):
    """(x, z) the straight run to string i's post starts from: beside the string retainer for the D and G, else the nut's
    back."""
    if i >= 2:
        z, x = L.TREE
        return x + (TREE_DX if i == 3 else -TREE_DX), z
    return L.NUT_X[i], Z_NUT_BACK


def strings_mesh():
    """Four strings as thin tubes (role `strings`): from the saddle over the nut (the D and G under the retainer's washer) to
    a post, which they are wound round one and a half times, and down through the bridge plate behind the saddle on the
    other side. The part between the saddle and the nut's front edge lies EXACTLY on the card's nut and bridge points."""
    out = []
    z_tree = L.TREE[0]
    for i in range(4):
        nut, sad = (np.array(p) for p in L.string_ends(i))
        back = np.array([nut[0], nut[1], Z_NUT_BACK])
        wraps = post_wraps(i, Y_FACE - 0.0080)
        tail = np.array([sad[0], -L.BRIDGE["t"] + 0.0002, L.STRING_HOLE_Z])
        run = [back, nut, sad, tail]
        if i >= 2:                                      # under the retainer's washer (7 mm over the face)
            gx, _gz = guide(i)
            run = [np.array([gx, Y_FACE - 0.0070 + L.STRING_R[i], z_tree])] + run
        path = np.concatenate([wraps[::-1], run])
        out.append(S.tube(path, L.STRING_R[i], sides=8, up=OUT, mat=M["strings"]))
    return S.merge(out)
