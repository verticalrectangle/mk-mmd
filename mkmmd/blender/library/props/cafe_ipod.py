"""Table-top props of the rainy cafe: an iPod classic, a ceramic bud vase with a silver-dollar eucalyptus sprig, and a
pair of white wired earbuds whose cord runs from the table's near edge into the iPod's headphone socket.

    cafe_ipod      the iPod lying screen up. Origin = centre of the bottom face (the contact point on the table top),
                   +X right, +Y the top edge (headphone socket side), +Z out of the screen. ONE mesh with 5 materials
                   (acrylic, chrome, screen, click wheel, recessed slots) and the vertex colour layer `ipcol` that the
                   screen and wheel materials read (the now-playing screen and the wheel marks are painted decals).
    cafe_vase      the ceramic bud vase. Origin = centre of the foot on the table top, +Z up. Its child `sprig` is the
                   eucalyptus (one mesh, 2 materials, vertex colour layer `ipleaf`), origin = centre of the vase mouth.
    cafe_earbuds   the earbuds, built in the WORLD-ALIGNED cafe frame: place the root at the origin with yaw 0.

Colours are blends of the palette slots (Kit.blend), fitted on Rose Pine Dawn; meshes, UV layers, vertex colours and the
procedural materials reproduce the reference props."""
import math
import random
from types import SimpleNamespace

import bmesh
import bpy
from mathutils import Vector
from mathutils import geometry as geo

from . import register
from .cafe_kit import Kit, L, N, bm_box, bm_lathe, bm_tube, bm_uv_sphere, catmull, principled

TABLE_Z = 0.74          # table-top height of the cafe layout the default positions below belong to


# ============================================================================================================== maths
def _rrect(cx, cy, w, h, r, n):
    """Counter-clockwise rounded-rectangle outline: arcs BR, TR, TL, BL with n+1 points each (so consecutive arcs are
    joined by the straight runs j = n, 2n+1, 3n+2, 4n+3 -> right, top, left, bottom)."""
    r = max(min(r, 0.5 * w - 1e-7, 0.5 * h - 1e-7), 1e-7)
    out = []
    for ax, ay, a0 in ((cx + 0.5 * w - r, cy - 0.5 * h + r, -90.0), (cx + 0.5 * w - r, cy + 0.5 * h - r, 0.0),
                       (cx - 0.5 * w + r, cy + 0.5 * h - r, 90.0), (cx - 0.5 * w + r, cy - 0.5 * h + r, 180.0)):
        for k in range(n + 1):
            a = math.radians(a0 + 90.0 * k / n)
            out.append((ax + r * math.cos(a), ay + r * math.sin(a)))
    return out


def _circle(cx, cy, r, n):
    return [(cx + r * math.cos(2 * math.pi * i / n), cy + r * math.sin(2 * math.pi * i / n)) for i in range(n)]


# =============================================================================================================== iPod
IPOD_L, IPOD_W, IPOD_T = 0.1035, 0.0618, 0.0105          # length (local Y), width (local X), thickness (local Z)
IPOD_JACK_X, IPOD_HOLD_X = -0.0165, 0.0125               # local x of the headphone socket / hold switch (top edge)
_ZC = -0.00015                                           # z (about the body centre) of the chrome wall's middle
IPOD_PLUG = (IPOD_JACK_X, IPOD_L / 2, IPOD_T / 2 + _ZC)  # centre of the headphone socket opening, prop frame
IPOD_PLUG_DIR = (0.0, 1.0, 0.0)                          # pointing out of the socket

# the mesh is modelled about the body centre and moved up by half the thickness at the end (origin on the bottom face)
_R = 0.0085                      # plan corner radius of the body
_NA = 8                          # arc segments per corner of the body outline
_WIN_W, _WIN_H, _WIN_R = 0.0510, 0.0390, 0.0040          # screen window (glass recess)
_WIN_CY = 0.02685
_WHL_CY, _WHL_R = -0.0210, 0.0195                        # click wheel centre / hole radius
_ZT, _ZB = IPOD_T / 2, -IPOD_T / 2
_ZR = _ZT - 0.00035              # click-wheel ring surface (just below the plate)
_ZG = _ZR - 0.0004               # groove floor
_ZBTN = _ZR + 0.0002             # centre button top rim
_ZF = _ZT - 0.0004               # screen recess floor
_PX = 0.126875                   # mm per UI pixel (40.6 mm active width / 320 px)
_SCREEN = (320 * _PX * 1e-3, 240 * _PX * 1e-3)           # the active picture (m)

# where the iPod lies in the reference layout (the earbuds' default route ends at its socket)
IPOD_LAYOUT_AT = (0.1950231, -0.6100154, TABLE_Z)
IPOD_LAYOUT_YAW = 0.3265814      # rad: the top edge aims at the seated figure's chest


def _ipod_colors(K):
    b, s = K.blend, K.slot
    return SimpleNamespace(
        acrylic=s("base"), chrome=b(hl_high=1.0, k=1.4), slot=s("muted"),
        bezel=b(muted=1.0, k=1.25),
        paper0=s("surface"), paper1=s("overlay"),
        head0=s("base"), head1=b(surface=0.7, iris=0.3), rule=b(surface=0.5, iris=0.5),
        play=s("rose"), title=s("subtle"), batt=s("subtle"), batt_in=b(base=0.85, iris=0.15), batt_fill=s("foam"),
        art_frame=s("hl_med"), sky0=s("iris"), sky1=b(rose=0.7, hl_med=0.3, chroma=0.8, k=1.3),
        sun=b(gold=0.8, surface=0.2, hue=10, chroma=1.1, k=1.1), hill=s("foam"), hill2=s("pine"),
        artist=s("muted"), album=b(subtle=0.6, hl_high=0.4, k=1.25),
        track=s("hl_med"), fill0=s("rose"), fill1=b(love=0.65, rose=0.35), knob=s("love"), knob_in=s("surface"),
        time=s("muted"), icon=b(surface=0.5, iris=0.5), icon2=b(subtle=0.6, hl_high=0.4, k=1.25),
        ring=s("hl_high"), ink=b(text=0.8, surface=0.2), groove=b(subtle=0.6, hl_high=0.4, k=1.3),
        button=s("hl_med"))


# ----------------------------------------------------------------------------------------------- iPod: mesh helpers
def _face(bm, verts, want, mat, smooth=False):
    """Create a face and flip it if its normal opposes `want`.  Returns None for degenerate/duplicate faces."""
    try:
        f = bm.faces.new(verts)
    except ValueError:
        return None
    f.normal_update()
    if f.normal.dot(want) < 0:
        f.normal_flip()
    f.material_index = mat
    f.smooth = smooth
    return f


def _paint(bm, faces, col, layer="ipcol"):
    lay = bm.loops.layers.float_color.get(layer) or bm.loops.layers.float_color.new(layer)
    for f in faces:
        if f is not None:
            for lp in f.loops:
                lp[lay] = col


def _col_at(spec, x, y):
    """Colour spec: linear rgba tuple, or ('g', (x0, y0), (x1, y1), rgba0, rgba1) linear gradient."""
    if spec[0] != "g":
        return spec
    _, (x0, y0), (x1, y1), c0, c1 = spec
    dx, dy = x1 - x0, y1 - y0
    t = ((x - x0) * dx + (y - y0) * dy) / (dx * dx + dy * dy)
    t = min(max(t, 0.0), 1.0)
    return tuple(a * (1 - t) + b * t for a, b in zip(c0, c1))


def _decal(bm, shapes, org, z, mat, layer="ipcol"):
    """Flat painter's-algorithm decal.  shapes = [(polygon_mm [(x, y)], colour_spec, drop)] in paint order (later = on
    top); the shapes are resolved with a constrained Delaunay arrangement so that the result is coplanar and
    non-overlapping (no z-fighting).  Triangles whose topmost shape has drop=True are discarded.  org = (x, y) in
    metres of mm-origin."""
    verts, faces = [], []
    for poly, _c, _d in shapes:
        pts = list(poly)
        area = sum(pts[i][0] * pts[(i + 1) % len(pts)][1] - pts[(i + 1) % len(pts)][0] * pts[i][1]
                   for i in range(len(pts)))
        if area < 0:
            pts = pts[::-1]
        idx = []
        for p in pts:
            verts.append(Vector(p))
            idx.append(len(verts) - 1)
        faces.append(idx)
    vc, _e, fc, _ov, _oe, of = geo.delaunay_2d_cdt(verts, [], faces, 1, 1e-5)
    lay = bm.loops.layers.float_color.get(layer) or bm.loops.layers.float_color.new(layer)
    cache = {}

    def bv(i):
        if i not in cache:
            cache[i] = bm.verts.new((org[0] + vc[i].x * 1e-3, org[1] + vc[i].y * 1e-3, z))
        return cache[i]
    made = []
    for tri, orig in zip(fc, of):
        if not orig:
            continue
        top = max(orig)
        _p, col, drop = shapes[top]
        if drop:
            continue
        P = [vc[i] for i in tri]
        if (P[1].x - P[0].x) * (P[2].y - P[0].y) - (P[1].y - P[0].y) * (P[2].x - P[0].x) < 0:
            tri = list(tri)[::-1]
        try:
            f = bm.faces.new([bv(i) for i in tri])
        except ValueError:
            continue
        f.material_index = mat
        f.smooth = False
        for i, lp in zip(tri, f.loops):
            lp[lay] = _col_at(col, vc[i].x, vc[i].y)
        made.append(f)
    return made


def _pocket(bm, org, ex, ey, nrm, outline, depth, cham, mat, smooth=False):
    """Cut-in pocket (walls + floor, faces looking into the void).  `outline(inset)` -> [(u, v)] CCW opening outline in
    the wall plane through `org` (u along ex, v along ey, nrm = outward wall normal).  Returns the opening ring verts
    (shared with the surrounding wall polygon).  smooth=True (round sockets) shades the walls smooth with crisp rims."""
    def ring(inset, dep):
        return [bm.verts.new(org + ex * u + ey * v - nrm * dep) for (u, v) in outline(inset)]
    r0, r1, r2 = ring(0.0, 0.0), ring(cham, cham * 0.9), ring(cham, depth)
    pts = outline(0.0)
    ax = org + ex * (sum(p[0] for p in pts) / len(pts)) + ey * (sum(p[1] for p in pts) / len(pts))
    M = len(r0)
    for a, b, up in ((r0, r1, 0.7), (r1, r2, 0.0)):
        for j in range(M):
            vs = (a[j], a[(j + 1) % M], b[(j + 1) % M], b[j])
            c = sum((v.co for v in vs), Vector()) / 4
            d = ax - c
            d -= nrm * d.dot(nrm)
            _face(bm, vs, d + nrm * up, mat, smooth=smooth)
    _face(bm, r2, nrm, mat)
    if smooth:                               # keep the countersink / bore / floor rims crisp
        for rg in (r0, r1, r2):
            for j in range(M):
                e = bm.edges.get((rg[j], rg[(j + 1) % M]))
                if e is not None:
                    e.smooth = False
    return r0


def _boss(bm, org, ex, ey, nrm, outline, base_dep, top_dep, mat):
    """Raised pad (hold-switch slider): walls facing away from its axis + top face."""
    def ring(dep):
        return [bm.verts.new(org + ex * u + ey * v - nrm * dep) for (u, v) in outline(0.0)]
    rb, rt = ring(base_dep), ring(top_dep)
    pts = outline(0.0)
    ax = org + ex * (sum(p[0] for p in pts) / len(pts)) + ey * (sum(p[1] for p in pts) / len(pts))
    M = len(rb)
    for j in range(M):
        vs = (rb[j], rb[(j + 1) % M], rt[(j + 1) % M], rt[j])
        c = sum((v.co for v in vs), Vector()) / 4
        d = c - ax
        d -= nrm * d.dot(nrm)
        _face(bm, vs, d, mat, smooth=True)
    _face(bm, rt, nrm, mat)
    _face(bm, rb, -nrm, mat)


def _ui_shapes(c):
    """Abstract now-playing screen (mm, origin = window centre, y up).  Painter's order."""
    P = _PX

    def X(px):
        return (px - 160.0) * P

    def Y(py):
        return (120.0 - py) * P

    def rr(x0, y0, x1, y1, r, n=4):
        return [(X(px), Y(py)) for px, py in _rrect(0.5 * (x0 + x1), 0.5 * (y0 + y1), x1 - x0, y1 - y0, r, n)]

    def cap(x0, y0, x1, y1, n=5):
        return rr(x0, y0, x1, y1, 0.5 * (y1 - y0), n)

    def circ(cx, cy, r, n=28):
        return [(X(px), Y(py)) for px, py in _circle(cx, cy, r, n)]

    def poly(*pp):
        return [(X(px), Y(py)) for px, py in pp]

    def grad(x0, y0, x1, y1, ca, cb):
        return ("g", (X(x0), Y(y0)), (X(x1), Y(y1)), ca, cb)

    wi, hi = _WIN_W * 1e3 - 0.6, _WIN_H * 1e3 - 0.6
    return [
        (_rrect(0, 0, wi, hi, _WIN_R * 1e3 - 0.3, 6), c.bezel, False),                              # bezel
        (rr(0, 0, 320, 240, 4, 3), grad(0, 0, 0, 240, c.paper0, c.paper1), False),                  # paper-white bg
        (rr(0, 0, 320, 28, 0.1, 1), grad(0, 0, 0, 28, c.head0, c.head1), False),                    # header
        (rr(0, 28, 320, 30, 0.1, 1), c.rule, False),                                                # header rule
        (poly((13, 8), (13, 21), (24, 14.5)), c.play, False),                                       # play glyph
        (cap(116, 10, 204, 18.5), c.title, False),                                                  # title bar
        (rr(268, 8, 298, 20, 3, 3), c.batt, False),                                                 # battery
        (rr(270, 10, 296, 18, 1.6, 3), c.batt_in, False),
        (rr(271.8, 11.8, 291.0, 16.2, 1.0, 2), c.batt_fill, False),
        (rr(298, 11.5, 301.5, 16.5, 1.2, 2), c.batt, False),
        (rr(14, 42, 108, 136, 2.5, 3), c.art_frame, False),                                         # album art frame
        (rr(16, 44, 106, 134, 1.5, 3), grad(0, 44, 0, 100, c.sky0, c.sky1), False),                 # sky
        (circ(61, 90, 21), c.sun, False),                                                           # sun
        (poly((16, 134), (16, 108), (36, 99), (58, 108), (82, 97), (106, 105), (106, 134)), c.hill, False),
        (poly((16, 134), (16, 121), (40, 114), (70, 123), (106, 116), (106, 134)), c.hill2, False),
        (cap(126, 50, 268, 62), c.title, False),                                                    # title line
        (cap(126, 72, 236, 81), c.artist, False),                                                   # artist line
        (cap(126, 91, 252, 100), c.album, False),                                                   # album line
        (cap(24, 164, 296, 173), c.track, False),                                                   # progress track
        (cap(24, 164, 122, 173), grad(24, 0, 122, 0, c.fill0, c.fill1), False),                     # progress fill
        (circ(122, 168.5, 7.0), c.knob, False),                                                     # knob
        (circ(122, 168.5, 2.8), c.knob_in, False),
        (cap(24, 183, 60, 189), c.time, False),                                                     # elapsed, remaining
        (cap(260, 183, 296, 189), c.time, False),
        (rr(132, 203, 146, 217, 3, 3), c.icon, False),                                              # footer icons
        (rr(153, 203, 167, 217, 3, 3), c.icon2, False),
        (rr(174, 203, 188, 217, 3, 3), c.icon, False),
    ]


def _wheel_shapes(c):
    """Click-wheel ring face + printed marks (mm, origin = wheel centre).  The inner disc is dropped (button/groove)."""
    ra0, ra1 = (_WHL_R - 0.00012) * 1e3, 9.8
    ink = c.ink
    S = [(_circle(0, 0, ra0, 72), c.ring, False), (_circle(0, 0, ra1, 72), c.ring, True)]

    def bar(x0, y0, x1, y1):
        return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]

    def mir(pp):
        return [(-x, y) for x, y in pp][::-1]
    # MENU (4 glyph-like bars) - top
    xs = (-2.0, -0.85, 0.2, 1.35)
    for x, w in zip(xs, (0.55, 0.4, 0.5, 0.45)):
        S.append((bar(x, 14.1, x + w, 15.5), ink, False))
    # previous (bar + two left-pointing triangles) - left
    S.append((bar(-16.5, -0.8, -16.1, 0.8), ink, False))
    S.append(([(-15.9, 0.0), (-14.6, -0.85), (-14.6, 0.85)], ink, False))
    S.append(([(-14.5, 0.0), (-13.2, -0.85), (-13.2, 0.85)], ink, False))
    # next (mirrored) - right
    S.append((mir(bar(-16.5, -0.8, -16.1, 0.8)), ink, False))
    S.append((mir([(-15.9, 0.0), (-14.6, -0.85), (-14.6, 0.85)]), ink, False))
    S.append((mir([(-14.5, 0.0), (-13.2, -0.85), (-13.2, 0.85)]), ink, False))
    # play / pause - bottom
    S.append(([(-2.0, -15.55), (-2.0, -14.05), (-0.6, -14.8)], ink, False))
    S.append((bar(0.35, -15.5, 0.8, -14.1), ink, False))
    S.append((bar(1.3, -15.5, 1.75, -14.1), ink, False))
    return S


def _ipod_mesh(c):
    """Whole iPod in one bmesh: materials 0 acrylic, 1 chrome, 2 screen, 3 click wheel, 4 recessed slots."""
    bm = bmesh.new()
    bm.loops.layers.float_color.new("ipcol")
    W, H, R, n = IPOD_W, IPOD_L, _R, _NA
    Zt, Zb = _ZT, _ZB
    M = 4 * (n + 1)
    run = {n, 2 * n + 1, 3 * n + 2, 4 * n + 3}
    j_top, j_bot = 2 * n + 1, 4 * n + 3
    # ---- side profile: (inset, z); rings 0..13
    rr_ = 0.00075
    prof = [(rr_, Zt)]
    for k in range(1, 5):
        th = math.radians(90.0 * k / 4)
        prof.append((rr_ * (1 - math.sin(th)), Zt - rr_ + rr_ * math.cos(th)))
    zw = 0.0031
    prof += [(0.0, zw), (0.00035, zw - 0.0002), (0.0, zw - 0.0004)]       # white wall end, parting groove (rings 5-7)
    z_lo = -0.0030
    prof.append((0.0, z_lo))                                             # ring 8: bottom of the straight chrome wall
    rb = z_lo - Zb
    for k in range(1, 6):
        th = math.radians(90.0 * k / 5)
        prof.append((rb * (1 - math.cos(th)), z_lo - rb * math.sin(th)))  # rings 9-13: back round-over
    rings = []
    for d, z in prof:
        rings.append([bm.verts.new((x, y, z)) for x, y in _rrect(0, 0, W - 2 * d, H - 2 * d, R - d, n)])
    steps = [(0, False)] * 4 + [(0, True), (0, True), (1, True), (1, True)] + [(1, False)] * 5
    WALL = 7                                                              # chrome wall step (pockets cut here)
    for k, (mat, flat) in enumerate(steps):
        for j in range(M):
            if k == WALL and j in (j_top, j_bot):
                continue
            j1 = (j + 1) % M
            f = bm.faces.new((rings[k][j], rings[k + 1][j], rings[k + 1][j1], rings[k][j1]))
            f.material_index = mat
            f.smooth = not (flat and j in run)
    # ---- back (chrome)
    _face(bm, rings[-1], Vector((0, 0, -1)), 1)
    # ---- front plate with the screen window and the wheel hole
    win_pts = _rrect(0, _WIN_CY, _WIN_W, _WIN_H, _WIN_R, 6)
    whl_pts = _circle(0, _WHL_CY, _WHL_R, 72)
    win_v = [bm.verts.new((x, y, Zt)) for x, y in win_pts]
    whl_v = [bm.verts.new((x, y, Zt)) for x, y in whl_pts]
    allv = list(rings[0]) + win_v + whl_v
    polys = [[Vector((v.co.x, v.co.y, 0.0)) for v in vs] for vs in (rings[0], win_v, whl_v)]
    for t in geo.tessellate_polygon(polys):
        _face(bm, (allv[t[0]], allv[t[1]], allv[t[2]]), Vector((0, 0, 1)), 0)
    # ---- screen recess (walls) + UI decal on its floor
    fi = 0.0003
    flo_v = [bm.verts.new((x, y, _ZF)) for x, y in
             _rrect(0, _WIN_CY, _WIN_W - 2 * fi, _WIN_H - 2 * fi, _WIN_R - fi, 6)]
    Mw = len(win_v)
    wrun = {6, 13, 20, 27}
    cwin = Vector((0, _WIN_CY, 0))
    for j in range(Mw):
        vs = (win_v[j], win_v[(j + 1) % Mw], flo_v[(j + 1) % Mw], flo_v[j])
        cc = sum((v.co for v in vs), Vector()) / 4
        d = cwin - cc
        d.z = 0.0
        _face(bm, vs, d + Vector((0, 0, 0.7)), 0, smooth=j not in wrun)
    _decal(bm, _ui_shapes(c), (0.0, _WIN_CY), _ZF, 2)
    # ---- click wheel: recess wall, ring (+ printed marks), groove, button
    ra0 = _WHL_R - 0.00012
    bm_lathe(bm, [(_WHL_R, Zt), (ra0, _ZR)], segs=72, mat=0, smooth=False, center=(0, _WHL_CY, 0), uv=False)
    _decal(bm, _wheel_shapes(c), (0.0, _WHL_CY), _ZR, 3)
    groove = [(0.0098, _ZR), (0.0097, _ZG), (0.0090, _ZG), (0.0090, _ZBTN - 0.0003)]
    before = set(bm.faces)
    bm_lathe(bm, groove, segs=72, mat=3, smooth=False, center=(0, _WHL_CY, 0), uv=False)
    _paint(bm, [f for f in bm.faces if f not in before], c.groove)
    btn = [(0.0090, _ZBTN - 0.0003)]
    for k in range(1, 4):
        a = math.radians(30.0 * k)
        btn.append((0.0087 + 0.0003 * math.cos(a), _ZBTN - 0.0003 + 0.0003 * math.sin(a)))
    for r in (0.0060, 0.0030, 0.0):
        btn.append((r, _ZBTN + 0.0001 * (1 - (r / 0.0087) ** 2)))
    before = set(bm.faces)
    bm_lathe(bm, btn, segs=72, mat=3, smooth=True, center=(0, _WHL_CY, 0), uv=False)
    _paint(bm, [f for f in bm.faces if f not in before], c.button)
    # ---- top edge: headphone socket + hold switch;  bottom edge: dock connector slot
    zc = _ZC
    ex, ey = Vector((1, 0, 0)), Vector((0, 0, 1))
    top_n, bot_n = Vector((0, 1, 0)), Vector((0, -1, 0))

    def jack(inset):
        return _circle(IPOD_JACK_X, zc, 0.0022 - inset, 40)

    def hold(inset):
        return _rrect(IPOD_HOLD_X, zc, 0.0105 - 2 * inset, 0.0030 - 2 * inset, 0.0015 - inset, 4)

    def dock(inset):
        return _rrect(0.0, zc, 0.0215 - 2 * inset, 0.0030 - 2 * inset, 0.0010 - inset, 3)
    top_org, bot_org = Vector((0, H / 2, 0)), Vector((0, -H / 2, 0))
    holes_top = [_pocket(bm, top_org, ex, ey, top_n, jack, 0.0022, 0.0004, 4, smooth=True),
                 _pocket(bm, top_org, ex, ey, top_n, hold, 0.0009, 0.00015, 4)]
    holes_bot = [_pocket(bm, bot_org, ex, ey, bot_n, dock, 0.0018, 0.0002, 4)]
    _boss(bm, top_org, ex, ey, top_n,
          lambda i: _rrect(IPOD_HOLD_X - 0.0027, zc, 0.0042, 0.0019, 0.0009, 3), 0.0009, -0.0003, 1)
    for j, holes, nrm in ((j_top, holes_top, top_n), (j_bot, holes_bot, bot_n)):
        j1 = (j + 1) % M
        outer = [rings[WALL][j], rings[WALL + 1][j], rings[WALL + 1][j1], rings[WALL][j1]]
        allv = outer + [v for h in holes for v in h]
        polys = [[Vector((v.co.x, v.co.z, 0.0)) for v in vs] for vs in [outer] + holes]
        for t in geo.tessellate_polygon(polys):
            _face(bm, (allv[t[0]], allv[t[1]], allv[t[2]]), nrm, 1)
    bm.normal_update()
    bmesh.ops.translate(bm, vec=(0.0, 0.0, IPOD_T / 2), verts=bm.verts)      # origin = centre of the bottom face
    return bm


# ------------------------------------------------------------------------------------------- shared node shorthands
def _mix(nt, loc, fac, a, b):
    """MixRGB helper: fac/a/b may be a socket or a constant (float / rgba)."""
    n = nt.nodes.new("ShaderNodeMixRGB")
    n.location = loc
    for key, v in (("Fac", fac), ("Color1", a), ("Color2", b)):
        if isinstance(v, bpy.types.NodeSocket):
            nt.links.new(v, n.inputs[key])
        else:
            n.inputs[key].default_value = v
    return n


def _mr(nt, loc, src, a, b, c=0.0, d=1.0, smooth=False):
    """Map Range (clamped) helper; returns the node (output 'Result')."""
    n = N(nt, "ShaderNodeMapRange", loc, inputs={"From Min": a, "From Max": b, "To Min": c, "To Max": d})
    n.clamp = True
    n.interpolation_type = "SMOOTHSTEP" if smooth else "LINEAR"
    if src is not None:
        nt.links.new(src, n.inputs["Value"])
    return n


def _ipod_materials(K, c):
    mats = []
    # the mesh origin is on the bottom face: shift the object coordinates back to the body centre so the noise fields
    # sit where they were when the mesh was modelled about its centre
    off = (0.0, 0.0, -IPOD_T / 2)
    # glossy acrylic front
    m, nt, out = K.new_mat("acrylic")
    tc = N(nt, "ShaderNodeTexCoord", (-1300, 0))
    mp = N(nt, "ShaderNodeMapping", (-1100, 0), inputs={"Location": off})
    L(nt, tc.outputs["Object"], mp.inputs["Vector"])
    nz = N(nt, "ShaderNodeTexNoise", (-800, 100), inputs={"Scale": 420.0, "Detail": 5.0, "Roughness": 0.65})
    L(nt, mp.outputs["Vector"], nz.inputs["Vector"])
    rm = _mr(nt, (-560, 100), nz.outputs["Fac"], 0.35, 0.72, 0.035, 0.14)
    bump = N(nt, "ShaderNodeBump", (-560, -150), inputs={"Strength": 0.02, "Distance": 0.0004})
    L(nt, nz.outputs["Fac"], bump.inputs["Height"])
    b = principled(nt, out, **{"Base Color": c.acrylic, "Roughness": 0.08, "Specular IOR Level": 0.5,
                               "Coat Weight": 0.55, "Coat Roughness": 0.03, "Sheen Weight": 0.12})
    L(nt, rm.outputs["Result"], b.inputs["Roughness"])
    L(nt, bump.outputs["Normal"], b.inputs["Normal"])
    mats.append(m)
    # chrome / brushed steel
    m, nt, out = K.new_mat("chrome")
    tc = N(nt, "ShaderNodeTexCoord", (-1000, 0))
    mp = N(nt, "ShaderNodeMapping", (-800, 0), inputs={"Scale": (1.0, 1.0, 1.0), "Location": off})
    L(nt, tc.outputs["Object"], mp.inputs["Vector"])
    nz = N(nt, "ShaderNodeTexNoise", (-600, 100), inputs={"Scale": 700.0, "Detail": 6.0, "Roughness": 0.7})
    L(nt, mp.outputs["Vector"], nz.inputs["Vector"])
    rm = _mr(nt, (-360, 100), nz.outputs["Fac"], 0.3, 0.7, 0.16, 0.25)
    b = principled(nt, out, **{"Base Color": c.chrome, "Metallic": 0.82, "Roughness": 0.2, "Specular IOR Level": 0.6})
    L(nt, rm.outputs["Result"], b.inputs["Roughness"])
    mats.append(m)
    # screen: vertex colour as albedo + soft backlight emission under a glass coat
    m, nt, out = K.new_mat("screen")
    at = N(nt, "ShaderNodeAttribute", (-600, 0), attribute_type="GEOMETRY", attribute_name="ipcol")
    b = principled(nt, out, **{"Roughness": 0.2, "Specular IOR Level": 0.5, "Coat Weight": 1.0,
                               "Coat Roughness": 0.02, "Coat IOR": 1.45, "Emission Strength": 0.42})
    L(nt, at.outputs["Color"], b.inputs["Base Color"])
    L(nt, at.outputs["Color"], b.inputs["Emission Color"])
    mats.append(m)
    # click wheel: satin plastic, colour from vertex colour
    m, nt, out = K.new_mat("wheel")
    at = N(nt, "ShaderNodeAttribute", (-600, 0), attribute_type="GEOMETRY", attribute_name="ipcol")
    b = principled(nt, out, **{"Roughness": 0.36, "Specular IOR Level": 0.45, "Coat Weight": 0.25,
                               "Coat Roughness": 0.2, "Sheen Weight": 0.2})
    L(nt, at.outputs["Color"], b.inputs["Base Color"])
    mats.append(m)
    # recessed slots (jack bore, hold slot, dock): soft lavender grey, never black
    mats.append(K.simple_mat("slot", c.slot, rough=0.55, spec=0.3, emit=c.slot, emit_strength=0.12))
    return mats


@register("cafe_ipod")
def cafe_ipod(name, coll, root, slots=None):
    """iPod classic lying screen up on a table. Origin = centre of the bottom face (the contact point on the table
    top), +X right, +Y the top edge (headphone socket side), +Z out of the screen; the reading side is -Y.
    Objects: `body` (one mesh, 5 materials, colour layer `ipcol`), `plug` (arrow empty at the headphone socket: its
    local +Z is the socket's outward direction (+Y of the body); a plug goes in along -Z), `col` (hidden collider).
    No custom properties. Card: `use.anchor` / `use.look` `plug` {point, dir out of the socket}, `use.look` `screen`,
    `use.surface` `screen` (the active picture, up = +Y), `use.rest` `top` (plane on the body top), a box collider."""
    K = Kit(name, coll, root, slots)
    c = _ipod_colors(K)
    body = K.to_obj("body", _ipod_mesh(c), _ipod_materials(K, c))
    plug = K.empty("plug", loc=IPOD_PLUG, rot=(-math.pi / 2, 0.0, 0.0), kind="ARROWS", size=0.01, parent=body)
    bm = bmesh.new()
    bm_box(bm, (IPOD_W, IPOD_L, IPOD_T), loc=(0.0, 0.0, IPOD_T / 2))
    col = K.collider(K.to_obj("col", bm))
    screen_c = [0.0, _WIN_CY, round(IPOD_T - 0.0004, 5)]
    use = {
        "anchor": [{"name": "plug", "point": list(IPOD_PLUG), "dir": list(IPOD_PLUG_DIR), "object": plug.name}],
        "look": [{"name": "plug", "point": list(IPOD_PLUG), "dir": list(IPOD_PLUG_DIR)},
                 {"name": "screen", "point": screen_c}],
        "surface": [{"name": "screen", "center": screen_c, "normal": [0.0, 0.0, 1.0], "up": [0.0, 1.0, 0.0],
                     "size": [round(_SCREEN[0], 5), round(_SCREEN[1], 5)]}],
        "rest": [{"name": "top", "type": "plane", "center": [0.0, 0.0, IPOD_T], "normal": [0.0, 0.0, 1.0],
                  "size": [IPOD_W, IPOD_L]}],
    }
    return K.card(use=use, colliders=[{"type": "box", "object": col.name, "rnd": 0.002, "tag": name}],
                  origin="table_top", front="-Y")


# =============================================================================================================== vase
VASE_H = 0.085                                   # height of the vase (foot to lip)
VASE_MOUTH = (0.0012, 0.0006, VASE_H)            # centre of the mouth in vase coordinates (handmade lean)
SPRIG_SEED = 11


def _vase_colors(K):
    b = K.blend
    return SimpleNamespace(
        # glaze: mid / light body tones, pooled bead, dark and pale speckles, thin glaze over the rim; raw clay foot
        g_mid=b(iris=0.9, hl_high=0.1, k=1.25), g_light=b(iris=0.8, hl_high=0.2, k=1.4),
        g_pool=b(iris=1.0, chroma=0.9, k=1.25), g_dark=b(iris=0.6, subtle=0.4, k=0.85),
        g_pale=b(base=0.5, iris=0.5, hue=-20, chroma=1.25, k=1.3), g_rim=b(surface=0.6, iris=0.4),
        clay0=b(hl_med=0.7, gold=0.3), clay1=b(overlay=0.75, gold=0.25),
        # eucalyptus stem (dusty rose-brown) and leaves (sage, silvery bloom, paler rim and veins, greener underside)
        stem0=b(rose=1.0, hue=10, chroma=0.6, k=0.9), stem1=b(rose=1.0, hue=10, chroma=0.5),
        stem2=b(rose=1.0, hue=10, chroma=0.7, k=0.7),
        sage0=b(foam=0.7, overlay=0.3, hue=-30, chroma=1.5, k=0.9),
        sage1=b(foam=0.6, overlay=0.4, hue=-30, chroma=1.5, k=1.05),
        bloom=b(foam=0.6, overlay=0.4, hue=-30, chroma=0.9, k=1.35),
        rim=b(foam=0.6, overlay=0.4, hue=-30, chroma=1.25, k=1.25),
        vein=b(foam=0.6, surface=0.4, hue=-30, chroma=0.8, k=1.35),
        under=b(foam=0.9, gold=0.1, hue=-30, chroma=1.1, k=1.35),
        sheen=b(surface=0.8, foam=0.2))


def _vase_profile():
    """(r, z) profile: outside bottom->top, over the rolled lip, down the inside; hard crease at the foot edge."""
    pts = [(0.0198, 0.0004), (0.0208, 0.0042), (0.0226, 0.0092), (0.0272, 0.0145), (0.0306, 0.0205),
           (0.0320, 0.0258), (0.0308, 0.0322), (0.0266, 0.0385), (0.0200, 0.0443), (0.0140, 0.0498),
           (0.0106, 0.0550), (0.0095, 0.0610), (0.0094, 0.0665), (0.0102, 0.0725), (0.0118, 0.0785),
           (0.0128, 0.0818)]
    outer = [(v.x, v.y) for v in catmull([(r, z, 0.0) for r, z in pts], 5)]
    lip_c = (0.0109, VASE_H - 0.0021)
    lip = [(lip_c[0] + 0.0021 * math.cos(math.radians(a)), lip_c[1] + 0.0021 * math.sin(math.radians(a)))
           for a in range(0, 181, 20)]
    ins = [(0.0088, 0.0818), (0.0084, 0.0790), (0.0075, 0.0730), (0.0066, 0.0670), (0.0064, 0.0610), (0.0072, 0.0555),
           (0.0105, 0.0500), (0.0160, 0.0442), (0.0222, 0.0385), (0.0268, 0.0322), (0.0288, 0.0258), (0.0276, 0.0195),
           (0.0240, 0.0145), (0.0190, 0.0112), (0.0110, 0.0094)]
    inner = [(v.x, v.y) for v in catmull([(r, z, 0.0) for r, z in ins], 5)]
    return [(0.0, 0.0), (0.0192, 0.0), (0.0192, 0.0)] + outer + lip[1:] + inner[1:] + [(0.0, 0.0090)]


def _vase_mesh():
    bm = bmesh.new()
    bm_lathe(bm, _vase_profile(), segs=72, mat=0)
    # hand-thrown irregularity: slightly oval, wobbling, leaning neck
    for v in bm.verts:
        x, y, z = v.co
        rho = math.hypot(x, y)
        if rho < 1e-6:
            continue
        th = math.atan2(y, x)
        k = (1.0 + 0.0060 * math.sin(th + 0.6 + 14.0 * z) + 0.0040 * math.sin(2 * th + 1.9 - 9.0 * z)
             + 0.0025 * math.sin(3 * th + 0.2 + 25.0 * z))
        t = z / VASE_H
        v.co.x = x * k + VASE_MOUTH[0] * t * t
        v.co.y = y * k + VASE_MOUTH[1] * t * t
    bm.normal_update()
    return bm


def _glaze_mat(K, c):
    """Satin iris/lavender glaze, pooled bead above a wavy unglazed foot, tiny speckles, throwing rings (bump)."""
    m, nt, out = K.new_mat("glaze")
    tc = N(nt, "ShaderNodeTexCoord", (-1800, 0))
    sep = N(nt, "ShaderNodeSeparateXYZ", (-1600, 300))
    L(nt, tc.outputs["Object"], sep.inputs["Vector"])
    z = sep.outputs["Z"]
    nz_a = N(nt, "ShaderNodeTexNoise", (-1600, 0), inputs={"Scale": 90.0, "Detail": 3.0, "Roughness": 0.55})
    L(nt, tc.outputs["Object"], nz_a.inputs["Vector"])
    sub = N(nt, "ShaderNodeMath", (-1400, 0), operation="SUBTRACT")
    sub.inputs[1].default_value = 0.5
    L(nt, nz_a.outputs["Fac"], sub.inputs[0])
    mul = N(nt, "ShaderNodeMath", (-1250, 0), operation="MULTIPLY")
    mul.inputs[1].default_value = 0.0018
    L(nt, sub.outputs[0], mul.inputs[0])
    edge = N(nt, "ShaderNodeMath", (-1100, 200), operation="ADD")
    L(nt, z, edge.inputs[0])
    L(nt, mul.outputs[0], edge.inputs[1])
    foot = _mr(nt, (-900, 300), edge.outputs[0], 0.0070, 0.0076, 1.0, 0.0, smooth=True)       # 1 on the raw foot
    b_up = _mr(nt, (-900, 100), edge.outputs[0], 0.0072, 0.0086, 0.0, 1.0, smooth=True)
    b_dn = _mr(nt, (-900, -50), edge.outputs[0], 0.0086, 0.0130, 1.0, 0.0, smooth=True)
    bead = N(nt, "ShaderNodeMath", (-700, 40), operation="MULTIPLY")
    L(nt, b_up.outputs[0], bead.inputs[0])
    L(nt, b_dn.outputs[0], bead.inputs[1])
    # large-scale glaze variation + height gradient (thin & light at the shoulder, pooled & deeper low down)
    nz_b = N(nt, "ShaderNodeTexNoise", (-1600, -300), inputs={"Scale": 7.0, "Detail": 2.0, "Roughness": 0.5})
    L(nt, tc.outputs["Object"], nz_b.inputs["Vector"])
    tz = N(nt, "ShaderNodeMath", (-1400, 300), operation="MULTIPLY")
    tz.inputs[1].default_value = 1.0 / VASE_H
    L(nt, z, tz.inputs[0])
    f0 = _mr(nt, (-1200, -300), nz_b.outputs["Fac"], 0.25, 0.75, -0.25, 0.25)
    f1 = _mr(nt, (-1200, 300), tz.outputs[0], 0.0, 1.0, -0.20, 0.35)
    fs = N(nt, "ShaderNodeMath", (-1000, -200), operation="ADD", use_clamp=True)
    fs.inputs[1].default_value = 0.55
    L(nt, f0.outputs["Result"], fs.inputs[0])
    fs2 = N(nt, "ShaderNodeMath", (-850, -200), operation="ADD", use_clamp=True)
    L(nt, fs.outputs[0], fs2.inputs[0])
    L(nt, f1.outputs["Result"], fs2.inputs[1])
    gl = _mix(nt, (-650, -200), fs2.outputs[0], c.g_mid, c.g_light)
    gl2 = _mix(nt, (-450, -200), bead.outputs[0], gl.outputs["Color"], c.g_pool)
    # speckles: dark-ish violet-grey and a few pale ones
    vo1 = N(nt, "ShaderNodeTexVoronoi", (-1600, -600), inputs={"Scale": 300.0, "Randomness": 1.0})
    L(nt, tc.outputs["Object"], vo1.inputs["Vector"])
    s1c = N(nt, "ShaderNodeSeparateColor", (-1400, -700))
    L(nt, vo1.outputs["Color"], s1c.inputs["Color"])
    s1d = _mr(nt, (-1400, -560), vo1.outputs["Distance"], 0.10, 0.17, 1.0, 0.0, smooth=True)
    s1r = _mr(nt, (-1200, -700), s1c.outputs["Red"], 0.70, 0.72, 0.0, 1.0)
    s1 = N(nt, "ShaderNodeMath", (-1000, -620), operation="MULTIPLY")
    L(nt, s1d.outputs["Result"], s1.inputs[0])
    L(nt, s1r.outputs["Result"], s1.inputs[1])
    s1m = N(nt, "ShaderNodeMath", (-850, -620), operation="MULTIPLY")
    s1m.inputs[1].default_value = 0.75
    L(nt, s1.outputs[0], s1m.inputs[0])
    gl3 = _mix(nt, (-250, -300), s1m.outputs[0], gl2.outputs["Color"], c.g_dark)
    vo2 = N(nt, "ShaderNodeTexVoronoi", (-1600, -900), inputs={"Scale": 650.0, "Randomness": 1.0, "W": 3.0},
            voronoi_dimensions="4D")
    L(nt, tc.outputs["Object"], vo2.inputs["Vector"])
    s2c = N(nt, "ShaderNodeSeparateColor", (-1400, -1000))
    L(nt, vo2.outputs["Color"], s2c.inputs["Color"])
    s2d = _mr(nt, (-1400, -860), vo2.outputs["Distance"], 0.07, 0.14, 1.0, 0.0, smooth=True)
    s2r = _mr(nt, (-1200, -1000), s2c.outputs["Red"], 0.80, 0.82, 0.0, 0.6)
    s2 = N(nt, "ShaderNodeMath", (-1000, -920), operation="MULTIPLY")
    L(nt, s2d.outputs["Result"], s2.inputs[0])
    L(nt, s2r.outputs["Result"], s2.inputs[1])
    gl4 = _mix(nt, (-50, -300), s2.outputs[0], gl3.outputs["Color"], c.g_pale)
    # glaze thins to pale body colour over the rim
    rim = _mr(nt, (-250, -520), z, VASE_H - 0.0030, VASE_H - 0.0004, 0.0, 0.45, smooth=True)
    gl5 = _mix(nt, (150, -300), rim.outputs["Result"], gl4.outputs["Color"], c.g_rim)
    # raw foot colour (warm stoneware)
    clay = N(nt, "ShaderNodeTexNoise", (-900, -1250), inputs={"Scale": 600.0, "Detail": 4.0, "Roughness": 0.6})
    L(nt, tc.outputs["Object"], clay.inputs["Vector"])
    clay_c = _mix(nt, (-650, -1250), clay.outputs["Fac"], c.clay0, c.clay1)
    L(nt, clay.outputs["Fac"], clay_c.inputs["Fac"])
    base = _mix(nt, (400, -300), foot.outputs["Result"], gl5.outputs["Color"], clay_c.outputs["Color"])
    # roughness: satin glaze vs dry foot
    rg = _mr(nt, (-650, -420), nz_b.outputs["Fac"], 0.3, 0.7, 0.27, 0.36)
    rough = N(nt, "ShaderNodeMix", (400, -520), data_type="FLOAT")
    L(nt, foot.outputs["Result"], rough.inputs["Factor"])
    L(nt, rg.outputs["Result"], rough.inputs[2])
    rough.inputs[3].default_value = 0.85
    # bump: soft, drifting throwing rings on the glaze (amplitude + phase vary around the pot), grain on the foot
    nzl = N(nt, "ShaderNodeTexNoise", (-1700, 900), inputs={"Scale": 22.0, "Detail": 1.0, "Roughness": 0.5})
    L(nt, tc.outputs["Object"], nzl.inputs["Vector"])
    dz = _mr(nt, (-1500, 900), nzl.outputs["Fac"], 0.0, 1.0, -0.0014, 0.0014)
    zz = N(nt, "ShaderNodeMath", (-1300, 900), operation="ADD")
    L(nt, z, zz.inputs[0])
    L(nt, dz.outputs["Result"], zz.inputs[1])
    czz = N(nt, "ShaderNodeCombineXYZ", (-1150, 900))
    L(nt, zz.outputs[0], czz.inputs["Z"])
    wv = N(nt, "ShaderNodeTexWave", (-1000, 900), inputs={"Scale": 220.0, "Distortion": 0.8, "Detail": 1.5,
                                                           "Detail Scale": 1.0, "Detail Roughness": 0.5},
           wave_type="BANDS", bands_direction="Z", wave_profile="SIN")
    L(nt, czz.outputs[0], wv.inputs["Vector"])
    nza = N(nt, "ShaderNodeTexNoise", (-1700, 1150), inputs={"Scale": 14.0, "Detail": 1.0, "Roughness": 0.5})
    L(nt, tc.outputs["Object"], nza.inputs["Vector"])
    amp = _mr(nt, (-1500, 1150), nza.outputs["Fac"], 0.3, 0.7, 0.35, 1.0)
    wva = N(nt, "ShaderNodeMath", (-800, 1000), operation="MULTIPLY")
    L(nt, wv.outputs["Fac"], wva.inputs[0])
    L(nt, amp.outputs["Result"], wva.inputs[1])
    gr = N(nt, "ShaderNodeTexNoise", (-1000, 500), inputs={"Scale": 900.0, "Detail": 3.0, "Roughness": 0.6})
    L(nt, tc.outputs["Object"], gr.inputs["Vector"])
    hmix = N(nt, "ShaderNodeMix", (-600, 700), data_type="FLOAT")
    L(nt, foot.outputs["Result"], hmix.inputs["Factor"])
    L(nt, wva.outputs[0], hmix.inputs[2])
    L(nt, gr.outputs["Fac"], hmix.inputs[3])
    bump = N(nt, "ShaderNodeBump", (-400, 700), inputs={"Strength": 1.0, "Distance": 0.00007})
    L(nt, hmix.outputs[0], bump.inputs["Height"])
    b = principled(nt, out, **{"Specular IOR Level": 0.5, "Coat Weight": 0.10, "Coat Roughness": 0.3})
    L(nt, base.outputs["Color"], b.inputs["Base Color"])
    L(nt, rough.outputs[0], b.inputs["Roughness"])
    L(nt, bump.outputs["Normal"], b.inputs["Normal"])
    return m


# ------------------------------------------------------------------------------------------------ eucalyptus sprig
def _frames(pts):
    """Tangents + parallel-transported normals/binormals along a dense polyline."""
    n = len(pts)
    T = [(pts[min(i + 1, n - 1)] - pts[max(i - 1, 0)]).normalized() for i in range(n)]
    N0 = Vector((1, 0, 0))
    N0 = (N0 - T[0] * N0.dot(T[0])).normalized()
    Ns = [N0]
    for i in range(1, n):
        p = Ns[-1] - T[i] * Ns[-1].dot(T[i])
        Ns.append(p.normalized())
    return T, Ns, [T[i].cross(Ns[i]) for i in range(n)]


def _leaf(bm, P, a, w, nrm, r, rnd, tint, mat=1, segs=32):
    """One round, cupped, slightly wavy coin leaf.  Base edge touches P; long axis a, width axis w, upper-face
    normal nrm."""
    rings = (0.0, 0.26, 0.52, 0.76, 0.92, 1.0)
    p1, p2, p3 = rnd.uniform(0, 6.283), rnd.uniform(0, 6.283), rnd.uniform(0, 6.283)
    bowl, fold, curl = 0.20 + rnd.uniform(-0.05, 0.06), 0.05 + rnd.uniform(-0.02, 0.03), 0.14 + rnd.uniform(-0.05, 0.09)
    uvl = bm.loops.layers.uv.verify()
    ccl = bm.loops.layers.float_color.get("ipleaf") or bm.loops.layers.float_color.new("ipleaf")
    info = {}

    def mk(rho, th):
        g = 1.0 + 0.035 * math.sin(3 * th + p1) + 0.022 * math.sin(5 * th + p2) + 0.012 * math.sin(2 * th + p3)
        lx, ly = rho * g * math.cos(th), rho * g * math.sin(th)
        h = (bowl * (lx * lx + ly * ly) + fold * abs(ly) - curl * max(lx, 0.0) ** 2
             + 0.012 * rho ** 3 * math.sin(7 * th + p1))
        v = bm.verts.new(P + a * (r * (0.93 + lx)) + w * (r * ly) + nrm * (r * h))
        info[v] = (0.5 + 0.5 * rho * math.sin(th), 0.5 + 0.5 * rho * math.cos(th))
        return v
    centre = mk(0.0, 0.0)
    grid = [[centre]] + [[mk(rho, 2 * math.pi * j / segs) for j in range(segs)] for rho in rings[1:]]
    faces = []
    for j in range(segs):
        j1 = (j + 1) % segs
        faces.append(_face(bm, (centre, grid[1][j], grid[1][j1]), nrm, mat, smooth=True))
        for k in range(1, len(rings) - 1):
            faces.append(_face(bm, (grid[k][j], grid[k][j1], grid[k + 1][j1], grid[k + 1][j]), nrm, mat, smooth=True))
    c3 = (tint, rnd.random(), rnd.random(), 1.0)
    for f in faces:
        if f is None:
            continue
        for lp in f.loops:
            lp[uvl].uv = info[lp.vert]
            lp[ccl] = c3
    return len(faces)


_STEMS = (
    # control points (sprig-local, origin = mouth centre), nodes, first node above mouth, leaf radius at base, phase
    dict(ctrl=[(0.001, 0.001, -0.045), (0.002, 0.0015, 0.0), (0.001, -0.001, 0.04), (-0.007, -0.006, 0.10),
               (-0.020, -0.011, 0.165), (-0.034, -0.013, 0.222), (-0.044, -0.011, 0.262)],
         nodes=9, h0=0.050, rbase=0.0172, phase=0.3),
    dict(ctrl=[(-0.002, 0.002, -0.045), (-0.001, 0.001, 0.0), (0.006, 0.004, 0.035), (0.020, 0.009, 0.090),
               (0.036, 0.012, 0.145), (0.048, 0.010, 0.190), (0.054, 0.006, 0.222)],
         nodes=6, h0=0.056, rbase=0.0158, phase=2.1),
    dict(ctrl=[(0.002, -0.001, -0.045), (0.001, -0.002, 0.0), (0.001, -0.010, 0.030), (-0.003, -0.026, 0.070),
               (-0.007, -0.040, 0.115), (-0.009, -0.047, 0.155), (-0.009, -0.050, 0.175)],
         nodes=4, h0=0.060, rbase=0.0145, phase=4.0),
)


def _sprig_mesh(seed=SPRIG_SEED):
    rnd = random.Random(seed)
    bm = bmesh.new()
    bm.loops.layers.uv.verify()
    bm.loops.layers.float_color.new("ipleaf")
    leaves = 0
    for st in _STEMS:
        pts = catmull(st["ctrl"], 14)
        T, Ns, _bs = _frames(pts)
        S = [0.0]
        for i in range(1, len(pts)):
            S.append(S[-1] + (pts[i] - pts[i - 1]).length)
        tot = S[-1]
        i_m = next(i for i in range(len(pts)) if pts[i].z >= 0.0)
        s_m = S[i_m]

        def at(s):
            i = max(1, min(next((k for k in range(len(S)) if S[k] >= s), len(S) - 1), len(S) - 1))
            u = (s - S[i - 1]) / max(S[i] - S[i - 1], 1e-9)
            P = pts[i - 1].lerp(pts[i], u)
            Tt = T[i - 1].lerp(T[i], u).normalized()
            Nn = Ns[i - 1].lerp(Ns[i], u)
            Nn = (Nn - Tt * Nn.dot(Tt)).normalized()
            return P, Tt, Nn, Tt.cross(Nn)
        bm_tube(bm, pts, lambda t: 0.0019 - 0.0011 * t, sides=7, mat=0, cap=("none", "round"), up=(1, 0, 0))
        s_first = s_m + st["h0"]
        s_last = tot - 0.014
        m = st["nodes"]
        phi = st["phase"]
        for i in range(m):
            t = i / (m - 1)
            s = s_first + (s_last - s_first) * t ** 0.92
            P, Tt, Nn, Bn = at(s)
            phi += math.pi / 2 + rnd.uniform(-0.25, 0.25)
            alpha = math.radians(66.0 - 34.0 * t + rnd.uniform(-6, 6))
            for side in (0, 1):
                ph = phi + side * math.pi
                o = Nn * math.cos(ph) + Bn * math.sin(ph)
                wt = -Nn * math.sin(ph) + Bn * math.cos(ph)
                r = st["rbase"] * (1.0 - 0.47 * t ** 0.95) * rnd.uniform(0.93, 1.07)
                a = Tt * math.cos(alpha + rnd.uniform(-0.08, 0.08)) + o * math.sin(alpha)
                delta = (0.50 * (1.0 - t) ** 1.2 + 0.10) * rnd.uniform(0.7, 1.25) * (r / 0.016)
                down = Vector((0, 0, -1))
                dp = down - a * down.dot(a)
                if dp.length > 1e-6:
                    a = (a * math.cos(delta) + dp.normalized() * math.sin(delta)).normalized()
                w = (wt - a * wt.dot(a)).normalized()
                nrm = a.cross(w)
                ref = Vector((0, 0, 0.5)) + Tt * 0.5 - o * 0.6
                if nrm.dot(ref) < 0:
                    nrm = -nrm
                leaves += 1 if _leaf(bm, P, a, w, nrm, r, rnd, rnd.random()) else 0
    bm.normal_update()
    return bm, leaves


def _euc_materials(K, c):
    """(stem, leaf) materials."""
    m1, nt, out = K.new_mat("stem")
    tc = N(nt, "ShaderNodeTexCoord", (-900, 0))
    nz = N(nt, "ShaderNodeTexNoise", (-700, 0), inputs={"Scale": 160.0, "Detail": 4.0, "Roughness": 0.6})
    L(nt, tc.outputs["Object"], nz.inputs["Vector"])
    sz = N(nt, "ShaderNodeSeparateXYZ", (-900, 250))
    L(nt, tc.outputs["Object"], sz.inputs["Vector"])
    zr = _mr(nt, (-700, 250), sz.outputs["Z"], 0.02, 0.24, 0.0, 1.0)
    c0 = _mix(nt, (-450, 150), zr.outputs["Result"], c.stem0, c.stem1)
    L(nt, zr.outputs["Result"], c0.inputs["Fac"])
    cc = _mix(nt, (-250, 0), nz.outputs["Fac"], c0.outputs["Color"], c.stem2)
    nzm = _mr(nt, (-450, -150), nz.outputs["Fac"], 0.35, 0.8, 0.0, 0.35)
    L(nt, nzm.outputs["Result"], cc.inputs["Fac"])
    b = principled(nt, out, **{"Roughness": 0.55, "Specular IOR Level": 0.35, "Sheen Weight": 0.08})
    L(nt, cc.outputs["Color"], b.inputs["Base Color"])

    m2, nt, out = K.new_mat("leaf")
    uv = N(nt, "ShaderNodeUVMap", (-1800, 300))
    sp = N(nt, "ShaderNodeSeparateXYZ", (-1600, 300))
    L(nt, uv.outputs["UV"], sp.inputs["Vector"])
    at = N(nt, "ShaderNodeAttribute", (-1800, -100), attribute_type="GEOMETRY", attribute_name="ipleaf")
    sc_ = N(nt, "ShaderNodeSeparateColor", (-1600, -100))
    L(nt, at.outputs["Color"], sc_.inputs["Color"])
    # |u - 0.5| and radial distance
    du0 = N(nt, "ShaderNodeMath", (-1400, 400), operation="SUBTRACT")
    du0.inputs[1].default_value = 0.5
    L(nt, sp.outputs["X"], du0.inputs[0])
    du = N(nt, "ShaderNodeMath", (-1250, 400), operation="ABSOLUTE")
    L(nt, du0.outputs[0], du.inputs[0])
    dv0 = N(nt, "ShaderNodeMath", (-1400, 250), operation="SUBTRACT")
    dv0.inputs[1].default_value = 0.5
    L(nt, sp.outputs["Y"], dv0.inputs[0])
    cx = N(nt, "ShaderNodeCombineXYZ", (-1250, 250))
    L(nt, du0.outputs[0], cx.inputs["X"])
    L(nt, dv0.outputs[0], cx.inputs["Y"])
    ln = N(nt, "ShaderNodeVectorMath", (-1100, 250), operation="LENGTH")
    L(nt, cx.outputs[0], ln.inputs[0])
    rho = N(nt, "ShaderNodeMath", (-950, 250), operation="MULTIPLY")
    rho.inputs[1].default_value = 2.0
    L(nt, ln.outputs["Value"], rho.inputs[0])
    # colour: per-leaf tint between the two sage tones + silvery bloom patches + paler rim
    base = _mix(nt, (-1300, -100), sc_.outputs["Red"], c.sage0, c.sage1)
    L(nt, sc_.outputs["Red"], base.inputs["Fac"])
    nz = N(nt, "ShaderNodeTexNoise", (-1500, -400), inputs={"Scale": 9.0, "Detail": 4.0, "Roughness": 0.55})
    cmb = N(nt, "ShaderNodeCombineXYZ", (-1700, -400))
    L(nt, sp.outputs["X"], cmb.inputs["X"])
    L(nt, sp.outputs["Y"], cmb.inputs["Y"])
    L(nt, sc_.outputs["Green"], cmb.inputs["Z"])
    L(nt, cmb.outputs[0], nz.inputs["Vector"])
    bl = _mr(nt, (-1300, -400), nz.outputs["Fac"], 0.35, 0.72, 0.0, 0.42)
    bloom = _mix(nt, (-1000, -100), bl.outputs["Result"], base.outputs["Color"], c.bloom)
    L(nt, bl.outputs["Result"], bloom.inputs["Fac"])
    rimf = _mr(nt, (-950, -300), rho.outputs[0], 0.55, 1.0, 0.0, 0.22, smooth=True)
    rimc = _mix(nt, (-700, -100), rimf.outputs["Result"], bloom.outputs["Color"], c.rim)
    L(nt, rimf.outputs["Result"], rimc.inputs["Fac"])
    # veins: midrib + a few lateral veins (pale, subtle)
    mid = _mr(nt, (-1100, 550), du.outputs[0], 0.010, 0.042, 1.0, 0.0, smooth=True)
    fade = _mr(nt, (-1100, 700), sp.outputs["Y"], 0.45, 0.92, 1.0, 0.0, smooth=True)
    midr = N(nt, "ShaderNodeMath", (-900, 600), operation="MULTIPLY")
    L(nt, mid.outputs["Result"], midr.inputs[0])
    L(nt, fade.outputs["Result"], midr.inputs[1])
    lm = N(nt, "ShaderNodeMath", (-1250, 850), operation="MULTIPLY")
    lm.inputs[1].default_value = 0.85
    L(nt, du.outputs[0], lm.inputs[0])
    lv = N(nt, "ShaderNodeMath", (-1100, 850), operation="SUBTRACT")
    L(nt, sp.outputs["Y"], lv.inputs[0])
    L(nt, lm.outputs[0], lv.inputs[1])
    lcx = N(nt, "ShaderNodeCombineXYZ", (-950, 850))
    L(nt, lv.outputs[0], lcx.inputs["X"])
    wv = N(nt, "ShaderNodeTexWave", (-800, 850), inputs={"Scale": 7.0, "Distortion": 0.6},
           wave_type="BANDS", bands_direction="X", wave_profile="SAW")
    L(nt, lcx.outputs[0], wv.inputs["Vector"])
    lat0 = _mr(nt, (-600, 850), wv.outputs["Fac"], 0.90, 0.99, 0.0, 1.0)
    side = _mr(nt, (-800, 1050), du.outputs[0], 0.28, 0.44, 1.0, 0.0, smooth=True)
    lat = N(nt, "ShaderNodeMath", (-420, 900), operation="MULTIPLY")
    L(nt, lat0.outputs["Result"], lat.inputs[0])
    L(nt, side.outputs["Result"], lat.inputs[1])
    vm = N(nt, "ShaderNodeMath", (-420, 700), operation="MULTIPLY")
    vm.inputs[1].default_value = 0.55
    L(nt, midr.outputs[0], vm.inputs[0])
    vl = N(nt, "ShaderNodeMath", (-250, 780), operation="MULTIPLY")
    vl.inputs[1].default_value = 0.22
    L(nt, lat.outputs[0], vl.inputs[0])
    vsum = N(nt, "ShaderNodeMath", (-80, 740), operation="ADD", use_clamp=True)
    L(nt, vm.outputs[0], vsum.inputs[0])
    L(nt, vl.outputs[0], vsum.inputs[1])
    top = _mix(nt, (-80, -100), vsum.outputs[0], rimc.outputs["Color"], c.vein)
    L(nt, vsum.outputs[0], top.inputs["Fac"])
    # underside: greener, less silvery
    ng = N(nt, "ShaderNodeNewGeometry", (-300, -400))
    under_m = _mix(nt, (-100, -250), 0.28, base.outputs["Color"], c.under)
    col = _mix(nt, (150, -100), ng.outputs["Backfacing"], top.outputs["Color"], under_m.outputs["Color"])
    bmp = N(nt, "ShaderNodeBump", (150, -450), inputs={"Strength": 0.25, "Distance": 0.0004})
    hh = N(nt, "ShaderNodeMath", (-80, -520), operation="ADD")
    L(nt, vm.outputs[0], hh.inputs[0])
    L(nt, nz.outputs["Fac"], hh.inputs[1])
    L(nt, hh.outputs[0], bmp.inputs["Height"])
    b = principled(nt, out, **{"Roughness": 0.44, "Specular IOR Level": 0.45, "Sheen Weight": 0.25,
                               "Sheen Roughness": 0.45, "Sheen Tint": c.sheen})
    L(nt, col.outputs["Color"], b.inputs["Base Color"])
    L(nt, bmp.outputs["Normal"], b.inputs["Normal"])
    return m1, m2


# the vase body as stacked vertical cylinders (name, radius, z from, z to): belly, shoulder, upper shoulder, neck
_VASE_COLLIDERS = (("col_belly", 0.0330, 0.0, 0.0340), ("col_shoulder", 0.0305, 0.0340, 0.0410),
                   ("col_upper", 0.0255, 0.0410, 0.0470), ("col_neck", 0.0170, 0.0470, VASE_H + 0.0002))


@register("cafe_vase")
def cafe_vase(name, coll, root, slots=None):
    """Ceramic bud vase (satin lavender glaze over an unglazed foot) holding a silver-dollar eucalyptus sprig. Origin =
    centre of the foot on the table top, +Z up. Objects: `pot` (the vase mesh), `sprig` (child of the pot: ONE mesh,
    2 materials, stems pass down into the neck; origin = centre of the vase mouth, 0.0012 / 0.0006 / 0.085 m in the
    vase frame; object property `leaf_count`; colour layer `ipleaf`, UV map), `col_*` (hidden collider cylinders).
    No custom properties. Card: `use.look` `sprig` (top of the sprig) and `mouth`, `use.rest` `lip` (an edge across
    the rolled lip), four stacked cylinder colliders that cover the belly, shoulders and neck."""
    K = Kit(name, coll, root, slots)
    c = _vase_colors(K)
    pot = K.to_obj("pot", _vase_mesh(), [_glaze_mat(K, c)])
    bm, nleaves = _sprig_mesh()
    tip = max((v.co for v in bm.verts), key=lambda p: p.z).copy()          # highest point of the sprig (sprig frame)
    sprig = K.to_obj("sprig", bm, list(_euc_materials(K, c)), loc=VASE_MOUTH, parent=pot)
    sprig["leaf_count"] = nleaves
    mx, my, mz = VASE_MOUTH
    cols = []
    for base, R, z0, z1 in _VASE_COLLIDERS:
        b = bmesh.new()
        bmesh.ops.create_cone(b, cap_ends=True, segments=24, radius1=R, radius2=R, depth=z1 - z0)
        o = K.collider(K.to_obj(base, b, loc=(0.0, 0.0, (z0 + z1) / 2)))
        cols.append({"type": "cylinder", "object": o.name, "R": R, "half_h": (z1 - z0) / 2, "rnd": 0.002, "tag": name})
    use = {
        "look": [{"name": "sprig", "point": [round(mx + tip.x, 4), round(my + tip.y, 4), round(mz + tip.z, 4)]},
                 {"name": "mouth", "point": [mx, my, mz]}],
        "rest": [{"name": "lip", "type": "edge", "a": [mx - 0.0109, my, mz], "b": [mx + 0.0109, my, mz],
                  "normal": [0.0, 0.0, 1.0]}],
    }
    return K.card(use=use, colliders=cols, origin="table_top", front="-Y")


# ============================================================================================================ earbuds
AOV = "earbud"
CORD_R = 0.0016
# default anchors in the world-aligned cafe frame. Body anchors: the points of the seated figure the cord hangs from
# (ears, jaw, throat, chest, belly) with the semantic bone each one belongs to; the cord then crosses the table's near
# edge and lies on the table top into the iPod's socket (see _table_route).
_BODY_ANCHORS = {
    "earL": ((0.04954, -0.13868, 1.01082), "head"), "earR": ((-0.07113, -0.03310, 1.07307), "head"),
    "jawL": ((-0.01364, -0.13661, 0.95669), "head"), "jawR": ((-0.09783, -0.06295, 1.00012), "head"),
    "split": ((-0.01772, -0.11651, 0.89339), "neck"),
    "chest": ((0.03507, -0.15392, 0.74756), "upper_body2"),
    "belly": ((0.07439, -0.11579, 0.67146), "upper_body"),
}
_CORDS = (("earL", "jawL", "split"), ("earR", "jawR", "split"),
          ("split", "chest", "belly", "edge", "lie1", "lie2", "plugin", "plug"))
# control points of the jaw, chest and belly anchors hang about 1 cm below their anchor, tilted as those body parts were
_SAG = {"jawL": (-0.002227, 0.002936, -0.009296), "jawR": (-0.002227, 0.002936, -0.009296),
        "chest": (0.000095, 0.002813, -0.009596), "belly": (0.000011, 0.001736, -0.009848)}
_BUD_ROT = (0.07437, 0.37036, -0.7188)                       # orientation of the seated head (Euler XYZ, rad)


def _table_route():
    """Waypoints of the cord from the table's near edge to the iPod socket in the reference layout (cafe frame)."""
    s, c = math.sin(IPOD_LAYOUT_YAW), math.cos(IPOD_LAYOUT_YAW)
    ax, ay, az = IPOD_LAYOUT_AT
    px, py, pz = IPOD_PLUG
    plug = Vector((ax + c * px - s * py, ay + s * px + c * py, az + pz))
    out = Vector((-s, c, 0.0))                                # out of the socket = the iPod's top-edge direction
    z = TABLE_Z + 0.0016
    return {
        "edge": Vector((0.150, -0.185, TABLE_Z + 0.004)),
        "lie1": Vector((0.175, -0.300, z)),
        "lie2": Vector((plug.x + out.x * 0.07, plug.y + out.y * 0.07, z)),
        "plugin": plug + out * 0.012,
        "plug": plug,
    }, out


def _setup_aov(view_layer):
    if AOV not in view_layer.aovs:
        a = view_layer.aovs.add()
        a.name = AOV
        a.type = "VALUE"


def _earbud_material(K):
    m = K.simple_mat("white", K.slot("surface"), rough=0.35, coat=0.3)
    aov = m.node_tree.nodes.new("ShaderNodeOutputAOV")
    aov.location = (900, -250)
    aov.aov_name = AOV
    aov.inputs["Value"].default_value = 1.0
    return m


@register("cafe_earbuds")
def cafe_earbuds(name, coll, root, slots=None):
    """White wired earbuds: a bud in each ear of a seated figure, a Y-split under the chin, one cord down the chest,
    over the table's near edge and along the table top into the iPod's headphone socket.

    Frame: the WORLD-ALIGNED cafe frame (place the root at the origin, yaw 0); the default positions are world positions
    of the reference layout: a seated figure facing -Y, table top z 0.74, an iPod at [0.19502, -0.61002, 0.74] with yaw
    18.7117 deg (a `cafe_ipod` placed there). That default cord belongs to the reference seated layout.

    Anchors: plain empties `earL earR jawL jawR split chest belly` (body) and `edge lie1 lie2 plugin plug` (table),
    named `<name>_<anchor>`. The cord (`cord`, NURBS) has HOOK modifiers to them, so moving an empty bends the cord.
    To follow a body, parent the body anchors to bones (the card's `use.anchor[].bone` names the semantic bone: head,
    neck, upper_body2, upper_body); to land on another iPod, move `plug` / `plugin` / `lie2`. Other objects:
    `earL_mesh`, `earR_mesh` (buds, children of their anchors), `jack` (plug sleeve, child of `plugin`).

    Every object has the object property `earbud` = 1; the material `white` writes the shader AOV `earbud` (= 1; the
    view-layer AOV is created here) so a silhouette pass can paint the cord white. No custom properties on the root."""
    K = Kit(name, coll, root, slots)
    _setup_aov(bpy.context.view_layer)
    mat = _earbud_material(K)

    def tag(o):
        o["earbud"] = 1
        o.visible_shadow = True
        return o

    route, out = _table_route()
    pos = {n: Vector(p) for n, (p, _bone) in _BODY_ANCHORS.items()}
    pos.update(route)
    anchors = {n: tag(K.empty(n, loc=p, size=0.01)) for n, p in pos.items()}
    bpy.context.view_layer.update()
    K.purge(K.oname("cord"))
    cu = bpy.data.curves.new(K.oname("cord"), "CURVE")
    cu.dimensions = "3D"
    cu.bevel_depth = CORD_R
    cu.bevel_resolution = 2
    cu.resolution_u = 12
    cord = K.obj("cord", cu)
    cu.materials.append(mat)
    idx, hooks = 0, []
    for chain in _CORDS:
        sp = cu.splines.new("NURBS")
        sp.points.add(len(chain) - 1)
        sp.order_u = 3
        sp.use_endpoint_u = True
        for k, n in enumerate(chain):
            sag = Vector(_SAG.get(n, (0.0, 0.0, 0.0)))
            sp.points[k].co = (*(pos[n] + sag), 1.0)
            hooks.append((idx, anchors[n]))
            idx += 1
    for i, e in hooks:
        h = cord.modifiers.new(f"hook{i}", "HOOK")
        h.object = e
        h.vertex_indices_set([i])
        h.matrix_inverse = e.matrix_world.inverted()
    tag(cord)
    for n in ("earL", "earR"):
        bm = bmesh.new()
        bm_uv_sphere(bm, 1.0, seg=14, rings=10)
        tag(K.to_obj(f"{n}_mesh", bm, [mat], rot=_BUD_ROT, scale=(0.0085, 0.0075, 0.0085), parent=anchors[n]))
    # the jack: a short white sleeve on the plug end
    bm = bmesh.new()
    bmesh.ops.create_cone(bm, cap_ends=True, segments=12, radius1=0.0028, radius2=0.0028, depth=0.016)
    for f in bm.faces:
        f.smooth = True
    yaw = out.to_track_quat("Z", "Y").to_euler().z
    tag(K.to_obj("jack", bm, [mat], rot=(math.pi / 2, 0.0, yaw), parent=anchors["plugin"]))
    use = {
        "anchor": [{"name": n, "point": [round(v, 5) for v in pos[n]], "object": anchors[n].name,
                    **({"bone": _BODY_ANCHORS[n][1]} if n in _BODY_ANCHORS else {})} for n in anchors],
        "look": [{"name": n, "point": [round(v, 5) for v in pos[n]]} for n in ("earL", "earR", "plug")],
    }
    return K.card(use=use, origin="world", front="-Y")
