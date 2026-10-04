"""bmesh building blocks of the bedroom tech props (helper module, registers no builder): rounded-rectangle slabs and
frames in a local plane, flat decals, pockets cut into a flat face and corner-colour painting.

A `Frame` maps (u, v, d) to space: u, v span the plane of a face, d is the depth measured outward (toward the viewer of
that face). u x v = d for a right-handed frame, so a loop that runs counter-clockwise in (u, v) faces +d.

Colour. Plastic parts take their colour per corner from the float-colour layer `vc` (RGBA, linear; alpha is a gain
that emissive materials read): every function that makes faces takes `col`, the RGBA of the new faces' corners."""
import math

import bmesh
from mathutils import Matrix, Vector

VC = "vc"                                   # name of the per-corner colour layer


class Frame:
    """origin `o`, plane axes `u`, `v` and outward depth axis `d` (unit vectors)."""

    def __init__(self, o, u, v, d):
        self.o, self.u, self.v, self.d = Vector(o), Vector(u), Vector(v), Vector(d)

    def pt(self, u, v, d=0.0):
        return self.o + self.u * u + self.v * v + self.d * d


def front_frame(y, z=0.0):
    """Frame of a face looking toward -Y: u = +X (right), v = +Z (up), depth outward = -Y."""
    return Frame((0.0, y, z), (1, 0, 0), (0, 0, 1), (0, -1, 0))


def top_frame(z, y=0.0):
    """Frame of a face looking up: u = +X, v = +Y, depth = +Z."""
    return Frame((0.0, y, z), (1, 0, 0), (0, 1, 0), (0, 0, 1))


def back_frame(y, z=0.0):
    """Frame of a face looking toward +Y: u = -X (so it is right-handed), v = +Z, depth = +Y."""
    return Frame((0.0, y, z), (-1, 0, 0), (0, 0, 1), (0, 1, 0))


# ================================================================= outlines
def rrect(cx, cy, w, h, r, n=5):
    """Counter-clockwise outline of a rounded rectangle, 4 * (n + 1) points: corner arcs bottom-right, top-right,
    top-left, bottom-left, n + 1 points each. -> [(u, v)]. The radius is limited to what fits."""
    hw, hh = w / 2.0, h / 2.0
    r = max(min(r, hw - 1e-6, hh - 1e-6), 1e-5)
    pts = []
    for ccx, ccy, a0 in ((cx + hw - r, cy - hh + r, -90.0), (cx + hw - r, cy + hh - r, 0.0),
                         (cx - hw + r, cy + hh - r, 90.0), (cx - hw + r, cy - hh + r, 180.0)):
        for k in range(n + 1):
            a = math.radians(a0 + 90.0 * k / n)
            pts.append((ccx + r * math.cos(a), ccy + r * math.sin(a)))
    return pts


def circle(cx, cy, r, n=24):
    return [(cx + r * math.cos(2 * math.pi * i / n), cy + r * math.sin(2 * math.pi * i / n)) for i in range(n)]


# ================================================================= faces
def layer(bm):
    return bm.loops.layers.float_color.get(VC) or bm.loops.layers.float_color.new(VC)


def make_face(bm, verts, want=None, mat=0, smooth=False, col=None):
    """A face on `verts`, flipped to look along `want` (a vector, or a callable of the face centre -> vector). None for
    a degenerate or duplicate face. `col` paints its corners."""
    try:
        f = bm.faces.new(verts)
    except ValueError:
        return None
    if want is not None:
        f.normal_update()
        wv = want(f.calc_center_median()) if callable(want) else want
        if f.normal.dot(wv) < 0.0:
            f.normal_flip()
    f.material_index = mat
    f.smooth = smooth
    if col is not None:
        lay = layer(bm)
        for lp in f.loops:
            lp[lay] = col
    return f


def bridge(bm, A, B, want=None, mat=0, smooth=False, col=None, close=True):
    """Quads between two loops of equal length (the loops may be closed or open)."""
    n = len(A)
    out = []
    for i in range(n if close else n - 1):
        j = (i + 1) % n
        out.append(make_face(bm, (A[i], A[j], B[j], B[i]), want, mat, smooth, col))
    return out


def loop_at(bm, fr, pts, d):
    return [bm.verts.new(fr.pt(x, y, d)) for x, y in pts]


# ================================================================= slab
def slab(bm, fr, cx, cy, w, h, r, d0, d1, lip=0.0, n=5, ln=4, mat=0, col=None, base=False, lip_mat=None, lip0=0.0,
         ln0=3):
    """A rounded-rectangle block (a convex solid) from depth d0 to d1 in the plane of `fr`: centre (cx, cy), size w x h,
    corner radius r. The outer edge at d1 is rounded with a fillet of radius `lip` (ln steps), the edge at d0 with one of
    radius `lip0` (ln0 steps). Flat faces are flat shaded, fillets smooth. `base` adds the cap at d0 (leave it off for a
    block standing on a face). Returns the front face."""
    lm = mat if lip_mat is None else lip_mat
    ctr = fr.pt(cx, cy, 0.5 * (d0 + d1))

    def outward(c):
        return c - ctr

    def ring(inset, d):
        return loop_at(bm, fr, rrect(cx, cy, w - 2 * inset, h - 2 * inset, max(r - inset, 1e-4), n), d)

    prof = []                                                    # (inset, depth, smooth face leading to this ring)
    if lip0 > 0.0:
        prof.append((lip0, d0, False))
        for k in range(1, ln0 + 1):
            ps = 0.5 * math.pi * k / ln0
            prof.append((lip0 * (1.0 - math.sin(ps)), d0 + lip0 * (1.0 - math.cos(ps)), True))
    else:
        prof.append((0.0, d0, False))
    if lip > 0.0:
        prof.append((0.0, d1 - lip, False))
        for k in range(1, ln + 1):
            ph = 0.5 * math.pi * k / ln
            prof.append((lip * (1.0 - math.cos(ph)), d1 - lip + lip * math.sin(ph), True))
    else:
        prof.append((0.0, d1, False))
    prev = ring(prof[0][0], prof[0][1])
    if base:
        make_face(bm, prev, outward, mat, False, col)
    for inset, d, sm in prof[1:]:
        cur = ring(inset, d)
        bridge(bm, prev, cur, outward, lm if sm else mat, sm, col)
        prev = cur
    return make_face(bm, prev, outward, mat, False, col)


def frame_ring(bm, fr, outer, inner, d_base, d_top, d_in, lip=0.0, n=5, ln=3, mat=0, col=None, inner_mat=None, lip0=0.0,
               ln0=3, draft=0.0, in_shrink=0.0, closed=True):
    """A window frame (an annulus solid): outer outline `outer` = (cx, cy, w, h, r) rises from d_base to d_top, the flat top
    runs to the opening `inner` = (cx, cy, w, h, r), whose wall goes down to depth d_in (may be below d_base: a deep bezel);
    the underside closes the ring unless `closed` is False (a frame whose back is hidden: no hard edge there). The outer top
    edge has a fillet `lip` (ln steps), the outer bottom edge one of `lip0` (ln0 steps); `draft` leans the outer wall: it is
    `draft` m inset at the bottom of the wall, full size at its top. `in_shrink` moves the lower edge of the opening wall in
    by that much per side (negative: out), so the wall is not vertical. Returns the top faces."""
    im = mat if inner_mat is None else inner_mat
    ocx, ocy, ow, oh, orr = outer
    icx, icy, iw, ih, irr = inner
    c_out = fr.pt(ocx, ocy, 0.0)
    c_in = fr.pt(icx, icy, 0.0)

    def o_out(c):
        v = c - c_out
        return v - fr.d * v.dot(fr.d)

    def o_in(c):
        v = c_in - c
        return v - fr.d * v.dot(fr.d)

    def oring(inset, d):
        return loop_at(bm, fr, rrect(ocx, ocy, ow - 2 * inset, oh - 2 * inset, max(orr - inset, 1e-4), n), d)

    def iring(d, shrink=0.0):
        return loop_at(bm, fr, rrect(icx, icy, iw - 2 * shrink, ih - 2 * shrink, max(irr - shrink, 1e-4), n), d)

    prof = []                                          # (inset, depth, smooth, side) of the outer rings; side +1 top, -1 bottom
    if lip0 > 0.0:
        prof.append((draft + lip0, d_base, False, -1))
        for k in range(1, ln0 + 1):
            ps = 0.5 * math.pi * k / ln0
            prof.append((draft + lip0 * (1.0 - math.sin(ps)), d_base + lip0 * (1.0 - math.cos(ps)), True, -1))
    else:
        prof.append((draft, d_base, False, -1))
    prof.append((0.0, d_top - lip, False, 0))
    if lip > 0.0:
        for k in range(1, ln + 1):
            ph = 0.5 * math.pi * k / ln
            prof.append((lip * (1.0 - math.cos(ph)), d_top - lip + lip * math.sin(ph), True, 1))
    base = oring(*prof[0][:2])
    prev = base
    for inset, d, sm, side in prof[1:]:
        cur = oring(inset, d)
        bridge(bm, prev, cur, (lambda c, s=side: o_out(c) + fr.d * s) if sm else o_out, mat, sm, col)
        prev = cur
    top_in = iring(d_top)
    tops = bridge(bm, prev, top_in, fr.d, mat, False, col)
    bot_in = iring(d_in, in_shrink)
    bridge(bm, top_in, bot_in, o_in, im, False, col)
    if closed:
        bridge(bm, bot_in, base, -fr.d, mat, False, col)
    return tops


def flat_quad(bm, fr, cx, cy, w, h, d, mat=0, col=None, r=0.0, n=3):
    """A flat rectangle (rounded when r > 0) facing +d at depth d: an n-gon face."""
    pts = rrect(cx, cy, w, h, r, n) if r > 0 else [(cx + w / 2, cy - h / 2), (cx + w / 2, cy + h / 2),
                                                   (cx - w / 2, cy + h / 2), (cx - w / 2, cy - h / 2)]
    return make_face(bm, loop_at(bm, fr, pts, d), fr.d, mat, False, col)


def flat_poly(bm, fr, pts, d, mat=0, col=None):
    """A flat polygon (convex or not too concave: one n-gon) given counter-clockwise or not, facing +d."""
    return make_face(bm, loop_at(bm, fr, pts, d), fr.d, mat, False, col)


def flat_disc(bm, fr, cx, cy, r, d, n=24, mat=0, col=None):
    return make_face(bm, loop_at(bm, fr, circle(cx, cy, r, n), d), fr.d, mat, False, col)


def ribbon(bm, fr, pts, width, d, mat=0, col=None):
    """A flat strip of constant width along the 2D polyline `pts` at depth d, facing +d (a pen stroke)."""
    n = len(pts)
    left, right = [], []
    for i in range(n):
        a = pts[max(i - 1, 0)]
        b = pts[min(i + 1, n - 1)]
        tx, ty = b[0] - a[0], b[1] - a[1]
        ln = math.hypot(tx, ty) or 1.0
        nx, ny = -ty / ln * width / 2, tx / ln * width / 2
        left.append(bm.verts.new(fr.pt(pts[i][0] + nx, pts[i][1] + ny, d)))
        right.append(bm.verts.new(fr.pt(pts[i][0] - nx, pts[i][1] - ny, d)))
    out = []
    for i in range(n - 1):
        out.append(make_face(bm, (right[i], right[i + 1], left[i + 1], left[i]), fr.d, mat, False, col))
    return out


# ================================================================= pockets
def coplanar_faces(bm, point, normal, tol=1e-6):
    """Faces lying in the plane through `point` with unit `normal` and facing along it."""
    out = []
    for f in bm.faces:
        if f.normal.dot(normal) > 0.9999 and abs((f.calc_center_median() - point).dot(normal)) < tol:
            out.append(f)
    return out


def pocket(bm, fr, rect, depth, mat=0, col=None, floor_mat=None, d_plane=0.0, draft_deg=0.0):
    """Cut a rectangular pocket into the flat face of `bm` that lies in the plane `d = d_plane` of `fr` and contains
    `rect` = (u0, u1, v0, v1): the face is split along the rectangle, the rectangle region is pushed `depth` deeper
    (against fr.d) and its floor drawn in by depth * tan(draft_deg) on every side, so the walls lean; walls get material
    `mat`, the floor `floor_mat`. The mesh must be a closed solid (the pocket walls take their orientation from it).
    Returns (floor_faces, wall_faces)."""
    u0, u1, v0, v1 = rect
    plane_pt = fr.pt(0.0, 0.0, d_plane)

    def region():
        bm.normal_update()                       # faces made by from_arrays or by an earlier cut have no normals yet
        return coplanar_faces(bm, plane_pt, fr.d)

    for co, no in ((fr.pt(u0, 0, d_plane), fr.u), (fr.pt(u1, 0, d_plane), fr.u),
                   (fr.pt(0, v0, d_plane), fr.v), (fr.pt(0, v1, d_plane), fr.v)):
        fs = region()
        geom = list(fs) + [e for f in fs for e in f.edges] + [v for f in fs for v in f.verts]
        bmesh.ops.bisect_plane(bm, geom=list({id(g): g for g in geom}.values()), dist=1e-7, plane_co=co, plane_no=no)
    inside = []
    for f in region():
        c = f.calc_center_median() - plane_pt
        if u0 - 1e-6 <= c.dot(fr.u) <= u1 + 1e-6 and v0 - 1e-6 <= c.dot(fr.v) <= v1 + 1e-6:
            inside.append(f)
    ext = bmesh.ops.extrude_face_region(bm, geom=inside)
    moved = [g for g in ext["geom"] if isinstance(g, bmesh.types.BMVert)]
    floor = [g for g in ext["geom"] if isinstance(g, bmesh.types.BMFace)]
    bmesh.ops.translate(bm, vec=-fr.d * depth, verts=moved)
    if draft_deg:
        t = depth * math.tan(math.radians(draft_deg))
        uc, vc, hu, hv = 0.5 * (u0 + u1), 0.5 * (v0 + v1), 0.5 * (u1 - u0), 0.5 * (v1 - v0)
        for v in moved:
            rel = v.co - plane_pt
            du, dv = rel.dot(fr.u) - uc, rel.dot(fr.v) - vc
            v.co += fr.u * (du * (hu - t) / hu - du) + fr.v * (dv * (hv - t) / hv - dv)
    bmesh.ops.delete(bm, geom=inside, context="FACES_ONLY")
    bmesh.ops.recalc_face_normals(bm, faces=list(bm.faces))
    fl = set(floor)
    walls = {f for v in moved for f in v.link_faces if f not in fl}
    lay = layer(bm)
    for group, m in ((floor, mat if floor_mat is None else floor_mat), (walls, mat)):
        for f in group:
            f.material_index = m
            f.smooth = False
            if col is not None:
                for lp in f.loops:
                    lp[lay] = col
    return floor, list(walls)


def from_arrays(bm, V, Q, T, col=None, mat=0, smooth=True):
    """Append the numpy mesh (V (n, 3), quads Q (m, 4), triangles T (k, 3)) to `bm`: faces smooth shaded when `smooth`, material
    `mat`, corner colour `col`. Returns the new verts."""
    verts = [bm.verts.new(tuple(float(c) for c in p)) for p in V]
    lay = layer(bm) if col is not None else None
    for ids in list(Q) + list(T):
        try:
            f = bm.faces.new([verts[int(i)] for i in ids])
        except ValueError:
            continue
        f.material_index = mat
        f.smooth = smooth
        if lay is not None:
            for lp in f.loops:
                lp[lay] = col
    return verts


def conform(verts, fn):
    """Move every vertex of `verts` by fn(co) -> Vector (a displacement): a small detail built on a plane takes the shape of
    the curved surface it sits on."""
    for v in verts:
        v.co += fn(v.co)


# ================================================================= placing round parts
def put(dst, src, loc=(0, 0, 0), rot=None, scale=1.0):
    """Append bmesh `src` to `dst` transformed by rotation (a 3x3 Matrix, default none), uniform scale and `loc`;
    colour layers are kept. Frees `src`. Returns the new verts."""
    M = Matrix.Translation(Vector(loc)) @ ((rot.to_4x4() if rot is not None else Matrix.Identity(4))
                                            @ Matrix.Scale(scale, 4))
    vmap = {}
    for v in src.verts:
        vmap[v] = dst.verts.new(M @ v.co)
    uv_s = src.loops.layers.uv.active
    uv_d = dst.loops.layers.uv.verify() if uv_s is not None else None
    cs = src.loops.layers.float_color.get(VC)
    cd = layer(dst) if cs is not None else None
    for f in src.faces:
        try:
            nf = dst.faces.new([vmap[v] for v in f.verts])
        except ValueError:
            continue
        nf.smooth = f.smooth
        nf.material_index = f.material_index
        for a, b in zip(f.loops, nf.loops):
            if uv_s is not None:
                b[uv_d].uv = a[uv_s].uv
            if cs is not None:
                b[cd] = a[cs]
    out = list(vmap.values())
    src.free()
    return out


def rot_z_to(direction):
    """3x3 rotation taking +Z onto `direction`."""
    return Vector((0, 0, 1)).rotation_difference(Vector(direction).normalized()).to_matrix()


def paint_all(bm, col):
    """Give every corner of every face the colour `col`."""
    lay = layer(bm)
    for f in bm.faces:
        for lp in f.loops:
            lp[lay] = col


def soften(bm, width, segments=2, angle=55.0, smooth=True):
    """Round the hard edges of the solids in `bm`: every edge shared by two faces whose normals differ by `angle` degrees or
    more (convex or concave) is beveled `width` m wide in `segments` steps (clamped so neighbouring bevels never overlap); the
    new faces are smooth shaded, the flat faces stay flat. Free decal faces and open borders have no such edge and are left
    alone. Returns the number of edges beveled."""
    bm.normal_update()
    lim = math.cos(math.radians(angle))
    edges = []
    for e in bm.edges:
        if len(e.link_faces) == 2:
            n0, n1 = e.link_faces[0].normal, e.link_faces[1].normal
            if n0.length > 0.5 and n1.length > 0.5 and n0.dot(n1) < lim:
                edges.append(e)
    if not edges:
        return 0
    res = bmesh.ops.bevel(bm, geom=edges, offset=width, offset_type="OFFSET", segments=segments, profile=0.5, affect="EDGES",
                          clamp_overlap=True)
    if smooth:
        for f in res["faces"]:
            f.smooth = True
    return len(edges)
