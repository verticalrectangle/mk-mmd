"""Meshes of the paper theatre (`[[set]] kind = "paper_theatre"`; its layout and motion are mkmmd.core.paper_theatre):
numpy only, no bpy, so tests import it. A helper module: it registers no builder.

Every paper plate is cut with the cut-out kit's scissors (parent_cutout_layout.scissor: straight snips that turn a few
degrees, their joints a hair off the drawn line) and made a closed plate with rolled edges (electric_guitar_geo.slab; a
plate with holes, the moon with its rabbit, by `holed_slab`); the backdrop is cut flat and bent onto its plan; the
curtains are accordion-folded sheets (their folds deepen where they are gathered) with gilt tie-backs; the floor's cut
front edge shows the corrugation's liners and flutes over its dark inside; pencil lines are strips on the paper."""
import math
import zlib

import numpy as np

from ....core import paper_theatre as PT
from ....core import shell as S
from ..props import electric_guitar_geo as GEO
from ..props import parent_cutout_layout as PCL

MAT = {r: i for i, r in enumerate(PT.MATS)}           # role -> material index
PENCIL_W = 0.005                                      # a pencil line's width
R_EDGE = 0.005                                        # the largest rolled edge (card); paper rolls 0.35 of its thickness


def seed_of(*key):
    return zlib.crc32(":".join(str(k) for k in key).encode())


def cut(P, seed, spec, hole=False):
    """The outline P as scissors cut it, `spec` (snip, wobble, step) (None: as drawn, counter-clockwise). The points
    inside a snip sag a hair into the paper (a hole's: away from the hole), so the face's fill keeps every edge."""
    if spec is None:
        return GEO.ccw(np.asarray(P, float))
    snip, wobble, step = spec
    return PCL.scissor(P, seed, step, snip=snip, wobble=wobble, least=min(0.004, 0.25 * snip),
                       sag=(-1.0 if hole else 1.0) * min(0.00025, 0.04 * step))


def _orient(V, T, direction):
    """The triangles T turned so their normals point along `direction`."""
    P = V[T]
    n = np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0]) @ np.asarray(direction, float)
    out = T.copy()
    out[n < 0] = out[n < 0][:, ::-1]
    return out


def holed_slab(outline, holes, top, bottom, r, spacing, mat, n=2):
    """GEO.slab with holes: the outer outline and every hole get the same rolled rim (a hole's rings run clockwise, so
    its rim faces into the hole), the faces are one fill of the plate between them."""
    loops = [GEO.ccw(outline)] + [GEO.ccw(h)[::-1].copy() for h in holes]
    d_t, h_t = GEO.fillet_profile(r, n)
    Vs, Qs, fronts, backs, inners, off = [], [], [], [], [], 0
    for W in loops:
        m = len(W)
        M = S.mitre_offsets(W)
        rings = [np.column_stack([(W + d * M)[:, 0], top(*(W + d * M).T) + h, (W + d * M)[:, 1]]) for d, h in zip(d_t, h_t)]
        rings += [np.column_stack([(W + d * M)[:, 0], bottom(*(W + d * M).T) - h, (W + d * M)[:, 1]])
                  for d, h in zip(d_t[::-1], h_t[::-1])]
        R = np.stack(rings)
        K = len(R)
        idx = off + np.arange(K * m).reshape(K, m)
        j = np.arange(m)
        j2 = (j + 1) % m
        Qs.append(np.concatenate([np.stack([idx[k][j], idx[k + 1][j], idx[k + 1][j2], idx[k][j2]], 1)
                                  for k in range(K - 1)]))
        Vs.append(R.reshape(-1, 3))
        off += K * m
        fronts.append(idx[0])
        backs.append(idx[K - 1])
        inners.append(W + r * M)
    outer, hs = inners[0], inners[1:]
    Q = GEO.lattice(outer, spacing, 0.7 * spacing)
    for h in hs:
        Q = Q[~GEO.inside(Q, h) & (GEO.distance(Q, h) >= 0.7 * spacing)] if len(Q) else Q
    ring = np.concatenate(inners)
    V2 = np.vstack([ring, Q]) if len(Q) else ring
    T = GEO.delaunay(V2)
    c = V2[T].mean(1)
    keep = GEO.inside(c, outer)
    for h in hs:
        keep &= ~GEO.inside(c, h)
    T = T[keep]
    ni = len(Q)
    Vtop = np.column_stack([Q[:, 0], top(Q[:, 0], Q[:, 1]), Q[:, 1]]) if ni else np.zeros((0, 3))
    Vbot = np.column_stack([Q[:, 0], bottom(Q[:, 0], Q[:, 1]), Q[:, 1]]) if ni else np.zeros((0, 3))
    V = np.concatenate(Vs + [Vtop, Vbot])
    front_ids = np.concatenate(fronts + [off + np.arange(ni)])
    back_ids = np.concatenate(backs + [off + ni + np.arange(ni)])
    Tf, Tb = _orient(V, front_ids[T], (0.0, -1.0, 0.0)), _orient(V, back_ids[T], (0.0, 1.0, 0.0))
    mesh = S.Mesh(V, np.concatenate(Qs), np.concatenate([Tf, Tb]), mat=mat)
    return mesh if mesh.volume() > 0 else mesh.flip()


def plate_mesh(p, seed):
    """A Plate (mkmmd.core.paper_theatre) as a closed mesh in its object's frame: cut, filled, rolled, bowed, placed."""
    ring = cut(p.outline, seed, p.cut)
    holes = [cut(h, seed + 1 + k, p.cut, hole=True) for k, h in enumerate(p.holes)]
    r = min(R_EDGE, 0.35 * p.thick)
    step = p.cut[2] if p.cut else 0.004
    size = ring.max(0) - ring.min(0)
    spacing = float(np.clip(0.22 * size.min(), max(1.6 * step, 0.004), 0.12))
    if holes:                                             # a lattice finer than the holes' features keeps their edges
        spacing = float(np.clip(0.08 * min((h.max(0) - h.min(0)).min() for h in holes), 0.004, spacing))

    def front(x, z):
        return p.y + PT.bow_y(p.bow, x)

    def back(x, z):
        return p.y + p.thick + PT.bow_y(p.bow, x)
    mat = MAT[p.role]
    if holes:
        m = holed_slab(ring, holes, front, back, r, spacing, mat)
    else:
        edge = None if p.edge is None else MAT[p.edge]
        m = GEO.slab(ring, front, back, r=r, n=2, wall=1, spacing=spacing, mat=mat, mat_wall=edge, mat_back=edge)
    if p.frame is not None:
        R, t = p.frame
        m = m.apply(np.asarray(R, float), t)
    return m


def tube(p0, p1, r, role, sides=10):
    return S.tube(np.array([p0, p1], float), r, sides=sides, mat=MAT[role])


def thread():
    """A thread a metre long from the origin straight down (its object is scaled along z to the thread's length)."""
    return S.tube(np.array([[0.0, 0.0, 0.0], [0.0, 0.0, -1.0]]), PT.THREAD_R, sides=6, mat=MAT["thread"])


def brad(x, z, y_head, y_back):
    m = PCL.brad((x, z), y_head, y_back)
    m.Qm[:] = MAT["brad"]
    m.Tm[:] = MAT["brad"]
    return m


def band(path, normal, width, thick, role, lift=0.0, closed=False):
    """A closed thin band along `path` lying on a surface whose normal is `normal` (one or one per point): `width`
    across, `thick` off the surface, `lift` clear of it (a pencil line, a strip of card glued on its edge). Closed, so it
    never leaves an open edge to a renderer's shadow volumes."""
    P = np.asarray(path, float)
    n = len(P)
    N = np.broadcast_to(np.asarray(normal, float), (n, 3)).copy()
    N /= np.linalg.norm(N, axis=1, keepdims=True)
    T = (np.roll(P, -1, 0) - np.roll(P, 1, 0)) if closed else np.gradient(P, axis=0)
    A = np.cross(N, T)
    A /= np.maximum(np.linalg.norm(A, axis=1, keepdims=True), 1e-12)
    c = P + N * lift
    ring = np.stack([c - 0.5 * width * A, c + 0.5 * width * A, c + 0.5 * width * A + thick * N,
                     c - 0.5 * width * A + thick * N], 1)                                   # (n, 4, 3) a section a point
    if closed:
        m = S.skin(np.concatenate([ring, ring[:1]]), caps=(False, False), closed=True, mat=MAT[role]).weld()
    else:
        m = S.skin(ring, caps=(True, True), closed=True, mat=MAT[role])
    return m if m.volume() > 0 else m.flip()


def _strip(path, normal, width=PENCIL_W, lift=0.0004, role="pencil", closed=False):
    """A pencil line: a band 0.4 mm thick along `path` on a surface whose normal (one or per point) is `normal`."""
    return band(path, normal, width, 0.0004, role, lift, closed)


def _line(a, b, step=0.02):
    a, b = np.asarray(a, float), np.asarray(b, float)
    n = max(int(np.linalg.norm(b - a) / step), 1) + 1
    return a + np.linspace(0.0, 1.0, n)[:, None] * (b - a)


def sky_pencil(lines, sky, seed):
    """The pencil lines on the backdrop: ruled lines, compass circles and tick crosses drawn in its unrolled (s, z),
    a hand's wobble on them, bent onto it."""
    rng = np.random.default_rng(seed)
    paths = []
    for a, b in lines["lines"]:
        p = _line(a, b)
        p[:, 1] += 0.004 * np.sin(np.linspace(0.0, rng.uniform(2, 4) * math.pi, len(p)))     # not quite ruled
        paths.append((p, False))
    for (cx, cz), r in lines["circles"]:
        t = np.linspace(0.0, 2.0 * math.pi, int(2 * math.pi * r / 0.015), endpoint=False)
        paths.append((np.stack([cx + r * np.cos(t), cz + r * np.sin(t)], 1), True))
    for (cx, cz), r in lines["crosses"]:
        paths += [(_line((cx - r, cz), (cx + r, cz), 0.01), False), (_line((cx, cz - r), (cx, cz + r), 0.01), False)]
    out = []
    for p, closed in paths:
        V = PT.sky_bend(np.stack([p[:, 0], np.zeros(len(p)), p[:, 1]], 1), sky)
        _x, _y, nx, ny = PT.sky_plan(p[:, 0], sky)
        N = np.stack([nx, ny, np.zeros(len(p))], 1)
        out.append(_strip(V, N, closed=closed))
    return out


def floor_pencil(lines):
    """Pencil lines on the floor's paper: (x, y) pairs."""
    out = []
    for a, b in lines:
        p = _line(a, b)
        out.append(_strip(np.stack([p[:, 0], p[:, 1], np.zeros(len(p))], 1), (0.0, 0.0, 1.0)))
    return out


def corrugated(x0, x1, y, z0, z1):
    """The floor's cut front edge at y between z0 and z1: its dark inside, the two liners and the flutes' wave (kraft
    strips over it), standing clear of the edge's scissor wobble."""
    pitch = PT.FLUTE
    h, zm = z1 - z0, 0.5 * (z0 + z1)
    out = [band(_line((x0, y - 0.0025, zm), (x1, y - 0.0025, zm), 0.2), (0.0, -1.0, 0.0), h, 0.0008, "cardboard_dark")]
    for zc in (z1 - 0.0035, z0 + 0.0035):
        out.append(band(_line((x0, y - 0.0035, zc), (x1, y - 0.0035, zc), 0.2), (0.0, -1.0, 0.0), 0.007, 0.0008,
                        "cardboard"))
    xs = np.arange(x0, x1, 0.004)
    zs = zm + 0.5 * (h - 0.016) * np.sin(2.0 * math.pi * (xs - x0) / pitch)
    out.append(band(np.stack([xs, np.full(len(xs), y - 0.0035), zs], 1), (0.0, -1.0, 0.0), 0.0045, 0.0008,
                    "cardboard"))
    return out


def part_mesh(part, seed):
    """One object's mesh (mkmmd.core.paper_theatre.Part) in its own frame."""
    meshes = []
    bend = part.special.get("bend")
    for k, p in enumerate(part.plates):
        m = plate_mesh(p, seed_of(seed, part.name, k))
        if bend is not None and k == 0:
            m = S.Mesh(PT.sky_bend(m.V, bend), m.Q, m.T, m.Qm, m.Tm)
            m = m if m.volume() > 0 else m.flip()
        meshes.append(m)
    meshes += [tube(p0, p1, r, role) for p0, p1, r, role in part.tubes]
    meshes += [brad(*b) for b in part.brads]
    if "corrugated" in part.special:
        x0, x1, y, z0, z1 = part.special["corrugated"]
        meshes += corrugated(x0, x1, y, z0, z1)
    if "pencil_floor" in part.special:
        meshes += floor_pencil(part.special["pencil_floor"])
    if "pencil_sky" in part.special:
        meshes += sky_pencil(part.special["pencil_sky"], bend, seed_of(seed, "sky pencil"))
    return S.merge(meshes)


def curtain_mesh(c, seed):
    """A curtain (mkmmd.core.paper_theatre.Curtain) in its hinge's frame: the accordion-folded sheet (a closed thin
    solid, its hem cut a little wavy), the gilt band it is gathered with and the tassel; and its flex shape (offsets of
    every vertex at the shape key's value 1)."""
    top = float(c.inner_z.max())
    hx, _hy, hz = c.hinge
    nz = int((top - 0.03) / 0.035) + 1
    z = np.linspace(0.03, top, nz)
    k = 2 * c.pleats
    u = np.linspace(0.0, 1.0, 2 * k + 1)
    U, Z = np.meshgrid(u, z)
    rng = np.random.default_rng(seed)
    Z[0] = 0.03 + 0.010 * np.abs(np.sin(math.pi * c.pleats * u)) + rng.uniform(0.0, 0.003, len(u))
    X, Y = PT.pleat(c, U, Z)
    t = PT.PAPER
    nr, nu = U.shape
    F = np.stack([X - hx, Y - 0.5 * t, Z - hz], -1).reshape(-1, 3)
    B = np.stack([X - hx, Y + 0.5 * t, Z - hz], -1).reshape(-1, 3)
    V = np.concatenate([F, B])
    off = nr * nu
    idF = np.arange(off).reshape(nr, nu)
    i, j = np.meshgrid(np.arange(nr - 1), np.arange(nu - 1), indexing="ij")
    qf = np.stack([idF[i, j], idF[i, j + 1], idF[i + 1, j + 1], idF[i + 1, j]], -1).reshape(-1, 4)
    qb = qf[:, ::-1] + off
    loop = ([idF[0, jj] for jj in range(nu)] + [idF[ii, nu - 1] for ii in range(1, nr)]
            + [idF[nr - 1, jj] for jj in range(nu - 2, -1, -1)] + [idF[ii, 0] for ii in range(nr - 2, 0, -1)])
    a = np.array(loop)
    b = np.roll(a, -1)
    rim = np.stack([b, a, a + off, b + off], 1)
    sheet = S.Mesh(V, np.concatenate([qf, qb, rim]), mat=MAT["curtain"])
    sheet = sheet if sheet.volume() > 0 else sheet.flip()
    flex = np.zeros_like(sheet.V)
    flex[:, 1] = -PT.curtain_billow(c, sheet.V[:, 2] + hz)

    # the gilt band round the gathered paper at the tie, and its tassel on the inner side
    zt = c.tie_z
    xi = float(np.interp(zt, c.inner_z, c.inner_x))
    _x, ys = PT.pleat(c, np.linspace(0.0, 1.0, 4 * k + 1), np.full(4 * k + 1, zt))
    cx, aw, ad = 0.5 * (xi + c.outer) - hx, 0.5 * abs(xi - c.outer) + 0.012, float(np.abs(ys).max()) + 0.010
    th = np.linspace(0.0, 2.0 * math.pi, 40, endpoint=False)
    path = np.stack([cx + aw * np.cos(th), ad * np.sin(th), np.full(len(th), zt - hz)], 1)
    band = S.sweep(path, np.array([[-0.0035, -0.028], [0.0, -0.028], [0.0, 0.028], [-0.0035, 0.028]]), closed=True,
                   mat=MAT["gilt"]).weld()                # a ring sweep repeats its first section: weld the seam
    band = band if band.volume() > 0 else band.flip()
    tassel = plate_mesh(c.part, seed + 7) if c.part is not None else S.Mesh()
    extras = S.merge([band, tassel])
    out = S.merge([sheet, extras])
    flex = np.concatenate([flex, np.zeros((len(extras.V), 3))])
    flex[len(sheet.V):, 1] = -PT.curtain_billow(c, np.full(len(extras.V), zt))         # the band rides the tie
    return out, flex


def proscenium_mesh(part, seed, height):
    """The proscenium with its valance (the carrier's frame) and its flex shape (offsets at value 1)."""
    m = part_mesh(part, seed)
    flex = np.zeros_like(m.V)
    flex[:, 1] = -PT.proscenium_flex(m.V[:, 2], height)
    return m, flex
