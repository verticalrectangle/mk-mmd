"""The bedroom furniture of `bedroom_decor`: the nightstand and the wall shelf (geometry, materials, cards). See
`bedroom_nightstand` and `wall_shelf` there for their contracts.

Forms. Nothing visible is a plain cuboid: slabs and carcasses are extrusions of rounded outlines (plan corners with a real
radius) with a Bevel modifier of several mm on their edges (>= 3 segments); the books are lofted from a profile (rounded
spine, boards larger than the page block, curved fore-edge); the cassettes are shells with rounded plan corners and a
bevel. Printing on a curved surface (spines, labels) is single-sided sheets laid 0.2 - 0.6 mm over it."""
import math
import random

import bmesh
from mathutils import Matrix, Vector

from ....core import shell as SH
from ..shell import mesh_object
from . import bedroom_decor_layout as LAY
from . import bedroom_decor_soft as SOFT
from . import bedroom_decor_prints as PR
from .bedroom_decor_field import Palette
from .bedroom_decor_nodes import pattern_material
from .cafe_kit import Kit, bevel_modifier, bm_box, bm_lathe, bm_tube


def _extrude(bm, pts, w0, w1, mats, to3=None, smooth=True):
    """Prism of the counter-clockwise outline `pts` [(u, v)] from w0 to w1 along the third axis; `to3(u, v, w)` places a
    vertex (default x, y, z = u, v, w; a right-handed mapping, so the faces stay outward). mats = material indices
    (at w0, at w1, sides). Returns the new verts."""
    to3 = to3 or (lambda u, v, w: (u, v, w))
    lo = [bm.verts.new(to3(u, v, w0)) for u, v in pts]
    hi = [bm.verts.new(to3(u, v, w1)) for u, v in pts]
    n = len(pts)
    mb, mt, ms = mats
    faces = [(lo[::-1], mb), (hi, mt)] + [((lo[i], lo[(i + 1) % n], hi[(i + 1) % n], hi[i]), ms) for i in range(n)]
    for vs, mi in faces:
        f = bm.faces.new(vs)
        f.material_index = mi
        f.smooth = smooth
    return lo + hi


def _rim(o, width, segments=3, angle=35.0):
    """Rounded edges: a Bevel modifier on the sharp edges of `o`."""
    bevel_modifier(o, width=width, segments=segments, limit="ANGLE", angle=angle)
    return o


def _lacquer(K, base, colour, rough=0.32, coat=0.35):
    return K.simple_mat(base, colour, rough=rough, coat=coat, spec=0.5)


def _sheet(bm, pts, mat):
    """One single-sided polygon (a printed decal): pts in order, counter-clockwise seen from the front."""
    f = bm.faces.new([bm.verts.new(p) for p in pts])
    f.material_index = mat
    f.smooth = False
    return f


def _disc(bm, cx, cy, z, r, mat, n=20):
    f = bm.faces.new([bm.verts.new((cx + r * math.cos(2 * math.pi * k / n), cy + r * math.sin(2 * math.pi * k / n), z))
                      for k in range(n)])
    f.material_index = mat
    f.smooth = False
    return f


# ================================================================= nightstand
def build_nightstand(name, coll, root, slots=None):
    K = Kit(name, coll, root, slots)
    pal = Palette(colors=K.colors)
    z0, z1 = LAY.NS_BODY_Z
    zc = (z0 + z1) / 2

    # ---- top slab: terrazzo laminate on top, a teal bullnose edge; plan corners 45 mm, both rims rolled over
    laminate = pattern_material(K, "laminate", pal, PR.laminate(), "xy", rough=0.34, coat=0.25, spec=0.5, bump=0.03)
    edge = _lacquer(K, "edge", K.blend(pine=.9, foam=.1))
    top_t = LAY.NS_TOP_T
    slab = SOFT.soft_slab(SH.rrect(LAY.NS_W, LAY.NS_D, 0.045, 6), top_t, 0.011, 0.006)
    SOFT.set_roles(slab, lambda c, n: ((n[:, 2] > 0.3) & (c[:, 2] > top_t - 0.0125)).astype(int) ^ 1)
    mesh_object(K.oname("top"), slab, coll, root, lambda r: (laminate, edge)[r], loc=(0.0, 0.0, LAY.NS_H - top_t))

    # ---- the drawer unit: a pale moulded carcass (plan corners 50 mm, top and bottom rims rolled over 26 / 22 mm), a dark
    # reveal, the Memphis drawer front (a crowned pad with a rolled rim), a tubular pull
    body_m = _lacquer(K, "body", pal.blend(paper=.78, subtle=.14, rose=.08), rough=0.42, coat=0.15)
    mesh_object(K.oname("body"), SOFT.soft_slab(SH.rrect(LAY.NS_BODY_W, LAY.NS_BODY_D, 0.05, 6), z1 - z0, 0.026, 0.022, 5),
                coll, root, lambda r: body_m, loc=(0.0, 0.0, z0))
    fw, fh = 0.340, 0.098
    face_y = -LAY.NS_BODY_D / 2
    face = SH.rot_matrix(rx=90.0)                                       # a pad built face-up (+z) turned to face -y
    reveal_m = K.simple_mat("reveal", K.blend(base=.85, overlay=.15), rough=0.6)
    reveal = SOFT.soft_slab(SH.rrect(fw + 0.008, fh + 0.008, 0.034, 6), 0.002, 0.0007, 0.0007, 2).apply(face)
    mesh_object(K.oname("reveal"), reveal, coll, root, lambda r: reveal_m, loc=(0.0, face_y + 0.0005, zc))
    drawer_m = pattern_material(K, "drawer", pal, PR.drawer_panel(fw / 2, fh / 2), "xz", rough=0.30, coat=0.35, spec=0.5)
    plain_m = _lacquer(K, "drawer_edge", K.blend(love=.75, rose=.25))
    pad_t, crown = 0.012, 0.004
    pad = SOFT.soft_slab(SH.rrect(fw, fh, 0.03, 6), pad_t, 0.005, 0.003, 4, dome=crown, cap_rings=3)
    SOFT.set_roles(pad, lambda c, n: ((n[:, 2] > 0.3) & (c[:, 2] > pad_t - 0.0055)).astype(int) ^ 1)
    mesh_object(K.oname("drawer"), pad.apply(face), coll, root, lambda r: (drawer_m, plain_m)[r],
                loc=(0.0, face_y + 0.0075, zc))                         # crown apex at face_y - 8.5 mm: the old front plane
    pull_m = K.simple_mat("pull", K.blend(gold=.85, rose=.15), rough=0.28, metal=0.7, spec=0.6)
    bm = bmesh.new()
    py = face_y - 0.0085
    bm_tube(bm, [Vector((-0.062, py - 0.0078, zc)), Vector((0.0, py - 0.0078, zc)), Vector((0.062, py - 0.0078, zc))], 0.0045,
            sides=14, cap="round")
    for sx in (-1, 1):
        bm_tube(bm, [Vector((sx * 0.052, py, zc)), Vector((sx * 0.052, py - 0.0078, zc))], 0.0034, sides=10, cap="none")
    K.to_obj("pull", bm, [pull_m])

    # ---- tubular legs, a lower shelf (plan corners r = 15 mm), metal feet
    leg_m = _lacquer(K, "leg", K.blend(pine=.9, foam=.1), rough=0.30, coat=0.4)
    foot_m = K.simple_mat("foot", K.blend(subtle=.6, text=.4), rough=0.3, metal=0.9, spec=0.6)
    bm, fm = bmesh.new(), bmesh.new()
    for sx in (-1, 1):
        for sy in (-1, 1):
            top, bot = (Vector(p) for p in LAY.leg_points(sx, sy))
            mid = (top + bot) * 0.5
            bm_tube(bm, [top, mid, bot], lambda t: 0.0115 - 0.0020 * t, sides=16, cap=("none", "flat"))
            bm_lathe(fm, [(0.0, 0.0), (0.0090, 0.0), (0.0108, 0.0016), (0.0114, 0.0040), (0.0114, 0.0100), (0.0098, 0.0125),
                          (0.0, 0.0125)], segs=20, center=(bot.x, bot.y, 0.0))
    K.to_obj("legs", bm, [leg_m])
    K.to_obj("feet", fm, [foot_m])
    shelf_m = _lacquer(K, "shelf", K.blend(iris=.85, text=.15))
    sx_, sy_ = LAY.leg_at(1, 1, LAY.NS_SHELF_Z)
    shelf_t = LAY.NS_SHELF_T                                            # a board with both rims rolled, plan corners 40 mm
    mesh_object(K.oname("shelf"), SOFT.soft_slab(SH.rrect(2 * sx_, 2 * sy_, 0.04, 6), shelf_t, 0.006, 0.004), coll, root,
                lambda r: shelf_m, loc=(0.0, 0.0, LAY.NS_SHELF_Z - shelf_t / 2))

    # ---- hidden collision boxes: the top slab and the drawer unit
    bm = bmesh.new()
    bm_box(bm, (LAY.NS_W, LAY.NS_D, LAY.NS_TOP_T), (0.0, 0.0, LAY.NS_H - LAY.NS_TOP_T / 2))
    col_top = K.collider(K.to_obj("col_top", bm, []))
    bm = bmesh.new()
    bm_box(bm, (LAY.NS_BODY_W, LAY.NS_BODY_D + 0.02, z1 - z0), (0.0, -0.01, zc))
    col_body = K.collider(K.to_obj("col_body", bm, []))

    top_z = LAY.NS_H
    shelf_top = LAY.NS_SHELF_Z + LAY.NS_SHELF_T / 2
    use = {"rest": [{"name": "top", "type": "plane", "center": [0.0, 0.0, top_z], "normal": [0, 0, 1],
                     "size": [LAY.NS_W, LAY.NS_D]},
                    {"name": "shelf", "type": "plane", "center": [0.0, 0.0, round(shelf_top, 4)], "normal": [0, 0, 1],
                     "size": [round(2 * sx_ - 0.04, 3), round(2 * sy_ - 0.04, 3)]}],
           "look": [{"name": "top", "point": [0.0, 0.0, top_z]}]}
    colliders = [{"type": "box", "object": col_top.name, "rnd": 0.004, "tag": name},
                 {"type": "box", "object": col_body.name, "rnd": 0.004, "tag": name}]
    return K.card(use=use, colliders=colliders, origin="floor_center", front="-Y")


# ================================================================= wall shelf
def _spine_sheet(bm, w, d, x0, x1, z0, z1, off, mat, n=8):
    """Printing on a book's round spine: a single-sided strip over x0..x1 (across the thickness) and z0..z1, laid `off`
    over the curve."""
    pts = [LAY.spine_point(w, d, x0 + (x1 - x0) * k / n, off) for k in range(n + 1)]
    lo = [bm.verts.new((x, y, z0)) for x, y in pts]
    hi = [bm.verts.new((x, y, z1)) for x, y in pts]
    for k in range(n):
        f = bm.faces.new((lo[k], lo[k + 1], hi[k + 1], hi[k]))
        f.material_index = mat
        f.smooth = True


def _book(bm, b, M, mi, rng):
    """One book in its own frame (x across the spine from 0, y from the fore-edge 0 to the spine at -d, z up) moved by M:
    the case (boards + round spine) a little taller and deeper than the page block, the page block with a bowed
    fore-edge, bands and a label printed on the spine."""
    w, h, d = b["w"], b["h"], b["d"]
    tc = LAY.BOOK_COVER_T * rng.uniform(0.9, 1.25)
    over = LAY.BOOK_OVER * rng.uniform(0.85, 1.25)
    ci = mi[b["slot"]]
    vs = _extrude(bm, LAY.book_case_outline(w, d, tc), 0.0, h, (ci, ci, ci))
    vs += _extrude(bm, LAY.book_block_outline(w, d, tc, over), over, h - over, (mi["pages"],) * 3)
    n0 = len(bm.verts)
    for z, key in [(h - 0.032, "gold"), (0.030, "gold"), (h - 0.044, "ink")][: b["bands"]]:
        _spine_sheet(bm, w, d, 0.0, w, z - 0.0025, z + 0.0025, 0.00025, mi[key])
    zl = h * 0.60
    _spine_sheet(bm, w, d, w * 0.13, w * 0.87, zl - 0.024, zl + 0.024, 0.0004, mi["paper"])
    for dz, f in ((-0.012, 0.52), (0.0, 0.42), (0.012, 0.52)):
        _spine_sheet(bm, w, d, w * (0.5 - 0.45 * f), w * (0.5 + 0.45 * f), zl + dz - 0.0018, zl + dz + 0.0018, 0.0006,
                     mi["ink"])
    bm.verts.ensure_lookup_table()
    vs += list(bm.verts)[n0:]
    bmesh.ops.transform(bm, matrix=M, verts=vs)


def _cassette(bm, cm, tp, top, mi):
    """A cassette lying flat (spine toward -y) at tp = {x, y, yaw}, centre height `z`: shell with rounded plan corners
    (the bevel modifier rounds its edges), a printed spine and, on the top one, the label, window and hubs."""
    W, D, H = LAY.CASSETTE
    z = tp["z"]
    vs = _extrude(bm, LAY.rounded_rect(W / 2, D / 2, LAY.CASSETTE_CORNER), -H / 2, H / 2, (mi["case"],) * 3)
    n0 = len(bm.verts)
    yf = -D / 2 - 0.0002
    _sheet(bm, [(-0.041, yf, -H / 2 + 0.0026), (0.041, yf, -H / 2 + 0.0026), (0.041, yf, H / 2 - 0.0026),
                (-0.041, yf, H / 2 - 0.0026)], mi["paper"])
    yf -= 0.0001
    _sheet(bm, [(-0.041, yf, -H / 2 + 0.0026), (-0.014, yf, -H / 2 + 0.0026), (-0.014, yf, H / 2 - 0.0026),
                (-0.041, yf, H / 2 - 0.0026)], mi[tp["slot"]])
    for x0, x1 in ((-0.006, 0.020), (0.026, 0.034)):
        _sheet(bm, [(x0, yf - 0.0001, -0.0006), (x1, yf - 0.0001, -0.0006), (x1, yf - 0.0001, 0.0006),
                    (x0, yf - 0.0001, 0.0006)], mi["ink"])
    if top:
        zt = H / 2 + 0.0002
        _sheet(bm, [(-0.0385, -0.0105, zt), (0.0385, -0.0105, zt), (0.0385, 0.0275, zt), (-0.0385, 0.0275, zt)], mi["paper"])
        zt += 0.0001
        _sheet(bm, [(-0.0385, 0.0205, zt), (0.0385, 0.0205, zt), (0.0385, 0.0275, zt), (-0.0385, 0.0275, zt)], mi[tp["slot"]])
        _sheet(bm, [(-0.0245, -0.0062, zt), (0.0245, -0.0062, zt), (0.0245, 0.0062, zt), (-0.0245, 0.0062, zt)], mi["case"])
        for y in (0.0135, 0.0165):
            _sheet(bm, [(-0.030, y, zt), (0.012, y, zt), (0.012, y + 0.0011, zt), (-0.030, y + 0.0011, zt)], mi["ink"])
        for sx in (-1, 1):
            _disc(bm, sx * 0.0128, 0.0, zt + 0.0001, 0.0057, mi["paper"])
            _disc(bm, sx * 0.0128, 0.0, zt + 0.0002, 0.0036, mi["case"])
    bm.verts.ensure_lookup_table()
    vs += list(bm.verts)[n0:]
    M = Matrix.Translation((tp["x"], tp["y"], z)) @ Matrix.Rotation(math.radians(tp["yaw"]), 4, "Z")
    bmesh.ops.transform(bm, matrix=M, verts=vs)


def build_wall_shelf(name, coll, root, slots=None):
    K = Kit(name, coll, root, slots)
    pal = Palette(colors=K.colors)
    seed = int(float(K.slots.get("seed", 0)))
    lay = LAY.shelf_layout(seed)
    d, t = LAY.SH_D, LAY.SH_T

    # ---- the board: pale lacquer with a teal front edge, plan corners r = 9 mm, edges rounded
    top_m = _lacquer(K, "board", pal.blend(paper=.72, gold=.12, rose=.16), rough=0.42, coat=0.2)
    edge_m = _lacquer(K, "edge", K.blend(pine=.9, foam=.1))
    bm = bmesh.new()
    pts = LAY.rounded_poly([(LAY.SH_W / 2, 0.0), (-LAY.SH_W / 2, 0.0), (-LAY.SH_W / 2, -d), (LAY.SH_W / 2, -d)],
                           [0.0, 0.0, 0.009, 0.009], 5)
    _extrude(bm, pts, -t, 0.0, (1, 0, 1))
    _rim(K.to_obj("board", bm, [top_m, edge_m]), 0.006, 4)

    # ---- the books: real books, one mesh, one material per cover colour
    used = list(dict.fromkeys(b["slot"] for b in lay["books"]))
    mats = [K.simple_mat(f"book_{s}", K.blend(**{s: .80, "overlay": .20}), rough=0.62, sheen=0.3) for s in used]
    mats += [K.simple_mat("pages", pal.blend(paper=.85, gold=.15), rough=0.85),
             K.simple_mat("band_gold", K.blend(gold=.9, rose=.1), rough=0.4, metal=0.5),
             K.simple_mat("band_ink", pal.blend(ink=1.0), rough=0.6),
             K.simple_mat("label", pal.blend(paper=.9, gold=.1), rough=0.7)]
    mi = {s: i for i, s in enumerate(used)}
    mi.update({"pages": len(used), "gold": len(used) + 1, "ink": len(used) + 2, "paper": len(used) + 3})
    rng = random.Random(seed * 101 + 5)
    bm = bmesh.new()
    for b in lay["books"]:
        pull = rng.choice((0.0, 0.0, 0.0, 0.004, 0.007))                # a few stand proud of the row
        M = Matrix.Translation((b["x"], -0.012 - pull, 0.0)) @ Matrix.Rotation(-math.radians(b["lean_deg"]), 4, "Y")
        _book(bm, b, M, mi, rng)
    _rim(K.to_obj("books", bm, mats), 0.0008, 2, angle=40.0)

    # ---- a stack of cassettes lying flat, spines out
    W, D, H = LAY.CASSETTE
    cm = [K.simple_mat("tape_case", K.blend(base=.8, overlay=.2), rough=0.25, coat=0.3),
          K.simple_mat("tape_label", pal.blend(paper=.9, gold=.1), rough=0.7),
          K.simple_mat("tape_ink", pal.blend(ink=1.0), rough=0.5)]
    cm += [K.simple_mat(f"tape_{s}", K.blend(**{s: .9, "text": .1}), rough=0.4) for s in LAY.TAPE_SLOTS]
    tmi = {"case": 0, "paper": 1, "ink": 2, **{s: 3 + i for i, s in enumerate(LAY.TAPE_SLOTS)}}
    bm = bmesh.new()
    for i, tp in enumerate(lay["tapes"]):
        _cassette(bm, cm, {**tp, "z": i * H + H / 2}, i == len(lay["tapes"]) - 1, tmi)
    _rim(K.to_obj("tapes", bm, cm), 0.0011, 3, angle=40.0)

    # ---- a small totem: a rounded cylinder, a ball, a cone with a soft tip
    bm = bmesh.new()
    ox, oy = lay["ornament_x"], -0.10
    bm_lathe(bm, [(0.0, 0.0), (0.0245, 0.0), (0.0272, 0.0020), (0.0280, 0.0050), (0.0280, 0.0200), (0.0272, 0.0235),
                  (0.0245, 0.0258), (0.0, 0.0260)], segs=32, mat=0, center=(ox, oy, 0.0))
    bm_lathe(bm, [(0.0, 0.0260), (0.0123, 0.0275), (0.0218, 0.0345), (0.0261, 0.0485), (0.0218, 0.0625), (0.0123, 0.0695),
                  (0.0, 0.0710)], segs=32, mat=1, center=(ox, oy, 0.0))
    bm_lathe(bm, [(0.0, 0.0705), (0.0212, 0.0705), (0.0234, 0.0716), (0.0236, 0.0735), (0.0200, 0.0830), (0.0100, 0.1010),
                  (0.0030, 0.1140), (0.0, 0.1180)], segs=32, mat=2, center=(ox, oy, 0.0))
    K.to_obj("ornament", bm, [K.simple_mat("orn_base", K.blend(foam=.85, text=.15), rough=0.35, coat=0.3),
                              K.simple_mat("orn_ball", K.blend(love=.9, rose=.1), rough=0.3, coat=0.4),
                              K.simple_mat("orn_cone", K.blend(gold=.9, rose=.1), rough=0.35, coat=0.3)])

    # ---- hidden collision boxes: the board and the row of books
    bm = bmesh.new()
    bm_box(bm, (LAY.SH_W, d, t), (0.0, -d / 2, -t / 2))
    col_board = K.collider(K.to_obj("col_board", bm, []))
    bx0, bx1 = lay["books_x"]
    top_h = max(b["h"] for b in lay["books"])
    bm = bmesh.new()
    bm_box(bm, (bx1 - bx0, 0.17, top_h), ((bx0 + bx1) / 2, -0.012 - 0.085, top_h / 2))
    col_books = K.collider(K.to_obj("col_books", bm, []))

    f0, f1 = lay["free"]
    use = {"rest": [{"name": "top", "type": "plane", "center": [0.0, -d / 2, 0.0], "normal": [0, 0, 1], "size": [LAY.SH_W, d]},
                    {"name": "free", "type": "plane", "center": [round((f0 + f1) / 2, 4), -d / 2, 0.0], "normal": [0, 0, 1],
                     "size": [round(f1 - f0, 4), round(d - 0.03, 4)]}],
           "look": [{"name": "top", "point": [0.0, -d / 2, 0.0]}]}
    colliders = [{"type": "box", "object": col_board.name, "rnd": 0.004, "tag": name},
                 {"type": "box", "object": col_books.name, "rnd": 0.004, "tag": name}]
    return K.card(use=use, colliders=colliders, origin="wall_center", front="-Y",
                  zones={"books": [round(v, 4) for v in lay["books_x"]], "tapes": [round(v, 4) for v in lay["tapes_x"]],
                         "ornament": [round(lay["ornament_x"] - 0.03, 4), round(lay["ornament_x"] + 0.03, 4)],
                         "free": [round(v, 4) for v in lay["free"]]})
