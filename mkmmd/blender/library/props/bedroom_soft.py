"""The 80s bedroom's soft furnishings: `bed_single` (a single bed with a Memphis quilt) and `rug_80s` (a Memphis rug).

Procedural props for the night room (palette `rose-pine-moon`): every colour is a palette slot blended in linear light,
every material is procedural shader nodes for EEVEE Next (no image files), the geometry is built with numpy and is
deterministic (no clock, no random state). Builder options are the non-colour entries of `[[prop]] slots` (below).
Helper modules (they register nothing): `bedroom_soft_geo` (cloth drape, shells, pillows, rug body and fringe),
`bedroom_soft_pattern` (the print: motifs, layout, lookup tables) and `bedroom_soft_frame` (the frame's parts and their
meshes, built with the shell toolkit `mkmmd.core.shell`), all pure numpy; `bedroom_soft_nodes` (a node expression builder
and the print's shader), `bedroom_soft_mats` (the materials). tests/test_bedroom_soft.py checks the pure maths.

Form. Nothing is a hard-edged cuboid: every board is a `shell.rounded_box` with real fillets (9-14 mm on the visible
ones, so no edge turns more than 22 degrees), the posts are soft-cornered tapered legs with rounded ends, the finials are
turned, cloth is wrapped from profiles. `mkmmd.core.form.analyse` on the whole bed and rug scores 0.01 (frame 0.00,
quilt 0.03, mattress 0.03, the rest 0.00) and the hard-edge audit (dihedral >= 75 degrees) finds 0 m on every object.

bed_single
==========
Frame. Origin = the floor under the centre of the footprint (1.00 x 2.00 m), +Z up, the HEAD end at local +Y (push it
against a wall), the foot end at -Y, front = -Y (the side a person faces). Mattress top z = 0.50; headboard top 0.95,
footboard and foot posts 0.55 (the quilt crosses the footboard and hangs outside it); side rails z 0.20-0.30, the
mattress (0.96 x 1.89 m) rests on their top edge. The visible bounds (the card's `size`) are about 1.14 x 2.05 x 1.00:
the quilt's hang adds up to 7 cm on each side and 5 cm at the foot, the finials the last 5 cm of height.

Parts, as objects named `<name>_<part>`:
  frame      one mesh of planed pine, 40 parts: four 55 mm posts (rounded-square section, legs tapering to 78 % at the
             floor, rounded ends), two side rails 25 x 120 mm, slatted head and foot boards (rails 30 x 65-80 mm and nine
             20 x 42 mm slats each, equal gaps), the cleats and base slats under the mattress. The posts are sweeps, the
             boards `rounded_box`es (the visible edges rounded over 9-14 mm like a bullnose plane, the hidden ones 12-14
             mm too), boards run 10 mm into the posts; custom normals (flat faces stay flat); UV: u along the board, v
             across (grain). `frame = "tube"`: powder-coated steel tubes (posts, an arched head rail, 13 bars each end,
             open ends inside the other parts) with the same rails and base.
  headboard  only with `headboard = "padded"`: a tufted slab (`rounded_box`, 32 mm fillets; UV = x, z in metres) instead
             of the slats.
  finials    turned gold finials (collar, cove, neck, ball: 24-sided lathes, 5.2 cm tall) on the head posts.
  mattress   a slab with 35 mm rounded edges and a diamond-stitched ticking; piping: two round seams (12-sided tubes,
             z 0.395 and 0.335).
  sheet      the fitted sheet: a thin shell over the top that wraps 6 cm down the four sides, slightly wrinkled.
  quilt      the key piece: ONE closed shell (pattern side = material 0, lining = 1, binding rim = 2) made by wrapping a
             flat 1.37 x 2.06 m sheet over the bed. It bends over the mattress edge and hangs 0.20 m below the mattress
             top on both long sides, crosses the footboard and hangs 0.12 m outside it, and is ROLLED BACK at y = +0.495
             (0.45 m from the head end of the mattress): a 0.27 m band of its lining lies over it (the fold's thickness
             is modelled, the band's side flaps hang outside the lower layer's), exposing the sheet and the pillows. The
             field sags and ripples softly (up to ~2 cm), the hang has pleats and a scalloped hem, and the corner draws
             in round the post. A signed-distance push keeps the cloth out of the mattress, rails, foot posts and
             footboard, then a constrained relaxation smooths the bulge the posts leave (no sharp fold).
             UV "UVMap" = the flat sheet in metres (u across from the left edge, v along from the foot hem),
             UV "edge" = the distance to the hem (u channel). About 22k triangles.
  pillow_a, pillow_b   two plump pillows (the second leans on the first against the headboard) with pinched seams and
             corners, a dent and an Oxford flange stitch 38 mm in; one pillowcase material.
  col_top, col_bed, col_head   hidden colliders (see Card).
About 47k triangles in all (quilt 22k, frame 12k, sheet 3k, pillows 3k each); the quilt's material has 360 nodes.

The print. The quilt's flat sheet carries a lattice of 5 x 8 cells (0.27 x 0.26 m), each with one motif or none: a
triangle, three dots, a ring, a squiggle, a zigzag, bold stripes, an arch (rainbow) or a plus, with a seeded random
rotation, size and offset, never the same motif or colour as the cell to its left or below. The shader reads the cell's
parameters from colour-ramp lookup tables (made by `bedroom_soft_pattern.layout / lut_stops`) and draws the motif as a
signed-distance function (anti-aliased over 4 mm). It adds big soft tone-on-tone blobs and small specks between the
motifs, the border bands (from the hem inwards: a 1.7 cm binding in `love`, a 3.4 cm band of ink = `base`, a thin `gold`
line), the diamond quilting (11.5 cm pitch: puffs by bump, stitch lines by darkening, flat toward the hem) and a faint
weave. Nothing is baked, so a project can change the palette.

Slots (builder options; every colour option is a palette slot NAME, never a hex value)
  pattern          "memphis" (default, all eight motifs), "zigzag" (waves and bars), "triangles", "plain" (only the
                   quilting and the border)
  seed             integer (default 7): the print's layout and the deterministic noise of the quilt, sheet and pillows
  quilt_colors     2-8 slot names (a comma separated string or a list) the motifs are drawn in; the first acts as ink
                   (default "base,love,gold,pine,iris,foam"); another list makes every colour equally likely (ink x2)
  quilt_ground     slot of the quilt's ground (default: a blend of text, iris, rose and overlay, pale lavender)
  pillowcase       slot of the pillowcase (default: a blend of rose and love, pink)
  frame            "wood" (default: painted pine boards) or "tube" (steel tubes)
  frame_color      slot of the paint or powder coat (default: a blend of pine, foam and text, a pale teal)
  headboard        "slats" (default) or "padded";  headboard_color: slot of the padded slab (default "iris")
The card's `options` echoes the resolved values.

Card (numbers for the defaults)
  size         [1.14, 2.05, 1.00]; origin "floor_center"; front "-Y"; slots: the palette slots used.
  use.rest     "top": a plane, the quilt's upper surface over the mattress: centre [0, -0.2025, 0.542], normal
               [0, 0, 1], size [0.90, 1.395] (x from -0.45 to 0.45, y from the foot edge -0.90 to the roll at +0.495
               where the lining band begins; the field is flat to a centimetre or two)
  use.look     "pillow": the highest point of the upper pillow, about [0.10, 0.85, 0.76]; "quilt": the centre of the
               field [0, -0.2025, 0.542]
  colliders    three hidden boxes tagged with the prop name: `<name>_col_top` (the mattress under its quilt, from the
               foot hang to the roll: x +-0.51, y -1.03 .. +0.525, z 0.30 .. 0.56; rnd 0.02), `<name>_col_bed`
               (mattress and sheet at the head end, behind the roll: x +-0.48, y 0.525 .. 0.945, z 0.30 .. 0.508;
               rnd 0.01) and `<name>_col_head` (the headboard: x +-0.50, y 0.97 .. 1.00, z 0.40 .. 0.95; rnd 0.01)
  frame_bounds the frame's own bounds {"min": [-0.5, -1.0, 0], "max": [0.5, 1.0, 0.95]}: the visible bounds are wider
               because of the quilt, so place the bed against a wall by its frame, not by the quilt's flap.
  options      the resolved builder options (pattern, seed, quilt_colors, frame, headboard).
  No lights, no custom properties.

rug_80s
=======
A rectangular wool rug. Origin = the floor under its centre, front = -Y, +Z up. The woven body is 1.80 (x) x 1.20 (y)
x 0.012 (z) and lies flat (the card says `"flat": true`: people and furniture stand on it); the fringe of the two short
edges reaches 5.8 cm past x = +-0.9, so the visible bounds (the card's `size`) are [1.916, 1.20, 0.013].

Objects: `<name>_body` (a closed slab with a half-round edge of 6 mm radius and softly rounded plan corners, the top
undulating by less than a millimetre; materials: wool on top, edge tape and felt for the rim and underside, about 5k
triangles) and `<name>_fringe` (74 tassels per short edge, every 16 mm: a soft knot, a 24-quad lump, and two round yarn
strands, 8-sided tubes tapering from 4 to 2 mm, 5.0-5.8 cm long beyond the edge, fanning out and wobbling, lying on the
floor; about 21k triangles; plied-twist yarn shader).

The wool shader: a Memphis print in the flat (x, y) metres of the top (UV "UVMap"): an 11.6 cm border (an ink edge
line, a lavender band, a `love` line, a sawtooth of ink triangles on `gold`, a `love` line) round a `pine` teal field
with the same lattice print as the quilt (7 x 4 cells; motifs in `base, love, gold, iris, foam` and a lavender),
tone-on-tone blobs and specks. The pile is two layers of noise (fibres, tufts) as bump and as a darker tone down
between the tufts, with a slow dye drift and a soft sheen. It reads as a mid-value teal rug with light and bright
accents on a dark floor.

Slots: `seed` (integer, default 5: the print and the fringe), `rug_ground` (slot of the field, default a blend of pine
and overlay), `rug_colors` (2-8 slot names or the default "base,love,gold,iris,foam,lavender", where "lavender" stands
for a blend of text and iris).

Card: size [1.916, 1.20, 0.013]; `use.rest` "top": plane centre [0, 0, 0.012], normal [0, 0, 1], size [1.8, 1.2]; no
colliders (an empty list); extra keys `flat` (true), `body` ([1.8, 1.2, 0.012]), `fringe` ({"length", "sides": ["-X",
"+X"], "tassels_per_side": 74}) and `options`.
"""
import math

import bmesh
import bpy
import numpy as np
from mathutils import Euler, Vector

from .. import shell as SH
from . import bedroom_soft_frame as FR
from . import bedroom_soft_geo as G
from . import bedroom_soft_mats as M
from . import bedroom_soft_pattern as PT
from . import register
from .cafe_kit import Kit, bm_lathe, bm_tube


# ================================================================================================ options
def opt(K, key, default):
    """A builder option: an entry of `slots` that is not a palette colour."""
    v = K.slots.get(key)
    return default if v in (None, "") else v


def slot_names(K, key, default):
    """An option of 2-8 slot names (a comma separated string or a list) -> list of names."""
    v = opt(K, key, default)
    names = [str(n).strip() for n in (v if isinstance(v, (list, tuple)) else str(v).split(",")) if str(n).strip()]
    if not 2 <= len(names) <= PT.MAX_COLORS:
        raise ValueError(f"{K.name}: {key} needs 2 to {PT.MAX_COLORS} slot names (got {names})")
    return names


def slot_color(K, name):
    """A palette slot given by name (an option's value) -> linear RGBA."""
    try:
        return K.slot(str(name))
    except KeyError:
        raise ValueError(f"{K.name}: {name!r} is not a palette slot") from None


# ================================================================================================ meshes
def shell_obj(K, base, shell, mats, loc=(0.0, 0.0, 0.0), rot=(0.0, 0.0, 0.0), uv_names=("UVMap", "edge")):
    """A pure-geometry `Shell` as a smooth-shaded mesh object <name>_<base> with the given materials (index = shell.mat)
    and the two UV maps."""
    name = K.oname(base)
    K.purge(name)
    me = bpy.data.meshes.new(name)
    me.from_pydata(shell.V.tolist(), [], shell.F.tolist())
    me.polygons.foreach_set("use_smooth", np.ones(len(shell.F), dtype=bool))
    me.polygons.foreach_set("material_index", shell.mat)
    flat = shell.F.reshape(-1)
    for nm, uv in zip(uv_names, (shell.uv0, shell.uv1)):
        layer = me.uv_layers.new(name=nm)
        layer.data.foreach_set("uv", uv[flat].astype(np.float32).reshape(-1))
    for m in mats:
        me.materials.append(m)
    me.update()
    return K.obj(base, me, loc=loc, rot=rot)


# ================================================================================================ the bed's parts
def _frame_object(K, mat, headboard):
    """The wooden frame: one object of planed boards and soft posts (shell toolkit), custom normals from the vertices."""
    return SH.mesh_object(K.oname("frame"), FR.frame_mesh(headboard), K.coll, K.root, lambda role: mat)


def _tube_object(K, mat, headboard):
    return SH.mesh_object(K.oname("frame"), FR.tube_mesh(headboard), K.coll, K.root, lambda role: mat)


def _padded_object(K, mat):
    """The padded headboard: a rounded slab, UV = (x, z) metres for the tufting."""
    return SH.mesh_object(K.oname("headboard"), FR.padded_mesh(), K.coll, K.root, lambda role: mat)


def _mattress(K, mat):
    bm = bmesh.new()
    g = bmesh.ops.create_cube(bm, size=1.0)
    bmesh.ops.scale(bm, vec=Vector((2 * G.MAT_HX, 2 * G.MAT_HY, G.MAT_Z1 - G.MAT_Z0)), verts=g["verts"])
    bmesh.ops.translate(bm, vec=Vector((0, 0, (G.MAT_Z0 + G.MAT_Z1) / 2)), verts=g["verts"])
    bmesh.ops.bevel(bm, geom=list(bm.edges), offset=G.MAT_R, segments=4, profile=0.5, affect="EDGES")
    bm.normal_update()
    uvl = bm.loops.layers.uv.verify()
    for f in bm.faces:
        n = f.normal
        f.smooth = max(abs(n.x), abs(n.y), abs(n.z)) < 0.985
        a = max(range(3), key=lambda i: abs(n[i]))
        for lp in f.loops:
            c = lp.vert.co
            lp[uvl].uv = (c.x + c.y, c.z) if a != 2 else (c.x, c.y)
    return K.to_obj("mattress", bm, [mat])


def _piping(K, mat):
    bm = bmesh.new()
    for z in (0.395, 0.335):
        pts = [Vector(p) for p in FR.rounded_rect_path(G.MAT_HX + 0.0008, G.MAT_HY + 0.0008, G.MAT_R, z)]
        bm_tube(bm, pts, 0.0065, sides=12, closed=True, smooth=True)
    return K.to_obj("piping", bm, [mat])


def _box_object(K, base, lo, hi):
    """A hidden collider cube spanning lo..hi (frame coordinates)."""
    bm = bmesh.new()
    g = bmesh.ops.create_cube(bm, size=1.0)
    bmesh.ops.scale(bm, vec=Vector([h - l for l, h in zip(lo, hi)]), verts=g["verts"])
    bmesh.ops.translate(bm, vec=Vector([(h + l) / 2 for l, h in zip(lo, hi)]), verts=g["verts"])
    return K.collider(K.to_obj(base, bm, []))


# ================================================================================================ the quilt's print
# colour indices of the default palette: 0 ink (the palette's base), 1 love, 2 gold, 3 pine, 4 iris (deepened), 5 foam
DEFAULT_PALETTE = ("base", "love", "gold", "pine", "iris", "foam")
PATTERNS = {
    "memphis": {"weights": [(PT.TRIANGLE, 1.3), (PT.DOTS, 1.0), (PT.RING, 0.9), (PT.SQUIGGLE, 1.2), (PT.ZIGZAG, 1.1),
                            (PT.STRIPES, 0.8), (PT.ARCH, 0.8), (PT.CROSS, 0.7)], "empty": 0.08},
    "zigzag": {"weights": [(PT.ZIGZAG, 3.0), (PT.SQUIGGLE, 2.0), (PT.STRIPES, 1.4), (PT.DOTS, 1.0), (PT.TRIANGLE, 0.8)],
               "empty": 0.05},
    "triangles": {"weights": [(PT.TRIANGLE, 4.0), (PT.DOTS, 1.2), (PT.RING, 0.8), (PT.STRIPES, 0.8)], "empty": 0.05},
    "plain": {"weights": [], "empty": 1.0},
}
DEFAULT_POOLS = {
    PT.TRIANGLE: [(2, 3), (1, 3), (3, 2), (4, 2), (0, 1)], PT.DOTS: [(1, 3), (2, 2), (3, 2), (0, 2), (4, 1)],
    PT.RING: [(3, 3), (1, 2), (4, 2), (0, 2)], PT.SQUIGGLE: [(0, 4), (1, 2), (3, 2), (2, 1)],
    PT.ZIGZAG: [(0, 3), (3, 3), (1, 2), (4, 1)], PT.STRIPES: [(0, 3), (1, 2), (3, 2), (2, 1)],
    PT.ARCH: [(2, 2), (1, 3), (4, 2), (3, 2)], PT.CROSS: [(1, 3), (0, 2), (3, 2), (2, 1)],
}


def quilt_palette(K, names):
    """The print's colours (linear RGBA) for palette slot names: the darkest is kept as ink, iris is deepened."""
    out = []
    for n in names:
        if n == "iris":
            out.append(K.blend(iris=0.6, overlay=0.4))
        elif n in ("gold", "love", "pine"):
            out.append(K.blend(**{n: 1.0, "chroma": 1.15}))
        else:
            out.append(slot_color(K, n))
    return out


def print_spec(K, info, pattern, names, seed):
    """Layout of the quilt's print over the flat sheet (info from G.quilt_shell) and the table stops for the shader."""
    nx, ny = 5, 8
    sx, sy = 2 * info["hw"] / nx, (info["t_hi"] - info["t_lo"]) / ny
    pat = PATTERNS[pattern]
    n = len(names)
    pools = DEFAULT_POOLS if tuple(names) == DEFAULT_PALETTE else {
        t: [(i, 2.0 if i == 0 else 1.0) for i in range(n)] for t in range(1, 9)}
    if pat["weights"]:
        cells = PT.layout(nx, ny, seed, pat["weights"], pools, empty=pat["empty"], n_colors=n)
    else:
        cells = [PT.Cell() for _ in range(nx * ny)]
    per = (PT.LUT_SIZE // nx) * nx
    tables = [PT.lut_stops(cells[k:k + per]) for k in range(0, len(cells), per)]
    return {"nx": nx, "ny": ny, "sx": sx, "sy": sy, "tables": tables, "cells": cells}


@register("bed_single")
def bed_single(name, coll, root, slots=None):
    K = Kit(name, coll, root, slots)
    seed = int(opt(K, "seed", 7))
    pattern = str(opt(K, "pattern", "memphis"))
    if pattern not in PATTERNS:
        raise ValueError(f"{name}: pattern {pattern!r} (have {sorted(PATTERNS)})")
    names = slot_names(K, "quilt_colors", ",".join(DEFAULT_PALETTE))
    ground = K.blend(text=0.5, iris=0.22, rose=0.1, overlay=0.18)
    if opt(K, "quilt_ground", None):
        ground = slot_color(K, opt(K, "quilt_ground", None))
    case = slot_color(K, opt(K, "pillowcase", None)) if opt(K, "pillowcase", None) else K.blend(rose=0.7, love=0.3)
    paint_c = K.blend(pine=0.6, foam=0.25, text=0.15)
    if opt(K, "frame_color", None):
        paint_c = slot_color(K, opt(K, "frame_color", None))
    # ---- the quilt first: its shape fixes the sheet coordinates the print lives in
    quilt, qinfo = G.quilt_shell(seed=seed)
    spec = print_spec(K, qinfo, pattern, names, seed)
    spec.update(ground=ground, colors=quilt_palette(K, names), ink=K.blend(base=1.0), gold=K.blend(gold=1.0),
                blobs=[K.slot("love"), K.blend(gold=1.0), K.slot("foam"), K.blend(iris=1.0)],
                specks=[K.slot("love"), K.blend(gold=1.0), K.slot("pine"), K.blend(iris=0.6, overlay=0.4),
                        K.blend(base=1.0)],
                binding=K.slot("love"), pitch=0.115, lining=K.blend(foam=0.45, text=0.5, pine=0.05))
    # ---- the frame
    frame_kind, headboard = str(opt(K, "frame", "wood")), str(opt(K, "headboard", "slats"))
    if frame_kind not in ("wood", "tube") or headboard not in ("slats", "padded"):
        raise ValueError(f"{name}: frame = wood | tube, headboard = slats | padded (got {frame_kind!r}, {headboard!r})")
    if frame_kind == "tube":
        _tube_object(K, M.tube(K, paint_c), headboard)
    else:
        _frame_object(K, M.paint(K, paint_c), headboard)
    if headboard == "padded":
        pad_c = slot_color(K, opt(K, "headboard_color", "iris"))
        _padded_object(K, M.padded(K, pad_c))
    bm = bmesh.new()
    finials = [(c, 1.0) for _, c in FR.finials()] if frame_kind == "wood" else [(c, k) for _, c, k in FR.tube_finials()]
    for c, k in finials:
        bm_lathe(bm, FR.finial_profile(k), segs=24, center=c)
    K.to_obj("finials", bm, [M.accent(K, K.blend(gold=1.0))])
    _mattress(K, M.ticking(K, K.blend(text=0.55, overlay=0.2, iris=0.25), (0.45, 0.45, 0.52, 1.0)))
    _piping(K, K.simple_mat("piping", K.blend(love=0.7, rose=0.3), rough=0.7))
    sheet, sinfo = G.sheet_shell(seed=seed + 4)
    sheet_m = M.sheet(K, K.blend(text=0.5, foam=0.2, rose=0.15, overlay=0.15))
    shell_obj(K, "sheet", sheet, [sheet_m] * 3)
    bind_m = M.binding(K, spec["binding"])
    shell_obj(K, "quilt", quilt, [M.quilt(K, spec), M.lining(K, spec), bind_m])
    pil_m = M.pillowcase(K, case)
    tops = []
    for spec_ in G.PILLOWS:
        local, Vb, R, loc = G.pillow_in_bed(spec_, seed + 5)
        rot = Euler(tuple(math.radians(a) for a in spec_[6]))
        shell_obj(K, spec_[0], local, [pil_m], loc=tuple(loc), rot=tuple(rot))
        tops.append(Vb[Vb[:, 2].argmax()])
    # ---- hidden colliders: the mattress with its quilt, the headboard
    ztop = qinfo["field_top_z"]
    y0, y1 = qinfo["y0"], qinfo["y_fold"]
    top_box = _box_object(K, "col_top", (-0.51, -1.03, G.MAT_Z0), (0.51, y1 + 0.03, ztop + 0.02))
    bed_box = _box_object(K, "col_bed", (-G.MAT_HX, y1 + 0.03, G.MAT_Z0), (G.MAT_HX, G.MAT_HY, G.SHEET_Z + 0.004))
    head_box = _box_object(K, "col_head", (-G.FRAME_HX, G.FRAME_HY - G.BOARD_T, 0.40),
                           (G.FRAME_HX, G.FRAME_HY, G.HEAD_TOP))
    yc = round((y0 + y1) / 2, 4)
    use = {"rest": [{"name": "top", "type": "plane", "center": [0.0, yc, round(ztop, 4)], "normal": [0, 0, 1],
                     "size": [round(2 * qinfo["a_edge"] + 0.01, 3), round(y1 - y0, 3)]}],
           "look": [{"name": "pillow", "point": [round(float(v), 4) for v in tops[-1]]},
                    {"name": "quilt", "point": [0.0, yc, round(ztop, 4)]}]}
    colliders = [{"type": "box", "object": top_box.name, "rnd": 0.02, "tag": name},
                 {"type": "box", "object": bed_box.name, "rnd": 0.01, "tag": name},
                 {"type": "box", "object": head_box.name, "rnd": 0.01, "tag": name}]
    return K.card(use=use, colliders=colliders, origin="floor_center", front="-Y",
                  options={"pattern": pattern, "seed": seed, "quilt_colors": names, "frame": frame_kind,
                           "headboard": headboard},
                  frame_bounds={"min": [-G.FRAME_HX, -G.FRAME_HY, 0.0], "max": [G.FRAME_HX, G.FRAME_HY, G.HEAD_TOP]})


# ================================================================================================ the rug
RUG_PALETTE = ("base", "love", "gold", "iris", "foam", "lavender")
RUG_POOLS = {
    PT.TRIANGLE: [(2, 3), (1, 3), (3, 2), (5, 2), (4, 1)], PT.DOTS: [(1, 3), (2, 2), (5, 2), (4, 1), (0, 1)],
    PT.RING: [(5, 3), (2, 2), (1, 2), (3, 1)], PT.SQUIGGLE: [(0, 3), (1, 3), (2, 2), (5, 2)],
    PT.ZIGZAG: [(0, 3), (2, 2), (5, 2), (1, 2)], PT.STRIPES: [(5, 3), (1, 2), (2, 2), (0, 2)],
    PT.ARCH: [(2, 2), (1, 3), (5, 2), (3, 2)], PT.CROSS: [(1, 3), (2, 2), (5, 2), (4, 2)],
}
RUG_BANDS = {"edge": 0.010, "band": (0.010, 0.040), "line": (0.0445, 0.0045), "saw": (0.049, 0.099), "saw_pitch": 0.075,
             "inner": (0.1045, 0.0045), "field": 0.116}


def rug_colors(K, names):
    out = []
    for n in names:
        if n == "lavender":
            out.append(K.blend(text=0.6, iris=0.4))
        elif n in ("gold", "love", "pine"):
            out.append(K.blend(**{n: 1.0, "chroma": 1.1}))
        else:
            out.append(slot_color(K, n))
    return out


@register("rug_80s")
def rug_80s(name, coll, root, slots=None):
    K = Kit(name, coll, root, slots)
    seed = int(opt(K, "seed", 5))
    names = slot_names(K, "rug_colors", ",".join(RUG_PALETTE))
    ground = K.blend(pine=0.85, overlay=0.15)
    if opt(K, "rug_ground", None):
        ground = slot_color(K, opt(K, "rug_ground", None))
    W, D = G.RUG_W, G.RUG_D
    fx = RUG_BANDS["field"]
    nx, ny = 7, 4
    sx, sy = (W - 2 * fx) / nx, (D - 2 * fx) / ny
    n = len(names)
    pools = RUG_POOLS if tuple(names) == RUG_PALETTE else {
        t: [(i, 2.0 if i == 0 else 1.0) for i in range(n)] for t in range(1, 9)}
    weights = [(PT.TRIANGLE, 1.2), (PT.DOTS, 1.0), (PT.RING, 0.9), (PT.SQUIGGLE, 1.2), (PT.ZIGZAG, 1.1),
               (PT.STRIPES, 0.8), (PT.ARCH, 0.8), (PT.CROSS, 0.8)]
    cells = PT.layout(nx, ny, seed, weights, pools, empty=0.06, n_colors=n)
    spec = {"nx": nx, "ny": ny, "sx": sx, "sy": sy, "ox": fx, "oy": fx, "tables": [PT.lut_stops(cells)],
            "colors": rug_colors(K, names), "ground": ground,
            "blobs": [K.blend(iris=1.0), K.slot("foam"), K.blend(text=1.0)],
            "specks": [K.blend(text=0.7, iris=0.3), K.blend(gold=1.0), K.slot("love"), K.slot("foam")],
            "width": W, "height": D, "bands": RUG_BANDS,
            "border": {"edge": K.blend(base=1.0), "band": K.blend(text=0.6, iris=0.4), "line": K.slot("love"),
                       "saw": K.blend(base=1.0), "saw2": K.blend(gold=1.0, chroma=1.1), "inner": K.slot("love")}}
    body = G.rug_shell(seed)
    edge_m = M.rug_edge(K, K.blend(base=0.7, overlay=0.3))
    shell_obj(K, "body", body, [M.rug_wool(K, spec), edge_m, edge_m])
    fringe, tassels = G.rug_fringe(seed)
    shell_obj(K, "fringe", fringe, [M.yarn(K, K.blend(text=0.75, gold=0.15, rose=0.1))], uv_names=("UVMap", "edge"))
    use = {"rest": [{"name": "top", "type": "plane", "center": [0.0, 0.0, G.RUG_T], "normal": [0, 0, 1],
                     "size": [W, D]}]}
    return K.card(use=use, colliders=[], origin="floor_center", front="-Y", flat=True, body=[W, D, G.RUG_T],
                  fringe={"length": round(float(np.abs(fringe.V[:, 0]).max()) - W / 2, 3), "sides": ["-X", "+X"],
                         "tassels_per_side": tassels},
                  options={"seed": seed, "rug_colors": names})
