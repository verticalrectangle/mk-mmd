"""Bedroom furniture of the 80s room: a writing desk and the desk chair that goes with it. Pure maths (the rake of the
backrest, outlines, bent-tube centre lines) lives in bedroom_furniture_geo.py; materials and meshes are built here with the
shared kit of the cafe props (cafe_kit.py). Both register themselves with @register when imported. Materials are
procedural (object coordinates, no image files); no custom properties, no lights.

Colours come from the project's palette slots (see cafe_kit.py); `[[prop]] slots = {pine = "#123456"}` overrides single
slots. The non-colour keys named below choose which slot plays which role (`slots = {edge = "love"}`); each takes slot
names, anything else is an error. Every part is an object `<name>_<part>` under the prop's root.

bedroom_desk  An 80s writing desk of 1.20 x 0.60 x 0.74 m. Origin = the floor under the centre of the top, front = -Y (the
              sitter's side), +Z up. The upper face of the top is at exactly z = 0.74. Options: `edge` (slot of the band
              round the top, default pine), `drawers` (slots of the three drawer fronts, top to bottom, a list or a comma
              separated string, default "foam,love,gold"). Parts:
                top        0.03 m slab (z 0.71 .. 0.74), rounded corners (r 0.04), pale speckled laminate with fine
                           scratches in patches and a thin clear coat; the edge band is the wall, a 6 mm round-over and an
                           11 mm border on the top; the underside is plain
                pedestal   the carcass of the three-drawer unit under the -X end, a moulded hollow tube (open at the foot
                           and under the slab): plan a rounded rectangle x in [-0.5816, -0.2904], y in [-0.2666, 0.285]
                           with 70 mm (front) / 100 mm (back) corner radii, z 0 .. 0.7099; the foot rolls inward (14 mm radius) over a recessed
                           dark kick plate (z < 0.077, set in 22 mm); pale laminate walls. Geometry:
                           bedroom_furniture_ped.py (shell toolkit, docs/modelling.md)
                drawers    three pastel plastic fronts that wrap round the corners of the carcass: 10 mm thick shells
                           (3.5 mm rounded rims), crowned 3 mm up their height on the flat of the front, running 35 mm
                           past the corner arcs along the sides; x in [-0.592, -0.28], front face at y = -0.28; heights
                           0.165 / 0.19 / 0.22 with 4 mm joints, the top one reaching z = 0.695, the lowest starting at
                           0.112
                joints     dark reveal strips on the carcass in the two 4 mm gaps between the fronts
                handles    three chrome bar pulls (0.064 m between the round ends, radius 5.5 mm) on two posts, 10.5 mm in
                           front of the fronts' middle row; the nearest point of the desk to the sitter is y = -0.2994, in
                           front of the top's front edge at -0.30
                frame      chrome-look tubular steel loop (tube radius 12.5 mm, bends 55 mm) in the plane x = 0.58 under
                           the +X end: two legs at y = +-0.255, a rail on the floor and a rail tight under the top
                panel      thin modesty panel at the back, x in [-0.37, 0.58] (its left end is buried in the rounded back
                           corner of the carcass), y in [0.264, 0.28], z in [0.30, 0.71]
              The kneehole is open: below the top (z < 0.66) nothing stands between x = -0.28 and x = 0.5675 except the
              back panel (y >= 0.264), so a chair can be pulled in more than 0.5 m. Card: size [1.2, 0.6, 0.74], origin
              "floor_center", front "-Y"; use.rest [top: plane, centre [0, 0, 0.74], normal [0, 0, 1], size [1.2, 0.6];
              front: edge from [-0.3, -0.30, 0.74] to [0.3, -0.30, 0.74], normal [0, 0, 1]], use.look [top: [0, 0, 0.74]];
              colliders (hidden boxes, rnd 0.008, tag = the prop name): `col_top` (the slab, 1.2 x 0.6 x 0.03) and
              `col_pedestal` (x in [-0.592, -0.28], y in [-0.28, 0.285], z in [0, 0.71]).

desk_chair    A chrome tubular-steel cantilever chair with a padded seat and a padded back, no arms, built to the numbers
              of `cafe_chair`: the same `use.sit`, `use.feet` and collider geometry, so a character that sits on one sits
              identically on the other. Origin = the floor under the seat's centre, front = -Y (the sitter faces -Y, the
              backrest is at +Y), seat top at z = 0.45. Options: `upholstery` (slot of the fabric, default iris),
              `accent` (slot of the stripe and the piping, default love). Parts:
                seat       0.42 x 0.42 x 0.052 cushion (corners r 0.08, 18 mm round-over, z 0.398 .. 0.45): the flat top
                           is at exactly z = 0.45 and nothing stands above it; accent piping (radius 3.5 mm) round the
                           top edge, a dark underside
                back       0.415 x 0.39 x 0.045 cushion (corners r 0.075) in the frame of the backrest collider (origin
                           [0, 0.2086, 0.66], tilted 10.4 deg back about X; x across, y through the cushion, z up its
                           face): the flat front face is the collider's front face (y = -0.015 there; the face a
                           character's back and hair rest against, nothing stands in front of it) and covers x in
                           +-0.1895, z in [-0.157, 0.197]; accent piping round the front edge, a horizontal accent stripe
                           (z = 0.06, 0.04 wide) with double top-stitching, a hard plastic shell on the rear face
                frame      one bent tube per side at x = +-0.2175 (radius 11.5 mm): a runner on the floor (y from 0.235
                           forward), the C-shaped front (apex y = -0.225), the seat rail under the cushion (z = 0.3865)
                           and the back post beside the backrest (through the middle of the cushion) with a rounded cap
                           above it; cross tubes under the seat (y = -0.02 and 0.09) and on the floor (y = 0.20)
                feet       dark plugs closing the runners; bolts   four chrome bolt heads on the posts
              Size about [0.46, 0.51, 0.88]. Card: use.sit [seat: hip [0, 0.035, 0.525], facing [0, -1, 0], seat_z 0.45,
              floor_z 0, pelvis_deg 6, back_deg 0, back_tilt_deg 10.4], use.feet [floor: L [0.115, -0.33, None], R
              [-0.105, -0.36, None], floor_z 0], use.rest [seat_front: edge from [-0.15, -0.14, 0.45] to [0.15, -0.14,
              0.45], normal [0, 0, 1]]; colliders (hidden boxes, rnd 0.012, tag = the prop name): `col_back` (0.38 x 0.030
              x 0.46, centre [0, 0.2086, 0.66], rotated -10.4 deg about X) and `col_seat` (0.42 x 0.42 x 0.04, centre
              [0, 0, 0.43]; a box, where cafe_chair describes the same mesh as a cylinder)."""
import math

import bmesh
from mathutils import Matrix, Vector

from .. import shell as SH
from . import bedroom_furniture_ped as PED
from . import register
from .bedroom_furniture_geo import BACK_C, BACK_TILT, SEAT_Z, pad_point, round_path, rrect
from .cafe_colors import DAWN, shade
from .cafe_kit import Kit, L, N, bm_append, bm_box, bm_lathe, bm_tube, principled, ramp

# ================================================================= desk dimensions
DESK_W, DESK_D, DESK_H = 1.20, 0.60, 0.74
TOP_T = 0.03
SLAB_Z0 = DESK_H - TOP_T
LEG_X = 0.58                             # plane of the tubular loop
LEG_Y = 0.255
LEG_R = 0.0125
PANEL_Y = (0.264, 0.28)
PANEL_Z0 = 0.30

# ================================================================= chair dimensions
SEAT_T = 0.052                           # thickness of the seat cushion (top 0.45)
SEAT_W = 0.42                            # = the seat collider
BACK_W, BACK_H, BACK_T = 0.415, 0.39, 0.045
BACK_ZC = 0.02                           # centre of the back cushion up the pad frame
BACK_FRONT = -0.015                      # front face of the cushion = front face of the collider (0.030 thick)
RAIL_X = 0.2175                          # side tubes
TUBE_R = 0.0115
RAIL_Z = SEAT_Z - SEAT_T - TUBE_R        # seat rail: its top touches the cushion
FRONT_APEX = -0.225                      # front of the C-shaped bend (tube centre)
POST_YP = 0.0075                         # post line: through the middle of the cushion (pad frame)
POST_TOP = 0.212                         # post end up the pad frame (the round cap adds the tube radius)
RUNNER_END = 0.235                       # where the floor runners stop (the plugs close them)
SEAM = 3                                 # ring of a cushion's round-over (45 deg) where the fabric meets the piping


# ================================================================= small helpers
def _opt(K, key, default):
    v = K.slots.get(key, default)
    if v not in DAWN:
        raise ValueError(f"{K.name}: option {key!r} must be a palette slot ({', '.join(sorted(DAWN))}), not {v!r}")
    return v


def _opt_list(K, key, default, n):
    v = K.slots.get(key, default)
    if isinstance(v, str):
        v = [s.strip() for s in v.split(",") if s.strip()]
    v = list(v) or list(default)
    out = [v[i % len(v)] for i in range(n)]
    for s in out:
        if s not in DAWN:
            raise ValueError(f"{K.name}: option {key!r} must name palette slots ({', '.join(sorted(DAWN))}), not {s!r}")
    return out


def _blend(K, *pairs, **kw):
    """K.blend of (slot, weight) pairs; a slot named twice adds up (a role may be played by any slot)."""
    w = {}
    for slot, wt in pairs:
        w[slot] = w.get(slot, 0.0) + wt
    return K.blend(**w, **kw)


def _pastel(K, slot, lift=0.18, chroma=1.0):
    """A palette slot lifted toward the text colour (a pastel plastic); `chroma` scales its saturation after the lift."""
    w = {"text": lift}
    w[slot] = w.get(slot, 0.0) + 1.0 - lift
    return K.blend(chroma=chroma, **w)


def _lofted(rings, mats, cap_a=None, cap_b=None, crisp=(False, False), smooth=True):
    """Closed body from rings (lists of 3D points of equal length): quad strips between consecutive rings (material
    mats[i] for strip i, smooth shaded) and n-gon caps on the first / last ring (material index or None; crisp = the
    rim of the cap is a sharp edge). Normals are made to point outward. -> bmesh."""
    bm = bmesh.new()
    vr = [[bm.verts.new(p) for p in ring] for ring in rings]
    n = len(vr[0])
    for i in range(len(vr) - 1):
        for j in range(n):
            f = bm.faces.new((vr[i][j], vr[i][(j + 1) % n], vr[i + 1][(j + 1) % n], vr[i + 1][j]))
            f.material_index = mats[i]
            f.smooth = smooth
    for ring, mat, hard in ((vr[0], cap_a, crisp[0]), (vr[-1], cap_b, crisp[1])):
        if mat is not None:
            f = bm.faces.new(ring)
            f.material_index = mat
            f.smooth = False
            if hard:
                for e in f.edges:
                    e.smooth = False
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces[:])
    return bm


def _rbox(size, r, mat=0, seg=2):
    """Box with rounded edges (radius r, seg segments), centred on the origin: flat faces flat, roundings smooth."""
    b = bmesh.new()
    g = bmesh.ops.create_cube(b, size=1.0)
    bmesh.ops.scale(b, vec=Vector(size), verts=g["verts"])
    bmesh.ops.bevel(b, geom=b.edges[:], offset=r, segments=seg, profile=0.5, affect="EDGES")
    b.normal_update()
    for f in b.faces:
        f.material_index = mat
        n = f.normal
        f.smooth = max(abs(n.x), abs(n.y), abs(n.z)) < 0.999
    return b


def _put(bm, src, loc=(0.0, 0.0, 0.0), rot=None):
    """Append (and free) `src` at `loc`; rot = a 3x3 / 4x4 matrix applied first."""
    m = Matrix.Translation(loc)
    if rot is not None:
        m = m @ rot.to_4x4()
    bm_append(bm, src, m)
    src.free()


def _tube(bm, pts, r, sides=16, mat=0, cap=("flat", "flat"), closed=False, up=(0, 0, 1)):
    return bm_tube(bm, [Vector(p) for p in pts], r, sides=sides, mat=mat, cap=cap, closed=closed, up=up, uv=False)


def _arc_ring(rs, n_arcs):
    """Profile of a round-over of radius rs: points (inset, depth) from the flat face to the wall (90 deg)."""
    return [(rs - rs * math.sin(math.radians(90.0 * k / n_arcs)), rs - rs * math.cos(math.radians(90.0 * k / n_arcs)))
            for k in range(n_arcs + 1)]


# ================================================================= materials
def _laminate_mat(K, light, dark, scratch):
    """Pale melamine laminate: a faint speckle in roughness and bump (object coordinates, metres), a slow drift of tone and
    two sets of fine long scratches, a thin clear coat for the sheen of a lamp on it."""
    m, nt, out = K.new_mat("laminate")
    tc = N(nt, "ShaderNodeTexCoord", (-1900, 0))
    sp = N(nt, "ShaderNodeTexNoise", (-1600, 450), inputs={"Scale": 420.0, "Detail": 5.0, "Roughness": 0.65})
    L(nt, tc.outputs["Object"], sp.inputs["Vector"])
    dr = N(nt, "ShaderNodeTexNoise", (-1600, 150), inputs={"Scale": 2.6, "Detail": 2.0, "Roughness": 0.5})
    L(nt, tc.outputs["Object"], dr.inputs["Vector"])
    tone = ramp(nt, (-1300, 150), [(0.35, light), (0.65, dark)])
    L(nt, dr.outputs["Fac"], tone.inputs["Fac"])
    lines = []
    for k, (rot, sx, sy, off) in enumerate(((0.35, 5.0, 340.0, 0.0), (-0.6, 4.0, 260.0, 17.0))):
        y = -250 - 330 * k
        rt = N(nt, "ShaderNodeMapping", (-1650, y), inputs={"Location": (off, off * 0.6, 0.0)})
        rt.inputs["Rotation"].default_value = (0.0, 0.0, rot)
        L(nt, tc.outputs["Object"], rt.inputs["Vector"])
        sc = N(nt, "ShaderNodeMapping", (-1400, y), inputs={"Scale": (sx, sy, 1.0)})
        L(nt, rt.outputs[0], sc.inputs["Vector"])
        nz = N(nt, "ShaderNodeTexNoise", (-1150, y), inputs={"Scale": 1.0, "Detail": 1.0, "Roughness": 0.5})
        L(nt, sc.outputs[0], nz.inputs["Vector"])
        d = N(nt, "ShaderNodeMath", (-900, y), operation="SUBTRACT", inputs={1: 0.58})
        L(nt, nz.outputs["Fac"], d.inputs[0])
        a = N(nt, "ShaderNodeMath", (-720, y), operation="ABSOLUTE")
        L(nt, d.outputs[0], a.inputs[0])
        ln = N(nt, "ShaderNodeMapRange", (-540, y), inputs={"From Min": 0.010, "From Max": 0.0035, "To Min": 0.0,
                                                           "To Max": 1.0}, clamp=True)
        L(nt, a.outputs[0], ln.inputs["Value"])
        lines.append(ln.outputs["Result"])
    both = N(nt, "ShaderNodeMath", (-340, -420), operation="MAXIMUM")
    L(nt, lines[0], both.inputs[0])
    L(nt, lines[1], both.inputs[1])
    pn = N(nt, "ShaderNodeTexNoise", (-900, -900), inputs={"Scale": 3.7, "Detail": 1.0, "Roughness": 0.5})
    L(nt, tc.outputs["Object"], pn.inputs["Vector"])
    pm = N(nt, "ShaderNodeMapRange", (-640, -900), inputs={"From Min": 0.42, "From Max": 0.62, "To Min": 0.0,
                                                          "To Max": 0.8}, clamp=True)
    L(nt, pn.outputs["Fac"], pm.inputs["Value"])
    scr = N(nt, "ShaderNodeMath", (-150, -520), operation="MULTIPLY")
    L(nt, both.outputs[0], scr.inputs[0])
    L(nt, pm.outputs["Result"], scr.inputs[1])
    wear = N(nt, "ShaderNodeMixRGB", (-300, 100), blend_type="MIX", inputs={"Color2": scratch})
    L(nt, scr.outputs[0], wear.inputs["Fac"])
    L(nt, tone.outputs["Color"], wear.inputs["Color1"])
    rough = N(nt, "ShaderNodeMapRange", (-1300, 450), inputs={"From Min": 0.35, "From Max": 0.65, "To Min": 0.30,
                                                             "To Max": 0.44}, clamp=True)
    L(nt, sp.outputs["Fac"], rough.inputs["Value"])
    rup = N(nt, "ShaderNodeMath", (-300, 420), operation="MULTIPLY_ADD", inputs={1: 0.22})
    L(nt, scr.outputs[0], rup.inputs[0])
    L(nt, rough.outputs["Result"], rup.inputs[2])
    h = N(nt, "ShaderNodeMath", (-300, -150), operation="MULTIPLY_ADD", inputs={1: -0.6})
    L(nt, scr.outputs[0], h.inputs[0])
    L(nt, sp.outputs["Fac"], h.inputs[2])
    bump = N(nt, "ShaderNodeBump", (0, -150), inputs={"Strength": 0.012, "Distance": 0.0004})
    L(nt, h.outputs[0], bump.inputs["Height"])
    b = principled(nt, out, loc=(350, 0), **{"Specular IOR Level": 0.5, "Coat Weight": 0.12, "Coat Roughness": 0.2})
    L(nt, wear.outputs["Color"], b.inputs["Base Color"])
    L(nt, rup.outputs[0], b.inputs["Roughness"])
    L(nt, bump.outputs["Normal"], b.inputs["Normal"])
    return m


def _paint_mat(K, base, color, rough=0.38, coat=0.18):
    """Satin enamel / lacquered plastic with a faint orange-peel bump."""
    m, nt, out = K.new_mat(base)
    tc = N(nt, "ShaderNodeTexCoord", (-900, 0))
    nz = N(nt, "ShaderNodeTexNoise", (-650, 0), inputs={"Scale": 260.0, "Detail": 3.0, "Roughness": 0.6})
    L(nt, tc.outputs["Object"], nz.inputs["Vector"])
    bump = N(nt, "ShaderNodeBump", (-350, -150), inputs={"Strength": 0.03, "Distance": 0.0004})
    L(nt, nz.outputs["Fac"], bump.inputs["Height"])
    b = principled(nt, out, loc=(0, 0), **{"Base Color": color, "Roughness": rough, "Coat Weight": coat,
                                          "Coat Roughness": 0.2, "Specular IOR Level": 0.5})
    L(nt, bump.outputs["Normal"], b.inputs["Normal"])
    return m


def _chrome_mat(K, rough=0.16, metal=0.9):
    """Chrome-look steel: metallic, low roughness, tinted from the text / subtle slots. A tenth of it is not metal, so a
    tube still has a body (and does not turn black) where the room offers it nothing to reflect."""
    return K.simple_mat("chrome", K.blend(text=.72, subtle=.28), rough=rough, metal=metal, spec=0.5)


def _fabric_mat(K, base, color, sheen, stripe=None, thread=None):
    """Upholstery: a woven fuzz (bump), blotchy tone, a soft sheen at grazing angles. `stripe` = (z0, width, colour): a
    horizontal band across the cushion in its own frame (object coordinates), edged by double top-stitching (dashes on the
    front face, `thread` colour)."""
    m, nt, out = K.new_mat(base)
    tc = N(nt, "ShaderNodeTexCoord", (-2300, 0))
    wv = N(nt, "ShaderNodeTexNoise", (-2000, 400), inputs={"Scale": 560.0, "Detail": 3.0, "Roughness": 0.7})
    L(nt, tc.outputs["Object"], wv.inputs["Vector"])
    bl = N(nt, "ShaderNodeTexNoise", (-2000, 100), inputs={"Scale": 9.0, "Detail": 3.0, "Roughness": 0.5})
    L(nt, tc.outputs["Object"], bl.inputs["Vector"])
    tone = ramp(nt, (-1700, 100), [(0.3, shade(color, 0.93)), (0.7, shade(color, 1.06))])
    L(nt, bl.outputs["Fac"], tone.inputs["Fac"])
    col = tone.outputs["Color"]
    height = wv.outputs["Fac"]
    if stripe is not None:
        z0, w, scol = stripe
        sep = N(nt, "ShaderNodeSeparateXYZ", (-1900, -300))
        L(nt, tc.outputs["Object"], sep.inputs["Vector"])
        sub = N(nt, "ShaderNodeMath", (-1700, -300), operation="SUBTRACT", inputs={1: z0})
        L(nt, sep.outputs["Z"], sub.inputs[0])
        dz = N(nt, "ShaderNodeMath", (-1520, -300), operation="ABSOLUTE")
        L(nt, sub.outputs[0], dz.inputs[0])
        inside = N(nt, "ShaderNodeMapRange", (-1340, -300), inputs={"From Min": w / 2 + 0.0006, "From Max": w / 2 - 0.0006,
                                                                    "To Min": 0.0, "To Max": 1.0}, clamp=True)
        L(nt, dz.outputs[0], inside.inputs["Value"])
        mixc = N(nt, "ShaderNodeMixRGB", (-1100, -100), blend_type="MIX", inputs={"Color2": scol})
        L(nt, inside.outputs["Result"], mixc.inputs["Fac"])
        L(nt, col, mixc.inputs["Color1"])
        # two rows of stitches along each seam, 4.5 mm inside and outside the stripe's edge, dashed along x and only
        # where the surface faces the sitter (object-space normal -Y)
        rows = []
        for k, off in enumerate((-0.0045, 0.0045)):
            d = N(nt, "ShaderNodeMath", (-1340, -560 - 160 * k), operation="SUBTRACT", inputs={1: w / 2 + off})
            L(nt, dz.outputs[0], d.inputs[0])
            a = N(nt, "ShaderNodeMath", (-1160, -560 - 160 * k), operation="ABSOLUTE")
            L(nt, d.outputs[0], a.inputs[0])
            ln = N(nt, "ShaderNodeMapRange", (-980, -560 - 160 * k), inputs={"From Min": 0.0009, "From Max": 0.0005,
                                                                            "To Min": 0.0, "To Max": 1.0}, clamp=True)
            L(nt, a.outputs[0], ln.inputs["Value"])
            rows.append(ln.outputs["Result"])
        both = N(nt, "ShaderNodeMath", (-800, -620), operation="MAXIMUM")
        L(nt, rows[0], both.inputs[0])
        L(nt, rows[1], both.inputs[1])
        fx = N(nt, "ShaderNodeMath", (-1340, -900), operation="MULTIPLY", inputs={1: 135.0})
        L(nt, sep.outputs["X"], fx.inputs[0])
        fr = N(nt, "ShaderNodeMath", (-1160, -900), operation="FRACT")
        L(nt, fx.outputs[0], fr.inputs[0])
        dash = N(nt, "ShaderNodeMath", (-980, -900), operation="LESS_THAN", inputs={1: 0.62})
        L(nt, fr.outputs[0], dash.inputs[0])
        sepn = N(nt, "ShaderNodeSeparateXYZ", (-1900, -1100))
        L(nt, tc.outputs["Normal"], sepn.inputs["Vector"])
        front = N(nt, "ShaderNodeMath", (-1700, -1100), operation="LESS_THAN", inputs={1: -0.75})
        L(nt, sepn.outputs["Y"], front.inputs[0])
        s1 = N(nt, "ShaderNodeMath", (-620, -700), operation="MULTIPLY")
        L(nt, both.outputs[0], s1.inputs[0])
        L(nt, dash.outputs[0], s1.inputs[1])
        s2 = N(nt, "ShaderNodeMath", (-440, -760), operation="MULTIPLY")
        L(nt, s1.outputs[0], s2.inputs[0])
        L(nt, front.outputs[0], s2.inputs[1])
        stitch = s2.outputs[0]
        mixt = N(nt, "ShaderNodeMixRGB", (-700, -100), blend_type="MIX", inputs={"Color2": thread or scol})
        L(nt, stitch, mixt.inputs["Fac"])
        L(nt, mixc.outputs["Color"], mixt.inputs["Color1"])
        col = mixt.outputs["Color"]
        hh = N(nt, "ShaderNodeMath", (-440, 300), operation="MULTIPLY_ADD", inputs={1: -0.9})
        L(nt, stitch, hh.inputs[0])
        L(nt, wv.outputs["Fac"], hh.inputs[2])
        height = hh.outputs[0]
    bump = N(nt, "ShaderNodeBump", (-200, 300), inputs={"Strength": 0.35, "Distance": 0.0006})
    L(nt, height, bump.inputs["Height"])
    b = principled(nt, out, loc=(400, 0), **{"Roughness": 0.62, "Sheen Weight": 0.8, "Sheen Roughness": 0.35,
                                            "Specular IOR Level": 0.3})
    b.inputs["Sheen Tint"].default_value = sheen
    L(nt, col, b.inputs["Base Color"])
    L(nt, bump.outputs["Normal"], b.inputs["Normal"])
    return m


# ================================================================= desk
def _desk_top(K, mats):
    """The slab: rounded rectangle, rings from the underside up. Material 0 laminate (the top inside the band), 1 edge
    band (wall, 6 mm round-over and a rim 11 mm wide on the top: a coloured border), 2 underside."""
    rt, band = 0.006, 0.011
    prof = [(0.0025, SLAB_Z0), (0.0, SLAB_Z0 + 0.0025)]
    prof += [(rt - rt * math.sin(math.radians(a)), DESK_H - rt + rt * math.cos(math.radians(a)))
             for a in (90.0, 67.5, 45.0, 22.5, 0.0)]
    prof.append((band, DESK_H))
    rings = [[(x, y, z) for x, y in rrect(DESK_W, DESK_D, 0.04, n=9, inset=i)] for i, z in prof]
    bm = _lofted(rings, [1] * (len(rings) - 1), cap_a=2, cap_b=0, crisp=(True, True))
    return K.to_obj("top", bm, mats)


def _shell_obj(K, base, mesh, mats):
    """A `core.shell.Mesh` as the object `<name>_<base>`; the roles of its faces index `mats`."""
    return SH.mesh_object(K.oname(base), mesh, K.coll, K.root, lambda role: mats[role])


def _desk_pedestal(K, mats):
    """The carcass: a hollow tube with big plan corners, laminate walls (role 0), the rolled foot and kick plate (role 1)."""
    return _shell_obj(K, "pedestal", PED.carcass(0, 1), mats)


def _desk_drawers(K, mats):
    """Three drawer fronts wrapped round the corners, one material each."""
    return _shell_obj(K, "drawers", PED.fronts((0, 1, 2)), mats)


def _desk_joints(K, mat):
    """Dark reveal strips in the gaps between the fronts."""
    return _shell_obj(K, "joints", PED.joints(0), [mat])


def _desk_handles(K, mat):
    """A chrome bar pull on the flat of each front, standing 5 mm off it on two posts."""
    bm = bmesh.new()
    xc, yf = PED.pull_centre()
    for za, zb in PED.ROWS:
        zc = (za + zb) / 2
        _tube(bm, [(xc - 0.032, yf - 0.0105, zc), (xc + 0.032, yf - 0.0105, zc)], 0.0055, sides=14, cap=("round", "round"))
        for s in (-1, 1):
            _tube(bm, [(xc + s * 0.022, yf + 0.001, zc), (xc + s * 0.022, yf - 0.0105, zc)], 0.004, sides=10,
                  cap=("none", "none"))
    return K.to_obj("handles", bm, [mat])


def _desk_frame(K, mat):
    """One closed loop of tube in the plane x = LEG_X: floor rail, two legs, a rail tight under the top."""
    bm = bmesh.new()
    z_lo, z_hi = LEG_R, SLAB_Z0 - LEG_R - 0.0004
    corners = [(LEG_X, -LEG_Y, z_lo), (LEG_X, -LEG_Y, z_hi), (LEG_X, LEG_Y, z_hi), (LEG_X, LEG_Y, z_lo)]
    path = round_path(corners, 0.055, arc_deg=7.5, step=0.05, closed=True)
    _tube(bm, path, LEG_R, sides=18, closed=True, up=(1, 0, 0))
    return K.to_obj("frame", bm, [mat])


def _desk_panel(K, mat):
    """The modesty panel at the back; its left end is buried in the rounded back corner of the carcass."""
    bm = bmesh.new()
    x0, x1 = -0.37, LEG_X
    z1 = SLAB_Z0 - 0.0001
    p = _rbox((x1 - x0, PANEL_Y[1] - PANEL_Y[0], z1 - PANEL_Z0), 0.0025, mat=0)
    _put(bm, p, ((x0 + x1) / 2, (PANEL_Y[0] + PANEL_Y[1]) / 2, (PANEL_Z0 + z1) / 2))
    return K.to_obj("panel", bm, [mat])


def _box_collider(K, base, size, centre, rot=(0.0, 0.0, 0.0)):
    """Hidden box (not rendered, drawn as wire) for the solvers: a spec {"type": "box", "object": ...} takes its bounds."""
    bm = bmesh.new()
    bm_box(bm, size, (0, 0, 0))
    return K.collider(K.to_obj(base, bm, [], loc=centre, rot=rot))


@register("bedroom_desk")
def bedroom_desk(name, coll, root, slots=None):
    K = Kit(name, coll, root, slots)
    edge_slot = _opt(K, "edge", "pine")
    drawer_slots = _opt_list(K, "drawers", ("foam", "love", "gold"), 3)
    laminate = _laminate_mat(K, K.blend(text=.62, subtle=.28, gold=.10), K.blend(text=.50, subtle=.40, gold=.10),
                             K.blend(text=.9, gold=.1))
    edge = _paint_mat(K, "edge", _pastel(K, edge_slot, 0.10), rough=0.4, coat=0.12)
    under = K.simple_mat("underside", K.blend(subtle=.3, overlay=.7), rough=0.65, spec=0.3)
    plinth = K.simple_mat("plinth", K.blend(overlay=.7, muted=.3), rough=0.6, spec=0.3)
    joint = K.simple_mat("joint", K.blend(base=.85, overlay=.15), rough=0.85, spec=0.1)
    drawers = [_paint_mat(K, f"drawer{i + 1}", _pastel(K, s), rough=0.36, coat=0.2) for i, s in enumerate(drawer_slots)]
    chrome = _chrome_mat(K)
    panel = _paint_mat(K, "panel", K.blend(text=.55, subtle=.30, gold=.08, overlay=.07), rough=0.5, coat=0.05)
    _desk_top(K, [laminate, edge, under])
    _desk_pedestal(K, [laminate, plinth])
    _desk_drawers(K, drawers)
    _desk_joints(K, joint)
    _desk_handles(K, chrome)
    _desk_frame(K, chrome)
    _desk_panel(K, panel)
    px0, px1 = PED.PED_X
    py0, py1 = PED.PED_Y
    col_top = _box_collider(K, "col_top", (DESK_W, DESK_D, TOP_T), (0.0, 0.0, DESK_H - TOP_T / 2))
    col_ped = _box_collider(K, "col_pedestal", (px1 - px0, py1 - py0, SLAB_Z0),
                            ((px0 + px1) / 2, (py0 + py1) / 2, SLAB_Z0 / 2))
    use = {
        "rest": [{"name": "top", "type": "plane", "center": [0.0, 0.0, DESK_H], "normal": [0, 0, 1],
                  "size": [DESK_W, DESK_D]},
                 {"name": "front", "type": "edge", "a": [-0.3, -DESK_D / 2, DESK_H], "b": [0.3, -DESK_D / 2, DESK_H],
                  "normal": [0, 0, 1]}],
        "look": [{"name": "top", "point": [0.0, 0.0, DESK_H]}],
    }
    colliders = [{"type": "box", "object": o.name, "rnd": 0.008, "tag": name} for o in (col_top, col_ped)]
    return K.card(use=use, colliders=colliders, origin="floor_center", front="-Y")


# ================================================================= chair
def _cushion_rings(w, h, rc, cmap, profile, n=10):
    """Rings of a cushion: one per profile entry (inset, depth), outline points mapped by cmap(u, v, depth)."""
    return [[cmap(u, v, t) for u, v in rrect(w, h, rc, n=n, inset=i)] for i, t in profile]


def _cushion_profile(rs, T, back_chamfer, steps=6):
    """(profile, strip materials) of a cushion of thickness T seen from its flat front: a round-over of radius rs in
    `steps` steps, the wall, a chamfer at the rear. Strips before the seam ring SEAM (45 deg of the round-over) are the
    front fabric (0), the rest the boxing strip (1); the caps are the front (0) and the rear (2)."""
    prof = _arc_ring(rs, steps) + [(0.0, T - back_chamfer), (back_chamfer, T)]
    return prof, [0] * SEAM + [1] * (len(prof) - 1 - SEAM)


def _chair_seat(K, fabric, underside, piping):
    prof, mats = _cushion_profile(0.018, SEAT_T, 0.007)
    rings = _cushion_rings(SEAT_W, SEAT_W, 0.08, lambda u, v, t: (u, v, SEAT_Z - t), prof)
    bm = _lofted(rings, mats, cap_a=0, cap_b=2, crisp=(False, True))
    _tube(bm, rings[SEAM], 0.0035, sides=8, mat=3, cap=("none", "none"), closed=True)
    return K.to_obj("seat", bm, [fabric, fabric, underside, piping])


def _chair_back(K, fabric, shell, piping):
    prof, mats = _cushion_profile(0.018, BACK_T, 0.010)
    rings = _cushion_rings(BACK_W, BACK_H, 0.075, lambda u, v, t: (u, BACK_FRONT + t, BACK_ZC + v), prof)
    bm = _lofted(rings, mats, cap_a=0, cap_b=2, crisp=(False, True))
    _tube(bm, rings[SEAM], 0.0035, sides=8, mat=3, cap=("none", "none"), closed=True, up=(0, 1, 0))
    return K.to_obj("back", bm, [fabric, fabric, shell, piping], loc=BACK_C, rot=(-BACK_TILT, 0, 0))


def _plug_bm():
    """Rubber plug closing the end of a runner (axis +Z, base at z = 0)."""
    b = bmesh.new()
    rp = TUBE_R + 0.0005                                     # a hair wider than the tube (its axis sits 0.5 mm higher)
    bm_lathe(b, [(0.0, 0.0), (rp - 0.0012, 0.0), (rp, 0.0012), (rp, 0.0115), (rp - 0.0012, 0.0135), (0.0, 0.0138)],
             segs=24, uv=False)
    return b


def _dome_bm():
    """Bolt head (axis +Z, base at z = 0)."""
    b = bmesh.new()
    bm_lathe(b, [(0.0, 0.0), (0.0075, 0.0), (0.0078, 0.0006), (0.0068, 0.0026), (0.0045, 0.0039), (0.0, 0.0043)],
             segs=20, uv=False)
    return b


def _chair_frame(K, chrome):
    """One bent tube per side (runner -> C-shaped front -> seat rail -> back post) and the cross tubes."""
    bm = bmesh.new()
    r_front = (RAIL_Z - TUBE_R) / 2
    c, s = math.cos(BACK_TILT), math.sin(BACK_TILT)
    y_corner = pad_point(0.0, POST_YP, (RAIL_Z - BACK_C[2] + POST_YP * s) / c)[1]
    y_top, z_top = pad_point(0.0, POST_YP, POST_TOP)[1:]
    for sx in (-1, 1):
        x = sx * RAIL_X
        pts = [(x, RUNNER_END, TUBE_R), (x, FRONT_APEX, TUBE_R), (x, FRONT_APEX, RAIL_Z), (x, y_corner, RAIL_Z),
               (x, y_top, z_top)]
        _tube(bm, round_path(pts, [0.0, r_front, r_front, 0.055, 0.0], arc_deg=7.5, step=0.05), TUBE_R, sides=18,
              cap=("flat", "round"))
    # cross tubes end on the axis of the side tubes, so they must meet the straight part of the rail (y from -0.0375 to
    # y_corner - 0.05) and the floor runner
    for y, z in ((-0.02, RAIL_Z), (0.09, RAIL_Z), (0.20, TUBE_R)):
        _tube(bm, [(-RAIL_X, y, z), (0.0, y, z), (RAIL_X, y, z)], TUBE_R * 0.85, sides=16, cap=("none", "none"))
    return K.to_obj("frame", bm, [chrome])


def _chair_bolts(K, chrome, zps=(0.12, -0.10)):
    """Chrome bolt heads on the outside of both posts, where the cushion is clamped."""
    bm = bmesh.new()
    for sx in (-1, 1):
        for zp in zps:
            _put(bm, _dome_bm(), pad_point(sx * (RAIL_X + TUBE_R - 0.0025), POST_YP, zp),
                 Matrix.Rotation(sx * math.pi / 2, 3, "Y"))
    return K.to_obj("bolts", bm, [chrome])


def _chair_feet(K, rubber):
    bm = bmesh.new()
    for sx in (-1, 1):
        _put(bm, _plug_bm(), (sx * RAIL_X, RUNNER_END, TUBE_R + 0.0005), Matrix.Rotation(-math.pi / 2, 3, "X"))
    return K.to_obj("feet", bm, [rubber])


@register("desk_chair")
def desk_chair(name, coll, root, slots=None):
    K = Kit(name, coll, root, slots)
    up_slot = _opt(K, "upholstery", "iris")
    acc_slot = _opt(K, "accent", "love")
    fab = _pastel(K, up_slot, 0.08, chroma=1.25)
    acc = _pastel(K, acc_slot, 0.12, chroma=1.1)
    sheen = _pastel(K, up_slot, 0.5)
    thread = _pastel(K, acc_slot, 0.7)
    z_stripe = 0.06
    fab_seat = _fabric_mat(K, "fabric_seat", fab, sheen)
    fab_back = _fabric_mat(K, "fabric_back", fab, sheen, stripe=(z_stripe, 0.04, acc), thread=thread)
    shell = K.simple_mat("shell", _blend(K, ("muted", .55), (up_slot, .45)), rough=0.42, spec=0.4)
    under = K.simple_mat("underside", K.blend(base=.7, overlay=.3), rough=0.8, spec=0.2)
    piping = K.simple_mat("piping", acc, rough=0.5, spec=0.4, sheen=0.5)
    rubber = K.simple_mat("rubber", K.blend(base=.75, overlay=.25), rough=0.75, spec=0.3)
    chrome = _chrome_mat(K)
    _chair_seat(K, fab_seat, under, piping)
    _chair_back(K, fab_back, shell, piping)
    _chair_frame(K, chrome)
    _chair_bolts(K, chrome)
    _chair_feet(K, rubber)
    col_back = _box_collider(K, "col_back", (0.38, 0.030, 0.46), BACK_C, rot=(-BACK_TILT, 0.0, 0.0))
    col_seat = _box_collider(K, "col_seat", (SEAT_W, SEAT_W, 0.04), (0.0, 0.0, SEAT_Z - 0.02))
    use = {
        "sit": [{"name": "seat", "hip": [0.0, 0.035, SEAT_Z + 0.075], "facing": [0, -1, 0], "seat_z": SEAT_Z,
                 "floor_z": 0.0, "pelvis_deg": 6.0, "back_deg": 0.0, "back_tilt_deg": round(math.degrees(BACK_TILT), 1)}],
        "feet": [{"name": "floor", "L": [0.115, -0.33, None], "R": [-0.105, -0.36, None], "floor_z": 0.0}],
        "rest": [{"name": "seat_front", "type": "edge", "a": [-0.15, -0.14, SEAT_Z], "b": [0.15, -0.14, SEAT_Z],
                  "normal": [0, 0, 1]}],
    }
    colliders = [{"type": "box", "object": o.name, "rnd": 0.012, "tag": name} for o in (col_back, col_seat)]
    return K.card(use=use, colliders=colliders, origin="floor_center", front="-Y")
