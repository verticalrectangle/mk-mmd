"""The compact cassette of the bedroom tech props (helper module, registers no builder): `build_cassette` makes the
mesh in the frame of the `cassette_tape` prop, `cassette_materials` its materials. The `cassette_player` shows the same
mesh standing in its bay.

Frame of the mesh: origin = centre of the bottom face, +X along the long side (0.100 m), +Y toward the far (closed) edge,
+Z up (0.0125 m); the label is on top (+Z) and the tape openings are on the -Y edge (the near side of a cassette lying
on a desk, the bottom of one standing in a deck).

Two meshes: the translucent `shell` (materials 0 smoky plastic, 1 clear window; it casts no shadow, so the inside is lit
through it) and the opaque `body` (materials 0 label: printed paper, colour per corner from the layer `vc`, 1 hub: white,
2 tape: brown, 3 dark: the floor, rollers and openings, 4 steel: spindle pins)."""
import math
from types import SimpleNamespace

import bmesh

from . import bedroom_tech_geo as G
from . import bedroom_tech_maths as M
from .cafe_colors import mix
from .cafe_kit import L, N, bm_lathe, principled

W, D, T = 0.100, 0.0635, 0.0125            # the cassette (x, y, z); the label's ink is the top
ZS = T - 0.00025                            # top face of the shell (decals stand on it, up to T)
CORNER = 0.0045                             # plan corner radius of the shell
LIP_TOP, LIP_BOT, DRAFT = 0.0035, 0.0025, 0.0016    # edge fillets of the shell (top, bottom) and the lean of its side walls
LABEL = (-0.046, 0.046, 0.0030, 0.0292)     # label sticker rectangle on the top face: x0, x1, y0, y1
WINDOW = (-0.0265, 0.0265, -0.0185, 0.0005)         # clear window below the label
HUB_X, HUB_Y = 0.0210, -0.0090              # hub centres (+-HUB_X, HUB_Y)
HUB_R, HUB_HOLE = 0.0056, 0.0031            # hub outer radius and the radius of its toothed bore
PACK_L, PACK_R = 0.0132, 0.0104             # radius of the tape wound on the left and right hub
ROLLER_X, ROLLER_Y, ROLLER_R = 0.0300, -0.0268, 0.0033
TAPE_Y = -0.0299                            # the exposed run of tape along the front edge

ACCENT_PAIR = {"love": "pine", "gold": "love", "foam": "iris", "iris": "gold", "rose": "foam", "pine": "gold"}


def accent(K, spec):
    """(rgba, slot name or None) of a label colour spec: a palette slot name, or a '#rrggbb' value."""
    if isinstance(spec, str) and spec.startswith("#"):
        from ....core import palette as PAL
        return (*PAL.linear(spec), 1.0), None
    return K.slot(spec), spec


def label_spec(K, name):
    """The label colour spec of a tape: slots['label'] when given, else picked from M.LABEL_SLOTS by the instance
    name (see M.label_index)."""
    given = K.slots.get("label")
    if given:
        return given
    return M.LABEL_SLOTS[M.label_index(name)]


def colours(K, spec):
    """The colours of one cassette: shell smoke, label ground / dark stripe / light stripe / second accent / paper /
    ink / rules, hub white, tape brown, floor. `spec`: the label colour (see label_spec)."""
    a, slot = accent(K, spec)
    other = K.slot(ACCENT_PAIR.get(slot, "gold"))
    paper = K.blend(text=.9, rose=.1)
    dark = mix(a, K.slot("base"), 0.72)
    return SimpleNamespace(
        smoke=K.blend(overlay=.55, hl_med=.45), glass=K.blend(hl_high=.6, text=.4),
        ground=mix(a, K.slot("text"), 0.20), dark=dark, light=K.blend(text=.9, gold=.1), second=other,
        paper=paper, ink=K.blend(pine=.30, base=.70), rule=mix(paper, K.slot("subtle"), 0.55),
        hub=K.blend(text=.9, gold=.05, rose=.05), tape=K.blend(k=.55, base=.80, gold=.12, rose=.08),
        floor=K.blend(surface=.7, base=.3), steel=K.blend(subtle=.5, text=.5), accent=a)


# ================================================================= materials
def cassette_materials(K, C, prefix=""):
    """The materials of the two meshes (see the module docstring), named `<prop>_<prefix><name>`:
    -> ([shell, window], [label, hub, tape, dark, steel])."""
    # shell: smoky translucent plastic
    m, nt, out = K.new_mat(prefix + "shell")
    m.surface_render_method = "BLENDED"
    m.use_backface_culling = False
    m.show_transparent_back = True
    principled(nt, out, **{"Base Color": C.smoke, "Roughness": 0.10, "Alpha": 0.62, "Specular IOR Level": 0.65,
                           "Coat Weight": 0.6, "Coat Roughness": 0.04})
    shell = m
    # window: clear
    m, nt, out = K.new_mat(prefix + "window")
    m.surface_render_method = "BLENDED"
    m.use_backface_culling = True
    m.show_transparent_back = False
    principled(nt, out, **{"Base Color": C.glass, "Roughness": 0.04, "Alpha": 0.10, "Specular IOR Level": 0.7})
    window = m
    # label: printed paper, colour per corner
    m, nt, out = K.new_mat(prefix + "label")
    at = N(nt, "ShaderNodeAttribute", (-700, 0), attribute_type="GEOMETRY", attribute_name=G.VC)
    nz = N(nt, "ShaderNodeTexNoise", (-900, -250), inputs={"Scale": 900.0, "Detail": 3.0})
    tc = N(nt, "ShaderNodeTexCoord", (-1100, -250))
    L(nt, tc.outputs["Object"], nz.inputs["Vector"])
    bump = N(nt, "ShaderNodeBump", (-500, -250), inputs={"Strength": 0.05, "Distance": 0.0002})
    L(nt, nz.outputs["Fac"], bump.inputs["Height"])
    b = principled(nt, out, **{"Roughness": 0.38, "Specular IOR Level": 0.5, "Coat Weight": 0.25, "Coat Roughness": 0.15})
    L(nt, at.outputs["Color"], b.inputs["Base Color"])
    L(nt, bump.outputs["Normal"], b.inputs["Normal"])
    label = m
    hub = K.simple_mat(prefix + "hub", C.hub, rough=0.34, spec=0.5, coat=0.15)
    tape = K.simple_mat(prefix + "tape", C.tape, rough=0.45, spec=0.35)
    dark = K.simple_mat(prefix + "dark", C.floor, rough=0.55, spec=0.35)
    steel = K.simple_mat(prefix + "steel", C.steel, rough=0.28, metal=0.85, spec=0.6)
    return [shell, window], [label, hub, tape, dark, steel]


# ================================================================= mesh
def _paint_label(bm, fr, C, rng):
    """The printed label on the top face: pastel ground, stripes, a paper writing field with ruled lines and a
    handwriting-like scribble (no text), at depths just above the shell."""
    x0, x1, y0, y1 = LABEL
    d0 = ZS + 0.00005
    step = 0.00005
    cx, cy, w, h = (x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0
    G.flat_quad(bm, fr, cx, cy, w, h, d0, 0, C.ground, r=0.0008)
    # stripes along the lower edge of the label: dark, light, second accent
    sy = y0
    for hh, col in ((0.0014, C.dark), (0.0006, C.light), (0.0018, C.second)):
        G.flat_quad(bm, fr, cx, sy + hh / 2, w, hh, d0 + step, 0, col)
        sy += hh
    # a small second-accent tab at the top right corner of the label, a dark one at the left
    G.flat_quad(bm, fr, x1 - 0.0075, y1 - 0.0033, 0.0100, 0.0042, d0 + step, 0, C.second, r=0.0010)
    G.flat_quad(bm, fr, x0 + 0.0050, y1 - 0.0033, 0.0060, 0.0042, d0 + step, 0, C.dark, r=0.0010)
    # writing field
    fx0, fx1, fy0, fy1 = x0 + 0.0045, x1 - 0.0045, y0 + 0.0075, y1 - 0.0075
    G.flat_quad(bm, fr, (fx0 + fx1) / 2, (fy0 + fy1) / 2, fx1 - fx0, fy1 - fy0, d0 + step, 0, C.paper, r=0.0005)
    n_rules = 3
    for i in range(n_rules):
        yy = fy0 + (fy1 - fy0) * (i + 0.8) / (n_rules + 0.1)
        G.flat_quad(bm, fr, (fx0 + fx1) / 2, yy, fx1 - fx0 - 0.002, 0.00022, d0 + 2 * step, 0, C.rule)
    # scribble on the field: two lines of cursive-like strokes in ink
    sx0, sy0 = fx0 + 0.0020, fy0 + 0.0008
    for pts in M.scribble(rng, fx1 - fx0 - 0.0040, fy1 - fy0 - 0.0016, lines=2, size=0.0028):
        G.ribbon(bm, fr, [(sx0 + p[0], sy0 + p[1]) for p in pts], 0.00034, d0 + 3 * step, 0, C.ink)


def _hub(bm, x, y):
    """One reel hub: a white tube standing from the floor to just under the window, with six flat white teeth reaching over
    its bore (thin tabs: no edges to round)."""
    zt, zb = T - 0.0013, 0.0016
    b = bmesh.new()
    bm_lathe(b, [(HUB_HOLE, zb), (HUB_R, zb), (HUB_R, zt), (HUB_HOLE, zt), (HUB_HOLE, zb)], segs=16, mat=1, uv=False)
    for k in range(6):
        a = 2 * math.pi * k / 6
        c, s = math.cos(a), math.sin(a)
        r0, r1, w = HUB_HOLE + 0.0004, HUB_HOLE - 0.0010, 0.0007            # a tab: base on the rim, tip toward the axis
        tab = [(r0 * c - w * s, r0 * s + w * c, zt), (r0 * c + w * s, r0 * s - w * c, zt), (r1 * c, r1 * s, zt)]
        b.faces.new([b.verts.new(p) for p in tab]).material_index = 1
    G.put(bm, b, loc=(x, y, 0.0))


def _pack(bm, x, y, r):
    """Tape wound on a hub: a brown ring from the hub to radius r (a visible rim, a groove-less side)."""
    z0, z1 = 0.0034, 0.0080
    b = bmesh.new()
    bm_lathe(b, [(HUB_R, z0), (r, z0), (r, z1), (HUB_R, z1)], segs=28, mat=2, uv=False)
    G.put(bm, b, loc=(x, y, 0.0))


def _tape_run(bm, p0, p1, z0=0.0034, z1=0.0072):
    """A length of tape standing on edge between two points: one open face (a ribbon: it has no edges to round)."""
    f = bm.faces.new([bm.verts.new((p0[0], p0[1], z0)), bm.verts.new((p1[0], p1[1], z0)),
                      bm.verts.new((p1[0], p1[1], z1)), bm.verts.new((p0[0], p0[1], z1))])
    f.material_index = 2


def build_cassette(K, C, rng, with_label=True):
    """The cassette as two bmeshes in the prop frame: (shell, body), materials as in the module docstring (`vc` corner
    colours on the label and the dark parts)."""
    shell = bmesh.new()
    G.layer(shell)
    bm = bmesh.new()
    G.layer(bm)
    fr = G.top_frame(0.0)
    wx0, wx1, wy0, wy1 = WINDOW
    grey = (0.5, 0.5, 0.5, 1.0)
    # shell: a plate with the window hole through the top (its wall tapers in), and the clear window a little below the top
    # face. Every edge is a real fillet and the side walls lean (13 deg), so no face of the shell stands on a box axis.
    G.frame_ring(shell, fr, (0.0, 0.0, W, D, CORNER), ((wx0 + wx1) / 2, (wy0 + wy1) / 2, wx1 - wx0, wy1 - wy0, 0.0012),
                 0.0, ZS, 0.0, lip=LIP_TOP, n=4, ln=4, mat=0, col=grey, lip0=LIP_BOT, ln0=3, draft=DRAFT, in_shrink=0.0030)
    G.flat_quad(shell, fr, (wx0 + wx1) / 2, (wy0 + wy1) / 2, wx1 - wx0, wy1 - wy0, ZS - 0.0004, 1, grey, r=0.0012)
    # inside: floor plate, hubs with their tape, rollers and the run of tape across the front
    G.flat_quad(bm, fr, 0.0, 0.0, W - 0.0050, D - 0.0050, 0.0014, 3, grey, r=0.0022)
    for sgn, r in ((-1, PACK_L), (1, PACK_R)):
        _pack(bm, sgn * HUB_X, HUB_Y, r)
        _hub(bm, sgn * HUB_X, HUB_Y)
    for sgn in (-1, 1):
        b = bmesh.new()
        bm_lathe(b, [(0.0, 0.0016), (ROLLER_R, 0.0016), (ROLLER_R, 0.0090), (0.0, 0.0090)], segs=14, mat=3, uv=False)
        G.put(bm, b, loc=(sgn * ROLLER_X, ROLLER_Y, 0.0))
    G.soften(bm, 0.0004, segments=1)           # the solid parts (a 0.4 mm chamfer); the pins and the tape ribbons below have none
    for sgn in (-1, 1):
        b = bmesh.new()
        bm_lathe(b, [(0.0, 0.0016), (0.0006, 0.0016), (0.0009, 0.0020), (0.0009, 0.0098), (0.0006, 0.0102), (0.0, 0.0102)],
                 segs=12, mat=4, uv=False)
        G.put(bm, b, loc=(sgn * ROLLER_X, ROLLER_Y, 0.0))
    # tape path: pack -> roller -> along the front edge (round the rollers' near side) -> roller -> pack
    for sgn, r in ((-1, PACK_L), (1, PACK_R)):
        _tape_run(bm, (sgn * (HUB_X + 0.0025), HUB_Y - r * 0.99), (sgn * ROLLER_X, ROLLER_Y + ROLLER_R * 0.9))
    _tape_run(bm, (-ROLLER_X, TAPE_Y + 0.0002), (ROLLER_X, TAPE_Y + 0.0002))
    # the near (-Y) edge: openings for the rollers and the head and the run of tape, just behind the shell's wall (seen
    # through it), so the cassette keeps its 63.5 mm depth
    fy = G.front_frame(-D / 2)
    for sgn in (-1, 1):
        G.flat_quad(bm, fy, sgn * ROLLER_X, 0.0063, 0.0090, 0.0064, -0.00030, 3, grey, r=0.0012)
    G.flat_quad(bm, fy, 0.0, 0.0063, 0.0200, 0.0068, -0.00030, 3, grey, r=0.0006)
    G.flat_quad(bm, fy, 0.0, 0.0063, 0.0640, 0.0010, -0.00020, 2)
    if with_label:
        _paint_label(bm, fr, C, rng)
    # three small screws on the top face below the window
    for sx in (-0.0455, 0.0, 0.0455):
        G.flat_disc(bm, fr, sx, -0.0262, 0.0017, ZS + 0.00004, 12, 3, grey)
    G.soften(shell, 0.0005)
    return shell, bm
