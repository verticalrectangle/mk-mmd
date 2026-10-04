"""The bedside alarm clock of `bedroom_decor` (geometry, materials, card). See `alarm_clock` there for the contract."""
import math

import bmesh
import bpy
from mathutils import Matrix, Vector

from . import bedroom_decor_layout as LAY
from ....core import shell as SH
from ..shell import mesh_object
from . import bedroom_decor_soft as SOFT
from .cafe_kit import Kit, L, N, bevel_modifier, bm_box, bm_lathe, drive, principled


ON, OFF = 3.4, 0.10                                 # emission strength of a lit and of an unlit segment (glow = 1)


def _flat(bm, pts, zfn):
    """A printed decal on the clock's top: a single-sided quad over the (x, y) corners `pts` (counter-clockwise seen
    from above), laid 0.2 mm over the top surface z = zfn(y). No thickness, so no edges."""
    f = bm.faces.new([bm.verts.new((x, y, zfn(y) + 0.0002)) for x, y in pts])
    f.smooth = False


def _prism(bm, pts, w0, w1, to3, mat=0, smooth=True):
    """Extrude the counter-clockwise polygon `pts` [(u, v)] from w0 to w1; `to3(u, v, w)` places a vertex."""
    lo = [bm.verts.new(to3(u, v, w0)) for u, v in pts]
    hi = [bm.verts.new(to3(u, v, w1)) for u, v in pts]
    n = len(pts)
    faces = [bm.faces.new(lo[::-1]), bm.faces.new(hi)] + [bm.faces.new((lo[i], lo[(i + 1) % n], hi[(i + 1) % n], hi[i]))
                                                           for i in range(n)]
    for f in faces:
        f.material_index = mat
        f.smooth = smooth
    return lo + hi


def _plastic(K, base, colour, rough=0.36, coat=0.25, bump=0.04):
    """Moulded plastic: satin, a clear coat and a very fine texture."""
    m, nt, out = K.new_mat(base)
    tc = N(nt, "ShaderNodeTexCoord", (-900, -200))
    nz = N(nt, "ShaderNodeTexNoise", (-700, -200), inputs={"Scale": 900.0, "Detail": 2.0, "Roughness": 0.6})
    L(nt, tc.outputs["Object"], nz.inputs["Vector"])
    bp = N(nt, "ShaderNodeBump", (-450, -200), inputs={"Strength": bump, "Distance": 0.0004})
    L(nt, nz.outputs["Fac"], bp.inputs["Height"])
    b = principled(nt, out, loc=(-150, 0), **{"Base Color": colour, "Roughness": rough, "Coat Weight": coat,
                                              "Coat Roughness": 0.15, "Specular IOR Level": 0.5})
    L(nt, bp.outputs["Normal"], b.inputs["Normal"])
    return m


def _led_mat(K, base, colour, dark, on, off, level=None):
    """Emissive segment material: emission strength = glow * (off + (on - off) * level), `glow` (and `level`, a 0/1
    custom property of the root such as `colon`) read through drivers; `off` is the faint strength of an unlit one."""
    m, nt, out = K.new_mat(base)
    glow = K.param(nt, "glow", (-1500, 300))
    if level is None:
        k = N(nt, "ShaderNodeMath", (-1100, 300), operation="MULTIPLY", inputs={1: on})
        L(nt, glow, k.inputs[0])
    else:
        lv = K.param(nt, level, (-1500, 100))
        ma = N(nt, "ShaderNodeMath", (-1250, 200), operation="MULTIPLY_ADD", inputs={1: on - off, 2: off})
        L(nt, lv, ma.inputs[0])
        k = N(nt, "ShaderNodeMath", (-1000, 300), operation="MULTIPLY")
        L(nt, ma.outputs[0], k.inputs[0])
        L(nt, glow, k.inputs[1])
    b = principled(nt, out, loc=(-600, 0), **{"Base Color": dark, "Roughness": 0.35, "Emission Color": colour})
    L(nt, k.outputs[0], b.inputs["Emission Strength"])
    return m


def _glass_mat(K, colour):
    m, nt, out = K.new_mat("glass")
    principled(nt, out, **{"Base Color": colour, "Roughness": 0.04, "Specular IOR Level": 0.8, "Alpha": 0.16})
    return m


def build(name, coll, root, slots=None):
    K = Kit(name, coll, root, slots)
    opts = K.slots
    chars = LAY.parse_time(opts.get("time", "02:47"))
    led = str(opts.get("led", "love"))
    light_w = float(opts.get("light", 0.2))
    K.prop("glow", 1.0, 0.0, 4.0, "gain of the display's emission (and of its small light)")
    K.prop("colon", 1, 0, 1, "the display's colon: 1 lit, 0 unlit (key it for a blinking colon)")
    K.prop("alarm", 1, 0, 1, "alarm slider and indicator: 1 on, 0 off")

    led_col = K.blend(**{led: 1.0})
    dark_seg = K.blend(base=.8, overlay=.2)
    # ---- the case: pale shell over a dark plinth, a gold stripe, rubber feet
    shell_m = _plastic(K, "shell", K.blend(text=.60, subtle=.40))
    plinth_m = _plastic(K, "plinth", K.blend(overlay=.85, base=.15), rough=0.45, coat=0.1)
    stripe_m = K.simple_mat("stripe", K.blend(gold=.9, rose=.1), rough=0.4, coat=0.2)
    hw = LAY.CLOCK_W / 2
    yz = lambda u, v, w: (w, u, v)                                                  # polygon in (y, z), extruded along x
    f1, b1 = LAY.CLOCK_FRONT_TOP, LAY.CLOCK_BACK_TOP
    # moulded case: the side profile has real fillets (3 / 9 / 11 / 5.5 mm) and both ends are rolled over (10 mm)
    mesh_object(K.oname("shell"), SOFT.clock_shell(), K.coll, K.root, lambda r: shell_m)
    z0, z1 = LAY.CLOCK_PLINTH
    plinth = SOFT.soft_slab(SH.rrect(2 * hw - 0.001, 0.0665, 0.014, 6), z1 + 0.0004 - z0, 0.0035, 0.003, 3)
    mesh_object(K.oname("plinth"), plinth, K.coll, K.root, lambda r: plinth_m, loc=(0.0, 0.00125, z0))
    bm = bmesh.new()
    _prism(bm, [(-0.0327, z1 + 0.0006), (-0.0321, z1 + 0.0006), (-0.0321, z1 + 0.0033), (-0.0327, z1 + 0.0033)],
           -0.069, 0.069, yz)
    bevel_modifier(K.to_obj("stripe", bm, [stripe_m]), width=0.00025, segments=2, limit="ANGLE", angle=35.0)
    bm = bmesh.new()
    for sx in (-1, 1):
        for sy in (-1, 1):
            bm_lathe(bm, [(0.0, 0.0), (0.0050, 0.0), (0.0060, 0.0004), (0.0064, 0.0012), (0.0060, 0.0019), (0.0, 0.0020)],
                     segs=16, center=(sx * 0.064, sy * 0.0235, 0.0))
    K.to_obj("feet", bm, [K.simple_mat("rubber", K.blend(overlay=.7, base=.3), rough=0.8)])

    # ---- the front face: a frame placed on the sloped face (x right, y up the face, z out of it)
    phi, (fy, fz), _, nrm, up = LAY.face_frame()
    face = K.empty("face", loc=(0.0, fy, fz), rot=(math.pi / 2 - phi, 0.0, 0.0), hidden=True)
    plate_m = _plastic(K, "plate", K.blend(base=.85, overlay=.15), rough=0.18, coat=0.6, bump=0.0)
    bm = bmesh.new()
    bm_box(bm, (LAY.PLATE_W, LAY.PLATE_H, 0.0012), (0.0, 0.0, 0.0006))
    bevel_modifier(K.to_obj("plate", bm, [plate_m], parent=face), width=0.0008, segments=2, limit="ANGLE", angle=35.0)
    bm = bmesh.new()
    bm_box(bm, (LAY.PLATE_W + 0.002, LAY.PLATE_H + 0.002, 0.0006), (0.0, 0.0, 0.0034))
    glass = K.to_obj("glass", bm, [_glass_mat(K, K.blend(base=.6, overlay=.4))], parent=face)
    glass.visible_shadow = False
    bevel_modifier(glass, width=0.0002, segments=2, limit="ANGLE", angle=35.0)

    # ---- the digits: lit segments in one mesh, unlit ones in another
    cells = LAY.display_cells()
    polys = LAY.segment_polygons()
    z_lo, z_hi = 0.0013, 0.0021
    on_bm, off_bm = bmesh.new(), bmesh.new()
    for cx, ch in zip(cells["digits"], chars):
        lit = LAY.DIGIT_SEGMENTS[ch]
        for seg, pts in polys.items():
            _prism(on_bm if seg in lit else off_bm, [(cx + x, y) for x, y in pts], z_lo, z_hi, lambda u, v, w: (u, v, w),
                   smooth=False)
    on_m = _led_mat(K, "seg_on", led_col, dark_seg, ON, ON)
    off_m = _led_mat(K, "seg_off", K.blend(**{led: .8}, base=.2), dark_seg, OFF, OFF)
    for obj in (K.to_obj("seg_on", on_bm, [on_m], parent=face), K.to_obj("seg_off", off_bm, [off_m], parent=face)):
        bevel_modifier(obj, width=0.00022, segments=2, limit="ANGLE", angle=35.0)      # lit segments: soft-edged lenses
    cbm = bmesh.new()
    k = math.tan(math.radians(LAY.SLANT))
    for sy in (-1, 1):
        y = sy * LAY.DIGIT_H / 4
        bm_box(cbm, (0.0034, 0.0034, z_hi - z_lo), (cells["colon"] + y * k, y, (z_lo + z_hi) / 2))
    bevel_modifier(K.to_obj("colon", cbm, [_led_mat(K, "colon", led_col, dark_seg, ON, OFF, level="colon")], parent=face),
                   width=0.00022, segments=2, limit="ANGLE", angle=35.0)
    lbm = bmesh.new()
    bmesh.ops.create_cone(lbm, cap_ends=True, segments=12, radius1=0.0016, radius2=0.0016, depth=z_hi - z_lo,
                          matrix=Matrix.Translation((-0.057, 0.0175, (z_lo + z_hi) / 2)))
    bevel_modifier(K.to_obj("alarm_led", lbm, [_led_mat(K, "alarm_led", K.blend(gold=.9, text=.1), dark_seg, ON * 0.7, OFF,
                                                         level="alarm")], parent=face),
                   width=0.00022, segments=2, limit="ANGLE", angle=35.0)

    # ---- top: two small buttons, a speaker grille, the alarm slider
    top_ang = math.atan2(b1[1] - f1[1], b1[0] - f1[0])
    btn = {"btn1": (-0.040, K.simple_mat("btn1", K.blend(gold=.85, rose=.15), rough=0.35, coat=0.3)),
           "btn2": (0.040, K.simple_mat("btn2", K.blend(iris=.9, text=.1), rough=0.35, coat=0.3))}
    for key, (bx, mat) in btn.items():
        bm = bmesh.new()
        _prism(bm, LAY.rounded_rect(0.0095, 0.0055, 0.003, 4), -0.0018, 0.0018, lambda u, v, w: (u, v, w))
        o = K.to_obj(key, bm, [mat], loc=(bx, -0.0055, LAY.top_z(-0.0055) + 0.0012), rot=(top_ang, 0.0, 0.0))
        bevel_modifier(o, width=0.0012, segments=3, limit="ANGLE", angle=35.0)
    grille_m = K.simple_mat("grille", K.blend(overlay=.8, base=.2), rough=0.5)
    bm = bmesh.new()                                                  # speaker slits: flush printed sheets, no edges
    for i in range(5):
        y = 0.0020 + 0.0030 * i
        _flat(bm, [(-0.0425, y - 0.0006), (0.0425, y - 0.0006), (0.0425, y + 0.0006), (-0.0425, y + 0.0006)], LAY.top_z)
    K.to_obj("grille", bm, [grille_m])
    bm = bmesh.new()
    _flat(bm, [(-0.017, 0.01775), (0.017, 0.01775), (0.017, 0.02525), (-0.017, 0.02525)], LAY.top_z)
    K.to_obj("slot", bm, [grille_m])
    bm = bmesh.new()
    _prism(bm, LAY.rounded_rect(0.0045, 0.0042, 0.0022, 4), -0.0017, 0.0017, lambda u, v, w: (u, v, w))
    knob = K.to_obj("knob", bm, [K.simple_mat("knob", K.blend(pine=.9, foam=.1), rough=0.35, coat=0.3)],
                    loc=(0.0, 0.0215, LAY.top_z(0.0215) + 0.0016), rot=(top_ang, 0.0, 0.0))
    bevel_modifier(knob, width=0.0010, segments=3, limit="ANGLE", angle=35.0)
    drive(knob, "location", "-0.0095 + 0.019 * p", index=0, var=("p", root, '["alarm"]'))

    # ---- a hidden collision box and the display's small light
    cbm = bmesh.new()
    bm_box(cbm, (LAY.CLOCK_W, LAY.CLOCK_D, 0.066), (0.0, 0.0, 0.033))
    col_body = K.collider(K.to_obj("col_body", cbm, []))
    if light_w > 0.0:
        ld = bpy.data.lights.new(K.oname("light"), "POINT")
        ld.color = tuple(K.light(**{led: 1.0})[:3])
        ld.energy = light_w
        ld.shadow_soft_size = 0.02
        ld.use_shadow = False
        ld.specular_factor = 0.0                                    # spill only: no round highlight in the glass
        K.obj("light", ld, loc=(0.0, -0.05, 0.045))
        drive(ld, "energy", f"{light_w} * p", var=("p", root, '["glow"]'))

    centre = Vector((0.0, fy, fz)) + Vector(nrm) * 0.0036
    use = {"look": [{"name": "display", "point": [round(v, 4) for v in centre]}],
           "surface": [{"name": "display", "center": [round(v, 4) for v in centre], "normal": [round(v, 4) for v in nrm],
                        "up": [round(v, 4) for v in up], "size": [LAY.PLATE_W, LAY.PLATE_H]}]}
    colliders = [{"type": "box", "object": col_body.name, "rnd": 0.004, "tag": name}]
    return K.card(use=use, colliders=colliders, origin="floor_center", front="-Y", time="".join(chars).replace(" ", ""),
                  properties={"glow": 1.0, "colon": 1, "alarm": 1})
