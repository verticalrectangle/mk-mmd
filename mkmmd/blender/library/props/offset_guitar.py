"""offset_guitar: an offset-waist electric guitar of the late 1950s kind, built procedurally (no downloaded model, no
textures): a long body with a skewed waist, a long bass horn and a shallow treble shoulder, forearm and belly contours, the
electric guitar's bolt-on neck (22 frets, a 6-in-line headstock), a big tortoiseshell pickguard, two wide flat pickups, a
rocking bridge with barrel saddles, a floating tremolo with its long arm, volume and tone knobs, a three-way toggle, a
slide switch and two thumb wheels, the jack on the guard with its plug and a hanging cord, strap buttons, strings, a
plectrum. Unbranded: no logo, no lettering.

    [[prop]]
    name = "guitar"
    card = "library:offset_guitar"
    wear = "rin"
    anchor_to = "rin"                    # the tremolo arm in her picking hand: it swings with her strumming
    slots = {body = "#efe6cf"}           # optional recolour by role; guard = "plain", arm = false or {swing, lift, length}

Frame, origin and the neck are the electric guitar's (`electric_guitar`): upright on its tail, Z up along the strings, the
face toward -Y, the low E and the long horn at -X, the origin on the face at the saddle line. The tail edge is at z = -0.195,
the long horn's tip at 0.293, the body 0.348 wide. Everything is built by `offset_guitar_parts` from `offset_guitar_layout`'s
numbers (see docs/modelling.md: the offset guitar).

Objects `<name>_<part>`: body, pickguard, neck, fretboard, frets, nut, tuners, truss, pickups (with the guard's screws),
bridge, tremolo (plate, string anchor, lock button, collet), trem_arm (the arm: its origin at its bend above the collet, it
aims its +Z at the empty `<name>_trem_aim`, which stands at the arm's tip), knobs, switch, jack, decals (graphic layers),
buttons, back_plates, string_1 .. string_6 (they ring when played, mkmmd.core.strings), plug, pick, cable, the empties
`<name>_strap_top`, `<name>_strap_bottom`, `<name>_jack`, `<name>_trem_aim` and the hidden colliders `<name>_col_body` /
`_horn` / `_neck` / `_head`. The material `<name>_strap` is made for the strap the pose stage builds.

Card keys: the electric guitar's (`use.grip` neck and strum, `use.anchor` strap_top, strap_bottom, jack, head, `use.look`,
`use.wear` stand, colliders), with the strum point between the two pickups, and `use.anchor` `trem`: the arm's tip, with
the bone (`wrist.R`) and the object (`<name>_trem_aim`) that `[[prop]] anchor_to = "<cast>"` parents to that hand. The
arm then follows the hand: holding it while strumming is how these guitars are played in shoegaze.

Colour roles (`slots`): the electric guitar's, with body (an old white gone cream), pickguard and guard_dark (the amber
flames and the dark ground of the tortoiseshell; `guard = "plain"`: the pickguard role alone), knobs (white plastic: the
knobs, the pickup covers, the toggle's and the arm's tips).
"""
import bpy
from mathutils import Matrix

from ..mesh import box as _collider_box
from . import electric_guitar_mats as EM
from . import offset_guitar_hw as H
from . import offset_guitar_layout as LAY
from . import offset_guitar_parts as PT
from . import register
from .. import shell as SH
from .convertible_palette import rgba


def _empty(name, point, coll, root, size=0.012):
    SH.purge(name)
    e = bpy.data.objects.new(name, None)
    e.empty_display_type, e.empty_display_size = "ARROWS", size
    coll.objects.link(e)
    e.parent = root
    e.location = tuple(float(v) for v in point)
    return e


def _cable(name, coll, root, mat):
    """`<name>_cable`: a NURBS curve with a round bevel (radius 3.2 mm) from the plug's end to the floor."""
    old = bpy.data.objects.get(f"{name}_cable")
    if old is not None:
        data = old.data
        bpy.data.objects.remove(old)
        if data is not None and data.users == 0:
            bpy.data.curves.remove(data)
    pts = H.cable_path()
    cu = bpy.data.curves.new(f"{name}_cable", "CURVE")
    cu.dimensions = "3D"
    cu.bevel_depth = LAY.CABLE_R
    cu.bevel_resolution = 2
    cu.resolution_u = 12
    cu.use_fill_caps = True
    sp = cu.splines.new("NURBS")
    sp.points.add(len(pts) - 1)
    sp.order_u = 4
    sp.use_endpoint_u = True
    for pt, p in zip(sp.points, pts):
        pt.co = (float(p[0]), float(p[1]), float(p[2]), 1.0)
    cu.materials.append(mat)
    o = bpy.data.objects.new(f"{name}_cable", cu)
    coll.objects.link(o)
    o.parent = root
    return o


def _tortoise(mat, flame, ground):
    """Tortoiseshell on the guard's material: flames of `flame` in the `ground`, blotches a few centimetres across drawn
    out along the guitar, from a noise in the object's own space pushed to a hard contrast."""
    nt = mat.node_tree
    b = nt.nodes.get("Principled BSDF")
    tc = nt.nodes.new("ShaderNodeTexCoord")
    mp = nt.nodes.new("ShaderNodeMapping")
    mp.inputs["Scale"].default_value = (36.0, 36.0, 20.0)
    nz = nt.nodes.new("ShaderNodeTexNoise")
    nz.inputs["Scale"].default_value = 1.0
    nz.inputs["Detail"].default_value = 3.0
    nz.inputs["Roughness"].default_value = 0.62
    gain = nt.nodes.new("ShaderNodeMath")
    gain.operation = "MULTIPLY_ADD"
    gain.use_clamp = True
    gain.inputs[1].default_value = 5.0                      # the noise is 0.5 +- 0.12: flames over two fifths of it
    gain.inputs[2].default_value = -2.4
    mix = nt.nodes.new("ShaderNodeMix")
    mix.data_type = "RGBA"
    mix.inputs[6].default_value = rgba(ground)
    mix.inputs[7].default_value = rgba(flame)
    nt.links.new(tc.outputs["Object"], mp.inputs["Vector"])
    nt.links.new(mp.outputs["Vector"], nz.inputs["Vector"])
    nt.links.new(nz.outputs["Fac"], gain.inputs[0])
    nt.links.new(gain.outputs[0], mix.inputs[0])
    nt.links.new(mix.outputs[2], b.inputs["Base Color"])
    nt.links.new(mix.outputs[2], b.inputs["Emission Color"])


@register("offset_guitar")
def offset_guitar(name, coll, root, slots=None):
    pal = LAY.Pal(slots)
    mats = EM.Materials(name, pal)
    if pal.guard == "tortoise":
        _tortoise(mats.role("pickguard"), pal.lin("pickguard"), pal.lin("guard_dark"))
    objs = {}
    for key, mesh in PT.parts().items():
        objs[key] = SH.mesh_object(f"{name}_{key}", mesh, coll, root, mats.of)
    for key in PT.GRAPHIC:
        SH.exempt(objs[key])
    loc, R = H.pick_rest()                                         # the pick lies on the pickguard, tip toward the neck
    pick = objs["pick"]
    pick.location = tuple(float(v) for v in loc)
    pick.rotation_euler = Matrix(R.tolist()).to_euler()
    _cable(name, coll, root, mats.role("cable"))
    # anchors: empties at the card's points
    _, _, jack_end = LAY.jack_geometry()
    _empty(f"{name}_strap_top", LAY.STRAP_TOP["point"], coll, root)
    _empty(f"{name}_strap_bottom", LAY.STRAP_BOTTOM["point"], coll, root)
    _empty(f"{name}_jack", jack_end, coll, root)
    if pal.arm is not None:                                        # the arm turns about its bend toward the aim at its tip
        bend, tip, Ra = LAY.arm_geometry(pal.arm)
        arm = SH.mesh_object(f"{name}_trem_arm", PT.arm(pal.arm), coll, root, mats.of)
        arm.location = tuple(float(v) for v in bend)
        arm.rotation_euler = Matrix(Ra.tolist()).to_euler()
        aim = _empty(f"{name}_trem_aim", tip, coll, root)
        track = arm.constraints.new("DAMPED_TRACK")
        track.target, track.track_axis = aim, "TRACK_Z"
    # hidden colliders for hair and skirt
    for key, (center, size) in LAY.COLLIDERS.items():
        SH.purge(f"{name}_col_{key}")
        _collider_box(f"{name}_col_{key}", center, size, coll, root, collider=True)
    lo, hi = PT.bounds(pal.arm, pal.arm is not None)
    return LAY.card(name, pal.roles(), size=hi - lo, arm=pal.arm, has_arm=pal.arm is not None)
