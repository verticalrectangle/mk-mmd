"""electric_bass: a 60s-70s short-scale bass of the Fender Mustang Bass kind, built procedurally (no downloaded model, no
textures): an offset solid body with rounded horns (forearm and belly contours), a bolt-on maple neck with a rosewood board,
a 30 inch scale and 19 frets, a Fender bass headstock with four in-line machine heads and clover keys, one split single coil
(two black covers), a string-through bridge with four saddles, a chrome control plate with two knobs and the jack, a pickguard,
strap buttons, four strings, a plectrum and a hanging cord; the 70s racing stripes on request. Unbranded: no logo, no
lettering.

    [[prop]]
    name = "bass"
    card = "library:electric_bass"
    slots = {body = "foam", stripe = "text"}     # optional recolour by role (below); a `stripe` adds the racing stripes

Frame (metres; everything in the card is in it; the guitar's, see electric_guitar): the bass stands UPRIGHT on its tail, Z up
along the strings toward the nut, the face (strings side) looks toward -Y, X across it with the low E at -X (seen from the
front). ORIGIN: on the centre line, on the body's flat front face (y = 0) at the nominal saddle line (z = 0): the nut is at
z = 0.762, fret n at 0.762 * 2 ** (-n / 12), the tail edge at z -0.072, the long horn's tip at 0.376, the headstock's tip at
0.991. The body slab fills y in [0, 0.038]; the board's crown plane is y = -0.0115. A right-handed player holds it as `wear`
says: turned +90 degrees about +Y (neck to her left). Everything is built by `electric_bass_parts` from `electric_bass_layout`'s
numbers (see docs/modelling.md: the bass).

Objects `<name>_<part>`: body, pickguard, control_plate, neck (wood with heel and headstock), fretboard (with its dots), frets,
nut, tuners (machine heads with clover keys), tree (the D and G strings' retainer), pickups (the two covers with their screws
and the guard's screws), bridge, knobs, plate_parts (the plate's screws and the jack's nut), decals (graphic layers: the string
holes), stripes (with a `stripe` slot; a graphic layer), buttons, back_plates (the neck plate and the string ferrules),
string_1 .. string_4 (one per string, 1 the G: they ring when played, mkmmd.core.strings), plug, pick (modelled in the grip
solver's pinch frame, lying on the face at rest), cable (a bevelled curve from the plug end down to the floor, z = -0.14: the
wear stage re-hangs it), the empties `<name>_strap_top`, `<name>_strap_bottom`, `<name>_jack` and
the hidden colliders `<name>_col_body` / `_horn` / `_neck` / `_head`. The material `<name>_strap` (a woven band) is made for the
strap the pose stage builds; no strap object exists here.

Card keys (the guitar's contract, four strings)
  size, origin ("bridge"), front ("-Y"), slots (the colour roles, below)
  use.grip    neck: frame {along (0,0,1) toward the nut, across (1,0,0) low E to G, normal (0,-1,0) out of the board}, thumb
              (0,1,0), scale, frets (the board's crown centre under each wire, index 0 = the nut), strings (low E first: nut and
              bridge points on the axis, radius, midi: the open string's pitch, object: its own `<name>_string_<n>`), tuning
              [28, 33, 38, 43] (E1 A1 D2 G2: what a `notes` line is placed with), section {width, depth (nut, last fret), p 2.6,
              board_width, board_radius} (the neck's back is exactly this superellipse), fret_height
              strum: center (the pick tip's rest point, on the string plane between the pickup and the neck's end), along,
              across, normal, zone [z0, z1] and pick {object, thickness, length, width, tip} (the pick object's frame: origin
              where the pads meet, +x toward the tip at x = tip, y across, z through the thickness)
  use.anchor  strap_top (the long horn's button), strap_bottom (the tail's), jack (the plug's end, dir out of the socket), head
  use.look    neck (fret 5), strum, head
  use.wear    stand: the chest bone, the pivot (the back at the saddle line), the placement numbers, the strap and the cable
  colliders   boxes on `_col_body`, `_col_horn`, `_col_head`, a capsule on `_col_neck`

Colour roles (`slots` of the prop; a palette slot, a hex, or a blend "gold:5,rose:3,overlay:2"): body (love), pickguard (text),
neck (a warm maple), fretboard (a darker warm wood), hardware (text: bright metal), strings (text:2,muted:1), knobs (base:3,
overlay:1: the black plastic of the knobs and the pickup covers), strap (base:2,iris:1: dark, never black), cable (text), pick
(gold), inlay (bone: nut and dots), dark (base: holes), stripe (the racing stripes: only with a `stripe` slot).
"""
import bpy
from mathutils import Matrix

from ..mesh import box as _collider_box
from . import electric_bass_hw as H
from . import electric_bass_layout as LAY
from . import electric_bass_mats as EM
from . import electric_bass_parts as PT
from . import register
from .. import shell as SH


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


@register("electric_bass")
def electric_bass(name, coll, root, slots=None):
    pal = LAY.Pal(slots)
    mats = EM.Materials(name, pal)
    objs = {}
    for key, mesh in PT.parts(pal.stripes).items():
        objs[key] = SH.mesh_object(f"{name}_{key}", mesh, coll, root, mats.of)
    for key in PT.GRAPHIC:
        if key in objs:
            SH.exempt(objs[key])
    loc, R = H.pick_rest()                                         # the pick lies on the face, tip toward the neck
    pick = objs["pick"]
    pick.location = tuple(float(v) for v in loc)
    pick.rotation_euler = Matrix(R.tolist()).to_euler()
    _cable(name, coll, root, mats.role("cable"))
    # anchors: empties at the card's points
    _, _, jack_end = LAY.jack_geometry()
    _empty(f"{name}_strap_top", LAY.STRAP_TOP["point"], coll, root)
    _empty(f"{name}_strap_bottom", LAY.STRAP_BOTTOM["point"], coll, root)
    _empty(f"{name}_jack", jack_end, coll, root)
    # hidden colliders for hair and skirt
    for key, (center, size) in LAY.COLLIDERS.items():
        SH.purge(f"{name}_col_{key}")
        _collider_box(f"{name}_col_{key}", center, size, coll, root, collider=True)
    lo, hi = PT.bounds()
    return LAY.card(name, pal.roles(), size=hi - lo)
