"""Bedroom tech props: a spring-arm desk lamp, a radio-cassette player and a compact cassette, for the 80s bedroom
(`card = "library:desk_lamp"`, `"library:cassette_player"`, `"library:cassette_tape"`). The builders register here; the
geometry and materials live in bedroom_tech_lamp.py, bedroom_tech_player.py and bedroom_tech_tape.py, shared bmesh blocks
in bedroom_tech_geo.py, the player's moulded cabinet (pure numpy) in bedroom_tech_cabinet.py and the pure maths (arm
kinematics, coil path, label colour pick, scribble) in bedroom_tech_maths.py (tests/test_bedroom_tech_maths.py and
tests/test_bedroom_tech_cabinet.py).

All three: metres, Z up. The origin is on the support surface (the desk top) under the middle of the base ("floor_center"),
the front is -Y (the side a person faces). Everything is named `<prop name>_<part>`, builds deterministically (no clock; the
random picks are seeded from the instance name) and colours only by palette slot (blended in linear light, so a project can
switch palettes; the dark tones are overlay / surface / base, never black). Materials are procedural Principled shaders for
EEVEE Next. `[[prop]] slots` may carry options next to the palette (non-colour keys, listed per prop). Every card has the
usual `size` (exact extent of the visible meshes), `origin`, `front`, `slots`, `use`, `colliders`, plus `bounds`
{min, max} in the prop frame (the lamp's origin is under its base, its arms reach over one side of it) and, where the prop
has custom properties on its root, `params` (their default values).

Form. Nothing visible is a box. Small edges: every hard edge (two faces turning by 55 deg or more, convex or concave) is
rounded by bedroom_tech_geo.soften (a bevel of 0.3-1.5 mm in one or two steps, 3.5 mm on the lamp's cross pieces; the new
strips are smooth shaded, the flat faces stay flat); parts too thin for a bevel (tape ribbons, hub teeth, spring wire, pins)
have no hard edges, and a frame whose back is hidden is built open. Big forms: the `form` check counts a part as a box when
most of its surface faces along three orthogonal axes, bevelled or not, so the player's cabinet and the cassette's shell are
moulded: real 5-12 mm fillets, domed ends, a dished underside and leaning walls, which keep less than 60 % of their surface
within 10 deg of three axes (see the cabinet and the shell below). Measured with mkmmd.core.form.analyse: player 0.00,
cassette 0.00, lamp 0.00 (a box = 1). The blockiness audit (hard edges = dihedral >= 75 deg) is 0 m on the player and the
cassette; on the lamp 0.06 m remain, the straight hook legs where each spring wire leaves its rivet.

desk_lamp   an articulated spring-arm desk lamp that owns a warm spot light.
  Pose: lower arm up and back, upper arm forward, the shade over the desk looking toward -Y, 35 deg below the horizon.
    pivot (0, 0.012, 0.056); lower arm 0.38 m long, 12 deg back from vertical; elbow (0, 0.091, 0.428); upper arm 0.36 m,
    3 deg below horizontal; head pivot (0, -0.269, 0.409); bulb centre (0, -0.328, 0.368). Size 0.160 x 0.512 x 0.446
    (base diameter 0.160, height 0.446 to the elbow knob, the shade's rim 0.402 m in front of the base axis, the lower arm
    leans 0.11 m behind it: bounds y -0.402 .. +0.110).
  Options: `paint` = slot name (default "love"; also "iris", "pine", "gold" ...) or "#rrggbb" of the enamel; `power` = the
    default of the custom property `power` (W).
  Objects, a chain that can be posed by turning each part about its local X: `base` (weighted disc, felt pad, pivot boss,
    two cheeks) > `arm_lower` (origin = base pivot, +Z along the arm: two bars, hinge barrels, steel pin, two springs) >
    `arm_upper` (origin = elbow: forks, elbow knob, springs) > `shade` (origin = bulb centre, +Z toward the socket neck, the
    opening toward -Z: dome with a rolled lip, enamel outside, glowing pale enamel inside, socket) > `bulb` (frosted glass,
    emissive, casts no shadow) and `light`. Materials `paint`, `inner`, `steel`, `rubber`, `knob`, `bulb`.
  Light: `<name>_light`, a SPOT in the shade frame at (0, 0, -0.002) (the bulb), colour of K.light(gold .75, rose .15, text
    .10) (about 1.0, 0.59, 0.31), spot_size 75 deg, blend 0.5, soft radius 0.03 m, shadows on, energy driven as
    `power * on`. A Blender spot shines along its local -Z, which is the shade's -Z: with the prop unrotated the beam axis is
    (0, -0.819, -0.574) in the prop frame (35 deg below the horizon toward -Y). With the bulb 0.368 m over the desk top the
    axis meets it 0.85 m from the base axis; the pool begins about 0.44 m from the base and runs on, fading, past 1.5 m.
  Aiming: rotate the prop about Z. yaw 0 looks toward -Y; the beam's horizontal heading is (sin yaw, -cos yaw): yaw 90 ->
    +X, -90 -> -X, 180 -> +Y. Keep the base about 0.15 m off a wall (the lower arm leans 0.11 m behind it).
  Custom properties on the root: `power` (60 W, >= 0) the spot's energy at `on` = 1; `on` (1, 0..1) multiplies the spot's
    energy, the bulb's emission (strength 7) and the glow of the shade's inside (up to 1.8), so keying `on` switches the lamp.
  Card: use.look `shade` (centre of the opening, 0, -0.362, 0.344) and `bulb` (0, -0.328, 0.368); `lights` ["<name>_light"];
    four hidden colliders: `col_base` cylinder R 0.080 half_h 0.0136 about the base, `col_arm_lower` / `col_arm_upper`
    capsules R 0.030 on the arm objects (a (0,0,0), b (0,0,0.38) / (0,0,0.36) in their frames), `col_shade` sphere R 0.0675
    on the shade. About 33k triangles.

cassette_player   a compact 80s boombox: radio-cassette player, 0.340 x 0.115 x 0.155 on 5 mm feet.
  Frame: origin = centre of the underside of the feet; size 0.346 x 0.127 x 0.160 (the domed ends reach x = +-0.172, the
    side switches 1.5 mm further on the right, keys and knobs stand up to 7 mm off the front, the folded antenna 4 mm off
    the back: bounds x -0.1723 .. +0.1735, y -0.065 .. +0.062); the top face is at z = 0.160.
  Options: `label` = the colour (slot name or hex) of the label of the cassette in the bay (default "iris").
  Cabinet (bedroom_tech_cabinet): a closed quad mesh swept along X from the side section (y, z): flat front (y = -0.0575, z
    0.017 - 0.152), flat top (z = 0.160, y -0.0495 .. +0.0455) and vertical upper back (y = +0.0575, z >= 0.085), fillets of
    8 mm (front-top) and 12 mm (top-back, front-bottom, back-bottom), a lower back that leans 16 mm in (13 deg) below z =
    0.085, an underside dished 6 mm in the middle (quartic: horizontal where it meets the fillets), and ends that run out
    in 5 mm quarter-round rims into a 4 mm parabolic dome (flat sides at x = +-0.168, the domes reach +-0.172). Every arc is
    sampled with an ease (short segments next to the flat faces), long straight edges are cut every 20 mm so the domes
    bend smoothly. The bay (a pocket 121 x 75 mm, 22 mm deep, walls leaning 12 deg) and the handle recess (240 x 22.5 mm, 4.5
    mm deep, walls leaning 15 deg) are cut into the flat front and top.
  Front, in the front plane y = -0.0575 (x to the right, z up): two speakers (centres x = +-0.1145, z = 0.0845: bezel radius
    0.0455, cloth radius 0.0405 with six concentric ribs and a centre dome); between them a control column of three dark
    plates 0.134 wide with 1 mm seams: the dial row (z 0.130 - 0.153: the dial window x -0.0605 .. 0.0175 in a chamfered
    bezel, a backlit gold scale with ticks and a red pointer, a level meter of two columns of six segments at x 0.0325 /
    0.0405, an LED at x 0.0555), the cassette door (z 0.049 - 0.129: a countersunk frame round a clear window x +-0.0585, z
    0.0537 - 0.1245, two hinge knuckles at its foot, a cassette with two white six-tooth hubs and brown tape standing in
    the pocket behind it) and the key row (z 0.015 - 0.048: five chunky keys at x -0.0551 + 0.0176 k, rec love, play pine,
    rew and ff iris, stop gold, with their symbols, then three small knobs at x 0.0345 / 0.0455 / 0.0565 with scale dots).
    Two thin accent stripes (pine, love) run along the foot of the front at z 0.0195 / 0.0221, behind the key plate.
    Body: silver-grey satin plastic (subtle / text / hl_high), dark trim (overlay / surface), cloth overlay / muted, rubber
    feet (four, at x +-0.145, y -0.035 / +0.018, standing 9.4 mm tall with their tops inside the dished underside). Sides
    (built on the end planes and carried onto the domes): a seam, on the right a headphone socket and three slide switches,
    on the left vent slots. Back: vents on the vertical part, the battery door, rating label and DC socket on the leaning
    part, a telescopic antenna folded along the back at z 0.1385 with its rounded hinge bar at the right. Top: a handle
    folded flat in the recess, flush with the top face.
  Objects: `body` (one mesh: shell, plates, keys, knobs, bezels, details), `grilles`, `glass` (door window and dial glass,
    clear, casts no shadow), `glow` (dial scale, ticks, LED, meter: emissive), `hinges`, `feet`, `antenna`, `cassette`
    (opaque parts of the cassette) and `cassette_shell` (its smoky translucent shell, no shadow), the hidden collider
    `col_body`. Materials `plastic`, `cloth`, `metal`, `rubber`, `glass`, `glow` and the cassette's `cass_*`.
  Custom property on the root: `glow` (1, 0..1): scales the emission of the dial (strength 0.9 at 1, brightest in the middle
    of the strip), the LED (strength 1.4) and the lit meter segments (0.9) through drivers; unlit segments keep 5 %.
  Card: use.look `dial` (-0.0215, -0.060, 0.1415) and `bay` (0, -0.0595, 0.0891); use.surface `dial` (centre (-0.0215, -0.0597,
    0.143), normal [0, -1, 0], up [0, 0, 1], size [0.072, 0.007]: the part of the strip above its tick marks, for text);
    use.rest `top` (plane, centre (0, 0, 0.160), normal +Z, size [0.30, 0.09]); one hidden box collider `col_body`
    (0.340 x 0.115 x 0.160, rnd 0.006). About 24k triangles.

cassette_tape   a compact cassette lying on the desk, 0.100 x 0.0635 x 0.0125.
  Frame: origin = centre of the underside, the long side along X, the label side up (+Z), the tape openings on the -Y edge.
  Options: `label` = the label colour: a slot name (love, gold, foam, iris, rose, pine ...) or "#rrggbb". Without it the
    colour is picked from (love, gold, foam, iris, rose, pine) by the instance name (bedroom_tech_maths.label_index: an md5
    of the name without its trailing number, plus that number): tape1 ... tape6 get six different colours, any name always
    gets the same one. The stripes and tabs use the colour's partner accent (love-pine, gold-love, foam-iris, iris-gold,
    rose-foam, pine-gold), the paper and ink come from text and pine / base.
  Objects: `shell` (smoky translucent overlay / hl_med plastic with a clear window; casts no shadow so the inside is lit
    through it) and `body` (the label: a pastel ground, dark / light / accent stripes, two tabs, a paper writing field with
    ruled lines and two lines of handwriting-like scribble that differ per instance, no text; two white hubs with six teeth
    and brown tape on them, rollers, the run of tape and the openings on the near edge, three screws). Materials `shell`,
    `window`, `label`, `hub`, `tape`, `dark`, `steel`. The shell is moulded, not a box: 3.5 mm top and 2.5 mm bottom
    fillets, side walls leaning 13 deg (the base is 3.9 mm narrower per side than the top of the wall), plan corners of 4.5
    mm, and a window hole whose wall tapers 14 deg; its size stays 0.100 x 0.0635 x 0.0125 (0.01245 to the label's ink).
  Card: use.rest `top` (plane, centre (0, 0, 0.0125), normal +Z, size [0.1, 0.0635]: tapes can be stacked), use.look `label`
    (0, 0.0161, 0.0125); no colliders; the extra key `label` is the colour spec used. About 2.7k triangles."""
import bpy
from mathutils import Vector

from . import register
from . import bedroom_tech_lamp as LAMP
from . import bedroom_tech_maths as M
from . import bedroom_tech_player as PLAYER
from . import bedroom_tech_tape as TAPE
from .cafe_kit import Kit


def _pt(p):
    return [round(float(v), 4) for v in p]


def _bounds(K):
    """Exact (min, max) of the visible meshes' vertices in the prop frame -> (size, {"min", "max"}). A mesh with a rotated frame
    would give a loose box through its bound_box; the origin of the lamp is not the centre of its footprint, so the card also
    carries the extent on each side."""
    bpy.context.view_layer.update()
    inv = K.root.matrix_world.inverted()
    lo, hi = Vector((1e9,) * 3), Vector((-1e9,) * 3)
    for o in K.root.children_recursive:
        if o.type != "MESH" or o.get("mk_collider"):
            continue
        M_ = inv @ o.matrix_world
        for v in o.data.vertices:
            p = M_ @ v.co
            lo = Vector(map(min, lo, p))
            hi = Vector(map(max, hi, p))
    r5 = [lambda v: [round(float(x), 5) for x in v]][0]       # 0.01 mm: a cassette's 0.0635 must stay 0.0635 (stacking)
    return r5(hi - lo), {"min": r5(lo), "max": r5(hi)}


@register("desk_lamp")
def desk_lamp(name, coll, root, slots=None):
    K = Kit(name, coll, root, slots)
    paint = K.slots.get("paint", "love")
    power = float(K.slots.get("power", LAMP.DEFAULT_POWER))
    parts = LAMP.assemble(K, paint, power)
    P = parts["P"]
    out = Vector(P["out"])
    rim = Vector(P["bulb"]) - out * LAMP.Z_RIM
    use = {"look": [{"name": "shade", "point": _pt(rim)}, {"name": "bulb", "point": _pt(P["bulb"])}]}
    size, bounds = _bounds(K)
    return K.card(use=use, colliders=parts["colliders"], origin="floor_center", front="-Y", size=size, bounds=bounds,
                  lights=[parts["light"].name], params={"power": K.root["power"], "on": K.root["on"]})


@register("cassette_tape")
def cassette_tape(name, coll, root, slots=None):
    K = Kit(name, coll, root, slots)
    spec = TAPE.label_spec(K, name)
    C = TAPE.colours(K, spec)
    rng = M.new_rng(name, "label")
    shell_bm, body_bm = TAPE.build_cassette(K, C, rng)
    shell_m, body_m = TAPE.cassette_materials(K, C)
    K.to_obj("body", body_bm, body_m)
    K.to_obj("shell", shell_bm, shell_m).visible_shadow = False
    x0, x1, y0, y1 = TAPE.LABEL
    use = {"rest": [{"name": "top", "type": "plane", "center": [0.0, 0.0, TAPE.T], "normal": [0, 0, 1],
                     "size": [TAPE.W, TAPE.D]}],
           "look": [{"name": "label", "point": [0.0, round((y0 + y1) / 2, 4), TAPE.T]}]}
    size, bounds = _bounds(K)
    return K.card(use=use, colliders=[], origin="floor_center", front="-Y", size=size, bounds=bounds, label=spec)


@register("cassette_player")
def cassette_player(name, coll, root, slots=None):
    K = Kit(name, coll, root, slots)
    spec = K.slots.get("label", "iris")
    use, colliders = PLAYER.assemble(K, spec, M.new_rng(name, "label"))
    size, bounds = _bounds(K)
    return K.card(use=use, colliders=colliders, origin="floor_center", front="-Y", size=size, bounds=bounds,
                  label=spec, params={"glow": K.root["glow"], "pump": K.root["pump"]})
