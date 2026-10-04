"""traffic_car: an 80s car or truck for the traffic on a night road: a lofted body in paint and dark glass, emissive head- and
taillights, amber markers, and optional real headlight beams that sweep over whatever they pass.

    [[prop]]
    name = "oncoming1"
    card = "library:traffic_car"
    slots = { body = "sedan", paint = "pine", beams = true }

    [[vehicle]]                     # driven like any car: an oncoming lane drives it against the path
    prop = "oncoming1"
    path = "road:road"
    lane = "opp1"
    speed = 27.0
    meet = { vehicle = "car", t = 6.4 }

Options (the `slots`; colours are palette slots or hex):
  body     "sedan" (a three-box 80s saloon), "wagon" (its estate), "semi" (a cab-over tractor and a 13.6 m box trailer)
  paint    body colour ("pine"; the trailer's is `trailer`, "subtle")      glass   window colour ("base")
  lamp     headlight colour ("text")     tail    taillight colour ("love")     marker  marker-light colour ("gold")
  lamps    emission of the headlights (30)     tails   of the taillights (9)     markers  of the marker lights (3.5)
  beams    real spot lights in the headlights (false), `beam_power` W each (900), `beam_angle` deg (55), `beam_tilt` deg
           down (5)
The prop faces -Y like the convertible, its origin on the ground under the middle of the car, so a [[vehicle]] drives it;
the card lists the wheels (they spin) and the beams (`lights`). A semi's trailer sides are `use.surface` entries `left` and
`right` (a company name in [[text]] makes the wall that slides past read as a truck).

Card: size, bounds, front "-Y", wheels, lights, use.surface (the semi), form_max (0.8 for the semi: a box trailer is a box
by nature)."""
import math

import bpy
import numpy as np

from ....core import shell as CS
from .. import shell as SH
from . import register
from .hearts import _colour

# (y, low, belt, top, half width, half roof width): body rings from the front face to the back (the car faces -Y)
SEDAN = [(-2.275, 0.30, 0.66, 0.665, 0.80, 0.78), (-2.20, 0.24, 0.74, 0.745, 0.86, 0.84), (-1.30, 0.22, 0.80, 0.805, 0.87, 0.85),
         (-0.85, 0.22, 0.86, 0.88, 0.87, 0.80), (-0.35, 0.22, 0.86, 1.34, 0.87, 0.66), (0.65, 0.22, 0.86, 1.36, 0.87, 0.66),
         (1.25, 0.22, 0.88, 0.92, 0.87, 0.78), (2.20, 0.25, 0.86, 0.865, 0.86, 0.84), (2.275, 0.32, 0.78, 0.785, 0.80, 0.78)]
WAGON = SEDAN[:5] + [(2.00, 0.22, 0.86, 1.38, 0.87, 0.68), (2.22, 0.26, 0.84, 1.30, 0.86, 0.66),
                     (2.275, 0.32, 0.80, 1.20, 0.82, 0.62)]
CAR = {"wheelbase": 2.62, "track": 0.74, "radius": 0.31, "width": 0.19}
# the semi: cab rings (y, low, top, half width), the trailer (front y, back y, low, high, half width), axles (y, radius, x)
CAB = [(-6.20, 1.05, 3.00, 1.18), (-6.12, 0.98, 3.16, 1.22), (-5.20, 0.98, 3.26, 1.22), (-4.45, 0.98, 3.26, 1.22),
       (-4.38, 1.05, 3.15, 1.18)]
TRAILER = (-3.70, 9.90, 1.25, 3.95, 1.27)
SEMI_AXLES = [(-5.60, 0.52, 1.02), (-3.30, 0.52, 0.96), (-2.05, 0.52, 0.96), (8.20, 0.52, 0.96), (9.45, 0.52, 0.96)]


def options(slots):
    s = dict(slots or {})
    body = s.get("body", "sedan")
    if body not in ("sedan", "wagon", "semi"):
        raise ValueError(f"traffic_car: body = sedan, wagon or semi, not {body!r}")
    num = {"lamps": 30.0, "tails": 9.0, "markers": 3.5, "beam_power": 900.0, "beam_angle": 55.0, "beam_tilt": 5.0}
    o = {k: float(s.get(k, v)) for k, v in num.items()}
    o.update(body=body, beams=bool(s.get("beams", False)))
    return o


def _ring(y, low, belt, top, hw, hr, c=0.07):
    """A body section at `y`: chamfered sill, sides up to the belt, the greenhouse up to the roof; counter-clockwise about
    +Y (the loft runs from the front to the back)."""
    P = [(-hw + c, low), (hw - c, low), (hw, low + c), (hw, belt), (hr, top), (-hr, top), (-hw, belt), (-hw, low + c)][::-1]
    return [(x, y, z) for x, z in P]


def _material(name, rgb, rough=0.35, coat=0.0, emit=0.0, metal=0.0):
    m = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    m.use_nodes = True
    b = m.node_tree.nodes.get("Principled BSDF")
    b.inputs["Base Color"].default_value = (*rgb, 1.0)
    b.inputs["Roughness"].default_value = rough
    b.inputs["Metallic"].default_value = metal
    b.inputs["Coat Weight"].default_value = coat
    b.inputs["Emission Color"].default_value = (*rgb, 1.0)
    b.inputs["Emission Strength"].default_value = emit
    m.diffuse_color = (*rgb, 1.0)
    return m


def _car_body(stations):
    """The lofted body: paint below the belt and on the roof, glass on the greenhouse (material 1)."""
    body = CS.skin([_ring(*st) for st in stations], caps=(True, True))
    belt = max(st[2] for st in stations)
    return body.assign(1, lambda c, n: (c[:, 2] > belt + 0.03) & (n[:, 2] < 0.92))


def slab(w, h, y0, y1, zc=0.0, corner=0.12, edge=0.05, steps=3, mat=0):
    """A board `w` (x) by `h` (z) from y0 to y1 with rounded corners (`corner`) and front and back edges rounded over
    `edge` m: rings of the rounded-rect section, inset along a quarter circle at each end (counter-clockwise about +Y)."""
    def ring(inset, y):
        sec = CS.rrect(h - 2 * inset, w - 2 * inset, max(corner - inset, 0.01), n=4)    # (u, v) = (z, x)
        return [(v, y, zc + u) for u, v in sec]
    ang = [math.pi / 2 * k / steps for k in range(steps + 1)]
    rings = [ring(edge * (1 - math.sin(a)), y0 + edge * (1 - math.cos(a))) for a in ang]
    rings += [ring(edge * (1 - math.sin(a)), y1 - edge * (1 - math.cos(a))) for a in reversed(ang)]
    return CS.skin(rings, caps=(True, True), mat=mat)


def _semi_body():
    cab = CS.skin([_ring(y, low, top - 1.05, top, hw, hw - 0.05) for y, low, top, hw in CAB], caps=(True, True))
    cab.assign(1, lambda c, n: (n[:, 1] < -0.6) & (c[:, 2] > 2.05))              # the windscreen
    y0, y1, z0, z1, hw = TRAILER
    trailer = slab(2 * hw, z1 - z0, y0, y1, (z0 + z1) / 2, corner=0.14, edge=0.16, mat=2)
    rails = CS.rounded_box((0.95, 3.4, 0.36), center=(0.0, -4.4, 0.78), r=0.05, mat=3)
    return CS.merge([cab, trailer, rails])


def _wheel(radius, width):
    prof = [(radius * 0.55, -width / 2), (radius - 0.04, -width / 2), (radius, -width / 2 + 0.04), (radius, width / 2 - 0.04),
            (radius - 0.04, width / 2), (radius * 0.55, width / 2)]
    return CS.lathe(prof, seg=20, closed_ends=True, mat=0).rot(ry=90.0)


def _lamp(name, coll, root, mat, centre, normal, size):
    o = SH.mesh_object(name, CS.rounded_box((size[0], 0.02, size[1]), r=min(size) * 0.3), coll, root, lambda _r: mat,
                       loc=centre, rot=(0.0, 0.0, 0.0 if abs(normal[1]) > 0.5 else 90.0))
    return o


@register("traffic_car")
def traffic_car(name, coll, root, slots=None):
    slots = dict(slots or {})
    o = options(slots)
    col = {k: _colour(slots.get(k, d), slots) for k, d in (("paint", "pine"), ("glass", "base"), ("lamp", "text"),
                                                            ("tail", "love"), ("marker", "gold"), ("trailer", "subtle"))}
    paint = _material(f"{name}_paint", col["paint"], 0.32, coat=0.6)
    glass = _material(f"{name}_glass", col["glass"], 0.06, coat=0.3)
    trailer = _material(f"{name}_trailer", col["trailer"], 0.5)
    rubber = _material(f"{name}_rubber", (0.02, 0.02, 0.025), 0.85)
    head = _material(f"{name}_headlamp", col["lamp"], 0.2, emit=o["lamps"])
    tail = _material(f"{name}_taillamp", col["tail"], 0.2, emit=o["tails"])
    mark = _material(f"{name}_marker", col["marker"], 0.2, emit=o["markers"])
    mats = [paint, glass, trailer, rubber]
    semi = o["body"] == "semi"
    cage = _semi_body() if semi else _car_body(WAGON if o["body"] == "wagon" else SEDAN)
    body = SH.shell_object(f"{name}_body", cage, coll, root, lambda role: mats[role] if isinstance(role, int) else paint,
                           levels=1, bake_it=True)
    lo, hi = SH.evaluated_bounds(body)
    front, back = lo[1] - 0.004, hi[1] + 0.004
    lamps, wheels, lights = [], [], []
    if semi:
        for x in (-0.85, 0.85):
            lamps.append(_lamp(f"{name}_head{'L' if x > 0 else 'R'}", coll, root, head, (x, front, 1.12), (0, -1, 0), (0.3, 0.16)))
        for i, x in enumerate(np.linspace(-0.5, 0.5, 5)):              # cab roof clearance lights
            lamps.append(_lamp(f"{name}_cab{i}", coll, root, mark, (x, CAB[0][0] + 0.06, 3.12), (0, -1, 0), (0.09, 0.05)))
        y0, y1, z0, z1, hw = TRAILER
        for i, y in enumerate(np.arange(y0 + 1.0, y1 - 0.5, 2.6)):      # trailer side markers, low and high
            for side, x in (("L", hw + 0.004), ("R", -hw - 0.004)):
                for k, z in enumerate((z0 + 0.08, z1 - 0.08)):
                    lamps.append(_lamp(f"{name}_side{side}{i}_{k}", coll, root, mark, (x, y, z), (1, 0, 0), (0.09, 0.05)))
        for side, x in (("L", 1.05), ("R", -1.05)):
            lamps.append(_lamp(f"{name}_tail{side}", coll, root, tail, (x, back, z0 + 0.2), (0, 1, 0), (0.26, 0.14)))
            lamps.append(_lamp(f"{name}_tailtop{side}", coll, root, tail, (x * 1.1, back, z1 - 0.08), (0, 1, 0), (0.09, 0.05)))
        axles, head_z, head_x = SEMI_AXLES, 1.12, 0.85
    else:
        for side, x in (("L", 0.56), ("R", -0.56)):
            lamps.append(_lamp(f"{name}_head{side}", coll, root, head, (x, front, 0.64), (0, -1, 0), (0.34, 0.14)))
            lamps.append(_lamp(f"{name}_tail{side}", coll, root, tail, (x * 0.95, back, 0.74), (0, 1, 0), (0.42, 0.12)))
            lamps.append(_lamp(f"{name}_mark{side}", coll, root, mark, (x * 1.45, front + 0.06, 0.62), (1, 0, 0), (0.07, 0.05)))
        wb, r = CAR["wheelbase"], CAR["radius"]
        axles = [(-wb / 2 - 0.05, r, 0.80), (wb / 2 - 0.05, r, 0.80)]
        head_z, head_x = 0.64, 0.56
    for i, (y, r, x) in enumerate(axles):
        for side, xs in (("L", x), ("R", -x)):
            w = SH.mesh_object(f"{name}_wheel{i}{side}", _wheel(r, CAR["width"] if not semi else 0.5), coll, root,
                               lambda _r: rubber, loc=(xs, y, r))
            wheels.append({"object": w.name, "radius": r, "axis": [1, 0, 0]})
    if o["beams"]:
        for side, x in (("L", head_x), ("R", -head_x)):
            ld = bpy.data.lights.get(f"{name}_beam{side}") or bpy.data.lights.new(f"{name}_beam{side}", "SPOT")
            ld.energy, ld.spot_size, ld.spot_blend = o["beam_power"], math.radians(o["beam_angle"]), 0.45
            ld.color = tuple(min(1.0, c / max(col["lamp"])) for c in col["lamp"])
            ld.use_shadow = False
            ob = bpy.data.objects.get(f"{name}_beam{side}") or bpy.data.objects.new(f"{name}_beam{side}", ld)
            if ob.name not in coll.objects:
                coll.objects.link(ob)
            ob.parent = root
            ob.location = (x, front - 0.02, head_z)
            ob.rotation_euler = (math.radians(-90.0 + o["beam_tilt"]), 0.0, 0.0)     # a spot shines along -Z: turned to -Y
            lights.append(ob.name)
    lo, hi = [float(v) for v in lo], [float(v) for v in hi]
    surfaces = []
    if semi:
        y0, y1, z0, z1, hw = TRAILER
        for side, sx in (("left", 1.0), ("right", -1.0)):
            surfaces.append({"name": side, "center": [sx * (hw + 0.004), (y0 + y1) / 2, (z0 + z1) / 2], "normal": [sx, 0.0, 0.0],
                             "up": [0.0, 0.0, 1.0], "size": [y1 - y0 - 1.2, z1 - z0 - 0.7]})
    return {"size": [round(h - l, 3) for l, h in zip(lo, hi)], "origin": "floor_center", "front": "-Y", "blocks": True,
            "bounds": {"min": [round(v, 3) for v in lo], "max": [round(v, 3) for v in hi]}, "wheels": wheels,
            "lights": lights, "lamps": [ob.name for ob in lamps], "form_max": 0.8 if semi else 0.25,
            "use": {"surface": surfaces},
            "form_exempt": [f"{name}_head*", f"{name}_tail*", f"{name}_mark*", f"{name}_cab*", f"{name}_side*"],
            "slots": {k: slots.get(k, d) for k, d in (("paint", "pine"), ("glass", "base"))}}
