"""convertible_80s: a 1980s American convertible, built procedurally (no downloaded model, no textures): the
Rin-and-Reisen night-highway car. Generic and unbadged, modelled in proportion on a 1984-86 Dodge 600 convertible (see
docs/modelling.md for how: the body is ONE lofted, subdivided shell, every other piece a profile, sweep or lathe): long flat
hood and deck, a shoulder crease along the whole side, flush rectangular headlamps beside a barred grille, a full-width
ribbed tail-lamp band, swept rubber bumpers with end caps, a two-tone body split at the rub strip, 15 inch turbine alloys, a
folded soft top under a padded boot, pleated bucket seats, a digital cluster, a cassette deck and a 2-spoke wheel.

    [[prop]]
    name = "car"
    card = "library:convertible_80s"
    slots = {body = "iris"}          # optional recolour by role (below)

Frame: Z up, ground z = 0, origin = the car's centre on the ground, FORWARD -Y, the driver on +X (left-hand drive).
Metres. Length 4.59, width 1.73 (mirrors 1.99), wheelbase 2.62, track 1.46, tyre radius 0.31, shoulder crease 0.83, door
top 0.915, hood 0.80 at the nose to 0.99 at its rear edge, the glass from 0.925 at y -0.65 up to a header at 1.32 (44 degrees).
`card = "library:car_mockup"` can be swapped for this card: every name the pipeline reads is the same.

Card keys (all in the prop's local frame)
  size, origin ("floor_center"), front ("-Y"), slots (the colour roles, below)
  use.sit      driver (+X) and passenger (-X): hip, facing, seat_z, floor_z, pelvis_deg, back_deg (the backrest
               recline: the sitter leans back by it). `hip` is where the pose stage puts the hips before it tips the
               pelvis (which moves the joints ~3.5 cm toward the nose), so it is 3.5 cm behind the final joint: the
               joint ends 15 cm above the cushion top, centred on the seat.
  use.feet     driver / passenger: L, R ankle positions (x, y), floor_z
  use.grip     wheel: a ring (center, axis toward the driver, radius, tube) riding the object `<name>_wheel`
  use.rest     sill_L, sill_R: the door-top edges (x +-0.805) a hand rests on
  use.look     road, mirror (rear-view mirror glass), dash (the cluster), cassette (the slot), headlamp_L / _R, tail
  use.surface  flat faces for kinetic type: speed (the digital speed readout), bars (the bar graph area), cluster (the
               whole display), radio (the radio's window), cassette_label (rides the cassette object), plate_front,
               plate_rear. Each: center, normal (faces the viewer), up (the text's up), size [w, h]. `speed` also has
               `grid`: the digit cells of its unlit ghost segments, columns 3 side by side at `pitch` 0.0417 m (cell k
               centred (k - 1) * pitch from the surface centre, u to the viewer's right), digits `digit_width` 0.034 by
               `digit_height` 0.052 m, `stroke` 0.0062, two decimal dots; a right-aligned numeral ends at the right
               edge of the last digit, (pitch - digit_width) / 2 = 0.0038 m inside the cell.
               The unlit cells (role vfd_ghost) are a faint hint, ~9 % of a lit cell's displayed brightness.
  use.anchor   slot (the cassette slot: point, dir of insertion), mirror (rear-view mirror), and the cabin's handles
               and pulls (shifter, handbrake, console_lid, armrest_L/R, door_pull_L/R, lock_pin_L/R)
  colliders    boxes on hidden objects `<name>_col_cushion_<side>`, `_col_back_<side>`, `_col_headrest_<side>`
               (side = driver | passenger), `_col_door_L`, `_col_door_R`, `_col_dash`, `_col_console`,
               `_col_floor`, `_col_tunnel`, `_col_windshield`, `_col_rear_cushion`, `_col_rear_back`, plus a `ring`
               on the wheel
  wheels       the four road wheels: {object, radius, axis} (the vehicles stage spins them about the local axis, +X: the
               car's left; a positive turn rolls the tyre forward)
  steering     {object: "<name>_wheel", axis (local), ratio}: the vehicles stage turns the object; hands that
               `ride = "<name>_wheel"` turn with it
  lights       the real light objects: headlamps (spots at the flush lamps), tail lamps, dash glow
  params       the animatable properties and their defaults

Animatable state: custom properties on the prop root, driving the parts (key them with `[[key]] target = "car"
prop = "lamps" keys = [[t, v], ...]`):
  lamps   0..1  the fixed headlamps: 0 off; at 1 the lenses glow and the two spot lights are on. They come up with a short
                warm-up flicker (a filament finding its feet) between about 0.15 and 0.9, so a ramp from 0 to 1 over 0.4 s
                reads as a switch being thrown
  brake   0..1  brake lamps (the tail band and the tail lights)
  tails   0..1  the running level of the tail band and the side markers (0.35)
  dash_on 0..1  the digital displays, the indicator lights, the dash glow light
  tape    0..1  the cassette: 0 held out in front of the slot, 1 pushed in
  bars    0..1  the bar graph level of the cluster
  visors  0..1  the sun visors (the object `<name>_visors`: both pads, their pivot rods and clips) turn about their rods:
                0 flipped up, level with the header and pointing back (the default: under a folded top a real visor is
                stowed, and a camera over the hood sees the faces), 1 down against the glass. The object can also be
                hidden by a project key: `[[key]] target = "car_visors" prop = "hide_render" keys = [[0, 1]]`

Colour roles (`slots` of the prop; a palette slot name, a hex, or a blend "gold:5,rose:3,overlay:2"): body (love),
lower (muted: the two-tone below the rub strip at z 0.43, and the bumpers), stripe (gold: the pinstripes and cassette stripe), trim (text: bright
metal), rubber (base: the darkest slot, never pure black), interior (a warm tan), display (foam), headlamp (text),
taillamp (love), glass (foam), wheel (subtle), boot (overlay:3,base:2).
"""
import importlib
import inspect
import math

import bpy
import numpy as np

from ..mesh import box as _collider_box
from . import convertible_layout as LAY
from . import convertible_mats as CM
from . import convertible_palette as CP
from . import register
from ....core import shell as S
from .. import shell as SH
from .cafe_kit import drive

PART_MODULES = ("convertible_body", "convertible_wheels", "convertible_exterior", "convertible_trim", "convertible_seats",
                "convertible_dash")

PARAMS = {                       # name: (default, doc)
    "lamps": (0.0, "headlamps: 0 off, 1 lit (lens glow and the spot lights, with a short warm-up flicker)"),
    "brake": (0.0, "brake lamps: 0 off, 1 full"),
    "tails": (0.35, "running level of the tail band and side markers"),
    "dash_on": (1.0, "digital displays, indicators and the dash glow"),
    "tape": (1.0, "cassette: 0 held out in front of the slot, 1 pushed in"),
    "bars": (0.6, "bar graph level of the cluster"),
    "visors": (0.0, "sun visors: 0 flipped up (level, pointing back from the header), 1 down against the glass"),
}

HEADLAMP_W = 300.0               # watts of one headlamp spot at lamps = 1
TAIL_BASE_W, TAIL_BRAKE_W = 4.0, 40.0
DASH_GLOW_W = 2.0
SIT = {"pelvis_deg": 12.0, "back_deg": LAY.BACK_RECLINE_DEG}
# the pose stage tips the pelvis after placing the hips, which moves the hip joints ~3.5 cm toward the nose: aim behind
HIP_AIM = 0.035
FEET = {"driver": {"L": (LAY.SEAT_X + 0.14, -0.50), "R": (LAY.SEAT_X - 0.08, -0.55)},
        "passenger": {"L": (-LAY.SEAT_X + 0.12, -0.38), "R": (-LAY.SEAT_X - 0.10, -0.40)}}


# ===================================================================================================================
# mesh -> object
# ===================================================================================================================
purge = SH.purge


def drive_vars(id_data, path, expr, vars, root, index=-1):
    """Scripted driver `expr` over several single-property variables {name: custom property of root}."""
    res = id_data.driver_add(path, index) if index >= 0 else id_data.driver_add(path)
    for fc in res if isinstance(res, list) else [res]:
        d = fc.driver
        d.type = "SCRIPTED"
        d.expression = expr
        for vn, prop in vars.items():
            v = d.variables.new()
            v.name = vn
            v.type = "SINGLE_PROP"
            v.targets[0].id_type = "OBJECT"
            v.targets[0].id = root
            v.targets[0].data_path = f'["{prop}"]'
    return res


def set_prop(root, name, value, doc):
    root[name] = value
    root.id_properties_ui(name).update(description=doc, default=value, min=0.0, soft_min=0.0, max=1.0, soft_max=1.0)


def _lst(v, nd=5):
    return [round(float(x), nd) for x in v]


# ===================================================================================================================
# the parts
# ===================================================================================================================
def load_modules():
    return {modname: importlib.import_module(f"{__package__}.{modname}") for modname in PART_MODULES}


def collect_shells(mods):
    """{key: core.shell.Shell} of the smooth shells every module defines (`static_shells()`: creased cages that become
    Subdivision Surface objects named `<prop>_<key>`)."""
    shells = {}
    for mod in mods.values():
        if hasattr(mod, "static_shells"):
            for k, sh in mod.static_shells().items():
                if k in shells:
                    raise ValueError(f"shell {k!r} is built twice")
                shells[k] = sh
    return shells


class Probes:
    """{shell key: core.shell.Probe} on the evaluated shell objects, built on first use (evaluating a shell takes time)."""

    def __init__(self, objs):
        self._objs, self._cache = objs, {}

    def __getitem__(self, key):
        if key not in self._cache:
            self._cache[key] = SH.probe(self._objs[key])
        return self._cache[key]

    def __contains__(self, key):
        return key in self._objs


def collect_parts(mods, probes):
    """From every part module: {group: [Mesh]} of the static parts (`static_parts()`; a module whose function takes a
    `probes` argument gets the shells' Probes, to put trims and lamps exactly on the smooth surface) and {key: Part} of
    the dynamic ones (`dynamic_parts()`)."""
    statics, dynamics = {}, {}
    for mod in mods.values():
        if hasattr(mod, "static_parts"):
            takes = "probes" in inspect.signature(mod.static_parts).parameters
            for g, m in (mod.static_parts(probes) if takes else mod.static_parts()).items():
                statics.setdefault(g, []).append(m)
        if hasattr(mod, "dynamic_parts"):
            for k, part in mod.dynamic_parts().items():
                if k in dynamics:
                    raise ValueError(f"dynamic part {k!r} is built twice")
                dynamics[k] = part
    return statics, dynamics


def _merged_dicts(mods, attr):
    """The union of a module-level dict (SURFACES, ANCHORS, LOOKS) over the part modules that define it."""
    out = {}
    for mod in mods.values():
        out.update(getattr(mod, attr, {}))
    return out


def part_bbox(part):
    """Bounding box (lo, hi) of a dynamic part in car coordinates."""
    m = part.mesh
    R = S.rot_matrix(*part.rot)
    V = m.V @ R.T + np.asarray(part.origin, float)
    return V.min(0), V.max(0)


# ===================================================================================================================
# the builder
# ===================================================================================================================
@register("convertible_80s")
def convertible_80s(name, coll, root, slots=None):
    pal = CP.Pal(slots)
    for pname, (val, doc) in PARAMS.items():
        set_prop(root, pname, val, doc)
    mats = CM.Materials(name, pal, root, list(PARAMS))
    role_mat = lambda moving: (lambda i: mats.for_object(LAY.MATS[i], moving))    # noqa: E731
    mods = load_modules()
    shells = collect_shells(mods)
    objs = {}

    # ---- smooth shells: a creased cage, a Subdivision Surface, optional Boolean cutters and a Bevel
    shell_bounds = []
    for key, sh in shells.items():
        cutters = [SH.cutter_object(f"{name}_{key}_cut{k}", cm, coll, root, role_mat(False)) for k, cm in enumerate(sh.cutters)]
        o = SH.shell_object(f"{name}_{key}", sh.mesh, coll, root, role_mat(False), levels=sh.levels, bevel=sh.bevel,
                            cutters=cutters, cutter_role=sh.cutter_role, solver=sh.solver, bake_it=True)       # the car moves: no live modifiers
        shell_bounds.append(SH.evaluated_bounds(o))
        objs[key] = o
    statics, dynamics = collect_parts(mods, Probes(objs))

    # ---- static objects: one per group (the group "decals" holds graphic layers, which the form check skips)
    flags = {"interior": "interior", "glass": "glass"}
    for g, meshes in statics.items():
        o = SH.mesh_object(f"{name}_{flags.get(g, g)}", S.merge(meshes), coll, root, role_mat(False))
        if g == "glass":
            o.visible_shadow = False
        if g == "decals":
            SH.exempt(o)
        objs[g] = o

    # ---- dynamic objects
    for key, part in dynamics.items():
        oname = f"{name}_wheel" if key == "steering_wheel" else f"{name}_{key}"
        o = SH.mesh_object(oname, part.mesh, coll, root, role_mat(True), loc=tuple(part.origin), rot=tuple(part.rot))
        objs[key] = o
    for k in ("steering_wheel", "cassette", "visors", "wheel_FL", "wheel_FR", "wheel_RL", "wheel_RR"):
        if k not in objs:
            raise ValueError(f"convertible_80s: no part {k!r} (have {sorted(objs)})")
    wheel_obj = objs["steering_wheel"]
    wheel_obj.rotation_mode = "QUATERNION"
    wheel_obj.rotation_quaternion = wheel_obj.rotation_euler.to_quaternion()

    # ---- the cassette slides along the slot with `tape`
    cas = objs["cassette"]
    drive_vars(cas, "location", f"{LAY.TAPE_OUT_Y:.5f} + {LAY.TAPE_IN_Y - LAY.TAPE_OUT_Y:.5f} * clamp(t, 0, 1)",
               {"t": "tape"}, root, index=1)

    # ---- the sun visors turn about their pivot rods (the object's local X) with `visors`: 1 down on the glass (the rest pose),
    # 0 flipped up, level and pointing back
    drive_vars(objs["visors"], "rotation_euler", f"{math.radians(LAY.VISOR_FLIP_DEG):.6f} * (1 - clamp(v, 0, 1))", {"v": "visors"},
               root, index=0)

    # ---- lights
    lights = build_lights(name, coll, root, pal, objs)

    # ---- colliders and the card
    cols = build_colliders(name, coll, root, mods)
    card = build_card(name, root, pal, statics, dynamics, shell_bounds, mods, cols, lights)
    return card


# ===================================================================================================================
# lights
# ===================================================================================================================
def build_lights(name, coll, root, pal, objs):
    out = []

    def add(lname, kind, parent, loc, rot=(0.0, 0.0, 0.0), colour=(1, 1, 1), watts=1.0, **kw):
        purge(lname)
        ld = bpy.data.lights.new(lname, kind)
        ld.color = colour
        ld.energy = watts
        ld.use_shadow = kw.pop("shadow", False)
        if kind == "SPOT":
            ld.spot_size = math.radians(kw.pop("spot", 80.0))
            ld.spot_blend = kw.pop("blend", 0.6)
            ld.shadow_soft_size = kw.pop("radius", 0.04)
        elif kind == "AREA":
            ld.shape = "RECTANGLE"
            ld.size, ld.size_y = kw.pop("size")
        elif kind == "POINT":
            ld.shadow_soft_size = kw.pop("radius", 0.04)
        o = bpy.data.objects.new(lname, ld)
        coll.objects.link(o)
        o.parent = parent
        o.location = loc
        o.rotation_euler = [math.radians(a) for a in rot]
        out.append(o)
        return o, ld

    head = pal.lin("headlamp")
    for side, sx in (("L", 1.0), ("R", -1.0)):
        # a spot looks along its local -Z: tipped from +90 to point down the road (-Y), two degrees toward the ground
        o, ld = add(f"{name}_headlamp_{side}", "SPOT", root, (sx * LAY.HEADLAMP_X, LAY.Y_NOSE - 0.03, LAY.HEADLAMP_Z),
                    rot=(-88.0, 0.0, 0.0), colour=head, watts=HEADLAMP_W, spot=75.0, blend=0.55, radius=0.06)
        drive_vars(ld, "energy", f"{HEADLAMP_W:.1f} * smoothstep(0.15, 0.9, p) * (1.0 - 0.3 * (1.0 - smoothstep(0.15, 0.9, p)) * "
                   "abs(sin(p * 70.0)))", {"p": "lamps"}, root)
    tail = pal.lin("taillamp")
    for side, x in (("L", 0.66), ("R", -0.66)):
        o, ld = add(f"{name}_taillamp_{side}", "AREA", root, (x, LAY.Y_TAIL + 0.03, 0.835), rot=(90.0, 0.0, 0.0),
                    colour=tail, watts=TAIL_BASE_W, size=(0.30, 0.07))
        drive_vars(ld, "energy", f"{TAIL_BASE_W:.1f} * (0.3 + 0.7 * tails) + {TAIL_BRAKE_W:.1f} * brake",
                   {"tails": "tails", "brake": "brake"}, root)
    disp = pal.lin("display")
    cn, up, _ = LAY.CLUSTER_FRAME[0], LAY.CLUSTER_FRAME[1], LAY.CLUSTER_FRAME[2]
    glow_at = tuple(np.asarray(LAY.BINNACLE_C) + np.asarray(cn) * 0.17 - np.asarray(up) * 0.03)   # clear of the glass
    o, ld = add(f"{name}_dash_glow", "POINT", root, glow_at, colour=disp, watts=DASH_GLOW_W, radius=0.05)
    drive(ld, "energy", f"{DASH_GLOW_W:.2f} * p", var=("p", root, '["dash_on"]'))
    return [o.name for o in out]


# ===================================================================================================================
# colliders
# ===================================================================================================================
def build_colliders(name, coll, root, mods):
    """Hidden box objects for the sitters' hair and clothes (docs/design.md: Colliders). Returns the card specs."""
    cols = []

    def add(base, center, size, rot=(0.0, 0.0, 0.0)):
        purge(f"{name}_col_{base}")
        o = _collider_box(f"{name}_col_{base}", tuple(center), tuple(size), coll, root, rot=rot, collider=True)
        cols.append({"type": "box", "object": o.name, "rnd": 0.01, "tag": name})

    seat = _merged_dicts(mods, "SEAT")                              # the seats module's own numbers (car coordinates)
    a = math.radians(LAY.BACK_RECLINE_DEG)
    ax = np.array([0.0, math.sin(a), math.cos(a)])                    # up the backrest
    nf = np.array([0.0, -math.cos(a), math.sin(a)])                   # out of its front face
    hy, hz = seat.get("headrest_center", (0.446, 0.5 * sum(LAY.HEADREST_Z)))
    hs = seat.get("headrest_size", (0.23, 0.10, LAY.HEADREST_Z[1] - LAY.HEADREST_Z[0]))
    for side, sx in (("driver", LAY.SEAT_X), ("passenger", -LAY.SEAT_X)):
        add(f"cushion_{side}", (sx, (LAY.CUSHION_Y[0] + LAY.CUSHION_Y[1]) / 2, LAY.CUSHION_TOP - 0.06),
            (LAY.CUSHION_W, LAY.CUSHION_Y[1] - LAY.CUSHION_Y[0], 0.12))
        base = np.array([sx, LAY.BACK_BASE_Y, LAY.CUSHION_TOP])
        c = base + ax * LAY.BACK_H / 2 - nf * 0.06
        add(f"back_{side}", c, (LAY.CUSHION_W, 0.12, LAY.BACK_H), rot=(-LAY.BACK_RECLINE_DEG, 0.0, 0.0))
        add(f"headrest_{side}", (sx, hy, hz), tuple(hs), rot=(-LAY.BACK_RECLINE_DEG, 0.0, 0.0))
    for side, sx in (("L", 1.0), ("R", -1.0)):
        add(f"door_{side}", (sx * (LAY.X_BELT_OUT - LAY.WALL_T / 2), (LAY.Y_COWL + LAY.Y_DOOR_REAR) / 2, 0.62),
            (LAY.WALL_T, LAY.Y_DOOR_REAR - LAY.Y_COWL, 0.74))
    y0, y1 = LAY.Y_COWL, LAY.Y_BOOT
    add("floor", (0.0, (y0 + y1) / 2, LAY.FLOOR_Z - 0.015), (2 * LAY.X_WALL_IN, y1 - y0, 0.03))      # carpet: cloth rests on it
    add("tunnel", (0.0, 0.125, (LAY.FLOOR_Z + 0.31) / 2), (0.32, 1.65, 0.31 - LAY.FLOOR_Z))
    add("dash", (0.0, 0.5 * (LAY.Y_COWL - 0.40), 0.805), (2 * LAY.DASH_X, -0.40 - LAY.Y_COWL, 0.41))   # the pad and the upper face; the knee bolster is clear
    add("console", (0.0, (LAY.CONSOLE_Y[0] + LAY.CONSOLE_Y[1]) / 2, 0.415),
        (2 * LAY.CONSOLE_X, LAY.CONSOLE_Y[1] - LAY.CONSOLE_Y[0], 0.37))
    (y0, z0), (y1, z1) = LAY.WS_BASE, LAY.WS_TOP
    glen = math.hypot(y1 - y0, z1 - z0)
    add("windshield", (0.0, (y0 + y1) / 2, (z0 + z1) / 2), (2 * LAY.WS_HALF_W_TOP, 0.03, glen),
        rot=(-(90.0 - LAY.WS_RAKE_DEG), 0.0, 0.0))
    cy0, cy1 = LAY.REAR_SEAT_Y
    add("rear_cushion", (0.0, (cy0 + cy1) / 2, LAY.REAR_SEAT_TOP - 0.06), (LAY.REAR_SEAT_W, cy1 - cy0, 0.12))
    add("rear_back", (0.0, cy1 + 0.07, 0.72), (LAY.REAR_SEAT_W, 0.12, 0.50),
        rot=(-LAY.REAR_BACK_RECLINE_DEG, 0.0, 0.0))
    cols.append({"type": "ring", "object": f"{name}_wheel", "radius": LAY.WHEEL_R, "tube": LAY.WHEEL_TUBE, "tag": name})
    return cols


# ===================================================================================================================
# the card
# ===================================================================================================================
def build_card(name, root, pal, statics, dynamics, shell_bounds, mods, cols, lights):
    lo, hi = np.full(3, 1e9), np.full(3, -1e9)
    for meshes in statics.values():
        for m in meshes:
            a, b = m.bbox()
            lo, hi = np.minimum(lo, a), np.maximum(hi, b)
    for a, b in shell_bounds:
        lo, hi = np.minimum(lo, a), np.maximum(hi, b)
    for part in dynamics.values():
        a, b = part_bbox(part)
        lo, hi = np.minimum(lo, a), np.maximum(hi, b)
    surf_src, anchor_src, look_src = (_merged_dicts(mods, k) for k in ("SURFACES", "ANCHORS", "LOOKS"))

    sits, feet = [], []
    for side, sx in (("driver", LAY.SEAT_X), ("passenger", -LAY.SEAT_X)):
        sits.append({"name": side, "hip": [sx, LAY.HIP_Y + HIP_AIM, LAY.HIP_Z], "facing": [0, -1, 0], "seat_z": LAY.CUSHION_TOP,
                     "floor_z": LAY.FLOOR_Z, **SIT})
        f = FEET[side]
        feet.append({"name": side, "L": [round(f["L"][0], 4), round(f["L"][1], 4), None],
                     "R": [round(f["R"][0], 4), round(f["R"][1], 4), None], "floor_z": LAY.FLOOR_Z})
    rest = [{"name": f"sill_{s}", "type": "edge", "a": [x, LAY.SILL_Y[0], LAY.Z_BELT], "b": [x, LAY.SILL_Y[1], LAY.Z_BELT],
             "normal": [0, 0, 1]} for s, x in (("L", LAY.SILL_X), ("R", -LAY.SILL_X))]
    grip = [{"name": "wheel", "type": "ring", "center": _lst(LAY.WHEEL_C), "axis": _lst(LAY.WHEEL_AXIS),
             "radius": LAY.WHEEL_R, "tube": LAY.WHEEL_TUBE, "object": f"{name}_wheel"}]
    looks = [{"name": "road", "point": list(LAY.LOOK_ROAD)}, {"name": "mirror", "point": _lst(LAY.MIRROR_C)},
             {"name": "dash", "point": _lst(LAY.BINNACLE_C)}, {"name": "cassette", "point": _lst(LAY.SLOT_C)},
             {"name": "headlamp_L", "point": [LAY.HEADLAMP_X, LAY.Y_FRONT, LAY.HEADLAMP_Z]},
             {"name": "headlamp_R", "point": [-LAY.HEADLAMP_X, LAY.Y_FRONT, LAY.HEADLAMP_Z]},
             {"name": "tail", "point": [0.0, LAY.Y_REAR, 0.84]}]
    looks += [{"name": k, "point": _lst(v)} for k, v in look_src.items() if k not in ("dash", "cassette")]
    surfaces = []
    grids = _merged_dicts(mods, "GRIDS")                          # extra layout a surface documents (the speed digit grid)
    for k, s in surf_src.items():
        e = {"name": k, "center": _lst(s["center"]), "normal": _lst(s["normal"]), "up": _lst(s["up"]),
             "size": _lst(s["size"], 4)}
        if k == "cassette_label":
            e["object"] = f"{name}_cassette"
        if k in grids:
            e["grid"] = grids[k]
        surfaces.append(e)
    for k, c, n in (("plate_front", LAY.PLATE_FRONT, (0, -1, 0)), ("plate_rear", LAY.PLATE_REAR, (0, 1, 0))):
        surfaces.append({"name": k, "center": _lst(c), "normal": list(n), "up": [0, 0, 1], "size": _lst(LAY.PLATE_SIZE, 4)})
    anchors = {"slot": {"point": LAY.SLOT_C, "dir": [0, -1, 0]}, "mirror": {"point": LAY.MIRROR_C}}
    for k, a in anchor_src.items():
        anchors[k] = {**anchors.get(k, {}), **a}
    anchors = [{"name": k, "point": _lst(a["point"]), **({"dir": _lst(a["dir"])} if "dir" in a else {})}
               for k, a in anchors.items()]
    # the wheel objects are unrotated and spin about their local X, which points to the car's left: that rolls forward
    wheels = [{"object": f"{name}_wheel_{k}", "radius": LAY.TYRE_R, "axis": [1.0, 0.0, 0.0]}
              for k in ("FL", "FR", "RL", "RR")]
    return {
        "size": _lst(hi - lo, 3), "origin": "floor_center", "front": "-Y", "slots": pal.roles(),
        "use": {"sit": sits, "feet": feet, "grip": grip, "rest": rest, "look": looks, "surface": surfaces,
                "anchor": anchors},
        "colliders": cols, "wheels": wheels,
        "steering": {"object": f"{name}_wheel", "axis": [0.0, 0.0, 1.0], "ratio": LAY.STEER_RATIO},
        "lights": lights, "params": {k: v[0] for k, v in PARAMS.items()},
    }
