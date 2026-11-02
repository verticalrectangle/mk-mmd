"""semi_truck: an American long-nose conventional tractor and a 53 ft dry van, built procedurally for close shots (no
downloaded model, no textures; generic and unbadged). The numbers every part agrees on are `semi_truck_layout` (proportions
from a long-hood tractor's body-builder drawing); the parts are bpy-free modules (`semi_truck_gear` the running gear and
frame, `semi_truck_body` the smooth shells, `semi_truck_exterior` the hardware and lamps, `semi_truck_cab` the cab's inside,
`semi_truck_trailer` the van), the colours `semi_truck_mats`.

    [[prop]]
    name = "truck"
    card = "library:semi_truck"
    slots = { body = "pine", mural = "art/veggies.png" }

    [[vehicle]]
    prop = "truck"
    path = "road:road"
    lane = "R1"
    speed = 22.0
    at = 150.0

Frame: metres, Z up, ground z = 0, FORWARD -Y, +X the truck's left (the driver's side). Origin on the ground under the
middle of the drive tandem. The tractor's bumper is at y -7.14, the van's rear at y +14.94.

Options (the `slots`; colours are palette slots, '#hex', blends or 'a:b:t' mixes):
  colour roles   body, accent, trim, chassis, rubber, glass, headlamp, marker, taillamp, van, van_trim, tape,
                 interior, dash, gauges, glow, display, wood, curtain (semi_truck_layout.ROLES)
  trailer        true: the van on the fifth wheel; false: the tractor alone (bobtail)
  beams          true: real spot lights in the headlamps (`beam_power` W each, 1100; `beam_angle` 50; `beam_tilt` 4 deg
                 down), driven by `lamps`
  mural          an image (a file, project-relative, absolute or ~/..., or an asset-registry slug whose path is an image)
                 painted on both of the van's sides; `mural_left`, `mural_right` override one side, `mural_back` paints
                 the rear doors; `mural_fit` "cover" (default: uniformly scaled and centred, the overflow cropped),
                 "fill" (stretched) or "contain" (all of it, the wall colour round it). The image's alpha shows the
                 wall's palette colour behind it; the joint strips stay on top
  lamps, markers, brake, dash_on, dome   starting values of the root properties (below)

Root properties (key with [[key]]): lamps (1: headlamps and beams), markers (1: clearance, marker and tail lamps),
brake (0), dash_on (1: gauges and the CB), dome (0: the cab's dome light).

Card: size, bounds, origin "floor_center", front "-Y", wheels (all ten wheel ends; dual pairs are one object each),
steering {object "<name>_wheel", axis, ratio, wheelbase}, lights (the beams and the cab's lights), use.sit driver /
passenger, use.feet, use.grip `wheel` (a ring riding `<name>_wheel`), use.rest sill_L / sill_R, use.look, use.surface
(door_L/R, sleeper_L/R, and with the van left, right, back), use.anchor, colliders, params, trailer {object, pivot}.
"""
import importlib
import inspect
import math

import bpy
import numpy as np

from ....core import photoworld as PW
from ....core import shell as S
from .. import shell as SH
from ..mesh import box as _collider_box
from . import convertible_palette as CP
from . import register
from . import semi_truck_card as C
from . import semi_truck_layout as L
from . import semi_truck_mats as SM
from .cafe_kit import drive

PART_MODULES = ("semi_truck_gear", "semi_truck_body", "semi_truck_exterior", "semi_truck_cab", "semi_truck_trailer")
TRAILER_MODULE = "semi_truck_trailer"
TRAILER_PIVOT = C.TRAILER_PIVOT
DOME_W, DASH_GLOW_W = 4.0, 1.5


def _set_prop(root, name, value, doc):
    root[name] = float(value)
    root.id_properties_ui(name).update(description=doc, default=float(value), min=0.0, soft_min=0.0, max=1.0, soft_max=1.0)


class Probes:
    """{shell key: probe} on the evaluated shell objects in the truck's frame (a trailer shell's probe is shifted from the
    trailer's frame), built on first use."""

    def __init__(self, objs):
        self._objs, self._cache = objs, {}

    def __getitem__(self, key):
        if key not in self._cache:
            p = SH.probe(self._objs[key])
            self._cache[key] = _Shifted(p, TRAILER_PIVOT) if L.is_trailer(key) else p
        return self._cache[key]

    def __contains__(self, key):
        return key in self._objs


class _Shifted:
    """A probe whose object lies in a frame offset by `off` from the truck's: rays in and hits out in the truck's frame."""

    def __init__(self, probe, off):
        self.p, self.off = probe, np.asarray(off, float)

    def ray(self, origin, direction):
        h = self.p.ray(np.asarray(origin, float) - self.off, direction)
        return None if h is None else (h[0], h[1] + self.off, h[2])

    def height(self, x, y, z0=10.0):
        h = self.ray((x, y, z0), (0.0, 0.0, -1.0))
        return None if h is None else float(h[1][2])


def load_modules(trailer=True):
    return {m: importlib.import_module(f"{__package__}.{m}") for m in PART_MODULES if trailer or m != TRAILER_MODULE}


def collect(mods, probes_for, trailer=True):
    """From every part module: {key: Shell} of its smooth shells (`static_shells()`), {group: [Mesh]} of its static parts
    (`static_parts(probes)`, the probes on the shells already built) and {key: Part} of its moving ones (`dynamic_parts()`).
    Without the trailer, every key starting with "trailer" is left out."""
    shells, statics, dynamics = {}, {}, {}
    keep = (lambda k: True) if trailer else (lambda k: not L.is_trailer(k))
    for mod in mods.values():
        if hasattr(mod, "static_shells"):
            for k, sh in mod.static_shells().items():
                if k in shells:
                    raise ValueError(f"semi_truck: shell {k!r} is built twice")
                if keep(k):
                    shells[k] = sh
    probes = probes_for(shells)
    for mod in mods.values():
        if hasattr(mod, "static_parts"):
            takes = "probes" in inspect.signature(mod.static_parts).parameters
            for g, m in (mod.static_parts(probes) if takes else mod.static_parts()).items():
                if keep(g):
                    statics.setdefault(g, []).append(m)
        if hasattr(mod, "dynamic_parts"):
            for k, part in mod.dynamic_parts().items():
                if k in dynamics:
                    raise ValueError(f"semi_truck: dynamic part {k!r} is built twice")
                if keep(k):
                    dynamics[k] = part
    return shells, statics, dynamics


def _mural_paths(murals):
    """{side: path} of the mural images: files (project-relative, absolute, ~/...) or registry slugs (photo_world's rules)."""
    from ...runtime import CTX
    root, assets = (CTX.get("project") or {}).get("root"), (CTX.get("config") or {}).get("assets")
    out = {}
    for side, ref in murals.items():
        try:
            out[side] = PW.resolve(ref, root, assets)
        except PW.PhotoWorldError as e:
            raise ValueError(f"semi_truck: mural_{side} {ref!r}: {str(e).split(': ', 1)[-1]}") from None
    return out


# ===================================================================================================================
# the builder
# ===================================================================================================================
@register("semi_truck")
def semi_truck(name, coll, root, slots=None):
    slots = dict(slots or {})
    o = L.options(slots)
    pal = CP.Pal(slots, roles=L.ROLES)
    for pname, (val, doc) in L.PARAMS.items():
        _set_prop(root, pname, slots.get(pname, val), doc)
    mats = SM.Materials(name, pal, root)
    pivot = np.asarray(TRAILER_PIVOT, float)
    trailer_root = None
    if o["trailer"]:
        SH.purge(f"{name}_trailer")
        trailer_root = bpy.data.objects.new(f"{name}_trailer", None)
        trailer_root.empty_display_size = 0.5
        coll.objects.link(trailer_root)
        trailer_root.parent = root
        trailer_root.location = tuple(pivot)
    murals = {}
    if o["murals"]:
        for side, path in _mural_paths(o["murals"]).items():
            murals[f"mural_{side}"] = mats.mural(side, path, o["mural_fit"], pivot)

    def mat_of(i):
        role = L.MATS[int(i)]
        if role.startswith("mural_"):
            return murals.get(role) or mats.get("trailer_skin")
        return mats.get(role)

    def parent_shift(key):
        return (trailer_root, pivot) if L.is_trailer(key) else (root, np.zeros(3))

    mods = load_modules(o["trailer"])
    objs, bounds = {}, []

    def build_shells(shells):
        for key, sh in shells.items():
            par, off = parent_shift(key)
            cage = sh.mesh.moved(-off)
            cutters = [SH.cutter_object(f"{name}_{key}_cut{k}", cm.moved(-off), coll, par, mat_of)
                       for k, cm in enumerate(sh.cutters)]
            ob = SH.shell_object(f"{name}_{key}", cage, coll, par, mat_of, levels=sh.levels, bevel=sh.bevel, cutters=cutters,
                                 cutter_role=sh.cutter_role, solver=sh.solver, bake_it=True)
            lo, hi = SH.evaluated_bounds(ob)
            bounds.append((lo + off, hi + off))
            objs[key] = ob
        return Probes(objs)

    _, statics, dynamics = collect(mods, build_shells, o["trailer"])

    # ---- static parts: one object per group ("glass" groups cast no shadow, "decals" groups are graphic layers)
    for g, meshes in statics.items():
        par, off = parent_shift(g)
        mesh = S.merge(meshes)
        a, b = mesh.bbox()
        bounds.append((a, b))
        ob = SH.mesh_object(f"{name}_{g}", mesh.moved(-off), coll, par, mat_of)
        if g == "glass" or g.endswith("_glass"):
            ob.visible_shadow = False
        if g == "decals" or g.endswith("_decals"):
            SH.exempt(ob)
        objs[g] = ob

    # ---- moving parts: the wheels, the steering wheel (object `<name>_wheel`)
    for key, part in dynamics.items():
        par, off = parent_shift(key)
        oname = f"{name}_wheel" if key == "steering_wheel" else f"{name}_{key}"
        ob = SH.mesh_object(oname, part.mesh, coll, par, mat_of, loc=tuple(np.asarray(part.origin, float) - off),
                            rot=tuple(part.rot))
        V = part.mesh.V @ S.rot_matrix(*part.rot).T + np.asarray(part.origin, float)
        bounds.append((V.min(0), V.max(0)))
        objs[key] = ob
    want = ["steering_wheel"] + C.wheel_keys(o["trailer"])
    missing = [k for k in want if k not in objs]
    if missing:
        raise ValueError(f"semi_truck: no part {missing} (have {sorted(objs)})")
    for key in want:
        ob = objs[key]
        ob.rotation_mode = "QUATERNION"
        ob.rotation_quaternion = ob.rotation_euler.to_quaternion()

    build_lights(name, coll, root, pal, o)
    for base, centre, size, rot in C.collider_boxes():
        SH.purge(f"{name}_col_{base}")
        _collider_box(f"{name}_col_{base}", tuple(centre), tuple(size), coll, root, rot=rot, collider=True)
    lo = np.min([b[0] for b in bounds], axis=0)
    hi = np.max([b[1] for b in bounds], axis=0)
    return C.card(name, pal.roles(), o, mods, lo, hi)


# ===================================================================================================================
# lights
# ===================================================================================================================
def build_lights(name, coll, root, pal, o):
    """The real lights, named as the card lists them (semi_truck_card.light_names): the headlamp beams (spots driven by
    `lamps`, with `beams`), the dome light (`dome`) and the dash's glow on the driver (`dash_on`)."""
    names = iter(C.light_names(name, o["beams"]))

    def add(kind, loc, rot=(0.0, 0.0, 0.0), colour=(1.0, 1.0, 1.0), watts=1.0, **kw):
        lname = next(names)
        SH.purge(lname)
        ld = bpy.data.lights.new(lname, kind)
        ld.color = colour
        ld.energy = watts
        ld.use_shadow = kw.pop("shadow", False)
        if kind == "SPOT":
            ld.spot_size = math.radians(kw.pop("spot", 60.0))
            ld.spot_blend = kw.pop("blend", 0.5)
        ld.shadow_soft_size = kw.pop("radius", 0.05)
        ob = bpy.data.objects.new(lname, ld)
        coll.objects.link(ob)
        ob.parent = root
        ob.location = loc
        ob.rotation_euler = [math.radians(a) for a in rot]
        return ld

    if o["beams"]:
        head = pal.lin("headlamp")
        peak = max(max(head), 1e-6)
        x, y, z = L.HEADLAMP
        for sx in (1.0, -1.0):
            # a spot looks along its local -Z: turned from +90 about X to look down the road (-Y), tipped `beam_tilt` down
            ld = add("SPOT", (sx * x, y - 0.03, z), rot=(-90.0 + o["beam_tilt"], 0.0, 0.0),
                     colour=tuple(c / peak for c in head), watts=o["beam_power"], spot=o["beam_angle"], blend=0.45,
                     radius=0.08)
            drive(ld, "energy", f"{o['beam_power']:.1f} * p", var=("p", root, '["lamps"]'))
    glow = pal.mix("glow", ("text", 1.0))
    peak = max(max(glow), 1e-6)
    ld = add("POINT", L.DOME_C, colour=tuple(c / peak for c in glow), watts=DOME_W, radius=0.06)
    drive(ld, "energy", f"{DOME_W:.2f} * p", var=("p", root, '["dome"]'))
    gx, gy, gz = L.GAUGES_C
    ld = add("POINT", (gx, gy + 0.22, gz + 0.10), colour=tuple(c / peak for c in glow), watts=DASH_GLOW_W, radius=0.05)
    drive(ld, "energy", f"{DASH_GLOW_W:.2f} * p", var=("p", root, '["dash_on"]'))


