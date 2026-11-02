"""semi_truck_exterior: the tractor's hardware and lamps (bpy-free, pure numpy on `core.shell`; the contract is
semi_truck_layout). Every piece is a lathe, a sweep, a framed panel or a formed sheet; pieces that sit on TruckBody's
smooth shells are put on them with the builder's ray-cast probes (`kit.Body`; a stand-in of rounded boxes without them).

    semi_truck_exterior_front    the chrome wraparound bumper on its brackets, tow hooks, the headlamp pods (two 4 x 6 in
                                 lenses in a chrome bezel and bucket) with amber turn signals, fender markers, the
                                 convex mirrors on the fender crowns
    semi_truck_exterior_cab      air cleaners and their ducts, West Coast mirrors (tube loop, spot mirror, CB antenna),
                                 air horns, clearance lamps on the visor and the sleeper roof, rocker markers, wipers,
                                 grab handles, door handles
    semi_truck_exterior_chassis  stacks (heat shields, bands, brackets), fuel tanks (straps, cradles, filler), step boxes,
                                 the deck plate, the air and electric lines with glad hands, the tail lamps
    semi_truck_exterior_kit      tubes along bent paths, lathes on any axis, sweeps with flat caps, coils, `Body`

Groups (one object `<prop>_<group>` each): ext_bumper, ext_lamps, ext_mirrors, ext_air, ext_horns, ext_trim,
ext_stacks, ext_tanks, ext_steps, ext_deck.
"""
from . import semi_truck_exterior_cab as CAB
from . import semi_truck_exterior_chassis as CH
from . import semi_truck_exterior_front as FRONT
from . import semi_truck_exterior_kit as K
from . import semi_truck_layout as L

GROUPS = ("ext_bumper", "ext_lamps", "ext_mirrors", "ext_air", "ext_horns", "ext_trim", "ext_stacks", "ext_tanks",
          "ext_steps", "ext_deck")

LOOKS = {
    "bumper": (0.0, L.Y_BUMPER, 0.5 * sum(L.BUMPER_Z)),
    "air_cleaner_L": (L.AIR_CLEANER[0], L.AIR_CLEANER[1], 0.5 * (L.AIR_CLEANER[2] + L.AIR_CLEANER[3])),
    "step_L": (CH.STEP["x_out"], 0.5 * sum(L.STEP_Y), 0.5 * sum(CH.STEP["z"])),
    "tank_L": (L.TANK_X + L.TANK_R, 0.5 * sum(L.TANK_Y), L.TANK_Z),
    "horns": (sum(CAB.HORNS["x"]) / 3.0, CAB.HORNS["y"], L.CAB_ROOF_Z + 0.08),
    "glad_hands": (0.0, L.TRAILER_Y[0] - 0.06, CH.COUPLE_Z),
}


def static_parts(probes=None):
    """{group: Mesh} of every piece of the exterior, in the truck's frame (see the module docstring)."""
    body = K.Body(probes)
    out = {}
    for mod in (FRONT, CAB, CH):
        for g, meshes in mod.parts(body).items():
            out.setdefault(g, []).extend(meshes)
    return {g: K.S.merge(out[g]) for g in GROUPS if g in out}
