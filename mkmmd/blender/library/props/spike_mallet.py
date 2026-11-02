"""The spike mallet (`card = "library:spike_mallet"`): `<name>_head` (the turned barrel, its origin at the head's middle),
`<name>_bands` (the two iron hoops, about the same middle) and `<name>_handle` (from its end at the root's origin up
through the head) under the root. A hand holds it by the card's `use.grip` `hold`; `use.anchor` `face` is the middle of
the striking (+X) face. Static: no live properties. Numbers, roles and the card: spike_mallet_layout.py.

Materials, one per part, every colour a colour role resolved from the slots this build is handed (so a breathing palette
drives the sockets; no colour is kept between builds): the head's wood shows growth rings round a pith a little off its
axis (a wave texture's rings along X in the head's own coordinates), long figure on the turned side and the end grain
(the `grain` colour, rougher) where the surface faces along the axis; the hoops are dark hammered iron with the
facing-angle glow of the library's metals; the handle is straight-grained, its butt and its tenon's end show end grain,
the wedge a dark line across the tenon. Each surface glows FLOOR of its colour, so it keeps its colour in a dark world."""
import math

from .. import shell as SH
from ..sets import nightkit as NK
from . import register
from . import spike_mallet_layout as LAY
from .cafe_kit import Kit
from .electric_guitar_mats import _metal_glow

FLOOR = 0.08                        # share of its colour a wooden surface always glows
PITH = (0.026, -0.034)              # the head's pith (y, z) off its axis: its rings are centred there
RING_HEAD, RING_HANDLE = 0.0062, 0.0019   # growth-ring spacing (m) of the head's log and of the handle's billet
IRON = 0.6                         # the hoops' iron: the `band` colour darkened by this


def _dark(c, k):
    return (c[0] * k, c[1] * k, c[2] * k, 1.0)


def _map(nb, vec, scale=(1.0, 1.0, 1.0), loc=(0.0, 0.0, 0.0)):
    return nb.node("ShaderNodeMapping", {"Vector": vec, "Location": loc, "Scale": scale}).outputs[0]


def _rings(nb, vec, direction, spacing):
    """0 .. 1, 1 on the dark late wood of each growth ring: wave rings about the axis `direction` of `vec`, wandering out
    of round across it (a slow noise bends them a ring spacing or two) and nearly straight along it (the noise is
    stretched along the axis, which the rings' distance ignores)."""
    stretch = tuple(0.12 if a == direction else 1.0 for a in "XYZ")
    w = nb.node("ShaderNodeTexWave", {"Vector": _map(nb, vec, scale=stretch), "Scale": 2.0 * math.pi / (20.0 * spacing),
                                      "Distortion": 14.0, "Detail": 2.0, "Detail Scale": 0.16, "Detail Roughness": 0.45},
                wave_type="RINGS", rings_direction=direction, wave_profile="SIN").outputs["Fac"]
    return nb.smooth(0.6, 0.95, w)


def _rays(nb, q):
    """0 .. 1: faint rays from the pith across the rings (the end grain's medullary rays), `q` the point from the pith
    (the pith's axis along X): a noise round the pith (cos, sin of the angle: no seam) stretched along the radius."""
    _, qy, qz = nb.sep(q)
    r = nb.vlen(nb.comb(0.0, qy, qz))
    cos, sin = nb.math("DIVIDE", qy, r), nb.math("DIVIDE", qz, r)
    around = nb.comb(nb.math("MULTIPLY", cos, 34.0), nb.math("MULTIPLY", sin, 34.0), nb.math("MULTIPLY", r, 18.0))
    return nb.smooth(0.58, 0.78, nb.noise(around, scale=1.0, detail=2.0, rough=0.5))


def _head_mat(name, wood, grain):
    m, nb = NK.new_material(f"{name}_head")
    co = nb.texcoord("Object")
    q = _map(nb, co, loc=(0.0, -PITH[0], -PITH[1]))
    ring = _rings(nb, q, "X", RING_HEAD)
    fibre = nb.smooth(0.45, 0.8, nb.noise(_map(nb, co, scale=(6.0, 150.0, 150.0)), scale=1.0, detail=4.0, rough=0.6))
    end = nb.smooth(0.5, 0.85, nb.math("ABSOLUTE", nb.sep(nb.texcoord("Normal"))[0]))   # the faces: end grain
    side = nb.mixc(nb.math("MULTIPLY_ADD", ring, 0.7, nb.math("MULTIPLY", fibre, 0.3)), wood, _dark(wood, 0.62))
    endc = nb.mixc(nb.math("MULTIPLY_ADD", _rays(nb, q), 0.3, ring), grain, _dark(grain, 0.58))
    col = nb.mixc(end, side, endc)
    bump = nb.node("ShaderNodeBump", {"Strength": 0.2, "Distance": 0.0004, "Height": ring}).outputs["Normal"]
    nb.output(nb.principled(col, rough=nb.mixf(end, 0.52, 0.8), spec=0.42, normal=bump, emission=col,
                            strength=FLOOR))
    return m


def _band_mat(name, band):
    """Hammered black iron: the `band` colour darkened (IRON), lighter where it is worn, dented."""
    m, nb = NK.new_material(f"{name}_band")
    co = nb.texcoord("Object")
    dents = nb.voronoi(co, scale=60.0, feature="F1", which="Distance")                 # the hammer's dents
    mottle = nb.noise(co, scale=35.0, detail=3.0, rough=0.55)
    worn = tuple(min(1.0, 1.5 * v + 0.01) for v in band[:3]) + (1.0,)
    col = nb.mixc(nb.math("MULTIPLY", mottle, 0.45), _dark(band, IRON), worn)
    bump = nb.node("ShaderNodeBump", {"Strength": 0.3, "Distance": 0.0008, "Height": dents}).outputs["Normal"]
    b = nb.principled(col, rough=nb.math("MULTIPLY_ADD", mottle, 0.25, 0.3), metal=0.9, normal=bump, emission=col,
                      strength=0.0)
    _metal_glow(nb.nt, b, lo=0.1, hi=0.6)
    nb.output(b)
    return m


def _handle_mat(name, handle, grain):
    m, nb = NK.new_material(f"{name}_handle")
    co = nb.texcoord("Object")
    x, _, z = nb.sep(co)
    fibre = nb.smooth(0.42, 0.78, nb.noise(_map(nb, co, scale=(220.0, 220.0, 5.0)), scale=1.0, detail=4.0, rough=0.6))
    side = nb.mixc(nb.math("MULTIPLY", fibre, 0.55), handle, _dark(handle, 0.7))
    ring = _rings(nb, _map(nb, co, loc=(-0.004, 0.006, 0.0)), "Z", RING_HANDLE)
    endc = nb.mixc(ring, grain, _dark(grain, 0.6))
    top = nb.smooth(LAY.LENGTH - 0.008, LAY.LENGTH - 0.0045, z)                        # the tenon's end on the head
    wedge = nb.math("MULTIPLY", top, nb.math("SUBTRACT", 1.0, nb.smooth(0.0007, 0.0014, nb.math("ABSOLUTE", x))))
    endc = nb.mixc(wedge, endc, _dark(grain, 0.28))
    end = nb.math("MAXIMUM", top, nb.math("SUBTRACT", 1.0, nb.smooth(0.0012, 0.0035, z)))   # ... and the butt
    col = nb.mixc(end, side, endc)
    bump = nb.node("ShaderNodeBump", {"Strength": 0.12, "Distance": 0.0003, "Height": fibre}).outputs["Normal"]
    nb.output(nb.principled(col, rough=nb.mixf(end, 0.48, 0.78), spec=0.45, normal=bump, emission=col,
                            strength=FLOOR))
    return m


@register("spike_mallet")
def spike_mallet(name, coll, root, slots=None):
    K = Kit(name, coll, root, slots)
    c = {role: K.role(role, LAY.ROLES) for role in LAY.ROLES}
    mats = {"head": _head_mat(name, c["wood"], c["grain"]), "band": _band_mat(name, c["band"]),
            "handle": _handle_mat(name, c["handle"], c["grain"])}
    mat_of = lambda i: mats[LAY.MATS[i]]
    for part, (mesh, loc) in LAY.meshes().items():
        SH.mesh_object(f"{name}_{part}", mesh, coll, root, mat_of, loc=loc)
    return LAY.card(name, {role: K.slots.get(role, LAY.ROLES[role]) for role in LAY.ROLES})
