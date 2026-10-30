"""butterflies: a swarm of butterflies round the root, flapping, in formations blended by keyable weights: a ring mandala
facing -Y, tilted orbits round a centre (round her head with `attach`), a wandering flock; `burst` flings them out and
away, `bloom` grows the rings in from the centre. The maths is mkmmd/core/butterflies.py (written once: numpy for the
tests, geometry nodes here through butterflies_nodes.py); one butterfly's parts, flap and colours are
mkmmd/core/butterfly.py and butterfly_parts.py.

    [[prop]]
    name = "swarm"
    card = "library:butterflies"
    at = [0.0, 0.3, 1.3]                       # the mandala's centre, behind her; it faces -Y like a character
    slots = { ring_counts = [8, 16, 24], spin = 0.05, styles = { morpho = 2, monarch = 1, blue = 1 } }

    [[key]]                                    # out on every kick, drifting back in until the next
    target = "swarm"
    prop = "burst"
    lane = "kick"
    pulse = [[0.0, 0.0], [0.8, 1.0]]

Objects: `<name>_swarm`, a mesh of one point per butterfly carrying its seeded constants (and an `id`, so motion blur
matches every instance from frame to frame) with the geometry-nodes modifier "Swarm" (tree `<name>_swarm`) that instances
the parts on the points; the hidden part sources `<name>_<style><way>_body`, `_wing_L`, `_wing_R` of every style and
colourway in use (materials `<name>_<style><way>_wing`, `_body`, `_eye`); with `level` the empty `<name>_level` between the
root and the swarm, which a Limit Rotation (world space, x and y held at 0) keeps level while the root tilts (yaw still
follows: a swarm riding a nodding head). Drivers carry the root's properties onto the modifier's inputs and the scene
frame onto its clip second `t`: nothing runs at render time and a freeze holds them mid-beat.

Options (the `slots`) and live root properties: mkmmd.core.butterflies (DEFAULTS, PARAMS) and docs/design.md: Props,
`butterflies`; the colour roles of mkmmd.core.butterfly (ground, under, ink, spot, accent, sheen, body, eye) in the slots
recolour every style. Card: size and bounds (the rings, orbits and drift, without the burst), origin "center", front
"-Y", blocks false, `rates` (flap -> beats, spin -> turns), params, `use.look` centre, `butterflies` {count, rings, styles,
combos, objects, sources, level, inputs}."""
import bpy
import numpy as np

from ....core import butterflies as BS
from ...keys import set_fcurve
from .. import shell as SH
from . import butterflies_nodes as BN
from . import butterfly_parts as BP
from . import register
from .cafe_kit import clear_drivers

MOD = "Swarm"
TAG = "mk_butterflies"                       # custom property of every object a swarm makes: the swarm's name
TILE = 384                                   # the wing pattern's pixels per wing: a swarm's butterflies are small on screen


def _clock(root, into, rate, f0, fps):
    """A clock `into` at a steady `rate` a second: 0 at clip second 0, linear both ways (the keys stage rewrites it when
    the rate is keyed)."""
    fc = set_fcurve(root, f'["{into}"]', 0, [f0, f0 + fps], [0.0, rate], interp="LINEAR")
    fc.extrapolation = "LINEAR"


def _drive(host, ident, expr, root=None, prop=None):
    """A driver `expr` on the modifier's input `ident`, reading the root's property `prop` by its name; refused unless it
    is a Blender simple expression (no Python at render time)."""
    fc = host.driver_add(f'modifiers["{MOD}"]["{ident}"]')
    d = fc.driver
    d.type = "SCRIPTED"
    d.expression = expr
    if prop is not None:
        v = d.variables.new()
        v.name, v.type = prop, "SINGLE_PROP"
        v.targets[0].id_type, v.targets[0].id = "OBJECT", root
        v.targets[0].data_path = f'["{prop}"]'
    if not (d.is_valid and d.is_simple_expression):
        raise ValueError(f"butterflies: the driver on {host.name}'s {ident} is not a Blender simple expression ({expr})")


def _object(name, data, coll, parent, swarm):
    """An object of the swarm (the old ones were purged by their tag: a purge by name here would take the fresh mesh of
    the same name with it)."""
    o = bpy.data.objects.new(name, data)
    coll.objects.link(o)
    o.parent = parent
    o[TAG] = swarm
    return o


def _points(name, L, pick):
    """The swarm's mesh: one vertex per butterfly where it sits on its ring at turns 0, with every attribute the tree
    reads (ATTRS), its instance index `pick` and its `id`."""
    old = bpy.data.meshes.get(name)
    if old is not None and old.users == 0:
        bpy.data.meshes.remove(old)
    me = bpy.data.meshes.new(name)
    me.vertices.add(L.N)
    me.vertices.foreach_set("co", L.rest().astype(np.float32).ravel())
    for k, kind in BS.ATTRS.items():
        a = me.attributes.new(k, kind, "POINT")
        a.data.foreach_set("vector" if kind == "FLOAT_VECTOR" else "value", L.attrs[k].astype(np.float32).ravel())
    me.attributes.new("pick", "INT", "POINT").data.foreach_set("value", np.asarray(pick)[L.combo].astype(np.int32))
    me.attributes.new("id", "INT", "POINT").data.foreach_set("value", np.arange(L.N, dtype=np.int32))
    me.update()
    return me


def _level(name, coll, root):
    """The empty that carries a level swarm: world-space Limit Rotation holding x and y at 0."""
    e = _object(f"{name}_level", None, coll, root, name)
    e.empty_display_type, e.empty_display_size = "PLAIN_AXES", 0.1
    c = e.constraints.new("LIMIT_ROTATION")
    c.owner_space = "WORLD"
    c.use_limit_x = c.use_limit_y = True
    c.min_x = c.max_x = c.min_y = c.max_y = 0.0
    return e


@register("butterflies")
def butterflies(name, coll, root, slots=None):
    slots = dict(slots or {})
    opts, P = BS.options(slots)
    L = BS.layout(opts)
    sc = bpy.context.scene
    f0, fps = float(sc.get("mk_frame0", sc.frame_start)), sc.render.fps / sc.render.fps_base
    for old in [o.name for o in bpy.data.objects if o.get(TAG) == name]:     # a rebuild in the same session
        SH.purge(old)
    clear_drivers(root)
    for k, (_d, unit, doc) in BS.PARAMS.items():
        root[k] = P[k]
        root.id_properties_ui(k).update(description=f"{doc}" + (f" ({unit})" if unit else ""), default=P[k])
    for into, doc in BS.CLOCKS.items():
        root[into] = 0.0
        root.id_properties_ui(into).update(description=doc)
    for r in BS.RATES:
        _clock(root, r["into"], P[r["rate"]], f0, fps)
    sources = {part: [] for part in BS.PARTS}
    for style, way in L.combos:
        prefix = f"{name}_{style}{way}"
        meshes = BP.part_meshes(prefix, style, slots, way, tile=TILE)
        for part in BS.PARTS:
            src = _object(f"{prefix}_{part}", meshes[part], coll, root, name)
            src.hide_render = src.hide_viewport = True
            sources[part].append(src)
    tree, idents, pick = BN.swarm_tree(f"{name}_swarm", L, sources)
    parent = _level(name, coll, root) if opts["level"] else root
    host = _object(f"{name}_swarm", _points(f"{name}_swarm", L, pick), coll, parent, name)
    mod = host.modifiers.new(MOD, "NODES")
    mod.node_group = tree
    values = dict(P, t=0.0, beats=0.0, turns=0.0)
    for k, ident in idents.items():
        mod[ident] = float(values[k])
        if k == "t":
            _drive(host, ident, f"(frame - {f0!r}) / {float(fps)!r}")
        else:
            _drive(host, ident, k, root, k)
    lo, hi = BS.bounds(opts, P, L)
    return {"size": [round(h - lo_, 4) for lo_, h in zip(lo, hi)], "origin": "center", "front": "-Y", "blocks": False,
            "bounds": {"min": [round(v, 4) for v in lo], "max": [round(v, 4) for v in hi]}, "layers": [],
            "use": {"look": [{"name": "centre", "point": [0.0, 0.0, 0.0]}]},
            "rates": [dict(r) for r in BS.RATES], "params": P,
            "butterflies": {"count": L.N, "rings": L.counts, "styles": {st: L.style.count(st) for st in sorted(set(L.style))},
                            "combos": [[st, w] for st, w in L.combos], "objects": [host.name],
                            "sources": {part: [o.name for o in obs] for part, obs in sources.items()},
                            "level": opts["level"], "inputs": idents}}
