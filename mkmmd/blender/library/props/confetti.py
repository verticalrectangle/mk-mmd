"""confetti: paper confetti and streamers fired from the root in volleys on the hits: gently curled squares, hearts and
stars (flat paper die-cuts) and curly streamers, flying out in a cone, slowed by the air, tumbling and fluttering as they
rain down, then shrinking away; drivers on the scene frame (the maths is mkmmd/core/confetti.py), so a freeze holds them
mid-air.

    [[prop]]
    name = "pop"
    card = "library:confetti"
    at = [0.0, 0.6, 1.0]                  # the mouth; the pieces fly up, a little toward the camera (-Y), and fall
    slots = { count = 60, volleys = 3, floor = -1.0, colors = ["love", "gold", "foam", "iris"] }

    [[key]]                               # a volley on every accent, dealt round the three volleys in turn
    target = "pop"
    prop = "start"
    lane = "accent"

Objects `<name>_<kind><i>` (kinds square, ribbon, heart, star; one mesh per kind and colour, `<name>_<kind>_<c>`, in the
matte paper `<name>_paper<c>` of colour c), every one tagged form-exempt (paper is flat by nature). Options (the `slots`):
count (60 per volley, up to 200), volleys (3, 1 .. 4), seed (1), mix ({kind: share}), colors (palette slots, '#hex' or
'a:b:t' mixes; love, gold, foam, iris, rose, pine, text), aim ([0, -0.25, 1]), cone (24 degrees), ribbon (0.5 m: a
streamer's length), glow (0.25), and the starting values of the live root properties start, reach, drag, fall, flutter,
sway, spin, size, life, floor, amount (core.confetti.PARAMS), which `[[key]]` keys. Volley k reads `start<k>`; the card's
`deals` has the keys stage deal the keys of `start` round them (unkeyed: start0 = start, the others never). Card keys:
size, bounds (the whole flight), origin "center", front "-Y", blocks false, params, deals, `confetti` {objects, volleys,
count, fps}."""
import re

import bpy

from ....core import confetti as CF
from .. import shell as SH
from . import register
from . import showkit as SK
from .cafe_kit import clear_drivers

NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
CLOCK = {"f0": "the frame of clip second 0", "fps": "frames per second"}


def _doc(k):
    if k in CF.PARAMS:
        _, unit, doc = CF.PARAMS[k]
        return doc + (f" ({unit})" if unit else "")
    if k in CLOCK:
        return CLOCK[k]
    return (f"the clip second volley {k[5:]} goes off (s): the keys stage deals the keys of `start` here; "
            f"{CF.NEVER:g}: never")


def _driver(owner, path, expr, root, live, index=-1):
    """A scripted driver `expr` on `owner[path]` (a piece): the root's properties `live` and the piece's helpers, by
    name, are its variables."""
    res = owner.driver_add(path, index) if index >= 0 else owner.driver_add(path)
    names = sorted(set(NAME.findall(expr)) - CF.FUNCS - {"frame"})
    for fc in res if isinstance(res, list) else [res]:
        d = fc.driver
        d.type = "SCRIPTED"
        d.expression = expr
        for nm in names:
            if nm not in live and nm not in CF.HELPERS:
                raise ValueError(f"confetti: expression for {path} uses {nm!r}, which is neither a root property nor a "
                                 f"helper")
            v = d.variables.new()
            v.name, v.type = nm, "SINGLE_PROP"
            v.targets[0].id_type = "OBJECT"
            v.targets[0].id = root if nm in live else owner
            v.targets[0].data_path = f'["{nm}"]'
        if not (d.is_valid and d.is_simple_expression):
            raise ValueError(f"confetti: the driver on {path} is not a Blender simple expression ({expr})")


def _paper(name, rgb, glow):
    """Matte paper of colour `rgb`, alike on both sides, with a small glow of its own so the colour reads at night: rough
    and dielectric, so a tumbling piece only ever glints."""
    m = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    m.use_nodes = True
    b = m.node_tree.nodes.get("Principled BSDF")
    b.inputs["Base Color"].default_value = (*rgb, 1.0)
    b.inputs["Roughness"].default_value = 0.62
    b.inputs["Specular IOR Level"].default_value = 0.3
    b.inputs["Emission Color"].default_value = (*rgb, 1.0)
    b.inputs["Emission Strength"].default_value = glow
    m.diffuse_color = (*rgb, 1.0)
    m.use_backface_culling = False
    return m


@register("confetti")
def confetti(name, coll, root, slots=None):
    slots = dict(slots or {})
    O = CF.options(slots)
    rgbs = [SK.rgb(spec, slots, f"confetti: colors[{i}]") for i, spec in enumerate(O.colors)]
    sc = bpy.context.scene
    fps = sc.render.fps / sc.render.fps_base
    for old in [o for o in bpy.data.objects if any(o.name.startswith(f"{name}_{k}") and o.name[len(name) + len(k) + 1:]
                                                     .isdigit() for k in CF.KINDS)]:
        SH.purge(old.name)
    clear_drivers(root)
    P = CF.root_values(O)
    values = dict(P, f0=float(sc.get("mk_frame0", sc.frame_start)), fps=float(fps))
    for k, v in values.items():
        root[k] = float(v)
        root.id_properties_ui(k).update(description=_doc(k))
    for k, v, doc in (("count", O.count, "pieces per volley (fixed at build)"),
                      ("volleys", O.volleys, "volleys, each reading its own start<k> (fixed at build)"),
                      ("seed", O.seed, "the layout's seed (fixed at build)")):
        root[k] = v
        root.id_properties_ui(k).update(description=doc)
    pieces = CF.layout(O)
    mats = [_paper(f"{name}_paper{c}", rgb, O.glow) for c, rgb in enumerate(rgbs)]
    used = sorted({(p["kind"], p["colour"]) for p in pieces})
    shapes = {kind: CF.mesh(kind, CF.length(O)) for kind in {k for k, _ in used}}
    meshes = {(kind, c): SH.mesh_data(f"{name}_{kind}_{c}", shapes[kind], lambda _role, m=mats[c]: m)[0]
              for kind, c in used}
    live = set(values)
    objs, at = [], {k: 0 for k in CF.KINDS}
    for p in pieces:
        kind = p["kind"]
        o = bpy.data.objects.new(f"{name}_{kind}{at[kind]}", meshes[kind, p["colour"]])
        at[kind] += 1
        coll.objects.link(o)
        o.parent = root
        o.rotation_mode = "XYZ"
        SH.exempt(o, "paper confetti: flat by nature")
        ex = CF.expressions(p)
        for h in CF.HELPERS:
            o[h] = 0.0
        for h in CF.HELPERS:
            _driver(o, f'["{h}"]', ex[h], root, live)
        _driver(o, "location", ex["x"], root, live, 0)
        _driver(o, "location", ex["y"], root, live, 1)
        _driver(o, "location", "z", root, live, 2)
        for i, ch in enumerate(("rx", "ry", "rz")):
            _driver(o, "rotation_euler", ex[ch], root, live, i)
        for i, ch in enumerate(("s", "s", "sz" if "sz" in ex else "s")):
            _driver(o, "scale", ex[ch], root, live, i)
        objs.append(o)
    lo, hi = CF.bounds(O, pieces)
    return {"size": [round(float(h - l), 4) for l, h in zip(lo, hi)], "origin": "center", "front": "-Y", "blocks": False,
            "bounds": {"min": [round(float(v), 4) for v in lo], "max": [round(float(v), 4) for v in hi]}, "layers": [],
            "slots": {"colors": list(O.colors)}, "params": P, "deals": CF.deals(O.volleys),
            "confetti": {"objects": [o.name for o in objs], "volleys": O.volleys, "count": O.count, "fps": fps}}
