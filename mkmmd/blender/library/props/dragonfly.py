"""dragonfly: a paper dragonfly on a steel wire, a toy theatre's rod puppet, 0.45 m across (the maths is
mkmmd/core/dragonfly.py; a `[[flight]]` flies it relative to a target and draws its wire: mkmmd/core/flight.py,
mkmmd/blender/build/flight.py). Drivers on the root's properties turn the wings: nothing runs at render time.

    [[prop]]
    name = "dfly"
    card = "library:dragonfly"
    at = [0.45, -0.35, 1.5]
    slots = { step = 12, step_from = 3.03 }       # its wings and its flight held on the set's stepped grid

Options (the `slots`): size (the span, 0.45 m), whirr (cycles a pose the wings advance, 0.41), amp (the stroke, 70
degrees), mid (its middle above flat, 10), lag (the hindwings behind the fore, 0.5 of a cycle), droop (the wings below
flat at rest, 4), step / step_from (poses a second and the grid's first clip second, as the paper theatre's: 0 every
frame), wire (its radius, 0.002 m), hand ([x, y, z]: the wire's far end in the prop's frame until a flight keys it),
film (the share of ink in the wings' film, 0.5), clear_wing (the cellophane's opacity lit, 0.42), smile (true), inks (the
film's colours, foam iris rose gold) and the colour roles of mkmmd.core.dragonfly.ROLES. Root properties for [[key]]:
phase (cycles, keyed by the builder: whirr a pose), rest (0 flying, 1 perched), amp, mid, lag, droop.

Objects `<name>_body` (the root's child, scaled to the span), `<name>_wing_fore_L` .. `_hind_R` (the body's children,
turning about their hinges), `<name>_wire` (a bevelled poly curve) and `<name>_hand` (an empty at its far end); materials
`<name>_<role>`. The cellophane is see-through lit (blended, its colour turning through the inks with the angle it is
seen at, as a thin film's does) and opaque to a flat look, which draws its film picture (a cel tone, `texture =
true`)."""
import bpy
import numpy as np

from ....core import dragonfly as DF
from ....core import flight as FL
from ....core import paper_theatre as PT
from ...keys import set_fcurve
from .. import shell as SH
from ..sets import nightkit as NK
from . import register
from . import showkit as SK
from .cafe_kit import clear_drivers

PAPER = {"straw": 0.06, "stripe": 0.06, "thorax": 0.08, "face": 0.07, "legs": 0.04, "vein": 0.05, "spot": 0.05}


def _rgba(c, a=1.0):
    return (*c[:3], a)


def paper(name, c, mottle=0.07):
    """Construction paper on a round form: the colour mottled by a cloud and flecked with fibres, a fine tooth in its
    bump, a soft sheen (the paper theatre's paper without its cut-edge colour, which looks for flat pieces)."""
    m, nb = NK.new_material(name)
    P = nb.texcoord("Object")
    cloud = nb.math("SUBTRACT", nb.noise(P, scale=22.0, detail=4.0, rough=0.6), 0.5)
    fibre = nb.smooth(0.62, 0.70, nb.noise(P, scale=260.0, detail=3.0, rough=0.55))
    k = nb.math("ADD", nb.math("MULTIPLY", cloud, 2.0 * mottle), nb.math("ADD", nb.math("MULTIPLY", fibre, 0.08), 1.0))
    col = nb.vmath("SCALE", _rgba(c), scale=k)
    bump = nb.node("ShaderNodeBump", {"Strength": 0.15, "Distance": 0.0004,
                                      "Height": nb.noise(P, scale=900.0, detail=4.0, rough=0.6)})
    b = nb.principled(col, rough=0.85, spec=0.25, normal=bump.outputs["Normal"])
    nb.set(b.inputs["Sheen Weight"], 0.3)
    nb.set(b.inputs["Sheen Roughness"], 0.5)
    nb.output(b)
    m.diffuse_color = _rgba(c)
    return m


def _gloss(name, c, glow=0.0, rough=0.18):
    """A googly eye's plastic: glossy under a clear coat."""
    m, nb = NK.new_material(name)
    b = nb.principled(_rgba(c), rough=rough, spec=0.5, coat=1.0, coat_rough=0.04, emission=_rgba(c), strength=glow)
    nb.output(b)
    m.diffuse_color = _rgba(c)
    return m


def _steel(name, c):
    m, nb = NK.new_material(name)
    b = nb.principled(_rgba(c), rough=0.32, metal=1.0, emission=_rgba(c), strength=0.08)
    nb.output(b)
    m.diffuse_color = _rgba(c)
    return m


def _film_image(name, inks, base, strength, variant=0):
    """The film's picture as a packed sRGB image (row 0 of mkmmd.core.dragonfly.film is its top)."""
    old = bpy.data.images.get(name)
    if old is not None:
        bpy.data.images.remove(old)
    pic = np.clip(DF.film(inks, base, strength, variant=variant), 0.0, 1.0)
    h, w = pic.shape[:2]
    srgb = np.where(pic <= 0.0031308, pic * 12.92, 1.055 * pic ** (1.0 / 2.4) - 0.055)      # as SS.linear_to_srgb
    px = np.concatenate([srgb, np.ones((h, w, 1))], axis=2)[::-1]                # Blender's rows run up
    img = bpy.data.images.new(name, w, h, alpha=False)
    img.colorspace_settings.name = "sRGB"
    img.pixels.foreach_set(np.ascontiguousarray(px, np.float32).ravel())
    img.pack()
    return img


def _cellophane(name, image, inks, base, clear):
    """The wings' cellophane: lit, see-through (blended) with a clear glint, its tint the film's picture turned
    through the inks by the angle it is seen at (a thin film's colours move as the wing whirrs); its viewport colour
    opaque, so a flat look draws it, with the picture (the active image node) as a cel tone's texture."""
    m, nb = NK.new_material(name)
    m.surface_render_method = "BLENDED"
    m.use_backface_culling = False
    m.show_transparent_back = True
    tex = nb.node("ShaderNodeTexImage", {0: nb.node("ShaderNodeUVMap", uv_map="UVMap").outputs[0]}, image=image,
                  interpolation="Linear", extension="EXTEND")
    nb.nt.nodes.active = tex
    lw = nb.node("ShaderNodeLayerWeight", {"Blend": 0.45})
    u = nb.sep(nb.node("ShaderNodeUVMap", uv_map="UVMap").outputs[0])[0]
    s = nb.math("FRACT", nb.math("ADD", nb.math("MULTIPLY", lw.outputs["Facing"], 1.4), nb.math("MULTIPLY", u, 0.55)))
    stops = [(i / len(inks), c) for i, c in enumerate(inks)] + [(1.0, inks[0])]
    turn = nb.ramp(s, [(p, _rgba(c)) for p, c in stops])
    col = nb.mixc(0.45, tex.outputs["Color"], turn)
    alpha = nb.math("ADD", clear, nb.math("MULTIPLY", lw.outputs["Fresnel"], 0.35), clamp=True)
    b = nb.principled(col, rough=0.12, spec=0.7, emission=col, strength=0.12, alpha=alpha)
    nb.output(b)
    m.diffuse_color = _rgba(base)
    return m


def _driver(owner, path, index, expr, root):
    """A scripted driver `expr` on owner.path[index] reading the root's properties; refused unless Blender keeps it a
    simple expression (docs/AGENTS.md: Pitfalls)."""
    fc = owner.driver_add(path, index)
    d = fc.driver
    d.type = "SCRIPTED"
    d.expression = expr
    for nm in DF.ROOT_PROPS:
        if nm in expr:
            v = d.variables.new()
            v.name, v.type = nm, "SINGLE_PROP"
            v.targets[0].id_type, v.targets[0].id = "OBJECT", root
            v.targets[0].data_path = f'["{nm}"]'
    if len(expr) > 255 or not (d.is_valid and d.is_simple_expression):
        raise ValueError(f"dragonfly: the driver on {owner.name}.{path}[{index}] is not a Blender simple expression")
    return fc


def key_films(variants, frame_start, frame_end, frame0, fps, step, step_from, hold=None):
    """Show one film variant of every wing a pose (pose k: variant k % FILMS; `variants` [[object, ...] per variant]),
    keyed CONSTANT on every frame or on the step grid; `hold` (n,) bool over every frame keeps the variant it had when
    the hold began (a perched dragonfly's wings stop shimmering)."""
    keys, times, _interp = PT.key_times(frame_start, frame_end, frame0, fps, step, step_from)
    shown = np.round(frame0 + times * fps)
    pose = DF.poses(shown, frame0, fps, step, step_from)
    if hold is not None:
        idx = np.clip((shown - frame_start).astype(int), 0, len(hold) - 1)
        held = np.asarray(hold, bool)[idx]
        for i in range(1, len(pose)):
            if held[i]:
                pose[i] = pose[i - 1]
    which = np.mod(pose, DF.FILMS)
    for v, objs in enumerate(variants):
        hidden = (which != v).astype(float)
        for o in objs:
            for attr in ("hide_render", "hide_viewport"):
                setattr(o, attr, bool(hidden[0]))
                set_fcurve(o, attr, 0, keys, hidden, interp="CONSTANT")


def key_phase(root, O, frame_start, frame_end, frame0, fps, step, step_from):
    """Key the root's `phase` (whirr cycles a pose) over the frames, on every frame (two keys, linear on) or held on the
    step grid (CONSTANT, as the paper theatre's)."""
    if step <= 0.0:
        fc = set_fcurve(root, '["phase"]', 0, [frame0, frame0 + 1.0], [0.0, O["whirr"]], interp="LINEAR")
        fc.extrapolation = "LINEAR"
        return
    keys, times, interp = PT.key_times(frame_start, frame_end, frame0, fps, step, step_from)
    shown = np.round(frame0 + times * fps)
    set_fcurve(root, '["phase"]', 0, keys, DF.poses(shown, frame0, fps, step, step_from) * O["whirr"], interp=interp)


def _wire_curve(name, coll, root, mat, radius, pts):
    SH.purge(name)
    cu = bpy.data.curves.get(name)
    if cu is not None:
        bpy.data.curves.remove(cu)
    cu = bpy.data.curves.new(name, "CURVE")
    cu.dimensions = "3D"
    cu.bevel_depth = radius
    cu.bevel_resolution = 2
    cu.use_fill_caps = True
    sp = cu.splines.new("POLY")
    sp.points.add(len(pts) - 1)
    for p, c in zip(sp.points, pts):
        p.co = (*c, 1.0)
    cu.materials.append(mat)
    o = bpy.data.objects.new(name, cu)
    coll.objects.link(o)
    o.parent = root
    return o


def _object(name, mesh, coll, parent, mat_of, loc=(0.0, 0.0, 0.0)):
    SH.purge(name)                                       # a rebuild in the same session: the old object and mesh go
    me, _ = SH.mesh_data(name, mesh, mat_of)
    o = bpy.data.objects.new(name, me)
    coll.objects.link(o)
    o.parent = parent
    o.location = loc
    o.rotation_mode = "XYZ"
    return o


@register("dragonfly")
def dragonfly(name, coll, root, slots=None):
    slots = dict(slots or {})
    try:
        O = DF.options(slots)
        roles = SK.roles(slots, DF.ROLES, "dragonfly")
        col = {k: SK.rgb(v, slots, f"dragonfly: {k}") for k, v in roles.items()}
        inks = [SK.rgb(c, slots, "dragonfly: inks") for c in O["inks"]]
    except ValueError as e:
        raise ValueError(str(e)) from None
    sc = bpy.context.scene
    f0, fps = float(sc.get("mk_frame0", sc.frame_start)), sc.render.fps / sc.render.fps_base
    if O["step"] > fps:
        raise ValueError(f"dragonfly: step = {O['step']:g} poses a second is more than the scene's {fps:g} frames")
    mats = {}
    for role in DF.MATS:
        nm = f"{name}_{role}"
        if role in PAPER:
            mats[role] = paper(nm, col[role], PAPER[role])
        elif role in ("eye", "pupil"):
            mats[role] = _gloss(nm, col[role], glow=0.05 if role == "eye" else 0.0, rough=0.2 if role == "eye" else 0.12)
        elif role == "shine":
            mats[role] = _gloss(nm, col[role], glow=1.5, rough=0.1)
        else:
            wing_mats = [_cellophane(nm if v == 0 else f"{nm}_{v}",
                                     _film_image(f"{name}_film" + (f"_{v}" if v else ""), inks, col["wing"], O["film"], v),
                                     inks, col["wing"], O["clear_wing"]) for v in range(DF.FILMS)]
            mats[role] = wing_mats[0]
    wire_mat = _steel(f"{name}_wire", col["wire"])
    clear_drivers(root)
    for k in DF.ROOT_PROPS:
        root[k] = 0.0 if k in ("phase", "rest") else float(O[k])
    root.id_properties_ui("phase").update(description="the wings' whirr: cycles (keyed: whirr a pose)")
    root.id_properties_ui("rest").update(description="0 flying, 1 perched: the wings laid out flat", min=0.0, max=1.0)
    P = DF.parts(O["smile"])
    ob = DF.objects(name)

    def mat_of(r):
        return mats[DF.MATS[r]]

    body = _object(ob["body"], P["body"], coll, root, mat_of)
    body.scale = (O["size"],) * 3
    ex = DF.expressions()
    variants = [[] for _ in range(DF.FILMS)]
    for key, names in ob["variants"].items():
        wing, side = key.split("_")
        for v, oname in enumerate(names):
            w = _object(oname, P[key], coll, body, lambda r, v=v: wing_mats[v] if DF.MATS[r] == "wing" else mat_of(r),
                        DF.hinge(wing, side))
            clear_drivers(w)
            _driver(w, "rotation_euler", 1, ex[key], root)
            variants[v].append(w)
    key_phase(root, O, sc.frame_start, sc.frame_end, f0, fps, O["step"], O["step_from"])
    key_films(variants, sc.frame_start, sc.frame_end, f0, fps, O["step"], O["step_from"])
    belly = np.array(DF.BELLY) * O["size"]
    hand = np.array(O["hand"], float)
    rest = FL.wire_rest(hand[None], belly[None], np.array([[0.0, 0.0, -1.0]]), np.array([0.0]), 32)[0]
    _wire_curve(ob["wire"], coll, root, wire_mat, O["wire"], rest)
    SH.purge(ob["hand"])
    h = bpy.data.objects.new(ob["hand"], None)
    h.empty_display_size = 0.05
    coll.objects.link(h)
    h.parent = root
    h.location = tuple(hand)
    return DF.card(name, O, roles)
