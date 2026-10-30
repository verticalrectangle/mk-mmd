"""Strings that ring (docs/design.md: Playing a worn guitar, Strings that ring): the perform stage keys the strings of a worn
guitar-like prop as its pick hand plays them. The maths is mkmmd.core.strings (numpy, tested); this is the Blender side.

Each string the card names (`use.grip` neck `strings[i].object`, its own mesh with a ring of vertices at every wire and press
point, as the library props build them) gets shape keys for what it does in the clip: `press_<fret>` (held down behind that
wire: straight from the nut to the finger, along the crown, up to the saddle) and `ring_<fret>` (vibrating from that wire,
or the nut for 0, to the saddle: a spindle of the string's swing, its key value the amplitude over REF), keyed on the
frames they move (LINEAR), and the custom properties its material reads: `mk_amp` (m, the blur's half-width at the
antinode), `mk_u0` (where the vibrating length starts, 0 at the nut, 1 at the saddle), `mk_phase` (the shimmer) and the
constant `mk_r` (its radius); the vertex attribute `mk_u` holds every vertex's place along it. The strings' materials get a
coverage mix (`ring_material`): a still string is opaque; a ringing one is a translucent band, its middle as covered as a
string swept over its blur is there (2 r / (pi a), a = its amplitude at that place), denser toward its edges where the string
turns (a rim on the facing ratio |N.I|), with a faint shimmer along it, drawn blended (sorted, no dither grain). Flat looks
(silhouette, vector, mk's renderer) and pixel maps see the spindle as geometry, opaque."""
import math

import bpy
import numpy as np

from ...core import fretting as FR
from ...core import strings as CS
from .. import keys as K
from . import BuildError

REF = 0.004                       # m: the amplitude a `ring_<fret>` key's value 1 shows (its slider goes to SLIDER)
SLIDER = 4.0
NODE = "mk_ring"                  # name prefix of the nodes `ring_material` adds (and the mark that it did)
RIM = (0.30, 0.92)               # 1 - |N.I| from which the band's edges get denser, and where they are densest
RIM_GAIN = 0.85                  # ... how much of the way to opaque they get there


def _string_objects(prop, neck, where):
    """{string number: object} of the card's neck strings that name an object of the scene."""
    n = len(neck["strings"])
    out = {}
    for i, st in enumerate(neck["strings"]):
        name = st.get("object")
        if not name:
            continue
        ob = bpy.data.objects.get(name)
        if ob is None or ob.type != "MESH":
            raise BuildError(f"{where}: the card's string {n - i} names {name!r}, which is not a mesh of the scene")
        out[n - i] = ob
    return out


def _centres(neck, entry):
    """{string: its place on the strum axis (m from the strum centre along `across`) at the strum centre}."""
    a, c = (np.asarray(entry[k], float) for k in ("along", "across"))
    a, c = a / np.linalg.norm(a), c / np.linalg.norm(c)
    centre = np.asarray(entry["center"], float)
    s_c = float(centre @ a)
    return {s: float((FR.string_point(neck, s, s_c) - centre) @ c) for s in range(1, len(neck["strings"]) + 1)}


def _fretting(m, prop):
    """(states, press) of the cast member's fretting hand on `prop`'s neck (the pose stage's `m.necks`), or ([], PRESS)."""
    for info in m.necks.values():
        if info.get("prop") == prop.name:
            return list(info["states"]), float(info.get("press", FR.PRESS))
    return [], FR.PRESS


def _keyed(values):
    """Indices of the frames a curve needs keys on (LINEAR): every frame it is not 0 and the 0 just before and after each
    such run (the first and the last frame always, so the curve is pinned)."""
    v = np.asarray(values, float)
    on = v != 0.0
    near = on | np.r_[on[1:], False] | np.r_[False, on[:-1]]
    near[0] = near[-1] = True
    return np.where(near)[0]


def _key(id_data, path, frames, values, interp="LINEAR"):
    idx = _keyed(values)
    K.set_fcurve(id_data, path, 0, np.asarray(frames, float)[idx], np.asarray(values, float)[idx], interp=interp)
    return len(idx)


def ring_material(mat):
    """Put the coverage mix into a string material once (see the module docstring): alpha = 1 for a still string or any
    object without `mk_amp`, else c + (1 - c) RIM_GAIN rim on its front faces (0 on its back faces): c = min(1, 2 r / (pi
    a)) the middle of the band, a = mk_amp p(u), p(u) = sin(pi (u - u0) / (1 - u0)) on u0..1, rim = smoothstep of 1 - |N.I|
    over RIM (the edges, where the swing turns), times a faint shimmer (noise along u, phased by mk_phase); Transparent
    mixed in by 1 - alpha."""
    if not mat or not mat.use_nodes or mat.node_tree is None:
        return False
    nt = mat.node_tree
    if f"{NODE}_mix" in nt.nodes:
        return False
    outs = [n for n in nt.nodes if n.bl_idname == "ShaderNodeOutputMaterial"]
    out = next((n for n in outs if n.is_active_output), outs[0] if outs else None)
    if out is None or not out.inputs["Surface"].links:
        return False
    surface = out.inputs["Surface"].links[0].from_socket
    x0, y0 = out.location.x - 900.0, out.location.y - 500.0
    made = []

    def node(kind, name, dx, dy, **props):
        n = nt.nodes.new(kind)
        n.name = n.label = f"{NODE}_{name}"
        n.location = (x0 + dx, y0 + dy)
        for k, v in props.items():
            setattr(n, k, v)
        made.append(n)
        return n

    def attr(name, kind, dy):
        return node("ShaderNodeAttribute", name, 0.0, dy, attribute_type=kind, attribute_name=name).outputs["Fac"]

    def math_(op, a, b=None, dx=0.0, dy=0.0, clamp=False, c=None):
        n = node("ShaderNodeMath", f"{op.lower()}_{len(made)}", dx, dy, operation=op, use_clamp=clamp)
        for i, v in enumerate((a, b, c)):
            if v is None:
                continue
            if isinstance(v, (int, float)):
                n.inputs[i].default_value = float(v)
            else:
                nt.links.new(v, n.inputs[i])
        return n.outputs[0]

    u, u0 = attr("mk_u", "GEOMETRY", 0.0), attr("mk_u0", "OBJECT", -160.0)
    amp, r0, ph = attr("mk_amp", "OBJECT", -320.0), attr("mk_r", "OBJECT", -480.0), attr("mk_phase", "OBJECT", -640.0)
    s = math_("DIVIDE", math_("SUBTRACT", u, u0, 200, 0), math_("MAXIMUM", math_("SUBTRACT", 1.0, u0, 200, -160), 1e-4, 380, -160),
              380, 0)
    inside = math_("MULTIPLY", math_("GREATER_THAN", s, -1e-6, 560, 80), math_("LESS_THAN", s, 1.000001, 560, -80), 740, 0)
    p = math_("MULTIPLY", math_("SINE", math_("MULTIPLY", math_("MINIMUM", math_("MAXIMUM", s, 0.0, 560, 240), 1.0, 740, 240),
                                              math.pi, 920, 240)), inside, 1100, 120)
    a = math_("MULTIPLY", amp, p, 1280, 0)
    geo = node("ShaderNodeNewGeometry", "geometry", 900.0, -500.0)
    dot = node("ShaderNodeVectorMath", "facing", 1100.0, -500.0, operation="DOT_PRODUCT")
    nt.links.new(geo.outputs["Normal"], dot.inputs[0])
    nt.links.new(geo.outputs["Incoming"], dot.inputs[1])
    edge = math_("SUBTRACT", 1.0, math_("ABSOLUTE", dot.outputs["Value"], None, 1280, -500), 1460, -500)
    rim = node("ShaderNodeMapRange", "rim", 1640.0, -500.0, interpolation_type="SMOOTHSTEP")
    nt.links.new(edge, rim.inputs["Value"])
    rim.inputs["From Min"].default_value, rim.inputs["From Max"].default_value = RIM
    centre = math_("DIVIDE", math_("MULTIPLY", r0, 2.0 / math.pi, 1460, -320), math_("ADD", a, 1e-7, 1460, -160), 1640, -240,
                   clamp=True)                                              # the band's middle: 2 r / (pi a)
    cov = math_("MULTIPLY_ADD", math_("MULTIPLY", math_("SUBTRACT", 1.0, centre, 1820, -400), rim.outputs["Result"], 2000,
                                      -400), RIM_GAIN, 2180, -320, c=centre)   # denser toward the edges, where it turns
    noise = node("ShaderNodeTexNoise", "shimmer", 1640.0, -760.0, noise_dimensions="4D")
    noise.inputs["Scale"].default_value = 1.0
    noise.inputs["Detail"].default_value = 1.0
    vec = node("ShaderNodeCombineXYZ", "shimmer_at", 1460.0, -760.0)
    nt.links.new(math_("MULTIPLY", u, 40.0, 1280, -760), vec.inputs["X"])
    nt.links.new(vec.outputs[0], noise.inputs["Vector"])
    nt.links.new(ph, noise.inputs["W"])
    shim = math_("MULTIPLY_ADD", noise.outputs["Fac"], 0.5, 2000, -600, c=0.75)      # about 0.85 .. 1.15
    still = math_("LESS_THAN", amp, 1e-6, 2000, 160)
    seen = math_("MULTIPLY", math_("MULTIPLY", cov, shim, 2360, -400, clamp=True),
                 math_("SUBTRACT", 1.0, geo.outputs["Backfacing"], 2180, -700), 2540, -400)
    alpha = math_("MAXIMUM", still, seen, 2720, -200)
    clear = node("ShaderNodeBsdfTransparent", "clear", 2720.0, 120.0)
    mix = node("ShaderNodeMixShader", "mix", 2900.0, 0.0)
    nt.links.new(alpha, mix.inputs["Fac"])
    nt.links.new(clear.outputs[0], mix.inputs[1])
    nt.links.new(surface, mix.inputs[2])
    nt.links.new(mix.outputs[0], out.inputs["Surface"])
    if hasattr(mat, "surface_render_method"):                   # sorted, smooth translucency (no dither grain)
        mat.surface_render_method = "BLENDED"
    if hasattr(mat, "use_transparent_shadow"):
        mat.use_transparent_shadow = True
    return True


def _shape(ob, name, deltas_local):
    """The shape key `name` of `ob` (made from the basis when missing) at basis + deltas (object space)."""
    if ob.data.shape_keys is None:
        ob.shape_key_add(name="Basis", from_mix=False)
    kb = ob.data.shape_keys.key_blocks.get(name) or ob.shape_key_add(name=name, from_mix=False)
    basis = np.empty(3 * len(ob.data.vertices))
    ob.data.shape_keys.key_blocks[0].data.foreach_get("co", basis)
    kb.data.foreach_set("co", (basis.reshape(-1, 3) + deltas_local).ravel())
    kb.slider_min, kb.slider_max = 0.0, SLIDER
    kb.value = 0.0
    return kb


def play(ctx, name, m, side, prop, strokes, keep, sigma, entry, neck):
    """Key the strings of `prop` that the pick hand `side` of cast member `name` plays with `strokes` (its strum plan:
    mkmmd.core.strum.plan, `keep` the path numbers, `sigma` half the strings' width, `entry` the strum point, `neck` the
    card's neck entry): the presses of its fretting hand (the pose stage's neck states), each string's hits and rings
    (mkmmd.core.strings), the shape keys and properties above on every frame of the build. Returns the report's numbers
    (None when the strum spec says `ring = false` or the card names no string object)."""
    where = f"perform.{name}.strum ({side})"
    if not strokes or neck is None:
        return None
    try:
        params = CS.ring_params(next((s for s in _specs(ctx, name) if s.get("hand", "R") == side), {}), neck.get("scale", 0.65))
    except CS.StringsError as e:
        raise BuildError(f"{where}: {e}") from None
    if params is None:
        return None
    objs = _string_objects(prop, neck, where)
    if not objs:
        ctx.log("WARNING", f"{where}: the card of {prop.name!r} names no string objects (neck strings[].object): no string rings")
        return None
    n = len(neck["strings"])
    states, press = _fretting(m, prop)
    try:
        pressed = CS.presses(states, n) if states else []
        picking = any(s.string is not None for s in strokes)
        hit = CS.hits(strokes, _centres(neck, entry), sigma, keep, None if picking else CS.sounding(states, n))
        rung = CS.rings(pressed, hit, mono=picking)
    except (CS.StringsError, FR.FrettingError) as e:
        raise BuildError(f"{where}: {e}") from None
    frames = np.asarray(ctx.frames)
    ts = np.array([ctx.time(f) for f in frames])
    shown = CS.frames(rung, pressed, ts, params, lambda s, f: CS.fret_u(neck, s, f), n)
    root_inv = np.array(prop.root.matrix_world.inverted())
    report, nkeys = {}, 0
    for s, ob in sorted(objs.items()):
        M = root_inv @ np.array(ob.matrix_world)                       # the string's own space -> the prop's frame (rigid)
        R, o = M[:3, :3], M[:3, 3]
        V = np.empty(3 * len(ob.data.vertices))
        ob.data.vertices.foreach_get("co", V)
        P = V.reshape(-1, 3) @ R.T + o
        Rinv = np.linalg.inv(R)
        u = CS.u_of(neck, s, P)
        at = ob.data.attributes.get("mk_u") or ob.data.attributes.new("mk_u", "FLOAT", "POINT")
        at.data.foreach_set("value", u.astype(np.float32))
        ob["mk_r"] = float(FR.string_entry(neck, s).get("radius", 0.0006))
        d = shown[s]
        for fret, v in sorted(d["press"].items()):
            _shape(ob, f"press_{fret}", CS.press_offsets(neck, s, P, fret, press) @ Rinv.T)
            nkeys += _key(ob.data.shape_keys, f'key_blocks["press_{fret}"].value', frames, v)
        for fret in sorted(set(int(f) for f in d["fret"][d["amp"] > 0])):
            _shape(ob, f"ring_{fret}", CS.ring_offsets(neck, s, P, CS.fret_u(neck, s, fret), REF) @ Rinv.T)
            v = np.where((d["fret"] == fret) & (d["amp"] > 0), d["amp"] / REF, 0.0)
            nkeys += _key(ob.data.shape_keys, f'key_blocks["ring_{fret}"].value', frames, v)
        for key, v, interp in (("mk_amp", d["amp"], "LINEAR"), ("mk_u0", np.where(d["amp"] > 0, d["u0"], 0.0), "CONSTANT"),
                               ("mk_phase", np.where(d["amp"] > 0, d["phase"], 0.0), "CONSTANT")):
            ob[key] = 0.0
            nkeys += _key(ob, f'["{key}"]', frames, v, interp)
        for mat in {sl.material for sl in ob.material_slots if sl.material is not None}:
            ring_material(mat)
        mine = [r for r in rung if r[2] == s]
        if mine or d["press"]:
            report[s] = {"rings": len(mine), "frets": sorted({r[3] for r in mine}), "held": sorted(d["press"])}
    return {"played": report, "hits": len(hit), "keys": nkeys, "amp_mm": round(params["amp"] * 1e3, 2)}


def _specs(ctx, name):
    spec = (ctx.section("perform", name) or {}).get("strum")
    return [spec] if isinstance(spec, dict) else list(spec or [])
