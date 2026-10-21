"""The space kit (the maths and meshes are mkmmd/core/space.py, which says what each prop is):

    [[prop]]
    name = "tightrope"
    card = "library:line"
    at = [-2.0, 0.0, 0.0]
    slots = { length = 40, ticks = 0.6, bar = 4, ripples = [[3.79, 4.5, -0.04]] }   # [t, x, amp]
    [[key]]                                   # the line retracts into its +X end and shoots back out
    target = "tightrope"
    prop = "extent"
    keys = [[17.36, 1.0], [17.42, 0.0], [17.83, 0.0], [17.86, 1.0]]

    [[prop]]
    name = "bh"
    card = "library:hole"                     # a black sphere: root property `breathe` scales it
    at = [38.0, 0.0, 0.0]
    slots = { radius = 1.6 }

    [[prop]]
    name = "board"
    card = "library:round_board"              # rings `board_r<i>` turn by omega_i * `phase`; overlays `board_hot_r<i>s<j>`
    at = [38.0, 0.0, 0.0]
    slots = { inner = 2.4, width = 0.9, rings = 6, sectors = 16 }

    [[prop]]
    name = "knight1"
    card = "library:chess"
    parent = "board_r3"                       # it rides the ring
    slots = { piece = "knight", height = 2.2, color = "#F4F6FF", metal = 0.6 }

    [[prop]]
    name = "sky"
    card = "library:stars"                    # flat stars facing -Y: [x, z, size, t] in the root's frame
    at = [0.0, 3.0, 0.0]
    slots = { points = [[1.0, 2.2, 1.0, 0.0], [1.6, 2.6, 0.8, 0.38]], lines = true }

    [[prop]]
    name = "ground"
    card = "library:floor"                    # to the horizon: mirror | chess | grid
    slots = { kind = "chess", cell = 2.0, color = "text", fade = [40, 220] }

    [[prop]]
    name = "warp"
    card = "library:starfield"                # streaks along -X past the root, two tiles leapfrogging
    slots = { speed = 14, streak = 1.2, surge = [[11.0, 0.0], [12.2, 90.0]] }

    [[prop]]
    name = "accretion"
    card = "library:disc"                     # a glowing ring round the hole, its streaks orbiting (Kepler)
    at = [38.0, 0.0, 0.0]
    rot = [12.0, 0.0, 0.0]
    slots = { inner = 2.0, outer = 7.5, color = "iris", spin = 0.3 }

Objects: line `<name>_seg<k>` (k from the anchor) and `<name>_tick<j>` (tagged `mk_line`), hole `<name>_shadow` (tagged
`mk_hole` = its radius), board `<name>_r<i>` and `<name>_hot_r<i>s<j>` (hidden: a [[key]] on `hide_render` shows an
attacked square), chess `<name>_piece`, stars `<name>_star<i>` and `<name>_line<i>` (from star i to i + 1), floor
`<name>_floor`, starfield `<name>_tile0` / `<name>_tile1`, disc `<name>_disc`.
Materials `<name>_line`, `<name>_shadow`, `<name>_light` / `<name>_dark` / `<name>_hot`, `<name>_piece`, `<name>_star`,
`<name>_floor`, `<name>_disc`: a vector look's `[vector] materials` rules give them tones by these names."""
import math
import re

import bpy
import numpy as np

from ....core import shell as CS
from ....core import space as SP
from ... import keys as K
from .. import shell as SH
from ..sets import nightkit as NK
from . import register
from .cafe_kit import clear_drivers
from .hearts import _colour, _material

NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def _err(fn):
    def wrapped(name, coll, root, slots=None):
        try:
            return fn(name, coll, root, slots)
        except SP.SpaceError as e:
            raise ValueError(f"prop {name!r}: {e}") from None
    wrapped.__name__ = fn.__name__
    return wrapped


def _purge(prefix):
    for o in [o for o in bpy.data.objects if o.name.startswith(prefix)]:
        SH.purge(o.name)


def _driver(owner, path, expr, root, prop, index=-1):
    """A driver `expr` on owner[path] whose one variable is the root's custom property `prop`, by its first letter."""
    res = owner.driver_add(path, index) if index >= 0 else owner.driver_add(path)
    for fc in res if isinstance(res, list) else [res]:
        d = fc.driver
        d.type = "SCRIPTED"
        d.expression = expr
        names = sorted(set(NAME.findall(expr)) - SP.FUNCS)
        for nm in names:
            v = d.variables.new()
            v.name, v.type = nm, "SINGLE_PROP"
            v.targets[0].id_type = "OBJECT"
            v.targets[0].id = root
            v.targets[0].data_path = f'["{prop}"]'
        if not (d.is_valid and d.is_simple_expression):
            raise ValueError(f"the driver on {owner.name}.{path} is not a Blender simple expression ({expr})")


def _live(root, prop, value, doc):
    root[prop] = float(value)
    root.id_properties_ui(prop).update(description=doc)


def _flat(name, rgb, rough=1.0, metal=0.0, glow=0.0):
    """A material of one colour (diffuse colour too: what Workbench and the flat looks see)."""
    m = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    m.use_nodes = True
    b = m.node_tree.nodes.get("Principled BSDF")
    b.inputs["Base Color"].default_value = (*rgb, 1.0)
    b.inputs["Roughness"].default_value = rough
    b.inputs["Metallic"].default_value = metal
    b.inputs["Emission Color"].default_value = (*rgb, 1.0)
    b.inputs["Emission Strength"].default_value = glow
    m.diffuse_color = (*rgb, 1.0)
    return m


def _clock():
    sc = bpy.context.scene
    fps = sc.render.fps / sc.render.fps_base
    f0 = float(sc.get("mk_frame0", sc.frame_start))
    frames = np.arange(sc.frame_start, sc.frame_end + 1)
    return frames, (frames - f0) / fps


def _bounds(lo, hi):
    return {"min": [round(float(v), 4) for v in lo], "max": [round(float(v), 4) for v in hi]}


# ================================================================================================================ line
@register("line")
@_err
def line(name, coll, root, slots=None):
    o = SP.line_options(slots)
    _purge(f"{name}_seg")
    _purge(f"{name}_tick")
    clear_drivers(root)
    _live(root, "extent", 1.0, "how much of the line there is, from its anchor end (0..1)")
    mat = _material(f"{name}_line", _colour(o["color"], slots or {}), 2.0)
    L, s, r, end = float(o["length"]), float(o["seg"]), float(o["radius"]), o["anchor"] == "end"
    toward = -1.0 if end else 1.0
    segs = []
    for k, (x0, x1, a) in enumerate(SP.segments(L, s, o["anchor"])):
        length = x1 - x0
        ob = SH.mesh_object(f"{name}_seg{k}", SP.segment_mesh(length, r, toward), coll, root, lambda _r: mat,
                            loc=(x1 if end else x0, 0.0, 0.0))
        ob["mk_line"] = True
        SH.exempt(ob, "a drawn line")
        _driver(ob, "scale", SP.segment_expr(a, length, L), root, "extent", 0)
        for i in (1, 2):                                            # gone, it leaves no end cap behind
            _driver(ob, "scale", SP.segment_expr(a, min(length, 0.005), L), root, "extent", i)
        segs.append((ob, x1 if end else x0, x0 if end else x1, length))
    hi_t = float(max(o["tick"]))
    for j, (x, long) in enumerate(SP.ticks(L, float(o["ticks"]), int(o["bar"]))):
        h = float(o["tick"][1] if long else o["tick"][0])
        ob = SH.mesh_object(f"{name}_tick{j}", SP.tick_mesh(h, float(o["tick_width"])), coll, root, lambda _r: mat,
                            loc=(x, 0.0, 0.0))
        ob["mk_line"] = True
        SH.exempt(ob, "a drawn tick")
        expr = SP.tick_expr((L - x) if end else x, L)
        for i in range(3):
            _driver(ob, "scale", expr, root, "extent", i)
    rip = o["ripple"]
    keyed = 0
    if o["ripples"]:
        frames, ts = _clock()
        ends = np.array([[a, b] for _, a, b, _ in segs])
        z = SP.ripple(ts, ends.ravel(), o["ripples"], rip["speed"], rip["decay"], rip["width"], rip["life"])
        z = z.reshape(len(ts), len(segs), 2)
        for k, (ob, _, _, length) in enumerate(segs):
            zk = z[:, k]
            if not np.abs(zk).max() > 1e-6:
                continue
            tilt = -toward * np.arctan2(zk[:, 1] - zk[:, 0], length)
            K.key_prop(ob, "location", frames, zk[:, 0], index=2, interp="LINEAR")
            K.key_prop(ob, "rotation_euler", frames, tilt, index=1, interp="LINEAR")
            keyed += 1
    return {"size": [round(L, 4), round(2 * r, 4), round(hi_t, 4)], "origin": "start", "front": "-Y", "blocks": False,
            "bounds": _bounds((0.0, -r, -hi_t / 2), (L, r, hi_t / 2)), "layers": [],
            "line": {"segments": len(segs), "anchor": o["anchor"], "ticks": len(SP.ticks(L, float(o["ticks"]),
                                                                                       int(o["bar"]))),
                     "rippled_segments": keyed}}


# ================================================================================================================ hole
@register("hole")
@_err
def hole(name, coll, root, slots=None):
    o = SP.hole_options(slots)
    clear_drivers(root)
    _live(root, "breathe", 1.0, "the shadow's scale (a kick's inhale: key it)")
    mat = _flat(f"{name}_shadow", _colour(o["color"], slots or {}), rough=1.0)
    b = mat.node_tree.nodes.get("Principled BSDF")
    b.inputs["Specular IOR Level"].default_value = 0.0                 # no highlight: the shadow of a hole
    R = float(o["radius"])
    ob = SH.mesh_object(f"{name}_shadow", SP.sphere(R), coll, root, lambda _r: mat)
    ob["mk_hole"] = R
    for i in range(3):
        _driver(ob, "scale", "b", root, "breathe", i)
    return {"size": [2 * R] * 3, "origin": "center", "front": "-Y", "blocks": False,
            "bounds": _bounds((-R, -R, -R), (R, R, R)), "layers": [], "hole": {"radius": R, "object": ob.name},
            "use": {"look": [{"name": "centre", "point": [0.0, 0.0, 0.0]}]}}


# ================================================================================================================ board
@register("round_board")
@_err
def round_board(name, coll, root, slots=None):
    o = SP.board_options(slots)
    _purge(f"{name}_r")
    _purge(f"{name}_hot_")
    clear_drivers(root)
    _live(root, "phase", 0.0, "how far the rings have turned: ring i is at omega_i * phase radians")
    s = slots or {}
    mats = [_flat(f"{name}_light", _colour(o["light"], s), rough=0.25),
            _flat(f"{name}_dark", _colour(o["dark"], s), rough=0.5),
            _flat(f"{name}_hot", _colour(o["hot"], s), rough=0.5, glow=1.0)]
    om = SP.omegas(o)
    rings = []
    for i in range(o["rings"]):
        ob = SH.mesh_object(f"{name}_r{i}", SP.ring_mesh(o, i), coll, root, lambda role: mats[role])
        ob.rotation_mode = "XYZ"
        SH.exempt(ob, "a board's cells")
        _driver(ob, "rotation_euler", f"p * {om[i]:.8g}", root, "phase", 2)
        rings.append(ob.name)
        for j in range(o["sectors"]):
            hot = SH.mesh_object(f"{name}_hot_r{i}s{j}", SP.hot_mesh(o, i, j), coll, ob, lambda _r: mats[2])
            hot.hide_render = True
            hot.hide_viewport = False
            SH.exempt(hot, "a highlight")
    R = o["inner"] + o["rings"] * o["width"]
    return {"size": [2 * R, 2 * R, o["thick"]], "origin": "center", "front": "-Y", "blocks": False,
            "bounds": _bounds((-R, -R, -o["thick"]), (R, R, 0.0)), "layers": [],
            "board": {"rings": rings, "omega": [round(w, 6) for w in om], "sectors": o["sectors"],
                      "radii": [[round(a, 4), round(b, 4)] for a, b in SP.radii(o["inner"], o["width"], o["rings"])]},
            "use": {"rest": [{"name": "top", "type": "plane", "center": [0.0, 0.0, 0.0], "radius": round(R, 4)}],
                    "look": [{"name": "centre", "point": [0.0, 0.0, 0.0]}]}}


# ================================================================================================================ chess
@register("chess")
@_err
def chess(name, coll, root, slots=None):
    o = SP.chess_options(slots)
    rgb = _colour(o["color"], slots or {})
    mat = _flat(f"{name}_piece", rgb, rough=float(o["rough"]), metal=float(o["metal"]))
    m = bpy.data.materials[f"{name}_piece"].node_tree.nodes.get("Principled BSDF")
    m.inputs["Coat Weight"].default_value = 0.6
    m.inputs["Coat Roughness"].default_value = 0.05
    h = float(o["height"])
    mesh = SP.piece_mesh(o["piece"], h)
    ob = SH.mesh_object(f"{name}_piece", mesh, coll, root, lambda _r: mat)
    lo, hi = mesh.bbox()
    return {"size": [round(float(v), 4) for v in hi - lo], "origin": "floor_center", "front": "+X", "blocks": False,
            "bounds": _bounds(lo, hi), "layers": [], "chess": {"piece": o["piece"], "object": ob.name}}


# ================================================================================================================ stars
@register("stars")
@_err
def stars(name, coll, root, slots=None):
    o = SP.stars_options(slots)
    _purge(f"{name}_star")
    _purge(f"{name}_line")
    mat = _material(f"{name}_star", _colour(o["color"], slots or {}), 3.0)
    frames, ts = _clock()
    star = SP.star_mesh(int(o["kind"]), float(o["inner"]), float(o["size"]), float(o["depth"]))
    pts = o["points"]
    me = None
    for i, (x, z, size, t) in enumerate(pts):
        if me is None:
            me, _ = SH.mesh_data(f"{name}_star", star, lambda _r: mat)
        ob = bpy.data.objects.new(f"{name}_star{i}", me)
        coll.objects.link(ob)
        ob.parent = root
        ob.location = (x, 0.0, z)
        ob.scale = (size, size, size)
        SH.exempt(ob, "a drawn star")
        if t is not None:
            s = SP.flash(ts, t, float(o["flash"]), float(o["gain"])) * size
            keep = np.flatnonzero(np.abs(np.diff(s, prepend=s[0])) > 1e-6)
            if len(keep):
                sel = np.unique(np.clip(np.concatenate([keep - 1, keep]), 0, len(frames) - 1))
                for c in range(3):
                    K.key_prop(ob, "scale", frames[sel], s[sel], index=c, interp="LINEAR")
    drawn = 0
    if o["lines"]:
        for i in range(len(pts) - 1):
            (x0, z0, _, _), (x1, z1, _, t1) = pts[i], pts[i + 1]
            length = float(np.hypot(x1 - x0, z1 - z0))
            if length < 1e-6:
                continue
            ob = SH.mesh_object(f"{name}_line{i}", SP.bar_mesh(length, float(o["line_width"])), coll, root,
                                lambda _r: mat, loc=(x0, 0.002, z0))
            ob.rotation_mode = "XYZ"
            ob.rotation_euler = (0.0, -float(np.arctan2(z1 - z0, x1 - x0)), 0.0)
            SH.exempt(ob, "a constellation's line")
            if t1 is not None:                                          # it draws itself toward the star as it flashes
                d = float(o["draw"])
                grow = np.clip((ts - (t1 - d)) / d, SP.MIN_SCALE, 1.0)
                K.key_prop(ob, "scale", frames, grow, index=0, interp="LINEAR")
            drawn += 1
    if pts:
        P = np.array([[p[0], p[1]] for p in pts])
        pad = float(o["size"]) * max(float(o["gain"]), 1.0) * max(p[2] for p in pts)
        lo, hi = P.min(0) - pad, P.max(0) + pad
    else:
        lo, hi = np.zeros(2), np.zeros(2)
    return {"size": [round(float(hi[0] - lo[0]), 4), round(float(o["depth"]), 4), round(float(hi[1] - lo[1]), 4)],
            "origin": "keep", "front": "-Y", "blocks": False, "layers": [],
            "bounds": _bounds((lo[0], -float(o["depth"]), lo[1]), (hi[0], 0.0, hi[1])),
            "stars": {"count": len(pts), "lines": drawn}}


# ================================================================================================================ floor
def _floor_material(name, o, slots):
    """The floor's shader (core.space.checker / grid_lines in nodes): the pattern from the object's x, y, softened by
    the camera distance, faded into `dark` between the `fade` distances; a mirror is the dark glossy surface alone."""
    m, nb = NK.new_material(name)
    light, dark = _colour(o["color"], slots), _colour(o["dark"], slots)
    x, y, _ = nb.sep(nb.texcoord("Object"))
    d = nb.cam_distance()
    soft = nb.math("MAXIMUM", 0.004, nb.math("MULTIPLY", d, 0.0025))
    far = nb.smooth(float(o["fade"][0]), float(o["fade"][1]), d)
    c = float(o["cell"])
    base, glow = dark + (1.0,), None
    if o["kind"] == "chess":
        s = nb.math("MULTIPLY", nb.math("SINE", nb.math("MULTIPLY", x, math.pi / c)),
                    nb.math("SINE", nb.math("MULTIPLY", y, math.pi / c)))
        k = nb.math("DIVIDE", c / math.pi, soft)
        mask = nb.math("ADD", 0.5, nb.math("MULTIPLY", 0.5, nb.math("TANH", nb.math("MULTIPLY", s, k))))
        base = nb.mixc(far, nb.mixc(mask, dark + (1.0,), light + (1.0,)), dark + (1.0,))
    elif o["kind"] == "grid":                  # core.space.grid_lines: each line box-filtered over `soft`, a trapezoid
        half = float(o["line"]) / 2.0
        cover = []
        for v in (x, y):
            q = nb.math("DIVIDE", v, c)
            dist = nb.math("MULTIPLY", nb.math("ABSOLUTE", nb.math("SUBTRACT", q, nb.math("ROUND", q))), c)
            hi = nb.math("MINIMUM", half, nb.math("ADD", dist, soft))
            lo = nb.math("MAXIMUM", -half, nb.math("SUBTRACT", dist, soft))
            cover.append(nb.math("DIVIDE", nb.math("MAXIMUM", 0.0, nb.math("SUBTRACT", hi, lo)),
                                 nb.math("MULTIPLY", soft, 2.0)))
        lines = nb.math("SUBTRACT", nb.math("ADD", cover[0], cover[1]), nb.math("MULTIPLY", cover[0], cover[1]))
        glow = nb.math("MULTIPLY", nb.math("MULTIPLY", lines, float(o["glow"])), nb.math("SUBTRACT", 1.0, far))
    sh = nb.principled(base, rough=float(o["rough"]), metal=float(o["metal"]), spec=1.0)
    if glow is not None:
        sh = nb.add_shader(sh, nb.emission(light + (1.0,), glow))
    nb.output(sh)
    m.diffuse_color = (*light, 1.0) if o["kind"] != "mirror" else (*dark, 1.0)
    return m


@register("floor")
@_err
def floor(name, coll, root, slots=None):
    o = SP.floor_options(slots)
    _purge(f"{name}_floor")
    R = float(o["size"])
    mat = _floor_material(f"{name}_floor", o, slots or {})
    plane = CS.Mesh([[-R, -R, 0.0], [R, -R, 0.0], [R, R, 0.0], [-R, R, 0.0]], [[0, 1, 2, 3]])
    ob = SH.mesh_object(f"{name}_floor", plane, coll, root, lambda _r: mat)
    SH.exempt(ob, "a floor to the horizon")
    e = bpy.context.scene.eevee                # what the camera sees is mirrored in it: EEVEE Next's screen-space tracing
    e.use_raytracing = True
    e.ray_tracing_method = "SCREEN"
    e.ray_tracing_options.resolution_scale = "1"
    e.ray_tracing_options.use_denoise = True
    e.ray_tracing_options.trace_max_roughness = max(e.ray_tracing_options.trace_max_roughness, 0.5)
    return {"size": [2 * R, 2 * R, 0.0], "origin": "center", "front": "-Y", "blocks": False, "layers": [],
            "bounds": _bounds((-R, -R, 0.0), (R, R, 0.0)), "floor": {"kind": o["kind"], "object": ob.name},
            "use": {"rest": [{"name": "top", "type": "plane", "center": [0.0, 0.0, 0.0], "size": [2 * R, 2 * R]}]}}


# ================================================================================================================ starfield
@register("starfield")
@_err
def starfield(name, coll, root, slots=None):
    o = SP.starfield_options(slots)
    _purge(f"{name}_tile")
    mat = _flat(f"{name}_star", _colour(o["color"], slots or {}), rough=1.0, glow=float(o["glow"]))
    me, _ = SH.mesh_data(f"{name}_stars", SP.starfield_mesh(o), lambda _r: mat, smooth=False, normals=False)
    frames, ts = _clock()
    s = SP.travel(ts, o)
    L = float(o["length"])
    tiles = []
    for k in range(2):                                # two tiles leapfrog round the root: the stars never run out
        ob = bpy.data.objects.new(f"{name}_tile{k}", me)
        coll.objects.link(ob)
        ob.parent = root
        SH.exempt(ob, "a field of stars")
        K.key_prop(ob, "location", frames, SP.tile_x(s, k, L), index=0, interp="LINEAR")
        tiles.append(ob.name)
    lo, hi = (-L, -o["width"] / 2.0, float(o["height"][0])), (L, o["width"] / 2.0, float(o["height"][1]))
    return {"size": [round(h - l, 4) for l, h in zip(lo, hi)], "origin": "keep", "front": "-Y", "blocks": False,
            "layers": [], "bounds": _bounds(lo, hi), "starfield": {"stars": 2 * o["count"], "tiles": tiles}}


# ================================================================================================================ disc
def _disc_material(name, o, slots):
    """The disc's glow: polar coordinates of the object's x, y, each radius turned back by how far it has orbited
    (core.space.disc_turn, the clock a driver on a Value node), a noise stretched along the orbit for its streaks, hot
    white inside going to `color` outward, added over what is behind it."""
    m, nb = NK.new_material(name, blended=True)
    m.use_backface_culling = False
    x, y, _ = nb.sep(nb.texcoord("Object"))
    r = nb.math("SQRT", nb.math("ADD", nb.math("MULTIPLY", x, x), nb.math("MULTIPLY", y, y)))
    clock = nb.node("ShaderNodeValue")
    sc = bpy.context.scene
    f0, fps = float(sc.get("mk_frame0", sc.frame_start)), sc.render.fps / sc.render.fps_base
    fc = clock.outputs[0].driver_add("default_value")
    fc.driver.type, fc.driver.expression = "SCRIPTED", f"(frame - {f0:g}) / {fps:g}"
    turn = nb.math("MULTIPLY", nb.math("MULTIPLY", 2.0 * math.pi * float(o["spin"]), nb.math(
        "POWER", nb.math("DIVIDE", r, float(o["inner"])), float(o["kepler"]))), clock.outputs[0])
    ang = nb.math("SUBTRACT", nb.math("ARCTAN2", y, x), turn)
    A, k = float(o["streaks"]), float(o["rings"])
    p = nb.comb(nb.math("MULTIPLY", nb.math("COSINE", ang), A), nb.math("MULTIPLY", nb.math("SINE", ang), A),
                nb.math("MULTIPLY", r, k / (o["outer"] - o["inner"])))
    n = nb.noise(p, scale=1.0, detail=4.0, rough=0.55, which="Fac")
    u = nb.remap(r, float(o["inner"]), float(o["outer"]))                      # 0 at the inner rim .. 1 outside
    body = nb.math("MULTIPLY", nb.smooth(0.0, 0.06, u), nb.math("POWER", nb.math("SUBTRACT", 1.0, u), 1.6))
    streak = nb.math("POWER", nb.remap(n, 0.3, 0.75), 1.5)
    colour = nb.ramp(u, [(0.0, _colour(o["hot"], slots) + (1.0,)), (0.22, _colour(o["color"], slots) + (1.0,)),
                         (1.0, tuple(0.35 * v for v in _colour(o["color"], slots)) + (1.0,))])
    strength = nb.math("MULTIPLY", nb.math("MULTIPLY", body, nb.math("ADD", 0.25, streak)), float(o["glow"]))
    nb.output(nb.add_shader(nb.transparent(), nb.emission(colour, strength)))
    return m


@register("disc")
@_err
def disc(name, coll, root, slots=None):
    o = SP.disc_options(slots)
    _purge(f"{name}_disc")
    mat = _disc_material(f"{name}_disc", o, slots or {})
    ob = SH.mesh_object(f"{name}_disc", SP.disc_mesh(o), coll, root, lambda _r: mat)
    SH.exempt(ob, "an accretion disc of light")
    R = float(o["outer"])
    return {"size": [2 * R, 2 * R, 0.0], "origin": "center", "front": "-Y", "blocks": False, "layers": [],
            "bounds": _bounds((-R, -R, 0.0), (R, R, 0.0)), "disc": {"object": ob.name},
            "use": {"look": [{"name": "centre", "point": [0.0, 0.0, 0.0]}]}}
