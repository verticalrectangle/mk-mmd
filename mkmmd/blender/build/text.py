r"""text: kinetic type on surfaces, rendered by EEVEE with no Python at render time (docs/design.md: Text).

Every [[text]] becomes one object: a geometry-nodes modifier (String to Curves -> Fill Curve) under the owner of its
surface, with a palette-coloured emissive material. What changes over time is keyed on node inputs (a number, the
typewriter's ramp, a blink gain), never driven by handlers, so `mk render` needs nothing but the saved .blend. Fonts
are packed into the .blend.

[[text]] keys (times in clip seconds; lengths in metres)
  name = "gantry_left"      object name (required, unique); [[key]] can target it (hide_render, location, ...)
  on = "road:gantry1_panel1"   a card surface, `<set or prop>:<surface>` (use.surface: centre, normal, up, size); the
                            surface name may be left out when the owner has exactly one
  mount = "car", at = [x, y, z], facing = [x, y, z], up = [0, 0, 1], box = [w, h]
                            free placement instead of `on`: panel centre `at` in the mount's frame (a set, prop or
                            object name; default the world), the direction the text faces, the direction of its top
                            edge, and the panel size to fit and align in (without a box, text is placed around `at`).
                            `box` also narrows a surface's panel
  text = "LINE ONE\nLINE TWO"   the string (a list of lines also works)
  value = {keys = [[t, v], ...], format = "{:.0f} MPH", interp = "BEZIER"}   a number keyed over time instead of
                            `text`; format: literal text around one {:[0][width][.decimals][f|d]} field
  font = "overpass_bold"    asset registry slug (kind font) or a font file; default Blender's built-in font
  size = 0.18               cap height; fit = 0.85  the ink of the widest string the text will ever show fills at
                            most this share of the panel (also the margin text aligns in; default 0.9). Give one
                            (size alone does not shrink to fit; both: size is the largest cap height fit allows)
  align = "left" | "center" | "right", valign = "top" | "middle" | "bottom"   (align may carry both words)
  offset = [u, v]           shift along the panel's right and up axes
  color = "text"            palette slot or #hex (slot:slot:t mixes two); glow = 1.0  emission strength;
                            lit = 1.0  how much of the scene's light the letters also reflect (0: flat ink)
  depth = 0.0               extrusion toward the viewer; lift = 0.002  distance in front of the surface
  tracking = 1.0            character spacing; word_spacing = 1.0  gap between words; leading = 1.5  line pitch / cap
                            height
  reveal = {from = t, to = t}   typewriter: first character at `from`, last at `to` (spaces cost no time)
  blink = {period = 1.0, duty = 0.5, low = 0.0, phase = 0.0, from, to}   emission gain steps on and off
  flicker = {amount = 0.5, rate = 12.0, dips = 0.25, seed = 1, from, to}  failing-tube dips in the emission gain
  fade = [[t, gain], ...]   keyed multiplier on the emission (a dash waking up, a sign switching on); linear, or
                            {keys, interp}
  ghost = true | 0.10 | {strength, text, color}   unlit segments behind the lit ones (display fonts): every letter and
                            digit of the widest string as an 8, a faint hint: `strength` is their brightness as a share
                            of the lit ones as displayed (default 0.10), colour the text's pulled halfway to muted
  halo = true | {strength, size}   a slight glow bleeding past the lit edges (additive copies of the lit shapes around
                            them): `strength` the share of the lit emission it adds (0.3), `size` how far it reaches,
                            in cap heights (0.04)
  haze = false | {distance, cap}   aerial perspective; text on a highway set fades into the road's haze by default
  back = true               the text on the back of the surface (the outer side of a pane): the frame turns about `up`
                            so it still reads correctly from behind
  kinetic = {...}           keyed motion of one text (arrival, letter spread, weight, drip, recolour, when it is on
                            screen), built from keys by `lyrics` below or written by hand; core/typefx.py lists the keys
  lyrics = {timeline = "audio/timeline.json", line = 3, words = [1, 4], style = "pop", from = t, to = t, ...}
                            one text per sung word, read from the timeline (never printed anywhere), each with its own
                            `kinetic` keys; the entry's placement and look keys go to every word: build/wordtype.py
  backing = {color = "rose", pattern = "stripe", ...}   a strip of tape behind the text (core/typefx.py): it is the
                            text's ink box plus a margin, moves with the word and fades with its opacity
  outline = {color = "text", width = 0.05, alpha = 0.5}   a soft ring of another colour behind the letters (width in em)
                            so they read on any background; it moves and fades with the word
  ink = {...}               handwriting on a surface: build/ink.py
  screen = true | {anchor = "top", side = "left", margin = 0.04, height = 0.25, width = 1.0}   type of the PICTURE, not of the
                            scene: laid over the finished frame of whichever shot is cutting (a second pass of
                            mkmmd/blender/styles.py, kept as `screen/<frame>.png` for `mk post`), measured in frame
                            heights from the picture's centre (x right, y up; `size`, `box`, `at`, `offset` too), so it
                            is placed once for every shot and both outputs; the band is the margin-inset frame (anchor: a
                            strip of `height` at the top, bottom or middle; side: against the left or right margin,
                            `width` the share of the room). Needs no surface
  knockout = true           screen type reversed out of a silhouette shot (the figure's ink on the colour, the colour on
                            the figure; with no strip or ring); in any other shot it is ordinary screen type
  aspect.<output> = {...}   an output's own version of the entry: its keys laid over the entry's (tables merge, anything
                            else is replaced). An entry with `screen` or `aspect` is built once per output (objects end
                            `@<output>` when the project has several, rendered for that output only)
  extends = "name", abstract = true   an entry that is the named entry with its own keys laid over it; an `abstract` entry
                            is only a base and is never built (a look that eight lyric lines share is written once)
The text frame is the card's: x right (up x normal), y up, z out of the surface; screen type's is the picture's."""
import json
import math
import os

import bpy
import numpy as np
from mathutils import Matrix

from ...core import screentype as SR
from ...core import wordtype as LY
from ...core import typefx as FX
from ...core import typeset as T
from .. import keys as K
from ..library.sets import highway as HIGHWAY
from ..library.sets import nightkit as NK
from . import BuildError, collection

KNOWN = {"name", "on", "mount", "at", "facing", "up", "box", "text", "value", "font", "size", "fit", "align", "valign",
         "offset", "color", "glow", "lit", "depth", "lift", "tracking", "word_spacing", "leading", "reveal", "blink",
         "flicker", "fade", "ghost", "halo", "haze", "back", "kinetic", "lyric", "backing", "outline", "screen", "knockout",
         "_aspect", "_multi", "_shot"}
MAX_NAME = 50                  # characters of a text's name: Blender keeps 63, and the node group's `mk_text_` takes 8
LIFT = 0.002                    # m in front of the surface
GHOST_BACK = 0.0006             # m the unlit segments sit behind the lit ones
GHOST_STRENGTH = 0.10           # brightness of the unlit segments as displayed, as a share of the lit ones
HALO_BACK = 0.0003              # m the halo sits behind the lit shapes (in front of the unlit segments)
HALO_STRENGTH, HALO_SIZE = 0.3, 0.04
HALO_RINGS = ((12, 1 / 3), (16, 2 / 3), (20, 1.0))     # (copies, radius share) of the rings around the glyph
HALO_COPIES = sum(n for n, _ in HALO_RINGS)
ALIGN_X = {"left": "LEFT", "center": "CENTER", "right": "RIGHT"}
FONT_EXT = (".ttf", ".otf", ".ttc", ".pfb", ".pfm")


# ------------------------------------------------------------------------------------------------------ nodes
class _G:
    """Tiny helper over a geometry node tree: nodes in a row, inputs set or linked."""

    def __init__(self, nt):
        self.nt = nt
        self.x = 0

    def node(self, kind, name=None, **props):
        n = self.nt.nodes.new(kind)
        n.location = (self.x, 0)
        self.x += 220
        if name:
            n.name = n.label = name
        for k, v in props.items():
            setattr(n, k, v)
        return n

    def put(self, sock, v):
        if isinstance(v, bpy.types.NodeSocket):
            self.nt.links.new(v, sock)
        else:
            sock.default_value = v

    def math(self, op, a, b=None):
        n = self.node("ShaderNodeMath", operation=op)
        self.put(n.inputs[0], a)
        if b is not None:
            self.put(n.inputs[1], b)
        return n.outputs[0]


def _new_tree(name):
    old = bpy.data.node_groups.get(name)
    if old is not None:
        bpy.data.node_groups.remove(old)
    nt = bpy.data.node_groups.new(name, "GeometryNodeTree")
    nt.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
    return nt


def _string_to_curves(g, font, align_x, tracking, word_spacing):
    n = g.node("GeometryNodeStringToCurves", align_x=align_x, align_y="TOP_BASELINE", overflow="OVERFLOW")
    if font is not None:
        n.font = font
    n.inputs["Character Spacing"].default_value = tracking
    n.inputs["Word Spacing"].default_value = word_spacing
    return n


def _filled(g, curves):
    """Curve instances -> one filled mesh in the text plane."""
    real = g.node("GeometryNodeRealizeInstances")
    g.put(real.inputs["Geometry"], curves)
    fill = g.node("GeometryNodeFillCurve")
    g.put(fill.inputs["Curve"], real.outputs[0])
    return fill.outputs[0]


class _Measure:
    """A throwaway geometry-nodes object that lays strings out exactly as the text object will (same node, font,
    alignment and spacing) and reports the ink box of each, so fit and alignment are exact for any font."""

    def __init__(self, coll, font, align_x, tracking, word_spacing, tag):
        self.nt = _new_tree(f"mk_text_measure_{tag}")
        self.s_id = self.nt.interface.new_socket("String", in_out="INPUT", socket_type="NodeSocketString").identifier
        self.l_id = self.nt.interface.new_socket("Line Spacing", in_out="INPUT",
                                                 socket_type="NodeSocketFloat").identifier
        g = _G(self.nt)
        gin = g.node("NodeGroupInput")
        stc = _string_to_curves(g, font, align_x, tracking, word_spacing)
        stc.inputs["Size"].default_value = 1.0
        g.put(stc.inputs["String"], gin.outputs["String"])
        g.put(stc.inputs["Line Spacing"], gin.outputs["Line Spacing"])
        out = g.node("NodeGroupOutput")
        g.put(out.inputs[0], _filled(g, stc.outputs["Curve Instances"]))
        self.me = bpy.data.meshes.new(f"mk_text_measure_{tag}")
        self.ob = bpy.data.objects.new(f"mk_text_measure_{tag}", self.me)
        coll.objects.link(self.ob)
        self.mod = self.ob.modifiers.new("Measure", "NODES")
        self.mod.node_group = self.nt
        self._cache = {}
        self.cap = None

    def ink(self, s, line_spacing):
        """(x0, y0, x1, y1) of the ink of `s` at em size 1, or None when it draws nothing."""
        key = (s, round(line_spacing, 6))
        if key in self._cache:
            return self._cache[key]
        self.mod[self.s_id] = s
        self.mod[self.l_id] = float(line_spacing)
        self.ob.update_tag()
        ev = self.ob.evaluated_get(bpy.context.evaluated_depsgraph_get())
        me = ev.to_mesh()
        n = len(me.vertices)
        res = None
        if n:
            co = np.empty(n * 3, np.float32)
            me.vertices.foreach_get("co", co)
            co = co.reshape(-1, 3)
            res = (float(co[:, 0].min()), float(co[:, 1].min()), float(co[:, 0].max()), float(co[:, 1].max()))
        ev.to_mesh_clear()
        self._cache[key] = res
        return res

    def cap_ratio(self):
        """Cap height per em: the height of a flat-topped capital (E, then H)."""
        if self.cap is None:
            for ch in ("E", "H"):
                b = self.ink(ch, 1.0)
                if b:
                    self.cap = b[3] - b[1]
                    break
            else:
                raise T.TextError("the font has neither E nor H to take a cap height from")
        return self.cap

    def close(self):
        for coll, item in ((bpy.data.objects, self.ob), (bpy.data.meshes, self.me), (bpy.data.node_groups, self.nt)):
            coll.remove(item)


# ------------------------------------------------------------------------------------------------------ inputs
def _registry(ctx):
    root = os.path.expanduser(ctx.assets or "~/mk-assets")
    path = os.path.join(root, "registry.json")
    if not os.path.exists(path):
        return root, {}
    with open(path, encoding="utf-8") as fh:
        return root, {e["slug"]: e for e in json.load(fh)}


def _font(ctx, ref, cache):
    """(VectorFont or None for the built-in font, label) for a registry slug or a file; fonts are packed."""
    if not ref:
        return None, "builtin"
    ref = str(ref)
    if ref in cache["fonts"]:
        return cache["fonts"][ref], ref
    if any(c in ref for c in "/\\~") or os.path.splitext(ref)[1].lower() in FONT_EXT:
        path = ctx.path(ref)
    else:
        root, reg = _registry(ctx)
        e = reg.get(ref)
        if e is None or e.get("kind") != "font":
            fonts = sorted(s for s, x in reg.items() if x.get("kind") == "font")
            raise BuildError(f"font {ref!r}: not a font file and not a font in the asset registry (have {fonts})")
        path = e["path"] if os.path.isabs(e["path"]) else os.path.join(root, e["path"])
    if not os.path.exists(path):
        raise BuildError(f"font {ref!r}: {path} does not exist")
    vf = bpy.data.fonts.load(path, check_existing=True)
    if vf.packed_file is None:
        vf.pack()
    cache["fonts"][ref] = vf
    return vf, ref


def _mount(ctx, name):
    if not name:
        return None
    if name in ctx.props:
        return ctx.props[name].root
    if name in ctx.sets:
        return ctx.sets[name].root
    ob = bpy.data.objects.get(name)
    if ob is None:
        raise BuildError(f"mount {name!r}: no prop, set or object of that name")
    return ob


def _output_size(ctx, output):
    """(w, h) pixels of a project output."""
    for o in ctx.project.get("outputs") or []:
        if o["name"] == output:
            return int(o["size"][0]), int(o["size"][1])
    sc = bpy.context.scene
    return int(sc.render.resolution_x), int(sc.render.resolution_y)


def _place(ctx, spec):
    """Where the text goes: parent object, its parent-inverse (None = identity), the frame (4x4, in the parent's frame
    or the world), the panel (w, h) or None, the owner (kind, name) and a label. Screen type has no parent: it stands on the
    plane z = 0 of the world, one unit per frame height (docs/design.md: Text, Screen type)."""
    name = spec["name"]
    screen = SR.screen_spec(spec)
    if screen is not None:
        clash = [k for k in ("on", "mount", "facing", "up") if spec.get(k) is not None]
        if clash:
            raise BuildError(f"text {name!r}: screen text is placed in frame heights (`at`, `box`, `screen = {{anchor, "
                             f"margin, height}}`); drop {clash}")
        output = spec.get("_aspect")
        if output is None:
            raise BuildError(f"text {name!r}: screen text is built per output (the text stage sets that up)")
        at, box = SR.panel(screen, _output_size(ctx, output), spec.get("at"), spec.get("box"))
        F = T.surface_matrix((at[0], at[1], 0.0), (0.0, 0.0, 1.0), (0.0, 1.0, 0.0), float(spec.get("lift", 0.0)))
        return None, None, F, tuple(box), ("screen", output), f"screen@{output}"
    lift = float(spec.get("lift", LIFT))
    if spec.get("on"):
        ref = str(spec["on"])
        owner, _, sname = ref.partition(":")
        if owner in ctx.props:
            kind, o = "prop", ctx.props[owner]
        elif owner in ctx.sets:
            kind, o = "set", ctx.sets[owner]
        else:
            raise BuildError(f"text {name!r}: on {ref!r}: no prop or set {owner!r} (props {sorted(ctx.props)}, sets "
                             f"{sorted(ctx.sets)})")
        surfaces = o.card.get("use", {}).get("surface", [])
        names = [s["name"] for s in surfaces]
        if sname:
            s = next((s for s in surfaces if s["name"] == sname), None)
            if s is None:
                raise BuildError(f"text {name!r}: {kind} {owner!r} has no surface {sname!r} (has {names})")
        elif len(surfaces) == 1:
            s = surfaces[0]
        else:
            raise BuildError(f"text {name!r}: on {ref!r}: {kind} {owner!r} has " +
                             (f"{len(surfaces)} surfaces, name one ({names})" if surfaces else "no surfaces"))
        normal = [-float(v) for v in s["normal"]] if spec.get("back") else s["normal"]      # back: the other side
        F = T.surface_matrix(s["center"], normal, s.get("up", (0.0, 0.0, 1.0)), lift)
        panel = tuple(float(v) for v in (spec.get("box") or s["size"]))
        parent, inverse = o.root, None
        carrier = bpy.data.objects.get(s["object"]) if s.get("object") else None
        if carrier is not None and carrier.parent == o.root:     # the surface rides an object of its own
            parent = carrier
            inverse = (carrier.matrix_parent_inverse @ carrier.matrix_basis).inverted()
        return parent, inverse, F, panel, (kind, owner), f"{owner}:{s['name']}"
    if spec.get("facing") is None:
        raise BuildError(f"text {name!r}: give `on` (a card surface) or `facing` (free placement with `at`)")
    F = T.surface_matrix(spec.get("at", (0.0, 0.0, 0.0)), spec["facing"], spec.get("up", (0.0, 0.0, 1.0)), lift)
    panel = tuple(float(v) for v in spec["box"]) if spec.get("box") else None
    mount = spec.get("mount")
    kind = "prop" if mount in ctx.props else "set" if mount in ctx.sets else "object"
    return _mount(ctx, mount), None, F, panel, (kind, mount), f"{mount or 'world'}@{list(spec.get('at', (0, 0, 0)))}"


def _content(ctx, spec):
    """(strings the text can show, number info or None). Number info: {nf, keys, interp}."""
    name = spec["name"]
    if ("text" in spec) == ("value" in spec):
        raise BuildError(f"text {name!r}: give exactly one of `text` and `value`")
    if "text" in spec:
        t = spec["text"]
        t = "\n".join(str(x) for x in t) if isinstance(t, list) else str(t)
        if not t.strip():
            raise BuildError(f"text {name!r}: the text is empty")
        return [t], None
    v = spec["value"]
    if not isinstance(v, dict) or not v.get("keys"):
        raise BuildError(f"text {name!r}: value needs keys = [[t, v], ...]")
    keys = sorted((float(t), float(x)) for t, x in v["keys"])
    nf = T.parse_format(v.get("format", "{:.0f}"))
    vals = [x for _, x in keys]
    interp = v.get("interp", "BEZIER")
    if interp not in K.INTERP:
        raise BuildError(f"text {name!r}: value interp {interp!r} must be one of {sorted(K.INTERP)}")
    return T.candidate_strings(min(vals), max(vals), nf), {"nf": nf, "keys": keys, "interp": interp}


def _haze(ctx, spec, owner):
    """{distance, cap, rgb} or None: `haze = false` none; a table (distance, cap) as given; absent: the set's own haze
    for text on a highway set, nothing for other owners."""
    h = spec.get("haze")
    if h is False:
        return None
    if h is None:
        s = next((s for s in ctx.data.get("set", []) if s.get("name") == owner[1]), None) if owner[0] == "set" else None
        if s is None or s.get("kind") != "highway":
            return None
        h = s.get("haze") or {}
    base = {**HIGHWAY.HAZE, **(h if isinstance(h, dict) else {})}
    return {"distance": float(base["distance"]), "cap": float(base["cap"]),
            "rgb": NK.rgb(ctx.palette, NK.haze_hex(ctx.palette))}


# ------------------------------------------------------------------------------------------------------ material
def _material(name, rgb, glow, haze, gains=("Gain",), lit=1.0):
    """Principled + emission, strength = glow x one keyable Value node per name in `gains` (Gain: blink and flicker,
    Fade: a ramp), faded into the haze with distance."""
    m, nb = NK.new_material(name)
    strength = float(glow)
    for gname in gains:
        v = nb.node("ShaderNodeValue")
        v.name = v.label = gname
        v.outputs[0].default_value = 1.0
        strength = nb.math("MULTIPLY", v.outputs[0], strength)
    sh = nb.principled(tuple(c * lit for c in rgb), rough=0.55, spec=0.3 * lit, emission=rgb, strength=strength)
    if haze:
        sh = NK.haze(nb, sh, haze["rgb"], haze["distance"], haze["cap"])
    nb.output(sh)
    return m


def _additive(name, rgb, glow, gains=()):
    """Light added on top of whatever is behind (nothing is hidden, nothing written to depth): the unlit segments of a
    display. Strength = glow x one keyable Value node per name in `gains`."""
    m, nb = NK.new_material(name, blended=True)
    m.use_backface_culling = False
    strength = float(glow)
    for gname in gains:
        v = nb.node("ShaderNodeValue")
        v.name = v.label = gname
        v.outputs[0].default_value = 1.0
        strength = nb.math("MULTIPLY", v.outputs[0], strength)
    nb.output(nb.add_shader(nb.transparent(), nb.emission(rgb, strength)))
    return m


def _kinetic_material(cache, rgb, glow, to_rgb, haze, lit=1.0, alpha=1.0):
    """The material of kinetic words, shared by every word with the same look: Principled + emission whose alpha is the
    geometry attribute `mk_alpha` (the node tree stores the keyed opacity and the drip's fade there, so the keys stay
    on the objects and the shader is compiled once) and, with `to_rgb`, whose colour mixes toward it by `mk_tint`.
    Alpha is hashed (DITHERED), not blended: no sorting against the glass."""
    key = (tuple(round(v, 5) for v in rgb), round(float(glow), 5), round(float(lit), 5), round(float(alpha), 5),
           None if to_rgb is None else tuple(round(v, 5) for v in to_rgb),
           None if not haze else (haze["distance"], haze["cap"]))
    m = cache["kmats"].get(key)
    if m is not None:
        return m
    m, nb = NK.new_material(f"mk_text_kinetic_{len(cache['kmats'])}")
    m.surface_render_method = "DITHERED"
    col = rgb
    if to_rgb is not None:
        col = nb.mixc(nb.attr("mk_tint", "Fac"), rgb + (1.0,), to_rgb + (1.0,))
    base = nb.mixc(lit, (0.0, 0.0, 0.0, 1.0), col) if lit < 1.0 else col
    a = nb.attr("mk_alpha", "Fac")
    if alpha < 1.0:
        a = nb.math("MULTIPLY", a, float(alpha))
    sh = nb.principled(base, rough=0.55, spec=0.3 * lit, emission=col, strength=float(glow), alpha=a)
    if haze:
        sh = NK.haze(nb, sh, haze["rgb"], haze["distance"], haze["cap"])
    nb.output(sh)
    cache["kmats"][key] = m
    return m


# ------------------------------------------------------------------------------------------------------ geometry
def _reveal(g, curves):
    """The typewriter: delete the character instances past the count the keyed `Reveal` ramp has typed."""
    ds = g.node("GeometryNodeAttributeDomainSize", component="INSTANCES")
    g.put(ds.inputs["Geometry"], curves)
    n_chars = ds.outputs["Instance Count"]
    ramp = g.node("ShaderNodeValue", name="Reveal")
    ramp.outputs[0].default_value = -1.0
    r = ramp.outputs[0]
    shown = g.math("ADD", g.math("FLOOR", g.math("ADD", g.math("MULTIPLY", r, g.math("SUBTRACT", n_chars, 1.0)),
                                                   1e-5)), 1.0)
    count = g.math("MULTIPLY", shown, g.math("GREATER_THAN", r, -1e-6))
    idx = g.node("GeometryNodeInputIndex")
    gone = g.math("GREATER_THAN", idx.outputs[0], g.math("SUBTRACT", count, 0.5))
    dele = g.node("GeometryNodeDeleteGeometry", domain="INSTANCE")
    g.put(dele.inputs["Geometry"], curves)
    g.put(dele.inputs["Selection"], gone)
    return dele.outputs[0]


def _text_mesh(g, P, string, reveal):
    """String -> instances (-> typewriter) -> filled mesh (-> extruded): the mesh socket, in the text plane."""
    stc = _string_to_curves(g, P["font"], P["align_x"], P["tracking"], P["words"])
    stc.inputs["Size"].default_value = P["em"]
    stc.inputs["Line Spacing"].default_value = P["ls"]
    g.put(stc.inputs["String"], string)
    curves = stc.outputs["Curve Instances"]
    if reveal:
        curves = _reveal(g, curves)
    mesh = _filled(g, curves)
    if P["depth"] > 0:
        ext = g.node("GeometryNodeExtrudeMesh", mode="FACES")
        g.put(ext.inputs["Mesh"], mesh)
        ext.inputs["Offset Scale"].default_value = P["depth"]
        mesh = ext.outputs["Mesh"]
    return mesh


def _placed(g, P, mesh, mat, dz, motion=None):
    """The mesh moved to its place in the panel (and `dz` along the normal), moved on by `motion` (kinetic text: a
    function geometry -> geometry), with its material."""
    tr = g.node("GeometryNodeTransform")
    g.put(tr.inputs["Geometry"], mesh)
    tr.inputs["Translation"].default_value = (P["tx"], P["ty"], dz)
    geo = tr.outputs[0]
    if motion is not None:
        geo = motion(geo)
    sm = g.node("GeometryNodeSetMaterial")
    g.put(sm.inputs["Geometry"], geo)
    sm.inputs["Material"].default_value = mat
    return sm.outputs[0]


def _halo(g, P, mesh, mat):
    """A soft glow: the lit mesh repeated around three rings (HALO_RINGS) with additive material, so
    the light bleeds past the edges by up to P["halo_r"] metres, brightest next to the glyph."""
    rings = []
    for (n, k), turn in zip(HALO_RINGS, (0.0, 0.13, 0.31)):
        circle = g.node("GeometryNodeMeshCircle")
        circle.inputs["Vertices"].default_value = n
        circle.inputs["Radius"].default_value = k * P["halo_r"]
        spin = g.node("GeometryNodeTransform")                 # turn each ring so the copies do not line up
        g.put(spin.inputs["Geometry"], circle.outputs["Mesh"])
        spin.inputs["Rotation"].default_value = (0.0, 0.0, turn)
        pts = g.node("GeometryNodeMeshToPoints", mode="VERTICES")
        g.put(pts.inputs["Mesh"], spin.outputs[0])
        rings.append(pts.outputs["Points"])
    join = g.node("GeometryNodeJoinGeometry")
    for r in rings:
        g.put(join.inputs["Geometry"], r)
    inst = g.node("GeometryNodeInstanceOnPoints")
    g.put(inst.inputs["Points"], join.outputs[0])
    g.put(inst.inputs["Instance"], mesh)
    real = g.node("GeometryNodeRealizeInstances")
    g.put(real.inputs["Geometry"], inst.outputs["Instances"])
    return _placed(g, P, real.outputs[0], mat, -HALO_BACK)


def _tree(name, P, content, mats):
    """The text's node group. content: the static string or the number info; mats: {"lit", "ghost", "halo"} (the last
    two may be missing)."""
    nt = _new_tree(name)
    g = _G(nt)
    if isinstance(content, str):
        string = content
    else:
        nf = content["nf"]
        val = g.node("ShaderNodeValue", name="Value")
        val.outputs[0].default_value = content["keys"][0][1]
        vs = g.node("FunctionNodeValueToString")
        vs.inputs["Decimals"].default_value = nf.decimals
        g.put(vs.inputs["Value"], val.outputs[0])
        parts = []
        if nf.prefix:
            pre = g.node("FunctionNodeInputString", string=nf.prefix)
            parts.append(pre.outputs[0])
        if nf.width:
            ln = g.node("FunctionNodeStringLength")
            g.put(ln.inputs[0], vs.outputs[0])
            pad = g.node("FunctionNodeInputString", string=nf.fill * nf.width)
            sl = g.node("FunctionNodeSliceString")
            g.put(sl.inputs["String"], pad.outputs[0])
            g.put(sl.inputs["Position"], ln.outputs[0])
            sl.inputs["Length"].default_value = nf.width
            parts.append(sl.outputs[0])
        parts.append(vs.outputs[0])
        if nf.suffix:
            suf = g.node("FunctionNodeInputString", string=nf.suffix)
            parts.append(suf.outputs[0])
        if len(parts) == 1:
            string = parts[0]
        else:
            join = g.node("GeometryNodeStringJoin")
            for p in reversed(parts):                  # a multi-input socket reads its links last to first
                g.put(join.inputs["Strings"], p)
            string = join.outputs[0]
    mesh = _text_mesh(g, P, string, P["reveal"])
    geos = [_placed(g, P, mesh, mats["lit"], 0.0)]
    if "ghost" in mats:
        geos.append(_placed(g, P, _text_mesh(g, P, P["ghost_text"], False), mats["ghost"], -GHOST_BACK))
    if "halo" in mats:
        geos.append(_halo(g, P, mesh, mats["halo"]))
    geo = geos[0]
    if len(geos) > 1:
        j = g.node("GeometryNodeJoinGeometry")
        for x in geos:
            g.put(j.inputs["Geometry"], x)
        geo = j.outputs[0]
    out = g.node("NodeGroupOutput")
    g.put(out.inputs[0], geo)
    return nt


def _key(id_data, path, keys, ctx, interp):
    try:
        id_data.path_resolve(path)
    except ValueError as e:
        raise BuildError(f"cannot key {path!r} on {id_data.name!r}: {e}")
    K.set_fcurve(id_data, path, 0, [ctx.frame(t) for t, _ in keys], [v for _, v in keys], interp=interp)


# ------------------------------------------------------------------------------------------------------ kinetic
# `kinetic = {...}` is validated by mkmmd/core/typefx.py (its header lists the keys); here its keys become Value nodes.
NODE_NAMES = {"scale": "Pop", "sx": "SquashX", "sy": "SquashY", "dx": "Dx", "dy": "Dy", "dxp": "DxPanel",
              "dyp": "DyPanel", "rot": "Rot", "tracking": "Tracking", "weight": "Weight", "opacity": "Opacity",
              "tint": "Tint"}
WEIGHT_COPIES = 12              # copies on the ring a keyed weight grows the filled letters with


def _val(g, name, v):
    """A keyable Value node (its output socket)."""
    n = g.node("ShaderNodeValue", name=name)
    n.outputs[0].default_value = float(v)
    return n.outputs[0]


def _xyz(g, x, y, z):
    n = g.node("ShaderNodeCombineXYZ")
    for i, v in enumerate((x, y, z)):
        g.put(n.inputs[i], v)
    return n.outputs[0]


def _smooth(g, x, lo, hi):
    """smoothstep(lo, hi, x) for constants lo < hi (Map Range, smoothstep interpolation)."""
    n = g.node("ShaderNodeMapRange", data_type="FLOAT", interpolation_type="SMOOTHSTEP", clamp=True)
    g.put(n.inputs["Value"], x)
    n.inputs["From Min"].default_value = lo
    n.inputs["From Max"].default_value = hi
    return n.outputs["Result"]


def _motion(g, P, kin):
    """geometry -> geometry: the word's keyed transform about its pivot (scale and squash, rotation, offsets), or None
    when nothing about it moves. The Value nodes are made once, here."""
    ch = {c: _val(g, NODE_NAMES[c], kin[c][0][1]) for c in FX.MOTION if c in kin}
    if not ch:
        return None
    em, (px, py) = P["em"], P["pivot"]
    pw, ph = P["panel"] or (0.0, 0.0)
    if ("dxp" in ch or "dyp" in ch) and not P["panel"]:
        raise BuildError("kinetic dxp / dyp are shares of the panel: give the text a surface or a `box`")

    def total(base, *terms):
        out = base
        for key, k in terms:
            if key in ch:
                out = g.math("ADD", out, g.math("MULTIPLY", ch[key], k))
        return out

    pop = ch.get("scale", 1.0)
    sx = g.math("MULTIPLY", pop, ch["sx"]) if "sx" in ch else pop
    sy = g.math("MULTIPLY", pop, ch["sy"]) if "sy" in ch else pop
    rot = g.math("RADIANS", ch["rot"]) if "rot" in ch else 0.0
    shift = (total(px, ("dx", em), ("dxp", pw)), total(py, ("dy", em), ("dyp", ph)))

    def apply(geo):
        a = g.node("GeometryNodeTransform")
        g.put(a.inputs["Geometry"], geo)
        a.inputs["Translation"].default_value = (-px, -py, 0.0)
        b = g.node("GeometryNodeTransform")
        g.put(b.inputs["Geometry"], a.outputs[0])
        g.put(b.inputs["Rotation"], _xyz(g, 0.0, 0.0, rot))
        g.put(b.inputs["Scale"], _xyz(g, sx, sy, pop))
        c = g.node("GeometryNodeTransform")
        g.put(c.inputs["Geometry"], b.outputs[0])
        g.put(c.inputs["Translation"], _xyz(g, shift[0], shift[1], 0.0))
        return c.outputs[0]

    return apply


def _letters(g, P, inst, kin):
    """Per-letter work on the filled character instances: the drip exit (stretch about each letter's top, fall, sideways
    wobble, fade) and the attributes the material reads, `mk_alpha` (keyed opacity x the drip's fade) and `mk_tint`."""
    em = P["em"]
    fade = 1.0
    d = kin.get("drip")
    if d:
        t = _val(g, "Drip", -1.0)                                  # seconds since the exit started; keyed
        idx = g.node("GeometryNodeInputIndex").outputs[0]
        rnd = g.node("FunctionNodeRandomValue", data_type="FLOAT")
        rnd.inputs[2].default_value, rnd.inputs[3].default_value = 0.0, 1.0       # float Min, Max
        g.put(rnd.inputs["ID"], idx)
        rnd.inputs["Seed"].default_value = d["seed"]
        r = rnd.outputs[1]
        stat = g.node("GeometryNodeAttributeStatistic", data_type="FLOAT", domain="INSTANCE")
        g.put(stat.inputs["Geometry"], inst)
        g.put(stat.inputs["Attribute"], r)
        span = g.math("MAXIMUM", g.math("SUBTRACT", stat.outputs["Max"], stat.outputs["Min"]), 1e-6)
        delay = g.math("MULTIPLY", g.math("DIVIDE", g.math("SUBTRACT", r, stat.outputs["Min"]), span), LY.DRIP_SPREAD)
        u = g.math("MAXIMUM", g.math("SUBTRACT", t, delay), 0.0)
        s = g.math("SUBTRACT", 1.0, g.math("EXPONENT", g.math("DIVIDE", g.math("MULTIPLY", u, -1.0), LY.DRIP_TAU)))
        sy = g.math("ADD", 1.0, g.math("MULTIPLY", s, d["stretch"]))
        sx = g.math("DIVIDE", 1.0, g.math("SQRT", sy))
        fall = g.math("MULTIPLY", g.math("MULTIPLY", g.math("MULTIPLY", u, u), 0.5 * d["g"] / LY.DRIP_FALL), em)
        wob = g.math("MULTIPLY", g.math("MULTIPLY", g.math("SINE", g.math("ADD", g.math("MULTIPLY", u, 9.0), idx)), s),
                     0.03 * em)
        sc = g.node("GeometryNodeScaleInstances")
        g.put(sc.inputs["Instances"], inst)
        g.put(sc.inputs["Scale"], _xyz(g, sx, sy, 1.0))
        g.put(sc.inputs["Center"], g.piv)
        sc.inputs["Local Space"].default_value = True
        tr = g.node("GeometryNodeTranslateInstances")
        g.put(tr.inputs["Instances"], sc.outputs[0])
        g.put(tr.inputs["Translation"], _xyz(g, wob, g.math("MULTIPLY", fall, -1.0), 0.0))
        tr.inputs["Local Space"].default_value = False
        inst = tr.outputs[0]
        fade = g.math("SUBTRACT", 1.0, _smooth(g, u, 0.14 * d["life"], d["life"]))
    g.opacity = _val(g, NODE_NAMES["opacity"], kin["opacity"][0][1]) if "opacity" in kin else 1.0
    alpha = g.math("MULTIPLY", g.opacity, fade) if "opacity" in kin else fade
    names = [("mk_alpha", alpha)] + ([("mk_tint", _val(g, NODE_NAMES["tint"], kin["tint"]["keys"][0][1]))]
                                     if "tint" in kin else [])
    for key, value in names:
        st = g.node("GeometryNodeStoreNamedAttribute", data_type="FLOAT", domain="INSTANCE")
        g.put(st.inputs["Geometry"], inst)
        st.inputs["Name"].default_value = key
        g.put(st.inputs["Value"], value)
        inst = st.outputs[0]
    return inst


def _grown(g, P, mesh, weight):
    """The mesh grown by the keyed radius `weight` (em): copies of it on a ring of that radius and in the middle, so
    the strokes get thicker the way a bolder master's would (counters close a little, corners round)."""
    circle = g.node("GeometryNodeMeshCircle", fill_type="TRIANGLE_FAN")
    circle.inputs["Vertices"].default_value = WEIGHT_COPIES
    g.put(circle.inputs["Radius"], g.math("MULTIPLY", weight, P["em"]))
    pts = g.node("GeometryNodeMeshToPoints", mode="VERTICES")
    g.put(pts.inputs["Mesh"], circle.outputs["Mesh"])
    inst = g.node("GeometryNodeInstanceOnPoints")
    g.put(inst.inputs["Points"], pts.outputs["Points"])
    g.put(inst.inputs["Instance"], mesh)
    real = g.node("GeometryNodeRealizeInstances")
    g.put(real.inputs["Geometry"], inst.outputs["Instances"])
    return real.outputs[0]


# ------------------------------------------------------------------------------------------------------ backing
TAPE_COLS, TAPE_ROWS = 14, 9                    # grid of the strip: the zigzag of the torn ends lives on its end columns


def _tape_material(name, b, rgb, pat_rgb, w, h):
    """Paper in `rgb` with a stripe / dot / check pattern in `pat_rgb` from the strip's own UV (so the pattern is fixed to
    the paper however the word moves), lit by the scene plus a little emission so it never goes dark; alpha is the word's
    `mk_alpha`."""
    m, nb = NK.new_material(name)
    m.surface_render_method = "DITHERED"
    col = rgb + (1.0,)
    if b["pattern"] != "plain":
        uv = nb.node("ShaderNodeUVMap", uv_map="UVMap").outputs[0]
        ux, uy, _ = nb.sep(uv)
        X, Y = nb.math("MULTIPLY", ux, w), nb.math("MULTIPLY", uy, h)
        pitch, duty = FX.TAPE_SCALE[b["pattern"]]
        pitch *= b["scale"]
        soft = 0.05                                 # edge softness in pattern units: no moire from hard thresholds
        if b["pattern"] == "stripe":
            t = nb.math("FRACT", nb.math("DIVIDE", nb.math("ADD", X, Y), pitch))
            mask = nb.remap(t, duty + soft, duty - soft)
        elif b["pattern"] == "check":
            wave = nb.math("MULTIPLY", nb.math("SINE", nb.math("MULTIPLY", nb.math("DIVIDE", X, pitch), math.pi)),
                           nb.math("SINE", nb.math("MULTIPLY", nb.math("DIVIDE", Y, pitch), math.pi)))
            mask = nb.math("ADD", 0.5, nb.math("MULTIPLY", nb.math("TANH", nb.math("MULTIPLY", wave, 6.0)), 0.5))
        else:
            cx = nb.math("DIVIDE", X, pitch)
            colx = nb.math("FLOOR", nb.math("ADD", cx, 0.5))
            gx = nb.math("SUBTRACT", cx, colx)
            cy = nb.math("ADD", nb.math("DIVIDE", Y, pitch), nb.math("MULTIPLY", nb.math("MODULO", colx, 2.0), 0.5))
            gy = nb.math("SUBTRACT", cy, nb.math("FLOOR", nb.math("ADD", cy, 0.5)))
            d = nb.math("SQRT", nb.math("ADD", nb.math("MULTIPLY", gx, gx), nb.math("MULTIPLY", gy, gy)))
            mask = nb.remap(d, duty + soft, duty - soft)
        col = nb.mixc(nb.math("MULTIPLY", mask, 0.9), rgb + (1.0,), pat_rgb + (1.0,))
    sh = nb.principled(col, rough=0.75, spec=0.15, emission=col, strength=b["glow"], alpha=nb.attr("mk_alpha", "Fac"))
    nb.output(sh)
    return m


def _tape_mesh(g, P, b):
    """The strip: a grid centred on the word's ink box, the two short ends torn (every other vertex of the end columns
    pulled in by a random share of `torn` x the height), `UVMap` stored for the pattern, `mk_alpha` = the word's opacity."""
    w, h = b["size"]
    cx, cy = b["centre"]
    grid = g.node("GeometryNodeMeshGrid")
    grid.inputs["Size X"].default_value, grid.inputs["Size Y"].default_value = w, h
    grid.inputs["Vertices X"].default_value, grid.inputs["Vertices Y"].default_value = TAPE_COLS, TAPE_ROWS
    uv = g.node("GeometryNodeStoreNamedAttribute", data_type="FLOAT2", domain="CORNER")
    g.put(uv.inputs["Geometry"], grid.outputs["Mesh"])
    uv.inputs["Name"].default_value = "UVMap"
    g.put(uv.inputs["Value"], grid.outputs["UV Map"])
    pos = g.node("GeometryNodeInputPosition").outputs[0]
    sep = g.node("ShaderNodeSeparateXYZ")
    g.put(sep.inputs[0], pos)
    x, y = sep.outputs[0], sep.outputs[1]
    row = g.math("ROUND", g.math("MULTIPLY", g.math("ADD", g.math("DIVIDE", y, h), 0.5), TAPE_ROWS - 1))
    odd = g.math("MODULO", row, 2.0)
    end = g.math("GREATER_THAN", g.math("ABSOLUTE", x), 0.5 * w - 1e-6)
    left = g.math("LESS_THAN", x, 0.0)
    rnd = g.node("FunctionNodeRandomValue", data_type="FLOAT")
    rnd.inputs[2].default_value, rnd.inputs[3].default_value = 0.4, 1.0
    g.put(rnd.inputs["ID"], g.math("ADD", row, g.math("MULTIPLY", left, 100.0)))
    rnd.inputs["Seed"].default_value = b["seed"]
    # left end: odd teeth pulled in; right end: even teeth (the two ends never mirror each other)
    tooth = g.math("ADD", g.math("MULTIPLY", left, odd), g.math("MULTIPLY", g.math("SUBTRACT", 1.0, left),
                                                               g.math("SUBTRACT", 1.0, odd)))
    pull = g.math("MULTIPLY", g.math("MULTIPLY", g.math("MULTIPLY", end, tooth), rnd.outputs[1]), b["torn"] * h)
    dx = g.math("MULTIPLY", pull, g.math("SUBTRACT", 1.0, g.math("MULTIPLY", left, 2.0)))   # +x on the left, -x right
    off = g.node("ShaderNodeCombineXYZ")
    g.put(off.inputs[0], dx)
    sp = g.node("GeometryNodeSetPosition")
    g.put(sp.inputs["Geometry"], uv.outputs[0])
    g.put(sp.inputs["Offset"], off.outputs[0])
    mv = g.node("GeometryNodeTransform")
    g.put(mv.inputs["Geometry"], sp.outputs[0])
    mv.inputs["Translation"].default_value = (cx, cy, 0.0)
    st = g.node("GeometryNodeStoreNamedAttribute", data_type="FLOAT", domain="POINT")
    g.put(st.inputs["Geometry"], mv.outputs[0])
    st.inputs["Name"].default_value = "mk_alpha"
    g.put(st.inputs["Value"], getattr(g, "opacity", 1.0))
    return st.outputs[0]


def _tree_kinetic(name, P, string, mat):
    """The node group of a kinetic word: String to Curves (tracking keyed) -> typewriter -> Fill Curve on the character
    instances -> per-letter drip and the attributes the material reads -> Realize -> weight -> extrude -> placed ->
    keyed transform about the pivot -> only while `Show` says so."""
    kin = P["kin"]
    nt = _new_tree(name)
    g = _G(nt)
    stc = _string_to_curves(g, P["font"], P["align_x"], P["tracking"], P["words"])
    stc.inputs["Size"].default_value = P["em"]
    stc.inputs["Line Spacing"].default_value = P["ls"]
    g.put(stc.inputs["String"], string)
    if "tracking" in kin:
        g.put(stc.inputs["Character Spacing"], _val(g, NODE_NAMES["tracking"], kin["tracking"][0][1] * P["tracking"]))
    if "drip" in kin:
        stc.pivot_mode = "TOP_CENTER"                             # each letter stretches from the middle of its top
        g.piv = stc.outputs["Pivot Point"]
    curves = stc.outputs["Curve Instances"]
    if P["reveal"]:
        curves = _reveal(g, curves)
    fill = g.node("GeometryNodeFillCurve")
    g.put(fill.inputs["Curve"], curves)
    inst = _letters(g, P, fill.outputs[0], kin)
    real = g.node("GeometryNodeRealizeInstances")
    g.put(real.inputs["Geometry"], inst)
    mesh = real.outputs[0]
    if "weight" in kin:
        mesh = _grown(g, P, mesh, _val(g, NODE_NAMES["weight"], kin["weight"][0][1]))
    if P["depth"] > 0:
        ext = g.node("GeometryNodeExtrudeMesh", mode="FACES")
        g.put(ext.inputs["Mesh"], mesh)
        ext.inputs["Offset Scale"].default_value = P["depth"]
        mesh = ext.outputs["Mesh"]
    motion = _motion(g, P, kin)
    geo = _placed(g, P, mesh, mat, 0.0, motion)
    if P.get("outline"):
        o = P["outline"]
        join = g.node("GeometryNodeJoinGeometry")
        g.put(join.inputs["Geometry"], geo)
        g.put(join.inputs["Geometry"], _placed(g, P, _grown(g, P, mesh, o["width"]), o["material"], -o["dz"], motion))
        geo = join.outputs[0]
    if P.get("backing"):
        b = P["backing"]
        join = g.node("GeometryNodeJoinGeometry")
        g.put(join.inputs["Geometry"], geo)
        g.put(join.inputs["Geometry"], _placed(g, P, _tape_mesh(g, P, b), b["material"], -b["dz"], motion))
        geo = join.outputs[0]
    if "show" in kin:
        sw = g.node("GeometryNodeSwitch", input_type="GEOMETRY")
        g.put(sw.inputs[0], g.math("GREATER_THAN", _val(g, "Show", 0.0), 0.5))
        g.put(sw.inputs[2], geo)                                  # inputs: Switch, False (empty), True
        geo = sw.outputs[0]
    out = g.node("NodeGroupOutput")
    g.put(out.inputs[0], geo)
    return nt


def _key_node(ctx, nt, node, keys, interp="LINEAR", scale=1.0):
    """Keys on a Value node's output: frames rounded to 1e-4 (a CONSTANT key must sit on its frame, not 1e-13 after
    it); a list whose values are all one number just sets the node's default."""
    pts = {}
    for t, v in keys:
        pts[round(ctx.frame(t), 4)] = float(v) * scale
    frames = sorted(pts)
    vals = [pts[f] for f in frames]
    n = nt.nodes[node]
    if len(set(round(v, 9) for v in vals)) == 1:
        n.outputs[0].default_value = vals[0]
        return
    K.set_fcurve(nt, f'nodes["{node}"].outputs[0].default_value', 0, frames, vals, interp=interp)


def _key_kinetic(ctx, nt, kin, P):
    """Key every channel of a kinetic word onto its Value nodes; returns how many keys were written."""
    n_keys = 0
    for ch in FX.KEYED:
        if ch in kin:
            _key_node(ctx, nt, NODE_NAMES[ch], kin[ch], scale=P["tracking"] if ch == "tracking" else 1.0)
            n_keys += len(kin[ch])
    if "tint" in kin:
        _key_node(ctx, nt, NODE_NAMES["tint"], kin["tint"]["keys"])
        n_keys += len(kin["tint"]["keys"])
    if "show" in kin:
        on = kin["show"][0]
        keys = [(on - 1.0 / ctx.fps, 0.0), (on, 1.0)] + ([(kin["show"][1], 0.0)] if len(kin["show"]) > 1 else [])
        _key_node(ctx, nt, "Show", keys, "CONSTANT")
        n_keys += len(keys)
    d = kin.get("drip")
    if d:
        end = d["t"] + d["life"] + LY.DRIP_SPREAD + 0.1
        _key_node(ctx, nt, "Drip", [(d["t"] - 1.0 / ctx.fps, -1.0 / ctx.fps), (end, end - d["t"])])
        n_keys += 2
    return n_keys


# ------------------------------------------------------------------------------------------------------ stage
def _ghost(spec, strings):
    """(settings, string of unlit segments) or (None, None): `ghost = true | 0.10` or {strength, text, color}."""
    g = spec.get("ghost")
    if g is None or g is False or (not isinstance(g, (dict, bool)) and float(g) == 0.0):
        return None, None
    g = dict(g) if isinstance(g, dict) else {} if isinstance(g, bool) else {"strength": float(g)}
    return g, g.get("text") or T.ghost_string(max(strings, key=lambda s: (len(s), s)))


def _fade(spec):
    """(keys [(t, gain)], interp) of `fade = [[t, gain], ...]` or {keys, interp}, or (None, None)."""
    f = spec.get("fade")
    if f is None:
        return None, None
    keys, interp = (f.get("keys"), f.get("interp", "LINEAR")) if isinstance(f, dict) else (f, "LINEAR")
    if not keys:
        raise BuildError(f"text {spec['name']!r}: fade needs keys = [[t, gain], ...]")
    if interp not in K.INTERP:
        raise BuildError(f"text {spec['name']!r}: fade interp {interp!r} must be one of {sorted(K.INTERP)}")
    return sorted((float(t), float(g)) for t, g in keys), interp


def _halo_spec(spec):
    """{strength, size} of `halo = true | {strength, size}`, or None."""
    h = spec.get("halo")
    if h is None or h is False:
        return None
    h = dict(h) if isinstance(h, dict) else {}
    return {"strength": float(h.get("strength", HALO_STRENGTH)), "size": float(h.get("size", HALO_SIZE))}


def _measurer(coll, cache, font, font_label, align, tracking, words):
    """The shared _Measure of a font / alignment / spacing combination."""
    tag = f"{font_label}|{align}|{tracking:g}|{words:g}"
    if tag not in cache["measure"]:
        cache["measure"][tag] = _Measure(coll, font, ALIGN_X[align], tracking, words, str(len(cache["measure"])))
    return cache["measure"][tag]


def _one(ctx, coll, spec, cache):
    name = spec.get("name")
    if not name:
        raise BuildError("[[text]] needs a name")
    unknown = sorted(set(spec) - KNOWN)
    if unknown:
        raise BuildError(f"text {name!r}: unknown keys {unknown} (known: {sorted(KNOWN)})")
    if len(name) > MAX_NAME:                    # Blender cuts names at 63 characters: two texts would end up sharing one
        raise BuildError(f"text {name!r}: the name is {len(name)} characters, over {MAX_NAME} (the object is 63 at most and its "
                         "node group adds a prefix); a lyric word's is <name>_l<line>w<word>[_<shot>][@<output>]")
    if bpy.data.objects.get(name) is not None:
        raise BuildError(f"text {name!r}: an object of that name exists already")
    kin = FX.kinetic_spec(spec)
    backing = FX.backing_spec(spec)
    if backing and SR.screen_spec(spec) is not None:
        if "glow" not in (spec["backing"] if isinstance(spec["backing"], dict) else {}):
            backing["glow"] = 1.0                   # the screen layer has no lights: the paper is its own emission
    outline = FX.outline_spec(spec)
    if (backing or outline) and kin is None:      # a strip or ring behind a plain text: the kinetic tree builds it too
        kin = FX.kinetic_spec({**spec, "kinetic": {}})
    lyric = spec.get("lyric")
    parent, inverse, F, panel, owner, label = _place(ctx, spec)
    strings, number = _content(ctx, spec)
    reveal = spec.get("reveal")
    if reveal is not None and not (isinstance(reveal, dict) and "from" in reveal and "to" in reveal):
        raise BuildError(f"text {name!r}: reveal needs {{from = t, to = t}} (clip seconds)")
    glow = float(spec.get("glow", 1.0))
    if ("blink" in spec or "flicker" in spec or "fade" in spec) and glow <= 0:
        raise BuildError(f"text {name!r}: blink, flicker and fade modulate the glow; set glow > 0")
    font, font_label = _font(ctx, spec.get("font"), cache)
    align, valign = T.parse_align(spec.get("align"), spec.get("valign"))
    tracking, words = float(spec.get("tracking", 1.0)), float(spec.get("word_spacing", 1.0))
    meas = _measurer(coll, cache, font, font_label, align, tracking, words)
    cap_ratio = meas.cap_ratio()
    ls = float(spec.get("leading", 1.5)) * cap_ratio

    # layout: the ink of everything the text can show (and of its ghost segments) decides size and place
    ghost, ghost_text = _ghost(spec, strings)
    boxes = [b for b in (meas.ink(s, ls) for s in strings + ([ghost_text] if ghost_text else [])) if b]
    if not boxes:
        shown = f"the word ({lyric[0]}, {lyric[1]})" if lyric is not None else repr(strings[0])
        raise BuildError(f"text {name!r}: the font draws nothing for {shown}")
    ref0 = (min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes))
    ref = ref0
    if kin and kin.get("headroom"):          # the letters spread: the fit leaves room for the widest the word gets
        h = 1.0 + kin["headroom"]
        ref = (ref0[0] * h, ref0[1], ref0[2] * h, ref0[3])
    lay = T.plan_layout(ref, cap_ratio, panel=panel, size=spec.get("size"), fit=spec.get("fit"), align=align,
                        valign=valign, offset=spec.get("offset", (0.0, 0.0)))
    warn = None
    if not lay.fits:
        warn = f"ink {lay.ink[0]:.3f} x {lay.ink[1]:.3f} m is larger than the {panel[0]:.3f} x {panel[1]:.3f} m panel"
        ctx.log(f"WARNING text {name}: {warn}")

    # look
    try:
        rgb = NK.rgb(ctx.palette, spec.get("color", "text"))
        grgb = None
        if ghost is not None:
            lit = NK.hexof(ctx.palette, spec.get("color", "text"))
            grgb = NK.rgb(ctx.palette, ghost["color"] if "color" in ghost
                          else T.ghost_colour(lit, ctx.palette["muted"]))
        to_rgb = NK.rgb(ctx.palette, kin["tint"]["color"]) if kin and "tint" in kin else None
        if backing:
            backing["rgb"] = NK.rgb(ctx.palette, backing["color"])
            backing["pattern_rgb"] = NK.rgb(ctx.palette, backing["pattern_color"])
        if outline:
            outline["rgb"] = NK.rgb(ctx.palette, outline["color"])
    except KeyError as e:
        raise BuildError(f"text {name!r}: {e.args[0]}")
    haze = _haze(ctx, spec, owner)
    fade, fade_interp = _fade(spec)
    gains = ("Gain", "Fade") if fade else ("Gain",)
    lit = float(spec.get("lit", 1.0))
    if not 0.0 <= lit <= 1.0:
        raise BuildError(f"text {name!r}: lit is a share of the scene's light, 0..1")
    if kin is not None:
        mats = {"lit": _kinetic_material(cache, rgb, glow, to_rgb, haze, lit)}
    else:
        mats = {"lit": _material(f"{name}_text", rgb, glow, haze, gains, lit)}
    if ghost is not None:
        g_glow = glow * T.display_share(ghost.get("strength", GHOST_STRENGTH))
        mats["ghost"] = _additive(f"{name}_ghost", grgb, g_glow, ("Fade",) if fade else ())
    halo = _halo_spec(spec)
    if halo is not None:
        mats["halo"] = _additive(f"{name}_halo", rgb, glow * halo["strength"] / HALO_COPIES, gains)

    # object: a modifier on an empty mesh, riding its owner
    P = {"font": font, "align_x": ALIGN_X[align], "tracking": tracking, "words": words, "em": lay.em, "ls": ls,
         "tx": lay.tx, "ty": lay.ty, "depth": float(spec.get("depth", 0.0)), "reveal": bool(reveal),
         "ghost_text": ghost_text, "halo_r": (halo["size"] * lay.cap) if halo else 0.0}
    if kin is not None:
        cx, cy = lay.tx + 0.5 * (ref0[0] + ref0[2]) * lay.em, lay.ty
        cy += {"center": 0.5 * (ref0[1] + ref0[3]), "bottom": ref0[1], "top": ref0[3]}[kin.get("pivot", "center")] * lay.em
        P.update(kin=kin, pivot=(cx, cy), panel=panel)
        if outline:
            outline["material"] = _kinetic_material(cache, outline["rgb"], 1.0, None, haze, 0.0, outline["alpha"])
            P["outline"] = outline
        if backing:
            ink_w = (ref0[2] - ref0[0]) * (1.0 + 0.8 * (kin.get("headroom") or 0.0))
            w, h = FX.tape_size(backing, ink_w, ref0[3] - ref0[1], lay.em)
            backing.update(size=(w, h), centre=(0.5 * (ref0[0] + ref0[2]) * lay.em, 0.5 * (ref0[1] + ref0[3]) * lay.em))
            backing["material"] = _tape_material(f"{name}_tape", backing, backing["rgb"], backing["pattern_rgb"], w, h)
            P["backing"] = backing
        nt = _tree_kinetic(f"mk_text_{name}", P, strings[0], mats["lit"])
    else:
        nt = _tree(f"mk_text_{name}", P, strings[0] if number is None else number, mats)
    ob = bpy.data.objects.new(name, bpy.data.meshes.new(name))
    coll.objects.link(ob)
    ob.modifiers.new("Text", "NODES").node_group = nt
    M = Matrix([list(map(float, r)) for r in F])
    if parent is None:
        ob.matrix_world = M
    else:
        ob.parent = parent
        if inverse is not None:
            ob.matrix_parent_inverse = inverse
        ob.matrix_basis = M
    ob.visible_shadow = P["depth"] > 0
    if SR.screen_spec(spec) is not None:           # type of the picture: drawn by the screen layer, never by a scene render
        ob["mk_screen"] = 1
        ob.hide_render = ob.hide_viewport = True
        show = (kin or {}).get("show")             # the frames it is on screen: the layer skips the pass without it
        if show:
            ob["mk_show"] = [int(round(ctx.frame(show[0]))),
                             int(round(ctx.frame(show[1]))) if len(show) > 1 else 1_000_000]
    if lyric is not None:                  # which sung word this is, as numbers (the text itself lives in the nodes)
        ob["mk_lyric"] = [int(lyric[0]), int(lyric[1])]
    if spec.get("_aspect") is not None and spec.get("_multi"):     # rendered in that output only (scene.bind_aspect)
        ob["mk_aspect"] = str(spec["_aspect"])
    if spec.get("knockout"):                       # reversed over silhouette shots whatever their `knockout` table says
        ob["mk_knockout"] = 1
    if spec.get("_shot") is not None:
        ob["mk_shot"] = str(spec["_shot"])

    # time: keys on node inputs (a number, the typewriter ramp, the emission gain, kinetic channels), nothing else moves
    n_keys = 0
    if number is not None:
        _key(nt, 'nodes["Value"].outputs[0].default_value', number["keys"], ctx, number["interp"])
    if reveal:
        _key(nt, 'nodes["Reveal"].outputs[0].default_value',
             T.reveal_keys(float(reveal["from"]), float(reveal["to"]), ctx.fps), ctx, "LINEAR")
    if kin is not None:
        n_keys = _key_kinetic(ctx, nt, kin, P)
    gain = None
    for kind, fn in (("blink", T.blink_keys), ("flicker", T.flicker_keys)):
        if kind in spec:
            keys = fn(spec[kind], ctx.time(ctx.start), ctx.time(ctx.end))
            gain = keys if gain is None else T.product_keys(gain, keys)
    if gain:
        _key(mats["lit"].node_tree, 'nodes["Gain"].outputs[0].default_value', gain, ctx, "CONSTANT")
        if "halo" in mats:
            _key(mats["halo"].node_tree, 'nodes["Gain"].outputs[0].default_value', gain, ctx, "CONSTANT")
    for m in mats.values():
        if fade:
            _key(m.node_tree, 'nodes["Fade"].outputs[0].default_value', fade, ctx, fade_interp)

    ctx.texts = getattr(ctx, "texts", {})
    ctx.texts[name] = ob
    info = {"on": label, "font": font_label, "cap_mm": round(lay.cap * 1000, 2), "em_mm": round(lay.em * 1000, 2),
            "ink_m": [round(v, 4) for v in lay.ink], "panel_m": list(panel) if panel else None, "fits": lay.fits,
            "strings_measured": len(strings)}
    if lyric is None:                    # a word of lyrics never leaves its text in a report
        widest = max(strings, key=len).replace("\n", " / ")
        info["widest"] = widest if len(widest) <= 40 else widest[:37] + "..."
    else:
        info["lyric"] = list(lyric)
        info["chars"] = len(strings[0])
    if kin:
        info["kinetic_keys"] = n_keys
        if "show" in kin:
            info["frames"] = [int(round(ctx.frame(kin["show"][0]))),
                              int(round(ctx.frame(kin["show"][1]))) if len(kin["show"]) > 1 else None]
    if warn:
        info["warning"] = warn
    if number is not None:
        info["value_keys"] = len(number["keys"])
    return info


def _measure_fn(ctx, coll, cache, spec):
    """callable(string[, font[, tracking]]) -> {w, y0, y1, cap}: the ink box of a string at em size 1 in the font (the entry's,
    or the registry slug / file `font`) and tracking (the entry's, or `tracking`), word spacing and leading of `spec`
    (centred), for the lyrics stage's layouts. Nothing about the string reaches an error message."""
    words = float(spec.get("word_spacing", 1.0))
    by_look = {}

    def tools(ref, tracking):
        if (ref, tracking) not in by_look:
            font, label = _font(ctx, ref, cache)
            meas = _measurer(coll, cache, font, label, "center", tracking, words)
            cap = meas.cap_ratio()
            by_look[(ref, tracking)] = (meas, cap, float(spec.get("leading", 1.5)) * cap)
        return by_look[(ref, tracking)]

    def measure(s, font=None, tracking=None):
        meas, cap, ls = tools(spec.get("font") if font is None else font,
                              float(spec.get("tracking", 1.0)) if tracking is None else float(tracking))
        b = meas.ink(s, ls)
        if b is None:
            raise T.TextError("the font draws nothing for one of the words")
        return {"w": b[2] - b[0], "y0": b[1], "y1": b[3], "cap": cap}

    return measure


def _outputs(ctx):
    return [o["name"] for o in (ctx.project.get("outputs") or [])] or ["main"]


def run(ctx):
    try:                                                  # `extends` / `abstract`: a look shared by several entries
        specs = SR.resolve_entries(ctx.data.get("text", []))
    except T.TextError as e:
        raise BuildError(str(e)) from None
    if not specs:
        return {}
    coll = collection("Text")
    cache = {"fonts": {}, "measure": {}, "kmats": {}}
    outputs = _outputs(ctx)
    out = {}
    try:
        for spec in specs:
            if spec.get("ink") is not None:               # handwriting ink on a surface: build/ink.py
                from . import ink as INK
                out[spec.get("name", "ink")] = INK.build(ctx, coll, spec)
                ctx.log("ink", spec.get("name", "ink"), out[spec.get("name", "ink")].get("strokes"))
                continue
            try:                                          # `screen` / `aspect.<output>`: one set of objects per output
                variants = SR.per_output(spec, outputs)
            except T.TextError as e:
                raise BuildError(f"text {spec.get('name', '?')!r}: {e}") from None
            for asp, vspec in variants:
                suffix = f"@{asp}" if asp is not None and len(outputs) > 1 else ""
                if asp is not None:
                    vspec = {**vspec, "_aspect": asp, "_multi": len(outputs) > 1}
                if vspec.get("lyrics") is not None:       # lyric type: one text per word, built from the timeline
                    from . import wordtype as LYB
                    report = {}
                    try:
                        subs = LYB.expand(ctx, vspec, _measure_fn(ctx, coll, cache, vspec), report)
                    except T.TextError as e:
                        raise BuildError(f"text {vspec.get('name', '?')!r}: {e}") from None
                    out[f"{vspec['name']}{suffix}"] = {"lyrics": report}
                else:
                    subs = [vspec]
                for sub in subs:
                    if suffix:
                        sub = {**sub, "name": f"{sub.get('name', '?')}{suffix}"}
                    name = sub.get("name", "?")
                    try:
                        out[name] = _one(ctx, coll, sub, cache)
                    except T.TextError as e:
                        raise BuildError(f"text {name!r}: {e}") from None
                    except BuildError as e:
                        if not str(e).startswith("text "):
                            raise BuildError(f"text {name!r}: {e}") from None
                        raise
                ctx.log("text", f"{vspec.get('name', '?')}{suffix}",
                        f"{len(subs)} object(s)" if len(subs) != 1 else out[subs[0]["name"] + suffix]["on"])
    finally:
        for m in cache["measure"].values():
            m.close()
    bpy.context.view_layer.update()
    return out
