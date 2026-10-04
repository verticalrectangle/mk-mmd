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
  color = "text"            palette slot or #hex (slot:slot:t mixes two); glow = 1.0  emission strength
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
The text frame is the card's: x right (up x normal), y up, z out of the surface."""
import json
import os

import bpy
import numpy as np
from mathutils import Matrix

from ...core import typeset as T
from .. import keys as K
from ..library.sets import highway as HIGHWAY
from ..library.sets import nightkit as NK
from . import BuildError, collection

KNOWN = {"name", "on", "mount", "at", "facing", "up", "box", "text", "value", "font", "size", "fit", "align", "valign",
         "offset", "color", "glow", "depth", "lift", "tracking", "word_spacing", "leading", "reveal", "blink",
         "flicker", "fade", "ghost", "halo", "haze"}
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


def _place(ctx, spec):
    """Where the text goes: parent object, its parent-inverse (None = identity), the frame (4x4, in the parent's frame
    or the world), the panel (w, h) or None, the owner (kind, name) and a label."""
    name = spec["name"]
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
        F = T.surface_matrix(s["center"], s["normal"], s.get("up", (0.0, 0.0, 1.0)), lift)
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
def _material(name, rgb, glow, haze, gains=("Gain",)):
    """Principled + emission, strength = glow x one keyable Value node per name in `gains` (Gain: blink and flicker,
    Fade: a ramp), faded into the haze with distance."""
    m, nb = NK.new_material(name)
    strength = float(glow)
    for gname in gains:
        v = nb.node("ShaderNodeValue")
        v.name = v.label = gname
        v.outputs[0].default_value = 1.0
        strength = nb.math("MULTIPLY", v.outputs[0], strength)
    sh = nb.principled(rgb, rough=0.55, spec=0.3, emission=rgb, strength=strength)
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


# ------------------------------------------------------------------------------------------------------ geometry
def _text_mesh(g, P, string, reveal):
    """String -> instances (-> typewriter) -> filled mesh (-> extruded): the mesh socket, in the text plane."""
    stc = _string_to_curves(g, P["font"], P["align_x"], P["tracking"], P["words"])
    stc.inputs["Size"].default_value = P["em"]
    stc.inputs["Line Spacing"].default_value = P["ls"]
    g.put(stc.inputs["String"], string)
    curves = stc.outputs["Curve Instances"]
    if reveal:
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
        curves = dele.outputs[0]
    mesh = _filled(g, curves)
    if P["depth"] > 0:
        ext = g.node("GeometryNodeExtrudeMesh", mode="FACES")
        g.put(ext.inputs["Mesh"], mesh)
        ext.inputs["Offset Scale"].default_value = P["depth"]
        mesh = ext.outputs["Mesh"]
    return mesh


def _placed(g, P, mesh, mat, dz):
    """The mesh moved to its place in the panel (and `dz` along the normal), with its material."""
    tr = g.node("GeometryNodeTransform")
    g.put(tr.inputs["Geometry"], mesh)
    tr.inputs["Translation"].default_value = (P["tx"], P["ty"], dz)
    sm = g.node("GeometryNodeSetMaterial")
    g.put(sm.inputs["Geometry"], tr.outputs[0])
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


def _one(ctx, coll, spec, cache):
    name = spec.get("name")
    if not name:
        raise BuildError("[[text]] needs a name")
    unknown = sorted(set(spec) - KNOWN)
    if unknown:
        raise BuildError(f"text {name!r}: unknown keys {unknown} (known: {sorted(KNOWN)})")
    if bpy.data.objects.get(name) is not None:
        raise BuildError(f"text {name!r}: an object of that name exists already")
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
    tag = f"{font_label}|{align}|{tracking:g}|{words:g}"
    if tag not in cache["measure"]:
        cache["measure"][tag] = _Measure(coll, font, ALIGN_X[align], tracking, words, str(len(cache["measure"])))
    meas = cache["measure"][tag]
    cap_ratio = meas.cap_ratio()
    ls = float(spec.get("leading", 1.5)) * cap_ratio

    # layout: the ink of everything the text can show (and of its ghost segments) decides size and place
    ghost, ghost_text = _ghost(spec, strings)
    boxes = [b for b in (meas.ink(s, ls) for s in strings + ([ghost_text] if ghost_text else [])) if b]
    if not boxes:
        raise BuildError(f"text {name!r}: the font draws nothing for {strings[0]!r}")
    ref = (min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes))
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
    except KeyError as e:
        raise BuildError(f"text {name!r}: {e.args[0]}")
    haze = _haze(ctx, spec, owner)
    fade, fade_interp = _fade(spec)
    gains = ("Gain", "Fade") if fade else ("Gain",)
    mats = {"lit": _material(f"{name}_text", rgb, glow, haze, gains)}
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

    # time: keys on node inputs (a number, the typewriter ramp, the emission gain), nothing else moves
    if number is not None:
        _key(nt, 'nodes["Value"].outputs[0].default_value', number["keys"], ctx, number["interp"])
    if reveal:
        _key(nt, 'nodes["Reveal"].outputs[0].default_value',
             T.reveal_keys(float(reveal["from"]), float(reveal["to"]), ctx.fps), ctx, "LINEAR")
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
    widest = max(strings, key=len).replace("\n", " / ")
    info = {"on": label, "font": font_label, "cap_mm": round(lay.cap * 1000, 2), "em_mm": round(lay.em * 1000, 2),
            "ink_m": [round(v, 4) for v in lay.ink], "panel_m": list(panel) if panel else None, "fits": lay.fits,
            "widest": widest if len(widest) <= 40 else widest[:37] + "...", "strings_measured": len(strings)}
    if warn:
        info["warning"] = warn
    if number is not None:
        info["value_keys"] = len(number["keys"])
    return info


def run(ctx):
    specs = ctx.data.get("text", [])
    if not specs:
        return {}
    coll = collection("Text")
    cache = {"fonts": {}, "measure": {}}
    out = {}
    try:
        for spec in specs:
            name = spec.get("name", "?")
            try:
                out[name] = _one(ctx, coll, spec, cache)
            except T.TextError as e:
                raise BuildError(f"text {name!r}: {e}") from None
            except BuildError as e:
                if not str(e).startswith("text "):
                    raise BuildError(f"text {name!r}: {e}") from None
                raise
            ctx.log("text", name, out[name]["on"])
    finally:
        for m in cache["measure"].values():
            m.close()
    bpy.context.view_layer.update()
    return out
