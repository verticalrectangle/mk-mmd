"""blocktype: type set in 3D toy alphabet blocks (docs/design.md: Text, Block type). Called by the text stage for a
[[text]] with `lyrics = {style = "blocks", ...}` (the sung words) or with `text` and `blocks = {...}` (a label).

mkmmd/core/blocktype.py decides the blocks (one per letter or digit), what their faces show, where they stand, how big they
are and how each moves; this module builds them:

  an empty `<name>` (`@<output>` for one output of several) where the text's panel is, turned by `blocks.view` and scaled by
       the block's edge, so the blocks below it live in edges; `[[key]]` can move the whole set by its name
  per block a mesh object `<name>_l<line>w<word>c<letter>` (a label's `<name>_c<k>`) under it, keyed: location, rotation
       (quaternion) and scale on the frames it moves (LINEAR, the straight runs dropped), its scale 0 before it drops in
       and after it is gone (CONSTANT steps), so a render needs no Python; it carries `mk_lyric = [line, word]` and
       `mk_block = letter`, and as screen type `mk_screen` and `mk_show` like any screen text
  one mesh per distinct block (a rounded cube with its glyphs, rim and pictures), the glyphs' filled outlines read off a
       throwaway String to Curves / Fill Curve object in the entry's font, so the letters become geometry and no string
       is left in the .blend
  palette materials shared by colour: the paint shades itself from the camera's side (a light over the viewer's left
       shoulder, in camera space) as emission times `glow`, so screen type, which is drawn with the lights off, still
       reads as solid blocks; in the scene `lit` adds the lit Principled surface on top

The timeline is read here and nowhere else; the letters never reach a log line, an error, a report or an object property."""
import bpy
import numpy as np
from mathutils import Matrix

from ...core import blocktype as BT
from ...core import lanes as LN
from ...core import palette as PAL
from ...core import screentype as SR
from ...core import typeset as T
from .. import keys as K
from ..library.sets import nightkit as NK
from . import BuildError

is_blocks = BT.is_blocks

LIGHT = (-0.3, 0.75, 0.55)            # the self-shading's light, in camera space: over the viewer's left shoulder
SHADE = (0.55, 1.12)                  # what a face lit from behind and one facing the light show of their colour
GLOSS = (0.22, 28.0)                  # the paint's highlight: strength, sharpness
ROUGH = {"wood": 0.6, "paint": 0.3, "ink": 0.3, "picture": 0.4}
GLOW = {True: 1.0, False: 0.3}        # default emission of the self-shaded paint, screen type / in the scene
LIT = {True: 0.0, False: 1.0}         # default share of the scene's light (screen type is drawn with the lights off)
MAX_OBJECT = 63                       # bytes Blender keeps of a name (UTF-8: a kana or kanji takes 3)


# ------------------------------------------------------------------------------------------------------ glyphs
class _Glyphs:
    """Each character's filled outline in a font at em size 1 (String to Curves -> Realize -> Fill Curve), read off a
    throwaway geometry-nodes object: (V2, CCW triangles) or None when the font draws nothing for it."""

    def __init__(self, coll, font, tag):
        self.nt = bpy.data.node_groups.new(f"mk_block_glyph_{tag}", "GeometryNodeTree")
        nt = self.nt
        nt.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
        self.sid = nt.interface.new_socket("String", in_out="INPUT", socket_type="NodeSocketString").identifier
        gin = nt.nodes.new("NodeGroupInput")
        stc = nt.nodes.new("GeometryNodeStringToCurves")
        stc.align_x, stc.align_y = "LEFT", "TOP_BASELINE"
        if font is not None:
            stc.font = font
        stc.inputs["Size"].default_value = 1.0
        nt.links.new(gin.outputs["String"], stc.inputs["String"])
        real = nt.nodes.new("GeometryNodeRealizeInstances")
        nt.links.new(stc.outputs["Curve Instances"], real.inputs["Geometry"])
        fill = nt.nodes.new("GeometryNodeFillCurve")
        fill.mode = "TRIANGLES"
        nt.links.new(real.outputs[0], fill.inputs["Curve"])
        out = nt.nodes.new("NodeGroupOutput")
        nt.links.new(fill.outputs[0], out.inputs[0])
        self.me = bpy.data.meshes.new(f"mk_block_glyph_{tag}")
        self.ob = bpy.data.objects.new(f"mk_block_glyph_{tag}", self.me)
        coll.objects.link(self.ob)
        self.mod = self.ob.modifiers.new("Glyph", "NODES")
        self.mod.node_group = nt
        self._cache = {}
        self._cap = None

    def get(self, c):
        if c in self._cache:
            return self._cache[c]
        self.mod[self.sid] = c
        self.ob.update_tag()
        ev = self.ob.evaluated_get(bpy.context.evaluated_depsgraph_get())
        me = ev.to_mesh()
        me.calc_loop_triangles()
        res = None
        if len(me.loop_triangles):
            co = np.empty(len(me.vertices) * 3, np.float32)
            me.vertices.foreach_get("co", co)
            tri = np.empty(len(me.loop_triangles) * 3, np.int32)
            me.loop_triangles.foreach_get("vertices", tri)
            V2 = co.reshape(-1, 3)[:, :2].astype(float)
            T3 = tri.reshape(-1, 3)
            a, b = V2[T3[:, 1]] - V2[T3[:, 0]], V2[T3[:, 2]] - V2[T3[:, 0]]
            cw = a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0] < 0
            T3[cw] = T3[cw][:, ::-1]                       # every triangle counter-clockwise, facing out of the face
            res = (V2, T3)
        ev.to_mesh_clear()
        self._cache[c] = res
        return res

    def draws(self, c):
        return self.get(c) is not None

    def cap(self):
        """The height of a capital per em (E, else H): what a block's letter is sized by."""
        if self._cap is None:
            for c in ("E", "H"):
                g = self.get(c)
                if g is not None:
                    self._cap = float(g[0][:, 1].max() - g[0][:, 1].min())
                    break
            else:
                raise T.TextError("the font has neither E nor H to take a cap height from")
        return self._cap

    def close(self):
        for coll, item in ((bpy.data.objects, self.ob), (bpy.data.meshes, self.me), (bpy.data.node_groups, self.nt)):
            coll.remove(item)


# ------------------------------------------------------------------------------------------------------ materials
def _material(cache, palette, colour, role, glow, lit):
    """The shared material of one colour: the colour shaded from the camera's side (a fixed light in camera space, a
    gloss highlight) as emission times `glow`, plus `lit` of a Principled surface lit by the scene."""
    hexcol = NK.hexof(palette, colour)
    key = (hexcol, role, round(glow, 4), round(lit, 4))
    mats = cache.setdefault("block_mats", {})
    if key in mats:
        return mats[key]
    m = bpy.data.materials.new(f"mk_block_{role}_{hexcol.lstrip('#')}")
    m.use_nodes = True
    nt = m.node_tree
    for n in list(nt.nodes):
        nt.nodes.remove(n)
    rgb = PAL.linear(hexcol)
    geo = nt.nodes.new("ShaderNodeNewGeometry")
    vt = nt.nodes.new("ShaderNodeVectorTransform")
    vt.vector_type, vt.convert_from, vt.convert_to = "NORMAL", "WORLD", "CAMERA"
    nt.links.new(geo.outputs["Normal"], vt.inputs[0])
    light = np.asarray(LIGHT, float) / np.linalg.norm(LIGHT)
    half = light + np.array([0.0, 0.0, 1.0])
    half /= np.linalg.norm(half)
    dot = nt.nodes.new("ShaderNodeVectorMath")
    dot.operation = "DOT_PRODUCT"
    nt.links.new(vt.outputs[0], dot.inputs[0])
    dot.inputs[1].default_value = tuple(light)
    ramp = nt.nodes.new("ShaderNodeMapRange")
    ramp.inputs["From Min"].default_value, ramp.inputs["From Max"].default_value = -1.0, 1.0
    ramp.inputs["To Min"].default_value, ramp.inputs["To Max"].default_value = SHADE
    nt.links.new(dot.outputs["Value"], ramp.inputs["Value"])
    hdot = nt.nodes.new("ShaderNodeVectorMath")
    hdot.operation = "DOT_PRODUCT"
    nt.links.new(vt.outputs[0], hdot.inputs[0])
    hdot.inputs[1].default_value = tuple(half)
    clamp = nt.nodes.new("ShaderNodeMath")
    clamp.operation = "MAXIMUM"
    nt.links.new(hdot.outputs["Value"], clamp.inputs[0])
    clamp.inputs[1].default_value = 0.0
    power = nt.nodes.new("ShaderNodeMath")
    power.operation = "POWER"
    nt.links.new(clamp.outputs[0], power.inputs[0])
    power.inputs[1].default_value = GLOSS[1]
    gloss = nt.nodes.new("ShaderNodeMath")
    gloss.operation = "MULTIPLY"
    nt.links.new(power.outputs[0], gloss.inputs[0])
    gloss.inputs[1].default_value = GLOSS[0] if role != "wood" else 0.5 * GLOSS[0]
    shaded = nt.nodes.new("ShaderNodeVectorMath")
    shaded.operation = "SCALE"
    shaded.inputs[0].default_value = rgb
    nt.links.new(ramp.outputs["Result"], shaded.inputs["Scale"])
    shine = nt.nodes.new("ShaderNodeCombineXYZ")
    for i in range(3):
        nt.links.new(gloss.outputs[0], shine.inputs[i])
    colour = nt.nodes.new("ShaderNodeVectorMath")
    colour.operation = "ADD"
    nt.links.new(shaded.outputs["Vector"], colour.inputs[0])
    nt.links.new(shine.outputs[0], colour.inputs[1])
    em = nt.nodes.new("ShaderNodeEmission")
    nt.links.new(colour.outputs["Vector"], em.inputs["Color"])
    em.inputs["Strength"].default_value = float(glow)
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    shader = em.outputs[0]
    if lit > 0:
        bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
        bsdf.inputs["Base Color"].default_value = (*rgb, 1.0)
        bsdf.inputs["Roughness"].default_value = ROUGH[role]
        lit_out = bsdf.outputs[0]
        if lit < 1:
            dark = nt.nodes.new("ShaderNodeEmission")
            dark.inputs["Strength"].default_value = 0.0
            mix = nt.nodes.new("ShaderNodeMixShader")
            mix.inputs["Fac"].default_value = float(lit)
            nt.links.new(dark.outputs[0], mix.inputs[1])
            nt.links.new(bsdf.outputs[0], mix.inputs[2])
            lit_out = mix.outputs[0]
        add = nt.nodes.new("ShaderNodeAddShader")
        nt.links.new(shader, add.inputs[0])
        nt.links.new(lit_out, add.inputs[1])
        shader = add.outputs[0]
    nt.links.new(shader, out.inputs["Surface"])
    m.diffuse_color = (*rgb, 1.0)
    mats[key] = m
    return m


# ------------------------------------------------------------------------------------------------------ meshes
def _mesh(name, m, n_slots):
    """A Blender mesh of a block (mkmmd.core.blocktype.Mesh): its polygons, the normal of every corner, `n_slots`
    material slots that each object fills with its own colours."""
    me = bpy.data.meshes.new(name)
    me.vertices.add(len(m.V))
    me.vertices.foreach_set("co", m.V.astype(np.float32).ravel())
    me.loops.add(len(m.loops))
    me.loops.foreach_set("vertex_index", m.loops.astype(np.int32))
    starts = np.concatenate([[0], np.cumsum(m.sizes)[:-1]]).astype(np.int32)
    me.polygons.add(len(m.sizes))
    me.polygons.foreach_set("loop_start", starts)
    me.polygons.foreach_set("loop_total", m.sizes.astype(np.int32))
    me.polygons.foreach_set("material_index", m.mats.astype(np.int32))
    me.update(calc_edges=True)
    me.shade_smooth()
    me.normals_split_custom_set([tuple(n) for n in m.normals.tolist()])
    for _ in range(n_slots):
        me.materials.append(None)
    return me


# ------------------------------------------------------------------------------------------------------ keys
def _key(ob, path, index, frames, values, steps=()):
    """LINEAR keys on whole frames, the straight runs dropped (mkmmd.core.blocktype.thin), the keys at `steps` (indices
    into frames) CONSTANT."""
    frames, values = np.asarray(frames, float), np.asarray(values, float)
    keep = BT.thin(frames, values)
    keep = np.union1d(keep, np.asarray(steps, int)) if len(steps) else keep
    fc = K.set_fcurve(ob, path, index, frames[keep], values[keep], interp="LINEAR")
    for i in steps:
        fc.keyframe_points[int(np.searchsorted(keep, i))].interpolation = "CONSTANT"


def _key_block(ctx, ob, b):
    """A block's keys: location and rotation on its frames, its scale stepped from 0 to the first key and back to 0 on the
    frame it is gone from."""
    k = b.keys
    f = ctx.frame0 + k["frames"].astype(float)
    for i in range(3):
        _key(ob, "location", i, f, k["loc"][:, i])
    for i in range(4):
        _key(ob, "rotation_quaternion", i, f, k["quat"][:, i])
    fs = np.concatenate([[f[0] - 1.0], f, [ctx.frame0 + b.off]])
    n = len(fs)
    for i in range(3):
        vs = np.concatenate([[0.0], k["scale"][:, i], [0.0]])
        _key(ob, "scale", i, fs, vs, steps=(0, n - 2))


# ------------------------------------------------------------------------------------------------------ build
def _hits(ctx, name, opts):
    if not opts["hits"]:
        return None
    rel = (ctx.data.get("audio") or {}).get("hits")
    if not rel:
        raise BuildError(f"text {name!r}: blocks.hits reads the [audio] hits file and the project names none (`mk timeline "
                         f"hits` writes it)")
    try:
        return LN.load(ctx.path(rel))
    except LN.LaneError as e:
        raise BuildError(f"text {name!r}: blocks.hits: {e}") from None


def _glyphs(coll, cache, font, font_label):
    """The glyph reader of a font, closed with the text stage's measurers at the end of the stage."""
    key = ("blocks", font_label)
    if key not in cache["measure"]:
        cache["measure"][key] = _Glyphs(coll, font, f"{len(cache['measure'])}")
    return cache["measure"][key]


def build(ctx, coll, spec, suffix, cache):
    """Build a [[text]] set in blocks (one output's version of it, `suffix` its `@<output>`); returns the stage report:
    the plan's summary in Blender frames, where it stands, the font, the edge and the words' frames, never their text."""
    from . import text as TX
    name = spec.get("name", "?")
    try:
        screen = SR.screen_spec(spec) is not None
        if SR.negative_spec(spec):
            raise T.TextError("negative does not apply to blocks (toy blocks are painted solids: their coverage is a "
                              "shape, not type to invert); drop negative or blocks")
        opts = BT.options(spec, screen)
    except (BT.BlockError, T.TextError) as e:
        msg = str(e)
        raise BuildError(msg if msg.startswith("text ") else f"text {name!r}: {msg}") from None
    parent, inverse, F, panel, owner, label = TX._place(ctx, spec)
    top = bottom = None
    if screen:                                         # the picture's top and bottom edges in the panel frame
        top, bottom = 0.5 - float(F[1][3]), -0.5 - float(F[1][3])
    tl = None
    if spec.get("lyrics") is not None:
        from . import wordtype as LYB
        tl = LYB._timeline(ctx, spec)
    hits = _hits(ctx, name, opts)
    font, font_label = TX._font(ctx, spec.get("font"), cache)
    glyphs = _glyphs(coll, cache, font, font_label)
    try:
        cap = glyphs.cap()
        plan = BT.plan(spec, fps=ctx.fps, frame0=ctx.frame0, tl=tl, panel=panel, top=top, bottom=bottom, hits=hits,
                       end=ctx.end - ctx.frame0 + 1, screen=screen, draws=glyphs.draws)
    except (BT.BlockError, T.TextError) as e:
        msg = str(e)
        raise BuildError(msg if msg.startswith("text ") else f"text {name!r}: {msg}") from None
    if not plan.blocks:
        raise BuildError(f"text {name!r}: no block is ever on screen (every word lands after the cut)")

    ename = f"{name}{suffix}"
    if bpy.data.objects.get(ename) is not None:
        raise BuildError(f"text {name!r}: an object named {ename!r} exists already")
    root = bpy.data.objects.new(ename, None)
    coll.objects.link(root)
    root.empty_display_size = 0.5
    M = Matrix([list(map(float, r)) for r in (np.asarray(F, float) @ plan.matrix)])
    if parent is None:
        root.matrix_world = M
    else:
        root.parent = parent
        if inverse is not None:
            root.matrix_parent_inverse = inverse
        root.matrix_basis = M
    multi = spec.get("_aspect") is not None and spec.get("_multi")
    if multi:
        root["mk_aspect"] = str(spec["_aspect"])

    glow = float(spec.get("glow", GLOW[screen]))
    lit = 0.0 if screen else float(spec.get("lit", LIT[screen]))
    if not 0.0 <= lit <= 1.0 or glow < 0:
        raise BuildError(f"text {name!r}: lit is a share of the scene's light, 0..1, and glow at least 0")
    meshes = cache.setdefault("block_meshes", {})
    words = {}
    for b in plan.blocks:
        oname = f"{name}_{b.tag}{suffix}"
        if len(oname.encode("utf-8")) > MAX_OBJECT:
            raise BuildError(f"text {name!r}: the block name {oname!r} takes {len(oname.encode('utf-8'))} bytes (UTF-8), "
                             f"over {MAX_OBJECT}: shorten the entry's name")
        if bpy.data.objects.get(oname) is not None:
            raise BuildError(f"text {name!r}: an object named {oname!r} exists already")
        mkey = (font_label, b.mesh)
        if mkey not in meshes:
            m = BT.block_mesh(b.mesh, glyphs.get, cap)
            meshes[mkey] = _mesh(f"mk_block_{len(meshes)}", m, len(b.colors))
        ob = bpy.data.objects.new(oname, meshes[mkey])
        coll.objects.link(ob)
        ob.parent = root
        ob.rotation_mode = "QUATERNION"
        try:
            for i, col in enumerate(b.colors):
                role = "wood" if i == BT.SLOT_WOOD else "ink" if i == BT.SLOT_INK else "paint" if i < BT.SLOT_INK \
                    else "picture"
                ob.material_slots[i].link = "OBJECT"
                ob.material_slots[i].material = _material(cache, ctx.palette, col, role, glow, lit)
        except KeyError as e:
            raise BuildError(f"text {name!r}: blocks: {e.args[0]}") from None
        _key_block(ctx, ob, b)
        ob["mk_block"] = int(b.char)
        if b.lyric is not None:                        # which sung word, as numbers (its letter is geometry, nothing else)
            ob["mk_lyric"] = [int(b.lyric[0]), int(b.lyric[1])]
        if multi:
            ob["mk_aspect"] = str(spec["_aspect"])
        if screen:                                     # type of the picture: drawn by the screen layer, never by a scene render
            ob["mk_screen"] = 1
            ob.hide_render = ob.hide_viewport = True
            ob["mk_show"] = [ctx.frame0 + int(b.on), ctx.frame0 + int(b.off)]
        w = words.setdefault(b.lyric or ("label",), {"blocks": 0, "land": None, "frames": [None, None]})
        w["blocks"] += 1
        land = ctx.frame0 + int(b.land)
        w["land"] = land if w["land"] is None else min(w["land"], land)
        w["frames"][0] = ctx.frame0 + b.on if w["frames"][0] is None else min(w["frames"][0], ctx.frame0 + b.on)
        w["frames"][1] = ctx.frame0 + b.off if w["frames"][1] is None else max(w["frames"][1], ctx.frame0 + b.off)
    ctx.texts = getattr(ctx, "texts", {})
    ctx.texts[ename] = root
    s = dict(plan.summary)
    for key in ("first_frame", "last_frame", "cut_frame"):
        if s.get(key) is not None:
            s[key] += ctx.frame0
    s.update(on=label, font=font_label, edge=round(plan.edge, 6), panel=list(panel) if panel else None, object=ename,
             per_word=[{"lyric": list(k), **v} if k != ("label",) else v for k, v in words.items()])
    ctx.log("blocks", ename, f"{s['blocks']} block(s) of {s['words']} word(s), lands Blender frames {s['first_frame']}.."
            f"{s['last_frame']}" + (f", gone by {s['cut_frame']}" if s.get("cut_frame") is not None else "")
            + f", edge {plan.edge:.4g}" + (f", {sum(s['moves'].values())} move(s) on hits" if s["moves"] else "")
            + (f", {len(s['tumbles'])} tumble(s)" if s["tumbles"] else ""))
    return s
