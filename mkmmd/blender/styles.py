"""Render-time looks of a shot (docs/design.md: Shots): the flat `silhouette`, the flat `vector` and the window
`reflection`, switched on and off per frame by `mk render` (ops_render) and `mk look` (ops_look) from the shot table that
`mk build` keeps in scene["mk_shots"] (mkmmd.blender.build.shots; the normalised specs are mkmmd.core.shotstyle's).

    looks = Looks(scene)
    for f in frames:
        scene.frame_set(f)
        if looks.prepare(f, aspect):            # enters / leaves / updates the look of frame f's shot
            looks.render(path)                  # one image in that look, written as the file type of `path`
        else:
            bpy.ops.render.render(write_still=True)
    looks.close()                               # puts back everything the looks changed

silhouette. Blender 4.2's EEVEE Next (and Workbench) ignore a view layer's `material_override`, so the flat look is built
from the passes that do work, all inside the one render call of a frame and composed with numpy (`compose_silhouette`):
  flat   Workbench, flat light, `Object.color`: the subject and the hard accents (thin cords) in their colours. Exact
         to one level of 255 (Workbench's colour transform), antialiased by Workbench.
  soft   EEVEE with an AOV carrying the opaque share of the accent objects' Mix Shader (a lightning bolt: Transparent
         mixed with Emission) and the compositor writing R = alpha, B = that AOV (Raw values): `soft_alpha` and `soft_aov`.
         The AOV is painted in the accent colour; what the bolt's alpha has beyond it (its soft shells) stays a glow in
         the subject colour (`sil = alpha - bolt` in `compose_silhouette`).
  type   EEVEE with the lights off and a black world: the type objects (the text stage's `mk_text_*` objects) as their
         emission shows them, keyed opacity (the alpha attribute of kinetic words) included.
  knock  the same, alpha only, for the `knockout` objects: ink colour on the background, background colour where it
         overlaps the silhouette.
  screen the text stage's screen type (objects with `mk_screen`, hidden from every scene render): the same pass through an
         orthographic camera one frame height tall, kept off the frame as its own RGBA layer file (`render(path, layer=...)`;
         `mk post` lays it over the cut and its effects; `prepare` answers "screen" for a shot with no look of its own); its
         `knockout` type goes into the knock pass of a silhouette, in the picture.
No file is left behind and nothing in the scene stays changed.

vector. A flat-vector drawing of the scene in the project's tones, composed with numpy (`compose_vector`) from Workbench
passes at SS.VECTOR_SCALE times the frame's size, boxed down:
  id     flat light, every material's viewport colour its id (`vector_table`), Raw, no antialiasing, a float EXR: exact;
         the compositor writes the Z pass to an EXR on the same render (the scene must have no compositor of its own)
  tex    flat light, the textures: what a drawn tone (eyes, mouth) takes its colours from (only when a tone is drawn)
  shade  the studio light, fixed in the world, with cast shadows from the look's `light`, on plain white (Raw)
  type   as for the silhouette, laid over the drawing.

reflection. A plane light probe at the glass object and, in the glass material, a mirror layer in front of the glass
shader (mkmmd.blender.mirror). EEVEE Next fills a glossy surface on a probe's plane with the mirrored view of the whole
scene, so what stands behind the camera shows up in the pane."""
import json
import os
import tempfile

import bpy
import numpy as np

from ..core import screentype as SR
from ..core import shotstyle as SS
from . import mirror as MR

WORKBENCH = "BLENDER_WORKBENCH"
EEVEE = "BLENDER_EEVEE_NEXT"
FLAT_TYPES = {"MESH", "CURVE", "SURFACE", "FONT", "META", "CURVES", "POINTCLOUD"}
TEXT_GROUP = "mk_text_"                  # the node groups of the text stage: its objects are type
AOV_NAME = "mk_soft"


# ================================================================================================= small helpers
class Restore:
    """Records the old value of every attribute it sets (and every callable it is given) and puts them back, last first."""

    def __init__(self):
        self.undo = []

    def attr(self, owner, name, value):
        old = getattr(owner, name)
        if hasattr(old, "__len__") and not isinstance(old, str):
            old = tuple(old)
        self.undo.append((owner, name, old))
        setattr(owner, name, value)

    def call(self, fn):
        self.undo.append((None, None, fn))

    def run(self):
        while self.undo:
            owner, name, old = self.undo.pop()
            if owner is None:
                old()
            else:
                try:
                    setattr(owner, name, old)
                except (ReferenceError, AttributeError):
                    pass                                  # the owner went away with its look


def _linear(rgb):
    return [SS.srgb_to_linear(c) for c in rgb]


def _read_rgba(path):
    """A PNG as (h, w, 4) float32, top row first: the stored values, straight alpha, no colour management."""
    img = bpy.data.images.load(path)
    try:
        img.colorspace_settings.name = "Non-Color"
        w, h = img.size
        px = np.empty(w * h * 4, np.float32)
        img.pixels.foreach_get(px)
    finally:
        bpy.data.images.remove(img)
    return px.reshape(h, w, 4)[::-1]


def _save(path, rgb, quality=90):
    """Write float RGB (h, w, 3) in 0..1, top row first, as the file type of `path`: PNG by hand (8-bit), anything else
    through Blender's own writer with the values untouched."""
    img8 = (np.clip(rgb, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    if path.lower().endswith(".png"):
        with open(path, "wb") as fh:
            fh.write(SS.png_bytes(img8))
        return
    h, w, _ = img8.shape
    buf = np.ones((h, w, 4), np.float32)
    buf[..., :3] = img8[::-1].astype(np.float32) / 255.0
    img = bpy.data.images.new("mk_look_out", w, h, alpha=False, float_buffer=True)
    try:
        img.colorspace_settings.name = "Non-Color"
        img.pixels.foreach_set(buf.reshape(-1))
        img.filepath_raw = path
        if path.lower().endswith((".jpg", ".jpeg")):
            img.file_format = "JPEG"
            img.save(quality=quality)
        else:
            img.file_format = "PNG"
            img.save()
    finally:
        bpy.data.images.remove(img)


def _save_rgba(path, rgba):
    """Write straight RGBA (h, w, 4) float in 0..1, top row first, as an 8-bit PNG with the values untouched (Blender's own
    writer, a byte image: no premultiplying); the file appears whole or not at all, and its folder is made."""
    h, w, _ = rgba.shape
    buf = np.clip(rgba, 0.0, 1.0).astype(np.float32)[::-1]
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = f"{path}.{os.getpid()}.tmp.png"
    img = bpy.data.images.new("mk_screen_out", w, h, alpha=True, float_buffer=False)
    try:
        img.colorspace_settings.name = "Non-Color"
        img.alpha_mode = "STRAIGHT"
        img.pixels.foreach_set(buf.reshape(-1))
        img.filepath_raw = tmp
        img.file_format = "PNG"
        img.save()
    finally:
        bpy.data.images.remove(img)
    os.replace(tmp, path)


def _truthy_props(ob):
    return frozenset(k for k in ob.keys() if not k.startswith("_") and not hasattr(ob[k], "to_dict") and bool(ob[k]))


def _collections(ob):
    """Names of the collections `ob` is in, with every ancestor collection."""
    parents = {}
    for c in bpy.data.collections:
        for ch in c.children:
            parents.setdefault(ch.name, []).append(c)
    out, seen, todo = [], set(), list(ob.users_collection)
    while todo:
        c = todo.pop()
        if c.name in seen:
            continue
        seen.add(c.name)
        out.append(c.name)
        todo.extend(parents.get(c.name, []))
    return tuple(out)


def object_records(scene=None):
    """What the object patterns match (mkmmd.core.shotstyle.Obj) for every renderable object of the scene."""
    sc = scene or bpy.context.scene
    return [SS.Obj(o.name, _collections(o), _truthy_props(o)) for o in sc.objects if o.type in FLAT_TYPES]


def _materials(ob):
    """Every material of the object: its slots and the Set Material nodes of its geometry-nodes modifiers."""
    mats = [s.material for s in ob.material_slots if s.material is not None]
    for mod in ob.modifiers:
        if mod.type == "NODES" and mod.node_group is not None:
            todo, seen = [mod.node_group], set()
            while todo:
                ng = todo.pop()
                if ng.name in seen:
                    continue
                seen.add(ng.name)
                for n in ng.nodes:
                    if n.bl_idname == "GeometryNodeSetMaterial" and n.inputs["Material"].default_value is not None:
                        mats.append(n.inputs["Material"].default_value)
                    if getattr(n, "node_tree", None) is not None:
                        todo.append(n.node_tree)
    out, seen = [], set()
    for m in mats:
        if m.name not in seen and m.use_nodes and m.node_tree is not None:
            seen.add(m.name)
            out.append(m)
    return out


def is_soft(ob):
    """An object whose material lets the scene through (a Transparent shader in it): a lightning bolt, a glow shell."""
    return any(n.bl_idname == "ShaderNodeBsdfTransparent" for m in _materials(ob) for n in m.node_tree.nodes)


def is_type(ob):
    return any(m.type == "NODES" and m.node_group is not None and m.node_group.name.startswith(TEXT_GROUP)
               for m in ob.modifiers)


def add_weight_aov(mat, aov):
    """An AOV Output node in `mat` that carries the opaque share of its surface: the factor of the Mix Shader that mixes
    a Transparent BSDF in (the factor itself when Transparent is the first shader, one minus it when it is the second).
    Returns the function that removes what it added; a material without such a mix gets nothing."""
    nt = mat.node_tree
    added = []
    for mix in [n for n in nt.nodes if n.bl_idname == "ShaderNodeMixShader"]:
        for slot in (1, 2):
            links = mix.inputs[slot].links
            if not (links and links[0].from_node.bl_idname == "ShaderNodeBsdfTransparent"):
                continue
            node = nt.nodes.new("ShaderNodeOutputAOV")
            node.aov_name = aov
            added.append(node)
            fac = mix.inputs[0]
            src = fac.links[0].from_socket if fac.links else None
            if slot == 1:
                if src is not None:
                    nt.links.new(src, node.inputs["Value"])
                else:
                    node.inputs["Value"].default_value = fac.default_value
            else:
                sub = nt.nodes.new("ShaderNodeMath")
                sub.operation = "SUBTRACT"
                sub.inputs[0].default_value = 1.0
                added.append(sub)
                if src is not None:
                    nt.links.new(src, sub.inputs[1])
                else:
                    sub.inputs[1].default_value = fac.default_value
                nt.links.new(sub.outputs[0], node.inputs["Value"])

    def undo():
        for n in added:
            nt.nodes.remove(n)
    return undo


# ================================================================================================= the looks
class Looks:
    """The look in force at each frame of a render or look session (see the module docstring)."""

    def __init__(self, scene=None):
        self.sc = scene or bpy.context.scene
        try:
            self.table = json.loads(self.sc.get("mk_shots", "[]"))
        except ValueError:
            self.table = []
        self.kind = None                  # "silhouette" | "vector" | "reflection" | None
        self.key = None                   # (shot name, aspect, kind) of the look that is entered
        self.spec = None
        self.rs = Restore()
        self.info = {}
        self.cl = {}
        self.tint_values = []
        self.screen_names = [o.name for o in self.sc.objects if "mk_screen" in o.keys()]
        self.screen = ([], [])            # (type over the frame, type reversed out of a silhouette) at the prepared frame
        self.screen_cam = None

    # ---------------------------------------------------------------- state machine
    def spec_at(self, frame, aspect=None, shot=None):
        """(shot name, kind, spec) of the look at `frame` for an output aspect, or (name, None, None). `shot` names the
        shot to look through instead of the one in the cut at `frame` (the plates of a transition or insert)."""
        if shot is None:
            entry = SS.shot_at(self.table, frame)
        else:
            entry = next((e for e in self.table if e["name"] == shot), None)
            if entry is None:
                raise RuntimeError(f"no shot named {shot!r} in the scene's shot table (run mk build)")
        if entry is None:
            return None, None, None
        styles = entry.get("styles") or {}
        look = styles.get(aspect) if aspect is not None else (next(iter(styles.values())) if styles else None)
        if not look:
            return entry["name"], None, None
        kind = next(iter(look))
        return entry["name"], kind, look[kind]

    def prepare(self, frame, aspect=None, shot=None):
        """Bring the scene into the look of `frame` (call after `frame_set`), or of `shot` when one is named (the plates of
        a transition or insert: no screen type). Returns the active kind, "screen" when only screen type is on the frame,
        or None."""
        name, kind, spec = self.spec_at(frame, aspect, shot)
        key = (name, aspect, kind)
        if key != self.key:
            self.leave()
            self.key = key
            if kind is not None:
                self.kind, self.spec = kind, spec
                getattr(self, "_enter_" + kind)(name, spec)
        if self.kind == "silhouette":
            self._update_silhouette()
        self.screen = self._screen_live(frame, aspect) if shot is None and self.screen_names else ([], [])
        if self.kind != "silhouette":              # nothing to reverse out of: all of it goes over the frame
            self.screen = (self.screen[0] + self.screen[1], [])
        return self.kind or ("screen" if self.screen[0] else None)

    def leave(self):
        if self.kind is not None:
            self.rs.run()
            self.kind, self.spec = None, None
        self.key = None

    def close(self):
        self.leave()
        if self.screen_cam is not None:
            data = self.screen_cam.data
            bpy.data.objects.remove(self.screen_cam)
            bpy.data.cameras.remove(data)
            self.screen_cam = None

    def render(self, path, layer=None):
        """Render the current frame to `path` in the active look (a plain render for the reflection, whose image settings
        are the scene's; the composed silhouette is written as the file type of `path`). The screen type of the frame goes
        into its own RGBA file `layer` (written first: a frame on disk always has its layer) and is left off the picture, so
        `mk post` can lay it over whatever the cut effects make of the frame; with no `layer` it is laid over the picture
        here. A silhouette's `knockout` type belongs to its figure and is always in the picture."""
        over = self.screen[0]
        rgba = self._screen_pass(over, "screen") if over else None
        if rgba is not None and layer is not None:
            _save_rgba(layer, rgba)
            rgba = None
        if self.kind in ("silhouette", "vector"):
            img = self.compose(self.passes())
        elif rgba is not None:
            img = self._shoot_scene()
        else:
            self.sc.render.filepath = path
            bpy.ops.render.render(write_still=True)
            return
        if rgba is not None:
            img = SR.composite(img, rgba)
        _save(path, img)

    # ---------------------------------------------------------------- screen type
    def _screen_live(self, frame, aspect):
        """(names of the screen type on screen at `frame` in an output, those of them to reverse out of a silhouette): the
        objects made for that output (`mk_aspect`) whose `mk_show` frames [on, off) hold the frame."""
        over, knock = [], []
        for n in self.screen_names:
            o = bpy.data.objects[n]
            if aspect is not None and "mk_aspect" in o.keys() and o["mk_aspect"] != aspect:
                continue
            show = o.get("mk_show")
            if show is not None and not show[0] <= frame < show[1]:
                continue
            (knock if "mk_knockout" in o.keys() else over).append(n)
        return over, knock

    def _screen_camera(self):
        """The orthographic camera that frames the plane z = 0 one frame height tall (made once, removed by `close`)."""
        if self.screen_cam is None:
            cd = bpy.data.cameras.new("mk_screen_layer")
            cd.type, cd.sensor_fit, cd.clip_start, cd.clip_end = "ORTHO", "AUTO", 0.1, 100.0
            ob = bpy.data.objects.new("mk_screen_layer", cd)
            ob.location = (0.0, 0.0, 10.0)                            # looks down -z at the plane, +y up
            self.sc.collection.objects.link(ob)
            self.screen_cam = ob
        return self.screen_cam

    def _screen_pass(self, names, tag):
        """The given screen type as its emission shows it (lights off, black world), through the screen camera -> straight
        RGBA, one pixel per pixel of the frame. The shot markers are set aside meanwhile: a render would switch the camera
        back to the shot's."""
        sc, rs = self.sc, Restore()
        try:
            cam = self._screen_camera()
            r = sc.render
            cam.data.ortho_scale = SR.ortho_scale((r.resolution_x, r.resolution_y))
            marks = [(m.name, m.frame, m.camera) for m in sc.timeline_markers]
            rs.attr(sc, "camera", cam)
            for m in list(sc.timeline_markers):
                sc.timeline_markers.remove(m)

            def put_back():
                for mname, f, c in marks:
                    sc.timeline_markers.new(mname, frame=f).camera = c
            rs.call(put_back)
            for n in names:
                rs.attr(bpy.data.objects[n], "hide_render", False)
            return self._type(names, tag=tag)
        finally:
            rs.run()

    def _shoot_scene(self):
        """The scene's own render of the frame (its engine, light and view transform) as display values -> (h, w, 3)."""
        rs, ims = Restore(), self.sc.render.image_settings
        try:
            for k, val in (("file_format", "PNG"), ("color_mode", "RGB"), ("color_depth", "8"), ("compression", 0)):
                rs.attr(ims, k, val)
            return self._shoot("scene")[..., :3]
        finally:
            rs.run()

    # ---------------------------------------------------------------- passes
    def _objects(self):
        return [o for o in self.sc.objects if o.type in FLAT_TYPES]

    def _visibility(self, rs, show=(), holdout=()):
        """Everything rendered but `show` (as it is) and `holdout` (invisible but hiding what is behind it, EEVEE only)
        is hidden for the pass. Objects the scene hides at this frame stay hidden."""
        show, holdout = set(show), set(holdout)
        for o in self._objects():
            if o.hide_render or o.name in show:
                continue
            if o.name in holdout:
                rs.attr(o, "is_holdout", True)
            else:
                rs.attr(o, "hide_render", True)

    def _film(self, rs, engine, view, samples=None):
        sc, r = self.sc, self.sc.render
        rs.attr(r, "engine", engine)
        rs.attr(r, "film_transparent", True)
        rs.attr(r, "use_motion_blur", False)
        rs.attr(r, "dither_intensity", 0.0)
        rs.attr(r, "use_compositing", False)
        v = sc.view_settings
        for k, val in (("look", "None"),                  # first: the look resets with the view transform, so it is
                       ("view_transform", view),          # restored last, when its transform is back
                       ("exposure", 0.0), ("gamma", 1.0), ("use_curve_mapping", False)):
            rs.attr(v, k, val)
        ims = r.image_settings
        for k, val in (("file_format", "PNG"), ("color_mode", "RGBA"), ("color_depth", "8"), ("compression", 0)):
            rs.attr(ims, k, val)
        if engine == EEVEE:
            rs.attr(sc.eevee, "taa_render_samples", int(samples or 16))
            rs.attr(sc.eevee, "use_raytracing", False)
        else:
            d = sc.display
            rs.attr(d, "render_aa", "16")
            sh = d.shading
            for k, val in (("light", "FLAT"), ("color_type", "OBJECT"), ("show_object_outline", False),
                           ("show_cavity", False), ("show_shadows", False), ("show_specular_highlight", False)):
                rs.attr(sh, k, val)

    def _shoot(self, tag, ext=".png"):
        """Render the current frame into a temporary file (the image settings' type, named with `ext`) and read it back ->
        (h, w, 4) float32."""
        path = os.path.join(tempfile.gettempdir(), f"mk_pass_{os.getpid()}_{tag}{ext}")
        self.sc.render.filepath = path
        try:
            bpy.ops.render.render(write_still=True)
            return _read_rgba(path)
        finally:
            if os.path.exists(path):
                os.unlink(path)

    def _flat(self):
        """Subject and hard accents in their flat colours -> straight RGBA (coverage in alpha, Workbench scale)."""
        rs = Restore()
        try:
            self._film(rs, WORKBENCH, "Standard")
            self._visibility(rs, show=self.cl["subject"] + self.cl["hard"])
            return self._shoot("flat")
        finally:
            rs.run()

    def matte(self, scale=2):
        """Coverage (h, w) in 0..1 of the subject alone, without cords or type: the figure a transition turns into a
        window. Antialiased by Workbench like the flat pass it is cut from, and drawn `scale` times the frame's size: the
        figure is zoomed far past the frame's resolution, so its edge wants every pixel of detail it can have."""
        rs = Restore()
        try:
            self._film(rs, WORKBENCH, "Standard")
            r = self.sc.render
            rs.attr(r, "resolution_x", r.resolution_x * int(scale))
            rs.attr(r, "resolution_y", r.resolution_y * int(scale))
            self._visibility(rs, show=self.cl["subject"])
            return np.clip(self._shoot("matte")[..., 3] / SS.ALPHA_FULL, 0.0, 1.0).astype(np.float32)
        finally:
            rs.run()

    def _soft(self, occlude=True):
        """The accent objects with a transparent material, hidden behind the subject (unless `occlude` is off) ->
        (alpha, aov weight), both 0..1."""
        sc, rs = self.sc, Restore()
        try:
            self._film(rs, EEVEE, "Raw", self.spec["samples"])
            rs.attr(sc.render.image_settings, "color_mode", "RGB")
            self._visibility(rs, show=self.cl["soft"], holdout=self.cl["subject"] + self.cl["hard"] if occlude else ())
            vl = bpy.context.view_layer
            aov = vl.aovs.add()
            aov.name, aov.type = AOV_NAME, "VALUE"
            rs.call(lambda: vl.aovs.remove(aov))
            for n in self.cl["soft"]:
                for m in _materials(bpy.data.objects[n]):
                    rs.call(add_weight_aov(m, AOV_NAME))
            self._aov_compositor(rs, AOV_NAME)
            px = self._shoot("soft")
            return np.clip(px[..., 0], 0.0, 1.0), np.clip(px[..., 2], 0.0, 1.0)
        finally:
            rs.run()

    def _aov_compositor(self, rs, aov):
        """Render Layers -> Combine (R = alpha, B = the AOV) -> Composite, on a scene that has no compositor of its own."""
        sc = self.sc
        ours = ("CompositorNodeRLayers", "CompositorNodeComposite")
        if sc.use_nodes and sc.node_tree is not None and any(n.bl_idname not in ours for n in sc.node_tree.nodes):
            raise RuntimeError("the scene has a compositor tree of its own; the silhouette's accent pass needs the compositor")
        rs.attr(sc, "use_nodes", True)
        rs.attr(sc.render, "use_compositing", True)
        nt = sc.node_tree
        mine = []

        def node(kind):
            n = nt.nodes.new(kind)
            mine.append(n)
            return n
        for n in list(nt.nodes):                                 # a freshly enabled tree holds a default Render Layers
            nt.nodes.remove(n)                                   # and Composite; the pass uses its own
        rl, comb = node("CompositorNodeRLayers"), node("CompositorNodeCombineColor")
        comp = node("CompositorNodeComposite")
        comb.mode = "RGB"
        comb.inputs["Alpha"].default_value = 1.0
        comb.inputs["Green"].default_value = 0.0
        comp.use_alpha = False
        nt.links.new(rl.outputs["Alpha"], comb.inputs["Red"])
        nt.links.new(rl.outputs[aov], comb.inputs["Blue"])
        nt.links.new(comb.outputs["Image"], comp.inputs["Image"])

        def undo():
            for n in mine:
                nt.nodes.remove(n)
        rs.call(undo)

    def _type(self, names, holdout=(), tag="type"):
        """The given objects as their emission shows them (lights off, black world) -> straight RGBA."""
        sc, rs = self.sc, Restore()
        try:
            self._film(rs, EEVEE, "Standard", (self.spec or {}).get("samples"))
            for o in sc.objects:
                if o.type == "LIGHT" and not o.hide_render:
                    rs.attr(o, "hide_render", True)
            world = bpy.data.worlds.new("mk_black")
            world.use_nodes = True
            bg = world.node_tree.nodes["Background"]
            bg.inputs["Color"].default_value = (0.0, 0.0, 0.0, 1.0)
            bg.inputs["Strength"].default_value = 0.0
            rs.attr(sc, "world", world)
            rs.call(lambda: bpy.data.worlds.remove(world))
            self._visibility(rs, show=names, holdout=holdout)
            return self._shoot(tag)
        finally:
            rs.run()

    # ---------------------------------------------------------------- silhouette
    def _enter_silhouette(self, name, spec):
        rs = self.rs
        objs = [o for o in self._objects() if not o.hide_render]
        recs = [SS.Obj(o.name, _collections(o), _truthy_props(o)) for o in objs]
        hide = set(SS.select(recs, spec["hide"])) - set(SS.select(recs, spec["keep"]))
        knock = set(SS.select(recs, spec["knockout"]["objects"])) if spec["knockout"] else set()
        accent = set(SS.select(recs, spec["accent"]))
        cols = spec["colors"]
        cl = {k: [] for k in ("hide", "subject", "hard", "soft", "type", "knock")}
        for o in objs:
            if o.name in hide:
                cl["hide"].append(o.name)
                rs.attr(o, "hide_render", True)
            elif o.name in knock:
                cl["knock"].append(o.name)
            elif is_type(o):
                cl["type"].append(o.name)
            elif o.name in accent and is_soft(o):
                cl["soft"].append(o.name)
            elif o.name in accent:
                cl["hard"].append(o.name)
                rs.attr(o, "color", (*_linear(cols["accent"]), 1.0))
            else:
                cl["subject"].append(o.name)
                rs.attr(o, "color", (*_linear(cols["subject"]), 1.0))
        self.cl = cl
        self.info = {k: len(v) for k, v in cl.items()}

    def _update_silhouette(self):
        """Per frame: the custom properties the background layers follow (a flash, a sun)."""
        vals = []
        for layer in self.spec["tint"]:
            holder = bpy.data.objects.get(layer["object"])
            vals.append(float(holder[layer["prop"]]) if holder is not None and layer["prop"] in holder.keys() else 0.0)
        self.tint_values = vals

    def passes(self):
        """The renders of the current frame that the flat look is composed from: {name: array}, for inspection."""
        if self.kind == "vector":
            return self._vector_passes()
        cl, out = self.cl, {}
        out["flat"] = self._flat()
        if cl["soft"]:
            out["soft_alpha"], out["soft_aov"] = self._soft()
        live = [n for n in cl["type"] if not bpy.data.objects[n].hide_render]
        if live:
            out["type"] = self._type(live, holdout=cl["subject"] + cl["hard"])
        live = [n for n in cl["knock"] if not bpy.data.objects[n].hide_render]
        if self.spec["knockout"] and live:
            out["knock"] = self._type(live, tag="knock")
        if self.screen[1]:                         # screen type to reverse out: ink on the background, the background on her
            if not self.spec["knockout"]:          # no table of the shot's: the figure's own ink
                self.spec = dict(self.spec, knockout={"objects": [], "color": self.spec["colors"]["subject"]})
            k = self._screen_pass(self.screen[1], "knock")
            out["knock"] = k if "knock" not in out else np.maximum(out["knock"], k)
        return out

    def compose(self, p, subject=True):
        """The finished flat frame (float RGB, display space) from the passes `passes()` made; `subject = False` leaves the
        figure out (the frame round a transition's window)."""
        if self.kind == "vector":
            img = SS.compose_vector(p, self.spec, self.vtable, subject=subject)
            if "type" in p:
                a = np.clip(p["type"][..., 3:4], 0.0, 1.0)
                img = img * (1.0 - a) + p["type"][..., :3] * a
            return img
        return SS.compose_silhouette(p, self.spec, self.tint_values, hard=bool(self.cl["hard"]), subject=subject)

    # ---------------------------------------------------------------- vector
    def _enter_vector(self, name, spec):
        rs = self.rs
        objs = [o for o in self._objects() if not o.hide_render]
        recs = [SS.Obj(o.name, _collections(o), _truthy_props(o)) for o in objs]
        hide = set(SS.select(recs, spec["hide"])) - set(SS.select(recs, spec["keep"]))
        cl = {"hide": [], "subject": [], "type": []}
        for o in objs:
            if o.name in hide:
                cl["hide"].append(o.name)
                rs.attr(o, "hide_render", True)
            else:
                cl["type" if is_type(o) else "subject"].append(o.name)
        mats = []
        for n in cl["subject"]:
            ob = bpy.data.objects[n]
            for m in [s.material for s in ob.material_slots if s.material is not None] + _materials(ob):
                if m.name not in mats:
                    mats.append(m.name)
        self.cl, self.vmats = cl, mats
        self.vtable = SS.vector_table(mats, spec)
        self.info = {**{k: len(v) for k, v in cl.items()}, "materials": len(mats)}

    def _vector_passes(self):
        """The passes `compose_vector` reads, at SS.VECTOR_SCALE times the frame's size (see the module docstring)."""
        r, s, out = self.sc.render, SS.VECTOR_SCALE, {}
        rs = Restore()
        try:
            rs.attr(r, "resolution_x", r.resolution_x * s)
            rs.attr(r, "resolution_y", r.resolution_y * s)
            self._visibility(rs, show=self.cl["subject"])
            out["id"], out["depth"] = self._vector_ids()
            if any(t["kind"] == "drawn" for t in self.spec["tones"].values()):
                out["tex"] = self._vector_pass("tex")[..., :3]
            out["shade"] = self._vector_pass("shade")[..., 0]
        finally:
            rs.run()
        live = [n for n in self.cl["type"] if not bpy.data.objects[n].hide_render]
        if live:
            out["type"] = self._type(live, holdout=self.cl["subject"])
        return out

    def _vector_ids(self):
        """(ids, depth): every material's viewport colour is its id (Raw, no antialiasing, a float EXR: exact; a material
        under half opaque is left out, as Workbench leaves it out of the other passes), and the compositor writes the Z
        pass of the same render to an EXR."""
        sc, rs = self.sc, Restore()
        try:
            self._film(rs, WORKBENCH, "Raw")
            rs.attr(sc.display, "render_aa", "OFF")
            rs.attr(sc.display.shading, "color_type", "MATERIAL")
            ims = sc.render.image_settings
            for k, val in (("file_format", "OPEN_EXR"), ("color_depth", "32"), ("exr_codec", "ZIP")):
                rs.attr(ims, k, val)
            for k, n in enumerate(self.vmats, 1):
                m = bpy.data.materials[n]
                rs.attr(m, "diffuse_color", ((k % 256) / 255.0, (k // 256) / 255.0, 0.0, float(m.diffuse_color[3] >= 0.5)))
            depth = self._depth_compositor(rs)
            px = self._shoot("vid", ".exr")
            ids = np.where(px[..., 3] > 0.5, np.rint(px[..., 0] * 255.0) + 256.0 * np.rint(px[..., 1] * 255.0), 0.0)
            return ids.astype(np.int32), depth()
        finally:
            rs.run()

    def _depth_compositor(self, rs):
        """Render Layers' Depth -> a File Output EXR in the temp folder, on a scene with no compositor of its own. Returns
        a function that reads it back after the render ((h, w) metres, top row first) and deletes it."""
        sc = self.sc
        ours = ("CompositorNodeRLayers", "CompositorNodeComposite")
        if sc.use_nodes and sc.node_tree is not None and any(n.bl_idname not in ours for n in sc.node_tree.nodes):
            raise RuntimeError("the scene has a compositor tree of its own; the vector look's depth pass needs the compositor")
        rs.attr(sc, "use_nodes", True)
        rs.attr(sc.render, "use_compositing", True)
        rs.attr(bpy.context.view_layer, "use_pass_z", True)
        nt = sc.node_tree
        for n in list(nt.nodes):                                 # a freshly enabled tree holds a default Render Layers
            nt.nodes.remove(n)                                   # and Composite; the pass uses its own
        rl, comp = nt.nodes.new("CompositorNodeRLayers"), nt.nodes.new("CompositorNodeComposite")
        fo = nt.nodes.new("CompositorNodeOutputFile")
        mine = [rl, comp, fo]
        stem = f"mk_pass_{os.getpid()}_depth_"
        fo.base_path = tempfile.gettempdir()
        fo.format.file_format, fo.format.color_depth, fo.format.color_mode = "OPEN_EXR", "32", "RGB"
        fo.file_slots[0].path = stem
        nt.links.new(rl.outputs["Image"], comp.inputs["Image"])
        nt.links.new(rl.outputs["Depth"], fo.inputs[0])
        rs.call(lambda: [nt.nodes.remove(n) for n in mine])
        path = os.path.join(tempfile.gettempdir(), f"{stem}{sc.frame_current:04d}.exr")

        def read():
            try:
                return _read_rgba(path)[..., 0]
            finally:
                if os.path.exists(path):
                    os.unlink(path)
        return read

    def _vector_pass(self, kind):
        """`tex`: the textures under flat light (display space); `shade`: the studio light, fixed in the world, with cast
        shadows from the look's `light`, on plain white (Raw)."""
        sc, rs = self.sc, Restore()
        try:
            sh, d = sc.display.shading, sc.display
            if kind == "tex":
                self._film(rs, WORKBENCH, "Standard")
                rs.attr(sh, "color_type", "TEXTURE")
            else:
                self._film(rs, WORKBENCH, "Raw")
                for k, val in (("light", "STUDIO"), ("color_type", "SINGLE"), ("single_color", (1.0, 1.0, 1.0)),
                               ("show_shadows", True), ("shadow_intensity", 1.0), ("use_world_space_lighting", True)):
                    rs.attr(sh, k, val)
                for k, val in (("light_direction", tuple(self.spec["light"])), ("shadow_shift", 0.1), ("shadow_focus", 0.0)):
                    rs.attr(d, k, val)
            return self._shoot("v" + kind)
        finally:
            rs.run()

    # ---------------------------------------------------------------- reflection
    def _enter_reflection(self, name, spec):
        rs = self.rs
        glass = bpy.data.objects.get(spec["object"])
        if glass is None:
            raise RuntimeError(f"shot {name!r}: reflection object {spec['object']!r} is not in the scene")
        plane = MR.glass_plane(glass)
        probe = MR.find_probe(self.sc, plane)
        made = probe is None
        if made:
            probe = MR.make_probe(self.sc, f"mk_reflect_{name}", plane, spec.get("probe") or {})
            rs.call(lambda p=probe: MR.remove_probe(p))
        objs = [o for o in self._objects() if not o.hide_render]
        recs = [SS.Obj(o.name, _collections(o), _truthy_props(o)) for o in objs]
        skip = set(SS.select(recs, spec["hide"]))
        if spec["only"]:
            skip |= {r.name for r in recs} - set(SS.select(recs, spec["only"]))
        for n in sorted(skip):
            rs.attr(bpy.data.objects[n], "hide_probe_plane", True)       # the pane does not show these
        if not spec["world"]:
            undo = MR.world_dark_to_camera(self.sc.world)
            if undo is not None:
                rs.call(undo)
        rs.call(MR.mirror_layer(glass, spec))
        self.info = {"object": glass.name, "probe": probe.name, "probe_made": made, "not_reflected": len(skip),
                     "plane_centre": [round(c, 4) for c in plane["centre"]], "half": [round(v, 4) for v in plane["half"]]}
