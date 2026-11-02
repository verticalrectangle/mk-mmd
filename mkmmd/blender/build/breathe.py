"""breathe: `[look] breathe` in the build (docs/design.md: Lights and the look): every colour the build took from the palette
keyed between its value with the project's palette and with the other one, by the mix over clip time.

A probe Blender builds this build's stages before this one (scene .. keys) with the other palette, beside this build from
its first stage (`start`, `Probe`), and saves them (`save_probe`). This stage waits for it, links the probe's materials,
node groups, worlds, lights and the meshes whose attributes a material reads, pairs them with this build's by name (a
material also through the object slot that holds it in both), and drives every difference from one scene property,
`mk_breathe`: the mix, keyed on every frame. The builders are deterministic, so a value that differs is one the palette
made.
  - Socket values of materials, worlds and shader node groups, and the values of RGB and Value nodes: a driver per channel
    that differs, colours mixed as c ** (1 / 2.2) (core.breathe.colour_expr: close to sRGB, as "a:b:t" mixes), other
    numbers linearly.
  - A colour ramp, an image or a mesh attribute that differs: the probe's beside this build's (a ramp node, a packed
    image, an attribute `<name>~breathe` on every mesh the material is on) and the two outputs mixed by the breath
    (colours through Gamma 1/2.2, Mix, Gamma 2.2).
  - Lights: their colour, and their strength mixed and times the `lights` factor (lerp(1, factor, mix)): a keyed
    strength is keyed again on every frame, a driven one's expression is multiplied. The world's colour and materials'
    viewport colours.
The cast's own materials are left alone. What cannot breathe (a value animated already, a node or a link only one build
has, a setting that is not a number) keeps this build's value and is listed in the report's `fixed` (a WARNING).
Renderers need nothing new: Blender evaluates the drivers on every frame, and the bake records every driven node value
and every light per frame (mkr draws from those)."""
import atexit
import json
import os
import shutil
import subprocess
import time
import uuid

import bpy
import numpy as np

from ...core import breathe as BR
from ...core import jobscope as JS
from .. import keys as K
from ..runtime import CTX
from . import STAGES, BuildError

PROP = "mk_breathe"          # the scene property every breathing value is driven by: the mix on the frame
SUFFIX = "~breathe"          # the probe's ramp, image or attribute beside this build's
VAR = "m"                    # the drivers' variable: the scene's mk_breathe
FIXED_LOG = 12               # how many of the values that cannot breathe the log names

ATTR_KEY = {"FLOAT": ("value", 1), "INT": ("value", 1), "INT8": ("value", 1), "BOOLEAN": ("value", 1),
            "FLOAT2": ("vector", 2), "INT32_2D": ("value", 2), "FLOAT_VECTOR": ("vector", 3), "FLOAT_COLOR": ("color", 4),
            "BYTE_COLOR": ("color", 4), "QUATERNION": ("value", 4)}
READS = {"ATTRIBUTE": "attribute_name", "VERTEX_COLOR": "layer_name"}   # nodes that read a mesh attribute by name
_NODE_BASE = None


def _spec(ctx):
    """The project's `[look] breathe` checked (core.breathe.spec), or None."""
    look = ctx.data.get("look") or {}
    try:
        sp = BR.spec(look)
        if sp is not None:
            BR.palettes(look)
    except BR.BreatheError as e:
        raise BuildError(str(e)) from None
    return sp


def start(ctx):
    """Start the probe build beside this one when the project breathes (build/__init__.py, before the first stage), or
    take the one `mk build` built before this build (`probe_ready`: when the memory would not hold both at once)."""
    sp = _spec(ctx)
    if sp is not None:
        ctx.breath_probe = Ready(ctx.probe_ready) if ctx.probe_ready else Probe(ctx, sp)


def run(ctx):
    sp = _spec(ctx)
    if sp is None:
        return {"breathes": False}
    probe = getattr(ctx, "breath_probe", None) or Probe(ctx, sp)
    sc = bpy.context.scene
    frames = [int(f) for f in ctx.frames]
    mixes = [BR.frame_mix(sp, ctx.fps, ctx.frame0, f) for f in frames]
    sc[PROP] = float(BR.frame_mix(sp, ctx.fps, ctx.frame0, ctx.frame0))
    K.set_fcurve(sc, f'["{PROP}"]', 0, frames, mixes, interp="LINEAR")
    t0 = time.time()
    try:
        blend = probe.wait()
        waited = round(time.time() - t0, 1)
        rep = Breath(ctx, sp, sc, frames, mixes).run(blend)
    finally:
        probe.stop()
        if not os.environ.get("MK_KEEP_JOBS"):
            shutil.rmtree(probe.folder, ignore_errors=True)
    rep = {"breathes": True, "palette": sp["palette"], "keys": len(sp["keys"]),
           "probe": {"stages": probe.stages, "seconds": probe.seconds, "waited": waited}, **rep}
    ctx.log(f"breathe: {rep['drivers']} drivers ({rep['sockets']} socket values, {rep['values']} RGB / Value nodes, "
            f"{rep['ramps']} ramps, {rep['images']} images, {rep['attributes']} attributes, {rep['lights']} lights); "
            f"the probe took {probe.seconds}s, {waited}s of them waited for")
    if rep["fixed"]:
        ctx.log(f"WARNING breathe: {len(rep['fixed'])} values keep this build's colour: "
                + "; ".join(rep["fixed"][:FIXED_LOG]) + (" ..." if len(rep["fixed"]) > FIXED_LOG else ""))
    return rep


# ----------------------------------------------------------------------------------------------------------- the probe
class Probe:
    """The probe build: this build's stages before `breathe` with the other palette, in a Blender of its own started
    from the file this one started from, running beside this build until `wait`. Solver results and the form cache
    are written whole (a temporary file renamed), so the two builds share the project's cache."""

    def __init__(self, ctx, sp):
        if not ctx.start_file:
            raise BuildError("breathe: this Blender started without a file; the probe build starts from the same one "
                             "(mk build starts every build from one)")
        self.palette = sp["palette"]
        self.stages = [s for s in STAGES[:STAGES.index("breathe")] if s in ctx.stages]
        self.folder = os.path.join(ctx.cache, "breathe", f"{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}")
        os.makedirs(self.folder, exist_ok=True)
        self.blend, job_path, self.result, self.log_path = (
            os.path.join(self.folder, n) for n in ("probe.blend", "job.json", "result.json", "blender.log"))
        job = {"op": "build", "args": {"stages": self.stages, "cache": ctx.cache, "probe": self.blend},
               "project": CTX["project"], "config": CTX["config"]}
        with open(job_path, "w", encoding="utf-8") as fh:
            json.dump(job, fh, ensure_ascii=False)
        root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        boot = f"import sys; sys.path.insert(0, {root!r}); from mkmmd.blender import runtime; runtime.main()"
        cmd = [bpy.app.binary_path, "-b", ctx.start_file, "-y", "--python-exit-code", "3", "--python-expr", boot, "--",
               job_path, self.result]
        try:
            self.mem = JS.share((CTX.get("config") or {}).get("blender_memory"))
        except JS.JobScopeError as e:
            raise BuildError(str(e)) from None
        cmd = JS.wrap(cmd, "Blender breathe probe", self.mem)     # its own scope: an overflow kills the probe alone
        self.log = open(self.log_path, "w", encoding="utf-8")
        self.t0 = time.time()
        self.seconds = None
        self.proc = subprocess.Popen(cmd, stdout=self.log, stderr=subprocess.STDOUT,
                                     env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"))
        atexit.register(self.stop)                 # a build that fails before `breathe` leaves no probe running
        ctx.log(f"breathe: the probe build with {self.palette} started ({' '.join(self.stages)})")

    def stop(self):
        if self.proc.poll() is None:
            self.proc.kill()
            self.proc.wait()
        if not self.log.closed:
            self.log.close()

    def wait(self):
        """The probe's saved .blend, once it is done; a BuildError with its error when it failed."""
        rc = self.proc.wait()
        self.log.close()
        self.seconds = round(time.time() - self.t0, 1)
        if not os.path.exists(self.result):
            why = JS.killed_message(rc, "breathe probe", self.mem)
            raise BuildError(f"breathe: {why or f'the probe build exited ({rc}) without a result'}; its log: "
                             f"{self.log_path}\n" + _tail(self.log_path))
        with open(self.result, encoding="utf-8") as fh:
            res = json.load(fh)
        if not res.get("ok"):
            raise BuildError(f"breathe: the probe build with {self.palette} failed (its log: {self.log_path}): "
                             f"{res.get('error')}\n{(res.get('trace') or '')[-2000:]}")
        if not os.path.exists(self.blend):
            raise BuildError(f"breathe: the probe build saved nothing at {self.blend}")
        return self.blend


class Ready:
    """A probe build `mk build` ran before this build (when the memory available would not hold both Blenders at once):
    its folder, its saved .blend, its stages and seconds; nothing to wait for or stop."""

    def __init__(self, ready):
        self.folder, self.blend = ready["folder"], ready["blend"]
        self.stages, self.seconds = ready["stages"], ready["seconds"]

    def wait(self):
        if not os.path.exists(self.blend):
            raise BuildError(f"breathe: the probe build saved nothing at {self.blend}")
        return self.blend

    def stop(self):
        pass


def _tail(path, n=30):
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            return "".join(fh.readlines()[-n:])
    except OSError:
        return ""


def save_probe(ctx):
    """A probe build's end (build/__init__.py): the scene saved where the build that started it reads it, and beside it
    (`<blend>.json`) the material each object's slots hold and the scene's world: an object's name is the same in both
    builds where a material named after its colour is not."""
    sc = bpy.context.scene
    slots = {o.name: [s.material.name if s.material else None for s in o.material_slots]
             for o in sc.objects if len(o.material_slots)}
    with open(ctx.probe + ".json", "w", encoding="utf-8") as fh:
        json.dump({"slots": slots, "world": sc.world.name if sc.world else None}, fh, ensure_ascii=False)
    sc.frame_set(ctx.frame0)
    bpy.ops.wm.save_as_mainfile(filepath=ctx.probe, compress=False)


# ------------------------------------------------------------------------------------------------------------ helpers
def _esc(name):
    return bpy.utils.escape_identifier(name)


def _vals(v):
    """A socket or property value as a tuple of floats, or None when it is not a number (a bool, a string, a pointer)."""
    if isinstance(v, bool) or isinstance(v, str) or v is None:
        return None
    if isinstance(v, (int, float)):
        return (float(v),)
    try:
        return tuple(float(x) for x in v)
    except TypeError:
        return None


def _differ(a, b):
    return any(abs(x - y) > 1e-7 for x, y in zip(a, b))


def _anim_paths(*owners, prefix=""):
    """The data paths keyed (actions, NLA strips) or driven on these IDs; `prefix` ("node_tree.") stripped."""
    out = set()
    for o in owners:
        ad = getattr(o, "animation_data", None) if o is not None else None
        if ad is None:
            continue
        paths = set()
        for act in [ad.action] + [s.action for t in ad.nla_tracks for s in t.strips]:
            if act is not None:
                paths.update(fc.data_path for fc in act.fcurves)
        paths.update(d.data_path for d in ad.drivers)
        out.update(p[len(prefix):] if prefix and p.startswith(prefix) else p for p in paths)
    return out


def _node_base():
    global _NODE_BASE
    if _NODE_BASE is None:
        _NODE_BASE = {p.identifier for p in bpy.types.Node.bl_rna.properties} | {"rna_type"}
    return _NODE_BASE


def _ramp(cr):
    return (cr.interpolation, cr.color_mode, cr.hue_interpolation,
            tuple((round(e.position, 7), tuple(round(c, 7) for c in e.color)) for e in cr.elements))


def _pixels(img):
    px = np.empty(len(img.pixels), np.float32)
    img.pixels.foreach_get(px)
    return px


def _same_image(ia, ib):
    """Whether two images (this build's, the probe's) hold the same picture."""
    if ia is None or ib is None:
        return ia is ib
    if ia.packed_file is not None and ib.packed_file is not None:
        return ia.packed_file.size == ib.packed_file.size and ia.packed_file.data == ib.packed_file.data
    if ia.source == "FILE" and ib.source == "FILE" and ia.packed_file is None and ib.packed_file is None:
        return os.path.normpath(bpy.path.abspath(ia.filepath)) == \
            os.path.normpath(bpy.path.abspath(ib.filepath, library=ib.library))
    if tuple(ia.size) != tuple(ib.size):
        return False
    pa, pb = _pixels(ia), _pixels(ib)
    return pa.shape == pb.shape and float(np.abs(pa - pb).max(initial=0.0)) <= 0.5 / 255.0


def _attr(me, name):
    """(domain, data_type, values) of a mesh attribute, or None."""
    at = me.attributes.get(name)
    if at is None or at.data_type not in ATTR_KEY:
        return None
    key, comp = ATTR_KEY[at.data_type]
    vals = np.empty(len(at.data) * comp, np.float32 if at.data_type.startswith(("FLOAT", "BYTE")) else np.int32)
    at.data.foreach_get(key, vals)
    return at.domain, at.data_type, vals


def _sock(coll, ident):
    return next(s for s in coll if s.identifier == ident)


def _trees_reading(nt, seen=None):
    """{attribute name: [(tree, node)]} of the nodes that read a mesh attribute by name in `nt` and its groups."""
    seen = set() if seen is None else seen
    out = {}
    if nt is None or nt.as_pointer() in seen:
        return out
    seen.add(nt.as_pointer())
    for n in nt.nodes:
        if n.type == "ATTRIBUTE" and n.attribute_type == "GEOMETRY":
            out.setdefault(n.attribute_name, []).append((nt, n))
        elif n.type == "VERTEX_COLOR":
            out.setdefault(n.layer_name, []).append((nt, n))
        elif n.type == "GROUP" and n.node_tree is not None:
            for k, v in _trees_reading(n.node_tree, seen).items():
                out.setdefault(k, []).extend(v)
    return out


# -------------------------------------------------------------------------------------------------------- the breath
class Breath:
    def __init__(self, ctx, sp, sc, frames, mixes):
        self.ctx, self.sp, self.sc = ctx, sp, sc
        self.frames, self.mixes = frames, np.asarray(mixes, float)
        self.n = {"drivers": 0, "sockets": 0, "values": 0, "ramps": 0, "images": 0, "attributes": 0, "lights": 0,
                  "world": 0, "viewport": 0}
        self.fixed = []
        cast = set()
        for m in ctx.cast.values():
            if m.root is not None:
                cast.update(o.name for o in [m.root] + list(m.root.children_recursive))
        self.cast_objects = cast
        self.cast_mats = {s.material.name for o in bpy.data.objects if o.name in cast for s in o.material_slots
                          if s.material is not None}
        self.done_trees = set()

    def fix(self, where, why):
        self.fixed.append(f"{where}: {why}")

    # -------------------------------------------------------------------------------------------------- drivers
    def drive(self, idb, path, index, expr):
        fc = idb.driver_add(path, index) if index is not None else idb.driver_add(path)
        d = fc.driver
        d.type = "SCRIPTED"
        v = d.variables.new()
        v.name, v.type = VAR, "SINGLE_PROP"
        t = v.targets[0]
        t.id_type, t.id, t.data_path = "SCENE", self.sc, f'["{PROP}"]'
        d.expression = expr
        self.n["drivers"] += 1
        return fc

    def drive_values(self, idb, path, colour, va, vb, scalar):
        """Drivers on the channels of `path` that differ: colour channels (the first three of a colour) as colours."""
        for k, (a, b) in enumerate(zip(va, vb)):
            if abs(a - b) <= 1e-7:
                continue
            expr = BR.colour_expr(a, b, VAR) if colour and k < 3 else BR.number_expr(a, b, VAR)
            self.drive(idb, path, None if scalar else k, expr)

    # ------------------------------------------------------------------------------------------------------ run
    def run(self, blend):
        with open(blend + ".json", encoding="utf-8") as fh:
            probe = json.load(fh)
        mats = {m.name: m for m in bpy.data.materials if m.library is None and m.name not in self.cast_mats}
        groups = {g.name: g for g in bpy.data.node_groups if g.library is None}
        lights = {la.name: la for la in bpy.data.lights if la.library is None and la.users}
        world = self.sc.world
        pairs = {}                                    # this build's material -> the probe's material name
        for o in self.sc.objects:
            if o.name in self.cast_objects or o.name not in probe["slots"]:
                continue
            theirs = probe["slots"][o.name]
            for i, s in enumerate(o.material_slots):
                if s.material is None or s.material.name not in mats or i >= len(theirs) or theirs[i] is None:
                    continue
                prev = pairs.setdefault(s.material.name, theirs[i])
                if prev != theirs[i]:
                    self.fix(f"material {s.material.name}", f"the probe has {prev!r} and {theirs[i]!r} where it is")
        for name in mats:
            pairs.setdefault(name, name)
        meshes = self._attribute_meshes(mats)
        with bpy.data.libraries.load(blend, link=True) as (src, dst):
            have = set(src.materials)
            mat_names = sorted({b for b in pairs.values() if b in have})
            dst.materials = list(mat_names)
            group_names = [n for n in src.node_groups if n in groups]
            dst.node_groups = list(group_names)
            light_names = [n for n in src.lights if n in lights]
            dst.lights = list(light_names)
            world_names = [n for n in src.worlds if world is not None and n == probe.get("world")]
            dst.worlds = list(world_names)
            mesh_names = [n for n in src.meshes if n in meshes]
            dst.meshes = list(mesh_names)
        linked = [x for x in list(dst.materials) + list(dst.node_groups) + list(dst.lights) + list(dst.worlds)
                  + list(dst.meshes) if x is not None]
        libs = {x.library for x in linked if x.library is not None}
        try:
            theirs = dict(zip(mat_names, dst.materials))
            for name, m in mats.items():
                mb = theirs.get(pairs[name])
                if mb is None:
                    if m.users and name not in self.cast_mats:
                        self.fix(f"material {name}", "the probe build has none like it")
                    continue
                self.material(m, mb)
            for name, gb in zip(group_names, dst.node_groups):
                ga = groups[name]
                if gb is None:
                    continue
                if ga.bl_idname != "ShaderNodeTree":
                    if self._tree_differs(ga, gb):
                        self.fix(f"node group {name}", "a geometry or other node group that differs does not breathe")
                    continue
                self.tree(ga, gb, f"node group {name}", None)
            if world is not None and world_names and dst.worlds[0] is not None:
                self.world(world, dst.worlds[0])
            for name, lb in zip(light_names, dst.lights):
                if lb is not None:
                    self.light(lights[name], lb)
            self.attributes(meshes, dict(zip(mesh_names, dst.meshes)))
        finally:
            for lib in libs:
                bpy.data.libraries.remove(lib)
        return {**self.n, "fixed": self.fixed}

    # ------------------------------------------------------------------------------------------------ materials
    def material(self, ma, mb):
        where = f"material {ma.name}"
        va, vb = _vals(ma.diffuse_color), _vals(mb.diffuse_color)
        if _differ(va, vb):
            if "diffuse_color" in _anim_paths(ma):
                self.fix(where, "its viewport colour is animated")
            else:
                self.drive_values(ma, "diffuse_color", True, va, vb, False)
                self.n["viewport"] += 1
        if ma.use_nodes and ma.node_tree is not None and mb.node_tree is not None:
            self.tree(ma.node_tree, mb.node_tree, where, ma)

    def world(self, wa, wb):
        where = f"world {wa.name}"
        va, vb = _vals(wa.color), _vals(wb.color)
        if _differ(va, vb):
            if "color" in _anim_paths(wa):
                self.fix(where, "its colour is animated")
            else:
                self.drive_values(wa, "color", True, va, vb, False)
                self.n["world"] += 1
        if wa.use_nodes and wa.node_tree is not None and wb.node_tree is not None:
            self.tree(wa.node_tree, wb.node_tree, where, wa)

    @staticmethod
    def _links(nt):
        return {(lk.from_node.name, lk.from_socket.identifier, lk.to_node.name, lk.to_socket.identifier)
                for lk in nt.links if lk.is_valid and not lk.is_muted}

    def _tree_differs(self, ta, tb):
        if {n.name for n in ta.nodes} != {n.name for n in tb.nodes} or self._links(ta) != self._links(tb):
            return True
        for a in ta.nodes:
            b = tb.nodes[a.name]
            for sa, sb in zip(a.inputs, b.inputs):
                if hasattr(sa, "default_value") and not sa.is_linked:
                    x, y = _vals(sa.default_value), _vals(sb.default_value)
                    if x is not None and y is not None and _differ(x, y):
                        return True
        return False

    def tree(self, ta, tb, where, owner):
        """Drive what differs between this build's node tree `ta` and the probe's `tb` (`owner`: the material or world
        that holds `ta`, whose own animation names its nodes as node_tree.nodes[...])."""
        if ta.as_pointer() in self.done_trees:
            return
        self.done_trees.add(ta.as_pointer())
        names_a, names_b = {n.name for n in ta.nodes}, {n.name for n in tb.nodes}
        if names_a != names_b:
            self.fix(where, f"nodes only in one build ({', '.join(sorted(names_a ^ names_b)[:4])})")
            return
        if self._links(ta) != self._links(tb):
            self.fix(where, "its links differ between the builds")
            return
        anim = _anim_paths(ta) | _anim_paths(owner, prefix="node_tree.")
        crossfades = []
        for a in list(ta.nodes):
            b = tb.nodes[a.name]
            nw = f"{where}: node {a.name}"
            if a.bl_idname != b.bl_idname:
                self.fix(nw, f"{a.bl_idname} here, {b.bl_idname} in the probe")
                continue
            crossfades += self.node_props(a, b, nw, anim)
            for io in ("inputs", "outputs"):
                sa_all, sb_all = getattr(a, io), getattr(b, io)
                if len(sa_all) != len(sb_all):
                    self.fix(nw, f"its {io} differ")
                    continue
                if io == "outputs" and a.type not in ("RGB", "VALUE"):
                    continue
                for i, (sa, sb) in enumerate(zip(sa_all, sb_all)):
                    if not hasattr(sa, "default_value") or sa.identifier != sb.identifier:
                        continue
                    if io == "inputs" and sa.is_linked:
                        continue
                    va, vb = _vals(sa.default_value), _vals(sb.default_value)
                    if va is None or vb is None:
                        if sa.default_value != sb.default_value:
                            self.fix(nw, f"{io} {sa.name}: {sa.default_value!r} / {sb.default_value!r}")
                        continue
                    if len(va) != len(vb) or not _differ(va, vb):
                        continue
                    path = f'nodes["{_esc(a.name)}"].{io}[{i}].default_value'
                    if path in anim:
                        self.fix(nw, f"{io} {sa.name} is animated")
                        continue
                    if sa.type in ("INT", "BOOLEAN"):
                        self.fix(nw, f"{io} {sa.name}: whole numbers {va} / {vb}")
                        continue
                    self.drive_values(ta, path, sa.type == "RGBA", va, vb, len(va) == 1 and not hasattr(
                        sa.default_value, "__len__"))
                    self.n["values" if io == "outputs" else "sockets"] += 1
        for kind, a, b in crossfades:
            if kind == "ramp":
                self.ramp(ta, a, b)
            else:
                self.image(ta, a, b, where)

    def node_props(self, a, b, where, anim):
        """Compare a node's own settings: a ramp or an image that differs is crossfaded (returned as ("ramp" | "image",
        a, b)); another setting that differs is fixed."""
        out = []
        base = _node_base()
        for p in a.bl_rna.properties:
            k = p.identifier
            if k in base or p.type == "COLLECTION":
                continue
            va, vb = getattr(a, k), getattr(b, k)
            if k == "color_ramp":
                if _ramp(va) != _ramp(vb):
                    if any(x.startswith(f'nodes["{_esc(a.name)}"].color_ramp') for x in anim):
                        self.fix(where, "its colour ramp is animated")
                    else:
                        out.append(("ramp", a, b))
                continue
            if k == "image":
                if not _same_image(va, vb):
                    out.append(("image", a, b))
                continue
            if p.type == "POINTER":
                if isinstance(va, bpy.types.ID) or isinstance(vb, bpy.types.ID):
                    if (va.name if va is not None else None) != (vb.name if vb is not None else None):
                        self.fix(where, f"{k}: {getattr(va, 'name', None)!r} / {getattr(vb, 'name', None)!r}")
                continue
            x, y = _vals(va), _vals(vb)
            if x is not None and y is not None:
                if len(x) != len(y) or _differ(x, y):
                    self.fix(where, f"{k}: {tuple(x)} / {tuple(y)}")
            elif va != vb:
                self.fix(where, f"{k}: {va!r} / {vb!r}")
        return out

    # ------------------------------------------------------------------------------------------------ crossfades
    def _node(self, nt, idname, name, loc):
        n = nt.nodes.new(idname)
        n.name = n.label = name
        n.location = loc
        return n

    def _mix(self, nt, sa, sb, tag, loc):
        """The output socket that is `sa` at mix 0 and `sb` at mix 1: colours through Gamma 1/2.2, Mix, Gamma 2.2."""
        x, y = loc
        mx = self._node(nt, "ShaderNodeMix", f"{tag}.mix", (x + 200, y))
        mx.data_type = {"RGBA": "RGBA", "VECTOR": "VECTOR"}.get(sa.type, "FLOAT")
        mx.blend_type, mx.clamp_factor = "MIX", True
        kind = {"RGBA": "Color", "VECTOR": "Vector"}.get(sa.type, "Float")
        ins = [_sock(mx.inputs, f"A_{kind}"), _sock(mx.inputs, f"B_{kind}")]
        if sa.type == "RGBA":
            for j, (s, inp) in enumerate(((sa, ins[0]), (sb, ins[1]))):
                g = self._node(nt, "ShaderNodeGamma", f"{tag}.in{j}", (x + 100, y - 60 * j))
                g.inputs["Gamma"].default_value = 1.0 / BR.GAMMA
                nt.links.new(s, g.inputs["Color"])
                nt.links.new(g.outputs["Color"], inp)
        else:
            nt.links.new(sa, ins[0])
            nt.links.new(sb, ins[1])
        fac = list(mx.inputs).index(_sock(mx.inputs, "Factor_Float"))
        self.drive(nt, f'nodes["{_esc(mx.name)}"].inputs[{fac}].default_value', None, VAR)
        out = _sock(mx.outputs, f"Result_{kind}")
        if sa.type == "RGBA":
            g = self._node(nt, "ShaderNodeGamma", f"{tag}.out", (x + 300, y))
            g.inputs["Gamma"].default_value = BR.GAMMA
            nt.links.new(out, g.inputs["Color"])
            out = g.outputs["Color"]
        return out

    def crossfade(self, nt, a, b_node):
        """Route every link from `a`'s outputs through a mix with the same output of `b_node` (the probe's version of
        `a`, beside it in `nt`)."""
        for sa in a.outputs:
            links = [lk for lk in nt.links if lk.from_socket == sa]
            if not links:
                continue
            sb = _sock(b_node.outputs, sa.identifier)
            out = self._mix(nt, sa, sb, f"{a.name}{SUFFIX}.{sa.identifier}", (a.location.x, a.location.y))
            for lk in links:
                to = lk.to_socket
                nt.links.remove(lk)
                nt.links.new(out, to)

    def _same_inputs(self, nt, a, b_node):
        """`b_node` reads what `a` reads: the same links into its inputs, or the same values."""
        for sa in a.inputs:
            sb = _sock(b_node.inputs, sa.identifier)
            if sa.is_linked:
                nt.links.new(sa.links[0].from_socket, sb)
            elif hasattr(sa, "default_value"):
                sb.default_value = sa.default_value

    def ramp(self, nt, a, b):
        n = self._node(nt, "ShaderNodeValToRGB", f"{a.name}{SUFFIX}", (a.location.x, a.location.y - 260))
        cr, src = n.color_ramp, b.color_ramp
        cr.interpolation, cr.color_mode, cr.hue_interpolation = src.interpolation, src.color_mode, src.hue_interpolation
        stops = sorted(((e.position, tuple(e.color)) for e in src.elements), key=lambda s: s[0])
        while len(cr.elements) > 1:                              # one stop, then each new one inserted where it goes
            cr.elements.remove(cr.elements[-1])
        cr.elements[0].position, cr.elements[0].color = stops[0]
        for p, c in stops[1:]:
            cr.elements.new(p).color = c
        self._same_inputs(nt, a, n)
        self.crossfade(nt, a, n)
        self.n["ramps"] += 1

    def image(self, nt, a, b, where):
        ib = b.image
        if ib is None or a.image is None:
            self.fix(f"{where}: node {a.name}", "an image in one build only")
            return
        img = self._local_image(a.image, ib)
        n = self._node(nt, "ShaderNodeTexImage", f"{a.name}{SUFFIX}", (a.location.x, a.location.y - 300))
        for k in ("interpolation", "projection", "projection_blend", "extension"):
            setattr(n, k, getattr(a, k))
        n.image = img
        self._same_inputs(nt, a, n)
        self.crossfade(nt, a, n)
        self.n["images"] += 1

    def _local_image(self, ia, ib):
        """This build's copy of the probe's picture `ib` (packed), named after `ia`."""
        name = f"{ia.name}{SUFFIX}"
        have = bpy.data.images.get(name)
        if have is not None and have.library is None:
            return have
        if ib.source == "FILE" and ib.packed_file is None:
            img = bpy.data.images.load(bpy.path.abspath(ib.filepath, library=ib.library), check_existing=False)
            img.name = name
        else:
            w, h = ib.size
            img = bpy.data.images.new(name, w, h, alpha=True, float_buffer=ib.is_float)
            img.pixels.foreach_set(_pixels(ib))
            img.pack()
        img.colorspace_settings.name = ib.colorspace_settings.name
        img.alpha_mode = ib.alpha_mode
        return img

    # ------------------------------------------------------------------------------------------------ attributes
    def _attribute_meshes(self, mats):
        """{mesh name: {attribute: [(tree, node)]}} of the meshes (not the cast's) whose materials read attributes."""
        out = {}
        for o in self.sc.objects:
            if o.type != "MESH" or o.name in self.cast_objects or o.data is None or o.data.library is not None:
                continue
            reads = {}
            for s in o.material_slots:
                m = s.material
                if m is None or m.name not in mats or not m.use_nodes:
                    continue
                for k, v in _trees_reading(m.node_tree).items():
                    reads.setdefault(k, []).extend(v)
            if reads:
                out.setdefault(o.data.name, {}).update(reads)
        return out

    def attributes(self, meshes, theirs):
        """Every attribute a material reads that differs on some mesh: the probe's values beside it as `<name>~breathe`
        on that mesh and on every other mesh the reading trees draw (their own values there), the trees reading both."""
        patch = {}                                   # attribute -> [(tree, node)] to crossfade
        differs = {}                                 # (mesh, attribute) -> the probe's values
        for mname, reads in meshes.items():
            me, mb = bpy.data.meshes.get(mname), theirs.get(mname)
            if me is None or mb is None:
                continue
            for att, nodes in reads.items():
                if not att:
                    continue
                xa, xb = _attr(me, att), _attr(mb, att)
                if xa is None or xb is None:
                    continue
                if xa[:2] != xb[:2] or xa[2].shape != xb[2].shape:
                    self.fix(f"mesh {mname}: attribute {att}", "its kind or size differs between the builds")
                    continue
                if np.array_equal(xa[2], xb[2]):
                    continue
                if xa[2].dtype.kind != "f":
                    self.fix(f"mesh {mname}: attribute {att}", "whole numbers that differ")
                    continue
                differs[(mname, att)] = xb[2]
                patch.setdefault(att, {})
                for nt, n in nodes:
                    patch[att][(nt.as_pointer(), n.name)] = (nt, n)
        if not patch:
            return
        for mname, reads in meshes.items():
            me = bpy.data.meshes.get(mname)
            for att in patch:
                if att not in reads or me is None:
                    continue
                xa = _attr(me, att)
                if xa is None:
                    continue
                vals = differs.get((mname, att), xa[2])
                new = me.attributes.get(att + SUFFIX) or me.attributes.new(att + SUFFIX, xa[1], xa[0])
                new.data.foreach_set(ATTR_KEY[xa[1]][0], vals)
                if (mname, att) in differs:
                    self.n["attributes"] += 1
        for att, nodes in patch.items():
            for nt, n in nodes.values():
                b = self._node(nt, n.bl_idname, f"{n.name}{SUFFIX}", (n.location.x, n.location.y - 220))
                if n.type == "ATTRIBUTE":
                    b.attribute_type = n.attribute_type
                setattr(b, READS[n.type], att + SUFFIX)
                self.crossfade(nt, n, b)

    # ---------------------------------------------------------------------------------------------------- lights
    def light(self, la, lb):
        users = [o.name for o in bpy.data.objects if o.data == la]
        where = f"light {users[0] if users else la.name}"
        anim = _anim_paths(la)
        va, vb = _vals(la.color), _vals(lb.color)
        if _differ(va, vb):
            if "color" in anim:
                self.fix(where, "its colour is animated")
            else:
                self.drive_values(la, "color", True, va, vb, False)
        k = 1.0
        for name in users + [la.name]:
            f = BR.light_factor(self.sp, name)
            if f != 1.0:
                k = f
                break
        ea, eb = float(la.energy), float(lb.energy)
        ad = la.animation_data
        driven = next((d for d in (ad.drivers if ad else ()) if d.data_path == "energy"), None)
        keyed = ad.action.fcurves.find("energy") if ad and ad.action else None
        changed = _differ(_vals(la.color), _vals(lb.color))
        if driven is not None:
            if k != 1.0:
                d = driven.driver
                if d.type != "SCRIPTED":
                    self.fix(where, "its strength is driven (not by an expression): no `lights` factor")
                else:
                    v = d.variables.new()
                    v.name, v.type = "mk_m", "SINGLE_PROP"
                    v.targets[0].id_type, v.targets[0].id, v.targets[0].data_path = "SCENE", self.sc, f'["{PROP}"]'
                    d.expression = f"({d.expression}) * lerp(1, {k:.9g}, mk_m)"
                    changed = True
        elif keyed is not None or "energy" in anim:
            bad = keyed is None
            fb = (lb.animation_data.action.fcurves.find("energy")
                  if lb.animation_data and lb.animation_data.action else None)
            if bad:
                self.fix(where, "its strength is animated by NLA strips")
            else:
                e_a = np.array([keyed.evaluate(f) for f in self.frames])
                e_b = np.array([fb.evaluate(f) for f in self.frames]) if fb is not None else np.full(len(e_a), eb)
                if k != 1.0 or not np.allclose(e_a, e_b, atol=1e-6):
                    m = self.mixes
                    vals = (e_a + (e_b - e_a) * m) * (1.0 + (k - 1.0) * m)
                    K.set_fcurve(la, "energy", 0, self.frames, vals, interp="LINEAR")
                    changed = True
        elif abs(ea - eb) > 1e-7 or k != 1.0:
            expr = BR.number_expr(ea, eb, VAR) if abs(ea - eb) > 1e-7 else f"{ea:.9g}"
            if k != 1.0:
                expr = f"{expr} * lerp(1, {k:.9g}, {VAR})"
            self.drive(la, "energy", None, expr)
            changed = True
        if changed:
            self.n["lights"] += 1
