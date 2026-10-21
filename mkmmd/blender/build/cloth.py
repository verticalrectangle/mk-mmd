"""cloth: the cloth copies a cast member's rig declares (rig.json `cloth`, mkmmd.model.cloth), or that its table makes
from the model's own garments (`make`, mkmmd.core.garment), simulated with Blender's cloth over the build's frames, the
garment's faces following the simulated cloth (docs/design.md: Cloth).

[cloth.<cast>] switches it on for that member (every key optional):
  enabled = true   false: no cloth (the table can stay)
  pieces = [...]   the copies to simulate, by name (default every one the rig declares or the table makes)
  quality, mass, tension, shear, bending, air, distance, hold, grip: the cloth, over the garment's own (rig.json): steps
                   a frame, kg a vertex, stiffness against stretching, shearing and bending, air damping, the gap it keeps
                   off the body (m), how much the free cloth is held where the armature carries it (0..1, each vertex
                   at least its copy's holds) and how hard what is held is pulled back there (Blender's pin stiffness)
  cache = true     reuse a simulation of the same copy, settings and motion (<cache>/cloth/<key>.pc2)
  [[cloth.<cast>.make]]  a copy made from the model's own faces (a model mk did not build): name, materials (the
                   garment: its outside is the cloth, the rest follows it), collide (the materials of the body it
                   collides with), trims (materials of what lies on it, a button or a badge, within `gap` m: each piece
                   rides the cloth whole), pin {above, fade}, sleeves {hold, collides}, inset, reach, thick
                   (mkmmd.core.garment.make), gap (0.01)

Per copy: <root>_<name>_cloth, the simulated cloth (hidden from render), rides the armature and plays its frames from
<blend>.cloth/<cast>.<name>.pc2 (a Mesh Cache modifier: the saved scene holds no physics), and the model's faces of the
copy's materials move to a mesh of their own, <root>_<name>, bound to that cloth on the first frame (Surface Deform) as
far as the copy is free there; a made copy's trims move to <root>_<name>_trims*, one object per piece, each parented to
the three vertices of the cloth triangle nearest it. The simulation: the copy skinned to the armature, pinned where its
pins say and the rest held by the larger of its holds and `hold`, its rest shape the skinned copy's on every frame
(Dynamic Mesh), colliding with the copy's skin (skinned too), baked from the build's first frame to its last. Fold
shapes on a piece the cloth simulates are left to it (the folds stage skips them)."""
import hashlib
import io
import json
import os
import shutil
import struct
import time

import bmesh
import bpy
import numpy as np
from mathutils import Matrix
from mathutils.bvhtree import BVHTree
from mathutils.kdtree import KDTree

from ...core import garment as GM
from .. import scene as S
from . import BuildError, collection
from .cast import material_name

STAGE = 10                      # the set-up's version: a cached simulation from an older one is not reused
PIN = "mk_pin"
DEFAULTS = {"quality": 8, "mass": 0.3, "tension": 10.0, "shear": 5.0, "bending": 0.5, "air": 1.0, "distance": 0.003,
            "hold": 0.3, "grip": 1.0}


def made(ctx, name):
    """{copy name: its table} of the copies `[[cloth.<name>.make]]` makes from the model's own faces (their keys are
    checked with the table's, mkmmd.core.tables.CLOTH)."""
    out = {}
    for e in (ctx.section("cloth", name) or {}).get("make") or []:
        if not e.get("name") or not e.get("materials") or not e.get("collide"):
            raise BuildError(f"cloth.{name}.make: each copy needs a name, its materials and the materials it collides with")
        if e["name"] in out:
            raise BuildError(f"cloth.{name}.make: two copies are named {e['name']!r}")
        out[e["name"]] = e
    return out


def simulated(ctx, name, m):
    """The names of the cloth copies this stage simulates for cast member `name`: none without its table (an empty one
    switches it on) or with enabled = false; else `pieces` (default every copy its rig declares or the table makes)."""
    if name not in ctx.section("cloth"):
        return []
    spec = ctx.section("cloth", name) or {}
    if not spec.get("enabled", True):
        return []
    have = [c["name"] for c in (m.rig or {}).get("cloth") or []]
    have += [k for k in made(ctx, name) if k not in have]
    want = list(spec.get("pieces") or have)
    unknown = [p for p in want if p not in have]
    if unknown:
        raise BuildError(f"cloth.{name}: no cloth copy {unknown} in its rig.json or its table's make (it has {have or 'none'})")
    return want


def load(m, entry):
    """(arrays {name: array}, sha1) of a cloth entry, read from its file beside the member's model."""
    path = os.path.join(os.path.dirname(m.pmx), entry["file"])
    try:
        with open(path, "rb") as fh:
            data = fh.read()
    except OSError:
        raise BuildError(f"cloth.{m.name}: the cloth copy {entry['name']!r} needs {path} (mk model build writes it)") from None
    sha = hashlib.sha1(data).hexdigest()
    if entry.get("sha1") and sha != entry["sha1"]:
        raise BuildError(f"cloth.{m.name}: {path} is not the cloth copy its rig.json describes (rebuild the model)")
    with np.load(io.BytesIO(data), allow_pickle=False) as z:
        return {k.split(".", 1)[1]: z[k] for k in entry["arrays"]}, sha


def _mesh(name, verts, faces, coll, arm, bones, weights):
    """A mesh object riding the armature (its local space the armature's), skinned with `weights` (n, len(bones))."""
    me = bpy.data.meshes.new(name)
    me.from_pydata(np.asarray(verts, float).tolist(), [], np.asarray(faces).tolist())
    me.update()
    ob = bpy.data.objects.new(name, me)
    coll.objects.link(ob)
    ob.parent = arm
    for j, b in enumerate(bones):
        idx = np.flatnonzero(weights[:, j] > 0.0)
        if len(idx):
            vg = ob.vertex_groups.get(b) or ob.vertex_groups.new(name=b)
            for i in idx:
                vg.add([int(i)], float(weights[i, j]), "REPLACE")
    md = ob.modifiers.new("armature", "ARMATURE")
    md.object = arm
    return ob


def unbutton(sim, opened, f0, f1):
    """A shape key `mk_open` on the copy `sim` with its points at `opened` (n, 3), eased in from frame f0 to f1: the
    unbuttoned shape the cloth's rest (Dynamic Mesh) and goal take on over the settle, so its front swings open."""
    sim.shape_key_add(name="Basis", from_mix=False)
    k = sim.shape_key_add(name="mk_open", from_mix=False)
    k.data.foreach_set("co", np.asarray(opened, np.float32).ravel())
    for f, v in ((f0, 0.0), (f1, 1.0)):
        k.value = v
        k.keyframe_insert("value", frame=f)


def _coords(ob, dg, buf):
    """(n, 3) the evaluated mesh's vertex coordinates (the object's own space)."""
    ev = ob.evaluated_get(dg)
    me = ev.to_mesh()
    me.vertices.foreach_get("co", buf)
    ev.to_mesh_clear()
    return buf.reshape(-1, 3).copy()


def motion_key(ctx, m, entry, sha, settings):
    """The hash of what a simulation depends on: the copy, its settings, the frames, and on every frame where each
    bone that carries it is in the world."""
    h = hashlib.sha256(json.dumps({"stage": STAGE, "blender": bpy.app.version_string, "name": entry["name"],
                                   "npz": sha, "settings": settings, "frames": [ctx.start, ctx.end], "fps": ctx.fps},
                                  sort_keys=True).encode())
    arm = m.arm
    idx = [arm.pose.bones.find(b) for b in entry["bones"]]
    buf = np.zeros(len(arm.pose.bones) * 16, np.float32)
    for f in ctx.frames:
        bpy.context.scene.frame_set(int(f))
        arm.pose.bones.foreach_get("matrix", buf)
        M = np.asarray(arm.matrix_world, float) @ buf.reshape(-1, 4, 4).transpose(0, 2, 1)[idx]
        h.update(np.round(M, 6).astype(np.float32).tobytes())
    return h.hexdigest()[:24]


def write_pc2(path, X):
    """Frames X (F, n, 3) as a PC2 point cache (what Blender's Mesh Cache modifier plays: sample i on its start frame
    plus i)."""
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "wb") as fh:
        fh.write(struct.pack("<12siiffi", b"POINTCACHE2\0", 1, X.shape[1], 0.0, 1.0, X.shape[0]))
        fh.write(np.ascontiguousarray(X, "<f4").tobytes())
    os.replace(tmp, path)


def edges(faces):
    """(e, 2) the unique edges of faces (f, k): quads, or a made copy's triangles."""
    Q = np.asarray(faces)
    return np.unique(np.sort(np.stack([Q, np.roll(Q, -1, axis=1)], 2).reshape(-1, 2), 1), axis=0)


def checks(X, G, faces, pins, collides, C, tris, frames, window):
    """What the simulation did over the clip's frames `window` (frame numbers, inclusive): X (F, n, 3) the cloth, G its
    goal (the copy where the armature carries it) and C (F, m, 3) the body it collides with on `frames`, in one space.
    Deepest the free cloth that collides goes into that body (mm) and on which frame, the share of its vertex-frames
    deeper than 2 mm, how far the free cloth trails its goal (mm: most of it, p99, and the most), the largest change of
    a vertex's frame-to-frame motion of the free cloth and of the pinned (mm: the pinned move as her skin does, so the
    free cloth's beyond them is the cloth's own jump), and the most an edge with a free end stretches beyond its goal
    (%); `pen_vertex` and `stretch_edge` say which (the copy's vertex indices)."""
    k = np.flatnonzero((frames >= window[0]) & (frames <= window[1]))
    free = np.flatnonzero(pins < 0.5)
    touch = np.flatnonzero((pins < 0.5) & (np.asarray(collides) > 0))
    deep, deep_at, deep_j, over = 0.0, int(frames[k[0]]), -1, 0
    T = np.asarray(tris)
    tri_e = np.sort(np.concatenate([T[:, [0, 1]], T[:, [1, 2]], T[:, [2, 0]]]), 1)
    _, inv, n = np.unique(tri_e, axis=0, return_inverse=True, return_counts=True)
    rim = (n[inv.reshape(3, -1)] == 1).any(0)                # a face on the skin's open edge: no inside there
    for i in k:
        tree = BVHTree.FromPolygons(C[i].tolist(), T.tolist(), all_triangles=True)
        for j in touch:
            loc, nrm, f, d = tree.find_nearest(X[i, j])
            if loc is not None and not rim[f] and (X[i, j] - np.asarray(loc)) @ np.asarray(nrm) < 0.0:
                over += d > 0.002
                if d > deep:
                    deep, deep_at, deep_j = d, int(frames[i]), int(j)
    A = np.linalg.norm(X[k][2:] - 2.0 * X[k][1:-1] + X[k][:-2], axis=2) if len(k) > 2 else np.zeros((1, len(pins)))
    E = edges(faces)
    E = E[(pins[E[:, 0]] < 0.5) | (pins[E[:, 1]] < 0.5)]
    L0 = np.linalg.norm(G[k][:, E[:, 0]] - G[k][:, E[:, 1]], axis=2)
    L = np.linalg.norm(X[k][:, E[:, 0]] - X[k][:, E[:, 1]], axis=2)
    lag = np.linalg.norm(X[k][:, free] - G[k][:, free], axis=2)
    pinned = np.flatnonzero(pins >= 0.5)
    ratio = (L / np.maximum(L0, 1e-9)).max(0)
    return {"pen_mm": round(deep * 1000.0, 2), "pen_frame": deep_at, "pen_vertex": deep_j,
            "pen_over_2mm": round(over / max(len(k) * len(touch), 1), 5),
            "lag_mm": [round(float(np.quantile(lag, q)) * 1000.0, 1) for q in (0.5, 0.99, 1.0)] if len(free) else [],
            "jerk_mm": round(float(A[:, free].max()) * 1000.0, 2) if len(free) else 0.0,
            "jerk_pinned_mm": round(float(A[:, pinned].max()) * 1000.0, 2) if len(pinned) else 0.0,
            "stretch_pct": round(float(ratio.max() - 1.0) * 100.0, 1) if len(E) else 0.0,
            "stretch_edge": [int(x) for x in E[int(np.argmax(ratio))]] if len(E) else []}


def simulate(ctx, sim, col, coll, A, settings):
    """Bake Blender's cloth on `sim` against `col` over the build's frames; returns (cloth (F, n, 3), its goal (F, n, 3),
    body (F, m, 3)) in the armature's space and the seconds the simulation took."""
    sc = bpy.context.scene
    pin = sim.vertex_groups.new(name=PIN)                 # goal: pinned, else held by the copy's holds or `hold`
    goal = A["pins"] + (1.0 - A["pins"]) * np.maximum(A["holds"], float(settings["hold"]))
    for i, g in enumerate(goal):
        pin.add([i], float(g), "REPLACE")
    col.modifiers.new("collision", "COLLISION")
    col.collision.thickness_outer = 0.001
    col.collision.cloth_friction = 1.0                      # skin is smooth: a bunched sleeve slides back down
    col.collision.use_culling = col.collision.use_normal = True   # cloth found inside goes back out along the normal
    md = sim.modifiers.new("cloth", "CLOTH")
    s, c = md.settings, md.collision_settings
    s.quality = int(settings["quality"])
    s.mass = float(settings["mass"])
    s.tension_stiffness = s.compression_stiffness = float(settings["tension"])
    s.shear_stiffness = float(settings["shear"])
    s.bending_stiffness = float(settings["bending"])
    s.air_damping = float(settings["air"])
    s.vertex_group_mass = PIN
    s.pin_stiffness = float(settings["grip"])
    s.use_dynamic_mesh = True                                # its rest shape is the armature's on every frame
    c.use_collision, c.use_self_collision = True, False
    c.distance_min = float(settings["distance"])
    c.collection = coll
    if not A["collides"].all():                              # Blender's group names the faces that do not collide
        free = sim.vertex_groups.new(name="mk_no_collide")
        free.add([int(i) for i in np.flatnonzero(A["collides"] == 0)], 1.0, "REPLACE")
        c.vertex_group_object_collisions = free.name
    pc = md.point_cache
    pc.frame_start, pc.frame_end = ctx.start, ctx.end
    t0 = time.time()
    with bpy.context.temp_override(scene=sc, object=sim, active_object=sim, point_cache=pc):
        bpy.ops.ptcache.bake(bake=True)
    if not pc.is_baked:
        raise BuildError(f"cloth: Blender did not bake {sim.name}")
    seconds = time.time() - t0
    frames = ctx.frames
    X = np.zeros((len(frames), len(sim.data.vertices), 3), np.float32)
    G = np.zeros_like(X)
    C = np.zeros((len(frames), len(col.data.vertices), 3), np.float32)
    bx, bc = np.zeros(X.shape[1] * 3, np.float32), np.zeros(C.shape[1] * 3, np.float32)
    for i, f in enumerate(frames):
        sc.frame_set(int(f))
        dg = bpy.context.evaluated_depsgraph_get()
        X[i], C[i] = _coords(sim, dg, bx), _coords(col, dg, bc)
    md.show_viewport = False                                 # the goal: where the armature alone carries it
    for i, f in enumerate(frames):
        sc.frame_set(int(f))
        G[i] = _coords(sim, bpy.context.evaluated_depsgraph_get(), bx)
    sim.modifiers.remove(md)
    return X, G, C, seconds


def separate(m, materials, name, near=None):
    """The faces of `materials` (PMX names) in the member's meshes moved to meshes of their own (named
    <root>_<name>), keeping their modifiers, weights and shape keys; with `near` = (KDTree of points in the armature's
    space, gap) only their loose pieces that come within `gap` metres of a point (a face centre that near), each whole:
    a brooch standing off the lapel goes all of it or none. Returns the new objects."""
    out = []
    inv = m.arm.matrix_world.inverted()
    for ob in m.meshes:
        slots = {i for i, s in enumerate(ob.material_slots) if s.material and material_name(s.material) in materials}
        if not slots or not any(p.material_index in slots for p in ob.data.polygons):
            continue
        to_arm = inv @ ob.matrix_world
        before = set(bpy.data.objects)
        vl = bpy.context.view_layer
        hidden = (ob.hide_get(), ob.hide_viewport, ob.hide_select)
        ob.hide_set(False)
        ob.hide_viewport = ob.hide_select = False
        for o in list(bpy.context.selected_objects):
            o.select_set(False)
        ob.select_set(True)
        vl.objects.active = ob
        bpy.ops.object.mode_set(mode="EDIT")
        if ob.mode != "EDIT":
            raise BuildError(f"cloth: could not edit {ob.name} to move its faces of {sorted(materials)} out")
        bm = bmesh.from_edit_mesh(ob.data)
        for seq in (bm.verts, bm.edges, bm.faces):
            for e in seq:
                e.select = False
        faces = [f for f in bm.faces if f.material_index in slots]
        if near is not None:                                        # the loose pieces of those faces, each whole
            piece, close = {}, {}
            for f0 in faces:
                if f0.index in piece:
                    continue
                piece[f0.index], stack = f0.index, [f0]
                while stack:
                    for v in stack.pop().verts:
                        for g in v.link_faces:
                            if g.material_index in slots and g.index not in piece:
                                piece[g.index] = f0.index
                                stack.append(g)
            for f in faces:
                d = near[0].find(to_arm @ f.calc_center_median())[2]
                close[piece[f.index]] = min(close.get(piece[f.index], np.inf), d)
            faces = [f for f in faces if close[piece[f.index]] <= near[1]]
        for f in faces:
            f.select_set(True)
        picked = len(faces)
        bmesh.update_edit_mesh(ob.data)
        if picked:
            bpy.ops.mesh.separate(type="SELECTED")
        bpy.ops.object.mode_set(mode="OBJECT")
        ob.select_set(False)
        ob.hide_set(hidden[0])
        ob.hide_viewport, ob.hide_select = hidden[1], hidden[2]
        for o in bpy.data.objects:
            if o not in before:
                o.name = f"{m.root.name}_{name}"
                out.append(o)
    return out


SKELETON = ("lower_body", "upper_body", "upper_body2", "neck", "head", "shoulder.L", "shoulder.R", "arm.L", "arm.R",
            "elbow.L", "elbow.R", "wrist.L", "wrist.R", "leg.L", "leg.R", "knee.L", "knee.R", "ankle.L", "ankle.R")


def _arrays(obs, arm, col, materials=None):
    """(V (n, 3) in the armature's space at rest, T (f, 3) triangles of the stacked vertices, W (n, len(col))) of the faces
    of `materials` (PMX names; every face when None) of the objects `obs`; `col` maps a bone (vertex group) to its
    column."""
    Vs, Ts, Ws, off = [], [], [], 0
    inv = arm.matrix_world.inverted()
    for ob in obs:
        me = ob.data
        me.calc_loop_triangles()
        co = np.zeros(len(me.vertices) * 3, np.float32)
        me.vertices.foreach_get("co", co)
        M = np.asarray(inv @ ob.matrix_world, float)
        V = co.reshape(-1, 3).astype(float) @ M[:3, :3].T + M[:3, 3]
        lt = me.loop_triangles
        tri = np.zeros(len(lt) * 3, np.int64)
        lt.foreach_get("vertices", tri)
        tri = tri.reshape(-1, 3)
        if materials is not None:
            mi = np.zeros(len(lt), np.int64)
            lt.foreach_get("material_index", mi)
            slots = [i for i, s in enumerate(ob.material_slots) if s.material and material_name(s.material) in materials]
            tri = tri[np.isin(mi, slots)]
        if not len(tri):
            continue
        used, inv_t = np.unique(tri, return_inverse=True)
        names = {g.index: g.name for g in ob.vertex_groups}
        W = np.zeros((len(used), len(col)))
        for i, vi in enumerate(used):
            for g in me.vertices[int(vi)].groups:
                c = col.get(names.get(g.group))
                if c is not None:
                    W[i, c] = g.weight
        Vs.append(V[used])
        Ts.append(inv_t.reshape(-1, 3) + off)
        Ws.append(W)
        off += len(used)
    if not Vs:
        return np.zeros((0, 3)), np.zeros((0, 3), np.int64), np.zeros((0, len(col)))
    return np.concatenate(Vs), np.concatenate(Ts), np.concatenate(Ws)


def make_copy(m, spec):
    """A copy `[[cloth.<cast>.make]]` makes from the member's own faces: (entry, arrays, sha1, the garment's objects, the
    trims' objects). The garment's faces move to their objects first (`separate`), so the body it collides with is the
    rest of the model."""
    piece, mats = spec["name"], set(spec["materials"])
    parts = separate(m, mats, piece)
    if not parts:
        raise BuildError(f"cloth.{m.name}.make {piece!r}: the model has no faces of {sorted(mats)}")
    arm = m.arm
    bones = [b.name for b in arm.data.bones if b.use_deform]
    col = {b: i for i, b in enumerate(bones)}
    V, T, W = _arrays(parts, arm, col)
    Vc, Tc, Wc = _arrays(m.meshes, arm, col, set(spec["collide"]))
    if not len(Tc):
        raise BuildError(f"cloth.{m.name}.make {piece!r}: collide: the model has no faces of {sorted(spec['collide'])}")
    smap = S.semantic_map(arm)
    seg = [arm.data.bones[smap[s]] for s in SKELETON if s in smap]
    A = np.array([b.head_local[:] for b in seg], float)
    B = np.array([b.tail_local[:] for b in seg], float)
    roots = [arm.data.bones[smap[s]] for s in ("arm.L", "arm.R") if s in smap]
    under = {b.name for r in roots for b in [r, *r.children_recursive]}
    try:
        arrays = GM.make(V, T, W, A, B, np.array([b in under for b in bones]), (Vc, Tc, Wc), pin=spec.get("pin"),
                         sleeves=spec.get("sleeves"), inset=float(spec.get("inset", 0.003)),
                         reach=float(spec.get("reach", 0.15)), thick=float(spec.get("thick", 0.003)))
    except (KeyError, ValueError) as e:
        raise BuildError(f"cloth.{m.name}.make {piece!r}: {e}") from None
    keep = np.flatnonzero((arrays["sim_weights"] > 0).any(0) | (arrays["collide_weights"] > 0).any(0))
    arrays["sim_weights"], arrays["collide_weights"] = arrays["sim_weights"][:, keep], arrays["collide_weights"][:, keep]
    entry = {"name": piece, "materials": sorted(mats), "bones": [bones[i] for i in keep], "settings": {}}
    h = hashlib.sha1(json.dumps(spec, sort_keys=True, default=str).encode())
    for k in sorted(arrays):
        h.update(np.ascontiguousarray(arrays[k]).tobytes())
    trims = []
    if spec.get("trims"):
        kd = KDTree(len(V))
        for i, p in enumerate(V):
            kd.insert(p, i)
        kd.balance()
        trims = separate(m, set(spec["trims"]), f"{piece}_trims", near=(kd, float(spec.get("gap", 0.01))))
    return entry, arrays, h.hexdigest(), parts, trims


def ride(trims, sim, rest, faces, frame):
    """Each loose piece of the `trims` objects as an object of its own, parented to the three vertices of the cloth
    triangle (of `sim`, rest positions `rest` and `faces`) nearest its centre: it moves and turns with that triangle,
    never stretched. Returns the pieces."""
    pieces = []
    vl = bpy.context.view_layer
    for ob in trims:
        before = set(bpy.data.objects)
        for o in list(bpy.context.selected_objects):
            o.select_set(False)
        ob.select_set(True)
        vl.objects.active = ob
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.mesh.select_all(action="SELECT")
        bpy.ops.mesh.separate(type="LOOSE")
        bpy.ops.object.mode_set(mode="OBJECT")
        ob.select_set(False)
        pieces += [ob] + [o for o in bpy.data.objects if o not in before]
    centres = np.asarray(rest, float)[np.asarray(faces)].mean(1)
    inv = sim.parent.matrix_world.inverted() if sim.parent else Matrix.Identity(4)
    bpy.context.scene.frame_set(int(frame))
    tri = []
    for ob in pieces:                                               # the triangle nearest each piece's centre
        co = np.zeros(len(ob.data.vertices) * 3, np.float32)
        ob.data.vertices.foreach_get("co", co)
        M = np.asarray(inv @ ob.matrix_world, float)
        c = (co.reshape(-1, 3).astype(float) @ M[:3, :3].T + M[:3, 3]).mean(0)
        tri.append(np.asarray(faces)[int(np.argmin(np.linalg.norm(centres - c, axis=1)))])
    for k, ob in enumerate(pieces):
        ob.name = f"{sim.name[:-len('_cloth')]}_trims{k}"
        keep_world = ob.matrix_world.copy()
        for md in list(ob.modifiers):
            ob.modifiers.remove(md)
        ob.parent = sim
        ob.parent_type = "VERTEX_3"
        ob.parent_vertices = tuple(int(x) for x in tri[k])
        ob.matrix_parent_inverse = Matrix.Identity(4)
        ob.matrix_basis = Matrix.Identity(4)
        dg = bpy.context.evaluated_depsgraph_get()
        dg.update()
        ob.matrix_parent_inverse = ob.evaluated_get(dg).matrix_world.inverted() @ keep_world
    return pieces


def follow(ob, rest, pins):
    """(n,) how much each vertex of `ob` follows the cloth: 1 minus the pins of the copy (`rest` (n, 3), in the space of
    `ob`'s own vertices) near it, by inverse distance over its three nearest: what a band holds keeps its skin's motion
    exactly, the free knit goes with the cloth."""
    kd = KDTree(len(rest))
    for i, p in enumerate(rest):
        kd.insert(p, i)
    kd.balance()
    out = np.zeros(len(ob.data.vertices))
    for v in ob.data.vertices:
        near = kd.find_n(v.co, 3)
        w = np.array([1.0 / max(d, 1e-6) for _, _, d in near])
        out[v.index] = 1.0 - float(np.dot(w, [pins[i] for _, i, _ in near]) / w.sum())
    return out


def group(ob, name, w):
    """A vertex group `name` on `ob` with the weights `w` (n,), its vertices of weight 0 left out; returns its name."""
    vg = ob.vertex_groups.new(name=name)
    for i in np.flatnonzero(w > 0.0).tolist():
        vg.add([i], float(w[i]), "REPLACE")
    return vg.name


def own_colours(ob, rest, faces, colours):
    """(n,) the part of a lapped copy (`rest` (m, 3), `faces`, its vertices' `colours`: garment.lap_colours) each vertex of
    `ob` follows (garment.bound_colours): one within 2 mm of the copy that of the face nearest it, any other its own
    side's through the garment's edges."""
    F = np.asarray(faces)
    tree = BVHTree.FromPolygons(np.asarray(rest, float).tolist(), F.tolist())
    fc = np.array([np.bincount(colours[f]).argmax() for f in F])
    n = len(ob.data.vertices)
    co = np.zeros(n * 3, np.float32)
    ob.data.vertices.foreach_get("co", co)
    R = co.reshape(-1, 3).astype(float)
    on, at = np.zeros(n, bool), np.zeros(n, np.int64)
    for i, p in enumerate(R.tolist()):
        _, _, fi, d = tree.find_nearest(p)
        if fi is not None:
            on[i], at[i] = d < 0.002, fc[fi]
    E = np.zeros(len(ob.data.edges) * 2, np.int32)
    ob.data.edges.foreach_get("vertices", E)
    return GM.bound_colours(R, E.reshape(-1, 2), on, at)


def play(ob, pc2, start):
    """The cloth's frames on `ob` (a copy, or a part of one: the same vertices): a Mesh Cache of `pc2` from frame
    `start`, triangulated the same way on every frame (what the garment binds to: triangles, always valid) and set a
    tenth of a millimetre off the garment it carries (the copy's corners are garment points, and one lying exactly on a
    cloth corner breaks Surface Deform's weights); hidden from render."""
    mc = ob.modifiers.new("frames", "MESH_CACHE")
    mc.cache_format, mc.filepath, mc.frame_start = "PC2", pc2, float(start)
    tri = ob.modifiers.new("faces", "TRIANGULATE")
    tri.quad_method = "FIXED"
    off = ob.modifiers.new("off", "DISPLACE")
    off.strength, off.mid_level = 1e-4, 0.0
    ob.hide_render = True
    ob.display_type = "WIRE"


def lapped_part(sim, faces, colours, k, coll, pc2, start):
    """Part `k` of a copy that laps over itself (`colours`: garment.lap_colours): an object with all of `sim`'s vertices
    (it plays the same frames) and those of its `faces` with a vertex of colour `k`, riding the armature like it: what the
    garment's vertices on that side of the lap bind to, so a jacket's closure does not tie one panel to the other."""
    co = np.zeros(len(sim.data.vertices) * 3, np.float32)
    sim.data.vertices.foreach_get("co", co)
    F = np.asarray(faces)
    me = bpy.data.meshes.new(f"{sim.name}_lap{k}")
    me.from_pydata(co.reshape(-1, 3).astype(float).tolist(), [], F[(colours[F] == k).any(1)].tolist())
    me.update()
    ob = bpy.data.objects.new(me.name, me)
    coll.objects.link(ob)
    ob.parent = sim.parent
    ob.matrix_parent_inverse = sim.matrix_parent_inverse.copy()
    ob.matrix_basis = sim.matrix_basis.copy()
    play(ob, pc2, start)
    return ob


def bind(ob, target, frame, group, sparse=False):
    """A Surface Deform on `ob` right after its armature, named after the vertex group `group`, following `target` as
    much as that group says, bound on `frame`; `sparse` binds the group's vertices only."""
    md = ob.modifiers.new(group, "SURFACE_DEFORM")
    md.target = target
    md.vertex_group = group
    md.use_sparse_bind = sparse
    at = next((i for i, x in enumerate(ob.modifiers) if x.type == "ARMATURE"), -1)
    ob.modifiers.move(len(ob.modifiers) - 1, at + 1)
    bpy.context.scene.frame_set(int(frame))
    with bpy.context.temp_override(object=ob, active_object=ob):
        bpy.ops.object.surfacedeform_bind(modifier=md.name)
    dg = bpy.context.evaluated_depsgraph_get()
    dg.update()
    ob.evaluated_get(dg)
    if not md.is_bound:
        raise BuildError(f"cloth: {ob.name} could not be bound to {target.name}")
    return md


def run(ctx):
    ctx.check_tables("cloth")
    out = {}
    sc = bpy.context.scene
    for name, m in ctx.cast.items():
        want = simulated(ctx, name, m)
        if not want:
            if name in ctx.section("cloth") and (ctx.section("cloth", name) or {}).get("enabled", True):
                ctx.log("WARNING", f"cloth.{name}: the model has no cloth copy (rig.json `cloth`) and the table makes "
                                   f"none, so nothing is simulated")
            continue
        spec = ctx.section("cloth", name) or {}
        entries = {c["name"]: c for c in (m.rig or {}).get("cloth") or []}
        makes = made(ctx, name)
        twice = sorted(set(entries) & set(makes))
        if twice:
            raise BuildError(f"cloth.{name}.make: {twice} is a cloth copy its rig.json has already: give it another name")
        out[name] = {}
        for piece in want:
            parts, trims = None, []
            if piece in makes:
                e, A, sha, parts, trims = make_copy(m, makes[piece])
            else:
                e = entries[piece]
                A, sha = load(m, e)
                missing = [b for b in e["bones"] if b not in m.arm.data.bones]
                if missing:
                    raise BuildError(f"cloth.{name}: the cloth copy {piece!r} is weighted to bones the armature lacks: "
                                     f"{missing}")
            settings = dict(DEFAULTS, **(e.get("settings") or {}))
            settings.update({k: spec[k] for k in DEFAULTS if k in spec})
            coll = collection(f"{m.root.name}_{piece}_cloth")
            sim = _mesh(f"{m.root.name}_{piece}_cloth", A["sim"], A["sim_faces"], coll, m.arm, e["bones"],
                        A["sim_weights"])
            op = makes[piece].get("open") if piece in makes else None
            if op:                                                  # unbuttoned: the front swings open over the settle
                o = op if isinstance(op, dict) else {"deg": op}
                free = (1.0 - A["pins"]) * (A["holds"] == 0.0)
                opened = GM.open_front(A["sim"], A["sim_faces"], free, float(o["deg"]), below=o.get("below"),
                                       fade=float(o.get("fade", 0.06)))
                unbutton(sim, opened, ctx.start, ctx.start + ctx.settle)
            key = motion_key(ctx, m, e, sha, settings)
            cdir = os.path.join(ctx.cache, "cloth")
            os.makedirs(cdir, exist_ok=True)
            cached = os.path.join(cdir, f"{key}.pc2")
            rep_path = os.path.join(cdir, f"{key}.json")
            if spec.get("cache", True) and os.path.exists(cached) and os.path.exists(rep_path):
                with open(rep_path, encoding="utf-8") as fh:
                    rep = dict(json.load(fh), cached=True)
            else:
                col = _mesh(f"{m.root.name}_{piece}_body", A["collide"], A["collide_faces"], coll, m.arm, e["bones"],
                            A["collide_weights"])
                X, G, C, seconds = simulate(ctx, sim, col, coll, A, settings)
                bpy.data.objects.remove(col, do_unlink=True)
                rep = {"seconds": round(seconds, 1), "per_frame_ms": round(seconds * 1000.0 / len(ctx.frames), 1),
                       **checks(X, G, A["sim_faces"], A["pins"], A["collides"], C, A["collide_faces"], ctx.frames,
                                (min(ctx.frame0, ctx.end - 2), ctx.end))}         # a still: its frame and two before
                write_pc2(cached, X)
                with open(rep_path, "w", encoding="utf-8") as fh:
                    json.dump(rep, fh)
                rep["cached"] = False
            sim.modifiers.remove(sim.modifiers["armature"])        # the frames hold where the armature took it too
            # the scene keeps the frames beside the .blend, played on the cloth by a Mesh Cache
            side = os.path.splitext(ctx.path(ctx.out))[0] + ".cloth" if ctx.out else cdir
            os.makedirs(side, exist_ok=True)
            pc2 = os.path.join(side, f"{name}.{piece}.pc2")
            if os.path.abspath(pc2) != os.path.abspath(cached):
                shutil.copyfile(cached, pc2)
            play(sim, pc2, ctx.start)
            parts = parts or separate(m, set(e["materials"]), piece)
            if not parts:
                raise BuildError(f"cloth.{name}: the model has no faces of {e['materials']} for {piece!r} to follow")
            # a copy that laps over itself (a jacket's closure): each side of the lap binds to its own part of the cloth
            laps = GM.laps(A["sim"], A["sim_faces"], np.asarray(A["pins"]) < 1.0)
            colours = GM.lap_colours(A["sim"], A["sim_faces"], laps)
            lapped = [lapped_part(sim, A["sim_faces"], colours, k, coll, pc2, ctx.start)
                      for k in range(int(colours.max()) + 1)] if colours.any() else []
            for ob in parts:
                w = follow(ob, A["sim"], A["pins"])
                if not lapped:
                    bind(ob, sim, ctx.start, group(ob, "mk_cloth", w))
                    continue
                own = own_colours(ob, A["sim"], A["sim_faces"], colours)
                for k, target in enumerate(lapped):
                    if (w * (own == k) > 0.0).any():
                        bind(ob, target, ctx.start, group(ob, f"mk_cloth{k}", w * (own == k)), sparse=True)
            riders = ride(trims, sim, A["sim"], A["sim_faces"], ctx.start) if trims else []
            sc.frame_set(ctx.frame0)
            rep.update(verts=len(A["sim"]), faces=len(A["sim_faces"]), collide_faces=len(A["collide_faces"]),
                       frames=[ctx.start, ctx.end], settings=settings, key=key, pc2=pc2,
                       follows={ob.name: len(ob.data.vertices) for ob in parts}, laps=len(laps),
                       lapped=[ob.name for ob in lapped])
            rest = np.asarray(A["sim"], float)                      # where on the garment, in its rest space
            if rep.get("pen_vertex", -1) >= 0:
                rep["pen_at"] = [round(float(x), 3) for x in rest[rep["pen_vertex"]]]
            if rep.get("stretch_edge"):
                rep["stretch_at"] = [round(float(x), 3) for x in rest[rep["stretch_edge"]].mean(0)]
            if piece in makes:
                rep["made"] = {"trims": len(riders), "pinned": int((A["pins"] >= 0.5).sum()),
                               "sleeves": int((A["holds"] > 0.0).sum())}
            out[name][piece] = rep
            ctx.log("cloth", name, piece, "cached" if rep["cached"] else f"simulated in {rep['seconds']} s",
                    f"pen {rep['pen_mm']} mm (frame {rep['pen_frame']}), lag {rep['lag_mm']} mm, jerk {rep['jerk_mm']} "
                    f"mm (pinned {rep['jerk_pinned_mm']}), stretch {rep['stretch_pct']} %")
    return out
