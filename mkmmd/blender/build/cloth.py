"""cloth: the cloth copies a cast member's rig declares (rig.json `cloth`, mkmmd.model.cloth) simulated with Blender's
cloth over the build's frames, the garment's faces following the simulated cloth (docs/design.md: Cloth).

[cloth.<cast>] switches it on for that member (every key optional):
  enabled = true   false: no cloth (the table can stay)
  pieces = [...]   the copies to simulate, by name (default every one the rig declares)
  quality, mass, tension, shear, bending, air, distance, hold, grip: the cloth, over the garment's own (rig.json): steps
                   a frame, kg a vertex, stiffness against stretching, shearing and bending, air damping, the gap it keeps
                   off the body (m), how much the free cloth is held where the armature carries it (0..1, each vertex
                   at least its copy's holds) and how hard what is held is pulled back there (Blender's pin stiffness)
  cache = true     reuse a simulation of the same copy, settings and motion (<cache>/cloth/<key>.pc2)

Per copy: <root>_<name>_cloth, the simulated cloth (hidden from render), rides the armature and plays its frames from
<blend>.cloth/<cast>.<name>.pc2 (a Mesh Cache modifier: the saved scene holds no physics), and the model's faces of the
copy's materials move to a mesh of their own, <root>_<name>, bound to that cloth on the first frame (Surface Deform) as
far as the copy is free there. The simulation: the copy skinned to the armature, pinned where its pins say and the rest
held by the larger of its holds and `hold`, its rest shape the skinned copy's on every frame (Dynamic Mesh), colliding
with the copy's skin (skinned too), baked from the build's first frame to its last. Fold shapes on a piece the cloth
simulates are left to it (the folds stage skips them)."""
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
from mathutils.bvhtree import BVHTree
from mathutils.kdtree import KDTree

from . import BuildError, collection

STAGE = 9                       # the set-up's version: a cached simulation from an older one is not reused
PIN = "mk_pin"
DEFAULTS = {"quality": 8, "mass": 0.3, "tension": 10.0, "shear": 5.0, "bending": 0.5, "air": 1.0, "distance": 0.003,
            "hold": 0.3, "grip": 1.0}


def simulated(ctx, name, m):
    """The names of the cloth copies this stage simulates for cast member `name`: none without its table (an empty one
    switches it on) or with enabled = false; else `pieces` (default every copy its rig declares)."""
    if name not in ctx.section("cloth"):
        return []
    spec = ctx.section("cloth", name) or {}
    if not spec.get("enabled", True):
        return []
    have = [c["name"] for c in (m.rig or {}).get("cloth") or []]
    want = list(spec.get("pieces") or have)
    unknown = [p for p in want if p not in have]
    if unknown:
        raise BuildError(f"cloth.{name}: no cloth copy {unknown} in its rig.json (it has {have or 'none'})")
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


def edges(quads):
    """(e, 2) the unique edges of quads (f, 4)."""
    Q = np.asarray(quads)
    return np.unique(np.sort(np.concatenate([Q[:, [0, 1]], Q[:, [1, 2]], Q[:, [2, 3]], Q[:, [3, 0]]]), 1), axis=0)


def checks(X, G, faces, pins, collides, C, tris, frames, window):
    """What the simulation did over the clip's frames `window` (frame numbers, inclusive): X (F, n, 3) the cloth, G its
    goal (the copy where the armature carries it) and C (F, m, 3) the body it collides with on `frames`, in one space.
    Deepest the free cloth that collides goes into that body (mm) and on which frame, the share of its vertex-frames
    deeper than 2 mm, how far the free cloth trails its goal (mm: most of it, p99, and the most), the largest change of
    a vertex's frame-to-frame motion of the free cloth and of the pinned (mm: the pinned move as her skin does, so the
    free cloth's beyond them is the cloth's own jump), and the most an edge with a free end stretches beyond its goal
    (%)."""
    k = np.flatnonzero((frames >= window[0]) & (frames <= window[1]))
    free = np.flatnonzero(pins < 0.5)
    touch = np.flatnonzero((pins < 0.5) & (np.asarray(collides) > 0))
    deep, deep_at, over = 0.0, int(frames[k[0]]), 0
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
                    deep, deep_at = d, int(frames[i])
    A = np.linalg.norm(X[k][2:] - 2.0 * X[k][1:-1] + X[k][:-2], axis=2) if len(k) > 2 else np.zeros((1, len(pins)))
    E = edges(faces)
    E = E[(pins[E[:, 0]] < 0.5) | (pins[E[:, 1]] < 0.5)]
    L0 = np.linalg.norm(G[k][:, E[:, 0]] - G[k][:, E[:, 1]], axis=2)
    L = np.linalg.norm(X[k][:, E[:, 0]] - X[k][:, E[:, 1]], axis=2)
    lag = np.linalg.norm(X[k][:, free] - G[k][:, free], axis=2)
    pinned = np.flatnonzero(pins >= 0.5)
    return {"pen_mm": round(deep * 1000.0, 2), "pen_frame": deep_at,
            "pen_over_2mm": round(over / max(len(k) * len(touch), 1), 5),
            "lag_mm": [round(float(np.quantile(lag, q)) * 1000.0, 1) for q in (0.5, 0.99, 1.0)] if len(free) else [],
            "jerk_mm": round(float(A[:, free].max()) * 1000.0, 2) if len(free) else 0.0,
            "jerk_pinned_mm": round(float(A[:, pinned].max()) * 1000.0, 2) if len(pinned) else 0.0,
            "stretch_pct": round(float((L / np.maximum(L0, 1e-9)).max() - 1.0) * 100.0, 1)}


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


def _material_name(mat):
    mm = getattr(mat, "mmd_material", None)
    return getattr(mm, "name_j", "") or mat.name


def separate(m, materials, name):
    """The faces of `materials` (PMX names) in the member's meshes moved to meshes of their own (named
    <root>_<name>), keeping their modifiers, weights and shape keys. Returns the new objects."""
    out = []
    for ob in m.meshes:
        slots = {i for i, s in enumerate(ob.material_slots) if s.material and _material_name(s.material) in materials}
        if not slots or not any(p.material_index in slots for p in ob.data.polygons):
            continue
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
        for f in bm.faces:
            if f.material_index in slots:
                f.select_set(True)
        bmesh.update_edit_mesh(ob.data)
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


def follow(ob, rest, pins):
    """A vertex group `mk_cloth` on `ob`: how much each of its vertices follows the cloth, 1 minus the pins of the copy
    (`rest` (n, 3), in the space of `ob`'s own vertices) near it, by inverse distance over its three nearest: what a band
    holds keeps its skin's motion exactly, the free knit goes with the cloth."""
    kd = KDTree(len(rest))
    for i, p in enumerate(rest):
        kd.insert(p, i)
    kd.balance()
    vg = ob.vertex_groups.new(name="mk_cloth")
    for v in ob.data.vertices:
        near = kd.find_n(v.co, 3)
        w = np.array([1.0 / max(d, 1e-6) for _, _, d in near])
        held = float(np.dot(w, [pins[i] for _, i, _ in near]) / w.sum())
        vg.add([v.index], 1.0 - held, "REPLACE")
    return vg.name


def bind(ob, target, frame, group):
    """A Surface Deform on `ob` right after its armature, following `target` as much as the vertex group `group` says,
    bound on `frame`."""
    md = ob.modifiers.new("mk_cloth", "SURFACE_DEFORM")
    md.target = target
    md.vertex_group = group
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
                ctx.log("WARNING", f"cloth.{name}: the model has no cloth copy (rig.json `cloth`), so nothing is simulated")
            continue
        spec = ctx.section("cloth", name) or {}
        entries = {c["name"]: c for c in m.rig["cloth"]}
        out[name] = {}
        for piece in want:
            e = entries[piece]
            A, sha = load(m, e)
            missing = [b for b in e["bones"] if b not in m.arm.data.bones]
            if missing:
                raise BuildError(f"cloth.{name}: the cloth copy {piece!r} is weighted to bones the armature lacks: {missing}")
            settings = dict(DEFAULTS, **(e.get("settings") or {}))
            settings.update({k: spec[k] for k in DEFAULTS if k in spec})
            coll = collection(f"{m.root.name}_{piece}_cloth")
            sim = _mesh(f"{m.root.name}_{piece}_cloth", A["sim"], A["sim_faces"], coll, m.arm, e["bones"],
                        A["sim_weights"])
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
                                (ctx.frame0, ctx.end))}
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
            mc = sim.modifiers.new("frames", "MESH_CACHE")
            mc.cache_format, mc.filepath, mc.frame_start = "PC2", pc2, float(ctx.start)
            tri = sim.modifiers.new("faces", "TRIANGULATE")       # what the knit binds to: triangles, always valid,
            tri.quad_method = "FIXED"                             # split the same way on every frame
            sim.hide_render = True
            sim.display_type = "WIRE"
            parts = separate(m, set(e["materials"]), piece)
            if not parts:
                raise BuildError(f"cloth.{name}: the model has no faces of {e['materials']} for {piece!r} to follow")
            for ob in parts:
                bind(ob, sim, ctx.start, follow(ob, A["sim"], A["pins"]))
            sc.frame_set(ctx.frame0)
            rep.update(verts=len(A["sim"]), faces=len(A["sim_faces"]), collide_faces=len(A["collide_faces"]),
                       frames=[ctx.start, ctx.end], settings=settings, key=key, pc2=pc2,
                       follows={ob.name: len(ob.data.vertices) for ob in parts})
            out[name][piece] = rep
            ctx.log("cloth", name, piece, "cached" if rep["cached"] else f"simulated in {rep['seconds']} s",
                    f"pen {rep['pen_mm']} mm (frame {rep['pen_frame']}), lag {rep['lag_mm']} mm, jerk {rep['jerk_mm']} "
                    f"mm (pinned {rep['jerk_pinned_mm']}), stretch {rep['stretch_pct']} %")
    return out
