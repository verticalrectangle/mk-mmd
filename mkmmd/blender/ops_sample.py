"""sample: posed data over frames into one .npz, for checks and solvers (they never touch bpy).

args:
  frames      [int]                          Blender frames (sorted, unique)
  out         path of the .npz to write
  bones       {armature or "": [bone, ...]}  Blender, semantic or PMX names; "" = the scene's only MMD armature
  families    {armature or "": [family]}     add every bone whose name classifies into these families
  objects     [object name]
  exprs       [expression]                   `mk q` expressions, numeric results only
  camera      bool                           the active camera per frame (markers respected) + its intrinsics
  colliders   [spec]                         scene collision shapes to resolve (docs/design.md: Colliders)
  meshes      [{objects, prop, exclude, frame}]   evaluated geometry, world space, once per request at its own frame
                                             (frame null: as the scene stands): `objects` by name plus what a render shows
                                             under the prop root(s) `prop` (a name or a list; no colliders, no hidden
                                             objects, no characters), minus `exclude` (names or fnmatch patterns)
npz keys: frames; per armature index a: bone_world_a (F,n,4,4) posed world matrices, bone_rest_a (n,4,4) rest
matrices (armature space), arm_world_a (F,4,4); obj_world (F,m,4,4); expr_i (F,...); cam_world (F,4,4), cam_lens,
cam_sensor (F,2), cam_shift (F,2), cam_clip (F,2); per mesh request k: mesh_v_k (n,3) vertices, mesh_t_k (m,3) triangle
corners (indices into mesh_v_k), mesh_o_k (m,) index of the owning object in the reply's meshes[k].objects. The JSON
reply names everything."""
import json
from contextlib import contextmanager
from fnmatch import fnmatchcase

import bpy
import numpy as np

from ..core import bonemap
from ..core import families as FAM
from . import scene as S
from .ops_core import _namespace
from .runtime import CTX, op


def _mat(m):
    return np.array([list(r) for r in m], float)


def _arm(name):
    return S.find_armature(name or None)


def resolve_collider(spec, default_arm):
    """One collider spec -> list of items {kind, source, tag, geometry...} (geometry in the source frame)."""
    t = spec["type"]
    tag = spec.get("tag") or spec.get("object") or spec.get("bone") or t
    rnd = float(spec.get("rnd", 0.0))
    if t == "floor":
        return [{"kind": "floor", "source": ["world", "", ""], "tag": "floor", "z": float(spec.get("z", 0.0))}]
    if t == "prop":                                      # every collider of a prop's card (stored on its root at save)
        root = bpy.data.objects.get(spec.get("prop", ""))
        if root is None or "mk_colliders" not in root.keys():
            raise ValueError(f"collider spec {spec!r}: no prop {spec.get('prop')!r} with stored colliders (rebuild)")
        return [it for s in json.loads(root["mk_colliders"]) for it in resolve_collider(s, default_arm)]
    if "object" in spec:
        ob = bpy.data.objects[spec["object"]]
        src = ["object", "", ob.name]
        sc = np.array(ob.matrix_world.to_scale())
        bb = np.array([list(v) for v in ob.bound_box]) * sc
        lo, hi = bb.min(0), bb.max(0)
        ctr, half = (lo + hi) / 2, (hi - lo) / 2
        if t == "box":
            M = np.eye(4)
            M[:3, 3] = ctr
            return [{"kind": "box", "source": src, "tag": tag, "M": M.tolist(), "half": half.tolist(), "rnd": rnd}]
        if t == "cylinder":
            M = np.eye(4)
            M[:3, 3] = spec.get("center", [0.0, 0.0, 0.0])
            R = float(spec.get("R", max(half[0], half[1])))
            hh = float(spec.get("half_h", half[2]))
            return [{"kind": "cylinder", "source": src, "tag": tag, "M": M.tolist(), "R": R, "hh": hh, "rnd": rnd}]
        if t == "capsule":
            return [{"kind": "capsule", "source": src, "tag": tag, "a": spec["a"], "b": spec["b"],
                     "R": float(spec["R"])}]
        if t == "sphere":
            return [{"kind": "sphere", "source": src, "tag": tag, "c": spec.get("c", ctr.tolist()),
                     "R": float(spec.get("R", half.max()))}]
        if t == "ring":                                  # a torus about the object's local Z: capsules round it
            n = int(spec.get("segments", 16))
            k = float(sc[0])
            R, tube = float(spec["radius"]) * k, float(spec["tube"]) * k
            a = 2 * np.pi * np.arange(n) / n
            pts = np.stack([R * np.cos(a), R * np.sin(a), np.zeros(n)], 1).tolist()
            return [{"kind": "capsule", "source": src, "tag": tag, "a": pts[i], "b": pts[(i + 1) % n], "R": tube}
                    for i in range(n)]
        raise ValueError(f"collider type {t!r} on an object: box cylinder capsule sphere ring")
    if t == "cylinder" and "center" in spec:
        M = np.eye(4)
        M[:3, 3] = spec["center"]
        return [{"kind": "cylinder", "source": ["world", "", ""], "tag": tag, "M": M.tolist(), "R": float(spec["R"]),
                 "hh": float(spec["half_h"]), "rnd": rnd}]
    if t in ("capsule", "fingers"):
        arm = _arm(spec.get("armature") or default_arm)
        mw = arm.matrix_world
        pairs = []
        if t == "capsule":
            pairs.append((S.resolve_bone(arm, spec["bone"]), S.resolve_bone(arm, spec["to"]), float(spec["R"])))
        else:                                            # every finger segment, and the palm: wrist -> finger bases
            radius = {"thumb": 0.010, "index": 0.009, "middle": 0.009, "ring": 0.0085, "little": 0.008, "palm": 0.016}
            radius.update(spec.get("radius", {}))
            smap = S.semantic_map(arm)
            for side in spec.get("sides", ["L", "R"]):
                for fing, sems in bonemap.FINGERS.items():
                    names = [smap[f"{s}.{side}"] for s in sems if f"{s}.{side}" in smap]
                    pairs += [(a, b, radius[fing]) for a, b in zip(names[:-1], names[1:])]
                wrist = smap.get(f"wrist.{side}")
                for base in ("index1", "middle1", "little1"):
                    if wrist and f"{base}.{side}" in smap:
                        pairs.append((wrist, smap[f"{base}.{side}"], radius["palm"]))
        out = []
        for b0, b1, R in pairs:
            rest = mw @ arm.data.bones[b0].matrix_local
            e = rest.inverted() @ (mw @ arm.data.bones[b1].head_local)
            out.append({"kind": "capsule", "source": ["bone", arm.name, b0], "tag": spec.get("tag", "hand"),
                        "a": [0.0, 0.0, 0.0], "b": list(e), "R": R})
        return out
    raise ValueError(f"collider spec {spec!r}: give an object, a world center, bone+to, fingers or floor")


# ---------------------------------------------------------------- evaluated geometry
GEOMETRY = {"MESH", "CURVE", "SURFACE", "FONT", "META"}      # object types whose evaluated geometry converts to a mesh


def _character_root(o):
    """An MMD model's root empty (mmd_tools stores its `mmd_type` enum as a number, ROOT = 2): what hangs below it is a
    character, not part of a prop."""
    return o.type == "EMPTY" and o.get("mmd_type") == 2


def _rendered(o):
    """Does a render show this object: not a collider shape, not hidden from render (itself or all its collections), and
    visible to the camera."""
    return (not o.get("mk_collider") and not o.hide_render and o.visible_camera
            and not (o.users_collection and all(c.hide_render for c in o.users_collection)))


def _visible_geometry(root):
    """The geometry objects below `root` that a render shows (`_rendered`), characters excluded (their subtrees are
    skipped); in name order."""
    out = []
    for c in sorted(root.children, key=lambda o: o.name):
        if _character_root(c):
            continue
        if c.type in GEOMETRY and _rendered(c):
            out.append(c)
        out += _visible_geometry(c)
    return out


def _find_object(name, what):
    ob = bpy.data.objects.get(name)
    if ob is None:
        near = [o.name for o in bpy.data.objects if name.lower() in o.name.lower()][:8]
        raise ValueError(f"mesh {what} {name!r}: no such object" + (f" (similar: {near})" if near else ""))
    return ob


def _mesh_objects(spec):
    """The objects of one mesh request, in order and without repeats."""
    obs = []
    props = spec.get("prop") or []
    for prop in [props] if isinstance(props, str) else props:
        obs += _visible_geometry(_find_object(prop, "prop"))
    for name in spec.get("objects") or []:
        ob = _find_object(name, "object")
        if ob.type not in GEOMETRY:
            raise ValueError(f"mesh object {name!r} is a {ob.type}, not geometry (the root of a prop goes in `prop`)")
        obs.append(ob)
    skip = list(spec.get("exclude") or [])
    out = {}
    for ob in obs:
        if not any(fnmatchcase(ob.name, pat) for pat in skip):
            out.setdefault(ob.name, ob)
    return list(out.values())


@contextmanager
def _render_settings(obs):
    """Modifiers as a render evaluates them: only the ones shown in renders, Subdivision Surface and Multires at their
    render levels. The viewport settings come back afterwards (`mk serve` keeps the scene between jobs)."""
    saved = []
    try:
        for ob in obs:
            for md in ob.modifiers:
                saved.append((md, "show_viewport", md.show_viewport))
                if md.type in ("SUBSURF", "MULTIRES"):
                    saved.append((md, "levels", md.levels))
                    md.levels = md.render_levels
                md.show_viewport = md.show_render
        yield
    finally:
        for md, attr, value in reversed(saved):
            setattr(md, attr, value)


def _world_mesh(ob, dg):
    """World-space vertices (n,3) and triangle corners (m,3) of the evaluated object: modifiers, shape keys, armature
    poses and geometry nodes applied, as a render at this frame sees it."""
    oe = ob.evaluated_get(dg)
    me = oe.to_mesh()
    if me is None:
        return np.zeros((0, 3)), np.zeros((0, 3), np.int32)
    try:
        co = np.empty(len(me.vertices) * 3, np.float32)
        me.vertices.foreach_get("co", co)
        me.calc_loop_triangles()
        tri = np.empty(len(me.loop_triangles) * 3, np.int32)
        me.loop_triangles.foreach_get("vertices", tri)
    finally:
        oe.to_mesh_clear()
    M = _mat(oe.matrix_world)
    tri = tri.reshape(-1, 3)
    if np.linalg.det(M[:3, :3]) < 0:                      # mirrored by a negative scale: keep the normals outward
        tri = tri[:, ::-1]
    return co.reshape(-1, 3).astype(float) @ M[:3, :3].T + M[:3, 3], tri


def _sample_meshes(specs, out):
    """Fill `out` with mesh_v_k / mesh_t_k / mesh_o_k for every request; returns their reply entries (objects, frame,
    the objects tagged `mk_form_exempt`, and the world matrix of every prop root asked for; `error` instead when a name
    does not exist: that request fails, the others do not). The scene's frame is put back afterwards."""
    sc = bpy.context.scene
    keep = sc.frame_current
    meta = []
    try:
        for k, spec in enumerate(specs):
            if spec.get("frame") is not None and int(spec["frame"]) != sc.frame_current:
                sc.frame_set(int(spec["frame"]))
            try:
                obs = _mesh_objects(spec)
            except ValueError as e:
                obs, err = [], str(e)
            else:
                err = None
            verts, tris, owner, base = [], [], [], 0
            with _render_settings(obs):
                bpy.context.view_layer.update()
                dg = bpy.context.evaluated_depsgraph_get()
                for j, ob in enumerate(obs):
                    V, T = _world_mesh(ob, dg)
                    verts.append(V)
                    tris.append(T + base)
                    owner.append(np.full(len(T), j, np.int32))
                    base += len(V)
            out[f"mesh_v_{k}"] = np.concatenate(verts) if verts else np.zeros((0, 3))
            out[f"mesh_t_{k}"] = np.concatenate(tris).astype(np.int32) if tris else np.zeros((0, 3), np.int32)
            out[f"mesh_o_{k}"] = np.concatenate(owner) if owner else np.zeros(0, np.int32)
            props = spec.get("prop") or []
            roots = {p: _mat(bpy.data.objects[p].matrix_world).tolist()
                     for p in ([props] if isinstance(props, str) else props) if err is None}
            meta.append({"objects": [o.name for o in obs], "frame": sc.frame_current, "roots": roots,
                         "exempt": [o.name for o in obs if o.get("mk_form_exempt")], **({"error": err} if err else {})})
    finally:
        if sc.frame_current != keep:
            sc.frame_set(keep)
    return meta


@op("sample")
def sample(args):
    frames = [int(f) for f in args["frames"]]
    sc = bpy.context.scene
    default_arm = args.get("armature")
    resolved_specs = [resolve_collider(spec, default_arm) for spec in args.get("colliders", [])]
    items = [it for group in resolved_specs for it in group]
    want_bones = {k: list(v) for k, v in (args.get("bones") or {}).items()}
    fam_sel = {}
    for arm_name, fams in (args.get("families") or {}).items():
        arm = _arm(arm_name)
        fam_sel[arm_name] = {f: [] for f in fams}
        for pb in arm.pose.bones:
            f = FAM.classify(pb.name, S.name_j(pb))
            if f in fams:
                fam_sel[arm_name][f].append(pb.name)
                want_bones.setdefault(arm_name, []).append(pb.name)
    objects = list(dict.fromkeys(args.get("objects", [])))
    for it in items:
        kind, owner, name = it["source"]
        if kind == "bone":
            want_bones.setdefault(owner, []).append(name)
        elif kind == "object" and name not in objects:
            objects.append(name)
    arms, meta_b = [], {}
    for k, (arm_name, names) in enumerate(want_bones.items()):
        arm = _arm(arm_name)
        resolved = {}
        for n in names:                                  # "?name": optional, skipped when the model lacks it
            try:
                resolved[n] = S.resolve_bone(arm, n[1:] if n.startswith("?") else n)
            except KeyError:
                if not n.startswith("?"):
                    raise
        res = list(dict.fromkeys(resolved.values()))
        arms.append((arm, res))
        B = arm.data.bones
        meta_b[arm_name] = {"index": k, "armature": arm.name, "names": res, "resolved": resolved,
                            "families": fam_sel.get(arm_name, {}),
                            "jp": [S.name_j(arm.pose.bones[n]) for n in res],
                            "parent": [B[n].parent.name if B[n].parent else None for n in res],
                            "length": [B[n].length for n in res],
                            "children": [[c.name for c in B[n].children] for n in res],
                            "scale": list(arm.matrix_world.to_scale())}
    exprs = args.get("exprs", [])
    ns = _namespace(default_arm) if exprs else None
    codes = [compile(e, "<mk sample>", "eval") for e in exprs]
    F = len(frames)
    out = {"frames": np.array(frames)}
    meshes = _sample_meshes(args.get("meshes") or [], out)
    for k, (arm, names) in enumerate(arms):
        out[f"bone_world_{k}"] = np.zeros((F, len(names), 4, 4))
        out[f"bone_rest_{k}"] = np.array([_mat(arm.data.bones[n].matrix_local) for n in names]).reshape(-1, 4, 4)
        out[f"arm_world_{k}"] = np.zeros((F, 4, 4))
    out["obj_world"] = np.zeros((F, len(objects), 4, 4))
    expr_vals = [[] for _ in exprs]
    want_cam = bool(args.get("camera"))
    cam_names = []
    if want_cam:
        for key in ("cam_world", "cam_lens", "cam_sensor", "cam_shift", "cam_clip"):
            out[key] = np.zeros({"cam_world": (F, 4, 4), "cam_lens": (F,), "cam_sensor": (F, 2), "cam_shift": (F, 2),
                                 "cam_clip": (F, 2)}[key])
    proj = CTX["project"] or {}
    for i, f in enumerate(frames):
        sc.frame_set(f)
        dg = bpy.context.evaluated_depsgraph_get()
        for k, (arm, names) in enumerate(arms):
            ae = arm.evaluated_get(dg)
            mw = ae.matrix_world
            out[f"arm_world_{k}"][i] = _mat(mw)
            for j, n in enumerate(names):
                out[f"bone_world_{k}"][i, j] = _mat(mw @ ae.pose.bones[n].matrix)
        for j, n in enumerate(objects):
            out["obj_world"][i, j] = _mat(bpy.data.objects[n].evaluated_get(dg).matrix_world)
        if codes:
            ns["frame"] = f
            ns["t"] = (f - proj["frame0"]) / proj["fps"] if proj.get("fps") else None
            for e, code in enumerate(codes):
                v = eval(code, ns)  # noqa: S307 - the caller's own expression
                expr_vals[e].append(np.array(v, float) if not isinstance(v, (int, float, bool)) else float(v))
        if want_cam:
            cam = sc.camera
            cam_names.append(cam.name if cam else None)
            if cam is not None:
                ce = cam.evaluated_get(dg)
                d = ce.data
                out["cam_world"][i] = _mat(ce.matrix_world)
                out["cam_lens"][i] = d.lens if d.type == "PERSP" else 0.0
                out["cam_sensor"][i] = (d.sensor_width, d.sensor_height)
                out["cam_shift"][i] = (d.shift_x, d.shift_y)
                out["cam_clip"][i] = (d.clip_start, d.clip_end)
    for e, vals in enumerate(expr_vals):
        out[f"expr_{e}"] = np.array(vals, float)
    np.savez(args["out"], **out)
    uses = {}
    for n in args.get("props") or []:
        root = bpy.data.objects.get(n)
        if root is None or "mk_use" not in root.keys():
            raise ValueError(f"prop {n!r}: no prop root with stored use points (rebuild the scene)")
        uses[n] = json.loads(root["mk_use"])
    reply = {"out": args["out"], "frames": len(frames), "bones": meta_b, "objects": objects, "exprs": exprs,
             "colliders": resolved_specs, "meshes": meshes, "props": uses}
    if want_cam:
        cams = {n for n in cam_names if n}
        reply["camera"] = {"names": cam_names, "sensor_fit": {n: bpy.data.objects[n].data.sensor_fit for n in cams},
                           "resolution": [sc.render.resolution_x, sc.render.resolution_y],
                           "pixel_aspect": [sc.render.pixel_aspect_x, sc.render.pixel_aspect_y]}
    return reply


@op("visibility")
def visibility(args):
    """Ray tests from the active camera. args: frames, points (exprs giving a point or a list of points per frame),
    ignore (object names whose hits do not count: the subject's own meshes). Per frame: the share of points whose ray
    is blocked and by what; and whether the camera sits inside a closed mesh (rays in 6 directions all hit back
    faces)."""
    sc = bpy.context.scene
    ns = _namespace(args.get("armature"))
    codes = [compile(e, "<mk visibility>", "eval") for e in args.get("points", [])]
    ignore = set(args.get("ignore", []))
    for arm_name in args.get("ignore_models", []):
        arm = _arm(arm_name)
        ignore |= {o.name for o in S.model_meshes(arm)}
    from mathutils import Vector
    dirs = [Vector(d) for d in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))]
    rows = []
    for f in args["frames"]:
        sc.frame_set(int(f))
        dg = bpy.context.evaluated_depsgraph_get()
        cam = sc.camera
        c = cam.evaluated_get(dg).matrix_world.translation.copy()
        ns["frame"] = f
        pts = []
        for code in codes:
            v = eval(code, ns)  # noqa: S307
            arr = np.array(v, float).reshape(-1, 3)
            pts += [Vector(p) for p in arr]
        blocked, by = 0, {}
        for p in pts:
            d = p - c
            dist = d.length
            if dist < 1e-6:
                continue
            origin, left = c, dist
            while True:
                hit, loc, nrm, _i, ob, _m = sc.ray_cast(dg, origin, d.normalized(), distance=left)
                if not hit:
                    break
                name = ob.original.name if hasattr(ob, "original") else ob.name
                if name in ignore or ob.hide_render or not ob.visible_camera:
                    step = (loc - origin).length + 1e-4
                    origin, left = loc + d.normalized() * 1e-4, left - step
                    if left <= 0:
                        break
                    continue
                blocked += 1
                by[name] = by.get(name, 0) + 1
                break
        inside_by = None                                   # inside a closed mesh a render shows (not a glow volume,
        hits = []                                          # a hidden collider or an ignored object)
        for dvec in dirs:
            origin, first = c, (False, None, False)
            for _ in range(64):
                hit, loc, nrm, _i, ob, _m = sc.ray_cast(dg, origin, dvec)
                if not hit:
                    break
                name = ob.original.name if hasattr(ob, "original") else ob.name
                if name in ignore or ob.hide_render or not ob.visible_camera:
                    origin = loc + dvec * 1e-4
                    continue
                first = (True, name, nrm.dot(dvec) > 0)
                break
            hits.append(first)
        if all(h[0] and h[2] for h in hits) and len({h[1] for h in hits}) == 1:
            inside_by = hits[0][1]
        rows.append({"frame": f, "points": len(pts), "blocked": blocked, "by": by, "camera": cam.name,
                     "inside": inside_by})
    return rows
