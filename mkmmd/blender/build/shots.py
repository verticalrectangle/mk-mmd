"""shots: the cut. One camera per shot and output aspect, keyed per frame (no constraints), cut with timeline markers.

[[shot]] keys (times in clip seconds; targets as in mkmmd.blender.build.targets)
  name, from, to
  mount = "car"                  object the camera rides (prop or set name; default the world)
  at = [x, y, z] | target        camera position in the mount's frame (or world); a dict target ({path = "road:road",
                                 s = 640, offset = 7, z = 1.2} for a roadside camera, {prop = ...}) is resolved per frame
  look = target                  what the camera points at (default: straight ahead along the mount's -Y)
  lens = 35, roll = 0 (deg), lag = 0.0 (s, operator lag on the aim), shake = 0.0 (deg, handheld)
  keys = [{t, at, look, lens, shift}]   moves inside the shot (eased)
  frame = {subject = [targets], fill = 0.45, solve = "lens" | "distance"}   automatic framing per aspect: the subject's
                                 height fills `fill` of the frame (lens or dolly solved per aspect)
  shift = [x, y]                 lens shift as fractions of the LARGER image side (Blender's shift_x / shift_y: the
                                 picture moves the other way), constant or keyed in `keys[].shift`. A 1:1 output that
                                 must be an exact crop of a 9:16 master keeps the master's camera with lens and dof
                                 fstop x 1920 / 1080 and shift = [0, (420 - top) / 1080] for a crop `top` px from the
                                 master's top (mkmmd.core.shotstyle.crop_camera)
  [shot.aspect.<output>]         per-output overrides of at / look / lens / roll / frame / shift / style / reflection
  dof = {focus = target, fstop = 2.8, offset = 0}   offset: metres the focus plane sits nearer the camera than the target
                                 (a face's surface is 4-8 cm in front of its eye bones)
  style = "silhouette"           the flat look of a shot, composed at render time (`mk render`, `mk look`) by
                                 mkmmd.blender.styles; colors = {background, subject, accent}, hide / keep / accent =
                                 [object patterns], tint = [{object, prop, color, gain, glow}], knockout = {objects,
                                 color}, grow, samples (docs/design.md: Shots)
  style = "vector"               the flat-vector look: the project's [vector] table (tones, materials, lines, shadow,
                                 light) with the shot's own colors = {background, line, inner}, tones, hide / keep
  reflection = {object = "<glass>", strength, dim, roughness, hide, only, bend, world, tint}   her image in a window
                                 pane: a plane light probe and a mirror layer on the glass object, made at render time
  plate = true                   not in the cut (no `from` / `to` needed): a shot that only a [[transition]] or [[insert]]
                                 takes frames from (docs/design.md: Transitions and inserts); a shot in the cut that
                                 such an effect takes frames from before its `from` is keyed over those frames too
[[ring]] = {at, center, dur, width, hold, ease, edge}: a disc or band sweeping out from a point of the frame or the scene,
flipping a vector shot to its look's `opposite` palette inside (mkmmd.core.rings); the stage keeps them in
scene["mk_rings"] with the clip's fps and frame0 and warns about one live over a shot without a vector look.
The scene keeps the shot table in scene["mk_shots"] (JSON; per output aspect the normalised style and reflection, colours
resolved; `keyed`: the frames the cameras are keyed over; `plate`) so `mk look` and `mk render` bind the markers to each
aspect's cameras and switch the look per frame."""
import fnmatch
import json
import math
import zlib

import bpy
import numpy as np
from mathutils import Matrix, Quaternion, Vector

from ...core import freeze as FZ
from ...core import perform as PF
from ...core import rings as RG
from ...core import shotspec as SP
from ...core import shotstyle as SS
from ...core import transition as TR
from .. import keys as K
from .. import styles as ST
from . import BuildError, collection, targets

SENSOR = 36.0


def _mount(ctx, name):
    if not name:
        return None
    if name in ctx.props:
        return ctx.props[name].root
    if name in getattr(ctx, "sets", {}):
        return ctx.sets[name].root
    ob = bpy.data.objects.get(name)
    if ob is None:
        raise BuildError(f"shot mount {name!r}: no prop, set or object of that name")
    return ob


def _vfov_scale(size):
    """Half-height of the image plane in sensor units at lens 1 (AUTO sensor fit: 36 mm on the long side)."""
    w, h = size
    return (SENSOR / 2) * (h / max(w, h))


def _styles(ctx, spec, outs, name):
    """{aspect: {"silhouette" | "vector" | "reflection": normalised spec}} for the shot's render-time looks (the table
    that mkmmd.blender.styles reads; colours resolved against the project palette, a vector look with the project's
    [vector] table)."""
    styles, seen = {}, set()
    recs = None
    for out in outs:
        asp = out["name"]
        try:
            norm = SS.normalize(SP.merged(spec, (spec.get("aspect") or {}).get(asp)), ctx.palette, ctx.data.get("vector"))
        except SS.StyleError as e:
            raise BuildError(f"shot {name!r} ({asp}): {e}") from None
        if "reflection" in norm:
            glass = bpy.data.objects.get(norm["reflection"]["object"])
            if glass is None or glass.type != "MESH":
                raise BuildError(f"shot {name!r} ({asp}): reflection object {norm['reflection']['object']!r} is not a "
                                 f"mesh in the scene (list them with `mk q BLEND --list objects`)")
        for kind, look in norm.items():                     # a pattern that matches nothing is a typo until proven otherwise
            for key in ("hide", "keep", "accent", "only"):
                for pat in look.get(key, []):
                    if (key, pat) in seen:
                        continue
                    seen.add((key, pat))
                    recs = recs if recs is not None else ST.object_records(bpy.context.scene)
                    if not SS.select(recs, [pat]):
                        ctx.log(f"WARNING shot {name!r}: {kind} {key} pattern {pat!r} matches no object")
            for i, rule in enumerate(look.get("materials", []) if kind == "vector" else []):
                if ("materials", i) in seen:
                    continue
                seen.add(("materials", i))
                names = {SS.vector_name(m.name) for m in bpy.data.materials}
                if not any(fnmatch.fnmatchcase(n, p) for n in names for p in rule["match"]):
                    ctx.log(f"WARNING [vector] materials[{i}] (tone {rule['tone']!r}): match {rule['match']} names no "
                            f"material in the scene")
        if norm:
            styles[asp] = norm
    return styles


def run(ctx):
    shots = ctx.data.get("shot", [])
    if not shots:
        return {}
    try:
        plan = TR.plan(ctx.data, ctx.fps, ctx.frame0, ctx.palette)
    except TR.TransitionError as e:
        raise BuildError(str(e)) from None
    needs = TR.needs(plan)                       # {shot: [first, last]}: frames a transition or insert takes from a shot
    sc = bpy.context.scene
    coll = collection("Cameras")
    outs = ctx.project.get("outputs") or [{"name": "main", "size": [sc.render.resolution_x, sc.render.resolution_y]}]
    for spec in shots:
        try:
            SP.check(spec, [o["name"] for o in outs])
        except SP.ShotError as e:
            raise BuildError(str(e)) from None
    table, report = [], {}
    for m in list(sc.timeline_markers):
        sc.timeline_markers.remove(m)
    for spec in shots:
        name = spec["name"]
        need, plate = needs.get(name), bool(spec.get("plate"))
        if plate and need is None:
            ctx.log(f"WARNING shot {name!r}: plate = true but no [[transition]] or [[insert]] takes frames from it; skipped")
            continue
        if "from" in spec:
            f_from, f_to = int(round(ctx.frame(float(spec["from"])))), int(round(ctx.frame(float(spec["to"]))))
        else:
            f_from, f_to = need                                      # a plate shot lives where it is wanted
        lo, hi = (min(f_from, need[0]), max(f_to, need[1])) if need else (f_from, f_to)
        if lo < ctx.start:
            raise BuildError(f"shot {name!r}: a transition or insert takes frames from it as early as {lo}, before the "
                             f"scene's first frame {ctx.start} (lower [scene] start)")
        frames = np.arange(max(ctx.start, lo - 2), min(ctx.end, hi + 2) + 1)    # its own frames and those it lends
        own = frames[(frames >= f_from - 2) & (frames <= f_to + 2)]
        cams = {}
        for out in outs:
            asp = out["name"]
            sp = SP.merged(spec, (spec.get("aspect") or {}).get(asp))
            mount = _mount(ctx, sp.get("mount"))
            cd = bpy.data.cameras.new(f"{name}@{asp}")
            cd.sensor_fit, cd.sensor_width = "AUTO", SENSOR
            cd.clip_start, cd.clip_end = 0.02, 5000.0
            cam = bpy.data.objects.new(f"{name}@{asp}", cd)
            coll.objects.link(cam)
            cam.rotation_mode = "QUATERNION"
            keys = sorted(sp.get("keys", []), key=lambda k: k["t"])
            base_shift = (0.0, 0.0)
            if sp.get("shift") is not None:
                base_shift = SS.shift_pair(sp["shift"], f"shot {name!r} ({asp}) shift")
            for k in keys:
                if "shift" in k:
                    SS.shift_pair(k["shift"], f"shot {name!r} ({asp}) keys[].shift")
            # per frame: camera position (world), aim point (world), lens, lens shift, mount matrix
            pos, aim, lens, mounts, shifts = [], [], [], [], []
            for f in frames:
                sc.frame_set(int(f))
                t = ctx.time(f)
                at, look, ln = sp.get("at", (0, 0, 0)), sp.get("look"), float(sp.get("lens", 35.0))
                sh = base_shift
                M = mount.matrix_world.copy() if mount is not None else Matrix()

                def place(a, M=M):                       # a list is in the mount's frame, a dict target is in the world
                    return targets.point(ctx, a) if isinstance(a, dict) else M @ Vector(a)
                if keys:
                    k1 = next((k for k in keys if k["t"] >= t), keys[-1])
                    k0 = next((k for k in reversed(keys) if k["t"] <= t), keys[0])
                    u = 0.0 if k1 is k0 else float(PF.smooth((t - k0["t"]) / max(k1["t"] - k0["t"], 1e-6)))
                    p = place(k0.get("at", at)).lerp(place(k1.get("at", at)), u)
                    ln = k0.get("lens", ln) * (1 - u) + k1.get("lens", ln) * u
                    look = k0.get("look", look) if u < 0.5 else k1.get("look", look)
                    sh = tuple(np.asarray(k0.get("shift", base_shift), float) * (1 - u)
                               + np.asarray(k1.get("shift", base_shift), float) * u)
                else:
                    p = place(at)
                if look is None:
                    a = p + (M.to_3x3() @ Vector((0, -1, 0))) * 10.0
                else:
                    a = targets.point(ctx, look)
                pos.append(tuple(p))
                aim.append(tuple(a))
                lens.append(ln)
                shifts.append(sh)
                mounts.append(M)
            pos, aim, lens = np.array(pos), np.array(aim), np.array(lens)
            if float(sp.get("lag", 0.0)) > 0:
                # operator lag is felt in the mount's frame: a camera in a car lags the subject, not the road
                local = np.array([tuple(Mi.inverted() @ Vector(a)) for Mi, a in zip(mounts, aim)])
                local = PF.lowpass(local, float(sp["lag"]) * ctx.fps)
                aim = np.array([tuple(Mi @ Vector(a)) for Mi, a in zip(mounts, local)])
            fr = sp.get("frame")
            if fr:
                if "subject" not in fr:
                    raise BuildError(f"shot {name!r} ({asp}): `frame` needs a `subject` (a target or a list of targets)")
                subj = fr["subject"] if isinstance(fr["subject"], list) else [fr["subject"]]
                hs = []
                for f in own[:: max(1, len(own) // 12)]:
                    sc.frame_set(int(f))
                    P = np.array([tuple(targets.point(ctx, s)) for s in subj])
                    k = int(np.argmin(np.abs(frames - f)))
                    d = np.linalg.norm(P.mean(0) - pos[k])
                    ext = max(P[:, 2].max() - P[:, 2].min(), 0.25)        # at least a head's height
                    hs.append((ext + 0.15) / d)                            # angular height (with headroom)
                ang = float(np.median(hs))
                fill = float(fr.get("fill", 0.45))
                half = _vfov_scale(out["size"])
                if fr.get("solve", "lens") == "lens":
                    lens[:] = fill * 2 * half / ang
                else:                                                      # dolly along the aim line
                    want = fill * 2 * half / lens.mean()
                    scale = ang / want
                    pos = aim + (pos - aim) * scale
            shake = float(sp.get("shake", 0.0))
            rng = np.random.default_rng(zlib.crc32(f"{name}@{asp}".encode()))
            jit = PF.lowpass(rng.normal(0, 1, (len(frames), 3)), 6.0) * math.radians(shake) * 2.5 if shake else None
            roll = math.radians(float(sp.get("roll", 0.0)))
            quats = []
            for i in range(len(frames)):
                d = Vector(aim[i]) - Vector(pos[i])
                q = d.to_track_quat("-Z", "Y")
                q = q @ Quaternion((0, 0, 1), roll)
                if jit is not None:
                    q = q @ Quaternion((1, 0, 0), jit[i, 0]) @ Quaternion((0, 1, 0), jit[i, 1]) @ \
                        Quaternion((0, 0, 1), jit[i, 2] * 0.5)
                quats.append(tuple(q))
            K.key_vec(cam, "location", frames, pos, interp="LINEAR")
            K.key_vec(cam, "rotation_quaternion", frames, K.continuous(quats), interp="LINEAR")
            K.set_fcurve(cd, "lens", 0, frames, lens, interp="LINEAR")
            if any("shift" in k for k in keys):
                shifts = np.array(shifts)
                K.set_fcurve(cd, "shift_x", 0, frames, shifts[:, 0], interp="LINEAR")
                K.set_fcurve(cd, "shift_y", 0, frames, shifts[:, 1], interp="LINEAR")
            else:
                cd.shift_x, cd.shift_y = base_shift
            if sp.get("dof"):
                cd.dof.use_dof = True
                cd.dof.aperture_fstop = float(sp["dof"].get("fstop", 2.8))
                off = float(sp["dof"].get("offset", 0.0))
                dist = []
                for i, f in enumerate(frames):
                    sc.frame_set(int(f))
                    dist.append(max((targets.point(ctx, sp["dof"]["focus"]) - Vector(pos[i])).length - off, 0.05))
                K.set_fcurve(cd, "dof.focus_distance", 0, frames, dist, interp="LINEAR")
            cams[asp] = cam.name
        styles = _styles(ctx, spec, outs, name)
        entry = {"name": name, "from": f_from, "to": f_to, "cameras": cams, "keyed": [int(frames[0]), int(frames[-1])]}
        if plate:
            entry["plate"] = True                                    # not in the cut: no marker, rendered where wanted
        else:
            mk = sc.timeline_markers.new(name, frame=f_from)
            mk.camera = bpy.data.objects[cams[outs[0]["name"]]]
        if styles:
            entry["styles"] = styles
        table.append(entry)
        report[name] = {"frames": [f_from, f_to], "keyed": entry["keyed"], "cameras": cams}
        if plate:
            report[name]["plate"] = True
        if styles:
            report[name]["styles"] = {asp: sorted(s) for asp, s in styles.items()}
        ctx.log("shot", name, f_from, f_to, "(plate)" if plate else "")
    sc["mk_shots"] = json.dumps(table)
    first = min((e for e in table if not e.get("plate")) or table, key=lambda s: s["from"])
    sc.camera = bpy.data.objects[first["cameras"][outs[0]["name"]]]
    if plan["transitions"] or plan["inserts"]:
        report["_cut_effects"] = TR.summary(plan)
    rings = _rings(ctx, table, outs)
    if rings:
        report["_rings"] = rings
    try:
        freezes = FZ.normalize(ctx.data.get("freeze", []), ctx.fps, ctx.frame0)
    except FZ.FreezeError as e:
        raise BuildError(str(e)) from None
    sc["mk_freeze"] = json.dumps(freezes)                        # mk render / mk look hold the world there
    if freezes:
        report["_freezes"] = freezes
    return report


def _rings(ctx, table, outs):
    """[[ring]] normalised into scene["mk_rings"] with the clip's fps and frame0 (mkmmd.blender.styles draws them) and
    checked against the cut: a ring is drawn in vector shots only (a WARNING names any other shot it is live over), and
    their look needs an `opposite` palette for the inside of the ring."""
    try:
        rings = RG.normalize(ctx.data.get("ring", []), ctx.palette)
    except RG.RingError as e:
        raise BuildError(str(e)) from None
    bpy.context.scene["mk_rings"] = json.dumps({"fps": ctx.fps, "frame0": ctx.frame0, "rings": rings})
    for r in rings:
        f0, f1 = ctx.frame0 + r["at"] * ctx.fps, ctx.frame0 + r["until"] * ctx.fps
        for e in table:
            if e.get("plate") or e["to"] <= f0 or e["from"] >= f1:
                continue
            for out in outs:
                look = (e.get("styles") or {}).get(out["name"]) or {}
                if "vector" not in look:
                    ctx.log(f"WARNING ring {r['index']} at {r['at']} s: shot {e['name']!r} ({out['name']}) has no vector "
                            f"look, so the ring is not drawn there")
                elif look["vector"]["opposite"] is None:
                    raise BuildError(f"ring {r['index']} at {r['at']} s: shot {e['name']!r} ({out['name']}) needs a "
                                     f"[vector] opposite palette for the inside of the ring")
    return {"rings": len(rings)} if rings else None
