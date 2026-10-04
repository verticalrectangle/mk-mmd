"""shots: the cut. One camera per shot and output aspect, keyed per frame (no constraints), cut with timeline markers.

[[shot]] keys (times in clip seconds; targets as in mkmmd.blender.build.targets)
  name, from, to
  mount = "car"                  object the camera rides (prop or set name; default the world)
  at = [x, y, z] | target        camera position in the mount's frame (or world); a dict target ({path = "road:road",
                                 s = 640, offset = 7, z = 1.2} for a roadside camera, {prop = ...}) is resolved per frame
  look = target                  what the camera points at (default: straight ahead along the mount's -Y)
  lens = 35, roll = 0 (deg), lag = 0.0 (s, operator lag on the aim), shake = 0.0 (deg, handheld)
  keys = [{t, at, look, lens}]   moves inside the shot (eased)
  frame = {subject = [targets], fill = 0.45, solve = "lens" | "distance"}   automatic framing per aspect: the subject's
                                 height fills `fill` of the frame (lens or dolly solved per aspect)
  [shot.aspect.<output>]         per-output overrides of at / look / lens / roll / frame
  dof = {focus = target, fstop = 2.8}
The scene keeps the shot table in scene["mk_shots"] (JSON) so `mk look` and `mk render` bind the markers to each
aspect's cameras."""
import json
import math
import zlib

import bpy
import numpy as np
from mathutils import Matrix, Quaternion, Vector

from ...core import perform as PF
from .. import keys as K
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


def _merge(base, over):
    out = dict(base)
    out.update(over or {})
    return out


def _vfov_scale(size):
    """Half-height of the image plane in sensor units at lens 1 (AUTO sensor fit: 36 mm on the long side)."""
    w, h = size
    return (SENSOR / 2) * (h / max(w, h))


def run(ctx):
    shots = ctx.data.get("shot", [])
    if not shots:
        return {}
    sc = bpy.context.scene
    coll = collection("Cameras")
    outs = ctx.project.get("outputs") or [{"name": "main", "size": [sc.render.resolution_x, sc.render.resolution_y]}]
    table, report = [], {}
    for m in list(sc.timeline_markers):
        sc.timeline_markers.remove(m)
    for spec in shots:
        name = spec["name"]
        f_from, f_to = int(round(ctx.frame(float(spec["from"])))), int(round(ctx.frame(float(spec["to"]))))
        frames = np.arange(max(ctx.start, f_from - 2), min(ctx.end, f_to + 2) + 1)
        cams = {}
        for out in outs:
            asp = out["name"]
            sp = _merge(spec, (spec.get("aspect") or {}).get(asp))
            mount = _mount(ctx, sp.get("mount"))
            cd = bpy.data.cameras.new(f"{name}@{asp}")
            cd.sensor_fit, cd.sensor_width = "AUTO", SENSOR
            cd.clip_start, cd.clip_end = 0.02, 5000.0
            cam = bpy.data.objects.new(f"{name}@{asp}", cd)
            coll.objects.link(cam)
            cam.rotation_mode = "QUATERNION"
            keys = sorted(sp.get("keys", []), key=lambda k: k["t"])
            # per frame: camera position (world), aim point (world), lens, mount matrix
            pos, aim, lens, mounts = [], [], [], []
            for f in frames:
                sc.frame_set(int(f))
                t = ctx.time(f)
                at, look, ln = sp.get("at", (0, 0, 0)), sp.get("look"), float(sp.get("lens", 35.0))
                if keys:
                    k1 = next((k for k in keys if k["t"] >= t), keys[-1])
                    k0 = next((k for k in reversed(keys) if k["t"] <= t), keys[0])
                    u = 0.0 if k1 is k0 else float(PF.smooth((t - k0["t"]) / max(k1["t"] - k0["t"], 1e-6)))
                    at = tuple(Vector(k0.get("at", at)).lerp(Vector(k1.get("at", at)), u))
                    ln = k0.get("lens", ln) * (1 - u) + k1.get("lens", ln) * u
                    look = k0.get("look", look) if u < 0.5 else k1.get("look", look)
                M = mount.matrix_world.copy() if mount is not None else Matrix()
                p = targets.point(ctx, at) if isinstance(at, dict) else M @ Vector(at)
                if look is None:
                    a = p + (M.to_3x3() @ Vector((0, -1, 0))) * 10.0
                else:
                    a = targets.point(ctx, look)
                pos.append(tuple(p))
                aim.append(tuple(a))
                lens.append(ln)
                mounts.append(M)
            pos, aim, lens = np.array(pos), np.array(aim), np.array(lens)
            if float(sp.get("lag", 0.0)) > 0:
                # operator lag is felt in the mount's frame: a camera in a car lags the subject, not the road
                local = np.array([tuple(Mi.inverted() @ Vector(a)) for Mi, a in zip(mounts, aim)])
                local = PF.lowpass(local, float(sp["lag"]) * ctx.fps)
                aim = np.array([tuple(Mi @ Vector(a)) for Mi, a in zip(mounts, local)])
            fr = sp.get("frame")
            if fr:
                subj = fr["subject"] if isinstance(fr["subject"], list) else [fr["subject"]]
                hs = []
                for f in frames[:: max(1, len(frames) // 12)]:
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
            if sp.get("dof"):
                cd.dof.use_dof = True
                cd.dof.aperture_fstop = float(sp["dof"].get("fstop", 2.8))
                dist = []
                for i, f in enumerate(frames):
                    sc.frame_set(int(f))
                    dist.append((targets.point(ctx, sp["dof"]["focus"]) - Vector(pos[i])).length)
                K.set_fcurve(cd, "dof.focus_distance", 0, frames, dist, interp="LINEAR")
            cams[asp] = cam.name
        mk = sc.timeline_markers.new(name, frame=f_from)
        mk.camera = bpy.data.objects[cams[outs[0]["name"]]]
        table.append({"name": name, "from": f_from, "to": f_to, "cameras": cams})
        report[name] = {"frames": [f_from, f_to], "cameras": cams}
        ctx.log("shot", name, f_from, f_to)
    sc["mk_shots"] = json.dumps(table)
    first = min(table, key=lambda s: s["from"])
    sc.camera = bpy.data.objects[first["cameras"][outs[0]["name"]]]
    return report
