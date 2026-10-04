"""vehicles: props that drive along a set's path (docs/design.md: Vehicles).

[[vehicle]] keys
  prop = "car"                  the prop to drive (its card may list wheels and a steering wheel, see below)
  path = "highway:road"         <set>:<path>
  lane = "R1" | 1.8             a lane name from the path's card, or metres left of the centerline. A lane whose card `dir`
                                is -1 (an oncoming lane) is driven against the path: the car faces the other way
  dir = 1 | -1                  the direction along the path when `lane` is a number (default 1: with it)
  speed = 24.0 | [[t, v], ...]  m/s (the car's own speed, positive), constant or keyed on clip seconds (linear between keys)
  at = 0.0                      arc length (m) where the vehicle is at clip time 0
  meet = {vehicle, t, ahead = 0.0}   instead of `at`: alongside that vehicle (an earlier [[vehicle]]) at clip time t, plus
                                `ahead` metres further along the path (a car to overtake, an oncoming car passing a cut)
  height = 0.0                  extra height above the path (m)
  leave = false                 true: the vehicle may run off the path's ends (traffic coming and going): there it waits
                                at the end, hidden from the render with everything it carries (its headlight beams too)
  roll = 1.2, pitch = 0.8       body roll / pitch (deg per g of lateral / longitudinal acceleration: the nose rises under
                                acceleration and dives under braking, the body leans out of a turn)
  wheelbase = 2.6, steer_ratio = 14.0
Card keys used: "wheels": [{object, radius, axis = [1, 0, 0] (local spin axis pointing to the car's left: the wheel
rolls forward)}], "steering": {object, axis =
[0, 0, 1] (local), ratio}. The steering object turns by atan(wheelbase * curvature) * ratio; the hands gripping it
follow when their IK targets ride on it (pose stage, grip on a ring with an object)."""
import math

import bpy
import numpy as np
from mathutils import Quaternion, Vector

from ...core.path import speed_profile
from .. import keys as K
from . import BuildError

G = 9.81


def lane_offset(card_path, lane):
    if isinstance(lane, (int, float)):
        return float(lane)
    for ln in card_path.get("lanes", []):
        if ln["name"] == lane:
            return float(ln["offset"])
    raise BuildError(f"lane {lane!r} not in path lanes {[ln['name'] for ln in card_path.get('lanes', [])]}")


def lane_dir(card_path, spec):
    """+1 when the vehicle drives with the path, -1 against it: the card's `dir` of a named lane, else the spec's `dir`."""
    lane = spec.get("lane", 0.0)
    if not isinstance(lane, (int, float)):
        for ln in card_path.get("lanes", []):
            if ln["name"] == lane:
                return -1 if float(ln.get("dir", 1)) < 0 else 1
    d = float(spec.get("dir", 1))
    if d not in (1.0, -1.0):
        raise BuildError(f"vehicle {spec.get('prop')!r}: dir = 1 or -1, not {d}")
    return int(d)


def _key_hidden(root, frames, hidden):
    """Key `hide_render` (and `hide_viewport`) of everything under `root` (it, its children, their children: a car's body,
    lamps, wheels and beams) on the frames where `hidden` changes (constant keys: the switch happens on the frame)."""
    stack, obs = [root], []
    while stack:
        o = stack.pop()
        obs.append(o)
        stack.extend(o.children)
    change = [0] + [i for i in range(1, len(frames)) if hidden[i] != hidden[i - 1]]
    for o in obs:
        for i in change:
            for attr in ("hide_render", "hide_viewport"):
                setattr(o, attr, bool(hidden[i]))
                o.keyframe_insert(attr, frame=int(frames[i]))
        for fc in (o.animation_data.action.fcurves if o.animation_data and o.animation_data.action else []):
            if fc.data_path in ("hide_render", "hide_viewport"):
                for k in fc.keyframe_points:
                    k.interpolation = "CONSTANT"


def run(ctx):
    out = {}
    frames = ctx.frames
    rel = frames - ctx.frame0                        # clip-relative frame numbers
    for spec in ctx.data.get("vehicle", []):
        name = spec["prop"]
        if name not in ctx.props:
            raise BuildError(f"vehicle: no prop {name!r}")
        prop = ctx.props[name]
        set_name, _, path_name = spec["path"].partition(":")
        if set_name not in ctx.sets:
            raise BuildError(f"vehicle {name!r}: no set {set_name!r}")
        st = ctx.sets[set_name]
        cp = st.card["paths"][path_name]
        path = st.path(path_name)
        speed = spec.get("speed", 20.0)
        keys = speed if isinstance(speed, list) else [[-1e6, float(speed)], [1e6, float(speed)]]
        s_rel, v = speed_profile(keys, ctx.fps, rel)
        sign = lane_dir(cp, spec)
        s_rel = sign * s_rel                             # along the path: an oncoming car's arc length falls
        at = float(spec.get("at", 0.0))
        if spec.get("meet"):
            mt = spec["meet"]
            other = getattr(ctx, "vehicles", {}).get(mt.get("vehicle"))
            if other is None:
                raise BuildError(f"vehicle {name!r}: meet.vehicle {mt.get('vehicle')!r} is not an earlier [[vehicle]] "
                                 f"(have {sorted(getattr(ctx, 'vehicles', {}))})")
            i = int(np.argmin(np.abs(frames - ctx.frame(float(mt["t"])))))
            at = float(other["s"][i]) + float(mt.get("ahead", 0.0)) - float(s_rel[i])
        s = at + s_rel
        off_path = (s < 0.0) | (s > path.length)
        if off_path.any() and not spec.get("leave"):
            raise BuildError(f"vehicle {name!r} leaves the path: arc length {s.min():.0f}..{s.max():.0f} m, path "
                             f"{path.length:.0f} m (adjust `at`, `meet`, `speed` or the path length; traffic: leave = true)")
        if off_path.all():
            raise BuildError(f"vehicle {name!r} is never on the path (arc length {s.min():.0f}..{s.max():.0f} m)")
        if off_path.any():
            s = np.clip(s, 0.0, path.length)
        off = lane_offset(cp, spec.get("lane", 0.0))
        pos = path.offset(s, off, float(spec.get("height", 0.0)))
        head = np.unwrap(path.heading(s))
        kappa = path.curvature(s)
        a_lat = v * v * kappa
        a_lon = np.gradient(v) * ctx.fps
        roll = np.radians(float(spec.get("roll", 1.2)) * a_lat / G) * sign      # its own left is the path's right
        pitch = np.radians(float(spec.get("pitch", 0.8)) * a_lon / G)
        yaw = head + math.pi / 2 + (math.pi if sign < 0 else 0.0)     # the prop's -Y faces the direction of travel
        root = prop.root
        root.rotation_mode = "XYZ"
        K.key_vec(root, "location", frames, pos, interp="LINEAR")
        # body: yaw about Z, then lean out of the turn (roll about the car's forward axis) and pitch about its left axis:
        # a positive Euler X tips the nose DOWN, so the nose goes up under acceleration (a_lon > 0) and dives under braking
        eul = np.stack([-pitch, -roll, yaw], 1)
        K.key_vec(root, "rotation_euler", frames, eul, interp="LINEAR")
        if off_path.any():                               # after the transform keys (they write the object's action)
            _key_hidden(root, frames, off_path)
        info = {"path": spec["path"], "lane_offset": off, "from_m": round(float(s[0]), 1),
                "to_m": round(float(s[-1]), 1), "max_lat_g": round(float(np.abs(a_lat).max() / G), 3)}
        # wheels spin with the distance travelled
        for w in prop.card.get("wheels", []):
            ob = bpy.data.objects.get(w["object"])
            if ob is None:
                raise BuildError(f"vehicle {name!r}: wheel object {w['object']!r} missing")
            ob.rotation_mode = "QUATERNION"
            q0 = ob.rotation_quaternion.copy()
            ax = Vector(w.get("axis", (1, 0, 0))).normalized()  # local axis pointing to the car's left (+X)
            ang = sign * s / float(w["radius"])                 # + about the left axis: the tyre top moves forward (-Y)
            K.key_vec(ob, "rotation_quaternion", frames,
                      K.continuous([tuple(q0 @ Quaternion(ax, a)) for a in ang]), interp="LINEAR")
        # steering wheel follows the road's curvature
        sw = prop.card.get("steering")
        if sw:
            ob = bpy.data.objects.get(sw.get("object", ""))
            if ob is None:
                raise BuildError(f"vehicle {name!r}: steering object {sw.get('object')!r} missing")
            ob.rotation_mode = "QUATERNION"
            q0 = ob.rotation_quaternion.copy()
            ax = Vector(sw.get("axis", (0, 0, 1))).normalized()
            ratio = float(spec.get("steer_ratio", sw.get("ratio", 14.0)))
            delta = np.arctan(float(spec.get("wheelbase", 2.6)) * kappa) * ratio * sign
            K.key_vec(ob, "rotation_quaternion", frames,
                      K.continuous([tuple(q0 @ Quaternion(ax, d)) for d in delta]), interp="LINEAR")
            info["steer_deg_max"] = round(float(np.degrees(np.abs(delta)).max()), 1)
        ctx.vehicles = getattr(ctx, "vehicles", {})
        ctx.vehicles[name] = {"s": s, "v": v, "pos": pos}
        out[name] = info
        ctx.log("vehicle", name, info)
    return out
