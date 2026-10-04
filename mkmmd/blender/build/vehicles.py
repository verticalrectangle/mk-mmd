"""vehicles: props that drive along a set's path (docs/design.md: Vehicles).

[[vehicle]] keys
  prop = "car"                  the prop to drive (its card may list wheels and a steering wheel, see below)
  path = "highway:road"         <set>:<path>
  lane = "R1" | 1.8             a lane name from the path's card, or metres left of the centerline
  speed = 24.0 | [[t, v], ...]  m/s, constant or keyed on clip seconds (linear between keys)
  at = 0.0                      arc length (m) where the vehicle is at clip time 0
  height = 0.0                  extra height above the path (m)
  roll = 1.2, pitch = 0.8       body roll / pitch (deg per g of lateral / longitudinal acceleration)
  wheelbase = 2.6, steer_ratio = 14.0
Card keys used: "wheels": [{object, radius, axis = [1, 0, 0] (local spin axis)}], "steering": {object, axis =
[0, 0, 1] (local), ratio}. The steering object turns by atan(wheelbase * curvature) * ratio; the hands gripping it
follow when their IK targets ride on it (pose stage, grip on a ring with an object)."""
import math

import bpy
import numpy as np
from mathutils import Quaternion, Vector

from ...core.path import Path, speed_profile
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
        path = Path(st.path_points(path_name))
        speed = spec.get("speed", 20.0)
        keys = speed if isinstance(speed, list) else [[-1e6, float(speed)], [1e6, float(speed)]]
        s_rel, v = speed_profile(keys, ctx.fps, rel)
        s = float(spec.get("at", 0.0)) + s_rel
        if s.min() < 0 or s.max() > path.length:
            raise BuildError(f"vehicle {name!r} leaves the path: arc length {s.min():.0f}..{s.max():.0f} m, path "
                             f"{path.length:.0f} m (adjust `at`, `speed` or the path length)")
        off = lane_offset(cp, spec.get("lane", 0.0))
        pos = path.offset(s, off, float(spec.get("height", 0.0)))
        head = np.unwrap(path.heading(s))
        kappa = path.curvature(s)
        a_lat = v * v * kappa
        a_lon = np.gradient(v) * ctx.fps
        roll = np.radians(float(spec.get("roll", 1.2)) * a_lat / G)
        pitch = np.radians(float(spec.get("pitch", 0.8)) * a_lon / G)
        yaw = head + math.pi / 2                      # the prop's -Y faces the direction of travel
        root = prop.root
        root.rotation_mode = "XYZ"
        K.key_vec(root, "location", frames, pos, interp="LINEAR")
        # body: yaw about Z, then lean out of the turn (roll about the car's forward axis) and nose dive (pitch)
        eul = np.stack([pitch, -roll, yaw], 1)
        K.key_vec(root, "rotation_euler", frames, eul, interp="LINEAR")
        info = {"path": spec["path"], "lane_offset": off, "from_m": round(float(s[0]), 1),
                "to_m": round(float(s[-1]), 1), "max_lat_g": round(float(np.abs(a_lat).max() / G), 3)}
        # wheels spin with the distance travelled
        for w in prop.card.get("wheels", []):
            ob = bpy.data.objects.get(w["object"])
            if ob is None:
                raise BuildError(f"vehicle {name!r}: wheel object {w['object']!r} missing")
            ob.rotation_mode = "QUATERNION"
            q0 = ob.rotation_quaternion.copy()
            ax = Vector(w.get("axis", (1, 0, 0))).normalized()
            ang = s / float(w["radius"])
            K.key_vec(ob, "rotation_quaternion", frames,
                      K.continuous([tuple(q0 @ Quaternion(ax, -a)) for a in ang]), interp="LINEAR")
        # steering wheel follows the road's curvature
        sw = prop.card.get("steering")
        if sw:
            ob = bpy.data.objects.get(sw["object"])
            ob.rotation_mode = "QUATERNION"
            q0 = ob.rotation_quaternion.copy()
            ax = Vector(sw.get("axis", (0, 0, 1))).normalized()
            ratio = float(spec.get("steer_ratio", sw.get("ratio", 14.0)))
            delta = np.arctan(float(spec.get("wheelbase", 2.6)) * kappa) * ratio
            K.key_vec(ob, "rotation_quaternion", frames,
                      K.continuous([tuple(q0 @ Quaternion(ax, d)) for d in delta]), interp="LINEAR")
            info["steer_deg_max"] = round(float(np.degrees(np.abs(delta)).max()), 1)
        ctx.vehicles = getattr(ctx, "vehicles", {})
        ctx.vehicles[name] = {"s": s, "v": v, "pos": pos}
        out[name] = info
        ctx.log("vehicle", name, info)
    return out
