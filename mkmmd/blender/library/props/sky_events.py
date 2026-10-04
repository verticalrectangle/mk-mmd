"""Things that happen in a night sky at a clip time: `meteor` (a shooting star) and `airplane` (an airliner's lights crossing
high up). Both are keyed at build on the song's clock: clip seconds are scene frames through the scene's `mk_frame0` (the
scene stage sets it) and fps, so nothing runs at render time but drivers for the strobe.

    [[prop]]
    name = "meteor1"
    card = "library:meteor"
    at = [0.0, -600.0, 0.0]                   # the point the sky is seen from (the middle of the action)
    slots = { t = 29.4, dur = 0.7, from = [-60.0, 34.0], to = [-38.0, 22.0] }

    [[prop]]
    name = "plane1"
    card = "library:airplane"
    at = [0.0, -600.0, 0.0]
    slots = { t0 = 19.0, t1 = 21.5, from = [-40.0, 18.0], to = [-10.0, 21.0] }

Directions are [azimuth, elevation] in degrees seen from the root (azimuth from +X toward +Y, as night_sky's `moon`), at
`distance` m (meteor 2500, airplane 3000).

meteor slots: t (clip seconds the streak appears), dur (0.6), from / to ([az, el] of its head at the start / end), length (deg of
sky its tail covers at the brightest, 7), color ("text"), strength (60), distance
airplane slots: t0, t1 (clip seconds it flies from `from` to `to`), lights (size of the light spheres, m: 9), strobe (white
strobe on 0.08 s of every 1.2 s), red, green, white (colours: love, foam, text), strength (40), distance"""
import math

import bpy
import numpy as np

from ....core import shell as CS
from .. import shell as SH
from . import register
from .hearts import _colour


def direction(az, el):
    a, e = math.radians(az), math.radians(el)
    return np.array([math.cos(e) * math.cos(a), math.cos(e) * math.sin(a), math.sin(e)])


def _clip_frame(t):
    sc = bpy.context.scene
    return float(sc.get("mk_frame0", sc.frame_start)) + float(t) * sc.render.fps / sc.render.fps_base


def _emitter(name, rgb, strength, gradient=None):
    """An emission material; `gradient` = length along the object's +X over which it fades out (the meteor's tail)."""
    m = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    em = nt.nodes.new("ShaderNodeEmission")
    em.inputs["Color"].default_value = (*rgb, 1.0)
    em.inputs["Strength"].default_value = strength
    nt.links.new(em.outputs[0], out.inputs["Surface"])
    if gradient:
        tc = nt.nodes.new("ShaderNodeTexCoord")
        sep = nt.nodes.new("ShaderNodeSeparateXYZ")
        ramp = nt.nodes.new("ShaderNodeMapRange")
        ramp.inputs["From Min"].default_value, ramp.inputs["From Max"].default_value = 0.0, gradient
        ramp.inputs["To Min"].default_value, ramp.inputs["To Max"].default_value = 1.0, 0.0
        pw = nt.nodes.new("ShaderNodeMath")
        pw.operation, pw.inputs[1].default_value = "POWER", 2.2
        mul = nt.nodes.new("ShaderNodeMath")
        mul.operation = "MULTIPLY"
        mul.inputs[1].default_value = strength
        nt.links.new(tc.outputs["Object"], sep.inputs[0])
        nt.links.new(sep.outputs["X"], ramp.inputs["Value"])
        nt.links.new(ramp.outputs[0], pw.inputs[0])
        nt.links.new(pw.outputs[0], mul.inputs[0])
        nt.links.new(mul.outputs[0], em.inputs["Strength"])
        return m, mul.inputs[1]
    return m, em.inputs["Strength"]


@register("meteor")
def meteor(name, coll, root, slots=None):
    s = dict(slots or {})
    t, dur = float(s.get("t", 0.0)), float(s.get("dur", 0.6))
    dist = float(s.get("distance", 2500.0))
    p0, p1 = direction(*s.get("from", [-60.0, 34.0])) * dist, direction(*s.get("to", [-40.0, 22.0])) * dist
    tail = math.radians(float(s.get("length", 7.0))) * dist
    m, strength = _emitter(f"{name}_streak", _colour(s.get("color", "text"), s), float(s.get("strength", 60.0)), gradient=tail)
    n = 9
    path = [(tail * k / (n - 1), 0.0, 0.0) for k in range(n)]
    radius = [dist * 0.0009 * (1.0 - 0.85 * k / (n - 1)) for k in range(n)]      # the head's width, tapering to the tail
    o = SH.mesh_object(f"{name}_streak", CS.tube(path, radius, sides=6), coll, root, lambda _r: m)
    SH.exempt(o, "a light streak in the sky")
    d = (p1 - p0) / max(np.linalg.norm(p1 - p0), 1e-9)              # the tail trails behind the head (+X opposite to motion)
    yaw, pitch = math.atan2(-d[1], -d[0]), math.asin(max(-1.0, min(1.0, -d[2])))
    o.rotation_euler = (0.0, -pitch, yaw)
    f0, f1 = _clip_frame(t), _clip_frame(t + dur)
    for f, p, sc, e in ((f0 - 1, p0, 0.05, 0.0), (f0, p0, 0.3, 1.0), (f0 + 0.35 * (f1 - f0), p0 + 0.35 * (p1 - p0), 1.0, 1.0),
                        (f1, p1, 0.7, 0.0)):
        o.location, o.scale = tuple(p), (sc, 1.0, 1.0)
        o.keyframe_insert("location", frame=f)
        o.keyframe_insert("scale", frame=f)
        strength.default_value = float(s.get("strength", 60.0)) * e
        strength.keyframe_insert("default_value", frame=f)
    lo, hi = np.minimum(p0, p1) - tail, np.maximum(p0, p1) + tail
    return {"size": [round(float(v), 1) for v in hi - lo], "origin": "center", "front": "-Y", "blocks": False,
            "bounds": {"min": [round(float(v), 1) for v in lo], "max": [round(float(v), 1) for v in hi]}, "layers": []}


@register("airplane")
def airplane(name, coll, root, slots=None):
    s = dict(slots or {})
    t0, t1 = float(s.get("t0", 0.0)), float(s.get("t1", 3.0))
    dist = float(s.get("distance", 3000.0))
    p0, p1 = direction(*s.get("from", [-40.0, 18.0])) * dist, direction(*s.get("to", [-10.0, 21.0])) * dist
    size, strength = float(s.get("lights", 9.0)), float(s.get("strength", 40.0))
    body = bpy.data.objects.get(f"{name}_body") or bpy.data.objects.new(f"{name}_body", None)
    if body.name not in coll.objects:
        coll.objects.link(body)
    body.parent = root
    d = (p1 - p0) / max(np.linalg.norm(p1 - p0), 1e-9)
    body.rotation_euler = (0.0, 0.0, math.atan2(d[1], d[0]))
    lamp = CS.lathe([(0.0, -1.0), (0.7, -0.7), (1.0, 0.0), (0.7, 0.7), (0.0, 1.0)], seg=10)
    for key, slot, at, blink in (("red", "love", (0.0, 4.0 * size, 0.0), False), ("green", "foam", (0.0, -4.0 * size, 0.0), False),
                                 ("white", "text", (-3.0 * size, 0.0, 0.5 * size), True)):
        m, st = _emitter(f"{name}_{key}", _colour(s.get(key, slot), s), strength)
        o = SH.mesh_object(f"{name}_{key}", lamp, coll, body, lambda _r, m=m: m, loc=at)
        o.scale = (size,) * 3
        SH.exempt(o, "an aircraft light")
        if blink:                                                     # a strobe: on 0.08 s of every 1.2 s
            fps = bpy.context.scene.render.fps / bpy.context.scene.render.fps_base
            dr = st.driver_add("default_value").driver
            dr.type = "SCRIPTED"
            dr.expression = f"{strength * 3:.1f} if fmod(frame, {1.2 * fps:.2f}) < {0.08 * fps:.2f} else 0.0"
    for f, p in ((_clip_frame(t0), p0), (_clip_frame(t1), p1)):
        body.location = tuple(p)
        body.keyframe_insert("location", frame=f)
    for fc in body.animation_data.action.fcurves:
        fc.extrapolation = "LINEAR"
        for k in fc.keyframe_points:
            k.interpolation = "LINEAR"
    lo, hi = np.minimum(p0, p1) - 8 * size, np.maximum(p0, p1) + 8 * size
    return {"size": [round(float(v), 1) for v in hi - lo], "origin": "center", "front": "-Y", "blocks": False,
            "bounds": {"min": [round(float(v), 1) for v in lo], "max": [round(float(v), 1) for v in hi]}, "layers": []}
