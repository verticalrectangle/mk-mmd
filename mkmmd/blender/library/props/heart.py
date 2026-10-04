"""heart: one big puffy heart that pounds on the beat ("ba-dump": a bump on the beat and a smaller one after it), with a ring
of little hearts bursting out of it every few beats. Animated by drivers on the scene frame (no Python at render time) on the
song's clock: `bpm` and a clip second `start` that falls on a beat (clip seconds are scene frames through the scene's
`mk_frame0`, the frame of clip second 0). The heart a lovestruck character wears on her chest in a silhouette shot.

    [[prop]]
    name = "doki"
    card = "library:heart"
    attach = "reisen_love:upper_body2"            # rides her chest (offset / attach_rot in the bone's head frame)
    slots = { size = 0.2, bpm = 127.05, start = 0.0, beat = 0.3, burst = 8, every = 4, reach = 0.45 }

Objects: `<name>_heart` (the big heart) and `<name>_burst<i>` (the little ones). They share one mesh (mkmmd.core.hearts: a
classic two-lobed heart inflated to a pillow whose face looks along -Y; the ring flies out in the face's plane, x and z) and are
tagged `mk_heart`, so a silhouette shot takes them as accents: `accent = ["prop:mk_heart"]`, `keep = [..., "doki_*"]`.

Options (the `slots`; the colour is a palette slot or hex):
  heart        colour (love)                         glow         emission strength (0.4; 0 = lit by the scene)
  size         width of the big heart, m (0.2)       puff         pillow thickness / width (0.42)
  bpm          the song's tempo (120)                start        a clip second on a beat (0.0)
  beat         how much the heart swells on the beat (0.25: 25 % wider; the after-beat bump is 60 % of it; 0 = still)
  burst        little hearts in a burst (8; 0 = none)              every   beats from one burst to the next (4: a bar)
  reach        how far they fly, m (0.45)            burst_size   their width as a share of the big heart's (0.32)
  turn         degrees the ring is turned in its plane (0: the first heart flies straight up)
The clock values are custom properties of the root (`bpm`, `start`, `beat`, `every`, `reach`), read by the drivers every
frame: `[[key]] target = "doki", prop = "beat"` calms or quickens it.

Frame: the root is the big heart's centre, its face looks along -Y. Card keys: size, bounds, front "-Y", params, `hearts`
{objects, accent, tag}."""
import math
import re

import bpy

from ....core import hearts as HR
from .. import shell as SH
from . import register
from .cafe_kit import clear_drivers
from .hearts import _colour, _material

NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
FUNCS = {"fmod", "min", "max", "abs", "pow"}
LIVE = {"f0": "frame of clip second 0", "fps": "frames per second", "bpm": "tempo", "start": "a clip second on a beat",
        "beat": "swell on the beat", "every": "beats between bursts", "reach": "how far the little hearts fly (m)",
        "size": "width of the big heart (m)", "bk": "width of a little heart (m)"}


def options(slots):
    """The prop's numbers from its slots, checked."""
    num = {"size": 0.2, "puff": HR.PUFF, "glow": 0.4, "bpm": 120.0, "start": 0.0, "beat": 0.25, "every": 4.0, "reach": 0.45,
           "burst_size": 0.32, "turn": 0.0}
    o = {k: float(slots.get(k, v)) for k, v in num.items()}
    o["burst"] = int(slots.get("burst", 8))
    for k in ("size", "puff", "bpm", "every"):
        if o[k] <= 0:
            raise ValueError(f"heart: {k} must be > 0, got {o[k]}")
    if o["burst"] < 0 or o["beat"] < 0 or o["reach"] < 0:
        raise ValueError("heart: burst, beat and reach must be >= 0")
    return o


def _driver(owner, path, expr, root, index=-1):
    """A scripted driver `expr` on `owner[path]`; its variables are the root's properties of the same names."""
    res = owner.driver_add(path, index) if index >= 0 else owner.driver_add(path)
    for fc in res if isinstance(res, list) else [res]:
        d = fc.driver
        d.type = "SCRIPTED"
        d.expression = expr
        for nm in sorted(set(NAME.findall(expr)) - FUNCS - {"frame"}):
            if nm not in LIVE:
                raise ValueError(f"heart: the expression for {path} uses {nm!r}, which is not a root property")
            v = d.variables.new()
            v.name, v.type = nm, "SINGLE_PROP"
            v.targets[0].id_type, v.targets[0].id = "OBJECT", root
            v.targets[0].data_path = f'["{nm}"]'
        if len(expr) > 255 or not (d.is_valid and d.is_simple_expression):
            raise ValueError(f"heart: the driver on {path} is not a Blender simple expression ({expr})")


@register("heart")
def heart(name, coll, root, slots=None):
    slots = dict(slots or {})
    o = options(slots)
    sc = bpy.context.scene
    colour = slots.get("heart", "love")
    mat = _material(f"{name}_heart", _colour(colour, slots), o["glow"])
    for old in [ob for ob in bpy.data.objects if ob.name == f"{name}_heart" or
                (ob.name.startswith(f"{name}_burst") and ob.name[len(name) + 6:].isdigit())]:
        SH.purge(old.name)
    me, _ = SH.mesh_data(f"{name}_heart", HR.heart_mesh(o["puff"]), lambda _role: mat)
    clear_drivers(root)
    values = {"f0": float(sc.get("mk_frame0", sc.frame_start)), "fps": sc.render.fps / sc.render.fps_base, "bpm": o["bpm"],
              "start": o["start"], "beat": o["beat"], "every": o["every"], "reach": o["reach"], "size": o["size"],
              "bk": o["size"] * o["burst_size"]}
    for k, v in values.items():
        root[k] = float(v)
        root.id_properties_ui(k).update(description=LIVE[k])
    big = bpy.data.objects.new(f"{name}_heart", me)
    coll.objects.link(big)
    big.parent = root
    big["mk_heart"] = True
    _driver(big, "scale", HR.heartbeat_expressions(0, 0, 0.0)["big"], root)
    objs = [big]
    for i in range(o["burst"]):
        ex = HR.heartbeat_expressions(i, o["burst"], o["turn"])
        b = bpy.data.objects.new(f"{name}_burst{i}", me)
        coll.objects.link(b)
        b.parent = root
        b["mk_heart"] = True
        b.location.y = -0.01                                 # just in front of the big heart
        a = math.radians(o["turn"]) + 2.0 * math.pi * i / o["burst"]
        b.rotation_euler.y = -0.35 * math.sin(a)            # leaning out the way it flies
        _driver(b, "location", ex["x"], root, 0)
        _driver(b, "location", ex["z"], root, 2)
        _driver(b, "scale", ex["s"], root)
        objs.append(b)
    half = 0.5 * o["size"] * (1 + 1.6 * o["beat"]) + (o["reach"] + 0.5 * o["size"] * o["burst_size"] if o["burst"] else 0.0)
    depth = 0.5 * o["puff"] * o["size"] * (1 + 1.6 * o["beat"])
    lo, hi = [-half, -depth - 0.01, -half], [half, depth, half]
    return {"size": [round(h - l, 4) for l, h in zip(lo, hi)], "origin": "center", "front": "-Y", "blocks": False,
            "bounds": {"min": [round(v, 4) for v in lo], "max": [round(v, 4) for v in hi]}, "layers": [],
            "slots": {"heart": colour}, "params": {k: values[k] for k in ("bpm", "start", "beat", "every", "reach")},
            "hearts": {"objects": [ob.name for ob in objs], "accent": f"{name}_*", "tag": "mk_heart"}}
