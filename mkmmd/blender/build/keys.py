"""keys: declarative keys on set, prop and object properties (storm flashes, fog, steam, light levels, visibility).

[[key]] keys
  target = "cafe"           a set or prop name (its root), or any object name
  prop = "fog"              a custom property of the target (set and prop cards document theirs), or an RNA data
                            path such as "location" / "hide_render" / "data.energy"
  index = 0                 array index for vector paths
  keys = [[t, v], ...]      clip seconds and values
  interp = "BEZIER" | "LINEAR" | "CONSTANT"
  relative = false          true: the values are offsets added to what the property already does at each key (a
                            value another stage solved or animated, e.g. a hand's grip target), and the existing
                            animation outside the keys' time span is kept
  lane = "kick"             instead of `keys`: on every hit of that lane of the [audio] hits file (mkmmd.core.lanes;
  min_db, every, from, to   at or above `min_db`, every n-th, inside from..to) the property takes the hit's own time
                            (CONSTANT by default: a prop's `start` goes off again on each hit) or, with
  pulse = [[dt, v], ...]    keys that many seconds after each hit (a flash on every kick), cut short by the next hit
  step = 0.6, start = 0     with `lane`: the property counts the hits instead (mkmmd.core.lanes.steps): `start` (default
                            its value as built) before the first, each hit adding `step`; a jump on the hit (CONSTANT)
                            or, with `pulse` = [[dt, share], ...], each step shaped by shares 0..1 of it (LINEAR)
  values = [v0, v1, ...]    with `lane`: on each hit the property takes the next value of the list instead (hit i the
                            value i % len(values), cycling; CONSTANT by default: a penlight's colour index stepping
                            through its colours on every accent); not with `pulse` or `step`

The objects a [[glitch]] names (docs/design.md: Shots: Glitches) are hidden from its `to` on (a CONSTANT `hide_render` key
added to what they already have), so they stay gone once it has broken them up.

A prop card's `rates` = [{rate, into, per}] (a turntable's `rpm` into its `turns`): once the keys have animated the root's
property `rate`, its property `into` is keyed on every frame with the running integral of rate / per over clip seconds,
0 at clip second 0, so a driver that reads `into` turns a part at the keyed speed and a speed change never jumps it.

A prop card's `deals` = [{from, into}] (confetti's `start` into its volleys' `start0`, `start1`, ...): once the keys have
animated the root's property `from`, its keyframes are dealt out in time order, in turn, into the properties `into` as
CONSTANT keys (keyframe i into into[i % len(into)]), so each volley falls its whole life while the next one goes off.
"""
import math

import bpy
import numpy as np

from ...core import lanes as LN
from ...core import shotstyle as SS
from ...core import transition as TR
from .. import keys as K
from ..styles import object_records
from . import BuildError


def _lane_keys(ctx, spec, where, base=None):
    """[[t, v], ...] and the default interpolation of a [[key]] that fires on a lane (`lane`, `pulse`, `step`, `start`,
    `values`); `base` is the property's value as built (a counting key's default `start`)."""
    if "keys" in spec:
        raise BuildError(f"{where}: give `keys` or `lane`, not both")
    hits_file = (ctx.data.get("audio") or {}).get("hits")
    if not hits_file:
        raise BuildError(f"{where}: `lane` reads the [audio] hits file (write it with `mk timeline hits`)")
    try:
        c = LN.choice(spec, where)
        ts = LN.pick(LN.load(ctx.path(hits_file)), c["lane"], c["min_db"], c["every"], c["t0"], c["t1"])
    except LN.LaneError as e:
        raise BuildError(f"{where}: {e}") from None
    if not ts:
        raise BuildError(f"{where}: lane {c['lane']!r} has no hit there (from, to, min_db)")
    pulse = spec.get("pulse")
    if pulse is not None and not (isinstance(pulse, list) and pulse and all(
            isinstance(k, list) and len(k) == 2 and all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in k)
            for k in pulse)):
        raise BuildError(f"{where}: pulse = [[dt, value], ...] (seconds after each hit)")
    values = spec.get("values")
    if values is not None and (pulse is not None or spec.get("step") is not None):
        raise BuildError(f"{where}: give `values` or `{'pulse' if pulse is not None else 'step'}`, not both (`values`: "
                         f"on each hit the property takes the next value of the list)")
    if spec.get("step") is not None:
        step, start = spec["step"], spec.get("start", base)
        for k, v in (("step", step), ("start", start)):
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                raise BuildError(f"{where}: {k} = {v!r}: a number (a counting key: `start`, then `step` more per hit)")
        if pulse is not None and any(dt < 0 for dt, _ in pulse):
            raise BuildError(f"{where}: a counting key's pulse = [[dt, share], ...] with dt >= 0 (seconds after the hit)")
        return LN.steps(ts, start, step, None if pulse is None else sorted(pulse)), \
            "CONSTANT" if pulse is None else "LINEAR"
    if "start" in spec:
        raise BuildError(f"{where}: `start` goes with `step` (a counting key)")
    if values is not None:
        if not (isinstance(values, list) and values and all(
                isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in values)):
            raise BuildError(f"{where}: values = [v0, v1, ...]: a list of numbers, the value the property takes on each "
                             f"hit in turn (got {values!r})")
        return [[t, float(values[i % len(values)])] for i, t in enumerate(ts)], "CONSTANT"
    if pulse is None:
        return [[t, t] for t in ts], "CONSTANT"
    out = []
    for t, nxt in zip(ts, ts[1:] + [None]):
        out += [[t + float(dt), float(v)] for dt, v in pulse if nxt is None or t + float(dt) < nxt - 1e-6]
    return out, "LINEAR"


def _target(ctx, name):
    if name in getattr(ctx, "sets", {}):
        return ctx.sets[name].root
    if name in ctx.props:
        return ctx.props[name].root
    ob = bpy.data.objects.get(name)
    if ob is None:
        raise BuildError(f"key target {name!r}: no set, prop or object of that name")
    return ob


def _base_values(id_data, path, index, frames):
    """The property's current value at each frame: its fcurve when animated, else its static value."""
    ad = getattr(id_data, "animation_data", None)
    fc = ad.action.fcurves.find(path, index=index) if ad and ad.action else None
    if fc is not None and len(fc.keyframe_points):
        return [fc.evaluate(f) for f in frames]
    val = id_data.path_resolve(path)
    try:
        val = val[index]
    except TypeError:
        pass
    return [float(val)] * len(frames)


def run(ctx):
    out = {}
    for spec in ctx.data.get("key", []):
        ob = _target(ctx, spec["target"])
        prop = spec["prop"]
        where = f"[[key]] {spec['target']}.{prop}"
        id_data, path = ob, prop
        if prop in ob.keys():
            path = f'["{prop}"]'
        elif prop.startswith("data.") and ob.data is not None:
            id_data, path = ob.data, prop[5:]
        interp = spec.get("interp", "BEZIER")
        if spec.get("lane") is not None:
            base = None
            if spec.get("step") is not None and "start" not in spec:
                base = _base_values(id_data, path, int(spec.get("index", 0)), [ctx.frame(0.0)])[0]
            pts, default = _lane_keys(ctx, spec, where, base)
            interp = spec.get("interp", default)
        elif "step" in spec or "start" in spec:
            raise BuildError(f"{where}: `step` and `start` count the hits of a `lane`")
        elif "values" in spec:
            raise BuildError(f"{where}: `values` steps through a list on the hits of a `lane` (or put the values in `keys`)")
        else:
            pts = sorted(spec.get("keys") or [])
        if not pts:
            raise BuildError(f"{where}: no keys")
        index = int(spec.get("index", 0))
        frames = [ctx.frame(float(t)) for t, _ in pts]
        values = [float(v) for _, v in pts]
        relative = bool(spec.get("relative", False))
        try:
            if relative:
                values = [b + v for b, v in zip(_base_values(id_data, path, index, frames), values)]
            K.key_prop(id_data, path, frames, values, index=index, interp=interp, replace=not relative)
        except (TypeError, RuntimeError, ValueError) as e:
            raise BuildError(f"[[key]] {spec['target']}.{prop}: {e}")
        out[f"{spec['target']}.{prop}"] = len(pts)
    out.update(_rates(ctx))
    out.update(_deals(ctx))
    if out:
        ctx.log("keys", len(out))
    hidden = _glitched(ctx)
    if hidden:
        out["_glitch_hidden"] = hidden
    return out


def _rates(ctx):
    """{"<prop>.<into>": frames keyed} for the props whose card has `rates` and whose `rate` is animated (see the module
    docstring): Simpson's rule over every frame of the build, linear beyond it."""
    out = {}
    frames = np.arange(ctx.start, ctx.end + 1, dtype=float)
    for name, p in ctx.props.items():
        for r in p.card.get("rates") or []:
            ad = p.root.animation_data
            fc = ad.action.fcurves.find(f'["{r["rate"]}"]') if ad and ad.action else None
            if fc is None or not len(fc.keyframe_points) or len(frames) < 2:
                continue
            v = np.array([fc.evaluate(f) for f in frames])
            mid = np.array([fc.evaluate(f + 0.5) for f in frames[:-1]])
            step = (v[:-1] + 4.0 * mid + v[1:]) / (6.0 * ctx.fps * float(r.get("per", 1.0)))
            total = np.concatenate([[0.0], np.cumsum(step)])
            total -= np.interp(ctx.frame0, frames, total)
            K.key_prop(p.root, f'["{r["into"]}"]', frames, total, interp="LINEAR").extrapolation = "LINEAR"
            out[f"{name}.{r['into']}"] = len(frames)
    return out


def _deals(ctx):
    """{"<prop>.<into>": keys dealt} for the props whose card has `deals` and whose `from` is animated (see the module
    docstring): its keyframes in time order, keyframe i into into[i % len(into)] (same frame and value, CONSTANT); an
    `into` that gets none keeps its value as built."""
    out = {}
    for name, p in ctx.props.items():
        for d in p.card.get("deals") or []:
            ad = p.root.animation_data
            fc = ad.action.fcurves.find(f'["{d["from"]}"]') if ad and ad.action else None
            if fc is None or not len(fc.keyframe_points):
                continue
            co = np.empty(2 * len(fc.keyframe_points))
            fc.keyframe_points.foreach_get("co", co)
            order = np.argsort(co[0::2], kind="stable")
            f, v, into = co[0::2][order], co[1::2][order], list(d["into"])
            for i, prop in enumerate(into[:len(f)]):
                K.key_prop(p.root, f'["{prop}"]', f[i::len(into)], v[i::len(into)], interp="CONSTANT")
                out[f"{name}.{prop}"] = len(f[i::len(into)])
    return out


def _glitched(ctx):
    """{glitch index: objects hidden}: every object a [[glitch]] names hidden from the frame its window ends on."""
    specs = ctx.data.get("glitch") or []
    specs = [specs] if isinstance(specs, dict) else specs
    recs = object_records(bpy.context.scene) if specs else []
    out = {}
    for i, g in enumerate(specs):
        pats = [g["objects"]] if isinstance(g["objects"], str) else list(g["objects"])
        names = SS.select(recs, pats)
        if not names:
            ctx.log("WARNING", f"glitch {i}: no object matches {pats}, so nothing is hidden after it")
            continue
        f1 = TR.frame_of(g["to"], ctx.fps, ctx.frame0)
        for n in names:
            ob = bpy.data.objects[n]
            before = _base_values(ob, "hide_render", 0, [f1 - 1])[0]          # as it was on the window's last frame
            K.key_prop(ob, "hide_render", [f1 - 1, f1], [before, 1.0], interp="CONSTANT", replace=False)
        out[i] = len(names)
    return out
