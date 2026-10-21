"""stepped: the cast on twos (docs/design.md: Stepped motion). `[[stepped]] = {from, to, cast, fps = 12}` holds each named
member's whole motion for 1 / fps s at a time over its window: every animated object under the member's root (the root,
the armature and its bones, the hands' goals, the morphs' placeholder) and the shape keys of its meshes keep the pose of
the window's last step (mkmmd.core.stepped) until the next one, keyed CONSTANT from half a frame before each step so no
frame's shutter spans a step, while the cameras keep every frame. A VMD playing on an NLA strip (the motion stage's) is
held through the strip's own clock (Animated Strip Time, keyed to play as it did). It runs after everything that keys
the cast (pose, motion, perform, keys, sim, folds), so the hair steps with her; a garment's cloth copy that Blender
simulates does not step, and a curve with modifiers (none of mk's own) is left as it is and counted."""
import numpy as np

from ...core import stepped as STP
from .. import keys as K
from . import BuildError

AUTO_CLAMPED = 4             # handle type (RNA enum) of the keys a window adds


def _owners(m):
    """The datablocks that animate member `m`: every object under its root (the root too) and its meshes' shape keys."""
    root = m.root or m.arm
    if root is None:
        return []
    out = []
    for ob in [root, *root.children_recursive]:
        out.append(ob)
        if ob.type == "MESH" and ob.data.shape_keys is not None:
            out.append(ob.data.shape_keys)
    return out


def _clock(strip):
    """An NLA strip's own time as an fcurve (Animated Strip Time), keyed to play as it did: its action's first frame at
    its start, its last at its end, linear between (mk's motion strips neither repeat nor run backward)."""
    if not strip.use_animated_time:
        strip.use_animated_time = True
        kp = strip.fcurves.find("strip_time").keyframe_points
        kp.clear()
        kp.add(2)
        kp.foreach_set("co", [strip.frame_start, strip.action_frame_start, strip.frame_end, strip.action_frame_end])
        kp.foreach_set("interpolation", [K.INTERP["LINEAR"]] * 2)
    fc = strip.fcurves.find("strip_time")
    fc.update()
    return fc


def _curves(m, windows):
    """[(fcurve, the windows it steps in)] of member `m`: every curve of the actions that move it, and the clock of
    each NLA strip that plays under it (a VMD's bones and morphs) with the windows that overlap the strip."""
    out, seen = [], set()
    for owner in _owners(m):
        ad = owner.animation_data
        if ad is None:
            continue
        if ad.action is not None and ad.action.name not in seen:
            seen.add(ad.action.name)
            out += [(fc, windows) for fc in ad.action.fcurves]
        for track in ad.nla_tracks:
            for s in track.strips:
                over = [w for w in windows if w["from"] < s.frame_end and w["to"] > s.frame_start]
                if over:
                    out.append((_clock(s), over))
    return out


def _get(kp, name, width):
    a = np.empty(len(kp) * width, np.float64 if name in ("co", "handle_left", "handle_right") else np.int32)
    kp.foreach_get(name, a)
    return a.reshape(-1, width) if width > 1 else a


def hold(fc, w):
    """Key the fcurve so it holds its value at each step of window `w` from half a frame before that step to half a frame
    before the next, and takes up its own curve again from half a frame before the window ends. The keys outside the
    window keep their values, interpolation and handles."""
    kp = fc.keyframe_points
    steps = w["steps"]
    frames = np.asarray([s - 0.5 for s in steps] + [w["to"] - 0.5], np.float64)
    values = np.asarray([fc.evaluate(s) for s in steps] + [fc.evaluate(w["to"] - 0.5)], np.float64)
    co = _get(kp, "co", 2)
    old = {k: _get(kp, k, 2) for k in ("handle_left", "handle_right")}
    old.update({k: _get(kp, k, 1) for k in ("interpolation", "handle_left_type", "handle_right_type", "easing")})
    keep = (co[:, 0] < frames[0] - 1e-6) | (co[:, 0] > frames[-1] + 1e-6)
    before = np.nonzero(co[:, 0] <= frames[-1] + 1e-6)[0]
    tail = int(old["interpolation"][before[-1]]) if len(before) else K.INTERP["LINEAR"]   # the closing key goes on as it did
    n = len(frames)
    new_co = np.stack([frames, values], 1)
    rows = {"co": (co[keep], new_co), "handle_left": (old["handle_left"][keep], new_co),
            "handle_right": (old["handle_right"][keep], new_co),
            "interpolation": (old["interpolation"][keep],
                              np.asarray([K.INTERP["CONSTANT"]] * (n - 1) + [tail], np.int32)),
            "handle_left_type": (old["handle_left_type"][keep], np.full(n, AUTO_CLAMPED, np.int32)),
            "handle_right_type": (old["handle_right_type"][keep], np.full(n, AUTO_CLAMPED, np.int32)),
            "easing": (old["easing"][keep], np.zeros(n, np.int32))}
    order = np.argsort(np.concatenate([co[keep][:, 0], frames]), kind="stable")
    kp.clear()
    kp.add(len(order))
    for name, (a, b) in rows.items():
        kp.foreach_set(name, np.concatenate([a, b])[order].ravel())
    fc.update()


def run(ctx):
    entries = ctx.data.get("stepped", [])
    if not entries:
        return {}
    try:
        windows = STP.normalize(entries, list(ctx.cast), ctx.fps, ctx.frame0, ctx.end)
    except STP.StepError as e:
        raise BuildError(str(e)) from None
    out = {}
    for name, m in ctx.cast.items():
        mine = [w for w in windows if name in w["cast"]]
        if not mine:
            continue
        held, skipped = 0, 0
        for fc, over in _curves(m, mine):
            if not len(fc.keyframe_points):
                continue
            if len(fc.modifiers):
                skipped += 1
                continue
            for w in over:
                hold(fc, w)
            held += 1
        out[name] = {"windows": [[w["from"], w["to"] - 1] for w in mine], "poses": sum(len(w["steps"]) for w in mine),
                     "curves": held}
        if skipped:
            out[name]["curves_with_modifiers"] = skipped
        ctx.log("stepped", name, out[name]["windows"], out[name]["poses"], "poses over", held, "curves")
    return out
