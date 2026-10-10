"""perform: a character's life on top of its pose: gaze (eyes lead, head and neck follow), breathing, sway, idle nod,
beat bob, startles, blinks and lids, expressions, lip sync, twitches.

[perform.<cast>] keys (times in clip seconds; targets as in mkmmd.blender.build.targets)
  look = target                      idle gaze target (default: ahead at eye height)
  gaze = [{t, at, hold = 1.0, back = 0.4, rise = 0.3}]   look-at events
  glance = [{t, at, dur = 0.8, pitch = 2.75, rise = 0.12, fall = 0.25, blink = true}]   eye-only glances (the eyes
                                     lead to `at`, the head lifts `pitch` deg, the lids open, a blink follows); the
                                     head and neck do not turn
  head_share = 0.7                   share of a gaze turn taken by head + neck (the eyes do the rest, up to eye_max)
  head_limits = {yaw = 75, up = 35, down = 45}   the head + neck's range (deg): a target overhead or behind is met
                                     by the eyes, not by an impossible head turn
  neck_share = 0.35                  share of the head's turn carried by the neck
  eye_max = 24 (deg)
  breath = {per_min = 16.5, deg = 0.6}, sway = {deg = 0.37, period = 2.5}, nod = {deg = 0.48, period = 2.3}
  bob = {deg = 1.5, beats = [t...] | timeline = "audio/timeline.json" (the default), downbeat_accent = 1.6}
  startle = [t...]
  lean = [[t, deg], ...], turn = [[t, deg], ...], tilt = [[t, deg], ...]   extra upper-body lean forward / turn
                                     toward the model's left / sideways tilt toward its left over time, eased between
                                     keys and held before the first and after the last, on top of the pose's base (a
                                     reach that leans in and settles back, a head on a shoulder; hand targets still hold,
                                     except hands whose pose `ride` is a chest bone: they go with it)
  head_tilt = [[t, deg], ...]        the head rolls toward the model's left (on top of the gaze), keyed the same way
  rock = {deg, period = 4.0, phase = 0.0, hips = 0.0}, head_rock = {...}   a periodic sideways lean of the upper body (like
                                     `tilt`) and roll of the head (like `head_tilt`) toward the model's left: deg * sin(2 pi t /
                                     period + phase) of clip time, ADDED to the keyed tilts; unlike them it goes on at any clip
                                     time. `hips` (m) shifts the hips toward the left with the same swing, the feet planted:
                                     the weight goes from foot to foot
  crouch = [[t, metres], ...]        the hips drop by that much, the feet planted (the knees bend: a landing, a squat), eased
                                     between keys like `rise`
  blink = {per_min = 15, seed = 0, extra = [[t, dur], ...]}; lids = 0.0 (base lowering 0..1)
  sing = {timeline = "audio/timeline.json" (the default), lines = [a, b], mouth = 0.8, lead = -0.03, voice = "en-gb"}
  expressions = [{morph = "smile_eyes" (semantic or the model's own name), keys = [[t, value], ...]}]
  twitch = [{bones = ["ear_root.L", ...] | family = "ears", t = 3.2, deg = 14, axis = [1, 0, 0], dur = 0.22}]
  strum = {hand = "R", prop = "guitar", rhythm = "onsets:other" | "beats:8" | [t, ...], from = 0.96, to = 3.86, accent = "downbeats",
           depth = 0.003, span = 0.09, attack = 0.075, timeline = "audio/timeline.json", ...}   or a list of such tables: the
                                     pick hand (its grip is `grip = "guitar:strum"`) strokes across the strings on the rhythm,
                                     down strokes down, up strokes up, the pick meeting the first string at each strike time;
                                     it rests between windows (mkmmd.core.strum; docs/design.md: Perform)
  drum = {hand = "R", fingers = [...], deg = 25, lift = 0.16, roll = 0.02, beats | timeline, from, to}   the fingers tap on
                                     the beats from where their keys hold them (a grip's curl): lifted before, down on it
A table that is empty does nothing (the log says so); a table named after no cast member or holding a key this stage does
not read is refused (mkmmd.core.tables).
Everything is composed in the armature's frame (correct for characters riding vehicles), keyed per frame (LINEAR)."""
import json
import math
import zlib

import bpy
import numpy as np
from mathutils import Matrix, Quaternion, Vector

from ...core import families as FAM
from ...core import lipsync as LS
from ...core import perform as PF
from ...core import timeline as TL
from .. import keys as K
from .. import scene as S
from . import BuildError, targets

SPINE = ("upper_body", "upper_body2", "neck", "head")


def _morph(m, name):
    return (m.rig or {}).get("morphs", {}).get(name, name)


def _qarr(q):
    return Quaternion((q[0], q[1], q[2], q[3]))


def _sample_pass(ctx, m, look_refs, frames):
    """Per frame: armature world matrix, eye midpoint and every target point, both in ARMATURE space."""
    sc = bpy.context.scene
    arm = m.arm
    smap = S.semantic_map(arm)
    eyes_b = [smap[k] for k in ("eye.L", "eye.R") if k in smap] or [smap["head"]]
    A, E, T = [], [], {i: [] for i in range(len(look_refs))}
    for f in frames:
        sc.frame_set(int(f))
        dg = bpy.context.evaluated_depsgraph_get()
        ae = arm.evaluated_get(dg)
        Aw = ae.matrix_world.copy()
        inv = Aw.inverted()
        e = sum((ae.pose.bones[b].head for b in eyes_b), Vector()) / len(eyes_b)
        A.append(Aw)
        E.append(e)
        for i, ref in enumerate(look_refs):
            T[i].append(inv @ targets.point(ctx, ref))
    return A, E, T


def _strum(ctx, name, m, specs):
    """[perform.<cast>] strum: key the pick hand's IK goal (relative to the grip the pose stage solved) so that the pick tip
    strokes across the strings of the guitar-like prop the hand grips: strokes from the timeline's onsets or the beat grid
    (mkmmd.core.strum.plan), the pick tip's path over them (`path`), turned into goals that move the hand as a rigid body: the
    tip goes where the path says, part of the sideways travel (`share`) made by turning the hand about the wrist, about an axis
    along the strings. Returns {side: numbers}."""
    from ...core import fretting as FR
    from ...core import strum as SM
    specs = [specs] if isinstance(specs, dict) else list(specs)
    sides = {}
    for sp in specs:
        sides.setdefault(sp.get("hand", "R"), []).append(sp)
    out = {}
    f1 = ctx.start + ctx.settle
    for side, entries in sides.items():
        where = f"perform.{name}.strum ({side})"
        first = entries[0]
        if side not in m.ik:
            raise BuildError(f"{where}: the pick hand needs `[pose.{name}.hands.{side}] grip = \"<prop>:strum\"` (no arm IK on that side)")
        tgt = m.ik[side][0]
        prop = ctx.props.get(first.get("prop"))
        if prop is None:
            raise BuildError(f"{where}: give `prop` = the name of the guitar the hand grips (props: {sorted(ctx.props)})")
        if tgt.parent is not prop.root:
            raise BuildError(f"{where}: the hand's goal must ride the prop {prop.name!r} (a grip on its strum zone does by default)")
        entry = prop.use("grip", first.get("grip", "strum"))
        a, c, n = (np.asarray(entry.get(k), float) for k in ("along", "across", "normal"))
        a, c, n = a / np.linalg.norm(a), c / np.linalg.norm(c), n / np.linalg.norm(n)
        centre = np.asarray(entry["center"], float)
        neck = next((g for g in prop.card.get("use", {}).get("grip", []) if g.get("type") == "neck"), None)
        if "sigma" in first:
            sigma = float(first["sigma"])
        elif neck is not None:                                 # half the width of the six strings at the strum centre
            lo, hi = (FR.string_point(neck, s, float(centre @ a)) @ c for s in (6, 1))
            sigma = abs(float(hi - lo)) / 2.0
        else:
            sigma = 0.0225
        tl = ctx.timeline(first, where)
        strokes = sorted((s for e in entries for s in SM.plan(e, tl, ctx.duration)), key=lambda s: s.t)
        keep = {k: first[k] for k in SM.DEFAULTS if k in first}
        win = SM.window(strokes, **keep)
        if win is None:
            ctx.log("WARNING", f"{where}: no strokes (rhythm and window select no strike time)")
            out[side] = {"strokes": 0}
            continue
        fr = np.arange(max(f1 + 1, math.floor(ctx.frame(win[0]))), min(ctx.end, math.ceil(ctx.frame(win[1]))) + 1)
        P = SM.path(strokes, [ctx.time(f) for f in fr], sigma, **keep)
        act = tgt.animation_data.action
        loc0 = np.array([act.fcurves.find("location", index=i).evaluate(f1) for i in range(3)])
        q0 = Quaternion([act.fcurves.find("rotation_quaternion", index=i).evaluate(f1) for i in range(4)])
        R0 = np.array(q0.to_matrix())
        wrist = S.semantic_map(m.arm)[f"wrist.{side}"]
        pivot = loc0 - R0 @ np.array([0.0, m.arm.data.bones[wrist].length, 0.0])      # the wrist head: the goal acts on the tail
        r = centre - pivot
        r = r - (r @ a) * a
        kappa = float(np.cross(a, r) @ c)                      # sideways travel of the tip per radian about the wrist
        share = float(first.get("share", SM.DEFAULTS["share"])) if abs(kappa) > 0.03 else 0.0
        locs, rots = [], []
        for u, h in zip(P["u"], P["h"]):
            th = float(np.clip(share * u / kappa, -0.6, 0.6)) if share else 0.0
            Rt = np.array(Quaternion(Vector(a), th).to_matrix())
            tip = centre + u * c + h * n                       # where the path wants the pick tip
            move = tip - (pivot + Rt @ (centre - pivot))
            locs.append(tuple(pivot + Rt @ (loc0 - pivot) + move))
            rots.append(tuple(Matrix((Rt @ R0).tolist()).to_quaternion()))
        rots = K.continuous(rots)
        if float(np.dot(rots[0], np.array(q0))) < 0:
            rots = -rots
        mills = []
        if first.get("windmill"):
            locs, rots, mills = _windmill(ctx, m, side, prop, n, fr, locs, rots, strokes,
                                          [float(t) for t in first["windmill"]], float(first.get("windmill_dur", 0.5)), where)
        K.key_vec(tgt, "location", fr, locs, interp="LINEAR", replace=False)
        K.key_vec(tgt, "rotation_quaternion", fr, rots, interp="LINEAR", replace=False)
        down = sum(1 for s in strokes if s.dir > 0)
        out[side] = {"strokes": len(strokes), "down": down, "up": len(strokes) - down, "frames": [int(fr[0]), int(fr[-1])],
                     "sigma_mm": round(sigma * 1e3, 1), "turn_share": share, "first": [round(s.t, 3) for s in strokes[:6]],
                     "max_u_mm": round(float(np.abs(P["u"]).max()) * 1e3, 1)}
        if mills:
            out[side]["windmills"] = mills
        if first.get("kick"):
            out[side]["kick"] = _neck_kick(ctx, prop, n, [s.t for s in strokes if s.accent], float(first["kick"]))
        ctx.log("strum", name, side, json.dumps(out[side]))
    return out


def _windmill(ctx, m, side, prop, normal, fr, locs, rots, strokes, times, dur, where):
    """A windmill: over the `dur` seconds before a down stroke (the one nearest each time in `times`) the pick hand swings a
    full circle. Its goal (position and orientation as at that strike) turns once about an axis through the shoulder along
    the guitar's face normal, coming down onto the strings at the strike, which then plays as planned; the arm IK follows.
    The circle eases in from wherever the strumming has the hand. `locs`, `rots` are the goals of the frames `fr` in the
    prop's frame. Returns (locs, rots, [strike times])."""
    sc = bpy.context.scene
    keep = sc.frame_current
    sc.frame_set(int(ctx.start + ctx.settle))
    dg = bpy.context.evaluated_depsgraph_get()
    ae = m.arm.evaluated_get(dg)
    shoulder_w = ae.matrix_world @ ae.pose.bones[S.semantic_map(m.arm)[f"arm.{side}"]].head
    Mp = prop.root.evaluated_get(dg).matrix_world.copy()
    sc.frame_set(keep)
    centre = np.array(Mp.inverted() @ shoulder_w)
    up_l = np.array(Mp.to_3x3().normalized().inverted() @ Vector((0.0, 0.0, 1.0)))
    axis = Vector(normal).normalized()
    L, Q = np.array(locs, float), np.array(rots, float)
    ts = np.array([ctx.time(f) for f in fr])
    done = []
    for tw in times:
        downs = [s for s in strokes if s.dir > 0 and abs(s.t - tw) <= 0.12]
        if not downs:
            raise BuildError(f"{where}: windmill {tw}: no down stroke within 0.12 s of it (the strike it comes down on)")
        t_s = min(downs, key=lambda s: abs(s.t - tw)).t
        j = int(np.argmin(np.abs(ts - t_s)))
        p_s, q_s = L[j].copy(), Quaternion(Q[j])

        def turned(th):
            R = Quaternion(axis, th)
            return centre + np.array(R @ Vector(p_s - centre)), R @ q_s

        sign = 1.0 if float((turned(-0.25)[0] - p_s) @ up_l) > 0 else -1.0     # the hand comes down onto the strings
        t0 = t_s - dur
        for k in np.where((ts >= t0) & (ts <= ts[j]))[0]:
            r = min((ts[k] - t0) / dur, 1.0)
            p, q = turned(-sign * 2.0 * math.pi * (1.0 - r ** 1.6))      # slow out of the strumming, fast down onto the strike
            w = min(1.0, r / 0.15)
            w = w * w * (3.0 - 2.0 * w)
            L[k] = L[k] * (1.0 - w) + p * w
            Q[k] = np.array(Quaternion(Q[k]).slerp(q, w))
        done.append(round(float(t_s), 3))
    return [tuple(v) for v in L], K.continuous([tuple(q) for q in Q]), done


def _neck_kick(ctx, prop, normal, times, deg):
    """The guitar's neck kicks up on the accented strokes: the prop's root turns `deg` about the face normal (through its
    origin) on each, easing back. The hands ride the prop, so they go with it."""
    root = prop.root
    if root.rotation_mode != "QUATERNION":
        q = root.matrix_basis.to_quaternion()
        root.rotation_mode = "QUATERNION"
        root.rotation_quaternion = q
    q_base = root.rotation_quaternion.copy()
    nl = Vector(normal).normalized()
    Rw = root.matrix_world.to_3x3().normalized()
    rise = (Rw @ (Quaternion(nl, 0.1) @ Vector((0.0, 0.0, 1.0)))).z - (Rw @ Vector((0.0, 0.0, 1.0))).z
    sign = 1.0 if rise > 0 else -1.0                   # the neck (+z of the prop) goes up
    ts = np.array([ctx.time(f) for f in ctx.frames])
    pulse = PF.beat_pulse(ts, times, 0.04, 0.2)
    rots = [tuple(q_base @ Quaternion(nl, sign * math.radians(deg) * float(p))) for p in pulse]
    K.key_vec(root, "rotation_quaternion", ctx.frames, K.continuous(rots), interp="LINEAR")
    return {"accents": len(times), "deg": deg}


def _beat_marks(ctx, b, where):
    """(beat times, per-beat weights) of a table with `beats = [...]` (weights 1) or a `timeline` (default
    audio/timeline.json): its beats, the downbeats weighted `downbeat_accent` (1.6)."""
    if "beats" in b:
        return [float(x) for x in b["beats"]], None
    beats, downbeats = TL.beats(ctx.timeline(b, where))
    downs = set(round(x, 3) for x in downbeats)
    return beats, [float(b.get("downbeat_accent", 1.6)) if round(x, 3) in downs else 1.0 for x in beats]


def _window(ts, b, ease=0.15):
    """1 inside a table's `from`..`to` (clip seconds; open when missing), easing in and out over `ease` seconds."""
    lo, hi = float(b.get("from", -1e9)), float(b.get("to", 1e9))
    return PF.smooth((ts - lo) / ease + 1.0) * PF.smooth((hi - ts) / ease + 1.0)


def _body(ctx, name, m, spec, ts, frames, settle):
    """bounce, crouch, rise, kick and the rock's hips: the hips (the `center` bone) and the feet (the leg IK bones) move on
    top of the offsets the pose stage gave them (eased in over the settle like them). `bounce` dips the hips on the beats
    while the feet stay planted (the knees bend: a punk's pump); `crouch = [[t, metres], ...]` lowers them so (a landing, a
    squat); `rock.hips` shifts them toward the model's left with the rock's own swing (the weight goes from foot to foot);
    `rise = [[t, metres], ...]` lifts the whole body, feet too (she floats); `kick` flicks one foot up and back on the
    beats, `hold` of the way lifted between them. Returns report numbers."""
    bn, rs, kk, cr = spec.get("bounce"), spec.get("rise"), spec.get("kick"), spec.get("crouch")
    rk = spec.get("rock") or {}
    hips = float(rk.get("hips", 0.0))
    if not (bn or rs or kk or cr or hips):
        return {}
    arm = m.arm
    smap = S.semantic_map(arm)
    W = arm.matrix_world.to_3x3().normalized()
    up_w, back_w, left_w = W @ Vector((0.0, 0.0, 1.0)), W @ Vector((0.0, 1.0, 0.0)), W @ Vector((1.0, 0.0, 0.0))
    base = m.base or {}
    centre0 = Vector(base.get("center", (0.0, 0.0, 0.0)))
    feet0 = {s: Vector(v) for s, v in (base.get("feet") or {}).items()}
    lift, dip, side, info = np.zeros(len(ts)), np.zeros(len(ts)), np.zeros(len(ts)), {}
    if bn:
        beats, acc = _beat_marks(ctx, bn, f"perform.{name}.bounce")
        dip = float(bn.get("depth", 0.03)) * _window(ts, bn) * PF.beat_pulse(
            ts, beats, float(bn.get("attack", 0.05)), float(bn.get("decay", 0.16)), acc)
        info["bounce_mm"] = round(float(dip.max()) * 1000, 1)
    if cr:
        down = PF.eased_keys(ts, cr)
        dip = dip + down
        info["crouch_mm"] = round(float(np.abs(down).max()) * 1000, 1)
    if hips:
        side = hips * np.sin(2.0 * np.pi * ts / float(rk.get("period", 4.0)) + float(rk.get("phase", 0.0)))
        info["hips_mm"] = round(abs(hips) * 1000, 1)
    if rs:
        lift = PF.eased_keys(ts, rs)
        info["rise_mm"] = round(float(np.abs(lift).max()) * 1000, 1)
    kick_side, kick = None, None
    if kk:
        kick_side = kk.get("foot", "L")
        if kick_side not in ("L", "R"):
            raise BuildError(f"perform.{name}.kick: foot = \"L\" or \"R\", not {kick_side!r}")
        beats, acc = _beat_marks(ctx, kk, f"perform.{name}.kick")
        hold = float(kk.get("hold", 0.0))
        pulse = np.minimum(PF.beat_pulse(ts, beats, float(kk.get("attack", 0.06)), float(kk.get("decay", 0.18)), acc), 1.5)
        kick = (hold + (1.0 - hold) * pulse) * _window(ts, kk)
        reach = up_w * float(kk.get("height", 0.12)) + back_w * float(kk.get("back", 0.08))
        info["kick"] = {"foot": kick_side, "peak_mm": round(float(kick.max()) * reach.length * 1000, 1)}
    centre = [tuple(centre0 * float(s) + up_w * float(lift[i] - dip[i]) + left_w * float(side[i]))
              for i, s in enumerate(settle)]
    K.key_bone_locs(arm, smap["center"], frames, centre, interp="LINEAR")
    if rs or kk:
        for side in ("L", "R"):
            ik = smap.get(f"leg_ik.{side}")
            if not ik:
                continue
            f0 = feet0.get(side, Vector())
            vals = []
            for i, s in enumerate(settle):
                v = f0 * float(s) + up_w * float(lift[i])
                if side == kick_side:
                    v = v + reach * float(kick[i])
                vals.append(tuple(v))
            K.key_bone_locs(arm, ik, frames, vals, interp="LINEAR")
    return info


DRUM_FINGERS = ("little", "ring", "middle", "index")     # the order a tap ripples in
DRUM_MID = 0.35                                           # share of the lift the middle joint takes


def _drum(ctx, name, m, d, ts, frames):
    """drum: the fingers of one hand tap on the beats (a driver drumming on the wheel). Each finger lifts at its base
    joint by `deg` (the middle joint by DRUM_MID of it) about its own flexion axis, from where its keys hold it (a
    grip's solved curl), and falls back onto the beat (core.perform.tap); the little finger leads the index by `roll`
    seconds per finger. Keyed every frame from the first lift to `to` (LINEAR), the hand's other keys kept."""
    from ...core import bonemap
    from .pose import _world_head, palm_normal
    where = f"perform.{name}.drum"
    side = d.get("hand", "R")
    if side not in ("L", "R"):
        raise BuildError(f"{where}: hand = \"L\" or \"R\", not {side!r}")
    fingers = list(d.get("fingers", DRUM_FINGERS))
    bad = sorted(set(fingers) - set(DRUM_FINGERS))
    if bad:
        raise BuildError(f"{where}: fingers {bad}: unknown (have {list(DRUM_FINGERS)})")
    fingers = [f for f in DRUM_FINGERS if f in fingers]
    beats, acc = _beat_marks(ctx, d, where)
    lo, hi = float(d.get("from", ts[0])), float(d.get("to", ts[-1]))
    keep = [i for i, b in enumerate(beats) if lo <= b <= hi]
    if not keep:
        ctx.log("WARNING", f"{where}: no beat between {lo} and {hi} s: nothing is drummed")
        return {}
    beats, acc = [beats[i] for i in keep], (None if acc is None else [acc[i] for i in keep])
    lift, deg, roll = float(d.get("lift", 0.16)), float(d.get("deg", 25.0)), float(d.get("roll", 0.02))
    lead = lift + roll * (len(fingers) - 1)
    sel = np.where((ts >= beats[0] - lead - 0.05) & (ts <= max(hi, beats[-1]) + 0.05))[0]
    fr, tsel = [frames[i] for i in sel], ts[sel]
    arm = m.arm
    smap = S.semantic_map(arm)
    palm = palm_normal(arm, smap, side)
    act = arm.animation_data.action if arm.animation_data else None
    for k, f in enumerate(fingers):
        bones = [smap[f"{s}.{side}"] for s in bonemap.FINGERS[f][:3] if f"{s}.{side}" in smap]
        if len(bones) < 2:
            continue
        axis = (_world_head(arm, bones[1]) - _world_head(arm, bones[0])).normalized().cross(palm).normalized()
        e = PF.tap(tsel, [b - roll * (len(fingers) - 1 - k) for b in beats], lift, acc)
        for j, b in enumerate(bones[:2]):
            B = K.rest_rot(arm, b)
            path = f'pose.bones["{b}"].rotation_quaternion'
            fcs = [act.fcurves.find(path, index=c) if act else None for c in range(4)]
            rest = tuple(arm.pose.bones[b].rotation_quaternion)
            share = 1.0 if j == 0 else DRUM_MID
            quats = []
            for i, fno in enumerate(fr):
                base = Quaternion([fcs[c].evaluate(fno) if fcs[c] else rest[c] for c in range(4)]).normalized()
                ext = (B.inverted() @ Quaternion(axis, -math.radians(deg * share * float(e[i]))).to_matrix()
                       @ B).to_quaternion()
                quats.append(tuple(ext @ base))
            K.key_bone_quats(arm, b, fr, quats, interp="LINEAR", replace=False)
    return {"hand": side, "fingers": fingers, "taps": len(beats), "deg": deg,
            "frames": [int(fr[0]), int(fr[-1])]}



def run(ctx):
    ctx.check_tables("perform")
    out = {}
    frames = ctx.frames
    ts = np.array([ctx.time(f) for f in frames])
    for name, m in ctx.cast.items():
        spec = ctx.section("perform", name)
        if not spec:
            if name in ctx.section("perform"):
                ctx.log("WARNING", f"perform.{name}: the table is empty, so nothing is performed (give it keys, or remove it)")
            continue
        arm = m.arm
        smap = S.semantic_map(arm)
        base = m.base.get("spine") or {}
        lat, back, up = Vector((1, 0, 0)), Vector((0, 1, 0)), Vector((0, 0, 1))     # armature axes (faces -Y)
        fwd = -back
        # ---- targets: idle look + gaze events (eye height, 3 m ahead by default)
        events = sorted(spec.get("gaze", []), key=lambda ev: ev["t"])
        glances = sorted(spec.get("glance", []), key=lambda g: g["t"])
        refs = [spec.get("look")] + [ev["at"] for ev in events] + [g["at"] for g in glances]
        idle_ahead = refs[0] is None
        refs = [r if r is not None else [0, 0, 0] for r in refs]
        A, E, T = _sample_pass(ctx, m, refs, frames)
        if idle_ahead:
            T[0] = [e + fwd * 3.0 for e in E]
        gz = PF.GazeEvents([(ev["t"], ev["t"] + ev.get("hold", 1.0), ev["t"] + ev.get("hold", 1.0) + ev.get("back", 0.4),
                             i + 1, ev.get("rise", 0.3)) for i, ev in enumerate(events)])
        W = gz.weights(ts) if events else np.zeros((0, len(ts)))
        W_eye = gz.weights(ts + 0.07) if events else W           # the eyes lead the head
        gl = [(g["t"], g.get("dur", 0.8), g.get("rise", 0.12), g.get("fall", 0.25)) for g in glances]
        G = PF.glance_weights(ts, gl) if gl else np.zeros((0, len(ts)))
        G_eye = PF.glance_weights(ts + 0.05, gl) if gl else G    # the eyes lead
        Tn = np.array([[tuple(v) for v in T[i]] for i in range(len(refs))])     # (targets, F, 3)
        En = np.array([tuple(e) for e in E])
        Dirs = Tn - En[None]                                     # unit directions from the eyes: targets at any
        Dirs /= np.maximum(np.linalg.norm(Dirs, axis=2, keepdims=True), 1e-9)   # distance blend evenly

        def blend(Wm, Wg=None):
            d = Dirs[0] * (1.0 - Wm.sum(0))[:, None]
            for i in range(Wm.shape[0]):
                d = d + Dirs[i + 1] * Wm[i][:, None]
            if Wg is not None and Wg.shape[0]:                   # a glance takes the eyes where no gaze event has them
                free = 1.0 - Wm.sum(0)
                for j in range(Wg.shape[0]):
                    w = (Wg[j] * free)[:, None]
                    d = d * (1.0 - w) + Dirs[len(events) + 1 + j] * w
            d /= np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-9)
            return En + d                                        # a point 1 m along the blended direction

        look_eye = PF.lowpass(blend(W_eye, G_eye), 1.5)
        look_head = PF.lowpass(blend(W), 3.0) + PF.drift(len(ts), seed=zlib.crc32(name.encode()) % 1000) * 0.3
        w_away = W.sum(0) if len(W) else np.zeros(len(ts))
        lift = np.zeros(len(ts))                                  # degrees the head lifts during glances
        if glances:
            lift = PF.lowpass((G * np.array([g.get("pitch", 2.75) for g in glances])[:, None]).sum(0)[:, None], 3.0)[:, 0]
        # ---- body life
        br = spec.get("breath", {})
        sw = spec.get("sway", {})
        nd = spec.get("nod", {})
        breath = PF.breathing(ts, br.get("per_min", 16.5), br.get("deg", 0.6))
        sway = PF.sway(ts, sw.get("period", 2.5), sw.get("deg", 0.37))
        nod = math.radians(nd.get("deg", 0.48)) * np.sin(2 * math.pi * ts / nd.get("period", 2.3)) * (1 - w_away)
        bob = np.zeros(len(ts))
        if spec.get("bob"):
            b = spec["bob"]
            beats, accent = _beat_marks(ctx, b, f"perform.{name}.bob")
            bob = PF.beat_bob(ts, beats, b.get("deg", 1.5), accent=accent)
        startle = PF.startles(ts, spec.get("startle", []))
        lean_x = np.radians(PF.eased_keys(ts, spec.get("lean", [])))
        turn_x = np.radians(PF.eased_keys(ts, spec.get("turn", [])))
        tilt_x = np.radians(PF.eased_keys(ts, spec.get("tilt", [])))
        head_tilt_x = np.radians(PF.eased_keys(ts, spec.get("head_tilt", [])))
        head_pitch_x, head_yaw_x = np.zeros(len(ts)), np.zeros(len(ts))
        for key, table in (("rock", spec.get("rock") or {}), ("head_rock", spec.get("head_rock") or {})):
            if not table:
                continue
            swing = PF.rock(ts, table.get("period", 4.0), table.get("deg", 0.0), table.get("phase", 0.0))
            axis = table.get("axis", "side")
            if axis not in ("side", "front", "turn"):
                raise BuildError(f"perform.{name}.{key}: axis = \"side\", \"front\" or \"turn\", not {axis!r}")
            if key == "rock":
                if axis == "side":
                    tilt_x = tilt_x + swing
                elif axis == "front":
                    lean_x = lean_x + swing
                else:
                    turn_x = turn_x + swing
            elif axis == "side":
                head_tilt_x = head_tilt_x + swing
            elif axis == "front":
                head_pitch_x = head_pitch_x + swing
            else:
                head_yaw_x = head_yaw_x + swing
        k_head = float(spec.get("head_share", 0.7))
        k_neck = float(spec.get("neck_share", 0.35))
        head_limits = spec.get("head_limits")
        eye_max = math.radians(float(spec.get("eye_max", 24)))
        settle = PF.smooth((frames - ctx.start) / max(ctx.settle, 1))
        has2 = "upper_body2" in smap
        keys = {k: [] for k in SPINE + ("eye",) if k in smap or k == "eye"}
        for i, f in enumerate(frames):
            s = float(settle[i])
            q1b = Quaternion().slerp(base.get("upper_body", Quaternion()), s)
            q2b = Quaternion().slerp(base.get("upper_body2", Quaternion()), s) if has2 else Quaternion()
            q3b = Quaternion().slerp(base.get("neck", Quaternion()), s)
            q4b = Quaternion().slerp(base.get("head", Quaternion()), s)
            q1 = (Quaternion(up, turn_x[i]) @ Quaternion(back, tilt_x[i]) @ Quaternion(lat, lean_x[i] - startle[i])
                  @ Quaternion(up, sway[i]) @ q1b)
            q2 = Quaternion(lat, breath[i]) @ q2b
            if not has2:                                   # breathing goes to the only chest bone
                q1, q2 = q1 @ q2, Quaternion()
            D2 = q1 @ q2
            Dn_base = D2 @ q3b
            Dh_base = Dn_base @ q4b
            eye_pos = E[i]
            look = (Vector(look_head[i]) - eye_pos).normalized()
            aim = PF.clamp_look(tuple((Dh_base @ fwd).normalized()), tuple((Dh_base @ up).normalized()), tuple(look),
                                k_head, head_limits)
            R_full = (Dh_base @ fwd).rotation_difference(look if aim is None else Vector(aim))
            pitch = Quaternion(lat, nod[i] + bob[i] + head_pitch_x[i] - math.radians(lift[i]))
            Dn = Quaternion().slerp(R_full, k_head * k_neck) @ Dn_base
            Dh = (Quaternion(up, head_yaw_x[i]) @ Quaternion(back, head_tilt_x[i]) @ pitch
                  @ Quaternion().slerp(R_full, k_head) @ Dh_base)
            R_eye = (Dh @ fwd).rotation_difference((Vector(look_eye[i]) - eye_pos).normalized())
            if R_eye.angle > eye_max:
                R_eye = Quaternion().slerp(R_eye, eye_max / R_eye.angle)
            qs = {"upper_body": q1, "upper_body2": q2, "neck": D2.inverted() @ Dn,
                  "head": Dn.inverted() @ Dh, "eye": Dh.inverted() @ (R_eye @ Dh)}
            for k in keys:
                keys[k].append(qs[k])
        for k, qs in keys.items():
            bones = [smap[f"eye.{s}"] for s in ("L", "R") if f"eye.{s}" in smap] if k == "eye" else [smap[k]]
            for b in bones:
                R = arm.data.bones[b].matrix_local.to_3x3().normalized()     # armature-space rest rotation
                Ri = R.inverted()
                loc = [tuple((Ri @ q.to_matrix() @ R).to_quaternion()) for q in qs]
                K.key_bone_quats(arm, b, frames, loc, interp="LINEAR")
        info = {"gaze_events": len(events), "glances": len(glances), "frames": len(frames)}
        info.update(_body(ctx, name, m, spec, ts, frames, settle))
        # ---- face: blinks + lids
        bl = spec.get("blink", {})
        blink_m = _morph(m, "blink")
        blinks = PF.blink_schedule(ts[0], ts[-1], bl.get("per_min", 15.0), seed=bl.get("seed", 0),
                                   extra=[tuple(x) for x in bl.get("extra", [])] +
                                   [(ev["t"] + 0.03, PF.BLINK_S) for ev in events if ev.get("blink", True)] +
                                   [(g["t"] + 0.02, PF.BLINK_S) for g in glances if g.get("blink", True)])
        lids = float(spec.get("lids", 0.0)) * (1 - w_away) * (1 - np.minimum(G.sum(0), 1.0))
        vals = np.maximum(PF.blink_curve(ts, blinks), lids * settle)
        info["blinks"] = len(blinks)
        holders = S.morph_holders(m.arm)                  # the meshes, or the bound sliders' placeholder
        n_blink = K.key_morph(holders, blink_m, frames, vals, interp="LINEAR")
        info["blink_morph"] = blink_m if n_blink else None
        if not n_blink:
            ctx.log(f"perform {name}: no blink morph ({blink_m}) on the model")
        # ---- expressions
        for ex in spec.get("expressions", []):
            mn = _morph(m, ex["morph"])
            pts = sorted(ex["keys"])
            n = K.key_morph(holders, mn, [ctx.frame(t) for t, _ in pts], [v for _, v in pts])
            if not n:
                raise BuildError(f"perform.{name}: morph {ex['morph']!r} ({mn}) is not on the model")
        # ---- lip sync
        if spec.get("sing"):
            sg = spec["sing"]
            tl = ctx.timeline(sg, f"perform.{name}.sing")
            try:
                kf = LS.keyframes(tl, lines=sg.get("lines"), words=sg.get("words"), mouth=sg.get("mouth", 0.8),
                                  fps=ctx.fps, voice=sg.get("voice", "en-gb"), offset=sg.get("lead", -0.03))
            except ValueError as e:
                raise BuildError(f"perform.{name}.sing: {e}") from None
            done = []
            for v, pts in kf.items():
                mn = _morph(m, v)
                if pts and K.key_morph(holders, mn, [ctx.frame(t) for t, _ in pts], [x for _, x in pts]):
                    done.append(mn)
            info["sing_morphs"] = len(done)
            info["sing_keys"] = sum(len(p) for p in kf.values())
        # ---- twitches (quick flicks of ears, tails...): bone-local rotations about an armature-space axis
        for tw in spec.get("twitch", []):
            if tw.get("family"):
                bones = [pb.name for pb in arm.pose.bones if FAM.classify(pb.name, S.name_j(pb)) == tw["family"]
                         and (pb.parent is None or FAM.classify(pb.parent.name, S.name_j(pb.parent)) != tw["family"])]
            else:
                bones = [S.resolve_bone(arm, b) for b in tw["bones"]]
            t0, dur, deg = float(tw["t"]), float(tw.get("dur", 0.22)), float(tw.get("deg", 14))
            axis = Vector(tw.get("axis", (1, 0, 0))).normalized()
            fr = [ctx.frame(t0 - 0.02), ctx.frame(t0 + dur * 0.25), ctx.frame(t0 + dur * 0.55), ctx.frame(t0 + dur)]
            amp = [0.0, 1.0, -0.25, 0.0]
            for b in bones:
                R = arm.data.bones[b].matrix_local.to_3x3().normalized()
                ax_l = (R.inverted() @ axis).normalized()
                K.key_bone_quats(arm, b, fr, [tuple(Quaternion(ax_l, math.radians(deg) * a)) for a in amp],
                                 interp="BEZIER", replace=False)
        if spec.get("strum"):
            info["strum"] = _strum(ctx, name, m, spec["strum"])
        if spec.get("drum"):
            info["drum"] = _drum(ctx, name, m, spec["drum"], ts, frames)
        out[name] = info
        ctx.log("perform", name, json.dumps(info, ensure_ascii=False))
    return out
