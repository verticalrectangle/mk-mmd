"""perform: a character's life on top of its pose: gaze (eyes lead, head and neck follow), breathing, sway, idle nod,
beat bob, startles, blinks and lids, expressions, lip sync, twitches.

[perform.<cast>] keys (times in clip seconds; targets as in mkmmd.blender.build.targets)
  look = target                      idle gaze target (default: ahead at eye height)
  gaze = [{t, at, hold = 1.0, back = 0.4, rise = 0.3}]   look-at events
  glance = [{t, at, dur = 0.8, pitch = 2.75, rise = 0.12, fall = 0.25, blink = true}]   eye-only glances (the eyes
                                     lead to `at`, the head lifts `pitch` deg, the lids open, a blink follows); the
                                     head and neck do not turn
  head_share = 0.7                   share of a gaze turn taken by head + neck (the eyes do the rest, up to eye_max)
  neck_share = 0.35                  share of the head's turn carried by the neck
  eye_max = 24 (deg)
  breath = {per_min = 16.5, deg = 0.6}, sway = {deg = 0.37, period = 2.5}, nod = {deg = 0.48, period = 2.3}
  bob = {deg = 1.5, timeline = "audio/timeline.json" | beats = [t...], downbeat_accent = 1.6}
  startle = [t...]
  blink = {per_min = 15, seed = 0, extra = [[t, dur], ...]}; lids = 0.0 (base lowering 0..1)
  sing = {timeline = "audio/timeline.json", lines = [a, b], mouth = 0.8, lead = -0.03, voice = "en-gb"}
  expressions = [{morph = "smile_eyes" (semantic or the model's own name), keys = [[t, value], ...]}]
  twitch = [{bones = ["ear_root.L", ...] | family = "ears", t = 3.2, deg = 14, axis = [1, 0, 0], dur = 0.22}]
Everything is composed in the armature's frame (correct for characters riding vehicles), keyed per frame (LINEAR)."""
import json
import math
import zlib

import bpy
import numpy as np
from mathutils import Quaternion, Vector

from ...core import families as FAM
from ...core import lipsync as LS
from ...core import perform as PF
from ...core import timeline as TL
from .. import keys as K
from .. import scene as S
from . import BuildError, targets

SPINE = ("upper_body", "upper_body2", "neck", "head")


def _timeline(ctx, path):
    with open(ctx.path(path), encoding="utf-8") as fh:
        return json.load(fh)


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


def run(ctx):
    out = {}
    frames = ctx.frames
    ts = np.array([ctx.time(f) for f in frames])
    for name, m in ctx.cast.items():
        spec = ctx.section("perform", name)
        if not spec:
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
            if "beats" in b:
                beats, accent = b["beats"], None
            else:
                tl = _timeline(ctx, b["timeline"])
                beats, downbeats = TL.beats(tl)
                downs = set(round(x, 3) for x in downbeats)
                accent = [b.get("downbeat_accent", 1.6) if round(x, 3) in downs else 1.0 for x in beats]
            bob = PF.beat_bob(ts, beats, b.get("deg", 1.5), accent=accent)
        startle = PF.startles(ts, spec.get("startle", []))
        k_head = float(spec.get("head_share", 0.7))
        k_neck = float(spec.get("neck_share", 0.35))
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
            q1 = Quaternion(lat, -startle[i]) @ Quaternion(up, sway[i]) @ q1b
            q2 = Quaternion(lat, breath[i]) @ q2b
            if not has2:                                   # breathing goes to the only chest bone
                q1, q2 = q1 @ q2, Quaternion()
            D2 = q1 @ q2
            Dn_base = D2 @ q3b
            Dh_base = Dn_base @ q4b
            eye_pos = E[i]
            R_full = (Dh_base @ fwd).rotation_difference((Vector(look_head[i]) - eye_pos).normalized())
            pitch = Quaternion(lat, nod[i] + bob[i] - math.radians(lift[i]))
            Dn = Quaternion().slerp(R_full, k_head * k_neck) @ Dn_base
            Dh = pitch @ Quaternion().slerp(R_full, k_head) @ Dh_base
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
        n_blink = K.key_morph(m.meshes, blink_m, frames, vals, interp="LINEAR")
        info["blink_morph"] = blink_m if n_blink else None
        if not n_blink:
            ctx.log(f"perform {name}: no blink morph ({blink_m}) on the model")
        # ---- expressions
        for ex in spec.get("expressions", []):
            mn = _morph(m, ex["morph"])
            pts = sorted(ex["keys"])
            n = K.key_morph(m.meshes, mn, [ctx.frame(t) for t, _ in pts], [v for _, v in pts])
            if not n:
                raise BuildError(f"perform.{name}: morph {ex['morph']!r} ({mn}) is not on the model")
        # ---- lip sync
        if spec.get("sing"):
            sg = spec["sing"]
            tl = _timeline(ctx, sg["timeline"])
            kf = LS.keyframes(tl, lines=sg.get("lines"), mouth=sg.get("mouth", 0.8), fps=ctx.fps,
                              voice=sg.get("voice", "en-gb"), offset=sg.get("lead", -0.03))
            done = []
            for v, pts in kf.items():
                mn = _morph(m, v)
                if pts and K.key_morph(m.meshes, mn, [ctx.frame(t) for t, _ in pts], [x for _, x in pts]):
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
        out[name] = info
        ctx.log("perform", name, json.dumps(info, ensure_ascii=False))
    return out
