"""marionette: a cast member on the threads of a `library:marionette` prop (docs/design.md: Marionettes;
mkmmd.core.marionette has the maths).

The perform stage calls `prepare` before it samples the gaze (so she can look up at the cross) and `hands` after the
strum: `prepare` reads her settled pose, puts the cross over her head and the leg bar in front of her and keys them,
ties the threads' ends to her bones (the `<prop>_end_<part>` empties, bone-parented where the thread meets her) and
returns her channels for perform to add to its own; `hands` turns the worn guitar with the hand bar, swings a hand whose
thread is cut on its arm, and keys the hand bar to follow the picking hand (the thread does the strumming).

The build's `marionette` stage (`run`, after `stepped`) draws the threads from where the anchors on the bars and the
ends on her are on every frame, once everything is keyed: so a stepped member's threads step with her, and the bars are
held on the same steps. Each thread's own length is its span at `from`: straight while it is pulled, sagging when she
comes closer, dangling under the cross as it falls in, parted at a cut (the piece on the cross yanked up out of sight,
the piece on her falling limp over her and off onto the floor)."""
import json
import math

import bpy
import numpy as np
from mathutils import Matrix, Quaternion, Vector

from ...core import marionette as MR
from ...core import perform as PF
from ...core import stepped as STP
from ...core import timeline as TL
from .. import keys as K
from .. import scope as SCOPE
from .. import scene as S
from . import BuildError
from . import stepped as STEPPED

ENDS = {"head": ("head", (0.0, 0.0, 0.175)), "back": ("upper_body2", None), "wrist_L": ("wrist.L", (0.0, 0.0, 0.02)),
        "wrist_R": ("wrist.R", (0.0, 0.0, 0.02)), "knee_L": ("knee.L", (0.0, -0.045, 0.0)),
        "knee_R": ("knee.R", (0.0, -0.045, 0.0))}       # where each thread meets her: a bone and an offset (her frame)
BACK = (0.0, 0.075, -0.045)     # the back's end: from the middle of her shoulders, behind and a little lower
SIDE = {"head": 1.0, "back": 1.0, "wrist_L": 1.0, "wrist_R": -1.0, "knee_L": 1.0, "knee_R": -1.0}


def _frame(m):
    """Her frame: the cast root's place and turn (x her left, y behind her, z up), no scale."""
    loc, rot, _ = m.root.matrix_world.decompose()
    return Matrix.Translation(loc) @ rot.to_matrix().to_4x4(), rot


def _spec(ctx, name):
    v = ctx.section("perform", name).get("marionette")
    return v if isinstance(v, dict) and v else None


def plan(ctx, name):
    """(P, prop, card numbers) of a member's marionette table; None without one."""
    spec = _spec(ctx, name)
    if spec is None:
        return None
    where = f"perform.{name}.marionette"
    beats = []
    try:
        beats = TL.beats(ctx.timeline(spec, where))[0]
    except BuildError:
        if any(isinstance(spec.get(k), dict) for k in ("rock", "tilt", "turn", "sway", "knees")):
            raise
    try:
        P = MR.normalize(spec, beats)
    except MR.MarionetteError as e:
        raise BuildError(f"perform.{name}.{e}" if str(e).startswith("marionette") else f"{where}: {e}") from None
    prop = ctx.props.get(P["prop"])
    if prop is None or "marionette" not in prop.card:
        raise BuildError(f"{where}: prop = {P['prop']!r} is no library:marionette prop (props: {sorted(ctx.props)})")
    return P, prop, prop.card["marionette"]


def _key_hidden(ob, frames, hidden):
    K.set_fcurve(ob, "hide_render", 0, frames, np.asarray(hidden, float), interp="CONSTANT")


def prepare(ctx, name, m, ts, frames):
    """Key the cross and the leg bar over her, tie the threads to her and return her channels (mkmmd.core.marionette.
    body_plan, her frame = the armature's axes) with what `hands` needs; None without a `marionette` table."""
    got = plan(ctx, name)
    if got is None:
        return None
    P, prop, card = got
    where = f"perform.{name}.marionette"
    sc = bpy.context.scene
    keep = sc.frame_current
    sc.frame_set(int(ctx.start + ctx.settle))
    arm = m.arm
    smap = S.semantic_map(arm)
    need = [b for b, _ in ENDS.values()] + ["arm.L", "arm.R", "leg.L", "leg.R", "ankle.L", "ankle.R"]
    missing = [b for b in need if b not in smap]
    if missing:
        raise BuildError(f"{where}: the model has no {', '.join(missing)} bone (the threads are tied there)")
    F, Frot = _frame(m)
    Fi = F.inverted()

    def her(b):
        return Fi @ (arm.matrix_world @ arm.pose.bones[smap[b]].head)

    ends = {}
    for part, (bone, off) in ENDS.items():
        p = (her("arm.L") + her("arm.R")) / 2 + Vector(BACK) if off is None else her(bone) + Vector(off)
        ends[part] = (smap[bone], p)
    a_head = Vector(card["anchors"]["head"]["point"])
    top = ends["head"][1]
    home = Vector((top.x - a_head.x, top.y - a_head.y, top.z + P["height"] - a_head.z))
    leg_home = home + Vector(card["leg_bar"]["home"])
    legs = {}
    for s in ("L", "R"):
        hip, knee, ankle = (arm.matrix_world @ arm.data.bones[smap[f"{b}.{s}"]].head_local for b in ("leg", "knee", "ankle"))
        toe = smap.get(f"toe.{s}") or smap.get(f"toe_ik.{s}")
        foot = (arm.matrix_world @ arm.data.bones[toe].tail_local - ankle).length if toe else 0.14
        legs[s] = {"thigh": (knee - hip).length, "shin": (ankle - knee).length, "foot": max(foot, 0.08)}
    cross = MR.cross_plan(ts, P, float(card["leg_bar"]["half"]))
    body = MR.body_plan(ts, P, cross, legs)
    # the bars, keyed in the prop root's frame
    bars = {k: bpy.data.objects[v] for k, v in card["bars"].items()}
    Ri = prop.root.matrix_world.inverted()
    Rrot_i = prop.root.matrix_world.to_quaternion().inverted()
    locs, rots, llocs, lrots = [], [], [], []
    for i in range(len(ts)):
        q = Quaternion((0, 0, 1), float(cross["turn"][i])) @ Quaternion((0, 1, 0), float(cross["tilt"][i])) \
            @ Quaternion((1, 0, 0), float(cross["rock"][i]))
        locs.append(tuple(Ri @ (F @ (home + Vector(cross["pos"][i])))))
        rots.append(tuple(Rrot_i @ Frot @ q))
        ql = Quaternion((0, 0, 1), 0.5 * float(cross["turn"][i])) @ Quaternion((0, 1, 0), -float(cross["leg_roll"][i]))
        llocs.append(tuple(Ri @ (F @ (leg_home + Vector(cross["leg_pos"][i])))))
        lrots.append(tuple(Rrot_i @ Frot @ ql))
    for ob, L, Q in ((bars["cross"], locs, rots), (bars["leg"], llocs, lrots)):
        ob.rotation_mode = "QUATERNION"
        K.key_vec(ob, "location", frames, L, interp="LINEAR")
        K.key_vec(ob, "rotation_quaternion", frames, K.continuous(Q), interp="LINEAR")
    for ob in bars.values():
        _key_hidden(ob, frames, ~cross["shown"])
    # the threads' ends, tied to her bones where they meet her at the settled pose
    for part, (bone, p) in ends.items():
        e = bpy.data.objects[f"{prop.name}_end_{part}"]
        e.parent = arm
        e.parent_type = "BONE"
        e.parent_bone = bone
        e.matrix_parent_inverse = Matrix.Identity(4)
        bpy.context.view_layer.update()
        e.matrix_world = Matrix.Translation(F @ p)
    sc.frame_set(keep)
    body["report"]["home"] = [round(v, 3) for v in home]
    return {"P": P, "prop": prop, "card": card, "body": body, "cross": cross, "F": F, "Frot": Frot,
            "shoulders": {s: her(f"arm.{s}") for s in ("L", "R")}}


def _grip(m, side):
    """The grip of a hand's pose table ("guitar:strum", ...), "" without one."""
    rig = m.hands.get(side)
    return str(((rig or {}).get("h") or {}).get("grip") or "")


def _keyed(ob, path, n_comp, frames, still):
    """(F, n_comp) values of an object's keyed property on `frames` (its value where it has no curve)."""
    act = ob.animation_data.action if ob.animation_data else None
    out = np.empty((len(frames), n_comp))
    for c in range(n_comp):
        fc = act.fcurves.find(path, index=c) if act is not None else None
        out[:, c] = [fc.evaluate(f) for f in frames] if fc is not None else still[c]
    return out


def _guitar(ctx, m, mp, frames, gname):
    """Turn the worn guitar about its face normal through the point between her hands by the plan's `guitar` (+ its
    neck up): the hand bar's tilt lifts one hand and lowers the other. The hands ride it. Keys on top of the root's own
    turn (a strum's neck kick)."""
    ang = mp["body"]["guitar"]
    g = ctx.props.get(gname)
    if g is None or np.abs(ang).max() < 1e-4:
        return None
    entry = next((u for u in g.card.get("use", {}).get("grip", []) if u.get("normal")), None)
    if entry is None:
        return None
    root = g.root
    sc = bpy.context.scene
    keep = sc.frame_current
    sc.frame_set(int(ctx.start + ctx.settle))
    gi = root.matrix_world.inverted()
    hands = [gi @ m.ik[s][0].matrix_world.translation for s in ("L", "R") if s in m.ik]
    pivot = sum(hands, Vector()) / len(hands) if hands else Vector()
    nl = Vector(entry["normal"]).normalized()
    Rw = root.matrix_world.to_3x3().normalized()
    rise = (Rw @ (Quaternion(nl, 0.1) @ Vector((0.0, 0.0, 1.0)))).z - (Rw @ Vector((0.0, 0.0, 1.0))).z
    sign = 1.0 if rise > 0 else -1.0
    sc.frame_set(keep)
    if root.rotation_mode != "QUATERNION":
        q = root.matrix_basis.to_quaternion()
        root.rotation_mode = "QUATERNION"
        root.rotation_quaternion = q
    q0 = _keyed(root, "rotation_quaternion", 4, frames, tuple(root.rotation_quaternion))
    l0 = _keyed(root, "location", 3, frames, tuple(root.location))
    scale = root.matrix_basis.to_scale()
    S3 = Matrix.Diagonal(scale)
    rots, locs = [], []
    for i in range(len(frames)):
        qb = Quaternion(q0[i]).normalized()
        qn = Quaternion(nl, sign * float(ang[i]))
        rots.append(tuple(qb @ qn))
        locs.append(tuple(Vector(l0[i]) + qb.to_matrix() @ (S3 @ (pivot - qn @ pivot))))
    K.key_vec(root, "rotation_quaternion", frames, K.continuous(rots), interp="LINEAR")
    K.key_vec(root, "location", frames, locs, interp="LINEAR")
    return {"deg": round(float(np.degrees(np.abs(ang).max())), 1)}


def _hang(ctx, m, mp, ts, frames, side, cut):
    """A hand whose thread is cut drops off what it played and swings on its arm into its hang (core.marionette.
    hang_path), in the frame its goal rides, turning with the arm as it swings about the shoulder (so the forearm is
    not wrung); with `back` it goes back to its own keys over that window."""
    goal, _pole = m.ik[side]
    sc = bpy.context.scene
    keep = sc.frame_current
    sc.frame_set(int(ctx.start + ctx.settle))
    Pf = goal.matrix_world @ goal.matrix_basis.inverted()        # the frame the goal's keys are in
    shoulder = np.asarray(Pf.inverted() @ (mp["F"] @ mp["shoulders"][side]))
    to_goal = Pf.to_3x3().inverted() @ mp["Frot"].to_matrix()
    sc.frame_set(keep)
    goal.rotation_mode = "QUATERNION"
    loc = _keyed(goal, "location", 3, frames, tuple(goal.location))
    rot = _keyed(goal, "rotation_quaternion", 4, frames, tuple(goal.rotation_quaternion))
    i0 = min(int(np.searchsorted(ts, cut["t"])), len(ts) - 1)
    start = loc[i0]
    hang = np.asarray(start) + np.asarray(to_goal @ Vector(cut["hang"]))
    path = MR.hang_path(ts, cut["t"], start, shoulder, hang)
    w = MR.step(ts, cut["t"])
    if "back" in cut:
        w = w * (1.0 - PF.smooth((ts - cut["back"][0]) / (cut["back"][1] - cut["back"][0])))
    new = loc + w[:, None] * (path - loc)
    r0 = Vector(start - shoulder)
    q_cut = Quaternion(rot[i0]).normalized()
    quats = []
    for i in range(len(ts)):
        q = Quaternion(rot[i]).normalized()
        if w[i] > 0.0:
            swung = r0.rotation_difference(Vector(path[i] - shoulder)) @ q_cut
            q = q.slerp(swung, float(w[i]))
        quats.append(tuple(q))
    K.key_vec(goal, "location", frames, [tuple(v) for v in new], interp="LINEAR")
    K.key_vec(goal, "rotation_quaternion", frames, K.continuous(quats), interp="LINEAR")
    return {"t": cut["t"], "drop_mm": round(float(np.linalg.norm(hang - start)) * 1000, 1)}


def _hand_bar(ctx, m, mp, ts, frames, gname):
    """The hand bar follows the picking hand: its end over that hand goes up and down with the hand's strokes (the
    thread does the strumming) while that thread holds, the bar turning about the other hand's eye so that thread keeps
    its pull."""
    g = ctx.props.get(gname)
    side = next((s for s in ("R", "L") if s in m.ik and m.ik[s][0].parent is (g.root if g else None)
                 and _grip(m, s).endswith(":strum")), None) if g else None
    if side is None:
        return None
    bar = bpy.data.objects[mp["card"]["bars"]["hand"]]
    anchors = mp["card"]["anchors"]
    x_hand = float(anchors[f"wrist_{side}"]["point"][0])
    pivot = Vector(anchors[f"wrist_{'L' if side == 'R' else 'R'}"]["point"])
    pivot.z = 0.0
    lever = x_hand - pivot.x
    goal = m.ik[side][0]
    loc = _keyed(goal, "location", 3, frames, tuple(goal.location))
    Rw = g.root.matrix_world.to_3x3()
    dz = np.array([(Rw @ Vector(v)).z for v in loc - loc[0]])
    dz = dz - PF.lowpass(dz, 30)                                 # the strokes, not where the hand rests
    held = MR.held(ts, mp["P"], f"wrist_{side}")
    ang = np.arcsin(np.clip(dz * held / abs(lever), -0.3, 0.3)) * (-1.0 if lever < 0 else 1.0)
    at = Vector(mp["card"]["hand_bar"]["at"])
    qs = [Quaternion((0, 1, 0), -float(a)) for a in ang]
    bar.rotation_mode = "QUATERNION"
    K.key_vec(bar, "rotation_quaternion", frames, [tuple(q) for q in qs], interp="LINEAR")
    K.key_vec(bar, "location", frames, [tuple(at + pivot - q @ pivot) for q in qs], interp="LINEAR")
    return {"side": side, "deg": round(float(np.degrees(np.abs(ang).max())), 1)}


def hands(ctx, name, m, mp, ts, frames):
    """After the strum: the guitar turned with the hand bar, the hands whose threads are cut swung on their arms, the
    hand bar following the picking hand. Returns report numbers."""
    spec = ctx.section("perform", name)
    st = spec.get("strum")
    st = st[0] if isinstance(st, list) and st else st
    gname = mp["P"]["guitar"] or (st.get("prop") if isinstance(st, dict) else None)
    out = {}
    if gname:
        out["guitar"] = _guitar(ctx, m, mp, frames, gname)
        out["hand_bar"] = _hand_bar(ctx, m, mp, ts, frames, gname)
    for c in mp["P"]["cuts"]:
        for th in c["threads"]:
            if th.startswith("wrist_"):
                side = th[-1]
                grip = _grip(m, side)
                if side in m.ik and not grip.endswith(":neck"):      # a hand on a neck holds on to it
                    out[f"hang_{side}"] = _hang(ctx, m, mp, ts, frames, side, c)
    return out


# ---- the stage: the threads


def _windows(ctx, name):
    entries = ctx.data.get("stepped", [])
    if not entries:
        return []
    try:
        return [w for w in STP.normalize(entries, list(ctx.cast), ctx.fps, ctx.frame0, ctx.end) if name in w["cast"]]
    except STP.StepError as e:
        raise BuildError(str(e)) from None


def _last(ctx, m):
    shows = m.spec.get("shows")
    if shows:
        return min(ctx.end, int(math.ceil(ctx.frame(max(float(b) for _a, b in shows)))))
    return ctx.end


def run(ctx):
    out = {}
    for name, m in ctx.cast.items():
        got = plan(ctx, name)
        if got is None or m.arm is None:
            continue
        P, prop, card = got
        bars = [bpy.data.objects[v] for v in card["bars"].values()]
        held = 0
        for w in _windows(ctx, name):                             # the bars step with her
            for ob in bars:
                ad = ob.animation_data
                for fc in (ad.action.fcurves if ad and ad.action else ()):
                    if len(fc.keyframe_points):
                        STEPPED.hold(fc, w)
                        held += 1
        f0 = max(ctx.start, int(math.floor(ctx.frame(P["from"] - P["drop"]))) - 1)
        f1 = _last(ctx, m)
        frames = np.arange(f0, f1 + 1)
        ts = np.array([ctx.time(f) for f in frames])
        anchors = [bpy.data.objects[card["anchors"][p]["object"]] for p in MR.PARTS]
        ends = [bpy.data.objects[f"{prop.name}_end_{p}"] for p in MR.PARTS]
        n, k = len(frames), len(MR.PARTS)
        A, B = np.empty((n, k, 3)), np.empty((n, k, 3))
        sc = bpy.context.scene
        keep = sc.frame_current
        bpy.context.view_layer.update()                          # objects the sim removed leave stale entries
        with SCOPE.only(anchors + ends + bars + [m.arm, m.root, prop.root]):
            for i, f in enumerate(frames):
                sc.frame_set(int(f))
                dg = bpy.context.evaluated_depsgraph_get()
                for j in range(k):
                    A[i, j] = anchors[j].evaluated_get(dg).matrix_world.translation
                    B[i, j] = ends[j].evaluated_get(dg).matrix_world.translation
        sc.frame_set(keep)
        F, Frot = _frame(m)
        Fr = np.array(Frot.to_matrix())
        floor = float(F.translation.z)
        i_on = int(np.clip(np.searchsorted(ts, P["from"]), 0, n - 1))
        lengths = np.linalg.norm(A[i_on] - B[i_on], axis=1)
        Ri = np.array(prop.root.matrix_world.inverted())
        npts = int(card["thread"]["points"])
        rep = {}
        for j, part in enumerate(MR.PARTS):
            tc = MR.cut_time(P, part)
            cut = next((c for c in P["cuts"] if part in c["threads"]), None)
            side = Fr @ np.array([SIDE[part], 0.0, 0.0])
            drape = Fr @ np.asarray(MR.DRAPE[part], float)
            main = np.empty((n, npts, 3))
            low = np.empty((n, npts, 3))
            show_main, show_low = np.zeros(n, bool), np.zeros(n, bool)
            snip = np.empty((n, 3))
            L = float(lengths[j])
            C = B_slide = None
            ta = P["from"] - P["drop"]
            for i, t in enumerate(ts):
                a, b = A[i, j], B[i, j]
                low[i] = b
                if t < ta:
                    main[i] = MR.dangle(a, L, 0.0, npts, side)
                elif t < P["from"] - MR.ATTACH:
                    main[i] = MR.dangle(a, L, t - ta, npts, side)
                    show_main[i] = True
                elif t < P["from"]:
                    s = float(PF.smooth((t - P["from"] + MR.ATTACH) / MR.ATTACH))
                    e = (1.0 - s) * (a + np.array([0.0, 0.0, -L])) + s * b
                    main[i] = (1.0 - s) * MR.dangle(a, L, t - ta, npts, side) + s * MR.sag_points(a, e, L * 1.03, npts, side)
                    show_main[i] = True
                elif t < tc:
                    main[i] = MR.sag_points(a, b, L, npts, side)
                    main[i] += np.outer(MR.twang(npts, t - P["from"]), side)
                    show_main[i] = True
                else:
                    if C is None:
                        C = MR.point_at(main[i - 1] if i else MR.sag_points(a, b, L, npts, side), cut["at"])
                    tau = t - tc
                    main[i] = MR.cut_upper(a, C, tau, npts, side)
                    show_main[i] = tau < MR.YANK + 0.08
                    if tau >= MR.SLIDE and B_slide is None:
                        B_slide = b.copy()
                    low[i] = MR.cut_lower(b, C, tau, drape, floor, B_slide, float(card["thread"]["radius"]), npts)
                    show_low[i] = True
                snip[i] = MR.point_at(main[i], cut["at"] if cut else 0.5) if C is None else C
            for kind, pts, shown in (("thread", main, show_main), ("cut", low, show_low)):
                ob = bpy.data.objects[f"{prop.name}_{kind}_{part}"]
                local = pts @ Ri[:3, :3].T + Ri[:3, 3]
                cu = ob.data
                for q in range(npts):
                    for c in range(3):
                        K.set_fcurve(cu, f"splines[0].points[{q}].co", c, frames, local[:, q, c], interp="LINEAR")
                _key_hidden(ob, frames, ~shown)
            sn = bpy.data.objects[f"{prop.name}_snip_{part}"]
            K.key_vec(sn, "location", frames, [tuple(v) for v in snip @ Ri[:3, :3].T + Ri[:3, 3]], interp="LINEAR")
            rep[part] = {"length_m": round(L, 3), "cut": None if tc == math.inf else round(tc, 3)}
        out[name] = {"prop": prop.name, "frames": [int(frames[0]), int(frames[-1])], "threads": rep,
                     "bar_curves_stepped": held}
        ctx.log("marionette", name, json.dumps(out[name]))
    return out
