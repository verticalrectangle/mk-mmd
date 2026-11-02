"""motion: VMD motions on a character's NLA, optionally retimed to the song's beats and limited to some bones.

[[motion.<cast>]] keys
  vmd = path or registry slug (kind motion)
  start = 0.0          clip seconds where source frame `from` plays
  from, to             source frame range (default: the whole motion)
  scale = 1.0          time scale; or retime = "beats" with timeline = "audio/timeline.json" (the default): the motion's own beat
                       (core.vmd.beat_fit: `bpm` when given, else core.vmd.tempo_phase's) is scaled to the song's beat
                       period over `per_beat` (1; 2 puts the motion's beats on the song's 8ths) and its phase snapped to
                       the nearest song beat (or 8th ...) after `start`
  bones = "all" | "upper" | "lower" | "arms" | [groups, semantic or Blender names]   which bones the motion drives
                       ("arms": shoulders, arms, elbows, wrists and their twist bones, not the fingers)
  legs = "ik"          "ik": the member's leg IK holds the feet where the IK bones are (a motion with FK legs, a BVH
                       conversion's, then keeps them planted); "fk": the IK bones are keyed where the motion's own legs
                       put the ankles (follow_legs), so it steps, kicks, turns and jumps as it was made
  morphs = true        also play the VMD's facial keys
  blend_in, blend_out  seconds; influence ramps on the strip
  influence = 1.0      the share of the motion the strip plays (0..1, eased in and out over the blends): a torso strip
                       at 0.5 turns and leans the body half as far as the motion does
  mirror = false       true: the motion mirrored left for right (core.vmd.mirror), the other hand's gestures
  extend = "hold"      "hold": the strip's first and last poses stay before and after it; "forward": only its last pose
                       stays, after it (a dance paused between two strips, the later one picking up where it stopped);
                       "none": it acts on its own frames only
Each entry is a track of its own, a later entry's above an earlier one's (where both play, the later one wins).
Strips play in a track under the active action: pose / perform keys win where both key a bone, except on a member with a
hand a `free` move lets go, whose own keys compose with its strips (the action's blend type is Combine: a walk strip's
arm swing under the pose stage's forearm pre-bend, which goes with the IK). Once the strips are laid, the hands let go to
them are keyed again where the strips have them (mkmmd.blender.build.pose.free_hands)."""
import json
import os
import tempfile

import bpy
from mathutils import Matrix, Quaternion, Vector

from ...core import footplant as FP
from ...core import timeline as TL
from ...core import vmd as V
from .. import scene as S
from . import BuildError
from . import pose as POSE

UPPER = {"upper_body", "upper_body2", "upper_body3", "neck", "head", "eyes", "eye", "shoulder_p", "shoulder",
         "shoulder_c", "arm", "arm_twist", "elbow", "wrist_twist", "wrist", "thumb", "index", "middle", "ring",
         "little"}
LOWER = {"center", "groove", "waist", "lower_body", "leg", "knee", "ankle", "toe", "leg_ik", "toe_ik",
         "leg_ik_parent", "leg_d", "knee_d", "ankle_d", "toe_ex", "waist_cancel"}
ARMS = {"shoulder_p", "shoulder", "shoulder_c", "arm", "arm_twist", "elbow", "wrist_twist", "wrist"}   # not the fingers
GROUPS = {"upper": UPPER, "lower": LOWER, "arms": ARMS}
EXTEND = {"hold": "HOLD", "forward": "HOLD_FORWARD", "none": "NOTHING"}


def _vmd_path(ctx, ref):
    p = ctx.path(ref)
    if os.path.exists(p):
        return p
    reg = os.path.join(os.path.expanduser(ctx.assets or "~/mk-assets"), "registry.json")
    if os.path.exists(reg):
        with open(reg, encoding="utf-8") as fh:
            for e in json.load(fh):
                if e["slug"] == ref:
                    return e["path"]
    raise BuildError(f"motion {ref!r}: no such file or registry slug")


def import_vmd(member, path, key):
    """VMD -> (bone action, morph action or None), detached from the model for NLA use. The member's IK is left as it
    was: mmd_tools switches a member's IK off for the whole scene (the IK constraints' influence) when the VMD's first
    frame turns it off, as FK motions (a BVH conversion's legs) do, which left the legs of every later stage without
    their IK; the switches the VMD keys are dropped too (nothing reads them during playback)."""
    root, arm = member.root, member.arm
    bpy.ops.object.select_all(action="DESELECT")
    root.select_set(True)
    bpy.context.view_layer.objects.active = root
    sc = bpy.context.scene
    keep = (sc.frame_start, sc.frame_end, sc.frame_current)
    prev = arm.animation_data.action if arm.animation_data else None
    if arm.animation_data:
        arm.animation_data.action = None
    toggles = {pb.name: pb.mmd_ik_toggle for pb in arm.pose.bones}
    influence = {(pb.name, c.name): c.influence for pb in arm.pose.bones for c in pb.constraints}
    sc.frame_current = 0                       # mmd_tools offsets imported keys by the current frame
    bpy.ops.mmd_tools.import_vmd(filepath=path, scale=0.08, bone_mapper="PMX", use_pose_mode=False,
                                 update_scene_settings=False, use_nla=False, create_new_action=True,
                                 log_level="ERROR")
    sc.frame_start, sc.frame_end, sc.frame_current = keep
    for name, on in toggles.items():
        pb = arm.pose.bones[name]
        if pb.mmd_ik_toggle != on:
            pb.mmd_ik_toggle = on              # its update sets the chain's constraints, which are restored next
    for (b, c), v in influence.items():
        arm.pose.bones[b].constraints[c].influence = v
    act = arm.animation_data.action if arm.animation_data else None
    if act:
        act.name = f"{key}_bone"
        act.use_fake_user = True
        for fc in [fc for fc in act.fcurves if fc.data_path.endswith(".mmd_ik_toggle")]:
            act.fcurves.remove(fc)
        near_rest(act)
    arm.animation_data.action = prev
    mact = None
    holders = S.morph_holders(arm)                 # the bound sliders' placeholder, or the meshes
    for mm in dict.fromkeys(holders + list(member.meshes)):   # mmd_tools keys the placeholder and every mesh
        sk = mm.data.shape_keys
        if sk and sk.animation_data and sk.animation_data.action:
            if mm in holders:
                mact = sk.animation_data.action
                mact.name = f"{key}_morph"
                mact.use_fake_user = True
            sk.animation_data.action = None
    return act, mact


def near_rest(act):
    """Turn every bone's quaternion track of `act` that sits on the far side of the rest (w below 0 over its keys) to
    the near side: the same rotations (q and -q are one), but a strip that blends with the rest pose or with another
    strip turns the short way. Blender blends quaternions component by component, and BVH conversions key whole tracks
    at w < 0, so a blend-in swung the neck and twisted the arms half a turn on the way."""
    tracks = {}
    for fc in act.fcurves:
        if fc.data_path.endswith(".rotation_quaternion"):
            tracks.setdefault(fc.data_path, {})[fc.array_index] = fc
    for fcs in tracks.values():
        w = fcs.get(0)
        if w is None or sum(k.co[1] for k in w.keyframe_points) >= 0.0:
            continue
        for fc in fcs.values():
            for k in fc.keyframe_points:
                k.co[1], k.handle_left[1], k.handle_right[1] = -k.co[1], -k.handle_left[1], -k.handle_right[1]
            fc.update()


def follow_legs(act, arm):
    """Key the leg IK bones of `act` where its own legs put the ankles (`legs = "fk"`): a motion that keys the legs'
    rotations and not their IK bones (FK legs, a BVH conversion's) then steps, kicks, turns and jumps as it was made,
    under the member's IK, where the IK would hold the feet where the IK bones are. Per frame the chain from the root to
    each ankle is posed from `act`'s keys (a bone it does not key at rest), and the IK bone goes with the ankle as one
    rigid body, keyed linear on every frame, so its child the toe IK brings the ankle's pitch along. A motion made on a
    body with other legs floats its feet above the floor or sinks them into it: when it keys both legs the whole body is
    moved down (the centre bone and the IK bones together, so the knees bend as the motion bends them) by how far the
    lower foot is above where it rests on the floor on the motion's lowest tenth of frames; and it slides a resting foot,
    which is held where it landed (core.footplant.plant). Constraints on the chain (a waist cancel's) are not followed.
    Returns (the sides keyed, metres moved down, the rests held per side); no sides when `act` keys no leg rotation."""
    sem = S.semantic_map(arm)
    fcs = {(fc.data_path, fc.array_index): fc for fc in act.fcurves}
    bones = arm.data.bones

    def keyed(name):
        return any(len(fcs[k].keyframe_points) > 1 for k in fcs if k[0] == f'pose.bones["{name}"].rotation_quaternion')

    def value(name, prop, i, f, rest):
        fc = fcs.get((f'pose.bones["{name}"].{prop}', i))
        return fc.evaluate(f) if fc else rest

    def posed(bone, f, memo):                      # the bone's pose matrix in the armature's space at frame f
        if bone is None:
            return Matrix()
        if bone.name not in memo:
            q = Quaternion([value(bone.name, "rotation_quaternion", i, f, 1.0 if i == 0 else 0.0) for i in range(4)])
            basis = Matrix.Translation(Vector([value(bone.name, "location", i, f, 0.0) for i in range(3)])) \
                @ q.normalized().to_matrix().to_4x4()
            memo[bone.name] = local_to_pose(bone, basis, posed(bone.parent, f, memo))
        return memo[bone.name]

    def local_to_pose(bone, m, parent_pose, invert=False):
        if bone.parent is None:
            return bone.convert_local_to_pose(m, bone.matrix_local, invert=invert)
        return bone.convert_local_to_pose(m, bone.matrix_local, parent_matrix=parent_pose,
                                          parent_matrix_local=bone.parent.matrix_local, invert=invert)

    legs = {}
    for side in ("L", "R"):
        names = [sem.get(f"{s}.{side}") for s in ("leg", "knee", "ankle", "leg_ik")]
        if None not in names and (keyed(names[0]) or keyed(names[1])):
            ankle, ik = bones[names[2]], bones[names[3]]
            legs[side] = (ankle, ik, ankle.matrix_local.inverted() @ ik.matrix_local)   # the IK bone on the ankle
    if not legs:
        return [], 0.0, {}
    center = bones.get(sem.get("center", ""))
    frames = range(int(act.frame_range[0]), int(act.frame_range[1] + 0.999) + 1)
    goals = {side: [] for side in legs}
    centers = []
    for f in frames:
        memo = {}
        for side, (ankle, ik, rigid) in legs.items():
            goals[side].append((posed(ankle, f, memo) @ rigid, posed(ik.parent, f, memo)))
        if center is not None:
            centers.append((posed(center, f, memo), posed(center.parent, f, memo)))
    above = sorted(min(goals[s][k][0].translation.z - legs[s][1].matrix_local.translation.z for s in legs)
                   for k in range(len(frames)))
    drop = above[int(0.1 * (len(above) - 1))] if centers and len(legs) == 2 else 0.0   # a foot left on its IK bone stands
    down = Matrix.Translation((0.0, 0.0, -drop))

    def key(bone, prop, vals, n):
        path = f'pose.bones["{bone.name}"].{prop}'
        for i in range(n):
            fc = fcs.get((path, i)) or act.fcurves.new(path, index=i, action_group=bone.name)
            fc.keyframe_points.clear()
            fc.keyframe_points.add(len(vals))
            fc.keyframe_points.foreach_set("co", [c for f, v in zip(frames, vals) for c in (f, v[i])])
            fc.keyframe_points.foreach_set("interpolation", [1] * len(vals))   # LINEAR
            fc.update()

    rests = {}
    for side, (_, ik, _) in legs.items():
        carried = center is not None and center in ik.parent_recursive   # moving the centre moves the IK bone's parent
        moved = [down @ goal for goal, _ in goals[side]]
        held, spans = FP.plant([m.translation for m in moved], ik.matrix_local.translation.z)
        rests[side] = len(spans)
        locs, quats, prev = [], [], None
        for m, (_, parent_pose), at in zip(moved, goals[side], held):
            m.translation = Vector(at)
            loc, q, _ = local_to_pose(ik, m, down @ parent_pose if carried else parent_pose, invert=True).decompose()
            if prev is not None and prev.dot(q) < 0.0:                   # the same hemisphere as the frame before
                q.negate()
            prev = q
            locs.append(loc)
            quats.append(q)
        key(ik, "location", locs, 3)
        key(ik, "rotation_quaternion", quats, 4)
    if centers and drop:
        key(center, "location", [local_to_pose(center, down @ m, p, invert=True).to_translation() for m, p in centers], 3)
    return sorted(legs), drop, rests


def _group(sem):
    """Body group stem of a semantic name: index2.L -> index, thumb_tip.R -> thumb, upper_body2 -> upper_body2."""
    stem = sem.split(".")[0]
    if stem in UPPER or stem in LOWER:
        return stem
    return stem.replace("_tip", "").rstrip("0123456789")


def mask_action(act, arm, which):
    """A copy of `act` keeping only the fcurves of the chosen bones: "all", a group (GROUPS: "upper", "lower", "arms")
    or a list of groups and bone names (semantic, Blender or PMX)."""
    if which == "all":
        return act
    inv = {b: _group(sem) for sem, b in S.semantic_map(arm).items()}
    names = [which] if isinstance(which, str) else list(which)
    if isinstance(which, str) and which not in GROUPS:
        raise BuildError(f"bones = {which!r}: all, {', '.join(GROUPS)} or a list")
    keep = set()
    for n in names:
        keep |= {b for b, stem in inv.items() if stem in GROUPS[n]} if n in GROUPS else {S.resolve_bone(arm, n)}
    new = act.copy()
    new.name = f"{act.name}_{which if isinstance(which, str) else 'sel'}"
    for fc in list(new.fcurves):
        dp = fc.data_path
        if dp.startswith('pose.bones["') and dp.split('"')[1] not in keep:
            new.fcurves.remove(fc)
    return new


def nla_strip(id_data, action, start, scale=1.0, src_start=None, src_end=None, track="motion", blend_in=0.0,
              blend_out=0.0, extend="hold", influence=1.0):
    ad = id_data.animation_data or id_data.animation_data_create()
    tr = ad.nla_tracks.get(track) or ad.nla_tracks.new()
    tr.name = track
    s = tr.strips.new(f"{action.name}@{int(start)}", int(round(start)), action)
    if src_start is not None:
        s.action_frame_start = src_start
    if src_end is not None:
        s.action_frame_end = src_end
    s.scale = scale
    s.frame_start_ui = start                   # exact (fractional) start, duration kept
    s.use_auto_blend = False                   # before the blends: turning auto-blend off zeroes them (Blender 4.2)
    s.blend_in, s.blend_out = blend_in, blend_out
    s.blend_type = "REPLACE"
    s.extrapolation = EXTEND[extend]           # "hold": first and last poses before and after; "forward": the last after
    if influence < 1.0:                        # a share of the motion: the influence keyed, eased over the blends
        s.use_animated_influence = True        # (Blender then reads the curve, not the blends)
        fc = s.fcurves.find("influence")
        fc.keyframe_points.clear()
        a, b = s.frame_start, s.frame_end
        for f, v in sorted({a: 0.0 if blend_in else influence, a + blend_in: influence, b - blend_out: influence,
                            b: 0.0 if blend_out else influence}.items()):
            fc.keyframe_points.insert(f, v, options={"FAST"}).interpolation = "BEZIER"
        fc.update()
    return s


def _mirrored(path):
    """A temporary copy of the VMD at `path` mirrored left for right (core.vmd.mirror); the caller removes it."""
    with open(path, "rb") as fh:
        data = V.mirror(fh.read())
    fd, out = tempfile.mkstemp(suffix=".vmd")
    with os.fdopen(fd, "wb") as fh:
        fh.write(data)
    return out


def beat_fit(path, timeline, fps, start_t, src_from, bpm=None, per_beat=1.0):
    """(scale, start frame relative to clip 0, info) so the motion's beats land on the song's (core.vmd.beat_fit)."""
    try:
        return V.beat_fit(V.read(path), TL.beats(timeline)[0], fps, start_t, src_from, bpm=bpm, per_beat=per_beat)
    except ValueError as e:
        raise BuildError(f"retime=beats: {e}") from None


def run(ctx):
    out = {}
    for name, items in (ctx.data.get("motion") or {}).items():
        m = ctx.cast.get(name)
        if m is None:
            raise BuildError(f"[motion.{name}]: no cast member {name!r}")
        for i, it in enumerate(items if isinstance(items, list) else [items]):
            path = _vmd_path(ctx, it["vmd"])
            key = f"{name}_{i}_{os.path.splitext(os.path.basename(path))[0]}"[:50]
            src = _mirrored(path) if it.get("mirror") else path
            try:
                act, mact = import_vmd(m, src, key)
            finally:
                if src != path:
                    os.unlink(src)
            if act is None:
                raise BuildError(f"motion {path}: no bone keys")
            src_from = float(it.get("from", act.frame_range[0]))
            src_to = float(it.get("to", act.frame_range[1]))
            info = {"vmd": os.path.basename(path), "source": [src_from, src_to]}
            if it.get("mirror"):
                info["mirror"] = True
            fit = {k: it[k] for k in ("bpm", "per_beat") if k in it}
            for k, v in fit.items():
                if isinstance(v, bool) or not isinstance(v, (int, float)) or v <= 0:
                    raise BuildError(f"motion.{name}[{i}] {k} = {v!r}: a positive number")
            if it.get("retime") == "beats":
                tl = ctx.timeline(it, f"motion.{name}[{i}] retime = \"beats\"")
                scale, start_rel, got = beat_fit(path, tl, ctx.fps, float(it.get("start", 0.0)), src_from,
                                                 bpm=fit.get("bpm"), per_beat=fit.get("per_beat", 1.0))
                start = ctx.frame0 + start_rel
                info.update(got)
            elif fit:
                raise BuildError(f"motion.{name}[{i}] {' and '.join(fit)}: they fit the motion to the song's beats, "
                                 f"with retime = \"beats\"")
            else:
                scale = float(it.get("scale", 1.0))
                start = ctx.frame(float(it.get("start", 0.0)))
            legs = it.get("legs", "ik")
            if legs not in ("ik", "fk"):
                raise BuildError(f"motion.{name}[{i}] legs = {legs!r}: \"ik\" or \"fk\"")
            if legs == "fk":
                sides, drop, rests = follow_legs(act, m.arm)
                if not sides:
                    raise BuildError(f"motion.{name}[{i}] legs = \"fk\": {os.path.basename(path)} keys no leg rotation "
                                     f"(its legs are its IK bones' already)")
                info["legs"] = {"fk": sides, "lowered_m": round(drop, 3), "rests": rests}
            act_m = mask_action(act, m.arm, it.get("bones", "all"))
            extend = it.get("extend", "hold")
            if extend not in EXTEND:
                raise BuildError(f"motion.{name}[{i}] extend = {extend!r}: {' or '.join(EXTEND)}")
            share = it.get("influence", 1.0)
            if isinstance(share, bool) or not isinstance(share, (int, float)) or not 0.0 < share <= 1.0:
                raise BuildError(f"motion.{name}[{i}] influence = {share!r}: the share of the motion it plays, above 0 "
                                 f"and at most 1")
            bi, bo = float(it.get("blend_in", 0.0)) * ctx.fps, float(it.get("blend_out", 0.0)) * ctx.fps
            s = nla_strip(m.arm, act_m, start, scale, src_from, src_to, track=f"motion{i}", blend_in=bi,
                          blend_out=bo, extend=extend, influence=float(share))
            info["strip"] = [round(s.frame_start, 2), round(s.frame_end, 2)]
            if share < 1.0:
                info["influence"] = float(share)
            if mact is not None and it.get("morphs", True):
                for mm in S.morph_holders(m.arm):
                    if mm.data.shape_keys:
                        nla_strip(mm.data.shape_keys, mact, start, scale, src_from, src_to, track=f"morph{i}",
                                  extend=extend)
                info["morphs"] = True
            out.setdefault(name, []).append(info)
            ctx.log("motion", name, json.dumps(info, ensure_ascii=False))
    handed = POSE.free_hands(ctx)                     # the hands let go to the strips, keyed where the strips have them
    if handed:
        out["free_hands"] = handed
    return out
