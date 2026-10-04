"""motion: VMD motions on a character's NLA, optionally retimed to the song's beats and limited to some bones.

[[motion.<cast>]] keys
  vmd = path or registry slug (kind motion)
  start = 0.0          clip seconds where source frame `from` plays
  from, to             source frame range (default: the whole motion)
  scale = 1.0          time scale; or retime = "beats" with timeline = "audio/timeline.json" (the default): the motion's own beat
                       (core.vmd.tempo_phase) is scaled to the song's beat period and its phase snapped to the nearest
                       song beat after `start`
  bones = "all" | "upper" | "lower" | [semantic or Blender names]   which bones the motion drives
  morphs = true        also play the VMD's facial keys
  blend_in, blend_out  seconds; influence ramps on the strip
Strips play in a track under the active action (pose / perform keys win where both key a bone)."""
import json
import os

import bpy
import numpy as np

from ...core import timeline as TL
from ...core import vmd as V
from .. import scene as S
from . import BuildError

UPPER = {"upper_body", "upper_body2", "upper_body3", "neck", "head", "eyes", "eye", "shoulder_p", "shoulder",
         "shoulder_c", "arm", "arm_twist", "elbow", "wrist_twist", "wrist", "thumb", "index", "middle", "ring",
         "little"}
LOWER = {"center", "groove", "waist", "lower_body", "leg", "knee", "ankle", "toe", "leg_ik", "toe_ik",
         "leg_ik_parent", "leg_d", "knee_d", "ankle_d", "toe_ex", "waist_cancel"}


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
    """VMD -> (bone action, morph action or None), detached from the model for NLA use."""
    root, arm = member.root, member.arm
    bpy.ops.object.select_all(action="DESELECT")
    root.select_set(True)
    bpy.context.view_layer.objects.active = root
    sc = bpy.context.scene
    keep = (sc.frame_start, sc.frame_end, sc.frame_current)
    prev = arm.animation_data.action if arm.animation_data else None
    if arm.animation_data:
        arm.animation_data.action = None
    sc.frame_current = 0                       # mmd_tools offsets imported keys by the current frame
    bpy.ops.mmd_tools.import_vmd(filepath=path, scale=0.08, bone_mapper="PMX", use_pose_mode=False,
                                 update_scene_settings=False, use_nla=False, create_new_action=True,
                                 log_level="ERROR")
    sc.frame_start, sc.frame_end, sc.frame_current = keep
    act = arm.animation_data.action if arm.animation_data else None
    if act:
        act.name = f"{key}_bone"
        act.use_fake_user = True
    arm.animation_data.action = prev
    mact = None
    for mm in member.meshes:
        sk = mm.data.shape_keys
        if sk and sk.animation_data and sk.animation_data.action:
            mact = sk.animation_data.action
            mact.name = f"{key}_morph"
            mact.use_fake_user = True
            sk.animation_data.action = None
    return act, mact


def _group(sem):
    """Body group stem of a semantic name: index2.L -> index, thumb_tip.R -> thumb, upper_body2 -> upper_body2."""
    stem = sem.split(".")[0]
    if stem in UPPER or stem in LOWER:
        return stem
    return stem.replace("_tip", "").rstrip("0123456789")


def mask_action(act, arm, which):
    """A copy of `act` keeping only the fcurves of the chosen bones."""
    if which == "all":
        return act
    inv = {b: _group(sem) for sem, b in S.semantic_map(arm).items()}
    if isinstance(which, str):
        groups = UPPER if which == "upper" else LOWER if which == "lower" else None
        if groups is None:
            raise BuildError(f"bones = {which!r}: all, upper, lower or a list")
        keep = {b for b, stem in inv.items() if stem in groups}
    else:
        keep = {S.resolve_bone(arm, n) for n in which}
    new = act.copy()
    new.name = f"{act.name}_{which if isinstance(which, str) else 'sel'}"
    for fc in list(new.fcurves):
        dp = fc.data_path
        if dp.startswith('pose.bones["') and dp.split('"')[1] not in keep:
            new.fcurves.remove(fc)
    return new


def nla_strip(id_data, action, start, scale=1.0, src_start=None, src_end=None, track="motion", blend_in=0.0,
              blend_out=0.0):
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
    s.blend_in, s.blend_out = blend_in, blend_out
    s.blend_type = "REPLACE"
    s.extrapolation = "HOLD"
    s.use_auto_blend = False
    return s


def beat_fit(path, timeline, fps, start_t, src_from):
    """(scale, start frame relative to clip 0, info) so the motion's beats land on the song's beats."""
    mo = V.read(path)
    tp = V.tempo_phase(mo)
    beats = np.asarray(TL.beats(timeline)[0], float)
    if tp is None or len(beats) < 2:
        raise BuildError(f"retime=beats: {'no beat in the motion' if tp is None else 'no beats in the timeline'}")
    period_v, phase_v = tp[0], tp[1]
    period_s = float(np.median(np.diff(beats))) * fps
    scale = period_s / period_v
    k = np.ceil((src_from - phase_v) / period_v)
    src_beat = phase_v + k * period_v                     # first motion beat at or after `from`
    want = start_t * fps + (src_beat - src_from) * scale   # where it would land unsnapped (frames from clip 0)
    j = int(np.argmin(np.abs(beats * fps - want)))
    land = beats[j] * fps
    return scale, land - (src_beat - src_from) * scale, {"motion_bpm": round(V.MMD_FPS * 60 / period_v, 2),
                                                        "song_bpm": round(fps * 60 / period_s, 2),
                                                        "scale": round(scale, 4)}


def run(ctx):
    out = {}
    for name, items in (ctx.data.get("motion") or {}).items():
        m = ctx.cast.get(name)
        if m is None:
            raise BuildError(f"[motion.{name}]: no cast member {name!r}")
        for i, it in enumerate(items if isinstance(items, list) else [items]):
            path = _vmd_path(ctx, it["vmd"])
            key = f"{name}_{i}_{os.path.splitext(os.path.basename(path))[0]}"[:50]
            act, mact = import_vmd(m, path, key)
            if act is None:
                raise BuildError(f"motion {path}: no bone keys")
            src_from = float(it.get("from", act.frame_range[0]))
            src_to = float(it.get("to", act.frame_range[1]))
            info = {"vmd": os.path.basename(path), "source": [src_from, src_to]}
            if it.get("retime") == "beats":
                tl = ctx.timeline(it, f"motion.{name}[{i}] retime = \"beats\"")
                scale, start_rel, fit = beat_fit(path, tl, ctx.fps, float(it.get("start", 0.0)), src_from)
                start = ctx.frame0 + start_rel
                info.update(fit)
            else:
                scale = float(it.get("scale", 1.0))
                start = ctx.frame(float(it.get("start", 0.0)))
            act_m = mask_action(act, m.arm, it.get("bones", "all"))
            bi, bo = float(it.get("blend_in", 0.0)) * ctx.fps, float(it.get("blend_out", 0.0)) * ctx.fps
            s = nla_strip(m.arm, act_m, start, scale, src_from, src_to, track=f"motion{i}", blend_in=bi,
                          blend_out=bo)
            info["strip"] = [round(s.frame_start, 2), round(s.frame_end, 2)]
            if mact is not None and it.get("morphs", True):
                for mm in m.meshes:
                    if mm.data.shape_keys:
                        nla_strip(mm.data.shape_keys, mact, start, scale, src_from, src_to, track=f"morph{i}")
                info["morphs"] = True
            out.setdefault(name, []).append(info)
            ctx.log("motion", name, json.dumps(info, ensure_ascii=False))
    return out
