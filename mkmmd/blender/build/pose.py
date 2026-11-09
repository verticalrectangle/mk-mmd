"""pose: each cast member's base pose, eased in from the rest pose over the first `settle_frames` of the pre-roll.

[pose.<cast>] keys
  sit = "prop:seat" | {hip, facing, floor_z, pelvis_deg, back_deg}   place the hips on a seat (root follows
        the prop); without sit the model stands where the cast stage put it
  sit_offset = [x, y, z]      slides the hip point on the seat (metres, in the seat prop's frame): sit further forward
  feet = "seat" | "floor" | "prop:feet" | {L = [x, y], R = [...]}      ankle targets (leg IK) at the model's own ankle
        height; a foot may also be a target reference (`{cast = "name", point = [x, y, 0]}`: only x and y count), so a
        standing pose written in the character's own frame moves with it.
        Seated, "seat" (the default) is the seat prop's own feet and "floor" puts the feet in front of the knees;
        standing, "floor" keeps the feet on the floor under the hips (what no `feet` does too) and "seat" is an error
  lift = {L = m, R = m}       each ankle goal that far above its rest height, eased in like the feet (with or without
        `feet`): a foot kicked up, both off the ground with `hips.shift` raised as far (a jump caught at its top)
  lean, turn (deg)            upper body forward lean / turn toward the model's left, on top of the seat's back angle
  lean_share, turn_share = 0.6   the share of each that `upper_body` takes (`upper_body2` takes the rest)
  hips = {shift = [x, y, z], roll = deg, yaw = deg}   the pelvis of a standing or seated body, eased in over the settle: `shift`
        slides it (metres, in the character's own frame like `{cast = "name", point}`: x its left, y behind it, z up: a drop
        bends the knees, the leg IK keeps the feet on `feet`), `roll` drops her left hip (+), `yaw` turns the pelvis toward her
        left about the vertical through it (keys on `center` and `lower_body`, on top of what a seat asks for)
  toes = {L = deg, R = deg}   foot yaw about the vertical through each ankle (the leg IK bone), + toward her left: toes in
        are negative on the left foot and positive on the right
  head = {pitch, yaw, roll, neck = 0.4}   base head rotation (deg, absolute); the neck carries the share `neck` of the
        turn from the chest to the head; perform's gaze adds on top
  [pose.<cast>.hands.<L|R>]   arm IK on the wrist through the twist bones:
     at = target ref, dir = [x, y, z] (hand direction), palm = [x, y, z] (palm normal), pole = target ref
     rest = "prop:edge" (+ along = 0..1, lift = m): the hand lies on an edge use point
     keys = [{t, at, dir, palm, sight}]  moving targets (clip seconds, eased); `sight = true` on consecutive keys holds
        the hand with the HEAD from the first to the last of them (`_sight_ride`: a ring held at the eye stays on it as
        the head leans, tilts, nods and rocks), handed over in SIGHT_RAMP seconds either side; the moves mark the keys of
        a place seen through (mkmmd.core.moves.SIGHTED)
     ride = "<object>" | "cast:NAME.BONE"   the goal FOLLOWS something: an object (a steering wheel), or a bone of this
        character (NAME is its own name: the chest, `cast:rin.upper_body2`). The goal and the elbow pole are then
        bone-parented to the bone (Blender parents to its TAIL) and their keys are computed from the world goal as the bone
        stands in the settled base pose (frame start + settle: this stage's own spine keys; perform's breathing, sway and
        lean / turn / tilt keys ride on top), so the hand is where the world target said at the end of the settle and then
        goes with the chest through every later lean, turn, tilt and sway; two hands riding the same bone never part. A
        bone of an arm is refused (its IK would depend on itself). `keys` ride too: they are read as world points of the
        settled pose
     fingers = "relaxed" | "curled" | "fist" | "flat" | "point" | {index = [a, b, c], middle, ring, little, thumb}
        a table gives the degrees of each finger's first three joints toward the palm; fingers and joints left out
        stay straight (mkmmd.core.fingers)
     grip = "prop:use" | "rest"   the hand HOLDS something: a solver (mkmmd.solvers.grip, on the model's own skin) finds
        the finger rotations and the hand's frame on the prop; the wrist goal follows, the finger rotations are keyed
        like a finger preset (they replace `fingers`) and `ride` still makes the goal follow a prop part.
        grip = "car:wheel"  a prop's use.grip entry, by its type: ring -> a power grip round the rim (prop radius,
           tube); clock = 10 (hours as the character sees the wheel: 12 top, 3 its right; default 10 for L, 2 for R,
           read at the first frame, the hand then rides the wheel), approach = 90 (the palm on the side of the rim that
           faces the character), wrap = -1 (fingers round the outside of the rim), seeds, skin_radius (0.16 m);
           pinch -> thumb-index pad pinch of the strap (prop width, span, length; edge = 0.004 m); pen -> a lateral
           tripod with the writing orientation of the whole hand, which needs `posture` (below) or a `track`; pen
           grips were tuned on one hand shape (two other reference hands also pass their gates; another hand can still miss them)
        grip = "pen:barrel" + track   the pen rides the hand and its nib FOLLOWS A PATH over the page:
           track = "nib" (the project's tracks/nib.json) | "tracks/nib.json" (any .json path), channel = "target":
              {"frames": [Blender frames], "<channel>": [[x, y, z] world metres per frame]}, held before its first and
              after its last frame. One grip is solved (finger rotations, the pen in the wrist, the writing orientation
              of the pen); then on every frame of the build the wrist goal is `pen_frame @ inv(target_in_wrist)` with
              the pen frame's origin on the track (LINEAR keys), and the pen prop is bone-parented to the wrist in the
              solved grip, so its origin (the nib) lands on the track within the IK's accuracy: pin it with a
              `contact` check of obj("pen").loc against the track. The stage logs the worst IK miss over the track.
           posture = {...}  keys of the writing posture (mkmmd.solvers.grip _Posture) the stage completes from the
              scene: nib (the track's mean x, y at paper_z), paper_z (the track's lowest z), shoulder (where the
              shoulder is in the seated pose at the end of the settle), facing (the character's heading), pole (the
              arm IK's elbow pole), upper / fore (the model's arm); table = "table:top" (a prop's use.rest plane ->
              {z, center, radius}) or {z, center, radius}: the forearm stays above it; target = {elevation, azimuth,
              tilt, extension, ulnar: [deg, tolerance]} the writing posture wanted
           wobble = deg | {deg, tau = 18 (frames), seed = 3}  slow random tilt of the pen about the world X and Y
              (the writing orientation is constant otherwise), the way a hand is never perfectly still
        grip = "rest" with rest = "prop:edge" (+ along, offset, lift = 0.0 m, face = "palm" | "back", dir): the relaxed
           hand lies ON the surface; the palm centre is at the rest point, `dir` is the hand's heading
        Every solve is cached (<project>/.mk/cache/grip); the stage output and the log carry each hand's digest
        (contact gaps, penetration, finger clash in mm, seconds, warnings past 3 / 1 / 1 mm).
  fingers = {L = preset, R = preset}  without arm IK
  [[pose.<cast>.drape]]       static cloth draped over a seat (a skirt, a scarf): chain = ["bone", ...] root -> tip (Blender,
        PMX or semantic names), dirs = [[x, y, z], ...] the direction each bone points along in the character's axes (x its
        left, y behind it, z up; the least rotation from the bone's rest direction, whatever the bones above do),
        scale = [1, 0.75, ...] optional length scale per bone (bunched cloth). Eased in over the settle like the rest of
        the base pose; the bones stay keyed (not simulated), so hair collides with their bodies
A table with no keys poses nothing (the log says so), but a prop `[[prop]] wear` puts on that member is still worn. A table
named after no cast member or holding a key this stage does not read is refused (mkmmd.core.tables).
World-axis rotations are keyed with mkmmd.blender.keys (q_child = D_parent^-1 D_want)."""
import json
import math
import os
import time

import bpy
import numpy as np
from mathutils import Euler, Matrix, Quaternion, Vector

from ...core import armreach as AR
from ...core import armtwist as AT
from ...core import fingers as FG
from ...core import fretting as FR
from ...core import gripframe as GF
from ...core import moves as MV
from ...core import pentrack as PT
from ...core import strum as SM
from ...core import wear as WR
from .. import keys as K
from .. import scene as S
from . import BuildError, collection, targets
from . import moves as MOVES
from . import wear as WEAR


def _q(axis, deg):
    return Quaternion(Vector(axis).normalized(), math.radians(deg))


def seat_frame(ctx, spec, offset=None):
    """World seat: hip point, facing (unit, horizontal), floor z, pelvis tilt, back angle, prop (or None). `offset`
    [x, y, z] (m) slides the hip point, in the seat prop's frame (in the world's for a `sit` table)."""
    off = Vector(offset) if offset is not None else Vector()
    if isinstance(spec, str):
        prop_name, _, use = spec.partition(":")
        if prop_name not in ctx.props:
            raise BuildError(f"sit {spec!r}: no prop {prop_name!r}")
        p = ctx.props[prop_name]
        u = p.use("sit", use or None)
        f = p.world_dir(u.get("facing", (0, -1, 0)))
        floor = (p.root.matrix_world @ Vector((0, 0, u.get("floor_z", 0.0)))).z
        return {"hip": p.world(Vector(u["hip"]) + off), "facing": Vector((f.x, f.y, 0)).normalized(),
                "floor_z": floor, "pelvis": float(u.get("pelvis_deg", 6.0)), "back": float(u.get("back_deg", 0.0)),
                "prop": p, "use": u}
    f = Vector(spec.get("facing", (0, -1, 0)))
    return {"hip": Vector(spec["hip"]) + off, "facing": Vector((f.x, f.y, 0)).normalized(),
            "floor_z": float(spec.get("floor_z", 0.0)), "pelvis": float(spec.get("pelvis_deg", 6.0)),
            "back": float(spec.get("back_deg", 0.0)), "prop": None, "use": spec}


def _place_root(m, seat):
    f = seat["facing"]
    yaw = math.atan2(f.x, -f.y)
    want = Matrix.Translation((seat["hip"].x, seat["hip"].y, seat["floor_z"])) @ Matrix.Rotation(yaw, 4, "Z")
    root = m.root
    if seat["prop"] is not None:
        root.parent = seat["prop"].root
        root.matrix_parent_inverse.identity()
        root.matrix_basis = seat["prop"].root.matrix_world.inverted() @ want
    else:
        root.matrix_world = want
    bpy.context.view_layer.update()


def _world_head(arm, bone):
    return arm.matrix_world @ arm.data.bones[bone].head_local


def _reparent(obj, parent):
    """Parent `obj` to `parent` keeping its world placement. matrix_world lags `.location` and `.parent` until the
    depsgraph updates, so a copy taken right after setting them is STALE and parenting would move the object (an
    elbow pole that ended at the world origin): update before copying and before writing it back."""
    bpy.context.view_layer.update()
    mw = obj.matrix_world.copy()
    obj.parent = parent
    obj.matrix_parent_inverse.identity()
    bpy.context.view_layer.update()
    obj.matrix_world = mw


def _ride_bone(m, smap, name, side, ref, f1):
    """`ride = "cast:NAME.BONE"` of a hand: (Blender bone, inverse of the bone's head frame in the world) for the bone as it
    stands in the settled base pose, frame `f1`, when the pose stage's spine keys exist; the character's own placement is
    the one of the current frame, like the world targets'. NAME is the posed character and BONE any name of one of its bones
    that is no part of an arm (the arm IK would depend on itself)."""
    where = f"pose.{name}.hands.{side}"
    who, _, bone = ref[5:].partition(".")
    if who != name or not bone:
        raise BuildError(f"{where}: ride {ref!r} must be cast:{name}.BONE, a bone of the posed character "
                         f"(its chest: cast:{name}.upper_body2)")
    arm = m.arm
    try:
        b = S.resolve_bone(arm, bone)
    except KeyError as e:
        raise BuildError(f"{where}: ride {ref!r}: {e}")
    tops = {smap[f"arm.{s}"] for s in ("L", "R") if f"arm.{s}" in smap}
    p = arm.data.bones[b]
    while p is not None:
        if p.name in tops:
            raise BuildError(f"{where}: ride {ref!r} is part of an arm, whose IK would depend on itself "
                             f"(ride the chest: cast:{name}.upper_body2)")
        p = p.parent
    sc = bpy.context.scene
    keep = sc.frame_current
    sc.frame_set(f1)
    head = arm.evaluated_get(bpy.context.evaluated_depsgraph_get()).pose.bones[b].matrix.copy()
    sc.frame_set(keep)
    return b, (arm.matrix_world @ head).inverted()


def _foot_point(ctx, ref):
    """An ankle target of `feet = {L, R}`: [x, y, (z)] in the world, or any target reference (a point in a character's own
    frame: `{cast = "name", point = [x, y, 0]}`). Only x and y count: the ankle's height is the model's own."""
    if isinstance(ref, (list, tuple)):
        return Vector(list(ref[:2]) + [0.0])
    return targets.point(ctx, ref)


def _axes(m):
    """The model's world axes at rest: lateral (its left), back, up."""
    R = m.arm.matrix_world.to_3x3().normalized()
    return R @ Vector((1, 0, 0)), R @ Vector((0, 1, 0)), Vector((0, 0, 1))


def _spine(m, smap, lean, turn, head, shares=(0.6, 0.6)):
    """Base rotations of the spine and head in the ARMATURE's axes (the model faces -Y): {semantic: rotation the bone
    adds} and the composed chain (D2 chest, Dn neck, Dh head). `shares` = the shares of the lean and the turn that
    `upper_body` takes (the rest is `upper_body2`'s; a model without one takes it all); `head["neck"]` = the share of
    the chest-to-head rotation the neck carries (0.4)."""
    lat, back, up = Vector((1, 0, 0)), Vector((0, 1, 0)), Vector((0, 0, 1))
    has2 = "upper_body2" in smap
    s_lean, s_turn = (shares if has2 else (1.0, 1.0))
    q_up1 = _q(up, turn * s_turn) @ _q(lat, lean * s_lean)
    q_up2 = _q(up, turn * (1.0 - s_turn)) @ _q(lat, lean * (1.0 - s_lean)) if has2 else Quaternion()
    D2 = q_up1 @ q_up2
    Dh = _q(back, head.get("roll", 0.0)) @ _q(up, head.get("yaw", 0.0)) @ _q(lat, head.get("pitch", 0.0))
    Dn = D2.slerp(Dh, float(head.get("neck", 0.4)))
    out = {"upper_body": q_up1, "neck": D2.inverted() @ Dn, "head": Dn.inverted() @ Dh}
    if has2:
        out["upper_body2"] = q_up2
    return out, {"D2": D2, "Dn": Dn, "Dh": Dh}


def _drape(ctx, m, name, specs, f0, f1):
    """[[pose.<cast>.drape]]: each bone of a chain points along a direction of the author's choosing (a skirt over the
    lap, a scarf), by the least rotation from its rest direction, whatever the bones above it do (mkmmd.core.drape).
    Keys ease in from the rest pose over the settle like the rest of the base pose. Returns the number of bones keyed."""
    from ...core import drape as DR
    arm, sc = m.arm, bpy.context.scene
    keep = sc.frame_current
    sc.frame_set(f1)                                       # the pelvis and spine are keyed: read what the chain hangs from
    ae = arm.evaluated_get(bpy.context.evaluated_depsgraph_get())
    n = 0
    for k, dr in enumerate(specs):
        where = f"pose.{name}.drape[{k}]"
        try:
            chain = [S.resolve_bone(arm, b) for b in dr["chain"]]
            dirs = [list(d) for d in dr["dirs"]]
            bones = [arm.data.bones[b] for b in chain]
            par = bones[0].parent
            parent = None if par is None else np.array(
                (ae.pose.bones[par.name].matrix.to_3x3().normalized() @
                 par.matrix_local.to_3x3().normalized().inverted()))
            keys, _ = DR.chain_keys([np.array(b.tail_local - b.head_local) for b in bones], dirs, parent)
        except (KeyError, ValueError) as e:
            raise BuildError(f"{where}: {e}")
        scale = dr.get("scale") or [1.0] * len(chain)
        if len(scale) != len(chain):
            raise BuildError(f"{where}: {len(scale)} scales for {len(chain)} bones")
        for b, q, s in zip(chain, keys, scale):
            K.key_bone_arm(arm, b, [f0, f1], [Quaternion(), Matrix(q.tolist()).to_quaternion()], interp="BEZIER")
            if abs(float(s) - 1.0) > 1e-9:                 # bunched cloth: the bone is shorter
                K.key_prop(arm, f'pose.bones["{b}"].scale', [f0, f1], [1.0, float(s)], index=1)
            n += 1
    sc.frame_set(keep)
    return n


def _chain_points(arm, smap, side, sems):
    """World rest points of a finger chain: the heads of its bones, then its tip (the tip bone's head, else the last
    bone's tail)."""
    bones = [smap[f"{s}.{side}"] for s in sems if f"{s}.{side}" in smap]
    pts = [np.array(_world_head(arm, b)) for b in bones]
    if bones and f"{sems[-1]}.{side}" not in smap:
        pts.append(np.array(arm.matrix_world @ arm.data.bones[bones[-1]].tail_local))
    return pts


_HANDS, _JOINTS = {}, {}         # the hands at rest, and the joints of each shape on them: a solved thumb is solved once


def hand_rest(arm, smap, side):
    """The hand `side` of `arm` at rest in the world, as mkmmd.core.fingers.joints reads it: {"points": {finger: its
    rest chain (`_chain_points`)}, "palm": the palm normal, "tips": {finger: where its skin ends}, "radii": {finger: its
    last segment's skin radius}} (mkmmd.core.fingers.finger_end, from the skin the last bone of each finger moves most;
    a finger without such skin, or a model whose hand the grip export cannot read, ends at its tip bone). None for a
    hand without wrist, index and little finger bones."""
    from ...core import bonemap
    from ..ops_hand import hand_arrays
    if not all(f"{k}.{side}" in smap for k in ("wrist", "index1", "little1")):
        return None
    key = (arm.name, side, tuple(round(v, 6) for row in arm.matrix_world for v in row))
    if key in _HANDS:
        return _HANDS[key]
    points = {f: np.array([list(p) for p in _chain_points(arm, smap, side, sems)])
              for f, sems in bonemap.FINGERS.items() if sum(f"{s}.{side}" in smap for s in sems) >= 2}
    tips, radii = {}, {}
    try:
        a = hand_arrays(arm, side)
    except (KeyError, ValueError):
        a = None
    if a is not None:
        owner = np.argmax(a["W"], axis=1)
        for k, f in enumerate(bonemap.FINGERS):
            last = 1 + 4 * k + 2                       # the chain's third bone: the one the fingertip's skin follows
            if f in points and a["names"][last]:
                tip, r = FG.finger_end(a["heads"][last - 2:last + 2], a["V"][owner == last])
                if tip is not None:
                    tips[f], radii[f] = tip, r
    _HANDS[key] = {"points": points, "palm": np.array(palm_normal(arm, smap, side)), "tips": tips, "radii": radii}
    return _HANDS[key]


def _finger_quats(arm, smap, side, preset):
    """Bone -> local quaternion of each finger joint of the hand `side` in the shape `preset` (a name of
    mkmmd.core.fingers.PRESETS or a table {finger: [a, b, c], spread}): each finger curled toward the palm about its own
    flexion axis (finger direction x palm normal, from the rest pose), its base joint first turned about the palm normal
    by its spread (away from the middle finger; the thumb away from the index); a solved thumb (mkmmd.core.fingers.THUMBS:
    "tuck" folds it over the curled middle finger, "ring" meets the index's tip with it, "cross" lays it across the
    index's tip) solved for this hand's own rest bones and skin (`hand_rest`, mkmmd.core.fingers.joints)."""
    from ...core import bonemap
    try:
        FG.curls(preset), FG.spreads(preset)
    except ValueError as e:
        raise BuildError(str(e))
    hand = hand_rest(arm, smap, side)
    if hand is None:
        return {}
    key = (id(hand), json.dumps(preset, sort_keys=True))
    if key not in _JOINTS:
        try:
            _JOINTS[key] = FG.joints(hand, preset, side)
        except ValueError as e:
            raise BuildError(f"{arm.name} {side} hand: {e}")
    out = {}
    for name, rots in _JOINTS[key].items():
        bones = [smap[f"{s}.{side}"] for s in bonemap.FINGERS[name] if f"{s}.{side}" in smap]
        for b, R in zip(bones[:3], rots):
            B = K.rest_rot(arm, b)
            out[b] = (B.inverted() @ Matrix([list(r) for r in R]) @ B).to_quaternion()
    return out


def _rest_bend(arm, base, fore, fold):
    """The way (unit, armature space) the elbow points from the shoulder-hand line in the pose the arm IK starts from
    (the rest arm with the forearm folded by `fold`, an armature-space rotation about the elbow: `_prebend`), with the
    line from the chain root `base` to the hand: the pole's direction there."""
    bones = arm.data.bones
    head, knee = bones[base].head_local, bones[fore].head_local
    axis = (knee + fold @ (bones[fore].tail_local - knee) - head).normalized()
    return ((knee - head) - axis * (knee - head).dot(axis)).normalized()


def _pole_angle(arm, base, fore, fold):
    """The IK pole angle (radians) that makes this arm's elbow point at its pole. Blender measures it about the line from
    the chain root to the target, from the root bone's X axis, and rigs roll that axis differently on the two sides (MMD
    rigs mirror it), so one fixed angle sends one elbow the wrong way. The angle returned is the one under which the pose
    the solver starts from (the rest arm with the forearm folded by `fold`, an armature-space rotation about the elbow:
    `_prebend`) solves with the pole where that elbow points (`_rest_bend`): the elbow then points at the pole in every
    pose."""
    bones = arm.data.bones
    root = bones[base]
    head, tail = root.head_local, root.tail_local
    knee = bones[fore].head_local
    reach = knee + fold @ (bones[fore].tail_local - knee) - head
    pole = knee + _rest_bend(arm, base, fore, fold) * reach.length
    projected = reach.cross(pole - head).cross(tail - head)
    x_axis = root.matrix_local.to_3x3().col[0]
    angle = x_axis.angle(projected)
    return -angle if x_axis.cross(projected).dot(tail - head) > 0 else angle


WRIST_ROLL = 15.0          # degrees the wrist turns the hand about its own length past the forearm twist bone, at most
ARM_TWIST = (AT.UPPER[0] + 2.0 * MV.TWIST_MARGIN, AT.UPPER[1] - 2.0 * MV.TWIST_MARGIN)   # degrees the upper arm's twist
                           # bone turns under the IK, at most: the human range less twice the moves' margin, which
                           # leaves room inside the range for the IK's own turn of the arm bone (up to ~10 at a hand-off)
ARM_RANGE = (AT.UPPER[0] + MV.TWIST_MARGIN, AT.UPPER[1] - MV.TWIST_MARGIN)   # degrees the upper arm's twist bone turns
                           # whatever drives the arm (a motion strip's own shrug while a hand is let go): the range less
                           # the margin left for the arm bone's own turn
FORE_RANGE = (AT.FOREARM[0] + WRIST_ROLL, AT.FOREARM[1] - WRIST_ROLL)        # degrees the forearm twist bone turns,
                           # whatever drives it: the range less what the wrist may add, so the hand stays in range too


def _twist_limit(arm, bone, side, limits, axis, name, order="AUTO"):
    """A LIMIT_ROTATION named `name` on `bone` holding its turn about its own length (its local Y) within `limits`
    (degrees, + supination / external rotation on either side: mkmmd.core.armtwist.SIGN), `axis` the arm's direction out
    along it at rest (armature space): which way the bone's Y points along the arm decides the sign."""
    k = AT.SIGN[side] * (1.0 if arm.data.bones[bone].matrix_local.to_3x3().col[1].dot(axis) >= 0.0 else -1.0)
    lo, hi = sorted((k * limits[0], k * limits[1]))
    c = arm.pose.bones[bone].constraints.new("LIMIT_ROTATION")
    c.name = name
    c.owner_space = "LOCAL"
    c.use_limit_y = True
    c.min_y, c.max_y = math.radians(lo), math.radians(hi)
    c.euler_order = order
    return c


def _arm_ik(ctx, m, smap, side):
    """The arm rig. The goal empty `<name>_hand.<side>` is the wrist bone's tail and orientation. A position-only IK on
    the forearm (chain up to the upper arm) puts the forearm's tail where the goal says the wrist's head is and bends the
    elbow toward the pole (`_pole_angle`: the pole is where the elbow points, on either side of any rig); the wrist copies
    the goal's rotation. The forearm twist bone, when the rig has one, rolls with the hand (a forearm pronates, a wrist
    only bends; MMD rigs spread the twist bone's roll down the forearm, so a sleeve turns with the hand instead of the
    hand turning inside the cuff) within FORE_RANGE, and the wrist turns the hand on past that at most WRIST_ROLL, so the
    hand stays inside mkmmd.core.armtwist.FOREARM and a goal that asks more leaves the hand short of its roll instead of
    wringing the sleeve (the moves library turns the elbow so that none does: mkmmd.core.moves.arm_for). The upper arm's
    twist bone stays within ARM_RANGE, and within ARM_TWIST while the IK holds the arm (the IK itself never turns it).
    These limits hold whatever drives the arm: a motion strip's own shrug while the hand is let go levels off at them.
    (A single IK that also matched the hand's rotation let the solver swing the elbow away from the pole.)
    Returns (goal, pole, [(bone, constraint name), ...]): the constraints whose influence the settle ramps up."""
    arm = m.arm
    bones = arm.data.bones
    wrist, upper = smap[f"wrist.{side}"], smap[f"arm.{side}"]
    fore = smap.get(f"elbow.{side}") or arm.pose.bones[wrist].parent.name
    n, b = 1, arm.pose.bones[fore]
    while b.parent is not None and b.name != upper:
        b = b.parent
        n += 1
    if b.name != upper:
        raise BuildError(f"{m.name}: {fore} does not hang from {upper}")
    coll = collection("Rig")
    tgt = bpy.data.objects.new(f"{m.name}_hand.{side}", None)
    pole = bpy.data.objects.new(f"{m.name}_elbow.{side}", None)
    reach = bpy.data.objects.new(f"{m.name}_wrist.{side}", None)
    for o, size in ((tgt, 0.05), (pole, 0.04), (reach, 0.02)):
        o.empty_display_size = size
        coll.objects.link(o)
    tgt.rotation_mode = "QUATERNION"
    wrist_rest = bones[wrist].matrix_local.to_3x3().normalized()     # armature space: the offsets below are too
    reach.parent = tgt                                 # where the forearm's tail goes: rigid with the hand's goal
    reach.location = wrist_rest.inverted() @ (bones[fore].tail_local - bones[wrist].tail_local)
    c = arm.pose.bones[fore].constraints.new("IK")
    c.name = "mk_arm_ik"
    c.target, c.pole_target = reach, pole
    fold, local_fold = _prebend(arm, fore)
    c.pole_angle = _pole_angle(arm, upper, fore, fold)
    c.chain_count = n
    c.use_rotation = False
    c.iterations = 500
    cons = [(fore, c.name)]
    r = arm.pose.bones[wrist].constraints.new("COPY_ROTATION")
    r.name = "mk_hand_rot"
    r.target = tgt
    r.owner_space = r.target_space = "WORLD"
    cons.append((wrist, r.name))
    at = arm.pose.bones[wrist].constraints.new("COPY_LOCATION")   # the wrist stays on the forearm's end: rigs put it a
    at.name = "mk_wrist_on_arm"                                    # fraction of a millimetre off the twist bone's axis,
    at.target, at.subtarget, at.head_tail = arm, fore, 1.0          # so rolling that bone would slide the hand off it
    at.owner_space = at.target_space = "WORLD"
    cons.append((wrist, at.name))
    along = bones[wrist].head_local - bones[fore].head_local       # out along the forearm, and the upper arm, at rest
    if f"arm_twist.{side}" in smap:
        tb = arm.pose.bones[smap[f"arm_twist.{side}"]]
        tb.lock_ik_x = tb.lock_ik_y = tb.lock_ik_z = True   # the IK turns the arm bone; this one keeps what it is given
        up = bones[fore].head_local - bones[upper].head_local
        _twist_limit(arm, tb.name, side, ARM_RANGE, up, "mk_arm_twist_range")    # always: a dance's shrug too
        lim = _twist_limit(arm, tb.name, side, ARM_TWIST, up, "mk_arm_twist_limit")
        cons.append((tb.name, lim.name))
    twist = smap.get(f"wrist_twist.{side}")
    if twist and twist != fore:
        aim = bpy.data.objects.new(f"{m.name}_twist.{side}", None)
        aim.empty_display_size = 0.02
        coll.objects.link(aim)
        aim.parent = tgt                               # where the twist bone's X axis points when it follows the hand
        x_twist = bones[twist].matrix_local.to_3x3().col[0]
        aim.location = reach.location + (wrist_rest.inverted() @ x_twist) * 0.2
        t = arm.pose.bones[twist].constraints.new("LOCKED_TRACK")
        t.name = "mk_forearm_twist"
        t.target = aim
        t.lock_axis, t.track_axis = "LOCK_Y", "TRACK_X"
        cons.append((twist, t.name))
        _twist_limit(arm, twist, side, FORE_RANGE, along, "mk_forearm_limit")      # always: IK, strip or neither
        _twist_limit(arm, wrist, side, (-WRIST_ROLL, WRIST_ROLL), along, "mk_wrist_roll", order="YXZ")
    return tgt, pole, cons, (fore, local_fold)


def _prebend(arm, fore):
    """(armature-space rotation, local quaternion) folding the forearm bone AT.PREBEND degrees toward the model's front
    (armature -Y). The IK solver starts from the pose it is given, and from a straight rest arm it bends the elbow
    whichever way the solve drifts: started folded the way an elbow folds, it keeps that fold, so the pole decides alone
    where the elbow points (and mkmmd.core.armtwist predicts its twist from that fold)."""
    b = arm.data.bones[fore]
    along = (b.tail_local - b.head_local).normalized()
    axis = along.cross(Vector((0.0, -1.0, 0.0)))
    if axis.length < 1e-6:                             # a forearm pointing straight ahead at rest: fold it up
        axis = along.cross(Vector((0.0, 0.0, 1.0)))
    B = b.matrix_local.to_3x3().normalized()
    q = Quaternion(axis.normalized(), math.radians(AT.PREBEND)).to_matrix()
    return q, (B.inverted() @ q @ B).to_quaternion()


def palm_normal(arm, smap, side):
    """The hand's palm normal in world at rest (MMD rest pose: palms face down)."""
    hw = _world_head(arm, smap[f"wrist.{side}"])
    n = (_world_head(arm, smap[f"index1.{side}"]) - hw).cross(_world_head(arm, smap[f"little1.{side}"]) - hw)
    n.normalize()
    return -n if n.z > 0 else n


def _hand_matrix(arm, smap, side, at, direction, palm=None):
    """IK goal for a wrist head at `at` with the hand pointing along `direction` and the palm facing `palm` (rolled
    about the hand axis; default: the least rotation from rest). IK goals act on the bone's tail, so the goal is the
    head frame moved along the bone by its length."""
    wrist = smap[f"wrist.{side}"]
    bone = arm.data.bones[wrist]
    B = K.rest_rot(arm, wrist)
    rest_dir = (B @ Vector((0, 1, 0))).normalized()
    q = rest_dir.rotation_difference(Vector(direction).normalized())
    R = q.to_matrix() @ B
    if palm is not None:
        n_local = B.inverted() @ palm_normal(arm, smap, side)
        y = (R @ Vector((0, 1, 0))).normalized()
        cur = (R @ n_local).normalized()
        want = Vector(palm) - Vector(palm).dot(y) * y
        cur = cur - cur.dot(y) * y
        if want.length > 1e-6 and cur.length > 1e-6:
            want.normalize()
            cur.normalize()
            ang = math.atan2(cur.cross(want).dot(y), cur.dot(want))
            R = Quaternion(y, ang).to_matrix() @ R
    M = R.to_4x4()
    M.translation = Vector(at)
    return M @ Matrix.Translation((0.0, bone.length, 0.0))


def _rest_point(p, u, h):
    """World point on a prop's use.rest entry: a plane's centre (+ `offset` in the prop's frame) or a point `along`
    an edge."""
    if u.get("type") == "plane" or "a" not in u:        # a plane: centre + offset in the prop's frame
        return p.world(Vector(u["center"]) + Vector(h.get("offset", (0.0, 0.0, 0.0))))
    return p.world(u["a"]).lerp(p.world(u["b"]), float(h.get("along", 0.5)))      # an edge: a point along it


def _ik_shares(ctx, name, side, ik, f0, f1):
    """(frames, shares) of a hand's arm IK: none at the pre-roll's start, all of it from the settle's end (f1) on, or as
    much as the hand's `ik` keys say ([[t, share], ...]: clip seconds and shares 0..1, eased between keys and held
    outside them; the share at the settle's end is the one the settle reaches)."""
    if ik is None:
        return [f0, f1], [0.0, 1.0]
    where = f"pose.{name}.hands.{side}.ik"
    if not isinstance(ik, list) or not ik or not all(isinstance(k, (list, tuple)) and len(k) == 2 for k in ik):
        raise BuildError(f"{where}: expected [[t, share], ...] (clip seconds, shares 0..1)")
    ts, vs = [float(k[0]) for k in ik], [float(k[1]) for k in ik]
    if any(not 0.0 <= v <= 1.0 for v in vs) or any(b < a for a, b in zip(ts, ts[1:])):
        raise BuildError(f"{where}: shares are 0..1 and the keys in time order")
    frames = [ctx.frame(t) for t in ts]
    later = [(f, v) for f, v in zip(frames, vs) if f > f1]
    return [f0, f1] + [f for f, _ in later], [0.0, float(np.interp(f1, frames, vs))] + [v for _, v in later]


def _key_goal(tgt, mats, par_inv, interp="BEZIER"):
    """Key a hand's goal empty on [(frame, world goal matrix)], read in its parent's frame `par_inv` (a bone ride's: the
    bone as it stands in the settled pose)."""
    frames, locs, rots = [], [], []
    for fr, M in sorted(mats, key=lambda fm: fm[0]):
        Ml = par_inv @ M
        frames.append(fr)
        locs.append(tuple(Ml.translation))
        rots.append(tuple(Ml.to_quaternion()))
    K.key_vec(tgt, "location", frames, locs, interp=interp)
    K.key_vec(tgt, "rotation_quaternion", frames, K.continuous(rots), interp=interp)


SIGHT_RAMP = 0.1          # s: a hand held with the head (keys marked `sight`) is handed to it and back this fast


def _sight_ride(ctx, m, smap, name, side, h, tgt, back, f1):
    """Carry the hand `side` with the head over each run of its keys marked `sight` (a place seen through,
    mkmmd.core.moves.SIGHTED: a ring held at the eye): a second goal `<name>_sight.<side>` riding the head bone holds
    those keys' goals, read in the head's frame as it stands in the settled pose (`_ride_bone`), and the hand's goal
    `tgt` copies it (its `mk_sight` constraint) from the run's first key to its last, handed over in SIGHT_RAMP seconds
    before the first and back after the last. Riding the chest, the ring would slide off the eye as the move leans the
    body and tilts the head, and with every nod on the beat or head rock perform adds; this way it stays on it. Keyed
    again (the goal and the constraint reused) when the moves are compiled again (`free_hands`). Returns the runs
    [(first frame, last frame)]."""
    keys = h.get("keys") or []
    ramp = SIGHT_RAMP * ctx.fps
    runs = []
    for k, prev in zip(keys, [None] + keys[:-1]):
        if not k.get("sight"):
            continue
        fr = ctx.frame(float(k["t"]))
        if prev is not None and prev.get("sight") and runs:
            runs[-1][1] = fr
        elif runs and fr - runs[-1][1] <= 2.0 * ramp:             # two holds too close to hand over between: one
            runs[-1][1] = fr
        else:
            runs.append([fr, fr])
    runs = [(a, b) for a, b in runs if b - a > 1e-6]
    c = tgt.constraints.get("mk_sight")
    if not runs:
        if c is not None:
            c.influence = 0.0
            K.key_prop(tgt, f'constraints["{c.name}"].influence', [f1], [0.0], interp="CONSTANT")
        return []
    arm = m.arm
    hb, head_inv = _ride_bone(m, smap, name, side, f"cast:{name}.head", f1)
    helper = bpy.data.objects.get(f"{m.name}_sight.{side}")
    if helper is None:
        helper = bpy.data.objects.new(f"{m.name}_sight.{side}", None)
        helper.empty_display_size = 0.03
        helper.rotation_mode = "QUATERNION"
        collection("Rig").objects.link(helper)
        attach_to_bone(helper, arm, hb, Matrix())
    mats = [(ctx.frame(float(k["t"])), _hand_matrix(arm, smap, side, targets.point(ctx, k["at"]),
                                                    k.get("dir", h.get("dir", tuple(-back))), k.get("palm", h.get("palm"))))
            for k in keys if k.get("sight")]
    _key_goal(helper, mats, Matrix.Translation((0.0, -arm.data.bones[hb].length, 0.0)) @ head_inv)
    if c is None:
        c = tgt.constraints.new("COPY_TRANSFORMS")
        c.name = "mk_sight"
        c.target = helper
    frames, shares = [], []
    for a, b in runs:
        frames += [a - ramp, a, b, b + ramp]
        shares += [0.0, 1.0, 1.0, 0.0]
    K.key_prop(tgt, f'constraints["{c.name}"].influence', frames, shares)
    return runs


def _key_ik(ctx, arm, name, side, rig, ik, f0, f1, unbent=()):
    """Key a hand's arm rig on its IK shares (`_ik_shares`): each constraint's influence (that share of its full one,
    `rig["full"]`) and the forearm's pre-bend. `unbent` [(frame, frame)]: stretches where a strip bends the elbow and
    the hand is let go to it, the IK easing off and on inside them: the pre-bend is off from one to the other (the IK
    starts from the strip's own bent arm there, and the arm it eases from and to is the strip's alone), back a frame
    outside, where the IK holds the arm. Returns (frames, shares)."""
    frames, shares = _ik_shares(ctx, name, side, ik, f0, f1)
    for (bone, cname), full in rig["full"].items():
        c = arm.pose.bones[bone].constraints[cname]
        K.key_prop(arm, f'pose.bones["{bone}"].constraints["{cname}"].influence', frames, [full * s for s in shares])
        c.influence = full
    bend = dict(zip(frames, shares))
    for a, b in unbent:
        bend = {f: s for f, s in bend.items() if not a - 1.0 < f < b + 1.0}
        bend.update({a - 1.0: 1.0, a: 0.0, b: 0.0, b + 1.0: 1.0})
    fb = sorted(bend)
    K.key_bone_quats(arm, rig["fore"], fb, [tuple(Quaternion().slerp(rig["bend"], bend[f])) for f in fb],
                     interp="BEZIER")
    return frames, shares


def _key_fingers(ctx, arm, smap, side, h, f0, f1, fade=None):
    """Key a hand's fingers: its own (`fingers`, else flat) from the settle's end, then each key's at its time, eased
    between (none before the settle ends). `fade` (IK frames, IK shares, bones): those bones give way by the hand's IK
    share, as its arm does, to the motion strip that plays them while the hand is let go (the strip's curl alone, not
    composed with the hand's: the action blends with Combine)."""
    seq = {f1: _finger_quats(arm, smap, side, h.get("fingers") or "flat")}
    for k in sorted(h.get("keys", []), key=lambda k: float(k["t"])):
        if k.get("fingers"):
            seq[max(ctx.frame(float(k["t"])), f1)] = _finger_quats(arm, smap, side, k["fingers"])
    seq = sorted(seq.items())
    frames = [f0] + [f for f, _ in seq]
    for b in seq[0][1]:
        quats = K.continuous([(1, 0, 0, 0)] + [tuple(q[b]) for _, q in seq])
        if fade is not None and b in fade[2]:
            K.key_bone_quats(arm, b, *_faded(frames, quats, fade[0], fade[1]), interp="BEZIER")
        else:
            K.key_bone_quats(arm, b, frames, quats, interp="BEZIER")


def _eased(frames, values, f, mix):
    """The value at frame f of keys (frames, values), eased in and out between two (a Bezier key's flat handles), held
    outside them; `mix(a, b, w)` blends two values."""
    i = int(np.searchsorted(frames, f, side="right"))
    if i == 0:
        return values[0]
    if i == len(frames):
        return values[-1]
    a, b = frames[i - 1], frames[i]
    w = min(max((f - a) / (b - a), 0.0), 1.0) if b > a else 1.0
    return mix(values[i - 1], values[i], w * w * (3.0 - 2.0 * w))


def _faded(frames, quats, ik_frames, shares):
    """(frames, quaternions) of a bone's keys (frames, quats) scaled by the IK share (ik_frames, shares): on its own
    frames, the shares' and every frame where the share is under one, each rotation that share of the way from none."""
    share = lambda f: _eased(ik_frames, shares, f, lambda a, b, w: a + (b - a) * w)                # noqa: E731
    qs = [Quaternion(q) for q in quats]
    want = {float(f) for f in frames} | {float(f) for f in ik_frames}
    want |= {float(f) for f in range(math.ceil(min(ik_frames)), math.floor(max(ik_frames)) + 1) if share(f) < 1.0 - 1e-6}
    out = sorted(want)
    return out, K.continuous([tuple(Quaternion().slerp(_eased(frames, qs, f, lambda a, b, w: a.slerp(b, w)), share(f)))
                              for f in out])


def _driven(arm):
    """The Blender names of the bones some NLA strip of the armature plays."""
    out = set()
    for tr in (arm.animation_data.nla_tracks if arm.animation_data else ()):
        for st in tr.strips:
            for fc in (st.action.fcurves if st.action else ()):
                if fc.data_path.startswith('pose.bones["'):
                    out.add(fc.data_path.split('"')[1])
    return out


POLE_OUT = 0.55           # m from the shoulder, at least, that a pole on its way between a strip's elbow and home passes
STRIP_MARGIN = 5.0        # degrees: a strip's arm is in range for a let-go hand to join or leave it when its twists keep
                          # this far inside the human range (the IK's arm on the strip's hand differs by a few)


STRAIGHT_FOLD = (10.0, 30.0)   # degrees of elbow fold: an arm folded less than the first shows the side its elbow points
                               # by its upper arm's turn, one folded more than the second by where its elbow is


def _pole_of(S, E, W, turned, out=0.3):
    """A pole `out` metres from the elbow E across the shoulder (S) - wrist (W) line, so the arm IK bends the elbow where
    E is and turns the upper arm as it is turned: across the line toward E for a folded arm, and for one too straight to
    show its side (a motion's arm out in a T: there it swings round on a millimetre) the way `turned` says it points
    (the IK's rest pole direction turned with the upper arm), blended between the folds of STRAIGHT_FOLD."""
    axis = (W - S).normalized()
    across = (E - S) - axis * (E - S).dot(axis)
    side = turned - axis * turned.dot(axis)
    fold = math.degrees((E - S).angle(W - E, 0.0)) if (E - S).length > 1e-6 and (W - E).length > 1e-6 else 0.0
    w = min(max((fold - STRAIGHT_FOLD[0]) / (STRAIGHT_FOLD[1] - STRAIGHT_FOLD[0]), 0.0), 1.0)
    w = w * w * (3.0 - 2.0 * w)
    way = (across.normalized() if across.length > 1e-6 else Vector()) * w + (
        side.normalized() if side.length > 1e-6 else Vector()) * (1.0 - w)
    return E + way.normalized() * out if way.length > 1e-6 else None


def arm_twists(arm, smap, side, pb):
    """(forearm, upper arm, elevation) degrees of the arm `side` of `arm` as its pose bones `pb` stand (an evaluated
    armature's): mkmmd.core.armtwist's twists, the hand against the elbow and the arm's twist bone (else the arm bone)
    against the chest, + supination / external rotation, and how far the upper arm is raised from the chest's down."""
    bones = arm.data.bones
    n = {k: smap.get(f"{k}.{side}") for k in ("arm", "arm_twist", "elbow", "wrist")}
    chest = smap.get("upper_body2") or smap["upper_body"]
    rot = lambda b: np.array(pb[b].matrix.to_3x3().normalized())                                  # noqa: E731
    rest = lambda b: np.array(bones[b].matrix_local.to_3x3().normalized())                        # noqa: E731
    head = lambda b: np.array(bones[b].head_local)                                                 # noqa: E731
    hi = n["arm_twist"] or n["arm"]
    fore = AT.relative_twist(rot(n["elbow"]), rest(n["elbow"]), rot(n["wrist"]), rest(n["wrist"]),
                             head(n["wrist"]) - head(n["elbow"]), side)
    upper = AT.relative_twist(rot(chest), rest(chest), rot(hi), rest(hi), head(n["elbow"]) - head(n["arm"]), side)
    down = rot(chest) @ rest(chest).T @ np.array(AT.DOWN)
    return fore, upper, AT.elevation(np.array(pb[n["elbow"]].head) - np.array(pb[n["arm"]].head), down)


def _strip_samples(ctx, m, smap, side, spans, inv):
    """Where the member's motion strips have the hand `side` over its free `spans` (clip seconds), at their ends and on
    every frame between: {"t", "at", "dir", "palm"} in the member's own frame (`inv`: the inverse of its root's world
    matrix), as mkmmd.core.moves places (the wrist joint, the way the hand points and the way its palm faces), "pole" (a
    pole in the world that bends the arm IK's elbow where the strip has it and turns its upper arm as the strip turns
    it: `_pole_of`) and "ok" (whether the strip keeps the arm's twists in the human range there: `arm_twists`,
    mkmmd.core.moves.overtwist, less STRIP_MARGIN). Each is read in the frame of the bone the goal rides as that bone
    stands in the settled pose, so a strip that turns the chest under the arm carries the goal with it. The arm rig (its
    constraints and the forearm's pre-bend) is muted meanwhile: the strip's arm."""
    arm, rig = m.arm, m.hands[side]
    wrist, upper = smap[f"wrist.{side}"], smap[f"arm.{side}"]
    times = sorted({round(t, 6) for t0, t1 in spans for t in [t0, t1] + [ctx.time(f) for f in range(
        math.ceil(ctx.frame(t0)), math.floor(ctx.frame(t1)) + 1)]})
    act = arm.animation_data.action if arm.animation_data else None
    path = f'pose.bones["{rig["fore"]}"].rotation_quaternion'
    muted = [arm.pose.bones[b].constraints[c] for b, c in rig["full"]] + [
        fc for fc in (act.fcurves if act else ()) if fc.data_path == path]
    was = [o.mute for o in muted]
    n_local = K.rest_rot(arm, wrist).inverted() @ palm_normal(arm, smap, side)
    bend = arm.data.bones[upper].matrix_local.to_3x3().inverted() @ _rest_bend(arm, upper, rig["fore"],
                                                                               _prebend(arm, rig["fore"])[0])
    turn = inv.to_3x3().normalized()
    sc = bpy.context.scene
    out = {k: [] for k in ("t", "at", "dir", "palm", "pole", "ok", "S", "E", "W")}
    try:
        for o in muted:
            o.mute = True
        for t in times:
            f = ctx.frame(t)
            sc.frame_set(math.floor(f), subframe=f - math.floor(f))
            ae = arm.evaluated_get(bpy.context.evaluated_depsgraph_get())
            mw, pb = ae.matrix_world, ae.pose.bones
            X = Matrix()
            if rig["ride_bone"] is not None:
                X = rig["ride_set"] @ (mw @ pb[rig["ride_bone"]].matrix).inverted()
            W = X @ mw @ pb[wrist].matrix
            R = W.to_3x3().normalized()
            out["t"].append(t)
            out["at"].append(list(inv @ W.translation))
            out["dir"].append(list(turn @ (R @ Vector((0.0, 1.0, 0.0)))))
            out["palm"].append(list(turn @ (R @ n_local)))
            S0, E = X @ (mw @ pb[upper].head), X @ (mw @ pb[rig["fore"]].head)
            out["pole"].append(_pole_of(S0, E, W.translation, (X @ mw @ pb[upper].matrix).to_3x3().normalized() @ bend))
            out["ok"].append(MV.overtwist(arm_twists(arm, smap, side, pb), STRIP_MARGIN) <= 0.0)
            out["S"].append(S0)
            out["E"].append(E)
            out["W"].append(W.translation.copy())
    finally:
        for o, w in zip(muted, was):
            o.mute = w
    good = [p for p in out["pole"] if p is not None]                 # a straight arm keeps its neighbour's
    prev = good[0] if good else None
    for i, p in enumerate(out["pole"]):
        out["pole"][i] = prev = p if p is not None else prev
    return out


def _settled(rig, ae, point):
    """A world point of the evaluated armature `ae` read in the frame of the bone the goal rides as it stands in the
    settled pose (`_strip_samples`)."""
    mw = ae.matrix_world
    if rig["ride_bone"] is None:
        return mw @ point
    return rig["ride_set"] @ (mw @ ae.pose.bones[rig["ride_bone"]].matrix).inverted() @ (mw @ point)


def _swivel(S, W, a, b):
    """Radians that turn point a about the line from S to W onto the half-plane of point b."""
    u = (W - S).normalized()
    pa, pb = (a - S) - u * (a - S).dot(u), (b - S) - u * (b - S).dot(u)
    return math.atan2(u.dot(pa.cross(pb)), pa.dot(pb)) if pa.length > 1e-6 and pb.length > 1e-6 else 0.0


def _key_move_poles(ctx, pole, keys, home, pinv):
    """Key the elbow's pole on a hand's keys when one of them holds the elbow its own way (a key's `pole`, a target: a
    move's place such as a lone hand resting on the skirt, mkmmd.core.moves POLES): that pole on its keys and the hand's
    own pole (`home`, world) on the others, eased between them as the goal is; read in the pole's parent frame `pinv`."""
    if not any(k.get("pole") is not None for k in keys):
        return
    want = {}
    for k in sorted(keys, key=lambda k: float(k["t"])):
        want[ctx.frame(float(k["t"]))] = tuple(pinv @ (targets.point(ctx, k["pole"]) if k.get("pole") is not None
                                                       else home))
    frames = sorted(want)
    K.key_vec(pole, "location", frames, [want[f] for f in frames], interp="BEZIER")


def _key_pole(ctx, m, rig, samples, keys, rounds=3):
    """Key the elbow's pole of a hand on a strip: on its keys there, where the strip bends the elbow (`samples["pole"]`),
    and back where it stood on the hand's keys before and after each stretch, so the IK bends the elbow as the strip
    does where it lets go and takes over. The IK aims its elbow at the pole only to some degrees, so at each stretch's
    ends (the IK holding the arm on the strip's hand) the pole is turned about the shoulder-wrist line by what the IK's
    elbow misses the strip's by, `rounds` times, and between them by a share of both. Only for a goal riding a bone (its
    pole rides it too). Returns the stretches."""
    if rig["ride_bone"] is None or samples["pole"][0] is None:
        return 0
    idx = {round(t, 4): i for i, t in enumerate(samples["t"])}
    runs, cur = [], None                                     # each stretch: [key before, [sample indices], key after]
    for i, k in enumerate(keys):
        if k.get("strip"):
            if cur is None:
                cur = [keys[i - 1]["t"] if i else None, [], None]
                runs.append(cur)
            cur[1].append(idx[round(k["t"], 4)])
        elif cur is not None:
            cur[2], cur = k["t"], None
    par_inv = rig["par_inv"]
    ad = rig["pole"].animation_data
    fcs = [ad.action.fcurves.find("location", index=i) if ad and ad.action else None for i in range(3)]

    def keyed(t):
        """The pole where the hand's own keys have it at clip time t (a move's place may hold the elbow its own way,
        `_key_move_poles`), or where it stands: read before the stretches are keyed into it."""
        f = ctx.frame(t)
        return Vector([fc.evaluate(f) if fc is not None else rig["pole_loc"][i] for i, fc in enumerate(fcs)])
    homes = [(keyed(before) if before is not None else None, keyed(after) if after is not None else None)
             for before, _, after in runs]
    turn = [[0.0, 0.0] for _ in runs]                        # the turn at each stretch's two ends

    def outside(S0, P, home):
        """Halfway between the strip's pole P and the hand's own (`home`), pushed out from the shoulder S0: the pole goes
        round the arm on its way, not through it (straight across, the IK's elbow flips where the pole passes the arm's
        line)."""
        d = 0.5 * (P + par_inv.inverted() @ home) - S0
        return tuple(par_inv @ (S0 + d.normalized() * max(d.length, POLE_OUT)))

    def poles():                                             # each stretch on its own: the hand's own keys stay round it
        for (before, ids, after), (a, b), (hb, ha) in zip(runs, turn, homes):
            t0, t1 = samples["t"][ids[0]], samples["t"][ids[-1]]
            at, want = {}, {}
            for i in ids:
                w = (samples["t"][i] - t0) / (t1 - t0) if t1 > t0 else 0.0
                S0, P = samples["S"][i], samples["pole"][i]
                q = Quaternion((samples["W"][i] - S0).normalized(), a + (b - a) * w)
                at[i] = S0 + q @ (P - S0)
                want[ctx.frame(samples["t"][i])] = tuple(par_inv @ at[i])
            if before is not None:
                want.setdefault(ctx.frame(before), tuple(hb))
                want.setdefault(ctx.frame(0.5 * (before + t0)), outside(samples["S"][ids[0]], at[ids[0]], hb))
            if after is not None:
                want[ctx.frame(0.5 * (t1 + after))] = outside(samples["S"][ids[-1]], at[ids[-1]], ha)
                want[ctx.frame(after)] = tuple(ha)
            frames = sorted(want)
            K.key_vec(rig["pole"], "location", frames, [want[f] for f in frames], interp="BEZIER", replace=False)

    sc = bpy.context.scene
    for _ in range(rounds):
        poles()
        for (_, ids, _), ends in zip(runs, turn):
            for j, i in enumerate((ids[0], ids[-1])):
                f = ctx.frame(samples["t"][i])
                sc.frame_set(math.floor(f), subframe=f - math.floor(f))
                ae = m.arm.evaluated_get(bpy.context.evaluated_depsgraph_get())
                E = _settled(rig, ae, ae.pose.bones[rig["fore"]].head)
                ends[j] += _swivel(samples["S"][i], samples["W"][i], E, samples["E"][i])
    poles()
    return len(runs)


def free_hands(ctx):
    """Hands let go to motion strips (docs/design.md: Moves, `free`), once the motion stage has laid the strips: where
    the strips have each let-go hand over its free spans is sampled (`_strip_samples`), the member's moves are compiled
    again with it (mkmmd.core.moves: the goal follows the strip there and joins and leaves it at the human pace), and
    the hand's goal, its IK shares, its fingers and its elbow's pole are keyed again: the IK lets go and takes over on
    the strip's own hand with the elbow where the strip has it, and finger bones a strip plays give way to it while the
    hand is let go. A hand whose arm no strip plays keeps its goal at its rest place. Returns {cast: {side: report}}."""
    out = {}
    sc = bpy.context.scene
    keep = sc.frame_current
    f0, f1 = ctx.start, ctx.start + ctx.settle
    for name, m in ctx.cast.items():
        plan = m.moves
        if not plan or not any(plan["free"].values()):
            continue
        driven = _driven(m.arm)
        smap = S.semantic_map(m.arm)
        sides = [s for s in ("L", "R") if plan["free"][s] and s in m.hands
                 and (m.hands[s]["fore"] in driven or smap.get(f"arm.{s}") in driven)]
        if not sides:
            continue
        sc.frame_set(f0)                                         # targets are read where the pose stage read them
        inv = m.root.matrix_world.inverted()
        strip = {s: _strip_samples(ctx, m, smap, s, plan["free"][s], inv) for s in sides}
        try:
            res = MV.compile(plan["entries"], plan["marks"], plan["beats"], yaw=plan["yaw"], clear=plan["clear"],
                             strip=strip, cloth=plan["cloth"], winks=plan["winks"], downbeats=plan["downbeats"],
                             props=plan["props"])
        except MV.MoveError as e:
            raise BuildError(f"move.{name}: {e}") from None
        sc.frame_set(f0)
        for s in sides:
            rig = m.hands[s]
            keys = res["hands"][s]
            h = dict(rig["h"], keys=MOVES.hand_keys(name, keys), ik=res["ik"].get(s))
            _key_goal(rig["tgt"], _hand_targets(ctx, name, s, h, m.arm, smap, rig["back"], rig["out_dir"], f1),
                      rig["par_inv"])
            _sight_ride(ctx, m, smap, name, s, h, rig["tgt"], rig["back"], f1)
            _key_move_poles(ctx, rig["pole"], h["keys"], rig["pole_w"], rig["pole_inv"])
            ik = h["ik"] or []
            spans = [(ik[i][0], ik[i + 3][0]) for i in range(0, len(ik), 4)]
            frames, shares = _key_ik(ctx, m.arm, name, s, rig, h["ik"], f0, f1,
                                     unbent=[(ctx.frame(a), ctx.frame(b)) for a, b in spans])
            if h.get("fingers") or any(k.get("fingers") for k in h["keys"]):
                _key_fingers(ctx, m.arm, smap, s, h, f0, f1, fade=(frames, shares, driven))
            plays = _strips_of(m.arm, {rig["fore"], smap[f"arm.{s}"]})
            for a, b in spans:
                if not all(any(_full(st, f) for st in plays)
                           for f in np.arange(math.ceil(ctx.frame(a)), math.floor(ctx.frame(b)) + 1)):
                    ctx.log("WARNING", f"pose.{name}.hands.{s}: let go at {a:g}-{b:g} s where no motion strip plays the "
                                       f"arm fully (one eases in or out there, or none plays): the IK lets go of a half-"
                                       f"played arm; let a strip of the arm cover the span, eased in before and out after")
            twisted = _past_range(strip[s], spans)
            if twisted:
                ctx.log("WARNING", f"pose.{name}.hands.{s}: let go to a strip that twists the arm past a human arm's range "
                                   f"at {', '.join(f'{a:g}-{b:g}' for a, b in twisted)} s (mkmmd.core.armtwist): let the "
                                   f"hand go over another stretch of the motion")
            out.setdefault(name, {})[s] = {"spans": [[a, b] for a, b in spans], "samples": len(strip[s]["t"]),
                                           "pole": _key_pole(ctx, m, rig, strip[s], keys),
                                           **({"twisted": twisted} if twisted else {})}
            ctx.log("pose", f"{name}.{s}: let go on the strips over {out[name][s]['spans']}")
    sc.frame_set(keep)
    return out


def _past_range(samples, spans):
    """[[t0, t1], ...] clip seconds where a strip's samples (`_strip_samples`) twist the arm past the human range within
    the spans (t0, t1) the hand is let go to it."""
    out = []
    for t, ok in zip(samples["t"], samples["ok"]):
        if ok or not any(a - 1e-6 <= t <= b + 1e-6 for a, b in spans):
            continue
        if out and t - out[-1][1] < 0.05:
            out[-1][1] = round(t, 3)
        else:
            out.append([round(t, 3), round(t, 3)])
    return out


def _strips_of(arm, bones):
    """The NLA strips of the armature whose action plays one of `bones` (Blender names)."""
    paths = tuple(f'pose.bones["{b}"]' for b in bones)
    return [st for tr in (arm.animation_data.nla_tracks if arm.animation_data else ()) if not tr.mute
            for st in tr.strips if st.action is not None and any(fc.data_path.startswith(paths) for fc in st.action.fcurves)]


def _full(st, f):
    """Whether NLA strip `st` plays at its full influence on frame f: inside it, past its blend in and before its blend
    out (or on its influence curve), or in a hold beyond it that has no blend."""
    if st.mute:
        return False
    a, b = st.frame_start, st.frame_end
    if st.use_animated_influence:
        fc = st.fcurves.find("influence")
        return a - 1e-3 <= f <= b + 1e-3 and fc is not None and fc.evaluate(f) >= 0.999
    if f < a:
        return st.extrapolation == "HOLD" and st.blend_in == 0.0
    if f > b:
        return st.extrapolation in ("HOLD", "HOLD_FORWARD") and st.blend_out == 0.0
    return a + st.blend_in - 1e-3 <= f <= b - st.blend_out + 1e-3


def _hand_targets(ctx, name, side, h, arm, smap, back, out_dir, f1):
    """[(frame, IK goal matrix)] of a hand without a grip: lying on an edge or plane, at a point, or moving keys."""
    mats = []
    if h.get("rest"):
        prop_name, _, use = h["rest"].partition(":")
        p = ctx.props[prop_name]
        u = p.use("rest", use)
        n_w = p.world_dir(u.get("normal", (0, 0, 1)))
        at = _rest_point(p, u, h) + n_w * float(h.get("lift", 0.03))
        d = Vector(h.get("dir", tuple(-back + out_dir * 0.3)))
        mats.append((f1, _hand_matrix(arm, smap, side, at, d, h.get("palm", (0, 0, -1)))))
    if h.get("at") is not None:
        at = targets.point(ctx, h["at"])
        mats.append((f1, _hand_matrix(arm, smap, side, at, h.get("dir", tuple(-back)), h.get("palm"))))
    for k in h.get("keys", []):
        mats.append((ctx.frame(float(k["t"])), _hand_matrix(arm, smap, side, targets.point(ctx, k["at"]),
                                                             k.get("dir", h.get("dir", tuple(-back))),
                                                             k.get("palm", h.get("palm")))))
    if not mats:
        raise BuildError(f"pose.{name}.hands.{side}: give at, rest, grip or keys")
    return mats


def _hand_track(ctx, where, h):
    """(frames, positions) of a hand's `track`: a bare name is the project's tracks/<name>.json, anything ending in
    .json a file path; `channel` (default "target") names the positions."""
    ref = str(h["track"])
    path = ref if ref.endswith(".json") else os.path.join("tracks", f"{ref}.json")
    try:
        return PT.load(ctx.path(path), h.get("channel", "target"))
    except (OSError, ValueError) as e:
        raise BuildError(f"{where}: track {ref!r}: {e}")


def _table_of(ctx, ref, where):
    """The posture's `table` from a prop's use.rest plane ("table:top")."""
    prop_name, _, use = ref.partition(":")
    if prop_name not in ctx.props:
        raise BuildError(f"{where}: posture table {ref!r}: no prop {prop_name!r}")
    p = ctx.props[prop_name]
    u = p.use("rest", use or None)
    if "center" not in u:
        raise BuildError(f"{where}: posture table {ref!r} is not a plane (a use.rest entry with a `center`)")
    return PT.table_spec({"center": tuple(p.world(u["center"])), "radius": u.get("radius")})


def _pen_posture(ctx, m, smap, side, h, positions, pole, where):
    """The solver's writing posture for a pen hand with a track (mkmmd.core.pentrack.posture): the user's `posture`
    keys win; the nib is the track's writing spot, the shoulder is where it really is in the settled seated pose, the
    heading is the character's, the elbow pole is the IK's own, and `table = "prop:use"` reads a prop's rest plane."""
    sc, arm = bpy.context.scene, m.arm
    keep = sc.frame_current
    sc.frame_set(ctx.start + ctx.settle)
    ae = arm.evaluated_get(bpy.context.evaluated_depsgraph_get())
    shoulder = ae.matrix_world @ ae.pose.bones[smap[f"arm.{side}"]].head
    back = ae.matrix_world.to_3x3() @ Vector((0.0, 1.0, 0.0))
    sc.frame_set(keep)
    user = dict(h.get("posture") or {})
    if isinstance(user.get("table"), str):
        user["table"] = _table_of(ctx, user["table"], where)
    if user.get("pole") is None and pole is not None:
        user["pole"] = list(pole)
    return PT.posture(user, positions, (-back.x, -back.y, 0.0), shoulder)


WHEEL_APPROACHES = (0.0, 30.0, 60.0, 90.0)       # palm on the rim's outer and driver sides: the grips a driver uses (deg)
WRIST_BEND_WARN = 35.0                            # deg: a wheel hand bent more than this against its forearm is a WARNING
ARM_BEND_WARN = 55.0                              # deg: a fretting or pick hand bent more than this against its forearm ...
FOREARM_CLIP_WARN = 10.0                          # mm: ... or a pick hand's forearm this deep in the prop's body
REACH_WARN = 0.98                                 # ... or a wrist this share of the arm's length from the shoulder
AVOID_CELL = 0.012                                # m: the prop geometry a forearm keeps clear of, one point per cube this big


def _elbow(S, W, P, a, b):
    """Where the arm IK puts the elbow of an arm from shoulder S to wrist W (upper arm a, forearm b): in the plane of S, W
    and the pole P, on P's side; the straight arm's when W is out of reach. None for a degenerate pole."""
    sw = W - S
    c = float(np.linalg.norm(sw))
    if c < 1e-6:
        return None
    u = sw / c
    if c >= a + b:
        return S + a * u
    v = (P - S) - float((P - S) @ u) * u
    n = float(np.linalg.norm(v))
    if n < 1e-9:
        return None
    ca = (a * a + c * c - b * b) / (2.0 * a * c)
    al = math.acos(max(-1.0, min(1.0, ca)))
    return S + a * (math.cos(al) * u + math.sin(al) * (v / n))


def _arm_frame(ctx, m, smap, side, p, out_dir):
    """The arm in the frame of the prop `p` it grips, as the settled pose has it (frame start + settle: the seat, the lean):
    shoulder, pole, upper arm and forearm lengths, the hand's direction (wrist head -> middle finger base) in the wrist
    bone's own frame, up and outward (`out_dir`, world) in the prop's frame, and the matrices between the prop's frame and
    the world NOW (the grip frame and the pole are placed now)."""
    arm, sc = m.arm, bpy.context.scene
    keep = sc.frame_current
    now = p.root.matrix_world.copy()
    sc.frame_set(ctx.start + ctx.settle)
    dg = bpy.context.evaluated_depsgraph_get()
    ae = arm.evaluated_get(dg)
    to_prop = p.root.evaluated_get(dg).matrix_world.inverted()
    shoulder = to_prop @ (ae.matrix_world @ ae.pose.bones[smap[f"arm.{side}"]].head)
    pole = to_prop @ m.ik[side][1].evaluated_get(dg).matrix_world.translation
    R = to_prop.to_3x3().normalized()
    sc.frame_set(keep)
    hd = {k: np.array(_world_head(arm, smap[f"{k}.{side}"])) for k in ("arm", "elbow", "wrist", "middle1")}
    B = np.array((arm.matrix_world @ arm.data.bones[smap[f"wrist.{side}"]].matrix_local).to_3x3().normalized())
    hand = hd["middle1"] - hd["wrist"]
    return {"S": np.array(shoulder), "P": np.array(pole), "a": float(np.linalg.norm(hd["elbow"] - hd["arm"])),
            "b": float(np.linalg.norm(hd["wrist"] - hd["elbow"])), "hand_local": B.T @ (hand / np.linalg.norm(hand)),
            "up": np.array(R @ Vector((0.0, 0.0, 1.0))), "out": np.array(R @ Vector(out_dir).normalized()),
            "to_prop": np.array(now.inverted()), "to_world": np.array(now)}


def _wheel_for_arm(ctx, where, G, prop, params, arrays, af, label, free_elbow, wraps=(-1, 1)):
    """An arm-chosen wheel grip: the grips a driver uses (palm on the rim's outer or driver side, WHEEL_APPROACHES, the
    `wraps`) are solved, and each is scored by how straight the wrist is on the forearm the arm makes: with the elbow in
    its pole's plane, or, `free_elbow`, at the best elbow the arm can have that is not inward of its shoulder and is below
    it (mkmmd.core.armreach). The straightest that meets the grip's gates wins. With a free elbow the solver is given the
    arm too (in the grip's frame), so it looks for grips the forearm can meet straight. Returns (params, res, rep, info,
    elbow): the elbow in the prop's frame when it was chosen, else None."""
    Gl = af["to_prop"] @ np.asarray(G, float)
    S, P, a, b = af["S"], af["P"], af["a"], af["b"]
    g0, Rg = Gl[:3, 3], Gl[:3, :3] / np.linalg.norm(Gl[:3, :3], axis=0)
    arm = {"shoulder": [float(v) for v in Rg.T @ (S - g0)], "a": float(a), "b": float(b),
           "up": [float(v) for v in Rg.T @ af["up"]], "out": [float(v) for v in Rg.T @ af["out"]]}
    tried = []
    for th in WHEEL_APPROACHES:
        for wrap in wraps:
            cand = dict(params, approach=float(th), wrap=int(wrap), **({"arm": arm} if free_elbow else {}))
            res, rep = ctx.solve("mkmmd.solvers.grip", arrays, {"style": "wheel", "prop": prop, "params": cand},
                                 f"{label}-a{th:.0f}w{wrap:+d}")
            r = rep.get("report", {})
            Wf = Gl @ np.linalg.inv(np.asarray(res["target_in_wrist"], float))      # the wrist head frame, prop frame
            W, h = Wf[:3, 3], Wf[:3, :3] @ af["hand_local"]
            elbows = AR.elbows(S, W, a, b, af["up"], af["out"]) if free_elbow else [_elbow(S, W, P, a, b)]
            scored = [(float(AR.bend(E, W, h)), E) for E in elbows if E is not None]
            bend, E = min(scored, key=lambda t: t[0]) if scored else (None, None)
            gaps = [v.get("gap_mm") for v in (r.get("contacts") or {}).values() if isinstance(v, dict)]
            ok = (all(g is not None and g <= 3.0 for g in gaps) and (r.get("penetration_mm") or 0.0) <= 1.0
                  and (r.get("finger_clash_mm") or 0.0) <= 1.0)
            reach = float(np.linalg.norm(W - S)) / (a + b)
            tried.append((bend is None, not ok, 999.0 if bend is None else bend, cand, res, rep, E, reach))
    tried.sort(key=lambda t: t[:3])
    none, _bad, bend, cand, res, rep, E, reach = tried[0]
    if none:                                              # no allowed elbow on any grip: the first grip, as the pole has it
        E = None
    info = {"approach": cand["approach"], "wrap": cand["wrap"], "wrist_bend_deg": None if none else round(bend, 1),
            "elbow": "chosen" if free_elbow and not none else "pole", "reach": round(reach, 2),
            "tried": [[t[3]["approach"], t[3]["wrap"], None if t[0] else round(t[2], 1)] for t in tried]}
    if reach > 1.0:
        ctx.log("WARNING", f"{where}: the wheel is {reach:.2f} arm lengths away: the arm stretches straight and the hand "
                           "does not reach the rim (move the seat closer, or lean in)")
    return cand, res, rep, info, (E if free_elbow else None)


def _prop_points(p, names, where, cell=AVOID_CELL):
    """The evaluated vertices of the prop's objects `names` (a card's `avoid`) in the prop's frame, one per `cell`-metre
    cube: the geometry a forearm keeps clear of."""
    dg = bpy.context.evaluated_depsgraph_get()
    inv = np.array(p.root.matrix_world.inverted())
    out = []
    for n in names:
        o = bpy.data.objects.get(n)
        if o is None or o.type != "MESH":
            raise BuildError(f"{where}: the card's avoid object {n!r} is not a mesh in the scene")
        e = o.evaluated_get(dg)
        me = e.to_mesh()
        V = np.empty(len(me.vertices) * 3)
        me.vertices.foreach_get("co", V)
        M = inv @ np.array(e.matrix_world)
        out.append(V.reshape(-1, 3) @ M[:3, :3].T + M[:3, 3])
        e.to_mesh_clear()
    P = np.concatenate(out)
    _, keep = np.unique(np.floor(P / cell).astype(np.int64), axis=0, return_index=True)
    return P[np.sort(keep)]


def _carrying_arm(ctx, m, smap, side, p, u, h, G, out_dir, where):
    """The `arm` a neck or strum grip is solved with (mkmmd.solvers.grip.arm_spec, in the grip frame G) and what the build
    needs to place the elbow afterwards: the fretting arm's elbow is the one its pole gives the IK; a pick hand's is chosen
    with the grip (out from the shoulder, below it, its forearm clear of the card's `avoid` objects: over the body's edge),
    unless the hand gives a `pole`. Returns (arm, frame) with frame {af, Gl, g0, Rg, free}."""
    af = _arm_frame(ctx, m, smap, side, p, out_dir)
    Gl = af["to_prop"] @ np.asarray(G, float)
    g0, Rg = Gl[:3, 3], Gl[:3, :3] / np.linalg.norm(Gl[:3, :3], axis=0)
    arm = {"shoulder": [float(c) for c in Rg.T @ (af["S"] - g0)], "a": af["a"], "b": af["b"]}
    free = u.get("type") == "strum" and h.get("pole") is None
    if free:
        arm.update(up=[float(c) for c in Rg.T @ af["up"]], out=[float(c) for c in Rg.T @ af["out"]])
    else:
        arm["pole"] = [float(c) for c in Rg.T @ (af["P"] - g0)]
    if u.get("type") == "strum" and u.get("avoid"):
        arm["avoid"] = np.round((_prop_points(p, u["avoid"], where) - g0) @ Rg, 5).tolist()
    return arm, {"af": af, "Gl": Gl, "g0": g0, "Rg": Rg, "free": free}


GRIP_TURN = 60.0          # degrees: the most `_grip_pole` turns a held grip's elbow round the shoulder-wrist line


def _grip_pole(m, smap, side, goals, pole, f1):
    """Turn the elbow's pole of a hand on a neck or strum grip round the shoulder, about the line to its wrist, the least
    way (5-degree steps, at most GRIP_TURN) that keeps the arm's twists mkmmd.core.moves.TWIST_MARGIN inside the human
    range at every goal of the grip (`goals`: world matrices; the fret changes, a pick hand's stroke ends,
    `_stroke_ends`), as the arm IK will pose them (mkmmd.core.armtwist.twists, the elbow mkmmd.core.armreach.pole_elbow
    gives) with the chest as it stands in the settled pose, frame `f1`; perform's sway and the IK's own few degrees use
    the margin. The grip puts the hand, the elbow only turns its arm: a fretting hand down by the nut turns its forearm
    over past what it can from an elbow hanging under the shoulder, a picking hand whose elbow sits forward of the body's
    edge turns its upper arm in past its range at the end of a down stroke. A turn that would bend the wrist past
    ARM_BEND_WARN counts against it. Returns the degrees turned (0: none needed, or none helps)."""
    arm, bones = m.arm, m.arm.data.bones
    rest = MOVES.arm_rest(m)
    if rest is None or not goals:
        return 0.0
    sc = bpy.context.scene
    keep = sc.frame_current
    sc.frame_set(f1)
    pb = arm.evaluated_get(bpy.context.evaluated_depsgraph_get()).pose.bones
    chest = smap.get("upper_body2") or smap["upper_body"]
    Aw, inv = np.array(arm.matrix_world), np.array(m.root.matrix_world.inverted())
    Rm = inv @ Aw                                                    # armature -> the character's frame
    X = Rm @ np.array(bones[chest].matrix_local) @ np.linalg.inv(np.array(pb[chest].matrix)) @ np.linalg.inv(Aw)
    P_w = np.array(pole.matrix_world.translation)
    S = (X @ np.r_[Aw[:3, :3] @ np.array(pb[smap[f"arm.{side}"]].head) + Aw[:3, 3], 1.0])[:3]
    sc.frame_set(keep)
    at = lambda b, end="head": (Rm @ np.r_[np.array(getattr(bones[smap[b]], f"{end}_local")), 1.0])[:3]   # noqa: E731
    a = float(np.linalg.norm(at(f"elbow.{side}") - at(f"arm.{side}")))
    b = float(np.linalg.norm(at(f"wrist.{side}") - at(f"elbow.{side}")))
    L = float(np.linalg.norm(at(f"wrist.{side}", "tail") - at(f"wrist.{side}")))
    unit = lambda M: M / np.linalg.norm(M, axis=0)                                          # noqa: E731
    R0 = unit((Rm @ np.array(bones[smap[f"wrist.{side}"]].matrix_local))[:3, :3])
    p0 = Rm[:3, :3] @ np.linalg.inv(Aw[:3, :3]) @ np.array(palm_normal(arm, smap, side))
    p0 = p0 / np.linalg.norm(p0)
    hands = []
    for G in goals:
        Gx = X @ np.array(G)
        R = unit(Gx[:3, :3])
        hands.append((Gx[:3, 3] - L * R[:, 1], R[:, 1], R @ R0.T @ p0))
    Px = (X @ np.r_[P_w, 1.0])[:3]
    u = np.mean([W for W, _, _ in hands], axis=0) - S
    u = u / np.linalg.norm(u)

    def score(P):
        worst = 0.0
        for W, d, p in hands:
            E = AR.pole_elbow(S, W, a, b, P)
            fore, upper = AT.twists(rest, side, S, E, W, d, p, P)
            limits = AT.upper_limits(AT.elevation(E - S))
            bend = MV.bend(d, (W - E) / np.linalg.norm(W - E))
            worst = max(worst, AT.excess(fore, AT.FOREARM, MV.TWIST_MARGIN) + AT.excess(upper, limits, MV.TWIST_MARGIN)
                        + max(bend - ARM_BEND_WARN, 0.0))
        return round(worst, 1)

    turns = sorted(range(-int(GRIP_TURN), int(GRIP_TURN) + 1, 5), key=abs)
    best = min(turns, key=lambda t: (score(S + AT.rotation(u, t) @ (Px - S)), abs(t)))
    if best == 0 or score(S + AT.rotation(u, best) @ (Px - S)) >= score(Px):
        return 0.0
    Pn = np.linalg.inv(X) @ np.r_[S + AT.rotation(u, best) @ (Px - S), 1.0]
    pole.matrix_world = Matrix.Translation(Vector(tuple(Pn[:3])))
    return float(best)


def strum_specs(ctx, name, side, u, where):
    """The `[perform.<name>] strum` tables of the hand `side` ([] when there are none), each with the `technique` of the first
    decided with the card's strum entry `u` (mkmmd.core.strum.resolved): a hand plays one way."""
    specs = ((ctx.data.get("perform") or {}).get(name) or {}).get("strum")
    mine = [s for s in ([specs] if isinstance(specs, dict) else list(specs or [])) if s.get("hand", "R") == side]
    if not mine:
        return []
    try:
        tech = SM.technique(mine[0], u)
    except SM.StrumError as e:
        raise BuildError(f"{where}: perform.{name}.strum: {e}") from None
    return [dict(s, technique=tech) for s in mine]


def picked_notes(ctx, name, side, prop, neck, entry, where):
    """[(t, c, w, string)] the strum hand plays with `rhythm = "notes"` (mkmmd.core.strum.pick_notes): the bass line of the
    other hand (`[pose.<cast>.hands.<other>] notes`, gripping this prop's neck), each note on the string the fretting hand
    plays it on, placed on the strum axis of the strum entry `entry`."""
    other = "L" if side == "R" else "R"
    h = ((ctx.section("pose", name) or {}).get("hands") or {}).get(other) or {}
    if h.get("notes") is None:
        raise BuildError(f"{where}: rhythm = \"notes\" plays the bass line of the fretting hand: give "
                         f"[pose.{name}.hands.{other}] notes = {{lane = \"...\" | times = [[t, midi], ...]}}")
    ref = str(h.get("grip", ""))
    fret_prop, _, use = ref.partition(":")
    if fret_prop != prop.name or neck is None:
        raise BuildError(f"{where}: rhythm = \"notes\": the fretting hand must grip the neck of {prop.name!r} "
                         f"(its grip is {ref!r})")
    placed, _keys = bass_line(ctx, h, prop.use("grip", use or None), where)
    return SM.pick_notes(placed, neck, entry)


def _stroke_ends(ctx, name, side, p, u, goal):
    """The goals of a pick hand at the two ends of its widest stroke: its grip's `goal` (a world matrix) moved along the
    strum zone's `across` (`u`, the prop `p`'s use.grip entry) by half the sweep of an accented stroke of the hand's
    `[perform.<name>] strum` (mkmmd.core.strum.params: `span`, `accent_gain`), as the strokes carry the hand; [] when
    perform strums with no hand of this side, or plays with the fingers (the hand stays at its stations)."""
    specs = strum_specs(ctx, name, side, u, f"pose.{name}.hands.{side}")
    if not specs or SM.fingerstyle(specs[0]) or u.get("across") is None:
        return []
    k = SM.params(specs[0])
    half = 0.5 * float(k["span"]) * max(float(k["accent_gain"]), 1.0)
    across = p.world_dir(u["across"])
    across = across / across.length
    return [Matrix.Translation(across * (s * half)) @ goal for s in (-1.0, 1.0)]


def _grip(ctx, m, smap, name, side, h, back, out_dir, pole=None, reference=None):
    """Solve the grip of one hand (docs/design.md: Grips): the grip frame G in the world from the prop's card and its
    current transform, the solver's finger rotations and `target_in_wrist`, and the IK goal that puts the wrist where
    G @ inv(target_in_wrist) says. Returns {goal (Matrix), bones {bone: quaternion}, digest}; a pen hand with a
    `track` also returns `pen` = {prop, T (target_in_wrist), R (the pen's writing orientation in the world), track}, a
    neck hand `ref` (what the next shape's solve takes as its `reference`: mkmmd.solvers.grip_neck, `_neck_grips`)."""
    from ..ops_hand import hand_arrays
    arm, where = m.arm, f"pose.{name}.hands.{side}"
    wrist = smap[f"wrist.{side}"]
    toward = (_world_head(arm, smap["arm.L"]) + _world_head(arm, smap["arm.R"])) / 2      # the character's side
    ref = h["grip"]
    track = pick = None
    auto_wheel = False
    try:
        if ref == "rest":
            if h.get("track"):
                raise BuildError(f"{where}: a track needs a pen grip (grip = \"pen:barrel\"), not \"rest\"")
            if not h.get("rest"):
                raise BuildError(f"{where}: grip = \"rest\" needs rest = \"prop:edge\" (a prop's use.rest point)")
            prop_name, _, use = h["rest"].partition(":")
            p = ctx.props[prop_name]
            u = p.use("rest", use)
            n_w = p.world_dir(u.get("normal", (0, 0, 1)))
            at = _rest_point(p, u, h) + n_w * float(h.get("lift", 0.0))
            G = GF.surface_frame(at, n_w, h.get("dir", tuple(-back + out_dir * 0.3)))
            style, prop, params = "rest", {"surface": "plane"}, {"face": h.get("face", "palm")}
            if h.get("seeds") is not None:
                params["seeds"] = int(h["seeds"])
            info = {"point": [round(v, 3) for v in at]}
        else:
            prop_name, _, use = ref.partition(":")
            if prop_name not in ctx.props:
                raise BuildError(f"{where}: grip {ref!r}: no prop {prop_name!r}")
            p = ctx.props[prop_name]
            u = p.use("grip", use or None)
            hh = h
            if h.get("track"):
                track = _hand_track(ctx, where, h)
                hh = dict(h, posture=_pen_posture(ctx, m, smap, side, h, track[1], pole, where))
            auto_wheel = u.get("type") == "ring" and hh.get("approach", "auto") == "auto"
            if auto_wheel:                                       # chosen for the arm below (_wheel_for_arm)
                hh = {k: v for k, v in hh.items() if k != "approach"}
            style, prop, params = GF.card_style(u, hh)
            if track is not None and style != "pen":
                raise BuildError(f"{where}: a track needs a pen grip, {ref!r} is a {style} grip")
            if u.get("type") == "neck":                          # the neck frame N: on the board under the position fret's wire
                origin, Rn = FR.neck_frame(u, int(hh["fret"]))
                N = np.eye(4)
                N[:3, :3], N[:3, 3] = Rn, origin
                G = np.array(p.root.matrix_world) @ N
                info = {"fret": int(hh["fret"]), "chord": hh["chord"]}
                if reference is not None:                        # the shape the hand comes from (_neck_grips)
                    params["ref"] = reference
            elif u.get("type") == "strum" and hh.get("technique") == "fingers":     # a plucking station: string `string`
                neck = next((g for g in p.card.get("use", {}).get("grip", []) if g.get("type") == "neck"), None)
                if neck is None:
                    raise BuildError(f"{where}: grip {ref!r}: fingerstyle plays the strings of the card's neck entry, and "
                                     f"{p.name!r} has none")
                try:
                    prop, (origin, Rs) = SM.pluck_prop(neck, u, int(hh["string"]))
                except SM.StrumError as e:
                    raise BuildError(f"{where}: grip {ref!r}: {e}") from None
                N = np.eye(4)
                N[:3, :3], N[:3, 3] = Rs, origin
                G = np.array(p.root.matrix_world) @ N
                style, params = "pluck", ({"seeds": int(hh["seeds"])} if hh.get("seeds") is not None else {})
                info = {"technique": "fingers", "string": int(hh["string"]), "anchor": prop["anchor"]["type"]}
            elif u.get("type") == "strum":                       # the pick's pinch frame: its tip on the strings at the strum centre
                tip = float(hh.get("tip", u["pick"].get("tip", 0.008)))
                G = GF.strum_frame(p.world(u["center"]), p.world_dir(u.get("normal", (0, -1, 0))),
                                   p.world_dir(u.get("along", (0, 0, 1))), p.world_dir(u.get("across", (1, 0, 0))), tip,
                                   hh.get("thumb", "neck"))
                hold = GF.hold_of(u, hh)
                G = GF.hand_frame(G, hold)                      # the hand's own frame; the pick stays in the strum frame
                info = {"thumb": hh.get("thumb", "neck"), "tip_mm": round(tip * 1000, 1), "hold": hold}
                pick = u["pick"].get("object")
                if u.get("palm"):                               # the heel of the hand over the card's patch (the bridge)
                    pm, Gi = u["palm"], np.linalg.inv(np.asarray(G, float))
                    turn = lambda v: GF.unit(Gi[:3, :3] @ np.asarray(p.world_dir(v), float)).tolist()      # noqa: E731
                    params["palm"] = {"center": (Gi @ np.r_[np.asarray(p.world(pm["center"]), float), 1.0])[:3].tolist(),
                                      "normal": turn(pm["normal"]), "axes": [turn(u["across"]), turn(u["along"])],
                                      "size": [float(v) for v in pm["size"]], "lift": float(pm.get("lift", 0.012))}
                    info["palm"] = True
            elif style == "wheel":
                G, info = GF.ring_frame(p.world(u["center"]), p.world_dir(u["axis"]), u["radius"],
                                        p.world_dir((0, 0, 1)), GF.clock_of(h, side), toward)
                info = {"clock": GF.clock_of(h, side), "axis_flipped": info["flipped"]}
            elif style == "pinch":
                G = GF.pinch_frame(p.world(u["center"]), p.world_dir(u["axis"]), p.world_dir(u["normal"]),
                                   u.get("span", 0.0), params.get("edge", 0.004), toward)
                info = {}
            else:                                                # the pen: the prop's own frame
                mw = p.root.matrix_world
                G = GF.frame_matrix(*(np.array(mw.to_3x3().normalized().col[i]) for i in range(3)),
                                    np.array(mw.translation))
                info = {}
    except (ValueError, KeyError) as e:
        raise BuildError(f"{where}: grip {ref!r}: {e}")
    arrays = hand_arrays(arm, side, float(h.get("skin_radius", 0.16)))
    t0 = time.time()
    if auto_wheel and style == "wheel" and side in m.ik:
        af = _arm_frame(ctx, m, smap, side, p, out_dir)
        params, res, rep, chose, elbow = _wheel_for_arm(ctx, where, G, prop, params, arrays, af, f"grip-{name}-{side}",
                                                        free_elbow=h.get("pole") is None,
                                                        wraps=(int(h["wrap"]),) if h.get("wrap") is not None else (-1, 1))
        info.update(chose)
        W = (af["to_prop"] @ np.asarray(G, float) @ np.linalg.inv(np.asarray(res["target_in_wrist"], float)))[:3, 3]
        u = (W - af["S"]) / np.linalg.norm(W - af["S"])
        foot = None if elbow is None else af["S"] + float((elbow - af["S"]) @ u) * u
        if foot is not None and np.linalg.norm(elbow - foot) > 1e-3:     # the pole goes where the chosen elbow points
            pole_l = elbow + 2.0 * (elbow - foot)                       # (a straight arm has no bend to aim)
            m.ik[side][1].matrix_world = Matrix.Translation(Vector(tuple((af["to_world"] @ np.r_[pole_l, 1.0])[:3])))
            info["pole"] = [round(float(v), 3) for v in pole_l]
        bend = chose["wrist_bend_deg"]
        if bend is None or bend > WRIST_BEND_WARN:
            ctx.log("WARNING", f"{where}: " + (f"the wrist bends {bend} deg against the forearm on the best grip of the "
                                               f"wheel (over {WRIST_BEND_WARN})" if bend is not None else
                                               "no grip of the wheel leaves the elbow out and below the shoulder")
                    + ": try another clock, move the seat, or give no `pole` so the elbow is chosen with the grip")
    elif style != "rest" and u.get("type") in ("neck", "strum") and side in m.ik:     # solved with the arm that carries it
        arm_in, fr = _carrying_arm(ctx, m, smap, side, p, u, h, G, out_dir, where)
        res, rep = ctx.solve("mkmmd.solvers.grip", arrays, {"style": style, "prop": prop, "params": dict(params, arm=arm_in)},
                             f"grip-{name}-{side}")
        r = rep.get("report", {})
        info.update({k: r[k] for k in ("wrist_bend_deg", "reach", "forearm_clear_mm", "knuckles_mm") if r.get(k) is not None})
        if style == "neck":                                        # each pad on its string (across) where it is pressed (along)
            info["pads_mm"] = {f: [c["across_mm"], c["dx_mm"]] for f, c in (r.get("contacts") or {}).items() if "dx_mm" in c}
            if r.get("from_ref"):
                info["from_ref"] = r["from_ref"]
        info["elbow"] = "chosen" if fr["free"] and r.get("elbow") is not None else "pole"
        if fr["free"] and r.get("elbow") is not None:              # the pole goes where the chosen elbow points
            S_ = fr["af"]["S"]
            elbow = fr["Rg"] @ np.asarray(r["elbow"], float) + fr["g0"]
            W = (fr["Gl"] @ np.linalg.inv(np.asarray(res["target_in_wrist"], float)))[:3, 3]
            d = (W - S_) / np.linalg.norm(W - S_)
            foot = S_ + float((elbow - S_) @ d) * d
            if np.linalg.norm(elbow - foot) > 1e-3:
                pole_l = elbow + 2.0 * (elbow - foot)
                m.ik[side][1].matrix_world = Matrix.Translation(Vector(tuple((fr["af"]["to_world"] @ np.r_[pole_l, 1.0])[:3])))
                info["pole"] = [round(float(v), 3) for v in pole_l]
        bend, clear, reach = r.get("wrist_bend_deg"), r.get("forearm_clear_mm"), r.get("reach")
        if bend is not None and bend > ARM_BEND_WARN:
            ctx.log("WARNING", f"{where}: the wrist bends {bend} deg against its forearm (over {ARM_BEND_WARN}): the arm "
                               "cannot bring the hand to the grip straight (move the prop closer, or give the hand a `pole`)")
        if clear is not None and clear < -FOREARM_CLIP_WARN:
            ctx.log("WARNING", f"{where}: the forearm passes {-clear:.0f} mm into {p.name!r} on its way to the strings: "
                               "the arm is too short to come over the body's edge there (wear it lower, or further to the "
                               "side the arm comes from: [[prop]] wear.at)")
        if reach is not None and reach > REACH_WARN:
            ctx.log("WARNING", f"{where}: the wrist is {reach:.2f} arm lengths from the shoulder: the arm stretches "
                               "straight (bring the prop closer: [[prop]] wear.at, neck_deg)")
    else:
        res, rep = ctx.solve("mkmmd.solvers.grip", arrays, {"style": style, "prop": prop, "params": params},
                             f"grip-{name}-{side}")
    seconds = round(time.time() - t0, 1)
    r = rep.get("report", {})
    contacts = {k: v.get("gap_mm") for k, v in (r.get("contacts") or {}).items() if isinstance(v, dict)}
    digest = {"style": style, "contacts_mm": contacts, "penetration_mm": r.get("penetration_mm"),
              "finger_clash_mm": r.get("finger_clash_mm"), "seconds": seconds, **info}
    warn = [f"contact {k} {v} mm" for k, v in contacts.items() if v is not None and v > 3.0]
    warn += [f"{k} {r[k]} mm" for k in ("penetration_mm", "finger_clash_mm") if (r.get(k) or 0.0) > 1.0]
    if warn:
        digest["warnings"] = warn
        ctx.log("WARNING", where, "grip misses its gates:", "; ".join(warn))
    _W, goal = GF.wrist_goal(G, res["target_in_wrist"], arm.data.bones[wrist].length)
    bones = {str(b): tuple(float(x) for x in q) for b, q in zip(res["bones"], res["quats"])}
    out = {"goal": K.mat(goal), "bones": bones, "digest": digest}
    if pick:                                                     # the object the hand carries (a pick): modelled in the grip frame
        if pick not in bpy.data.objects:
            raise BuildError(f"{where}: grip {ref!r}: the card's pick object {pick!r} is not in the scene")
        T = np.asarray(res["target_in_wrist"], float)                # the hand's frame in the wrist; the pick's own in it
        if info.get("hold") == "across":
            T = T @ np.linalg.inv(GF.HOLD_TURN)
        out["pick"] = {"object": bpy.data.objects[pick], "T": T}
    if "pose_names" in res:                                      # a plucking station: each finger's poses (grip style pluck)
        out["poses"] = {str(n): {str(b): tuple(float(x) for x in q) for b, q in zip(bs, qs)}
                        for n, bs, qs in zip(res["pose_names"], res["pose_bones"], res["pose_quats"])}
        out["travel"] = {str(f): {str(p): float(u) for p, u in v.items()} for f, v in (r.get("travel") or {}).items()}
        out["curves"] = {str(f): {str(p): [float(u) for u in us] for p, us in v.items()} for f, v in (r.get("curves") or {}).items()}
    if track is not None:
        if "frame_world_quat" not in rep:
            raise BuildError(f"{where}: the pen solve returned no writing orientation (frame_world_quat)")
        digest["writing"] = r.get("writing")
        digest["track"] = {"frames": [int(track[0][0]), int(track[0][-1])],
                           "nib_mm": [round(float(v) * 1000, 1) for v in np.ptp(track[1], axis=0)]}
        out["pen"] = {"prop": p, "T": np.asarray(res["target_in_wrist"], float),
                      "R": PT.quat_matrix(rep["frame_world_quat"]), "track": track}
    if style == "neck":
        out["ref"] = {"angles": r["angles_deg"], "target_in_wrist": np.asarray(res["target_in_wrist"], float).tolist(),
                      "targets": {t["finger"]: [t["string"], t["fret"]] for t in prop["targets"]}, "position": int(hh["fret"])}
    return out


def _grip_card(ctx, h):
    """(prop, use.grip entry) of a hand's `grip = "prop:use"`, or (None, None) for `rest` and unknown names (`_grip` reports those)."""
    ref = h.get("grip")
    if not isinstance(ref, str) or ref == "rest":
        return None, None
    prop_name, _, use = ref.partition(":")
    p = ctx.props.get(prop_name)
    if p is None:
        return None, None
    try:
        return p, p.use("grip", use or None)
    except BuildError:
        return p, None


def lane_hits(ctx, h, where):
    """The [audio] hits file's lanes when a fretting hand's `notes` name a `lane` (mkmmd.core.lanes), else None."""
    from ...core import lanes as LN
    spec = h.get("notes")
    if not (hasattr(spec, "get") and spec.get("lane") is not None):
        return None
    hits_file = (ctx.data.get("audio") or {}).get("hits")
    if not hits_file:
        raise BuildError(f"{where}: notes lane {spec['lane']!r} reads the [audio] hits file (write it with "
                         f"`mk timeline hits --notes NAME=STEM:LO-HI` and name it in [audio] hits)")
    try:
        return LN.load(ctx.path(hits_file))
    except LN.LaneError as e:
        raise BuildError(f"{where}: {e}") from None


def bass_line(ctx, h, entry, where):
    """(placed notes, hand keys) of a fretting hand's `notes` (mkmmd.core.fretting.line) on its neck entry: the times from the
    table's `times`, or from the `lane` it names in the [audio] hits file (`lane_hits`)."""
    hits = lane_hits(ctx, h, where)
    try:
        return FR.line(h["notes"], entry, hits, float(h.get("move", FR.MOVE)))
    except FR.FrettingError as e:
        raise BuildError(f"{where}: {e}") from None


def _neck_grips(ctx, m, smap, name, side, h, back, out_dir, pole):
    """The grips of a fretting hand over time (grip = "guitar:neck"): its `fret` (the position, under the index finger) and
    `chord` at the start, then `keys = [{t, fret, chord, move}]`: the shape that is in place at clip time t, reached over `move`
    seconds (default 0.12) before it. A key without `fret` or `chord` keeps the previous one; without `fret` and `chord` in the
    hand table the first key is the start. `notes` (a bass line, `bass_line`) makes the keys instead: a shape of one finger
    per note, the first note's from the start (mkmmd.core.fretting.hand_states). Every distinct (fret, chord) is solved once
    (cached), in time order, each from the shape it follows (its `reference`, mkmmd.solvers.grip_neck): the first shape solved
    at its position, so that every shape there keeps that hand and moves only its own fingers, or, at a new position, the
    shape before it (the hand slides there). A change passes through both shapes eased off the strings (`FR.lifted`, solved
    from them): the fingers lift over the first FR.LIFT of the move, the hand moves (along the neck to a new position) and
    they land over the last FR.LIFT (keyed by FR.change_keys). Returns the first grip with `changes` = [(landing frame, move
    frames, goal, bones, via)], via = [(share of the move, goal, bones)] the keys inside the move, `states` (the shapes in
    time) and every solved shape's digest in `digest["states"]` (and the line's numbers in `digest["notes"]`)."""
    where = f"pose.{name}.hands.{side}"
    entry = _grip_card(ctx, h)[1]
    line = None
    try:
        states, placed = FR.hand_states(h, entry, lane_hits(ctx, h, where))
    except FR.FrettingError as e:
        raise BuildError(f"{where}: {e}") from None
    if placed is not None:
        line = {"notes": len(placed), "shapes": len(states), "strings": sorted({n["string"] for n in placed}),
                "positions": sorted({n["position"] for n in placed}), "open": sum(1 for n in placed if not n["fret"]),
                "frets": [min(n["fret"] for n in placed), max(n["fret"] for n in placed)],
                "span_s": [round(placed[0]["t"], 3), round(placed[-1]["t"], 3)]}
    solved, anchor, n_str = {}, {}, len(entry["strings"])

    def grip_of(fret, chord, after):
        key = json.dumps([fret, chord], sort_keys=True)
        if key not in solved:
            solved[key] = _grip(ctx, m, smap, name, side, dict(h, fret=fret, chord=chord), back, out_dir, pole,
                                reference=None if after is None else after["ref"])
            anchor.setdefault(fret, solved[key])
        return solved[key]

    seq, prev = [], None
    for t, fret, chord, move in states:
        g = grip_of(fret, chord, anchor.get(fret, None if prev is None else prev[2]))
        via = []
        if prev is not None and json.dumps(prev[:2], sort_keys=True) != json.dumps([fret, chord], sort_keys=True):
            via = [(FR.LIFT, grip_of(prev[0], FR.lifted(prev[1], n_str), prev[2])),
                   (1.0 - FR.LIFT, grip_of(fret, FR.lifted(chord, n_str), g))]
        seq.append((t, move, g, via))
        prev = (fret, chord, g)
    out = dict(seq[0][2])
    out["changes"] = [(ctx.frame(t), max(1.0, move * ctx.fps), g["goal"], g["bones"],
                       [(s, v["goal"], v["bones"]) for s, v in via]) for t, move, g, via in seq[1:]]
    out["digest"] = dict(seq[0][2]["digest"], states=[g["digest"] for g in solved.values()])
    out["states"] = states                                   # the shapes in time: the strings they hold (mkmmd.core.strings)
    if line is not None:
        out["digest"]["notes"] = line
    return out


def _pluck_grips(ctx, m, smap, name, side, h, back, out_dir, pole):
    """The stations of a strum hand that plays with the fingers (`[perform.<name>] strum` technique "fingers", grip =
    "<prop>:strum"): one grip of style pluck (mkmmd.solvers.grip_pluck) per string the bass line plays (`picked_notes`,
    planned as perform plays it), each solved with the arm, the first string played solved last so that its chosen elbow
    places the pole. Returns the first string's grip (the hand settles there) with `pluck` = {"stations": {string: {goal,
    bones, poses}}, "travel": {string: {finger: {pose: m}}} (where the fingertip is across the string in each pose: what
    times a stroke, mkmmd.core.strum.pluck_keys), "first": string} and every station's digest in `digest["stations"]`."""
    where = f"pose.{name}.hands.{side}"
    p, u = _grip_card(ctx, h)
    specs = strum_specs(ctx, name, side, u, where)
    neck = next((g for g in p.card.get("use", {}).get("grip", []) if g.get("type") == "neck"), None)
    if not all(SM.picking(s) for s in specs):
        raise BuildError(f"{where}: perform.{name}.strum plays with the fingers (technique \"fingers\", the card's default "
                         f"for {p.name!r}), which plucks the fretting hand's bass line: give every table of this hand "
                         f"rhythm = \"notes\", or technique = \"pick\" to strum")
    notes = picked_notes(ctx, name, side, p, neck, u, where)
    tl = ctx.timeline(specs[0], f"perform.{name}.strum")
    try:
        strokes = sorted((s for sp in specs for s in SM.plan(sp, tl, ctx.duration, notes)), key=lambda s: s.t)
    except SM.StrumError as e:
        raise BuildError(f"perform.{name}.strum ({side}): {e}") from None
    order = list(dict.fromkeys(s.string for s in strokes)) or [int(notes[0][3]) if notes else len(neck["strings"])]
    solved = {}
    for s in reversed(order):
        solved[s] = _grip(ctx, m, smap, name, side, dict(h, technique="fingers", string=s), back, out_dir, pole)
    out = dict(solved[order[0]])
    out["pluck"] = {"stations": {s: {"goal": g["goal"], "bones": g["bones"], "poses": g["poses"]} for s, g in solved.items()},
                    "travel": {s: g["travel"] for s, g in solved.items()}, "curves": {s: g["curves"] for s, g in solved.items()},
                    "first": order[0]}
    out["digest"] = dict(out["digest"], stations=[solved[s]["digest"] for s in order])
    return out


def _pen_goals(ctx, m, wrist, h, grip):
    """The pen rides the wrist bone in the solved grip, and the wrist goes where the pen frame on the track says:
    [(frame, IK goal matrix)] for every frame of the build. The writing orientation is the solver's (docs/design.md:
    Grips); `wobble = deg` or {deg, tau (frames), seed} tilts it slowly (the hand is never perfectly still)."""
    pen, arm = grip["pen"], m.arm
    frames = ctx.frames
    pos = PT.positions_at(pen["track"][0], pen["track"][1], frames)
    wob = h.get("wobble") or {}
    wob = {"deg": wob} if isinstance(wob, (int, float)) else wob
    tilt = PT.tilt_wobble(len(frames), float(wob.get("deg", 0.0)), float(wob.get("tau", 18.0)), int(wob.get("seed", 3)))
    R = PT.pen_rotations(pen["R"], tilt if wob.get("deg") else None, len(frames))
    goals = PT.goals(PT.pen_frames(pos, R), pen["T"], arm.data.bones[wrist].length)
    attach_to_bone(pen["prop"].root, arm, wrist, K.mat(pen["T"]))
    return [(int(f), K.mat(g)) for f, g in zip(frames, goals)]


def run(ctx):
    moves = MOVES.apply(ctx)                              # [[move.<cast>]] -> the pose and perform tables
    ctx.check_tables("pose")
    out = {"moves": moves} if moves else {}
    f0, f1 = ctx.start, ctx.start + ctx.settle
    probe = {}                                            # (cast, side) -> the frames its IK miss is measured on
    bpy.context.scene.frame_set(f0)                       # vehicles may have moved the props: one consistent frame
    for name, m in ctx.cast.items():
        spec = ctx.section("pose", name)
        if not spec:                                      # no keys, nothing to pose: but a prop it wears is still put on it
            worn = WEAR.apply(ctx, name, m)
            if worn:
                out[name] = {"wear": worn}
            elif name in ctx.section("pose"):
                ctx.log("WARNING", f"pose.{name}: the table is empty, so nothing is posed (give it keys, or remove it)")
            continue
        arm = m.arm
        smap = S.semantic_map(arm)
        info = {}
        seat = seat_frame(ctx, spec["sit"], spec.get("sit_offset")) if spec.get("sit") else None
        if seat:
            _place_root(m, seat)
            m.seat = seat
        lat, back, up = _axes(m)
        # hips onto the seat; `hips` slides and turns the pelvis of a standing body too
        hp = spec.get("hips") or {}
        hip_move = m.root.matrix_world.to_3x3().normalized() @ Vector(hp.get("shift", (0.0, 0.0, 0.0)))   # her frame -> world
        hip_turn = (_q((0, 0, 1), float(hp.get("yaw", 0.0))) @ _q((0, 1, 0), float(hp.get("roll", 0.0)))
                    @ _q((1, 0, 0), float(hp.get("pitch", 0.0))))   # the armature's axes; pitch + tips it forward
        centre = Vector()                                  # the hips' offset (world) at the end of the settle, for perform
        if seat:
            hips = (_world_head(arm, smap["leg.L"]) + _world_head(arm, smap["leg.R"])) / 2
            off = seat["hip"] - hips + hip_move
            K.key_bone_locs(arm, smap["center"], [f0, f1], [(0, 0, 0), tuple(off)])
            K.key_bone_arm(arm, smap["lower_body"], [f0, f1], [Quaternion(), hip_turn @ _q((1, 0, 0), -seat["pelvis"])],
                           interp="BEZIER")
            info["hip_offset"] = [round(v, 4) for v in off]
            centre = off
        elif hp:
            if hip_move.length > 0.0:
                K.key_bone_locs(arm, smap["center"], [f0, f1], [(0, 0, 0), tuple(hip_move)])
                centre = hip_move
            if hp.get("roll") or hp.get("yaw") or hp.get("pitch"):
                K.key_bone_arm(arm, smap["lower_body"], [f0, f1], [Quaternion(), hip_turn], interp="BEZIER")
            info["hips"] = {k: hp[k] for k in hp}
        # spine and head
        lean = float(spec.get("lean", 0.0)) + (-seat["back"] if seat else 0.0)
        base, chain = _spine(m, smap, lean, float(spec.get("turn", 0.0)), spec.get("head", {}),
                             (float(spec.get("lean_share", 0.6)), float(spec.get("turn_share", 0.6))))
        for sem, q in base.items():
            K.key_bone_arm(arm, smap[sem], [f0, f1], [Quaternion(), q], interp="BEZIER")
        m.base = {"spine": base, "chain": chain, "center": centre, "feet": {}}
        # feet on the floor / pedals (leg IK bones move to the ankle targets)
        feet = spec.get("feet", "seat" if seat else None)
        if feet == "seat" and not seat:
            raise BuildError(f"pose.{name}.feet = \"seat\" is the feet of the seat the character sits on, and there is no `sit`: "
                             f"write feet = \"floor\" (standing: the feet stay on the floor under the hips), \"prop:feet\" or "
                             f"{{L = [x, y], R = [x, y]}}")
        lift = spec.get("lift") or {}
        if not isinstance(lift, dict) or set(lift) - {"L", "R"}:
            raise BuildError(f"pose.{name}.lift = {lift!r}: {{L = m, R = m}}, how far above its rest height each ankle goes")
        if feet or lift:
            if not feet:
                pts = {"L": None, "R": None}
            elif feet == "seat" or feet == "floor":
                fd = None
                if seat and seat["prop"] is not None and feet == "seat":
                    for key in (seat["use"].get("name"), None):     # the seat's own feet, else the only feet
                        try:
                            fd = seat["prop"].use("feet", key)
                            break
                        except BuildError:
                            continue
                pts = {s: (seat["prop"].world(list(fd[s][:2]) + [0.0]) if fd else None) for s in ("L", "R")}
            elif isinstance(feet, str):
                prop_name, _, use = feet.partition(":")
                if prop_name not in ctx.props:
                    raise BuildError(f"pose.{name}.feet = {feet!r}: no prop {prop_name!r} (props: {sorted(ctx.props)})")
                p = ctx.props[prop_name]
                fd = p.use("feet", use or None)
                pts = {s: p.world(fd[s][:2] + [0.0]) for s in ("L", "R")}
            else:
                if not isinstance(feet, dict) or any(s not in feet for s in ("L", "R")):
                    raise BuildError(f"pose.{name}.feet: a table gives both feet, {{L = [x, y], R = [x, y]}} (or a target each)")
                pts = {s: _foot_point(ctx, feet[s]) for s in ("L", "R")}
            for s in ("L", "R"):
                ik = smap.get(f"leg_ik.{s}")
                if not ik:
                    continue
                rise = float(lift.get(s, 0.0))
                if pts[s] is None and seat is None and not rise:  # standing on `floor`: the foot stays under the hips
                    continue
                rest = _world_head(arm, ik)
                if pts[s] is None and seat is not None:          # in front of the knee, legs as relaxed as the seat allows
                    knee_fwd = seat["hip"] - back * (rest - _world_head(arm, smap[f"leg.{s}"])).length * 0.55
                    side_off = lat * (0.1 if s == "L" else -0.1)
                    tgt = Vector((knee_fwd.x + side_off.x, knee_fwd.y + side_off.y, 0.0))
                else:
                    tgt = pts[s] if pts[s] is not None else rest
                goal = Vector((tgt.x, tgt.y, rest.z + rise))        # the model's own ankle height, `lift` above it
                K.key_bone_locs(arm, ik, [f0, f1], [(0, 0, 0), tuple(goal - rest)])
                m.base["feet"][s] = goal - rest
        for s, deg in (spec.get("toes") or {}).items():          # foot yaw about the vertical through the ankle (L and R)
            if smap.get(f"leg_ik.{s}") and float(deg):
                K.key_bone_arm(arm, smap[f"leg_ik.{s}"], [f0, f1], [Quaternion(), _q((0, 0, 1), float(deg))], interp="BEZIER")
                m.base.setdefault("toes", {})[s] = float(deg)    # perform's air and step_turn key the foot on top of it
            info.setdefault("toes", {})[s] = float(deg)
        # cloth draped over the seat: chains whose bones point along chosen directions
        if spec.get("drape"):
            info["drape_bones"] = _drape(ctx, m, name, spec["drape"], f0, f1)
        # props the character wears (a guitar on its strap): on their bone before the hands go to them
        worn = WEAR.apply(ctx, name, m)
        if worn:
            info["wear"] = worn
        # arms
        hands = spec.get("hands", {})
        for side in ("L", "R"):
            h = hands.get(side)
            if not h:
                continue
            tgt, pole, cons, (fore, bend) = _arm_ik(ctx, m, smap, side)
            m.ik[side] = (tgt, pole)
            probe[name, side] = np.array([f1])
            wrist = smap[f"wrist.{side}"]
            sh = _world_head(arm, smap[f"arm.{side}"])
            out_dir = lat if side == "L" else -lat
            gp, gu = _grip_card(ctx, h)
            if h.get("pole") is not None:
                pole.location = targets.point(ctx, h["pole"])
            elif gu is not None and gu.get("type") == "strum":     # a strumming elbow: out to the side, over the body's edge
                pole.location = sh + out_dir * 0.45 + back * 0.12 - up * 0.08
            elif gu is not None and gu.get("type") == "neck":      # a fretting elbow hangs under the shoulder
                pole.location = sh + out_dir * 0.15 + back * 0.08 - up * 0.40
            else:
                pole.location = sh + out_dir * AR.POLE[0] + back * AR.POLE[1] - up * AR.POLE[2]
            pole_w = pole.location.copy()
            if seat and seat["prop"] is not None:          # IK goals ride with the prop
                for o in (tgt, pole):
                    _reparent(o, seat["prop"].root)
            ride_inv = ride_bone = ride_set = None         # the frame a bone ride's keys are relative to
            ride = h.get("ride")
            if ride is None and gu is not None and gu.get("type") in ("neck", "strum"):
                ride = gp.root.name                        # a hand on the neck or strings goes with the prop (it is worn: it moves)
            if ride:                                       # ... or with one of its parts (a steering wheel), or a bone
                if str(ride).startswith("cast:"):          # of the character itself (its chest): see _ride_bone
                    ride_bone, head_inv = _ride_bone(m, smap, name, side, ride, f1)
                    ride_set = head_inv.inverted()         # the bone's head frame in the settled pose
                    attach_to_bone(tgt, arm, ride_bone, Matrix())
                    attach_to_bone(pole, arm, ride_bone, head_inv @ Matrix.Translation(pole_w))
                    ride_inv = Matrix.Translation((0.0, -arm.data.bones[ride_bone].length, 0.0)) @ head_inv
                    info.setdefault("ride", {})[side] = ride_bone
                else:
                    ob = bpy.data.objects.get(ride)
                    if ob is None:
                        raise BuildError(f"pose.{name}.hands.{side}: ride object {ride!r} not found")
                    _reparent(tgt, ob)
            grip, interp = None, "BEZIER"
            if h.get("notes") is not None and not (h.get("grip") and gu is not None and gu.get("type") == "neck"):
                raise BuildError(f"pose.{name}.hands.{side}: `notes` is a fretting hand's bass line: it needs a neck grip "
                                 f"(grip = \"<prop>:neck\")")
            if h.get("grip"):
                neck = gu is not None and gu.get("type") == "neck"
                if h.get("keys") and not neck:
                    raise BuildError(f"pose.{name}.hands.{side}: grip and keys cannot be combined (only a neck grip takes "
                                     f"keys: {{t, fret, chord}})")
                pluck = (gu is not None and gu.get("type") == "strum"
                         and any(SM.fingerstyle(s) for s in strum_specs(ctx, name, side, gu, f"pose.{name}.hands.{side}")))
                grip = (_neck_grips if neck else _pluck_grips if pluck else _grip)(ctx, m, smap, name, side, h, back,
                                                                                  out_dir, pole_w)
                if neck:                                   # the strings it holds down, for the strings that ring
                    m.necks[side] = {"prop": gp.name, "entry": gu, "states": grip["states"],
                                     "press": float(h.get("press", FR.PRESS))}
                if grip.get("pen"):                        # the nib follows a track: a goal on every frame
                    mats, interp = _pen_goals(ctx, m, wrist, h, grip), "LINEAR"
                    probe[name, side] = np.union1d(probe[name, side], ctx.frames[ctx.frames >= f1][::6])
                else:
                    mats = FR.change_keys(grip["goal"], grip.get("changes", []), f1, 0)     # a fretting hand's next shapes
                    probe[name, side] = np.union1d(probe[name, side], [math.ceil(fr) for fr, *_ in grip.get("changes", [])])   # reach at every landing
                if grip.get("pick"):                       # the pick rides the wrist in the solved pinch
                    attach_to_bone(grip["pick"]["object"], arm, wrist, K.mat(grip["pick"]["T"]))
                if gu is not None and gu.get("type") in ("neck", "strum") and not grip.get("pen"):
                    goals = [M for _, M in mats] + (_stroke_ends(ctx, name, side, gp, gu, grip["goal"])
                                                    if gu.get("type") == "strum" else [])
                    goals += [st["goal"] for st in (grip.get("pluck") or {}).get("stations", {}).values()]
                    turned = _grip_pole(m, smap, side, goals, m.ik[side][1], f1)
                    if turned:                             # the elbow turned to keep the arm's twists in range
                        grip["digest"]["pole_turn_deg"] = turned
                info.setdefault("grip", {})[side] = grip["digest"]
                ctx.log("grip", name, side, grip["digest"])
            else:
                mats = _hand_targets(ctx, name, side, h, arm, smap, back, out_dir, f1)
            mats.sort(key=lambda fm: fm[0])
            par_inv = ride_inv if ride_inv is not None else (tgt.parent.matrix_world.inverted() if tgt.parent else Matrix())
            _key_goal(tgt, mats, par_inv, interp)
            sight = _sight_ride(ctx, m, smap, name, side, h, tgt, back, f1)
            if sight:
                info.setdefault("sight", {})[side] = [[round(a, 1), round(b, 1)] for a, b in sight]
            rig = {"tgt": tgt, "pole": pole, "fore": fore, "bend": bend, "par_inv": par_inv, "ride_bone": ride_bone,
                   "ride_set": ride_set, "pole_loc": pole.location.copy(), "h": h, "back": back, "out_dir": out_dir,
                   "full": {(bone, cname): arm.pose.bones[bone].constraints[cname].influence for bone, cname in cons},
                   "pole_w": pole_w, "pole_inv": ride_inv if ride_inv is not None else (
                       (pole.parent.matrix_world @ pole.matrix_parent_inverse).inverted() if pole.parent else Matrix())}
            if grip and grip.get("pluck"):                 # a plucking hand's stations, its goals in the goal's parent frame
                rig["pluck"] = dict(grip["pluck"], stations={s: dict(st, local=par_inv @ st["goal"])
                                                             for s, st in grip["pluck"]["stations"].items()})
            m.hands[side] = rig
            _key_move_poles(ctx, pole, h.get("keys") or [], pole_w, rig["pole_inv"])
            _key_ik(ctx, arm, name, side, rig, h.get("ik"), f0, f1)
            if h.get("ik") is not None:                    # a hand let go follows the member's strips (a walk's arm
                arm.animation_data.action_blend_type = "COMBINE"   # swing), which compose with its own keys (the
                info.setdefault("ik", {})[side] = len(h["ik"])     # pre-bend goes with the IK)
            fp = h.get("fingers")
            if grip:
                seq = FR.change_keys(grip["bones"], grip.get("changes", []), f1, 1)
                for b in grip["bones"]:
                    K.key_bone_quats(arm, b, [f0] + [f for f, _ in seq], [(1, 0, 0, 0)] + [bn[b] for _, bn in seq],
                                     interp="BEZIER")
            elif fp or any(k.get("fingers") for k in h.get("keys", [])):
                _key_fingers(ctx, arm, smap, side, h, f0, f1)
        for side, fp in (spec.get("fingers") or {}).items():
            for b, q in _finger_quats(arm, smap, side, fp).items():
                K.key_bone_quats(arm, b, [f0, f1], [(1, 0, 0, 0), tuple(q)], interp="BEZIER")
        out[name] = info
        ctx.log("pose", name)
    out["attached"] = attach_props(ctx)
    held = MOVES.attach_held(ctx)                              # what the gag moves hold: in a fist, hugged to the chest
    if held:
        out["held"] = held
    cables = WEAR.fit_cables(ctx)                              # worn props' leads hang from their jacks to the floor
    if cables:
        out["cables"] = cables
    bpy.context.view_layer.update()
    errs = ik_errors(ctx, probe)
    if errs:
        out["ik_error_mm"] = errs
    return out


IK_TOLERANCE_MM = 5.0


def ik_errors(ctx, probe):
    """How far each arm IK's wrist tail ends from its goal (mm), keyed "<cast>.<side>": the worst over that hand's own frames
    in `probe` ((cast, side) -> frames: the settled frame, plus a moving track's or a fretting hand's landings). Another
    member's frames are not its own: a hand let go (free) is off its parked goal there. A goal beyond the arm's reach
    leaves the hand short of the prop, so it is logged as a warning."""
    sc = bpy.context.scene
    keep = sc.frame_current
    at = {}
    for hand, frames in probe.items():
        for fr in np.atleast_1d(frames):
            at.setdefault(int(fr), []).append(hand)
    smaps = {name: S.semantic_map(ctx.cast[name].arm) for name in {name for name, _side in probe}}
    errs = {}
    for fr in sorted(at):
        sc.frame_set(fr)
        dg = bpy.context.evaluated_depsgraph_get()
        for name, side in at[fr]:
            m = ctx.cast[name]
            ae = m.arm.evaluated_get(dg)
            tail = ae.matrix_world @ ae.pose.bones[smaps[name][f"wrist.{side}"]].tail
            miss = round((tail - m.ik[side][0].evaluated_get(dg).matrix_world.translation).length * 1000, 1)
            errs[f"{name}.{side}"] = max(miss, errs.get(f"{name}.{side}", 0.0))
    for key, miss in errs.items():
        if miss > IK_TOLERANCE_MM:
            name, _, side = key.rpartition(".")
            ctx.log("WARNING", f"pose.{name}.hands.{side}: the wrist ends {miss} mm short of its goal (out of "
                               f"reach: move the seat, lean forward or bring the prop closer)")
    sc.frame_set(keep)
    return errs


def attach_to_bone(obj, arm, bone, rel):
    """Bone-parent obj with `rel` its matrix in the bone's HEAD frame (Blender bone parenting uses the tail)."""
    obj.parent = arm
    obj.parent_type = "BONE"
    obj.parent_bone = bone
    obj.matrix_parent_inverse.identity()
    obj.matrix_basis = Matrix.Translation((0.0, -arm.data.bones[bone].length, 0.0)) @ rel


def attach_props(ctx):
    """[[prop]] attach = "cast:bone" (semantic or Blender name), offset = [x, y, z] (m), attach_rot = [x, y, z] (deg,
    XYZ Euler) in the bone's head frame: the prop rides on the bone. With `cable` (true, or a worn cable's table) its
    card's cord is hung from its jack and swung like a worn prop's (wear.fit_cables, sim.simulate_cables): a mic in a hand.
    [[prop]] anchor_to = "cast": the card's use.anchor entries that name a `bone` (semantic) and an `object` are
    bone-parented to that cast member keeping their world placement at the settled base pose (earbuds in the ears,
    a cord on the chest)."""
    done = {}
    sc = bpy.context.scene
    for spec in ctx.data.get("prop", []):
        cable = spec.get("cable")
        if cable is not None and not spec.get("attach"):
            raise BuildError(f"prop {spec['name']!r}: `cable` goes with `attach` (a worn prop's is `wear.cable`)")
        if spec.get("attach"):
            cast_name, _, bone = spec["attach"].partition(":")
            m = ctx.cast.get(cast_name)
            if m is None:
                raise BuildError(f"prop {spec['name']!r}: attach to unknown cast member {cast_name!r}")
            b = S.resolve_bone(m.arm, bone)
            rel = Matrix.Translation(Vector(spec.get("offset", (0, 0, 0)))) @ \
                Euler([math.radians(a) for a in spec.get("attach_rot", (0, 0, 0))]).to_matrix().to_4x4()
            attach_to_bone(ctx.props[spec["name"]].root, m.arm, b, rel)
            done[spec["name"]] = f"{cast_name}:{b}"
            if cable is not None:
                cable = {"anchor": "jack"} if cable is True else cable          # truthy: fit_cables and the sim skip {}
                if not isinstance(cable, dict) or set(cable) - set(WR.TABLES["cable"]):
                    raise BuildError(f"prop {spec['name']!r}: cable = {spec['cable']!r}: true or a table of "
                                     f"{', '.join(WR.TABLES['cable'])}")
                ctx.props[spec["name"]].worn = {"cast": cast_name, "cable": dict(cable)}
        if spec.get("anchor_to"):
            m = ctx.cast.get(spec["anchor_to"])
            if m is None:
                raise BuildError(f"prop {spec['name']!r}: anchor_to unknown cast member {spec['anchor_to']!r}")
            sc.frame_set(ctx.start + ctx.settle)
            n = 0
            for u in ctx.props[spec["name"]].card.get("use", {}).get("anchor", []):
                if not (u.get("bone") and u.get("object")):
                    continue
                ob = bpy.data.objects[u["object"]]
                mw = ob.matrix_world.copy()
                ob.parent = m.arm
                ob.parent_type = "BONE"
                ob.parent_bone = S.resolve_bone(m.arm, u["bone"])
                ob.matrix_world = mw
                n += 1
            done[f"{spec['name']}.anchors"] = n
    return done
