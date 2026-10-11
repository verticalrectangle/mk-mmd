"""Metrics on a cast member's hands while its moves hold them (docs/design.md: Moves, Checks), measured on the built
scene with the rules the moves library keeps (mkmmd.core.moves): wrist_bend (a hand bends at most MV.WRIST degrees from
its forearm), hand_mirror (a two-hand move's hands mirror each other across the body), mic_palm (a fist round a mic or a
hammer keeps its palm toward the body) and hand_room (a hand at a place that does not touch the body stays out of its
collision bodies).

The hands ride the upper body (the pose stage keys their places on it), so they are measured in its frame: the world
with the upper body's motion undone, in the armature's axes (x its left, y behind it, z up; a mirror image is x turned
over). A move holds a hand from its `t` to `t + dur` (its own places; a two-hand move leaves a hand that rests at the
mic there, unless the move raises it), and a hand counts while it holds still at its place there (under STILL mm a
frame in that frame): one setting off early for its next move is travelling, which the rules do not cover."""
import numpy as np

from ..core import moves as MV
from ..solvers import geom
from . import CheckError, Metric, metric
from .props import TRUNK

SIDES = ("L", "R")
RIDE = ("?upper_body2", "upper_body")     # the bone the hands ride: upper_body2 when the rig has it
STILL = 1.5                               # mm a frame: a hand moving less than this in the ride frame holds its place
INWARD = {"L": 1.0, "R": -1.0}            # a palm toward the body has s * x < 0 (mkmmd.core.moves)
PALM = {"L": -1.0, "R": 1.0}              # the palm's side of cross(fingers, index -> little knuckles)


def _list(v):
    if v is None:
        return []
    return [v] if isinstance(v, str) else list(v)


def plays(ctx, args):
    """The cast member's moves, one row per hand: (t0, t1, side, move name, its places, both hands)."""
    if ctx.project is None:
        raise CheckError("hand metrics read the cast member's [[move.<cast>]]: run them in a project")
    try:
        member = ctx.project.cast_member(args.get("cast"))
    except Exception as e:  # noqa: BLE001 - a ProjectError names what is wrong
        raise CheckError(str(e))
    entries = ((ctx.project.data.get("move") or {}).get(member["name"])) or []
    rest = {"L": "rest", "R": "rest"}
    for e in entries:
        if isinstance(e, dict) and e.get("name") == "rest" and e.get("hand") in rest:
            rest[e["hand"]] = e.get("place", "rest")
    rows = []
    for e in entries:
        if not isinstance(e, dict) or e.get("name") not in MV.MOVES or e.get("name") == "rest":
            continue
        m = MV.MOVES[e["name"]]
        if m.get("hand") not in ("places", "beats"):
            continue
        t0 = float(e["t"])
        t1 = t0 + float(e.get("dur", 1.0))
        sides = SIDES if m.get("both") else (e.get("hand", "R"),)
        if m.get("both") and not m.get("mic_ok"):
            sides = tuple(s for s in sides if rest[s] != "mic")
        places = ([p for p, _ in m["places"]] if m["hand"] == "places"
                  else sorted({m["place"], m.get("off_place", m["place"])}))
        for s in sides:
            rows.append((t0, t1, s, e["name"], places, len(sides) == 2))
    if not rows:
        raise CheckError(f"{member['name']} has no hand moves ([[move.{member['name']}]])")
    return rows


def _unit(v):
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12)


def ride_space(data, arm, frames):
    """Per frame (F, 4, 4): the world to the armature's frame with the ride bone's motion since its rest undone (metres)."""
    W, R0, _ = data.bones(arm, list(RIDE), frames)
    A = data.arm_world(arm, frames)
    rest = geom.unscaled(A @ R0[0])                      # the ride bone at rest, where the armature is on each frame
    return np.linalg.inv(geom.unscaled(A)) @ rest @ np.linalg.inv(geom.unscaled(W[:, 0]))


def _apply(M, P):
    return np.einsum("fij,fj->fi", M[:, :3, :3], P) + M[:, :3, 3]


class _Hands(Metric):
    bones = ("elbow", "wrist", "middle1")
    keep = staticmethod(lambda row: True)
    args = {"cast": "cast member (its [[move.<cast>]] and rig)", "sides": "L R (default both)"}

    def needs(self, args, ctx, need):
        arm, _ = ctx.cast(args)
        sides = _list(args.get("sides")) or list(SIDES)
        need.bone(arm, *RIDE, *[f"{b}.{s}" for s in sides for b in ("wrist",) + tuple(self.bones)])
        return {"arm": arm, "sides": sides, "rows": plays(ctx, args)}

    def points(self, data, arm, side, frames):
        """World positions (F, 3) of the side's `bones` (a '?' one is left out when the rig lacks it)."""
        W, _, _ = data.bones(arm, [f"{n}.{side}" for n in self.bones], frames)
        return [W[:, j, :3, 3] for j in range(W.shape[1])]

    def holds(self, ctx, data, st, frames):
        """(ride frame per frame, {side: the move holding that hand still at its place on each frame, else None})."""
        M = ride_space(data, st["arm"], frames)
        f = np.asarray(frames, float)
        ts = [ctx.project.time(x) for x in frames]
        out = {}
        for s in st["sides"]:
            W, _, _ = data.bones(st["arm"], [f"wrist.{s}"], frames)
            q = _apply(M, W[:, 0, :3, 3]) * 1000.0
            if len(q) > 1:
                v = np.gradient(q, f, axis=0)                      # mm a frame, the frames as checked
                still = np.linalg.norm(v, axis=1) < STILL
            else:
                still = np.ones(len(q), bool)
            held = [None] * len(frames)
            for t0, t1, side, name, places, both in st["rows"]:
                if side != s or not self.keep((t0, t1, side, name, places, both)):
                    continue
                for k, t in enumerate(ts):
                    if t0 - 1e-6 <= t <= t1 + 1e-6 and still[k]:
                        held[k] = name
            out[s] = held
        if not any(any(h) for h in out.values()):
            raise CheckError(f"{self.name}: no move holds a hand still at its place in the checked frames")
        return M, out


@metric
class WristBend(_Hands):
    name = "wrist_bend"
    doc = ("How far a hand bends away from its forearm while a move holds it at its place: the angle between the forearm "
           "(elbow to wrist) and the hand (wrist to the middle finger's knuckle). The moves library keeps it under "
           f"{MV.WRIST:g}. Value: the largest, in degrees.")
    default_max = MV.WRIST

    def compute(self, args, ctx, data, frames, st):
        _, held = self.holds(ctx, data, st, frames)
        worst = (-1.0, None, None, None)
        for s in st["sides"]:
            el, wr, mid = self.points(data, st["arm"], s, frames)
            ang = np.degrees(np.arccos(np.clip(np.sum(_unit(wr - el) * _unit(mid - wr), 1), -1.0, 1.0)))
            for k, move in enumerate(held[s]):
                if move is not None and ang[k] > worst[0]:
                    worst = (float(ang[k]), int(frames[k]), s, move)
        return worst[0], {"at_frame": worst[1], "side": worst[2], "move": worst[3]}


@metric
class HandMirror(_Hands):
    name = "hand_mirror"
    doc = ("A two-hand move's hands mirror each other across the body: each hand's wrist and middle knuckle against the "
           "other's turned over the body's middle, in the frame of the upper body they ride, on the frames such a move "
           "holds both hands at their places. Value: the largest mismatch in mm (the angle between the hands' "
           "directions, mirrored, in detail).")
    args = {"cast": "cast member (its [[move.<cast>]] and rig)"}
    default_max = 5.0
    keep = staticmethod(lambda row: row[5])

    def needs(self, args, ctx, need):
        return super().needs(dict(args, sides=None), ctx, need)

    def compute(self, args, ctx, data, frames, st):
        M, held = self.holds(ctx, data, st, frames)
        both = np.array([held["L"][k] is not None and held["R"][k] is not None for k in range(len(frames))])
        if not both.any():
            raise CheckError("hand_mirror: no two-hand move holds both hands at their places in the checked frames")
        _, wl, ml = (_apply(M, p) for p in self.points(data, st["arm"], "L", frames))
        _, wr, mr = (_apply(M, p) for p in self.points(data, st["arm"], "R", frames))
        X = np.array([-1.0, 1.0, 1.0])
        mm = np.maximum(np.linalg.norm(wl - wr * X, axis=1), np.linalg.norm(ml - mr * X, axis=1)) * 1000.0
        deg = np.degrees(np.arccos(np.clip(np.sum(_unit(ml - wl) * _unit((mr - wr) * X), 1), -1.0, 1.0)))
        k = int(np.argmax(np.where(both, mm, -1.0)))
        return float(mm[k]), {"at_frame": int(frames[k]), "move": held["L"][k], "deg": round(float(deg[k]), 2),
                              "deg_max": round(float(deg[both].max()), 2), "frames": int(both.sum())}


@metric
class MicPalm(_Hands):
    name = "mic_palm"
    doc = ("A fist round a mic or a hammer (the moves library's aimed places) keeps its palm toward the body, never "
           "twisted out: the palm's direction (from the fingers and the index to little knuckles) in the frame of the "
           "upper body the hand rides, its share toward the middle, while the move holds it. Value: the smallest share "
           "(the library keeps it over 0.3).")
    default_min = 0.3
    bones = ("wrist", "middle1", "index1", "little1")
    keep = staticmethod(lambda row: any(p in MV.AIM for p in row[4]))

    def compute(self, args, ctx, data, frames, st):
        M, held = self.holds(ctx, data, st, frames)
        worst = (np.inf, None, None, None)
        for s in st["sides"]:
            wr, mid, ind, lit = (_apply(M, p) for p in self.points(data, st["arm"], s, frames))
            palm = PALM[s] * _unit(np.cross(_unit(mid - wr), _unit(lit - ind)))
            inward = -INWARD[s] * palm[:, 0]
            for k, move in enumerate(held[s]):
                if move is not None and inward[k] < worst[0]:
                    worst = (float(inward[k]), int(frames[k]), s, move)
        return worst[0], {"at_frame": worst[1], "side": worst[2], "move": worst[3]}


@metric
class HandRoom(_Hands):
    name = "hand_room"
    doc = ("A hand at a place that does not touch the body stays out of it: the smallest distance (mm) from the wrist "
           "and the finger knuckles and tips to the character's collision bodies (rig.json: the torso, hips, legs, neck "
           "and head, not the arms), while a move holds it at its place; places that touch the body are left out. The "
           "bodies are a little fatter than the skin, so a hand a few mm off them rests near it. Value: the smallest.")
    default_min = 0.0
    bones = ("wrist", "?index1", "?middle1", "?little1", "?index3", "?middle3", "?little3")
    keep = staticmethod(lambda row: not any(p in MV.TOUCH for p in row[4]))

    def needs(self, args, ctx, need):
        st = super().needs(args, ctx, need)
        _, rig = ctx.cast(args)
        if rig is None:
            raise CheckError("hand_room: needs the model's rig.json (cast with rig/asset)")
        wanted = [rig["map"].get(b, b) for b in TRUNK if rig["map"].get(b, b) in rig["bones"]]
        shapes = geom.Shapes()
        used = geom.model_shapes(shapes, dict(rig, bodies=[b for b in rig["bodies"] if b.get("bone") in wanted]), st["arm"])
        if not used:
            raise CheckError(f"hand_room: the model has no collision bodies on {wanted}")
        need.bone(st["arm"], *sorted({b["bone"] for b in used.values()}))
        return dict(st, shapes=shapes, rig=rig)

    def compute(self, args, ctx, data, frames, st):
        _, held = self.holds(ctx, data, st, frames)
        P = st["shapes"].pack()
        labels = [f"body:{st['rig']['bodies'][m]['bone']}" if m is not None else "?" for kind in geom.KINDS
                  for m in P[kind]["model"]]
        Rs, ps, _, _ = data.source_frames(st["shapes"].sources, frames)
        pts = {s: np.stack(self.points(data, st["arm"], s, frames), 1) for s in st["sides"]}
        worst = (np.inf, None, None, None, None)
        for k in range(len(frames)):
            sides = [s for s in st["sides"] if held[s][k] is not None]
            if not sides:
                continue
            X = np.concatenate([pts[s][k] for s in sides])
            n = pts[sides[0]].shape[1]
            res = geom.penetration(X, np.zeros(len(X)), geom.world(P, Rs[k], ps[k]), None)
            S = np.concatenate([res[kind][0] for kind in geom.KINDS if kind in res], 1)
            i, j = np.unravel_index(int(np.argmax(S)), S.shape)
            room = -float(S[i, j]) * 1000.0
            if room < worst[0]:
                s = sides[i // n]
                worst = (room, int(frames[k]), s, held[s][k], labels[j])
        return worst[0], {"at_frame": worst[1], "side": worst[2], "move": worst[3], "nearest": worst[4]}
