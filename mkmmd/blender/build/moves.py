"""moves: `[[move.<cast>]]` (docs/design.md: Moves) compiled by mkmmd.core.moves into the tables the pose and perform stages
read: the hand keys into `[pose.<cast>.hands.L/R] keys` (places as `{cast, point}` targets), lean and tilt keys, twitches and
expressions into `[perform.<cast>]`. Runs at the start of the pose stage. The landmarks the places are built from come from
the member's rest pose in its root's frame, the beats from the project's timeline (audio/timeline.json)."""
import json
import math
import os

from mathutils import Vector

from ...core import moves as MV
from .. import scene as S
from . import BuildError

MOUTH = (0.0, -0.06, 0.012)                 # the mouth from the head bone's head (an MMD head sits on the neck's top)
EYES = (0.0, -0.05, 0.07)                   # the eyes from there, for a rig without eye bones


def marks(m):
    """The landmarks of cast member `m` in its own frame (mkmmd.core.moves.MARKS) from its armature's rest pose."""
    inv = m.root.matrix_world.inverted()
    arm = m.arm

    def head(sem):
        b = S.resolve_bone(arm, sem)
        return inv @ (arm.matrix_world @ arm.data.bones[b].head_local)

    out = {"arm.L": head("arm.L"), "arm.R": head("arm.R"), "chest": head("upper_body2")}
    h = head("head")
    out["mouth"] = h + Vector(MOUTH)
    try:
        out["eye"] = (head("eye.L") + head("eye.R")) / 2
    except (KeyError, ValueError):
        out["eye"] = h + Vector(EYES)
    out = {k: [round(float(c), 4) for c in v] for k, v in out.items()}
    out["reach"] = round(sum((head(f"wrist.{s}") - head(f"arm.{s}")).length for s in "LR") / 2, 4)
    return out


def _beats(ctx):
    path = ctx.path("audio/timeline.json")
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as fh:
        return [float(b) for b in json.load(fh).get("beats", [])]


def apply(ctx):
    """Lay every member's compiled moves into ctx.data's pose and perform tables. Returns {cast: report}."""
    moves = ctx.data.get("move") or {}
    if not isinstance(moves, dict):
        raise BuildError("[[move.<cast>]]: expected entries named after cast members")
    out, beats = {}, None
    for name, entries in moves.items():
        m = ctx.cast.get(name)
        if m is None:
            raise BuildError(f"[[move.{name}]]: no cast member {name!r} (cast: {', '.join(sorted(ctx.cast))})")
        beats = _beats(ctx) if beats is None else beats
        yaw = math.degrees(m.root.matrix_world.to_euler().z)
        try:
            res = MV.compile(list(entries), marks(m), beats, yaw=yaw)
        except MV.MoveError as e:
            raise BuildError(f"move.{name}: {e}") from None
        pose = ctx.data.setdefault("pose", {}).setdefault(name, {})
        for side, keys in res["hands"].items():
            if not keys:
                continue
            hand = pose.setdefault("hands", {}).setdefault(side, {})
            if any(k in hand for k in ("at", "keys", "grip", "rest")):
                raise BuildError(f"[pose.{name}.hands.{side}] places that hand and [[move.{name}]] moves it")
            hand["keys"] = [dict(k, at={"cast": name, "point": k["at"]}) for k in keys]
        perf = ctx.data.setdefault("perform", {}).setdefault(name, {})
        for key in ("lean", "tilt"):
            if res[key]:
                if perf.get(key):
                    raise BuildError(f"[perform.{name}] {key} and [[move.{name}]] both key it")
                perf[key] = res[key]
        for key in ("twitch", "expressions"):
            if res[key]:
                perf[key] = list(perf.get(key) or []) + res[key]
        out[name] = {"moves": len(entries), "hand_keys": {s: len(k) for s, k in res["hands"].items() if k},
                     "expressions": len(res["expressions"]), "twitches": len(res["twitch"])}
        ctx.log("moves", name, json.dumps(out[name]))
    return out
