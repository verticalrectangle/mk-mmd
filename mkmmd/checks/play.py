"""string_play: every note a cast member plays on a worn guitar-like prop, played as planned (docs/design.md: Checks; Playing
a worn guitar, Strings that ring). The notes are the project's own: the strum spec (`[perform.<cast>] strum`) replayed on its
timeline (mkmmd.core.strum.plan) with the fretting hand's shapes (mkmmd.core.fretting.hand_states), each string a stroke
lets go of a note (mkmmd.core.strings.hits: a picked or plucked note on its string, a strum on every string it rings). Per
note the scene must show:

    ring     the note's string ringing from the note's fret (its object's `mk_amp` up and `mk_u0` the fret's place on the
             first frame after the note) and, for a line played one note at a time, no other string starting with it
    finger   a fretted note's finger on it: the closest skin of that finger's distal phalanx (the fretting hand's) within
             `finger_mm` across of the string, between the wire and the press window behind it (mkmmd.core.fretting
             PRESS_LO..PRESS_HI of the gap, `finger_mm` of slack) and touching it (`touch_mm` from its surface)
    strike   the pick tip (or a fingerstyle stroke's plucking fingertip skin) crossing the string within `time_ms` of the
             note and passing within `pick_mm` of its axis

The value is the number of notes that miss any of them (max 0); `detail` has the worst numbers and the first misses."""
import json

import numpy as np

from ..core import fretting as FR
from ..core import lanes as LN
from ..core import strings as CS
from ..core import strum as SM
from . import CheckError, Metric, metric
from .props import _picked, _specs, _timeline

FINGER_MM, TOUCH_MM, TIME_MS, PICK_MM = 3.0, 2.5, 8.0, 5.0


def _use_expr(prop, key):
    """An expression that reads `key` of every string object the prop's card names (lowest first; -1 when missing)."""
    return (f'[bpy.data.objects[s["object"]].get("{key}", 0.0) if s.get("object") in bpy.data.objects else -1.0 '
            f'for g in __import__("json").loads(bpy.data.objects["{prop}"]["mk_use"])["grip"] if g.get("type") == "neck" '
            f'for s in g["strings"]]')


def _lane_hits(ctx, h):
    spec = h.get("notes")
    if not (hasattr(spec, "get") and spec.get("lane") is not None):
        return None
    hits_file = (ctx.project.data.get("audio") or {}).get("hits")
    if not hits_file:
        raise CheckError("string_play: the notes lane reads the [audio] hits file, and the project names none")
    try:
        return LN.load(ctx.project.path(hits_file))
    except LN.LaneError as e:
        raise CheckError(f"string_play: {e}") from None


def _crossing(ts, u, level, d, t0, span=0.12):
    """The first time near t0 that u passes `level` moving the way d (+1 / -1), interpolated, or None."""
    near = np.where((ts >= t0 - span) & (ts <= t0 + span))[0]
    for k in near[:-1]:
        if (u[k] - level) * d <= 0.0 < (u[k + 1] - level) * d:
            return float(ts[k] + (level - u[k]) / (u[k + 1] - u[k]) * (ts[k + 1] - ts[k]))
    return None


@metric
class StringPlay(Metric):
    name = "string_play"
    doc = ("Every note of a worn guitar-like prop played as planned: the note's string rings from the note's fret (and no "
           "other string starts with a note of a line), a fretted note's finger lies on that string just behind its fret, "
           "and the pick tip or the plucking fingertip crosses that string on time. The notes are the project's strum spec "
           "and fretting hand replayed (mkmmd.core.strings). Value: the notes that miss any of it.")
    args = {"cast": "cast member who plays (default: the only one)", "hand": "R (default) | L: the pick hand",
            "pick": "the pick object (default <prop>_pick)", "finger_mm": f"across / along slack of a fretting finger (default {FINGER_MM:g})",
            "touch_mm": f"the fretting finger's skin from the string's surface (default {TOUCH_MM:g})",
            "time_ms": f"the strike's timing (default {TIME_MS:g})", "pick_mm": f"the strike's distance from the string's axis (default {PICK_MM:g})"}
    default_max = 0.0

    def needs(self, args, ctx, need):
        cast, specs = _specs(ctx, args)
        prop = specs[0].get("prop")
        if not prop:
            raise CheckError("string_play: the strum spec names no `prop`")
        arm, _rig = ctx.cast({"cast": cast})
        side = args.get("hand", "R")
        other = "L" if side == "R" else "R"
        hands = ((ctx.project.data.get("pose") or {}).get(cast) or {}).get("hands") or {}
        h = hands.get(other) or {}
        fretting = str(h.get("grip", "")).partition(":")[0] == prop
        pick = args.get("pick", f"{prop}_pick")
        need.prop_use(prop)
        need.obj(prop, pick)
        regions = {f"fret.{f}": {"bones": [f"{f}3.{other}"], "stop": []} for f in FR.FINGERS} if fretting else {}
        regions.update({f"pluck.{f}": {"bones": [f"{f}3.{side}"], "stop": []} for f in ("index", "middle")})
        return {"cast": cast, "specs": specs, "prop": prop, "pick": pick, "arm": arm, "side": side, "hand": h if fretting else None,
                "amp": need.expr(_use_expr(prop, "mk_amp")), "u0": need.expr(_use_expr(prop, "mk_u0")),
                "skin": need.skin(arm, regions)}

    def compute(self, args, ctx, data, frames, st):
        use = data.use(st["prop"])
        grips = {g["name"]: g for g in use.get("grip", [])}
        entry = grips.get(st["specs"][0].get("grip", "strum"))
        neck = next((g for g in grips.values() if g.get("type") == "neck"), None)
        if entry is None or neck is None:
            raise CheckError(f"string_play: prop {st['prop']!r} needs use.grip entries `strum` and one of type neck")
        n = len(neck["strings"])
        frames = np.asarray(frames)
        ts = np.array([ctx.project.time(f) for f in frames])
        amp, u0 = data.expr(st["amp"], frames), data.expr(st["u0"], frames)        # (F, n): the lowest string first
        if amp.ndim != 2 or amp.shape[1] != n or (amp < 0).any():
            raise CheckError(f"string_play: the card of {st['prop']!r} does not name {n} string objects of the scene")
        a, c, nrm = (np.asarray(entry[k], float) for k in ("along", "across", "normal"))
        a, c, nrm = a / np.linalg.norm(a), c / np.linalg.norm(c), nrm / np.linalg.norm(nrm)
        centre = np.asarray(entry["center"], float)
        s_c = float(centre @ a)
        cs = {s: float((FR.string_point(neck, s, s_c) - centre) @ c) for s in range(1, n + 1)}
        sigma = abs(cs[1] - cs[n]) / 2.0
        # the plan: strokes, the fretting hand's shapes, the strings each stroke lets go of
        try:
            states = FR.hand_states(st["hand"], neck, _lane_hits(ctx, st["hand"]))[0] if st["hand"] else []
        except FR.FrettingError as e:
            raise CheckError(f"string_play: the fretting hand: {e}") from None
        specs = [SM.resolved(sp, entry) for sp in st["specs"]]     # the card's technique (the bass plays with its fingers)
        tl = _timeline(ctx, specs)
        picking = any(SM.picking(sp) for sp in specs)
        picked = _picked(ctx, st["cast"], st["side"], st["prop"], neck, entry) if picking else None
        strokes = sorted((s for sp in specs for s in SM.plan(sp, tl, ctx.project.duration, picked)), key=lambda s: s.t)
        p = SM.params(specs[0])
        pressed = CS.presses(states, n) if states else []
        hits = CS.hits(strokes, cs, sigma, p, None if picking else CS.sounding(states, n))
        owner = {}
        for s in strokes:                                       # which stroke lets go of each string (for its direction)
            for t, string, _g in CS.hits([s], cs, sigma, p, None):
                owner[(round(t, 6), string)] = s
        # the scene in the prop's frame
        Mroot = data.obj(st["prop"], frames)
        inv = np.linalg.inv(Mroot)
        to_prop = lambda W, P: np.einsum("fij,fnj->fni", W[:, :3, :3], P) + W[:, None, :3, 3]   # noqa: E731
        A = data.arm_world(st["arm"], frames)
        sk = data.skin(st["skin"], frames)
        world = to_prop(A, sk["posed"])
        skin = to_prop(inv, world)                              # (F, nv, 3) prop frame
        region = np.asarray(sk["region"])
        names = list(sk["regions"])
        part = {nm: np.where(region == i)[0] for i, nm in enumerate(names)}
        Mpick = data.obj(st["pick"], frames)
        tip_off = float((entry.get("pick") or {}).get("tip", 0.008))
        tip_w = np.einsum("fij,j->fi", Mpick[:, :3, :3], np.array([tip_off, 0.0, 0.0])) + Mpick[:, :3, 3]
        tip = np.einsum("fij,fj->fi", inv[:, :3, :3], tip_w) + inv[:, :3, 3]
        lim = {k: float(args.get(k, d)) for k, d in (("finger_mm", FINGER_MM), ("touch_mm", TOUCH_MM), ("time_ms", TIME_MS),
                                                     ("pick_mm", PICK_MM))}
        rows, worst = [], {"finger_mm": 0.0, "touch_mm": 0.0, "time_ms": 0.0, "pick_mm": 0.0}
        dt = 1.0 / float(ctx.project.fps)
        for t, string, _g in hits:
            if not (ts[0] + dt <= t <= ts[-1] - 2 * dt):
                continue
            col = n - string
            fret = CS.fretted(pressed, string, t)
            row = {"t": round(t, 3), "string": string, "fret": fret, "miss": []}
            k = int(np.searchsorted(ts, t + 1e-6))                          # the first frame after the note
            if amp[k, col] <= 0.0:
                row["miss"].append("silent")
            elif abs(u0[k, col] - CS.fret_u(neck, string, fret)) > 2e-3:
                row["miss"].append("wrong_fret")
            if picking and k > 0 and any(amp[k, j] > amp[k - 1, j] + 1e-6 for j in range(n) if j != col):
                row["miss"].append("other_string")
            nut, bridge = CS.axis(neck, string)
            d_ax = (bridge - nut) / np.linalg.norm(bridge - nut)
            if fret:                                                         # the finger that holds it there
                _tt, pos, chord, _mv = max((s for s in states if s[0] is None or s[0] - s[3] <= t),
                                           key=lambda s: -1e9 if s[0] is None else s[0])
                held = [f for f, (s2, off) in FR.shape(chord, n).items()
                        if f not in FR.hovering(chord) and s2 == string and int(pos) + int(off) == fret]
                tg = FR.finger_targets(neck, {held[0]: [string, fret - int(pos)]}, int(pos))[0] if held else None
                if tg is None or f"fret.{held[0]}" not in part:
                    row["miss"].append("no_finger")
                else:
                    j = int(np.argmin(np.abs(ts - t)))
                    P = skin[j, part[f"fret.{held[0]}"]]
                    axis_at = CS.pressed_axis(neck, string, fret, CS.u_of(neck, string, P),
                                              float(st["hand"].get("press", FR.PRESS)))
                    off = P - axis_at
                    off_perp = off - (off @ d_ax)[:, None] * d_ax
                    gap = np.linalg.norm(off_perp, axis=1) - float(tg["radius"])
                    i = int(np.argmin(gap))
                    across = float(off_perp[i] @ c) * 1000.0
                    s_i = float(P[i] @ a)
                    along = max(tg["lo"] - s_i, s_i - tg["hi"], 0.0) * 1000.0
                    row.update(finger_mm=round(max(abs(across), along), 2), touch_mm=round(float(gap[i]) * 1000.0, 2))
                    worst["finger_mm"] = max(worst["finger_mm"], row["finger_mm"])
                    worst["touch_mm"] = max(worst["touch_mm"], row["touch_mm"])
                    if row["finger_mm"] > lim["finger_mm"] or row["touch_mm"] > lim["touch_mm"]:
                        row["miss"].append("finger")
            s = owner.get((round(t, 6), string))
            finger = getattr(s, "finger", None) if s is not None else None
            if finger and f"pluck.{finger}" in part:                          # a fingertip: its skin nearest the string
                Pf = skin[:, part[f"pluck.{finger}"]]
                near = np.argmin(np.linalg.norm(np.cross(Pf - nut, d_ax), axis=2), axis=1)
                point = Pf[np.arange(len(ts)), near]
            else:
                point = tip
            za, zb = float(nut @ a), float(bridge @ a)                      # the string where the tip is along it (the
            on = nut + ((point @ a - za) / (zb - za))[:, None] * (bridge - nut)   # strings fan out toward the bridge)
            u = (point - on) @ c
            d = s.dir if s is not None else 1
            cross = _crossing(ts, u, 0.0, d, t)
            if cross is None:
                row["miss"].append("no_strike")
            else:
                at = np.array([np.interp(cross, ts, point[:, i]) for i in range(3)])
                dist = float(np.linalg.norm(np.cross(at - nut, d_ax))) * 1000.0
                row.update(time_ms=round((cross - t) * 1000.0, 1), pick_mm=round(dist, 2))
                worst["time_ms"] = max(worst["time_ms"], abs(row["time_ms"]))
                worst["pick_mm"] = max(worst["pick_mm"], dist)
                if abs(row["time_ms"]) > lim["time_ms"] or dist > lim["pick_mm"]:
                    row["miss"].append("strike")
            rows.append(row)
        if not rows:
            raise CheckError("string_play: no note falls inside the checked frames")
        missed = [r for r in rows if r["miss"]]
        detail = {"notes": len(rows), "missed": len(missed), "worst": {k: round(v, 2) for k, v in worst.items()},
                  "limits": lim, "strings": sorted({r["string"] for r in rows}),
                  "fretted": sum(1 for r in rows if r["fret"]), "misses": missed[:8]}
        if missed:
            detail["at_frame"] = int(round(ctx.project.frame(missed[0]["t"])))
        return float(len(missed)), json.loads(json.dumps(detail))
