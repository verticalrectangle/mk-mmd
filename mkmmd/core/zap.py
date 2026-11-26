"""Zaps, bpy-free (docs/design.md: Frame effects, Zaps): the `[[fx]]` kind `zap` checked and timed; mkmmd/zap.py draws it.

    zap   a sheet's live sticker is somewhere else ON a time instead of travelling there. Over the frames before the time
          the sticker tears into horizontal slices that slide apart, its colours coming apart into two inks; on the
          time it lies where its sheet lays it then (a `step` in that sheet's keys, or another sheet: another plate,
          another pose, another size), torn only a little, and its slices snap shut over the next frames; thin rows of
          its pixels are stretched along the line from where it was for a few frames; and where it was its last
          picture stays, frozen, tinted toward an ink, breaking up in blocks dragged along the motion of the picture
          under it until it is gone (`life`), or bursting apart from a point when something goes through it (`touch`).
          A zap that `fail`s tears and comes back to the same place, `tries` times, stuttering.

Times are clip seconds, places frame fractions (0, 0 the top left), sizes in the sticker's own heights. Every part of a
zap is a function of the frame and the seed but the ghost's drag, which follows the picture under it from one frame to
the next (worked out again from the zap's frame when frames come out of order: FrameFx.safe keeps mk post's runs from
starting inside it)."""
import math

import numpy as np

KINDS = ("zap",)
TEAR = {"count": 14, "shift": 0.3, "share": 0.65, "split": 0.045, "inks": ["love", "foam"]}
GHOST = {"ink": "iris", "tint": 0.7, "alpha": 0.65, "tear": 0.4, "block": 20.0, "drop": 0.9, "jitter": 8.0,
         "scatter": 0.08, "life": 0.6}
STREAK = {"frames": 4, "rows": 0.6, "width": 3.5, "tint": 0.3}
BURST = {"speed": 1.4, "life": 0.3, "ripple": 3.0}
STUTTER = {"tries": 2, "gap": 0.2}
ZAP = {"t": None, "life": None, "fail": False, "touch": None}       # one zap's keys (`life` the ghost's, default its)
KEYS = {"zap": {"sheet", "zaps", "lead", "snap", "land", "tear", "ghost", "streak", "burst", "stutter", "seed"}}
DEFAULTS = {"zap": {"sheet": None, "zaps": None, "lead": 0.1, "snap": 0.05, "land": 0.3, "tear": {}, "ghost": {},
                    "streak": {}, "burst": {}, "stutter": {}, "seed": 1}}
MOVING = (0.35, 1.0)   # a sliding slice's share of `shift`, drawn between these
RESTING = 0.08         # how far a slice that does not slide still wobbles, a share of `shift`
NOISE = 0.25           # each frame's own wobble on a slice's slide, a share of it (digital: a new one every frame)
FADE = 1.5             # a ghost's opacity goes as (1 - age / life) ** FADE
RETRACT = 0.75         # a streak's far end comes in toward the sticker as (frame / frames) ** RETRACT


# ================================================================================================= the entry
def normalize(spec, out, palette, what):
    """A zap's own keys checked and filled in, in place (`out` has its defaults under `spec`)."""
    from . import framefx as FX          # framefx imports this module; its helpers are all there by the time this runs
    names = [out["sheet"]] if isinstance(out["sheet"], str) else out["sheet"]
    if not isinstance(names, (list, tuple)) or not names or not all(isinstance(n, str) and n.strip() for n in names):
        raise FX.FxError(f"{what}: sheet = {out['sheet']!r}: the `name` of the sheet whose live sticker zaps, or a list "
                         f"of them (she leaves the one that lays her on the frame before a zap for the one that lays "
                         f"her on its frame)")
    out["sheet"] = list(names)
    if out["rect"] is not None:
        raise FX.FxError(f"{what}: a zap draws in the stickers of its sheets, not in a part of the frame: no `rect`")
    for k, lo, hi in (("lead", 0.0, 2.0), ("snap", 0.0, 1.0), ("land", 0.0, 1.0)):
        out[k] = FX._num(out[k], f"{what} {k}", lo, hi)
    out["seed"] = FX._num(out["seed"], f"{what} seed", lo=0, integer=True)
    t = _sub(FX, out["tear"], TEAR, what, "tear")
    if not (isinstance(t["inks"], (list, tuple)) and len(t["inks"]) == 2):
        raise FX.FxError(f"{what}: tear inks = {t['inks']!r}: two colours, the one her colours part to the left and "
                         f"the one to the right")
    out["tear"] = {"count": FX._num(t["count"], f"{what} tear count", 1, 64, integer=True),
                   "shift": FX._num(t["shift"], f"{what} tear shift", 0.0, 3.0),
                   "share": FX._num(t["share"], f"{what} tear share", 0.0, 1.0),
                   "split": FX._num(t["split"], f"{what} tear split", 0.0, 1.0),
                   "inks": [FX._colour(c, palette, f"{what} tear inks") for c in t["inks"]]}
    g = _sub(FX, out["ghost"], GHOST, what, "ghost")
    out["ghost"] = {"ink": FX._colour(g["ink"], palette, f"{what} ghost ink"),
                    "tint": FX._num(g["tint"], f"{what} ghost tint", 0.0, 1.0),
                    "alpha": FX._num(g["alpha"], f"{what} ghost alpha", 0.0, 1.0),
                    "tear": FX._num(g["tear"], f"{what} ghost tear", 0.0, 1.0),
                    "block": FX._num(g["block"], f"{what} ghost block", 2.0, 400.0),
                    "drop": FX._num(g["drop"], f"{what} ghost drop", 0.0, 1.0),
                    "jitter": FX._num(g["jitter"], f"{what} ghost jitter", 0.0, 200.0),
                    "scatter": FX._num(g["scatter"], f"{what} ghost scatter", 0.0, 5.0),
                    "life": FX._num(g["life"], f"{what} ghost life", 1e-3, 30.0)}
    s = _sub(FX, out["streak"], STREAK, what, "streak")
    out["streak"] = {"frames": FX._num(s["frames"], f"{what} streak frames", 0, 30, integer=True),
                     "rows": FX._num(s["rows"], f"{what} streak rows", 0.0, 1.0),
                     "width": FX._num(s["width"], f"{what} streak width", 0.25, 40.0),
                     "tint": FX._num(s["tint"], f"{what} streak tint", 0.0, 1.0)}
    b = _sub(FX, out["burst"], BURST, what, "burst")
    out["burst"] = {"speed": FX._num(b["speed"], f"{what} burst speed", 0.0, 20.0),
                    "life": FX._num(b["life"], f"{what} burst life", 1e-3, 5.0),
                    "ripple": FX._num(b["ripple"], f"{what} burst ripple", 1e-3, 100.0)}
    u = _sub(FX, out["stutter"], STUTTER, what, "stutter")
    out["stutter"] = {"tries": FX._num(u["tries"], f"{what} stutter tries", 1, 8, integer=True),
                      "gap": FX._num(u["gap"], f"{what} stutter gap", 1e-3, 2.0)}
    out["zaps"] = _zaps(FX, out["zaps"], out, what)


def _sub(FX, v, defaults, what, name):
    if v is None or v is True:
        v = {}
    if not isinstance(v, dict):
        raise FX.FxError(f"{what}: {name} = {v!r}: expected a table ({{{', '.join(sorted(defaults))}}})")
    extra = sorted(set(v) - set(defaults))
    if extra:
        raise FX.FxError(f"{what}: {name}: unknown keys {extra} (known: {sorted(defaults)})")
    return {**defaults, **v}


def _zaps(FX, v, out, what):
    if v is None:
        raise FX.FxError(f"{what}: `zaps` is required: [{{t, life, fail, touch}}, ...], each `t` the clip second she is "
                         f"at the new place")
    items = [v] if isinstance(v, dict) else v
    if not isinstance(items, (list, tuple)) or not items or not all(isinstance(z, dict) for z in items):
        raise FX.FxError(f"{what}: zaps = {v!r}: expected [{{t, life, fail, touch}}, ...]")
    got = []
    for z in items:
        extra = sorted(set(z) - set(ZAP))
        if extra:
            raise FX.FxError(f"{what} zaps: unknown keys {extra} (known: {sorted(ZAP)})")
        if z.get("t") is None:
            raise FX.FxError(f"{what} zaps: {z!r} has no `t` (the clip second she is at the new place)")
        e = {"t": FX._num(z["t"], f"{what} zaps t"), "fail": z.get("fail", False),
             "life": out["ghost"]["life"] if z.get("life") is None else FX._num(z["life"], f"{what} zaps life",
                                                                                 1e-3, 30.0)}
        if not isinstance(e["fail"], bool):
            raise FX.FxError(f"{what} zaps: fail = {e['fail']!r}: true (she tears and comes back to the same place, "
                             f"`stutter` tries times) or false")
        tc = z.get("touch")
        if tc is not None:
            if e["fail"]:
                raise FX.FxError(f"{what}: the zap at {e['t']:g} s fails: it leaves no ghost to touch")
            if not isinstance(tc, dict) or tc.get("t") is None or set(tc) - {"t", "at"}:
                raise FX.FxError(f"{what} zaps: touch = {tc!r}: expected {{t, at}} (the clip second the ghost bursts "
                                 f"and the frame point [x, y] it bursts from; no `at`: its middle)")
            tc = {"t": FX._num(tc["t"], f"{what} zaps touch t"),
                  "at": None if tc.get("at") is None else FX._pair(tc["at"], f"{what} zaps touch at")}
            if not e["t"] <= tc["t"] < e["t"] + e["life"]:
                raise FX.FxError(f"{what}: the zap at {e['t']:g} s: touch at {tc['t']:g} s: its ghost is there from "
                                 f"{e['t']:g} to {e['t'] + e['life']:g} s")
        e["touch"] = tc
        got.append(e)
    got.sort(key=lambda e: e["t"])
    return got


# ================================================================================================= timing
def timed(fx, fps, frame0):
    """Each zap of entry `fx` (and of its aspect entries) gets its frames: `frame` (its time's, where she is at the new
    place), `lead` (the frames her sticker tears over before it, at least one), `snap` (those its slices take to snap
    shut from it), `tries` (a failed zap's: the frames of each try, `stutter.gap` apart), `ghost` (the frames its ghost
    shows, from its frame), `touch` (the frame a touch bursts the ghost on) and `burst` (the frames a burst takes, its
    ripple across the ghost's reach included at worst). An FxError when a zap reaches outside the span or two zaps'
    tears overlap."""
    from . import framefx as FX
    for e in [fx, *fx["aspect"].values()]:
        lead = max(1, int(round(e["lead"] * fps)))
        snap = int(round(e["snap"] * fps))
        for j, z in enumerate(e["zaps"]):
            f = FX.frame_of(z["t"], fps, frame0)
            tries = [FX.frame_of(z["t"] + k * e["stutter"]["gap"], fps, frame0)
                     for k in range(e["stutter"]["tries"] if z["fail"] else 1)]
            z.update(frame=f, lead=lead, snap=snap, tries=tries,
                     ghost=0 if z["fail"] else max(1, int(round(z["life"] * fps))),
                     touch_frame=None if z["touch"] is None else FX.frame_of(z["touch"]["t"], fps, frame0),
                     burst=int(math.ceil((e["burst"]["life"] + 1.0 / e["burst"]["ripple"]) * fps)))
            first, last = tries[0] - lead, max(tries[-1] + snap, f + z["ghost"]) - 1
            if z["touch_frame"] is not None:
                last = min(last, z["touch_frame"] + z["burst"])
            if first < fx["f0"] or last > fx["f1"] - 1:
                raise FX.FxError(f"fx {fx['index']} (zap): the zap at {z['t']:g} s works on frames {first}..{last} "
                                 f"(its tear from {lead} frames before it to its ghost's end), outside the span "
                                 f"({fx['f0']}..{fx['f1'] - 1})")
            for a, b in zip(tries, tries[1:]):
                if b - lead < a + snap:
                    raise FX.FxError(f"fx {fx['index']} (zap): the zap at {z['t']:g} s: its tries {e['stutter']['gap']:g} "
                                     f"s apart overlap: each tears over {lead} frames and snaps shut over {snap}")
            z["first"], z["last"] = first, last
        for a, b in zip(e["zaps"], e["zaps"][1:]):
            if b["tries"][0] - lead < a["tries"][-1] + snap:
                raise FX.FxError(f"fx {fx['index']} (zap): the zaps at {a['t']:g} and {b['t']:g} s overlap: each tears "
                                 f"over {lead} frames before it and snaps shut over {snap} after")


def link(fxs, fps, frame0):
    """Each zap (and its aspect entries) gets `laid`, [(index, f0, f1)] of the sheets of its `sheet` in table order (the
    frames each lays her on), and for every zap of its `zaps` `before` and `after`: the indices of the sheets that lay
    her on the frame before it and on its frame (the last of them in the table where two do: drawer()). An FxError
    when no sheet of that name lays a live sticker of its own, when none of them lays her on one of those frames or on
    a frame its ghost shows on (that sheet draws it), when a zap leaves her in the same place (the same sheet, its
    keys not moving her there) or a failed one moves her to another sheet."""
    from . import framefx as FX
    if not any(fx["kind"] in KINDS for fx in fxs):
        return
    for fx in fxs:
        if fx["kind"] not in KINDS:
            continue
        what = f"fx {fx['index']} (zap)"
        sheets = {}
        for name in fx["sheet"]:
            got = [s for s in fxs if s["kind"] == "sheet" and s.get("name") == name]
            if not got or got[0].get("objects") is None:
                raise FX.FxError(f"{what}: sheet = {name!r}: no sheet of that `name` lays a live sticker (`objects`)")
            sheets[got[0]["index"]] = got[0]
        for e in [fx, *fx["aspect"].values()]:
            if e["sheet"] != fx["sheet"] or [z["t"] for z in e["zaps"]] != [z["t"] for z in fx["zaps"]]:
                raise FX.FxError(f"{what}: an aspect.<output> table may not change the sheets or the zaps' times")
            e["laid"] = [(i, s["f0"], s["f1"]) for i, s in sorted(sheets.items())]
        for z in fx["zaps"]:
            for g in [z["tries"][0] - 1] + list(range(z["tries"][0], z["tries"][-1] + max(z["snap"], 1))):
                if drawer(fx, g) is None:
                    raise FX.FxError(f"{what}: the zap at {z['t']:g} s: none of its sheets ({', '.join(fx['sheet'])}) "
                                     f"lays her on frame {g}")
            before, after = drawer(fx, z["frame"] - 1), drawer(fx, z["frame"])
            if z["fail"] and any(drawer(fx, g) != before for g in range(z["tries"][0], z["tries"][-1] + 1)):
                raise FX.FxError(f"{what}: the zap at {z['t']:g} s fails: she stays where she is, so one sheet lays her "
                                 f"there through its tries")
            if not z["fail"] and before == after and not _moves(sheets[before], z["frame"] - 1, z["frame"], fps,
                                                                frame0):
                raise FX.FxError(f"{what}: the zap at {z['t']:g} s: sheet {sheets[before]['name']!r} lays her in the "
                                 f"same place on frames {z['frame'] - 1} and {z['frame']}: step its keys there "
                                 f"(`ease = \"step\"` at {z['t']:g}), zap to another sheet, or give it `fail = true`")
            gone = [g for g in range(z["frame"], z["frame"] + z["ghost"]) if g <= z["last"] and drawer(fx, g) is None]
            if gone:
                raise FX.FxError(f"{what}: the ghost of the zap at {z['t']:g} s shows on frames {z['frame']}.."
                                 f"{min(z['frame'] + z['ghost'], z['last'] + 1) - 1}, but none of its sheets lays her on "
                                 f"frame {gone[0]} (the sheet that does draws it): name the sheet she is on then in "
                                 f"`sheet`, or give the zap a shorter `life`")
        for e in [fx, *fx["aspect"].values()]:
            for z, z0 in zip(e["zaps"], fx["zaps"]):
                z.update(before=drawer(fx, z0["frame"] - 1), after=drawer(fx, z0["frame"]))


def drawer(fx, frame):
    """The index of the sheet of zap `fx`'s `sheet` that lays her on `frame` (the last of them in the table: the one
    on top, which draws the zap's ghosts there), or None."""
    on = [i for i, f0, f1 in fx["laid"] if f0 <= frame < f1]
    return on[-1] if on else None


def _moves(sheet, g0, g1, fps, frame0):
    """Whether a sheet's keys (any output's) move its sticker between frames g0 and g1 (a `step` on the frame of its
    second, as FrameFx keys it)."""
    from . import framefx as FX
    for e in [sheet, *(sheet.get("aspect") or {}).values()]:
        t0, t1 = (g0 - frame0) / fps, (g1 - frame0) / fps
        if FX.move_at(e["keys"], t0, 0.5 / fps) != FX.move_at(e["keys"], t1, 0.5 / fps):
            return True
    return False


# ================================================================================================= what it reads
def ghost_age(z, frame):
    """The frames since zap `z`'s frame while its ghost shows on `frame` (0 on the zap's own), else None."""
    a = frame - z["frame"]
    if z["fail"] or a < 0 or a >= z["ghost"] or frame > z["last"]:
        return None
    return a


def sources(fx, frame):
    """The other frames of the cut a zap reads at `frame`: for each ghost there, the frame before its zap (the sticker
    it leaves, as its sheet lays it there) and the frames since (the picture under it, whose motion drags it)."""
    out = []
    for z in fx["zaps"]:
        if ghost_age(z, frame) is not None:
            out += [g for g in range(z["frame"] - 1, frame) if g not in out]
    return sorted(out)


def safe(fx, frame):
    """Whether `frame` may start a run of mk post's frames: not inside a ghost after its first frame (it is dragged
    from the frame before)."""
    return not any((ghost_age(z, frame) or 0) > 0 for z in fx["zaps"])


# ================================================================================================= the tear
def tear_at(fx, frame):
    """[(zap index, side, amount)]: how far `frame` tears the sticker for each zap there: on `before` (the sheet she
    leaves) over the `lead` frames before the zap, rising as u ** 2 to 1 on the frame before it; on `after` (the sheet
    she arrives in) `land` on its frame, snapping shut as ((snap - k) / snap) ** 2 over the `snap` frames from it (k
    frames on): whole on the snap's last. A failed zap's tries tear and close on the sheet that keeps her, each so."""
    out = []
    for j, z in enumerate(fx["zaps"]):
        for f in z["tries"]:
            k = frame - f
            if -z["lead"] <= k < 0:
                u = (k + z["lead"] + 1) / z["lead"]
                out.append((j, "before", u * u))
            elif 0 <= k < z["snap"]:
                out.append((j, "after", fx["land"] * ((z["snap"] - k) / z["snap"]) ** 2))
    return out


def slices(fx, j, frame=None):
    """The slices zap j of `fx` tears the sticker into: [(top, bottom, slide)], top and bottom shares of the sticker's
    height (0 its top, rising, covering it), slide its sideways move at a full tear, in the sticker's heights (right
    positive): `share` of them slide `shift` times MOVING, the rest wobble RESTING; with a `frame`, each slide is off by
    that frame's own NOISE of it (seeded by the frame: a new wobble every frame, the same every time)."""
    t = fx["tear"]
    rng = np.random.default_rng([fx["seed"], fx["index"], j, 7])
    n = t["count"]
    cuts = np.sort(rng.uniform(0.0, 1.0, n - 1)) if n > 1 else np.zeros(0)
    edges = np.concatenate([[0.0], cuts, [1.0]])
    sign = 1.0 if rng.random() < 0.5 else -1.0
    out = []
    for i in range(n):
        sign = -sign if rng.random() < 0.7 else sign                 # mostly turn about: a tear, not a shear
        mag = rng.uniform(*MOVING) if rng.random() < t["share"] else RESTING * rng.uniform(-1.0, 1.0)
        out.append([float(edges[i]), float(edges[i + 1]), sign * mag * t["shift"]])
    if frame is not None:
        wob = np.random.default_rng([fx["seed"], fx["index"], j, int(frame), 11]).uniform(-1.0, 1.0, n)
        for s, w in zip(out, wob):
            s[2] += NOISE * w * max(abs(s[2]), RESTING * t["shift"])
    return [tuple(s) for s in out]


def fail_amount(fx, z, frame):
    """A failed zap's tear on `frame`, stuttering: its tries' tears with the amount held for two frames at a time
    (every other frame repeats the one before), so the slices jerk out and back instead of gliding."""
    held = frame - ((frame - z["tries"][0]) % 2)
    return max([a for j, s, a in tear_at({**fx, "zaps": [z]}, held)] or [0.0])


# ================================================================================================= the ghost and the streak
def ghost_alpha(fx, z, age):
    """The ghost's opacity `age` frames after its zap: `alpha` at first, going as (1 - u) ** FADE over its life."""
    u = age / max(z["ghost"], 1)
    return fx["ghost"]["alpha"] * max(1.0 - u, 0.0) ** FADE


def block_deaths(fx, j, count):
    """When each of `count` ghost blocks drops out, a share of the ghost's life: `drop` of them over it (the first a
    tenth of the way on, more and more after), the rest at its end."""
    rng = np.random.default_rng([fx["seed"], fx["index"], j, 13])
    r = rng.random(count)
    early = rng.random(count) < fx["ghost"]["drop"]
    return np.where(early, 0.04 + 0.96 * r ** 0.7, 1.0)


def block_drift(fx, j, shape):
    """Each ghost block's own drift, ((rows, cols, 2) frame heights a second, x right, y down): up to `scatter` in a
    direction of its own (seeded), so the blocks wander apart as the ghost breaks up, even over a still picture."""
    rng = np.random.default_rng([fx["seed"], fx["index"], j, 31])
    a = rng.uniform(0.0, 2.0 * math.pi, shape)
    r = fx["ghost"]["scatter"] * np.sqrt(rng.random(shape))
    return np.stack([r * np.cos(a), r * np.sin(a)], -1)


def streak_reach(fx, k):
    """How much of the line from where she was to where she is a streak covers `k` frames after the zap (1 on its
    frame), and its opacity: its far end coming in toward her as (k / frames) ** RETRACT, fading as it does."""
    n = fx["streak"]["frames"]
    if k < 0 or k >= n:
        return 0.0, 0.0
    u = k / n
    return 1.0 - u ** RETRACT if k else 1.0, (1.0 - u) ** 0.6


def burst_at(fx, z, frame, fps, dist):
    """A touched ghost's blocks on `frame`: (how far each has flown, frame heights; its opacity times) for blocks at
    `dist` frame heights from the touch: each lets go `dist / ripple` s after the touch and flies out at `speed`
    (easing out) over the burst's `life`, fading as (1 - u) ** 2; (0, 1) before it lets go."""
    if z.get("touch_frame") is None:
        return np.zeros_like(dist), np.ones_like(dist)
    b = fx["burst"]
    tau = (frame - z["touch_frame"]) / fps - np.asarray(dist) / b["ripple"]
    u = np.clip(tau / b["life"], 0.0, 1.0)
    gone = tau >= b["life"]
    fly = np.where(tau > 0, b["speed"] * b["life"] * (1.0 - (1.0 - u) ** 2), 0.0)
    return fly, np.where(gone, 0.0, np.where(tau > 0, (1.0 - u) ** 2, 1.0))
