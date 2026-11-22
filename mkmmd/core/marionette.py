"""Marionettes, bpy-free (docs/design.md: Marionettes): a cast member hung on threads from a wooden control cross (the
`library:marionette` prop) and moved from outside, late and loose. `[perform.<cast>] marionette` keys the cross and her
body (mkmmd/blender/build/marionette.py); the build's `marionette` stage draws the threads from where the cross and her
body are once everything is keyed (after `stepped`, so they step with her).

Frames and signs. Everything here is in the member's own frame: x her left, y behind her, z up, metres; angles in
radians unless a key says degrees. The cross's `rock` turns its front end down (about +x: she is pulled into a bow),
its `tilt` its left end down (about +y), its `turn` it toward her left (about +z). Her channels add to perform's own:
`lean` (forward), `tilt` (toward her left), `turn`, `head_tilt`, `head_nod` (down), `head_turn`, `rise` (the whole
body, feet too), `side` / `behind` (the hips, the feet planted), `dip` (the hips down, the knees bend), the ankles'
offsets `feet` and the toes' pitch `point` (toes down), and `guitar` (the worn guitar's turn about its face normal,
+ its neck up).

The model. Each part a thread holds (PARTS) is pulled toward where its thread wants it while the thread is taut and
hangs by its own weight when it is slack (a thread cannot push); it gets there late, through a damped spring of its own
period (`follow`): the body's weight on the head and back threads (0.34 s, a little bounce), the shoulders under the
cross (0.45 s) and the hips under the shoulders (0.55 s, so they swing on after it), the chest's turn after the
cross's, the head a beat behind its chest and swinging past it (`loll`), a knee under the leg bar (0.28 s) with its
shin swinging under it (0.4 s: the foot trails, lands forward and skids back). A cut thread lets its part go
(`cuts`): a hand drops off what it played and swings on its arm into a hang (`hang_path`), the head drops chin to
chest until she takes it back (`own`), the knees give and she catches her weight, and the last cuts leave her on her
own feet while the cross is thrown aside (`throw`).

The threads (`sag_points`, `dangle`, `cut_upper`, `cut_lower`): straight when taut, a parabola sagging under gravity to
its own length when slack; before `from` they hang under the falling cross and land on her; a cut one parts at `at`
(0 the cross, 1 her): the piece on the cross is yanked up through it and out above within YANK s, the piece on her
falls limp over her toward its `drape` side and slides off onto the floor."""
import math

import numpy as np

from . import perform as PF

PARTS = ("head", "back", "wrist_L", "wrist_R", "knee_L", "knee_R")
SIDES = ("L", "R")
SWING_KEYS = ("deg", "m", "from", "to", "every", "lead", "beats", "first")
KNEE_KEYS = ("t", "side", "height", "from", "to", "every", "first", "beats", "lead")
CUT_KEYS = ("t", "threads", "at", "own", "back", "hang", "give", "catch")
HOP_KEYS = ("t", "height", "land")
# the keys that move her in the world (mkmmd.core.marionette_world): she lies, floats, is flown, falls, lands
FLOAT_KEYS = ("from", "to", "bob", "period", "dips", "dip", "ease")
KICKS_KEYS = ("from", "to", "period", "height", "first", "wander", "seed")
SWIM_KEYS = ("from", "to", "period", "stroke", "kick", "first")
REACH_KEYS = ("from", "to", "at", "ease")
RELEASE_KEYS = ("t", "tumble", "spin", "terminal")
LAND_KEYS = ("t", "at", "give", "radius", "clatter", "speed", "approach")
YANK_KEYS = ("t", "v", "at")
KNOT_KEYS = ("t", "swing", "loop")
WORLD_KEYS = ("lie", "float", "kicks", "arms", "reach", "swim", "release", "land", "yank", "knot")
KEYS = {"prop": None, "from": None, "drop": None, "height": None, "lift": None, "hang": None, "lag": None,
        "timeline": None, "rock": SWING_KEYS, "tilt": SWING_KEYS, "turn": SWING_KEYS, "sway": SWING_KEYS, "raise": None,
        "hops": HOP_KEYS, "knees": KNEE_KEYS, "cuts": CUT_KEYS, "throw": ("t", "v", "spin"), "guitar": None,
        "over": None, "width": None, "lie": None, "float": FLOAT_KEYS, "kicks": KICKS_KEYS, "arms": None, "reach": REACH_KEYS,
        "swim": SWIM_KEYS, "release": RELEASE_KEYS, "land": LAND_KEYS, "yank": YANK_KEYS, "knot": KNOT_KEYS}
DEFAULTS = {"drop": 0.35, "height": 0.9, "lift": 0.03, "hang": 0.012, "lag": 1.0, "over": 0.4}
ARMS = ("shoulders", "chest")     # where the threads hold her wrists while she lies on her back

DROP_HEIGHT = 2.6           # m the cross falls from before it is caught over her
ATTACH = 0.08               # s: the falling threads' ends land on her parts over this before `from`
SNAP = (0.05, 0.03, 0.22)   # s: the jerk that pulls the threads taut: up, held, back down
KNEE_PULL = (0.09, 0.08, 0.18)     # s: the leg bar's lift of a knee: up, held, down; it sets off LEAD before the beat
KNEE_LEAD = 0.18           # s: so the knee is up on the beat (the bar's lift plus the knee's lag)
HOP = (0.08, 0.05, 0.28)    # s: a hop's quick lift of the cross: up, held, down
GRAVITY = 9.81
TORSO = 0.45                # m: hips to shoulders, for the chest's lean between them
# (period s, damping) of each part's spring
BODY = (0.34, 0.45)
SHOULDER = (0.45, 0.32)
HIPS = (0.55, 0.25)
CHEST = (0.40, 0.35)
HEAD = (0.70, 0.22)
HEAD_DELAY = 0.12           # s: the head's spring hears the chest this late (it lolls a beat behind)
KNEE = (0.28, 0.50)
SHIN = (0.40, 0.30)
GUITAR = (0.35, 0.35)
DROP = (0.50, 0.42)         # a head whose thread is cut falls so (chin to chest, a small bounce: the hair stays on her)
GIVE = (0.30, 0.50)         # knees whose threads are cut give so, and she catches herself
SWING = (0.60, 0.16)        # a cut hand's pendulum on its arm
HEAD_DROP = math.radians(38.0)   # chin to chest
HEAD_HELD = math.radians(4.0)    # a held head droops this much on its thread
SNAP_NOD = math.radians(-9.0)    # the snap jerks the head up
SNAP_NECK = math.radians(4.0)    # and the guitar's neck
SNAP_KNEE = 0.03                 # m: and twitches the knees
YANK = 0.25                 # s: a cut piece on the cross is out above it in this time
FALL = 0.30                 # s: a cut piece on her swings down over her in this time
SLIDE = 0.40                # s after the cut it slides off her and drops
DRAPE = {"head": (0.0, 0.5, -1.0), "back": (0.0, 0.6, -1.0), "wrist_L": (0.3, -0.2, -1.0),
         "wrist_R": (-0.3, -0.2, -1.0), "knee_L": (0.1, -0.5, -1.0), "knee_R": (-0.1, -0.5, -1.0)}
HANG = (-0.03, -0.08, -0.2)      # a cut hand's hang from where it was, her frame: lower and in front of what it played


class MarionetteError(ValueError):
    """A `marionette` table the build cannot use."""


# ---- curves


def pulse(ts, t0, up, hold, down):
    """0 -> 1 over `up` s from t0 (smoothstep), held `hold` s, back to 0 over `down` s."""
    ts = np.asarray(ts, float)
    return PF.smooth((ts - t0) / max(up, 1e-4)) * (1.0 - PF.smooth((ts - t0 - up - hold) / max(down, 1e-4)))


def step(ts, t0):
    return (np.asarray(ts, float) >= t0).astype(float)


def follow(ts, target, spring, lag=1.0, h=1.0 / 240.0):
    """A damped spring following `target` ((n,) or (n, k), sampled at the ascending times `ts`): x'' = w^2 (g - x) -
    2 zeta w x', w = 2 pi / (period * lag), from rest on the first sample, integrated on an `h` s grid (semi-implicit
    Euler). The lag and the swing past of a part hung on a thread."""
    ts = np.asarray(ts, float)
    g = np.asarray(target, float)
    flat = g.ndim == 1
    g2 = g[:, None] if flat else g
    if len(ts) < 2:
        return g.copy()
    period, zeta = spring
    w = 2.0 * math.pi / (float(period) * float(lag))
    grid = np.arange(ts[0], ts[-1] + h, h)
    G = np.stack([np.interp(grid, ts, g2[:, j]) for j in range(g2.shape[1])], 1)
    x, v = G[0].copy(), np.zeros(g2.shape[1])
    out = np.empty_like(G)
    a, b = h * w * w, h * 2.0 * zeta * w
    for i in range(len(grid)):
        v += a * (G[i] - x) - b * v
        x = x + h * v
        out[i] = x
    res = np.stack([np.interp(ts, grid, out[:, j]) for j in range(g2.shape[1])], 1)
    return res[:, 0] if flat else res


def delayed(ts, x, dt):
    """`x` sampled at ts heard `dt` s late (held at its first value before)."""
    return np.interp(np.asarray(ts, float) - dt, ts, x)


HOP_FLIGHT = 0.36          # s: a hop from a stand is in the air this long; one after a landing takes off RECOVER after it
HOP_RECOVER = 0.06


def landing_hops(ts, landings):
    """(rise, dip, lead) (n,) m of hops that land on their times [(t, height)]: each a parabola `height` high that
    leaves the floor HOP_RECOVER after the landing before it (or HOP_FLIGHT before its own, from a stand: a brace in
    the knees first) and comes down on t into the knees (dip, a third of its height, back up in 0.14 s), the bars
    jerked up ahead of it (`lead`: the cross's lift, the hand moving first)."""
    ts = np.asarray(ts, float)
    rise, dip, lead = np.zeros(len(ts)), np.zeros(len(ts)), np.zeros(len(ts))
    prev = -math.inf
    for t, h in landings or ():
        stand = t - prev > HOP_FLIGHT + HOP_RECOVER + 0.25
        t0 = t - HOP_FLIGHT if stand else prev + HOP_RECOVER
        span = max(t - t0, 0.1)
        u = (ts - t0) / span
        rise = np.maximum(rise, np.where((u > 0.0) & (u < 1.0), 4.0 * h * u * (1.0 - u), 0.0))
        dip = dip + h / 3.0 * pulse(ts, t, 0.05, 0.0, 0.14)
        if stand:
            dip = dip + 0.2 * h * pulse(ts, t0 - 0.14, 0.1, 0.0, 0.04)
        lead = lead + 1.2 * h * pulse(ts, t0 - 0.07, 0.07, 0.03, max(span - 0.1, 0.08))
        prev = t
    return rise, dip, lead


def _num(v, what, lo=None, hi=None):
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(float(v)):
        raise MarionetteError(f"marionette.{what} = {v!r}: expected a number")
    v = float(v)
    if (lo is not None and v < lo) or (hi is not None and v > hi):
        raise MarionetteError(f"marionette.{what} = {v}: expected {lo} .. {hi}")
    return v


def _beats_in(beats, lo, hi, every, first=0):
    sel = [b for b in sorted(beats) if lo - 1e-6 <= b < hi - 1e-6]
    return sel[first::max(int(every), 1)]


def swing_keys(v, beats, what, unit="deg"):
    """Keys [[t, value], ...] of a cross channel: keys as they are, or a beat table {deg (or m), from, to, every = 1,
    lead = 0.05, first = 1}: the value swings to +deg, -deg, +deg ... on every `every`-th beat in the window (`lead` s
    before it: the hand moves on the beat, the puppet answers after), from 0 a swing before the first and back to 0 a
    swing after the last; `first` = -1 starts the other way. Values in the table's unit (degrees or metres)."""
    if not v:
        return []
    if isinstance(v, list):
        try:
            return [[float(t), float(x)] for t, x in v]
        except (TypeError, ValueError):
            raise MarionetteError(f"marionette.{what}: keys [[t, {unit}], ...] or a table {{{unit}, from, to, every}}")
    if not isinstance(v, dict):
        raise MarionetteError(f"marionette.{what}: keys [[t, {unit}], ...] or a table {{{unit}, from, to, every}}")
    unknown = sorted(set(v) - set(SWING_KEYS))
    if unknown:
        raise MarionetteError(f"marionette.{what}: unknown keys {unknown} (known: {list(SWING_KEYS)})")
    amp = _num(v.get(unit, 0.0), f"{what}.{unit}")
    lo, hi = _num(v.get("from", -1e9), f"{what}.from"), _num(v.get("to", 1e9), f"{what}.to")
    lead = _num(v.get("lead", 0.15), f"{what}.lead", 0.0, 0.5)
    sign = -1.0 if _num(v.get("first", 1), f"{what}.first") < 0 else 1.0
    bs = _beats_in(v.get("beats") or beats, lo, hi, v.get("every", 1))
    if not bs:
        return []
    gap = (bs[1] - bs[0]) if len(bs) > 1 else 0.5
    keys = [[bs[0] - lead - gap, 0.0]]
    for k, b in enumerate(bs):
        keys.append([b - lead, sign * amp * (1.0 if k % 2 == 0 else -1.0)])
    keys.append([bs[-1] - lead + gap, 0.0])
    return keys


def knee_events(v, beats):
    """[(t, side, height)] of `knees`: a list [{t, side, height = 0.12}] or a beat table {from, to, every = 2, first =
    "L", height = 0.12, lead}: a knee on every `every`-th beat of the window, the sides in turn."""
    if not v:
        return []
    out = []
    if isinstance(v, list):
        for i, e in enumerate(v):
            if not isinstance(e, dict) or "t" not in e:
                raise MarionetteError(f"marionette.knees[{i}]: a table {{t, side, height}}")
            side = e.get("side", "L")
            if side not in SIDES:
                raise MarionetteError(f"marionette.knees[{i}].side = {side!r}: \"L\" or \"R\"")
            out.append((_num(e["t"], f"knees[{i}].t"), side, _num(e.get("height", 0.12), f"knees[{i}].height", 0.0, 0.4)))
        return sorted(out)
    if not isinstance(v, dict):
        raise MarionetteError("marionette.knees: a list of {t, side, height} or a table {from, to, every, first, height}")
    unknown = sorted(set(v) - set(KNEE_KEYS) - {"t", "side"})
    if unknown:
        raise MarionetteError(f"marionette.knees: unknown keys {unknown} (known: {list(KNEE_KEYS)})")
    first = v.get("first", "L")
    if first not in SIDES:
        raise MarionetteError(f"marionette.knees.first = {first!r}: \"L\" or \"R\"")
    h = _num(v.get("height", 0.12), "knees.height", 0.0, 0.4)
    bs = _beats_in(v.get("beats") or beats, _num(v.get("from", -1e9), "knees.from"), _num(v.get("to", 1e9), "knees.to"),
                   v.get("every", 2))
    other = "R" if first == "L" else "L"
    return [(b, first if k % 2 == 0 else other, h) for k, b in enumerate(bs)]


def normalize(spec, beats):
    """The plan's numbers from a `[perform.<cast>] marionette` table (KEYS) and the beats (clip seconds): the defaults
    filled, beat tables made keys, the cuts checked (each thread cut once, after `from`). Raises MarionetteError."""
    if not isinstance(spec, dict):
        raise MarionetteError("marionette: expected a table {prop, from, ...}")
    unknown = sorted(set(spec) - set(KEYS))
    if unknown:
        raise MarionetteError(f"marionette: unknown keys {unknown} (known: {sorted(KEYS)})")
    if "prop" not in spec or "from" not in spec:
        raise MarionetteError("marionette: needs `prop` (a library:marionette prop) and `from` (the clip second the threads "
                              "land on her and pull taut)")
    P = {k: _num(spec.get(k, d), k, 0.0) for k, d in DEFAULTS.items() if k != "over"}
    over = spec.get("over", DEFAULTS["over"])
    try:
        P["over"] = [[float(t), float(x)] for t, x in over] if isinstance(over, list) else _num(over, "over", 0.0)
    except (TypeError, ValueError):
        raise MarionetteError("marionette.over: metres, or keys [[t, metres], ...]") from None
    P["lag"] = max(P["lag"], 0.2)
    P["prop"] = str(spec["prop"])
    P["guitar"] = str(spec["guitar"]) if spec.get("guitar") else None
    P["from"] = _num(spec["from"], "from")
    if P["drop"] < ATTACH + 0.05:
        raise MarionetteError(f"marionette.drop = {P['drop']}: at least {ATTACH + 0.05} s for the cross to fall in")
    for k, unit in (("rock", "deg"), ("tilt", "deg"), ("turn", "deg"), ("sway", "m")):
        P[k] = swing_keys(spec.get(k), beats, k, unit)
    try:
        P["raise"] = [[float(t), float(x)] for t, x in (spec.get("raise") or [])]
    except (TypeError, ValueError):
        raise MarionetteError("marionette.raise: keys [[t, metres], ...]") from None
    try:
        P["width"] = [[float(t), float(x)] for t, x in (spec.get("width") or [])]
    except (TypeError, ValueError):
        raise MarionetteError("marionette.width: keys [[t, metres], ...] (the threads' radius)") from None
    if any(not 0.0002 <= x <= 0.01 for _t, x in P["width"]):
        raise MarionetteError("marionette.width: a thread's radius is 0.0002 .. 0.01 m")
    P["hops"], P["landings"] = _hops(spec.get("hops"))
    P["knees"] = knee_events(spec.get("knees"), beats)
    cuts, seen = [], {}
    for i, c in enumerate(spec.get("cuts") or []):
        where = f"cuts[{i}]"
        if not isinstance(c, dict) or "t" not in c or not c.get("threads"):
            raise MarionetteError(f"marionette.{where}: a table {{t, threads = [...], at}}")
        unknown = sorted(set(c) - set(CUT_KEYS))
        if unknown:
            raise MarionetteError(f"marionette.{where}: unknown keys {unknown} (known: {list(CUT_KEYS)})")
        threads = [c["threads"]] if isinstance(c["threads"], str) else list(c["threads"])
        for th in threads:
            if th not in PARTS:
                raise MarionetteError(f"marionette.{where}.threads: {th!r} is no thread (threads: {list(PARTS)})")
            if th in seen:
                raise MarionetteError(f"marionette.{where}: the {th} thread is cut already by cuts[{seen[th]}]")
            seen[th] = i
        t = _num(c["t"], f"{where}.t")
        if t <= P["from"]:
            raise MarionetteError(f"marionette.{where}.t = {t}: a thread is cut after it is on her (from = {P['from']})")
        cut = {"t": t, "threads": threads, "at": _num(c.get("at", 0.5), f"{where}.at", 0.05, 0.95),
               "give": _num(c.get("give", 0.08), f"{where}.give", 0.0, 0.3),
               "catch": _num(c.get("catch", 0.3), f"{where}.catch", 0.0, 1.0),
               "hang": [float(x) for x in c.get("hang", HANG)]}
        for k in ("own", "back"):
            if k in c:
                a, b = (_num(x, f"{where}.{k}") for x in c[k])
                if not t <= a < b:
                    raise MarionetteError(f"marionette.{where}.{k} = [{a}, {b}]: a window after the cut ({t})")
                cut[k] = (a, b)
        cuts.append(cut)
    P["cuts"] = sorted(cuts, key=lambda c: c["t"])
    th = spec.get("throw")
    if th:
        if not isinstance(th, dict) or "t" not in th:
            raise MarionetteError("marionette.throw: a table {t, v = [x, y, z] m/s, spin = deg/s}")
        P["throw"] = {"t": _num(th["t"], "throw.t"), "v": [float(x) for x in th.get("v", (-3.0, 0.4, 2.6))],
                      "spin": _num(th.get("spin", 540.0), "throw.spin")}
    else:
        P["throw"] = None
    P["world"] = _world(spec, P)
    if P["throw"] and P["world"]["release"]:
        raise MarionetteError("marionette: `throw` throws the cross aside and `release` lets it fall after her on her "
                              "head thread: give one of them")
    return P


def _hops(v):
    """(hops at their top on t, hops that land on t) of `hops`: a list [{t, height = 0.07, land = false}] or a table
    {land = [t, ...], height} (every hop lands on its time)."""
    if not v:
        return [], []
    if isinstance(v, dict):
        unknown = sorted(set(v) - {"land", "height"})
        if unknown or not isinstance(v.get("land"), list):
            raise MarionetteError("marionette.hops: a list [{t, height, land}] or a table {land = [t, ...], height}")
        h = _num(v.get("height", 0.07), "hops.height", 0.0, 0.5)
        return [], sorted((_num(t, "hops.land"), h) for t in v["land"])
    if not isinstance(v, list):
        raise MarionetteError("marionette.hops: a list [{t, height, land}] or a table {land = [t, ...], height}")
    top, land = [], []
    for i, h in enumerate(v):
        if not isinstance(h, dict) or "t" not in h:
            raise MarionetteError(f"marionette.hops[{i}]: a table {{t, height, land}}")
        e = (_num(h["t"], f"hops[{i}].t"), _num(h.get("height", 0.07), f"hops[{i}].height", 0.0, 0.5))
        (land if h.get("land") else top).append(e)
    return top, sorted(land)


def _table(v, what, keys, need=()):
    if not isinstance(v, dict):
        raise MarionetteError(f"marionette.{what}: a table {{{', '.join(keys)}}}")
    unknown = sorted(set(v) - set(keys))
    if unknown:
        raise MarionetteError(f"marionette.{what}: unknown keys {unknown} (known: {list(keys)})")
    missing = [k for k in need if k not in v]
    if missing:
        raise MarionetteError(f"marionette.{what}: needs {', '.join(missing)}")
    return v


def _window(v, what, lo=-1e9, hi=1e9):
    a, b = _num(v.get("from", lo), f"{what}.from"), _num(v.get("to", hi), f"{what}.to")
    if b <= a:
        raise MarionetteError(f"marionette.{what}: from ({a}) before to ({b})")
    return a, b


def _side(v, what):
    if v not in SIDES:
        raise MarionetteError(f"marionette.{what} = {v!r}: \"L\" or \"R\"")
    return v


def _world(spec, P):
    """The keys that move her in the world (WORLD_KEYS; mkmmd.core.marionette_world plays them), checked, defaults
    filled; `on` whether any is given (then her root is keyed: she can lie, be flown, fall and land)."""
    W = {"on": any(spec.get(k) is not None for k in WORLD_KEYS)}
    try:
        W["lie"] = [[float(t), float(x)] for t, x in (spec.get("lie") or [])]
    except (TypeError, ValueError):
        raise MarionetteError("marionette.lie: keys [[t, deg], ...] (-90 on her back, face up; 90 on her front)") from None
    W["float"] = None
    if spec.get("float"):
        f = _table(spec["float"], "float", FLOAT_KEYS)
        a, b = _window(f, "float")
        W["float"] = {"from": a, "to": b, "bob": _num(f.get("bob", 0.02), "float.bob", 0.0, 0.2),
                      "period": _num(f.get("period", 3.2), "float.period", 0.5, 30.0),
                      "dips": sorted(_num(t, "float.dips") for t in (f.get("dips") or [])),
                      "dip": _num(f.get("dip", 0.012), "float.dip", 0.0, 0.2),
                      "ease": _num(f.get("ease", 0.8), "float.ease", 0.05, 5.0)}
    W["kicks"] = None
    if spec.get("kicks"):
        k = _table(spec["kicks"], "kicks", KICKS_KEYS)
        a, b = _window(k, "kicks")
        W["kicks"] = {"from": a, "to": b, "period": _num(k.get("period", 2.4), "kicks.period", 0.6, 20.0),
                      "height": _num(k.get("height", 0.07), "kicks.height", 0.0, 0.3),
                      "first": _side(k.get("first", "L"), "kicks.first"),
                      "wander": _num(k.get("wander", 0.25), "kicks.wander", 0.0, 0.6),
                      "seed": int(_num(k.get("seed", 0), "kicks.seed"))}
    arms = spec.get("arms", "shoulders")
    if arms not in ARMS:
        raise MarionetteError(f"marionette.arms = {arms!r}: one of {list(ARMS)} (where her wrists are held while she "
                              f"lies on her back)")
    W["arms"] = arms
    reach = spec.get("reach") or []
    reach = [reach] if isinstance(reach, dict) else reach
    if not isinstance(reach, list):
        raise MarionetteError("marionette.reach: a list [{from, to, at}]")
    W["reach"] = []
    for i, r in enumerate(reach):
        r = _table(r, f"reach[{i}]", REACH_KEYS, ("from", "to", "at"))
        a, b = _window(r, f"reach[{i}]")
        W["reach"].append({"from": a, "to": b, "at": r["at"], "ease": _num(r.get("ease", 0.3), f"reach[{i}].ease",
                                                                           0.05, 3.0)})
    W["swim"] = None
    if spec.get("swim"):
        s = _table(spec["swim"], "swim", SWIM_KEYS)
        a, b = _window(s, "swim")
        W["swim"] = {"from": a, "to": b, "period": _num(s.get("period", 0.8), "swim.period", 0.3, 5.0),
                     "stroke": _num(s.get("stroke", 0.2), "swim.stroke", 0.0, 0.5),
                     "kick": _num(s.get("kick", 0.08), "swim.kick", 0.0, 0.3),
                     "first": _side(s.get("first", "L"), "swim.first")}
    W["release"] = W["land"] = W["yank"] = W["knot"] = None
    if spec.get("release"):
        r = _table(spec["release"], "release", RELEASE_KEYS, ("t",))
        t = _num(r["t"], "release.t")
        if t <= P["from"]:
            raise MarionetteError(f"marionette.release.t = {t}: the control is let go after the threads are on her")
        W["release"] = {"t": t, "tumble": _num(r.get("tumble", 0.0), "release.tumble"),
                        "spin": _num(r.get("spin", 240.0), "release.spin"),
                        "terminal": _num(r.get("terminal", 4.0), "release.terminal", 0.5, 60.0)}
    if spec.get("land"):
        g = _table(spec["land"], "land", LAND_KEYS, ("t", "at"))
        t = _num(g["t"], "land.t")
        if not W["release"] or t < W["release"]["t"] + 0.6:
            raise MarionetteError(f"marionette.land.t = {t}: she lands from a fall, at least 0.6 s after `release`")
        W["land"] = {"t": t, "at": g["at"], "give": _num(g.get("give", 0.1), "land.give", 0.0, 0.4),
                     "radius": _num(g.get("radius", 0.4), "land.radius", 0.0, 50.0),
                     "clatter": _num(g.get("clatter", 0.35), "land.clatter", 0.1, 2.0),
                     "speed": _num(g.get("speed", 2.5), "land.speed", 0.2, 20.0),
                     "approach": _num(g.get("approach", 0.6), "land.approach", 0.1, 3.0)}
    if spec.get("yank"):
        y = _table(spec["yank"], "yank", YANK_KEYS, ("t",))
        t = _num(y["t"], "yank.t")
        if not W["land"] or t <= W["land"]["t"] + 0.1:
            raise MarionetteError(f"marionette.yank.t = {t}: she is yanked by her head thread from where she landed: "
                                  f"after `land`")
        v = y.get("v", (0.6, 0.0, 7.0))
        if not isinstance(v, (list, tuple)) or len(v) != 3:
            raise MarionetteError("marionette.yank.v: [x, y, z] m/s (her frame)")
        W["yank"] = {"t": t, "v": [float(x) for x in v], "at": _num(y.get("at", 0.6), "yank.at", 0.05, 0.95)}
    if spec.get("knot"):
        k = _table(spec["knot"], "knot", KNOT_KEYS, ("t",))
        t = _num(k["t"], "knot.t")
        last = max([P["from"]] + [c["t"] for c in P["cuts"]] + [W[e]["t"] for e in ("release", "land") if W[e]]
                   + ([W["yank"]["t"] + 1.2] if W["yank"] else []))
        if t <= last:
            raise MarionetteError(f"marionette.knot.t = {t}: her threads are knotted on again after the last cut, the "
                                  f"fall and the yank (after {last})")
        loop = _num(k["loop"], "knot.loop", t) if k.get("loop") is not None else None
        W["knot"] = {"t": t, "swing": _num(k.get("swing", 8.0), "knot.swing", 0.0, 45.0), "loop": loop}
    return W


def knot_time(P):
    """The clip second her threads are whole and on her again (inf: never)."""
    k = (P.get("world") or {}).get("knot")
    return k["t"] if k else math.inf


def cut_time(P, part):
    """The clip second the part's thread is cut (inf: never)."""
    return next((c["t"] for c in P["cuts"] if part in c["threads"]), math.inf)


def held(ts, P, part):
    """(n,) 1 while the part's thread holds it (from `from` to its cut, and again once it is knotted on), else 0."""
    ts = np.asarray(ts, float)
    return (((ts >= P["from"]) & (ts < cut_time(P, part))) | (ts >= knot_time(P))).astype(float)


# ---- the cross


def cross_plan(ts, P, half_leg=0.12):
    """The cross over the clip, relative to its home over her head (her frame): `pos` (n, 3) m, `rock`, `tilt`, `turn`
    (n,) rad, `lift` (n,) m (its height over home that pulls her: raise, hops and the snap, not the fall in), `sway`
    (n,) m (sideways), `fall` (n,) m (the fall in from above, caught), and the leg bar (held in the other hand, falling
    in with the cross): `leg_pos` (n, 3) and `leg_roll` (n,) rad (its left end up +), with `knee` {side: (n,) m} how
    far each knee's anchor rises (`half_leg` m: half the leg bar). `shown` (n,) bool: from the moment it starts to fall
    in. After a `throw` the cross flies off on a ballistic arc, spinning."""
    ts = np.asarray(ts, float)
    n = len(ts)
    t0, drop = P["from"], P["drop"]
    ta, tl = t0 - drop, t0 - ATTACH
    u = np.clip((ts - ta) / max(tl - ta, 1e-3), 0.0, 1.0)
    fall = DROP_HEIGHT * (1.0 - u * u)
    after = np.maximum(ts - tl, 0.0)
    fall = fall - 0.035 * np.exp(-after / 0.12) * np.sin(2.0 * math.pi * after / 0.3) * (ts > tl)    # caught: a dip
    lift = P["lift"] * 1.2 * pulse(ts, t0 - SNAP[0], *SNAP) + PF.eased_keys(ts, P["raise"])
    for t, h in P["hops"]:
        lift = lift + h * pulse(ts, t - HOP[0] - 0.06, *HOP)       # she is up on the beat: late, swinging past it
    sway = PF.eased_keys(ts, P["sway"])
    rock = np.radians(PF.eased_keys(ts, P["rock"]))
    tilt = np.radians(PF.eased_keys(ts, P["tilt"]))
    turn = np.radians(PF.eased_keys(ts, P["turn"]))
    pos = np.zeros((n, 3))
    pos[:, 0] = sway
    lead = landing_hops(ts, P.get("landings"))[2]                 # the hand jerks the bars up ahead of each hop
    pos[:, 2] = fall + lift + lead
    knee = {s: np.zeros(n) for s in SIDES}
    leg_z, leg_roll = np.zeros(n), np.zeros(n)
    for t, side, h in P["knees"]:
        p = pulse(ts, t - KNEE_LEAD, *KNEE_PULL)
        knee[side] = np.maximum(knee[side], h * p)
    for side, sgn in (("L", 1.0), ("R", -1.0)):                    # the bar goes up by half and rolls the rest
        leg_roll = leg_roll + sgn * np.arcsin(np.clip(knee[side] / (2.0 * half_leg), -1.0, 1.0))
    leg_z = 0.5 * (knee["L"] + knee["R"])
    leg_pos = np.zeros((n, 3))
    leg_pos[:, 0] = 0.6 * sway
    leg_pos[:, 2] = fall + lift + leg_z + lead
    th = P.get("throw")
    if th:
        tau = np.maximum(ts - th["t"], 0.0)
        on = ts >= th["t"]
        v = np.asarray(th["v"], float)
        fly = np.outer(tau, v) + np.outer(0.5 * tau * tau, (0.0, 0.0, -GRAVITY))
        pos = pos + fly * on[:, None]
        leg_pos = leg_pos + fly * on[:, None] * 0.8
        tilt = tilt + np.radians(th["spin"]) * tau * on * (-1.0 if v[0] < 0 else 1.0)
        rock = rock + np.radians(0.35 * th["spin"]) * tau * on
    shown = ts >= ta - 1e-6
    return {"pos": pos, "rock": rock, "tilt": tilt, "turn": turn, "lift": lift, "sway": sway, "fall": fall,
            "leg_pos": leg_pos, "leg_roll": leg_roll, "knee": knee, "shown": shown}


# ---- her body


def body_plan(ts, P, cross, legs, stand=None):
    """Her channels (see the module's docstring) over `ts` from the cross's plan. `legs` {side: {thigh, shin, foot}}
    m: the thigh and shin lengths and the ankle to the toes' tip. `stand` (n,) 0..1: how much she stands on her feet
    on the floor (default all the time: the knees give at a cut only so; mkmmd.core.marionette_world passes its own).
    Returns {lean, tilt, turn, head_tilt, head_nod, head_turn, rise, side, behind, dip, feet {side: (n, 3)}, point
    {side: (n,)}, knee {side: (n,) m the leg bar lifts it}, guitar, report}."""
    ts = np.asarray(ts, float)
    n = len(ts)
    lag = P["lag"]
    t0 = P["from"]
    stand = np.ones(n) if stand is None else np.asarray(stand, float)
    hd, bk = held(ts, P, "head"), held(ts, P, "back")
    body = np.maximum(hd, bk)
    knees_on = np.minimum(held(ts, P, "knee_L") + (ts < t0), held(ts, P, "knee_R") + (ts < t0))
    hang = P["hang"] * body * np.minimum(knees_on, 1.0)           # toes barely on the floor while she hangs
    # the weight: lifted by the cross (a thread pulls, it cannot push), late, with a little bounce
    zt = body * (hang + np.maximum(cross["lift"], 0.0))
    z = follow(ts, zt, BODY, lag)
    hop, hop_dip, _ = landing_hops(ts, P.get("landings"))
    rise = np.maximum(z, 0.0) + hop
    dip = 0.8 * np.maximum(-z, 0.0) + hop_dip                     # a landing goes into the knees
    # the shoulders under the cross, the hips under the shoulders
    sh = follow(ts, np.stack([cross["sway"] * body, np.zeros(n)], 1), SHOULDER, lag)
    hp = follow(ts, sh, HIPS, lag)
    held_tilt = follow(ts, cross["tilt"] * body, CHEST, lag)
    held_rock = follow(ts, cross["rock"] * body, CHEST, lag)
    tilt = 0.55 * held_tilt + np.arctan2(sh[:, 0] - hp[:, 0], TORSO)
    lean = 0.6 * held_rock - np.arctan2(sh[:, 1] - hp[:, 1], TORSO)
    turn = 0.6 * follow(ts, cross["turn"] * body, CHEST, lag)
    # the head: its world roll and pitch follow the chest's a beat late and swing past (it lolls), drooping when held
    head_in_t = delayed(ts, tilt + 0.5 * hd * cross["tilt"], HEAD_DELAY * lag)
    head_in_n = delayed(ts, lean + 0.4 * hd * cross["rock"], HEAD_DELAY * lag)
    snap_nod = SNAP_NOD * pulse(ts, t0 - SNAP[0], *SNAP) * hd.max()
    head_tilt = follow(ts, head_in_t, HEAD, lag) - tilt
    head_nod = follow(ts, head_in_n + snap_nod, HEAD, lag) - lean + HEAD_HELD * follow(ts, hd, BODY, lag)
    head_turn = 0.4 * follow(ts, delayed(ts, turn, HEAD_DELAY * lag), HEAD, lag) - 0.4 * turn
    # the knees: each under its end of the leg bar, the shin swinging under it
    feet, point, knee = {}, {}, {}
    knee_peak = {}
    for s in SIDES:
        k = held(ts, P, f"knee_{s}")
        target = k * (cross["knee"][s] + SNAP_KNEE * pulse(ts, t0 - SNAP[0], *SNAP))
        h = np.maximum(follow(ts, target, KNEE, lag), 0.0)
        thigh = float(legs[s]["thigh"])
        th = np.arccos(np.clip(1.0 - h / thigh, -1.0, 1.0))
        fwd = follow(ts, thigh * np.sin(th), SHIN, lag)
        f = np.zeros((n, 3))
        f[:, 1] = -fwd                                             # forward is -y
        f[:, 2] = h
        feet[s] = f
        knee[s] = h
        toe = np.arcsin(np.clip(rise / max(float(legs[s].get("foot", 0.14)), 0.05), 0.0, 0.6))
        point[s] = 0.55 * th + toe
        knee_peak[s] = round(float(h.max()) * 1000, 1)
    side = hp[:, 0] - 0.1 * (feet["L"][:, 2] - feet["R"][:, 2])   # the weight over the foot that stays down
    behind = hp[:, 1].copy()
    guitar = -0.5 * follow(ts, cross["tilt"] * np.maximum(held(ts, P, "wrist_L"), held(ts, P, "wrist_R")), GUITAR, lag)
    guitar = guitar + SNAP_NECK * follow(ts, pulse(ts, t0 - SNAP[0], *SNAP), GUITAR, lag) * held(ts, P, "wrist_L").max()
    report = {"rise_mm": round(float(rise.max()) * 1000, 1), "sway_mm": round(float(np.abs(hp[:, 0]).max()) * 1000, 1),
              "knee_mm": knee_peak}
    # the cuts
    cut_dip = np.zeros(n)
    for c in P["cuts"]:
        tc = c["t"]
        fade = np.ones(n)
        if "own" in c:                                             # she takes it back herself, slowly
            fade = 1.0 - PF.smooth((ts - c["own"][0]) / (c["own"][1] - c["own"][0]))
        if "head" in c["threads"]:
            fall = follow(ts, step(ts, tc), DROP, lag) * fade
            head_nod = head_nod + HEAD_DROP * fall
            head_tilt = head_tilt + math.radians(7.0) * fall
            lean = lean + math.radians(5.0) * fall
        if "knee_L" in c["threads"] or "knee_R" in c["threads"]:   # standing: the knees give, she catches her weight
            give = follow(ts, step(ts, tc) - (1.0 - c["catch"]) * step(ts, tc + 0.18), GIVE, lag) * fade
            cut_dip = cut_dip + c["give"] * give * stand
        if "wrist_L" in c["threads"]:                              # the neck sinks with the hand, and she lifts it
            guitar = guitar - math.radians(5.0) * follow(ts, pulse(ts, tc, 0.08, 0.15, 0.4), GUITAR, lag)
        if "back" in c["threads"]:                                 # she lands on her own feet
            cut_dip = cut_dip + 0.015 * follow(ts, pulse(ts, tc, 0.06, 0.05, 0.3), GIVE, lag) * stand
    dip = dip + cut_dip
    report["cuts"] = [[c["t"], c["threads"]] for c in P["cuts"]]
    return {"lean": lean, "tilt": tilt, "turn": turn, "head_tilt": head_tilt, "head_nod": head_nod,
            "head_turn": head_turn, "rise": rise, "side": side, "behind": behind, "dip": dip, "cut_dip": cut_dip,
            "feet": feet, "point": point, "knee": knee, "guitar": guitar, "report": report}


def _unit(v, fallback=(0.0, 0.0, 1.0)):
    v = np.asarray(v, float)
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else np.asarray(fallback, float)


def slerp(a, b, t):
    """The unit vector a fraction `t` of the way from a to b on the great circle (t past 1 goes on past b; opposite
    vectors turn through the horizontal, or x)."""
    a, b = _unit(a), _unit(b)
    c = float(np.clip(a @ b, -1.0, 1.0))
    ang = math.acos(c)
    if ang < 1e-6:
        return a.copy()
    axis = np.cross(a, b)
    if np.linalg.norm(axis) < 1e-6:                               # opposite: through the horizontal
        side = np.array([1.0, 0.0, 0.0]) if abs(a[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        axis = np.cross(a, side)
    k = _unit(axis)
    th = ang * float(t)
    return a * math.cos(th) + np.cross(k, a) * math.sin(th) + k * (k @ a) * (1.0 - math.cos(th))


def hang_path(ts, t_cut, start, shoulder, hang, spring=SWING):
    """(n, 3): a hand whose thread is cut at t_cut falls from `start` (where it was then) and swings on its arm about
    `shoulder` into `hang`, past it and back, settling (a pendulum's step response, in direction and reach; `start`
    before the cut)."""
    ts = np.asarray(ts, float)
    start, shoulder, hang = (np.asarray(v, float) for v in (start, shoulder, hang))
    r0, r1 = start - shoulder, hang - shoulder
    period, zeta = spring
    w = 2.0 * math.pi / period
    wd = w * math.sqrt(1.0 - zeta * zeta)
    tau = np.maximum(ts - t_cut, 0.0)
    e = np.exp(-zeta * w * tau) * (np.cos(wd * tau) + zeta / math.sqrt(1.0 - zeta * zeta) * np.sin(wd * tau))
    out = np.empty((len(ts), 3))
    for i, x in enumerate(1.0 - e):
        r = (1.0 - min(max(x, 0.0), 1.0)) * np.linalg.norm(r0) + min(max(x, 0.0), 1.0) * np.linalg.norm(r1)
        out[i] = shoulder + r * slerp(r0, r1, x)
    return out


# ---- the threads


def sag_points(A, B, length, n=24, side=(1.0, 0.0, 0.0)):
    """(n, 3) points of a thread from A to B whose own length is `length`: straight when it is taut (length <= |AB|),
    else a parabola sagging under gravity, square to the chord, as deep as its length asks (L = d + 8 h^2 / 3d); a
    thread near the vertical bows toward `side` instead (gravity has no side to pull it to)."""
    A, B = np.asarray(A, float), np.asarray(B, float)
    u = np.linspace(0.0, 1.0, n)[:, None]
    d = B - A
    dist = float(np.linalg.norm(d))
    P = A + u * d
    if dist < 1e-6 or length <= dist:
        return P
    h = math.sqrt(3.0 / 8.0 * dist * (length - dist))
    c = d / dist
    g = np.array([0.0, 0.0, -1.0])
    g = g - (g @ c) * c
    sd = np.asarray(side, float)
    sd = sd - (sd @ c) * c
    k = float(np.linalg.norm(g))
    bow = _unit(g * min(k / 0.3, 1.0) + _unit(sd, (1.0, 0.0, 0.0)) * max(0.0, 1.0 - k / 0.3), (1.0, 0.0, 0.0))
    return P + 4.0 * u * (1.0 - u) * h * bow


def dangle(A, length, tau, n=24, side=(1.0, 0.0, 0.0)):
    """(n, 3): a thread hanging from A, falling with its cross: straight down `length`, fluttering sideways (`tau` s
    since it started to fall)."""
    A = np.asarray(A, float)
    s = np.linspace(0.0, 1.0, n)[:, None]
    flutter = 0.06 * length * np.sin(math.pi * s) * math.sin(9.0 * tau + 0.7) + 0.03 * length * s ** 2 * math.sin(
        14.0 * tau)
    return A + s * length * np.array([0.0, 0.0, -1.0]) + flutter * _unit(side)


def cut_upper(A, C, tau, n=24, side=(1.0, 0.0, 0.0)):
    """(n, 3): the piece left on the cross `tau` s after its thread was cut at C: yanked up through its anchor A and out
    above it in YANK s (the material a distance s from A sits s - D(tau) down the old line, or above A once that is
    negative), its free end whipping sideways."""
    A, C = np.asarray(A, float), np.asarray(C, float)
    length = float(np.linalg.norm(C - A))
    down = _unit(C - A, (0.0, 0.0, -1.0))
    D = (length + 0.6) * (max(tau, 0.0) / YANK) ** 1.6
    out = np.empty((n, 3))
    sd = _unit(side)
    for i, s in enumerate(np.linspace(0.0, length, n)):
        q = s - D
        p = A + q * down if q >= 0.0 else A + np.array([0.0, 0.0, -q])
        whip = 0.12 * (s / max(length, 1e-6)) ** 2 * math.sin(20.0 * tau) * math.exp(-5.0 * tau)
        out[i] = p + whip * sd
    return out


def cut_lower(B, C, tau, drape, floor_z=0.0, B_slide=None, radius=0.0015, n=24):
    """(n, 3): the piece left on her `tau` s after its thread was cut at C (its other end B on her): it swings down
    over her from the line it held toward `drape` (its far end first) over FALL s, and SLIDE s after the cut slides
    off (from where B was then, `B_slide`) and drops onto the floor at `floor_z`, lying there in loose curls."""
    B, C = np.asarray(B, float), np.asarray(C, float)
    length = float(np.linalg.norm(C - B))
    up = _unit(C - B)
    dd = _unit(drape, (0.0, 0.0, -1.0))
    s = np.linspace(0.0, 1.0, n)
    f = np.clip((max(tau, 0.0) / FALL) ** 2 * (0.55 + 0.45 * s), 0.0, 1.0)
    ds = length / (n - 1)
    pts = np.empty((n, 3))
    root = B if (tau < SLIDE or B_slide is None) else np.asarray(B_slide, float)
    pts[0] = root
    for i in range(1, n):
        pts[i] = pts[i - 1] + ds * slerp(up, dd, f[i])
    if tau >= SLIDE:
        drop = 0.5 * GRAVITY * (tau - SLIDE) ** 2
        pts[:, 2] -= drop
        down = pts[:, 2] <= floor_z + radius
        pts[:, 2] = np.maximum(pts[:, 2], floor_z + radius)
        across = _unit(np.cross(dd, (0.0, 0.0, 1.0)), (1.0, 0.0, 0.0))     # a thread on the floor lies in curls
        coil = min(1.0, (tau - SLIDE) / 0.6) * 0.18 * length
        pts += np.outer(down * coil * np.sin(2.0 * math.pi * 1.5 * s) * s, across)
    return pts


def point_at(P, u):
    """The point a share `u` along a polyline P (n, 3) by its length."""
    P = np.asarray(P, float)
    seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    return np.array([np.interp(u * cum[-1], cum, P[:, j]) for j in range(3)])


def twang(n, tau, amp=0.012, freq=11.0, decay=0.12):
    """(n,) the sideways shake of a thread snapping taut `tau` s ago: its first mode, dying out."""
    if tau < 0.0:
        return np.zeros(n)
    return amp * math.exp(-tau / decay) * math.sin(2.0 * math.pi * freq * tau) * np.sin(math.pi * np.linspace(0, 1, n))


# ---- the prop (library:marionette): a control and its threads

CONTROLS = ("aeroplane", "stirrers")
MAIN = (-0.20, 0.28)        # m: the main bar along y (front end, back end), at z = 0 of the cross's frame
SLAT = (0.034, 0.014)       # m: a slat's width and thickness
HEAD_BAR = 0.16             # m: half the head bar (the wings), across x at y = 0, on top of the main bar
HAND_BAR = {"at": (0.0, -0.205, -0.034), "half": 0.25, "radius": 0.0085}      # a dowel under the main bar's front
LEG_BAR = {"home": (0.0, -1.15, -0.5), "half": 0.28}   # held in the other hand, in front of her and lower: wide, so the
#                                                        knee threads pass outside her face seen from the front
EYE = (0.0055, 0.0012)      # m: a screw eye's ring and wire radius
ANCHORS = {"head": ("cross", (0.0, 0.0, -0.007 - EYE[0])), "back": ("cross", (0.0, 0.265, -0.007 - EYE[0])),
           "wrist_L": ("hand", (0.235, 0.0, -0.0085 - EYE[0])), "wrist_R": ("hand", (-0.235, 0.0, -0.0085 - EYE[0])),
           "knee_L": ("leg", (0.265, 0.0, -0.007 - EYE[0])), "knee_R": ("leg", (-0.265, 0.0, -0.007 - EYE[0]))}
# two coffee stirrers crossed and tied with thread (the main one along her facing, the other across it in front: the
# hand bar), the leg bar a third: flat birch sticks with round ends, the threads knotted round them
STIRRER = {"length": 1.0, "width": 0.032, "thick": 0.008, "cross_at": -0.30, "leg": 1.0, "knot": 0.005,
           "wrap": 0.0014}             # a coffee stirrer (178 x 5.5 x 1.5 mm) at the theatre's scale: flat, round ends
THREAD = {"radius": 0.0015, "points": 24}
ROLES = {"wood": "gold:base:0.45", "eye": "text:base:0.45", "thread": "text"}
STIRRER_WOOD = "gold:base:0.66"    # birch, paler than the aeroplane's stained wood
MATS = ("wood", "eye", "thread")


def layout(control="aeroplane"):
    """{anchors, hand_bar, leg_bar, main}: where each thread is tied (its bar and the point in that bar's frame), the
    hand bar (its place on the cross, half its length), the leg bar (its home from the cross, half its length) and the
    main bar's ends, for a control `control` (CONTROLS)."""
    if control == "aeroplane":
        return {"anchors": ANCHORS, "hand_bar": HAND_BAR, "leg_bar": LEG_BAR, "main": MAIN}
    if control != "stirrers":
        raise MarionetteError(f"marionette: control = {control!r}: one of {list(CONTROLS)}")
    st = STIRRER
    half, t, k = st["length"] / 2.0, st["thick"], st["knot"]
    under = -t / 2.0 - k
    anchors = {"head": ("cross", (0.0, 0.0, under)), "back": ("cross", (0.0, half - 0.025, under)),
               "wrist_L": ("hand", (half - 0.03, 0.0, under)), "wrist_R": ("hand", (-(half - 0.03), 0.0, under)),
               "knee_L": ("leg", (st["leg"] / 2.0 - 0.03, 0.0, under)),
               "knee_R": ("leg", (-(st["leg"] / 2.0 - 0.03), 0.0, under))}
    return {"anchors": anchors, "hand_bar": {"at": (0.0, st["cross_at"], 0.95 * t), "half": half, "radius": t / 2.0},
            "leg_bar": {"home": LEG_BAR["home"], "half": st["leg"] / 2.0}, "main": (-half, half)}


def _slat(y0, y1, w, t, axis="y", mat=0):
    """A wooden slat from y0 to y1 along y (or, `axis` "x", along x), w wide and t thick: its long edges rounded, its
    ends round in plan (a stadium)."""
    from . import shell as S
    r = min(w / 2.0, (y1 - y0) / 2.0)
    phi = np.linspace(0.0, math.pi / 2, 7)[1:]
    ys = np.concatenate([y0 + r * (1.0 - np.cos(phi)), (y1 - r * (1.0 - np.cos(phi)))[::-1]])
    su = np.maximum(np.concatenate([np.sin(phi), np.sin(phi)[::-1]]), 0.15)
    path = np.stack([np.zeros(len(ys)), ys, np.zeros(len(ys))], 1)
    m = S.sweep(path, S.rrect(w, t, min(0.4 * t, 0.005), n=3), scale=np.stack([su, np.ones(len(su))], 1),
                up=(0.0, 0.0, 1.0), mat=mat)
    return m.rot(rz=-90.0) if axis == "x" else m


def _eye(p, axis="y", mat=1):
    """A screw eye under the wood: its ring hanging at p (the thread's knot), the ring's plane across `axis`."""
    from . import shell as S
    plane = "xz" if axis == "y" else "yz"
    ring = S.tube(S.torus_path(EYE[0], 16, plane, p), EYE[1], sides=6, closed=True, mat=mat)
    shank = S.tube(np.array([[p[0], p[1], p[2] + EYE[0]], [p[0], p[1], p[2] + EYE[0] + 0.006]]), EYE[1] * 1.3,
                   sides=6, mat=mat)
    return S.merge([ring, shank])


def _ring(center, axis, a, b, r, mat=2, n=14):
    """A loop of thread round a stick: an ellipse of semi-axes a (across) and b (up) round `center`, square to the
    stick's `axis` ("x" or "y"), wound of a thread r thick."""
    from . import shell as S
    phi = np.linspace(0.0, 2.0 * math.pi, n, endpoint=False)
    across = np.array([1.0, 0.0, 0.0]) if axis == "y" else np.array([0.0, 1.0, 0.0])
    pts = np.asarray(center, float) + np.outer(a * np.cos(phi), across) + np.outer(b * np.sin(phi), (0.0, 0.0, 1.0))
    return S.tube(pts, r, sides=5, closed=True, mat=mat)


def _knot(p, axis, mat=2):
    """A thread tied round a stirrer at p (the point it hangs from, under the stick): two turns round the stick and the
    knot under it."""
    from . import shell as S
    st = STIRRER
    w, t, r = st["width"], st["thick"], st["wrap"]
    c = (p[0], p[1], 0.0)
    step = np.array([0.0, 2.2 * r, 0.0]) if axis == "y" else np.array([2.2 * r, 0.0, 0.0])
    turns = [_ring(np.asarray(c) + k * step, axis, w / 2.0 + r, t / 2.0 + r, r, mat) for k in (-0.5, 0.5)]
    knot = S.tube(np.array([[p[0], p[1], -t / 2.0 - r], [p[0], p[1], p[2] + r]]), 1.8 * r, sides=6, mat=mat)
    return S.merge(turns + [knot])


def _lashing(center, mat=2):
    """The thread that ties the two stirrers where they cross: two loops round both sticks, along the diagonals."""
    from . import shell as S
    st = STIRRER
    w, t, r = st["width"], st["thick"], st["wrap"]
    loops = []
    for d in (np.array([1.0, 1.0, 0.0]), np.array([1.0, -1.0, 0.0])):
        d = d / np.linalg.norm(d)
        phi = np.linspace(0.0, 2.0 * math.pi, 16, endpoint=False)
        pts = (np.asarray(center, float) + np.outer(0.78 * w * np.cos(phi), d)
               + np.outer((1.05 * t + r) * np.sin(phi), (0.0, 0.0, 1.0)))
        loops.append(S.tube(pts, r, sides=5, closed=True, mat=mat))
    return S.merge(loops)


def cross_meshes(control="aeroplane"):
    """{"cross": the main bar, the head bar on top of it, the hand bar's peg and their eyes (the cross's frame), "hand":
    the hand bar, a dowel, and its eyes (its own frame, at the hand bar's `at`), "leg": the leg bar and its eyes (its
    own frame)}: shell Meshes, material indices of MATS (0 wood, 1 the eyes' metal, 2 thread). `control` "stirrers":
    the main stirrer with the head and back threads knotted round it and the lashing where the other crosses it, that
    other one (the hand bar) with the wrists' threads, a third with the knees'."""
    from . import shell as S
    if control == "stirrers":
        st, lay = STIRRER, layout("stirrers")
        w, t = st["width"], st["thick"]
        A = lay["anchors"]
        half, at = st["length"] / 2.0, lay["hand_bar"]["at"]
        cross = S.merge([_slat(-half, half, w, t), _lashing((0.0, at[1], 0.5 * at[2]))]
                        + [_knot(A[p][1], "y") for p in ("head", "back")])
        hand = S.merge([_slat(-half, half, w, t, axis="x")] + [_knot(A[s][1], "x") for s in ("wrist_L", "wrist_R")])
        leg = S.merge([_slat(-st["leg"] / 2.0, st["leg"] / 2.0, w, t, axis="x")]
                      + [_knot(A[s][1], "x") for s in ("knee_L", "knee_R")])
        return {"cross": cross, "hand": hand, "leg": leg}
    w, t = SLAT
    hb = HAND_BAR
    cross = S.merge([_slat(MAIN[0], MAIN[1], w, t), _slat(-HEAD_BAR, HEAD_BAR, w * 0.9, t, axis="x").moved((0.0, 0.0, t * 0.95)),
                     S.tube(np.array([[0.0, hb["at"][1], -t / 2], [0.0, hb["at"][1], hb["at"][2] + hb["radius"]]]), 0.004,
                            sides=8, mat=1)]
                    + [_eye(ANCHORS[p][1]) for p in ("head", "back")])
    hand = S.merge([S.tube(np.array([[-hb["half"], 0.0, 0.0], [hb["half"], 0.0, 0.0]]), hb["radius"], sides=12)]
                   + [_eye(ANCHORS[s][1], "x") for s in ("wrist_L", "wrist_R")])
    leg = S.merge([_slat(-LEG_BAR["half"], LEG_BAR["half"], w, t, axis="x")]
                  + [_eye(ANCHORS[s][1], "x") for s in ("knee_L", "knee_R")])
    return {"cross": cross, "hand": hand, "leg": leg}


REST = (1.6, 2.6, 0.04)     # m: the surface she lies on (a collider for the sim): across her, along her, thick; top at 0


def card(name, roles, control="aeroplane"):
    """The prop card (docs/design.md: Prop card): the look points that ride the bars (`cross`, `legbar`), the thread
    anchors, the surface she rests on in the world (`<name>_rest`) and where a yank takes her thread (`<name>_grab`),
    and the objects (docs/design.md: Marionette)."""
    lay = layout(control)
    bars = {"cross": f"{name}_cross", "hand": f"{name}_handbar", "leg": f"{name}_legbar"}
    hb, lb, main = lay["hand_bar"], lay["leg_bar"], lay["main"]
    thick = STIRRER["thick"] if control == "stirrers" else SLAT[1]
    return {"size": [round(2 * hb["half"], 3), round(main[1] - main[0], 3), thick * 2], "origin": "center",
            "front": "-Y", "slots": dict(roles), "form_max": 0.5, "form_exempt": [f"{name}_rest"],
            "use": {"look": [{"name": "cross", "object": bars["cross"], "point": [0.0, 0.0, 0.0]},
                             {"name": "legbar", "object": bars["leg"], "point": list(lb["home"])},
                             {"name": "grab", "object": f"{name}_grab", "point": [0.0, 0.0, 0.0]}]},
            "marionette": {"anchors": {p: {"bar": b, "object": f"{name}_at_{p}", "point": list(pt)}
                                       for p, (b, pt) in lay["anchors"].items()},
                           "bars": bars, "hand_bar": dict(hb, at=list(hb["at"])),
                           "leg_bar": dict(lb, home=list(lb["home"])), "thread": dict(THREAD), "control": control,
                           "rest": f"{name}_rest", "grab": f"{name}_grab"},
            "objects": list(bars.values()) + [f"{name}_{k}_{p}" for p in PARTS for k in ("thread", "cut")]
            + [f"{name}_rest", f"{name}_grab"]}
