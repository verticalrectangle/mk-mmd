"""Flights, bpy-free (docs/design.md: Flights): a prop flown relative to a target by a `[[flight]]` table, keyed by the
build's `flight` stage (before perform, so a gaze finds it) and its `wire` stage (after everything that moves the target;
mkmmd/blender/build/flight.py). A flight is a list of moves on clip seconds, each a place relative to the target in the
target's frame (a cast member's: x her left, y behind her, z up; else the world's axes):

  in      {t, from, to, dur = 0.7}       flies in from `from` (where it waits before) and is at `to` ON t
  dart    {t, to, dur = 0.22}            a quick dash that is at `to` ON t, runs a little past it and settles back
  glide   {t, to, dur = 1.0}             an eased glide that is at `to` ON t
  circle  {t, dur = 1.6, turns = 1, radius = 0.4, height = 0, tilt = 20, dir = 1, to}   from t round the target
                                         `turns` times, on a circle `radius` out at `height` (its plane tipped `tilt`
                                         degrees, the side toward the audience, -y, lower), ending at `to` (default:
                                         where it began); `dir` 1 counter-clockwise seen from above
  perch   {t, on = [0, 0, 0], dur = 0.5} lands on the target (`on` from its point) ON t and rides it, wings at rest,
                                         until the next move takes it off
  hover   {t, darts = 0.05, every = 0.5, bob = 0.012}   from t on: the small darts and the bob of its hovering

Between moves it hovers where the last one left it: a quick small dart to another point within `darts` m of the spot
about every `every` s, and a bob. Every move but a perch follows the target late (a critically damped follow `lag` s
behind: a hand on a wire follows her, not every hop of her head); a perch rides it. The prop keeps `clear` m from the
target except perched, faces where it flies and, slow, the target (perched: where she faces); its back tips `show`
degrees toward the audience (the toy theatre's -y) and it banks into its turns.

The wire (a prop whose card has one: the dragonfly): from `hand` (its far end, off stage: a world point or keys
[[t, x, y, z], ...]) to the belly, a cubic from the hand's aim to the belly's way down; its middle lags and overshoots as
spring steel does (each point a damped spring toward that shape, weighted by sin(pi s): the ends are held), and it never
comes within `clear` of the target (nor within an `avoid` sphere's radius of its point)."""
import math

import numpy as np

KINDS = ("in", "dart", "glide", "circle", "perch", "hover")
MOVE_KEYS = {"in": {"from": None, "to": None, "dur": 0.7},
             "dart": {"to": None, "dur": 0.22},
             "glide": {"to": None, "dur": 1.0},
             "circle": {"dur": 1.6, "turns": 1.0, "radius": 0.4, "height": 0.0, "tilt": 20.0, "dir": 1, "to": "start"},
             "perch": {"on": (0.0, 0.0, 0.0), "dur": 0.5},
             "hover": {"darts": 0.05, "every": 0.5, "bob": 0.012}}
KEYS = ("prop", "target", "moves", "hand", "lag", "clear", "show", "avoid", "start", "seed", "wire", "face")
DEFAULTS = {"lag": 0.3, "clear": 0.15, "show": 30.0, "seed": 1}
WIRE = {"freq": 2.4, "damp": 0.2}                   # spring steel: its middle's own wobble (Hz) and damping ratio
POINTS = 32                                         # points along a wire (its card may say otherwise)
HOVER = {"darts": 0.05, "every": 0.5, "bob": 0.012}
DART_ZETA = 0.6                 # a dart's overshoot: 9.5 % past its place, back within a few frames
FLIT = (0.08, 0.55)             # a hover dart: its time and damping
GRAVITY = 9.81


class FlightError(ValueError):
    """A [[flight]] table the build cannot fly: what is wrong and what to change."""


# ===================================================================================================================
# the table
# ===================================================================================================================
def _num(v, what, lo=-math.inf, hi=math.inf):
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(float(v)):
        raise FlightError(f"{what} = {v!r}: a number")
    v = float(v)
    if not lo <= v <= hi:
        raise FlightError(f"{what} = {v:g}: from {lo:g} to {hi:g}")
    return v


def _vec(v, what):
    if not isinstance(v, (list, tuple)) or len(v) != 3:
        raise FlightError(f"{what} = {v!r}: [x, y, z] in metres")
    return tuple(_num(x, what) for x in v)


def _hand(v, where):
    """`hand`: a world point [x, y, z] or keys [[t, x, y, z], ...] in time order -> ((t,), (x, y, z)) lists."""
    if v is None:
        return None
    if isinstance(v, (list, tuple)) and len(v) == 3 and all(not isinstance(x, (list, tuple)) for x in v):
        return [(0.0, _vec(v, f"{where}.hand"))]
    if not isinstance(v, (list, tuple)) or not v:
        raise FlightError(f"{where}.hand = {v!r}: a world point [x, y, z] or keys [[t, x, y, z], ...]")
    out = []
    for i, k in enumerate(v):
        if not isinstance(k, (list, tuple)) or len(k) != 4:
            raise FlightError(f"{where}.hand[{i}] = {k!r}: [t, x, y, z] (clip seconds, a world point)")
        out.append((_num(k[0], f"{where}.hand[{i}] t"), _vec(k[1:], f"{where}.hand[{i}]")))
    if any(b[0] <= a[0] for a, b in zip(out, out[1:])):
        raise FlightError(f"{where}.hand: the keys' times must rise")
    return out


def normalize(spec, index=0):
    """The [[flight]] table `spec` checked, with its defaults: {prop, target, moves (with their windows: `t0`, `t1`),
    hand, lag, clear, show, avoid, start, seed, wire}. `target` is kept as written (the build resolves it)."""
    where = f"flight {index}"
    if not isinstance(spec, dict):
        raise FlightError(f"{where}: expected a table {{prop, target, moves, ...}}")
    unknown = sorted(set(spec) - set(KEYS))
    if unknown:
        raise FlightError(f"{where}: unknown keys {unknown} (known: {', '.join(KEYS)})")
    if not isinstance(spec.get("prop"), str) or not spec["prop"]:
        raise FlightError(f"{where}: prop = {spec.get('prop')!r}: the name of the [[prop]] it flies")
    where = f"flight {index} ({spec['prop']})"
    F = {"prop": spec["prop"], "target": spec.get("target"), "where": where,
         "lag": _num(spec.get("lag", DEFAULTS["lag"]), f"{where}.lag", 0.0, 5.0),
         "clear": _num(spec.get("clear", DEFAULTS["clear"]), f"{where}.clear", 0.0, 5.0),
         "show": _num(spec.get("show", DEFAULTS["show"]), f"{where}.show", -80.0, 80.0),
         "hand": _hand(spec.get("hand"), where),
         "start": None if spec.get("start") is None else _vec(spec["start"], f"{where}.start")}
    seed = spec.get("seed", DEFAULTS["seed"])
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise FlightError(f"{where}.seed = {seed!r}: a whole number, 0 or more")
    F["seed"] = seed
    wire = spec.get("wire", {})
    if not isinstance(wire, dict) or set(wire) - set(WIRE):
        raise FlightError(f"{where}.wire = {wire!r}: a table of {', '.join(WIRE)}")
    F["wire"] = {"freq": _num(wire.get("freq", WIRE["freq"]), f"{where}.wire.freq", 0.2, 20.0),
                 "damp": _num(wire.get("damp", WIRE["damp"]), f"{where}.wire.damp", 0.0, 2.0)}
    avoid = spec.get("avoid", [])
    if not isinstance(avoid, list) or not all(isinstance(a, dict) and set(a) <= {"at", "radius"} and "at" in a
                                              for a in avoid):
        raise FlightError(f"{where}.avoid = {avoid!r}: a list of {{at = <target>, radius = m}}")
    F["avoid"] = [{"at": a["at"], "radius": _num(a.get("radius", 0.15), f"{where}.avoid radius", 0.0, 10.0)}
                  for a in avoid]
    face = spec.get("face")
    if face is not None:
        if not isinstance(face, dict) or "from" not in face or set(face) - {"from", "margin"} or not (
                isinstance(face["from"], str) or isinstance(face["from"], (list, tuple))):
            raise FlightError(f"{where}.face = {face!r}: {{from = <a camera's object name or [x, y, z]>, margin = m}}")
        F["face"] = {"from": face["from"] if isinstance(face["from"], str) else _vec(face["from"], f"{where}.face.from"),
                     "margin": _num(face.get("margin", 0.25), f"{where}.face.margin", 0.0, 5.0)}
    else:
        F["face"] = None
    moves = spec.get("moves")
    if not isinstance(moves, list) or not moves:
        raise FlightError(f"{where}.moves: a list of moves {{t, do, ...}} ({', '.join(KINDS)})")
    F["moves"] = _moves(moves, where, F["clear"])
    return F


def _moves(moves, where, clear):
    out = []
    for i, mv in enumerate(moves):
        w = f"{where}.moves[{i}]"
        if not isinstance(mv, dict) or mv.get("do") not in KINDS:
            raise FlightError(f"{w}: a table {{t, do, ...}} with do one of {', '.join(KINDS)} (got {mv!r})")
        kind = mv["do"]
        known = MOVE_KEYS[kind]
        unknown = sorted(set(mv) - set(known) - {"t", "do"})
        if unknown:
            raise FlightError(f"{w} ({kind}): unknown keys {unknown} (known: t, do, {', '.join(known)})")
        m = {"do": kind, "t": _num(mv.get("t"), f"{w}.t"), "index": i}
        for k, d in known.items():
            v = mv.get(k, d)
            if v is None:
                raise FlightError(f"{w} ({kind}): needs `{k}`")
            if k in ("from", "to", "on"):
                m[k] = v if v == "start" else _vec(v, f"{w}.{k}")
            elif k == "dir":
                if v not in (1, -1):
                    raise FlightError(f"{w}.dir = {v!r}: 1 (counter-clockwise seen from above) or -1")
                m[k] = int(v)
            else:
                lo = {"dur": 0.02, "turns": 0.0, "radius": 0.0, "every": 0.05}.get(k, -math.inf if k in
                                                                                    ("height", "tilt") else 0.0)
                m[k] = _num(v, f"{w}.{k}", lo, 89.0 if k == "tilt" else math.inf)
        if kind == "circle" and m["radius"] < clear + 0.02:
            raise FlightError(f"{w}: radius {m['radius']:g} m passes through the target (clear = {clear:g} m): make "
                              f"it {clear + 0.02:g} or more")
        if kind == "circle":
            m["t0"], m["t1"] = m["t"], m["t"] + m["dur"]
        elif kind == "hover":
            m["t0"] = m["t1"] = m["t"]
        else:
            m["t0"], m["t1"] = m["t"] - m["dur"], m["t"]
        out.append(m)
    out.sort(key=lambda m: (m["t0"], m["index"]))
    travel = [m for m in out if m["do"] != "hover"]
    for a, b in zip(travel, travel[1:]):
        if b["t0"] < a["t1"] - 1e-9:
            raise FlightError(f"{where}.moves[{b['index']}] ({b['do']} {'from' if b['do'] == 'circle' else 'on'} "
                              f"{b['t']:g} s) starts at {b['t0']:g} s, before moves[{a['index']}] ({a['do']}) ends at "
                              f"{a['t1']:g} s: one move at a time")
    return out


# ===================================================================================================================
# shapes of a move in time
# ===================================================================================================================
def smoothstep(a, b, x):
    t = np.clip((np.asarray(x, float) - a) / (b - a), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def smootherstep(u):
    u = np.clip(np.asarray(u, float), 0.0, 1.0)
    return u * u * u * (u * (6.0 * u - 15.0) + 10.0)


def arrive(tau, dur, zeta=DART_ZETA):
    """A damped spring's step response that first reaches 1 at tau = dur (it starts still, runs past 1 and settles):
    1 - exp(-s tau) (cos(w tau) + s / w sin(w tau)), with w = (pi - acos(zeta)) / dur and s = zeta w / sqrt(1 - zeta^2)."""
    tau = np.maximum(np.asarray(tau, float), 0.0)
    w = (math.pi - math.acos(zeta)) / dur
    s = zeta * w / math.sqrt(1.0 - zeta * zeta)
    return 1.0 - np.exp(-s * tau) * (np.cos(w * tau) + (s / w) * np.sin(w * tau))


def trapezoid(u, a=0.2):
    """0..1 over u in 0..1 at a speed that ramps up over the first `a` and down over the last `a` (a whirl round)."""
    u = np.clip(np.asarray(u, float), 0.0, 1.0)
    k = 1.0 / (2.0 * a * (1.0 - a))
    return np.where(u < a, k * u * u, np.where(u > 1.0 - a, 1.0 - k * (1.0 - u) ** 2, (u - 0.5 * a) / (1.0 - a)))


def _window(t, t0, t1, ramp):
    """1 over [t0, t1], easing to 0 over `ramp` s either side."""
    return smoothstep(t0 - ramp, t0, t) * (1.0 - smoothstep(t1, t1 + ramp, t))


def _circle_axes(tilt_deg):
    """(e1, e2, n): the circle's plane (e1 along x, e2 back and up) and its normal, tipped toward -y by tilt."""
    a = math.radians(tilt_deg)
    return np.array([1.0, 0.0, 0.0]), np.array([0.0, math.cos(a), math.sin(a)]), np.array([0.0, -math.sin(a), math.cos(a)])


def _circle(m, p_from, t):
    """Positions on a `circle` move at times t (its window) from p_from, round and on to its end."""
    e1, e2, n = _circle_axes(m["tilt"])
    c = np.array([0.0, 0.0, m["height"]])
    p_to = np.asarray(p_from if m["to"] == "start" else m["to"], float)

    def polar(p):
        d = p - c
        x, y = d @ e1, d @ e2
        return math.hypot(x, y), math.atan2(y, x), float(d @ n)

    rs, ths, hs = polar(np.asarray(p_from, float))
    re, the, he = polar(p_to)
    gap = (the - ths) % (2.0 * math.pi)
    sweep = m["dir"] * (2.0 * math.pi * m["turns"] + (gap if m["dir"] > 0 else (2.0 * math.pi - gap) % (2.0 * math.pi)))
    u = np.clip((np.asarray(t, float) - m["t0"]) / m["dur"], 0.0, 1.0)
    th = ths + sweep * trapezoid(u)
    w = smoothstep(0.0, 0.22, u) * (1.0 - smoothstep(0.78, 1.0, u))
    r = (1.0 - w) * (rs + (re - rs) * u) + w * m["radius"]
    h = (1.0 - w) * (hs + (he - hs) * u)
    return c + (r * np.cos(th))[:, None] * e1 + (r * np.sin(th))[:, None] * e2 + h[:, None] * n


def _flits(F, t_lo, t_hi):
    """The hover's small darts: (times (k,), places (k, 3)) from a seeded walk of places within `darts` of the spot,
    with the `hover` params in force at each time."""
    rng = np.random.default_rng(1000 + F["seed"])
    params = [(m["t"], m) for m in F["moves"] if m["do"] == "hover"]
    times, places = [], []
    t = t_lo + rng.random() * HOVER["every"]
    while t < t_hi:
        p = dict(HOVER)
        for tp, m in params:
            if tp <= t:
                p = {k: m[k] for k in HOVER}
        d = rng.normal(size=3) * np.array([1.0, 0.6, 0.45])
        d /= max(np.linalg.norm(d), 1e-9)
        times.append(t)
        places.append(d * p["darts"] * (0.45 + 0.55 * rng.random()))
        t += max(p["every"], 0.05) * (0.6 + 0.8 * rng.random())
    return np.array(times), np.array(places).reshape(-1, 3)


def _bob_amp(F, t):
    amp = np.full(len(t), HOVER["bob"])
    for m in F["moves"]:
        if m["do"] == "hover":
            amp[t >= m["t"]] = m["bob"]
    return amp


# ===================================================================================================================
# the path relative to the target
# ===================================================================================================================
def path(F, ts):
    """The flight relative to the target at clip seconds ts (n,): {"O" (n, 3) offsets in the target's frame (the moves
    and the hover), "perch" (n,) 0..1 riding the raw target, "rest" (n,) 0..1 the wings at rest, "hover" (n,) 0..1,
    "legs" [{do, t0, t1, from, to}]}."""
    t = np.asarray(ts, float)
    n = len(t)
    travel = [m for m in F["moves"] if m["do"] != "hover"]
    first = travel[0] if travel else None
    start = F["start"]
    if start is None:
        start = (first.get("from") or first.get("to") or (0.3, -0.3, 0.0)) if first else (0.3, -0.3, 0.0)
        start = (0.3, -0.3, 0.0) if start == "start" else start
    O = np.tile(np.asarray(start, float), (n, 1))
    perch, rest, busy = np.zeros(n), np.zeros(n), np.zeros(n)
    legs = []
    fn = None                                       # the current leg: f(t) -> (k, 3)

    def at(time):
        return np.asarray(start, float) if fn is None else fn(np.array([time]))[0]

    for i, m in enumerate(travel):
        p0 = np.asarray(m["from"], float) if m["do"] == "in" else at(m["t0"])
        if m["do"] == "circle":
            def fn(tt, m=m, p0=p0):
                tt = np.asarray(tt, float)
                inside = _circle(m, p0, np.clip(tt, m["t0"], m["t1"]))
                return np.where((tt < m["t0"])[:, None], p0, inside)
            p1 = fn(np.array([m["t1"]]))[0]
        else:
            p1 = np.asarray(m["on"] if m["do"] == "perch" else m["to"], float)
            if m["do"] == "dart":
                def fn(tt, m=m, p0=p0, p1=p1):
                    s = arrive(np.asarray(tt, float) - m["t0"], m["dur"])
                    return p0 + (p1 - p0) * s[:, None]
            else:
                def fn(tt, m=m, p0=p0, p1=p1):
                    s = smootherstep((np.asarray(tt, float) - m["t0"]) / m["dur"])
                    out = p0 + (p1 - p0) * s[:, None]
                    if m["do"] == "perch":                  # it lands: a small give and back
                        tau = np.maximum(np.asarray(tt, float) - m["t1"], 0.0)
                        out[:, 2] -= 0.008 * np.exp(-tau / 0.09) * np.sin(math.pi * np.minimum(tau / 0.18, 1.0))
                    return out
        nxt = travel[i + 1]["t0"] if i + 1 < len(travel) else math.inf
        sel = (t >= m["t0"]) & (t < nxt)
        if m["do"] == "in":
            O[t < m["t0"]] = p0
        if sel.any():
            O[sel] = fn(t[sel])
        busy = np.maximum(busy, _window(t, m["t0"], m["t1"] + (0.12 if m["do"] == "dart" else 0.05), 0.07))
        if m["do"] == "perch":
            def off_at(ramp, nxt=nxt):                    # 1 until the next move takes it off (none: for good)
                return np.ones(n) if math.isinf(nxt) else 1.0 - smoothstep(nxt, nxt + ramp, t)
            perch = np.maximum(perch, smoothstep(m["t0"], m["t1"], t) * off_at(0.15))
            rest = np.maximum(rest, smoothstep(m["t1"] - 0.03, m["t1"] + 0.2, t) * off_at(0.06))
        legs.append({"do": m["do"], "t0": round(m["t0"], 4), "t1": round(m["t1"], 4),
                     "from": [round(float(v), 4) for v in p0], "to": [round(float(v), 4) for v in p1]})
    hover = (1.0 - busy) * (1.0 - smoothstep(0.0, 0.5, perch))
    lo, hi = (float(t[0]), float(t[-1])) if n else (0.0, 0.0)
    ft, fp = _flits(F, lo, hi + 1.0)
    J = np.zeros((n, 3))
    prev = np.zeros(3)
    for tk, pk in zip(ft, fp):
        J += (pk - prev)[None] * arrive(t - tk, FLIT[0], FLIT[1])[:, None] * (t >= tk)[:, None]
        prev = pk
    amp = _bob_amp(F, t)
    ph = 2.0 * math.pi * (F["seed"] * 0.37 % 1.0)
    J[:, 2] += amp * np.sin(2.0 * math.pi * 1.7 * t + ph)
    J[:, 0] += 0.4 * amp * np.sin(2.0 * math.pi * 1.1 * t + 2.0 * ph)
    O = O + J * hover[:, None]
    return {"O": O, "perch": perch, "rest": rest, "hover": hover, "legs": legs}


# ===================================================================================================================
# in the world
# ===================================================================================================================
def follow(X, dt, tau):
    """A causal, critically damped follow of X (n, d) sampled every dt s, `tau` s behind it, starting on X[0]:
    y'' = (X - y) / tau^2 - 2 y' / tau (a hand that follows late, never past where it is going)."""
    X = np.asarray(X, float)
    if tau <= 0.0 or len(X) == 0:
        return X.copy()
    w = 1.0 / tau
    sub = max(1, int(math.ceil(4.0 * dt * w)))
    h = dt / sub
    y, v = X[0].copy(), np.zeros_like(X[0])
    out = np.empty_like(X)
    for i in range(len(X)):
        for _ in range(sub):
            v = v + (w * w * (X[i] - y) - 2.0 * w * v) * h
            y = y + v * h
        out[i] = y
    return out


def off_line(P, A, B, r, keep=None, down=0.35):
    """Points P (n, 3) pushed out of the capsule of radius r round the segment from A (n, 3) to B (n, 3) (the line
    between a camera and her face), straight away from it and a little downward (under her chin rather than over her
    eyes); `keep` (n,) bool: frames left alone."""
    P = np.array(P, float)
    A, B = np.asarray(A, float), np.asarray(B, float)
    u = B - A
    L = np.maximum(np.linalg.norm(u, axis=-1, keepdims=True), 1e-9)
    u = u / L
    s = np.clip(((P - A) * u).sum(-1, keepdims=True), 0.0, L)
    c = A + u * s
    d = P - c
    dist = np.linalg.norm(d, axis=-1)
    inside = (dist < r) & (s[..., 0] < L[..., 0] - 1e-6)
    if keep is not None:
        inside &= ~np.asarray(keep, bool).reshape(inside.shape[:1] + (1,) * (inside.ndim - 1))
    if not inside.any():
        return P
    dn = np.where((dist > 1e-6)[..., None], d / np.maximum(dist, 1e-9)[..., None], 0.0)
    dn = dn + np.array([0.0, 0.0, -down])
    dn = dn - (dn * u).sum(-1, keepdims=True) * u
    dn /= np.maximum(np.linalg.norm(dn, axis=-1, keepdims=True), 1e-9)
    P[inside] = (c + dn * r)[inside]
    return P


def place(F, ts, T, R, P_rel=None, eye=None):
    """World positions (n, 3) of the flown prop at clip seconds ts: the target T (n, 3) followed `lag` late, plus the
    path's offsets turned by R (3, 3: the target's frame), riding the raw target where it perches, and kept `clear` of
    it; with `eye` (n, 3: the camera of `face`) kept `face.margin` off the line from it to the target. `P_rel`: the path
    (`path(F, ts)`) when it is at hand."""
    ts = np.asarray(ts, float)
    T = np.asarray(T, float)
    pr = path(F, ts) if P_rel is None else P_rel
    dt = float(ts[1] - ts[0]) if len(ts) > 1 else 1.0 / 60.0
    Ts = follow(T, dt, F["lag"])
    off = pr["O"] @ np.asarray(R, float).T
    w = pr["perch"][:, None]
    P = (1.0 - w) * (Ts + off) + w * (T + off)
    if eye is not None and F.get("face"):
        P = off_line(P, eye, T, F["face"]["margin"], keep=pr["perch"] >= 0.5)
        P = (1.0 - w) * follow(P, dt, 0.03) + w * P           # the push eased in, never a jump
    d = P - T
    dist = np.linalg.norm(d, axis=1)
    push = (dist < F["clear"]) & (pr["perch"] < 0.5)
    if push.any():                                    # never in the target: out along the way it came, up if dead on
        dd = np.where(dist[:, None] > 1e-6, d / np.maximum(dist, 1e-9)[:, None], np.array([0.0, 0.0, 1.0]))
        P[push] = T[push] + dd[push] * F["clear"]
    return P


def attitude(P, T, ts, R, perch, show_deg):
    """The flown prop's frame (n, 3, 3): columns its left, back and up in the world (its head toward -back). It faces
    where it flies, and the target when it is slow (perched: where the target's frame faces, -y), its back tipped
    `show_deg` toward the audience (-y of R), banking into its turns."""
    P, T = np.asarray(P, float), np.asarray(T, float)
    n = len(P)
    ts = np.asarray(ts, float)
    dt = float(ts[1] - ts[0]) if n > 1 else 1.0 / 60.0
    R = np.asarray(R, float)
    if n < 2:
        V = np.zeros((n, 3))
    else:
        V = follow(np.gradient(P, dt, axis=0), dt, 0.04)
    speed = np.linalg.norm(V, axis=1)
    trav = V / np.maximum(speed, 1e-9)[:, None]
    trav[:, 2] = np.clip(trav[:, 2], -0.45, 0.45)
    ahead = R @ np.array([0.0, -1.0, 0.0])
    to_t = T - P
    to_t[:, 2] = 0.0
    k = np.linalg.norm(to_t, axis=1)
    to_t = np.where(k[:, None] > 1e-6, to_t / np.maximum(k, 1e-9)[:, None], ahead)
    w = smoothstep(0.12, 0.45, speed)[:, None]
    f = (1.0 - w) * to_t + w * trav
    pw = np.asarray(perch, float)[:, None]
    f = (1.0 - pw) * f + pw * ahead
    f /= np.maximum(np.linalg.norm(f, axis=1, keepdims=True), 1e-9)
    f = follow(f, dt, 0.05)
    f /= np.maximum(np.linalg.norm(f, axis=1, keepdims=True), 1e-9)
    s = math.radians(show_deg)
    up0 = R @ np.array([0.0, -math.sin(s), math.cos(s)])
    up = up0[None] - (f @ up0)[:, None] * f
    weak = np.linalg.norm(up, axis=1) < 0.2
    if weak.any():
        z = np.array([0.0, 0.0, 1.0])
        up[weak] = z - (f[weak] @ z)[:, None] * f[weak]
    up /= np.maximum(np.linalg.norm(up, axis=1, keepdims=True), 1e-9)
    left = np.cross(up, f)
    A = np.gradient(V, dt, axis=0) if n > 1 else np.zeros((n, 3))
    bank = np.clip(0.9 * np.arctan2((A * left).sum(1), GRAVITY), -0.6, 0.6) * (1.0 - pw[:, 0])
    bank = follow(bank[:, None], dt, 0.05)[:, 0]
    up = up + left * np.tan(bank)[:, None]                 # into the turn: the back toward the turn's centre
    up /= np.maximum(np.linalg.norm(up, axis=1, keepdims=True), 1e-9)
    left = np.cross(up, f)
    left /= np.maximum(np.linalg.norm(left, axis=1, keepdims=True), 1e-9)
    return np.stack([left, -f, up], axis=2)


def hand_track(F, ts, default):
    """The wire's far end (n, 3) at clip seconds ts: the `hand` keys eased between (smoothstep), held past the ends;
    `default` (a world point) without any."""
    ts = np.asarray(ts, float)
    keys = F.get("hand") or [(0.0, tuple(default))]
    kt = np.array([k[0] for k in keys])
    kp = np.array([k[1] for k in keys], float)
    if len(kt) == 1:
        return np.tile(kp[0], (len(ts), 1))
    i = np.clip(np.searchsorted(kt, ts, side="right") - 1, 0, len(kt) - 2)
    u = smoothstep(0.0, 1.0, (ts - kt[i]) / (kt[i + 1] - kt[i]))
    return kp[i] + (kp[i + 1] - kp[i]) * u[:, None]


# ===================================================================================================================
# the wire
# ===================================================================================================================
def _bezier(P0, P1, P2, P3, s):
    s = s[None, :, None]
    a, b = (1.0 - s) ** 3, 3.0 * (1.0 - s) ** 2 * s
    c, d = 3.0 * (1.0 - s) * s * s, s ** 3
    return a * P0[:, None] + b * P1[:, None] + c * P2[:, None] + d * P3[:, None]


def wire_rest(H, D, down, ts, points=POINTS, reach=0.25):
    """The wire's shape at rest (n, points, 3) from the hand H (n, 3) to the belly D (n, 3): a cubic leaving the hand
    along its aim (toward where the belly has been, half a second late) and the belly downward (`down` (n, 3), the
    prop's -z) and toward the hand."""
    H, D, down = (np.asarray(a, float) for a in (H, D, down))
    ts = np.asarray(ts, float)
    dt = float(ts[1] - ts[0]) if len(ts) > 1 else 1.0 / 60.0
    aim = follow(D, dt, 0.5) - H
    aim /= np.maximum(np.linalg.norm(aim, axis=1, keepdims=True), 1e-9)
    L = np.linalg.norm(D - H, axis=1)
    back = H - D
    back[:, 2] = 0.0
    back /= np.maximum(np.linalg.norm(back, axis=1, keepdims=True), 1e-9)
    leave = 0.7 * down + 0.7 * back
    leave /= np.maximum(np.linalg.norm(leave, axis=1, keepdims=True), 1e-9)
    P1 = H + aim * (0.4 * L)[:, None]
    P2 = D + leave * np.minimum(0.3 * L, reach)[:, None]
    return _bezier(H, P1, P2, D, np.linspace(0.0, 1.0, points))


def keep_out(W, spheres, passes=4, capsules=()):
    """The wire's points W (n, k, 3) pushed out of every sphere (centre (n, 3), radius) and every capsule (A (n, 3),
    B (n, 3), radius) and smoothed, its ends held: the last pass only pushes, so no point is left inside."""
    W = np.array(W, float)
    for p in range(passes + 1):
        for c, r in spheres:
            d = W - np.asarray(c, float)[:, None]
            dist = np.linalg.norm(d, axis=2)
            inside = dist < r
            inside[:, 0] = inside[:, -1] = False
            if not inside.any():
                continue
            dd = np.where((dist > 1e-6)[..., None], d / np.maximum(dist, 1e-9)[..., None], np.array([0.0, 0.0, 1.0]))
            dd = dd + np.array([0.0, 0.0, 0.15]) * inside[..., None]          # lifted over it rather than round it
            dd /= np.maximum(np.linalg.norm(dd, axis=2, keepdims=True), 1e-9)
            W[inside] = (np.asarray(c, float)[:, None] + dd * r)[inside]
        for a, b, r in capsules:
            inner = off_line(W[:, 1:-1], np.asarray(a, float)[:, None], np.asarray(b, float)[:, None], r, down=0.0)
            W[:, 1:-1] = inner
        if p < passes:
            W[:, 1:-1] = 0.5 * W[:, 1:-1] + 0.25 * (W[:, :-2] + W[:, 2:])
    return W


def wire(H, D, down, ts, spheres=(), freq=WIRE["freq"], damp=WIRE["damp"], points=POINTS, reach=0.25, capsules=()):
    """The wire (n, points, 3) from the hand H to the belly D on every frame of ts: its shape at rest (`wire_rest`) with
    each point a damped spring (`freq` Hz, `damp`) toward it, so the middle lags behind and overshoots as spring steel
    does, the ends held (weight sin(pi s)), then kept out of the `spheres` [(centres (n, 3), radius), ...]."""
    ts = np.asarray(ts, float)
    S = wire_rest(H, D, down, ts, points, reach)
    n = len(S)
    dt = float(ts[1] - ts[0]) if n > 1 else 1.0 / 60.0
    w = 2.0 * math.pi * freq
    sub = max(1, int(math.ceil(8.0 * dt * w)))
    h = dt / sub
    X, V = S[0].copy(), np.zeros_like(S[0])
    out = np.empty_like(S)
    for i in range(n):
        for _ in range(sub):
            V = V + (w * w * (S[i] - X) - 2.0 * damp * w * V) * h
            X = X + V * h
        out[i] = X
    m = np.sin(math.pi * np.linspace(0.0, 1.0, points))[None, :, None]
    L = np.linalg.norm(np.asarray(D, float) - np.asarray(H, float), axis=1)[:, None, None]
    dev = (out - S) * m
    big = np.linalg.norm(dev, axis=2, keepdims=True)
    dev = dev * np.minimum(1.0, 0.3 * L / np.maximum(big, 1e-9))     # a wire bends, it does not fold
    return keep_out(S + dev, spheres, capsules=capsules)


def clearance(W, spheres):
    """The least distance (m) from the wire's inner points W (n, k, 3) to the spheres' surfaces (negative: inside),
    and the frame index it happens on."""
    best, at = math.inf, -1
    for c, r in spheres:
        d = np.linalg.norm(np.asarray(W, float)[:, 1:-1] - np.asarray(c, float)[:, None], axis=2) - r
        i = int(np.argmin(d.min(1)))
        if float(d[i].min()) < best:
            best, at = float(d[i].min()), i
    return best, at
