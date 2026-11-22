"""A paper planet on a skewer, bpy-free (the library prop `paper_planet`, docs/design.md: the library table): a ball of
paper gores as a crafter makes a globe (twelve gores in two tones, puffed a little between seams that stand as creases,
a calotte over the top pole flattened into a landing spot), cut-paper craters glued on in two layers, on a wooden
skewer that comes up from below, and a toothpick flag stuck in its shoulder (a paper cocktail flag, striped, no
lettering); and its motion: it rises into place by `up`, turns slowly, and when something lands on it (`land`) it sinks
on its skewer, bobs, tips and settles a little lower, the flag whipping.

Frame: the origin is the ball's middle in place (risen, nothing on it), +Z up the skewer, front -Y. The meshes are at
unit diameter; a builder scales them by `size` (0.8 m: a planet a figure 1.5 m tall stands on; the skewer, pick and
flag in a real craft ball's proportions: a 3 mm skewer and a 65 mm cocktail pick on an 80 mm ball, made ten times).
Photographs used: paper globes built from gores (a gore's seams, the polar calotte), cocktail flags on picks stuck in
food (the flag on the pick's top third, about 3 : 2, the pick standing out three flags' heights)."""
import functools
import math

import numpy as np

from . import flight as FL
from . import shell as S

R = 0.5                          # the ball's radius at unit diameter
GORES = 12
PUFF = 0.012                     # a gore's puff between its seams (a share of R)
FLAT = 0.03125                   # the top flattened this far (unit diameter; 25 mm at 0.8 m): a landing spot
Z_TOP = R - FLAT
SKEWER = {"r": 0.019, "below": 2.4, "tip": 0.22}       # its radius, how far it runs down from the middle, its tip in
PICK = {"r": 0.011, "length": 0.78, "in": 0.13}         # the toothpick: radius, length, how deep it is stuck in
FLAG = {"w": 0.36, "h": 0.24, "thick": 0.004, "top": 0.035}   # the flag: out from the pick, along it, paper, its top
FLAG_AT = (42.0, 35.0)           # where the pick goes in: degrees down from the top, round from +X toward +Y (back)
MATS = ("gore", "gore2", "cap", "crater", "crater_in", "skewer", "pick")
MI = {k: i for i, k in enumerate(MATS)}
ROLES = {"gore": "love", "gore2": "love:rose:0.4", "cap": "gold", "crater": "rose:gold:0.3", "crater_in": "love:text:0.45",
         "skewer": "gold:base:0.55", "pick": "gold:base:0.7"}
ROLE_DOCS = {"gore": "every other gore", "gore2": "the gores between", "cap": "the calotte over the landing spot",
             "crater": "the craters' outer paper (their rims)", "crater_in": "the craters' inner paper",
             "skewer": "the wooden skewer", "pick": "the toothpick"}
STRIPES = ("foam", "paper", "iris")                     # the flag's stripes, top to bottom
DEFAULTS = {"size": 0.8, "up": None, "rise": 0.7, "depth": 1.6, "turn": 0.03, "land": (), "sink": 0.08,
            "settle": 0.018, "whip": 1.5, "step": 0.0, "step_from": 0.0, "seed": 3, "craters": 7}
BOB = {"hz": 2.1, "decay": 0.3, "tip": 3.0}              # a landing's bob: its rate, its decay (s), its tip (degrees)
WHIP = {"hz": 3.2, "decay": 0.35}                        # the flag's whip


# ===================================================================================================================
# the ball
# ===================================================================================================================
def puff(phi, theta):
    """The ball's radius over R at azimuth phi and polar angle theta: each gore puffed between its seams."""
    u = (np.asarray(phi, float) * GORES / (2.0 * math.pi)) % 1.0
    return 1.0 + PUFF * np.sin(math.pi * u) ** 0.6 * np.sin(np.asarray(theta, float)) ** 2


def surface(dirs):
    """Points on the ball's surface (k, 3) along unit directions `dirs` (k, 3): the gores' puff, the top flattened."""
    d = np.asarray(dirs, float)
    th = np.arccos(np.clip(d[:, 2], -1.0, 1.0))
    ph = np.arctan2(d[:, 1], d[:, 0])
    p = d * (R * puff(ph, th))[:, None]
    over = p[:, 2] > Z_TOP
    p[over] *= (Z_TOP / p[over, 2])[:, None]
    return p


THETA_CAP = math.acos(Z_TOP / R)


def ball_mesh(cols_per_gore=12, rings=72):
    """The ball (closed): a grid of rings from the top pole to the bottom one (an apex at each), a ring on the calotte's
    edge; the gores in their two tones, the calotte over the top."""
    m = GORES * cols_per_gore
    cap_rings = 6
    th = np.concatenate([np.linspace(0.0, THETA_CAP, cap_rings + 1)[1:],
                         np.linspace(THETA_CAP, math.pi, rings - cap_rings + 1)[1:-1]])
    ph = 2.0 * math.pi * np.arange(m) / m
    TH, PH = np.meshgrid(th, ph, indexing="ij")
    d = np.stack([np.sin(TH) * np.cos(PH), np.sin(TH) * np.sin(PH), np.cos(TH)], -1).reshape(-1, 3)
    V = np.vstack([[0.0, 0.0, Z_TOP], surface(d), [0.0, 0.0, -R]])
    nr = len(th)
    idx = 1 + np.arange(nr * m).reshape(nr, m)
    j, j2 = np.arange(m), (np.arange(m) + 1) % m
    T_top = np.stack([np.zeros(m, int), idx[0][j2], idx[0][j]], 1)            # seen from outside: counter-clockwise
    bot = len(V) - 1
    T_bot = np.stack([np.full(m, bot), idx[-1][j], idx[-1][j2]], 1)
    Q = np.concatenate([np.stack([idx[i][j], idx[i][j2], idx[i + 1][j2], idx[i + 1][j]], 1) for i in range(nr - 1)])
    mesh = S.Mesh(V, Q, np.concatenate([T_top, T_bot]))
    if mesh.volume() < 0:
        mesh = mesh.flip()
    ring = np.repeat(np.arange(nr - 1), m)
    gore = np.tile(j // cols_per_gore, nr - 1) % 2
    mesh.Qm = np.where(ring < cap_rings - 1, MI["cap"], np.where(gore == 0, MI["gore"], MI["gore2"])).astype(np.int32)
    mesh.Tm = np.concatenate([np.full(m, MI["cap"]), np.where((j // cols_per_gore) % 2 == 0, MI["gore"], MI["gore2"])]
                             ).astype(np.int32)
    return mesh


# ===================================================================================================================
# craters: cut-paper discs glued over the gores, a rim disc and a smaller one in it
# ===================================================================================================================
def _tangent_frame(c):
    c = np.asarray(c, float) / np.linalg.norm(c)
    a = np.array([0.0, 0.0, 1.0]) if abs(c[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    e1 = np.cross(a, c)
    e1 /= np.linalg.norm(e1)
    return c, e1, np.cross(c, e1)


def paper_disc(centre, radius, lift, thick, mat, seed, wobble=0.07, seg=40, rings=5):
    """A cut-paper disc (closed) glued on the ball: its outline a scissor-cut circle (a few wavy lobes), its two faces
    conformed to the surface `lift` and `lift + thick` over it, the cut edge between them."""
    c, e1, e2 = _tangent_frame(centre)
    rng = np.random.default_rng(seed)
    a = 2.0 * math.pi * np.arange(seg) / seg
    k = rng.integers(3, 6)
    rr = radius * (1.0 + wobble * (0.6 * np.sin(k * a + rng.random() * 6.3) + 0.4 * np.sin((k + 2) * a + rng.random() * 6.3)))
    fr = np.linspace(0.0, 1.0, rings + 1)[1:]

    def on(r_frac, h):
        pts = []
        for f in r_frac:
            off = (f * rr / R)[:, None] * (np.cos(a)[:, None] * e1 + np.sin(a)[:, None] * e2)
            d = c + off
            d /= np.linalg.norm(d, axis=1, keepdims=True)
            pts.append(surface(d) + d * h)
        return np.concatenate(pts)

    top = np.vstack([surface(c[None]) + c * (lift + thick), on(fr, lift + thick)])
    bot = np.vstack([surface(c[None]) + c * lift, on(fr, lift)])
    nv = len(top)
    V = np.vstack([top, bot])
    j, j2 = np.arange(seg), (np.arange(seg) + 1) % seg
    Ts, Qs = [np.stack([np.zeros(seg, int), 1 + j, 1 + j2], 1)], []
    for i in range(rings - 1):
        r0, r1 = 1 + i * seg, 1 + (i + 1) * seg
        Qs.append(np.stack([r0 + j, r1 + j, r1 + j2, r0 + j2], 1))
    Tb = [np.stack([np.full(seg, nv), nv + 1 + j2, nv + 1 + j], 1)]
    Qb = [np.stack([nv + r0 + j2, nv + r1 + j2, nv + r1 + j, nv + r0 + j], 1)
          for r0, r1 in ((1 + i * seg, 1 + (i + 1) * seg) for i in range(rings - 1))]
    edge_t, edge_b = 1 + (rings - 1) * seg, nv + 1 + (rings - 1) * seg
    Qe = [np.stack([edge_t + j, edge_b + j, edge_b + j2, edge_t + j2], 1)]
    mesh = S.Mesh(V, np.concatenate(Qs + Qb + Qe), np.concatenate(Ts + Tb), mat=mat)
    return mesh if mesh.volume() >= 0 else mesh.flip()


def crater_spots(n=7, seed=3):
    """n crater places [(direction (3,), radius)] round the ball, clear of the landing spot, of the flag's hole and of
    each other, more on the upper half (seen from a camera at the planet's height or above)."""
    rng = np.random.default_rng(seed)
    flag = flag_point()
    out = []
    tries = 0
    while len(out) < n and tries < 4000:
        tries += 1
        z = rng.uniform(-0.5, 0.75)
        a = -0.5 * math.pi + rng.uniform(-1.9, 1.9) if len(out) < n - 2 else rng.uniform(0.0, 2.0 * math.pi)
        d = np.array([math.sqrt(1.0 - z * z) * math.cos(a), math.sqrt(1.0 - z * z) * math.sin(a), z])
        r = rng.uniform(0.06, 0.12)                      # most face the audience (-y), where a picture sees them
        ang = r / R
        if math.acos(min(1.0, d[2])) < THETA_CAP + ang + 0.06:
            continue
        if math.acos(float(np.clip(d @ flag, -1.0, 1.0))) < ang + 0.16:
            continue
        if any(math.acos(float(np.clip(d @ e, -1.0, 1.0))) < ang + re / R + 0.05 for e, re in out):
            continue
        out.append((d, r))
    return out


def craters_mesh(n=7, seed=3):
    parts = []
    for k, (d, r) in enumerate(crater_spots(n, seed)):
        parts.append(paper_disc(d, r, 0.0015, 0.0035, MI["crater"], seed * 100 + k))
        parts.append(paper_disc(d, 0.62 * r, 0.0052, 0.0035, MI["crater_in"], seed * 100 + 50 + k, wobble=0.09))
    return S.merge(parts)


# ===================================================================================================================
# the skewer, the toothpick and its flag
# ===================================================================================================================
def skewer_mesh():
    """The skewer along -Z from its tip inside the ball down out of the picture: a plain round stick, a little thicker
    as it goes down, its tip a point."""
    r, below, tip = SKEWER["r"], SKEWER["below"], SKEWER["tip"]
    prof = [(0.0, tip), (0.35 * r, tip - 0.02), (0.8 * r, tip - 0.06), (r, tip - 0.12), (1.06 * r, -below + 0.004),
            (0.6 * r, -below), (0.0, -below)]
    return S.lathe(np.array(prof[::-1]), seg=20, mat=MI["skewer"])


def flag_point():
    """The unit direction from the ball's middle to where the pick goes in."""
    down, round_ = (math.radians(v) for v in FLAG_AT)
    return np.array([math.sin(down) * math.cos(round_), math.sin(down) * math.sin(round_), math.cos(down)])


def pick_frame():
    """(origin (3,), axis (3,), out (3,)) of the toothpick: where it goes in (on the surface), along it (up and out),
    and the way the flag stands off it (across the pick, toward +X: seen face on from the front)."""
    d = flag_point()
    o = surface(d[None])[0]
    a = d + np.array([0.0, 0.0, 0.45])
    a /= np.linalg.norm(a)
    x = np.array([1.0, 0.0, 0.0]) - a * a[0]
    x /= np.linalg.norm(x)
    return o, a, x


def pick_mesh():
    """The toothpick in its own frame: along +Z from its point stuck in the ball (z = -PICK in) out to its other
    point, double-pointed and turned."""
    r, L, inside = PICK["r"], PICK["length"], PICK["in"]
    z0, z1 = -inside, L - inside
    prof = [(0.0, z0), (0.5 * r, z0 + 0.03), (r, z0 + 0.07), (r, z1 - 0.07), (0.5 * r, z1 - 0.03), (0.0, z1)]
    return S.lathe(np.array(prof), seg=14, mat=MI["pick"])


def flag_sheet(stripes=len(STRIPES), nu=18, nv=12):
    """The paper flag in the flag's frame (x out from the pick, y along it, z through the paper): a sheet
    FLAG w x h x thick from x = the pick's radius, its stripes across y (top first, material i the i-th stripe);
    closed. Returns (mesh, the shape key `whip`'s offsets (k, 3): the free edge curled back over z, the more the
    further out)."""
    w, h, t = FLAG["w"], FLAG["h"], FLAG["thick"]
    x0 = PICK["r"] * 0.9
    xs = x0 + np.linspace(0.0, w, nu + 1)
    ys = np.linspace(-h, 0.0, nv * stripes + 1)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    top = np.stack([X, Y, np.full_like(X, 0.5 * t)], -1).reshape(-1, 3)
    bot = np.stack([X, Y, np.full_like(X, -0.5 * t)], -1).reshape(-1, 3)
    nx, ny = X.shape
    idx = np.arange(nx * ny).reshape(nx, ny)
    Qs, mats = [], []
    for i in range(nx - 1):
        for k in range(ny - 1):
            q = (idx[i, k], idx[i + 1, k], idx[i + 1, k + 1], idx[i, k + 1])
            band = stripes - 1 - min(stripes - 1, k // nv)                     # y rises: the top stripe is band 0
            Qs.append(q)
            mats.append(band)
    Qt = np.array(Qs)
    off = nx * ny
    Qb = Qt[:, ::-1] + off
    ring = np.concatenate([idx[:, 0], idx[-1, 1:], idx[::-1, -1][1:], idx[0, ::-1][1:-1]])
    r2 = np.roll(ring, -1)
    Qe = np.stack([ring, ring + off, r2 + off, r2], 1)
    edge_band = np.zeros(len(Qe), int)
    mesh = S.Mesh(np.vstack([top, bot]), np.concatenate([Qt, Qb, Qe]),
                  Qm=np.concatenate([mats, mats, edge_band]).astype(np.int32))
    if mesh.volume() < 0:
        mesh = mesh.flip()
    V = mesh.V
    u = np.clip((V[:, 0] - x0) / w, 0.0, 1.0)
    offs = np.zeros_like(V)
    offs[:, 2] = 0.32 * h * u ** 2 * (1.0 + 0.25 * np.sin(math.pi * (V[:, 1] / h)))
    offs[:, 0] = -0.12 * w * u ** 3
    return mesh, offs


def flag_place():
    """Where the flag's top edge is along the pick (its frame's z, unit diameter): FLAG top under its outer point."""
    return PICK["length"] - PICK["in"] - FLAG["top"]


@functools.cache
def parts(craters=7, seed=3):
    """The meshes at unit diameter (cached: read-only): {"ball", "craters", "skewer", "pick", "flag": Mesh, "whip":
    (k, 3) offsets of the flag's shape key}."""
    flag, whip = flag_sheet()
    return {"ball": ball_mesh(), "craters": craters_mesh(craters, seed), "skewer": skewer_mesh(), "pick": pick_mesh(),
            "flag": flag, "whip": whip}


# ===================================================================================================================
# the motion
# ===================================================================================================================
def _num(v, what, lo, hi):
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(float(v)) or not lo <= v <= hi:
        raise ValueError(f"paper_planet: {what} is a number from {lo:g} to {hi:g} (got {v!r})")
    return float(v)


def options(slots):
    """The prop's options from its `slots`, checked: {size, up, rise, depth, turn, land (sorted clip seconds), sink,
    settle, whip, step, step_from, seed, craters, stripes}."""
    slots = dict(slots or {})
    O = {k: _num(slots.get(k, DEFAULTS[k]), k, lo, hi) for k, lo, hi in (
        ("size", 0.01, 20.0), ("rise", 0.05, 10.0), ("depth", 0.0, 50.0), ("turn", -5.0, 5.0), ("sink", 0.0, 2.0),
        ("settle", 0.0, 2.0), ("whip", 0.0, 3.0), ("step", 0.0, 240.0), ("step_from", -1e4, 1e4))}
    up = slots.get("up", DEFAULTS["up"])
    O["up"] = None if up is None else _num(up, "up", -1e4, 1e4)
    land = slots.get("land", DEFAULTS["land"])
    land = [land] if isinstance(land, (int, float)) and not isinstance(land, bool) else land
    if not isinstance(land, (list, tuple)):
        raise ValueError(f"paper_planet: land is a clip second or a list of them (got {land!r})")
    O["land"] = sorted(_num(v, "land", -1e4, 1e4) for v in land)
    for k in ("seed", "craters"):
        v = slots.get(k, DEFAULTS[k])
        if isinstance(v, bool) or not isinstance(v, int) or not 0 <= v <= (10 ** 6 if k == "seed" else 30):
            raise ValueError(f"paper_planet: {k} is a whole number (got {v!r})")
        O[k] = v
    stripes = slots.get("stripes", list(STRIPES))
    if not isinstance(stripes, (list, tuple)) or not 1 <= len(stripes) <= 6 or not all(isinstance(c, str) for c in stripes):
        raise ValueError(f"paper_planet: stripes is a list of one to six colours, the flag's stripes from the top "
                         f"(got {stripes!r})")
    O["stripes"] = [str(c) for c in stripes]
    return O


def motion(O, t):
    """The planet's motion at clip seconds t (n,): {"z" its rise and bob (m), "tip_x", "tip_y" its
    tipping (radians, about X and Y), "spin" (radians about Z), "whip" (the flag's shape key, -1.5..1.5), "yaw" (the flag
    turning about its pick, radians)}. Everything that travels arrives on its time: in place ON `up` (it sets off `rise`
    before), and a landing's sink starts ON its `land`."""
    t = np.asarray(t, float)
    z = np.zeros(len(t))
    whip = 0.18 * np.sin(2.0 * math.pi * 0.9 * t) * min(1.0, abs(O["turn"]) * 20.0)
    yaw = np.zeros(len(t))
    tip_x, tip_y = np.zeros(len(t)), np.zeros(len(t))
    if O["up"] is not None:
        t0 = O["up"] - O["rise"]
        z += -O["depth"] * (1.0 - FL.arrive(t - t0, O["rise"], 0.8))
        z[t < t0] = -O["depth"]
        tau = np.maximum(t - O["up"], 0.0)
        on = t >= O["up"]
        kick = 0.7 * O["whip"] * np.exp(-tau / WHIP["decay"]) * np.sin(2.0 * math.pi * WHIP["hz"] * tau) * on
        whip += kick
        yaw += 0.25 * kick
    w = 2.0 * math.pi * BOB["hz"]
    for k, tl in enumerate(O["land"]):
        tau = np.maximum(t - tl, 0.0)
        on = (t >= tl).astype(float)
        dec = np.exp(-tau / BOB["decay"])
        z += (-O["sink"] * dec * np.sin(w * tau) - O["settle"] * (1.0 - np.exp(-tau / 0.2))) * on
        tip = math.radians(BOB["tip"]) * dec * np.sin(w * tau) * on            # from 0: nothing jumps on the landing
        a = 0.9 + 1.7 * k
        tip_x += tip * math.cos(a)
        tip_y += tip * math.sin(a)
        kick = O["whip"] * np.exp(-tau / WHIP["decay"]) * np.sin(2.0 * math.pi * WHIP["hz"] * tau) * on
        whip += kick
        yaw += 0.35 * kick
    spin = 2.0 * math.pi * O["turn"] * (t - (O["up"] or 0.0))      # the flag where it was made to stand when in place
    return {"z": z, "tip_x": tip_x, "tip_y": tip_y, "spin": spin, "whip": np.clip(whip, -1.5, 1.5),
            "yaw": yaw}


def top(O, t):
    """Where the landing spot's middle is (n, 3) in the prop's frame at clip seconds t: the ball's top riding its rise,
    bob and tip (what a figure standing on it keys its feet or root to)."""
    m = motion(O, t)
    h = Z_TOP * O["size"]
    p = np.zeros((len(m["z"]), 3))
    p[:, 0] = h * np.sin(m["tip_y"])
    p[:, 1] = -h * np.sin(m["tip_x"])
    p[:, 2] = m["z"] + h * np.cos(m["tip_x"]) * np.cos(m["tip_y"])
    return p


# ===================================================================================================================
# the card
# ===================================================================================================================
def objects(name):
    return {"lift": f"{name}_lift", "ball": f"{name}_ball", "craters": f"{name}_craters", "skewer": f"{name}_skewer",
            "pick": f"{name}_pick", "flag": f"{name}_flag", "top": f"{name}_top", "tip": f"{name}_flag_tip"}


def flag_tip():
    """The flag's free top corner in the flag's own frame (x out from the pick, y along it; unit diameter)."""
    return np.array([PICK["r"] * 0.9 + FLAG["w"], 0.0, 0.0])


def card(name, O, roles):
    s = O["size"]
    ob = objects(name)
    o, a, x = pick_frame()
    tip_world = o + a * flag_place() + x * flag_tip()[0]          # the flag's corner, the ball not turned
    end = o + a * (PICK["length"] - PICK["in"])
    lo = [-R * 1.02, -R * 1.02, -SKEWER["below"]]
    hi = [max(R * 1.02, float(tip_world[0])), R * 1.02, max(R, float(end[2]))]
    b = {"min": [round(v * s, 5) for v in lo], "max": [round(v * s, 5) for v in hi]}
    ztop = round(Z_TOP * s, 5)
    return {"size": [round(h - l, 5) for l, h in zip(b["min"], b["max"])], "origin": "center", "front": "-Y",
            "blocks": False, "bounds": b, "layers": [], "colliders": [], "slots": dict(roles),
            "use": {"anchor": [{"name": "top", "point": [0.0, 0.0, ztop], "dir": [0.0, 0.0, 1.0], "object": ob["top"]},
                               {"name": "flag", "point": [round(float(v) * s, 5) for v in tip_world],
                                "object": ob["tip"]}],
                    "look": [{"name": "planet", "point": [0.0, 0.0, 0.0]},
                             {"name": "top", "point": [0.0, 0.0, ztop], "object": ob["top"]},
                             {"name": "flag", "point": [round(float(v) * s, 5) for v in tip_world],
                              "object": ob["tip"]}],
                    "rest": [{"name": "top", "type": "plane", "center": [0.0, 0.0, ztop],
                              "radius": round(R * math.sin(THETA_CAP) * s, 5)}],
                    "feet": [{"name": "top", "L": [round(0.085 * s / 0.8, 5), 0.0, None],
                              "R": [round(-0.085 * s / 0.8, 5), 0.0, None], "floor_z": ztop}]},
            "planet": {"size": s, "top": [0.0, 0.0, ztop], "flat_radius": round(R * math.sin(THETA_CAP) * s, 5),
                       "up": O["up"], "rise": O["rise"], "land": O["land"], "sink": O["sink"], "settle": O["settle"],
                       "turn": O["turn"], "step": O["step"], "step_from": O["step_from"], "objects": ob}}
