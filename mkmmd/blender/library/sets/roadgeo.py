"""Pure-numpy road layout for the `highway` set: the cross-section (lanes, shoulders, median, paint), lamp layouts,
terrain profile, tunnel profiles and portal facades. Lateral offsets x are metres to the LEFT of the direction of
travel (like `core.path.Path.offset`), so the right-hand side of the road has negative x; heights are metres above the
road surface."""
import math

import numpy as np

# material kinds of the road surface ribbon
ASPHALT, SHOULDER, MEDIAN, WHITE, YELLOW = 0, 1, 2, 3, 4
KIND_MAT = {"lane": ASPHALT, "shoulder": SHOULDER, "median": MEDIAN, "white": WHITE, "yellow": YELLOW}


class Section:
    """A road cross-section: contiguous `bands` (dicts x0 < x1, kind, uc (uv origin), dash, under) from the right
    edge (x_min) to the left edge (x_max), the lane list for the card and the extents the highway builds on."""

    def __init__(self):
        self.bands, self.lanes = [], []
        self.x_min = self.x_max = 0.0
        self.carriageways = []            # {"dir", "x_in", "x_out", "lanes"}: x_in next to the median / centre line
        self.divided = False
        self.median = 0.0
        self.lane_width = 3.6

    @property
    def width(self):
        return self.x_max - self.x_min

    @property
    def half(self):
        return max(abs(self.x_min), abs(self.x_max))

    def dashed(self):
        """The distinct (on, period, phase) dash patterns in use."""
        return sorted({b["dash"] for b in self.bands if b["dash"]})


def _cut(structure, paint):
    """Cut paint bands out of structural ones and add them: a contiguous, sorted list of bands."""
    out = []
    for b in structure:
        pieces = [(b["x0"], b["x1"])]
        for p in paint:
            nxt = []
            for a, c in pieces:
                lo, hi = max(a, p["x0"]), min(c, p["x1"])
                if hi - lo <= 1e-9:
                    nxt.append((a, c))
                    continue
                if lo - a > 1e-9:
                    nxt.append((a, lo))
                if c - hi > 1e-9:
                    nxt.append((hi, c))
            pieces = nxt
        out += [dict(b, x0=a, x1=c) for a, c in pieces]
    for p in paint:
        mid = 0.5 * (p["x0"] + p["x1"])
        under = next((b for b in structure if b["x0"] - 1e-9 <= mid <= b["x1"] + 1e-9 and b["kind"] == "lane"),
                     next((b for b in structure if b["x0"] - 1e-9 <= mid <= b["x1"] + 1e-9), structure[0]))
        out.append(dict(p, under=under["kind"], uc=under["uc"]))
    return sorted(out, key=lambda b: b["x0"])


def cross_section(lanes=2, oncoming=None, lane_width=3.6, shoulder=2.5, inner_shoulder=0.9, median=3.0, divided=True,
                  drive="right", line_width=0.15, dash=(3.0, 12.0), dash_phase=0.0):
    """Lay out the road. `lanes` lanes run with the path direction (dir +1), `oncoming` against it (default the same
    number; 0 = one-way). Drive "right": the +1 lanes are on the right of the path (negative offsets); "left" mirrors
    it. Divided roads get a median strip (`median` m wide) with inner shoulders; undivided roads a double yellow line.
    Lane names are fwd1.. / opp1.., 1 = the lane next to the median / centre line."""
    oncoming = lanes if oncoming is None else int(oncoming)
    lanes = int(lanes)
    w, lw = float(lane_width), float(line_width)
    sec = Section()
    sec.divided = bool(divided and lanes > 0 and oncoming > 0)
    sec.lane_width = w
    xm = median / 2 if sec.divided else 0.0
    one_way = lanes == 0 or oncoming == 0
    si = inner_shoulder if (sec.divided or one_way) else 0.0
    sec.median = median if sec.divided else 0.0
    structure, paint, lane_list, cws = [], [], [], []
    dash_t = (float(dash[0]), float(dash[1]), float(dash_phase))

    def carriage(sign, n, name, d):
        """One carriageway on the side `sign` (-1 right of the centre, +1 left)."""
        if n <= 0:
            return
        x_in = sign * xm
        edge = sign * (xm + si)
        if si:
            structure.append({"x0": min(x_in, edge), "x1": max(x_in, edge), "kind": "shoulder", "uc": (x_in + edge) / 2,
                              "dash": None})
            paint.append({"x0": edge - lw / 2, "x1": edge + lw / 2, "kind": "yellow", "dash": None})   # inner edge
        for i in range(n):
            a, b = sign * (xm + si + i * w), sign * (xm + si + (i + 1) * w)
            structure.append({"x0": min(a, b), "x1": max(a, b), "kind": "lane", "uc": (a + b) / 2, "dash": None})
            lane_list.append({"name": f"{name}{i + 1}", "offset": (a + b) / 2, "dir": d})
            if i:
                paint.append({"x0": a - lw / 2, "x1": a + lw / 2, "kind": "white", "dash": dash_t})   # lane divider
        x_end = sign * (xm + si + n * w)
        x_out = x_end + sign * shoulder
        structure.append({"x0": min(x_end, x_out), "x1": max(x_end, x_out), "kind": "shoulder",
                          "uc": (x_end + x_out) / 2, "dash": None})
        paint.append({"x0": x_end - lw / 2, "x1": x_end + lw / 2, "kind": "white", "dash": None})     # outer edge
        cws.append({"dir": d, "x_in": x_in, "x_out": x_out, "lanes": n, "side": sign})

    carriage(-1, lanes, "fwd", +1)
    carriage(+1, oncoming, "opp", -1)
    if sec.divided:
        structure.append({"x0": -xm, "x1": xm, "kind": "median", "uc": 0.0, "dash": None})
    elif lanes > 0 and oncoming > 0:                                              # double yellow centre line
        paint.append({"x0": -lw * 1.5, "x1": -lw * 0.5, "kind": "yellow", "dash": None})
        paint.append({"x0": lw * 0.5, "x1": lw * 1.5, "kind": "yellow", "dash": None})
    structure.sort(key=lambda b: b["x0"])
    bands = _cut(structure, paint)
    x_min, x_max = bands[0]["x0"], bands[-1]["x1"]
    shift = 0.0 if (lanes > 0 and oncoming > 0) else -(x_min + x_max) / 2        # one-way: centre the roadway
    mirror = -1.0 if drive == "left" else 1.0
    for b in bands:
        x0, x1 = (b["x0"] + shift) * mirror, (b["x1"] + shift) * mirror
        b["x0"], b["x1"] = min(x0, x1), max(x0, x1)
        b["uc"] = (b["uc"] + shift) * mirror
        b["dash"] = tuple(b["dash"]) if b["dash"] else None
        b.setdefault("under", b["kind"])
    bands.sort(key=lambda b: b["x0"])
    for ln in lane_list:
        ln["offset"] = (ln["offset"] + shift) * mirror
    for c in cws:
        c["x_in"], c["x_out"] = (c["x_in"] + shift) * mirror, (c["x_out"] + shift) * mirror
        c["side"] = c["side"] * mirror
    sec.bands, sec.lanes, sec.carriageways = bands, lane_list, cws
    sec.x_min, sec.x_max = min((x_min + shift) * mirror, (x_max + shift) * mirror), max(
        (x_min + shift) * mirror, (x_max + shift) * mirror)
    return sec


def band_material(b, s_mid):
    """Material index per quad of band `b` for the quad mid arc lengths `s_mid` (dashes alternate paint / ground)."""
    paint, under = KIND_MAT[b["kind"]], KIND_MAT[b["under"]]
    if b["dash"] is None:
        return np.full(len(s_mid), paint)
    on, period, phase = b["dash"]
    return np.where(np.mod(np.asarray(s_mid, float) - phase, period) < on, paint, under)


# ---------------------------------------------------------------------------------------------------- lamps


def lamp_layout(sec, length, layout="auto", spacing=40.0, first=None, skip=(), margin=12.0, verge=1.0):
    """Where the street lamps stand (arc length `s`, lateral `x` of each pole) and which way each arm points.
    layout: "median" (one pole on the median with two arms; divided roads), "outer" (a pole behind each edge),
    "left" / "right" (that side only), "stagger" (alternate sides), "auto" (median when divided, else outer).
    Poles within `margin` of a `skip` range (tunnels) are dropped. Returns dict: pole_s, pole_x (per pole),
    arm_pole (pole index per arm), sign (+1 the arm points left (+x), -1 right) per arm, and per-arm s, x."""
    if layout == "auto":
        layout = "median" if sec.divided else "outer"
    if layout == "median" and not sec.divided:
        layout = "outer"
    if layout not in ("median", "outer", "left", "right", "stagger"):
        raise ValueError(f"lamps layout {layout!r} (median, outer, left, right, stagger, auto)")
    first = spacing * 0.5 if first is None else float(first)
    xr, xl = sec.x_min - verge + 0.2, sec.x_max + verge - 0.2          # behind the right / left edge
    poles = []                                                         # (s, x, arm signs)
    for k, s in enumerate(np.arange(first, length - 1e-6, spacing)):
        if any(a - margin <= s <= b + margin for a, b in skip):
            continue
        if layout == "median":
            poles.append((s, 0.0, (+1.0, -1.0)))
        elif layout == "outer":
            poles += [(s, xr, (+1.0,)), (s, xl, (-1.0,))]
        elif layout == "right":
            poles.append((s, xr, (+1.0,)))
        elif layout == "left":
            poles.append((s, xl, (-1.0,)))
        else:
            poles.append((s, xr, (+1.0,)) if k % 2 == 0 else (s, xl, (-1.0,)))
    pole_s = np.array([p[0] for p in poles], float)
    pole_x = np.array([p[1] for p in poles], float)
    arm_pole = np.array([i for i, p in enumerate(poles) for _ in p[2]], int)
    sign = np.array([sg for p in poles for sg in p[2]], float)
    return {"layout": layout, "pole_s": pole_s, "pole_x": pole_x, "arm_pole": arm_pole, "sign": sign,
            "s": pole_s[arm_pole] if len(arm_pole) else np.zeros(0),
            "x": pole_x[arm_pole] if len(arm_pole) else np.zeros(0)}


def real_light_mask(s, every=4, window=None, max_lights=24):
    """Which lamps (arrays ordered along the road) get a real light: every `every`-th lamp (0 = none), only within
    the arc-length `window` (a, b) when given, at most `max_lights`. Boolean array."""
    s = np.asarray(s, float)
    mask = np.zeros(len(s), bool)
    if every <= 0 or not len(s):
        return mask
    mask[np.arange(len(s)) % every == 0] = True
    if window is not None:
        mask &= (s >= window[0]) & (s <= window[1])
    if mask.sum() > max_lights:
        keep = np.flatnonzero(mask)[:max_lights]
        mask[:] = False
        mask[keep] = True
    return mask


# ---------------------------------------------------------------------------------------------------- terrain


def edge_offsets(drop=0.6, verge=1.0, run=3.0, far=(2.0, 5.0, 10.0, 20.0, 40.0, 75.0)):
    """Distances d (m outward from the paved edge) of the lines of the roadside ribbon: the verge, the embankment foot
    and then `far` further out."""
    base = verge + run * drop
    return np.array([0.0, verge, base] + [base + f for f in far])


def ground_dz(d, drop=0.6, verge=1.0, run=3.0):
    """Height (relative to the road) at distance d outward from the paved edge: level verge, a slope down at 1:run,
    then the ground at -drop."""
    d = np.asarray(d, float)
    return np.where(d <= verge, 0.0, np.where(d >= verge + run * drop, -drop, -(d - verge) / run))


# ---------------------------------------------------------------------------------------------------- tunnels


def tube_profile(x_left, x_right, wall_h=3.9, crown_h=6.4, n=14, rungs=(1.2, 2.4, 3.0)):
    """Inner outline of one tunnel tube from the foot of the left wall up it, over the arch and down the right wall:
    (m, 2) points (lateral, height), x_left > x_right (left is positive). `rungs` are extra heights on the walls so
    baked light has vertices to fall off between."""
    xc, hw = 0.5 * (x_left + x_right), 0.5 * (x_left - x_right)
    t = np.linspace(0.0, math.pi, n + 1)
    arch = np.stack([xc + hw * np.cos(t), wall_h + (crown_h - wall_h) * np.sin(t)], 1)
    up = [z for z in rungs if 0.0 < z < wall_h]
    left = np.array([[x_left, 0.0]] + [[x_left, z] for z in up])
    right = np.array([[x_right, z] for z in up[::-1]] + [[x_right, 0.0]])
    return np.vstack([left, arch, right])


def mound_profile(x_left, x_right, crown, ground_z, spread=24.0, cap=3.5, n=9):
    """Outline of the hill over a tunnel from its foot on the left over the crest to the right foot: (2n + 2, 2), a
    rounded shoulder on each side (a quarter of a superellipse)."""
    top = crown + cap
    t = np.linspace(0.0, 1.0, n)
    run = (1.0 - t) ** 2.2                                     # 1 at the foot -> 0 at the crest edge
    xs = np.concatenate([x_left + spread * run, x_right - spread * run[::-1]])
    ys = np.concatenate([ground_z + (top - ground_z) * (1.0 - run), ground_z + (top - ground_z) * (1.0 - run[::-1])])
    return np.stack([xs, ys], 1)


def facade(openings, outline, ground_z):
    """Quads (m, 4, 2) of a portal wall in the (lateral x, height z) plane: the region under `outline` (a mound_profile,
    ground at ground_z) minus the tube `openings` (tube_profile outlines). Corners run left-bottom, right-bottom,
    right-top, left-top as seen by traffic approaching the wall, so the normal points back along -T."""
    outline = np.asarray(outline, float)
    ox, oz = outline[::-1, 0], outline[::-1, 1]                    # increasing x for np.interp
    xs = set(np.round(ox, 6))
    holes = []
    for o in openings:
        o = np.asarray(o, float)
        xs |= set(np.round(o[:, 0], 6))
        arch = o[1:-1]
        order = np.argsort(arch[:, 0])
        holes.append((o[:, 0].min(), o[:, 0].max(), arch[order, 0], arch[order, 1]))
    xs = np.array(sorted(xs))
    quads = []
    for xa, xb in zip(xs[:-1], xs[1:]):
        if xb - xa < 1e-6:
            continue
        mid = 0.5 * (xa + xb)
        top_a, top_b = np.interp([xa, xb], ox, oz)
        bot_a = bot_b = ground_z
        for lo, hi, ax, az in holes:
            if lo - 1e-6 <= mid <= hi + 1e-6:
                bot_a, bot_b = np.interp([xa, xb], ax, az)
        if top_a - bot_a <= 1e-6 and top_b - bot_b <= 1e-6:
            continue
        quads.append([[xb, bot_b], [xa, bot_a], [xa, top_a], [xb, top_b]])
    return np.array(quads)
