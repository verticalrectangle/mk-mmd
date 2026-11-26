"""The paper theatre café's furniture (`card = "library:paper_cafe_chair"` and `"library:paper_cafe_table"`; docs/design.md:
Props), numpy only: the options, the pieces (card and paper, as the theatre's), the cup's rattle and the cards. The
Blender side is mkmmd/blender/library/props/paper_cafe.py; a flat piece is a paper_theatre Plate (scissor-cut, rolled,
meshed there by paper_theatre_parts.plate_mesh), a bent strip or a turned piece a shell Mesh made here.

Sized round a figure 1.5 m tall (the theatre's), the chair's frame as `library:cafe_chair`'s: its origin on the floor
under the seat's middle, the seated figure facing -Y, +X her left.

  * the chair: a bentwood café chair of card. A round seat (a disc of card, a pale cane seat with its weave pencilled
    on, a bent band of card round its edge), four legs of bent card strips splaying a little to the floor, a ring of
    card joining them, and the back: two strips rising from the seat's back and arching over (leaning back a little)
    round a smaller loop, brass split pins where the legs meet the seat;
  * the table: a small round café table: a pale top with a gilt band round its edge on a pedestal of two card
    silhouettes slotted across each other (curled feet, a column, a flare under the top), and on it a cup on its
    saucer, coffee in it with a paper lily pad floating on the coffee; `rattle` (clip seconds) makes the cup and the
    saucer jump and chatter, settling in half a second (held on the `step` grid as the theatre's motion is); the
    place where the scissors lie (`use.anchor` `scissors`) is on the top on her side."""
import math

import numpy as np

from . import paper_theatre as PT
from . import shell as S

# --- the chair
SEAT_Z = 0.44                  # the seat's top
SEAT_R = 0.19                  # its radius
SEAT_T = 0.012                 # its card
BAND = (0.035, 0.006)          # the band round the seat's edge: its height, its card
STRIP = (0.032, 0.007)         # a leg's strip of card: its width, its thickness
LEGS = {"front_L": ((0.13, -0.12), (0.162, -0.155)), "front_R": ((-0.13, -0.12), (-0.162, -0.155)),
        "back_L": ((0.135, 0.12), (0.158, 0.172)), "back_R": ((-0.135, 0.12), (-0.158, 0.172))}   # under the seat, foot
RING_Z = 0.2                   # the ring joining the legs
BACK = (0.135, 0.13, 0.86, 9.0)  # the back's half width at the seat, its foot's y, its top, how far it leans back (deg)
LOOP = (0.075, 0.56, 0.8)      # the inner loop's half width, its foot and its top
CHAIR_ROLES = {"wood": "gold:text:0.42", "cane": "gold:surface:0.55", "cane_line": "gold:text:0.6", "brad": "gold"}

# --- the table
TOP_Z = 0.62                   # the top's face
TOP_R = 0.24                   # its radius
TOP_T = 0.014                  # its card
RIM = (0.024, 0.006)           # the gilt band round its edge
FOOT = 0.2                     # the pedestal's half spread on the floor
COLUMN = 0.026                 # its column's half width
CUP_AT = (-0.07, -0.07)        # the cup on its saucer (on the top, away from her and toward the front)
SCISSORS_AT = (0.1, 0.04)      # where the scissors lie: on her side of the top
CUP = (0.032, 0.046, 0.066, 0.0025)   # the cup's foot and rim radii, its height, its card
SAUCER = (0.078, 0.013)        # the saucer's radius and height
COFFEE = 0.052                 # the coffee's height in the cup
PAD_R = 0.022                  # the paper lily pad floating on it
RATTLE = (0.5, 0.012, 0.006, 5.0, 7.0)   # s a rattle takes to settle; the cup's hop and the saucer's (m); the cup's tilt
#                                          (deg), how fast it chatters (Hz)
TABLE_ROLES = {"top": "overlay:surface:0.4", "rim": "gold", "base": "pine:text:0.55", "cup": "surface",
               "cup_band": "love", "saucer": "surface", "coffee": "gold:text:0.62", "pad": "pine:foam:0.45"}
TABLE = {"rattle": (), "step": 0.0, "step_from": 0.0, "seed": 1}

CHAIR_MATS = tuple(CHAIR_ROLES)
TABLE_MATS = tuple(TABLE_ROLES)


class CafeError(ValueError):
    """A paper_cafe_* prop's slots the builder cannot use."""


def table_options(slots):
    """The table's own slots (TABLE: rattle, step, step_from, seed) checked and filled; colour slots are left to the
    builder."""
    o = dict(TABLE)
    r = slots.get("rattle", ())
    if isinstance(r, (int, float)) and not isinstance(r, bool):
        r = [r]
    if not isinstance(r, (list, tuple)):
        raise CafeError(f"paper_cafe_table: rattle = {r!r}: a clip second or a list of them")
    o["rattle"] = tuple(PT._number(v, "paper_cafe_table: rattle") for v in r)
    for k in ("step", "step_from"):
        if k in slots:
            o[k] = PT._number(slots[k], f"paper_cafe_table: {k}", 0.0 if k == "step" else None,
                              240.0 if k == "step" else None)
    if "seed" in slots:
        o["seed"] = int(slots["seed"])
    return o


# ===================================================================================================================
# the chair
# ===================================================================================================================
def _strip(path, up, mat, width=STRIP[0], thick=STRIP[1], closed=False):
    """A strip of card bent along `path`, its broad face facing `up` (made square to the path), rounded edges."""
    return S.sweep(np.asarray(path, float), S.rrect(thick, width, 0.4 * thick, 2), up=up, closed=closed, mat=mat)


def _curve(points, n=24):
    """A smooth curve through the points (Catmull-Rom), n points a span."""
    P = np.asarray(points, float)
    P = np.vstack([2 * P[0] - P[1], P, 2 * P[-1] - P[-2]])
    out = []
    u = np.linspace(0.0, 1.0, n, endpoint=False)[:, None]
    for i in range(1, len(P) - 2):
        p0, p1, p2, p3 = P[i - 1], P[i], P[i + 1], P[i + 2]
        out.append(0.5 * (2 * p1 + (-p0 + p2) * u + (2 * p0 - 5 * p1 + 4 * p2 - p3) * u ** 2
                          + (-p0 + 3 * p1 - 3 * p2 + p3) * u ** 3))
    out.append(P[-2][None])
    return np.concatenate(out)


def back_path():
    """The back's strip: up from the seat's back on her right, over the top, down on her left (leaning back BACK[3])."""
    hw, y0, top, lean = BACK
    t = math.tan(math.radians(lean))
    z0 = SEAT_Z - 0.01

    def at(x, z):
        return (x, y0 + (z - z0) * t, z)
    pts = [at(-hw, z0), at(-hw - 0.006, 0.6), at(-hw + 0.004, 0.75), at(-0.1, top - 0.035), at(0.0, top),
           at(0.1, top - 0.035), at(hw - 0.004, 0.75), at(hw + 0.006, 0.6), at(hw, z0)]
    return _curve(pts, 14)


def loop_path():
    """The inner loop of the back, closed."""
    hw, z0, z1 = LOOP
    _hw, y0, _top, lean = BACK
    t = math.tan(math.radians(lean))
    zc, rz = 0.5 * (z0 + z1), 0.5 * (z1 - z0)
    a = np.linspace(0.0, 2.0 * math.pi, 48, endpoint=False)
    z = zc + rz * np.sin(a)
    return np.stack([hw * np.cos(a), y0 + (z - (SEAT_Z - 0.01)) * t + 0.003, z], 1)


def chair_pieces():
    """The chair's pieces: {object part: [Mesh or Plate]} in the chair's frame: `body` (the legs, the ring, the seat's
    band, the back and its loop), `seat` (the cane seat and its pencilled weave), and the split pins (`brads`: (x, z,
    y_head, y_back) on a plate facing -y, turned to face out)."""
    wood, cane, line = (CHAIR_MATS.index(r) for r in ("wood", "cane", "cane_line"))
    body = []
    for name, ((x0, y0), (x1, y1)) in LEGS.items():
        z_top = SEAT_Z - 0.012
        path = _curve([(x0, y0, z_top), (0.5 * (x0 + x1) + 0.004 * np.sign(x0), 0.5 * (y0 + y1), 0.22),
                       (x1, y1, 0.008)], 16)
        out = np.array([x0, y0, 0.0]) / math.hypot(x0, y0)
        body.append(_strip(path, out, wood))
    # the ring joining the legs, on edge
    a = np.linspace(0.0, 2.0 * math.pi, 64, endpoint=False)
    cy, rx, ry = 0.012, 0.165, 0.152
    ring = np.stack([rx * np.cos(a), cy + ry * np.sin(a), np.full(64, RING_Z)], 1)
    body.append(S.sweep(ring, S.rrect(STRIP[1], 0.02, 0.003, 2), up=(0.0, 0.0, 1.0), closed=True, mat=wood).weld())
    # the band round the seat's edge
    ring = np.stack([(SEAT_R + 0.003) * np.cos(a), (SEAT_R + 0.003) * np.sin(a), np.full(64, SEAT_Z - 0.022)], 1)
    body.append(S.sweep(ring, S.rrect(BAND[1], BAND[0], 0.0025, 2), up=(0.0, 0.0, 1.0), closed=True,
                        mat=wood).weld())
    # the back: its strip and its loop, their broad faces to the front
    lean = math.radians(BACK[3])
    face = (0.0, -math.cos(lean), math.sin(lean))
    body.append(_strip(back_path(), face, wood, width=0.03))
    body.append(_strip(loop_path(), face, wood, width=0.022, closed=True).weld())
    # the seat: a disc of card, its cane top, the weave pencilled across it
    disc = S.rounded_panel(S.ellipse(SEAT_R, SEAT_R, 72), SEAT_T, r_edge=0.004, dome=0.004, mat=cane)
    disc = disc.moved((0.0, 0.0, SEAT_Z - SEAT_T))
    seat = [disc]
    for turn in (45.0, -45.0):
        b = math.radians(turn)
        d, n = np.array([math.cos(b), math.sin(b)]), np.array([-math.sin(b), math.cos(b)])
        for k in np.arange(-3.5, 4.0, 1.0):
            off = k * 0.042
            half = math.sqrt(max((SEAT_R - 0.025) ** 2 - off * off, 0.0))
            if half < 0.03:
                continue
            p0, p1 = off * n - half * d, off * n + half * d
            s = np.linspace(0.0, 1.0, 16)[:, None]
            pts = p0 + s * (p1 - p0)
            r2 = (pts ** 2).sum(1)
            dome = 0.004 * (1.0 - r2 / SEAT_R ** 2)
            path = np.stack([pts[:, 0], pts[:, 1], np.full(len(pts), SEAT_Z) + dome + 0.0004], 1)
            seat.append(S.ribbon(path, 0.0035, (0.0, 0.0, 1.0), thick=0.0004, mat=line))
    brads = []
    for name, ((x0, y0), _f) in LEGS.items():
        r = math.hypot(x0, y0)
        brads.append((x0 / r, y0 / r))
    return {"body": body, "seat": seat, "brads": brads}


def chair_card(name, roles, colliders):
    """The chair's card: `library:cafe_chair`'s shape (use.sit `seat`, use.feet `seat`, use.rest `seat_front`)."""
    use = {"sit": [{"name": "seat", "hip": [0.0, 0.035, round(SEAT_Z + 0.075, 4)], "facing": [0, -1, 0],
                    "seat_z": SEAT_Z, "floor_z": 0.0, "pelvis_deg": 6.0, "back_deg": 0.0,
                    "back_tilt_deg": BACK[3]}],
           "feet": [{"name": "seat", "L": [0.11, -0.33, None], "R": [-0.11, -0.33, None], "floor_z": 0.0}],
           "rest": [{"name": "seat_front", "type": "edge", "a": [-0.15, -0.14, SEAT_Z], "b": [0.15, -0.14, SEAT_Z],
                     "normal": [0, 0, 1]}],
           "look": [{"name": "back_top", "point": [0.0, round(BACK[1] + (BACK[2] - SEAT_Z) *
                                                         math.tan(math.radians(BACK[3])), 4), BACK[2]]}]}
    return {"size": [0.36, 0.36, BACK[2]], "origin": "floor_center", "front": "-Y", "slots": dict(roles),
            "use": use, "colliders": colliders}


# ===================================================================================================================
# the table
# ===================================================================================================================
def pedestal_outline():
    """A pedestal silhouette (x, z): curled feet on the floor, the column, a flare under the top."""
    zt = TOP_Z - TOP_T
    right = [(FOOT, 0.0), (FOOT + 0.012, 0.018), (FOOT - 0.004, 0.036), (FOOT - 0.04, 0.03), (0.09, 0.05),
             (0.045, 0.1), (COLUMN, 0.16), (COLUMN, zt - 0.14), (0.04, zt - 0.07), (0.075, zt - 0.02), (0.08, zt)]
    pts = _curve(np.array([[x, z] for x, z in right]), 6)
    P = np.vstack([pts, (pts * (-1.0, 1.0))[::-1]])
    return PT.ccw(np.vstack([P, [[-FOOT * 0.55, 0.0], [FOOT * 0.55, 0.0]]]))


def table_pieces():
    """The table's pieces in its frame: {"table": [Mesh or Plate]} (the top, its band, the pedestal's two silhouettes,
    the bearer) and {"saucer": [Mesh], "cup": [Mesh]} in their own frames (each on its foot's middle)."""
    m = {r: TABLE_MATS.index(r) for r in TABLE_MATS}
    top = S.rounded_panel(S.ellipse(TOP_R, TOP_R, 96), TOP_T, r_edge=0.004, mat=m["top"])
    top = top.apply(None, (0.0, 0.0, TOP_Z - TOP_T))
    a = np.linspace(0.0, 2.0 * math.pi, 96, endpoint=False)
    band = np.stack([(TOP_R + 0.002) * np.cos(a), (TOP_R + 0.002) * np.sin(a), np.full(96, TOP_Z - 0.5 * TOP_T)], 1)
    rim = S.sweep(band, S.rrect(RIM[1], RIM[0], 0.0025, 2), up=(0.0, 0.0, 1.0), closed=True, mat=m["rim"]).weld()
    P = pedestal_outline()
    flat = []
    for turn in (0.0, 90.0):
        b = math.radians(turn)
        R = ((math.cos(b), -math.sin(b), 0.0), (math.sin(b), math.cos(b), 0.0), (0.0, 0.0, 1.0))
        flat.append(PT._plate("base", P, -0.006, 0.012, PT.SNIP, frame=(R, (0.0, 0.0, 0.0))))
    bearer = S.rounded_panel(S.ellipse(0.085, 0.085, 48), 0.01, r_edge=0.003, mat=m["base"])
    bearer = bearer.apply(None, (0.0, 0.0, TOP_Z - TOP_T - 0.01))
    # the saucer and the cup, turned
    sr, sh = SAUCER
    saucer = S.lathe([(0.0, 0.0), (0.045, 0.0), (0.05, 0.003), (sr - 0.006, sh - 0.004), (sr, sh - 0.001),
                      (sr - 0.002, sh), (sr - 0.012, sh - 0.003), (0.04, 0.006), (0.0, 0.006)], seg=48, mat=m["saucer"])
    r0, r1, h, t = CUP
    z = np.linspace(0.004, h, 10)
    rr = r0 + (r1 - r0) * ((z - 0.004) / (h - 0.004)) ** 0.8
    outer = list(zip(rr, z))
    inner = list(zip(rr[::-1] - t, z[::-1] - 0.5 * t))
    cup = S.lathe([(0.0, 0.0), (r0 * 0.85, 0.0), (r0, 0.004)] + outer[1:] + [(r1 + 0.0008, h + 0.001)] + inner +
                  [(0.0, 0.004 + t)], seg=48, mat=m["cup"])
    zb = 0.7 * h
    ring = np.stack([(r0 + (r1 - r0) * ((zb - 0.004) / (h - 0.004)) ** 0.8 + 0.0008) * np.cos(a[::2]),
                     (r0 + (r1 - r0) * ((zb - 0.004) / (h - 0.004)) ** 0.8 + 0.0008) * np.sin(a[::2]),
                     np.full(48, zb)], 1)
    stripe = S.sweep(ring, S.rrect(0.0012, 0.009, 0.0005, 1), up=(0.0, 0.0, 1.0), closed=True,
                     mat=m["cup_band"]).weld()
    rc = r0 + (r1 - r0) * ((COFFEE - 0.004) / (h - 0.004)) ** 0.8 - t - 0.0005
    coffee = S.lathe([(0.0, COFFEE - 0.004), (rc, COFFEE - 0.004), (rc, COFFEE), (0.0, COFFEE)], seg=48,
                     mat=m["coffee"])
    q = np.linspace(0.25, 2.0 * math.pi - 0.25, 40)
    pad_outline = np.vstack([[[0.15 * PAD_R, 0.0]], np.stack([PAD_R * np.cos(q), PAD_R * np.sin(q)], 1)])
    pad = S.rounded_panel(pad_outline, 0.0012, r_edge=0.0005, mat=m["pad"])
    pad = pad.apply(S.rot_matrix(rz=35.0), (0.004, -0.003, COFFEE + 0.0002))
    hc = np.array([r1 + 0.002, 0.0, 0.5 * h])
    t_ = np.linspace(-0.5 * math.pi, 0.5 * math.pi, 20)
    handle_path = hc + np.stack([0.016 * np.cos(t_), np.zeros(20), 0.017 * np.sin(t_)], 1) - (0.004, 0.0, 0.0)
    handle = S.sweep(handle_path, S.rrect(0.004, 0.007, 0.0015, 2), up=(0.0, 1.0, 0.0), mat=m["cup"])
    cup_m = [x if x.volume() > 0 else x.flip() for x in (cup, stripe, coffee, pad, handle)]
    return {"table": [top, rim, bearer] + flat, "saucer": [saucer if saucer.volume() > 0 else saucer.flip()],
            "cup": cup_m}


def rattle(opts, t):
    """The cup's and the saucer's jump on every `rattle` time: (cup z, cup tilt x, cup tilt y, saucer z) at the clip
    seconds t: each a quick hop off the mark and a chatter dying away in RATTLE[0] s."""
    t = np.asarray(t, float)
    secs, cup, saucer, tilt, hz = RATTLE
    cz, tx, ty, sz = (np.zeros(t.shape) for _ in range(4))
    rng = np.random.default_rng(int(opts["seed"]) + 77)
    for k, t0 in enumerate(opts["rattle"]):
        u = t - t0
        on = u >= 0.0
        env = np.where(on, np.exp(-np.maximum(u, 0.0) * math.log(30.0) / secs), 0.0)
        hop = np.abs(np.sin(2.0 * math.pi * hz * np.maximum(u, 0.0)))
        cz += cup * env * hop
        sz += saucer * env * np.abs(np.sin(2.0 * math.pi * hz * 1.3 * np.maximum(u, 0.0) + 0.6))
        a, b = rng.uniform(0, 2 * math.pi, 2)
        tx += math.radians(tilt) * env * np.sin(2.0 * math.pi * hz * 0.8 * u + a) * on
        ty += math.radians(tilt) * env * np.sin(2.0 * math.pi * hz * 0.9 * u + b) * on
    return cz, tx, ty, sz


def table_card(name, opts, roles, cup_object, colliders):
    """The table's card: use.anchor `scissors` (where the scissors' pivot lies: their blades away from her), use.look
    `cup` (riding the cup), use.rest `top`."""
    sx, sy = SCISSORS_AT
    d = np.array([-1.0, -0.35, 0.0])
    d /= np.linalg.norm(d)
    use = {"anchor": [{"name": "scissors", "point": [sx, sy, round(TOP_Z + 0.0024, 4)],   # the pin: the cards on the top
                       "dir": [round(float(v), 4) for v in d], "normal": [0.0, 0.0, 1.0]}],
           "look": [{"name": "cup", "point": [CUP_AT[0], CUP_AT[1], round(TOP_Z + SAUCER[1] + COFFEE, 4)],
                     "object": cup_object}],
           "rest": [{"name": "top", "type": "plane", "center": [0.0, 0.0, TOP_Z], "radius": TOP_R}]}
    return {"size": [2 * TOP_R, 2 * TOP_R, TOP_Z], "origin": "floor_center", "front": "-Y", "slots": dict(roles),
            "use": use, "colliders": colliders, "rattle": list(opts["rattle"])}
