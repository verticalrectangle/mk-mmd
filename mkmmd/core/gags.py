"""Gag moves, bpy-free (docs/design.md: Moves): the cartoon routines of a kids' show in the moves library. Its hand places,
finger shapes, poles and aims are merged into mkmmd.core.moves' own tables, its moves into `MOVES`, and `compile` plays
them as it plays the others (their hands travel, rest, stay natural and keep off the body the same way):
- `yawn`: a big stretch, both fists up high and then the arms opening out wide, the back arched and the head tilted, the
  mouth opening slowly into a yawn and closing (`a`, `o`; the eyes shut, `calm`). With `prop` (a pillow: `use.anchor`
  `hug`, the middle of its back, and `use.rest` `hold_L` / `hold_R`, its two ends) she hugs it to her chest instead, a
  palm on each end, leaning onto it with her head tilted (`pillow_places`);
- `chomp`: a held prop (a giant carrot: `use.grip` `hold`, `use.bite` `bite1`, `bite2` ...) brought to the mouth and bitten
  on each of the move's hits (or beats): bite k's point at the lips on the k-th hit of every chomp with that prop, the mouth
  opening wide as it comes and snapping shut on the hit, chewing after; the prop's root property `bite<k>` is set to that
  clip second, so its bite mark carves in and its chunks fly then (`chomp_places`);
- `brush`: a held toothbrush (`use.grip` `hold`, `use.anchor` `brush`: the middle of the bristles' tips and the way they
  face) with its bristles on the teeth, scrubbing side to side on the beats; the teeth bared (`i`);
- `run_in_place`: a cartoon run, the knees coming up in turn on each beat and halfway between (perform's `walk` on the
  spot: stride 0, each foot coming down where it rose), the fists pumping opposite them and the body leaning in;
- `wag`: a scolding index finger raised by the face, wagging out on the beats and in between, the other hand on the hip;
- `wink_point`: a wink and a point at the lens, leaning in;
- `bounce`: an EDM bounce: the knees pump on the hits (perform's `bounce` on them), the fists in front of the chest dip
  with them, and on each accent (default: the downbeats) both fists punch up and she takes a little jump (`rise`, the legs
  tucking: `air`, the knees giving on the landing: `crouch`);
- `chores`: simple mimed loops, `chore = "sweep"` (two fists up a broom handle, sweeping to the low hand's side on the
  beats), `"stir"` (a fist circling a turn a beat over the bowl of the other, cupped hand) or `"water"` (a fist round a
  watering can's handle held out low, tipped forward on the beats).

A held prop rides the fist of the hand that holds it (`held`: the build bone-parents it to the wrist) with the grip's
middle in the fist's hole (`fist`: the member's own fingers closed round a handle of the grip's radius) and the grip's
`axis` (toward the business end: a carrot's tip, a toothbrush's head, a can's spout) out of the fist's thumb side, as a
fist holds a mic. The fist closes on the span the card's `use.grip` `hold` gives (`center`, `axis`, `length`, `radius`,
optionally `radius_tip` and `profile` [[s, r], ...] from the business end): where along it depends on the move (a carrot
as far from its tip as the arm still brings the first bite to the mouth, `GRIP_REACH`; a toothbrush by its neck). One
grip per prop: the first move that holds it sets how it sits in the fist (its front to the lens, the bristles to the
teeth), and every place of it is solved with that grip (`fit`): the wrist moved until the prop's part is where the place
wants it, the hand continuing its forearm as far as the fist round the handle allows (mkmmd.core.moves.aimed) and the
elbow turned to keep the arm's twists in range (mkmmd.core.moves.arm_for). A hugged prop (the pillow) rides the chest
instead, its back against the body (kept `HUG_GAP` off it).

Everything is in the character's frame (x its left, y behind it, z up, metres from its root), the places written for
the left hand and mirrored for the right, like mkmmd.core.moves' own. `plan` makes one gag entry's keys: the hands'
places on the clock, the places that depend on a prop (resolved here), the perform keys of the legs (`bounce`, `rise`,
`crouch`, `air`, `walk`), timed faces (`curves`), the spans where a move has the mouth (`mouth`: lip sync gives way
there), the props to attach and the props' root properties to set."""
import math

import numpy as np

from . import fingers as FG


class GagError(ValueError):
    """A gag move that cannot be played as asked (mkmmd.core.moves.compile reports it as a MoveError)."""


def _mv():
    """mkmmd.core.moves, which imports this module for its tables (so it is imported when it is needed, not before)."""
    from . import moves
    return moves


def _v(*a):
    return np.array(a, float)


def _unit(v):
    v = np.asarray(v, float)
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-12 else v


# ---------------------------------------------------------------------------------------------------------- the tables
OUT = {  # place: (direction from the arm joint for the left hand, x mirrored for the right; share of the reach)
    "stretch": ((0.28, 0.12, 1.0), 0.97),         # the yawn's stretch: the fists up high, a little out and back
    "stretch_out": ((1.0, 0.12, 0.38), 0.96),     # ... then the arms open out wide, a little up and back
    "pump_up": ((0.16, -0.22, 1.0), 0.92),        # a fist punched up on an accent
    "water": ((0.45, -0.6, -0.7), 0.92),          # a watering can held out in front at the side, at the waist
    "water_tilt": ((0.45, -0.6, -0.7), 0.92),
}
AIM = {  # a mimed fist round a handle: where the handle points out of its thumb side, for the left hand (as
    # mkmmd.core.moves.AIM): the ways a fist holds one there with the hand continuing its forearm and the arm's twists in
    # range (a fist below the shoulder holding a handle that points down has its thumb up the handle)
    "water": (0.1, -0.2, 0.97),                   # a watering can's handle, held out low ...
    "water_tilt": (0.1, -0.75, 0.65),             # ... the fist tipped forward 40 degrees: the spout dips
    "stir": (0.0, 0.17, 0.98),                    # a spoon, its bowl down in the other hand's bowl
    "sweep_hi": (0.3, 0.17, 0.94),                # a broom handle, both thumbs up it: down to its head at the low hand's
    "sweep_lo": (-0.3, 0.17, 0.94),               # side (the high hand's place mirrors with the hand that holds high)
}
POLES = {  # places that hold the elbow their own way: its pole from the arm joint, (out, behind, below) (moves.POLES)
    "pump_back": (0.25, 0.5, 0.15),               # a fist swung back by the hip: the elbow back
    "club": (0.4, 0.05, 0.35), "pump_fwd": (0.35, 0.05, 0.35),   # fists before the chest: the elbows down at the sides
    "wag_out": (0.15, -0.3, 0.45), "wag_in": (0.15, -0.3, 0.45),   # a finger raised by the face: the elbow under it,
}                                                                  # so the forearm stands upright
SHAPES = {                                         # finger shapes (mkmmd.core.fingers tables) of the gag places
    "pillow": {"index": [10, 12, 8], "middle": [10, 12, 8], "ring": [12, 14, 8], "little": [14, 16, 10],
               "thumb": [10, 10, 5], "spread": 10},       # a palm spread on a soft pillow, the fingers curving round it
    "bowl": {"index": [30, 30, 20], "middle": [30, 30, 20], "ring": [32, 32, 20], "little": [34, 34, 22],
             "thumb": [20, 15, 10], "spread": -4},        # cupped under a bowl
}
BOX = {"pillow": (1.0, 0.012), "bowl": (0.85, 0.02)}   # as mkmmd.core.moves.BOX
STRAIGHT = {}
FIST = (35.0, 50.0, 35.0)                          # the curl (deg per joint) a held fist starts from (`curled`), ...
FIST_SCALE = (0.25, 2.2)                           # ... scaled within this range to close round the handle (`fist`)
FIST_THUMB = (20.0, 25.0, 20.0)                    # the thumb's curl round a handle
GRIP_REACH = 0.88                                  # a held place keeps its wrist this share of `reach` from the shoulder
GRIP_ROUNDS = 6                                    # the halvings that look for the furthest grip still in that reach
BITE_ROOM = 0.07                                   # m: the fist stays this far from the last bite's point (the teeth)
LIP_INSET = 0.008                                  # m: a bite's point this far behind the lips (between the teeth) ...
BRUSH_INSET = 0.002                                # ... and the bristles' tips this far (on the teeth)
LIP_MARGIN = 0.003                                 # m: the lips are where the face first comes this close from the front
CHOMP_DIR = (0.5, -0.55, -0.55)                    # a chomp's prop from the mouth toward the fist (left hand; x mirrored)
BRUSH_DIR = (0.7, -0.55, -0.35)                    # a toothbrush from the teeth toward the fist (across the mouth) ...
BRUSH_GRIP = 0.72                                  # ... held this share of `reach` from the bristles (down its handle)
READY_TILT = 20.0                                  # degrees a chomp's prop tips down about the fist before a bite, after
HUG_GAP = 0.012                                    # m: a hugged pillow's back this far off the body
HUG_DROP = 0.06                                    # m: its top this far below the mouth (under the chin)
PALM = 0.012                                       # m: the palm's skin from the wrist bone's axis, toward the palm
RUN = {"lift": 0.5, "dip": 0.02, "lean": 7.0}      # run_in_place: the feet rise this share of `reach`; the hips dip (m)
BOUNCE = {"depth": 0.045, "attack": 0.05, "decay": 0.18, "dip": 0.035, "up": 0.3}   # the knees' pump (m, s), the fists'
#                                                    dip with it (m), how long the fists stay up after an accent (s)
JUMP = {"height": 0.07, "flight": 0.3, "land": 0.04, "tuck": 0.07, "back": 0.03, "again": 0.25}   # an accent's hop: m,
#                                                    s, the knees giving on the landing (m), the legs' tuck in the air
#                                                    (m), and the least time from a landing to the next take-off (s)
CHOMP_GAP = 0.25                                   # s: the least time between two bites (the mouth opens and shuts)
SCRUB = 0.02                                       # m: a toothbrush's scrub each way from the middle of the teeth
SWEEP = {"hi": 0.025, "lo": 0.06}                  # m: how far the high and the low fist sweep each way
STIR = 0.04                                        # m: the radius of a stirring fist's circle
CHORES = ("sweep", "stir", "water")
MOUTH_MORPHS = ("a", "i", "u", "e", "o")           # the vowels lip sync keys: a move that has them has the mouth

# hand: "gag" (keys made by `plan`); both: both hands (pair "mirror": a mirror image, the hand_mirror check holds them so;
# "anti": in turns); other: the hand that is not `hand` plays too; prop: "need" / "may": a held prop (`[[move]] prop`);
# hits: it plays on its own hits; the rest as mkmmd.core.moves.MOVES
MOVES = {
    "yawn": {"hand": "gag", "both": True, "pair": "mirror", "prop": "may", "mirror": True, "lean": -9.0,
             "head_tilt": 12.0},
    "chomp": {"hand": "gag", "prop": "need", "hits": True, "lean": 5.0, "face": [("smile_eyes", 0.5)]},
    "brush": {"hand": "gag", "prop": "need", "hits": True, "lean": 3.0, "face": [("i", 0.7), ("smile_eyes", 0.3)]},
    "run_in_place": {"hand": "gag", "both": True, "pair": "anti", "hits": True, "lean": RUN["lean"],
                     "face": [("cheerful", 0.4), ("mouth_smile", 0.4)]},
    "wag": {"hand": "gag", "other": True, "hits": True, "mirror": True, "lean": 4.0, "head_tilt": 7.0,
            "face": [("jito", 0.6), ("mouth_down", 0.5), ("angry", 0.25)]},
    "wink_point": {"hand": "places", "places": (("point", 0.0),), "mirror": True, "wink": True, "lean": 7.0,
                   "head_tilt": 9.0, "twitch": {"bones": ["head"], "deg": 7.0, "axis": [1, 0, 0], "dur": 0.3},
                   "face": [("mouth_smile", 0.7)]},
    "bounce": {"hand": "gag", "both": True, "pair": "mirror", "hits": True, "accents": True, "lean": 4.0,
               "face": [("cheerful", 0.5), ("mouth_smile", 0.5)]},
    "chores": {"hand": "gag", "hits": True, "chore": True, "lean": 8.0, "face": [("calm", 0.3)]},
}
# the plain places of each gag move per hand: "hand" the move's hand, "other" the other one, "both" each
PLAYS = {"yawn": {"both": ("stretch", "stretch_out")}, "run_in_place": {"both": ("pump_fwd", "pump_back")},
         "wag": {"hand": ("wag_out", "wag_in"), "other": ("hip",)}, "bounce": {"both": ("club", "pump_up")},
         "sweep": {"hand": ("sweep_hi",), "other": ("sweep_lo",)}, "stir": {"hand": ("stir",), "other": ("bowl",)},
         "water": {"hand": ("water", "water_tilt")}}
PLACES = ("stretch", "stretch_out", "pump_up", "club", "pump_fwd", "pump_back", "wag_out", "wag_in", "sweep_hi",
          "sweep_lo", "stir", "bowl", "water", "water_tilt")


def place(name, side, marks):
    """(point, dir, palm, shape) of the gag place `name` (PLACES) for the hand `side` as written, in the character's
    frame (as mkmmd.core.moves.place: the wrist, the way the fingers point, the way the palm faces, the finger shape)."""
    s = 1.0 if side == "L" else -1.0
    arm, chest, eye = (np.asarray(marks[k], float) for k in (f"arm.{side}", "chest", "eye"))
    R = float(marks["reach"])

    def out(n):
        d = _unit(OUT[n][0])
        return arm + OUT[n][1] * R * np.array([s * d[0], d[1], d[2]])

    wag = eye + R * _v(s * 0.40, -0.30, -0.30)                 # beside the face, the forearm upright under it
    P = {
        "stretch": (out("stretch"), _v(s * 0.12, 0.25, 1.0), _v(0, -1, 0), "fist"),
        "stretch_out": (out("stretch_out"), _v(s, 0.15, 0.3), _v(0, -1, 0.3), "open"),
        "pump_up": (out("pump_up"), _v(s * 0.08, -0.12, 1.0), _v(0, -1, 0), "fist"),
        "club": (chest + R * _v(s * 0.30, -0.62, 0.22), _v(-s * 0.35, -0.75, 0.55), _v(-s, 0, 0), "fist"),
        "pump_fwd": (chest + R * _v(s * 0.20, -0.58, 0.18), _v(-s * 0.4, -0.7, 0.6), _v(-s, 0, 0), "fist"),
        "pump_back": (chest + R * _v(s * 0.50, 0.32, -0.33), _v(s * 0.05, -0.45, -0.89), _v(-s, 0, 0), "fist"),
        "wag_out": (wag, _v(s * 0.42, 0, 0.91), _v(0, -1, 0), "point"),          # the index up, tipped out ...
        "wag_in": (wag, _v(-s * 0.42, 0, 0.91), _v(0, -1, 0), "point"),         # ... and in, the palm to the lens
        "sweep_hi": (chest + R * _v(-s * 0.06, -0.40, 0.12), _v(0, -0.5, -0.8), _v(-s, 0, 0), "curled"),
        "sweep_lo": (chest + R * _v(s * 0.18, -0.46, -0.24), _v(0, -0.6, -0.8), _v(-s, 0, 0), "curled"),
        "stir": (chest + R * _v(s * 0.06, -0.50, -0.05), _v(-s * 0.3, -0.8, -0.4), _v(-s, 0, 0), "curled"),
        "bowl": (chest + R * _v(s * 0.12, -0.42, -0.22), _v(-s * 0.5, -0.85, 0.0), _v(0, 0, 1), "bowl"),
        "water": (out("water"), _v(s * 0.2, -0.6, -0.75), _v(-s, 0, 0), "curled"),
        "water_tilt": (out("water_tilt"), _v(s * 0.2, -0.75, -0.6), _v(-s, 0, 0), "curled"),
    }
    return P[name]


# ------------------------------------------------------------------------------------------------------- a held fist
_FISTS = {}


def _circle(P):
    """(centre, radius) of the circle nearest the points P (n, 3), fitted in their plane."""
    c = P.mean(0)
    _, _, Vt = np.linalg.svd(P - c)
    xy = (P - c) @ Vt[:2].T
    A = np.c_[xy, np.ones(len(xy))]
    a, b, k = np.linalg.lstsq(A, -(xy ** 2).sum(1), rcond=None)[0]
    centre = np.array([-0.5 * a, -0.5 * b])
    return c + Vt[:2].T @ centre, float(math.sqrt(max(centre @ centre - k, 0.0)))


def fist(marks, radius):
    """(finger shape, hole) of the member's left hand (mkmmd.core.moves.hand_rest) closed round a handle `radius` m thick:
    the curl of FIST scaled (within FIST_SCALE) until the middle finger's inner skin lies on the handle, the circle its
    joints and tip curl round fitted in their plane; `hole` the handle's axis there, in the hand's own frame (x along the
    hand, y its palm, z its little finger's side): its centre at the middle of the four fingers' spread."""
    MV = _mv()
    hand = MV.hand_rest(marks)
    key = (round(float(radius), 4), tuple(round(float(v), 5) for f in sorted(hand["points"])
                                          for v in np.ravel(hand["points"][f])))
    if key in _FISTS:
        return _FISTS[key]
    pts = {f: np.asarray(v, float) for f, v in hand["points"].items()}
    mid = pts["middle"]
    seg = float(np.linalg.norm(np.diff(mid, axis=0), axis=1).sum())
    skin = (hand.get("radii") or {}).get("middle") or FG.RADIUS * seg
    palm = _unit(hand["palm"])
    best = None
    for k in np.linspace(FIST_SCALE[0], FIST_SCALE[1], 40):
        ang = [min(a * k, 110.0) for a in FIST]
        P = FG.posed(mid, palm, ang)
        centre, r = _circle(P)
        err = abs((r - skin) - float(radius))
        if best is None or err < best[0]:
            best = (err, k, centre)
    _, k, centre = best
    ang = [round(min(a * k, 110.0), 1) for a in FIST]
    shape = {f: list(ang) for f in ("index", "middle", "ring", "little")}
    shape["thumb"] = list(FIST_THUMB)
    base = np.mean([pts[f][0] for f in ("index", "middle", "ring", "little") if f in pts], axis=0)
    hole = np.array([centre[0], centre[1], base[2]])
    _FISTS[key] = (shape, hole)
    return _FISTS[key]


# ------------------------------------------------------------------------------------------------------- prop geometry
def _scaled(v, k):
    return np.asarray(v, float) * float(k)


def hold_of(prop, geom):
    """The grip span of a held prop: {centre, axis (unit, toward the business end), length, radius(s)}: its card's
    `use.grip` `hold` (type "hold"), scaled to the prop."""
    use, k = geom.get("use") or {}, float(geom.get("scale", 1.0))
    for g in use.get("grip", []):
        if g.get("name") == "hold" and g.get("type", "hold") == "hold":
            if "center" not in g or "axis" not in g:
                raise GagError(f"prop {prop!r}: its use.grip `hold` needs a center and an axis")
            prof = [[float(s) * k, float(r) * k] for s, r in (g.get("profile") or [])]
            return {"centre": _scaled(g["center"], k), "axis": _unit(g["axis"]), "length": float(g.get("length", 0.0)) * k,
                    "radius": float(g.get("radius", 0.015)) * k,
                    "radius_tip": None if g.get("radius_tip") is None else float(g["radius_tip"]) * k,
                    "profile": sorted(prof)}
    raise GagError(f"prop {prop!r} has no use.grip `hold` (type \"hold\": center, axis, length, radius) for a fist")


def radius_at(hold, s):
    """The handle's radius `s` m from the business end of the grip span (its `profile`, else from `radius_tip` there to
    `radius` at the other end, else `radius`)."""
    if hold["profile"]:
        ss, rr = zip(*hold["profile"])
        return float(np.interp(s, ss, rr))
    if hold["radius_tip"] is not None and hold["length"] > 1e-6:
        u = min(max(s / hold["length"], 0.0), 1.0)
        return hold["radius_tip"] + u * (hold["radius"] - hold["radius_tip"])
    return hold["radius"]


def _anchor(prop, geom, name, kind="anchor"):
    k = float(geom.get("scale", 1.0))
    for a in (geom.get("use") or {}).get(kind, []):
        if a.get("name") == name:
            return a, k
    raise GagError(f"prop {prop!r} has no use.{kind} `{name}`")


def bites_of(prop, geom):
    """[point, ...] of a prop's bites in order (`use.bite` `bite1`, `bite2` ...: where the k-th is bitten), scaled."""
    k = float(geom.get("scale", 1.0))
    out = []
    for b in (geom.get("use") or {}).get("bite", []):
        n = str(b.get("name", ""))
        if n.startswith("bite") and n[4:].isdigit() and "point" in b:
            out.append((int(n[4:]), _scaled(b["point"], k), str(b.get("prop", n))))
    if not out:
        raise GagError(f"prop {prop!r} has no use.bite points (bite1, bite2 ...) to chomp")
    out.sort(key=lambda x: x[0])
    if [i for i, _, _ in out] != list(range(1, len(out) + 1)):
        raise GagError(f"prop {prop!r}: its use.bite points are not bite1, bite2 ... in a row")
    return [(p, key) for _, p, key in out]


def _frame(a, b):
    """The rotation whose columns are the unit `a`, `b` made square to it, and their cross."""
    a = _unit(a)
    b = _unit(np.asarray(b, float) - (np.asarray(b, float) @ a) * a)
    return np.stack([a, b, np.cross(a, b)], 1)


def turned(a_prop, b_prop, a_to, b_to):
    """The rotation that turns the prop's own `a_prop` onto `a_to` and its `b_prop` as near onto `b_to` as square to it."""
    return _frame(a_to, b_to) @ _frame(a_prop, b_prop).T


# ---------------------------------------------------------------------------------------------------------- holding
def _hand_frame(d, p):
    """The hand's frame as a rotation: columns its direction, its palm and their cross (mkmmd.core.moves._basis)."""
    return _frame(d, p)


def lips(marks, clear):
    """Where the lips are in the character's frame: the first point coming in from 15 cm in front of the mouth landmark
    that is within LIP_MARGIN of the face (`clear`), else the mouth landmark itself (no body to measure)."""
    MV = _mv()
    mouth = np.asarray(marks["mouth"], float)
    if clear is None:
        return mouth
    path = MV.line(mouth, _v(0.0, -1.0, 0.0))
    k = MV.sweep(path, lambda w: w[None], clear, LIP_MARGIN, 0.15, 0.002)
    return path(k)


def grip_point(hold, s):
    """The point `s` m along the grip span from its business end, in the prop's frame."""
    return hold["centre"] + hold["axis"] * (0.5 * hold["length"] - s)


def fit(side, marks, target, part, m, grip, rel=None, roll=None, rounds=8, pole=None):
    """A held place: (wrist, dir, palm, prop rotation, prop origin, forearm, pole) of the hand `side` holding a prop so
    that its point `part` (prop frame) is at `target` (character frame), the grip's axis (toward the business end) along
    `m` out of the fist's thumb side. `grip` (shape, hole, point on the prop's axis, the axis in the prop's frame) says
    where the fist holds it. Without `rel` the prop's roll about its axis is set by `roll` ((its own vector, the way it
    should face), e.g. its front to the lens) and the hand then continues its forearm as far as the fist allows (moves.
    aimed); with `rel` (the prop in the hand's frame, (R, T): a grip already set) the hand continues its forearm and the
    prop turns with it. The elbow is turned to keep the arm's twists in range (moves.arm_for), from `pole` (a point: the
    elbow of the place before, so a swing keyed place by place keeps its elbow) while that keeps them, else each round
    from the pose stage's own; the wrist is moved until the part is on its target."""
    MV = _mv()
    _, hole, G, axis = grip
    m = _unit(m)
    target = np.asarray(target, float)
    Rp = turned(axis, roll[0], m, roll[1]) if rel is None else None
    W = target - (Rp @ (np.asarray(part, float) - G) if Rp is not None else m * float(np.linalg.norm(part - G)))
    d = p = None
    start = pole
    for _ in range(rounds):
        d, p, _, pole = MV.arm_for(side, marks, W, lambda ff: MV.aimed(side, ff, m), start, MV.HELD, None)
        start = pole if start is not None else None
        H = _hand_frame(d, p)
        off = MV._off(hole, d, p, side)
        if rel is None:
            R, origin = Rp, W + off - Rp @ G
        else:
            R, origin = H @ rel[0], W + H @ rel[1]
        miss = target - (origin + R @ np.asarray(part, float))
        W = W + miss
        if float(np.linalg.norm(miss)) < 1e-5:
            break
    H = _hand_frame(d, p)
    if rel is None:
        R, origin = Rp, W + MV._off(hole, d, p, side) - Rp @ G
    else:
        R, origin = H @ rel[0], W + H @ rel[1]
    return W, d, p, R, origin, MV.forearm(side, marks, W, pole), pole


def _resolved(side, marks, W, d, p, f, pole, shape):
    """The tuple mkmmd.core.moves.resolve gives (point, dir, palm, shape, moved, bend, pole) of a solved held place."""
    MV = _mv()
    return (np.asarray(W, float), _unit(d), _unit(p), shape, 0.0, MV.bend(d, f), pole)


def _reach_ok(side, marks, W):
    S = np.asarray(marks[f"arm.{side}"], float)
    return float(np.linalg.norm(np.asarray(W, float) - S)) <= GRIP_REACH * float(marks["reach"])


def _grip(side, marks, hold, s):
    """(shape, hole, grip point, axis) of a fist closed on the grip span `s` m from its business end."""
    shape, hole = fist(marks, radius_at(hold, s))
    return shape, hole, grip_point(hold, s), hold["axis"]


def _tilted(m, deg):
    """The unit `m` turned `deg` degrees up toward +z about the horizontal square to it (negative: down)."""
    up = _v(0.0, 0.0, 1.0) - float(m @ _v(0.0, 0.0, 1.0)) * m
    if float(np.linalg.norm(up)) < 1e-6:
        return m
    t = math.radians(deg)
    return _unit(m * math.cos(t) + _unit(up) * math.sin(t))


def chomp_places(prop, geom, side, marks, clear, bites, held, total=0):
    """The places of a chomp of `prop` with the hand `side`: {name: resolved} for `bite<k>@<prop>` (bite k's point between
    the lips) and `ready<k>@<prop>` (the prop tipped READY_TILT degrees down about the fist from there: before the bite
    and while chewing) for each bite number in `bites`, and the held prop's attach (`held`: the grip a first hold sets,
    reused after). The prop lies along CHOMP_DIR from the mouth to the fist, its front (-Y) to the lens; the fist holds
    the grip span as far from the business end as still brings the first bite to the lips with the wrist in reach
    (GRIP_REACH), and BITE_ROOM past the deepest of the prop's `total` bites (every chomp's), else as near the business
    end as that room allows."""
    s = 1.0 if side == "L" else -1.0
    hold = hold_of(prop, geom)
    pts = bites_of(prop, geom)
    total = max(int(total), max(bites))
    if total > len(pts):
        raise GagError(f"prop {prop!r} has {len(pts)} bites (use.bite) and is chomped {total} times")
    lip = lips(marks, clear) + _v(0.0, LIP_INSET, 0.0)
    m = -_unit(_v(s * CHOMP_DIR[0], CHOMP_DIR[1], CHOMP_DIR[2]))     # the prop from the fist to the mouth
    roll = (_v(0.0, -1.0, 0.0), _v(0.0, -1.0, 0.0))               # its front to the lens
    grip, rel = held.get("grip"), held.get("rel")
    if grip is None:
        deepest = min(float((pts[b][0] - hold["centre"]) @ hold["axis"]) for b in range(total))   # along the axis
        lo = min(max(0.5 * hold["length"] - deepest + BITE_ROOM, 0.0), hold["length"])   # the nearest the bites allow
        if float((grip_point(hold, lo) - hold["centre"]) @ hold["axis"]) > deepest - BITE_ROOM + 1e-6:
            raise GagError(f"prop {prop!r}: its grip span leaves no room for a fist {BITE_ROOM:g} m from its bites")

        def reaches(sv):
            return _reach_ok(side, marks, fit(side, marks, lip, pts[0][0], m, _grip(side, marks, hold, sv), roll=roll)[0])

        hi = hold["length"]
        if reaches(hi):                                            # held at the far end of the span, still in reach
            lo = hi
        elif reaches(lo):                                          # the furthest from the bites that reaches
            for _ in range(GRIP_ROUNDS):
                mid = 0.5 * (lo + hi)
                lo, hi = (mid, hi) if reaches(mid) else (lo, mid)
        grip = _grip(side, marks, hold, lo)
    out = {}
    for b in sorted(set(bites)):
        part = pts[b - 1][0]
        W, d, p, R, origin, f, pole = fit(side, marks, lip, part, m, grip, rel, roll)
        if rel is None:                                              # the first hold sets how it sits in the fist
            H = _hand_frame(d, p)
            rel = (H.T @ R, H.T @ (origin - W))
            held.update(grip=grip, rel=rel, ref=(W, d, p, R, origin))
        out[f"bite{b}@{prop}"] = _resolved(side, marks, W, d, p, f, pole, grip[0])
        hole = W + _mv()._off(grip[1], d, p, side)                   # tipped down about the fist: away from the mouth
        tip = _tilted(m, -READY_TILT)
        W, d, p, _, _, f, pole = fit(side, marks, hole + tip * float(np.linalg.norm(lip - hole)), part, tip, grip, rel)
        out[f"ready{b}@{prop}"] = _resolved(side, marks, W, d, p, f, pole, grip[0])
    return out


def brush_places(prop, geom, side, marks, clear, held):
    """The place `brush@<prop>` of a toothbrush held by the hand `side`: its bristles' middle (`use.anchor` `brush`) on
    the teeth, facing them, the brush along BRUSH_DIR from the teeth to the fist, the fist on the grip span where it is
    nearest BRUSH_GRIP of `reach` from the bristles (a giant toothbrush is held down its handle, its hand clear of the
    face)."""
    s = 1.0 if side == "L" else -1.0
    hold = hold_of(prop, geom)
    a, k = _anchor(prop, geom, "brush")
    part, face = _scaled(a["point"], k), _unit(a.get("dir", (0.0, -1.0, 0.0)))
    target = lips(marks, clear) + _v(0.0, BRUSH_INSET, 0.0)
    m = -_unit(_v(s * BRUSH_DIR[0], BRUSH_DIR[1], BRUSH_DIR[2]))     # the brush from the fist to the teeth
    grip, rel = held.get("grip"), held.get("rel")
    if grip is None:
        want = BRUSH_GRIP * float(marks["reach"])
        span = np.linspace(0.0, hold["length"], max(int(hold["length"] / 0.005), 1) + 1)
        sv = float(min(span, key=lambda x: abs(float(np.linalg.norm(part - grip_point(hold, x))) - want)))
        grip = _grip(side, marks, hold, sv)
    W, d, p, R, origin, f, pole = fit(side, marks, target, part, m, grip, rel, (face, _v(0.0, 1.0, 0.0)))
    if rel is None:
        H = _hand_frame(d, p)
        held.update(grip=grip, rel=(H.T @ R, H.T @ (origin - W)), ref=(W, d, p, R, origin))
    return {f"brush@{prop}": _resolved(side, marks, W, d, p, f, pole, grip[0])}


def hug_pose(prop, geom, marks, clear):
    """(rotation, origin) of a hugged prop (a pillow) in the character's frame: upright, its front (`use.anchor` `hug`'s
    dir) ahead, the middle of its back (`hug`'s point) on the body's midline, its top HUG_DROP under the mouth and its
    back HUG_GAP off the body and clothes (`clear`; without one, a fifth of `reach` in front of the chest)."""
    MV = _mv()
    a, k = _anchor(prop, geom, "hug")
    hug, front = _scaled(a["point"], k), _unit(a.get("dir", (0.0, -1.0, 0.0)))
    R = turned(front, _v(0.0, 0.0, 1.0), _v(0.0, -1.0, 0.0), _v(0.0, 0.0, 1.0))
    b = geom.get("bounds") or {}
    lo, hi = (_scaled(b[x], k) if x in b else None for x in ("min", "max"))
    if lo is None or hi is None:
        raise GagError(f"prop {prop!r}: a hugged prop needs its card's bounds (min, max)")
    corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])]) @ R.T
    top = float(corners[:, 2].max()) - float((R @ hug)[2])          # its top above the hug point
    mouth, chest = np.asarray(marks["mouth"], float), np.asarray(marks["chest"], float)
    at = _v(0.0, float(chest[1]) - 0.2 * float(marks["reach"]), float(mouth[2]) - HUG_DROP - top)
    if clear is not None:                                    # its back comes in from the front onto the body
        back = np.array([R @ _v(x - hug[0], 0.0, z - hug[2]) for x in np.linspace(lo[0], hi[0], 7) * 0.9
                         for z in np.linspace(lo[2], hi[2], 5) * 0.9])     # its back's plane about the hug point
        path = MV.line(_v(0.0, float(chest[1]) + 0.05, float(at[2])), _v(0.0, -1.0, 0.0))
        sk = MV.sweep(path, lambda w_: w_[None] + back, clear, HUG_GAP, 0.45, 0.01)
        at = path(sk)
    origin = at - R @ hug
    return R, origin


def pillow_places(prop, geom, marks, clear, held):
    """The places `pillow@<prop>` of both hands hugging a pillow to the chest (`hug_pose`): each palm on its own end
    (`use.rest` `hold_L` / `hold_R`: a plane on the end, its normal out), the fingers forward along it and a little down.
    The pillow's pose goes to `held` (it rides the chest)."""
    MV = _mv()
    if held.get("pose") is None:
        held["pose"] = hug_pose(prop, geom, marks, clear)
    R, origin = held["pose"]
    out = {}
    for side in ("L", "R"):
        a, k = _anchor(prop, geom, f"hold_{side}", "rest")
        c, n = origin + R @ _scaled(a["center"], k), _unit(R @ _unit(a.get("normal", (1.0 if side == "L" else -1.0,
                                                                                            0.0, 0.0))))
        d0 = _unit(_v(0.0, -1.0, -0.3) - (_v(0.0, -1.0, -0.3) @ n) * n)
        W = c + n * PALM - 0.45 * float(marks["hand"]) * d0
        d, p, f, pole = MV.arm_for(side, marks, W, lambda ff, d0=d0, n=n: MV.natural(d0, -n, ff), None, MV.HELD, d0)
        out[side] = {f"pillow@{prop}": _resolved(side, marks, W, d, p, f, pole, "pillow")}
    return out


# ---------------------------------------------------------------------------------------------------------- the clock
def _times(e, t0, t1, beats, what="hits"):
    """The clip seconds a gag plays on: the entry's own `hits` (checked to fall inside it) or the beats inside it."""
    if "hits" in e:
        hs = e["hits"]
        if isinstance(hs, dict):
            raise GagError("hits = {lane = ...} is read off the [audio] hits file by the build: give clip seconds here")
        if not isinstance(hs, list) or not hs or not all(isinstance(h, (int, float)) and not isinstance(h, bool)
                                                         for h in hs):
            raise GagError("hits is a list of clip seconds")
        bs = sorted(float(h) for h in hs)
        if bs[0] < t0 - 1e-6 or bs[-1] >= t1 - 1e-6:
            raise GagError(f"its hits must fall inside it ({t0:g} .. {t1:g} s)")
        return bs
    bs = [float(b) for b in beats if t0 - 1e-6 <= b < t1 - 1e-6]
    if not bs:
        raise GagError(f"no beat between {t0:g} and {t1:g} s (the member's timeline gives the beats, or give hits)")
    return bs


def _halves(bs, t1):
    """[(time, the time halfway to the next one, or to t1 after the last)]."""
    return [(b, 0.5 * (b + nb)) for b, nb in zip(bs, list(bs[1:]) + [t1])]


def _increasing(seq, gap=0.02):
    """`seq` [(t, place, offset)] with keys closer than `gap` to the one before left out (the hand is there already)."""
    out = []
    for k in sorted(seq, key=lambda k: k[0]):
        if out and k[0] - out[-1][0] < gap:
            continue
        out.append(k)
    return out


def bite_numbers(entries, beats):
    """{prop: [clip second of each of its bites, in time order]}: the hits of every chomp entry, numbered on across them
    (bite k is the k-th)."""
    out = {}
    for e in entries:
        if not (isinstance(e, dict) and e.get("name") == "chomp" and isinstance(e.get("prop"), str)):
            continue
        t0 = float(e.get("t", 0.0))
        t1 = t0 + float(e.get("dur", 1.0))
        try:
            hs = _times(e, t0, t1, beats)
        except GagError:
            continue                                      # compile reports it on the entry itself
        out.setdefault(e["prop"], []).extend(hs)
    return {k: sorted(v) for k, v in out.items()}


def plan(e, m, t0, t1, beats, downbeats, hand, mic, marks, props, clear, bites, held):
    """The keys of the gag entry `e` (its move `m`: MOVES) from clip second t0 to t1 played with the hand `hand` (the
    member's hands at the mic in `mic`, {side: bool}): {"plays": {side: [(t, place, offset)]}, "places": {side: {name:
    resolved}} (the prop places, mkmmd.core.moves.resolve's tuples), "perform": {"bounce": [t], "rise", "crouch": [[t,
    m]], "air": [span], "walk": {..}}, "curves": {morph: [[t, v]]}, "mouth": [(t0, t1)], "values": {prop: {key:
    value}}}. `props` {name: {"use", "bounds", "scale"}}: the cards of the props it may hold; `bites` (`bite_numbers`);
    `held` {prop: {...}}: each held prop's grip and attach, set by the first move that holds it and kept for the rest."""
    name = e["name"]
    other = "R" if hand == "L" else "L"
    dur = t1 - t0
    out = {"plays": {}, "places": {"L": {}, "R": {}}, "perform": {}, "curves": {}, "mouth": [], "values": {}}
    prop = e.get("prop")
    if prop is not None:
        if not m.get("prop"):
            raise GagError(f"`prop` is for a move that holds one ({', '.join(k for k, v in MOVES.items() if v.get('prop'))})")
        if not isinstance(prop, str) or prop not in props:
            raise GagError(f"prop = {prop!r}: no such prop ([[prop]] names: {', '.join(sorted(props)) or 'none'})")
    elif m.get("prop") == "need":
        raise GagError("needs `prop`, the [[prop]] it holds")
    chore = e.get("chore", "sweep" if m.get("chore") else None)
    if "chore" in e and not m.get("chore"):
        raise GagError("`chore` is for `chores`")
    if m.get("chore") and chore not in CHORES:
        raise GagError(f"chore = {chore!r}: one of {', '.join(CHORES)}")
    if "accents" in e and not m.get("accents"):
        raise GagError("`accents` is for `bounce`")

    def need(*sides):
        for sd in sides:
            if mic[sd]:
                raise GagError(f"needs the {sd} hand, and it holds the mic")

    def free(sides):
        return [sd for sd in sides if not mic[sd]]

    plays = out["plays"]
    if name == "yawn":
        if prop is not None:
            need("L", "R")
            h = held.setdefault(prop, {})
            if h.get("bone") == "wrist":
                raise GagError(f"prop {prop!r} is held in a hand by another move: a hugged prop rides the chest")
            got = pillow_places(prop, props[prop], marks, clear, h)
            for sd in ("L", "R"):
                out["places"][sd].update(got[sd])
                plays[sd] = [(t0, f"pillow@{prop}", (0, 0, 0))]
            h["bone"] = "chest"
        else:
            for sd in free(("L", "R")):
                plays[sd] = [(t0, "stretch", (0, 0, 0)), (t0 + 0.55 * dur, "stretch_out", (0, 0, 0))]
        out["curves"] = {"a": [[t0, 0.0], [t0 + 0.35 * dur, 1.0], [t0 + 0.7 * dur, 0.95], [t0 + 0.9 * dur, 0.0]],
                         "o": [[t0, 0.0], [t0 + 0.4 * dur, 0.35], [t0 + 0.7 * dur, 0.3], [t0 + 0.9 * dur, 0.0]],
                         "calm": [[t0 + 0.1 * dur, 0.0], [t0 + 0.3 * dur, 0.9], [t0 + 0.8 * dur, 0.9],
                                  [t0 + 0.95 * dur, 0.0]]}
        out["mouth"].append((t0, t1))
    elif name == "chomp":
        need(hand)
        hs = _times(e, t0, t1, beats)
        if any(b - a < CHOMP_GAP - 1e-6 for a, b in zip(hs, hs[1:])):
            raise GagError(f"its hits are bites: at least {CHOMP_GAP:g} s apart (the mouth opens and shuts on each)")
        order = bites.get(prop, hs)
        ks = [order.index(h) + 1 for h in hs]
        h = held.setdefault(prop, {})
        if h.get("side", hand) != hand or h.get("bone") == "chest":
            raise GagError(f"prop {prop!r} is held {'in the ' + h['side'] + ' hand' if 'side' in h else 'to the chest'}"
                           f" by another move")
        out["places"][hand].update(chomp_places(prop, props[prop], hand, marks, clear, ks, h, len(order)))
        h.update(side=hand, bone="wrist")
        names = bites_of(prop, props[prop])
        seq = [(t0, f"ready{ks[0]}@{prop}", (0, 0, 0))]
        a = []
        for i, (hit, k) in enumerate(zip(hs, ks)):
            nxt = hs[i + 1] if i + 1 < len(hs) else t1
            seq += [(max(hit - 0.2, t0 + 0.02), f"ready{k}@{prop}", (0, 0, 0)), (hit, f"bite{k}@{prop}", (0, 0, 0))]
            after = f"ready{ks[i + 1]}@{prop}" if i + 1 < len(ks) else f"ready{k}@{prop}"
            if nxt - hit > 0.35:
                seq.append((hit + 0.16, after, (0, 0, 0)))
            a += [[hit - 0.22, 0.0], [hit - 0.04, 1.0], [hit + 0.02, 0.0]]     # wide open as it comes, shut on the hit
            for dt, v in ((0.14, 0.3), (0.26, 0.05), (0.38, 0.25), (0.5, 0.0)):  # ... then chewing
                if hit + dt < nxt - 0.24:
                    a.append([hit + dt, v])
            out["values"].setdefault(prop, {})[names[k - 1][1]] = round(hit, 4)
        plays[hand] = _increasing(seq)
        out["curves"]["a"] = a
        out["mouth"].append((t0, t1))
    elif name == "brush":
        need(hand)
        bs = _times(e, t0, t1, beats)
        h = held.setdefault(prop, {})
        if h.get("side", hand) != hand or h.get("bone") == "chest":
            raise GagError(f"prop {prop!r} is held {'in the ' + h['side'] + ' hand' if 'side' in h else 'to the chest'}"
                           f" by another move")
        out["places"][hand].update(brush_places(prop, props[prop], hand, marks, clear, h))
        h.update(side=hand, bone="wrist")
        at = f"brush@{prop}"
        seq = [(t0, at, (0, 0, 0))] if bs[0] - t0 > 0.05 else []
        for b, half in _halves(bs, t1):
            seq += [(b, at, (SCRUB, 0.0, 0.0)), (half, at, (-SCRUB, 0.0, 0.0))]
        plays[hand] = _increasing(seq)
        out["mouth"].append((t0, t1))
    elif name == "run_in_place":
        bs = _times(e, t0, t1, beats)
        knees = []
        for b, half in _halves(bs, t1):
            knees += [b, half] if half - b > 0.12 else [b]
        gap = float(np.median(np.diff(knees))) if len(knees) > 1 else 0.4
        knees = [k for k in knees if k <= t1 - 0.45 * gap + 1e-9]
        knees = knees[:len(knees) - len(knees) % 2] or knees   # an even count: each run starts on the left knee
        steps = [round(k + 0.45 * gap, 4) for k in knees]
        out["perform"]["walk"] = {"steps": steps, "stride": 0.0, "lift": round(RUN["lift"] * float(marks["reach"]), 3),
                                  "dip": RUN["dip"], "slide": 1.0, "roll": 0.15, "first": "L"}
        for sd in free(("L", "R")):
            seq = []
            for i, k in enumerate(knees):                    # the L knee up on the even ones: its arm back, the other's
                up = "L" if i % 2 == 0 else "R"              # forward
                seq.append((k, "pump_back" if sd == up else "pump_fwd", (0, 0, 0)))
            plays[sd] = _increasing(seq)
    elif name == "wag":
        need(hand)
        bs = _times(e, t0, t1, beats)
        seq = [(t0, "wag_in", (0, 0, 0))] if bs[0] - t0 > 0.05 else []      # its wind-up, unless it starts on a beat
        for b, half in _halves(bs, t1):
            seq += [(b, "wag_out", (0, 0, 0)), (half, "wag_in", (0, 0, 0))]
        plays[hand] = _increasing(seq)
        if not mic[other]:
            plays[other] = [(t0, "hip", (0, 0, 0))]
    elif name == "bounce":
        hs = _times(e, t0, t1, beats)
        if "accents" in e:
            ac = e["accents"]
            if isinstance(ac, dict):
                raise GagError("accents = {lane = ...} is read off the [audio] hits file by the build: give clip seconds")
            if not isinstance(ac, list) or not all(isinstance(a, (int, float)) and not isinstance(a, bool) for a in ac):
                raise GagError("accents is a list of clip seconds")
            acc = sorted(float(a) for a in ac)
            if acc and (acc[0] < t0 - 1e-6 or acc[-1] >= t1 - 1e-6):
                raise GagError(f"its accents must fall inside it ({t0:g} .. {t1:g} s)")
        else:
            acc = [float(b) for b in downbeats if t0 - 1e-6 <= b < t1 - 1e-6]
        events = sorted({round(h, 4) for h in hs} | {round(a, 4) for a in acc})
        accents = {round(a, 4) for a in acc}
        for sd in free(("L", "R")):
            seq = [(t0, "club", (0, 0, 0))] if events[0] - t0 > 0.05 else []
            for ev, nxt in zip(events, events[1:] + [t1]):
                gap = nxt - ev
                if ev in accents:
                    seq.append((ev, "pump_up", (0, 0, 0)))
                    if gap > 2.0 * BOUNCE["up"]:
                        seq.append((ev + BOUNCE["up"], "pump_up", (0, 0, 0)))
                        seq.append((ev + min(BOUNCE["up"] + 0.25, 0.5 * (gap + BOUNCE["up"])), "club", (0, 0, 0)))
                else:
                    seq += [(ev, "club", (0.0, 0.0, -BOUNCE["dip"])), (ev + 0.5 * gap, "club", (0, 0, 0))]
            plays[sd] = _increasing(seq)
        out["perform"]["bounce"] = [round(h, 4) for h in hs]
        rise, crouch, air = [], [], []
        last = -math.inf
        for a in acc:
            T = JUMP["flight"]
            if a - last < T + JUMP["again"]:
                continue                                    # still in the air from the last one, or just landed
            last = a
            rise += [[round(a, 4), 0.0], [round(a + 0.5 * T, 4), JUMP["height"]], [round(a + T, 4), 0.0]]
            crouch += [[round(a + T - 0.02, 4), 0.0], [round(a + T + 0.06, 4), JUMP["land"]],
                       [round(a + T + 0.24, 4), 0.0]]
            air.append({"from": round(a + 0.01, 4), "to": round(a + T - 0.01, 4), "tuck": JUMP["tuck"],
                        "back": JUMP["back"], "rise": 0.09, "extend": 0.1, "point": 15.0})
        if acc:
            out["perform"].update(rise=rise, crouch=crouch, air=air)
    elif name == "chores":
        bs = _times(e, t0, t1, beats)
        if chore == "sweep":
            need(hand, other)
            lo = 1.0 if other == "L" else -1.0               # the low hand's side: the broom's head sweeps out to it
            lead = bs[0] - t0 > 0.05
            hi_seq, lo_seq = [(t0, "sweep_hi", (0, 0, 0))] * lead, [(t0, "sweep_lo", (0, 0, 0))] * lead
            for b, half in _halves(bs, t1):
                hi_seq += [(b, "sweep_hi", (lo * SWEEP["hi"], 0.0, 0.0)), (half, "sweep_hi", (-lo * SWEEP["hi"], 0.0, 0.0))]
                lo_seq += [(b, "sweep_lo", (lo * SWEEP["lo"], -0.01, 0.0)), (half, "sweep_lo", (-lo * SWEEP["lo"], 0.01, 0.0))]
            plays[hand], plays[other] = _increasing(hi_seq), _increasing(lo_seq)
        elif chore == "stir":
            need(hand, other)
            seq = []
            for b, nb in zip(bs, bs[1:] + [t1]):
                for j, (x, y) in enumerate(((1.0, 0.0), (0.0, -1.0), (-1.0, 0.0), (0.0, 1.0))):
                    seq.append((b + 0.25 * j * (nb - b), "stir", (STIR * x, STIR * y, 0.0)))
            start = [(t0, "stir", (0, 0, 0))] if bs[0] - t0 > 0.05 else []
            plays[hand], plays[other] = _increasing(start + seq), [(t0, "bowl", (0, 0, 0))]
        else:
            need(hand)
            seq = [(t0, "water", (0, 0, 0))] if bs[0] - t0 > 0.05 else []
            for b, half in _halves(bs, t1):
                seq += [(b, "water_tilt", (0, 0, 0)), (half, "water", (0, 0, 0))]
            plays[hand] = _increasing(seq)
    else:
        raise GagError(f"no gag {name!r}")
    return out


def legs(got, mouths):
    """The perform keys of every gag move of a member at once: `got` {key: [what each move gave]} (`plan`'s "perform"),
    `mouths` [(t0, t1)] the spans the moves have the mouth. {"bounce": {beats, depth, attack, decay} (the knees pump on
    every bounce's hits), "rise", "crouch": [[t, m], ...] (the jumps, the landings), "air": [span] (the legs tuck),
    "walk": {steps, ...} (every run's steps), "mouth": [[t0, t1], ...] (merged)}, only what some move asks for."""
    out = {}
    beats = sorted({t for v in got.get("bounce", []) for t in v})
    if beats:
        out["bounce"] = {"beats": beats, "depth": BOUNCE["depth"], "attack": BOUNCE["attack"], "decay": BOUNCE["decay"]}
    for key in ("rise", "crouch"):
        keys = sorted(k for v in got.get(key, []) for k in v)
        if keys:
            out[key] = keys
    air = sorted((s for v in got.get("air", []) for s in v), key=lambda s: s["from"])
    if air:
        out["air"] = air
    walks = got.get("walk", [])
    if walks:
        out["walk"] = dict(walks[0], steps=sorted(t for w in walks for t in w["steps"]))
    merged = []
    for a, b in sorted(mouths):
        if merged and a <= merged[-1][1] + 1e-6:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    if merged:
        out["mouth"] = [[round(a, 4), round(b, 4)] for a, b in merged]
    return out


def attachments(grips):
    """{prop: {"bone": "wrist", "side", "rel": [rotation, offset]} | {"bone": "chest", "pose": [rotation, origin]}} of
    the props the gag moves hold (`plan`'s `held`): a held one in the frame of the hand that holds it (columns its
    direction, its palm and their cross: the wrist bone's axes as the pose stage sets them), a hugged one in the
    character's frame, riding the chest from the settled pose."""
    out = {}
    for prop, h in grips.items():
        if h.get("bone") == "chest" and h.get("pose") is not None:
            R, origin = h["pose"]
            out[prop] = {"bone": "chest", "pose": [np.round(R, 6).tolist(), np.round(origin, 5).tolist()]}
        elif h.get("bone") == "wrist" and h.get("rel") is not None:
            R, T = h["rel"]
            out[prop] = {"bone": "wrist", "side": h["side"], "rel": [np.round(R, 6).tolist(), np.round(T, 5).tolist()]}
    return out


def rows(e, mic):
    """[(side, places, mirrored)] of the hands a gag entry plays (the hand_room / hand_mirror / wrist_bend checks hold its
    hands at these places; `mic` {side: bool} the hands that stay at the mic): the places as `plan` names them."""
    name = e.get("name")
    m = MOVES.get(name, {})
    hand = e.get("hand", "R")
    other = "R" if hand == "L" else "L"
    prop = e.get("prop")
    if name == "yawn" and prop is not None:
        return [(sd, [f"pillow@{prop}"], True) for sd in ("L", "R")]
    if name == "chomp":
        return [(hand, [f"chomp@{prop}"], False)]
    if name == "brush":
        return [(hand, [f"brush@{prop}"], False)]
    table = PLAYS.get(e.get("chore", "sweep") if name == "chores" else name, {})
    out = []
    for key, places in table.items():
        sides = ("L", "R") if key == "both" else (hand,) if key == "hand" else (other,)
        for sd in sides:
            if not mic.get(sd):
                out.append((sd, list(places), key == "both" and m.get("pair") == "mirror"))
    return out
