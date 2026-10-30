"""Moves, bpy-free (docs/design.md: Moves): `[[move.<cast>]]` entries place named moves of a small library on the clock,
and `compile` turns them into what the pose and perform stages already read: hand keys (`pose.<cast>.hands.L/R.keys`:
places in the character's frame, directions, finger shapes), upper-body lean and tilt keys, head twitches and keyed
expressions (`perform.<cast>`). A move is keys like any other, so a project can still key anything by hand beside them.

Hand places are built from the member's landmarks in its own frame (x its left, y behind it, z up, metres from its root;
the build reads them from the rest pose: `arm.L` / `arm.R` the arm joints, `chest`, `mouth`, `eye`, `upper` and `fore`
the arm's two lengths, `reach` from the arm joint to the wrist, `hand` from the wrist to the middle fingertip, and
optionally `eye_dx`, half the distance between the eyes, `pupil`, the left eye's visible pupil (its own mesh, in front
of and beside the eye bone), and `fingers`, the hand at rest: `hand_rest`), so one library fits any body: a place near
the body is an offset from a landmark, one with the arm out (OUT) a direction from the arm joint and a share of
`reach`, so it is never out of reach. Places by the face and the chest are on the hand's own side, clear of a mic held
at the mouth and of its cord. A gesture (GESTURES: a ring to peek through, a finger heart, half of a two-hand heart, a
butterfly's wing) is placed by where the member's own hand, posed in its shape (mkmmd.core.fingers.pose), has the part
that must read: the ring's hole (`hole`) on the line from the lens to the pupil (`pupil_at`) and its opening to the
lens, the finger heart's crossing by the cheek with its tips up and its face to the lens, the fingertips of a two-hand
heart on the body's midline, the hooked thumbs of the butterfly across it; `gesture` turns the hand to show it and sets
the wrist so that part lands there. The keys of a place seen through (SIGHTED: the ring) are marked `sight`: the pose
stage carries the hand with the head while it holds it there, so the ring stays on the eye as the move leans the body
and tilts the head, and through whatever else turns the head (a nod on the beat, a head rock).

Every place is made natural before it is keyed (`resolve`):
- the wrist: the elbow an arm IK gives (the pose stage's default pole, mkmmd.core.armreach, or a place's own, POLES)
  fixes the forearm, and a hand that would bend more than HELD degrees from it (WRIST, the rule, less the few the IK
  poses the arm off the plan) is turned toward it; a place for a fist round a held thing (a mic, a hammer) is aimed
  instead (AIM): the hand continues the forearm as far as a grip on a handle pointing that way allows (the mic at the
  mouth leans back to the lips from a fist under them, so the forearm rises to it); two hands that meet in one shape
  (SHAPED, the two-hand heart) keep the orientation written for them, sized to the arm so it suits any body;
- the twist: where that elbow would leave the forearm or the upper arm turned past what an arm turns
  (mkmmd.core.armtwist, less TWIST_MARGIN), the elbow is turned round the shoulder-wrist line the least way to where it
  does not (`arm_for`: up and out for a hand turned palm out, down and in under one turned palm up), and the key carries
  that elbow's pole; keys halfway through a trip whose elbows sit far apart keep the pole round the shoulder and the
  forearm turning the short way between its ends' twists (`_via`, `_midway`);
- the skin: given the build's distance to what a hand must stay off (`clear`: the member's own body and clothes, less
  the loose chains the sim moves, hair, ears, a tail, a skirt, ribbons, which make way for a hand), the hand comes in
  from outside and stops where it first comes within its margin: an arm held out (OUT) swings in round the shoulder, so
  it stays in reach (a hanging hand rises off a wide skirt the sim does not move), a hand by the body comes in straight
  (TOUCH, else straight out from the body's axis); a touching place (a pat on the chest, hands over the ears) ends on the
  surface, any other one where it was asked unless that is too close to, or inside, the body. A place on a skirt
  (CLOTH: pinching it, hands clasped behind the back) measures to the skirt too, simulated or not (`cloth`), and ends on
  it, so it rests on the cloth, not under it.
A hand that has played its move goes back to its rest place (`rest`: hanging by the hip; `dainty`: resting lightly on
the front of the skirt or thighs, the two hands together; `behind`: the hands clasped behind the back; `clasp`: the
hands clasped together at the chest; `skirt`: each hand pinching the skirt at its side; `mic`: holding a mic at the
mouth). `dainty`, `behind` and `clasp` are poses for two (LONE): a hand there alone, its partner away or resting
elsewhere, rests at its lone place instead (`dainty_one` on its own thigh, its wrist straight and its elbow out;
`behind_one` the back of the hand on the small of the back; `clasp_one` the hand on the chest), `_alone`. A rest entry
with a time (`t`) changes where the hand waits from then on: it is there on that time, as at a move's first place
(`_hand`). It goes home when it has time to stay there
REST_MIN seconds before its next move, else it goes straight on to that move. A hand travels at a human pace (`travel`:
TRAVEL_MIN, plus its wrist's way at TRAVEL_SPEED, or as long as its palm takes to turn over at TURN_SPEED) and reaches
each move's first place on the move's time, so it sets off before it, leaving the place it holds early if it must; a
move that leaves the hand too little time to get there is refused. A move that is a reaction (`react`: `flung`, arms
thrown up by a blast) is the other way round: the hand leaves where it is on the move's time and is thrown there
(`fling`: FLING_MIN, plus its way at FLING_SPEED). A hand that rests at the mic keeps its grip in every move: a two-hand
move leaves it at the mic (except one that raises it, `hands_up`, `flung`), and one that needs both hands free
(`heart_push`, `butterfly`, `bloom`) is refused.

A `free` move lets its hand go for the span: the arm IK eases off over FREE_EASE seconds and back on, and between the
arm follows whatever else drives it, a `[[motion]]` strip (a walk's arm swing, a singing motion's gestures). Given where
the strip has the hand over the span (`compile`'s `strip`: samples of its wrist, the hand's direction and its palm), the
hand's goal follows them: the hand travels to where the strip has it at the span's start (setting off early, at the
human pace, as to any place), the goal stays on the strip's hand while the IK eases off and back, so nothing moves where
the IK lets go and takes over, and it leaves the strip as late as still gets it to its next move on time (at the span's
end, or before). Without samples the goal waits at its rest place meanwhile, so the hand comes back from there.
`compile` gives such a hand `ik` keys, the share of its IK it feels, over the stretch it is let go."""
import math

import numpy as np

from . import armreach as AR
from . import armtwist as AT
from . import fingers as FG
from . import gags as GG


class MoveError(ValueError):
    """A bad `[[move.<cast>]]`."""


EASE = 0.12                      # seconds a move's lean or tilt takes to come in, and to go back
TRAVEL_MIN = 0.1                 # seconds the shortest move of a hand from one place to another takes ...
TRAVEL_SPEED = 1.6               # m/s: ... plus its wrist's way at this pace (eased in and out: it peaks near twice that)
TURN_SPEED = 600.0               # deg/s: ... or as long as turning its palm or its fingers' way at this pace takes
REST_MIN = 0.15                  # seconds a hand stays at its rest place at least, or it goes straight on to the next move
FREE_EASE = 0.25                 # seconds a `free` move takes to let the arm IK go, and to take it back
FLING_MIN = 0.06                 # seconds a hand thrown by a reaction (`react`) is on its way at least ...
FLING_SPEED = 4.5                # m/s: ... plus its wrist's way at this pace
WRIST = 60.0                     # degrees: the most a hand bends away from its forearm (the `wrist_bend` check's rule)
HELD = WRIST - 4.0               # ... and the most a place bends it as planned: the IK poses the arm a few degrees off
FAR = 0.3                        # m: how far out a hand by the body starts its approach to it
STEP = 0.01                      # m: the approach's step (the contact is then found to 0.003 mm by halving)
SWING = math.radians(70.0)       # how far up and away an arm held out starts its swing in to its place
SWING_STEP = math.radians(2.0)   # ... and its step
MARGIN = 0.012                   # m: the room a place that does not touch keeps from the body and clothes (`clear`)
TOUCH_MARGIN = 0.003             # m: ... and a touching one
STRAIGHT = {"dainty_one": 12.0,  # degrees: places whose hand continues its forearm closer than HELD (alone on a thigh;
            "peek": 48.0}        # a ring at the eye, which the head carries, `sight`: its tilts and nods bend the wrist)
TWIST_MARGIN = 10.0              # degrees: a place keeps its arm's twists this far inside the human range (AT.FOREARM,
                                 # AT.upper_limits), turning its elbow round the shoulder-wrist line to (`arm_for`)
POLE_OFF = 0.3                   # m: the pole of an elbow turned so: this far beyond it from the shoulder-wrist line
ELBOW_RISE = 0.12                # ... the elbow at most this share of `reach` above the shoulder (or as high as its own)
TORSO = (0.8, 1.3)               # ... and outside the torso: half its width and depth at the chest, shares of the arm
                                 # joint's distance from the body's middle
POLE_TURN = 60.0                 # degrees: two keys' elbow poles further apart than this round the shoulder pass a key
                                 # halfway with the pole round the shoulder between them (`_midway`)
REACH = 0.37                     # m: the arm (arm joint to wrist) the places' own poles (POLES) are written for
TWIST_STEP = 120.0               # degrees: a key halfway through a trip keeps the arm's twists less than this from the
                                 # keys about it, so the forearm turns between them the short way (`_midway`)
POLES = {  # places that hold the elbow their own way: its pole from the arm joint, (out, behind, below) as AR.POLE
    "dainty_one": (0.45, 0.0, 0.10),                  # alone on the skirt: the elbow out
    "peek": (0.3, -0.35, -0.05),                      # at the eye: the elbow out at about the shoulder's height, the
                                                      # forearm coming in from the side, under the ring and clear of the mouth
    "giggle": (0.15, -0.2, 0.5),                      # at the lips: the elbow low in front, the forearm upright
    "butterfly": (0.1, -0.15, 0.45), "butterfly_shut": (0.1, -0.15, 0.45),   # crossed before the chest: the elbows low
    "heart": (-0.15, -0.37, 0.2),                     # by the cheek: the elbow low in front, the forearm rising to it
    "bonk_up": (0.3, -0.24, -0.24),                   # a hammer cocked over the shoulder: the elbow out, forward and up
    "hair": (0.33, 0.02, -0.31),                      # in the hair: the elbow up and out to the side
    "hair_out": (0.42, 0.0, -0.16),                   # flicked out of it: the elbow still out to the side, a little up
    "ears": (0.42, -0.15, 0.0),                       # over the ears: the elbows out wide at the shoulders' height
    "heart_push": (0.35, 0.05, 0.3),                   # in front; the two-hand heart: the elbows out, the forearms inward
    "bud": (0.25, -0.1, 0.5), "bloom": (0.3, -0.05, 0.5), "clasp": (0.25, 0.0, 0.5),   # before the chest: the elbow down
    "behind": (0.3, 0.45, 0.25), "behind_one": (0.3, 0.45, 0.25),                       # behind the back: the elbow back
}
SHAPES = {                       # finger shapes (mkmmd.core.fingers tables) the moves use
    "relaxed": "relaxed", "flat": "flat", "fist": "fist", "point": "point", "curled": "curled",
    "peace": {"index": [0, 0, 0], "middle": [0, 0, 0], "ring": [80, 95, 65], "little": [80, 95, 65],
              "thumb": [30, 40, 40], "spread": 14},
    "paw": {"index": [55, 70, 45], "middle": [55, 70, 45], "ring": [55, 70, 45], "little": [55, 70, 45],
            "thumb": [20, 30, 20]},
    "heart": {"index": [80, 30, 10], "middle": [80, 95, 65], "ring": [80, 95, 65], "little": [80, 95, 65],
              "thumb": "cross"},                               # the finger heart: the thumb across the index's tip
    "sparkle": {"thumb": [0, 5, 5], "spread": 24},
    "heart2": {"index": [5, 55, 35], "middle": [10, 65, 40], "ring": [20, 70, 45], "little": [30, 75, 50],
               "thumb": [70, 0, 0]},             # half of a two-hand heart: the fingers curled over, the thumb down
    "gun": {"index": [0, 0, 0], "middle": [85, 95, 65], "ring": [85, 95, 65], "little": [85, 95, 65],
            "thumb": [0, 0, 0], "spread": {"thumb": 28}},       # the finger gun: index out, the thumb up, the rest curled
    "ring": {"index": [45, 55, 40], "middle": [8, 8, 4], "ring": [10, 10, 5], "little": [12, 12, 6], "thumb": "ring",
             "spread": {"ring": 10, "little": 22}},       # the thumb and index tips meet in a ring, the rest fanned
    "wings": {"index": [4, 4, 2], "middle": [4, 4, 2], "ring": [5, 5, 3], "little": [6, 6, 4], "thumb": [0, 10, 15],
              "spread": {"index": 6, "ring": 6, "little": 14, "thumb": 55}},   # a butterfly's wing, the thumb out to hook
    "bud": {"index": [25, 30, 20], "middle": [25, 30, 20], "ring": [28, 32, 20], "little": [30, 35, 22],
            "thumb": [20, 15, 10], "spread": -4},              # cupped round a flower bud
    "open": {"index": [10, 12, 8], "middle": [10, 12, 8], "ring": [12, 14, 8], "little": [14, 16, 10],
             "thumb": [5, 5, 5], "spread": 22},                # a flower open: the fingers fanned, curving like petals
    "reach": {"index": [6, 8, 4], "middle": [6, 8, 4], "ring": [8, 10, 5], "little": [10, 12, 6], "thumb": [5, 8, 5],
              "spread": 16},                                    # an open hand reaching out
    "giggle": {"index": [15, 20, 12], "middle": [15, 20, 12], "ring": [17, 22, 12], "little": [20, 24, 14],
               "thumb": [15, 15, 10], "spread": -2},           # the fingers together, curled a little, over the lips
    "clasp": {"index": [55, 35, 20], "middle": [55, 35, 20], "ring": [55, 35, 20], "little": [55, 35, 20],
              "thumb": [15, 10, 10], "spread": 6},              # two hands' fingers laced together
    "pinch": {"index": [40, 45, 25], "middle": [35, 40, 25], "ring": [40, 45, 30], "little": [45, 50, 30],
              "thumb": [35, 20, 15]},                           # the thumb and index pinching cloth
}
BOX = {  # shape: (how far the hand reaches along its direction, a share of `hand`; half its thickness, m)
    "flat": (1.0, 0.012), "relaxed": (0.95, 0.014), "sparkle": (1.0, 0.012), "peace": (1.0, 0.016),
    "point": (0.95, 0.018), "curled": (0.75, 0.022), "paw": (0.7, 0.024), "heart": (0.65, 0.024), "fist": (0.6, 0.025),
    "heart2": (0.7, 0.024), "gun": (0.95, 0.022), "ring": (0.95, 0.014), "wings": (1.0, 0.012), "bud": (0.85, 0.02),
    "open": (1.0, 0.012), "reach": (1.0, 0.013), "giggle": (0.95, 0.014), "clasp": (0.8, 0.022), "pinch": (0.8, 0.02),
}
WIDTH = 0.21                     # half the hand's width, a share of `hand`
MARKS = ("arm.L", "arm.R", "chest", "mouth", "eye", "reach", "upper", "fore", "hand")
EYE_DX = 0.2                     # half the distance between the eyes, a share of `hand`, for a member without `eye_dx`
GESTURES = {  # places made from the posed hand (`gesture`): their shape
    "peek": "ring", "heart": "heart", "heart_push": "heart2", "butterfly": "wings", "butterfly_shut": "wings"}
PEEK = (0.065, -15.0)            # the ring's hole this far (m) in front of the pupil; the fanned fingers this many degrees
                                 # from upright, + out to the hand's own side, - in toward the brow
SIGHTED = ("peek",)              # places seen through: their part (`sighted`) on the line from the lens to the pupil
SIGHT_ROUNDS, SIGHT_TOL = 6, 2e-4   # `resolve` moves a seen-through place back onto its line this often, to this (m)
HEART = (0.11, -0.12, 0.03, -20.0)   # the finger heart's crossing from the mouth (out, behind, up; m), its tips this many
                                     # degrees from upright (in), its index's side to the lens
PUSH = (-0.48, 0.59)             # the two-hand heart's top (the index fingertips) from the chest: behind and up, shares of
                                 # `reach`
BUTTERFLY = {"behind": -0.35,    # the hooked thumbs from the chest, a share of `reach`, at the height of the arm joints
             "below": 0.0,       # ... less this share of `reach`: before the collarbone, the wrists crossed below them
             "hook": 0.012,      # m: each thumb's tip past the midline (the other side's)
             "wing": 35.0,       # degrees the wings (the fingers) lean out from upright ...
             "lean": 15.0,       # ... and forward, toward the lens
             "shut": 30.0}       # degrees they fold forward about the thumbs on the beats
CROSS = {"butterfly": 0.05}      # m: two-hand moves whose wrists cross, the right hand this far in front of the left
OUT = {  # place: (direction from the arm joint for the left hand, x mirrored for the right; share of the reach)
    "rest": ((0.13, 0.03, -1.0), 0.97),
    "mic_lens": ((-0.05, -1.0, 0.05), 0.9),
    "mic_up": ((0.12, -0.6, 0.8), 0.9),
    "bounce": ((0.26, -0.75, -0.61), 0.85),
    "point": ((-0.14, -1.0, 0.05), 0.95),
    "up": ((0.17, -0.09, 1.0), 0.93),
    "mic_across": ((-0.6, -0.8, 0.0), 0.85),
    "bonk_up": ((0.35, -0.4, 1.0), 0.8),
    "bonk": ((0.12, -1.0, -0.75), 0.95),
    "gun": ((-0.12, -1.0, 0.18), 0.98),
    "float_low": ((0.62, -0.3, -0.72), 0.92),
    "float": ((0.86, -0.32, 0.12), 0.9),
    "reach": ((0.12, -1.0, 0.12), 0.9),
    "wings_up": ((0.92, 0.05, 0.38), 0.92),
    "wings_down": ((0.92, 0.0, -0.4), 0.92),
    "skirt": ((0.22, -0.12, -1.0), 0.99),
}
AIM = {  # a fist round a handle (a mic, a hammer): where the handle points, for the left hand, x mirrored for the right. A
    # fist holds a handle across its palm, so a straight wrist keeps it square to the forearm: the aims lean from upright
    "mic": (0.0, 0.77, 0.64),                          # at the mouth: leaning back to the lips from the fist under them
    "mic_lens": (0.0, -0.4, 0.92),                     # held out, leaning to the lens
    "mic_up": (0.0, 0.08, 1.0),                        # raised forward and up, the mic upright over it (the palm in)
    "mic_across": (-0.25, -0.3, 0.92),                 # held out to the other side, upright for a partner there
    "bonk_up": (0.0, 0.55, 0.83),                      # a hammer cocked up and back over the shoulder
    "bonk": (0.0, -0.55, 0.83),                        # struck: its head's face comes forward and down on what is in
                                                       # front at the height of the chest (a box, a door)
}
TOUCH = {  # a place on the body: the way the hand comes in to it, for the left hand, x mirrored for the right
    "chest": (0.0, -1.0, 0.0),
    "dainty_one": (0.5, -1.0, 0.0),
    "ears": (1.0, 0.0, 0.0),
    "dainty": (0.0, -1.0, 0.0),
    "hair": (1.0, 0.0, 0.0),
    "hip": (1.0, 0.0, 0.0),
    "behind": (0.0, 1.0, 0.0),
    "behind_one": (0.4, 1.0, 0.0),
    "clasp_one": (0.0, -1.0, 0.0),
}
SHAPED = ("heart_push", "butterfly", "butterfly_shut")   # two hands that meet in one shape: kept as written (`gesture`)
CLOTH = ("skirt", "behind", "behind_one")   # places on a skirt even where the sim moves it (`cloth`): they end on it
for _table, _gags in ((OUT, GG.OUT), (AIM, GG.AIM), (POLES, GG.POLES), (SHAPES, GG.SHAPES), (BOX, GG.BOX),
                      (STRAIGHT, GG.STRAIGHT)):
    _table.update(_gags)                    # the gag moves' places (mkmmd.core.gags) are places like these


def _v(*a):
    return np.array(a, float)


def _unit(v):
    v = np.asarray(v, float)
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-12 else v


def place(name, side, marks):
    """(point, dir, palm, shape) of a named hand place for the hand `side` ("L" / "R") as written, in the character's
    frame: `point` where the wrist goes, `dir` the way the fingers point, `palm` the way the palm faces. `resolve` makes
    it natural."""
    s = 1.0 if side == "L" else -1.0
    arm, chest, mouth, eye = (np.asarray(marks[k], float) for k in (f"arm.{side}", "chest", "mouth", "eye"))
    R = marks["reach"]

    def out(name):
        d = _unit(OUT[name][0])
        return arm + OUT[name][1] * marks["reach"] * np.array([s * d[0], d[1], d[2]])

    P = {
        "rest": (out("rest"), _v(0, 0, -1), _v(-s, 0, 0), "relaxed"),
        "dainty": (chest + marks["reach"] * _v(s * 0.186, -0.1, -0.461), _v(-s * 0.6, -0.15, -0.78), _v(0, 1, 0),
                   "relaxed"),                         # written inside: the approach rests it on the skirt or thighs
        "dainty_one": (chest + marks["reach"] * _v(s * 0.27, -0.05, -0.49), _v(-s * 0.2, -0.15, -0.97), _v(-s * 0.5, 1, 0),
                       "relaxed"),                     # alone on the skirt: on its own thigh, down along it
        "mic": (mouth + _v(s * 0.03, -0.10, -0.13), _v(-s * 0.45, -0.45, 0.77), _v(-s, 0, 0), "curled"),   # under the lips
        "mic_lens": (out("mic_lens"), _v(0, -1, 0), _v(-s, 0, 0), "curled"),
        "mic_up": (out("mic_up"), _v(0, 0, 1), _v(-s, 0, 0), "curled"),
        "mic_across": (out("mic_across"), _v(-s, -1, 0), _v(0, 0, 1), "curled"),
        "chest": (chest + _v(s * 0.08, 0.02, 0.0), _v(-s * 0.4, 0, 0.9), _v(0, 1, 0), "flat"),
        "bounce": (out("bounce"), _v(0, -1, 0), _v(0, 0, -1), "relaxed"),
        "point": (out("point"), _v(-s * 0.15, -1, 0.05), _v(-s, 0, 0), "point"),
        "peace": (eye + _v(s * 0.10, -0.12, -0.12), _v(-s * 0.2, 0, 1), _v(0, -1, 0), "peace"),
        "paw": (mouth + _v(s * 0.09, -0.12, -0.06), _v(0, -0.4, -0.9), _v(0, 1, 0), "paw"),
        "sparkle": (eye + _v(s * 0.14, -0.12, -0.12), _v(s * 0.2, -0.2, 1), _v(0, -1, 0), "sparkle"),
        "up": (out("up"), _v(s * 0.1, 0, 1), _v(0, -1, 0), "flat"),
        "drip_in": (chest + _v(s * 0.12, -0.26, 0.02), _v(-s * 0.6, -0.2, 0.7), _v(0, 1, 0), "relaxed"),
        "drip_out": (chest + _v(s * 0.12, -0.28, 0.04), _v(-s * 0.6, -0.2, 0.7), _v(0, -1, 0), "relaxed"),
        "ears": (eye + _v(s * 0.13, 0.0, -0.10), _v(-s * 0.375, 0.427, 0.823), _v(-s * 0.921, -0.276, -0.276),
                 "flat"),                             # written out to the side: it comes in onto the ear, the elbow wide
        "bunny": (eye + _v(s * 0.09, 0.01, 0.17), _v(s * 0.1, -0.2, 1), _v(0, -1, 0), "paw"),
        "bonk_up": (out("bonk_up"), _v(0, -0.5, 0.85), _v(-s, 0, 0), "curled"),
        "bonk": (out("bonk"), _v(0, -0.8, -0.5), _v(-s, 0, 0), "curled"),
        "hair": (eye + _v(s * 0.09, 0.10, 0.03), _v(-s * 0.2, 0.662, 0.722), _v(-s * 0.957, -0.289, 0),
                 "relaxed"),                          # in it, behind the ear, the fingers up and back over the head
        "hair_out": (eye + _v(s * 0.18, 0.06, 0.05), _v(s * 0.4, 0.5, 0.6), _v(0, 0.6, 0.8), "flat"),   # flicked out,
                                                      # the fingers up and back, the palm up: the upper arm keeps its
                                                      # turn out (a toss behind the head would turn it past what it can)
        "hip": (chest + marks["reach"] * _v(s * 0.40, 0.08, -0.17), _v(-s * 0.15, -0.75, -0.65), _v(-s, 0, 0), "flat"),
        "gun": (out("gun"), _v(-s * 0.12, -1, 0.12), _v(-s, 0, 0), "gun"),
        "float_low": (out("float_low"), _v(s * 0.5, -0.3, -0.8), _v(-s, 0.3, 0), "relaxed"),   # weightless: drifting
        "float": (out("float"), _v(s, -0.3, -0.12), _v(0, 0.3, -1), "relaxed"),               # out, then up and open
        "bud": (chest + R * _v(s * 0.12, -0.40, 0.05), _v(-s * 0.1, -0.6, 0.79), _v(-s, 0, 0), "bud"),   # cupped
        "bloom": (chest + R * _v(s * 0.40, -0.55, 0.35), _v(s * 0.45, -0.25, 0.86), _v(-s * 0.6, 0.3, 0.75), "open"),
        "reach": (out("reach"), _v(-s * 0.05, -0.6, 0.8), _v(0, -1, 0), "reach"),                # the palm to the lens
        "wings_up": (out("wings_up"), _v(s * 0.97, 0, -0.12), _v(0, 0.3, -1), "relaxed"),      # the hand trailing down
        "wings_down": (out("wings_down"), _v(s * 0.92, 0, 0.38), _v(0, 0.2, -1), "relaxed"),   # ... and up
        "blow": (mouth + _v(s * 0.04, -0.17, -0.07), _v(-s * 0.25, -0.9, 0.2), _v(0, 0.1, 1), "flat"),   # palm up
        "blow_out": (mouth + _v(s * 0.06, -0.3, -0.06), _v(-s * 0.1, -0.9, 0.35), _v(0, 0.3, 1), "open"),
        "giggle": (mouth + _v(s * 0.03, -0.04, -0.11), _v(-s * 0.3, -0.1, 0.95), _v(0, 1, 0), "giggle"),   # the lips
        "behind": (chest + R * _v(s * 0.07, 0.40, -0.45), _v(-s * 0.5, 0.15, -0.85), _v(-s, 0, 0), "clasp"),
        "behind_one": (chest + R * _v(s * 0.22, 0.38, -0.47), _v(-s * 0.45, 0.1, -0.89), _v(0, 1, 0), "relaxed"),
        "clasp": (chest + R * _v(s * 0.07, -0.33, 0.10), _v(-s * 0.2, -0.25, 0.95), _v(-s, 0, 0), "clasp"),
        "clasp_one": (chest + _v(s * 0.07, 0.0, 0.05), _v(-s * 0.45, 0, 0.9), _v(0, 1, 0), "relaxed"),  # on the chest
        "skirt": (out("skirt"), _v(s * 0.15, -0.2, -0.97), _v(-s, 0.1, 0), "pinch"),           # pinching the hem
    }
    if name in GESTURES:
        return gesture(name, side, marks)
    if name in GG.PLACES:
        return GG.place(name, side, marks)
    if name not in P:
        raise MoveError(f"no hand place {name!r} (have {', '.join(sorted(set(P) | set(GESTURES) | set(GG.PLACES)))})")
    return P[name]


def hand_rest(marks):
    """The member's left hand at rest in its own frame, as mkmmd.core.fingers.joints reads it: x the way its wrist bone
    points (where a place's `dir` points it), y its palm normal square to x (where `palm` turns it), z = x cross y (its
    little finger's side), metres from the wrist joint: the landmark `fingers` the build reads off the model ({points,
    palm, tips, radii}: its fingers' chains, the palm normal, where their skin ends and how thick it is; the right hand
    mirrored onto the left and the two averaged), else a typical hand (mkmmd.core.fingers.HAND) as long as `hand`."""
    f = marks.get("fingers")
    if f:
        return {"points": {k: np.asarray(v, float) for k, v in f["points"].items()}, "palm": _unit(f["palm"]),
                "tips": {k: np.asarray(v, float) for k, v in (f.get("tips") or {}).items()},
                "radii": {k: float(v) for k, v in (f.get("radii") or {}).items()}}
    pts = {k: np.asarray(v, float) * float(marks["hand"]) for k, v in FG.HAND.items()}
    n = _unit(np.cross(pts["index"][0], pts["little"][0]))
    return {"points": pts, "palm": n if n[1] > 0 else -n}


_POSED = {}


def posed(marks, shape):
    """The member's left hand (`hand_rest`) posed in the shape `shape` (SHAPES), in its own frame: mkmmd.core.fingers.pose
    ({"points": {finger: joints and tip}, "tips": {finger: where its skin ends}})."""
    hand = hand_rest(marks)
    key = (shape, tuple(round(float(v), 6) for f in sorted(hand["points"]) for v in np.ravel(hand["points"][f])),
           tuple(round(float(v), 6) for f in sorted(hand.get("tips", {})) for v in hand["tips"][f]),
           tuple(sorted((hand.get("radii") or {}).items())))
    if key not in _POSED:
        _POSED[key] = FG.pose(hand, SHAPES[shape], "L")
    return _POSED[key]


def _plane(points, toward):
    """The unit normal of the plane nearest `points` (n, 3), on the side of `toward`."""
    P = np.asarray(points, float)
    n = np.linalg.svd(P - P.mean(0))[2][2]
    return n if n @ toward > 0 else -n


def _turned(a, b, to_a, to_b):
    """The rotation that turns the hand's own direction `a` onto `to_a`, and `b` as near onto `to_b` as square to it."""
    return _basis(to_a, to_b) @ _basis(a, b).T


def _off(point, d, p, side):
    """Where a point of the left hand's own frame (`hand_rest`) is from the wrist of the hand `side` pointing along `d`,
    its palm toward `p` (the right hand is the left's mirror image: its z, the little finger's side, is turned over)."""
    x, y, z = np.asarray(point, float)
    return x * d + y * p + (1.0 if side == "L" else -1.0) * z * np.cross(d, p)


def _crossing(po):
    """(where, the way it faces, the way its tips point) of a finger heart (`posed` "heart"): the middle of the closest
    points of the thumb's and the index's last segments, ending where their skin does; the normal of their plane on the
    thumb's side; the way halfway between the two."""
    T, I, tt, it = po["points"]["thumb"][-2], po["points"]["index"][-2], po["tips"]["thumb"], po["tips"]["index"]
    u = np.linspace(0.0, 1.0, 41)
    A, B = T + np.outer(u, tt - T), I + np.outer(u, it - I)
    d = np.linalg.norm(A[:, None] - B[None], axis=2)
    i, j = np.unravel_index(int(np.argmin(d)), d.shape)
    face = _unit(np.cross(tt - T, it - I))
    return 0.5 * (A[i] + B[j]), (face if face @ (A[i] - B[j]) > 0 else -face), _unit(_unit(tt - T) + _unit(it - I))


def gesture(name, side, marks):
    """(point, dir, palm, shape) of a gesture (GESTURES) for the hand `side`, from where the member's own hand has the
    part of it that must read in its shape (`posed`; the left hand's, mirrored for the right):
    - `peek` ("ring"): the ring of the thumb and index, its opening (the normal of the loop through the index's joints,
      the two fingertips and the thumb's joints) to the lens with the thumb toward the face (the side a forearm coming
      in from the side can turn to: thumb to the lens it would have to turn over past what a forearm turns), its hole
      (`hole`) PEEK[0] in front of the hand's own pupil (`pupil_at`), the hand's own direction (the fanned fingers) up,
      PEEK[1] degrees out from upright (in, over the brow);
    - `heart` (the finger heart): the crossing of the thumb and index (`_crossing`) at HEART from the mouth, its tips
      straight up and its face (the thumb's side) to the lens;
    - `heart_push` (half of a two-hand heart, "heart2"): the line from the thumb's tip to the index's upright and on the
      body's midline (each tip half a finger's radius off it: the two hands' tips touch), the half heart's face (its little
      finger's side: a hand whose fingers curl over to the middle shows that side) to the lens, its top at PUSH;
    - `butterfly` ("wings"), crossed at the wrists: the hand on the other side, the back of the hand to the lens, the
      fingers leaning out BUTTERFLY["wing"] degrees from upright, the thumb's tip BUTTERFLY["hook"] past the midline at
      BUTTERFLY's place before the collarbone, the right hand CROSS["butterfly"] in front of the left; `butterfly_shut`
      is the same hand turned BUTTERFLY["shut"] degrees forward about the thumbs (the wings folded)."""
    s = 1.0 if side == "L" else -1.0
    M = _v(s, 1.0, 1.0)                                      # the left hand's world vectors -> this side's
    chest, mouth = (np.asarray(marks[k], float) for k in ("chest", "mouth"))
    shape = GESTURES[name]
    po = posed(marks, shape)
    radial = _v(0.0, 0.0, -1.0)                              # the left hand's thumb side, in its own frame
    if name == "peek":
        I, T = po["points"]["index"], po["points"]["thumb"]
        loop = np.array([I[0], I[1], I[2], po["tips"]["index"], po["tips"]["thumb"], T[2], T[1]])
        part = hole(po)
        t = math.radians(PEEK[1])
        R = _turned(_plane(loop, radial), _v(1.0, 0.0, 0.0), _v(0.0, 1.0, 0.0), _v(math.sin(t), 0.0, math.cos(t)))
        at = pupil_at(side, marks) + _v(0.0, -PEEK[0], 0.0)
    elif name == "heart":
        part, face, up = _crossing(po)
        t = math.radians(HEART[3])
        R = _turned(up, face, _v(math.sin(t), 0.0, math.cos(t)), _v(0.0, 1.0, 0.0))
        at = mouth + _v(s * HEART[0], HEART[1], HEART[2])
    elif name == "heart_push":
        I, T = po["points"]["index"], po["points"]["thumb"]
        part, low = po["tips"]["index"], po["tips"]["thumb"]
        half = np.array([*I[:-1], part, low, *T[1:-1]])
        R = _turned(part - low, _plane(half, -radial), _v(0.0, 0.0, 1.0), _v(0.0, -1.0, 0.0))
        md = hand_rest(marks)["points"]["middle"]
        r = (hand_rest(marks).get("radii") or {}).get("index") or FG.RADIUS * float(np.linalg.norm(np.diff(md, axis=0),
                                                                                                  axis=1).sum())
        at = _v(s * 0.5 * r, chest[1] + PUSH[0] * marks["reach"], chest[2] + PUSH[1] * marks["reach"])
    else:
        part = po["tips"]["thumb"]
        w = math.radians(BUTTERFLY["wing"])
        R = FG._rot(_v(1.0, 0.0, 0.0), BUTTERFLY["lean"]) @ _turned(
            _v(1.0, 0.0, 0.0), _v(0.0, 1.0, 0.0), _v(-math.sin(w), 0.0, math.cos(w)), _v(0.0, 1.0, 0.0))
        if name == "butterfly_shut":                         # folded forward about the thumb (its base to its tip)
            hinge = _unit(R @ (part - po["points"]["thumb"][0]))
            R = FG._rot(hinge, BUTTERFLY["shut"] * (1.0 if hinge[0] > 0 else -1.0)) @ R
        arms = 0.5 * (np.asarray(marks["arm.L"], float) + np.asarray(marks["arm.R"], float))
        at = _v(s * BUTTERFLY["hook"], chest[1] + BUTTERFLY["behind"] * marks["reach"] + s * 0.5 * CROSS["butterfly"],
                arms[2] - BUTTERFLY["below"] * marks["reach"])
    d, p = M * R[:, 0], M * R[:, 1]
    return at - _off(part, d, p, side), d, p, shape


def hole(po):
    """The middle of the hole of a ring hand (`posed` "ring") in the left hand's own frame: the centre of the circle
    through the index's curl and the thumb's pad (its last two joints and tip, the thumb's tip and two joints), fitted
    in their plane (not the joints' mean, which the knuckles pull out of the hole)."""
    I, T = po["points"]["index"], po["points"]["thumb"]
    P = np.array([I[1], I[2], po["tips"]["index"], po["tips"]["thumb"], T[2], T[1]], float)
    c = P.mean(0)
    _, _, Vt = np.linalg.svd(P - c)
    xy = (P - c) @ Vt[:2].T
    A = np.c_[xy, np.ones(len(xy))]
    a, b, _ = np.linalg.lstsq(A, -(xy ** 2).sum(1), rcond=None)[0]
    return c + Vt[:2].T @ np.array([-0.5 * a, -0.5 * b])


def sighted(name, side, marks, point, d, palm):
    """Where the part of a place seen through (SIGHTED) is in the character's frame for the hand `side` with its wrist at
    `point`, pointing along `d`, its palm toward `palm`: the ring's hole (`hole`)."""
    return np.asarray(point, float) + _off(hole(posed(marks, GESTURES[name])), d, palm, side)


def pupil_at(side, marks):
    """Where the visible pupil of the hand `side`'s eye is in the character's frame at rest: the landmark `pupil` (the
    eye's own mesh, a few centimetres in front of the eye bone and beside it), else the eye bone `eye_dx` to its side."""
    s = 1.0 if side == "L" else -1.0
    p = marks.get("pupil")
    if p is None:
        return np.asarray(marks["eye"], float) + _v(s * float(marks.get("eye_dx", EYE_DX * marks["hand"])), 0.0, 0.0)
    return _v(s * p[0], p[1], p[2])


def pole_point(side, marks, o=None):
    """The point (in the character's frame) of an elbow's pole `o`, (out, behind, below) from the arm joint of the hand
    `side` as AR.POLE, written for an arm of REACH and scaled to the member's (None: AR.POLE itself, unscaled: the pose
    stage's own pole)."""
    k = 1.0 if o is None else float(marks["reach"]) / REACH
    o = AR.POLE if o is None else o
    return np.asarray(marks[f"arm.{side}"], float) + k * _v((1.0 if side == "L" else -1.0) * o[0], o[1], -o[2])


def elbow(side, marks, wrist, pole=None):
    """Where an arm IK puts the elbow of the hand `side` for its wrist at `wrist`: with its pole at the point `pole` (None:
    the one the pose stage gives it by default, `pole_point`)."""
    S = np.asarray(marks[f"arm.{side}"], float)
    P = pole_point(side, marks) if pole is None else np.asarray(pole, float)
    return AR.pole_elbow(S, np.asarray(wrist, float), marks["upper"], marks["fore"], P)


def forearm(side, marks, wrist, pole=None):
    """The forearm's direction (elbow -> wrist, unit) an arm IK gives the hand `side` for its wrist at `wrist` (`elbow`)."""
    return _unit(np.asarray(wrist, float) - elbow(side, marks, wrist, pole))


def elbow_pole(name, side, marks):
    """The point (in the character's frame) the elbow's pole goes to at a place that holds the elbow its own way (POLES),
    None at the others (the pose stage's own pole)."""
    if not isinstance(name, str) or name not in POLES:
        return None
    return pole_point(side, marks, POLES[name])


def arm_rest(marks):
    """The member's left arm at rest, as mkmmd.core.armtwist.twists reads it: {upper, fore, dir, palm}, the upper arm's
    and the forearm's directions, the way the hand points and its palm faces, in its own frame (the landmark `arm_rest`
    the build reads off the model, the right arm mirrored onto the left and the two averaged; else AT.REST)."""
    return marks.get("arm_rest") or AT.REST


def twisted(side, marks, wrist, d, p, pole=None):
    """(forearm, upper arm, elevation) degrees of the arm an IK gives the hand `side` with its wrist at `wrist`, the hand
    pointing along `d` and its palm toward `p`, and its elbow's pole at the point `pole` (None: the pose stage's own):
    its two twists (mkmmd.core.armtwist: + supination, external rotation) and how far its upper arm is raised."""
    S = np.asarray(marks[f"arm.{side}"], float)
    P = pole_point(side, marks) if pole is None else np.asarray(pole, float)
    E = AR.pole_elbow(S, np.asarray(wrist, float), marks["upper"], marks["fore"], P)
    return AT.twists(arm_rest(marks), side, S, E, wrist, d, p, P) + (AT.elevation(E - S),)


def overtwist(tw, margin=TWIST_MARGIN):
    """Degrees an arm's twists lie outside the human range narrowed by `margin`, summed: `tw` (forearm, upper arm,
    elevation) as `twisted` gives them, against AT.FOREARM and AT.upper_limits."""
    return AT.excess(tw[0], AT.FOREARM, margin) + AT.excess(tw[1], AT.upper_limits(tw[2]), margin)


def arm_for(side, marks, wrist, hand, pole=None, limit=HELD, written=None, toward=None):
    """(dir, palm, forearm, pole) of the hand `side` with its wrist at `wrist`: `hand(f)` -> (dir, palm) the hand takes on
    the forearm f (`natural`, `aimed` or as written), with its elbow's pole at `pole` (a point; None: the pose stage's
    own) when the arm's twists then stay TWIST_MARGIN inside the human range (`twisted`, `overtwist`), else with the
    elbow turned round the shoulder-wrist line to where they do, the least way from where `pole` puts it, rather than the
    forearm twisting past what it can: up and out for a hand turned over palm out (a ring at the eye, side-on to the
    lens), down and in under it for one turned palm up (a palm held up before the mouth). Among such elbows one whose
    forearm the hand `written` (its direction as the place has it) bends no more than `limit` from is preferred, so the
    hand keeps its orientation (a gesture still reads); the elbow rises at most ELBOW_RISE of `reach` above the shoulder
    (a person lifts it to about the shoulder's height), or as high as the wrist for a hand above that (a hand in the
    hair), stays on its own side of the body's middle and out of the torso (TORSO). With `toward` (forearm, upper arm
    degrees, either None) the elbow whose twists come nearest those is preferred before all but the range, the given
    pole included (a key on a trip, turning between its ends' twists through the middle). The pole returned puts the
    IK's elbow there (POLE_OFF beyond it); when no elbow keeps the twists in range the closest one wins."""
    S, W = np.asarray(marks[f"arm.{side}"], float), np.asarray(wrist, float)
    a, b = marks["upper"], marks["fore"]
    P0 = pole_point(side, marks) if pole is None else np.asarray(pole, float)
    E0 = AR.pole_elbow(S, W, a, b, P0)
    rest = arm_rest(marks)
    u = _unit(W - S)
    v0 = _unit((P0 - S) - ((P0 - S) @ u) * u)

    def judge(E, P, v):
        f = _unit(W - E)
        d, p = hand(f)
        tw = AT.twists(rest, side, S, E, W, d, p, P) + (AT.elevation(E - S),)
        over = max(bend(d if written is None else written, f) - limit, 0.0)
        near = 0.0 if toward is None else sum(abs(tw[i] - t) for i, t in enumerate(toward) if t is not None)
        return (round(overtwist(tw) + (10.0 * over if written is None else 0.0), 1), round(near / 10.0),
                math.ceil(over / 5.0), math.degrees(math.acos(float(np.clip(v @ v0, -1.0, 1.0))))), (d, p, f)

    key, (d, p, f) = judge(E0, P0, v0)
    if key[0] <= 0.0 and toward is None:
        return d, p, f, pole
    best = (key, (d, p, f, pole))
    sx, chest = abs(float(S[0])), np.asarray(marks["chest"], float)
    top = max(float(S[2]) + ELBOW_RISE * marks["reach"], float(W[2]), float(E0[2]))
    for E, v in zip(*AR.circle(S, W, a, b)):
        if E[2] > top + 1e-9 or E[0] * S[0] < 0.0 or (
                (E[0] / (TORSO[0] * sx)) ** 2 + ((E[1] - chest[1]) / (TORSO[1] * sx)) ** 2 < 1.0):
            continue                                        # above the shoulder, past the body's middle, in the torso
        P = E + POLE_OFF * v
        key, (d, p, f) = judge(E, P, v)
        if key < best[0]:
            best = (key, (d, p, f, P))
    return best[1]


def _turn(v, axis, ang):
    """v turned by `ang` radians about the unit `axis` (right hand)."""
    return v * math.cos(ang) + np.cross(axis, v) * math.sin(ang) + axis * float(axis @ v) * (1.0 - math.cos(ang))


def bend(d, f):
    """Degrees between a hand's direction and its forearm's."""
    return float(np.degrees(np.arccos(np.clip(_unit(d) @ _unit(f), -1.0, 1.0))))


def natural(d, p, f, limit=HELD):
    """(dir, palm): the hand (pointing along `d`, the palm toward `p`) turned toward its forearm `f` as a whole until it
    bends at most `limit` degrees from it."""
    d = _unit(d)
    p = _unit(np.asarray(p, float) - (np.asarray(p, float) @ d) * d)
    th = bend(d, f)
    if th <= limit:
        return d, p
    axis = np.cross(d, _unit(f))
    if np.linalg.norm(axis) < 1e-9:                     # straight back along the forearm: turn it about the palm's side
        axis = np.cross(d, p)
    axis = _unit(axis)
    ang = math.radians(th - limit)
    return _unit(_turn(d, axis, ang)), _unit(_turn(p, axis, ang))


def aimed(side, f, aim):
    """(dir, palm) of a fist round a mic pointing along `aim` (the grille's way) whose hand continues the forearm `f` as
    far as the grip allows: the mic crosses the palm on the thumb side (as at the mic place)."""
    m = _unit(aim)
    d = np.asarray(f, float) - (np.asarray(f, float) @ m) * m
    if np.linalg.norm(d) < 0.15:                         # the mic along the forearm: the hand points the way it can
        d = np.cross(m, _v(1, 0, 0)) if abs(m[0]) < 0.9 else np.cross(m, _v(0, 1, 0))
    d = _unit(d)
    return d, (1.0 if side == "R" else -1.0) * _unit(np.cross(m, d))


def hand_box(d, p, length, shape):
    """(18, 3) points of the hand relative to its wrist: a box along `d` as far as the shape reaches, across the palm and
    through it (along `p`)."""
    reach, half = BOX.get(shape if isinstance(shape, str) else "", (0.75, 0.022))
    d, p = _unit(d), _unit(p)
    r = _unit(np.cross(d, p))
    return np.array([t * reach * length * d + a * WIDTH * length * r + b * half * p
                     for t in (0.25, 0.6, 1.0) for a in (-1.0, 0.0, 1.0) for b in (-1.0, 1.0)])


def sweep(path, hand, clear, margin, far, step):
    """The approach's parameter where the hand stops: the hand (`hand(wrist)`: its points (n, 3) with its wrist there, as
    it turns on the way) comes in along `path(s)` (the wrist at s; s = 0 is the place asked for) from s = `far` and
    stops where `clear(points)` (the smallest distance from points to the body) first falls to `margin`: 0 when it never
    does on the way, `far` when it touches even there. Coming from outside, it never ends inside the body."""
    def touches(s):
        return clear(hand(path(s))) <= margin

    if touches(far):
        return far
    s = far
    while s > 0.0:
        nxt = max(s - step, 0.0)
        if touches(nxt):
            lo, hi = nxt, s
            for _ in range(12):
                mid = 0.5 * (lo + hi)
                lo, hi = (mid, hi) if touches(mid) else (lo, mid)
            return hi
        s = nxt
    return 0.0


def line(point, approach):
    """A straight approach to `point` from along `approach` (s in metres)."""
    a, point = _unit(approach), np.asarray(point, float)
    return lambda s: point + a * s


def swing(point, shoulder, out):
    """An approach to `point` swinging the whole arm about `shoulder` from the side `out` (s in radians): the hand stays
    as far from the shoulder as at the place, so it stays in reach."""
    S = np.asarray(shoulder, float)
    v = np.asarray(point, float) - S
    axis = np.cross(v, _unit(out))
    if np.linalg.norm(axis) < 1e-9:
        axis = np.cross(v, _v(0, -1, 0))
    axis = _unit(axis)
    return lambda s: S + _turn(v, axis, s)


def resolve(name, side, marks, clear=None, cloth=None):
    """The place `name` for the hand `side` made natural (module docstring): (point, dir, palm, shape, moved, bend, pole):
    `moved` the metres the approach kept it out by, `bend` its wrist's degrees from the forearm, `pole` the point (in the
    character's frame) the elbow's pole goes to there, None for the pose stage's own: a place's own (POLES), or one
    turned to keep the arm's twists in range (`arm_for`). A place on a skirt (CLOTH) is kept off `cloth` instead of
    `clear` when it is given: the body and clothes with the skirt the sim moves. A place seen through (SIGHTED) is moved
    back square to the line of sight, after the wrist was turned toward its forearm and the approach kept it out, until
    its part (`sighted`) is on the line from the lens to the pupil (`pupil_at`) with the body upright."""
    pt, d, p, shape = place(name, side, marks)
    s = 1.0 if side == "L" else -1.0
    if name in CLOTH and cloth is not None:
        clear = cloth

    def hand(f):
        if name in AIM:
            a = AIM[name]
            return aimed(side, f, (s * a[0], a[1], a[2]))
        if name in SHAPED:
            dd = _unit(d)
            return dd, _unit(np.asarray(p, float) - (np.asarray(p, float) @ dd) * dd)
        return natural(d, p, f, STRAIGHT.get(name, HELD))

    written = None if name in AIM else d
    limit = STRAIGHT.get(name, HELD)
    dd, pp, f, pole = arm_for(side, marks, pt, hand, elbow_pole(name, side, marks), limit, written)
    moved = 0.0
    if clear is not None:
        out = _v(pt[0], pt[1] - float(marks["chest"][1]), 0.0)
        out = out if np.linalg.norm(out) > 0.03 else _v(0, -1, 0)
        if name in TOUCH:
            t = TOUCH[name]
            path, margin, far, step = line(pt, (s * t[0], t[1], t[2])), TOUCH_MARGIN, FAR, STEP
        elif name in OUT:                                # the arm out: it swings up and away round the shoulder
            path, margin, far, step = (swing(pt, marks[f"arm.{side}"], out), TOUCH_MARGIN if name in CLOTH else MARGIN,
                                       SWING, SWING_STEP)
        elif name in GESTURES:                           # straight from the front: the gesture stays on its line of sight
            path, margin, far, step = line(pt, _v(0, -1, 0)), MARGIN, FAR, STEP
        else:
            path, margin, far, step = line(pt, out), MARGIN, FAR, STEP
        k = sweep(path, lambda w: w + hand_box(*hand(forearm(side, marks, w, pole)), marks["hand"], shape), clear,
                  margin, far, step)                     # on its way in the elbow's pole is the one at the place
        if k:
            new = path(k)
            moved = float(np.linalg.norm(new - pt))
            pt = new
            f = forearm(side, marks, pt, pole)               # the elbow it came in with (its twists a few degrees off)
            dd, pp = hand(f)
    if name in SIGHTED:                                  # its part back on the line of sight, as seen from the front
        goal = pupil_at(side, marks)
        for _ in range(SIGHT_ROUNDS):
            miss = (goal - sighted(name, side, marks, pt, dd, pp)) * _v(1.0, 0.0, 1.0)
            if float(np.linalg.norm(miss)) < SIGHT_TOL:
                break
            pt = pt + miss
            f = forearm(side, marks, pt, pole)
            dd, pp = hand(f)
    return pt, dd, pp, shape, moved, bend(dd, f), pole


# hand: "places" ((place, share of dur), ...) or "beats" (`place` + `on` on each beat inside it, or on the entry's `hits`,
# `off_place` (default `place`) + `off` halfway to the next; offsets in the character's frame; a move with an off place of
# its own starts there: its wind-up; every: on every n-th of those beats, not on the entry's own hits); both: both hands
# play it (mic_ok: the mic hand too; two_hands: it needs both free);
# lean / tilt / turn / head_tilt / head_turn: degrees held over the move (BODY); twitch: a head flick on its start; face:
# (semantic morph, value) held over it; wink: the eye on the other side from the hand winks; mirror: the turns, tilts and
# the twitch are written for the left hand, and go the other way for the right one (a pose done with either hand); free:
# the hand is let go to whatever else drives the arm for the move's span (`ik` keys, see the module docstring)
MOVES = {
    "bounce_hand": {"hand": "beats", "place": "bounce", "on": (0, 0, -0.05), "off": (0, 0, 0)},
    "chest_pat": {"hand": "beats", "place": "chest", "on": (0, 0, 0), "off": (0, -0.035, 0.01), "face": [("blush", 0.6)]},
    "paws": {"hand": "beats", "place": "paw", "on": (0, 0, -0.035), "off": (0, 0, 0), "both": True, "face": [("omega", 0.8)]},
    "point": {"hand": "places", "places": (("point", 0.0),)},
    "heart_wink": {"hand": "places", "places": (("heart", 0.0),), "face": [("wink_r", 1.0), ("mouth_smile", 0.6)]},
    "peace_eye": {"hand": "places", "places": (("peace", 0.0),), "face": [("wink_l", 1.0)]},
    "sparkle": {"hand": "places", "places": (("sparkle", 0.0),), "both": True, "face": [("smile_eyes", 0.7)]},
    "drip_check": {"hand": "places", "places": (("drip_in", 0.0), ("drip_out", 0.5)), "face": [("jito", 0.5)]},
    "hands_up": {"hand": "places", "places": (("up", 0.0),), "both": True, "mic_ok": True},
    "flung": {"hand": "places", "places": (("up", 0.0),), "both": True, "mic_ok": True, "react": True},
    "mic_lens": {"hand": "places", "places": (("mic_lens", 0.0),)},
    "mic_up": {"hand": "places", "places": (("mic_up", 0.0),)},
    "mic_across": {"hand": "places", "places": (("mic_across", 0.0),)},
    "ears": {"hand": "places", "places": (("ears", 0.0),), "both": True, "face": [("hau", 1.0)]},
    "heart_push": {"hand": "places", "places": (("heart_push", 0.0),), "both": True, "two_hands": True,
                   "face": [("cheerful", 0.8)]},
    "bunny_paws": {"hand": "places", "places": (("bunny", 0.0),), "both": True, "face": [("wink_r", 1.0)]},
    "bonk": {"hand": "beats", "place": "bonk", "off_place": "bonk_up", "on": (0, 0, 0), "off": (0, 0, 0)},
    "into_lens": {"lean": 14.0},
    "lean_back": {"lean": -11.0, "tilt": 7.0},
    "stank": {"twitch": {"bones": ["head"], "deg": -12.0, "axis": [1, 0, 0], "dur": 0.3}, "face": [("hau", 1.0)]},
    "blown": {"lean": -16.0, "twitch": {"bones": ["head"], "deg": -14.0, "axis": [1, 0, 0], "dur": 0.4},
              "face": [("surprised", 1.0)]},
    "hair_flip": {"hand": "places", "places": (("hair", 0.0), ("hair_out", 0.4)), "head_tilt": -10.0, "mirror": True,
                  "twitch": {"bones": ["head"], "deg": 12.0, "axis": [0, 1, 0], "dur": 0.4},
                  "face": [("calm", 0.6), ("mouth_smile", 0.3)]},
    "over_shoulder": {"turn": 25.0, "head_turn": 45.0, "head_tilt": 6.0, "mirror": True,
                      "face": [("calm", 0.4), ("mouth_smile", 0.35)]},
    "hand_hip": {"hand": "places", "places": (("hip", 0.0),), "head_tilt": 12.0, "tilt": -5.0, "mirror": True,
                 "face": [("smile_eyes", 0.35), ("mouth_smile", 0.5)]},
    "finger_gun": {"hand": "places", "places": (("gun", 0.0),), "lean": 6.0, "head_tilt": 8.0, "mirror": True,
                   "wink": True, "face": [("mouth_smile", 0.6)]},
    "free": {"hand": "free"},
    "float": {"hand": "places", "places": (("float_low", 0.0), ("float", 0.75))},
    "peek": {"hand": "places", "places": (("peek", 0.0),), "lean": 4.0, "head_tilt": 5.0, "mirror": True, "wink": True,
             "face": [("mouth_smile", 0.5)]},
    "butterfly": {"hand": "beats", "place": "butterfly_shut", "off_place": "butterfly", "on": (0, 0, 0), "off": (0, 0, 0),
                  "both": True, "two_hands": True, "face": [("smile_eyes", 0.6), ("mouth_smile", 0.5)]},
    "bloom": {"hand": "places", "places": (("bud", 0.0), ("bloom", 0.4)), "both": True, "two_hands": True,
              "face": [("smile_eyes", 0.6), ("mouth_smile", 0.5)]},
    "reach": {"hand": "places", "places": (("reach", 0.0),), "lean": 8.0, "head_tilt": 6.0, "mirror": True,
              "face": [("calm", 0.35), ("mouth_smile", 0.35)]},
    "wings": {"hand": "beats", "place": "wings_down", "off_place": "wings_up", "on": (0, 0, 0), "off": (0, 0, 0),
              "both": True, "every": 2, "face": [("calm", 0.5)]},
    "blow": {"hand": "places", "places": (("blow", 0.0), ("blow_out", 0.5)), "face": [("u", 0.7), ("calm", 0.4)]},
    "giggle": {"hand": "places", "places": (("giggle", 0.0),), "head_tilt": 6.0, "mirror": True,
               "face": [("smile_eyes", 0.8), ("blush", 0.5)]},
    "face": {},
}
MOVES.update(GG.MOVES)                                          # the gag moves (mkmmd.core.gags), played by `compile` too
BODY = ("lean", "tilt", "turn", "head_tilt", "head_turn")       # perform keys a move holds, degrees
MIRRORED = ("tilt", "turn", "head_tilt", "head_turn")           # ... those that a mirrored move turns the other way
KEYS = {"name", "t", "dur", "hand", "morph", "value", "hits", "prop", "chore", "accents"}
REST_PLACES = ("rest", "dainty", "mic", "behind", "clasp", "skirt")
LONE = {"dainty": "dainty_one", "behind": "behind_one", "clasp": "clasp_one"}   # a pose for two: a hand there alone
STRIP = "strip"                  # the place of a hand on a strip: (STRIP, i) is its i-th sample
WINKS = {"L": "wink_l", "R": "wink_r"}   # the semantic wink that shuts the left eye, and the right (a model may swap them)
SIDES = ("L", "R")


def compile(entries, marks, beats=(), yaw=0.0, clear=None, strip=None, cloth=None, winks=None, downbeats=(), props=None):
    """The keys of one cast member's moves: {"hands": {side: [key]}, "ik": {side: [[t, share]]} (only the hands a `free`
    move lets go), "lean", "tilt", "turn", "head_tilt", "head_turn": [[t, deg]] (BODY), "twitch": [...], "expressions":
    [{morph, keys}], "places": {side: {place: {moved_mm, wrist_deg, twist_deg}}}} (`twist_deg`: the forearm's and the
    upper arm's twists there, `twisted`). `entries` are its `[[move]]` tables {name, t, dur = 1, hand = "R", morph,
    value, hits}, and `{name = "rest", hand, place, t}` where that hand waits from clip time t on (REST_PLACES; default
    "rest"; without `t` from the start); `marks` its landmarks (MARKS); `beats` clip seconds (a beat move pats on each
    inside it, or on its entry's `hits`: clip seconds inside it); `yaw` the member's (degrees: hand directions are turned
    into the world);
    `clear(points)` the smallest distance from points (n, 3) in the character's frame to what a hand stays off, its body
    and clothes (None: hands are not kept off anything), and `cloth(points)` the same with a skirt the sim moves (CLOTH
    places rest on it; None: on what `clear` measures). `strip` {side: {"t", "at", "dir", "palm"}}: where a motion strip
    has that hand at clip seconds `t` over the spans its `free` moves let it go (the wrist joint, the way the hand points
    and the way its palm faces, in the character's frame): its goal follows them there (module docstring). A key on the
    strip has `"strip": True` and no fingers (the strip's own play). `winks` {"L": morph, "R": morph}: the semantic wink
    that shuts the member's left eye and its right (WINKS when None; models name them from either side).
    A gag move (mkmmd.core.gags) also gives "perform": the keys of its legs and its mouth ({"bounce": {beats, depth,
    attack, decay}, "rise", "crouch": [[t, m]], "air": [span], "walk": {steps, ...}, "mouth": [[t0, t1]]}), "held":
    {prop: {"bone": "wrist" | "chest", "side", "rel" | "pose"}} (the props it holds: where each rides the hand that holds
    it, (rotation, offset) in the hand's frame, columns its direction, its palm and their cross, or the hugged one's pose
    in the character's frame) and "props": {prop: {root property: value}} (a carrot's `bite<k>` clip seconds), with
    `downbeats` (clip seconds: a bounce's accents by default) and `props` {name: {"use", "bounds", "scale"}}: the cards of
    the props its moves hold."""
    rests, told = _rests(entries)                                     # told: hands given a rest place of their own
    mic = {sd: rests[sd][0][1] == "mic" for sd in SIDES}              # a hand at the mic holds it all clip long
    homes = _homes(rests)                                             # where each hand waits, a pair's lone places too
    entries = [e for e in entries if not (isinstance(e, dict) and e.get("name") == "rest")]
    missing = [k for k in MARKS if k not in marks]
    if missing:
        raise MoveError(f"landmarks missing: {missing}")
    c, s = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
    world = lambda d: [round(c * d[0] - s * d[1], 4), round(s * d[0] + c * d[1], 4), round(float(d[2]), 4)]   # noqa: E731
    plays = {"L": [], "R": []}                                       # (t0, t1, [(t, place, offset)])
    frees = {"L": [], "R": []}                                       # (t0, t1, joined) of the spans a `free` move lets go
    body, faces = [], {}
    out = {"hands": {"L": [], "R": []}, "ik": {}, **{k: [] for k in BODY}, "twitch": [], "expressions": [],
           "places": {"L": {}, "R": {}}, "perform": {}, "held": {}, "props": {}}
    bites = GG.bite_numbers(entries, beats)                          # each prop's bites, numbered on across the chomps
    grips, fixed, legs, curves, mouths = {}, {"L": {}, "R": {}}, {}, {}, []   # a gag's grips, prop places, legs, faces
    for i, e in enumerate(entries):
        what = f"move {i} ({e.get('name')!r})" if isinstance(e, dict) else f"move {i}"
        if not isinstance(e, dict) or set(e) - KEYS or "name" not in e or "t" not in e:
            raise MoveError(f"{what}: expected {{name, t, dur, hand, morph, value, hits, prop, chore, accents}}")
        if e["name"] not in MOVES:
            raise MoveError(f"{what}: no such move (have {', '.join(sorted(MOVES))})")
        m, t0, dur = MOVES[e["name"]], float(e["t"]), float(e.get("dur", 1.0))
        hand = e.get("hand", "R")
        if dur <= 0 or hand not in ("L", "R"):
            raise MoveError(f"{what}: dur must be positive and hand \"L\" or \"R\"")
        t1 = t0 + dur
        sides = ("L", "R") if m.get("both") else (hand,)
        if m.get("both"):
            held = [sd for sd in sides if mic[sd]]
            if held and m.get("two_hands"):
                raise MoveError(f"{what}: needs both hands free, and the {held[0]} hand holds the mic")
            if not m.get("mic_ok"):
                sides = tuple(sd for sd in sides if not mic[sd])              # the mic hand keeps the mic at the mouth
        if "hits" in e and m.get("hand") != "beats" and not m.get("hits"):
            raise MoveError(f"{what}: hits are for a move played on beats "
                            f"({', '.join(k for k, v in MOVES.items() if v.get('hand') == 'beats' or v.get('hits'))})")
        for k in ("prop", "chore", "accents"):
            if k in e and m.get("hand") != "gag":
                raise MoveError(f"{what}: `{k}` is for a gag move ({', '.join(sorted(GG.MOVES))})")
        react = bool(m.get("react"))
        if m.get("hand") == "free":
            if mic[hand]:
                raise MoveError(f"{what}: the {hand} hand holds the mic and is never let go")
            on = _strip_seq((strip or {}).get(hand), t0, t1)                 # on the strip, or its goal waits at rest
            plays[hand].append((t0, t1, on or [(t0, _home_at(homes[hand], t0), (0, 0, 0))], False))
            frees[hand].append((t0, t1, on[0][0] if on else t0))             # ... let go from where it joins
        elif m.get("hand") == "places":
            for side in sides:
                plays[side].append((t0, t1, [(t0 + share * dur, p, (0, 0, 0)) for p, share in m["places"]], react))
        elif m.get("hand") == "beats":
            if "hits" in e:
                hs = e["hits"]
                if not isinstance(hs, list) or not hs or not all(isinstance(h, (int, float)) for h in hs):
                    raise MoveError(f"{what}: hits is a list of clip seconds")
                bs = sorted(float(h) for h in hs)
                if bs[0] < t0 - 1e-6 or bs[-1] >= t1 - 1e-6:
                    raise MoveError(f"{what}: its hits must fall inside it ({t0} .. {t1} s)")
            else:
                bs = [b for b in beats if t0 - 1e-6 <= b < t1 - 1e-6][::int(m.get("every", 1))]
                if not bs:
                    raise MoveError(f"{what}: no beat between {t0} and {t1} s (the member's timeline gives the beats)")
            on, off = m["place"], m.get("off_place", m["place"])
            for side in sides:
                seq = [(t0, off, m["off"])] if off != on and bs[0] - t0 > 1e-6 else []     # the wind-up
                for b, nb in zip(bs, bs[1:] + [t1]):
                    seq += [(b, on, m["on"]), (0.5 * (b + nb), off, m["off"])]
                plays[side].append((t0, t1, seq, react))
        elif m.get("hand") == "gag":
            try:
                got = GG.plan(e, m, t0, t1, beats, downbeats, hand, mic, marks, props or {}, clear, bites, grips)
            except GG.GagError as ex:
                raise MoveError(f"{what}: {ex}") from None
            for side, seq in got["plays"].items():
                if seq:
                    plays[side].append((t0, t1, seq, False))
                fixed[side].update(got["places"][side])
            for key, v in got["perform"].items():
                legs.setdefault(key, []).append(v)
            for morph, keys in got["curves"].items():
                curves.setdefault(morph, []).extend(keys)
            mouths += got["mouth"]
            for prop, vals in got["values"].items():
                out["props"].setdefault(prop, {}).update(vals)
        flip = -1.0 if m.get("mirror") and hand == "R" else 1.0
        if any(k in m for k in BODY):
            body.append((t0, t1, {k: float(m.get(k, 0.0)) * (flip if k in MIRRORED else 1.0) for k in BODY}, what))
        if "twitch" in m:
            out["twitch"].append(dict(m["twitch"], t=t0, deg=float(m["twitch"]["deg"]) * flip))
        if e["name"] == "face" and "morph" not in e:
            raise MoveError(f"{what}: a face needs `morph` (a morph's semantic name)")
        wink = [((winks or WINKS)["R" if hand == "L" else "L"], 1.0)] if m.get("wink") else []   # the other eye
        for morph, value in (m.get("face", []) + wink
                             + ([(str(e["morph"]), float(e.get("value", 1.0)))] if "morph" in e else [])):
            faces.setdefault(morph, []).append([t0, t1, value])
    made = {sd: dict(fixed[sd]) for sd in SIDES}                     # a gag's prop places, solved with their grips

    def at(side, name):
        if isinstance(name, tuple):                                           # (STRIP, i): where the strip has the hand
            s, i = strip[side], name[1]
            return np.asarray(s["at"][i], float), _unit(s["dir"][i]), _unit(s["palm"][i]), None, 0.0, 0.0, None
        if name not in made[side]:                                            # resolved, with its elbow's pole if it has one
            made[side][name] = resolve(name, side, marks, clear, cloth)
        return made[side][name]

    moved = {sd: list(plays[sd]) for sd in SIDES}
    for a, b, two in _paired(homes):                                          # a pose for two: alone, its lone place
        for sd, other in (("L", "R"), ("R", "L")):
            after = None if b is None else _home_at(homes[sd], b)
            plays[sd] += _alone(moved[sd], moved[other], sd, at, two, (a, b), after)
    for side in SIDES:
        left = {}                                                             # a free move's start: when it leaves the strip
        out["hands"][side] = _hand(sorted(plays[side], key=lambda p: p[0]), side, homes[side], world,
                                   lambda name, side=side: at(side, name), side in told,
                                   mid=_midway(side, marks), left=left, mic=mic[side])
        out["places"][side] = {n: {"moved_mm": round(r[4] * 1000.0, 1), "wrist_deg": round(r[5], 1),
                                   "twist_deg": [round(v, 1) for v in twisted(side, marks, r[0], r[1], r[2], r[6])[:2]]}
                               for n, r in sorted(made[side].items())}
        spans = [(join, left.get(t0, t1)) for t0, t1, join in frees[side]]
        if any(t1 - t0 > 1e-3 for t0, t1 in spans):
            out["ik"][side] = ik_keys([(t0, t1) for t0, t1 in spans if t1 - t0 > 1e-3])
    body.sort(key=lambda b: b[0])
    for (a0, a1, _, wa), (b0, _, _, wb) in zip(body, body[1:]):
        if b0 < a1 - 1e-6:
            raise MoveError(f"{wa} and {wb} both move the body at {b0} s")
    for t0, t1, vals, _ in body:
        for key, v in vals.items():
            if v:
                out[key] += [[t0, 0.0], [min(t0 + EASE, 0.5 * (t0 + t1)), v], [max(t1 - EASE, 0.5 * (t0 + t1)), v], [t1, 0.0]]
    for morph in list(faces) + [k for k in curves if k not in faces]:
        merged = []
        for t0, t1, v in sorted(faces.get(morph, [])):
            if merged and t0 < merged[-1][1] + 0.12:                 # close enough to hold through
                merged[-1][1], merged[-1][2] = max(merged[-1][1], t1), max(merged[-1][2], v)
            else:
                merged.append([t0, t1, v])
        keys = []
        for t0, t1, v in merged:
            keys += [[t0 - 0.06, 0.0], [t0 + 0.04, v], [max(t1 - 0.06, t0 + 0.05), v], [t1 + 0.06, 0.0]]
        keys = sorted(keys + curves.get(morph, []))                   # a gag's timed face (a yawn, a bite) beside them
        kept = []
        for t, v in keys:
            if kept and abs(t - kept[-1][0]) < 1e-3:
                kept[-1][1] = max(kept[-1][1], v)
            else:
                kept.append([t, v])
        out["expressions"].append({"morph": morph, "keys": [[round(t, 4), round(v, 3)] for t, v in kept]})
    out["perform"] = GG.legs(legs, mouths)
    out["held"] = GG.attachments(grips)
    for key in BODY:
        out[key] = [[round(t, 4), round(v, 3)] for t, v in out[key]]
    return out


def ik_keys(spans):
    """[[t, share], ...] of a hand's arm IK over the spans (t0, t1) a `free` move lets it go: all of it outside them,
    none inside, eased over FREE_EASE seconds at each end (half the span when it is shorter); spans that touch are one."""
    merged = []
    for t0, t1 in sorted(spans):
        if merged and t0 <= merged[-1][1] + 1e-6:
            merged[-1][1] = max(merged[-1][1], t1)
        else:
            merged.append([t0, t1])
    keys = []
    for t0, t1 in merged:
        e = min(FREE_EASE, 0.5 * (t1 - t0))
        keys += [[t0, 1.0], [t0 + e, 0.0], [t1 - e, 0.0], [t1, 1.0]]
    return [[round(t, 4), v] for t, v in keys]


def travel(a, b):
    """Seconds a hand takes from one place to another (each (point, dir, palm, ...) as `resolve` gives it, the point
    with any offset): TRAVEL_MIN plus its wrist's way at TRAVEL_SPEED, or as long as turning its fingers' way or its palm
    at TURN_SPEED takes when that is longer."""
    way = float(np.linalg.norm(np.asarray(b[0], float) - np.asarray(a[0], float)))
    turn = max(bend(a[1], b[1]), bend(a[2], b[2]))
    return max(TRAVEL_MIN + way / TRAVEL_SPEED, turn / TURN_SPEED)


def fling(a, b):
    """Seconds a hand thrown from one place to another by a reaction takes: FLING_MIN plus its wrist's way at
    FLING_SPEED."""
    return FLING_MIN + float(np.linalg.norm(np.asarray(b[0], float) - np.asarray(a[0], float))) / FLING_SPEED


def _rests(entries):
    """({side: [(t, place)]}, told): where each hand waits from clip time t on, as its rest entries say (`{name =
    "rest", hand, place, t}`; the first from the start, t None: "rest" unless an entry without `t` says otherwise), and
    the hands given a rest entry of their own. A hand that rests at the mic holds it from the start to the end."""
    out, told = {sd: [] for sd in SIDES}, set()
    for e in entries:
        if not (isinstance(e, dict) and e.get("name") == "rest"):
            continue
        t = e.get("t")
        if set(e) - {"name", "hand", "place", "t"} or e.get("hand") not in SIDES or e.get("place") not in REST_PLACES \
                or (t is not None and (isinstance(t, bool) or not isinstance(t, (int, float)))):
            raise MoveError(f"rest = {e!r}: expected {{name = \"rest\", hand = \"L\" | \"R\", place = "
                            f"{' | '.join(REST_PLACES)}, t = the clip seconds it waits there from (default: the start)}}")
        out[e["hand"]].append((None if t is None else float(t), e["place"]))
        told.add(e["hand"])
    for sd, rs in out.items():
        rs.sort(key=lambda r: -math.inf if r[0] is None else r[0])
        for (a, _), (b, _) in zip(rs, rs[1:]):
            if a == b:
                raise MoveError(f"the {sd} hand has two rest places from {'the start' if a is None else f'{a:g} s'}")
        if not rs or rs[0][0] is not None:
            rs.insert(0, (None, "rest"))
        if len(rs) > 1 and any(p == "mic" for _, p in rs):
            raise MoveError(f"the {sd} hand rests at the mic: it holds the mic from the start to the end, so its rest "
                            f"place cannot change")
    return out, told


def _homes(rests):
    """{side: [(t, place, own)]}: where each hand waits from clip time t on (the first from the start, t None), its rest
    places (`_rests`) with a pose for two (LONE) at its lone place while the other hand rests elsewhere; `own`: the change
    is the hand's own (a rest entry's `t`), not its partner's leaving or joining the pose."""
    def base(sd, t):
        p = rests[sd][0][1]
        for tt, pl in rests[sd][1:]:
            if t is not None and tt <= t:
                p = pl
        return p

    out = {}
    for sd, other in (("L", "R"), ("R", "L")):
        def eff(t, sd=sd, other=other):
            p = base(sd, t)
            return LONE[p] if p in LONE and base(other, t) != p else p
        mine = {t for t, _ in rests[sd][1:]}
        seq = [(None, eff(None), True)]
        for t in sorted(mine | {t for t, _ in rests[other][1:]}):
            if eff(t) != seq[-1][1]:
                seq.append((t, eff(t), t in mine))
        out[sd] = seq
    return out


def _home_at(homes, t):
    """The place a hand waits at at clip time t (`homes`: its [(t, place, own)], `_homes`)."""
    p = homes[0][1]
    for tt, pl, _ in homes[1:]:
        if tt <= t + 1e-9:
            p = pl
    return p


def _paired(homes):
    """[(t0, t1, place)]: the stretches both hands rest together at a pose for two (LONE; t0 None: from the start, t1
    None: to the end). A hand rests there, not at its lone place, only while the other one does too (`_homes`)."""
    seq = homes["L"]
    return [(t0, nxt[0], p) for (t0, p, _), nxt in zip(seq, seq[1:] + [(None, None, None)]) if p in LONE]


def _alone(mine, theirs, side, at, two="dainty", span=(None, None), after=None):
    """The plays (t0, t1, [(t, place, offset)], False) that rest a hand at a pose for two (`two`: "dainty", ... LONE)
    whose partner rests there too at its lone place (LONE: "dainty_one", on its own thigh) while it is alone: the partner
    away from the pose (from when it sets off on a move, or is let go, to when it is back), outside this hand's own moves
    and the trips they need. It glides there from the two-hand spot as its partner leaves, arriving as the partner gets
    to its move, and is back on the spot as its partner is. The partner's absences closer than the hand's round trip plus
    REST_MIN are one (it does not go back for a moment), and a stretch shorter than REST_MIN is left out (it stays on the
    two-hand spot). `mine` and `theirs` are plays (t0, t1, [(t, place, offset)], react); `at(side, place)` resolves one;
    `span` (t0, t1) the clip seconds both rest at the pose (None: from the start, to the end; `_paired`), the hand
    leaving its lone place in time to be at `after`, where it rests next, at t1."""
    other = "R" if side == "L" else "L"
    lone = LONE[two]
    one = at(side, lone)
    trip = travel(at(side, two), one)
    lo = -math.inf if span[0] is None else span[0]
    hi = math.inf if span[1] is None else span[1] - travel(one, at(side, after))
    away = []                                                        # [it leaves, it is back, it gets to its move]
    for t0, t1, seq, react in sorted(theirs, key=lambda p: p[0]):
        home = at(other, two)
        a = t0 if react else seq[0][0] - travel(home, at(other, seq[0][1]))
        b = t1 + travel(at(other, seq[-1][1]), home)
        if away and a - away[-1][1] < REST_MIN + 2.0 * trip:
            away[-1][1] = max(away[-1][1], b)
        else:
            away.append([a, b, max(seq[0][0], a + trip)])
    own = [(t0 if react else seq[0][0] - travel(one, at(side, seq[0][1])), t1 + travel(at(side, seq[-1][1]), one))
           for t0, t1, seq, react in mine]
    out = []
    for a, b, there in away:
        pieces = [(max(a, lo), min(b - trip, hi))]
        for c, d in own:
            pieces = [q for p, r in pieces for q in ((p, min(r, c)), (max(p, d), r)) if q[1] > q[0]]
        for p, r in pieces:
            if r - p < REST_MIN:
                continue
            arrive = max(there, p + trip)
            glide = (abs(p - a) < 1e-9 or abs(p - lo) < 1e-9) and arrive < r   # from the two-hand spot, with its partner
            seq = [(p, two, (0, 0, 0)), (arrive, lone, (0, 0, 0))] if glide else [(p, lone, (0, 0, 0))]
            out.append((p, r, seq, False))
    return out


def _strip_seq(s, t0, t1):
    """The places of a hand on a strip over a free move's span (t0, t1): its samples there, [(t, (STRIP, i), no offset)]
    (`s` the side's samples {"t", ..., "ok"} or None: none), less those at either end where the strip twists the arm
    past the human range (`ok` False: a dance's shrug the span starts in): the hand joins the strip after them and
    leaves it before them, unless the strip is past the range all through."""
    if s is None:
        return []
    seq = [(float(t), (STRIP, i), (0, 0, 0)) for i, t in enumerate(s["t"]) if t0 - 1e-6 <= float(t) <= t1 + 1e-6]
    ok = s.get("ok")
    if ok is None or not any(ok[i] for _, (_, i), _ in seq):
        return seq
    a = next(j for j, (_, (_, i), _) in enumerate(seq) if ok[i])
    b = max(j for j, (_, (_, i), _) in enumerate(seq) if ok[i])
    return seq[a:b + 1]


def _said(place):
    """A place as an error message names it."""
    return "the strip" if isinstance(place, tuple) else repr(place)


def _hand(spans, side, homes, world, at, told=False, mid=None, left=None, mic=False):
    """The keys of one hand: its moves' places on their times (`at(place)` resolves one, as `resolve`: a key at a place
    whose elbow is held its own way carries its pole, `pole`), travelling between them at a human pace (`travel`), home
    to its rest place between two moves when it can stay there REST_MIN seconds. `spans` are (t0, t1, [(t, place,
    offset)], react): a reacting move's hand leaves where it is on t0 and arrives when `fling` gets
    it there. `homes` [(t, place, own)]: where it rests from clip time t on (`_homes`, the first from the start). A rest
    place of its own (`own`, a rest entry's `t`) it reaches on its time, setting off early as for a move: a move it plays
    across that time, or one that leaves it too little time to get there or to go on from there, is refused; one that
    starts on that time takes it there after it. A change of its partner's (a pose for two left or joined, its lone
    place: `_homes`) it makes when it is free, as soon as it can, and leaves out when its next move comes too soon (it
    goes home to the new place after the move). A hand with no move waits at its rest place when it has one of its own
    (`told`: a rest entry), else it is left alone; one that rests at the mic (`mic`) holds it in every move. With `mid`
    (`_midway`) a trip whose hand would fold over its wrist or wring its arm on the way passes a key halfway
    (`_via`). A free move on a strip (its places the strip's samples) keeps the hand on the strip to the end of the move,
    or leaves it at the last sample from which the hand still gets to the next move on time; `left` (a dict) gets {the
    move's t0: when it leaves}. A key at a place seen through (SIGHTED) is marked `sight`: the pose stage carries the
    hand with the head from the first such key to the last of a run of them."""
    grip = SHAPES["curled"] if mic else None                        # a mic hand never lets go of the mic
    keys = []

    def where(name, off=(0, 0, 0)):
        point, d, palm, *_ = at(name)
        return np.asarray(point, float) + np.asarray(off, float), d, palm

    def key(t, name, off=(0, 0, 0)):
        point, d, palm, shape, *more = at(name)
        k = {"t": round(t, 4), "at": [round(float(p + o), 4) for p, o in zip(point, off)], "dir": world(d),
             "palm": world(palm), "_d": d, "_p": palm}
        if shape is None:
            k["strip"] = True                                        # on a strip: the fingers are the strip's
        else:
            k["fingers"] = grip if grip is not None else SHAPES[shape] if isinstance(shape, str) else shape
        if name in SIGHTED:
            k["sight"] = True                                        # held with the head (the pose stage)
        if len(more) > 2 and more[2] is not None:
            k["pole"] = [round(float(v), 4) for v in more[2]]        # the elbow held its own way there (POLES)
        keys.append(k)

    if not spans and len(homes) == 1:
        if homes[0][1] != "rest" or told:
            key(0.0, homes[0][1])                                    # it waits there from the start
        return _via(keys, world, mid)
    for k in range(1, len(spans)):
        if spans[k][0] < spans[k - 1][1] - 1e-6:
            raise MoveError(f"the {side} hand plays two moves at once (at {spans[k - 1][0]} and {spans[k][0]} s)")
    plan = [[t0, t1, seq, react, None] for t0, t1, seq, react in spans]        # None: a move; else a rest change
    for t, p, own in homes[1:]:
        if any(abs(sp[2][0][0] - t) < 1e-6 for sp in spans):
            continue                                                 # a move on its time: it goes there after it
        busy = next((sp for sp in spans if sp[0] + 1e-6 < t < sp[1] - 1e-6), None)
        if busy is not None and own:
            raise MoveError(f"the {side} hand plays a move from {busy[0]:g} to {busy[1]:g} s and its rest place changes "
                            f"to {p!r} at {t:g} s: change it before the move or after it")
        if busy is None:
            plan.append([t, t, [(t, p, (0, 0, 0))], False, "own" if own else "partner"])
    plan.sort(key=lambda e: e[0])

    def home(k):
        """The place the hand rests at before entry k of the plan: where the last rest change took it, or after a move
        the rest place it has at the move's end."""
        if k == 0:
            return homes[0][1]
        return plan[k - 1][2][0][1] if plan[k - 1][4] else _home_at(homes, plan[k - 1][1])

    kept = []                                                        # a partner's change: made when there is time
    for i, e in enumerate(plan):
        if e[4] != "partner":
            kept.append(e)
            continue
        prev = kept[-1] if kept else None
        to = where(e[2][0][1])
        if prev is None:
            t, resting = e[0], homes[0][1]
        elif prev[4]:
            t, resting = max(e[0], prev[0] + travel(where(prev[2][0][1]), to)), prev[2][0][1]
        else:
            t, resting = max(e[0], prev[1] + travel(where(*prev[2][-1][1:]), to)), _home_at(homes, prev[1])
        nxt = next((n for n in plan[i + 1:] if n[4] != "partner"), None)
        if e[2][0][1] == resting or (nxt is not None and t + REST_MIN + (
                0.0 if nxt[3] else travel(to, where(*nxt[2][0][1:]))) > nxt[2][0][0] + 1e-6):
            continue                                                 # there already, or no time before its next move
        kept.append([t, t, [(t, e[2][0][1], (0, 0, 0))], False, "partner"])
    plan = kept
    straight = [False]                                               # from entry k - 1 straight on to entry k
    for k in range(1, len(plan)):
        (_, t1, pseq, _, pkind), (t0, _, nseq, react, _) = plan[k - 1], plan[k]
        if pkind:                                                    # at a rest place it changed to: it is home
            straight.append(False)
            continue
        last, rest = where(*pseq[-1][1:]), where(home(k))
        if react:                                                    # it leaves on t0 from where it is by then
            straight.append(t0 - t1 < travel(last, rest) + REST_MIN)
        else:
            straight.append(nseq[0][0] - t1 < travel(last, rest) + REST_MIN + travel(rest, where(*nseq[0][1:])))
    for k, (t0, _, seq, react, _) in enumerate(plan):               # a reaction arrives when its fling gets it there
        if react:
            frm = where(*plan[k - 1][2][-1][1:]) if straight[k] else where(home(k))
            arrive = t0 + fling(frm, where(*seq[0][1:]))
            plan[k][2] = [(max(t, arrive), p, off) for t, p, off in seq]

    def trip(k, a, b):
        return fling(a, b) if plan[k][3] else travel(a, b)

    for k, (t0, t1, seq, _, kind) in enumerate(plan):
        rest = home(k)
        if not straight[k]:                                          # it sets off from home to arrive on time
            leave = seq[0][0] - trip(k, where(rest), where(*seq[0][1:]))
            if k and plan[k - 1][4] and leave < plan[k - 1][0] - 1e-6:
                raise MoveError(f"the {side} hand needs {seq[0][0] - leave:.2f} s to get from its rest place {rest!r} "
                                f"(at {plan[k - 1][0]:g} s) to {_said(seq[0][1])} (at {seq[0][0]:g} s) and has "
                                f"{seq[0][0] - plan[k - 1][0]:.2f} s: change the rest place earlier or start the move "
                                f"later")
            leave = max(leave, 0.0) if k == 0 else leave
            if leave < seq[0][0] - 1e-6:
                key(leave, rest)
        on = isinstance(seq[0][1], tuple)                            # on a strip: it stays as long as it can
        if on and k + 1 < len(plan) and straight[k + 1]:
            nt, nplace, noff = plan[k + 1][2][0]
            dest = where(nplace, noff)
            ok = [j for j, (t, p, off) in enumerate(seq) if t + trip(k + 1, where(p, off), dest) <= nt + 1e-6]
            if not ok:
                raise MoveError(f"the {side} hand needs {trip(k + 1, where(*seq[0][1:]), dest):.2f} s to get from the "
                                f"strip (at {seq[0][0]:g} s) to {_said(nplace)} (at {nt:g} s): let it go later or start "
                                f"the next move later")
            seq = seq[:ok[-1] + 1]
        for t, p, off in seq:
            key(t, p, off)
        if on and left is not None:
            left[t0] = seq[-1][0]
        last, lt = where(*seq[-1][1:]), seq[-1][0]
        end = lt if on or kind else t1                               # the last place holds to the end of the move ...
        if k + 1 < len(plan) and straight[k + 1]:
            nt, nplace, noff = plan[k + 1][2][0]
            need = trip(k + 1, last, where(nplace, noff))
            end = min(end, nt - need)                                # ... or the hand leaves it in time for the next
            if end < lt - 1e-6:
                then = ("change the rest place later or end the move sooner" if plan[k + 1][4]
                        else "start the next move later")
                raise MoveError(f"the {side} hand needs {need:.2f} s to get from {_said(seq[-1][1])} (at {lt:g} s) to "
                                f"{_said(nplace)} (at {nt:g} s) and has {nt - lt:.2f} s: {then}")
        if end - lt > 1e-6:
            key(end, seq[-1][1], seq[-1][2])
        if not kind and (k + 1 == len(plan) or not straight[k + 1]):
            back = _home_at(homes, t1)
            key(max(t1, lt) + travel(last, where(back)), back)
    return _via(keys, world, mid)


def halfway(a, b):
    """(dir, palm) halfway along the shortest turn from hand orientation `a` to `b` (each (dir, palm)): the chordal mean
    of the two rotations, which for two of them is the midpoint of the turn."""
    U, _, Vt = np.linalg.svd(_basis(*a) + _basis(*b))
    if np.linalg.det(U @ Vt) < 0:
        U[:, -1] *= -1.0
    R = U @ Vt
    return R[:, 0], R[:, 1]


def _basis(d, p):
    """The hand's frame as a rotation: columns its direction, its palm (square to it) and their cross."""
    d = _unit(d)
    p = _unit(np.asarray(p, float) - (np.asarray(p, float) @ d) * d)
    return np.stack([d, p, np.cross(d, p)], 1)


def _via(keys, world, mid, depth=2):
    """`keys` (each with its directions in the character's frame as "_d" / "_p", dropped here) with a key halfway through
    every trip whose hand, turned halfway from one end's orientation to the other's (`halfway`), would bend more than
    HELD degrees from the forearm the arm has there or twist the arm past the human range, or whose elbow's pole swings
    far round the shoulder (`mid`, `_midway`): the wrist halfway between the two places, the hand turned back toward its
    forearm (`natural`) and the elbow turned to keep the twists in range (`arm_for`), its pole there too; the two halves
    are checked again, `depth` times. Interpolating only the two ends' orientations while the arm swings round under
    them folds the hand over its wrist on the way (a hand going up to the hair from beside the hip) or wrings the
    forearm, and a pole keyed straight across from behind the back to in front passes by the shoulder, where the IK's
    elbow flips round. A hand holding its place needs none; a trip to or from a strip is checked for the bend only (the
    pose stage turns the elbow's pole to the strip's elbow there)."""
    out = []
    for a, b in zip(keys, keys[1:] + [None]):
        out.append({k: v for k, v in a.items() if k not in ("_d", "_p")})
        if b is not None:
            out += _halfway_keys(a, b, world, mid, depth)
    return out


def _halfway_keys(a, b, world, mid, depth):
    """The keys `_via` puts between keys a and b (none, one halfway, or more about it in its halves)."""
    if mid is None or depth <= 0 or b["t"] - a["t"] < 1e-3 or (a.get("strip") and b.get("strip")) or (
            a["at"] == b["at"] and a["dir"] == b["dir"] and a["palm"] == b["palm"]):
        return []                                                    # (along a strip the hand is the strip's)
    got = mid(a, b, not (a.get("strip") or b.get("strip")))
    if got is None:
        return []
    w, d, p, pole = got
    k = {"t": round(0.5 * (a["t"] + b["t"]), 4), "at": [round(float(v), 4) for v in w], "dir": world(d),
         "palm": world(p), "_d": d, "_p": p, **({"pole": [round(float(v), 4) for v in pole]} if pole is not None else {})}
    return (_halfway_keys(a, k, world, mid, depth - 1) + [{n: v for n, v in k.items() if n not in ("_d", "_p")}]
            + _halfway_keys(k, b, world, mid, depth - 1))


def _midway(side, marks):
    """`mid(a, b, twist)` for `_via`: None when the hand `side` halfway between keys a and b (the wrist halfway round the
    shoulder, at the mean of the two ends' distances from it, the hand turned halfway, its elbow's pole halfway between
    the two keys' poles: points in the character's frame, a key without one has the pose stage's own, AR.POLE) bends at
    most HELD from its forearm and, with `twist`, keeps the arm's twists in the human range (`twisted`, `overtwist`) and
    less than TWIST_STEP from the keys' on either side, so they turn the short way between them, through the middle;
    else (wrist, dir, palm, pole) of a key there that does: its elbow turned to bring the upper arm's twist nearest the
    middle of the two keys' (`arm_for`), the hand turned toward its forearm (`natural`) and rolled about its own length
    to the middle of their forearm twists (`rolled`): the forearm turns over with the arm on the way, not as the two ends'
    orientations interpolate while the arm swings round under them (the pole None when neither key has one and the elbow
    need not turn). Two poles more than POLE_TURN apart round the shoulder have their halfway point round it too, on the
    outside (and a key there even when the hand is fine), not on the straight line between them; two keys whose twists
    are TWIST_STEP or more apart get the halfway key even when it is fine too, so the IK turns the arm through it. The
    wrist goes round the shoulder as an arm swings, not on the straight line between the two places, which passes by the
    shoulder on a trip from the hip up behind the head: an arm folded that far wrings the upper arm past its range
    whichever way the elbow points."""
    S, home = np.asarray(marks[f"arm.{side}"], float), pole_point(side, marks)
    out = _v(1.0 if side == "L" else -1.0, 0.0, 0.0)

    def twists(k):
        return twisted(side, marks, k["at"], k["_d"], k["_p"], k.get("pole"))

    def mid(a, b, twist=True):
        ra, rb = np.asarray(a["at"], float) - S, np.asarray(b["at"], float) - S
        la, lb = float(np.linalg.norm(ra)), float(np.linalg.norm(rb))
        m = (ra / la if la > 1e-6 else 0.0) + (rb / lb if lb > 1e-6 else 0.0)
        w = S + 0.5 * (la + lb) * _unit(m if float(np.linalg.norm(m)) > 1e-6 else out)
        d, p = halfway((a["_d"], a["_p"]), (b["_d"], b["_p"]))
        pa, pb = a.get("pole"), b.get("pole")
        pole, turned = None, False
        if pa is not None or pb is not None:                    # the poles as seen round the arm's line (the elbow's
            va, vb = (np.asarray(home if q is None else q, float) - S for q in (pa, pb))   # swivel), not the shoulder
            u = _unit(w - S)
            ra, rb = va - (va @ u) * u, vb - (vb @ u) * u
            la, lb = float(np.linalg.norm(ra)), float(np.linalg.norm(rb))
            turned = la < 1e-6 or lb < 1e-6 or float(ra @ rb) < la * lb * math.cos(math.radians(POLE_TURN))
            if turned:                                          # round the arm's line, on its outer side
                m = (ra / la if la > 1e-6 else 0.0) + (rb / lb if lb > 1e-6 else 0.0)
                if np.linalg.norm(m) < 1e-6:
                    m = out - (out @ u) * u
                pole = S + 0.5 * float((va + vb) @ u) * u + _unit(m) * max(0.5 * (la + lb), POLE_OFF)
            else:
                pole = S + 0.5 * (va + vb)
        f = forearm(side, marks, w, pole)
        if not twist:
            if bend(d, f) <= HELD:
                return (w, d, p, pole) if turned else None
            return (w,) + natural(d, p, f) + (pole,)
        ta, tb, tw = twists(a), twists(b), twisted(side, marks, w, d, p, pole)
        if bend(d, f) <= HELD and overtwist(tw, 0.0) <= 0.0 and all(
                abs(tw[i] - t[i]) < TWIST_STEP for t in (ta, tb) for i in (0, 1)):
            far = any(abs(ta[i] - tb[i]) >= TWIST_STEP for i in (0, 1))      # the ends far apart: keyed through the
            return (w, d, p, pole) if turned or far else None             # middle, as the IK might take the other way
        dd, pp, _, pole = arm_for(side, marks, w, lambda f: natural(d, p, f), pole, HELD, d,
                                  toward=(None, 0.5 * (ta[1] + tb[1])))
        return (w,) + rolled(side, marks, w, dd, pp, pole, 0.5 * (ta[0] + tb[0])) + (pole,)
    return mid


def rolled(side, marks, wrist, d, p, pole, target, rounds=4):
    """(dir, palm) of the hand `side` (pointing along `d`, its palm toward `p`, its wrist at `wrist` and its elbow's pole
    at `pole`) rolled about its own length until its forearm twists `target` degrees (`twisted`)."""
    d = _unit(d)
    for _ in range(rounds):
        err = target - twisted(side, marks, wrist, d, p, pole)[0]
        err = (err + 180.0) % 360.0 - 180.0
        if abs(err) < 0.5:
            break
        p = _turn(np.asarray(p, float), d, math.radians(AT.SIGN[side] * err))   # SIGN: + supination about the arm
    return d, _unit(np.asarray(p, float) - (np.asarray(p, float) @ d) * d)
