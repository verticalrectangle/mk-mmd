"""The per-member tables of mk.toml (mkmmd/core/tables.py): `[pose.<cast>]`, `[perform.<cast>]` and `[sim.<cast>]` must be a
cast member's, and hold only keys their stage reads. The Blender half (the stages calling the check) is
tests/test_build_stages.py."""
import tomllib

import pytest

from mkmmd.core import tables as TB

CAST = ["rin", "bob"]


def check(text, section="pose", cast=CAST):
    TB.check_section(section, tomllib.loads(text).get(section), cast)


def test_a_table_must_be_a_cast_members_and_the_error_lists_the_cast():
    with pytest.raises(TB.TableError, match=r"\[pose\.rn\]: no cast member 'rn' \(cast: rin, bob\)"):
        check("[pose.rn]\nlean = 3\n")
    with pytest.raises(TB.TableError, match=r"\[sim\.rin\]: no cast member 'rin' \(cast: none: the project has no \[\[cast\]\]\)"):
        check("[sim.rin]\nfloor = false\n", "sim", cast=[])
    with pytest.raises(TB.TableError, match=r"\[perform\] holds tables named after cast members, \[perform\.<cast>\]"):
        TB.check_section("perform", [1, 2], CAST)
    with pytest.raises(TB.TableError, match=r"\[pose\] has `lean` as a plain key"):
        check("[pose]\nlean = 3\n")                                             # a key written where a cast name goes


def test_an_unknown_key_is_refused_with_the_known_ones():
    with pytest.raises(TB.TableError) as e:
        check("[pose.rin]\nlean = 3\nsiting = 'car:seat'\n")
    msg = str(e.value)
    assert msg.startswith("[pose.rin]: unknown key 'siting' (known: ") and "sit, sit_offset" in msg and "lean" in msg
    with pytest.raises(TB.TableError, match=r"\[perform\.bob\]: unknown keys 'blinks', 'gaz' \(known: "):
        check("[perform.bob]\nblinks = {per_min = 12}\ngaz = []\nlids = 0.1\n", "perform")
    with pytest.raises(TB.TableError, match=r"\[sim\.rin\]: unknown key 'flor' \(known: anchor_free, colliders"):
        check("[sim.rin]\nflor = 0.3\n", "sim")


def test_keys_below_the_table_are_checked_where_the_stage_reads_them_all():
    for text, where in [("[pose.rin.hands.left]\nat = [0, 0, 1]\n", r"\[pose\.rin\] hands: unknown key 'left' \(known: L, R\)"),
                        ("[pose.rin.hands.L]\ngripp = 'car:wheel'\n", r"\[pose\.rin\] hands\.L: unknown key 'gripp' \(known: "),
                        ("[pose.rin.hands.R]\nkeys = [{t = 1, pos = [0, 0, 1]}]\n", r"hands\.R\.keys\[0\]: unknown key 'pos'"),
                        ("[pose.rin.hands.R]\ngrip = 'pen:barrel'\nwobble = {dg = 2}\n", r"hands\.R\.wobble: unknown key 'dg'"),
                        ("[pose.rin]\nhead = {pitc = 3}\n", r"\[pose\.rin\] head: unknown key 'pitc'"),
                        ("[pose.rin]\nhips = {shft = [0, 0, -0.1]}\n", r"\[pose\.rin\] hips: unknown key 'shft'"),
                        ("[pose.rin]\nfeet = {left = [0, 0]}\n", r"\[pose\.rin\] feet: unknown key 'left'"),
                        ("[pose.rin]\nfingers = {left = 'fist'}\n", r"\[pose\.rin\] fingers: unknown key 'left'"),
                        ("[[pose.rin.drape]]\nchain = ['a']\ndirs = [[0, 0, -1]]\nscal = [1]\n", r"drape\[0\]: unknown key 'scal'"),
                        ("[pose.rin]\nsit = {hip = [0, 0, 0.5], height = 0.4}\n", r"\[pose\.rin\] sit: unknown key 'height'")]:
        with pytest.raises(TB.TableError, match=where):
            check(text)
    for text, where in [("[perform.rin]\nblink = {per_minute = 12}\n",
                         r"\[perform\.rin\] blink: unknown key 'per_minute' \(known: extra, per_min, seed\)"),
                        ("[perform.rin]\ngaze = [{t = 1, at = [0, 0, 1]}, {t = 2, at = [0, 0, 1], hld = 3}]\n", r"gaze\[1\]: unknown key 'hld'"),
                        ("[perform.rin]\nbob = {deg = 2, timelines = 'x'}\n", r"\[perform\.rin\] bob: unknown key 'timelines'"),
                        ("[perform.rin]\nsing = {timeline = 'x', mouths = 1}\n", r"\[perform\.rin\] sing: unknown key 'mouths'"),
                        ("[perform.rin]\nstrum = {hand = 'R', rhytm = 'beats:8'}\n", r"\[perform\.rin\] strum: unknown key 'rhytm'"),
                        ("[perform.rin]\nhead_limits = {yaww = 60}\n", r"head_limits: unknown key 'yaww' \(known: down, up, yaw\)"),
                        ("[perform.rin]\ntwitch = [{t = 1, family = 'ears', degs = 5}]\n", r"twitch\[0\]: unknown key 'degs'")]:
        with pytest.raises(TB.TableError, match=where):
            check(text, "perform")


POSE_ALL = """
[pose.rin]
sit = {hip = [0, 0, 0.5], facing = [0, -1, 0], floor_z = 0.0, pelvis_deg = 6, back_deg = 0}
sit_offset = [0, 0.03, 0]
feet = {L = [0.1, 0.0], R = {cast = "rin", point = [0, 0, 0]}}
hips = {shift = [0, 0, -0.03], roll = 3, yaw = -4}
toes = {L = -8, R = 10}
lean = 12
turn = -6
lean_share = 0.5
turn_share = 0.5
head = {pitch = -4, yaw = 2, roll = 1, neck = 0.4}
fingers = {L = "relaxed", R = {index = [8, 10]}}
[pose.rin.hands.L]
grip = "guitar:neck"
fret = 3
chord = "power"
press = 0.3
move = 0.12
keys = [{t = 1.9, fret = 5}, {t = 3.8, fret = 1, chord = "E", move = 0.2}]
ride = "cast:rin.upper_body2"
pole = [0, 0, 1]
clock = 10
approach = 90
wrap = -1
seeds = 6
skin_radius = 0.16
[pose.rin.hands.R]
at = {cast = "rin", point = [0.25, -0.15, 1.05]}
dir = [0, -1, 0]
palm = [0, 0, -1]
rest = "table:edge"
along = 0.5
offset = [0, 0, 0]
lift = 0.03
fingers = "fist"
face = "palm"
edge = 0.004
thumb = "neck"
tip = 0.008
track = "nib"
channel = "target"
posture = {nib = [0, 0, 0.7], table = "table:top"}
wobble = {deg = 2, tau = 18, seed = 3}
keys = [{t = 0.0, at = [0.3, -0.6, 1.0], dir = [0, -1, 0], palm = [0, 0, -1]}]
[[pose.rin.drape]]
chain = ["a", "b"]
dirs = [[0, -0.7, -0.7], [0, -1, 0]]
scale = [1, 0.8]
"""

PERFORM_ALL = """
[perform.rin]
look = {cast = "bob", point = [0, 0, 1.5]}
gaze = [{t = 1, at = "cast:bob", hold = 1.0, back = 0.4, rise = 0.3, blink = false}]
glance = [{t = 2, at = [0, 0, 1], dur = 0.8, pitch = 2.75, rise = 0.12, fall = 0.25, blink = true}]
head_share = 0.7
head_limits = {yaw = 75, up = 35, down = 45}
neck_share = 0.35
eye_max = 24
breath = {per_min = 16.5, deg = 0.6}
sway = {deg = 0.37, period = 2.5}
nod = {deg = 0.48, period = 2.3}
bob = {deg = 1.5, timeline = "audio/timeline.json", beats = [1, 2], downbeat_accent = 1.6}
startle = [1.0]
lean = [[0, 0], [1, 5]]
turn = [[0, 0]]
tilt = [[0, 0]]
head_tilt = [[0, 0]]
rock = {deg = 3, period = 4, phase = 0}
head_rock = {deg = 3, period = 4, phase = 0}
blink = {per_min = 15, seed = 1, extra = [[1.0, 0.2]]}
lids = 0.1
sing = {timeline = "t.json", lines = [1, 2], mouth = 0.8, lead = -0.03, voice = "en-gb"}
expressions = [{morph = "smile", keys = [[0, 0], [1, 1]]}]
twitch = [{bones = ["ear_root.L"], family = "ears", t = 3.2, deg = 14, axis = [1, 0, 0], dur = 0.22}]
strum = [{hand = "R", prop = "guitar", grip = "strum", timeline = "t.json", sigma = 0.02, rhythm = "beats:8", from = 1.0, to = 3.0, accent = "downbeats", span = 0.09, depth = 0.003, share = 0.5, min_gap = 0.05}]
"""

SIM_ALL = """
[sim.rin]
families = ["back_hair"]
params = {back_hair = {sag = [20, 85]}}
colliders = "set"
props = true
fingers = false
floor = false
wind = {carrier = "car", exposure = 0.3}
use_masks = false
anchor_free = 0.25
substeps = 10
settle_s = 1.5
engine = "auto"
"""


def test_every_key_a_stage_reads_passes():
    check(POSE_ALL)
    check(PERFORM_ALL, "perform")
    check(SIM_ALL, "sim")
    check("", "pose")                                                                  # no section at all
    check("[pose.rin]\n")                                                              # an empty table has no unknown key
