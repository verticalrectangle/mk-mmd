"""The mosh moves, bpy-free (mkmmd.core.mosh through mkmmd.core.moves.compile): every mosh place natural and in reach and
the right hand's the left's mirror image, a pound's face meeting its target on the blow with both fists on the handle,
a headbang's nods and a bass face on the hits through their own body curves, a pogo's ballistic jumps, wotagei's
penlights in the fists and its routines, a whistle's pull, the hand checks' rows, and a trip from the hip up over the
head going round the shoulder. The Blender half (the props bone-parented, the keys laid, the checks) is proved on
scratch builds."""
import numpy as np
import pytest

from mkmmd.core import gags as GG
from mkmmd.core import mosh as MO
from mkmmd.core import moves as MV
from mkmmd.core import perform as PF

MARKS = {"arm.L": [0.0914, 0.006, 1.1795], "arm.R": [-0.0914, 0.006, 1.1795], "chest": [0.0, -0.0504, 1.019],
         "mouth": [0.0, -0.0647, 1.2795], "eye": [0.0, -0.044, 1.3229], "reach": 0.374, "upper": 0.209, "fore": 0.166,
         "hand": 0.166, "waist": [0.0, -0.0535, 0.9347]}
BEAT = 0.8225
BEATS = [round(BEAT * k, 4) for k in range(40)]
MALLET = {"scale": 1.0, "bounds": None,
          "use": {"grip": [{"name": "hold", "type": "hold", "center": [0, 0, 0.4], "axis": [0, 0, 1], "length": 0.64,
                            "radius": 0.0208}],
                  "anchor": [{"name": "face", "point": [0.23, 0.0, 1.12], "dir": [1, 0, 0]}]}}
PEN = {"scale": 1.0, "bounds": None,
       "use": {"grip": [{"name": "hold", "type": "hold", "center": [0, 0, 0.05], "axis": [0, 0, 1], "length": 0.075,
                         "radius": 0.0155}]}}
PROPS = {"mallet": MALLET, "pen_l": PEN, "pen_r": PEN}
PENS = {"L": "pen_l", "R": "pen_r"}


def compile(entries, marks=MARKS):
    return MV.compile(entries, marks, BEATS, downbeats=BEATS[::3], props=PROPS)


def body(scale):
    return {k: (np.asarray(v, float) * scale).tolist() if isinstance(v, list) else v * scale for k, v in MARKS.items()}


@pytest.mark.parametrize("name", MO.PLACES)
def test_every_mosh_place_keeps_the_wrist_natural_the_arm_untwisted_in_reach_and_mirrored(name):
    for scale in (0.85, 1.0, 1.2):
        m = body(scale)
        got = {}
        for side in ("L", "R"):
            pt, d, p, _, _, bend, pole = MV.resolve(name, side, m)
            assert bend <= MV.HELD + 1e-6
            assert MV.overtwist(MV.twisted(side, m, pt, d, p, pole), 0.0) == 0.0
            assert np.linalg.norm(pt - np.array(m[f"arm.{side}"])) <= m["reach"] + 1e-6
            got[side] = (pt, d, p)
        M = np.array([-1.0, 1.0, 1.0])
        for a, b in zip(got["L"], got["R"]):
            assert a * M == pytest.approx(b, abs=1e-9)


def test_the_chest_carries_a_place_written_where_carried_says_to_the_world_point():
    p = np.array([0.1, -0.9, 0.4])
    for lean, crouch in ((0.0, 0.0), (30.0, 0.0), (37.0, 0.08)):
        assert MO.lifted(MO.carried(p, MARKS, lean, crouch), MARKS, lean, crouch) == pytest.approx(p)
    assert MO.lifted(np.array(MARKS["waist"]), MARKS, 40.0) == pytest.approx(np.array(MARKS["waist"]))
    assert MO.lifted(np.array(MARKS["chest"]), MARKS, 20.0)[1] < MARKS["chest"][1]     # leaning forward: toward -Y


def _face_at_key(res, held, side, key):
    """Where the mallet's face is with the hand `side` at `key` (its place in the frame the hands ride)."""
    R, T = (np.array(v) for v in held["rel"])
    H = GG._frame(np.array(key["dir"]), np.array(key["palm"]))
    return np.array(key["at"]) + H @ T + H @ R @ np.array([0.23, 0.0, 1.12])


@pytest.mark.parametrize("target", [[0.0, -1.0, 0.43], [0.0, -1.2, 0.56], [0.15, -1.05, 0.3]])
def test_a_pound_brings_the_face_down_on_its_target_on_each_hit_leaning_into_it(target):
    res = compile([{"name": "pound", "t": 1.4, "dur": 3.0, "prop": "mallet", "target": target}])
    held = res["held"]["mallet"]
    assert held["bone"] == "wrist" and held["side"] == "R"
    lean = PF.eased_keys(np.array(BEATS[2:5]), res["lean"])
    brace = PF.eased_keys(np.array(BEATS[2:5]), res["perform"]["crouch"])
    keys = {k["t"]: k for k in res["hands"]["R"]}
    for hit, ln, br in zip(BEATS[2:5], lean, brace):                  # the blows on the beats 1.645, 2.4675, 3.29
        face = MO.lifted(_face_at_key(res, held, "R", keys[hit]), MARKS, ln, br)
        assert face == pytest.approx(np.array(target), abs=0.003)     # the chest carries the place onto the target
    assert 5.0 <= lean[0] <= 45.0
    assert max(v for _, v in res["perform"]["crouch"]) >= MO.POUND["give"]   # the knees give after it


def test_a_pound_keeps_both_fists_on_the_handle_at_every_key_with_no_key_halfway():
    res = compile([{"name": "pound", "t": 1.4, "dur": 3.0, "prop": "mallet", "target": [0.0, -1.0, 0.43]}])
    R, T = (np.array(v) for v in res["held"]["mallet"]["rel"])
    hold = GG.hold_of("mallet", MALLET)
    _, hole = GG.fist(MARKS, 0.0208)
    left = {k["t"]: k for k in res["hands"]["L"] if 1.4 <= k["t"] <= 4.4}
    right = [k for k in res["hands"]["R"] if 1.4 <= k["t"] <= 4.4]
    assert len(right) > 10 and set(left) == {k["t"] for k in right}  # the same keys, none halfway on one hand only
    for k in right:
        assert "fingers" in k                                         # every key at a place (none halfway)
        H = GG._frame(np.array(k["dir"]), np.array(k["palm"]))
        axis = H @ R @ hold["axis"]
        base = np.array(k["at"]) + H @ T + H @ R @ hold["centre"]     # a point on the handle's line
        o = left[k["t"]]
        fist = np.array(o["at"]) + MV._off(hole, np.array(o["dir"]), np.array(o["palm"]), "L")
        r = fist - base
        assert np.linalg.norm(r - (r @ axis) * axis) < 0.004          # the left fist's hole on the handle's line
        assert MV.bend(o["dir"], MV.forearm("L", MARKS, o["at"], o.get("pole"))) <= MV.HELD + 1.0


def test_a_pound_starts_wound_up_and_refuses_a_target_out_of_reach_or_a_blow_too_soon():
    res = compile([{"name": "pound", "t": 1.4, "dur": 3.0, "prop": "mallet"}])
    first = next(k for k in res["hands"]["R"] if k["t"] == 1.4)
    blow = next(k for k in res["hands"]["R"] if k["t"] == BEATS[2])
    assert first["at"][2] > blow["at"][2] + 0.25                      # the wind-up is up high
    with pytest.raises(MV.MoveError, match="out of reach of the mallet"):
        compile([{"name": "pound", "t": 1.4, "dur": 3.0, "prop": "mallet", "target": [0.0, -2.5, 0.2]}])
    with pytest.raises(MV.MoveError, match="start it before the hit"):
        compile([{"name": "pound", "t": 1.6, "dur": 2.0, "prop": "mallet"}])
    with pytest.raises(MV.MoveError, match="needs `prop`"):
        compile([{"name": "pound", "t": 1.4, "dur": 2.0}])
    with pytest.raises(MV.MoveError, match="`target` is for pound"):
        compile([{"name": "chomp", "t": 1.4, "dur": 2.0, "target": [0, -1, 0.5]}])


def test_a_headbang_nods_the_body_and_head_down_on_each_hit_as_deep_as_its_depth():
    hits = [1.0, 1.41, 1.82, 2.23]
    res = compile([{"name": "headbang", "t": 0.7, "dur": 2.0, "hits": hits, "hands": "hips"}])
    lean, nod = dict(map(tuple, res["lean"])), dict(map(tuple, res["head_nod"]))
    assert [lean[h] for h in hits] == [MO.BANG["lean"]] * 4 and [nod[h] for h in hits] == [MO.BANG["nod"]] * 4
    assert min(lean.values()) < 0.0 and res["lean"][0][1] == 0.0 and res["lean"][-1] == [2.7, 0.0]
    half = compile([{"name": "headbang", "t": 0.7, "dur": 2.0, "hits": hits, "depth": 0.5}])
    assert dict(map(tuple, half["head_nod"]))[1.0] == pytest.approx(0.5 * MO.BANG["nod"])
    assert not half["hands"]["L"] and not half["hands"]["R"]             # hands free: no keys of the move's own
    hip = [k for k in res["hands"]["L"] if 0.7 <= k["t"] <= 2.7 and "fingers" in k]
    assert hip and all(k["fingers"] == MV.SHAPES["flat"] for k in hip)  # on the hip
    with pytest.raises(MV.MoveError, match="depth"):
        compile([{"name": "headbang", "t": 0.7, "dur": 2.0, "depth": 3}])
    with pytest.raises(MV.MoveError, match="start it before the hit"):
        compile([{"name": "headbang", "t": 0.82, "dur": 2.0}])


def test_body_channels_overlap_only_when_two_moves_key_different_ones():
    both = compile([{"name": "pound", "t": 1.4, "dur": 3.0, "prop": "mallet"},
                    {"name": "bass_face", "t": 1.5, "dur": 2.0}])          # lean and head_nod: apart
    assert both["lean"] and both["head_nod"]
    with pytest.raises(MV.MoveError, match="head_nod"):
        compile([{"name": "headbang", "t": 0.7, "dur": 2.0}, {"name": "bass_face", "t": 1.5, "dur": 2.0}])


def test_a_pogo_jumps_up_on_each_hit_as_high_as_a_body_flies_in_the_time_it_has():
    res = compile([{"name": "pogo", "t": 0.5, "dur": 3.0}])
    rise, air = res["perform"]["rise"], res["perform"]["air"]
    takeoffs = [b for b in BEATS if 0.5 <= b < 3.5 - MO.POGO["ground"] - MO.POGO["least"]]   # each with time to fly
    assert [round(a["from"] - 0.01, 4) for a in air] == takeoffs
    T = MO.POGO["flight"]
    top = min(MO.POGO["height"], MO.GRAVITY * T * T / 8.0)
    assert max(v for _, v in rise) == pytest.approx(top, abs=1e-4)
    peak = [t for t, v in rise if v == max(v for _, v in rise)]
    assert peak[0] == pytest.approx(takeoffs[0] + 0.5 * T, abs=1e-3)  # a parabola: up and down in its flight
    assert all(a["tuck"] == MO.POGO["tuck"] for a in air) and max(v for _, v in res["perform"]["crouch"]) > 0.03
    L, R = ([k for k in res["hands"][s] if k["t"] >= 0.5] for s in ("L", "R"))
    for kl, kr in zip(L, R):
        assert np.array(kl["at"]) * [-1, 1, 1] == pytest.approx(np.array(kr["at"]), abs=1e-4)
    eights = compile([{"name": "pogo", "t": 0.5, "dur": 1.5, "hits": [0.8225, 1.23375, 1.645]}])
    assert max(v for _, v in eights["perform"]["rise"]) < top          # less time between the hits: lower hops


@pytest.mark.parametrize("routine", list(MO.ROUTINES))
def test_wotagei_holds_a_penlight_in_each_fist_and_plays_its_routine_on_the_beats(routine):
    res = compile([{"name": "wotagei", "t": 0.5, "dur": 2.5, "props": PENS, "routine": routine}])
    for side, prop in PENS.items():
        h = res["held"][prop]
        assert h["bone"] == "wrist" and h["side"] == side
        R = np.array(h["rel"][0])
        thumb = np.array([0.0, 0.0, -1.0 if side == "L" else 1.0])   # mkmmd.core.moves.aimed: the thumb's side
        assert R @ np.array([0.0, 0.0, 1.0]) == pytest.approx(thumb)  # the light out of the thumb's side
        places = {n for n in res["places"][side] if n.endswith("@" + prop)}
        assert places == {f"{p}@{prop}" for p in MO.ROUTINES[routine][0]}
        for n in places:
            assert res["places"][side][n]["wrist_deg"] <= MV.HELD + 1e-6
    keys = {s: [k for k in res["hands"][s] if 0.5 <= k["t"] <= 3.0] for s in ("L", "R")}
    if routine == "punch":                                             # in turns, on the eighths
        out = [k["t"] for k in keys["L"] if k["at"][1] < -0.25]
        assert out and all(any(abs(t - b) < 1e-3 for b in BEATS + [b + 0.41125 for b in BEATS]) for t in out)
    if routine == "eights":                                            # mirror images
        for kl, kr in zip(keys["L"], keys["R"]):
            assert np.array(kl["at"]) * [-1, 1, 1] == pytest.approx(np.array(kr["at"]), abs=1e-4)
    if routine == "arcs":                                              # both lights to one side, then the other
        on = [k for k in keys["L"] if any(abs(k["t"] - b) < 1e-3 for b in BEATS)]
        assert on[0]["at"][0] > 0.15 and on[1]["at"][0] < 0.05
    with pytest.raises(MV.MoveError, match="props"):
        compile([{"name": "wotagei", "t": 0.5, "dur": 2.5, "props": {"L": "pen_l"}}])


def test_a_whistle_pulls_the_cord_down_on_each_hit_the_other_hand_on_the_hip():
    for hand, s in (("R", -1.0), ("L", 1.0)):
        res = compile([{"name": "whistle", "t": 0.5, "dur": 2.0, "hand": hand}])
        keys = {k["t"]: k for k in res["hands"][hand]}
        up, down = keys[0.5], keys[BEATS[1]]
        assert up["at"][2] > down["at"][2] + 0.04 and up["at"][2] > MARKS["eye"][2]   # pulled down from up by the head
        assert s * up["at"][0] > 0.1
        other = "L" if hand == "R" else "R"
        assert res["places"][other]["hip"]["wrist_deg"] <= MV.HELD
        assert res["head_tilt"][1][1] == pytest.approx(5.0 * (1.0 if hand == "L" else -1.0))


def test_a_bass_face_scowls_and_nods_on_each_hit_and_a_pout_has_the_mouth():
    hits = [1.0, 1.3, 2.5]
    res = compile([{"name": "bass_face", "t": 0.8, "dur": 2.2, "hits": hits}])
    faces = {e["morph"]: dict(map(tuple, e["keys"])) for e in res["expressions"]}
    assert set(faces) == {"angry", "jito", "mouth_down"}
    assert all(faces["angry"][round(h + 0.02, 4)] == 0.9 for h in hits)
    assert faces["angry"][round(1.0 + 0.5 * 0.3, 4)] == pytest.approx(MO.ACCENT["low"] * 0.9)   # held between close hits
    nod = dict(map(tuple, res["head_nod"]))
    assert all(nod[round(h + MO.ACCENT["nod_at"], 4)] == MO.ACCENT["nod"] for h in (1.0, 2.5))
    assert "mouth" not in res["perform"]
    pout = compile([{"name": "bass_face", "t": 0.8, "dur": 2.2, "style": "pout"}])
    assert pout["perform"]["mouth"] == [[0.8, 3.0]]
    with pytest.raises(MV.MoveError, match="style"):
        compile([{"name": "bass_face", "t": 0.8, "dur": 2.2, "style": "grin"}])


def test_the_hand_checks_see_the_mosh_moves_hands_at_their_places():
    free = {"L": False, "R": False}
    assert MO.rows({"name": "pogo"}, free) == [("L", ["pump_up"], True), ("R", ["pump_up"], True)]
    assert MO.rows({"name": "headbang"}, free) == []
    assert MO.rows({"name": "headbang", "hands": "hips"}, free) == [("L", ["hip"], False), ("R", ["hip"], False)]
    assert MO.rows({"name": "wotagei", "routine": "eights"}, free) == [("L", ["wota_eight"], True),
                                                                        ("R", ["wota_eight"], True)]
    assert MO.rows({"name": "whistle", "hand": "L"}, free) == [("L", ["cord", "cord_pull"], False),
                                                               ("R", ["hip"], False)]
    assert MO.rows({"name": "bass_face"}, free) == []


def test_a_trip_from_the_hip_up_over_the_head_goes_round_the_shoulder():
    """The straight line from a hand on the hip to fists up over the head passes a few cm from the shoulder, where the
    arm would fold through itself: a key halfway round the shoulder keeps the arm out."""
    res = compile([{"name": "headbang", "t": 0.7, "dur": 1.5, "hands": "hips"}, {"name": "pogo", "t": 2.5, "dur": 1.6}])
    S = np.array(MARKS["arm.R"])
    keys = [k for k in res["hands"]["R"] if 2.0 <= k["t"] <= 2.5]
    assert any("fingers" not in k for k in keys)                       # a key halfway
    path = [np.array(k["at"]) for k in res["hands"]["R"] if 1.9 <= k["t"] <= 2.5]
    for a, b in zip(path, path[1:]):
        g = b - a
        u = min(max(-float((a - S) @ g) / float(g @ g), 0.0), 1.0)
        assert np.linalg.norm(a - S + u * g) > 0.12                    # no straight stretch passes by the shoulder
