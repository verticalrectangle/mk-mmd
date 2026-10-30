"""Moves, bpy-free (mkmmd.core.moves): `[[move.<cast>]]` entries compiled to hand keys, lean / tilt keys, twitches and
expressions, every hand place made natural (the wrist against the forearm an arm IK gives, a mic aimed, the hand kept off
the body). The Blender half (the keys laid into the pose and perform tables, a move held on a real arm) is checked in
tests/test_build_stages.py."""
import math

import numpy as np
import pytest

from mkmmd.core import armreach as AR
from mkmmd.core import armtwist as AT
from mkmmd.core import moves as MV

MARKS = {"arm.L": [0.093, -0.02, 1.25], "arm.R": [-0.093, -0.02, 1.25], "chest": [0.0, -0.065, 1.12],
         "mouth": [0.0, -0.071, 1.35], "eye": [0.0, -0.045, 1.41], "reach": 0.37, "upper": 0.195, "fore": 0.176,
         "hand": 0.146}
BEATS = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
PLACES = ("rest", "dainty", "dainty_one", "mic", "mic_lens", "mic_up", "mic_across", "chest", "bounce", "point", "heart",
          "peace", "paw", "sparkle", "up", "drip_in", "drip_out", "ears", "heart_push", "bunny", "bonk_up", "bonk", "hair",
          "hair_out", "hip", "gun", "float_low", "float", "peek", "butterfly", "butterfly_shut", "bud", "bloom", "reach",
          "wings_up", "wings_down", "blow", "blow_out", "giggle", "behind", "behind_one", "clasp", "clasp_one", "skirt")
DREAMY = ("peek", "butterfly", "bloom", "reach", "wings", "blow", "giggle")      # the moves of the dreamy set


def near(p):
    return pytest.approx(p, abs=1e-4)                                   # the keys are rounded to 0.1 mm


def ball(centre, radius):
    """clear(points) of a ball: the smallest distance from the points to it (negative inside)."""
    c = np.asarray(centre, float)
    return lambda P: float((np.linalg.norm(np.asarray(P, float).reshape(-1, 3) - c, axis=1) - radius).min())


def placed(keys):
    """A hand's keys at its places (a key halfway through a trip carries no fingers: they ease through it)."""
    return [k for k in keys if "fingers" in k]


def body(scale=1.0):
    """Landmarks of a body `scale` times MARKS (positions and lengths)."""
    return {k: (np.asarray(v, float) * scale).tolist() if isinstance(v, list) else v * scale for k, v in MARKS.items()}


# ---------------------------------------------------------------- the arm and the wrist
@pytest.mark.parametrize("W", [(0.25, -0.25, 1.1), (0.1, -0.3, 1.45), (0.25, 0.05, 1.0)])
def test_the_ik_elbow_keeps_both_lengths_and_bends_toward_the_pole(W):
    S, a, b = np.array([0.093, -0.02, 1.25]), 0.195, 0.176
    P = S + np.array(AR.POLE) * [1, 1, -1]
    E = AR.pole_elbow(S, W, a, b, P)
    assert np.linalg.norm(E - S) == pytest.approx(a) and np.linalg.norm(E - np.array(W)) == pytest.approx(b)
    u = (np.array(W) - S) / np.linalg.norm(np.array(W) - S)
    side = lambda X: (X - S) - ((X - S) @ u) * u                          # noqa: E731
    assert side(E) @ side(P) > 0                                        # on the pole's side of the shoulder-wrist line
    far = AR.pole_elbow(S, S + [0.5, 0, 0], a, b, P)                     # out of reach: the straight arm
    assert far == pytest.approx(S + [a, 0, 0])


@pytest.mark.parametrize("scale", [0.85, 1.0, 1.2])
@pytest.mark.parametrize("name", PLACES)
def test_no_place_bends_a_wrist_past_the_limit(name, scale):
    m = body(scale)
    for side in ("L", "R"):
        pt, d, p, _, _, bend, pole = MV.resolve(name, side, m)
        assert bend <= MV.WRIST + 1e-6
        assert bend == pytest.approx(MV.bend(d, MV.forearm(side, m, pt, pole)), abs=1e-6)
        assert abs(d @ p) < 1e-9 and np.linalg.norm(d) == pytest.approx(1) and np.linalg.norm(p) == pytest.approx(1)


@pytest.mark.parametrize("name", PLACES)
def test_the_right_hand_is_the_left_s_mirror_image_in_every_place(name):
    """Mirror images, the butterfly's crossed wrists apart: its right hand is MV.CROSS in front of the left."""
    clear = ball((0.0, -0.02, 1.0), 0.17)                                # a body to keep off, the same for both
    ahead = np.array([0.0, MV.CROSS["butterfly"] if name.startswith("butterfly") else 0.0, 0.0])
    for c in (None, clear):
        (pl, dl, ql, *_), (pr, dr, qr, *_) = (MV.resolve(name, s, MARKS, c) for s in ("L", "R"))
        M = np.array([-1.0, 1.0, 1.0])
        assert pl * M - ahead == pytest.approx(pr) and dl * M == pytest.approx(dr) and ql * M == pytest.approx(qr)


@pytest.mark.parametrize("side, s", [("L", 1.0), ("R", -1.0)])
@pytest.mark.parametrize("name", sorted(MV.AIM))
def test_a_hand_holding_a_mic_up_or_out_has_its_palm_toward_the_body_never_twisted_out(name, side, s):
    for scale in (0.85, 1.0, 1.2):
        _, _, p, *_ = MV.resolve(name, side, body(scale))
        assert s * p[0] < -0.3                                             # the palm toward the midline


def test_a_hand_bent_too_far_turns_toward_its_forearm_as_a_whole():
    """... to HELD, the wrist rule less the few degrees the IK poses an arm off the plan (the `wrist_bend` check holds
    the built hand to WRIST)."""
    f = np.array([0.0, 0.0, 1.0])
    d, p = MV.natural([0.0, -1.0, -1.0], [0.0, -1.0, 1.0], f)            # 135 degrees down: a paw past breaking
    assert MV.bend(d, f) == pytest.approx(MV.HELD) and MV.HELD < MV.WRIST
    assert d @ p == pytest.approx(0.0, abs=1e-9)                       # the palm turned with it
    assert MV.natural([0.0, -0.5, 1.0], [0.0, 1.0, 0.5], f)[0] == pytest.approx(np.array([0, -0.5, 1]) / math.sqrt(1.25))


def _world(name, side, m, part):
    """Where the place `name` puts the hand `side`'s posed point `part` (MV.posed, in the left hand's own frame)."""
    pt, d, p, *_ = MV.resolve(name, side, m)
    return pt + MV._off(part, d, p, side)


@pytest.mark.parametrize("chest_z", [1.18, 1.12, 1.05])               # a long torso bends the heart's wrists past the limit
def test_the_two_hand_heart_keeps_its_shape_its_hands_mirrored_and_meeting_on_any_body(chest_z):
    """Each hand's index fingertip at the top and its thumb's tip at the bottom of one heart, on the body's midline
    (half a finger's radius off it, so the two touch), the heart upright and facing the lens, as written."""
    m = dict(MARKS, chest=[0.0, -0.065, chest_z])
    (pl, dl, ql, _, _, bl, _), (pr, dr, qr, _, _, br, _) = (MV.resolve("heart_push", s, m) for s in ("L", "R"))
    assert pl == pytest.approx([-pr[0], pr[1], pr[2]]) and dl == pytest.approx([-dr[0], dr[1], dr[2]])
    po = MV.posed(m, "heart2")
    r = MV.FG.RADIUS * float(np.linalg.norm(np.diff(MV.hand_rest(m)["points"]["middle"], axis=0), axis=1).sum())
    for side in ("L", "R"):
        top, low = (_world("heart_push", side, m, po["tips"][f]) for f in ("index", "thumb"))
        assert abs(top[0]) == pytest.approx(0.5 * r, abs=1e-6) and abs(low[0]) == pytest.approx(0.5 * r, abs=1e-6)
        assert (top - low) / np.linalg.norm(top - low) == pytest.approx([0.0, 0.0, 1.0], abs=1e-6)   # upright
        _, wd, wp, _ = MV.place("heart_push", side, m)
        assert MV.resolve("heart_push", side, m)[1] == pytest.approx(wd / np.linalg.norm(wd))   # as written, never turned
    assert max(bl, br) <= MV.WRIST


@pytest.mark.parametrize("side", ["L", "R"])
@pytest.mark.parametrize("name", sorted(MV.AIM))
def test_an_aimed_mic_points_where_aimed_across_a_palm_that_continues_the_forearm(name, side):
    pt, d, p, _, _, bend, pole = MV.resolve(name, side, MARKS)
    sigma = 1.0 if side == "R" else -1.0
    a = MV.AIM[name]
    aim = np.array([(1.0 if side == "L" else -1.0) * a[0], a[1], a[2]])
    assert sigma * np.cross(d, p) == pytest.approx(aim / np.linalg.norm(aim))   # the thumb side, as at the mic place
    f = MV.forearm(side, MARKS, pt, pole)                                  # the place's own elbow (a fist at the eye)
    assert bend == pytest.approx(90.0 - math.degrees(math.acos(abs(f @ aim) / np.linalg.norm(aim))), abs=1e-6)


# ---------------------------------------------------------------- the hands off the body
def test_a_hand_coming_in_stops_at_its_margin_and_one_clear_of_the_body_stays():
    clear = ball((0.0, 0.0, 1.2), 0.1)
    box = MV.hand_box([0, 0, 1], [0, 1, 0], 0.15, "flat")
    hand = lambda w: w + box                                              # noqa: E731
    path = MV.line([0.0, -0.05, 1.12], [0, -1, 0])                        # to a wrist inside the ball
    s = MV.sweep(path, hand, clear, 0.01, MV.FAR, MV.STEP)
    assert clear(hand(path(s))) == pytest.approx(0.01, abs=1e-4) and s > 0.05
    assert MV.sweep(MV.line([0.0, -0.3, 1.12], [0, -1, 0]), hand, clear, 0.01, MV.FAR, MV.STEP) == 0.0


def test_a_hand_stops_in_front_of_what_is_in_its_way_not_where_it_was_asked():
    hair = ball((0.0, -0.12, 1.3), 0.04)                                  # a lock of hair in front of the place
    box = MV.hand_box([0, 0, 1], [0, 1, 0], 0.15, "curled")
    path = MV.line([0.0, -0.02, 1.25], [0, -1, 0])                        # asked for behind it, clear of it
    s = MV.sweep(path, lambda w: w + box, hair, 0.01, MV.FAR, MV.STEP)
    assert path(s)[1] < -0.12 - 0.04 and s > 0                           # in front of it, never through it


def test_an_arm_held_out_swings_clear_of_a_skirt_and_stays_in_reach():
    skirt = ball((0.0, -0.02, 0.62), 0.3)                                 # a wide skirt round a hanging hand
    m = dict(MARKS)
    pt0 = MV.place("rest", "L", m)[0]
    pt, d, p, shape, moved, _, _ = MV.resolve("rest", "L", m, skirt)
    assert skirt(pt0 + MV.hand_box(d, p, m["hand"], shape)) < MV.MARGIN < moved           # it was in the skirt
    assert skirt(pt + MV.hand_box(d, p, m["hand"], shape)) == pytest.approx(MV.MARGIN, abs=1e-3)
    S = np.asarray(m["arm.L"])
    assert np.linalg.norm(pt - S) == pytest.approx(np.linalg.norm(pt0 - S))   # as far from the shoulder: in reach
    assert pt[0] > pt0[0]                                                  # out to its side, up off the skirt


def test_touching_places_end_on_the_body_and_the_others_keep_their_room():
    head = ball(MARKS["eye"], 0.11)                                       # a head as big as the hair round it
    torso = ball((0.0, -0.02, 1.15), 0.12)
    clear = lambda P: min(head(P), torso(P))                             # noqa: E731
    for name in ("chest", "ears"):
        pt, d, p, shape, moved, _, _ = MV.resolve(name, "L", MARKS, clear)
        assert moved > 0 and clear(pt + MV.hand_box(d, p, MARKS["hand"], shape)) == pytest.approx(MV.TOUCH_MARGIN, abs=5e-4)
    big = ball(MARKS["mouth"], 0.2)                                       # a place by the face inside a big hood
    pt, d, p, shape, moved, _, _ = MV.resolve("heart", "L", MARKS, big)
    assert moved > 0 and big(pt + MV.hand_box(d, p, MARKS["hand"], shape)) == pytest.approx(MV.MARGIN, abs=5e-4)
    assert MV.resolve("point", "L", MARKS, clear)[4] == 0.0                # clear of the body: where it was asked


def test_places_by_the_face_and_chest_stay_on_their_own_side_clear_of_a_mic_at_the_mouth():
    for name in ("chest", "heart", "paw"):
        for side, s in (("L", 1.0), ("R", -1.0)):
            pt = MV.place(name, side, MARKS)[0]
            assert s * pt[0] - MV.WIDTH * MARKS["hand"] > 0.02, (name, side)    # the hand's inner edge off the midline


# ---------------------------------------------------------------- the clock
def test_a_move_holds_its_last_place_to_its_end_and_the_hand_goes_home_after_it():
    out = MV.compile([{"name": "point", "t": 1.0, "dur": 0.8, "hand": "L"}], MARKS)
    keys = placed(out["hands"]["L"])
    home, point = MV.resolve("rest", "L", MARKS), MV.resolve("point", "L", MARKS)
    there = MV.travel(home, point)
    assert there > MV.TRAVEL_MIN + 0.1                                    # a hand by the hip to the arm held out
    assert [k["t"] for k in keys] == pytest.approx([1.0 - there, 1.0, 1.8, 1.8 + MV.travel(point, home)], abs=1e-4)
    assert [k["at"] for k in keys] == [near(p[0]) for p in (home, point, point, home)]
    assert keys[1]["fingers"] == "point" and keys[0]["fingers"] == "relaxed"
    assert set(out["places"]["L"]) == {"rest", "point"}


def test_moves_close_together_go_straight_on_and_far_apart_go_home_between():
    def ats(gap):
        es = [{"name": "point", "t": 1.0, "dur": 0.5, "hand": "L"},
              {"name": "heart_wink", "t": 1.5 + gap, "dur": 0.5, "hand": "L"}]
        return [k["at"] for k in MV.compile(es, MARKS)["hands"]["L"]]
    home = near(MV.resolve("rest", "L", MARKS)[0])
    assert ats(0.2).count(home) == 2                                      # in before the first, out after the last
    assert ats(1.2).count(home) == 4                                      # and home between them


def test_a_hand_travels_at_a_human_pace_and_sets_off_early_to_arrive_on_time():
    """Moves laid end to end (a point held to 1.5 s, a heart at 1.52 s) made the hand jump between them within a frame."""
    es = [{"name": "point", "t": 1.0, "dur": 0.5, "hand": "L"}, {"name": "heart_wink", "t": 1.52, "dur": 0.4, "hand": "L"},
          {"name": "peace_eye", "t": 2.1, "dur": 0.3, "hand": "L"}]
    keys = placed(MV.compile(es, MARKS)["hands"]["L"])
    places = {n: MV.resolve(n, "L", MARKS) for n in ("rest", "point", "heart", "peace")}
    at = lambda k: next(n for n, r in places.items() if k["at"] == near(r[0]))   # noqa: E731
    for a, b in zip(keys, keys[1:]):                                      # never faster than a quick gesture, on average
        assert np.linalg.norm(np.subtract(b["at"], a["at"])) <= 1.6 * (b["t"] - a["t"]) + 1e-3, (at(a), at(b))
        assert b["t"] - a["t"] >= MV.TRAVEL_MIN - 1e-3, (at(a), at(b))
    assert [k["t"] for k in keys if at(k) == "heart"][0] == pytest.approx(1.52)       # on its time
    assert [k["t"] for k in keys if at(k) == "peace"][0] == pytest.approx(2.1)
    assert [at(k) for k in keys] == ["rest", "point", "point", "heart", "heart", "peace", "peace", "rest"]


def test_a_move_that_leaves_the_hand_too_little_time_to_get_there_is_refused():
    with pytest.raises(MV.MoveError, match=r"the L hand needs 0\.\d\d s to get from 'point' \(at 1 s\) to 'heart'"):
        MV.compile([{"name": "point", "t": 1.0, "dur": 0.1, "hand": "L"},
                    {"name": "heart_wink", "t": 1.15, "dur": 0.4, "hand": "L"}], MARKS)     # 18 cm in 0.15 s


def test_one_hand_cannot_play_two_moves_at_once():
    with pytest.raises(MV.MoveError, match="the L hand plays two moves at once"):
        MV.compile([{"name": "point", "t": 1.0, "dur": 1.0, "hand": "L"},
                    {"name": "heart_wink", "t": 1.5, "dur": 1.0, "hand": "L"}], MARKS)
    out = MV.compile([{"name": "point", "t": 1.0, "dur": 1.0, "hand": "L"},
                      {"name": "point", "t": 1.5, "dur": 1.0, "hand": "R"}], MARKS)
    assert out["hands"]["L"] and out["hands"]["R"]                        # the other hand may


def test_a_beat_move_pats_on_each_beat_inside_it_and_needs_one():
    keys = MV.compile([{"name": "chest_pat", "t": 0.9, "dur": 1.0, "hand": "R"}], MARKS, BEATS)["hands"]["R"]
    on = MV.resolve("chest", "R", MARKS)[0]
    pats = [k["t"] for k in keys if k["at"] == near(on)]
    assert pats == pytest.approx([1.0, 1.5])                              # 0.5 and 2.0 are outside it
    with pytest.raises(MV.MoveError, match="no beat between"):
        MV.compile([{"name": "chest_pat", "t": 1.1, "dur": 0.3}], MARKS, BEATS)


def test_a_beat_move_on_the_body_pats_from_the_surface_it_was_brought_to():
    torso = ball((0.0, -0.02, 1.15), 0.12)
    keys = MV.compile([{"name": "chest_pat", "t": 0.9, "dur": 1.0, "hand": "L"}], MARKS, BEATS, clear=torso)["hands"]["L"]
    on = MV.resolve("chest", "L", MARKS, torso)[0]
    pats = [k for k in keys if k["at"] == near(on)]
    offs = [k for k in keys if k["at"] == near(on + np.array(MV.MOVES["chest_pat"]["off"]))]
    assert len(pats) == 2 and len(offs) >= 2                               # on the chest, then lifted off it


def test_a_bonk_winds_up_then_strikes_on_its_own_hits_not_the_beats_and_they_are_checked():
    es = [{"name": "bonk", "t": 0.9, "dur": 0.6, "hand": "R", "hits": [1.1, 1.3]}]
    keys = MV.compile(es, MARKS, BEATS)["hands"]["R"]
    up, hit = (near(MV.resolve(p, "R", MARKS)[0]) for p in ("bonk_up", "bonk"))
    assert [k["t"] for k in keys if k["at"] == hit] == pytest.approx([1.1, 1.3])          # not on the beat at 1.0
    assert [k["t"] for k in keys if k["at"] == up] == pytest.approx([0.9, 1.2, 1.4, 1.5])  # cocked first, between, after
    for bad, frag in (([1.1, 1.6], "inside it"), ([], "list of clip seconds"), ("1.1", "list of clip seconds")):
        with pytest.raises(MV.MoveError, match=frag):
            MV.compile([dict(es[0], hits=bad)], MARKS, BEATS)
    with pytest.raises(MV.MoveError, match="hits are for a move played on beats"):
        MV.compile([{"name": "point", "t": 1.0, "hits": [1.2]}], MARKS, BEATS)


@pytest.mark.parametrize("side", ["L", "R"])
def test_a_bonk_is_cocked_up_over_the_shoulder_and_its_face_comes_forward_and_down(side):
    up, du, pu, _, _, bend_up, _ = MV.resolve("bonk_up", side, MARKS)
    pt, d, _, _, _, bend, _ = MV.resolve("bonk", side, MARKS)
    handle = (1.0 if side == "R" else -1.0) * np.cross(du, pu)    # the fist's hole: where the handle points
    assert up[2] > MARKS[f"arm.{side}"][2] + 0.1 and handle[1] > 0.3 and handle[2] > 0.3   # raised, the hammer up and back
    assert d[1] < -0.5 and d[2] < -0.2                     # the hand, so the head's face, points forward and down
    assert pt[2] < MARKS["chest"][2] and max(bend, bend_up) <= MV.WRIST


# ---------------------------------------------------------------- the mic hand
def test_a_two_hand_move_leaves_the_mic_hand_at_the_mic_unless_it_raises_it():
    mic = near(MV.resolve("mic", "R", MARKS)[0])
    out = MV.compile([{"name": "rest", "hand": "R", "place": "mic"}, {"name": "sparkle", "t": 1.0, "dur": 0.5}], MARKS)
    assert out["hands"]["L"] and [k["at"] for k in out["hands"]["R"]] == [mic]      # held there from the start
    up = MV.compile([{"name": "rest", "hand": "R", "place": "mic"}, {"name": "hands_up", "t": 1.0}], MARKS)["hands"]
    assert placed(up["R"])[1]["at"] == near(MV.resolve("up", "R", MARKS)[0])
    assert {k["fingers"] for k in placed(up["R"])} == {"curled"} and placed(up["L"])[1]["fingers"] == "flat"
    with pytest.raises(MV.MoveError, match="needs both hands free, and the R hand holds the mic"):
        MV.compile([{"name": "rest", "hand": "R", "place": "mic"}, {"name": "heart_push", "t": 1.0}], MARKS)
    with pytest.raises(MV.MoveError, match="rest"):
        MV.compile([{"name": "rest", "hand": "R", "place": "chest"}], MARKS)


def test_a_reaction_leaves_on_its_own_time_from_where_the_hand_is_and_is_thrown_there():
    """Thrown up by the blast, the arms would have set off before it, as a planned move does to arrive on time: a
    reaction (`flung`) leaves on its time from the place the hand holds (the mic held across, kept until then) or from
    home, arrives at a thrown pace, and goes home when it ends (the hands come down as they land)."""
    es = [{"name": "rest", "hand": "R", "place": "mic"}, {"name": "mic_across", "t": 8.4, "dur": 0.992, "hand": "R"},
          {"name": "flung", "t": 9.392, "dur": 0.96}]
    hands = MV.compile(es, MARKS)["hands"]
    across, up, mic = (MV.resolve(n, "R", MARKS) for n in ("mic_across", "up", "mic"))
    R = placed(hands["R"])
    assert max(k["t"] for k in R if k["at"] == near(across[0])) == pytest.approx(9.392)
    arrive = min(k["t"] for k in R if k["at"] == near(up[0]))
    assert arrive == pytest.approx(9.392 + MV.fling(across, up), abs=1e-4) and arrive - 9.392 < MV.travel(across, up)
    assert R[-1]["at"] == near(mic[0]) and R[-1]["t"] > 9.392 + 0.96 and {k["fingers"] for k in R} == {"curled"}
    home, up_l, L = MV.resolve("rest", "L", MARKS), MV.resolve("up", "L", MARKS), placed(hands["L"])
    assert L[0]["at"] == near(home[0]) and L[0]["t"] == pytest.approx(9.392)
    assert L[1]["at"] == near(up_l[0]) and L[1]["t"] == pytest.approx(9.392 + MV.fling(home, up_l), abs=1e-3)


def test_a_hand_that_would_fold_over_its_wrist_on_its_way_passes_a_key_halfway_turned_back():
    """Going from beside the hip up into the hair (the hair flip) the hand folded back over its wrist: only the two ends'
    orientations were keyed, and they were interpolated while the arm swung round under them. The keys halfway go round
    the shoulder at the mean of the two ends' distances from it: on the straight line from the hip up behind the head the
    wrist passed 8 cm from the shoulder, an arm folded so far that the upper arm turned past its range whichever way the
    elbow pointed (the IK wrung it round through 180 degrees); round the shoulder every key keeps both twists in range."""
    keys = MV.compile([{"name": "hair_flip", "t": 1.0, "dur": 1.5, "hand": "L"}], MARKS)["hands"]["L"]
    vias = [i for i, k in enumerate(keys) if "fingers" not in k]
    S = np.asarray(MARKS["arm.L"], float)

    def round_mid(a, b):
        ra, rb = np.array(a["at"]) - S, np.array(b["at"]) - S
        la, lb = np.linalg.norm(ra), np.linalg.norm(rb)
        return S + 0.5 * (la + lb) * MV._unit(ra / la + rb / lb)
    assert vias
    for i in vias:                                                      # halfway between two keys about it (the halves
        v = keys[i]                                                     # of a long trip get keys of their own)
        assert any(v["t"] == pytest.approx(0.5 * (a["t"] + b["t"]), abs=1e-4) and v["at"] == near(round_mid(a, b))
                   for a in keys[:i] for b in keys[i + 1:])
        f = MV.forearm("L", MARKS, v["at"], v.get("pole"))
        assert MV.bend(v["dir"], f) <= MV.WRIST + 0.5                                              # turned back in reach
        assert np.linalg.norm(np.array(v["at"]) - S) > 0.2                                         # clear of the shoulder
    assert all(MV.overtwist(MV.twisted("L", MARKS, k["at"], k["dir"], k["palm"], k.get("pole")), 0.0) == 0.0
               for k in keys)
    calm = MV.compile([{"name": "point", "t": 1.0, "dur": 1.0, "hand": "L"}], MARKS)["hands"]["L"]
    assert all("fingers" in k for k in calm)                               # a trip that does not fold needs none


def test_a_mic_at_the_mouth_leans_back_up_to_the_lips_from_a_fist_under_them_so_the_elbow_hangs():
    """The mic was held upright from a fist turned square across the face, so the forearm lay level and the elbow stuck
    out at the height of the shoulder."""
    S = np.asarray(MARKS["arm.R"], float)
    pt, d, p, _, _, bend, _ = MV.resolve("mic", "R", MARKS)
    E = AR.pole_elbow(S, pt, MARKS["upper"], MARKS["fore"], S + np.array(AR.POLE) * [-1, 1, -1])
    assert S[2] - E[2] > 0.12 and abs(E[0] - S[0]) < 0.1 and bend < 20.0   # the elbow hangs by the side, the wrist straight
    mic = np.cross(d, p)                                                   # the right fist's hole: toward the grille
    up = np.asarray(MARKS["mouth"], float) - pt
    assert pt[2] < MARKS["mouth"][2] - 0.08 and mic @ up / np.linalg.norm(up) > 0.9   # under the lips, aimed up at them


def test_a_dainty_hand_waits_lightly_on_the_front_of_the_body_and_goes_back_there():
    torso = ball((0.0, -0.02, 0.9), 0.15)                                  # a skirt round the hips
    both = [{"name": "rest", "hand": "L", "place": "dainty"}, {"name": "rest", "hand": "R", "place": "dainty"}]
    keys = MV.compile(both + [{"name": "point", "t": 1.0, "dur": 0.5, "hand": "L"}], MARKS, clear=torso)["hands"]["L"]
    home = MV.resolve("dainty", "L", MARKS, torso)
    assert keys[0]["at"] == near(home[0]) and keys[-1]["at"] == near(home[0])
    assert torso(home[0] + MV.hand_box(home[1], home[2], MARKS["hand"], home[3])) == pytest.approx(MV.TOUCH_MARGIN, abs=5e-4)
    assert home[0][1] < MARKS["chest"][1] and 0 < home[0][0] < MARKS["arm.L"][0] + 0.02   # in front, inside the shoulder
    waits = MV.compile(both, MARKS)["hands"]["L"]
    assert len(waits) == 1 and waits[0]["t"] == 0.0                         # it waits there from the start


def test_a_dainty_hand_alone_rests_on_its_own_thigh_its_wrist_straight_and_its_elbow_out():
    """Two hands resting on the skirt front ("dainty") are a pose for two: alone there a hand bent in at the wrist and its
    forearm lay on the front of the body. A dainty hand alone (its partner away on a move, or resting elsewhere) rests on
    its own thigh ("dainty_one"), a few cm from the two-hand spot, the hand continuing its forearm and the elbow out to the
    side (its key carries the elbow's pole); it glides there as its partner leaves, arriving as the partner gets to its
    move, and is back on the spot as the partner is, and a short return of the partner between two moves does not send
    it back for a moment."""
    both = [{"name": "rest", "hand": "L", "place": "dainty"}, {"name": "rest", "hand": "R", "place": "dainty"}]
    names = ("dainty", "dainty_one", "rest")
    res = {n: MV.resolve(n, "R", MARKS) for n in names}
    one, two = res["dainty_one"], res["dainty"]
    assert one[5] <= MV.STRAIGHT["dainty_one"] + 1e-6 and one[5] < two[5] - 10       # the hand continues its forearm
    assert np.linalg.norm(one[0] - two[0]) < 0.09 and one[0][1] < MARKS["chest"][1]   # a few cm away, in front
    out = lambda r, n: -MV.elbow("R", MARKS, r[0], r[6])[0]                            # noqa: E731 (her right: -x)
    assert out(one, "dainty_one") > max(out(two, "dainty"), -MARKS["arm.R"][0] + 0.05)  # the elbow out to the side

    def where(entries):
        keys = placed(MV.compile(entries, MARKS)["hands"]["R"])
        return [(k["t"], next((n for n in names if k["at"] == near(res[n][0])), "?"), k.get("pole")) for k in keys]
    seen = where(both + [{"name": "point", "t": 2.0, "dur": 1.0, "hand": "L"}])
    assert seen[0][1] == "dainty" and seen[-1][1] == "dainty" and {p for _, p, _ in seen} == {"dainty", "dainty_one"}
    point, home = MV.resolve("point", "L", MARKS), MV.resolve("dainty", "L", MARKS)
    lone = [t for t, p, _ in seen if p == "dainty_one"]
    left = max(t for t, p, _ in seen if p == "dainty" and t < lone[0])
    assert left == pytest.approx(2.0 - MV.travel(home, point), abs=1e-3)              # it leaves as the left does ...
    assert lone[0] == pytest.approx(2.0, abs=1e-3)                                     # ... is there as it points
    assert seen[-1][0] == pytest.approx(3.0 + MV.travel(point, home), abs=1e-3)        # and is back with it
    assert all((pl is not None) == (p == "dainty_one") for _, p, pl in seen)           # its elbow held out there only
    gap = where(both + [{"name": "point", "t": 2.0, "dur": 0.6, "hand": "L"},
                        {"name": "heart_wink", "t": 3.6, "dur": 0.5, "hand": "L"}])
    assert not [t for t, p, _ in gap if p == "dainty" and 2.0 < t < 4.1]               # no flip back between them
    alone = where([both[1], {"name": "point", "t": 2.0, "dur": 1.0, "hand": "L"}])
    assert {p for _, p, _ in alone} == {"dainty_one"}                                  # its partner rests elsewhere


def test_the_member_s_yaw_turns_the_hand_directions_into_the_world():
    k0 = MV.compile([{"name": "point", "t": 1.0, "hand": "L"}], MARKS)["hands"]["L"][1]
    k90 = MV.compile([{"name": "point", "t": 1.0, "hand": "L"}], MARKS, yaw=90.0)["hands"]["L"][1]
    assert k0["at"] == k90["at"]                                          # places stay in the member's own frame
    assert k90["dir"] == pytest.approx([-k0["dir"][1], k0["dir"][0], k0["dir"][2]], abs=1e-4)


# ---------------------------------------------------------------- faces and the body
def test_faces_close_together_hold_through_and_a_face_needs_a_morph():
    out = MV.compile([{"name": "face", "t": 1.0, "dur": 0.5, "morph": "smile_eyes", "value": 0.5},
                      {"name": "face", "t": 1.55, "dur": 0.5, "morph": "smile_eyes"}], MARKS)
    (ex,) = out["expressions"]
    assert ex["morph"] == "smile_eyes" and len(ex["keys"]) == 4           # one held span, not two
    assert max(v for _, v in ex["keys"]) == 1.0
    with pytest.raises(MV.MoveError, match="needs `morph`"):
        MV.compile([{"name": "face", "t": 1.0}], MARKS)


def test_body_moves_ease_in_and_out_and_two_at_once_are_refused():
    out = MV.compile([{"name": "lean_back", "t": 1.0, "dur": 1.0}], MARKS)
    assert [v for _, v in out["lean"]] == [0.0, -11.0, -11.0, 0.0] and out["lean"][-1][0] == 2.0
    with pytest.raises(MV.MoveError):
        MV.compile([{"name": "lean_back", "t": 1.0, "dur": 1.0}, {"name": "into_lens", "t": 1.5, "dur": 1.0}], MARKS)
    with pytest.raises(MV.MoveError):                                     # a head turn and a hip tilt at once, too
        MV.compile([{"name": "over_shoulder", "t": 1.0, "dur": 1.0}, {"name": "hand_hip", "t": 1.9, "dur": 1.0}], MARKS)


@pytest.mark.parametrize("name", ["hair_flip", "over_shoulder", "hand_hip", "finger_gun", "peek", "reach", "giggle"])
def test_a_pose_with_the_right_hand_turns_the_other_way_and_leans_the_same(name):
    left = MV.compile([{"name": name, "t": 1.0, "dur": 0.8, "hand": "L"}], MARKS)
    right = MV.compile([{"name": name, "t": 1.0, "dur": 0.8, "hand": "R"}], MARKS)
    turned = [k for k in MV.MIRRORED if left[k]]
    assert turned                                                          # every pose turns or tilts something
    for k in turned:
        assert [v for _, v in right[k]] == [-v for _, v in left[k]]
    assert right["lean"] == left["lean"]
    assert [t["deg"] for t in right["twitch"]] == [-t["deg"] for t in left["twitch"]]


def test_the_finger_gun_winks_the_eye_away_from_the_hand_and_points_ahead_at_eye_height():
    for hand, eye in (("R", "wink_l"), ("L", "wink_r")):
        out = MV.compile([{"name": "finger_gun", "t": 1.0, "hand": hand}], MARKS)
        assert eye in [x["morph"] for x in out["expressions"]]
        held = next(k for k in out["hands"][hand] if k["t"] == 1.0)
        assert held["dir"][1] < -0.9 and held["fingers"] == MV.SHAPES["gun"]  # out in front, the gun shape
        assert held["at"][2] > MARKS["arm.L"][2]                              # raised to aim, above the shoulder joint


def test_a_hand_on_the_hip_comes_in_from_the_side_and_rests_on_the_body():
    hips = ball((0.0, -0.02, 1.0), 0.13)
    pt, d, p, shape, moved, _, _ = MV.resolve("hip", "L", MARKS, hips)
    assert hips(pt + MV.hand_box(d, p, MARKS["hand"], shape)) == pytest.approx(MV.TOUCH_MARGIN, abs=5e-4)
    assert pt[0] > 0.1 and pt[2] < MARKS["chest"][2]                         # on her own side, below the chest


# ---------------------------------------------------------------- the dreamy set
@pytest.mark.parametrize("name", DREAMY)
def test_each_move_of_the_dreamy_set_plays_its_places_on_its_time_at_a_human_pace(name):
    """A move of the dreamy set (a ring at the eye, butterfly wings, a flower opening, a reach to the lens, wings, a blow,
    a giggle) puts its hands at its places on its time with its fingers (a beat move on the beats inside it, `wings` on
    every second one), both hands for a two-hand move, mirror images of each other; the hand comes from its rest place
    and goes back to it at the human pace and never leaves the arm's reach."""
    m = MV.MOVES[name]
    out = MV.compile([{"name": name, "t": 1.0, "dur": 1.6, "hand": "L"}], MARKS, BEATS)
    sides = ("L", "R") if m.get("both") else ("L",)
    assert [sd for sd in "LR" if out["hands"][sd]] == list(sides)
    if m["hand"] == "places":
        want = [(1.0 + share * 1.6, p) for p, share in m["places"]]
    else:
        bs = [b for b in BEATS if 1.0 <= b < 2.6][::m.get("every", 1)]
        want = [(1.0, m["off_place"])] if bs[0] > 1.0 else []          # the wind-up
        for b, nb in zip(bs, bs[1:] + [2.6]):
            want += [(b, m["place"]), (0.5 * (b + nb), m["off_place"])]
    for sd in sides:
        keys = placed(out["hands"][sd])
        for t, p in want:
            r = MV.resolve(p, sd, MARKS)
            k = next(k for k in keys if k["t"] == pytest.approx(t))
            assert k["at"] == near(r[0]) and k["fingers"] == MV.SHAPES[r[3]], (sd, t, p)
        for a, b in zip(keys, keys[1:]):
            assert np.linalg.norm(np.subtract(b["at"], a["at"])) <= 1.6 * (b["t"] - a["t"]) + 1e-3, (a["t"], b["t"])
        S = np.asarray(MARKS[f"arm.{sd}"])
        assert all(np.linalg.norm(np.subtract(k["at"], S)) <= MARKS["upper"] + MARKS["fore"] for k in keys)
        home = near(MV.resolve("rest", sd, MARKS)[0])
        assert keys[0]["at"] == home and keys[-1]["at"] == home and keys[-1]["t"] > 2.6
    if m.get("both"):                       # mirror images at the move's places (a crossed pair: the right hand ahead)
        (L, R), X = (placed(out["hands"][sd]) for sd in "LR"), np.array([-1.0, 1.0, 1.0])
        ahead = np.array([0.0, MV.CROSS.get(name, 0.0), 0.0])
        at = [(t, p) for t, p in want]
        for t, _ in at:
            kl, kr = (next(k for k in keys if k["t"] == pytest.approx(t)) for keys in (L, R))
            assert np.asarray(kl["at"]) * X - ahead == near(kr["at"])


def test_the_dreamy_set_s_faces_and_poses_and_its_two_hand_shapes_need_both_hands():
    peek = MV.compile([{"name": "peek", "t": 1.0, "hand": "L"}], MARKS)
    assert {x["morph"] for x in peek["expressions"]} == {"wink_r", "mouth_smile"}       # the other eye shut
    assert [v for _, v in peek["lean"]] == [0.0, 4.0, 4.0, 0.0] and [v for _, v in peek["head_tilt"]][1] == 5.0
    blow = MV.compile([{"name": "blow", "t": 1.0, "hand": "R"}], MARKS)
    assert {x["morph"] for x in blow["expressions"]} == {"u", "calm"} and not blow["lean"]   # pursed lips, blowing
    for name in ("butterfly", "bloom"):                                  # a butterfly, a flower: one shape of two hands
        with pytest.raises(MV.MoveError, match="needs both hands free, and the R hand holds the mic"):
            MV.compile([{"name": "rest", "hand": "R", "place": "mic"}, {"name": name, "t": 1.0}], MARKS, BEATS)
    wings = MV.compile([{"name": "rest", "hand": "R", "place": "mic"}, {"name": "wings", "t": 1.0, "dur": 1.6,
                                                                         "hits": [1.2, 1.7, 2.2]}], MARKS, BEATS)
    down = near(MV.resolve("wings_down", "L", MARKS)[0])
    assert [k["t"] for k in placed(wings["hands"]["L"]) if k["at"] == down] == pytest.approx([1.2, 1.7, 2.2])  # its hits
    assert len(wings["hands"]["R"]) == 1                                 # the mic hand stays at the mic


def test_a_peek_on_the_clip_s_first_frame_already_holds_its_ring_there():
    """A video cut to start on its first sung word plays the peek from clip 0: the hand travels in the pre-roll, so its
    first key is the ring at its eye on 0.0 (the pose stage holds a first key's target from the start), fingers formed."""
    keys = MV.compile([{"name": "peek", "t": 0.0, "dur": 1.0, "hand": "L"}], MARKS)["hands"]["L"]
    ring = MV.resolve("peek", "L", MARKS)
    assert keys[0]["t"] == 0.0 and keys[0]["at"] == near(ring[0]) and keys[0]["fingers"] == MV.SHAPES["ring"]
    assert all(k["at"] == near(ring[0]) for k in keys if k["t"] <= 1.0)


@pytest.mark.parametrize("side", ["L", "R"])
def test_a_finger_heart_by_the_cheek_stands_upright_its_crossing_facing_the_lens(side):
    """The finger heart's two tips point up from their crossing, leaning in toward the face by HEART[3], and its plane
    faces the lens, the index's side to it, on either hand: the place used to show the lens the little finger's side of
    a fist, and with the thumb's side to the lens no elbow keeps the forearm in range."""
    s = 1.0 if side == "L" else -1.0
    _, face, up = MV._crossing(MV.posed(MARKS, "heart"))
    _, d, p, *_ = MV.resolve("heart", side, MARKS)
    tips, front = MV._off(up, d, p, side), MV._off(face, d, p, side)
    assert tips[2] > 0.9 and s * tips[0] < 0.0 and front[1] > 0.95


@pytest.mark.parametrize("side, s", [("L", 1.0), ("R", -1.0)])
def test_a_peek_holds_a_ring_at_its_own_eye_and_shuts_the_other_whichever_way_a_model_names_its_winks(side, s):
    """The ring the thumb and index close is held at the hand's own eye, in front of it, its hole on the eye's line to the
    lens (a member without a `pupil` landmark: the eye bone `eye_dx` to its side), its opening to the lens (the hand
    side-on, the fanned fingers up); the other eye shuts. A model names its winks from either side (`ウィンク右` shuts the
    right eye of one and the eye on the right of the picture of another): `winks` says which semantic wink shuts which
    eye."""
    po = MV.posed(MARKS, "ring")
    I, T = po["points"]["index"], po["points"]["thumb"]
    loop = np.array([I[0], I[1], I[2], po["tips"]["index"], po["tips"]["thumb"], T[2], T[1]])
    pt, d, p, *_ = MV.resolve("peek", side, MARKS)
    hole = MV.sighted("peek", side, MARKS, pt, d, p)
    eye = np.asarray(MARKS["eye"]) + [s * MV.EYE_DX * MARKS["hand"], 0.0, 0.0]
    assert abs(hole[0] - eye[0]) < 1e-3 and abs(hole[2] - eye[2]) < 1e-3 and hole[1] < eye[1] - 0.05
    ring = np.array([_world("peek", side, MARKS, q) for q in loop])
    normal = np.linalg.svd(ring - ring.mean(0))[2][2]
    assert abs(normal[1]) > 0.95                                         # the ring's opening to the lens
    assert d[2] > 0.5 and s * d[0] < 0.0                                 # the hand up and in over the brow, side-on, ...
    assert MV._off(MV._plane(loop, MV._v(0.0, 0.0, -1.0)), d, p, side)[1] > 0.9   # ... the thumb toward the face
    for winks, shut in ((None, {"L": "wink_r", "R": "wink_l"}), ({"L": "wink_r", "R": "wink_l"}, {"L": "wink_l", "R": "wink_r"})):
        out = MV.compile([{"name": "peek", "t": 1.0, "hand": side}], MARKS, winks=winks)
        morphs = {x["morph"] for x in out["expressions"]}
        assert shut[side] in morphs and ({"wink_l", "wink_r"} - {shut[side]}).isdisjoint(morphs)


@pytest.mark.parametrize("side, s", [("L", 1.0), ("R", -1.0)])
def test_the_peek_s_ring_hole_is_on_the_line_from_the_lens_to_the_visible_pupil_held_with_the_head(side, s):
    """An MMD eye's visible pupil is a mesh of its own about 1 cm out from the eye bone and 4 cm in front of it (the
    landmark `pupil`): the hole of the ring (the circle through the index's curl and the thumb's pad, `hole`; the joints'
    mean sits ~1 cm off it, toward the knuckles) is on the pupil's line to the lens within 3 mm as seen from the front,
    on the hand's own side, not on the eye bone's. Both its keys, the arrival and the end of the hold, are there and
    marked `sight` (the pose stage holds the hand with the head between them, so the ring stays on the eye as the body
    leans and the head tilts); the trips in and out are not. The forearm and upper arm keep in the human range and the
    elbow stays out at the shoulder."""
    po = MV.posed(MARKS, "ring")
    I, T = po["points"]["index"], po["points"]["thumb"]
    curl = np.array([I[1], I[2], po["tips"]["index"], po["tips"]["thumb"], T[2], T[1]])
    r = np.linalg.norm(curl - MV.hole(po), axis=1)
    assert r.max() - r.min() < 0.4 * r.mean() and np.linalg.norm(curl.mean(0) - MV.hole(po)) > 0.005   # its middle
    marks = dict(MARKS, pupil=[0.0425, -0.083, 1.412])
    pupil = np.array([s * 0.0425, -0.083, 1.412])
    pt, d, p, _, _, _, pole = MV.resolve("peek", side, marks)
    hole = MV.sighted("peek", side, marks, pt, d, p)
    assert np.linalg.norm((hole - pupil)[[0, 2]]) < 0.003 and hole[1] < pupil[1] - 0.04
    keys = MV.compile([{"name": "peek", "t": 1.0, "dur": 1.0, "hand": side}], marks)["hands"][side]
    held = [k for k in keys if k.get("sight")]
    assert [k["t"] for k in held] == [1.0, 2.0] and all(k["at"] == near(pt) for k in held)
    assert all(not k.get("sight") for k in keys if not 1.0 <= k["t"] <= 2.0)
    fore, upper, *_ = MV.twisted(side, marks, pt, d, p, pole)
    assert AT.FOREARM[0] + MV.TWIST_MARGIN <= fore <= AT.FOREARM[1] - MV.TWIST_MARGIN
    assert AT.UPPER[0] + MV.TWIST_MARGIN <= upper <= AT.UPPER[1] - MV.TWIST_MARGIN
    S, E = np.asarray(marks[f"arm.{side}"], float), MV.elbow(side, marks, pt, pole)
    assert s * (E[0] - S[0]) > 0.08 and abs(E[2] - S[2]) < 0.08


def test_a_held_place_with_an_elbow_of_its_own_stays_still_with_no_key_halfway():
    """A key halfway through a trip (`_via`) measured the hand's bend with the stage's own elbow, so a held peek (its
    elbow low in front) got one turning the hand while it held still: it wobbled."""
    for name in ("peek", "giggle"):
        keys = MV.compile([{"name": name, "t": 1.0, "dur": 1.0, "hand": "R"}], MARKS)["hands"]["R"]
        held = [k for k in keys if 1.0 <= k["t"] <= 2.0]
        assert len(held) == 2 and held[0]["at"] == held[1]["at"] and held[0]["dir"] == held[1]["dir"], name


@pytest.mark.parametrize("name", [n for n in PLACES if PLACES.index(n) >= PLACES.index("peek")])
def test_the_new_places_keep_off_the_body_and_its_clothes(name):
    head, torso = ball(MARKS["eye"], 0.105), ball((0.0, -0.02, 1.13), 0.125)
    hips = ball((0.0, -0.02, 0.88), 0.13)
    clear = lambda P: min(head(P), torso(P), hips(P))                   # noqa: E731
    margin = MV.TOUCH_MARGIN if name in MV.TOUCH or name in MV.CLOTH else MV.MARGIN
    for side in ("L", "R"):
        pt, d, p, shape, _, _, _ = MV.resolve(name, side, MARKS, clear)
        assert clear(pt + MV.hand_box(d, p, MARKS["hand"], shape)) >= margin - 5e-4, side


def test_a_hand_on_a_skirt_rests_on_the_cloth_even_where_the_sim_moves_it():
    """A skirt the sim moves makes way for a hand, so the body under it alone kept a hand off (`clear`): pinching it at
    the side or clasped behind the back the hand ended inside the skirt. Those places (CLOTH) measure to the cloth too
    (`cloth`) and end on it; other places do not see it."""
    hips = ball((0.0, -0.02, 0.86), 0.12)
    skirt = ball((0.0, -0.02, 0.8), 0.2)                                # the cloth flaring out round the hips
    cloth = lambda P: min(hips(P), skirt(P))                            # noqa: E731
    for name in ("skirt", "behind", "behind_one"):
        under, on = MV.resolve(name, "L", MARKS, hips), MV.resolve(name, "L", MARKS, hips, cloth)
        box = lambda r: r[0] + MV.hand_box(r[1], r[2], MARKS["hand"], r[3])   # noqa: E731
        assert cloth(box(under)) < 0.0, name                             # inside the skirt
        assert cloth(box(on)) == pytest.approx(MV.TOUCH_MARGIN, abs=5e-4), name   # on it
    assert MV.resolve("dainty", "L", MARKS, hips, cloth)[0] == pytest.approx(MV.resolve("dainty", "L", MARKS, hips)[0])


# ---------------------------------------------------------------- rest places, and changing them
def _named(keys, side, names):
    """A hand's keys at places as (t, place name)."""
    res = {n: MV.resolve(n, side, MARKS)[0] for n in names}
    return [(k["t"], next((n for n in names if k["at"] == near(res[n])), "?")) for k in placed(keys)]


@pytest.mark.parametrize("two", ["behind", "clasp"])
def test_hands_at_a_pose_for_two_rest_together_and_one_alone_rests_at_its_lone_place(two):
    """Clasped behind the back or at the chest, two hands rest together; one alone (its partner away on a move, or
    resting elsewhere) rests at its lone place (the back of the hand on the small of the back, the hand on the chest),
    gliding there as its partner leaves and back as it returns, as a dainty hand does."""
    lone = MV.LONE[two]
    both = [{"name": "rest", "hand": "L", "place": two}, {"name": "rest", "hand": "R", "place": two}]
    seen = _named(MV.compile(both + [{"name": "peek", "t": 2.0, "dur": 1.0, "hand": "L"}], MARKS)["hands"]["R"], "R",
                  (two, lone))
    assert seen[0][1] == two and seen[-1][1] == two and {p for _, p in seen} == {two, lone}
    assert [t for t, p in seen if p == lone][0] == pytest.approx(2.0, abs=1e-3)            # alone as its partner peeks
    back = 3.0 + MV.travel(MV.resolve("peek", "L", MARKS), MV.resolve(two, "L", MARKS))
    assert seen[-1][0] == pytest.approx(back, abs=1e-3)                                     # and back with it
    alone = MV.compile([both[1], {"name": "peek", "t": 2.0, "dur": 1.0, "hand": "L"}], MARKS)["hands"]["R"]
    assert {p for _, p in _named(alone, "R", (two, lone))} == {lone}                       # its partner rests elsewhere
    r = MV.resolve(two, "R", MARKS)
    assert r[3] == "clasp" and r[5] <= MV.WRIST                                             # the fingers laced


def test_a_rest_entry_with_a_time_takes_the_hand_to_its_new_place_on_that_time_at_a_human_pace():
    """`{name = "rest", hand, place, t}` changes where the hand waits from t on: it waits at the old place before,
    setting off to be at the new one on t (the trip at the human pace), as it sets off for a move; entries without `t`
    are where it waits from the start. Two hands moving together to a pose for two meet there."""
    es = [{"name": "rest", "hand": "L", "place": "dainty"}, {"name": "rest", "hand": "R", "place": "dainty"},
          {"name": "rest", "hand": "L", "place": "behind", "t": 5.0}, {"name": "rest", "hand": "R", "place": "behind", "t": 5.0}]
    out = MV.compile(es, MARKS)
    for sd in "LR":
        two, back = MV.resolve("dainty", sd, MARKS), MV.resolve("behind", sd, MARKS)
        keys = placed(out["hands"][sd])
        assert [k["at"] for k in keys] == [near(two[0]), near(back[0])]                    # dainty from the start
        assert [k["t"] for k in keys] == pytest.approx([5.0 - MV.travel(two, back), 5.0], abs=1e-4)
        assert keys[1]["pole"] == near(list(MV.elbow_pole("behind", sd, MARKS)))          # the elbow back there
    from_rest = MV.compile([{"name": "rest", "hand": "L", "place": "skirt", "t": 2.0}], MARKS)["hands"]["L"]
    assert _named(from_rest, "L", ("rest", "skirt")) == [(pytest.approx(2.0 - MV.travel(
        MV.resolve("rest", "L", MARKS), MV.resolve("skirt", "L", MARKS)), abs=1e-4), "rest"), (2.0, "skirt")]


def test_a_rest_change_a_move_leaves_no_time_for_is_refused_saying_how_long_the_trip_takes():
    base = [{"name": "rest", "hand": "L", "place": "dainty"}, {"name": "rest", "hand": "L", "place": "behind", "t": 5.0}]
    with pytest.raises(MV.MoveError, match=r"the L hand needs 0\.\d\d s to get from 'point' \(at 4\.8 s\) to "
                                           r"'behind_one' \(at 5 s\) and has 0\.20 s: change the rest place later"):
        MV.compile(base + [{"name": "point", "t": 4.8, "dur": 0.15, "hand": "L"}], MARKS)      # still pointing
    with pytest.raises(MV.MoveError, match=r"the L hand needs 0\.\d\d s to get from its rest place 'behind_one' "
                                           r"\(at 5 s\) to 'point' \(at 5\.1 s\) and has 0\.10 s"):
        MV.compile(base + [{"name": "point", "t": 5.1, "dur": 0.5, "hand": "L"}], MARKS)       # due to point at once
    with pytest.raises(MV.MoveError, match=r"the L hand plays a move from 4 to 6 s and its rest place changes to "
                                           r"'behind_one' at 5 s"):
        MV.compile(base + [{"name": "point", "t": 4.0, "dur": 2.0, "hand": "L"}], MARKS)       # pointing across it
    names = ("dainty_one", "behind_one", "point")                        # alone: its partner rests by the hip
    fine = _named(MV.compile(base + [{"name": "point", "t": 3.0, "dur": 1.0, "hand": "L"}], MARKS)["hands"]["L"], "L",
                  names)
    assert [p for _, p in fine] == ["dainty_one", "point", "point", "dainty_one", "dainty_one", "behind_one"]
    on_time = _named(MV.compile(base + [{"name": "point", "t": 5.0, "dur": 1.0, "hand": "L"}], MARKS)["hands"]["L"],
                     "L", names)
    assert on_time[-1][1] == "behind_one" and on_time[-1][0] > 6.0      # a move on its time: there after it


def test_a_hand_left_at_a_pose_for_two_follows_its_partner_s_rest_changes_when_it_has_the_time():
    """When its partner changes rest place a hand at a pose for two goes to its lone place, arriving as the partner gets
    to its new one, and back when the partner returns; busy with a move then, it goes there when the move is done."""
    es = [{"name": "rest", "hand": "L", "place": "dainty"}, {"name": "rest", "hand": "R", "place": "dainty"},
          {"name": "rest", "hand": "R", "place": "clasp", "t": 3.0}, {"name": "rest", "hand": "R", "place": "dainty", "t": 6.0}]
    out = MV.compile(es, MARKS)["hands"]
    two, one = MV.resolve("dainty", "L", MARKS), MV.resolve("dainty_one", "L", MARKS)
    assert _named(out["L"], "L", ("dainty", "dainty_one")) == [
        (pytest.approx(3.0 - MV.travel(two, one), abs=1e-4), "dainty"), (3.0, "dainty_one"),
        (pytest.approx(6.0 - MV.travel(one, two), abs=1e-4), "dainty_one"), (6.0, "dainty")]
    assert [p for _, p in _named(out["R"], "R", ("dainty", "clasp_one"))] == ["dainty", "clasp_one", "clasp_one",
                                                                             "dainty"]   # alone at the chest
    busy = MV.compile(es + [{"name": "point", "t": 2.5, "dur": 1.0, "hand": "L"}], MARKS)["hands"]["L"]
    seen = _named(busy, "L", ("dainty", "dainty_one", "point"))
    after = next(i for i, (t, p) in enumerate(seen) if p == "point" and t > 3.0)
    assert seen[after + 1][1] == "dainty_one"                            # home to its lone place after the move


def test_rest_entries_are_checked_and_a_hand_at_the_mic_keeps_it():
    for es, frag in (([{"name": "rest", "hand": "L", "place": "behind", "t": "5"}], "expected"),
                     ([{"name": "rest", "hand": "L", "place": "dainty_one"}], "expected"),
                     ([{"name": "rest", "hand": "L", "place": "behind", "t": 5.0},
                       {"name": "rest", "hand": "L", "place": "clasp", "t": 5.0}], "the L hand has two rest places from 5 s"),
                     ([{"name": "rest", "hand": "L", "place": "dainty"}, {"name": "rest", "hand": "L", "place": "clasp"}],
                      "two rest places from the start"),
                     ([{"name": "rest", "hand": "R", "place": "mic"}, {"name": "rest", "hand": "R", "place": "clasp", "t": 2.0}],
                      "holds the mic from the start to the end")):
        with pytest.raises(MV.MoveError, match=frag):
            MV.compile(es, MARKS)


# ---------------------------------------------------------------- letting a hand go
def test_a_free_hand_lets_its_ik_go_over_the_span_and_eases_it_back_in_time_for_the_next_move():
    """While a free move lets the hand go (a walk strip swings the arm) its IK share eases from all to none and back over
    FREE_EASE, and the goal waits at the rest place, so the hand comes back from there to its next move."""
    es = [{"name": "free", "t": 1.0, "dur": 2.0, "hand": "L"}, {"name": "point", "t": 4.0, "dur": 0.5, "hand": "L"}]
    out = MV.compile(es, MARKS)
    E = MV.FREE_EASE
    assert out["ik"] == {"L": [[1.0, 1.0], [1.0 + E, 0.0], [3.0 - E, 0.0], [3.0, 1.0]]}       # the right hand is not let go
    home = MV.resolve("rest", "L", MARKS)[0]
    during = [k for k in out["hands"]["L"] if 1.0 <= k["t"] <= 3.0]
    assert during and all(k["at"] == near(home) for k in during)                                # the goal waits at rest
    assert next(k for k in out["hands"]["L"] if k["t"] == 4.0)["at"] == near(MV.resolve("point", "L", MARKS)[0])


def test_free_spans_that_touch_are_one_and_a_short_one_eases_over_half_its_length():
    assert MV.ik_keys([(2.0, 3.0), (1.0, 2.0)]) == [[1.0, 1.0], [1.25, 0.0], [2.75, 0.0], [3.0, 1.0]]
    assert MV.ik_keys([(1.0, 1.3)]) == [[1.0, 1.0], [1.15, 0.0], [1.15, 0.0], [1.3, 1.0]]
    with pytest.raises(MV.MoveError, match="holds the mic"):
        MV.compile([{"name": "rest", "hand": "R", "place": "mic"}, {"name": "free", "t": 1.0, "hand": "R"}], MARKS)
    with pytest.raises(MV.MoveError, match="two moves at once"):
        MV.compile([{"name": "free", "t": 1.0, "dur": 2.0, "hand": "L"}, {"name": "point", "t": 2.0, "hand": "L"}], MARKS)


def _strip(t0, t1, centre, radius=0.08):
    """A strip's hand over (t0, t1) on every 30th of a second: the wrist round a circle in front of the chest, the hand
    pointing up, the palm forward."""
    ts = np.round(np.arange(t0, t1 + 1e-9, 1.0 / 30.0), 6)
    a = 2.0 * np.pi * (ts - t0) / max(t1 - t0, 1e-9)
    at = np.asarray(centre, float) + radius * np.stack([np.cos(a), np.zeros_like(a), np.sin(a)], 1)
    return {"t": ts.tolist(), "at": at.tolist(), "dir": [[0.0, 0.0, 1.0]] * len(ts), "palm": [[0.0, -1.0, 0.0]] * len(ts)}


def test_a_free_hand_on_a_strip_follows_it_and_joins_and_leaves_it_at_a_human_pace():
    """Given where a strip has the hand over a free span, the goal follows it there (so the IK lets go and takes over on
    the strip's own hand: nothing moves at the hand-offs), the hand travels to the strip's first place from its last move
    at the human pace, and leaves the strip at its end to reach the next move; the IK is let go over exactly that."""
    strip = {"L": _strip(1.5, 3.0, (0.18, -0.25, 1.2))}
    es = [{"name": "point", "t": 0.5, "dur": 0.9, "hand": "L"}, {"name": "free", "t": 1.5, "dur": 1.5, "hand": "L"},
          {"name": "heart_wink", "t": 3.6, "dur": 0.5, "hand": "L"}]
    out = MV.compile(es, MARKS, strip=strip)
    keys = out["hands"]["L"]
    on = [k for k in keys if k.get("strip")]
    assert [k["t"] for k in on] == pytest.approx(strip["L"]["t"], abs=1e-4)                 # every sample, in order
    assert [k["at"] for k in on] == [near(p) for p in strip["L"]["at"]]
    assert not any("fingers" in k for k in on) and all("fingers" in k for k in placed(keys))
    rest = MV.resolve("rest", "L", MARKS)[0]
    assert not [k for k in keys if 1.4 <= k["t"] <= 3.6 and k["at"] == near(rest)]          # never parked at rest
    point = MV.resolve("point", "L", MARKS)
    gone = MV.travel(point, (np.asarray(strip["L"]["at"][0]), np.array([0.0, 0.0, 1.0]), np.array([0.0, -1.0, 0.0])))
    before = [k for k in placed(keys) if k["t"] < 1.5]
    assert 1.5 - gone < 1.4                                                                 # it must set off early ...
    assert before[-1]["at"] == near(point[0]) and before[-1]["t"] == pytest.approx(1.5 - gone, abs=1e-3)   # and does
    E = MV.FREE_EASE
    assert out["ik"]["L"] == [[1.5, 1.0], [1.5 + E, 0.0], [3.0 - E, 0.0], [3.0, 1.0]]
    heart = next(k for k in keys if k["t"] == pytest.approx(3.6))
    assert heart["at"] == near(MV.resolve("heart", "L", MARKS)[0])


def test_a_hand_leaves_its_strip_early_for_a_move_that_comes_soon_and_one_too_soon_is_refused():
    strip = {"L": _strip(1.0, 2.5, (0.2, -0.2, 0.95))}
    late = MV.compile([{"name": "free", "t": 1.0, "dur": 1.5, "hand": "L"},
                       {"name": "peace_eye", "t": 2.62, "dur": 0.5, "hand": "L"}], MARKS, strip=strip)
    peace = MV.resolve("peace", "L", MARKS)
    on = [k for k in late["hands"]["L"] if k.get("strip")]
    leave = on[-1]["t"]
    s = (np.asarray(on[-1]["at"]), np.array([0.0, 0.0, 1.0]), np.array([0.0, -1.0, 0.0]))
    assert leave < 2.5 and leave + MV.travel(s, peace) <= 2.62 + 1e-6                    # the latest it can stay
    assert late["ik"]["L"][-1] == [round(leave, 4), 1.0]                                   # the IK is back as it leaves
    home = MV.resolve("rest", "L", MARKS)
    alone = MV.compile([{"name": "free", "t": 1.0, "dur": 1.5, "hand": "L"}], MARKS, strip=strip)
    last = alone["hands"]["L"][-1]
    s = (np.asarray(strip["L"]["at"][-1]), np.array([0.0, 0.0, 1.0]), np.array([0.0, -1.0, 0.0]))
    assert last["at"] == near(home[0]) and last["t"] == pytest.approx(2.5 + MV.travel(s, home), abs=1e-3)   # home after
    with pytest.raises(MV.MoveError, match=r"the L hand needs 0\.\d\d s to get from the strip \(at 1 s\) to 'peace'"):
        MV.compile([{"name": "free", "t": 1.0, "dur": 0.1, "hand": "L"},
                    {"name": "peace_eye", "t": 1.15, "dur": 0.5, "hand": "L"}], MARKS, strip=strip)


@pytest.mark.parametrize("entries, marks, frag", [
    ([{"name": "moonwalk", "t": 1.0}], MARKS, "no such move"),
    ([{"name": "point", "t": 1.0, "speed": 2}], MARKS, "expected {name, t"),
    ([{"name": "point", "t": 1.0, "dur": 0}], MARKS, "dur must be positive"),
    ([{"name": "point", "t": 1.0}], {k: v for k, v in MARKS.items() if k != "fore"}, "landmarks missing"),
])
def test_bad_moves_are_refused_naming_the_fault(entries, marks, frag):
    with pytest.raises(MV.MoveError, match=frag):
        MV.compile(entries, marks)


# ---------------------------------------------------------------- arm twist
def _rot(axis, deg):
    return AT.rotation(axis, deg)


def test_twist_is_the_turn_about_the_axis_whatever_the_swing_and_signed_for_supination_on_either_side():
    """The shared measure (mkmmd.core.armtwist, the joint_limits check's too): a turn about the forearm read through any
    swing square to it, + supination on both sides (the left arm's axis points to +x, the right's to -x)."""
    a = np.array([1.0, 0.0, 0.0])
    for deg in (-170.0, -60.0, 0.0, 45.0, 150.0):
        D = _rot([0.0, 0.3, 1.0], 35.0) @ _rot(a, deg)                  # a swing about an axis not square to it, then
        swing = _rot(np.cross(a, [0.0, 0.3, 1.0]), 35.0) @ _rot(a, deg)  # one square to it: only the latter is a swing
        assert AT.twist(swing, a) == pytest.approx(deg, abs=1e-6)
        assert abs(AT.twist(D, a)) < 181.0
    I, R = np.eye(3), _rot([0.0, 0.0, 1.0], 30.0)
    turn = _rot(a, -40.0)                                               # the left palm turned from down to forward
    assert AT.relative_twist(R, I, R @ turn, I, a, "L") == pytest.approx(40.0)
    assert AT.relative_twist(R, I, R @ _rot(-a, 40.0), I, -a, "R") == pytest.approx(40.0)   # its mirror image
    assert AT.upper_limits(45.0) == AT.UPPER and AT.upper_limits(150.0)[1] == AT.UPPER[1] + 60.0


def test_the_ik_hinge_model_reads_no_twist_at_rest_and_a_neutral_forearm_as_neutral():
    """The arm IK is a hinge folding forward from rest: the rest arm twists 0, and an elbow bent 90 degrees at the side
    with the thumb up (the palm toward the body, anatomy's neutral) twists 0 too; the palm turned up reads +90."""
    rest = AT.REST
    S = np.zeros(3)
    W0 = S + 0.35 * np.asarray(rest["upper"])
    assert MV.overtwist(AT.twists(rest, "L", S, S + 0.2 * np.asarray(rest["upper"]), W0, rest["dir"], rest["palm"])
                        + (60.0,), 0.0) == 0.0
    E, W = np.array([0.02, 0.0, -0.2]), np.array([0.02, -0.18, -0.2])  # the elbow under the shoulder, the forearm forward
    fore, _ = AT.twists(rest, "L", S, E, W, [0.0, -1.0, 0.0], [-1.0, 0.0, 0.0])   # thumb up, palm to the body
    assert fore == pytest.approx(0.0, abs=8.0)                          # (the A-pose rest tilts the hinge a little)
    assert AT.twists(rest, "L", S, E, W, [0.0, -1.0, 0.0], [0.0, 0.0, 1.0])[0] == pytest.approx(90.0, abs=8.0)
    assert AT.twists(rest, "R", S * [-1, 1, 1], E * [-1, 1, 1], W * [-1, 1, 1], [0.0, -1.0, 0.0],
                     [1.0, 0.0, 0.0])[0] == pytest.approx(fore, abs=1e-6)                  # the right arm, mirrored


@pytest.mark.parametrize("scale", [0.85, 1.0, 1.2])
@pytest.mark.parametrize("name", PLACES)
def test_every_place_keeps_both_arms_twists_in_the_human_range(name, scale):
    """No place asks a forearm or an upper arm to turn past what it can (mkmmd.core.armtwist), on either side, any body:
    the elbow is turned round the shoulder-wrist line where its own would wring the forearm (a ring at the eye side-on,
    a palm held up before the mouth), and the places a forearm could not reach as first written are written so it can
    (the finger heart, the hammer cocked over the shoulder, the hand in the hair and tossing it, the hands over the
    ears)."""
    m = body(scale)
    for side in ("L", "R"):
        pt, d, p, _, _, _, pole = MV.resolve(name, side, m)
        assert MV.overtwist(MV.twisted(side, m, pt, d, p, pole), 0.0) == 0.0, (side, MV.twisted(side, m, pt, d, p, pole))


@pytest.mark.parametrize("side", ["L", "R"])
def test_the_peek_lifts_the_elbow_out_at_the_shoulder_and_the_blow_brings_it_in_under_the_palm(side):
    """The ring at the eye side-on needed ~180 degrees of forearm turn from an elbow low in front (the sleeve wrung into a
    pillar before her mouth); the elbow goes out to the side at about the shoulder's height, the forearm comes in from
    the side and the twist is comfortable. A palm up before the mouth from an elbow out and back wanted ~150: its elbow
    comes in under the hand."""
    s = 1.0 if side == "L" else -1.0
    S = np.asarray(MARKS[f"arm.{side}"], float)
    pt, d, p, _, _, _, pole = MV.resolve("peek", side, MARKS)
    E = MV.elbow(side, MARKS, pt, pole)
    fore, upper, _ = MV.twisted(side, MARKS, pt, d, p, pole)
    assert s * (E[0] - S[0]) > 0.08 and abs(E[2] - S[2]) < 0.08 and E[1] < S[1]    # out, at the shoulder, in front
    assert abs(fore) < 75.0 and abs(upper) < 75.0
    assert s * (pt[0] - E[0]) < 0.0                                     # the forearm comes in from the side
    pt, d, p, _, _, _, pole = MV.resolve("blow", side, MARKS)
    E = MV.elbow(side, MARKS, pt, pole)
    assert E[2] < pt[2] and s * (E[0] - S[0]) < 0.05                    # under the hand, in from the shoulder
    assert MV.overtwist(MV.twisted(side, MARKS, pt, d, p, pole), 0.0) == 0.0


def test_a_trip_between_far_apart_elbows_turns_the_forearm_the_short_way_through_the_middle():
    """From behind the back to the ring at the eye, and from the arm out as a wing to the palm up before the mouth, the
    elbow swings round the shoulder and the forearm turns over: keys halfway keep the pole round the shoulder (keyed
    straight across it passes the arm's line and the IK's elbow flips) and the twists between their neighbours', so no
    stretch of the trip turns the forearm the long way round past what it can."""
    es = [{"name": "rest", "hand": "L", "place": "behind"}, {"name": "peek", "t": 1.0, "dur": 0.6, "hand": "L"},
          {"name": "wings", "t": 2.5, "dur": 1.0}, {"name": "blow", "t": 4.0, "dur": 1.2, "hand": "L"}]
    keys = MV.compile(es, MARKS, BEATS)["hands"]["L"]
    tw = [MV.twisted("L", MARKS, k["at"], k["dir"], k["palm"], k.get("pole")) for k in keys]
    assert all(MV.overtwist(t, 0.0) == 0.0 for t in tw)
    assert all(abs(b[0] - a[0]) < MV.TWIST_STEP and abs(b[1] - a[1]) < MV.TWIST_STEP for a, b in zip(tw, tw[1:]))
    S = np.asarray(MARKS["arm.L"], float)
    poles = [np.asarray(k["pole"], float) - S if "pole" in k else MV.pole_point("L", MARKS) - S for k in keys]
    for a, b in zip(poles, poles[1:]):                                  # never straight across the shoulder
        assert a @ b > np.linalg.norm(a) * np.linalg.norm(b) * math.cos(math.radians(120.0))


def test_a_hand_joins_and_leaves_a_strip_only_where_the_strip_keeps_the_arm_in_range():
    """A dance whose arms shrug past what an upper arm turns where a free move starts (the strip eased in under the IK,
    the goal on its hand: 125 degrees out): the hand joins it after that, and the IK lets go from there."""
    strip = {"L": _strip(1.0, 2.5, (0.2, -0.2, 0.95))}
    strip["L"]["ok"] = [not (t < 1.2 or t > 2.35) for t in strip["L"]["t"]]
    out = MV.compile([{"name": "free", "t": 1.0, "dur": 1.5, "hand": "L"}], MARKS, strip=strip)
    on = [k for k in out["hands"]["L"] if k.get("strip")]
    assert on[0]["t"] >= 1.2 - 1e-6 and on[-1]["t"] <= 2.35 + 1e-6
    assert out["ik"]["L"][0] == [pytest.approx(on[0]["t"], abs=1e-4), 1.0]
