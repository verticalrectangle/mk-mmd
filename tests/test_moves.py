"""Moves, bpy-free (mkmmd.core.moves): `[[move.<cast>]]` entries compiled to hand keys, lean / tilt keys, twitches and
expressions, every hand place made natural (the wrist against the forearm an arm IK gives, a mic aimed, the hand kept off
the body). The Blender half (the keys laid into the pose and perform tables, a move held on a real arm) is checked in
tests/test_build_stages.py."""
import math

import numpy as np
import pytest

from mkmmd.core import armreach as AR
from mkmmd.core import moves as MV

MARKS = {"arm.L": [0.093, -0.02, 1.25], "arm.R": [-0.093, -0.02, 1.25], "chest": [0.0, -0.065, 1.12],
         "mouth": [0.0, -0.071, 1.35], "eye": [0.0, -0.045, 1.41], "reach": 0.37, "upper": 0.195, "fore": 0.176,
         "hand": 0.146}
BEATS = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
PLACES = ("rest", "dainty", "mic", "mic_lens", "mic_up", "mic_across", "chest", "bounce", "point", "heart", "peace",
          "paw", "sparkle", "up", "drip_in", "drip_out", "ears", "heart_push", "bunny", "bonk_up", "bonk")


def near(p):
    return pytest.approx(p, abs=1e-4)                                   # the keys are rounded to 0.1 mm


def ball(centre, radius):
    """clear(points) of a ball: the smallest distance from the points to it (negative inside)."""
    c = np.asarray(centre, float)
    return lambda P: float((np.linalg.norm(np.asarray(P, float).reshape(-1, 3) - c, axis=1) - radius).min())


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
        pt, d, p, _, _, bend = MV.resolve(name, side, m)
        assert bend <= MV.WRIST + 1e-6
        assert bend == pytest.approx(MV.bend(d, MV.forearm(side, m, pt)), abs=1e-6)
        assert abs(d @ p) < 1e-9 and np.linalg.norm(d) == pytest.approx(1) and np.linalg.norm(p) == pytest.approx(1)


@pytest.mark.parametrize("name", PLACES)
def test_the_right_hand_is_the_left_s_mirror_image_in_every_place(name):
    clear = ball((0.0, -0.02, 1.0), 0.17)                                # a body to keep off, the same for both
    for c in (None, clear):
        (pl, dl, ql, *_), (pr, dr, qr, *_) = (MV.resolve(name, s, MARKS, c) for s in ("L", "R"))
        M = np.array([-1.0, 1.0, 1.0])
        assert pl * M == pytest.approx(pr) and dl * M == pytest.approx(dr) and ql * M == pytest.approx(qr)


@pytest.mark.parametrize("side, s", [("L", 1.0), ("R", -1.0)])
@pytest.mark.parametrize("name", sorted(MV.AIM))
def test_a_hand_holding_a_mic_up_or_out_has_its_palm_toward_the_body_never_twisted_out(name, side, s):
    for scale in (0.85, 1.0, 1.2):
        _, _, p, *_ = MV.resolve(name, side, body(scale))
        assert s * p[0] < -0.3                                             # the palm toward the midline


def test_a_hand_bent_too_far_turns_toward_its_forearm_as_a_whole():
    f = np.array([0.0, 0.0, 1.0])
    d, p = MV.natural([0.0, -1.0, -1.0], [0.0, -1.0, 1.0], f)            # 135 degrees down: a paw past breaking
    assert MV.bend(d, f) == pytest.approx(MV.WRIST)
    assert d @ p == pytest.approx(0.0, abs=1e-9)                       # the palm turned with it
    assert MV.natural([0.0, -0.5, 1.0], [0.0, 1.0, 0.5], f)[0] == pytest.approx(np.array([0, -0.5, 1]) / math.sqrt(1.25))


@pytest.mark.parametrize("chest_z", [1.18, 1.12, 1.05])               # a long torso bends the heart's wrists past the limit
def test_the_two_hand_heart_keeps_its_shape_its_hands_mirrored_and_meeting_on_any_body(chest_z):
    m = dict(MARKS, chest=[0.0, -0.065, chest_z])
    (pl, dl, ql, _, _, _), (pr, dr, qr, _, _, _) = (MV.resolve("heart_push", s, m) for s in ("L", "R"))
    assert pl == pytest.approx([-pr[0], pr[1], pr[2]]) and dl == pytest.approx([-dr[0], dr[1], dr[2]])
    for pt, d, p, side in ((pl, dl, ql, "L"), (pr, dr, qr, "R")):
        assert abs((pt + MV.BOX["heart2"][0] * m["hand"] * d)[0]) < 0.015   # the fingertips meet at the midline
        _, wd, wp, _ = MV.place("heart_push", side, m)
        assert d == pytest.approx(wd / np.linalg.norm(wd))                   # the heart as written, never turned
        assert p == pytest.approx(wp / np.linalg.norm(wp), abs=0.01)


@pytest.mark.parametrize("side", ["L", "R"])
@pytest.mark.parametrize("name", sorted(MV.AIM))
def test_an_aimed_mic_points_where_aimed_across_a_palm_that_continues_the_forearm(name, side):
    pt, d, p, _, _, bend = MV.resolve(name, side, MARKS)
    sigma = 1.0 if side == "R" else -1.0
    a = MV.AIM[name]
    aim = np.array([(1.0 if side == "L" else -1.0) * a[0], a[1], a[2]])
    assert sigma * np.cross(d, p) == pytest.approx(aim / np.linalg.norm(aim))   # the thumb side, as at the mic place
    f = MV.forearm(side, MARKS, pt)
    assert bend == pytest.approx(90.0 - math.degrees(math.acos(abs(f @ aim) / np.linalg.norm(aim))), abs=1e-6)
    m_d, m_p = MV.place("mic", side, MARKS)[1:3]
    assert sigma * np.cross(m_d, m_p) / np.linalg.norm(np.cross(m_d, m_p)) == pytest.approx([0, 0, 1])   # the mic place's mic is upright


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
    pt, d, p, shape, moved, _ = MV.resolve("rest", "L", m, skirt)
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
        pt, d, p, shape, moved, _ = MV.resolve(name, "L", MARKS, clear)
        assert moved > 0 and clear(pt + MV.hand_box(d, p, MARKS["hand"], shape)) == pytest.approx(MV.TOUCH_MARGIN, abs=5e-4)
    big = ball(MARKS["mouth"], 0.2)                                       # a place by the face inside a big hood
    pt, d, p, shape, moved, _ = MV.resolve("heart", "L", MARKS, big)
    assert moved > 0 and big(pt + MV.hand_box(d, p, MARKS["hand"], shape)) == pytest.approx(MV.MARGIN, abs=5e-4)
    assert MV.resolve("heart", "L", MARKS, clear)[4] == 0.0                # clear of the body: where it was asked


def test_places_by_the_face_and_chest_stay_on_their_own_side_clear_of_a_mic_at_the_mouth():
    for name in ("chest", "heart", "paw"):
        for side, s in (("L", 1.0), ("R", -1.0)):
            pt = MV.place(name, side, MARKS)[0]
            assert s * pt[0] - MV.WIDTH * MARKS["hand"] > 0.02, (name, side)    # the hand's inner edge off the midline


# ---------------------------------------------------------------- the clock
def test_a_move_holds_its_last_place_to_its_end_and_the_hand_goes_home_after_it():
    out = MV.compile([{"name": "point", "t": 1.0, "dur": 0.8, "hand": "L"}], MARKS)
    keys = out["hands"]["L"]
    home = MV.resolve("rest", "L", MARKS)[0]
    point = MV.resolve("point", "L", MARKS)[0]
    assert [k["t"] for k in keys] == pytest.approx([1.0 - MV.EASE, 1.0, 1.8, 1.8 + MV.EASE])
    assert [k["at"] for k in keys] == [near(p) for p in (home, point, point, home)]
    assert keys[1]["fingers"] == "point" and keys[0]["fingers"] == "relaxed"
    assert set(out["places"]["L"]) == {"rest", "point"}


def test_moves_close_together_go_straight_on_and_far_apart_go_home_between():
    def ats(gap):
        es = [{"name": "point", "t": 1.0, "dur": 0.5, "hand": "L"},
              {"name": "heart_wink", "t": 1.5 + gap, "dur": 0.5, "hand": "L"}]
        return [k["at"] for k in MV.compile(es, MARKS)["hands"]["L"]]
    home = near(MV.resolve("rest", "L", MARKS)[0])
    assert ats(0.2).count(home) == 2                                      # in before the first, out after the last
    assert ats(1.0).count(home) == 4                                      # and home between them


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
    up, du, pu, _, _, bend_up = MV.resolve("bonk_up", side, MARKS)
    pt, d, _, _, _, bend = MV.resolve("bonk", side, MARKS)
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
    assert up["R"][1]["at"] == near(MV.resolve("up", "R", MARKS)[0])
    assert {k["fingers"] for k in up["R"]} == {"curled"} and up["L"][1]["fingers"] == "flat"   # the mic stays in the fist
    with pytest.raises(MV.MoveError, match="needs both hands free, and the R hand holds the mic"):
        MV.compile([{"name": "rest", "hand": "R", "place": "mic"}, {"name": "heart_push", "t": 1.0}], MARKS)
    with pytest.raises(MV.MoveError, match="rest"):
        MV.compile([{"name": "rest", "hand": "R", "place": "chest"}], MARKS)


def test_a_dainty_hand_waits_lightly_on_the_front_of_the_body_and_goes_back_there():
    torso = ball((0.0, -0.02, 0.9), 0.15)                                  # a skirt round the hips
    es = [{"name": "rest", "hand": "L", "place": "dainty"}, {"name": "point", "t": 1.0, "dur": 0.5, "hand": "L"}]
    keys = MV.compile(es, MARKS, clear=torso)["hands"]["L"]
    home = MV.resolve("dainty", "L", MARKS, torso)
    assert keys[0]["at"] == near(home[0]) and keys[-1]["at"] == near(home[0])
    assert torso(home[0] + MV.hand_box(home[1], home[2], MARKS["hand"], home[3])) == pytest.approx(MV.TOUCH_MARGIN, abs=5e-4)
    assert home[0][1] < MARKS["chest"][1] and 0 < home[0][0] < MARKS["arm.L"][0] + 0.02   # in front, inside the shoulder
    waits = MV.compile([{"name": "rest", "hand": "L", "place": "dainty"}], MARKS)["hands"]["L"]
    assert len(waits) == 1 and waits[0]["t"] == 0.0                         # it waits there from the start


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
    with pytest.raises(MV.MoveError, match="both lean the body"):
        MV.compile([{"name": "lean_back", "t": 1.0, "dur": 1.0}, {"name": "into_lens", "t": 1.5, "dur": 1.0}], MARKS)


@pytest.mark.parametrize("entries, marks, frag", [
    ([{"name": "moonwalk", "t": 1.0}], MARKS, "no such move"),
    ([{"name": "point", "t": 1.0, "speed": 2}], MARKS, "expected {name, t"),
    ([{"name": "point", "t": 1.0, "dur": 0}], MARKS, "dur must be positive"),
    ([{"name": "point", "t": 1.0}], {k: v for k, v in MARKS.items() if k != "fore"}, "landmarks missing"),
])
def test_bad_moves_are_refused_naming_the_fault(entries, marks, frag):
    with pytest.raises(MV.MoveError, match=frag):
        MV.compile(entries, marks)
