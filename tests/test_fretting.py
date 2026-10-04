"""Fretting tables and neck targets, the neck grip solver, wearing a prop (placement, strap), a hanging cable and the strum
frame: pure maths on a synthetic neck card (an upright guitar-like neck: z toward the nut, the board facing -y)."""
import math

import numpy as np
import pytest

from mkmmd.core import cable as CB
from mkmmd.core import fretting as FR
from mkmmd.core import gripframe as GF
from mkmmd.core import wear as WR

SCALE = 0.648


def neck_entry(nfrets=22, board_y=-0.011):
    """A neck card entry in the contract's frame: origin at the saddle line, +z toward the nut, the board's surface plane at
    y = board_y facing -y, low E on -x."""
    z = lambda n: SCALE * 2.0 ** (-n / 12.0)                              # noqa: E731
    nut_x = np.array([-17.5, -10.5, -3.5, 3.5, 10.5, 17.5]) * 1e-3
    bridge_x = np.array([-26.25, -15.75, -5.25, 5.25, 15.75, 26.25]) * 1e-3
    radius = [0.0009, 0.0007, 0.0005, 0.0004, 0.0003, 0.00025]
    return {"name": "neck", "type": "neck", "frame": {"along": [0, 0, 1], "across": [1, 0, 0], "normal": [0, -1, 0]},
            "thumb": [0, 1, 0], "scale": SCALE, "frets": [[0.0, board_y, z(n)] for n in range(nfrets + 1)],
            "strings": [{"name": "EADGBe"[i], "nut": [nut_x[i], board_y - 0.0016, z(0)],
                         "bridge": [bridge_x[i], board_y - 0.001, 0.0], "radius": radius[i]} for i in range(6)],
            "section": {"width": [0.043, 0.056], "depth": [0.021, 0.024], "p": 2.6}, "fret_height": 0.0012}


# ------------------------------------------------------------------------------------------------ chords and targets
def test_every_chord_has_distinct_strings_and_valid_fingers():
    for name, table in FR.CHORDS.items():
        shp = FR.shape(name)
        assert shp == {k: tuple(v) for k, v in table.items()}
        assert len({s for s, _ in shp.values()}) == len(shp), name
        assert set(shp) <= set(FR.FINGERS)


def test_shape_accepts_tables_and_rejects_bad_ones():
    assert FR.shape({"index": [5, 0], "ring": [4, 2]}) == {"index": (5, 0), "ring": (4, 2)}
    for bad in ("nonsense", {"thumb": [1, 0]}, {"index": [7, 0]}, {"index": [3, -1]}, {"index": [3, 0], "ring": [3, 2]}, {}, 5):
        with pytest.raises(FR.FrettingError):
            FR.shape(bad)


def test_fret_positions_follow_equal_temperament():
    z = FR.fret_z(neck_entry())
    assert z[0] == pytest.approx(SCALE) and z[12] == pytest.approx(SCALE / 2)
    d = SCALE - z                                                           # distance from the nut
    assert d[1] == pytest.approx(0.03637, abs=1e-4)                         # the first fret's classic 36.4 mm
    assert np.all(np.diff(z) < 0)


def test_neck_frame_is_a_right_handed_frame_on_the_board():
    e = neck_entry()
    o, R = FR.neck_frame(e, 5)
    assert o == pytest.approx(e["frets"][5])
    assert R.T @ R == pytest.approx(np.eye(3)) and np.linalg.det(R) == pytest.approx(1.0)
    assert R[:, 0] == pytest.approx([0, 0, 1]) and R[:, 2] == pytest.approx([0, -1, 0])     # toward the nut; out of the board
    assert R[:, 1] == pytest.approx([-1, 0, 0])                                              # y = z cross x: toward the low E
    with pytest.raises(FR.FrettingError):
        FR.neck_frame(e, 40)


def test_power_chord_targets_sit_on_their_strings_behind_their_wires():
    e = neck_entry()
    tg = {t["finger"]: t for t in FR.finger_targets(e, "power", 5)}
    assert set(tg) == {"index", "ring", "little"}
    assert (tg["index"]["fret"], tg["ring"]["fret"], tg["little"]["fret"]) == (5, 7, 7)
    assert (tg["index"]["string"], tg["ring"]["string"], tg["little"]["string"]) == (6, 5, 4)
    zs = FR.fret_z(e)
    for t in tg.values():
        p = np.array(t["point"])
        k = t["fret"]
        assert zs[k] < p[2] < zs[k - 1]                                                    # between the wire it plays and the next toward the nut
        assert (p[2] - zs[k]) / (zs[k - 1] - zs[k]) == pytest.approx(FR.PRESS)
        assert t["lo"] < p[2] < t["hi"]
        st = e["strings"][6 - t["string"]]
        line = FR.string_point(e, t["string"], p[2])
        assert p[0] == pytest.approx(line[0], abs=1e-9)                                    # on the string's line across the neck
        assert p[1] == pytest.approx(-0.011 - (FR.FRET_HEIGHT + st["radius"]))           # pressed on the wire's crown, out of the board (-y)
    assert tg["index"]["point"][0] < tg["ring"]["point"][0] < tg["little"]["point"][0]     # low E, A, D from -x to +x


def test_open_chords_press_the_classic_frets_at_position_one():
    e = neck_entry()
    got = {name: sorted((t["string"], t["fret"]) for t in FR.finger_targets(e, name, 1)) for name in ("E", "A", "D", "G", "C")}
    assert got["E"] == [(3, 1), (4, 2), (5, 2)]
    assert got["A"] == [(2, 2), (3, 2), (4, 2)]
    assert got["D"] == [(1, 2), (2, 3), (3, 2)]
    assert got["G"] == [(1, 3), (5, 2), (6, 3)]
    assert got["C"] == [(2, 1), (4, 2), (5, 3)]


def test_targets_refuse_the_nut_and_the_end_of_the_neck():
    e = neck_entry(nfrets=12)
    with pytest.raises(FR.FrettingError, match="starts at 1"):
        FR.finger_targets(e, "power", 0)
    with pytest.raises(FR.FrettingError, match="neck has 12"):
        FR.finger_targets(e, "power", 11)


def test_solver_prop_is_the_card_seen_from_the_neck_frame():
    e = neck_entry()
    prop, (o, R) = FR.solver_prop(e, "power", 3)
    assert [t["finger"] for t in prop["targets"]] == ["index", "ring", "little"]
    for t, c in zip(prop["targets"], FR.finger_targets(e, "power", 3)):
        assert o + R @ np.array(t["point"]) == pytest.approx(c["point"])
        assert t["x"][0] < t["point"][0] < t["x"][1]
        assert t["point"][2] == pytest.approx(FR.FRET_HEIGHT + c["radius"])               # z is the height over the board
    assert prop["frets"][3] == pytest.approx(0.0) and prop["frets"][0] > prop["frets"][3] > prop["frets"][10]
    s = prop["section"]
    assert s["x"][0] < min(t["point"][0] for t in prop["targets"]) and s["x"][1] > max(t["point"][0] for t in prop["targets"])
    assert 0.043 <= s["width"][1] <= s["width"][0] <= 0.056 and s["p"] == 2.6      # wider toward the body
    st = prop["strings"][0]
    assert o + R @ np.array(st[0]) == pytest.approx(e["strings"][0]["nut"])


def test_section_is_linear_between_nut_and_last_fret():
    e = neck_entry()
    zs = FR.fret_z(e)
    assert FR.section_at(e, zs[0]) == pytest.approx((0.043, 0.021))
    assert FR.section_at(e, zs[-1]) == pytest.approx((0.056, 0.024))
    assert FR.section_at(e, 2.0) == pytest.approx((0.043, 0.021))                          # held beyond the nut
    w, d = FR.section_at(e, 0.5 * (zs[0] + zs[-1]))
    assert w == pytest.approx(0.0495) and d == pytest.approx(0.0225)


def test_card_style_gives_the_neck_and_pick_problems():
    e = neck_entry()
    style, prop, params = GF.card_style(e, {"fret": 3, "chord": "power"})
    assert style == "neck" and len(prop["targets"]) == 3 and params == {}
    with pytest.raises(ValueError, match="fret"):
        GF.card_style(e, {"chord": "power"})
    strum = {"name": "strum", "type": "strum", "pick": {"thickness": 0.0008, "length": 0.031, "width": 0.026, "tip": 0.008}}
    style, prop, params = GF.card_style(strum, {})
    assert style == "pinch" and prop == {"width": 0.0008, "depth": 0.031, "length": 0.026}
    assert params["edge"] == pytest.approx(0.023)                                           # the pads sit 23 mm from the base: 8 mm of tip
    assert GF.card_style(strum, {"tip": 0.012})[2]["edge"] == pytest.approx(0.019)
    with pytest.raises(ValueError, match="inside its length"):
        GF.card_style(strum, {"tip": 0.05})
    with pytest.raises(ValueError, match="pick"):
        GF.card_style({"name": "s", "type": "strum"}, {})


def test_strum_frame_puts_the_pick_tip_on_the_strings():
    centre, normal, along = np.array([0.0, -0.02, 0.4]), np.array([0.0, -1.0, 0.0]), np.array([0.0, 0.0, 1.0])
    G = GF.strum_frame(centre, normal, along, 0.008, "neck")
    assert G[:3, :3].T @ G[:3, :3] == pytest.approx(np.eye(3)) and np.linalg.det(G[:3, :3]) == pytest.approx(1.0)
    assert G[:3, 0] == pytest.approx([0, 1, 0])                                             # x: the pick's direction toward its tip = into the face
    assert G[:3, 2] == pytest.approx([0, 0, 1])                                             # z: the squeeze axis, thumb toward the neck
    tip = G[:3, 3] + 0.008 * G[:3, 0]
    assert tip == pytest.approx(centre)
    assert GF.strum_frame(centre, normal, along, 0.008, "bridge")[:3, 2] == pytest.approx([0, 0, -1])
    with pytest.raises(ValueError):
        GF.strum_frame(centre, normal, along, 0.008, "left")


# ------------------------------------------------------------------------------------------------ the neck solver
@pytest.mark.parametrize("side", ["L"])
def test_neck_grip_presses_the_strings_with_arched_fingers_and_the_thumb_behind(side):
    from test_grip import synthetic_hand
    from mkmmd.solvers import grip as G
    hand = synthetic_hand(side)
    prop, _frame = FR.solver_prop(neck_entry(), "power", 3)
    res = G.solve("neck", hand, prop, seeds=2, workers=1)
    rep = res["report"]
    assert res["style"] == "neck" and len(res["bones"]) == 15
    for f in ("index", "ring", "little"):
        c = rep["contacts"][f]
        assert abs(c["gap_mm"]) < 1.0 and abs(c["across_mm"]) < 3.5 and c["along_mm"] < 1.5, (f, c)
        assert c["distal_deg"] < 50.0                                                       # arched onto the board
    assert rep["contacts"]["thumb"]["gap_mm"] < 1.5 and rep["contacts"]["thumb"]["facing_deg"] < 60.0
    assert rep["penetration_mm"] < 0.5 and rep["finger_clash_mm"] < 0.5
    assert rep["hovering_clear_mm"] > 5.0                                                   # the middle finger hovers over the strings
    wrist = rep["wrist_in_N_mm"]
    assert wrist[1] < 0 and wrist[2] < 0                                                    # the wrist below the neck and behind the board
    T = np.array(res["target_in_wrist"])
    assert T[:3, :3].T @ T[:3, :3] == pytest.approx(np.eye(3), abs=1e-6)


def test_neck_solver_rejects_bad_problems():
    from test_grip import synthetic_hand
    from mkmmd.solvers import grip as G
    hand = synthetic_hand("L")
    prop, _ = FR.solver_prop(neck_entry(), "power", 3)
    with pytest.raises(ValueError, match="targets"):
        G.solve("neck", hand, {"section": prop["section"]}, workers=1)
    with pytest.raises(ValueError, match="unknown neck tuning"):
        G.solve("neck", hand, prop, workers=1, bogus=1)
    with pytest.raises(ValueError, match="finger"):
        G.solve("neck", hand, {"targets": [{"finger": "thumb", "point": [0, 0, 0], "x": [0, 0]}], "section": prop["section"]},
                workers=1)


# ------------------------------------------------------------------------------------------------ wearing
WEAR = {"name": "stand", "bone": "upper_body2", "pivot": [0.0, 0.045, 0.0], "ref": {"top": 1.70, "shoulder_width": 0.18},
        "at": [-0.06, -0.10, -0.12], "scale": ["shoulder_width", "top", "top"], "neck_deg": 25.0, "yaw_deg": 10.0, "roll_deg": 0.0}


def test_orientation_brings_the_neck_to_the_left_and_up():
    R = WR.orientation(25.0, 0.0, 0.0)
    assert R.T @ R == pytest.approx(np.eye(3)) and np.linalg.det(R) == pytest.approx(1.0)
    neck, face = R[:, 2], -R[:, 1]                                                          # the prop's +z (neck) and its face (-y)
    assert neck == pytest.approx([math.cos(math.radians(25)), 0.0, math.sin(math.radians(25))])     # left (+x) and up
    assert face == pytest.approx([0.0, -1.0, 0.0])                                          # the face looks forward (-y)
    assert R[:, 0][2] < 0                                                                   # the prop's x (toward the high e) points down
    yawed = WR.orientation(25.0, 20.0, 0.0)[:, 2]
    assert yawed[1] < 0 and yawed[0] < neck[0]                                              # swung toward her front
    rolled = -WR.orientation(25.0, 0.0, 15.0)[:, 1]
    assert rolled[2] > 0.2                                                                  # a positive roll turns the face up
    assert rolled @ neck == pytest.approx(0.0, abs=1e-9)


def test_placement_scales_with_the_wearer_and_puts_the_pivot_at_the_belly():
    e = WEAR
    small = {"top": 1.70, "shoulder_width": 0.18}
    big = {"top": 1.87, "shoulder_width": 0.198}
    R, at, root = WR.placement(e, small)
    assert at == pytest.approx([-0.06, -0.10, -0.12])
    assert root + R @ np.array(e["pivot"]) == pytest.approx(at)                            # the pivot lands on `at`
    R2, at2, _ = WR.placement(e, big)
    assert at2 == pytest.approx([-0.06 * 1.1, -0.10 * 1.1, -0.12 * 1.1])
    with pytest.raises(WR.WearError, match="scale"):
        WR.placement(e, {"top": 1.7})
    over = WR.params(e, {"neck_deg": 30})
    assert over["neck_deg"] == 30 and WR.params(e)["neck_deg"] == 25.0
    with pytest.raises(WR.WearError, match="unknown key"):
        WR.params(e, {"bogus": 1})


def test_matrix_is_the_root_in_the_armature_frame():
    M = WR.matrix(WEAR, {"top": 1.70, "shoulder_width": 0.18}, [0.0, -0.05, 1.02])
    R, at, root = WR.placement(WEAR, {"top": 1.70, "shoulder_width": 0.18})
    assert M[:3, :3] == pytest.approx(R) and M[:3, 3] == pytest.approx(np.array([0.0, -0.05, 1.02]) + root)
    assert M[3] == pytest.approx([0, 0, 0, 1])


def test_entry_picks_by_name():
    use = {"wear": [WEAR, dict(WEAR, name="low", neck_deg=10.0)]}
    assert WR.entry(use, "low")["neck_deg"] == 10.0
    with pytest.raises(WR.WearError, match="name one"):
        WR.entry(use)
    with pytest.raises(WR.WearError, match="no wear entry"):
        WR.entry(use, "high")
    with pytest.raises(WR.WearError, match="no use.wear"):
        WR.entry({})
    assert WR.entry({"wear": [WEAR]})["name"] == "stand"


def test_strap_goes_over_the_shoulder_round_the_back_and_clears_the_torso():
    spine = WR.spine_axis([[0, -0.053, 0.935], [0, -0.05, 1.019], [0, -0.039, 1.168], [0, -0.053, 0.79]])
    top = (np.array([0.30, -0.22, 1.02]), np.array([-0.5, 0.2, 0.8]))
    bottom = (np.array([-0.10, -0.17, 0.80]), np.array([0.0, 0.5, 0.2]))
    shoulder = np.array([0.091, 0.006, 1.18])
    path = WR.strap_path(top, bottom, shoulder, spine, radius=0.0998, over_side=1.0)
    assert path.shape == (48, 3)
    assert path[0] == pytest.approx(top[0]) and path[-1] == pytest.approx(bottom[0])
    assert np.all(np.isfinite(path))
    assert path[:, 2].max() > shoulder[2] + 0.02                                            # over the shoulder, above the joint
    k = int(np.argmax(path[:, 2]))
    assert 0.03 < path[k, 0] < 0.13 and abs(path[k, 1] - shoulder[1]) < 0.06
    back = path[(path[:, 1] > spine(1.0)[1] + 0.05)]
    assert len(back) > 5 and back[:, 0].min() < 0.0 < back[:, 0].max() + 0.1                # the diagonal crosses her back
    # beyond the two ends the band stays out of the torso: every point of the middle is farther than its radius from the axis
    mid = path[3:-3]
    ax = np.array([spine(z) for z in mid[:, 2]])
    r = np.hypot(mid[:, 0] - ax[:, 0], mid[:, 1] - ax[:, 1])
    inside = (mid[:, 2] < shoulder[2]) & (mid[:, 2] > 0.85)
    assert r[inside].min() > 0.0998 * 0.9
    mirrored = WR.strap_path((top[0] * [-1, 1, 1], top[1] * [-1, 1, 1]), (bottom[0] * [-1, 1, 1], bottom[1] * [-1, 1, 1]),
                             shoulder * [-1, 1, 1], spine, radius=0.0998, over_side=-1.0)
    assert mirrored[:, 0] == pytest.approx(-path[:, 0], abs=1e-9)                           # a right shoulder: the mirror image


# ------------------------------------------------------------------------------------------------ the cable
def test_cable_hangs_to_the_floor_and_lies_along_it():
    top = np.array([0.55, -0.20, 0.95])
    P = CB.hang(top, [0.0, -0.3, -0.95], floor_z=0.0, tail=(-1.0, 1.0), radius=0.0032)
    assert P[0] == pytest.approx(top) and len(P) == 9
    assert P[:, 2].min() == pytest.approx(0.0032, abs=1e-9)                                  # the cord's centre never sinks into the floor
    assert np.all(P[5:, 2] < 0.2)                                                           # the far part lies on the floor
    assert np.linalg.norm(P[-1, :2] - P[5, :2]) == pytest.approx(0.9, rel=0.05)             # tail_len of it along the floor
    d = (P[-1] - P[5])[:2]
    assert d[0] < 0 < d[1] and abs(d[0] + d[1]) < 0.4 * np.linalg.norm(d)                    # trailing toward (-1, 1)
    assert CB.length(P) > 0.95 + 0.9                                                         # longer than the drop and the lie
    with pytest.raises(ValueError, match="no room"):
        CB.hang([0, 0, 0.03], [0, 0, -1], 0.0)
