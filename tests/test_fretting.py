"""Fretting tables and neck targets, the neck grip solver, wearing a prop (placement, strap), a hanging cable and the strum
frame: pure maths on a synthetic neck card (an upright guitar-like neck: z toward the nut, the board facing -y)."""
import math

import numpy as np
import pytest

from mkmmd.core import cable as CB
from mkmmd.core import fretting as FR
from mkmmd.core import gripframe as GF
from mkmmd.core import wear as WR
from mkmmd.solvers import grip_neck as GN

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
    tx = {t["finger"]: t["point"][0] for t in prop["targets"]}
    assert prop["slots"][0] == pytest.approx(tx["index"]) and prop["slots"][2] == pytest.approx(tx["ring"])   # one finger per fret
    assert tx["index"] > prop["slots"][1] > tx["ring"] and prop["slots"][3] < tx["little"]
    assert FR.solver_prop(e, {"index": [6, 0]}, 21)[0]["slots"][2:] == [None, None]          # frets 23 and 24 are not on the neck
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
    assert params["stop"] == pytest.approx(0.008 - GF.STRUM_CLEAR)                         # the hand stays above the strings (x = tip)
    assert params["heading"] == GF.STRUM_HEADING                                            # the arm places the wrist, not a heading
    assert GF.card_style(strum, {"tip": 0.012})[2]["edge"] == pytest.approx(0.019)
    assert GF.card_style(dict(strum, clear=0.006), {})[2]["stop"] == pytest.approx(0.002)                # a card may ask more room
    assert GF.card_style(dict(strum, clear=0.006), {"tip": 0.013, "clear": 0.010})[2]["stop"] == pytest.approx(0.003)   # the hand wins
    with pytest.raises(ValueError, match="clear"):
        GF.card_style(strum, {"clear": 0.02})
    # held as a plectrum: the thumb across it. The slab turns with the hand (the pick's width under the fingers, the pads
    # on its middle line, its longer end across them) and the strings stop the skin along y, where the tip now points
    style, prop, params = GF.card_style(dict(strum, hold="across"), {"tip": 0.018, "clear": 0.009})
    assert style == "pinch" and prop == {"width": 0.0008, "depth": 0.026, "length": pytest.approx(0.036)}
    assert params["edge"] == pytest.approx(0.013) and params["stop"] == pytest.approx(0.009) and params["stop_axis"] == "y"
    assert "stop_axis" not in GF.card_style(dict(strum, hold="across"), {"hold": "along"})[2]          # the hand's key wins
    with pytest.raises(ValueError, match="hold"):
        GF.card_style(strum, {"hold": "sideways"})
    G = GF.strum_frame((0.0, -0.0155, 0.091), (0, -1, 0), (0, 0, 1), (1, 0, 0), 0.018)
    H = GF.hand_frame(G, "across")
    assert GF.hand_frame(G, "along") is not None and np.allclose(GF.hand_frame(G, "along"), G)
    assert np.allclose(H[:3, 1], G[:3, 0]) and np.allclose(H[:3, 2], G[:3, 2]) and np.allclose(H[:3, 3], G[:3, 3])   # y to the tip
    assert H[:3, 0] @ np.array([0.0, 1.0, 0.0]) == pytest.approx(0.0) and H[2, 0] > 0.5 and H[0, 0] > 0.5   # the fingers lie along
    assert np.allclose(H @ np.linalg.inv(GF.HOLD_TURN), G)                                             # the face, toward the nut
    with pytest.raises(ValueError, match="inside its length"):
        GF.card_style(strum, {"tip": 0.05})
    with pytest.raises(ValueError, match="pick"):
        GF.card_style({"name": "s", "type": "strum"}, {})


def test_strum_frame_puts_the_pick_tip_on_the_strings_with_the_thumb_on_top():
    centre, normal = np.array([0.0, -0.02, 0.4]), np.array([0.0, -1.0, 0.0])
    along, across = np.array([0.0, 0.0, 1.0]), np.array([1.0, 0.0, 0.0])                   # toward the nut; lowest string -> highest
    G = GF.strum_frame(centre, normal, along, across, 0.008, "neck")
    assert G[:3, :3].T @ G[:3, :3] == pytest.approx(np.eye(3)) and np.linalg.det(G[:3, :3]) == pytest.approx(1.0)
    assert G[:3, 0] == pytest.approx([0, 1, 0])                                             # x: the pick's direction toward its tip = into the face
    s = math.sqrt(0.5)
    assert G[:3, 2] == pytest.approx([-s, 0, s])                                            # z: the thumb on the bass side, leaning to the neck
    tip = G[:3, 3] + 0.008 * G[:3, 0]
    assert tip == pytest.approx(centre)
    assert GF.strum_frame(centre, normal, along, across, 0.008, "bridge")[:3, 2] == pytest.approx([-s, 0, -s])
    assert GF.strum_frame(centre, normal, along, across, 0.008, lean=0.0)[:3, 2] == pytest.approx([-1, 0, 0])    # square to the sweep
    assert GF.strum_frame(centre, normal, along, across, 0.008, lean=90.0)[:3, 2] == pytest.approx([0, 0, 1])    # edge on
    skewed = GF.strum_frame(centre, normal, along + 0.3 * normal, across - 0.2 * normal, 0.008)                # off the face plane
    assert skewed[:3, 2] @ skewed[:3, 0] == pytest.approx(0.0, abs=1e-12)
    with pytest.raises(ValueError):
        GF.strum_frame(centre, normal, along, across, 0.008, "left")
    with pytest.raises(ValueError, match="normal"):
        GF.strum_frame(centre, normal, along, normal, 0.008)


# ------------------------------------------------------------------------------------------------ the neck solver
@pytest.mark.parametrize("side", ["L"])
def test_neck_grip_presses_the_strings_with_arched_fingers_and_the_thumb_behind(side):
    from test_grip import synthetic_hand
    from mkmmd.solvers import grip as G
    hand = synthetic_hand(side)
    prop, _frame = FR.solver_prop(neck_entry(), "power", 3)
    res = G.solve("neck", hand, prop, seeds=2, workers=1)
    rep = res["report"]
    assert res["style"] == "neck" and len(res["bones"]) == 15 and rep["misses"] == []
    for f in ("index", "ring", "little"):
        c = rep["contacts"][f]
        tol_x = GN.PAD_TOL_X if f == "index" else GN.PAD_TOL_CROWD                       # ring and little side by side on fret 5
        assert abs(c["gap_mm"]) < 1.0 and c["along_mm"] < 0.5, (f, c)
        assert abs(c["across_mm"]) < GN.PAD_TOL_Y * 1e3 + 0.5 and abs(c["dx_mm"]) < tol_x * 1e3 + 1.0, (f, c)   # on the string, at its fret
        assert c["pad_mm"] < GN.PAD_TOUCH * 1e3 + 0.2                                        # touching with the pad
        assert c["distal_deg"] < 50.0                                                       # arched onto the board
    assert rep["contacts"]["ring"]["dx_mm"] - rep["contacts"]["little"]["dx_mm"] > 4.0        # side by side: staggered along the fret
    assert rep["contacts"]["thumb"]["gap_mm"] < 1.5 and rep["contacts"]["thumb"]["facing_deg"] < 60.0
    assert rep["penetration_mm"] < 0.5 and rep["finger_clash_mm"] < 0.5
    assert rep["hovering_clear_mm"] > 5.0                                                   # the middle finger hovers over the strings
    assert rep["knuckles_mm"] < GN.KNUCKLE_Z * 1e3 + 3.0                                    # under the neck, not standing in front of it
    mid = rep["hovering"]["middle"]
    half = 0.5 * FR.section_at(neck_entry(), FR.fret_z(neck_entry())[4])[0] * 1e3
    assert mid["gap_mm"] < GN.HOVER_MAX * 1e3 + 1.5 and abs(mid["y_mm"]) < half             # just over the strings, over the board
    assert abs(mid["x_mm"] - prop["slots"][1] * 1e3) < GN.HOME_TOL * 1e3 + 1.0               # ... over its own fret
    assert mid["distal_deg"] < 65.0                                                         # curled toward them, not sticking up
    xh = 0.5 * (prop["slots"][0] + prop["slots"][1]) * 1e3                                   # the thumb behind the index and middle
    assert xh + GN.THUMB_X[0] * 1e3 - 1.0 < rep["contacts"]["thumb"]["x_mm"] < xh + GN.THUMB_X[1] * 1e3 + 1.0
    wrist = rep["wrist_in_N_mm"]
    assert wrist[1] < 0 and wrist[2] < 0                                                    # the wrist below the neck and behind the board
    T = np.array(res["target_in_wrist"])
    assert T[:3, :3].T @ T[:3, :3] == pytest.approx(np.eye(3), abs=1e-6)


def _ref(res, prop):
    """The reference a shape's solve takes from the solved shape before it (as the pose stage builds it)."""
    return {"angles": res["report"]["angles_deg"], "target_in_wrist": np.asarray(res["target_in_wrist"]).tolist(),
            "targets": {t["finger"]: [t["string"], t["fret"]] for t in prop["targets"]}, "position": prop["position"]}


def test_the_reach_keeps_the_hand_out_of_the_neck():
    """The pads first reach their targets with the neck solid seen softly: a reach that ignored it left the hand deep in the
    neck (a note on the far string, a stretch to the little finger), and the full problem then pushed it out in front of the
    board, a claw over the fretboard instead of a hand under the neck."""
    from scipy.optimize import least_squares
    from test_grip import synthetic_hand
    from mkmmd.solvers import grip as G
    hand = synthetic_hand("L")
    arm = G.arm_spec({"shoulder": [-0.20, 0.16, -0.26], "a": 0.25, "b": 0.20, "pole": [-0.24, -0.33, -0.36]})
    for chord, fret in (({"index": [6, 0]}, 3), ({"little": [6, 3]}, 3), ({"middle": [6, 1]}, 9)):
        prop, _ = FR.solver_prop(neck_entry(), chord, fret)
        prob = GN._Neck(hand, prop["targets"], prop["section"], arm=arm, slots=prop["slots"], position=fret)
        deg = np.clip(prob.design, prob.lo[:21] + 1e-6, prob.hi[:21] - 1e-6)
        prob.R0, prob.p0 = prob.start_frame(deg)
        res = least_squares(lambda v, p=prob: p.residuals(v, "reach"), np.r_[deg, np.zeros(6)], bounds=(prob.lo, prob.hi),
                            x_scale=prob.X_SCALE, max_nfev=GN.NFEV[0], xtol=1e-12, ftol=1e-6)
        depth = -prob.analyse(res.x)["sdf"][prob.skin].min()
        assert depth < 0.002, (chord, fret, depth)


def test_a_reference_keeps_the_hand_and_moves_only_the_playing_finger():
    """A line at one position: each shape is solved from the one before it (`ref`). The hand stays (the palm a few
    millimetres along the neck at most, the thumb behind it), the finger that played lifts off its string, the free fingers
    keep their joints, and only the next note's finger goes down onto its string."""
    from test_grip import synthetic_hand
    from mkmmd.solvers import grip as G
    hand = synthetic_hand("L")
    a_prop, _ = FR.solver_prop(neck_entry(), {"index": [6, 0]}, 3)
    a = G.solve("neck", hand, a_prop, seeds=1, workers=1)
    b_prop, _ = FR.solver_prop(neck_entry(), {"middle": [5, 1]}, 3)
    b = G.solve("neck", hand, b_prop, seeds=1, workers=1, ref=_ref(a, a_prop))
    ra, rb = a["report"], b["report"]
    assert rb["misses"] == [] and rb["starts"] == 2                                          # the reference's start and a design one
    palm = rb["from_ref"]["palm_mm"]
    assert abs(palm[0]) < 8.0 and abs(palm[1]) < 4.0 and abs(palm[2]) < 4.0 and rb["from_ref"]["turn_deg"] < 4.0
    assert abs(rb["contacts"]["thumb"]["x_mm"] - ra["contacts"]["thumb"]["x_mm"]) < 8.0
    for f in ("ring", "little"):                                                             # free before and after
        for j in ("mcp", "pip", "dip"):
            assert abs(rb["angles_deg"][f"{f}_{j}"] - ra["angles_deg"][f"{f}_{j}"]) < 4.0, (f, j)
    assert rb["hovering"]["index"]["gap_mm"] > GN.CLEAR_HOVER * 1e3 - 1.0                    # the index left its string
    c = rb["contacts"]["middle"]
    assert abs(c["gap_mm"]) < 1.0 and abs(c["across_mm"]) < GN.PAD_TOL_Y * 1e3 + 0.5 and abs(c["dx_mm"]) < GN.PAD_TOL_X * 1e3 + 1.0
    again = G.solve("neck", hand, b_prop, seeds=1, workers=1, ref=_ref(a, a_prop))
    assert again["target_in_wrist"] == pytest.approx(b["target_in_wrist"])                   # the same inputs give the same shape


def test_a_lifted_shape_eases_the_fingers_off_and_a_shift_slides_the_hand():
    """The shapes a keyed change passes through: the chord eased off the strings (`FR.lifted`) solved from it keeps the
    hand and lifts every finger HOVER over its string; the chord at another position solved from it slides the hand along
    the neck (its start, the hand as it was relative to the board, wins and passes the gates)."""
    from test_grip import synthetic_hand
    from mkmmd.solvers import grip as G
    hand = synthetic_hand("L")
    p8, _ = FR.solver_prop(neck_entry(), "power", 8)
    g8 = G.solve("neck", hand, p8, seeds=1, workers=1)
    off, _ = FR.solver_prop(neck_entry(), FR.lifted("power"), 8)
    lift = G.solve("neck", hand, off, seeds=1, workers=1, ref=_ref(g8, p8))["report"]
    assert lift["misses"] == [] and max(abs(v) for v in lift["from_ref"]["palm_mm"]) < 2.0
    for t_off, t_on in zip(off["targets"], p8["targets"]):
        assert t_off["point"][2] - t_on["point"][2] > FR.HOVER                                # each pad over its string ...
        assert abs(lift["contacts"][t_off["finger"]]["gap_mm"]) < 1.0                          # ... where its target says
    p3, _ = FR.solver_prop(neck_entry(), "power", 3)
    shift = G.solve("neck", hand, p3, seeds=1, workers=1, ref=_ref(g8, p8))["report"]
    assert shift["misses"] == [] and shift["from_ref"]["turn_deg"] < 15.0
    assert max(abs(v) for v in shift["from_ref"]["palm_mm"]) < 25.0                           # slid along: the same hand at fret 3


def test_neck_grip_with_the_arm_keeps_the_wrist_on_the_forearm_and_within_reach():
    """The fretting arm of a standing player (in N at fret 3: the shoulder toward the body and behind the board, the elbow's
    pole below it): the solver keeps the wrist's bend against the forearm the IK makes within its free angle."""
    from test_grip import synthetic_hand
    from mkmmd.solvers import grip as G
    hand = synthetic_hand("L")
    prop, _frame = FR.solver_prop(neck_entry(), "power", 3)
    arm = {"shoulder": [-0.20, 0.16, -0.26], "a": 0.25, "b": 0.20, "pole": [-0.24, -0.33, -0.36]}
    res = G.solve("neck", hand, prop, seeds=2, workers=1, arm=arm)
    rep = res["report"]
    for f in ("index", "ring", "little"):
        assert abs(rep["contacts"][f]["gap_mm"]) < 1.0, (f, rep["contacts"][f])
    assert rep["elbow_from"] == "pole" and len(rep["elbow"]) == 3
    assert rep["wrist_bend_deg"] < GN.NECK_ARM_FREE + 5.0 and rep["reach"] < G.ARM_REACH + 0.01
    W, S = np.array(rep["wrist_in_N_mm"]) * 1e-3, np.array(arm["shoulder"])
    assert np.linalg.norm(W - S) / (arm["a"] + arm["b"]) == pytest.approx(rep["reach"], abs=2e-3)
    with pytest.raises(ValueError, match="pole"):
        G.solve("neck", hand, prop, seeds=1, workers=1, arm={"shoulder": [0, 0, 0], "a": 0.2, "b": 0.2})


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
    with pytest.raises(ValueError, match="ref"):
        G.solve("neck", hand, prop, workers=1, ref={"angles": 3})
    with pytest.raises(ValueError, match="slots"):
        G.solve("neck", hand, dict(prop, slots=[0.0, 0.01]), workers=1)


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
    # a long body worn low and close, its strap standing well off a jacket: the band still comes to the lower button from
    # behind (the prop's face is 2 cm in front of the button), never out through the prop
    near = (np.array([-0.30, -0.10, 0.85]), np.array([-0.8, 0.0, -0.2]))
    low = WR.strap_path(top, near, shoulder, spine, radius=0.0998, over_side=1.0, standoff=0.03)
    assert low[-10:, 1].min() > near[0][1] - 0.012


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
