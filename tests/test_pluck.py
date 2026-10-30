"""Playing a bass line with the fingers (mkmmd.core.strum, technique "fingers"; the grip style `pluck`, mkmmd.solvers.grip_pluck):
the technique from the spec or the card, rest strokes alternating the index and middle fingers toward the lower strings, the
fingertip crossing its string at the stroke's time, the hand's stations and where its thumb rests; pure maths on the library
bass's card and a synthetic hand."""
import json
import math

import numpy as np
import pytest

from mkmmd.blender.library.props import electric_bass_layout as BL
from mkmmd.blender.library.props import electric_guitar_layout as GL
from mkmmd.core import fretting as FR
from mkmmd.core import jsonx
from mkmmd.core import strum as SM
from mkmmd.core import tables as TB
from mkmmd.solvers import grip as G
from mkmmd.solvers import grip_pluck as GP
from test_grip import HANDS

CARD = BL.card("bass")
NECK, STRUM = (next(g for g in CARD["use"]["grip"] if g["type"] == k) for k in ("neck", "strum"))
BEAT = 0.5
TL = {"beats": [round(i * BEAT, 4) for i in range(80)], "downbeats": [round(i * BEAT, 4) for i in range(0, 80, 4)]}
SPEC = {"rhythm": "notes", "hand": "R", "prop": "bass", "technique": "fingers"}


def notes_on(pairs):
    """[(t, c, w, string)] of notes given as (t, string) on the library bass."""
    return SM.pick_notes([{"t": t, "string": s} for t, s in pairs], NECK, STRUM)


# ------------------------------------------------------------------------------------------------ which way it is played
def test_the_technique_is_the_specs_then_the_cards_then_the_pick():
    assert SM.technique({}) == "pick" and SM.technique({}, {}) == "pick"
    assert SM.technique({}, STRUM) == "fingers"                                     # the bass is played with the fingers
    guitar = next(g for g in GL.card("guitar")["use"]["grip"] if g["type"] == "strum")
    assert SM.technique({}, guitar) == "pick"                                       # the guitar with a pick
    assert SM.technique({"technique": "pick"}, STRUM) == "pick"                     # the project's table wins
    with pytest.raises(SM.StrumError, match="fingers"):
        SM.technique({"technique": "slap"}, STRUM)
    assert SM.resolved({"rhythm": "notes"}, STRUM)["technique"] == "fingers"
    TB.check({"strum": dict(SPEC, hold=0.2, recover=0.1, shift=0.1, pause=1.0)}, TB.PERFORM, "[perform.rin]")


def test_the_fingers_have_their_own_numbers():
    p = SM.params(SPEC)
    assert p["attack"] == SM.FINGERS["attack"] and p["hold"] == SM.FINGERS["hold"] and p["pause"] == SM.FINGERS["pause"]
    assert SM.params(dict(SPEC, hold=0.1))["hold"] == 0.1                          # the table's own number wins
    assert "hold" not in SM.params({"rhythm": "notes", "technique": "pick"})


def test_fingers_play_a_bass_line_only():
    with pytest.raises(SM.StrumError, match="rhythm = \"notes\""):
        SM.plan(dict(SPEC, rhythm="beats:8"), TL)


# ------------------------------------------------------------------------------------------------ the strokes
def test_index_and_middle_alternate_with_rest_strokes_toward_the_lower_strings():
    times = [1.0, 1.25, 1.5, 1.75, 2.0, 4.0, 4.2, 4.4]                            # a run, a pause, a run
    st = SM.plan(SPEC, TL, notes=notes_on([(t, 3) for t in times]))
    assert [s.t for s in st] == pytest.approx(times)
    assert all(s.dir == -1 for s in st)                                             # every rest stroke pulls toward the lower strings
    assert [s.finger for s in st] == ["index", "middle", "index", "middle", "index", "index", "middle", "index"]
    assert all(s.string == 3 for s in st)
    pick = SM.plan(dict(SPEC, technique="pick"), TL, notes=notes_on([(t, 3) for t in times]))
    assert all(s.finger is None for s in pick) and pick[0].dir == 1                 # a pick alternates down and up


def test_a_rest_stroke_crosses_its_string_at_the_strokes_time_at_a_steady_speed():
    s = SM.plan(SPEC, TL, notes=notes_on([(2.0, 2)]))[0]
    travel = {"hover": 0.007, "cross": 0.0004, "rest": -0.014}
    w = SM.stroke_times(s, 0.11, travel)
    assert w["hover"] < w["cross"] < w["rest"] and w["rest"] - w["hover"] == pytest.approx(0.11)
    v = (travel["hover"] - travel["rest"]) / 0.11                                   # m/s across the string
    u = lambda t: np.interp(t, [w[p] for p in SM.POSES], [travel[p] for p in SM.POSES])    # noqa: E731
    assert u(2.0) == pytest.approx(0.0, abs=1e-9)                                   # the fingertip is over the axis at t
    assert (u(2.0 - 0.01) - u(2.0 + 0.01)) / 0.02 == pytest.approx(v)               # ... moving at the swing's speed
    for phase in np.linspace(0.0, 1.0 / 30.0, 7, endpoint=False):                   # sampled at 30 fps, the crossing is found
        frames = 2.0 + phase + np.arange(-3, 4) / 30.0                              # where it is, whatever the frame phase
        k = int(np.where(np.diff(np.sign(u(frames))) < 0)[0][0])
        a, b = u(frames[k]), u(frames[k + 1])
        assert frames[k] + a / (a - b) * (frames[k + 1] - frames[k]) == pytest.approx(2.0, abs=1e-3)
    assert SM.stroke_times(s, 0.11, {"hover": -0.001}) == SM.stroke_times(s, 0.11)     # a pose out of order: TRAVEL's


def test_pluck_keys_swing_each_finger_through_its_string_and_back_to_hovering():
    st = SM.plan(SPEC, TL, notes=notes_on([(1.0, 4), (1.3, 3), (1.6, 3), (3.5, 1)]))
    keys = SM.pluck_keys(st, None, **SM.params(SPEC))
    for s in st:
        k = keys[s.finger]
        assert (s.t, "cross") in k                                                  # the crossing keyed at the stroke's time
        i = k.index((s.t, "cross"))
        assert k[i - 1][1] == "hover" and k[i + 1][1] == "rest"
        p0, p1, w = SM.pose_track(k, [s.t - 1e-6, s.t + 1e-6])
        assert {p0[0], p1[0]} == {"hover", "cross"} and {p0[1], p1[1]} == {"cross", "rest"}
    for f in ("index", "middle"):
        t = [a for a, _p in keys[f]]
        assert t == sorted(t) and keys[f][-1][1] == "hover"                       # it ends hovering, ready
    p0, p1, w = SM.pose_track(keys["middle"], [1.0])                                 # the other finger hovers meanwhile
    assert p0[0] == p1[0] == "hover"
    s0, s1, ws = SM.station_track(keys["station"], [s.t for s in st])
    assert s0 == s1 == [4, 3, 3, 1]                                                 # at each stroke the hand is at its string's
    first, last = SM.pluck_window(keys)
    assert first < 1.0 - 0.05 and last > 3.5 + SM.FINGERS["hold"]
    with pytest.raises(SM.StrumError, match="no finger"):
        SM.pluck_keys(SM.plan(dict(SPEC, technique="pick"), TL, notes=notes_on([(1.0, 4)])))


def test_fast_notes_cut_the_swing_and_still_lift_in_between():
    st = SM.plan(SPEC, TL, notes=notes_on([(1.0 + 0.09 * i, 2) for i in range(8)]))          # sixteenths at 167 bpm
    keys = SM.pluck_keys(st, None, **SM.params(SPEC))
    for f in ("index", "middle"):
        k = keys[f]
        assert [a for a, _p in k] == sorted(a for a, _p in k)
        crossings = [a for a, p in k if p == "cross"]
        assert len(crossings) == 4
        between = [p for a, p in k if crossings[0] < a < crossings[1]]
        assert "hover" in between                                                   # it lifts over the strings to start again


# ------------------------------------------------------------------------------------------------ the stations
def test_the_thumb_rests_on_the_pickup_for_the_lowest_string_and_on_it_for_the_others():
    prop, (o, R) = SM.pluck_prop(NECK, STRUM, 4)
    assert prop["string"] == 0 and prop["anchor"]["type"] == "plane"
    a = prop["anchor"]
    assert a["normal"] == pytest.approx([0.0, 0.0, 1.0])                            # the cover's top looks out of the face
    e_at = R.T @ (FR.string_point(NECK, 4, STRUM["fingers"]["anchor"]["center"][2]) - o)
    assert a["center"][0] < e_at[0] - BL.STRING_R[0]                                # beyond the low E, toward the bass side
    assert -0.025 < a["center"][2] < -0.003                                         # under the strings' plane, on the cover
    for s in (1, 2, 3):
        p, _ = SM.pluck_prop(NECK, STRUM, s)
        assert p["anchor"] == {"type": "string", "string": 0} and p["string"] == 4 - s
    assert np.allclose(R, np.stack([[1, 0, 0], [0, 0, 1], [0, -1, 0]], 1), atol=1e-12)     # across, along, out of the face
    st = prop["strings"][0]                                                         # the played string runs through the origin
    a0, b0 = np.asarray(st["a"]), np.asarray(st["b"])
    e = (b0 - a0) / np.linalg.norm(b0 - a0)
    assert np.linalg.norm(-a0 - (-a0 @ e) * e) < 1e-9
    assert prop["zone"][0] < 0.0 < prop["zone"][1]
    assert [b["axes"] for b in prop["blocks"]] == [R.T.tolist()] * 2
    with pytest.raises(SM.StrumError, match="fingers"):
        SM.pluck_prop(NECK, {k: v for k, v in STRUM.items() if k != "fingers"}, 2)


# ------------------------------------------------------------------------------------------------ the grip style
def _station(string):
    """The library bass's station problem for a string (the solver works in its own frame: where the hand model sits does
    not matter)."""
    return SM.pluck_prop(NECK, STRUM, string)[0]


@pytest.fixture(scope="module")
def solved():
    return GP.solve_pluck(HANDS["R"], _station(3), seeds=1, workers=1, nfev=(30, 60, 30))


def test_the_pluck_style_solves_a_station_and_every_pose(solved):
    rep = solved["report"]
    assert solved["style"] == "pluck" and set(rep["contacts"]) == {"thumb", "index", "middle"}
    assert set(solved["poses"]) == {f"{f}.{p}" for f in GP.PLUCKERS for p in GP.POSES}
    for f in GP.PLUCKERS:
        names = {HANDS["R"].names[i] for i in HANDS["R"].F[f][:3]}
        assert all(set(q) == names for k, q in solved["poses"].items() if k.startswith(f + "."))
        t = rep["travel"][f]
        assert t["hover"] > t["cross"] > t["rest"]                                  # hovering toward the higher strings, resting past
    assert json.loads(json.dumps(jsonx.to_jsonable(rep)))["travel"] == rep["travel"]


def test_the_pluck_style_is_registered_and_checks_its_input():
    assert "pluck" in G.STYLES
    with pytest.raises(ValueError, match="pluck prop"):
        G.solve("pluck", HANDS["R"], {"strings": []}, seeds=1, workers=1)
    with pytest.raises(ValueError, match="tuning"):
        G.solve("pluck", HANDS["R"], _station(2), seeds=1, workers=1, colour=3)
    with pytest.raises(ValueError, match="anchor"):
        GP._Pluck(HANDS["R"], **{**_station(2), "anchor": {"type": "cloud"}})


def test_retime_makes_the_fingertip_move_at_a_steady_speed():
    us = [0.006, 0.0058, 0.0052, 0.004, 0.0021, 0.0009, 0.0003, 0.0001, 0.0]      # slow at the start, as a blend can be
    grid = np.linspace(0.0, 1.0, len(us))
    for w in (0.0, 0.2, 0.5, 0.9, 1.0):
        at = SM.retime(w, us)
        assert np.interp(at, grid, us) == pytest.approx(0.006 * (1.0 - w), abs=1e-9)  # where a steady swing has it
    assert SM.retime(0.3, [0.0, 0.002, 0.001]) == 0.3                              # not monotone: left alone
    assert SM.retime(0.4, list(reversed(us))) == pytest.approx(1.0 - SM.retime(0.6, us))


def test_a_picking_hand_turns_about_its_wrist_across_the_strings():
    centre, across = np.array([0.0, -0.017, 0.10]), np.array([1.0, 0.0, 0.0])
    pivot = centre + np.array([-0.05, -0.04, -0.06])                               # the wrist above the bass side, toward the bridge
    axis, lever = SM.wrist_turn(pivot, centre, across)
    r = centre - pivot
    r = r - (r @ across) * across
    assert lever == pytest.approx(np.linalg.norm(r)) and abs(axis @ across) < 1e-12 and abs(axis @ r) < 1e-12
    th = 1e-4                                                                       # a small turn moves the tip across the strings
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    moved = (np.eye(3) + math.sin(th) * K + (1 - math.cos(th)) * K @ K) @ (centre - pivot) - (centre - pivot)
    assert moved @ across / th == pytest.approx(lever, rel=1e-4)                     # ... by `lever` metres per radian
    assert SM.wrist_turn(centre + 0.01 * across, centre, across) == (None, 0.0)    # no lever to turn
    assert SM.params({"rhythm": "notes", "technique": "pick"})["share"] == SM.PICK["share"]
