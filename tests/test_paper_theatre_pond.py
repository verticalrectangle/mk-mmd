"""The paper theatre's pond and its machinery (mkmmd.core.paper_theatre_pond and its flipbooks in
mkmmd.blender.library.sets.paper_theatre_parts, no bpy): what lands on its moment lands on the held step that shows it
(a pad under her feet, the lilies open, the strokes drawn, the sun up), the night is uncovered whole, the ground and the
cut-loose pieces leave the opening and come back at day, and every new piece of paper is a closed solid."""
import math

import numpy as np
import pytest

from mkmmd.blender.library.sets import paper_theatre_parts as P
from mkmmd.core import palette as PAL
from mkmmd.core import paper_theatre as PT
from mkmmd.core import paper_theatre_pond as PP
from mkmmd.core import shotstyle as SS

HOPS = [7.19, 7.58, 7.95, 8.35, 8.71, 9.12]
STROKES = [9.46, 9.83, 10.21, 10.57, 10.96, 11.34, 11.73]
SPEC = {"name": "theatre", "kind": "paper_theatre", "width": 2.6, "step": 12, "step_from": 3.03,
        "pond": {"lilies": 4.54, "ripples": {"touch": 0.0, "then": [1.53, 4.54]}},
        "panorama": {"hops": HOPS}, "rise": {"from": 9.15, "to": 10.6},
        "night": {"from": 9.15, "to": 10.6, "strokes": STROKES}, "moon": {"shape": "croissant", "drops": 10.99},
        "loose": 12.08, "sun": {"pops": 20.47}, "day": 20.2, "house": [[3.03, 22.68]]}


def _scene(**kw):
    o = PT.options(dict(SPEC, **kw))
    lay, pond = PT.scene(o)
    return o, lay, pond


def _held(o, t):
    """The clip second of the held step that shows clip second t (60 fps, frame0 181, the set's step grid)."""
    _keys, times, _ = PT.key_times(1, 1800, 181, 60.0, o["step"], o["step_from"])
    frame = 181 + round(t * 60.0)
    shown = (np.round(times * 60.0) + 181).astype(int)
    return float(times[np.searchsorted(shown, frame, side="right") - 1])


def _values(o, lay, pond, t):
    ch, hidden, flips = PP.motion(o, lay, pond, np.atleast_1d(np.asarray(t, float)))
    return {(c.object, c.path, c.index): c.values for c in ch}, hidden, flips


def test_options_check_the_pond_keys_and_a_plain_theatre_is_left_alone():
    with pytest.raises(PT.TheatreError) as e:
        PT.options(dict(SPEC, panorama={"hops": HOPS, "towards": 1}))
    assert "towards" in str(e.value) and "toward" in str(e.value)
    for bad in ({"rise": {"from": 9.0}}, {"panorama": {"hops": [8.0, 7.0]}}, {"night": {"from": 1, "to": 2,
                                                                                       "strokes": [1.0, 1.0]}},
                {"moon": {"shape": "bagel"}}, {"night": {"from": 1, "to": 2, "strokes": list(range(1, 10))}}):
        with pytest.raises(PT.TheatreError):
            PT.options(dict(SPEC, **bad))
    plain = PT.options({"name": "t", "kind": "paper_theatre"})
    lay, pond = PT.scene(plain)
    assert pond is None and lay is PT.layout(2.2, 3.2, 2.35, 1)
    card = PT.card("t", plain, lay, pond)
    assert "rest" not in card["use"] and set(card["materials"]) == {f"t_{r}" for r in PT.ROLES}


def test_a_pad_is_under_her_feet_on_every_hop_and_the_lane_goes_the_other_way():
    """On the held step that shows each landing a landing pad's middle is within a centimetre of the origin; between
    landings the lane moves against her travel and keeps its pads out of the keep-out of her threads (they lie flat)."""
    for toward in (1, -1):
        o, lay, pond = _scene(panorama={"hops": HOPS, "toward": toward})
        assert len(pond.landings) == len(HOPS) and pond.landings[0] == PP.PAD_X
        for k, hop in enumerate(HOPS):
            val, _h, _f = _values(o, lay, pond, _held(o, hop))
            lane_x = val[("lane", "location", 0)][0]
            assert abs(pond.landings[k] + lane_x) < 0.01, (toward, k, pond.landings[k] + lane_x)
        val, _h, _f = _values(o, lay, pond, np.linspace(HOPS[0], HOPS[-1], 200))
        assert np.all(np.diff(val[("lane", "location", 0)]) * toward <= 1e-9)
        pads = [p for p in pond.lane if p.name.startswith("pad")]
        assert all(abs(p.y) + p.r < min(abs(y) for y in PP.WING_Y) for p in pads[1:]) and pads[0].r == PP.PAD[0]
        assert PP.PAD[1] < min(abs(y) for y in PP.WING_Y)          # the main pad, an oval, passes the wings too
        lane0 = _values(o, lay, pond, [3.5])[0][("lane", "location", 0)][0]
        assert lane0 == 0.0 and PP.PAD_X + PP.PAD[0] > 0.0 > PP.PAD_X - PP.PAD[0] + 0.75   # lying: her head on it


def test_the_lilies_are_shut_until_the_tab_is_pulled_and_open_on_it():
    o, lay, pond = _scene()
    t = np.array([3.5, 4.3, _held(o, 4.54), 6.0])
    val, hidden, flips = _values(o, lay, pond, t)
    states, index = flips["lily1"]
    openness = states[index]
    assert openness[0] == 0.0 and openness[1] == 0.0
    assert openness[2] >= 1.0 and openness[3] == pytest.approx(1.0, abs=0.02)
    tab = val[("tab", "location", 2)]
    assert tab[0] == pytest.approx(pond.tab_at[2]) and tab[2] <= pond.tab_at[2] - 0.95 * PP.TAB_PULL[0]
    look = {u["name"]: u for u in PT.card("theatre", o, lay, pond)["use"]["look"]}
    assert look["tab"]["object"] == "theatre_tab" and look["crank"]["object"] == "theatre_crank_knob"
    m, positions = P.lily_mesh(pond.lilies[0], 3)
    shut, op = positions(0.0), positions(1.0)
    assert shut.shape == op.shape == m.V.shape and m.is_closed()
    assert np.hypot(shut[:, 0], shut[:, 1]).max() < 0.06 < 0.15 < np.hypot(op[:, 0], op[:, 1]).max()


def test_the_night_is_uncovered_whole_and_each_stroke_lands_on_its_note():
    """The day halves part from the middle: by the first stroke its stars are uncovered, by `to` both halves are wound
    past their spools; each stroke is drawn whole on the held step that shows its time and not before it starts."""
    o, lay, pond = _scene()
    a, R, D, yf, S = pond.sky
    halves = {h.name: P.half_mesh(h, 1) for h in pond.halves}
    for k, stroke in enumerate(pond.strokes):
        t_on = _held(o, STROKES[k])
        _val, hidden, flips = _values(o, lay, pond, [t_on - 0.25, t_on])
        states, index = flips[stroke.name]
        assert states[index[1]] == pytest.approx(1.0) and hidden[stroke.name][0]
        s_stars = [s for seg in stroke.segments for s, _z in seg]
        first = stroke.segments[0][0][0]
        for when, stars in ((t_on - PP.DRAW, [first]), (t_on, s_stars)):    # its first star as it starts, all at the end
            left, right = PP.halves_offset(o, pond, np.array([when]))
            assert left[0] < min(stars) - PP.STAR_R and max(stars) + PP.STAR_R < -0.02 + right[0], k
    left, right = PP.halves_offset(o, pond, np.array([10.6, 15.0]))
    for name, d in (("day_L", left), ("day_R", right)):
        u = halves[name].V[:, 0]
        for dd in d:
            s = u + dd
            assert (s > S).all() or (s < -S).all(), name
    _val, _h, flips = _values(o, lay, pond, [20.0, 20.3])
    for name in ("day_L", "day_R"):
        states, index = flips[name]
        assert states[index[1]] == 0.0                     # back at day


def test_the_halves_wind_onto_their_spools_without_a_jump():
    o, lay, pond = _scene()
    a, R, D, yf, S = pond.sky
    s = np.linspace(S - 0.05, S + 0.12, 341)                # up to just short of half a turn on the spool
    for y in PP.HALF_Y:
        V = PP.wind(np.stack([s, np.full_like(s, y), np.full_like(s, 1.0)], 1), 0.0, pond.sky)
        step = np.linalg.norm(np.diff(V, axis=0), axis=1)
        assert step.max() < 1.5 * (s[1] - s[0])            # paper bends round the spool, it never tears
        off = np.hypot(V[:, 0] - PP.spool_x(pond.sky), V[:, 1] - yf)
        assert off[s > S + 0.005].min() >= PP.SPOOL_R       # wound on the spool, never through it


def test_the_ground_sinks_out_and_the_clouds_go_down_past_her_then_all_comes_back_at_day():
    o, lay, pond = _scene()
    t = np.array([9.15, 10.6, 11.0, 20.25])
    val, hidden, _f = _values(o, lay, pond, t)
    floor = next(p for p in lay.statics if p.name == "floor")
    z = val[("floor", "location", 2)]
    assert z[0] == pytest.approx(floor.at[2]) and z[1] == pytest.approx(floor.at[2] - o["rise"]["sink"])
    for nm in ("floor", "wave1", "wave5", "wing_L2", "hills_back", "sun"):
        assert list(hidden[nm]) == [False, True, True, False], nm
    assert z[3] == pytest.approx(floor.at[2])
    sky = next(p for p in lay.statics if p.name == "sky")
    low = sky.plates[0].outline[:, 1].min()                    # sky under her once the floor has gone, in the cellar
    assert low == pytest.approx(pond.z_low, abs=1e-6) and low < -o["rise"]["sink"]
    for h in lay.hangs:
        if h.kind == "cloud":
            bottom = h.anchor[2] + val[(h.name, "location", 2)][1] - h.size[1]
            assert bottom < 1.0 < h.anchor[2] - h.length - h.size[1], h.name    # from over her to under her


def test_cut_loose_every_piece_falls_out_and_the_croissant_hangs_clear_of_her_threads():
    o, lay, pond = _scene()
    t = np.array([12.0, 12.4, 13.5, 20.25])
    val, hidden, _f = _values(o, lay, pond, t)
    for h in lay.hangs:
        z = val[(h.name + "_hang", "location", 2)]
        assert z[0] == pytest.approx(h.anchor[2]) and z[1] < h.anchor[2] - 0.5
        assert hidden[h.name][2], h.name                    # out of the opening a second later
        assert not hidden[h.name][3] if h.kind == "cloud" else hidden[h.name][3]      # day: clouds back, the rest up
    moon = next(h for h in lay.hangs if h.name == "moon")
    assert moon.kind == "croissant" and P.part_mesh(moon.part, 1).is_closed()
    kx, ky = PT.KEEP_OUT
    side, fb, twist = PT.amplitudes(o, moon)
    rx = np.abs(np.concatenate([p.outline for p in moon.part.plates])[:, 0]).max()
    reach = (moon.length + moon.size[1]) * math.sin(side)
    assert abs(moon.anchor[0]) - rx - reach >= kx or abs(moon.anchor[1]) - rx * math.sin(twist) >= ky


def test_the_sun_pops_up_on_its_time_and_spins_down():
    o, lay, pond = _scene()
    on = _held(o, 20.47)
    t = np.array([4.0, 20.22, on, on + 1.5])
    val, hidden, _f = _values(o, lay, pond, t)
    z = val[("sun", "location", 2)]
    assert z[0] == pytest.approx(pond.sun_up) and z[1] == pytest.approx(pond.sun_down)
    assert z[2] >= pond.sun_up - 1e-6 and abs(z[3] - pond.sun_up) < 0.01
    spin = val[("sun_rays", "rotation_euler", 1)] - 2.0 * math.pi * o["sun_rpm"] / 60.0 * t
    assert spin[1] == 0.0 and spin[3] == pytest.approx(2.0 * math.pi * PP.SUN_SPIN[0], rel=0.02)
    rays = next(p for p in lay.statics if p.name == "sun_rays").plates[0].outline
    hw = 0.5 * lay.width
    humps = [(-1.05 * hw / 1.1, 0.62, 1.0), (0.42 * hw / 1.1, 0.55, 1.1), (1.45, 0.40, 0.8)]
    ridge = float(PT.ridge(np.array([lay.sun[0]]), humps, 0.30)[0]) - PP.W0
    assert pond.sun_down + rays[:, 1].max() < ridge < pond.sun_up        # hidden behind the hills, then over them


def test_every_new_paper_piece_is_a_closed_outward_solid():
    o, lay, pond = _scene()
    parts = [r.part for r in pond.rows] + pond.wings + [p.part for p in pond.lane] + [pond.tab, pond.crank, pond.knob]
    parts += [p for p in lay.statics if p.name in ("starlight", "sun", "sun_rays", "spool_L", "cellar", "cover", "base")]
    parts.append(next(h.part for h in lay.hangs if h.name == "moon"))
    for part in parts:
        m = P.part_mesh(part, 1)
        assert m.is_closed() and m.volume() > 0, part.name
    for h in pond.halves:
        assert P.half_mesh(h, 1).is_closed()
    for s in pond.strokes[:2]:
        m, positions = P.stroke_mesh(s, pond.sky)
        assert m.is_closed() and positions(0.5).shape == m.V.shape


def test_the_house_goes_on_the_cuts_and_leaves_water_pads_and_buds():
    """Outside its windows only the water (2 m and more round her), the pads, the lilies and the ripples stay, and the
    house switches on the cut frames themselves; the lilies are buds at 0 and at the end (the loop's join)."""
    o, lay, pond = _scene()
    ks, gone = PP.house_keys(o, 181, 60.0, 1)
    assert list(ks) == [0.5, 181 + 182 - 0.5, 181 + 1361 - 0.5] and list(gone) == [True, False, True]
    keep = PP.keeps(pond)
    assert {"cover", "pad1", "lily1", "ripple1"} <= keep
    assert not keep & {"floor", "base", "sky", "wave1", "wing_L1", "reeds1", "tab", "cellar", "day_L", "starlight"}
    cover = next(p for p in lay.statics if p.name == "cover").plates[0].outline
    assert np.hypot(cover[:, 0], cover[:, 1]).min() > 2.0
    assert 0.0 < PP.RIDE_GIVE < PP.W0 - 0.002              # riding her, the main pad's top never goes under the water
    _val, _hidden, flips = _values(o, lay, pond, [0.0, 25.69])
    states, index = flips["lily1"]
    assert states[index[0]] == 0.0 and states[index[1]] == 0.0


def test_three_ripples_leave_the_pad_on_the_touch_and_turn_each_their_own_way():
    o, lay, pond = _scene()
    rings = pond.ripples
    assert [r.t0 for r in rings[:3]] == [0.0, PP.RIPPLE_GAP, 2 * PP.RIPPLE_GAP] and len(rings) == 5
    assert rings[3].reach < rings[0].reach and rings[0].reverse != rings[1].reverse
    _val, hidden, flips = _values(o, lay, pond, [-0.1, 0.0, 1.0, 2.1])
    assert list(hidden["ripple1"]) == [True, False, False, True]
    m, positions = P.ripple_mesh(rings[0], PP.RIPPLE[3])
    rim = np.hypot(positions(0.0)[:, 0] / PP.PAD[0], positions(0.0)[:, 1] / PP.PAD[1]).min()
    out = np.hypot(positions(0.99)[:, 0], positions(0.99)[:, 1]).max()
    assert m.is_closed() and 0.99 < rim < 1.03 and out > PP.PAD[0] + 0.85 * rings[0].reach
    roles = [PT.MATS[i] for i in m.Qm[::6]]
    assert roles[:4] == ["ripple1", "ripple2", "ripple3", "ripple4"]
    back = [PT.MATS[i] for i in P.ripple_mesh(rings[1], PP.RIPPLE[3])[0].Qm[::6]]
    assert back[:4] == ["ripple4", "ripple3", "ripple2", "ripple1"]


def test_the_ripples_read_as_water_first_their_tones_close_in_the_drifts_order():
    """By default a ring's four tones keep `contrast` of the roles' spread about the same mean, still light, light
    middle, dark, dark middle (the order that seems to turn); `contrast = 1` is the roles' own, outside 0..1 an error."""
    dawn, tones = PAL.PALETTES["rose-pine-dawn"], ("ripple1", "ripple2", "ripple3", "ripple4")

    def luma(c):
        return float(np.dot([0.2126, 0.7152, 0.0722], c))
    pond = dict(SPEC["pond"], ripples={"touch": 0.0, "contrast": 1.0})
    own = PT.colours(PT.options(dict(SPEC, pond=pond)), dawn)
    soft = PT.colours(PT.options(SPEC), dawn)
    assert all(np.allclose(own[r], SS.resolve_colour(PT.POND_ROLES[r][0], dawn)) for r in tones)
    L_own, L = [luma(own[r]) for r in tones], [luma(soft[r]) for r in tones]
    assert L[0] > L[1] > L[3] > L[2]
    assert np.isclose(max(L) - min(L), PP.RIPPLES["contrast"] * (max(L_own) - min(L_own)))
    assert np.allclose(np.mean([soft[r] for r in tones], 0), np.mean([own[r] for r in tones], 0))
    with pytest.raises(PT.TheatreError):
        PT.options(dict(SPEC, pond=dict(SPEC["pond"], ripples={"touch": 0.0, "contrast": 1.5})))
