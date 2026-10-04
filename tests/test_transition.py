"""Cut effects, bpy-free (mkmmd.core.transition, mkmmd.core.tween): the specs and their errors, the plan against the cut, what
each frame needs rendered, the files, and the timing of a transition and an insert. Fixtures use made-up shot names."""
import re

import pytest

from mkmmd.core import palette as PAL
from mkmmd.core import transition as TR
from mkmmd.core import tween as TW

FPS, F0 = 30, 31
MOON = PAL.get("rose-pine-moon")


# ---------------------------------------------------------------- curves
@pytest.mark.parametrize("kind", TW.EASES)
def test_eases_run_from_0_to_1_without_going_back(kind):
    us = [i / 50 for i in range(51)]
    es = [TW.ease(u, kind) for u in us]
    assert es[0] == 0.0 and es[-1] == 1.0
    assert all(b >= a for a, b in zip(es, es[1:]))


def test_ease_in_is_slow_first_and_out_is_fast_first():
    assert TW.ease(0.5, "in") < 0.5 < TW.ease(0.5, "out")
    assert TW.ease(0.5, "inout") == pytest.approx(0.5)
    with pytest.raises(ValueError):
        TW.ease(0.5, "bounce")


def test_geometric_steps_are_equal_ratios():
    assert TW.geometric(1, 18, 0.0) == 1 and TW.geometric(1, 18, 1.0) == pytest.approx(18)
    a, b, c = (TW.geometric(2, 32, p) for p in (0.25, 0.5, 0.75))
    assert b / a == pytest.approx(c / b)


@pytest.mark.parametrize("overshoot", [0.0, 0.05, 0.12, 0.4])
def test_pop_overshoots_by_the_asked_amount_and_lands_on_one(overshoot):
    us = [i / 2000 for i in range(2001)]
    ys = [TW.pop(u, overshoot) for u in us]
    assert ys[0] == pytest.approx(0.0, abs=1e-9) and ys[-1] == pytest.approx(1.0)
    assert max(ys) == pytest.approx(1.0 + overshoot, abs=2e-3)


def test_stagger_starts_late_and_finishes_early():
    assert TW.stagger(0.1, 0.2, 0.3) == 0.0 and TW.stagger(0.35, 0.2, 0.3) == pytest.approx(0.5) and TW.stagger(0.9, 0.2, 0.3) == 1.0


# ---------------------------------------------------------------- a project to plan against
def shot(name, a, b, **kw):
    return {"name": name, "from": a, "to": b, **kw}


SIL = {"style": "silhouette", "colors": {"background": "love", "subject": "base", "accent": "text"}}


def project(**extra):
    """s1 (silhouette) 0-2 s, l1 2-4, s2 (silhouette) 4-6, l2 6-8, plus a plate shot."""
    data = {"output": [{"name": "16x9"}, {"name": "9x16"}],
            "shot": [shot("s1", 0.0, 2.0, **SIL), shot("l1", 2.0, 4.0), shot("s2", 4.0, 6.0, **SIL), shot("l2", 6.0, 8.0),
                     {"name": "pl", "plate": True, **SIL}]}
    data.update(extra)
    return data


def plan(**extra):
    return TR.plan(project(**extra), FPS, F0, MOON)


# ---------------------------------------------------------------- normalising
def test_an_expand_gets_its_defaults():
    t = TR.normalize_transition({"at": 2.0, "kind": "expand"}, MOON)
    assert t["dur"] == 0.4 and t["scale"] == [1.0, None] and t["turn"] == [0.0, 90.0] and t["ease"] == "in"
    assert t["center"] == {"mode": "subject"} and t["edge"] is None


def test_a_slash_gets_its_defaults_and_resolves_its_colours():
    t = TR.normalize_transition({"at": 2.0, "kind": "slash", "color": "gold", "second": {"color": "#ffffff"}}, MOON)
    assert t["dur"] == 0.18 and t["angle"] == -20.0 and t["width"] == 0.3 and t["dir"] == "right" and t["ease"] == "inout"
    assert t["color"] == pytest.approx(PAL.srgb(MOON["gold"]), abs=1e-5)
    assert t["second"] == {"color": [1.0, 1.0, 1.0], "width": 0.04, "offset": -0.03}


def test_edge_colours_and_widths():
    t = TR.normalize_transition({"at": 1, "kind": "collapse", "edge": {"color": "love", "width": 6}}, MOON)
    assert t["edge"] == {"color": pytest.approx(PAL.srgb(MOON["love"]), abs=1e-5), "width": 6.0}
    t = TR.normalize_transition({"at": 1, "kind": "collapse", "edge": {}}, MOON)
    assert t["edge"]["width"] == 4.0 and t["edge"]["color"] == pytest.approx(PAL.srgb(MOON["text"]), abs=1e-5)


@pytest.mark.parametrize("center, want", [
    ("subject", {"mode": "subject"}), (None, {"mode": "subject"}),
    ([0.4, 0.6], {"mode": "frame", "at": [0.4, 0.6]}),
    ([1, 2, 3], {"mode": "point", "expr": "(1.0, 2.0, 3.0)"}),
    ('bone("spine", "A_arm").head', {"mode": "point", "expr": 'bone("spine", "A_arm").head'})])
def test_centres(center, want):
    spec = {"at": 1, "kind": "expand"}
    if center is not None:
        spec["center"] = center
    assert TR.normalize_transition(spec, MOON)["center"] == want


@pytest.mark.parametrize("spec, frag", [
    ({"kind": "expand"}, "needs `at`"),
    ({"at": 1, "kind": "wipe"}, "kind = 'wipe'"),
    ({"at": 1, "kind": "expand", "angle": 10}, "unknown keys ['angle']"),
    ({"at": 1, "kind": "slash", "scale": [1, 2]}, "unknown keys ['scale']"),
    ({"at": 1, "kind": "expand", "dur": 0}, "dur must be positive"),
    ({"at": 1, "kind": "expand", "ease": "bounce"}, "ease = 'bounce'"),
    ({"at": 1, "kind": "expand", "scale": [18, 1]}, "0 < lo < hi"),
    ({"at": 1, "kind": "expand", "scale": [1]}, "expected [lo, hi]"),
    ({"at": 1, "kind": "expand", "center": [1, 2, 3, 4]}, "center = "),
    ({"at": 1, "kind": "expand", "edge": {"color": "nope"}}, "edge"),
    ({"at": 1, "kind": "expand", "edge": {"width": 0}}, "width must be positive"),
    ({"at": 1, "kind": "expand", "edge": {"thick": 3}}, "unknown keys ['thick']"),
    ({"at": 1, "kind": "slash", "width": 0}, "fraction of the frame diagonal"),
    ({"at": 1, "kind": "slash", "dir": "up"}, "dir = 'up'"),
    ({"at": 1, "kind": "slash", "second": {"width": -1}}, "second.width"),
])
def test_bad_transition_specs_say_what_is_wrong(spec, frag):
    with pytest.raises(TR.TransitionError, match=re.escape(frag)):
        TR.normalize_transition(spec, MOON)


INSERT = {"from": 1.0, "to": 2.0, "shot": "pl", "anchor": 'bone("head").head'}


def test_an_insert_gets_its_defaults():
    ins = TR.normalize_insert(INSERT, MOON)
    assert ins["shape"] == "thought" and ins["size"] == 0.34 and ins["offset"] == [0.12, -0.30] and ins["out"] == "pop"
    assert ins["pop"] == {"dur": 0.3, "overshoot": 0.12} and ins["outline"]["width"] == 5.0 and ins["ratio"] == 1.35
    assert ins["expand"] == {"dur": 0.4, "turn": 0.0, "ease": "in"}


def test_an_aspect_overrides_the_inserts_geometry_only():
    ins = TR.normalize_insert({**INSERT, "aspect": {"9x16": {"size": 0.2, "offset": [0, -0.3]}}}, MOON)
    assert TR.insert_for(ins, "9x16") == {"size": 0.2, "offset": [0.0, -0.3], "ratio": 1.35}
    assert TR.insert_for(ins, "16x9") == {"size": 0.34, "offset": [0.12, -0.3], "ratio": 1.35}
    with pytest.raises(TR.TransitionError, match="unknown keys"):
        TR.normalize_insert({**INSERT, "aspect": {"9x16": {"out": "pop"}}}, MOON)


@pytest.mark.parametrize("patch, frag", [
    ({"shot": None}, "needs `shot`"), ({"to": 0.5}, "`to` must come after"), ({"shape": "star"}, "shape = 'star'"),
    ({"size": 0}, "size = 0"), ({"size": 1.5}, "size = 1.5"), ({"ratio": 0}, "ratio must be positive"),
    ({"out": "fade"}, "out = 'fade'"), ({"pop": {"dur": 0}}, "pop.dur must be positive"),
    ({"pop": {"speed": 2}}, "unknown keys"), ({"expand": {"dur": 0.3}}, "only applies to out = 'expand'"),
    ({"outline": {"width": -1}}, "width must be positive"), ({"extra": 1}, "unknown keys")])
def test_bad_inserts_say_what_is_wrong(patch, frag):
    spec = dict(INSERT)
    for k, v in patch.items():
        if v is None:
            spec.pop(k)
        else:
            spec[k] = v
    with pytest.raises(TR.TransitionError, match=frag):
        TR.normalize_insert(spec, MOON)


# ---------------------------------------------------------------- the plan
def test_a_project_without_effects_plans_nothing():
    p = plan()
    assert p["transitions"] == [] and p["inserts"] == [] and p["plates"] == ["pl"]
    assert [c["name"] for c in p["cuts"]] == ["s1", "l1", "s2", "l2"]
    assert p["cuts"][1] == {"name": "l1", "from": F0 + 60, "to": F0 + 120}


def test_the_window_is_the_dur_seconds_that_end_at_the_cut():
    p = plan(transition=[{"at": 2.0, "kind": "expand", "dur": 0.4}])
    t = p["transitions"][0]
    assert (t["cut"], t["n"], t["first"], t["last"]) == (F0 + 60, 12, F0 + 48, F0 + 59)
    assert (t["out"], t["in"]) == ("s1", "l1")


def test_expand_takes_the_figure_from_the_outgoing_shot_collapse_from_the_incoming_one():
    e, c = plan(transition=[{"at": 2.0, "kind": "expand"}, {"at": 4.0, "kind": "collapse"}])["transitions"]
    assert (e["matte"], e["plate"]) == ("s1", "l1") and (c["matte"], c["plate"]) == ("s2", "l1")


def test_a_slash_has_no_matte_and_shows_the_incoming_shot_behind_the_band():
    t = plan(transition=[{"at": 6.0, "kind": "slash"}])["transitions"][0]
    assert t["matte"] is None and t["plate"] == "l2" and t["n"] == 5 and (t["out"], t["in"]) == ("s2", "l2")


@pytest.mark.parametrize("spec, frag", [
    ({"at": 2.5, "kind": "slash"}, "no shot start at that frame"),
    ({"at": 0.0, "kind": "slash"}, "no shot before the cut"),
    ({"at": 2.0, "kind": "expand", "dur": 2.5}, "starts before shot 's1' does"),
    ({"at": 4.0, "kind": "expand"}, "must be a silhouette"),               # l1 is lit: no figure to grow
    ({"at": 6.0, "kind": "collapse"}, "must be a silhouette")])            # l2 is lit
def test_transitions_that_do_not_fit_the_cut_are_refused(spec, frag):
    with pytest.raises(TR.TransitionError, match=frag):
        plan(transition=[spec])


def test_a_figure_must_be_a_silhouette_in_every_output():
    data = project(transition=[{"at": 2.0, "kind": "expand"}])
    data["shot"][0]["aspect"] = {"9x16": {"style": "none"}}
    with pytest.raises(TR.TransitionError, match="'9x16'"):
        TR.plan(data, FPS, F0, MOON)


def test_windows_may_not_overlap():
    with pytest.raises(TR.TransitionError, match="overlap"):
        plan(transition=[{"at": 2.0, "kind": "expand", "dur": 0.4}], insert=[{**INSERT, "from": 1.2, "to": 1.9}])


def test_plans_are_json_clean():
    import json
    p = plan(transition=[{"at": 2.0, "kind": "expand", "edge": {}}, {"at": 6.0, "kind": "slash", "second": {}}],
             insert=[{**INSERT, "from": 4.5, "to": 5.5}])
    assert json.loads(json.dumps(p)) == p


# ---------------------------------------------------------------- inserts in the plan
def test_an_insert_sits_in_its_host_and_counts_its_pop_and_exit():
    ins = plan(insert=[{**INSERT, "from": 4.5, "to": 5.5}])["inserts"][0]
    assert (ins["host"], ins["f0"], ins["f1"]) == ("s2", F0 + 135, F0 + 165) and (ins["n_in"], ins["n_exit"]) == (9, 9)


def test_an_expanding_insert_ends_where_its_shot_begins_in_the_cut():
    p = plan(insert=[{"from": 2.5, "to": 4.0, "shot": "s2", "anchor": "x", "out": "expand"}])
    ins = p["inserts"][0]
    assert ins["host"] == "l1" and ins["f1"] == F0 + 120 and ins["n_exit"] == 12
    with pytest.raises(TR.TransitionError, match="so `shot` must be that shot, not 'pl'"):
        plan(insert=[{"from": 2.5, "to": 4.0, "shot": "pl", "anchor": "x", "out": "expand"}])
    with pytest.raises(TR.TransitionError, match="grows into the shot that starts at `to`"):
        plan(insert=[{"from": 2.5, "to": 3.5, "shot": "s2", "anchor": "x", "out": "expand"}])


@pytest.mark.parametrize("patch, frag", [
    ({"from": 1.5, "to": 2.5}, "inside one shot"),                      # a cut at 2.0
    ({"shot": "nobody"}, "no shot named 'nobody'"),
    ({"shot": "s1"}, "host shot itself"),
    ({"from": 0.5, "to": 0.9}, "too few")])
def test_inserts_that_do_not_fit_are_refused(patch, frag):
    with pytest.raises(TR.TransitionError, match=frag):
        plan(insert=[{**INSERT, "from": 0.2, "to": 1.0, **patch}])


# ---------------------------------------------------------------- what a frame needs
def items(d, f):
    return sorted((i["kind"], i["shot"]) for i in d.get(f, []))


def test_an_expand_needs_the_figure_and_the_incoming_plate():
    p = plan(transition=[{"at": 2.0, "kind": "expand"}])
    d = TR.demands(p)
    assert sorted(d) == list(range(F0 + 48, F0 + 60))
    assert items(d, F0 + 50) == [("matte", "s1"), ("plate", "l1")]


def test_a_collapse_needs_the_incoming_figure_but_not_a_plate_of_the_shot_in_the_cut():
    d = TR.demands(plan(transition=[{"at": 4.0, "kind": "collapse"}]))
    assert items(d, F0 + 115) == [("matte", "s2")]                      # the plate is the cut's own frame


def test_a_slash_needs_only_the_incoming_plate():
    assert items(TR.demands(plan(transition=[{"at": 6.0, "kind": "slash"}])), F0 + 177) == [("plate", "l2")]


def test_a_centre_that_is_a_point_of_the_scene_is_projected_through_the_figures_camera():
    p = plan(transition=[{"at": 4.0, "kind": "collapse", "center": 'bone("hips").head'}])
    point = [i for i in TR.demands(p)[F0 + 115] if i["kind"] == "point"]
    assert point == [{"kind": "point", "key": "t0", "expr": 'bone("hips").head', "shot": "s2"}]


def test_an_insert_needs_its_plate_and_its_anchor_in_the_host_camera_for_every_frame():
    p = plan(insert=[{**INSERT, "from": 4.5, "to": 5.5}])
    d = TR.demands(p)
    assert sorted(d) == list(range(F0 + 135, F0 + 165))
    assert items(d, F0 + 140) == [("plate", "pl"), ("point", "s2")]
    assert [i for i in d[F0 + 140] if i["kind"] == "point"][0]["key"] == "i0"


def test_demands_can_be_limited_to_some_frames():
    p = plan(transition=[{"at": 2.0, "kind": "expand"}])
    assert sorted(TR.demands(p, [F0 + 10, F0 + 50, F0 + 51])) == [F0 + 50, F0 + 51]


def test_cameras_are_keyed_over_the_frames_other_shots_take_from_them():
    p = plan(transition=[{"at": 2.0, "kind": "expand"}], insert=[{**INSERT, "from": 6.5, "to": 7.5}])
    n = TR.needs(p)
    assert n["l1"] == [F0 + 48, F0 + 59] and n["s1"] == [F0 + 48, F0 + 59] and n["pl"] == [F0 + 195, F0 + 224]
    assert "l2" not in n                                                 # the host's own frames are its own


# ---------------------------------------------------------------- files
def test_each_item_has_its_files_and_the_finishing_one_is_last():
    assert TR.rel_paths({"kind": "plate", "shot": "l1"}, 7) == ["plate/l1/00007.png"]
    assert TR.rel_paths({"kind": "matte", "shot": "s1"}, 12) == ["back/s1/00012.png", "matte/s1/00012.png"]
    assert TR.rel_paths({"kind": "point", "key": "i0"}, 12345) == ["point/i0/12345.json"]
    assert (TR.plate_rel("a", 1), TR.matte_rel("a", 1), TR.back_rel("a", 1), TR.point_rel("k", 1)) == (
        "plate/a/00001.png", "matte/a/00001.png", "back/a/00001.png", "point/k/00001.json")


def test_pending_items_are_the_ones_without_a_finished_file_and_stale_claims_go(tmp_path):
    d = TR.demands(plan(transition=[{"at": 2.0, "kind": "expand"}]), [F0 + 50, F0 + 51])
    assert len(TR.pending_items(tmp_path, d)) == 4
    for rel in TR.rel_paths({"kind": "plate", "shot": "l1"}, F0 + 50):
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_bytes(b"png")
    (tmp_path / TR.matte_rel("s1", F0 + 50)).parent.mkdir(parents=True)
    (tmp_path / TR.matte_rel("s1", F0 + 50)).write_bytes(b"")           # claimed, never finished
    left = TR.pending_items(tmp_path, d)
    assert (F0 + 50, {"kind": "matte", "shot": "s1"}) in left and len(left) == 3
    TR.clear_claims(tmp_path)
    assert not (tmp_path / TR.matte_rel("s1", F0 + 50)).exists() and (tmp_path / TR.plate_rel("l1", F0 + 50)).exists()


# ---------------------------------------------------------------- timing
def tr(kind="expand", **kw):
    kw.setdefault("scale", [1, 18])
    return plan(transition=[{"at": 2.0 if kind == "expand" else 4.0, "kind": kind, **kw}])["transitions"][0]


def test_the_top_scale_may_be_left_to_whatever_fills_the_frame():
    t = tr(scale=[1, "fill"])
    assert t["scale"] == [1.0, None]
    with pytest.raises(ValueError, match="pass hi"):
        TR.pose(t, 3)
    assert TR.pose(t, t["n"] - 1, hi=7.5) == pytest.approx((7.5, 90.0))


def test_an_expand_runs_from_the_figure_itself_to_the_filled_frame():
    t = tr()
    assert TR.pose(t, 0) == pytest.approx((1.0, 0.0)) and TR.pose(t, t["n"] - 1) == pytest.approx((18.0, 90.0))
    scales = [TR.pose(t, k)[0] for k in range(t["n"])]
    assert all(b > a for a, b in zip(scales, scales[1:]))


def test_a_collapse_starts_filled_and_lands_on_the_figure():
    c = tr("collapse")
    assert TR.pose(c, 0) == pytest.approx((18.0, 90.0)) and TR.pose(c, c["n"] - 1) == pytest.approx((1.0, 0.0))
    scales = [TR.pose(c, k)[0] for k in range(c["n"])]
    assert all(b < a for a, b in zip(scales, scales[1:]))


def test_a_collapse_is_an_expand_played_backwards_when_the_ease_is_symmetric():
    e, c = tr(ease="inout"), tr("collapse", ease="inout")
    for k in range(e["n"]):
        assert TR.pose(c, k) == pytest.approx(TR.pose(e, e["n"] - 1 - k))


def test_ease_in_grows_slowly_at_first_and_out_quickly():
    slow, fast = tr(ease="in"), tr(ease="out")
    assert TR.pose(slow, 3)[0] < TR.pose(fast, 3)[0]
    assert TR.pose(slow, 3)[1] < TR.pose(fast, 3)[1]


def test_the_top_scale_can_be_raised_and_the_range_is_the_specs():
    t = tr(scale=[2, 8], turn=[10, -30])
    assert TR.pose(t, 0) == pytest.approx((2.0, 10.0)) and TR.pose(t, t["n"] - 1) == pytest.approx((8.0, -30.0))
    assert TR.pose(t, t["n"] - 1, hi=50.0)[0] == pytest.approx(50.0)


def test_the_matte_must_fill_the_frame_at_the_end_of_an_expand_and_the_start_of_a_collapse():
    e, c = tr(), tr("collapse")
    assert TR.covering_frame(e) == e["last"] and TR.covering_frame(c) == c["first"]


def test_a_slash_is_on_screen_in_every_frame_of_its_window_and_sweeps_forward():
    t = plan(transition=[{"at": 6.0, "kind": "slash"}])["transitions"][0]
    p = [TR.slash_progress(t, k) for k in range(t["n"])]
    assert 0.0 < p[0] and p[-1] < 1.0 and all(b > a for a, b in zip(p, p[1:]))


def ins(**kw):
    return plan(insert=[{**INSERT, "from": 4.5, "to": 5.5, **kw}])["inserts"][0]


def test_an_insert_pops_in_holds_and_pops_out():
    i = ins()
    states = [TR.insert_state(i, f) for f in range(i["f0"], i["f1"])]
    assert [s["phase"] for s in states[:9]] == ["in"] * 9 and [s["phase"] for s in states[-9:]] == ["out"] * 9
    assert all(s["phase"] == "hold" for s in states[9:-9])
    first, last = states[0], states[-1]
    assert first["dots"][0] > 0.2 and first["bubble"] == 0.0              # the circles come first
    assert last["bubble"] == 0.0 and last["dots"] == [0.0, 0.0, 0.0]      # and nothing is left on the last frame
    assert states[8]["bubble"] == pytest.approx(1.0) and states[8]["dots"] == pytest.approx([1.0, 1.0, 1.0])


def test_the_circles_pop_one_after_another_before_the_bubble():
    i = ins(pop={"dur": 0.6, "overshoot": 0.0}, **{"from": 4.1, "to": 5.9})
    seen = []
    for f in range(i["f0"], i["f0"] + i["n_in"]):
        s = TR.insert_state(i, f)
        seen.append((s["dots"][0], s["dots"][1], s["dots"][2], s["bubble"]))
    starts = [next(k for k, v in enumerate(col) if v > 0) for col in zip(*seen)]
    assert starts == sorted(starts) and starts[0] < starts[3]            # dot 0, dot 1, dot 2, then the bubble


def test_a_pop_overshoots_before_settling():
    i = ins(pop={"dur": 0.5, "overshoot": 0.2})
    peak = max(TR.insert_state(i, f)["bubble"] for f in range(i["f0"], i["f0"] + i["n_in"]))
    assert peak > 1.05


def test_an_expanding_insert_holds_then_grows_to_fill_the_frame_on_the_last_frame():
    p = plan(insert=[{"from": 2.5, "to": 4.0, "shot": "s2", "anchor": "x", "out": "expand"}])
    i = p["inserts"][0]
    states = [TR.insert_state(i, f) for f in range(i["f0"], i["f1"])]
    grow = states[-i["n_exit"]:]
    assert [s["phase"] for s in grow] == ["expand"] * i["n_exit"]
    assert grow[0]["p"] == 0.0 and grow[-1]["p"] == pytest.approx(1.0)
    assert all(b["p"] > a["p"] for a, b in zip(grow, grow[1:]))
    assert grow[0]["dots"] == pytest.approx([1.0, 1.0, 1.0]) and grow[-1]["dots"] == [0.0, 0.0, 0.0]
    assert all(s["bubble"] == 1.0 for s in grow)
    assert TR.insert_state(i, i["f0"] - 1)["phase"] == "none"


# ---------------------------------------------------------------- the head the trail starts at
def test_an_insert_stands_for_a_head_of_radius_metres_and_never_a_negative_one():
    assert TR.normalize_insert(INSERT, MOON)["radius"] == 0.11
    assert TR.normalize_insert({**INSERT, "radius": 0.2}, MOON)["radius"] == 0.2
    assert TR.normalize_insert({**INSERT, "radius": 0}, MOON)["radius"] == 0.0
    with pytest.raises(TR.TransitionError, match="radius"):
        TR.normalize_insert({**INSERT, "radius": -0.1}, MOON)


def test_a_point_file_without_its_scale_is_stale_and_goes_with_the_claims(tmp_path):
    import json
    good, old, empty = (tmp_path / "point/i0/00001.json", tmp_path / "point/i0/00002.json", tmp_path / "point/i0/00003.json")
    good.parent.mkdir(parents=True)
    good.write_text(json.dumps({"p": [0.5, 0.5], "depth": 3.0, "m": 0.8}))
    old.write_text(json.dumps({"p": [0.5, 0.5], "depth": 3.0}))
    empty.write_bytes(b"")
    plate = tmp_path / "plate/s/00001.png"
    plate.parent.mkdir(parents=True)
    plate.write_bytes(b"png")
    assert [TR.stale_point(p) for p in (good, old, empty)] == [False, True, True]
    TR.clear_claims(tmp_path)
    assert good.exists() and not old.exists() and not empty.exists() and plate.exists()
