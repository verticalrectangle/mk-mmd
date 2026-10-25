"""The slice and spin transitions (mkmmd.core.transition, mkmmd.cutfx): their specs and errors, their windows in the plan (a
spin's straddles the cut), what they need rendered, their timing, and the composite on tiny synthetic frames: the halves of a
slice parting across the cut and off the frame by it, the edge on their cut sides, the cut through a figure's centroid; a
spin turning faster to the cut and slowing to rest after it, smeared along its turn. Fixtures use made-up shot names."""
import math
import re

import cv2
import numpy as np
import pytest

from mkmmd import cutfx as CF
from mkmmd.core import palette as PAL
from mkmmd.core import transition as TR

FPS, F0 = 30, 1
W, H = 96, 54
MOON = PAL.get("rose-pine-moon")
SIL = {"style": "silhouette"}


def project(transitions, inserts=(), outputs=("t",)):
    """a (silhouette) 0-2 s, b 2-4, c 4-6."""
    return {"output": [{"name": o} for o in outputs],
            "shot": [{"name": "a", "from": 0.0, "to": 2.0, **SIL}, {"name": "b", "from": 2.0, "to": 4.0},
                     {"name": "c", "from": 4.0, "to": 6.0}],
            "transition": list(transitions), "insert": list(inserts)}


def plan(*transitions, **kw):
    return TR.plan(project(transitions, **kw), FPS, F0, MOON)


# ---------------------------------------------------------------- specs
def test_a_slice_gets_its_defaults():
    t = TR.normalize_transition({"at": 2.0, "kind": "slice"}, MOON)
    assert t["dur"] == 0.3 and t["ease"] == "in" and t["angle"] == -20.0 and t["edge"] is None
    assert t["center"] == {"mode": "frame", "at": [0.5, 0.5]}


def test_a_slice_cuts_through_a_point_or_the_figure_with_an_edge():
    t = TR.normalize_transition({"at": 2.0, "kind": "slice", "center": "figure", "edge": {"color": "#ffffff", "width": 14}},
                                MOON)
    assert t["center"] == {"mode": "figure"} and t["edge"] == {"color": [1.0, 1.0, 1.0], "width": 14.0}
    assert TR.normalize_transition({"at": 2.0, "kind": "slice", "center": [0.3, 0.6]}, MOON)["center"] == {
        "mode": "frame", "at": [0.3, 0.6]}


def test_a_spin_gets_its_defaults_and_has_no_ease():
    t = TR.normalize_transition({"at": 2.0, "kind": "spin"}, MOON)
    assert (t["dur"], t["after"], t["turns"], t["blur"], t["scale"], t["back"]) == (0.3, 0.3, 1.5, 1.0, 0.9, "mirror")
    assert t["center"] == {"mode": "frame", "at": [0.5, 0.5]} and "ease" not in t
    t = TR.normalize_transition({"at": 2.0, "kind": "spin", "dur": 0.2, "back": "#000000"}, MOON)
    assert t["after"] == 0.2 and t["back"] == [0.0, 0.0, 0.0]                  # `after` defaults to `dur`


@pytest.mark.parametrize("spec, frag", [
    ({"kind": "slice", "center": [1.2, 0.5]}, "frame fractions in 0..1"),
    ({"kind": "slice", "center": "subject"}, "center = 'subject'"),
    ({"kind": "slice", "center": [0.5, 0.5, 1.0]}, "center = "),
    ({"kind": "slice", "turns": 2}, "unknown keys ['turns']"),
    ({"kind": "slice", "edge": {"width": 0}}, "width must be positive"),
    ({"kind": "slice", "ease": "bounce"}, "ease = 'bounce'"),
    ({"kind": "spin", "ease": "in"}, "unknown keys ['ease']"),
    ({"kind": "spin", "turns": 0}, "turns = 0"),
    ({"kind": "spin", "blur": 5}, "blur = 5.0"),
    ({"kind": "spin", "scale": 0}, "scale = 0.0"),
    ({"kind": "spin", "after": 0}, "after = 0.0"),
    ({"kind": "spin", "back": "nope"}, "back"),
    ({"kind": "spin", "center": [0.5, -0.1]}, "frame fractions in 0..1"),
])
def test_bad_slices_and_spins_say_what_is_wrong(spec, frag):
    with pytest.raises(TR.TransitionError, match=re.escape(frag)):
        TR.normalize_transition({"at": 2.0, **spec}, MOON)


# ---------------------------------------------------------------- the plan
def test_a_slice_window_ends_at_the_cut_and_needs_only_the_incoming_plates():
    p = plan({"at": 2.0, "kind": "slice"})
    t = p["transitions"][0]
    assert (t["first"], t["last"], t["n"], t["cut"]) == (F0 + 51, F0 + 59, 9, F0 + 60)
    assert (t["out"], t["in"], t["plate"], t["matte"]) == ("a", "b", "b", None)
    d = TR.demands(p)
    assert sorted(d) == list(range(t["first"], t["last"] + 1))
    assert all(items == [{"kind": "plate", "shot": "b"}] for items in d.values())


def test_a_slice_through_the_figure_reads_the_outgoing_figure_on_its_first_frame():
    p = plan({"at": 2.0, "kind": "slice", "center": "figure"})
    t = p["transitions"][0]
    assert t["matte"] == "a"
    d = TR.demands(p, [t["first"] + 4])                                      # a look at one frame inside the window ...
    assert d[t["first"]] == [{"kind": "matte", "shot": "a"}]                 # ... still gets the figure the cut runs through
    assert d[t["first"] + 4] == [{"kind": "plate", "shot": "b"}]
    assert TR.demands(p, [t["first"] - 5]) == {}                             # nothing outside the window
    assert TR.needs(p)["a"] == [t["first"], t["first"]]


def test_a_slice_through_the_figure_needs_a_flat_outgoing_shot():
    with pytest.raises(TR.TransitionError, match="must have a flat look.*centroid"):
        plan({"at": 4.0, "kind": "slice", "center": "figure"})              # b is lit
    data = project([{"at": 2.0, "kind": "slice", "center": "figure"}], outputs=("16x9", "9x16"))
    data["shot"][0]["aspect"] = {"9x16": {"style": "none"}}
    with pytest.raises(TR.TransitionError, match="'9x16'"):
        TR.plan(data, FPS, F0, MOON)


def test_a_spin_window_straddles_the_cut_and_needs_no_layers():
    p = plan({"at": 2.0, "kind": "spin", "dur": 0.3, "after": 0.4})
    t = p["transitions"][0]
    assert (t["first"], t["cut"], t["last"], t["n"]) == (F0 + 51, F0 + 60, F0 + 71, 21)
    assert (t["out"], t["in"], t["plate"], t["matte"]) == ("a", "b", None, None)
    assert TR.demands(p) == {} and TR.needs(p) == {}
    assert TR.summary(p)["transitions"][0]["frames"] == [F0 + 51, F0 + 71]


def test_a_spin_must_come_to_rest_inside_the_incoming_shot():
    with pytest.raises(TR.TransitionError, match=re.escape("runs past the end of shot 'b' (it is 60 frames long); shorten `after`")):
        plan({"at": 2.0, "kind": "spin", "after": 2.5})
    plan({"at": 2.0, "kind": "spin", "after": 2.0})                          # 60 frames: the whole shot


def test_a_spin_after_the_cut_may_not_overlap_an_insert_there():
    ins = {"from": 2.2, "to": 3.0, "shot": "c", "shape": "split", "in": 0, "out": 0}
    with pytest.raises(TR.TransitionError, match="overlap"):
        plan({"at": 2.0, "kind": "spin", "after": 0.4}, inserts=[ins])
    plan({"at": 2.0, "kind": "spin", "after": 0.2}, inserts=[ins])          # 6 frames: done before it


# ---------------------------------------------------------------- timing
def test_slice_reach_is_from_the_cut_to_the_farthest_corner():
    assert TR.slice_reach((100, 50), (49.5, 24.5), TR.slice_normal(0)) == pytest.approx((51.0, 51.0))
    assert TR.slice_reach((100, 50), (49.5, 24.5), TR.slice_normal(90)) == pytest.approx((26.0, 26.0))
    plus, minus = TR.slice_reach((100, 50), (19.5, 24.5), TR.slice_normal(0))      # off centre: the far half goes farther
    assert (plus, minus) == pytest.approx((81.0, 21.0))
    nx, ny = TR.slice_normal(20)                                             # clockwise: the cut's top leans right
    assert nx > 0 and ny > 0


def test_slice_progress_is_inside_0_1_in_every_window_frame_and_moves_on():
    t = plan({"at": 2.0, "kind": "slice"})["transitions"][0]
    p = [TR.slice_progress(t, k) for k in range(t["n"])]
    assert 0.0 < p[0] and p[-1] < 1.0 and all(b > a for a, b in zip(p, p[1:]))
    assert TR.slice_progress(t, t["n"]) == 1.0                               # gone at the cut


def spin(**kw):
    return plan({"at": 2.0, "kind": "spin", **kw})["transitions"][0]


def test_a_spin_turns_its_turns_from_rest_to_rest_faster_to_the_cut_and_slower_after():
    t = spin(dur=0.3, after=0.5, turns=1.5)
    assert TR.spin_angle(t, t["first"] - 1) == 0.0 and TR.spin_angle(t, t["last"] + 1) == pytest.approx(540.0)
    ts = [t["first"] - 1 + k * 0.25 for k in range(4 * (t["last"] - t["first"] + 2) + 1)]
    a = [TR.spin_angle(t, x) for x in ts]
    speed = np.diff(a)
    assert all(s >= 0 for s in speed)
    tc = t["cut"] - 0.5
    before = [s for x, s in zip(ts, speed) if x + 0.25 <= tc]
    after = [s for x, s in zip(ts, speed) if x >= tc]
    assert all(b > a_ for a_, b in zip(before, before[1:])) and all(b < a_ for a_, b in zip(after, after[1:]))
    eps = 1e-4                                                               # the same top speed on both sides of the cut
    left = (TR.spin_angle(t, tc) - TR.spin_angle(t, tc - eps)) / eps
    right = (TR.spin_angle(t, tc + eps) - TR.spin_angle(t, tc)) / eps
    assert left == pytest.approx(right, rel=1e-3) and left == pytest.approx(2 * 540 / (t["n"] + 1), rel=1e-3)


def test_a_spin_shows_the_outgoing_shot_then_the_incoming_one_landing_upright_with_a_dip_in_scale():
    t = spin(dur=0.3, after=0.3, scale=0.8)
    first, last, cut = t["first"], t["last"], t["cut"]
    assert TR.spin_state(t, cut - 1)["picture"] == "out" and TR.spin_state(t, cut)["picture"] == "in"
    assert TR.spin_state(t, last + 1)["turn"] == pytest.approx(0.0)         # at rest, upright
    assert TR.spin_state(t, first - 1)["turn"] == 0.0
    scales = [TR.spin_state(t, f)["scale"] for f in range(first - 1, last + 2)]
    assert scales[0] == pytest.approx(1.0) and scales[-1] == pytest.approx(1.0) and min(scales) == pytest.approx(0.8, abs=0.01)
    smear = [np.diff(TR.spin_state(t, f)["smear"])[0] for f in range(first, last + 1)]
    k = cut - first
    assert max(smear) in (smear[k - 1], smear[k]) and smear[0] < smear[k] / 4 and smear[-1] < smear[k] / 4


def test_negative_turns_spin_counter_clockwise_and_no_blur_is_sharp():
    t = spin(turns=-1.0, blur=0.0)
    assert TR.spin_angle(t, t["cut"]) < 0 and TR.spin_state(t, t["cut"])["smear"] == [0.0, 0.0]


# ---------------------------------------------------------------- compositing
def q8(c):
    return tuple(round(v * 255) / 255 for v in c)


A, B, BACK = q8((0.9, 0.3, 0.4)), q8((0.2, 0.4, 0.9)), q8((0.1, 0.8, 0.2))


def ramp(colour):
    """A picture whose blue channel counts the column: where a pixel came from can be read back."""
    img = np.empty((H, W, 3), np.float32)
    img[:] = colour
    img[..., 2] = (np.arange(W, dtype=np.float32) / (W - 1))[None, :]
    return img


def save_rgb(path, img):
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), (np.clip(img, 0, 1)[..., ::-1] * 255 + 0.5).astype(np.uint8))


def fx_for(tmp_path, spec, figure=None):
    """A CutFx over a slice or a spin at 2 s, its incoming plates flat B and, for a cut through the figure, a disc as the
    outgoing shot's figure (`figure` = its centre, frame pixels; the matte is drawn at twice the frame's size)."""
    p = plan(spec)
    for f, items in TR.demands(p).items():
        for it in items:
            if it["kind"] == "plate":
                save_rgb(tmp_path / TR.plate_rel(it["shot"], f), np.broadcast_to(np.float32(B), (H, W, 3)))
            elif it["kind"] == "matte":
                a = np.zeros((2 * H, 2 * W), np.uint8)
                if figure is not None:
                    cv2.circle(a, (int(2 * figure[0]), int(2 * figure[1])), 16, 255, -1)
                path = tmp_path / TR.matte_rel(it["shot"], f)
                path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(path), a)
                save_rgb(tmp_path / TR.back_rel(it["shot"], f), np.broadcast_to(np.float32(A), (H, W, 3)))
    return p["transitions"][0], CF.CutFx(p, tmp_path, (W, H), "t")


def is_(img, colour, tol=0.02):
    return np.abs(img - np.asarray(colour, np.float32)).max(axis=2) < tol


def test_a_slice_opens_a_gap_onto_the_incoming_shot_and_the_halves_leave_by_the_cut(tmp_path):
    t, fx = fx_for(tmp_path, {"at": 2.0, "kind": "slice", "angle": 0})
    main = ramp(A)
    shown = []
    for f in range(t["first"], t["last"] + 1):
        out = fx.frame(f, main)
        incoming, outgoing = is_(out, B).mean(), np.isclose(out[..., 0], A[0], atol=0.01).mean()
        assert incoming + outgoing > 0.95 and outgoing > 0.0                  # both halves on screen, a soft seam between
        shown.append(incoming)
    assert all(b > a for a, b in zip(shown, shown[1:])) and shown[0] < 0.1 and shown[-1] > 0.6
    k = 4                                                                    # the left half has moved left, the right right
    d = TR.slice_progress(t, k) * TR.slice_reach((W, H), ((W - 1) / 2, (H - 1) / 2), (1.0, 0.0))[0]
    out = fx.frame(t["first"] + k, main)
    x = 10
    assert out[20, x, 2] == pytest.approx(main[20, x + int(round(d)), 2], abs=0.02)
    assert out[20, W - 1 - x, 2] == pytest.approx(main[20, W - 1 - x - int(round(d)), 2], abs=0.02)


def test_a_slice_draws_its_edge_on_both_cut_sides(tmp_path):
    t, fx = fx_for(tmp_path, {"at": 2.0, "kind": "slice", "angle": 0, "edge": {"color": "#ffffff", "width": 40}})
    out = fx.frame(t["first"] + 2, ramp(A))                                  # 40 px at 1080: 2 px on this frame
    row = out[27]
    white = np.where(is_(row[None], (1.0, 1.0, 1.0), 0.05)[0])[0]
    gap = np.where(is_(row[None], B)[0])[0]
    assert len(gap) and white.min() < gap.min() and white.max() > gap.max()    # white on both sides of the gap
    assert 2 <= len(white) <= 6


def test_a_tilted_slice_parts_across_its_line(tmp_path):
    t, fx = fx_for(tmp_path, {"at": 2.0, "kind": "slice", "angle": 30})
    out = fx.frame(t["first"] + 3, ramp(A))
    gap = is_(out, B)
    ys, xs = np.nonzero(gap[10:44])                                          # the gap runs along the cut, leaning right at the top
    top, bottom = xs[ys < 6].mean(), xs[ys > 28].mean()
    assert top > bottom + 10


def test_a_slice_through_the_figure_cuts_at_its_centroid(tmp_path):
    t, fx = fx_for(tmp_path, {"at": 2.0, "kind": "slice", "angle": 0, "center": "figure"}, figure=(25, 27))
    fx.check([t["first"] + 1])
    c = (2 * 25 + 0.5) / 2 - 0.5                                             # the disc's centre, from the matte drawn twice as big
    plus, minus = TR.slice_reach((W, H), (c, 27.0), (1.0, 0.0))
    p = TR.slice_progress(t, 1)
    cols = np.nonzero(is_(fx.frame(t["first"] + 1, ramp(A)), B)[27])[0]
    assert len(cols) and c - p * minus - 1 <= cols.min() and cols.max() <= c + p * plus + 1    # the gap opens about the
    assert cols.max() < W / 2 - 10                                                             # disc, not the middle


def test_a_slice_through_a_missing_figure_is_an_error_that_says_so(tmp_path):
    t, fx = fx_for(tmp_path, {"at": 2.0, "kind": "slice", "center": "figure"}, figure=None)
    with pytest.raises(TR.TransitionError, match="shows no figure at frame"):
        fx.check([t["last"]])


def bar_picture():
    """A white bar from the middle to the right edge on dark grey: where it points says how far the picture has turned."""
    img = np.full((H, W, 3), 0.1, np.float32)
    img[H // 2 - 1:H // 2 + 1, W // 2:] = 1.0
    return img


def bar_angle(img, r=20.0):
    """The angle (degrees, clockwise on screen from the right) the bar points to, and how spread its light is round a circle."""
    th = np.radians(np.arange(0, 360, 1.0))
    xs, ys = (W - 1) / 2 + r * np.cos(th), (H - 1) / 2 + r * np.sin(th)
    v = cv2.remap(img[..., 0], xs.astype(np.float32)[None], ys.astype(np.float32)[None], cv2.INTER_LINEAR)[0] - 0.1
    v = np.clip(v, 0, None)
    c, s = (v * np.cos(th)).sum(), (v * np.sin(th)).sum()
    return math.degrees(math.atan2(s, c)) % 360, float((v > 0.05).sum())


def test_a_spin_turns_the_cut_frame_by_its_angle_and_smears_it_most_at_the_cut(tmp_path):
    t, fx = fx_for(tmp_path, {"at": 2.0, "kind": "spin", "dur": 0.4, "after": 0.4, "back": "#000000", "turns": 1.0})
    pic = bar_picture()
    spread = {}
    for f in range(t["first"], t["last"] + 1):
        st = TR.spin_state(t, f)
        got, spread[f] = bar_angle(fx.frame(f, pic))
        want = (st["turn"] + sum(st["smear"]) / 2.0) % 360                 # the smear's middle
        assert abs((got - want + 180) % 360 - 180) < 4.0
    assert max(spread, key=spread.get) in (t["cut"] - 1, t["cut"])
    assert spread[t["cut"]] > 2 * spread[t["first"]]


def test_a_spin_shows_its_back_colour_beyond_the_turned_picture_or_mirrors_the_frame(tmp_path):
    t, fx = fx_for(tmp_path, {"at": 2.0, "kind": "spin", "back": "#1acc33", "blur": 0.0})
    f = min(range(t["first"], t["cut"]), key=lambda g: abs(TR.spin_state(t, g)["turn"] % 180 - 90))   # near a quarter turn
    out = fx.frame(f, np.broadcast_to(np.float32(A), (H, W, 3)).copy())
    assert is_(out[:5, :5], BACK, 0.03).all() and is_(out[24:30, 45:51], A).all()
    _, fx2 = fx_for(tmp_path / "m", {"at": 2.0, "kind": "spin", "blur": 0.0})
    assert is_(fx2.frame(f, np.broadcast_to(np.float32(A), (H, W, 3)).copy()), A).all()      # mirrored: the frame all round


def test_a_spin_is_not_composited_outside_its_window(tmp_path):
    t, fx = fx_for(tmp_path, {"at": 2.0, "kind": "spin"})
    main = bar_picture()
    assert fx.active(t["last"]) and fx.active(t["cut"]) and not fx.active(t["last"] + 1) and not fx.active(t["first"] - 1)
    assert fx.missing(range(t["first"], t["last"] + 1)) == []
    assert abs((bar_angle(fx.frame(t["last"], main))[0] + 180) % 360 - 180) < 6.0    # nearly at rest, upright
    assert fx.frame(t["last"] + 1, main) is main
