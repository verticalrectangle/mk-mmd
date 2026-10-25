"""The split-screen insert (`[[insert]] shape = "split"`; mkmmd.core.transition, mkmmd.cutfx): its spec and errors, its
window in the plan (the slides in and out, the swaps), the layout of its panels along the split, and the composite on tiny
synthetic frames: each half cropped about its focus, the inner half sliding in from its side, trading sides with the host
like a carousel and sliding back out, the divider, the side a tall frame takes, and a video half read through a stand-in
for mkmmd.footage. Fixtures use made-up shot names."""
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
SPLIT = {"from": 1.0, "to": 3.0, "shot": "p", "shape": "split"}


def project(inserts, transitions=()):
    """h 0-4 s (the host), n 4-6 s, and a plate shot p."""
    return {"output": [{"name": "t"}],
            "shot": [{"name": "h", "from": 0.0, "to": 4.0}, {"name": "n", "from": 4.0, "to": 6.0}, {"name": "p", "plate": True}],
            "insert": list(inserts), "transition": list(transitions)}


def plan(**kw):
    return TR.plan(project([{**SPLIT, **kw}]), FPS, F0, MOON)


def split(**kw):
    return plan(**kw)["inserts"][0]


# ---------------------------------------------------------------- the spec
def test_a_split_gets_its_defaults():
    s = TR.normalize_insert(SPLIT, MOON)
    assert (s["shape"], s["shot"], s["clip"], s["side"], s["share"], s["divider"]) == ("split", "p", None, None, 0.5, None)
    assert s["focus"] == {"host": [0.5, 0.5], "inner": [0.5, 0.5]} and (s["in"], s["out"], s["swap"]) == (0.3, 0.3, [])


def test_a_split_takes_a_video_by_name_or_with_how_it_plays():
    s = TR.normalize_insert({"from": 1.0, "to": 3.0, "shape": "split", "clip": "footage/a.mp4"}, MOON)
    assert s["shot"] is None and s["clip"] == {"src": "footage/a.mp4", "in": 0.0, "rate": 1.0, "loop": False, "fit": "cover",
                                               "mirror": False, "back": pytest.approx(PAL.srgb(MOON["base"]), abs=1e-5)}
    spec = {"from": 1.0, "to": 3.0, "shape": "split",
            "clip": {"src": "pexels_1", "in": 2.5, "rate": 0.5, "loop": True, "fit": "contain", "mirror": True, "back": "#000000"}}
    c = TR.normalize_insert(spec, MOON)["clip"]
    assert (c["in"], c["rate"], c["loop"], c["fit"], c["mirror"], c["back"]) == (2.5, 0.5, True, "contain", True, [0.0, 0.0, 0.0])


def test_swaps_are_sorted_seconds_and_aspects_override_side_share_and_focus():
    s = TR.normalize_insert({**SPLIT, "swap": [2.5, 1.5]}, MOON)
    assert s["swap"] == [1.5, 2.5] and TR.normalize_insert({**SPLIT, "swap": 2}, MOON)["swap"] == [2.0]
    s = TR.normalize_insert({**SPLIT, "focus": {"inner": [0.4, 0.3]},
                             "aspect": {"9x16": {"side": "top", "share": 0.4, "focus": {"host": [0.5, 0.2]}}}}, MOON)
    assert TR.split_for(s, "9x16", (1080, 1920)) == {"side": "top", "share": 0.4,
                                                     "focus": {"host": [0.5, 0.2], "inner": [0.4, 0.3]}}
    assert TR.split_for(s, "16x9", (1920, 1080)) == {"side": "right", "share": 0.5,
                                                     "focus": {"host": [0.5, 0.5], "inner": [0.4, 0.3]}}


def test_no_side_is_right_in_a_wide_or_square_frame_and_bottom_in_a_tall_one():
    s = TR.normalize_insert(SPLIT, MOON)
    assert [TR.split_for(s, o, size)["side"] for o, size in (("a", (1920, 1080)), ("b", (1080, 1080)), ("c", (1080, 1920)))] \
        == ["right", "right", "bottom"]
    s = TR.normalize_insert({**SPLIT, "side": "left"}, MOON)                  # a side given holds in every output
    assert TR.split_for(s, "c", (1080, 1920))["side"] == "left"


def bad(**kw):
    spec = {**SPLIT, **kw}
    return {k: v for k, v in spec.items() if v is not None}


@pytest.mark.parametrize("spec, frag", [
    (bad(clip="a.mp4"), "exactly one of `shot`"),
    (bad(shot=None), "exactly one of `shot`"),
    (bad(side="middle"), "side = 'middle'"),
    (bad(share=1.0), "share = 1.0"),
    (bad(share=0), "share = 0"),
    (bad(focus=[0.5, 0.5]), "focus = [0.5, 0.5]"),
    (bad(focus={"host": [2, 0]}), "frame fractions in 0..1"),
    (bad(focus={"middle": [0.5, 0.5]}), "unknown keys ['middle']"),
    (bad(**{"in": -1}), "in = -1.0"),
    (bad(swap=[0.5]), "swap at 0.5 s"),
    (bad(swap="soon"), "swap = 'soon'"),
    (bad(anchor="x"), "unknown keys ['anchor']"),
    (bad(divider={"width": 0}), "width must be positive"),
    (bad(aspect={"9x16": {"size": 0.3}}), "unknown keys ['size']"),
    (bad(aspect={"9x16": {"side": "up"}}), "side = 'up'"),
    (bad(shot=None, clip={"src": "a.mp4", "fit": "stretch"}), "clip.fit = 'stretch'"),
    (bad(shot=None, clip={"src": "a.mp4", "rate": -1}), "clip.rate = -1.0"),
    (bad(shot=None, clip={"src": "a.mp4", "loop": 1}), "true or false"),
    (bad(shot=None, clip={"in": 2.0}), "clip = "),
    (bad(shot=None, clip={"src": "a.mp4", "speed": 2}), "unknown keys ['speed']"),
    (bad(to=0.5), "`to` must come after"),
])
def test_bad_splits_say_what_is_wrong(spec, frag):
    with pytest.raises(TR.TransitionError, match=re.escape(frag)):
        TR.normalize_insert(spec, MOON)


# ---------------------------------------------------------------- the plan
def test_a_split_sits_in_its_host_and_needs_only_its_shots_plates():
    p = plan()
    s = p["inserts"][0]
    assert (s["host"], s["f0"], s["f1"], s["n_in"], s["n_exit"], s["swaps"]) == ("h", F0 + 30, F0 + 90, 9, 9, [])
    d = TR.demands(p)
    assert sorted(d) == list(range(F0 + 30, F0 + 90)) and all(v == [{"kind": "plate", "shot": "p"}] for v in d.values())
    assert TR.needs(p) == {"p": [F0 + 30, F0 + 89]}
    assert TR.summary(p)["inserts"][0] == {"index": 0, "shape": "split", "frames": [F0 + 30, F0 + 89], "host": "h",
                                           "shot": "p", "out": 0.3}


def test_a_split_with_a_video_needs_nothing_rendered():
    p = TR.plan(project([{"from": 1.0, "to": 3.0, "shape": "split", "clip": "a.mp4"}]), FPS, F0, MOON)
    assert TR.demands(p) == {} and TR.summary(p)["inserts"][0]["clip"] == "a.mp4"


def test_slides_of_zero_seconds_are_cuts_and_swaps_land_on_their_frames():
    s = split(**{"in": 0, "out": 0, "swap": [1.5, 2.0]})
    assert (s["n_in"], s["n_exit"], s["swaps"]) == (0, 0, [F0 + 45, F0 + 60])


@pytest.mark.parametrize("kw, frag", [
    ({"swap": [1.2]}, "the swap at 1.2 s (landing on frame 37 after a 9-frame slide) comes too soon after the slide in"),
    ({"swap": [2.0, 2.1]}, "comes too soon after the swap at 2.0 s"),
    ({"swap": [2.9]}, "lands on frame 88, too late: a swap must land by frame 82"),
    ({"in": 0, "out": 0, "swap": [2.0, 2.0]}, "too soon after the swap at 2.0 s"),
    ({"to": 1.4}, "12 frames are too few for a 9-frame slide in and a 9-frame slide out"),
    ({"shot": "h"}, "the host shot itself"),
    ({"shot": "nope"}, "no shot named 'nope'"),
    ({"to": 4.5}, "inside one shot"),
])
def test_splits_that_do_not_fit_are_refused(kw, frag):
    with pytest.raises(TR.TransitionError, match=re.escape(frag)):
        plan(**kw)


def test_a_split_may_not_overlap_a_transition():
    with pytest.raises(TR.TransitionError, match="overlap"):
        TR.plan(project([{**SPLIT, "from": 3.0, "to": 3.9}], [{"at": 4.0, "kind": "slice"}]), FPS, F0, MOON)


# ---------------------------------------------------------------- timing and layout
def test_the_inner_half_slides_in_holds_and_slides_out_moving_on_every_slide_frame():
    s = split()
    us = [TR.split_state(s, f)["u"] for f in range(s["f0"], s["f1"])]
    assert all(0 < u < 1 for u in us[:9]) and all(b > a for a, b in zip(us[:9], us[1:9]))
    assert us[9:-9] == [1.0] * (len(us) - 18)
    assert all(0 < u < 1 for u in us[-9:]) and all(b < a for a, b in zip(us[-9:], us[-8:]))
    assert TR.split_state(s, s["f0"] - 1)["phase"] == "none" and TR.split_state(s, s["f1"])["phase"] == "none"


def test_a_swap_slides_as_long_as_the_slide_in_and_lands_on_its_frame():
    s = split(swap=[2.0])
    fs = s["swaps"][0]
    qs = [TR.split_state(s, f)["q"] for f in range(fs - 10, fs + 2)]
    assert qs[0] == 0.0 and all(0 < q < 1 for q in qs[1:10]) and all(b > a for a, b in zip(qs[1:10], qs[2:10]))
    assert qs[10:] == [0.0, 0.0]
    assert [TR.split_state(s, f)["swaps"] for f in (fs - 1, fs)] == [0, 1]
    s = split(swap=[2.0], **{"in": 0})                                        # a cut
    assert all(TR.split_state(s, f)["q"] == 0.0 for f in range(s["f0"], s["f1"]))


def panels(u=1.0, q=0.0, swaps=0, share=0.5, side="right", L=96):
    return TR.split_panels({"u": u, "q": q, "swaps": swaps}, share, L, side)


def test_panels_at_rest_put_the_inner_half_on_its_side():
    assert panels() == [("host", 0.0, 48.0), ("inner", 48.0, 48.0)]
    assert panels(side="left") == [("inner", 0.0, 48.0), ("host", 48.0, 48.0)]
    assert panels(share=0.25) == [("host", 0.0, 72.0), ("inner", 72.0, 24.0)]
    assert panels(share=0.25, side="top") == [("inner", 0.0, 24.0), ("host", 24.0, 72.0)]
    assert panels(swaps=1, share=0.25) == [("inner", 0.0, 24.0), ("host", 24.0, 72.0)]


@pytest.mark.parametrize("side", TR.SIDES)
@pytest.mark.parametrize("u, q, swaps", [(0.1, 0, 0), (0.6, 0, 0), (1, 0.3, 0), (1, 0.8, 1), (0.4, 0, 1), (1, 0, 2)])
def test_panels_tile_the_frame_in_every_state(side, u, q, swaps):
    ps = panels(u, q, swaps, 0.4, side)
    assert all(abs(s1 - (s0 + n0)) < 1e-9 for (_, s0, n0), (_, s1, _) in zip(ps, ps[1:]))
    assert ps[0][1] <= 0.0 and ps[-1][1] + ps[-1][2] >= 96.0 - 1e-9
    assert all(a != b for (a, _, _), (b, _, _) in zip(ps, ps[1:]))


def test_the_inner_half_enters_from_its_side_and_the_carousel_runs_on_the_same_way():
    starts = [dict((r, s) for r, s, _ in panels(u=u))["inner"] for u in (0.2, 0.5, 0.9)]
    assert starts == sorted(starts, reverse=True)                            # from the right, moving left
    left = [panels(u=u, side="left")[0] for u in (0.2, 0.9)]
    assert left[0][1] + left[0][2] < left[1][1] + left[1][2]                  # from the left, moving right
    mid = panels(q=0.5)                                                      # [host | inner] -> [inner | host]: leftward
    assert [r for r, _, _ in mid] == ["host", "inner", "host"] and mid[1][1] == pytest.approx(24.0)
    back = panels(q=0.5, swaps=1)                                            # and on: [inner | host] -> [host | inner]
    assert [r for r, _, _ in back] == ["inner", "host", "inner"] and back[1][1] == pytest.approx(24.0)


def test_dividers_run_on_the_seams_inside_the_frame():
    assert TR.split_dividers(panels(), 96) == [48.0]
    assert TR.split_dividers(panels(q=0.5), 96) == [pytest.approx(24.0), pytest.approx(72.0)]
    assert TR.split_dividers(panels(u=0.0), 96) == []


def test_a_picture_is_cropped_about_its_focus_as_far_as_it_covers_the_panel():
    assert TR.panel_offset(48, 48, 0.5, 96) == 24.0                          # its middle mid-panel
    assert TR.panel_offset(48, 48, 0.0, 96) == 48.0                          # its left edge at the panel's
    assert TR.panel_offset(48, 48, 1.0, 96) == 0.0                           # its right edge at the panel's
    assert TR.panel_offset(0, 96, 0.2, 96) == 0.0                            # the whole frame: as it is


# ---------------------------------------------------------------- compositing
def coded(g, w=W, h=H):
    """A picture whose red channel counts the column, blue the row, green says whose it is."""
    img = np.empty((h, w, 3), np.float32)
    img[..., 0] = (np.arange(w, dtype=np.float32) / (w - 1))[None, :]
    img[..., 1] = g
    img[..., 2] = (np.arange(h, dtype=np.float32) / (h - 1))[:, None]
    return img


HOST, INNER = 0.2, 0.8


def save_rgb(path, img):
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), (np.clip(img, 0, 1)[..., ::-1] * 255 + 0.5).astype(np.uint8))


def fx_for(tmp_path, size=(W, H), footage_of=None, **kw):
    p = TR.plan(project([{**SPLIT, **kw}]), FPS, F0, MOON)
    for f, items in TR.demands(p).items():
        for it in items:
            save_rgb(tmp_path / TR.plate_rel(it["shot"], f), coded(INNER, *size))
    return p["inserts"][0], CF.CutFx(p, tmp_path, size, "t", footage_of=footage_of)


def test_at_rest_each_half_shows_its_own_picture_cropped_about_its_middle(tmp_path):
    s, fx = fx_for(tmp_path, divider={"color": "#ffffff", "width": 40})
    out = fx.frame(s["f0"] + 12, coded(HOST))
    assert np.allclose(out[:, 5:45, 1], HOST, atol=0.01) and np.allclose(out[:, 51:, 1], INNER, atol=0.01)
    assert out[30, 10, 0] == pytest.approx((10 + 24) / (W - 1), abs=0.01)  # host columns 24..71 on the left
    assert out[30, 60, 0] == pytest.approx((60 - 48 + 24) / (W - 1), abs=0.01)   # the inner shot's own middle on the right
    assert np.allclose(out[:, 47:49], 1.0, atol=0.01)                        # a 2 px divider on the seam (40 px at 1080)
    assert out[17, 10, 2] == pytest.approx(17 / (H - 1), abs=0.01)          # not squeezed: the rows are the frame's


def test_a_focus_moves_the_crop_and_stops_at_the_pictures_edge(tmp_path):
    s, fx = fx_for(tmp_path, focus={"inner": [0.0, 0.5], "host": [0.3, 0.5]})
    out = fx.frame(s["f0"] + 12, coded(HOST))
    assert out[30, 48, 0] == pytest.approx(0.0, abs=0.01)                   # the inner picture's left edge at the seam
    assert out[30, 24, 0] == pytest.approx(round(0.3 * 96) / (W - 1), abs=0.015)   # the host's focus mid-panel


def test_the_inner_half_slides_in_from_the_right_trades_sides_and_slides_out_left(tmp_path):
    s, fx = fx_for(tmp_path, swap=[2.0])
    host = coded(HOST)
    first = fx.frame(s["f0"], host)
    assert first[30, W - 1, 1] == pytest.approx(INNER, abs=0.01) and first[30, W // 2, 1] == pytest.approx(HOST, abs=0.01)
    swapped = fx.frame(s["swaps"][0] + 2, host)
    assert swapped[30, 10, 1] == pytest.approx(INNER, abs=0.01) and swapped[30, 80, 1] == pytest.approx(HOST, abs=0.01)
    last = fx.frame(s["f1"] - 1, host)
    assert last[30, 0, 1] == pytest.approx(INNER, abs=0.01) and last[30, 10, 1] == pytest.approx(HOST, abs=0.01)
    assert not fx.active(s["f1"]) and not fx.active(s["f0"] - 1)


def test_a_tall_frame_splits_top_and_bottom(tmp_path):
    s, fx = fx_for(tmp_path, size=(H, W))                                    # 54 x 96: taller than wide
    out = fx.frame(s["f0"] + 12, coded(HOST, H, W))
    assert np.allclose(out[5:45, :, 1], HOST, atol=0.01) and np.allclose(out[51:, :, 1], INNER, atol=0.01)
    assert out[10, 20, 2] == pytest.approx((10 + 24) / (W - 1), abs=0.01)  # the host cropped about its middle, vertically
    assert out[10, 20, 0] == pytest.approx(20 / (H - 1), abs=0.01)          # and whole across


class FakeClip:
    """Stands in for mkmmd.footage.Clip: a flat picture of the asked size, the calls written down."""

    def __init__(self, colour=(0.9, 0.5, 0.1)):
        self.colour, self.calls = colour, []

    def index(self, t, start=0.0, rate=1.0, loop=False):
        self.calls.append(("index", t, start, rate, loop))
        return 0

    def picture(self, t, size, **kw):
        self.calls.append(("picture", t, tuple(size), kw))
        return np.broadcast_to(np.asarray(self.colour, np.float32), (size[1], size[0], 3))


def video_fx(tmp_path, footage_of, **clip):
    p = TR.plan(project([{"from": 1.0, "to": 3.0, "shape": "split", "share": 0.4, "clip": {"src": "v.mp4", **clip}}]),
                FPS, F0, MOON)
    return p["inserts"][0], CF.CutFx(p, tmp_path, (W, H), "t", footage_of=footage_of)


def test_a_video_half_plays_from_its_in_point_fitted_to_the_half(tmp_path):
    clip = FakeClip()
    s, fx = video_fx(tmp_path, lambda src: clip, **{"in": 2.0, "rate": 0.5, "fit": "contain"})
    fx.check(range(s["f0"], s["f1"]))
    assert clip.calls[0] == ("index", 0.0, 2.0, 0.5, False)                  # the span must start inside the video
    out = fx.frame(s["f0"] + 15, coded(HOST))
    _, t, size, kw = clip.calls[-1]
    assert t == pytest.approx(15 / FPS) and size == (39, H)                  # 0.4 of 96 wide, the frame's height
    assert kw["start"] == 2.0 and kw["rate"] == 0.5 and kw["fit"] == "contain" and kw["focus"] == (0.5, 0.5)
    assert np.allclose(out[:, 62:, :], clip.colour, atol=0.01) and np.allclose(out[:, :50, 1], HOST, atol=0.01)


def test_a_video_that_cannot_be_read_is_an_error_naming_the_insert(tmp_path):
    def missing(src):
        raise FileNotFoundError(f"clip {src!r}: no such file")
    s, fx = video_fx(tmp_path, missing)
    with pytest.raises(TR.TransitionError, match=r"insert 0 \(split 1.0-3.0 s\): clip 'v.mp4': clip 'v.mp4': no such file"):
        fx.check([s["f0"]])
    _, bare = video_fx(tmp_path, None)
    with pytest.raises(TR.TransitionError, match="no video source"):
        bare.frame(s["f0"] + 3, coded(HOST))
