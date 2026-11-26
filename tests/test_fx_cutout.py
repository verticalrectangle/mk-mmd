"""Cutouts ([[fx]] kind cutout: mkmmd.core.videosticker and mkmmd.videosticker): a clip cut out by its own matte (the
`<stem>.matte<suffix>` video beside it, mkmmd.footage), played stepped and synced to a beat, its anchor at `at` (a frame
point or a scene point), turned, with its edge, per output."""
import re

import cv2
import numpy as np
import pytest

from mkmmd import footage as FT
from mkmmd import framefx as FF
from mkmmd.core import framefx as FX
from mkmmd.core import palette as PAL
from mkmmd.core import transition as TR
from mkmmd.core import videosticker as VS

FPS, F0 = 60, 1
DAWN = PAL.get("rose-pine-dawn")
W, H = 160, 90
GREY = (0.4, 0.4, 0.4)


def entry(**spec):
    return {"kind": "cutout", "from": 1.0, "to": 2.0, "clip": "hand.mp4", **spec}


def plan(*fxs):
    return TR.plan({"project": {"duration": 4.0}, "output": [{"name": "16x9", "size": [W, H]},
                                                             {"name": "9x16", "size": [H, W]}],
                    "shot": [{"name": "a", "from": 0.0, "to": 4.0}], "fx": list(fxs)}, FPS, F0, DAWN)


def frame(t):
    return FX.frame_of(t, FPS, F0)


def grey(size=(W, H)):
    return np.full((size[1], size[0], 3), GREY, np.float32)


def video(path, pictures, fps=24.0):
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (pictures[0].shape[1], pictures[0].shape[0]))
    assert vw.isOpened()
    for p in pictures:
        vw.write(p)
    vw.release()
    return path


def hand(root, matte=None, n=24, size=(48, 32)):
    """hand.mp4 whose frame i is flat grey 16 + 9 i, and its matte hand.matte.mp4 (default: its left half white)."""
    video(root / "hand.mp4", [np.full((size[1], size[0], 3), 16 + 9 * i, np.uint8) for i in range(n)])
    if matte is None:
        matte = np.zeros((size[1], size[0], 3), np.uint8)
        matte[:, :size[0] // 2] = 255
    video(root / "hand.matte.mp4", [matte] * n)


def which(rgb):
    return int(round((float(np.mean(rgb)) * 255.0 - 16.0) / 9.0))


def fxs(pl, root, output="16x9", point_of=None):
    size = (W, H) if output == "16x9" else (H, W)
    return FF.FrameFx(pl["fx"], size, fps=FPS, frame0=F0, frame_of=lambda g: None, point_of=point_of, output=output,
                      footage_of=FT.source(root))


# ================================================================================================= keys
@pytest.mark.parametrize("spec, frag", [
    ({"kind": "cutout", "from": 1, "to": 2}, "`clip` is required"),
    (entry(sync=[0.5, 1.5], **{"in": 1.0}), "give `in`"),
    (entry(sync=[-0.5, 1.5]), "[video second (0 or more), clip second]"),
    (entry(sync=0.5), "expected [x, y]"),
    (entry(fps=500), "fps = 500: expected 0..120"),
    (entry(at={"pt": "x"}), "expected {point = \"<expression>\"} or [x, y]"),
    (entry(grade={"warmth": 1}), "expected {gain, lift, warm, saturation, toward}"),
    (entry(at={"peel": 2.0, "grip": 1}), "expected {peel = <second>}, {tear = <second>, grip = 0 or 1} or"),
    (entry(at={"tear": 2.0, "grip": 2}), "grip = 2: 0 or 1"),
    (entry(extend="yes"), "extend is true or false"),
    (entry(grade={"lift": 2}), "lift = 2: expected 0..1"),
    (entry(shape="heart"), "unknown keys ['shape']"),
    (entry(focus=[0.2, 0.2]), "unknown keys ['focus']"),
    (entry(aspect={"9x16": {"anchor": [1]}}), "aspect.9x16 (cutout) anchor"),
    (entry(anchor=[[1.0, 0.2, 0.5]]), "a track is [[video second, x, y], ...], two rows or more"),
    (entry(anchor=[[2.0, 0.2, 0.5], [1.0, 0.4, 0.5]]), "a track's video seconds rise"),
])
def test_bad_entries_say_what_is_wrong(spec, frag):
    with pytest.raises(FX.FxError, match=re.escape(frag)):
        FX.normalize(spec, DAWN)


# ================================================================================================= clock
def test_it_steps_at_its_fps_from_its_sync_a_step_starting_on_the_beat_s_frame():
    fx = FX.normalize(entry(fps=12, sync=[2.0, 1.585]), DAWN)       # the beat at 1.585 s falls on frame 95.1 -> 95
    beat = (frame(1.585) - F0) / FPS
    assert VS.clock(fx, beat, FPS) == (pytest.approx(beat), pytest.approx(2.0))          # the moment, on the beat
    for k in range(1, 5):                                           # held for the 5 frames of a twelfth of a second
        assert VS.clock(fx, beat + k / FPS, FPS)[1] == pytest.approx(2.0)
    assert VS.clock(fx, beat + 5 / FPS, FPS)[1] == pytest.approx(2.0 + 1 / 12)
    assert VS.clock(fx, beat - 1 / FPS, FPS)[1] == pytest.approx(2.0 - 1 / 12)          # the pose before it
    assert fx["in"] == pytest.approx(2.0 - 0.585)                   # the video's second on the span's first frame
    smooth = FX.normalize(entry(fps=0, rate=-2.0, **{"in": 3.0}), DAWN)                  # every frame, played backwards
    assert VS.clock(smooth, 1.1, FPS) == (pytest.approx(1.1), pytest.approx(2.8))


def test_a_real_clip_shows_the_frame_of_its_held_pose(tmp_path):
    hand(tmp_path)
    pl = plan(entry(sync=[0.5, 1.5], fps=12))                      # video 0.5 s (frame 12 at 24 fps) on clip 1.5 s
    ffx = fxs(pl, tmp_path)
    for t, want in ((1.5, 12), (1.5 + 4 / 60, 12), (1.5 + 5 / 60, 14), (1.5 - 1 / 60, 10)):
        out = ffx.frame(frame(t), grey())
        assert which(out[40:50, 30:50]) == want
        assert out[45, 120].tolist() == pytest.approx(list(GREY))  # the matte's black half: the picture


# ================================================================================================= shape and place
def test_its_shape_is_its_matte_with_an_edge_round_it_and_an_empty_matte_leaves_the_picture(tmp_path):
    hand(tmp_path)
    pl = plan(entry(edge={"color": "#ffffff", "width": 100}))      # 100 px at 1080: 8.3 px here
    out = fxs(pl, tmp_path).frame(frame(1.5), grey())
    assert which(out[40:50, 30:50]) == 12                           # the box: 135 x 90 about the middle, its left half
    assert out[45, 84].tolist() == pytest.approx([1.0, 1.0, 1.0], abs=0.02)   # the rim past the matte's edge (79.5)
    assert out[45, 100].tolist() == pytest.approx(list(GREY))
    empty = tmp_path / "e"
    empty.mkdir()
    hand(empty, matte=np.zeros((32, 48, 3), np.uint8))
    assert fxs(plan(entry(edge={"color": "#ffffff", "width": 100})), empty).frame(frame(1.5), grey()).tolist() == \
        grey().tolist()


def test_its_anchor_lands_on_at_and_it_turns_about_it_mirrored_with_the_clip():
    fx = FX.normalize(entry(anchor=[0.25, 0.75], size=0.5, turn=90), DAWN)
    M, box, scale = VS.place_cutout(fx, 1.5, (W, H), 1.5, (50.0, 40.0), 0.0)
    assert box == (pytest.approx(67.5), pytest.approx(45.0)) and scale == 1.0
    a = np.array([(0.25 - 0.5) * 67.5, (0.75 - 0.5) * 45.0])
    M = np.asarray(M)
    assert M @ [*a, 1.0] == pytest.approx([50.0, 40.0])
    assert M @ [a[0] + 10.0, a[1], 1.0] == pytest.approx([50.0, 30.0])      # right of it turns up: counter-clockwise
    fx = FX.normalize(entry(anchor=[0.25, 0.75], size=0.5, mirror=True), DAWN)
    M = np.asarray(VS.place_cutout(fx, 1.5, (W, H), 1.5, (50.0, 40.0), 0.0)[0])
    assert M @ [(0.75 - 0.5) * 67.5, a[1], 1.0] == pytest.approx([50.0, 40.0])


def test_a_tracked_anchor_keeps_the_point_that_moves_in_the_clip_on_at():
    fx = FX.normalize(entry(anchor=[[1.0, 0.2, 0.5], [2.0, 0.4, 0.9]], size=0.5), DAWN)   # a fingertip, 1 s to 2 s
    for video, (x, y) in ((1.5, (0.3, 0.7)), (0.0, (0.2, 0.5)), (5.0, (0.4, 0.9))):       # between rows; held past them
        M = np.asarray(VS.place_cutout(fx, 1.5, (W, H), 1.5, (50.0, 40.0), video)[0])
        assert M @ [(x - 0.5) * 67.5, (y - 0.5) * 45.0, 1.0] == pytest.approx([50.0, 40.0])


def test_at_may_be_a_scene_point_and_each_output_places_it_its_own_way(tmp_path):
    hand(tmp_path, matte=np.full((32, 48, 3), 255, np.uint8))
    pl = plan(entry(at={"point": "head"}, anchor=[0.0, 0.0], size=0.2, extend=False,
                    aspect={"9x16": {"at": [0.5, 0.5], "anchor": [0.5, 0.5]}}))
    fx = pl["fx"][0]
    assert FX.points(fx) == [("x0", "head")]
    out = fxs(pl, tmp_path, point_of=lambda key, g: ((30.0 + (g - frame(1.0)), 20.0), 1.0)).frame(frame(1.1), grey())
    assert out[17, 33].tolist() == pytest.approx(list(GREY))       # the point moved 6 px right by now: up-left of it
    assert which(out[24:30, 40:50]) == 2                            # down-right of it: the clip
    tall = fxs(pl, tmp_path, "9x16", point_of=None).frame(frame(1.1), grey((H, W)))
    assert which(tall[75:85, 40:50]) == 2 and tall[5, 5].tolist() == pytest.approx(list(GREY))


def test_where_its_matte_meets_the_clip_s_border_the_arm_goes_on_to_the_picture_s_edge_with_no_rim(tmp_path):
    matte = np.zeros((32, 48, 3), np.uint8)
    matte[16:] = 255                                                # the bottom half: it meets left, right and bottom
    hand(tmp_path, matte=matte)
    kw = {"size": 0.3, "edge": {"color": "#ffffff", "width": 100}}  # box 40.5 x 27 about the middle, bottom at 58
    out = fxs(plan(entry(**kw)), tmp_path).frame(frame(1.5), grey())
    assert which(out[75:85, 70:90]) == 12                           # under the box: the arm, its border line repeated
    assert which(out[60:63, 70:90]) == 12                           # no rim where it crosses the box's own edge
    assert out[40, 80].tolist() == pytest.approx([1.0, 1.0, 1.0], abs=0.02)   # the rim round its top, as ever
    off = fxs(plan(entry(extend=False, **kw)), tmp_path).frame(frame(1.5), grey())
    assert off[80, 80].tolist() == pytest.approx(list(GREY))
    soft = tmp_path / "soft"                                        # a matte softened by its maker fades in its last lines
    soft.mkdir()
    matte[-2:] = 60
    hand(soft, matte=matte)
    assert which(fxs(plan(entry(**kw)), soft).frame(frame(1.5), grey())[75:85, 70:90]) == 12   # the arm goes on all the same


def test_it_rides_a_peel_s_corner_and_a_tear_s_grips(tmp_path, monkeypatch):
    fx = FX.normalize(entry(at={"tear": 2.0, "grip": 1}), DAWN)
    fx["index"] = 0
    with pytest.raises(FX.FxError, match=r"rides the tear at 2 s; the tear transitions are at \[3.0\]"):
        VS.ride([fx], [{"kind": "tear", "at": 3.0}, {"kind": "peel", "at": 2.0}], FPS)
    tr = {"kind": "tear", "at": 2.004}
    VS.ride([fx], [{"kind": "peel", "at": 2.0}, tr], FPS)
    assert fx["at"]["tr"] is tr
    hand(tmp_path, matte=np.full((32, 48, 3), 255, np.uint8))
    from mkmmd.core import peel as PL
    monkeypatch.setattr(PL, "corner", lambda tr, g, size: {"p": [0.25, 0.5], "z": 0.0, "scale": 1.0})
    pl = {"fx": FX.plan({"fx": [entry(at={"peel": 1.0}, size=0.2, extend=False)]}, FPS, F0, DAWN)}
    VS.ride(pl["fx"], [{"kind": "peel", "at": 1.0}], FPS)
    out = fxs(pl, tmp_path).frame(frame(1.5), grey())
    assert which(out[42:48, 36:44]) == 12 and out[45, 80].tolist() == pytest.approx(list(GREY))


def test_it_is_a_sticker_with_the_sheet_s_rim_and_shadow_unless_it_says_otherwise():
    ST, fx = FX.ST, FX.normalize(entry(), DAWN)
    white = FX._colour(ST.EDGE["color"], DAWN, "")
    assert fx["edge"] == {"color": white, "width": ST.EDGE["width"]}
    assert fx["shadow"] == {**ST.SHADOW, "color": FX._colour(ST.SHADOW["color"], DAWN, "")}
    assert FX.normalize(entry(edge={"width": 12}), DAWN)["edge"] == {"color": white, "width": 12.0}   # over the sheet's
    bare = FX.normalize(entry(edge=False, shadow=False, grade=False), DAWN)
    assert (bare["edge"], bare["shadow"], bare["grade"]) == (None, None, None)


def test_its_grade_brings_skin_up_and_leaves_near_white_paper_as_shot():
    from mkmmd import videosticker as VSF
    g = FX.normalize(entry(grade={"gain": 1.2, "lift": 0.0, "warm": 0.0, "saturation": 1.0}), DAWN)["grade"]
    skin, card = [0.6, 0.45, 0.4], [0.95, 0.94, 0.9]
    out = VSF.grade(np.array([[skin, card]], np.float32), g)
    assert out[0, 0].tolist() == pytest.approx([0.72, 0.54, 0.48], abs=1e-5)
    assert out[0, 1].tolist() == pytest.approx(card, abs=1e-6)


def test_post_checks_its_matte_before_encoding(tmp_path):
    video(tmp_path / "hand.mp4", [np.full((32, 48, 3), 90, np.uint8)] * 4)
    with pytest.raises(FT.FootageError, match=r"fx 0 \(cutout\): hand.mp4: no matte beside it"):
        FF.check_footage(plan(entry())["fx"], FT.source(tmp_path))
    video(tmp_path / "hand.matte.mp4", [np.full((32, 48, 3), 255, np.uint8)] * 3)
    with pytest.raises(FT.FootageError, match="a matte is its clip's size and length"):
        FF.check_footage(plan(entry())["fx"], FT.source(tmp_path))
