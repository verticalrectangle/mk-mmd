"""Video stickers ([[fx]] kind clip: mkmmd.core.videosticker and mkmmd.videosticker): their keys, the shapes' outlines and
masks, where a sticker stands and how its keys move it, the clip fitted into its box, how it is laid on the picture
(over, screen, add) with its edge and shadow, the per-output placement, and the checks and credits its clip gets."""
import re

import cv2
import numpy as np
import pytest

from mkmmd import assets as A
from mkmmd import footage as FT
from mkmmd import framefx as FF
from mkmmd import gpufx as GFX
from mkmmd import postgpu as PG
from mkmmd import videosticker as VSF
from mkmmd.cli.assets import footage_clips, project_credits
from mkmmd.core import breathe as BR
from mkmmd.core import framefx as FX
from mkmmd.core import palette as PAL
from mkmmd.core import transition as TR
from mkmmd.core import videosticker as VS

FPS, F0 = 30, 1
MOON, DAWN = PAL.get("rose-pine-moon"), PAL.get("rose-pine-dawn")
W, H = 160, 90
GREY, RED, WHITE, BLACK = (0.4, 0.4, 0.4), (1.0, 0.0, 0.0), (1.0, 1.0, 1.0), (0.0, 0.0, 0.0)


def entry(**spec):
    return {"kind": "clip", "from": 1.0, "to": 2.0, "clip": "night.mp4", **spec}


def plan(*fxs):
    return TR.plan({"project": {"duration": 4.0}, "output": [{"name": "16x9", "size": [W, H]},
                                                             {"name": "9x16", "size": [H, W]}],
                    "shot": [{"name": "a", "from": 0.0, "to": 4.0}], "fx": list(fxs)}, FPS, F0, DAWN)


def frame(t):
    return FX.frame_of(t, FPS, F0)


class Clip:
    """A clip stand-in: every picture flat `colour`; records what each picture was asked for."""

    def __init__(self, colour=RED):
        self.colour, self.asked = colour, []

    def index(self, t, start=0.0, rate=1.0, loop=False):
        return 0

    def picture(self, t, size, start=0.0, rate=1.0, loop=False, fit="cover", focus=(0.5, 0.5), mirror=False,
                back=(0.0, 0.0, 0.0), crop=None):
        self.asked.append({"t": round(t, 4), "size": tuple(size), "start": start, "rate": rate, "loop": loop,
                           "fit": fit, "focus": tuple(focus), "mirror": mirror})
        return np.full((size[1], size[0], 3), self.colour, np.float32)


def run(pl, clip, output="16x9"):
    size = (W, H) if output == "16x9" else (H, W)
    return FF.FrameFx(pl["fx"], size, fps=FPS, frame0=F0, frame_of=lambda g: None, point_of=None, output=output,
                      footage_of=lambda c: clip)


def grey(size=(W, H)):
    return np.full((size[1], size[0], 3), GREY, np.float32)


def grey_clip(path, n=25, fps=24.0, size=(48, 32)):
    """A clip whose frame i is flat grey 16 + 9 i (the codec takes a few levels off)."""
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    assert vw.isOpened()
    for i in range(n):
        vw.write(np.full((size[1], size[0], 3), 16 + 9 * i, np.uint8))
    vw.release()
    return path


def which(rgb):
    """The frame number of a picture of grey_clip (its mean grey)."""
    return int(round((float(np.mean(rgb)) * 255.0 - 16.0) / 9.0))


def box_of(m, at=0.5):
    """(x0, y0, x1, y1): the pixels where `m` is over `at`, inclusive."""
    ys, xs = np.nonzero(m > at)
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


# ================================================================================================= keys
@pytest.mark.parametrize("spec, frag", [
    ({"kind": "clip", "from": 1, "to": 2}, "`clip` is required"),
    (entry(shape="hexagon"), "shape 'hexagon' is none of rect, circle, heart, star"),
    (entry(shape="heart", radius=0.2), "radius rounds a rect's corners; a heart has none"),
    (entry(radius=0.7), "radius = 0.7: expected 0..0.5"),
    (entry(size=0), "size = 0: expected 0.01..10"),
    (entry(ratio=-1), "ratio = -1: expected 0.05..20"),
    (entry(at=[0.5]), "expected [x, y]"),
    (entry(blend="multiply"), "blend 'multiply' is none of"),
    (entry(blend="negative"), "a negative sticker shows the picture under it inverted: give it no clip"),
    (entry(blend="screen", shadow={"alpha": 0.5}), "a screen sticker's face is light and casts no shadow"),
    (entry(blend="add", shadow={}, edge={"color": "#ffffff", "width": 0}), "a add sticker's face is light"),
    (entry(fit="contain"), "unknown keys ['fit']"),
    (entry(objects="Rin_*"), "unknown keys ['objects']"),
    (entry(edge={"width": 14}), "expected {color, width}"),
    (entry(shadow={"colour": "base"}), "expected {color, offset, blur, alpha}"),
    (entry(focus=[1.5, 0.5]), "[x, y] in 0..1 of the video"),
    (entry(rate=20), "rate = 20: expected 0..16"),
    (entry(loop="yes"), "loop is true or false"),
    (entry(crop=[0.5, 0.2, 0.4, 0.6]), "x1 must be right of x0"),
    (entry(keys=[{"turn": 30}]), "has no `t`"),
    (entry(keys=[{"t": 1.2, "scale": 0}]), "expected a positive number"),
    (entry(aspect={"9x16": {"from": 1.5}}), "may not set ['from']"),
    (entry(aspect={"9x16": {"shape": "disc"}}), "aspect.9x16 (clip): shape 'disc' is none of"),
])
def test_bad_entries_say_what_is_wrong(spec, frag):
    with pytest.raises(FX.FxError, match=re.escape(frag)):
        FX.normalize(spec, DAWN)


def test_its_defaults_and_an_entry_the_same_in_any_palette_but_its_colours():
    assert "clip" in FX.KINDS and "clip" in FX.CLIPS and {"rect", "feather", "strength", "aspect"} <= FX.KEYS["clip"]
    fx = FX.normalize(entry(), DAWN)
    assert {k: fx[k] for k in ("shape", "radius", "at", "size", "ratio", "keys", "edge", "shadow", "blend")} == {
        "shape": "rect", "radius": 0.12, "at": [0.5, 0.5], "size": 0.4, "ratio": 1.0, "keys": [], "edge": None,
        "shadow": None, "blend": "over"}
    assert (fx["in"], fx["rate"], fx["loop"], fx["focus"], fx["mirror"]) == (0.0, 1.0, False, [0.5, 0.5], False)
    ratio = {s: FX.normalize(entry(shape=s), DAWN)["ratio"] for s in VS.OUTLINED}   # each shape's own proportions
    assert ratio["rect"] == ratio["circle"] == 1.0
    assert ratio["heart"] == pytest.approx(1.106, abs=0.005) and ratio["star"] == pytest.approx(1.044, abs=0.005)
    assert FX.normalize(entry(shape="heart", ratio=1.5), DAWN)["ratio"] == 1.5
    spec = entry(shape="star", blend="screen", edge={"color": "text", "width": 14}, shadow={"color": "base"},
                 keys=[{"t": 1.2, "turn": 10}], aspect={"9x16": {"at": [0.5, 0.8], "edge": {"color": "love"}}})
    a, b = FX.normalize(spec, MOON), FX.normalize(spec, DAWN)
    assert a["edge"]["color"] != b["edge"]["color"] and all(isinstance(c, float) for c in a["shadow"]["color"])
    paths = [p for p, _, _ in BR.Pair(a, b).diffs]           # a palette that breathes mixes these numbers, all colours
    assert paths and all(p[-2] == "color" for p in paths)
    assert {p[0] for p in paths} == {"edge", "shadow", "aspect"}


# ================================================================================================= shapes
@pytest.mark.parametrize("shape", VS.OUTLINED)
def test_each_outline_fills_its_box_and_has_more_points_the_bigger_it_shows(shape):
    p = VS.outline(shape, 200.0, 120.0, 0.2)
    assert p.min(0) == pytest.approx([-100.0, -60.0]) and p.max(0) == pytest.approx([100.0, 60.0])
    assert len(VS.outline(shape, 400.0, 240.0, 0.2)) > len(p)
    if shape in ("circle", "heart"):                           # curved all round: a point every step or so
        gaps = np.hypot(*np.diff(np.vstack([p, p[:1]]), axis=0).T)
        assert gaps.max() <= 1.6 * VS.STEP
    with pytest.raises(ValueError):
        VS.outline("disc", 10.0, 10.0)


def mask(shape, radius=0.0):
    """A 100 x 100 px box of `shape` filled in the middle of a 140 x 140 picture: it spans pixels 20 .. 119."""
    return VSF.cover(VS.outline(shape, 100.0, 100.0, radius) + 69.5, (140, 140))


def test_the_masks_are_each_shape_antialiased_by_its_share_of_each_pixel():
    square, rounded, circle, heart, star = (mask("rect"), mask("rect", 0.25), mask("circle"), mask("heart"),
                                            mask("star"))
    assert square.sum() == pytest.approx(10000.0, rel=0.001) and square[69, 20] > 0.99 and square[69, 19] < 0.01
    off = VSF.cover(VS.outline("rect", 100.0, 100.0) + 69.75, (140, 140))    # a quarter of a pixel to the right
    assert off[69, 20] == pytest.approx(0.75, abs=0.01) and off[69, 120] == pytest.approx(0.25, abs=0.01)
    assert off[20, 69] == pytest.approx(0.75, abs=0.01) and off[120, 69] == pytest.approx(0.25, abs=0.01)
    assert rounded.sum() == pytest.approx(10000.0 - (4.0 - np.pi) * 25.0 ** 2, rel=0.002)
    assert rounded[20, 20] < 0.01 and rounded[20, 69] > 0.99 and rounded[69, 20] > 0.99      # the corners cut round
    assert circle.sum() == pytest.approx(np.pi * 50.0 ** 2, rel=0.002)
    assert circle[69, 21] > 0.99 and circle[21, 69] > 0.99 and circle[24, 24] < 0.01
    assert heart[30, 69] < 0.01 and heart[30, 70] < 0.01                     # the cleft between its lobes
    assert heart[26, 39] > 0.99 and heart[26, 100] > 0.99                    # the tops of its lobes
    assert heart[112, 69] > 0.9 and heart[112, 70] > 0.9 and heart[112, 30] < 0.01    # its point down the middle
    assert star[24, 69] > 0.99 and star[69, 69] > 0.99                       # its top point and its middle
    assert star[117, 69] < 0.01 and star[20, 25] < 0.01                      # between its lower points, a corner
    for m in (rounded, circle, heart, star):
        soft = (m > 0.02) & (m < 0.98)
        assert soft.sum() > 100 and 0.0 <= m.min() and m.max() <= 1.0          # an antialiased rim, not a staircase


# ================================================================================================= where it stands
def test_it_stands_at_at_size_tall_ratio_wide_and_its_keys_turn_shift_and_scale_it_about_its_middle():
    fx = FX.normalize(entry(at=[0.25, 0.5], size=0.5, ratio=2.0, keys=[
        {"t": 1.0}, {"t": 2.0, "turn": 90, "dx": 0.1, "scale": 0.5, "ease": "linear"}]), DAWN)
    M, box, s = VS.place(fx, 0.5, (W, H))
    assert box == pytest.approx((90.0, 45.0)) and s == 1.0
    assert np.array(M) == pytest.approx(np.array([[1.0, 0.0, 39.5], [0.0, 1.0, 44.5]]))
    M, box, s = VS.place(fx, 2.0, (W, H))
    assert s == 0.5 and box == pytest.approx((90.0, 45.0))
    right = np.array(M) @ [45.0, 0.0, 1.0]           # its right side's middle: turned up (ccw), halved, 0.1 x 90 px on
    assert right == pytest.approx([39.5 + 9.0, 44.5 - 22.5])


def test_an_opaque_sticker_covers_its_box_and_turns_about_its_middle():
    pl = plan(entry(ratio=2.0, radius=0.0, keys=[{"t": 1.0}, {"t": 1.5, "turn": 90, "ease": "linear"},
                                                 {"t": 1.8, "dx": 0.1}]))
    ffx = run(pl, Clip(RED))
    rest, up, on = (ffx.frame(frame(t), grey())[..., 1] < 0.2 for t in (1.0, 1.5, 1.8))   # red: no green
    assert box_of(rest) == (44, 27, 115, 62)          # 72 x 36 px about (79.5, 44.5): 0.4 frame heights, twice as wide
    assert box_of(up) == (62, 9, 97, 80)              # stood on end about its middle
    assert box_of(on) == (71, 9, 106, 80)             # and 9 px on
    assert ffx.frame(frame(0.9), grey()).tolist() == grey().tolist()      # before its span: the picture as it is


def test_the_clip_is_fitted_to_cover_the_box_at_the_size_it_shows_at_from_its_in():
    pl = plan(entry(ratio=2.0, focus=[0.2, 0.7], mirror=True, rate=0.5, loop=True,
                    keys=[{"t": 1.0}, {"t": 1.5, "scale": 2.0, "ease": "linear"}], **{"in": 3.0}))
    clip = Clip()
    ffx = run(pl, clip)
    for t in (1.0, 1.5):
        ffx.frame(frame(t), grey())
    a, b = clip.asked
    assert a == {"t": 0.0, "size": (72, 36), "start": 3.0, "rate": 0.5, "loop": True, "fit": "cover",
                 "focus": (0.2, 0.7), "mirror": True}
    assert b["t"] == pytest.approx(0.5) and b["size"] == (144, 72)     # twice as large on screen: fitted twice as large


def test_a_real_clip_shows_its_frame_of_the_time_in_the_sticker(tmp_path):
    grey_clip(tmp_path / "g.mp4")
    pl = plan(entry(clip="g.mp4", shape="circle", rate=0.5, **{"in": 0.25}))
    ffx = FF.FrameFx(pl["fx"], (W, H), fps=FPS, frame0=F0, frame_of=lambda g: None, point_of=None, output="16x9",
                     footage_of=FT.source(tmp_path))
    for k in (0, 12, 29):
        out = ffx.frame(frame(1.0) + k, grey())
        t = 0.25 + 0.5 * k / FPS                                       # `in` on the span's first frame, half speed
        assert which(out[38:52, 72:88]) == int(np.floor(t * 24 + 1e-6))
        assert out[5, 5].tolist() == pytest.approx(list(GREY))         # round it: the picture


def test_a_crop_makes_a_part_of_the_clip_s_frame_the_sticker_s_video(tmp_path):
    vw = cv2.VideoWriter(str(tmp_path / "desk.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 24.0, (64, 32))
    for _ in range(4):                                # a white screen with a black note, the crop well inside the note
        pic = np.full((32, 64, 3), 255, np.uint8)
        pic[0:16, 32:64] = 0
        vw.write(pic)
    vw.release()
    pl = plan(entry(clip="desk.mp4", crop=[40 / 64, 4 / 32, 60 / 64, 12 / 32], ratio=2.5, radius=0.0, size=0.4))
    ffx = FF.FrameFx(pl["fx"], (W, H), fps=FPS, frame0=F0, frame_of=lambda g: None, point_of=None, output="16x9",
                     footage_of=FT.source(tmp_path))
    out = ffx.frame(frame(1.0), grey())
    assert out[30:60, 40:120].max() < 0.1 and out[5, 5].tolist() == pytest.approx(list(GREY))   # the note alone


# ================================================================================================= how it is laid on
def face(blend, colour, **spec):
    """The sticker's middle pixel over GREY: its face as `blend` lays a flat clip of `colour` there."""
    pl = plan(entry(blend=blend, **spec))
    return run(pl, Clip(colour)).frame(frame(1.0), grey())[44, 79].tolist()


def test_over_lays_the_clip_opaque_screen_and_add_lay_its_light_so_its_black_vanishes():
    assert face("over", RED) == pytest.approx(list(RED))
    assert face("over", BLACK) == pytest.approx(list(BLACK))
    assert face("screen", BLACK) == pytest.approx(list(GREY))                   # the clip's black: gone
    assert face("screen", (0.5, 0.0, 0.0)) == pytest.approx([1.0 - 0.6 * 0.5, 0.4, 0.4])
    assert face("add", BLACK) == pytest.approx(list(GREY))
    assert face("add", (0.5, 0.0, 0.0)) == pytest.approx([0.9, 0.4, 0.4])
    assert face("add", (0.8, 0.0, 0.0)) == pytest.approx([1.0, 0.4, 0.4])       # clipped
    assert face("over", RED, strength=0.5) == pytest.approx([0.7, 0.2, 0.2])    # mixed in halfway


def test_a_negative_face_is_the_picture_under_it_inverted_and_reads_no_clip():
    spec = {k: v for k, v in entry(blend="negative", radius=0.0).items() if k != "clip"}
    pic = np.zeros((H, W, 3), np.float32)
    pic[..., 0] = np.linspace(0.0, 1.0, W, dtype=np.float32)                 # a red ramp under it, left to right
    pic[..., 2] = 0.25

    def no_footage(clip):
        raise AssertionError(f"a negative sticker read the clip {clip!r}")

    pl = plan(spec)
    FF.check_footage(pl["fx"], no_footage)                                     # mk post's check passes it by
    out = FF.FrameFx(pl["fx"], (W, H), fps=FPS, frame0=F0, frame_of=lambda g: None, point_of=None, output="16x9",
                     footage_of=no_footage).frame(frame(1.0), pic)
    for x in (65, 94):                                                         # the face spans x 61.5 .. 97.5
        assert out[44, x].tolist() == pytest.approx([1.0 - pic[44, x, 0], 1.0, 0.75], abs=1e-5)
    assert out[5, 5].tolist() == pytest.approx(pic[5, 5].tolist())             # outside it, the picture as it was


def test_its_edge_is_paper_round_the_shape_and_what_is_opaque_casts_the_shadow_down_right():
    edge, shadow = {"color": "#ffffff", "width": 60}, {"offset": [0.05, 0.05], "blur": 12, "alpha": 0.6}
    # 5 px of edge at 90 px; the shadow 4.5 px right and down, a 1 px blur; the face spans x 61.5 .. 97.5, y 26.5 .. 62.5
    over = run(plan(entry(radius=0.0, edge=edge, shadow=shadow)), Clip(BLACK)).frame(frame(1.0), grey())
    light = run(plan(entry(radius=0.0, blend="screen", edge=edge, shadow=shadow)), Clip(BLACK)).frame(frame(1.0),
                                                                                                      grey())
    for out in (over, light):
        assert out[44, 100].tolist() == pytest.approx(list(WHITE), abs=1e-3)       # the edge, 3 px out: paper
        assert out[17, 52].tolist() == pytest.approx(list(GREY), abs=1e-3)         # up and left: no shadow
        assert out[66, 104, 0] < 0.2                                               # down and right: its shadow
    assert over[44, 79].tolist() == pytest.approx(list(BLACK))                      # an opaque face over its shadow
    assert over[40, 63].tolist() == pytest.approx(list(BLACK))
    assert light[44, 79].tolist() == pytest.approx(list(GREY))                      # light casts none: the picture
    assert light[40, 63, 0] < 0.2                                                   # but the rim's shadow falls in it


def test_each_output_has_its_own_placement():
    pl = plan(entry(radius=0.0, aspect={"9x16": {"at": [0.5, 0.8], "size": 0.1}}))
    wide = run(pl, Clip(RED)).frame(frame(1.0), grey())[..., 1] < 0.2
    tall = run(pl, Clip(RED), "9x16").frame(frame(1.0), grey((H, W)))[..., 1] < 0.2
    assert box_of(wide) == (62, 27, 97, 62)
    assert box_of(tall) == (37, 120, 52, 135)          # 16 px about (44.5, 127.5) of the tall frame


# ================================================================================================= its clip
def test_credits_count_its_clip_and_post_checks_it_before_encoding(tmp_path, monkeypatch):
    monkeypatch.setenv("MK_ASSETS", str(tmp_path / "assets"))
    reg = A.Registry()
    reg.add({"kind": "reference", "slug": "city", "path": str(tmp_path / "footage" / "city.mp4"), "author": "a"})
    data = {"fx": [{"kind": "clip", "clip": "night", "aspect": {"9x16": {"clip": "footage/city.mp4"}}},
                   {"kind": "footage", "clip": "road"}, {"kind": "grid", "clip": "nope"}]}
    assert footage_clips(data) == ["night", "footage/city.mp4", "road"]

    class Proj:
        root = tmp_path
        cast = []

    Proj.data = data
    assert project_credits(Proj, reg) == (["night", "city", "road"], [])     # a slug, or a file the registry knows
    grey_clip(tmp_path / "g.mp4")
    pl = plan(entry(clip="g.mp4", **{"in": 0.5}), entry(clip="g.mp4", **{"in": 2.0}))
    src = FT.source(tmp_path)
    FF.check_footage(pl["fx"][:1], src)
    with pytest.raises(FT.FootageError, match=r"fx 1 \(clip\): g.mp4: in = 2 s is past its end"):
        FF.check_footage(pl["fx"], src)
    with pytest.raises(FT.FootageError, match=r"fx 0 \(clip\): clip 'none.mp4'"):
        FF.check_footage(plan(entry(clip="none.mp4"))["fx"], src)
    with pytest.raises(FT.FootageError, match=r"fx 0 \(clip\): clip 'none.mp4'"):    # an output's own clip too
        FF.check_footage(plan(entry(clip="g.mp4", aspect={"9x16": {"clip": "none.mp4"}}))["fx"], src)
    ffx = FF.FrameFx(pl["fx"], (W, H), fps=FPS, frame0=F0, frame_of=lambda g: None, point_of=None)
    with pytest.raises(FileNotFoundError, match="no footage to read"):
        ffx.frame(frame(1.0), grey())


def test_it_reads_no_layers_and_mk_post_draws_it_on_the_cpu():
    pl = plan(entry(keys=[{"t": 1.2, "turn": 10}]))
    fx = pl["fx"][0]
    assert not FX.uses_figure(fx) and FX.points(fx) == [] and FX.sources(fx, frame(1.5), F0) == []
    assert "clip" not in GFX.KINDS and PG.ops(run(pl, Clip()), fx, frame(1.5)) is None
