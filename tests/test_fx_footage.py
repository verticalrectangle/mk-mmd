"""Footage plates (mkmmd.footage: a clip read frame by frame and fitted to a picture; the `footage` frame effect laying a
figure over it as a sticker), figure moves for collages (`keys` about a `pivot`), per-output `aspect.<output>` tables of
[[fx]], `mk ref footage` (Pexels videos into a project and the asset registry) and the credits a footage clip earns."""
import json
import types

import cv2
import numpy as np
import pytest

from mkmmd import assets as A
from mkmmd import footage as FT
from mkmmd import framefx as FF
from mkmmd.cli import main as MAIN
from mkmmd.cli.assets import project_credits
from mkmmd.cli.export import NO_REDISTRIBUTION
from mkmmd.core import framefx as FX
from mkmmd.core import palette as PAL
from mkmmd.core import transition as TR
from mkmmd.gpufx import footage as GFT
from mkmmd.ref import footage as RF
from mkmmd.ref import pexels as PX

FPS, F0 = 30, 1
DAWN = PAL.get("rose-pine-dawn")
W, H = 64, 36
WHITE, BLACK, RED = [1.0, 1.0, 1.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]


def project(fx=(), **extra):
    return {"project": {"duration": 4.0}, "output": [{"name": "16x9", "size": [W, H]}, {"name": "9x16", "size": [H, W]}],
            "shot": [{"name": "a", "from": 0.0, "to": 2.0}, {"name": "b", "from": 2.0, "to": 4.0}],
            "fx": list(fx), **extra}


def plan(fx=(), **extra):
    return TR.plan(project(fx, **extra), FPS, F0, DAWN)


def footage(**kw):
    return {"kind": "footage", "from": 1.0, "to": 2.0, "objects": "Fig*", "clip": "clip.mp4", **kw}


def grey_clip(path, n=25, fps=24.0, size=(48, 32)):
    """A clip whose frame i is flat grey 16 + 9 i (the codec takes a few levels off): which frame a picture came from
    can be read off it."""
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    assert vw.isOpened()
    for i in range(n):
        vw.write(np.full((size[1], size[0], 3), 16 + 9 * i, np.uint8))
    vw.release()
    return path


def which(rgb):
    """The frame number of a picture of grey_clip (its mean grey)."""
    return int(round((float(np.mean(rgb)) * (255.0 if np.asarray(rgb).dtype != np.uint8 else 1.0) - 16.0) / 9.0))


def square(x0, x1, y0, y1, size=(W, H)):
    m = np.zeros((size[1], size[0]), np.float32)
    m[y0:y1, x0:x1] = 1.0
    return m


class Plate:
    """A clip stand-in: every picture flat `colour`; records the times and starts it was asked for."""

    def __init__(self, colour=(0.2, 0.4, 0.6)):
        self.colour, self.asked = colour, []

    def index(self, t, start=0.0, rate=1.0, loop=False):
        return 0

    def picture(self, t, size, start=0.0, rate=1.0, loop=False, fit="cover", focus=(0.5, 0.5), mirror=False,
                back=(0.0, 0.0, 0.0)):
        self.asked.append((round(t, 4), start, tuple(focus)))
        return np.full((size[1], size[0], 3), self.colour, np.float32)


# ---------------------------------------------------------------- the entries
@pytest.mark.parametrize("spec, frag", [
    ({"kind": "footage", "from": 1, "to": 2, "objects": "Fig*"}, "`clip` is required"),
    (footage(fit="stretch"), "none of cover, contain"),
    (footage(focus=[0.5, 1.5]), "in 0..1 of the video"),
    (footage(rate=-1), "rate = -1: expected 0..16"),
    (footage(loop="yes"), "loop is true or false"),
    (footage(keys=[{"turn": 90}]), "has no `t`"),
    (footage(keys=[{"t": 1.2, "spin": 3}]), "unknown keys ['spin']"),
    (footage(keys=[{"t": 1.2, "ease": "bounce"}]), "none of inout, in, out, linear, back"),
    (footage(keys=[{"t": 1.2, "scale": 0}]), "expected a positive number"),
    (footage(keys=[{"t": 1.2}, {"t": 1.2, "turn": 4}]), "two keys at the same second"),
    (footage(shadow={"colour": "text"}), "expected {color, offset, blur, alpha}"),
    (footage(shadow={"alpha": 2}), "alpha = 2: expected 0..1"),
    (footage(pivot="feet"), "\"base\", \"center\" or [x, y]"),
    (footage(aspect={"9x16": {"from": 1.5}}), "may not set ['from']"),
    (footage(aspect={"9x16": 3}), "expected a table of the entry's keys"),
    (footage(aspect={"9x16": {"focal": [0.2, 0.2]}}), "unknown keys ['focal']"),
    ({"kind": "stack", "from": 1, "to": 2, "objects": "Fig*", "aspect": {"9x16": {"delay": 2}}}, "reads frames the entry"),
])
def test_bad_footage_and_aspect_entries_say_what_is_wrong(spec, frag):
    with pytest.raises(TR.TransitionError, match=frag.replace("[", r"\[").replace("]", r"\]")):
        plan([spec])


def test_an_aspect_table_is_the_entry_for_its_output_and_frame_fx_draws_the_output_s_own():
    pl = plan([footage(focus=[0.3, 0.3], aspect={"9x16": {"focus": [0.7, 0.2], "mirror": True}})])
    fx = pl["fx"][0]
    tall = FX.for_output(fx, "9x16")
    assert tall["focus"] == [0.7, 0.2] and tall["mirror"] is True and fx["focus"] == [0.3, 0.3]
    assert (tall["index"], tall["f0"], tall["f1"]) == (fx["index"], fx["f0"], fx["f1"])     # one span for every output
    assert FX.for_output(fx, "16x9") is fx and FX.for_output(fx, None) is fx
    clips = {}
    for out, size in (("16x9", (W, H)), ("9x16", (H, W))):
        plate = clips[out] = Plate()
        run = FF.FrameFx(pl["fx"], size, fps=FPS, frame0=F0, frame_of=lambda g: None, point_of=None, output=out,
                         figure_of=lambda key, g, size=size: np.zeros((size[1], size[0]), np.float32),
                         footage_of=lambda clip, plate=plate: plate)
        run.frame(F0 + 30, np.zeros((size[1], size[0], 3), np.float32))
    assert clips["16x9"].asked[0][2] == (0.3, 0.3) and clips["9x16"].asked[0][2] == (0.7, 0.2)


# ---------------------------------------------------------------- the clip
def test_a_clip_reads_any_frame_in_any_order_and_maps_span_seconds_to_its_frames(tmp_path):
    clip = FT.Clip(grey_clip(tmp_path / "g.mp4"))
    assert (clip.fps, clip.count, clip.size) == (24.0, 25, (48, 32)) and clip.duration == pytest.approx(25 / 24)
    for i in (0, 1, 2, 20, 5, 6, 24, 0, 17):                          # on, a jump ahead, back, the end, the start
        assert which(clip.frame(i)) == i
    assert clip.frame(17) is clip.frame(17)                           # kept
    with pytest.raises(IndexError):
        clip.frame(40)
    t = np.arange(12) / 30.0                                          # a 30 fps span over a 24 fps clip
    assert [clip.index(x) for x in t] == [int(np.floor(x * 24 + 1e-6)) for x in t]
    assert clip.index(0.25, start=0.5) == 18 and clip.index(0.25, start=0.5, rate=0.5) == 15
    assert clip.index(5.0) == 24 and clip.index(5.0, loop=True) == 120 % 25        # past the end: held, or round
    assert clip.index(3.0, start=0.2, rate=0.0) == 4                  # rate 0: a still
    with pytest.raises(FT.FootageError, match="past its end"):
        clip.index(0.0, start=1.1)
    assert clip.index(0.0, start=1.1, loop=True) == 26 % 25
    pic = clip.picture(0.25, (24, 16), start=0.5)
    assert pic.dtype == np.float32 and pic.shape == (16, 24, 3) and which(pic) == 18
    assert clip.picture(0.25, (24, 16), start=0.5) is pic             # the same frame at the same size: kept


def test_a_later_output_shows_the_clip_frames_an_earlier_one_decoded_as_its_own_fit_makes_them(tmp_path):
    path = tmp_path / "clip.mp4"
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 24.0, (48, 32))
    for i in range(12):                                               # a ramp across that moves on each frame
        vw.write(np.broadcast_to(((np.arange(48) * 5 + i * 9) % 256).astype(np.uint8)[None, :, None], (32, 48, 3)).copy())
    vw.release()
    span = {"clip": "clip.mp4", "in": 0.0, "rate": 1.0, "loop": False, "f0": 10, "f1": 20, "fit": "cover",
            "focus": [0.5, 0.5], "mirror": False}
    first, later, other = span, {**span, "focus": [0.0, 0.5], "mirror": True}, {**span, "in": 0.2}
    (tmp_path / "kept").mkdir()
    kept = GFT.Kept(tmp_path / "kept", [("a", lambda: (48, 32), {0: first}), ("b", lambda: (16, 32), {0: later}),
                                        ("c", lambda: (16, 32), {0: other})])
    src = FT.source(tmp_path)

    def hand(name, size, e, i):
        op = {"op": "footage", "clip": {"name": "clip.mp4", "index": i, "fit": e["fit"], "focus": e["focus"],
                                        "mirror": e["mirror"], "entry": 0}}
        kept.begin(name)
        GFT.hand(types.SimpleNamespace(size=size, _footage_of=src), op, tmp_path, kept=kept)
        got = np.fromfile(op["clip"]["path"], np.uint8).reshape(size[1], size[0], 3)
        want, _ = FT.fitted(FT.Clip(path).frame(i), size, e["fit"], e["focus"], e["mirror"])
        assert np.array_equal(got, want), (name, i)                   # as the output's own decode and fit make it
        return op["clip"].get("keep", False)
    try:
        assert [hand("a", (48, 32), first, i) for i in (3, 4)] == [False, False]
        assert [hand("b", (16, 32), later, i) for i in (3, 4, 5)] == [True, True, False]   # 5: decoded for b itself
        assert not hand("c", (16, 32), other, 3)                      # another span: not taken from a's frames
    finally:
        kept.close()
    assert not (tmp_path / "kept").exists()


def test_a_missing_or_broken_clip_and_a_span_past_its_end_are_errors_that_name_the_entry(tmp_path):
    (tmp_path / "junk.mp4").write_bytes(b"not a video")
    with pytest.raises(FT.FootageError, match="does not exist"):
        FT.resolve("footage/none.mp4", tmp_path)
    with pytest.raises(FT.FootageError, match="not a video OpenCV can read"):
        FT.Clip(tmp_path / "junk.mp4")
    grey_clip(tmp_path / "g.mp4")
    pl = plan([footage(clip="g.mp4", **{"in": 0.5}), footage(clip="g.mp4", **{"in": 2.0})])
    src = FT.source(tmp_path)
    FF.check_footage(pl["fx"][:1], src)
    with pytest.raises(FT.FootageError, match=r"fx 1 \(footage\): g.mp4: in = 2 s is past its end"):
        FF.check_footage(pl["fx"], src)
    with pytest.raises(FT.FootageError, match=r"fx 0 \(footage\): clip 'none.mp4'"):
        FF.check_footage(plan([footage(clip="none.mp4")])["fx"], src)
    run = FF.FrameFx(pl["fx"], (W, H), fps=FPS, frame0=F0, frame_of=lambda g: None, point_of=None,
                     figure_of=lambda key, g: np.zeros((H, W), np.float32))
    with pytest.raises(FileNotFoundError, match="no footage to read"):
        run.frame(F0 + 30, np.zeros((H, W, 3), np.float32))


def test_a_clip_is_a_file_or_a_registry_slug_whose_path_is_a_video(tmp_path, monkeypatch):
    monkeypatch.setenv("MK_ASSETS", str(tmp_path / "assets"))
    clip = grey_clip(tmp_path / "g.mp4")
    reg = A.Registry()
    reg.add({"kind": "reference", "slug": "pexels_7", "path": str(clip), "license": "x", "author": "y"})
    reg.add({"kind": "font", "slug": "type", "path": str(tmp_path / "type.ttf"), "license": "x", "author": "y"})
    reg.save()
    assert FT.resolve("pexels_7") == clip and FT.resolve("g.mp4", tmp_path) == clip
    assert FT.resolve(str(clip)) == clip                               # absolute
    assert FT.is_file("a/b") and FT.is_file("x.MOV") and FT.is_file("~/x") and not FT.is_file("pexels_7")
    with pytest.raises(FT.FootageError, match="the videos there: pexels_7.*mk ref footage"):
        FT.resolve("pexels_8")
    with pytest.raises(FT.FootageError, match="is not a video"):
        FT.resolve("type")
    src = FT.source(tmp_path)
    assert src("pexels_7") is src("pexels_7") and src("pexels_7").count == 25


def test_cover_fills_the_frame_about_the_focus_contain_shows_it_whole_and_mirror_flips_it():
    xx, yy = np.meshgrid(np.arange(200), np.arange(100))
    frame = np.dstack([xx * 255 // 199, yy * 255 // 99, np.zeros_like(xx)]).astype(np.uint8)   # red: column, green: row
    mid = FT.fit(frame, (50, 50))                                     # a square of a 2:1 frame: its middle half
    assert mid.shape == (50, 50, 3) and mid[25, 25, 0] == pytest.approx(0.5, abs=0.02)
    left = FT.fit(frame, (50, 50), focus=(0.1, 0.5))                   # the focus kept as central as the crop allows
    assert left[25, 0, 0] == pytest.approx(0.0, abs=0.02) and left[25, 25, 0] == pytest.approx(0.25, abs=0.02)
    right = FT.fit(frame, (50, 50), focus=(0.6, 0.5))
    assert right[25, 25, 0] == pytest.approx(0.6, abs=0.02)
    edge = FT.fit(frame, (50, 50), focus=(0.9, 0.5))                  # the crop stops at the frame's edge
    assert edge[25, 49, 0] == pytest.approx(1.0, abs=0.01) and edge[25, 25, 0] == pytest.approx(0.75, abs=0.02)
    flipped = FT.fit(frame, (50, 50), focus=(0.6, 0.5), mirror=True)  # the same point, the picture flipped
    assert np.allclose(flipped, right[:, ::-1], atol=0.02)
    whole = FT.fit(frame, (100, 100), "contain", back=(0.0, 0.0, 1.0))
    assert whole[0, 50].tolist() == [0.0, 0.0, 1.0] and whole[99, 50].tolist() == [0.0, 0.0, 1.0]   # bars above, below
    assert whole[25:75, 0, 0].max() < 0.02 and whole[25:75, 99, 0].min() > 0.98                      # all of it between
    tall = FT.fit(frame, (30, 120))                                    # 9:16-ish out of 2:1: a narrow slice, upscaled
    assert tall.shape == (120, 30, 3) and tall[60, 15, 0] == pytest.approx(0.5, abs=0.03)


# ---------------------------------------------------------------- the sticker
def test_a_sticker_is_its_picture_in_a_white_border_casting_a_soft_shadow_down_right_on_the_plate():
    plate = np.full((H, W, 3), 0.5, np.float32)
    pic = np.zeros((H, W, 3), np.float32)
    pic[..., 0] = 1.0
    m = square(24, 40, 8, 24)
    out = FF._paste(plate, pic, m, None, edge=(WHITE, 3.0), shadow=(BLACK, (4.0, 4.0), 1.5, 0.5))
    assert out[15, 30].tolist() == RED                                # the figure: opaque, never blended
    assert out[15, 41].tolist() == pytest.approx(WHITE, abs=1e-3) and out[6, 30].tolist() == pytest.approx(WHITE, abs=1e-3)
    assert 0.2 < out[28, 44, 0] < 0.35                                # its shadow below right, past the border, soft
    assert out[2, 18].tolist() == pytest.approx([0.5] * 3, abs=1e-3)  # above left: the plate as it was
    assert out[15, 2].tolist() == pytest.approx([0.5] * 3, abs=1e-3)
    ring = FF._grow(m, 3.0)
    assert ring[15, 41] > 0.99 and ring[15, 43] < 0.01 and 0.2 < ring[15, 42] < 0.8      # 3 px out, antialiased


def test_footage_lays_the_figure_over_the_clip_s_frame_of_the_time_as_a_sticker(tmp_path):
    grey_clip(tmp_path / "g.mp4")
    pl = plan([footage(clip="g.mp4", rate=0.5, edge={"color": "#ffffff", "width": 60}, **{"in": 0.25})])  # 2 px at 36
    m = square(20, 30, 10, 26)
    run = FF.FrameFx(pl["fx"], (W, H), fps=FPS, frame0=F0, frame_of=lambda g: None, point_of=None, output="16x9",
                     figure_of=lambda key, g: m, footage_of=FT.source(tmp_path))
    img = np.zeros((H, W, 3), np.float32)
    img[..., 1] = 1.0                                                  # the frame: her, green
    for k in (0, 12, 29):
        out = run.frame(F0 + 30 + k, img)
        t = 0.25 + 0.5 * k / FPS                                       # `in` on the span's first frame, half speed
        assert which(out[2:8, 40:60]) == int(np.floor(t * 24 + 1e-6))
        assert out[18, 25].tolist() == [0.0, 1.0, 0.0] and out[18, 30].tolist() == pytest.approx(WHITE, abs=1e-3)


# ---------------------------------------------------------------- figure moves
def test_a_move_eases_from_key_to_key_by_the_key_it_arrives_at_and_holds_outside():
    keys = FX.normalize(footage(keys=[{"t": 1.6, "turn": -90, "ease": "in", "dx": 0.1}, {"t": 1.2},
                                      {"t": 1.8, "scale": 2.0, "ease": "linear"}, {"t": 1.9, "turn": 0, "ease": "back"}]),
                        DAWN)["keys"]
    assert [k["t"] for k in keys] == [1.2, 1.6, 1.8, 1.9]
    assert keys[2]["turn"] == -90 and keys[2]["dx"] == 0.1 and keys[3]["scale"] == 2.0     # left out: as before
    at = lambda t: FX.move_at(keys, t)                                                    # noqa: E731
    assert at(0.0) == FX.MOVE and at(1.2) == FX.MOVE                  # held before the first (at rest)
    assert at(1.4)["turn"] == pytest.approx(-90 * 0.25)               # `in`: u^2
    assert at(1.7)["scale"] == pytest.approx(1.5) and at(1.7)["turn"] == -90               # `linear`
    assert at(1.88)["turn"] > 0.0 and at(1.9)["turn"] == pytest.approx(0.0)                # `back` overshoots, settles
    assert at(5.0) == {"turn": 0.0, "dx": 0.1, "dy": 0.0, "scale": 2.0}
    io = FX.normalize(footage(keys=[{"t": 1.0}, {"t": 2.0, "dy": 1.0}]), DAWN)["keys"]
    assert FX.move_at(io, 1.25)["dy"] == pytest.approx(0.15625) and FX.move_at(io, 1.5)["dy"] == pytest.approx(0.5)
    M = np.array(FX.move_matrix({"turn": 90.0, "dx": 0.0, "dy": 0.5, "scale": 2.0}, (10.0, 20.0), 36.0))
    assert M @ [11.0, 20.0, 1.0] == pytest.approx([10.0, 20.0 - 2.0 + 18.0])   # right of the pivot: up (ccw), x2, dy


def test_a_figure_tips_over_about_its_base_counter_clockwise_and_pops_back():
    pl = plan([footage(keys=[{"t": 1.0}, {"t": 1.5, "turn": 90, "ease": "linear"}, {"t": 1.8, "turn": 0}])])
    m = square(30, 34, 10, 30)                                         # a standing bar, its base at (31.5, 29.5)
    plate = Plate((0.0, 0.0, 0.0))
    run = FF.FrameFx(pl["fx"], (W, H), fps=FPS, frame0=F0, frame_of=lambda g: None, point_of=None,
                     figure_of=lambda key, g: m, footage_of=lambda clip: plate)
    img = np.ones((H, W, 3), np.float32)
    up, flat, back = (run.frame(F0 + f, img)[..., 0] for f in (30, 45, 54))
    assert up[12, 32] == 1.0 and up[28, 20] == 0.0
    assert flat[12, 32] == 0.0 and flat[29, 16] == 1.0 and flat[29, 33] == 0.0 and run._pivots[0] == (31.5, 29.5)
    assert back[29, 16] == 0.0 and back[12, 32] == 1.0
    mid = run.frame(F0 + 37, img)[..., 0]                              # 7 frames of 15: about 42 degrees over
    ys, xs = np.nonzero(mid > 0.5)
    tip = np.argmin(ys)
    assert xs[tip] < 25 and 11 < ys[tip] < 20


def test_a_pivot_found_on_the_figure_is_read_on_the_span_s_first_frame():
    pl = plan([footage(keys=[{"t": 1.2, "turn": 10}]), footage(keys=[{"t": 1.2, "turn": 10}], pivot=[0.5, 0.9]),
               footage(clip="b.mp4")])
    assert [FX.sources(fx, F0 + 40, F0) for fx in pl["fx"]] == [[F0 + 30], [], []]
    assert FX.sources(pl["fx"][0], F0 + 30, F0) == []
    key = FX.figure_key(pl["fx"][0])
    got = TR.fx_frames(pl, [F0 + 40])
    assert {"kind": "figure", "key": key, "objects": ["Fig*"], "shot": "a"} in got[F0 + 30]
    gone = plan([footage(keys=[{"t": 1.2, "turn": 10}])])
    run = FF.FrameFx(gone["fx"], (W, H), fps=FPS, frame0=F0, frame_of=lambda g: None, point_of=None,
                     figure_of=lambda key, g: np.zeros((H, W), np.float32), footage_of=lambda clip: Plate())
    with pytest.raises(FX.FxError, match="not in the frame on the span's first frame"):
        run.frame(F0 + 40, np.zeros((H, W, 3), np.float32))


def test_every_collage_moves_its_figure_by_its_keys():
    pl = plan([{"kind": "grid", "from": 1.0, "to": 2.0, "objects": "Fig*", "cells": 1, "back": "#000000",
                "fill": "#ffffff", "keys": [{"t": 1.0, "dx": 0.25}], "pivot": [0.5, 0.5]}])
    run = FF.FrameFx(pl["fx"], (W, H), fps=FPS, frame0=F0, frame_of=lambda g: None, point_of=None,
                     figure_of=lambda key, g: square(10, 14, 10, 20))
    out = run.frame(F0 + 35, np.zeros((H, W, 3), np.float32))
    assert out[15, 11].tolist() == BLACK and out[15, 20].tolist() == WHITE        # 0.25 x 36 = 9 px to the right


# ---------------------------------------------------------------- mk ref footage and the credits
def video(cid=4254119, w=4096, h=2160, sizes=((960, 506, 3e6), (2048, 1080, 9e6), (4096, 2160, 38e6), (7680, 4320, 4e8))):
    return {"id": cid, "url": f"https://www.pexels.com/video/driving-on-a-forest-road-{cid}/", "width": w, "height": h,
            "duration": 13, "user": {"name": "K", "url": "https://www.pexels.com/@k"},
            "video_files": [{"file_type": "video/mp4", "width": a, "height": b, "fps": 23.98, "size": int(s),
                             "link": f"https://videos.pexels.com/{a}.mp4"} for a, b, s in sizes]}


def test_the_largest_rendition_under_the_cap_and_a_registry_entry_that_credits_its_author():
    v = video()
    best = RF.pick_best(PX.mp4_files(v), 1080)
    assert (best["width"], best["height"]) == (4096, 2160)             # the 8K one is over the cap
    assert RF.pick_best(PX.mp4_files(v), 2500) is None
    assert RF.orientation(1080, 1920) == "portrait" and RF.orientation(5, 5) == "square"
    assert RF.title(v) == "Driving on a forest road" and RF.title({"id": 3, "url": "x/video/3/"}) == "Pexels video 3"
    e = RF.entry(v, "/tmp/footage/pexels_4254119.mp4", best)
    assert e["slug"] == "pexels_4254119" and e["kind"] == "reference" and e["author_url"] == "https://www.pexels.com/@k"
    assert e["license"] == "Pexels License (https://www.pexels.com/license/)" and "K" in e["credit"]
    assert not NO_REDISTRIBUTION.search(e["restrictions"])            # it may still be exported (stock sites only)


def mk(argv, capsys):
    with pytest.raises(SystemExit) as e:
        MAIN.main(argv)
    return e.value.code, json.loads(capsys.readouterr().out)


def test_mk_ref_footage_downloads_into_the_project_registers_the_asset_and_credits_follow(tmp_path, monkeypatch,
                                                                                          capsys):
    monkeypatch.setenv("MK_ASSETS", str(tmp_path / "assets"))
    monkeypatch.setenv("MK_CACHE", str(tmp_path / "cache"))
    root = tmp_path / "proj"
    root.mkdir()
    (root / "mk.toml").write_text('[project]\nname = "p"\nfps = 30\nframe0 = 1\nduration = 4.0\n\n[[fx]]\n'
                                  'kind = "footage"\nfrom = 1.0\nto = 2.0\nobjects = "Fig*"\nclip = "pexels_4254119"\n')
    calls = []
    monkeypatch.setattr(PX, "api_key", lambda: "SECRET")
    monkeypatch.setattr(PX, "video", lambda cid, key=None: video(int(cid), 1920, 1080) if cid != 9 else video(9, 1080, 1920))

    def download(url, dest, max_bytes, expected=None, label=None):
        calls.append(url)
        dest.parent.mkdir(parents=True, exist_ok=True)
        grey_clip(dest)
        return dest.stat().st_size
    monkeypatch.setattr(PX, "download", download)
    code, out = mk(["ref", "footage", "4254119", "9", "--orientation", "landscape", "--min-height", "32",   # 48x32 files
                    "--project", str(root)], capsys)
    assert code == 1 and out["failed"] == [{"id": 9, "error": "it is portrait (1080x1920), not landscape"}]
    clip = out["clips"][0]
    assert clip["slug"] == "pexels_4254119" and clip["path"] == "footage/pexels_4254119.mp4"
    assert clip["seconds"] == pytest.approx(25 / 24, abs=0.01) and clip["size"] == [48, 32]
    assert calls == ["https://videos.pexels.com/4096.mp4"] and "SECRET" not in json.dumps(out)
    e = A.Registry().get("pexels_4254119")
    assert e["path"] == str((root / "footage" / "pexels_4254119.mp4").resolve()) and e["author"] == "K"
    reg = A.Registry()
    reg.update("pexels_4254119", tags=["mine"])
    reg.save()
    code, out = mk(["ref", "footage", "4254119", "--min-height", "32", "--project", str(root)], capsys)
    assert code == 0 and A.Registry().get("pexels_4254119")["tags"] == ["mine", "footage", "pexels"]
    code, out = mk(["assets", "credits", "--project", str(root)], capsys)
    assert code == 0 and out["assets"] == ["pexels_4254119"] and "Driving on a forest road" in out["text"]


def test_mk_ref_footage_registers_the_file_s_own_size_and_not_a_file_under_min_height(tmp_path, monkeypatch, capsys):
    """Pexels can label a rendition bigger than the file it serves (a "1920x1080" that is 1280x720): the registry keeps
    the file's own size, and a file whose short side is under --min-height is not registered."""
    monkeypatch.setenv("MK_ASSETS", str(tmp_path / "assets"))
    monkeypatch.setenv("MK_CACHE", str(tmp_path / "cache"))
    root = tmp_path / "proj"
    root.mkdir()
    (root / "mk.toml").write_text('[project]\nname = "p"\nfps = 30\nframe0 = 1\nduration = 4.0\n')
    monkeypatch.setattr(PX, "api_key", lambda: "SECRET")
    monkeypatch.setattr(PX, "video", lambda cid, key=None: video(int(cid), 1920, 1080))

    def download(url, dest, max_bytes, expected=None, label=None):    # labelled 4096x2160, the file is 64x36
        dest.parent.mkdir(parents=True, exist_ok=True)
        grey_clip(dest, size=(64, 36))
        return dest.stat().st_size
    monkeypatch.setattr(PX, "download", download)
    code, out = mk(["ref", "footage", "5", "--min-height", "36", "--project", str(root)], capsys)
    assert code == 0 and out["clips"][0]["size"] == [64, 36]
    assert A.Registry().get("pexels_5")["extra"]["size"] == [64, 36]
    code, out = mk(["ref", "footage", "6", "--project", str(root)], capsys)                 # --min-height 1080
    assert code == 1 and out["clips"] == [] and out["failed"][0]["id"] == 6
    assert "4096x2160" in out["failed"][0]["error"] and "64x36" in out["failed"][0]["error"]
    assert [e["slug"] for e in A.Registry().entries] == ["pexels_5"]
    assert (root / "footage" / "pexels_6.mp4").is_file()                                # the download stays


def test_credits_count_footage_videos_and_photo_world_photos_by_slug_or_by_a_registered_file(tmp_path, monkeypatch):
    monkeypatch.setenv("MK_ASSETS", str(tmp_path / "assets"))
    reg = A.Registry()
    reg.add({"kind": "reference", "slug": "road", "path": str(tmp_path / "proj" / "footage" / "r.mp4"), "author": "a"})
    reg.add({"kind": "reference", "slug": "plate", "path": str(tmp_path / "proj" / "plates" / "p.jpg"), "author": "b"})

    class Proj:
        root = tmp_path / "proj"
        cast = [{"asset": "girl"}]
        data = {"fx": [{"kind": "footage", "clip": "footage/r.mp4", "aspect": {"9x16": {"clip": "sea"}}},
                       {"kind": "grid", "clip": "nope"}, {"kind": "footage", "clip": "unregistered/x.mp4"}],
                "insert": [{"shape": "split", "clip": {"src": "river", "in": 2.0}}, {"shape": "split", "clip": "sea"},
                           {"shape": "thought", "shot": "head"}],
                "set": [{"kind": "photo_world", "image": "street"}, {"kind": "photo_world", "image": "plates/p.jpg"},
                        {"kind": "photo_world", "image": "sky.png"}, {"kind": "night_sky", "image": "nope"},
                        {"kind": "photo_world", "image": "street"}],
                "credits": {"assets": ["font", "sea"], "lines": ["Song"]}}
    assert project_credits(Proj, reg) == (["girl", "road", "sea", "river", "street", "plate", "font"], ["Song"])
