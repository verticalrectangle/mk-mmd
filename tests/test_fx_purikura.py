"""The purikura look of a photo booth (`look = "purikura"`, mkmmd.core.purikura, mkmmd.purikura) and the gel-pen hand
it writes with (mkmmd.core.hand): its keys and presets, colours that keep one structure in a dark and a light palette,
the stamps and the writing laid out on a photo, the stamps' parts, the flashes thinned to three a second, and the strip
drawn the same in mk post's order and out of order."""
import numpy as np
import pytest

from mkmmd import framefx as FF
from mkmmd.core import framefx as FX
from mkmmd.core import hand as HD
from mkmmd.core import palette as PAL
from mkmmd.core import picturefx as PC
from mkmmd.core import purikura as PK
from mkmmd.core import transition as TR

FPS, F0 = 30, 1
MOON, DAWN = PAL.get("rose-pine-moon"), PAL.get("rose-pine-dawn")


def booth(**spec):
    return {"kind": "booth", "from": 1.0, "to": 3.0, "times": [1.2, 1.6, 2.0, 2.4], "look": "purikura", **spec}


@pytest.mark.parametrize("spec, frag", [
    ({"look": "sepia"}, "none of plain, purikura"),
    ({"look": "plain", "purikura": {}}, "give look = \"purikura\""),
    ({"purikura": {"sparkles": 3}}, "unknown keys"),
    ({"purikura": {"write": ["\u3042"]}}, "the pen does not write"),
    ({"purikura": {"write": "x" * 25}}, "short text"),
    ({"purikura": {"stamps": ["moon"]}}, "none of star"),
    ({"purikura": {"marks": [{"shape": "bow"}]}}, "`at` is required"),
    ({"purikura": {"pastel": 1.5}}, "pastel"),
    ({"purikura": {"frames": "nope"}}, "frames"),
])
def test_bad_entries_say_what_is_wrong(spec, frag):
    with pytest.raises(FX.FxError, match=frag.replace("(", r"\(")):
        FX.normalize(booth(**spec), DAWN)


def test_the_look_presets_the_booths_keys_and_the_plain_booth_stays_as_it_was():
    fx = FX.normalize(booth(), MOON)
    assert fx["doodles"] == PK.BOOTH["doodles"] and fx["border"] == PK.BOOTH["border"]
    assert fx["back"] == FX._colour(PK.BOOTH["back"], MOON, "back")
    assert FX.normalize(booth(doodles=1, back="pine"), MOON)["doodles"] == 1          # what the entry gives wins
    plain = FX.normalize({k: v for k, v in booth().items() if k != "look"}, MOON)
    assert plain["look"] == "plain" and plain["purikura"] is None
    assert plain["paper"] == [1.0, 1.0, 1.0] and plain["doodles"] == 2 and plain["border"] == 16.0


def _shape_of(v):
    if isinstance(v, dict):
        return {k: _shape_of(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_shape_of(x) for x in v]
    return v if isinstance(v, (str, bool, type(None))) or isinstance(v, int) else float


def test_its_colours_are_pastel_in_a_dark_palette_and_a_light_one_with_one_structure():
    spec = booth(purikura={"write": ["11.01"], "marks": [{"shape": "bunny_ears", "at": [0.5, 0.1]}]})
    a, b = FX.normalize(spec, MOON), FX.normalize(spec, DAWN)
    assert _shape_of(a) == _shape_of(b)                                       # a palette breath mixes them number by number
    for fx, pal in ((a, MOON), (b, DAWN)):
        light = PK.light(FX, pal, "light")
        assert fx["purikura"]["light"] == light
        for frame, slot in zip(fx["purikura"]["frames"], PK.DEFAULTS["frames"]):
            assert PK.luma(frame) > PK.luma(FX._colour(slot, pal, slot)) and PK.luma(frame) > 0.6
        rose = PK.mix(FX._colour(PK.BOOTH["paper"], pal, "paper"), light, PK.DEFAULTS["pastel"])
        assert PK.luma(fx["paper"]) > PK.luma(rose)                          # the paper lighter still


def test_stamps_sit_round_the_edges_and_leave_the_foot_to_the_writing():
    fx = FX.normalize(booth(doodles=4, purikura={"write": ["kawaii", ""], "marks": [{"shape": "bow", "at": [0.5, 0.1]}]}),
                      DAWN)
    fx["index"] = 0
    for cell, writes in ((0, True), (1, False), (2, False)):
        ds = PK.stamps(fx, cell, (100.0, 125.0), 1.2)
        assert len(ds) == 5 and ds[-1]["shape"] == "bow" and ds[-1]["at"] == (50.0, 12.5)
        for d in ds[:4]:
            u, v = d["at"][0] / 100.0 - 0.5, d["at"][1] / 125.0 - 0.5
            assert d["shape"] in fx["purikura"]["stamps"] and 0.32 < np.hypot(u, v) < 0.44
            assert d["t"] > 1.2 + fx["drop"]
            if writes:
                assert not (v > 0.2 and abs(u) < 0.25)                            # not over the writing in its foot
        assert PK.stamps(fx, cell, (100.0, 125.0), 1.2) == ds                   # seeded


def test_the_writing_fits_the_photo_and_starts_once_it_has_landed():
    fx = FX.normalize(booth(purikura={"write": ["2026.11.01", "", "BFF \u2661"], "speed": 10.0}), MOON)
    fx["index"] = 0
    w = PK.writing(fx, 0, (80.0, 100.0), 1.2)
    assert w["cap"] * HD.width(w["text"]) <= PK.WRITE_WIDTH * 80.0 + 1e-9 and w["t"] > 1.2 + fx["drop"]
    assert w["dur"] == pytest.approx(sum(HD.lengths(HD.strokes(w["text"], w["seed"]))) / 10.0)
    assert PK.writing(fx, 1, (80.0, 100.0), 1.6) is None and PK.writing(fx, 3, (80.0, 100.0), 2.4) is None
    assert PK.writing(fx, 2, (80.0, 100.0), 2.0)["cap"] == pytest.approx(fx["purikura"]["letters"] * 100.0)


def test_every_letter_of_the_hand_sits_in_its_box_and_the_pen_writes_it_in_order():
    for c, (adv, strokes) in HD.glyphs().items():
        for p in strokes:
            assert p.ndim == 2 and len(p) >= 2, c
            assert p[:, 0].min() > -0.1 and p[:, 0].max() < adv + 0.1 and p[:, 1].min() > -0.1 and p[:, 1].max() < 1.42
    with pytest.raises(HD.HandError, match="does not write"):
        HD.check("ok\u00e9")
    s = HD.strokes("Hi 11.01", seed=4)
    assert np.concatenate(s)[:, 0].min() == pytest.approx(-np.concatenate(s)[:, 0].max(), abs=0.15)   # about its middle
    lens = HD.lengths(s)
    inked = [sum(HD.lengths(HD.written(s, u, lens))) for u in np.linspace(0.0, 1.0, 11)]
    assert inked[0] == 0.0 and all(b >= a for a, b in zip(inked, inked[1:]))
    assert inked[5] == pytest.approx(0.5 * sum(lens), rel=1e-6) and inked[-1] == pytest.approx(sum(lens))


@pytest.mark.parametrize("name, parts", [("heart", ["fill"]), ("flower", ["fill", "hole"]),
                                         ("blush", ["fill", "mark", "mark", "mark"]), ("arrow", ["line", "line"]),
                                         ("crown", ["fill", "mark", "fill", "fill", "fill"]),
                                         ("bubble", ["fill", "hole", "hole", "hole"]), ("bunny_ears", ["line"] * 4)])
def test_a_stamp_fills_the_closed_outlines_and_lightens_what_lies_across_them(name, parts):
    assert PK.stamp_parts(PC.shape(name)) == parts


def test_the_purikura_flash_goes_off_three_times_a_second_at_most():
    times = [1.0, 1.2, 1.4, 1.6, 1.8, 2.5]
    fx = FX.normalize(booth(times=times), MOON)
    kept = PK.flashing(fx, times)
    assert kept == [1.0, 1.2, 1.4, 2.5]
    assert all(sum(1 for k in kept if t <= k < t + 1.0) <= PK.FLASH_RATE for t in np.arange(0.0, 3.0, 0.01))
    assert PK.flashing({**fx, "purikura": None}, times) == times             # the plain booth: every capture


def clip_frames(size):
    w, h = size
    return {F0 + k: np.full((h, w, 3), [(k % 7) / 7.0, (k % 11) / 11.0, 0.5], np.float32) for k in range(130)}


def test_the_strip_comes_out_the_same_in_post_order_and_out_of_order():
    spec = booth(times=[1.2, 1.5, 1.8], purikura={"write": ["11.01", "ok"], "marks": [{"shape": "blush", "at": [0.3, 0.5]}]})
    data = {"project": {"duration": 4.0}, "output": [{"name": "1x1", "size": [216, 216]}],
            "shot": [{"name": "a", "from": 0.0, "to": 4.0}], "fx": [spec]}
    pl = TR.plan(data, FPS, F0, MOON)
    clip = clip_frames((216, 216))
    post = FF.FrameFx(pl["fx"], (216, 216), fps=FPS, frame0=F0, frame_of=clip.get, point_of=None, output="1x1")
    f1 = FX.frame_of(2.9, FPS, F0)
    outs = {f: post.frame(f, clip[f]) for f in range(pl["fx"][0]["f0"], f1 + 1)}
    look = FF.FrameFx(pl["fx"], (216, 216), fps=FPS, frame0=F0, frame_of=clip.get, point_of=None, output="1x1")
    for f in (f1, FX.frame_of(2.0, FPS, F0), FX.frame_of(1.6, FPS, F0)):
        assert np.array_equal(look.frame(f, clip[f]), outs[f])
    plain = TR.plan({**data, "fx": [{k: v for k, v in spec.items() if k not in ("look", "purikura")}]}, FPS, F0, MOON)
    other = FF.FrameFx(plain["fx"], (216, 216), fps=FPS, frame0=F0, frame_of=clip.get, point_of=None, output="1x1")
    assert not np.allclose(other.frame(f1, clip[f1]), outs[f1], atol=0.05)
