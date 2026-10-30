"""The paper world's post work and type: the activity book (`[[fx]] kind = "activity"`, mkmmd.core.activity,
mkmmd.activity: a page, tracing, connect-the-dots, colour by number stepping on the hits), editorial labels (`kind =
"label"`, mkmmd.core.labels, mkmmd.labels) and huge kinetic words (lyrics `layout = {kind = "bleed"}` and the kinetic
`yaw`, `pitch`, `persp`; mkmmd.core.wordtype, mkmmd.core.typefx). Made-up sprites, fonts of this machine, synthetic
timelines: no lyric text."""
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from mkmmd import framefx as FF
from mkmmd.core import activity as AC
from mkmmd.core import framefx as FX
from mkmmd.core import labels as LB
from mkmmd.core import palette as PAL
from mkmmd.core import typefx as TF
from mkmmd.core import wordtype as WT

DAWN = PAL.get("rose-pine-dawn")
FPS = 30
SIZE = (320, 180)


def system_font():
    for p in ("/usr/share/fonts/TTF/DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
              "/usr/share/fonts/TTF/DejaVuSansMono.ttf"):
        if Path(p).is_file():
            return p
    pytest.skip("no TrueType font on this machine")


@pytest.fixture
def sprite(tmp_path):
    """A two-colour disc with alpha: a red half and a green half."""
    n = 120
    y, x = np.mgrid[0:n, 0:n]
    a = ((x - 59.5) ** 2 + (y - 59.5) ** 2 < 55 ** 2).astype(np.uint8) * 255
    rgb = np.where((x < 60)[..., None], [200, 40, 40], [40, 160, 60]).astype(np.uint8)
    p = tmp_path / "disc.png"
    Image.fromarray(np.dstack([rgb, a])).save(p)
    return str(p)


def page(**spec):
    return {"kind": "activity", "from": 0.0, "to": 4.0, "times": [0.5, 1.0, 1.5, 2.0], **spec}


def fxfx(entries, sprite_path=None, font=None):
    fxs = FX.plan({"fx": entries}, FPS, 0, DAWN, last_frame=200)
    return FF.FrameFx(fxs, SIZE, fps=FPS, frame0=0, frame_of=lambda g: None,
                      point_of=lambda key, g: ((100.0, 60.0), 1.0), font_of=lambda ref: font or system_font(),
                      sprite_of=lambda refs: [sprite_path] if sprite_path else [])


def blank():
    return np.full((SIZE[1], SIZE[0], 3), 0.97, np.float32)


# ---------------------------------------------------------------- the activity book: its keys and timing
@pytest.mark.parametrize("spec, frag", [
    ({"items": []}, "`items` is required"),
    ({"items": [{"kind": "colour"}]}, "kind = trace"),
    ({"items": [{"kind": "trace"}]}, "give one of `shape`"),
    ({"items": [{"kind": "trace", "shape": "star", "points": [[0, 0], [1, 1]]}]}, "give one of"),
    ({"items": [{"kind": "paint", "points": [[0, 0], [1, 1]]}]}, "needs regions"),
    ({"items": [{"kind": "dots", "shape": "figure"}]}, "none of"),
    ({"items": [{"kind": "trace", "shape": "figure"}]}, "give the entry its `objects`"),
    ({"objects": "Rin_*", "items": [{"kind": "trace", "shape": "star"}]}, "no item does"),
    ({"items": [{"kind": "trace", "shape": "star", "eyes": True}]}, "a sprite's"),
    ({"items": [{"kind": "dots", "shape": "heart", "count": 2}]}, "count"),
    ({"items": [{"kind": "trace", "shape": "star", "t": 9.0}]}, "outside the page's span"),
    ({"picture": "overlay", "items": [{"kind": "trace", "shape": "star"}]}, "picture"),
    ({"rule": {"pitch": 0}, "items": [{"kind": "trace", "shape": "star"}]}, "rule pitch"),
])
def test_bad_pages_say_what_is_wrong(spec, frag):
    with pytest.raises(FX.FxError, match=frag):
        FX.normalize(page(**spec), DAWN)


def test_an_item_steps_once_a_hit_from_its_own_time_eased_and_never_past_its_steps():
    it = FX.normalize(page(items=[{"kind": "trace", "shape": "star", "t": 0.9}]), DAWN)["items"][0]
    times = [0.5, 1.0, 1.5, 2.0]
    assert AC.item_hits(it, times) == [1.0, 1.5, 2.0] and AC.steps(it, times) == 3
    assert AC.progress(it, 0.95, times, 0.2) == 0.0
    assert 0.0 < AC.progress(it, 1.1, times, 0.2) < 1.0                       # easing in after the hit
    assert AC.progress(it, 1.3, times, 0.2) == pytest.approx(1.0)
    assert AC.progress(it, 9.0, times, 0.2) == pytest.approx(3.0)
    assert AC.per_step({**it, "per": None}, 12, times) == 4 and AC.per_step({**it, "per": 2}, 12, times) == 2


def test_outlines_start_at_the_top_run_clockwise_and_resample_evenly():
    ccw = [(0.0, 1.0), (1.0, 1.0), (1.0, 0.0), (0.0, 0.0)]                   # counter-clockwise on screen (y down)
    p = AC.start_top(ccw)
    assert p[0] == (0.0, 0.0) and p[1] == (1.0, 0.0)                          # from the top left, to the right
    q = AC.resample(p, 8, closed=True)
    assert len(q) == 8 and q[0] == (0.0, 0.0) and q[2] == pytest.approx((1.0, 0.0))
    h = AC.head(p, 0.5, closed=True)
    assert h[-1] == pytest.approx((1.0, 1.0))                                # half way round a square


def test_a_page_reads_her_figure_only_when_a_trace_follows_her():
    fx = FX.normalize(page(objects="Rin_*", items=[{"kind": "trace", "shape": "figure"}]), DAWN)
    assert FX.uses_figure(fx) and not FX.uses_figure(FX.normalize(page(items=[{"kind": "trace", "shape": "star"}]),
                                                                  DAWN))


# ---------------------------------------------------------------- the activity book: drawn
def _ink(a, b):
    return float((np.abs(a - b).sum(-1) > 0.15).mean())


def test_the_page_is_cream_paper_with_rules_and_the_picture_drawn_on_it():
    ff = fxfx([page(items=[{"kind": "trace", "shape": "star", "at": [0.2, 0.3], "size": 0.3}])])
    img = blank()
    img[80:100, 150:170] = 0.1                                               # a dark mark on the drawing's paper
    out = ff.frame(1, img)
    assert out[90, 160].sum() < 0.6                                          # the drawing shows on the page
    paper = np.asarray(FX.normalize(page(items=[{"kind": "trace", "shape": "star"}]), DAWN)["paper"])
    assert np.abs(out[40:60, 250:300].mean((0, 1)) - paper).max() < 0.08    # cream, wherever nothing is drawn


def test_a_trace_is_crayoned_further_on_each_hit():
    ff = fxfx([page(rule=False, margin=False, items=[{"kind": "trace", "shape": "circle", "at": [0.5, 0.5],
                                                       "size": 0.6, "color": "#FF0000", "tip": False}])])
    red = [float((ff.frame(f, blank())[..., 0] - ff.frame(f, blank())[..., 2] > 0.3).mean()) for f in (10, 25, 40, 70)]
    assert red[0] < red[1] < red[2] < red[3]                                  # 0.33 s, 0.83, 1.33, 2.33: 0..4 hits


def test_connect_the_dots_join_on_the_hits_and_reveal_the_sprite(sprite):
    fx = page(rule=False, margin=False, items=[{"kind": "dots", "sprites": "disc.png", "at": [0.5, 0.5],
                                                "size": 0.7, "count": 8, "color": "#0000FF"}])
    ff = fxfx([fx], sprite)
    early, late = ff.frame(20, blank()), ff.frame(110, blank())              # 0.67 s: one hit; 3.67 s: all, revealed
    assert _ink(early, blank()) > 0.005                                       # printed dots and numbers from the start
    green = lambda o: float(((o[..., 1] - o[..., 0]) > 0.3).mean())          # noqa: E731
    assert green(early) < 0.01 and green(late) > 0.1                         # the sprite appears once they are joined


def test_colour_by_number_cuts_a_sprite_by_its_colours_and_fills_a_region_a_hit(sprite):
    from mkmmd import activity as ACF
    from mkmmd import mandala as MDF
    tile = MDF.sprite_tile(sprite, 100, 0.0)
    lab, regs = ACF.regions(tile.rgb, tile.a, 2)
    assert sorted(r["number"] for r in regs) == [1, 2] and lab.max() == 2
    ff = fxfx([page(rule=False, margin=False, items=[{"kind": "paint", "sprites": "disc.png", "at": [0.5, 0.5],
                                                      "size": 0.8, "colors": 2, "legend": False}])], sprite)
    reds = [float(((ff.frame(f, blank())[..., 0] - ff.frame(f, blank())[..., 1]) > 0.25).mean()) for f in (10, 40)]
    assert reds[1] > reds[0] + 0.02                                           # a region coloured in after its hit


# ---------------------------------------------------------------- labels
def label(**spec):
    return {"kind": "label", "from": 0.0, "to": 4.0, "labels": [{"text": "1967", "at": [0.5, 0.5]}], **spec}


@pytest.mark.parametrize("spec, frag", [
    ({"labels": []}, "`labels` is required"),
    ({"labels": [{"at": [0.5, 0.5]}]}, "short line"),
    ({"labels": [{"text": "x" * 41, "at": [0.5, 0.5]}]}, "short line"),
    ({"labels": [{"text": "A"}]}, "`at` is required"),
    ({"labels": [{"text": "A", "at": [0.5, 0.5], "place": [0, 0], "offset": [0, 0]}]}, "`place`"),
    ({"labels": [{"text": "A", "at": [0.5, 0.5], "align": "up"}]}, "align"),
    ({"leader": {"dash": 3}}, "leader"),
])
def test_bad_labels_say_what_is_wrong(spec, frag):
    with pytest.raises(FX.FxError, match=frag):
        FX.normalize(label(**spec), DAWN)


def test_a_label_at_a_scene_point_asks_for_it_projected():
    fx = FX.plan({"fx": [label(labels=[{"text": "A", "at": {"point": "obj('x').loc"}}, {"text": "B", "at": [0.1, 0.1]}])]},
                 FPS, 0, DAWN, last_frame=200)[0]
    assert FX.points(fx) == [(LB.point_key(fx, 0), "obj('x').loc")]


def test_a_label_types_out_then_draws_its_leader_to_its_point():
    lb = FX.normalize(label(labels=[{"text": "SIDE A", "at": [0.5, 0.5], "t": 1.0}]), DAWN)["labels"][0]
    assert LB.shown(lb, 0.9, 0.25) == 0 and 0 < LB.shown(lb, 1.1, 0.25) < 6 and LB.shown(lb, 1.3, 0.25) == 6
    assert LB.drawn(lb, 1.2, 0.25, 0.2) == 0.0 and LB.drawn(lb, 1.6, 0.25, 0.2) == 1.0


def test_a_label_is_drawn_beside_its_point_kept_inside_the_frame_with_a_dot_at_the_point():
    ff = fxfx([label(color="#000000", labels=[{"text": "TRACK 14", "at": [0.95, 0.5], "offset": [0.1, 0.0],
                                              "align": "left"}])])
    out = ff.frame(60, blank())
    dark = out.sum(-1) < 1.5
    ys, xs = np.nonzero(dark)
    assert xs.max() < SIZE[0] - 2                                             # slid in from past the right edge
    assert dark[88:92, 302:306].any()                                         # the dot at its point


# ---------------------------------------------------------------- huge kinetic words
def test_kinetic_yaw_pitch_and_persp_are_keys_of_a_word():
    k = TF.kinetic_spec({"kinetic": {"yaw": [[0.0, -20.0]], "pitch": [[0.0, 8.0]], "persp": 3}})
    assert k["yaw"] == [(0.0, -20.0)] and k["persp"] == 3.0
    with pytest.raises(Exception, match="give one"):
        TF.kinetic_spec({"kinetic": {"persp": 3}})
    with pytest.raises(Exception, match="0.5 .. 1000"):
        TF.kinetic_spec({"kinetic": {"yaw": [[0.0, 1.0]], "persp": 0}})


def test_bleed_sets_each_word_to_run_off_the_panel_in_its_band():
    opts = WT.bleed_options({"rows": 2, "align": ["left", "right", "center"], "yaw": [-20, 10], "pitch": 5})
    boxes = [(4.0, 0.0, 0.7), (6.0, 0.0, 0.7), (1.0, -0.2, 0.7)]
    out = WT.layout_bleed(boxes, 0.7, (1.6, 1.0), opts)
    ink0 = 4.0 * out[0]["size"] / 0.7
    assert ink0 == pytest.approx(1.2 * 1.6)                                   # wider than the panel: the frame crops it
    assert out[0]["offset"][0] - ink0 / 2 == pytest.approx(-0.8 - 0.04 * ink0)   # just off the left edge
    ink1 = 6.0 * out[1]["size"] / 0.7
    assert out[1]["offset"][0] + ink1 / 2 == pytest.approx(0.8 + 0.04 * ink1)    # just off the right edge
    assert out[0]["offset"][1] == pytest.approx(0.25) and out[1]["offset"][1] == pytest.approx(-0.25)   # two bands
    assert out[2]["size"] == pytest.approx(0.8 * 0.5)                         # a short word: capped by its band
    assert [o["yaw"] for o in out] == [-20.0, 10.0, -20.0] and out[2]["pitch"] == 5.0


@pytest.mark.parametrize("lay, frag", [({"rows": 0}, "rows"), ({"align": "up"}, "align"), ({"width": 0}, "width"),
                                       ({"at": 3}, "at"), ({"yaw": "x"}, "yaw")])
def test_bad_bleeds_say_what_is_wrong(lay, frag):
    with pytest.raises(WT.LyricsError, match=frag):
        WT.bleed_options(lay)


def test_a_bleed_word_swings_into_its_tilt():
    k = WT.depth_keys({"yaw": -20.0, "pitch": 6.0, "persp": 3.0, "swing": 30.0}, 1.0, 30)
    assert list(k["yaw"][0]) == [1.0, 10.0] and k["yaw"][-1][1] == pytest.approx(-20.0) and k["persp"] == 3.0
    assert WT.depth_keys({"yaw": -20.0, "pitch": 6.0, "persp": 3.0, "swing": 0.0}, 1.0, 30)["yaw"] == [[1.0, -20.0]]


def _timeline():
    words = [{"start": 0.5 + 0.6 * i, "end": 0.9 + 0.6 * i, "text": "ab" * (i + 1)} for i in range(3)]
    return {"lines": [{"words": words}], "beats": [0.5, 1.0, 1.5, 2.0], "vocal_db": [-20.0] * 120}


def test_a_bleed_entry_gives_every_word_its_own_size_tilt_and_no_fit():
    measure = lambda s, font=None, tracking=None: {"w": 0.6 * len(s), "y0": 0.0, "y1": 0.7, "cap": 0.7}  # noqa: E731
    entry = {"name": "huge", "fit": 0.9, "screen": True,
             "lyrics": {"timeline": "t.json", "line": 1, "style": "slam", "recycle": True,
                        "layout": {"kind": "bleed", "yaw": [-20, 15], "pitch": 4}}}
    specs, summary = WT.expand(entry, _timeline(), fps=30, panel=(1.6, 0.92), measure=measure)
    assert summary["layout"] == "bleed" and len(specs) == 3
    assert all("fit" not in s and s["size"] > 0 for s in specs)
    assert [s["kinetic"]["yaw"][0][1] for s in specs] == [-20.0, 15.0, -20.0]
    assert all(s["kinetic"]["persp"] == WT.BLEED["persp"] for s in specs)
