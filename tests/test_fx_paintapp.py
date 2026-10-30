"""The toy paint app (mkmmd.core.paintapp: the entry checked, its chrome laid out per output, what its tools do and when,
the cursor's way, the stamps' and fills' animation, the eraser's confetti and strokes, flash safety; mkmmd.paintapp: the
chrome, stamps, fills, undo and both erasers drawn over synthetic cuts) and the layers and credits it needs."""
import math

import numpy as np
import pytest
from PIL import Image

from mkmmd import assets as A
from mkmmd import framefx as FF
from mkmmd import mandala as MDF
from mkmmd import paintapp as PAF
from mkmmd.cli.assets import project_credits
from mkmmd.core import framefx as FX
from mkmmd.core import paintapp as PA
from mkmmd.core import palette as PAL
from mkmmd.core import transition as TR

FPS, F0 = 30, 1
DAWN = PAL.get("rose-pine-dawn")
W, H = 320, 180
TEAL, CREAM, PINK = (0.2, 0.55, 0.6), (0.95, 0.9, 0.8), (0.9, 0.5, 0.6)


def project(fx=(), size=(W, H), cut=2.0):
    return {"project": {"duration": 4.0}, "output": [{"name": "o", "size": list(size)}],
            "shot": [{"name": "a", "from": 0.0, "to": cut}, {"name": "b", "from": cut, "to": 4.0}], "fx": list(fx)}


def plan(fx=(), size=(W, H)):
    return TR.plan(project(fx, size), FPS, F0, DAWN)


def app(**kw):
    return {"kind": "paintapp", "from": 0.0, "to": 4.0, **kw}


def entry(spec, size=(W, H)):
    return plan([spec], size)["fx"][0]


def frame_at(t):
    return F0 + int(round(t * FPS))


def disc_sprite(path, colour=(255, 0, 0), size=40, box=60):
    """A PNG `box` px square, clear but for a disc `size` px across in its middle."""
    yy, xx = np.mgrid[0:box, 0:box].astype(np.float32)
    a = np.clip(size / 2.0 - np.hypot(xx - (box - 1) / 2.0, yy - (box - 1) / 2.0) + 0.5, 0.0, 1.0)
    rgba = np.dstack([np.full((box, box), c, np.float32) for c in colour] + [a * 255.0]).astype(np.uint8)
    Image.fromarray(rgba, "RGBA").save(path)
    return str(path)


def flat(colour, size=(W, H)):
    out = np.empty((size[1], size[0], 3), np.float32)
    out[:] = colour
    return out


def two_shots(size=(W, H)):
    """The cut's frames: teal with a pink square before the cut at 2 s, cream after it."""
    a = flat(TEAL, size)
    a[size[1] // 3:2 * size[1] // 3, size[0] // 3:size[0] // 2] = PINK
    b = flat(CREAM, size)
    return lambda g: a if g < frame_at(2.0) else b


def runner(pl, size=(W, H), frames=None, hits=None, point_of=None):
    return FF.FrameFx(pl["fx"], size, fps=FPS, frame0=F0, frame_of=frames or two_shots(size), point_of=point_of,
                      hits=hits, sprite_of=MDF.sprite_source())


# ---------------------------------------------------------------- the entry
@pytest.mark.parametrize("spec", [
    app(toolbar="middle"),
    app(colour="love"),
    app(swatches=["love"] * 13),
    app(ui="yes"),
    app(cursor={"kind": "mouse"}),
    app(cursor={"keys": [{"t": 1.0}]}),
    app(cursor={"keys": [{"t": 1.0, "at": [0.5, 0.5]}, {"t": 1.0, "at": [0.2, 0.2]}]}),
    app(stamps={"lane": "kick"}),
    app(stamps={"sprites": "a.png"}),
    app(stamps={"sprites": "a.png", "times": [1.0], "lane": "kick"}),
    app(stamps={"sprites": "a.png", "times": [1.0], "size": 2.0}),
    app(stamps={"sprites": "a.png", "times": [1.0], "glow": 1}),
    app(stamps={"sprites": "a.png", "times": [1.0], "eyes": "big"}),
    app(bucket={"times": [1.0]}),
    app(bucket={"times": [1.0], "at": [0.5, 0.5], "grow": 0.05}),
    app(bucket={"times": [1.0], "at": [[0.5, 0.5], {"pt": "x"}]}),
    app(eraser={"dur": 0.5}),
    app(eraser={"at": 1.0, "mode": "fade"}),
    app(eraser={"at": 1.0, "dur": 0.1}),
    app(eraser={"at": 3.8, "dur": 0.5}),
    app(eraser=[{"at": 1.0}, {"at": 1.3}]),
    app(undo={"times": [1.0]}),
])
def test_a_bad_entry_is_an_error(spec):
    with pytest.raises(TR.TransitionError):
        plan([spec])


def test_an_eraser_needs_a_frame_before_it_to_hold():
    with pytest.raises(TR.TransitionError):
        plan([app(eraser={"at": 0.0})])
    assert entry(app(eraser={"at": 0.1}))["eraser"][0]["frame"] == frame_at(0.1)


def test_the_entry_fills_in_its_defaults_and_resolves_its_colours():
    fx = entry(app())
    assert fx["toolbar"] == "auto" and fx["ui"] is True and fx["eraser"] == [] and fx["stamps"] is None
    assert fx["cursor"]["kind"] == "hand" and fx["cursor"]["keys"] == []
    assert len(fx["swatches"]) == len(PA.SWATCHES) and all(len(c) == 3 for c in fx["swatches"])
    assert entry(app(cursor=False))["cursor"] is None
    fx = entry(app(stamps={"sprites": {"tag": "clover"}, "lane": "kick", "at": [0.3, 0.4], "edge": False}))
    assert fx["stamps"]["refs"] == [{"tag": ["clover"]}] and fx["stamps"]["at"] == [[0.3, 0.4]]
    assert fx["stamps"]["lane"] == "kick" and fx["stamps"]["times"] is None and fx["stamps"]["edge"] is None


# ---------------------------------------------------------------- the chrome
@pytest.mark.parametrize("size, side", [((1920, 1080), "left"), ((1080, 1080), "left"), ((1080, 1920), "bottom")])
def test_the_toolbar_stands_at_the_bottom_of_a_tall_frame_and_the_left_of_others(size, side):
    fx = entry(app(), size)
    lay = PA.layout(fx, size)
    assert lay["side"] == side
    w, h = size

    def inside(b, box):
        return box[0] - 1e-6 <= b[0] and box[1] - 1e-6 <= b[1] and b[2] <= box[2] + 1e-6 and b[3] <= box[3] + 1e-6

    def apart(a, b):
        return a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1]
    frame = (0.0, 0.0, w, h)
    assert inside(lay["canvas"], frame) and inside(lay["toolbar"], frame) and inside(lay["strip"], frame)
    assert apart(lay["canvas"], lay["toolbar"]) and apart(lay["canvas"], lay["strip"])
    assert apart(lay["toolbar"], lay["strip"])
    assert [t for t, _ in lay["buttons"]] == list(PA.TOOLS)
    assert all(inside(b, lay["toolbar"]) for _, b in lay["buttons"])
    assert all(inside((x - r + 0.5, y - r + 0.5, x + r + 0.5, y + r + 0.5), lay["strip"]) for x, y, r in lay["swatches"])
    x0, y0, x1, y1 = lay["canvas"]
    assert (x1 - x0) * (y1 - y0) > 0.6 * w * h                     # the picture keeps most of the frame
    ys = [b[1] for _, b in lay["buttons"]]
    xs = [b[0] for _, b in lay["buttons"]]
    assert (ys == sorted(ys) and len(set(xs)) == 1) if side == "left" else (xs == sorted(xs) and len(set(ys)) == 1)


def test_a_toolbar_too_long_for_its_side_shrinks_to_fit():
    fx = entry(app(toolbar="right", scale=2.0), (1920, 1080))
    lay = PA.layout(fx, (1920, 1080))
    assert lay["side"] == "right" and lay["toolbar"][2] <= 1920 and lay["toolbar"][0] > lay["canvas"][2]
    assert lay["toolbar"][1] >= lay["canvas"][1] - 1e-6 and lay["toolbar"][3] <= lay["canvas"][3] + 1e-6


# ---------------------------------------------------------------- what happens, and when
def tools(stamps=(), bucket=(), undo=()):
    return {"stamps": list(stamps), "bucket": list(bucket), "undo": list(undo)}


def test_undo_takes_back_the_last_stamp_or_fill_still_on_and_an_eraser_clears_the_canvas():
    fx = entry(app(stamps={"sprites": "a.png", "times": [0.5, 1.0, 2.6]}, bucket={"times": [0.8], "at": [0.5, 0.5]},
                   undo={"times": [1.5, 1.6, 1.7, 1.8, 3.0]}, eraser={"at": 2.0}))
    hist = PA.history(fx, tools([0.5, 1.0, 2.6], [0.8], [1.5, 1.6, 3.0]))
    by = {(it["kind"], it["j"]): it for it in hist["items"]}
    assert (by["stamp", 1]["how"], by["stamp", 1]["end"]) == ("undo", 1.5)          # the last one first
    assert (by["fill", 0]["how"], by["fill", 0]["end"]) == ("undo", 1.6)
    assert (by["stamp", 0]["how"], by["stamp", 0]["end"]) == ("erase", 2.0)         # wiped with the picture
    assert (by["stamp", 2]["how"], by["stamp", 2]["end"]) == ("undo", 3.0)
    assert [a["kind"] for a in hist["actions"]] == ["stamp", "fill", "stamp", "undo", "undo", "erase", "stamp", "undo"]
    empty = PA.history(fx, tools(undo=[1.0]))                       # nothing to take back: it only clicks
    assert empty["items"] == [] and empty["actions"] == [{"t": 1.0, "kind": "undo", "j": 0}, {"t": 2.0, "kind": "erase",
                                                                                              "j": 0}]


def test_a_stamp_with_a_life_fades_out_and_is_not_taken_back_once_gone():
    fx = entry(app(stamps={"sprites": "a.png", "times": [0.5, 1.0], "life": 0.4}, undo={"times": [1.2]}))
    hist = PA.history(fx, tools([0.5, 1.0], undo=[1.2]))
    first, second = hist["items"]
    assert (first["how"], first["end"]) == ("fade", 0.9) and (second["how"], second["end"]) == ("undo", 1.2)
    assert PA.stamp_state(fx, first, 0.8)[2] == pytest.approx(0.4)          # its fade: a quarter second
    assert PA.stamp_state(fx, first, 0.95) is None


def test_the_cursor_picks_the_tool_and_the_colour_on_its_way_when_there_is_time():
    fx = entry(app(stamps={"sprites": "a.png", "times": [1.0, 1.2]}, bucket={"times": [3.0], "at": [0.5, 0.5],
                                                                              "color": "foam"}))
    hist = PA.history(fx, tools([1.0, 1.2], [3.0]))
    clicks, tools_, picked = PA.clicks(fx, hist)
    ons = [c["on"] for c in clicks]
    foam = PA.swatch_of(fx, fx["bucket"]["color"][0])
    assert ons == [("tool", "stamp"), ("stamp", 0), ("stamp", 1), ("tool", "bucket"), ("swatch", foam), ("fill", 0)]
    assert [c["t"] for c in clicks][3:5] == pytest.approx([1.2 + 1.8 / 3, 1.2 + 3.6 / 3])
    assert PA.in_force(tools_, 0.1) == "pencil" and PA.in_force(tools_, 1.1) == "stamp"
    assert PA.in_force(tools_, 2.0) == "bucket" and PA.in_force(picked, 2.9) == foam
    quick = entry(app(stamps={"sprites": "a.png", "times": [0.3]}))      # no time to pick: it changes as it acts
    clicks, tools_, _ = PA.clicks(quick, PA.history(quick, tools([0.3])))
    assert [c["on"] for c in clicks] == [("stamp", 0)] and PA.in_force(tools_, 0.3) == "stamp"
    bare = entry(app(ui=False, stamps={"sprites": "a.png", "times": [3.0]}))
    assert [c["on"] for c in PA.clicks(bare, PA.history(bare, tools([3.0])))[0]] == [("stamp", 0)]


def test_the_cursor_waits_glides_on_an_arc_and_arrives_on_time():
    stops = [(0.0, (0.0, 0.0)), (1.0, (100.0, 0.0)), (1.2, (100.0, 100.0))]
    assert PA.glide(stops, -1.0, 0.3) == (0.0, 0.0) and PA.glide(stops, 0.69, 0.3) == (0.0, 0.0)
    assert PA.glide(stops, 1.0, 0.3) == pytest.approx((100.0, 0.0)) and PA.glide(stops, 5.0, 0.3) == (100.0, 100.0)
    mid = PA.glide(stops, 0.85, 0.3)
    assert mid[0] == pytest.approx(50.0) and abs(mid[1]) > 5.0                     # half way, off the straight line
    assert PA.glide(stops, 1.04, 0.3) == pytest.approx((100.0, 0.0))                # the short leg: 0.8 of its time
    assert PA.glide(stops, 1.2, 0.3) == pytest.approx((100.0, 100.0))
    assert PA.press([1.0], 1.0) == pytest.approx(1.0) and PA.press([1.0], 0.5) == 0.0
    assert min(PA.press([1.0], 1.0 + s) for s in np.linspace(0.0, PA.RELEASE, 40)) < 0.0   # it springs past rest
    assert PA.ripples([1.0], 1.1) == pytest.approx([0.1 / PA.RIPPLE]) and PA.ripples([1.0], 2.0) == []


def test_a_stamp_pops_in_overshooting_and_is_taken_back_in_reverse():
    fx = entry(app(stamps={"sprites": "a.png", "times": [1.0]}, undo={"times": [2.0]}))
    it = PA.history(fx, tools([1.0], undo=[2.0]))["items"][0]
    assert PA.stamp_state(fx, it, 0.99) is None
    pop = [PA.stamp_state(fx, it, 1.0 + s)[0] for s in np.linspace(0.0, 0.3, 31)]
    assert pop[0] == pytest.approx(0.0) and max(pop) == pytest.approx(1.0 + PA.POP_OVER, abs=0.01)
    assert pop[-1] == pytest.approx(1.0) and PA.stamp_state(fx, it, 1.9)[:3] == pytest.approx((1.0, 0.0, 1.0))
    back = [PA.stamp_state(fx, it, 2.0 + s)[0] for s in np.linspace(0.0, 0.29, 30)]
    assert back[0] == pytest.approx(1.0) and max(back) > 1.1 and back[-1] < 0.25 and back[-1] < back[-2]
    assert PA.stamp_state(fx, it, 2.31) is None


def test_a_fill_spreads_out_over_its_grow_and_draws_back_when_taken_back():
    fx = entry(app(bucket={"times": [1.0], "at": [0.5, 0.5], "grow": 0.4}, undo={"times": [2.0]}))
    it = PA.history(fx, tools(bucket=[1.0], undo=[2.0]))["items"][0]
    spread = [PA.fill_state(fx, it, 1.0 + s)[0] for s in (0.0, 0.1, 0.2, 0.4, 0.9)]
    assert spread[0] == 0.0 and spread == sorted(spread) and spread[-2:] == [1.0, 1.0]
    assert PA.fill_state(fx, it, 2.15)[0] < 1.0 and PA.fill_state(fx, it, 2.31) is None


def test_more_than_three_fills_or_erasers_in_a_second_are_not_flash_safe():
    fx = entry(app(bucket={"times": [0.5], "at": [0.5, 0.5]}, eraser={"at": 2.0}))
    PA.flash_check(fx, PA.history(fx, tools(bucket=[0.5, 0.9, 1.3])))
    with pytest.raises(FX.FxError):
        PA.flash_check(fx, PA.history(fx, tools(bucket=[1.2, 1.5, 1.8])))
    with pytest.raises(FX.FxError):
        run = runner(plan([app(bucket={"times": [1.0, 1.2, 1.4, 1.6], "at": [0.5, 0.5]})]))
        run.frame(frame_at(1.0), flat(TEAL))


# ---------------------------------------------------------------- the eraser's geometry
@pytest.mark.parametrize("size", [(320, 180), (180, 320), (200, 200)])
def test_the_confetti_tiles_the_frame_and_flies_away_by_the_end(size):
    polys = PA.confetti(size, 9, 1, (0,))
    area = sum(0.5 * abs(float(np.dot(p[:, 0], np.roll(p[:, 1], -1)) - np.dot(p[:, 1], np.roll(p[:, 0], -1))))
               for p in polys)
    assert area == pytest.approx(size[0] * size[1], rel=1e-6)
    assert all(p[:, 0].min() >= 0 and p[:, 0].max() <= size[0] and p[:, 1].min() >= 0 and p[:, 1].max() <= size[1]
               for p in polys)
    fls = PA.flights(polys, (0.5 * size[0], 0.5 * size[1]), size, 0.5, 1, (0,))
    assert min(f["t0"] for f in fls) >= 0.0 and max(f["t0"] for f in fls) <= (PA.WAVE + PA.LAG) * 0.5 + 1e-9
    near = min(fls, key=lambda f: math.dist(f["c"], (0.5 * size[0], 0.5 * size[1])))
    far = max(fls, key=lambda f: math.dist(f["c"], (0.5 * size[0], 0.5 * size[1])))
    assert near["t0"] < far["t0"]                                     # it blows out from the centre
    assert PA.flight_pose(far, -0.01, 0.0, min(size), 0.5) is None
    assert all(PA.flight_pose(f, 0.5 - f["t0"], 1.0, min(size), 0.5) is None for f in fls)
    assert PA.confetti(size, 9, 1, (0,))[3].tolist() == polys[3].tolist()       # seeded
    assert PA.confetti(size, 9, 2, (0,))[3].tolist() != polys[3].tolist()


@pytest.mark.parametrize("size", [(320, 180), (180, 320)])
def test_the_wipe_strokes_cover_the_whole_frame_by_the_end(size):
    import cv2
    width = 0.3 * min(size)
    path = PA.wipe_path(size, width)
    mask = np.zeros((size[1], size[0]), np.uint8)
    cv2.polylines(mask, [np.rint(path * 16).astype(np.int32)], False, 255, int(round(width)), cv2.LINE_AA, 4)
    assert mask.min() >= 128
    half = np.zeros_like(mask)
    cv2.polylines(half, [np.rint(PA.upto(path, 0.5) * 16).astype(np.int32)], False, 255, int(round(width)),
                  cv2.LINE_AA, 4)
    assert 0.3 < (half > 128).mean() < 0.75
    assert PA.along(path, 0.0) == pytest.approx(tuple(path[0])) and PA.along(path, 1.0) == pytest.approx(tuple(path[-1]))


def test_an_eraser_reads_the_frame_before_it_and_the_points_its_fills_read_there():
    pl = plan([app(bucket={"times": [0.5], "at": {"point": "(0, 1, 1)"}}, eraser={"at": 2.0, "dur": 0.5})])
    fx = pl["fx"][0]
    g = frame_at(2.0) - 1
    assert FX.holds(fx) and FX.points(fx) == [("x0b0", "(0, 1, 1)")]
    assert FX.sources(fx, frame_at(2.0), F0) == [g] and FX.sources(fx, frame_at(2.0) + 14, F0) == [g]
    assert FX.sources(fx, frame_at(2.0) + 15, F0) == [] and FX.sources(fx, g, F0) == []
    items = TR.fx_frames(pl, [frame_at(2.2)])[g]
    assert {"kind": "plate", "shot": "a"} in items
    assert {"kind": "point", "key": "x0b0", "expr": "(0, 1, 1)", "shot": "a"} in items


# ---------------------------------------------------------------- drawn
def test_the_chrome_frames_the_canvas_and_leaves_the_picture_inside_it():
    fx_spec = app(cursor=False)
    run = runner(plan([fx_spec]))
    img = flat(TEAL)
    out = run.frame(frame_at(1.0), img)
    lay = PA.layout(entry(fx_spec), (W, H))
    x0, y0, x1, y1 = (int(v) for v in lay["canvas"])
    inner = out[y0 + 8:y1 - 8, x0 + 8:x1 - 8]
    assert np.allclose(inner, TEAL, atol=1e-5)                                   # the picture, as it was
    assert np.allclose(out[1, W - 2], entry(fx_spec)["border"], atol=0.02)        # the border round it
    tx0, ty0, tx1, ty1 = lay["buttons"][0][1]
    assert not np.allclose(out[int(ty0):int(ty1), int(tx0):int(tx1)], TEAL, atol=0.05)
    assert np.array_equal(run.frame(frame_at(1.0), img), out)                    # the same frame comes out the same
    assert np.array_equal(runner(plan([app(strength=0.0)])).frame(frame_at(1.0), img), img)


def test_stamps_land_on_their_hits_where_the_cursor_clicks_and_undo_takes_one_back(tmp_path):
    sprite = disc_sprite(tmp_path / "red.png")
    stamps = {"sprites": sprite, "lane": "kick", "at": [[0.3, 0.5], [0.7, 0.5]], "size": 0.3, "sparkle": False,
              "shadow": False, "angle": 0.0}
    hits = {"kick": [{"t": 1.0, "db": 0.0}, {"t": 1.5, "db": -2.0}, {"t": 3.9, "db": -30}]}
    run = runner(plan([app(ui=False, cursor=False, stamps=stamps, undo={"times": [2.0]})]), hits=hits)
    teal = flat(TEAL)
    a, b = (int(0.3 * W - 0.5), int(0.5 * H - 0.5)), (int(0.7 * W - 0.5), int(0.5 * H - 0.5))
    before = run.frame(frame_at(0.9), teal)
    assert np.allclose(before[a[1], a[0]], TEAL, atol=0.01)
    after = run.frame(frame_at(1.8), teal)
    assert np.allclose(after[a[1], a[0]], (1.0, 0.0, 0.0), atol=0.02)
    assert np.allclose(after[b[1], b[0]], (1.0, 0.0, 0.0), atol=0.02)
    gone = run.frame(frame_at(2.4), teal)                                       # undone: the last one, at b
    assert np.allclose(gone[a[1], a[0]], (1.0, 0.0, 0.0), atol=0.02) and np.allclose(gone[b[1], b[0]], TEAL, atol=0.01)
    hand = runner(plan([app(ui=False, stamps=stamps)]), hits=hits).frame(frame_at(1.5), teal)
    bare = runner(plan([app(ui=False, cursor=False, stamps=stamps)]), hits=hits).frame(frame_at(1.5), teal)
    near = (slice(b[1] - 4, b[1] + 30), slice(b[0] - 4, b[0] + 30))           # the cursor clicks where it lands
    assert not np.allclose(hand[near], bare[near], atol=0.05)
    assert np.allclose(hand[:, :W // 2], bare[:, :W // 2], atol=1e-6)


def test_a_fill_floods_the_colour_region_round_its_point_and_keeps_out_of_others():
    spec = app(ui=False, cursor=False, bucket={"times": [1.0], "at": [0.1, 0.1], "color": "gold", "grow": 0.2})
    run = runner(plan([spec]))
    img = two_shots()(frame_at(1.0))
    gold = entry(spec)["bucket"]["color"][0]
    out = run.frame(frame_at(1.5), img)
    assert np.allclose(out[5, 5], gold, atol=0.02) and np.allclose(out[H - 5, W - 5], gold, atol=0.02)
    assert np.allclose(out[H // 2, int(W * 0.42)], PINK, atol=0.02)             # the square is another region
    early = run.frame(frame_at(1.0) + 2, img)                                   # spreading from its point
    assert np.allclose(early[5, 5], gold, atol=0.02) and np.allclose(early[H - 5, W - 5], TEAL, atol=0.02)


@pytest.mark.parametrize("mode", ["confetti", "wipe"])
def test_the_eraser_takes_the_held_picture_off_the_incoming_shot(mode):
    spec = app(ui=False, cursor=False, eraser={"at": 2.0, "dur": 0.5, "mode": mode})
    frames = two_shots()
    run = runner(plan([spec]), frames=frames)
    g = frame_at(2.0)

    def share_of_old(out):
        return float(np.mean(np.abs(out - np.asarray(CREAM, np.float32)).max(-1) > 0.05))
    first = run.frame(g, frames(g))
    assert share_of_old(first) > 0.9                                             # the outgoing picture, held
    mid = run.frame(g + 7, frames(g + 7))
    assert 0.05 < share_of_old(mid) < 0.95
    assert np.allclose(run.frame(g + 15, frames(g + 15)), CREAM, atol=1e-5)      # gone: the incoming shot
    fresh = runner(plan([spec]), frames=frames)                                  # out of order, as mk look asks
    assert np.allclose(fresh.frame(g + 7, frames(g + 7)), mid, atol=1e-5)


def test_the_held_picture_carries_the_paint_on_it_and_the_paint_after_the_eraser_stays_live(tmp_path):
    sprite = disc_sprite(tmp_path / "red.png")
    spec = app(ui=False, cursor=False, eraser={"at": 2.0, "dur": 0.5, "mode": "wipe"},
               stamps={"sprites": sprite, "times": [1.0, 2.6], "at": [0.5, 0.25], "size": 0.25, "sparkle": False,
                       "shadow": False})
    frames = two_shots()
    run = runner(plan([spec]), frames=frames)
    x, y = int(0.5 * W - 0.5), int(0.25 * H - 0.5)
    g = frame_at(2.0)
    assert np.allclose(run.frame(g, frames(g))[y, x], (1.0, 0.0, 0.0), atol=0.02)    # the stamp, held
    assert np.allclose(run.frame(g + 15, frames(g + 15))[y, x], CREAM, atol=0.01)      # rubbed out with it
    assert np.allclose(run.frame(frame_at(3.2), frames(frame_at(3.2)))[y, x], (1.0, 0.0, 0.0), atol=0.02)


def test_the_art_is_drawn_in_its_palette_with_an_ink_outline():
    art = entry(app())["art"]
    for name, fn in PAF.ICONS.items():
        rgb, a, origin = PAF.draw(fn(art, art["text"]), 64.0, 3.0, art["text"])
        assert a.max() == pytest.approx(1.0) and a[0].max() < 0.05 and a[-1].max() < 0.05     # room round it
        assert 0.0 <= origin[0] < a.shape[1] and 0.0 <= origin[1] < a.shape[0]
    rgb, a, origin = PAF.draw(PAF._hand(art, art["text"]), 100.0, 4.0, art["text"])
    x, y = int(round(origin[0])), int(round(origin[1]))
    assert a[y + 3, x] > 0.5 and a[max(y - 8, 0), x] < 0.5                      # its fingertip is the hot spot


# ---------------------------------------------------------------- credits
def test_credits_count_the_sprites_a_paint_app_stamps(tmp_path, monkeypatch):
    monkeypatch.setenv("MK_ASSETS", str(tmp_path / "assets"))
    root = tmp_path / "proj"
    (root / "sprites").mkdir(parents=True)
    reg = A.Registry()
    for name, tags in (("leaf1", ["sprite", "clover"]), ("leaf2", ["sprite", "clover"]), ("fly", ["sprite"])):
        disc_sprite(root / "sprites" / f"{name}.png")
        reg.add({"kind": "texture", "slug": name, "path": str(root / "sprites" / f"{name}.png"), "tags": tags,
                 "author": "a"})

    class Proj:
        pass
    Proj.root, Proj.cast = root, []
    Proj.data = {"fx": [{"kind": "paintapp", "stamps": {"sprites": {"tag": "clover"}},
                         "aspect": {"9x16": {"stamps": {"sprites": "fly"}}}}]}
    assert project_credits(Proj, reg) == (["leaf1", "leaf2", "fly"], [])
