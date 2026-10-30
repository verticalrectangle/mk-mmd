"""The page turn (`[[transition]] kind = "page"`; mkmmd.core.pageturn, mkmmd.pageturn and their hooks in
mkmmd.core.transition and mkmmd.cutfx): its spec and errors, its window at a cut and at the loop's wrap (the clip's end
turned over onto its first frame, held) and the layers each asks for, the curl's normal and travel, and the composite on
tiny synthetic frames through CutFx: the page lifting at a corner and turning toward its `dir`, the next picture
uncovered a little more on every frame and the page not gone before the cut, its paper back with the picture showing
through it mirrored; and the wrap, no cut effect: the finished frame turned over frame 0 as finished, held, the screen
type laid on after the grade and the keep mask turning with the page. Fixtures use made-up shot names."""
import json
import math
import re

import cv2
import numpy as np
import pytest

from mkmmd import cutfx as CF
from mkmmd import pageturn as PAGE
from mkmmd.core import pageturn as PG
from mkmmd.core import palette as PAL
from mkmmd.core import transition as TR

FPS, F0 = 30, 1
DAWN = PAL.get("rose-pine-dawn")
SIZES = {"16x9": (192, 108), "1x1": (108, 108), "9x16": (108, 192)}
PAGE_AT = {"at": 3.0, "kind": "page"}
WRAP = {"at": 6.0, "kind": "page"}
RED, BLUE = (0.9, 0.1, 0.1), (0.1, 0.1, 0.9)


def project(transitions, duration=6.0, shots=None):
    """a 0-3 s and b 3-6 s (or `shots`), a clip of `duration` seconds (None: not given), the three outputs."""
    shots = shots or [{"name": "a", "from": 0.0, "to": 3.0}, {"name": "b", "from": 3.0, "to": 6.0}]
    return {"project": {} if duration is None else {"duration": duration}, "output": [{"name": o} for o in SIZES],
            "shot": shots, "transition": list(transitions)}


def plan(spec=PAGE_AT, **kw):
    return TR.plan(project([{**spec, **kw}]), FPS, F0, DAWN)


# ---------------------------------------------------------------- the spec
def test_a_page_turn_gets_its_defaults():
    t = TR.normalize_transition(PAGE_AT, DAWN)
    assert (t["dur"], t["ease"], t["dir"], t["angle"], t["radius"], t["through"], t["shadow"]) == (
        0.6, "inout", "left", 20.0, 0.12, 0.18, 0.45)
    assert t["back"] == pytest.approx(PAL.srgb(DAWN["surface"]), abs=1e-6)     # Dawn's lightest: its paper
    t = TR.normalize_transition({**PAGE_AT, "dir": "up", "angle": -30, "back": "gold", "dur": 0.4, "ease": "out"}, DAWN)
    assert (t["dir"], t["angle"], t["dur"], t["ease"]) == ("up", -30.0, 0.4, "out")
    assert t["back"] == pytest.approx(PAL.srgb(DAWN["gold"]), abs=1e-6)


@pytest.mark.parametrize("kw, frag", [
    ({"dir": "sideways"}, "dir = 'sideways'"),
    ({"angle": 75}, "angle = 75.0"),
    ({"angle": "steep"}, "angle = 'steep'"),
    ({"radius": 0}, "radius = 0.0"),
    ({"radius": 0.6}, "radius = 0.6"),
    ({"through": 1.5}, "through = 1.5"),
    ({"shadow": -0.1}, "shadow = -0.1"),
    ({"back": "nope"}, "back"),
    ({"edge": {}}, "unknown keys ['edge']"),
    ({"ease": "bounce"}, "ease = 'bounce'"),
    ({"dur": 0}, "dur must be positive"),
])
def test_bad_page_turns_say_what_is_wrong(kw, frag):
    with pytest.raises(TR.TransitionError, match=re.escape(frag)):
        TR.normalize_transition({**PAGE_AT, **kw}, DAWN)


# ---------------------------------------------------------------- the plan
def test_a_page_turn_at_a_cut_needs_the_next_shots_plates_over_its_window():
    p = plan()
    t = p["transitions"][0]
    assert (t["cut"], t["first"], t["last"], t["n"]) == (F0 + 90, F0 + 72, F0 + 89, 18)
    assert (t["out"], t["in"], t["plate"], t["matte"], t["still"]) == ("a", "b", "b", None, None)
    d = TR.demands(p)
    assert sorted(d) == list(range(F0 + 72, F0 + 90)) and all(v == [{"kind": "plate", "shot": "b"}] for v in d.values())
    assert TR.needs(p) == {"b": [F0 + 72, F0 + 89]}
    assert json.loads(json.dumps(p)) == p


def test_at_the_clips_end_it_is_the_loops_wrap_onto_the_first_frame_held():
    p = plan(WRAP)
    t = p["transitions"][0]
    assert (t["cut"], t["first"], t["last"]) == (F0 + 180, F0 + 162, F0 + 179)
    assert (t["out"], t["in"], t["plate"], t["still"]) == ("b", "a", "a", F0)
    assert TR.wrap_of(p) is t and TR.wrap_of(plan()) is None and TR.wrap_of(None) is None
    assert TR.demands(p) == {} and TR.needs(p) == {}          # no layers: the finished frames turn over frame 0 as it is
    assert TR.summary(p)["transitions"][0]["in"] == "a" and json.loads(json.dumps(p)) == p
    one = TR.plan(project([WRAP], shots=[{"name": "a", "from": 0.0, "to": 6.0}]), FPS, F0, DAWN)["transitions"][0]
    assert (one["out"], one["in"], one["still"]) == ("a", "a", F0)            # a one-shot loop turns onto itself


def test_the_wrap_needs_the_clips_length_and_is_the_page_turns_alone():
    with pytest.raises(TR.TransitionError, match=re.escape("or at the clip's end ([project] duration)")):
        TR.plan(project([WRAP], duration=None), FPS, F0, DAWN)
    with pytest.raises(TR.TransitionError, match=r"no shot start at that frame \(cuts at 0.00, 3.00 s\)$"):
        TR.plan(project([{"at": 6.0, "kind": "slash"}]), FPS, F0, DAWN)
    with pytest.raises(TR.TransitionError, match=re.escape("window starts before shot 'c' does (it is 6 frames long)")):
        TR.plan(project([WRAP], shots=[{"name": "a", "from": 0.0, "to": 3.0}, {"name": "b", "from": 3.0, "to": 5.8},
                                      {"name": "c", "from": 5.8, "to": 6.0}]), FPS, F0, DAWN)   # it turns the last shot


# ---------------------------------------------------------------- the geometry
def test_the_curls_normal_points_from_the_flat_page_toward_its_lifted_edge():
    t = TR.normalize_transition(PAGE_AT, DAWN)
    for d, n in (("left", (1.0, 0.0)), ("right", (-1.0, 0.0)), ("up", (0.0, 1.0)), ("down", (0.0, -1.0))):
        assert PG.normal({**t, "dir": d, "angle": 0.0}) == pytest.approx(n)
    nx, ny = PG.normal(t)                                                      # left, 20: the bottom right corner first
    assert (nx, ny) == pytest.approx((math.cos(math.radians(20)), math.sin(math.radians(20))))
    w, h = 192, 108
    corners = {(x, y): x * nx + y * ny for x in (-0.5, w - 0.5) for y in (-0.5, h - 0.5)}
    assert max(corners, key=corners.get) == (w - 0.5, h - 0.5)
    c0, c1 = PG.travel(t, (w, h))
    assert c0 == pytest.approx(corners[(w - 0.5, h - 0.5)])
    assert c1 == pytest.approx(corners[(-0.5, -0.5)] - 0.12 * h - 1.0)


def test_the_curl_travels_on_every_frame_and_is_never_at_either_end_inside_the_window():
    t = plan()["transitions"][0]
    ps = [PG.progress(t, k) for k in range(t["n"])]
    assert 0.0 < ps[0] and ps[-1] < 1.0 and all(b > a for a, b in zip(ps, ps[1:]))


# ---------------------------------------------------------------- compositing
def flat(colour, w, h):
    return np.broadcast_to(np.asarray(colour, np.float32), (h, w, 3)).copy()


def coded(g, w, h):
    """A picture whose red channel counts the column, blue the row, green says whose it is."""
    img = np.empty((h, w, 3), np.float32)
    img[..., 0] = (np.arange(w, dtype=np.float32) / (w - 1))[None, :]
    img[..., 1] = g
    img[..., 2] = (np.arange(h, dtype=np.float32) / (h - 1))[:, None]
    return img


def save_rgb(path, img):
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), (np.clip(img, 0, 1)[..., ::-1] * 255 + 0.5).astype(np.uint8))


def build(root, out, spec=PAGE_AT, incoming=None, shots=None, **kw):
    """The plan, its plates written to `root` (`incoming`, flat blue by default) and a CutFx for output `out`."""
    p = TR.plan(project([{**spec, **kw}], shots=shots), FPS, F0, DAWN)
    w, h = SIZES[out]
    incoming = flat(BLUE, w, h) if incoming is None else incoming
    for f, items in TR.demands(p).items():
        for it in items:
            save_rgb(root / TR.plate_rel(it["shot"], f), incoming)
    return p["transitions"][0], CF.CutFx(p, root, (w, h), out)


def kinds(img):
    """Masks of what each pixel shows: the red page's front, the blue picture under it, the whitish paper back."""
    r, g, b = img[..., 0], img[..., 1], img[..., 2]
    return r > b + 0.3, b > r + 0.3, (img.min(axis=2) > 0.55)


@pytest.mark.parametrize("out", list(SIZES))
def test_the_page_uncovers_the_next_picture_a_little_more_on_every_frame(tmp_path, out):
    t, fx = build(tmp_path, out)
    w, h = SIZES[out]
    main = flat(RED, w, h)
    assert fx.frame(t["first"] - 1, main) is main and fx.frame(t["cut"], main) is main
    shares, backs = [], []
    for f in range(t["first"], t["last"] + 1):
        img = fx.frame(f, main)
        assert fx.active(f) and not np.array_equal(img, main), f"frame {f}: the page has not lifted"
        assert np.abs(img - np.asarray(fx._plate("b", f, main))).max() > 0.02, f"frame {f}: all gone before the cut"
        front, under, back = kinds(img)
        shares.append(float(under.mean()))
        backs.append(float(back.mean()))
    assert all(b >= a for a, b in zip(shares, shares[1:])) and shares[-1] > 0.85 and shares[0] < 0.05
    assert max(backs) > 0.15                                                  # its paper back shows on the way


@pytest.mark.parametrize("d", ["left", "right", "up", "down"])
def test_the_page_turns_over_toward_its_dir(tmp_path, d):
    t, fx = build(tmp_path, "16x9", dir=d, angle=0.0)
    w, h = SIZES["16x9"]
    img = fx.frame(t["first"] + 9, flat(RED, w, h))
    front, under, _ = kinds(img)
    ys, xs = np.mgrid[0:h, 0:w]
    assert front.any() and under.any()
    if d == "left":                                                           # lifted at the right, turned to the left
        assert xs[under].min() > xs[front].max()
    elif d == "right":
        assert xs[under].max() < xs[front].min()
    elif d == "up":
        assert ys[under].min() > ys[front].max()
    else:
        assert ys[under].max() < ys[front].min()


def test_with_an_angle_one_corner_lifts_first(tmp_path):
    for angle, corner in ((20.0, (1, 1)), (-20.0, (1, 0))):                  # (right, bottom) and (right, top)
        t, fx = build(tmp_path / str(angle), "16x9", angle=angle)
        w, h = SIZES["16x9"]
        main = flat(RED, w, h)
        img = fx.frame(t["first"] + 3, main)
        changed = np.abs(img - main).max(axis=2) > 0.05
        ys, xs = np.nonzero(changed)
        assert len(xs) and (xs.mean() > w / 2) == bool(corner[0]) and (ys.mean() > h / 2) == bool(corner[1])


def test_its_paper_back_shows_the_picture_through_it_mirrored(tmp_path):
    t, fx = build(tmp_path, "16x9", dir="left", angle=0.0, through=0.5, shadow=0.0)
    w, h = SIZES["16x9"]
    main = coded(0.3, w, h)                                                   # red counts the columns
    k = 9
    img = fx.frame(t["first"] + k, main)
    c = PG.line(t, k, (w, h))
    R = PG.roll(t, (w, h))
    xs = np.arange(w, dtype=np.float32)
    xo = 2.0 * c + math.pi * R - xs                                           # the page point the turned part shows
    turned = (xs < c - 2.0) & (xo > 2.0) & (xo < w - 3.0)
    assert turned.sum() > 10
    row = img[h // 2, turned, 0]
    paper = np.float32(PAL.srgb(DAWN["surface"]))[0]
    want = paper * (0.5 + 0.5 * xo[turned] / (w - 1))
    assert np.abs(row - want).max() < 0.06                                    # its grain is all that is left
    assert np.polyfit(xs[turned], row, 1)[0] < 0.0 < np.polyfit(xs, main[h // 2, :, 0], 1)[0]   # mirrored
    t0, plain = build(tmp_path / "plain", "16x9", dir="left", angle=0.0, through=0.0, shadow=0.0)
    flat_back = plain.frame(t0["first"] + k, main)
    assert np.abs(flat_back[h // 2, turned, 0] - paper).max() < 0.05           # through = 0: the paper alone


def test_the_wrap_is_no_cut_effect_and_turns_the_finished_frame_over_frame_0_held(tmp_path):
    """CutFx leaves the wrap's window alone and needs nothing drawn for it; `wrap` turns the frame as finished over frame
    0 as finished, held: frame 0 itself under the page, under its shadow at most, and nothing else of it is touched."""
    w, h = SIZES["16x9"]
    p = TR.plan(project([WRAP]), FPS, F0, DAWN)
    t = p["transitions"][0]
    fx = CF.CutFx(p, tmp_path / "empty", (w, h), "16x9")
    window = range(t["first"], t["last"] + 1)
    main, still = flat(RED, w, h), flat(BLUE, w, h)                         # the clip's end and frame 0, as finished
    assert not any(fx.active(f) for f in window) and fx.missing(window) == [] and fx.frame(t["last"], main) is main
    for f in (t["first"] + 6, t["last"]):
        img, layer, mask = PAGE.wrap(t, f, (main, None, None), (still, None, None))
        assert layer is None and mask is None
        under = (img[..., 0] < 0.15) & (img[..., 1] < 0.15) & (img[..., 2] > 0.3)   # the blue picture alone
        under = cv2.erode(under.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)   # not its antialiased edge
        assert under.sum() > 50
        ratio = img[under] / still[under]
        assert float(ratio.max()) <= 1.0 + 1e-5                               # the first frame, under a shadow at most
        assert np.abs(ratio - ratio[:, :1]).max() < 1e-3                     # darkened evenly: its own colour
    last = PAGE.wrap(t, t["last"], (main, None, None), (still, None, None))[0]
    assert float((np.abs(last - still).max(axis=2) < 1e-5).mean()) > 0.6     # mostly frame 0 itself by the end


def test_the_type_laid_on_after_the_grade_and_the_keep_mask_turn_with_the_page():
    """At the wrap the screen type laid over the graded picture (straight RGBA) and `[post] keep`'s mask turn as the
    picture does: the page's own on the flat page, frame 0's past the roll, none where the page's back lies; under the
    shadow the type is darker, not fainter, and a mask is as it was."""
    w, h = SIZES["16x9"]
    t = plan(WRAP, dir="left", angle=0.0)["transitions"][0]
    k = 6
    turn = PAGE.Turn(t, t["first"] + k, (w, h))
    c, R, (c0, _) = PG.line(t, k, (w, h)), PG.roll(t, (w, h)), PG.travel(t, (w, h))
    xs = np.arange(w, dtype=np.float32)
    past = xs > c + R + 2.0                                                   # frame 0 shows
    page = xs < c - max(c0 - c - math.pi * R, 0.0) - 2.0                      # the flat page
    up = 2.0 * c + math.pi * R - xs                                           # the page point its turned-over back shows
    back = (xs < c - 2.0) & (up > 2.0) & (up < w - 3.0)
    assert past.sum() > 10 and page.sum() > 10 and back.sum() > 10
    lit = turn.keep[..., 0]
    assert (lit[:, past] < 0.98).any() and (lit[:, page] < 0.98).any()       # the shadow falls on both
    mine, theirs = np.empty((h, w, 4), np.float32), np.empty((h, w, 4), np.float32)
    mine[...], theirs[...] = (0.9, 0.8, 0.1, 0.6), (0.1, 0.3, 0.9, 1.0)
    out = turn.layer(mine, theirs)
    for region, own in ((page, mine), (past, theirs)):
        assert np.allclose(out[:, region, :3], own[:, region, :3] * lit[:, region, None], atol=1e-6)
        assert np.array_equal(out[:, region, 3], own[:, region, 3])
    assert (out[:, back, 3] == 0.0).all()
    alone = turn.layer(None, theirs)
    assert (alone[:, page, 3] == 0.0).all() and (alone[:, past, 3] == 1.0).all()
    m = turn.mask(np.full((h, w), 0.25, np.float32), np.full((h, w), 0.75, np.float32))
    assert (m[:, page] == 0.25).all() and (m[:, past] == 0.75).all() and (m[:, back] == 0.0).all()
    assert turn.layer(None, None) is None and turn.mask(None, None) is None
