"""The iris insert (`[[insert]] shape = "iris"`; mkmmd.core.iris, mkmmd.iris and their hooks in mkmmd.core.transition and
mkmmd.cutfx): its spec and errors, its window in the plan and the layers it asks for, its timing, the circle in every
output (on the projected anchor, sized in metres or as a share of the short side, per output), and the composite on tiny
synthetic frames through CutFx: no trail anywhere, the opening uncovering the picture, the rim (a vignette darker toward
the edge, a glint brighter at the upper left), an edge ring, a video inside, and the growth into the full frame that
ends on the inner picture itself. Fixtures use made-up shot names."""
import json
import re

import cv2
import numpy as np
import pytest

from mkmmd import cutfx as CF
from mkmmd import matte as M
from mkmmd import post as P
from mkmmd.core import iris as IR
from mkmmd.core import palette as PAL
from mkmmd.core import transition as TR

FPS, F0 = 30, 1
DAWN = PAL.get("rose-pine-dawn")
ANCHOR = 'obj("toy_eyehole").loc'
IRIS = {"from": 1.0, "to": 3.0, "shot": "p", "shape": "iris", "anchor": ANCHOR}
EXPAND = {"from": 3.0, "to": 4.0, "shot": "n", "shape": "iris", "anchor": ANCHOR, "out": "expand"}
SIZES = {"16x9": (192, 108), "1x1": (108, 108), "9x16": (108, 192)}


def project(inserts, transitions=()):
    """h 0-4 s (the host), n 4-6 s, and a plate shot p; the three outputs."""
    return {"output": [{"name": o} for o in SIZES],
            "shot": [{"name": "h", "from": 0.0, "to": 4.0}, {"name": "n", "from": 4.0, "to": 6.0}, {"name": "p", "plate": True}],
            "insert": list(inserts), "transition": list(transitions)}


def plan(spec=IRIS, **kw):
    return TR.plan(project([{**spec, **kw}]), FPS, F0, DAWN)


def iris(spec=IRIS, **kw):
    return plan(spec, **kw)["inserts"][0]


def with_clip(spec, clip):
    """`spec` showing the video `clip` instead of its shot."""
    return {**{k: v for k, v in spec.items() if k != "shot"}, "clip": clip}


# ---------------------------------------------------------------- the spec
def test_an_iris_gets_its_defaults():
    s = TR.normalize_insert(IRIS, DAWN)
    assert (s["shape"], s["shot"], s["clip"], s["anchor"]) == ("iris", "p", None, ANCHOR)
    assert (s["radius"], s["size"], s["out"], s["edge"], s["aspect"]) == (None, 0.3, "close", None, {})
    assert s["open"] == {"dur": 0.25, "ease": "out"} and s["expand"] == {"dur": 0.6, "ease": "in"}
    rim = s["rim"]
    assert (rim["width"], rim["dark"], rim["glint"], rim["fringe"]) == (0.3, 0.8, 0.85, 2.0)
    assert rim["color"] == pytest.approx(PAL.srgb(PAL.darkest(DAWN)), abs=1e-6)    # Dawn's darkest: its text colour
    assert rim["color"] == pytest.approx(PAL.srgb(DAWN["text"]), abs=1e-6)
    assert rim["light"] == pytest.approx(PAL.srgb(DAWN["surface"]), abs=1e-6)      # the glint: its lightest


def test_a_radius_replaces_the_size_and_an_aspect_gives_an_output_its_own():
    s = TR.normalize_insert({**IRIS, "radius": 0.004, "aspect": {"9x16": {"size": 0.5}}}, DAWN)
    assert (s["radius"], s["size"]) == (0.004, None)
    assert IR.extent(s, "9x16") == {"radius": None, "size": 0.5} and IR.extent(s, "16x9") == {"radius": 0.004, "size": None}
    s = TR.normalize_insert({**IRIS, "aspect": {"1x1": {"radius": 0.01}}}, DAWN)
    assert IR.extent(s, "1x1") == {"radius": 0.01, "size": None} and IR.extent(s, "9x16") == {"radius": None, "size": 0.3}


def test_a_rim_can_be_turned_off_or_set_and_an_edge_added():
    s = TR.normalize_insert({**IRIS, "rim": False, "edge": {}}, DAWN)
    assert s["rim"] is None and s["edge"] == {"color": pytest.approx(PAL.srgb(DAWN["text"]), abs=1e-6), "width": 4.0}
    s = TR.normalize_insert({**IRIS, "rim": {"width": 0.5, "color": "pine", "glint": 0, "fringe": 0}}, DAWN)
    assert (s["rim"]["width"], s["rim"]["dark"], s["rim"]["glint"], s["rim"]["fringe"]) == (0.5, 0.8, 0.0, 0.0)
    assert s["rim"]["color"] == pytest.approx(PAL.srgb(DAWN["pine"]), abs=1e-6)


def test_a_video_inside_takes_the_splits_forms():
    s = TR.normalize_insert(with_clip(IRIS, {"src": "v.mp4", "in": 2.0, "loop": True}), DAWN)
    assert s["shot"] is None and (s["clip"]["src"], s["clip"]["in"], s["clip"]["loop"], s["clip"]["fit"]) == (
        "v.mp4", 2.0, True, "cover")


def bad(**kw):
    spec = {**IRIS, **kw}
    return {k: v for k, v in spec.items() if v is not None}


@pytest.mark.parametrize("spec, frag", [
    (bad(radius=0.01, size=0.3), "radius and size together"),
    (bad(offset=[0, 0]), "unknown keys ['offset']"),
    (bad(pop={"dur": 0.3}), "unknown keys ['pop']"),
    (bad(open={"dur": 0.2, "overshoot": 0.1}), "unknown keys ['overshoot']"),
    (bad(open={"dur": -0.1}), "open.dur = -0.1"),
    (bad(open={"ease": "bounce"}), "open.ease = 'bounce'"),
    (bad(open=0.3), "open = 0.3"),
    (bad(expand={"dur": 0.5}), "only applies to out = 'expand'"),
    (bad(out="expand", expand={"dur": 0}), "expand.dur = 0.0"),
    (bad(out="pop"), "out = 'pop'"),
    (bad(rim={"width": 1.0}), "rim.width = 1.0"),
    (bad(rim={"dark": 1.5}), "rim.dark = 1.5"),
    (bad(rim={"glint": -0.1}), "rim.glint = -0.1"),
    (bad(rim={"fringe": -1}), "rim.fringe = -1.0"),
    (bad(rim={"tint": 1}), "unknown keys ['tint']"),
    (bad(rim="soft"), "rim = 'soft'"),
    (bad(rim={"color": "nope"}), "rim.color"),
    (bad(edge={"width": 0}), "width must be positive"),
    (bad(edge=3), "edge = 3"),
    (bad(clip="a.mp4"), "exactly one of `shot`"),
    (bad(shot=None), "exactly one of `shot`"),
    (bad(shot=None, clip={"src": "a.mp4", "fit": "stretch"}), "clip.fit = 'stretch'"),
    (bad(anchor=None), "needs `anchor`"),
    (bad(anchor=""), "anchor = ''"),
    (bad(size=0), "size = 0.0"),
    (bad(size=1.5), "size = 1.5"),
    (bad(radius=0), "radius = 0.0"),
    (bad(radius="big"), "radius = 'big'"),
    (bad(aspect={"9x16": {"radius": 0.1, "size": 0.2}}), "aspect.9x16: radius and size together"),
    (bad(aspect={"9x16": {"offset": [0, 0]}}), "unknown keys ['offset']"),
    (bad(aspect={"9x16": {}}), "expected {radius} or {size}"),
    (bad(aspect={"9x16": {"size": 2}}), "aspect.9x16: size = 2.0"),
    (bad(to=0.5), "`to` must come after"),
])
def test_bad_irises_say_what_is_wrong(spec, frag):
    with pytest.raises(TR.TransitionError, match=re.escape(frag)):
        TR.normalize_insert(spec, DAWN)


# ---------------------------------------------------------------- the plan
def test_an_iris_needs_its_plate_and_its_anchor_through_the_host_camera_on_every_frame():
    p = plan()
    s = p["inserts"][0]
    assert (s["host"], s["f0"], s["f1"], s["n_in"], s["n_exit"]) == ("h", F0 + 30, F0 + 90, 8, 8)
    d = TR.demands(p)
    assert sorted(d) == list(range(F0 + 30, F0 + 90))
    assert all(v == [{"kind": "plate", "shot": "p"}, {"kind": "point", "key": "i0", "expr": ANCHOR, "shot": "h"}]
               for v in d.values())
    assert TR.needs(p) == {"p": [F0 + 30, F0 + 89]}
    assert TR.summary(p)["inserts"][0] == {"index": 0, "shape": "iris", "frames": [F0 + 30, F0 + 89], "host": "h",
                                           "shot": "p", "out": "close"}
    rich = plan(EXPAND, edge={}, aspect={"9x16": {"radius": 0.01}})
    assert json.loads(json.dumps(rich)) == rich                           # JSON-clean, as every plan


def test_an_iris_with_a_video_needs_only_its_anchor():
    p = TR.plan(project([with_clip(IRIS, "a.mp4")]), FPS, F0, DAWN)
    d = TR.demands(p)
    assert sorted(d) == list(range(F0 + 30, F0 + 90)) and all(v == [{"kind": "point", "key": "i0", "expr": ANCHOR,
                                                                       "shot": "h"}] for v in d.values())
    assert TR.summary(p)["inserts"][0]["clip"] == "a.mp4" and TR.needs(p) == {}


def test_an_expand_grows_into_the_shot_that_starts_at_to():
    s = iris(EXPAND)
    assert (s["host"], s["f1"], s["n_in"], s["n_exit"]) == ("h", F0 + 120, 8, 18)
    with pytest.raises(TR.TransitionError, match=re.escape("so `shot` must be that shot, not 'p'")):
        plan(EXPAND, shot="p")
    with pytest.raises(TR.TransitionError, match=re.escape("grows into the shot that starts at `to` (no shot starts there)")):
        plan(EXPAND, to=3.9)
    video = {**with_clip(EXPAND, "a.mp4"), "from": 2.0, "to": 3.5}
    assert TR.plan(project([video]), FPS, F0, DAWN)["inserts"][0]["n_exit"] == 18     # a video: nothing to land on


def test_an_iris_that_opens_on_its_first_frame_and_cuts_out():
    s = iris(open={"dur": 0})
    assert (s["n_in"], s["n_exit"]) == (0, 0)
    assert IR.state(s, s["f0"]) == {"phase": "hold", "open": 1.0, "p": 0.0}
    assert IR.state(s, s["f1"] - 1)["open"] == 1.0 and IR.state(s, s["f1"])["phase"] == "none"


@pytest.mark.parametrize("spec, kw, frag", [
    (IRIS, {"to": 1.4}, "12 frames are too few: its opening takes 8 and its close 8, at least 16 in all"),
    (EXPAND, {"from": 3.5}, "15 frames are too few: its opening takes 8 and its expand 18, at least 26 in all"),
    (IRIS, {"open": {"dur": 0}, "to": 1.04}, "1 frames are too few: its opening takes 0 and its close 0, at least 2"),
    (IRIS, {"to": 4.5}, "inside one shot"),
    (IRIS, {"shot": "h"}, "the host shot itself"),
    (IRIS, {"shot": "nope"}, "no shot named 'nope'"),
])
def test_irises_that_do_not_fit_are_refused(spec, kw, frag):
    with pytest.raises(TR.TransitionError, match=re.escape(frag)):
        plan(spec, **kw)


def test_an_iris_may_not_overlap_another_window():
    with pytest.raises(TR.TransitionError, match="overlap"):
        TR.plan(project([{**IRIS, "from": 3.0, "to": 3.9}], [{"at": 4.0, "kind": "slice"}]), FPS, F0, DAWN)
    with pytest.raises(TR.TransitionError, match="overlap"):
        TR.plan(project([IRIS, {**IRIS, "from": 2.5, "to": 3.8}]), FPS, F0, DAWN)


# ---------------------------------------------------------------- timing
def test_it_opens_from_a_small_circle_to_rest_holds_and_closes_back_onto_its_point():
    s = iris()
    states = [IR.state(s, f) for f in range(s["f0"], s["f1"])]
    opening, closing = states[:8], states[-8:]
    assert [st["phase"] for st in opening] == ["in"] * 8 and [st["phase"] for st in closing] == ["out"] * 8
    assert all(st == {"phase": "hold", "open": 1.0, "p": 0.0} for st in states[8:-8])
    sizes = [st["open"] for st in opening]
    assert 0.0 < sizes[0] < 0.35 and sizes[-1] == 1.0 and all(b > a for a, b in zip(sizes, sizes[1:]))
    assert [st["open"] for st in closing] == sizes[::-1]                   # the opening played backwards
    assert 0.0 < closing[-1]["open"] < 0.35                                # a small circle on the last frame, gone on `to`
    assert IR.state(s, s["f1"])["phase"] == "none" and IR.state(s, s["f0"] - 1)["phase"] == "none"


def test_an_expand_moves_on_every_frame_and_reaches_one_exactly_on_the_last():
    s = iris(EXPAND)
    grow = [IR.state(s, f) for f in range(s["f1"] - 18, s["f1"])]
    ps = [st["p"] for st in grow]
    assert all(st["phase"] == "expand" and st["open"] == 1.0 for st in grow)
    assert 0.0 < ps[0] < 0.01 and ps[-1] == 1.0 and all(b > a for a, b in zip(ps, ps[1:]))   # eased in: a slow start
    assert IR.state(s, s["f1"] - 19)["phase"] == "hold"


# ---------------------------------------------------------------- the circle in each output
def state_at(s, f, out, anchor=(120.0, 40.0), per_metre=10.0):
    return IR.circle(s, IR.state(s, f), out, SIZES[out], anchor, per_metre)


@pytest.mark.parametrize("out", list(SIZES))
def test_the_circle_sits_on_the_projected_anchor_at_its_rest_size(out):
    s = iris()
    w, h = SIZES[out]
    c = state_at(s, s["f0"] + 20, out, anchor=(0.6 * w, 0.4 * h))
    assert c["centre"] == (0.6 * w, 0.4 * h) and c["at"] == c["centre"]
    assert c["radius"] == pytest.approx(0.3 * min(w, h) / 2.0)             # size: the diameter, a share of the short side
    k0 = 2.0 * c["radius"] / min(w, h) * 1.04                              # the picture just covers it (and its edge pixels)
    assert k0 <= c["k"] <= k0 * 1.002
    assert c["glint"] == 0.85 and c["p"] == 0.0


def test_a_radius_in_metres_is_seen_at_the_anchors_depth_and_an_aspect_sizes_its_output():
    s = iris(radius=0.004, aspect={"9x16": {"size": 0.6}})
    f = s["f0"] + 20
    assert state_at(s, f, "16x9", per_metre=12.0)["radius"] == pytest.approx(0.004 * 12.0 * 108)
    assert state_at(s, f, "16x9", per_metre=24.0)["radius"] == pytest.approx(0.004 * 24.0 * 108)   # a push: it grows
    assert state_at(s, f, "1x1", per_metre=12.0)["radius"] == pytest.approx(0.004 * 12.0 * 108)
    assert state_at(s, f, "9x16", per_metre=12.0)["radius"] == pytest.approx(0.6 * 108 / 2.0)      # its own size


def test_the_opening_and_the_close_scale_the_circle_only():
    s = iris()
    rest = state_at(s, s["f0"] + 20, "16x9")
    for f in list(range(s["f0"], s["f0"] + 8)) + list(range(s["f1"] - 8, s["f1"])):
        c = state_at(s, f, "16x9")
        assert c["radius"] == pytest.approx(rest["radius"] * IR.state(s, f)["open"])
        assert (c["k"], c["at"]) == (rest["k"], rest["at"])                # the iris uncovers a picture that stays put


@pytest.mark.parametrize("out", list(SIZES))
def test_an_expand_ends_centred_with_its_vignette_off_the_frame_and_grows_every_frame(out):
    s = iris(EXPAND)
    w, h = SIZES[out]
    cs = [state_at(s, f, out) for f in range(s["f1"] - 18, s["f1"])]
    last = cs[-1]
    assert last["centre"] == ((w - 1) / 2.0, (h - 1) / 2.0) and last["at"] == last["centre"] and last["k"] == 1.0
    assert last["radius"] * (1.0 - 0.3) > np.hypot(w, h) / 2.0 and last["glint"] == 0.0
    assert all(b["radius"] > a["radius"] for a, b in zip(cs, cs[1:]))
    assert all(b["glint"] < a["glint"] for a, b in zip(cs, cs[1:]))        # the glint fades as it grows
    assert IR.fill_radius((w, h), None) == pytest.approx(np.hypot(w, h) / 2.0 * 1.02)


def test_the_picture_never_shows_its_edge_inside_the_circle():
    rng = np.random.default_rng(3)
    for _ in range(400):
        w, h = SIZES[rng.choice(list(SIZES))]
        centre = (rng.uniform(-0.3, 1.3) * w, rng.uniform(-0.3, 1.3) * h)
        r = rng.uniform(0.5, 1.5) * max(w, h)
        k, at = IR.cover((w, h), centre, r, rng.uniform(0.05, 1.0))
        hx, hy = k * (w - 1) / 2.0, k * (h - 1) / 2.0
        ys, xs = np.mgrid[0:h, 0:w]
        seen = np.hypot(xs - centre[0], ys - centre[1]) <= r + 0.5             # the circle's pixels in the frame
        if seen.any():
            assert (np.abs(xs[seen] - at[0]) <= hx + 1e-6).all() and (np.abs(ys[seen] - at[1]) <= hy + 1e-6).all()


# ---------------------------------------------------------------- compositing
def coded(g, w, h):
    """A picture whose red channel counts the column, blue the row, green says whose it is."""
    img = np.empty((h, w, 3), np.float32)
    img[..., 0] = (np.arange(w, dtype=np.float32) / (w - 1))[None, :]
    img[..., 1] = g
    img[..., 2] = (np.arange(h, dtype=np.float32) / (h - 1))[:, None]
    return img


def flat(colour, w, h):
    return np.broadcast_to(np.asarray(colour, np.float32), (h, w, 3)).copy()


def save_rgb(path, img):
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), (np.clip(img, 0, 1)[..., ::-1] * 255 + 0.5).astype(np.uint8))


def build(root, out, spec=IRIS, inner=None, anchor=(0.6, 0.4), per_metre=10.0, footage_of=None, **kw):
    """The plan, the layers it asks for written to `root` (the plate `inner`, the anchor at `anchor` frame fractions) and
    a CutFx for output `out`."""
    p = plan(spec, **kw)
    w, h = SIZES[out]
    inner = coded(0.8, w, h) if inner is None else inner
    for f, items in TR.demands(p).items():
        for it in items:
            if it["kind"] == "plate":
                save_rgb(root / TR.plate_rel(it["shot"], f), inner)
            else:
                (root / TR.point_rel(it["key"], f)).parent.mkdir(parents=True, exist_ok=True)
                (root / TR.point_rel(it["key"], f)).write_text(json.dumps({"p": list(anchor), "depth": 2.0, "m": per_metre}))
    return p["inserts"][0], CF.CutFx(p, root, (w, h), out, footage_of=footage_of)


def plate(root, s, f, out):
    return P.read(root / TR.plate_rel(s["shot"], f), SIZES[out])


def circle_px(s, f, out, anchor=(0.6, 0.4), per_metre=10.0):
    w, h = SIZES[out]
    return IR.circle(s, IR.state(s, f), out, (w, h), M.frac_to_px(anchor, (w, h)), per_metre)


def dist(c, out):
    w, h = SIZES[out]
    ys, xs = np.mgrid[0:h, 0:w]
    return np.hypot(xs - c["centre"][0], ys - c["centre"][1])


@pytest.mark.parametrize("out", list(SIZES))
@pytest.mark.parametrize("spec, kw", [(IRIS, {}), (EXPAND, {}), (IRIS, {"edge": {"color": "love", "width": 60}})])
def test_no_trail_anywhere_every_pixel_outside_the_circle_and_its_edge_is_the_hosts(tmp_path, out, spec, kw):
    s, fx = build(tmp_path, out, spec, **kw)
    w, h = SIZES[out]
    host = coded(0.2, w, h)
    ring = kw.get("edge", {}).get("width", 0) * min(w, h) / 1080.0
    for f in range(s["f0"], s["f1"]):
        img = fx.frame(f, host)
        c = circle_px(s, f, out)
        outside = dist(c, out) >= c["radius"] + ring + 0.5
        assert np.array_equal(img[outside], host[outside]), f"frame {f}: something drawn outside the circle"
        assert fx.active(f)
    assert fx.frame(s["f1"], host) is host and fx.frame(s["f0"] - 1, host) is host


@pytest.mark.parametrize("out", list(SIZES))
def test_an_expand_grows_into_the_whole_frame_and_its_last_frame_is_the_inner_picture(tmp_path, out):
    s, fx = build(tmp_path, out, EXPAND)
    w, h = SIZES[out]
    host = coded(0.2, w, h)
    shares = []
    for f in range(s["f1"] - 18, s["f1"]):
        img = fx.frame(f, host)
        shares.append(float((np.abs(img - host).max(axis=2) > 1e-6).mean()))
    assert all(b > a or b == 1.0 for a, b in zip(shares, shares[1:])) and shares[-1] == 1.0
    assert shares[-2] == 1.0                                               # the frame covered before the end, vignette and all
    last = fx.frame(s["f1"] - 1, host)
    assert np.abs(last - plate(tmp_path, s, s["f1"] - 1, out)).max() < 1e-6     # the cut lands on that very picture


def test_the_opening_uncovers_a_picture_that_stays_put(tmp_path):
    s, fx = build(tmp_path, "16x9", rim=False)
    host = coded(0.2, *SIZES["16x9"])
    rest = fx.frame(s["f0"] + 20, host)
    first = fx.frame(s["f0"], host)
    c = circle_px(s, s["f0"], "16x9")
    inside = dist(c, "16x9") <= c["radius"] - 0.5
    assert 0 < inside.sum() and np.abs(first[inside] - rest[inside]).max() < 1e-3      # the same picture, warped alike
    assert (np.abs(first - host).max(axis=2) > 1e-6).sum() < (np.abs(rest - host).max(axis=2) > 1e-6).sum()


def test_the_middle_of_the_circle_is_the_picture_whatever_the_rim(tmp_path):
    for rim in (False, {}):
        s, fx = build(tmp_path / str(bool(rim)), "1x1", inner=flat((0.9, 0.85, 0.8), *SIZES["1x1"]), rim=rim)
        img = fx.frame(s["f0"] + 20, coded(0.2, *SIZES["1x1"]))
        c = circle_px(s, s["f0"] + 20, "1x1")
        d = dist(c, "1x1")
        keep = d <= (c["radius"] - 0.5 if rim is False else c["radius"] * 0.7 - 0.5)
        assert np.abs(img[keep] - plate(tmp_path / str(bool(rim)), s, s["f0"] + 20, "1x1")[keep]).max() < 1e-4


def luma(img):
    return img @ np.float32([0.2126, 0.7152, 0.0722])


def rim_frame(tmp_path, out="16x9", **kw):
    """(the frame at rest, its circle, the inner colour as the plate holds it) of a big flat light iris mid-frame."""
    s, fx = build(tmp_path, out, inner=flat((0.9, 0.85, 0.8), *SIZES[out]), size=0.8, anchor=(0.5, 0.5), **kw)
    img = fx.frame(s["f0"] + 20, coded(0.2, *SIZES[out]))
    return img, circle_px(s, s["f0"] + 20, out, anchor=(0.5, 0.5)), plate(tmp_path, s, s["f0"] + 20, out)[0, 0]


def test_the_vignette_darkens_toward_the_edge_toward_the_rims_colour(tmp_path):
    img, c, inner = rim_frame(tmp_path)
    cx, cy, r = c["centre"][0], c["centre"][1], c["radius"]                # 43 px: the vignette from 30 px out
    row = luma(img[int(round(cy))])
    xs = [int(round(cx + q * r)) for q in (0.5, 0.75, 0.85, 0.93, 0.985)]  # out to the right, clear of both glints
    ls = [float(row[x]) for x in xs]
    assert ls[0] == pytest.approx(float(luma(inner)), abs=1e-4)
    assert all(b < a - 0.005 for a, b in zip(ls, ls[1:]))
    text = np.float32(PAL.srgb(DAWN["text"]))
    edge = img[int(round(cy)), xs[-1]]
    assert np.abs(edge - text).sum() < np.abs(inner - text).sum() * 0.35   # most of the way to Dawn's darkest


def test_the_glint_is_brightest_at_the_upper_left_fainter_opposite_and_gone_without_a_rim(tmp_path):
    def arc(img, c, deg):
        a = np.radians(deg)
        pts = [(c["centre"][0] + q * c["radius"] * np.cos(a), c["centre"][1] - q * c["radius"] * np.sin(a))
               for q in np.linspace(0.88, 0.98, 6)]
        return max(float(luma(img[int(round(y)), int(round(x))])) for x, y in pts)

    img, c, _ = rim_frame(tmp_path / "rim")
    upper_left, lower_right, left, right = arc(img, c, 135), arc(img, c, -45), arc(img, c, 180), arc(img, c, 0)
    assert upper_left > lower_right > right + 0.02 and upper_left > left + 0.1
    dull, c, _ = rim_frame(tmp_path / "dull", rim={"glint": 0})
    assert arc(dull, c, 135) == pytest.approx(arc(dull, c, 45), abs=0.02)   # no glint: the rim alike all round


def test_an_edge_rings_the_outside_of_the_circle(tmp_path):
    s, fx = build(tmp_path, "16x9", rim=False, edge={"color": "#ff0000", "width": 40}, size=0.6, anchor=(0.5, 0.5))
    img = fx.frame(s["f0"] + 20, coded(0.2, *SIZES["16x9"]))
    c = circle_px(s, s["f0"] + 20, "16x9", anchor=(0.5, 0.5))
    y = int(round(c["centre"][1]))
    x_ring = int(round(c["centre"][0] + c["radius"] + 2))                # 4 px of ring at 108 px on the short side
    assert np.abs(img[y, x_ring] - [1.0, 0.0, 0.0]).max() < 0.02
    x_out = int(round(c["centre"][0] + c["radius"] + 7))
    assert np.array_equal(img[y, x_out], coded(0.2, *SIZES["16x9"])[y, x_out])


def test_a_circle_under_half_a_pixel_draws_nothing(tmp_path):
    s, fx = build(tmp_path, "16x9", radius=0.001, per_metre=0.002)
    host = coded(0.2, *SIZES["16x9"])
    assert fx.frame(s["f0"] + 20, host) is host


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


def video(tmp_path, footage_of, out="9x16", **clip):
    spec = with_clip(IRIS, {"src": "v.mp4", **clip})
    return build(tmp_path, out, spec, footage_of=footage_of)


def test_a_video_inside_plays_from_its_in_point_at_the_frames_size(tmp_path):
    clip = FakeClip()
    s, fx = video(tmp_path, lambda src: clip, **{"in": 2.0, "rate": 0.5})
    fx.check(range(s["f0"], s["f1"]))
    assert clip.calls[0] == ("index", 0.0, 2.0, 0.5, False)
    img = fx.frame(s["f0"] + 15, coded(0.2, *SIZES["9x16"]))
    _, t, size, kw = clip.calls[-1]
    assert t == pytest.approx(15 / FPS) and size == SIZES["9x16"]
    assert kw["start"] == 2.0 and kw["rate"] == 0.5 and kw["fit"] == "cover" and kw["focus"] == (0.5, 0.5)
    c = circle_px(s, s["f0"] + 15, "9x16")
    x, y = (int(round(v)) for v in c["centre"])
    assert np.abs(img[y, x] - clip.colour).max() < 1e-4


def test_a_video_that_cannot_be_read_is_an_error_naming_the_iris(tmp_path):
    def missing(src):
        raise FileNotFoundError(f"clip {src!r}: no such file")
    s, fx = video(tmp_path, missing)
    with pytest.raises(TR.TransitionError, match=r"insert 0 \(iris 1.0-3.0 s\): clip 'v.mp4': clip 'v.mp4': no such file"):
        fx.check([s["f0"]])
