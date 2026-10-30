"""Op-art ([[fx]] kind "spiral", mkmmd.core.fields and mkmmd.fields) and the collages' pattern backs: the keys and what
they refuse, each pattern's symmetry (a spiral's arms), antialiased edges and a too-fine middle drawn as the colours' mean,
turning on `speed` and stepping on hits, smooth from frame to frame, the same in any order, kept behind her by a ground
mask, and a grid's cells each on a pattern of its own that steps on the grid's hits."""
import math
import re

import cv2
import numpy as np
import pytest

from mkmmd import fields as FL
from mkmmd import framefx as FF
from mkmmd.core import fields as CF
from mkmmd.core import framefx as FX
from mkmmd.core import palette as PAL
from mkmmd.core import transition as TR

FPS, F0 = 30, 1
DAWN = PAL.get("rose-pine-dawn")
W, H = 96, 54
RED, BLUE, WHITE, BLACK = [1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [1.0, 1.0, 1.0], [0.0, 0.0, 0.0]


def plan(*fx, size=(W, H)):
    data = {"project": {"duration": 4.0}, "output": [{"name": "t", "size": list(size)}],
            "shot": [{"name": "a", "from": 0.0, "to": 4.0}], "fx": list(fx)}
    return TR.plan(data, FPS, F0, DAWN)


def runner(pl, figure=None, size=(W, H)):
    return FF.FrameFx(pl["fx"], size, fps=FPS, frame0=F0, frame_of=lambda g: None, point_of=None,
                      figure_of=(lambda key, g: figure) if figure is not None else None)


def spiral(**kw):
    return {"kind": "spiral", "from": 0.0, "to": 3.0, **kw}


def grey():
    return np.full((H, W, 3), 0.5, np.float32)


@pytest.mark.parametrize("spec, frag", [
    (spiral(pattern="zigzag"), "pattern 'zigzag' is none of spiral, rings, moire, sunburst, checker"),
    (spiral(colors=["rose"]), "expected a list of 2 to 4 colours"),
    (spiral(colors=["rose", "gold", "pine", "foam", "iris"]), "2 to 4 colours"),
    (spiral(colors="rose"), "expected a list of 2 to 4 colours"),
    (spiral(pattern="rings", twist=2), "twist (a key of spiral) does not shape a rings: it takes size"),
    (spiral(pattern="spiral", size=0.1), "size (a key of rings or moire or checker) does not shape a spiral"),
    (spiral(arms=0), "arms = 0: expected 1..64"),
    (spiral(pattern="moire", moire="dots"), "moire 'dots' is none of rings, lines"),
    (spiral(blend="add"), "blend 'add' is none of over, screen, multiply, soft-light"),
    (spiral(opacity=1.5), "opacity = 1.5: expected 0..1"),
    (spiral(mask="ground"), "needs the figure's `objects`"),
    (spiral(times=[0.5], lane="snare"), "one of them"),
    (spiral(min_db=-3), "give the `lane`"),
    ({"kind": "grid", "from": 0, "to": 1, "objects": "R*", "back": {"colors": ["rose", "base"]}},
     "a pattern back names its `pattern`"),
    ({"kind": "grid", "from": 0, "to": 1, "objects": "R*", "back": {"pattern": "rings", "mask": "ground"}},
     "unknown keys ['mask'] in a pattern back"),
    ({"kind": "grid", "from": 0, "to": 1, "objects": "R*", "back": ["love", {"pattern": "rings", "size": 0}]},
     "back 1 size = 0: expected 0.002..2"),
])
def test_bad_entries_say_what_is_wrong(spec, frag):
    with pytest.raises(FX.FxError, match=re.escape(frag)):
        FX.normalize(spec, DAWN)


def test_each_pattern_takes_its_own_keys_and_defaults():
    norm = lambda **kw: FX.normalize(spiral(**kw), DAWN)                                   # noqa: E731
    s = norm()
    assert (s["pattern"], s["arms"], s["twist"]) == ("spiral", 5, 1.0) and "size" not in s and "warp" not in s
    assert norm(pattern="sunburst")["arms"] == 12 and "twist" not in norm(pattern="sunburst")
    m = norm(pattern="moire", moire="lines", angle=4)
    assert (m["moire"], m["angle"], m["size"]) == ("lines", 4.0, 0.018)
    assert norm(colors=["love", "#ffffff", [0, 0, 1]])["colors"][1:] == [WHITE, BLUE]
    assert norm(center={"point": "bone('head').head"})["center"] == {"mode": "point", "expr": "bone('head').head"}
    back = FX.normalize({"kind": "orbit", "from": 0, "to": 1, "objects": "R*",
                         "back": [{"pattern": "checker", "center": [0.3, 0.6]}, "love"]}, DAWN)["back"]
    assert back[0]["pattern"] == "checker" and back[0]["center"] == [0.3, 0.6] and back[0]["warp"] == 0.6
    assert back[1] == [round(c, 6) for c in PAL.srgb(DAWN["love"])]


def _ring(pic, centre, radius, n=2000):
    """Colours sampled (bilinear) round a circle about `centre` (px), counter-clockwise on screen from the right."""
    a = np.linspace(0.0, 2.0 * math.pi, n, endpoint=False)
    xs = (centre[0] + radius * np.cos(a)).astype(np.float32)[None, :]
    ys = (centre[1] - radius * np.sin(a)).astype(np.float32)[None, :]
    return cv2.remap(pic, xs, ys, cv2.INTER_LINEAR)[0]


@pytest.mark.parametrize("arms, colors", [(3, [RED, BLUE]), (5, [RED, BLUE]), (4, [RED, BLUE, WHITE])])
def test_a_spiral_has_its_arms_and_their_symmetry(arms, colors):
    spec = {"pattern": "spiral", "colors": colors, "arms": arms, "twist": 1.5}
    size, c = (401, 401), (200.0, 200.0)
    pic = FL.pattern(spec, size, 0.0, c, 401)
    ring = _ring(pic, c, 150.0)
    nearest = np.argmin(((ring[:, None, :] - np.asarray(colors)[None]) ** 2).sum(-1), 1)
    changes = int(np.count_nonzero(nearest != np.roll(nearest, 1)))
    assert changes == arms * len(colors)                       # every colour once per arm round the circle
    turned = np.roll(ring, -len(ring) // arms, axis=0)        # a turn of 1 / arms of a circle: the same pattern
    assert np.mean(np.abs(turned - ring) > 0.5) < 0.01
    half = np.roll(ring, -len(ring) // (2 * arms), axis=0)    # half of that: not
    assert np.mean(np.abs(half - ring) > 0.5) > 0.3


def test_edges_are_antialiased_and_a_too_fine_middle_is_the_mean():
    spec = {"pattern": "sunburst", "colors": [BLACK, WHITE], "arms": 16}
    pic = FL.pattern(spec, (300, 300), 0.0, (149.5, 149.5), 300)
    v = pic[..., 0]
    grey_ = (v > 0.02) & (v < 0.98)
    assert 0.02 < grey_.mean() < 0.25                          # edges in between, the rays flat
    assert np.allclose(pic[..., 0], pic[..., 2])               # black to white through greys, no colour fringes
    assert abs(float(v[147:153, 147:153].mean()) - 0.5) < 0.05  # 32 rays in a few px: their mean, not noise
    rings = {"pattern": "rings", "colors": [BLACK, WHITE], "size": 0.05}
    a = FL.pattern(rings, (300, 300), 0.0, (149.5, 149.5), 300)
    b = FL.pattern(rings, (300, 300), 0.02, (149.5, 149.5), 300)   # the rings out by a third of a px
    assert 0.0 < float(np.abs(a - b).max()) < 0.45             # every edge px moves a little: none flips whole


@pytest.mark.parametrize("pattern", CF.PATTERNS)
def test_every_pattern_paints_its_colours_in_every_aspect(pattern):
    own = {"size": 0.06} if pattern in ("rings", "moire") else {}
    spec = FX.normalize(spiral(pattern=pattern, colors=["love", "base"], **own), DAWN)
    for size in ((192, 108), (108, 192), (120, 120)):
        pic = FL.pattern(spec, size, 0.0, (size[0] / 2 - 0.5, size[1] / 2 - 0.5), min(size))
        assert pic.shape == (size[1], size[0], 3) and pic.dtype == np.float32
        cs = np.asarray(spec["colors"], np.float32)
        near = np.min(np.abs(pic[:, :, None, :] - cs[None, None]).sum(-1), -1)
        assert (near < 0.02).mean() > 0.4                       # mostly its own colours, not a wash
        assert np.all((pic >= -1e-5) & (pic <= 1 + 1e-5))


def test_phase_flows_at_speed_and_steps_on_hits():
    spec = {"speed": 0.5, "step": 1.0}
    assert CF.phase(spec, 2.0, 1.0, []) == pytest.approx(0.5)
    assert CF.phase(spec, 1.6, 1.0, [1.4]) == pytest.approx(0.3 + 1.0)             # the step done after STEP
    mid = CF.phase(spec, 1.4 + CF.STEP / 2, 1.0, [1.4]) - 0.5 * (0.4 + CF.STEP / 2)
    assert 0.5 < mid < 1.0                                     # eased out: more than half way at half its time
    assert CF.phase(spec, 1.3, 1.0, [1.4]) == pytest.approx(0.15)                   # not before it


def test_a_step_moves_the_stripes_on_by_one():
    pl = plan(spiral(pattern="rings", colors=[RED, BLUE], speed=0.0, times=[1.0], size=0.1))
    r = runner(pl)
    before, after = r.frame(F0 + 25, grey()), r.frame(F0 + 40, grey())
    assert np.allclose(before[:, :, 0], after[:, :, 2], atol=1e-4)   # one stripe on: the two colours traded
    assert not np.allclose(before, after)


def test_it_turns_smoothly_from_frame_to_frame_and_the_same_in_any_order():
    pl = plan(spiral(pattern="spiral", colors=["love", "base"], speed=1.0))
    r = runner(pl)
    seq = [r.frame(F0 + 30 + j, grey()) for j in range(8)]
    steps = [float(np.abs(a - b).mean()) for a, b in zip(seq, seq[1:])]
    assert max(steps) < 0.06 and min(steps) > 0.0
    back = runner(pl)
    assert np.array_equal(back.frame(F0 + 37, grey()), seq[7]) and np.array_equal(back.frame(F0 + 30, grey()), seq[0])


def test_a_ground_mask_keeps_her_and_asks_for_her_layer():
    m = np.zeros((H, W), np.float32)
    m[10:40, 30:60] = 1.0
    pl = plan(spiral(mask="ground", objects="Reisen*"))
    img = np.random.default_rng(2).random((H, W, 3)).astype(np.float32)
    out = runner(pl, figure=m).frame(F0 + 10, img)
    assert np.array_equal(out[10:40, 30:60], img[10:40, 30:60])
    assert not np.allclose(out[45:, :], img[45:, :])
    assert FX.uses_figure(pl["fx"][0]) and TR.demands(pl)
    assert not TR.demands(plan(spiral()))


def test_blend_modes():
    img = np.full((2, 2, 3), 0.4, np.float32)
    pic = np.full((2, 2, 3), 0.5, np.float32)
    assert np.allclose(FL.blend(img, pic, 1.0, "over"), 0.5)
    assert np.allclose(FL.blend(img, pic, 1.0, "multiply"), 0.2)
    assert np.allclose(FL.blend(img, pic, 1.0, "screen"), 1 - 0.6 * 0.5)
    assert np.allclose(FL.blend(img, pic, 1.0, "soft-light"), 0.4)          # grey leaves it as it is
    assert np.allclose(FL.blend(img, pic, 0.0, "screen"), img)
    a = np.array([[0.0, 1.0], [0.5, 0.25]], np.float32)
    assert np.allclose(FL.blend(img, pic, a, "over")[..., 0], 0.4 + 0.1 * a)


# ---------------------------------------------------------------- pattern backs
def _figure():
    m = np.zeros((H, W), np.float32)
    m[18:40, 42:54] = 1.0
    return m


def test_a_grid_paints_each_cell_on_a_pattern_of_its_own():
    back = {"pattern": "rings", "colors": [RED, BLUE], "size": 0.08, "speed": 0.0}
    pl = plan({"kind": "grid", "from": 0.0, "to": 3.0, "objects": "Fig*", "cells": [[2, 2]], "back": back})
    img = np.zeros((H, W, 3), np.float32)
    img[..., 1] = 1.0                                          # her picture: green
    out = runner(pl, figure=_figure()).frame(F0 + 5, img)
    cells = [out[j * 27:(j + 1) * 27, i * 48:(i + 1) * 48] for j in range(2) for i in range(2)]
    for c in cells:
        assert np.allclose(c, cells[0], atol=1e-5)             # each cell its own pattern, centred in it
        corner = c[:4, :4]
        assert corner[..., 1].max() < 1e-6 and (corner[..., 0] + corner[..., 2]).min() > 0.99   # ground: the rings
    whole = FL.pattern(FX.normalize({"kind": "grid", "from": 0, "to": 1, "objects": "F", "back": back}, DAWN)["back"][0],
                       (48, 27), 0.0, (23.5, 13.0), 27)
    assert np.allclose(cells[0][:3], whole[:3], atol=1e-5)    # the cell's pattern at the cell's own size


def test_a_list_of_backs_mixes_patterns_and_colours_and_a_pattern_back_steps_on_the_hits():
    back = [{"pattern": "rings", "colors": [RED, BLUE], "size": 0.1, "speed": 0.0}, WHITE]
    pl = plan({"kind": "grid", "from": 0.0, "to": 3.0, "objects": "Fig*", "cells": [[2, 1]], "back": back,
               "times": [1.0]})
    r = runner(pl, figure=_figure())
    img = np.zeros((H, W, 3), np.float32)
    first = r.frame(F0 + 20, img)
    assert np.allclose(first[:6, 48:54], 1.0)                  # cell (1, 0): white
    assert first[:6, :6, 1].max() < 1e-6                      # cell (0, 0): the rings
    late = r.frame(F0 + 50, img)                              # after the hit the list moves on a step
    assert np.allclose(late[:6, :6], 1.0)
    single = plan({"kind": "orbit", "from": 0.0, "to": 3.0, "objects": "Fig*", "times": [1.0],
                   "back": {"pattern": "rings", "colors": [RED, BLUE], "size": 0.1, "speed": 0.0}})
    o = runner(single, figure=np.zeros((H, W), np.float32))
    a, b = o.frame(F0 + 20, img), o.frame(F0 + 50, img)
    assert np.allclose(a[..., 0], b[..., 2], atol=1e-4)       # one back, its stripes stepped on by one


@pytest.mark.parametrize("kind", ["orbit", "mirror", "panels", "stack"])
def test_every_flat_back_collage_takes_a_pattern(kind):
    back = {"pattern": "checker", "colors": [RED, BLUE], "speed": 0.0}
    pl = plan({"kind": kind, "from": 0.0, "to": 3.0, "objects": "Fig*", "back": back})
    out = runner(pl, figure=_figure()).frame(F0 + 5, np.zeros((H, W, 3), np.float32))
    ground = out[:8, :8]
    assert ground[..., 1].max() < 1e-6 and ground[..., 0].std() > 0.1     # a checkerboard, not a flat colour


def test_a_contained_clip_sits_on_a_pattern_back(tmp_path):
    path = tmp_path / "clip.mp4"
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 24.0, (32, 32))
    for _ in range(30):
        vw.write(np.full((32, 32, 3), 128, np.uint8))
    vw.release()
    back = {"pattern": "checker", "colors": [RED, BLUE], "size": 0.2, "warp": 0.0, "speed": 0.0}
    pl = plan({"kind": "footage", "from": 0.0, "to": 1.0, "objects": "Fig*", "clip": str(path), "fit": "contain",
               "back": back})
    r = FF.FrameFx(pl["fx"], (W, H), fps=FPS, frame0=F0, frame_of=lambda g: None, point_of=None,
                   figure_of=lambda key, g: np.zeros((H, W), np.float32), footage_of=lambda c: FF.FT.Clip(c))
    out = r.frame(F0 + 3, np.zeros((H, W, 3), np.float32))
    assert np.allclose(out[:, 40:56], 128 / 255, atol=0.03)               # the square clip in the middle
    side = out[:, :18]
    assert side[..., 1].max() < 1e-6 and side[..., 0].std() > 0.2        # round it, the checkerboard
