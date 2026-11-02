"""Op-art ([[fx]] kind "spiral", mkmmd.core.fields and mkmmd.fields) and the collages' pattern backs: the keys and what
they refuse, each pattern's symmetry (a spiral's arms), antialiased edges and a too-fine middle drawn as the colours' mean,
turning on `speed` and stepping on hits, smooth from frame to frame, the same in any order, kept behind her by a ground
mask, and a grid's cells each on a pattern of its own that steps on the grid's hits. The drift: its tones taken by luma,
the darkest, dark, lightest, light steps the same way round every motif and the other way on the next ring or row, a
whole number of motifs round a ring, its creep and its paced steps, and the GPU leaving it to the CPU."""
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
    (spiral(pattern="zigzag"), "pattern 'zigzag' is none of spiral, rings, moire, sunburst, checker, drift"),
    (spiral(colors=["rose"]), "expected a list of 2 to 4 colours"),
    (spiral(colors=["rose", "gold", "pine", "foam", "iris"]), "2 to 4 colours"),
    (spiral(colors="rose"), "expected a list of 2 to 4 colours"),
    (spiral(pattern="rings", twist=2), "twist (a key of spiral) does not shape a rings: it takes size"),
    (spiral(pattern="spiral", size=0.1), "size (a key of rings or moire or checker or drift) does not shape a spiral"),
    (spiral(pattern="drift", arms=3), "arms (a key of spiral or sunburst) does not shape a drift: it takes angle, "
                                      "layout, motif, size"),
    (spiral(pattern="drift", layout="grid"), "layout 'grid' is none of rings, rows"),
    (spiral(pattern="drift", motif="egg"), "motif 'egg' is none of tie, wheel"),
    (spiral(pattern="rings", motif="tie"), "motif (a key of drift) does not shape a rings: it takes size"),
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
    own = {"size": 0.06} if pattern in ("rings", "moire") else {"size": 0.3} if pattern == "drift" else {}
    spec = FX.normalize(spiral(pattern=pattern, colors=["love", "base"], **own), DAWN)
    for size in ((192, 108), (108, 192), (120, 120)):
        pic = FL.pattern(spec, size, 0.0, (size[0] / 2 - 0.5, size[1] / 2 - 0.5), min(size))
        assert pic.shape == (size[1], size[0], 3) and pic.dtype == np.float32
        cs = np.asarray(FL.drift_tones(spec["colors"]) if pattern == "drift" else spec["colors"], np.float32)
        near = np.min(np.abs(pic[:, :, None, :] - cs[None, None]).sum(-1), -1)
        assert (near < 0.02).mean() > 0.4                       # mostly its own colours (a drift's tones), not a wash
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


# ---------------------------------------------------------------- the drift
TONES = [BLACK, BLUE, [0.0, 1.0, 0.0], WHITE]       # darkest, dark, light, lightest: no mix of two looks like a third
CYCLE = [2, 0, 1, 3]                                 # forward through a motif: the light ground, darkest, dark, lightest


def _seq(vals):
    """The tones (indices into TONES) of the px `vals` (n, 3), runs merged, px between tones (edges) left out."""
    d = np.abs(np.asarray(vals)[:, None, :] - np.asarray(TONES, np.float32)[None]).sum(-1)
    idx = np.argmin(d, 1)
    out = []
    for i in idx[d[np.arange(len(idx)), idx] < 0.02]:
        if not out or out[-1] != i:
            out.append(int(i))
    return out


def _cyclic(seq, cycle):
    """Whether `seq` runs round `cycle` (each entry followed by the next of the cycle), at least twice round."""
    return len(seq) >= 8 and all(cycle[(cycle.index(a) + 1) % 4] == b for a, b in zip(seq, seq[1:]))


def test_drift_takes_its_tones_by_luma_whatever_their_order_and_palette():
    want = [np.asarray(c, np.float32) for c in TONES]
    for order in ([3, 0, 2, 1], [1, 2, 3, 0], [0, 1, 2, 3]):
        got = FL.drift_tones([TONES[i] for i in order])
        assert all(np.allclose(g, e) for g, e in zip(got, want))
    two = FL.drift_tones([WHITE, BLACK])                                  # mixed between: 0.3 and 0.7 of the way
    assert [round(float(t[0]), 3) for t in two] == [0.0, 0.3, 0.7, 1.0]
    dark = FL.drift_tones([WHITE, [0.2] * 3, BLACK])                      # the middle one nearer black: the dark tone
    assert np.allclose(dark[1], 0.2) and np.allclose(dark[2], 0.6)
    light = FL.drift_tones([WHITE, [0.8] * 3, BLACK])                     # nearer white: the light one
    assert np.allclose(light[2], 0.8) and np.allclose(light[1], 0.4)
    for name in ("rose-pine-moon", "rose-pine-dawn"):                    # base and text trade places: still in order
        spec = FX.normalize(spiral(pattern="drift"), PAL.get(name))
        assert (spec["speed"], spec["step"], spec["layout"], spec["motif"]) == (0.0, 0.25, "rings", "tie")
        lum = [float(np.asarray(t) @ FL.LUMA) for t in FL.drift_tones(spec["colors"])]
        assert lum == sorted(lum) and len(spec["colors"]) == 4
    other = FX.normalize(spiral(pattern="rings"), DAWN)
    assert (other["speed"], other["step"]) == (0.5, 1.0)                  # the others keep the shared defaults


@pytest.mark.parametrize("motif", CF.DRIFT_MOTIFS)
def test_drift_steps_run_one_way_round_every_motif_and_the_other_way_on_the_next_row(motif):
    spec = {"pattern": "drift", "colors": TONES, "layout": "rows", "motif": motif, "size": 0.1, "angle": 0.0}
    pic = FL.pattern(spec, (600, 300), 0.0, (299.5, 149.5), 300)        # rows 30 px wide, row 0 about y = 149.5
    row0, row1 = pic[149 + 5], pic[179 + 5]                              # 5 px off their middles: clear of a wheel's hub
    assert _cyclic(_seq(row0), CYCLE)                                    # left to right: forward on row 0
    assert _cyclic(_seq(row1), CYCLE[::-1])                              # backward on row 1
    turned = FL.pattern({**spec, "angle": 90.0}, (300, 600), 0.0, (149.5, 299.5), 300)
    assert _cyclic(_seq(turned[::-1, 149 + 5]), CYCLE)                   # rows turned up the frame: forward is up


def test_drift_rings_hold_a_whole_number_of_motifs_turning_alternate_ways():
    spec = {"pattern": "drift", "colors": TONES, "layout": "rings", "motif": "tie", "size": 0.1, "angle": 0.0}
    pic = FL.pattern(spec, (401, 401), 0.0, (200.0, 200.0), 300)        # rings 30 px wide from a hub of 15 px
    a = np.linspace(0.0, 2.0 * math.pi, 4000, endpoint=False)           # counter-clockwise on screen from the right
    for k, cycle in ((1, CYCLE[::-1]), (2, CYCLE)):
        rad = (k + 0.5 + FL.DRIFT_HUB) * 30.0 + 5.0
        xs = np.rint(200.0 + rad * np.cos(a)).astype(int)
        ys = np.rint(200.0 - rad * np.sin(a)).astype(int)
        seq = _seq(pic[ys, xs])
        assert _cyclic(seq, cycle)                                       # odd rings turn the other way
        n = round(2.0 * math.pi * (k + 0.5 + FL.DRIFT_HUB) * 30.0 / (30.0 * FL.DRIFT["tie"]["pitch"]))
        assert seq.count(0) in (n, n + 1)                                # a whole number round: no seam at the left
    assert np.allclose(pic[195:206, 195:206], TONES[2])                  # the hub: the light ground


def test_drift_creeps_each_row_its_own_way_and_steps_three_times_a_second_at_most():
    spec = {"pattern": "drift", "colors": TONES, "layout": "rows", "motif": "tie", "size": 0.16, "angle": 0.0}
    a = FL.pattern(spec, (600, 300), 0.0, (299.5, 149.5), 300)          # rows 48 px, ties every 24 px
    b = FL.pattern(spec, (600, 300), 0.25, (299.5, 149.5), 300)         # a quarter of a cell on: 6 px
    assert np.allclose(b[150, 100:500], a[150, 94:494], atol=1e-4)      # row 0 forward: right
    assert np.allclose(b[198, 100:500], a[198, 106:506], atol=1e-4)     # row 1 forward: left
    drift = FX.normalize(spiral(pattern="drift", step=1.0, times=[1.0]), DAWN)
    assert CF.phase(drift, 2.0, 0.0, [1.0, 1.1, 1.2, 1.4]) == pytest.approx(2.0)   # 1.1 and 1.2 too soon
    rings = FX.normalize(spiral(pattern="rings", speed=0.0, times=[1.0]), DAWN)
    assert CF.phase(rings, 2.0, 0.0, [1.0, 1.1, 1.2, 1.4]) == pytest.approx(4.0)   # the others step on every hit
    pl = plan(spiral(pattern="drift", colors=TONES, size=0.3, times=[1.0, 1.1]))
    paced = plan(spiral(pattern="drift", colors=TONES, size=0.3, times=[1.0]))
    assert np.array_equal(runner(pl).frame(F0 + 50, grey()), runner(paced).frame(F0 + 50, grey()))


def test_a_drift_too_fine_for_its_pixels_is_its_tones_mean_and_the_gpu_leaves_it_to_the_cpu():
    from mkmmd.gpufx import fields as GFL
    from mkmmd.gpufx import spiral as GS
    tiny = FL.pattern({"pattern": "drift", "colors": TONES, "layout": "rings", "motif": "wheel", "size": 0.004,
                       "angle": 0.0}, (200, 200), 0.0, (99.5, 99.5), 200)
    mean = sum(np.float32(s) * np.asarray(c, np.float32) for s, c in zip(FL.DRIFT_AREA, TONES))
    assert np.abs(tiny[30:170, 30:170] - mean).max() < 0.02
    pl = plan(spiral(pattern="drift"), spiral(pattern="checker"))
    r = runner(pl)
    assert GS.ops(r, r.fxs[0], F0 + 3) is None and GS.ops(r, r.fxs[1], F0 + 3)[0]["op"] == "spiral"
    grid = FX.normalize({"kind": "stack", "from": 0, "to": 1, "objects": "R*",
                         "back": ["love", {"pattern": "drift"}]}, DAWN)
    assert not GFL.backs_drawable(grid) and GFL.backs_drawable({**grid, "back": grid["back"][:1]})


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
