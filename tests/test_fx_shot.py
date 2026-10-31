"""A collage's figure taken from another shot's own render (`shot`): the plates and the figure through that shot's camera
that `mk render` draws and `mk look` previews (mkmmd.core.transition), the figure set into the frame by `at` and `height`
about its pivot by one map for the whole span (mkmmd.framefx), and a stack of it echoing and mirrored about a point of the
frame it is laid in, on synthetic pictures."""
import re

import numpy as np
import pytest
from PIL import Image

from mkmmd import framefx as FF
from mkmmd import postframes as PF
from mkmmd.core import framefx as FX
from mkmmd.core import palette as PAL
from mkmmd.core import transition as TR

FPS, F0 = 30, 1
MOON = PAL.get("rose-pine-moon")
W, H = 64, 36
RED, BLACK = [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]


def project(fx=(), **extra):
    return {"project": {"duration": 4.0}, "output": [{"name": "16x9", "size": [W, H]}, {"name": "9x16", "size": [H, W]}],
            "shot": [{"name": "a", "from": 0.0, "to": 2.0}, {"name": "b", "from": 2.0, "to": 4.0},
                     {"name": "far", "plate": True}],
            "fx": list(fx), **extra}


def plan(fx=(), **extra):
    return TR.plan(project(fx, **extra), FPS, F0, MOON)


def stack(**kw):
    return {"kind": "stack", "from": 1.0, "to": 1.5, "objects": "Gui*", "shot": "far", **kw}


def guitarist(arm=0, size=(W, H), x=10):
    """Her coverage in her own shot's frame: a body 6 px wide and 21 tall standing on row 30 at columns x .. x + 5, with
    an arm `arm` px long held out to the right on rows 12 and 13 (her box grows as it strums out)."""
    m = np.zeros((size[1], size[0]), np.float32)
    m[10:31, x:x + 6] = 1.0
    if arm:
        m[12:14, x + 6:x + 6 + arm] = 1.0
    return m


def ramp(size=(W, H)):
    """Her shot's picture: red is the column, green the row (0..1), so where a pixel came from can be read off it."""
    xx, yy = np.meshgrid(np.linspace(0, 1, size[0], dtype=np.float32), np.linspace(0, 1, size[1], dtype=np.float32))
    return np.dstack([xx, yy, np.zeros_like(xx)])


def runner(pl, figure, size=(W, H), output=None, plate=None, point=None):
    return FF.FrameFx(pl["fx"], size, fps=FPS, frame0=F0, frame_of=lambda g: None, output=output,
                      point_of=lambda key, g: (point, None), figure_of=lambda key, g: figure(g),
                      plate_of=(lambda shot, g: ramp(size)) if plate is None else plate)


def box(cov):
    """(x0, y0, x1, y1) of the pixels a coverage covers half or more of."""
    ys, xs = np.nonzero(cov > 0.5)
    return xs.min(), ys.min(), xs.max(), ys.max()


# ---------------------------------------------------------------- the entries and the layers
@pytest.mark.parametrize("spec, frag", [
    (stack(shot=3), "expected the name of a [[shot]]"),
    (stack(shot="nope"), "there is no [[shot]] of that name"),
    (stack(shot="a"), "is the shot in the cut on frames 31..45"),
    (stack(at=0.5), "at = 0.5: expected [x, y]"),
    (stack(height=0), "height = 0: expected 0.01..10"),
    (stack(aspect={"9x16": {"shot": "b"}}), "may not set ['shot']"),
])
def test_bad_entries_say_what_is_wrong(spec, frag):
    with pytest.raises(TR.TransitionError, match=re.escape(frag)):
        plan([spec])


def test_its_plates_and_its_figure_through_its_shot_are_drawn_on_every_frame_it_reads_and_its_camera_keyed_there():
    pl = plan([stack(count=2, delay=3), {"kind": "grid", "from": 1.0, "to": 1.1, "objects": "Gui*"}])
    fx, plain = pl["fx"]
    assert FX.figure_key(fx) != FX.figure_key(plain)            # the same objects through another camera: another layer
    figure = {"kind": "figure", "key": FX.figure_key(fx), "objects": ["Gui*"], "shot": "far"}
    dem = TR.demands(pl)
    reads = set(range(fx["f0"] - 6, fx["f1"]))                  # its span, 31 .. 45, and the poses 3 and 6 frames before
    for g in reads:
        assert {"kind": "plate", "shot": "far"} in dem[g] and figure in dem[g]
    assert not [g for g, items in dem.items() if g not in reads and any(it.get("shot") == "far" for it in items)]
    assert {"kind": "figure", "key": FX.figure_key(plain), "objects": ["Gui*"], "shot": "a"} in dem[31]
    assert TR.needs(pl)["far"] == [25, 45]                      # mk build keys its camera over them
    assert TR.fx_frames(pl, [33]) == {30: [figure, {"kind": "plate", "shot": "far"}],    # mk look draws its poses
                                      27: [figure, {"kind": "plate", "shot": "far"}]}    # before, not the cut's frames


def test_placed_without_keys_it_still_reads_its_pivot_on_the_span_s_first_frame():
    fx = plan([stack(at=[0.3, 0.9])])["fx"][0]
    assert FX.sources(fx, fx["f0"] + 4, F0) == [fx["f0"]]
    assert FX.sources(plan([stack()])["fx"][0], fx["f0"] + 4, F0) == []                 # laid where her shot shows her
    assert FX.sources(plan([stack(at=[0.3, 0.9], pivot=[0.2, 0.8])])["fx"][0], fx["f0"] + 4, F0) == []
    only = plan([stack(aspect={"9x16": {"at": [0.5, 0.9], "height": 0.3}})])["fx"][0]   # one output sets her elsewhere:
    assert FX.sources(only, fx["f0"] + 4, F0) == [fx["f0"]]                             # the entry reads that frame too
    assert not FX.placed(only) and FX.placed(only["aspect"]["9x16"])


def test_placement_scales_about_the_anchor_and_puts_it_where_it_is_asked_and_composes_with_a_move():
    P = FX.placement((12.5, 30.5), (15.5, 31.9), 0.5)
    assert np.allclose(np.array(P) @ [12.5, 30.5, 1.0], [15.5, 31.9])
    assert np.allclose(np.array(P) @ [14.5, 30.5, 1.0], [16.5, 31.9])                  # half as large about it
    K = FX.move_matrix({"turn": 90.0, "dx": 0.0, "dy": 0.0, "scale": 1.0}, (15.5, 31.9), H)
    C = np.array(FX.compose(K, P))
    assert np.allclose(C @ [12.5, 30.5, 1.0], [15.5, 31.9])
    assert np.allclose(C @ [14.5, 30.5, 1.0], [15.5, 30.9])                            # then turned about it


# ---------------------------------------------------------------- the pictures
def test_it_is_set_into_the_frame_by_its_feet_at_its_height_by_one_map_so_strumming_never_moves_it():
    pl = plan([stack(kind="grid", cells=1, back="#000000", fill="#FF0000", at=[0.25, 0.9], height=0.5)])
    f0 = pl["fx"][0]["f0"]
    arms = {f0: 0, f0 + 5: 8}                                   # the arm strums out: her box changes, her feet do not
    run = runner(pl, lambda g: guitarist(arms[g]))
    host = np.full((H, W, 3), 0.5, np.float32)
    a, b = run.frame(f0, host)[..., 0], run.frame(f0 + 5, host)[..., 0]
    x0, y0, x1, y1 = box(a)
    assert y1 - y0 + 1 == pytest.approx(0.5 * H, abs=1.0)        # half the frame's height
    assert ((x0 + x1) / 2.0, y1 + 0.5) == pytest.approx((0.25 * W - 0.5, 0.9 * H - 0.5), abs=0.75)   # feet at `at`
    assert np.array_equal(a[:, :17], b[:, :17])                 # the body: the same pixels on every frame
    assert a[:, 20:].max() == 0.0 and b[16:18, 20:24].min() > 0.5     # and the arm out beside it


def test_its_picture_is_its_own_shot_s_scaled_with_it_and_its_poses_before_come_from_that_shot_too():
    pl = plan([stack(back="#000000", count=1, delay=3, bands=["picture"], step=[0.25, 0.0], at=[0.25, 0.9],
                     height=0.5)])
    f0 = pl["fx"][0]["f0"]
    plates = {f0 + 5: ramp(), f0 + 2: ramp()[:, ::-1].copy()}   # 3 frames before, her shot showed another picture
    run = runner(pl, lambda g: guitarist(), plate=lambda shot, g: plates[g] if shot == "far" else None)
    out = run.frame(f0 + 5, np.full((H, W, 3), 0.5, np.float32))
    s, anchor, to = 0.5 * H / 21, (12.5, 30.5), (0.25 * W - 0.5, 0.9 * H - 0.5)
    y = 20
    src_y = anchor[1] + (y - to[1]) / s                           # the row of her shot a row of the frame shows
    assert out[y, 14:18, 1] == pytest.approx(np.full(4, src_y / (H - 1)), abs=0.03)
    x = 14 + 9                                                    # the copy, 0.25 x 36 = 9 px to the right
    src_x = anchor[0] + (14 - to[0]) / s
    assert out[y, x, 0] == pytest.approx(1.0 - src_x / (W - 1), abs=0.04)      # from the picture 3 frames before


def test_each_output_sets_it_where_its_aspect_table_says():
    pl = plan([stack(kind="grid", cells=1, back="#000000", fill="#FF0000", at=[0.25, 0.9], height=0.5,
                     aspect={"9x16": {"at": [0.5, 0.45], "height": 0.25}})])
    f0 = pl["fx"][0]["f0"]
    tall = runner(pl, lambda g: guitarist(size=(H, W), x=15), size=(H, W), output="9x16")
    x0, y0, x1, y1 = box(tall.frame(f0, np.zeros((W, H, 3), np.float32))[..., 0])
    assert y1 - y0 + 1 == pytest.approx(0.25 * W, abs=1.0)
    assert ((x0 + x1) / 2.0, y1 + 0.5) == pytest.approx((0.5 * H - 0.5, 0.45 * W - 0.5), abs=0.75)


def test_its_stack_echoes_and_mirrors_about_a_point_of_the_frame_it_is_laid_in_over_the_picture():
    pl = plan([stack(back="picture", mirror=True, center={"point": "bone('chest').head"}, count=2, step=[-4 / 36, 0.0],
                     bands=["#0000FF", "#00FF00"], fill="#FF0000", at=[0.25, 0.9], height=0.5)])
    f0 = pl["fx"][0]["f0"]
    run = runner(pl, lambda g: guitarist(), point=(31.5, 17.5))
    host = np.full((H, W, 3), 0.5, np.float32)
    out = run.frame(f0, host)
    assert np.allclose(out, out[:, ::-1], atol=1e-5)            # her and her mirror image about the frame's middle
    row = out[22]
    reds = np.nonzero(np.all(np.abs(row - RED) < 0.02, -1))[0]
    blues = np.nonzero(np.all(np.abs(row - [0.0, 0.0, 1.0]) < 0.02, -1))[0]
    left, right = reds[reds < 32], reds[reds >= 32]
    assert blues[blues < 32].max() < left.min() and blues[blues >= 32].min() > right.max()   # both echoes trail outward
    assert np.array_equal(out[:2], host[:2])                    # the picture round them as it is


def test_without_its_plates_it_says_what_is_missing():
    pl = plan([stack()])
    f0 = pl["fx"][0]["f0"]
    bare = FF.FrameFx(pl["fx"], (W, H), fps=FPS, frame0=F0, frame_of=lambda g: None, point_of=None,
                      figure_of=lambda key, g: guitarist())
    with pytest.raises(FileNotFoundError, match="no plates to read"):
        bare.frame(f0, np.zeros((H, W, 3), np.float32))
    gone = runner(pl, lambda g: guitarist(), plate=lambda shot, g: None)
    with pytest.raises(FileNotFoundError, match=r"shot 'far' at frame 31: its plate is missing"):
        gone.frame(f0, np.zeros((H, W, 3), np.float32))


def test_mk_post_reads_a_shot_from_its_plates_and_from_the_cut_s_own_frames_where_it_is_in_the_cut(tmp_path):
    pl = plan([stack()])
    Image.new("RGB", (W, H), (255, 0, 0)).save(tmp_path / "00080.png")                  # the cut's frame 80: shot b
    (tmp_path / "plate" / "far").mkdir(parents=True)
    Image.new("RGB", (W, H), (0, 0, 255)).save(tmp_path / TR.plate_rel("far", 40))
    get = PF.plates(pl, tmp_path, {80: tmp_path / "00080.png"}, (W, H))
    assert get("far", 40)[0, 0].tolist() == [0.0, 0.0, 1.0] and get("b", 80)[0, 0].tolist() == [1.0, 0.0, 0.0]
    assert get("far", 41) is None
