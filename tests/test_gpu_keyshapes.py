"""mkr-post's shapes and keys against the CPU path (mkmmd.gpufx: shapes, key) on synthetic pictures: a case for each
branch of the ops and shaders (each shape, ink pick, back, mask, pop; each distance, plane matte, band, spill, back and
burn), each frame drawn by both and compared within a level (tests/gpu_fixtures.py). Skipped without the mkr-post binary
or a GPU."""
import cv2
import numpy as np
import pytest

import gpu_fixtures as GF
from mkmmd import footage as FT
from mkmmd import gpufx as GFX
from mkmmd.core import framefx as FX
from mkmmd.core import palette as PAL
from mkmmd.core import transition as TR

FPS, F0 = 30, 1
SMALL = (192, 108)
MOON = PAL.get("rose-pine-moon")
CREAM, PLUM = (0.98, 0.96, 0.93), (0.34, 0.32, 0.47)
WORST = {}                      # test id -> the largest level difference it saw (the report: pytest -s)


def plan(*fx, duration=3.0, size=SMALL):
    data = {"project": {"duration": duration}, "output": [{"name": "t", "size": list(size)}],
            "shot": [{"name": "a", "from": 0.0, "to": duration}], "fx": list(fx)}
    return TR.plan(data, FPS, F0, MOON)


def picture(seed, size=SMALL):
    """Smooth colour noise with hard-edged blocks over it."""
    w, h = size
    rng = np.random.default_rng(seed)
    t = cv2.GaussianBlur(rng.random((h, w, 3)).astype(np.float32), (0, 0), 2.0)
    t = (t - t.min()) / (t.max() - t.min())
    blocks = (rng.random((h // 6 + 1, w // 6 + 1, 3)) > 0.6).astype(np.float32)
    hard = np.kron(blocks, np.ones((6, 6, 1), np.float32))[:h, :w]
    return np.ascontiguousarray(np.where(hard > 0, 1.0 - t, t), np.float32)


def on_cream(seed, size=SMALL):
    """A picture of colour noise on a flat cream ground (a disc of the noise in its middle, a bar of it at the side)."""
    w, h = size
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    inside = (np.hypot(xx - w * 0.45, yy - h * 0.5) < h * 0.36) | (xx > w * 0.85)
    return np.where(inside[..., None], picture(seed, size), np.array(CREAM, np.float32)).astype(np.float32)


def disc(g, size=SMALL):
    """A figure's coverage on frame g: an antialiased disc walking across."""
    w, h = size
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cx, cy, r = w * (0.3 + 0.005 * (g % 60)), h * 0.5, h * 0.3
    return np.clip(r + 0.5 - np.hypot(xx - cx, yy - cy), 0.0, 1.0)


@pytest.fixture(scope="module")
def gpu(tmp_path_factory):
    g = GF.GPU(SMALL, tmp_path_factory.mktemp("gpu"))
    yield g
    g.close()


def check(gpu, folder, spec, frames_to, pic=picture, masked=False, frames=None, **kw):
    """Every frame of `frames_to` drawn by both paths over `pic(1000 + frame)`, within a level; and the effect changed
    the picture on some of them. `masked`: the entry's figure layers are disc(g); `frames` the cut's frames ({g:
    picture}); `kw` go to the FrameFx. The largest level difference."""
    pl = plan(spec)
    figures = None
    if masked:
        key = FX.figure_key(pl["fx"][0])
        figures = {(key, g): disc(g) for g in range(F0, F0 + 100)}
    ffx = GF.frame_fx(pl["fx"], SMALL, folder, frames=frames or {g: picture(g) for g in range(F0, F0 + 100)},
                      figures=figures, fps=FPS, frame0=F0, hits=kw.pop("hits", {}), **kw)
    fx = ffx.fxs[0]
    drew, most = 0, 0
    for f in frames_to:
        img = pic(1000 + f)
        worst, over = gpu.check(ffx, fx, f, img)
        assert worst <= 1 and over == 0, f"{spec['kind']} frame {f}: {worst} levels, {over} pixels off by more than 1"
        most = max(most, worst)
        drew += int(np.abs(ffx._apply(fx, f, img) - img).max() > 1.0 / 255.0)
    assert drew, f"{spec['kind']}: changed no frame"
    return most


def frames_of(spec):
    fx = plan(spec)["fx"][0]
    return list(range(fx["f0"], fx["f1"]))


def note(request, worst):
    WORST[request.node.name] = worst
    print(f"\n{request.node.name}: max level difference {worst}")


# ================================================================================================= shapes
@pytest.mark.parametrize("keys, pic", [
    ({"cell": 0.12, "shape": ["circle", "triangle", "square", "diamond"], "turn": 40.0, "spin": 90.0, "size": 0.8},
     picture),
    ({"cell": 0.3, "shape": "triangle", "layout": "grid", "turn": 180.0, "size": 1.0}, picture),     # a box blur per group
    ({"cell": [[0.0, 0.2], [1.0, 0.06]], "shape": ["square", "diamond"], "layout": "grid", "seed": 5}, picture),
    ({"cell": 0.5, "shape": ["circle", "diamond"], "size": 0.9, "turn": 30.0}, picture),
    ({"cell": 0.03, "shape": "circle", "size": 0.7}, picture),                                      # 2 px of box
    ({"cell": 0.1, "shape": ["circle", "square"], "inks": ["#1f1d2e", "#d7827e", "#f6c177", "#56949f"]}, picture),
    ({"cell": 0.1, "shape": ["circle", "square", "triangle"], "pick": "shape", "turn": 20.0,
      "inks": ["#1f1d2e", "#d7827e", "#f6c177"]}, picture),
    ({"cell": 0.1, "shape": ["circle", "square"], "pick": "deal", "inks": ["#1f1d2e", "#d7827e"], "seed": 11}, picture),
    ({"cell": 0.1, "back": list(CREAM), "shape": ["circle", "square"]}, on_cream),                   # a flat back, skip
    ({"cell": 0.1, "back": list(CREAM), "skip": 0.2, "inks": ["#1f1d2e", "#d7827e"], "layout": "grid"}, on_cream),
    ({"cell": 0.1, "back": "pine", "shape": "diamond", "strength": [[0.0, 0.3], [1.0, 1.0]]}, picture),
])
def test_shapes(gpu, tmp_path, request, keys, pic):
    spec = {"kind": "shapes", "from": 0.0, "to": 1.0, "pop": 0.0, **keys}
    note(request, check(gpu, tmp_path, spec, frames_of(spec)[::6], pic=pic))


@pytest.mark.parametrize("keys", [
    {"cell": 0.1, "times": [0.2, 0.6], "pop": 0.3, "stagger": 0.2, "size": 1.0},       # popping in past the cell's size
    {"cell": [[0.0, 0.3], [1.0, 0.05]], "times": [0.5], "pop": 0.25, "back": list(CREAM), "turn": 25.0},
])
def test_shapes_pop_in_on_the_hits(gpu, tmp_path, request, keys):
    spec = {"kind": "shapes", "from": 0.0, "to": 1.0, **keys}
    pic = on_cream if "back" in keys else picture
    note(request, check(gpu, tmp_path, spec, frames_of(spec)[::2], pic=pic))


@pytest.mark.parametrize("keys, pic", [
    ({"cell": [[0.0, 0.3], [1.0, 0.07]], "times": [0.2, 0.6], "pop": 0.3, "stagger": 0.25}, picture),
    ({"cell": [[0.0, 0.1], [1.0, 0.25]], "times": [0.3], "pop": 0.2, "stagger": 0.3, "back": list(CREAM),
      "inks": ["#1f1d2e", "#d7827e", "#f6c177"], "turn": 30.0, "spin": 40.0, "layout": "grid"}, on_cream),
])
def test_shapes_keep_the_old_layout_until_the_new_shapes_pop(gpu, tmp_path, request, keys, pic):
    """Every frame from a hit while its shapes pop in: the layout before shows where a new cell has not begun."""
    spec = {"kind": "shapes", "from": 0.0, "to": 1.0, **keys}
    fx = plan(spec)["fx"][0]
    first = F0 + int(round(keys["times"][0] * FPS))
    note(request, check(gpu, tmp_path, spec, [f for f in range(first - 1, first + 22) if f < fx["f1"]], pic=pic))


@pytest.mark.parametrize("mask", ["figure", "ground"])
def test_shapes_under_a_mask(gpu, tmp_path, request, mask):
    spec = {"kind": "shapes", "from": 0.0, "to": 1.0, "cell": 0.12, "pop": 0.0, "mask": mask, "objects": "a*",
            "strength": 0.7}
    note(request, check(gpu, tmp_path, spec, frames_of(spec)[::6], masked=True))


# ================================================================================================= key
GREEN = (0.1, 0.6, 0.3)


def ramp(seed, size=SMALL):
    """The key colour fading across the frame into colour noise: every distance from it shows, with hard blocks."""
    w, _ = size
    t = np.linspace(0.0, 1.0, w, dtype=np.float32)[None, :, None]
    return np.ascontiguousarray(np.array(CREAM, np.float32) * (1.0 - t) + picture(seed, size) * t, np.float32)


def cut(missing=(F0 + 7,)):
    """The cut's frames: a ramp of their own each, but for the ones never rendered."""
    return {g: ramp(g) for g in range(F0, F0 + 100) if g not in missing}


def clip(path, n=40, fps=24.0, size=(192, 108)):
    """A clip whose frames differ (a bar sweeping over colour noise)."""
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    assert vw.isOpened()
    base = (picture(77, size) * 255.0).astype(np.uint8)
    for i in range(n):
        f = base.copy()
        x = (7 * i) % size[0]
        f[:, x:x + 9] = (40 * i % 255, 255 - 3 * i % 255, 128)
        vw.write(f)
    vw.release()
    return path


def key(**kw):
    return {"kind": "key", "from": 0.0, "to": 1.0, "color": list(CREAM), "back": list(GREEN),
            "tolerance": [[0.0, 0.08], [1.0, 0.35]], **kw}


@pytest.mark.parametrize("keys", [
    {},                                                                                  # rgb, soft, a flat back
    {"space": "chroma", "soft": 0.05, "split": 0.03, "gain": [1.4, 0.7, 1.2]},
    {"space": "luma", "tolerance": [[0.0, 0.05], [1.0, 0.25]], "soft": 0.2, "bands": 4},
    {"soft": 0.0, "split": [0.04, 0.0, -0.06]},                                         # a hard step, planes apart
    {"bands": 3, "soft": 0.15, "split": -0.05, "strength": [[0.0, 0.4], [1.0, 1.0]]},
    {"spill": {"color": "#d7827e", "amount": 0.7, "reach": 0.2}},
    {"spill": "gold", "space": "chroma", "soft": 0.0},
    {"gain": [2.0, 0.5, 1.0], "color": "#f2e9e1", "back": "#2a273f"},
])
def test_key(gpu, tmp_path, request, keys):
    spec = key(**keys)
    note(request, check(gpu, tmp_path, spec, frames_of(spec)[::6], pic=ramp, frames=cut()))


@pytest.mark.parametrize("keys", [
    {"burn": 3, "fade": 0.6},
    {"burn": 5, "fade": 0.85, "bands": 4, "soft": 0.2, "space": "chroma", "split": 0.03},
    {"burn": 2, "fade": 0.5, "pump": 0.3, "decay": 0.2, "times": [0.2, 0.6], "mask": "ground", "objects": "a*"},
])
def test_key_burn_keeps_the_cut_s_frames_before(gpu, tmp_path, request, keys):
    spec = key(**keys)
    frames = frames_of(spec)
    note(request, check(gpu, tmp_path, spec, [F0, F0 + 1, F0 + 7, F0 + 8, F0 + 9] + frames[12::5], pic=ramp,
                        frames=cut(), masked="mask" in keys))


def test_key_burn_at_the_tolerance_of_each_frame_before(gpu, tmp_path, request):
    spec = key(burn=4, fade=0.9, tolerance=[[0.0, 0.02], [0.5, 0.4], [1.0, 0.1]])      # every frame its own tolerance
    note(request, check(gpu, tmp_path, spec, frames_of(spec)[3::4], pic=ramp, frames=cut()))


@pytest.mark.parametrize("keys", [
    {"delay": 3, "back": None},
    {"delay": 40, "back": None, "gain": [1.2, 1.0, 0.8], "burn": 1},
])
def test_key_shows_the_cut_s_frame_before(gpu, tmp_path, request, keys):
    spec = key(**keys)
    spec.pop("back")
    note(request, check(gpu, tmp_path, spec, [F0, F0 + 2, F0 + 9, F0 + 10, F0 + 21, F0 + 27], pic=ramp,
                        frames=cut()))


@pytest.mark.parametrize("keys", [
    {"in": 0.2},
    {"loop": True, "rate": 2.0, "mirror": True, "focus": [0.3, 0.6], "space": "chroma", "spill": "love"},
])
def test_key_shows_a_clip(gpu, tmp_path, request, keys):
    clip(tmp_path / "clip.mp4")
    spec = key(clip="clip.mp4", **keys)
    spec.pop("back")
    note(request, check(gpu, tmp_path, spec, frames_of(spec)[::4], pic=ramp, frames=cut(),
                        footage_of=FT.source(tmp_path)))


@pytest.mark.parametrize("keys", [
    {"spread": 0.6},                                                                    # the key opens from the middle
    {"spread": 1.2, "center": [0.2, 0.3], "burn": 3, "fade": 0.7, "soft": 0.2},         # and its trail with it
    {"spread": 0.5, "center": [0.8, 0.7], "space": "luma", "bands": 4, "spill": "gold", "soft": 0.1},
    {"spread": 0.4, "center": [0.5, 0.5], "soft": 0.0, "split": 0.03},
])
def test_key_opens_from_a_point(gpu, tmp_path, request, keys):
    spec = key(tolerance=[[0.0, 0.3], [1.0, 0.6]], **keys)
    note(request, check(gpu, tmp_path, spec, frames_of(spec)[::5], pic=ramp, frames=cut()))


@pytest.mark.parametrize("keys", [
    {"rect": [0.0, 0.0, 0.5, 1.0], "feather": 8.0},                                      # a feathered half of the frame
    {"rect": [0.5, 0.0, 1.0, 1.0], "feather": 0.0, "burn": 3, "fade": 0.6},              # a hard edge, with a trail
    {"rect": [0.2, 0.15, 0.85, 0.7], "feather": 14.0, "spread": 0.8, "center": [0.3, 0.4], "bands": 4, "soft": 0.1,
     "spill": "gold", "mask": "ground", "objects": "a*", "strength": [[0.0, 0.5], [1.0, 1.0]]},
])
def test_key_limited_to_a_rect(gpu, tmp_path, request, keys):
    """A key with a `rect` (the bros silhouettes' half frames) is bracketed by mkr-post's generic rect ops: the frame
    outside it as it was, a feathered edge as the CPU's rect_mask makes it, a trail and a mask inside."""
    spec = key(tolerance=[[0.0, 0.15], [1.0, 0.45]], **keys)
    frames = frames_of(spec)
    note(request, check(gpu, tmp_path, spec, frames[3::5], pic=ramp, frames=cut(), masked="mask" in keys))


def test_key_opens_from_a_point_of_the_scene_that_moves(gpu, tmp_path, request):
    spec = key(tolerance=0.3, spread=0.8, center={"point": "any"}, burn=2, fade=0.6)
    note(request, check(gpu, tmp_path, spec, frames_of(spec)[::4], pic=ramp, frames=cut(),
                        point_of=lambda k, g: ((40.0 + 2.0 * g, 30.0 + 0.5 * g), None)))


def test_key_follows_a_stem_and_the_burn_trail_follows_it_too(gpu, tmp_path, request):
    sr = 1000
    x = np.zeros(4 * sr, np.float32)
    x[sr // 2:] = 0.5 * np.sin(np.arange(len(x) - sr // 2) * 0.05)
    x[sr:] *= np.linspace(0.2, 1.0, len(x) - sr)
    spec = key(tolerance=0.05, follow={"stem": "vocals", "amount": 0.3, "smooth": 0.05}, burn=3, fade=0.7)
    note(request, check(gpu, tmp_path, spec, frames_of(spec)[::5], pic=ramp, frames=cut(),
                        stem_of=lambda timeline, stem: (x, sr, 0.0)))


def test_key_on_a_frame_the_gpu_cannot_read_stays_with_the_cpu(gpu, tmp_path):
    pl = plan(key(burn=2))
    ffx = GF.frame_fx(pl["fx"], SMALL, tmp_path, frames=cut(), fps=FPS, frame0=F0, hits={})
    fx = ffx.fxs[0]
    ffx.paths = None
    assert GFX.KINDS["key"].ops(ffx, fx, F0 + 5) is None
