"""mkr-post's warps, remaps and temporal effects against the CPU path (mkmmd.gpufx: echo, slitscan, lens, shock, push,
slab, split, melt, kaleido, tunnel, trails, scope, pixelsort, mosh) on synthetic pictures: a case for each branch of a
kind's op and shader (keys that only change its numbers ride along in one), what the remaps share under a mask once
(the figure's place filled, the figure laid back as a sticker), each frame drawn by both and compared within a level
(tests/gpu_fixtures.py). Skipped without the mkr-post binary or a GPU."""
import cv2
import numpy as np
import pytest

import gpu_fixtures as GF
from mkmmd import gpufx as GFX
from mkmmd import postgpu as PG
from mkmmd.core import framefx as FX
from mkmmd.core import palette as PAL
from mkmmd.core import transition as TR

FPS, F0 = 30, 1
SMALL, LARGE = (160, 90), (256, 144)       # the masked remaps' stickers want room: a quarter-size shadow, a fill
MOON = PAL.get("rose-pine-moon")


def plan(*fx, duration=2.0, size=SMALL):
    data = {"project": {"duration": duration}, "output": [{"name": "t", "size": list(size)}],
            "shot": [{"name": "a", "from": 0.0, "to": duration}], "fx": list(fx)}
    return TR.plan(data, FPS, F0, MOON)


def picture(seed, size=SMALL):
    """Smooth colour noise with hard-edged blocks over it: sampling between pixels shows on both."""
    w, h = size
    rng = np.random.default_rng(seed)
    t = cv2.GaussianBlur(rng.random((h, w, 3)).astype(np.float32), (0, 0), 2.0)
    t = (t - t.min()) / (t.max() - t.min())
    blocks = (rng.random((h // 6 + 1, w // 6 + 1, 3)) > 0.6).astype(np.float32)
    hard = np.kron(blocks, np.ones((6, 6, 1), np.float32))[:h, :w]
    return np.ascontiguousarray(np.where(hard > 0, 1.0 - t, t), np.float32)


def frames(size):
    return {g: picture(g, size) for g in range(F0, F0 + 60) if g != F0 + 7}     # frame F0 + 7 was never rendered


def figure(g, size, kind):
    """A figure's coverage on frame g, antialiased: a disc walking across ("disc"), a band from the left edge across
    most of the frame, so some of its place has no ground on either side to mirror in ("band"), or nothing ("none")."""
    w, h = size
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    if kind == "disc":
        cx, cy, r = w * (0.3 + 0.01 * (g % 30)), h * 0.5, h * 0.3
        return np.clip(r + 0.5 - np.hypot(xx - cx, yy - cy), 0.0, 1.0)
    if kind == "band":
        return np.clip(np.minimum(w * 0.8 - xx, np.abs(yy - h * 0.5) * -1.0 + h * 0.25) + 0.5, 0.0, 1.0)
    return np.zeros((h, w), np.float32)


@pytest.fixture(scope="module")
def gpu(tmp_path_factory):
    g = GF.GPU(SMALL, tmp_path_factory.mktemp("gpu"))
    yield g
    g.close()


@pytest.fixture(scope="module")
def gpu_large(tmp_path_factory):
    g = GF.GPU(LARGE, tmp_path_factory.mktemp("gpu_large"))
    yield g
    g.close()


def check(gpu, folder, spec, frames_to, size=SMALL, shape=None, **kw):
    """Every frame of `frames_to` drawn by both paths from a picture of its own, within a level; and the effect
    changed the picture on some of them (the comparison means something). With `shape` the entry's figure layers are
    figure(g, size, shape); `kw` go to the FrameFx (a scope's stems)."""
    pl = plan(spec, size=size)
    figures = None
    if shape is not None:
        key = FX.figure_key(pl["fx"][0])
        figures = {(key, g): figure(g, size, shape) for g in range(F0, F0 + 60)}
    ffx = GF.frame_fx(pl["fx"], size, folder, frames=frames(size), figures=figures, fps=FPS, frame0=F0, hits={}, **kw)
    fx = ffx.fxs[0]
    drew = 0
    for f in frames_to:
        img = picture(1000 + f, size)
        worst, over = gpu.check(ffx, fx, f, img)
        assert worst <= 1 and over == 0, f"{spec['kind']} frame {f}: {worst} levels, {over} pixels off by more than 1"
        drew += int(np.abs(ffx._apply(fx, f, img) - img).max() > 1.0 / 255.0)
    assert drew, f"{spec['kind']}: changed no frame"


def frames_of(spec):
    fx = plan(spec)["fx"][0]
    return list(range(fx["f0"], fx["f1"]))


@pytest.mark.parametrize("keys", [
    {"blend": "darken", "count": 3, "every": 2, "fade": 0.7},
    {"tint": "love", "count": 5, "every": 1},
    {"blend": "darken", "tint": "#3080ff", "strength": [[0.0, 0.0], [0.5, 1.0]]},
])
def test_echo(gpu, tmp_path, keys):
    spec = {"kind": "echo", "from": 0.0, "to": 1.0, **keys}
    check(gpu, tmp_path, spec, frames_of(spec)[::3])


@pytest.mark.parametrize("keys", [
    {"now": "left", "depth": 9, "strength": [[0.0, 0.2], [1.0, 1.0]]}, {"now": "bottom"},
])
def test_slitscan(gpu, tmp_path, keys):
    spec = {"kind": "slitscan", "from": 0.0, "to": 1.5, **keys}
    check(gpu, tmp_path, spec, frames_of(spec)[::4])


@pytest.mark.parametrize("keys", [
    {"radius": 0.12, "shadow": 0.6}, {"radius": 0.3, "center": [0.05, 0.9], "strength": [[0.0, 0.0], [1.0, 1.0]]},
])
def test_lens(gpu, tmp_path, keys):
    spec = {"kind": "lens", "from": 0.0, "to": 1.0, "center": [0.4, 0.55], **keys}
    check(gpu, tmp_path, spec, frames_of(spec)[::5])


@pytest.mark.parametrize("keys", [
    {"glow": 0.0, "amount": 0.03},
    {"speed": 0.8, "width": 0.08, "color": "love", "life": 0.9, "center": [0.9, 0.1],
     "strength": [[0.0, 1.0], [1.0, 0.3]]},
])
def test_shock(gpu, tmp_path, keys):
    spec = {"kind": "shock", "from": 0.0, "to": 1.0, "times": [0.05, 0.3], **keys}
    check(gpu, tmp_path, spec, frames_of(spec)[::2])


@pytest.mark.parametrize("keys", [
    {"ease": "linear", "play": True, "inside": 0.5, "center": [0.9, 0.2], "radius": 0.05},
])
def test_push(gpu, tmp_path, keys):
    spec = {"kind": "push", "from": 0.5, "to": 1.5, "center": [0.3, 0.4], **keys}
    check(gpu, tmp_path, spec, frames_of(spec)[::3] + frames_of(spec)[-1:])


@pytest.mark.parametrize("keys", [
    {"dir": "down", "slant": -10, "angle": 20, "alpha": 0.7, "size": [0.5, 0.3], "at": [0.2, 0.7], "color": "gold",
     "strength": 0.6},
])
def test_slab(gpu, tmp_path, keys):
    spec = {"kind": "slab", "from": 0.0, "to": 0.8, **keys}
    check(gpu, tmp_path, spec, frames_of(spec))


@pytest.mark.parametrize("keys", [
    {"bands": 30, "shift": 0.2, "share": 0.8, "rgb": 20, "hold": 0.1, "seed": 7, "strength": [[0.0, 0.0], [1.0, 1.0]]},
])
def test_split(gpu, tmp_path, keys):
    spec = {"kind": "split", "from": 0.0, "to": 1.0, **keys}
    check(gpu, tmp_path, spec, frames_of(spec)[::2])


@pytest.mark.parametrize("keys", [
    {"mode": "breathe", "amount": -0.3, "speed": 0.5, "center": [0.3, 0.6]},
    {"mode": "swirl", "amount": 200, "strength": [[0.0, 0.0], [1.0, 1.0]]},
    {"mode": "ripple", "amount": 0.03},
    {"mode": "ripple", "amount": 0.05, "times": [0.1, 0.5], "decay": 0.3},
    {"mode": "drip", "amount": 0.3, "speed": 1.5},
    {"mode": "noise", "amount": 0.05, "scale": 0.3, "speed": 2.0},
])
def test_melt(gpu, tmp_path, keys):
    spec = {"kind": "melt", "from": 0.0, "to": 1.0, **keys}
    check(gpu, tmp_path, spec, frames_of(spec)[::3])


@pytest.mark.parametrize("keys", [
    {"segments": 7, "mirror": False, "spin": 40, "turn": 30, "zoom": [[0.0, 0.4], [1.0, 2.5]],
     "source": {"radius": 0.15}},
    {"segments": 8, "zoom": [[0.0, 0.3], [1.0, 0.9]], "source": {"center": [0.4, 0.4], "angle": 30}},
    {"times": [0.1, 0.4], "pulse": 0.5, "step": 45, "glide": 0.1, "strength": [[0.0, 0.0], [1.0, 1.0]]},
])
def test_kaleido(gpu, tmp_path, keys):
    spec = {"kind": "kaleido", "from": 0.0, "to": 1.0, **keys}
    check(gpu, tmp_path, spec, frames_of(spec)[::3])


@pytest.mark.parametrize("keys", [
    {"twist": 15, "scale": 0.6, "speed": 1.2}, {"frame": {"color": ["love", "gold"], "width": 30}},
    {"times": [0.2], "step": 2, "depth": 4, "center": [0.3, 0.6], "strength": [[0.0, 0.0], [1.0, 1.0]]},
])
def test_tunnel(gpu, tmp_path, keys):
    spec = {"kind": "tunnel", "from": 0.0, "to": 1.0, **keys}
    check(gpu, tmp_path, spec, frames_of(spec)[::3])


MASKS = [
    ({"mask": "ground", "edge": {"color": "#ffffff", "width": 40}, "shadow": {"alpha": 0.5, "blur": 60}}, "disc"),
    ({"mask": "ground", "edge": {"color": "gold", "width": 20}}, "band"),
    ({"mask": "figure"}, "disc"),
    ({"mask": "figure", "back": "love", "edge": {"color": "#ffffff", "width": 30}, "shadow": {"blur": 8}}, "disc"),
    ({"mask": "figure", "back": "base"}, "none"),
    ({"mask": "ground", "shadow": {}}, "none"),
]


# the variants of what the masked remaps share (fx/masked.rs) on kaleido, which adds nothing of its own under a mask;
# tunnel's figure queue and melt's keep pass once each
@pytest.mark.parametrize("kind, own, keys, shape", [
    *(("kaleido", {"segments": 6, "turn": 40}, keys, shape) for keys, shape in MASKS),
    ("tunnel", {"frame": {"color": ["love", "foam"], "width": 20}}, {"mask": "figure"}, "disc"),
    ("melt", {"mode": "swirl", "amount": 120}, {"mask": "ground"}, "disc"),
])
def test_masked(gpu_large, tmp_path, kind, own, keys, shape):
    spec = {"kind": kind, "from": 0.0, "to": 1.0, "objects": "Fig*", **own, **keys}
    if shape == "none" and keys["mask"] == "figure":          # nothing of her: the picture as it is, on both
        pl = plan(spec, size=LARGE)
        key = FX.figure_key(pl["fx"][0])
        ffx = GF.frame_fx(pl["fx"], LARGE, tmp_path, frames=frames(LARGE),
                          figures={(key, g): figure(g, LARGE, shape) for g in range(F0, F0 + 60)}, fps=FPS, frame0=F0,
                          hits={})
        for f in frames_of(spec)[::5]:
            assert gpu_large.check(ffx, ffx.fxs[0], f, picture(1000 + f, LARGE)) == (0, 0)
        return
    check(gpu_large, tmp_path, spec, frames_of(spec)[::5], size=LARGE, shape=shape)


@pytest.mark.parametrize("keys", [
    {},
    {"blend": "screen", "tint": 0.5, "step": [0.03, -0.02]},
    {"blend": "add", "count": 3, "delay": 1, "colors": ["love", "gold"], "spread": 0.5, "cycle": 0.3, "fade": 0.6,
     "times": [0.2], "snap": 0.1, "strength": [[0.0, 0.2], [1.0, 1.0]]},
])
def test_trails(gpu, tmp_path, keys):
    spec = {"kind": "trails", "from": 0.0, "to": 1.0, "objects": "Fig*", **keys}
    check(gpu, tmp_path, spec, frames_of(spec)[::4], shape="disc")


def stem(timeline, name):
    """A made-up stem: a chirp with a beat, 8 kHz, a second before clip time 0."""
    sr = 8000
    t = np.arange(4 * sr) / sr
    return (np.sin(2 * np.pi * (110 + 60 * t) * t) * (0.6 + 0.4 * np.sin(2 * np.pi * 2 * t))).astype(np.float32), sr, 1.0


@pytest.mark.parametrize("keys", [
    {"lag": 0, "glow": 0, "width": 4},
    {"at": [0.2, 0.7], "size": 0.25, "color": "love", "window": 0.02, "strength": [[0.0, 0.3], [1.0, 1.0]]},
])
def test_scope(gpu, tmp_path, keys):
    spec = {"kind": "scope", "from": 0.0, "to": 1.0, **keys}
    check(gpu, tmp_path, spec, frames_of(spec)[::4], stem_of=stem)


ODD = (170, 96)                  # a row OpenCV's RGB2HSV does not cover in whole vectors


@pytest.fixture(scope="module")
def gpu_odd(tmp_path_factory):
    g = GF.GPU(ODD, tmp_path_factory.mktemp("gpu_odd"))
    yield g
    g.close()


@pytest.mark.parametrize("keys, shape", [
    ({"mask": "figure", "objects": "Fig*", "axis": "y", "reverse": True, "lo": 0.2, "hi": 0.7}, "disc"),
    ({"by": "hue"}, None),
    ({"mask": "ground", "objects": "Fig*", "by": "sat", "axis": "y", "lo": 0.1, "hi": 0.6}, "disc"),
    ({"times": [0.1, 0.4], "decay": 0.2, "strength": [[0.0, 0.0], [1.0, 1.0]]}, None),
])
def test_pixelsort(gpu, gpu_odd, tmp_path, keys, shape):
    spec = {"kind": "pixelsort", "from": 0.0, "to": 1.0, **keys}
    check(gpu, tmp_path / "small", spec, frames_of(spec)[::4], shape=shape)
    if keys.get("by") in ("hue", "sat"):
        check(gpu_odd, tmp_path / "odd", spec, frames_of(spec)[::6], size=ODD, shape=shape)


@pytest.mark.parametrize("keys, shape", [
    ({"block": 40}, None), ({"mask": "ground", "objects": "Fig*"}, "disc"),
    ({"mask": "figure", "objects": "Fig*", "strength": [[0.2, 0.0], [0.5, 1.0]]}, "disc"),
])
def test_mosh(gpu, tmp_path, keys, shape):
    """Frame after frame (mkr-post carries its held picture one step on) and with frames skipped (from the start); past
    the frame the cut never rendered (no step into or out of it). On that frame itself the CPU draws it."""
    spec = {"kind": "mosh", "from": 0.2, "to": 1.0, **keys}
    frames_to = [f for f in frames_of(spec) if f != F0 + 7]
    check(gpu, tmp_path / "on", spec, frames_to[:8], shape=shape)
    check(gpu, tmp_path / "skips", spec, frames_to[::5], shape=shape)


@pytest.mark.parametrize("keys, shape", [
    ({"flow": {"kind": "noise", "size": 0.5, "amount": 70, "steps": 5, "jitter": 0.5, "stuck": 0.2}, "block": 40,
      "mask": "ground", "objects": "Fig*"}, "disc"),
])
def test_mosh_flow(gpu, tmp_path, keys, shape):
    """A made-up flow, frame after frame from the span's first: mkr-post holds the picture as it reached the mosh there
    and carries what it holds on; past the frame the cut never rendered too (a flow needs none of the cut's frames)."""
    spec = {"kind": "mosh", "from": 0.1, "to": 1.0, **keys}
    check(gpu, tmp_path, spec, frames_of(spec)[:10], shape=shape)


def test_a_flowing_mosh_goes_on_only_from_what_mkr_post_holds(tmp_path):
    """Its frame before skipped, or its ops dropped on the way (a layer missing): the CPU draws it."""
    pl = plan({"kind": "mosh", "from": 0.1, "to": 1.0, "flow": {"kind": "drift", "amount": 60}})
    ffx = GF.frame_fx(pl["fx"], SMALL, tmp_path, frames=frames(SMALL), fps=FPS, frame0=F0, hits={})
    fx = ffx.fxs[0]
    f0, mosh = fx["f0"], GFX.KINDS["mosh"]
    assert mosh.ops(ffx, fx, f0 + 1) is None
    assert mosh.ops(ffx, fx, f0)[0]["flow"]["fresh"]
    assert not mosh.ops(ffx, fx, f0 + 1)[0]["flow"]["fresh"]
    PG.dropped(ffx, f0 + 1)
    assert mosh.ops(ffx, fx, f0 + 2) is None
    assert mosh.ops(ffx, fx, f0)[0]["flow"]["fresh"]


def test_a_pixel_sort_is_the_first_op_mkr_post_draws_on_a_frame_it_sorts():
    """It is chaotic in its input (mkmmd.gpufx.pixelsort): the entries before it are drawn on the CPU."""
    op = {"op": "pixelsort"}
    assert PG.first({"kind": "pixelsort"}, [op])
    assert not PG.first({"kind": "pixelsort"}, [])                  # a sort too weak to draw lets the walk go on
    assert not PG.first({"kind": "kaleido"}, [{"op": "kaleido"}])
