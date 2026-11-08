"""mkr-post's painted and collage effects against the CPU path (mkmmd.gpufx: tint, invert, halftone, film, camcorder,
doodle, label, flood, liquid, spiral, grid, stack, mandala, footage) on synthetic pictures: one case per distinct code
path of a kind (a mode, mask, blend, pattern, fill or stage switched on or off; keys that only change numbers ride along
in a case on their path), the masked ones and the collages over a figure layer, each frame drawn by both and compared
within a level (tests/gpu_fixtures.py). Skipped without the mkr-post binary or a GPU."""
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

import gpu_fixtures as GF
from mkmmd import footage as FT
from mkmmd import mandala as MDF
from mkmmd import picturefx as PFX
from mkmmd import postgpu as PG
from mkmmd.core import framefx as FX
from mkmmd.core import palette as PAL
from mkmmd.core import transition as TR

FPS, F0 = 30, 1
SMALL = (192, 108)
MOON = PAL.get("rose-pine-moon")


def plan(*fx, duration=3.0, size=SMALL):
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


def disc(g, size=SMALL):
    """A figure's coverage on frame g: an antialiased disc walking across."""
    w, h = size
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cx, cy, r = w * (0.3 + 0.005 * (g % 60)), h * 0.5, h * 0.3
    return np.clip(r + 0.5 - np.hypot(xx - cx, yy - cy), 0.0, 1.0)


def system_font():
    """A TTF of this machine (a stamp or a label needs a real font to draw), else the test is skipped."""
    if shutil.which("fc-match"):
        p = subprocess.run(["fc-match", "-f", "%{file}", "monospace"], capture_output=True, text=True).stdout.strip()
        if p.lower().endswith((".ttf", ".otf")) and Path(p).is_file():
            return p
    for p in ("/usr/share/fonts/TTF/DejaVuSansMono.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"):
        if Path(p).is_file():
            return p
    pytest.skip("no TrueType font on this machine")


@pytest.fixture(scope="module")
def gpu(tmp_path_factory):
    g = GF.GPU(SMALL, tmp_path_factory.mktemp("gpu"))
    yield g
    g.close()


def check(gpu, folder, spec, frames_to, masked=False, **kw):
    """Every frame of `frames_to` (in order) drawn by both paths from a picture of its own, within a level; and the
    effect changed the picture on some of them (the comparison means something). `masked`: the entry's figure layers
    are disc(g); `kw` go to the FrameFx (a font)."""
    pl = plan(spec)
    figures = None
    if masked:
        key = FX.figure_key(pl["fx"][0])
        figures = {(key, g): disc(g) for g in range(F0, F0 + 100)}
    frames = {g: picture(g) for g in range(F0, F0 + 100)}
    ffx = GF.frame_fx(pl["fx"], SMALL, folder, frames=frames, figures=figures, fps=FPS, frame0=F0, hits={}, **kw)
    fx = ffx.fxs[0]
    drew = 0
    for f in frames_to:
        img = picture(1000 + f)
        worst, over = gpu.check(ffx, fx, f, img)
        assert worst <= 1 and over == 0, f"{spec['kind']} frame {f}: {worst} levels, {over} pixels off by more than 1"
        drew += int(np.abs(ffx._apply(fx, f, img) - img).max() > 1.0 / 255.0)
    assert drew, f"{spec['kind']}: changed no frame"


def frames_of(spec):
    fx = plan(spec)["fx"][0]
    return list(range(fx["f0"], fx["f1"]))


@pytest.mark.parametrize("keys", [
    {"color": "love", "dark": "#102030", "light": "#f0e0d0", "strength": [[0.0, 0.0], [0.6, 1.0]]},
])
def test_tint(gpu, tmp_path, keys):
    spec = {"kind": "tint", "from": 0.0, "to": 1.0, **keys}
    check(gpu, tmp_path, spec, frames_of(spec)[::4])


@pytest.mark.parametrize("keys, masked", [
    ({"times": [0.2, 0.5], "hue": 140.0, "in": 0.02, "hold": 0.2, "out": 0.3}, False),
    ({"times": [0.3], "hue": -60.0, "mask": "figure", "objects": "a*"}, True),
    ({"times": [0.3], "mask": "ground", "objects": "a*", "strength": 0.6}, True),
])
def test_invert(gpu, tmp_path, keys, masked):
    spec = {"kind": "invert", "from": 0.0, "to": 1.2, **keys}
    check(gpu, tmp_path, spec, frames_of(spec)[::2], masked=masked)


@pytest.mark.parametrize("keys", [
    {"cell": 14.0, "angle": 80.0, "level": 0.8, "ink": "pine", "lift": 0.9, "offset": [-2.0, 4.5],
     "strength": [[0.0, 0.2], [0.5, 1.0]]},
])
def test_halftone(gpu, tmp_path, keys):
    spec = {"kind": "halftone", "from": 0.0, "to": 1.0, **keys}
    check(gpu, tmp_path, spec, frames_of(spec)[::5])


@pytest.mark.parametrize("keys", [
    {"mono": True, "fps": 12, "grain": {"amount": 0.12, "size": 2.0}, "flicker": 0.2, "weave": 3.0, "dust": 6,
     "scratches": 3, "hair": 1.0, "vignette": 0.6, "contrast": 1.4, "seed": 3},
    {"fps": 8, "saturation": 1.6, "dye": 0.8, "halation": 1.2, "grain": 0.0, "dust": 0, "scratches": 0,
     "vignette": 0.0, "contrast": 0.0, "weave": 0.0, "strength": [[0.0, 0.3], [0.8, 1.0]]},
    {"fps": 15, "rect": [0.2, 0.15, 0.85, 0.7], "feather": 12.0, "weave": 4.0, "halation": 0.8, "dust": 5,
     "strength": [[0.0, 0.5], [0.6, 1.0]]},
])
def test_film(gpu, tmp_path, keys):
    spec = {"kind": "film", "from": 0.0, "to": 1.0, **keys}
    check(gpu, tmp_path, spec, frames_of(spec))         # every frame: mkr-post keeps the print a step's first one made


def test_a_rect_goes_to_the_gpu_for_every_kind_bracketed_by_the_rect_ops_unless_it_draws_its_box(tmp_path):
    pl = plan({"kind": "film", "from": 0.0, "to": 1.0, "rect": [0.0, 0.0, 0.5, 1.0]},
              {"kind": "tint", "from": 0.0, "to": 1.0, "rect": [0.0, 0.0, 0.5, 1.0], "feather": 10.0})
    ffx = GF.frame_fx(pl["fx"], SMALL, tmp_path, frames={g: picture(g) for g in range(F0, F0 + 40)}, fps=FPS,
                      frame0=F0, hits={})
    film, tint = ffx.fxs
    film_ops = PG.ops(ffx, film, film["f0"])
    assert film_ops is not None and [o["op"] for o in film_ops] == ["film"]          # it prints the rect's box itself
    got = PG.ops(ffx, tint, tint["f0"])
    assert [o["op"] for o in got] == ["rect", "tint", "rect"] and got[0]["phase"] == "begin"
    end = got[2]
    ramps = np.fromfile(end["ramps"], np.float32)
    w, h = SMALL
    assert end["phase"] == "end" and end["count"] == len(ramps) == w + h
    rows, cols = PFX.rect_ramps(SMALL, (0.0, 0.0, 0.5, 1.0), 10.0 * ffx.px)
    assert np.array_equal(ramps[:h], rows) and np.array_equal(ramps[h:], cols)
    assert PG.ops(ffx, tint, tint["f0"])[2]["ramps"] == end["ramps"]                 # written once, by what they are


@pytest.mark.parametrize("rect, feather", [([0.0, 0.0, 0.5, 1.0], 10.0), ([0.3, 0.2, 0.9, 0.8], 0.0)])
def test_a_tint_limited_to_a_rect_is_kept_inside_it_as_the_cpu_keeps_it(gpu, tmp_path, rect, feather):
    spec = {"kind": "tint", "from": 0.0, "to": 1.0, "color": "love", "rect": rect, "feather": feather}
    check(gpu, tmp_path, spec, frames_of(spec)[::5])


@pytest.mark.parametrize("keys", [
    {"corner": "top-right", "indicator": "none", "date": "MAY 5", "tracking": 0.0, "chroma": 0.0, "soft": 0.0,
     "noise": 0.0, "jitter": 0.0, "scanlines": 0.0, "saturation": 1.4},
    {"corner": "bottom-left", "indicator": "REC", "time": "11:59:58 PM", "tick": True,
     "tracking": {"amount": 0.8, "speed": 0.6}, "switching": 1.0, "size": 0.09, "chroma": 14.0, "soft": 3.0,
     "noise": 0.2, "jitter": 4.0, "scanlines": 0.6, "saturation": 0.3, "seed": 9,
     "strength": [[0.0, 0.0], [0.5, 1.0]]},
])
def test_camcorder(gpu, tmp_path, keys):
    spec = {"kind": "camcorder", "from": 0.0, "to": 2.0, "date": "OCT. 23 1999", "time": "8:41 PM", **keys}
    font = system_font()
    check(gpu, tmp_path, spec, frames_of(spec)[::5], font_of=lambda slug: font)


@pytest.mark.parametrize("keys", [
    {"marks": [{"shape": "heart", "at": [0.3, 0.4], "size": 0.3, "color": "love", "angle": 20.0, "t": 0.1},
               {"shape": "star", "at": [0.75, 0.6], "size": 0.25, "color": "gold", "t": 0.4}]},
    {"marks": [{"shape": "x", "at": [0.5, 0.5], "size": 0.4}], "fps": 12, "width": 14.0, "outline": 0.0,
     "wobble": 4.0, "seed": 5, "strength": 0.7},
])
def test_doodle(gpu, tmp_path, keys):
    spec = {"kind": "doodle", "from": 0.0, "to": 1.0, **keys}
    check(gpu, tmp_path, spec, frames_of(spec)[::3])


@pytest.mark.parametrize("keys", [
    {"labels": [{"text": "LEFT", "at": [0.7, 0.3], "align": "right", "color": "gold"},
                {"text": "two", "at": [0.2, 0.7], "offset": [0.1, 0.1], "t": 0.3}],
     "box": {"color": "base", "pad": 6, "alpha": 0.9}, "leader": False, "size": 0.05, "tracking": 1.5,
     "strength": [[0.0, 0.5], [0.8, 1.0]]},
])
def test_label(gpu, tmp_path, keys):
    spec = {"kind": "label", "from": 0.0, "to": 1.2, **keys}
    font = system_font()
    check(gpu, tmp_path, spec, frames_of(spec)[::4], font_of=lambda slug: font)


@pytest.mark.parametrize("keys, masked", [
    ({"times": [0.1, 0.9], "color": "foam", "center": [0.2, 0.7], "grow": 0.5, "hold": 0.1, "recede": 0.3,
      "soft": 6.0, "wobble": 1.0, "seed": 4}, False),
    ({"times": [0.2], "fade": 0.4, "recede": 0.0, "mask": "figure", "objects": "a*"}, True),
    ({"times": [0.2], "mask": "ground", "objects": "a*", "strength": 0.5}, True),
])
def test_flood(gpu, tmp_path, keys, masked):
    spec = {"kind": "flood", "from": 0.0, "to": 2.0, **keys}
    check(gpu, tmp_path, spec, frames_of(spec)[::5], masked=masked)


@pytest.mark.parametrize("keys, masked", [
    ({"inks": ["pine", "rose"], "light": "text", "blend": "screen", "scale": 0.5, "speed": 2.0, "pulse": 0.6,
      "times": [0.3], "cells": 0.8, "bubbles": 0, "center": [0.3, 0.6], "seed": 7}, False),
    ({"blend": "multiply", "opacity": 0.6, "bubbles": 12, "mask": "figure", "objects": "a*"}, True),
    ({"blend": "soft-light", "scale": 3.0, "mask": "ground", "objects": "a*"}, True),
])
def test_liquid(gpu, tmp_path, keys, masked):
    spec = {"kind": "liquid", "from": 0.0, "to": 1.0, **keys}
    check(gpu, tmp_path, spec, frames_of(spec)[::6], masked=masked)


@pytest.mark.parametrize("keys, masked", [
    ({"pattern": "rings", "size": 0.08, "colors": ["love", "gold", "pine"], "speed": 1.5, "times": [0.4]}, False),
    ({"pattern": "moire", "moire": "lines", "angle": 12.0, "blend": "screen"}, False),
    ({"pattern": "moire", "offset": 9.0, "center": [0.3, 0.4], "blend": "multiply"}, False),
    ({"pattern": "sunburst", "arms": 7, "opacity": 0.7, "mask": "figure", "objects": "a*"}, True),
    ({"pattern": "checker", "size": 0.15, "warp": 1.0, "angle": 30.0, "blend": "soft-light", "mask": "ground",
      "objects": "a*"}, True),
    ({"pattern": "spiral", "arms": 3, "twist": 2.5, "step": 0.5, "times": [0.2, 0.6]}, False),
])
def test_spiral(gpu, tmp_path, keys, masked):
    spec = {"kind": "spiral", "from": 0.0, "to": 1.0, **keys}
    check(gpu, tmp_path, spec, frames_of(spec)[::6], masked=masked)


EDGE, SHADOW = {"color": "#ffffff", "width": 14}, {"color": "#575279", "alpha": 0.35}


@pytest.mark.parametrize("keys", [
    {"cells": [[2, 1], [3, 3]], "times": [0.4], "divide": 0.25, "back": "picture", "edge": EDGE, "shadow": SHADOW},
    {"cells": [1, [2, 1], [2, 2]], "times": [0.3, 0.7], "divide": 0.2, "fill": "silkscreen",
     "inks": [["#1f1d2e", "#d7827e", "#f6c177"], ["#1f1d2e", "#56949f"], ["#1f1d2e", "#b4637a", "#ea9d34", "#9ccfd8"]],
     "back": ["gold", "rose", "foam"], "edge": EDGE, "shadow": SHADOW},
    {"cells": [[2, 2]], "fill": ["silkscreen", "#faf4ed"], "offset": [0.0, 0.0], "piece": "love", "times": [0.5],
     "back": "picture"},
])
def test_grid(gpu, tmp_path, keys):
    spec = {"kind": "grid", "from": 0.0, "to": 1.2, "objects": "a*", **keys}
    check(gpu, tmp_path, spec, frames_of(spec)[::3], masked=True)


@pytest.mark.parametrize("keys", [
    {"count": 4, "step": [0.03, -0.01], "delay": 2, "bands": ["love", "gold", "foam", "iris"], "times": [0.3, 0.6],
     "back": "picture", "edge": EDGE, "shadow": SHADOW},
    {"count": 3, "mirror": True, "center": [0.4, 0.6], "bands": ["pine", "rose"], "edge": EDGE},
    {"count": 3, "recede": {"count": 3, "scale": 0.92, "speed": 0.7}, "bands": ["gold", "love"], "shadow": SHADOW,
     "strength": [[0.0, 0.4], [0.6, 1.0]]},
])
def test_stack(gpu, tmp_path, keys):
    spec = {"kind": "stack", "from": 0.0, "to": 1.2, "objects": "a*", **keys}
    check(gpu, tmp_path, spec, frames_of(spec)[::3], masked=True)


def disc_sprite(path, colour=(255, 0, 0), size=40, box=60):
    """A PNG `box` px square, clear but for a disc `size` px across in its middle."""
    yy, xx = np.mgrid[0:box, 0:box].astype(np.float32)
    a = np.clip(size / 2.0 - np.hypot(xx - (box - 1) / 2.0, yy - (box - 1) / 2.0) + 0.5, 0.0, 1.0)
    rgba = np.dstack([np.full((box, box), c, np.float32) for c in colour] + [a * 255.0]).astype(np.uint8)
    Image.fromarray(rgba, "RGBA").save(path)
    return path


@pytest.mark.parametrize("keys", [
    {"rings": [{"count": 6, "radius": 0.3, "source": {"shape": ["clover", "heart"], "colors": ["love", "gold"]}},
               {"count": 4, "radius": 0.15, "scale": 0.2}], "back": "pine", "edge": EDGE, "shadow": SHADOW},
    {"rings": [{"count": 8, "radius": 0.35, "scale": 0.15, "mirror": True}], "back": ["love", "gold"],
     "times": [0.2, 0.7], "grow": {"times": [0.3], "dur": 0.3, "gap": 0.04, "order": "around"}},
    {"rings": [{"count": 4, "radius": 0.3, "scale": 0.3, "turn": 25.0, "orient": "upright",
                "source": {"sprites": "s*.png", "eyes": True, "mouth": True}}],
     "back": "picture", "center": [0.4, 0.55], "times": [0.3, 0.6], "bloom": 0.2, "center_figure": True,
     "edge": EDGE, "shadow": SHADOW},
])
def test_mandala(gpu, tmp_path, keys):
    disc_sprite(tmp_path / "s1.png")
    disc_sprite(tmp_path / "s2.png", colour=(30, 200, 90), size=50)
    spec = {"kind": "mandala", "from": 0.0, "to": 1.2, "objects": "a*", **keys}
    check(gpu, tmp_path, spec, frames_of(spec)[::3], masked=True, sprite_of=MDF.sprite_source(tmp_path))


def clip(path, n=60, fps=24.0, size=(256, 144)):
    """A clip whose frames differ (a bar sweeping over colour noise): a frame read one off shows."""
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


@pytest.mark.parametrize("keys, clip_size", [
    ({"in": 0.5, "loop": True, "rate": 2.0, "mirror": True, "fill": "#1f1d2e",
      "strength": [[0.0, 0.3], [0.5, 1.0]]}, (256, 144)),
    ({"edge": EDGE, "shadow": SHADOW, "focus": [0.3, 0.6]}, (96, 64)),  # a clip enlarged to cover (OpenCV's bicubic)
    ({"fit": "contain", "back": "gold", "mirror": True}, (80, 60)),     # and to fit inside
])
def test_footage(gpu, tmp_path, keys, clip_size):
    clip(tmp_path / "clip.mp4", size=clip_size)
    spec = {"kind": "footage", "from": 0.0, "to": 1.2, "objects": "a*", "clip": "clip.mp4", **keys}
    check(gpu, tmp_path, spec, frames_of(spec), masked=True, footage_of=FT.source(tmp_path))
