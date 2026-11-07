"""mkr-post's paint app stamps against the CPU path (mkmmd.gpufx.paintapp) on synthetic sprites: stamps popping in (mid-pop,
overshoot), settled, fading out, scrolling, with googly eyes, sparkles, edge and shadow, several overlapping in the order
they landed, under a `strength` and with an `aspect` table's own `at`, each frame drawn by both and compared within a level
(tests/gpu_fixtures.py); and what stays on the CPU (the app's chrome, cursor, fills, erasers, undo). Skipped without the
mkr-post binary or a GPU."""
import numpy as np
import pytest
from PIL import Image

import gpu_fixtures as GF
from mkmmd import gpufx as GFX
from mkmmd import mandala as MDF
from mkmmd import postgpu as PG
from mkmmd.core import palette as PAL
from mkmmd.core import transition as TR

FPS, F0 = 30, 1
SIZE = (320, 180)
TALL = (144, 256)
MOON = PAL.get("rose-pine-moon")
WORST = {}                      # test id -> the largest level difference it saw (the report: pytest -s)


def plan(spec, size=SIZE, name="t", duration=3.0):
    data = {"project": {"duration": duration}, "output": [{"name": name, "size": list(size)}],
            "shot": [{"name": "a", "from": 0.0, "to": duration}], "fx": [spec]}
    return TR.plan(data, FPS, F0, MOON)


def picture(seed, size=SIZE):
    """Smooth colour ramps with hard-edged blocks over them (a ground the stickers' edges and shadows show on)."""
    w, h = size
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    base = np.dstack([0.3 + 0.5 * xx / w, 0.35 + 0.4 * yy / h, 0.5 + 0.3 * np.sin(xx / 17.0 + seed)]).astype(np.float32)
    blocks = (rng.random((h // 9 + 1, w // 9 + 1, 3)) > 0.7).astype(np.float32)
    hard = np.kron(blocks, np.ones((9, 9, 1), np.float32))[:h, :w]
    return np.ascontiguousarray(np.clip(np.where(hard > 0, 1.0 - base, base), 0.0, 1.0), np.float32)


def sprite(path, colour=(230, 60, 90), box=64, seed=0):
    """An RGBA PNG `box` px square: a blob with a bump and a notch (it shows its turn), its colour shading across."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:box, 0:box].astype(np.float32)
    cx = cy = (box - 1) / 2.0
    ang = np.arctan2(yy - cy, xx - cx)
    r = box * 0.34 * (1.0 + 0.25 * np.cos(ang - 0.5) + 0.12 * np.cos(3.0 * ang + rng.random()))
    a = np.clip(r - np.hypot(xx - cx, yy - cy) + 0.5, 0.0, 1.0)
    shade = 0.6 + 0.4 * xx / box
    rgb = np.dstack([np.full((box, box), c, np.float32) * shade for c in colour])
    Image.fromarray(np.dstack([rgb, a * 255.0]).astype(np.uint8), "RGBA").save(path)
    return path


@pytest.fixture(scope="module")
def gpu(tmp_path_factory):
    g = GF.GPU(SIZE, tmp_path_factory.mktemp("gpu"))
    yield g
    g.close()


@pytest.fixture(scope="module")
def gpu_tall(tmp_path_factory):
    g = GF.GPU(TALL, tmp_path_factory.mktemp("gpu_tall"))
    yield g
    g.close()


def app(**stamps):
    """A paint app of stamps alone over 0 .. 2.4 s (sprites s*.png)."""
    stamps = {"sprites": "s*.png", "times": [0.2, 0.7, 1.3], "size": 0.28, **stamps}
    return {"kind": "paintapp", "from": 0.0, "to": 2.4, "ui": False, "cursor": False, "stamps": stamps}


def runner(spec, folder, size=SIZE, name="t", **kw):
    sprite(folder / "s1.png")
    sprite(folder / "s2.png", colour=(40, 190, 110), seed=1)
    sprite(folder / "s3.png", colour=(240, 190, 50), seed=2)
    pl = plan(spec, size, name)
    return GF.frame_fx(pl["fx"], size, folder, frames={g: picture(g, size) for g in range(F0, F0 + 100)}, fps=FPS,
                       frame0=F0, hits={}, sprite_of=MDF.sprite_source(folder), **kw)


def check(gpu, folder, spec, every=2, size=SIZE, **kw):
    """Every `every`th frame of the entry drawn by both paths over a picture, within a level; and the stamps changed the
    picture on some of them. The largest level difference."""
    ffx = runner(spec, folder, size, **kw)
    fx = ffx.fxs[0]
    drew, most = 0, 0
    for f in range(fx["f0"], fx["f1"], every):
        img = picture(1000 + f, size)
        worst, over = gpu.check(ffx, fx, f, img)
        assert worst <= 1 and over == 0, f"frame {f}: {worst} levels, {over} pixels off by more than 1"
        most = max(most, worst)
        drew += int(np.abs(ffx._apply(fx, f, img) - img).max() > 1.0 / 255.0)
    assert drew, "the stamps changed no frame"
    return most


def note(request, worst):
    WORST[request.node.name] = worst
    print(f"\n{request.node.name}: max level difference {worst}")


# ================================================================================================= what the GPU draws
@pytest.mark.parametrize("stamps", [
    {"sparkle": False},                                                         # pop, overshoot, settled
    {"sparkle": False, "pop": 1.0, "angle": 40.0},                              # a slow pop with a wide turn
    {"sparkle": False, "pop": 0.0},                                             # no pop: there at once
    {},                                                                         # the sparkles, as they fly
    {"sparkle": False, "edge": False, "shadow": False},                         # bare sprites
    {"sparkle": False, "edge": {"color": "#2a273f", "width": 14.0},
     "shadow": {"color": "love", "offset": [0.01, 0.02], "blur": 12.0, "alpha": 0.6}},
    {"sparkle": False, "life": 0.8, "fade": 0.4},                               # fading out (and gone)
    {"life": 0.6, "fade": 0.0},                                                 # leaving at once
    {"sparkle": False, "scroll": [0.1, -0.06], "at": [[0.3, 0.4], [0.7, 0.6]]},  # drifting over the frame
    {"sparkle": False, "at": [[0.5, 0.5]], "times": [0.2, 0.35, 0.5, 0.65, 0.8], "size": 0.4},  # a pile, in deal order
    {"sprites": "s1.png", "at": [[0.5, 0.5], [0.52, 0.5], [0.5, 0.53]], "times": [0.2, 0.3, 0.4], "size": 0.5,
     "angle": 0.0},
    {"at": [[-0.02, 0.5], [1.01, 0.2], [0.5, 1.02]]},                           # half off the frame, sparkles on it
])
def test_stamps(gpu, tmp_path, request, stamps):
    note(request, check(gpu, tmp_path, app(**stamps)))


@pytest.mark.parametrize("stamps", [
    {"eyes": True, "mouth": True, "sparkle": False},
    {"eyes": {"look": "viewer", "size": 1.2}, "mouth": False},
    {"eyes": {"look": [0.2, 0.8], "wobble": 2.0}, "mouth": True, "life": 0.9, "fade": 0.5},
    {"eyes": True, "mouth": True, "at": [[0.5, 0.5], [0.55, 0.52]], "times": [0.2, 0.4, 0.6], "size": 0.45,
     "scroll": [0.05, 0.03]},
])
def test_stamps_with_eyes(gpu, tmp_path, request, stamps):
    note(request, check(gpu, tmp_path, app(**stamps)))


@pytest.mark.parametrize("keys", [
    {"strength": [[0.0, 0.3], [2.4, 1.0]]},
    {"strength": 0.5},
])
def test_stamps_under_a_strength(gpu, tmp_path, request, keys):
    note(request, check(gpu, tmp_path, {**app(), **keys}))


def test_the_cases_reach_what_they_name(tmp_path):
    """Over a fading stamp with eyes and sparkles the op carries fades, pupils and sparkles (inks), a partial pop."""
    ffx = runner(app(life=0.8, fade=0.4, eyes=True, mouth=True, pop=0.5), tmp_path)
    fx = ffx.fxs[0]
    got = [GFX.KINDS["paintapp"].ops(ffx, fx, f) for f in range(fx["f0"], fx["f1"])]
    stickers = [s for ops in got for op in ops for s in op["stickers"]]
    assert any(s["opacity"] < 1.0 for s in stickers)
    assert any(s["inks"] for s in stickers if s["opacity"] == 1.0)
    assert any(len(s["inks"]) > 4 for s in stickers)                        # pupils and glints, and sparkles over them
    assert any(s["warp"]["m"][0] < 0.9 * max(abs(s2["warp"]["m"][0]) for s2 in stickers) for s in stickers)   # a pop


def test_stamps_follow_the_aspect_tables_own_spots(gpu_tall, tmp_path, request):
    spec = {**app(at=[[0.5, 0.5]]), "aspect": {"t": {"stamps": {
        "sprites": "s*.png", "times": [0.2, 0.7, 1.3], "size": 0.5, "at": [[0.25, 0.25], [0.75, 0.6], [0.4, 0.85]]}}}}
    ffx = runner(spec, tmp_path, TALL, output="t")
    assert ffx.fxs[0]["stamps"]["at"] == [[0.25, 0.25], [0.75, 0.6], [0.4, 0.85]]
    note(request, check(gpu_tall, tmp_path, spec, size=TALL, output="t"))


def test_stamps_before_any_lands_draw_nothing(gpu, tmp_path):
    ffx = runner(app(times=[1.0]), tmp_path)
    fx = ffx.fxs[0]
    assert GFX.KINDS["paintapp"].ops(ffx, fx, F0 + 10) == []
    assert PG.ops(ffx, fx, F0 + 40)[0]["stickers"]


def test_stamps_share_a_tile_a_sprite_and_a_size(gpu, tmp_path):
    ffx = runner(app(sprites="s1.png", times=[0.1, 0.3, 0.5, 0.7], sparkle=False), tmp_path)
    op = GFX.KINDS["paintapp"].ops(ffx, ffx.fxs[0], F0 + 30)[0]
    assert len(op["stickers"]) == 4 and len(op["tiles"]) == 1
    assert {s["tile"] for s in op["stickers"]} == {0}


# ================================================================================================= what stays on the CPU
@pytest.mark.parametrize("extra", [
    {"ui": True},
    {"cursor": True},
    {"cursor": {"kind": "pencil", "keys": [{"t": 0.5, "at": [0.5, 0.5], "click": True}]}},
    {"bucket": {"at": [0.5, 0.5], "times": [0.5]}},
    {"eraser": {"at": 1.0, "dur": 0.5}},
    {"undo": {"times": [1.0]}},
])
def test_a_paint_app_with_more_than_stamps_stays_with_the_cpu(tmp_path, extra):
    spec = {**app(), **extra}
    ffx = runner(spec, tmp_path)
    fx = ffx.fxs[0]
    assert all(GFX.KINDS["paintapp"].ops(ffx, fx, f) is None for f in (F0 + 10, F0 + 40))
    assert PG.ops(ffx, fx, F0 + 40) is None


def test_a_paint_app_without_stamps_stays_with_the_cpu(tmp_path):
    spec = {"kind": "paintapp", "from": 0.0, "to": 2.4, "ui": False, "cursor": False,
            "bucket": {"at": [0.5, 0.5], "times": [0.5]}}
    ffx = runner(spec, tmp_path)
    assert GFX.KINDS["paintapp"].ops(ffx, ffx.fxs[0], F0 + 40) is None


def test_stamps_without_a_scratch_folder_stay_with_the_cpu(tmp_path):
    ffx = runner(app(), tmp_path)
    ffx.paths = None
    assert GFX.KINDS["paintapp"].ops(ffx, ffx.fxs[0], F0 + 40) is None
