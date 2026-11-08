"""Negative screen type through `mk post` (docs/design.md: Text, Screen type: `negative = true`): the coverage of the type is a
layer of its own (`negative/<frame>.png`, only its alpha read) that inverts the picture under the glyphs after the ordinary
screen layer is laid, before the grade (`[post] screen = "graded"`) or after it (`"over"`). The CPU path (mkmmd.postframes) is
checked against the formula on synthetic pixels; mkr-post (mkr/mkr-post, the GPU path) against the CPU path within a level
(that half runs under `mk gpu run -- uv run --with pytest pytest ...`, and skips without the binary or a GPU). Tiny synthetic
frames and glyph coverages, no Blender; a flat grade (nothing the grade changes), so the numbers are the composite's."""
import numpy as np
import pytest
from PIL import Image

from mkmmd import post as P
from mkmmd import postframes as PF
from mkmmd import postgpu as PG
from mkmmd.cli import common as CM
from mkmmd.core import screentype as SRT
from mkmmd.project import Project

FPS, F0, W, H = 10, 1, 64, 36
FRAMES = range(F0, F0 + 4)
FLAT = 'saturation = 1.0\ncontrast = 1.0\nvignette = 0.0\nfloor = false\ngrain = { amount = 0.0 }\n'


def backdrop(f):
    """The picture of frame `f`: a gradient (a slope per channel, drifting with the frame) and a hard vertical step."""
    x = np.linspace(0.0, 1.0, W, dtype=np.float32)[None, :]
    y = np.linspace(0.0, 1.0, H, dtype=np.float32)[:, None]
    r, g, b = x + 0 * y, 0.2 + 0.6 * y + 0 * x, 1.0 - x + 0 * y
    pic = np.stack([r, g, b], axis=2) * (0.5 + 0.1 * (f - F0))
    pic[:, 40:] = (0.9, 0.1, 0.5)                                    # the step
    return np.clip(pic, 0.0, 1.0)


def coverage():
    """A synthetic glyph's coverage: solid across the step, an antialiased rim of quarter steps round it."""
    a = np.zeros((H, W), np.float32)
    a[10:26, 30:50] = 1.0
    a[9, 30:50], a[26, 30:50] = 0.25, 0.75
    a[10:26, 29], a[10:26, 50] = 0.5, 0.5
    return a


def ordinary():
    """An ordinary screen layer: an opaque green box on the left (a little under the glyph's), a half red one below."""
    layer = np.zeros((H, W, 4), np.uint8)
    layer[4:20, 4:34] = (0, 200, 0, 255)
    layer[28:34, 4:20] = (255, 0, 0, 128)
    return layer


def save_rgba(path, rgba):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgba, "RGBA").save(path)


def project(root, screen):
    root.mkdir(parents=True, exist_ok=True)
    (root / "mk.toml").write_text(
        f'[project]\nname = "t"\nfps = {FPS}\nframe0 = {F0}\nduration = 0.4\n[[output]]\nname = "wide"\nsize = [{W}, {H}]\n'
        f'[[shot]]\nname = "a"\nfrom = 0.0\nto = 0.4\n[post]\nscreen = "{screen}"\n' + FLAT, encoding="utf-8")
    src = root / "renders" / "draft" / "wide"
    src.mkdir(parents=True, exist_ok=True)
    cover = (coverage() * 255 + 0.5).astype(np.uint8)
    neg = np.zeros((H, W, 4), np.uint8)
    neg[..., 0], neg[..., 1], neg[..., 2], neg[..., 3] = 230, 20, 100, cover          # colour is not used
    for f in FRAMES:
        Image.fromarray((backdrop(f) * 255 + 0.5).astype(np.uint8)).save(src / f"{f:05d}.png")
    # frame 1: no type; 2: ordinary only; 3: negative only (its screen layer fully transparent); 4: both
    save_rgba(src / SRT.layer_rel(F0 + 1), ordinary())
    save_rgba(src / SRT.layer_rel(F0 + 2), np.zeros((H, W, 4), np.uint8))
    save_rgba(src / SRT.negative_rel(F0 + 2), neg)
    save_rgba(src / SRT.layer_rel(F0 + 3), ordinary())
    save_rgba(src / SRT.negative_rel(F0 + 3), neg)
    return src


def output(root, screen, gpu=None):
    src = project(root, screen)
    proj = Project.load(root)
    files = {f: src / f"{f:05d}.png" for f in FRAMES}
    return PF.Output(proj, CM.cut_plan(proj), "wide", src, files, sorted(files), (W, H), gpu=gpu), src


def expected(src, f):
    """Frame `f` as the formula says, from the files themselves (float64): the ordinary layer over the picture, the picture
    under the negative coverage inverted (`bg * (1 - a) + (1 - bg) * a`)."""
    bg = P.read(src / f"{f:05d}.png").astype(np.float64)
    lp, npth = src / SRT.layer_rel(f), src / SRT.negative_rel(f)
    layer = P.read_rgba(lp) if lp.exists() else None
    if layer is not None:
        a = layer[..., 3:4].astype(np.float64)
        bg = bg * (1.0 - a) + layer[..., :3] * a
    if npth.exists():
        a = P.read_rgba(npth)[..., 3:4].astype(np.float64)
        bg = bg * (1.0 - a) + (1.0 - bg) * a
    return bg


def picture(raw):
    return np.frombuffer(raw, np.uint8).reshape(H, W, 3).astype(int)


@pytest.mark.parametrize("screen", ["graded", "over"])
def test_post_inverts_the_picture_under_the_negative_coverage_and_nothing_else(tmp_path, screen):
    out, src = output(tmp_path, screen)
    for k, f in enumerate(FRAMES):
        got = picture(out.finish(k)[0])
        want = P.to_u8(expected(src, f)).astype(int)
        assert np.abs(got - want).max() <= 1, (screen, f)
    bare = picture(out.finish(0)[0])
    assert np.abs(bare - P.to_u8(P.read(src / f"{F0:05d}.png")).astype(int)).max() <= 1       # the flat grade changes nothing
    neg = picture(out.finish(2)[0])
    pic = P.to_u8(P.read(src / f"{F0 + 2:05d}.png")).astype(int)
    a = coverage()
    assert np.abs(neg[a == 1.0] - (255 - pic[a == 1.0])).max() <= 1                              # solid: the negative
    assert np.abs(neg[a == 0.0] - pic[a == 0.0]).max() <= 1                                      # elsewhere: the picture
    assert (neg[12, 38] == 255 - pic[12, 38]).all() and (neg[12, 45] == 255 - pic[12, 45]).all()  # both sides of the step
    assert pic[12, 38].tolist() != pic[12, 45].tolist() and pic[12, 45].tolist() == [230, 26, 128]
    assert np.abs(neg[9, 40] - (0.75 * pic[9, 40] + 0.25 * (255 - pic[9, 40]))).max() <= 1.5     # a quarter-covered rim pixel
    both = picture(out.finish(3)[0])
    green = P.to_u8(P.read_rgba(src / SRT.layer_rel(F0 + 3))[..., :3]).astype(int)
    assert np.abs(both[12, 32] - (255 - green[12, 32])).max() <= 1                               # the negative of the green box
    assert np.abs(both[12, 10] - green[12, 10]).max() <= 1 and green[12, 10].tolist() == [0, 200, 0]   # the box clear of it


def test_a_frame_without_a_negative_layer_is_as_it_always_was(tmp_path):
    out, src = output(tmp_path, "graded")
    ordinary_only = picture(out.finish(1)[0])
    layer = P.read_rgba(src / SRT.layer_rel(F0 + 1))
    want = SRT.composite(P.read(src / f"{F0 + 1:05d}.png"), layer)
    assert np.abs(ordinary_only - P.to_u8(want).astype(int)).max() <= 0
    assert not (src / SRT.negative_rel(F0 + 1)).exists()


    assert np.abs(ordinary_only - P.to_u8(want).astype(int)).max() <= 1
    folder = tmp_path / "gpu"
    folder.mkdir()
    out, src = output(tmp_path / "p", "graded", gpu=str(folder))
    jobs = [out.plan(k)[1] for k in range(4)]
    assert ["layer" in j for j in jobs] == [False, True, True, True]
    assert [j.get("negative") for j in jobs] == [None, None, str(src / SRT.negative_rel(F0 + 2)),
                                                str(src / SRT.negative_rel(F0 + 3))]
    assert [out.plan(k)[2][1] for k in range(4)] == [0, 1, 1, 1]                                 # screen type frames


@pytest.mark.parametrize("screen", ["graded", "over"])
def test_mkr_post_inverts_as_the_cpu_does(tmp_path, screen):
    import gpu_fixtures as GF                     # skips without the binary or a GPU (run it as `mk gpu run -- ...`)
    why = GF.usable()
    if why:
        pytest.skip(why)
    folder = tmp_path / "gpu"
    folder.mkdir()
    out, src = output(tmp_path / "p", screen, gpu=str(folder))
    cpu, _ = output(tmp_path / "c", screen)
    server = PG.Server(out, folder)
    try:
        for k, f in enumerate(FRAMES):
            route, job, _ = out.plan(k)
            assert route == "gpu"
            server.send(job)
            got_k, _, got = server.receive()
            want = picture(cpu.finish(k)[0])
            assert got_k == k and np.abs(picture(got) - want).max() <= 1, (screen, f)
            assert np.abs(picture(got) - P.to_u8(expected(src, f)).astype(int)).max() <= 1, (screen, f)   # and the formula
    finally:
        server.close()
