"""Outfit textures (mkmmd.model.parts.outfit_tex): shapes, tileability, determinism, palette statistics, toon ramps,
sphere maps and the build step. bpy-free; the images are small (128..512 px) so the file runs in a few seconds."""
import ast
import inspect
from pathlib import Path

import numpy as np
import pytest

from mkmmd.model.parts import outfit_tex as ot

W = np.array([0.2126, 0.7152, 0.0722], np.float32)


def lum(img):
    return img[..., :3].astype(np.float32) @ W


def pair_stat(img, axis):
    """(wrap-around pair, mean of the interior pairs, std of the interior pairs) of the mean abs difference between
    neighbouring columns (axis 1) or rows (axis 0)."""
    a = img[..., :3].astype(np.float32)
    ext = np.concatenate([a, a.take([0], axis=axis)], axis=axis)
    d = np.abs(np.diff(ext, axis=axis)).mean(axis=(1 - axis, 2))
    return float(d[-1]), float(d[:-1].mean()), float(d[:-1].std())


def seamless(img, axis):
    wrap, mean, std = pair_stat(img, axis)
    return wrap <= mean + 4.0 * std


@pytest.fixture(scope="module")
def dress256():
    return ot.dress_pattern(size=256)


@pytest.fixture(scope="module")
def dress512():
    return ot.dress_pattern(size=512)


@pytest.fixture(scope="module")
def tex():
    """Every non-dress image at 256 px."""
    return {
        "frill": ot.frill_texture(size=256),
        "frill_inner": ot.frill_texture(size=256, inner=True),
        "satin": ot.satin_texture(size=256),
        "satin_print": ot.satin_texture(size=256, print=True),
        "leather": ot.leather_texture(size=256),
        "sole": ot.leather_texture(size=256, sole=True),
    }


# ------------------------------------------------------------------------------------------------ contract
def test_constants():
    assert ot.DEFAULT_COLORS == {
        "dress_base": "#141b17", "dress_leaf": "#2f5a3a", "dress_leaf_hi": "#3d6e48", "dress_leaf_dark": "#18291d",
        "dress_accent": "#2d5a6e", "frill": "#3f8f4f", "frill_inner": "#9ccb8f", "satin": "#17151a",
        "leather": "#1c1a1f", "sole": "#0f0e11"}
    assert ot.TILE_M == {"dress": 0.30, "frill_u": 0.08, "satin_u": 0.04, "leather": 0.10}


def test_module_is_numpy_only():
    """The module must run inside Blender's Python: stdlib + numpy only (no scipy, no PIL, no bpy)."""
    tree = ast.parse(Path(ot.__file__).read_text())
    roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0, "relative imports would pull in other modules"
            roots.add((node.module or "").split(".")[0])
    assert roots <= {"functools", "math", "dataclasses", "numpy"}, roots


def test_signatures_and_default_sizes():
    sig = inspect.signature
    assert sig(ot.dress_pattern).parameters["size"].default == 2048
    assert sig(ot.dress_pattern).parameters["seed"].default == 11
    assert sig(ot.frill_texture).parameters["size"].default == 1024
    assert sig(ot.frill_texture).parameters["seed"].default == 3
    assert sig(ot.satin_texture).parameters["size"].default == 512
    assert sig(ot.satin_texture).parameters["seed"].default == 5
    assert sig(ot.satin_texture).parameters["print"].default is False
    assert sig(ot.leather_texture).parameters["size"].default == 512
    assert sig(ot.leather_texture).parameters["seed"].default == 7
    assert sig(ot.toon_ramp).parameters["size"].default == 32
    assert sig(ot.sphere_map).parameters["size"].default == 128


def test_shapes_dtypes_alpha(dress256, tex):
    for name, img in dict(tex, dress=dress256).items():
        assert img.dtype == np.uint8, name
        assert img.shape == (256, 256, 4), name
        assert (img[..., 3] == 255).all(), name
    assert ot.dress_pattern(size=96).shape == (96, 96, 4)


def test_bad_arguments():
    with pytest.raises(ValueError):
        ot.toon_ramp("nope")
    with pytest.raises(ValueError):
        ot.sphere_map("nope")
    with pytest.raises(ValueError):
        ot.dress_pattern({"dress_base": "green"}, size=32)


# ----------------------------------------------------------------------------------------------- tileability
@pytest.mark.parametrize("axis", [0, 1])
def test_dress_tileable(dress256, dress512, axis):
    assert seamless(dress256, axis)
    assert seamless(dress512, axis)


def test_tileability_metric_sees_a_seam():
    """Control: the metric must fail on a real seam (two unrelated prints spliced; at most one of three seed pairs may
    slip through, because a column pair with little print on both sides differs little) and on a brightness step."""
    hits = 0
    for sa, sb in ((1, 2), (7, 8), (11, 12)):
        a, b = ot.dress_pattern(size=256, seed=sa), ot.dress_pattern(size=256, seed=sb)
        spliced = np.concatenate([a[:, :128], b[:, 128:]], axis=1)
        hits += (not seamless(spliced, 1)) and pair_stat(spliced, 1)[0] > 1.5 * pair_stat(a, 1)[1]
    assert hits >= 2
    assert not seamless(np.concatenate([a[:, :128], np.clip(a[:, 128:].astype(int) + 40, 0, 255).astype(np.uint8)],
                                       axis=1), 1)


def test_frill_satin_leather_tileable(tex):
    assert seamless(tex["frill"], 1)
    assert seamless(tex["frill_inner"], 1)
    assert seamless(tex["satin"], 1)
    assert seamless(tex["satin_print"], 1)
    for name in ("leather", "sole"):
        assert seamless(tex[name], 1) and seamless(tex[name], 0)


# ------------------------------------------------------------------------------------------------ determinism
def test_deterministic_by_seed(dress256, tex):
    assert np.array_equal(ot.dress_pattern(size=256), dress256)
    assert not np.array_equal(ot.dress_pattern(size=256, seed=12), dress256)
    assert np.array_equal(ot.frill_texture(size=256), tex["frill"])
    assert not np.array_equal(ot.frill_texture(size=256, seed=4), tex["frill"])
    assert np.array_equal(ot.satin_texture(size=256, print=True), tex["satin_print"])
    assert not np.array_equal(ot.satin_texture(size=256, seed=6), tex["satin"])
    assert np.array_equal(ot.leather_texture(size=256, sole=True), tex["sole"])
    assert not np.array_equal(ot.leather_texture(size=256, seed=8), tex["leather"])


# ----------------------------------------------------------------------------------------------------- dress
DRESS_BASE = np.array([0x14, 0x1B, 0x17], np.float32)                # dress_base #141b17


def dress_masks(img):
    rgb = img[..., :3].astype(np.float32)
    covered = np.abs(rgb - DRESS_BASE).sum(-1) > 30.0
    green = covered & (rgb[..., 1] > rgb[..., 2] + 4.0)
    teal = covered & (rgb[..., 2] > rgb[..., 1] + 4.0)
    return covered, green, teal


def test_dress_palette_statistics(dress512):
    rgb = dress512[..., :3].astype(np.float32)
    median = np.median(rgb.reshape(-1, 3), axis=0)
    assert np.abs(median - DRESS_BASE).max() <= 5.0                      # the ground is the near-black base colour
    assert rgb.reshape(-1, 3).mean(0).max() < 50.0                       # a black-ish dress overall
    covered, green, teal = dress_masks(dress512)
    assert 0.18 <= covered.mean() <= 0.25                                # a sparse floral print: ~21 % of the cloth
    assert green.sum() / covered.sum() >= 0.65                           # mostly leaf green
    assert 0.10 <= teal.sum() / covered.sum() <= 0.28                    # the big teal flowers
    # the leaf colours themselves: body (dress_leaf at 55 %), light half (dress_leaf_hi at 65 %), nothing glaring
    g = rgb[..., 1]
    assert abs(float(np.median(g[green])) - 64.0) < 8.0
    assert (g > 70).mean() > 0.05
    assert (g > 100).mean() < 0.02


def test_dress_leaf_family_is_dark_green():
    pal = ot._palette(None)
    base, leaf, hi, dk, acc = (pal[k] for k in ("dress_base", "dress_leaf", "dress_leaf_hi", "dress_leaf_dark",
                                                "dress_accent"))
    fam = ot._families(pal)
    assert np.allclose(fam["leaf"]["body"], base + 0.55 * (leaf - base))      # body 55 % of the way from the cloth
    assert np.allclose(fam["leaf"]["hi"], base + 0.65 * (hi - base))          # light half 65 %
    assert np.allclose(fam["leaf"]["dark"], base + 0.45 * (dk - base))        # veins / shadow side: same ratio to body
    assert np.allclose(fam["accent"]["body"], base + 0.70 * (acc - base))     # the teal family at 70 %
    assert np.allclose(fam["ghost"]["body"], base + 0.07 * (leaf - base))     # faint tone-on-tone underlayer
    luma = lambda c: float(c @ W)
    body, dark, light = fam["leaf"]["body"], fam["leaf"]["dark"], fam["leaf"]["hi"]
    assert luma(dark) < luma(body) < luma(light)
    assert 0.45 < luma(dark) / luma(body) < 0.65                              # the former shadow-to-body ratio (0.54)
    assert luma(fam["ghost"]["dark"]) < luma(base) < luma(fam["ghost"]["body"]) < 1.2 * luma(base)
    assert luma(light) < 0.7 * luma(ot._hex(ot.DEFAULT_COLORS["frill"]))      # the frills stay the brightest greens


def test_dress_print_is_not_busy(dress512):
    covered, _, _ = dress_masks(dress512)
    assert 0.18 <= covered.mean() <= 0.25                                     # pixel coverage ~21 %
    assert 0.200 <= ot._dress_layout(11).coverage <= 0.225                    # layout grid target ~0.21


def polyline_length(stem):
    return float(np.hypot(*np.diff(stem.pts, axis=0).T).sum())


def test_dress_layout_statistics():
    lay = ot._dress_layout(11)
    motifs = [p for p in lay.prims if isinstance(p, (ot._Flower, ot._Dot)) or
              (isinstance(p, ot._Leaf) and p.kind in ("leaf", "bud"))]
    accent = [p for p in motifs if p.accent]
    assert 0.08 <= len(accent) / len(motifs) <= 0.14                     # the flowers, and hardly anything else
    flowers = [p for p in motifs if isinstance(p, ot._Flower)]
    assert 4 <= len(flowers) <= 5                                        # flowers per 30 cm tile
    assert all(3.5 <= 2 * f.radius <= 4.5 for f in flowers)              # rosettes 3.5-4.5 cm across
    assert all(f.accent for f in flowers)
    leaves = [p for p in lay.prims if isinstance(p, ot._Leaf) and p.kind == "leaf"]
    assert len(leaves) >= 14
    assert all(4.5 <= lf.length <= 7.0 for lf in leaves)                 # leaves 4.5-7 cm
    assert sum(lf.accent for lf in leaves) <= 3 and any(not lf.accent for lf in leaves)
    assert 38 <= len(motifs) <= 50                                       # about 45 motifs per tile
    dots = sum(isinstance(p, ot._Dot) for p in motifs)
    assert 3 <= dots <= 12                                               # one berry twig and a few seeds
    twigs = [polyline_length(p) for p in lay.prims if isinstance(p, ot._Stem)]
    assert sum(9.0 <= t <= 13.0 for t in twigs) >= 4                     # three leafy twigs and the berry twig
    assert max(twigs) <= 13.0 + 1e-6
    assert any(isinstance(p, ot._Leaf) and p.kind == "ghost" for p in lay.prims)
    assert 0.200 <= lay.coverage <= 0.225
    assert ot._dress_layout(11) is lay                                   # cached per seed


def test_dress_layout_counts_over_seeds():
    for seed in range(6):
        lay = ot._dress_layout(seed)
        motifs = [p for p in lay.prims if isinstance(p, (ot._Flower, ot._Dot)) or
                  (isinstance(p, ot._Leaf) and p.kind in ("leaf", "bud"))]
        assert 4 <= sum(isinstance(p, ot._Flower) for p in motifs) <= 5, seed
        assert 34 <= len(motifs) <= 52, seed
        assert sum(isinstance(p, ot._Dot) for p in motifs) <= 12, seed
        assert 0.200 <= lay.coverage <= 0.225, seed


def test_dress_cloth_modulation_within_three_percent():
    m = ot._dress_cloth(256, ot._rng(11, 202))
    assert m.shape == (256, 256)
    assert 0.97 <= m.min() and m.max() <= 1.03
    assert m.std() > 0.002                                               # it is there, just invisible


def test_dress_independent_of_resolution(dress256, dress512):
    """The print is laid out in centimetres: the same motifs at any size."""
    small = lum(dress512).reshape(256, 2, 256, 2).mean(axis=(1, 3))
    ref = lum(dress256)
    c = np.corrcoef(small.ravel(), ref.ravel())[0, 1]
    assert c > 0.9


def test_dress_colors_override(dress256):
    img = ot.dress_pattern({"dress_base": "#404040", "dress_leaf": "#802020", "unknown_key": "#000000"}, size=256)
    median = np.median(img[..., :3].reshape(-1, 3), axis=0)
    assert np.abs(median - 0x40).max() <= 14.0
    _, green, _ = dress_masks(dress256)
    assert img[..., 0][green].mean() > dress256[..., 0][green].mean() + 15        # the leaves turned red
    assert ot.dress_pattern({"nonsense": "#123456"}, size=64).tolist() == ot.dress_pattern(None, size=64).tolist()


def test_dress_leaf_motifs_have_two_tones(dress512):
    """Leaves are two-tone with darker veins: the green channel inside the print is not flat."""
    _, green, _ = dress_masks(dress512)
    g = dress512[..., 1][green].astype(np.float32)
    assert np.percentile(g, 90) - np.percentile(g, 10) > 20.0


# ------------------------------------------------------------------------------------------------------ frill
def band(img, a, b):
    s = img.shape[0]
    return lum(img)[int(a * s):int(b * s)]


def test_frill_structure(tex):
    for name in ("frill", "frill_inner"):
        img = tex[name]
        body, hem, top = band(img, 0.45, 0.70).mean(), band(img, 0.975, 1.0).mean(), band(img, 0.0, 0.03).mean()
        assert hem > body * 1.06                                          # lighter hem band
        assert top < body * 0.80                                          # darker where it is sewn on (see below)
        # stitch lines: round dashes along x, so the row peak varies much more than in a plain cloth row
        ref = lum(img)[124:133].max(axis=0)
        for y in (0.06, 0.93):
            peak = lum(img)[int(y * 256) - 4:int(y * 256) + 5].max(axis=0)
            spread = np.percentile(peak, 90) - np.percentile(peak, 10)
            assert spread > 5.0 and spread > 1.8 * (np.percentile(ref, 90) - np.percentile(ref, 10))
    outer, inner = tex["frill"], tex["frill_inner"]
    assert lum(inner).mean() > lum(outer).mean() + 30
    mo, mi = outer[..., :3].reshape(-1, 3).mean(0), inner[..., :3].reshape(-1, 3).mean(0)
    assert mi[0] / mi[2] > mo[0] / mo[2]                                  # inner frill is warmer / yellower
    # the faint white scallop line near the hem exists on the inner frill only
    lace_rows = lum(inner)[int(0.78 * 256):int(0.90 * 256)]
    plain_rows = lum(outer)[int(0.78 * 256):int(0.90 * 256)]
    assert (lace_rows > np.median(lace_rows) + 8).mean() > 0.02
    assert (plain_rows > np.median(plain_rows) + 8).mean() < 0.005


@pytest.mark.parametrize("name, top, seam", [("frill", 0.50, 0.40), ("frill_inner", 0.72, 0.65)])
def test_frill_shaded_underside(tex, name, top, seam):
    """Where the ruffle meets the dress: a smooth gradient from x`top` at row 0 to x1.0 at 22 % of the rows, with a thin
    darker seam line (x`seam`) inside the first 2.5 % of the rows. The inner frill is gentler."""
    rows = lum(tex[name]).mean(axis=1)
    n = len(rows)
    r = rows / rows[int(0.45 * n):int(0.70 * n)].mean()
    k = int(0.025 * n)
    assert abs(r[0] - top) < 0.07                                         # row 0
    assert abs(r[:k].min() - seam) < 0.07 and int(np.argmin(r[:k])) > 0   # seam line, not at the very edge
    gradient = [r[int(f * n)] for f in (0.04, 0.08, 0.12, 0.17, 0.22)]
    assert all(a < b for a, b in zip(gradient, gradient[1:]))             # smooth, rising
    assert abs(r[int(0.22 * n)] - 1.0) < 0.07 and abs(r[int(0.30 * n)] - 1.0) < 0.07
    assert r[:int(0.22 * n)].mean() < 0.9 and r[int(0.35 * n):int(0.9 * n)].min() > 0.9


def test_frill_inner_underside_is_gentler(tex):
    ro, ri = lum(tex["frill"]).mean(axis=1), lum(tex["frill_inner"]).mean(axis=1)
    ro, ri = ro / ro[115:180].mean(), ri / ri[115:180].mean()
    assert ri[0] > ro[0] + 0.15 and ri[:6].min() > ro[:6].min() + 0.15
    assert (ri[:int(0.15 * 256)] > ro[:int(0.15 * 256)]).all()           # the two meet again near 22 %


def test_frill_colours(tex):
    m = tex["frill"][..., :3].reshape(-1, 3).mean(0)
    assert m[1] > m[0] + 40 and m[1] > m[2] + 30                            # green
    base = np.array([0x3F, 0x8F, 0x4F], np.float32)
    mid = band(tex["frill"], 0.4, 0.7).mean()
    assert abs(mid - float(base @ W)) < 0.12 * float(base @ W)


# ------------------------------------------------------------------------------------------------------ satin
def test_satin_ribbon(tex):
    img = tex["satin"]
    L = lum(img)
    edge = np.concatenate([L[:int(0.04 * 256)].ravel(), L[-int(0.04 * 256):].ravel()]).mean()
    centre = L[int(0.15 * 256):int(0.85 * 256)].mean()
    assert edge > centre + 5.0                                            # lighter selvedge line
    assert abs(centre - float(np.array([24, 24, 29], np.float32) @ W)) < 3.0   # near-black satin colour
    blocks = L[int(0.1 * 256):int(0.9 * 256)][:192].reshape(24, 8, 32, 8).mean(axis=(1, 3))
    assert np.abs(blocks / blocks.mean() - 1.0).max() < 0.05              # sheen within a few percent


def test_satin_print_is_sparse_and_faint(tex):
    plain, printed = tex["satin"], tex["satin_print"]
    diff = np.abs(printed[..., :3].astype(np.int32) - plain[..., :3].astype(np.int32)).max(-1)
    assert 0.01 < (diff > 3).mean() < 0.15
    assert diff.max() <= 0.22 * 255 + 2                                     # alpha 22 % of at most white
    assert (printed[..., :3].astype(np.int32) >= plain[..., :3].astype(np.int32) - 1).all()   # only ever lighter


# ---------------------------------------------------------------------------------------------------- leather
def test_leather_and_sole(tex):
    leather, sole = tex["leather"], tex["sole"]
    ml, ms = leather[..., :3].reshape(-1, 3).mean(0), sole[..., :3].reshape(-1, 3).mean(0)
    assert np.abs(ml - [0x1C, 0x1A, 0x1F]).max() < 4.0
    assert np.abs(ms - [0x0F, 0x0E, 0x11]).max() < 4.0
    for img in (leather, sole):
        L = lum(img)
        c = L[96:160, 96:160].mean()
        corners = np.mean([L[:48, :48].mean(), L[:48, -48:].mean(), L[-48:, :48].mean(), L[-48:, -48:].mean()])
        assert c > corners + 0.3                                         # barely visible lighter wear in the middle
        assert c < corners + 8.0
        assert L.std() > 0.3                                              # a fine grain


# ------------------------------------------------------------------------------------------------ toon / sphere
@pytest.mark.parametrize("kind", ["dress", "frill", "satin", "leather"])
def test_toon_ramp(kind):
    t = ot.toon_ramp(kind)
    assert t.shape == (32, 32, 4) and t.dtype == np.uint8
    assert (t == t[:, :1]).all()                                          # one colour per row
    assert (t[..., 3] == 255).all()
    assert (t[0, 0] == 255).all()                                         # top: fully lit, pure white
    col = t[:, 0, :3].astype(np.int32)
    assert (np.diff(col, axis=0) <= 0).all()                              # monotone: lit -> shaded
    assert (col[-1] < 235).all()
    assert ot.toon_ramp(kind, 16).shape == (16, 16, 4)
    total = (col[0] - col[-1]).max()
    if kind in ("dress", "frill"):
        assert np.abs(np.diff(col, axis=0)).max() < 0.65 * total          # a narrow but smooth step, not a cliff


def test_toon_ramp_tints_and_multipliers():
    dress = ot.toon_ramp("dress")[-1, 0, :3] / 255.0
    assert dress[1] > dress[0] + 0.1 and dress[2] > dress[0] + 0.1        # shadow tinted teal-green, not grey
    assert dress[1] >= dress[2]
    assert np.allclose(dress, [0.42, 0.57, 0.56], atol=0.01)
    frill = ot.toon_ramp("frill")[-1, 0, :3] / 255.0
    assert abs(frill[1] - 0.62) < 0.01 and frill[0] < frill[1] and frill[2] < frill[1]
    assert frill @ W > dress @ W + 0.04                                      # the frill's shadow is the lighter one
    leather = ot.toon_ramp("leather")[-1, 0, :3] / 255.0
    assert leather[2] > leather[0] + 0.1                                    # slightly bluish shadow
    satin = ot.toon_ramp("satin")
    assert satin[-1, 0, 0] > 150                                            # very soft: the shadow stays light
    steps = np.abs(np.diff(satin[:, 0, 0].astype(int)))
    assert steps.max() <= 12


SPHERE_PEAK = {"satin": 41, "leather": 82}               # 0.16 / 0.32 of white: anything brighter greys near-black cloth


@pytest.mark.parametrize("kind", ["satin", "leather"])
def test_sphere_map(kind):
    s = ot.sphere_map(kind)
    assert s.shape == (128, 128, 4) and s.dtype == np.uint8
    assert (s[..., 3] == 255).all()
    ring = np.concatenate([s[0, :, :3].ravel(), s[-1, :, :3].ravel(), s[:, 0, :3].ravel(), s[:, -1, :3].ravel()])
    assert (ring == 0).all()                                                # black border
    L = s[..., :3].astype(int).sum(-1)
    y, x = np.unravel_index(np.argmax(L), L.shape)
    assert x < 64 and y < 64                                                # highlight high left
    peak = int(s[..., :3].max())
    assert 0.6 * SPHERE_PEAK[kind] <= peak <= SPHERE_PEAK[kind]             # a weak highlight: it is ADDED to black
    assert ot.sphere_map(kind, 32).shape == (32, 32, 4)


def test_sphere_leather_sharper_than_satin():
    sat, lea = ot.sphere_map("satin")[..., 0].astype(float), ot.sphere_map("leather")[..., 0].astype(float)
    assert lea.max() > sat.max()
    assert (lea > 0.5 * lea.max()).sum() < 0.3 * (sat > 0.5 * sat.max()).sum()


# --------------------------------------------------------------------------------------------- build_textures
class StubCtx:
    def __init__(self, spec=None):
        self.spec = spec if spec is not None else {}
        self.saved = {}

    def save_png(self, name, rgba):
        assert rgba.dtype == np.uint8 and rgba.ndim == 3 and rgba.shape[2] == 4
        self.saved[name] = rgba
        return "model_" + name


KEYS = ["dress", "frill", "frill_inner", "satin", "satin_print", "leather", "sole", "toon_dress", "toon_frill",
        "toon_satin", "toon_leather", "sphere_satin", "sphere_leather"]
SMALL = {"dress": 96, "frill": 64, "satin": 64, "leather": 64}


def test_build_textures_returns_every_key():
    ctx = StubCtx()
    out = ot.build_textures(ctx, sizes=SMALL)
    assert sorted(out) == sorted(KEYS)
    assert out["dress"] == "model_outfit_dress.png"
    assert sorted(ctx.saved) == sorted(f"outfit_{k}.png" for k in KEYS)
    assert ctx.saved["outfit_dress.png"].shape == (96, 96, 4)
    assert ctx.saved["outfit_toon_dress.png"].shape == (32, 32, 4)
    assert ctx.saved["outfit_sphere_satin.png"].shape == (128, 128, 4)
    assert ot._SIZES["dress"] == 2048 and ot._SIZES["frill"] == 1024
    assert ot._SIZES["satin"] == 512 and ot._SIZES["leather"] == 512


def test_build_textures_black_items_follow_the_project_palette():
    spec = {"colors": {"black": {"ribbon": "#401010", "shoe": "#104010", "shoe_sole": "#101040"},
                       "outfit": {"leather": "#202020"}}}
    ctx = StubCtx(spec)
    ot.build_textures(ctx, sizes=SMALL)
    mean = {k: ctx.saved[f"outfit_{k}.png"][..., :3].reshape(-1, 3).mean(0) for k in ("satin", "leather", "sole")}
    assert mean["satin"][0] > mean["satin"][1] + 12 and mean["satin"][0] > mean["satin"][2] + 12      # ribbon -> satin
    assert mean["sole"][2] > mean["sole"][0] + 12 and mean["sole"][2] > mean["sole"][1] + 12          # shoe_sole -> sole
    assert abs(mean["leather"][0] - mean["leather"][1]) < 6                                           # colors.outfit beat shoe
    plain = StubCtx()
    ot.build_textures(plain, sizes=SMALL)
    ps = plain.saved["outfit_sole.png"][..., :3].reshape(-1, 3).mean(0)
    assert np.abs(ps - [0x0F, 0x0E, 0x11]).max() < 6                                                  # default: black, not brown


def test_build_textures_colour_precedence():
    spec = {"colors": {"outfit": {"frill": "#c04040", "satin": "#303030"}}}
    ctx = StubCtx(spec)
    ot.build_textures(ctx, colors={"satin": "#505050"}, sizes=SMALL)
    fr = ctx.saved["outfit_frill.png"][..., :3].reshape(-1, 3).mean(0)
    assert fr[0] > fr[1] + 40                                              # the spec colour reached the frill
    sa = ctx.saved["outfit_satin.png"][..., :3].reshape(-1, 3).mean(0)
    assert abs(sa.mean() - 0x50) < 14                                       # the argument beat the spec
    assert StubCtx().spec == {} and ot.build_textures(StubCtx({"colors": None}), sizes=SMALL)
    plain = StubCtx()
    ot.build_textures(plain, sizes=SMALL)
    assert np.array_equal(plain.saved["outfit_dress.png"], ctx.saved["outfit_dress.png"])
