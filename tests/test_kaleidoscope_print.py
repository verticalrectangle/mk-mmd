"""The kaleidoscope toy's printed paper (mkmmd/core/kaleidoscope_print.py), without Blender: the colour roles, the print of
every pattern (sizes, determinism, millimetre layouts, the ground and every ink showing, a dark palette still reading), the
wear map and the view through the eyepiece."""
import ast
from pathlib import Path

import numpy as np
import pytest

from mkmmd.core import kaleidoscope_print as KP
from mkmmd.core import palette as PAL

DAWN = PAL.PALETTES["rose-pine-dawn"]
MOON = PAL.PALETTES["rose-pine-moon"]


def _lum(img):
    """Relative luminance (linear light) of uint8 sRGB pixels."""
    c = np.asarray(img, float) / 255.0
    lin = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    return lin @ np.array([0.2126, 0.7152, 0.0722])


def _blocks(img, n):
    """Means over n x n pixel blocks (the image cropped to whole blocks)."""
    h, w = img.shape[0] // n * n, img.shape[1] // n * n
    a = np.asarray(img, float)[:h, :w]
    return a.reshape(h // n, n, w // n, n, *a.shape[2:]).mean((1, 3))


def _shares(img, cols, tol=0.09):
    """{role: share of the pixels whose nearest role colour it is, within tol (sRGB 0..1)}."""
    px = img.reshape(-1, 3).astype(float) / 255.0
    names = list(cols)
    C = np.array([cols[n] for n in names])
    d = ((px[:, None, :] - C[None]) ** 2).sum(-1)
    near, ok = d.argmin(1), d.min(1) < tol * tol
    return {n: float(((near == i) & ok).mean()) for i, n in enumerate(names)}


# ===================================================================================================================
# colours
# ===================================================================================================================
def test_roles_list_the_ground_first_for_every_pattern():
    assert KP.PATTERNS == ("roses", "memphis", "marbled") and set(KP.ROLES) == set(KP.PATTERNS)
    for p in KP.PATTERNS:
        assert list(KP.ROLES[p])[0] == "ground" and len(KP.ROLES[p]) >= 4


@pytest.mark.parametrize("name", sorted(PAL.PALETTES))
def test_colors_resolve_the_default_slots_in_role_order(name):
    pal = PAL.PALETTES[name]
    for p in KP.PATTERNS:
        c = KP.colors(p, pal)
        assert list(c) == list(KP.ROLES[p])
        for role, slot in KP.ROLES[p].items():
            assert c[role] == pytest.approx(PAL.srgb(pal[slot]))
            assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in c[role])


def test_colors_take_slot_and_hex_overrides():
    c = KP.colors("roses", DAWN, {"leaf": "iris", "rose": "#FF0080"})
    assert c["leaf"] == pytest.approx(PAL.srgb(DAWN["iris"]))
    assert c["rose"] == pytest.approx((1.0, 0.0, 128 / 255))
    assert c["petal"] == pytest.approx(PAL.srgb(DAWN["rose"]))           # the roles not overridden keep their slot
    assert KP.colors("memphis", DAWN, {"line": "love"})["line"] == pytest.approx(PAL.srgb(DAWN["love"]))


@pytest.mark.parametrize("pattern, over", [("roses", {"stem": "pine"}), ("roses", {"leaf": "emerald"}),
                                           ("roses", {"leaf": "#12345"}), ("memphis", {"rose": "love"}),
                                           ("marbled", {"line": "text"}), ("plaid", None)])
def test_colors_refuse_unknown_roles_slots_and_patterns(pattern, over):
    with pytest.raises(ValueError):
        KP.colors(pattern, DAWN, over)


def test_print_refuses_an_unknown_pattern_missing_colours_and_an_empty_sheet():
    cols = KP.colors("roses", DAWN)
    with pytest.raises(ValueError):
        KP.paper_print("plaid", cols)
    with pytest.raises(ValueError):
        KP.paper_print("memphis", cols)                                  # roses' colours lack memphis' inks
    with pytest.raises(ValueError):
        KP.paper_print("roses", cols, size_mm=(0.0, 20.0), px_per_mm=5.0)
    with pytest.raises(ValueError):
        KP.paper_wear(size_mm=(20.0, 20.0), px_per_mm=0.0)
    with pytest.raises(ValueError):
        KP.view(cols, px=4)


def test_module_needs_nothing_but_numpy():
    """It runs inside Blender's Python: no PIL, cv2 or scipy, nothing but math, numpy and the palettes."""
    tree = ast.parse(Path(KP.__file__).read_text())
    mods = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            mods |= ({"." + a.name for a in node.names} if node.level else {(node.module or "").split(".")[0]})
    assert mods <= {"math", "numpy", ".palette"}, mods


# ===================================================================================================================
# the print
# ===================================================================================================================
@pytest.mark.parametrize("size, ppm", [((153.94, 192.0), 2.0), ((154.0, 204.0), 1.5), ((33.3, 21.7), 3.3)])
def test_sheet_size_follows_the_millimetres(size, ppm):
    W, H = round(size[0] * ppm), round(size[1] * ppm)
    for p in KP.PATTERNS:
        img = KP.paper_print(p, KP.colors(p, DAWN), seed=2, size_mm=size, px_per_mm=ppm)
        assert img.shape == (H, W, 3) and img.dtype == np.uint8
    w = KP.paper_wear(seed=2, size_mm=size, px_per_mm=ppm)
    assert w.shape == (H, W) and w.dtype == np.float32


@pytest.mark.parametrize("pattern", KP.PATTERNS)
def test_print_is_deterministic_and_seeded(pattern):
    cols = KP.colors(pattern, DAWN)
    a = KP.paper_print(pattern, cols, seed=5, size_mm=(60.0, 70.0), px_per_mm=4.0)
    assert np.array_equal(a, KP.paper_print(pattern, cols, seed=5, size_mm=(60.0, 70.0), px_per_mm=4.0))
    b = KP.paper_print(pattern, cols, seed=6, size_mm=(60.0, 70.0), px_per_mm=4.0)
    assert np.abs(a.astype(int) - b).mean() > 4.0                       # another layout, not another grain


@pytest.mark.parametrize("pattern", KP.PATTERNS)
def test_layout_is_in_millimetres(pattern):
    """5 and 10 px/mm print the same motifs in the same places: averaged over millimetre blocks they agree far more
    closely than two seeds do."""
    cols = KP.colors(pattern, DAWN)
    size = (64.0, 80.0)
    fine = _blocks(KP.paper_print(pattern, cols, seed=3, size_mm=size, px_per_mm=10.0), 10)
    coarse = _blocks(KP.paper_print(pattern, cols, seed=3, size_mm=size, px_per_mm=5.0), 5)
    other = _blocks(KP.paper_print(pattern, cols, seed=4, size_mm=size, px_per_mm=5.0), 5)
    same, apart = np.abs(fine - coarse).mean(), np.abs(other - coarse).mean()
    assert same < 3.0 and same < 0.15 * apart, (same, apart)


@pytest.mark.parametrize("name", ["rose-pine-dawn", "rose-pine-moon"])
@pytest.mark.parametrize("pattern", KP.PATTERNS)
def test_ground_dominates_and_every_ink_shows(pattern, name):
    """The ground is the biggest colour and every role's ink prints somewhere, in a light and in a dark palette (on a dark
    ground the inks are knocked out of it and still show their own colours)."""
    cols = KP.colors(pattern, PAL.PALETTES[name])
    sh = _shares(KP.paper_print(pattern, cols, seed=1, px_per_mm=3.0), cols)
    ground = sh.pop("ground")
    assert ground > 0.4 and ground > 2.0 * max(sh.values()), (ground, sh)
    assert min(sh.values()) > 0.002, sh


def test_the_rose_has_depth_darker_than_its_ink():
    """The rose's depth is its main ink printed twice: pixels darker than the main ink in its hue show up."""
    cols = KP.colors("roses", DAWN)
    img = KP.paper_print("roses", cols, seed=1, px_per_mm=3.0).reshape(-1, 3) / 255.0
    deep = np.asarray(cols["rose"]) ** 2
    assert (np.abs(img - deep).max(1) < 0.08).mean() > 0.003


# ===================================================================================================================
# wear
# ===================================================================================================================
@pytest.fixture(scope="module")
def wear():
    return KP.paper_wear(seed=2, size_mm=(154.0, 204.0), px_per_mm=5.0)


def test_wear_is_strongest_at_the_ends_and_the_seam(wear):
    ppm = 5
    assert wear.dtype == np.float32 and wear.min() >= 0.0 and wear.max() <= 1.0
    assert wear[:ppm // 2].mean() > 0.6 and wear[-(ppm // 2):].mean() > 0.6        # the rims
    assert wear[:, :1].mean() > 0.4 and wear[:, -1:].mean() > 0.4                     # the seam
    mid = wear[15 * ppm:-15 * ppm, 10 * ppm:-10 * ppm]
    assert np.median(mid) < 0.01 and (mid > 0.1).mean() < 0.05
    assert mid.max() > 0.2                                               # a few scuffs and scratches all the same


def test_rim_wear_is_a_ragged_band_not_a_gradient(wear):
    ppm = 5
    for band in (wear[:8 * ppm], wear[-8 * ppm:][::-1]):
        depth = (band >= 0.5).sum(0) / ppm                               # mm rubbed through, column by column
        assert 1.0 < depth.mean() < 5.0 and depth.std() > 0.25, (depth.mean(), depth.std())
        rows = band.mean(1)
        assert rows[0] > 0.6 and rows[-1] < 0.05                         # gone within a few mm


def test_wear_is_deterministic_seeded_and_in_millimetres():
    a = KP.paper_wear(seed=3, size_mm=(60.0, 80.0), px_per_mm=10.0)
    assert np.array_equal(a, KP.paper_wear(seed=3, size_mm=(60.0, 80.0), px_per_mm=10.0))
    b = KP.paper_wear(seed=3, size_mm=(60.0, 80.0), px_per_mm=5.0)
    c = KP.paper_wear(seed=4, size_mm=(60.0, 80.0), px_per_mm=5.0)
    same, apart = np.abs(_blocks(a, 10) - _blocks(b, 5)).mean(), np.abs(_blocks(c, 5) - _blocks(b, 5)).mean()
    assert same < 0.35 * apart, (same, apart)


# ===================================================================================================================
# the view through the eyepiece
# ===================================================================================================================
def _turned(img, deg):
    """img sampled at its pixels turned by deg about the centre (nearest pixel)."""
    n = img.shape[0]
    yy, xx = (np.mgrid[0:n, 0:n] + 0.5 - 0.5 * n)
    a = np.radians(deg)
    xs = np.cos(a) * xx - np.sin(a) * yy + 0.5 * n
    ys = np.sin(a) * xx + np.cos(a) * yy + 0.5 * n
    return img[np.clip(np.floor(ys).astype(int), 0, n - 1), np.clip(np.floor(xs).astype(int), 0, n - 1)]


def _radius(n):
    yy, xx = (np.mgrid[0:n, 0:n] + 0.5 - 0.5 * n)
    return np.hypot(xx, yy) / (KP.VIEW_RADIUS * n)


@pytest.mark.parametrize("pattern, name, seed", [("roses", "rose-pine-dawn", 1), ("memphis", "rose-pine-moon", 7),
                                                 ("marbled", "rose-pine-dawn", 3)])
def test_view_is_a_six_fold_star(pattern, name, seed):
    img = KP.view(KP.colors(pattern, PAL.PALETTES[name]), seed=seed, px=384).astype(float)
    inside = _radius(384) < 0.9
    d60 = np.abs(_turned(img, 60.0) - img)[inside].mean()
    d23 = np.abs(_turned(img, 23.0) - img)[inside].mean()
    assert d60 < 0.3 * d23, (d60, d23)
    assert np.abs(img[:, ::-1] - img)[inside].mean() < 0.5                # mirrored left to right
    assert np.abs(img[::-1] - img)[inside].mean() < 0.5                   # and top to bottom


@pytest.mark.parametrize("name", ["rose-pine-dawn", "rose-pine-moon"])
def test_view_is_dark_outside_its_circle_with_a_soft_edge(name):
    cols = KP.colors("roses", PAL.PALETTES[name])
    img = KP.view(cols, seed=2, px=256)
    L, r = _lum(img), _radius(256)
    assert L[r > 1.02].max() < 0.02                                      # near-black outside
    assert L[r < 0.6].mean() > 3.0 * L[r > 1.02].mean() + 0.01
    assert L[(r > 0.95) & (r < 1.0)].mean() < L[(r > 0.5) & (r < 0.7)].mean()      # darkening into the rim


def test_view_is_deterministic_and_seeded():
    cols = KP.colors("roses", DAWN)
    a = KP.view(cols, seed=4, px=200)
    assert a.shape == (200, 200, 3) and a.dtype == np.uint8
    assert np.array_equal(a, KP.view(cols, seed=4, px=200))
    assert np.abs(a.astype(int) - KP.view(cols, seed=5, px=200)).mean() > 4.0


def test_view_shows_the_inks():
    cols = KP.colors("memphis", DAWN)
    sh = _shares(KP.view(cols, seed=1, px=384), cols, tol=0.12)
    assert sum(v > 0.005 for k, v in sh.items() if k != "ground") >= 4, sh
