"""Tests for mkmmd.model.tex: colours, images, blends, noise, gradients, the antialiased Canvas, MMD toon / sphere
maps and the atlas packer. Everything is bpy-free and writes only into pytest's tmp_path."""
import numpy as np
import pytest
from PIL import Image

from mkmmd.model import tex

RED, GREEN, BLUE, WHITE, BLACK = "#ff0000", "#00ff00", "#0000ff", "#ffffff", "#000000"


def area(img):
    """Sum of alpha = covered area in pixels."""
    return float(img[..., 3].sum())


# ====================================================================================================== colours
def test_color_parsing():
    assert np.allclose(tex.color("#f80"), [1.0, 136 / 255, 0.0, 1.0])
    assert np.allclose(tex.color("#336699"), [0.2, 0.4, 0.6, 1.0])
    assert np.allclose(tex.color("#33669980"), [0.2, 0.4, 0.6, 128 / 255])
    assert np.allclose(tex.color("#abcd"), np.array([0xAA, 0xBB, 0xCC, 0xDD]) / 255)
    assert np.allclose(tex.color("336699"), [0.2, 0.4, 0.6, 1.0])                    # '#' optional
    c = tex.color("#336699")
    assert c.dtype == np.float32 and c.shape == (4,)


def test_color_from_numbers_and_arrays():
    assert np.allclose(tex.color((1, 0, 0)), [1, 0, 0, 1])
    assert np.allclose(tex.color((0.1, 0.2, 0.3, 0.4)), [0.1, 0.2, 0.3, 0.4])
    assert np.allclose(tex.color([0.5, 0.5, 0.5]), [0.5, 0.5, 0.5, 1])
    assert np.allclose(tex.color(0.25), [0.25, 0.25, 0.25, 1])
    arr = np.array([1.0, 0.5, 0.25, 0.5], np.float32)
    out = tex.color(arr)
    out[0] = 0.0
    assert arr[0] == 1.0                                                              # a copy, never the input
    assert np.allclose(tex.color(np.array([255, 128, 0], np.uint8)), [1, 128 / 255, 0, 1])
    assert np.allclose(tex.color(np.array([1.5, -0.2, 0.5])), [1.0, 0.0, 0.5, 1.0])   # arrays are clipped silently


@pytest.mark.parametrize("bad", ["#12", "#12345", "red", "#gggggg", (255, 0, 0), (1, 2), (1, 0, 0, 0, 0), None, [-1, 0, 0]])
def test_color_rejects_garbage(bad):
    with pytest.raises((ValueError, TypeError)):
        tex.color(bad)


def test_srgb_roundtrip_and_known_values():
    x = np.linspace(0.0, 1.0, 1001)
    assert np.abs(tex.linear_to_srgb(tex.srgb_to_linear(x)) - x).max() < 1e-9
    assert np.isclose(tex.srgb_to_linear(0.5), 0.2140411, atol=1e-6)
    assert np.isclose(tex.linear_to_srgb(0.2140411), 0.5, atol=1e-6)
    assert tex.srgb_to_linear(0.0) == 0.0 and np.isclose(tex.srgb_to_linear(1.0), 1.0)
    assert np.isclose(tex.srgb_to_linear(0.02), 0.02 / 12.92)                          # linear toe
    assert np.all(np.diff(tex.srgb_to_linear(x)) > 0)
    assert tex.srgb_to_linear(np.float32([0.3])).dtype == np.float32
    assert tex.srgb_to_linear(np.array([-1.0, 2.0])).tolist() == [0.0, 1.0]           # clipped


def test_mix():
    a, b = tex.color("#102030"), tex.color("#a0b0c0")
    assert np.allclose(tex.mix(a, b, 0.0), a, atol=1e-6) and np.allclose(tex.mix(a, b, 1.0), b, atol=1e-6)
    assert np.allclose(tex.mix(BLACK, WHITE, 0.5), [0.73536] * 3 + [1.0], atol=1e-4)          # linear light
    assert np.allclose(tex.mix(BLACK, WHITE, 0.5, linear=False), [0.5, 0.5, 0.5, 1.0])
    assert tex.mix(a, b, 0.3).shape == (4,) and tex.mix(a, b, 0.3).dtype == np.float32
    t = np.linspace(0, 1, 6).reshape(2, 3)
    m = tex.mix(RED, BLUE, t, linear=False)
    assert m.shape == (2, 3, 4) and np.allclose(m[1, 2], tex.color(BLUE)) and np.allclose(m[0, 0], tex.color(RED))
    img = tex.new(2, 2, GREEN)
    assert tex.mix(img, tex.new(2, 2, RED), 0.5, linear=False).shape == (2, 2, 4)
    # alpha is interpolated and colours are weighted by alpha: fading to a transparent colour keeps the red
    f = tex.mix(RED, "#0000ff00", 0.5, linear=False)
    assert np.allclose(f, [1, 0, 0, 0.5])


def test_shade():
    c = tex.color("#336699")
    assert np.allclose(tex.shade(c, 1.0), c, atol=1e-6)
    assert np.allclose(tex.shade("#808080", 0.5)[:3], [0.3622] * 3, atol=1e-3)         # halved in linear light
    assert np.allclose(tex.shade("#336699", 0.0)[:3], 0.0)
    assert np.allclose(tex.shade("#336699", 100.0)[:3], 1.0)
    assert np.isclose(tex.shade("#33669940", 0.5)[3], 64 / 255)                         # alpha kept
    darker, lighter = tex.shade(c, 0.7), tex.shade(c, 1.3)
    assert np.all(darker[:3] < c[:3]) and np.all(lighter[:3] > c[:3])
    assert tex.shade(tex.new(3, 2, "#808080"), 0.5).shape == (2, 3, 4)


def test_shift_hue():
    assert np.allclose(tex.shift_hue(RED, 120), tex.color(GREEN), atol=1e-6)
    assert np.allclose(tex.shift_hue(RED, 240), tex.color(BLUE), atol=1e-6)
    assert np.allclose(tex.shift_hue(RED, 360), tex.color(RED), atol=1e-6)
    assert np.allclose(tex.shift_hue(RED, -120), tex.color(BLUE), atol=1e-6)
    assert np.allclose(tex.shift_hue("#808080", 77), tex.color("#808080"), atol=1e-6)      # grey has no hue
    assert np.allclose(tex.shift_hue("#336699", 30), tex.color("#333399"), atol=1e-6)      # 210 deg -> 240 deg
    assert np.isclose(tex.shift_hue("#ff000040", 90)[3], 64 / 255)
    img = tex.new(2, 2, RED)
    assert np.allclose(tex.shift_hue(img, 120)[1, 1], tex.color(GREEN), atol=1e-6)


def test_lighten_darken_hsv_hex():
    c = tex.color("#336699")
    assert np.allclose(tex.lighten(c, 1.0), [1, 1, 1, 1]) and np.allclose(tex.darken(c, 1.0), [0, 0, 0, 1])
    assert np.allclose(tex.lighten(c, 0.0), c, atol=1e-6) and np.allclose(tex.darken(c, 0.0), c, atol=1e-6)
    assert np.all(tex.lighten(c, 0.3)[:3] > c[:3]) and np.all(tex.darken(c, 0.3)[:3] < c[:3])
    assert np.isclose(tex.lighten("#33669080", 0.5)[3], 128 / 255)                      # alpha kept
    assert np.isclose(tex.darken(c, 0.5, linear=False)[0], 0.1)
    assert np.allclose(tex.hsv(0, 1, 1), [1, 0, 0, 1]) and np.allclose(tex.hsv(120, 1, 1), [0, 1, 0, 1])
    assert np.allclose(tex.hsv(240, 0.5, 1.0, 0.5), [0.5, 0.5, 1.0, 0.5])
    assert tex.to_hex("#336699") == "#336699" and tex.to_hex((1, 1, 1, 0.5)) == "#ffffff80"
    assert tex.to_hex(tex.color("#f1e7d6")) == "#f1e7d6"
    s = tex.smoothstep(0.2, 0.6, np.array([0.0, 0.2, 0.4, 0.6, 1.0]))
    assert np.allclose(s, [0, 0, 0.5, 1, 1]) and tex.smoothstep(0.5, 0.5, 0.4) == 0.0


# ======================================================================================================= images
def test_new_and_conversions():
    img = tex.new(5, 3, "#ff800080")
    assert img.shape == (3, 5, 4) and img.dtype == np.float32
    assert np.allclose(img[2, 4], [1, 128 / 255, 0, 128 / 255])
    assert tex.new(2, 2).sum() == 0.0
    u = tex.to_uint8(np.array([[[0.0, 0.5, 1.0, 1 / 255]]], np.float32))
    assert u.dtype == np.uint8 and u.tolist() == [[[0, 128, 255, 1]]]                   # rounds to nearest
    assert tex.to_uint8(np.full((1, 1, 4), 2.0)).tolist() == [[[255] * 4]]               # clipped
    assert tex.to_uint8(np.full((1, 1, 4), -1.0)).tolist() == [[[0] * 4]]
    grey = tex.to_uint8(np.full((2, 3), 0.5, np.float32))
    assert grey.shape == (2, 3, 4) and grey[0, 0].tolist() == [128, 128, 128, 255]
    rgb = tex.to_uint8(np.full((2, 3, 3), 1.0, np.float32))
    assert rgb.shape == (2, 3, 4) and rgb[0, 0].tolist() == [255, 255, 255, 255]
    one = tex.to_uint8(np.full((2, 3, 1), 1.0, np.float32))
    assert one.shape == (2, 3, 4) and one[0, 0].tolist() == [255, 255, 255, 255]
    src = np.arange(24, dtype=np.uint8).reshape(2, 3, 4)
    out = tex.to_uint8(src)
    assert np.array_equal(out, src) and out is not src                                  # uint8 passes through (copy)
    assert tex.to_uint8(np.full((2, 2, 3), 200, np.uint8))[0, 0].tolist() == [200, 200, 200, 255]
    f = tex.to_float(src)
    assert f.dtype == np.float32 and f.shape == (2, 3, 4) and np.isclose(f[0, 0, 3], 3 / 255)
    assert tex.to_float(np.full((2, 2), 3.0)).max() == 1.0
    assert tex.to_float(np.zeros((2, 2, 3), np.float64)).shape == (2, 2, 4)
    assert tex.to_float(np.zeros((2, 2, 3), np.float64))[0, 0, 3] == 1.0
    with pytest.raises(TypeError):
        tex.to_float(np.zeros((2, 2, 4), np.int64))
    with pytest.raises(ValueError):
        tex.to_float(np.zeros((2, 2, 2), np.float32))
    rt = np.random.default_rng(0).random((8, 9, 4)).astype(np.float32)
    assert np.abs(tex.to_float(tex.to_uint8(rt)) - rt).max() <= 0.5 / 255 + 1e-6


def test_dither_keeps_exact_values_and_is_deterministic():
    flat = tex.to_float(np.full((8, 8, 4), 100, np.uint8))
    assert np.array_equal(tex.to_uint8(flat, dither=True), np.full((8, 8, 4), 100, np.uint8))
    ramp = np.tile(np.linspace(0.30, 0.31, 256, dtype=np.float32)[None, :, None], (4, 1, 4))
    d1, d2 = tex.to_uint8(ramp, dither=True), tex.to_uint8(ramp, dither=True)
    assert np.array_equal(d1, d2)
    assert abs(d1[..., 0].mean() / 255 - ramp[..., 0].mean()) < 0.002                    # unbiased on average
    assert np.all(d1[..., 3] == tex.to_uint8(ramp)[..., 3])                              # alpha never dithered


def test_png_roundtrip_rgba8_straight_alpha(tmp_path):
    rng = np.random.default_rng(1)
    u8 = rng.integers(0, 256, (7, 11, 4), dtype=np.uint8)
    u8[0, 0] = (255, 0, 0, 0)                                                            # invisible but red
    u8[1, 1] = (51, 102, 153, 128)                                                       # half transparent
    img = tex.to_float(u8)
    path = tex.save_png(tmp_path / "a" / "b" / "t.png", img)                              # creates the folders
    assert path.exists() and path.parent.name == "b"
    with Image.open(path) as im:
        assert im.mode == "RGBA" and im.size == (11, 7)
        stored = np.asarray(im)
    assert np.array_equal(stored, u8)                                                    # no premultiplication
    assert np.array_equal(tex.to_uint8(tex.load_png(path)), u8)
    loaded = tex.load_png(path)
    assert loaded.dtype == np.float32 and loaded.shape == (7, 11, 4)
    assert np.allclose(loaded[1, 1], [0.2, 0.4, 0.6, 128 / 255], atol=1e-6)
    assert np.allclose(loaded[0, 0], [1, 0, 0, 0])


def test_png_save_accepts_other_inputs_and_load_other_modes(tmp_path):
    p = tex.save_png(str(tmp_path / "g.png"), np.full((4, 4), 0.5, np.float32))           # grey float, str path
    assert tex.load_png(p)[0, 0].tolist() == pytest.approx([128 / 255] * 3 + [1.0])
    Image.new("L", (3, 2), 100).save(tmp_path / "l.png")
    assert np.allclose(tex.load_png(tmp_path / "l.png")[1, 2], [100 / 255] * 3 + [1.0])
    pal = Image.new("P", (2, 2), 1)
    pal.putpalette([0, 0, 0, 255, 0, 0])
    pal.save(tmp_path / "p.png", transparency=1)
    assert np.allclose(tex.load_png(tmp_path / "p.png")[0, 0], [1, 0, 0, 0])
    assert tex.save_png(tmp_path / "d.png", tex.new(2, 2, WHITE), dither=True).exists()


# ===================================================================================================== blending
def test_composite_normal_and_opacity():
    blue = tex.new(2, 1, BLUE)
    red_half = tex.new(2, 1, "#ff000080")
    out = tex.composite(blue, red_half)
    a = 128 / 255
    assert out.dtype == np.float32 and np.allclose(out[0, 0], [a, 0, 1 - a, 1], atol=1e-6)
    assert np.allclose(tex.composite(blue, tex.new(2, 1, RED), opacity=0.5)[0, 1], [0.5, 0, 0.5, 1], atol=1e-6)
    assert np.allclose(tex.composite(blue, tex.new(2, 1, RED), opacity=0.0), blue)
    mask = np.array([[1.0, 0.0]], np.float32)                                            # per-pixel opacity
    m = tex.composite(blue, tex.new(2, 1, RED), opacity=mask)
    assert np.allclose(m[0, 0], [1, 0, 0, 1]) and np.allclose(m[0, 1], [0, 0, 1, 1])
    transparent = tex.new(2, 1)
    onto = tex.composite(transparent, red_half)                                          # onto nothing = the source
    assert np.allclose(onto[0, 0], [1, 0, 0, a], atol=1e-6)
    both = tex.composite(tex.new(1, 1, "#33445500"), tex.new(1, 1, "#00000000"))
    assert both[0, 0, 3] == 0 and np.allclose(both[0, 0, :3], tex.color("#334455")[:3])  # RGB kept where nothing is
    over = tex.composite(tex.new(1, 1, "#ff000080"), tex.new(1, 1, "#0000ff80"))         # alpha = a + b (1 - a)
    assert np.isclose(over[0, 0, 3], a + a * (1 - a), atol=1e-6)
    assert over[0, 0, 2] > over[0, 0, 0]                                                 # the top layer weighs more


def test_composite_blend_modes():
    g = tex.new(1, 1, "#808080")
    o = tex.new(1, 1, "#ff8000")
    gv, ov = 128 / 255, np.array([1.0, 128 / 255, 0.0])
    assert np.allclose(tex.composite(g, o, "multiply")[0, 0, :3], gv * ov, atol=1e-6)
    assert np.allclose(tex.composite(g, o, "screen")[0, 0, :3], 1 - (1 - gv) * (1 - ov), atol=1e-6)
    assert np.allclose(tex.composite(g, tex.new(1, 1, WHITE), "add")[0, 0, :3], 1.0)             # clipped
    assert np.allclose(tex.composite(g, tex.new(1, 1, "#202020"), "add")[0, 0, :3], gv + 32 / 255, atol=1e-6)
    dark, light = tex.new(1, 1, "#404040"), tex.new(1, 1, "#c0c0c0")
    s = tex.new(1, 1, "#808080")
    assert np.allclose(tex.composite(dark, s, "overlay")[0, 0, :3], 2 * (64 / 255) * (128 / 255), atol=1e-6)
    assert np.allclose(tex.composite(light, s, "overlay")[0, 0, :3], 1 - 2 * (1 - 192 / 255) * (1 - 128 / 255), atol=1e-6)
    # multiply with white changes nothing, screen with black changes nothing
    assert np.allclose(tex.composite(o, tex.new(1, 1, WHITE), "multiply"), o)
    assert np.allclose(tex.composite(o, tex.new(1, 1, BLACK), "screen"), o)
    # half-opacity multiply is the lerp between backdrop and product
    half = tex.composite(g, o, "multiply", opacity=0.5)[0, 0, :3]
    assert np.allclose(half, 0.5 * gv + 0.5 * gv * ov, atol=1e-6)
    # erase cuts alpha away
    er = tex.composite(tex.new(1, 1, RED), tex.new(1, 1, "#ffffff80"), "erase")
    assert np.allclose(er[0, 0], [1, 0, 0, 1 - 128 / 255], atol=1e-6)
    # blend modes onto a transparent backdrop show the source
    for mode in ("multiply", "screen", "add", "overlay"):
        assert np.allclose(tex.composite(tex.new(1, 1), o, mode)[0, 0], o[0, 0], atol=1e-6)


def test_composite_validation_and_purity():
    a, b = tex.new(2, 2, RED), tex.new(2, 2, BLUE)
    a0, b0 = a.copy(), b.copy()
    out = tex.composite(a, b, "multiply", 0.5)
    assert np.array_equal(a, a0) and np.array_equal(b, b0) and out is not a
    with pytest.raises(ValueError):
        tex.composite(a, tex.new(3, 2, BLUE))
    with pytest.raises(ValueError):
        tex.composite(a, b, "burn")
    # uint8 inputs are accepted
    assert tex.composite(tex.to_uint8(a), tex.to_uint8(b)).shape == (2, 2, 4)


def test_paste_uv_orientation_and_purity():
    dst = tex.new(64, 64, BLACK)
    red = tex.new(8, 8, RED)
    out = tex.paste(dst, red, (0.0, 0.5, 0.5, 1.0))                                      # the upper-left quadrant
    assert np.allclose(out[0, 0], [1, 0, 0, 1]) and np.allclose(out[31, 31], [1, 0, 0, 1])
    assert np.allclose(out[32, 0], [0, 0, 0, 1]) and np.allclose(out[0, 32], [0, 0, 0, 1])
    assert dst.sum() == 64 * 64 and out is not dst                                       # dst untouched
    low = tex.paste(dst, red, (0.5, 0.0, 1.0, 0.5))                                      # lower right = bottom rows
    assert np.allclose(low[63, 63], [1, 0, 0, 1]) and np.allclose(low[0, 63], [0, 0, 0, 1])
    # corners in any order give the same result
    assert np.array_equal(tex.paste(dst, red, (0.5, 1.0, 0.0, 0.5)), out)


def test_paste_resampling_modes_and_cropping():
    dst = tex.new(40, 40, WHITE)
    src = np.zeros((4, 4, 4), np.float32)
    src[:, :2] = (1, 0, 0, 1)
    src[:, 2:] = (0, 0, 1, 1)
    out = tex.paste(dst, src, (0.0, 0.0, 1.0, 1.0))
    assert np.allclose(out[20, 2], [1, 0, 0, 1], atol=0.06) and np.allclose(out[20, 37], [0, 0, 1, 1], atol=0.06)   # (Lanczos rings)
    # same-size paste is an exact copy
    exact = tex.paste(tex.new(8, 8), tex.to_float(np.random.default_rng(0).integers(0, 255, (8, 8, 4), np.uint8)), (0, 0, 1, 1))
    assert exact.shape == (8, 8, 4)
    # multiply mode
    m = tex.paste(tex.new(16, 16, "#808080"), tex.new(4, 4, "#808080"), (0, 0, 1, 1), mode="multiply")
    assert np.allclose(m[5, 5, :3], (128 / 255) ** 2, atol=1e-5)
    # a rect hanging out of the destination is cropped, not an error
    crop = tex.paste(tex.new(64, 64, BLACK), tex.new(8, 8, RED), (0.9, 0.9, 1.5, 1.5))
    assert np.allclose(crop[0, 63], [1, 0, 0, 1]) and np.allclose(crop[0, 50], [0, 0, 0, 1])
    assert np.allclose(crop[10, 63], [0, 0, 0, 1])
    far = tex.paste(tex.new(8, 8, BLACK), tex.new(2, 2, RED), (2.0, 2.0, 3.0, 3.0))
    assert far[..., 0].sum() == 0
    empty = tex.paste(tex.new(8, 8, BLACK), tex.new(2, 2, RED), (0.5, 0.5, 0.5, 0.9))   # zero width
    assert empty[..., 0].sum() == 0


def test_resample_is_alpha_correct():
    s = np.zeros((16, 16, 4), np.float32)                                                # left half white, right nothing
    s[:, :8] = (1, 1, 1, 1)
    out = tex.paste(tex.new(24, 24), s, (0, 0, 1, 1))
    visible = out[..., 3] > 0.01
    assert out[..., :3][visible].min() > 0.999                                           # no dark fringe at the edge
    assert np.any((out[..., 3] > 0.05) & (out[..., 3] < 0.95))                           # but a soft alpha edge
    r = tex.resize(s, 7, 5)
    assert r.shape == (5, 7, 4) and r[..., :3][r[..., 3] > 0.01].min() > 0.999
    c = tex.resize(tex.new(10, 10, "#336699"), 4, 3)
    assert np.allclose(c, tex.color("#336699"), atol=1e-5)
    same = tex.resize(s, 16, 16)
    assert np.array_equal(same, s) and same is not s


# ============================================================================================ blur and bleed
def gauss_blur_reference(img, sigma, wrap=False):
    """Slow independent reference: np.convolve per row / column with the documented kernel (radius ceil(3.5 sigma))."""
    r = max(1, int(np.ceil(3.5 * sigma)))
    x = np.arange(-r, r + 1)
    k = np.exp(-0.5 * (x / sigma) ** 2)
    k /= k.sum()
    out = img.astype(np.float64)
    for axis in (0, 1):
        pad = [(0, 0)] * out.ndim
        pad[axis] = (r, r)
        p = np.pad(out, pad, mode="wrap" if wrap else "edge")
        out = np.apply_along_axis(lambda v: np.convolve(v, k, mode="valid"), axis, p)
    return out


def test_blur_matches_reference_small_and_fft_kernels():
    rng = np.random.default_rng(3)
    a = rng.random((30, 41)).astype(np.float32)
    for sigma in (1.0, 2.5, 6.0):                                                        # 6.0 uses the FFT path
        assert np.abs(tex.blur(a, sigma) - gauss_blur_reference(a, sigma)).max() < 2e-6
    assert np.abs(tex.blur(a, 6.0, wrap=True) - gauss_blur_reference(a, 6.0, wrap=True)).max() < 2e-6
    assert tex.blur(a, 2.0).dtype == np.float32 and tex.blur(a, 2.0).shape == a.shape
    assert np.array_equal(tex.blur(a, 0.0), a) and tex.blur(a, 0.0) is not a


def test_blur_conservation_symmetry_and_wrap():
    const = np.full((20, 30), 0.7, np.float32)
    assert np.abs(tex.blur(const, 3.0) - 0.7).max() < 1e-6 and np.abs(tex.blur(const, 14.0) - 0.7).max() < 1e-6
    imp = np.zeros((41, 41), np.float32)
    imp[20, 20] = 1.0
    b = tex.blur(imp, 2.0)
    assert np.isclose(b.sum(), 1.0, atol=1e-5) and np.allclose(b, b[::-1, ::-1], atol=1e-7)
    assert np.allclose(b, b.T, atol=1e-7) and b.argmax() == 20 * 41 + 20
    std = np.sqrt((b.sum(axis=0) * (np.arange(41) - 20) ** 2).sum())                     # sigma comes out right
    assert abs(std - 2.0) < 0.05
    edge = np.zeros((10, 20), np.float32)
    edge[5, 0] = 1.0
    assert tex.blur(edge, 2.0, wrap=False)[5, 19] < 1e-6 and tex.blur(edge, 2.0, wrap=True)[5, 19] > 0.01


def test_blur_is_alpha_weighted():
    img = tex.new(64, 64, "#00000000")                                                   # black transparent surround
    img[20:40, 20:40] = (1, 0, 0, 1)
    b = tex.blur(img, 3.0)
    seen = b[..., 3] > 0.01
    assert np.allclose(b[..., :3][seen], [1, 0, 0], atol=1e-4)                           # colour not darkened
    assert np.isclose(b[..., 3].sum(), img[..., 3].sum(), rtol=1e-3)                    # coverage is conserved
    assert 0.4 < b[20, 30, 3] < 0.6                                                      # half alpha on the edge
    far = tex.blur(tex.new(8, 8, "#33445500"), 2.0)                                      # transparent: RGB left alone
    assert np.allclose(far[0, 0, :3], tex.color("#334455")[:3]) and far[..., 3].max() == 0
    rgb = tex.blur(np.zeros((9, 9, 3), np.float32), 1.0)                                 # non-RGBA: per channel
    assert rgb.shape == (9, 9, 3)


def test_bleed_fills_transparent_rgb_only():
    img = tex.new(16, 16, "#00000000")
    img[6:10, 6:10] = (1.0, 0.5, 0.2, 1.0)
    out = tex.bleed(img, 3)
    assert np.array_equal(out[..., 3], img[..., 3])                                      # alpha untouched
    assert np.array_equal(out[6:10, 6:10], img[6:10, 6:10])                              # opaque pixels untouched
    assert np.allclose(out[5, 5, :3], [1, 0.5, 0.2]) and np.allclose(out[10, 12, :3], [1, 0.5, 0.2])  # 3 px reach
    assert np.allclose(out[3, 7, :3], [1, 0.5, 0.2]) and np.allclose(out[2, 7, :3], 0.0)           # and no more
    assert np.allclose(out[0, 0, :3], 0.0)
    assert np.array_equal(img[5, 5], [0, 0, 0, 0])                                       # input not modified
    one = tex.bleed(img, 1)
    assert np.allclose(one[5, 7, :3], [1, 0.5, 0.2]) and np.allclose(one[4, 7, :3], 0.0)


def test_bleed_averages_neighbours_and_edge_cases():
    img = tex.new(5, 1, "#00000000")
    img[0, 0] = (1, 0, 0, 1)
    img[0, 4] = (0, 0, 1, 1)
    out = tex.bleed(img, 1)
    assert np.allclose(out[0, 1, :3], [1, 0, 0]) and np.allclose(out[0, 3, :3], [0, 0, 1])
    assert np.allclose(out[0, 2, :3], 0.0)                                               # not reached after one round
    two = tex.bleed(img, 2)
    assert np.allclose(two[0, 2, :3], [0.5, 0, 0.5])                                     # mean of both sides
    semi = tex.new(3, 1, "#00000000")
    semi[0, 1] = (0, 1, 0, 0.5)
    assert np.allclose(tex.bleed(semi, 1)[0, 0, :3], [0, 1, 0])                           # partly transparent counts
    assert np.array_equal(tex.bleed(img, 0), img)
    empty, full = tex.new(4, 4), tex.new(4, 4, RED)
    assert np.array_equal(tex.bleed(empty), empty) and np.array_equal(tex.bleed(full), full)


# ======================================================================================================= noise
def test_value_noise_basics():
    n = tex.value_noise(96, 64, scale=6.0, seed=1, octaves=3)
    assert n.dtype == np.float32 and n.shape == (64, 96)
    assert 0.0 <= n.min() and n.max() <= 1.0 and n.std() > 0.05 and 0.3 < n.mean() < 0.7
    assert np.array_equal(n, tex.value_noise(96, 64, scale=6.0, seed=1, octaves=3))      # deterministic
    assert not np.array_equal(n, tex.value_noise(96, 64, scale=6.0, seed=2, octaves=3))
    assert not np.array_equal(tex.value_noise(32, 32, seed=1), tex.value_noise(32, 32, seed=-1))
    st = tex.value_noise(64, 64, scale=4.0, seed=3, normalize=True)
    assert st.min() == 0.0 and st.max() == 1.0
    one, many = tex.value_noise(64, 64, 8, seed=4, octaves=1), tex.value_noise(64, 64, 8, seed=4, octaves=4)
    assert np.abs(np.diff(many, axis=1)).mean() > np.abs(np.diff(one, axis=1)).mean()    # finer detail
    coarse, fine = tex.value_noise(64, 64, 2.0, seed=4), tex.value_noise(64, 64, 16.0, seed=4)
    assert np.abs(np.diff(fine, axis=1)).mean() > 3 * np.abs(np.diff(coarse, axis=1)).mean()
    smooth = np.abs(np.diff(n, axis=1)).max()
    assert smooth < 0.2                                                                  # no white-noise jumps


def test_value_noise_independent_of_resolution():
    a = tex.value_noise(64, 64, 4.0, seed=3)
    b = tex.value_noise(256, 256, 4.0, seed=3)
    assert np.corrcoef(a.ravel(), b[2::4, 2::4].ravel())[0, 1] > 0.999                  # same picture, finer sampling
    c = tex.value_noise(100, 50, (6.0, 3.0), seed=3)
    assert c.shape == (50, 100)


def test_value_noise_tileable_wraps_seamlessly():
    t = tex.value_noise(64, 48, 4.0, seed=5, octaves=3, tileable=True)
    side = np.concatenate([t, t], axis=1)
    top = np.concatenate([t, t], axis=0)
    seam_u = np.abs(side[:, 63] - side[:, 64]).mean()
    seam_v = np.abs(top[47] - top[48]).mean()
    inner_u = np.abs(np.diff(t, axis=1)).mean()
    inner_v = np.abs(np.diff(t, axis=0)).mean()
    assert seam_u < 1.5 * inner_u and seam_v < 1.5 * inner_v
    plain = tex.value_noise(64, 48, 4.0, seed=5, octaves=3, tileable=False)
    assert np.abs(plain[:, 0] - plain[:, -1]).mean() > 5 * seam_u                        # the plain one does not wrap
    assert np.abs(t[:, 0] - t[:, -1]).mean() < 3 * inner_u


def test_streaks_are_anisotropic():
    s = tex.streaks(128, 128)
    assert s.dtype == np.float32 and s.shape == (128, 128) and 0.0 <= s.min() and s.max() <= 1.0
    along_u = np.abs(np.diff(s, axis=1)).mean()                                          # slow along u ...
    along_v = np.abs(np.diff(s, axis=0)).mean()                                          # ... fast along v
    assert along_v > 8 * along_u
    flipped = tex.streaks(128, 128, scale=(64.0, 2.0))
    assert np.abs(np.diff(flipped, axis=1)).mean() > 8 * np.abs(np.diff(flipped, axis=0)).mean()
    assert np.array_equal(s, tex.streaks(128, 128)) and not np.array_equal(s, tex.streaks(128, 128, seed=1))


# ==================================================================================================== gradients
def test_linear_gradient_default_top_to_bottom():
    g = tex.linear_gradient(8, 100, [(0, RED), (1, BLUE)])
    assert g.shape == (100, 8, 4) and g.dtype == np.float32
    assert np.allclose(g[0, 0], tex.color(RED), atol=0.006) and np.allclose(g[-1, 0], tex.color(BLUE), atol=0.006)
    assert np.allclose(g[49:51, 0].mean(axis=0), [0.5, 0, 0.5, 1], atol=1e-6)            # midpoint of the sRGB numbers
    assert np.all(np.diff(g[:, 0, 0]) < 0) and np.all(np.diff(g[:, 0, 2]) > 0)           # monotonic, red falls
    assert np.allclose(g[:, 0], g[:, 7])                                                 # constant along the width
    assert g[0, 0, 0] > g[1, 0, 0]


def test_linear_gradient_direction_stops_and_space():
    h = tex.linear_gradient(100, 4, [(0, BLACK), (1, WHITE)], p0=(0, 0.5), p1=(1, 0.5))
    assert h[0, 0, 0] < 0.01 and h[0, -1, 0] > 0.99 and np.allclose(h[0, 49:51, 0].mean(), 0.5, atol=1e-6)
    d = tex.linear_gradient(50, 50, [(0, BLACK), (1, WHITE)], p0=(0, 0), p1=(1, 1))       # bottom-left -> top-right
    assert d[-1, 0, 0] < 0.02 and d[0, -1, 0] > 0.98 and np.isclose(d[25, 25, 0], 0.5, atol=0.02)
    assert np.isclose(d[0, 0, 0], d[-1, -1, 0], atol=0.02)                               # the other diagonal is flat
    uns = tex.linear_gradient(4, 40, [(1, BLUE), (0, RED), (0.5, GREEN)])               # unsorted stops
    assert np.allclose(uns[19:21, 0].mean(axis=0), tex.color(GREEN), atol=0.03)
    out = tex.linear_gradient(4, 40, [(0.25, RED), (0.75, BLUE)])                        # clamps outside the stops
    assert np.allclose(out[0, 0], tex.color(RED)) and np.allclose(out[-1, 0], tex.color(BLUE))
    one = tex.linear_gradient(3, 3, [(0.5, "#336699")])
    assert np.allclose(one, tex.color("#336699"))
    # p0 / p1 inside the image: clamped colour beyond
    part = tex.linear_gradient(4, 100, [(0, RED), (1, BLUE)], p0=(0.5, 0.8), p1=(0.5, 0.2))
    assert np.allclose(part[0, 0], tex.color(RED)) and np.allclose(part[-1, 0], tex.color(BLUE))
    with pytest.raises(ValueError):
        tex.linear_gradient(4, 4, [(0, RED), (1, BLUE)], p0=(0.5, 0.5), p1=(0.5, 0.5))
    with pytest.raises(ValueError):
        tex.linear_gradient(4, 4, [])


def test_gradient_interpolation_space_and_alpha():
    s = tex.linear_gradient(2, 101, [(0, BLACK), (1, WHITE)], linear=False)
    lin = tex.linear_gradient(2, 101, [(0, BLACK), (1, WHITE)], linear=True)
    assert np.isclose(s[50, 0, 0], 0.5, atol=1e-6) and np.isclose(lin[50, 0, 0], 0.73536, atol=1e-3)
    fade = tex.linear_gradient(2, 51, [(0, "#ff0000"), (1, "#ff000000")])
    assert fade[0, 0, 3] > 0.98 and fade[-1, 0, 3] < 0.02 and np.allclose(fade[10, 0, :3], [1, 0, 0], atol=1e-5)
    # fading to transparent BLACK still fades red (alpha weighted), it does not go through dark red
    fade2 = tex.linear_gradient(2, 51, [(0, "#ff0000"), (1, "#00000000")])
    assert np.allclose(fade2[25, 0, :3], [1, 0, 0], atol=1e-5) and np.isclose(fade2[25, 0, 3], 0.5, atol=0.02)


def test_radial_gradient():
    r = tex.radial_gradient(101, 101, [(0, WHITE), (1, BLACK)])
    assert np.allclose(r[50, 50], [1, 1, 1, 1], atol=1e-6)
    assert r[0, 0, 0] == 0.0 and r[0, 50, 0] < 0.02                                      # clamped outside the radius
    assert np.isclose(r[50, 75, 0], 0.5, atol=0.02) and np.isclose(r[25, 50, 0], 0.5, atol=0.02)
    assert np.allclose(r[50, 75], r[25, 50], atol=1e-3) and np.allclose(r[75, 50], r[25, 50], atol=1e-3)
    wide = tex.radial_gradient(200, 100, [(0, WHITE), (1, BLACK)], radius=0.25)           # circular in pixels
    assert np.isclose(wide[50, 100 + 25, 0], 0.5, atol=0.03) and np.isclose(wide[50 - 25, 100, 0], 0.5, atol=0.03)
    ell = tex.radial_gradient(100, 100, [(0, WHITE), (1, BLACK)], radius=0.4, aspect=2.0)  # twice as wide as tall
    assert np.isclose(ell[50, 50 + 20, 0], 0.5, atol=0.03) and np.isclose(ell[50 - 10, 50, 0], 0.5, atol=0.05)
    off = tex.radial_gradient(100, 100, [(0, WHITE), (1, BLACK)], center=(0.25, 0.75), radius=0.2)
    assert off[25, 25, 0] > 0.95                                                         # v = 0.75 is row 25
    with pytest.raises(ValueError):
        tex.radial_gradient(4, 4, [(0, WHITE)], radius=0.0)


# ======================================================================================================= canvas
def test_canvas_disc_area_and_antialiasing():
    cv = tex.Canvas(256, 256)
    assert cv.circle((0.5, 0.5), 0.3, fill=RED) is cv                                    # chaining
    img = cv.image()
    assert img.shape == (256, 256, 4) and img.dtype == np.float32
    expected = np.pi * (0.3 * 256) ** 2
    assert abs(area(img) - expected) / expected < 0.01
    a = img[..., 3]
    edge = (a > 0) & (a < 1)
    assert edge.sum() > 200                                                              # antialiased: partial alpha
    assert np.allclose(img[a > 0][:, :3], [1, 0, 0], atol=1e-5)                         # colour is not darkened
    assert a[128, 128] == 1.0 and a[0, 0] == 0.0
    ys, xs = np.nonzero(edge)
    r = np.hypot(xs + 0.5 - 128, ys + 0.5 - 128)
    assert abs(r.mean() - 0.3 * 256) < 1.0                                               # the edge sits at the radius


def test_canvas_without_antialiasing_is_binary():
    img = tex.Canvas(64, 64, ss=1).circle((0.5, 0.5), 0.3, fill=RED).image()
    assert set(np.unique(img[..., 3]).tolist()) <= {0.0, 1.0}
    smooth = tex.Canvas(64, 64, ss=8).circle((0.5, 0.5), 0.3, fill=RED).image()[..., 3]
    coarse = tex.Canvas(64, 64, ss=2).circle((0.5, 0.5), 0.3, fill=RED).image()[..., 3]
    assert len(np.unique(smooth)) > len(np.unique(coarse)) > 2                           # more samples, more coverage levels
    assert abs(area(smooth[..., None].repeat(4, axis=2)) - np.pi * (0.3 * 64) ** 2) / (np.pi * (0.3 * 64) ** 2) < 0.01


def test_canvas_polygon_areas():
    # a rectangle on the sample grid has an exact area
    cv = tex.Canvas(100, 50).polygon([(0.25, 0.2), (0.75, 0.2), (0.75, 0.6), (0.25, 0.6)], fill=WHITE)
    assert area(cv.image()) == pytest.approx(0.5 * 0.4 * 100 * 50, abs=1e-3)
    # a triangle: area = base * height / 2
    cv = tex.Canvas(200, 100).polygon([(0.1, 0.1), (0.9, 0.1), (0.5, 0.9)], fill=WHITE)
    expected = 0.8 * 0.8 / 2 * 200 * 100
    assert area(cv.image()) == pytest.approx(expected, rel=0.004)
    # winding does not matter, concave shapes work: an L shape
    pts = [(0.1, 0.1), (0.9, 0.1), (0.9, 0.3), (0.3, 0.3), (0.3, 0.9), (0.1, 0.9)]
    exp_l = (0.8 * 0.2 + 0.2 * 0.6) * 80 * 80
    for p in (pts, pts[::-1]):
        assert area(tex.Canvas(80, 80).polygon(p, fill=WHITE).image()) == pytest.approx(exp_l, rel=0.003)
    # a ring list is a polygon with a hole (even-odd)
    outer = [(0.1, 0.1), (0.9, 0.1), (0.9, 0.9), (0.1, 0.9)]
    inner = [(0.3, 0.3), (0.7, 0.3), (0.7, 0.7), (0.3, 0.7)]
    holed = tex.Canvas(100, 100).polygon([outer, inner], fill=WHITE).image()
    assert area(holed) == pytest.approx((0.64 - 0.16) * 10000, rel=0.002) and holed[50, 50, 3] == 0
    solid = tex.Canvas(100, 100).polygon([outer, inner], fill=WHITE, rule="nonzero").image()
    assert area(solid) == pytest.approx(0.64 * 10000, rel=0.002)
    # a self-intersecting bow tie
    bow = tex.Canvas(100, 100).polygon([(0.1, 0.1), (0.9, 0.9), (0.9, 0.1), (0.1, 0.9)], fill=WHITE).image()
    assert bow[50, 25, 3] == 1.0 and bow[25, 50, 3] == 0.0 and bow[75, 50, 3] == 0.0


def test_canvas_uv_orientation_v_is_up():
    cv = tex.Canvas(64, 64).rect(0.1, 0.8, 0.9, 0.95, fill=WHITE)                        # near v = 1 -> TOP rows
    img = cv.image()
    rows = np.nonzero(img[..., 3].sum(axis=1) > 0)[0]
    cols = np.nonzero(img[..., 3].sum(axis=0) > 0)[0]
    assert rows.min() == 3 and rows.max() == 12                                          # (1 - 0.95) * 64 = 3.2
    assert cols.min() == 6 and cols.max() == 57                                          # u runs left to right
    low = tex.Canvas(64, 64).circle((0.2, 0.1), 0.05, fill=WHITE).image()
    ys, xs = np.nonzero(low[..., 3])
    assert ys.mean() > 55 and xs.mean() < 20                                             # bottom left
    cv = tex.Canvas(50, 100).polygon([(0, 1), (0.5, 1), (0, 0.5)], fill=WHITE).image()   # top-left triangle
    assert cv[0, 0, 3] == 1.0 and cv[0, 49, 3] == 0.0 and cv[99, 0, 3] == 0.0 and cv[24, 0, 3] == 1.0


def test_canvas_stroke_widths_are_in_uv_width_units_both_ways():
    cv = tex.Canvas(200, 100).polyline([(0.1, 0.5), (0.9, 0.5)], 0.04, WHITE, cap="butt")
    img = cv.image()
    assert area(img) == pytest.approx(0.8 * 200 * 0.04 * 200, abs=1e-3)                  # 160 x 8 px
    assert np.count_nonzero(img[:, 100, 3]) == 8
    v = tex.Canvas(200, 100).polyline([(0.5, 0.1), (0.5, 0.9)], 0.04, WHITE, cap="butt").image()
    assert area(v) == pytest.approx(0.8 * 100 * 0.04 * 200, abs=1e-3)                    # 80 x 8 px: 8 px both ways
    assert np.count_nonzero(v[50, :, 3]) == 8
    wide = tex.Canvas(200, 100).polyline([(0.1, 0.5), (0.9, 0.5)], 0.1, WHITE, cap="butt").image()
    assert area(wide) == pytest.approx(160 * 20, abs=1e-3)
    diag = tex.Canvas(200, 200).polyline([(0.1, 0.1), (0.9, 0.9)], 0.04, WHITE, cap="butt").image()
    length = 0.8 * np.sqrt(2) * 200
    assert area(diag) == pytest.approx(length * 8, rel=0.01)


def test_canvas_caps():
    base = 0.6 * 200 * 8                                                                 # length x width in px^2
    seg = [(0.2, 0.5), (0.8, 0.5)]
    butt = area(tex.Canvas(200, 100).polyline(seg, 0.04, WHITE, cap="butt").image())
    rnd = area(tex.Canvas(200, 100).polyline(seg, 0.04, WHITE, cap="round").image())
    sq = area(tex.Canvas(200, 100).polyline(seg, 0.04, WHITE, cap="square").image())
    assert butt == pytest.approx(base, abs=1e-3)
    assert rnd == pytest.approx(base + np.pi * 4 ** 2, rel=0.003)                        # two half discs
    assert sq == pytest.approx(base + 8 * 8, abs=1e-3)                                   # two half squares
    dot = tex.Canvas(100, 100).polyline([(0.5, 0.5)], 0.2, WHITE).image()                # one point: a round dot
    assert area(dot) == pytest.approx(np.pi * 10 ** 2, rel=0.01)
    assert area(tex.Canvas(100, 100).polyline([(0.5, 0.5)], 0.2, WHITE, cap="butt").image()) == 0.0
    with pytest.raises(ValueError):
        tex.Canvas(8, 8).polyline(seg, 0.1, WHITE, cap="triangle")


def test_canvas_joins_at_a_right_angle():
    pts = [(0.2, 0.2), (0.8, 0.2), (0.8, 0.8)]
    r = 0.1 * 200 / 2                                                                    # half width in px
    areas = {}
    for join in ("bevel", "round", "miter"):
        areas[join] = area(tex.Canvas(200, 200).polyline(pts, 0.1, WHITE, cap="butt", join=join).image())
    assert areas["bevel"] == pytest.approx(2 * 120 * 2 * r - r * r + r * r / 2, rel=0.002)   # 2 bars - overlap + triangle
    assert areas["miter"] - areas["bevel"] == pytest.approx(r * r / 2, rel=0.05)         # square vs triangle corner
    assert areas["round"] - areas["bevel"] == pytest.approx((np.pi / 4 - 0.5) * r * r, rel=0.08)
    assert areas["bevel"] < areas["round"] < areas["miter"]
    # the outer corner pixel of a miter join is filled, of the bevel is not
    m = tex.Canvas(200, 200).polyline(pts, 0.1, WHITE, cap="butt", join="miter").image()
    b = tex.Canvas(200, 200).polyline(pts, 0.1, WHITE, cap="butt", join="bevel").image()
    assert m[169, 169, 3] == 1.0 and b[169, 169, 3] == 0.0
    # a very sharp turn: miter falls back to bevel (no spike)
    spike = tex.Canvas(100, 100).polyline([(0.1, 0.5), (0.9, 0.5), (0.1, 0.52)], 0.02, WHITE, cap="butt", join="miter").image()
    assert spike[..., 3].sum(axis=0).nonzero()[0].max() < 94
    with pytest.raises(ValueError):
        tex.Canvas(8, 8).polyline(pts, 0.1, WHITE, join="spiky")


def test_canvas_closed_polyline_and_polygon_stroke():
    sq = [(0.2, 0.2), (0.8, 0.2), (0.8, 0.8), (0.2, 0.8)]
    ring = tex.Canvas(100, 100).polyline(sq, 0.04, WHITE, closed=True, cap="butt", join="miter").image()
    outer, inner = (0.6 + 0.04) ** 2, (0.6 - 0.04) ** 2
    assert area(ring) == pytest.approx((outer - inner) * 10000, rel=0.003)
    assert ring[50, 50, 3] == 0.0 and ring[20, 50, 3] == 1.0
    poly = tex.Canvas(100, 100).polygon(sq, stroke=WHITE, width=0.04).image()           # same via polygon()
    assert area(poly) == pytest.approx((outer - inner) * 10000, rel=0.003)
    both = tex.Canvas(100, 100).polygon(sq, fill=RED, stroke=BLUE, width=0.04).image()
    assert np.allclose(both[50, 50], [1, 0, 0, 1]) and np.allclose(both[20, 50], [0, 0, 1, 1])


def test_canvas_ellipse_circle_and_rect():
    e = tex.Canvas(200, 200).ellipse((0.5, 0.5), (0.3, 0.1), fill=WHITE).image()
    assert area(e) == pytest.approx(np.pi * 60 * 20, rel=0.005)
    rot = tex.Canvas(200, 200).ellipse((0.5, 0.5), (0.3, 0.1), fill=WHITE, angle=90).image()
    assert area(rot) == pytest.approx(np.pi * 60 * 20, rel=0.005)
    ys, xs = np.nonzero(rot[..., 3] > 0.5)
    assert (ys.max() - ys.min()) > 2.5 * (xs.max() - xs.min())                          # now taller than wide
    tilt = tex.Canvas(200, 200).ellipse((0.5, 0.5), (0.3, 0.05), fill=WHITE, angle=45).image()
    assert tilt[60, 140, 3] > 0.5 and tilt[140, 140, 3] == 0.0                          # CCW: up-right (row 60, col 140)
    ring = tex.Canvas(200, 200).circle((0.5, 0.5), 0.3, stroke=WHITE, width=0.02).image()
    assert area(ring) == pytest.approx(2 * np.pi * 60 * 4, rel=0.01) and ring[100, 100, 3] == 0.0
    elring = tex.Canvas(200, 200).ellipse((0.5, 0.5), (0.35, 0.1), stroke=WHITE, width=0.02).image()
    assert area(elring) == pytest.approx(  # uniform stroke: perimeter x width (Ramanujan)
        np.pi * (3 * (70 + 20) - np.sqrt((3 * 70 + 20) * (70 + 3 * 20))) * 4, rel=0.015)
    both = tex.Canvas(100, 100).circle((0.5, 0.5), 0.3, fill=RED, stroke=BLUE, width=0.04).image()
    assert np.allclose(both[50, 50], [1, 0, 0, 1]) and np.allclose(both[50, 20], [0, 0, 1, 1], atol=0.02)
    r = tex.Canvas(100, 100).rect(0.1, 0.2, 0.9, 0.7, fill=WHITE).image()
    assert area(r) == pytest.approx(0.8 * 0.5 * 10000, abs=1e-3)
    rr = tex.Canvas(100, 100).rect(0.1, 0.2, 0.9, 0.7, fill=WHITE, radius=0.1).image()
    assert area(rr) == pytest.approx(0.8 * 0.5 * 10000 - (4 - np.pi) * 100, rel=0.003)
    assert rr[30, 10, 3] == 0.0 and rr[50, 10, 3] == 1.0                                 # the corner is rounded off
    huge = tex.Canvas(100, 100).rect(0.1, 0.2, 0.9, 0.7, fill=WHITE, radius=5.0).image()  # clipped to half the side
    assert area(huge) == pytest.approx(np.pi * 25 * 25 + 0.3 * 100 * 50, rel=0.02)
    sr = tex.Canvas(100, 100).rect(0.2, 0.2, 0.8, 0.8, stroke=WHITE, width=0.04).image()
    assert area(sr) == pytest.approx(((0.64) ** 2 - (0.56) ** 2) * 10000, rel=0.003)
    flip = tex.Canvas(100, 100).rect(0.9, 0.7, 0.1, 0.2, fill=WHITE).image()             # corners in any order
    assert np.array_equal(flip, r)


def test_canvas_blend_modes():
    def paint(mode, bg, c, **kw):
        return tex.Canvas(8, 8, bg=bg).rect(0, 0, 1, 1, fill=c, mode=mode, **kw).image()[4, 4]

    a = 128 / 255
    assert np.allclose(paint("normal", "#808080", "#ff000080"), [a + a * (1 - a), a * (1 - a), a * (1 - a), 1], atol=1e-4)
    assert np.allclose(paint("multiply", "#808080", "#ff8000")[:3], [0.50196, 0.25197, 0.0], atol=1e-4)
    assert np.allclose(paint("screen", "#808080", "#ff8000")[:3], [1.0, 0.75197, 0.50196], atol=1e-4)
    assert np.allclose(paint("add", "#808080", "#404040")[:3], 128 / 255 + 64 / 255, atol=1e-4)
    assert np.allclose(paint("add", "#808080", "#ffffff")[:3], 1.0, atol=1e-4)
    assert np.allclose(paint("overlay", "#404040", "#808080")[:3], 2 * (64 / 255) * (128 / 255), atol=1e-4)
    assert paint("erase", RED, WHITE)[3] == 0.0
    half = paint("erase", RED, "#ffffff80")
    assert np.isclose(half[3], 1 - 128 / 255, atol=1e-4) and np.allclose(half[:3], [1, 0, 0], atol=1e-4)
    # on a transparent canvas every mode just shows the colour
    for mode in ("multiply", "screen", "add", "overlay"):
        got = tex.Canvas(8, 8).rect(0, 0, 1, 1, fill="#ff8000", mode=mode).image()[4, 4]
        assert np.allclose(got, [1, 128 / 255, 0, 1], atol=1e-4)
    # erase a hole into an opaque fill, with antialiased edges
    img = tex.Canvas(64, 64, bg=RED).circle((0.5, 0.5), 0.25, fill=WHITE, mode="erase").image()
    assert img[32, 32, 3] == 0.0 and img[0, 0, 3] == 1.0 and np.any((img[..., 3] > 0) & (img[..., 3] < 1))
    with pytest.raises(ValueError):
        tex.Canvas(8, 8).circle((0.5, 0.5), 0.2, fill=RED, mode="dodge")
    # a mode applies to the stroke as well
    cv = tex.Canvas(32, 32, bg="#808080").circle((0.5, 0.5), 0.3, stroke="#808080", width=0.05, mode="multiply")
    assert np.allclose(cv.image()[16, 16 - 10, :3], (128 / 255) ** 2, atol=1e-3)


def test_canvas_translucent_fills_accumulate():
    cv = tex.Canvas(8, 8, bg=WHITE).rect(0, 0, 1, 1, fill="#ff000080").rect(0, 0, 1, 1, fill="#ff000080")
    a = 128 / 255
    one_layer = 1 - a
    expected_g = one_layer * (1 - a)                                                     # white seen through two layers
    assert np.allclose(cv.image()[4, 4], [1, expected_g, expected_g, 1], atol=1e-3)
    # premultiplied accumulation: transparent bg, no fringes: all visible pixels are exactly the paint colour
    img = tex.Canvas(64, 64).circle((0.5, 0.5), 0.3, fill="#ff800080").image()
    seen = img[..., 3] > 0
    assert np.allclose(img[seen][:, :3], [1, 128 / 255, 0], atol=1e-5)
    assert np.isclose(img[32, 32, 3], 128 / 255, atol=1e-5)


def test_canvas_background():
    img = tex.Canvas(8, 6, bg="#336699").image()
    assert img.shape == (6, 8, 4) and np.allclose(img, tex.color("#336699"))
    clear = tex.Canvas(8, 8, bg="#ff000000").circle((0.5, 0.5), 0.2, fill=BLUE).image()
    assert clear[0, 0, 3] == 0.0 and np.allclose(clear[0, 0, :3], [1, 0, 0])             # RGB of a transparent bg kept
    assert np.allclose(clear[4, 4], [0, 0, 1, 1])
    assert np.array_equal(tex.Canvas(4, 4).image(), np.zeros((4, 4, 4), np.float32))
    filled = tex.Canvas(8, 8).fill_all("#102030").image()
    assert np.allclose(filled, tex.color("#102030"))
    assert tex.Canvas(8, 8, bg=RED).fill_all(WHITE, mode="erase").image()[..., 3].max() == 0.0


def test_canvas_curve_bezier_and_widths():
    pts = [(0.1, 0.5), (0.35, 0.7), (0.6, 0.3), (0.9, 0.5)]
    img = tex.Canvas(200, 100).curve(pts, 0.03, WHITE).image()
    for u, v in pts:                                                                     # passes through every point
        assert img[int(round((1 - v) * 100 - 0.5)), int(round(u * 200 - 0.5)), 3] > 0.9
    assert img[..., 3].sum() > 0
    closed = tex.Canvas(100, 100).curve([(0.5, 0.2), (0.8, 0.5), (0.5, 0.8), (0.2, 0.5)], 0.03, WHITE, closed=True).image()
    assert closed[50, 50, 3] == 0.0 and closed[80, 50, 3] > 0.9 and closed[50, 80, 3] > 0.9
    tap = tex.Canvas(200, 50).curve([(0.1, 0.5), (0.5, 0.5), (0.9, 0.5)], 0.02, WHITE, widths=[0.002, 0.04, 0.002], cap="butt").image()
    thickness = (tap[..., 3] > 0.5).sum(axis=0)
    assert thickness[100] > 2.5 * thickness[30] > 0 and thickness[100] > 2.5 * thickness[170]
    assert abs(int(thickness[100]) - 8) <= 1                                             # 0.04 * 200 px in the middle
    two = tex.Canvas(100, 100).curve([(0.1, 0.1), (0.9, 0.9)], 0.04, WHITE).image()      # straight line
    assert two[50, 50, 3] == 1.0 and two[50, 90, 3] == 0.0
    with pytest.raises(ValueError):
        tex.Canvas(8, 8).curve(pts, 0.03, WHITE, widths=[0.01, 0.02])
    bz = tex.Canvas(200, 100).bezier((0.1, 0.5), (0.3, 0.9), (0.7, 0.1), (0.9, 0.5), 0.02, WHITE).image()
    assert bz[50, 20, 3] > 0.9 and bz[50, 180, 3] > 0.9 and bz[50, 100, 3] > 0.9        # start, end and the middle
    assert bz[10, 100, 3] == 0.0
    tb = tex.Canvas(200, 100).bezier((0.1, 0.5), (0.4, 0.5), (0.6, 0.5), (0.9, 0.5), (0.001, 0.05), WHITE, cap="butt").image()
    th = (tb[..., 3] > 0.5).sum(axis=0)
    assert th[170] > 3 * th[40] and th[170] >= 7


def test_catmull_rom_and_bezier_points():
    pts = np.array([[0.0, 0.0], [1.0, 0.5], [2.0, 0.0], [3.0, 1.0]])
    c, param = tex.catmull_rom(pts, samples=8, return_param=True)
    assert c.shape == (3 * 8 + 1, 2) and np.allclose(c[0], pts[0]) and np.allclose(c[-1], pts[-1])
    for i in range(4):
        assert np.allclose(c[np.argmin(np.abs(param - i))], pts[i])                      # interpolates the controls
    cl = tex.catmull_rom(pts, samples=8, closed=True)
    assert cl.shape == (4 * 8, 2) and np.allclose(cl[0], pts[0])
    assert tex.catmull_rom(pts[:1]).shape == (1, 2)
    line = tex.catmull_rom(pts[:2], samples=4)
    assert np.allclose(line[:, 1], line[:, 0] * 0.5, atol=1e-9)                           # two points: a straight line
    dup = tex.catmull_rom(np.array([[0, 0], [0, 0], [1, 1], [2, 0]], float), samples=4)   # coincident points are safe
    assert np.all(np.isfinite(dup))
    tight = tex.catmull_rom(np.array([[0, 0], [1, 0], [1.1, 0.01], [2, 0]], float), samples=16)
    assert np.all(np.abs(tight[:, 1]) < 0.1)                                              # centripetal: no loops / overshoot
    b = tex.bezier_points((0, 0), (0, 1), (1, 1), (1, 0), n=10)
    assert b.shape == (11, 2) and np.allclose(b[0], [0, 0]) and np.allclose(b[-1], [1, 0])
    assert np.allclose(b[5], [0.5, 0.75])                                                 # B(0.5) = (P0 + 3P1 + 3P2 + P3) / 8


def test_canvas_gradient_fills():
    cv = tex.Canvas(100, 100)
    cv.fill_gradient([(0.2, 0.2), (0.8, 0.2), (0.8, 0.8), (0.2, 0.8)], [(0, BLACK), (1, WHITE)], (0.2, 0.2), (0.8, 0.2))
    img = cv.image()
    assert img[50, 10, 3] == 0.0 and img[50, 90, 3] == 0.0                               # clipped to the polygon
    assert img[50, 21, 0] < 0.05 and img[50, 78, 0] > 0.95 and np.isclose(img[50, 50, 0], 0.5, atol=0.02)
    assert abs(area(img) - 0.36 * 10000) < 1.0
    cv = tex.Canvas(100, 100).fill_gradient(None, [(0, RED), (1, BLUE)], (0.5, 1.0), (0.5, 0.0))    # whole canvas
    ref = tex.linear_gradient(100, 100, [(0, RED), (1, BLUE)])
    assert np.abs(cv.image() - ref).max() < 0.006                                        # same as the array version
    lin = tex.Canvas(100, 100).fill_gradient(None, [(0, BLACK), (1, WHITE)], (0.5, 1.0), (0.5, 0.0), linear=True).image()
    assert np.isclose(lin[50, 10, 0], 0.7354, atol=0.01)
    rad = tex.Canvas(100, 100).fill_radial([(0, 0), (1, 0), (1, 1), (0, 1)], [(0, WHITE), (1, BLACK)], (0.5, 0.5), 0.5).image()
    ref = tex.radial_gradient(100, 100, [(0, WHITE), (1, BLACK)])
    assert np.abs(rad - ref).max() < 0.015
    clipped = tex.Canvas(100, 100).fill_radial([(0.4, 0.4), (0.6, 0.4), (0.6, 0.6), (0.4, 0.6)], [(0, RED), (1, BLUE)], (0.5, 0.5), 0.1)
    assert np.allclose(clipped.image()[50, 50], tex.color(RED), atol=0.1) and clipped.image()[10, 10, 3] == 0.0
    mul = tex.Canvas(50, 50, bg="#808080").fill_gradient(None, [(0, WHITE), (1, "#808080")], (0.5, 1), (0.5, 0), mode="multiply").image()
    assert mul[0, 0, 0] > mul[-1, 0, 0] and mul[-1, 0, 0] < 0.3
    fade = tex.Canvas(50, 50, bg=WHITE).fill_radial(None, [(0, "#ff0000ff"), (1, "#ff000000")], (0.5, 0.5), 0.4).image()
    assert np.allclose(fade[25, 25], [1, 0.038, 0.038, 1], atol=0.005) and np.allclose(fade[0, 0], [1, 1, 1, 1], atol=1e-5)
    with pytest.raises(ValueError):
        tex.Canvas(8, 8).fill_radial(None, [(0, RED), (1, BLUE)], (0.5, 0.5), 0.0)


def test_canvas_stamp():
    src = np.zeros((8, 8, 4), np.float32)
    src[:4, :4] = (1, 0, 0, 1)                                                           # top-left red
    src[:4, 4:] = (0, 1, 0, 1)                                                           # top-right green
    src[4:, :4] = (0, 0, 1, 1)                                                           # bottom-left blue
    src[4:, 4:] = (1, 1, 1, 0)                                                           # bottom-right: nothing
    cv = tex.Canvas(40, 40).stamp(src, (0.25, 0.25, 0.75, 0.75))                         # rect = pixels 10..30
    src[:] = 0.0                                                                         # the stamp keeps its own copy
    img = cv.image()
    assert img[0, 0, 3] == 0.0 and img[39, 39, 3] == 0.0 and img[9, 15, 3] == 0.0       # only inside the rect
    assert np.allclose(img[11, 11], [1, 0, 0, 1], atol=1e-3) and np.allclose(img[11, 28], [0, 1, 0, 1], atol=1e-3)
    assert np.allclose(img[28, 11], [0, 0, 1, 1], atol=1e-3) and img[28, 28, 3] < 1e-3
    # alpha-correct: a stamp with a transparent half over white leaves no dark fringe
    half = np.zeros((8, 8, 4), np.float32)
    half[:, :4] = (1, 0, 0, 1)
    over = tex.Canvas(32, 32, bg=WHITE).stamp(half, (0, 0, 1, 1)).image()
    assert over[..., 1].min() >= 0.0 and np.all(over[16, :, 1] >= -1e-6) and np.all(over[16, :, 3] > 0.999)
    assert np.allclose(over[16, 31], [1, 1, 1, 1], atol=1e-3) and np.allclose(over[16, 2], [1, 0, 0, 1], atol=1e-3)
    mid = over[16, 14:18]
    assert np.all(mid[:, 1] >= -1e-6) and np.all((mid[:, 1] == mid[:, 2]))                # a plain red/white blend
    mult = tex.Canvas(16, 16, bg="#808080").stamp(tex.new(4, 4, "#808080"), (0, 0, 1, 1), mode="multiply").image()
    assert np.allclose(mult[8, 8, :3], (128 / 255) ** 2, atol=1e-4)
    assert tex.Canvas(8, 8).stamp(tex.new(2, 2, RED), (0.5, 0.5, 0.5, 0.8)).image()[..., 3].sum() == 0.0


def test_canvas_shape_mask():
    cv = tex.Canvas(100, 80)
    m = cv.shape_mask([(0.25, 0.25), (0.75, 0.25), (0.75, 0.75), (0.25, 0.75)])
    assert m.shape == (80, 100) and m.dtype == np.float32 and m.min() >= 0.0 and m.max() <= 1.0
    assert m.sum() == pytest.approx(0.5 * 0.5 * 100 * 80, abs=1e-3) and m[40, 50] == 1.0 and m[0, 0] == 0.0
    tri = cv.shape_mask([(0.1, 0.1), (0.9, 0.1), (0.5, 0.9)])
    assert tri.sum() == pytest.approx(0.8 * 0.8 / 2 * 100 * 80, rel=0.005)
    assert np.any((tri > 0) & (tri < 1))                                                  # antialiased
    ring = cv.shape_mask([[(0.1, 0.1), (0.9, 0.1), (0.9, 0.9), (0.1, 0.9)], [(0.3, 0.3), (0.7, 0.3), (0.7, 0.7), (0.3, 0.7)]])
    assert ring[40, 50] == 0.0 and ring[40, 15] == 1.0
    assert cv.shape_mask([(2, 2), (3, 2), (3, 3)]).sum() == 0.0                           # outside: empty
    assert not cv._ops                                                                    # nothing is drawn by it
    # per-pixel work: mask * noise through colorize, stamped back
    noise = tex.value_noise(100, 80, 8, seed=1)
    img = tex.colorize(m * noise, "#ff8800")
    assert img.shape == (80, 100, 4) and img[0, 0, 3] == 0.0 and np.allclose(img[40, 50, :3], [1, 136 / 255, 0])
    assert np.isclose(img[40, 50, 3], noise[40, 50], atol=1e-6)
    out = tex.Canvas(100, 80).stamp(img, (0, 0, 1, 1)).image()
    assert np.abs(out[40, 50, 3] - noise[40, 50]) < 0.02


def test_canvas_image_caching_and_copy_semantics():
    cv = tex.Canvas(32, 32).circle((0.5, 0.5), 0.3, fill=RED)
    a = cv.image()
    b = cv.image()
    assert np.array_equal(a, b) and a is not b
    a[:] = 0.5                                                                           # the caller owns the result
    assert np.array_equal(cv.image(), b)
    cv.circle((0.5, 0.5), 0.1, fill=BLUE)                                                # drawing again invalidates
    c = cv.image()
    assert not np.array_equal(b, c) and np.allclose(c[16, 16], [0, 0, 1, 1])


def test_canvas_tiles_render_identically():
    def scene(cv):
        cv.fill_all("#f1e7d6")
        cv.circle((0.3, 0.6), 0.2, fill="#dd5555", stroke="#552222", width=0.02)
        cv.ellipse((0.7, 0.4), (0.25, 0.1), fill="#3355dd", stroke="#112266", width=0.015, angle=30)
        cv.rect(0.1, 0.1, 0.6, 0.35, fill="#22aa44", stroke="#115522", width=0.01, radius=0.04)
        cv.polygon([(0.2, 0.2), (0.9, 0.3), (0.6, 0.9), (0.1, 0.7)], stroke=BLACK, width=0.01)
        cv.curve([(0.05, 0.9), (0.3, 0.95), (0.6, 0.8), (0.95, 0.95)], 0.02, "#222", widths=[0.002, 0.02, 0.01, 0.002])
        cv.bezier((0.1, 0.5), (0.3, 0.9), (0.7, 0.1), (0.9, 0.5), (0.03, 0.002), "#a0a")
        cv.fill_gradient([(0.4, 0.4), (0.8, 0.4), (0.8, 0.8), (0.4, 0.8)], [(0, WHITE), (1, BLACK)], (0.4, 0.4), (0.8, 0.8), mode="multiply")
        cv.fill_radial(None, [(0, "#ffffff80"), (1, "#00000000")], (0.5, 0.5), 0.3, mode="screen")
        cv.stamp(np.random.default_rng(0).random((20, 20, 4)).astype(np.float32), (0.45, 0.05, 0.65, 0.25))
        cv.polyline([(0.1, 0.1), (0.5, 0.12), (0.2, 0.3), (0.9, 0.2)], 0.03, "#ff0", cap="butt", join="miter", mode="add")
        cv.circle((0.5, 0.5), 0.2, fill=WHITE, mode="erase")
        return cv

    ref = scene(tex.Canvas(130, 90)).image()
    for tile in (20, 32, 64):
        c = tex.Canvas(130, 90)
        c.TILE = tile
        assert np.abs(scene(c).image() - ref).max() < 1e-5                               # tiles are exact crops
    assert ref[..., 3].min() == 0.0 and ref.std() > 0.05                                 # the scene is not trivial


def test_canvas_edge_cases_and_validation(tmp_path):
    cv = tex.Canvas(32, 32)
    cv.circle((5.0, 5.0), 0.1, fill=RED).polyline([(-3, -3), (-2, -2)], 0.1, RED).rect(2, 2, 3, 3, fill=RED)
    assert cv.image()[..., 3].sum() == 0.0                                               # far outside: nothing, no crash
    big = tex.Canvas(32, 32).rect(-5, -5, 6, 6, fill=RED).image()
    assert np.all(big[..., 3] == 1.0)                                                    # covering everything is fine
    with pytest.raises(ValueError):
        tex.Canvas(0, 8)
    with pytest.raises(ValueError):
        tex.Canvas(8, 8, ss=0)
    with pytest.raises(ValueError):
        tex.Canvas(8, 8).polygon([(0, 0), (1, 1)], fill=RED)                              # needs 3 points to fill
    with pytest.raises(ValueError):
        tex.Canvas(8, 8).polygon([(0, 0), (np.nan, 1), (1, 0)], fill=RED)
    with pytest.raises(ValueError):
        tex.Canvas(8, 8).polyline([(0, 0), (1, 1)], 0.1, "not a colour")
    tex.Canvas(8, 8).polygon([(0, 0), (1, 1), (1, 0)])                                    # neither fill nor stroke: no-op
    tex.Canvas(8, 8).polyline([(0.5, 0.5), (0.5, 0.5), (0.5, 0.5)], 0.1, RED)             # all points coincide
    tex.Canvas(8, 8).polyline([(0.1, 0.1), (0.1, 0.1), (0.9, 0.9), (0.9, 0.9)], 0.1, RED, closed=True)
    tex.Canvas(8, 8).ellipse((0.5, 0.5), (0.0, 0.0), fill=RED, stroke=BLUE)               # degenerate radii
    path = tex.Canvas(16, 16, bg=RED).save(tmp_path / "out" / "c.png")
    assert path.exists() and np.allclose(tex.load_png(path)[5, 5], tex.color(RED))
    odd = tex.Canvas(7, 5, ss=3).circle((0.5, 0.5), 0.3, fill=RED).image()               # sizes not divisible by anything
    assert odd.shape == (5, 7, 4) and odd[2, 3, 3] == 1.0
    ss1, ss4 = tex.Canvas(64, 64, ss=1), tex.Canvas(64, 64, ss=4)
    for c in (ss1, ss4):
        c.polygon([(0.1, 0.1), (0.9, 0.2), (0.5, 0.9)], fill=WHITE)
    assert abs(area(ss1.image()) - area(ss4.image())) / area(ss4.image()) < 0.02
    tex.Canvas(8, 8).polyline([], 0.1, RED)                                              # nothing to draw is fine
    assert tex.Canvas(8, 8).polyline([], 0.1, RED).image()[..., 3].sum() == 0.0


def test_canvas_width_validation():
    with pytest.raises(ValueError):
        tex.Canvas(8, 8).polyline([(0, 0), (1, 1)], float("nan"), RED)
    with pytest.raises(ValueError):
        tex.Canvas(8, 8).polyline([(0, 0), (1, 1), (0.5, 0.2)], 0.1, RED, widths=[0.1, 0.2])
    with pytest.raises(ValueError):
        tex.Canvas(8, 8).polyline([(0, 0), (1, 1)], 0.1, RED, widths=[0.1, float("inf")])
    assert tex.Canvas(8, 8).polyline([(0, 0), (1, 1)], -0.1, RED).image()[..., 3].sum() == 0.0   # negative width: nothing
    closed = tex.Canvas(128, 128).curve([(0.5, 0.2), (0.8, 0.5), (0.5, 0.8), (0.2, 0.5)], 0.02, WHITE, closed=True,
                                        widths=[0.01, 0.1, 0.01, 0.1]).image()
    right = int((closed[64, 90:, 3] > 0.5).sum())                                        # stroke is vertical here: 0.1 * 128 px
    top = int((closed[:40, 64, 3] > 0.5).sum())                                          # horizontal here: 0.01 * 128 px
    assert 11 <= right <= 14 and 1 <= top <= 2 and closed[64, 64, 3] == 0.0


def brute_winding(poly, xs, ys):
    """Even-odd crossing parity and winding number of a polygon (pixel space) at points, by ray casting."""
    wn = np.zeros(xs.shape, int)
    cr = np.zeros(xs.shape, int)
    for i in range(len(poly)):
        (x0, y0), (x1, y1) = poly[i], poly[(i + 1) % len(poly)]
        up, dn = (y0 <= ys) & (y1 > ys), (y0 > ys) & (y1 <= ys)
        xi = x0 + (ys - y0) / (y1 - y0 if y1 != y0 else 1.0) * (x1 - x0)
        hit = (up | dn) & (xi > xs)
        cr += hit
        wn += np.where(hit & up, 1, 0) - np.where(hit & dn, 1, 0)
    return wn, cr


def test_canvas_polygon_fill_matches_point_in_polygon():
    rng = np.random.default_rng(5)
    w, h = 64, 48
    rows, cols = np.mgrid[0:h, 0:w]
    cx, cy = cols + 0.5, rows + 0.5
    for trial in range(40):
        n = int(rng.integers(3, 10))
        uv = np.stack([rng.random(n) * 1.4 - 0.2, rng.random(n) * 1.4 - 0.2], axis=1)   # partly outside the canvas
        px = np.stack([uv[:, 0] * w, (1 - uv[:, 1]) * h], axis=1)
        wn, cr = brute_winding(px, cx, cy)
        eo = tex.Canvas(w, h, ss=1).polygon(uv, fill=WHITE, rule="evenodd").image()[..., 3] > 0.5
        nz = tex.Canvas(w, h, ss=1).polygon(uv, fill=WHITE, rule="nonzero").image()[..., 3] > 0.5
        assert np.array_equal(eo, cr % 2 == 1) and np.array_equal(nz, wn != 0)


def test_canvas_strokes_match_the_distance_field():
    """A round-capped, round-joined stroke is exactly {p : distance(p, polyline) <= width / 2}."""
    rng = np.random.default_rng(6)
    w, h = 80, 56
    rows, cols = np.mgrid[0:h, 0:w]
    cx, cy = cols + 0.5, rows + 0.5
    for trial in range(40):
        n = int(rng.integers(2, 8))
        closed = trial % 3 == 0 and n >= 3
        px = np.stack([rng.random(n) * 70 + 5, rng.random(n) * 46 + 5], axis=1)
        r = float(rng.uniform(0.6, 8.0))
        uv = np.stack([px[:, 0] / w, 1 - px[:, 1] / h], axis=1)
        got = tex.Canvas(w, h, ss=1).polyline(uv, 2 * r / w, WHITE, closed=closed).image()[..., 3] > 0.5
        segs = list(zip(px[:-1], px[1:])) + ([(px[-1], px[0])] if closed else [])
        dist = np.full(cx.shape, 1e9)
        for a, b in segs:
            ab = b - a
            t = np.clip(((cx - a[0]) * ab[0] + (cy - a[1]) * ab[1]) / max((ab ** 2).sum(), 1e-12), 0, 1)
            dist = np.minimum(dist, np.hypot(cx - (a[0] + t * ab[0]), cy - (a[1] + t * ab[1])))
        wrong = got != (dist <= r)
        assert not np.any(wrong & (np.abs(dist - r) > 0.08)), trial                      # only on the very boundary


def test_canvas_window_independence():
    """Rasterising a window equals cropping the full raster (what the tile renderer relies on)."""
    rng = np.random.default_rng(8)
    uv = np.stack([rng.random(9) * 1.2 - 0.1, rng.random(9) * 1.2 - 0.1], axis=1)
    full = tex.Canvas(90, 70, ss=3).polygon(uv, fill=WHITE).polyline(uv, 0.03, RED).image()
    tiled = tex.Canvas(90, 70, ss=3)
    tiled.TILE = 21                                                                       # 7 px tiles
    tiled.polygon(uv, fill=WHITE).polyline(uv, 0.03, RED)
    assert np.abs(tiled.image() - full).max() < 1e-5


# ============================================================================================= MMD helpers
def test_toon_ramp_rows_and_threshold():
    r = tex.toon_ramp("#c8b4c8")
    assert r.shape == (128, 32, 4) and r.dtype == np.float32
    assert np.allclose(r[0], [1, 1, 1, 1]) and np.allclose(r[-1], tex.color("#c8b4c8"))   # top = lit, bottom = shadow
    assert np.allclose(r[:, 0], r[:, 31])                                                  # constant across the width
    lum = r[:, 0, 1]
    assert np.all(np.diff(lum) <= 1e-7)                                                    # monotonic dark-ward
    mid = 0.5 * (1.0 + tex.color("#c8b4c8")[1])
    row = int(np.argmax(lum < mid))
    assert abs(row - 0.5 * 127) <= 1                                                       # the edge is at the threshold
    low = tex.toon_ramp("#808080", threshold=0.25, softness=0.0, size=(4, 100))
    step = int(np.argmax(low[:, 0, 0] < 0.9))
    assert abs(step - 0.25 * 99) <= 1
    assert np.allclose(np.unique(np.round(low[:, 0, 0], 3)), [128 / 255, 1.0], atol=1e-3)
    high = tex.toon_ramp("#808080", threshold=0.8, softness=0.0, size=(4, 100))
    assert int(np.argmax(high[:, 0, 0] < 0.9)) > 70                                        # more light, later edge
    soft = tex.toon_ramp("#808080", softness=0.4)
    hard = tex.toon_ramp("#808080", softness=0.02)
    assert (soft[:, 0, 0] < 0.99).sum() - (soft[:, 0, 0] < 0.52).sum() > 3 * ((hard[:, 0, 0] < 0.99).sum() - (hard[:, 0, 0] < 0.52).sum())
    custom = tex.toon_ramp("#101010", light="#ffeedd", size=(8, 16))
    assert np.allclose(custom[0, 0], tex.color("#ffeedd")) and np.allclose(custom[-1, 0], tex.color("#101010"))
    assert tex.toon_ramp("#808080", size=(1, 1)).shape == (1, 1, 4)


def test_toon_ramp_deep_and_bands():
    plain = tex.toon_ramp("#808080")
    deep = tex.toon_ramp("#808080", deep="#201060")
    assert np.allclose(deep[-1, 0], tex.color("#201060"))                                  # exactly at the bottom
    t = np.arange(128) / 127.0
    start = (0.5 + 0.05 / 2 + 1.0) / 2                                                     # default deep_start
    assert np.allclose(deep[t <= start], plain[t <= start])                                # untouched above deep_start
    red = deep[t > start][:, 0, 0]                                                         # fading towards the deep colour
    assert np.all(np.diff(red) < 0) and red[0] < plain[t > start][0, 0, 0] and np.isclose(red[-1], 0x20 / 255)
    late = tex.toon_ramp("#808080", deep="#201060", deep_start=0.9)
    assert np.allclose(late[:110], plain[:110]) and np.allclose(late[-1, 0], tex.color("#201060"))
    b = tex.toon_bands([(0, WHITE), (0.5, WHITE), (0.5, "#888888"), (1, "#444444")])
    assert b.shape == (128, 32, 4)
    assert np.allclose(b[0], tex.color(WHITE)) and np.allclose(b[-1], tex.color("#444444"))
    assert np.allclose(b[63, 0], tex.color(WHITE)) and b[64, 0, 0] < 0.6                    # a hard edge in the middle
    assert tex.toon_bands([(0, RED), (1, BLUE)], size=(2, 5)).shape == (5, 2, 4)
    assert np.allclose(tex.toon_bands([(0, RED), (1, BLUE)], size=(2, 3))[1, 0], [0.5, 0, 0.5, 1])


def test_sphere_map_pixel_mapping_and_outside_disc():
    def fn(nx, ny, nz):
        return np.stack([nx * 0.5 + 0.5, ny * 0.5 + 0.5, nz, np.ones_like(nz)], axis=-1)

    n = 8
    m = tex.sphere_map(n, fn)
    assert m.shape == (8, 8, 4) and m.dtype == np.float32
    for row, col in ((1, 6), (3, 4), (6, 2), (4, 3)):
        nx = 2 * (col + 0.5) / n - 1
        ny = 1 - 2 * (row + 0.5) / n                                                       # the top row is ny = +1
        assert np.isclose(m[row, col, 0], 0.5 + 0.5 * nx, atol=1e-6)
        assert np.isclose(m[row, col, 1], 0.5 + 0.5 * ny, atol=1e-6)
        assert np.isclose(m[row, col, 2], np.sqrt(max(0.0, 1 - nx * nx - ny * ny)), atol=1e-6)
    assert m[0, 4, 1] > m[7, 4, 1] and m[3, 7, 0] > m[3, 0, 0]                              # up is up, right is +x
    corner = m[0, 0]                                                                       # outside the disc: the rim
    assert np.isclose(corner[2], 0.0, atol=1e-6)
    assert np.isclose(np.hypot(2 * corner[0] - 1, 2 * corner[1] - 1), 1.0, atol=1e-6)
    assert np.isclose(2 * corner[0] - 1, -np.sqrt(0.5), atol=1e-6) and np.isclose(2 * corner[1] - 1, np.sqrt(0.5), atol=1e-6)
    centre = tex.sphere_map(9, fn)[4, 4]
    assert np.isclose(centre[2], 1.0, atol=1e-6) and np.allclose(centre[:2], 0.5, atol=1e-6)    # the middle faces us
    rect = tex.sphere_map((12, 6), lambda nx, ny, nz: np.ones(nx.shape + (3,)))
    assert rect.shape == (6, 12, 4) and np.all(rect[..., 3] == 1.0)
    assert np.allclose(tex.sphere_map(4, lambda nx, ny, nz: np.array([1.0, 0.0, 0.0])), [1, 0, 0, 1])
    assert tex.sphere_map(4, lambda nx, ny, nz: nx * 0.0 + 0.25)[0, 0, 0] == 0.25            # grey (h, w)
    with pytest.raises(ValueError):
        tex.sphere_map(4, lambda nx, ny, nz: np.zeros((3, 3, 4)))


def test_sphere_highlight_band():
    size = 256
    h = tex.sphere_highlight(size)
    assert h.shape == (size, size, 4) and np.all(h[..., 3] == 1.0)
    prof = h[:, size // 2, 0]
    expected_row = (1 - 0.35) / 2 * size - 0.5                                             # ny = 0.35 -> row 82.7
    assert abs(int(prof.argmax()) - expected_row) <= 1 and prof.max() > 0.999
    ny_rows = 1 - 2 * (np.arange(size) + 0.5) / size
    assert np.all(prof[ny_rows > 0.35 + 0.12] == 0.0) and np.all(prof[ny_rows < 0.35 - 0.12] == 0.0)   # black outside
    assert np.all(np.diff(prof[: int(prof.argmax()) + 1]) >= -1e-7)                       # smooth rise, then a fall
    assert np.all(np.diff(prof[int(prof.argmax()):]) <= 1e-7)
    assert np.allclose(h[..., 0], h[..., 1]) and np.allclose(h[..., 0], h[..., 2])          # white colour, grey values
    centre = h[:, size // 2, 0]
    side = h[:, 20, 0]                                                                     # inside the disc: same band
    assert np.allclose(centre[70:95], side[70:95], atol=1e-5)
    tint = tex.sphere_highlight(64, band=(0.0, 0.2), color="#ff8040", strength=0.5)
    peak = tint[31:33, 32, :3].max(axis=0)
    assert np.allclose(peak, np.array([1.0, 128 / 255, 64 / 255]) * 0.5, atol=0.02)         # colour x strength, additive
    assert np.allclose(tint[0, 32], [0, 0, 0, 1])
    faded = tex.sphere_highlight(64, side_fade=0.8)
    plain_band = tex.sphere_highlight(64)
    assert faded[:, 32, 0].max() > 0.95 and faded[:, 3, 0].max() < 0.5                       # dimmer towards the limb
    assert plain_band[:, 3, 0].max() > 0.95 and np.allclose(plain_band[:, 3, 0], plain_band[:, 32, 0])   # uniform in nx
    half = tex.sphere_highlight(64, color=(1, 1, 1, 0.5))
    assert np.isclose(half[:, 32, 0].max(), 0.5, atol=0.02)                                 # colour alpha scales it
    mul = tex.sphere_highlight(64, mode="mul", color="#806040")
    assert np.allclose(mul[0, 0], [1, 1, 1, 1]) and np.allclose(mul[63, 63], [1, 1, 1, 1])  # white = no change
    assert np.allclose(mul[:, 32, :3].min(axis=0), [128 / 255, 96 / 255, 64 / 255], atol=0.02)
    assert np.all(mul[..., :3] <= 1.0) and mul[..., 0].min() < 0.6
    white = tex.sphere_highlight(32, mode="mul")                                            # white band: no effect
    assert np.allclose(white, 1.0)
    with pytest.raises(ValueError):
        tex.sphere_highlight(8, mode="screen")


def test_sphere_rim_and_flat():
    n = 65
    rim = tex.sphere_rim(n, power=3.0, strength=0.6)
    assert rim.shape == (n, n, 4) and np.all(rim[..., 3] == 1.0)
    assert np.allclose(rim[n // 2, n // 2, :3], 0.0, atol=1e-6)                             # dark where it faces us
    edge = rim[n // 2, 0, 0]                                                               # nz small at the rim
    assert 0.3 < edge <= 0.6 and np.allclose(rim[0, 0, :3], 0.6, atol=1e-5)                 # corner = rim value
    assert np.all(np.diff(rim[n // 2, : n // 2 + 1, 0]) <= 1e-7)                            # grows towards the rim
    nx = 2 * (10 + 0.5) / n - 1
    nz = np.sqrt(1 - nx ** 2 - (1 - 2 * (32 + 0.5) / n) ** 2)
    assert np.isclose(rim[32, 10, 0], 0.6 * (1 - nz) ** 3, atol=1e-5)                       # the documented profile
    sharp = tex.sphere_rim(n, power=8.0, strength=0.6)
    assert sharp[n // 2, 8, 0] < rim[n // 2, 8, 0]                                          # a higher power is narrower
    mul = tex.sphere_rim(32, color="#808080", mode="mul", strength=1.0)
    assert np.allclose(mul[16, 16, :3], 1.0, atol=0.01) and mul[0, 0, 0] < 0.55
    flat = tex.sphere_flat(8, "#ff000080")
    assert flat.shape == (8, 8, 4) and np.allclose(flat, [1, 0, 0, 128 / 255])
    assert tex.sphere_flat().shape == (64, 64, 4) and np.allclose(tex.sphere_flat(), 1.0)
    with pytest.raises(ValueError):
        tex.sphere_rim(8, mode="burn")


# ======================================================================================================= atlas
def rects_px(atlas):
    return {n: tex.uv_rect_pixels(r, atlas.w, atlas.h) for n, r in atlas.rects.items()}


def test_atlas_alloc_no_overlap_inside_bounds():
    rng = np.random.default_rng(7)
    atlas = tex.Atlas(512, 512, padding=2)
    sizes = [(int(rng.integers(16, 120)), int(rng.integers(16, 120))) for _ in range(40)]
    sizes.sort(key=lambda s: -s[1])
    placed = []
    for i, (w, h) in enumerate(sizes):
        try:
            rect = atlas.alloc(w, h, "s%d" % i)
        except ValueError:
            break
        placed.append((w, h, rect))
    assert len(placed) >= 15
    boxes = []
    for w, h, rect in placed:
        x0, y0, x1, y1 = tex.uv_rect_pixels(rect, 512, 512)
        assert (x1 - x0, y1 - y0) == (w, h)                                                # exact pixel size
        assert 2 <= x0 and x1 <= 512 - 2 and 2 <= y0 and y1 <= 512 - 2                      # gutter inside the atlas
        boxes.append((x0 - 2, y0 - 2, x1 + 2, y1 + 2))                                      # footprint incl. gutter
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            a, b = boxes[i], boxes[j]
            assert a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1], (i, j)    # footprints never overlap
    assert set(atlas.rects) == {"s%d" % i for i in range(len(placed))}


def test_atlas_uv_orientation_and_rects():
    atlas = tex.Atlas(256, 128, padding=2)
    r0 = atlas.alloc(100, 40, "first")
    u0, v0, u1, v1 = r0
    assert (u0, u1) == (2 / 256, 102 / 256)                                                # u from the left
    assert v1 == pytest.approx(1 - 2 / 128) and v0 == pytest.approx(1 - 42 / 128)           # the first shelf is the TOP
    assert atlas.rects["first"] == r0
    r1 = atlas.alloc(50, 30, "second")                                                     # same shelf, to the right
    assert r1[0] == pytest.approx(106 / 256) and r1[3] == pytest.approx(1 - 2 / 128)
    r2 = atlas.alloc(200, 20)                                                              # wider: next shelf below
    assert r2[3] < r0[1] and "second" in atlas.rects and len(atlas.rects) == 2             # unnamed ones are not stored
    assert tex.uv_rect_pixels(r0, 256, 128) == (2, 2, 102, 42)
    assert tex.uv_rect_pixels((0.0, 0.0, 1.0, 1.0), 64, 32) == (0, 0, 64, 32)
    assert tex.uv_rect_pixels((0.5, 0.5, 0.0, 1.0), 64, 32) == (0, 0, 32, 16)
    tight = tex.Atlas(64, 64, padding=0)
    assert tight.alloc(64, 64) == (0.0, 0.0, 1.0, 1.0)                                       # no padding: it can fill all
    with pytest.raises(ValueError):
        tight.alloc(1, 1)


def test_atlas_shelf_reuse_and_growth():
    atlas = tex.Atlas(100, 100, padding=0)
    a = atlas.alloc(40, 30, "a")
    b = atlas.alloc(40, 50, "b")                                                           # the open shelf grows to 50
    c = atlas.alloc(20, 10, "c")                                                           # fits at the end of shelf 1
    px = rects_px(atlas)
    assert px["a"][1] == px["b"][1] == px["c"][1] == 0 and px["c"][0] == 80
    d = atlas.alloc(60, 20, "d")                                                           # shelf 2 starts below the 50
    assert rects_px(atlas)["d"][1] == 50 and a != b and c != d
    e = atlas.alloc(30, 20, "e")                                                           # reuses the room beside it
    assert rects_px(atlas)["e"][1] == 50 and rects_px(atlas)["e"][0] == 60


def test_atlas_errors():
    atlas = tex.Atlas(128, 128, padding=2)
    atlas.alloc(30, 30, "x")
    with pytest.raises(ValueError):
        atlas.alloc(10, 10, "x")                                                           # duplicate name
    with pytest.raises(ValueError):
        atlas.alloc(125, 10)                                                               # + gutter does not fit
    with pytest.raises(ValueError):
        atlas.alloc(0, 10)
    n = 0
    with pytest.raises(ValueError, match="full"):
        for _ in range(100):
            atlas.alloc(30, 30)
            n += 1
    assert n == 8                                                                          # 3 x 3 slots of 34 px in 128 x 128
    small = tex.Atlas(32, 32, padding=2)
    with pytest.raises(ValueError):
        small.alloc(40, 40)
    with pytest.raises(ValueError):
        tex.Atlas(0, 8)
    with pytest.raises(ValueError):
        tex.Atlas(8, 8, padding=-1)


def test_atlas_blit_places_pixels_and_fills_gutter():
    atlas = tex.Atlas(64, 64, padding=2)
    atlas.alloc(20, 10, "a")
    atlas.alloc(16, 16, "b")
    img = np.zeros((10, 20, 4), np.float32)
    img[:5] = (1, 0, 0, 1)                                                                 # top half red
    img[5:] = (0, 0, 1, 1)                                                                 # bottom half blue
    assert atlas.blit("a", img) == atlas.rects["a"]
    x0, y0, x1, y1 = tex.uv_rect_pixels(atlas.rects["a"], 64, 64)
    out = atlas.image()
    assert np.allclose(out[y0, x0], [1, 0, 0, 1]) and np.allclose(out[y1 - 1, x1 - 1], [0, 0, 1, 1])
    assert np.allclose(out[y0 + 4, x0 + 3], [1, 0, 0, 1]) and np.allclose(out[y0 + 5, x0 + 3], [0, 0, 1, 1])
    # the gutter repeats the edge pixels (a bilinear tap beside the slot sees the island's colour)
    assert np.allclose(out[y0 - 1, x0 + 5], [1, 0, 0, 1]) and np.allclose(out[y0 - 2, x0 + 5], [1, 0, 0, 1])
    assert np.allclose(out[y1 + 1, x0 + 5], [0, 0, 1, 1]) and np.allclose(out[y0 + 2, x0 - 2], [1, 0, 0, 1])
    assert np.allclose(out[y0 - 2, x0 - 2], [1, 0, 0, 1]) and np.allclose(out[y1 + 1, x1 + 1], [0, 0, 1, 1])
    assert out[0, 63, 3] == 0.0                                                            # elsewhere untouched
    bx0, by0, bx1, by1 = tex.uv_rect_pixels(atlas.rects["b"], 64, 64)
    assert out[by0 + 3, bx0 + 3, 3] == 0.0                                                 # b was never written
    atlas.blit("b", tex.new(4, 4, GREEN))                                                  # another size is resampled
    out = atlas.image()
    assert np.allclose(out[by0, bx0], [0, 1, 0, 1], atol=1e-5) and np.allclose(out[by1 - 1, bx1 - 1], [0, 1, 0, 1], atol=1e-5)
    # blit into an arbitrary UV rect, and with a blend mode
    atlas.blit(atlas.rects["b"], tex.new(16, 16, "#808080"), mode="multiply")
    assert np.allclose(atlas.img[by0 + 3, bx0 + 3], [0, 128 / 255, 0, 1], atol=1e-3)
    live = atlas.image()
    live[:] = 0.0
    assert atlas.img.max() > 0                                                             # image() returns a copy
    with pytest.raises(KeyError):
        atlas.blit("nope", img)


def test_atlas_without_gutter_and_map_uv():
    atlas = tex.Atlas(32, 32, padding=0)
    r = atlas.alloc(8, 8, "p")
    atlas.blit("p", tex.new(8, 8, RED))
    x0, y0, x1, y1 = tex.uv_rect_pixels(r, 32, 32)
    assert (atlas.img[..., 0] > 0).sum() == 64 and atlas.img[y0, x0, 0] == 1.0
    uv = np.array([[0.0, 0.0], [1.0, 1.0], [0.5, 0.25]])
    mapped = atlas.map_uv("p", uv)
    assert np.allclose(mapped[0], [r[0], r[1]]) and np.allclose(mapped[1], [r[2], r[3]])
    assert np.allclose(mapped[2], [r[0] + 0.5 * (r[2] - r[0]), r[1] + 0.25 * (r[3] - r[1])])
    assert np.allclose(tex.remap_uv(uv, (0.5, 0.0, 1.0, 0.5))[1], [1.0, 0.5])
    assert np.allclose(uv, [[0, 0], [1, 1], [0.5, 0.25]])                                    # the input is untouched
    stack = tex.remap_uv(np.zeros((4, 3, 2)), (0.1, 0.2, 0.3, 0.4))
    assert stack.shape == (4, 3, 2) and np.allclose(stack[..., 0], 0.1)
    assert np.allclose(atlas.map_uv((0.0, 0.0, 0.5, 0.5), [0.5, 0.5]), [0.25, 0.25])


def test_uv_grid_and_colorize():
    u, v = tex.uv_grid(4, 2)
    assert u.shape == v.shape == (2, 4) and u.dtype == np.float32
    assert np.allclose(u[0], [0.125, 0.375, 0.625, 0.875]) and np.allclose(v[:, 0], [0.75, 0.25])   # v is up
    mask = ((u - 0.5) ** 2 + (v - 0.5) ** 2) < 0.1
    assert mask.shape == (2, 4)
    c = tex.colorize(np.array([[0.0, 0.5, 2.0]], np.float32), "#ff000080")
    assert c.shape == (1, 3, 4) and np.allclose(c[0, :, :3], [1, 0, 0])
    assert np.allclose(c[0, :, 3], [0.0, 0.5 * 128 / 255, 128 / 255])                       # clipped mask x colour alpha
    base = tex.linear_gradient(6, 4, [(0, "#336699"), (1, "#996633")])
    mask = np.random.default_rng(2).random((4, 6)).astype(np.float32)
    via_image = tex.composite(base, tex.colorize(mask, "#ff8800"), "multiply")
    via_opacity = tex.composite(base, tex.new(6, 4, "#ff8800"), "multiply", opacity=mask)
    assert np.allclose(via_image, via_opacity, atol=1e-6)                                    # the two documented routes


def test_shade_blur_and_noise_arguments():
    img = tex.new(4, 3, "#808080")
    mottle = 0.5 + np.linspace(0.0, 1.0, 12, dtype=np.float32).reshape(3, 4)               # per-pixel factor (h, w)
    out = tex.shade(img, mottle)
    assert out.shape == (3, 4, 4) and out[0, 0, 0] < out[2, 3, 0]
    assert np.allclose(out[2, 3, :3], tex.shade("#808080", 1.5)[:3], atol=1e-6)
    u8 = np.zeros((6, 6, 4), np.uint8)
    u8[2:4, 2:4] = (255, 0, 0, 255)
    assert tex.blur(u8, 1.0).dtype == np.float32 and np.isclose(tex.blur(u8, 1.0)[..., 3].sum(), 4.0, rtol=0.02)
    tile = tex.value_noise(32, 32, scale=1.0, seed=3, tileable=True)                       # tileable: at least 2 cells
    assert tile.std() > 0.01
    for bad in (0.0, -1.0, float("nan"), (4.0, 0.0)):
        with pytest.raises(ValueError):
            tex.value_noise(8, 8, scale=bad)
    assert tex.value_noise(8, 8, seed=2 ** 40).shape == (8, 8)                               # any int seed works


def test_translucent_stroke_does_not_double_up_where_it_crosses_itself():
    alpha = 128 / 255
    cross = [(0.1, 0.5), (0.9, 0.5), (0.5, 0.9), (0.5, 0.1)]                                # the path crosses itself
    img = tex.Canvas(100, 100).polyline(cross, 0.1, (1, 1, 1, alpha), cap="butt", join="round").image()
    assert np.isclose(img[50, 50, 3], alpha, atol=1e-5)                                      # one layer, not 1 - (1-a)^2
    assert np.isclose(img[50, 20, 3], alpha, atol=1e-5) and np.isclose(img[20, 50, 3], alpha, atol=1e-5)
    erased = tex.Canvas(100, 100, bg=WHITE).polyline(cross, 0.1, (1, 1, 1, alpha), cap="butt", mode="erase").image()
    assert np.isclose(erased[50, 50, 3], 1 - alpha, atol=1e-5)


def sample_uv(img, u, v):
    """Nearest texel at UV (u, v), v up, as a renderer with the PMX v-flip applied would read the PNG."""
    h, w = img.shape[:2]
    return img[min(int((1.0 - v) * h), h - 1), min(int(u * w), w - 1)]


def test_part_builder_workflow_keeps_v_up_through_canvas_atlas_and_png(tmp_path):
    """The way builders use the module: draw islands, pack them, remap the mesh UVs, save, read back, sample."""
    eye = tex.Canvas(96, 64, bg="#ffffff")
    eye.circle((0.25, 0.75), 0.06, fill=RED)                                               # upper left of the island
    eye.circle((0.75, 0.25), 0.06, fill=BLUE)                                              # lower right of the island
    skin = tex.linear_gradient(128, 128, [(0, "#f9f0e1"), (1, "#c8a080")])                  # light at v = 1
    atlas = tex.Atlas(256, 256, padding=2)
    r_skin = atlas.alloc(128, 128, "skin")
    r_eye = atlas.alloc(96, 64, "eye")
    atlas.blit("skin", skin)
    atlas.blit("eye", eye.image())
    path = tex.save_png(tmp_path / "tex" / "face.png", tex.bleed(atlas.image()))
    back = tex.load_png(path)
    mesh_uv = np.array([[0.25, 0.75], [0.75, 0.25], [0.5, 0.95], [0.5, 0.05]])              # per-corner island UVs
    atlas_uv = atlas.map_uv("eye", mesh_uv)
    assert np.allclose(atlas_uv, tex.remap_uv(mesh_uv, r_eye))
    assert sample_uv(back, *atlas_uv[0])[:3] == pytest.approx([1, 0, 0], abs=0.01)           # the red dot is at (.25, .75)
    assert sample_uv(back, *atlas_uv[1])[:3] == pytest.approx([0, 0, 1], abs=0.01)
    assert sample_uv(back, *atlas_uv[2])[:3] == pytest.approx([1, 1, 1], abs=0.01)           # white elsewhere in the island
    top, bottom = sample_uv(back, *tex.remap_uv([0.5, 0.95], r_skin)), sample_uv(back, *tex.remap_uv([0.5, 0.05], r_skin))
    assert top[0] > bottom[0] and top[1] > bottom[1]                                        # v = 1 is the light end
    assert np.allclose(sample_uv(back, 0.99, 0.01), 0.0)                                    # unused corner stays empty
    assert back.shape == (256, 256, 4) and back.dtype == np.float32
    pix = tex.uv_rect_pixels(r_eye, 256, 256)
    assert np.allclose(back[pix[1]:pix[3], pix[0]:pix[2], 3], 1.0)                          # the whole island is opaque


def test_module_docstring_example_runs(tmp_path):
    cv = tex.Canvas(512, 512, bg="#f1e7d6")
    cv.circle((0.5, 0.5), 0.2, fill="#dd5555", stroke="#552222", width=0.01)
    cv.curve([(0.2, 0.8), (0.5, 0.9), (0.8, 0.8)], 0.012, "#222", widths=[0.002, 0.012, 0.002])
    grain = tex.colorize(tex.value_noise(512, 512, 24, seed=3) * 0.15, "#000")
    path = tex.save_png(tmp_path / "skin.png", tex.composite(cv.image(), grain))
    img = tex.load_png(path)
    assert img.shape == (512, 512, 4) and np.all(img[..., 3] == 1.0)
    assert sample_uv(img, 0.5, 0.5)[0] > 0.7 and sample_uv(img, 0.5, 0.5)[1] < 0.5               # the red disc
    assert sample_uv(img, 0.05, 0.05)[0] > 0.7 and sample_uv(img, 0.05, 0.05)[2] > 0.5         # the skin colour
