"""Kinetic type, bpy-free: surface frames, alignment words, number formats, typewriter and blink keys, fit and layout
(mkmmd.core.typeset). The Blender half of the `text` stage is checked by building and looking (docs/design.md: Text)."""

import numpy as np
import pytest

from mkmmd.core import typeset as T


# ---------------------------------------------------------------- frames
def test_surface_matrix_is_a_right_handed_frame_with_right_equal_up_cross_normal():
    m = T.surface_matrix([1.0, 2.0, 3.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0])
    r, u, n, p = m[:3, 0], m[:3, 1], m[:3, 2], m[:3, 3]
    assert np.allclose(n, [-1, 0, 0]) and np.allclose(u, [0, 0, 1])
    assert np.allclose(r, np.cross(u, n)) and np.allclose(r, [0, -1, 0])     # a driver heading +X sees -Y on his right
    assert np.allclose(np.cross(r, u), n) and np.linalg.det(m[:3, :3]) == pytest.approx(1.0)
    assert np.allclose(p, [1, 2, 3])


def test_surface_matrix_lifts_along_the_normal_and_orthogonalises_up():
    m = T.surface_matrix([0, 0, 0], [0, 1, 0], [0, 1, 1], lift=0.002)
    assert np.allclose(m[:3, 3], [0, 0.002, 0])
    assert np.allclose(m[:3, 1], [0, 0, 1])                      # the slanted hint loses its part along the normal
    assert np.allclose(m[:3, :3].T @ m[:3, :3], np.eye(3))


def test_surface_matrix_facing_up_without_a_usable_up_picks_plus_y():
    m = T.surface_matrix([0, 0, 0], [0, 0, 1])
    assert np.allclose(m[:3, 1], [0, 1, 0]) and np.allclose(m[:3, 0], [1, 0, 0])      # looking down, +Y up, +X right
    with pytest.raises(T.TextError):
        T.surface_matrix([0, 0, 0], [0, 0, 0])


# ---------------------------------------------------------------- alignment words
@pytest.mark.parametrize("align,valign,want", [
    (None, None, ("center", "middle")),
    ("left", None, ("left", "middle")),
    ("right", "top", ("right", "top")),
    ("bottom", None, ("center", "bottom")),
    ("center middle", None, ("center", "middle")),
    ("top left", None, ("left", "top")),
    (["right", "bottom"], None, ("right", "bottom")),
    ("left", "center", ("left", "middle")),
    ("centre", "bottom", ("center", "bottom")),
    ("left, top", None, ("left", "top")),
])
def test_parse_align(align, valign, want):
    assert T.parse_align(align, valign) == want


@pytest.mark.parametrize("bad", ["left right", "up", "top bottom"])
def test_parse_align_rejects_nonsense(bad):
    with pytest.raises(T.TextError):
        T.parse_align(bad)


# ---------------------------------------------------------------- number formats
def test_parse_format_reads_prefix_suffix_precision_and_padding():
    assert T.parse_format("{:.0f}") == T.NumberFormat("", "", 0, 0, " ")
    assert T.parse_format("{:.1f} MPH") == T.NumberFormat("", " MPH", 1, 0, " ")
    assert T.parse_format("T{:03d}") == T.NumberFormat("T", "", 0, 3, "0")
    assert T.parse_format("{:4.0f}%") == T.NumberFormat("", "%", 0, 4, " ")
    assert T.parse_format("{:d}").decimals == 0


@pytest.mark.parametrize("bad", ["{}", "{:x}", "{:+.1f}", "{:.1f}{:.1f}", "no field", "{:>5}", "{:.2d}", "{x}"])
def test_parse_format_rejects_what_nodes_cannot_rebuild(bad):
    with pytest.raises(T.TextError):
        T.parse_format(bad)


def test_number_format_matches_python_for_the_supported_forms():
    for fmt in ("{:.0f}", "{:.1f}", "{:.2f}", "{:03.0f}", "{:5.1f}", "{:d}"):
        nf = T.parse_format(fmt)
        for v in (0, 3, 58, 71.4, 9.96, 100):
            want = fmt.format(int(round(v)) if fmt == "{:d}" else v)
            assert nf.format(v) == want, (fmt, v)
    assert T.parse_format("{:.0f} MPH").format(58.4) == "58 MPH"
    assert T.parse_format("{:.0f}").format(-0.2) == "0"           # no "-0" on a display
    assert T.parse_format("{:04.0f}").format(-5) == "-005"


def test_candidate_strings_cover_every_value_of_a_sweep():
    nf = T.parse_format("{:.0f}")
    s = T.candidate_strings(58, 71, nf)
    assert s[:2] == ["58", "71"] and set(s) == {str(i) for i in range(58, 72)}
    assert T.candidate_strings(71, 58, nf) == s
    tenths = T.candidate_strings(0.0, 1.0, T.parse_format("{:.1f}"))
    assert len(tenths) == 11 and "0.5" in tenths
    assert T.candidate_strings(3, 3, nf) == ["3"]
    assert T.candidate_strings(0.2, 0.4, nf) == ["0"]


def test_candidate_strings_keep_the_longest_when_there_are_too_many():
    nf = T.parse_format("{:.0f}")
    s = T.candidate_strings(0, 2000, nf, limit=20)
    assert len(s) <= 20 and "0" in s and "2000" in s
    assert max(len(x) for x in s) == 4 and sum(len(x) == 4 for x in s) >= 5


def test_ghost_string_lights_every_segment_and_keeps_punctuation():
    assert T.ghost_string("58") == "88"
    assert T.ghost_string("12:30.5 MPH") == "88:88.8 888"
    assert T.ghost_string("A\nB") == "8\n8"


def test_ghost_colour_is_darker_and_less_saturated_than_the_lit_colour():
    from mkmmd.core import palette as P
    pal = P.get("rose-pine-moon")

    def luma_sat(hexstr):
        r, g, b = P.srgb(hexstr)
        return 0.2126 * r + 0.7152 * g + 0.0722 * b, (max(r, g, b) - min(r, g, b)) / max(r, g, b)

    for slot in ("gold", "love", "foam", "text"):
        lit = pal[slot]
        ghost = T.ghost_colour(lit, pal["muted"])
        assert luma_sat(ghost)[0] < luma_sat(lit)[0]
        assert luma_sat(ghost)[1] < luma_sat(lit)[1] or slot == "text"      # text is already nearly grey
    assert T.ghost_colour(pal["gold"], pal["muted"], 0.0) == pal["gold"]
    assert T.ghost_colour(pal["gold"], pal["muted"], 1.0) == pal["muted"]


def test_display_share_turns_a_displayed_brightness_into_an_emission():
    assert T.display_share(1.0) == 1.0 and T.display_share(0.0) == 0.0
    assert T.display_share(0.10) == pytest.approx(0.10 ** 2.2)
    shares = [T.display_share(s) for s in np.linspace(0, 1, 11)]
    assert shares == sorted(shares) and shares[5] < 0.5          # half as bright to the eye is far less light
    for bad in (-0.1, 1.5):
        with pytest.raises(T.TextError):
            T.display_share(bad)


# ---------------------------------------------------------------- typewriter
def test_reveal_keys_ramp():
    k = T.reveal_keys(24.94, 25.9, 30)
    assert [t for t, _ in k] == pytest.approx([24.94 - 1 / 30, 24.94, 25.9])
    assert [r for _, r in k] == [-1.0, 0.0, 1.0]
    with pytest.raises(T.TextError):
        T.reveal_keys(2.0, 2.0, 30)


def test_reveal_count_types_the_first_character_at_from_and_the_last_at_to():
    n = 12
    assert T.reveal_count(-1.0, n) == 0
    assert T.reveal_count(0.0, n) == 1
    assert T.reveal_count(1.0, n) == n
    assert T.reveal_count(0.5, 1) == 1
    for k in range(n):                                            # character k (0-based) is up at r = k / (n - 1)
        r = k / (n - 1)
        assert T.reveal_count(r, n) == k + 1
        assert T.reveal_count(r - 1e-3, n) == k if k else True
    counts = [T.reveal_count(r, n) for r in np.linspace(-1, 1.2, 200)]
    assert counts == sorted(counts) and max(counts) == n


# ---------------------------------------------------------------- blink and flicker
def test_blink_keys_alternate_between_on_and_low_inside_the_window():
    k = T.blink_keys({"period": 1.0, "duty": 0.5, "low": 0.1, "from": 2.0, "to": 5.0}, 0.0, 10.0)
    ts = [t for t, _ in k]
    assert ts == sorted(ts)
    gains = {round(t, 3): g for t, g in k}
    assert gains[2.0] == 1.0 and gains[2.5] == 0.1 and gains[3.0] == 1.0 and gains[3.5] == 0.1
    assert k[-1] == (5.0, 1.0)                                     # on again after the window


def test_blink_defaults_span_the_clip():
    k = T.blink_keys({"period": 2.0}, 0.0, 6.0)
    assert k[0][0] < 0 and k[-1][0] == 6.0
    with pytest.raises(T.TextError):
        T.blink_keys({"period": 0}, 0, 1)
    with pytest.raises(T.TextError):
        T.blink_keys({"duty": 0}, 0, 1)


def test_flicker_keys_are_seeded_bounded_and_end_lit():
    spec = {"amount": 0.6, "rate": 10, "seed": 4, "from": 1.0, "to": 4.0}
    a, b = T.flicker_keys(spec, 0, 10), T.flicker_keys(spec, 0, 10)
    assert a == b and a != T.flicker_keys(dict(spec, seed=5), 0, 10)
    g = np.array([v for _, v in a])
    assert g.min() >= 1 - 0.6 - 1e-9 and g.max() <= 1.0 and g.min() < 0.9
    assert a[-1] == (4.0, 1.0) and a[0][1] == 1.0
    assert all(a[i][0] < a[i + 1][0] for i in range(len(a) - 1))
    with pytest.raises(T.TextError):
        T.flicker_keys({"amount": 2.0}, 0, 1)


def test_product_keys_multiply_two_step_gains():
    a = [(0.0, 1.0), (1.0, 0.5), (3.0, 1.0)]
    b = [(0.0, 1.0), (2.0, 0.2), (4.0, 1.0)]
    p = dict(T.product_keys(a, b))
    assert p[0.0] == 1.0 and p[1.0] == 0.5 and p[2.0] == pytest.approx(0.1) and p[3.0] == pytest.approx(0.2)
    assert p[4.0] == 1.0


# ---------------------------------------------------------------- layout
def box(w, h, x0=0.0, y0=0.0):
    return (x0, y0, x0 + w, y0 + h)


def test_fit_fills_the_limiting_dimension_to_the_margin():
    # a wide string on a 4.2 x 2.4 panel: width limits, 85 % of 4.2 m
    lay = T.plan_layout(box(8.0, 1.0), 0.7, panel=(4.2, 2.4), fit=0.85)
    assert lay.ink[0] == pytest.approx(4.2 * 0.85) and lay.ink[1] < 2.4 * 0.85 and lay.fits
    assert lay.cap == pytest.approx(0.7 * lay.em)
    # a tall block: height limits
    tall = T.plan_layout(box(1.0, 4.0), 0.7, panel=(4.2, 2.4), fit=0.85)
    assert tall.ink[1] == pytest.approx(2.4 * 0.85) and tall.ink[0] < 4.2 * 0.85


def test_centred_text_has_its_ink_centred_on_the_panel_plus_offset():
    ref = box(3.0, 0.5, x0=0.1, y0=-0.02)
    lay = T.plan_layout(ref, 0.5, panel=(2.0, 1.0), fit=0.8, offset=(0.05, -0.1))
    cx = lay.tx + (ref[0] + ref[2]) / 2 * lay.em
    cy = lay.ty + (ref[1] + ref[3]) / 2 * lay.em
    assert cx == pytest.approx(0.05) and cy == pytest.approx(-0.1)


@pytest.mark.parametrize("align,valign", [("left", "top"), ("right", "bottom"), ("left", "bottom"), ("right", "top")])
def test_edge_alignment_puts_the_ink_on_the_edge_of_the_fit_region(align, valign):
    ref = box(2.0, 0.6, x0=0.03, y0=-0.1)
    lay = T.plan_layout(ref, 0.5, panel=(4.0, 2.0), size=0.2, fit=0.9, align=align, valign=valign)
    x0, x1 = lay.tx + ref[0] * lay.em, lay.tx + ref[2] * lay.em
    y0, y1 = lay.ty + ref[1] * lay.em, lay.ty + ref[3] * lay.em
    aw, ah = 4.0 * 0.9, 2.0 * 0.9
    assert (x0 if align == "left" else x1) == pytest.approx(-aw / 2 if align == "left" else aw / 2)
    assert (y0 if valign == "bottom" else y1) == pytest.approx(-ah / 2 if valign == "bottom" else ah / 2)


def test_size_sets_the_cap_height_and_fit_caps_it():
    ref = box(5.0, 1.0)
    lay = T.plan_layout(ref, 0.7, panel=(4.0, 2.0), size=0.1)
    assert lay.cap == pytest.approx(0.1) and lay.em == pytest.approx(0.1 / 0.7)
    capped = T.plan_layout(ref, 0.7, panel=(4.0, 2.0), size=0.5, fit=0.5)       # 0.5 m caps would be 3.5 m wide
    assert capped.ink[0] == pytest.approx(4.0 * 0.5) and capped.cap < 0.5
    big = T.plan_layout(ref, 0.7, panel=(1.0, 1.0), size=0.5)                    # no fit given: size wins, overflows
    assert big.cap == pytest.approx(0.5) and not big.fits


def test_free_text_without_a_panel_needs_a_size_and_anchors_at_the_origin():
    ref = box(2.0, 0.5, x0=0.1)
    with pytest.raises(T.TextError):
        T.plan_layout(ref, 0.5)
    lay = T.plan_layout(ref, 0.5, size=0.1, align="left", valign="bottom")
    assert lay.tx + ref[0] * lay.em == pytest.approx(0.0) and lay.ty + ref[1] * lay.em == pytest.approx(0.0)
    assert lay.fits and lay.usable == (0.0, 0.0)


def test_layout_rejects_inkless_text_and_bad_fit():
    with pytest.raises(T.TextError):
        T.plan_layout((0, 0, 0, 0), 0.5, panel=(1, 1))
    with pytest.raises(T.TextError):
        T.plan_layout(box(1, 1), 0.5, panel=(1, 1), fit=1.5)
    with pytest.raises(T.TextError):
        T.plan_layout(box(1, 1), 0.0, panel=(1, 1))
