"""Screen type, the numbers (mkmmd/core/screentype.py): the entry per output, the band, the camera that frames the picture and
the compositing. Synthetic numbers only; the Blender half is checked by a build and a render (docs/design.md)."""
import numpy as np
import pytest

from mkmmd.core import screentype as SR
from mkmmd.core.typeset import TextError

S169, S916 = (1920, 1080), (1080, 1920)


# ------------------------------------------------------------------------------------------------------ the entry
def test_deep_merge_merges_tables_and_replaces_everything_else_and_leaves_both_alone():
    base = {"a": 1, "t": {"x": 1, "y": [1, 2], "n": {"p": 1}}, "l": [1, 2]}
    over = {"t": {"y": [3], "n": {"q": 2}}, "l": [9], "b": 2}
    got = SR.deep_merge(base, over)
    assert got == {"a": 1, "t": {"x": 1, "y": [3], "n": {"p": 1, "q": 2}}, "l": [9], "b": 2}
    assert base == {"a": 1, "t": {"x": 1, "y": [1, 2], "n": {"p": 1}}, "l": [1, 2]}
    assert over["t"] == {"y": [3], "n": {"q": 2}}
    assert SR.deep_merge(base, None) == base and SR.deep_merge(base, None) is not base


def test_a_table_replaces_a_scalar_and_a_scalar_replaces_a_table():
    assert SR.deep_merge({"a": 1}, {"a": {"x": 1}}) == {"a": {"x": 1}}
    assert SR.deep_merge({"a": {"x": 1}}, {"a": 2}) == {"a": 2}


def test_an_entry_with_no_aspect_and_no_screen_is_one_object_for_every_output():
    spec = {"name": "t", "text": "aa"}
    assert SR.per_output(spec, ["16x9", "9x16"]) == [(None, spec)]


def test_screen_or_aspect_builds_the_entry_once_per_output_with_the_overlay_laid_over_it():
    spec = {"name": "t", "screen": {"anchor": "top", "height": 0.2}, "size": 0.05,
            "aspect": {"9x16": {"screen": {"height": 0.3}, "size": 0.04}}}
    got = dict(SR.per_output(spec, ["16x9", "9x16"]))
    assert list(got) == ["16x9", "9x16"]
    assert got["16x9"] == {"name": "t", "screen": {"anchor": "top", "height": 0.2}, "size": 0.05}
    assert got["9x16"] == {"name": "t", "screen": {"anchor": "top", "height": 0.3}, "size": 0.04}
    assert "aspect" in spec                                           # the entry itself is untouched


def test_aspect_without_screen_still_splits_per_output_and_unknown_outputs_are_errors():
    spec = {"name": "t", "aspect": {"9x16": {"size": 0.04}}}
    assert [n for n, _ in SR.per_output(spec, ["16x9", "9x16"])] == ["16x9", "9x16"]
    with pytest.raises(TextError, match="no output '1x1'"):
        SR.per_output({"name": "t", "aspect": {"1x1": {}}}, ["16x9"])
    with pytest.raises(TextError, match="table"):
        SR.per_output({"name": "t", "aspect": 3}, ["16x9"])
    with pytest.raises(TextError, match="table of overrides"):
        SR.per_output({"name": "t", "aspect": {"16x9": 3}}, ["16x9"])


# ------------------------------------------------------------------------------------------------------ screen spec
def test_screen_true_is_all_defaults_and_absent_or_false_is_not_screen_type():
    assert SR.screen_spec({}) is None and SR.screen_spec({"screen": False}) is None
    assert SR.screen_spec({"screen": True}) == {"anchor": None, "side": "center", "margin": SR.MARGIN, "height": None,
                                                "width": 1.0}


@pytest.mark.parametrize("screen, frag", [
    ({"anchr": "top"}, "unknown screen keys \\['anchr'\\]"),
    ({"anchor": "left"}, "anchor is one of"),
    ({"side": "top"}, "side is one of"),
    ({"margin": 0.5}, "margin"),
    ({"width": 0.0}, "width"),
    ({"height": 1.5}, "height"),
    (3, "true or a table"),
])
def test_screen_errors_name_the_key(screen, frag):
    with pytest.raises(TextError, match=frag):
        SR.screen_spec({"screen": screen})


# ------------------------------------------------------------------------------------------------------ the frame
def test_the_frame_is_w_over_h_frame_heights_wide_and_the_camera_that_frames_it_is_that_big_on_its_long_side():
    assert SR.frame_width(S169) == pytest.approx(16 / 9) and SR.frame_width(S916) == pytest.approx(9 / 16)
    assert SR.ortho_scale(S169) == pytest.approx(16 / 9)           # landscape: the width is the larger side
    assert SR.ortho_scale(S916) == 1.0                              # portrait: the height is
    assert SR.ortho_scale((1000, 1000)) == 1.0


def test_the_whole_frame_inside_the_margin_when_there_is_no_anchor():
    sc = SR.screen_spec({"screen": {"margin": 0.05}})
    (u, v), (w, h) = SR.panel(sc, S169)
    assert (u, v) == (0.0, 0.0)
    assert w == pytest.approx(16 / 9 - 0.1) and h == pytest.approx(0.9)
    (u, v), (w, h) = SR.panel(sc, S916)
    assert w == pytest.approx(9 / 16 - 0.1) and h == pytest.approx(0.9)


def test_a_band_sits_against_the_edge_inside_the_margin_and_follows_the_frame_width():
    top = SR.screen_spec({"screen": {"anchor": "top", "margin": 0.05, "height": 0.2}})
    (u, v), (w, h) = SR.panel(top, S169)
    assert (u, h) == (0.0, 0.2) and v == pytest.approx(0.5 - 0.05 - 0.1) and w == pytest.approx(16 / 9 - 0.1)
    (u, v), (w, h) = SR.panel(top, S916)
    assert v == pytest.approx(0.35) and w == pytest.approx(9 / 16 - 0.1)
    bottom = SR.screen_spec({"screen": {"anchor": "bottom", "margin": 0.05, "height": 0.2}})
    assert SR.panel(bottom, S169)[0][1] == pytest.approx(-0.35)
    centre = SR.screen_spec({"screen": {"anchor": "center", "height": 0.3}})
    assert SR.panel(centre, S169)[0] == (0.0, 0.0) and SR.panel(centre, S169)[1][1] == 0.3
    assert SR.panel(SR.screen_spec({"screen": {"anchor": "top"}}), S169)[1][1] == SR.BAND


def test_width_is_a_share_of_what_the_margins_leave_and_at_and_box_win_over_the_band():
    sc = SR.screen_spec({"screen": {"anchor": "top", "width": 0.5, "margin": 0.0, "height": 0.1}})
    assert SR.panel(sc, S169)[1][0] == pytest.approx(16 / 9 * 0.5)
    assert SR.panel(sc, S169, at=[0.2, -0.1], box=[0.5, 0.3]) == ((0.2, -0.1), (0.5, 0.3))
    assert SR.panel(sc, S169, box=[0.5, 0.3])[0] == (0.0, pytest.approx(0.45))       # box alone keeps the band's place


# ------------------------------------------------------------------------------------------------------ compositing
def test_composite_is_straight_alpha_over_and_leaves_its_inputs_alone():
    base = np.full((2, 2, 3), 0.5, np.float32)
    layer = np.zeros((2, 2, 4), np.float32)
    layer[0, 0] = (1.0, 0.0, 0.0, 1.0)             # opaque red
    layer[0, 1] = (0.0, 1.0, 0.0, 0.5)             # half green
    layer[1, 0] = (1.0, 1.0, 1.0, 0.0)             # nothing
    layer[1, 1] = (0.2, 0.4, 0.6, 2.0)             # alpha is clipped to 1
    keep_b, keep_l = base.copy(), layer.copy()
    out = SR.composite(base, layer)
    assert out.shape == (2, 2, 3) and out.dtype == np.float32
    assert out[0, 0].tolist() == [1.0, 0.0, 0.0]
    assert out[0, 1].tolist() == pytest.approx([0.25, 0.75, 0.25])
    assert out[1, 0].tolist() == [0.5, 0.5, 0.5]
    assert out[1, 1].tolist() == pytest.approx([0.2, 0.4, 0.6])
    assert np.array_equal(base, keep_b) and np.array_equal(layer, keep_l)


def test_composite_takes_the_rgb_of_a_base_that_has_an_alpha_channel_too():
    base = np.full((1, 1, 4), 0.25, np.float32)
    layer = np.zeros((1, 1, 4), np.float32)
    assert SR.composite(base, layer).shape == (1, 1, 3)


# ------------------------------------------------------------------------------------------------------ negative type
NH, NW = 8, 16


def backdrop():
    """A synthetic picture: a gradient left to right (a different slope in each channel, rising rows) with a sharp edge, a
    bright slab from column 10 whose rows 2..5 hold a hard step (white) in it. Float32 in 0..1, (NH, NW, 3)."""
    x = np.linspace(0.0, 1.0, NW, dtype=np.float32)[None, :, None]
    y = np.linspace(0.1, 0.4, NH, dtype=np.float32)[:, None, None]
    bg = np.concatenate([np.broadcast_to(x, (NH, NW, 1)), np.broadcast_to(1.0 - x, (NH, NW, 1)) * 0.5 + y,
                         np.broadcast_to(x * 0.3 + y, (NH, NW, 1))], axis=2).astype(np.float32)
    bg[2:6, 10:] = 1.0
    return bg


def glyphs():
    """A synthetic glyph's coverage (NH, NW): solid in a box that straddles the hard edge, an antialiased rim of 0.25 / 0.5 /
    0.75 round it, nothing elsewhere. Returned as the negative layer: RGBA with arbitrary colour, the alpha the coverage."""
    a = np.zeros((NH, NW), np.float32)
    a[2:6, 8:14] = 1.0
    a[1, 8:14], a[6, 8:14] = 0.25, 0.75
    a[2:6, 7], a[2:6, 14] = 0.5, 0.5
    layer = np.zeros((NH, NW, 4), np.float32)
    layer[..., :3] = (0.9, 0.1, 0.4)                 # the glyph's colour: not used
    layer[..., 3] = a
    return layer


def test_negative_type_inverts_the_covered_pixels_and_leaves_the_rest_bit_for_bit():
    bg, neg = backdrop(), glyphs()
    keep_b, keep_n = bg.copy(), neg.copy()
    out = SR.composite(bg, None, neg)
    a = neg[..., 3]
    assert out.shape == (NH, NW, 3) and out.dtype == np.float32
    assert np.array_equal(out[a == 1.0], 1.0 - bg[a == 1.0])                  # solid coverage: the photographic negative
    assert np.array_equal(out[a == 0.0], bg[a == 0.0])                        # outside the glyph: the same bits
    assert (a == 1.0).sum() == 24 and (a == 0.0).sum() > 80                   # (a real mix of both)
    assert np.array_equal(bg, keep_b) and np.array_equal(neg, keep_n)         # inputs untouched


def test_negative_type_follows_the_coverage_formula_where_it_is_partial():
    bg, neg = backdrop(), glyphs()
    a = neg[..., 3:4].astype(np.float64)
    want = bg.astype(np.float64) * (1.0 - a) + (1.0 - bg.astype(np.float64)) * a
    out = SR.composite(bg, None, neg)
    assert np.abs(out - want).max() < 1e-6
    rim = (a[..., 0] > 0) & (a[..., 0] < 1)
    assert rim.sum() == 4 * 2 + 6 + 6 and np.abs(out[rim] - bg[rim]).max() > 0.01       # it moved: not just the solid part
    mid = out[3, 7]                                                                     # a half-covered pixel of the gradient
    assert mid.tolist() == pytest.approx((0.5 * bg[3, 7] + 0.5 * (1.0 - bg[3, 7])).tolist(), abs=1e-6)
    assert mid.tolist() == pytest.approx([0.5, 0.5, 0.5], abs=1e-6)                     # half-way between a picture and its negative is grey


def test_the_edge_of_the_picture_inverts_on_both_sides_of_it():
    bg, neg = backdrop(), glyphs()
    out = SR.composite(bg, None, neg)
    assert bg[3, 9].tolist() != [1.0, 1.0, 1.0] and bg[3, 10].tolist() == [1.0, 1.0, 1.0]       # the hard step in the picture
    assert out[3, 10].tolist() == [0.0, 0.0, 0.0]                                                # white under the glyph: black
    assert out[3, 9].tolist() == pytest.approx((1.0 - bg[3, 9]).tolist())                        # and the gradient beside it
    assert out[3, 10].tolist() == [0.0, 0.0, 0.0] and out[3, 15].tolist() == [1.0, 1.0, 1.0]      # past the glyph: the white again


def test_negative_and_ordinary_type_coexist_ordinary_first_and_negative_over_it():
    bg, neg = backdrop(), glyphs()
    layer = np.zeros((NH, NW, 4), np.float32)
    layer[0:4, 0:10] = (0.0, 0.8, 0.0, 1.0)                    # an opaque green box, partly under the negative glyph
    layer[6:8, 0:6] = (1.0, 0.0, 0.0, 0.5)                      # half red, clear of it
    plain = SR.composite(bg, layer)
    out = SR.composite(bg, layer, neg)
    a = neg[..., 3]
    assert np.array_equal(out[a == 0.0], plain[a == 0.0])      # where the negative is absent: exactly today's composite
    assert np.array_equal(SR.composite(bg, layer, None), plain)
    assert out[2, 9].tolist() == pytest.approx((1.0 - plain[2, 9]).tolist(), abs=1e-7)   # negative over green: green's negative
    assert out[2, 9].tolist() == pytest.approx([1.0, 0.2, 1.0], abs=1e-6)
    assert out[3, 12].tolist() == [0.0, 0.0, 0.0]               # negative over the white slab, no ordinary type there
    assert out[0, 0].tolist() == pytest.approx([0.0, 0.8, 0.0])  # ordinary type clear of the glyph: untouched by it


def test_overlapping_negative_type_is_one_coverage_and_inverts_once():
    """Two negative words that overlap are drawn into one layer, as the glyphs of one word that overlap are: straight alpha
    laid over, `1 - (1 - a1)(1 - a2)`. Where they overlap the picture is inverted once (a negative of a negative would give
    the picture back, a hole in the type); a half of each makes three quarters."""
    bg = backdrop()
    first, second = np.zeros((NH, NW), np.float32), np.zeros((NH, NW), np.float32)
    first[1:5, 2:8], second[3:7, 5:11] = 1.0, 1.0                  # overlap: rows 3..4, columns 5..7
    first[7, 0], second[7, 0] = 0.5, 0.5
    cover = 1.0 - (1.0 - first) * (1.0 - second)
    layer = np.dstack([np.ones((NH, NW, 3), np.float32), cover])
    out = SR.composite(bg, None, layer)
    both = (first == 1.0) & (second == 1.0)
    assert both.sum() == 6 and np.array_equal(out[both], 1.0 - bg[both])
    assert np.array_equal(out[(first == 1.0) ^ (second == 1.0)], 1.0 - bg[(first == 1.0) ^ (second == 1.0)])
    assert out[7, 0].tolist() == pytest.approx((0.25 * bg[7, 0] + 0.75 * (1.0 - bg[7, 0])).tolist(), abs=1e-6)


def test_negative_type_clips_its_coverage_and_takes_a_base_with_an_alpha_channel():
    base = np.full((1, 2, 4), 0.25, np.float32)
    layer = np.zeros((1, 2, 4), np.float32)
    layer[0, 0, 3], layer[0, 1, 3] = 3.0, -1.0
    out = SR.composite(base, None, layer)
    assert out.shape == (1, 2, 3)
    assert out[0, 0].tolist() == [0.75, 0.75, 0.75] and out[0, 1].tolist() == [0.25, 0.25, 0.25]


@pytest.mark.parametrize("spec, frag", [
    ({"negative": "yes", "screen": True}, r"negative is true or false, not 'yes'"),
    ({"negative": 1, "screen": True}, r"negative is true or false, not 1"),
    ({"negative": True}, r"negative = true needs `screen`"),
    ({"negative": True, "screen": False}, r"negative = true needs `screen`"),
    ({"negative": True, "screen": True, "knockout": True}, r"negative and knockout both reverse type"),
    ({"negative": True, "screen": True, "backing": {"color": "x"}}, r"negative and backing cannot go together"),
    ({"negative": True, "screen": True, "outline": {"width": 0.1}}, r"negative and outline cannot go together"),
    ({"negative": True, "screen": True, "lyrics": {"backing": {"color": "x"}}},
     r"negative and lyrics.backing cannot go together"),
])
def test_negative_misuse_is_refused_naming_the_key(spec, frag):
    with pytest.raises(TextError, match=frag):
        SR.negative_spec(spec)


def test_negative_is_a_flag_of_screen_type_and_false_or_absent_is_ordinary():
    assert SR.negative_spec({"screen": True, "negative": True}) is True
    assert SR.negative_spec({"screen": {"anchor": "top"}, "negative": True, "knockout": False, "backing": False}) is True
    for spec in ({}, {"screen": True}, {"screen": True, "negative": False}, {"negative": False}, {"negative": None}):
        assert SR.negative_spec(spec) is False


def test_negative_is_checked_on_each_outputs_version_of_the_entry():
    spec = {"name": "t", "screen": True, "negative": True, "aspect": {"9x16": {"knockout": True}}}
    with pytest.raises(TextError, match="negative and knockout"):
        SR.per_output(spec, ["16x9", "9x16"])
    spec = {"name": "t", "screen": True, "aspect": {"9x16": {"negative": True}}}
    got = dict(SR.per_output(spec, ["16x9", "9x16"]))
    assert SR.negative_spec(got["9x16"]) is True and SR.negative_spec(got["16x9"]) is False
    with pytest.raises(TextError, match="negative = true needs `screen`"):
        SR.per_output({"name": "t", "negative": True}, ["16x9"])


def test_negative_layer_is_a_file_of_its_own_beside_the_screen_layer():
    assert SR.negative_rel(7) == "negative/00007.png" and SR.layer_rel(7) == "screen/00007.png"


# ------------------------------------------------------------------------------------------------------ extends
def test_an_entry_that_extends_another_is_it_with_its_own_keys_laid_over_it_and_a_base_is_not_built():
    base = {"name": "ly", "abstract": True, "font": "marker", "size": 0.05, "lyrics": {"style": "slide", "wow": {}, "line": 1},
            "aspect": {"9x16": {"size": 0.04}}}
    one = {"name": "ly2", "extends": "ly", "size": 0.06, "lyrics": {"line": 2}, "aspect": {"9x16": {"screen": {"height": 0.3}}}}
    got = SR.resolve_entries([base, one])
    assert [e["name"] for e in got] == ["ly2"]
    assert got[0] == {"name": "ly2", "font": "marker", "size": 0.06, "lyrics": {"style": "slide", "wow": {}, "line": 2},
                      "aspect": {"9x16": {"size": 0.04, "screen": {"height": 0.3}}}}
    assert base["abstract"] is True and "extends" in one                           # the inputs are left alone


def test_extends_chains_and_a_concrete_base_is_built_too_and_abstract_is_not_inherited():
    a = {"name": "a", "color": "love", "text": "x"}
    b = {"name": "b", "extends": "a", "size": 0.1}
    c = {"name": "c", "extends": "b", "color": "gold"}
    got = {e["name"]: e for e in SR.resolve_entries([a, b, c])}
    assert set(got) == {"a", "b", "c"} and got["c"] == {"name": "c", "color": "gold", "text": "x", "size": 0.1}
    base = {"name": "base", "abstract": True, "text": "x"}
    kid = {"name": "kid", "extends": "base"}
    assert "abstract" not in SR.resolve_entries([base, kid])[0]
    both = {"name": "mid", "extends": "base", "abstract": True}
    assert [e["name"] for e in SR.resolve_entries([base, both, {"name": "leaf", "extends": "mid"}])] == ["leaf"]


def test_order_does_not_matter_for_the_base():
    kid = {"name": "kid", "extends": "base", "size": 2}
    base = {"name": "base", "abstract": True, "size": 1, "color": "x"}
    assert SR.resolve_entries([kid, base]) == [{"name": "kid", "size": 2, "color": "x"}]


@pytest.mark.parametrize("entries, frag", [
    ([{"name": "a", "extends": "nope"}], "extends 'nope', which no \\[\\[text\\]\\] is named"),
    ([{"name": "a", "extends": "b"}, {"name": "b", "extends": "a"}], "extends loops"),
    ([{"name": "a", "extends": "a"}], "extends loops"),
    ([{"extends": "a"}, {"name": "a"}], "needs a name of its own"),
])
def test_extends_errors_name_the_entries(entries, frag):
    with pytest.raises(TextError, match=frag):
        SR.resolve_entries(entries)


def test_entries_without_extends_pass_through_unchanged():
    es = [{"name": "a", "text": "x"}, {"name": "b", "ink": {}}]
    assert SR.resolve_entries(es) == es


def test_side_puts_the_band_against_that_edge_inside_the_margin():
    left = SR.screen_spec({"screen": {"anchor": "top", "side": "left", "width": 0.5, "margin": 0.05, "height": 0.2}})
    (u, v), (w, h) = SR.panel(left, S169)
    assert w == pytest.approx((16 / 9 - 0.1) * 0.5) and u == pytest.approx(-(16 / 18 - 0.05 - w / 2))
    assert u - w / 2 == pytest.approx(-(16 / 18 - 0.05))                               # its left edge sits on the margin
    right = SR.screen_spec({"screen": {"anchor": "top", "side": "right", "width": 0.5, "margin": 0.05, "height": 0.2}})
    assert SR.panel(right, S169)[0][0] == pytest.approx(-u) and SR.panel(right, S169)[0][1] == v
    full = SR.screen_spec({"screen": {"anchor": "top", "side": "left", "margin": 0.05}})
    assert SR.panel(full, S169)[0][0] == pytest.approx(0.0)                             # a band as wide as the frame has no room to slide
    assert SR.panel(left, S916)[0][0] < 0.0
