"""Shot styles and lens shift, bpy-free (mkmmd.core.shotstyle): the crop arithmetic behind `shift`, colour and object
resolution, the normalised `silhouette` / `reflection` specs, the cut lookup and the numpy composition of a silhouette
frame. The Blender half (mkmmd.blender.styles, build/shots.py) is checked by rendering and looking (docs/design.md: Shots).
Fixtures use made-up object names and colours only."""
import io

import numpy as np
import pytest
from PIL import Image

from mkmmd.core import palette as PAL
from mkmmd.core import shotstyle as SS

MASTER, SQUARE = (1080, 1920), (1080, 1080)
DAWN = PAL.get("rose-pine-dawn")


# ---------------------------------------------------------------- lens shift
@pytest.mark.parametrize("square_y", [0, 420, 560, 600, 700, 840])
def test_crop_shift_is_the_briefs_formula(square_y):
    x, y = SS.crop_shift(MASTER, SQUARE, top=square_y)
    assert x == 0.0
    assert y == pytest.approx((960 - (square_y + 540)) / 1080)


def test_centred_crop_needs_no_shift_and_a_crop_below_the_centre_shifts_up():
    assert SS.crop_shift(MASTER, SQUARE, top=420) == (0.0, 0.0)
    assert SS.crop_shift(MASTER, SQUARE, top=700)[1] < 0.0          # the axis rides high in a crop that starts low


@pytest.mark.parametrize("lens", [22.0, 24.0, 62.0, 110.0])
@pytest.mark.parametrize("top", [0, 420, 560, 600, 700, 840])
def test_the_square_camera_is_an_exact_crop_of_the_master(lens, top):
    cam = SS.crop_camera(MASTER, SQUARE, top=top)
    rng = np.random.default_rng(top + int(lens))
    for _ in range(40):
        z = rng.uniform(0.4, 30.0)
        p = (rng.uniform(-0.6, 0.6) * z, rng.uniform(-1.0, 1.0) * z, -z)
        mx, my = SS.pinhole_px(p, lens, MASTER)
        cx, cy = SS.pinhole_px(p, lens * cam["lens_scale"], SQUARE, cam["shift"])
        assert cx == pytest.approx(mx, abs=1e-9)                    # same column
        assert cy == pytest.approx(my - top, abs=1e-9)             # rows move up by the crop's top


def test_a_crop_with_an_x_offset_is_exact_too():
    master, crop, top, left = (1080, 1920), (540, 540), 300, 200
    cam = SS.crop_camera(master, crop, top=top, left=left)
    assert cam["lens_scale"] == pytest.approx(1920 / 540)
    p = (0.31, -0.2, -2.0)
    mx, my = SS.pinhole_px(p, 50.0, master)
    cx, cy = SS.pinhole_px(p, 50.0 * cam["lens_scale"], crop, cam["shift"])
    assert (cx, cy) == pytest.approx((mx - left, my - top), abs=1e-9)


def test_lens_and_fstop_scale_keep_the_blur_in_master_pixels():
    """A defocus blur of f^2 / N * g millimetres (g depends on the distances only) is f^2 / N * g / 36 * max_side pixels."""
    cam = SS.crop_camera(MASTER, SQUARE, top=560)
    f, n, g = 62.0, 2.8, 0.0123
    master_px = f * f / n * g / 36.0 * max(MASTER)
    crop_px = (f * cam["lens_scale"]) ** 2 / (n * cam["fstop_scale"]) * g / 36.0 * max(SQUARE)
    assert crop_px == pytest.approx(master_px)
    assert cam["lens_scale"] == pytest.approx(1920 / 1080) and cam["fstop_scale"] == pytest.approx(1920 / 1080)


def test_blender_shift_convention_as_probed():
    """Probed in Blender 4.2.3 with world_to_camera_view on a square frame: shift_x = +0.25 puts the on-axis point at u = 0.25
    (left), shift_y = +0.25 at v = 0.25 counted from the bottom (so lower in the picture)."""
    assert SS.pinhole_px((0, 0, -10), 50.0, (1080, 1080), (0.25, 0.0)) == pytest.approx((270.0, 540.0))
    assert SS.pinhole_px((0, 0, -10), 50.0, (1080, 1080), (0.0, 0.25)) == pytest.approx((540.0, 810.0))
    assert SS.pinhole_px((0, 0, -10), 50.0, (1080, 1920), (0.0, 0.25))[1] == pytest.approx(960 + 0.25 * 1920)


def test_shift_pair_accepts_pairs_only():
    assert SS.shift_pair([0, -0.25]) == (0.0, -0.25)
    for bad in (0.25, [1], [1, 2, 3], "ab", None):
        with pytest.raises(SS.StyleError):
            SS.shift_pair(bad)


# ---------------------------------------------------------------- colours
def test_colours_from_slots_hex_mixes_and_triples():
    assert SS.resolve_colour("foam", DAWN) == pytest.approx(PAL.srgb("#56949f"))
    assert SS.resolve_colour("#eef2fb", DAWN) == pytest.approx(PAL.srgb("#eef2fb"))
    assert SS.resolve_colour([0.25, 0.5, 1.0], DAWN) == (0.25, 0.5, 1.0)
    mid = SS.resolve_colour("gold:iris:0.5", DAWN)
    a, b = PAL.srgb(DAWN["gold"]), PAL.srgb(DAWN["iris"])
    assert mid == pytest.approx([(x + y) / 2 for x, y in zip(a, b)], abs=1 / 255)


@pytest.mark.parametrize("bad", ["nonsense", "foam:iris", "foam:iris:2", "foam:iris:x", 12, [1, 2], [0.5, 0.5, 1.5], None])
def test_bad_colours_are_style_errors(bad):
    with pytest.raises(SS.StyleError):
        SS.resolve_colour(bad, DAWN)


def test_srgb_roundtrip():
    for c in (0.0, 0.02, 0.2, 0.5, 1.0):
        assert SS.linear_to_srgb(SS.srgb_to_linear(c)) == pytest.approx(c)
    assert SS.srgb_to_linear(0.5) == pytest.approx(0.21404, abs=1e-4)


# ---------------------------------------------------------------- selecting objects
OBJS = [SS.Obj("room_WallBack", ("room", "root")), SS.Obj("room_WallFront", ("room", "root")), SS.Obj("room_Floor", ("room",)),
        SS.Obj("tableA_body", ("props",)), SS.Obj("cordA", ("props",), frozenset({"earbud"})), SS.Obj("room_BoltSide0_core", ("sky",)),
        SS.Obj("zz_l1w3", ("Text",))]


def test_patterns_match_names_collections_and_properties():
    assert SS.select(OBJS, ["room_Wall*"]) == ["room_WallBack", "room_WallFront"]
    assert SS.select(OBJS, ["room_Floor", "tableA_body"]) == ["room_Floor", "tableA_body"]
    assert SS.select(OBJS, ["@room"]) == ["room_WallBack", "room_WallFront", "room_Floor"]
    assert SS.select(OBJS, ["prop:earbud"]) == ["cordA"]
    assert SS.select(OBJS, ["prop:missing", "nothing*"]) == []
    assert SS.select(OBJS, ["*_l?w*"]) == ["zz_l1w3"]
    assert SS.select(OBJS, []) == [] and SS.select(OBJS, None) == []


def test_name_patterns_are_case_sensitive_and_anchored():
    assert SS.select(OBJS, ["room_wall*"]) == []
    assert SS.select(OBJS, ["Wall*"]) == []
    assert SS.select(OBJS, ["*Wall*"]) == ["room_WallBack", "room_WallFront"]


# ---------------------------------------------------------------- normalised specs
SIL = {"name": "s", "style": "silhouette", "colors": {"background": "foam", "subject": "text", "accent": "surface"},
       "hide": ["room_Wall*"], "accent": ["prop:earbud", "room_Bolt*"],
       "tint": [{"object": "root", "prop": "flash", "color": "#eef2fb", "gain": 0.8}]}


def test_silhouette_spec_is_normalised_with_resolved_colours():
    out = SS.normalize(SIL, DAWN)
    sil = out["silhouette"]
    assert sil["colors"]["background"] == pytest.approx(PAL.srgb("#56949f"), abs=1e-5)
    assert sil["colors"]["subject"] == pytest.approx(PAL.srgb("#575279"), abs=1e-5)
    assert sil["hide"] == ["room_Wall*"] and sil["keep"] == [] and sil["knockout"] is None
    assert sil["tint"][0]["gain"] == 0.8 and sil["tint"][0]["glow"] is None
    assert sil["samples"] == 16 and sil["grow"] == 1.0
    assert "reflection" not in out


def test_silhouette_defaults_and_string_patterns():
    sil = SS.normalize({"name": "s", "style": "silhouette", "hide": "one*"}, DAWN)["silhouette"]
    assert sil["hide"] == ["one*"]
    assert sil["colors"]["background"] == pytest.approx(PAL.srgb(DAWN["base"]), abs=1e-5)
    assert sil["colors"]["accent"] == pytest.approx(PAL.srgb(DAWN["surface"]), abs=1e-5)


def test_glow_and_knockout_specs():
    spec = dict(SIL, tint=[{"object": "root", "prop": "sun", "color": "surface", "gain": 0.55,
                            "glow": {"at": [0.15, 0.25], "size": [0.9, 0.6]}}],
                knockout={"objects": ["zz_l*"], "color": "text"})
    sil = SS.normalize(spec, DAWN)["silhouette"]
    assert sil["tint"][0]["glow"] == {"at": [0.15, 0.25], "size": [0.9, 0.6]}
    assert sil["knockout"]["objects"] == ["zz_l*"]
    assert sil["knockout"]["color"] == pytest.approx(PAL.srgb("#575279"), abs=1e-5)


@pytest.mark.parametrize("patch", [
    {"style": "sepia"},
    {"colors": {"sky": "foam"}},
    {"colors": {"background": "nonsense"}},
    {"hide": [3]},
    {"samples": 0},
    {"grow": -1},
    {"knockout": {"color": "text"}},
    {"tint": [{"object": "c", "prop": "p"}]},
    {"tint": [{"object": "c", "prop": "p", "color": "foam", "extra": 1}]},
    {"tint": [{"object": "c", "prop": "p", "color": "foam", "glow": {"size": [0, 1]}}]},
    {"reflection": {"object": "g"}},
])
def test_bad_silhouette_specs_raise_style_errors(patch):
    with pytest.raises(SS.StyleError):
        SS.normalize(dict(SIL, **patch), DAWN)


def test_reflection_spec_defaults_and_validation():
    out = SS.normalize({"name": "r", "reflection": {"object": "glass"}}, DAWN)["reflection"]
    assert out["object"] == "glass" and out["strength"] == 0.5 and out["dim"] == 0.0 and out["roughness"] == 0.0
    assert out["hide"] == [] and out["only"] == [] and out["bend"] is False and out["world"] is False
    full = SS.normalize({"name": "r", "reflection": {"object": "g", "strength": 0.4, "hide": ["a*"], "only": ["b*"],
                                                    "bend": True, "world": True, "tint": "gold"}}, DAWN)["reflection"]
    assert full["strength"] == 0.4 and full["hide"] == ["a*"] and full["only"] == ["b*"] and full["bend"] and full["world"]
    assert full["tint"] == pytest.approx(PAL.srgb(DAWN["gold"]), abs=1e-5)
    for bad in ({"strength": 1.5}, {"dim": -0.1}, {"roughness": 2}, {"unknown": 1}):
        with pytest.raises(SS.StyleError):
            SS.normalize({"name": "r", "reflection": dict({"object": "g"}, **bad)}, DAWN)
    with pytest.raises(SS.StyleError):
        SS.normalize({"name": "r", "reflection": {"strength": 0.3}}, DAWN)


def test_a_shot_without_a_look_normalises_to_nothing_and_style_none_switches_one_off():
    assert SS.normalize({"name": "plain"}, DAWN) == {}
    assert SS.normalize(dict(SIL, style="none"), DAWN) == {}


# ---------------------------------------------------------------- the cut
TABLE = [{"name": "a", "from": 100, "to": 130}, {"name": "b", "from": 130, "to": 160}, {"name": "c", "from": 160, "to": 200}]


def test_shot_at_gives_a_cut_to_the_incoming_shot():
    assert SS.shot_at(TABLE, 100)["name"] == "a" and SS.shot_at(TABLE, 129)["name"] == "a"
    assert SS.shot_at(TABLE, 130)["name"] == "b" and SS.shot_at(TABLE, 199)["name"] == "c"
    assert SS.shot_at(TABLE, 5000)["name"] == "c"              # past the end: the last shot holds
    assert SS.shot_at(TABLE, 10)["name"] == "a"                # before the first cut: the first shot
    assert SS.shot_at(list(reversed(TABLE)), 140)["name"] == "b"
    assert SS.shot_at([], 1) is None


# ---------------------------------------------------------------- the background
FLASH = {"object": "root", "prop": "flash", "color": [0.9, 0.95, 1.0], "gain": 0.8, "glow": None}
SUN = {"object": "root", "prop": "sun", "color": [1.0, 1.0, 0.8], "gain": 0.55, "glow": {"at": [0.15, 0.25], "size": [0.9, 0.6]}}


def test_a_flat_layer_moves_the_background_by_gain_times_value_clipped_to_one():
    base = [0.2, 0.4, 0.6]
    assert SS.background_rgb(base, [FLASH], [0.0]) == pytest.approx(base)
    w = 0.8 * 0.5
    assert SS.background_rgb(base, [FLASH], [0.5]) == pytest.approx([a + (b - a) * w for a, b in zip(base, FLASH["color"])])
    assert SS.background_rgb(base, [FLASH], [3.0]) == pytest.approx(FLASH["color"])           # weight clips at one
    assert SS.background_rgb(base, [SUN], [1.0]) == pytest.approx(base)                       # glow layers are per pixel


def test_a_glow_layer_is_strongest_at_its_centre_and_follows_the_value():
    base = [0.0, 0.0, 0.0]
    img = SS.background_image((100, 200), base, [SUN], [1.0])
    assert img.shape == (200, 100, 3) and img.dtype == np.float32
    cy, cx = int(0.25 * 200), int(0.15 * 100)
    far = img[-1, -1]
    assert img[cy, cx].mean() > far.mean() > 0.0
    assert np.all(img[cy, cx] <= np.asarray(SUN["color"]) * 0.55 + 1e-3)                     # at most gain x colour
    dark = SS.background_image((100, 200), base, [SUN], [0.0])
    assert np.allclose(dark, 0.0)


# ---------------------------------------------------------------- image helpers
def test_dilate_grows_a_pixel_into_a_square():
    a = np.zeros((9, 9), np.float32)
    a[4, 4] = 1.0
    d = SS.dilate(a, 2)
    assert d.sum() == 25 and d[2:7, 2:7].all() and d[1, 4] == 0.0
    assert SS.dilate(a, 0) is a


def test_dilate_keeps_the_maximum_and_clips_at_the_border():
    a = np.zeros((5, 5), np.float32)
    a[0, 0], a[0, 1] = 0.5, 1.0
    d = SS.dilate(a, 1)
    assert d[0, 0] == 1.0 and d[1, 0] == 1.0 and d[1, 2] == 1.0 and d[2, 3] == 0.0 and d.shape == (5, 5)


def test_png_bytes_is_a_valid_8_bit_rgb_png():
    rng = np.random.default_rng(3)
    img = rng.integers(0, 256, (7, 11, 3), dtype=np.uint8)
    back = np.asarray(Image.open(io.BytesIO(SS.png_bytes(img))).convert("RGB"))
    assert back.shape == (7, 11, 3) and np.array_equal(back, img)


# ---------------------------------------------------------------- composing a silhouette frame
def _spec(**kw):
    s = dict(SIL, tint=[], **kw)
    return SS.normalize(s, DAWN)["silhouette"]


def _flat(h, w, subject_px=(), colour=None, alpha=SS.ALPHA_FULL):
    f = np.zeros((h, w, 4), np.float32)
    for y, x in subject_px:
        f[y, x, :3] = colour
        f[y, x, 3] = alpha
    return f


def test_background_and_subject_are_painted_flat():
    spec = _spec()
    bg, subj = np.asarray(spec["colors"]["background"], np.float32), np.asarray(spec["colors"]["subject"], np.float32)
    flat = _flat(6, 6, [(2, 2), (2, 3)], subj)
    out = SS.compose_silhouette({"flat": flat}, spec, [])
    assert np.allclose(out[0, 0], bg, atol=1e-6)
    assert np.allclose(out[2, 2], subj, atol=1e-6) and np.allclose(out[2, 3], subj, atol=1e-6)     # alpha 254 / 255 is full


def test_a_partly_covered_pixel_blends_background_and_subject():
    spec = _spec()
    bg, subj = np.asarray(spec["colors"]["background"], np.float32), np.asarray(spec["colors"]["subject"], np.float32)
    flat = _flat(4, 4, [(1, 1)], subj, alpha=0.5 * SS.ALPHA_FULL)
    out = SS.compose_silhouette({"flat": flat}, spec, [])
    assert np.allclose(out[1, 1], 0.5 * bg + 0.5 * subj, atol=1e-5)


def test_the_tint_follows_the_property_value():
    spec = SS.normalize(dict(SIL, tint=[{"object": "root", "prop": "flash", "color": "#ffffff", "gain": 1.0}]), DAWN)["silhouette"]
    flat = np.zeros((3, 3, 4), np.float32)
    bg = np.asarray(spec["colors"]["background"], np.float32)
    assert np.allclose(SS.compose_silhouette({"flat": flat}, spec, [0.0])[0, 0], bg, atol=1e-6)
    assert np.allclose(SS.compose_silhouette({"flat": flat}, spec, [0.25])[0, 0], bg + (1 - bg) * 0.25, atol=1e-6)
    assert np.allclose(SS.compose_silhouette({"flat": flat}, spec, [1.0])[0, 0], 1.0, atol=1e-6)


def test_a_bolt_paints_its_weight_in_the_accent_colour_and_the_rest_of_its_alpha_as_subject_glow():
    spec = _spec()
    bg = np.asarray(spec["colors"]["background"], np.float32)
    subj = np.asarray(spec["colors"]["subject"], np.float32)
    acc = np.asarray(spec["colors"]["accent"], np.float32)
    alpha, aov = np.zeros((4, 4), np.float32), np.zeros((4, 4), np.float32)
    alpha[1, 1], aov[1, 1] = 0.9, 0.4           # core: weight 0.4, 0.5 left over as glow
    alpha[2, 2], aov[2, 2] = 0.0, 0.0
    out = SS.compose_silhouette({"flat": np.zeros((4, 4, 4), np.float32), "soft_alpha": alpha, "soft_aov": aov}, spec, [])
    glow = 0.9 - 0.4
    want = bg + (subj - bg) * glow
    want = want + (acc - want) * 0.4
    assert np.allclose(out[1, 1], want, atol=1e-6)
    assert np.allclose(out[2, 2], bg, atol=1e-6)


def test_a_bolt_behind_the_subject_leaves_no_glow_over_it():
    spec = _spec()
    subj = np.asarray(spec["colors"]["subject"], np.float32)
    flat = _flat(3, 3, [(1, 1)], subj)
    alpha, aov = np.zeros((3, 3), np.float32), np.zeros((3, 3), np.float32)
    alpha[1, 1] = 0.7                                     # a stray alpha under the subject: capped by what the subject leaves
    out = SS.compose_silhouette({"flat": flat, "soft_alpha": alpha, "soft_aov": aov}, spec, [])
    assert np.allclose(out[1, 1], subj, atol=1e-6)


def test_cords_are_the_accent_colour_and_grow_by_the_wire_width():
    spec = SS.normalize(dict(SIL, tint=[], grow=135.0), DAWN)["silhouette"]       # 135 px at 1080 wide = 1 px at 8 wide
    subj, acc = np.asarray(spec["colors"]["subject"], np.float32), np.asarray(spec["colors"]["accent"], np.float32)
    flat = np.zeros((9, 8, 4), np.float32)
    flat[:, :4, :3], flat[:, :4, 3] = subj, SS.ALPHA_FULL            # a slab of subject on the left
    flat[4, 2, :3] = acc                                             # a one pixel cord lying on it
    out = SS.compose_silhouette({"flat": flat}, spec, [], hard=True)
    assert np.allclose(out[4, 2], acc, atol=1e-5)
    assert np.allclose(out[3, 1], acc, atol=1e-5) and np.allclose(out[5, 3], acc, atol=1e-5)    # grown by one pixel
    assert np.allclose(out[1, 1], subj, atol=1e-5)                                               # plain subject stays
    plain = SS.compose_silhouette({"flat": flat}, spec, [], hard=False)
    assert np.allclose(plain[3, 1], subj, atol=1e-5)


def test_workbench_colour_noise_is_not_taken_for_an_accent():
    spec = SS.normalize(dict(SIL, tint=[], grow=135.0), DAWN)["silhouette"]
    subj = np.asarray(spec["colors"]["subject"], np.float32)
    flat = np.zeros((4, 4, 4), np.float32)
    flat[..., :3], flat[..., 3] = subj + 1.0 / 255.0, SS.ALPHA_FULL
    out = SS.compose_silhouette({"flat": flat}, spec, [], hard=True)
    assert np.allclose(out, flat[..., :3], atol=1e-6)               # one level of 255 off, untouched


def test_type_is_painted_over_everything_with_its_own_alpha():
    spec = _spec()
    bg = np.asarray(spec["colors"]["background"], np.float32)
    t = np.zeros((3, 3, 4), np.float32)
    t[1, 1] = (1.0, 0.0, 0.0, 0.25)
    out = SS.compose_silhouette({"flat": np.zeros((3, 3, 4), np.float32), "type": t}, spec, [])
    assert np.allclose(out[1, 1], bg * 0.75 + np.array([0.25, 0.0, 0.0]), atol=1e-6)
    assert np.allclose(out[0, 0], bg, atol=1e-6)


def test_knockout_type_is_ink_on_the_background_and_background_over_the_silhouette():
    spec = SS.normalize(dict(SIL, tint=[], knockout={"objects": ["w*"], "color": "gold"}), DAWN)["silhouette"]
    bg = np.asarray(spec["colors"]["background"], np.float32)
    subj = np.asarray(spec["colors"]["subject"], np.float32)
    ink = np.asarray(spec["knockout"]["color"], np.float32)
    flat = _flat(3, 4, [(1, 2)], subj)
    k = np.zeros((3, 4, 4), np.float32)
    k[1, 1, 3] = 1.0                      # letter over the open background
    k[1, 2, 3] = 1.0                      # letter over the silhouette
    k[0, 3, 3] = 0.5                      # half a letter over the background
    out = SS.compose_silhouette({"flat": flat, "knock": k}, spec, [])
    assert np.allclose(out[1, 1], ink, atol=1e-6)
    assert np.allclose(out[1, 2], bg, atol=1e-6)
    assert np.allclose(out[0, 3], 0.5 * bg + 0.5 * ink, atol=1e-6)
    assert np.allclose(out[2, 2], bg, atol=1e-6)


def test_the_composition_is_float32_rgb_of_the_pass_size():
    spec = _spec()
    out = SS.compose_silhouette({"flat": np.zeros((5, 7, 4), np.float32)}, spec, [])
    assert out.shape == (5, 7, 3) and out.dtype == np.float32 and 0.0 <= out.min() and out.max() <= 1.0
