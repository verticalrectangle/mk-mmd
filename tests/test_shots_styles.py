"""Shot styles and lens shift, bpy-free (mkmmd.core.shotstyle): the crop arithmetic behind `shift`, colour and object
resolution, the normalised `silhouette` / `vector` / `reflection` specs, the cut lookup and the numpy composition of a
silhouette and of a vector frame. The Blender half (mkmmd.blender.styles, build/shots.py) is checked by rendering and
looking (docs/design.md: Shots).
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


def test_reflection_false_switches_an_inherited_reflection_off_for_one_aspect():
    shot = {"name": "r", "reflection": {"object": "glass"}, "aspect": {"1x1": {"reflection": False}}}
    assert "reflection" in SS.normalize(shot, DAWN)
    assert SS.normalize(dict(shot, **shot["aspect"]["1x1"]), DAWN) == {}                   # the aspect's keys laid over the shot
    only_here = SS.normalize(dict(SIL, style="none", reflection={"object": "glass"}), DAWN)
    assert set(only_here) == {"reflection"}                                                # a silhouette shot can show one
    assert SS.normalize(dict(SIL, reflection=False), DAWN).keys() == {"silhouette"}        # false next to a look is no conflict
    with pytest.raises(SS.StyleError):                                                     # true is not a reflection
        SS.normalize({"name": "r", "reflection": True}, DAWN)


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


# ---------------------------------------------------------------- the vector look
VEC = {"colors": {"background": "#FFD21F", "line": "#1B2A6B", "inner": "#0000FF"},
       "tones": {"white": {"lit": "#FFFFFF", "shade": "#8FB4F0"}, "dark": "#1B2A6B",
                 "eyes": {"colors": ["#000000", "#8FB4F0", "#FFFFFF"], "at": [0.4, 0.8], "grey": 0.2}},
       "materials": [{"match": ["skin", "face"], "tone": "white", "group": "skin"}, {"match": "Yellow*", "tone": "dark"},
                     {"match": "eye", "tone": "eyes", "group": "skin"}]}


def _vec(shot=None, project=VEC):
    return SS.normalize({"name": "v", "style": "vector", **(shot or {})}, DAWN, project)["vector"]


def _rgb(h):
    return np.asarray(SS.resolve_colour(h, DAWN), np.float32)


def test_a_vector_shot_lays_its_colours_and_tones_over_the_projects():
    v = _vec({"colors": {"line": "#FFD21F", "background": "#1B2A6B"}, "tones": {"dark": "#FFD21F"}})
    assert np.allclose(v["colors"]["background"], _rgb("#1B2A6B")) and np.allclose(v["colors"]["line"], _rgb("#FFD21F"))
    assert np.allclose(v["colors"]["inner"], _rgb("#0000FF"))                 # the project's own inner stays
    assert np.allclose(v["tones"]["dark"]["lit"], _rgb("#FFD21F")) and v["tones"]["white"]["kind"] == "lit"
    assert v["default"] == "white"                                            # no `fill` tone: the first one
    plain = _vec(project={"colors": {"line": "#1B2A6B"}})
    assert np.allclose(plain["colors"]["inner"], plain["colors"]["line"])    # inner defaults to the line colour
    assert plain["default"] == "fill" and plain["materials"] == []


@pytest.mark.parametrize("shot, project, frag", [
    ({}, dict(VEC, outlines=2), "unknown keys"),
    ({}, dict(VEC, tones={"t": {"lit": "#fff", "grey": 0.2}}), "unknown keys"),
    ({}, dict(VEC, tones={"t": {"colors": ["#000", "#fff"], "at": [0.5, 0.6]}}), "n - 1 brightness steps"),
    ({}, dict(VEC, tones={"t": {"colors": ["#000", "#888", "#fff"], "at": [0.6, 0.5]}}), "increasing"),
    ({}, dict(VEC, materials=[{"match": "skin", "tone": "nope"}]), "not one of the tones"),
    ({}, dict(VEC, materials=[{"tone": "white"}]), "expected"),
    ({}, dict(VEC, lines={"outline": -1}), "negative"),
    ({}, dict(VEC, shadow=1.5), "0..1"),
    ({}, dict(VEC, light=[0, 0]), "light"),
    ({"grow": 2}, VEC, "silhouette keys"),
    ({"colors": {"subject": "#fff"}}, VEC, "expected a table with background, line, inner")])
def test_bad_vector_specs_are_style_errors(shot, project, frag):
    with pytest.raises(SS.StyleError, match=frag):
        _vec(shot, project)


def test_tones_belong_to_the_vector_style_only():
    with pytest.raises(SS.StyleError, match="tones belong to the vector style"):
        SS.normalize(dict(SIL, tones={"dark": "#000"}), DAWN)
    with pytest.raises(SS.StyleError, match="tones belong to the vector style"):
        SS.normalize({"name": "lit", "tones": {"dark": "#000"}}, DAWN)


def test_the_material_table_takes_the_first_rule_and_ignores_blenders_suffix():
    tones, groups = SS.vector_table(["skin", "face.001", "Yellow2", "hair", "Yellow2.001", "eye"], _vec())
    assert tones == [None, "white", "white", "dark", "white", "dark", "eyes"]   # hair: no rule, the default tone
    assert groups[1] == groups[2] == groups[6]                                 # one group: no line between them
    assert groups[3] == groups[5] != groups[1]                                 # no group: the material's own name
    assert groups[4] not in (groups[1], groups[3]) and groups[0] == 0


def _passes(ids, depth=3.0, shade=0.4, tex=None):
    ids = np.asarray(ids, np.int32)
    p = {"id": ids, "depth": np.broadcast_to(np.asarray(depth, np.float64), ids.shape).copy(),
         "shade": np.broadcast_to(np.asarray(shade, np.float32), ids.shape).copy()}
    if tex is not None:
        p["tex"] = tex
    return p


def _compose(p, mats, **kw):
    spec = _vec(**kw)
    return SS.compose_vector(p, spec, SS.vector_table(mats, spec), scale=1), spec


def test_a_lit_tone_takes_its_shade_colour_under_the_shadow_threshold():
    ids = np.zeros((40, 60), np.int32)
    ids[5:35, 5:55] = 1
    shade = np.full(ids.shape, 0.4, np.float32)
    shade[:, 30:] = 0.3 * SS.LIGHT_FULL - 0.01                                # just under the look's default shadow
    out, spec = _compose(_passes(ids, shade=shade), ["skin"])
    assert np.allclose(out[20, 15], _rgb("#FFFFFF")) and np.allclose(out[20, 45], _rgb("#8FB4F0"))
    assert np.allclose(out[1, 1], _rgb("#FFD21F"))                              # the background, flat


def test_a_one_pixel_lit_sliver_in_a_shadow_is_taken_out():
    ids = np.zeros((40, 60), np.int32)
    ids[5:35, 5:55] = 1
    shade = np.zeros(ids.shape, np.float32)
    shade[:, 30] = 0.4
    out, _ = _compose(_passes(ids, shade=shade), ["skin"])
    assert np.allclose(out[10:30, 10:50], _rgb("#8FB4F0"))


def test_a_drawn_tone_picks_its_colour_by_the_textures_brightness_and_greys_take_the_last():
    ids = np.zeros((40, 80), np.int32)
    ids[5:35, 5:75] = 1
    tex = np.zeros((40, 80, 3), np.float32)
    tex[:, :20] = 0.2                                                         # dark: the first colour
    tex[:, 20:40] = 0.6                                                       # the middle band
    tex[:, 40:60] = 0.95                                                      # bright and grey: the last colour
    tex[:, 60:] = [1.0, 0.92, 0.75]                                           # bright but tinted (luma 0.92): one down
    out, _ = _compose(_passes(ids, tex=tex), ["eye"])
    assert np.allclose(out[20, 12], _rgb("#000000")) and np.allclose(out[20, 30], _rgb("#8FB4F0"))
    assert np.allclose(out[20, 50], _rgb("#FFFFFF")) and np.allclose(out[20, 68], _rgb("#8FB4F0"))


def _two_blocks(mats, line=10.0):
    ids = np.zeros((216, 216), np.int32)
    ids[20:196, 20:108], ids[20:196, 108:196] = 1, 2
    return _compose(_passes(ids), mats, project=dict(VEC, lines={"outline": 30.0, "inner": line}))


def test_an_inner_line_runs_where_two_groups_meet_and_none_inside_one_group():
    out, spec = _two_blocks(["skin", "Yellow2"])
    inner = _rgb("#0000FF")
    assert np.allclose(out[100, 107], inner) and np.allclose(out[100, 60], _rgb("#FFFFFF"))
    out, _ = _two_blocks(["skin", "face"])                                    # one group: no line between them
    assert not np.isclose(out[30:186, 30:186], inner).all(-1).any()


def test_the_outline_is_drawn_round_the_figure_in_the_line_colour():
    out, _ = _two_blocks(["skin", "Yellow2"])
    line = _rgb("#1B2A6B")
    assert np.allclose(out[100, 20], line) and np.allclose(out[100, 18], line) and np.allclose(out[100, 22], line)
    assert np.allclose(out[100, 10], _rgb("#FFD21F"))


def test_a_depth_step_draws_a_line_and_a_surface_turning_away_does_not():
    ids = np.zeros((216, 216), np.int32)
    ids[20:196, 20:196] = 1
    inner = _rgb("#0000FF")
    step = np.full(ids.shape, 3.0)
    step[:, 108:] = 2.7                                                       # a part 30 cm in front of the rest
    out, _ = _compose(_passes(ids, depth=step), ["skin"])
    assert np.allclose(out[100, 107], inner) or np.allclose(out[100, 108], inner)
    slope = np.broadcast_to(np.linspace(1.0, 5.3, 216), ids.shape)            # 2 cm a pixel: steep, but smooth
    out, _ = _compose(_passes(ids, depth=slope), ["skin"])
    assert not np.isclose(out[30:186, 30:186], inner).all(-1).any()


def test_without_the_subject_the_vector_frame_is_its_background_at_the_frames_size():
    ids = np.zeros((40, 60), np.int32)
    ids[5:35, 5:55] = 1
    spec = _vec()
    out = SS.compose_vector(_passes(ids), spec, SS.vector_table(["skin"], spec), scale=2, subject=False)
    assert out.shape == (20, 30, 3) and out.dtype == np.float32
    assert np.allclose(out, _rgb("#FFD21F"))


OPP = dict(VEC, opposite={"colors": {"background": "#1B2A6B", "line": "#FFD21F"}, "tones": {"dark": "#FFD21F"}})


def test_the_opposite_palette_lays_its_colours_over_the_looks_and_keeps_the_rest():
    v = _vec(project=dict(OPP, opposite=dict(OPP["opposite"], tones={"dark": "#FFD21F",
                                                                  "eyes": {"colors": ["#FFFFFF", "#000000", "#8FB4F0"]}})))
    o = v["opposite"]
    assert np.allclose(o["colors"]["background"], _rgb("#1B2A6B")) and np.allclose(o["colors"]["line"], _rgb("#FFD21F"))
    assert np.allclose(o["colors"]["inner"], _rgb("#0000FF"))                # not given: the look's own inner
    assert np.allclose(o["tones"]["dark"]["lit"], _rgb("#FFD21F")) and o["tones"]["white"] == v["tones"]["white"]
    assert o["tones"]["eyes"]["at"] == v["tones"]["eyes"]["at"] and o["tones"]["eyes"]["grey"] == 0.2
    assert _vec()["opposite"] is None


@pytest.mark.parametrize("opp, frag", [
    ({"tones": {"nope": "#FFFFFF"}}, "not one of the tones"),
    ({"tones": {"white": "#FFFFFF"}}, "keeps its kind"),
    ({"tones": {"eyes": {"colors": ["#000000", "#FFFFFF"]}}}, "keeps its kind"),
    ({"colours": {}}, "expected {colors, tones}")])
def test_bad_opposite_palettes_are_refused(opp, frag):
    with pytest.raises(SS.StyleError, match=frag):
        _vec(project=dict(VEC, opposite=opp))


def test_flipped_pixels_take_the_opposite_palette_and_only_they():
    ids = np.zeros((40, 80), np.int32)
    ids[5:35, 5:75] = 1
    flip = np.zeros(ids.shape, bool)
    flip[:, 40:] = True
    spec = _vec(project=OPP)
    out = SS.compose_vector(_passes(ids), spec, SS.vector_table(["Yellow1"], spec), scale=1, flip=flip)
    assert np.allclose(out[20, 20], _rgb("#1B2A6B")) and np.allclose(out[20, 60], _rgb("#FFD21F"))    # the dark tone
    assert np.allclose(out[1, 20], _rgb("#FFD21F")) and np.allclose(out[1, 60], _rgb("#1B2A6B"))      # the background
    with pytest.raises(ValueError, match="opposite"):
        SS.compose_vector(_passes(ids), _vec(), SS.vector_table(["Yellow1"], _vec()), scale=1, flip=flip)
