"""One butterfly, bpy-free (mkmmd.core.butterfly, mkmmd.core.butterfly_pattern): the wing outlines and their pose, the meshes
at unit span, the flap maths and the single prop's driver expressions against it, the options, the colours and the
patterns. The Blender half (the drivers giving the same angles, the keys stage's rates, a hit lane) is
tests/test_butterfly_build.py."""
import math
import re

import numpy as np
import pytest

from mkmmd.core import butterfly as BF
from mkmmd.core import butterfly_pattern as BP
from mkmmd.core import palette as PAL
from mkmmd.core import shell as S


def _clamp(x, lo, hi):
    return max(lo, min(hi, x))


def _smoothstep(a, b, x):
    t = _clamp((x - a) / (b - a), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


BLENDER = {"radians": math.radians, "cos": math.cos, "smoothstep": _smoothstep, "max": max}


def _inside(p, poly):
    """Even-odd point in polygon."""
    x, y = p
    a, b = poly, np.roll(poly, -1, axis=0)
    hit = (a[:, 1] > y) != (b[:, 1] > y)
    xs = a[hit, 0] + (y - a[hit, 1]) * (b[hit, 0] - a[hit, 0]) / (b[hit, 1] - a[hit, 1])
    return int(np.sum(xs > x)) % 2 == 1


# ---------------------------------------------------------------------------------------------------- outlines
@pytest.mark.parametrize("style", BF.STYLES)
def test_outlines_are_simple_ccw_polygons_at_unit_span_rooted_on_the_body(style):
    reach = 0.0
    for wing in ("fore", "hind"):
        P = BF.outline(style, wing)
        assert S.polygon_area(P) > 0 and not S.section_crosses_itself(P)
        L = BF.landmarks(style)[wing]
        assert np.allclose(P[0], L[0]) and np.allclose(P[-2], L[-1])                  # the root's corners kept
        assert max(L[0][0], L[-1][0]) < 0.04                                          # the root sits on the thorax
        reach = max(reach, float(P[:, 0].max()))
    assert reach == pytest.approx(0.5, abs=2e-3)                                     # unit span (resampled curve)


@pytest.mark.parametrize("style", BF.STYLES)
def test_the_face_fill_covers_the_outline_exactly_tails_and_scallops_too(style):
    for wing in ("fore", "hind"):
        P = BF.outline(style, wing, 200)
        V, T = BF.fill(P)
        a = V[T]
        e1, e2 = a[:, 1] - a[:, 0], a[:, 2] - a[:, 0]
        area = 0.5 * (e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0])
        assert area.min() > 0 and area.sum() == pytest.approx(S.polygon_area(P), rel=1e-9)
        edges = {tuple(sorted(e)) for t in T for e in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0]))}
        m = len(P)
        assert all(tuple(sorted((i, (i + 1) % m))) in edges for i in range(m))      # the outline is kept whole


@pytest.mark.parametrize("style", BF.STYLES)
def test_the_forewing_s_hind_edge_is_swept_back_by_the_set_pose(style):
    m = BF._MEASURED[style]
    F = BF.landmarks(style)["fore"]
    root = 0.5 * (F[0] + F[-1])
    edge = F[m["tornus"]] - root
    assert math.degrees(math.atan2(edge[1], edge[0])) == pytest.approx(m["sweep"], abs=1e-6)


@pytest.mark.parametrize("style", BF.STYLES)
def test_the_hindwing_s_leading_edge_is_tucked_under_the_forewing(style):
    """No gap between the wings: the hindwing's front edge, from past the thorax to where it comes out from under the
    forewing, lies inside the forewing's outline or within a hair of its edge."""
    F, H = BF.outline(style, "fore"), BF.outline(style, "hind")
    s = BF.landmark_s(style, "hind")
    out0 = BP.ANCHORS[style]["hind"]["outer"][0]
    n = len(H) - 1
    hidden = [H[i] for i in range(n) if i / (n - 1) < s[out0] - 0.02 and H[i][0] > 0.1]       # past the thorax
    out = np.array([p for p in hidden if not _inside(p, F)]).reshape(-1, 2)
    gap = BF._edge_distance(out, F).max() if len(out) else 0.0
    assert hidden and gap < 0.02                          # at most a hairline where it comes out at the tornus


def test_only_the_swallowtail_has_tails_and_they_are_narrow():
    def tail(style):
        """How far the backmost point of the hindwing sticks out behind the outline 3-6 cm-units to either side of it."""
        H = BF.outline(style, "hind")
        tip = H[int(np.argmax(H[:, 1]))]
        side = H[(np.abs(H[:, 0] - tip[0]) > 0.03) & (np.abs(H[:, 0] - tip[0]) < 0.06)]
        return tip[1] - side[:, 1].max(), tip
    out, tip = tail("swallowtail")
    assert out > 0.04 and tip[0] > 0.12                   # a narrow tail, out from the body
    for style in ("monarch", "morpho", "blue"):
        assert tail(style)[0] < 0.02                      # round hindwings


def test_scallops_bite_into_the_outer_edge_by_their_depth():
    smooth = BF._resample(BF._open_curve(BF.landmarks("morpho")["hind"]), 239)
    waved = BF.outline("morpho", "hind")[:-1]
    bite = np.linalg.norm(waved - smooth, axis=1)
    depth = BF._SCALLOPS["morpho"]["hind"][0][3]
    assert bite.max() == pytest.approx(depth, rel=0.15) and np.median(bite) < 0.5 * depth
    assert np.allclose(BF.outline("blue", "hind")[:-1], BF._resample(BF._open_curve(BF.landmarks("blue")["hind"]), 239))


def test_an_unknown_style_is_refused():
    with pytest.raises(ValueError, match="no style"):
        BF.landmarks("moth")


# ---------------------------------------------------------------------------------------------------- meshes
@pytest.mark.parametrize("style", BF.STYLES)
def test_parts_are_closed_and_the_right_wing_mirrors_the_left(style):
    P = BF.parts(style)
    for k in ("body", "wing_L", "wing_R"):
        assert P[k].is_closed() and P[k].volume() > 0, k
    L, R = P["wing_L"], P["wing_R"]
    assert np.allclose(R.V, L.V * [-1.0, 1.0, 1.0])
    assert P["hinge_L"] == BF.hinge("L") and P["hinge_R"] == (-BF.HINGE_X, 0.0, BF.HINGE_Z)
    assert L.V[:, 0].min() > -0.01 and L.V[:, 0].max() == pytest.approx(0.5 - BF.HINGE_X, abs=0.003)
    B = P["body"]
    assert np.allclose(np.sort(B.V[:, 0]), np.sort(-B.V[:, 0]), atol=1e-9)        # the body is symmetric
    assert set(np.unique(np.concatenate([B.Qm, B.Tm]))) == {0, 1}                    # body and eyes


@pytest.mark.parametrize("style", BF.STYLES)
def test_the_upper_side_is_marked_and_the_forewing_lies_over_the_hindwing(style):
    m, dorsal = BF.wing_mesh(style, "L")
    top, bot = m.V[dorsal == 1.0], m.V[dorsal == 0.0]
    assert len(top) and len(bot) and set(np.unique(dorsal)) == {0.0, 0.5, 1.0}
    for wing in ("fore", "hind"):                         # each face's upper vertices lie over its underside's
        V, _Q, _T, _uv, d = BF._wing_surface(style, wing)
        assert V[d == 1.0][:, 2].mean() > V[d == 0.0][:, 2].mean()
    Vf, *_rest, df = BF._wing_surface(style, "fore")
    Vh, *_rest2, dh = BF._wing_surface(style, "hind")
    F = BF.outline(style, "fore")
    over = np.array([_inside(p, F) for p in Vh[:, :2]])
    hind_top = Vh[over & (dh == 1.0)]
    fore_bottom = np.interp(hind_top[:, 0], np.sort(Vf[df == 0.0][:, 0]),
                            np.sort(Vf[df == 0.0][:, 2]))   # z of the forewing's underside along x
    assert len(hind_top) and np.all(fore_bottom - hind_top[:, 2] > 0.001)          # no intersection where they overlap


@pytest.mark.parametrize("style", BF.STYLES)
def test_wing_uvs_land_in_their_tiles_of_the_atlas(style):
    for wing in ("fore", "hind"):
        _V, _Q, _T, uv, _d = BF._wing_surface(style, wing)
        assert uv.min() >= 0.0 and uv.max() <= 1.0 and uv[:, 1].min() >= 0.5           # the upper side's row
        assert (uv[:, 0].max() <= 0.5) if wing == "fore" else (uv[:, 0].min() >= 0.5)


def test_the_body_has_its_head_in_front_and_its_antennae_reach_forward():
    B = BF.body_mesh("monarch")
    eyes = B.V[np.unique(np.concatenate([B.T[B.Tm == 1].ravel(), B.Q[B.Qm == 1].ravel()]))]
    assert eyes[:, 1].mean() < -0.06 and B.V[:, 1].min() < -0.2 and B.V[:, 1].max() > 0.25
    assert B.V[:, 2].min() < -0.08                                                  # legs reach down to a perch


# ---------------------------------------------------------------------------------------------------- flap maths
def test_the_wing_sweeps_its_stroke_about_the_middle():
    ph = np.linspace(0.0, 1.0, 101)
    a = np.degrees(BF.wing_angle(ph, 100.0, 20.0))
    assert a.max() == pytest.approx(70.0) and a.min() == pytest.approx(-30.0)
    assert a[0] == pytest.approx(70.0) and a[50] == pytest.approx(-30.0)             # phase 0 up, 0.5 down


def test_a_hit_adds_exactly_one_beat_smoothly():
    u = np.linspace(-1.0, 2.0, 301)
    h = BF.hit_flap(u)
    assert h[u <= 0].max() == 0.0 and h[u >= 1].min() == 1.0
    assert np.all(np.diff(h) >= 0) and np.abs(np.diff(h)).max() < 0.02             # monotone, no jump
    P = {k: v[0] for k, v in BF.FLAP_PARAMS.items()}
    P.update(hit=2.0, hit_time=0.3, phase=0.25)
    before, after = BF.flap_phase(0.0, P, 1.99), BF.flap_phase(0.0, P, 2.31)
    assert after - before == pytest.approx(1.0)                                    # the same pose, one beat later
    mid = BF.wing_angle(BF.flap_phase(0.0, P, 2.08), 100.0, 20.0)
    assert mid != pytest.approx(BF.wing_angle(before, 100.0, 20.0))                 # it moved in between


def test_the_body_rises_on_the_down_stroke():
    assert BF.body_bob(0.5) == pytest.approx(0.5 * BF.BOB) and BF.body_bob(0.0) == pytest.approx(-0.5 * BF.BOB)


class _Ops:
    """A stand-in backend: the flap maths must use only `cos`, `smoothstep` and arithmetic on what it is given."""

    @staticmethod
    def cos(x):
        return math.cos(x)

    @staticmethod
    def smoothstep(a, b, x):
        return _smoothstep(a, b, x)


def test_the_flap_maths_runs_on_another_backend():
    assert BF.wing_angle(0.3, 90.0, 10.0, _Ops()) == pytest.approx(float(BF.wing_angle(0.3, 90.0, 10.0)))
    assert BF.hit_flap(0.4, _Ops()) == pytest.approx(float(BF.hit_flap(0.4)))
    assert BF.body_bob(0.7, _Ops()) == pytest.approx(float(BF.body_bob(0.7)))


@pytest.mark.parametrize("P", [{"beats": 0.0, "phase": 0.0, "hit": -1000.0, "hit_time": 0.3, "flap_mid": 20.0,
                                "flap_amp": 100.0},
                               {"beats": 3.7, "phase": 0.25, "hit": 1.2, "hit_time": 0.25, "flap_mid": 35.0,
                                "flap_amp": 60.0}])
def test_driver_expressions_equal_the_numpy_reference(P):
    f0, fps, size = 91.0, 30.0, 0.09
    ex = BF.expressions(f0, fps, size)
    for frame in (60, 91, 100, 126, 127, 128, 130, 135, 160):
        env = dict(BLENDER, frame=frame, **P)
        t = (frame - f0) / fps
        ph = BF.flap_phase(P["beats"], P, t)
        want = float(BF.wing_angle(ph, P["flap_amp"], P["flap_mid"]))
        assert eval(ex["wing_R"], {"__builtins__": {}}, env) == pytest.approx(want, abs=1e-6)
        assert eval(ex["wing_L"], {"__builtins__": {}}, env) == pytest.approx(-want, abs=1e-6)
        assert eval(ex["bob"], {"__builtins__": {}}, env) == pytest.approx(
            float(BF.body_bob(ph)) * size, abs=1e-8)


def test_driver_expressions_are_blender_simple_expressions():
    ok = set(BF.NAMES) | BF.FUNCS | {"frame"}
    for text in BF.expressions(91.0, 30.0, 0.09).values():
        assert len(text) <= 255 and "**" not in text and "%" not in text
        assert set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", text)) <= ok


# ---------------------------------------------------------------------------------------------------- options, card
@pytest.mark.parametrize("slots, frag", [({"style": "moth"}, "style 'moth'"), ({"size": 0}, "size is the wingspan"),
                                         ({"way": 9}, "colourway"), ({"sheen": -1}, "sheen"), ({"tile": 10}, "tile"),
                                         ({"flap": -2}, "flap is wing beats"), ({"hit_time": 0}, "hit_time"),
                                         ({"glow": -0.1}, "glow"), ({"flap_amp": "big"}, "flap_amp is a number")])
def test_bad_options_are_refused_saying_what_to_change(slots, frag):
    with pytest.raises(ValueError, match=frag):
        BF.options(slots)


def test_options_take_the_starting_values_of_the_live_properties():
    O, P = BF.options({"style": "blue", "flap": 5, "phase": 0.25, "size": 0.12})
    assert O["style"] == "blue" and O["size"] == 0.12 and O["sheen"] == BF.SHEEN["blue"] and O["glow"] == BF.GLOW
    assert P["flap"] == 5.0 and P["phase"] == 0.25 and P["flap_amp"] == BF.FLAP_PARAMS["flap_amp"][0]


def test_the_card_points_scale_with_the_size_and_bound_the_stroke():
    P = BF.options({})[1]
    U = BF.use_points("monarch", 0.18)
    tip = next(u for u in U["look"] if u["name"] == "tip_L")["point"]
    assert tip[0] == pytest.approx(0.5 * 0.18, abs=0.003)
    b = BF.bounds("monarch", 0.09, P)
    up = 0.09 * (BF.HINGE_Z + 0.488 * math.sin(math.radians(70.0)))
    assert b["max"][2] >= up - 0.002 and b["min"][0] == pytest.approx(-0.045)


# ---------------------------------------------------------------------------------------------------- colours
def test_dark_and_light_stand_for_the_palette_s_ends():
    for name in ("rose-pine-dawn", "rose-pine-moon"):
        pal = PAL.get(name)
        dark, light = BF.colour("dark", pal), BF.colour("light", pal)
        lum = [BF._luma(PAL.linear(v)) for v in pal.values()]
        assert BF._luma(dark) < 0.5 * min(lum) and BF._luma(light) == pytest.approx(max(lum))


def test_roles_blend_in_linear_light_and_slots_override_them():
    mid = BF.colour("gold:1,love:3")
    g, lv = PAL.linear(BF.DAWN["gold"]), PAL.linear(BF.DAWN["love"])
    assert mid == pytest.approx(tuple(0.25 * a + 0.75 * b for a, b in zip(g, lv)))
    assert BF.roles("monarch", {"ground": "#123456"})["ground"] == "#123456"
    assert BF.roles("monarch", {}, 1)["ground"] == BF.WAYS["monarch"][1]["ground"]
    moon = BF.colour("gold", PAL.get("rose-pine-moon"))
    assert moon != pytest.approx(BF.colour("gold"))                                  # the project's palette recolours
    with pytest.raises(ValueError, match="colourways"):
        BF.roles("monarch", {}, 7)


# ---------------------------------------------------------------------------------------------------- patterns
@pytest.fixture(scope="module")
def atlases():
    return {s: BP.masks(s, 128) for s in BF.STYLES}


def test_masks_are_deterministic_and_in_range(atlases):
    M = atlases["morpho"]
    assert M.shape == (256, 256, 4) and M.dtype == np.float32 and 0.0 <= M.min() and M.max() <= 1.0
    assert np.array_equal(M, BP.masks("morpho", 128))


def _tile(M, style, wing, under):
    t = M.shape[0] // 2
    r, c = (1 if under else 0), (0 if wing == "fore" else 1)
    return M[r * t:(r + 1) * t, c * t:(c + 1) * t]


def _px(style, wing, xy, t=128):
    x0, y0, size = BF.atlas(style)[wing]
    return int((xy[1] - y0) / size * t), int((xy[0] - x0) / size * t)


def test_a_monarch_has_black_borders_with_white_dots_and_black_veins(atlases):
    T = BP.Tile("monarch", "fore", 128)
    BP.PAINTERS["monarch"](T, False)
    ink, spot = T.ch["ink"], T.ch["spot"]
    border = (T.d < 0.02) & T.inside & (T.s > 0.6) & (T.s < 0.8)
    middle = T.inside & (T.d > 0.08)
    assert ink[border].mean() > 0.6 and spot[border].max() > 0.9                    # dots in the black border
    assert 0.05 < ink[middle].mean() < 0.6                                         # veins over orange, not a blot


def test_a_morpho_shines_inside_its_black_border_and_has_eyes_underneath(atlases):
    up = _tile(atlases["morpho"], "morpho", "hind", False)
    T = BP.Tile("morpho", "hind", 128)
    mid = T.inside & (T.d >= BP.BAND - 1e-9)
    assert up[..., 3][mid].mean() > 0.9                                            # the sheen
    edge = T.inside & (T.d < 0.03) & (T.s > 0.4) & (T.s < 0.8)
    assert up[..., 0][edge].mean() > 0.7                                           # the black border over it
    under = _tile(atlases["morpho"], "morpho", "hind", True)
    assert under[..., 2].max() > 0.9 and under[..., 2].sum() > 3.0 * up[..., 2].sum()   # gold-ringed eyes below


def test_painting_a_channel_clears_the_ones_above_it():
    T = BP.Tile("blue", "fore", 64)
    full = np.ones((64, 64), np.float32)
    T.paint("ink", full)
    T.paint("spot", full)
    T.paint("accent", full * 0.5)
    assert T.ch["spot"].max() == pytest.approx(0.5) and T.ch["ink"].max() == pytest.approx(0.5)
    T.paint("sheen", full)
    assert T.ch["accent"].max() == 0.0 and T.ch["ink"].max() == 0.0


def test_the_swallowtail_s_tail_is_black_and_the_blue_has_ringed_dots_underneath(atlases):
    st = _tile(atlases["swallowtail"], "swallowtail", "hind", False)
    H = BF.outline("swallowtail", "hind")
    tip = H[int(np.argmax(H[:, 1]))]
    r, c = _px("swallowtail", "hind", tip + np.array([0.0, -0.012]))
    assert st[r, c, 0] > 0.8
    T = BP.Tile("blue", "hind", 256)
    BP.PAINTERS["blue"](T, True)
    assert T.ch["ink"][T.inside].max() > 0.95 and T.ch["spot"][T.inside].max() > 0.95
    assert T.ch["accent"][T.inside].max() > 0.9                                    # the orange lunules


def test_cached_masks_are_kept_and_read_back(tmp_path):
    a = BP.cached_masks("blue", 64, str(tmp_path))
    files = list(tmp_path.glob("blue_64_*.npy"))
    assert len(files) == 1 and np.array_equal(np.load(files[0]), a)
    assert np.array_equal(BP.cached_masks("blue", 64, str(tmp_path)), a)
    assert BP.masks_key("blue", 64) != BP.masks_key("blue", 128) != BP.masks_key("monarch", 128)
