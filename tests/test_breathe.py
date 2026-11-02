"""The breathing palette's maths (mkmmd/core/breathe.py, docs/design.md: Lights and the look): `[look] breathe` checked, the
two palettes, the mix over clip time, the light factors, the build's driver expressions, and `Pair`, which mixes the same
thing resolved with each palette (post's per-frame colours)."""
import math

import numpy as np
import pytest

from mkmmd.core import breathe as BR
from mkmmd.core import palette as PAL
from mkmmd.core import shotstyle as SS

LOOK = {"palette": "rose-pine-dawn",
        "breathe": {"palette": "rose-pine-moon", "keys": [[0.0, 0.0], [1.6, 1.0], [3.0, 0.0]]}}


def look(**breathe):
    return {"palette": "rose-pine-dawn", "breathe": {"palette": "rose-pine-moon", "keys": [[0, 0], [1, 1]], **breathe}}


# ------------------------------------------------------------------------------------------------------------- spec
def test_no_breathe_is_none_and_both_palettes_are_the_projects():
    assert BR.spec({"palette": "rose-pine-moon"}) is None
    assert BR.spec({"breathe": False}) is None
    a, b = BR.palettes({"palette": "rose-pine-moon"})
    assert a == PAL.get("rose-pine-moon") and b is None


def test_spec_fills_in_the_defaults():
    sp = BR.spec(LOOK)
    assert sp["palette"] == "rose-pine-moon" and sp["ease"] == "smooth" and sp["slots"] == {}
    assert sp["keys"] == [[0.0, 0.0, "smooth"], [1.6, 1.0, "smooth"], [3.0, 0.0, "smooth"]]
    assert sp["lights"] == [["*", 1.0]]


@pytest.mark.parametrize("breathe, words", [
    ({"palette": "rose-pine-noon"}, "palette = 'rose-pine-noon'"),
    ({"keys": []}, "keys = [[t, mix]"),
    ({"keys": [[0, 1.5]]}, "mix = 1.5"),
    ({"keys": [[1, 0], [0.5, 1]]}, "comes before"),
    ({"keys": [[0, 0, "bounce"]]}, "ease = 'bounce'"),
    ({"ease": "cubic"}, "ease = 'cubic'"),
    ({"keys": [[0, "a"]]}, "expected a number"),
    ({"lights": {"sun": -1}}, "lights.sun = -1.0"),
    ({"slots": {"gold": "orange"}}, "slots.gold"),
    ({"slots": {"neon": "#ff00ff"}}, "slots neon: not slots of rose-pine-moon"),
    ({"tempo": 2}, "unknown key tempo"),
])
def test_spec_errors_say_what_is_wrong(breathe, words):
    with pytest.raises(BR.BreatheError) as e:
        BR.palettes(look(**breathe))
    assert words in str(e.value)


def test_the_projects_slots_stay_its_own_and_its_own_names_carry_over():
    lk = {"palette": "rose-pine-moon", "slots": {"gold": "#f2c46d", "neon": "#ff00ff"},
          "breathe": {"palette": "rose-pine-dawn", "keys": [[0, 0]], "slots": {"love": "#c0405f"}}}
    a, b = BR.palettes(lk)
    assert a["gold"] == "#f2c46d" and a["neon"] == "#ff00ff"
    assert b["gold"] == PAL.PALETTES["rose-pine-dawn"]["gold"]          # an override of a palette slot is the project's
    assert b["neon"] == "#ff00ff"                                       # a slot of its own carries over
    assert b["love"] == "#c0405f"                                       # the other palette's own override
    assert set(a) == set(b)


# -------------------------------------------------------------------------------------------------------------- mix
def test_mix_holds_outside_the_keys_and_eases_between():
    sp = BR.spec(LOOK)
    assert BR.mix_at(sp, -1.0) == 0.0 and BR.mix_at(sp, 9.0) == 0.0
    assert BR.mix_at(sp, 0.8) == pytest.approx(0.5)                     # smoothstep's middle
    assert BR.mix_at(sp, 0.4) == pytest.approx(0.15625)                 # 3u^2 - 2u^3 at u = 1/4
    assert BR.mix_at(sp, 1.6) == 1.0
    assert BR.mix_at(sp, 2.3) == pytest.approx(0.5)


def test_a_keys_own_ease_shapes_the_way_to_the_next_and_two_keys_at_one_time_jump():
    sp = BR.spec(look(keys=[[0, 0, "linear"], [1, 1, "hold"], [2, 0], [2, 1], [3, 0]]))
    assert BR.mix_at(sp, 0.25) == pytest.approx(0.25)                   # linear
    assert BR.mix_at(sp, 1.99) == 1.0                                   # held to the next key
    assert BR.mix_at(sp, 2.0) == 1.0                                    # the later of two keys at t = 2
    assert BR.mix_at(sp, 1.999999) == 1.0
    assert BR.mix_at(sp, 2.5) == pytest.approx(0.5)


def test_frame_mix_reads_clip_seconds_of_blender_frames():
    sp = BR.spec(LOOK)
    assert BR.frame_mix(sp, 30, 1, 1) == 0.0
    assert BR.frame_mix(sp, 30, 1, 25) == pytest.approx(0.5)            # t = 0.8
    assert BR.frame_mix(sp, 30, 1, 49) == 1.0                           # t = 1.6


def test_mixed_palette_is_each_slot_mixed_in_srgb():
    a, b = PAL.get("rose-pine-dawn"), PAL.get("rose-pine-moon")
    m = BR.mixed(a, b, 0.3)
    assert set(m) == set(a)
    assert m["love"] == PAL.mix(a["love"], b["love"], 0.3)
    assert BR.mixed(a, b, 0.0) == a and BR.mixed(a, b, 1.0) == b


def test_light_factor_is_the_first_pattern_that_matches():
    sp = BR.spec(look(lights={"sun*": 3.0, "lamp": 0.5, "*": 0.8}))
    assert BR.light_factor(sp, "sun_main") == 3.0
    assert BR.light_factor(sp, "lamp") == 0.5
    assert BR.light_factor(sp, "lamp2") == 0.8
    assert BR.light_factor(BR.spec(look(lights=2.0)), "anything") == 2.0
    assert BR.light_factor(BR.spec(look(lights={"key": 2.0})), "rim") == 1.0


# ------------------------------------------------------------------------------------------------- driver maths
def _eval(expr, m):
    """Blender's simple expressions as the drivers use them (pow, lerp)."""
    return eval(expr, {"pow": pow, "lerp": lambda a, b, t: a + (b - a) * t, "m": m})   # noqa: S307 - our own text


def test_colour_drivers_are_the_two_values_at_the_ends_and_mix_near_srgb_between():
    for a, b in ((0.8, 0.02), (0.0, 0.5), (3.0, 0.1)):
        e = BR.colour_expr(a, b)
        assert "**" not in e                                            # Blender's simple expressions have no **
        assert _eval(e, 0.0) == pytest.approx(a, abs=1e-7)
        assert _eval(e, 1.0) == pytest.approx(b, abs=1e-7)
        assert _eval(e, 0.5) == pytest.approx(BR.colour_at(a, b, 0.5), rel=1e-6)
    a, b = PAL.linear("#faf4ed")[0], PAL.linear("#232136")[0]          # Dawn's base and Moon's, red
    mid = SS.srgb_to_linear((SS.linear_to_srgb(a) + SS.linear_to_srgb(b)) / 2)   # the sRGB mix "a:b:0.5" makes
    assert BR.colour_at(a, b, 0.5) == pytest.approx(mid, rel=0.06)
    assert abs(BR.colour_at(a, b, 0.5) - (a + b) / 2) > 0.1             # not a linear mix: far brighter


def test_number_drivers_mix_linearly():
    e = BR.number_expr(2.0, 6.0)
    assert [_eval(e, m) for m in (0.0, 0.25, 1.0)] == [2.0, 3.0, 6.0]


# ------------------------------------------------------------------------------------------------------------- Pair
def test_pair_mixes_numbers_and_hex_colours_and_shares_the_rest():
    shared = {"font": "x", "size": [1, 2]}
    a = {"color": [0.2, 0.4, 0.6], "n": 3, "t": (1.0, "#000000"), "deep": [[0.0], [1.0]], "same": shared}
    b = {"color": [0.6, 0.4, 0.2], "n": 3, "t": (3.0, "#ffffff"), "deep": [[0.0], [3.0]], "same": dict(shared)}
    p = BR.Pair(a, b, "fx[0]")
    assert not p.static and len(p.diffs) == 5
    assert p.at(0.0) is a and p.at(1.0) is b
    h = p.at(0.5)
    assert h["color"] == pytest.approx([0.4, 0.4, 0.4])
    assert h["t"] == (2.0, "#808080") and isinstance(h["t"], tuple)
    assert h["deep"][1] == [2.0] and h["deep"][0] is a["deep"][0]       # only what differs is copied
    assert h["same"] is shared and h["n"] == 3
    assert a["color"] == [0.2, 0.4, 0.6]                                # `a` itself is untouched


def test_pair_of_equal_things_is_static():
    p = BR.Pair({"a": [1.0, "x"]}, {"a": [1.0, "x"]})
    assert p.static and p.at(0.5) is p.a


def test_pair_mixes_float_arrays():
    p = BR.Pair({"x": np.zeros(3, np.float32)}, {"x": np.ones(3, np.float32)})
    assert np.allclose(p.at(0.25)["x"], 0.25)


@pytest.mark.parametrize("a, b, words", [
    ({"k": "a"}, {"k": "b"}, "only numbers and colours"),
    ({"k": [1, 2]}, {"k": [1]}, "2 items with one palette"),
    ({"k": 1}, {"k": 2}, "a whole number cannot be mixed"),
    ({"k": True}, {"k": False}, "only numbers and colours"),
    ({"k": 1}, {"j": 1}, "only in one palette's"),
    ({"k": None}, {"k": 0.5}, "only numbers and colours"),
])
def test_pair_refuses_what_is_not_the_same_thing(a, b, words):
    with pytest.raises(BR.BreatheError) as e:
        BR.Pair(a, b, "fx[2]")
    assert words in str(e.value) and str(e.value).startswith("fx[2]['k']" if "k" in a and "k" in b else "fx[2]")


def test_pair_of_resolved_slots_is_the_mixed_palettes_colour():
    a, b = PAL.get("rose-pine-dawn"), PAL.get("rose-pine-moon")
    p = BR.Pair(list(SS.resolve_colour("love:gold:0.25", a)), list(SS.resolve_colour("love:gold:0.25", b)))
    want = SS.resolve_colour("love:gold:0.25", BR.mixed(a, b, 0.4))
    assert p.at(0.4) == pytest.approx(list(want), abs=2.0 / 255.0)      # hex rounding, twice
    assert math.isclose(p.at(0.4)[0], 0.6 * p.a[0] + 0.4 * p.b[0])
