"""Type effects (mkmmd/core/typefx.py): validation of `kinetic`, `backing` and `outline` and the size of a strip of tape. The
errors say what is wrong, never what the text says."""
import pytest

from mkmmd.core import typefx as FX
from mkmmd.core.typeset import TextError


def test_kinetic_keys_come_back_sorted_and_numeric():
    k = FX.kinetic_spec({"kinetic": {"show": [1, 2], "scale": [[0.2, 1], [0.1, 1.3]], "rot": [[0, 3]], "pivot": "bottom",
                                     "tint": {"color": "foam", "keys": [[1, 0], [0, 0.5]]}, "headroom": 0.2,
                                     "drip": {"t": 1.5}}})
    assert k["show"] == [1.0, 2.0] and k["scale"] == [(0.1, 1.3), (0.2, 1.0)] and k["rot"] == [(0.0, 3.0)]
    assert k["tint"] == {"color": "foam", "keys": [(0.0, 0.5), (1.0, 0.0)]}
    assert k["pivot"] == "bottom" and k["headroom"] == 0.2
    assert k["drip"] == {"t": 1.5, "g": 2600.0, "stretch": 3.2, "life": 0.55, "seed": 1}
    assert FX.kinetic_spec({}) is None and FX.kinetic_spec({"kinetic": {}}) == {}


@pytest.mark.parametrize("kin, frag", [
    ("x", "must be a table"),
    ({"pop": 1}, "unknown kinetic keys ['pop']"),
    ({"scale": []}, "scale has no keys"),
    ({"scale": [[0, "a"]]}, "scale is a list of [t, value] keys"),
    ({"scale": 3}, "scale is a list of [t, value] keys"),
    ({"show": [1, 2, 3]}, "show is [t_on] or [t_on, t_off]"),
    ({"pivot": "left"}, "pivot is one of"),
    ({"tint": {"color": "foam"}}, "tint is {color"),
    ({"drip": {"g": 1}}, "t required"),
    ({"drip": {"t": 1, "speed": 2}}, "t required"),
])
def test_kinetic_errors_say_what_is_wrong(kin, frag):
    with pytest.raises(TextError, match=frag.replace("[", r"\[").replace("{", r"\{").replace("]", r"\]")):
        FX.kinetic_spec({"kinetic": kin})


@pytest.mark.parametrize("look", ["value", "blink", "flicker", "fade", "ghost", "halo"])
def test_a_kinetic_text_refuses_the_looks_that_need_their_own_material(look):
    with pytest.raises(TextError, match=f"cannot be combined with `{look}`"):
        FX.kinetic_spec({"kinetic": {}, look: {"x": 1}})


def test_backing_defaults_and_validation():
    assert FX.backing_spec({}) is None and FX.backing_spec({"backing": False}) is None
    b = FX.backing_spec({"backing": True})
    assert b == {"color": "surface", "pattern": "plain", "pattern_color": "surface", "pad": [0.55, 0.3], "height": None,
                 "torn": 0.1, "dz": 0.0006, "glow": 0.3, "scale": 1.0, "seed": 1}
    b = FX.backing_spec({"backing": {"color": "rose", "pattern": "dots", "pad": [1, 2], "height": 1.5, "glow": 0.5}})
    assert b["pattern"] == "dots" and b["pad"] == [1.0, 2.0] and b["height"] == 1.5 and b["glow"] == 0.5
    for bad, frag in [({"stripes": 1}, "unknown backing keys"), ({"pattern": "plaid"}, "pattern must be one of"),
                      ({"pad": [1]}, "pad is"), ({"height": -1}, "must be positive"), ({"torn": 0.7}, "torn in"),
                      ("rose", "backing is a table")]:
        with pytest.raises(TextError, match=frag):
            FX.backing_spec({"backing": bad})


def test_tape_size_is_the_ink_plus_the_pad_or_a_fixed_height():
    b = FX.backing_spec({"backing": {"pad": [0.5, 0.25]}})
    w, h = FX.tape_size(b, 2.0, 0.75, 0.04)
    assert w == pytest.approx((2.0 + 1.0) * 0.04) and h == pytest.approx((0.75 + 0.5) * 0.04)
    b = FX.backing_spec({"backing": {"pad": [0.5, 0.25], "height": 1.57}})
    assert FX.tape_size(b, 2.0, 0.75, 0.04) == (pytest.approx(0.12), pytest.approx(1.57 * 0.04))


def test_outline_defaults_and_validation():
    assert FX.outline_spec({}) is None and FX.outline_spec({"outline": False}) is None
    assert FX.outline_spec({"outline": True}) == {"color": "text", "width": 0.05, "alpha": 0.5, "dz": 0.0003}
    o = FX.outline_spec({"outline": {"color": "surface", "width": 0.1, "alpha": 1.0}})
    assert o["color"] == "surface" and o["width"] == 0.1 and o["alpha"] == 1.0
    for bad, frag in [({"glow": 1}, "unknown outline keys"), ({"width": 0}, "width must be positive"),
                      ({"alpha": 0}, "alpha in"), ({"alpha": 1.5}, "alpha in"), ("x", "outline is a table")]:
        with pytest.raises(TextError, match=frag):
            FX.outline_spec({"outline": bad})
