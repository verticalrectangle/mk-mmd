"""The paper scissors, bpy-free (mkmmd.core.paper_scissors and its meshes, mkmmd.blender.library.props.
paper_scissors_parts): the cut cards meet at the tips shut and open into a V of 60 degrees as the drivers turn them,
the tip and cut points lie between the blades, the hold between the loops, and the cards never pass through each
other."""
import math

import numpy as np
import pytest

from mkmmd.blender.library.props import paper_scissors_parts as PSP
from mkmmd.core import gripframe as GF
from mkmmd.core import paper_scissors as PS
from mkmmd.core import paper_theatre as PT


def _eval(expr, open_):
    return eval(expr, {"__builtins__": {}}, {"open": open_, "cos": math.cos, "sin": math.sin,
                                              "clamp": lambda v, lo, hi: min(max(v, lo), hi)})


def _turned3(V, angle):
    c, s = math.cos(angle), math.sin(angle)
    return np.stack([c * V[:, 0] + s * V[:, 2], V[:, 1], -s * V[:, 0] + c * V[:, 2]], 1)


@pytest.fixture(scope="module")
def parts():
    return PSP.parts()


def test_the_cut_cards_meet_at_the_tips_shut_and_open_a_v_of_sixty(parts):
    tips = {}
    for side in "ab":
        V = parts[f"blade_{side}"].V
        tips[side] = V[np.argmax(V[:, 2])]
        assert parts[f"blade_{side}"].is_closed()
    assert np.hypot(*(tips["a"] - tips["b"])[[0, 2]]) < 0.0015                 # shut: one tip
    assert tips["a"][2] == pytest.approx(PS.TIP, abs=0.001)
    ex = PS.expressions()
    for o in (-0.3, 0.0, 0.25, 0.5, 1.0, 1.4):                                 # the drivers turn as the numpy says
        a, b = PS.angles(o)
        assert _eval(ex["blade_a"], o) == pytest.approx(float(a)) and _eval(ex["blade_b"], o) == pytest.approx(float(b))
        assert _eval(ex["tip"], o) == pytest.approx(PS.tip(o)[0, 2])
        assert (_eval(ex["hold_x"], o), _eval(ex["hold_z"], o)) == pytest.approx(tuple(PS.hold(o)[0, [0, 2]]))
    a, b = PS.angles(1.0)
    ta, tb = _turned3(tips["a"][None], float(a))[0], _turned3(tips["b"][None], float(b))[0]
    v = math.degrees(math.atan2(ta[0], ta[2]) - math.atan2(tb[0], tb[2]))
    assert v == pytest.approx(2.0 * PS.TURN, abs=1.5)                          # open: a V of 60 degrees
    assert ta[0] > 0.02 and tb[0] < -0.02                                      # piece a's blade to +X
    lo, hi = np.array(PS.bounds()["min"]), np.array(PS.bounds()["max"])
    for o in (0.0, 0.5, 1.0):                                                  # the card's bounds hold every opening
        a, b = PS.angles(o)
        for key, t in (("blade_a", a), ("blade_b", b), ("pin", b)):
            V = _turned3(parts[key].V, float(t))
            assert (V >= lo - 1e-6).all() and (V <= hi + 1e-6).all(), (key, o)


def test_tip_and_cut_lie_between_the_blades_and_shutting_cuts(parts):
    O = PS.outlines()

    def on_blade(p, o):
        """Is the prop-frame point p (x, z) on blade a, on blade b at the opening o?"""
        a, b = PS.angles(o)
        return (bool(PT.inside(PS.turned(np.array([p]), -float(a)), O["blade_a"])[0]),
                bool(PT.inside(PS.turned(np.array([p]), -float(b)), O["blade_b"])[0]))

    cut = (0.0, PS.CUT)
    for o in (0.35, 0.6, 1.0):
        x, _y, z = PS.tip(o)[0]
        assert on_blade((x, z), o) == (False, False), o
        assert on_blade(cut, o) == (False, False), o
        a, b = PS.angles(o)                                                    # in the V: past each cutting edge
        assert PS.turned(np.array([cut]), -float(a))[0, 0] < -PS.EDGE < 0.0 < PS.EDGE < \
            PS.turned(np.array([cut]), -float(b))[0, 0]
    assert on_blade(cut, 0.0) == (True, True)                                  # shut over the thread: cut
    assert on_blade(cut, 0.95 * PS.cut_shut()) == (True, True)
    assert on_blade(cut, 1.05 * PS.cut_shut()) == (False, False)
    lay = PS.layers()
    assert lay["blade_a"][1] < 0.0 < lay["blade_b"][0]                         # the empties ride between the cards
    ya, yb = parts["blade_a"].V[:, 1], parts["blade_b"].V[:, 1]
    assert ya.max() < 0.0 < yb.min()                                           # never through each other
    yp = parts["pin"].V[:, 1]
    assert yp.min() < ya.min() and yp.max() > yb.max()                         # the pin through all four cards


def test_the_hold_lies_between_the_loops_and_its_pinch_takes_their_bottom():
    O = PS.outlines()
    for o in (0.0, 0.5, 1.0):
        a, b = PS.angles(o)
        hx, _hy, hz = PS.hold(o)[0]
        ca = PS.turned(np.array([PS.LOOPS["a"][0]]), float(a))[0]
        cb = PS.turned(np.array([PS.LOOPS["b"][0]]), float(b))[0]
        assert ca[0] < hx < cb[0]
        for side, t in (("a", a), ("b", b)):                                   # in the gap, on neither handle
            assert not PT.inside(PS.turned(np.array([[hx, hz]]), -float(t)), O[f"handle_{side}"])[0], (o, side)
    card = PS.card("s", PS.options({}), PS.ROLES)
    g = card["use"]["grip"][0]
    assert (g["name"], g["object"], g["dir"], g["palm"]) == ("hold", "s_hold", [0.0, 0.0, 1.0], [0.0, -1.0, 0.0])
    assert g["point"] == pytest.approx(PS.hold(0.0)[0].tolist(), abs=1e-5)
    lay = PS.layers()
    assert g["width"] == pytest.approx(lay["handle_b"][1] - lay["handle_a"][0])
    G = GF.pinch_frame(g["center"], g["axis"], g["normal"], g["span"], 0.004)
    assert G[2, 3] == pytest.approx(PS.TIP - PS.LENGTH + 0.004, abs=1e-4)     # the pads 4 mm up from the loops' bottom
    assert abs(G[1, 2]) == pytest.approx(1.0)                                  # squeezing through the card
    assert card["length"] == PS.LENGTH
    assert [u["object"] for u in card["use"]["look"]] == ["s_pivot", "s_tip", "s_cut"]


def test_options_refuse_what_they_cannot_build():
    assert PS.options({})["open"] == 0.0 and PS.options({"open": 1})["open"] == 1.0
    for bad in (1.5, -0.1, "0.5", True, float("nan")):
        with pytest.raises(ValueError, match="open"):
            PS.options({"open": bad})
