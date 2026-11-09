"""The boom microphone stand (pure numpy: the Blender side is checked by building and looking): the grille where the
options say, the microphone over the boom's end in its clip, the stand on the floor, and options that cannot stand."""
import numpy as np
import pytest

from mkmmd.blender.library.props import handheld_mic_layout as MIC
from mkmmd.blender.library.props import mic_stand_layout as L


def test_the_grille_is_where_the_options_put_it_and_the_mic_lies_over_the_boom():
    opt = {"height": 1.24, "angle": 35.0, "boom": 0.30}
    f, parts, card = L.frame(opt), L.meshes(opt), L.card("mic", opt=opt)
    for k, m in parts.items():
        assert m.is_closed() and m.volume() > 0, k
    ball = parts["mic_grille"].V
    top = ball[np.argmax(ball @ f["d"])]                                      # the top of the ball, along the boom
    centre = top - MIC.BALL_R * f["d"]
    assert centre == pytest.approx([0.0, f["reach"], 1.24], abs=1e-4)
    look = {e["name"]: e["point"] for e in card["use"]["look"]}
    assert look["grille"] == pytest.approx(centre, abs=1e-4) and card["reach"] == pytest.approx(f["reach"], abs=1e-5)
    assert f["d"] == pytest.approx([0.0, np.cos(np.radians(35)), np.sin(np.radians(35))])
    body = parts["mic_body"].V - f["tail"]
    radial = np.linalg.norm(body - np.outer(body @ f["d"], f["d"]), axis=1)
    assert radial.max() == pytest.approx(MIC.COLLAR[0], abs=3e-4)             # round about the boom's direction
    boom_axis = f["pivot"] + np.outer(np.linspace(0, 1, 9), f["end"] - f["pivot"])
    gap = np.min(np.linalg.norm(parts["mic_body"].V[:, None] - boom_axis[None], axis=2))
    assert gap > L.BOOM_R + 0.004                                             # over the boom, not skewered on it
    assert parts["base"].V[:, 2].min() == pytest.approx(0.0, abs=1e-6)
    pole = parts["pole"].V
    assert pole[:, 2].max() == pytest.approx(f["pivot"][2], abs=1e-6) and np.abs(pole[:, :2]).max() < 0.02
    path = L.cable_path(opt)
    assert path[0] == pytest.approx(f["tail"] - 0.002 * f["d"]) and path[-1][2] == pytest.approx(L.CABLE_R)


def test_options_that_cannot_stand_are_refused():
    assert L.options() == L.OPTIONS and L.options({"height": 1.1, "stand": "#ffffff"})["height"] == 1.1
    with pytest.raises(ValueError, match="under its clutch"):
        L.frame({"height": 0.8, "angle": 60})
    with pytest.raises(ValueError, match="boom"):
        L.options({"boom": 2.0})
