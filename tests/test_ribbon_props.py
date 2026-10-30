"""The giant toothbrush and toothpaste tube (library:toothbrush, library:toothpaste_tube; their layouts are numpy only):
closed forms, cards that match the meshes (the hold on the handle, the bristles' tips, the nozzle in the open mouth),
a tube that keeps its perimeter as it flattens toward the crimp and when it is squeezed. No Blender."""
import math

import numpy as np
import pytest

from mkmmd.blender.library.props import toothbrush_layout as TB
from mkmmd.blender.library.props import toothpaste_tube_layout as TT


def test_the_toothbrush_is_one_closed_form_from_the_handle_s_end_to_the_head_s_tip():
    m = TB.body_mesh()
    assert m.is_closed() and m.volume() > 0
    lo, hi = m.bbox()
    assert lo[2] == pytest.approx(0.0, abs=0.005) and hi[2] == pytest.approx(TB.LENGTH, abs=0.005)
    assert set(np.unique(m.Qm)) == {TB.MATS.index("handle"), TB.MATS.index("grip")}
    w, t, yc, _ = TB.section(np.array([0.17, 0.55, 0.67]))
    assert w[0] > 0.04 and t[0] > 0.028                                  # a fist's handle
    assert w[1] < 0.022 and w[2] > 0.044                                 # a thin neck, a wide head
    assert yc[2] < yc[0] - 0.01                                          # the head sits forward of the handle's line


def test_the_hold_is_the_handle_and_the_brush_point_is_the_middle_of_the_bristles_tips():
    card = TB.card("b")
    hold = card["use"]["grip"][0]
    assert hold["name"] == "hold" and hold["type"] == "hold" and hold["axis"] == [0.0, 0.0, 1.0]
    z0, z1 = hold["center"][2] - hold["length"] / 2, hold["center"][2] + hold["length"] / 2
    assert z0 >= 0.0 and z1 <= 0.53                                      # the handle, short of the head
    w, t, _, _ = TB.section(np.array([hold["center"][2]]))
    assert hold["radius"] == pytest.approx(0.25 * (w[0] + t[0]), abs=1e-4)
    brush = card["use"]["anchor"][0]
    assert brush["name"] == "brush" and brush["dir"] == [0.0, -1.0, 0.0]
    parts = TB.bristle_meshes()
    lo = min(p.V[:, 1].min() for p in parts.values())
    assert lo <= brush["point"][1] <= lo + 0.005                         # at the tips (the tallest a little further)
    zs = np.concatenate([p.V[:, 2] for p in parts.values()])
    assert brush["point"][2] == pytest.approx(0.5 * (zs.min() + zs.max()), abs=0.002)


def test_every_tuft_stands_out_of_the_head_s_front_from_inside_it():
    for x, z, _, _ in TB.tufts():
        assert -0.03 < TB.surface_y(x, z) < -0.02                        # the head's front
    parts = TB.bristle_meshes()
    for p in parts.values():
        assert p.is_closed()
        assert p.V[:, 1].max() == pytest.approx(max(TB.surface_y(x, z) + TB.SINK for x, z, _, _ in TB.tufts()), abs=0.003)
    assert len(TB.tufts()) == 36


def test_the_tube_keeps_its_perimeter_flattening_to_the_crimp_and_when_squeezed():
    z = np.linspace(TT.CRIMP, TT.SHOULDER, 40)
    for s in (0.0, 1.0):
        b = TT.body_b(z, s)
        a = TT.half_width(b)
        assert TT.perimeter(a, b) == pytest.approx(np.full(len(z), 2.0 * math.pi * TT.R), rel=1e-6)
        assert np.all(a >= b - 1e-12)
    assert TT.body_b(np.array([TT.CRIMP]))[0] == pytest.approx(TT.SEAL) and TT.body_b(np.array([TT.SHOULDER]))[0] == \
        pytest.approx(TT.R)
    pressed = TT.body_b(z, 1.0) / TT.body_b(z, 0.0)
    assert pressed.min() == pytest.approx(1.0 - TT.SQUEEZE[0], abs=0.02)
    assert z[np.argmin(pressed)] == pytest.approx(TT.SQUEEZE[1], abs=0.015)


def test_the_squeeze_moves_the_body_not_the_crimp_shoulder_or_neck():
    move = np.linalg.norm(TT.squeeze_offsets(1.0), axis=1)
    V = TT.body_mesh(0.0).V
    assert move[(V[:, 2] < TT.CRIMP)].max() < 1e-9 and move[(V[:, 2] > TT.SHOULDER)].max() < 1e-9
    assert 0.02 < move.max() < 0.04
    assert TT.body_mesh(0.0).is_closed() and TT.body_mesh(1.0).is_closed()


def test_the_nozzle_is_the_open_mouth_and_lets_out_paste_that_fits_it():
    card = TT.card("t")
    nozzle = card["use"]["anchor"][0]
    assert nozzle["name"] == "nozzle" and nozzle["point"] == [0.0, 0.0, TT.LENGTH] and nozzle["dir"] == [0.0, 0.0, 1.0]
    assert nozzle["radius"] == TT.PASTE_R < TT.BORE_R < TT.NECK_R
    V = TT.body_mesh().V
    near = V[np.hypot(V[:, 0], V[:, 1]) < TT.BORE_R - 1e-4]
    assert near[:, 2].max() <= TT.LENGTH - TT.BORE_DEPTH + 1e-6             # nothing in the mouth above the bore's floor
    hold = card["use"]["grip"][0]
    assert hold["type"] == "hold" and TT.CRIMP <= hold["center"][2] - hold["length"] / 2
    assert hold["center"][2] + hold["length"] / 2 <= TT.SHOULDER and card["params"] == {"squeeze": 0.0}
    assert TT.card("t", capped=True)["objects"] == ["t_body", "t_cap"] and TT.cap_mesh().is_closed()
