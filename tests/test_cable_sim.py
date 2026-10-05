"""The cord solver (mkmmd.solvers.cable): a lead hanging from a moving plug swings with it, keeps its length, stays out of
the floor and the body beside it, and a still plug leaves it still."""
import math

import numpy as np
import pytest

from mkmmd.core import cable as CB
from mkmmd.solvers import cable as RC
from mkmmd.solvers import geom

FPS = 30.0


def _swing(plug, leg=True):
    F = len(plug)
    d = np.tile([0.0, -0.3, -1.0], (F, 1))
    shape = CB.hang(plug[0], d[0], 0.0, tail=(-1.0, 1.0))
    shapes = geom.Shapes()
    if leg:                                                        # a shin-sized capsule right beside the drop
        shapes.add("capsule", ("bone", "arm", "leg"), "body", model=0, a=[0.0, 0.0, 0.0], b=[0.0, 0.0, -0.8], R=0.06)
    src_pos = np.tile(np.array([[0.09, 0.02, 0.85]]), (F, 1, 1))
    src_R = np.tile(np.eye(3), (F, 1, 1, 1))
    x, rep = RC.simulate(shape, plug, d, shapes.pack(), src_pos, src_R, 0.0, 0.0032, FPS)
    return x, rep, shapes, src_pos, src_R


def test_a_cord_swings_with_a_bouncing_plug_keeps_its_length_and_stays_out_of_the_floor_and_a_leg():
    t = np.arange(90) / FPS
    plug = np.stack([0.02 * np.sin(math.pi * t), np.zeros(len(t)), 0.9 - 0.035 * np.abs(np.sin(math.pi * t * 127.05 / 60))], 1)
    x, rep, shapes, src_pos, src_R = _swing(plug)
    assert x[:, 0] == pytest.approx(plug)                                      # the plug end is the plug
    mid = x.shape[1] // 3
    assert np.ptp(x[:, mid, 2]) > 0.02 and np.ptp(x[:, mid, 0]) > 0.02          # the hanging cord moves with it
    seg = np.linalg.norm(np.diff(x, axis=1), axis=2)
    assert seg.max() < 1.04 * np.median(seg)                                   # no bungee: under 4 % stretch anywhere
    assert x[..., 2].min() >= 0.0032 - 1e-6                                     # never into the floor
    W = geom.world(shapes.pack(), src_R[0], src_pos[0])
    beyond = np.arange(x.shape[1]) * rep["segment_mm"] / 1000 > RC.ANCHOR_FREE
    for f in range(len(x)):
        pen, _n = geom.penetration(x[f][beyond], np.full(beyond.sum(), 0.0032), W, None)["capsule"]
        assert pen.max() < 1e-3                                                # nor through the leg beside it
    assert np.ptp(x[:, -1], axis=0).max() < 1e-9                               # the far end lies where it lies


def test_a_still_plug_leaves_the_cord_still_once_it_has_settled():
    plug = np.tile([0.0, 0.0, 0.9], (60, 1))
    x, _rep, *_ = _swing(plug, leg=False)
    assert np.abs(x[-1] - x[-30]).max() < 5e-4                                 # no swing left, no creep on the floor
