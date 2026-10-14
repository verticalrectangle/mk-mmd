"""Fold shapes a joint angle drives (mkmmd.core.folds): the drivers' angles and weights. bpy-free."""
import numpy as np
import pytest

from mkmmd.core import folds as FD


def rot(axis, deg):
    a = np.asarray(axis, float) / np.linalg.norm(axis)
    t = np.radians(deg)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(t) * K + (1 - np.cos(t)) * K @ K


def frames(*Rs):
    return np.stack([np.asarray(R, float) for R in Rs])


ELBOW = {"morph": "elbow", "driver": {"kind": "angle", "bone": "fore", "ref": "upper", "dir": [1.0, 0.0, 0.0],
                                      "from": 20.0, "to": 110.0}}
TWIST = {"morph": "twist", "driver": {"kind": "twist", "bone": "chest", "ref": "hips", "axis": [0.0, 0.0, 1.0],
                                      "from": 10.0, "to": 40.0}}
DIRS = {"fore": [1.0, 0.0, 0.0], "upper": [1.0, 0.0, 0.0], "chest": [0.0, 0.0, 1.0], "hips": [0.0, 0.0, 1.0]}


def test_an_elbow_fold_grows_with_the_bend_not_with_the_arm():
    """The elbow's fold follows the angle between the forearm and the upper arm: none straight, all of it bent past
    `to`, half way at the middle of the ramp; the whole arm raised or turned unbent shows none of it (measured against
    a fixed direction, a raised straight arm would read as a bend)."""
    bend = [rot([0, -1, 0], d) for d in (0.0, 65.0, 120.0)]
    w = FD.weights([ELBOW], {"fore": frames(*bend), "upper": frames(*[np.eye(3)] * 3)}, DIRS)["elbow"]
    assert np.allclose(w, [0.0, 0.5, 1.0])
    up = rot([0, 1, 0], -80.0)                                          # the straight arm raised 80 degrees
    w = FD.weights([ELBOW], {"fore": frames(up, up @ rot([0, -1, 0], 65.0)), "upper": frames(up, up)}, DIRS)["elbow"]
    assert np.allclose(w, [0.0, 0.5])


def test_a_twist_fold_reads_the_turn_about_the_axis_alone():
    """The waist's fold follows how far the chest has turned on the hips about the vertical, one way only: the same turn
    while she also leans (a swing) reads the same, the other way round reads nothing, and turning hips and chest
    together is no twist at all."""
    turn = rot([0, 0, 1], 25.0)
    lean = rot([1, 0, 0], 30.0)
    w = FD.weights([TWIST], {"chest": frames(turn, lean @ turn, rot([0, 0, 1], -25.0), turn),
                             "hips": frames(np.eye(3), np.eye(3), np.eye(3), turn)}, DIRS)["twist"]
    assert w[0] == pytest.approx(0.5) and w[1] == pytest.approx(w[0], abs=0.02)
    assert w[2] == 0.0 and w[3] == pytest.approx(0.0, abs=1e-9)


def test_a_malformed_fold_is_refused():
    with pytest.raises(ValueError):
        FD.check({"morph": "x", "driver": {"kind": "bend", "bone": "a", "ref": "b", "dir": [0, 0, 1], "from": 0, "to": 1}})
    with pytest.raises(ValueError):
        FD.check({"morph": "x", "driver": {"kind": "angle", "bone": "a", "ref": "b", "dir": [0, 1], "from": 0, "to": 1}})
