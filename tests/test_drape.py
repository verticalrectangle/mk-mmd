import numpy as np
import pytest

from mkmmd.core import drape as DR


def _rot(axis, deg):
    a = np.asarray(axis, float)
    a = a / np.linalg.norm(a)
    th = np.radians(deg)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K


def test_min_rotation_takes_one_direction_onto_the_other_by_the_least_angle():
    a, b = np.array([0.0, -0.3, -0.95]), np.array([0.62, -0.25, -0.74])
    R = DR.min_rotation(a, b)
    assert R @ DR.unit(a) == pytest.approx(DR.unit(b))
    assert np.allclose(R @ R.T, np.eye(3)) and np.linalg.det(R) == pytest.approx(1.0)
    angle = np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))
    assert angle == pytest.approx(np.degrees(np.arccos(DR.unit(a) @ DR.unit(b))))   # no twist about the direction


def test_min_rotation_of_equal_and_opposite_directions():
    a = np.array([0.2, -0.5, -0.8])
    assert np.allclose(DR.min_rotation(a, 3 * a), np.eye(3))
    R = DR.min_rotation(a, -a)                                                       # a half turn
    assert R @ DR.unit(a) == pytest.approx(-DR.unit(a))
    assert np.allclose(R @ R.T, np.eye(3)) and np.linalg.det(R) == pytest.approx(1.0)
    with pytest.raises(ValueError, match="zero length"):
        DR.min_rotation([0, 0, 0], a)


def test_every_bone_of_a_chain_ends_up_along_its_direction_whatever_is_above_it():
    rest = [np.array([0.0, -0.2, -0.98]), np.array([0.1, -0.1, -0.99]), np.array([0.0, 0.1, -1.0])]
    want = [[0.0, -0.70, -0.71], [0.0, -0.93, -0.37], [0.0, -0.85, -0.53]]
    parent = _rot([1, 0, 0], -6.0)                                                   # the pelvis tilt above the chain
    keys, deltas = DR.chain_keys(rest, want, parent)
    D = parent
    for q, r, w in zip(keys, rest, want):
        D = D @ q                                                                    # D_i = D_(i-1) q_i
        assert D @ DR.unit(r) == pytest.approx(DR.unit(w), abs=1e-9)
    assert all(np.allclose(d, DR.min_rotation(r, w)) for d, r, w in zip(deltas, rest, want))


def test_the_first_key_undoes_the_parents_rotation():
    rest, want = [np.array([0.0, 0.0, -1.0])], [[0.0, -1.0, 0.0]]
    parent = _rot([0, 0, 1], 30.0)
    (q,), _ = DR.chain_keys(rest, want, parent)
    assert parent @ q == pytest.approx(DR.min_rotation(rest[0], want[0]))
    (q0,), _ = DR.chain_keys(rest, want)
    assert q0 == pytest.approx(DR.min_rotation(rest[0], want[0]))


def test_a_chain_needs_one_direction_per_bone():
    with pytest.raises(ValueError, match="2 directions for 3 bones"):
        DR.chain_keys([[0, 0, -1]] * 3, [[0, -1, 0]] * 2)
