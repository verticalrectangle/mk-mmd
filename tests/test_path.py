import math

import numpy as np
import pytest

from mkmmd.core.path import Path, gentle_road, speed_profile


def test_straight_path_frames_and_offsets():
    p = Path([[0, 0, 0], [0, -100, 0]])
    assert p.length == pytest.approx(100.0)
    T, L, U = p.frame(50.0)
    assert T == pytest.approx([0, -1, 0]) and U == pytest.approx([0, 0, 1])
    assert L == pytest.approx([1, 0, 0])                      # travelling -Y, the left is +X
    assert p.offset(50.0, lateral=2.0, height=1.0) == pytest.approx([2.0, -50.0, 1.0])
    assert p.heading(10.0) == pytest.approx(-math.pi / 2)


def test_circle_curvature_sign_and_magnitude():
    R = 200.0
    a = np.linspace(0, math.pi, 37)                           # 5 deg apart: the spline follows the circle closely
    ccw = np.stack([R * np.cos(a), R * np.sin(a), 0 * a], 1)
    p = Path(ccw, samples_per_seg=48)
    k = p.curvature(p.length / 2)
    assert k == pytest.approx(1 / R, rel=0.03)                # turning left: positive
    assert Path(ccw[::-1], samples_per_seg=48).curvature(p.length / 2) == pytest.approx(-1 / R, rel=0.03)


def test_project_finds_arc_length():
    p = Path([[0, 0, 0], [0, -100, 0], [50, -150, 0]])
    s = p.project([p.point(70.0)])
    assert s[0] == pytest.approx(70.0, abs=0.5)


def test_gentle_road_is_seeded_and_turns_gently():
    a, b = gentle_road(1000, seed=3), gentle_road(1000, seed=3)
    assert np.allclose(a, b)
    p = Path(a)
    k = np.abs(p.curvature(np.linspace(20, p.length - 20, 200)))
    assert k.max() < 0.01                                     # radius > 100 m everywhere


def test_speed_profile_distance():
    s, v = speed_profile([[0.0, 20.0], [10.0, 20.0]], 30, np.arange(-30, 301))
    assert s[30] == pytest.approx(0.0) and s[-1] == pytest.approx(200.0) and s[0] == pytest.approx(-20.0)
