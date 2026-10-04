"""Pure-numpy parts of the night_sky set: sky directions and frames, the world-to-root conversion, the star density
solver (against a Monte Carlo count on a jittered grid, the layout of the Voronoi node) and the spec parser.
(The world shader itself is checked by building and looking; see docs.)"""
import math

import numpy as np
import pytest

from mkmmd.blender.library.sets import night_sky as N
from mkmmd.core.path import Path


def test_direction_follows_the_heading_convention():
    assert N.direction(0) == pytest.approx([1, 0, 0])
    assert N.direction(90) == pytest.approx([0, 1, 0])
    assert N.direction(-90) == pytest.approx([0, -1, 0])            # where the default road runs
    assert N.direction(0, 90) == pytest.approx([0, 0, 1], abs=1e-12)
    assert np.linalg.norm(N.direction(33.0, 21.0)) == pytest.approx(1.0)
    path = Path(np.array([[0.0, 0.0, 0.0], [-30.0, -80.0, 0.0], [-40.0, -200.0, 0.0]]))
    for s in (10.0, 100.0):
        heading = math.degrees(float(path.heading(s)))
        assert N.direction(heading) == pytest.approx(path.tangent(s)[:2].tolist() + [0.0], abs=1e-9)


@pytest.mark.parametrize("az,el", [(0, 0), (-74, 13), (120, 60), (10, 89.0)])
def test_tangent_frame_is_orthonormal_horizontal_and_right_handed(az, el):
    m = N.direction(az, el)
    e1, e2 = N.tangent_frame(m)
    assert np.linalg.norm(e1) == pytest.approx(1.0) and np.linalg.norm(e2) == pytest.approx(1.0)
    assert e1 @ m == pytest.approx(0.0, abs=1e-12) and e2 @ m == pytest.approx(0.0, abs=1e-12)
    assert e1[2] == pytest.approx(0.0, abs=1e-12) and e2[2] > 0.0
    assert np.cross(e1, e2) == pytest.approx(m)


def test_tangent_frame_left_and_zenith():
    e1, e2 = N.tangent_frame(np.array([1.0, 0.0, 0.0]))             # looking along +X, left is +Y
    assert e1 == pytest.approx([0, 1, 0]) and e2 == pytest.approx([0, 0, 1])
    e1, e2 = N.tangent_frame(np.array([0.0, 0.0, 1.0]))             # straight up: no horizontal direction to follow
    assert e1 == pytest.approx([1, 0, 0])


def test_to_root_undoes_the_roots_yaw():
    v = N.direction(-74, 13)
    for yaw in (0.0, 0.7, -2.0):
        r = N.to_root(v, yaw)
        c, s = math.cos(yaw), math.sin(yaw)
        assert np.array([c * r[0] - s * r[1], s * r[0] + c * r[1], r[2]]) == pytest.approx(v)
    assert N.to_root([1, 0, 0], math.pi / 2) == pytest.approx([0, -1, 0], abs=1e-12)    # a root turned to face +Y


def test_star_scale_grows_with_density_and_shrinks_with_size():
    assert N.star_scale(8, 0.07) > N.star_scale(4, 0.07) > N.star_scale(2, 0.07)
    assert N.star_scale(4, 0.14) < N.star_scale(4, 0.07) < N.star_scale(4, 0.035)
    assert 20 < N.star_scale(N.STARS["density"], N.STARS["size"]) < 100


@pytest.mark.parametrize("density,size,brightness", [(4.0, 0.07, 1.0), (10.0, 0.05, 2.0), (1.5, 0.12, 0.7)])
def test_star_scale_gives_the_requested_density(density, size, brightness):
    """Count the visible stars of a jittered 3D grid (one feature point per cell, the Voronoi node's layout) around a
    sphere of radius S cells: a star is visible when its brightness times exp(-d^2 / 2 sigma^2) clears the floor."""
    S = N.star_scale(density, size, brightness)
    rng = np.random.default_rng(1)
    g = np.arange(-int(math.ceil(S)) - 2, int(math.ceil(S)) + 2)
    cells = np.stack(np.meshgrid(g, g, g, indexing="ij"), -1).reshape(-1, 3)
    d = np.linalg.norm(cells + rng.random(cells.shape), axis=1) - S      # radial offset from the unit sphere, in cells
    d = d[np.abs(d) < 2.0]
    b = rng.random(len(d)) ** N.STAR_GAMMA
    sigma = math.radians(size) / 2.3548 * S * N.size_factor(b)
    seen = int((N.STAR_PEAK * brightness * b * np.exp(-d ** 2 / (2 * sigma ** 2)) > N.STAR_FLOOR).sum())
    assert seen == pytest.approx(density * N.SPHERE_SQDEG / 100.0, rel=0.08)


def test_parse_fills_defaults_and_switches_layers():
    cfg = N.parse({"name": "sky", "kind": "night_sky", "at": [0, 0, 0], "yaw": 30})
    assert cfg["zenith"] == "base" and cfg["horizon"] is None
    assert cfg["stars"] == N.STARS and cfg["moon"] == N.MOON and cfg["glow"] == N.GLOW
    assert cfg["clouds"] == N.CLOUDS and cfg["eevee"] == N.EEVEE
    cfg = N.parse({"stars": False, "moon": {"az": -60}, "clouds": True, "eevee": False})
    assert cfg["stars"] is None and cfg["eevee"] is None and cfg["clouds"] == N.CLOUDS
    assert cfg["moon"]["az"] == -60 and cfg["moon"]["el"] == N.MOON["el"]
    assert N.MOON["az"] == -74                                           # the defaults are not modified


@pytest.mark.parametrize("spec", [
    {"colour": "base"}, {"stars": {"dens": 1}}, {"moon": 5}, {"glow": [1]},
    {"band": 0.0}, {"stars": {"density": 0}}, {"clouds": {"cover": 1.5}}, {"moon": {"size": 0}},
    {"stars": {"fade": [20, 5]}}, {"stars": {"fade": [5]}},
    {"eevee": {"trace_scale": 3}}, {"eevee": {"probe": 1000}}, {"eevee": {"quality": 2}},
])
def test_parse_rejects_bad_specs(spec):
    with pytest.raises(ValueError, match="night_sky"):
        N.parse(spec)
