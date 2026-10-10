"""The cartoon fuse's layout (mkmmd.blender.library.props.fuse_layout, pure Python + numpy): its options, the spark and the
card. The Blender half (the cord showing up to min(grow, 1 - burn), the spark riding its end on a moved, scaled prop) is
checked in tests/test_build_stages.py."""
import numpy as np
import pytest

from mkmmd.blender.library.props import fuse_layout as LAY


def test_the_defaults_are_a_grown_unlit_curl_and_a_rope_of_its_own_points_is_taken():
    P, r, start = LAY.options({})
    assert P == pytest.approx(np.array(LAY.CURL)) and r == LAY.RADIUS
    assert start == {"grow": 1.0, "burn": 0.0, "lit": 0.0}                # built whole: the form guard sees the rope
    P, r, start = LAY.options({"points": [[0, 0, 0], [0, 0, 0.2], [0.1, 0, 0.3]], "radius": 0.02, "lit": 0.5})
    assert P.shape == (3, 3) and r == 0.02 and start["lit"] == 0.5


@pytest.mark.parametrize("slots, frag", [
    ({"points": [[0, 0, 0]]}, "at least two"),
    ({"points": [[0.1, 0, 0], [0, 0, 0.2]]}, "the first point is the root"),
    ({"points": [[0, 0, 0], [0, 0, 0.2], [0, 0, 0.2]]}, "coincide"),
    ({"points": "curly"}, "points must be"),
    ({"radius": 0.5}, "radius is metres"),
    ({"grow": 1.5}, "grow is 0..1"),
    ({"lit": -0.1}, "lit is 0..1"),
])
def test_a_bad_fuse_is_refused_saying_what_to_change(slots, frag):
    with pytest.raises(ValueError, match=frag):
        LAY.options(slots)


def test_the_spark_is_a_closed_star_with_twelve_spikes_facing_out():
    V, T = LAY.spark_mesh(0.05)
    edges = {}
    for a, b, c in T:
        for e in ((a, b), (b, c), (c, a)):
            edges[tuple(sorted(e))] = edges.get(tuple(sorted(e)), 0) + 1
    assert set(edges.values()) == {2}                                      # closed: every edge between two faces
    vol = sum(np.dot(V[a], np.cross(V[b], V[c])) for a, b, c in T) / 6.0
    assert vol > 0                                                         # faces wound outward
    r = np.sort(np.linalg.norm(V, axis=1))[::-1]
    assert r[:12] == pytest.approx(0.05) and r[12] < 0.5 * 0.05           # twelve spikes over a small body


def test_the_card_looks_at_the_tip_and_bounds_the_rope_the_socket_and_a_full_spark():
    P, r, _ = LAY.options({})
    card = LAY.card("fz", {}, P, r)
    assert card["use"]["look"][1] == {"name": "tip", "point": pytest.approx(list(P[-1]))}
    lo, hi = np.array(card["bounds"]["min"]), np.array(card["bounds"]["max"])
    assert (lo <= P.min(0) - LAY.SPARK + 1e-4).all() and (hi >= P.max(0) + LAY.SPARK - 1e-4).all()
    assert lo[2] <= -LAY.CAP[1] + 1e-4                                     # the socket under the root
    assert card["fuse"]["length"] == pytest.approx(np.linalg.norm(np.diff(P, axis=0), axis=1).sum(), abs=1e-4)
