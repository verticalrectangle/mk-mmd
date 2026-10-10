"""Rings, bpy-free (mkmmd.core.rings): `[[ring]]` normalised and sorted, which rings are on screen when, their discs and
bands in frame pixels as they sweep out, and the flip and edge masks the vector look paints with. The Blender half (the
centres projected through the shot's camera, mkmmd.blender.styles) is checked by rendering and looking."""
import math

import numpy as np
import pytest

from mkmmd.core import palette as PAL
from mkmmd.core import rings as RG

DAWN = PAL.get("rose-pine-dawn")


def _rings(*specs):
    return RG.normalize(list(specs), DAWN)


def test_rings_are_sorted_and_end_when_swept_or_when_the_next_ring_starts_if_held():
    r = _rings({"at": 2.0, "center": [0.5, 0.5], "hold": True}, {"at": 1.0, "center": [0.5, 0.5], "dur": 0.5},
               {"at": 3.0, "center": "obj('boombox').location", "hold": True})
    assert [x["at"] for x in r] == [1.0, 2.0, 3.0] and [x["index"] for x in r] == [1, 0, 2]
    assert r[0]["until"] == 1.5 and r[1]["until"] == 3.0 and r[2]["until"] == math.inf
    assert r[2]["center"] == {"mode": "point", "expr": "obj('boombox').location"}
    assert RG.live(r, 1.6) == [] and [x["at"] for x in RG.live(r, 2.9)] == [2.0]


@pytest.mark.parametrize("spec, frag", [
    ({"center": [0.5, 0.5]}, "needs `at`"),
    ({"at": 1.0}, "needs `center`"),
    ({"at": 1.0, "center": [0.5]}, "center"),
    ({"at": 1.0, "center": [0.5, 0.5], "dur": 0}, "positive"),
    ({"at": 1.0, "center": [0.5, 0.5], "ease": "bounce"}, "ease"),
    ({"at": 1.0, "center": [0.5, 0.5], "edge": {"colour": "#fff"}}, "edge"),
    ({"at": 1.0, "center": [0.5, 0.5], "size": 1}, "unknown keys")])
def test_bad_rings_are_refused(spec, frag):
    with pytest.raises(RG.RingError, match=frag):
        _rings(spec)


def test_a_band_sweeps_from_its_centre_until_it_has_left_the_frame():
    r = _rings({"at": 1.0, "center": [0.25, 0.5], "dur": 1.0, "width": 0.2, "ease": "linear"})
    size, c = (100, 200), {0: (0.25, 0.5)}
    far = math.hypot(75.0, 100.0)                                             # from (25, 100) to a right-hand corner
    s = RG.shapes(r, 1.0, c, size)[0]
    assert s["c"] == (25.0, 100.0) and s["outer"] == 0.0 and s["inner"] == 0.0
    s = RG.shapes(r, 1.5, c, size)[0]
    assert s["outer"] == pytest.approx(0.5 * (far + 40.0)) and s["inner"] == pytest.approx(s["outer"] - 40.0)
    s = RG.shapes(r, 1.9999, c, size)[0]
    assert s["inner"] >= far - 0.05                                           # nothing of the band is left on the frame
    assert RG.shapes(r, 2.0, c, size) == []
    assert RG.shapes(r, 1.5, {0: None}, size) == []                          # a centre behind the camera: no ring


def test_a_held_disc_fills_the_frame_and_stays_until_the_next_ring():
    r = _rings({"at": 1.0, "center": [0.5, 0.5], "hold": True, "dur": 0.5}, {"at": 3.0, "center": [0.5, 0.5]})
    c, size = {0: (0.5, 0.5), 1: (0.5, 0.5)}, (40, 60)
    shapes = RG.shapes(r, 2.5, c, size)
    assert RG.masks(shapes, size, 1)[0].all() and shapes[0]["edge"] is None   # swept past the frame: no wavefront left
    assert not RG.masks(RG.shapes(r, 3.0, c, size), size, 1)[0].any()       # the next ring starts: the disc is gone


def test_overlapping_rings_cut_back_into_each_other():
    discs = [{"c": (20.0, 20.0), "outer": 10.0, "inner": 0.0, "edge": None},
             {"c": (30.0, 20.0), "outer": 10.0, "inner": 0.0, "edge": None}]
    flip, edges = RG.masks(discs, (50, 40), 1)
    assert flip[20, 14] and flip[20, 35] and not flip[20, 25] and not flip[5, 5] and edges == []


def test_the_wavefront_runs_on_the_outer_rim_as_wide_as_asked_at_1080():
    band = [{"c": (0.0, 0.0), "outer": 50.0, "inner": 30.0, "edge": {"color": [1.0, 1.0, 1.0], "width": 10.8}}]
    flip, edges = RG.masks(band, (108, 108), 2)                              # 10.8 px at 1080 is 1.08 px on 108 px
    (m, rgb), = edges
    ys, xs = np.nonzero(m)
    d = np.hypot((xs + 0.5) / 2.0, (ys + 0.5) / 2.0)
    assert len(d) and np.all(np.abs(d - 50.0) <= 0.54 + 1e-9) and rgb == [1.0, 1.0, 1.0]
    assert flip[2 * 40, 2 * 5] and not flip[2 * 10, 2 * 10] and not flip[2 * 60, 2 * 60]   # the band, not its middle
