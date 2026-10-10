"""The squeaky toy hammer's layout (mkmmd.blender.library.props.squeaky_hammer_layout, no bpy): closed outward meshes in
the card's frame, the pleated head across the top of the handle, and the squash. The Blender half (the head's scale driven
by the root's `squash`) is checked in tests/test_build_stages.py."""
import numpy as np
import pytest

from mkmmd.blender.library.props import squeaky_hammer_layout as LAY


def _closed(m):
    edges = {}
    for f in list(m.Q) + list(m.T):
        for a, b in zip(f, np.roll(f, -1)):
            e = tuple(sorted((int(a), int(b))))
            edges[e] = edges.get(e, 0) + 1
    return set(edges.values()) == {2}


def test_the_head_is_a_closed_pleated_bellows_across_the_handle_capped_at_both_ends():
    m = LAY.head_mesh()
    assert _closed(m) and m.volume() > 0                                           # closed, facing out
    lo, hi = m.bbox()
    assert hi - lo == pytest.approx([LAY.HEAD_L, 2 * LAY.CAP_R, 2 * LAY.CAP_R], abs=2e-3)   # its axis along X
    assert (lo + hi) / 2 == pytest.approx([0.0, 0.0, 0.0], abs=1e-9)              # centred: a squash keeps it in place
    caps = np.isin(np.arange(len(m.V)), np.unique(np.concatenate(
        [m.Q[m.Qm == 1].ravel(), m.T[m.Tm == 1].ravel()]).astype(int)))
    assert np.abs(m.V[caps, 0]).min() >= LAY.HEAD_L / 2 - LAY.CAP_T - 1e-9        # the caps are the two ends
    ring = np.hypot(m.V[:, 1], m.V[:, 2])
    mid = (~caps) & (np.abs(m.V[:, 0]) < LAY.HEAD_L / 2 - LAY.CAP_T) & (ring > 0.5 * LAY.HEAD_R)
    x, r = m.V[mid, 0], ring[mid]
    order = np.argsort(x)
    rr = np.round(r[order], 6)
    folds = np.sum((rr[1:-1] < rr[:-2]) & (rr[1:-1] <= rr[2:]))
    assert r.min() == pytest.approx(LAY.HEAD_R - LAY.PLEAT_D, abs=2e-4) and folds >= LAY.PLEATS   # pleated
    n, end = m.vertex_normals(), np.abs(m.V[:, 0]) > LAY.HEAD_L / 2 - 1e-4
    assert end.sum() > 0 and (np.sign(n[end, 0]) == np.sign(m.V[end, 0])).all()   # both caps' faces look outward


def test_the_handle_rises_from_its_end_at_the_origin_into_the_head():
    m = LAY.handle_mesh()
    assert _closed(m) and m.volume() > 0
    lo, hi = m.bbox()
    assert lo[2] == pytest.approx(0.0, abs=1e-9) and hi[2] == pytest.approx(LAY.HEAD_Z)
    assert 0.0 < LAY.GRIP_Z < LAY.HEAD_Z - LAY.CAP_R                              # a fist closes on the handle, under the head


def test_a_squash_shortens_the_head_and_swells_it_round_and_stays_in_its_range():
    assert LAY.squash(0.0) == (1.0, 1.0)
    along, round_ = LAY.squash(1.0)
    assert along < 0.7 and round_ > 1.1
    assert LAY.squash(2.0) == LAY.squash(1.0) and LAY.squash(-1.0) == LAY.squash(0.0)
