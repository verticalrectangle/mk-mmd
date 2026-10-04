"""Pure maths of the desk's drawer pedestal: the plan outline as a path, the carcass tube, the wrap-around drawer fronts, and
the `form` metric on them (the pedestal must not be a box). Blender-side building is checked by building and looking."""
import math

import numpy as np
import pytest

from mkmmd.blender.library.props import bedroom_furniture_ped as P
from mkmmd.core import form as F
from mkmmd.core import shell as CS


def test_outline_is_a_closed_ccw_rounded_rectangle_starting_mid_front():
    o = P.outline()
    assert CS.polygon_area(o) > 0
    assert o[0] == pytest.approx([(P.CARC_X[0] + P.CARC_X[1]) / 2, P.CARC_Y[0]])
    assert o.min(0) == pytest.approx([P.CARC_X[0], P.CARC_Y[0]]) and o.max(0) == pytest.approx([P.CARC_X[1], P.CARC_Y[1]])
    rf, rb = P.PLAN_R
    cx, cy = P.CARC_X[1] - rf, P.CARC_Y[0] + rf                                # the front right corner arc
    arc = o[(o[:, 0] > cx + 1e-9) & (o[:, 1] < cy - 1e-9)]
    assert len(arc) >= 10
    assert np.linalg.norm(arc - [cx, cy], axis=1) == pytest.approx(rf)
    cx, cy = P.CARC_X[1] - rb, P.CARC_Y[1] - rb                                # the back right corner arc
    arc = o[(o[:, 0] > cx + 1e-9) & (o[:, 1] > cy + 1e-9)]
    assert len(arc) >= 10
    assert np.linalg.norm(arc - [cx, cy], axis=1) == pytest.approx(rb)
    with pytest.raises(AssertionError):
        P.outline(r=(0.1, 0.1))                                                # too big for the front


def test_fronts_stay_inside_the_outer_extents_and_the_kneehole_is_free():
    """Fronts, carcass and joints stay in x <= -0.28 (the kneehole starts there) and x >= -0.592 (the slab's edge is at
    -0.6), and nothing is nearer the sitter than the front face at y = -0.28."""
    for m in (P.carcass(), P.fronts(), P.joints()):
        lo, hi = m.bbox()
        assert lo[0] >= P.PED_X[0] - 1e-6 and hi[0] <= P.PED_X[1] + 1e-6
        assert lo[1] >= P.PED_Y[0] - 1e-6 and hi[1] <= P.PED_Y[1] + 1e-6
        assert lo[2] >= 0.0 and hi[2] <= 0.71
    lo, _ = P.fronts().bbox()
    assert lo[1] == pytest.approx(P.PED_Y[0])                                  # the crowned middle of the front
    assert P.pull_centre()[1] == pytest.approx(P.PED_Y[0])


def test_path_positions_and_normals():
    path = P.Path(P.outline())
    pos, n = path.at([0.0, path.total])
    assert pos[0] == pytest.approx(pos[1]) and n[0] == pytest.approx([0.0, -1.0])
    s = np.linspace(-path.total / 2, path.total / 2, 401)
    pos, n = path.at(s)
    assert np.linalg.norm(n, axis=1) == pytest.approx(1.0)
    ctr = P.Path(P.outline()).P.mean(0)
    assert ((pos - ctr) * n).sum(1).min() > 0                                  # outward everywhere
    step = np.linalg.norm(np.diff(pos, axis=0), axis=1)
    assert step.max() < 1.5 * path.total / 400                                 # no jumps along the path


def test_half_wrap_covers_front_corner_and_a_little_of_the_side():
    path = P.Path(P.outline())
    half = P.half_wrap(path)
    rf = P.PLAN_R[0]
    front_half = (P.CARC_X[1] - P.CARC_X[0]) / 2 - rf
    assert half == pytest.approx(front_half + math.pi / 2 * rf + P.WRAP, abs=2e-3)
    end, _ = path.at([half])
    assert end[0, 0] == pytest.approx(P.CARC_X[1])                             # on the right wall
    assert end[0, 1] == pytest.approx(P.CARC_Y[0] + rf + P.WRAP, abs=2e-3)


def test_drawer_rows_and_joints():
    rows = P.ROWS
    assert len(rows) == 3 and rows[0][1] == pytest.approx(0.695)
    for (b0, t0), (b1, t1) in zip(rows, rows[1:]):
        assert t0 - b0 > 0.15 and b0 - t1 == pytest.approx(0.004)
    assert rows[-1][0] > P.WALL_Z                                              # the lowest front clears the foot roll
    assert rows[0][1] < P.Z_TOP
    assert P.JOINTS == [(rows[1][1], rows[0][0]), (rows[2][1], rows[1][0])]


def test_carcass_is_a_vertical_outward_tube_with_a_kick_plate():
    c = P.carcass(0, 1)
    assert not c.is_closed()                                                   # hollow: open at the foot and under the slab
    lo, hi = c.bbox()
    assert lo[2] == 0.0 and hi[2] == pytest.approx(P.Z_TOP)
    cq, _ = c.face_centres()
    nq, _ = c.face_vectors()
    wall = cq[:, 2] > P.WALL_Z + 0.02
    ctr = np.array([np.mean(P.CARC_X), np.mean(P.CARC_Y), 0.0])
    out = ((cq[wall] - ctr)[:, :2] * nq[wall][:, :2]).sum(1)
    assert (out > 0).all() and np.abs(nq[wall][:, 2]).max() < 1e-9             # outward and exactly vertical
    roles = c.Qm
    assert set(np.unique(roles)) == {0, 1}
    assert (cq[roles == 1][:, 2] < P.WALL_Z).all() and (cq[roles == 0][:, 2] > P.WALL_Z - 1e-6).all()
    foot = c.V[c.V[:, 2] == 0.0]
    assert np.abs(foot[:, 0] - np.mean(P.CARC_X)).max() < (P.CARC_X[1] - P.CARC_X[0]) / 2 - (P.ROLL_R + P.COVE_R) + 1e-6


def test_fronts_and_joints_are_closed_solids_with_outward_faces():
    for m in (P.fronts(), P.joints()):
        assert m.is_closed() and m.volume() > 0
    f = P.fronts((0, 1, 2))
    assert set(np.unique(f.Qm)) == {0, 1, 2}
    v = P.front(P.Path(P.outline()), *P.ROWS[1], P.half_wrap(P.Path(P.outline())))
    assert v.is_closed()
    assert v.volume() == pytest.approx(v.area() * P.FRONT_T / 2, rel=0.25)    # a thin shell: V ~ A/2 x t


def test_crown_is_on_the_flat_of_the_front_only():
    """The crown lifts the middle row of the flat front by CROWN (so its face is at y = PED_Y[0], the edges of the front
    3 mm further in) and never the sides, which stay inside x <= -0.28."""
    path = P.Path(P.outline())
    z0, z1 = P.ROWS[0]
    f = P.front(path, z0, z1, P.half_wrap(path))
    zc = (z0 + z1) / 2
    flat = np.abs(f.V[:, 0] - (P.CARC_X[0] + P.CARC_X[1]) / 2) < 0.02            # the flat of the front, x near the middle
    mid_row = flat & (np.abs(f.V[:, 2] - zc) < 0.004)
    edge_row = flat & (f.V[:, 2] > z1 - 0.004)
    assert f.V[:, 1].min() == pytest.approx(P.PED_Y[0], abs=1e-9)
    assert f.V[mid_row, 1].min() < f.V[edge_row, 1].min() - 0.0025
    side = f.V[:, 0] > P.CARC_X[1] - 1e-6
    assert f.V[side, 0].max() == pytest.approx(P.PED_X[1], abs=1e-5)


def test_pedestal_is_not_a_box_for_the_form_metric():
    """Carcass, fronts and joints next to a slab the size of the desk top (the metric's diameter is that of the whole
    prop): no cuboid parts, no sharp-rimmed flat patches, a low score; the old box pedestal scores 1.0."""
    rail = CS.tube(np.array([[-0.6, 0.0, 0.5], [0.6, 0.0, 0.5]]), 0.0125, sides=12)          # sets the prop's diameter
    parts = [rail, P.carcass(), P.fronts(), P.joints()]
    V = np.concatenate([m.V for m in parts])
    off = np.cumsum([0] + [len(m.V) for m in parts[:-1]])
    T = np.concatenate([m.triangles() + o for m, o in zip(parts, off)])
    owner = np.concatenate([np.full(len(m.triangles()), i) for i, m in enumerate(parts)])
    r = F.analyse(V, T, owner, ["rail", "carcass", "fronts", "joints"])
    assert r["score"] < 0.15, r["components"][:3]
    by = r["by_object"]
    assert by["carcass"]["cuboid"] < 0.3 and by["fronts"]["score"] < 0.05 and by["joints"]["score"] < 0.05
    box = CS.rounded_box((0.292, 0.557, 0.65), center=(-0.436, 0.0, 0.38), r=0.003, k=1)
    rb = F.analyse(np.concatenate([rail.V, box.V]),
                   np.concatenate([rail.triangles(), box.triangles() + len(rail.V)]),
                   np.concatenate([np.zeros(len(rail.triangles()), int), np.ones(len(box.triangles()), int)]),
                   ["rail", "box"])
    assert rb["by_object"]["box"]["cuboid"] > 0.9
