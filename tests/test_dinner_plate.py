"""The dinner plate's layout (mkmmd.blender.library.props.dinner_plate_layout, no bpy): the lathed plate (foot ring, well, the
rim flaring up and out, a rolled lip) closed in the card's frame (its foot on z = 0), the veggies lying on it, and the card's
rest plane and rim pinch read off the profile."""
import math

import numpy as np
import pytest

from mkmmd.blender.library.props import dinner_plate_layout as LAY
from mkmmd.core import shell as S


def _parts(m):
    parent = np.arange(len(m.V))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for f in list(m.Q) + list(m.T):
        r0 = find(int(f[0]))
        for v in f[1:]:
            r = find(int(v))
            if r != r0:
                parent[r] = r0
    roots = np.array([find(i) for i in range(len(m.V))])
    return [np.nonzero(roots == r)[0] for r in np.unique(roots)]


def _sub(m, ids):
    keep = np.zeros(len(m.V), bool)
    keep[ids] = True
    remap = np.cumsum(keep) - 1
    return S.Mesh(m.V[keep], remap[m.Q[keep[m.Q].all(1)]], remap[m.T[keep[m.T].all(1)]])


def _top():
    """(r, z) of the plate's top surface, from the axis out to the lip's outermost point."""
    P = LAY.profile()
    k = int(np.argmax(P[:, 0]))
    return P[k:][::-1]


def test_the_plate_is_a_closed_lathe_on_its_foot_0_27_across():
    m = LAY.plate_mesh()
    assert m.is_closed() and m.volume() > 0
    lo, hi = m.bbox()
    assert lo[2] == pytest.approx(0.0, abs=1e-12) and 0.02 < hi[2] < 0.03
    assert hi[0] - lo[0] == pytest.approx(0.27, abs=1e-3)
    foot = m.V[np.abs(m.V[:, 2]) < 1e-9]
    r = np.hypot(foot[:, 0], foot[:, 1])
    assert LAY.FOOT[0] - 0.002 < r.min() and r.max() < LAY.FOOT[1] + 0.002            # it stands on its foot ring


def test_the_well_is_flat_and_the_rim_flares_up_and_out_to_a_rolled_lip():
    top = _top()
    well = top[top[:, 0] <= LAY.R_WELL]
    assert np.allclose(well[:, 1], LAY.Z_WELL, atol=1e-9)                              # the rest plane IS the floor
    rim = top[(top[:, 0] > LAY.RIM_TOP_IN[0]) & (top[:, 0] < LAY.R_EDGE - 0.006)]
    slope = np.degrees(np.arctan2(np.diff(rim[:, 1]), np.diff(rim[:, 0])))
    assert (slope > 10.0).all() and (slope < 30.0).all()                               # up and out
    P = LAY.profile()
    k = int(np.argmax(P[:, 0]))
    assert P[k, 0] == pytest.approx(LAY.R_EDGE, abs=1e-4)
    lip = P[P[:, 0] > LAY.R_EDGE - 0.004]
    turn = np.degrees(np.unwrap(np.arctan2(np.diff(lip[:, 1]), np.diff(lip[:, 0]))))
    assert turn.max() - turn.min() > 170.0                                             # rolled round, not cut off


def test_every_veggie_is_closed_and_lies_on_the_plate_inside_the_well():
    for seed in (1, 2, 9):
        v = LAY.veggies_mesh(seed)
        for p in _parts(v):
            sub = _sub(v, p)
            assert sub.is_closed() and sub.volume() > 0, seed
        r = np.hypot(v.V[:, 0], v.V[:, 1])
        assert r.max() < LAY.R_WELL + 0.012                                            # in the well
        top = _top()
        assert (v.V[:, 2] >= np.interp(r, top[:, 0], top[:, 1]) - 0.0005).all()        # on it, never through it
        assert v.V[:, 2].min() == pytest.approx(LAY.Z_WELL - 0.0003, abs=1e-9)
    names = {LAY.MATS[i] for i in np.unique(np.concatenate([v.Qm, v.Tm]))}
    assert names == {"floret", "stalk", "coin", "pea", "tomato", "calyx"}
    assert not np.allclose(LAY.veggies_mesh(1).V[:200], LAY.veggies_mesh(2).V[:200])   # the seed moves them


def test_a_coin_s_uv_is_its_share_of_the_radius():
    m = LAY.coin_mesh(0.012, 0.005, 3)
    r = np.hypot(m.V[:, 0], m.V[:, 1])
    assert m.UV[:, 0].min() == 0.0 and m.UV[:, 0].max() == pytest.approx(1.0)
    assert m.UV[np.argmin(r), 0] == 0.0 and m.UV[np.argmax(r), 0] == pytest.approx(1.0)
    assert m.bbox()[0][2] == 0.0 and m.bbox()[1][2] == pytest.approx(0.005)


def test_the_card_s_rest_plane_and_rim_pinch_are_the_plate_s():
    card = LAY.card("pl", LAY.options({})["roles"], [0.27, 0.27, 0.024])
    assert card["origin"] == "floor_center" and card["front"] == "-Y"
    assert card["objects"] == ["pl_plate", "pl_veggies"]
    (well,) = card["use"]["rest"]
    assert well == {"name": "well", "type": "plane", "center": [0.0, 0.0, LAY.Z_WELL], "normal": [0.0, 0.0, 1.0],
                    "radius": LAY.R_WELL}
    assert card["use"]["look"] == [{"name": "centre", "point": [0.0, 0.0, LAY.Z_WELL]}]
    (rim,) = card["use"]["grip"]
    assert rim["name"] == "rim" and rim["type"] == "pinch"
    n, a = np.array(rim["normal"]), np.array(rim["axis"])
    assert np.linalg.norm(n) == pytest.approx(1.0, abs=1e-4) and abs(n @ a) < 1e-9
    assert n[1] > 0.9 and n[2] > 0.2                                                   # out over the back edge, up the rim
    squeeze = np.cross(a, n)
    assert squeeze[2] > 0.9                                                            # thumb over, fingers under
    c = np.array(rim["center"])
    r = np.hypot(c[0], c[1])
    top = _top()
    P = LAY.profile()
    k = int(np.argmax(P[:, 0]))
    under = P[:k + 1]
    zt, zu = np.interp(r, top[:, 0], top[:, 1]), np.interp(r, under[:, 0], under[:, 1])
    assert zu < c[2] < zt and c[2] == pytest.approx(0.5 * (zt + zu), abs=1e-5)         # in the rim's thickness
    assert rim["width"] == pytest.approx((zt - zu) * math.cos(math.asin(n[2])), abs=5e-4)
    assert 0.003 < rim["width"] < 0.007 and rim["span"] > 0 and rim["length"] > 0
    assert r + 0.5 * rim["span"] == pytest.approx(LAY.R_EDGE, abs=1e-6)


def test_options_take_veggies_seed_and_colours():
    o = LAY.options({"veggies": False, "seed": 4, "band": "love", "base": "#000000"})
    assert o["veggies"] is False and o["seed"] == 4 and o["roles"]["band"] == "love"
    assert LAY.card("pl", o["roles"], [0.27, 0.27, 0.024], veggies=False)["objects"] == ["pl_plate"]
    for slots, frag in (({"veggies": "no"}, "veggies"), ({"seed": -1}, "seed"), ({"seed": 1.5}, "seed"),
                        ({"tomato": "red"}, "tomato")):
        with pytest.raises(ValueError, match=frag):
            LAY.options(slots)
