"""Pure-numpy half of the skyline set: layout, building parts, wall geometry, tree line, glow profile. (The shaders
and objects are checked by building and looking; see docs.)"""
import math

import numpy as np
import pytest

from mkmmd.blender.library.sets import skyline as SK

WIN = [np.array([0.9, 0.5, 0.2]), np.array([0.8, 0.3, 0.3]), np.array([0.3, 0.6, 0.7]), np.array([0.8, 0.8, 0.9])]


def city(n=60, seed=3, **kw):
    L = SK.layout(n, seed, **kw)
    rng = np.random.default_rng([seed, 7])
    S = SK.style(L, rng, 0.35, WIN, [6.0, 1.0, 1.2, 0.8], np.array([0.02, 0.02, 0.04]), np.array([0.04, 0.04, 0.08]),
                 [np.array([0.5, 0.3, 0.7])], 1.5)
    P = SK.parts(L, S, rng)
    return L, S, P, rng


def test_rects_overlap():
    c, h = np.array([0.0, 0.0]), np.array([5.0, 5.0])
    C, H, Y = np.array([[13.0, 0.0], [30.0, 0.0]]), np.array([[5.0, 5.0]] * 2), np.zeros(2)     # 3 m and 20 m apart
    assert not SK.rects_overlap(c, h, 0.0, C, H, Y)
    assert SK.rects_overlap(c, h, 0.0, C, H, Y, gap=4.0)                                 # closer than 4 m
    assert SK.rects_overlap(c, h, 0.0, np.array([[6.0, 0.0]]), H[:1], Y[:1])             # overlapping
    assert SK.rects_overlap(c, np.array([5.0, 1.0]), math.pi / 4, np.array([[6.0, 0.0]]), np.array([[1.0, 5.0]]),
                            np.array([math.pi / 4]))                                    # turned: still crossing
    assert not SK.rects_overlap(c, np.array([5.0, 1.0]), 0.0, np.array([[0.0, 8.0]]), np.array([[5.0, 1.0]]),
                                np.array([0.0]))                                         # side by side along y
    assert not SK.rects_overlap(c, h, 0.0, np.zeros((0, 2)), np.zeros((0, 2)), np.zeros(0))


def test_layout_is_seeded_and_in_range():
    a = SK.layout(80, 3, distance=900.0, depth=200.0, arc=100.0, az=-90.0, height=(40.0, 220.0))
    b = SK.layout(80, 3, distance=900.0, depth=200.0, arc=100.0, az=-90.0, height=(40.0, 220.0))
    assert all(np.allclose(a[k], b[k]) for k in a)
    assert not np.allclose(a["x"], SK.layout(80, 4, distance=900.0, depth=200.0, arc=100.0, az=-90.0)["x"])
    assert len(a["x"]) == 80
    r = np.hypot(a["x"], a["y"])
    assert r.min() >= 800 - 1e-6 and r.max() <= 1000 + 1e-6
    assert a["H"].min() >= 8.0 and a["H"].max() <= 220.0 + 1e-6
    ang = np.degrees(np.arctan2(a["y"], a["x"]))
    assert ang.min() >= -90 - 50.01 and ang.max() <= -90 + 50.01


def test_layout_band_runs_across_the_heading():
    L = SK.layout(60, 2, shape="band", az=0.0, width=1000.0, distance=700.0, depth=100.0)
    assert L["x"].min() >= 650 - 1e-6 and L["x"].max() <= 750 + 1e-6              # along the heading (+X)
    assert np.abs(L["y"]).max() <= 500 + 1e-6 and np.abs(L["y"]).max() > 350      # across it
    face = np.array([(math.sin(t), -math.cos(t)) for t in L["yaw"]])              # local -Y: back toward the root
    assert (face[:, 0] < -0.5).all()


def test_layout_keeps_buildings_apart_when_there_is_room():
    L = SK.layout(40, 5, distance=800.0, depth=300.0, footprint=(16.0, 40.0), gap=3.0)
    half = np.stack([L["fw"], L["fd"]], 1) / 2
    for i in range(len(L["x"])):
        rest = np.delete(np.arange(len(L["x"])), i)
        assert not SK.rects_overlap(np.array([L["x"][i], L["y"][i]]), half[i], L["yaw"][i],
                                    np.stack([L["x"][rest], L["y"][rest]], 1), half[rest], L["yaw"][rest], 2.9)


def test_layout_core_puts_the_tall_ones_in_the_middle():
    L = SK.layout(120, 7, core=1.0, arc=110.0, height=(20.0, 170.0))
    mid = np.abs(L["s"]) < 0.2
    assert L["H"][mid].mean() > 1.5 * L["H"][~mid].mean()
    flat = SK.layout(120, 7, core=0.0, arc=110.0, height=(20.0, 170.0))
    assert flat["H"][np.abs(flat["s"]) < 0.2].mean() < 1.5 * flat["H"][np.abs(flat["s"]) >= 0.2].mean()


def test_layout_one_building_and_crowded():
    one = SK.layout(1, 1)
    assert len(one["x"]) == 1
    crowded = SK.layout(150, 1, arc=5.0, depth=30.0, footprint=(30.0, 60.0))        # no room: shrinks, then overlaps
    assert len(crowded["x"]) == 150 and crowded["fw"].min() > 0


def test_style_thresholds_and_ranges():
    L, S, P, _ = city()
    n = len(L["x"])
    assert (np.diff(np.concatenate([S["cth"], np.ones((n, 1))], 1), axis=1) >= -1e-9).all()      # cumulative weights
    assert ((S["pf"] > 0) & (S["pf"] <= 1) & (S["pc"] > 0) & (S["pc"] <= 1)).all()
    assert (S["ww"] > 0.3).all() and (S["ww"] < 1.0).all() and (S["ch"] >= 3.0).all() and (S["ch"] <= 4.2).all()
    dark, light, tint = np.array([0.02, 0.02, 0.04]), np.array([0.04, 0.04, 0.08]), np.array([0.5, 0.3, 0.7])
    assert (S["wall"] >= dark - 1e-9).all() and (S["wall"] <= np.maximum(light, tint) + 1e-9).all()
    assert (S["avg"] > 0).all()


def test_parts_cover_every_building_and_stand_on_the_ground():
    L, S, P, _ = city()
    n = len(L["x"])
    assert set(P["b"].tolist()) == set(range(n))
    assert P["z0"].min() == -SK.SUNK
    assert (P["h"] > 0).all() and (P["w"] > 0).all() and (P["d"] > 0).all()
    tops = P["z0"] + P["h"] + np.abs(P["sl"])
    for i in range(n):
        assert tops[P["b"] == i].max() == pytest.approx(P["top"][i])
    assert (P["top"] >= 0.5 * L["H"]).all()
    # whole floors: the first part of every building ends on a floor line (ground level is z = 0)
    first = np.array([np.where(P["b"] == i)[0][0] for i in range(n)])
    floors = (P["z0"][first] + P["h"][first]) / S["ch"]
    assert np.allclose(floors, np.round(floors))


def test_beacon_tips_are_above_the_roofs():
    L, S, P, _ = city()
    assert (P["tip"][:, 2] > 0.5 * L["H"]).all()
    r = np.hypot(P["tip"][:, 0] - L["x"], P["tip"][:, 1] - L["y"])
    assert (r < L["fw"] + L["fd"]).all()                                          # at the building, in the world frame


def test_facades_are_closed_outward_quads_with_whole_window_columns():
    L, S, P, rng = city()
    V, Q, UV, A = SK.facades(P, S, rng)
    n = len(P["b"])
    assert V.shape == (n * SK.VERTS, 3) and Q.shape == (n * 5, 4) and UV.shape == (n * SK.VERTS, 2)
    assert Q.max() == len(V) - 1
    assert all(len(a) == len(V) for a in A.values())
    Qp = Q.reshape(n, 5, 4)
    for i in range(0, n, 3):
        ctr = V[i * SK.VERTS:(i + 1) * SK.VERTS].mean(0)
        for f in range(5):
            p = V[Qp[i, f]]
            nrm = np.cross(p[1] - p[0], p[3] - p[0])
            if f < 4:
                assert np.dot(nrm, p.mean(0) - ctr) > 0                          # walls look outward
                u = UV[Qp[i, f], 0]
                span = u[1] - u[0]
                assert span >= 1 and abs(span - round(span)) < 1e-9 and u[2] == u[1] and u[3] == u[0]
            else:
                assert nrm[2] > 0 and (UV[Qp[i, f]] == 0).all()                  # roofs look up and carry no grid
    assert (A["bid"][::SK.VERTS] == P["b"]).all()
    assert set(np.unique(A["wm"])) <= {0.0, 1.0}


def test_wall_v_is_height_in_floors():
    L, S, P, rng = city()
    V, Q, UV, A = SK.facades(P, S, rng)
    walls = np.concatenate([np.arange(k * SK.VERTS, k * SK.VERTS + 16) for k in range(len(P["b"]))])
    assert np.allclose(UV[walls, 1] * A["win"][walls, 1], V[walls, 2])


def test_tree_line_is_seeded_bounded_and_bumpy():
    x, h, t = SK.tree_line(np.random.default_rng(1), 800.0, 2.5, (5.0, 16.0), (3.0, 12.0))
    x2, h2, _ = SK.tree_line(np.random.default_rng(1), 800.0, 2.5, (5.0, 16.0), (3.0, 12.0))
    assert np.allclose(h, h2) and len(x) == len(h) == len(t)
    assert h.min() >= 5.0 * 0.35 - 1e-9 and h.max() <= 16.0 * 1.35 + 1e-9
    assert h.std() > 1.0 and ((t >= 0) & (t <= 1)).all()


def test_city_profile_peaks_over_the_buildings():
    L = SK.layout(60, 2, core=1.0)
    s = np.linspace(-0.6, 0.6, 121)
    g, reach = SK.city_profile(L, s, 0.06)
    assert g.max() == pytest.approx(1.0) and g.min() >= 0
    assert abs(s[g.argmax()]) < 0.3
    assert g[np.abs(s) > 0.58].max() < 0.5 * g.max() and reach.max() <= L["H"].max() + 1e-9


def test_frontage_arc_and_band():
    p = SK.frontage("arc", -90.0, 120.0, 0.0, 800.0, np.array([-0.5, 0.0, 0.5]))
    assert np.allclose(np.hypot(p[:, 0], p[:, 1]), 800.0)
    assert p[1] == pytest.approx([0.0, -800.0], abs=1e-9)
    assert math.degrees(math.atan2(p[2, 1], p[2, 0])) == pytest.approx(-30.0)     # counter-clockwise from -90
    q = SK.frontage("band", 0.0, 0.0, 1000.0, 700.0, np.array([-0.5, 0.5]))
    assert q == pytest.approx(np.array([[700.0, -500.0], [700.0, 500.0]]))
