"""Pure-numpy geometry of the night sets: primitives face outward, instancing, merged meshes, stations, lamp light,
skyline layout. (Blender-side builders are checked by building and looking; see docs.)"""
import math

import numpy as np
import pytest

from mkmmd.blender.library.sets import nightgeo as G
from mkmmd.blender.library.sets import roadgeo as R


def face_normals(g):
    n = []
    for q in g.Q:
        a, b, c = g.V[q[0]], g.V[q[1]], g.V[q[2]]
        n.append(np.cross(b - a, c - b))
    for t in g.T:
        a, b, c = g.V[t[0]], g.V[t[1]], g.V[t[2]]
        n.append(np.cross(b - a, c - b))
    return np.array(n)


def face_centres(g):
    return np.array([g.V[q].mean(0) for q in g.Q] + [g.V[t].mean(0) for t in g.T])


@pytest.mark.parametrize("make", [lambda: G.box((2, 3, 4)), lambda: G.box((1, 1, 1), base=True),
                                  lambda: G.cylinder(0.5, 0.3, 3.0, 8), lambda: G.cylinder(0.5, 0.5, 2.0, 12, base=False),
                                  lambda: G.sphere(1.5, 10, 6)])
def test_primitives_face_outward(make):
    g = make()
    ctr = g.V.mean(0)
    n, c = face_normals(g), face_centres(g)
    assert (np.einsum("ij,ij->i", n, c - ctr) > 0).all()


def test_box_uv_is_metres_and_base_sits_on_zero():
    g = G.box((2.0, 3.0, 4.0), base=True)
    assert g.V[:, 2].min() == pytest.approx(0.0) and g.V[:, 2].max() == pytest.approx(4.0)
    side = g.UV[:4]                                   # the +X face: u along y (3 m), v up (4 m)
    assert side[:, 0].max() == pytest.approx(3.0) and side[:, 1].max() == pytest.approx(4.0)
    top = g.UV[16:20]                                 # +Z face: x by y
    assert top[:, 0].max() == pytest.approx(2.0) and top[:, 1].max() == pytest.approx(3.0)


def test_boxes_many_sizes():
    g = G.boxes([(1, 1, 1), (2, 2, 5)], base=True)
    assert len(g.V) == 48 and len(g.Q) == 12
    assert g.V[24:, 2].max() == pytest.approx(5.0)
    assert g.inst.tolist() == [0] * 24 + [1] * 24


def test_instance_transform_and_indices():
    g = G.cylinder(0.1, 0.1, 1.0, 6)
    R = G.rot_z([0.0, math.pi / 2])
    t = np.array([[0, 0, 0], [10, 0, 0]])
    h = G.instance(g, R, t)
    assert len(h.V) == 2 * len(g.V) and h.Q.max() < len(h.V)
    assert h.T.max() < len(h.V)
    # a point at +x in the unrotated copy goes to +y when rotated a quarter turn
    i = int(np.argmax(g.V[:, 0]))
    assert h.V[len(g.V) + i] - np.array([10, 0, 0]) == pytest.approx([g.V[i, 1], g.V[i, 0], g.V[i, 2]], abs=1e-9)


def test_instance_scale_and_frame_from_z():
    g = G.box((1, 1, 1), base=True)
    R = G.frame_from_z([[1.0, 0.0, 0.0]])
    assert R[0][:, 2] == pytest.approx([1, 0, 0])                  # local +Z now points along +X
    assert np.linalg.det(R[0]) == pytest.approx(1.0)
    h = G.instance(g, R, [[0, 0, 0]], scale=[[1, 1, 3.0]])
    assert h.V[:, 0].max() == pytest.approx(3.0) and h.V[:, 0].min() == pytest.approx(0.0)


def test_bars_span_endpoints():
    g = G.bars([[0, 0, 0], [0, 0, 2]], [[4, 0, 0], [0, 3, 2]], 0.1, 0.2)
    assert g.V[:24, 0].max() == pytest.approx(4.0, abs=0.11)
    assert g.V[24:, 1].max() == pytest.approx(3.0, abs=0.11)


def test_place_moves_each_block_separately():
    g = G.boxes([(1, 1, 1), (1, 1, 1)], base=True)
    h = G.place(g, G.rot_z([0.0, math.pi / 2]), [[0, 0, 0], [10, 0, 0]])
    assert h.V[:24, 0].max() == pytest.approx(0.5) and h.V[24:, 0].mean() == pytest.approx(10.0)


def test_mesh_merge_offsets_materials_and_attrs():
    m = G.Mesh()
    m.add(G.box((1, 1, 1)), mat=2, a=1.0)
    m.add(G.cylinder(0.5, 0.5, 1.0, 6), mat=(3, 4), smooth=True, a=np.arange(len(G.cylinder(0.5, 0.5, 1.0, 6).V)) * 0.0 + 2.0,
          rgb=np.ones((len(G.cylinder(0.5, 0.5, 1.0, 6).V), 3)))
    a = m.arrays()
    assert len(a["V"]) == 24 + 6 * 2 + 2 * 7
    assert a["Q"].max() < len(a["V"]) and a["T"].max() < len(a["V"])
    assert (a["QM"][:6] == 2).all() and (a["QM"][6:] == 3).all() and (a["TM"] == 4).all()
    assert a["QS"][:6].sum() == 0 and a["QS"][6:].all()
    assert a["attrs"]["a"][:24].tolist() == [1.0] * 24 and a["attrs"]["a"][24] == 2.0
    assert a["attrs"]["rgb"].shape == (len(a["V"]), 3) and a["attrs"]["rgb"][0].tolist() == [0, 0, 0]
    with pytest.raises(ValueError):
        m.add(G.box(), bad=np.zeros(3))


def test_grid_faces_up_and_flip():
    s = np.linspace(0, 10, 6)
    lat = np.array([-2.0, 0.0, 2.0])
    V = np.stack([np.broadcast_to(s[:, None], (6, 3)), np.broadcast_to(lat[None], (6, 3)), np.zeros((6, 3))], -1)
    m = G.Mesh()
    m.add_grid(V, mat=np.arange(5 * 2).reshape(5, 2))
    a = m.arrays()
    assert a["Q"].shape == (10, 4) and a["QM"].tolist() == list(range(10))
    q = a["Q"][0]
    nrm = np.cross(a["V"][q[1]] - a["V"][q[0]], a["V"][q[2]] - a["V"][q[1]])
    assert nrm[2] > 0                                              # e_i = +x, e_j = +y -> +z
    m2 = G.Mesh()
    m2.add_grid(V, flip=True)
    q = m2.arrays()["Q"][0]
    assert np.cross(m2.arrays()["V"][q[1]] - m2.arrays()["V"][q[0]], m2.arrays()["V"][q[2]] - m2.arrays()["V"][q[1]])[2] < 0


def test_stations_include_extras_and_are_sorted():
    s = G.stations(100.0, 3.0, extra=[10.0, 10.02, 55.5, 400.0])
    assert s[0] == 0.0 and s[-1] == 100.0
    assert (np.diff(s) > 0).all() and np.diff(s).max() <= 3.0 + 1e-6
    assert 55.5 in s and 10.0 in s and 10.02 not in s              # closer than min_gap is merged


def test_dashes():
    st, en = G.dash_edges(0, 40, on=3.0, period=12.0)
    assert en[0] - st[0] == 3.0 and np.diff(st).tolist() == [12.0] * (len(st) - 1)
    assert G.in_dash([1.0, 4.0, 13.0], 3.0, 12.0).tolist() == [True, False, True]


def test_spot_light_peaks_below_the_lamp_and_matches_the_formula():
    lamp = np.array([[0.0, 0.0, 9.0]])
    pts = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [40.0, 0.0, 0.0]])
    n = np.tile([0.0, 0.0, 1.0], (3, 1))
    e = G.spot_light(pts, n, lamp, [0, 0, -1.0], 1000.0)
    assert e[0] == pytest.approx(1000.0 / (4 * math.pi ** 2) / 81.0, rel=1e-9)   # P/(4 pi^2) / d^2, full mask
    assert e[0] > e[1] > e[2]
    assert e[2] == 0.0                                             # outside the 140 degree cone
    far = G.spot_light(pts[:1], n[:1], lamp, [0, 0, -1.0], 1000.0, reach=5.0)
    assert far[0] == 0.0                                           # beyond reach
    assert G.spot_light(pts, n, np.zeros((0, 3)), [0, 0, -1.0], 1.0).tolist() == [0, 0, 0]


def test_spot_light_is_additive_and_respects_normals():
    pts = np.array([[0.0, 0.0, 0.0]])
    lamps = np.array([[0.0, 0.0, 6.0], [0.0, 0.0, 6.0]])
    one = G.spot_light(pts, [[0, 0, 1.0]], lamps[:1], [0, 0, -1.0], 500.0)
    two = G.spot_light(pts, [[0, 0, 1.0]], lamps, [0, 0, -1.0], 500.0)
    assert two[0] == pytest.approx(2 * one[0])
    assert G.spot_light(pts, [[0, 0, -1.0]], lamps[:1], [0, 0, -1.0], 500.0)[0] == 0.0   # facing away


def test_city_layout_is_seeded_and_in_range():
    a = G.city_layout(80, 3, 120.0, 900.0, 200.0, height=(40, 220), az0=-90.0)
    b = G.city_layout(80, 3, 120.0, 900.0, 200.0, height=(40, 220), az0=-90.0)
    assert np.allclose(a["center"], b["center"]) and np.allclose(a["size"], b["size"])
    r = np.hypot(a["center"][:, 0], a["center"][:, 1])
    assert r.min() >= 800 - 1e-6 and r.max() <= 1000 + 1e-6
    assert a["size"][:, 2].min() >= 40 and a["size"][:, 2].max() <= 220
    ang = np.degrees(np.arctan2(a["center"][:, 1], a["center"][:, 0]))
    assert ang.min() >= -90 - 60.1 and ang.max() <= -90 + 60.1
    mid = np.abs(ang + 90) < 20
    assert a["size"][mid, 2].mean() > a["size"][~mid, 2].mean()    # taller towards the middle (core)
    c = G.city_layout(80, 4, 120.0, 900.0, 200.0)
    assert not np.allclose(a["center"], c["center"])


# ------------------------------------------------------------------------------------------------ roadgeo


def test_divided_road_layout_and_lane_names():
    sec = R.cross_section(2)                                      # 2 + 2, divided, drive on the right
    assert sec.divided and sec.median == 3.0
    assert sec.x_min == pytest.approx(-sec.x_max)
    lanes = {ln["name"]: ln for ln in sec.lanes}
    assert sorted(lanes) == ["fwd1", "fwd2", "opp1", "opp2"]
    assert lanes["fwd1"]["offset"] == pytest.approx(-4.2) and lanes["fwd2"]["offset"] == pytest.approx(-7.8)
    assert lanes["opp1"]["offset"] == pytest.approx(4.2) and lanes["fwd1"]["dir"] == 1 and lanes["opp1"]["dir"] == -1
    # bands are contiguous from the right edge to the left edge
    xs = [(b["x0"], b["x1"]) for b in sec.bands]
    assert xs[0][0] == pytest.approx(sec.x_min) and xs[-1][1] == pytest.approx(sec.x_max)
    assert all(a[1] == pytest.approx(b[0]) for a, b in zip(xs, xs[1:]))
    assert sec.width == pytest.approx(2 * (1.5 + 0.9 + 2 * 3.6 + 2.5))


def test_paint_is_inside_the_road_and_dashes_only_between_same_direction_lanes():
    sec = R.cross_section(3)
    kinds = [b["kind"] for b in sec.bands]
    assert kinds.count("yellow") == 2 and kinds.count("white") == 2 * 2 + 2   # 2 dividers per side + 2 edge lines
    dashed = [b for b in sec.bands if b["dash"]]
    assert len(dashed) == 4 and all(b["kind"] == "white" and b["under"] == "lane" for b in dashed)
    for b in sec.bands:
        assert b["x1"] > b["x0"]


def test_undivided_road_has_a_double_yellow_centre_and_left_drive_mirrors():
    sec = R.cross_section(1, divided=False)
    assert not sec.divided and sec.median == 0.0
    assert [b["kind"] for b in sec.bands].count("yellow") == 2
    left = R.cross_section(2, drive="left")
    f1 = next(ln for ln in left.lanes if ln["name"] == "fwd1")
    assert f1["offset"] == pytest.approx(4.2) and f1["dir"] == 1                # with the path, on its left
    assert left.x_min == pytest.approx(-left.x_max)


def test_one_way_road_is_centred_on_the_path():
    sec = R.cross_section(3, oncoming=0)
    assert not sec.divided and len(sec.lanes) == 3
    assert sec.x_min == pytest.approx(-sec.x_max)
    offs = sorted(ln["offset"] for ln in sec.lanes)
    assert np.diff(offs).tolist() == pytest.approx([3.6, 3.6])
    assert [b["kind"] for b in sec.bands].count("shoulder") == 2


def test_band_material_alternates_along_dashes():
    sec = R.cross_section(2)
    b = next(b for b in sec.bands if b["dash"])
    mats = R.band_material(b, [1.0, 5.0, 13.0, 17.0])
    assert mats.tolist() == [R.WHITE, R.ASPHALT, R.WHITE, R.ASPHALT]
    solid = next(b for b in sec.bands if b["kind"] == "yellow")
    assert R.band_material(solid, [1.0, 5.0]).tolist() == [R.YELLOW, R.YELLOW]


def test_lamp_layouts():
    div, und = R.cross_section(2), R.cross_section(1, divided=False)
    m = R.lamp_layout(div, 400.0, "auto", spacing=40.0, first=20.0)
    assert m["layout"] == "median" and len(m["s"]) == 2 * len(m["pole_s"]) and set(m["sign"]) == {1.0, -1.0}
    assert m["pole_s"][0] == 20.0 and np.diff(m["pole_s"]).tolist() == [40.0] * (len(m["pole_s"]) - 1)
    o = R.lamp_layout(und, 400.0, "auto", spacing=40.0)
    assert o["layout"] == "outer" and len(o["s"]) == len(o["pole_s"]) == 2 * len(np.unique(o["pole_s"]))
    # an outer arm points toward the road: right-hand pole (negative x) reaches left (+1)
    assert (o["sign"] == -np.sign(o["x"])).all()
    st = R.lamp_layout(und, 400.0, "stagger", spacing=40.0)
    assert len(st["s"]) == len(st["pole_s"]) and set(np.sign(st["x"])) == {1.0, -1.0}
    assert R.lamp_layout(div, 400.0, "right", spacing=40.0)["x"].min() < div.x_min
    with pytest.raises(ValueError):
        R.lamp_layout(div, 400.0, "nonsense")


def test_lamps_skip_tunnels():
    sec = R.cross_section(2)
    m = R.lamp_layout(sec, 600.0, "median", spacing=40.0, first=20.0, skip=[(200.0, 300.0)], margin=12.0)
    assert not ((m["pole_s"] > 188.0) & (m["pole_s"] < 312.0)).any()
    assert 180.0 in m["pole_s"] or 140.0 in m["pole_s"]


def test_real_light_mask_every_window_and_cap():
    s = np.arange(20.0, 1000.0, 40.0)
    assert R.real_light_mask(s, 0).sum() == 0
    assert R.real_light_mask(s, 4).sum() == len(s[::4])
    w = R.real_light_mask(s, 1, window=(100.0, 300.0))
    assert s[w].min() >= 100 and s[w].max() <= 300
    assert R.real_light_mask(s, 1, max_lights=5).sum() == 5


def test_edge_profile_and_ground_height():
    d = R.edge_offsets(0.6)
    assert d[0] == 0.0 and (np.diff(d) > 0).all()
    dz = R.ground_dz(d, 0.6)
    assert dz[0] == 0.0 and dz[-1] == pytest.approx(-0.6) and (np.diff(dz) <= 0).all()


def test_tube_profile_arch_and_rungs():
    p = R.tube_profile(5.0, -5.0, wall_h=4.0, crown_h=6.5, rungs=(1.0, 2.0))
    assert p[0].tolist() == [5.0, 0.0] and p[-1].tolist() == [-5.0, 0.0]
    assert p[:, 1].max() == pytest.approx(6.5) and p[np.argmax(p[:, 1]), 0] == pytest.approx(0.0, abs=1e-9)
    left = p[p[:, 0] == 5.0]
    assert left[:, 1].tolist()[:3] == [0.0, 1.0, 2.0]


def test_facade_covers_outline_minus_openings():
    tubes = [R.tube_profile(10.0, 1.5, 3.9, 6.4, rungs=()), R.tube_profile(-1.5, -10.0, 3.9, 6.4, rungs=())]
    mp = R.mound_profile(10.0, -10.0, 6.4, -0.6)
    q = R.facade(tubes, mp, -0.6)
    assert q.ndim == 3 and q.shape[1:] == (4, 2)
    xs = q[..., 0]
    assert xs.max() == pytest.approx(mp[:, 0].max()) and xs.min() == pytest.approx(mp[:, 0].min())
    # no facade quad inside an opening: at x = 5.75 (middle of the left tube) the wall starts at the arch crown
    mid = [qq for qq in q if qq[:, 0].min() <= 5.75 <= qq[:, 0].max()]
    assert mid and all(qq[:, 1].min() > 5.0 for qq in mid)
    # between the tubes the wall reaches the ground
    pier = [qq for qq in q if qq[:, 0].min() <= 0.0 <= qq[:, 0].max()]
    assert pier and min(qq[:, 1].min() for qq in pier) == pytest.approx(-0.6)
    # area check: the facade is the outline polygon minus the two tube areas
    def area(quad):
        x, z = quad[:, 0], quad[:, 1]
        return 0.5 * abs(np.dot(x, np.roll(z, -1)) - np.dot(z, np.roll(x, -1)))
    total = sum(area(qq) for qq in q)
    poly = np.vstack([mp, mp[:1]])
    outline = 0.5 * abs(np.dot(poly[:-1, 0], poly[1:, 1]) - np.dot(poly[:-1, 1], poly[1:, 0]))
    assert 0.0 < total < outline
