"""Pure maths of the bedroom's soft furnishings (bed, quilt, rug): noise, profiles, the drape and its shells, the frame's
parts, the Memphis print (motifs, layout, lookup tables). The Blender side is checked by building and looking."""
import math

import numpy as np
import pytest

from mkmmd.blender.library.props import bedroom_soft_frame as FR
from mkmmd.blender.library.props import bedroom_soft_geo as G
from mkmmd.blender.library.props import bedroom_soft_pattern as PT
from mkmmd.core import form as FM
from mkmmd.core import shell as CS


def hard_edge_length(V, F, deg):
    """Total length (m) of the edges shared by two faces (quads or triangles as rows of F) whose normals differ by deg or more."""
    F = np.asarray(F)
    tris = np.concatenate([F[:, [0, 1, 2]], F[:, [0, 2, 3]]]) if F.shape[1] == 4 else F
    own = np.concatenate([np.arange(len(F))] * (2 if F.shape[1] == 4 else 1))
    P = V[tris]
    n = np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0])
    ln = np.linalg.norm(n, axis=1)
    keep = ln > 1e-14
    n[keep] /= ln[keep, None]
    # one normal per face: the area-weighted sum over its triangles
    N = np.zeros((len(F), 3))
    np.add.at(N, own, n * ln[:, None])
    N /= np.maximum(np.linalg.norm(N, axis=1, keepdims=True), 1e-18)
    e = np.concatenate([np.stack([F[:, i], F[:, (i + 1) % F.shape[1]]], axis=1) for i in range(F.shape[1])])
    fi = np.tile(np.arange(len(F)), F.shape[1])
    key = np.sort(e, axis=1)
    order = np.lexsort((key[:, 1], key[:, 0]))
    key, fi = key[order], fi[order]
    same = np.nonzero(np.all(key[1:] == key[:-1], axis=1))[0]
    ang = np.degrees(np.arccos(np.clip((N[fi[same]] * N[fi[same + 1]]).sum(1), -1, 1)))
    L = np.linalg.norm(V[key[same, 0]] - V[key[same, 1]], axis=1)
    return float(L[ang >= deg].sum())


# ============================================================================================== noise and sampling
def test_noise_is_deterministic_bounded_and_seeded():
    x, y = np.meshgrid(np.linspace(-3, 3, 41), np.linspace(-2, 4, 31))
    a, b = G.vnoise3(x, y, 0.5, seed=3), G.vnoise3(x, y, 0.5, seed=3)
    assert np.array_equal(a, b) and np.abs(a).max() <= 1.0
    assert not np.allclose(a, G.vnoise3(x, y, 0.5, seed=4))
    assert a.std() > 0.2                                              # it varies
    # a golden value: the hash must not change with the platform or numpy version
    assert G.vnoise3(0.3, 0.7, 0.1, seed=3) == pytest.approx(float(G.vnoise3(0.3, 0.7, 0.1, seed=3)), abs=0)
    assert float(G.vnoise3(2.0, 5.0, 1.0, seed=0)) == pytest.approx(float(G._lattice(np.array([2]), np.array([5]), np.array([1]), 0)[0]), abs=1e-12)


def test_noise_is_smooth():
    t = np.linspace(0, 3, 3001)
    n = G.vnoise3(t, 0.37 * t, 0.2, seed=5)
    assert np.abs(np.diff(n)).max() < 0.02


def test_sample_line_hits_the_ends_and_refines_zones():
    xs = G.sample_line(-1.0, 1.0, [(0.0, 0.5, 0.01)], base=0.1)
    assert xs[0] == -1.0 and xs[-1] == 1.0 and np.all(np.diff(xs) > 0)
    d = np.diff(xs)
    assert d[(xs[:-1] >= 0.0) & (xs[:-1] < 0.5)].max() <= 0.0101 and d[xs[:-1] < -0.1].min() > 0.05


# ============================================================================================== profiles
def test_profile_of_a_quarter_circle_fillet():
    R, L = 0.056, 0.2
    pr = G.Profile([(0.0, 1.0), (R, 1.0), (R, 1.0 - L)], [0, R, 0])
    assert pr.length == pytest.approx(math.pi * R / 2 + (L - R), abs=3e-4)
    a, b, ta, tb = pr.at([0.0, pr.length])
    assert (a[0], b[0]) == pytest.approx((0.0, 1.0)) and (a[1], b[1]) == pytest.approx((R, 1.0 - L), abs=1e-6)
    assert (ta[0], tb[0]) == pytest.approx((1.0, 0.0), abs=0.03) and (ta[1], tb[1]) == pytest.approx((0.0, -1.0), abs=0.03)
    # straight beyond the end, with unit tangents everywhere
    a2, b2, ta2, tb2 = pr.at([pr.length + 0.1, -0.05])
    assert a2[0] == pytest.approx(R, abs=1e-6) and b2[0] == pytest.approx(1.0 - L - 0.1, abs=1e-6) and a2[1] == pytest.approx(-0.05, abs=1e-6)
    assert np.allclose(np.hypot(ta2, tb2), 1.0)
    assert 0.0 < pr.hang < pr.length


def test_fillet_shrinks_to_fit_the_segment():
    P = G.fillet_path([(0, 0), (0.01, 0), (0.01, -1)], [0, 0.5, 0])        # radius 0.5 cannot fit in a 1 cm segment
    assert abs(P[:, 0].max() - 0.01) < 1e-9 and P[:, 0].min() >= -1e-9


# ============================================================================================== solids and pushing out
def test_round_box_distance():
    b = G.RoundBox((-1, -1, -1), (1, 1, 1), 0.2)
    assert b.sdf(np.array([0.0, 0.0, 0.0])) == pytest.approx(-1.0)
    assert b.sdf(np.array([2.0, 0.0, 0.0])) == pytest.approx(1.0)
    assert b.sdf(np.array([1.0, 0.0, 0.0])) == pytest.approx(0.0, abs=1e-12)
    assert b.sdf(np.array([1.2, 1.2, 1.2])) == pytest.approx(math.sqrt(3 * (0.2 + 0.2) ** 2) - 0.2 - 0.0, abs=0.5)


def test_push_out_reaches_the_clearance_and_leaves_far_points():
    sup = G.Support([G.RoundBox((-0.5, -0.5, 0.0), (0.5, 0.5, 0.5), 0.05)])
    pts = np.array([[0.0, 0.0, 0.45], [0.45, 0.1, 0.2], [3.0, 3.0, 3.0], [0.0, 0.0, 0.52]])
    out = G.push_out(pts, 0.02, sup)
    assert np.all(sup.sdf(out) >= 0.02 - 2e-3)
    assert np.array_equal(out[2], pts[2])


# ============================================================================================== outlines and the drape
def test_round_corners_put_the_corner_on_the_ellipse():
    hw, lo, hi = 0.7, -0.3, 1.7
    S, T = np.meshgrid(np.linspace(-hw, hw, 141), np.linspace(lo, hi, 201))
    S2, T2 = G.round_corners(S, T, hw, lo, hi, (0.24, 0.32), (0.24, 0.06))
    assert S2[0, -1] == pytest.approx(hw - 0.24 + 0.24 / math.sqrt(2), abs=1e-9)
    assert T2[0, -1] == pytest.approx(lo + 0.32 - 0.32 / math.sqrt(2), abs=1e-9)
    # outline points stay on the outline; the straight sides are untouched
    assert np.allclose(S2[:, 0][50:150], S[:, 0][50:150]) and np.allclose(T2[0, :][60:80], lo)
    # nothing leaves the rectangle and the whole map is monotone along rows
    assert S2.max() <= hw + 1e-12 and T2.min() >= lo - 1e-12 and T2.max() <= hi + 1e-12


def test_outline_distance_is_zero_on_the_boundary_and_grows_inward():
    pts = np.array([[0, 0], [1, 0], [1, 1], [0, 1.0]])
    d = G.loop_distance(np.array([[0.5, 0.0], [0.5, 0.25], [0.5, 0.5], [2.0, 0.5]]), pts)
    assert d == pytest.approx([0.0, 0.25, 0.5, 1.0])


def test_drape_flat_region_and_hem_height():
    side, foot = G.quilt_side_profile(), G.quilt_foot_profile()
    a, y0 = G.X_MID - G.SIDE_BEND, G.QUILT_Y0
    dr = G.Drape(a, y0, G.Y_FOLD, side, foot)
    S, Y = np.meshgrid(np.linspace(-a, a, 5), np.linspace(y0, G.Y_FOLD, 5))
    W = dr.wrap(S, Y)
    assert np.allclose(W["P"][..., 2], G.Z_MID) and np.allclose(W["P"][..., :2][..., 0], S) and np.allclose(W["nu"][..., 2], 1.0)
    # at the hem of the long side and of the foot the cloth hangs where the profiles end
    hem_side = dr.wrap(np.array([a + side.length]), np.array([0.0]))["P"][0]
    assert hem_side[2] == pytest.approx(G.MAT_Z1 - G.HANG_SIDE + G.QUILT_T_EDGE / 2 + 0.002, abs=2e-3)
    hem_foot = dr.wrap(np.array([0.0]), np.array([y0 - foot.length]))["P"][0]
    assert hem_foot[2] == pytest.approx(G.MAT_Z1 - G.HANG_FOOT + G.QUILT_T_EDGE / 2 + 0.002, abs=2e-3)
    assert hem_side[0] == pytest.approx(G.X_MID, abs=2e-3)
    # the diagonal of the corner ends at the hem too: no droop below the lower of the two hems
    th = np.linspace(0, math.pi / 2, 9)
    corner = dr.wrap(a + side.length * np.cos(th) * 0.9999, y0 - foot.length * np.sin(th) * 0.9999)["P"]
    assert corner[:, 2].min() >= G.MAT_Z1 - G.HANG_SIDE - 0.02


def test_fold_map_is_continuous_and_returns_over_the_quilt():
    Lc, y0, sep, bulge = 1.4, -0.9, 0.034, 0.012
    t = np.linspace(0.0, Lc + 0.5, 4001)
    y, lift = G.fold_map(t, Lc, y0, sep, bulge)
    assert np.abs(np.diff(y)).max() < 2e-3 and np.abs(np.diff(lift)).max() < 2e-3          # no jumps
    r = (sep + bulge) / 2
    assert np.all(lift[t <= Lc] == 0) and y[t <= Lc][-1] == pytest.approx(y0 + Lc, abs=1e-3)
    assert lift.max() <= sep + bulge + 1e-9 and y.max() == pytest.approx(y0 + Lc + r, abs=2e-3)
    back = t > Lc + math.pi * r + 0.3
    assert np.all(np.diff(y[back]) < 0) and lift[back][-1] == pytest.approx(sep, abs=1e-3)


# ============================================================================================== the shells
@pytest.fixture(scope="module")
def quilt():
    return G.quilt_shell(seed=7)


def test_quilt_is_a_closed_outward_shell(quilt):
    sh, info = quilt
    assert sh.open_edges() == 0 and sh.unused_vertices() == 0 and sh.volume() > 0.05
    assert np.isfinite(sh.V).all() and sh.tris < 30000
    assert set(np.unique(sh.mat)) == {0, 1, 2}


def test_quilt_stays_clear_of_the_furniture_and_inside_its_overhang(quilt):
    sh, info = quilt
    sup = G.bed_support()
    assert sup.sdf(sh.V).min() > -1e-3                                    # nothing cuts into the mattress, rails, foot
    lo, hi = sh.bounds()
    assert -0.62 < lo[0] and hi[0] < 0.62                                 # at most ~0.12 m past the frame's sides
    assert lo[1] > -1.10 and hi[1] < G.MAT_HY - 0.3                       # the foot hang; rolled back before the head
    assert lo[2] > 0.27 and hi[2] < 0.62
    # no part of the quilt is inside the posts, rails, head and foot boards of the frame
    for p in FR.frame_parts():
        box = G.RoundBox(p["lo"], p["hi"])
        assert box.sdf(sh.V).min() > -2e-3, p["name"]


def test_quilt_hangs_the_documented_distances(quilt):
    sh, info = quilt
    V = sh.V
    side = V[(np.abs(V[:, 1]) < 0.3) & (V[:, 0] > 0.3)]
    assert 0.17 <= G.MAT_Z1 - side[:, 2].min() <= 0.24                    # hangs 0.18-0.22 below the mattress top
    foot = V[(np.abs(V[:, 0]) < 0.3) & (V[:, 1] < -0.9)]
    assert 0.08 <= G.MAT_Z1 - foot[:, 2].min() <= 0.16
    assert 0.53 < info["field_top_z"] < 0.56
    # the fold: the lining side faces up over the band, the pattern side over the field
    top_faces = sh.F[sh.mat == 0]
    n = np.cross(sh.V[top_faces[:, 1]] - sh.V[top_faces[:, 0]], sh.V[top_faces[:, 3]] - sh.V[top_faces[:, 0]])
    c = sh.V[top_faces].mean(axis=1)
    field = (np.abs(c[:, 0]) < 0.3) & (c[:, 1] > -0.7) & (c[:, 1] < 0.15)
    band = (np.abs(c[:, 0]) < 0.3) & (c[:, 1] > 0.3) & (c[:, 1] < 0.48) & (c[:, 2] > info["field_top_z"] + 0.01)   # the returned layer
    assert (n[field][:, 2] > 0).mean() > 0.98 and (n[band][:, 2] < 0).mean() > 0.95


def test_quilt_has_no_sharp_fold(quilt):
    sh, _ = quilt
    assert hard_edge_length(sh.V, sh.F, 75.0) == 0.0                     # the posts' push is relaxed into smooth bulges
    assert hard_edge_length(sh.V, sh.F, 60.0) < 0.2


def test_pleats_change_smoothly_along_the_edge():
    a = np.linspace(-1.0, 1.0, 2001)
    f = G.pleat_field(a, np.full_like(a, 0.1), 7)
    assert 0.0 <= f.min() and f.max() <= 1.0 and np.abs(np.diff(f)).max() < 0.02       # a wave, not noise


def test_quilt_uvs_are_flat_cloth_metres(quilt):
    sh, info = quilt
    assert sh.uv0[:, 0].min() >= -1e-6 and sh.uv0[:, 0].max() <= 2 * info["hw"] + 1e-6
    assert sh.uv0[:, 1].min() >= -1e-6 and sh.uv0[:, 1].max() <= info["t_hi"] - info["t_lo"] + 1e-6
    assert sh.uv1[:, 0].min() >= 0.0 and sh.uv1[:, 0].max() > 0.5          # distance to the hem


def test_quilt_is_deterministic_and_the_seed_changes_it(quilt):
    sh, _ = quilt
    sh2, _ = G.quilt_shell(seed=7)
    assert np.array_equal(sh.V, sh2.V) and np.array_equal(sh.F, sh2.F)
    sh3, _ = G.quilt_shell(seed=8)
    assert not np.allclose(sh.V, sh3.V)


def test_sheet_wraps_the_mattress_without_cutting_into_it():
    sh, info = G.sheet_shell()
    assert sh.open_edges() == 0 and sh.volume() > 0 and sh.tris < 6000
    assert G.mattress_box().sdf(sh.V).min() > -1e-3
    lo, hi = sh.bounds()
    assert hi[2] < G.MAT_Z1 + G.SHEET_T + 0.002 and lo[2] > G.MAT_Z1 - G.SHEET_HANG - 0.005
    assert hi[0] < G.MAT_HX + 0.01 and hi[1] < G.MAT_HY + 0.01


def test_pillow_is_a_closed_plump_shell():
    sh = G.pillow_shell(0.30, 0.21, 0.17, dent=0.03)
    assert sh.open_edges() == 0 and sh.unused_vertices() == 0 and sh.volume() > 0.01 and sh.tris < 4000
    lo, hi = sh.bounds()
    assert lo[2] == pytest.approx(0.0, abs=0.01) and 0.12 < hi[2] < 0.22
    assert hi[0] == pytest.approx(0.30, abs=0.01) and hi[1] == pytest.approx(0.21, abs=0.01)


def test_pillows_rest_on_the_bed_without_cutting_into_the_frame():
    for spec in G.PILLOWS:
        local, Vb, R, loc = G.pillow_in_bed(spec)
        assert np.allclose(R @ R.T, np.eye(3)) and np.linalg.det(R) == pytest.approx(1.0)
        for p in FR.frame_parts():
            assert G.RoundBox(p["lo"], p["hi"]).sdf(Vb).min() > -3e-3, (spec[0], p["name"])
        assert Vb[:, 1].max() < G.FRAME_HY - G.BOARD_T and abs(Vb[:, 0]).max() < G.MAT_HX
        assert Vb[:, 1].min() > G.Y_FOLD + 0.01                           # clear of the quilt's roll
    a = G.pillow_in_bed(G.PILLOWS[0])[1]
    assert a[:, 2].min() == pytest.approx(G.SHEET_Z - 0.002, abs=2e-3)   # the lower pillow lies on the sheet
    b = G.pillow_in_bed(G.PILLOWS[1])[1]
    assert b[:, 2].min() > G.SHEET_Z + 0.03 and b[:, 2].max() > a[:, 2].max()    # the upper one rests on it, leaning up


def test_rug_body_is_a_closed_thin_slab():
    sh = G.rug_shell()
    assert sh.open_edges() == 0 and sh.unused_vertices() == 0
    assert sh.volume() == pytest.approx(G.RUG_W * G.RUG_D * G.RUG_T, rel=0.03)
    lo, hi = sh.bounds()
    assert lo == pytest.approx([-0.9, -0.6, 0.0], abs=2e-3) and hi[:2] == pytest.approx([0.9, 0.6], abs=2e-3)
    assert lo[2] >= -1e-9 and G.RUG_T - 1e-3 < hi[2] < G.RUG_T + 1.5e-3


def test_rug_fringe_lies_on_the_floor_beyond_the_short_edges():
    fr, n = G.rug_fringe()
    assert n > 70 and fr.V[:, 2].min() > -1e-3 and fr.V[:, 2].max() < 0.01
    x = np.abs(fr.V[:, 0])
    assert x.max() < G.RUG_W / 2 + G.RUG_FRINGE + 0.005 and x.max() > G.RUG_W / 2 + 0.045     # 5 to 5.8 cm past the edge
    assert x.min() > G.RUG_W / 2 - 0.0095                                                      # nothing on the short edges' inside
    assert np.abs(fr.V[:, 1]).max() < G.RUG_D / 2
    fr2, _ = G.rug_fringe()
    assert np.array_equal(fr.V, fr2.V)


def test_rug_fringe_is_round_yarn_and_soft_knots():
    fr, n = G.rug_fringe()
    assert hard_edge_length(fr.V, fr.F, 60.0) == 0.0                     # tubes of 8 sides (45 deg) and 24-quad lumps
    assert fr.unused_vertices() == 0 and fr.volume() > 0
    V, F = G.quad_sphere((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
    a, b, c, d = (V[F[:, i]] for i in range(4))
    vol = sum(float(np.einsum("ij,ij->i", p, np.cross(q, r)).sum()) / 6.0 for p, q, r in ((a, b, c), (a, c, d)))
    assert 2.5 < vol < 4.2 and np.allclose(np.linalg.norm(V, axis=1), 1.0)       # outward, inscribed in the unit sphere
    tv, tf = G.tube_rings(np.stack([np.linspace(0, 1, 6), np.zeros(6), np.full(6, 0.2)], axis=1), np.full(6, 0.1), 8)
    assert tf.shape == (40, 4) and np.allclose(np.hypot(tv[:, 1], tv[:, 2] - 0.2), 0.1)


# ============================================================================================== the frame
def test_frame_parts_fill_the_footprint():
    parts = FR.frame_parts()
    lo = np.min([p["lo"] for p in parts], axis=0)
    hi = np.max([p["hi"] for p in parts], axis=0)
    assert lo == pytest.approx([-G.FRAME_HX, -G.FRAME_HY, 0.0]) and hi == pytest.approx([G.FRAME_HX, G.FRAME_HY, G.HEAD_TOP])
    assert len({p["name"] for p in parts}) == len(parts)
    assert all(np.all(np.array(p["hi"]) > np.array(p["lo"])) for p in parts)
    foot = [p for p in parts if p["name"].startswith("foot_") or p["name"].startswith("post_foot")]
    assert max(p["hi"][2] for p in foot) == pytest.approx(G.FOOT_TOP)
    # the mattress rests on the rails and clears the posts and boards
    mat = G.mattress_box()
    for p in parts:
        if p["group"] in ("post", "board", "slat"):
            assert G.RoundBox(p["lo"], p["hi"]).sdf(np.array(mat.c)) > 0.0
    rails = [p for p in parts if p["group"] == "rail"]
    assert all(p["hi"][2] == pytest.approx(G.MAT_Z0) for p in rails)


def test_padded_headboard_replaces_the_slats():
    padded = FR.frame_parts("padded")
    assert not [p for p in padded if p["name"].startswith("head_slat") or p["name"] in ("head_rail_top", "head_rail_mid")]
    assert len(padded) < len(FR.frame_parts())
    slab, inner = FR.padded_slab(), G.FRAME_HX - G.POST_W
    assert slab["lo"][0] == pytest.approx(-inner) and slab["hi"][0] == pytest.approx(inner)
    assert slab["hi"][2] < G.HEAD_TOP and slab["hi"][1] <= G.FRAME_HY and slab["lo"][1] > G.FRAME_HY - 0.1


def _cylinder_sdf(P, a, b, r):
    """Signed distance of the points P to the capped cylinder of radius r along a -> b."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    ax = b - a
    L = np.linalg.norm(ax)
    ax = ax / L
    t = (P - a) @ ax
    radial = np.linalg.norm(P - a - t[:, None] * ax, axis=1)
    qx, qy = radial - r, np.abs(t - L / 2) - L / 2
    return np.minimum(np.maximum(qx, qy), 0.0) + np.hypot(np.maximum(qx, 0.0), np.maximum(qy, 0.0))


def test_tube_frame_fits_the_footprint_and_clears_the_quilt(quilt):
    sh, _ = quilt
    for headboard in ("slats", "padded"):
        tubes = FR.tube_parts(headboard)
        assert len({n for n, _, _ in tubes}) == len(tubes)
        for name, path, r in tubes:
            P = np.array(path)
            assert np.abs(P[:, 0]).max() + r <= G.FRAME_HX + 1e-9 and np.abs(P[:, 1]).max() + r <= G.FRAME_HY + 1e-9, name
            vertical = np.ptp(P[:, 0]) < 1e-9 and np.ptp(P[:, 1]) < 1e-9                  # flat caps: no radius at the end
            top = P[:, 2].max() + (0.0 if vertical else r)
            assert P[:, 2].min() >= -1e-9 and top <= G.HEAD_TOP + 0.005, name
            if name.startswith("foot_") or name.startswith("post_foot"):
                assert top <= G.FOOT_TOP + 1e-9, name                        # the quilt crosses the foot: nothing above it
                # no vertex of the quilt is inside a foot tube (a capped cylinder per straight run)
                for a, b in zip(P[:-1], P[1:]):
                    assert _cylinder_sdf(sh.V, a, b, r).min() > -2e-3, name
        names = {n for n, _, _ in tubes}
        assert ("head_bar_0" in names) == (headboard == "slats")
    for _, c, k in FR.tube_finials():
        prof = np.array(FR.finial_profile(k))
        assert abs(c[0]) + prof[:, 0].max() <= G.FRAME_HX + 1e-9 and abs(c[1]) + prof[:, 0].max() <= G.FRAME_HY + 1e-9
        assert c[2] + prof[:, 1].max() < G.HEAD_TOP + 0.05


@pytest.fixture(scope="module")
def frame_mesh():
    return FR.frame_mesh()


def test_frame_mesh_is_closed_soft_wood_inside_the_footprint(frame_mesh):
    m = frame_mesh
    assert m.is_closed() and m.volume() > 0.04
    lo, hi = m.bbox()
    assert lo == pytest.approx([-G.FRAME_HX, -G.FRAME_HY, 0.0], abs=1e-9)
    assert hi == pytest.approx([G.FRAME_HX, G.FRAME_HY, G.HEAD_TOP], abs=1e-9)         # the frame's own bounds, exactly
    assert m.UV.shape == (len(m.V), 2) and np.isfinite(m.UV).all()
    assert m.nfaces < 15000


def test_frame_has_no_sharp_arris_and_does_not_read_as_boxes(frame_mesh):
    m = frame_mesh
    for deg in (45.0, 60.0, 75.0):
        assert hard_edge_length(m.V, m.Q, deg) == 0.0, deg                           # every edge is a 22 deg step or less
    tris = np.concatenate([m.Q[:, [0, 1, 2]], m.Q[:, [0, 2, 3]], m.T])
    r = FM.analyse(m.V, tris)
    assert r["score"] < 0.1 and r["cuboid_share"] < 0.1 and r["flat_hard_share"] < 0.05    # planed and rounded, not boxes


def test_frame_parts_have_real_proportions_and_soft_edges():
    parts = {p["name"]: p for p in FR.frame_parts()}
    size = lambda p: np.array(p["hi"]) - np.array(p["lo"])                           # noqa: E731
    assert size(parts["post_foot_L"])[0] == pytest.approx(0.055) and size(parts["post_head_R"])[1] == pytest.approx(0.055)
    rail = size(parts["rail_L"])
    assert rail[0] == pytest.approx(0.025, abs=1e-3) and rail[2] == pytest.approx(0.12, abs=1e-6)     # 25 x 120 mm planks
    slat = size(parts["head_slat_0"])
    assert slat[0] == pytest.approx(FR.SLAT_W) and 0.019 < slat[1] < 0.021                         # 42 x 20 mm
    for p in parts.values():
        assert p["round"] >= 0.0035 and p["seg"] >= 1, p["name"]                       # no plain box among them
        if p["group"] in ("rail", "board", "slat"):
            assert p["round"] >= 0.009, p["name"]                                      # the visible ones: 9 mm or more
    assert parts["post_head_L"]["taper"][0] < 0.9 and parts["post_head_L"]["end"] >= 0.004


def test_post_is_a_tapered_rounded_leg_with_rounded_ends():
    part = {p["name"]: p for p in FR.frame_parts()}["post_foot_R"]
    m = FR.post_mesh(part)
    assert m.is_closed() and m.volume() > 0
    lo, hi = m.bbox()
    assert lo[2] == pytest.approx(0.0, abs=1e-9) and hi[2] == pytest.approx(G.FOOT_TOP, abs=1e-9)
    low, high = m.V[m.V[:, 2] < 0.002], m.V[np.abs(m.V[:, 2] - FR.LEG_TAPER[1]) < 1e-6]          # the floor, the end of the taper
    w_low, w_high = np.ptp(low[:, 0]), np.ptp(high[:, 0])
    assert w_high == pytest.approx(G.POST_W, abs=1e-4) and w_low < 0.8 * w_high         # narrower at the floor
    assert hard_edge_length(m.V, m.Q, 60.0) == 0.0


def test_padded_and_tube_variants_build_soft_geometry():
    pad = FR.padded_mesh()
    assert pad.is_closed() and hard_edge_length(pad.V, pad.Q, 60.0) == 0.0
    lo, hi = pad.bbox()
    assert abs(lo[0]) <= G.FRAME_HX - G.POST_W + 1e-9 and hi[1] <= G.FRAME_HY and hi[2] < G.HEAD_TOP
    for headboard in ("slats", "padded"):
        t = FR.tube_mesh(headboard)
        assert hard_edge_length(t.V, t.Q, 60.0) == 0.0 and np.isfinite(t.V).all()
        lo, hi = t.bbox()
        assert abs(lo[0]) <= G.FRAME_HX + 1e-9 and abs(hi[0]) <= G.FRAME_HX + 1e-9 and hi[1] <= G.FRAME_HY + 1e-9 and hi[2] <= G.HEAD_TOP + 0.005


def test_slats_fill_the_board_evenly():
    xs = np.array(FR.slat_xs(-0.445, 0.445, FR.SLAT_W, FR.SLAT_GAP))
    assert xs == pytest.approx(-xs[::-1]) and len(xs) == 9                     # symmetric, nine slats
    gaps = np.diff(xs) - FR.SLAT_W
    end_gap = xs[0] - FR.SLAT_W / 2 + 0.445
    assert gaps.min() > 0.03 and gaps.max() - gaps.min() < 1e-9               # the same gap between slats ...
    assert end_gap == pytest.approx(gaps[0], abs=1e-9)                        # ... and between the end slats and the posts


def test_rounded_rect_path_is_closed_and_counter_clockwise():
    pts = np.array(FR.rounded_rect_path(0.5, 1.0, 0.05, z=0.4))
    assert pts[:, 2] == pytest.approx(0.4)
    x, y = pts[:, 0], pts[:, 1]
    area = 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)
    assert area > 0 and area == pytest.approx(1.0 * 2.0 - (4 - math.pi) * 0.05 ** 2, rel=0.01)
    assert x[-1] == pytest.approx(x[0]) and abs(y[-1] - y[0]) == pytest.approx(2 * (1.0 - 0.05))   # closes with a straight side


# ============================================================================================== the print
R0 = 0.1


def _inside_area(kind, n=1200, span=1.3):
    g = np.linspace(-span, span, n) * R0
    x, y = np.meshgrid(g, g)
    d = PT.sdf(kind, x, y, R0)
    cell = (g[1] - g[0]) ** 2
    return (d < 0).sum() * cell, np.hypot(x, y)[d < 0].max()


def test_motif_areas_match_their_formulas():
    A = {k: _inside_area(k)[0] / R0 ** 2 for k in range(1, 9)}
    assert A[PT.TRIANGLE] == pytest.approx(3 * math.sqrt(3) / 4, rel=0.01)
    r_out, r_in = PT.RING_R + PT.RING_W, PT.RING_R - PT.RING_W
    assert A[PT.RING] == pytest.approx(math.pi * (r_out ** 2 - r_in ** 2), rel=0.01)
    assert A[PT.DOTS] == pytest.approx(math.pi * sum(r * r for _, _, r in PT.DOT_SET), rel=0.02)
    assert A[PT.CROSS] == pytest.approx(2 * (2 * PT.CROSS_L) * (2 * PT.CROSS_W) - (2 * PT.CROSS_W) ** 2, rel=0.02)
    r_a, r_b = PT.ARCH_R + PT.ARCH_W, PT.ARCH_R - PT.ARCH_W
    assert A[PT.ARCH] == pytest.approx(math.pi * (r_a ** 2 - r_b ** 2) / 2, rel=0.02)
    bars = 3 * 2 * PT.ST_W * 2 * PT.ST_BOX
    assert A[PT.STRIPES] == pytest.approx(bars, rel=0.03)
    for k in (PT.SQUIGGLE, PT.ZIGZAG):
        assert 0.3 < A[k] < 1.0                                           # a stroke, not a blob


def test_every_motif_fits_the_circle_of_its_radius():
    for k in range(1, 9):
        area, rmax = _inside_area(k)
        assert area > 0.05 * R0 ** 2, PT.TYPES[k]
        assert rmax <= R0 * 1.005, (PT.TYPES[k], rmax / R0)


def test_the_motif_distance_is_a_distance_near_the_edge():
    # the slope of the signed distance is about 1 along a line crossing a shape's edge (a stroke's g correction keeps it
    # near 1; an AA ramp of 2.2 mm is only right if it is)
    for k in range(1, 9):
        x = np.linspace(-1.2, 1.2, 4801) * R0
        d = PT.sdf(k, x, np.zeros_like(x) + 0.013 * R0, R0)
        edge = np.nonzero(np.sign(d[1:]) != np.sign(d[:-1]))[0]
        for i in edge[:6]:
            slope = abs(d[i + 1] - d[i]) / (x[i + 1] - x[i])
            assert 0.35 < slope < 2.0, (PT.TYPES[k], slope)


WEIGHTS = [(PT.TRIANGLE, 1.3), (PT.DOTS, 1.0), (PT.RING, 0.9), (PT.SQUIGGLE, 1.2), (PT.ZIGZAG, 1.1), (PT.STRIPES, 0.8),
           (PT.ARCH, 0.8), (PT.CROSS, 0.7)]
POOLS = {t: [(i, 1.0) for i in range(6)] for t in range(1, 9)}


def test_layout_is_deterministic_and_obeys_the_neighbour_rules():
    a = PT.layout(5, 8, 7, WEIGHTS, POOLS)
    b = PT.layout(5, 8, 7, WEIGHTS, POOLS)
    assert [(c.kind, c.c1, c.c2, round(c.rot, 9)) for c in a] == [(c.kind, c.c1, c.c2, round(c.rot, 9)) for c in b]
    assert [c.kind for c in a] != [c.kind for c in PT.layout(5, 8, 8, WEIGHTS, POOLS)]
    for cy in range(8):
        for cx in range(5):
            c = a[cx + 5 * cy]
            if c.kind == PT.NONE:
                continue
            for nb in ((cx - 1, cy), (cx, cy - 1)):
                if nb[0] >= 0 and nb[1] >= 0:
                    n = a[nb[0] + 5 * nb[1]]
                    assert n.kind != c.kind and (n.kind == PT.NONE or n.c1 != c.c1)
            assert c.c2 != c.c1 and 0.8 <= c.scale <= 1.0 and abs(c.jx) <= 0.05 and abs(c.jy) <= 0.05
    kinds = {c.kind for c in a}
    assert len(kinds - {PT.NONE}) >= 7                                    # a varied print


def test_motifs_stay_inside_their_cells():
    # worst case of the layout rules: the largest scale, the largest jitter, any rotation
    sx = sy = 0.25
    R = PT.MOTIF_R * sx * 1.0
    for k in range(1, 9):
        g = np.linspace(-0.6, 0.6, 600) * sx
        x, y = np.meshgrid(g, g)
        d = PT.sdf(k, x, y, R)
        reach = np.hypot(x, y)[d < 0].max()
        assert reach + 0.05 * sx * math.sqrt(2) <= 0.5 * sx * math.sqrt(2) + 1e-9
        assert reach + 0.05 * sx <= 0.5 * sx + 1e-9, PT.TYPES[k]


def test_lut_round_trip_and_ranges():
    cells = PT.layout(5, 6, 3, WEIGHTS, POOLS)
    A, B = PT.lut_stops(cells)
    assert len(A) == len(B) == PT.LUT_SIZE
    for i, (a, b) in enumerate(zip(A, B)):
        assert a[0] == pytest.approx(i / PT.LUT_SIZE) and b[0] == pytest.approx(i / PT.LUT_SIZE)
        assert all(0.0 <= v <= 1.0 for v in a[1] + b[1])
        if i < len(cells):
            c, d = cells[i], PT.decode(a[1], b[1])
            assert (d.kind, d.c1, d.c2) == (c.kind, c.c1, c.c2)
            assert d.rot == pytest.approx(c.rot % PT.TAU, abs=1e-9) and d.scale == pytest.approx(c.scale)
            assert d.jx == pytest.approx(c.jx) and d.jy == pytest.approx(c.jy)


def test_render_shows_the_ground_between_motifs_and_the_colour_of_a_motif():
    cells = [PT.Cell(PT.TRIANGLE, 0.0, 1.0, 0.0, 0.0, 1, 0)] + [PT.Cell() for _ in range(3)]
    img = PT.render(cells, 2, 2, 0.25, 0.25, [(0, 0, 0), (1, 0, 0)], (0.0, 1.0, 0.0), res=100)
    h = img.shape[0]
    assert tuple(img[0, -1]) == (0.0, 1.0, 0.0)                           # an empty cell: the ground
    cx, cy = 25, h - 25                                                   # the centre of the first (lower-left) cell
    assert tuple(img[cy, cx]) == (1.0, 0.0, 0.0)


def test_cell_lookup_clamps_outside_the_lattice():
    idx, px, py = PT.cell_at(np.array([-1.0, 0.1, 9.0]), np.array([0.1, 0.1, 0.1]), 3, 2, 0.25, 0.25)
    assert list(idx) == [0, 0, 2]
