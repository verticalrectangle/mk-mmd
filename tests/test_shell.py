"""The shell toolkit (`mkmmd.core.shell`): lofts, cages, panels and frames, cutters, probes, the creased-cage contract, and the
code blocks of docs/modelling.md. Pure numpy; the smoothness checks evaluate the Subdivision Surface with `mkmmd.model.subdiv`
(the same Catmull-Clark rules as Blender's modifier), so they need scipy and are skipped without it."""
import math
import re
from pathlib import Path

import numpy as np
import pytest

from mkmmd.core import shell as S


# ======================================================================================================== helpers
def key(V):
    return set(map(tuple, np.round(np.asarray(V), 7)))


def box_section(w, z0, z1, zt=None, crease=0.0):
    """A half section of a box: underside, side, top; names keel, bot, mid, top, tc."""
    zt = z1 if zt is None else zt
    return [S.pt("keel", 0.0, z0), S.pt("bot", w, z0), S.pt("mid", w, 0.5 * (z0 + z1), crease=crease), S.pt("top", w, z1),
            S.pt("tc", 0.0, zt)]


def pod(spacing=0.2, **kw):
    """A closed lozenge-ish body from five stations."""
    st = [S.station(-2.0, box_section(0.4, 0.2, 0.6)), S.station(-1.0, box_section(0.8, 0.2, 0.8)),
          S.station(0.0, box_section(0.9, 0.2, 0.9)), S.station(1.0, box_section(0.8, 0.2, 0.8)),
          S.station(2.0, box_section(0.4, 0.2, 0.6))]
    return S.loft(st, spacing=spacing, **kw)


def cage_faces(m):
    return [list(q) for q in m.Q] + [list(t) for t in m.T]


def evaluate(m, levels=2):
    sub = pytest.importorskip("mkmmd.model.subdiv")
    return sub.subdivide(m.V, cage_faces(m), levels=levels, creases={(int(a), int(b)): float(w) for (a, b), w in zip(m.Ce, m.Cw)})


# ======================================================================================================== interpolation
def test_pchip_hits_the_stations_and_does_not_overshoot():
    x = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    y = np.array([[0.0], [1.0], [1.0], [1.0], [3.0]])                  # a plateau: a cubic spline would ring around it
    q = np.linspace(0.0, 4.0, 81)
    v = S.pchip(x, y, q)[:, 0]
    assert np.allclose(S.pchip(x, y, x)[:, 0], y[:, 0])
    assert v.min() >= -1e-12 and v.max() <= 3.0 + 1e-12
    assert np.all(np.diff(v) >= -1e-12)                                # monotone data stays monotone
    assert np.allclose(v[(q >= 1.0) & (q <= 3.0)], 1.0)                # and the plateau stays flat


def test_sample_stations_never_leaves_a_gap_wider_than_spacing():
    ys = S.sample_stations(np.array([0.0, 0.3, 1.0, 2.5]), 0.2)
    assert np.all(np.diff(ys) <= 0.2 + 1e-9)
    assert {0.0, 0.3, 1.0, 2.5} <= set(np.round(ys, 9))


# ======================================================================================================== loft
def test_loft_is_a_closed_outward_mirrored_cage():
    lf = pod()
    m = lf.mesh
    assert m.is_closed() and m.volume() > 0
    lo, hi = m.bbox()
    assert hi[0] == pytest.approx(0.9) and lo[0] == pytest.approx(-0.9)
    assert lo[1] == pytest.approx(-2.0) and hi[1] == pytest.approx(2.0)
    assert key(m.V) == key(m.V * (-1, 1, 1))                           # the -x half is the exact mirror image
    assert lf.C == 2 * len(lf.names) - 2 and lf.R == len(lf.ys)


def test_loft_rows_include_every_station_and_respect_the_spacing():
    lf = pod(spacing=0.15)
    assert all(any(abs(y - lf.ys) < 1e-9) for y in (-2.0, -1.0, 0.0, 1.0, 2.0))
    assert np.all(np.diff(lf.ys) <= 0.15 + 1e-9)


def test_loft_addressing_reads_the_cage_back():
    lf = pod()
    assert lf.point("top", 0.0) == pytest.approx((0.9, 0.0, 0.9))
    assert lf.point("top", 0.0, side=-1) == pytest.approx((-0.9, 0.0, 0.9))
    x, y = lf.x_at(-1.0, 0.5)                                          # the side crosses z = 0.5 on the vertical wall
    assert x == pytest.approx(0.8) and y == pytest.approx(-1.0)
    assert lf.x_at(-1.0, 5.0) is None
    assert lf.vid(1.0, "bot") == lf.grid[lf.row(1.0), lf.col("bot")]


def test_loft_keeps_a_station_exactly_and_interpolates_between_without_overshoot():
    lf = pod(spacing=0.05)
    xs = [lf.point("top", y)[0] for y in lf.ys]
    assert max(xs) == pytest.approx(0.9)                               # never above the widest station
    assert xs[0] == pytest.approx(0.4) and xs[-1] == pytest.approx(0.4)
    mid = [lf.point("top", y)[2] for y in lf.ys]
    assert max(mid) == pytest.approx(0.9) and min(mid) == pytest.approx(0.6)


def test_loft_point_and_ring_creases_are_tagged():
    st = [S.station(-1.0, box_section(0.5, 0.2, 0.6, crease=1.0), ring=0.7), S.station(0.0, box_section(0.5, 0.2, 0.6, crease=1.0)),
          S.station(1.0, box_section(0.5, 0.2, 0.6, crease=1.0))]
    lf = S.loft(st, spacing=0.5)
    cd = lf.mesh.crease_dict()
    assert set(cd.values()) == {0.7, 1.0}
    # the point crease runs along the loft through 'mid' on both sides: 2 columns x (R - 1) rows
    assert sum(1 for w in cd.values() if w == 1.0) == 2 * (lf.R - 1)
    assert sum(1 for w in cd.values() if w == 0.7) == lf.C                       # the whole first ring


def test_a_hard_station_turns_sharply_instead_of_smoothing():
    soft = [S.station(0.0, box_section(0.5, 0.2, 0.5)), S.station(1.0, box_section(0.5, 0.2, 0.9)),
            S.station(2.0, box_section(0.5, 0.2, 0.5))]
    hard = [soft[0], S.station(1.0, box_section(0.5, 0.2, 0.9), hard=True, ring=1.0), soft[2]]
    a, b = S.loft(soft, spacing=0.25), S.loft(hard, spacing=0.25)
    za = [a.point("top", y)[2] for y in (0.5,)]
    zb = [b.point("top", y)[2] for y in (0.5,)]
    assert zb[0] == pytest.approx(0.7)                                 # linear up to the hard station
    assert za[0] > zb[0] + 0.01                                        # the smooth loft bulges above the straight line
    assert any(w == 1.0 for w in b.mesh.Cw)


def test_a_gap_is_a_creased_v_groove_with_its_own_material():
    plain = pod()
    g = S.gap(0.5, width=0.02, depth=0.01, cols=("mid", "tc"), mat=3)
    lf = pod(gaps=[g])
    m = lf.mesh
    assert m.is_closed() and np.unique(m.Qm).tolist() == [0, 3]
    r = lf.row(0.5)
    assert lf.ys[r] == pytest.approx(0.5) and lf.ys[r + 1] - lf.ys[r] == pytest.approx(0.01)
    p0, p1 = plain.point("top", 0.5), lf.point("top", 0.5)
    assert np.linalg.norm(p1 - p0) == pytest.approx(0.01, abs=0.002)   # pushed in by the depth, along the section normal
    assert np.linalg.norm(lf.point("mid", 0.5, side=-1)[[0, 2]]) < np.linalg.norm(plain.point("mid", 0.5, side=-1)[[0, 2]])
    assert lf.point("keel", 0.5) == pytest.approx(plain.point("keel", 0.5))     # the underside is outside the groove
    for rr in (r - 1, r + 1):                                          # the two rims of the groove are creased edges of one ring each
        ids = set(lf.grid[rr].tolist())
        sel = np.array([a in ids and b in ids for a, b in m.Ce])
        assert sel.sum() > 0 and set(m.Cw[sel]) == {1.0}


def test_an_open_top_region_makes_a_u_section():
    def u(w):                                                            # the top points run down the inner wall to a floor
        return [S.pt("keel", 0.0, 0.2), S.pt("bot", w, 0.2), S.pt("mid", w, 0.5), S.pt("top", w, 0.8), S.pt("in", w - 0.1, 0.8),
                S.pt("floor", w - 0.1, 0.3), S.pt("tc", 0.0, 0.3)]
    def c(w, zt):
        return [S.pt("keel", 0.0, 0.2), S.pt("bot", w, 0.2), S.pt("mid", w, 0.5), S.pt("top", w, zt), S.pt("in", 0.7 * w, zt),
                S.pt("floor", 0.3 * w, zt), S.pt("tc", 0.0, zt)]
    st = [S.station(-2.0, c(0.8, 0.8)), S.station(-1.0, c(0.8, 0.8), hard=True, ring=1.0), S.station(-0.95, u(0.8), hard=True),
          S.station(0.95, u(0.8), hard=True), S.station(1.0, c(0.8, 0.8), hard=True, ring=1.0), S.station(2.0, c(0.8, 0.8))]
    closed = S.loft([S.station(y, c(0.8, 0.8)) for y in (-2.0, 2.0)], spacing=0.5).mesh
    cabin = S.loft(st, spacing=0.3).mesh
    assert cabin.is_closed() and cabin.volume() > 0
    assert cabin.volume() < closed.volume() - 0.5                      # the opening takes volume out of the solid


def test_a_section_that_crosses_itself_is_refused():
    assert not S.section_crosses_itself(S.rrect(1.0, 0.5, 0.1))
    bow = np.array([[0, 0], [1, 1], [1, 0], [0, 1.0]])                       # a bow tie
    assert S.section_crosses_itself(bow)
    # a floor below the underside: the half section dips through itself
    bad = [S.pt("keel", 0.0, 0.3), S.pt("bot", 0.8, 0.3), S.pt("top", 0.8, 0.8), S.pt("floor", 0.6, 0.2), S.pt("tc", 0.0, 0.2)]
    ok = [S.pt("keel", 0.0, 0.1), S.pt("bot", 0.8, 0.1), S.pt("top", 0.8, 0.8), S.pt("floor", 0.6, 0.2), S.pt("tc", 0.0, 0.2)]
    with pytest.raises(ValueError, match="crosses itself"):
        S.loft([S.station(0.0, bad), S.station(1.0, bad)])
    assert S.loft([S.station(0.0, ok), S.station(1.0, ok)]).mesh.is_closed()


def test_loft_rejects_mismatched_stations():
    with pytest.raises(ValueError):
        S.loft([S.station(0.0, box_section(0.5, 0.2, 0.6)), S.station(1.0, box_section(0.5, 0.2, 0.6)[:-1] + [S.pt("oops", 0.0, 0.7)])])
    with pytest.raises(ValueError):
        S.loft([S.station(1.0, box_section(0.5, 0.2, 0.6)), S.station(0.0, box_section(0.5, 0.2, 0.6))])


def test_loft_assign_and_uv():
    lf = pod(uv=True)
    lf.assign(5, y=(-0.5, 0.5), cols=("bot", "mid"), side=+1)
    m = lf.mesh
    assert m.UV.shape == (len(m.V), 2) and 0.0 <= m.UV.min() and m.UV.max() <= 1.0
    cq, _ = m.face_centres()
    sel = m.Qm == 5
    assert sel.any() and np.all(cq[sel][:, 0] > 0) and np.all(np.abs(cq[sel][:, 1]) <= 0.6)


# ======================================================================================================== smoothness (limit surface)
def test_the_subdivided_loft_is_smooth_where_it_should_be_and_sharp_where_creased():
    st = [S.station(y, box_section(0.9 - 0.1 * abs(y), 0.2, 0.8, crease=1.0)) for y in (-1.0, -0.5, 0.0, 0.5, 1.0)]
    m = S.loft(st, spacing=0.25).mesh
    sd = evaluate(m, 2)
    V, F = sd.verts, np.array(sd.faces)
    N = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 3]] - V[F[:, 0]])
    N /= np.linalg.norm(N, axis=1)[:, None]
    # every edge: the angle between its two faces; the creased line (x = side, z = mid) must stay sharp, the rest smooth
    from collections import defaultdict
    ef = defaultdict(list)
    for fi, f in enumerate(F):
        for k in range(4):
            ef[tuple(sorted((f[k], f[(k + 1) % 4])))].append(fi)
    ang = {e: math.degrees(math.acos(np.clip(N[a] @ N[b], -1, 1))) for e, (a, b) in ((e, fs) for e, fs in ef.items() if len(fs) == 2)}
    on_crease = lambda e: all(abs(abs(V[i][0]) - (0.9 - 0.1 * abs(V[i][1]))) < 0.02 and abs(V[i][2] - 0.5) < 0.01 for i in e)
    assert max(a for e, a in ang.items() if not on_crease(e) and abs(V[list(e)][:, 2].mean() - 0.2) > 0.05
               and abs(V[list(e)][:, 1].mean()) < 0.9) < 30.0


def test_subdivision_keeps_the_station_extent_and_pulls_in_the_corners():
    lf = pod(spacing=0.1)
    sd = evaluate(lf.mesh, 2)
    lo, hi = sd.verts.min(0), sd.verts.max(0)
    assert hi[0] <= 0.9 + 1e-9 and hi[2] <= 0.9 + 1e-9 and lo[2] >= 0.2 - 1e-9     # a Catmull-Clark surface stays inside its cage
    assert hi[0] > 0.85                                                             # but the widest station is still close to it


# ======================================================================================================== cages
def test_cage_tube_with_caps_is_closed_and_faces_out():
    th = np.linspace(0, 2 * np.pi, 16, endpoint=False)
    rings = np.stack([np.stack([np.full(16, x), 0.1 * np.cos(th), 0.1 * np.sin(th)], 1) for x in (0.0, 0.2, 0.4)])
    for rr in (rings, rings[:, ::-1]):                                  # either ring direction ends up outward
        m = S.cage(rr, caps=("fan", "fan"), cap_bulge=(0.02, 0.02), crease_rings=[1, 0, 1])
        assert m.is_closed() and m.volume() > 0
        assert m.bbox()[1][0] == pytest.approx(0.42) and m.bbox()[0][0] == pytest.approx(-0.02)    # the caps dome out
        assert len(m.Ce) == 2 * 16 and set(m.Cw) == {1.0}


def test_cage_torus_and_open_strip():
    n, m_ = 24, 12
    a, b = np.linspace(0, 2 * np.pi, n, endpoint=False), np.linspace(0, 2 * np.pi, m_, endpoint=False)
    R, r = 0.3, 0.06
    rings = np.stack([np.stack([(R + r * np.cos(bb)) * np.cos(aa), (R + r * np.cos(bb)) * np.sin(aa), np.full(m_, 0.0) + r * np.sin(bb)], 1)
                      for aa in a for bb in [b]]).reshape(n, m_, 3)
    tor = S.cage(rings, closed=True, loop=True)
    assert tor.is_closed() and tor.volume() > 0
    assert tor.volume() == pytest.approx(2 * math.pi ** 2 * R * r * r, rel=0.06)
    strip = S.cage(rings[:6], closed=False)
    assert not strip.is_closed() and len(strip.Q) == 5 * (m_ - 1)


def test_cage_crease_weights_per_column_and_ring():
    th = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    rings = np.stack([np.stack([np.full(8, x), np.cos(th), np.sin(th)], 1) for x in (0.0, 1.0, 2.0)])
    cc = np.zeros(8)
    cc[3] = 0.6
    m = S.cage(rings, crease_cols=cc, crease_rings=[0, 0.4, 0])
    assert sorted(set(m.Cw)) == [0.4, 0.6] and (m.Cw == 0.6).sum() == 2 and (m.Cw == 0.4).sum() == 8


# ======================================================================================================== panels, frames, outlines
def test_superellipse_and_stadium_areas():
    e = S.superellipse(0.2, 0.1, 2.0, 200)
    assert abs(S.polygon_area(e)) == pytest.approx(math.pi * 0.1 * 0.05, rel=0.01)
    r = S.superellipse(0.2, 0.1, 40.0, 400)
    assert abs(S.polygon_area(r)) == pytest.approx(0.2 * 0.1, rel=0.03)
    st = S.stadium(0.3, 0.1, 12)
    assert abs(S.polygon_area(st)) == pytest.approx(0.2 * 0.1 + math.pi * 0.05 ** 2, rel=0.01)


def test_rounded_panel_is_a_closed_plate_with_a_rolled_rim():
    outline = S.rrect(0.10, 0.06, 0.012, 4)
    m = S.rounded_panel(outline, 0.012, r_edge=0.004, n=4, dome=0.003, uv=True)
    assert m.is_closed() and m.volume() > 0
    lo, hi = m.bbox()
    assert lo[2] == pytest.approx(0.0) and hi[2] == pytest.approx(0.015)          # thickness + dome
    assert hi[0] == pytest.approx(0.05) and hi[1] == pytest.approx(0.03)
    flat = abs(S.polygon_area(outline)) * 0.012
    assert flat * 0.93 < m.volume() - 0.003 * abs(S.polygon_area(outline)) * 0.5 < flat                    # a rolled rim loses a little
    assert m.UV.min() >= -1e-9 and m.UV.max() <= 1.0 + 1e-9
    # no sharp rim: the top edge is rolled, so no face of the panel's side is within 80 degrees of the face above it
    nq, _ = m.face_vectors()
    assert (np.abs(nq[:, 2]) / np.linalg.norm(nq, axis=1)).max() > 0.99          # flat top and bottom faces exist
    assert np.linalg.norm(nq, axis=1).min() > 0


def test_rounded_frame_through_and_with_a_pocket():
    outer, inner = S.rrect(0.12, 0.08, 0.014, 4), S.rrect(0.09, 0.05, 0.008, 4)
    thru = S.rounded_frame(outer, inner, 0.01)
    pocket = S.rounded_frame(outer, inner, 0.01, floor=0.006)
    assert thru.is_closed() and pocket.is_closed() and thru.volume() > 0 and pocket.volume() > thru.volume()
    ring_area = abs(S.polygon_area(outer)) - abs(S.polygon_area(inner))
    assert thru.volume() == pytest.approx(ring_area * 0.01, rel=0.1)
    floor_area = abs(S.polygon_area(inner))
    assert pocket.volume() - thru.volume() == pytest.approx(floor_area * 0.004, rel=0.15)   # 10 - 6 mm of pocket is filled
    assert pocket.bbox()[1][2] == pytest.approx(0.01)


def test_rounded_frame_resamples_outlines_of_different_length():
    f = S.rounded_frame(S.rrect(0.12, 0.08, 0.014, 6), S.rrect(0.09, 0.05, 0.008, 3), 0.01)
    assert f.is_closed()


def test_resample_and_offset_rings():
    sq = np.array([[0, 0], [1, 0], [1, 1], [0, 1.0]])
    r = S.resample_loop(sq, 8)
    assert len(r) == 8 and np.allclose(r[0], [0, 0]) and np.allclose(r[2], [1, 0])
    rings = S.offset_rings(sq, [(0.0, 0.0), (0.1, 0.5)])
    assert rings.shape == (2, 4, 3)
    assert rings[1][:, :2].min() == pytest.approx(0.1) and rings[1][:, :2].max() == pytest.approx(0.9)
    assert np.allclose(rings[1][:, 2], 0.5)
    flipped = S.offset_rings(sq[::-1], [(0.1, 0.0)])                    # a clockwise outline is turned around first
    assert flipped[0][:, :2].min() == pytest.approx(0.1)


# ======================================================================================================== sweeps, cutters, kit
def test_sweep_with_a_taper_rounds_the_ends_off():
    path = np.stack([np.linspace(-1, 1, 30), np.zeros(30), np.zeros(30)], 1)
    taper = np.minimum(1.0, np.sin(np.linspace(0, np.pi, 30)) * 6.0)
    m = S.sweep(path, S.rrect(0.1, 0.06, 0.02, 3), scale=np.stack([taper, taper], 1), up=(0, 0, 1))
    assert m.is_closed()
    assert m.volume() > 0 or m.flip().volume() > 0
    ends = m.V[np.abs(m.V[:, 0]) > 0.99]
    assert np.ptp(ends[:, 1]) < 0.06 and np.ptp(ends[:, 2]) < 0.04       # the end is smaller than the full section


def test_arch_cutter_is_a_closed_flared_cylinder():
    c = S.arch_cutter((1.3, 0.31), 0.36, 0.46, 0.9, flare=0.03, lip=0.055)
    assert c.is_closed() and c.volume() > 0
    lo, hi = c.bbox()
    assert lo[0] == pytest.approx(0.46) and hi[0] == pytest.approx(0.9)
    assert (hi[1] - lo[1]) / 2 == pytest.approx(0.39, abs=0.002)            # the lip flares to R + flare at the outer end
    r_in = np.linalg.norm(c.V[np.abs(c.V[:, 0] - 0.46) < 1e-6][:, 1:] - (1.3, 0.31), axis=1)       # the inner end ring
    assert r_in.max() == pytest.approx(0.36, abs=0.002)
    left = S.arch_cutter((1.3, 0.31), 0.36, -0.46, -0.9)
    assert left.is_closed() and left.volume() > 0 and left.bbox()[1][0] == pytest.approx(-0.46)


def test_probe_casts_rays_against_a_surface():
    m = S.rounded_box((1.0, 0.6, 0.4), (0.0, 0.0, 0.2), r=0.02)
    p = S.Probe.from_mesh(m)
    t, pt_, n = p.ray((3.0, 0.0, 0.2), (-1.0, 0.0, 0.0))
    assert t == pytest.approx(2.5, abs=1e-6) and pt_[0] == pytest.approx(0.5) and n == pytest.approx((1, 0, 0), abs=1e-6)
    assert p.height(0.0, 0.0) == pytest.approx(0.4)
    assert p.ray((3.0, 5.0, 0.2), (-1.0, 0.0, 0.0)) is None
    t2, _, n2 = p.ray((0.0, 0.0, 0.2), (0.0, 1.0, 0.0))                     # from inside: the wall it leaves, normal toward the origin
    assert t2 == pytest.approx(0.3) and n2 == pytest.approx((0, -1, 0), abs=1e-6)


def test_place_puts_a_part_on_a_surface():
    m = S.place(S.rounded_panel(S.rrect(0.1, 0.05, 0.01), 0.01), (1.0, 2.0, 3.0), (0.0, -1.0, 0.0), (0.0, 0.0, 1.0))
    lo, hi = m.bbox()
    assert lo == pytest.approx((0.95, 1.99, 2.975), abs=1e-9) and hi == pytest.approx((1.05, 2.0, 3.025), abs=1e-9)
    # its face looks along the normal, its y along up
    assert m.V[np.argmin(m.V[:, 1])][1] == pytest.approx(1.99)


def test_merge_and_weld_carry_creases():
    a = S.rounded_box((0.2, 0.2, 0.2), r=0.02)
    a.crease([(0, 1)], 0.5)
    b = a.moved((1.0, 0.0, 0.0))
    m = S.merge([a, b])
    assert len(m.Ce) == 2 and m.Ce[1][0] == len(a.V) + a.Ce[0][0]
    w = m.weld()
    assert len(w.Ce) == 2 and set(w.Cw) == {0.5}


def test_the_kit_primitives_stay_closed_and_outward():
    for m in (S.rounded_box((0.4, 0.3, 0.2), r=0.03, k=2), S.lathe([(0.0, 0.0), (0.1, 0.0), (0.1, 0.1), (0.0, 0.1)], 16, closed_ends=False),
              S.tube(np.array([[0, 0, 0], [0, 0, 0.5], [0.2, 0, 1.0]]), 0.03, sides=10), S.torus_path(0.2) is None or S.rounded_box((0.1,) * 3)):
        assert m.is_closed() and m.volume() > 0


def test_shell_contract_and_part_tuple():
    cage = pod().mesh
    sh = S.smooth(cage, levels=3, bevel=(0.003, 2), cutters=[S.arch_cutter((0, 0.3), 0.3, 0.4, 0.8)], cutter_role=4)
    assert sh.mesh is cage and sh.levels == 3 and sh.bevel == (0.003, 2) and len(sh.cutters) == 1 and sh.cutter_role == 4
    assert sh.solver == "EXACT" and S.smooth(cage, solver="FAST").solver == "FAST"
    assert S.Part(cage, (0, 0, 0), (0, 0, 0)).origin == (0, 0, 0)


# ======================================================================================================== the primitives
def test_rounded_box_volume_against_the_steiner_formula():
    a, b, c, r = 0.4, 0.6, 0.2, 0.03
    m = S.rounded_box((a, b, c), r=r, k=4)
    p, q, s = a - 2 * r, b - 2 * r, c - 2 * r                       # the rounded box is the inner box grown by r
    exact = p * q * s + 2 * (p * q + q * s + s * p) * r + math.pi * r ** 2 * (p + q + s) + 4 / 3 * math.pi * r ** 3
    assert m.is_closed()
    assert exact * 0.995 < m.volume() <= exact                       # inscribed facets: a hair under the exact value


def test_rounded_box_keeps_its_extent_and_is_symmetric():
    m = S.rounded_box((0.5, 0.3, 0.2), center=(1.0, 2.0, 3.0), r=0.02, k=2, div=0.05)
    lo, hi = m.bbox()
    assert lo == pytest.approx((0.75, 1.85, 2.9)) and hi == pytest.approx((1.25, 2.15, 3.1))
    assert m.is_closed()
    c = m.V - (1.0, 2.0, 3.0)
    assert key(c) == key(c * (-1, 1, 1)) == key(c * (1, -1, 1))


def test_sharp_extrusion_volume_is_area_times_width():
    bottom = np.array([[-1.0, 0.2], [0.0, 0.2], [1.0, 0.2]])
    top = np.array([[-1.0, 0.8], [0.5, 0.9], [1.0, 0.8]])
    m = S.extrude_profile(bottom, top, -0.5, 0.5, 0.0, 0.0)
    area = 0.5 * (0.6 + 0.7) * 1.5 + 0.5 * (0.7 + 0.6) * 0.5
    assert m.is_closed() and m.volume() == pytest.approx(area * 1.0)


def test_rounded_extrusion_loses_only_the_rim():
    bottom = np.array([[-1.0, 0.2], [1.0, 0.2]])
    top = np.array([[-1.0, 0.8], [1.0, 0.8]])
    r = 0.05
    m = S.extrude_profile(bottom, top, -0.5, 0.5, r, r, 4, 4, flush=(False, False))
    assert m.is_closed()
    sharp = 2.0 * 0.6 * 1.0
    perimeter = 2 * (2.0 + 0.6)
    rim_total = 2 * (1 - math.pi / 4) * r ** 2 * perimeter           # both sides lose a rounded rim all round
    assert m.volume() == pytest.approx(sharp - rim_total, abs=0.15 * rim_total)


def test_lathe_cylinder_volume():
    m = S.lathe([(0.3, 0.0), (0.3, 0.5)], seg=96, closed_ends=True)
    assert m.is_closed() and m.volume() == pytest.approx(math.pi * 0.09 * 0.5, rel=0.002)


def test_tube_volume_and_ring_closure():
    path = np.stack([np.zeros(40), np.zeros(40), np.linspace(0, 1.0, 40)], 1)
    t = S.tube(path, 0.05, sides=48)
    assert t.is_closed() and t.volume() == pytest.approx(math.pi * 0.0025 * 1.0, rel=0.01)
    ring = S.tube(S.torus_path(0.2, 72), 0.02, sides=24, closed=True).weld()
    # a 24-gon section holds 98.9 % of the circle's area
    assert ring.is_closed() and ring.volume() == pytest.approx(2 * math.pi ** 2 * 0.2 * 0.02 ** 2 * 0.9886, rel=0.005)


def test_fillet_arcs_are_tangent_with_the_requested_radius():
    pts = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]])
    f = S.fillet(pts, [0, 0.2, 0], n=16)
    centre = np.array([0.8, 0.2])
    on_arc = np.abs(np.hypot(f[:, 0] - centre[0], f[:, 1] - centre[1]) - 0.2) < 1e-9
    assert on_arc.sum() == 17                                        # n + 1 points of the arc
    assert f[0] == pytest.approx((0, 0)) and f[-1] == pytest.approx((1, 1))
    assert f[on_arc][0] == pytest.approx((0.8, 0.0)) and f[on_arc][-1] == pytest.approx((1.0, 0.2))   # tangent points
    assert not np.any(np.all(np.isclose(f, (1.0, 0.0)), axis=1))      # the sharp corner is gone


def test_weld_turns_collapsed_quads_into_triangles():
    V = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [1, 1, 0.0], [0, 1, 0]], float)      # vertices 2 and 3 coincide
    m = S.Mesh(V, [[0, 1, 2, 4], [1, 3, 4, 4]]).weld()
    assert len(m.V) == 4
    assert len(m.Q) == 1 and len(m.T) == 1                           # one honest quad, one collapsed to a triangle
    assert S.Mesh(V, [[2, 3, 3, 2]]).weld().nfaces == 0              # a fully collapsed face is dropped


def test_vertex_normals_are_area_weighted():
    # a big flat floor next to a tiny steep strip: the shared vertices keep the floor's normal
    V = np.array([[0, 0, 0], [10, 0, 0], [10, 10, 0], [0, 10, 0], [10, 10.001, 0.001], [0, 10.001, 0.001]], float)
    m = S.Mesh(V, [[0, 1, 2, 3], [3, 2, 4, 5]])
    n = m.vertex_normals()
    assert n[0] == pytest.approx((0, 0, 1))
    assert n[2][2] > 0.999


# ======================================================================================================== docs/modelling.md
def test_the_code_blocks_of_the_modelling_guide_run():
    text = (Path(__file__).resolve().parents[1] / "docs" / "modelling.md").read_text()
    blocks = re.findall(r"```python  # doc-test\n(.*?)```", text, re.S)
    assert len(blocks) >= 2
    for b in blocks:
        exec(compile(b, "docs/modelling.md", "exec"), {})
