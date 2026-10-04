"""Pure maths of the bedroom furniture: the chair's backrest frame, rounded-rectangle outlines, bent-tube centre lines
and stitch marks. (The Blender-side builders are checked by building and looking.)"""
import math

import pytest

from mkmmd.blender.library.props import bedroom_furniture_geo as G


def area(pts):
    return 0.5 * sum(a[0] * b[1] - b[0] * a[1] for a, b in zip(pts, pts[1:] + pts[:1]))


def seg_dist(p, a, b):
    ab = [y - x for x, y in zip(a, b)]
    ap = [y - x for x, y in zip(a, p)]
    t = max(0.0, min(1.0, sum(u * v for u, v in zip(ap, ab)) / sum(u * u for u in ab)))
    return math.dist(p, [x + t * u for x, u in zip(a, ab)])


# ----------------------------------------------------------------- the chair's backrest
def test_backrest_plane_is_the_cafe_chair_plane():
    assert G.back_y(0.45) == pytest.approx(0.170)
    assert G.back_y(0.853) == pytest.approx(0.244)
    assert math.degrees(G.BACK_TILT) == pytest.approx(10.4, abs=0.05)
    assert G.BACK_C == pytest.approx((0.0, G.back_y(0.66), 0.66))


def test_pad_frame_leans_back_and_keeps_its_axes_orthonormal():
    ex, ey, ez = G.pad_dir(1, 0, 0), G.pad_dir(0, 1, 0), G.pad_dir(0, 0, 1)
    assert ex == pytest.approx((1, 0, 0))
    assert math.dist(ey, (0, 0, 0)) == pytest.approx(1) and math.dist(ez, (0, 0, 0)) == pytest.approx(1)
    assert sum(a * b for a, b in zip(ey, ez)) == pytest.approx(0, abs=1e-12)
    assert ez[1] > 0 and ez[2] > 0                       # up the cushion = up and back
    assert ey[1] > 0 and ey[2] < 0                       # through the cushion = back and down
    assert G.pad_point(0, 0, 0) == pytest.approx(G.BACK_C)
    p = G.pad_point(0.1, 0.02, -0.05)
    o = G.pad_point(0, 0, 0)
    assert tuple(a - b for a, b in zip(p, o)) == pytest.approx(tuple(0.1 * a + 0.02 * b - 0.05 * c
                                                                      for a, b, c in zip(ex, ey, ez)))


def test_front_face_of_the_pad_lies_on_the_collider_front_plane():
    t = math.tan(G.BACK_TILT)
    f = G.pad_point(0, -0.015, 0)                        # front face of the 0.030 thick collider box
    for x, zp in ((0, 0.2), (0.19, -0.17), (-0.19, 0.215)):
        p = G.pad_point(x, -0.015, zp)
        assert p[1] == pytest.approx(f[1] + t * (p[2] - f[2]))
    n = tuple(-a for a in G.pad_dir(0, 1, 0))            # normal of the front face points at the sitter, up a little
    assert n[1] < -0.9 and n[2] > 0.1


# ----------------------------------------------------------------- rounded rectangles
def test_rrect_counts_bounds_and_orientation():
    ring = G.rrect(0.42, 0.36, 0.07, n=6)
    assert len(ring) == 28 and area(ring) > 0           # counter-clockwise
    xs, ys = [p[0] for p in ring], [p[1] for p in ring]
    assert (max(xs), min(xs)) == pytest.approx((0.21, -0.21)) and (max(ys), min(ys)) == pytest.approx((0.18, -0.18))
    exact = 0.42 * 0.36 - (4 - math.pi) * 0.07 ** 2          # the chords of the arcs cut slightly inside
    assert area(ring) < exact and area(ring) == pytest.approx(exact, rel=3e-3)


def test_rrect_insets_keep_the_point_count_and_stay_nested():
    base = G.rrect(0.42, 0.42, 0.075, n=8)
    for inset in (0.004, 0.02, 0.07, 0.1, 0.2):
        ring = G.rrect(0.42, 0.42, 0.075, n=8, inset=inset)
        assert len(ring) == len(base)
        assert max(abs(p[0]) for p in ring) == pytest.approx(0.21 - inset)
        assert max(max(abs(p[0]), abs(p[1])) for p in ring) <= 0.21 - inset + 1e-12
    with pytest.raises(ValueError):
        G.rrect(0.42, 0.42, 0.075, inset=0.21)


def test_rrect_corners_are_circular_arcs():
    r = 0.05
    ring = G.rrect(0.4, 0.3, r, n=9)
    centres = [(0.2 - r, 0.15 - r), (r - 0.2, 0.15 - r), (r - 0.2, r - 0.15), (0.2 - r, r - 0.15)]
    for k, c in enumerate(centres):
        for p in ring[k * 10:(k + 1) * 10]:
            assert math.dist(p, c) == pytest.approx(r)


# ----------------------------------------------------------------- bent tubes
def test_round_path_corner_has_the_fillet_length_and_radius():
    r = 0.05
    pts = [(0, 0, 0), (1.0, 0, 0), (1.0, 1.0, 0)]
    path = G.round_path(pts, r, arc_deg=3.0, step=0.05)
    assert path[0] == pytest.approx((0, 0, 0)) and path[-1] == pytest.approx((1.0, 1.0, 0))
    length = sum(math.dist(a, b) for a, b in zip(path, path[1:]))
    exact = 2.0 - 2 * r + r * math.pi / 2
    assert length == pytest.approx(exact, rel=2e-3) and length < exact + 1e-9           # chords are slightly short
    arc = [p for p in path if p[0] > 1.0 - r - 1e-9 and p[1] < r + 1e-9 and (p[0] < 1.0 - 1e-9 or p[1] > 1e-9)]
    assert len(arc) >= 20
    for p in arc:
        assert math.dist(p, (1.0 - r, r, 0.0)) == pytest.approx(r)


def test_round_path_points_are_distinct_and_evenly_dense():
    R, zr = 0.0115, 0.3865
    Rb = (zr - R) / 2
    pts = [(0.2175, 0.235, R), (0.2175, -0.225, R), (0.2175, -0.225, zr), (0.2175, 0.166, zr), (0.2175, 0.255, 0.87)]
    path = G.round_path(pts, [0, Rb, Rb, 0.055, 0], arc_deg=7.5, step=0.05)
    gaps = [math.dist(a, b) for a, b in zip(path, path[1:])]
    assert min(gaps) > 1e-4 and max(gaps) <= 0.05 + 1e-9        # the two half-turns of the front curve meet without a repeat
    assert {round(p[0], 9) for p in path} == {0.2175}            # stays in its plane
    assert min(p[2] for p in path) == pytest.approx(R) and min(p[1] for p in path) == pytest.approx(-0.225)
    # tangent turns by at most the arc step between neighbouring points
    dirs = [tuple((b[i] - a[i]) / math.dist(a, b) for i in range(3)) for a, b in zip(path, path[1:])]
    for u, v in zip(dirs, dirs[1:]):
        assert math.degrees(math.acos(max(-1, min(1, sum(a * b for a, b in zip(u, v)))))) <= 7.5 + 1e-6


def test_round_path_closed_loop():
    loop = G.round_path([(0, 0, 0), (0.5, 0, 0), (0.5, 0.5, 0), (0, 0.5, 0)], 0.25, arc_deg=5.0, closed=True)
    assert math.dist(loop[0], loop[-1]) > 1e-6                                  # the first point is not repeated
    for p in loop:                                                              # radius = half the side: a circle
        assert math.dist(p, (0.25, 0.25, 0.0)) == pytest.approx(0.25)
    loop2 = G.round_path([(0, 0, 0), (0.5, 0, 0), (0.5, 0.3, 0), (0, 0.3, 0)], 0.05, closed=True)
    assert min(math.dist(a, b) for a, b in zip(loop2, loop2[1:] + loop2[:1])) > 1e-4


def test_round_path_shortens_a_fillet_that_does_not_fit_and_skips_straight_corners():
    path = G.round_path([(0, 0, 0), (0.1, 0, 0), (0.1, 1.0, 0)], 0.5)            # asks for more than the run allows
    assert max(p[0] for p in path) == pytest.approx(0.1) and min(p[0] for p in path) == pytest.approx(0)
    straight = G.round_path([(0, 0, 0), (0.5, 0, 0), (1.0, 0, 0)], 0.1, step=0.25)
    assert straight == [pytest.approx(p) for p in [(0, 0, 0), (0.25, 0, 0), (0.5, 0, 0), (0.75, 0, 0), (1.0, 0, 0)]]
    with pytest.raises(ValueError):
        G.round_path([(0, 0, 0), (1, 0, 0), (0, 0, 0)], 0.1)                      # doubles back
    with pytest.raises(ValueError):
        G.round_path([(0, 0, 0)], 0.1)


# ----------------------------------------------------------------- stitch marks
def test_dash_marks_are_evenly_spaced_on_the_outline():
    ring = G.rrect(0.42, 0.42, 0.075, n=8, inset=0.01)
    L = G.perimeter(ring)
    marks = G.dash_marks(ring, 0.008)
    assert len(marks) == round(L / 0.008)
    for p, t in marks:
        assert math.hypot(*t) == pytest.approx(1.0)
        assert min(seg_dist(p, a, b) for a, b in zip(ring, ring[1:] + ring[:1])) < 1e-9
    # arc-length spacing is even: measure along the outline
    cum = [0.0]
    for a, b in zip(ring, ring[1:] + ring[:1]):
        cum.append(cum[-1] + math.dist(a, b))
    s = []
    for p, _ in marks:
        k = min(range(len(ring)), key=lambda i: seg_dist(p, ring[i], ring[(i + 1) % len(ring)]))
        s.append(cum[k] + math.dist(ring[k], p))
    steps = [b - a for a, b in zip(s, s[1:])]
    assert max(steps) - min(steps) < 1e-9 and steps[0] == pytest.approx(L / len(marks))


def test_dash_marks_work_in_3d_and_on_open_polylines():
    marks = G.dash_marks([(0, 0, 0), (0, 0, 1.0)], 0.25, closed=False)
    assert [round(p[2], 6) for p, _ in marks] == [0.125, 0.375, 0.625, 0.875]
    assert all(t == pytest.approx((0, 0, 1)) for _, t in marks)
