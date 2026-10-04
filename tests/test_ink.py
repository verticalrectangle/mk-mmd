"""Handwriting ink, bpy-free: strokes files, the ribbon (width, lift, caps, bends, dots, loops, write times), the frame of
a page, windows and the coarse track fallback (mkmmd.core.ink). The Blender half (mesh, shader, clock key) is checked by
building and looking (docs/design.md: Text, Ink)."""
import json

import numpy as np
import pytest

from mkmmd.core import ink as I

PAGE = (0.16, 0.22)                      # a letter page, metres
W, LIFT = 0.00042, 0.00018


def stroke(p, t0=0.0, dt=0.01):
    p = np.asarray(p, float)
    return I.Stroke(t0 + dt * np.arange(len(p)), p)


def line(x0, x1, y, n, t0=0.0, dur=1.0):
    return I.Stroke(t0 + dur * np.linspace(0, 1, n), np.stack([np.linspace(x0, x1, n), np.full(n, y)], 1))


def rows(rb):
    """Left and right vertex of every cross-section, (R, 2, 3), and its time (R,)."""
    return rb.verts.reshape(-1, 2, 3), rb.tw.reshape(-1, 2)[:, 0]


def areas(rb):
    """Signed area of every quad (positive: counter-clockwise seen from +z)."""
    q = rb.verts[:, :2][rb.faces]
    x, y = q[..., 0], q[..., 1]
    return 0.5 * (x * np.roll(y, -1, 1) - np.roll(x, -1, 1) * y).sum(1)


def refused(fn, *args, **kw):
    with pytest.raises(I.InkError) as e:
        fn(*args, **kw)
    return str(e.value)


# ---------------------------------------------------------------- frames
def test_page_millimetres_map_to_the_surface_frame_from_the_top_left_corner():
    p = I.to_surface([[0, 0], [160, 220], [80, 110], [20, 25]], PAGE)
    assert np.allclose(p[0], [-0.08, 0.11]) and np.allclose(p[1], [0.08, -0.11])        # top-left, bottom-right
    assert np.allclose(p[2], [0, 0])
    assert np.allclose(p[3], [(20 - 80) * 1e-3, (110 - 25) * 1e-3])                     # cafe_page.page_to_local(20, 25)
    assert np.allclose(I.to_page(p, PAGE), [[0, 0], [160, 220], [80, 110], [20, 25]])


def test_frame_follows_the_panel_size_not_the_page():
    assert np.allclose(I.to_surface([[0, 0], [50, 100]], (0.05, 0.1)), [[-0.025, 0.05], [0.025, -0.05]])


# ---------------------------------------------------------------- the ribbon
def test_straight_stroke_is_a_strip_of_the_width_lifted_with_half_width_caps():
    rb = I.ribbon([line(20, 30, 40, 11)], PAGE, W, LIFT)
    v, _ = rows(rb)
    assert rb.verts.shape == (2 * (11 + 2), 3) and rb.faces.shape == (11 + 1, 4)
    assert np.allclose(rb.verts[:, 2], LIFT)
    x0, y0 = (20 - 80) * 1e-3, (110 - 40) * 1e-3
    assert np.allclose(v[:, 0, 1] - v[:, 1, 1], W) and np.allclose(v[:, :, 1].mean(1), y0)      # W wide, centred
    assert v[0, 0, 0] == pytest.approx(x0 - W / 2) and v[-1, 0, 0] == pytest.approx(x0 + 10e-3 + W / 2)   # caps
    assert np.allclose(np.diff(v[:, 0, 0])[1:-1], 1e-3)                                          # points 1 mm apart
    assert (areas(rb) > 0).all() and areas(rb).sum() == pytest.approx((10e-3 + W) * W)


def test_left_is_the_left_of_the_direction_of_writing_whatever_the_direction():
    for ang in np.linspace(0, 2 * np.pi, 9)[:-1]:
        d = np.array([np.cos(ang), np.sin(ang)])
        rb = I.ribbon([stroke(np.array([50.0, 50.0]) + np.outer(np.arange(6), d))], PAGE, W, LIFT)
        v, _ = rows(rb)
        ds = np.array([d[0], -d[1]])                                       # page y is down, the surface's is up
        left = np.array([-ds[1], ds[0]])
        assert np.allclose((v[:, 0, :2] - v[:, 1, :2]) @ left, W)
        assert (areas(rb) > 0).all()                                       # counter-clockwise seen from +z


def test_write_time_is_monotone_along_every_stroke_and_shared_by_both_sides():
    s1 = stroke([[10, 10], [12, 11], [14, 10], [16, 12]], t0=1.0, dt=0.2)
    s2 = stroke([[10, 30], [11, 30], [12, 31]], t0=3.0, dt=0.05)
    rb = I.ribbon([s1, s2], PAGE, W, LIFT)
    assert np.array_equal(rb.tw[0::2], rb.tw[1::2])
    _, t = rows(rb)
    assert np.allclose(t[:6], [1.0, 1.0, 1.2, 1.4, 1.6, 1.6])               # the caps share the time of their end point
    assert np.allclose(t[6:], [3.0, 3.0, 3.05, 3.1, 3.1])
    assert (np.diff(t[:6]) >= 0).all() and (np.diff(t[6:]) >= 0).all()


def test_no_face_joins_two_strokes():
    rb = I.ribbon([line(10, 20, 10, 6), line(10, 20, 30, 4, t0=5.0), line(30, 40, 50, 8, t0=9.0)], PAGE, W, LIFT)
    assert len(rb.faces) == 6 + 4 + 8 + 3                                   # points + one for the caps of each stroke
    row = rb.faces // 2
    assert (row.max(1) - row.min(1) == 1).all()                             # a face spans two neighbouring cross-sections
    t = rb.tw[rb.faces]
    assert (t.max(1) - t.min(1)).max() <= 1 / 3 + 1e-9                      # never the pause between strokes (4 s and more)


def test_width_is_kept_around_a_smooth_curve():
    a = np.linspace(0, 1.5 * np.pi, 400)
    rb = I.ribbon([stroke(np.stack([60 + 5 * np.cos(a), 60 + 5 * np.sin(a)], 1))], PAGE, W, LIFT)   # 5 mm arc, 0.06 mm apart
    v, _ = rows(rb)
    inner = slice(2, -2)
    assert np.allclose(np.hypot(*(v[inner, 0, :2] - v[inner, 1, :2]).T), W, rtol=1e-3)
    c = I.to_surface([[60, 60]], PAGE)[0]
    r = np.hypot(*(v[inner, :, :2] - c).transpose(2, 0, 1))
    assert np.allclose(np.abs(r - 5e-3), W / 2, rtol=2e-3)                  # the edges lie half a width off the circle


def test_a_right_angle_corner_keeps_its_width_and_a_hairpin_never_spikes():
    rb = I.ribbon([stroke([[40, 40], [42, 40], [44, 40], [44, 42], [44, 44]])], PAGE, W, LIFT)
    v, _ = rows(rb)
    assert np.isclose(np.hypot(*(v[3, 0, :2] - v[3, 1, :2])), W * np.sqrt(2))     # miter: the edges meet square
    pts = [[40, 40], [42, 40], [44, 40], [42, 40.1], [40, 40.2]]
    rb = I.ribbon([stroke(pts)], PAGE, W, LIFT)
    v, _ = rows(rb)
    assert np.isfinite(rb.verts).all() and np.isfinite(rb.tw).all()
    centre = I.to_surface(pts, PAGE)
    assert (np.hypot(*(v[1:-1, :, :2] - centre[:, None]).transpose(2, 0, 1)) <= W / 2 * I.MITER + 1e-12).all()


def test_a_nib_that_doubles_back_exactly_stays_square_to_the_way_in():
    rb = I.ribbon([stroke([[40, 40], [42, 40], [44, 40], [42, 40], [40, 40]])], PAGE, W, LIFT)
    v, _ = rows(rb)
    assert np.isfinite(rb.verts).all()
    assert np.isclose(np.hypot(*(v[3, 0, :2] - v[3, 1, :2])), W)            # the turning point: one width, no spike


def test_a_one_point_stroke_is_a_dot_of_about_a_width():
    rb = I.ribbon([I.Stroke(np.array([2.0]), np.array([[50.0, 60.0]]))], PAGE, W, LIFT)
    assert rb.verts.shape == (8, 3) and rb.faces.shape == (3, 4) and rb.points == 2
    assert np.isfinite(rb.verts).all() and (areas(rb) > 0).all()
    c = I.to_surface([[50, 60]], PAGE)[0]
    assert np.abs(rb.verts[:, 0] - c[0]).max() <= W * (0.5 + I.DOT) + 1e-12
    assert np.ptp(rb.verts[:, 1]) == pytest.approx(W)
    assert rb.tw.min() == 2.0 and rb.tw.max() == pytest.approx(2.0 + I.DOT_S)


def test_repeated_points_merge_so_a_resting_pen_leaves_no_zero_length_quads():
    base = stroke([[10, 10], [11, 10], [12, 11]])
    rest = I.Stroke(np.array([0.0, 0.01, 0.02, 0.03, 0.04]), np.array([[10, 10], [10, 10], [11, 10], [11, 10], [12, 11.0]]))
    a, b = I.ribbon([base], PAGE, W, LIFT), I.ribbon([rest], PAGE, W, LIFT)
    assert b.points == 3 and np.allclose(b.verts, a.verts) and (areas(b) > 0).all()
    assert np.allclose(b.tw.reshape(-1, 2)[:, 0], [0.0, 0.0, 0.02, 0.04, 0.04])     # the time of the first of a run


def test_a_closed_loop_is_a_ring_of_the_width():
    a = np.linspace(0, 2 * np.pi, 361)
    pts = np.stack([80 + 4 * np.cos(a), 110 + 4 * np.sin(a)], 1)            # a 4 mm circle around the surface origin
    pts[-1] = pts[0]
    rb = I.ribbon([stroke(pts)], PAGE, W, LIFT)
    assert np.isfinite(rb.verts).all() and (areas(rb) > 0).all()
    r = np.hypot(rb.verts[:, 0], rb.verts[:, 1])
    assert r.min() >= 4e-3 - W / 2 - 1e-7 and r.max() <= 4e-3 + W / 2 + 1e-5       # bends add a hair; square caps poke out
    _, t = rows(rb)
    assert t[0] == t[1] and t[-1] == t[-2] and (np.diff(t) >= 0).all()


def test_hundreds_of_strokes_build_in_one_go_and_stay_well_formed():
    rng = np.random.default_rng(1)
    strokes = []
    for k in range(300):
        n = int(rng.integers(1, 60))
        head = np.cumsum(rng.normal(0, 0.2, n))                             # a hand-like path: smooth bends, 0.15 mm steps
        strokes.append(stroke(np.array([80.0, 100.0]) + np.cumsum(0.15 * np.stack([np.cos(head), np.sin(head)], 1), 0),
                              t0=float(k)))
    rb = I.ribbon(strokes, PAGE, W, LIFT)
    assert rb.strokes == 300 and np.isfinite(rb.verts).all() and (areas(rb) > 0).all()
    assert len(rb.faces) == rb.points + 300 and len(rb.verts) == 2 * (rb.points + 2 * 300)
    assert rb.faces.dtype == np.int32 and rb.faces.max() == len(rb.verts) - 1


def test_no_strokes_make_an_empty_ribbon_and_a_bad_width_is_refused():
    rb = I.ribbon([], PAGE)
    assert rb.verts.shape == (0, 3) and rb.faces.shape == (0, 4) and rb.strokes == 0
    assert "width" in refused(I.ribbon, [line(0, 1, 0, 3)], PAGE, width=0.0)


def test_the_shown_part_of_the_ink_ends_at_the_nib():
    s = line(20, 60, 40, 401, t0=1.0, dur=4.0)                              # 0.1 mm apart, written over four seconds
    rb = I.ribbon([s], PAGE, W, LIFT)
    for t in (1.5, 2.5, 4.9):
        x_nib = (I.front([s], t)[0] - 80) * 1e-3
        assert x_nib - 1.01e-4 <= rb.verts[rb.tw <= t][:, 0].max() <= x_nib + 1e-12     # within one point's spacing
    assert (rb.tw <= 0.5).sum() == 0 and (rb.tw <= 5.0).sum() == len(rb.tw)


# ---------------------------------------------------------------- strokes files
def doc(*strokes, **extra):
    return {"unit": "mm", "strokes": [{"t": list(t), "p": [list(q) for q in p]} for t, p in strokes], **extra}


def test_parse_reads_strokes_in_order_and_defaults_the_unit():
    d = doc(([0.0, 0.5], [(1, 2), (3, 4)]), ([2.0], [(5, 6)]))
    s = I.parse(d)
    assert [len(x.t) for x in s] == [2, 1] and s[0].p.tolist() == [[1, 2], [3, 4]] and s[1].t.tolist() == [2.0]
    del d["unit"]
    assert len(I.parse(d)) == 2


@pytest.mark.parametrize("bad, fragment", [
    ([], "`strokes` list"),
    ({"strokes": "x"}, "`strokes` list"),
    ({"strokes": 3}, "`strokes` list"),
    ({"unit": "m", "strokes": [{"t": [0], "p": [[0, 0]]}]}, "millimetres"),
    ({"strokes": []}, "no strokes"),
    ({"strokes": [{"t": [0]}]}, "stroke 0: needs `t`"),
    ({"strokes": [{"t": [0], "p": [[0, 0]]}, {"p": [[0, 0]], "t": []}]}, "stroke 1: needs at least one"),
    ({"strokes": [{"t": [0, 1], "p": [[0, 0]]}]}, "2 times but `p` has shape [1, 2]"),
    ({"strokes": [{"t": [0, 1], "p": [[0, 0, 0], [1, 1, 1]]}]}, "shape [2, 3]"),
    ({"strokes": [{"t": [0, 1], "p": [[0, 0], [1]]}]}, "must be numbers"),
    ({"strokes": [{"t": ["a", 1], "p": [[0, 0], [1, 1]]}]}, "must be numbers"),
    ({"strokes": [{"t": [0, float("nan")], "p": [[0, 0], [1, 1]]}]}, "finite"),
    ({"strokes": [{"t": [0, 1], "p": [[0, 0], [float("inf"), 1]]}]}, "finite"),
    ({"strokes": [{"t": [0, 1, 0.5], "p": [[0, 0], [1, 1], [2, 2]]}]}, "stroke 0: times must not decrease"),
])
def test_parse_refuses_bad_documents_naming_the_stroke(bad, fragment):
    assert fragment in refused(I.parse, bad, "ink.json")
    assert refused(I.parse, bad, "ink.json").startswith("ink.json: ")


def test_equal_times_are_allowed_a_pen_can_rest():
    assert len(I.parse(doc(([1.0, 1.0, 2.0], [(0, 0), (1, 0), (2, 0)])))) == 1


def test_load_reads_a_file_and_its_page_and_names_the_file_in_errors(tmp_path):
    p = tmp_path / "ink.json"
    ok = doc(([0, 1], [(0, 0), (1, 1)]))
    p.write_text(json.dumps({**ok, "page": [160, 220]}), encoding="utf-8")
    strokes, page = I.load(p)
    assert len(strokes) == 1 and page == (160.0, 220.0)
    p.write_text(json.dumps(ok), encoding="utf-8")
    assert I.load(p)[1] is None
    p.write_text("{not json", encoding="utf-8")
    assert "ink.json: not valid JSON" in refused(I.load, p)
    assert "missing.json" in refused(I.load, tmp_path / "missing.json")
    p.write_text(json.dumps({**ok, "page": [160]}), encoding="utf-8")
    assert "`page` must be [width, height]" in refused(I.load, p)
    p.write_text(json.dumps({**ok, "page": [0, 220]}), encoding="utf-8")
    assert "positive" in refused(I.load, p)
    p.write_text(json.dumps(doc(([1, 0], [(0, 0), (1, 1)]))), encoding="utf-8")
    assert "ink.json: stroke 0: times must not decrease" in refused(I.load, p)


# ---------------------------------------------------------------- windows
def two_lines():
    return [line(0, 10, 5, 11, t0=0.0, dur=10.0), line(0, 10, 8, 11, t0=20.0, dur=10.0)]      # x = t, then x = t - 20


def test_window_cuts_strokes_exactly_at_the_bounds():
    w = I.window(two_lines(), 3.5, 25.0)
    assert len(w) == 2
    assert w[0].t[0] == 3.5 and w[0].p[0].tolist() == [3.5, 5.0] and w[0].t[-1] == 10.0     # cut between points 3 and 4
    assert w[1].t[-1] == 25.0 and w[1].p[-1].tolist() == [5.0, 8.0]
    assert np.diff(w[0].t).min() > 0 and np.diff(w[1].t).min() > 0


def test_window_drops_what_lies_outside_and_keeps_all_without_bounds():
    s = two_lines()
    assert len(I.window(s)) == 2 and I.window(s)[0].t.tolist() == s[0].t.tolist()
    assert len(I.window(s, 11.0, 19.0)) == 0
    assert len(I.window(s, 10.0, 20.0)) == 2                                   # touching the bounds still counts
    only = I.window(s, 0.0, 15.0)
    assert len(only) == 1 and only[0].t.tolist() == s[0].t.tolist()
    late = I.window(s, 21.0)
    assert len(late) == 1 and late[0].t[0] == 21.0


def test_window_inside_one_segment_leaves_a_sliver_and_reversed_bounds_are_an_error():
    w = I.window(two_lines()[:1], 2.25, 2.75)
    assert len(w) == 1 and w[0].t.tolist() == [2.25, 2.75] and np.allclose(w[0].p, [[2.25, 5], [2.75, 5]])
    assert "before it starts" in refused(I.window, two_lines(), 5.0, 4.0)


# ---------------------------------------------------------------- the coarse track
def test_from_track_makes_one_stroke_per_run_of_pen_down_frames():
    t = np.arange(10) / 30.0
    xy = np.stack([np.arange(10) * 3.0, np.zeros(10)], 1)
    s = I.from_track(t, xy, [0, 1, 1, 1, 0, 0, 1, 0, 1, 1])
    assert [len(x.t) for x in s] == [3, 1, 2]
    assert s[0].t.tolist() == t[1:4].tolist() and s[2].p.tolist() == xy[8:].tolist()
    assert I.from_track(t, xy, [0] * 10) == [] and len(I.from_track(t, xy, [1] * 10)) == 1


@pytest.mark.parametrize("t, xy, down, fragment", [
    ([0, 1, 2], np.zeros((3, 3)), [1, 1, 1], "positions [3, 3]"),
    ([0, 1, 2], np.zeros((3, 2)), [1, 1], "2 flags"),
    ([0, 2, 1], np.zeros((3, 2)), [1, 1, 1], "must increase"),
])
def test_from_track_refuses_mismatched_samples(t, xy, down, fragment):
    assert fragment in refused(I.from_track, t, xy, down)


def test_front_and_length_report_the_nib_and_the_ink():
    s = [line(0, 10, 5, 11, t0=1.0, dur=10.0), line(0, 4, 8, 5, t0=20.0, dur=4.0)]
    assert np.allclose(I.front(s, 3.5), [2.5, 5.0]) and I.front(s, 15.0) is None and np.allclose(I.front(s, 24.0), [4, 8])
    assert I.length_mm(s) == pytest.approx(14.0)
