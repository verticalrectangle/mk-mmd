"""Matte maths of the cut effects (mkmmd.matte), on synthetic shapes: the signed distance of an antialiased coverage image, how
a field turns and scales about a pivot (the edge stays one pixel wide, straight and true to the shape at any zoom), the scale
that fills a frame, the pivot kept inside the shape, the cloud of a thought bubble and the painting helpers."""
import numpy as np
import pytest

from mkmmd import matte as M


# ---------------------------------------------------------------- shapes with antialiased coverage
def coverage(h, w, inside, ss=16):
    """Coverage (h, w) of the region where inside(X, Y) holds, by supersampling; X, Y are index coordinates."""
    ys, xs = np.mgrid[0:h * ss, 0:w * ss]
    X, Y = (xs + 0.5) / ss - 0.5, (ys + 0.5) / ss - 0.5
    return inside(X, Y).astype(np.float32).reshape(h, ss, w, ss).mean(axis=(1, 3))


def halfplane(h, w, angle_deg, offset):
    """The side of the line through the frame's middle where (p - mid) . n + offset <= 0; returns (coverage, true distance)."""
    t = np.radians(angle_deg)
    nx, ny = np.cos(t), np.sin(t)
    mid = ((w - 1) / 2.0, (h - 1) / 2.0)
    cov = coverage(h, w, lambda X, Y: (X - mid[0]) * nx + (Y - mid[1]) * ny + offset <= 0)
    ys, xs = np.mgrid[0:h, 0:w]
    return cov, (xs - mid[0]) * nx + (ys - mid[1]) * ny + offset


def disc(h, w, cx, cy, r):
    cov = coverage(h, w, lambda X, Y: (X - cx) ** 2 + (Y - cy) ** 2 <= r * r)
    ys, xs = np.mgrid[0:h, 0:w]
    return cov, np.hypot(xs - cx, ys - cy) - r


def l_shape(h, w, x0, y0, arm, thick):
    """An L: a vertical bar from (x0, y0) down `arm` and a horizontal bar along its foot; (X, Y) -> bool."""
    def inside(X, Y):
        vert = (X >= x0) & (X <= x0 + thick) & (Y >= y0) & (Y <= y0 + arm)
        horiz = (X >= x0) & (X <= x0 + arm) & (Y >= y0 + arm - thick) & (Y <= y0 + arm)
        return vert | horiz
    return coverage(h, w, inside), inside


# ---------------------------------------------------------------- signed distance
@pytest.mark.parametrize("angle", [0, 7, 22.5, 45, 63, 90, 123])
@pytest.mark.parametrize("offset", [0.0, 0.31, -0.77])
def test_the_distance_of_a_straight_edge_is_exact_where_the_coverage_says_so(angle, offset):
    cov, true = halfplane(60, 80, angle, offset)
    sdf = M.signed_distance(cov)
    partial = (cov > M.EPS) & (cov < 1 - M.EPS)
    inner = np.zeros_like(partial)
    inner[4:-4, 4:-4] = True
    if (partial & inner).any():                                              # (none when the edge runs between two columns)
        assert np.abs(sdf - true)[partial & inner].max() < 0.03              # the pixels the edge passes through
    near = (np.abs(true) < 2.5) & inner
    assert np.abs(sdf - true)[near].max() < 0.2 and np.abs(sdf - true)[near].mean() < 0.06


def test_the_distance_of_a_disc_has_the_right_sign_size_and_shape():
    cov, true = disc(80, 80, 40.3, 39.6, 20.4)
    sdf = M.signed_distance(cov)
    assert sdf[40, 40] < -18 and sdf[0, 0] > 25                              # inside negative, outside positive
    near = np.abs(true) < 2
    assert np.abs(sdf - true)[near].mean() < 0.06 and np.abs(sdf - true)[near].max() < 0.3


def test_a_picture_without_antialiasing_puts_the_edge_between_the_pixels():
    a = np.zeros((10, 12), np.float32)
    a[:, :5] = 1.0                                                          # columns 0..4 inside
    sdf = M.signed_distance(a)
    assert sdf[5, 4] == pytest.approx(-0.5, abs=0.02) and sdf[5, 5] == pytest.approx(0.5, abs=0.02)
    assert sdf[5, 0] == pytest.approx(-4.5, abs=0.1) and sdf[5, 11] == pytest.approx(6.5, abs=0.2)


def test_empty_and_full_pictures_are_all_one_side():
    assert (M.signed_distance(np.zeros((6, 7), np.float32)) > 100).all()
    assert (M.signed_distance(np.ones((6, 7), np.float32)) < -100).all()


# ---------------------------------------------------------------- turning and scaling a field
def test_a_scaled_disc_is_a_disc_of_the_scaled_radius_about_the_pivot():
    cov, _ = disc(90, 120, 59.5, 44.5, 9.7)
    field = M.AlphaField(M.signed_distance(cov))
    d = M.placed(field, (120, 90), (59.5, 44.5), (59.5, 44.5), 4.0)
    area = float(M.coverage(d).sum())
    assert area == pytest.approx(np.pi * (4 * 9.7) ** 2, rel=0.01)
    ys, xs = np.mgrid[0:90, 0:120]
    c = M.coverage(d)
    assert (c * xs).sum() / c.sum() == pytest.approx(59.5, abs=0.05) and (c * ys).sum() / c.sum() == pytest.approx(44.5, abs=0.05)


def test_the_pivot_may_land_elsewhere():
    cov, _ = disc(60, 60, 20.0, 30.0, 8.0)
    d = M.placed(M.AlphaField(M.signed_distance(cov)), (60, 60), (20.0, 30.0), (40.0, 15.0), 1.0)
    ys, xs = np.mgrid[0:60, 0:60]
    c = M.coverage(d)
    assert (c * xs).sum() / c.sum() == pytest.approx(40.0, abs=0.05) and (c * ys).sum() / c.sum() == pytest.approx(15.0, abs=0.05)


def test_a_turned_and_scaled_shape_follows_the_shape_and_not_its_box():
    h, w = 120, 160
    cov, inside = l_shape(h, w, 50.0, 30.0, 40.0, 12.0)
    field = M.AlphaField(M.signed_distance(cov))
    pivot, scale, turn = (56.0, 55.0), 2.5, 33.0
    d = M.placed(field, (w, h), pivot, pivot, scale, turn)
    got = M.coverage(d)

    t = np.radians(turn)

    def want_inside(X, Y):                                                  # the L, turned clockwise by `turn` and scaled
        dx, dy = X - pivot[0], Y - pivot[1]
        qx = pivot[0] + (np.cos(t) * dx + np.sin(t) * dy) / scale
        qy = pivot[1] + (-np.sin(t) * dx + np.cos(t) * dy) / scale
        return inside(qx, qy)

    want = coverage(h, w, want_inside, ss=8)
    both, either = np.minimum(got, want).sum(), np.maximum(got, want).sum()
    assert both / either > 0.985
    ys, xs = np.nonzero(want > 0.5)
    box = np.zeros((h, w), bool)
    box[ys.min():ys.max() + 1, xs.min():xs.max() + 1] = True
    notch = box & (want < 0.01)
    assert notch.sum() > 0.3 * box.sum() and got[notch].max() < 0.25 and got[notch].mean() < 0.01    # an L is not its box


def test_a_clockwise_turn_takes_the_top_of_a_bar_to_the_right():
    h, w = 80, 80
    cov = coverage(h, w, lambda X, Y: (np.abs(X - 40) <= 3) & (np.abs(Y - 40) <= 20))      # a tall bar
    field = M.AlphaField(M.signed_distance(cov))
    c = M.coverage(M.placed(field, (w, h), (40.0, 40.0), (40.0, 40.0), 1.0, 90.0))
    ys, xs = np.nonzero(c > 0.5)
    assert xs.max() - xs.min() > 35 and ys.max() - ys.min() < 10            # now a wide bar
    top = M.coverage(M.placed(field, (w, h), (40.0, 40.0), (40.0, 40.0), 1.0, 20.0))
    ys, xs = np.nonzero(top > 0.5)
    assert xs[ys.argmin()] > 40 and xs[ys.argmax()] < 40                    # the top leans right, the foot left


@pytest.mark.parametrize("angle", [0, 12, 30, 45])
def test_zoomed_edges_stay_one_pixel_wide_and_straight(angle):
    h, w, zoom = 54, 96, 16.0
    cov, _ = halfplane(h, w, angle, 0.37)
    field = M.AlphaField(M.signed_distance(cov))
    mid = ((w - 1) / 2.0, (h - 1) / 2.0)
    c = M.coverage(M.placed(field, (w, h), mid, mid, zoom))
    t = np.radians(angle)
    nx, ny = np.cos(t), np.sin(t)
    rows = np.arange(8, h - 8, 3)
    dev, ramp = [], []
    for y in rows:
        row = c[y]
        k = np.nonzero((row[:-1] >= 0.5) & (row[1:] < 0.5))[0]
        assert k.size == 1
        x = k[0] + (row[k[0]] - 0.5) / (row[k[0]] - row[k[0] + 1])
        # the line, in the zoomed frame: n . (mid + (p - mid) / zoom - mid) + 0.37 = 0
        want = mid[0] + zoom * (-0.37 - ny * (y - mid[1]) / zoom) / nx if abs(nx) > 1e-6 else None
        if want is not None:
            dev.append((x - want) * nx)
        ramp.append(int(((row > 0.02) & (row < 0.98)).sum()))
    assert max(ramp) <= 3                                                   # antialiased over a pixel or two, not 16
    if angle != 90:
        assert np.abs(dev).max() < 0.5                                      # straight to half a pixel at 16x


def test_a_finer_map_answers_in_frame_pixels_and_is_the_more_accurate():
    h, w = 40, 60
    cov1, _ = disc(h, w, 30.0, 20.0, 9.0)
    cov2 = coverage(2 * h, 2 * w, lambda X, Y: (X - 60.5) ** 2 + (Y - 40.5) ** 2 <= 18.0 ** 2, ss=8)   # the same disc, twice as fine
    a = M.placed(M.AlphaField(M.signed_distance(cov1)), (w, h), (30.0, 20.0), (30.0, 20.0), 3.0)
    b = M.placed(M.AlphaField(M.signed_distance(cov2), 2.0), (w, h), (30.0, 20.0), (30.0, 20.0), 3.0)
    ys, xs = np.mgrid[0:h, 0:w]
    true = 3.0 * (np.hypot(xs - 30.0, ys - 20.0) / 3.0 - 9.0)                # the disc of radius 27 about the pivot
    near = np.abs(true) < 4
    ea, eb = np.abs(a - true)[near], np.abs(b - true)[near]
    assert eb.max() < 0.4 and ea.max() < 1.2                                # both are the same disc in frame pixels
    assert eb.mean() < ea.mean()


# ---------------------------------------------------------------- filling the frame
def test_the_fill_scale_of_a_disc_is_the_frame_corner_over_its_radius():
    h, w, r = 54, 96, 7.5
    mid = ((w - 1) / 2.0, (h - 1) / 2.0)
    cov, _ = disc(h, w, mid[0], mid[1], r)
    field = M.AlphaField(M.signed_distance(cov))
    s = M.cover_scale(field, (w, h), mid, mid)
    assert s == pytest.approx((np.hypot(mid[0], mid[1]) + 1.5) / r, rel=0.02)
    assert M.coverage(M.placed(field, (w, h), mid, mid, s)).min() == 1.0
    assert M.coverage(M.placed(field, (w, h), mid, mid, s * 0.9)).min() < 1.0


def test_a_pivot_outside_the_shape_can_never_fill_the_frame():
    cov, _ = disc(40, 60, 30.0, 20.0, 8.0)
    field = M.AlphaField(M.signed_distance(cov))
    assert M.cover_scale(field, (60, 40), (5.0, 5.0), (5.0, 5.0)) == float("inf")


def test_a_turned_shape_fills_the_frame_at_the_scale_it_reports():
    h, w = 54, 96
    cov, _ = l_shape(h, w, 30.0, 8.0, 36.0, 14.0)
    field = M.AlphaField(M.signed_distance(cov))
    pivot = (36.0, 30.0)                                                     # inside the L's foot
    for turn in (0.0, 40.0, 90.0):
        s = M.cover_scale(field, (w, h), pivot, pivot, turn)
        assert np.isfinite(s)
        assert M.coverage(M.placed(field, (w, h), pivot, pivot, s * 1.01, turn)).min() == 1.0


# ---------------------------------------------------------------- the pivot
def test_a_centre_inside_the_shape_stays_and_one_outside_moves_in():
    cov, _ = disc(60, 80, 40.0, 30.0, 15.0)
    sdf = M.signed_distance(cov)
    assert M.inside_point(sdf, (40.0, 30.0)) == (40.0, 30.0)
    x, y = M.inside_point(sdf, (70.0, 30.0), depth=2.0)
    assert 52 < x < 56 and y == 30.0 and sdf[int(y), int(x)] <= -2.0        # the nearest pixel at least 2 px deep
    assert M.inside_point(np.full((5, 5), 9.0, np.float32), (2.0, 2.0)) is None


def test_the_deepest_pixel_is_used_when_nothing_is_as_deep_as_asked():
    cov, _ = disc(30, 30, 15.0, 15.0, 3.0)
    sdf = M.signed_distance(cov)
    x, y = M.inside_point(sdf, (0.0, 0.0), depth=10.0)
    assert abs(x - 15) <= 1 and abs(y - 15) <= 1


def test_the_centroid_is_the_centre_of_mass_and_an_empty_frame_has_its_middle():
    a = np.zeros((20, 40), np.float32)
    a[4:8, 10:20] = 1.0
    assert M.centroid(a) == pytest.approx((14.5, 5.5))
    assert M.centroid(np.zeros((20, 40), np.float32)) == pytest.approx((19.5, 9.5))
    assert M.frac_to_px((0.5, 0.5), (40, 20)) == (19.5, 9.5)


# ---------------------------------------------------------------- the cloud and painting
def test_the_cloud_is_scalloped_not_an_ellipse():
    cloud = M.CloudField(120.0, 80.0)
    xs, ys = np.meshgrid(np.arange(-160, 161, dtype=np.float32), np.arange(-120, 121, dtype=np.float32))
    d = cloud.sample(xs, ys)
    assert d[120, 160] < -40 and d[0, 0] > 30                               # deep in the middle, outside at the corner
    ang = np.linspace(0, 2 * np.pi, 720, endpoint=False)
    radius = []
    for a in ang:                                                           # where each ray leaves the shape
        r = np.arange(0, 200)
        px = np.clip((160 + r * np.cos(a)).astype(int), 0, 320)
        py = np.clip((120 + r * np.sin(a)).astype(int), 0, 240)
        radius.append(r[np.nonzero(d[py, px] > 0)[0][0]])
    radius = np.array(radius, float)
    peaks = int(((radius > np.roll(radius, 1)) & (radius >= np.roll(radius, -1)) & (radius > radius.mean())).sum())
    assert peaks >= 8 and radius.max() <= cloud.radius() + 1                # one lump per bump
    assert radius.max() > 0.9 * 120 and radius.min() > 0.7 * 80


def test_a_disc_helper_matches_the_exact_distance_and_crops_to_its_neighbourhood():
    d = M.disc((60, 40), (30.0, 20.0), 6.0)
    ys, xs = np.mgrid[0:40, 0:60]
    true = np.hypot(xs - 30.0, ys - 20.0) - 6.0
    near = np.abs(true) < 3
    assert np.abs(d - true)[near].max() < 1e-5 and (d[true > 30] == M.FAR).all()
    assert np.allclose(M.disc((60, 40), (30.0, 20.0), 6.0, crop=False), true, atol=1e-5)


def test_coverage_and_bands_antialias_over_one_pixel():
    d = np.array([-2.0, -0.5, -0.25, 0.0, 0.25, 0.5, 2.0], np.float32)
    assert M.coverage(d) == pytest.approx([1, 1, 0.75, 0.5, 0.25, 0, 0])
    band = M.inside_band(np.array([-1.0, 0.0, 2.4, 2.9, 3.0, 3.25, 3.5, 5.0], np.float32), 3.0)    # the shape grown by 3 px
    assert band == pytest.approx([1, 1, 1, 0.6, 0.5, 0.25, 0, 0], abs=1e-6)


def test_paint_lays_a_colour_or_a_picture_by_coverage():
    base = np.zeros((2, 2, 3), np.float32)
    out = M.paint(base, (1.0, 0.5, 0.0), np.array([[1.0, 0.5], [0.0, 0.25]], np.float32))
    assert out[0, 0] == pytest.approx([1, 0.5, 0]) and out[0, 1] == pytest.approx([0.5, 0.25, 0])
    assert out[1, 0] == pytest.approx([0, 0, 0]) and out[1, 1] == pytest.approx([0.25, 0.125, 0])
    pic = np.ones((2, 2, 3), np.float32)
    assert M.paint(base, pic, np.full((2, 2), 0.3, np.float32)) == pytest.approx(np.full((2, 2, 3), 0.3))


def test_a_rim_is_the_band_just_outside_the_shape():
    cov, _ = disc(80, 80, 40.0, 40.0, 15.0)
    d = M.placed(M.AlphaField(M.signed_distance(cov)), (80, 80), (40.0, 40.0), (40.0, 40.0), 1.0)
    rim = M.inside_band(d, 4.0) - M.coverage(d)
    ys, xs = np.mgrid[0:80, 0:80]
    r = np.hypot(xs - 40.0, ys - 40.0)
    assert rim[(r > 15.6) & (r < 18.4)].min() > 0.95                        # solid rim, 4 px wide
    assert rim[r < 14.4].max() < 0.01 and rim[r > 20.0].max() < 0.01         # none inside, none beyond
    assert rim.sum() == pytest.approx(np.pi * (19.0 ** 2 - 15.0 ** 2), rel=0.03)


def test_a_picture_scaled_about_its_centre_lands_where_asked():
    img = np.zeros((40, 60, 3), np.float32)
    img[18:22, 28:32] = 1.0                                                # a mark at the centre
    same = M.warp_scaled(img, (29.5, 19.5), 1.0)
    assert np.abs(same - img).max() < 1e-5
    moved = M.warp_scaled(img, (45.0, 10.0), 1.0)
    ys, xs = np.nonzero(moved[..., 0] > 0.5)
    assert xs.mean() == pytest.approx(45.0, abs=0.6) and ys.mean() == pytest.approx(10.0, abs=0.6)
    small = M.warp_scaled(img, (29.5, 19.5), 0.5)
    ys, xs = np.nonzero(small[..., 0] > 0.2)
    assert xs.max() - xs.min() <= 3 and abs(xs.mean() - 29.5) < 1.0


def test_a_picture_shrunk_far_is_averaged_not_aliased():
    stripes = np.zeros((64, 64, 3), np.float32)
    stripes[:, ::2] = 1.0                                                  # one-pixel stripes
    small = M.warp_scaled(stripes, (31.5, 31.5), 0.1)
    core = small[28:36, 28:36]
    assert core.std() < 0.1 and 0.35 < core.mean() < 0.65
