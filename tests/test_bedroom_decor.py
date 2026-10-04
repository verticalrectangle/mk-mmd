"""Pure maths of the bedroom decor props: the field-expression language (numpy back end), the shape library, the poster
layout (paper bow, pins, tape) and the poster designs (every style at several aspects: palette contrast, distinctness,
node budget). The Blender side (shader compiler, builders) is checked by building and looking."""
import math

import numpy as np
import pytest

from mkmmd.blender.library.props import bedroom_decor_field as F
from mkmmd.blender.library.props import bedroom_decor_layout as LAY
from mkmmd.blender.library.props import bedroom_decor_prints as PR
from mkmmd.blender.library.props import bedroom_decor_shapes as S
from mkmmd.core import palette as PAL

MOON = F.Palette(PAL.get("rose-pine-moon"))


def grid(n=160, aspect=1.0):
    return F.design_grid(aspect, n)


def ev(expr, n=64, aspect=1.0, pal=MOON):
    x, y = grid(n, aspect)
    return F.Evaluator(x, y, pal).run([expr])[0], x, y


def area(mask, n, aspect):
    """Area (design units^2) covered by a coverage mask rendered on an n-row grid."""
    return float(mask.sum()) * (1.0 / n) * (aspect / max(1, round(n * aspect)))


# ------------------------------------------------------------------------------------------------ field
def test_expressions_are_interned_and_folded():
    a, b = F.X + 1.0, F.X + 1.0
    assert a is b
    assert F.c(2.0) + F.c(3.0) is F.c(5.0)
    assert F.mul(F.X, 1.0) is F.X
    assert F.mul(F.Y, 0.0) is F.c(0.0)
    assert F.add(F.Y, 0.0) is F.Y
    assert (F.X * 2.0) is (2.0 * F.X)                       # constants sort second
    assert F.col(love=1.0) is F.col(love=1.0)


def test_topological_order_puts_dependencies_first():
    e = F.sin(F.X * 3.0) + F.len2(F.X, F.Y)
    order = F.topo([e])
    pos = {id(n): i for i, n in enumerate(order)}
    assert all(pos[id(a)] < pos[id(n)] for n in order for a in n.args)
    assert order[-1] is e


def test_scalar_ops_match_numpy():
    x, y = grid(40, 1.5)
    exprs = {
        "sub": (F.X - F.Y, x - y),
        "div0": (F.X / (F.Y - F.Y), np.zeros_like(x)),                  # x / 0 = 0 as in Blender
        "fract": (F.fract(F.X * 3.3), (x * 3.3) - np.floor(x * 3.3)),
        "abs": (abs(F.X), np.abs(x)),
        "sin": (F.sin(F.Y * 5.0), np.sin(y * 5.0)),
        "atan2": (F.atan2(F.Y, F.X), np.arctan2(y, x)),
        "min": (F.fmin(F.X, F.Y), np.minimum(x, y)),
        "len2": (F.len2(F.X, F.Y), np.hypot(x, y)),
        "dot2": (F.dot2(F.X, F.Y, 0.6, -0.8), x * 0.6 - y * 0.8),
    }
    got = F.Evaluator(x, y, MOON).run([e for e, _ in exprs.values()])
    for (name, (_, want)), g in zip(exprs.items(), got):
        assert np.allclose(g, want, atol=2e-5), name


def test_pingpong_is_a_triangle_wave():
    x = np.linspace(-3, 3, 241, dtype=np.float32)
    y = np.zeros_like(x)
    g = F.Evaluator(x, y, MOON).run([F.pingpong(F.X, 0.5)])[0]
    assert g.min() >= -1e-6 and g.max() <= 0.5 + 1e-6
    assert abs(g[120]) < 1e-5                                      # 0 at 0
    assert abs(float(g[np.argmin(np.abs(x - 0.5))]) - 0.5) < 1e-5  # peak at the scale
    assert abs(float(g[np.argmin(np.abs(x - 1.0))])) < 1e-4        # period 2 * scale


def test_remap_clamps_and_accepts_reversed_ranges():
    x = np.array([[-1.0, 0.0, 0.5, 1.0, 2.0]], dtype=np.float32)
    y = np.zeros_like(x)
    r = F.Evaluator(x, y, MOON).run([F.remap(F.X, 0.0, 1.0, 0.0, 10.0), F.remap(F.X, 1.0, 0.0, 0.0, 1.0),
                                     F.remap(F.X, 0.0, 1.0, 0.0, 1.0, smooth=True)])
    assert np.allclose(r[0], [0, 0, 5, 10, 10])
    assert np.allclose(r[1], [1, 1, 0.5, 0, 0])
    assert np.allclose(r[2], [0, 0, 0.5, 1, 1])


def test_xf_places_rotates_and_scales():
    x, y = grid(50, 1.0)
    px, py = F.xf(F.X, F.Y, (0.2, -0.1), math.radians(90), 2.0)
    gx, gy = F.Evaluator(x, y, MOON).run([px, py])
    # the frame's local x axis points along +y after a quarter turn; local = R^T (p - loc) / scale
    assert np.allclose(gx, (y + 0.1) / 2.0, atol=1e-5)
    assert np.allclose(gy, -(x - 0.2) / 2.0, atol=1e-5)


def test_ramps():
    x = np.linspace(0, 1, 11, dtype=np.float32)[None, :]
    y = np.zeros_like(x)
    stops = [(0.0, 0.0), (0.5, 1.0), (1.0, 0.0)]
    lin, const = F.Evaluator(x, y, MOON).run([F.ramp(F.X, stops), F.ramp(F.X, stops, "CONSTANT")])
    assert np.allclose(lin[0], [0, .2, .4, .6, .8, 1, .8, .6, .4, .2, 0], atol=1e-5)
    assert set(np.unique(const)) <= {0.0, 1.0}
    assert const[0, 0] == 0.0 and const[0, 6] == 1.0 and const[0, 10] == 0.0


def test_colour_ops():
    x = np.zeros((2, 3), dtype=np.float32)
    y = np.zeros_like(x)
    a, b = F.col(love=1.0), F.col(foam=1.0)
    half = F.cmix(a, b, 0.5)
    ev_ = F.Evaluator(x, y, MOON)
    ca, cb, ch = ev_.run([a, b, half])
    assert np.allclose(ch, (ca + cb) / 2, atol=1e-6)
    assert F.cmix(a, b, 0.0) is a and F.cmix(a, b, 1.0) is b
    ramp = F.cramp(F.X, [(0.0, a), (1.0, b)])
    r = F.Evaluator(np.array([[0.0, 1.0, 0.5]], np.float32), np.zeros((1, 3), np.float32), MOON).run([ramp])[0]
    assert np.allclose(r[0, 0], ca[0, 0], atol=1e-5) and np.allclose(r[0, 1], cb[0, 0], atol=1e-5)


def test_palette_roles_follow_the_palette():
    moon, dawn = F.Palette(PAL.get("rose-pine-moon")), F.Palette(PAL.get("rose-pine-dawn"))
    assert moon.roles == {"paper": "text", "ink": "base"}               # dark palette: light text, dark base
    assert dawn.roles["paper"] in ("surface", "base") and dawn.roles["ink"] == "text"
    lum = lambda p, role: sum(w * v for w, v in zip((0.2126, 0.7152, 0.0722), p.rgb(F.col(**{role: 1.0}).attrs)))
    assert lum(moon, "paper") > 5 * lum(moon, "ink") and lum(dawn, "paper") > 5 * lum(dawn, "ink")
    assert moon.blend(paper=0.5, gold=0.5)[3] == 1.0


# ------------------------------------------------------------------------------------------------ shapes
def test_shape_areas():
    n, a = 400, 1.0
    cases = {
        "disc": (S.disc(0.2), math.pi * 0.04),
        "ring": (S.ring(0.3, 0.05), math.pi * (0.325 ** 2 - 0.275 ** 2)),
        "rect": (S.rect(0.25, 0.1), 0.5 * 0.2),
        "round rect": (S.rect(0.25, 0.1, rad=0.05), 0.5 * 0.2 - (4 - math.pi) * 0.05 ** 2),
        "triangle": (S.cov(S.D_tri(0.5)(S.P0)), math.sqrt(3) / 4 * 0.25),
        "band": (S.band(F.Y, -0.1, 0.2), 0.3 * 1.0),
        "capsule": (S.line((-0.2, 0), (0.2, 0), 0.1), 0.4 * 0.1 + math.pi * 0.05 ** 2),
    }
    x, y = grid(n, a)
    got = F.Evaluator(x, y, MOON).run([m for m, _ in cases.values()])
    for (name, (_, want)), g in zip(cases.items(), got):
        assert area(g, n, a) == pytest.approx(want, rel=0.03), name


def test_masks_stay_in_unit_range():
    x, y = grid(100, 0.8)
    for m in (S.sprinkle(S.P0, 0.05, 0.01, 0.5), S.zigzag(S.P0, 0.1, 0.03, 0.02), S.checker(S.P0, 0.07),
              S.rays(S.at(S.P0, 0.1, 0.1), 12), S.sparkle(0.1), S.dots(S.P0, 0.06, 0.02, 10, stagger=True)):
        g = F.Evaluator(x, y, MOON).run([m])[0]
        assert np.isfinite(g).all() and g.min() >= -1e-6 and g.max() <= 1 + 1e-6


def test_sprinkle_density_and_determinism():
    n, a = 300, 1.0
    x, y = grid(n, a)
    lo = F.Evaluator(x, y, MOON).run([S.sprinkle(S.P0, 0.05, 0.012, keep=0.2, seed=1)])[0]
    hi = F.Evaluator(x, y, MOON).run([S.sprinkle(S.P0, 0.05, 0.012, keep=0.8, seed=1)])[0]
    other = F.Evaluator(x, y, MOON).run([S.sprinkle(S.P0, 0.05, 0.012, keep=0.8, seed=2)])[0]
    assert 0.1 < lo.sum() / hi.sum() < 0.6 and hi.sum() > 0
    assert not np.allclose(hi, other)
    again = F.Evaluator(x, y, MOON).run([S.sprinkle(S.P0, 0.05, 0.012, keep=0.8, seed=1)])[0]
    assert np.array_equal(hi, again)


def test_sprinkled_shapes_stay_in_their_cell():
    """With r <= (0.5 - jitter) * cell a shape never reaches a cell border, so it covers no point of the border."""
    cell, r = 0.1, 0.015
    for shape in ("disc", "ring", "diamond", "plus"):
        m = S.sprinkle(S.P0, cell, r, keep=1.0, seed=3, shape=shape)
        xs = np.linspace(-0.45, 0.45, 400, dtype=np.float32)
        x, y = np.meshgrid(xs, np.array([-0.3, 0.0, 0.2], np.float32))
        g = F.Evaluator(x, y, MOON).run([m])[0]
        near = np.abs(x - np.round(x / cell) * cell) < 0.0075            # within 7.5 mm of a vertical cell border
        assert g[near].max() < 0.02, shape


def test_block_row_is_inside_its_box_and_chromes_in_unit_height():
    row = S.BlockRow(0.0, 0.1, 0.05, [0.06, 0.04, 0.08], gap=0.01, slant=12.0, seed=3)
    x, y = grid(300, 1.0)
    m, t = F.Evaluator(x, y, MOON).run([row.mask(), row.fill_t()])
    assert m.max() == pytest.approx(1.0, abs=1e-4)
    cover = m > 0.5
    lean = 0.05 / 2 * math.tan(math.radians(12.0))
    assert y[cover].min() >= 0.1 - 0.025 - 1e-3 and y[cover].max() <= 0.1 + 0.025 + 1e-3
    assert x[cover].min() >= row.lo - lean - 2e-3 and x[cover].max() <= row.hi + lean + 2e-3
    inside = t[(y > 0.1 - 0.025) & (y < 0.1 + 0.025)]
    assert inside.min() >= -1e-6 and inside.max() <= 1 + 1e-6


def test_bands_pick_the_intervals():
    y = np.linspace(-0.5, 0.5, 201, dtype=np.float32)[:, None]
    x = np.zeros_like(y)
    m = F.Evaluator(x, y, MOON).run([S.bands(F.Y, -0.5, 0.5, [(-0.3, -0.2), (0.1, 0.25)])])[0][:, 0]
    inside = ((y[:, 0] > -0.3) & (y[:, 0] < -0.2)) | ((y[:, 0] > 0.1) & (y[:, 0] < 0.25))
    assert (m[inside & (np.abs(y[:, 0] + 0.25) < 0.03)] == 1).all() and (m[~inside & (np.abs(y[:, 0]) < 0.05)] == 0).all()
    assert abs(m.mean() - 0.25 / 1.0) < 0.02


def test_ridge_and_poisson():
    pts = S.ridge(5, 16, 0.0, 0.15, valley=0.5, width=0.24, sink=0.05)
    assert len(pts) == 17 and pts[0][0] == 0.0 and pts[-1][0] == 1.0
    assert all(0.0 <= h <= 0.15 for _, h in pts)
    assert all(b[0] > a[0] for a, b in zip(pts, pts[1:]))
    assert max(h for t, h in pts if abs(t - 0.5) < 0.04) <= 0.15 * 0.2          # the valley is low
    assert pts == S.ridge(5, 16, 0.0, 0.15, valley=0.5, width=0.24, sink=0.05)
    p = S.poisson(1, 30, 0.7, 1.0, 0.1, avoid=[(0.0, 0.0, 0.2)], margin=0.05)
    assert len(p) > 10
    assert all(math.hypot(x, y) >= 0.2 and abs(x) <= 0.3 + 1e-9 and abs(y) <= 0.45 + 1e-9 for x, y in p)
    assert min(math.hypot(a[0] - b[0], a[1] - b[1]) for i, a in enumerate(p) for b in p[i + 1:]) >= 0.1


# ------------------------------------------------------------------------------------------------ layout
@pytest.mark.parametrize("w,h", [(0.5, 0.7), (0.6, 0.9), (0.4, 0.3), (0.25, 0.25)])
@pytest.mark.parametrize("mount", ["pins", "tape", "none"])
def test_paper_never_enters_the_wall(w, h, mount):
    zs = np.linspace(-h / 2, h / 2, 41)
    xs = np.linspace(-w / 2, w / 2, 41)
    gaps = np.array([[LAY.sheet_gap(x, z, w, h, mount) for x in xs] for z in zs])
    assert gaps.min() >= LAY.GAP_PIN - 1e-9
    assert gaps.max() <= LAY.GAP_PIN + 0.0085                                      # bow stays shallow
    assert 0.0015 < gaps.mean() < 0.0055                                           # the mean plane stands ~3 mm off


def test_bow_vanishes_at_the_pins_and_the_taped_edge():
    w, h = 0.5, 0.7
    for x, z in LAY.pin_points(w, h):
        assert LAY.bow(x, z, w, h, "pins") == pytest.approx(0.0, abs=1e-12)
    assert len(LAY.pin_points(w, h)) == 4 and len(LAY.pin_points(w, h, torn=("TR", "BL"))) == 2
    for x in np.linspace(-w / 2, w / 2, 9):
        assert LAY.bow(x, h / 2, w, h, "tape") == pytest.approx(0.0, abs=1e-12)
    assert LAY.bow(0.0, -h / 2, w, h, "tape") > 0.003                                # the free end curls out
    corners = LAY.corner_points(w, h)
    assert set(corners) == {"TL", "TR", "BR", "BL"} and corners["TL"][0] < 0 < corners["TR"][0]
    assert corners["TL"][2] > 0 > corners["BL"][2] and all(p[1] < 0 for p in corners.values())


def test_tape_lies_on_the_paper_and_reaches_the_wall():
    w, h = 0.5, 0.7
    for x0, ang in LAY.tape_specs(w, h, 7):
        verts, quads, uvs, org = LAY.tape_strip(x0, ang, w, h, "tape")
        assert len(verts) == len(uvs) == 9 * 25 and len(quads) == 8 * 24
        assert all(0 <= i < len(verts) for q in quads for i in q)
        ys = [v[1] + org[1] for v in verts]
        assert max(ys) < 0.0 and min(ys) > -0.02
        assert any(abs(y + 0.0005) < 1e-4 for y in ys)                                # the wall end
        for v in verts:
            x, y, z = v[0] + org[0], v[1] + org[1], v[2] + org[2]
            if abs(x) <= w / 2 and abs(z) <= h / 2:
                assert y <= -LAY.sheet_gap(x, z, w, h, "tape") + 1e-9                  # never behind the paper's face
    assert len(LAY.tape_specs(w, h, 7, torn=("TL",))) == 1


def test_sheet_grid_resolution():
    assert LAY.sheet_grid(0.5, 0.7) == (25, 35)
    nx, ny = LAY.sheet_grid(5.0, 5.0)
    assert nx <= 72 and ny <= 72


# ------------------------------------------------------------------------------------------------ designs
ASPECTS = (0.4 / 0.6, 0.5 / 0.7, 0.6 / 0.9, 1.0, 0.4 / 0.3)


def count_stops(roots):
    n = 0
    for e in F.topo(roots):
        if e.op == "ramp":
            st = e.attrs[0]
            pos = [p for p, _ in st]
            n += 1
            assert len(st) <= 32 and all(0.0 <= p <= 1.0 for p in pos)
            assert all(b > a for a, b in zip(pos, pos[1:])), "ramp stops must increase"
    return n


@pytest.mark.parametrize("style", sorted(PR.STYLES))
@pytest.mark.parametrize("aspect", ASPECTS)
def test_every_style_builds_and_renders_finite(style, aspect):
    d = PR.build(style, aspect, 0, 0)
    fin = PR.finish(d, aspect, aspect + 0.1, 1.1, "cross", ("TR",))
    roots = [d.color] + list(fin[k] for k in ("albedo", "rough", "spec", "height", "alpha"))
    assert 0 < len(F.topo(roots)) < 1400
    count_stops(roots)
    img, alb, rough, spec, hgt, alpha = F.render(roots, aspect + 0.1, 48, MOON)
    for a in (img, alb, rough, spec, hgt, alpha):
        assert np.isfinite(a).all()
    assert rough.min() >= 0.0 and rough.max() <= 1.0 and spec.min() >= 0.0 and spec.max() <= 1.0
    assert alpha.min() < 0.05 and alpha.max() > 0.95                                  # a torn corner and a sheet


@pytest.mark.parametrize("style", sorted(PR.STYLES))
def test_designs_read_in_a_dim_room(style):
    """Mid-value ink with bright accents: neither murky nor blown out, nothing near-black, on the moon palette."""
    a = 0.5 / 0.7
    d = PR.build(style, a, 0, 0)
    img = F.render([d.color], a, 160, MOON)[0]
    lum = img @ np.array([0.2126, 0.7152, 0.0722], np.float32)
    base = MOON.rgb(F.col(base=1.0).attrs)
    base_lum = 0.2126 * base[0] + 0.7152 * base[1] + 0.0722 * base[2]
    assert lum.min() >= 0.8 * base_lum                                               # the darkest tone is the palette's base
    assert (lum < 1.6 * base_lum).mean() < 0.06                                      # almost nothing near the base
    assert 0.06 < float(np.median(lum)) < 0.6                                        # mid-value body
    assert np.percentile(lum, 98) > 2.0 * float(np.median(lum)) or np.percentile(lum, 98) > 0.45    # bright accents
    assert F.srgb8(img).reshape(-1, 3).std(axis=0).min() > 18                         # colourful, not flat


def test_styles_are_distinct_and_variants_differ():
    a = 0.5 / 0.7
    imgs = {s: F.render([PR.build(s, a, 0, 0).color], a, 64, MOON)[0] for s in PR.STYLES}
    names = sorted(imgs)
    assert len(names) >= 5 and {"sunset_grid", "memphis", "trio", "car", "sunburst"} <= set(names)
    for i, s in enumerate(names):
        for t in names[i + 1:]:
            assert np.abs(imgs[s] - imgs[t]).mean() > 0.04, (s, t)
    for s in names:
        v1 = F.render([PR.build(s, a, 0, 1).color], a, 64, MOON)[0]
        assert np.abs(v1 - imgs[s]).mean() > 0.01, s


def test_designs_are_deterministic_and_seed_matters():
    a = 0.5 / 0.7
    r = lambda seed: F.render([PR.build("sunset_grid", a, seed, 0).color], a, 64, MOON)[0]
    assert np.array_equal(r(0), r(0))
    assert not np.array_equal(r(0), r(3))


def test_unknown_style_is_an_error():
    with pytest.raises(KeyError):
        PR.build("no_such_style", 0.7)


def test_finish_folds_margin_and_torn_corners():
    a, sa, sh = 0.7, 0.8, 1.1
    d = PR.build("memphis", a, 0, 0)
    fin = PR.finish(d, a, sa, sh, "cross", ("TR",))
    roots = [fin["height"], fin["alpha"], fin["ink"]]
    x = np.array([[0.0, 0.2, sa / 2 - 0.005, 0.0]], np.float32)
    y = np.array([[0.3, 0.0, sh / 2 - 0.005, 0.0]], np.float32)
    h, al, ink = F.Evaluator(x, y, MOON).run(roots)
    assert h[0, 0] == pytest.approx(-1.0, abs=1e-5) and h[0, 1] == pytest.approx(-1.0, abs=1e-5)   # on the creases
    assert al[0, 2] == 0.0 and al[0, 0] == 1.0                                                   # torn corner, sheet
    assert ink[0, 0] == 1.0 and ink[0, 2] == 0.0                                                   # margin is paper
    plain = PR.finish(d, a, sa, sh, "none", ())
    h2, al2 = F.Evaluator(x, y, MOON).run([plain["height"], plain["alpha"]])
    assert (h2 == 0).all() and (al2 == 1).all()


# ------------------------------------------------------------------------------------------------ clock, nightstand, shelf
def test_parse_time():
    assert LAY.parse_time("02:47") == ("0", "2", "4", "7")
    assert LAY.parse_time("2:47") == (" ", "2", "4", "7")
    assert LAY.parse_time(" 23:59 ") == ("2", "3", "5", "9")
    for bad in ("2447", "24:00", "12:5", "ab:cd", "1:2:3", "", "12:60"):
        with pytest.raises(ValueError):
            LAY.parse_time(bad)


def test_seven_segment_digits_and_geometry():
    assert set("".join(LAY.DIGIT_SEGMENTS.values())) <= set("abcdefg")
    assert len(LAY.DIGIT_SEGMENTS["8"]) == 7 and LAY.DIGIT_SEGMENTS[" "] == "" and len(LAY.DIGIT_SEGMENTS) == 11
    assert {len(LAY.DIGIT_SEGMENTS[str(n)]) for n in range(10)} == {2, 3, 4, 5, 6, 7}        # 1 | 7 | 4 | 2 3 5 | 0 6 9 | 8
    polys = LAY.segment_polygons()
    assert sorted(polys) == list("abcdefg")
    for s, pts in polys.items():
        assert len(pts) == 6 and LAY.poly_area(pts) > 0, s                                   # counter-clockwise hexagons
    xs = [x for pts in polys.values() for x, _ in pts]
    ys = [y for pts in polys.values() for _, y in pts]
    assert max(ys) <= LAY.DIGIT_H / 2 + 1e-9 and min(ys) >= -LAY.DIGIT_H / 2 - 1e-9
    assert max(xs) - min(xs) < LAY.DIGIT_W * 1.2
    cells = LAY.display_cells()
    assert cells["width"] < LAY.PLATE_W and cells["colon"] == pytest.approx(0.0, abs=1e-12)
    assert list(cells["digits"]) == sorted(cells["digits"]) and cells["digits"][0] == pytest.approx(-cells["digits"][3])


def test_clock_face_and_top():
    phi, (fy, fz), length, n, up = LAY.face_frame()
    assert 0.2 < phi < 0.35 and length > LAY.PLATE_H
    assert sum(a * b for a, b in zip(n, up)) == pytest.approx(0.0, abs=1e-9) and n[1] < 0 < n[2]
    assert LAY.top_z(LAY.CLOCK_FRONT_TOP[0]) == pytest.approx(LAY.CLOCK_FRONT_TOP[1])
    assert LAY.top_z(LAY.CLOCK_BACK_TOP[0]) == pytest.approx(LAY.CLOCK_BACK_TOP[1])


def test_nightstand_legs_fit_under_the_top():
    for sx in (-1, 1):
        for sy in (-1, 1):
            top, bot = LAY.leg_points(sx, sy)
            assert abs(bot[0]) + 0.0125 <= LAY.NS_W / 2 and abs(bot[1]) + 0.0125 <= LAY.NS_D / 2          # foot cap radius
            assert top[2] > LAY.NS_BODY_Z[0] and bot[2] < 0.02
            x, y = LAY.leg_at(sx, sy, LAY.NS_SHELF_Z)
            assert bot[0] * sx > x * sx > top[0] * sx and abs(y) < abs(bot[1])
    assert LAY.NS_BODY_W < LAY.NS_W and LAY.NS_BODY_D < LAY.NS_D and LAY.NS_BODY_Z[1] < LAY.NS_H - LAY.NS_TOP_T + 1e-9


def test_shelf_layout_is_deterministic_and_fits():
    a, b = LAY.shelf_layout(0), LAY.shelf_layout(0)
    assert a == b and a != LAY.shelf_layout(3)
    books = a["books"]
    assert len(books) == 9
    for p, q in zip(books, books[1:]):
        assert q["x"] >= p["x"] + p["w"] - 1e-9                                             # no overlap along the board
    assert all(0.15 <= bk["h"] <= 0.27 and 0.13 <= bk["d"] <= 0.165 for bk in books)
    assert books[-1]["lean_deg"] > 5 and all(bk["lean_deg"] == 0 for bk in books[:-1])
    assert books[-1]["h"] < books[-2]["h"] + 0.05 or books[-2]["h"] > 0.9 * books[-1]["h"]   # the neighbour is tall enough
    assert a["books_x"][0] >= -LAY.SH_W / 2 + 0.02
    (t0, t1), (f0, f1) = a["tapes_x"], a["free"]
    assert a["books_x"][1] < t0 < t1 < f0 < f1 < a["ornament_x"] - 0.03
    assert f1 - f0 >= 0.18 and a["ornament_x"] + 0.03 < LAY.SH_W / 2                         # room for the 16 cm clock
    assert len(a["tapes"]) == 4 and {t["slot"] for t in a["tapes"]} <= set(LAY.TAPE_SLOTS)


def test_rounded_outlines():
    sq = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
    assert LAY.rounded_poly(sq, 0.0) == sq                                              # radius 0 keeps sharp corners
    r = LAY.rounded_rect(0.05, 0.03, 0.0028)
    assert len(r) == 24 and LAY.poly_area(r) > 0
    assert LAY.poly_area(r) == pytest.approx(0.1 * 0.06 - (4 - math.pi) * 0.0028 ** 2, rel=0.002)
    assert all(abs(x) <= 0.05 + 1e-12 and abs(y) <= 0.03 + 1e-12 for x, y in r)
    big = LAY.rounded_poly(sq, 5.0)                                                     # too large a radius is clamped
    assert all(-1e-9 <= x <= 1 + 1e-9 and -1e-9 <= y <= 1 + 1e-9 for x, y in big)
    assert min(math.hypot(a[0] - b[0], a[1] - b[1]) for a, b in zip(big, big[1:] + big[:1])) > 1e-4   # no zero-length edge
    one = LAY.rounded_poly(sq, [0.2, 0.0, 0.0, 0.0], 4)                                 # only the first corner rounded
    assert len(one) == 5 + 3 and (1.0, 0.0) in one and (0.0, 1.0) in one and (0.0, 0.0) not in one


def test_book_profiles():
    for w, d in ((0.016, 0.13), (0.026, 0.15), (0.038, 0.165)):
        case, block = LAY.book_case_outline(w, d), LAY.book_block_outline(w, d)
        assert LAY.poly_area(case) > 0 and LAY.poly_area(block) > 0
        sag = LAY.book_sag(w)
        assert min(y for _, y in case) == pytest.approx(-d) and 0.0 < sag <= 0.0065
        assert LAY.spine_point(w, d, 0.0) == pytest.approx((0.0, -d + sag)) and LAY.spine_point(w, d, w)[1] == pytest.approx(-d + sag)
        assert LAY.spine_point(w, d, w / 2) == pytest.approx((w / 2, -d))
        assert LAY.spine_point(w, d, w / 2, 0.001)[1] == pytest.approx(-d - 0.001)         # an offset moves outward
        assert min(x for x, _ in block) > LAY.BOOK_COVER_T and max(x for x, _ in block) < w - LAY.BOOK_COVER_T
        assert max(y for _, y in block) <= -LAY.BOOK_OVER + 0.0008 + 1e-9                  # fore-edge inset by the overhang
        assert min(y for _, y in block) > -d + 0.0008                                      # inside the spine
        assert LAY.poly_area(block) > 0.55 * w * d * 0.8                                   # the page block fills the case
    assert LAY.CASSETTE[0] <= LAY.TAPE_W and LAY.CASSETTE[2] * 4 < 0.06 and LAY.CASSETTE_CORNER < LAY.CASSETTE[2]


# ------------------------------------------------------------------------------------------------ soft forms and `form`
def _tris(mesh):
    T = mesh.triangles()
    return mesh.V, T


def test_soft_slab_is_closed_rolled_and_outward():
    from mkmmd.blender.library.props import bedroom_decor_soft as SOFT
    from mkmmd.core import shell as SH
    m = SOFT.soft_slab(SH.rrect(0.4, 0.35, 0.045, 6), 0.025, 0.011, 0.006)
    assert m.is_closed() and m.volume() > 0
    lo, hi = m.bbox()
    assert np.allclose(lo, [-0.2, -0.175, 0.0]) and np.allclose(hi, [0.2, 0.175, 0.025])
    prof = SOFT.slab_profile(0.025, 0.011, 0.006)
    assert prof[0] == pytest.approx((0.006 * 1.0 * min(1.0, 0.98 * 0.025 / 0.017), 0.0), abs=1e-9) and prof[-1][1] == pytest.approx(0.025)
    assert all(b[1] >= a[1] - 1e-12 for a, b in zip(prof, prof[1:]))                       # rises monotonically
    crowned = SOFT.soft_slab(SH.rrect(0.34, 0.098, 0.03, 6), 0.012, 0.005, 0.003, 4, dome=0.004, cap_rings=3)
    assert crowned.is_closed() and crowned.bbox()[1][2] == pytest.approx(0.012 + 0.004, abs=1e-6)
    roles = SOFT.set_roles(m.copy(), lambda c, n: (n[:, 2] > 0.3).astype(int))
    assert set(np.unique(np.concatenate([roles.Qm, roles.Tm]))) == {0, 1}


def test_clock_shell_closed_and_inside_its_box():
    from mkmmd.blender.library.props import bedroom_decor_soft as SOFT
    m = SOFT.clock_shell()
    assert m.is_closed() and m.volume() > 0
    lo, hi = m.bbox()
    assert lo[0] == pytest.approx(-LAY.CLOCK_W / 2) and hi[0] == pytest.approx(LAY.CLOCK_W / 2)
    assert LAY.CLOCK_FRONT_BOTTOM[0] <= lo[1] < LAY.CLOCK_FRONT_BOTTOM[0] + 0.0015     # the 3 mm fillet rounds the corner
    assert hi[1] == pytest.approx(LAY.CLOCK_BACK_TOP[0])
    assert lo[2] == pytest.approx(LAY.CLOCK_FRONT_BOTTOM[1]) and LAY.CLOCK_BACK_TOP[1] - 0.004 < hi[2] <= LAY.CLOCK_BACK_TOP[1]
    prof = np.asarray(LAY.clock_profile())
    assert LAY.poly_area([tuple(p) for p in prof]) > 0                                      # counter-clockwise (y, z)


def test_rolled_parts_are_not_boxes():
    """The parts that used to be bevelled cuboids score below 0.1 on the `form` metric (core.form, as the check)."""
    from mkmmd.blender.library.props import bedroom_decor_soft as SOFT
    from mkmmd.core import form as FORM
    from mkmmd.core import shell as SH

    def score(mesh):
        V, T = _tris(mesh)
        return FORM.analyse(V, T)["score"]

    carcass = SOFT.soft_slab(SH.rrect(LAY.NS_BODY_W, LAY.NS_BODY_D, 0.05, 6), LAY.NS_BODY_Z[1] - LAY.NS_BODY_Z[0], 0.026, 0.022, 5)
    assert score(carcass) < 0.1
    assert score(SOFT.clock_shell()) < 0.1
    plinth = SOFT.soft_slab(SH.rrect(LAY.CLOCK_W - 0.001, 0.0665, 0.014, 6), 0.0124, 0.0035, 0.003, 3)
    assert score(plinth) < 0.1
    box = SH.rounded_box((LAY.NS_BODY_W, LAY.NS_BODY_D, 0.12), r=0.003)                     # a plain bevelled cuboid
    assert score(box) > 0.5
