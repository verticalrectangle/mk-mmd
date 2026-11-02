"""The mile post (no bpy): the numerals drawn in code (mkmmd.core.numerals: every digit's skeleton in its ink box, the flat
ribbons and the raised strokes made from it) and the prop's layout (mkmmd.blender.library.props.mile_post_layout: the
options and their errors, closed outward solids, the plate growing with the digits, the numerals on the plate, the card's
surface on the plate's face, the form guard)."""
import json

import numpy as np
import pytest

from mkmmd.blender.library.props import mile_post_layout as L
from mkmmd.core import form as F
from mkmmd.core import numerals as NU
from mkmmd.core import shell as S

NAME = "mp"
LOOPS = {"0": 1, "6": 1, "8": 2, "9": 1}            # closed strokes per digit


@pytest.mark.parametrize("d", list(NU.DIGITS))
def test_every_skeleton_keeps_half_a_stroke_inside_its_ink_box_and_reaches_it(d):
    for condense, stroke in ((1.0, NU.STROKE), (L.RR_CONDENSE, L.RR_STROKE)):
        w = NU.width(d, condense, stroke)
        P = np.concatenate([p for p, _ in NU.strokes(d, 1.0, condense, stroke)])
        lo, hi = P.min(0), P.max(0)
        assert lo.min() >= 0.5 * stroke - 1e-9 and hi[0] <= w - 0.5 * stroke + 1e-9 and hi[1] <= 1.0 - 0.5 * stroke + 1e-9
        assert lo[1] == pytest.approx(0.5 * stroke, abs=1e-4) and hi[1] == pytest.approx(1.0 - 0.5 * stroke, abs=1e-4)
        assert hi[0] - lo[0] == pytest.approx(w - stroke, abs=0.02)         # as wide as its box


@pytest.mark.parametrize("d", list(NU.DIGITS))
def test_every_digit_has_its_strokes_and_loops_and_is_deterministic(d):
    st = NU.strokes(d, 0.25)
    assert sum(closed for _, closed in st) == LOOPS.get(d, 0) and len(st) >= 1
    for P, closed in st:
        step = np.linalg.norm(np.diff(np.vstack([P, P[:1]]) if closed else P, axis=0), axis=1)
        assert step.max() <= NU.STEP * 0.25 * 1.01 and step.min() > 0              # evenly sampled, no repeats
    again = NU.strokes(d, 0.25)
    assert all(np.array_equal(a, b) for (a, _), (b, _) in zip(st, again))


def test_series_e_widths_and_condensing_keeps_the_stroke():
    assert NU.width("2") == 0.81 and NU.width("4") == 0.94 and NU.width("0") == 0.84 and NU.width("1") == 0.30
    assert NU.width("2", 0.7, 0.15) == pytest.approx(0.15 + 0.66 * 0.7)
    assert NU.width("1", 0.5) == 0.30                                               # the 1 keeps its width
    with pytest.raises(ValueError, match="numerals: 'x' is not a digit"):
        NU.strokes("x")
    with pytest.raises(ValueError, match="not a string of digits"):
        NU.layout("12a")


@pytest.mark.parametrize("d", list(NU.DIGITS))
def test_ribbons_fill_the_ink_box_flat_and_facing_out(d):
    h = 0.2
    m = NU.ribbons(NU.strokes(d, h), NU.STROKE * h, lift=0.001)
    lo, hi = m.bbox()
    assert lo[:2].min() >= -1e-6 and hi[0] <= NU.width(d) * h + 1e-6               # the ink stays in its box ...
    assert lo[1] == pytest.approx(0.0, abs=1e-3) and hi[1] == pytest.approx(h, abs=1e-3)   # ... cap height to baseline
    assert hi[0] - lo[0] == pytest.approx(NU.width(d) * h, abs=0.02 * h)          # and as wide as it
    assert np.allclose(m.V[:, 2], 0.001)
    nq, nt = m.face_vectors()
    assert (nq[:, 2] > 0).all() and (nt[:, 2] > 0).all()


@pytest.mark.parametrize("d", list(NU.DIGITS))
def test_relief_strokes_are_closed_solids_standing_out_of_the_surface(d):
    h, w = 0.1, 0.015
    m = NU.relief(NU.strokes(d, h, 0.7, 0.15), w, 0.0035, sink=0.0008)
    assert m.is_closed() and m.volume() > 0
    assert m.V[:, 2].min() == pytest.approx(-0.0008) and m.V[:, 2].max() == pytest.approx(0.0035)
    lo, hi = m.bbox()
    assert lo[1] == pytest.approx(0.0, abs=2e-4) and hi[1] == pytest.approx(h, abs=2e-4)


def test_layout_centres_a_row_and_a_column_on_the_origin():
    row = np.concatenate([p for p, _ in NU.layout("707", 1.0)])
    W, H = NU.extent(3, 1.0)
    assert W == pytest.approx(3 * NU.CELL + 2 * NU.GAP) and H == 1.0
    assert (row.min(0) + row.max(0)) / 2 == pytest.approx([0.0, 0.0], abs=0.02)
    col = np.concatenate([p for p, _ in NU.layout("123", 1.0, gap=0.3, column=True)])
    assert NU.extent(3, 1.0, gap=0.3, column=True) == pytest.approx((NU.CELL, 3.6))
    assert col[:, 1].max() == pytest.approx(1.8 - 0.5 * NU.STROKE, abs=1e-6)       # the first numeral on top
    assert col[:, 1].min() == pytest.approx(-1.8 + 0.5 * NU.STROKE, abs=1e-6)


# ================================================================================================= options
def test_the_defaults_are_a_blank_railroad_plate_for_two_numerals():
    o = L.options({"base": "#232136"})                                             # palette slots pass through
    assert (o["style"], o["number"], o["digits"]) == ("railroad", "", 2)
    assert o["roles"] == L.ROLES["railroad"]


@pytest.mark.parametrize("slots, want", [
    ({"number": 7}, ("7", 1)), ({"number": 0}, ("0", 1)), ({"number": 123}, ("123", 3)), ({"number": "007"}, ("007", 3)),
    ({"number": "42", "digits": 3}, ("42", 3)), ({"number": "", "digits": 1}, ("", 1)),
    ({"style": "interstate", "number": 95}, ("95", 2))])
def test_number_and_digits(slots, want):
    o = L.options(slots)
    assert (o["number"], o["digits"]) == want


@pytest.mark.parametrize("slots, msg", [
    ({"style": "highway"}, r"mile_post: style is 'railroad' or 'interstate', got 'highway'"),
    ({"number": -1}, r"mile_post: number -1 is out of range"),
    ({"number": 1000}, r"mile_post: number 1000 is out of range"),
    ({"number": 7.5}, r"mile_post: number is a whole number 0 \.\. 999 or a string"),
    ({"number": True}, r"mile_post: number is a whole number"),
    ({"number": "12a"}, r"mile_post: number '12a' is not a string of 1 to 3 digits"),
    ({"number": "1234"}, r"mile_post: number '1234' is not a string of 1 to 3 digits"),
    ({"digits": 4}, r"mile_post: digits is how many numerals the plate is sized for, 1 \.\. 3, got 4"),
    ({"digits": "2"}, r"mile_post: digits is how many numerals"),
    ({"number": 123, "digits": 2}, r"mile_post: number '123' has 3 digits but the plate is sized for digits = 2"),
    ({"plate": "chartreuse"}, r"mile_post: colour role plate = 'chartreuse'"),
    ({"style": "interstate", "border": "#12"}, r"mile_post: colour role border = '#12'")])
def test_a_bad_option_is_refused_saying_what_to_give(slots, msg):
    with pytest.raises(ValueError, match=msg):
        L.options(slots)


def test_each_style_has_its_own_roles_from_the_palette():
    rr, is_ = L.options({})["roles"], L.options({"style": "interstate"})["roles"]
    assert rr == {"post": "paper", "plate": "ink", "numerals": "paper", "wood": "muted:gold:0.25",
                  "grime": "ink:gold:0.3", "foot": "ink"}
    assert is_ == {"post": "muted:paper:0.35", "plate": "pine", "numerals": "paper", "border": "paper"}
    assert L.options({"style": "interstate", "plate": "foam:pine:0.5", "wood": "nonsense"})["roles"]["plate"] == "foam:pine:0.5"


# ================================================================================================= meshes
STYLES = [{"number": 7}, {"number": "123"}, {}, {"style": "interstate", "number": 123}, {"style": "interstate", "number": 7},
          {"style": "interstate"}]


@pytest.fixture(scope="module", params=range(len(STYLES)), ids=[json.dumps(s) for s in STYLES])
def built(request):
    o = L.options(STYLES[request.param])
    return o, L.parts(o)


def test_every_solid_part_is_closed_and_outward_and_the_sheeting_is_flat(built):
    o, parts = built
    for key, m in parts.items():
        if o["style"] == "interstate" and key in L.GRAPHIC:
            c, n, _ = L.plate_frame("interstate", o["digits"])
            off = (m.V - c) @ n
            assert 0.0 < off.min() and off.max() < 0.001, key                      # just over the panel's face
            nq, nt = m.face_vectors()
            assert all((F_ @ n > 0).all() for F_ in (nq, nt) if len(F_)), key       # facing out of it
            continue
        assert m.is_closed() and m.volume() > 0, key


def test_the_parts_and_the_card_s_objects(built):
    o, parts = built
    want = ["post", "plate"] if o["style"] == "railroad" else ["post", "panel", "border"]
    want += ["numerals"] if o["number"] else []
    assert list(parts) == want
    assert L.card(NAME, o)["objects"] == [f"{NAME}_{k}" for k in want]


def test_the_railroad_post_is_a_six_inch_timber_with_a_hipped_top():
    m = L.post_mesh()
    lo, hi = m.bbox()
    assert lo == pytest.approx([-0.076, -0.076, 0.0], abs=1e-9)
    assert hi == pytest.approx([0.076, 0.076, L.POST_H + L.CAP_RISE], abs=1e-5)
    top = m.V[(m.V[:, 2] > L.POST_H + 1e-3) & (m.V[:, 2] < L.POST_H + 0.88 * L.CAP_RISE - 1e-6)]
    r = np.max(np.abs(top[:, :2]), axis=1)
    assert np.all(r <= 0.076 * (1.0 - (top[:, 2] - L.POST_H) / L.CAP_RISE) + 1e-6)   # the hips run straight in to the point
    w = L.post_check()
    assert len(w) == len(m.V) and w.min() == 0.0 and w.max() == pytest.approx(1.0)   # every check opens fully somewhere
    for k, (face, _a, z0, z1, _w, d) in enumerate(L.CHECKS):
        assert L.check_depth(k, 0.5 * (z0 + z1)) == pytest.approx(d)
        assert L.check_depth(k, z0) == 0.0 and L.check_depth(k, z1) == 0.0
        assert z1 < L.POST_H - L.PLATE_UNDER - L.plate_size("railroad", 3)[1] or face != 0   # the front check stays under the plate


@pytest.mark.parametrize("style", L.STYLES)
def test_the_plate_grows_with_the_digits(style):
    sizes = [L.plate_size(style, n) for n in (1, 2, 3)]
    if style == "railroad":
        assert sizes[0][0] < sizes[1][0] < sizes[2][0] and len({s[1] for s in sizes}) == 1
        assert sizes[0][0] < L.POST_W < sizes[2][0]
    else:
        assert {s[0] for s in sizes} == {12 * L.INCH}
        assert [s[1] / L.INCH for s in sizes] == pytest.approx([17.0, 30.0, 43.0])  # D10 freeway sizes without MILE


def test_the_interstate_panel_stands_on_its_u_channel_four_feet_up():
    o = L.options({"style": "interstate", "number": 42})
    parts = L.parts(o)
    panel, post = parts["panel"].V, parts["post"].V
    w, h = L.plate_size("interstate", 2)
    assert panel[:, 2].min() == pytest.approx(L.PANEL_LOW, abs=1e-6)
    assert panel[:, 2].max() == pytest.approx(L.PANEL_LOW + h, abs=1e-6)
    assert np.ptp(panel[:, 0]) == pytest.approx(w, abs=1e-6)
    assert post[:, 2].min() == 0.0 and post[:, 2].max() == pytest.approx(L.PANEL_LOW + h - L.POST_TOP_UNDER)
    assert np.ptp(post[:, 0]) == pytest.approx(L.CH_B, abs=1e-6) and np.ptp(post[:, 1]) == pytest.approx(L.CH_A, abs=1e-6)
    assert panel[np.abs(panel[:, 0]) < 0.1, 1].max() == pytest.approx(post[:, 1].min(), abs=1e-6)   # its back on the web


# ================================================================================================= the card
@pytest.mark.parametrize("slots", [{"number": 7}, {"digits": 3}, {"style": "interstate", "number": 123},
                                   {"style": "interstate"}])
def test_the_surface_is_the_numeral_area_on_the_plate_s_face(slots):
    o = L.options(slots)
    card = L.card(NAME, o)
    json.dumps(card)
    assert card["origin"] == "floor_center" and card["front"] == "-Y" and card["slots"] == o["roles"]
    (s,) = card["use"]["surface"]
    c, n, up = (np.array(s[k]) for k in ("center", "normal", "up"))
    assert s["name"] == "plate" and n @ up == 0.0 and np.cross(up, n) == pytest.approx([1.0, 0.0, 0.0])  # reads from -Y
    assert s["size"] == pytest.approx(list(L.numeral_area(o["style"], o["digits"])), abs=1e-5)
    parts = L.parts(o)
    plate = S.Probe.from_mesh(parts["plate" if o["style"] == "railroad" else "panel"])
    w, h = s["size"]
    for a, b in ((0.0, 0.0), (0.45 * w, 0.45 * h), (-0.45 * w, -0.45 * h), (0.45 * w, -0.45 * h)):
        p = c + a * np.array([1.0, 0.0, 0.0]) + b * up
        hit = plate.ray(p - 0.1 * n, n)                                            # from behind the face: hits its back
        front = plate.ray(p + 0.1 * n, -n)
        assert hit is not None and front is not None
        assert front[0] == pytest.approx(0.1, abs=2e-4), (a, b)                   # the face, flush with the surface
    pw, ph = L.plate_size(o["style"], o["digits"])
    assert w < pw and h < ph
    if o["number"]:                                                               # the numerals fill the surface
        V = parts["numerals"].V - c
        u, v = V @ np.array([1.0, 0.0, 0.0]), V @ up
        assert np.abs(u).max() <= 0.5 * w + 1e-4 and np.abs(v).max() <= 0.5 * h + 1e-4
        assert np.abs(v).max() == pytest.approx(0.5 * h, abs=1e-3)
        out = V @ n
        assert out.min() > -0.0015 and out.max() < 0.005                           # on the face
    look = {e["name"]: e["point"] for e in card["use"]["look"]}
    assert look["plate"] == s["center"] and look["top"] == [0.0, 0.0, pytest.approx(L.post_top(o["style"], o["digits"]))]
    V = np.concatenate([m.V for m in parts.values()])
    assert card["size"] == pytest.approx(list(V.max(0) - V.min(0)), abs=1e-4)


# ================================================================================================= form
@pytest.mark.parametrize("slots", [{"number": 7}, {}, {"style": "interstate", "number": 123}, {"style": "interstate"}])
def test_form_passes_the_hero_limit_without_a_raise(slots):
    o = L.options(slots)
    V, T, owner, names, off = [], [], [], [], 0
    for k, m in L.parts(o).items():
        V.append(m.V)
        t = m.triangles()
        T.append(t + off)
        owner.append(np.full(len(t), len(names)))
        names.append(k)
        off += len(m.V)
    exempt = L.GRAPHIC if o["style"] == "interstate" else ()
    r = F.analyse(np.concatenate(V), np.concatenate(T), np.concatenate(owner), names, exempt=exempt)
    assert r["score"] < 0.15, (r["score"], r["components"][:3])
    assert "form_max" not in L.card(NAME, o)
