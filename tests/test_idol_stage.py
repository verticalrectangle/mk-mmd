"""The idol stage's layout (numpy only, no Blender): closed outward meshes and a riser that is no box, the card against the
layout (the deck's rest plane, the feet marks, the heart's text surface, the rates), the LED chase maths (u per pattern,
the band travelling with `chase`, a panel's light steady over a cycle) and the option checks."""
import math

import numpy as np
import pytest

from mkmmd.blender.library.props import idol_stage_layout as LAY
from mkmmd.core import form as FM
from mkmmd.core import palette as PAL

PAL_MOON = dict(PAL.get("rose-pine-moon"))


def opts(**kw):
    return LAY.options({**PAL_MOON, **kw})


@pytest.fixture(scope="module")
def stage():
    O = opts()
    return O, LAY.meshes(O)


@pytest.mark.parametrize("steps", ["front", "sides", "none"])
def test_every_part_is_a_closed_outward_mesh(steps):
    M = LAY.meshes(opts(steps=steps))
    assert ("steps" in M) == (steps != "none") and ("bars" in M)
    for k, m in M.items():
        assert m.is_closed() and m.volume() > 0, k


def test_the_parts_stand_where_the_layout_says(stage):
    O, M = stage
    H, (w, d) = O["height"], O["size"]
    lo, hi = M["riser"].bbox()
    assert hi[2] == pytest.approx(H) and lo[2] == pytest.approx(0.0)
    assert hi[0] == pytest.approx(w / 2 + LAY.OVER, abs=0.01) and -lo[1] == pytest.approx(d / 2 + LAY.OVER, abs=0.01)
    n, rise = LAY.flight(O)
    assert n * rise == pytest.approx(H) and rise <= LAY.RISE_MAX
    s_lo, s_hi = M["steps"].bbox()
    assert s_hi[2] == pytest.approx((n - 1) * rise + LAY.STEP_DOME, abs=1e-6)                 # the last tread under the deck
    assert s_lo[1] == pytest.approx(-d / 2 - (n - 1) * LAY.GOING, abs=1e-3)                   # the first step out front
    for i, p in enumerate(O["panels"]):
        V = M[f"panel{i}"].V
        up = V[:, 2] > H + 0.1                                      # above the posts' feet
        assert V[:, 2].min() == pytest.approx(H) and V[up, 1].min() == pytest.approx(p["face_y"], abs=1e-6)
        assert V[up, 1].max() == pytest.approx(p["y"], abs=1e-6)


def test_the_riser_and_the_whole_stage_pass_the_form_guard(stage):
    _O, M = stage
    names = list(M)
    Vs, Ts, owner, off = [], [], [], 0
    for k, nm in enumerate(names):
        T = M[nm].triangles()
        Vs.append(M[nm].V)
        Ts.append(T + off)
        owner.append(np.full(len(T), k))
        off += len(M[nm].V)
    r = FM.analyse(np.concatenate(Vs), np.concatenate(Ts), np.concatenate(owner), names)
    assert r["score"] < 0.05 and r["by_object"]["riser"]["score"] < 0.05


def test_the_card_matches_the_layout(stage):
    O, M = stage
    H = O["height"]
    card = LAY.card("st", O, LAY.roles(PAL_MOON), M)
    assert card["origin"] == "floor_center" and card["front"] == "-Y"
    assert card["rates"] == [{"rate": "speed", "into": "chase", "per": 1.0}]
    assert card["params"] == {"speed": 0.5, "on": 1.0, "bars": 1.0}
    assert card["objects"] == ["st_" + k for k in M]
    lo = np.min([m.V.min(0) for m in M.values()], axis=0)
    hi = np.max([m.V.max(0) for m in M.values()], axis=0)
    assert card["size"] == pytest.approx(hi - lo, abs=1e-3)
    (rest,) = card["use"]["rest"]                                   # the deck: a rectangle on the flat top
    cx, cy, cz = rest["center"]
    w, d = rest["size"]
    flat = LAY.flat_outline(O)
    corners = np.array([(cx + sx * w / 2, cy + sy * d / 2) for sx in (-1, 1) for sy in (-1, 1)])
    assert cz == H and rest["normal"] == [0.0, 0.0, 1.0] and LAY.inside(flat, corners[:, 0], corners[:, 1]).all()
    assert w > 0.8 * O["size"][0] and d > 0.6 * O["size"][1]
    back = min(p["y"] - LAY.DEPTH / 2 - LAY.FOOT_R for p in O["panels"])
    assert cy + d / 2 < back                                        # clear of the panels' post feet
    feet = {f["name"]: f for f in card["use"]["feet"]}
    assert sorted(feet) == ["centre", "left", "right"]
    for f in feet.values():
        assert f["floor_z"] == H and f["L"][2] is None and f["L"][0] > f["R"][0]           # her left is +X
        x, y = (f["L"][0] + f["R"][0]) / 2, f["L"][1]
        assert abs(x - cx) < w / 2 and abs(y - cy) < d / 2                                # on the deck's rest plane
    (surf,) = card["use"]["surface"]                                # the big heart's face, for [[text]]
    p = O["panels"][LAY.big_heart(O)]
    assert surf["name"] == "heart" and surf["normal"] == [0.0, -1.0, 0.0] and surf["up"] == [0.0, 0.0, 1.0]
    assert surf["center"][1] == p["face_y"] and surf["object"] == "st_panel0"
    sx, _sy, sz = surf["center"]
    tw, th = surf["size"]
    pts = np.array([(sx + a * tw / 2, sz + b * th / 2) for a in (-1, 1) for b in (-1, 1)])
    assert LAY.inside(LAY.face(p), pts[:, 0] * 0.999 + sx * 0.001, pts[:, 1] * 0.999 + sz * 0.001).all()
    assert tw * th > 0.2 * p["size"] ** 2
    looks = {lk["name"]: lk["point"] for lk in card["use"]["look"]}
    assert looks["centre"] == [0.0, 0.0, H] and looks["heart"] == [p["at"][0], p["face_y"], p["at"][1]]
    assert card["slots"]["deck"] == "rose" and card["slots"]["ground"] == "ink"


def test_bigger_panels_stand_further_back_on_posts_on_the_flat_deck(stage):
    O, _M = stage
    flat = LAY.flat_outline(O)
    by_size = sorted(O["panels"], key=lambda p: -p["size"])
    assert [p["y"] for p in by_size] == sorted([p["y"] for p in by_size], reverse=True)
    for p in O["panels"]:
        for x, top in p["posts"]:
            y = p["y"] - LAY.DEPTH / 2
            ring = np.array([(x + LAY.FOOT_R * math.cos(a), y + LAY.FOOT_R * math.sin(a))
                             for a in np.linspace(0, 2 * math.pi, 16)])
            assert LAY.inside(flat, ring[:, 0], ring[:, 1]).all()
            assert LAY._lowest(LAY.outline(p["shape"]) * p["size"] + p["at"], x) + LAY.INSERT == pytest.approx(top, abs=1e-4)


# ---------------------------------------------------------------------------------------------------- the chase maths
def panel(pattern, shape="heart", size=2.2, at=(0.0, 2.25)):
    return LAY.options({**PAL_MOON, "panels": [{"shape": shape, "at": list(at), "size": size, "pattern": pattern}]})


def cells(O):
    p = O["panels"][0]
    i, j, x, z = LAY.dots(p, O["pitch"])
    return p, LAY.field(p, O), i, j, x, z


def test_u_per_pattern():
    O = panel("sweep")
    p, f, i, j, x, z = cells(O)
    assert LAY.u_of(LAY.NP, f, i, j) == pytest.approx(x / O["wave"])                       # the stage's x
    zc = 0.33
    gx = (np.arange(-40, 40) + 0.3) * O["pitch"]                    # inside the pixels, off their edges
    _dot, lv = LAY.pixel(LAY.NP, gx, np.full_like(gx, zc), 0.3, O["pitch"], O["wave"], zc)  # the bars: the same sweep
    cx = (np.floor(gx / O["pitch"]) + 0.5) * O["pitch"]
    assert lv == pytest.approx(LAY.band(LAY.NP, LAY.NP.fract(0.3 - cx / O["wave"]), "sweep"))
    for pattern in ("rise", "ring", "spin"):
        O = panel(pattern)
        p, f, i, j, x, z = cells(O)
        u = LAY.u_of(LAY.NP, f, i, j)
        if pattern == "rise":                                       # up the panel, the bands share its dots evenly
            assert np.all(np.diff(u[np.argsort(z)]) >= -1e-9) and u.min() < 0.05 * f["n"] and u.max() > 0.95 * f["n"]
            assert np.histogram(u, bins=f["n"], range=(0, f["n"]))[0] == pytest.approx(len(u) / f["n"], rel=0.12)
        if pattern == "ring":                                       # out from the middle in scaled copies of the face
            r = np.hypot(x - p["at"][0], z - p["at"][1])
            assert u[np.argmin(r)] < 0.02 and u.max() == pytest.approx(f["n"], abs=0.08 * f["n"])
            up = (x > p["at"][0]) & (x - p["at"][0] < 0.03) & (z > p["at"][1])     # one column, up from the middle
            assert np.all(np.diff(u[up][np.argsort(z[up])]) > 0)
        if pattern in ("ring", "spin"):                             # rho: the face's own radius at the dot's angle
            dx, dz = x - p["at"][0], z - p["at"][1]
            R = LAY._polar(LAY.face(p), p["at"], np.arctan2(dz, np.abs(dx)))
            rho2 = (dx * dx + dz * dz) / (R * R)
            assert rho2.max() < 1.0 + 1e-9 and rho2.max() > 0.85    # the dots fill the face out to its rim
            want = f["n"] * rho2 if pattern == "ring" else f["arms"] * np.arctan2(dz, dx) / (2 * math.pi) + rho2
            err = LAY.NP.fract(u - want + 0.5) - 0.5                  # spin: counter-clockwise (u grows with theta)
            assert np.abs(err).max() < 0.02
    O = panel("blink")
    p, f, i, j, x, z = cells(O)
    u = LAY.u_of(LAY.NP, f, i, j)
    assert set(np.round(u, 6)) == {0.0, 0.5} and np.all(u == 0.5 * ((i + j) % 2))


def test_the_band_travels_with_the_chase():
    O = panel("sweep", size=2.2)
    p, f, i, j, x, z = cells(O)
    row = np.abs(z - p["at"][1]) < 0.5 * O["pitch"]
    xs = x[row]
    heads = []
    for chase in (0.0, 0.1, 0.2, 0.3):
        lv = LAY.levels(f, i[row], j[row], chase)
        mid = (xs > -0.4) & (xs < 0.4)
        heads.append(xs[mid][np.argmax(lv[mid])])
    steps = np.diff(heads)
    assert np.all(steps > 0) and steps == pytest.approx(0.1 * O["wave"], abs=1.5 * O["pitch"])   # left to right
    O = panel("ring", size=1.0)                                     # one ring: its head at u = chase - BAND[0]
    p, f, i, j, x, z = cells(O)
    assert f["n"] == 1
    r = np.hypot(x - p["at"][0], z - p["at"][1])
    lit_r = [float(np.average(r, weights=LAY.levels(f, i, j, c) ** 8)) for c in (0.35, 0.55, 0.75)]
    assert lit_r[0] < lit_r[1] < lit_r[2]                           # the ring runs out


@pytest.mark.parametrize("k", range(len(LAY.DEFAULT_PANELS)))
def test_a_default_panel_s_light_is_steady_over_a_cycle(k):
    O = opts()
    p = O["panels"][k]
    f = LAY.field(p, O)
    i, j, _x, _z = LAY.dots(p, O["pitch"])
    mean = np.array([LAY.levels(f, i, j, c).mean() for c in np.linspace(0.0, 1.0, 241)[:-1]])
    assert (mean.max() - mean.min()) / mean.mean() < 0.1           # no panel flashes as the bands move
    assert 0.3 < mean.mean() < 0.6


@pytest.mark.parametrize("pattern", ["ring", "spin", "blink", "rise"])
def test_every_even_pattern_is_steady_on_every_default_panel(pattern):
    for p0 in LAY.DEFAULT_PANELS:
        O = panel(pattern, p0["shape"], p0["size"], p0["at"])
        _p, f, i, j, _x, _z = cells(O)
        mean = np.array([LAY.levels(f, i, j, c).mean() for c in np.linspace(0.0, 1.0, 241)[:-1]])
        assert (mean.max() - mean.min()) / mean.mean() < (0.3 if pattern == "rise" else 0.15), (pattern, p0)


def test_a_steady_speed_is_one_band_per_wave_a_cycle():
    O = panel("sweep")
    _p, f, i, j, _x, _z = cells(O)
    a = LAY.levels(f, i, j, 0.25)
    b = LAY.levels(f, i, j, 1.25)
    assert a == pytest.approx(b)                                    # a whole cycle later, the same picture


# ---------------------------------------------------------------------------------------------------- options
@pytest.mark.parametrize("slots, words", [
    ({"steps": "back"}, "steps is one of"),
    ({"size": [6.0]}, "size is [width, depth]"),
    ({"height": 3.0}, "height is a number 0.35 .. 1.6"),
    ({"speed": "fast"}, "speed is a number"),
    ({"on": 2.0}, "on is a number 0 .. 1"),
    ({"panels": [{"shape": "moon", "at": [0, 2]}]}, "shape is \"heart\" or \"star\""),
    ({"panels": [{"shape": "heart", "at": [0, 2], "spin": 1}]}, "unknown keys ['spin']"),
    ({"panels": [{"shape": "heart", "at": 2}]}, "at is [x, z]"),
    ({"panels": [{"shape": "heart", "at": [0, 1.0], "size": 2.0}]}, "raise at's z"),
    ({"panels": [{"shape": "heart", "at": [5.0, 2.0]}]}, "off the deck"),
    ({"panels": [{"shape": "star", "at": [0, 2], "pattern": "wave"}]}, "pattern is one of"),
    ({"panels": [{"shape": "star", "at": [0, 2], "color": "plaid"}]}, "panels[0]: color"),
    ({"deck": "plaid"}, "idol_stage: deck"),
])
def test_a_bad_option_says_what_to_change(slots, words):
    with pytest.raises(ValueError) as e:
        LAY.options({**PAL_MOON, **slots})
    assert words in str(e.value)


def test_a_panel_colour_may_name_a_role():
    O = opts(heart="foam", panels=[{"shape": "heart", "at": [0.0, 2.0], "color": "heart", "alt": "#102030"}])
    assert O["panels"][0]["color"] == "foam" and O["panels"][0]["alt"] == "#102030"
    assert opts(panels=[])["panels"] == [] and "surface" not in LAY.card("st", opts(panels=[]), LAY.roles(PAL_MOON),
                                                                         LAY.meshes(opts(panels=[])))["use"]
