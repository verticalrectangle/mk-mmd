"""The turntable prop (pure numpy: the Blender half is in tests/test_turntable_build.py): the card's label surfaces and points,
the colour roles, the tonearm's kinematics (the rest, the lead-in groove, the side played, the cue's lift), the drivers'
expressions against that maths, the meshes against the numbers they share (the stylus on the playing surface, the rest and
the cue bar under the tube) and the form guard."""
import json
import math
import re
import time

import numpy as np
import pytest

from mkmmd.blender.library.props import turntable_layout as L
from mkmmd.blender.library.props import turntable_parts as P
from mkmmd.core import form as F
from mkmmd.core import palette as PAL
from mkmmd.core import shell as S

NAME = "deck"


@pytest.fixture(scope="module")
def parts():
    return P.parts()


def radius(p):
    return math.hypot(p[0] - L.SPINDLE[0], p[1] - L.SPINDLE[1])


def world(part, R=None, t=None):
    """A part's mesh in the prop's frame (its base rotation, or the rotation R and origin t given)."""
    R = S.rot_matrix(*part.rot) if R is None else R
    return part.mesh.V @ np.asarray(R).T + np.asarray(part.origin if t is None else t)


# ======================================================================================================= the card
def test_the_card_is_plain_json_with_its_uses_rates_and_starting_values():
    card = L.card(NAME, L.Pal().roles(), L.options({"rpm": 45}))
    json.dumps(card)
    assert card["origin"] == "floor_center" and card["front"] == "-Y"
    assert {k: [e["name"] for e in v] for k, v in card["use"].items()} == {
        "surface": ["label", "label_lower"], "look": ["record", "stylus"], "anchor": ["spindle", "stylus"]}
    assert card["rates"] == [{"rate": "rpm", "into": "turns", "per": 60.0}]
    assert card["params"] == {"rpm": 45.0, "arm": 0.0, "cue": 0.0}
    assert set(card["slots"]) == set(L.ROLES)


def test_the_label_panels_ride_the_platter_and_fit_between_the_adapter_and_the_label_edge():
    for s in L.label_panels(NAME):
        assert s["object"] == f"{NAME}_platter"
        assert s["normal"] == [0.0, 0.0, 1.0] and s["up"] == [0.0, 1.0, 0.0]          # read from the front, from above
        cx, cy, cz = s["center"]
        assert cz == pytest.approx(L.Z_LABEL, abs=1e-5) and L.Z_LABEL > L.Z_MAT + L.LABEL_AREA_T   # on the paper
        w, h = s["size"]
        corners = [(cx + sx * w / 2, cy + sy * h / 2) for sx in (-1, 1) for sy in (-1, 1)]
        far = max(radius(c) for c in corners)
        near = min(abs(cy - L.SPINDLE[1]) - h / 2, 1.0)                                 # the edge nearest the hole
        assert far <= L.LABEL_R - 0.0015                                                # inside the printed area
        assert near >= L.ADAPTER_R + 0.002                                              # clear of the adapter
    up, low = L.label_panels(NAME)
    assert up["center"][1] > L.SPINDLE[1] > low["center"][1]


def test_the_look_and_anchor_points():
    card = L.card(NAME)
    rec = next(e for e in card["use"]["look"] if e["name"] == "record")
    assert rec["point"] == pytest.approx([L.SPINDLE[0], L.SPINDLE[1], L.Z_LABEL], abs=1e-5)
    for kind in ("look", "anchor"):
        st = next(e for e in card["use"][kind] if e["name"] == "stylus")
        assert st["object"] == f"{NAME}_stylus" and st["point"] == pytest.approx(list(L.stylus(0, 0)), abs=1e-5)
    sp = next(e for e in card["use"]["anchor"] if e["name"] == "spindle")
    assert sp["point"][2] == pytest.approx(L.SPINDLE_TOP, abs=1e-5) and sp["dir"] == [0.0, 0.0, 1.0]


@pytest.mark.parametrize("slots, frag", [
    ({"arm": 2.5}, "arm is 0"),
    ({"arm": -0.1}, "arm is 0"),
    ({"cue": 1.5}, "cue is 0"),
    ({"rpm": "fast"}, "rpm must be a number"),
    ({"cue": True}, "cue must be a number"),
    ({"rpm": float("nan")}, "rpm must be a number"),
])
def test_a_bad_starting_value_is_refused_saying_what_it_is(slots, frag):
    with pytest.raises(ValueError, match=frag):
        L.options(slots)


def test_negative_rpm_is_allowed_a_record_turned_back():
    assert L.options({"rpm": -45})["rpm"] == -45.0


# ======================================================================================================= colours
def lum(c):
    return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]


@pytest.mark.parametrize("palette", sorted(PAL.PALETTES))
def test_ink_is_dark_and_paper_light_in_a_light_palette_as_in_a_dark_one(palette):
    pal = L.Pal(PAL.get(palette))
    ink, paper = pal.lin("vinyl"), pal.lin("paper")
    assert lum(ink) < 0.05 and min(ink) > 0.0                                          # near black, never black
    assert lum(paper) > 0.6
    assert lum(pal.lin("body")) == pytest.approx(lum(ink))
    assert lum(pal.lin("plate")) > 4 * lum(ink)                                        # the silver deck on a dark base


def test_roles_take_slots_hexes_blends_and_other_roles():
    pal = L.Pal({**PAL.get("rose-pine-dawn"), "label": "pine", "vinyl": "#ff0000"})
    assert pal.lin("label") == pytest.approx(PAL.linear(PAL.get("rose-pine-dawn")["pine"]))
    assert pal.lin("vinyl") == pytest.approx(PAL.linear("#ff0000"))
    mix = [(3 * a + b) / 4 for a, b in zip(pal.lin("label"), pal.lin("ink"))]
    assert pal.lin("print") == pytest.approx(mix)                                      # print = label:3,ink:1 follows the label
    with pytest.raises(ValueError, match="loop"):
        L.Pal({"label": "print", "print": "label:2,ink:1"}).lin("label")


# ======================================================================================================= the tonearm
def test_the_pivot_is_the_overhang_short_of_the_arm_from_the_spindle():
    assert radius(L.PIVOT) == pytest.approx(L.ARM_L - L.OVERHANG)
    tip = (L.PIVOT[0] + L.ARM_L * math.cos(L.THETA_SP), L.PIVOT[1] + L.ARM_L * math.sin(L.THETA_SP))
    assert radius(tip) == pytest.approx(L.OVERHANG)                                    # aimed at the spindle: past it


def test_on_its_rest_the_stylus_is_off_the_platter_and_over_the_lead_in_at_one():
    rest = L.stylus(0.0, 0.0)
    assert radius(rest) > L.PLATTER_R + 0.04
    assert abs(rest[0]) < L.W / 2 - 0.02 and abs(rest[1]) < L.D / 2 - 0.02            # still over the plinth
    lead = L.stylus(1.0, 0.0)
    assert radius(lead) == pytest.approx(L.LEADIN_R, abs=1e-9)
    assert L.GROOVE_OUT < radius(lead) < L.REC_R                                       # outside the music, on the record
    assert radius(L.stylus(2.0, 0.0)) == pytest.approx(L.GROOVE_IN, abs=1e-9)
    assert radius(L.stylus(3.0, 0.0)) == pytest.approx(L.GROOVE_IN, abs=1e-9)          # held past the last groove


def test_the_swing_is_even_in_angle_then_the_stylus_moves_in_at_a_steady_pitch():
    a = np.linspace(0.0, 1.0, 11)
    th = np.array([L.yaw(v) for v in a])
    assert np.diff(th) == pytest.approx(np.full(10, (L.THETA_LEAD - L.THETA_REST) / 10))
    assert L.THETA_LEAD < L.THETA_REST                                                 # clockwise seen from above
    b = np.linspace(1.0, 2.0, 11)
    r = np.array([radius(L.stylus(v, 0.0)) for v in b])
    assert np.diff(r) == pytest.approx(np.full(10, (L.GROOVE_IN - L.LEADIN_R) / 10), abs=1e-9)


def test_down_the_stylus_is_on_the_playing_surface_and_the_cue_lifts_it():
    for arm in (0.0, 0.5, 1.0, 1.4, 2.0):
        assert L.stylus(arm, 0.0)[2] == pytest.approx(L.Z_REC, abs=1e-12)
        up = L.stylus(arm, 1.0)
        assert up[2] - L.Z_REC == pytest.approx(L.LIFT_MM / 1000.0, abs=2e-4)
        assert L.LIFT_MM > 8.0
    assert L.stylus(1.0, 2.0)[2] == pytest.approx(L.stylus(1.0, 1.0)[2])               # clamped


def test_a_stylus_frame_is_a_rotation():
    R = L.arm_matrix(1.3, 0.6)
    assert R @ R.T == pytest.approx(np.eye(3), abs=1e-12) and np.linalg.det(R) == pytest.approx(1.0)


# ======================================================================================================= drivers
VARS = {"yaw": "arm", "lift": "cue", "lever": "cue", "bar": "cue", "spin": "turns", "led33": "rpm", "led45": "rpm"}
ALLOWED = {"clamp", "acos", "pow", "abs", "if", "else"}


@pytest.mark.parametrize("key", sorted(VARS))
def test_every_driver_is_short_and_uses_only_its_property_and_simple_functions(key):
    expr = L.EXPR[key]
    assert len(expr) <= 255
    assert set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", expr)) - ALLOWED == {VARS[key]}
    assert "**" not in expr and "%" not in expr


def ev(key, value):
    return eval(L.EXPR[key], dict(L.SIMPLE), {VARS[key]: value})       # noqa: S307 - our own expression


def test_the_drivers_compute_the_layout_s_maths():
    for arm in np.linspace(-0.5, 2.5, 31):
        assert ev("yaw", arm) == pytest.approx(L.yaw(arm), abs=1e-6)
    for cue in np.linspace(-0.5, 1.5, 9):
        assert ev("lift", cue) == pytest.approx(-L.lift(cue), abs=1e-7)
        assert ev("bar", cue) == pytest.approx(L.BAR_RISE * min(max(cue, 0.0), 1.0), abs=1e-7)
    assert ev("spin", 0.25) == pytest.approx(-0.5 * math.pi, abs=1e-6)                 # clockwise from above
    for rpm, lit in ((0.0, "led33"), (33.3, "led33"), (45.0, "led45"), (-45.0, "led45")):
        assert ev(lit, rpm) == 1.0 and ev("led45" if lit == "led33" else "led33", rpm) == 0.0


# ======================================================================================================= meshes
def test_every_solid_part_is_closed_and_outward(parts):
    for key, part in parts.items():
        if key in P.GRAPHIC:
            continue
        assert part.mesh.is_closed(), key
        assert part.mesh.volume() > 0, key


def test_the_arm_s_lowest_point_is_the_stylus_tip(parts):
    V = parts["arm"].mesh.V
    i = int(np.argmin(V[:, 2]))
    assert V[i] == pytest.approx(np.array(L.STYLUS_LOCAL), abs=2e-4)


def test_down_on_the_record_no_part_of_the_arm_goes_into_it(parts):
    arm = parts["arm"]
    for a in (1.0, 1.5, 2.0):
        V = world(arm, L.arm_matrix(a, 0.0))
        on_record = np.hypot(V[:, 0] - L.SPINDLE[0], V[:, 1] - L.SPINDLE[1]) < L.REC_R
        assert V[on_record, 2].min() >= L.Z_REC - 1e-6


def test_the_record_s_surface_at_the_lead_in_is_the_playing_surface(parts):
    rec = parts["record"]
    probe = S.Probe(world(rec), rec.mesh.triangles())
    for r in (L.LEADIN_R, 0.07, L.GROOVE_IN):
        z = probe.height(L.SPINDLE[0] + r, L.SPINDLE[1])
        assert z == pytest.approx(L.Z_REC, abs=2e-5), r
    assert probe.height(L.SPINDLE[0] + L.REC_R - 0.0005, L.SPINDLE[1]) > L.Z_REC        # the bead at the rim
    assert probe.height(L.SPINDLE[0] + 0.03, L.SPINDLE[1]) > L.Z_REC                   # the raised label area


def test_the_parked_tube_lies_on_its_rest_and_the_lifted_tube_clears_its_fork(parts):
    base = parts["arm_base"].mesh
    rx, ry, rz = P.rest_point()
    near = np.hypot(base.V[:, 0] - rx, base.V[:, 1] - ry) < 0.0075
    top = base.V[near, 2].max()                                                        # the fork's prongs
    pad = base.V[np.hypot(base.V[:, 0] - rx, base.V[:, 1] - ry) < 0.0045, 2].max()
    under = rz - L.R_TUBE
    assert 0.0 < under - pad < 0.001                                                   # rests (a hair above the pad)
    lifted = under + L.REST_S * math.sin(math.radians(L.LIFT_DEG))
    assert lifted > top + 0.0005                                                       # the cue lifts it out of the fork


def test_the_cue_bar_waits_under_the_tube_and_rises_to_hold_it_up():
    p = P.tube_point(L.CUE_S)
    s = math.hypot(p[0], p[1])
    under_down = L.Z_ARM - L.R_TUBE
    under_up = under_down + s * math.sin(math.radians(L.LIFT_DEG))
    assert under_down - P.bar_rest_z() == pytest.approx(L.BAR_GAP)
    assert P.bar_rest_z() + L.BAR_RISE == pytest.approx(under_up, abs=2e-4)


def test_the_platter_and_the_record_stack_up(parts):
    plat, mat, rec, lab, ad = (parts[k].mesh for k in ("platter", "mat", "record", "label", "adapter"))
    top = L.PLATTER_TOP - L.Z_PLATE
    assert mat.V[:, 2].min() == pytest.approx(top) and mat.V[:, 2].max() == pytest.approx(top + L.MAT_T)
    assert rec.V[:, 2].min() == pytest.approx(L.Z_MAT - L.Z_PLATE)
    assert np.hypot(rec.V[:, 0], rec.V[:, 1]).max() == pytest.approx(L.REC_R, abs=1e-4)
    assert np.hypot(rec.V[:, 0], rec.V[:, 1]).min() == pytest.approx(L.HOLE_R, abs=1e-4)
    assert np.hypot(ad.V[:, 0], ad.V[:, 1]).max() < L.HOLE_R                           # the adapter fits the hole
    assert ad.V[:, 2].max() == pytest.approx(L.ADAPTER_TOP - L.Z_PLATE, abs=1e-4)
    assert lab.V[:, 2].min() > rec.V[:, 2].max() - 1e-9                                # the paper lies on the record
    assert np.hypot(plat.V[:, 0], plat.V[:, 1]).max() == pytest.approx(L.PLATTER_R, abs=1e-4)


def test_the_plinth_and_feet_stand_on_the_floor_and_fill_the_footprint(parts):
    V = np.concatenate([parts["plinth"].mesh.V, parts["feet"].mesh.V])
    lo, hi = V.min(0), V.max(0)
    assert lo[2] == pytest.approx(0.0) and hi[2] == pytest.approx(L.Z_PLATE)
    assert hi[0] - lo[0] == pytest.approx(L.W) and hi[1] - lo[1] == pytest.approx(L.D)


# ======================================================================================================= form and time
def test_form_is_far_from_boxy(parts):
    V, T, owner, names, off = [], [], [], [], 0
    for k, part in parts.items():
        v, t = world(part), part.mesh.triangles()
        V.append(v)
        T.append(t + off)
        owner.append(np.full(len(t), len(names)))
        names.append(k)
        off += len(v)
    r = F.analyse(np.concatenate(V), np.concatenate(T), np.concatenate(owner), names, exempt=P.GRAPHIC)
    assert r["score"] < 0.25, (r["score"], r["components"][:4])
    assert r["score"] < 0.05, "keep the deck well inside the hero limit"


def test_the_meshes_build_quickly():
    t0 = time.time()
    P.parts()
    assert time.time() - t0 < 2.0
