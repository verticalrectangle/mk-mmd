"""The semi truck's layout, options and card (pure Python and numpy; the Blender side is checked by building and looking):
the numbers the parts agree on, the steering wheel's frame and reach, the options a project gives in `slots` (the trailer,
the beams, the murals), the card's surfaces and mural fits, the colour roles in every palette, the card itself with its
wheels, steering, lights, use points and colliders, and the whole rig's wheels standing on the ground on their hubs. Each
part module has its own tests (tests/test_semi_truck_<part>.py)."""
import ast
import inspect
import math
from pathlib import Path

import numpy as np
import pytest

from mkmmd.blender.library.props import convertible_palette as CP
from mkmmd.blender.library.props import semi_truck_card as C
from mkmmd.blender.library.props import semi_truck_layout as L
from mkmmd.core import palette as PAL
from mkmmd.core import shell as S

PROPS = Path(L.__file__).parent
PART_MODULES = ("semi_truck_gear", "semi_truck_body", "semi_truck_exterior", "semi_truck_cab", "semi_truck_trailer")
IN = 0.0254


# ======================================================================================== the layout
def test_the_tractor_has_the_proportions_of_the_long_hood_drawing():
    assert L.BOC_Y - L.Y_BUMPER == pytest.approx(130.9 * IN, abs=0.003)            # bumper to back of cab
    assert L.AXLE_S_Y - L.Y_BUMPER == pytest.approx(31 * IN, abs=0.003)            # bumper to the set-forward axle
    assert L.BOC_Y - L.AXLE_S_Y == pytest.approx(99.9 * IN, abs=0.003)
    assert L.AXLE_S_Y - L.FRAME_Y[0] == pytest.approx(19.6 * IN, abs=0.003)        # front of frame
    assert L.CAB_ROOF_Z - L.FRAME_Z[0] == pytest.approx(81.8 * IN, abs=0.003)      # frame bottom to roof
    assert 2 * L.FENDER_HALF_W == pytest.approx(94.1 * IN, abs=0.01)
    assert L.WHEELBASE == pytest.approx(-L.AXLE_S_Y) and L.AXLE_D_Y[1] - L.AXLE_D_Y[0] == pytest.approx(52 * IN, abs=0.003)
    assert L.CAB_BOTTOM_Z < L.CAB_FLOOR_Z < L.WS_BASE[1] < L.WS_TOP[1] < L.CAB_ROOF_Z < L.SLEEPER_ROOF_Z
    assert L.HOOD_Z[0] < L.HOOD_Z[1] < L.WS_BASE[1]                               # the hood rises to the cowl, under the glass
    assert L.GRILLE_Z[1] == pytest.approx(L.HOOD_Z[0], abs=0.01) and L.GRILLE_Y > L.Y_BUMPER
    assert L.COWL_Y < L.WS_BASE[0] < L.WS_TOP[0] < L.BOC_Y and 20.0 < L.WS_RAKE_DEG < 28.0
    assert L.CAB_FRONT_Y < L.DOOR_Y[0] < L.DOOR_GLASS[0][0] < L.DOOR_GLASS[1][0] < L.DOOR_Y[1] < L.BOC_Y
    assert L.SLEEPER_Y[0] - L.BOC_Y == pytest.approx(1.8 * IN, abs=0.001) and L.SLEEPER_Y[1] - L.SLEEPER_Y[0] == pytest.approx(63 * IN, abs=0.003)


def test_every_wheel_stands_on_the_ground_inside_the_legal_width():
    assert L.WHEEL_Z == L.TYRE_R                                                 # the tyre radius sits on the ground
    assert 2 * L.TYRE_R == pytest.approx(1.054, abs=0.002) and L.RIM_R == pytest.approx(22.5 * IN / 2, abs=1e-4)
    hubs = L.wheel_hubs()
    assert len(hubs) == 10 and set(hubs) == set(C.wheel_keys(True))
    for key, (x, y, z) in hubs.items():
        assert z == L.TYRE_R
        xs = L.tyre_centres(key)
        assert len(xs) == (1 if key.startswith("wheel_S") else 2)
        assert np.mean(xs) == pytest.approx(x)                                    # a dual's origin is midway between its tyres
        if len(xs) == 2:
            assert abs(xs[1] - xs[0]) == pytest.approx(L.DUAL_SPACING)
        assert max(abs(v) for v in xs) + L.TYRE_W / 2 < 102 * IN / 2 + 1e-6       # 102 in overall
        mirror = key[:-1] + ("R" if key.endswith("L") else "L")
        assert hubs[mirror] == pytest.approx((-x, y, z))
    # the duals clear each other and the frame; the steer tyres clear the frame too
    inner = min(abs(v) for k in hubs for v in L.tyre_centres(k)) - L.TYRE_W / 2
    assert inner > L.FRAME_X + 0.1


def test_the_coupling_and_the_van_fit_together():
    assert L.TRAILER_Y[1] - L.TRAILER_Y[0] == pytest.approx(53 * 12 * IN, abs=0.002)
    assert 2 * L.TRAILER_HALF_W == pytest.approx(102 * IN, abs=0.002) and L.TRAILER_TOP_Z == pytest.approx(13.5 * 12 * IN, abs=0.002)
    assert L.KINGPIN_Y - L.TRAILER_Y[0] == pytest.approx(36 * IN, abs=0.001)
    assert L.AXLE_D_Y[0] < L.KINGPIN_Y < L.AXLE_D_Y[1]                           # the fifth wheel sits over the tandem
    assert L.FIFTH_TOP_Z > 2 * L.TYRE_R + 0.1 > L.FRAME_Z[1]                      # the van's floor clears the drive tyres
    assert L.TRAILER_Y[0] - L.SLEEPER_Y[1] > 0.8                                 # room for the van to swing behind the sleeper
    assert L.LANDING_Y > L.FRAME_Y[1] and L.LANDING_Y > L.AXLE_D_Y[1] + L.TYRE_R + 0.3   # the legs stand behind the tractor
    assert L.TRAILER_AXLE_Y[1] - L.TRAILER_AXLE_Y[0] == pytest.approx(49 * IN, abs=0.002)
    assert L.TRAILER_AXLE_Y[1] - L.KINGPIN_Y < 40 * 12 * IN                      # kingpin to rear axle within 40 ft
    assert L.SLIDER_Y[0] < L.TRAILER_AXLE_Y[0] < L.TRAILER_AXLE_Y[1] < L.SLIDER_Y[1] < L.TRAILER_Y[1]
    assert L.SLIDER_X + 0.05 < L.TRAILER_TRACK / 2 - L.DUAL_SPACING / 2 - L.TYRE_W / 2    # the rails between the inner tyres
    assert L.WHEEL_Z < L.SLIDER_Z[0] < L.SLIDER_Z[1] <= L.TRAILER_FLOOR_Z - 0.1          # under the crossmembers, over the axles
    y0, y1, z0, z1 = L.SIDE_PANEL
    assert L.TRAILER_Y[0] < y0 < y1 < L.TRAILER_Y[1] and L.TRAILER_RAIL_Z[1] == z0 < z1 == L.TRAILER_TOP_RAIL_Z[0]
    hw, bz0, bz1 = L.BACK_PANEL
    assert hw < L.TRAILER_HALF_W and L.TRAILER_FLOOR_Z < bz0 < bz1 < L.TRAILER_TOP_RAIL_Z[0]


def test_the_steering_wheel_frame_and_the_drivers_reach():
    R = S.rot_matrix(*L.WHEEL_ROT)
    assert R @ [0.0, 0.0, 1.0] == pytest.approx(L.WHEEL_AXIS, abs=1e-12)          # +Z: the column, toward the driver
    up = R @ [0.0, 1.0, 0.0]
    assert up[2] > 0.5 and up[1] < 0.0                                            # +Y: 12 o'clock, up and forward
    assert R @ [1.0, 0.0, 0.0] == pytest.approx((-1.0, 0.0, 0.0), abs=1e-12)      # +X: 3 o'clock, the driver's right
    assert math.degrees(math.acos(L.WHEEL_AXIS[2])) == pytest.approx(L.WHEEL_TILT_DEG)
    c, hip = np.array(L.WHEEL_C), np.array(L.HIP)
    # the hands at 10 and 2 o'clock within half a metre of a small driver's shoulders (an MMD girl's ~0.42 m torso)
    shoulder = hip + (0.0, 0.07, 0.41)
    for clock in (10, 2):
        a = math.radians(90.0 - 30.0 * clock)                                    # 12 o'clock = +90 degrees in the wheel plane
        grip = c + R @ (L.WHEEL_R * math.cos(a), L.WHEEL_R * math.sin(a), 0.0)
        side = np.sign(grip[0] - c[0])
        assert np.linalg.norm(grip - (shoulder + (0.17 * side, 0.0, 0.0))) < 0.52
    # the lower rim clears the thighs (rising 14 degrees from the hip, 7 cm thick) by 6 cm
    bottom = c - R @ (0.0, L.WHEEL_R + L.WHEEL_TUBE, 0.0)
    ahead = hip[1] - bottom[1]
    assert bottom[2] - (hip[2] + ahead * math.tan(math.radians(14.0)) + 0.07) > 0.06
    assert L.CUSHION_Y[0] < hip[1] < L.CUSHION_Y[1] and hip[2] > L.CUSHION_TOP
    assert L.WHEEL_C[1] > L.DASH_Y[1]                                           # the wheel stands behind the dash's face


# ======================================================================================== options
def test_options_default_to_the_whole_rig_with_beams_and_no_mural():
    o = L.options({})
    assert o["trailer"] is True and o["beams"] is True and o["murals"] == {} and o["mural_fit"] == "cover"
    assert o["beam_power"] == L.NUM_OPTIONS["beam_power"]


def test_a_mural_paints_both_sides_and_each_panel_may_have_its_own():
    assert L.options({"mural": "a.png"})["murals"] == {"left": "a.png", "right": "a.png"}
    o = L.options({"mural": "a.png", "mural_right": "b.png", "mural_back": "c.png", "mural_fit": "fill"})
    assert o["murals"] == {"left": "a.png", "right": "b.png", "back": "c.png"} and o["mural_fit"] == "fill"
    assert L.options({"mural_left": "x"})["murals"] == {"left": "x"}


@pytest.mark.parametrize("slots, err", [({"mural_fit": "stretch"}, ValueError), ({"trailer": "no"}, TypeError),
                                        ({"beams": 1}, TypeError), ({"trailer": False, "mural": "a.png"}, ValueError)])
def test_bad_options_are_refused(slots, err):
    with pytest.raises(err):
        L.options(slots)


# ======================================================================================== surfaces and murals
def test_the_surfaces_lie_on_the_panels_a_mural_fills():
    by = {s["name"]: s for s in L.surfaces(True)}
    assert set(by) == {"door_L", "door_R", "sleeper_L", "sleeper_R", "left", "right", "back"}
    assert {s["name"] for s in L.surfaces(False)} == {"door_L", "door_R", "sleeper_L", "sleeper_R"}
    y0, y1, z0, z1 = L.SIDE_PANEL
    for name, sx in (("left", 1.0), ("right", -1.0)):
        s = by[name]
        assert s["normal"] == [sx, 0.0, 0.0] and s["up"] == [0.0, 0.0, 1.0]
        assert s["center"] == pytest.approx([sx * (L.TRAILER_HALF_W + L.SURFACE_LIFT), (y0 + y1) / 2, (z0 + z1) / 2], abs=1e-4)
        assert s["size"] == pytest.approx([y1 - y0, z1 - z0], abs=1e-3)
    hw, bz0, bz1 = L.BACK_PANEL
    assert by["back"]["normal"] == [0.0, 1.0, 0.0] and by["back"]["size"] == pytest.approx([2 * hw, bz1 - bz0], abs=1e-3)
    for side, sx in (("L", 1.0), ("R", -1.0)):
        d = by[f"door_{side}"]
        assert d["center"][0] == pytest.approx(sx * (L.CAB_HALF_W + L.SURFACE_LIFT))
        assert L.DOOR_Y[0] < d["center"][1] - d["size"][0] / 2 and d["center"][1] + d["size"][0] / 2 < L.DOOR_Y[1]
        assert d["center"][2] + d["size"][1] / 2 < L.DOOR_GLASS[0][1]             # below the window
        sl = by[f"sleeper_{side}"]
        assert L.SLEEPER_Y[0] < sl["center"][1] - sl["size"][0] / 2 and sl["center"][1] + sl["size"][0] / 2 < L.SLEEPER_Y[1]


@pytest.mark.parametrize("panel, image", [(6.0, 2.0), (6.0, 9.0), (0.5, 2.0), (2.0, 2.0)])
def test_mural_fits_cover_contain_and_fill(panel, image):
    su, sv = L.mural_scale("fill", panel, image)
    assert (su, sv) == (1.0, 1.0)
    su, sv = L.mural_scale("cover", panel, image)
    assert max(su, sv) == pytest.approx(1.0) and min(su, sv) <= 1.0 + 1e-12       # one axis spans the image, the other crops it
    assert (su / sv) == pytest.approx(panel / image)                             # uniform: the image keeps its aspect
    su, sv = L.mural_scale("contain", panel, image)
    assert min(su, sv) == pytest.approx(1.0) and (su / sv) == pytest.approx(panel / image)   # all of it shows


def test_the_mural_reads_the_right_way_round_on_both_sides():
    """A viewer in front of a surface sees `right = up x normal` to the right: the rear on `left` (+X), the front on
    `right`: the mural shader's u (semi_truck_mats.mural) must run that way."""
    by = {s["name"]: s for s in L.surfaces(True)}
    for name, along in (("left", 1.0), ("right", -1.0), ("back", None)):
        s = by[name]
        right = np.cross(s["up"], s["normal"])
        if along is not None:
            assert right == pytest.approx((0.0, along, 0.0))
        else:
            assert right == pytest.approx((-1.0, 0.0, 0.0))
    src = (PROPS / "semi_truck_mats.py").read_text()
    assert 'if side == "left"' in src and "hw - px, pos[0]" in src


# ======================================================================================== colours and materials
@pytest.mark.parametrize("palette", sorted(PAL.PALETTES))
def test_every_colour_role_resolves_in_every_palette(palette):
    pal = CP.Pal(PAL.get(palette), roles=L.ROLES)
    for role in L.ROLES:
        rgb = pal.lin(role)
        assert len(rgb) == 3 and all(0.0 <= v <= 1.0 for v in rgb)
    assert pal.lin("gold:base:0.5") == pytest.approx(PAL.linear(PAL.mix(PAL.get(palette)["gold"], PAL.get(palette)["base"], 0.5)))
    assert pal.roles() == {k: str(v) for k, v in L.ROLES.items()}


def test_every_material_role_has_its_material_and_the_card_roles_are_slot_specs():
    assert len(set(L.MATS)) == len(L.MATS) and L.M["mural_left"] != L.M["mural_right"]
    tree = ast.parse((PROPS / "semi_truck_mats.py").read_text())
    methods = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    for role in L.MATS:
        assert role.startswith("mural_") or f"m_{role}" in methods, role
    assert "mural" in methods
    for spec in L.ROLES.values():
        assert "#" not in spec                                                 # palette slots only: the palette recolours it


# ======================================================================================== the card
def _card(**slots):
    pal = CP.Pal(PAL.get("rose-pine-moon"), roles=L.ROLES)
    mods = {"m": type("M", (), {"LOOKS": {"cb": (0.0, -4.8, 1.6)}, "ANCHORS": {"cb_mic": {"point": (0.1, -4.8, 1.7), "dir": (0, 0, -1)}}})}
    return C.card("truck", pal.roles(), L.options(slots), mods, np.array([-1.3, -7.2, 0.0]), np.array([1.3, 14.94, 4.115]))


def test_the_card_lists_every_wheel_the_steering_and_the_lights():
    c = _card()
    assert [w["object"] for w in c["wheels"]] == [f"truck_{k}" for k in C.wheel_keys(True)]
    assert all(w["radius"] == L.TYRE_R and w["axis"] == [1.0, 0.0, 0.0] for w in c["wheels"])
    assert c["steering"] == {"object": "truck_wheel", "axis": [0.0, 0.0, 1.0], "ratio": L.STEER_RATIO, "wheelbase": L.WHEELBASE}
    assert c["lights"] == ["truck_beam_L", "truck_beam_R", "truck_dome", "truck_dash_glow"]
    assert c["trailer"] == {"object": "truck_trailer", "pivot": [0.0, L.KINGPIN_Y, L.FIFTH_TOP_Z]}
    assert c["front"] == "-Y" and c["origin"] == "floor_center" and c["size"] == pytest.approx([2.6, 22.14, 4.115])
    bob = _card(trailer=False, beams=False)
    assert len(bob["wheels"]) == 6 and "trailer" not in bob and bob["lights"] == ["truck_dome", "truck_dash_glow"]
    assert {s["name"] for s in bob["use"]["surface"]} == {"door_L", "door_R", "sleeper_L", "sleeper_R"}


def test_the_card_seats_the_driver_at_the_wheel():
    c = _card()
    sit = {s["name"]: s for s in c["use"]["sit"]}
    assert sit["driver"]["hip"][0] == L.SEAT_X > 0 and sit["passenger"]["hip"][0] == -L.SEAT_X     # left-hand drive
    assert sit["driver"]["facing"] == [0, -1, 0] and sit["driver"]["seat_z"] == L.CUSHION_TOP
    assert sit["driver"]["floor_z"] == L.CAB_FLOOR_Z and sit["driver"]["back_deg"] == L.BACK_RECLINE_DEG
    g = c["use"]["grip"][0]
    assert g["name"] == "wheel" and g["type"] == "ring" and g["object"] == "truck_wheel" == c["steering"]["object"]
    assert g["center"] == pytest.approx(L.WHEEL_C) and g["axis"] == pytest.approx(L.WHEEL_AXIS, abs=1e-5)
    assert g["radius"] == L.WHEEL_R and g["tube"] == L.WHEEL_TUBE
    feet = {f["name"]: f for f in c["use"]["feet"]}
    assert feet["driver"]["L"][0] > feet["driver"]["R"][0] and feet["passenger"]["L"][0] > feet["passenger"]["R"][0]
    assert feet["passenger"]["L"][0] == pytest.approx(-feet["driver"]["R"][0])
    rest = {r["name"]: r for r in c["use"]["rest"]}
    assert rest["sill_L"]["a"][0] == pytest.approx(L.CAB_HALF_W - L.CAB_WALL_T / 2) and rest["sill_L"]["a"][2] < L.DOOR_GLASS[0][1]
    looks = {k["name"]: k["point"] for k in c["use"]["look"]}
    assert {"road", "dash", "grille", "mirror_L", "fifth_wheel", "cb", "tail", "wheel_S_L", "trailer_wheel_T2_R"} <= set(looks)
    assert {a["name"] for a in c["use"]["anchor"]} == {"cb_mic"}
    slots = c["slots"]
    assert slots["body"] == L.ROLES["body"] and slots["trailer"] is True and slots["mural_fit"] == "cover"
    assert _card(mural="a.png")["slots"]["mural_left"] == "a.png"


def test_the_colliders_sit_where_the_driver_does():
    boxes = {b[0]: b for b in C.collider_boxes()}
    _, centre, size, _ = boxes["cushion_driver"]
    assert centre[0] == L.SEAT_X and centre[2] + size[2] / 2 == pytest.approx(L.CUSHION_TOP)
    assert centre[1] - size[1] / 2 <= L.HIP[1] <= centre[1] + size[1] / 2
    _, centre, size, _ = boxes["floor"]
    assert centre[2] + size[2] / 2 == pytest.approx(L.CAB_FLOOR_Z)
    _, centre, size, _ = boxes["dash"]
    knee_top = L.HIP[2] + 0.12
    assert centre[2] - size[2] / 2 <= knee_top <= L.DASH_TOP_Z                    # the dash over the knees, under its top
    c = _card()
    names = [col["object"] for col in c["colliders"]]
    assert names[:-1] == [f"truck_col_{b}" for b in boxes] and c["colliders"][-1]["type"] == "ring"


# ======================================================================================== the part modules
def test_the_part_modules_are_bpy_free_and_offer_the_builders_interface():
    import importlib
    builder = (PROPS / "semi_truck.py").read_text()
    assert "PART_MODULES = (" + ", ".join(f'"{m}"' for m in PART_MODULES) + ")" in builder
    for m in PART_MODULES:
        src = (PROPS / f"{m}.py").read_text()
        assert "import bpy" not in src, m
        mod = importlib.import_module(f"mkmmd.blender.library.props.{m}")
        assert any(hasattr(mod, f) for f in ("static_shells", "static_parts", "dynamic_parts")), m
        if hasattr(mod, "static_parts"):
            params = inspect.signature(mod.static_parts).parameters
            assert set(params) <= {"probes"}, m
