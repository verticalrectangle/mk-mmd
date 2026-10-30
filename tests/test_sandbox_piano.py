"""The sandbox piano prop (pure numpy: the layout and meshes of mkmmd.blender.library.props.sandbox_piano_layout and
_parts): the card's seat, feet, keys and sand against the meshes, the contract's sizes, the keyboard, the legs going into
the sand, the lid on its stick, closed outward meshes and the form guard."""
import itertools
import json
import math
import time

import numpy as np
import pytest

from mkmmd.blender.library.props import sandbox_piano_layout as L
from mkmmd.blender.library.props import sandbox_piano_parts as P
from mkmmd.core import form as F
from mkmmd.core import palette as PAL
from mkmmd.core import shell as S

NAME = "sb"


@pytest.fixture(scope="module")
def parts():
    return P.parts()


def world(part):
    return part.mesh.V @ S.rot_matrix(*part.rot).T + np.asarray(part.origin)


def probe(part):
    return S.Probe(world(part), part.mesh.triangles())


def test_the_card_is_plain_json_with_the_chair_s_uses():
    card = L.card(NAME, L.Pal().roles())
    json.dumps(card)
    assert card["origin"] == "floor_center" and card["front"] == "-Y"
    assert {k: [e["name"] for e in v] for k, v in card["use"].items()} == {
        "sit": ["bench"], "feet": ["bench"], "rest": ["keys", "sand"], "look": ["keys"]}
    sit = card["use"]["sit"][0]
    assert sit["facing"] == [0.0, -1.0, 0.0] and sit["floor_z"] == L.Z_SAND and sit["seat_z"] == pytest.approx(L.SEAT_Z)
    assert set(sit) >= {"hip", "seat_z", "facing", "floor_z", "pelvis_deg", "back_deg"}
    assert set(card["slots"]) == set(L.ROLES)
    assert card["objects"] == [f"{NAME}_{k}" for k in P.parts()]


def test_the_contract_sizes():
    assert (L.BOX_W, L.BOX_D) == (2.2, 2.4) and L.BOARD_TOP == pytest.approx(0.30)
    assert L.Z_SAND == pytest.approx(0.22)
    assert L.KEYS_Z - L.Z_SAND == pytest.approx(0.70) and L.SEAT_Z - L.Z_SAND == pytest.approx(0.45)
    size = L.card(NAME)["size"]
    assert size[:2] == [L.BOX_W, L.BOX_D]
    top = max(world(p)[:, 2].max() for p in P.parts().values())
    assert size[2] == pytest.approx(top, abs=0.01)                                      # the open lid's edge


def test_every_part_is_closed_and_outward(parts):
    for key, part in parts.items():
        assert part.mesh.is_closed(), key
        assert part.mesh.volume() > 0, key


def test_she_sits_on_the_bench_facing_the_keys_with_her_feet_on_level_sand(parts):
    card = L.card(NAME)
    sit, feet = card["use"]["sit"][0], card["use"]["feet"][0]
    hx, hy, hz = sit["hip"]
    top = probe(parts["bench"]).height(hx, hy)
    assert top == pytest.approx(L.SEAT_Z, abs=0.004)                                   # the cushion under her hips
    assert hz == pytest.approx(L.SEAT_Z + 0.075)
    assert hy - L.KEY_Y0 == pytest.approx(L.HIP_TO_KEYS)                               # the keys in front of her (-Y)
    sand = probe(parts["sand"])
    assert feet["floor_z"] == L.Z_SAND
    for s in ("L", "R"):
        x, y, z = feet[s]
        assert z is None and sand.height(x, y) == pytest.approx(L.Z_SAND, abs=1e-6)    # level where her feet go
        assert L.sand_height(x, y) == pytest.approx(L.Z_SAND, abs=1e-12)
    assert feet["L"][0] > 0 > feet["R"][0]                                             # her left is +X
    assert L.Z_BED0 - L.Z_SAND > 0.60                                                   # her knees go under the key bed


def test_the_keys_rest_edge_lies_on_the_white_keys_along_the_keyboard_front(parts):
    edge = L.card(NAME)["use"]["rest"][0]
    assert edge["type"] == "edge" and edge["normal"] == [0.0, 0.0, 1.0]
    a, b = np.array(edge["a"]), np.array(edge["b"])
    assert a[2] == b[2] == pytest.approx(L.KEYS_Z) and a[1] == b[1] and a[0] < 0 < b[0]
    assert L.KEY_Y0 - L.KEY_L < a[1] < L.KEY_Y0
    keys = probe(parts["keys"])
    for i in range(L.WHITE):
        x = L.white_x(i)
        if abs(x) <= abs(a[0]) and all(abs(x - bx) > L.BLACK_W for _, bx in L.black_keys()):
            assert keys.height(x, a[1]) == pytest.approx(L.KEYS_Z, abs=1e-6), i
    look = L.card(NAME)["use"]["look"][0]["point"]
    assert look[2] == pytest.approx(L.KEYS_Z) and L.KEY_Y0 - L.KEY_L < look[1] < L.KEY_Y0


def test_the_keyboard_is_two_octaves_c_to_c_lowest_at_her_left():
    assert L.WHITE == 15 and len(L.black_keys()) == 10
    xs = [L.white_x(i) for i in range(L.WHITE)]
    assert xs[0] > xs[-1] and xs[0] == pytest.approx(L.KEYS_W / 2 - L.KEY_PITCH / 2)       # bass at +X
    after = [i for i, _ in L.black_keys()]
    assert after == [0, 1, 3, 4, 5, 7, 8, 10, 11, 12]                                  # groups of two and three
    for i, x in L.black_keys():
        assert x == pytest.approx(0.5 * (xs[i] + xs[i + 1]))
    assert L.CASE_W > L.KEYS_W + 2 * L.CHEEK_W - 1e-9


def test_the_sand_fills_the_box_at_its_level_with_gentle_bumps(parts):
    V = world(parts["sand"])
    assert np.abs(V[:, 0]).max() <= L.INNER_W / 2 + L.SAND_TUCK + 1e-9
    assert np.abs(V[:, 1]).max() <= L.INNER_D / 2 + L.SAND_TUCK + 1e-9
    xs, ys = np.meshgrid(np.linspace(-1.0, 1.0, 41), np.linspace(-1.1, 1.1, 45))
    h = L.sand_height(xs, ys)
    assert abs(float(np.median(h)) - L.Z_SAND) < 0.01 and 0.03 < h.max() - h.min() < 0.12
    rest = L.card(NAME)["use"]["rest"][1]
    assert rest["type"] == "plane" and rest["center"][2] == L.Z_SAND and rest["size"] == [L.INNER_W, L.INNER_D]


def test_the_boards_and_the_corner_seats(parts):
    B, Sx = world(parts["boards"]), world(parts["seats"])
    assert B[:, 2].min() == pytest.approx(0.0, abs=2e-3) and B[:, 2].max() == pytest.approx(L.BOARD_TOP, abs=2e-3)
    assert B[:, 0].max() == pytest.approx(L.BOX_W / 2) and B[:, 1].max() == pytest.approx(L.BOX_D / 2)
    assert Sx[:, 2].min() == pytest.approx(L.BOARD_TOP) and Sx[:, 2].max() == pytest.approx(L.BOARD_TOP + L.SEAT_T)


def test_the_legs_go_into_the_sand_and_up_into_the_case(parts):
    V = world(parts["trim"])
    for x, b in L.LEGS:
        lx, ly = x * L.CASE_W, L.CASE_FRONT - b
        leg = V[np.hypot(V[:, 0] - lx, V[:, 1] - ly) < L.LEG_R + 1e-6]
        assert leg[:, 2].min() == pytest.approx(L.Z_SAND - L.SINK)
        assert leg[:, 2].min() < L.sand_height(lx, ly) - 0.03                            # sunk into the heaped sand
        assert leg[:, 2].max() > L.RIM_Z0                                               # into the case
    Bv = world(parts["bench"])
    assert Bv[:, 2].min() == pytest.approx(L.Z_SAND - L.SINK)


def test_the_lid_stands_open_on_its_stick(parts):
    foot, head = P.stick_ends()
    assert probe(parts["case"]).height(foot[0], foot[1]) == pytest.approx(L.RIM_Z1, abs=1e-3)   # on the rim's top
    axis = (head - foot) / np.linalg.norm(head - foot)
    hit = probe(parts["lid"]).ray(foot + 0.05 * axis, axis)
    assert hit is not None and hit[1] == pytest.approx(head, abs=2e-3)                 # up to the lid's underside
    assert math.degrees(math.acos(axis[2])) < 30.0 and axis[0] > 0                     # leaning in toward the spine
    stick = world(parts["trim"])
    d = stick - foot
    along = d @ axis
    on = np.linalg.norm(d - along[:, None] * axis, axis=1) < 0.021
    assert along[on].max() == pytest.approx(np.linalg.norm(head - foot) + 0.004, abs=1e-6)
    assert np.allclose(L.lid_matrix(), S.rot_matrix(0.0, L.LID_DEG, 0.0))
    lid = world(parts["lid"])
    assert lid[:, 0].max() == pytest.approx(L.HINGE[0], abs=0.03)                       # hinged along the spine
    edge = lid[lid[:, 0] < lid[:, 0].min() + 0.01]
    assert edge[:, 2].min() > L.RIM_Z1 + 0.4                                            # the treble side up
    d = L.DESK
    for x in (-0.03 - d["w"] / 2, -0.03 + d["w"] / 2):
        assert L.RIM_Z1 + 0.01 + d["h"] < L.lid_under_z(x)                              # the desk clears the lid


def test_roles_never_shadow_a_palette_slot_and_a_bad_colour_is_refused():
    assert not set(L.ROLES) & set(PAL.PALETTES["rose-pine-dawn"])
    pal = L.Pal({**PAL.get("rose-pine-dawn"), "piano": "pine"})
    assert pal.lin("lid") == pytest.approx(pal.lin("pine"))                            # the lid follows the piano
    assert set(pal.checked()) == set(L.ROLES)
    with pytest.raises(ValueError, match="sandbox_piano: colour role bench = 'plaid'"):
        L.Pal({"bench": "plaid"}).checked()


def test_form_is_far_from_boxy(parts):
    V, T, owner, names, off = [], [], [], [], 0
    for k, part in parts.items():
        v, t = world(part), part.mesh.triangles()
        V.append(v)
        T.append(t + off)
        owner.append(np.full(len(t), len(names)))
        names.append(k)
        off += len(v)
    r = F.analyse(np.concatenate(V), np.concatenate(T), np.concatenate(owner), names)
    assert r["score"] < 0.05, (r["score"], r["components"][:4])


def test_the_meshes_build_quickly():
    t0 = time.time()
    P.parts()
    assert time.time() - t0 < 1.5


def test_the_plan_is_a_grand_s():
    Q = L.plan()
    assert S.polygon_area(Q) > 0 and not S.section_crosses_itself(Q)
    assert Q[:, 0].max() == pytest.approx(L.CASE_W / 2, abs=1e-3) and Q[:, 1].max() == pytest.approx(L.CASE_FRONT)
    assert L.CASE_FRONT - Q[:, 1].min() == pytest.approx(L.CASE_L, abs=0.01)
    spine = Q[Q[:, 0] > L.CASE_W / 2 - 1e-3]
    assert spine[:, 1].min() < L.CASE_FRONT - 0.8                                       # a long straight spine at +X
    xs = [L.plan_back(x, Q) for x in np.linspace(-0.45, 0.30, 16)]
    assert all(b < a for a, b in itertools.pairwise(xs))                                # deeper toward the tail
    (x1, y1), (x2, y2), (x3, y3) = ((P._bentside_x(L.CASE_FRONT - b), L.CASE_FRONT - b) for b in (0.30, 0.64, 0.95))
    assert x2 > x1 + (x3 - x1) * (y2 - y1) / (y3 - y1) + 0.01                           # the bentside bends in (an S)
