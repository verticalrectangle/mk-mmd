"""The record treadmill prop (pure numpy: the Blender half is in tests/test_record_treadmill_build.py): the card's use
points against the meshes (the run spot and the rest plane on the record's real top, the label panel inside the print),
the clockwise spin and its driver, the starting values, the colour roles, the label's print and the form guard."""
import json
import math
import time

import numpy as np
import pytest

from mkmmd.blender.library.props import record_treadmill_layout as L
from mkmmd.core import form as F
from mkmmd.core import palette as PAL
from mkmmd.core import shell as S

NAME = "rec"


@pytest.fixture(scope="module")
def parts():
    return L.parts()


def world(key, parts):
    """A part's vertices in the prop's frame (the record and label ride the platter at the deck)."""
    p = parts[key]
    t = parts["platter"].origin if key in L.SPIN else p.origin
    return p.mesh.V @ S.rot_matrix(*p.rot).T + np.asarray(t)


@pytest.fixture(scope="module")
def record_top(parts):
    return S.Probe(world("record", parts), parts["record"].mesh.triangles())


def test_the_card_is_plain_json_with_its_uses_rates_and_starting_values():
    card = L.card(NAME, L.Pal().roles(), L.options({"rpm": 45}))
    json.dumps(card)
    assert card["origin"] == "floor_center" and card["front"] == "-Y"
    assert {k: [e["name"] for e in v] for k, v in card["use"].items()} == {
        "feet": ["run"], "rest": ["top"], "surface": ["label"], "look": ["centre"]}
    assert card["rates"] == [{"rate": "rpm", "into": "turns", "per": 60.0}]
    assert card["params"] == {"rpm": 45.0} and L.card(NAME)["params"] == {"rpm": 33.3}
    assert set(card["slots"]) == set(L.ROLES)
    assert card["objects"] == [f"{NAME}_{k}" for k in ("plinth", "feet", "platter", "record", "label")]


def test_the_contract_numbers():
    assert L.Z_TOP == pytest.approx(0.16) and L.REC_R == pytest.approx(0.80)
    assert 1.55 <= 2 * L.PLATTER_R <= 1.75                                              # a platter ~1.6 m across
    feet = L.card(NAME)["use"]["feet"][0]
    assert feet["L"] == [L.RUN[0] + 0.10, 0.0, None] and feet["R"] == [L.RUN[0] - 0.10, 0.0, None]
    assert L.RUN == (-0.45, 0.0)


def test_every_solid_part_is_closed_and_outward(parts):
    for key, part in parts.items():
        if key in L.GRAPHIC:
            continue
        assert part.mesh.is_closed(), key
        assert part.mesh.volume() > 0, key


def test_she_runs_on_the_record_s_real_top_at_the_card_s_height(record_top):
    card = L.card(NAME)
    feet, top = card["use"]["feet"][0], card["use"]["rest"][0]
    assert top["type"] == "plane" and top["normal"] == [0.0, 0.0, 1.0]
    assert top["center"][2] == feet["floor_z"]
    for x, y in (L.RUN, feet["L"][:2], feet["R"][:2]):
        for dy in (-0.15, 0.0, 0.10):                                                   # under the whole foot
            assert record_top.height(x, y + dy) == pytest.approx(feet["floor_z"], abs=1e-6), (x, y + dy)
    for r in (L.GROOVE_IN, 0.5, top["radius"] - 0.006):                                 # the plane: the flat top
        assert record_top.height(r, 0.0) == pytest.approx(top["center"][2], abs=1e-6), r
    assert record_top.height(L.REC_R - 0.006, 0.0) > L.Z_TOP                           # the bead at the rim
    assert record_top.height(0.1, 0.0) > L.Z_TOP                                        # the raised label area


def test_the_run_spot_and_the_feet_are_inside_the_grooved_band():
    feet = L.card(NAME)["use"]["feet"][0]
    for x, y in (L.RUN, feet["L"][:2], feet["R"][:2]):
        for dy in (-0.15, 0.10):
            assert L.GROOVE_IN < math.hypot(x, y + dy) < L.GROOVE_OUT


def test_the_record_turns_clockwise_and_carries_her_feet_backward():
    vx, vy = L.surface_speed(33.3)
    assert vx == pytest.approx(0.0, abs=1e-12) and vy > 1.0                             # toward +Y: behind her
    assert L.spin(0.25) == pytest.approx(-math.pi / 2)                                  # clockwise from above
    for t in (-1.3, 0.0, 0.37, 2.0):
        assert eval(L.EXPR["spin"], {}, {"turns": t}) == pytest.approx(L.spin(t), abs=1e-6)   # our own expression
    assert len(L.EXPR["spin"]) <= 255 and "**" not in L.EXPR["spin"] and "%" not in L.EXPR["spin"]


def test_the_label_panel_rides_the_platter_inside_the_print_clear_of_the_spindle(parts):
    s = L.card(NAME)["use"]["surface"][0]
    assert s["object"] == f"{NAME}_platter" and s["normal"] == [0.0, 0.0, 1.0] and s["up"] == [0.0, 1.0, 0.0]
    cx, cy, cz = s["center"]
    assert cz == pytest.approx(world("label", parts)[:, 2].max(), abs=1e-5)             # on the paper
    w, h = s["size"]
    corners = [(cx + sx * w / 2, cy + sy * h / 2) for sx in (-1, 1) for sy in (-1, 1)]
    assert max(math.hypot(*c) for c in corners) <= L.LABEL_R - L.LABEL_MARGIN + 1e-9    # inside the print
    assert cy - h / 2 >= L.SPINDLE_R + 0.02                                             # clear of the spindle
    lab = world("label", parts)
    assert np.hypot(lab[:, 0], lab[:, 1]).max() == pytest.approx(L.LABEL_R, abs=1e-6)
    assert lab[:, 2].min() > world("record", parts)[:, 2].max() - 1e-9                 # the paper on the record


def test_the_platter_the_record_and_the_spindle_stack_up(parts):
    plat, rec = world("platter", parts), world("record", parts)
    assert rec[:, 2].min() == pytest.approx(L.Z_REC0) and rec[:, 2].min() == pytest.approx(plat[np.hypot(
        plat[:, 0], plat[:, 1]) > 0.5, 2].max())                                       # the record lies on the platter
    assert np.hypot(rec[:, 0], rec[:, 1]).max() == pytest.approx(L.REC_R, abs=5e-4)    # the rim's bead is rounded
    assert np.hypot(rec[:, 0], rec[:, 1]).min() == pytest.approx(L.HOLE_R, abs=1e-4)
    assert np.hypot(plat[:, 0], plat[:, 1]).max() == pytest.approx(L.PLATTER_R, abs=1e-4)
    assert plat[:, 2].max() == pytest.approx(L.SPINDLE_TOP, abs=1e-6)                  # the spindle's tip
    assert L.SPINDLE_R < L.HOLE_R                                                       # through the hole


def test_the_plinth_and_feet_stand_on_the_floor_under_the_platter(parts):
    V = np.concatenate([world("plinth", parts), world("feet", parts)])
    lo, hi = V.min(0), V.max(0)
    assert lo[2] == pytest.approx(0.0) and hi[2] == pytest.approx(L.Z_PLINTH)
    assert hi[0] - lo[0] == pytest.approx(L.PLINTH_W, abs=1e-3) and hi[1] - lo[1] == pytest.approx(L.PLINTH_D, abs=1e-3)
    assert world("platter", parts)[:, 2].min() > L.Z_PLINTH                             # turns clear of the deck


@pytest.mark.parametrize("slots, frag", [({"rpm": "fast"}, "rpm must be a number"), ({"rpm": True}, "rpm must"),
                                         ({"rpm": float("nan")}, "rpm must"), ({"rpm": 500}, "rpm is -80")])
def test_a_bad_starting_value_is_refused_saying_what_it_is(slots, frag):
    with pytest.raises(ValueError, match=f"record_treadmill: {frag}"):
        L.options(slots)


def test_roles_take_slots_hexes_blends_and_other_roles_and_never_shadow_a_palette_slot():
    assert not set(L.ROLES) & set(PAL.PALETTES["rose-pine-dawn"])
    pal = L.Pal({**PAL.get("rose-pine-dawn"), "label": "pine", "vinyl": "#ff0000", "rays": "label:1,gold:1"})
    assert pal.lin("label") == pytest.approx(PAL.linear(PAL.get("rose-pine-dawn")["pine"]))
    assert pal.lin("vinyl") == pytest.approx(PAL.linear("#ff0000"))
    want = [(a + b) / 2 for a, b in zip(pal.lin("label"), pal.lin("gold"))]
    assert pal.lin("rays") == pytest.approx(want)
    assert set(pal.checked()) == set(L.ROLES)
    with pytest.raises(ValueError, match="record_treadmill: colour role print = 'chartreuse'"):
        L.Pal({"print": "chartreuse"}).checked()
    with pytest.raises(ValueError, match="loop"):
        L.Pal({"label": "rays", "rays": "label:2,paper:1"}).lin("label")


def test_the_label_print_is_its_roles_colours_with_the_beets_clear_of_the_text_panel():
    cols = {"label": (0.9, 0.6, 0.2), "rays": (1.0, 0.9, 0.8), "print": (0.7, 0.3, 0.4), "leaf": (0.2, 0.6, 0.6),
            "paper": (1.0, 1.0, 0.95)}
    n = 512
    img = L.label_image(cols, n)
    assert img.shape == (n, n, 3) and img.min() >= 0.0 and img.max() <= 1.0
    c = (np.arange(n) + 0.5) * 2.0 / n - 1.0
    x, y = np.meshgrid(c, -c)

    def at(fx, fy):
        return img[int((1.0 - fy) * n / 2), int((fx + 1.0) * n / 2)]

    assert at(0.16, 0.0) == pytest.approx(cols["paper"], abs=1e-3)                    # the cream centre
    assert at(0.0, -0.95) == pytest.approx(cols["print"], abs=1e-3)                   # the band at the edge
    leaf = np.all(np.abs(img - np.array(cols["leaf"])) < 1e-3, axis=-1)
    assert leaf.sum() > 300 and (y[leaf] < 0.0).all()                                   # the beets: the lower half
    p = L.LABEL_PANEL
    panel = (np.abs(x) <= p["w"] / 2 / L.LABEL_R) & (y >= p["y0"] / L.LABEL_R) & (y <= p["y1"] / L.LABEL_R)
    assert not (leaf & panel).any()


def test_form_is_far_from_boxy(parts):
    V, T, owner, names, off = [], [], [], [], 0
    for k in parts:
        v, t = world(k, parts), parts[k].mesh.triangles()
        V.append(v)
        T.append(t + off)
        owner.append(np.full(len(t), len(names)))
        names.append(k)
        off += len(v)
    r = F.analyse(np.concatenate(V), np.concatenate(T), np.concatenate(owner), names, exempt=L.GRAPHIC)
    assert r["score"] < 0.05, (r["score"], r["components"][:4])


def test_the_meshes_and_the_print_build_quickly():
    t0 = time.time()
    L.parts()
    L.label_image({k: (0.5, 0.5, 0.5) for k in ("label", "rays", "print", "leaf", "paper")}, 1024)
    assert time.time() - t0 < 2.0
