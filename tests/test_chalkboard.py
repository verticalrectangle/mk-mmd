"""The chalkboard prop (pure numpy: mkmmd.blender.library.props.chalkboard_layout): the card's board surface on the
slate's face (so `[[text]] on = "<name>:board"` writes on it, reading from the front), the tray's edge, the board standing
on the ledge and leaning on the legs, the noughts-and-crosses drawing on the slate, the option, closed outward meshes and
the form guard."""
import json
import math
import time

import numpy as np
import pytest

from mkmmd.blender.library.props import chalkboard_layout as L
from mkmmd.core import form as F
from mkmmd.core import palette as PAL
from mkmmd.core import shell as S

NAME = "cb"


@pytest.fixture(scope="module")
def parts():
    return L.parts("noughts")


def probe(mesh):
    return S.Probe(mesh.V, mesh.triangles())


def test_the_card_is_plain_json_with_its_board_tray_and_look():
    card = L.card(NAME, L.Pal().roles(), L.options({"drawing": "noughts"}))
    json.dumps(card)
    assert card["origin"] == "floor_center" and card["front"] == "-Y"
    assert {k: [e["name"] for e in v] for k, v in card["use"].items()} == {
        "surface": ["board"], "rest": ["tray"], "look": ["board"]}
    assert set(card["slots"]) == set(L.ROLES)
    assert card["objects"] == [f"{NAME}_{k}" for k in ("board", "easel", "chalk", "drawing")]
    assert L.card(NAME)["objects"] == [f"{NAME}_{k}" for k in ("board", "easel", "chalk")]


@pytest.mark.parametrize("slots, want", [({}, "none"), ({"drawing": "noughts"}, "noughts"), ({"drawing": "none"}, "none")])
def test_the_drawing_option(slots, want):
    assert L.options(slots) == {"drawing": want}
    assert ("drawing" in L.parts(want)) == (want == "noughts")


def test_a_bad_drawing_is_refused_saying_what_there_is():
    with pytest.raises(ValueError, match="chalkboard: drawing is 'none'.*'noughts'.*got 'hangman'"):
        L.options({"drawing": "hangman"})


def test_every_solid_part_is_closed_and_outward(parts):
    for key, part in parts.items():
        if key in L.GRAPHIC:
            continue
        assert part.mesh.is_closed(), key
        assert part.mesh.volume() > 0, key


def test_the_board_surface_is_the_slate_s_face_reading_from_the_front(parts):
    s = L.card(NAME)["use"]["surface"][0]
    c, n, up = (np.array(s[k]) for k in ("center", "normal", "up"))
    assert np.linalg.norm(n) == pytest.approx(1.0, abs=1e-4) and np.linalg.norm(up) == pytest.approx(1.0, abs=1e-4)
    assert n @ up == pytest.approx(0.0, abs=1e-4)
    assert np.cross(up, n) == pytest.approx([1.0, 0.0, 0.0], abs=1e-4)                 # the text's x: +X, read from -Y
    assert n[1] < 0 and math.degrees(math.acos(up[2])) == pytest.approx(L.LEAN, abs=0.01)   # leaning back 12 degrees
    assert s["size"] == [L.SLATE_W, L.SLATE_H] == [0.90, 0.65]
    board = probe(parts["board"].mesh)
    for a, b in ((0.0, 0.0), (0.4, 0.28), (-0.4, -0.28)):                               # the slate's face, to the corners
        p = c + np.array([a, 0.0, 0.0]) + b * up
        hit = board.ray(p + 0.1 * n, -n)
        assert hit is not None and hit[0] == pytest.approx(0.1, abs=2e-4), (a, b)
    look = L.card(NAME)["use"]["look"][0]["point"]
    assert look == pytest.approx(s["center"])


def test_the_board_stands_on_the_ledge_and_leans_on_the_legs(parts):
    V = parts["board"].mesh.V
    assert V[:, 2].min() == pytest.approx(L.TRAY_Z, abs=1e-3)
    back = (V - L.CENTRE) @ L.NORMAL
    assert back.min() == pytest.approx(-L.FRAME_BACK, abs=1e-4)
    for z in (0.9, 1.2, 1.45):                                                         # the frame's back, the legs' front
        y_back = (L.CENTRE + (z - L.CENTRE[2]) / L.UP[2] * L.UP - L.FRAME_BACK * L.NORMAL)
        assert 0.0 < L.legs_front_y(z) - y_back[1] + 1e-9 < 0.004


def test_the_easel_stands_on_the_floor_about_1_6_m_tall(parts):
    E = parts["easel"].mesh.V
    assert E[:, 2].min() == pytest.approx(0.0, abs=1e-9)
    V = np.concatenate([p.mesh.V for p in parts.values()])
    lo, hi = V.min(0), V.max(0)
    assert 1.5 < hi[2] < 1.75
    assert (lo[1] + hi[1]) / 2 == pytest.approx(0.0, abs=0.01) and (lo[0] + hi[0]) / 2 == pytest.approx(0.0, abs=1e-6)
    assert L.card(NAME)["size"] == pytest.approx(list(hi - lo), abs=0.01)


def test_the_chalk_and_the_eraser_lie_in_the_tray_and_the_tray_edge_is_its_lip(parts):
    C = parts["chalk"].mesh.V
    assert C[:, 2].min() == pytest.approx(L.TRAY_Z, abs=1e-4)
    board = parts["board"].mesh.V
    front = board[board[:, 2] < L.TRAY_Z + 0.01, 1].min()                               # the board's foot
    lip_in = L.TRAY_BACK_Y - L.TRAY_D + 0.008 + L.LIP[0]
    assert lip_in < C[:, 1].min() and C[:, 1].max() < front                            # in the trough
    edge = L.card(NAME)["use"]["rest"][0]
    a, b = np.array(edge["a"]), np.array(edge["b"])
    assert edge["type"] == "edge" and edge["normal"] == [0.0, 0.0, 1.0] and a[0] < 0 < b[0]
    easel = probe(parts["easel"].mesh)
    for p in (a, b, 0.5 * (a + b)):
        assert easel.height(p[0], p[1]) == pytest.approx(p[2], abs=1e-3)                # on the lip's top


def test_the_noughts_and_crosses_lie_on_the_slate_inside_the_surface(parts):
    s = L.card(NAME)["use"]["surface"][0]
    w, h = s["size"]
    strokes = L.drawing_strokes()
    assert len(strokes) == 4 + 2 * 2 + 2                                               # grid, two crosses, two noughts
    P = np.concatenate(strokes)
    assert np.abs(P[:, 0]).max() < w / 2 - 0.05 and np.abs(P[:, 1]).max() < h / 2 - 0.03
    D = parts["drawing"].mesh.V
    off = (D - L.CENTRE) @ L.NORMAL
    assert 0.0 < off.min() and off.max() < 0.002                                       # just over the slate's face


def test_roles_never_shadow_a_palette_slot_and_a_bad_colour_is_refused():
    assert not set(L.ROLES) & set(PAL.PALETTES["rose-pine-dawn"])
    pal = L.Pal({**PAL.get("rose-pine-dawn"), "slate": "pine"})
    assert pal.lin("slate") == pytest.approx(pal.lin("pine"))
    assert set(pal.checked()) == set(L.ROLES)
    with pytest.raises(ValueError, match="chalkboard: colour role chalk = 'chalky'"):
        L.Pal({"chalk": "chalky"}).checked()


def test_form_is_far_from_boxy(parts):
    V, T, owner, names, off = [], [], [], [], 0
    for k, part in parts.items():
        V.append(part.mesh.V)
        t = part.mesh.triangles()
        T.append(t + off)
        owner.append(np.full(len(t), len(names)))
        names.append(k)
        off += len(part.mesh.V)
    r = F.analyse(np.concatenate(V), np.concatenate(T), np.concatenate(owner), names, exempt=L.GRAPHIC)
    assert r["score"] < 0.05, (r["score"], r["components"][:4])


def test_the_meshes_build_quickly():
    t0 = time.time()
    L.parts("noughts")
    assert time.time() - t0 < 1.0
