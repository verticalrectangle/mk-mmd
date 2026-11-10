"""The offset guitar prop (pure numpy: the Blender side is checked by building and looking): what its card adds to the
electric guitar's (the strum point between its two pickups, its anchors, the tremolo arm's tip for a hand), the meshes
against the layout (the photograph's landmarks, the strings over the saddles and into the tremolo), the arm's frame and
options, and the form guard."""
import json

import numpy as np
import pytest

from mkmmd.blender.library.props import electric_guitar_geo as G
from mkmmd.blender.library.props import electric_guitar_layout as EL
from mkmmd.blender.library.props import offset_guitar_body as B
from mkmmd.blender.library.props import offset_guitar_hw as H
from mkmmd.blender.library.props import offset_guitar_layout as L
from mkmmd.blender.library.props import offset_guitar_parts as P
from mkmmd.core import form as F
from mkmmd.core import shell as S

NAME = "guitar"


@pytest.fixture(scope="module")
def parts():
    return P.parts()


@pytest.fixture(scope="module")
def card(parts):
    lo, hi = P.bounds()
    return L.card(NAME, size=hi - lo)


def entry(card, kind, name):
    return next(e for e in card["use"][kind] if e["name"] == name)


def vec(v):
    return np.asarray(v, float)


def test_card_is_the_electric_guitars_with_this_bodys_points(card, parts):
    json.dumps(card)
    theirs = EL.card(NAME)
    assert entry(card, "grip", "neck") == entry(theirs, "grip", "neck")                    # the same neck: the same contract
    strum = entry(card, "grip", "strum")
    c = vec(strum["center"])
    covers = sorted(s["z"] for s in L.PICKUPS.values())
    assert covers[0] + 0.5 * L.PICKUP_W < strum["zone"][0] < c[2] < strum["zone"][1] < covers[1] - 0.5 * L.PICKUP_W
    assert c[1] == pytest.approx(np.mean([L.string_at(i, c[2])[1] for i in range(6)]), abs=1e-5) and c[0] == 0.0
    pick = parts["pick"].V                                                                 # in its own frame: +x to the tip
    tip = strum["pick"]["tip"]
    assert pick[:, 0].max() == pytest.approx(tip, abs=1e-4) and pick[:, 0].min() == pytest.approx(tip - strum["pick"]["length"], abs=1e-4)
    assert strum["hold"] == "across" and tip > 0.5 * strum["pick"]["length"] and 0.006 < tip - strum["clear"] < 0.012
    assert np.ptp(pick[:, 1]) == pytest.approx(strum["pick"]["width"], abs=2e-4) and strum["pick"]["width"] <= 0.02   # a small pick
    o, n = B.outline(), G.edge_normals(B.outline())
    for name in ("strap_top", "strap_bottom"):                                             # on the body's edge, pointing out
        a = entry(card, "anchor", name)
        p, d = vec(a["point"]), vec(a["dir"])
        assert G.distance(np.array([[p[0], p[2]]]), o)[0] < 0.0008 and p[1] == pytest.approx(0.5 * L.BODY_T)
        i = int(np.argmin(np.linalg.norm(o - np.array([p[0], p[2]]), axis=1)))
        assert d[[0, 2]] @ n[i] > 0.9 * np.linalg.norm(d[[0, 2]])                             # out of the edge, seen from the front
    jack = entry(card, "anchor", "jack")
    along = (parts["plug"].V - vec(jack["point"])) @ vec(jack["dir"])
    assert along.max() == pytest.approx(0.0, abs=1e-4) and vec(jack["dir"])[1] < -0.5      # the plug's end, out of the face
    wear = card["use"]["wear"][0]
    assert wear["strap"]["material"] == f"{NAME}_strap" and wear["cable"]["object"] == f"{NAME}_cable"
    assert {c["object"] for c in card["colliders"]} == {f"{NAME}_col_{k}" for k in L.COLLIDERS}


def test_the_arms_tip_is_an_anchor_a_hand_can_take(card):
    bend, tip, R = L.arm_geometry()
    a = entry(card, "anchor", "trem")
    assert a["bone"] == "wrist.R" and a["object"] == f"{NAME}_trem_aim" and np.allclose(a["point"], tip, atol=1e-5)
    assert np.allclose(R.T @ R, np.eye(3), atol=1e-12) and np.linalg.det(R) == pytest.approx(1.0)
    assert np.allclose(bend + R[:, 2] * L.ARM["length"], tip)                               # +Z runs from the bend to the tip
    assert R[1, 1] < -0.9                                                                   # +Y out of the face
    arm = P.arm()
    world = arm.V @ R.T + bend
    assert arm.is_closed() and world[:, 2].max() == pytest.approx(tip[2], abs=0.002)
    assert world[:, 1].max() < -L.TREM["t"] - 0.004                                         # the arm clear of the plate
    c = vec(entry(card, "grip", "strum")["center"])
    assert abs(tip[2] - c[2]) < 0.02 and L.SADDLE_X[-1] < tip[0] < 0.09                     # beside the strum point, treble side
    far = L.arm_geometry({"swing": 40.0, "length": 0.18})[1]
    assert far[0] > tip[0] + 0.06 and np.linalg.norm(far - bend) == pytest.approx(0.18)
    assert not any(e["name"] == "trem" for e in L.card(NAME, has_arm=False)["use"]["anchor"])


def test_options_guard_and_arm():
    assert L.Pal().guard == "tortoise" and L.Pal({"guard": "plain"}).guard == "plain" and L.Pal().arm == L.ARM
    assert L.Pal({"arm": False}).arm is None and L.Pal({"arm": {"swing": 30}}).arm == dict(L.ARM, swing=30.0)
    with pytest.raises(ValueError, match="guard"):
        L.Pal({"guard": "pearl"})
    with pytest.raises(ValueError, match="swing, lift, length"):
        L.Pal({"arm": {"bend": 3}})
    assert L.Pal({"body": "iris"}).lin("body") == L.Pal().slot_lin("iris")


def test_the_body_is_the_photographs(parts):
    V = parts["body"].V
    assert V[:, 2].min() == pytest.approx(-0.195, abs=0.001) and V[:, 2].max() == pytest.approx(0.293, abs=0.001)
    assert V[:, 0].max() - V[:, 0].min() == pytest.approx(0.348, abs=0.002)                 # the lower bout
    o = B.outline()

    def width(z):
        x = [np.interp(z, [a[1], b[1]][::int(np.sign(b[1] - a[1])) or 1], [a[0], b[0]][::int(np.sign(b[1] - a[1])) or 1])
             for a, b in zip(o, np.roll(o, -1, 0)) if min(a[1], b[1]) <= z < max(a[1], b[1])]
        return min(x), max(x)

    assert width(0.125)[0] == pytest.approx(-0.1152, abs=0.001) and width(0.035)[1] == pytest.approx(0.1182, abs=0.001)
    assert width(0.125)[0] > width(-0.04)[0] + 0.05 and width(0.035)[1] < width(-0.115)[1] - 0.05    # the offset waist
    assert V[V[:, 2] > 0.26][:, 0].max() < -0.07                                            # the long horn is the bass side's
    guard = B.guard_outline()
    assert G.inside(guard, o).all() and G.distance(guard, o).min() > 0.002
    for part in ("pickups", "bridge", "knobs", "jack"):                                     # what stands on the guard
        P2 = parts[part].V[:, [0, 2]]
        assert G.inside(P2, guard).mean() > 0.97, part
    assert G.inside(parts["tremolo"].V[:, [0, 2]], o).all() and not G.inside(H.trem_outline() + [0, sum(L.TREM["z"]) / 2], guard).any()
    assert B.face(np.array([-0.165]), np.array([-0.04]))[0] > 0.004 and B.face(np.array([0.165]), np.array([-0.04]))[0] == 0.0
    assert B.back(np.array([-0.105]), np.array([0.085]))[0] < L.BODY_T - 0.01               # the belly cut, bass side


def test_strings_lie_on_their_saddles_clear_the_covers_and_end_in_the_tremolo(parts):
    y_cover = H.Y_G - L.PICKUP_H
    lo, hi = parts["bridge"].bbox()
    for i in range(6):
        nut, sad = (vec(p) for p in L.string_ends(i))
        V = parts[f"string_{6 - i}"].V
        r = L.STRING_R[i]
        span = V[(V[:, 2] < nut[2] - 1e-6) & (V[:, 2] > sad[2] + 1e-6)]
        t = (span[:, 2] - nut[2]) / (sad[2] - nut[2])
        axis = nut + t[:, None] * (sad - nut)
        assert np.abs(np.linalg.norm(span - axis, axis=1) - r).max() < 1e-4                 # exactly on the card's line
        for spec in L.PICKUPS.values():
            assert L.string_at(i, spec["z"])[1] + r < y_cover - 0.002                       # 2 mm over the covers
        c = H.saddle_axis(i)
        assert c[1] - L.SADDLE["r"] == pytest.approx(sad[1] + r) and lo[2] < c[2] < hi[2]   # the string on its barrel's crest
        end = V[V[:, 2] < L.ANCHOR["z"] + 1e-6]
        assert len(end) and abs(end[:, 0].mean() - sad[0]) < 1e-4
        assert -L.TREM["t"] - L.ANCHOR["h"] < end[:, 1].mean() < -L.TREM["t"]               # inside the anchor bar
    assert L.COLLET["at"][0] - L.COLLET["r"] > L.SADDLE_X[-1] + 0.002                       # the collet beside the high e


def test_every_part_is_a_closed_solid_and_far_from_boxy(parts):
    assert set(parts) >= {"body", "pickguard", "neck", "pickups", "bridge", "tremolo", "knobs", "switch", "jack", "buttons",
                          *(f"string_{n}" for n in range(1, 7)), "plug", "pick"}
    for k, m in parts.items():
        assert m.is_closed() and m.volume() > 0, k
    loc, R = H.pick_rest()
    bend, _tip, Ra = L.arm_geometry()
    extra = [("trem_arm", P.arm(), Ra, bend), ("cable", S.tube(H.cable_path(), L.CABLE_R, sides=8, up=(0.0, 0.0, 1.0)), None, None)]
    V, T, owner, names, off = [], [], [], [], 0
    for k, m, rot, at in [(k, m, R if k == "pick" else None, loc if k == "pick" else None) for k, m in parts.items()] + extra:
        v = m.V if rot is None else m.V @ rot.T + at
        t = m.triangles()
        V.append(v)
        T.append(t + off)
        owner.append(np.full(len(t), len(names)))
        names.append(k)
        off += len(v)
    r = F.analyse(np.concatenate(V), np.concatenate(T), np.concatenate(owner), names, exempt=P.GRAPHIC)
    assert r["score"] < 0.15, (r["score"], r["components"][:4])
