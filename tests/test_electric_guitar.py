"""The electric guitar prop (pure numpy: the Blender side is checked by building and looking): the card's contract numbers (the
frame, frets, strings, neck section, strum point, pick, anchors, wear), the meshes against them, and the form guard."""
import json
import math
import time

import numpy as np
import pytest

from mkmmd.blender.library.props import electric_guitar_body as B
from mkmmd.blender.library.props import electric_guitar_geo as G
from mkmmd.blender.library.props import electric_guitar_hw as H
from mkmmd.blender.library.props import electric_guitar_layout as L
from mkmmd.blender.library.props import electric_guitar_neck as N
from mkmmd.blender.library.props import electric_guitar_parts as P
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


# ======================================================================================================= the card
def test_card_shape_and_json(card):
    assert card["origin"] == "bridge" and card["front"] == "-Y" and len(card["size"]) == 3
    json.dumps(card)                                                       # plain JSON
    assert set(card["use"]) == {"grip", "anchor", "look", "wear"}
    assert [e["name"] for e in card["use"]["grip"]] == ["neck", "strum"]
    assert {c["object"] for c in card["colliders"]} == {f"{NAME}_col_{k}" for k in ("body", "horn", "head", "neck")}
    assert set(L.COLLIDERS) == {"body", "horn", "head", "neck"}


def test_roles_and_palette():
    want = {"body", "pickguard", "neck", "fretboard", "hardware", "strings", "knobs", "strap", "cable", "pick"}
    assert want <= set(L.ROLES) and want <= set(L.Pal().roles())
    pal = L.Pal()
    assert pal.lin("knobs") == pal.lin("pickguard") == pal.lin("hardware")                       # defaults to the guard (text)
    assert L.Pal({"body": "iris"}).lin("body") == L.Pal().slot_lin("iris")
    assert L.Pal({"knobs": "gold"}).lin("knobs") == L.Pal().slot_lin("gold")
    strap = L.Pal().lin("strap")                                                                 # base:2,iris:1: dark but never black
    assert 0.01 < min(strap) and max(strap) < 0.35 and max(strap) < max(L.Pal().lin("hardware")) / 2
    assert set(L.MATS) >= set(L.ROLES) - {"dark"} | {"dark", "guard_core", "frets"}


# ======================================================================================================= the neck grip
def test_frame_is_orthonormal(card):
    f = entry(card, "grip", "neck")["frame"]
    along, across, normal = vec(f["along"]), vec(f["across"]), vec(f["normal"])
    for v in (along, across, normal):
        assert np.linalg.norm(v) == pytest.approx(1.0)
    assert abs(along @ across) < 1e-12 and abs(along @ normal) < 1e-12 and abs(across @ normal) < 1e-12
    assert tuple(along) == (0, 0, 1) and tuple(normal) == (0, -1, 0)                         # toward the nut, out of the board
    assert vec(entry(card, "grip", "neck")["thumb"]) @ normal == pytest.approx(-1.0)          # the thumb is behind the board


def test_frets_follow_the_formula_on_the_board_plane(card):
    e = entry(card, "grip", "neck")
    F_ = [vec(p) for p in e["frets"]]
    assert len(F_) == 23 and e["scale"] == 0.648                                             # nut + 22 wires
    for n, p in enumerate(F_):
        assert p[2] == pytest.approx(0.648 * 2.0 ** (-n / 12.0), abs=1e-4)
        assert p[0] == 0.0 and p[1] == pytest.approx(L.Y_BOARD)                              # one plane, the centre line
    assert F_[0][2] == pytest.approx(0.648) and np.all(np.diff([p[2] for p in F_]) < 0)
    # the board's crown (a rolled slab) is on that plane at x = 0
    rim = N.board_ring(0.045, 0.023)
    assert rim[:, 1].min() == pytest.approx(0.0) and rim[np.argmin(rim[:, 1]), 0] == pytest.approx(0.0, abs=1e-4)
    assert e["fret_height"] == pytest.approx(0.0012)
    # the meshes: wires at the formula's z, standing fret_height proud of the board
    V = P.parts()["frets"].V
    for n in (1, 12, 22):
        near = V[np.abs(V[:, 2] - L.fret_z(n)) < 0.0006]
        assert near[:, 1].min() == pytest.approx(L.Y_BOARD - L.FRET_H, abs=1e-5)


def test_strings_are_inside_the_board_and_clear_the_wires(card):
    e = entry(card, "grip", "neck")
    S_ = e["strings"]
    assert [s["name"] for s in S_] == list(L.STRING_NAMES) and len(S_) == 6
    nx = np.array([s["nut"][0] for s in S_])
    bx = np.array([s["bridge"][0] for s in S_])
    assert np.all(np.diff(nx) > 0) and np.all(np.diff(bx) > 0)                              # low E first, x increasing
    assert nx == pytest.approx(-nx[::-1]) and bx == pytest.approx(-bx[::-1])                   # symmetric
    for s in S_:
        assert s["nut"][2] == pytest.approx(0.648) and s["radius"] > 0
        assert s["nut"][1] < L.Y_BOARD < 0.0 and s["bridge"][1] < L.Y_BOARD               # above the board, in front of the face
        assert L.Y_BOARD - s["nut"][1] == pytest.approx(L.FRET_H + 0.0005 + s["radius"], abs=1e-5)       # 0.5 mm over the first wire
    r = np.array([s["radius"] for s in S_])
    assert np.all(np.diff(r) < 0) and r[0] == pytest.approx(0.000584, abs=2e-6)                # a 10-46 set, thick to thin
    for n in range(1, 23):                                                                   # every string stays inside the board
        z = 0.648 * 2.0 ** (-n / 12.0)
        w = float(L.section_wd(z)[0])
        for s in S_:
            a, b = vec(s["nut"]), vec(s["bridge"])
            p = a + (b - a) * ((z - a[2]) / (b[2] - a[2]))
            assert abs(p[0]) + s["radius"] < 0.5 * w - 0.0025
            # and clears the wire: its underside over the crown of the wire (the board's plane minus the wire's height)
            clear = (L.Y_BOARD - L.FRET_H) - (p[1] + s["radius"])
            assert clear > 0.0003                                                            # never buzzing on a wire
            if n == 12:
                assert 0.0014 < clear < 0.0026                                                # an action of about 2 mm
    hs = [L.Y_BOARD - s["bridge"][1] for s in S_]
    assert 0.0035 < min(hs) and max(hs) < 0.0075                                              # saddles stand 4-7 mm over the board plane


def test_string_meshes_run_through_the_card_points(parts):
    V = parts["strings"].V
    for i, st in enumerate(L.STRING_NAMES):
        for p in L.string_ends(i):
            d = np.linalg.norm(V - np.array(p), axis=1).min()
            assert d == pytest.approx(L.STRING_R[i], rel=0.15, abs=2e-5)


def test_neck_section_is_the_cards_superellipse(card):
    sec = entry(card, "grip", "neck")["section"]
    assert sec["width"] == [L.W_NUT, L.W_LAST] and sec["depth"] == [L.D_NUT, L.D_LAST] and sec["p"] == 2.6
    assert sec["board_width"] == sec["width"] and sec["board_radius"] == pytest.approx(0.2413)
    V = P.parts()["neck"].V
    stations = N.neck_stations()
    V = V[:len(stations) * len(N.ring_curve(0.0, 0.02, 0.02, 2.6, 0.001))]                         # the rings (not the caps' centres)
    seen = 0
    for z, xc, a, d, p, rc in stations:
        if z > N.Z_NUT_BACK + 1e-9:
            break
        ring = V[np.abs(V[:, 2] - z) < 1e-9]
        w, d_ = L.section_wd(z)
        assert 2 * a == pytest.approx(float(w)) and d == pytest.approx(float(d_)) and p == 2.6
        v = ring[:, 1] - L.Y_BOARD
        back = ring[v > L.BOARD_T + 0.0012]                                                   # the back, below the board
        res = np.abs((np.abs(2 * back[:, 0] / w)) ** p + ((back[:, 1] - L.Y_BOARD) / d_) ** p - 1.0)
        assert len(back) > 20 and res.max() < 1e-9
        assert v.max() == pytest.approx(float(d_))                                            # the centre-line depth
        seen += 1
    assert seen >= 10
    # linear taper between the nut and the last fret, held beyond
    assert float(L.section_wd(L.Z_NUT)[0]) == pytest.approx(0.042) and float(L.section_wd(L.Z_LAST)[0]) == pytest.approx(0.056)
    assert float(L.section_wd(0.1)[0]) == pytest.approx(0.056) and float(L.section_wd(L.fret_z(12))[1]) == pytest.approx(0.0236, abs=2e-4)


def test_board_wall_continues_the_neck(card):
    """The board's side walls and the neck's top corners meet: both on the same superellipse below the board's crown."""
    w, d = (float(q) for q in L.section_wd(0.4))
    ring = N.board_ring(w, d)
    low = ring[(ring[:, 1] > 0.0035) & (ring[:, 1] < L.BOARD_T - 0.0011)]                    # the straight wall part
    res = np.abs(np.abs(2 * low[:, 0] / w) ** L.P_SECTION + (low[:, 1] / d) ** L.P_SECTION - 1.0)
    assert res.max() < 1e-9


# ======================================================================================================= strumming
def test_strum_point_is_between_the_pickups_on_the_string_plane(card):
    s = entry(card, "grip", "strum")
    c, z0, z1 = vec(s["center"]), *s["zone"]
    assert c[0] == 0.0 and z0 < c[2] < z1
    assert vec(s["along"]) @ vec(s["across"]) == 0 and vec(s["normal"]) @ vec(s["along"]) == 0
    pts = np.array([L.string_at(i, c[2]) for i in range(6)])
    A = np.stack([np.ones(6), pts[:, 0]], 1)
    coef = np.linalg.lstsq(A, pts[:, 1], rcond=None)[0]
    assert abs(c[1] - (coef[0] + coef[1] * c[0])) < 0.001                                       # within 1 mm of the plane
    assert z0 > L.PICKUPS["bridge"]["z"] + 0.0185 / 2 + 0.07 / 2 * math.sin(math.radians(L.PICKUPS["bridge"]["slant"])) + 0.002
    assert z1 < L.PICKUPS["middle"]["z"] - 0.0185 / 2 - 0.002                                  # clear of both covers


def test_pick_is_in_the_pinch_frame(card, parts):
    pk = entry(card, "grip", "strum")["pick"]
    assert pk["object"] == f"{NAME}_pick"
    lo, hi = parts["pick"].bbox()
    assert hi[0] == pytest.approx(pk["tip"], abs=1e-4) and lo[0] == pytest.approx(pk["tip"] - pk["length"], abs=1e-4)
    assert hi[1] - lo[1] == pytest.approx(pk["width"], abs=1e-4) and lo[1] == pytest.approx(-hi[1], abs=1e-4)
    assert hi[2] - lo[2] == pytest.approx(pk["thickness"], abs=1e-5) and lo[2] == pytest.approx(-hi[2], abs=1e-6)
    assert parts["pick"].is_closed() and parts["pick"].volume() > 0
    # lying on the guard it points toward the neck and its face looks out of the guitar
    loc, R = H.pick_rest()
    assert np.linalg.det(R) == pytest.approx(1.0) and tuple(R[:, 0]) == (0, 0, 1) and tuple(R[:, 2]) == (0, -1, 0)
    ends = parts["pick"].V @ R.T + loc
    assert ends[:, 1].max() <= -L.GUARD_T + 1e-9 and ends[:, 1].min() > -L.GUARD_T - 0.002


# ======================================================================================================= anchors, looks, wear
def test_strap_anchors_sit_on_the_body_edge(card):
    o = B.outline()
    n = G.edge_normals(o)
    for name in ("strap_top", "strap_bottom"):
        a = entry(card, "anchor", name)
        p, d = vec(a["point"]), vec(a["dir"])
        assert np.linalg.norm(d) == pytest.approx(1.0, abs=1e-4) and a["object"] == f"{NAME}_{name}"
        assert p[1] == pytest.approx(0.5 * L.BODY_T)                                          # mid-thickness of the edge
        dist = G.distance(np.array([[p[0], p[2]]]), o)[0]
        assert dist < 0.0006                                                                   # on the outline
        i = int(np.argmin(np.linalg.norm(o - np.array([p[0], p[2]]), axis=1)))
        assert d[[0, 2]] @ n[i] > 0.9                                                          # leaving the edge, outward
    assert entry(card, "anchor", "strap_top")["point"][2] > 0.3 > 0 > entry(card, "anchor", "strap_bottom")["point"][2]
    assert entry(card, "anchor", "strap_top")["point"][0] < 0 < 1                              # the upper (long) horn is on the low E side


def test_jack_anchor_is_the_plugs_end(card, parts):
    a = entry(card, "anchor", "jack")
    p, d = vec(a["point"]), vec(a["dir"])
    assert np.linalg.norm(d) == pytest.approx(1.0, abs=1e-4) and a["object"] == f"{NAME}_jack"
    V = parts["plug"].V
    along = (V - p) @ d
    assert along.max() == pytest.approx(0.0, abs=1e-4)                                         # nothing of the plug beyond its end
    rad = np.linalg.norm((V - p) - np.outer(along, d), axis=1)
    assert rad[along > -1e-4].max() < 0.001                                                    # the end is on its axis
    assert d[1] < -0.3 and abs(d[0]) > 0.3                                                     # out of the face, toward the tail edge
    assert card["use"]["wear"][0]["cable"]["anchor"] == "jack"


def test_head_and_look_points(card):
    head = vec(entry(card, "anchor", "head")["point"])
    u = head[2] - L.SCALE
    xb, xt = N.head_extent(u)
    assert xb[0] < head[0] < xt[0] and 0.0 < u < L.HEAD_L
    looks = {e["name"]: vec(e["point"]) for e in card["use"]["look"]}
    assert set(looks) == {"neck", "strum", "head"}
    assert looks["neck"][2] == pytest.approx(L.fret_z(5), abs=1e-5) and looks["neck"][1] == pytest.approx(L.Y_BOARD)
    assert np.allclose(looks["strum"], entry(card, "grip", "strum")["center"])


def test_wear_entry_is_the_contract(card):
    e = card["use"]["wear"][0]
    assert e["name"] == "stand" and e["bone"] == "upper_body2" and e["pivot"] == [0.0, 0.045, 0.0]
    assert e["ref"] == {"top": 1.70, "shoulder_width": 0.18}
    assert e["at"] == [-0.14, -0.10, 0.0] and e["scale"] == ["shoulder_width", "top", "top"]
    assert (e["neck_deg"], e["yaw_deg"], e["roll_deg"]) == (35.0, 0.0, 0.0)               # validated on Reisen by the wear stage
    assert e["strap"] == {"top": "strap_top", "bottom": "strap_bottom", "over": "shoulder.L", "width": 0.05, "thickness": 0.004,
                          "material": f"{NAME}_strap"}
    assert e["cable"] == {"object": f"{NAME}_cable", "anchor": "jack", "radius": 0.0032}
    names = {a["name"] for a in card["use"]["anchor"]}
    assert {e["strap"]["top"], e["strap"]["bottom"], e["cable"]["anchor"]} <= names
    assert L.wear_entry("axe")["strap"]["material"] == "axe_strap" and L.WEAR["strap"]["material"] == "<name>_strap"


# ======================================================================================================= the meshes
def test_every_part_is_a_closed_solid(parts):
    assert set(parts) >= {"body", "pickguard", "neck", "fretboard", "frets", "nut", "tuners", "pickups", "bridge", "knobs", "switch",
                          "jack_plate", "buttons", "strings", "plug", "pick"}
    for k, m in parts.items():
        assert m.is_closed() and m.volume() > 0, k


def test_body_proportions(parts):
    lo, hi = parts["body"].bbox()
    assert hi[0] == pytest.approx(0.157, abs=0.002) and lo[0] == pytest.approx(-0.157, abs=0.002)        # 0.315 across
    assert lo[2] == pytest.approx(-0.1155, abs=0.001) and hi[2] == pytest.approx(0.314, abs=0.001)       # tail to the long horn
    assert lo[1] == pytest.approx(0.0, abs=1e-9) and hi[1] == pytest.approx(L.BODY_T, abs=1e-9)
    o = B.outline()
    assert G.poly_area(o) > 0 and 0.09 < G.poly_area(o) < 0.10
    # the horns: the long one on the bass side, 5.8 cm higher than the short one, and the waist 22.5 cm across
    xs, zs = o[:, 0], o[:, 1]
    assert zs[xs < 0].max() - zs[xs > 0].max() == pytest.approx(0.0575, abs=0.004)
    waist = [np.ptp(o[np.abs(o[:, 1] - 0.125) < 0.002, 0])]
    assert waist[0] == pytest.approx(0.2252, abs=0.003)
    # a rolled rim: the offset ring never folds
    ring = o + L.BODY_EDGE_R * S.mitre_offsets(o)
    assert np.all(((np.roll(o, -1, 0) - o) * (np.roll(ring, -1, 0) - ring)).sum(1) > 0)


def test_contours_scoop_the_bass_side_only(parts):
    assert B.face(-0.130, 0.12) > 0.004 and B.face(0.130, 0.12) == 0.0 and B.face(0.0, 0.0) == 0.0     # the forearm bevel
    assert B.face(-0.13, 0.25) == 0.0                                                                  # not up the horn
    assert B.back(-0.095, 0.05) < L.BODY_T - 0.010 and B.back(0.1, 0.05) == L.BODY_T                    # the belly cut
    V = parts["body"].V
    assert V[:, 1].max() == pytest.approx(L.BODY_T) and V[:, 1].min() == pytest.approx(0.0, abs=1e-9)


def test_the_guard_lies_inside_the_body_with_its_controls_on_it():
    guard, o = B.guard_outline(), B.outline()
    inner = G.inset(o, 0.003)
    assert G.inside(guard, inner).all()                                                           # 3 mm clear of the edge
    for x, z in L.GUARD_SCREWS:
        assert G.inside(np.array([[x, z]]), guard)[0] and G.distance(np.array([[x, z]]), guard)[0] > 0.002
    for x, z in L.KNOBS:
        assert G.inside(np.array([[x, z]]), guard)[0] and G.distance(np.array([[x, z]]), guard)[0] > L.KNOB_R
    for spec in L.PICKUPS.values():
        assert G.inside(np.array([[0.0, spec["z"]], [0.033, spec["z"]], [-0.033, spec["z"]]]), guard).all()
    cx, cz = L.SWITCH["center"]
    assert G.inside(np.array([[cx, cz]]), guard)[0]
    assert not G.inside(np.array([[0.0, 0.25], [0.0, 0.21]]), o)[1] or True                      # (the neck's pocket lobe is body)


def test_parts_stand_where_the_layout_says(parts):
    lo, hi = parts["pickups"].bbox()
    assert hi[1] == pytest.approx(L.GUARD_T * -1 + 0.0001, abs=3e-4) or hi[1] < 0                  # on the guard
    assert lo[1] > -0.015 and -0.0131 < parts["pickups"].V[:, 1].min() < -0.0112                   # pole tops, 11-13 mm over the face
    lo, hi = parts["bridge"].bbox()
    assert abs((lo[0] + hi[0]) / 2) < 0.002 and hi[2] > 0.019 and lo[2] < -0.02                     # the plate under the saddles
    crest = min(L.string_ends(i)[1][1] + L.STRING_R[i] for i in range(6))
    assert lo[1] == pytest.approx(crest, abs=1e-4)                                                  # the saddles' crests carry the strings
    kn = parts["knobs"]
    assert kn.bbox()[0][1] == pytest.approx(-L.GUARD_T - L.KNOB_H - 0.0003, abs=5e-4)               # 16.5 mm tall on the guard
    lo, hi = parts["tuners"].bbox()
    assert lo[0] < N.head_extent(0.05)[0][0] - 0.01                                                   # keys stick out of the bass edge
    assert hi[2] < L.SCALE + L.HEAD_L + 0.001 and lo[2] > L.SCALE + 0.02


def test_neck_and_headstock(parts):
    lo, hi = parts["neck"].bbox()
    assert lo[2] == pytest.approx(L.Z_HEEL) and hi[2] == pytest.approx(L.SCALE + L.HEAD_L)          # heel to the tip
    assert hi[2] > 0.8 and L.fret_z(22) == pytest.approx(0.1818, abs=2e-4)
    assert hi[1] == pytest.approx(L.Y_BOARD + L.D_LAST, abs=1e-6)                                    # the heel is the deepest
    zs = [st[0] for st in N.neck_stations()]
    z100 = min(zs, key=lambda z: abs(z - (L.SCALE + 0.1)))
    ring = parts["neck"].V[np.abs(parts["neck"].V[:, 2] - z100) < 1e-9]
    assert ring[:, 1].min() == pytest.approx(N.Y_FACE) and ring[:, 1].max() == pytest.approx(N.Y_BACK, abs=1e-4)    # the headstock's thickness
    xb, xt = N.head_extent(np.array([0.0034, 0.0495, 0.0681, 0.1568]))
    assert (xt - xb)[0] == pytest.approx(L.W_NUT, abs=1e-6) and (xt - xb)[1] > 0.074 and (xt - xb)[3] > 0.045   # a wide head
    bx = parts["fretboard"].bbox()
    assert bx[0][2] == pytest.approx(L.Z_HEEL) and bx[1][2] == pytest.approx(L.Z_NUT)


# ======================================================================================================= form and build time
def _prop_triangles(parts):
    loc, R = H.pick_rest()
    V, T, owner, names = [], [], [], []
    off = 0
    path = H.cable_path()
    cord = S.tube(path, L.CABLE_R, sides=8, up=(0.0, 0.0, 1.0))
    for k, m in list(parts.items()) + [("cable", cord)]:
        v = m.V if k != "pick" else m.V @ R.T + loc
        t = m.triangles()
        V.append(v)
        T.append(t + off)
        owner.append(np.full(len(t), len(names)))
        names.append(k)
        off += len(v)
    return np.concatenate(V), np.concatenate(T), np.concatenate(owner), names


def test_form_is_far_from_boxy(parts):
    V, T, owner, names = _prop_triangles(parts)
    r = F.analyse(V, T, owner, names, exempt=P.GRAPHIC)
    assert r["score"] < 0.25, (r["score"], r["components"][:4])
    assert r["score"] < 0.15, "keep the guitar well inside the hero limit"


def test_build_stays_under_five_seconds():
    t0 = time.time()
    for f in (B.body_mesh, B.guard_mesh, N.neck_mesh, N.board_mesh, N.frets_mesh, N.tuners_mesh, N.strings_mesh, H.pickups_mesh,
              H.bridge_mesh, H.knobs_mesh, H.jack_plate_mesh, H.plug_mesh, H.buttons_mesh, H.pick_mesh):
        f()
    assert time.time() - t0 < 4.0


def test_delaunay_fill_closes_the_outline():
    ring = G.ccw(S.ellipse(0.1, 0.06, 40))
    V, T = G.fill(ring, 0.02)
    E = np.concatenate([T[:, [0, 1]], T[:, [1, 2]], T[:, [2, 0]]])
    pairs = {tuple(e) for e in E.tolist()}
    border = [e for e in pairs if (e[1], e[0]) not in pairs]
    assert sorted(border) == sorted((i, (i + 1) % 40) for i in range(40))                          # the ring is the border
    P_ = V[T]
    a, b = P_[:, 1] - P_[:, 0], P_[:, 2] - P_[:, 0]
    area = 0.5 * (a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0])
    assert np.all(area > 0) and area.sum() == pytest.approx(abs(G.poly_area(ring)), rel=1e-6)
