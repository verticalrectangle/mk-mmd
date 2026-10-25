"""The electric bass prop (pure numpy: the Blender side is checked by building and looking): the card's contract numbers (the
frame, the 19 frets of a 30 inch scale, four strings and their tuning, the neck section, the strum point, the pick, anchors,
wear), the meshes against them (body proportions of the photograph, the guard, plate, pickup and bridge where the layout says,
the headstock and its keys), the optional stripes, and the form guard."""
import json
import math
import time

import numpy as np
import pytest

from mkmmd.blender.library.props import electric_bass_body as B
from mkmmd.blender.library.props import electric_bass_hw as H
from mkmmd.blender.library.props import electric_bass_layout as L
from mkmmd.blender.library.props import electric_bass_neck as N
from mkmmd.blender.library.props import electric_bass_parts as P
from mkmmd.blender.library.props import electric_guitar_geo as G
from mkmmd.core import form as F
from mkmmd.core import fretting as FR
from mkmmd.core import shell as S

NAME = "bass"


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
    json.dumps(card)
    assert set(card["use"]) == {"grip", "anchor", "look", "wear"}
    assert [e["name"] for e in card["use"]["grip"]] == ["neck", "strum"]
    assert {c["object"] for c in card["colliders"]} == {f"{NAME}_col_{k}" for k in ("body", "horn", "head", "neck")}
    assert set(L.COLLIDERS) == {"body", "horn", "head", "neck"}
    assert 1.10 < card["size"][2] < 1.16 and 0.39 < card["size"][0] < 0.42       # tail to tip with the cord, keys to the treble edge


def test_roles_palette_and_stripes():
    want = {"body", "pickguard", "neck", "fretboard", "hardware", "strings", "knobs", "strap", "cable", "pick", "stripe"}
    assert want <= set(L.ROLES) and want <= set(L.Pal().roles()) and set(L.MATS) >= set(L.ROLES) | {"guard_core", "frets"}
    assert L.Pal({"body": "iris"}).lin("body") == L.Pal().slot_lin("iris")
    knobs = L.Pal().lin("knobs")                                                     # dark plastic, never black
    assert 0.0 < min(knobs) and max(knobs) < max(L.Pal().lin("hardware")) / 2
    assert not L.Pal().stripes and not L.Pal({"love": "#ff0000"}).stripes and L.Pal({"stripe": "text"}).stripes
    assert "stripes" not in P.parts() and "stripes" in P.parts(stripes=True)


# ======================================================================================================= the neck grip
def test_frame_tuning_and_frets_follow_the_short_scale(card):
    e = entry(card, "grip", "neck")
    f = e["frame"]
    along, across, normal = vec(f["along"]), vec(f["across"]), vec(f["normal"])
    assert tuple(along) == (0, 0, 1) and tuple(across) == (1, 0, 0) and tuple(normal) == (0, -1, 0)
    assert vec(e["thumb"]) @ normal == pytest.approx(-1.0)
    assert e["tuning"] == [28, 33, 38, 43] and [s["midi"] for s in e["strings"]] == e["tuning"]
    assert FR.tuning_of({}, e) == [28, 33, 38, 43]
    fr = [vec(p) for p in e["frets"]]
    assert len(fr) == 20 and e["scale"] == 0.762                                      # nut + 19 wires
    for n, p in enumerate(fr):
        assert p[2] == pytest.approx(0.762 * 2.0 ** (-n / 12.0), abs=1e-5) and p[1] == pytest.approx(L.Y_BOARD) and p[0] == 0.0
    V = P.parts()["frets"].V
    for n in (1, 12, 19):                                                             # the wires at the formula's z, fret_height proud
        near = V[np.abs(V[:, 2] - L.fret_z(n)) < 0.0005]
        assert near[:, 1].min() == pytest.approx(L.Y_BOARD - L.FRET_H, abs=1e-5)
    assert L.Z_HEEL == pytest.approx(L.fret_z(19) - 0.011)                           # the board runs on past the last wire


def test_strings_are_inside_the_board_and_clear_the_wires(card):
    S_ = entry(card, "grip", "neck")["strings"]
    assert [s["name"] for s in S_] == ["E", "A", "D", "G"]
    nx, bx = np.array([s["nut"][0] for s in S_]), np.array([s["bridge"][0] for s in S_])
    assert np.all(np.diff(nx) > 0) and np.all(np.diff(bx) > 0) and nx == pytest.approx(-nx[::-1])
    assert np.diff(bx) == pytest.approx(0.0177) and np.diff(nx) == pytest.approx(0.0106)      # 17.7 mm at the saddles
    r = np.array([s["radius"] for s in S_])
    assert np.all(np.diff(r) < 0) and r[0] == pytest.approx(0.5 * 0.0254 * 0.105, abs=1e-6)  # a 45-105 set
    for s in S_:
        assert s["nut"][2] == pytest.approx(0.762) and s["bridge"][2] < 0.0                  # saddles behind the nominal line
        assert L.Y_BOARD - s["nut"][1] == pytest.approx(L.FRET_H + L.NUT_CLEAR + s["radius"], abs=1e-5)
    for n in range(1, 20):
        z = L.fret_z(n)
        w = float(L.section_wd(z)[0])
        for s in S_:
            a, b = vec(s["nut"]), vec(s["bridge"])
            p = a + (b - a) * ((z - a[2]) / (b[2] - a[2]))
            assert abs(p[0]) + s["radius"] < 0.5 * w - 0.002                              # inside the board
            clear = (L.Y_BOARD - L.FRET_H) - (p[1] + s["radius"])
            assert clear > 0.0005                                                         # never buzzing on a wire
            if n == 12:
                assert 0.0022 < clear < 0.0032                                            # a bass's action at the 12th
    hs = [L.Y_BOARD - s["bridge"][1] for s in S_]
    assert 0.005 < min(hs) and max(hs) < 0.009                                            # the saddles stand that high over the board


def test_string_meshes_run_through_the_card_points(parts):
    V = parts["strings"].V
    for i in range(4):
        for p in L.string_ends(i):
            assert np.linalg.norm(V - np.array(p), axis=1).min() == pytest.approx(L.STRING_R[i], rel=0.15, abs=2e-5)


def test_neck_section_is_the_cards_superellipse(card, parts):
    sec = entry(card, "grip", "neck")["section"]
    assert sec["width"] == [0.0413, 0.0570] and sec["depth"] == [L.D_NUT, L.D_LAST] and sec["p"] == 2.6
    assert sec["board_radius"] == pytest.approx(0.18415)                                  # 7.25 inches
    stations = N.neck_stations()
    V = parts["neck"].V[:len(stations) * len(N.GN.ring_curve(0.0, 0.02, 0.02, 2.6, 0.001))]
    seen = 0
    for z, xc, a, d, p, rc in stations:
        if z > N.Z_NUT_BACK + 1e-9:
            break
        ring = V[np.abs(V[:, 2] - z) < 1e-9]
        w, d_ = L.section_wd(z)
        assert 2 * a == pytest.approx(float(w)) and d == pytest.approx(float(d_))
        back = ring[ring[:, 1] - L.Y_BOARD > L.BOARD_T + 0.0012]
        res = np.abs(np.abs(2 * back[:, 0] / w) ** p + ((back[:, 1] - L.Y_BOARD) / d_) ** p - 1.0)
        assert len(back) > 20 and res.max() < 1e-9
        seen += 1
    assert seen >= 12
    ring = N.board_ring(*(float(q) for q in L.section_wd(0.5)))                          # the board's wall continues the neck
    low = ring[(ring[:, 1] > 0.0035) & (ring[:, 1] < L.BOARD_T - 0.0011)]
    w, d = (float(q) for q in L.section_wd(0.5))
    assert np.abs(np.abs(2 * low[:, 0] / w) ** 2.6 + (low[:, 1] / d) ** 2.6 - 1.0).max() < 1e-9


# ======================================================================================================= picking
def test_strum_point_is_between_the_pickup_and_the_neck_on_the_string_plane(card):
    s = entry(card, "grip", "strum")
    c, (z0, z1) = vec(s["center"]), s["zone"]
    assert c[0] == 0.0 and z0 < c[2] < z1
    pts = np.array([L.string_at(i, c[2]) for i in range(4)])
    assert abs(c[1] - pts[:, 1].mean()) < 1e-5 and np.ptp(pts[:, 1]) < 0.0025            # on the strings' plane (graded action)
    top_of_covers = max(spec["center"][1] + 0.5 * L.PICKUP_W for spec in L.PICKUPS.values())
    assert z0 > top_of_covers + 0.005 and z1 < L.Z_HEEL - 0.005                           # clear of the covers and the neck's end
    neck = entry(card, "grip", "neck")
    lo, hi = (FR.string_point(neck, k, c[2])[0] for k in (4, 1))
    assert 0.021 < 0.5 * (hi - lo) < 0.027                                                # sigma: half the four strings' width


def test_pick_is_in_the_pinch_frame(card, parts):
    pk = entry(card, "grip", "strum")["pick"]
    assert pk["object"] == f"{NAME}_pick"
    lo, hi = parts["pick"].bbox()
    assert hi[0] == pytest.approx(pk["tip"], abs=1e-4) and lo[0] == pytest.approx(pk["tip"] - pk["length"], abs=1e-4)
    assert hi[2] - lo[2] == pytest.approx(pk["thickness"], abs=1e-5) and lo[2] == pytest.approx(-hi[2], abs=1e-6)
    loc, R = H.pick_rest()
    assert np.linalg.det(R) == pytest.approx(1.0) and tuple(R[:, 2]) == (0, -1, 0)
    rest = parts["pick"].V @ R.T + loc
    assert rest[:, 1].max() <= 1e-9 and rest[:, 1].min() > -0.002                         # lying on the face
    assert G.inside(rest[:, [0, 2]], B.outline()).all() and not G.inside(rest[:, [0, 2]], B.guard_outline()).any()


# ======================================================================================================= anchors, looks, wear
def test_strap_anchors_sit_on_the_body_edge(card):
    o = B.outline()
    n = G.edge_normals(o)
    for name in ("strap_top", "strap_bottom"):
        a = entry(card, "anchor", name)
        p, d = vec(a["point"]), vec(a["dir"])
        assert np.linalg.norm(d) == pytest.approx(1.0, abs=1e-4) and p[1] == pytest.approx(0.5 * L.BODY_T)
        assert G.distance(np.array([[p[0], p[2]]]), o)[0] < 0.0008
        i = int(np.argmin(np.linalg.norm(o - np.array([p[0], p[2]]), axis=1)))
        assert d[[0, 2]] @ n[i] > 0.85
    top = entry(card, "anchor", "strap_top")["point"]
    assert top[0] < 0 and top[2] == pytest.approx(L.fret_z(12), abs=0.008)               # the long horn reaches the 12th fret


def test_jack_anchor_is_the_plugs_end_straight_out_of_the_plate(card, parts):
    a = entry(card, "anchor", "jack")
    p, d = vec(a["point"]), vec(a["dir"])
    assert tuple(np.round(d, 6)) == (0.0, -1.0, 0.0) and card["use"]["wear"][0]["cable"]["anchor"] == "jack"
    V = parts["plug"].V
    along = (V - p) @ d
    assert along.max() == pytest.approx(0.0, abs=1e-4)
    assert np.linalg.norm((V - p) - np.outer(along, d), axis=1)[along > -1e-4].max() < 0.001
    assert G.inside(np.array([[p[0], p[2]]]), B.plate_outline())[0]                       # on the control plate


def test_head_and_look_points(card):
    head = vec(entry(card, "anchor", "head")["point"])
    xb, xt = N.head_extent(head[2] - L.SCALE)
    assert xb[0] < head[0] < xt[0]
    looks = {e["name"]: vec(e["point"]) for e in card["use"]["look"]}
    assert set(looks) == {"neck", "strum", "head"}
    assert looks["neck"][2] == pytest.approx(L.fret_z(5), abs=1e-5) and np.allclose(looks["strum"], entry(card, "grip", "strum")["center"])


def test_wear_entry_is_the_contract(card):
    e = card["use"]["wear"][0]
    assert e["name"] == "stand" and e["bone"] == "upper_body2" and e["pivot"] == [0.0, L.BODY_T, 0.0]
    assert e["ref"] == {"top": 1.70, "shoulder_width": 0.18} and e["scale"] == ["shoulder_width", "top", "top"]
    assert e["strap"]["top"] == "strap_top" and e["strap"]["bottom"] == "strap_bottom" and e["strap"]["material"] == f"{NAME}_strap"
    assert e["cable"] == {"object": f"{NAME}_cable", "anchor": "jack", "radius": L.CABLE_R}
    assert {e["strap"]["top"], e["strap"]["bottom"], e["cable"]["anchor"]} <= {a["name"] for a in card["use"]["anchor"]}


# ======================================================================================================= the meshes
def test_every_part_is_a_closed_solid():
    every = P.parts(stripes=True)
    assert set(every) >= {"body", "pickguard", "control_plate", "neck", "fretboard", "frets", "nut", "tuners", "tree", "pickups",
                          "bridge", "knobs", "plate_parts", "buttons", "back_plates", "strings", "plug", "pick", "decals", "stripes"}
    for k, m in every.items():
        assert m.is_closed() and m.volume() > 0, k


def test_body_proportions_are_the_photographs(parts):
    lo, hi = parts["body"].bbox()
    assert hi[0] - lo[0] == pytest.approx(0.314, abs=0.002)                              # 12 3/8 inches at the lower bout
    assert lo[2] == pytest.approx(-0.0718, abs=0.001) and hi[2] == pytest.approx(0.376, abs=0.0015)
    assert lo[1] == pytest.approx(0.0, abs=1e-9) and hi[1] == pytest.approx(L.BODY_T, abs=1e-9)
    o = B.outline()
    xs, zs = o[:, 0], o[:, 1]
    assert zs[xs < -0.07].max() - zs[xs > 0.07].max() == pytest.approx(0.066, abs=0.004)   # the long horn over the short one
    upper = (zs > 0.15) & (zs < 0.25)
    assert -xs[upper & (xs < 0)].max() == pytest.approx(0.0955 * L.X_SCALE, abs=0.0015)      # the bass waist
    mid = (zs > 0.12) & (zs < 0.20)
    assert xs[mid & (xs > 0)].min() == pytest.approx(0.1055 * L.X_SCALE, abs=0.0015)         # the treble waist, lower
    bay_b = (xs > -0.06) & (xs < -0.035) & (zs > 0.3)
    bay_t = (xs > 0.05) & (xs < 0.07) & (zs > 0.2)
    assert zs[bay_b].min() == pytest.approx(0.326, abs=0.002) and zs[bay_t].min() == pytest.approx(0.279, abs=0.002)
    ring = o + L.BODY_EDGE_R * S.mitre_offsets(o)
    assert np.all(((np.roll(o, -1, 0) - o) * (np.roll(ring, -1, 0) - ring)).sum(1) > 0)   # the rolled rim never folds


def test_contours_scoop_the_bass_side_only(parts):
    assert B.face(-0.140, 0.10) > 0.003 and B.face(0.140, 0.10) == 0.0 and B.face(0.0, 0.0) == 0.0
    assert B.face(-0.11, 0.30) == 0.0                                                     # not up the horn
    assert B.back(-0.075, 0.17) < L.BODY_T - 0.008 and B.back(0.10, 0.0) == L.BODY_T
    V = parts["body"].V
    assert V[:, 1].max() == pytest.approx(L.BODY_T) and V[:, 1].min() == pytest.approx(0.0, abs=1e-9)


def test_the_guard_and_the_plate_lie_on_the_body_with_their_parts():
    o, guard, plate = B.outline(), B.guard_outline(), B.plate_outline()
    assert G.inside(guard, G.inset(o, 0.003)).all() and G.inside(plate, G.inset(o, 0.002)).all()
    assert not G.inside(plate, guard).any() and not G.inside(guard, plate).any()          # side by side, not stacked
    assert G.distance(plate, guard).min() > 0.0004
    for q in B.guard_screw_points():                                                      # the screws stand on the guard
        assert G.inside(q[None], guard)[0] and G.distance(q[None], guard)[0] > 0.0035
    assert len(B.guard_screw_points()) == len(L.GUARD_SCREWS)
    for spec in L.PICKUPS.values():
        cover = S.stadium(L.PICKUP_L, L.PICKUP_W, 8) + np.array(spec["center"])
        assert G.inside(cover, guard).all()
    for x, z in L.KNOBS + (L.JACK["center"],):
        q = np.array([[x, z]])
        assert G.inside(q, plate)[0] and G.distance(q, plate)[0] > 0.008
    w = float(L.section_wd(L.Z_HEEL)[0])                                                  # the guard clears the neck's end
    assert not G.inside(np.array([[0.0, L.Z_HEEL + 0.001], [0.5 * w - 0.001, 0.27], [-0.5 * w + 0.001, 0.27]]), guard).any()


def test_the_split_pickup_sits_under_its_pairs_of_strings():
    for spec, strings in ((L.PICKUPS["bass"], (0, 1)), (L.PICKUPS["treble"], (2, 3))):
        cx, cz = spec["center"]
        top = -L.GUARD_T - L.PICKUP_H
        xs = [L.string_at(i, cz)[0] for i in strings]
        assert all(abs(x - cx) < 0.5 * L.PICKUP_L - 0.006 for x in xs)                    # both strings over the cover
        for i in strings:
            gap = top - (L.string_at(i, cz)[1] + L.STRING_R[i])
            assert 0.003 < gap < 0.006                                                    # 3-6 mm under the strings
    assert L.PICKUPS["bass"]["center"][1] > L.PICKUPS["treble"]["center"][1]              # the bass half nearer the neck


def test_the_bridge_carries_the_strings_through_the_body(parts):
    crest = min(L.string_ends(i)[1][1] + L.STRING_R[i] for i in range(4))
    assert parts["bridge"].bbox()[0][1] == pytest.approx(crest, abs=1e-4)                # the saddles' crests carry the strings
    lo, hi = parts["bridge"].bbox()
    assert lo[2] < L.TAIL["z"] - 0.005 and hi[2] == pytest.approx(L.BRIDGE["top"], abs=0.001)
    for x in L.SADDLE_X:                                                                  # a ferrule on the back behind every string
        V = parts["back_plates"].V
        assert (np.hypot(V[:, 0] - x, V[:, 2] - L.STRING_HOLE_Z) < 0.005).any()
    assert L.STRING_HOLE_Z < min(-s for s in L.SETBACK) - 0.02


def test_the_headstock_and_its_keys(parts):
    lo, hi = parts["neck"].bbox()
    assert lo[2] == pytest.approx(L.Z_HEEL) and hi[2] == pytest.approx(L.SCALE + N.U_END)
    assert L.SCALE + L.HEAD_L == pytest.approx(0.991, abs=0.001)
    xb, xt = N.head_extent(np.array([0.0, 0.05, 0.20]))
    assert (xt - xb)[0] == pytest.approx(L.W_NUT, abs=1e-4) and (xt - xb)[1] > 0.08 and (xt - xb)[2] > 0.055
    pz, px = np.array([p[0] for p in L.POSTS]), np.array([p[1] for p in L.POSTS])
    ang = math.degrees(math.atan2(px[-1] - px[0], pz[-1] - pz[0]))
    assert ang == pytest.approx(12.8, abs=0.5) and np.diff(pz) == pytest.approx(0.0461, abs=0.001)
    T = parts["tuners"].V
    for i, (z, x) in enumerate(L.POSTS):                                                  # every key sticks out of the bass edge
        _gear, key = N.key_point(i)
        edge = float(N.head_extent(key[2] - L.SCALE)[0][0])
        near = T[np.abs(T[:, 2] - key[2]) < 0.004]
        assert near[:, 0].min() < edge - 0.025 and key[0] < x - 0.03
    for i in range(4):                                                                    # the posts stand inside the headstock
        z, x = L.POSTS[i]
        b, t = N.head_extent(z - L.SCALE)
        assert b[0] + L.POST_R + 0.002 < x < t[0] - L.POST_R - 0.002


def test_the_stripes_lie_on_the_body_and_follow_its_edge():
    st = P.parts(stripes=True)["stripes"]
    V = st.V
    assert G.inside(V[:, [0, 2]], B.outline()).all()
    y = B.surface_y(V[:, 0], V[:, 2])
    assert (y - V[:, 1] > 0.00004).all() and (y - V[:, 1] < 0.0004).all()                # 0.05-0.35 mm over the surface
    d = B.edge_distance(V[:, 0], V[:, 2])
    assert d.min() < 0.001 and (V[:, 1] > 0.001).any()                                    # out to the rolled edge
    assert not G.inside(V[:, [0, 2]], H.bridge_outline()).any()                           # clear of the bridge


# ======================================================================================================= form and build time
def _prop_triangles(parts):
    loc, R = H.pick_rest()
    V, T, owner, names = [], [], [], []
    off = 0
    cord = S.tube(H.cable_path(), L.CABLE_R, sides=8, up=(0.0, 0.0, 1.0))
    for k, m in list(parts.items()) + [("cable", cord)]:
        v = m.V if k != "pick" else m.V @ R.T + loc
        t = m.triangles()
        V.append(v)
        T.append(t + off)
        owner.append(np.full(len(t), len(names)))
        names.append(k)
        off += len(v)
    return np.concatenate(V), np.concatenate(T), np.concatenate(owner), names


def test_form_is_far_from_boxy():
    V, T, owner, names = _prop_triangles(P.parts(stripes=True))
    r = F.analyse(V, T, owner, names, exempt=P.GRAPHIC)
    assert r["score"] < 0.15, (r["score"], r["components"][:4])


def test_build_stays_under_four_seconds():
    t0 = time.time()
    for f in (B.body_mesh, B.guard_mesh, B.plate_mesh, B.stripes_mesh, N.neck_mesh, N.board_mesh, N.frets_mesh, N.tuners_mesh,
              N.strings_mesh, H.pickups_mesh, H.bridge_mesh, H.knobs_mesh, H.plug_mesh, H.buttons_mesh, H.back_mesh, H.pick_mesh):
        f()
    assert time.time() - t0 < 4.0
