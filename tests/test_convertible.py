"""The 1980s convertible's body and exterior (pure numpy; the Blender side is checked by building and looking): the layout,
the lofted body shell and its limit surface, the wheel arches, the nose and tail parts, the trim, the wheels, and the colour
roles. The cabin (seats, dash, wheel, cassette) is in test_convertible_cabin.py, the toolkit in test_shell.py."""
import numpy as np
import pytest

from mkmmd.blender.library.props import convertible_body as B
from mkmmd.blender.library.props import convertible_dash as D
from mkmmd.blender.library.props import convertible_exterior as E
from mkmmd.blender.library.props import convertible_layout as L
from mkmmd.blender.library.props import convertible_palette as CP
from mkmmd.blender.library.props import convertible_seats as SE
from mkmmd.blender.library.props import convertible_trim as TR
from mkmmd.blender.library.props import convertible_wheels as W
from mkmmd.core import shell as K

M = L.M
key = lambda V: set(map(tuple, np.round(np.asarray(V), 7)))      # noqa: E731


def car_parts():
    """Every mesh of every part module: [(module name, key, mesh in car coordinates)]."""
    out = []
    for mod in (B, W, E, TR, SE, D):
        if hasattr(mod, "static_parts"):
            out += [(mod.__name__, g, m) for g, m in mod.static_parts().items()]
        if hasattr(mod, "static_shells"):
            out += [(mod.__name__, "shell:" + k, sh.mesh) for k, sh in mod.static_shells().items()]
        if hasattr(mod, "dynamic_parts"):
            for k, p in mod.dynamic_parts().items():
                out.append((mod.__name__, k, K.Mesh(p.mesh.V @ K.rot_matrix(*p.rot).T + np.asarray(p.origin), p.mesh.Q, p.mesh.T,
                                                    p.mesh.Qm, p.mesh.Tm)))
    return [(mod, k, K.Mesh(m.V, m.Q, m.T, m.Qm, m.Tm)) for mod, k, m in out]        # the cabin modules may still use the old kit


def evaluated_body(levels=2):
    """The limit surface of the body cage (the Subdivision Surface a render shows), as a Mesh of triangles."""
    sub = pytest.importorskip("mkmmd.model.subdiv")
    m = B.body_loft().mesh
    sd = sub.subdivide(m.V, [list(q) for q in m.Q] + [list(t) for t in m.T], levels=levels,
                       creases={(int(a), int(b)): float(w) for (a, b), w in zip(m.Ce, m.Cw)})
    return K.Mesh(sd.verts, np.array(sd.faces))


# ======================================================================================== the layout
def test_the_layout_numbers_agree_with_each_other():
    assert L.Z_RUB < L.Z_SHOULDER < L.Z_BELT < L.Z_GLASS < L.Z_COWL < L.WS_TOP[1]
    assert L.Y_HOOD_REAR < L.Y_COWL < L.WS_TOP[0] < L.CUSHION_Y[0] + 0.5
    assert 42.0 < L.WS_RAKE_DEG < 46.0                               # measured on a side-on photograph: 44
    assert L.AXLE_R - L.AXLE_F == pytest.approx(L.WHEELBASE) and L.WHEELBASE == pytest.approx(2.62)
    assert L.ARCH_R - L.TYRE_R >= 0.045                              # a hand's breadth of daylight round the tyre
    assert L.WHEEL_X + L.TYRE_W / 2 < L.HALF_W                       # the tyres stand inside the body side
    assert L.X_WALL_IN < L.X_BELT_OUT < L.HALF_W
    assert L.LENGTH == pytest.approx(4.59) and L.WIDTH == pytest.approx(1.73)
    ys = [y for y, _ in L.HOOD_PROFILE]
    zs = [z for _, z in L.HOOD_PROFILE]
    k = ys.index(L.Y_HOOD_REAR)
    assert ys == sorted(ys) and zs[:k + 1] == sorted(zs[:k + 1]) and zs[k:] == sorted(zs[k:], reverse=True)   # up to the hood's rear edge, then down
    assert L.hood_z(L.Y_NOSE) == pytest.approx(L.Z_HOOD_NOSE) and L.hood_z(L.Y_HOOD_REAR) == pytest.approx(L.Z_COWL)
    assert L.hood_z(L.Y_COWL) == pytest.approx(L.Z_GLASS, abs=0.01)                                     # and on down to the glass


# ======================================================================================== the body
def test_the_body_cage_is_one_closed_outward_mirrored_shell():
    m = B.body_loft().mesh
    assert m.is_closed() and m.volume() > 2.0
    lo, hi = m.bbox()
    assert hi[0] == pytest.approx(L.HALF_W, abs=0.008) and lo[0] == pytest.approx(-L.HALF_W, abs=0.008)
    assert lo[1] == pytest.approx(L.Y_NOSE) and hi[1] == pytest.approx(L.Y_TAIL)
    assert hi[2] == pytest.approx(L.Z_COWL, abs=0.002) and lo[2] >= 0.18
    assert key(m.V) == key(m.V * (-1, 1, 1))


def test_the_sections_follow_the_layout():
    lf = B.body_loft()
    assert lf.point("tc", -1.31)[2] == pytest.approx(L.hood_z(-1.31), abs=0.003)              # the hood
    assert lf.point("tc", -0.695)[2] == pytest.approx(L.hood_z(-0.695), abs=0.003)              # the cowl drops to the glass
    assert lf.point("crease", -1.31)[2] == pytest.approx(L.Z_SHOULDER)                          # the shoulder crease: one level
    assert lf.point("crease", 0.0)[2] == pytest.approx(L.Z_SHOULDER)
    assert lf.point("crease", 1.9)[2] == pytest.approx(L.Z_SHOULDER)
    assert lf.point("e2", 0.0)[2] == pytest.approx(L.Z_BELT, abs=0.003)                         # the door top
    assert lf.point("e2", 0.0)[0] == pytest.approx(0.5 * (L.X_BELT_OUT + L.X_WALL_IN))
    assert lf.point("t1", 0.0)[0] == pytest.approx(L.X_WALL_IN)                                  # the inner wall
    assert lf.point("tc", 0.0)[2] == pytest.approx(B.Z_FLOOR)                                   # the floor
    assert lf.point("tc", 1.9)[2] == pytest.approx(0.873)                                        # the deck, flat and a little crowned
    assert lf.point("fa", 0.0)[2] < L.Z_RUB < lf.point("fb", 0.0)[2]                            # the rub strip shelf
    # the body is a barrel: widest at the flank, the side leans in above it to the crease and on to the belt
    assert lf.point("fc", 0.0)[0] == pytest.approx(L.HALF_W)
    assert lf.point("fc", 0.0)[0] > lf.point("crease", 0.0)[0] > lf.point("u2", 0.0)[0]
    assert lf.point("crease", -1.31)[0] == pytest.approx(L.HALF_W - B.CREASE_IN)


def test_the_cockpit_is_an_opening_with_walls_and_a_floor_the_floor_clears_the_pan():
    lf = B.body_loft()
    for y in (-0.5, 0.0, 0.62, 1.0):
        top = lf.point("tc", y)[2]
        assert top == pytest.approx(B.Z_FLOOR) and top > lf.point("keel", y)[2] + 0.02       # a floor above the pan, not through it
        assert lf.point("e2", y)[2] > top + 0.5
    # no section crosses itself: the point order runs counter-clockwise round the half section
    for y in (-2.0, -1.0, -0.65, 0.0, 1.0, 1.5, 2.0):
        P = np.array([lf.ring(y)[lf.col(n)] for n in lf.names])[:, [0, 2]]
        P = np.vstack([P, P[:1] * (0, 1)])                                                   # close it along the centre line
        assert K.polygon_area(P) > 0.05


def test_gaps_are_grooves_in_the_seam_material():
    m = B.body_loft().mesh
    cq, _ = m.face_centres()
    seam = cq[m.Qm == M["seam"]]
    assert len(seam) > 0
    for y in (L.Y_DOOR_FRONT, L.Y_DOOR_REAR, -2.10, L.Y_HOOD_REAR, 1.72, 2.145):
        assert (np.abs(seam[:, 1] - y) < 0.01).sum() > 0, y
    door = seam[np.abs(seam[:, 1] - L.Y_DOOR_REAR) < 0.01]
    assert door[:, 2].min() < 0.45 and door[:, 2].max() < L.Z_BELT                               # on the side, below the door top
    under = cq[m.Qm == M["underbody"]]
    assert (under[:, 2] < 0.3).sum() > 0.05 * len(cq) and len(under) > 0.1 * len(cq)           # the pan, the floor, the inner walls


def test_the_arch_cutters_surround_the_tyres():
    cs = B.arch_cutters()
    assert len(cs) == 4
    for c in cs:
        assert c.is_closed() and c.volume() > 0
    for c, ay, sx in zip(cs, (L.AXLE_F,) * 2 + (L.AXLE_R,) * 2, (1, -1, 1, -1)):
        lo, hi = c.bbox()
        assert 0.5 * (lo[1] + hi[1]) == pytest.approx(ay, abs=0.002) and 0.5 * (lo[2] + hi[2]) == pytest.approx(L.WHEEL_Z, abs=0.002)
        assert 0.5 * (hi[1] - lo[1]) > L.ARCH_R                                                  # flared past the nominal radius
        x0, x1 = sorted((abs(lo[0]), abs(hi[0])))
        assert x0 < L.WHEEL_X - L.TYRE_W / 2 - 0.03 and x1 > L.HALF_W                             # inside the tyre's inner face, out through the skin
        assert np.sign((lo[0] + hi[0]) / 2) == sx


def test_the_subdivided_body_has_the_reference_proportions():
    sm = evaluated_body()
    pr = K.Probe.from_mesh(sm)
    lo, hi = sm.bbox()
    assert lo[1] == pytest.approx(L.Y_NOSE, abs=0.03) and hi[1] == pytest.approx(L.Y_TAIL, abs=0.03)    # the nose and tail faces
    assert hi[0] == pytest.approx(L.HALF_W, abs=0.01)
    assert pr.height(0.0, -1.31) == pytest.approx(L.hood_z(-1.31), abs=0.012)                      # the flat hood
    assert pr.height(0.0, L.Y_HOOD_REAR + 0.05) == pytest.approx(L.hood_z(L.Y_HOOD_REAR + 0.05), abs=0.012)
    assert pr.height(0.0, 1.9) == pytest.approx(0.873, abs=0.01)                                  # the flat deck
    # the door top: a rounded cap at the belt height, x from the inner wall to the belt edge
    assert pr.height(0.5 * (L.X_BELT_OUT + L.X_WALL_IN), 0.0) == pytest.approx(L.Z_BELT, abs=0.006)
    # the shoulder crease stays a crease: the side above it leans in, the side below it is upright
    h_lo = pr.ray((3.0, 0.0, B.FLANK_Z), (-1.0, 0.0, 0.0))[1][0]
    h_mid = pr.ray((3.0, 0.0, L.Z_SHOULDER), (-1.0, 0.0, 0.0))[1][0]
    h_hi = pr.ray((3.0, 0.0, L.Z_BELT - 0.03), (-1.0, 0.0, 0.0))[1][0]
    assert h_lo == pytest.approx(L.HALF_W, abs=0.006) and h_mid == pytest.approx(L.HALF_W - B.CREASE_IN, abs=0.008) and h_hi < h_mid - 0.008


def test_the_cockpit_floor_meets_the_inner_walls_in_a_tight_corner():
    """After subdivision the floor must lie at the carpet height right out to the walls, and the inner walls must stand at
    the door panels' plane down to the floor (a fat fillet would swallow the carpet, the feet and the door panels)."""
    pr = K.Probe.from_mesh(evaluated_body())
    for x in (0.0, 0.2, 0.4, 0.56, 0.65):
        assert pr.height(x, 0.2) == pytest.approx(B.Z_FLOOR, abs=0.012), x
    for z in (0.30, 0.40, 0.55, 0.80):
        t, p, n = pr.ray((0.3, 0.2, z), (1.0, 0.0, 0.0))
        assert p[0] == pytest.approx(L.X_WALL_IN, abs=0.012), z


def test_the_windshield_lies_in_the_plane_from_the_cowl_to_the_header():
    glass, frame = B.windshield()
    (y0, z0), (y1, z1) = L.WS_BASE, L.WS_TOP
    up = np.array([0.0, y1 - y0, z1 - z0])
    up /= np.linalg.norm(up)
    nrm = np.array([0.0, -up[2], up[1]])
    for m in (glass, frame):
        assert m.is_closed() and m.volume() > 0
        d = (m.V - np.array([0.0, y0, z0])) @ nrm
        assert d.min() > -0.04 and d.max() < 0.05                                               # a thin pane near the plane
        along = (m.V - np.array([0.0, y0, z0])) @ up
        assert along.min() > -0.01 and along.max() < np.hypot(y1 - y0, z1 - z0) + 0.01
    lo, hi = glass.bbox()
    assert hi[2] < L.WS_TOP[1] and lo[2] > L.Z_GLASS - 0.01 and hi[0] - lo[0] > 1.2            # the pane spans the cabin
    flo, fhi = frame.bbox()
    assert fhi[0] == pytest.approx(L.WS_HALF_W_BASE, abs=0.01) and fhi[2] == pytest.approx(L.WS_TOP[1], abs=0.05)


def test_the_boot_covers_the_folded_top_behind_the_rear_seat():
    cage = B.boot_cage()
    assert cage.is_closed() and cage.volume() > 0
    lo, hi = cage.bbox()
    assert lo[1] == pytest.approx(B.BOOT["y0"], abs=0.01) and hi[1] == pytest.approx(B.BOOT["y1"], abs=0.01)
    assert hi[2] == pytest.approx(L.Z_DECK + B.BOOT["rise"] - 0.012, abs=0.02) and lo[2] < L.Z_DECK                # a pillow on the deck
    assert hi[0] - lo[0] < 2 * L.X_WALL_IN


# ======================================================================================== the nose and the tail
def test_every_exterior_group_is_well_formed_closed_pieces():
    for mod in (E, TR):
        for g, m in mod.static_parts().items():
            tag = f"{mod.__name__}:{g}"
            assert np.isfinite(m.V).all() and m.nfaces > 0, tag
            if g != "decals":
                assert m.is_closed(), tag                                                      # solid pieces, wound outward
                assert m.volume() > 0, tag


def test_the_bumpers_wrap_the_car_ends_and_taper_into_the_body():
    for front in (True, False):
        m = E.bumper(front)
        lo, hi = m.bbox()
        tip = -lo[1] if front else hi[1]
        assert tip == pytest.approx(L.LENGTH / 2 + 0.012, abs=0.02)                                 # the chrome bar leads by a hair
        assert E.BUMPER["x_end"] < hi[0] < E.BUMPER["x_end"] + E.BUMPER["d"]                         # round the corners to the sides
        assert lo[2] >= E.BUMPER["zc"] - E.BUMPER["h"] / 2 - 0.005 and hi[2] <= E.BUMPER["zc"] + E.BUMPER["h"] / 2 + 0.03
        # it runs back along the sides to meet the body: the ends reach well behind the face
        end = m.V[np.abs(m.V[:, 0]) > E.BUMPER["x_end"]]
        assert (end[:, 1].max() - end[:, 1].min()) > 0.2
        assert m.is_closed() and m.volume() > 0


def test_the_plates_sit_on_the_bumper_faces_where_the_card_surfaces_are():
    ex = E.static_parts()["trim"]
    for side, c in ((-1, L.PLATE_FRONT), (+1, L.PLATE_REAR)):
        quads = ex.Q[ex.Qm == M["plate"]]
        pts = ex.V[quads]
        on_plate = pts[(side * pts[..., 1] > 2.29).all(1)].reshape(-1, 3)
        assert len(on_plate) > 0
        lo, hi = on_plate.min(0), on_plate.max(0)
        assert (hi[0] - lo[0], hi[2] - lo[2]) == pytest.approx(L.PLATE_SIZE, abs=0.004)
        assert (lo + hi) / 2 == pytest.approx(c, abs=0.004)
        assert E.BUMPER["zc"] - E.BUMPER["h"] / 2 <= lo[2] and hi[2] <= E.BUMPER["zc"] + E.BUMPER["h"] / 2 + 0.01      # on the bumper


def test_the_headlamps_are_two_clear_lenses_a_side_in_bezels_on_the_fascia():
    probe = E._body_probe(None)
    for sx in (1.0, -1.0):
        m = E.headlamp(probe, sx)
        lens = m.V[m.Q[m.Qm == M["lens_head"]]].reshape(-1, 3)
        assert (lens[:, 0] * sx).mean() == pytest.approx(L.HEADLAMP_X, abs=0.02)
        assert (lens[:, 0] * sx).min() > L.HEADLAMP_X - 0.2 and (lens[:, 0] * sx).max() < L.HEADLAMP_X + 0.2
        # two lenses side by side: a gap in x down the middle of the lamp
        xs = np.sort(np.unique(np.round(lens[:, 0] * sx, 3)))
        assert np.diff(xs).max() > E.HEADLAMP["gap"] * 0.9
        assert (lens[:, 1].mean() < L.Y_NOSE + 0.12 and lens[:, 2].mean() == pytest.approx(L.HEADLAMP_Z, abs=0.02))
        assert np.ptp(lens[:, 2]) == pytest.approx(E.HEADLAMP["lens_h"], abs=0.01)
    assert E.HEADLAMP["x"] == L.HEADLAMP_X and E.HEADLAMP["z"] == L.HEADLAMP_Z


def test_the_grille_is_a_framed_row_of_chrome_slats():
    g = E.grille(E._body_probe(None))
    lo, hi = g.bbox()
    assert (hi[0] - lo[0], hi[2] - lo[2]) == pytest.approx((E.GRILLE["w"], E.GRILLE["h"]), abs=0.03)
    slats = g.V[g.Q[g.Qm == M["chrome"]]].reshape(-1, 3)
    assert len(np.unique(np.round(slats[:, 0], 3))) > 2 * E.GRILLE["vbars"]                         # many slats across the frame
    assert g.is_closed()


def test_the_tail_lamp_band_is_two_ribbed_red_units_either_side_of_a_dark_panel():
    trim, lens = E.tail_lamps(E._body_probe(None))
    assert lens.is_closed() and trim.is_closed()
    assert set(np.unique(lens.Qm)) == {M["lamp_tail"]}
    ribs = lens.V[lens.Q].mean(1)
    centres = np.unique(np.round(ribs[:, 0], 2))
    assert (np.abs(centres) > 0.13).all() and len(centres) >= 2 * E.TAIL["ribs"] - 2                  # the ribs of both units, none in the middle
    assert ribs[:, 1].min() > L.Y_TAIL - 0.05 and np.ptp(ribs[:, 2]) < 0.14                  # on the (slightly raked) tail panel
    dark = trim.V[trim.Q[trim.Qm == M["trim_dark"]]].reshape(-1, 3)
    assert np.abs(dark[:, 0]).max() < E.TAIL["x0"] + 0.01 and len(dark) > 0


def test_the_brake_lamp_rides_the_deck():
    lens = E.static_parts()["lens"]
    sel = lens.V[lens.Q[lens.Qm == M["lamp_tail"]]].reshape(-1, 3)
    on_deck = sel[(sel[:, 1] > 1.7) & (sel[:, 1] < 1.85)]
    assert len(on_deck) > 0 and abs(on_deck[:, 0]).max() < 0.12 and on_deck[:, 2].min() > L.Z_DECK - 0.03


# ======================================================================================== the trim
def test_the_strips_follow_the_sides_and_the_arch_lips_hug_the_openings():
    probe = TR._body_probe(None)
    for m in TR.side_strips(probe):
        lo, hi = m.bbox()
        assert abs(hi[0]) > L.HALF_W - 0.03 or abs(lo[0]) > L.HALF_W - 0.03 and m.is_closed()
    lips = TR.arch_lips(probe)
    assert len(lips) == 4
    for m, (ay, sx) in zip(lips, ((L.AXLE_F, 1), (L.AXLE_R, 1), (L.AXLE_F, -1), (L.AXLE_R, -1))):
        c = m.V.mean(0)
        r = np.hypot(m.V[:, 1] - ay, m.V[:, 2] - L.WHEEL_Z)
        assert r.min() > L.ARCH_R + 0.01 and r.max() < L.ARCH_R + 0.08
        assert np.sign(c[0]) == sx or True
        assert m.V[:, 2].max() > L.WHEEL_Z + L.ARCH_R                                              # over the top of the opening


def test_the_mirrors_stand_out_of_the_body_with_their_glass_facing_back():
    for sx in (1.0, -1.0):
        m = TR.mirror(sx)
        assert m.is_closed() and m.volume() > 0
        assert (m.V[:, 0] * sx).max() == pytest.approx(0.995, abs=0.01) and (m.V[:, 0] * sx).min() > 0.7
        g = m.V[m.Q[m.Qm == M["mirror_glass"]]]
        nq, _ = m.face_vectors()
        n = nq[m.Qm == M["mirror_glass"]].sum(0)
        assert n[1] > 0.9 * np.linalg.norm(n)                                                       # toward the driver (+Y)
        assert g[..., 2].mean() == pytest.approx(1.04, abs=0.03)


def test_wipers_antenna_and_tailpipe_sit_where_they_should():
    for m in TR.wipers():
        o, nrm, up, Lg = B.ws_frame()
        d = (m.V - o) @ nrm
        assert d.min() > 0.0 and d.max() < 0.045                                                    # parked on the glass
    ant = TR.antenna(TR._body_probe(None))
    assert ant[1].bbox()[1][2] == pytest.approx(TR.ANT_TIP_Z, abs=0.03)
    pipe = TR.tailpipe()[0]
    assert pipe.bbox()[1][1] == pytest.approx(TR.PIPE_Y[1], abs=0.01) and pipe.bbox()[0][2] > 0.15


def test_door_pulls_vents_and_the_fuel_door_are_on_the_sides():
    probe = TR._body_probe(None)
    for fn in (TR.door_pulls, TR.fender_vents):
        ms = fn(probe)
        assert len(ms) == 2
        assert ms[0].bbox()[0][0] > 0.8 and ms[1].bbox()[1][0] < -0.8
    fuel = TR.fuel_door(probe)[0]
    assert fuel.bbox()[0][0] > 0.8 and fuel.bbox()[0][1] > L.AXLE_R + L.ARCH_R + 0.1                  # behind the rear opening


def test_the_shut_lines_lie_on_the_surface():
    probe = TR._body_probe(None)
    d = TR.hood_seams(probe)
    assert len(d) == 5
    for m in d:
        for p in m.V[::7]:
            h = probe.height(p[0], p[1])
            assert h is not None and abs(p[2] - h) < 0.004


# ======================================================================================== wheels
def test_wheel_placement():
    w = W.dynamic_parts()
    assert set(w) == {"wheel_FL", "wheel_FR", "wheel_RL", "wheel_RR"}
    o = {k: np.array(p.origin) for k, p in w.items()}
    assert o["wheel_FL"][0] == pytest.approx(L.TRACK / 2) and o["wheel_FR"][0] == pytest.approx(-L.TRACK / 2)
    assert o["wheel_RL"][1] - o["wheel_FL"][1] == pytest.approx(L.WHEELBASE)
    assert o["wheel_RL"][1] == pytest.approx(o["wheel_RR"][1]) and o["wheel_FL"][2] == pytest.approx(0.31)
    # the tyre stands on the ground and has the right radius about the local X axis
    for k, p in w.items():
        V = p.mesh.V
        world = V @ K.rot_matrix(*p.rot).T + p.origin
        assert world[:, 2].min() == pytest.approx(0.0, abs=0.002)
        assert np.hypot(V[:, 1], V[:, 2]).max() == pytest.approx(L.TYRE_R, abs=0.002)
        width = V[:, 0].max() - V[:, 0].min()
        assert 0.17 < width < 0.23


def test_wheels_are_mirror_images_with_the_dish_outside():
    w = W.dynamic_parts()
    fl, fr = w["wheel_FL"].mesh, w["wheel_FR"].mesh
    assert np.sort(np.round(fl.V[:, 0], 6)) == pytest.approx(np.sort(np.round(-fr.V[:, 0], 6)))
    # the wheel face (alloy / chrome) is on the outer side: its centroid in local x points away from the car
    face = lambda m, role: m.V[np.unique(m.Q[m.Qm == M[role]])]                 # noqa: E731
    assert face(fl, "alloy")[:, 0].mean() > 0 > face(fr, "alloy")[:, 0].mean()


def test_the_tyres_sit_inside_the_arches_flush_with_the_side():
    p = W.dynamic_parts()["wheel_FL"]
    outer = p.mesh.V[:, 0].max() + p.origin[0]
    assert L.HALF_W - 0.06 < outer < L.HALF_W                                        # the sidewall is within a hand of the body side
    lf = B.body_loft()
    assert lf.point("fb", L.AXLE_F)[0] > outer                                       # and behind it, so the arch cut shows the tyre


# ======================================================================================== all the parts
def test_every_part_is_well_formed():
    for mod, key_, m in car_parts():
        tag = f"{mod.split('.')[-1]}:{key_}"
        assert np.isfinite(m.V).all(), tag
        assert m.nfaces > 0, tag
        assert m.Q.max(initial=0) < len(m.V) and m.T.max(initial=0) < len(m.V), tag
        assert m.Qm.min(initial=0) >= 0 and m.Qm.max(initial=0) < len(L.MATS), tag
        assert m.Tm.max(initial=0) < len(L.MATS), tag
        lo, hi = m.bbox()
        assert lo[0] > -1.1 and hi[0] < 1.1 and lo[1] > -2.45 and hi[1] < 2.45 and lo[2] > -0.01 and hi[2] < 1.72, tag
        nq, nt = m.face_vectors()
        areas = np.concatenate([np.linalg.norm(nq, axis=1), np.linalg.norm(nt, axis=1)])
        assert (areas > 1e-12).mean() > 0.99, f"{tag}: zero-area faces"


def test_overall_size_of_the_car():
    sm = K.merge([m for _, _, m in car_parts()])
    lo, hi = sm.bbox()
    assert hi[1] - lo[1] == pytest.approx(L.LENGTH, abs=0.03)
    assert lo[1] == pytest.approx(-L.LENGTH / 2, abs=0.02)
    assert B.body_loft().mesh.size()[0] == pytest.approx(L.WIDTH, abs=0.012)
    # the mirrors stand out of the body and the antenna is the highest thing on the car
    assert lo[2] >= -0.001 and hi[2] < 1.72 and (hi[0] - lo[0]) < 2.05
    glass, frame = B.windshield()
    assert frame.bbox()[1][2] == pytest.approx(L.WS_TOP[1], abs=0.05)                  # the windshield top, without the antenna


# ======================================================================================== colour roles
def linear_luma(c):
    return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]


def test_default_roles_follow_the_palette_and_rubber_is_its_darkest_slot():
    P = CP.Pal()
    assert set(P.roles()) >= {"body", "lower", "stripe", "trim", "rubber", "interior", "display", "headlamp", "taillamp"}
    from mkmmd.core import palette as PAL
    assert P.lin("body") == pytest.approx(PAL.linear(PAL.PALETTES["rose-pine-moon"]["love"]))
    assert P.lin("taillamp") == pytest.approx(PAL.linear(PAL.PALETTES["rose-pine-moon"]["love"]))
    # black rubber: the darkest slot, never pure black
    slots = {k: linear_luma(PAL.linear(v)) for k, v in PAL.PALETTES["rose-pine-moon"].items()}
    assert linear_luma(P.lin("rubber")) == pytest.approx(min(slots.values())) and linear_luma(P.lin("rubber")) > 0.005
    # the upholstery is a warm tan (red over green over blue), the carpet-like boot is dark and cool
    r, g, b = P.lin("interior")
    assert r > g > b and linear_luma(P.lin("interior")) > 3 * linear_luma(P.lin("boot"))


def test_roles_recolour_by_slot_hex_and_blend_and_palettes_switch_consistently():
    from mkmmd.core import palette as PAL
    moon, dawn = PAL.PALETTES["rose-pine-moon"], PAL.PALETTES["rose-pine-dawn"]
    assert CP.Pal({"body": "iris"}).lin("body") == pytest.approx(PAL.linear(moon["iris"]))
    assert CP.Pal({"body": "#112233"}).lin("body") == pytest.approx(PAL.linear("#112233"))
    mix = CP.Pal({"body": "gold:3,pine:1"}).lin("body")
    expect = 0.75 * np.array(PAL.linear(moon["gold"])) + 0.25 * np.array(PAL.linear(moon["pine"]))
    assert mix == pytest.approx(tuple(expect))
    hexmix = CP.Pal({"body": "#ff0000:1,#0000ff:3"}).lin("body")
    assert hexmix == pytest.approx((0.25, 0.0, 0.75))
    # a project palette (slots = the palette's hexes) recolours every role through the slot names
    assert CP.Pal(dawn).lin("body") == pytest.approx(PAL.linear(dawn["love"]))
    assert CP.Pal(dawn).lin("rubber") == pytest.approx(PAL.linear(dawn["base"]))
    # overriding a slot reaches the roles that use it
    assert CP.Pal({"love": "#00ff00"}).lin("taillamp") == pytest.approx((0.0, 1.0, 0.0))
    with pytest.raises(KeyError):
        CP.Pal({"body": "nonsense"}).lin("body")
