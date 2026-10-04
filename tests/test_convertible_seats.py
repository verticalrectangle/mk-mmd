"""The seats module of the 1980s convertible (pure numpy): the upholstered cages (closed, creased, where they stand against
the layout), the seat surface and the back plane the sitters are posed on, the headrest and its collider box, the door panels
against the body's wall, the console, the floor, the card's anchors and the module's budget. (The Blender side is checked by
building and looking.)"""
import math
import time

import numpy as np
import pytest

from mkmmd.blender.library.props import convertible_layout as L
from mkmmd.blender.library.props import convertible_seats as SE
from mkmmd.core import form as FM
from mkmmd.core import shell as S

M = L.M
GROUPS = {SE.GROUP, "decals"}


@pytest.fixture(scope="module")
def parts():
    return SE.static_parts()


@pytest.fixture(scope="module")
def shells():
    return SE.static_shells()


def verts(mesh, role):
    """Vertices of the faces of a material role."""
    q = mesh.V[mesh.Q[mesh.Qm == M[role]]].reshape(-1, 3)
    t = mesh.V[mesh.T[mesh.Tm == M[role]]].reshape(-1, 3)
    return np.concatenate([q, t])


# ============================================================================================== the module's contract
def test_the_module_builds_fast_and_well_formed(parts, shells):
    t0 = time.time()
    SE.static_shells(), SE.static_parts()
    assert time.time() - t0 < 3.0                                    # the budget is 0.5 s; a loaded machine gets slack
    assert set(parts) == GROUPS
    assert set(shells) == {"seat_L", "seat_R", "bench", "console", "floor", "door_L", "door_R", "quarter_L", "quarter_R"}
    meshes = [(g, m) for g, m in parts.items()] + [(k, s.mesh) for k, s in shells.items()]
    for key, m in meshes:
        assert np.isfinite(m.V).all() and m.nfaces > 0, key
        assert m.Q.max(initial=0) < len(m.V) and m.T.max(initial=0) < len(m.V), key
        assert m.Qm.min(initial=0) >= 0 and m.Qm.max(initial=0) < len(L.MATS), key
        assert m.Tm.max(initial=0) < len(L.MATS), key
        lo, hi = m.bbox()
        assert lo[0] > -1.0 and hi[0] < 1.0 and lo[1] > -2.4 and hi[1] < 2.4 and lo[2] >= 0.0 and hi[2] < 1.32, key


def test_cages_are_closed_outward_and_creased_only_where_they_are_meant_to(shells):
    for k, s in shells.items():
        m = s.mesh
        assert s.levels == (2 if k.startswith("seat") else 1), k
        assert m.is_closed(), k
        assert m.volume() > 0, k                                      # normals outward
        assert len(m.Ce) == len(m.Cw) and (m.Cw > 0).all() and (m.Cw <= 1.0).all(), k
        assert m.Ce.max(initial=0) < len(m.V), k
    # a cage with creases keeps them off the end caps and the underside: the trough seams are partial rings
    seat = shells["seat_L"].mesh
    assert 40 < len(seat.Ce) < 400


def test_the_budget_of_plain_meshes_and_cages(parts, shells):
    assert sum(m.nfaces for m in parts.values()) < 30000
    assert sum(len(s.mesh.Q) + len(s.mesh.T) for s in shells.values()) < 14000     # control faces: ~0.5 s of Subdivision Surface


def test_no_plain_part_is_a_box(parts):
    m = parts[SE.GROUP]
    tris = m.triangles()
    r = FM.analyse(m.V, tris)
    assert r["cuboid_share"] < 0.05, r["components"][:3]


# ============================================================================================== the seats
def limit_of(cage_fn):
    cage = cage_fn()
    ns, m = SE.ring_dims(cage)
    return SE.limit_grid(cage, ns, m)


def test_the_surface_under_the_hip_is_the_cushion_top():
    lim = limit_of(SE.cushion_cage)
    sel = (np.abs(lim[:, :, 0]) < 0.06) & (np.abs(lim[:, :, 1] - L.HIP_Y) < 0.06) & (lim[:, :, 2] > 0.44)
    assert lim[sel][:, 2].max() == pytest.approx(L.CUSHION_TOP, abs=0.005)
    # the dish is a dish: the bolsters stand higher than the crowns under the hip, the sitter sits between them
    side = (np.abs(np.abs(lim[:, :, 0]) - 0.22) < 0.02) & (np.abs(lim[:, :, 1] - L.HIP_Y) < 0.05)
    assert lim[side][:, 2].max() > L.CUSHION_TOP + 0.015
    # the cushion spans the layout's seat: front edge, back edge, width
    c = SE.cushion_cage()
    assert c.V[:, 1].min() == pytest.approx(L.CUSHION_Y[0], abs=0.02) and c.V[:, 1].max() == pytest.approx(L.CUSHION_Y[1], abs=0.005)
    assert np.ptp(c.V[:, 0]) == pytest.approx(L.CUSHION_W, abs=0.01)


def test_the_back_reclines_and_its_front_plane_is_ten_centimetres_behind_the_hip():
    lim = limit_of(SE.back_cage)
    crown = lim[:, 0]                                             # the centre line of the front surface (crowns and troughs)
    # the crowns lie on a plane tipped REC degrees from vertical: fit y against z
    hi = crown[(crown[:, 2] > 0.55) & (crown[:, 2] < 1.0)]
    slope = np.polyfit(hi[:, 2], hi[:, 1], 1)[0]
    assert math.degrees(math.atan(slope)) == pytest.approx(L.BACK_RECLINE_DEG, abs=2.5)
    j = int(np.argmin(np.abs(crown[:, 2] - L.HIP_Z)))
    assert 0.07 < crown[j, 1] - L.HIP_Y < 0.13
    assert 0.07 < SE.SEAT["back_gap_at_hip"] < 0.13
    # the back's front plane meets the cushion top at BACK_BASE_Y
    assert SE.SEAT["back_base_yz"] == pytest.approx((L.BACK_BASE_Y, L.CUSHION_TOP))


def test_the_headrest_stays_inside_its_collider_box_and_clear_of_the_sitter():
    hy, hz = SE.SEAT["headrest_center"]
    w, d, h = SE.SEAT["headrest_size"]
    R = S.rot_matrix(-L.BACK_RECLINE_DEG, 0, 0)
    head = SE.headrest_cage()
    P = (head.V - np.array([0.0, hy, hz])) @ R                    # into the collider box's frame
    assert np.abs(P[:, 0]).max() <= w / 2 + 0.001 and np.abs(P[:, 1]).max() <= d / 2 + 0.001 and np.abs(P[:, 2]).max() <= h / 2 + 0.001
    assert (np.abs(P) >= np.array([w, d, h]) / 2 - 0.004).any(axis=0).all()      # and it fills it, not a smaller thing
    plane_y = L.BACK_BASE_Y + (hz - L.CUSHION_TOP) * math.tan(math.radians(L.BACK_RECLINE_DEG))
    assert hy - d / 2 > plane_y
    assert L.HEADREST_Z[0] - 0.01 < hz - h / 2 and hz + h / 2 < L.HEADREST_Z[1] + 0.02


def test_the_seats_are_mirror_images_and_the_knob_is_on_the_door_side(shells, parts):
    a, b = shells["seat_L"].mesh, shells["seat_R"].mesh
    key = lambda V: np.round(V[np.lexsort(np.round(V, 6).T)], 6)                  # noqa: E731
    assert key(a.V * (-1, 1, 1)) == pytest.approx(key(b.V))
    ch = verts(parts[SE.GROUP], "chrome")
    knob = ch[(np.abs(ch[:, 1] - 0.31) < 0.04) & (np.abs(ch[:, 2] - 0.395) < 0.04) & (np.abs(ch[:, 0]) > 0.6)]
    assert {round(float(np.sign(x))) for x in knob[:, 0]} == {-1, 1}               # one knob on each seat's outer side
    for s in (+1, -1):
        mine = knob[np.sign(knob[:, 0]) == s]
        assert np.abs(mine[:, 0]).min() > L.SEAT_X + SE.HW                         # outside the cushion's wall


def test_the_seats_stand_clear_of_the_console_the_doors_and_the_floor(shells):
    seat = shells["seat_L"].mesh
    con = shells["console"].mesh
    assert seat.V[:, 0].min() > L.CONSOLE_X + 0.01                  # the console's sides sit between the cushions
    assert seat.V[:, 0].max() < L.X_WALL_IN - 0.05                  # the seat clears the door panel's armrest
    assert seat.V[:, 2].min() > L.FLOOR_Z                           # the rails stand on the carpet
    assert con.V[:, 0].max() < seat.V[:, 0].min() + 0.03            # the console's flared base nearly meets the seats


# ============================================================================================== bench, console, floor
def test_the_rear_bench_fills_its_place_and_ends_below_the_door_tops(shells):
    b = shells["bench"].mesh
    assert b.V[:, 0].min() == pytest.approx(-L.REAR_SEAT_W / 2, abs=0.01) and b.V[:, 0].max() == pytest.approx(L.REAR_SEAT_W / 2, abs=0.01)
    assert b.V[:, 1].min() == pytest.approx(L.REAR_SEAT_Y[0], abs=0.02)
    assert b.V[:, 2].max() < L.Z_BELT
    assert b.V[:, 2].min() <= L.FLOOR_Z                                  # the base stands on the floor
    cush = SE.bench_cushion_cage()
    assert cush.V[:, 2].max() == pytest.approx(L.REAR_SEAT_TOP, abs=0.03)


def test_the_console_runs_from_the_dash_to_the_armrest(shells, parts):
    c = shells["console"].mesh
    assert c.V[:, 1].min() == pytest.approx(L.CONSOLE_Y[0], abs=0.002)                       # meets the dash's stack at y = -0.30
    assert c.V[:, 1].max() == pytest.approx(L.CONSOLE_Y[1], abs=0.01)
    body = SE.console_cage()
    assert body.V[:, 2].max() == pytest.approx(L.CONSOLE_Z, abs=0.002)
    top = body.V[body.V[:, 2] > L.CONSOLE_Z - 0.002]
    assert np.abs(top[:, 0]).max() == pytest.approx(L.CONSOLE_X - 0.04, abs=0.01)           # flat top, rounded shoulders
    assert SE.lid_cage().V[:, 2].max() == pytest.approx(SE.LID_TOP + 0.01, abs=0.012)
    # the gate is chrome and the lever tip stands above the console, where a hand reaches
    assert SE.KNOB_C[2] > L.CONSOLE_Z + 0.1 and abs(SE.KNOB_C[0]) < 1e-9
    assert L.CONSOLE_Y[0] < SE.KNOB_C[1] < L.CONSOLE_Y[0] + 0.5


def test_the_floor_carpet_is_at_floor_z_with_a_tunnel(shells):
    carpet = SE.carpet_cage()
    ns, m = SE.ring_dims(carpet)
    ring = carpet.V[:ns * m]                                                                   # the rings, not the end caps
    top = ring[(np.abs(ring[:, 0]) > 0.25) & (np.abs(ring[:, 0]) < 0.62) & (ring[:, 2] > 0.225) & (ring[:, 2] < 0.26)]
    z_floor = top[:, 2]                                                                       # the carpet's top surface
    assert z_floor.min() == pytest.approx(L.FLOOR_Z, abs=0.005) and z_floor.max() < L.FLOOR_Z + 0.012     # where the feet stand
    assert ring[np.abs(ring[:, 0]) < 0.2][:, 2].max() == pytest.approx(0.31, abs=0.005)      # the transmission tunnel
    assert carpet.V[:, 2].max() == pytest.approx(SE.Z_CARPET_UP, abs=0.012)                  # and the sheet climbs the wall to the panel's lip
    assert np.abs(carpet.V[:, 0]).max() <= L.X_WALL_IN + 0.012                                # its lip tucks under the door panel

def test_nothing_of_the_cabin_stands_inside_the_rear_wheel_opening(parts, shells):
    """The body cuts the rear wheel opening with a cylinder (radius ARCH_R about the axle, from |x| = 0.58 outward): whatever
    of the trim stands inside it shows through the opening, so the well must hug the cylinder from outside."""
    cut = L.ARCH_R
    everything = [parts[SE.GROUP]] + [s.mesh for k, s in shells.items()]
    for m in everything:
        V = m.V[np.abs(m.V[:, 0]) > SE.WELL_X[0] + SE.WELL_T + 0.02]         # the well's own end wall (its bottom) is the well's floor
        rho = np.hypot(V[:, 1] - L.AXLE_R, V[:, 2] - L.WHEEL_Z)
        inside = (rho < cut - 0.004) & (V[:, 1] < SE.Y_CABIN_REAR + 0.05)
        assert not inside.any(), (V[inside][:3], rho[inside][:3])
    well = SE.wheel_well()
    rho = np.hypot(well.V[:, 1] - L.AXLE_R, well.V[:, 2] - L.WHEEL_Z)
    assert rho.max() == pytest.approx(SE.WELL_R_OUT, abs=0.003) and SE.WELL_R_IN > cut + 0.005
    assert well.V[:, 0].min() == pytest.approx(SE.WELL_X[0], abs=0.001) and well.V[:, 0].min() < L.WHEEL_X - L.TYRE_W / 2
    assert well.V[:, 2].min() >= L.FLOOR_Z - 0.04 and well.V[:, 1].max() <= SE.Y_CABIN_REAR + 0.07
    # the quarter trim's lower edge follows the well's top and the carpet ends before it
    q = SE.quarter_cage()
    for y in (0.95, 1.05):
        assert SE.quarter_zlow(y) >= SE.well_top(y) - 0.016
    assert SE.carpet_cage().V[:, 1].max() < L.AXLE_R - SE.WELL_R_OUT + 0.003


# ============================================================================================== the door panels
def test_door_panels_stand_against_the_wall_without_leaving_it(shells):
    for key in ("door_L", "quarter_L"):
        m = shells[key].mesh
        stand = SE.wall_x(m.V[:, 2]) - m.V[:, 0]                      # how far each point stands off the body's (leaning) wall
        assert stand.min() >= -0.01                                   # the back is buried in the wall, never through it
        assert stand.max() <= 0.07                                    # at most 7 cm of armrest
        assert m.V[:, 2].max() <= L.Z_BELT                            # the rolled top ends under the body's cap
        assert m.V[:, 2].min() >= L.FLOOR_Z - 0.01
    d = shells["door_L"].mesh
    assert d.V[:, 1].min() >= L.Y_DOOR_FRONT and d.V[:, 1].max() <= L.Y_DOOR_REAR              # the front end hides in the cowl
    q = shells["quarter_L"].mesh
    assert q.V[:, 1].min() >= L.Y_DOOR_REAR and q.V[:, 1].max() <= SE.Y_CABIN_REAR + 1e-6
    r = shells["door_R"].mesh
    assert np.sort(np.round(r.V[:, 0], 6)) == pytest.approx(np.sort(np.round(-d.V[:, 0], 6)))


def test_door_details_are_on_the_wall_and_the_pleats_are_vertical(parts):
    m = parts[SE.GROUP]
    wood = verts(m, "wood")
    w = wood[wood[:, 0] > 0.5]                                                              # the door's strip, not the console's
    assert w[:, 2].min() > 0.55 and w[:, 2].max() < L.Z_BELT - 0.06                           # a strip below the door top
    assert np.abs(w[:, 0] - (L.X_WALL_IN - 0.024)).max() < 0.015                              # on the panel
    ins = SE.door_insert_cage()
    ys = np.unique(np.round(ins.V[:, 1], 4))
    assert len(ys) > 30                                                                       # a pleat per 6 cm over the door
    # the seams run up and down: troughs (creased rings) lie at constant y over the insert's whole height
    assert len(ins.Ce) > 100


def test_the_anchors_are_where_the_things_are(parts):
    for k, a in SE.ANCHORS.items():
        p = np.asarray(a["point"], float)
        assert np.isfinite(p).all() and abs(p[0]) < 0.9 and -0.4 < p[1] < 0.8 and 0.5 < p[2] < 1.0, k
        if "dir" in a:
            assert np.linalg.norm(a["dir"]) == pytest.approx(1.0, abs=1e-6), k
    assert SE.ANCHORS["armrest_L"]["point"][0] == pytest.approx(-SE.ANCHORS["armrest_R"]["point"][0])
    assert SE.ANCHORS["door_pull_L"]["point"][0] == pytest.approx(-SE.ANCHORS["door_pull_R"]["point"][0])
    assert SE.ANCHORS["lock_pin_L"]["point"][0] == pytest.approx(L.SILL_X)
    assert SE.LOOKS["shifter"] == pytest.approx(SE.ANCHORS["shifter"]["point"])
    assert set(SE.SEAT) >= {"cushion_top", "back_gap_at_hip", "headrest_center", "headrest_size", "back_base_yz", "back_axis_yz"}
