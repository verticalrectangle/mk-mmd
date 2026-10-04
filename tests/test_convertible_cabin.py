"""The cabin of the 1980s convertible: the seats, the dash, the steering part and the cassette against the layout (pure
numpy; the Blender side is checked by building and looking). These are the checks of the cabin part modules
(`convertible_seats`, `convertible_dash`); they move with those modules when they are rebuilt."""
import math

import numpy as np
import pytest

from mkmmd.blender.library.props import convertible_dash as D
from mkmmd.blender.library.props import convertible_layout as L
from mkmmd.blender.library.props import convertible_seats as S
from mkmmd.core import shell as SH

M = L.M

# measures of the character that needs the most room (the rig's `measure`, metres)
SITTER = {"thigh": 0.327, "shin": 0.437, "upper_arm": 0.233, "forearm": 0.229, "hand": 0.139, "torso": 0.431,
          "ankle_height": 0.096, "shoulder_half": 0.1125}


def test_the_seat_surface_under_the_hip_is_the_cushion_top_and_the_back_reclines():
    assert 14 <= L.BACK_RECLINE_DEG <= 18
    # the seats are Subdivision Surfaces: read the limit surface of the cushion's cage (the seat is built about x = 0)
    cushion = S.cushion_cage()
    ns, m = S.ring_dims(cushion)
    lim = S.limit_grid(cushion, ns, m)
    near = lim[(np.abs(lim[:, :, 0]) < 0.06) & (np.abs(lim[:, :, 1] - L.HIP_Y) < 0.06) & (lim[:, :, 2] < L.HIP_Z - 0.05)]
    # the seat's highest surface under the hip joint (which stands 15 cm above it) is the cushion top
    assert near[:, 2].max() == pytest.approx(L.HIP_Z - 0.15, abs=0.01)
    # the back's front surface passes ~10 cm behind the hip joint at hip height
    assert 0.07 < S.SEAT["back_gap_at_hip"] < 0.13
    # and the headrest stands clear of the sitter's head: its front face is further back than the back's front plane
    hy, hz = S.SEAT["headrest_center"]
    plane_y = L.BACK_BASE_Y + (hz - L.CUSHION_TOP) * math.tan(math.radians(L.BACK_RECLINE_DEG))
    assert hy - S.SEAT["headrest_size"][1] / 2 > plane_y


def test_knees_and_feet_fit_under_the_dash():
    s = SITTER
    knee = np.array([L.HIP_Y - s["thigh"] * math.cos(math.radians(8)), L.HIP_Z + s["thigh"] * math.sin(math.radians(8))])
    dash_face_y = float(D.dash_polygon()[0][D.DASH_IDX["ledge"]][0])    # the knee ledge of the dash profile
    assert knee[0] > dash_face_y + 0.15
    # lower leg: from the knee down to an ankle at the floor + ankle height
    ankle_z = L.FLOOR_Z + s["ankle_height"]
    drop = knee[1] - ankle_z
    assert drop < s["shin"]                                   # the shin can reach the floor
    ankle_y = knee[0] - math.sqrt(s["shin"] ** 2 - drop ** 2)
    assert dash_face_y - 0.30 < ankle_y < knee[0]             # feet end in front of the knees but behind the toe board


def test_driver_reaches_both_sides_of_the_wheel_with_a_bent_elbow():
    s = SITTER
    a = math.radians(L.BACK_RECLINE_DEG)
    shoulder = np.array([L.SEAT_X, L.HIP_Y + s["torso"] * math.sin(a), L.HIP_Z + s["torso"] * math.cos(a)])
    axis = np.array(L.WHEEL_AXIS)
    right = np.array([1.0, 0.0, 0.0])
    for sgn in (+1, -1):
        grip = np.array(L.WHEEL_C) + sgn * L.WHEEL_R * right
        sh = shoulder + sgn * np.array([s["shoulder_half"], 0, 0])
        d = np.linalg.norm(grip - sh)
        arm = s["upper_arm"] + s["forearm"]
        assert 0.28 < d - s["hand"] / 2 < arm - 0.02          # the wrist is reachable, the elbow stays bent
    # the rim's lowest point clears the thighs (hip height + thickness) by a hand's breadth
    up_in_plane = np.cross(axis, right)
    bottom = np.array(L.WHEEL_C) - L.WHEEL_R * (up_in_plane if up_in_plane[2] > 0 else -up_in_plane)
    thigh_top = L.HIP_Z + 0.09
    assert bottom[2] - L.WHEEL_TUBE > thigh_top


def test_the_steering_part_maps_local_z_onto_the_column_axis():
    p = D.dynamic_parts()["steering_wheel"]
    assert tuple(p.origin) == pytest.approx(L.WHEEL_C)
    z = SH.rot_matrix(*p.rot) @ np.array([0.0, 0.0, 1.0])
    assert z == pytest.approx(L.WHEEL_AXIS, abs=1e-6)
    V = p.mesh.V
    rim = V[np.abs(V[:, 2]) < 0.03]
    assert np.hypot(rim[:, 0], rim[:, 1]).max() == pytest.approx(L.WHEEL_R + L.WHEEL_TUBE, abs=0.003)


def test_cassette_travels_into_the_slot_and_leaves_a_lip():
    assert L.TAPE_OUT_Y > L.TAPE_IN_Y
    half = L.CASSETTE[1] / 2
    assert L.TAPE_OUT_Y - half > L.DECK_FACE_Y + 0.02          # held out clear of the face plate
    lip = L.TAPE_IN_Y + half - L.DECK_FACE_Y
    assert 0.0 < lip < 0.012                                   # a few mm of it shows when pushed in
    p = D.dynamic_parts()["cassette"]
    assert p.origin[1] == pytest.approx(L.TAPE_IN_Y) and p.origin[2] == pytest.approx(L.DECK_Z)
    size = p.mesh.size()
    assert size == pytest.approx(L.CASSETTE, abs=0.004)


def test_dash_surfaces_face_the_driver_and_sit_in_the_cabin():
    for name, s in D.SURFACES.items():
        n, up = np.array(s["normal"]), np.array(s["up"])
        assert np.linalg.norm(n) == pytest.approx(1) and np.linalg.norm(up) == pytest.approx(1) and abs(n @ up) < 1e-6
        assert s["size"][0] > 0.02 and s["size"][1] > 0.01
        if name != "cassette_label":
            assert n[1] > 0.5                                  # toward the driver (+Y)
            assert -0.8 < s["center"][0] < 0.8 and -0.7 < s["center"][1] < -0.2 and 0.6 < s["center"][2] < 1.15


def test_bar_graph_cells_are_the_row_the_shader_lights_by_position():
    """The bar-graph shader lights cells by their position along convertible_dash.BAR_GRAPH (first -> last): the
    `vfd_lit` quads of the cluster must be exactly that evenly spaced row."""
    dash = D.static_parts()["decals"]                           # the display layers: flat quads lifted off the glass
    cells = dash.V[dash.Q[dash.Qm == M["vfd_lit"]]].mean(1)
    bg = D.BAR_GRAPH
    first, last = np.array(bg["first"]), np.array(bg["last"])
    assert len(cells) == bg["cells"] == 24
    order = np.argsort(cells @ np.array(bg["axis"]))
    expect = first + (last - first) * (np.arange(24) / 23.0)[:, None]
    assert cells[order] == pytest.approx(expect, abs=1e-6)
    assert np.dot(last - first, bg["axis"]) > 0                 # cell 0 is the leftmost as the driver sees it


def test_the_cassette_label_surface_matches_the_label_geometry():
    s = D.SURFACES["cassette_label"]
    p = D.dynamic_parts()["cassette"]
    label = p.mesh.V[p.mesh.Q[p.mesh.Qm == M["label"]]].reshape(-1, 3)
    lo, hi = label.min(0), label.max(0)
    w, h = s["size"]
    # the text area (centred on the part's origin) lies inside the label paper, which the hub window cuts short
    assert lo[0] <= -w / 2 + 0.001 and hi[0] >= w / 2 - 0.001 and lo[1] <= -h / 2 + 0.001 and hi[1] >= h / 2 - 0.001
    assert np.array(p.origin) == pytest.approx((0, L.TAPE_IN_Y, L.DECK_Z))
    assert s["center"][1] == pytest.approx(L.TAPE_IN_Y, abs=0.002) and s["center"][2] == pytest.approx(L.DECK_Z, abs=0.002)
    # the label looks up (+Z) and its text runs toward -X as the driver sees it: right = up x normal
    assert np.cross(s["up"], s["normal"]) == pytest.approx((-1, 0, 0), abs=1e-6)


def test_speed_surface_grid_matches_the_ghost_digits():
    """The card documents the speed readout's digit grid (a kinetic-type stage right-aligns numerals on it): three
    seven-segment cells at the documented pitch, each digit_width wide and digit_height tall."""
    s, g = D.SURFACES["speed"], D.SPEED_GRID
    assert g["columns"] == 3 and g["pitch"] * g["columns"] == pytest.approx(s["size"][0], abs=1e-4)
    assert g["digit_height"] == pytest.approx(s["size"][1])
    mesh = D.static_parts()["decals"]
    ghost = np.concatenate([mesh.Q[mesh.Qm == M["vfd_ghost"]].ravel(), mesh.T[mesh.Tm == M["vfd_ghost"]].ravel()])
    P = mesh.V[np.unique(ghost)] - np.array(s["center"])
    n, up = np.array(s["normal"]), np.array(s["up"])
    right = np.cross(up, n)
    u, v, d = P @ right, P @ up, P @ n
    on_speed = (np.abs(d) < 0.003) & (np.abs(u) <= s["size"][0] / 2 + 0.002) & (np.abs(v) <= s["size"][1] / 2 + 0.002)
    for k in range(3):
        c = (k - 1) * g["pitch"]
        cell = on_speed & (np.abs(u - c) <= g["digit_width"] / 2 + 0.0005)
        assert u[cell].min() == pytest.approx(c - g["digit_width"] / 2, abs=0.0015)
        assert u[cell].max() == pytest.approx(c + g["digit_width"] / 2, abs=0.0015)
        assert v[cell].max() - v[cell].min() == pytest.approx(g["digit_height"], abs=0.002)
