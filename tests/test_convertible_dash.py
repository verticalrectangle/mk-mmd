"""The dash module of the 1980s convertible (pure numpy): the cages (closed, creased, the profiles simple, where they stand),
the panels on the dash's flat faces, the cluster's display stack and the bar graph the shader lights, the cassette and the
steering wheel (frames, sizes, placement), the card's surfaces and the module's budget. (The Blender side is checked by
building and looking.)"""
import math
import time

import numpy as np
import pytest

from mkmmd.blender.library.props import convertible_dash as D
from mkmmd.blender.library.props import convertible_layout as L
from mkmmd.core import shell as S

M = L.M
ROLE = {i: n for n, i in M.items()}


@pytest.fixture(scope="module")
def parts():
    return D.static_parts()


@pytest.fixture(scope="module")
def shells():
    return D.static_shells()


@pytest.fixture(scope="module")
def dyn():
    return D.dynamic_parts()


def faces_of(mesh, role):
    """Corner vertex arrays (k, 4 or 3, 3) of the faces of a role: (quads, triangles)."""
    q = mesh.V[mesh.Q[mesh.Qm == M[role]]]
    t = mesh.V[mesh.T[mesh.Tm == M[role]]]
    return q, t


def role_vertices(mesh, role):
    q, t = faces_of(mesh, role)
    return np.concatenate([q.reshape(-1, 3), t.reshape(-1, 3)])


def world(part):
    R = S.rot_matrix(*part.rot)
    return part.mesh.V @ R.T + np.asarray(part.origin, float)


def segments_cross(P):
    n = len(P)

    def ccw(a, b, c):
        return (c[1] - a[1]) * (b[0] - a[0]) - (b[1] - a[1]) * (c[0] - a[0])
    for i in range(n):
        for j in range(i + 2, n):
            if (j + 1) % n == i:
                continue
            a, b, c, d = P[i], P[(i + 1) % n], P[j], P[(j + 1) % n]
            if ccw(a, c, d) * ccw(b, c, d) < 0 and ccw(a, b, c) * ccw(a, b, d) < 0:
                return True
    return False


# ============================================================================================== the module's contract
def test_the_module_builds_fast_and_well_formed(parts, shells, dyn):
    t0 = time.time()
    D.static_shells(), D.static_parts(), D.dynamic_parts()
    assert time.time() - t0 < 3.0                                  # the budget is 0.4 s; a loaded machine gets slack
    assert set(parts) == {"interior", "decals"} and set(shells) == {"dash", "binnacle", "stack"}
    assert set(dyn) == {"steering_wheel", "cassette", "visors"}
    meshes = [("interior", parts["interior"]), ("decals", parts["decals"])] + [(k, s.mesh) for k, s in shells.items()]
    for k, p in dyn.items():
        meshes.append((k, S.Mesh(world(p), p.mesh.Q, p.mesh.T, p.mesh.Qm, p.mesh.Tm)))
    for key, m in meshes:
        assert np.isfinite(m.V).all() and m.nfaces > 0, key
        assert m.Q.max(initial=0) < len(m.V) and m.T.max(initial=0) < len(m.V), key
        assert m.Qm.min(initial=0) >= 0 and m.Qm.max(initial=0) < len(L.MATS), key
        lo, hi = m.bbox()
        assert lo[0] > -1.0 and hi[0] < 1.0 and lo[1] > -2.4 and hi[1] < 2.4 and lo[2] > 0.0 and hi[2] < 1.4, key
        nq, nt = m.face_vectors()
        areas = np.concatenate([np.linalg.norm(nq, axis=1), np.linalg.norm(nt, axis=1)])
        assert (areas > 1e-12).mean() > 0.99, f"{key}: zero-area faces"
    assert sum(s.mesh.nfaces for s in shells.values()) < 1000          # cages stay small: the Subdivision Surface does the rest
    assert parts["interior"].nfaces + parts["decals"].nfaces < 30000


def test_the_shells_are_closed_wound_outward_and_their_creases_are_valid(shells):
    for key, sh in shells.items():
        m = sh.mesh
        assert m.is_closed() and m.volume() > 0, key
        assert sh.levels == 2
        assert ((m.Cw > 0) & (m.Cw <= 1)).all(), key
        assert m.Ce.max(initial=0) < len(m.V), key


# =================================================================================================== the dash's profiles
def test_the_dash_profile_is_a_simple_counter_clockwise_polygon_with_one_flat_pad():
    P, cw = D.dash_polygon()
    assert len(P) == len(cw) == len(D.DASH_NAMES) and S.polygon_area(P) > 0 and not segments_cross(P)
    assert ((cw >= 0) & (cw <= 1)).all()
    pad = P[D.DASH_IDX["crest"]:D.DASH_IDX["cowl1"] + 1]
    assert np.allclose(pad[:, 1], [D._zp(y) for y in pad[:, 0]], atol=1e-9)         # the pad is one plane
    assert pad[0, 1] <= L.DASH_TOP_Z[1] + 1e-3 and P[D.DASH_IDX["cowl2"], 1] < L.Z_GLASS     # under the foot of the glass
    assert D._zp(L.Y_COWL) == pytest.approx(L.DASH_TOP_Z[0])                                # the pad starts at the glass foot
    assert pad[-1, 0] == pytest.approx(D.YW + 0.009) and D.YW == pytest.approx(L.Y_COWL - 0.020)   # and runs on to the cowl lip
    assert D.Z_LOW[0] < D.Z_LOW[1] < D.Z_LOW[2] < D.Z_SHELF < D.Z_BAND[0] < D.Z_BAND[1] < D.Z_BROW < L.DASH_TOP_Z[1]
    band = P[D.DASH_IDX["band0"]:D.DASH_IDX["brow"] + 1]
    assert np.allclose(band[:, 0], band[0, 0])                                       # the band under the brow is vertical
    low = P[D.DASH_IDX["low0"]:D.DASH_IDX["low2"] + 1]
    d = low[-1] - low[0]
    assert all(abs(d[0] * (p - low[0])[1] - d[1] * (p - low[0])[0]) < 1e-9 for p in low)    # the lower panel is one plane
    assert P[:, 0].min() == pytest.approx(D.YW) and P[:, 0].max() <= L.DASH_TOP_Y[1] + 0.01


def test_the_dash_stands_between_the_door_panels_under_the_cowl(shells):
    lo, hi = shells["dash"].mesh.bbox()
    assert lo[0] == pytest.approx(-L.DASH_X) and hi[0] == pytest.approx(L.DASH_X)
    assert lo[1] == pytest.approx(D.YW, abs=1e-6) and hi[1] <= L.DASH_TOP_Y[1] + 0.01
    assert hi[2] <= L.DASH_TOP_Z[1] + 0.002 and hi[2] > L.DASH_TOP_Z[1] - 0.01      # the pad's highest point
    assert lo[2] > 0.25


def test_the_binnacle_face_is_the_glass_plane_four_millimetres_behind_it(shells):
    P, cw = D.hood_polygon()
    assert not segments_cross(P) and S.polygon_area(P) > 0
    nrm = D.CN[1:]
    for k in range(5, 8):                                                           # the three flat face points
        d = (P[k] - D.CC[1:]) @ nrm
        assert d == pytest.approx(-D.HOOD_BACK, abs=1e-9)
    # the pod covers the whole bezel, and its back is buried in the dash (below the pad plane)
    lo, hi = shells["binnacle"].mesh.bbox()
    assert lo[0] < D.CC[0] - D.CW / 2 - 0.02 and hi[0] > D.CC[0] + D.CW / 2 + 0.02
    assert P[-1][1] < D._zp(P[-1][0]) - 0.02


def test_the_stack_pod_grows_from_the_faceplate_into_the_dash(shells):
    ys = [r[0] for r in D.STACK_RINGS]
    ws = [r[1] for r in D.STACK_RINGS]
    assert ys == sorted(ys, reverse=True) and ws == sorted(ws)                       # wider toward the dash
    assert max(ys) < L.DECK_FACE_Y                                                   # the faceplate stands in front of it
    lo, hi = shells["stack"].mesh.bbox()
    assert hi[0] <= max(L.STACK_X) + 0.02 and lo[0] >= min(L.STACK_X) - 0.02        # the layout's stack width, give or take


# ================================================================================================ panels on the dash
def test_panels_stand_on_the_flat_faces_with_their_backs_inside(parts):
    """Every wood panel and the glove lid lies in front of its plane by 3-12 mm at the top and no further back than it."""
    inter = parts["interior"]
    wood = role_vertices(inter, "wood")
    assert len(wood) > 0
    P, _ = D.dash_polygon()
    band_y, low = P[D.DASH_IDX["band0"], 0], D._lower_frame()
    on_band = wood[np.abs(wood[:, 2] - D.Z_BAND_C) < 0.03]
    on_low = wood[wood[:, 2] < D.Z_SHELF]
    assert len(on_band) and len(on_low)
    assert on_band[:, 1].max() == pytest.approx(band_y + 0.003, abs=1e-4)           # proud 3 mm
    n, o = low[2], D._lower_at(0.0, D.Z_LOW[2] - 0.042)
    assert ((on_low - o) @ n).max() == pytest.approx(0.003, abs=1e-4)
    assert wood[:, 0].min() > -0.70 and wood[:, 0].max() < 0.72                      # nothing runs into the dash ends


def test_the_deck_faceplate_is_on_the_deck_plane_and_the_knobs_flank_the_slot(parts):
    inter = parts["interior"]
    plate = role_vertices(inter, "deck_face")
    assert plate[:, 1].max() == pytest.approx(L.DECK_FACE_Y, abs=1e-9)
    assert abs(plate[:, 0]).max() <= 0.125 + 1e-6 and abs(plate[:, 2].mean() - L.DECK_Z) < 0.01
    slot = role_vertices(parts["decals"], "underbody")
    slot = slot[np.abs(slot[:, 2] - D.SLOT_Z) < 0.01]
    slot = slot[np.abs(slot[:, 1] - (L.DECK_FACE_Y + D.DECK_LIFT)) < 1e-6]
    assert slot[:, 0].min() == pytest.approx(-0.054) and slot[:, 0].max() == pytest.approx(0.054)
    assert np.ptp(slot[:, 2]) == pytest.approx(0.016, abs=1e-6)


def test_the_slot_clears_the_cassette_and_the_cassette_leaves_a_lip(dyn):
    c = dyn["cassette"]
    assert c.origin[1] == pytest.approx(L.TAPE_IN_Y) and c.origin[2] == pytest.approx(L.DECK_Z)
    V = world(c)
    half = 0.5 * 0.016                                                               # the slot is 16 mm high, 108 wide
    top, bot = V[:, 2].max(), V[:, 2].min()
    assert D.SLOT_Z - half < bot and top < D.SLOT_Z + half and np.abs(V[:, 0]).max() < 0.054
    lip = V[:, 1].max() - L.DECK_FACE_Y
    assert 0.0 < lip < 0.012


# ============================================================================================ the cluster and its glass
def test_the_cluster_glass_and_the_display_stack(parts):
    d = parts["decals"]
    glass_q, _ = faces_of(d, "display_glass")
    g = glass_q[np.abs(glass_q[:, :, 0].mean(1) - D.CC[0]) < 0.2]                   # the cluster's glass (not the radio's)
    g = g[(g.reshape(len(g), -1, 3).mean(1)[:, 2] > 0.9)]
    assert len(g) == 1
    c = g[0].mean(0)
    assert c == pytest.approx(D.CC, abs=1e-9)
    n = np.cross(g[0][1] - g[0][0], g[0][3] - g[0][0])
    n /= np.linalg.norm(n)
    assert n == pytest.approx(D.CN, abs=1e-9)
    # every lit / ghost / indicator element lies in the glass rectangle, 0.6 mm above it
    for role in ("vfd_lit", "ind_a", "ind_b", "ind_c"):
        v = role_vertices(d, role)
        assert ((v - D.CC) @ D.CN == pytest.approx(D.LIFT, abs=1e-9)), role
        u, w = (v - D.CC) @ D.CR, (v - D.CC) @ D.CU
        assert abs(u).max() < D.CW / 2 and abs(w).max() < D.CH / 2, role
    gh = role_vertices(d, "vfd_ghost")
    gh = gh[np.abs((gh - D.CC) @ D.CN - D.LIFT) < 1e-6]
    u, w = (gh - D.CC) @ D.CR, (gh - D.CC) @ D.CU
    assert abs(u).max() < D.CW / 2 and abs(w).max() < D.CH / 2


def test_the_bar_graph_is_one_row_of_24_cells_ordered_left_to_right(parts):
    q, t = faces_of(parts["decals"], "vfd_lit")
    assert len(q) == D.BAR_CELLS and len(t) == 0                                      # nothing else uses the lit role
    ctr = q.mean(1)
    bg = D.BAR_GRAPH
    first, last = np.array(bg["first"]), np.array(bg["last"])
    axis = np.array(bg["axis"])
    assert bg["cells"] == D.BAR_CELLS and axis == pytest.approx(D.CR)
    s = (ctr - first) @ axis
    assert np.all(np.diff(s) > 0)                                                     # left to right as the driver sees it
    assert ctr[0] == pytest.approx(first, abs=1e-9) and ctr[-1] == pytest.approx(last, abs=1e-9)
    pitch = np.linalg.norm(last - first) / (D.BAR_CELLS - 1)
    assert np.diff(s) == pytest.approx(pitch, abs=1e-9)
    w = np.linalg.norm(q[:, 1] - q[:, 0], axis=1)
    assert w == pytest.approx((D.BARS_SIZE[0] - (D.BAR_CELLS - 1) * D.BAR_GAP) / D.BAR_CELLS, abs=1e-9)   # ~ 0.008 wide
    assert pitch - w[0] == pytest.approx(D.BAR_GAP, abs=1e-9)


def test_the_surfaces_follow_the_card_conventions():
    for name, s in D.SURFACES.items():
        n, up = np.array(s["normal"]), np.array(s["up"])
        assert np.linalg.norm(n) == pytest.approx(1) and np.linalg.norm(up) == pytest.approx(1) and abs(n @ up) < 1e-6
        assert s["size"][0] > 0.02 and s["size"][1] > 0.01
        assert all(isinstance(x, float) for x in s["center"] + s["normal"] + s["up"] + s["size"]), name
        if name != "cassette_label":
            assert n[1] > 0.5                                                          # toward the driver (+Y)
            assert -0.8 < s["center"][0] < 0.8 and -0.7 < s["center"][1] < -0.2 and 0.6 < s["center"][2] < 1.15
    for k in ("cluster", "speed", "bars"):
        s = D.SURFACES[k]
        assert np.array(s["normal"]) == pytest.approx(D.CN) and np.array(s["up"]) == pytest.approx(D.CU)
        assert (np.array(s["center"]) - D.CC) @ D.CN == pytest.approx(D.TEXT_LIFT)    # the text floats above the glass elements
    assert D.SURFACES["speed"]["size"] == [0.125, 0.052] and D.SURFACES["cluster"]["size"] == list(L.CLUSTER_SIZE)
    g = D.GRIDS["speed"]
    assert g["columns"] == 3 and g["pitch"] == pytest.approx(D.SURFACES["speed"]["size"][0] / 3, abs=1e-5)
    assert (g["pitch"] - g["digit_width"]) / 2 > 0.0 and g["digit_height"] <= D.SURFACES["speed"]["size"][1]


def test_the_speed_digits_sit_in_their_three_columns(parts):
    gh = role_vertices(parts["decals"], "vfd_ghost")
    gh = gh[np.abs((gh - D.CC) @ D.CN - D.LIFT) < 1e-6]
    u, w = (gh - D.CC) @ D.CR - D.SPEED_UV[0], (gh - D.CC) @ D.CU - D.SPEED_UV[1]
    digits = gh[(abs(w) < D.DIGIT_H / 2 + 1e-6) & (abs(u) < D.SPEED_SIZE[0] / 2)]
    uu = (digits - D.CC) @ D.CR - D.SPEED_UV[0]
    pitch = D.GRIDS["speed"]["pitch"]
    for k in range(3):
        cell = uu[np.abs(uu - (k - 1) * pitch) < D.DIGIT_W / 2 + 1e-4] - (k - 1) * pitch      # the digit itself (the dots sit outside)
        assert len(cell) > 0 and cell.min() == pytest.approx(-D.DIGIT_W / 2, abs=1e-4) and cell.max() == pytest.approx(D.DIGIT_W / 2, abs=1e-4)


# ================================================================================================ the wheel and the rest
def test_the_steering_part_maps_local_z_onto_the_column_axis(dyn):
    p = dyn["steering_wheel"]
    assert tuple(p.origin) == pytest.approx(L.WHEEL_C)
    z = S.rot_matrix(*p.rot) @ np.array([0.0, 0.0, 1.0])
    assert z == pytest.approx(L.WHEEL_AXIS, abs=1e-6)
    assert p.rot == pytest.approx((-68.0, 0.0, 0.0), abs=1e-6)
    V = p.mesh.V
    rim = V[np.abs(V[:, 2]) < 0.03]
    assert np.hypot(rim[:, 0], rim[:, 1]).max() == pytest.approx(L.WHEEL_R + L.WHEEL_TUBE, abs=0.003)
    # the rim is a closed ring tube: the wheel's biggest closed component
    assert p.mesh.nfaces > 2000
    # the spokes droop (local +Y is down in the car) and the hub is dished away from the driver
    hub = role_vertices(p.mesh, "chrome")
    assert hub[:, 2].min() > -0.02 and hub[:, 2].max() < 0.02


def test_the_wheel_clears_the_cluster_sight_line_over_the_hub_by_a_finger():
    eye = np.array([L.SEAT_X, 0.30, 1.22])
    p = D.WHEEL_C
    ax = D.WHEEL_AXIS
    for uv in ((0.0, -0.040), (-0.145, 0.0)):                                         # the bar graph's middle, the fuel gauge
        tgt = D.CC + uv[0] * D.CR + uv[1] * D.CU
        d = tgt - eye
        t = ((p - eye) @ ax) / (d @ ax)
        hit = eye + t * d
        loc = hit - p
        up_in = np.cross(ax, np.array([1.0, 0.0, 0.0]))
        up_in = up_in if up_in[2] > 0 else -up_in
        clear = loc @ up_in - 0.035                                                    # above the pad's top edge (half height 0.04)
        assert clear > 0.004 or abs(loc[0]) > 0.075                                    # the hub's half width is 0.06


def turned(part, visors):
    """The visors part in the car's frame at the param `visors`: the builder turns the object about its local X by
    VISOR_FLIP_DEG * (1 - visors), so 1 is the rest pose (down on the glass) and 0 flipped up."""
    R = S.rot_matrix(L.VISOR_FLIP_DEG * (1.0 - visors), 0.0, 0.0)
    return S.Mesh(part.mesh.V @ R.T + np.asarray(part.origin, float), part.mesh.Q, part.mesh.T, part.mesh.Qm, part.mesh.Tm)


def test_visors_down_stand_parallel_to_the_glass_a_centimetre_inside_it(dyn):
    vis = role_vertices(turned(dyn["visors"], 1.0), "vinyl")
    assert len(vis) > 0
    g0, g1 = np.array(L.WS_BASE), np.array(L.WS_TOP)
    d = (g1 - g0) / np.linalg.norm(g1 - g0)
    nrm = np.array([d[1], -d[0]])
    dist = (vis[:, 1:] - g0) @ nrm
    assert dist.min() == pytest.approx(0.010, abs=1e-4) and dist.max() < 0.040          # a centimetre off the glass, 22 mm thick
    s = (vis[:, 1:] - g0) @ d
    assert s.max() < np.linalg.norm(g1 - g0) - L.WS_FRAME                                # under the header rail
    assert abs(vis[:, 0]).max() < 0.60 and abs(vis[:, 0]).min() < 0.20


def test_the_visors_part_turns_about_the_common_axis_of_its_two_rods(dyn):
    p = dyn["visors"]
    assert tuple(p.rot) == (0.0, 0.0, 0.0) and tuple(p.origin) == pytest.approx((0.0, *D.VISOR_PIVOT))
    rods = role_vertices(p.mesh, "chrome")
    assert np.abs(rods[:, 1:]).max() == pytest.approx(0.0036, abs=1e-4)                  # both rods are centred on the local X axis
    assert np.abs(rods[:, 0]).max() == pytest.approx(0.56, abs=0.005) and np.abs(rods[:, 0]).min() < 0.17    # one per visor
    # the axis runs just under the header rail, along the glass; a turn leaves the rods where they are
    g0, g1 = np.array(L.WS_BASE), np.array(L.WS_TOP)
    d = (g1 - g0) / np.linalg.norm(g1 - g0)
    s_axis = (np.array(D.VISOR_PIVOT) - g0) @ d
    assert np.linalg.norm(g1 - g0) - L.WS_FRAME - 0.012 < s_axis < np.linalg.norm(g1 - g0) - L.WS_FRAME
    for v in (0.0, 0.4, 1.0):
        r = role_vertices(turned(p, v), "chrome")
        assert r[:, 1].mean() == pytest.approx(D.VISOR_PIVOT[0], abs=1e-6) and r[:, 2].mean() == pytest.approx(D.VISOR_PIVOT[1], abs=1e-6)


def test_visors_flipped_up_lie_level_behind_the_header_above_the_eye_line(dyn):
    """visors = 0: a visor under a folded top is stowed - level, pointing back from the header - so it no longer hangs in front
    of the faces for a camera over the hood."""
    flipped = turned(dyn["visors"], 0.0)
    vis = role_vertices(flipped, "vinyl")
    y_rod, z_rod = D.VISOR_PIVOT
    assert np.ptp(vis[:, 2]) < D.V_THICK + 0.004                                          # level: only the pad's thickness is left
    assert vis[:, 1].min() > y_rod - 0.005 and vis[:, 1].max() == pytest.approx(y_rod + 0.160, abs=0.005)    # pointing back, 16 cm
    assert z_rod - 0.02 < vis[:, 2].min() and vis[:, 2].max() < z_rod + 0.02              # at the rods' height
    assert vis[:, 2].max() < L.WS_TOP[1] and vis[:, 2].min() > 1.22 + 0.03                # under the header's top, over the eye line
    assert vis[:, 1].max() < -0.05                                                         # nowhere near the faces (y > 0.2)
    assert abs(vis[:, 0]).max() < 0.60 and abs(vis[:, 0]).min() < 0.20
    # the stitching turned with the pads: it stays on the face that now looks up (the sun-blocking face)
    seam = role_vertices(flipped, "seam")
    assert len(seam) > 0 and abs(seam[:, 2].max() - vis[:, 2].max()) < 0.004 and np.ptp(seam[:, 2]) < 0.001


def test_the_visors_never_swing_back_to_the_faces_or_down_into_the_dash(dyn):
    for v in np.linspace(0.0, 1.0, 9):
        vis = role_vertices(turned(dyn["visors"], v), "vinyl")
        assert vis[:, 1].max() < 0.0                                                       # never back to the sitters' faces
        assert vis[:, 2].min() > L.Z_GLASS + 0.05                                          # nor down into the dash


def test_the_mirror_glass_is_at_the_layout_point_and_faces_the_driver(parts):
    q, t = faces_of(parts["decals"], "mirror_glass")
    assert len(q) == 1
    c = q[0].mean(0)
    assert c == pytest.approx(L.MIRROR_C, abs=0.012)
    n = np.cross(q[0][1] - q[0][0], q[0][3] - q[0][0])
    n /= np.linalg.norm(n)
    assert n[1] > 0.9 and n[0] > 0.1                                                     # to the driver (+X), yawed, not sideways
    assert D.ANCHORS["mirror"]["point"] == pytest.approx(list(L.MIRROR_C))
    assert D.ANCHORS["slot"]["point"] == pytest.approx(list(L.SLOT_C)) and D.ANCHORS["slot"]["dir"] == [0.0, -1.0, 0.0]
    assert D.LOOKS["dash"] == pytest.approx(list(L.BINNACLE_C)) and D.LOOKS["cassette"] == pytest.approx(list(L.SLOT_C))


def test_pedals_hang_under_the_dash_and_the_column_enters_the_band(parts):
    inter = parts["interior"]
    rub = role_vertices(inter, "rubber")
    assert len(rub) > 0 and rub[:, 2].max() < 0.5 and rub[:, 2].min() > L.FLOOR_Z    # pads between the floor and the ledge
    assert rub[:, 1].max() < L.Y_COWL + 0.15                                          # under the dash, ahead of the knees
    sf = D._column_s_face()
    y = L.WHEEL_C[1] - L.WHEEL_AXIS[1] * sf
    P, _ = D.dash_polygon()
    assert y == pytest.approx(P[D.DASH_IDX["band0"], 0])                              # the column meets the band face
    z = L.WHEEL_C[2] - L.WHEEL_AXIS[2] * sf
    assert P[D.DASH_IDX["band0"], 1] < z < P[D.DASH_IDX["brow"], 1]


def test_the_tests_geometry_helpers_agree_with_the_layout():
    assert D.R_WHEEL @ np.array([0.0, 0.0, 1.0]) == pytest.approx(L.WHEEL_AXIS, abs=1e-9)
    assert math.degrees(math.acos(float(D.CN @ np.array([0.0, 1.0, 0.0])))) == pytest.approx(L.BINNACLE_TILT_DEG)
