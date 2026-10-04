"""The moulded cabinet of the bedroom player (bedroom_tech_cabinet.py): section, closed outward-facing mesh, the flat faces the
front plates, the top rest plane and the back decals rely on, the dome of the ends, and its form score (not a box)."""
import math
from collections import Counter

import numpy as np

from mkmmd.blender.library.props import bedroom_tech_cabinet as C
from mkmmd.core import form as F

# the player's values (bedroom_tech_player.py): front plane, depth, underside and top heights, the fillets, the ends
YF, YB, Z0, Z1 = -0.0575, 0.0575, 0.005, 0.160
R_FT, R_BT, R_FB, R_BB, R_END = 0.008, 0.012, 0.012, 0.012, 0.005
DISH, LEAN, KNEE = 0.006, 0.016, 0.080
L, DOME = 0.168, 0.004


def _section():
    return C.section(YF, YB, Z0, Z1, R_FT, R_BT, R_FB, R_BB, dish=DISH, lean=LEAN, knee=KNEE)


def _mesh():
    return C.cabinet(L, _section(), R_END, dome=DOME)


def _tris(V, Q, T):
    return np.concatenate([T, Q[:, [0, 1, 2]], Q[:, [0, 2, 3]]])


# ----------------------------------------------------------------------------------------------- section
def test_section_is_counter_clockwise_and_inside_the_box():
    P = _section()
    area = 0.5 * float(np.sum(P[:, 0] * np.roll(P[:, 1], -1) - np.roll(P[:, 0], -1) * P[:, 1]))
    assert area > 0                                             # counter-clockwise in (y, z)
    assert abs(P[:, 0].min() - YF) < 1e-9 and abs(P[:, 0].max() - YB) < 1e-9
    assert abs(P[:, 1].min() - Z0) < 1e-9 and abs(P[:, 1].max() - Z1) < 1e-9
    assert np.linalg.norm(np.diff(np.vstack([P, P[:1]]), axis=0), axis=1).min() > 1e-8     # no repeated points


def test_section_has_the_flat_faces_the_layout_needs():
    P = _section()
    front = P[np.abs(P[:, 0] - YF) < 1e-9]
    assert abs(front[:, 1].min() - (Z0 + R_FB)) < 1e-9 and abs(front[:, 1].max() - (Z1 - R_FT)) < 1e-9     # z 0.017 - 0.152
    top = P[np.abs(P[:, 1] - Z1) < 1e-9]
    assert abs(top[:, 0].min() - (YF + R_FT)) < 1e-9 and abs(top[:, 0].max() - (YB - R_BT)) < 1e-9
    assert top[:, 0].min() <= -0.045 and top[:, 0].max() >= 0.045        # the rest plane 0.09 m deep fits on the flat top
    back = P[np.abs(P[:, 0] - YB) < 1e-9]
    assert abs(back[:, 1].min() - (Z0 + KNEE)) < 1e-9                     # the vertical back starts at the knee


def test_the_lower_back_leans_in_and_the_underside_is_dished():
    P = _section()
    slope = P[(P[:, 0] > YB - LEAN - 1e-9) & (P[:, 1] > Z0 + R_BB - 1e-9) & (P[:, 1] < Z0 + KNEE + 1e-9)]
    lean_deg = math.degrees(math.atan2(LEAN, KNEE - R_BB))
    assert 10.0 < lean_deg < 16.0                                         # beyond the 10 deg that still counts as a box face
    assert slope[:, 0].min() < YB - LEAN * 0.5
    bottom = P[(P[:, 1] < Z0 + DISH + 1e-9) & (np.abs(P[:, 0]) < 0.03)]
    assert abs(bottom[:, 1].max() - (Z0 + DISH)) < 0.1 * DISH             # the middle of the underside is lifted by `dish`


def test_resample_cuts_long_edges_evenly():
    P = np.array([[0.0, 0.0], [0.1, 0.0], [0.1, 0.01], [0.0, 0.01]])
    Q = C.resample(P, 0.02)
    assert len(Q) == 5 + 1 + 5 + 1
    d = np.linalg.norm(np.diff(np.vstack([Q, Q[:1]]), axis=0), axis=1)
    assert d.max() <= 0.02 + 1e-12


# ----------------------------------------------------------------------------------------------- mesh
def test_the_mesh_is_closed_and_faces_outward():
    V, Q, T = _mesh()
    cnt = Counter()
    for f in list(Q) + list(T):
        for i in range(len(f)):
            cnt[(int(f[i]), int(f[(i + 1) % len(f)]))] += 1
    assert all(c == 1 for c in cnt.values())                             # no directed edge twice
    assert all(cnt.get((b, a), 0) == 1 for (a, b) in cnt)                # every edge has its opposite: closed, consistent
    assert C.volume(V, Q, T) > 0.005                                      # outward (a 5.5 litre body)
    assert all(len(set(f.tolist())) == len(f) for f in list(Q) + list(T))


def test_bounds_match_the_card():
    V, _, _ = _mesh()
    lo, hi = V.min(0), V.max(0)
    assert abs(hi[0] - (L + DOME)) < 1e-9 and abs(lo[0] + (L + DOME)) < 1e-9     # the domes reach x = +-0.172
    assert abs(lo[1] - YF) < 1e-9 and abs(hi[1] - YB) < 1e-9
    assert abs(lo[2] - Z0) < 1e-9 and abs(hi[2] - Z1) < 1e-9


def test_the_front_and_top_planes_are_exactly_flat_where_the_layout_stands_on_them():
    V, _, _ = _mesh()
    x_flat = L - R_END
    on_front = V[np.abs(V[:, 1] - YF) < 1e-9]
    assert on_front[:, 0].min() <= -x_flat + 1e-9 and on_front[:, 0].max() >= x_flat - 1e-9
    on_top = V[np.abs(V[:, 2] - Z1) < 1e-9]
    assert on_top[:, 0].min() <= -x_flat + 1e-9 and on_top[:, 1].max() >= 0.045 and on_top[:, 1].min() <= -0.045


def test_the_ends_are_domed_and_symmetric():
    V, _, _ = _mesh()
    for sign in (1, -1):
        end = V[sign * V[:, 0] > L - 1e-9]
        assert abs(sign * end[:, 0].max() - (L + DOME)) < 1e-9 or abs(end[:, 0].min() + (L + DOME)) < 1e-9
    # the dome is highest in the middle of the cap, level with the end plane on its boundary
    P = _section()
    lift = C.cap_lift(P, R_END, DOME)
    B = P + _mitre(P) * R_END
    c = B.mean(0)
    assert abs(lift(c[0], c[1]) - DOME) < 1e-12
    assert 0.0 <= lift(c[0] + 0.01, c[1] + 0.02) < DOME
    assert abs(lift(B[0][0], B[0][1])) < 1e-6                              # on the cap boundary
    assert lift(c[0] + 0.5, c[1]) == 0.0                                   # far outside: no lift


def _mitre(P):
    from mkmmd.core import shell as S
    return S.mitre_offsets(P)


# ----------------------------------------------------------------------------------------------- form
def test_the_cabinet_is_not_a_box():
    V, Q, T = _mesh()
    r = F.analyse(V, _tris(V, Q, T), worst=2)
    assert r["score"] < 0.15, r["score"]                                   # a bevelled box scores 0.7 and more
    assert r["cuboid_share"] < 0.12


def test_without_the_dish_and_the_lean_it_would_be_one():
    flat = C.section(YF, YB, Z0, Z1, R_FT, R_BT, R_FB, R_BB)
    V, Q, T = C.cabinet(L, flat, R_END, dome=0.0)
    assert F.analyse(V, _tris(V, Q, T), worst=1)["score"] > 0.5            # the guard above is not vacuous
