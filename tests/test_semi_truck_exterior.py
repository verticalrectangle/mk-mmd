"""The semi truck's exterior hardware (pure numpy; the Blender side is checked by building and looking): closed outward
pieces in the `ext_*` groups, the bumper on the layout's face line, the headlamps on L.HEADLAMP in front of the beams,
the tanks and step boxes where the layout puts them, the mirror heads inside 3.05 m overall, the glad hands and plug at
the van's couplers, nothing within 3 cm of a tyre, parts following the shells they are probed on, and the budget."""
import numpy as np
import pytest

from mkmmd.blender.library.props import semi_truck_exterior as E
from mkmmd.blender.library.props import semi_truck_exterior_cab as CAB
from mkmmd.blender.library.props import semi_truck_exterior_chassis as CH
from mkmmd.blender.library.props import semi_truck_exterior_front as F
from mkmmd.blender.library.props import semi_truck_exterior_kit as K
from mkmmd.blender.library.props import semi_truck_layout as L
from mkmmd.core import shell as S

M = L.M
BUDGET = 90_000                                  # evaluated triangles of the whole exterior (the brief's share of the rig)
CLEAR = 0.03                                     # nothing nearer a tyre than this


@pytest.fixture(scope="module")
def parts():
    return E.static_parts()


def faces_of(m, role):
    """(k, n, 3) corner points of the faces of a material role (quads and triangles)."""
    q = m.V[m.Q[m.Qm == M[role]]] if len(m.Q) else np.zeros((0, 4, 3))
    t = m.V[m.T[m.Tm == M[role]]] if len(m.T) else np.zeros((0, 3, 3))
    return q.reshape(-1, 3), t.reshape(-1, 3)


def points_of(m, role):
    q, t = faces_of(m, role)
    return np.concatenate([q, t])


def tris(m):
    return 2 * len(m.Q) + len(m.T)


def tyre_gap(P):
    """The smallest distance of the points P (n, 3) from the solid cylinder of any tyre (negative inside one)."""
    P = np.asarray(P, float)
    best = np.inf
    for key, (hx, hy, hz) in L.wheel_hubs().items():
        for xc in L.tyre_centres(key):
            dr = np.hypot(P[:, 1] - hy, P[:, 2] - hz) - L.TYRE_R
            da = np.abs(P[:, 0] - xc) - L.TYRE_W / 2
            d = np.where(da <= 0, dr, np.where(dr <= 0, da, np.hypot(np.maximum(dr, 0), np.maximum(da, 0))))
            best = min(best, float(d.min()))
    return best


# ======================================================================================== well formed
def test_the_groups_are_closed_outward_solids_in_the_exterior_namespace(parts):
    assert set(parts) == set(E.GROUPS)
    for g, m in parts.items():
        assert g.startswith("ext_") and not L.is_trailer(g), g
        assert np.isfinite(m.V).all() and m.nfaces > 0, g
        assert m.is_closed(), g                                                   # every piece a closed solid
        assert m.volume() > 0, g
        used = set(np.unique(np.concatenate([m.Qm, m.Tm])).tolist())
        assert used <= set(range(len(L.MATS))), g


def test_every_piece_is_closed_and_outward_on_its_own():
    body = K.Body(None)
    for mod in (F, CAB, CH):
        for g, meshes in mod.parts(body).items():
            for i, m in enumerate(meshes):
                assert m.is_closed() and m.volume() > 0, (mod.__name__, g, i)


def test_the_exterior_stays_inside_its_triangle_budget(parts):
    total = sum(tris(m) for m in parts.values())
    assert 40_000 < total <= BUDGET


def test_the_exterior_roles_it_alone_uses_are_drawn(parts):
    used = set()
    for m in parts.values():
        used |= set(np.unique(np.concatenate([m.Qm, m.Tm])).tolist())
    for role in ("chrome", "alloy", "lens_head", "lamp_amber", "lamp_red", "mirror_glass", "tread_plate", "shield",
                 "hose_red", "hose_blue"):
        assert M[role] in used, role


# ======================================================================================== clearance and boxes
def test_nothing_comes_within_3_cm_of_a_tyre(parts):
    for g, m in parts.items():
        cq, ct = m.face_centres()
        assert tyre_gap(np.concatenate([m.V, cq, ct])) >= CLEAR, g


# ======================================================================================== the nose
def test_the_bumper_face_stands_on_the_layouts_face_line(parts):
    b = F.bumper()
    lo, hi = b.bbox()
    assert lo[1] == pytest.approx(L.Y_BUMPER, abs=0.001)                         # the crowned face's peak
    assert hi[1] <= L.BUMPER_Y[1] + 1e-6 and hi[1] > L.BUMPER_Y[1] - 0.01         # the end wraps reach back to their mark
    assert hi[0] <= L.BUMPER_HALF_W and hi[0] > L.BUMPER_HALF_W - 0.01 and lo[0] == pytest.approx(-hi[0])
    assert (lo[2], hi[2]) == pytest.approx(L.BUMPER_Z, abs=0.001)
    # a formed sheet: thin everywhere (the section is a C a few mm thick, not a block)
    sec = F.bumper_section()
    assert abs(S.polygon_area(sec)) < 0.004 * 0.6
    # straight across the face: at the centre line its front is the face line from top to bottom roll
    face = b.V[(np.abs(b.V[:, 0]) < 0.5) & (b.V[:, 1] < L.Y_BUMPER + 0.01)]
    assert np.ptp(face[:, 2]) > 0.25


def test_the_headlamps_sit_on_the_layout_with_the_beams_in_front(parts):
    lamps = parts["ext_lamps"]
    lens = points_of(lamps, "lens_head")
    x, y, z = L.HEADLAMP
    for sx in (1.0, -1.0):
        mine = lens[lens[:, 0] * sx > 0]
        assert (mine[:, 0] * sx).mean() == pytest.approx(x, abs=0.01)
        assert mine[:, 2].mean() == pytest.approx(z, abs=0.01)
        assert mine[:, 1].min() == pytest.approx(y, abs=0.006)                    # the lenses' fronts in the lamp plane
        assert mine[:, 1].min() > y - 0.03                                        # the beam's spot stands in front of them
        xs = np.sort(np.unique(np.round(mine[:, 0] * sx, 3)))
        assert np.diff(xs).max() > F.LAMP["lens_gap"] * 0.9                       # two lenses side by side
    amber = points_of(lamps, "lamp_amber")
    signal = amber[(amber[:, 1] < y + 0.02) & (np.abs(amber[:, 2] - z) < 0.06)]
    assert len(signal) and np.abs(signal[:, 0]).min() > x + 0.18                 # the turn signals outboard of the pods


def test_the_fender_mirrors_stand_on_the_fender_crowns(parts):
    m = parts["ext_mirrors"]
    glass = points_of(m, "mirror_glass")
    front = glass[glass[:, 1] < L.CAB_FRONT_Y - 0.6]
    for sx in (1.0, -1.0):
        mine = front[front[:, 0] * sx > 0]
        assert len(mine) and L.FENDER_TOP_Z < mine[:, 2].min() < mine[:, 2].max() < L.FENDER_TOP_Z + 0.45
        assert L.AXLE_S_Y - 0.5 < mine[:, 1].mean() < L.AXLE_S_Y + 0.2


# ======================================================================================== round the cab
def test_the_mirror_heads_reach_118_inches_and_stay_inside_3_05_m(parts):
    m = parts["ext_mirrors"]
    half = np.abs(m.V[:, 0]).max()
    assert 2 * half <= 3.05 and 2 * half == pytest.approx(118.5 * 0.0254, abs=0.03)
    glass = points_of(m, "mirror_glass")
    for sx in (1.0, -1.0):
        head = glass[(glass[:, 0] * sx > 1.2) & (glass[:, 2] > L.MIRROR[2] - 0.3)]
        lo, hi = head.min(0), head.max(0)
        assert (0.5 * (lo + hi))[[0, 2]] * (sx, 1) == pytest.approx((L.MIRROR[0], L.MIRROR[2]), abs=0.02)
        assert hi[2] - lo[2] == pytest.approx(CAB.HEAD["h"] - 0.022, abs=0.01)
    whips = points_of(m, "rubber")
    assert whips[:, 2].max() > L.SLEEPER_ROOF_Z + 0.3                             # the CB whips stand over the roof


def test_the_clearance_lamps_are_five_on_the_visor_and_five_on_the_sleeper(parts):
    m = parts["ext_lamps"]
    amber = points_of(m, "lamp_amber")
    for zlo, zhi, ylo, yhi in ((2.6, 3.0, -5.3, -5.0), (3.0, 3.5, L.SLEEPER_Y[0], L.SLEEPER_Y[1])):
        sel = amber[(amber[:, 2] > zlo) & (amber[:, 2] < zhi) & (amber[:, 1] > ylo) & (amber[:, 1] < yhi)]
        xs = np.unique(np.round(sel[:, 0], 1))
        assert len(sel) and len(xs) >= 5
        assert np.round(sel[:, 0], 1).min() == pytest.approx(-0.6, abs=0.05)


def test_three_horns_on_the_roof(parts):
    h = parts["ext_horns"]
    lo, hi = h.bbox()
    assert lo[2] > L.CAB_ROOF_Z - 0.03 and hi[2] < L.CAB_ROOF_Z + 0.2
    assert L.CAB_FRONT_Y < lo[1] and hi[1] < L.BOC_Y
    bells = h.V[h.V[:, 1] < CAB.HORNS["y"] + 0.006]
    assert len(np.unique(np.round(bells[:, 0], 1))) >= 3


# ======================================================================================== along the frame
def test_the_tanks_are_round_on_the_layout(parts):
    t = parts["ext_tanks"]
    shell = points_of(t, "alloy")
    for sx in (1.0, -1.0):
        mine = shell[shell[:, 0] * sx > 0]
        mid = mine[(mine[:, 1] > L.TANK_Y[0] + 0.1) & (mine[:, 1] < L.TANK_Y[1] - 0.1)]
        r = np.hypot(mid[:, 0] - sx * L.TANK_X, mid[:, 2] - L.TANK_Z)
        assert r.min() == pytest.approx(L.TANK_R, abs=0.001) and r.max() == pytest.approx(L.TANK_R, abs=0.001)
        assert mine[:, 1].min() == pytest.approx(L.TANK_Y[0] - 0.026, abs=0.002)
        assert mine[:, 1].max() == pytest.approx(L.TANK_Y[1] + 0.026, abs=0.002)
    lo, hi = t.bbox()
    assert hi[2] < L.CAB_BOTTOM_Z and lo[2] > 0.3                                # under the rocker, off the ground
    assert np.abs(t.V[:, 0]).min() >= L.FRAME_X                                   # the cradles hang off the rails' webs
    assert np.abs(t.V[:, 0]).max() < 102 * 0.0254 / 2


def test_the_step_boxes_stand_under_the_doors(parts):
    s = parts["ext_steps"]
    lo, hi = s.bbox()
    assert lo[1] >= L.STEP_Y[0] - 0.003 and hi[1] <= L.STEP_Y[1] + 0.003
    tread = points_of(s, "tread_plate")
    levels = np.unique(np.round(tread[:, 2], 2))
    assert tread[:, 2].max() == pytest.approx(CH.STEP["z"][1] - 0.003, abs=0.002)
    assert levels.min() < CH.STEP["lower_z"] < levels.max()                       # the lower tread and the box's top
    assert np.abs(s.V[:, 0]).min() >= L.FRAME_X and np.abs(s.V[:, 0]).max() < 1.2


def test_the_deck_lies_on_the_rails_behind_the_sleeper(parts):
    d = parts["ext_deck"]
    plate = points_of(d, "tread_plate")
    assert plate[:, 1].min() == pytest.approx(L.DECK_Y[0], abs=0.002) and plate[:, 1].max() == pytest.approx(L.DECK_Y[1], abs=0.002)
    assert plate[:, 2].min() == pytest.approx(L.FRAME_Z[1] + 0.001, abs=0.001)


def test_the_glad_hands_and_the_plug_meet_the_vans_couplers(parts):
    d = parts["ext_deck"]
    y_seal = L.TRAILER_Y[0] - 0.06
    for role, sx in (("hose_red", 1.0), ("hose_blue", -1.0)):
        p = points_of(d, role)
        hand = p[p[:, 1] > L.TRAILER_Y[0] - 0.2]
        assert len(hand) and np.abs(hand[:, 0] - sx * CH.GLAD_X).max() < 0.06
        line = p[p[:, 1] < y_seal - 0.1]
        assert line[:, 1].min() < L.SLEEPER_Y[1] + 0.05                          # the line runs from the sleeper's back
    seals = points_of(d, "rubber")
    near = seals[seals[:, 1] > y_seal - 0.02]
    assert near[:, 1].max() == pytest.approx(y_seal, abs=0.001)                  # seal faces face to face with the van's
    assert d.V[:, 1].max() <= L.FRAME_Y[1] + 0.05
    plug = seals[(np.abs(seals[:, 0]) < 0.03) & (seals[:, 1] > L.TRAILER_Y[0] - 0.1)]
    assert plug[:, 1].max() == pytest.approx(L.TRAILER_Y[0] - 0.07, abs=0.001)


def test_the_tail_lamps_are_red_at_the_frames_end(parts):
    red = points_of(parts["ext_lamps"], "lamp_red")
    assert red[:, 1].min() > L.FRAME_Y[1] and red[:, 1].max() < L.FRAME_Y[1] + 0.1
    assert set(np.sign(np.round(red[:, 0], 3))) == {-1.0, 1.0}


# ======================================================================================== following the shells
class _Shifted:
    """A probe on the stand-in moved by `d`: the body as if TruckBody had built it there."""

    def __init__(self, mesh, d):
        self.p, self.d = S.Probe.from_mesh(mesh.moved(d)), np.asarray(d, float)

    def ray(self, o, v):
        return self.p.ray(o, v)


def test_parts_follow_the_shells_they_are_probed_on():
    base = E.static_parts()
    shells = K.standin_shells()
    moved = E.static_parts({"cab": _Shifted(shells["cab"], (0.0, 0.0, 0.05)), "hood": _Shifted(shells["hood"], (0.0, -0.02, 0.0)),
                            "sleeper": _Shifted(shells["sleeper"], (0.0, 0.0, 0.0))})
    # the visor lamps ride 5 cm higher with the cab's roof, the stacks and tanks stay where the layout puts them
    a0, a1 = points_of(base["ext_lamps"], "lamp_amber"), points_of(moved["ext_lamps"], "lamp_amber")
    vis0 = a0[(a0[:, 1] < -5.0) & (a0[:, 2] > 2.6)][:, 2].max()
    vis1 = a1[(a1[:, 1] < -5.0) & (a1[:, 2] > 2.6)][:, 2].max()
    assert vis1 - vis0 == pytest.approx(0.05, abs=0.002)
    assert np.allclose(CH.stack_pipe(1.0).V, CH.stack_pipe(1.0).V)
    assert np.abs(moved["ext_tanks"].V - base["ext_tanks"].V).max() < 1e-12
    # the headlamp buckets reach back to the fender's face wherever it is
    h0, h1 = base["ext_lamps"], moved["ext_lamps"]
    assert h1.V[:, 1].min() == pytest.approx(h0.V[:, 1].min(), abs=1e-9)          # the lens plane is the layout's
