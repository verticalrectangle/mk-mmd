"""The paper plane, bpy-free (mkmmd.core.paper_plane): the folded sheet reads as a dart (keel under level wings, the
leading edge two plies, the tips outermost), she sits on its spine with her legs over the wing, the bank and pitch turn
it about its own axes and carry her, and its keys hold on the paper theatre's step grid, landing ON their times."""
import math

import numpy as np
import pytest

from mkmmd.core import paper_plane as PL
from mkmmd.core import paper_theatre as PT
from mkmmd.core import shell as S


def test_the_folded_sheet_is_a_dart_with_its_plies():
    O = PL.options({})
    F = PL.Fold()
    P = PL.parts()
    for key, m in P.items():
        assert m.is_closed() and m.volume() > 0, key
    lo, hi = np.min([m.V.min(0) for m in P.values()], 0), np.max([m.V.max(0) for m in P.values()], 0)
    assert hi[1] - lo[1] == pytest.approx(O["size"], rel=0.05)                 # as long as the sheet
    assert lo[1] == pytest.approx(-0.5 * O["size"], abs=0.01)                  # the nose at the front, -Y
    body, wing = P["body"].V, P["wing_L"].V
    assert body[:, 2].min() < -0.1 * O["size"]                                  # the keel hangs under the wings...
    assert body[body[:, 2] < -0.08 * O["size"], 1].min() > 0.0                  # ...deepest toward the tail
    assert np.ptp(wing[:, 2]) < 4.0 * F.t                                       # the wings level, a few plies thick
    for side in "LR":
        w = P[f"wing_{side}"].V
        far = np.abs(w[:, 0]).max()
        assert abs(PL.tip(F, side)[0]) == pytest.approx(far, abs=F.t + F.g)    # the tip is the wing's outermost point
    bead = PL._bead(F)                                                          # the leading edge: the sheet turned back
    assert np.ptp(bead.V[:, 2]) == pytest.approx(2.0 * F.t + F.g, rel=1e-6)    # over itself, two plies thick
    assert PL.MI["edge"] in bead.Qm and PL.MI["inner"] in bead.Qm


def test_she_sits_on_the_spine_side_saddle_her_feet_below_the_wing():
    F = PL.Fold()
    P = PL.parts()
    probe = S.Probe.from_mesh(S.merge(list(P.values())))
    for side, sx in (("L", 1.0), ("R", -1.0)):
        st = PL.seat(F, PL.options({"side": side}))
        hip = st["hip"]
        for dx, dy in ((0.03, 0.0), (-0.03, 0.0), (0.04, 0.06), (-0.04, -0.06)):  # the paper's top under her (her hips
            z = probe.height(hip[0] + dx, hip[1] + dy)                          # straddle the slit along the spine)
            assert z == pytest.approx(st["seat_z"], abs=1e-6)
        assert hip[2] - st["seat_z"] == pytest.approx(PL.HIP)
        f = st["facing"]
        assert sx * f[0] > 0.9                                                  # sideways, over the `side` wing, the
        assert (sx * np.array([f[1], -f[0]]))[1] < -0.9                         # nose on her right (L) or her left (R)
        _d, o = F.le_dir()
        o = np.array([sx * o[0], o[1]])
        nose = F.N3[:2]
        assert (hip[:2] - nose) @ o == pytest.approx(-PL.REACH, abs=1e-6)      # the leading edge REACH ahead of her
        for k in "LR":                                                          # her feet past it, below the wing
            foot = st["feet"][k]
            assert (foot[:2] - nose) @ o > 0.1
            assert foot[2] < -0.3
    assert 0.3 < PL.seat_b(F) / F.L < 0.7


def test_the_colliders_hold_her_hair_on_the_paper_and_end_at_its_edges():
    F = PL.Fold()
    C = PL.colliders(F)
    poly = PL.outline(F)
    card = PL.card("plane", PL.options({}), PL.ROLES)
    assert {c["object"] for c in card["colliders"]} == {f"plane_col_{k}" for k in C}
    assert all(c["type"] == "box" for c in card["colliders"])
    xs, ys = np.meshgrid(np.linspace(-0.5, 0.5, 101), np.linspace(-0.8, 0.85, 166))
    grid = np.stack([xs.ravel(), ys.ravel()], 1)
    paper = PT.inside(grid, poly) & (PT.edge_distance(grid, poly) > 0.02)
    covered = np.zeros(len(grid), bool)
    for c, R, size in C.values():
        top = c[2] + 0.5 * size[2]
        assert top == pytest.approx(2.0 * F.t + F.g)                           # its top on the paper's top
        h = 0.5 * size
        corners = np.array([c + R @ [sx * h[0], sy * h[1], h[2]] for sx in (-1, 1) for sy in (-1, 1)])[:, :2]
        out = ~PT.inside(corners, poly)                                        # never past the paper's edge (where
        assert np.all(PT.edge_distance(corners[out], poly) < 1e-6)             # hair would lie on air): on it at most
        q = (grid - c[:2]) @ R[:2, :2]
        covered |= (np.abs(q[:, 0]) <= h[0]) & (np.abs(q[:, 1]) <= h[1])
    for side in "LR":
        st = PL.seat(F, PL.options({"side": side}))
        assert st["seat_z"] == pytest.approx(2.0 * F.t + F.g)
        rel = grid - st["hip"][:2]
        dist = np.hypot(rel[:, 0], rel[:, 1])
        assert covered[paper & (dist < 0.35)].all()                            # the paper round her seat, and behind
        assert covered[paper & (dist < 0.6) & (rel @ st["facing"][:2] < 0)].all()   # her where her long hair falls


def test_bank_and_pitch_turn_it_about_its_own_axes_and_carry_her():
    O = PL.options({"bank": 30.0})
    F = PL.Fold()
    level = PL.tips(PL.options({"flutter": 0}), [0.0])[0]
    for yaw in (0.0, 0.5 * math.pi):                                           # whatever its heading...
        banked = PL.tips(PL.options({"bank": 30.0, "flutter": 0}), [0.0], yaw=yaw)[0]
        drop = level[0, 0] * math.sin(math.radians(30.0))                       # ...+bank rolls its left wing down
        assert banked[0, 2] == pytest.approx(level[0, 2] - drop, abs=0.01)     # about its own long axis
        assert banked[1, 2] == pytest.approx(level[1, 2] + drop, abs=0.01)
    up = PL.root_matrix(0.0, math.radians(20.0))
    nose = up[:3, :3] @ F.N3
    assert nose[2] > 0.2 and nose[1] < 0                                        # +pitch lifts the nose
    st = PL.seat(F, O)
    O = PL.options({"bank": [[19.3, 0.0], [19.75, 22.0], [20.15, -17.0]], "pitch": [[20.0, 0.0], [20.4, 15.0]],
                    "land": 19.05, "flutter": 2.0})
    t = np.arange(18.9, 21.0, 1.0 / 60.0)
    m = PL.motion(O, t)
    paper = np.array([0.0, st["hip"][1], st["seat_z"], 1.0])                   # the paper under her and her hips, both
    hip = np.append(st["hip"], 1.0)                                             # held by the root: she rides every bank,
    for i in range(0, len(t), 7):                                               # pitch and dip
        M = PL.root_matrix(m["bank"][i], m["pitch"][i], 0.3, m["dip"][i], (1.0, 2.0, 3.0))
        a, b = M @ paper, M @ hip
        assert np.linalg.norm(b - a) == pytest.approx(PL.HIP)
        assert (b - a)[:3] / PL.HIP == pytest.approx(M[:3, 2], abs=1e-9)
    assert m["dip"][t < 19.05].max() == 0.0 and m["dip"].min() < -0.5 * O["sink"]


def test_keys_hold_on_the_step_grid_and_land_on_their_times():
    frame0, fps = 361, 60.0
    O = PL.options({"step": 12, "step_from": 19.05, "land": 19.05, "bank": [[19.05, 0.0], [19.55, 30.0]]})
    keys, times, interp = PT.key_times(1, 2227, frame0, fps, O["step"], O["step_from"])
    assert interp == "CONSTANT" and np.allclose(np.diff(keys[1:]), 5.0)        # a pose every 5 frames, held
    m = PL.motion(O, times)
    at = {s: int(np.argmin(np.abs(times - s))) for s in (19.05, 19.05 + 1.0 / 12.0, 19.55)}
    assert times[at[19.55]] == pytest.approx(19.55)
    assert math.degrees(m["bank"][at[19.55]]) == pytest.approx(30.0)          # ON its key's time
    assert m["dip"][at[19.05]] == pytest.approx(0.0) and m["dip"][: at[19.05]].max() == 0.0
    assert m["dip"][at[19.05 + 1.0 / 12.0]] < -0.01                              # the landing gives from its own step
    flex = m["flex_L"][at[19.05]: at[19.05] + 12]
    assert np.ptp(flex) > math.radians(2.0)                                     # the wings flex step to step
    assert PL.moves(PL.options({"flutter": 0})) == {"turn": False, "dip": False, "flex": False}


def test_options_refuse_what_they_cannot_build():
    with pytest.raises(ValueError, match="bank"):
        PL.options({"bank": [[20.0, 10.0], [19.0, 0.0]]})
    with pytest.raises(ValueError, match="side"):
        PL.options({"side": "left"})
    with pytest.raises(ValueError, match="sheet"):
        PL.options({"sheet": 0.9})
