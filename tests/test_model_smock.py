"""The smock's cuts and the dress parts' helpers (mkmmd/model/parts/smock.py, bubble_skirt.py, zori.py, outfit_rig.py):
pure functions, no build."""
import numpy as np
import pytest

from mkmmd.model.parts import bubble_skirt as BS
from mkmmd.model.parts import outfit_rig as ORIG
from mkmmd.model.parts import smock as SM
from mkmmd.model.parts import zori as ZO


def test_tail_tube_cuts_only_round_its_path():
    # a tail leaving her back at 40 degrees: the cloth round it is cut, the cloth on her front (on the line of the
    # path carried on backwards through her) is not (that line once cut a hole in her front)
    a = np.radians(40.0)
    root = np.array([0.0, 0.027, 0.837])
    path = [root + s * np.array([0.0, np.cos(a), np.sin(a)]) for s in (0.0, 0.1, 0.2)]
    tb = SM.tube(path, [[0.04, 0.036], [0.06, 0.055], [0.09, 0.08]])
    on_path = root + 0.1 * np.array([0.0, np.cos(a), np.sin(a)])
    front = root - 0.15 * np.array([0.0, np.cos(a), np.sin(a)])
    d = SM.tube_field(np.array([on_path, on_path + [0.0, 0.0, 0.2], front]), tb)
    assert d[0] < 0.0 < d[1]
    assert d[2] > 0.1


def test_neckline_keeps_the_shoulders_and_cuts_the_neck():
    # the plane through the neck's base: over it, the cloth on her neck goes and the shoulder's top stays
    r = np.full(96, 0.04)
    axis, n = np.array([0.0, 0.0, 1.14]), np.array([0.0, 0.0, 1.0])
    pts = np.array([[0.0, -0.03, 1.15],          # on the neck's front, over the plane: cut
                    [0.07, 0.0, 1.15],           # the top of her shoulder, over the plane: kept
                    [0.0, -0.03, 1.12]])         # under the plane: kept
    f = SM.neck_field(pts, axis, n, r)
    assert f[0] < 0.0 and f[1] > 0.0 and f[2] > 0.0


def test_hang_ring_takes_each_turn_its_own_radius():
    # cloth hanging 0.10 from the axis at her front and back and 0.15 at her sides: the hem's ring (where the lace is
    # sewn) must be 0.15 out at her side, not the front's 0.10 (the lace once sat inside the cloth at her sides)
    th = np.linspace(0.0, 2.0 * np.pi, 192, endpoint=False)
    R = np.tile(0.10 + 0.05 * np.sin(th) ** 2, (3, 1))
    zs = np.array([0.90, 0.85, 0.80])
    ring = SM.hang_ring(zs, R, 0.0, np.array([0.0, 0.5 * np.pi]), 0.85)
    assert abs(np.hypot(*ring[0, :2]) - 0.10) < 1e-6
    assert abs(ring[1, 0] - 0.15) < 1e-6


def test_lantern_creases_are_where_the_panels_meet():
    c = {"segments": {"count": 16}}
    assert BS.panels(np.array([0.0]), c)[0] > 0.999          # a panel's crest at her front middle
    k = BS.creases(c)
    assert len(k) == 16 and BS.panels(k, c).max() < 1e-6      # the painted creases sit in the folds


def test_ring_takes_a_firmer_physics():
    rig = ORIG.Rig()
    names = [[f"x{c}_{r}" for c in range(3)] for r in range(2)]
    pts = np.array([[[np.sin(t) * 0.2, -np.cos(t) * 0.2, z] for z in (0.8, 0.7, 0.6)]
                    for t in np.linspace(0.0, 2 * np.pi, 3, endpoint=False)])
    rig.ring(names, pts, "下半身", "skirt", 0.02, anchor_body="下半身", physics={"spring": (24.0, 24.0, 12.0)})
    chain_joints = [j for j in rig.joints if j.name.count("_") == 2]          # J_<bone>; the ties are J_<a>_<b>
    assert len(chain_joints) == 6 and all(j.spring_rot == (24.0, 24.0, 12.0) for j in chain_joints)


def test_print_motif_lands_where_it_is_set():
    col = {"base": np.array([0.64, 0.25, 0.28]), "print": np.array([0.79, 0.66, 0.47]),
           "print_shadow": np.array([0.6, 0.48, 0.33]), "shadow": np.zeros(3), "light": np.ones(3)}
    c = {"print": {"motifs": [{"kind": "spiral", "at": 0.0, "z": 0.0, "size": 0.15}], "stroke": 0.02, "soft": 0.001,
                   "line": 0.0015}}
    img, mpp = BS.picture(col, c, 1.3, 0.5, lambda z: 0.25, 0.0, 1.0)
    rgb = img[..., :3] / 255.0
    front = rgb[:, BS.PX // 2 - 60:BS.PX // 2 + 60]          # her front (u 0.5): the spiral is there
    back = np.concatenate([rgb[:, :40], rgb[:, -40:]], 1)    # the seam at her back: plain ground
    assert np.abs(front - col["print"]).sum(-1).min() < 0.05
    assert np.abs(back - col["base"]).sum(-1).max() < 0.02


def _covered_columns(img, colour):
    rgb = img[..., :3] / 255.0
    return np.flatnonzero((np.abs(rgb - colour).sum(-1) < 0.05).any(0))


def test_a_motif_keeps_its_width_where_the_cloth_is_wider_round_her():
    """A flared skirt is twice as wide round her at its hem as under the smock: a motif painted where the cloth is
    wider takes half as many of the picture's columns, so on the cloth it is as wide as it was drawn."""
    col = {"base": np.array([0.45, 0.10, 0.23]), "print": np.array([0.65, 0.70, 0.29]),
           "print_shadow": np.array([0.43, 0.48, 0.17]), "shadow": np.zeros(3), "light": np.ones(3)}
    c = {"print": {"motifs": [{"kind": "spiral", "at": 0.0, "z": 0.0, "size": 0.15}], "stroke": 0.02, "soft": 0.001,
                   "line": 0.0015}}
    narrow, _ = BS.picture(col, c, 1.3, 0.5, lambda z: 0.25, 0.0, 1.0)
    wide, _ = BS.picture(col, c, 1.3, 0.5, lambda z: 0.25, 0.0, 1.0, width_of=lambda d: np.full(np.shape(d), 2.6))
    a, b = _covered_columns(narrow, col["print"]), _covered_columns(wide, col["print"])
    assert 0.4 < np.ptp(b) / np.ptp(a) < 0.6


def test_a_flame_rises_where_it_is_set_in_its_own_colour():
    col = {"base": np.array([0.45, 0.10, 0.23]), "print": np.array([0.65, 0.70, 0.29]),
           "print_shadow": np.array([0.43, 0.48, 0.17]), "accent": np.array([0.81, 0.60, 0.32]), "accent_shadow": None,
           "shadow": np.zeros(3), "light": np.ones(3)}
    c = {"print": {"motifs": [{"kind": "flame", "at": 0.0, "z": 0.0, "size": 0.2, "color": "accent"}], "stroke": 0.02,
                   "soft": 0.001, "line": 0.0015}}
    img, mpp = BS.picture(col, c, 1.3, 0.5, lambda z: 0.25, 0.0, 1.0)
    rgb = img[..., :3] / 255.0
    hit = np.abs(rgb - col["accent"]).sum(-1) < 0.05
    rows = np.flatnonzero(hit.any(1))
    assert hit[:, BS.PX // 2 - 40:BS.PX // 2 + 40].any()                 # at her front
    assert (0.25 - rows.min() * mpp) > 0.04 and (rows.max() * mpp - 0.25) > 0.10   # it rises from well under its middle
    with pytest.raises(ValueError, match="kind"):
        BS.picture(col, {"print": {**c["print"], "motifs": [{"kind": "wave", "at": 0.0, "z": 0.0}]}}, 1.3, 0.5,
                   lambda z: 0.25, 0.0, 1.0)


def test_sole_outline_is_wider_than_the_foot_all_round():
    rng = np.random.default_rng(1)
    foot = np.column_stack([rng.uniform(-0.03, 0.03, 300), rng.uniform(-0.13, 0.03, 300)])
    ol = ZO.outline(foot, 0.005, 0.01)
    seg = np.roll(ol, -1, 0) - ol
    # every point of the foot inside the outline (counter-clockwise: left of every edge) and at least ~5 mm in
    cross = seg[None, :, 0] * (foot[:, None, 1] - ol[None, :, 1]) - seg[None, :, 1] * (foot[:, None, 0] - ol[None, :, 0])
    assert (cross > 0).all() or (cross < 0).all()
    d = np.min(np.abs(cross) / np.linalg.norm(seg, axis=1)[None], axis=1)
    assert d.min() > 0.0045
