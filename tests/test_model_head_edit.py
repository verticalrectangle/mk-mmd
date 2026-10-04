"""Face edits of the head part (bpy-free): the eye tilt keeps blinks closing, the jaw edit moves only the lower cheeks, the fang rides the
lip, the highlight narrows, and the texture ops (warp, almond) reshape a painted shape. All on tiny synthetic data."""
import numpy as np
import pytest

from mkmmd.model import part as P, pmx_recolor as RC, pmx_take as T
from mkmmd.model.parts import head_edit as HE


def piece(verts, weights=None, morphs=None, faces=None, normals=None):
    V = np.asarray(verts, float)
    n = len(V)
    return T.Piece(["m"], V, faces or [], np.zeros((0, 2)), np.tile([0.0, -1.0, 0.0], (n, 1)) if normals is None else np.asarray(normals, float),
                   np.zeros(0, int), weights or {}, morphs or {}, np.arange(n), np.zeros((n, 2)))


CENTRE, WIDTH = (0.045, 1.26), 0.05            # eye centre (x, z) and width of the left eye


def ring(c, r, n=24, y=-0.09):
    a = 2 * np.pi * np.arange(n) / n
    return np.stack([c[0] + r * np.cos(a), np.full(n, y), c[1] + r * np.sin(a)], -1)


# ------------------------------------------------------------------------------------------------------------------ eyes
def test_eye_field_is_one_inside_zero_outside_and_never_overlaps_the_other_eye():
    pts = np.array([[CENTRE[0] + d, -0.09, CENTRE[1]] for d in (0.0, 0.7 * WIDTH, 1.0 * WIDTH, 1.3 * WIDTH, 2.0 * WIDTH)])
    w = HE.eye_field(pts, CENTRE, WIDTH, +1.0)
    assert w[0] == pytest.approx(1.0) and w[1] == pytest.approx(1.0)
    assert 0.0 < w[2] < 1.0 and w[3] == pytest.approx(0.0, abs=1e-12) and w[4] == 0.0
    assert w[1] >= w[2] >= w[3]
    mid = np.array([[0.0, -0.09, CENTRE[1]], [-0.03, -0.09, CENTRE[1]], [0.01, -0.09, CENTRE[1]], [0.02, -0.09, CENTRE[1]]])
    assert HE.eye_field(mid[:2], CENTRE, WIDTH, +1.0).max() == 0.0                  # the midline and the other side are untouched
    f = HE.eye_field(mid[2:], CENTRE, WIDTH, +1.0)
    assert 0.0 < f[0] < f[1] < 1.0                                                  # a smooth fade towards the midline
    assert HE.eye_field(mid[1:2], (-CENTRE[0], CENTRE[1]), WIDTH, -1.0)[0] > 0.9     # and the mirrored eye owns the other side


def test_the_outer_corner_goes_up_for_both_eyes():
    out_l = np.array([[CENTRE[0] + 0.02, -0.09, CENTRE[1]]])
    out_r = np.array([[-CENTRE[0] - 0.02, -0.09, CENTRE[1]]])
    pl, pr = piece(out_l), piece(out_r)
    HE.tilt_eyes({"skin": pl}, "skin", {"L": (CENTRE, WIDTH)}, dict(tilt_deg=4.0), None)
    HE.tilt_eyes({"skin": pr}, "skin", {"R": ((-CENTRE[0], CENTRE[1]), WIDTH)}, dict(tilt_deg=4.0), None)
    assert pl.verts[0, 2] - CENTRE[1] == pytest.approx(0.02 * np.sin(np.radians(4.0)), abs=1e-9)
    assert pr.verts[0, 2] - CENTRE[1] == pytest.approx(0.02 * np.sin(np.radians(4.0)), abs=1e-9)         # mirrored: the outer corner is up too
    assert np.hypot(pl.verts[0, 0] - CENTRE[0], pl.verts[0, 2] - CENTRE[1]) == pytest.approx(0.02)       # a rotation: distances stay


def test_points_beyond_the_reach_do_not_move_and_the_iris_is_not_in_the_listed_pieces():
    far = np.array([[CENTRE[0] + 1.4 * WIDTH, -0.09, CENTRE[1]], [CENTRE[0], -0.09, CENTRE[1] + 1.5 * WIDTH]])
    iris = piece(ring(CENTRE, 0.012))
    skin = piece(np.vstack([far, ring(CENTRE, 0.03)]))
    before = (skin.verts.copy(), iris.verts.copy())
    HE.tilt_eyes({"skin": skin, "iris": iris}, "skin", {"L": (CENTRE, WIDTH)}, dict(tilt_deg=4.0, pieces=["white"]), None)
    assert np.array_equal(skin.verts[:2], before[0][:2])                              # beyond 1.3 widths
    assert not np.array_equal(skin.verts[2:], before[0][2:])
    assert np.array_equal(iris.verts, before[1])                                      # not listed: stays round and untilted


def test_vertices_weighted_to_the_brow_bone_stay_put():
    v = ring(CENTRE, 0.03, 8)
    brow = np.zeros(8)
    brow[:4] = 1.0
    p = piece(v, weights={"brow": brow, "head": 1.0 - brow})
    HE.tilt_piece(p, CENTRE, WIDTH, +1.0, np.radians(4.0), {}, "brow")
    assert np.array_equal(p.verts[:4], v[:4]) and not np.allclose(p.verts[4:], v[4:])


def test_morph_offsets_turn_with_the_vertices_so_a_blink_still_closes_on_itself():
    rng = np.random.default_rng(5)
    up = np.stack([CENTRE[0] + rng.uniform(-0.2, 0.2, 12) * WIDTH, np.full(12, -0.09), CENTRE[1] + rng.uniform(0.01, 0.06, 12) * WIDTH * 0.5], -1)
    lo = up.copy()
    lo[:, 2] = CENTRE[1] - 0.004                                                       # the lower lid they close onto
    blink = lo - up                                                                    # upper lid vertices travel down to the lower lid
    pu = piece(up, morphs={"blink": blink})
    pl = piece(lo, morphs={"blink": np.zeros_like(blink)})
    HE.tilt_eyes({"u": pu, "l": pl}, "u", {"L": (CENTRE, WIDTH)}, dict(tilt_deg=4.0, pieces=["l"]), None)
    closed_u = pu.verts + pu.morphs["blink"]
    assert np.allclose(closed_u, pl.verts, atol=1e-9)                                  # no gap, nothing white showing
    assert not np.allclose(pu.morphs["blink"], blink)                                  # the offsets did turn
    assert np.allclose(np.linalg.norm(pu.morphs["blink"], axis=1), np.linalg.norm(blink, axis=1))


def test_normals_turn_too_and_a_zero_tilt_changes_nothing():
    v = ring(CENTRE, 0.02, 6)
    n = np.tile([0.0, -0.6, 0.8], (6, 1))
    p = piece(v, normals=n)
    HE.tilt_eyes({"s": p}, "s", {"L": (CENTRE, WIDTH)}, dict(tilt_deg=4.0), None)
    assert np.allclose(np.linalg.norm(p.normals, axis=1), 1.0, atol=1e-9) and not np.allclose(p.normals, n)
    q = piece(v)
    HE.tilt_eyes({"s": q}, "s", {"L": (CENTRE, WIDTH)}, dict(tilt_deg=0.0), None)
    assert np.allclose(q.verts, v)


# ------------------------------------------------------------------------------------------------------------------- jaw
def face_grid():
    """A coarse frontal face surface: x -0.09..0.09, z 1.15..1.33, y bulging forward in the middle; normals point forward and out."""
    xs, zs = np.linspace(-0.09, 0.09, 37), np.linspace(1.17, 1.33, 33)
    X, Z = np.meshgrid(xs, zs)
    hw = 0.03 + 0.06 * np.clip((Z - 1.17) / 0.12, 0.0, 1.0)                              # a V chin widening to the temples
    Xf = np.clip(X, -hw, hw)
    Y = -0.08 - 0.02 * (1 - (Xf / hw) ** 2)
    V = np.stack([Xf.ravel(), Y.ravel(), Z.ravel()], -1)
    N = np.tile([0.0, -1.0, 0.0], (len(V), 1))
    N[:, 0] = 0.4 * np.sign(V[:, 0]) * np.abs(V[:, 0]) / 0.09
    N /= np.linalg.norm(N, axis=1, keepdims=True)
    return V, N


def test_jaw_weights_are_zero_at_the_midline_the_eyes_the_neck_and_the_back():
    V, N = face_grid()
    w = HE.jaw_weights(V, N, eye_bottom=1.30, neck_z=1.17, centre_y=0.005, cfg={})
    mid = np.abs(V[:, 0]) < 0.005
    assert w[mid].max() == 0.0                                                         # nose / lips / chin point strip
    assert w[V[:, 2] > 1.29].max() == 0.0                                              # at and just under the eyes
    assert w[V[:, 2] < 1.173].max() < 1e-3                                             # at the neck (neck_z + the 4 mm gap)
    assert w[V[:, 2] > 1.20][np.abs(V[V[:, 2] > 1.20][:, 0]) > 0.05].max() > 0.99      # and the full weight above its ramp
    cheek = (np.abs(V[:, 0]) > 0.05) & (V[:, 2] > 1.2) & (V[:, 2] < 1.26)
    assert w[cheek].max() == pytest.approx(1.0, abs=1e-6) and w[cheek].mean() > 0.5
    behind = V.copy()
    behind[:, 1] = 0.05
    assert HE.jaw_weights(behind, N, 1.30, 1.17, 0.005, {}).max() == 0.0               # nothing on the back of the head
    assert np.all((w >= 0) & (w <= 1))


def test_soften_jaw_moves_along_normals_by_the_amount_and_leaves_morph_offsets_alone():
    V, N = face_grid()
    off = np.random.default_rng(1).normal(size=V.shape) * 1e-3
    p = piece(V, morphs={"a": off.copy()}, normals=N)
    w = HE.soften_jaw(p, 1.30, 1.17, 0.005, dict(amount=0.002))
    d = p.verts - V
    assert np.allclose(np.linalg.norm(d, axis=1), 0.002 * w, atol=1e-12)
    assert np.allclose(d, 0.002 * w[:, None] * N, atol=1e-12)
    assert np.array_equal(p.morphs["a"], off)
    assert w.max() == pytest.approx(1.0, abs=1e-6)


# --------------------------------------------------------------------------------------------------------------- highlight
def test_narrow_highlight_scales_the_largest_shape_on_each_side_only():
    def blob(cx, cz, w, h):
        return [[cx - w, -0.09, cz - h], [cx + w, -0.09, cz - h], [cx + w, -0.09, cz + h], [cx - w, -0.09, cz + h]]
    V = np.array(blob(0.04, 1.27, 0.006, 0.009) + blob(0.05, 1.255, 0.002, 0.002) + blob(-0.04, 1.27, 0.006, 0.009) + blob(-0.05, 1.255, 0.002, 0.002))
    faces = [(4 * k, 4 * k + 1, 4 * k + 2, 4 * k + 3) for k in range(4)]
    p = piece(V, faces=faces)
    HE.narrow_highlight(p, 0.8)
    assert np.ptp(p.verts[0:4, 0]) == pytest.approx(0.8 * 0.012) and np.ptp(p.verts[8:12, 0]) == pytest.approx(0.8 * 0.012)
    assert np.allclose(p.verts[4:8], V[4:8]) and np.allclose(p.verts[12:16], V[12:16])             # the small round ones stay
    assert np.allclose(p.verts[0:4, 2], V[0:4, 2]) and p.verts[0:4, 0].mean() == pytest.approx(0.04)


# ------------------------------------------------------------------------------------------------------------------ fang
def teeth_and_lip(faces=True):
    """A teeth mesh, and a skin piece around a closed mouth: a height field whose front surface is y = -0.096 on the lower lip (z <= 1.2065)
    and bulges forward by up to 4 mm over the upper lip above the slit (z = 1.2062 .. 1.2068); the slit's boundary vertices are extra."""
    xs = np.linspace(-0.02, 0.02, 9)
    teeth_v = np.array([[x, -0.09, 1.20] for x in xs] + [[x, -0.09, 1.21] for x in xs])
    tf = [(i, i + 1, 9 + i + 1, 9 + i) for i in range(8)]
    uv = np.array([[0.1 * k + 0.5, 0.2] for f in tf for k in range(len(f))])
    mesh = P.Mesh("t", teeth_v, tf, uv=uv, face_mat=np.zeros(8, int), mats=["t"], weights={"頭": np.linspace(0.2, 1.0, 18)},
                  normals=np.tile([0.0, -1.0, 0.0], (18, 1)), morphs={"keep": np.zeros((18, 3))})
    gx, gz = np.meshgrid(np.linspace(-0.03, 0.03, 31), np.linspace(1.19, 1.22, 31))
    gy = -0.096 - 0.004 * np.clip((gz - 1.2068) / 0.004, 0.0, 1.0)
    G = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], -1)
    faces_g = [(j * 31 + i, j * 31 + i + 1, (j + 1) * 31 + i + 1, (j + 1) * 31 + i) for j in range(30) for i in range(30)]
    a = 2 * np.pi * np.arange(16) / 16                                                  # the slit: an ellipse, 0.6 mm tall
    lip = np.stack([0.02 * np.cos(a), np.full(16, -0.0958), 1.2065 + 0.0003 * np.sin(a)], -1)
    V = np.vstack([G, lip])
    loop = len(G) + np.arange(16)
    morphs = {"a": np.zeros((len(V), 3)), "o": np.zeros((len(V), 3)), "eyes": np.zeros((len(V), 3))}
    morphs["a"][loop] = [0.0, -0.001, 0.003]
    morphs["o"][loop] = [0.0, -0.002, 0.0]
    skin = piece(V, faces=faces_g if faces else [], morphs=morphs)
    return mesh, skin, loop


CFG = dict(side="L", x=0.012, width=0.004, length=0.0028, peek=0.001, over=0.0004, tuck=0.0012, outline_width=0.0005, reveal_down=0.0014,
           reveal_forward=0.0003, reveal={"o": 1.0, "eyes": 0.8})


def test_the_fang_hangs_below_the_upper_lip_in_front_of_the_lower_lip_with_a_pink_outline():
    teeth, skin, loop = teeth_and_lip()
    f = HE.make_fang(skin, loop, ["a", "o"], CFG, teeth, ["teeth", "line"])
    assert f.name == "fang" and f.mats == ["teeth", "line"] and list(f.face_mat) == [0, 1] and len(f.verts) == 6 and len(f.faces) == 2
    edge_z = 1.2065 + 0.0003 * np.sin(np.pi / 4)                                        # the upper-lip vertex nearest x = 0.012 (k = 2 of 16)
    tip = f.verts[2]
    assert tip[0] == pytest.approx(0.012) and tip[2] == pytest.approx(edge_z - 0.001, abs=1e-9)
    assert tip[1] == pytest.approx(-0.096 - 0.0004, abs=1e-6)                           # 0.4 mm in front of the lower lip's surface
    assert f.verts[0, 2] == pytest.approx(tip[2] + 0.0028) and f.verts[0, 1] == pytest.approx(tip[1] + 0.0012)     # the base is tucked behind and above
    assert f.verts[0, 0] == pytest.approx(0.012 - 0.002) and f.verts[1, 0] == pytest.approx(0.012 + 0.002)
    # the outline is a larger triangle a hair behind the white one
    area = lambda a, b, c: 0.5 * np.linalg.norm(np.cross(b - a, c - a))
    assert area(*f.verts[3:6]) > 1.3 * area(*f.verts[0:3]) and np.all(f.verts[3:6, 1] > f.verts[0:3, 1])
    for tri in f.faces:                                                                 # both face forward (-y), like the teeth
        n = np.cross(f.verts[tri[1]] - f.verts[tri[0]], f.verts[tri[2]] - f.verts[tri[0]])
        assert n[1] < 0
    # weights and uv come from the nearest tooth vertex
    near = int(np.argmin(np.linalg.norm(teeth.verts - tip, axis=1)))
    assert np.allclose(f.weights["頭"], teeth.weights["頭"][near]) and f.uv.shape == (6, 2) and f.normals.shape == (6, 3)


def test_the_fang_rides_the_lip_in_every_morph_and_peeks_more_where_it_should():
    teeth, skin, loop = teeth_and_lip()
    f = HE.make_fang(skin, loop, ["a", "o", "eyes", "unmoved"], CFG, teeth, ["t", "l"])
    donor_off = np.array([0.0, -0.001, 0.003])
    assert np.allclose(f.morphs["a"], donor_off)                                          # a rigid copy of the upper-lip vertex's offset, all six corners
    reveal = np.array([0.0, -0.0003, -0.0014])
    o = f.morphs["o"]                                                                     # donor offset + the reveal on the two tips only
    assert np.allclose(o[[0, 1, 3, 4]], [0.0, -0.002, 0.0]) and np.allclose(o[[2, 5]], np.array([0.0, -0.002, 0.0]) + reveal)
    e = f.morphs["eyes"]                                                                  # not moved by the lip, still revealed (factor 0.8)
    assert np.allclose(e[[0, 1, 3, 4]], 0.0) and np.allclose(e[[2, 5]], 0.8 * reveal)
    assert "unmoved" not in f.morphs and set(f.morphs) == {"a", "o", "eyes"}


def test_the_fang_goes_to_the_requested_side_and_works_without_skin_faces():
    teeth, skin, loop = teeth_and_lip(faces=False)
    f = HE.make_fang(skin, loop, [], dict(CFG, side="R"), teeth, ["t", "l"])
    assert f.verts[2, 0] == pytest.approx(-0.012)
    assert f.verts[2, 1] == pytest.approx(-0.0958 - 0.0004, abs=2e-5)                     # no surface to cast onto: the lip edge's own depth


# --------------------------------------------------------------------------------------------------------------- texture ops
def test_warp_narrows_a_painted_blob_about_its_centre_and_leaves_the_rim_alone():
    img = np.zeros((100, 200, 4), np.uint8)
    img[..., 3] = 255
    img[30:70, 70:130, :3] = 255                                                       # a 60 x 40 white blob
    out = RC.recolor(img, [dict(warp=dict(centre=[0.5, 0.5], radius=[0.4, 0.5], sx=0.5, sy=1.0))])
    w_before = (img[50, :, 0] > 128).sum()
    w_after = (out[50, :, 0] > 128).sum()
    assert 0.45 * w_before < w_after < 0.65 * w_before
    assert (out[50, 100, :3] > 200).all() and abs(np.nonzero(out[50, :, 0] > 128)[0].mean() - 99.5) < 1.5          # still centred
    assert np.array_equal(out[:, :10], img[:, :10]) and (out[..., 3] == 255).all()


def test_almond_makes_a_pointed_vertical_lens_inside_its_rect_only():
    img = np.zeros((200, 200, 4), np.uint8)
    img[..., 3] = 255
    img[..., 0] = 200
    out = RC.recolor(img, [dict(rect=[0.25, 0.1, 0.75, 0.9], almond=dict(centre=[0.5, 0.5], half=[0.1, 0.35], soft=0.3, tip=1.7))])
    a = out[..., 3].astype(int)
    assert a[100, 100] == 255 and a[100, 128] == 0 and a[100, 72] == 0                   # 0.1 wide each side at the middle (20 px), nothing at 28 px
    assert a[100 - 30, 106] == 255 and a[100 - 60, 106] == 0 and a[100 - 60, 100] == 255   # narrows towards the points (the axis stays inside)
    assert a[100, 40] == 255 and a[10, 100] == 255                                         # outside the rect: untouched
    assert (a[:, :, None] == out[..., 3:4]).all() and out[100, 128, 0] == 200
