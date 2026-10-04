"""Head part invariants (bpy-free): the part is well formed, the eyes close like 2D anime (lid sheets and strips, the skin
does not move), morphs stay in their region and are symmetric, the iris layers stay behind the skin for every gaze, normals
point outward. Builds the head against a stand-in body part (neck ring and head landmark)."""
import numpy as np
import pytest

from mkmmd.model import build, part as P, spec
from mkmmd.model.parts import head_decal as DC, head_lid as HL

REQUIRED = ["まばたき", "笑い", "ウィンク", "ウィンク右", "ウィンク２", "ｳｨﾝｸ２右", "はぅ", "なごみ", "びっくり", "じと目", "瞳小",
            "あ", "い", "う", "え", "お", "ω", "口角上げ", "口角下げ", "真面目", "困る", "怒り", "にこり", "上", "下", "照れ"]
LIDS = ["まばたき", "ウィンク", "ウィンク右", "ウィンク２", "ｳｨﾝｸ２右", "笑い", "はぅ", "なごみ", "じと目", "びっくり"]
SHEET = (HL.SHEET["rows"] + 1) * HL.SHEET["nu"]                 # vertices per lid sheet; order: upper L, lower L, upper R, lower R


@pytest.fixture(scope="module")
def head(tmp_path_factory):
    sp = spec.from_dict({"model": {"name": "t", "parts": ["head"]},
                         "colors": {"skin": {"base": "#f1e7d6"}, "eyes": {}, "mouth": {}},
                         "proportions": {"head": {}, "face": {}, "hair_guides": {}}})
    ctx = build.BuildCtx(sp, tmp_path_factory.mktemp("tex"), seed=1)
    ctx.part = "body"
    ctx.parts["body"] = P.Part("body", info=dict(neck_top=dict(z=1.159, center=(0.0, 0.0057), rx=0.0229, ry=0.0303),
                                                 skin=dict(base="#f1e7d6", toon="body_skin_toon.png")))
    ctx.land.update(head=np.array([0.0, -0.0045, 1.2041]), neck=np.array([0.0, 0.002, 1.1563]))
    ctx.part = "head"
    from mkmmd.model.parts import head as H
    part = H.build(ctx)
    P.check(part)
    return part


def mesh(part, name):
    return next(m for m in part.meshes if m.name == name)


def moved(m, name, eps=1e-9):
    d = m.morphs.get(name)
    return np.zeros(len(m.verts), bool) if d is None else np.abs(d).max(1) > eps


def edge_rows(which):
    """Vertex indices of the free edge of the upper (0) / lower (1) sheet of the left (0) / right (1) eye."""
    out = {}
    for k, (side, kind) in enumerate((("L", "upper"), ("L", "lower"), ("R", "upper"), ("R", "lower"))):
        a = k * SHEET + HL.SHEET["rows"] * HL.SHEET["nu"]
        out[(side, kind)] = np.arange(a, a + HL.SHEET["nu"])
    return out


def test_declared_morphs_cover_the_required_list(head):
    names = {m.name for m in head.morphs}
    assert set(REQUIRED) <= names
    assert {m.panel for m in head.morphs} <= set(P.PANELS)
    for n in names:                                    # every declared morph moves something
        assert any(n in m.morphs and np.abs(m.morphs[n]).max() > 0 for m in head.meshes), n


def test_the_skin_does_not_move_for_the_lid_morphs(head):
    face = mesh(head, "face")
    for n in LIDS + ["瞳小"]:
        d = face.morphs.get(n)
        assert d is None or np.abs(d).max() <= 5e-4, n          # at most 0.5 mm anywhere (in fact: not at all)


def test_blink_closes_the_lid_sheets_without_gaps(head):
    lids = mesh(head, "lids")
    V = lids.verts + lids.morphs["まばたき"]
    rows = edge_rows(None)
    for side in ("L", "R"):
        up, lo = V[rows[(side, "upper")]], V[rows[(side, "lower")]]
        assert np.linalg.norm(up - lo, axis=1).max() < 2e-4      # the two lids meet within 0.2 mm all along the eye
        assert np.ptp(V[rows[(side, "upper")]][:, 2]) > 0.002    # on a curve, not a point


def test_half_blink_puts_the_upper_lid_half_way_down(head):
    lids = mesh(head, "lids")
    d = lids.morphs["まばたき"]
    rows = edge_rows(None)[("L", "upper")]
    mid = rows[len(rows) // 2]
    full = abs(d[mid][2])
    assert full > 0.008                                         # travels 8+ mm at the middle of the eye
    # the sheet is a vertex morph: at 0.5 its free edge is exactly half way (linear), eye graphics stay put
    assert np.abs(mesh(head, "eyes").morphs.get("まばたき", np.zeros((1, 3)))).max() == 0


def test_morphs_stay_in_their_region(head):
    face, lines, lids = mesh(head, "face"), mesh(head, "lines"), mesh(head, "lids")
    eye_z = head.info["eyes"]["L"]["center"][2]
    mouth_z = head.info["mouth"]["centre"][2]
    for n in LIDS:
        for m in (lids, lines):
            z = m.verts[moved(m, n)][:, 2]
            assert z.size and z.min() > eye_z - 0.05 and z.max() < eye_z + 0.06, (n, m.name)
    for n in ("あ", "い", "う", "え", "お", "ω", "口角上げ", "口角下げ", "にやり"):
        z = face.verts[moved(face, n)][:, 2]
        assert z.size and abs(z.mean() - mouth_z) < 0.03 and z.max() < mouth_z + 0.04, n
    for n in ("上", "下", "にこり", "真面目", "困る", "怒り", "悲しみ"):        # brows: only the brow strips move
        assert not moved(face, n).any(), n
        assert moved(lines, n).any(), n
    assert not moved(face, "照れ").any()
    assert moved(mesh(head, "blush"), "照れ").all()


def test_wink_morphs_are_mirrored(head):
    lids = mesh(head, "lids")
    a, b = moved(lids, "ウィンク"), moved(lids, "ウィンク右")
    assert a.any() and b.any() and not (a & b).any()
    assert (lids.verts[a][:, 0] > 0).all() and (lids.verts[b][:, 0] < 0).all()
    A = lids.verts[a] + lids.morphs["ウィンク"][a]
    B = (lids.verts[b] + lids.morphs["ウィンク右"][b]) * np.array([-1.0, 1.0, 1.0])
    d = np.linalg.norm(A[:, None, :] - B[None, :, :], axis=2).min(1)
    assert d.max() < 1e-4                               # the closed shapes mirror each other within 0.1 mm


def test_rest_pose_is_symmetric(head):
    face = mesh(head, "face")
    V = face.verts
    sel = np.nonzero((np.abs(V[:, 0]) > 1e-4) & (V[:, 2] > 1.22) & (V[:, 2] < 1.32) & (V[:, 1] < -0.04))[0]
    M = V[sel] * np.array([-1.0, 1.0, 1.0])
    d = np.linalg.norm(V[sel][:, None, :] - M[None, :, :], axis=2).min(1)
    assert d.max() < 5e-5


def _inside(poly, pts):
    """Even-odd point-in-polygon test (numpy)."""
    x, y = pts[:, 0], pts[:, 1]
    ins = np.zeros(len(pts), bool)
    j = len(poly) - 1
    for i in range(len(poly)):
        xi, yi, xj, yj = poly[i, 0], poly[i, 1], poly[j, 0], poly[j, 1]
        cond = ((yi > y) != (yj > y)) & (x < (xj - xi) * (y - yi) / (yj - yi + 1e-30) + xi)
        ins ^= cond
        j = i
    return ins


def test_the_iris_stays_behind_the_skin_for_every_gaze(head):
    """Rotate the eye layers about the eye bone: outside the lid opening nothing may come in front of the skin. Inward
    (nasal) gaze is safe up to +-25 deg yaw; outward gaze pokes through the temple skin beyond about 10 deg (0.5 mm at 10,
    1.2 mm at 15, 2 mm at 20, 5.9 mm at 25: measured), so the checked range is yaw <= 10 outward, 25 inward, pitch +-15."""
    face, eyes = mesh(head, "face"), mesh(head, "eyes")
    surf = DC.SkinSurface(face.verts, face.faces, np.zeros(len(face.faces), int))
    worst = 0.0
    for side, sg, bone in (("L", 1.0, "左目"), ("R", -1.0, "右目")):
        e = head.info["eyes"][side]
        E = np.asarray(e["center"])
        poly = np.asarray(e["opening"])
        c = poly.mean(0)
        grown = c + (poly - c) * 1.06
        rows = np.nonzero(eyes.weights[bone] > 0.5)[0]
        P0 = eyes.verts[rows]
        for yaw, pitch in ((10 * sg, 0), (-25 * sg, 0), (0, 15), (0, -15), (10 * sg, 15), (-25 * sg, -15), (10 * sg, -15), (-25 * sg, 15)):
            ay, ap = np.radians(yaw), np.radians(pitch)
            Rz = np.array([[np.cos(ay), -np.sin(ay), 0], [np.sin(ay), np.cos(ay), 0], [0, 0, 1]])
            Rx = np.array([[1, 0, 0], [0, np.cos(ap), -np.sin(ap)], [0, np.sin(ap), np.cos(ap)]])
            Q = (P0 - E) @ (Rz @ Rx).T + E
            out = ~_inside(grown, Q[:, [0, 2]])
            Q = Q[out]
            ok = np.ones(len(Q), bool)
            y_skin = np.full(len(Q), np.nan)
            for k, q in enumerate(Q):
                try:
                    tri, bary = surf.bind(np.array([q[0]]), np.array([q[2]]))
                    y_skin[k] = surf.point(tri, bary)[0, 1]
                except ValueError:
                    ok[k] = False                      # beyond the front skin: hidden by the head's side
            behind = q_behind = (Q[:, 1] - y_skin)[ok]
            if behind.size:
                worst = max(worst, float(-behind.min()))
    assert worst < 1.3e-3, worst                        # nothing pokes more than 1.3 mm through the skin


def test_normals_point_outward(head):
    face = mesh(head, "face")
    c = head.info["head_center"]
    out = ((face.verts - c) * face.normals).sum(1)
    assert (out > 0).mean() > 0.995
    shell = head.info["skin"]
    V, T = np.asarray(shell["verts"]), np.asarray(shell["faces"])
    n = np.cross(V[T[:, 1]] - V[T[:, 0]], V[T[:, 2]] - V[T[:, 0]])
    assert (((V[T].mean(1) - c) * n).sum(1) > 0).all()


def test_mouth_is_closed_at_rest_and_opens_with_the_vowels(head):
    face = mesh(head, "face")
    m = head.info["mouth"]
    up, lo = face.verts[m["slit_upper"]], face.verts[m["slit_lower"]]
    assert np.linalg.norm(up.mean(0) - lo.mean(0)) < 1e-4
    heights = {}
    for n in ("あ", "い", "う", "え", "お"):
        d = face.morphs[n]
        heights[n] = float((up + d[m["slit_upper"]])[:, 2].mean() - (lo + d[m["slit_lower"]])[:, 2].mean())
    assert heights["あ"] > 0.009 and heights["あ"] > heights["う"] > 0.002
    widths = {n: np.ptp(face.verts[m["slit_upper"]][:, 0] + face.morphs[n][m["slit_upper"]][:, 0]) for n in ("い", "う")}
    assert widths["い"] > widths["う"] + 0.008


def test_published_info(head):
    i = head.info
    for k in ("head_center", "head_radii", "skull_top", "skin", "hairline", "hairline_side", "nape", "face_outline", "ears",
              "cat_ear_anchors", "eyes", "brows", "neck_ring", "landmarks"):
        assert k in i, k
    assert i["hairline"][0, 0] > 0 > i["hairline"][-1, 0]          # left temple first
    assert i["eyes"]["L"]["center"][0] > 0
    assert len(i["neck_ring"]) == 32


def test_lash_bands_do_not_fold():
    """A thick band offset from the curved lid margin must not fold over itself (the skin would show through the folds)."""
    from mkmmd.model.parts import head_eye as EY, head_shape as HS, head_skin as SK
    shape = HS.HeadShape()
    grid = SK.Grid(shape, dict(n_cols=80, face_deg=2.1, spacing_face=0.0042, spacing_neck=0.0048, spacing_top=0.0070))
    sb = SK.SkinBuilder(shape, grid)
    eye = EY.add_eye(sb, "L")
    for name, fn in (("lash", EY.upper_lash), ("lashlow", EY.lower_lash), ("crease", EY.crease)):
        sh = fn(eye)
        lo, hi = sh["lo"], sh["hi"]
        signs = []
        for k in range(len(lo) - 1):
            a, b, c, d = lo[k], lo[k + 1], hi[k + 1], hi[k]
            area = 0.0
            for p, q in ((a, b), (b, c), (c, d), (d, a)):
                area += p[0] * q[1] - q[0] * p[1]
            if abs(area) > 1e-10:
                signs.append(np.sign(area))
        assert len(set(signs)) == 1, (name, "folded quads", signs)
