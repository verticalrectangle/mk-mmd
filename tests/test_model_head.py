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
NOSE_TIP = (0.0, -0.1170, 1.2330)                               # model space: the spec's [proportions.face] nose_tip [y, z]


@pytest.fixture(scope="module")
def head(tmp_path_factory):
    sp = spec.from_dict({"model": {"name": "t", "parts": ["head"]},
                         "colors": {"skin": {"base": "#f1e7d6"}, "eyes": {}, "mouth": {}},
                         "proportions": {"head": {}, "face": {"nose_tip": list(NOSE_TIP[1:])}, "hair_guides": {}}})
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


def test_the_closed_lids_cover_the_iris(head):
    """The iris, the pupil and the highlights sit just behind the opening (as drawn eyes do), so a closed eye must still
    cover them: in every closed-eye drawing, each of their points lies behind the closed lid sheets or the skin."""
    lids, eyes, face = mesh(head, "lids"), mesh(head, "eyes"), mesh(head, "face")
    skin = [f for f, mi in zip(face.faces, face.face_mat) if face.mats[mi] == "顔"]
    layers = np.unique([i for f, mi in zip(eyes.faces, eyes.face_mat) if eyes.mats[mi] != "白目" for i in f])
    P = eyes.verts[layers]
    skin_surf = DC.SkinSurface(face.verts, skin, np.zeros(len(skin), int))
    for n in ("まばたき", "笑い", "はぅ", "なごみ"):
        lid_surf = DC.SkinSurface(lids.verts + lids.morphs[n], lids.faces, np.zeros(len(lids.faces), int))
        y_front = np.full(len(P), np.inf)
        for s in (lid_surf, skin_surf):
            tri, bary = s.bind(P[:, 0], P[:, 2], strict=False)
            ok = tri >= 0
            y_front[ok] = np.minimum(y_front[ok], s.point(tri[ok], bary[ok])[:, 1])
        assert np.isfinite(y_front).all(), n                      # every point is under the closed lids or the skin
        assert (P[:, 1] - y_front).min() > 2e-4, (n, (P[:, 1] - y_front).min())      # and behind them (forward is -y)


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


def test_the_lash_strips_stay_in_front_of_the_skin_and_the_lids_in_every_lid_morph(head):
    """A lid morph moves the lines round the eye (the lash band and its blades, the lower lash, the crease) over the lid
    sheets and partly onto the skin round the eye (the surprised eye lifts the band clear of the lid, the moods hang it from
    a lowered lid): none of their faces may dip behind the skin, or the skin shows through the lashes in holes, nor behind
    the lid sheets, or a closed-eye drawing breaks up where it crosses them."""
    face, lines, lids = mesh(head, "face"), mesh(head, "lines"), mesh(head, "lids")
    skin = [f for f, mi in zip(face.faces, face.face_mat) if face.mats[mi] == "顔"]
    surf = DC.SkinSurface(face.verts, skin, np.zeros(len(skin), int))
    strips = [f for f, mi in zip(lines.faces, lines.face_mat) if lines.mats[mi] in ("睫毛", "下睫毛", "二重")]
    t = [(a, b) for a in np.linspace(0, 1, 5) for b in np.linspace(0, 1, 5) if a + b <= 1]
    for n in [None] + LIDS + ["困る", "悲しみ", "怒り", "にこり", "真面目"]:
        V = lines.verts if n is None else lines.verts + lines.morphs.get(n, 0.0)
        Q = np.array([V[f[0]] + a * (V[f[1]] - V[f[0]]) + b * (V[f[2]] - V[f[0]]) for f in strips for a, b in t])
        tri, bary = surf.bind(Q[:, 0], Q[:, 2], strict=False)
        ok = tri >= 0                                          # inside the opening the strips lie over the lids, not skin
        behind = Q[ok, 1] - surf.point(tri[ok], bary[ok])[:, 1]   # > 0: behind the skin (forward is -y)
        assert behind.max() < 2.5e-4, (n, behind.max())
        if n is None:
            continue                                           # at rest the sheets are tucked away behind the margins
        sheets = DC.SkinSurface(lids.verts + lids.morphs.get(n, 0.0), lids.faces, np.zeros(len(lids.faces), int))
        tri, bary = sheets.bind(Q[:, 0], Q[:, 2], strict=False)
        ok = tri >= 0
        assert (Q[ok, 1] - sheets.point(tri[ok], bary[ok])[:, 1]).max() < 0.0, n


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
    """Rotate the eye layers about the eye bone: outside the lid opening nothing may come in front of the skin. The iris is
    a flat disc just behind the opening, pushed back as far as the skin round the opening needs for the eye's gaze range
    (head_eye DEFAULTS gaze), so it stays behind the skin at 15 deg outward, 25 deg inward and 15 deg up or down."""
    face, eyes = mesh(head, "face"), mesh(head, "eyes")
    surf = DC.SkinSurface(face.verts, face.faces, np.zeros(len(face.faces), int))
    worst = -1.0
    for side, sg, bone in (("L", 1.0, "左目"), ("R", -1.0, "右目")):
        e = head.info["eyes"][side]
        E = np.asarray(e["center"])
        poly = np.asarray(e["opening"])
        c = poly.mean(0)
        grown = c + (poly - c) * 1.06
        rows = np.nonzero(eyes.weights[bone] > 0.5)[0]
        P0 = eyes.verts[rows]
        for yaw, pitch in ((15 * sg, 0), (-25 * sg, 0), (0, 15), (0, -15), (15 * sg, 15), (-25 * sg, -15), (15 * sg, -15), (-25 * sg, 15)):
            ay, ap = np.radians(yaw), np.radians(pitch)
            Rz = np.array([[np.cos(ay), -np.sin(ay), 0], [np.sin(ay), np.cos(ay), 0], [0, 0, 1]])
            Rx = np.array([[1, 0, 0], [0, np.cos(ap), -np.sin(ap)], [0, np.sin(ap), np.cos(ap)]])
            Q = (P0 - E) @ (Rz @ Rx).T + E
            Q = Q[~_inside(grown, Q[:, [0, 2]])]
            tri, bary = surf.bind(Q[:, 0], Q[:, 2], strict=False)
            ok = tri >= 0                                      # beyond the front skin: hidden by the head's side
            if ok.any():
                worst = max(worst, float(-(Q[ok, 1] - surf.point(tri[ok], bary[ok])[:, 1]).min()))
    assert worst < 0.0, worst                           # nothing comes in front of the skin


def test_the_white_stays_inside_the_head(head):
    """The white is a pocket that opens out behind the lids: outside the opening none of it may come in front of the skin
    (it would show through the cheek or the temple)."""
    face, eyes = mesh(head, "face"), mesh(head, "eyes")
    skin = [f for f, mi in zip(face.faces, face.face_mat) if face.mats[mi] == "顔"]
    surf = DC.SkinSurface(face.verts, skin, np.zeros(len(skin), int))
    white = np.unique([i for f, mi in zip(eyes.faces, eyes.face_mat) if eyes.mats[mi] == "白目" for i in f])
    P = eyes.verts[white]
    out = np.ones(len(P), bool)
    for side in ("L", "R"):
        poly = np.asarray(head.info["eyes"][side]["opening"])
        c = poly.mean(0)
        out &= ~_inside(c + (poly - c) * 1.02, P[:, [0, 2]])
    Q = P[out]
    tri, bary = surf.bind(Q[:, 0], Q[:, 2], strict=False)
    ok = tri >= 0
    assert ok.sum() > 50                                     # the pocket opens out behind the skin around the opening
    gap = Q[ok, 1] - surf.point(tri[ok], bary[ok])[:, 1]
    assert gap.min() > 0.0, gap.min()


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


def test_the_nose_point_is_where_the_spec_puts_it(head):
    """[proportions.face] nose_tip is the front-most point of the face at its height (it was ignored: a flat face), and the
    nose stands out: nothing on the midline below it (the upper lip, the mouth, the chin) comes out as far."""
    V = mesh(head, "face").verts
    near = V[(np.abs(V[:, 0]) < 0.002) & (np.abs(V[:, 2] - NOSE_TIP[2]) < 0.004)]
    tip = near[np.argmin(near[:, 1])]
    assert np.abs(tip - NOSE_TIP)[1:].max() < 0.0012, tip
    below = V[(np.abs(V[:, 0]) < 0.002) & (V[:, 2] < NOSE_TIP[2] - 0.014) & (V[:, 2] > NOSE_TIP[2] - 0.060)]
    assert below[:, 1].min() > NOSE_TIP[1] + 0.003, below[:, 1].min()


def test_the_jaw_shadow_stays_under_the_jaw(head):
    """The texture's shadow strip (the skin under the jaw and the neck) never reaches the chin or the cheeks seen from the
    front: the chin and the throat behind it share azimuth and height, which one cylinder map cannot tell apart."""
    from mkmmd.model.parts import head_tex as TX
    face = mesh(head, "face")
    skin = [i for i, mi in enumerate(face.face_mat) if face.mats[mi] == "顔"]
    start = np.concatenate([[0], np.cumsum([len(f) for f in face.faces])])
    V = face.verts
    for i in skin:
        f = face.faces[i]
        P = V[list(f)]
        n = np.cross(P[1] - P[0], P[2] - P[0])
        n = n / np.linalg.norm(n)
        v = face.uv[start[i]:start[i + 1], 1]
        if n[1] < -0.6 and n[2] > -0.3 and P[:, 1].max() < -0.07:   # the face's front: in the face part of the map
            assert v.min() > TX.UNDER_V, (P.mean(0), v)
        if P[:, 2].max() < 1.165:                                # the neck at the seam ring: in the shadow strip
            assert v.max() < TX.UNDER_V, (P.mean(0), v)


def test_the_mouth_interior_stays_behind_the_skin_when_it_opens(head):
    """The bag behind the lips follows them further than the skin under the lower lip follows: it must stay behind that
    skin (it showed through the chin as dark patches)."""
    face, mouth = mesh(head, "face"), mesh(head, "mouth")
    skin = [f for f, mi in zip(face.faces, face.face_mat) if face.mats[mi] == "顔"]
    bag = np.unique([i for f, mi in zip(mouth.faces, mouth.face_mat) if mouth.mats[mi] == "口内" for i in f])
    for name in ("あ", "い", "う", "え", "お", "ω"):
        surf = DC.SkinSurface(face.verts + face.morphs[name], skin, np.zeros(len(skin), int))
        Q = mouth.verts[bag] + mouth.morphs[name][bag]
        tri, bary = surf.bind(Q[:, 0], Q[:, 2], strict=False)
        ok = tri >= 0
        gap = Q[ok, 1] - surf.point(tri[ok], bary[ok])[:, 1]
        gap = gap[gap > -0.01]                                   # the skin right in front, not the throat seen through the lips
        assert gap.min() > 0.0005, (name, gap.min())


def test_the_blush_stays_under_the_skin_in_every_mouth_shape(head):
    """The 照れ patches wait 1.2 mm under the cheeks: every mouth shape must carry them along with the skin (ω pulled the
    cheeks in and they showed through as dark spots)."""
    face, blush = mesh(head, "face"), mesh(head, "blush")
    skin = [f for f, mi in zip(face.faces, face.face_mat) if face.mats[mi] == "顔"]
    for name in ("あ", "い", "う", "え", "お", "ω", "口角上げ", "口角下げ", "にやり"):
        surf = DC.SkinSurface(face.verts + face.morphs[name], skin, np.zeros(len(skin), int))
        Q = blush.verts + blush.morphs.get(name, np.zeros_like(blush.verts))
        tri, bary = surf.bind(Q[:, 0], Q[:, 2], strict=False)
        ok = tri >= 0
        assert ok.sum() > 100
        gap = Q[ok, 1] - surf.point(tri[ok], bary[ok])[:, 1]
        assert gap.min() > 0.0005, (name, gap.min())


def test_the_corner_shapes_leave_the_middle_of_the_lips_still():
    """口角上げ / 口角下げ lift and lower the corners, 1.2 cm from the midline: the middle of the lips stays (np.sign(0) put
    a "corner" on the midline column, and the middle of the upper lip rose 2.8 mm on its own: a crease)."""
    from mkmmd.model.parts import head_mouth as MO
    c = dict(MO.DEFAULTS)
    P = np.array([[x, -0.1, c["z"] + dz] for dz in np.linspace(-0.006, 0.006, 7) for x in np.linspace(-0.004, 0.004, 9)])
    for key in ("mouth_smile", "mouth_down"):
        d = MO.lip_field(P, np.zeros(len(P)), c, **MO.SHAPES[key])
        assert np.abs(d[P[:, 0] == 0.0]).max() < 2e-4, key


def test_the_skin_arrives_at_the_inner_corner_of_each_eye_without_diving(head):
    """Between the nose and the inner corner of each eye the skin sets back early and arrives nearly facing forward, as
    drawn eyes (and the imported face) do: it dived 3.4 mm over the last 6 mm into the corner, a dark crease. It never
    goes below the corner either (a pit)."""
    face = mesh(head, "face")
    skin = [f for f, mi in zip(face.faces, face.face_mat) if face.mats[mi] == "顔"]
    surf = DC.SkinSurface(face.verts, skin, np.zeros(len(skin), int))
    s = np.array([6, 5, 4, 3, 2.5, 2, 1.5, 1, 0.5, 0.25, 0.05]) / 1000          # before the corner, towards the nose
    for side, sg in (("L", 1.0), ("R", -1.0)):
        poly = np.asarray(head.info["eyes"][side]["opening"])
        c = poly[np.argmin(sg * poly[:, 0])]
        tri, bary = surf.bind(c[0] - sg * s, np.full(len(s), c[1]))
        rel = surf.point(tri, bary)[:, 1] - surf.point(tri[-1:], bary[-1:])[0, 1]   # + = deeper than at the rim
        assert -rel[0] < 0.0025, (side, -rel[0])
        assert rel[1:-1].max() < 0.0001, (side, rel[1:-1].max())


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
    for name, sh in dict(EY.upper_lines(eye), lashlow=EY.lower_lash(eye)).items():
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
