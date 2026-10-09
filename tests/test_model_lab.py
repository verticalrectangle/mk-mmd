"""The model lab (mkmmd.model.lab): the renderer's depth order, scale and small-triangle path, posing through the bone
tree (children, rotation grants, finger curls on both hands), one scale across models, girths on the skin under a
sleeve, and a red line on a screenshot of a sheet read back in millimetres."""
import json

import numpy as np
import pytest

from mkmmd.model import lab as LAB
from mkmmd.model import pmx_io as X

pytest.importorskip("cv2")

BOX_F = [(0, 1, 3), (0, 3, 2), (4, 6, 7), (4, 7, 5), (0, 4, 5), (0, 5, 1), (2, 3, 7), (2, 7, 6), (0, 2, 6), (0, 6, 4),
         (1, 5, 7), (1, 7, 3)]


def box(c, h):
    """Corners (8, 3) of an axis-aligned box (model space) and its triangles."""
    P = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], float) * np.asarray(h, float)
    return P + np.asarray(c, float), BOX_F


def cylinder(r, z0, z1, n=64, seam=False):
    """Side of a z cylinder; `seam` splits the first column's vertices the way a UV seam splits a PMX mesh."""
    th = np.linspace(0, 2 * np.pi, n, endpoint=False)
    cols = [np.stack([r * np.cos(t) * np.ones(2), r * np.sin(t) * np.ones(2), [z0, z1]], -1) for t in th]
    if seam:
        cols.append(cols[0].copy())
    P = np.concatenate(cols)
    m = len(cols)
    F = []
    for i in range(m if not seam else m - 1):
        j = (i + 1) % m
        a, b, c, d = 2 * i, 2 * i + 1, 2 * j, 2 * j + 1
        F += [(a, c, d), (a, d, b)]
    return P, F


def pmx_of(parts, bones=(("root", (0, 0, 0), -1, None),), weights=None):
    """A PmxModel (unit 1) of `parts` [(points (k, 3) model space, triangles, rgb or rgba 0..1)]; bones [(name, head,
    (grant parent, ratio) | None)]; `weights` one bone index per vertex (default the first bone)."""
    verts, faces, mats = [], [], []
    for i, (P, F, rgb) in enumerate(parts):
        base = len(verts)
        for p in P:
            b = 0 if weights is None else int(weights[len(verts)])
            verts.append(X.PmxVertex(pos=(p[0], p[2], p[1]), normal=(0.0, 1.0, 0.0), uv=(0.0, 0.0), bones=(b,),
                                     weights=(1.0,)))
        faces += [base + k for f in F for k in f]
        mats.append(X.PmxMaterial(name=f"m{i}", diffuse=tuple(rgb) if len(rgb) == 4 else (*rgb, 1.0),
                                  index_count=3 * len(F)))
    pb = [X.PmxBone(name=n, pos=(h[0], h[2], h[1]), parent=p, grant_rotate=g is not None,
                    grant_parent=g[0] if g else -1, grant_ratio=g[1] if g else 0.0) for n, h, p, g in bones]
    return X.PmxModel(vertices=verts, faces=faces, materials=mats, bones=pb)


def model_of(*a, **kw):
    return LAB.from_pmx(pmx_of(*a, **kw), root=".", unit=1.0)


FRONT = dict(centre=(0.0, 0.0, 0.0), right=(1.0, 0.0, 0.0), up=(0.0, 0.0, 1.0), towards=(0.0, -1.0, 0.0), half=1.0)


def test_render_keeps_the_nearer_surface_at_scale():
    near = box((0, -0.5, 0), (0.2, 0.05, 0.2))                     # the camera is on the -y side
    far = box((0, 0.5, 0), (0.5, 0.05, 0.5))
    m = model_of([(*near, (1, 0, 0)), (*far, (0, 0, 1))])
    T, lit, shade = LAB.scene_colours(m)
    im, mask = LAB.render(m.V, m.N, T, lit, shade, FRONT, size=200, outline=False)
    px = np.asarray(im).astype(int)
    assert px[100, 100, 0] > 2 * px[100, 100, 2]                    # the centre shows the near red box
    assert px[100, 140, 2] > 2 * px[100, 140, 0]                    # 0.4 m right of it: the far blue one
    cols = np.nonzero(mask[100])[0]
    assert abs((cols.max() - cols.min() + 1) - 100) <= 2            # 1 m wide in a 2 m frame of 200 px


def test_small_triangles_cover_what_large_ones_do():
    P, F = box((0, 0, 0), (0.5, 0.001, 0.5))
    big = model_of([(P, F, (1, 1, 1))])
    n = 40                                                         # the same square as 2 x 40 x 40 small triangles
    g = np.linspace(-0.5, 0.5, n + 1)
    Q = np.array([[x, -0.001, z] for z in g for x in g])
    G = [(i * (n + 1) + j, i * (n + 1) + j + 1, (i + 1) * (n + 1) + j + 1) for i in range(n) for j in range(n)]
    G += [(i * (n + 1) + j, (i + 1) * (n + 1) + j + 1, (i + 1) * (n + 1) + j) for i in range(n) for j in range(n)]
    small = model_of([(Q, G, (1, 1, 1))])
    masks = [LAB.render(m.V, m.N, *LAB.scene_colours(m), FRONT, size=200, ss=1)[1] for m in (big, small)]
    assert (masks[0] & masks[1]).sum() / (masks[0] | masks[1]).sum() > 0.995   # no cracks where triangles meet


def test_bones_turn_their_children_and_grants_turn_part_way():
    bones = [("root", (0, 0, 0), -1, None), ("b1", (0, 0, 1), 0, None), ("b2", (1, 0, 1), 1, None),
             ("g", (0, 0, 1), 0, (1, 0.5))]
    m = model_of([(np.array([[2.0, 0.0, 1.0], [1.0, 0.0, 1.0]]), [], (1, 1, 1))], bones=bones, weights=[2, 3])
    V, _ = m.skin(m.V, m.N, m.matrices({1: LAB.rot((0, 0, 1), 90)}))
    assert np.allclose(V[0], (0.0, 2.0, 1.0), atol=1e-9)           # on b2: turns with b1 about b1's head
    assert np.allclose(V[1], (np.sqrt(0.5), np.sqrt(0.5), 1.0), atol=1e-9)   # granted half of b1's turn


@pytest.fixture(scope="module")
def girl():
    return LAB.load("base:girl", parts=["body"])


def test_fingers_curl_into_the_palm_and_spread_on_both_hands(girl):
    rest, _ = girl.deform("rest")
    curled, _ = girl.deform("curled")
    fist, _ = girl.deform("fist")
    spread, _ = girl.deform("spread")
    for side in "LR":
        a, r, n = LAB.hand_axes(girl, side)
        for f in ("index", "middle", "ring", "little"):
            sel = girl.dominant == girl.sem[f"{f}3.{side}"]
            assert ((curled[sel] - rest[sel]) @ n).mean() > 0.01, (side, f)    # the tips go to the palm side
            assert ((fist[sel] - rest[sel]) @ a).mean() < -0.03, (side, f)     # and fold back towards the wrist
        th = girl.dominant == girl.sem[f"thumb2.{side}"]
        d = (fist[th] - rest[th]).mean(axis=0)
        assert d @ r > max(0.02, 2 * (d @ n)), side                 # the thumb folds across the fingers, not out
        i3, l3 = (girl.dominant == girl.sem[f"{f}3.{side}"] for f in ("index", "little"))
        gap = [float(np.linalg.norm(V[i3].mean(0) - V[l3].mean(0))) for V in (rest, spread)]
        assert gap[1] > gap[0] + 0.01, side


def test_t_pose_lays_both_arms_out_level_and_leaves_the_legs(girl):
    G = girl.matrices(LAB.pose_rotations(girl, "tpose"))

    def head(name):
        b = girl.sem[name]
        return (G[b] @ np.r_[girl.heads[b], 1.0])[:3]
    for side, out in (("L", 1.0), ("R", -1.0)):
        a, e, w = (head(f"{k}.{side}") for k in ("arm", "elbow", "wrist"))
        assert abs(LAB._below(e - a)) < 0.05 and abs(LAB._below(w - e)) < 0.05, side   # upper arm and forearm level
        assert out * (w - a)[0] > 0.9 * np.linalg.norm(w - a), side                   # straight out to her side
    rest, _ = girl.deform()
    tpose, _ = girl.deform("tpose")
    legs = np.isin(girl.dominant, sorted(girl.subtree(girl.sem["leg.L"]) | girl.subtree(girl.sem["leg.R"])))
    assert np.abs(tpose[legs] - rest[legs]).max() < 1e-9


def test_side_by_side_cells_share_one_scale_and_floor():
    small = model_of([(*box((0, 0, 0.5), (0.15, 0.1, 0.5)), (0.9, 0.8, 0.7))])
    tall = model_of([(*box((0, 0, 0.75), (0.15, 0.1, 0.75)), (0.9, 0.8, 0.7))])
    img, layout, cover = LAB.sheet([small, tall], region="body", views=["front"], size=200)
    cov = np.asarray(cover) > 127
    spans = []
    for cell in layout["cells"]:
        x, y, w, h = cell["box"]
        rows = np.nonzero(cov[y:y + h, x:x + w].any(axis=1))[0]
        spans.append((rows.min(), rows.max()))
    (t0, b0), (t1, b1) = spans
    assert abs(b0 - b1) <= 1                                       # both stand on one floor line
    assert abs((b1 - t1 + 1) / (b0 - t0 + 1) - 1.5) < 0.03         # 1.5 m against 1 m


def test_limb_girth_is_cut_on_the_skin_inside_a_sleeve():
    # the skin a sleeve always covers is hidden (alpha 0), as models do, and sits 1 cm off the bone line; the sleeve's
    # centre is on it, so the nearest loop would be the sleeve
    limb = cylinder(0.05, 0.1, 1.0, seam=True)
    limb = (limb[0] + (0.01, 0.0, 0.0), limb[1])
    sleeve = cylinder(0.08, 0.3, 0.9)
    bones = [("root", (0, 0, 0), -1, None), ("左足", (0, 0, 1.0), 0, None), ("左ひざ", (0, 0, 0.55), 1, None),
             ("左足首", (0, 0, 0.1), 2, None)]
    m = model_of([(*limb, (1, 0.9, 0.8, 0.0)), (*sleeve, (0.2, 0.2, 0.3))], bones=bones,
                 weights=[1] * (len(limb[0]) + len(sleeve[0])))
    nums = LAB.measure(m, "leg", "L")
    assert abs(nums["girth at 35 %"] - 2000 * np.pi * 0.05) < 0.01 * 2000 * np.pi * 0.05


def test_an_open_cut_through_a_sheet_is_not_a_loop_round_the_limb():
    """A skirt piece beside the leg cuts into an open chain: however often a ray from the leg's point crosses it, it does
    not enclose the point, so the girth stays the skin's even where the piece is shorter than the skin's loop."""
    limb = cylinder(0.05, 0.1, 1.0)
    th = np.radians(np.linspace(60, 120, 9))                        # an arc 0.25 m in front of the leg, crossing -Y
    piece = (np.array([[0.15 * np.cos(t), -0.4 + 0.15 * np.sin(t), z] for t in th for z in (0.3, 0.9)]),
             [f for i in range(len(th) - 1) for f in ((2 * i, 2 * i + 2, 2 * i + 3), (2 * i, 2 * i + 3, 2 * i + 1))])
    bones = [("root", (0, 0, 0), -1, None), ("左足", (0, 0, 1.0), 0, None), ("左ひざ", (0, 0, 0.55), 1, None),
             ("左足首", (0, 0, 0.1), 2, None)]
    m = model_of([(*limb, (1, 0.9, 0.8)), (*piece, (0.2, 0.2, 0.3))], bones=bones,
                 weights=[1] * (len(limb[0]) + len(piece[0])))
    assert LAB.measure(m, "leg", "L")["girth at 35 %"] == pytest.approx(2000 * np.pi * 0.05, rel=0.01)


def test_a_red_line_on_a_screenshot_reads_back_in_millimetres(girl, tmp_path):
    cv2 = LAB._cv()
    img, layout, cover = LAB.sheet([girl], region="hand", poses=["rest", "relaxed"], views=["back", "palm"], size=300)
    sheet = tmp_path / "hand.png"
    img.save(sheet)
    cover.save(tmp_path / "hand.mask.png")
    (tmp_path / "hand.json").write_text(json.dumps(layout))
    cell = layout["cells"][0]
    x, y, w, h = cell["box"]
    sil = np.asarray(cover)[y:y + h, x:x + w] > 127
    crop = cv2.cvtColor(np.asarray(img)[y:y + h, x:x + w], cv2.COLOR_RGB2BGR)
    for mm in (4.0, -2.0):                                         # inside the outline, then outside it
        d = mm / 1000 * layout["px_per_m"]
        rows = range(int(h * 0.5), int(h * 0.8))
        line = np.array([(np.nonzero(sil[r])[0][0] + d, r) for r in rows if sil[r].any()])
        scale, border = 1.6, 40                                    # a zoomed screenshot inside a window
        shot = np.full((int(h * scale) + 2 * border, int(w * scale) + 2 * border, 3), (52, 50, 48), np.uint8)
        shot[border:border + int(h * scale), border:border + int(w * scale)] = cv2.resize(crop, (int(w * scale),
                                                                                                  int(h * scale)))
        cv2.polylines(shot, [(line * scale + border).astype(np.int32).reshape(-1, 1, 2)], False, (0, 0, 255), 3)
        cv2.imwrite(str(tmp_path / "marked.png"), shot)
        rep = LAB.trace(tmp_path / "marked.png", sheet, layout)
        assert rep["cell"] == cell["label"]
        assert abs(float(np.median([s["inset_mm"] for s in rep["samples"]])) - mm) < 0.8, (mm, rep["inset_mm"])
