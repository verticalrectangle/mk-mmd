"""The body part in mesh mode (mkmmd.model.parts.body_donor): the girl base's CC0 donor body fitted to a character's
skeleton. It must be a standard body (rig, landmarks, colliders), closed but for the seam the head builds on, standing in
the shoes the outfit makes for her foot, its limbs on her joints and weighted like the generated body. bpy-free."""
from collections import Counter

import numpy as np
import pytest

from mkmmd.core import bonemap
from mkmmd.model import build as BD
from mkmmd.model import part as P
from mkmmd.model import skeleton as SK
from mkmmd.model import spec as SP

MESH = ['body.source="mesh"', 'body.mesh="base:girl/body.npz"']


def make(tmp_path, *overrides):
    spec = SP.load("base:girl", MESH + list(overrides))
    return BD.run(spec, only="body", tex_dir=tmp_path)[0]


@pytest.fixture(scope="module")
def donor(tmp_path_factory):
    return make(tmp_path_factory.mktemp("donor"))


def skin_edges(mesh):
    """Edge use counts of the skin (material 0; the nail plates are open patches on top of it)."""
    c = Counter()
    for f, m in zip(mesh.faces, mesh.face_mat):
        if m == 0:
            for i in range(len(f)):
                c[(min(f[i], f[(i + 1) % len(f)]), max(f[i], f[(i + 1) % len(f)]))] += 1
    return c


def assert_closed_but_for_the_seam(part):
    """The skin is closed except one loop, and that loop lies on the seam ellipse the head part builds its neck on."""
    mesh, nt = part.meshes[0], part.info["neck_top"]
    c = skin_edges(mesh)
    assert max(c.values()) == 2
    nxt = {}
    for a, b in (e for e, k in c.items() if k == 1):
        nxt.setdefault(a, []).append(b)
        nxt.setdefault(b, []).append(a)
    assert all(len(v) == 2 for v in nxt.values())
    loop, prev = [next(iter(nxt))], None
    while True:
        step = [v for v in nxt[loop[-1]] if v != prev]
        prev = loop[-1]
        if step[0] == loop[0]:
            break
        loop.append(step[0])
    assert len(loop) == len(nxt)                                          # one loop: no hole at a wrist or anywhere
    V = mesh.verts[loop]
    cx, cy = nt["center"]
    assert np.allclose(V[:, 2], nt["z"], atol=1e-6)
    assert np.allclose(((V[:, 0] - cx) / nt["rx"]) ** 2 + ((V[:, 1] - cy) / nt["ry"]) ** 2, 1.0, atol=1e-6)


def test_mesh_body_is_a_standard_body(donor):
    P.check(donor)
    names = {b.name for b in donor.bones}
    smap = bonemap.build_map({n: n for n in names})
    assert not [s for s in bonemap.REQUIRED if s not in smap]
    for must in ("腕捩", "手捩", "つま先ＩＫ"):
        assert any(must in n for n in names)
    assert [m.name for m in donor.materials] == ["skin", "nail"]
    assert donor.info["mesh"] == "body" and donor.info["source"] == "mesh"
    land = donor.info["landmarks"]
    assert not [n for n in SK.required_landmarks() if n not in land]
    for k, v in land.items():
        if k.endswith(".L"):
            assert np.allclose(land[k[:-2] + ".R"], [-v[0], v[1], v[2]], atol=1e-4), k
    for extra in ("tail_root", "tail_root.L", "tail_root.R", "bust_front", "toe_end.L"):
        assert extra in land
    assert np.asarray(donor.info["tail_normal"])[1] > 0.5                  # the tail root faces backwards (+Y)
    bodies = {b.name: b for b in donor.bodies}
    for r in ("shoulder", "arm", "forearm", "hand", "thigh", "calf", "foot"):
        a, b = bodies[f"col_{r}_L"], bodies[f"col_{r}_R"]
        assert a.size == b.size and np.allclose(b.location, np.asarray(a.location) * [-1, 1, 1], atol=1e-6), r
    assert all(b.bone in names for b in donor.bodies)


def test_mesh_body_is_closed_but_for_the_seam_the_head_builds_on(donor):
    """The forearms are cut and joined to the hands, the head cut off at the neck: no hole anywhere but the neck seam,
    whose every point lies on the ellipse the head part's neck starts from."""
    assert_closed_but_for_the_seam(donor)


def test_mesh_body_stands_in_her_shoes(donor):
    """The donor's flat foot stands on the shoes' inner floor (`foot_inner_floor_z`), as long and as wide as her foot:
    the outfit's shoes are made around it (a foot on tiptoe or sunk into the sole breaks them)."""
    V, F = donor.meshes[0].verts, SP.load("base:girl")["proportions"]["sections"]["foot"]
    land = donor.info["landmarks"]
    foot = V[(V[:, 0] > 0.0) & (V[:, 2] < land["ankle.L"][2] - 0.02) & (V[:, 0] < 0.15)]
    assert abs(foot[:, 2].min() - F["foot_inner_floor_z"]) < 5e-4
    assert abs(np.ptp(foot[:, 1]) - F["foot_length"]) < 0.006
    assert abs(np.ptp(foot[:, 0]) - F["foot_width"]) < 0.008


def test_mesh_body_limbs_sit_on_her_joints(donor):
    """Across each elbow, wrist, knee and ankle the skin's section (on the plane that halves the bend) is centred on her
    joint within 12 mm: the fit carries the donor's limbs onto her skeleton, so the rig bends them where they bend."""
    V, land = donor.meshes[0].verts, donor.info["landmarks"]
    for prev, joint, nxt in (("arm", "elbow", "wrist"), ("elbow", "wrist", "middle1"), ("leg", "knee", "ankle"),
                             ("knee", "ankle", "toe")):
        for side, sg in (("L", 1.0), ("R", -1.0)):
            J = np.asarray(land[f"{joint}.{side}"])
            a = J - np.asarray(land[f"{prev}.{side}"])
            b = np.asarray(land[f"{nxt}.{side}"]) - J
            d = a / np.linalg.norm(a) + b / np.linalg.norm(b)
            d /= np.linalg.norm(d)
            ring = V[(np.abs((V - J) @ d) < 0.004) & (np.linalg.norm(V - J, axis=1) < 0.07) & (V[:, 0] * sg > 0.02)]
            c = 0.5 * (ring.min(0) + ring.max(0))
            off = (c - J) - ((c - J) @ d) * d
            assert len(ring) > 8 and np.linalg.norm(off) < 0.012, (joint, side, np.linalg.norm(off))


def test_mesh_body_is_weighted_like_the_generated_body(donor):
    """Valid PMX weights; the upper arm from 7 cm past the shoulder joint (where the generated arm's shoulder blend ends)
    moves with the arm alone, not the shoulder or the chest, or raising the arm bends it in the middle; thigh, shin,
    foot and forearm on their own bones; the seam ring all neck."""
    mesh, land = donor.meshes[0], donor.info["landmarks"]
    names = list(mesh.weights)
    M = np.stack([mesh.weights[n] for n in names], 1)
    assert (M >= 0).all() and np.allclose(M.sum(1), 1.0, atol=1e-6) and ((M > 0).sum(1) <= 4).all()
    bones = {b.name: b for b in donor.bones}
    assert all(n in bones and bones[n].deform for n in names)
    V = mesh.verts
    A, E = np.asarray(land["arm.L"]), np.asarray(land["elbow.L"])
    d = (E - A) / np.linalg.norm(E - A)
    s = (V - A) @ d
    upper = (s > 0.07) & (s < np.linalg.norm(E - A) - 0.04) & (np.linalg.norm((V - A) - np.outer(s, d), axis=1) < 0.05)
    arm = {"左腕", "左腕捩", "左腕捩1", "左腕捩2", "左腕捩3"}
    assert upper.sum() > 50 and M[upper][:, [i for i, n in enumerate(names) if n in arm]].sum(1).min() > 0.999

    def dominant(mask):
        return Counter(np.array(names)[np.argmax(M[mask], axis=1)]).most_common(1)[0][0]
    x, z = V[:, 0], V[:, 2]
    left = (x > 0.02) & (x < 0.16)
    assert dominant(left & (z > land["knee.L"][2] + 0.08) & (z < land["leg.L"][2] - 0.10)) == "左足"
    assert dominant(left & (z > land["ankle.L"][2] + 0.06) & (z < land["knee.L"][2] - 0.06)) == "左ひざ"
    assert dominant(left & (z < land["ankle.L"][2] - 0.04)) == "左足首"
    W, El = np.asarray(land["wrist.L"]), np.asarray(land["elbow.L"])
    t = (V - El) @ (W - El) / np.linalg.norm(W - El) ** 2
    fore = (t > 0.2) & (t < 0.8) & (np.linalg.norm(V - (El + np.outer(t, W - El)), axis=1) < 0.04)
    assert dominant(fore) in ("左ひじ", "左手捩")
    nt = donor.info["neck_top"]
    seam = np.abs(V[:, 2] - nt["z"]) < 1e-6
    assert seam.sum() > 20 and (mesh.weights["首"][seam] > 0.999).all()


def test_mesh_body_is_symmetric(donor):
    """Every vertex has a mirror twin, and its weights are the twin's with left and right swapped."""
    mesh = donor.meshes[0]
    V = mesh.verts
    key = lambda P: [tuple(k) for k in np.round(P / 1e-5).astype(np.int64)]
    at = {k: i for i, k in enumerate(key(V))}
    twin = np.array([at.get(k, -1) for k in key(V * [-1.0, 1.0, 1.0])])
    assert (twin >= 0).all()
    for n, w in mesh.weights.items():
        m = n.replace("左", "\0").replace("右", "左").replace("\0", "右")
        assert np.allclose(w[twin], mesh.weights.get(m, np.zeros(len(V))), atol=1e-5), n


def _front_radii(V):
    """The skin's most forward point near the left breast and its two principal radii of curvature there (metres, the
    smaller first), from a quadric y(x, z) fitted to the front-facing skin within 3 cm of it."""
    sel = (V[:, 0] > 0.005) & (V[:, 0] < 0.13) & (np.abs(V[:, 2] - 1.04) < 0.07) & (V[:, 1] < 0.0)
    f = V[sel][np.argmin(V[sel][:, 1])]
    near = (np.abs(V[:, 0] - f[0]) < 0.03) & (np.abs(V[:, 2] - f[2]) < 0.03) & (V[:, 1] < f[1] + 0.03) & (V[:, 0] > 0.003)
    X, Z = V[near, 0] - f[0], V[near, 2] - f[2]
    c = np.linalg.lstsq(np.stack([np.ones_like(X), X, Z, X * X, Z * Z, X * Z], 1), V[near, 1], rcond=None)[0]
    return f, np.sort(1.0 / np.linalg.eigvalsh(np.array([[2 * c[3], c[5]], [c[5], 2 * c[4]]])))


def test_bust_grows_round_breasts_forward_and_nothing_else(tmp_path, donor):
    """[body] bust moves the front of each breast forward by that much, the two alike (mirror images), keeping the
    donor's own round shape: at the new front the skin is curved at least half as gently as the donor's is at hers (not
    drawn out into a point); the two meet in the cleavage without adding up; nothing 14 cm or more from the fronts
    moves, nor anything of her back."""
    A = np.asarray(donor.meshes[0].verts)
    B = np.asarray(make(tmp_path, "body.bust=0.05").meshes[0].verts)
    move = B - A
    dist = np.linalg.norm(move, axis=1) / 0.05
    assert 0.97 < dist.max() < 1.005
    fronts = [A[np.argmax(np.where(sg * A[:, 0] > 0, dist, -1.0))] for sg in (1.0, -1.0)]
    assert fronts[0][0] == pytest.approx(-fronts[1][0], abs=1e-6) and move[np.argmax(dist), 1] < -0.045
    (_, before), (_, after) = _front_radii(A), _front_radii(B)
    assert after[0] > 0.5 * before[0], (before, after)
    cleavage = np.argmin(np.linalg.norm(A - 0.5 * (fronts[0] + fronts[1]), axis=1))
    assert dist[cleavage] < 0.9                                         # the two meet there, not add up
    far = np.min([np.linalg.norm(A - f, axis=1) for f in fronts], axis=0) > 0.14
    assert (dist[far] == 0.0).all() and (dist[A[:, 1] > fronts[0][1] + 0.1] == 0.0).all()
    key = lambda P: [tuple(k) for k in np.round(P / 1e-5).astype(np.int64)]
    at = {k: i for i, k in enumerate(key(A))}
    twin = np.array([at.get(k, -1) for k in key(A * [-1.0, 1.0, 1.0])])
    assert (twin >= 0).all() and np.allclose(move[twin] * [-1.0, 1.0, 1.0], move, atol=1e-6)


DIGIT_END = {"index": ("index3", "左人指３"), "middle": ("middle3", "左中指３"), "ring": ("ring3", "左薬指３"),
             "little": ("little3", "左小指３"), "thumb": ("thumb2", "左親指２")}


def _fingertips(part):
    """Per digit of the left hand: (how far its skin runs past the tip joint, its width 3 mm short of its end) in metres,
    measured on the skin its last bone drives, along that bone."""
    m, L = part.meshes[0], part.info["landmarks"]
    V, out = np.asarray(m.verts), {}
    for f, (joint, bone) in DIGIT_END.items():
        j, tip = np.asarray(L[f"{joint}.L"], float), np.asarray(L[f"{f}_tip.L"], float)
        ln = float(np.linalg.norm(tip - j))
        q = V[np.asarray(m.weights[bone]) > 0.5] - j
        along = q @ ((tip - j) / ln)
        across = np.linalg.norm(q - along[:, None] * ((tip - j) / ln), axis=1)
        near = np.abs(along - (along.max() - 0.003)) < 0.0008
        out[f] = (float(along.max()) - ln, 2.0 * float(across[near].max()))
    return out


def test_claws_draw_the_fingertips_out_into_sharp_points(tmp_path):
    """[body] claws.point draws every fingertip out into a point: each digit's skin runs on about that far past its tip
    joint and, 3 mm short of its end, is under 60 % as wide as the round fingertip is there; with nails = false there
    are no nail plates (the dark points are the claws), and nothing off the digits' last bones moves."""
    (tmp_path / "round").mkdir()
    (tmp_path / "sharp").mkdir()
    plain = make(tmp_path / "round", "body.nails=false")
    part = make(tmp_path / "sharp", "body.nails=false", "body.claws.point=0.008")
    assert "nails" not in {m.name_en for m in part.materials} | {m.name_en for m in plain.materials}
    round_, sharp = _fingertips(plain), _fingertips(part)
    for f in DIGIT_END:
        assert 0.0065 < sharp[f][0] < 0.0095 and abs(round_[f][0]) < 0.0005, (f, sharp[f], round_[f])
        assert sharp[f][1] < 0.6 * round_[f][1], (f, sharp[f], round_[f])
    W = plain.meshes[0].weights
    last = np.max([np.asarray(W[b.replace("左", s)]) for _, b in DIGIT_END.values() for s in ("左", "右")], axis=0)
    A, B = np.asarray(plain.meshes[0].verts), np.asarray(part.meshes[0].verts)
    assert A.shape == B.shape and np.array_equal(A[last == 0.0], B[last == 0.0])


def test_mesh_body_takes_the_designed_hand_too(tmp_path):
    """Without a hand mesh the body's own designed hand is welded on at the wrists, as onto the generated arm."""
    part = make(tmp_path, 'body.hand={}')
    assert_closed_but_for_the_seam(part)
    assert "左人指３" in part.meshes[0].weights


def test_mesh_body_needs_its_asset(tmp_path):
    spec = SP.load("base:girl", ['body.source="mesh"', 'body.mesh=""'])
    with pytest.raises(BD.BuildError, match="needs `mesh`"):
        BD.run(spec, only="body", tex_dir=tmp_path)
    spec = SP.load("base:girl", ['body.source="mesh"', f'body.mesh="{tmp_path / "nowhere.npz"}"'])
    with pytest.raises(BD.BuildError, match="does not exist"):
        BD.run(spec, only="body", tex_dir=tmp_path)


@pytest.mark.parametrize("source", ["mesh", "procedural"])
def test_claws_fade_the_hands_towards_the_fingertips(tmp_path, source):
    """[body] claws paints the hands' tiles of the skin atlas: the skin colour at the wrist, the claw colour at the
    fingertips (the right hand shares the left one's tiles), and leaves the rest of the body as it was."""
    from PIL import Image
    over = [f'body.source="{source}"', 'body.claws={color = "#00ff00", start = 0.3, full = 0.9}']
    part = BD.run(SP.load("base:girl", over), only="body", tex_dir=tmp_path)[0]
    plain = BD.run(SP.load("base:girl", [f'body.source="{source}"']), only="body", tex_dir=tmp_path / "plain")[0]
    m = part.meshes[0]
    img = np.asarray(Image.open(tmp_path / part.materials[0].texture).convert("RGB"), float) / 255.0
    ref = np.asarray(Image.open(tmp_path / "plain" / plain.materials[0].texture).convert("RGB"), float) / 255.0
    V = np.asarray(m.verts)
    corners = np.array([i for f in m.faces for i in f])
    skin = np.repeat(np.asarray(m.face_mat), [len(f) for f in m.faces]) == 0
    L = part.info["landmarks"]

    def colour(k):
        u, v = m.uv[k]
        return img[int((1.0 - v) * (img.shape[0] - 1)), int(u * (img.shape[1] - 1))]
    for side in ("L", "R"):
        w = np.asarray(L[f"wrist.{side}"], float)
        out = w - np.asarray(L[f"elbow.{side}"], float)
        d = np.linalg.norm(V[corners] - w, axis=1)
        hand = skin & ((V[corners] - w) @ out > 0) & (d < 0.25)         # beyond the wrist and near it: the hand
        d = np.where(hand, d, np.nan)
        tip = int(np.nanargmax(d))
        assert colour(tip)[1] > 0.9 and colour(tip)[0] < 0.2           # green at the fingertip
        near = int(np.nanargmin(np.where(d > 0.01, d, np.nan)))
        c = colour(near)
        assert c[0] > 0.8 and c[1] - c[0] < 0.05                          # skin by the wrist
    changed = np.abs(img - ref).max(2) > 0.02
    assert 0 < changed.mean() < 0.4                         # the hands' tiles only (up to a third of the atlas)
