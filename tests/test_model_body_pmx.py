"""The body part in pmx mode (mkmmd.model.parts.body_pmx): a PMX made here by the procedural body part plus a face shell and
a decoy garment is taken back, scaled, split into visible and covered skin, fitted with colliders. bpy-free."""
import numpy as np
import pytest

from mkmmd.core import bonemap
from mkmmd.model import assemble, build, geo, pmx_io
from mkmmd.model import part as P
from mkmmd.model.parts import body_pmx
from mkmmd.model.parts import body_pmx_fit as F

S_BODY, S_HEAD = 0.95, 0.985
HIDE = {"groups": ["torso", "shoulder", "arm"], "neck_z": 1.07, "wrist_back": 0.045, "leg_z": 0.50}


HEAD_SPHERE = (0.0, -0.012, 1.31, 0.0861)


def procedural(tmp, **body):
    spec = {"model": {"name": "t", "parts": ["body"], "seed": 1}}
    if body:
        spec["body"] = body
    return build.run(spec, only="body", tex_dir=tmp)[0]


@pytest.fixture(scope="module")
def source(tmp_path_factory):
    """(path of a source PMX, the part it was made from). Materials: skin, under (the torso and arms), face, clothes (a box
    on a decoy bone chain that the take has to prune)."""
    tmp = tmp_path_factory.mktemp("src")
    body = procedural(tmp / "tex", nails=False)
    m = body.meshes[0]
    names = [b.name for b in body.bones]
    groups = F.bone_groups(names)
    G = F.vertex_groups(m.weights, groups, len(m.verts))
    fg = np.array([int(np.argmax(G[list(f)].sum(0))) for f in m.faces])
    covered = np.isin(fg, [F.GROUPS.index(g) for g in ("torso", "shoulder", "arm")])
    skin = P.Mesh("skin", m.verts, m.faces, uv=m.uv, face_mat=covered.astype(int), mats=["skin", "under"],
                  weights=m.weights, normals=m.normals)
    head = np.asarray(body.info["landmarks"]["head"], float)
    sph = geo.uv_sphere(0.09, center=tuple(head + np.array([0.0, -0.01, 0.09])), segments=16, rings=8)
    face = sph.to_mesh("face", mats=["face"], weights={"頭": np.ones(sph.n_verts)})
    box = geo.uv_sphere(0.12, center=(0.0, -0.05, 1.0), segments=8, rings=4)
    cloth = box.to_mesh("clothes", mats=["clothes"], weights={"上半身": np.full(box.n_verts, 0.5), "スカート1": np.full(box.n_verts, 0.5)})
    decoy = P.Bone("スカート1", (0.0, -0.05, 0.9), (0.0, -0.05, 0.8), parent="下半身")
    mats = [P.Material(n, diffuse=(1, 1, 1, 1), ambient=(0.5, 0.5, 0.5)) for n in ("skin", "under", "face", "clothes")]
    part = P.Part("src", meshes=[skin, face, cloth], materials=mats, bones=list(body.bones) + [decoy], frames=body.frames)
    asm = assemble.assemble([part], name="src", scale=0.08)
    path = tmp / "src.pmx"
    pmx_io.write(asm.pmx, str(path))
    return path, body


@pytest.fixture(scope="module")
def part(source, tmp_path_factory):
    path, _ = source
    spec = {"model": {"name": "t", "parts": ["body"], "seed": 1}, "proportions": {"s_body": S_BODY, "s_head": S_HEAD},
            "body": {"source": "pmx", "pmx": {"path": str(path), "materials": ["skin", "under", "face"],
                                              "skin": ["skin", "under"], "hide": dict(HIDE),
                                              "landmark_bones": {"toe": "つま先"}, "head_sphere": list(HEAD_SPHERE)}}}
    return build.run(spec, only="body", tex_dir=tmp_path_factory.mktemp("tex"))[0]


def test_part_is_valid_and_complete(part, source):
    P.check(part)
    names = {b.name for b in part.bones}
    smap = bonemap.build_map({n: n for n in names})
    assert not [s for s in bonemap.REQUIRED if s not in smap]
    assert "スカート1" not in names                                      # the decoy chain only served a dropped material
    assert [m.name for m in part.materials] == ["skin", "skin_hidden"]
    assert part.meshes[0].name == "body" and part.info["mesh"] == "body" and part.info["source"] == "pmx"
    assert part.info["take"] is part.info["base"]
    assert all(b.semantic == smap_inv for b, smap_inv in [(b, {v: k for k, v in smap.items()}.get(b.name, "")) for b in part.bones])
    assert set(part.frames) and all(set(v) <= names for v in part.frames.values())


def test_scale_and_head_blend(part, source):
    _, body = source
    src = {k: np.asarray(v, float) for k, v in body.info["landmarks"].items()}
    land = part.info["landmarks"]
    for k in ("wrist.L", "knee.R", "leg_ik.L", "neck", "head", "upper_body", "index3.L"):
        assert np.allclose(land[k], S_BODY * src[k], atol=2e-6), k            # the body scales about the floor
    head = S_BODY * src["head"]
    assert np.allclose(land["head"], head, atol=2e-6)                          # the head pivot stays where the body puts it
    tip = head + (S_HEAD / S_BODY) * (S_BODY * src["head_tip"] - head)
    assert np.allclose(land["head_tip"], tip, atol=2e-6)                       # the head bone's tail grows to the head scale
    assert abs(float(part.info["take"].verts[:, 2].max()) - part.info["height"]) < 1e-9


def test_weights_valid_and_every_vertex_weighted(part):
    m = part.meshes[0]
    names = list(m.weights)
    M = np.stack([m.weights[n] for n in names], 1)
    assert (M >= 0).all() and np.allclose(M.sum(1), 1.0, atol=1e-6)
    assert ((M > 1e-6).sum(1) <= 4).all()
    bones = {b.name: b for b in part.bones}
    assert all(n in bones and bones[n].deform for n in names)


def test_covered_skin_is_split_off(part):
    m = part.meshes[0]
    V = np.asarray(m.verts)
    tris = np.array(m.faces, int).reshape(-1, 3)
    c = V[tris].mean(1)
    hidden = np.asarray(m.face_mat) == 1
    land = part.info["landmarks"]
    assert set(np.unique(m.face_mat)) == {0, 1} and m.mats == ["skin", "skin_hidden"]
    chest = (np.abs(c[:, 0]) < 0.06) & (c[:, 2] > 0.95) & (c[:, 2] < 1.05)
    assert chest.sum() > 20 and hidden[chest].all()                              # the bodice area is covered
    shin = (c[:, 2] < 0.45) & (c[:, 2] > 0.15)
    assert shin.sum() > 50 and not hidden[shin].any()                            # knees and shins stay skin
    thigh = (c[:, 2] > 0.55) & (c[:, 2] < 0.70) & (np.abs(c[:, 0]) > 0.04) & (np.abs(c[:, 0]) < 0.10)
    assert hidden[thigh].mean() > 0.9                                            # under the skirt
    hand = np.linalg.norm(c - land["middle2.L"], axis=1) < 0.03
    assert hand.sum() > 5 and not hidden[hand].any()
    e, w = np.asarray(land["elbow.L"]), np.asarray(land["wrist.L"])
    u = (w - e) / np.linalg.norm(w - e)
    s = (c - e) @ u
    forearm = (np.linalg.norm((c - e) - np.outer(s, u), axis=1) < 0.04) & (c[:, 0] > 0)
    end = forearm & (s > np.linalg.norm(w - e) - 0.040) & (s < np.linalg.norm(w - e) - 0.004)
    mid = forearm & (s > 0.3 * np.linalg.norm(w - e)) & (s < 0.6 * np.linalg.norm(w - e))
    assert end.sum() > 3 and not hidden[end].any()                               # inside the cuff the arm continues
    assert mid.sum() > 3 and hidden[mid].all()
    neck = (np.abs(c[:, 0]) < 0.03) & (c[:, 2] > HIDE["neck_z"]) & (c[:, 2] < land["head"][2])
    assert neck.sum() > 3 and not hidden[neck].any()
    assert part.info["hidden"]["faces"] == int(hidden.sum())


def test_skin_materials_copy_the_source(part):
    vis, hid = part.materials
    assert vis.name == "skin" and hid.name == "skin_hidden"
    assert hid.diffuse[3] == 0.0 and not hid.edge and hid.alpha_blend
    assert hid.diffuse[:3] == vis.diffuse[:3] and hid.ambient == vis.ambient and hid.specular == vis.specular
    assert vis.diffuse[3] == 1.0


def test_colliders_hug_and_mirror(part):
    bodies = {b.name: b for b in part.bodies}
    need = ["col_head", "col_neck", "col_upper_body", "col_upper_body2", "col_lower_body"]
    need += [f"col_{r}_{s}" for r in ("shoulder", "arm", "forearm", "hand", "thigh", "foot") for s in "LR"]
    need += [f"{n}_{s}" for n in F.LOWER_LEG for s in "LR"]
    assert not [n for n in need if n not in bodies]
    bones = {b.name for b in part.bones}
    for b in bodies.values():
        assert b.mode == "static" and b.group == 0 and b.bone in bones and b.no_collide == (0,)
        assert max(b.size) < 0.4
    for stem in [f"col_{r}" for r in ("shoulder", "arm", "forearm", "hand", "thigh", "foot")] + list(F.LOWER_LEG):
        assert bodies[f"{stem}_L"].size == bodies[f"{stem}_R"].size
        assert np.allclose(bodies[f"{stem}_R"].location, np.asarray(bodies[f"{stem}_L"].location) * [-1, 1, 1], atol=1e-6)
    head = bodies["col_head"]                                                      # the configured sphere, not the fitted one
    assert head.shape == "sphere" and np.allclose(head.location, HEAD_SPHERE[:3]) and abs(head.size[0] - HEAD_SPHERE[3]) < 1e-9
    assert 0.02 < bodies["col_arm_L"].size[0] < 0.045 and 0.015 < bodies["col_forearm_L"].size[0] < 0.04
    assert 0.04 < bodies["col_thigh_L"].size[0] < 0.075 and 0.025 < bodies["col_shin_L"].size[0] < 0.05
    m = part.meshes[0]
    V = np.asarray(m.verts)
    arm = bodies["col_arm_L"]
    a = np.asarray(arm.location)
    land = part.info["landmarks"]
    d, t = F._seg_dist(V, np.asarray(land["arm.L"]), np.asarray(land["elbow.L"]))
    on = (t > 0.1) & (t < 0.9) & (d < 0.06) & (V[:, 0] > 0)
    assert on.sum() > 30 and (d[on] < arm.size[0] * 1.15).mean() > 0.9           # the capsule really sits on the arm


def _capsule_inside(b, P):
    """How far inside capsule `b` (a `capsule` body: size = (radius, cylinder length), z along the body) the points P are,
    metres (negative: outside)."""
    R = np.array([[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]])
    rx, ry, rz = b.rotation
    cx, sx, cy, sy, cz, sz = np.cos(rx), np.sin(rx), np.cos(ry), np.sin(ry), np.cos(rz), np.sin(rz)
    R = np.array([[cz * cy, cz * sy * sx - sz * cx, cz * sy * cx + sz * sx],
                  [sz * cy, sz * sy * sx + cz * cx, sz * sy * cx - cz * sx], [-sy, cy * sx, cy * cx]])
    q = (np.asarray(P) - np.asarray(b.location)) @ R
    z = np.clip(q[:, 2], -0.5 * b.size[1], 0.5 * b.size[1])
    return b.size[0] - np.linalg.norm(q - np.stack([np.zeros(len(q)), np.zeros(len(q)), z], 1), axis=1)


def test_lower_leg_collider_tapers_and_covers_the_shin(part):
    """The shin is a stack of capsules, not one fat one: the skin of the shin sits inside it everywhere (a couple of mm at
    least) and the ankle end is clearly slimmer than the calf."""
    bodies = {b.name: b for b in part.bodies}
    assert len(F.LOWER_LEG) >= 4
    for side in "LR":
        stack = [bodies[f"{n}_{side}"] for n in F.LOWER_LEG]
        assert all(b.shape == "capsule" and b.bone == ("左ひざ" if side == "L" else "右ひざ") for b in stack)
        radii = [b.size[0] for b in stack]
        assert max(radii) - min(radii) > 0.012 and radii[0] > radii[-1] + 0.008          # the taper
        zs = [b.location[2] for b in stack]
        assert zs == sorted(zs, reverse=True)                                             # from the knee down
        m = part.meshes[0]
        V = np.asarray(m.verts)
        land = part.info["landmarks"]
        kn, an = np.asarray(land[f"knee.{side}"]), np.asarray(land[f"ankle.{side}"])
        d, t = F._seg_dist(V, kn, an)
        G = F.vertex_groups(m.weights, part.info["bone_groups"], len(V))
        on = (t > 0.15) & (t < 0.9) & (d < 0.08) & (np.argmax(G, 1) == F.GROUPS.index("leg")) & (G.sum(1) > 0.5)
        on &= (V[:, 0] * (1 if side == "L" else -1)) > 0
        assert on.sum() > 40
        inside = np.max([_capsule_inside(b, V[on]) for b in stack], axis=0)
        assert inside.min() > 0.0015 and np.median(inside) < 0.02                         # covered, and not by a fat tube


def test_tapered_chain_follows_a_cone():
    """A cone-shaped limb (radius 4.5 cm falling to 2.5 cm): the stack hugs it from outside, closer than one capsule could."""
    rng = np.random.default_rng(1)
    z = rng.uniform(0.0, 1.0, 3000)
    ang = rng.uniform(0, 2 * np.pi, 3000)
    r = 0.045 - 0.02 * z
    P = np.stack([r * np.cos(ang), r * np.sin(ang), 0.4 - 0.4 * z], 1)
    a, b = np.array([0.0, 0, 0.4]), np.array([0.0, 0, 0.0])
    caps = F.tapered_chain(P, a, b, pieces=5, t_fit=(0.05, 0.95))
    assert len(caps) == 5
    radii = [c[2] for c in caps]
    assert radii == sorted(radii, reverse=True) and radii[0] < 0.05 and radii[-1] > 0.024
    inside = np.max([_capsule_inside(F.capsule("c", "x", *c), P[(P[:, 2] > 0.03) & (P[:, 2] < 0.37)]) for c in caps], axis=0)
    assert inside.min() > 0.0015 and np.median(inside) < 0.008                            # a single capsule would leave ~2 cm


def test_info_for_the_other_parts(part):
    land = part.info["landmarks"]
    need = ["tail_root", "tail_root.L", "tail_root.R", "bust_front", "toe_end.L", "head_tip", "waist"]
    assert not [k for k in need if k not in land]
    m = part.meshes[0]
    V = np.asarray(m.verts)
    tris = np.array(m.faces, int).reshape(-1, 3)
    tr = np.asarray(land["tail_root"])
    hit = F.raycast(V, tris, tr + np.array([0.0, 0.05, 0.0]), (0.0, -1.0, 0.0))
    assert hit is not None and abs(hit[0] - 0.05) < 1e-4                           # the tail root is ON the skin
    n = np.asarray(part.info["tail_normal"])
    assert abs(np.linalg.norm(n) - 1) < 1e-6 and n[1] > 0.5                       # facing backwards (+Y), a little up
    assert land["bust_front"][2] > land["upper_body2"][2] and land["bust_front"][1] < land["upper_body"][1]
    tp = part.info["torso_profile"]
    assert set(tp) == {"rows", "half_width", "y_front", "y_back"} and len(tp["rows"]) > 10
    assert (tp["y_front"] < tp["y_back"]).all() and (tp["half_width"] > 0.03).all()
    nt = part.info["neck_top"]
    ring = np.asarray(nt["ring"])
    assert ring.shape == (nt["n"], 3) and np.allclose(ring[:, 2], nt["z"])
    sk = part.info["skin"]
    assert sk["material"] == "skin" and sk["hidden"] == "skin_hidden"
    assert set(part.info["regions"]) == set(F.GROUPS) and len(part.info["regions"]["torso"]) > 100


def test_resolve_hide_derives_the_neck_line_from_the_collar():
    h = body_pmx.resolve_hide({"leg_z": 0.5, "neck_inside": 0.02}, {"outfit_guides": {"collar_top_z": 1.1593}})
    assert abs(h["neck_z"] - 1.1393) < 1e-9 and h["leg_z"] == 0.5
    assert body_pmx.resolve_hide({"neck_z": 1.0}, {"outfit_guides": {"collar_top_z": 1.2}})["neck_z"] == 1.0
    assert "neck_z" not in body_pmx.resolve_hide({}, {})


def test_missing_configuration_is_an_error(tmp_path):
    spec = {"model": {"name": "t", "parts": ["body"]}, "body": {"source": "pmx", "pmx": {"materials": ["a"]}}}
    with pytest.raises(build.BuildError, match="path"):
        build.run(spec, only="body", tex_dir=tmp_path)
    spec = {"model": {"name": "t", "parts": ["body"]}, "body": {"source": "nonsense"}}
    with pytest.raises(build.BuildError, match="source"):
        build.run(spec, only="body", tex_dir=tmp_path)


def test_bone_groups_by_semantics_and_by_name():
    g = F.bone_groups(["上半身", "左腕捩2", "右ひじ", "左手首", "右人指３", "左足", "右足つま先", "首", "頭", "左肩C", "リボン"])
    assert g["上半身"] == "torso" and g["左腕捩2"] == "arm" and g["右ひじ"] == "arm" and g["左手首"] == "hand"
    assert g["右人指３"] == "hand" and g["左足"] == "leg" and g["右足つま先"] == "leg"
    assert g["首"] == "neck" and g["頭"] == "head" and g["左肩C"] == "shoulder" and "リボン" not in g


def test_raycast_and_slice_on_a_cube():
    V = np.array([[x, y, z] for x in (-1.0, 1.0) for y in (-1.0, 1.0) for z in (0.0, 2.0)])
    box = geo.box_mesh if hasattr(geo, "box_mesh") else None
    quads = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
    T = np.array([t for q in quads for t in ((q[0], q[1], q[2]), (q[0], q[2], q[3]))])
    t, k, _ = F.raycast(V, T, (0.2, 5.0, 1.0), (0.0, -1.0, 0.0))
    assert abs(t - 4.0) < 1e-9
    assert F.raycast(V, T, (3.0, 5.0, 1.0), (0.0, -1.0, 0.0)) is None
    seg = F.slice_z(V, T, 1.0)
    pts = seg.reshape(-1, 3)
    assert np.allclose(pts[:, 2], 1.0) and abs(np.abs(pts[:, 0]).max() - 1.0) < 1e-9 and abs(np.abs(pts[:, 1]).max() - 1.0) < 1e-9
    e = F.section_extent(V, T, 1.0)
    assert e == (-1.0, 1.0, -1.0, 1.0) and F.section_extent(V, T, 5.0) is None


def test_fits_recover_known_shapes():
    rng = np.random.default_rng(0)
    pts = rng.normal(size=(400, 3))
    pts = 0.1 * pts / np.linalg.norm(pts, axis=1, keepdims=True) + np.array([0.0, -0.02, 1.3])
    c, r = F.sphere_fit(pts)
    assert np.allclose(c, [0.0, -0.02, 1.3], atol=1e-6) and abs(r - 0.1) < 1e-6
    ang = rng.uniform(0, 2 * np.pi, 600)
    z = rng.uniform(0.0, 1.0, 600)
    cyl = np.stack([0.03 * np.cos(ang), 0.03 * np.sin(ang), z], 1)
    assert abs(F.capsule_fit(cyl, np.array([0.0, 0, 0.0]), np.array([0.0, 0, 1.0]), q=0.95) - 0.03) < 1e-6
    c, half = F.box_fit(np.array([[x, y, z] for x in (-0.1, 0.1) for y in (-0.05, 0.05) for z in (0.0, 0.2)]), np.eye(3), q=1.0, pad=0.0)
    assert np.allclose(c, [0, 0, 0.1]) and np.allclose(half, [0.1, 0.05, 0.1])
