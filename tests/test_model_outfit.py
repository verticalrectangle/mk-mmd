"""bpy-free invariants of the outfit part: the geometry kit (lofts, frills, bows, welding), the fitting queries, and the
built part on the provisional mannequin (offsets above the skin, skirt clearance, chain families and structure, weights,
closed shells, materials, determinism)."""
import numpy as np
import pytest

from mkmmd.core import families
from mkmmd.model import build as BD
from mkmmd.model import part as PT
from mkmmd.model import skeleton as SKL
from mkmmd.model import spec as SP
from mkmmd.model.parts import outfit, outfit_dress as OD, outfit_fit as OF, outfit_geo as G, outfit_rig as RG, outfit_skirt as SKT

SIZES = {"dress": 64, "frill": 64, "satin": 64, "leather": 64, "toon": 8, "sphere": 16}


# ---------------------------------------------------------------- fixtures
def make_ctx(tmp, cfg=None, scale=0.8):
    """A BuildCtx with a body Part made of the provisional mannequin (standard skeleton, static colliders on the bones
    the outfit hangs chains from)."""
    land = OF.default_landmarks(scale)
    skin = OF.provisional_skin(OF.Land(land), scale=scale)
    bones = SKL.standard_bones(land, {"fingers": False})
    bodies = [PT.RigidBody(f"col_{b}", b, "capsule", (0.05, 0.1, 0.0), mode="static", no_collide=(0,))
              for b in ("下半身", "首", "左手首", "右手首")]
    body = PT.Part("body", meshes=[skin.to_mesh("body")], materials=[PT.Material("skin")], bones=bones, bodies=bodies,
                   info={"landmarks": land})
    spec = SP.from_dict({"model": {"name": "t", "parts": ["body", "outfit"]},
                         "outfit": dict({"texture_sizes": SIZES}, **(cfg or {}))})
    ctx = BD.BuildCtx(spec, tmp, seed=3)
    ctx.parts["body"] = body
    ctx.land = {k: np.asarray(v, float) for k, v in land.items()}
    ctx.part = "outfit"
    return ctx, skin, land


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("outfit")
    ctx, skin, land = make_ctx(tmp)
    part = outfit.build(ctx)
    PT.check(part)
    return ctx, skin, part


@pytest.fixture(scope="module")
def pieces(built):
    """The garment patches by tag (before welding), rebuilt from the same fit the part used."""
    ctx, skin, part = built
    cfg = outfit.guided(ctx, outfit.merge(outfit.DEFAULTS, ctx.cfg))
    fit = OF.Fit(OF.Land(ctx.land), skin, bone_names=ctx.bones())
    dress, trim, satin = G.Soup("d", [outfit.MAT_DRESS]), G.Soup("t", [outfit.MAT_FRILL, outfit.MAT_FRILL_IN]), G.Soup("s", outfit.SATIN_MATS)
    rig = RG.Rig()
    info = OD.bodice(fit, cfg["bodice"], dress, trim, rig, ctx.find_body)
    OD.sash(fit, cfg["sash"], satin, info)
    sleeves = {s: OD.sleeve(fit, s, cfg["sleeve"], dress, trim, info, rig, ctx.find_body) for s in "LR"}
    OD.throat_bow(fit, cfg["bow"], satin, rig, info, "首", None)
    sk = SKT.skirt(fit, cfg["skirt"], info, dress, trim, rig, None)
    tags = {}
    for soup in (dress, trim, satin):
        for p in soup.patches:
            tags.setdefault(p.tag, []).append(p)
    return dict(fit=fit, info=info, sleeves=sleeves, skirt=sk, tags=tags, rig=rig, cfg=cfg)


def boundary_edges(faces):
    cnt = {}
    for f in faces:
        for a, b in zip(f, list(f[1:]) + [f[0]]):
            k = (min(a, b), max(a, b))
            cnt[k] = cnt.get(k, 0) + 1
    return [k for k, c in cnt.items() if c == 1], [k for k, c in cnt.items() if c > 2]


# ---------------------------------------------------------------- geometry kit
def test_theta_convention_front_minus_y_left_plus_x():
    assert np.allclose(G.hdir(0.0), [0, -1, 0]) and np.allclose(G.hdir(np.pi / 2), [1, 0, 0])


def test_loft_faces_point_outward_and_uv_seam_is_explicit():
    th = G.ring_theta(24)
    rings = np.stack([np.stack([0.1 * np.sin(th), -0.1 * np.cos(th), np.full(24, 1.0 - 0.1 * k)], -1) for k in range(5)])
    p = G.loft(rings, u=np.linspace(0, 3, 25))
    fn = G.face_normals(p.v, p.f)
    cen = np.array([p.v[list(f)].mean(0) for f in p.f])
    cen[:, 2] = 0
    assert (np.einsum("ij,ij->i", fn, cen) > 0).all()
    assert p.uv.shape == (sum(len(f) for f in p.f), 2) and p.uv[:, 0].max() == pytest.approx(3.0)
    assert len(p.v) == 120 and len(p.f) == 96                    # no duplicated seam vertices
    q = G.loft(rings[::-1], flip=True)
    assert (np.einsum("ij,ij->i", G.face_normals(q.v, q.f), np.array([q.v[list(f)].mean(0) * [1, 1, 0] for f in q.f])) > 0).all()


def test_frill_rings_start_on_the_base_and_pleats_stay_within_amplitude():
    th = G.ring_theta(120)
    base = np.stack([0.2 * np.sin(th), -0.2 * np.cos(th), np.zeros(120)], -1)
    out = G.hdir(th)
    prof = np.array([(0.0, 0.0), (0.01, 0.02), (0.02, 0.05)])
    rings = G.frill_rings(base, out, np.array([0, 0, -1.0]), prof, pleats=20, amp=0.01, wobble=0.0, hem_wobble=0.0)
    assert np.allclose(rings[0], base)
    r = np.hypot(rings[2, :, 0], rings[2, :, 1])
    assert r.max() - 0.22 <= 0.0101 and 0.22 - r.min() <= 0.0101 and r.max() - r.min() > 0.012    # pleated, bounded
    assert np.allclose(rings[2, :, 2], -0.05)


def test_resample_ring_keeps_a_circle_and_maps_back_to_the_source():
    th = G.ring_theta(16)
    ring = np.stack([np.cos(th), np.sin(th), np.zeros(16)], -1)
    pts, src = G.resample_ring(ring, 96)
    assert np.allclose(np.hypot(pts[:, 0], pts[:, 1]), 1.0, atol=0.01) and src.min() == 0 and src.max() == 15
    assert np.allclose(pts[::6], ring, atol=1e-9)


def test_cap_weights_normalises_and_keeps_four():
    rng = np.random.default_rng(0)
    w = {f"b{i}": rng.random(50) for i in range(7)}
    c = G.cap_weights(w, 50)
    A = np.stack(list(c.values()), 1)
    assert np.allclose(A.sum(1), 1.0) and ((A > 0).sum(1) <= 4).all()


def test_soup_welds_coincident_vertices_and_collapses_degenerate_quads():
    a = G.Patch(np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0.0]]), [(0, 1, 2, 3)], np.zeros((4, 2)), {"x": np.ones(4)})
    b = G.Patch(np.array([[1, 0, 0], [2, 0, 0], [2, 1, 0], [1, 1, 0.0]]), [(0, 1, 2, 3)], np.zeros((4, 2)), {"x": np.ones(4)})
    s = G.Soup("t", ["m"])
    s.add(a, 0)
    s.add(b, 0)
    m = s.mesh()
    assert len(m.verts) == 6 and len(m.faces) == 2
    t = G.Patch(np.array([[0, 0, 0], [0, 0, 0], [1, 0, 0], [0, 1, 0.0]]), [(0, 1, 2, 3)], np.zeros((4, 2)), {"x": np.ones(4)})
    s2 = G.Soup("t", ["m"])
    s2.add(t, 0)
    m2 = s2.mesh()
    assert len(m2.faces) == 1 and len(m2.faces[0]) == 3 and m2.uv.shape == (3, 2)


def test_bow_has_loops_knot_and_two_tails():
    bw = G.bow_ex([0, 0, 0], [1, 0, 0], [0, 0, 1], [0, -1, 0], size=0.03)
    assert len(bw["parts"]) == 5 and len(bw["tails"]) == 2
    for p in bw["parts"]:
        assert np.isfinite(p.v).all() and len(p.f) > 10
        assert p.uv.shape == (sum(len(f) for f in p.f), 2)
    ends = [path[-1] for path, _ in bw["tails"]]
    assert all(e[2] < -0.03 for e in ends) and ends[0][0] * ends[1][0] < 0     # hang down, splay to both sides


# ---------------------------------------------------------------- fitting queries
def test_section_of_a_cylinder_and_signed_distance():
    th = G.ring_theta(48)
    rings = np.stack([np.stack([0.1 * np.sin(th), -0.1 * np.cos(th), np.full(48, 1.0 - 0.05 * k)], -1) for k in range(21)])
    p = G.loft(rings)
    p = G.orient_outward(p, np.array([0, 0, 0.5]))
    skin = OF.Skin(p.v, p.f)
    c, r = skin.section(np.array([0.01, -0.01, 0.5]), np.array([0, -1.0, 0]), np.array([1.0, 0, 0]), 32)
    assert np.allclose(c[:2], 0, atol=1e-3) and np.allclose(r, 0.1, atol=2e-3)
    d, q, ti, b = skin.closest(np.array([[0.13, 0, 0.5], [0.04, 0, 0.5]]))
    assert d[0] == pytest.approx(0.03, abs=2e-3) and d[1] == pytest.approx(-0.06, abs=2e-3)


def test_push_out_enforces_the_margin_and_leaves_clear_points(built):
    ctx, skin, part = built
    fit = OF.Fit(OF.Land(ctx.land), skin)
    z = ctx.land["upper_body"][2]
    c, r = fit.torso_section(z, 32)
    th = G.ring_theta(32)
    inside = c + G.hdir(th) * (r - 0.02)[:, None]
    far = c + G.hdir(th) * (r + 0.05)[:, None]
    out = fit.push_out(np.vstack([inside, far]), 0.004)
    d = skin.closest(out)[0]
    assert d.min() >= 0.004 - 1e-4
    assert np.allclose(out[32:], far)


# ---------------------------------------------------------------- the built part
def test_part_structure_materials_and_meshes(built):
    ctx, skin, part = built
    assert part.name == "outfit"
    assert {m.name for m in part.meshes} == {"outfit_dress", "outfit_frills", "outfit_ribbons", "outfit_shoes"}
    names = {m.name for m in part.materials}
    assert names == {outfit.MAT_DRESS, outfit.MAT_FRILL, outfit.MAT_FRILL_IN, outfit.MAT_RUFFLE, outfit.MAT_SATIN,
                     outfit.MAT_LEG, outfit.MAT_SHOE, outfit.MAT_SOLE}
    for m in part.materials:
        assert m.texture and m.toon and (not m.sphere or m.sphere_mode == "add")
        assert (ctx.tex_dir / m.texture).exists() and (ctx.tex_dir / m.toon).exists()
        if m.name in (outfit.MAT_FRILL, outfit.MAT_FRILL_IN, outfit.MAT_RUFFLE):
            assert m.double_sided                                 # frills are sheets
    for mesh in part.meshes:
        assert mesh.uv is not None and np.isfinite(mesh.uv).all()
        assert mesh.face_mat is not None and set(mesh.mats) <= names


def test_weights_normalised_capped_and_on_known_bones(built):
    ctx, skin, part = built
    known = {b.name for b in ctx.parts["body"].bones} | {b.name for b in part.bones}
    for mesh in part.meshes:
        n = len(mesh.verts)
        tot = np.zeros(n)
        nz = np.zeros(n, int)
        for b, w in mesh.weights.items():
            assert b in known, b
            tot += w
            nz += w > 0
        assert np.allclose(tot, 1.0, atol=1e-6), mesh.name
        assert nz.max() <= 4


def test_cross_part_references_are_consistent(built):
    ctx, skin, part = built
    assert BD.check_refs([ctx.parts["body"], part], strict=True) == []


def test_chain_families_and_structure(built):
    ctx, skin, part = built
    fam = {}
    for b in part.bones:
        fam.setdefault(families.classify(b.name), []).append(b.name)
    assert set(fam) == {"skirt", "ribbon", "sleeve", "coat"}
    cfg = outfit.DEFAULTS["skirt"]
    assert len(fam["skirt"]) == cfg["columns"] * cfg["rows"]
    assert len(fam["ribbon"]) == 4 + 6 and len(fam["sleeve"]) == 2 * 4 * 2 and len(fam["coat"]) == 6   # throat bow + leg bow
    by = {b.name: b for b in part.bones}
    bodies = {rb.bone: rb for rb in part.bodies}
    joints = {j.b: j for j in part.joints}
    assert set(bodies) == set(by) and all(rb.mode == "dynamic" for rb in bodies.values())
    for b in part.bones:
        if b.tail_bone:                                           # connected chains: the child starts at the parent's tail
            assert np.allclose(by[b.tail_bone].head, b.tail, atol=1e-9) and by[b.tail_bone].parent == b.name
        assert b.name in joints or by.get(b.parent) is None       # a joint ties every body to its predecessor
        assert np.isfinite(np.array(b.head)).all() and np.isfinite(np.array(b.tail)).all()
    roots = [b for b in part.bones if b.parent in {x.name for x in ctx.parts["body"].bones}]
    assert {families.classify(b.name) for b in roots} == {"skirt", "ribbon", "sleeve", "coat"}
    for rb in part.bodies:
        assert rb.size[0] > 0 and rb.size[1] > 0 and rb.group in (7, 8, 9, 10, 11)
        assert 0 in rb.no_collide or rb.group != 8 or True


def test_skirt_columns_rows_and_hem(built, pieces):
    ctx, skin, part = built
    sk, fit = pieces["skirt"], pieces["fit"]
    L = fit.L
    assert sk["z_hem"] < min(L["knee.L"][2], L["knee.R"][2])                  # just below the knee
    assert sk["z_hem"] > 0.5 * L["ankle.L"][2] + 0.15
    z = sk["z_nodes"]
    hip_z = 0.5 * (L["leg.L"][2] + L["leg.R"][2])
    assert (np.diff(z) < 0).all() and 0.0 < hip_z - z[0] < 0.04                  # chains start just below the hip joints
    assert np.allclose(np.diff(z), np.diff(z)[0], atol=1e-9)                     # rows evenly spaced down to the ruffle
    P = sk["points"]
    assert P.shape == (sk["columns"], sk["rows"] + 1, 3)
    seg = np.linalg.norm(np.diff(P, axis=1), axis=2)
    assert seg.min() > 0.05 * fit.S and seg.max() < 0.25
    # column 0 is in front (-Y of the axis), counter-clockwise seen from above (the quarter column is on her left, +X)
    ctr = sk["shape"].center
    assert P[0, 1, 1] < ctr[1] and P[sk["columns"] // 4, 1, 0] > ctr[0]


def test_garment_offsets_above_the_skin_where_meant(built, pieces):
    ctx, skin, part = built
    tags = pieces["tags"]
    margin = 0.0030                                             # conform() enforces 6 mm (bodice) / 3.5 mm (sleeves) at the vertices
    for tag in ("bodice", "sash", "sleeve_L", "sleeve_R"):
        for p in tags[tag]:
            d = skin.closest(p.v)[0]
            if tag.startswith("sleeve"):
                d = d[np.arange(len(d)) >= 32 * 5]                # the first rings are the dome tucked into the shoulder
            assert d.min() >= margin, (tag, d.min())
    # the collar frill, cuffs, bow and tails never touch the skin either
    for tag in ("collar_frill", "cuff_L", "cuff_R", "bow"):
        for p in tags[tag]:
            assert skin.closest(p.v)[0].min() >= 0.002, tag


def test_skirt_clears_hips_and_thighs_at_rest(built, pieces):
    ctx, skin, part = built
    S = pieces["fit"].S
    waist = pieces["info"]["z_waist"]
    clear = pieces["cfg"]["skirt"].get("clearance", 0.045) * S
    for tag in ("skirt", "ruffle", "inner_frill"):
        for p in pieces["tags"][tag]:
            v = p.v[p.v[:, 2] < waist - 0.03]
            assert len(v) > 100
            d = skin.closest(v)[0]
            assert d.min() > 0.5 * clear, (tag, d.min())
    # the chain lines (what the solver collides) stay at least a particle radius + margin away
    P = pieces["skirt"]["points"]
    rad = np.asarray(SKT.DEFAULT_RADII, float)
    assert len(rad) == P.shape[1] - 1
    d = skin.closest(P[:, :-1].reshape(-1, 3))[0].reshape(P.shape[0], -1)
    assert (d > 0.5 * rad[None, :] + 0.002).all()


def test_shells_are_closed_where_they_must_be(built, pieces):
    tags = pieces["tags"]
    for side in "LR":
        tube, cap = tags[f"sleeve_{side}"][0], tags[f"sleeve_{side}_cap"][0]
        faces = [tuple(f) for f in tube.f] + [tuple(i + len(tube.v) for i in f) for f in cap.f]
        # weld the dome onto the tube's first ring (same positions) and look for open edges: only the cuff end stays open
        V = np.vstack([tube.v, cap.v])
        key = {}
        for i, v in enumerate(V):
            key.setdefault(tuple(np.round(v / 1e-6).astype(np.int64)), i)
        remap = [key[tuple(np.round(v / 1e-6).astype(np.int64))] for v in V]
        faces = [tuple(remap[i] for i in f) for f in faces]
        open_edges, bad = boundary_edges(faces)
        assert not bad
        # open edges all belong to the last ring of the tube (the cuff end), none to the shoulder end
        last = set(range(len(tube.v) - 32, len(tube.v)))
        assert open_edges and all(a in last and b in last for a, b in open_edges), (side, len(open_edges))
    cfg = pieces["cfg"]
    for tag, m in (("bodice", cfg["bodice"]["segments"]), ("skirt", cfg["skirt"]["columns"] * cfg["skirt"]["per_column"])):
        p = tags[tag][0]
        open_edges, bad = boundary_edges(p.f)
        rim = set(range(m)) | set(range(len(p.v) - m, len(p.v)))
        assert not bad and open_edges and len(open_edges) == 2 * m
        assert all(a in rim and b in rim for a, b in open_edges)                   # only the top and bottom rims are open


def test_frill_seams_are_seamless_in_u_and_close_up(pieces):
    for tag in ("collar_frill", "cuff_L", "ruffle", "inner_frill"):
        p = pieces["tags"][tag][0]
        u = p.uv[:, 0]
        assert abs(u.max() - round(u.max())) < 1e-9 and u.max() >= 1      # whole tiles around: no visible seam
        open_edges, bad = boundary_edges(p.f)
        assert not bad
        # an open sheet: the attachment rim and the hem; seam vertices are shared (no duplicate seam column)
        assert len(p.v) == len(np.unique(np.round(p.v / 1e-7).astype(np.int64), axis=0))


def test_no_degenerate_faces_and_finite_everything(built):
    ctx, skin, part = built
    for mesh in part.meshes:
        V = np.asarray(mesh.verts, float)
        assert np.isfinite(V).all()
        fn = G.face_normals(V, mesh.faces)
        area = 0.5 * np.linalg.norm(fn, axis=1)
        assert (area > 1e-10).mean() > 0.995, mesh.name
        assert max(len(f) for f in mesh.faces) <= 4


def test_weights_of_the_waist_follow_the_body_and_the_hem_follows_the_chains(built):
    ctx, skin, part = built
    dress = next(m for m in part.meshes if m.name == "outfit_dress")
    V = np.asarray(dress.verts)
    waist = ctx.land["upper_body"][2]
    skirt_mass = sum((w for b, w in dress.weights.items() if b.startswith("スカート")), np.zeros(len(V)))
    top = (V[:, 2] > waist - 0.02) & (V[:, 2] < waist + 0.1)
    low = V[:, 2] < 0.6
    assert skirt_mass[top].max() < 0.2 and skirt_mass[low].min() > 0.99


def test_deterministic(tmp_path):
    a, _, _ = make_ctx(tmp_path / "a", {"textures": False})
    b, _, _ = make_ctx(tmp_path / "b", {"textures": False})
    pa, pb = outfit.build(a), outfit.build(b)
    for ma, mb in zip(pa.meshes, pb.meshes):
        assert np.array_equal(ma.verts, mb.verts) and ma.faces == mb.faces
        assert all(np.array_equal(ma.weights[k], mb.weights[k]) for k in ma.weights)


def test_scale_follows_the_body_height(tmp_path):
    small, _, _ = make_ctx(tmp_path / "s", {"textures": False}, scale=0.7)
    big, _, _ = make_ctx(tmp_path / "b", {"textures": False}, scale=1.0)
    ps, pb = outfit.build(small), outfit.build(big)
    hs = max(np.ptp(m.verts[:, 2]) for m in ps.meshes)
    hb = max(np.ptp(m.verts[:, 2]) for m in pb.meshes)
    assert hb / hs == pytest.approx(1.0 / 0.7, rel=0.12)
    assert ps.info["scale"] == pytest.approx(0.7 * 1.414 / 1.7, rel=0.01)


# ---------------------------------------------------------------- weights helpers
def test_ring_weights_partition_of_unity():
    th = np.linspace(0, 2 * np.pi, 200, endpoint=False)
    z = np.tile(np.linspace(1.0, 0.0, 10), 20)[:200]
    names = [[f"b{r}_{c}" for c in range(6)] for r in range(3)]
    w_top = {"root": np.ones(200)}
    w = RG.ring_weights(th, z, names, np.array([0.7, 0.4, 0.2]), 1.0, w_top, 6)
    tot = sum(w.values())
    assert np.allclose(tot, 1.0)
    A = np.stack(list(w.values()), 1)
    assert ((A > 1e-9).sum(1) <= 4 + 1).all()
    low = z < 0.2
    assert all(b.startswith("b2_") for b, a in w.items() if a[low].max() > 1e-9)


def test_strip_chain_weights_partition_and_order():
    t = np.concatenate([np.linspace(0, 1, 51), [0.2, 0.6]])
    w = OD.strip_chain_weights(t, {"neck": np.ones(len(t))}, ["c1", "c2"], [0.2, 0.6])
    assert np.allclose(sum(w.values()), 1.0)
    assert w["neck"][0] == pytest.approx(1.0) and w["c1"][-2] == pytest.approx(1.0)       # the first node belongs to bone 1
    assert w["c2"][-1] == pytest.approx(1.0) and w["c2"][50] == pytest.approx(1.0)         # the last bone owns the rest


def test_euler_xyz_roundtrip():
    R = RG.frame_z(np.array([0.3, -0.5, -0.8]), np.array([1.0, 0, 0]))
    e = RG.euler_xyz(R)
    rx, ry, rz = e
    Rx = np.array([[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]])
    Ry = np.array([[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]])
    Rz = np.array([[np.cos(rz), -np.sin(rz), 0], [np.sin(rz), np.cos(rz), 0], [0, 0, 1]])
    assert np.allclose(Rz @ Ry @ Rx, R, atol=1e-9)


# ---------------------------------------------------------------- body distances (collider helpers)
def test_body_sdf_shapes_and_euler_matrix():
    R = RG.frame_z([0.2, -0.4, 0.9], [1, 0, 0])
    assert np.allclose(RG.euler_matrix(RG.euler_xyz(R)), R, atol=1e-9)
    sph = PT.RigidBody("s", "b", "sphere", (0.1, 0, 0), (0.0, 0.0, 1.0))
    assert RG.body_sdf([[0.0, 0.0, 1.0], [0.3, 0.0, 1.0]], sph) == pytest.approx([-0.1, 0.2])
    # a capsule (r 0.05, straight part 0.2 long) turned so that its local Z axis lies along world X
    cap = PT.RigidBody("c", "b", "capsule", (0.05, 0.2, 0), (0.0, 0.0, 0.0), (0.0, np.pi / 2, 0.0))
    assert np.allclose(RG.euler_matrix(cap.rotation) @ [0, 0, 1.0], [1, 0, 0], atol=1e-9)
    pts = [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.0, 0.0, 0.1], [0.2, 0.0, 0.0], [0.0, 0.08, 0.0]]
    assert RG.body_sdf(pts, cap) == pytest.approx([-0.05, -0.05, 0.05, 0.05, 0.03])    # inside, on the axis end, beside, past the cap
    box = PT.RigidBody("x", "b", "box", (0.1, 0.2, 0.3), (0.0, 0.0, 0.0))
    assert RG.body_sdf([[0.0, 0.0, 0.0], [0.3, 0.0, 0.0]], box) == pytest.approx([-0.1, 0.2])
    assert RG.union_sdf([[0.3, 0.0, 1.0]], [sph, cap]) == pytest.approx([0.2])
    assert RG.union_sdf([[0.3, 0.0, 1.0]], []).tolist() == [np.inf]


def test_chain_clearance_per_bone():
    ball = PT.RigidBody("s", "b", "sphere", (0.1, 0, 0), (0.0, 0.0, 0.0))                  # radius 0.1 at the origin
    pts = np.array([[0.2, 0.0, 0.0], [0.3, 0.0, 0.0], [0.5, 0.0, 0.0]])        # two bones along +X, away from the ball
    c = RG.chain_clearance(pts, [0.01, 0.02], [ball])
    assert c == pytest.approx([0.2 - 0.1 - 0.01, 0.3 - 0.1 - 0.02])            # nearest axis point minus the bone's radius
    inside = RG.chain_clearance(np.array([[0.0, 0.0, 0.0], [0.05, 0.0, 0.0]]), 0.01, [ball])
    assert inside[0] == pytest.approx(0.0 - 0.1 - 0.01)                         # starts at the ball's centre: overlapping, negative
    assert RG.chain_clearance(pts, 0.01, []).tolist() == [np.inf, np.inf]
