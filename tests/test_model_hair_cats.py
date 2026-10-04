"""The cat ears and tails of the hair part (mkmmd/model/parts/ears.py, tails.py): bones and chains classify as ears / tail, the
twitch bones are exactly the ear-family bones whose parent is not an ear bone, weights sum to 1, `part.check` passes, bones
are connected, clearances against the body's static colliders and the head, determinism, vertex budgets.

bpy-free: the stand-in head and body of tests/hair_fixtures.py (plus, when the project's spec is at hand, the real body part).
Run: ~/.local/share/uv/tools/mk-mmd/bin/python -m pytest -q tests/test_model_hair_cats.py"""
from pathlib import Path

import numpy as np
import pytest

from hair_fixtures import make_ctx
from mkmmd.core import families
from mkmmd.model import build as BD
from mkmmd.model import part as P
from mkmmd.model.parts import cat_common as CC
from mkmmd.model.parts import cat_ear_geo as EG
from mkmmd.model.parts import ears as E
from mkmmd.model.parts import hair_fit, hair_rig, hair_tex, tails as T


def build(tmp_path, ears=None, tails=None, which=("ears", "tails")):
    ctx = make_ctx(tmp_path, {})
    fit = hair_fit.HeadFit(ctx.parts["head"].info)
    rig = hair_rig.Rig()
    pal = hair_tex.palette()
    pieces = {}
    if "ears" in which:
        pieces["ears"] = E.build_ears(ctx, fit, dict(ears or {}), rig, pal)
    if "tails" in which:
        pieces["tails"] = T.build_tails(ctx, fit, dict(tails or {}), rig, pal)
    part = P.Part("hair", meshes=[m for p in pieces.values() for m in p.meshes],
                  materials=[m for p in pieces.values() for m in p.materials], bones=rig.bones, bodies=rig.bodies,
                  joints=rig.joints, info={k: v.info for k, v in pieces.items()})
    return ctx, fit, rig, pieces, part


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    return build(tmp_path_factory.mktemp("cats"))


def verts(piece):
    return np.concatenate([m.verts for m in piece.meshes], 0)


# ------------------------------------------------------------------------------------------------------ helpers
def test_pchip_is_monotone_and_hits_the_knots():
    x = np.array([0.0, 0.2, 0.5, 1.0])
    y = np.array([0.0, 0.1, 0.9, 1.0])
    f = CC.pchip(x, y)
    assert np.allclose(f(x), y)
    q = np.linspace(0, 1, 101)
    assert (np.diff(f(q)) >= -1e-12).all()


def test_rmf_frames_are_orthonormal_and_follow_the_curve():
    t = np.linspace(0, 1, 60)
    pts = np.stack([0.3 * np.sin(3 * t), 0.2 * t ** 2, t], -1)
    T_, N, B = CC.rmf(pts, [1.0, 0.0, 0.0])
    for a, b in ((T_, N), (T_, B), (N, B)):
        assert np.abs((a * b).sum(1)).max() < 1e-6
    assert np.allclose(np.linalg.norm(N, axis=1), 1) and np.allclose(np.linalg.norm(B, axis=1), 1)
    assert np.abs(np.diff(N, axis=0)).max() < 0.2                    # no sudden twist


def test_body_sdf_matches_the_shapes():
    sph = P.RigidBody("s", "b", "sphere", (0.1, 0, 0), (0, 0, 1.0))
    cap = P.RigidBody("c", "b", "capsule", (0.05, 0.2, 0), (0, 0, 0))
    box = P.RigidBody("x", "b", "box", (0.1, 0.2, 0.3), (0, 0, 0))
    assert CC.body_sdf(sph, [[0, 0, 1.3]])[0] == pytest.approx(0.2)
    assert CC.body_sdf(cap, [[0.15, 0, 0.0]])[0] == pytest.approx(0.10)
    assert CC.body_sdf(cap, [[0.0, 0.0, 0.25]])[0] == pytest.approx(0.25 - 0.1 - 0.05)
    assert CC.body_sdf(box, [[0.3, 0, 0]])[0] == pytest.approx(0.2)
    assert CC.body_sdf(box, [[0.0, 0, 0]])[0] == pytest.approx(-0.1)
    rot = P.RigidBody("r", "b", "box", (0.1, 0.2, 0.3), (0, 0, 0), (0, 0, np.pi / 2))
    assert CC.body_sdf(rot, [[0.4, 0, 0]])[0] == pytest.approx(0.2)         # rotated 90 deg about z: the long side along x


def test_mesh_distance_of_a_cube():
    v = np.array([[x, y, z] for x in (0, 1) for y in (0, 1) for z in (0, 1)], float)
    f = [[0, 2, 3, 1], [4, 5, 7, 6], [0, 1, 5, 4], [2, 6, 7, 3], [0, 4, 6, 2], [1, 3, 7, 5]]
    d = CC.mesh_distance([[0.5, 0.5, 1.5], [0.5, 0.5, 0.5], [2.0, 0.5, 0.5]], v, f)
    assert abs(d[0]) == pytest.approx(0.5)
    assert abs(d[1]) == pytest.approx(0.5)
    assert abs(d[2]) == pytest.approx(1.0)
    assert d[0] * d[1] < 0                                           # outside and inside have opposite signs


# ------------------------------------------------------------------------------------------------------ ears: rig
EAR_NAMES = {"L": ["猫耳左", "猫耳左1", "猫耳左2", "猫耳左3"], "R": ["猫耳右", "猫耳右1", "猫耳右2", "猫耳右3"]}


def test_ear_bones_classify_and_twitch_bones_are_the_roots(built):
    ctx, fit, rig, pieces, part = built
    info = pieces["ears"].info
    names = [b.name for b in rig.bones if "耳" in b.name]
    assert sorted(names) == sorted(EAR_NAMES["L"] + EAR_NAMES["R"])
    by = {b.name: b for b in rig.bones}
    for n in names:
        assert families.classify(n) == "ears"
    twitch = [n for n in names if families.classify(by[n].parent) != "ears"]
    assert sorted(twitch) == ["猫耳右", "猫耳左"] and sorted(info["twitch"]) == ["猫耳右", "猫耳左"]
    for n in twitch:
        assert by[n].parent == "頭"
    for side, ns in EAR_NAMES.items():
        assert [by[n].parent for n in ns[1:]] == [ns[0], ns[1], ns[2]]


def test_ear_bones_are_connected_and_point_up_the_ear(built):
    ctx, fit, rig, pieces, part = built
    by = {b.name: b for b in rig.bones}
    for side, ns in EAR_NAMES.items():
        for a, b in zip(ns[:-1], ns[1:]):
            assert np.allclose(by[a].tail, by[b].head)                # connected: tail of a is the head of b
            assert by[a].tail_bone == b
        sd = pieces["ears"].info["sides"][side]
        assert np.allclose(by[ns[0]].head, sd["base_centre"])        # the twitch bone pivots about the ear base centre
        axis = np.asarray(by[ns[0]].tail) - np.asarray(by[ns[0]].head)
        axis /= np.linalg.norm(axis)
        assert axis @ sd["axis"] > 0.995                              # along the ear's up direction
        tip = np.asarray(by[ns[-1]].tail)
        assert np.allclose(tip, sd["tip"], atol=1e-6)                 # the chain ends at the tip of the shell


def test_ear_static_joint_partner_and_dynamic_chain(built):
    ctx, fit, rig, pieces, part = built
    bodies = {b.name: b for b in rig.bodies}
    for side, ns in EAR_NAMES.items():
        st = bodies[ns[0]]
        assert st.mode == "static" and st.bone == ns[0] and st.shape == "sphere" and st.size[0] <= 0.01
        for n in ns[1:]:
            assert bodies[n].mode == "dynamic" and bodies[n].bone == n
        joints = {j.name: j for j in rig.joints}
        assert joints[f"J_{ns[1]}"].a == ns[0] and joints[f"J_{ns[1]}"].b == ns[1]      # the twitch body is the partner
        assert joints[f"J_{ns[2]}"].a == ns[1]


def test_ears_are_in_the_ears_family_group(built):
    ctx, fit, rig, pieces, part = built
    for n in EAR_NAMES["L"][1:] + EAR_NAMES["R"][1:]:
        body = next(b for b in rig.bodies if b.name == n)
        assert body.group in (hair_rig.GROUPS["ears"], hair_rig.ROOT_GROUP)


# ------------------------------------------------------------------------------------------------------ ears: mesh
def test_ear_part_checks_and_weights_sum_to_one(built):
    ctx, fit, rig, pieces, part = built
    P.check(part)
    for m in part.meshes:
        tot = sum(np.asarray(w, float) for w in m.weights.values())
        assert np.allclose(tot, 1.0, atol=1e-6), (m.name, tot.min(), tot.max())
    body = ctx.parts["body"]
    BD.check_refs([body, part])                                       # unknown parents / weights / materials would raise


def test_ear_vertex_budget_and_materials(built):
    ctx, fit, rig, pieces, part = built
    ears = pieces["ears"]
    n = sum(len(m.verts) for m in ears.meshes)
    assert 150 < n <= 900, n
    assert [m.name for m in ears.materials] == ["猫耳", "猫耳内", "猫耳毛"]
    for m in ears.materials:
        assert m.edge and m.texture and m.toon
    used = set()
    for m in ears.meshes:
        used |= {m.mats[i] for i in set(m.face_mat.tolist())}
    assert used == {"猫耳", "猫耳内", "猫耳毛"}
    assert (ctx.tex_dir / ears.materials[0].texture).exists() and (ctx.tex_dir / ears.materials[1].toon).exists()


def test_ear_size_follows_the_head_and_the_config(tmp_path):
    ctx, fit, rig, pieces, part = build(tmp_path, ears={"width": 0.08, "height": 0.10}, which=("ears",))
    sz = pieces["ears"].info["size"]
    assert sz["width"] == pytest.approx(0.08) and sz["height"] == pytest.approx(0.10)
    ctx2, fit2, rig2, p2, part2 = build(tmp_path / "b", which=("ears",))
    assert p2["ears"].info["size"]["width"] == pytest.approx(E.DEFAULTS["width_ratio"] * sz["head_width"])
    assert p2["ears"].info["size"]["height"] == pytest.approx(E.DEFAULTS["height_ratio"] * sz["head_width"])
    assert 0.17 < sz["head_width"] < 0.21


def test_ears_stand_on_the_scalp_tilted_outward_and_forward(built):
    ctx, fit, rig, pieces, part = built
    for side, sg in (("L", 1.0), ("R", -1.0)):
        sd = pieces["ears"].info["sides"][side]
        u = sd["axis"]
        tilt_out = np.degrees(np.arctan2(sg * u[0], u[2]))
        tilt_fwd = np.degrees(np.arctan2(-u[1], u[2]))
        assert 15 <= tilt_out <= 20, tilt_out
        assert 3 <= tilt_fwd <= 12, tilt_fwd
        assert np.linalg.norm(sd["origin"] - fit.cat[side]["origin"]) < 1e-9
        assert abs(fit.skull.height_dist(sd["origin"])) < 0.003       # the anchor is on the skin
        # the base centre is sunk about 5 mm into the skin
        assert sd["base_centre"] @ u == pytest.approx(sd["origin"] @ u - 0.005)
        assert sd["tip"][2] > sd["origin"][2] + 0.10                   # a tall shell above the skin
        assert sg * (sd["tip"][0] - sd["origin"][0]) > 0.02             # the tip leans outward
        assert sd["tip"][1] < sd["origin"][1]                           # and forward (-y)


def test_ears_are_buried_in_the_hair_and_never_pierce_the_skin(built):
    ctx, fit, rig, pieces, part = built
    for side in ("L", "R"):
        sd = pieces["ears"].info["sides"][side]
        fp = sd["footprint"]                                           # the base ring, clipped onto the sunk skin
        hd = fit.skull.height_dist(fp)
        assert hd.min() > -0.0065 and hd.max() < 0.0 + 0.012, (hd.min(), hd.max())
        # the whole shell is above the sunk skin (height_dist >= -sink) except the closing cap vertex
    V = verts(pieces["ears"])
    hd = fit.skull.height_dist(V)
    assert np.sort(hd)[2] > -0.0075                                    # only the two cap-centre vertices dip lower
    # the shell reaches clear of the hair volume: the tips stand about the configured height above the hair hull
    for side in ("L", "R"):
        sd = pieces["ears"].info["sides"][side]
        th, ph = hair_fit.angles(sd["tip"] - fit.center)
        above = fit.skull.height_dist(sd["tip"]) - float(fit.vol(th, ph))
        assert 0.07 < above < 0.13, above


def test_ears_mirror_each_other(built):
    ctx, fit, rig, pieces, part = built
    s = pieces["ears"].info["sides"]
    for k in ("origin", "base_centre", "tip"):
        a, b = s["L"][k], s["R"][k]
        assert np.allclose([a[0], a[1], a[2]], [-b[0], b[1], b[2]], atol=0.003), k


def test_ear_base_is_rigid_on_the_twitch_bone_and_the_tip_bends(built):
    ctx, fit, rig, pieces, part = built
    m = pieces["ears"].meshes[0]
    for side, ns in EAR_NAMES.items():
        sd = pieces["ears"].info["sides"][side]
        w = {b: np.asarray(m.weights.get(b, np.zeros(len(m.verts))), float) for b in ns}
        # base ring + everything below the first chain point: 100 % twitch bone
        base = sd["footprint"]
        idx = [int(np.argmin(np.linalg.norm(m.verts - p, axis=1))) for p in base]
        assert (w[ns[0]][idx] > 0.999).all()
        tip = int(np.argmin(np.linalg.norm(m.verts - sd["tip"], axis=1)))
        assert w[ns[3]][tip] > 0.6 and w[ns[0]][tip] < 0.01
        for b in ns[1:]:
            assert w[b].max() > 0.5                                    # every chain bone owns part of the shell


def test_ear_inner_face_is_recessed_behind_the_rim(built):
    ctx, fit, rig, pieces, part = built
    m = pieces["ears"].meshes[0]
    sd = pieces["ears"].info["sides"]["L"]
    inner = m.mats.index("猫耳内")
    face_idx = np.flatnonzero(m.face_mat == inner)
    assert len(face_idx) > 40
    ey = sd["facing"]
    V = m.verts
    # the inner faces lie behind the rim crest as seen along the facing direction
    inner_v = np.unique([i for f in face_idx for i in m.faces[f]])
    dots = (V[inner_v] - sd["base_centre"]) @ ey
    black = np.unique([i for f in np.flatnonzero(m.face_mat == m.mats.index("猫耳")) for i in m.faces[f]])
    assert dots.size and (V[black] @ ey).max() > (V[inner_v] @ ey).max() - 1e-9


def test_tufts_are_pale_locks_at_the_lower_inner_edge(built):
    ctx, fit, rig, pieces, part = built
    for side in ("L", "R"):
        sd = pieces["ears"].info["sides"][side]
        assert 3 <= len(sd["tufts"]) <= 5
        for t in sd["tufts"]:
            L = np.linalg.norm(np.diff(t, axis=0), axis=1).sum()
            assert 0.015 < L < 0.05
            # roots in the lower half of the visible ear, on the inner (head-midline) side of its axis
            root = t[0]
            assert (root - sd["base_centre"]) @ sd["axis"] < 0.55 * (sd["tip"] - sd["base_centre"]) @ sd["axis"]
            assert (root - sd["base_centre"]) @ sd["outward"] < 0.0


def test_ear_clearance_from_head_colliders_and_skin(built):
    ctx, fit, rig, pieces, part = built
    cols = CC.static_colliders(ctx)
    assert cols
    for side, ns in EAR_NAMES.items():
        ch = pieces["ears"].info["sides"][side]["chain"]
        by = {b.name: b for b in rig.bodies}
        r = [by[n].size[0] for n in ns[1:]]
        gaps = CC.chain_clearance(cols, ch, r)
        assert min(g for g, _ in gaps) >= 0.005, gaps
        for k in range(len(ch) - 1):
            hd = fit.skull.height_dist(ch[k:k + 2])
            assert hd.min() - r[k] >= 0.005                           # clear of the head skin too


def test_ears_determinism(tmp_path):
    a = build(tmp_path / "a", which=("ears",))[3]["ears"]
    b = build(tmp_path / "b", which=("ears",))[3]["ears"]
    for ma, mb in zip(a.meshes, b.meshes):
        assert np.array_equal(ma.verts, mb.verts) and ma.faces == mb.faces and np.array_equal(ma.uv, mb.uv)
        for k in ma.weights:
            assert np.array_equal(ma.weights[k], mb.weights[k])


def test_ears_can_be_disabled(tmp_path):
    ctx, fit, rig, pieces, part = build(tmp_path, ears={"enabled": False}, tails={"enabled": False})
    assert not pieces["ears"].meshes and not pieces["tails"].meshes and not rig.bones


def test_ear_tuft_count_is_configurable(tmp_path):
    n0 = len(build(tmp_path / "a", ears={"tufts": 0}, which=("ears",))[3]["ears"].meshes[0].verts)
    n5 = len(build(tmp_path / "b", ears={"tufts": 5}, which=("ears",))[3]["ears"].meshes[0].verts)
    assert n5 > n0 + 100


def test_ear_shell_planform_is_a_triangle_with_a_point():
    sh = EG.EarShape(EG.EarParams())
    wid = [float(np.subtract(*sh.edges(t)[::-1])) for t in (0.0, 0.25, 0.5, 0.75, 0.95)]
    assert wid == sorted(wid, reverse=True) and wid[0] == pytest.approx(0.0739, abs=1e-3) and wid[-1] < 0.015


# ------------------------------------------------------------------------------------------------------ tails: rig
def tail_names(i, n=9):
    return [f"尻尾{i}_{k + 1}" for k in range(n)]


def test_tail_bones_classify_and_chain_from_the_lower_body(built):
    ctx, fit, rig, pieces, part = built
    by = {b.name: b for b in rig.bones}
    for i in (1, 2):
        ns = tail_names(i)
        assert all(n in by for n in ns)
        for n in ns:
            assert families.classify(n) == "tail"
        assert by[ns[0]].parent == "下半身"
        assert [by[n].parent for n in ns[1:]] == ns[:-1]
        for a, b in zip(ns[:-1], ns[1:]):
            assert np.allclose(by[a].tail, by[b].head) and by[a].tail_bone == b
    assert [b.name for b in rig.bones if families.classify(b.name) == "tail"] == tail_names(1) + tail_names(2)


def test_tail_sides_left_is_plus_x(built):
    ctx, fit, rig, pieces, part = built
    ch = pieces["tails"].info["chains"]
    assert ch[1][0][0] > 0 > ch[2][0][0] and ch[1][-1][0] > 0.15 and ch[2][-1][0] < -0.15
    assert np.allclose(ch[1] * [-1, 1, 1], ch[2], atol=1e-6)         # mirror images


def test_tail_bodies_and_joints(built):
    ctx, fit, rig, pieces, part = built
    bodies = {b.name: b for b in rig.bodies}
    joints = {j.name: j for j in rig.joints}
    low = ctx.find_body("下半身")
    for i in (1, 2):
        ns = tail_names(i)
        assert joints[f"J_{ns[0]}"].a == low.name and joints[f"J_{ns[0]}"].b == ns[0]
        for a, b in zip(ns[:-1], ns[1:]):
            assert joints[f"J_{b}"].a == a
        for n in ns:
            assert bodies[n].mode == "dynamic"
            assert bodies[n].group in (hair_rig.GROUPS["tail"], hair_rig.ROOT_GROUP)


def test_tail_part_checks_and_weights_sum_to_one(built):
    ctx, fit, rig, pieces, part = built
    P.check(part)
    m = pieces["tails"].meshes[0]
    tot = sum(np.asarray(w, float) for w in m.weights.values())
    assert np.allclose(tot, 1.0, atol=1e-6)
    assert set(m.weights) <= {"下半身"} | set(tail_names(1)) | set(tail_names(2))
    BD.check_refs([ctx.parts["body"], part])


def test_tail_mesh_is_a_closed_tapered_tube(built):
    ctx, fit, rig, pieces, part = built
    t = pieces["tails"]
    m = t.meshes[0]
    assert 300 < len(m.verts) <= 1100, len(m.verts)
    assert [x.name for x in t.materials] == ["尻尾"] and t.materials[0].edge and t.materials[0].texture
    # every edge shared by two faces: a union of closed shells (the tubes and the spikes)
    from collections import Counter
    cnt = Counter()
    for f in m.faces:
        for a, b in zip(f, list(f[1:]) + [f[0]]):
            cnt[(min(a, b), max(a, b))] += 1
    assert set(cnt.values()) == {2}
    # radius: 2.4 cm at the root, 6 mm just before the rounded cap, a point at the tip, thinning past the fur bulge
    cfg = T.DEFAULTS
    L = cfg["length"]
    cap = T.TIP_ROUND * cfg["radius_tip"]
    assert cfg["radius_root"] == pytest.approx(0.024) and cfg["radius_tip"] == pytest.approx(0.006)
    assert T.radius_profile(0.0, L, cfg) == pytest.approx(cfg["radius_root"])
    assert T.radius_profile(L - cap, L, cfg) == pytest.approx(cfg["radius_tip"], rel=1e-6)
    assert T.radius_profile(L, L, cfg) == pytest.approx(0.0, abs=1e-9)
    r = T.radius_profile(np.linspace(0, L, 200), L, cfg)
    assert r.max() < 1.3 * cfg["radius_root"] and (np.diff(r[70:]) <= 1e-9).all()
    assert T.radius_profile(0.5 * L, L, cfg) < 0.65 * cfg["radius_root"]        # clearly tapered by the middle


def test_tail_length_and_shape(built):
    ctx, fit, rig, pieces, part = built
    info = pieces["tails"].info
    for i in (1, 2):
        pts = info["chains"][i]
        assert len(pts) == 10
        seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        assert np.ptp(seg) < 0.03 * seg.mean()                          # (nearly) equal bones
        L = seg.sum() + info["roots"][i]["chain_start"]
        assert L == pytest.approx(T.DEFAULTS["length"], abs=0.01)
        assert pts[:, 2].max() > pts[0][2] + 0.12                       # rises from the lower back ...
        assert pts[:, 2].max() < 1.12                                   # ... but the apex stays below the shoulder line
        assert pts[-1][1] > pts[0][1]                                   # and ends behind the root
        ang = [np.degrees(np.arccos(np.clip(np.dot(a, b), -1, 1))) for a, b in
               zip(np.diff(pts, axis=0)[:-1] / seg[:-1, None], np.diff(pts, axis=0)[1:] / seg[1:, None])]
        assert max(ang) < 45                                            # a smooth S, no kinks


def test_tails_are_a_soft_s_curve(built):
    """Rises, bows outward and comes back in at the tip; the profile is steep at the root, flat at the apex (below the
    shoulder line, so the tails never frame the face) and the tip dips a little."""
    ctx, fit, rig, pieces, part = built
    for i in (1, 2):
        sg = 1.0 if i == 1 else -1.0
        pts = pieces["tails"].info["chains"][i]
        x = sg * pts[:, 0]
        k = int(np.argmax(x))
        assert 3 <= k < len(pts) - 1, k                                  # the widest point is not at the tip
        assert x[k] - x[-1] > 0.008, x                                   # the tip has come back in
        assert pts[-1][2] < pts[:, 2].max() - 0.01                       # and dips below the apex
        d = np.diff(pts, axis=0)
        el = np.degrees(np.arcsin(d[:, 2] / np.linalg.norm(d, axis=1)))  # elevation of every bone
        assert el[0] > 35 and el[-1] < el[0] - 30, el                    # steep at the root, falling at the tip
    kn = T.DEFAULTS["path"]                                              # the knots themselves
    el, hd = np.array(kn["elev"]), np.array(kn["head"])
    assert el[0] - el[-1] >= 40 and hd.max() > 45 and hd[-1] < 0


def test_tail_tip_spikes(built):
    """3-5 thin pointed shells fanning from the last 6-8 cm, 12-20 mm long, rigid on the last two bones."""
    ctx, fit, rig, pieces, part = built
    info = pieces["tails"].info
    m = pieces["tails"].meshes[0]
    W = {b: np.asarray(w, float) for b, w in m.weights.items()}
    for i in (1, 2):
        sp = info["spikes"][i]
        assert 3 <= len(sp) <= 5
        last2 = tail_names(i)[-2:]
        for s in sp:
            off, n = s["verts"]
            assert 0.0115 <= s["visible"] <= 0.0205, s["visible"]
            assert s["tip_out"] > 0.003, s["tip_out"]                      # the tip stands clear of the tube
            assert 0.012 <= np.linalg.norm(s["tip"] - s["root"]) <= 0.045
            assert np.linalg.norm(s["root"] - info["chains"][i][-1]) < 0.085   # rooted in the last 6-8 cm
            assert np.allclose(sum(w[off:off + n] for w in W.values()), 1.0, atol=1e-6)
            for b, w in W.items():
                if b not in last2:
                    assert w[off:off + n].max() < 1e-9, b                    # nothing but the last two bones
            assert sum(W[b][off:off + n].min() for b in last2) > 0.999
        assert len({round(float(s["fan_deg"]), 1) for s in sp}) == len(sp)    # they fan: every angle differs


def test_tail_spikes_can_be_switched_off(tmp_path):
    n_on = len(build(tmp_path / "a", which=("tails",))[3]["tails"].meshes[0].verts)
    off = build(tmp_path / "b", tails={"spikes": 0}, which=("tails",))[3]["tails"]
    assert off.info["spikes"] == {1: [], 2: []} and n_on - len(off.meshes[0].verts) >= 3 * 21 * 2


def test_tail_vertices_have_at_most_four_bones(built):
    ctx, fit, rig, pieces, part = built
    m = pieces["tails"].meshes[0]
    cnt = sum((np.asarray(w) > 1e-6).astype(int) for w in m.weights.values())
    assert cnt.max() <= 4 and cnt.min() >= 1


def test_tail_material_shares_the_ears_look(built):
    ctx, fit, rig, pieces, part = built
    mt, ear = pieces["tails"].materials[0], pieces["ears"].materials[0]
    assert mt.toon == ear.toon and mt.edge_color == ear.edge_color       # the ears' toon ramp and outline colour
    assert mt.sphere and mt.sphere_mode == "add" and (ctx.tex_dir / mt.sphere).exists()
    from PIL import Image
    sph = np.asarray(Image.open(ctx.tex_dir / mt.sphere).convert("RGB"), float) / 255.0
    assert 0.05 < sph.max() <= 0.2 and sph[0, 0].max() == 0.0             # a faint glow, black outside the disk
    fur = np.asarray(Image.open(ctx.tex_dir / mt.texture).convert("RGB"), float) / 255.0
    outer = hair_tex.srgb(CC.COLORS["outer"])
    assert np.abs(np.median(fur.reshape(-1, 3), axis=0) - outer).max() < 0.06      # colors.ears.outer
    assert fur.mean(axis=2).max() < 0.35                                  # never leaves the near-black family


def test_tail_clearance_from_the_body_colliders(built):
    ctx, fit, rig, pieces, part = built
    cols = CC.static_colliders(ctx)
    bodies = {b.name: b for b in rig.bodies}
    worst = np.inf
    for i in (1, 2):
        ns = tail_names(i)
        pts = pieces["tails"].info["chains"][i]
        gaps = CC.chain_clearance(cols, pts, [bodies[n].size[0] for n in ns])
        worst = min(worst, min(g for g, _ in gaps))
        assert np.allclose([g for g, _ in gaps], pieces["tails"].info["clearance"][i])
    assert worst >= 0.005, worst


def test_tail_axis_is_outside_the_body_skin(built):
    ctx, fit, rig, pieces, part = built
    body = ctx.parts["body"].meshes[0]
    for i in (1, 2):
        pts = pieces["tails"].info["chains"][i]
        d = CC.mesh_distance(pts, body.verts, body.faces)
        assert (d > 0.004).all(), d                                     # the chain axis never dips under the skin


def test_tails_rise_over_the_back_and_stay_off_the_head(built):
    ctx, fit, rig, pieces, part = built
    info = pieces["tails"].info
    V = verts(pieces["tails"])
    body = ctx.parts["body"].meshes[0]
    for i in (1, 2):
        root = info["roots"][i]["root"]
        far = V[np.linalg.norm(V - root, axis=1) > 0.12]               # beyond the stump that runs into the body
        d = CC.mesh_distance(far, body.verts, body.faces)
        assert d.min() > 0.002, d.min()                                 # no tube vertex is inside the body there
    hd = fit.skull.height_dist(V)
    assert hd.min() > 0.05                                              # well clear of the head (and its hair)


def test_tail_roots_and_frames_published(built):
    ctx, fit, rig, pieces, part = built
    land = ctx.land
    info = pieces["tails"].info
    assert np.allclose(info["roots"][1]["root"], land.get("tail_root.L", land["tail_root"] + [T.DEFAULTS["root_x"], 0, 0]))
    assert np.allclose(info["roots"][2]["root"], land.get("tail_root.R", land["tail_root"] - [T.DEFAULTS["root_x"], 0, 0]))
    assert info["bones"] == tail_names(1) + tail_names(2)


def test_tails_determinism_and_switch(tmp_path):
    a = build(tmp_path / "a", which=("tails",))[3]["tails"]
    b = build(tmp_path / "b", which=("tails",))[3]["tails"]
    assert np.array_equal(a.meshes[0].verts, b.meshes[0].verts) and np.array_equal(a.meshes[0].uv, b.meshes[0].uv)
    for k in a.meshes[0].weights:
        assert np.array_equal(a.meshes[0].weights[k], b.meshes[0].weights[k])
    off = build(tmp_path / "c", tails={"enabled": False}, which=("tails",))
    assert not off[3]["tails"].meshes and not [x for x in off[2].bones if "尻尾" in x.name]


def test_tail_path_config_changes_the_shape(tmp_path):
    ctx, fit, rig, pieces, part = build(tmp_path, tails={"length": 0.45}, which=("tails",))
    pts = pieces["tails"].info["chains"][1]
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1).sum() + pieces["tails"].info["roots"][1]["chain_start"]
    assert seg == pytest.approx(0.45, abs=0.01)


# ------------------------------------------------------------------------------------------------------ the real body
SPEC = Path.home() / "Projects/mk-tests/rin_model/model.toml"


@pytest.mark.skipif(not SPEC.exists(), reason="the project spec of the Rin model is not at hand")
def test_tails_against_the_real_body_part(tmp_path):
    from mkmmd.model import spec as SP
    try:
        body = BD.run(SP.load(str(SPEC)), only="body", tex_dir=str(tmp_path / "tex"))[0]
    except Exception as e:                                              # the real body is under construction elsewhere
        pytest.skip(f"real body part does not build: {e}")
    ctx = make_ctx(tmp_path, {})
    ctx.parts["body"] = body
    ctx.land.clear()
    for k, v in body.info["landmarks"].items():
        ctx.land[k] = np.asarray(v, float)
    fit = hair_fit.HeadFit(ctx.parts["head"].info)
    rig = hair_rig.Rig()
    piece = T.build_tails(ctx, fit, {}, rig, hair_tex.palette())
    cols = CC.static_colliders(ctx)
    assert cols
    bodies = {b.name: b for b in rig.bodies}
    for i in (1, 2):
        ns = tail_names(i)
        gaps = CC.chain_clearance(cols, piece.info["chains"][i], [bodies[n].size[0] for n in ns])
        assert min(g for g, _ in gaps) >= 0.005
        sk = body.meshes[0]
        d = CC.mesh_distance(piece.info["chains"][i], sk.verts, sk.faces)
        assert (d > 0).all()
    part = P.Part("hair", meshes=piece.meshes, materials=piece.materials, bones=rig.bones, bodies=rig.bodies, joints=rig.joints)
    P.check(part)
    BD.check_refs([body, part])
