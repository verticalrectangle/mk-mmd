"""bpy-free invariants of the hair part: head fit, swept clumps, chain rigging, the built part (families, weights, scalp
coverage, bangs vs eyes, clearance, determinism). The head and body are the stand-ins of hair_fixtures.py."""
import numpy as np
import pytest

import hair_fixtures as HF
from local_project import SPEC, WHY
from mkmmd.core import families
from mkmmd.model import build as BD
from mkmmd.model import part as PT
from mkmmd.model.parts import hair, hair_geo, hair_head, hair_tex
from mkmmd.model.parts.hair_fit import HeadFit, Skull, angles, dirs, raycast_first
from mkmmd.model.parts.hair_rig import DYNAMIC, Rig, euler_for_axis

# the sub-builders (braids, ears, tails) have their own test files; the part-level checks here run on the full part
SLICES = {"braids": {"enabled": True}, "ears": {"enabled": True}, "tails": {"enabled": True}}


@pytest.fixture(scope="module")
def head_fit():
    return HeadFit(HF.head_info())


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("hair")
    ctx = HF.make_ctx(tmp, {"hair": dict(SLICES)})
    part = hair.build(ctx)
    PT.check(part)
    return ctx, part


@pytest.fixture(scope="module")
def head_only(tmp_path_factory):
    """The head hair alone (no braids, ears, tails): cheap and independent of the other slices."""
    tmp = tmp_path_factory.mktemp("hair_head")
    ctx = HF.make_ctx(tmp)
    fit = HeadFit(ctx.parts["head"].info)
    rig = Rig()
    piece = hair_head.build_head_hair(ctx, fit, {}, rig, hair_tex.palette(ctx.get("colors.hair")))
    return ctx, fit, rig, piece


# ---------------------------------------------------------------- fit
def test_dirs_angles_roundtrip():
    th = np.radians([-170, -90, -10, 0, 45, 90, 179])
    ph = np.radians([5, 40, 90, 120, 60, 100, 170])
    t2, p2 = angles(dirs(th, ph))
    assert np.allclose(t2, th, atol=1e-9) and np.allclose(p2, ph, atol=1e-9)
    assert np.allclose(dirs(0.0, np.pi / 2), [0, -1, 0])          # front is -Y
    assert np.allclose(dirs(np.pi / 2, np.pi / 2), [1, 0, 0])     # the character's left is +X


def test_skull_from_mesh_matches_a_sphere():
    s = Skull.from_function(np.array([0.0, 0.0, 1.0]), lambda D: np.full(D.shape[:-1], 0.1))
    d = np.array([[0.3, -0.5, 0.2], [0, 0, 1], [-1, 0.2, -0.3]])
    assert np.allclose(s.radius(d), 0.1)
    p = s.point(d, 0.01)
    assert np.allclose(np.linalg.norm(p - s.center, axis=1), 0.11)
    n = s.normal(d)
    assert np.allclose(n, d / np.linalg.norm(d, axis=1, keepdims=True), atol=1e-3)
    q = np.array([[0.0, 0.0, 1.05], [0.0, 0.0, 1.5]])               # inside the shell (radius 0.1 about z = 1)
    out = s.push_out(q, 0.02)
    assert np.allclose(out[0], [0, 0, 1.12]) and np.allclose(out[1], q[1])


def test_raycast_first_hits_a_box():
    v = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], float)
    f = [[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1], [2, 3, 7], [2, 7, 6], [0, 2, 6], [0, 6, 4],
         [1, 5, 7], [1, 7, 3]]
    t = raycast_first(np.zeros(3), np.array([[1.0, 0, 0], [0, 0, -1.0]]), v, np.array(f))
    assert np.allclose(t, [1.0, 1.0])


def test_head_fit_region(head_fit):
    th = np.radians(np.arange(-180, 180, 10))
    cp = head_fit.cap_phi(th)
    assert np.isfinite(cp).all() and cp.min() > np.radians(20) and cp.max() < np.radians(175)
    # the front hairline is high on the forehead, the nape low
    assert head_fit.cap_phi(0.0) < head_fit.cap_phi(np.pi)
    D = head_fit.hair_region_dirs()
    assert len(D) > 300 and np.allclose(np.linalg.norm(D, axis=1), 1.0)
    v = head_fit.vol(th, np.full(len(th), np.radians(70)))
    assert (v > 0.01).all() and (v < 0.06).all()
    assert head_fit.vol(0.0, 0.0) == pytest.approx(head_fit.vol.top)


# ---------------------------------------------------------------- strips
@pytest.mark.parametrize("tip", ["point", "round", "blunt"])
def test_sweep_is_a_closed_outward_shell(tip):
    P = np.stack([np.linspace(0, 0.1, 30) * 0, np.zeros(30), 1.0 - np.linspace(0, 0.1, 30)], -1)
    N = np.array([0.0, -1.0, 0.0])
    st = hair_geo.sweep(P, N, 0.03, 0.008, tip=tip, curv=8.0, tilt=0.2 if tip == "blunt" else 0.0)
    n = len(st.verts)
    assert all(len(set(f)) == len(f) and 0 <= min(f) and max(f) < n for f in st.faces)
    used = np.zeros(n, bool)
    for f in st.faces:
        used[f] = True
    assert used.all()
    # watertight: every edge is shared by exactly two faces
    edges = {}
    for f in st.faces:
        for a, b in zip(f, f[1:] + f[:1]):
            edges[tuple(sorted((a, b)))] = edges.get(tuple(sorted((a, b))), 0) + 1
    assert set(edges.values()) == {2}
    fn = hair_geo.face_normals(st.verts, st.faces)
    cen = np.array([st.verts[f].mean(0) for f in st.faces])
    ctr = np.array([st.ctr[f].mean(0) for f in st.faces])
    assert ((cen - ctr) * fn).sum(1).__gt__(0).mean() > 0.95
    assert st.arc.min() == 0 and st.arc.max() == pytest.approx(st.length)
    assert np.isfinite(st.verts).all()


def test_sweep_width_taper_and_uv_ranges():
    P = np.stack([np.zeros(20), np.zeros(20), np.linspace(1.0, 0.9, 20)], -1)
    st = hair_geo.sweep(P, [0, -1, 0], 0.04, 0.008, tip="point", tip_start=0.5)
    widths = []
    for r in range(st.rings):
        ring = st.verts[st.ring == r]
        widths.append(np.ptp(ring[:, 0]))
    assert widths[0] <= widths[len(widths) // 3] + 1e-9 and widths[-1] < widths[len(widths) // 2]
    acc = hair_geo.MeshAccum("t", ["m"])
    uv = np.stack([(st.q + 1) / 2, st.arc / st.length], -1)
    acc.add(st.verts, st.faces, uv, 0, weights={"b": np.ones(len(st.verts))})
    m = acc.to_mesh()
    assert m.uv.shape == (sum(len(f) for f in m.faces), 2)
    PT.check(PT.Part("x", meshes=[m], materials=[PT.Material("m")], bones=[PT.Bone("b", (0, 0, 0))]))


# ---------------------------------------------------------------- rig
def test_chain_weights_partition_of_unity():
    rig = Rig()
    pts = np.array([[0, 0, 1.3], [0, 0, 1.25], [0, 0, 1.2], [0, 0, 1.15], [0, 0, 1.1]])
    ch = rig.chain("後髪1_", pts, "頭", kind="hair", radius=0.01, anchor_body="col_head")
    P = np.stack([np.zeros(60), np.zeros(60), np.linspace(1.35, 1.08, 60)], -1)
    w = ch.weights(P)
    tot = sum(w.values())
    assert np.allclose(tot, 1.0)
    assert set(w) == {"頭", *ch.names}
    # smooth: no jump between neighbouring samples; rigid at the very root, last bone at the tip
    for a in w.values():
        assert np.abs(np.diff(a)).max() < 0.35
    assert w["頭"][0] == pytest.approx(1.0) and w[ch.names[-1]][-1] == pytest.approx(1.0)
    # at most 3 bones share a vertex
    stack = np.stack(list(w.values()))
    assert ((stack > 1e-6).sum(0) <= 3).all()


def test_rig_chain_structure():
    rig = Rig()
    pts = np.array([[0.1, 0, 1.3], [0.1, 0, 1.25], [0.1, 0, 1.2], [0.1, 0, 1.1]])
    rig.chain("横髪左1_", pts, "頭", kind="hair", radius=0.012, anchor_body="col_head")
    assert [b.name for b in rig.bones] == ["横髪左1_1", "横髪左1_2", "横髪左1_3"]
    assert rig.bones[0].parent == "頭" and rig.bones[1].parent == "横髪左1_1"
    assert rig.bones[0].tail_bone == "横髪左1_2" and rig.bones[-1].tail_bone == ""
    assert np.allclose(rig.bones[-1].tail, pts[-1])
    assert [j.a for j in rig.joints] == ["col_head", "横髪左1_1", "横髪左1_2"]
    assert rig.bodies[0].group == 8 and 0 in rig.bodies[0].no_collide and rig.bodies[1].group == 4
    assert all(b.mode == "dynamic" for b in rig.bodies)
    assert all(rb.shape == "capsule" and rb.size[0] > 0 and rb.size[1] > 0 for rb in rig.bodies)


@pytest.mark.parametrize("d", [[0, 0, 1], [1, 0, 0], [0, 1, 0], [0.3, -0.5, -0.8], [0, 0, -1], [-0.7, 0.2, 0.1]])
def test_euler_for_axis_maps_z_to_direction(d):
    d = np.asarray(d, float) / np.linalg.norm(d)
    rx, ry, rz = euler_for_axis(d)
    Rx = np.array([[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]])
    Ry = np.array([[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]])
    Rz = np.array([[np.cos(rz), -np.sin(rz), 0], [np.sin(rz), np.cos(rz), 0], [0, 0, 1]])
    assert np.allclose(Rz @ Ry @ Rx @ [0, 0, 1], d, atol=1e-9)


# ---------------------------------------------------------------- textures
def test_textures():
    pal = hair_tex.palette()
    atlas = hair_tex.make_atlas(pal, np.random.default_rng(0))
    assert atlas.shape == (1024, 512, 4) and atlas.dtype == np.uint8 and (atlas[..., 3] == 255).all()
    ramp = hair_tex.toon_ramp()
    assert ramp[0, 0, :3].astype(int).sum() > ramp[-1, 0, :3].astype(int).sum()      # top lit, bottom shadow (mmd_tools)
    ring = hair_tex.sphere_ring(pal)
    assert ring.shape == (256, 256, 4) and ring[..., :3].max() > 100 and (ring[0, :, :3] == 0).all()
    # the baked crown sheen: a wide, feathered, light pink-coral band (B close to G: never peach, never white) with no hard
    # edge anywhere, fainter on the inner tiles, none low down (the fringe) and none on the cap tile
    T, N = hair_tex.TILE_PX, hair_tex.N_TILES
    f_ring, w_ring = 0.16, 0.05
    head = hair_tex.make_atlas(pal, np.random.default_rng(0), ring_f=f_ring, ring_w=w_ring)[:512].astype(int)
    tile0, tile2, cap = head[:, 4:T - 4], head[:, 2 * T + 4:3 * T - 4], head[:, N * T:]
    r_core, c_core = np.unravel_index(np.argmax(tile0[:, :, 1]), tile0[:, :, 1].shape)
    assert abs(r_core / 512 - f_ring) < 0.07
    core = tile0[r_core, c_core]
    assert core[0] > 215 and 130 < core[1] < 215 and abs(core[2] - core[1]) < 30, core
    col = tile0[:, c_core, 1]
    base_g = col[min(r_core + 160, 511)]
    assert (col > (col.max() + base_g) / 2).sum() >= 20                           # wide: >= 20 rows at half strength
    assert np.abs(np.diff(head[:, :, 1], axis=0)).max() <= 10                      # no hard edge along the height ...
    inside = np.ones(head.shape[1] - 1, bool)
    inside[[T - 1, 2 * T - 1, 3 * T - 1, N * T - 1]] = False                        # (tile borders are different clumps)
    assert np.abs(np.diff(head[:, :, 1], axis=1))[:, inside].max() <= 16            # ... nor across a clump (feathered window)
    assert tile2[:, :, 1].max() < tile0[:, :, 1].max() - 25                         # the inner tiles are fainter
    assert head[int(0.60 * 512):, :2 * T, 1].max() < 120                            # nothing low down
    assert cap[:, :, 1].max() < 70                                                  # the cap tile has no sheen and is dark
    ramp = hair_tex.toon_ramp(hair_tex.TOON_SHADOW, edge=0.52, soft=0.20)[:, 0, :3].astype(int)
    assert np.abs(np.diff(ramp, axis=0)).max() <= 20                                # the toon step is a soft slope
    z = np.linspace(1.4, 1.1, 5)
    v = hair_tex.head_v(z, 1.4, 1.1)
    assert (np.diff(v) < 0).all() and v.max() <= 1.0 and v.min() >= 0.5
    u0, u1 = hair_tex.clump_u(np.random.default_rng(1))
    assert 0 <= u0 < u1 <= hair_tex.CAP_U0


def test_crown_sheen_follows_the_colour_family():
    """The baked sheen keeps Rin's exact pink-coral for her own family, takes the hue of another family's highlight (a
    dark-blue family gets a blue sheen, not a pink band), yields to explicit `ring` keys, and can be switched off."""
    rin = hair_tex.palette()
    assert np.array_equal(rin["ring"], hair_tex.SHEEN) and np.array_equal(rin["ring_core"], hair_tex.SHEEN_CORE)
    blue = hair_tex.palette({"base": "#1e3063", "shadow": "#101a3a", "light": "#314980", "highlight": "#738dba",
                             "highlight_core": "#afbed1"})
    for key in ("ring", "ring_core"):
        r, g, b = blue[key]
        assert b > g > r, (key, blue[key])
    set_ = hair_tex.palette({"ring": "#806040", "ring_core": "#a08060"})
    assert np.allclose(set_["ring"] * 255, [0x80, 0x60, 0x40]) and np.allclose(set_["ring_core"] * 255, [0xa0, 0x80, 0x60])
    T = hair_tex.TILE_PX
    on = hair_tex.make_atlas(rin, np.random.default_rng(0))[:512, :T].astype(int)
    off = hair_tex.make_atlas(rin, np.random.default_rng(0), ring=False)[:512, :T].astype(int)
    assert on[:, :, 1].max() > off[:, :, 1].max() + 40                       # the sheen is the lightest thing on the crown
    assert np.array_equal(on[int(0.6 * 512):], off[int(0.6 * 512):])          # and nothing else changes


def test_chains_inside_the_body_colliders_are_reported():
    """`hair.overlaps` names every chain whose capsules overlap a static collider at rest, with its deepest bone; a
    clear chain, and a chain whose only overlap is its first body (that one ignores the colliders), are not reported."""
    box = PT.RigidBody(name="col_upper_body2", bone="上半身2", shape="box", size=(0.08, 0.06, 0.10),
                       location=(0.0, 0.0, 1.0), mode="static")
    rig = Rig()
    rig.chain("in_", [[0.0, 0.12, 1.30], [0.0, 0.10, 1.15], [0.0, 0.05, 1.05], [0.0, 0.04, 0.98]], "頭", radius=0.01)
    rig.chain("clear_", [[0.0, 0.20, 1.30], [0.0, 0.20, 1.15], [0.0, 0.20, 1.00]], "頭", radius=0.01)
    rig.chain("root_", [[0.0, 0.05, 1.05], [0.0, 0.10, 1.20], [0.0, 0.10, 1.35]], "頭", radius=0.01)
    found = hair.overlaps(rig, [box])
    assert [(base, bone, which) for _, base, bone, which in found] == [("in_", "in_3", "col_upper_body2")]
    assert -0.032 < found[0][0] < -0.025               # its deepest bone: axis 2 cm inside the box's back face, radius 1 cm


# ---------------------------------------------------------------- the head hair
def test_head_hair_chain_families(head_only):
    _, _, rig, piece = head_only
    exp = {"前髪": "bangs", "横髪": "side_hair", "後髪": "back_hair"}
    for b in rig.bones:
        assert families.classify(b.name) == exp[b.name[:2]], b.name
    assert set(piece.info["chains"]) == set(rig.chains.__len__() and piece.info["chains"])      # id -> names
    D = hair_head.DEFAULTS
    assert len(rig.chains) == D["bangs"]["count"] + 2 + (D["back"]["count"] - 1)            # bangs, side, back


def test_head_hair_weights_and_mesh(head_only):
    ctx, _, rig, piece = head_only
    m = piece.meshes[0]
    bones = {"頭"} | {b.name for b in rig.bones}
    assert set(m.weights) <= bones
    tot = sum(np.asarray(w) for w in m.weights.values())
    assert np.allclose(tot, 1.0, atol=1e-6)
    stack = np.stack([np.asarray(w) for w in m.weights.values()])
    assert ((stack > 1e-6).sum(0) <= 4).all()
    assert len(m.verts) < 12000
    assert np.isfinite(m.verts).all()
    assert m.uv.shape == (sum(len(f) for f in m.faces), 2) and 0 <= m.uv.min() and m.uv.max() <= 1.0
    mats = {x.name: x for x in piece.materials}
    # the angel ring is baked into the texture (a camera-relative sphere ring crosses the bangs like a headband)
    assert mats["髪"].sphere_mode == "none" and not mats["髪"].sphere and mats["髪"].texture and mats["髪"].toon
    for f in (mats["髪"].texture, mats["髪"].toon):
        assert (ctx.tex_dir / f).exists()


def test_scalp_has_no_gaps(head_only):
    """Rays from the head centre through the hair region always meet the hair beyond the skin, within 6 cm of it."""
    _, fit, _, piece = head_only
    m = piece.meshes[0]
    tris = np.array([[f[0], f[k], f[k + 1]] for f in m.faces for k in range(1, len(f) - 1)])
    D = fit.hair_region_dirs(step_deg=7.0)
    t = raycast_first(fit.center, D, m.verts, tris)
    skin = fit.skull.radius(D)
    assert np.isfinite(t).all(), f"{int((~np.isfinite(t)).sum())} rays escape between the hair and the scalp"
    assert (t > skin + 0.001).all() and (t < skin + 0.06).all()


def test_human_ears_are_hidden(head_only):
    """The innermost hair surface passes outside every point of the human ears (they sit under the hair)."""
    _, fit, _, piece = head_only
    m = piece.meshes[0]
    tris = np.array([[f[0], f[k], f[k + 1]] for f in m.faces for k in range(1, len(f) - 1)])
    pts = np.array([np.asarray(e[k], float) for e in fit.ears.values() for k in ("top", "lobe", "front", "back")])
    d = pts - fit.center
    t = raycast_first(fit.center, d / np.linalg.norm(d, axis=1, keepdims=True), m.verts, tris)
    assert (t > np.linalg.norm(d, axis=1) + 0.001).all()


def test_crown_lift_and_ring_height(head_only):
    """The crown stands well above the skull and more at the back than at the front; the baked ring sits on the upper
    crown, far above the fringe."""
    _, fit, _, piece = head_only
    assert piece.info["top_z"] - fit.top[2] >= 0.035
    ph = np.radians(28.0)
    assert fit.vol(np.pi, ph) > fit.vol(0.0, ph) + 0.01
    assert piece.info["ring_z"] - piece.info["bangs_bottom_z"] >= 0.06
    assert piece.info["ring_z"] > fit.hairline[:, 2].min()


def test_fringe_is_separate_pointed_clumps(head_only):
    """9-12 separate clumps across the forehead, tips within +-5 mm of a straight cut line (which follows the forehead a
    little), a darker back layer, 2-3 thin loose strands."""
    _, fit, rig, piece = head_only
    inf = piece.info
    assert 9 <= inf["bangs_front"] <= 12 and inf["bangs_back"] == inf["bangs_front"] - 1 and 2 <= inf["bangs_loose"] <= 3
    tips = np.array(inf["bangs_tip_z"])
    assert len(tips) == inf["bangs_front"]
    assert 0.006 <= np.ptp(tips) <= 0.016                                          # lengths vary
    assert abs(tips.mean() - inf["bangs_cut_z"]) < 0.004 and tips.std() < 0.005      # around one straight line
    chains = [k for k in inf["chains"] if k.startswith("bangs")]
    assert len(chains) == inf["bangs_front"]                                       # one chain per front clump


def test_fringe_shadow_decal(head_only):
    """Two cel-shading steps under the fringe: opaque, 0.3-1 mm above the skin, above the eyes, weighted to the head."""
    ctx, fit, rig, piece = head_only
    m = next(x for x in piece.meshes if x.name == "hair_shadow")
    mats = {x.name: x for x in piece.materials}
    assert set(m.mats) == {"髪影", "髪影淡"} and set(m.mats) <= set(mats)
    assert all(not mats[n].alpha_blend and not mats[n].edge for n in m.mats)
    assert set(m.face_mat.tolist()) == {0, 1}
    d = m.verts - fit.center
    gap = np.linalg.norm(d, axis=1) - fit.skull.radius(d)
    assert 0.0002 < gap.min() and gap.max() < 0.0012, (gap.min(), gap.max())
    lids = max(fit.lid_top(s)[2] for s in ("L", "R"))
    assert m.verts[:, 2].min() > lids + 0.004
    assert m.verts[:, 2].min() > piece.info["bangs_bottom_z"] - 0.012               # no lower than ~1 cm under the tips
    assert np.allclose(m.weights["頭"], 1.0)


def test_bangs_leave_the_eyes_open(head_only):
    """The bangs end above the upper lids (+8 mm) and below the hairline, so they may cross the brows only."""
    _, fit, _, piece = head_only
    m = piece.meshes[0]
    wb = np.sum([np.asarray(w) for b, w in m.weights.items() if b.startswith("前髪")], axis=0)
    low = m.verts[wb > 0.5][:, 2].min()
    lids = max(fit.lid_top(s)[2] for s in ("L", "R"))
    assert low >= lids + 0.008 - 1e-6
    assert low <= fit.brow_z() + 0.01                       # but they do reach the brow line (a blunt fringe)
    assert piece.info["bangs_bottom_z"] > lids + 0.008


def test_side_and_back_hair_lengths(head_only):
    _, fit, _, piece = head_only
    m = piece.meshes[0]
    wsum = lambda pre: np.sum([np.asarray(w) for b, w in m.weights.items() if b.startswith(pre)], axis=0)
    side = m.verts[wsum("横髪") > 0.5]
    back = m.verts[wsum("後髪") > 0.5]
    assert side[:, 2].min() == pytest.approx(fit.chin[2], abs=0.012)               # chin length
    assert fit.chin[2] - 0.045 < back[:, 2].min() < fit.chin[2] + 0.04              # nape length (covers the neck)
    assert side[:, 0].max() < 0.16 and back[:, 1].max() < 0.2


def test_hair_normals_follow_a_smooth_proxy(head_only):
    """One smooth terminator across the whole mass: the normals of the outward-facing hair vertices stay close to those of
    the smooth head-shaped proxy (no step per clump); the crown tilts up, the sides are horizontal, the lower hair tilts down."""
    ctx, fit, rig, piece = head_only
    m = piece.meshes[0]
    sh = hair_head.DEFAULTS["shade"]
    P, N = m.verts, m.normals
    proxy = hair_head.shade_proxy(P, fit.center, fit.center[2] + sh["z_hi"], fit.center[2] + sh["z_lo"])
    out = (N * (P - fit.center)).sum(1) > 0
    ang = np.degrees(np.arccos(np.clip((N * proxy).sum(1), -1, 1)))
    assert out.mean() > 0.4
    assert np.median(ang[out]) < 12 and np.percentile(ang[out], 95) < 32
    top = hair_head.shade_proxy(np.array([[0.0, 0.0, fit.top[2] + 0.03], [0.1, 0.0, fit.center[2]],
                                          [0.1, 0.0, fit.center[2] - 0.1]]), fit.center, fit.center[2], fit.center[2] - 0.035)
    assert top[0, 2] > 0.95 and abs(top[1, 2]) < 0.05 and top[2, 2] < -0.4


def test_chain_clearance_from_static_bodies(head_only):
    """Every chain capsule is >= 5 mm clear of the body part's static sphere / capsule colliders at rest."""
    ctx, _, rig, _ = head_only
    from mkmmd.model.parts.hair_rig import euler_for_axis as _  # noqa: F401  (axis convention used by the capsules)

    def segment(rb):
        r, h = rb.size[0], rb.size[1]
        rx, ry, rz = rb.rotation
        Rx = np.array([[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]])
        Ry = np.array([[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]])
        Rz = np.array([[np.cos(rz), -np.sin(rz), 0], [np.sin(rz), np.cos(rz), 0], [0, 0, 1]])
        ax = Rz @ Ry @ Rx @ np.array([0, 0, 1.0])
        c = np.asarray(rb.location)
        return c - ax * h / 2, c + ax * h / 2, r

    def seg_seg(a0, a1, b0, b1):
        u, v, w = a1 - a0, b1 - b0, a0 - b0
        a, b, c, d, e = u @ u, u @ v, v @ v, u @ w, v @ w
        den = a * c - b * b
        s = np.clip((b * e - c * d) / den, 0, 1) if den > 1e-12 else 0.0
        t = np.clip((b * s + e) / c, 0, 1) if c > 1e-12 else 0.0
        s = np.clip((b * t - d) / a, 0, 1) if a > 1e-12 else 0.0
        return float(np.linalg.norm(a0 + s * u - (b0 + t * v)))

    statics = [rb for rb in ctx.parts["body"].bodies if rb.shape in ("sphere", "capsule")]
    worst = 1.0
    for rb in rig.bodies:
        a0, a1, r = segment(rb)
        for st in statics:
            if st.shape == "sphere":
                b0 = b1 = np.asarray(st.location)
                rr = st.size[0]
            else:
                b0, b1, rr = segment(st)
            worst = min(worst, seg_seg(a0, a1, b0, b1) - r - rr)
    assert worst >= 0.005, f"closest chain capsule to a static collider: {worst * 1000:.1f} mm"


# ---------------------------------------------------------------- against the real head part (skipped without MK_TEST_RIN_MODEL)
@pytest.fixture(scope="module")
def real_head(tmp_path_factory):
    """The head hair built on the real head part of the character project (whatever face it currently has)."""
    if SPEC is None:
        pytest.skip(WHY)
    from mkmmd.model import spec as SP
    sp = SP.load(SPEC)
    tmp = tmp_path_factory.mktemp("hair_real")
    ctx = BD.BuildCtx(sp, tmp / "tex", seed=1)
    try:
        BD.run(sp, only="head", ctx=ctx)
    except BD.BuildError as e:
        pytest.skip(f"the real head part does not build here: {e}")
    ctx.part = "hair"
    head = ctx.parts["head"]
    fit = HeadFit(head.info, meshes=head.meshes)
    rig = Rig()
    piece = hair_head.build_head_hair(ctx, fit, {}, rig, hair_tex.palette(ctx.get("colors.hair")))
    return ctx, fit, rig, piece


def _tris(m):
    return np.array([[f[0], f[k], f[k + 1]] for f in m.faces for k in range(1, len(f) - 1)])


def test_real_head_scalp_ears_and_eyes(real_head):
    """On the real head: no gap between the hair and the scalp in the hair region, the human ears are hidden, the fringe stays
    >= 8 mm above the upper lids, every hair vertex clears the skin."""
    ctx, fit, rig, piece = real_head
    m = piece.meshes[0]
    tris = _tris(m)
    D = fit.hair_region_dirs(step_deg=7.0)
    t = raycast_first(fit.center, D, m.verts, tris)
    gap = t - fit.skull.radius(D)
    assert np.isfinite(t).all(), f"{int((~np.isfinite(t)).sum())} rays escape between the hair and the scalp"
    assert gap.min() > 0.001 and gap.max() < 0.06
    dv = m.verts - fit.center
    assert (np.linalg.norm(dv, axis=1) - fit.skull.radius(dv)).min() > 0.002
    pts = np.array([np.asarray(e[k], float) for e in fit.ears.values() for k in ("top", "lobe", "front", "back")])
    d = pts - fit.center
    te = raycast_first(fit.center, d / np.linalg.norm(d, axis=1, keepdims=True), m.verts, tris)
    assert (te > np.linalg.norm(d, axis=1) + 0.001).all()
    wb = np.sum([np.asarray(w) for b, w in m.weights.items() if b.startswith("前髪")], axis=0)
    lids = max(fit.lid_top(s)[2] for s in ("L", "R"))
    assert m.verts[wb > 0.5][:, 2].min() >= lids + 0.008 - 1e-6


def test_real_head_fringe_shadow_sits_on_the_face(real_head):
    """The forehead shadow strips lie on the head part's `face` mesh (every vertex bound to it, carrying its morph offsets),
    above the eyes, and within 3 mm of the skin shell."""
    ctx, fit, rig, piece = real_head
    s = next(x for x in piece.meshes if x.name == "hair_shadow")
    P, uv, mat, hit = fit.on_face(s.verts)
    assert hit is not None and hit[3].mean() > 0.97, "the strips do not lie on the face mesh"
    assert mat in {m.name for m in ctx.parts["head"].materials}
    assert uv.shape == (len(s.verts), 2) and np.isfinite(uv).all()
    assert s.morphs and all(np.asarray(v).shape == (len(s.verts), 3) for v in s.morphs.values())
    face_morphs = next(x for x in ctx.parts["head"].meshes if x.name == "face").morphs
    assert set(s.morphs) <= set(face_morphs)
    dd = s.verts - fit.center
    g = np.linalg.norm(dd, axis=1) - fit.skull.radius(dd)
    assert -0.0005 < g.min() and g.max() < 0.003, (g.min(), g.max())
    assert s.verts[:, 2].min() > max(fit.lid_top(sd)[2] for sd in ("L", "R")) + 0.004


def test_real_head_cap_boundary_is_smooth(real_head):
    """Between two cap columns (5 deg of azimuth) the lower boundary changes by at most 35 deg of polar angle: the chord of
    such a quad sags at most 0.11 m x (1 - cos 17.5 deg) = 5 mm, less than the cap's offset above the skin (>= 6 mm), so
    no cap quad cuts through the head."""
    _, fit, _, piece = real_head
    th = np.radians(np.arange(-180.0, 180.0, 5.0))
    ph = np.degrees(fit.cap_phi(th))
    assert np.abs(np.diff(np.concatenate([ph, ph[:1]]))).max() <= 35.0
    m = piece.meshes[0]
    d = m.verts - fit.center
    assert (np.linalg.norm(d, axis=1) - fit.skull.radius(d)).min() > 0.002          # nothing of the hair inside the skin


# ---------------------------------------------------------------- on the girl base (her mesh body)
def test_girl_head_hair_keeps_off_her_body(tmp_path):
    """The girl's back locks hang to the nape, where her body's neck widens into the shoulders sooner than the skull hull
    the hanging hair is kept off (the head's neck, straight on under it): every head-hair vertex the body's skin answers
    for (below the neck seam, or up to the gap above it beside the neck) is hair_head.BODY_GAP off that skin, outside."""
    from mkmmd.model import spec as SP
    from mkmmd.model.parts.outfit_fit import Skin
    spec = SP.load("base:girl")
    ctx = BD.BuildCtx(spec, tmp_path, seed=SP.model_cfg(spec)["seed"])
    BD.run(spec, only=["hair"], ctx=ctx)
    body, nt = ctx.parts["body"], ctx.parts["body"].info["neck_top"]
    X = next(m for m in ctx.parts["hair"].meshes if m.name == "hair_head").verts
    over = ((X[:, 0] - nt["center"][0]) / nt["rx"]) ** 2 + ((X[:, 1] - nt["center"][1]) / nt["ry"]) ** 2 <= 1.0
    near = (X[:, 2] < nt["z"]) | ((X[:, 2] < nt["z"] + hair_head.BODY_GAP) & ~over)
    assert near.any()                                                # the nape's locks reach down to the body
    d = Skin.from_meshes([m for m in body.meshes if len(m.verts)], "body").closest(X[near])[0]
    assert d.min() >= hair_head.BODY_GAP - 1e-4


# ---------------------------------------------------------------- the full part
def test_full_part_validity(built):
    ctx, part = built
    assert part.name == "hair"
    BD.check_refs([ctx.parts["body"], ctx.parts["head"], part])
    names = {b.name for b in part.bones}
    for m in part.meshes:
        assert set(m.weights) <= names | {b.name for b in ctx.parts["body"].bones}
    assert len(names) == len(part.bones)
    assert {j.name for j in part.joints}.__len__() == len(part.joints)
    assert part.info["hair_top_z"] > ctx.parts["head"].info["skull_top"][2]


def test_full_part_families_and_chain_roots(built):
    """Chains classify into the intended families; chain roots hang from non-dynamic bones; the twitch bones are the
    ear-family bones whose parent is not an ear bone."""
    ctx, part = built
    dyn = {rb.bone for rb in part.bodies if rb.mode in ("dynamic", "dynamic_bone")}
    parent = {b.name: b.parent for b in part.bones}
    expect = {"前髪": "bangs", "横髪": "side_hair", "後髪": "back_hair", "三つ編": "braid", "猫耳": "ears", "尻尾": "tail",
              "リボン": "ribbon"}
    roots = [b for b in dyn if parent[b] not in dyn]
    assert roots
    for r in roots:
        fam = families.classify(r)
        pre = next(k for k in expect if r.startswith(k))
        assert fam == expect[pre] or (pre == "リボン" and fam == "ribbon"), (r, fam)
    got = {families.classify(r) for r in roots}
    assert {"bangs", "side_hair", "back_hair", "braid", "ears", "tail"} <= got
    ear_bones = [b.name for b in part.bones if families.classify(b.name) == "ears"]
    twitch = [b for b in ear_bones if families.classify(parent[b]) != "ears"]
    assert len(twitch) == 2 and all(b not in dyn for b in twitch)


def test_full_part_weights_normalised(built):
    _, part = built
    for m in part.meshes:
        tot = sum(np.asarray(w) for w in m.weights.values())
        assert np.allclose(tot, 1.0, atol=1e-5), m.name
        stack = np.stack([np.asarray(w) for w in m.weights.values()])
        assert ((stack > 1e-6).sum(0) <= 4).all(), m.name


def test_full_part_chains_are_connected_and_jointed(built):
    ctx, part = built
    bones = {b.name: b for b in part.bones}
    bodies = {rb.name: rb for rb in part.bodies}
    for b in part.bones:
        if b.tail_bone:
            assert np.allclose(b.tail if b.tail is not None else bones[b.tail_bone].head, bones[b.tail_bone].head) \
                or b.tail is None
    static_names = {rb.name for p in ctx.parts.values() for rb in p.bodies if rb.mode == "static"} | \
        {rb.name for rb in part.bodies if rb.mode == "static"}
    for j in part.joints:
        assert j.a in bodies or j.a in static_names
        assert j.b in bodies
    dynamic = [rb for rb in part.bodies if rb.mode != "static"]
    assert {j.b for j in part.joints} >= {rb.name for rb in dynamic}      # every dynamic body hangs from a joint


def test_full_part_is_deterministic(built, tmp_path):
    _, part = built
    ctx2 = HF.make_ctx(tmp_path, {"hair": dict(SLICES)})
    part2 = hair.build(ctx2)
    assert [m.name for m in part.meshes] == [m.name for m in part2.meshes]
    for a, b in zip(part.meshes, part2.meshes):
        assert np.array_equal(a.verts, b.verts) and a.faces == b.faces
    assert [b.name for b in part.bones] == [b.name for b in part2.bones]


def test_tails_can_be_switched_off(tmp_path):
    ctx = HF.make_ctx(tmp_path, {"hair": {"tails": {"enabled": False}}})
    part = hair.build(ctx)
    assert not any(families.classify(b.name) == "tail" for b in part.bones)
    assert any(families.classify(b.name) == "braid" for b in part.bones)
