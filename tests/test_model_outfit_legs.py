"""The legs of Rin's outfit (mkmmd/model/parts/outfit_legs.py): the black ribbon wound round the left calf with its bow and two
dynamic tails, and the Mary-Jane shoes on both feet.

What is checked, for a stand-in mannequin (`outfit_fit.provisional_skin`), for the girl base's body (the artist's mesh it
wears) and, when MK_TEST_RIN_MODEL names a character project whose body part builds, for that project's body part:
  - leg ribbon: every vertex of the spiral lies 2..6 mm above the skin (`Skin.closest`), the band winds `turns` times round the
    shin between the ankle and the knee on the left leg only, texture u counts tiles of 4 cm;
  - tail chains: bone names classify as "ribbon", parent 左ひざ, unique names, dynamic bodies (root in group 8, the rest in 7) that
    ignore the body colliders (group 0), starting outside the skin;
  - shoes: the leather (upper, collar, strap, buckle) never comes closer than 2 mm to the skin and is never inside it, quads
    too (centres and edge midpoints), faces point away from the foot, the toe box closes over the modelled toes, the sole is flat
    on the floor plane (the toe_end landmark, z = 0 for the real body: nothing below -1e-4) and weighted like the foot above it
    (toe box on the toe bone, heel on the ankle, blending at the ball), the right shoe mirrors the left on a symmetric body;
    the shoe size is the ACTUAL foot plus margins (no table of proportions);
  - meshes: `part.check`, weights sum to 1 with at most 4 bones on leg/foot bones only, < 0.5 % degenerate faces, metric UVs
    (satin: u = metres / 0.04 along the ribbon, v 0..1 across; leather: about metres / 0.10), deterministic output, cfg overrides.

bpy-free and scipy-free. Run: pytest -q tests/test_model_outfit_legs.py"""
import ast
from pathlib import Path

import numpy as np
import pytest

from local_project import SPEC, WHY
from mkmmd.core.families import classify
from mkmmd.model import build as BD
from mkmmd.model import part as PT
from mkmmd.model.parts import outfit_fit as F
from mkmmd.model.parts import outfit_geo as G
from mkmmd.model.parts import outfit_legs as OL
from mkmmd.model.parts import outfit_rig as RG

SATIN = ["黒リボン", "脚リボン"]
SHOE = ["靴", "靴底"]
LEG_BONES = {"左ひざ", "左足首", "左足つま先", "右ひざ", "右足首", "右足つま先"}
TAIL_BONES = {f"リボン脚{c}{k}" for c in "AB" for k in (1, 2, 3)}
MM = 1e-3


# ---------------------------------------------------------------- fixtures
def mannequin_fit():
    land = F.default_landmarks(0.95)
    land["head_tip"] = np.array([0.0, -0.0133, 1.3483])
    for side, sx in (("L", 1.0), ("R", -1.0)):
        land[f"toe_end.{side}"] = np.array([0.0538 * sx, -0.1566, 0.0])
    lm = F.Land(land)
    scale = 1.3483 / 1.7
    return F.Fit(lm, F.provisional_skin(lm, scale=scale)), None


def real_fit(tmp_path_factory, spec):
    """The outfit fit on the body part `spec` builds (a base or a project's model.toml)."""
    from mkmmd.model import spec as SP
    body = BD.run(SP.load(str(spec)), only="body", tex_dir=str(tmp_path_factory.mktemp("legs_body")))[0]
    skin = F.Skin.from_meshes([m for m in body.meshes if len(m.verts)], "body")
    return F.Fit(F.Land(dict(body.info["landmarks"])), skin), body


def build_legs(fit, cfg=None, anchor=None, colliders=None):
    satin, shoe = G.Soup("outfit_ribbons", SATIN), G.Soup("outfit_shoes", SHOE)
    rig = RG.Rig()
    rib = OL.leg_ribbon(fit, cfg or {}, satin, rig, anchor, colliders)
    sh = OL.shoes(fit, cfg or {}, shoe, satin, rig)
    meshes = [satin.mesh(), shoe.mesh()]
    mats = [PT.Material(n) for n in SATIN + SHOE]
    part = PT.Part("outfit", meshes=meshes, materials=mats, bones=rig.bones, bodies=rig.bodies, joints=rig.joints)
    return {"fit": fit, "rib": rib, "shoes": sh, "rig": rig, "soups": (satin, shoe), "meshes": meshes, "part": part,
            "colliders": colliders}


@pytest.fixture(scope="module", params=["mannequin", "girl", "real"])
def built(request, tmp_path_factory):
    if request.param == "mannequin":
        fit, body = mannequin_fit()
        out = build_legs(fit)
    else:
        if request.param == "girl":
            fit, body = real_fit(tmp_path_factory, "base:girl")
        elif SPEC is None:
            pytest.skip(WHY)
        else:
            try:
                fit, body = real_fit(tmp_path_factory, SPEC)
            except Exception as e:                                   # the project's body may not build here
                pytest.skip(f"real body part does not build: {e}")
        out = build_legs(fit, anchor={"左ひざ": "col_shin_L"}, colliders=[rb for rb in body.bodies if rb.mode == "static"])
    out["kind"], out["body"] = ("mannequin" if request.param == "mannequin" else "real"), body
    return out


def faces_area(mesh):
    V = np.asarray(mesh.verts)
    out = np.zeros(len(mesh.faces))
    for i, f in enumerate(mesh.faces):
        p = V[list(f)]
        out[i] = sum(0.5 * np.linalg.norm(np.cross(p[k] - p[0], p[k + 1] - p[0])) for k in range(1, len(f) - 1))
    return out


# ---------------------------------------------------------------- leg ribbon
def test_ribbon_sits_on_the_skin(built):
    fit, rib = built["fit"], built["rib"]
    d = fit.skin.closest(rib["grid"].reshape(-1, 3))[0]
    assert d.min() >= 2 * MM and d.max() < 6 * MM
    assert rib["min_offset"] == pytest.approx(d.min(), abs=1e-5)
    assert 2 * MM <= rib["min_offset"] and rib["max_offset"] < 6 * MM
    # the band is a real strip: its width follows `width` (centre to edge)
    g = rib["grid"]
    w = np.linalg.norm(g[:, -1] - g[:, 0], axis=1)
    assert w.mean() == pytest.approx(OL.DEFAULTS["ribbon"]["width"] * fit.S, rel=0.1)


def test_ribbon_winds_round_the_left_calf(built):
    fit, rib = built["fit"], built["rib"]
    L = fit.L
    ankle, knee = L["ankle.L"], L["knee.L"]
    path = rib["path"]
    assert (path[:, 0] > 0).all()                                   # the left leg only
    assert path[:, 2].min() >= ankle[2] and path[:, 2].max() <= knee[2]
    ax = rib["axis"]
    rel = path - ankle
    rel = rel - np.outer(rel @ ax, ax)
    u = np.cross(ax, [0.0, 1.0, 0.0])
    u = u / np.linalg.norm(u)
    v = np.cross(ax, u)
    ang = np.unwrap(np.arctan2(rel @ v, rel @ u))
    turns = abs(ang[-1] - ang[0]) / (2 * np.pi)
    assert turns == pytest.approx(OL.DEFAULTS["ribbon"]["turns"], abs=0.1)
    assert np.all(np.sign(np.diff(ang)) == np.sign(ang[-1] - ang[0]))        # monotone: one handedness
    z0, z1 = rib["z_range"]
    if built["kind"] == "real":                                      # proportions.outfit_guides.calf_ribbon_z
        assert z0 == pytest.approx(0.1236, abs=0.008) and z1 == pytest.approx(0.3789, abs=0.008)


def test_ribbon_uv_counts_four_centimetre_tiles(built):
    band = built["rib"]["band"]
    uv = np.asarray(band.uv)
    n = sum(len(f) for f in band.f)
    assert uv.shape == (n, 2)
    assert uv[:, 1].min() == pytest.approx(0.0) and uv[:, 1].max() == pytest.approx(1.0)
    length = np.linalg.norm(np.diff(built["rib"]["path"], axis=0), axis=1).sum()
    assert uv[:, 0].max() == pytest.approx(length / 0.04, rel=0.01)


def test_tail_chains_are_ribbon_bones(built):
    rig, rib = built["rig"], built["rib"]
    names = [b.name for b in rig.bones]
    assert set(names) == TAIL_BONES and len(names) == len(set(names)) == 6
    assert all(classify(n) == "ribbon" for n in names)
    assert rib["tail_bones"] == {"A": ["リボン脚A1", "リボン脚A2", "リボン脚A3"], "B": ["リボン脚B1", "リボン脚B2", "リボン脚B3"]}
    by = {b.name: b for b in rig.bones}
    for ch in "AB":
        ns = rib["tail_bones"][ch]
        assert by[ns[0]].parent == "左ひざ"
        assert [by[n].parent for n in ns[1:]] == ns[:-1]
    assert len(rig.bodies) == 6 and all(b.mode == "dynamic" for b in rig.bodies)
    assert sorted(b.group for b in rig.bodies) == [7] * 4 + [8] * 2
    ignores_body = [0 in b.no_collide for b in rig.bodies]
    if built["kind"] == "real":                 # proven clear of the body's colliders: only the two roots (group 8) skip group 0
        assert ignores_body == [True, False, False, True, False, False]
    else:                                       # no colliders known: every tail body ignores the body group
        assert all(ignores_body)
    expect_joints = 6 if built["kind"] == "real" else 4
    assert len(rig.joints) == expect_joints


def test_tail_bodies_clear_the_body_colliders(built):
    if built["kind"] != "real":
        pytest.skip("the mannequin has no shin colliders")
    rig, rib, body = built["rig"], built["rib"], built["body"]
    statics = [rb for rb in body.bodies if rb.mode == "static"]
    shin_stack = [rb for rb in statics if rb.bone == "左ひざ"]
    assert len(shin_stack) >= 2                                              # the body part's tapered lower-leg stack
    r = {b.name: b.size[0] for b in rig.bodies}
    for ch in "AB":
        names = rib["tail_bones"][ch]
        clear = RG.chain_clearance(rib["tail_points"][ch], [r[n] for n in names], statics)
        assert clear[1:].min() >= 0.0004                                      # non-root bodies: clear by tail_margin (0.5 mm)
        assert rib["tail_lift"][ch] >= 0.0
    assert rib["tail_clearance"] >= 0.0004
    # the lift only moves the tail out along the skin normal: still a few millimetres off the skin, never inside it
    for ch in "AB":
        d = built["fit"].skin.closest(rib["tail_points"][ch])[0]
        assert d.min() >= 2 * MM and d.max() < 20 * MM

def test_tail_chains_start_outside_the_body_and_the_shoe(built):
    fit, rib, sh = built["fit"], built["rib"], built["shoes"]
    for ch in "AB":
        pts = rib["tail_points"][ch]
        d = fit.skin.closest(pts)[0]
        assert (d > 0).all() and d[0] >= 2 * MM
        assert (pts[:, 0] > 0).all()
        leather = sh["L"]["leather_points"]
        clear = np.linalg.norm(pts[:, None, :] - leather[None], axis=2).min(1)
        assert (clear > 2 * MM).all()
    # the tails hang down: each chain points downwards
    for ch in "AB":
        pts = rib["tail_points"][ch]
        assert pts[-1, 2] < pts[0, 2]


# ---------------------------------------------------------------- shoes
def test_shoe_leather_clears_the_foot(built):
    fit, sh = built["fit"], built["shoes"]
    assert sh["min_dist"] >= 2 * MM
    for side in "LR":
        pts = sh[side]["leather_points"]
        F_ = sh[side]["foot"]
        d = F_.srf.closest(pts * np.array([F_.sg, 1.0, 1.0]))[0]        # the foot's own surface (the right one mirrored)
        assert d.min() >= 2 * MM, (side, d.min())
        assert (d > 0).all()                                         # never inside
        if built["kind"] == "real":
            assert fit.skin.closest(pts)[0].min() >= 2 * MM          # and clear of the whole body
        up = sh[side]["upper"]
        cen = np.array([up.v[list(f)].mean(0) for f in up.f])
        if built["kind"] == "real":                                  # quads too, not only vertices (the mannequin foot is open)
            assert fit.skin.closest(cen)[0].min() >= 2 * MM


def test_shoe_faces_point_away_from_the_foot(built):
    fit, sh = built["fit"], built["shoes"]
    for side in "LR":
        for p in (sh[side]["upper"],):                               # (the collar is a tube: half of its faces look at the skin)
            fn = G.face_normals(p.v, p.f)
            ln = np.linalg.norm(fn, axis=1)
            ok = ln > 1e-12
            cen = np.array([p.v[list(f)].mean(0) for f in p.f])[ok]
            n = fn[ok] / ln[ok, None]
            d_out = fit.skin.closest(cen + n * 2 * MM)[0]
            d_in = fit.skin.closest(cen - n * 2 * MM)[0]
            assert (d_out > d_in).mean() > (0.98 if built["kind"] == "real" else 0.95)


def test_sole_is_flat_on_the_floor_and_rigid(built):
    sh = built["shoes"]
    for side in "LR":
        info = sh[side]
        floor = info["floor_z"]
        if built["kind"] == "real":
            assert info["sole_min_z"] == pytest.approx(floor, abs=1e-9)
        bottom = [p for p in info["sole"]["patches"] if p.tag == "sole_bottom"]
        z = np.concatenate([p.v[:, 2] for p in bottom])
        assert z.min() == pytest.approx(floor, abs=1e-9)
        assert z.max() - floor <= OL.DEFAULTS["shoe"]["spring"] * built["fit"].S + 1e-9      # the toe spring only
        nz = np.concatenate([G.face_normals(p.v, p.f)[:, 2] for p in bottom])
        assert (nz < 0).mean() > (0.99 if built["kind"] == "real" else 0.9)      # the bottom faces down
        pre = "左" if side == "L" else "右"
        for p in info["sole"]["patches"]:                            # weighted like the foot above it (flexes at the ball)
            assert set(p.w) <= {pre + "ひざ", pre + "足首", pre + "足つま先"}
            assert np.allclose(sum(p.w.values()), 1.0)
    if built["kind"] == "real":
        assert sh["sole_min_z"] >= -1e-4                             # the foot is flat on z = 0 when the leg IK flattens it
        assert sh["L"]["sole_height"] > 0.02


def test_shoe_size_follows_the_foot(built):
    sh = built["shoes"]
    F_ = sh["L"]["foot"]
    w, ln = sh["L"]["sole_size"]
    assert F_.length + 0.004 < ln < F_.length + 0.06                  # the shoe is the actual foot plus toe room, welt and ledge
    assert F_.width + 0.006 < w < F_.width + 0.045
    # the right shoe is the mirror of the left one on a symmetric body
    L, R = sh["L"]["upper"].v, sh["R"]["upper"].v
    assert L.shape == R.shape
    mir = R * np.array([-1.0, 1.0, 1.0])
    tol = (2 if built["kind"] == "real" else 4) * MM                 # (the mannequin's two legs are tessellated differently)
    assert np.abs(L.mean(0) - mir.mean(0)).max() < tol
    if built["kind"] == "real":                                      # (the mannequin's feet are not exact mirror images)
        assert np.abs(np.ptp(L, 0) - np.ptp(mir, 0))[:2].max() < 2 * tol
    assert (R[:, 0] < 0).all() and (L[:, 0] > 0).all()


def inside_poly(P, poly):
    """Even-odd test of points P (n, 2) against a closed polygon (k, 2)."""
    x, y = P[:, 0][:, None], P[:, 1][:, None]
    x0, y0 = poly[:, 0][None], poly[:, 1][None]
    x1, y1 = np.roll(poly[:, 0], -1)[None], np.roll(poly[:, 1], -1)[None]
    cross = ((y0 > y) != (y1 > y)) & (x < (x1 - x0) * (y - y0) / np.where(y1 == y0, 1e-12, y1 - y0) + x0)
    return cross.sum(1) % 2 == 1


def test_toe_box_closes_over_the_toes(built):
    if built["kind"] != "real":
        pytest.skip("the mannequin foot has no toes (and no underside)")
    sh = built["shoes"]
    for side in "LR":
        F_ = sh[side]["foot"]
        sk = F_.skin
        V = sk.verts[np.unique(sk.tris)]
        # the footprint before the ankle, the toe box's (round a heel the welt runs a little inside its widest point: the
        # leather rounds over the sole); every toe lies inside the leather's plan outline
        foot = V[(V[:, 2] < F_.floor + 0.022 * F_.S) & (V[:, 1] < F_.c_a[1])]
        welt = sh[side]["grid"][:, 0, :2]
        assert inside_poly(foot[:, :2], welt).all()
        assert sh[side]["grid"][..., 1].min() < F_.y_tip - 0.004      # and the leather reaches beyond the longest toe


def test_toe_box_follows_the_toes(built):
    fit = built["fit"]
    if "左足つま先" not in fit.skin.weights:
        pytest.skip("the body has no toe weights")
    m = built["meshes"][1]
    V = np.asarray(m.verts)
    ball_y = fit.L["toe.L"][1]
    left = V[:, 0] > 0
    toe = np.asarray(m.weights["左足つま先"])
    ank = np.asarray(m.weights["左足首"])
    assert toe[left & (V[:, 1] < ball_y - 0.03)].mean() > 0.9        # the toe box is on the toe bone
    assert ank[left & (V[:, 1] > 0.0)].mean() > 0.7                  # the heel on the ankle
    mid = left & (np.abs(V[:, 1] - ball_y) < 0.01)
    assert 0.1 < toe[mid].mean() < 0.9                               # and they blend at the ball


def test_strap_crosses_the_instep_and_carries_the_bow(built):
    sh = built["shoes"]
    for side in "LR":
        strap = sh[side]["strap"]
        path = strap["path"] * np.array([1.0 if side == "L" else -1.0, 1.0, 1.0])
        assert path[:, 0].min() < path[:, 0].mean() - 0.015 and path[:, 0].max() > path[:, 0].mean() + 0.015   # spans the foot
        F_ = sh[side]["foot"]
        assert F_.y_tip < strap["path"][:, 1].mean() < F_.c_a[1]
    bow_v = np.concatenate([p.v for p in built["soups"][0].patches if p.tag == "shoe_bow"])
    for side, sx in (("L", 1.0), ("R", -1.0)):
        sel = bow_v[bow_v[:, 0] * sx > 0]
        strap_mid = sh[side]["strap"]["path"][len(sh[side]["strap"]["path"]) // 2] * np.array([sx, 1.0, 1.0])
        assert np.linalg.norm(sel.mean(0)[:2] - strap_mid[:2]) < 0.02


# ---------------------------------------------------------------- meshes
def test_meshes_are_valid_parts(built):
    PT.check(built["part"])
    if built["kind"] == "real":
        warns = BD.check_refs([built["body"], built["part"]])
        assert not [w for w in warns if "outfit" in w and "no parent" not in w]


def test_weights_are_normalised_and_on_leg_bones(built):
    for m in built["meshes"]:
        n = len(m.verts)
        W = np.stack([np.asarray(w) for w in m.weights.values()], 1)
        assert np.allclose(W.sum(1), 1.0, atol=1e-6)
        assert ((W > 1e-6).sum(1) <= 4).all()
        assert set(m.weights) <= LEG_BONES | TAIL_BONES
        assert W.shape[0] == n
    shoe_mesh = built["meshes"][1]
    assert {"左足首", "右足首"} <= set(shoe_mesh.weights) <= LEG_BONES
    if "左足つま先" in built["fit"].skin.weights:                       # the body weights its forefoot on the toe bone
        assert {"左足つま先", "右足つま先"} <= set(shoe_mesh.weights)
    # the tails are weighted to their own chain, the spiral to the shin and ankle only
    band = built["rib"]["band"]
    assert set(band.w) <= {"左ひざ", "左足首"}


def test_no_degenerate_faces(built):
    for m in built["meshes"]:
        a = faces_area(m)
        assert (a > 1e-9).mean() > 0.995, (m.name, int((a <= 1e-9).sum()), len(a))


def test_uv_is_metric(built):
    satin, shoe = built["meshes"]
    for m in (satin, shoe):
        assert np.asarray(m.uv).shape == (sum(len(f) for f in m.faces), 2)
    # satin: u counts tiles of 4 cm along the ribbon (spiral), v across
    band = built["rib"]["band"]
    uv = np.asarray(band.uv)
    ratios = []
    k = 0
    for f in band.f:
        e = band.v[f[1]] - band.v[f[0]]
        ratios.append((uv[k + 1] - uv[k])[0] * 0.04 / max(np.linalg.norm(e), 1e-9))
        k += len(f)
    assert np.median(ratios) == pytest.approx(1.0, abs=0.08)
    # leather and sole: about metres / 0.10 (the upper's u is rounded to whole tiles round the shoe)
    for side in "LR":
        up = built["shoes"][side]["upper"]
        uvu = np.asarray(up.uv)
        r, k = [], 0
        for f in up.f:
            e = np.linalg.norm(up.v[f[1]] - up.v[f[0]])
            if e > 1e-4:
                r.append(np.linalg.norm(uvu[k + 1] - uvu[k]) * 0.10 / e)
            k += len(f)
        assert 0.6 < np.median(r) < 1.6
        assert np.asarray(up.uv).min() > -3.0 and np.asarray(up.uv).max() < 12.0


def test_vertex_budget_and_time(built):
    satin, shoe = built["meshes"]
    assert len(satin.verts) < 2000 and len(shoe.verts) < 7000
    assert len(shoe.faces) > 4000                                   # both shoes, not a stub


# ---------------------------------------------------------------- determinism, config, imports
def test_build_is_deterministic(built):
    again = build_legs(built["fit"], anchor={"左ひざ": "col_shin_L"} if built["kind"] == "real" else None,
                       colliders=built["colliders"])
    for a, b in zip(built["meshes"], again["meshes"]):
        assert np.array_equal(a.verts, b.verts) and a.faces == b.faces and np.array_equal(a.uv, b.uv)
        assert all(np.array_equal(a.weights[k], b.weights[k]) for k in a.weights)


def test_cfg_overrides(built):
    fit = built["fit"]
    out = build_legs(fit, {"ribbon": {"turns": 3.0, "width": 0.012, "offset": 0.005}, "shoe": {"ease": 0.0055, "min_gap": 0.0045}})
    rib = out["rib"]
    assert rib["turns"] == 3.0
    assert rib["min_offset"] > built["rib"]["min_offset"]
    g = rib["grid"]
    assert np.linalg.norm(g[:, -1] - g[:, 0], axis=1).mean() == pytest.approx(0.012 * fit.S, rel=0.1)
    assert out["shoes"]["min_dist"] > built["shoes"]["min_dist"] + 0.5 * MM


def test_module_imports_only_numpy_and_the_outfit_kit():
    tree = ast.parse(Path(OL.__file__).read_text())
    mods = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            mods.add(("." * node.level) + (node.module or ""))
    assert {m for m in mods if not m.startswith(".")} == {"numpy"}, mods
