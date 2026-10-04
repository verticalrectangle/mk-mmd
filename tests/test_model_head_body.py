"""The head part in body mode (`[head] source = "body"`, bpy-free): the skin cap that closes an imported face shell, the radial
map used to publish its surface, and the builder end to end on a synthetic take (never on a real model).

The synthetic head is a sphere-like shell: a polar grid of rings from a smooth outer boundary (high in front, low behind) down
to a neck opening, with two eye holes and a mouth hole, a tube of body skin under the neck sharing the ring vertices, eyeball
and brow pieces and three vertex morphs (one of them on the body skin only)."""
import numpy as np
import pytest
from PIL import Image

from mkmmd.model import build, part as P, pmx_io as X, pmx_take as T, spec
from mkmmd.model.parts import head_cap as CP, head_shell as SH

UNIT = 0.08
RADIUS = 0.09
CENTRE = np.array([0.0, 0.005, 1.255])                  # sphere centre in model space (the head bone sits at z = 1.2)
N_AZ, N_RING = 24, 9
PSI_NECK = np.radians(148.0)


def _dir(psi, theta):
    return np.array([np.sin(psi) * np.sin(theta), -np.sin(psi) * np.cos(theta), np.cos(psi)])


def _psi_loop(theta):
    return np.radians(45.0 + 40.0 * (1.0 - np.cos(theta)) / 2.0)


def _pmx(p):
    return (float(p[0]) / UNIT, float(p[2]) / UNIT, float(p[1]) / UNIT)


class Synth:
    def __init__(self, tmp):
        (tmp / "tex").mkdir()
        g = np.linspace(0, 1, 256)
        skin = np.stack([np.tile(235 + 20 * g, (256, 1)), np.tile(200 + 40 * g[:, None], (1, 256)), np.tile(190 + 30 * g, (256, 1))], -1).astype(np.uint8)
        Image.fromarray(skin).save(tmp / "tex" / "skin.png")
        eye = np.zeros((32, 32, 4), np.uint8)
        eye[..., :3] = (200, 30, 140)
        eye[..., 3] = 255
        eye[:8, :8, 3] = 0
        Image.fromarray(eye).save(tmp / "tex" / "eye.png")
        toon = np.zeros((32, 32, 3), np.uint8)
        toon[:16] = 255
        toon[16:] = (243, 211, 215)
        Image.fromarray(toon).save(tmp / "tex" / "toon.bmp")
        bones = ["全ての親", "首", "頭", "両目", "左目", "右目", "brow"]
        bi = {n: i for i, n in enumerate(bones)}
        V, B, UV = [], [], []

        def add(p, bone, uv=(0.5, 0.5)):
            V.append(np.asarray(p, float))
            B.append(bone)
            UV.append(uv)
            return len(V) - 1

        th = 2 * np.pi * np.arange(N_AZ) / N_AZ
        grid = np.zeros((N_RING + 1, N_AZ), int)
        for m in range(N_RING + 1):
            for j in range(N_AZ):
                psi = _psi_loop(th[j]) + (PSI_NECK - _psi_loop(th[j])) * m / N_RING
                p = CENTRE + RADIUS * _dir(psi, th[j])
                grid[m, j] = add(p, "首" if m == N_RING else "頭", (0.5 + 3.0 * p[0], 0.3 - 3.0 * (p[2] - CENTRE[2])))
        holes = {"eyeL": (np.radians(30), np.radians(82)), "eyeR": (np.radians(-30), np.radians(82)), "mouth": (0.0, np.radians(110))}
        self.hole_centres = {k: CENTRE + RADIUS * _dir(p0, t0) for k, (t0, p0) in holes.items()}
        tris = []
        for m in range(N_RING):
            for j in range(N_AZ):
                j1 = (j + 1) % N_AZ
                q = [grid[m, j], grid[m, j1], grid[m + 1, j1], grid[m + 1, j]]
                c = np.mean([V[i] for i in q], axis=0) - CENTRE
                c /= np.linalg.norm(c)
                psi_c, th_c = np.arccos(c[2]), np.arctan2(c[0], -c[1])
                gone = False
                for nm, (t0, p0) in holes.items():
                    dth = (th_c - t0 + np.pi) % (2 * np.pi) - np.pi
                    if np.hypot(dth * np.sin(p0), psi_c - p0) < np.radians(9.5 if nm != "mouth" else 11.0):
                        gone = True
                if not gone:
                    tris += [(q[0], q[1], q[2]), (q[0], q[2], q[3])]
        ccw = lambda tr: [t if np.cross(V[t[1]] - V[t[0]], V[t[2]] - V[t[0]]) @ (V[t[0]] - CENTRE) > 0 else (t[0], t[2], t[1]) for t in tr]
        skin_tris = ccw(tris)
        # body skin: a tube under the neck ring, sharing the ring vertices
        ring = grid[N_RING]
        lower = [add(V[i] - (0, 0, 0.04), "首", (0.2, 0.2)) for i in ring]
        body_tris = []
        for j in range(N_AZ):
            j1 = (j + 1) % N_AZ
            a, b, c, d = ring[j], ring[j1], lower[j1], lower[j]
            body_tris += [(a, b, c), (a, c, d)]

        def outward(t):                                                                 # CCW seen from outside the tube
            n = np.cross(V[t[1]] - V[t[0]], V[t[2]] - V[t[0]])
            mid = (V[t[0]] + V[t[1]] + V[t[2]]) / 3
            return t if n @ np.array([mid[0] - CENTRE[0], mid[1] - CENTRE[1], 0.0]) > 0 else (t[0], t[2], t[1])
        body_tris = [outward(t) for t in body_tris]
        eye_tris, self.eye_ids = [], {}
        for bone, nm in (("左目", "eyeL"), ("右目", "eyeR")):
            c = CENTRE + 0.8 * (self.hole_centres[nm] - CENTRE)
            ids = [add(c, bone)] + [add(c + 0.012 * np.array([np.cos(a), 0.0, np.sin(a)]), bone, (0.2 + 0.1 * k, 0.5)) for k, a in enumerate(np.pi / 2 * np.arange(4))]
            self.eye_ids[bone] = ids
            eye_tris += [(ids[0], ids[1 + k], ids[1 + (k + 1) % 4]) for k in range(4)]
        brow_tris, self.brow_ids = [], []
        for sg in (1.0, -1.0):
            base = CENTRE + RADIUS * _dir(np.radians(66), sg * np.radians(30))
            ids = [add(base + np.array([sg * dx, -0.002, dz]), "brow") for dx, dz in ((-0.012, 0.0), (0.0, 0.003), (0.012, 0.0), (-0.012, 0.002), (0.0, 0.005), (0.012, 0.002))]
            self.brow_ids += ids
            brow_tris += [(ids[0], ids[1], ids[4]), (ids[0], ids[4], ids[3]), (ids[1], ids[2], ids[5]), (ids[1], ids[5], ids[4])]
        self.V = np.array(V)
        self.skin_tris, self.body_tris = skin_tris, body_tris
        verts = [X.PmxVertex(_pmx(p), (0.0, 1.0, 0.0), (float(u[0]), float(u[1])), (bi[b],), (1.0,)) for p, b, u in zip(V, B, UV)]
        order = [skin_tris, body_tris, eye_tris, brow_tris]
        faces = [int(i) for tr in order for t in tr for i in t[::-1]]
        textures = ["tex\\skin.png", "tex\\eye.png", "tex\\toon.bmp"]
        mats = [X.PmxMaterial("skin", texture=0, toon=2, edge=True, index_count=3 * len(skin_tris)),
                X.PmxMaterial("bodyskin", texture=0, toon=2, index_count=3 * len(body_tris)),
                X.PmxMaterial("eyeball", texture=1, toon=2, index_count=3 * len(eye_tris)),
                X.PmxMaterial("brow", texture=0, index_count=3 * len(brow_tris))]
        pos = {"全ての親": (0, 0, 0), "首": (0.0, 0.0, 1.15), "頭": (0.0, 0.0, 1.2), "両目": (0.0, -0.12, 1.4), "左目": (0.03, -0.05, 1.255),
               "右目": (-0.03, -0.05, 1.255), "brow": (0.0, -0.1, 1.3)}
        par = {"全ての親": -1, "首": 0, "頭": 1, "両目": 2, "左目": 2, "右目": 2, "brow": 2}
        pb = [X.PmxBone(n, "", _pmx(np.array(pos[n])), par[n]) for n in bones]
        sk0 = int(grid[3, 1])
        vm = lambda name, panel, items: X.PmxMorph(name, "", panel, "vertex", [(i, _pmx(d)) for i, d in items])
        morphs = [vm("blink", 2, [(sk0, (0, -0.001, -0.004)), (self.eye_ids["左目"][0], (0, 0, -0.002))]),
                  vm("smile", 3, [(int(grid[6, 0]), (0.003, 0, 0.001))]),
                  vm("body only", 4, [(lower[0], (0.001, 0, 0))]),
                  vm("brows up", 1, [(self.brow_ids[1], (0, 0, 0.004))])]
        self.model = X.PmxModel(name="synthetic", vertices=verts, faces=faces, textures=textures, materials=mats, bones=pb, morphs=morphs)
        self.path = tmp / "head.pmx"
        X.write(self.model, str(self.path))
        self.tc = dict(materials=["skin", "eyeball", "brow"], skin="skin", brow_bone="brow", brow_material="brow", eyes=dict(iris="eyeball"),
                       recolor=[dict(texture="tex/eye.png", hue=[[0, 0], [200, 200], [290, 350], [330, 358]], protect=[0.05, 0.2])])


@pytest.fixture(scope="module")
def synth(tmp_path_factory):
    return Synth(tmp_path_factory.mktemp("src"))


# ---------------------------------------------------------------------------------------------------------- geometry
def _sphere_shell():
    th = 2 * np.pi * np.arange(N_AZ) / N_AZ
    V, F = [], []
    for m in range(N_RING + 1):
        for j in range(N_AZ):
            psi = _psi_loop(th[j]) + (PSI_NECK - _psi_loop(th[j])) * m / N_RING
            V.append(CENTRE + RADIUS * _dir(psi, th[j]))
    for m in range(N_RING):
        for j in range(N_AZ):
            a, b = m * N_AZ + j, m * N_AZ + (j + 1) % N_AZ
            F.append((a, b, b + N_AZ, a + N_AZ))
    V = np.array(V)
    out = [f if np.cross(V[f[1]] - V[f[0]], V[f[2]] - V[f[0]]) @ (V[f[0]] - CENTRE) > 0 else tuple(reversed(f)) for f in F]
    return V, out


def _closed_sphere(n_psi=24, n_th=48):
    V = [CENTRE + np.array([0, 0, RADIUS]), CENTRE - np.array([0, 0, RADIUS])]
    for k in range(1, n_psi):
        for j in range(n_th):
            V.append(CENTRE + RADIUS * _dir(np.pi * k / n_psi, 2 * np.pi * j / n_th))
    Tr = []
    for j in range(n_th):
        Tr.append((0, 2 + j, 2 + (j + 1) % n_th))
        Tr.append((1, 2 + (n_psi - 2) * n_th + (j + 1) % n_th, 2 + (n_psi - 2) * n_th + j))
    for k in range(n_psi - 2):
        for j in range(n_th):
            a, b = 2 + k * n_th + j, 2 + k * n_th + (j + 1) % n_th
            Tr += [(a, a + n_th, b), (b, a + n_th, b + n_th)]
    return np.array(V), Tr


def test_boundary_loops_find_the_outer_loop_and_the_neck_opening():
    V, F = _sphere_shell()
    loops = CP.boundary_loops(V, F)
    assert sorted(len(w) for w, o in loops) == [N_AZ, N_AZ]
    dup = {F[0][0]: len(V), F[0][1]: len(V) + 1}                                       # two vertices of one face duplicated (a map seam)
    V2 = np.vstack([V, V[F[0][0]], V[F[0][1]]])
    F2 = list(F)
    F2[0] = tuple(dup.get(v, v) for v in F2[0])
    assert sorted(len(w) for w, o in CP.boundary_loops(V2, F2)) == [N_AZ, N_AZ]       # coincident vertices are welded


def test_cap_starts_on_the_loop_stays_outward_and_closes_the_shell():
    V, F = _sphere_shell()
    loops = CP.boundary_loops(V, F)
    outer = max(loops, key=lambda l: float(V[l[1]][:, 2].max()))
    L = V[outer[1]]
    prior = lambda d: CENTRE + 0.088 * np.asarray(d, float)                           # a slightly smaller sphere
    new, faces = CP.build_cap(L, CENTRE, prior, rings=7)
    n = len(L)
    assert len(new) == 7 * n + 1
    allv = np.vstack([L, new])
    r_apex = np.linalg.norm(new[-1] - CENTRE)
    assert np.allclose(new[-1][:2], CENTRE[:2], atol=1e-9) and abs(r_apex - 0.088) < 1e-6        # the apex sits on the pole, on the prior
    r_loop = np.linalg.norm(L - CENTRE, axis=1)
    assert np.all(np.abs(np.linalg.norm(new[:n] - CENTRE, axis=1) - r_loop) < 0.0016)            # the first ring is still the loop's own radius
    capF = CP.orient_outward(allv, [tuple(f) for f in faces], CENTRE)
    for f in capF:
        nrm = np.cross(allv[f[1]] - allv[f[0]], allv[f[2]] - allv[f[0]])
        assert nrm @ (allv[list(f)].mean(0) - CENTRE) >= 0
    glob = np.concatenate([outer[1], len(V) + np.arange(len(new))])
    left = CP.boundary_loops(np.vstack([V, new]), F + [tuple(int(glob[i]) for i in f) for f in capF])
    assert [len(w) for w, o in left] == [N_AZ]                                         # only the neck opening is left


def test_hole_fan_closes_a_hole():
    a = 2 * np.pi * np.arange(12) / 12
    ring = CENTRE + RADIUS * np.stack([0.2 * np.cos(a), -np.ones(12), 0.2 * np.sin(a)], -1) / 1.02
    apex, faces = CP.hole_fan(ring, CENTRE)
    assert len(faces) == 12 and abs(np.linalg.norm(apex - CENTRE) - np.linalg.norm(ring - CENTRE, axis=1).mean()) < 1e-9


def test_radial_shape_reproduces_a_sphere():
    sph = SH.RadialShape(CENTRE, np.full((91, 180), 0.09))
    assert sph.phi(CENTRE + np.array([0.0, 0.0, 0.05])) < 0 < sph.phi(CENTRE + np.array([0.0, 0.0, 0.12]))
    p = sph.surface(np.array([[0.3, -0.5, 0.8]]))[0]
    assert np.linalg.norm(p - CENTRE) == pytest.approx(0.09, abs=1e-9)
    assert np.allclose(sph.normal(CENTRE + 0.09 * np.array([0.6, 0.0, 0.8])), [0.6, 0.0, 0.8], atol=2e-2)
    V, Tr = _closed_sphere()
    sh = SH.RadialShape.from_shell(CENTRE, V, Tr, res_deg=6.0)
    assert np.allclose(np.linalg.norm(sh.surface(np.array([[0.2, -0.9, 0.1], [0.0, 0.3, 0.95]])) - CENTRE, axis=1), 0.09, atol=1.5e-3)
    hw, yf, yb = sh.prof.at(CENTRE[2])
    assert hw == pytest.approx(0.09, abs=1.5e-3) and yb - yf == pytest.approx(0.18, abs=2e-3)
    assert sh.ztop == pytest.approx(CENTRE[2] + 0.09, abs=3e-3)


def test_radial_shape_reports_an_open_shell():
    V, F = _sphere_shell()
    Tr = [(f[0], f[k], f[k + 1]) for f in F for k in range(1, len(f) - 1)]
    with pytest.raises(ValueError, match="open"):
        SH.RadialShape.from_shell(CENTRE, V, Tr, res_deg=6.0)


# ----------------------------------------------------------------------------------------------------- the whole mode
@pytest.fixture(scope="module")
def head_part(synth, tmp_path_factory):
    from mkmmd.model.parts import head as H
    d = tmp_path_factory.mktemp("tex")
    sp = spec.from_dict({"model": {"name": "t", "parts": ["head"]}, "proportions": {"head": {}, "face": {}, "hair_guides": {}},
                         "head": {"source": "body", "take": synth.tc}})
    ctx = build.BuildCtx(sp, d, seed=1)
    ctx.part = "body"
    tk = T.take(synth.path, ["skin", "bodyskin", "eyeball", "brow"], scale=1.0)
    ctx.parts["body"] = P.Part("body", bones=tk.bones, info=dict(take=tk))
    ctx.land.update(head=np.array(tk.bone("頭").head))
    ctx.part = "head"
    part = H.build(ctx)
    P.check(part)
    part.ctx = ctx
    part.take = tk
    return part


def _mesh(part, name):
    return next(m for m in part.meshes if m.name == name)


def test_body_mode_needs_the_published_take(synth, tmp_path):
    from mkmmd.model.parts import head as H
    sp = spec.from_dict({"model": {"name": "t", "parts": ["head"]}, "head": {"source": "body", "take": synth.tc}})
    ctx = build.BuildCtx(sp, tmp_path, seed=1)
    ctx.part = "body"
    ctx.parts["body"] = P.Part("body", info={})
    ctx.part = "head"
    with pytest.raises(ValueError, match="publish"):
        H.build(ctx)


def test_builds_a_valid_part_with_the_face_meshes_and_no_bones(head_part):
    names = [m.name for m in head_part.meshes]
    assert names[0] == "face" and {"eyeball", "brow", "ears"} <= set(names) and "bodyskin" not in names
    assert [m.name for m in head_part.materials] == ["skin", "eyeball", "brow"]
    assert head_part.bones == []                                                       # the bones are the body part's
    assert not head_part.ctx.warnings


def test_every_vertex_morph_is_declared_once_by_the_head_part(head_part):
    assert sorted(m.name for m in head_part.morphs) == ["blink", "body only", "brows up", "smile"]
    assert {m.name: m.panel for m in head_part.morphs} == {"blink": "eye", "smile": "mouth", "body only": "other", "brows up": "brow"}
    face = _mesh(head_part, "face")
    assert set(face.morphs) == {"blink", "smile"} and all(v.shape == (len(face.verts), 3) for v in face.morphs.values())
    assert set(_mesh(head_part, "brow").morphs) == {"brows up"}
    assert "body only" not in set().union(*[set(m.morphs) for m in head_part.meshes])   # that one lives on the body part's mesh


def test_the_skin_mesh_keeps_its_neck_ring_open_and_is_closed_elsewhere(head_part, synth):
    face = _mesh(head_part, "face")
    loops = CP.boundary_loops(face.verts, face.faces)
    sizes = sorted(len(w) for w, o in loops)
    assert len(sizes) == 4 and sizes[-1] == N_AZ                                       # the neck ring (shared with the body skin) + 3 openings
    n_skin = len(head_part.take.piece(["skin"]).verts)
    assert len(face.verts) > n_skin                                                    # the cap's vertices follow the skin's
    assert np.allclose(face.verts[:n_skin], head_part.take.piece(["skin"]).verts)       # the imported skin is untouched


def test_the_cap_carries_weights_uv_and_normals(head_part):
    face = _mesh(head_part, "face")
    n_skin = len(head_part.take.piece(["skin"]).verts)
    assert np.allclose(sum(face.weights.values()), 1.0)
    assert np.allclose(face.weights["頭"][n_skin:], 1.0)
    assert np.linalg.norm(face.normals, axis=1).min() > 0.99
    assert face.uv.shape == (sum(len(f) for f in face.faces), 2) and face.uv.min() >= 0.0 and face.uv.max() <= 1.0
    assert face.face_mat.max() == 0 and face.mats == ["skin"]


def test_textures_are_edited_copies_and_the_patch_is_painted_into_the_skin_one(head_part, synth):
    tex = head_part.ctx.tex_dir
    skin = next(m for m in head_part.materials if m.name == "skin")
    img = np.asarray(Image.open(tex / skin.texture).convert("RGB")).astype(int)
    orig = np.asarray(Image.open(synth.path.parent / "tex" / "skin.png").convert("RGB")).astype(int)
    assert img.shape == orig.shape and (np.abs(img - orig).sum(-1) > 0).any()           # patches painted into free texels
    changed = np.abs(img - orig).sum(-1) > 0
    ys, xs = np.nonzero(changed)
    assert ys.min() > 0.5 * img.shape[0]                                               # in the unused lower half (the bottom right)
    eye = next(m for m in head_part.materials if m.name == "eyeball")
    ey = np.asarray(Image.open(tex / eye.texture).convert("RGBA")).astype(int)
    assert ey[20, 20][0] > ey[20, 20][2] and ey[20, 20][1] < 60 and eye.alpha_blend      # the magenta became red


def test_the_published_info_is_measured_on_the_imported_surface(head_part, synth):
    i = head_part.info
    for k in ("head_center", "skin", "hairline", "hairline_side", "nape", "ears", "eyes", "brows", "neck_ring", "neck_center", "face_outline",
              "cat_ear_anchors", "landmarks", "skull_top", "chin", "brow_front", "lid_margin", "mouth"):
        assert k in i, k
    e = i["eyes"]["L"]
    tk = head_part.take
    assert np.allclose(e["center"], tk.bone("左目").head) and e["radius"] == pytest.approx(np.linalg.norm(e["pupil"] - e["center"]))
    assert e["lid_top"][2] > e["pupil"][2] and e["opening"].shape[1] == 2
    assert i["landmarks"]["eye.L"][0] > 0 > i["landmarks"]["eye.R"][0]
    assert len(i["brows"]["L"]) >= 3 and i["brows"]["L"][:, 0].min() > 0
    sk = i["skin"]
    assert len(sk["faces"]) > 100
    pv = np.asarray(i["frame"]["pivot"])
    sh = SH.RadialShape.from_shell(np.asarray(i["head_center"]) - pv, np.asarray(sk["verts"]) - pv, [tuple(t) for t in sk["faces"]], res_deg=4.0, phi_open_deg=147.0)
    for k in ("L", "R"):
        assert abs(sh.phi(np.asarray(i["cat_ear_anchors"][k]["origin"]) - pv)) < 0.004   # the anchors lie on the closed shell
    assert abs(i["skull_top"][2] - np.asarray(sk["verts"])[:, 2].max()) < 0.01
    assert i["neck_ring"].shape[1] == 3


def test_a_mouth_less_or_eye_less_skin_still_builds(synth, tmp_path):
    """No eye data is published when the iris material is not named (the hair part falls back to its own estimate)."""
    from mkmmd.model.parts import head as H
    tc = dict(synth.tc)
    tc["eyes"] = {}
    sp = spec.from_dict({"model": {"name": "t", "parts": ["head"]}, "proportions": {"head": {}, "face": {}, "hair_guides": {}}, "head": {"source": "body", "take": tc}})
    ctx = build.BuildCtx(sp, tmp_path, seed=1)
    ctx.part = "body"
    tk = T.take(synth.path, ["skin", "bodyskin", "eyeball", "brow"])
    ctx.parts["body"] = P.Part("body", bones=tk.bones, info=dict(take=tk))
    ctx.land.update(head=np.array(tk.bone("頭").head))
    ctx.part = "head"
    part = H.build(ctx)
    assert part.info["eyes"] == {}


# ------------------------------------------------------------------------------------------------------ the [head.edit] edits
def _build(synth, tmp_path, edit):
    from mkmmd.model.parts import head as H
    tc = dict(synth.tc)
    sp = spec.from_dict({"model": {"name": "t", "parts": ["head"]}, "proportions": {"head": {}, "face": {"eye_outer_tilt_deg": 5.0}, "hair_guides": {}},
                         "head": {"source": "body", "take": tc, "edit": edit}})
    ctx = build.BuildCtx(sp, tmp_path, seed=1)
    ctx.part = "body"
    tk = T.take(synth.path, ["skin", "bodyskin", "eyeball", "brow"])
    ctx.parts["body"] = P.Part("body", bones=tk.bones, info=dict(take=tk))
    ctx.land.update(head=np.array(tk.bone("頭").head))
    ctx.part = "head"
    part = H.build(ctx)
    P.check(part)
    return part, tk


def tk_open(part, side):
    """The eye opening polygon (x, z) as published: it is the EDITED skin's hole, so the tilt turned it with the lids."""
    return part.info["eyes"][side]["opening"]


def test_edits_off_leave_the_imported_skin_untouched(synth, tmp_path):
    part, tk = _build(synth, tmp_path, {"enabled": False, "eyes": {"enabled": True}, "jaw": {"enabled": True}})
    face = _mesh(part, "face")
    n = len(tk.piece(["skin"]).verts)
    assert np.array_equal(face.verts[:n], tk.piece(["skin"]).verts)


def test_the_eye_tilt_turns_the_skin_and_its_morph_offsets_about_each_eye_and_leaves_the_iris_alone(synth, tmp_path):
    part, tk = _build(synth, tmp_path, {"eyes": {"tilt_deg": 5.0}})                       # the default tilt would come from the spec: 5 deg here
    face, plain = _mesh(part, "face"), tk.piece(["skin"])
    n = len(plain.verts)
    d = face.verts[:n] - plain.verts
    moved = np.abs(d).max(1) > 1e-9
    assert moved.sum() > 20 and np.abs(d[moved]).max() < 0.02
    assert np.array_equal(_mesh(part, "eyeball").verts, tk.piece(["eyeball"]).verts)      # not a listed piece: round and untilted
    e = part.info["eyes"]["L"]
    assert np.allclose(e["center"], tk.bone("左目").head)
    # the left eye turns counter-clockwise seen from the front, the right one clockwise (the outer corners go up)
    for side, sg in (("L", 1.0), ("R", -1.0)):
        c = np.array([np.asarray(tk_open(part, side))[:, 0].mean(), 0.0, np.asarray(tk_open(part, side))[:, 1].mean()])   # the opening's centre
        rel0 = plain.verts - c
        sel = moved & (sg * plain.verts[:, 0] > 0)
        a0 = np.arctan2(rel0[sel, 2], sg * rel0[sel, 0])
        rel1 = face.verts[:n] - c
        a1 = np.arctan2(rel1[sel, 2], sg * rel1[sel, 0])
        turn = np.angle(np.exp(1j * (a1 - a0)))                                            # wrapped to (-pi, pi]
        assert np.all(turn > -1e-6) and turn.max() < np.radians(5.0) + 1e-6 and turn.max() > np.radians(4.0)
    # the morph offsets of the moved vertices were turned with them: their lengths are unchanged
    for name, off in plain.morphs.items():
        got = face.morphs[name][:n]
        assert np.allclose(np.linalg.norm(got, axis=1), np.linalg.norm(off, axis=1), atol=1e-9)


def test_the_jaw_edit_pushes_the_lower_face_out_by_at_most_the_amount(synth, tmp_path):
    part, tk = _build(synth, tmp_path, {"jaw": {"amount": 0.002, "strip": [0.1, 0.3], "below_eye": 0.0, "ramp_top": 0.01, "neck_gap": 0.0, "ramp_neck": 0.01}})
    face, plain = _mesh(part, "face"), tk.piece(["skin"])
    n = len(plain.verts)
    d = np.linalg.norm(face.verts[:n] - plain.verts, axis=1)
    assert d.max() <= 0.002 + 1e-9 and (d > 1e-5).sum() > 10
    assert np.array_equal(face.morphs["blink"][:n], plain.morphs["blink"])                  # the offsets stay as they are
