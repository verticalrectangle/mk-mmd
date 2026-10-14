"""Garments cut from the body's own skin (mkmmd.model.parts.outfit_shell) and the parts made of them: sweater, pants,
socks, sneakers and choker, on the girl base (her breasts grown as [body] bust lets a character's be). bpy-free."""
import numpy as np
import pytest

from mkmmd import pixmap as PX
from mkmmd.model import build as BD
from mkmmd.model import spec as SP
from mkmmd.model.parts import outfit as OU
from mkmmd.model.parts import outfit_fit as OF
from mkmmd.model.parts import outfit_ink as INK
from mkmmd.model.parts import outfit_shell as SH
from mkmmd.model.parts import sneakers as SN
from mkmmd.model.parts import sweater as SW

WEAR = 'model.parts=["body", "head", "sweater", "pants", "socks", "sneakers", "choker"]'


@pytest.fixture(scope="module")
def dressed(tmp_path_factory):
    """The girl base wearing every garment part: (ctx, {part name: Part}, the body's Fit)."""
    spec = SP.load("base:girl", [WEAR, "body.bust=0.02"])
    ctx = BD.BuildCtx(spec, tmp_path_factory.mktemp("wear"), seed=1)
    parts = {p.name: p for p in BD.run(spec, ctx=ctx)}
    ctx.part = "test"
    return ctx, parts, OU.make_fit(ctx)


@pytest.fixture(scope="module")
def webbed(tmp_path_factory):
    """The girl base in the sweater with its web under each arm switched on (it is off by default): (ctx, parts, Fit)."""
    spec = SP.load("base:girl", ['model.parts=["body", "sweater"]', "body.bust=0.02", "sweater.web.side=0.075"])
    ctx = BD.BuildCtx(spec, tmp_path_factory.mktemp("webbed"), seed=1)
    parts = {p.name: p for p in BD.run(spec, ctx=ctx)}
    ctx.part = "test"
    return ctx, parts, OU.make_fit(ctx)


def _grid(n=6):
    """A unit square in the z = 0 plane, n x n quads split into triangles, facing +z; weights a = x, b = 1 - x."""
    t = np.linspace(0.0, 1.0, n + 1)
    X, Y = np.meshgrid(t, t, indexing="ij")
    V = np.stack([X.ravel(), Y.ravel(), np.zeros(X.size)], 1)
    idx = np.arange(V.shape[0]).reshape(n + 1, n + 1)
    tris = []
    for i in range(n):
        for j in range(n):
            a, b, c, d = idx[i, j], idx[i + 1, j], idx[i + 1, j + 1], idx[i, j + 1]
            tris += [(a, b, c), (a, c, d)]
    W = np.stack([V[:, 0], 1.0 - V[:, 0]], 1)
    return SH.Shell(V, np.array(tris), np.tile([0.0, 0.0, 1.0], (len(V), 1)), ["a", "b"], W)


def _area_normals(sh):
    a, b, c = (sh.verts[sh.tris[:, k]] for k in range(3))
    n = np.cross(b - a, c - a)
    return 0.5 * np.linalg.norm(n, axis=1), n


def test_a_cut_follows_the_level_line_and_keeps_what_the_skin_carries():
    """Clipping keeps exactly the part where the field is >= 0: the cut runs along the level line (new vertices on it,
    interpolated), the kept area is exact, every triangle keeps its winding and the weights stay linear across the cut."""
    sh = _grid()
    f = sh.verts[:, 0] + 0.5 * sh.verts[:, 1] - 0.62               # a slanted line through the cells
    out = sh.clip(f)
    g = out.verts[:, 0] + 0.5 * out.verts[:, 1] - 0.62
    assert g.min() > -1e-12
    on = np.abs(g) < 1e-12
    assert on.sum() >= 7                                           # the cut made vertices on the line
    area, n = _area_normals(out)
    # the region x + y / 2 >= 0.62 of the unit square: 1 - the trapezoid below the line
    assert area.sum() == pytest.approx(1.0 - (0.62 - 0.25), abs=1e-12)
    assert (n[:, 2] > 0).all()
    assert np.allclose(out.W[:, 0], out.verts[:, 0]) and np.allclose(out.W.sum(1), 1.0)


def test_a_band_is_cut_on_both_sides():
    sh = _grid()
    out = sh.band(sh.verts[:, 0], 0.23, 0.71)
    assert out.verts[:, 0].min() == pytest.approx(0.23) and out.verts[:, 0].max() == pytest.approx(0.71)
    assert _area_normals(out)[0].sum() == pytest.approx(0.48)
    assert len(out.loops()) == 1                                   # a strip of the square: one boundary loop


def test_a_rim_closes_an_edge_back_towards_the_skin():
    """A rim adds a strip under the chosen boundary, `rim_depth` along the skin's normal below it, wound like the shell."""
    sh = _grid(4).band(_grid(4).verts[:, 0], 0.0, 0.5)
    loop = sh.loops()[0]
    P = sh.verts + sh.normals * 0.01
    p = sh.patch(P, np.zeros(len(P)), np.zeros(len(P)), rims=[loop], rim_depth=0.004)
    inner = p.v[len(np.unique(sh.tris)):]
    assert len(inner) == len(loop)
    assert np.allclose(inner[:, 2], 0.006)
    assert len(p.f) == len(sh.tris) + len(loop)


def test_a_finer_shell_keeps_the_surface_its_winding_and_what_the_skin_carries():
    """Subdividing splits every triangle in four at the middles of its edges: the same surface, every triangle wound as
    before, one new vertex per edge (neighbours share it: no cracks), weights and fields interpolated."""
    sh = _grid(3)
    sh.fields["x"] = sh.verts[:, 0].copy()
    out = sh.subdivide()
    assert len(out.tris) == 4 * len(sh.tris) and len(out.verts) == len(sh.verts) + len(sh.edges())
    area, n = _area_normals(out)
    assert area.sum() == pytest.approx(1.0) and (n[:, 2] > 0).all()
    assert np.allclose(out.W[:, 0], out.verts[:, 0]) and np.allclose(out.fields["x"], out.verts[:, 0])
    assert len(out.loops()) == 1 and len(out.loops()[0]) == 2 * len(sh.loops()[0])


def test_loose_cloth_that_starts_above_its_skin_keeps_that_skins_radius():
    """outfit_shell.drape: where the skin under the cloth starts lower in front than at the back (a knit hanging from a
    band that dips in front), the cloth's front starts at the front skin's own radius: it neither falls onto the axis
    nor takes the back's radius (which would push the front out like a barrel)."""
    a = (np.arange(144) + 0.5) / 144
    A, Z = np.meshgrid(a, np.linspace(0.0, 0.3, 31))
    back = np.abs(A - 0.5) < 0.25
    keep = back | (Z >= 0.06)                                              # the front's skin only from 6 cm down
    r = np.where(back, 0.15, 0.10)
    R = SH.drape(Z[keep], A[keep], r[keep], np.array([0.0, 0.03, 0.2]), np.zeros(3), taper=0.0)
    assert np.allclose(R, 0.10, atol=0.004)


def _tube(r=0.1, h=0.1, n=40, rows=8, jitter=0.0):
    """A vertical tube (radius r, z 0..h), n vertices round each of rows + 1 rings, wound outwards; fields u (the angle,
    a turn = 1) and z; each vertex `jitter` of the spacing off its ring position round the tube."""
    rng = np.random.default_rng(0)
    a = (np.arange(n)[None, :] + jitter * rng.uniform(-0.5, 0.5, (rows + 1, n))) / n
    z = np.repeat(np.linspace(0.0, h, rows + 1), n)
    V = np.stack([r * np.sin(2 * np.pi * a.ravel()), -r * np.cos(2 * np.pi * a.ravel()), z], 1)
    tris = []
    for i in range(rows):
        for j in range(n):
            p, q = i * n + j, i * n + (j + 1) % n
            tris += [(p, q, q + n), (p, q + n, p + n)]
    N = V * [1.0, 1.0, 0.0] / r
    return SH.Shell(V, np.array(tris), N, ["a"], np.ones((len(V), 1)), {"u": a.ravel() % 1.0, "z": z})


@pytest.mark.parametrize("jitter", [0.0, 0.6])
def test_rib_lines_cross_a_ribbed_piece_every_spacing(jitter):
    """outfit_ink.ribs draws a line across a ribbed piece (a tube here, across = its height) every `spacing` round it: as
    many as fit, evenly, each straight across nearly the whole depth on the outside, wherever the mesh's vertices lie
    (a line placed at a vertex's own value must still cross the piece, not collapse onto the vertex)."""
    sh = _tube(jitter=jitter)
    lines = INK.ribs(sh, sh.verts, "u", 0.02, "z", 0.0, 0.1, np.random.default_rng(3), broken=0.0)
    assert abs(len(lines) - round(2 * np.pi * 0.1 / 0.02)) <= 1
    ang = []
    for pts, nn, _, fl in lines:
        t = np.arctan2(pts[:, 0], -pts[:, 1])
        assert np.ptp(fl["z"]) > 0.7 * 0.1 and np.ptp(np.unwrap(t)) < 0.15          # across, ends at the last crossing
        assert (np.einsum("ij,ij->i", nn, pts * [1.0, 1.0, 0.0]) > 0).all()
        ang.append(np.mean(np.unwrap(t)) % (2 * np.pi))
    gaps = np.diff(np.sort(ang), append=np.min(ang) + 2 * np.pi) * 0.1
    assert 0.4 * 0.02 < gaps.min() and gaps.max() < 1.6 * 0.02


@pytest.mark.parametrize("name, gap", [("sweater", 0.0020), ("pants", 0.0002)])
def test_tight_cloth_stays_off_the_skin_between_its_vertices_too(dressed, name, gap):
    """A smoothed shell must keep its triangles, not only its vertices, off the skin: a bump of the skin between the
    vertices (a knee cap, a hip bone) shows through as skin-coloured spots. (The pants come closest in the crotch, where
    the nearest skin flips between the thighs; their rolled waist edge reaches towards the skin on purpose.)"""
    parts = dressed[1]
    assert parts[name].info["min_gap"] > gap


def test_the_sweater_leaves_the_shoulders_bare_and_ends_at_the_wrists(dressed):
    _, parts, fit = dressed
    V = np.asarray(parts["sweater"].meshes[0].verts)
    for side, sg in (("L", 1.0), ("R", -1.0)):
        sh, arm, elbow, wrist = (np.asarray(fit.L[f"{n}.{side}"], float) for n in ("shoulder", "arm", "elbow", "wrist"))
        # the top of the shoulder between the neck and the arm joint is bare
        near = (np.abs(V[:, 0] - 0.5 * (sh[0] + arm[0])) < 0.02) & (np.abs(V[:, 1] - sh[1]) < 0.04)
        assert not (near & (V[:, 2] > arm[2] - 0.01)).any()
        # nothing on the hand: past the wrist along the forearm only the cuff's few millimetres
        ax = (wrist - elbow) / np.linalg.norm(wrist - elbow)
        assert ((V[sg * V[:, 0] > 0.15] - wrist) @ ax).max() < 0.008 * fit.S
    w = parts["sweater"].meshes[0].weights
    assert any(b.endswith("腕") for b in w) and "上半身2" in w


def test_with_her_arms_down_the_band_is_one_level_ring(dressed):
    """The band wraps the top of her body as a real off-shoulder sweater's folded collar does: with her arms hanging down
    it runs round her chest, the outside of each upper arm and her back at one height (within 1 cm), so it never stops
    at a shoulder or rises over one, and between her chest and her arms no knit stands above the band's top around it
    (within 2.5 cm; the knit under each arm rose into a fin there while the band's line lay below the armpit). The arms
    are hung as the skin's weights would carry them: what the arm bones carry turns about each shoulder joint until the
    upper arm points straight down."""
    _, parts, fit = dressed
    m = parts["sweater"].meshes[0]
    V = np.asarray(m.verts, float)
    arm_w = sum(np.asarray(w, float) for b, w in m.weights.items() if OF.is_arm(b))
    P = V.copy()
    for side, sg in (("L", 1.0), ("R", -1.0)):
        j, e = (np.asarray(fit.L[f"{n}.{side}"], float) for n in ("arm", "elbow"))
        phi = -sg * np.arctan2(sg * (e[0] - j[0]), j[2] - e[2])             # brings the elbow under the joint
        on = sg * V[:, 0] > 0.0
        d = V[on] - j
        hung = np.stack([j[0] + d[:, 0] * np.cos(phi) - d[:, 2] * np.sin(phi), V[on, 1],
                         j[2] + d[:, 0] * np.sin(phi) + d[:, 2] * np.cos(phi)], 1)
        P[on] += arm_w[on, None] * (hung - V[on])
    rib = m.mats.index(SW.MAT_RIB)
    R = P[np.unique([i for f, k in zip(m.faces, m.face_mat) if k == rib for i in f])]
    R = R[R[:, 2] > float(fit.L["bust_front"][2]) - 0.05]                    # the band, not the waistband or cuffs
    tops = [R[(np.abs(R[:, 0]) < 0.02) & (R[:, 1] < 0.0), 2].max(), R[(np.abs(R[:, 0]) < 0.02) & (R[:, 1] > 0.0), 2].max()]
    for side, sg in (("L", 1.0), ("R", -1.0)):
        j = np.asarray(fit.L[f"arm.{side}"], float)
        tops.append(R[(sg * R[:, 0] > abs(j[0]) + 0.02) & (np.abs(R[:, 1] - j[1]) < 0.02), 2].max())
    assert np.ptp(tops) < 0.01, np.round(tops, 4)
    K = P[np.unique([i for f, k in zip(m.faces, m.face_mat) if k == m.mats.index(SW.MAT_KNIT) for i in f])]
    for side, sg in (("L", 1.0), ("R", -1.0)):
        j = np.asarray(fit.L[f"arm.{side}"], float)
        for p in K[(np.abs(sg * K[:, 0] - abs(j[0])) < 0.05) & (np.abs(K[:, 1] - j[1]) < 0.05)]:
            around = np.linalg.norm(R[:, :2] - p[:2], axis=1) < 0.025
            assert not around.any() or p[2] < R[around, 2].max(), (side, np.round(p, 4))


def test_the_sleeves_run_into_the_body_under_the_arms(dressed):
    """An off-shoulder sweater, not a tube top with arm rings: the band goes over each upper arm and opens under it, so
    the knit is one piece (the sleeves run into the body under the arms: one piece of knit reaches both wrists and her
    waist; the webs under the arms are pieces of their own) and the skin straight under each shoulder joint, 3 cm along
    the arm, is under the knit, not the band."""
    ctx, parts, fit = dressed
    S = fit.S
    m = parts["sweater"].meshes[0]
    V = np.asarray(m.verts)
    knit = [f for f, k in zip(m.faces, m.face_mat) if k == 0]
    used = np.unique([i for f in knit for i in f])
    parent = {int(i): int(i) for i in used}

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for f in knit:
        for j in f[1:]:
            parent[find(int(j))] = find(int(f[0]))
    roots = np.array([find(int(i)) for i in used])
    main = roots == np.bincount(roots).argmax()
    wrist = abs(float(fit.L["wrist.L"][0]))
    Vm = V[used[main]]
    assert (Vm[:, 0] > 0.8 * wrist).any() and (Vm[:, 0] < -0.8 * wrist).any() and (np.abs(Vm[:, 0]) < 0.02).any()
    sh = SH.Shell.from_part(parts["body"])
    c = SP.merge(SW.DEFAULTS, ctx.spec.get("sweater"))
    band = SW.fields(sh, fit, c)["band"]
    s, V = SH.both_sides(sh.verts, fit.L), sh.verts
    for sg in (1.0, -1.0):
        arm = np.asarray(fit.L["arm.L" if sg > 0 else "arm.R"], float)
        under = (sg * V[:, 0] > 0) & (np.abs(s - 0.03 * S) < 0.006) & (V[:, 2] < arm[2] - 0.02 * S) \
            & (np.abs(V[:, 1] - arm[1]) < 0.01)
        assert under.sum() >= 1 and (band[under] > c["band"]["height"] * S).all(), band[under]


def _front_map(parts, z, x=0.0, dist=1.2, size=500):
    """The pixel map (mkmmd.pixmap) of the dressed parts at rest from straight in front, at height z."""
    P, T, tm, mats = [], [], [], []
    off = 0
    for part in parts.values():
        for mesh in part.meshes:
            if not len(mesh.verts):
                continue
            ids = [mats.index(n) if n in mats else mats.append(n) or len(mats) - 1 for n in mesh.mats]
            for f, k in zip(mesh.faces, mesh.face_mat):
                for j in range(1, len(f) - 1):
                    T.append((f[0] + off, f[j] + off, f[j + 1] + off))
                    tm.append(ids[k])
            P.append(np.asarray(mesh.verts, float))
            off += len(mesh.verts)
    P = np.concatenate(P)
    M = np.eye(4)
    M[:3, 0], M[:3, 1], M[:3, 2], M[:3, 3] = (1, 0, 0), (0, 0, 1), (0, -1, 0), (x, -dist, z)
    geo = {"posed": P, "rest": P, "tris": np.asarray(T), "tri_object": np.zeros(len(T), np.int32),
           "tri_material": np.asarray(tm, np.int32), "tri_face": np.zeros(len(T), np.int32),
           "objects": np.array(["model"]), "materials": np.array(mats)}
    return PX.Map.make(geo, {"matrix": M.tolist(), "lens": 50.0, "sensor": [36.0, 24.0], "size": [size, size]})


def test_the_band_runs_on_across_the_front_of_each_armpit(dressed):
    """The band is one piece round the top of her body: seen from the front at rest, from her chest over each arm it runs
    on across the armpit's front, with no knit showing through the middle of its depth there (a band that opened across
    the front of the armpit would show the knit through a notch from its top to its bottom edge)."""
    _, parts, fit = dressed
    S = fit.S
    arm = np.asarray(fit.L["arm.L"], float)
    m = _front_map(parts, arm[2] - 0.05 * S)
    band, knit = m.mask([SW.MAT_RIB, SW.MAT_INK]), m.mask(SW.MAT_KNIT)
    _, ys, _ = m.cam.project(np.array([[0.0, -0.2, arm[2] + 0.02 * S], [0.0, -0.2, arm[2] - 0.12 * S]]))
    rows = np.arange(int(ys[0]), int(ys[1]) + 1)                           # the collar's heights, not the waistband's
    for sg in (1.0, -1.0):
        xs, _, _ = m.cam.project(np.array([[sg * 0.6 * arm[0], 0.0, arm[2]], [sg * 1.2 * arm[0], 0.0, arm[2]]]))
        shown = 0
        for col in range(int(min(xs)), int(max(xs)) + 1):
            hit = rows[band[rows, col]]
            if len(hit) > 1:
                third = (hit[-1] - hit[0]) // 3
                shown += int(knit[hit[0] + third:hit[-1] - third + 1, col].sum())
        assert shown == 0, (sg, shown)


def _arm_share(mesh):
    w = mesh.weights
    return sum(np.asarray(v, float) for b, v in w.items() if OF.is_arm(b))


def test_a_web_joins_each_sleeve_to_her_side_below_the_armpit(webbed):
    """Switched on, under each arm the knit reaches from the sleeve's underside to her side below the armpit, as the
    sweater in the photo does: seen from the front at rest, what shows halfway from the armpit's corner to the web's
    free edge is that knit (not her side or the space beyond); along its free edge its weights go from her side's (no
    arm) at the side end to the sleeve's (all arm) at the sleeve end, so it folds between them when the arm comes down
    and spans the gap when it goes up."""
    _, parts, fit = webbed
    S = fit.S
    info = parts["sweater"].info["web"]
    mesh = parts["sweater"].meshes[0]
    V = np.asarray(mesh.verts)
    share = _arm_share(mesh)
    arm = np.asarray(fit.L["arm.L"], float)
    m = _front_map(parts, arm[2] - 0.08 * S)
    for side, sg in (("L", 1.0), ("R", -1.0)):
        corner, at_side, at_sleeve = (np.asarray(info[side][k], float) for k in ("corner", "side", "sleeve"))
        assert corner[2] - at_side[2] > 0.06 * S                                # well below the armpit
        middle = corner + 0.5 * (0.5 * (at_side + at_sleeve) - corner)
        x, y, _ = m.cam.project(middle[None])
        a = m.at(int(x[0]), int(y[0]))
        assert a["material"] == SW.MAT_KNIT and np.linalg.norm(np.asarray(a["world"]) - middle) < 0.03 * S, (side, a)
        edge = np.flatnonzero((np.abs(V[:, 1] - corner[1]) < 1e-9) & (sg * V[:, 0] > 0.0))   # the free edge: in the plane
        edge = edge[np.linalg.norm(V[edge] - 0.5 * (at_side + at_sleeve), axis=1) < np.linalg.norm(at_sleeve - at_side)]
        by = np.argsort(np.linalg.norm(V[edge] - at_side, axis=1))
        assert len(edge) >= 8
        assert share[edge[by[:2]]].max() < 0.2 and share[edge[by[-2:]]].min() > 0.8, (side, share[edge[by]])


def _knit(part):
    """The sweater's knit vertices (n, 3)."""
    m = part.meshes[0]
    k = m.mats.index(SW.MAT_KNIT)
    return np.asarray(m.verts)[np.unique([i for f, fm in zip(m.faces, m.face_mat) if fm == k for i in f])]


def test_the_knit_spans_the_armpits_instead_of_going_into_them(dressed):
    """Under each arm the knit runs from the sleeve into the body in a wide curve, as cloth does, instead of following
    the skin's crease up into the armpit: in the upright plane through the shoulder joint, the knit between her side
    and the arm is everywhere at least 6 cm below the joint (following the crease it would reach to 3.5 cm under it)."""
    _, parts, fit = dressed
    K = _knit(parts["sweater"])
    S = fit.S
    for side, sg in (("L", 1.0), ("R", -1.0)):
        arm = np.asarray(fit.L[f"arm.{side}"], float)
        under = (np.abs(K[:, 1] - arm[1]) < 0.004) & (np.abs(sg * K[:, 0] - sg * arm[0]) < 0.03 * S) \
            & (K[:, 2] < arm[2] - 0.01 * S)
        assert under.sum() >= 3
        assert K[under, 2].max() < arm[2] - 0.06, (side, arm[2] - K[under, 2].max())


def test_the_knit_spans_the_cleavage(dressed):
    """Over her breasts the knit goes straight across from one to the other, as a sweater does, instead of dipping into
    the cleavage: at the height of the bust its middle is no more than 1.5 mm behind the most forward knit beside it."""
    _, parts, fit = dressed
    K = _knit(parts["sweater"])
    z = float(fit.L["bust_front"][2])
    row = np.abs(K[:, 2] - z) < 0.01
    middle = row & (np.abs(K[:, 0]) < 0.004)
    beside = row & (np.abs(K[:, 0]) > 0.02) & (np.abs(K[:, 0]) < 0.06)
    assert middle.sum() >= 2 and beside.sum() >= 4
    assert K[middle, 1].min() - K[beside, 1].min() < 0.0015, (K[middle, 1].min(), K[beside, 1].min())



def test_the_knit_lies_on_the_breasts_and_follows_her_waist_in_loosely(dressed):
    """The loose knit shows her figure as the drawing does: on their lower half (under the band) it lies on her breasts,
    and below them it follows her in towards the waist, loosely, instead of falling straight like a barrel: at her waist
    its sides stand 6 to 16 mm off her skin, its front 6 to 30 mm, and seen from the front it is at most a fifth wider
    than she is there."""
    ctx, parts, fit = dressed
    S = fit.S
    c = SP.merge(SW.DEFAULTS, ctx.spec.get("sweater"))
    m = parts["sweater"].meshes[0]
    V = np.asarray(m.verts)
    K = V[np.unique([i for f, k in zip(m.faces, m.face_mat) if k == 0 for i in f])]
    d = fit.skin.closest(K)[0]
    skin = SH.Shell.from_part(parts["body"])
    skin.fields["arm"] = skin.share(SH.ARM)
    fronts = SW.breasts(skin, fit)
    cl, z_front = c["body"]["cling"] * S, min(float(f[2]) for f in fronts)
    on = (SW._from_fronts(K, fronts) < 0.6 * cl) & (K[:, 2] < z_front)
    assert len(fronts) == 2 and on.sum() > 20
    assert np.median(d[on]) < c["body"]["offset"] * S + 0.003
    z_w = 0.5 * (z_front - cl + parts["sweater"].info["z_waistband"])        # her waist, under the breasts
    at = np.abs(K[:, 2] - z_w) < 0.01
    az = SH.azimuth(K, (0.0, float(np.mean(K[at, 1]))))
    for a0, lo, hi in ((0.0, 0.006, 0.03), (0.25, 0.006, 0.016), (0.75, 0.006, 0.016)):   # front, her left, her right
        sel = at & (np.abs((az - a0 + 0.5) % 1.0 - 0.5) < 0.04)
        assert lo < np.median(d[sel]) < hi, (a0, np.median(d[sel]))
    B = np.asarray(parts["body"].meshes[0].verts)
    waist = (np.abs(B[:, 2] - z_w) < 0.01) & (np.abs(B[:, 0]) < 0.2)       # her trunk there, not her hands
    assert np.ptp(K[at & (np.abs(K[:, 0]) < 0.2), 0]) < 1.2 * np.ptp(B[waist, 0])


@pytest.mark.parametrize("name", ["sweater", "pants", "socks"])
def test_drawn_lines_lie_just_outside_what_they_are_drawn_on(dressed, name):
    """Every pen stroke of a garment lies a fraction of a millimetre outside its cloth: never inside it, where it cannot
    be seen, never floating off it. (A stroke's edge may dip into a tight crease, the band in an armpit: a few
    vertices.)"""
    _, parts, _ = dressed
    m = parts[name].meshes[0]
    ink = m.mats.index(next(n for n in m.mats if n.endswith("線")))
    V = np.asarray(m.verts)
    fm = np.asarray(m.face_mat)
    cloth = OF.Skin(V, [tuple(f) for f, k in zip(m.faces, fm) if k != ink])
    iv = np.unique([i for f, k in zip(m.faces, fm) if k == ink for i in f])
    near = cloth.closest(V[iv])[0]
    assert len(iv) > 50 and parts[name].info["lines"] > 5
    assert (near <= 0.0).mean() < 0.03 and np.percentile(near, 95) < 0.0012


def test_no_line_outlines_her_breasts(dressed):
    """The knit lying on her breasts shows them; no pen stroke outlines them: none on the knit within `cling` + 1 cm of
    a breast's front (the band's ribs above them are the band's), and none on her skin (the cleavage above the band)."""
    ctx, parts, fit = dressed
    m = parts["sweater"].meshes[0]
    ink = m.mats.index(SW.MAT_INK)
    V, fm = np.asarray(m.verts), np.asarray(m.face_mat)
    knit = OF.Skin(V, [tuple(f) for f, k in zip(m.faces, fm) if k == 0])
    rib = OF.Skin(V, [tuple(f) for f, k in zip(m.faces, fm) if k == 1])
    iv = np.unique([i for f, k in zip(m.faces, fm) if k == ink for i in f])
    dk, dr = np.abs(knit.closest(V[iv])[0]), np.abs(rib.closest(V[iv])[0])
    assert np.min([dk, dr], axis=0).max() < 0.003                     # every stroke on cloth, none on her skin
    skin = SH.Shell.from_part(parts["body"])
    skin.fields["arm"] = skin.share(SH.ARM)
    fronts = SW.breasts(skin, fit)
    c = SP.merge(SW.DEFAULTS, ctx.spec.get("sweater"))
    on_knit = dk < dr
    near = SW._from_fronts(V[iv], fronts) < c["body"]["cling"] * fit.S + 0.01
    assert len(fronts) == 2 and not (on_knit & near).any()


def test_pants_run_from_the_waist_into_the_socks(dressed):
    _, parts, fit = dressed
    P = np.asarray(parts["pants"].meshes[0].verts)
    S = np.asarray(parts["socks"].meshes[0].verts)
    hip, ankle = float(fit.L["leg.L"][2]), float(fit.L["ankle.L"][2])
    assert P[:, 2].max() == pytest.approx(hip + 0.080 * fit.S, abs=0.003)
    assert P[:, 2].min() < S[:, 2].max() - 0.015                     # tucked in: the sock's cuff is over the hem
    assert S[:, 2].min() < ankle - 0.05                               # the socks go down over the feet


def test_socks_stand_over_the_tucked_pants(dressed):
    """Where a sock covers the pants it stands further off the skin than they do, so the pants never show through."""
    _, parts, fit = dressed
    pants = parts["pants"].meshes[0]
    socks = parts["socks"].meshes[0]
    Vp, Vs = np.asarray(pants.verts), np.asarray(socks.verts)
    lo = float(Vp[:, 2].min())
    sel = (Vs[:, 2] > lo + 0.005) & (Vs[:, 2] < float(Vs[:, 2].max()) - 0.01)
    ds = fit.skin.closest(Vs[sel])[0]
    dp = fit.skin.closest(Vp[Vp[:, 2] < float(Vs[:, 2].max()) - 0.01])[0]
    assert ds.min() > dp.max()


def test_the_sneakers_face_sits_on_the_foot(dressed):
    """The painted face is in the upper's grid coordinates: u = 0.5 at the toe, v = 0 at the welt; the toe cap is the
    cap colour there, the stripes run along both sides, the laces cross the front higher up; the edges are drawn in ink
    as in the design: the toe cap's and the foxing's, and each lace a light bar between two ink lines."""
    parts = dressed[1]
    m = parts["sneakers"].meshes[0]
    V = np.asarray(m.verts)
    corners = np.array([i for f in m.faces for i in f])
    fm = np.repeat(np.asarray(m.face_mat), [len(f) for f in m.faces])
    up = fm == 0
    left = V[corners, 0] > 0
    toe = np.argmin(np.where(up & left, V[corners, 1], np.inf))
    assert m.uv[toe][0] == pytest.approx(0.5, abs=0.03)
    c = SP.merge(SN.DEFAULTS, None)
    col = {k: np.asarray(v, float) for k, v in {"upper": (0.9, 0.85, 0.9), "cap": (1.0, 1.0, 0.9), "sole": (1, 1, 1),
                                                "lace": (0.95, 0.95, 0.95), "speckle": (0.6, 0.6, 0.6),
                                                "stripe_front": (0.0, 0.0, 1.0), "stripe_back": (1.0, 0.0, 0.0),
                                                "tread": (0.5, 0.5, 0.5), "ink": (0.1, 0.0, 0.1)}.items()}
    c["speckles"]["count"] = 0
    img = SN.face_texture(col, c, size=(720, 200)).astype(float) / 255.0

    def at(deg, v):
        return img[int((1.0 - v) * 199), int((deg / 360.0 + 0.5) * 719), :3]
    assert np.allclose(at(0.0, 0.1), col["cap"], atol=0.01)
    for sgn in (1, -1):                                              # both sides, low at the front, up the heel
        assert np.allclose(at(sgn * 70.0, 0.21), col["stripe_front"] * 0.85 + col["stripe_back"] * 0.15, atol=0.15)
        assert at(sgn * 170.0, 0.84)[0] > 0.8
        assert np.allclose(at(sgn * 120.0, c["foxing"]), col["ink"], atol=0.02)       # the foxing's edge
    assert np.allclose(at(0.0, 0.5 + 0.42 / 8), col["lace"], atol=0.01)
    assert np.allclose(at(0.0, c["cap"]["v"]), col["ink"], atol=0.02)                 # the toe cap's edge
    toe_col = img[:, 360, :3]
    lace, ink = (np.all(np.abs(toe_col - col[k]) < 0.02, 1) for k in ("lace", "ink"))
    rows = np.flatnonzero(lace)
    assert len(rows) >= int(c["laces"]["count"])
    assert all(ink[r - 2:r].any() and ink[r + 1:r + 3].any() for r in rows)


def test_the_choker_rings_the_neck_with_its_flower_in_front(dressed):
    _, parts, fit = dressed
    ch = parts["choker"]
    V = np.asarray(ch.meshes[0].verts)
    neck, head = float(fit.L["neck"][2]), float(fit.L["head"][2])
    assert neck < V[:, 2].min() and V[:, 2].max() < head
    flower = np.asarray(ch.info["flower"])
    assert flower[1] < float(fit.L["neck"][1]) - 0.02 and flower[0] > 0.0      # at the throat, towards her left
    assert ch.info["min_gap"] > 0.0
