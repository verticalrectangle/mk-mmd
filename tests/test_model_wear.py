"""Garments cut from the body's own skin (mkmmd.model.parts.outfit_shell) and the parts made of them: sweater, pants,
socks, sneakers and choker, on the girl base. bpy-free."""
import numpy as np
import pytest

from mkmmd.model import build as BD
from mkmmd.model import spec as SP
from mkmmd.model.parts import outfit as OU
from mkmmd.model.parts import outfit_shell as SH
from mkmmd.model.parts import sneakers as SN

WEAR = 'model.parts=["body", "head", "sweater", "pants", "socks", "sneakers", "choker"]'


@pytest.fixture(scope="module")
def dressed(tmp_path_factory):
    """The girl base wearing every garment part: (ctx, {part name: Part}, the body's Fit)."""
    spec = SP.load("base:girl", [WEAR])
    ctx = BD.BuildCtx(spec, tmp_path_factory.mktemp("wear"), seed=1)
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
    cap colour there, the stripes run along both sides, the laces cross the front higher up."""
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
                                                "lace": (0.3, 0.3, 0.3), "speckle": (0.6, 0.6, 0.6),
                                                "stripe_front": (0.0, 0.0, 1.0), "stripe_back": (1.0, 0.0, 0.0),
                                                "tread": (0.5, 0.5, 0.5)}.items()}
    c["speckles"]["count"] = 0
    img = SN.face_texture(col, c, size=(720, 200)).astype(float) / 255.0

    def at(deg, v):
        return img[int((1.0 - v) * 199), int((deg / 360.0 + 0.5) * 719), :3]
    assert np.allclose(at(0.0, 0.1), col["cap"], atol=0.01)
    for sgn in (1, -1):                                              # both sides, low at the front, up the heel
        assert np.allclose(at(sgn * 70.0, 0.21), col["stripe_front"] * 0.85 + col["stripe_back"] * 0.15, atol=0.15)
        assert at(sgn * 170.0, 0.84)[0] > 0.8
    assert np.allclose(at(0.0, 0.5 + 0.42 / 8), col["lace"], atol=0.01)


def test_the_choker_rings_the_neck_with_its_flower_in_front(dressed):
    _, parts, fit = dressed
    ch = parts["choker"]
    V = np.asarray(ch.meshes[0].verts)
    neck, head = float(fit.L["neck"][2]), float(fit.L["head"][2])
    assert neck < V[:, 2].min() and V[:, 2].max() < head
    flower = np.asarray(ch.info["flower"])
    assert flower[1] < float(fit.L["neck"][1]) - 0.02 and flower[0] > 0.0      # at the throat, towards her left
    assert ch.info["min_gap"] > 0.0
