"""Round animal ears drawn soft (mkmmd/model/parts/ears.py `shape = "round"`, `look = "soft"`; cat_ear_geo, cat_tex): the
planform rises into a round arc of the asked radius and has no point anywhere, the shell keeps the cat ears' rig, the
look is the [colors.ears] palette in smooth gradients with outlines no darker than its shadow, the tuft ends round.
The cat ears' defaults are tests/test_model_hair_cats.py's. bpy-free: the stand-in head of tests/hair_fixtures.py.
Run: pytest -q tests/test_model_hair_ears.py"""
import numpy as np
import pytest
from PIL import Image

from hair_fixtures import make_ctx
from mkmmd.core import families
from mkmmd.model import part as P
from mkmmd.model.parts import cat_ear_geo as EG
from mkmmd.model.parts import ears as E
from mkmmd.model.parts import hair_fit, hair_rig, hair_tex

EARS = {"outer": "#4a2a26", "outer_shadow": "#2f1917", "outer_sheen": "#7a4a40", "inner": "#b06a5c",
        "inner_light": "#d99a8a", "tuft": "#e6c9a8"}
ROUND = {"shape": "round", "look": "soft", "width": 0.074, "height": 0.058, "tip_radius": 0.024, "tufts": 1}


def build(tmp_path, ears=None, colors=EARS):
    ctx = make_ctx(tmp_path, {"colors": {"ears": dict(colors)}})
    fit = hair_fit.HeadFit(ctx.parts["head"].info)
    rig = hair_rig.Rig()
    piece = E.build_ears(ctx, fit, dict(ROUND, **(ears or {})), rig, hair_tex.palette())
    return ctx, fit, rig, piece


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    return build(tmp_path_factory.mktemp("round_ears"))


def outline(sh, n=400):
    """The planform's outline (x across, h height above the hull) from the outer base corner over the top to the inner."""
    t = np.linspace(0.0, 1.0, n)
    xi, xo = sh.edges(t)
    h = t * sh.p.H
    return np.concatenate([np.stack([xo, h], -1), np.stack([xi, h], -1)[::-1]])


def test_a_round_planform_rises_into_an_arc_of_its_radius_with_no_point():
    p = EG.EarParams(W=0.074, H=0.058, shape="round", tip_r=0.024, apex_shift=0.0)
    sh = EG.EarShape(p)
    xi, xo = sh.edges(np.array([0.0, 1.0]))
    assert xo[0] - xi[0] == pytest.approx(0.074) and abs(xo[1] - xi[1]) < 1e-9      # full width at the hull, 0 at the top
    O = outline(sh)
    top = O[O[:, 1] > p.H - 0.6 * p.tip_r]                       # the arc: points within its upper part
    centre = np.array([0.0, p.H - p.tip_r])
    assert np.allclose(np.linalg.norm(top - centre, axis=1), p.tip_r, atol=2e-4)
    d = np.diff(O, axis=0)
    cross = d[:-1, 0] * d[1:, 1] - d[:-1, 1] * d[1:, 0]
    turn = np.degrees(np.abs(np.arctan2(cross, (d[:-1] * d[1:]).sum(1))))
    assert turn.max() < 6.0                                       # a smooth outline: no corner, no point


def test_a_round_ears_rings_follow_its_arc_to_the_top():
    p = EG.EarParams(W=0.074, H=0.058, shape="round", tip_r=0.024, apex_shift=0.0)
    shell = EG.build_shell(p, lambda x, y: np.full(np.shape(x), -0.03))
    R = shell["R"]
    rows = np.array(shell["ring_t"][1:])
    assert (np.diff(rows) > 0).all() and rows[-1] > 0.98            # rings up to just under the top
    last = shell["verts"][(shell["rings"] - 1) * R:shell["rings"] * R]
    assert np.ptp(last[:, 0]) < 0.35 * p.tip_r                       # the last ring is small: the apex closes a dome
    ang = np.degrees(np.arcsin(np.clip((rows * p.H - (p.H - p.tip_r)) / p.tip_r, -1.0, 1.0)))
    arc = ang[rows * p.H > p.H - p.tip_r]
    assert np.diff(arc).max() < 12.0                                 # round the arc in steps of a few degrees


def test_the_point_planform_stays_the_cats():
    a, b = EG.EarShape(EG.EarParams()), EG.EarShape(EG.EarParams(shape="point"))
    t = np.linspace(-0.2, 1.0, 50)
    assert np.array_equal(np.stack(a.edges(t)), np.stack(b.edges(t))) and a.rows == EG.T_ROWS
    with pytest.raises(ValueError, match="shape"):
        EG.EarShape(EG.EarParams(shape="square"))


def test_round_soft_ears_keep_the_twitch_bones_and_the_ears_chain(built):
    ctx, fit, rig, piece = built
    names = sorted(b.name for b in rig.bones)
    assert names == sorted(["猫耳左", "猫耳左1", "猫耳左2", "猫耳左3", "猫耳右", "猫耳右1", "猫耳右2", "猫耳右3"])
    by = {b.name: b for b in rig.bones}
    assert all(families.classify(n) == "ears" for n in names)
    assert sorted(piece.info["twitch"]) == ["猫耳右", "猫耳左"] and by["猫耳左"].parent == "頭"
    part = P.Part("hair", meshes=piece.meshes, materials=piece.materials, bones=rig.bones, bodies=rig.bodies,
                  joints=rig.joints)
    P.check(part)
    for m in piece.meshes:
        assert np.allclose(sum(np.asarray(w, float) for w in m.weights.values()), 1.0, atol=1e-6)


def test_round_ears_have_the_asked_size_and_a_round_top(built):
    ctx, fit, rig, piece = built
    m = piece.meshes[0]
    V = np.asarray(m.verts, float)
    for side, sd in piece.info["sides"].items():
        u, ex, O = sd["axis"], sd["outward"], sd["base_centre"]
        mine = np.sign(V[:, 0]) == np.sign(O[0])
        Q = V[mine] - (O + sd["z_hull"] * u)                         # from the hull crossing of the axis
        top = float((Q @ u).max())
        assert top == pytest.approx(0.058, abs=0.002)
        near = np.abs(Q @ u) < 0.002
        assert np.ptp(Q[near] @ ex) == pytest.approx(0.074, abs=0.004)
        # the shell's outline in the ear's plane near its top (each ring's extent across) lies on one circle of about
        # the asked radius
        h = np.round(Q @ u, 5)
        rings = [r for r in np.unique(h) if top - 0.020 < r < top - 0.0005 and (h == r).sum() >= 8]
        A = np.array([(f(Q[h == r] @ ex), r) for r in rings for f in (np.min, np.max)] +
                     [(float(Q[np.argmax(h)] @ ex), top)])
        sol, *_ = np.linalg.lstsq(np.column_stack([2 * A[:, 0], 2 * A[:, 1], np.ones(len(A))]), (A ** 2).sum(1),
                                  rcond=None)
        assert np.sqrt(sol[2] + sol[0] ** 2 + sol[1] ** 2) == pytest.approx(0.024, abs=0.003)


def test_the_soft_look_paints_the_palette_without_fibres_and_outlines_no_darker_than_its_shadow(built):
    ctx, fit, rig, piece = built
    assert [m.name for m in piece.materials] == ["猫耳", "猫耳内", "猫耳毛"]
    shadow = hair_tex.srgb(EARS["outer_shadow"])
    for mt in piece.materials:
        assert min(mt.edge_color[:3]) >= shadow.min() - 1e-6 and sum(mt.edge_color[:3]) >= shadow.sum() - 1e-6
        img = np.asarray(Image.open(ctx.tex_dir / mt.texture).convert("RGB"), float) / 255.0
        assert np.abs(np.diff(img, 2, axis=1)).max() < 0.012             # no fibres: smooth across every row
    fur = np.asarray(Image.open(ctx.tex_dir / piece.materials[0].texture).convert("RGB"), float) / 255.0
    assert np.abs(fur[fur.shape[0] // 3] - hair_tex.srgb(EARS["outer"])).max() < 0.03        # the outer colour
    inner = np.asarray(Image.open(ctx.tex_dir / piece.materials[1].texture).convert("RGB"), float) / 255.0
    mid = inner[int(inner.shape[0] * 0.25), inner.shape[1] // 2]
    assert np.abs(mid - hair_tex.srgb(EARS["inner_light"])).max() < 0.03                     # light up its middle
    ramp = np.asarray(Image.open(ctx.tex_dir / piece.materials[0].toon).convert("RGB"), float)[-1, 0] / 255.0
    assert np.allclose(ramp, shadow / hair_tex.srgb(EARS["outer"]), atol=0.02)               # shaded to outer_shadow


def test_a_round_ears_tuft_is_one_soft_puff_with_a_round_end(built):
    ctx, fit, rig, piece = built
    for side, sd in piece.info["sides"].items():
        assert len(sd["tufts"]) == 1
        t = sd["tufts"][0]
        assert 0.008 < np.linalg.norm(np.diff(t, axis=0), axis=1).sum() < 0.020
        assert (t[0] - sd["base_centre"]) @ sd["outward"] < 0.0           # at the inner edge
    sp = EG.tuft_specs(EG.EarShape(EG.EarParams(shape="round", W=0.074, H=0.058)), 1, 0.074, np.random.default_rng(1))
    st = EG.build_tufts(sp)[0]
    assert sp[0]["tip"] == "round"
    near_end = st["verts"][st["arc"] > 0.9 * st["length"]]
    assert np.ptp(near_end, axis=0).max() > 0.002                         # it ends in a round end, not a needle


def test_the_soft_look_without_the_extra_colours_derives_them(tmp_path):
    ctx, fit, rig, piece = build(tmp_path, colors={"outer": "#4a2a26", "inner": "#b06a5c", "tuft": "#e6c9a8"})
    assert piece.materials[0].edge_color[0] == pytest.approx(0x4a / 255 * 0.64, abs=1e-6)


def test_an_unknown_shape_or_look_is_refused(tmp_path):
    with pytest.raises(ValueError, match="shape"):
        build(tmp_path, ears={"look": "felt"})


def test_a_round_ear_stood_over_the_skin_has_no_vertex_inside_the_head(tmp_path):
    ctx, fit, rig, piece = build(tmp_path, ears={"sink": -0.005})
    V = np.concatenate([m.verts for m in piece.meshes], 0)
    assert fit.skull.height_dist(V).min() > 0.0005                        # every vertex over the skin, none inside
