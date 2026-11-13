"""The nightgown part (mkmmd/model/parts/gown.py) on the girl base: its stripes, its hem, its capelet over a raised arm,
its chains and colliders, and no skin showing through it. bpy-free."""
import numpy as np
import pytest

from mkmmd import pixmap as PX
from mkmmd.model import assemble as AS
from mkmmd.model import build as BD
from mkmmd.model import lab as LAB
from mkmmd.model import part as PT
from mkmmd.model import spec as SP
from mkmmd.model.parts import gown as GW
from mkmmd.model.parts import outfit as OU
from mkmmd.model.parts import outfit_geo as G
from mkmmd.model.parts import outfit_legs as LG
from mkmmd.model.parts import outfit_rig as ORIG
from mkmmd.model.parts.hair_tex import srgb


@pytest.fixture(scope="module")
def dressed(tmp_path_factory):
    """The girl base in the gown: (ctx, {part name: Part}, the body's Fit)."""
    spec = SP.load("base:girl", ['model.parts=["body", "head", "gown"]'])
    ctx = BD.BuildCtx(spec, tmp_path_factory.mktemp("gown"), seed=1)
    parts = {p.name: p for p in BD.run(spec, ctx=ctx)}
    ctx.part = "test"
    return ctx, parts, OU.make_fit(ctx)


@pytest.fixture(scope="module")
def posable(dressed, tmp_path_factory):
    """The dressed girl as one model the lab can pose."""
    asm = AS.assemble(list(dressed[1].values()), name="girl", scale=0.08)
    return LAB.from_pmx(asm.pmx, tmp_path_factory.mktemp("posable"), 0.08, "girl")


def _look_at(eye, at):
    back = (np.asarray(eye, float) - at) / np.linalg.norm(np.asarray(eye, float) - at)
    right = np.cross([0.0, 0.0, 1.0], back)
    right /= np.linalg.norm(right)
    M = np.eye(4)
    M[:3, 0], M[:3, 1], M[:3, 2], M[:3, 3] = right, np.cross(back, right), back, eye
    return M


def _map(m, P, M, size=400):
    geo = {"posed": P, "rest": m.V, "tris": m.T, "tri_object": np.zeros(len(m.T), np.int32),
           "tri_material": m.tri_mat.astype(np.int32), "tri_face": np.zeros(len(m.T), np.int32),
           "objects": np.array(["model"]), "materials": np.array(m.mat_names)}
    return PX.Map.make(geo, {"matrix": np.asarray(M).tolist(), "lens": 50.0, "sensor": [36.0, 24.0],
                             "size": [size, size]})


def test_the_gown_passes_the_part_checks_and_builds_its_pieces(dressed):
    """The part is well formed (part.check) and has its pieces: the striped gown, the capelet, the sleeves, the frills,
    the bow, the lining, socks and shoes."""
    _, parts, _ = dressed
    p = parts["gown"]
    PT.check(p)
    mats = {n for m in p.meshes for n in m.mats}
    assert {GW.MAT_GOWN, GW.MAT_ROBE, GW.MAT_SLEEVE, GW.MAT_TRIM, GW.MAT_BOW, GW.MAT_LINING, GW.MAT_SOCK,
            GW.MAT_SHOE} <= mats, mats


def test_the_hem_is_at_its_height_and_ellipse(dressed):
    """The striped cloth ends `hem.above` over the ankle joints, its ring rx by ry round the axis (within 1.5 cm: its
    folds wave it), and the frill hangs below it."""
    ctx, parts, fit = dressed
    c = SP.merge(GW.DEFAULTS, ctx.spec.get("gown"))
    info = parts["gown"].info
    z_ank = 0.5 * (float(fit.L["ankle.L"][2]) + float(fit.L["ankle.R"][2]))
    assert abs(info["z_hem"] - (z_ank + c["hem"]["above"] * fit.S)) < 1e-6
    ring = info["rings"][-1]
    assert abs(float(ring[:, 2].min()) - info["z_hem"]) < 0.002
    rx = float(np.abs(ring[:, 0]).max())
    ry = float(info["axis_y"] - ring[:, 1].min())
    assert abs(rx - c["hem"]["rx"] * fit.S) < 0.015 and abs(ry - c["hem"]["ry"] * fit.S) < 0.015, (rx, ry)
    assert float(info["hem_rings"][-1][:, 2].mean()) < info["z_hem"] - 0.6 * c["frill"]["height"] * fit.S


def test_the_stripes_run_straight_down_and_are_counted(dressed):
    """The stripes are painted in the gown's picture, `count` round it at `share` of the pitch, none on the yoke, and
    each column of the body's rings keeps its turn round the axis all the way down (within a degree, where it is
    kept off her skin), so they run straight down the
    folds (a column that turned would wind the stripes into a spiral)."""
    ctx, parts, fit = dressed
    c = SP.merge(GW.DEFAULTS, ctx.spec.get("gown"))
    info = parts["gown"].info
    col = {k: np.array(srgb(v), float) for k, v in GW.COLORS.items()}
    pic = GW.picture(col, c, lambda th, z: np.zeros((len(np.atleast_1d(z)), len(th))), info["z_hem"], info["z_top"],
                     info["z_seam"], fit.S)[..., :3].astype(float)
    rows = pic.shape[0]
    seam_row = int((info["z_top"] - info["z_seam"]) / (info["z_top"] - info["z_hem"]) * rows)
    lilac = np.abs(pic - np.array(srgb(GW.COLORS["stripe"])) * 255).sum(-1) < 40
    row = lilac[(seam_row + rows) // 2]
    runs = int(np.count_nonzero(row & ~np.roll(row, 1)))
    assert runs == c["stripes"]["count"], runs
    assert abs(row.mean() - c["stripes"]["share"]) < 0.02, row.mean()
    assert not lilac[: max(seam_row - 3, 1)].any()                                # the yoke is plain
    assert (lilac[seam_row + 5: int(0.9 * rows)] == row[None, :]).mean() > 0.99   # the same columns all the way down
    th = G.theta_of(info["rings"].reshape(-1, 3), info["axis_y"]).reshape(info["rings"].shape[:2])
    spread = np.abs((th - th[:1] + np.pi) % (2 * np.pi) - np.pi)
    assert spread.max() < 0.02, spread.max()                       # within a degree (kept off her skin)


def test_its_chains_start_clear_of_the_colliders_they_meet(dressed):
    """Every dynamic body of the gown's chains past a chain's first lies clear of the static colliders it collides with
    at rest (her body's and the gown's own), so no chain warning is logged; the skirt hangs on at least 14 chains of
    5 and the sleeves' bags on the gown-sized colliders (group 12)."""
    ctx, parts, _ = dressed
    p = parts["gown"]
    statics = [rb for q in parts.values() for rb in q.bodies if rb.mode == "static"]
    rig = ORIG.Rig()
    rig.bodies = p.bodies
    assert GW.overlaps(rig, statics) == []
    assert not [w for w in ctx.warnings if "gown chain" in w]
    assert p.info["skirt"]["chains"] >= 14 and p.info["skirt"]["rows"] >= 5
    assert any(rb.group == ORIG.WORN for rb in p.bodies) and p.info["chains"] >= 4


def test_no_skin_shows_through_at_rest(posable, dressed):
    """Seen from eight directions round her at rest, none of the skin the gown covers shows: no hidden skin (a hole)
    and no drawn skin between her knees and the neckline away from her hands (a gap between pieces)."""
    m = posable
    _, parts, fit = dressed
    z_high = float(parts["gown"].info["neck"][:, 2].min()) - 0.01
    x_hand = abs(float(fit.L["wrist.L"][0])) - 0.015
    at = np.array([0.0, 0.0, 0.85])
    for k in range(8):
        a = 2 * np.pi * k / 8
        mp = _map(m, m.V, _look_at(at + np.array([2.6 * np.sin(a), -2.6 * np.cos(a), 0.25]), at))
        hidden = int(mp.mask(["skin_hidden"]).sum()) if "skin_hidden" in m.mat_names else 0
        ys, xs = np.nonzero(mp.mask(["skin"]))
        R = mp.points(xs, ys, rest=True) if len(xs) else np.zeros((0, 3))
        drawn = int(((R[:, 2] > 0.45) & (R[:, 2] < z_high) & (np.abs(R[:, 0]) < x_hand)).sum())
        assert hidden < 40 and drawn < 40, (k, hidden, drawn)


def test_the_capelet_covers_the_shoulder_with_the_arm_raised(posable, dressed):
    """With an arm raised 90 degrees out to her side the capelet goes up with the shoulder and covers it: seen from in
    front and from above, no skin (drawn or hidden) the gown covers shows round that shoulder joint (her neck above
    the neckline is skin, and stays)."""
    m = posable
    z_neck = float(dressed[1]["gown"].info["neck"][:, 2].min()) - 0.01
    for jp, sg in (("左", 1.0), ("右", -1.0)):
        j = m.heads[m.bones.index(f"{jp}腕")]
        P = m.skin(m.V, m.N, m.matrices({m.bones.index(f"{jp}腕"): LAB.rot((0.0, -1.0, 0.0), sg * 90.0)}))[0]
        for eye in ((sg * 0.1, -0.6, 0.1), (sg * 0.05, -0.1, 0.6)):
            mp = _map(m, P, _look_at(j + np.array(eye), j))
            ys, xs = np.nonzero(mp.mask([n for n in ("skin", "skin_hidden") if n in m.mat_names]))
            near = (np.linalg.norm(mp.points(xs, ys) - j, axis=1) < 0.06) & (mp.points(xs, ys, rest=True)[:, 2]
                                                                             < z_neck) if len(xs) else np.zeros(0, bool)
            assert int(near.sum()) < 30, (jp, eye, int(near.sum()))


def test_a_flat_shoe_has_no_strap_buckle_or_bow(dressed):
    """outfit_legs.shoes with `strap.enabled` false makes a slipper: no strap, buckle or bow (they ride the strap)."""
    _, _, fit = dressed
    shoes, satin = G.Soup("s", ["shoe", "sole"]), G.Soup("b", ["bow"])
    res = LG.shoes(fit, {"shoe": {"strap": {"enabled": False}}}, shoes, satin)
    assert res["L"]["strap"]["patches"] == [] and not satin.patches
    assert all(p.tag != "buckle" for p in shoes.patches)
