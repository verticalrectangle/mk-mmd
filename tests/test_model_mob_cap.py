"""bpy-free invariants of the mob cap part, built on the girl base bald and over her hair: the band runs where its keys
say and hugs what is under it, nothing of the skull or the hair comes through its cloth, the frill hangs under the band
off the hair, the crescent and the bow sit on the cloth, the crown slouches as asked; and the crescent's own shape."""
import numpy as np
import pytest

from mkmmd.model import build as BD
from mkmmd.model import part as PT
from mkmmd.model import spec as SP
from mkmmd.model.parts import mob_cap as MC
from mkmmd.model.parts import outfit_fit as OF
from mkmmd.model.parts.hair_fit import HeadFit

C = SP.merge(MC.DEFAULTS, {})


def _build(tmp, parts):
    spec = SP.load("base:girl", ["model.parts=[" + ", ".join(f'"{p}"' for p in parts) + "]"])
    ctx = BD.BuildCtx(spec, tmp, seed=1)
    return ctx, {p.name: p for p in BD.run(spec, ctx=ctx)}


@pytest.fixture(scope="module")
def bald(tmp_path_factory):
    return _build(tmp_path_factory.mktemp("cap_bald"), ["body", "head", "mob_cap"])


@pytest.fixture(scope="module")
def over_hair(tmp_path_factory):
    return _build(tmp_path_factory.mktemp("cap_hair"), ["body", "head", "hair", "mob_cap"])


def under(parts, z_min):
    """What the cap was fitted over (the parts built before it)."""
    return MC.under({k: p for k, p in parts.items() if k != "mob_cap"}, z_min)


def piece(cap, *names):
    """(vertices, triangles) of the cap's faces in the materials `names`, the triangles indexing all its vertices."""
    m = cap.meshes[0]
    ks = [m.mats.index(n) for n in names]
    F = [f for f, k in zip(m.faces, m.face_mat) if k in ks]
    return np.asarray(m.verts, float), np.array([[f[0], f[i], f[i + 1]] for f in F for i in range(1, len(f) - 1)])


def inside_cloth(cap, pts):
    """Whether each of `pts` (n, 3) lies inside the cap's cloth (band and crown): nearer the crown's axis than the cloth
    is in the same direction at the same height."""
    V, T = piece(cap, "帽子", "帽子バンド")
    cloth = OF.Skin(V, T)
    axis = cap.info["band"]["seam"].mean(0)
    out = np.zeros(len(pts), bool)
    zs = np.round(pts[:, 2] / 0.001).astype(int)
    for z in np.unique(zs):
        sel = zs == z
        c = np.array([axis[0], axis[1], z * 0.001])
        r = cloth.outline(c, np.array([0.0, -1.0, 0.0]), np.array([1.0, 0.0, 0.0]), 720, "near", despike=0)
        P = pts[sel] - c
        ang = np.arctan2(P[:, 0], -P[:, 1]) % (2.0 * np.pi)
        out[sel] = np.hypot(P[:, 0], P[:, 1]) < np.interp(ang, 2.0 * np.pi * np.arange(721) / 720, np.append(r, r[0]))
    return out


@pytest.mark.parametrize("which", ["bald", "over_hair"])
def test_the_band_runs_where_its_keys_say_and_hugs_what_is_under_it(which, request):
    _, parts = request.getfixturevalue(which)
    cap = parts["mob_cap"]
    PT.check(cap)
    assert [m.name for m in cap.materials] == list(MC.MATS) and list(cap.meshes[0].weights) == ["頭"]
    assert np.allclose(cap.meshes[0].weights["頭"], 1.0)
    fit = HeadFit(parts["head"].info, meshes=parts["head"].meshes)
    front = np.mean([np.asarray(e["center"], float)[2] for e in fit.eyes.values()]) + C["edge"]
    band = cap.info["band"]
    assert band["front"] == pytest.approx(front, abs=5e-4)                      # its lower edge at her front ...
    assert band["back"] == pytest.approx(front - C["drop"], abs=5e-4)            # ... and `drop` lower at her back
    assert np.abs((band["lower_edge"] - band["centre"]) @ band["normal"]).max() < 1e-6     # a ring in one plane
    assert np.allclose((band["seam"] - band["centre"]) @ band["normal"], C["band"], atol=1e-6)
    V, T = piece(cap, "帽子バンド")
    d = np.abs(under(parts, front - 0.1).closest(V[np.unique(T)])[0])
    assert C["gap"] - 3e-4 < d.min() < C["gap"] + 5e-4                           # hugging, `gap` off


@pytest.mark.parametrize("which", ["bald", "over_hair"])
def test_nothing_of_the_skull_or_the_hair_comes_through_the_cloth(which, request):
    _, parts = request.getfixturevalue(which)
    cap = parts["mob_cap"]
    band = cap.info["band"]
    pts = [np.asarray(parts["head"].info["skin"]["verts"], float)]
    if "hair" in parts:
        pts += [np.asarray(m.verts, float) for m in parts["hair"].meshes]
    P = np.vstack(pts)
    P = P[(P - band["centre"]) @ band["normal"] > 0.001]                         # above the band's lower edge
    assert len(P) > 100 and inside_cloth(cap, P).all()
    V, T = piece(cap, "帽子", "帽子バンド")
    d = np.abs(under(parts, band["back"] - 0.1).closest(V[np.unique(T)])[0])
    assert d.min() > C["gap"] - 3e-4                                             # and none of the cloth near them


def test_the_frill_hangs_under_the_band_off_the_hair(over_hair):
    _, parts = over_hair
    cap = parts["mob_cap"]
    band = cap.info["band"]
    V, T = piece(cap, "帽子フリル")
    F = V[np.unique(T)]
    h = (F - band["centre"]) @ band["normal"]
    assert h.max() <= 0.5 * C["thick"] + 1e-6                                    # sewn inside the band's rolled lip
    assert -h.min() == pytest.approx(C["frill"]["depth"], abs=5e-4)               # and as deep under it as asked
    assert cap.info["frill"]["depth"][0] == pytest.approx(C["frill"]["depth"], abs=5e-4)
    d = np.abs(under(parts, band["back"] - 0.1).closest(F)[0])
    assert d.min() > C["frill"]["margin"] - 3e-4                                 # no pleat on the hair


def test_the_crescent_and_the_bow_sit_on_the_cloth(over_hair):
    _, parts = over_hair
    cap = parts["mob_cap"]
    mo, bo = cap.info["moon"], cap.info["bow"]
    V, T = piece(cap, "帽子", "帽子バンド", "帽子フリル")
    cloth = OF.Skin(V, T)
    Vm, Tm = piece(cap, "帽子月")
    moon = Vm[np.unique(Tm)]
    d = cloth.closest(moon)[0]
    assert d.min() > 0.0004 and d.max() < 0.008                                 # on the cloth, nothing of it far off
    assert 0.0008 < mo["back"][0] and mo["back"][1] < 0.006
    assert mo["height"] == pytest.approx(C["moon"]["size"], abs=0.001) and mo["thick"] == pytest.approx(0.006, abs=1e-4)
    assert mo["theta"] == pytest.approx(C["moon"]["at"], abs=2.0)
    assert mo["above_roll"] == pytest.approx(C["moon"]["lift"])
    Vb, Tb = piece(cap, "帽子リボン")
    bow = Vb[np.unique(Tb)]
    db = cloth.closest(bow)[0]
    assert db.min() > 0.0005 and db.min() < 0.002                                # lying on the band, not in it
    assert bo["width"] == pytest.approx(C["bow"]["span"], abs=0.001) and bo["theta"] == pytest.approx(C["bow"]["at"],
                                                                                                    abs=2.0)
    assert np.allclose(bo["tails"], C["bow"]["tails"], atol=0.0015)


def test_the_crown_slouches_toward_her_back_left(bald):
    _, parts = bald
    cr = parts["mob_cap"].info["crown"]
    cc = C["crown"]
    assert cr["overhang_front_right"] == pytest.approx(cc["overhang"], abs=0.0005)
    assert cr["overhang_back_left"] == pytest.approx(cc["overhang"] + cc["slump"], abs=0.0005)
    assert cr["droop_front_right"] == pytest.approx(cc["hang"], abs=0.0005)
    assert cr["droop_back_left"] == pytest.approx(cc["hang"] + cc["droop"], abs=0.0005)
    assert cr["lean"] == pytest.approx(cc["lean"], abs=0.3) and cr["lean_toward"] == pytest.approx(cc["lean_toward"],
                                                                                                    abs=3.0)
    fit = HeadFit(parts["head"].info, meshes=parts["head"].meshes)
    assert cr["top"] == pytest.approx(float(fit.top[2]) + cc["top"], abs=1e-4)


def test_the_crescent_is_two_arcs_with_rounded_horns():
    mc = C["moon"]
    V, F, uv, outline = MC.crescent(mc)
    r = np.linalg.norm(outline, axis=1)
    assert r.max() == pytest.approx(0.5 * mc["size"], abs=2e-4)                  # the outer arc round its centre
    back = outline[np.abs(outline[:, 1]) < 0.002]
    assert np.ptp(back[:, 0]) == pytest.approx(mc["width"], abs=4e-4)            # as wide as asked in its middle
    assert np.ptp(V[:, 2]) == pytest.approx(mc["thick"], abs=1e-6)
    P, Q = np.roll(outline, 1, 0), np.roll(outline, -1, 0)                       # no corner sharper than a soft turn
    a, b = G_unit(outline - P), G_unit(Q - outline)
    assert np.degrees(np.arccos(np.clip(np.einsum("ij,ij->i", a, b), -1.0, 1.0))).max() < 40.0
    tips = outline[np.argsort(r)[:4]]                                            # its horns end in rounds 4 mm across
    assert np.linalg.norm(outline[None] - tips[:, None], axis=2).max() > 0.02
    sec = MC.slab(0.006, 0.003, mc["round"])
    corner = sec[(sec[:, 0] > 0.006 - mc["round"]) & (sec[:, 1] > 0.003 - mc["round"])]
    assert np.allclose(np.linalg.norm(corner - [0.006 - mc["round"], 0.003 - mc["round"]], axis=1), mc["round"])
    assert len(F) and np.isfinite(uv).all()


def G_unit(v):
    return v / np.linalg.norm(v, axis=1, keepdims=True)
