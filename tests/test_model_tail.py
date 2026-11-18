"""The big bushy tail (mkmmd/model/parts/tail.py): its thickness runs through its stations, it leaves the skin without a
vertex inside the body, the aim bone is the one `twitch family = "tail"` turns, the chain is stiff at the root and soft
at the tip, a vertex on the inside of a bend takes the bones of its own stretch of the tail, the fur stands off the
smooth body by the configured heights and the bands' edges follow the tufts, the end parts into a brush of lobes of
different lengths with parted, round ends in one closed surface, and the gap check tells inside a closed part by rays,
not by the nearest face.

bpy-free: the stand-in body of tests/hair_fixtures.py. Run: pytest -q tests/test_model_tail.py"""
from types import SimpleNamespace

import numpy as np
import pytest

from hair_fixtures import make_ctx
from mkmmd.core import families
from mkmmd.model import part as P
from mkmmd.model import spec as SP
from mkmmd.model.parts import cat_common as CC
from mkmmd.model.parts import tail as TL


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    ctx = make_ctx(tmp_path_factory.mktemp("tail"), {})
    ctx.part = "tail"
    return ctx, TL.build(ctx)


def surface(cfg, P_, s_):
    """The tail's Surface (no skin) along the centreline P_, s_."""
    prof = TL.Profile(cfg)
    frames = TL.Frames(P_, s_, float(cfg["embed"]))
    brush = TL.Brush(cfg, frames, prof, int(cfg["around"]))
    fur = TL.Fur(cfg, prof.L, TL.band_bounds(cfg, prof.L), cfg["bands"]["pattern"], prof, np.random.default_rng(1),
                 brush.s0)
    return TL.Surface(cfg, frames, prof, fur, brush, None), brush


def test_profile_hits_the_stations_and_holds_the_last():
    cfg = SP.merge(TL.DEFAULTS, {})
    prof = TL.Profile(cfg)
    L = prof.L
    for f, r in cfg["radius"]:
        assert abs(float(prof(f * L)) - r) < 1e-9
    assert abs(float(prof(L)) - cfg["radius"][-1][1]) < 1e-9


def test_no_vertex_inside_the_skin_and_the_part_checks(built):
    ctx, part = built
    P.check(part)
    mesh = part.meshes[0]
    skin = ctx.parts["body"].meshes[0]
    d = CC.mesh_distance(mesh.verts, skin.verts, skin.faces)
    near = np.linalg.norm(mesh.verts - ctx.land["tail_root"], axis=1) < 0.25
    assert d[near].min() > -0.0005                                       # cut at the skin, not run into the body
    w = np.zeros(len(mesh.verts))
    for v in mesh.weights.values():
        w += v
    assert np.allclose(w, 1.0)
    held = mesh.weights["下半身"]                                         # the base bends with her, not into her
    from_root = np.linalg.norm(mesh.verts - ctx.land["tail_root"], axis=1)
    on_skin = np.abs(d) < 0.0015
    assert held[on_skin & (from_root < 0.06)].min() > 0.99
    assert held[from_root > 0.15].max() == 0.0


def test_aim_bone_is_the_twitch_bone_and_the_chain_is_stiff_at_the_root(built):
    _, part = built
    bones = {b.name: b for b in part.bones}
    aim = bones[TL.AIM]
    assert aim.parent == "下半身"
    tail_family = [b.name for b in part.bones if families.classify(b.name) == "tail"]
    tops = [n for n in tail_family if families.classify(bones[n].parent) != "tail"]
    assert tops == [TL.AIM]                                              # what twitch family = "tail" turns
    bodies = {b.name: b for b in part.bodies}
    assert bodies[TL.AIM].mode == "static"
    chain = [b.name for b in part.bones if b.name != TL.AIM]
    assert len(chain) == TL.DEFAULTS["rig"]["bones"] and all(bodies[n].mode == "dynamic" for n in chain)
    joints = sorted(part.joints, key=lambda j: chain.index(j.b))
    assert joints[0].a == TL.AIM
    springs = np.array([j.spring_rot[0] for j in joints])
    swings = np.array([j.rot_hi[0] for j in joints])
    assert (np.diff(springs) < 0).all() and (np.diff(swings) > 0).all()
    assert min(part.info["chain"]["clearance"]) >= TL.DEFAULTS["rig"]["clear"] - 1e-4


def test_a_curled_tail_skins_each_vertex_to_its_own_stretch():
    cfg = SP.merge(TL.DEFAULTS, {"path": {"s": [0.0, 0.25, 0.5, 0.75, 1.0], "elev": [30.0, 120.0, 210.0, 300.0, 360.0]}})
    P_, s_ = TL.centreline(cfg, np.array([0.0, 0.01, 0.9]), np.array([0.0, 1.0, 0.0]))
    surf, _ = surface(cfg, P_, s_)
    V, _, _, s_v, _ = surf.arrays(0.3)
    L = float(cfg["length"])
    s_pts = np.linspace(0.08, L, 9)
    ch = TL.Rig().chain("尻尾", np.stack([np.interp(s_pts, s_, P_[:, k]) for k in range(3)], -1), anchor="尻尾",
                        kind="tail", radius=0.03, dynamic=False, hold=0.3)

    def own(w):                                                          # share of vertices whose main bone is theirs
        W = np.stack([w[n] for n in ch.names], 1)
        top, heavy = W.argmax(1), W.max(1) > 0.6
        lo, hi = s_pts[top[heavy]], s_pts[top[heavy] + 1]
        return ((s_v[heavy] > 1.5 * lo - 0.5 * hi) & (s_v[heavy] < 1.5 * hi - 0.5 * lo)).mean()
    assert own(TL.chain_weights(ch, s_pts, s_v)) > 0.999
    assert own(ch.weights(P=V)) < 0.98                                   # the nearest point of the chain is not enough


def test_fur_is_a_ring_at_each_band_edge_and_a_few_singles_before_the_brush():
    cfg = SP.merge(TL.DEFAULTS, {})
    prof = TL.Profile(cfg)
    L = prof.L
    bounds = TL.band_bounds(cfg, L)
    end = L - cfg["brush"]["length"]
    fur = TL.Fur(cfg, L, bounds, cfg["bands"]["pattern"], prof, np.random.default_rng(3), end)
    kinds = [r["kind"] for r in fur.rows]
    tc = cfg["tufts"]
    assert kinds.count("ring") == len(cfg["bands"]["pattern"]) - 1 and kinds.count("single") == tc["single"]["count"]
    for r in fur.rows:
        assert r["e"] < end                                              # no tuft runs into the brush
        if r["kind"] == "ring":
            assert tc["ring"]["count"][0] <= r["n"] <= tc["ring"]["count"][1]
            assert any(abs(r["e"] - b) < 1e-9 for b in bounds[1:-1])
        else:                                                            # outside the curve (phi pi), not inside
            assert np.cos(r["phase"]) < -0.3
    s = np.linspace(0.0, L, 3000)
    h, _, _, _ = fur.field(s, np.zeros_like(s), prof)                    # the inside of the curve, root to tip
    assert h.max() <= tc["max"] + 1e-9
    far = np.ones_like(s, bool)                                          # away from every ring the body is smooth
    for r in [r for r in fur.rows if r["kind"] == "ring"]:              # the singles sit outside the curve, not here
        far &= (s < r["s0"] - 0.012) | (s > r["e"] + 0.012)
    assert far.sum() > 300 and np.abs(h[far]).max() == 0.0
    with pytest.raises(ValueError, match="past where the brush parts"):
        TL.Fur(cfg, L, bounds, cfg["bands"]["pattern"], prof, np.random.default_rng(3), bounds[-2])


def test_band_edges_follow_the_tufts_not_a_ring():
    cfg = SP.merge(TL.DEFAULTS, {})
    prof = TL.Profile(cfg)
    L = prof.L
    bounds = TL.band_bounds(cfg, L)
    fur = TL.Fur(cfg, L, bounds, cfg["bands"]["pattern"], prof, np.random.default_rng(3), L - cfg["brush"]["length"])
    phi = np.linspace(0, 2 * np.pi, 360, endpoint=False)
    s = np.linspace(bounds[1] - 0.09, bounds[1] + 0.03, 600)
    _, band, _, _ = fur.field(s[:, None], phi[None, :], prof)
    band = np.where(band < 0, TL._band_of(np.repeat(s[:, None], len(phi), 1), fur.band_bounds), band)   # as painted
    edge = np.array([s[np.argmax(band[:, j] >= 1)] for j in range(len(phi))])   # where the second band starts
    assert 0.008 < edge.max() - edge.min() < 0.07                        # a zigzag a tuft deep, never a straight ring


def test_the_brush_parts_into_lobes_of_different_lengths_with_parted_round_ends(built):
    _, part = built
    mesh = part.meshes[0]
    use = {}
    for f in mesh.faces:                                                 # one closed surface: every edge twice
        for a in range(len(f)):
            e = tuple(sorted((f[a], f[(a + 1) % len(f)])))
            use[e] = use.get(e, 0) + 1
    assert set(use.values()) == {2}
    cfg = SP.merge(TL.DEFAULTS, {})
    br, L = part.info["brush"], part.info["length"]
    ends, pts = np.asarray(br["ends"]), np.asarray(br["points"])
    n = cfg["brush"]["lobes"]
    assert len(ends) == n + 1 and abs(ends[-1] - L) < 1e-9               # the middle lobe is the tail's end
    assert len(np.unique(np.round(ends, 4))) == n + 1 and ends[:-1].max() < L - 0.02   # all of different lengths
    for k in range(n):                                                   # neighbours' ends stand apart
        assert np.linalg.norm(pts[k] - pts[(k + 1) % n]) > 0.03
    P_, s_ = part.info["centreline"], part.info["s"]
    surf, brush = surface(cfg, P_, s_)
    from scipy.spatial import cKDTree
    middle = cKDTree(np.concatenate([surf.lobes[n][1].reshape(-1, 3), surf.lobes[n][2][None]]))
    for k in range(n):                                                   # each closes on the middle one, then parts
        s_k, rings, end = surf.lobes[k]
        half = len(s_k) // 2
        gap_half, gap_end = middle.query(rings[half])[0].min(), middle.query(rings[-1])[0].min()
        assert gap_half > 0.0 and gap_end > gap_half + 0.004
        width = np.linalg.norm(rings - rings.mean(axis=1, keepdims=True), axis=2).max(axis=1)
        assert width[-1] < 0.35 * width[0]                               # narrowed to its point
        step = np.linalg.norm(end - rings[-1].mean(axis=0))
        assert step < 0.6 * width[-1] + 0.004                           # a round end, not a spike past its last ring


def test_part_gaps_tells_inside_a_closed_part_by_rays_not_by_its_faces():
    u, v = np.meshgrid(np.linspace(0, np.pi, 13)[1:-1], np.linspace(0, 2 * np.pi, 24, endpoint=False), indexing="ij")
    ring = np.stack([np.sin(u) * np.cos(v), np.sin(u) * np.sin(v), np.cos(u)], -1).reshape(-1, 3)
    V = np.concatenate([ring, [[0, 0, 1.0], [0, 0, -1.0]]]) * 0.1          # a ball of radius 0.1, faces wound inward
    F = []
    for i in range(10):
        for j in range(24):
            a, b, c, d = i * 24 + j, i * 24 + (j + 1) % 24, (i + 1) * 24 + (j + 1) % 24, (i + 1) * 24 + j
            F.append([a, b, c, d])
    top, bot = 11 * 24, 11 * 24 + 1
    F += [[j, top, (j + 1) % 24] for j in range(24)] + [[10 * 24 + (j + 1) % 24, bot, 10 * 24 + j] for j in range(24)]
    ball = SimpleNamespace(meshes=[SimpleNamespace(verts=V, faces=F)])
    ctx = SimpleNamespace(parts={"ball": ball})
    pts = np.array([[0.0, 0.0, 0.11], [0.0, 0.0, 0.12], [0.0, 0.0, 0.09]])   # two outside, one inside
    got = TL.part_gaps(ctx, pts, np.ones(3), 0.0, "tail")["ball"]
    assert got["behind"] == 1 and got["gap"] == pytest.approx(-0.01, abs=0.002)
    assert CC.mesh_distance(pts, V, F)[0] < 0                             # the inward faces alone would say inside
