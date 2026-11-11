"""bpy-free invariants of the lock hairdo (hair_locks, hair_paint): hair combed from a parting is rooted on it and
leaves it across, a group's locks lie in one surface and cover the room between them, the sheen is one band round the
head. The head and body are the stand-ins of hair_fixtures.py."""
import numpy as np
import pytest

import hair_fixtures as HF
from mkmmd.core import families
from mkmmd.model import part as PT
from mkmmd.model.parts import hair, hair_locks as HL, hair_paint as HP, hair_tex
from mkmmd.model.parts.hair_fit import HeadFit, unit
from mkmmd.model.parts.hair_geo import arclen
from mkmmd.model.parts.hair_rig import Rig

PART = [[30, 40], [50, 30], [100, 20], [150, 14]]
LIFT = [[0.0, 0.0], [0.025, 0.014], [0.09, 0.008]]
LOCKS = {
    "sweep": {"family": "bangs", "count": 6, "layer": 2, "lift": LIFT, "free": 0.04, "chain": {"bones": 2, "every": 2},
              "guides": [{"root": ["part", 0.0], "to": [-10, 75], "length": 0.04},
                         {"root": ["part", 0.5], "to": [-80, 90], "length": 0.12}]},
    "fall": {"family": "side", "count": 5, "layer": 2, "lift": LIFT, "swell": 0.5,
             "guides": [{"root": ["part", 0.0], "to": [60, 82], "length": 0.10},
                        {"root": ["part", 1.0], "to": [130, 100], "length": 0.20}]},
    "fore": {"family": "bangs", "count": 4, "layer": 1,
             "guides": [{"root": ["hairline", 20], "to": [0, 62], "length": 0.02},
                        {"root": ["hairline", -30], "to": [-50, 80], "length": 0.06}]},
}


def spec(locks=None, partings=None):
    off = {"enabled": False}                             # the head hair alone: the other slices have their own tests
    return {"hair": {"style": "locks", "partings": partings or {"part": PART}, "locks": LOCKS if locks is None else locks,
                     "braids": off, "ears": off, "tails": off}}


@pytest.fixture(scope="module")
def planned(tmp_path_factory):
    ctx = HF.make_ctx(tmp_path_factory.mktemp("locks"), spec())
    hh = HL.LockHair(ctx, HeadFit(ctx.parts["head"].info), ctx.cfg, Rig(), hair_tex.palette(ctx.get("colors.hair")))
    return hh, {name: locks for name, _, locks in hh.plan()}


def test_lift_is_spread_over_the_way_or_told_in_metres_from_the_root():
    spread = HL.profile([-0.01, 0.0], "lift")
    assert np.allclose(spread(np.array([0.0, 0.1, 0.2]), 0.2), [-0.01, -0.005, 0.0])
    metres = HL.profile([[0.0, 0.0], [0.02, 0.01]], "lift")
    assert np.allclose(metres(np.array([0.0, 0.01, 0.02, 0.3]), 0.5), [0.0, 0.005, 0.01, 0.01])   # level after the last
    with pytest.raises(HL.HairdoError, match="lift"):
        HL.profile([[0.02, 0.0], [0.01, 0.01]], "lift")


def test_hair_combed_from_a_parting_is_rooted_on_it_and_leaves_it_across(planned):
    hh, groups = planned
    line, _ = hh.partings["part"]
    c = hh.skull.center
    for lk in groups["sweep"] + groups["fall"]:
        root = unit(lk.path[0] - c)
        assert np.degrees(np.arccos(np.clip(line @ root, -1.0, 1.0)).min()) < 0.6, lk.name
    for name, sign in (("sweep", -1.0), ("fall", 1.0)):                  # the two sides part: one to her right, one left
        for lk in groups[name][2:4]:                                     # locks in the middle of the group's stretch
            root = unit(lk.path[0] - c)
            k = int(np.argmax(line @ root))
            along = unit(line[min(k + 1, len(line) - 1)] - line[max(k - 1, 0)])
            go = unit(lk.path[int(np.searchsorted(arclen(lk.path), 0.02))] - lk.path[0])
            go = unit(go - root * (go @ root))
            assert abs(go @ along) < 0.4, (lk.name, go @ along)           # across the parting, not along it
            assert sign * np.cross(along, go) @ root < 0.0, lk.name


def test_a_groups_locks_lie_in_one_surface_and_leave_no_room_between_them(planned):
    hh, groups = planned
    c = hh.skull.center
    locks = groups["sweep"]
    first = locks[2].extra["lie"]
    for lk in locks[1:5]:
        st = hh.clump(lk)                                                 # where it lies with the others: past its root,
        end = min(lk.extra["s_leave"] - 0.04, st.length - 1.6 * lk.extra["free"])    # before it leaves or hangs free
        on = (st.arc > 0.03) & (st.arc < end) & (st.ring >= 0)
        assert on.sum() > 20
        D = unit(st.verts[on] - c)
        want = hh.skull.radius(D) + hh.lying(D, hh.away(D, first["line"]), first)
        # told by ONE lock's surface, every lock's points are on it: within the hair's breadth that says who is on top
        assert np.abs(np.linalg.norm(st.verts[on] - c, axis=1) - want).max() < 3 * HL.HAIR + 4e-4, lk.name
    for a, b in zip(locks[:-1], locks[1:]):
        sa, sb = hh.clump(a), hh.clump(b)

        def half(st, at):                                                 # half its width at the ring nearest `at`
            k = int(np.argmin(np.linalg.norm(st.ctr - at, axis=1)))
            ring = st.verts[st.ring == st.ring[k]]
            return 0.5 * np.linalg.norm(ring.max(0) - ring.min(0))
        for s_at in np.linspace(0.02, a.extra["s_leave"], 6):
            p = a.path[int(np.searchsorted(arclen(a.path), s_at))]
            gap = np.linalg.norm(b.path - p, axis=1)
            q = b.path[int(gap.argmin())]
            assert half(sa, p) + half(sb, q) > gap.min(), (a.name, round(float(s_at), 3))
    st = hh.clump(locks[3])
    assert np.linalg.norm(np.ptp(st.verts[:st.R], axis=0)) < 0.6 * locks[3].width     # a fan: narrow where its roots crowd


def test_a_root_lies_under_the_scalp_cap(planned):
    hh, groups = planned
    c = hh.skull.center
    for lk in [g[0] for g in groups.values()]:
        d = unit(lk.path[0] - c)
        th, ph = HL.angles(d)
        off = np.linalg.norm(lk.path[0] - c) - hh.skull.radius(d)
        assert HL.MIN_OFF - 1e-6 <= off < float(hh.cap_off(th, ph)), lk.name


def test_the_part_is_built_and_rigged(tmp_path):
    ctx = HF.make_ctx(tmp_path, spec())
    part = hair.build(ctx)
    PT.check(part)
    names = {b.name for b in part.bones}
    assert {"前髪1_1", "前髪1_2"} <= names and not any(n.startswith("横髪") for n in names)   # only `sweep` has chains
    mesh = part.meshes[0]
    assert mesh.uv.min() >= 0.0 and mesh.uv.max() <= 1.0
    fit = HeadFit(ctx.parts["head"].info)
    r = np.linalg.norm(mesh.verts - fit.skull.center, axis=1) - fit.skull.radius(unit(mesh.verts - fit.skull.center))
    assert r[mesh.verts[:, 2] > fit.skull.center[2]].min() > 0.0          # no hair inside the head


def test_a_lock_can_be_a_plait_on_a_chain_of_its_own(tmp_path):
    braid = {"family": "braid", "count": 1, "width": 0.040, "lift": [-0.012, -0.006], "tuck": 0.0,
             "guides": [{"root": [-112, 96], "to": [-96, 116], "length": 0.40}],
             "plait": {"enabled": True, "pitch": 0.04}, "chain": {"bones": 6, "radius": 0.012}}
    part = hair.build(HF.make_ctx(tmp_path, spec({**LOCKS, "braid": braid})))
    PT.check(part)
    assert {"髪", "三つ編", "髪リボン"} <= {m.name for m in part.materials}
    chain = [b.name for b in part.bones if b.name.startswith("三つ編")]
    assert chain == [f"三つ編右1_{k}" for k in range(1, 7)] and families.classify(chain[0]) == "braid"
    mesh = next(m for m in part.meshes if m.name == "hair_braid")
    z, x = mesh.verts[:, 2], mesh.verts[:, 0]
    assert x.max() < 0.0 and 0.33 < z.max() - z.min() < 0.42              # on her right, from under her ear down
    low = mesh.verts[(z > z.min() + 0.08) & (z < z.min() + 0.18)]         # where it hangs straight, above its tuft
    turn = np.radians(np.arange(0.0, 180.0, 15.0))
    across = max(np.ptp(low[:, 0] * np.cos(a) + low[:, 1] * np.sin(a)) for a in turn)
    assert 0.030 < across < 0.050                                         # a plait as wide as the lock was to be
    assert set(mesh.weights) <= {"頭", *chain} and mesh.weights[chain[-1]].max() > 0.9
    short = {**braid, "guides": [{"root": [-112, 96], "to": [-96, 116], "length": 0.10}]}
    with pytest.raises(ValueError, match="too short to plait"):
        hair.build(HF.make_ctx(tmp_path, spec({"braid": short})))


def test_a_hairdo_that_cannot_be_laid_says_what_is_wrong(tmp_path):
    one = [{"root": [0, 20], "to": [0, 80]}]
    for locks, partings, said in (({}, None, "needs groups of locks"),
                                  ({"a": {"guides": [{"root": ["seam", 0.5], "to": [0, 80]}]}}, None, "no parting 'seam'"),
                                  ({"a": {"count": 1, "guides": one}}, None, "needs a `width`"),
                                  ({"a": {"family": "mane", "guides": one}}, None, "family"),
                                  (LOCKS, {"hairline": [[0, 30], [20, 30]]}, "head's own line")):
        with pytest.raises(ValueError, match=said):
            hair.build(HF.make_ctx(tmp_path, spec(locks, partings)))


def test_the_sheen_is_one_band_round_the_head():
    look = {**HP.PAINT, "strands": 0.0, "lines": 0}
    pal = hair_tex.palette({})
    sheen = HP.Sheen(1.40, 0.018, look["tooth"], look["glow"], np.random.default_rng(3))
    band = sheen.on(np.array([1.40, 1.48, 1.32]), np.zeros(3), 0.0)[0]
    assert band[0] > 0.9 and band[1] == 0.0 and band[2] == 0.0
    seam = [sheen.on(np.full(40, 1.40) + np.linspace(-0.02, 0.02, 40), np.full(40, a), 0.0) for a in (np.pi, -np.pi)]
    assert all(np.allclose(x, y) for x, y in zip(*seam))                  # it closes behind her head
    edge = sheen.on(np.full(400, 1.4054), np.linspace(0.0, 1.0, 400), 0.0)[0]
    assert edge.max() - edge.min() > 0.4                                  # toothed: its upper edge goes up and down
    z = np.broadcast_to(np.linspace(1.43, 1.37, 64)[:, None], (64, 32))

    def cell(a0, a1):                                                     # a lock lying across the band, a0..a1 round
        around = np.broadcast_to(np.linspace(a0, a1, 32)[None, :], (64, 32))
        return HP.paint_lock(HP.Cell(0, 0, 32, 64, 256), z, around, np.linspace(0.05, 0.15, 64), np.zeros(64), pal, look,
                             np.random.default_rng(1), 1.45, 1.20, sheen)
    left, right = cell(0.10, 0.20), cell(0.20, 0.30)
    assert np.abs(left[:, -1] - right[:, 0]).max() < 1e-9                 # two locks side by side carry one picture


def test_a_lock_can_end_in_another_colour_and_cross_a_band_of_shade():
    pal = hair_tex.palette({"base": "#2060c0", "shadow": "#102050", "tip": "#c020c0"})
    plain = {**HP.PAINT, "strands": 0.0, "lines": 0, "root": 0.0, "tip": 0.0, "ramp": [[0.0, "base"], [1.0, "base"]]}
    z = np.broadcast_to(np.linspace(1.40, 1.20, 100)[:, None], (100, 16))
    around, arc = np.zeros((100, 16)), np.linspace(0.0, 0.20, 100)

    def paint(look, **more):
        return HP.paint_lock(HP.Cell(0, 0, 16, 100, 256), z, around, arc, np.zeros(100), pal, look,
                             np.random.default_rng(1), 1.45, 1.15, None, **more)
    flat = paint(plain)
    tipped = paint({**plain, "tips": {"colour": "tip", "length": 0.05, "strength": 1.0}})
    assert np.allclose(tipped[:70], flat[:70]) and np.allclose(tipped[-1, 8], pal["tip"], atol=0.02)   # its last 5 cm
    shade = HP.Sheen(1.30, 0.02, 0.012, 0.0, np.random.default_rng(2))
    banded = paint(plain, bands=[(shade, pal["shadow"], 1.0)])
    assert np.allclose(banded[50, 8], pal["shadow"], atol=0.05) and np.allclose(banded[[5, 95]], flat[[5, 95]])
