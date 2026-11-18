"""bpy-free invariants of the lock hairdo (hair_locks, hair_paint): hair combed from a parting is rooted on it and
leaves it across, a group's locks lie in one surface and cover the room between them, hair gathered to a tie ends in
it and a tail hangs from it, the sheen is one band round the head. The head and body are the stand-ins of
hair_fixtures.py."""
import numpy as np
import pytest

import hair_fixtures as HF
from mkmmd.core import families
from mkmmd.model import part as PT
from mkmmd.model.parts import hair, hair_braids_tex as HBT, hair_locks as HL, hair_paint as HP, hair_tex
from mkmmd.model.parts.hair_fit import HeadFit, Volume, unit
from mkmmd.model.parts.hair_geo import arclen, frames, solid_normals, sweep
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


def spec(locks=None, partings=None, ties=None):
    off = {"enabled": False}                             # the head hair alone: the other slices have their own tests
    return {"hair": {"style": "locks", "partings": partings or {"part": PART}, "locks": LOCKS if locks is None else locks,
                     "ties": ties or {}, "braids": off, "ears": off, "tails": off}}


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


def test_hair_gathered_to_a_tie_ends_in_it_and_a_tail_hangs_from_it(tmp_path):
    locks = {"drawn": {"count": 7, "layer": 1, "lift": LIFT,
                       "guides": [{"root": ["part", 0.0], "via": [[100, 80], [170, 100]], "to": "knot"},
                                  {"root": ["part", 1.0], "to": "knot"},
                                  {"root": ["part", 0.5], "via": [[-80, 50]], "to": "knot"}]},
             "tail": {"count": 3, "width": 0.03, "chain": {"bones": 3},
                      "guides": [{"root": "knot", "length": 0.25}, {"root": "knot", "out": [180, 150], "length": 0.20}]}}
    ctx = HF.make_ctx(tmp_path, spec(locks, ties={"knot": {"at": [-130, 100]}}))
    hh = HL.LockHair(ctx, HeadFit(ctx.parts["head"].info), ctx.cfg, Rig(), hair_tex.palette(ctx.get("colors.hair")))
    groups = {name: got for name, _, got in hh.plan()}
    knot = hh.on_hull(hh.ties["knot"]["dir"][None])[0]
    c = hh.skull.center
    for lk in groups["drawn"]:                                            # it lies on the head all its way into the tie
        assert np.linalg.norm(lk.path[-1] - knot) < 0.035 and not lk.chain and lk.extra["free"] == 0.0
        head = lk.path[arclen(lk.path) < lk.extra["s_leave"] - 0.01]
        over = np.linalg.norm(head - c, axis=1) - np.linalg.norm(hh.on_hull(unit(head - c)) - c, axis=1)
        assert -0.02 < over.min() and over.max() < 0.03
    tails = groups["tail"]
    for lk in tails:                                                      # it starts in the tie and hangs on a chain
        assert np.linalg.norm(lk.path[0] - knot) < 0.02 and lk.path[0, 2] - lk.path[-1, 2] > 0.15 and lk.chain
    first, last = (unit(lk.path[3] - lk.path[0]) for lk in (tails[0], tails[-1]))
    assert first @ last < 0.99 and last[1] > first[1]                     # each the way its guide's `out` points
    part = hair.build(HF.make_ctx(tmp_path / "built", spec(locks, ties={"knot": {"at": [-130, 100]}})))
    PT.check(part)


def test_a_plaits_crossings_are_as_close_as_it_is_wide(tmp_path):
    def crossings(width):                                                 # of 0.42 m of plait over a tuft as long as it is wide
        braid = {"family": "braid", "count": 1, "width": width, "jitter": {"length": 0.0},
                 "guides": [{"root": "knot", "length": 0.42 + width}], "plait": {"enabled": True, "tuft_len": 0.034}}
        ctx = HF.make_ctx(tmp_path / f"w{width}", spec({"braid": braid}, ties={"knot": {"at": [-130, 100]}}))
        hh = HL.LockHair(ctx, HeadFit(ctx.parts["head"].info), ctx.cfg, Rig(), hair_tex.palette(ctx.get("colors.hair")))
        (_, g, (lk,)), = hh.plan()
        return hh.plait(lk, g, None)[1]["lobes"]
    assert crossings(0.03) == pytest.approx(2 * crossings(0.06), abs=1)   # the same length: twice as many
    assert 0.2 * 0.06 < 0.42 / crossings(0.06) < 0.4 * 0.06               # a crossing every third of its width or so


def test_a_plait_of_hair_that_ends_in_another_colour_ends_in_it():
    pal = hair_tex.palette({"base": "#2060c0", "shadow": "#102050", "deep": "#081030", "light": "#4080e0",
                            "highlight": "#80b0ff", "highlight_core": "#c0d8ff", "rim": "#6090f0", "tip": "#c020c0"})
    weave = {"K": 30.0, "phase": 0.0, "dz": 0.0}
    plain = HBT.braid_atlas(pal, np.random.default_rng(1), weave).astype(float)
    ended = HBT.braid_atlas(pal, np.random.default_rng(1), {**weave, "tips": (pal["tip"], 0.3, 0.9)}).astype(float)
    rows, tile = hair_tex.HEAD_ROWS, hair_tex.TILE_PX                     # a strand's tile: the strand runs down it

    def red(img, part):
        return img[part, :tile, 0].mean() / img[part, :tile, 2].mean()
    above, end, tuft = slice(0, int(0.6 * rows)), slice(int(0.9 * rows), int(0.96 * rows)), slice(rows, None)
    assert np.array_equal(plain[above, :tile], ended[above, :tile])       # its own colour down to its last third
    assert red(ended, end) > 2.0 * red(plain, end) and red(ended, tuft) > 2.0 * red(plain, tuft)


def test_the_hull_comes_in_to_the_nape_behind_the_ears_only(planned):
    hh, _ = planned
    loose = Volume(hh.skull, side=0.04, back=0.04, taper=0.0)
    drawn = Volume(hh.skull, side=0.04, back=0.04, taper=0.0, nape=0.012, hairline=hh.fit.cap_phi)
    back, side = np.pi, 0.5 * np.pi
    low = float(hh.fit.cap_phi(back))
    assert drawn(back, low) == pytest.approx(0.012) and loose(back, low) == pytest.approx(0.04)
    assert drawn(back, low - np.radians(35.0)) == loose(back, low - np.radians(35.0))     # the back of her head keeps its volume
    assert drawn(side, float(hh.fit.cap_phi(side))) == loose(side, float(hh.fit.cap_phi(side)))   # and so do her ears' sides


def test_a_hairdo_that_cannot_be_laid_says_what_is_wrong(tmp_path):
    one = [{"root": [0, 20], "to": [0, 80]}]
    for locks, partings, said in (({}, None, "needs groups of locks"),
                                  ({"a": {"guides": [{"root": ["seam", 0.5], "to": [0, 80]}]}}, None, "no parting 'seam'"),
                                  ({"a": {"count": 1, "guides": one}}, None, "needs a `width`"),
                                  ({"a": {"family": "mane", "guides": one}}, None, "family"),
                                  (LOCKS, {"hairline": [[0, 30], [20, 30]]}, "head's own line"),
                                  ({"a": {"guides": [{"root": [0, 20], "to": "bun"}]}}, None, "no tie 'bun'"),
                                  ({"a": {"guides": [{"root": [0, 20], "to": "knot", "length": 0.1}]}}, None, "ends in it"),
                                  ({"a": {"width": 0.02, "guides": [{"root": "knot", "to": [0, 80]}]}}, None, "no `to`"),
                                  ({"a": {"guides": [{"root": [0, 20], "to": "knot"}, *one]}}, None, "one tie")):
        with pytest.raises(ValueError, match=said):
            hair.build(HF.make_ctx(tmp_path, spec(locks, partings, {"knot": {"at": [180, 100]}})))


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


def test_a_flipped_end_turns_out_and_up(tmp_path):
    def tips(flip):
        g = {"family": "back", "count": 3, "layer": 0, "tuck": 0.0, "free": 0.04, "cut": 1.17, "flip": flip,
             "guides": [{"root": ["part", 1.0], "to": [140, 104]}, {"root": ["part", 1.0], "to": [-140, 104]}]}
        ctx = HF.make_ctx(tmp_path, spec({"nape": g}))
        hh = HL.LockHair(ctx, HeadFit(ctx.parts["head"].info), ctx.cfg, Rig(), hair_tex.palette(ctx.get("colors.hair")))
        return [lk.path for _, _, locks in hh.plan() for lk in locks]
    out = tips({"len": 0.04, "out": 0.02})
    up = tips({"len": 0.04, "out": 0.02, "up": 0.02})
    for a, b in zip(out, up):
        assert b[-1, 2] - a[-1, 2] == pytest.approx(0.02, abs=0.004)       # the end rises by `up`
        assert a[-1, 2] < a[-2, 2] and b[-1, 2] > b[-2, 2]                 # out and down; out and up
        above = b[b[:, 2] > 1.25]                                          # above its last `len` it lies as it did
        assert np.linalg.norm(above[:, None] - a[None], axis=2).min(1).max() < 0.0015


def test_staggered_locks_end_apart_and_each_flips_its_own_way(tmp_path):
    def paths(**more):
        g = {"family": "back", "count": 5, "layer": 0, "tuck": 0.0, "free": 0.04, "cut": 1.17,
             "flip": {"len": 0.04, "out": 0.03, "up": 0.015}, **more,
             "guides": [{"root": ["part", 1.0], "to": [140, 104]}, {"root": ["part", 1.0], "to": [-140, 104]}]}
        ctx = HF.make_ctx(tmp_path, spec({"nape": g}))
        hh = HL.LockHair(ctx, HeadFit(ctx.parts["head"].info), ctx.cfg, Rig(), hair_tex.palette(ctx.get("colors.hair")))
        return [lk.path for _, _, locks in hh.plan() for lk in locks]
    even = paths()
    staggered = paths(stagger=[0.0, 0.012])
    for i, (a, b) in enumerate(zip(even, staggered)):
        assert b[-1, 2] == pytest.approx(a[-1, 2] - (0.012 if i % 2 else 0.0), abs=0.002)   # every other one 12 mm lower
    rise = [p[-1, 2] - p[-6, 2] for p in paths(jitter={"flip": 0.6})]
    assert np.ptp(rise) > 0.006                                                # the ends turn up each their own way


BACK = {"family": "back", "count": 3, "layer": 1, "width": 0.04, "free": 0.04, "tuck": 0.0, "cup": 0.0,
        "lift": [[0.0, 0.0], [0.05, 0.003]], "thick_max": 0.02,
        "guides": [{"root": ["part", 1.0], "to": [140, 90], "length": 0.08},
                   {"root": ["part", 1.0], "to": [-140, 90], "length": 0.08}]}


def laid(tmp_path, locks, **hair_tables):
    """(LockHair, {group: [Lock]}) of `locks` laid on the stand-in head; `hair_tables` more [hair] tables."""
    s = spec(locks)
    s["hair"].update(hair_tables)
    ctx = HF.make_ctx(tmp_path, s)
    hh = HL.LockHair(ctx, HeadFit(ctx.parts["head"].info), ctx.cfg, Rig(), hair_tex.palette(ctx.get("colors.hair")))
    return hh, {name: got for name, _, got in hh.plan()}


def ring_at(st, at):
    """(the ring of a swept lock nearest `at` metres from its root, its q round it, the way across it, through it)."""
    n, R = st.rings, st.R
    k = int(np.argmin(np.abs(st.arc[:n * R:R] - at)))
    ring, q = st.verts[k * R:(k + 1) * R], st.q[:R]
    across = unit(ring[np.argmax(q)] - ring[np.argmin(q)])
    t = unit(st.ctr[min(k + 1, n - 1) * R] - st.ctr[max(k - 1, 0) * R])
    return ring, q, across, unit(np.cross(t, across))


def test_a_solid_lock_is_a_round_volume_on_the_head_and_where_it_hangs(tmp_path):
    hh, thin = laid(tmp_path / "thin", {"back": BACK})
    lk = thin["back"][1]
    ring, _, across, through = ring_at(hh.clump(lk), 0.5 * lk.extra["s_leave"])
    assert np.ptp(ring @ through) < 0.06 * np.ptp(ring @ across)            # by default paper-thin where it lies
    hh, solid = laid(tmp_path / "solid", {"back": {**BACK, "solid": 0.32}})
    lk = solid["back"][1]
    st = hh.clump(lk)
    for at in (0.5 * lk.extra["s_leave"], st.length - 0.5 * lk.extra["free"]):     # on the head, hanging free
        ring, q, across, through = ring_at(st, at)
        assert np.ptp(ring @ through) / np.ptp(ring @ across) == pytest.approx(0.32, abs=0.03)
        for e, (a, b) in ((int(np.argmin(q)), (1, -1)), (int(np.argmax(q)), (-1, 1))):   # each edge is round,
            u, v = unit(ring[(e + a) % len(q)] - ring[e]), unit(ring[(e + b) % len(q)] - ring[e])   # not a knife
            assert np.degrees(np.arccos(np.clip(u @ v, -1.0, 1.0))) > 75.0
    n = solid_normals(st, hh.proxy_normals(st.verts), 0.88)
    edges = (np.abs(st.q) == 1.0) & (st.ring >= 0) & (st.arc > 0.2 * st.length) & (st.arc < 0.8 * st.length)
    across = unit(st.verts[edges] - st.ctr[edges])
    assert (np.einsum("ij,ij->i", n[edges], across) > 0.9).all()           # its edges shade by their own round form


def test_a_solid_lock_tucked_under_keeps_its_inner_face_off_the_skin(tmp_path):
    tucked = {**BACK, "solid": 0.32, "lift": [-0.012, 0.0]}
    hh, got = laid(tmp_path, {"back": tucked}, volume={"top": 0.008, "front": 0.004, "side": 0.010, "back": 0.010})
    c = hh.skull.center
    for lk in got["back"]:
        V = hh.clump(lk).verts
        assert (np.linalg.norm(V - c, axis=1) - hh.skull.radius(unit(V - c))).min() > 0.0, lk.name


def test_a_curl_bends_the_lock_about_its_width_and_never_twists_it():
    a = np.radians(np.linspace(0.0, 90.0, 40))                             # down, then out (+y): a quarter circle
    P = np.stack([np.zeros_like(a), 0.05 * (1.0 - np.cos(a)), 1.30 - 0.05 * np.sin(a)], 1)
    turn = np.radians(np.linspace(0.0, 60.0, 40))                          # the head's facing turns on the way
    N = np.stack([np.sin(turn), np.cos(turn), np.zeros_like(turn)], 1)
    _, B, _ = frames(P, N)
    assert abs(B[-1] @ B[0]) < 0.6                                          # faced the head's way, the end twists
    _, B, _ = frames(P, HL.curled(P, N, 0.0))
    assert (np.abs(B @ B[0]) > 0.999).all()                                 # curled, its width keeps its way
    st = sweep(P, HL.curled(P, N, 0.0), 0.03, 0.009, solid=0.3, section="round", k_outer=7)
    _, _, across, _ = ring_at(st, st.length - 0.006)
    assert abs(across @ B[0]) > 0.99


def test_a_flipped_lock_remembers_its_curl(tmp_path):
    hh, got = laid(tmp_path, {"back": {**BACK, "cut": 1.20, "flip": {"len": 0.05, "out": 0.04, "up": 0.02}}})
    assert all(lk.extra["curl"] == pytest.approx(0.05) for lk in got["back"])
    _, plain = laid(tmp_path / "plain", {"back": BACK})
    assert all(lk.extra["curl"] == 0.0 for lk in plain["back"])


def test_a_lock_lying_on_another_shades_it_deep_under_its_end(tmp_path):
    over, under = {**BACK, "layer": 2, "cut": 1.25}, {**BACK, "layer": 1, "cut": 1.18}
    hh, got = laid(tmp_path, {"over": over, "under": under})
    locks = got["under"] + got["over"]
    cells, _ = hh.atlas.cells([(lk.width, float(arclen(lk.path)[-1])) for lk in locks], [HL.CAP_PX])
    acc = HL.MeshAccum("hair_head", ["髪"])
    for lk, cell in zip(locks, cells):
        lk.extra["cell"] = cell
        hh.emit(acc, lk)
    deep = hh.occlusion(locks, 0.016)
    top, low = got["over"][1], got["under"][1]
    arcs, q, occ = deep[locks.index(low)]
    z = low.extra["rings"][3][:, len(q) // 2, 2]                            # the middle of its outer side, down it
    end = top.path[-1, 2]
    assert occ[(z > end - 0.012) & (z < end + 0.002), len(q) // 2].max() > 0.2     # just under the upper lock's end
    assert occ[np.abs(z - (end - 0.04)) < 0.006, len(q) // 2].max() < 0.1         # well below it, out in the open
    arcs, q, occ = deep[locks.index(top)]
    free = int(np.argmin(np.abs(arcs - (arcs[-1] - 0.5 * top.extra["free"]))))
    assert occ[free, len(q) // 2] < 0.05                                    # its own ridge, nothing over it
    pal = hair_tex.palette({})
    look = {**HP.PAINT, "crease": {"strength": 1.0, "reach": 0.016}}
    rgb = HP.paint_lock(HP.Cell(0, 0, 8, 16, 64), np.full((16, 8), 1.30), np.zeros((16, 8)), np.linspace(0, 0.1, 16),
                        np.zeros(16), pal, look, np.random.default_rng(1), 1.45, 1.15, crease=np.ones((16, 8)))
    assert np.allclose(rgb, pal["deep"])                                    # the crease's colour is the deep one


def test_the_locks_break_the_sheen_up():
    pal = hair_tex.palette({})
    z = np.broadcast_to(np.linspace(1.43, 1.37, 61)[:, None], (61, 21))
    around = np.zeros((61, 21))

    def lock(rng, **more):
        look = {**HP.PAINT, "strands": 0.0, "lines": 0, "edge": 0.0, "rim": 0.0, "root": 0.0, "tip": 0.0,
                "form": 0.0, "tooth": 0.0, **more}
        sheen = HP.Sheen(1.40, 0.02, 0.0, 0.0, np.random.default_rng(5))
        return HP.paint_lock(HP.Cell(0, 0, 21, 61, 128), z, around, np.linspace(0.05, 0.11, 61), np.zeros(61), pal,
                             look, np.random.default_rng(rng), 1.45, 1.20, sheen).sum(2)
    whole, broken = lock(1), lock(1, sheen_break=0.8)
    assert np.allclose(whole[30, 0], whole[30, 10])                         # one even band across the lock
    assert broken[30, 10] - broken[30, 0] > 0.5 * (whole[30, 10] - whole[0, 10])   # its piece fades to its edges
    rows = [int(np.argmax(lock(seed, sheen_shift=0.006)[:, 10])) for seed in (1, 2, 3, 4)]
    assert np.ptp(rows) >= 3                                                # each lock's piece a little up or down
