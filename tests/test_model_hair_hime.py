"""bpy-free invariants of the lock hairdo's cut hair (hair_locks, hair_geo, hair_paint, hair_fit, hair_clear): rounded
ends, a level cut, bows tied round locks, a mass that keeps its thickness, a plain soft sheen that comes down round
the head, a hull kept thin under a cap's band and the drape a heavy mass lies on. The head and body are the stand-ins
of hair_fixtures.py."""
import numpy as np

import hair_fixtures as HF
from mkmmd.model import part as PT
from mkmmd.model.parts import hair, hair_clear as HC, hair_locks as HL, hair_paint as HP, hair_tex
from mkmmd.model.parts.hair_fit import HeadFit, Volume, dirs
from mkmmd.model.parts.hair_geo import ROUND_RINGS, arclen, sweep
from mkmmd.model.parts.hair_rig import Rig

PART = [[0, 36], [0, 18], [0, 2], [180, 16], [180, 34]]
FRINGE = {"family": "bangs", "count": 9, "layer": 1, "free": 0.015, "finger": 0.85, "tip": "round", "tip_len": 0.008,
          "cut": 1.30, "guides": [{"root": ["part", 0.3], "to": [-45, 82]}, {"root": ["part", 0.0], "to": [0, 80]},
                                  {"root": ["part", 0.3], "to": [45, 82]}]}
LOCK = {"family": "side", "count": 1, "width": 0.035, "thick": 0.5, "thick_max": 0.018, "finger": 1.0, "free": 0.2,
        "tip": "round", "tip_len": 0.035, "cut": 1.0, "mass": {"enabled": True},
        "guides": [{"root": ["hairline", 55], "to": [62, 80]}],
        "bow": {"enabled": True, "above": 0.05, "span": 0.055, "ribbon": "#3f6fd8", "shade": "#284696"},
        "chain": {"bones": 5, "every": 1, "radius": 0.008}}


def spec(locks):
    off = {"enabled": False}
    return {"hair": {"style": "locks", "partings": {"part": PART}, "locks": locks, "braids": off, "ears": off,
                     "tails": off}}


def planned(tmp_path, locks):
    ctx = HF.make_ctx(tmp_path, spec(locks))
    hh = HL.LockHair(ctx, HeadFit(ctx.parts["head"].info), ctx.cfg, Rig(), hair_tex.palette(ctx.get("colors.hair")))
    return hh, {name: got for name, _, got in hh.plan()}


def test_a_round_tip_is_round_however_long_the_lock():
    """A round end has rings of its own round it: a short rounded end on a long lock is a half ellipse, not a cone
    from the last ring (what one ring every centimetre left of a 6 mm end)."""
    P = np.stack([np.zeros(80), np.zeros(80), np.linspace(1.0, 0.2, 80)], -1)
    L, end = 0.8, 0.006
    st = sweep(P, np.array([0.0, -1.0, 0.0]), 0.03, 0.006, tip="round", tip_start=1.0 - end / L, ring_step=0.01)
    arcs = st.arc[:st.rings * st.R:st.R]
    assert (arcs > L - end - 1e-9).sum() == ROUND_RINGS
    half = np.array([np.ptp(st.verts[i * st.R:(i + 1) * st.R, 0]) / 2 for i in range(st.rings)])
    t = np.clip((arcs - (L - end)) / end, 0.0, 1.0)
    assert np.allclose(half[-ROUND_RINGS:], 0.015 * np.sqrt(1.0 - t[-ROUND_RINGS:] ** 2), atol=2e-4)


def test_a_cut_group_ends_level_on_the_head_and_falling(tmp_path):
    """`cut` ends every lock at one height: the fringe while it lies on the forehead, a long lock where it falls."""
    hh, groups = planned(tmp_path, {"fringe": FRINGE, "lock": LOCK})
    for name, z in (("fringe", 1.30), ("lock", 1.0)):
        ends = np.array([lk.path[-1, 2] for lk in groups[name]])
        assert np.abs(ends - z).max() < 0.001
    assert all(lk.extra["s_leave"] >= arclen(lk.path)[-1] - 1e-6 for lk in groups["fringe"])   # cut on the head


def test_a_bow_is_tied_round_its_lock_on_its_chain(tmp_path):
    """The bow is `span` wide, in its ribbon's own material, tied `above` over the lock's end and riding its chain;
    the lock is gathered at the tie."""
    other = {**LOCK, "guides": [{"root": ["hairline", -55], "to": [-62, 80]}],
             "bow": {**LOCK["bow"], "ribbon": "#d8344a", "shade": "#8f1d33"}}
    part = hair.build(HF.make_ctx(tmp_path, spec({"left": LOCK, "right": other})))
    PT.check(part)
    assert {"結びリボン1", "結びリボン2"} <= {m.name for m in part.materials}
    bows = part.info["head"]["bows"]
    assert len(bows) == 2
    for b in bows:
        assert abs(b["span"] - 0.055) < 0.008 and b["above_end"] == 0.05
    mesh = next(m for m in part.meshes if m.name == "hair_bow")
    assert set(mesh.weights) & {b.name for b in part.bones if b.name.startswith("横髪")}


def test_a_mass_keeps_its_thickness_where_it_hangs(tmp_path):
    """A lock of a mass is as thick as itself from where it leaves the head; without `mass` it is paper-flat until
    its free end (FLAT)."""
    back = {"family": "back", "count": 5, "free": 0.05, "thick": 0.3, "thick_max": 0.02, "cut": 0.9,
            "guides": [{"root": ["part", 0.6], "to": [140, 112]}, {"root": ["part", 1.0], "to": [178, 114]}]}
    depth = {}
    for flag in (False, True):
        hh, groups = planned(tmp_path / str(flag), {"back": {**back, "mass": {"enabled": flag}}})
        lk = groups["back"][2]
        st = hh.clump(lk)
        arcs = st.arc[:st.rings * st.R:st.R]
        i = int(np.argmin(np.abs(arcs - (lk.extra["s_leave"] + 0.08))))
        ring = st.verts[i * st.R:(i + 1) * st.R] - st.ctr[i * st.R]
        depth[flag] = np.ptp(ring @ np.linalg.svd(ring - ring.mean(0))[2][1])     # across its thickness
    assert depth[True] > 4.0 * depth[False] and depth[True] > 0.005


def test_a_plain_sheen_is_a_soft_straight_band_that_comes_down_round_the_sides():
    """With `tooth` 0 the sheen is the same all round in front (no teeth, no wander), fades to nothing `width` off its
    middle, and is `fall` lower past `turn`."""
    sh = HP.Sheen(1.32, 0.008, 0.0, 0.0, np.random.default_rng(1), fall=0.06, turn=(48.0, 66.0))
    front = np.radians(np.linspace(-40.0, 40.0, 41))
    band = sh.on(np.full(41, 1.32), front, 0.0)[0]
    assert np.ptp(band) < 1e-9 and band.min() > 0.99
    assert sh.on(np.array([1.32 + 0.0081, 1.32 - 0.0081]), np.zeros(2), 0.0)[0].max() < 1e-6
    side = sh.on(np.array([1.26, 1.32]), np.radians([90.0, 90.0]), 0.0)[0]
    assert side[0] > 0.99 and side[1] < 1e-6
    toothed = HP.Sheen(1.32, 0.008, 0.012, 0.0, np.random.default_rng(1)).on(np.full(41, 1.325), front, 0.0)[0]
    assert np.ptp(toothed) > 0.2                                       # what a toothed band does


def test_the_hull_is_thin_under_a_caps_band_and_full_below_it(tmp_path):
    ctx = HF.make_ctx(tmp_path, spec({}))
    fit = HeadFit(ctx.parts["head"].info)
    vol = Volume(fit.skull, top=0.012, front=0.02, side=0.03, back=0.026, band=(1.335, 0.02, 0.016, 0.02))
    th, ph = np.radians(np.repeat([0.0, 90.0, 180.0], 60)), np.radians(np.tile(np.linspace(5.0, 120.0, 60), 3))
    z = fit.skull.point(dirs(th, ph))[:, 2]
    line = 1.335 - 0.02 * 0.5 * (1.0 - np.cos(th))
    v = vol(th, ph)
    assert v[z >= line].max() <= 0.016 + 1e-9
    full = Volume(fit.skull, top=0.012, front=0.02, side=0.03, back=0.026)(th, ph)
    assert np.allclose(v[z < line - 0.02], full[z < line - 0.02])


def test_a_drape_moves_hair_straight_out_and_bridges_hollows():
    """A drape's reach recedes no faster than its slope going down (a hollow under a bulge is bridged), and `clear`
    only ever moves points straight out from its axis, never round it, up, down or in."""
    a = np.linspace(0.0, 2.0 * np.pi, 400, endpoint=False)
    rings = []
    for z in np.arange(0.6, 1.2, 0.005):
        r = 0.12 + (0.04 if 0.95 < z < 1.0 else 0.0) + 0.004 * np.sin(9 * a)       # a bulge, and folds
        rings.append(np.stack([r * np.cos(a), r * np.sin(a), np.full_like(a, z)], -1))
    dr = HC.Drape(np.vstack(rings), (0.0, 0.0), 1.2)
    col = dr.reach(np.stack([np.full(60, 0.1), np.zeros(60), np.linspace(1.15, 0.62, 60)], -1))
    assert (np.diff(col) >= -HC.DRAPE["slope"] * (0.53 / 59) - 1e-9).all()
    X = np.random.default_rng(2).uniform([-0.2, -0.2, 0.65], [0.2, 0.2, 1.15], (300, 3))
    Y = dr.clear(X, 0.01)
    ang = lambda P: np.arctan2(P[:, 1], P[:, 0])   # noqa: E731
    assert np.allclose(Y[:, 2], X[:, 2]) and np.allclose(np.cos(ang(Y) - ang(X)), 1.0)
    assert (np.hypot(Y[:, 0], Y[:, 1]) >= np.hypot(X[:, 0], X[:, 1]) - 1e-12).all()
    assert (dr.over(Y) >= 0.01 - 1e-9).all()


def test_a_cut_can_vary_a_little_and_a_mass_thins_over_its_last_metres(tmp_path):
    """`jitter.cut` ends the locks a little apart (within it of the cut), and `mass.over` thins a mass only over
    its last metres: higher up it keeps its full thickness."""
    back = {"family": "back", "count": 6, "free": 0.05, "thick": 0.3, "thick_max": 0.02, "cut": 0.9,
            "jitter": {"cut": 0.003}, "mass": {"enabled": True, "thin": 0.5, "over": 0.10},
            "guides": [{"root": ["part", 0.6], "to": [140, 112]}, {"root": ["part", 1.0], "to": [178, 114]}]}
    hh, groups = planned(tmp_path, {"back": back})
    ends = np.array([lk.path[-1, 2] for lk in groups["back"]])
    assert np.abs(ends - 0.9).max() <= 0.003 + 1e-6 and np.ptp(ends) > 0.0005
    lk = groups["back"][2]
    st = hh.clump(lk)
    arcs = st.arc[:st.rings * st.R:st.R]

    def depth(at):
        i = int(np.argmin(np.abs(arcs - at)))
        ring = st.verts[i * st.R:(i + 1) * st.R] - st.ctr[i * st.R]
        return np.ptp(ring @ np.linalg.svd(ring - ring.mean(0))[2][1])
    high, mid, low = depth(lk.extra["s_leave"] + 0.06), depth(st.length - 0.12), depth(st.length - 0.02)
    assert abs(high - mid) < 0.15 * high and low < 0.75 * mid
