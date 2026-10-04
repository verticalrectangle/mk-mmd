import json
from types import SimpleNamespace

import numpy as np
import pytest

from mkmmd.core import families
from mkmmd.solvers import geom, strands

needs_numba = pytest.mark.skipif(not strands.HAVE_NUMBA, reason="numba is not installed")
TOP = 1.5                                           # height of the anchor (m)


# ---------------------------------------------------------------- small scenes
def straight_rig(n=5, seg=0.1, chains=1, family="hair"):
    """`chains` strands of n bones hanging straight down from anchors 0.15 m apart along x, no collision bodies."""
    bones = {}
    spec = []
    for c in range(chains):
        x = 0.15 * c
        bones[f"anchor{c}"] = {"head": [x, 0, TOP]}
        names = [f"c{c}b{i}" for i in range(n)]
        for i, name in enumerate(names):
            bones[name] = {"head": [x, 0, TOP - i * seg]}
        spec.append({"family": family, "root": names[0], "anchor": f"anchor{c}", "bones": names,
                     "parents": list(range(-1, n - 1)), "ends": [[x, 0, TOP - (i + 1) * seg] for i in range(n)],
                     "body_radius": [0.02] * n})
    return {"bones": bones, "chains": spec, "bodies": []}


def two_family_rig():
    """One hair bone and one tail bone, each hanging from its own anchor."""
    rig = straight_rig(1)
    rig["bones"]["t0"] = {"head": [0.3, 0, TOP]}
    rig["bones"]["anchor1"] = {"head": [0.3, 0, TOP]}
    rig["chains"].append({"family": "tail", "root": "t0", "anchor": "anchor1", "bones": ["t0"], "parents": [-1],
                          "ends": [[0.3, 0, TOP - 0.1]], "body_radius": [0.02]})
    return rig


def rot_z(deg):
    h = np.radians(deg) / 2
    return np.array([np.cos(h), 0, 0, np.sin(h)])


def scene(rig, F, shapes=None, poses=None):
    """Inputs for simulate: every chain hangs from the same world-fixed anchor source unless `poses` {source:
    (pos (F,3), quat (F,4))} moves some. Shapes already added to `shapes` keep their sources at the identity."""
    chains = geom.Chains(rig, None)
    shapes = geom.Shapes() if shapes is None else shapes
    anchor = shapes.src("world", "", "")
    S = len(shapes.sources)
    pos, quat = np.zeros((F, S, 3)), np.zeros((F, S, 4))
    quat[..., 0] = 1.0
    for j, (p, q) in (poses or {}).items():
        pos[:, j], quat[:, j] = p, q
    return dict(chains=chains, shapes=shapes, src_pos=pos, src_quat=quat, rest_R=np.tile(np.eye(3), (S, 1, 1)),
                rest_p=np.zeros((S, 3)), anchor_src=np.full(len(chains), anchor), rig=rig, F=F)


def run(sc, params=None, wind=None, engine="numpy", **kw):
    return strands.simulate(sc["chains"], sc["shapes"], sc["src_pos"], sc["src_quat"], sc["rest_R"], sc["rest_p"],
                            sc["anchor_src"], params, (10, 10 + sc["F"] - 1), sc["rig"], substeps=5, settle_s=0.5,
                            wind=wind, engine=engine, **kw)


def tip_offset(res):
    """Where the last particle ends up relative to straight down from the anchor, at the last frame."""
    return res.x[-1, 4] - np.array([0, 0, TOP - 0.5])


# ---------------------------------------------------------------- parameters
def test_every_family_has_defaults_and_overrides_merge():
    assert set(families.FAMILIES) <= set(strands.FAMILY_DEFAULTS)
    for p in strands.FAMILY_DEFAULTS.values():
        assert set(p) <= set(strands.PARAM_KEYS)
        assert {"sag", "drag", "zeta", "radius", "radius_max", "friction"} <= set(p)
    out = strands.resolve_params(["back_hair", "hair", "mystery"], {"back_hair": {"drag": 9.0}})
    assert out["back_hair"]["drag"] == 9.0 and out["back_hair"]["wind_drag"] == 9.0      # wind_drag follows drag
    assert out["back_hair"]["sag"] == strands.FAMILY_DEFAULTS["back_hair"]["sag"]
    assert out["mystery"] == strands.resolve_params(["other"])["other"]                  # unknown families: "other"
    assert strands.resolve_params(["hair"], {"hair": {"wind_drag": 2.0}})["hair"]["wind_drag"] == 2.0


@pytest.mark.parametrize("params", [{"back_hair": {"dragg": 1.0}}, {"backhair": {"drag": 1.0}},
                                    {"back_hair": {"zeta": -1.0}}, {"back_hair": {"radius": 0.0}}])
def test_misspelt_or_impossible_parameters_fail_loudly(params):
    with pytest.raises(ValueError):
        strands.resolve_params(["back_hair"], params)


# ---------------------------------------------------------------- wind
def test_air_is_constant_wind_plus_the_share_of_the_carrier_it_follows():
    carrier = np.linspace(0.0, 10.0, 31)[:, None] * np.array([[1.0, 0.0, 0.0]])       # 0 -> 10 m/s over one second
    X = np.array([[0.0, 0.0, 0.0], [5.0, -3.0, 2.0]])
    cabin = strands.Wind({"direction": [0, 2, 0], "speed": 3.0, "exposure": 0.3}, carrier, fps=30.0)
    assert cabin.velocity(0.5, X) == pytest.approx(np.array([[3.5, 3.0, 0.0]] * 2))             # 0.7 * 5 m/s + wind
    assert cabin.velocity(0.5 + 0.5 / 30, X[0]) == pytest.approx([0.7 * 5.1666667, 3.0, 0.0])    # linear between rows
    assert cabin.velocity(9.0, X[0]) == pytest.approx([7.0, 3.0, 0.0])                         # held after the last
    assert strands.Wind({"exposure": 0.0}, carrier, 30.0).velocity(1.0, X[0]) == pytest.approx([10.0, 0.0, 0.0])
    open_air = strands.Wind({"exposure": 1.0, "speed": 2.0, "direction": [0, 0, 1]}, carrier, 30.0)
    assert open_air.velocity(1.0, X[0]) == pytest.approx([0.0, 0.0, 2.0])
    assert strands.Wind({}).velocity(3.0, X) == pytest.approx(np.zeros((2, 3)))
    # the turbulence pattern drifts with the mean flow: the integral of constant + (1 - exposure) * carrier, no gust
    assert cabin.displacement_at([0.5, 1.0, 2.0]) == pytest.approx(
        np.array([[0.875, 1.5, 0.0], [3.5, 3.0, 0.0], [10.5, 6.0, 0.0]]))


def test_gusts_swell_along_the_wind_and_never_beyond_their_amplitude():
    w = strands.Wind({"direction": [0, 1, 0], "speed": 4.0, "gust": 2.0, "gust_period": 3.0, "seed": 5})
    t = np.linspace(0.0, 60.0, 601)
    u = np.array([w.velocity(s, [0, 0, 0]) for s in t])
    assert np.abs(u[:, [0, 2]]).max() < 1e-12                                  # only along direction
    assert u[:, 1].max() <= 6.0 + 1e-9 and u[:, 1].min() >= 2.0 - 1e-9         # speed +- amplitude
    assert u[:, 1].max() - u[:, 1].min() > 1.0                                 # and it does swell


def test_without_a_direction_gusts_blow_along_the_carriers_headwind():
    w = strands.Wind({"gust": 1.0, "exposure": 1.0, "seed": 2}, np.tile([8.0, 0.0, 0.0], (10, 1)), 30.0)
    u = np.array([w.velocity(s, [0, 0, 0]) for s in np.linspace(0, 20, 200)])
    assert np.abs(u[:, 1:]).max() < 1e-12                                       # on the carrier's axis
    assert np.ptp(u[:, 0]) > 0.5 and np.abs(u[:, 0]).max() <= 1.0 + 1e-9        # swelling by its amplitude


def test_turbulence_is_deterministic_per_seed_smooth_and_divergence_free():
    spec = {"speed": 2.0, "direction": [1, 0, 0], "gust": 1.0, "gust_period": 2.0, "turbulence": 1.5, "scale": 0.4,
            "seed": 11}
    w = strands.Wind(spec)
    X = np.random.default_rng(0).uniform(-1, 1, (60, 3))
    u = w.velocity(1.0, X)
    assert np.array_equal(u, strands.Wind(spec).velocity(1.0, X))
    assert not np.allclose(u, strands.Wind(dict(spec, seed=12)).velocity(1.0, X))
    # first-order smooth: a 10x smaller step changes the velocity 10x less (in time, then in space)
    big_t, small_t = (np.abs(w.velocity(1.0 + h, X) - u).max() for h in (1e-3, 1e-4))
    big_x, small_x = (np.abs(w.velocity(1.0, X + h) - u).max() for h in (1e-3, 1e-4))
    for big, small in ((big_t, small_t), (big_x, small_x)):
        assert 0 < big < 0.5 and small == pytest.approx(big / 10, rel=0.1)
    h, div = 1e-5, 0.0
    for i in range(3):
        e = np.eye(3)[i] * h
        div = div + (w.velocity(1.0, X + e)[:, i] - w.velocity(1.0, X - e)[:, i]) / (2 * h)
    assert np.abs(div).max() < 1e-6


def test_turbulence_has_the_requested_rms_speed():
    w = strands.Wind({"turbulence": 2.0, "scale": 0.5, "seed": 1})
    u = w.velocity(0.0, np.random.default_rng(3).uniform(-20, 20, (20000, 3)))
    assert np.sqrt((u ** 2).sum(1).mean()) == pytest.approx(2.0, rel=0.25)       # 16 random modes: a loose match


@pytest.mark.parametrize("spec", [{"gustt": 1.0}, {"exposure": 1.5}, {"direction": [0, 0, 0]}, {"turbulence": -1.0},
                                  {"gust": 1.0, "gust_period": 0.0}])
def test_wind_rejects_nonsense(spec):
    with pytest.raises(ValueError):
        strands.Wind(spec)


# ---------------------------------------------------------------- a strand in the air
def test_a_strand_hangs_straight_down_in_still_air():
    res = run(scene(straight_rig(), 21))
    assert np.abs(res.x[:, :, :2]).max() < 1e-9
    assert tip_offset(res)[2] == pytest.approx(0.0, abs=2e-3)            # a hair longer than at rest, under its weight
    assert res.pen.max() == 0.0 and all(w[2] == "-" for w in res.worst)


def test_side_wind_blows_the_strand_downwind_and_harder_wind_blows_it_further():
    angle = []
    for speed in (0.0, 1.0, 3.0, 8.0):
        res = run(scene(straight_rig(), 16), wind=strands.Wind({"direction": [1, 0, 0], "speed": speed}))
        tip = tip_offset(res)
        assert abs(tip[1]) < 1e-9
        angle.append(np.degrees(np.arctan2(tip[0], 0.5 - tip[2])))                # from the vertical, downwind
    assert angle[0] == pytest.approx(0.0, abs=1e-6)
    assert all(a < b for a, b in zip(angle, angle[1:])) and angle[1] > 1.0 and angle[-1] < 90.0
    res = run(scene(straight_rig(), 16), wind=strands.Wind({"direction": [0, -1, 0], "speed": 3.0}))
    assert tip_offset(res)[1] < -0.05 and abs(tip_offset(res)[0]) < 1e-9


def test_stiffer_families_deflect_less_and_wind_drag_sets_the_push():
    wind = strands.Wind({"direction": [1, 0, 0], "speed": 3.0})
    soft = tip_offset(run(scene(straight_rig(), 16), {"hair": dict(sag=(20.0, 80.0))}, wind))[0]
    stiff = tip_offset(run(scene(straight_rig(), 16), {"hair": dict(sag=(0.5, 1.0))}, wind))[0]
    gentle = tip_offset(run(scene(straight_rig(), 16), {"hair": dict(sag=(20.0, 80.0), wind_drag=1.0)}, wind))[0]
    assert 0 < stiff < 0.3 * soft and 0 < gentle < soft


def test_gusts_and_turbulence_move_a_strand_in_time_and_repeat_for_a_seed():
    spec = {"direction": [1, 0, 0], "speed": 1.0, "gust": 3.0, "gust_period": 0.8, "turbulence": 1.5, "seed": 3}
    a = run(scene(straight_rig(), 31), wind=strands.Wind(spec))
    b = run(scene(straight_rig(), 31), wind=strands.Wind(spec))
    c = run(scene(straight_rig(), 31), wind=strands.Wind(dict(spec, seed=4)))
    assert np.array_equal(a.x, b.x) and not np.allclose(a.x, c.x, atol=1e-3)
    assert a.x[:, 4, 0].std() > 0.01 and np.abs(a.x[:, 4, 1]).max() > 1e-3     # swings in time, off the wind axis too


def test_a_wind_that_blows_nothing_changes_nothing():
    sc = scene(straight_rig(), 16)
    assert np.array_equal(run(sc).x, run(sc, wind=strands.Wind({})).x)
    carried = strands.Wind({}, np.zeros((16, 3)))
    assert np.array_equal(run(sc).x, run(sc, wind=carried).x)


def test_riding_in_a_carrier_that_the_air_follows_is_the_same_as_standing_still():
    F, v = 31, np.array([0.0, 6.0, 0.0])
    t = np.arange(F) / 30.0
    still = scene(straight_rig(), F)
    moving = scene(straight_rig(), F, poses={0: (t[:, None] * v[None], np.tile([1.0, 0, 0, 0], (F, 1)))})
    lag = {}
    for exposure in (0.0, 0.3, 1.0):
        wind = strands.Wind({"exposure": exposure}, np.tile(v, (F, 1)), 30.0)
        res = run(moving, wind=wind)
        rel = res.x - moving["src_pos"][:, [0]]                                    # relative to the anchor
        lag[exposure] = rel[-1, 4, 1]
        if exposure == 0.0:
            assert np.allclose(rel, run(still).x, atol=1e-9)                       # Galilean invariance
    assert lag[0.0] == pytest.approx(0.0, abs=1e-9)
    assert lag[1.0] < lag[0.3] < lag[0.0] and lag[0.3] < -0.02                    # open air blows the hair back
    ground = run(moving)                                                          # no carrier given: still air
    assert (ground.x - moving["src_pos"][:, [0]])[-1, 4, 1] < -0.02


def test_a_collider_holds_the_strand_back_and_the_penetration_is_reported():
    shapes = geom.Shapes()
    shapes.add("sphere", ("world", "", ""), "ball", c=[0.13, 0.0, 1.25], R=0.1)
    wind = strands.Wind({"direction": [1, 0, 0], "speed": 4.0})
    free = run(scene(straight_rig(), 21), wind=wind)
    res = run(scene(straight_rig(), 21, shapes), wind=wind)
    assert res.x[-1, 3, 0] < free.x[-1, 3, 0] - 0.05                              # the ball is in the way
    assert 0.0 <= res.pen[:, 0].max() < 0.002 and res.labels == ["ball", "floor"]
    assert max(w[0] for w in res.worst) == res.pen[:, 0].max()
    assert all(np.linalg.norm(res.x[-1, i] - [0.13, 0, 1.25]) > 0.1 for i in (3, 4))      # outside the ball


def test_the_floor_holds_a_blown_strand_lowered_onto_it_and_it_lies_along_the_floor():
    F = 24
    drop = np.stack([0 * np.arange(F), 0 * np.arange(F), -0.15 * np.arange(F) / (F - 1)], 1)       # anchor sinks 15 cm
    poses = {0: (drop, np.tile([1.0, 0, 0, 0], (F, 1)))}
    wind = strands.Wind({"direction": [1, 0, 0], "speed": 3.0})        # a straight strand cannot fold alone
    free = run(scene(straight_rig(), F, poses=poses), wind=wind)
    shapes = geom.Shapes()
    shapes.floor_z = 1.15
    res = run(scene(straight_rig(), F, shapes, poses=poses), wind=wind)
    r = 0.5 * 0.02                                                         # particle radius: radius 0.5 x body 0.02
    assert free.x[:, :, 2].min() < 1.15 - 0.05                             # it would hang through the floor
    assert res.x[-1, 4, 2] == pytest.approx(1.15 + r, abs=3e-3)            # held up on it, the tip lying along it
    assert res.x[:, :, 2].min() > 1.15 + r - 0.002 and res.pen[:, 0].max() < 0.002
    assert np.abs(np.linalg.norm(res.x[:, 1:] - res.x[:, :-1], axis=2) / 0.1 - 1.0).max() < 1e-9      # lengths intact


def rendered_positions(sc, res):
    """Where the keyed bones put the particles: each bone's rest offset turned by its D, from the end of its parent."""
    chains = sc["chains"]
    seg = chains.end - chains.head
    out = np.zeros_like(res.x)
    for i in range(len(chains)):
        head = chains.head[i] if chains.parent[i] < 0 else out[:, chains.parent[i]]
        out[:, i] = head + np.einsum("fij,j->fi", res.D[:, i], seg[i])
    return out


@pytest.mark.parametrize("engine", ["numpy", pytest.param("numba", marks=needs_numba)])
def test_strands_stay_inextensible_in_a_gale_so_the_bones_render_the_particles(engine):
    rig = straight_rig(n=27, seg=0.034)                                          # a long lock of back hair
    shapes = geom.Shapes()
    wall = np.eye(4)
    wall[:3, 3] = [0.3, 0.0, 1.0]
    shapes.add("box", ("world", "", ""), "wall", M=wall, half=[0.02, 0.5, 0.8], rnd=0.005)
    sc = scene(rig, 25, shapes)
    gale = strands.Wind({"direction": [1, 0, 0], "speed": 14.0, "gust": 4.0, "gust_period": 0.6, "turbulence": 2.0,
                         "seed": 1})
    res = run(sc, wind=gale, engine=engine)
    assert res.x[-1, 13, 0] > 0.15                                               # it is blown hard toward the wall
    assert np.abs(np.linalg.norm(res.x[:, 1:] - res.x[:, :-1], axis=2) / 0.034 - 1.0).max() < 1e-9
    assert np.abs(rendered_positions(sc, res) - res.x).max() < 1e-9              # so the bones end where it was solved
    # and the penetration reported is the penetration of those positions
    chains, P = sc["chains"], sc["shapes"].pack()
    W = geom.world(P, sc["rest_R"], sc["rest_p"])
    own, frac = np.repeat(np.arange(len(chains)), 3), np.tile(geom.SAMPLES, len(chains))
    fk = rendered_positions(sc, res)
    closest = -1.0
    for f in range(res.x.shape[0]):
        head = np.concatenate([chains.head[:1], fk[f, :-1]])
        X = head[own] + frac[:, None] * (fk[f][own] - head[own])
        pen, _ = geom.measure(X, np.full(len(X), 0.01), W, None, P)
        assert max(pen.max(), 0.0) == pytest.approx(res.pen[f, 0], abs=1e-9)
        closest = max(closest, pen.max())
    assert -0.003 < closest < 0.005                                              # it leans on the wall, and is held


def test_input_mistakes_are_refused():
    sc = scene(straight_rig(), 8)
    bad = dict(sc, src_quat=sc["src_quat"] * 2.0)
    with pytest.raises(ValueError, match="unit quaternions"):
        run(bad)
    with pytest.raises(ValueError, match="src_pos"):
        run(dict(sc, src_pos=sc["src_pos"][:-1]))
    with pytest.raises(ValueError, match="anchor_src"):
        run(dict(sc, anchor_src=sc["anchor_src"] + 5))
    flipped = sc["src_quat"].copy()
    flipped[4:] *= -1.0                                             # q and -q are the same turn, but not for a lerp
    with pytest.raises(ValueError, match="sign-continuous"):
        run(dict(sc, src_quat=flipped))
    with pytest.raises(ValueError, match="fps"):
        run(sc, wind=strands.Wind({}, None, fps=24.0))
    with pytest.raises(TypeError):
        run(sc, wind={"speed": 1.0})
    rig = straight_rig(3)
    rig["chains"][0]["parents"] = [-1, 0, 0]
    with pytest.raises(ValueError, match="branching"):
        run(scene(rig, 8))
    with pytest.raises(ValueError):
        run(sc, {"hair": {"dragg": 1.0}})


def test_without_numba_auto_falls_back_to_numpy_and_asking_for_numba_fails(monkeypatch):
    monkeypatch.setattr(strands, "HAVE_NUMBA", False)
    sc = scene(straight_rig(), 6)
    assert run(sc, engine="auto").engine == "numpy"
    with pytest.raises(RuntimeError, match="numba"):
        run(sc, engine="numba")
    with pytest.raises(ValueError, match="engine"):
        run(sc, engine="gpu")


def test_quaternions_from_matrices_are_unit_and_continuous_through_a_half_turn():
    angle = np.linspace(0.0, 2.5 * np.pi, 40)                       # about z, past the half turn where w changes sign
    Rs = np.stack([geom.quat_to_mat(np.array([[np.cos(a / 2), 0, 0, np.sin(a / 2)]]))[0] for a in angle])[:, None]
    q = strands.quats_from_matrices(Rs)
    assert q.shape == (40, 1, 4) and np.allclose(np.linalg.norm(q, axis=2), 1.0)
    assert np.allclose(q[:, 0, 0], np.cos(angle / 2)) and np.allclose(geom.quat_to_mat(q[:, 0]), Rs[:, 0])


# ---------------------------------------------------------------- numba kernels against the numpy engine
def busy_scene(F=31):
    """Three strands swinging on a moving anchor among every kind of shape, in a gusty carrier's air."""
    rig = straight_rig(chains=3)
    shapes = geom.Shapes()
    shapes.add("sphere", ("world", "", ""), "ball", c=[0.12, 0.0, 1.3], R=0.07)
    shapes.add("capsule", ("world", "", ""), "rod", a=[0.3, -0.1, 1.15], b=[0.3, 0.1, 1.15], R=0.05)
    shapes.add("cylinder", ("world", "", ""), "post", M=np.array([[1, 0, 0, 0.06], [0, 1, 0, 0.06], [0, 0, 1, 1.15],
                                                                  [0, 0, 0, 1.0]]), R=0.05, hh=0.1, rnd=0.01)
    shapes.add("box", ("object", "", "crate"), "crate", M=np.eye(4), half=[0.05, 0.05, 0.08], rnd=0.01)
    shapes.floor_z = 1.08
    t = np.arange(F) / 30.0
    box_pos = np.stack([0.18 + 0.05 * np.sin(2 * np.pi * t), 0.02 + 0 * t, 1.2 + 0.03 * np.cos(2 * np.pi * t)], 1)
    box_q = np.array([rot_z(40.0 * np.sin(2 * np.pi * t[i])) for i in range(F)])
    sc = scene(rig, F, shapes)
    anchor_pos = np.stack([0.04 * np.sin(2 * np.pi * t), 0 * t, 0 * t], 1)
    anchor_q = np.array([rot_z(15.0 * np.sin(3 * np.pi * t[i])) for i in range(F)])
    j = shapes.sources.index(("object", "", "crate"))
    sc["src_pos"][:, 0], sc["src_quat"][:, 0] = anchor_pos, anchor_q
    sc["src_pos"][:, j], sc["src_quat"][:, j] = box_pos, box_q
    wind = strands.Wind({"direction": [1, 0.3, 0], "speed": 2.0, "exposure": 0.5, "gust": 1.0, "gust_period": 1.0,
                         "turbulence": 1.0, "seed": 2}, np.tile([1.0, 0.0, 0.0], (F, 1)), 30.0)
    return sc, wind


@needs_numba
def test_numba_engine_reproduces_the_numpy_engine_whatever_the_thread_count():
    sc, wind = busy_scene()
    a = run(sc, wind=wind, engine="numpy")
    b = run(sc, wind=wind, engine="numba", threads=1)
    c = run(sc, wind=wind, engine="numba", threads=3)
    assert a.engine == "numpy" and b.engine == "numba"
    assert np.array_equal(b.x, c.x) and np.array_equal(b.D, c.D) and np.array_equal(b.pen, c.pen)   # independent chains
    assert np.abs(a.x - b.x).max() < 1e-7 and np.abs(a.D - b.D).max() < 1e-6
    assert a.pen[:, 0].max() > 1e-6 and np.abs(a.pen - b.pen).max() < 1e-7                         # and they do touch
    assert [w[2] for w in a.worst if w[0] > 1e-6] == [w[2] for w in b.worst if w[0] > 1e-6]
    assert not np.allclose(a.x[:, 4, 1], 0.0, atol=1e-3)                                           # the air moved them


@needs_numba
@pytest.mark.parametrize("sweeps, ftl", [(1, 0.0), (2, 0.5), (6, 1.0)])
def test_numba_engine_follows_the_length_settings_like_the_numpy_engine(monkeypatch, sweeps, ftl):
    monkeypatch.setattr(strands, "SWEEPS", sweeps)
    monkeypatch.setattr(strands, "FTL_DAMP", ftl)
    sc, wind = busy_scene()
    a = run(sc, wind=wind, engine="numpy")
    b = run(sc, wind=wind, engine="numba", threads=1)
    assert np.abs(a.x - b.x).max() < 1e-7 and np.abs(a.D - b.D).max() < 1e-6


@needs_numba
def test_numba_pair_distances_and_normals_match_geom_penetration_for_every_shape_kind():
    rng = np.random.default_rng(7)

    def placed():
        q = rng.normal(size=(1, 4))
        M = np.eye(4)
        M[:3, :3], M[:3, 3] = geom.quat_to_mat(q / np.linalg.norm(q))[0], rng.uniform(-0.2, 0.2, 3)
        return M

    shapes = geom.Shapes()
    w = ("world", "", "")
    shapes.add("sphere", w, "ball", c=[0.05, -0.1, 0.0], R=0.15)
    shapes.add("capsule", w, "rod", a=[-0.2, 0.0, 0.1], b=[0.2, 0.1, -0.1], R=0.07)
    shapes.add("box", w, "round box", M=placed(), half=[0.2, 0.1, 0.15], rnd=0.03)
    shapes.add("box", w, "sharp box", M=placed(), half=[0.1, 0.2, 0.08], rnd=0.0)
    shapes.add("cylinder", w, "post", M=placed(), R=0.12, hh=0.2, rnd=0.02)
    sc = scene(straight_rig(), 1, shapes)
    M = strands._Model(sc["chains"], shapes, sc["src_pos"], sc["src_quat"], sc["rest_R"], sc["rest_p"],
                       sc["anchor_src"], None, (0, 0), sc["rig"], 30.0, 10, None, False, geom.ANCHOR_FREE)
    eng = strands._NumbaEngine(M, None, 1)
    blk = M.block([0], [0.0], [0.0])
    sc_, sb_, sm_, _, _ = eng._shapes(blk)
    X = rng.uniform(-0.3, 0.3, (500, 3))                                         # many inside the shapes, many outside
    r = rng.uniform(0.0, 0.03, 500)
    ref = geom.penetration(X, r, strands._world_row(blk.W, 0), None)
    inside = []
    for k, kind in enumerate(geom.KINDS):
        pen, nrm = ref[kind]
        for j in range(pen.shape[1]):
            col = sum(ref[kk][0].shape[1] for kk in geom.KINDS[:k]) + j
            got = np.array([strands._nb_dist(k, col, *X[m], r[m], sc_[0], sb_[0], sm_[0], eng.sk)
                            for m in range(len(X))])
            assert got[:, 0] == pytest.approx(-pen[:, j], abs=1e-12)
            assert got[:, 1:] == pytest.approx(nrm[:, j], abs=1e-9)
            inside.append(int((pen[:, j] > 0).sum()))
    assert len(inside) == 5 and min(inside) > 20                           # the inside branches are exercised too
    eng.close()


# ---------------------------------------------------------------- bone keys
def random_rotations(rng, n):
    q = rng.normal(size=(n, 4))
    return geom.quat_to_mat(q / np.linalg.norm(q, axis=1, keepdims=True))


def test_local_quats_invert_the_forward_kinematics_of_a_chain():
    rng = np.random.default_rng(5)
    F, N = 12, 4
    chains = geom.Chains(straight_rig(N), None)
    B = random_rotations(rng, N)                                                # bone rest rotations
    A = random_rotations(rng, 1)[0]                                             # the anchor's rest rotation
    Ra = random_rotations(rng, F)                                               # the anchor, posed on each frame
    # the keyed rotations: about random axes, the first bone's turning through a half turn (w changes sign)
    axis = rng.normal(size=(N, 3))
    axis /= np.linalg.norm(axis, axis=1, keepdims=True)
    angle = np.linspace(0.5, 1.5, F)[:, None] * np.pi * np.ones((1, N)) * np.array([1.0, 0.1, 0.2, 0.3])
    q_true = np.concatenate([np.cos(angle / 2)[..., None], np.sin(angle / 2)[..., None] * axis[None]], -1)
    D = np.zeros((F, N, 3, 3))
    Dp = Ra @ A.T
    for i in range(N):
        D[:, i] = Dp @ B[i] @ geom.quat_to_mat(q_true[:, i]) @ B[i].T
        Dp = D[:, i]
    res = SimpleNamespace(x=np.zeros((F, N, 3)), D=D, anchor_rot=np.repeat(Ra[:, None], N, 1),
                          anchor_rest_rot=np.repeat(A[None], N, 0), frames=(10, 10 + F - 1))
    q = strands.local_quats(chains, res, B)
    sign = np.einsum("fnj,fnj->fn", q, q_true)
    assert np.allclose(np.abs(sign), 1.0, atol=1e-12)                           # the same rotations
    assert (np.einsum("fnj,fnj->fn", q[1:], q[:-1]) > 0).all()                  # sign-continuous through the half turn
    assert np.all(np.sign(sign) == np.sign(sign[0]))                            # so each bone keeps one sign overall


def test_keys_from_a_simulation_pose_the_bones_where_the_solver_put_them():
    """Blender poses a child as parent_world @ (parent_rest^T @ rest) @ local rotation: playing the keys back through
    that must give D @ rest for every bone, whatever the (arbitrary) rest rotations of the bones and the anchor."""
    rng = np.random.default_rng(2)
    F, N = 21, 5
    B, A = random_rotations(rng, N), random_rotations(rng, 1)[0]
    sc = scene(straight_rig(N), F)
    t = np.arange(F) / 30.0
    Ra = np.array([geom.quat_to_mat(rot_z(40.0 * np.sin(5 * s))[None])[0] @ A for s in t])           # the anchor swings
    sc["rest_R"][0], sc["src_quat"][:, 0] = A, strands.quats_from_matrices(Ra[:, None])[:, 0]
    res = run(sc, wind=strands.Wind({"direction": [1, 0, 0], "speed": 2.0}))
    q = strands.local_quats(sc["chains"], res, B)
    world = np.zeros((F, N, 3, 3))
    for i in range(N):
        parent_world, parent_rest = (Ra, A) if i == 0 else (world[:, i - 1], B[i - 1])
        world[:, i] = parent_world @ parent_rest.T @ B[i] @ geom.quat_to_mat(q[:, i])
    assert np.abs(world - res.D @ B[None]).max() < 1e-9
    assert np.abs(res.D[:, 1:] - res.D[:, :-1]).max() > 0.01                                         # and they do bend


def test_local_quats_reports_the_frame_of_a_non_finite_rotation():
    chains = geom.Chains(straight_rig(2), None)
    D = np.tile(np.eye(3), (4, 2, 1, 1))
    D[2, 1] = np.nan
    res = SimpleNamespace(x=np.zeros((4, 2, 3)), D=D, anchor_rot=np.tile(np.eye(3), (4, 2, 1, 1)),
                          anchor_rest_rot=np.tile(np.eye(3), (2, 1, 1)), frames=(100, 103))
    with pytest.raises(FloatingPointError, match="c0b1.*102"):
        strands.local_quats(chains, res, np.tile(np.eye(3), (2, 1, 1)))


# ---------------------------------------------------------------- report
def test_report_measures_jerk_speed_and_penetration_in_the_anchor_frame():
    F = 12
    chains = geom.Chains(two_family_rig(), None)
    f = np.arange(F)
    x = np.zeros((F, 2, 3))
    x[:, 0, 0] = 0.5 * 0.4 * f ** 2 * 1e-3                    # hair: constant acceleration 0.4 mm / frame^2
    x[:, 1, 1] = 2.0e-3 * f                                   # tail: constant velocity 2 mm / frame
    Ra = np.tile(np.eye(3), (F, 2, 1, 1))
    Ra[:, :, :2, :2] = [[0.0, -1.0], [1.0, 0.0]]              # the anchor turned a quarter about z: x = R^T (x - p)
    anchor_pos = np.tile([10.0, 0.0, 0.0], (F, 2, 1))
    res = SimpleNamespace(frames=(100, 100 + F - 1), x=x + anchor_pos, anchor_pos=anchor_pos, anchor_rot=Ra,
                          pen=np.zeros((F, 3)), worst=[(0.0, "h0", "-")] * F)
    res.pen[4] = [0.0041, 0.0005, 2]
    res.pen[7] = [0.0103, 0.0, 1]
    res.worst[4], res.worst[7] = (0.0041, "h0", "body:neck"), (0.0103, "t0", "ball")
    rep = strands.report(chains, res, (100, 100 + F - 1))
    speed = 0.4 * (np.arange(F - 1) + 0.5)                    # mm / frame
    assert rep["hair"]["jerk_mm"] == pytest.approx(0.4)
    assert rep["hair"]["speed_mm"] == pytest.approx(np.median(speed), abs=1e-3)
    assert rep["hair"]["ratio"] == pytest.approx(0.4 / np.median(speed), abs=0.01)
    assert rep["tail"] == dict(jerk_mm=0.0, speed_mm=2.0, ratio=0.0, jerk_p95_mm=0.0)
    pen = rep["penetration_mm"]
    assert (pen["max"], pen["at_frame"], pen["bone"], pen["into"]) == (10.3, 107, "t0", "ball")
    assert pen["frames_over_2mm_by_shape"] == {"body:neck": 1, "ball": 1}
    inner = strands.report(chains, res, (103, 105))           # a window: only what happens inside it counts
    assert inner["penetration_mm"]["max"] == 4.1
    assert inner["penetration_mm"]["frames_over_2mm_by_shape"] == {"body:neck": 1}
    with pytest.raises(ValueError):
        strands.report(chains, res, (100, 101))               # too short to take a second difference
    with pytest.raises(ValueError):
        strands.report(chains, res, (90, 110))                # outside the simulated frames

# ---------------------------------------------------------------- command line
def test_the_command_line_runs_a_job_from_an_npz_and_writes_the_keys(tmp_path):
    rig = straight_rig(4)
    chains = geom.Chains(rig, None)
    F, N = 11, len(chains)
    t = np.arange(F) / 30.0
    src_pos = np.zeros((F, 1, 3))
    src_pos[:, 0] = np.array([0.0, 0.0, TOP]) + np.stack([0.03 * np.sin(6 * t), 0 * t, 0 * t], 1)
    src_quat = np.tile([1.0, 0, 0, 0], (F, 1, 1))
    rest_R, rest_p = np.eye(3)[None], np.array([[0.0, 0.0, TOP]])
    wind = {"direction": [1, 0, 0], "speed": 3.0, "gust": 1.0, "seed": 1}
    spec = {"rig": rig, "armature": "Arm", "frames": [1, F], "fps": 30, "substeps": 5, "settle_s": 0.5,
            "sources": [["bone", "Arm", "anchor0"]], "wind": wind, "engine": "numpy", "window": [1, F],
            "params": {"hair": {"drag": 6.0}}}
    B = np.tile(np.eye(3), (N, 1, 1))
    inp, out = tmp_path / "in.npz", tmp_path / "out.npz"
    np.savez(inp, head=chains.head, end=chains.end, bone_rest_R=B, src_pos=src_pos, src_quat=src_quat, rest_R=rest_R,
             rest_p=rest_p, spec=np.array(json.dumps(spec)))
    assert strands.main([str(inp), str(out)]) == 0
    z = np.load(out)
    shapes = geom.Shapes()
    a = shapes.src("bone", "Arm", "anchor0")
    res = strands.simulate(chains, shapes, src_pos, src_quat, rest_R, rest_p, np.full(N, a), spec["params"], (1, F),
                           rig, 30.0, 5, 0.5, strands.Wind(wind, None, 30.0), engine="numpy")
    assert z["quats"].shape == (F, N, 4) and np.array_equal(z["quats"], strands.local_quats(chains, res, B))
    assert np.array_equal(z["x"], res.x) and np.array_equal(z["pen"], res.pen)
    info = json.loads(str(z["report"]))
    assert info["bones"] == chains.bones and info["frames"] == [1, F] and info["engine"] == "numpy"
    assert "ratio" in info["report"]["hair"] and "penetration_mm" in info["report"]


def test_the_command_line_refuses_bad_jobs_with_a_usage_code(tmp_path, capsys):
    assert strands.main([]) == 2
    rig = straight_rig(2)
    chains = geom.Chains(rig, None)
    inp = tmp_path / "in.npz"
    spec = {"rig": rig, "armature": "Arm", "frames": [1, 4], "sources": []}           # the anchor is not a source
    np.savez(inp, head=chains.head, end=chains.end, bone_rest_R=np.tile(np.eye(3), (2, 1, 1)),
             src_pos=np.zeros((4, 0, 3)), src_quat=np.zeros((4, 0, 4)), rest_R=np.zeros((0, 3, 3)),
             rest_p=np.zeros((0, 3)), spec=np.array(json.dumps(spec)))
    assert strands.main([str(inp), str(tmp_path / "out.npz")]) == 2
    assert "spec.sources lacks" in capsys.readouterr().err


# ---------------------------------------------------------------- lateral coupling of sheet-like families (skirts)
def ring_rig(n=8, nb=3, radius=0.12, seg=0.08, shared=True, family="skirt", angles=None):
    """A skirt: n chains of nb bones hanging straight down from a ring of roots (radius, height TOP) around the z axis,
    all from one anchor bone `hips` (shared) or from one anchor bone each."""
    angles = 2.0 * np.pi * np.arange(n) / n if angles is None else np.asarray(angles, float)
    bones, chains = {"hips": {"head": [0, 0, TOP]}}, []
    for c, th in enumerate(angles):
        x, y = radius * np.cos(th), radius * np.sin(th)
        anchor = "hips" if shared else f"anchor{c}"
        bones.setdefault(anchor, {"head": [x, y, TOP]})
        names = [f"s{c}b{i}" for i in range(nb)]
        for i, name in enumerate(names):
            bones[name] = {"head": [x, y, TOP - i * seg]}
        chains.append({"family": family, "root": names[0], "anchor": anchor, "bones": names,
                       "parents": list(range(-1, nb - 1)), "ends": [[x, y, TOP - (i + 1) * seg] for i in range(nb)],
                       "body_radius": [0.02] * nb})
    return {"bones": bones, "chains": chains, "bodies": []}


def ring_scene(rig, F, shapes=None, poses=None):
    """Like scene() but every anchor bone is its own source (an object frame, identity at rest); poses maps an anchor
    bone to (positions (F,3), quaternions (F,4))."""
    chains = geom.Chains(rig, None)
    shapes = geom.Shapes() if shapes is None else shapes
    src = {a: shapes.src("object", "", a) for a in sorted(set(chains.anchor))}
    S = len(shapes.sources)
    pos, quat = np.zeros((F, S, 3)), np.zeros((F, S, 4))
    quat[..., 0] = 1.0
    for a, (p, q) in (poses or {}).items():
        pos[:, src[a]], quat[:, src[a]] = p, q
    return dict(chains=chains, shapes=shapes, src_pos=pos, src_quat=quat, rest_R=np.tile(np.eye(3), (S, 1, 1)),
                rest_p=np.zeros((S, 3)), anchor_src=np.array([src[a] for a in chains.anchor]), rig=rig, F=F)


def circle(angles_deg, radius=0.12):
    th = np.radians(angles_deg)
    root = np.stack([radius * np.cos(th), radius * np.sin(th), np.full(len(th), TOP)], 1)
    return root, root - np.array([0.0, 0.0, 0.3])


def test_ring_neighbours_follow_the_angle_around_the_skirt_and_close_a_full_ring():
    root, tip = circle(np.arange(8) * 45.0)
    assert strands.ring_neighbours(root, tip) == [(0, 1), (0, 7), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7)]
    order = np.random.default_rng(3).permutation(8)                     # the chains' order in the rig does not matter
    got = strands.ring_neighbours(root[order], tip[order])
    assert got == sorted({tuple(sorted((int(np.where(order == a)[0][0]), int(np.where(order == b)[0][0]))))
                          for a, b in strands.ring_neighbours(root, tip)})
    assert strands.ring_neighbours(*circle([0.0, 120.0, 240.0])) == [(0, 1), (0, 2), (1, 2)]      # three close up too


def test_ring_neighbours_leave_a_slit_an_arc_and_distant_chains_open():
    keep = [a for a in np.arange(12) * 30.0 if a not in (0.0, 30.0)]            # a 90 deg slit in a 12-chain skirt
    pairs = strands.ring_neighbours(*circle(keep))
    assert len(pairs) == 9 and (0, 9) not in pairs                              # first and last chain are not tied
    assert len(strands.ring_neighbours(*circle([0.0, 30.0, 60.0, 90.0, 120.0]))) == 4          # an arc: n - 1 links
    assert strands.ring_neighbours(*circle([0.0, 180.0])) == []                 # opposite sides of the body
    assert strands.ring_neighbours(*circle([0.0, 40.0])) == []                  # two chains are no ring
    assert strands.ring_neighbours(*circle([10.0])) == []


def test_ring_neighbours_keep_groups_apart():
    root, tip = circle(np.concatenate([np.arange(6) * 60.0, np.arange(6) * 60.0 + 7.0]))
    pairs = strands.ring_neighbours(root, tip, group=[0] * 6 + [1] * 6)
    assert len(pairs) == 12 and all((a < 6) == (b < 6) for a, b in pairs)


def test_lateral_defaults_couple_skirts_only_and_are_validated():
    assert strands.resolve_params(["skirt"])["skirt"]["lateral"] > 0
    assert all(strands.resolve_params([f])[f]["lateral"] == 0.0 for f in strands.FAMILY_DEFAULTS if f != "skirt")
    assert strands.resolve_params(["hair"], {"hair": {"lateral": 0.3}})["hair"]["lateral_collide"] == 1.0
    for bad in ({"lateral": 1.5}, {"lateral": -0.1}, {"lateral_collide": 0.5}):
        with pytest.raises(ValueError):
            strands.resolve_params(["skirt"], {"skirt": bad})


def test_without_lateral_coupling_chains_do_not_know_each_other():
    F = 16
    ramp = np.zeros((F, 3))
    ramp[:, 1] = np.linspace(0.0, 0.08, F)                                  # chain 0's anchor slides to its neighbour
    poses = {"anchor0": (ramp, np.tile([1.0, 0, 0, 0], (F, 1)))}
    off = {"skirt": {"lateral": 0.0}}
    ring = run(ring_scene(ring_rig(shared=False), F, poses=poses), off)
    alone = run(ring_scene(ring_rig(n=1, shared=False), F, poses=poses), off)
    still = run(ring_scene(ring_rig(shared=False), F), off)
    assert ring.links.shape == (0, 2)
    assert np.array_equal(ring.x[:, :3], alone.x)                           # chain 0 behaves as if it hung alone
    assert np.array_equal(ring.x[:, 3:], still.x[:, 3:])                    # and the others do not notice it moving
    coupled = run(ring_scene(ring_rig(shared=False), F, poses=poses))
    assert coupled.links.shape == (8 * 3, 2) and np.abs(coupled.x[:, 21:] - ring.x[:, 21:]).max() > 1e-3


@pytest.mark.parametrize("engine", ["numpy", pytest.param("numba", marks=needs_numba)])
def test_lateral_ties_drag_the_neighbours_along_and_keep_lengths_exact(engine):
    F = 21
    ramp = np.zeros((F, 3))
    ramp[:, 1] = np.linspace(0.0, 0.08, F)                                  # chain 0's anchor slides 8 cm toward +y
    q = np.tile([1.0, 0, 0, 0], (F, 1))
    tips = {}
    for lateral in (0.0, 0.8):
        sc = ring_scene(ring_rig(shared=False), F, poses={"anchor0": (ramp, q)})
        res = run(sc, {"skirt": {"lateral": lateral}}, engine=engine)
        tips[lateral] = res.x[:, [2, 5, 8, 11, 14, 17, 20, 23][::1]][:, :, :]
        seg = np.linalg.norm(res.x[:, 1:] - res.x[:, :-1], axis=2)[:, [i for i in range(23) if i % 3 != 2]]
        assert np.abs(seg / 0.08 - 1.0).max() < 1e-9                        # still inextensible with the ties
    moved = lambda t: np.linalg.norm(t[-1, 7] - t[0, 7])                     # the tip of chain 7, next to chain 0
    assert moved(tips[0.0]) < 1e-3 < 0.01 < moved(tips[0.8])
    gap = lambda t: np.linalg.norm(t[-1, 0] - t[-1, 7])                      # chain 0 to its neighbour: its tie
    assert gap(tips[0.8]) < gap(tips[0.0]) - 0.01


@pytest.mark.parametrize("engine", ["numpy", pytest.param("numba", marks=needs_numba)])
def test_the_cloth_between_chains_is_pushed_out_of_a_pillar_its_chord_cuts(engine):
    """Three strands hang against a pillar 60 deg apart: each touches it, but the straight cloth between two of them
    cuts 14 mm deep into it. The midpoints must push the pair out; the report counts the cloth too."""
    shapes = geom.Shapes()
    M = np.eye(4)
    M[2, 3] = TOP - 0.5
    shapes.add("cylinder", ("world", "", ""), "pillar", M=M, R=0.1, hh=0.9, rnd=0.0)
    rig = ring_rig(n=3, nb=4, radius=0.1105, angles=np.radians([0.0, 60.0, 120.0]))
    deep = {}
    for collide in (0.0, 1.0):
        res = run(ring_scene(rig, 12, shapes), {"skirt": {"lateral": 0.6, "lateral_collide": collide}}, engine=engine)
        deep[collide] = res.pen[:, 0].max()
    assert res.links.shape == (2 * 4, 2)                                    # two links of four rows (an arc, not a ring)
    assert deep[0.0] > 0.012                                                # the chord cuts in, and the report says so
    assert deep[1.0] < 0.004                                                # pushed out


def test_a_pair_whose_cloth_overlaps_a_body_at_rest_does_not_collide_with_it():
    """A sphere that clears every particle at rest but pokes through the cloth between them is the author's own overlap:
    the pair skips it, as a chain skips the bodies it overlaps at rest."""
    rig = ring_rig(n=8, nb=3)
    rig["bodies"] = [{"bone": "hips", "group": 0, "no_collide": [], "geom": {"kind": "sphere", "c": [0, 0, TOP - 0.1],
                                                                            "R": 0.105}}]
    shapes = geom.Shapes()
    chains = geom.Chains(rig, None)
    geom.model_shapes(shapes, rig, "", skip_bones=chains.chain_bones)
    sc = ring_scene(rig, 4, shapes)
    sc["rest_p"][shapes.sources.index(("bone", "", "hips"))] = [0.0, 0.0, 0.0]
    args = (sc["chains"], shapes, sc["src_pos"], sc["src_quat"], sc["rest_R"], sc["rest_p"], sc["anchor_src"])
    model = strands._Model(*args, None, (0, 3), rig, 30.0, 5, None, False, geom.ANCHOR_FREE)
    assert model.enable["sphere"].all()          # a coupled sheet collides with the body it hangs from near its roots
    row0 = model.link_a % 3 == 0                                             # the links of the first row of every chain
    low = model.link_a % 3 == 2                                              # the last row of every chain
    assert not model.link_enable["sphere"][~low].any()                       # the cloth of the upper rows overlaps it
    assert model.link_enable["sphere"][low].all()                            # the cloth lower down does not: collides
    assert row0.sum() == 8
    loose = strands._Model(*args, {"skirt": {"lateral": 0.0}}, (0, 3), rig, 30.0, 5, None, False, geom.ANCHOR_FREE)
    assert not loose.enable["sphere"].any()      # uncoupled chains keep the rule: the anchor's body is skipped


@needs_numba
def test_numba_engine_reproduces_the_numpy_engine_with_lateral_links(monkeypatch):
    shapes = geom.Shapes()
    M = np.eye(4)
    M[:3, 3] = [0.1, 0.1, TOP - 0.2]
    shapes.add("cylinder", ("world", "", ""), "post", M=M, R=0.05, hh=0.3, rnd=0.01)
    shapes.add("sphere", ("world", "", ""), "ball", c=[-0.14, 0.0, TOP - 0.12], R=0.06)
    shapes.floor_z = TOP - 0.3
    F = 25
    t = np.arange(F) / 30.0
    swing = np.array([rot_z(25.0 * np.sin(6 * s)) for s in t])
    hips = (np.stack([0.03 * np.sin(8 * t), 0 * t, 0 * t], 1), swing)
    sc = ring_scene(ring_rig(n=6, nb=4, shared=True), F, shapes, poses={"hips": hips})
    wind = strands.Wind({"direction": [1, 0.4, 0], "speed": 3.0, "gust": 1.5, "turbulence": 1.0, "seed": 2})
    a = run(sc, wind=wind, engine="numpy")
    b = run(sc, wind=wind, engine="numba", threads=1)
    c = run(sc, wind=wind, engine="numba", threads=4)
    assert a.links.shape[0] == 6 * 4 and np.array_equal(a.links, b.links)
    assert np.array_equal(b.x, c.x) and np.array_equal(b.pen, c.pen)
    assert np.abs(a.x - b.x).max() < 1e-7 and np.abs(a.D - b.D).max() < 1e-6 and np.abs(a.pen - b.pen).max() < 1e-7
    assert a.pen[:, 0].max() > 1e-4                                          # the skirt does touch the shapes
