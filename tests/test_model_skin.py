import numpy as np
import pytest

from mkmmd.model import skin
from mkmmd.model.part import Mesh


# ---------------------------------------------------------------- helpers
def _tube(rings, segs, radius=0.1, height=1.0, phase=0.0):
    """Open tube along +Z, quads, outward normals."""
    z = np.linspace(0.0, height, rings)
    a = phase + 2 * np.pi * np.arange(segs) / segs
    verts = np.array([[radius * np.cos(t), radius * np.sin(t), zz] for zz in z for t in a])
    faces = []
    for i in range(rings - 1):
        for j in range(segs):
            j2 = (j + 1) % segs
            faces.append([i * segs + j, i * segs + j2, (i + 1) * segs + j2, (i + 1) * segs + j])
    return verts, faces


def _grid(nx, ny):
    """(nx + 1) x (ny + 1) vertices on the unit square, quads."""
    xs, ys = np.meshgrid(np.linspace(0, 1, nx + 1), np.linspace(0, 1, ny + 1))
    verts = np.stack([xs.ravel(), ys.ravel(), np.zeros(xs.size)], axis=1)
    idx = np.arange((nx + 1) * (ny + 1)).reshape(ny + 1, nx + 1)
    quads = np.stack([idx[:-1, :-1].ravel(), idx[:-1, 1:].ravel(), idx[1:, 1:].ravel(), idx[1:, :-1].ravel()], axis=1)
    return verts, [list(map(int, q)) for q in quads]


def _total(w):
    return sum(w.values())


def _ericson(p, a, b, c):
    """Closest point on a triangle: Ericson's region test (independent of the module's implementation)."""
    ab, ac, ap = b - a, c - a, p - a
    d1, d2 = ab @ ap, ac @ ap
    if d1 <= 0 and d2 <= 0:
        return a
    bp = p - b
    d3, d4 = ab @ bp, ac @ bp
    if d3 >= 0 and d4 <= d3:
        return b
    vc = d1 * d4 - d3 * d2
    if vc <= 0 and d1 >= 0 and d3 <= 0:
        return a + ab * (d1 / (d1 - d3))
    cp = p - c
    d5, d6 = ab @ cp, ac @ cp
    if d6 >= 0 and d5 <= d6:
        return c
    vb = d5 * d2 - d1 * d6
    if vb <= 0 and d2 >= 0 and d6 <= 0:
        return a + ac * (d2 / (d2 - d6))
    va = d3 * d6 - d5 * d4
    if va <= 0 and (d4 - d3) >= 0 and (d5 - d6) >= 0:
        return b + (c - b) * ((d4 - d3) / ((d4 - d3) + (d5 - d6)))
    den = 1.0 / (va + vb + vc)
    return a + ab * (vb * den) + ac * (vc * den)


def _brute_distance(points, verts, faces):
    tris = [(f[0], f[i], f[i + 1]) for f in faces for i in range(1, len(f) - 1)]
    out = np.empty(len(points))
    for i, p in enumerate(points):
        out[i] = min(np.linalg.norm(p - _ericson(p, *verts[list(t)])) for t in tris)
    return out


# ---------------------------------------------------------------- normalise
def test_normalise_sums_to_one_and_keeps_only_used_bones():
    w = {"a": np.array([2.0, 0.0, 0.3]), "b": np.array([2.0, 1.0, 0.0]), "unused": np.zeros(3)}
    r = skin.normalise(w, 3)
    assert list(r) == ["a", "b"]                                  # all-zero bone dropped, input order kept
    assert all(v.dtype == np.float64 and v.shape == (3,) for v in r.values())
    assert np.allclose(_total(r), 1.0)
    assert np.allclose(r["a"], [0.5, 0.0, 1.0]) and np.allclose(r["b"], [0.5, 1.0, 0.0])
    assert w["a"].tolist() == [2.0, 0.0, 0.3]                     # the input is untouched


def test_normalise_cap_keeps_the_largest_and_floor_drops_the_small():
    row = [0.30, 0.25, 0.20, 0.15, 0.06, 0.04]
    w = {f"b{i}": np.array([x]) for i, x in enumerate(row)}
    r = skin.normalise(w, 1)
    assert sorted(r) == ["b0", "b1", "b2", "b3"]
    assert [float(r[f"b{i}"][0]) for i in range(4)] == pytest.approx([x / 0.9 for x in row[:4]])
    r2 = skin.normalise(w, 1, cap=2)
    assert sorted(r2) == ["b0", "b1"] and r2["b0"][0] == pytest.approx(0.30 / 0.55)
    # floor: raw weights below it are dropped, and so are weights that fall below it once renormalised
    f = skin.normalise({"a": np.array([1.0, 1.0]), "b": np.array([5e-5, 1.0001e-4])}, 2)
    assert list(f) == ["a"] and np.allclose(f["a"], 1.0)
    g = skin.normalise({"a": np.array([1.0]), "b": np.array([0.01])}, 1, floor=0.05)
    assert list(g) == ["a"]
    h = skin.normalise({"a": np.array([1.0]), "b": np.array([0.01])}, 1, floor=0.0)
    assert h["b"][0] == pytest.approx(0.01 / 1.01)


def test_normalise_cap_ties_prefer_earlier_bones_and_result_is_idempotent():
    w = {k: np.array([0.2]) for k in "abcde"}
    r = skin.normalise(w, 1)
    assert sorted(r) == ["a", "b", "c", "d"] and r["a"][0] == pytest.approx(0.25)
    rng = np.random.default_rng(0)
    big = {f"b{i}": rng.random(200) * (rng.random(200) < 0.4) for i in range(9)}
    big["b0"] += 0.01                                              # every vertex has something
    once = skin.normalise(big, 200)
    twice = skin.normalise(once, 200)
    assert once.keys() == twice.keys()
    assert all(np.allclose(once[k], twice[k]) for k in once)
    st = skin.check(once, 200)
    assert st["max_bones_per_vertex"] <= 4 and st["ok"] and st["sum_min"] == pytest.approx(1.0)


def test_normalise_fallback_and_unweighted_error():
    w = {"a": np.array([1.0, 0.0, 0.0, 5e-5]), "b": np.array([1.0, 0.0, 0.0, 0.0])}
    with pytest.raises(ValueError, match=r"\b3 of 4\b"):
        skin.normalise(w, 4)                                       # vertices 1, 2 and 3 are unweighted (5e-5 < floor)
    r = skin.normalise(w, 4, fallback="root")
    assert list(r) == ["a", "b", "root"]                           # fallback bone appended
    assert np.allclose(r["root"], [0, 1, 1, 1]) and np.allclose(_total(r), 1.0)
    assert np.allclose(r["a"], [0.5, 0, 0, 0])
    r = skin.normalise(w, 4, fallback="a")                         # fallback that already exists
    assert list(r) == ["a", "b"] and np.allclose(r["a"], [0.5, 1, 1, 1])
    with pytest.raises(ValueError, match="3 of 3"):
        skin.normalise({}, 3)
    assert skin.normalise({}, 0) == {}
    assert skin.normalise({}, 2, fallback="r")["r"].tolist() == [1.0, 1.0]


def test_normalise_rejects_bad_input():
    with pytest.raises(ValueError, match="shape"):
        skin.normalise({"a": np.ones(4)}, 3)
    with pytest.raises(ValueError, match="non-finite"):
        skin.normalise({"a": np.array([1.0, np.nan])}, 2)
    with pytest.raises(ValueError, match="cap"):
        skin.normalise({"a": np.ones(2)}, 2, cap=0)
    r = skin.normalise({"a": np.array([1.0, -0.5]), "b": np.array([0.0, 1.0])}, 2)   # negatives count as zero
    assert np.allclose(r["a"], [1, 0]) and np.allclose(r["b"], [0, 1])


# ---------------------------------------------------------------- weights algebra
def test_matrix_roundtrip_and_name_order():
    w = {"a": np.array([1.0, 0.0, 0.5]), "b": np.array([0.0, 0.0, 0.5]), "c": np.array([0.0, 1.0, 0.0])}
    names, W = skin.to_matrix(w, 3)
    assert names == ["a", "b", "c"] and W.shape == (3, 3) and W.dtype == np.float64
    assert np.allclose(W, [[1, 0, 0], [0, 0, 1], [0.5, 0.5, 0]])
    names2, W2 = skin.to_matrix(w, 3, names=["c", "z", "a", "b"])      # own order, missing bone = zero column
    assert names2 == ["c", "z", "a", "b"] and np.allclose(W2[:, 1], 0) and np.allclose(W2[:, 0], w["c"])
    with pytest.raises(ValueError):
        skin.to_matrix(w, 3, names=["a", "b"])                         # 'c' would be lost
    names3, W3 = skin.to_matrix(w, 3, names=["a", "b"], strict=False)
    assert names3 == ["a", "b"] and W3.shape == (3, 2)
    back = skin.from_matrix(names2, W2)
    assert list(back) == ["c", "a", "b"]                               # zero column 'z' dropped
    assert all(np.array_equal(back[k], w[k]) for k in back)
    assert "z" in skin.from_matrix(names2, W2, keep_zero=True)
    W[:, 0] = 9.0
    assert w["a"][0] == 1.0                                            # to_matrix copies
    with pytest.raises(ValueError):
        skin.from_matrix(["a"], np.zeros((3, 2)))
    with pytest.raises(ValueError):
        skin.from_matrix(["a", "a"], np.zeros((3, 2)))
    assert skin.from_matrix([], np.zeros((4, 0))) == {}


def test_add_scale_mix_rigid_algebra():
    a = {"x": np.array([1.0, 0.5, 0.0]), "y": np.array([0.0, 0.5, 1.0])}
    b = {"y": np.array([1.0, 1.0, 0.0]), "z": np.array([0.0, 0.0, 1.0])}
    s = skin.add(a, b)
    assert list(s) == ["x", "y", "z"] and np.allclose(s["y"], [1.0, 1.5, 1.0]) and np.allclose(s["z"], [0, 0, 1])
    s["x"][0] = 99.0
    assert a["x"][0] == 1.0                                            # no aliasing of inputs
    assert skin.add() == {} and list(skin.add(a)) == ["x", "y"]
    with pytest.raises(ValueError):
        skin.add(a, {"x": np.ones(4)})
    sc = skin.scale(a, 2.0)
    assert np.allclose(sc["x"], [2, 1, 0])
    sc = skin.scale(a, np.array([1.0, 0.0, 2.0]))
    assert np.allclose(sc["y"], [0, 0, 2])
    with pytest.raises(ValueError):
        skin.scale(a, np.ones(5))
    m = skin.mix(a, b, 0.25)
    assert list(m) == ["x", "y", "z"]
    assert np.allclose(m["x"], 0.75 * a["x"]) and np.allclose(m["z"], 0.25 * b["z"])
    assert np.allclose(m["y"], 0.75 * a["y"] + 0.25 * b["y"])
    t = np.array([0.0, 0.5, 1.0])
    m = skin.mix(a, b, t)
    assert np.allclose(m["x"], [1.0, 0.25, 0.0]) and np.allclose(m["z"], [0, 0, 1])
    assert np.allclose(m["y"], [0.0, 0.75, 0.0])
    full = skin.mix(skin.normalise(a, 3, fallback="x"), skin.normalise(b, 3, fallback="y"), t)
    assert np.allclose(_total(full), 1.0)                              # a mix of normalised weights is normalised
    rb = skin.region_blend(a, b, t)
    assert all(np.allclose(rb[k], m[k]) for k in m)
    r = skin.rigid(4, "bone")
    assert list(r) == ["bone"] and r["bone"].shape == (4,) and r["bone"].tolist() == [1.0] * 4
    with pytest.raises(ValueError):
        skin.mix(a, b, np.ones(5))


# ---------------------------------------------------------------- ramps
def test_smoothstep_and_ramp_known_points():
    assert skin.smoothstep(-1) == 0.0 and skin.smoothstep(2) == 1.0 and skin.smoothstep(0.5) == 0.5
    assert skin.smoothstep(0.25) == pytest.approx(0.15625)
    x = np.array([-1.0, 0.0, 0.25, 0.5, 1.0, 3.0])
    assert np.allclose(skin.smoothstep(x), [0, 0, 0.15625, 0.5, 1, 1])
    assert np.allclose(skin.ramp(x, 0.0, 1.0), skin.smoothstep(x))
    assert np.allclose(skin.ramp(x, 0.0, 1.0, smooth=False), [0, 0, 0.25, 0.5, 1, 1])
    assert np.allclose(skin.ramp(x, 1.0, 0.0, smooth=False), [1, 1, 0.75, 0.5, 0, 0])        # falling ramp
    assert np.allclose(skin.ramp(x, 1.0, 0.0), 1.0 - skin.smoothstep(x))
    assert np.allclose(skin.ramp(np.array([0.0, 1.0, 2.0]), 1.0, 1.0), [0, 1, 1])               # hard step
    assert np.allclose(skin.ramp(np.array([0.5, 1.5]), np.array([0.0, 1.0]), np.array([1.0, 2.0])), 0.5)
    assert np.isscalar(skin.ramp(0.5, 0.0, 1.0)) and skin.ramp(0.5, 0.0, 1.0) == 0.5


def test_plane_ramp_front_back_width_and_offset():
    v = np.array([[0, 0, z] for z in (-1.0, -0.1, -0.05, 0.0, 0.025, 0.05, 0.1, 1.0)], float)
    r = skin.plane_ramp(v, [0, 0, 0], [0, 0, 2.0], 0.1)                # normal need not be unit length
    assert r.shape == (8,)
    assert np.allclose(r, [0, 0, 0, 0.5, skin.smoothstep(0.75), 1, 1, 1])
    r = skin.plane_ramp(v, [0, 0, 0], [0, 0, -1], 0.1)                 # facing the other way
    assert np.allclose(r, [1, 1, 1, 0.5, skin.smoothstep(0.25), 0, 0, 0])
    r = skin.plane_ramp(v, [5, 5, 0], [0, 0, 1], 0.1, offset=0.5)      # plane at z = 0.5; point x/y irrelevant
    assert np.allclose(r, [0, 0, 0, 0, 0, 0, 0, 1])
    r = skin.plane_ramp(v, [0, 0, 0], [0, 0, 1], 0.0)                  # hard step
    assert set(np.unique(r)) == {0.0, 1.0} and r[3] == 1.0 and r[2] == 0.0
    lin = skin.plane_ramp(v, [0, 0, 0], [0, 0, 1], 0.1, smooth=False)
    assert lin[4] == pytest.approx(0.75)
    diag = skin.plane_ramp(np.array([[1.0, 0, 0], [-1.0, 0, 0], [0, 1.0, 0]]), [0, 0, 0], [1, 1, 0], 0.2)
    assert np.allclose(diag, [1, 0, 1])
    with pytest.raises(ValueError):
        skin.plane_ramp(v, [0, 0, 0], [0, 0, 0], 0.1)
    with pytest.raises(ValueError):
        skin.plane_ramp(v, [0, 0, 0], [0, 0, 1], -0.1)


def test_sphere_and_capsule_ramp():
    v = np.array([[0, 0, 0], [0.1, 0, 0], [0.15, 0, 0], [0.2, 0, 0], [0, 0.5, 0]], float)
    r = skin.sphere_ramp(v, [0, 0, 0], 0.1, 0.2)
    assert np.allclose(r, [1, 1, 0.5, 0, 0])
    assert np.allclose(skin.sphere_ramp(v, [0, 0, 0], 0.1, 0.2, smooth=False), [1, 1, 0.5, 0, 0])
    assert np.allclose(skin.sphere_ramp(v, [0, 0, 0], 0.1, 0.3, smooth=False)[2:4], [0.75, 0.5])
    hard = skin.sphere_ramp(v, [0, 0, 0], 0.15, 0.15)
    assert np.allclose(hard, [1, 1, 0, 0, 0])
    with pytest.raises(ValueError):
        skin.sphere_ramp(v, [0, 0, 0], 0.3, 0.2)
    a, b = [0, 0, 0], [0, 0, 1]
    pts = np.array([[0.05, 0, 0.5], [0.15, 0, 0.5], [0.3, 0, 0.5], [0, 0, 1.15], [0, 0, -0.3], [0.1, 0, 1.0]])
    c = skin.capsule_ramp(pts, a, b, 0.1, 0.2)
    assert np.allclose(c, [1, 0.5, 0, 0.5, 0, 1])
    deg = skin.capsule_ramp(pts, a, a, 0.1, 0.2)                       # zero-length capsule = sphere
    assert np.allclose(deg, skin.sphere_ramp(pts, a, 0.1, 0.2))


def test_twist_ramp_splits_a_limb_along_its_axis():
    a, b = [0.0, 0.0, 0.0], [0.0, 0.0, 2.0]
    v = np.array([[0.3, 0, -1.0], [0, 0, 0.0], [0.1, 0, 0.5], [0, 0.2, 1.0], [0, 0, 1.5], [0, 0, 2.0], [0, 0, 9.0]])
    r = skin.twist_ramp(v, a, b)                                       # parameter 0 .. 1
    assert np.allclose(r, [0, 0, skin.smoothstep(0.25), 0.5, skin.smoothstep(0.75), 1, 1])
    r = skin.twist_ramp(v, a, b, start=0.25, end=0.75, smooth=False)
    assert np.allclose(r, [0, 0, 0, 0.5, 1, 1, 1])
    r = skin.twist_ramp(v, a, b, start=1.0, end=0.0, smooth=False)     # reversed
    assert np.allclose(r, [1, 1, 0.75, 0.5, 0.25, 0, 0])
    with pytest.raises(ValueError):
        skin.twist_ramp(v, a, a)
    w = skin.mix(skin.rigid(len(v), "arm"), skin.rigid(len(v), "twist"), skin.twist_ramp(v, a, b, 0.25, 0.75))
    assert np.allclose(w["arm"] + w["twist"], 1.0) and w["twist"][3] == pytest.approx(0.5)


# ---------------------------------------------------------------- chains
def test_chain_param_straight_and_bent():
    J = [[0, 0, 0], [0, 0, 1], [0, 0, 2], [0, 0, 3]]
    v = np.array([[0.3, 0, -1.0], [0, 0, 1.5], [0.2, 0, 2.25], [0, 0, 5.0], [0.0, 0.1, 1.0]])
    s, d = skin.chain_param(v, J)
    assert np.allclose(s, [0, 1.5, 2.25, 3, 1.0])
    assert np.allclose(d, [np.hypot(0.3, 1.0), 0, 0.2, 2.0, 0.1])
    # L-shaped chain: up one metre, then along +x; sweep a circle around the joint: s is continuous and monotone
    L = [[0, 0, 0], [0, 0, 1], [1, 0, 1]]
    th = np.radians(np.linspace(0, 270, 91))
    ring = np.array([0.0, 0.0, 1.0]) + 0.3 * np.stack([np.cos(th), np.zeros_like(th), np.sin(th)], axis=1)
    s, d = skin.chain_param(ring, L)
    assert s[0] == pytest.approx(1.3) and s[-1] == pytest.approx(0.7)
    assert np.all(np.diff(s) <= 1e-12) and np.abs(np.diff(s)).max() < 0.05
    outer = (th >= np.radians(90)) & (th <= np.radians(180))
    assert np.allclose(s[outer], 1.0) and np.allclose(d[outer], 0.3)   # the joint is the nearest point there
    with pytest.raises(ValueError):
        skin.chain_param(v, [[0, 0, 0]])


def test_chain_weights_straight_three_bone_chain():
    J = [[0, 0, 0], [0, 0, 1], [0, 0, 2], [0, 0, 3]]
    bones = ["a", "b", "c"]
    z = np.linspace(-1.0, 4.0, 51)
    v = np.stack([0.05 * np.sin(7 * z), 0.03 * np.cos(3 * z), z], axis=1)       # off-axis wobble must not matter
    w = skin.chain_weights(v, bones, J, blend=0.4)
    assert list(w) == bones and all(x.shape == (51,) and x.dtype == np.float64 for x in w.values())
    assert np.allclose(_total(w), 1.0, atol=1e-12)
    for x in w.values():
        assert x.min() >= 0.0 and x.max() <= 1.0
    ref_a = 1.0 - skin.smoothstep((z - 0.8) / 0.4)
    ref_c = skin.smoothstep((z - 1.8) / 0.4)
    assert np.allclose(w["a"], ref_a, atol=1e-12) and np.allclose(w["c"], ref_c, atol=1e-12)
    assert np.allclose(w["b"], 1.0 - ref_a - ref_c, atol=1e-12)

    def at(zz):
        got = skin.chain_weights([[0, 0, zz]], bones, J, blend=0.4)
        return {b: float(got[b][0]) if b in got else 0.0 for b in bones}

    j1, j2 = at(1.0), at(2.0)
    assert j1["a"] == pytest.approx(0.5) and j1["b"] == pytest.approx(0.5) and j1["c"] == 0.0     # 50/50 at a joint
    assert j2["b"] == pytest.approx(0.5) and j2["c"] == pytest.approx(0.5) and j2["a"] == 0.0
    assert at(0.8)["a"] == 1.0 and at(1.2)["a"] == 0.0 and at(0.9)["a"] == pytest.approx(0.84375)   # width honoured
    assert at(-5.0) == {"a": 1.0, "b": 0.0, "c": 0.0} and at(7.0) == {"a": 0.0, "b": 0.0, "c": 1.0}  # clamped ends
    assert np.all(np.diff(w["a"]) <= 1e-12) and np.all(np.diff(w["c"]) >= -1e-12)                  # monotone
    assert np.allclose(skin.chain_weights(v, bones, J, blend=0.0)["a"], (z < 1.0).astype(float))      # hard split


def test_chain_weights_blend_list_clamp_and_lone_bone():
    J = [[0, 0, 0], [0, 0, 1], [0, 0, 2], [0, 0, 3]]
    z = np.linspace(0, 3, 301)
    v = np.stack([np.zeros_like(z), np.zeros_like(z), z], axis=1)
    w = skin.chain_weights(v, ["a", "b", "c"], J, blend=[0.2, 0.6])
    assert np.allclose(w["a"], 1.0 - skin.smoothstep((z - 0.9) / 0.2), atol=1e-12)
    assert np.allclose(w["c"], skin.smoothstep((z - 1.7) / 0.6), atol=1e-12)
    huge = skin.chain_weights(v, ["a", "b", "c"], J, blend=50.0)       # clamped to the segment length (1 m)
    assert np.allclose(huge["a"], 1.0 - skin.smoothstep(z - 0.5), atol=1e-12)
    assert np.allclose(huge["c"], skin.smoothstep(z - 1.5), atol=1e-12)
    assert huge["a"][50] == 1.0 and huge["a"][150] == 0.0 and huge["b"][150] == pytest.approx(1.0)
    uneven = skin.chain_weights(v, ["a", "b", "c"], [[0, 0, 0], [0, 0, 1], [0, 0, 1.1], [0, 0, 3]], blend=0.4)
    assert np.allclose(_total(uneven), 1.0) and min(x.min() for x in uneven.values()) >= 0.0
    assert uneven["b"].max() == pytest.approx(1.0, abs=1e-6)           # the short middle bone is fully faded in at its middle
    one = skin.chain_weights(v, ["only"], [[0, 0, 0], [0, 0, 3]])
    assert list(one) == ["only"] and np.allclose(one["only"], 1.0)
    skipped = skin.chain_weights(v[:20], ["a", "b", "c"], J, blend=0.2)
    assert list(skipped) == ["a"]                                      # bones without any weight are not returned
    assert skin.chain_weights(np.zeros((0, 3)), ["a", "b"], [[0, 0, 0], [0, 0, 1], [0, 0, 2]]) == {}
    # a zero-length segment in the chain must not produce NaN
    deg = skin.chain_weights(v, ["a", "b", "c"], [[0, 0, 0], [0, 0, 1], [0, 0, 1], [0, 0, 3]], blend=0.2)
    assert all(np.isfinite(x).all() for x in deg.values()) and np.allclose(_total(deg), 1.0)


def test_chain_weights_l_shaped_chain_follows_arc_length():
    J = np.array([[0, 0, 0], [0, 0, 1], [1, 0, 1]], float)
    u = np.linspace(0.0, 2.0, 81)                                      # arc length: 1 m up, then 1 m along +x
    pts = np.array([[0, 0, x] if x <= 1 else [x - 1, 0, 1] for x in u])
    w = skin.chain_weights(pts, ["up", "side"], J, blend=0.4)
    ref = skin.smoothstep((u - 0.8) / 0.4)
    assert np.allclose(w["side"], ref, atol=1e-12) and np.allclose(w["up"], 1.0 - ref, atol=1e-12)
    assert np.allclose(_total(w), 1.0, atol=1e-12)
    assert np.all(np.diff(w["side"]) >= -1e-12)                        # monotone along the chain
    j = int(np.argmin(np.abs(u - 1.0)))
    assert w["up"][j] == pytest.approx(0.5) and w["side"][j] == pytest.approx(0.5)               # 50/50 at the joint
    assert w["up"][np.argmin(np.abs(u - 0.8))] == 1.0 and w["side"][np.argmin(np.abs(u - 1.2))] == 1.0
    # clamped ends: before the start and beyond the end
    ends = skin.chain_weights(np.array([[0, 0, -2.0], [-0.1, 0, -0.5], [5.0, 0, 1.0], [2.0, 0.3, 1.0]]),
                              ["up", "side"], J, blend=0.4)
    assert np.allclose(ends["up"], [1, 1, 0, 0]) and np.allclose(ends["side"], [0, 0, 1, 1])
    # off the chain: smooth on the inside of the bend, 50/50 along the bisector, saturated on either side of it
    n = np.array([1.0, 0.0, 1.0]) / np.sqrt(2.0)
    tau = np.linspace(-0.3, 0.3, 61)
    inner = np.array([0.2, 0.0, 0.8]) + tau[:, None] * n               # crosses the bisector plane inside the bend
    wi = skin.chain_weights(inner, ["up", "side"], J, blend=0.4)
    assert wi["side"][30] == pytest.approx(0.5) and wi["side"][0] == 0.0 and wi["side"][-1] == 1.0
    assert np.all(np.diff(wi["side"]) >= -1e-12) and np.abs(np.diff(wi["side"])).max() < 0.1
    outer = np.array([[-0.3, 0, 1.3], [-0.3, 0, 1.0], [0.5, 0, 1.3], [0.3, 0, 1.6]])   # outside of the bend
    wo = skin.chain_weights(outer, ["up", "side"], J, blend=0.4)
    assert np.allclose(wo["side"], [0.5, 0, 1, 1]) and np.allclose(wo["up"], [0.5, 1, 0, 0])


def test_chain_weights_curled_chain_and_random_chains():
    # a half circle of 8 segments: the middle of segment i is owned by bone i all the way round
    k = 8
    ang = np.linspace(0, np.pi, k + 1)
    J = np.stack([np.sin(ang), np.zeros(k + 1), -np.cos(ang)], axis=1)
    mid_ang = (ang[:-1] + ang[1:]) / 2
    mids = 1.05 * np.stack([np.sin(mid_ang), np.zeros(k), -np.cos(mid_ang)], axis=1)
    names = [f"b{i}" for i in range(k)]
    w = skin.chain_weights(mids, names, J, blend=0.1)
    _, W = skin.to_matrix(w, k, names)
    assert (W.argmax(axis=1) == np.arange(k)).all() and np.allclose(W.max(axis=1), 1.0)
    # random chains: weights are valid everywhere (even with hairpin bends); on the chain itself they follow the
    # arc-length formula (bends below the hairpin clamp)
    rng = np.random.default_rng(5)

    def walk(kk, bend):
        d = np.array([0.0, 0.0, 1.0])
        pts = [np.zeros(3)]
        for _ in range(kk):
            step = rng.normal(size=3)
            d = d + bend * step / np.linalg.norm(step)
            d /= np.linalg.norm(d)
            pts.append(pts[-1] + d * rng.uniform(0.2, 0.7))
        return np.array(pts)

    for trial in range(8):
        wild = trial % 4 == 3
        kk = int(rng.integers(2, 7))
        J = walk(kk, 3.0 if wild else 0.9)                  # bend <= asin(0.9) = 64 deg unless wild
        names = [f"j{i}" for i in range(kk)]
        blend = rng.uniform(0.0, 0.6, kk - 1)
        far = J[0] + rng.normal(size=(300, 3))
        wf = skin.chain_weights(far, names, J, blend=blend)
        assert np.allclose(_total(wf), 1.0, atol=1e-12) and min(x.min() for x in wf.values()) >= 0.0
        if wild:
            continue
        ln = np.linalg.norm(np.diff(J, axis=0), axis=1)
        cum = np.concatenate([[0.0], np.cumsum(ln)])
        wid = np.minimum(blend, np.minimum(ln[:-1], ln[1:]))
        u = rng.uniform(0.0, cum[-1], 200)
        seg = np.clip(np.searchsorted(cum, u, side="right") - 1, 0, kk - 1)
        on = J[seg] + (u - cum[seg])[:, None] / ln[seg][:, None] * (J[seg + 1] - J[seg])
        won = skin.chain_weights(on, names, J, blend=blend)
        passed = [np.ones_like(u)] + [skin.ramp(u, cum[j] - wid[j - 1] / 2, cum[j] + wid[j - 1] / 2)
                                      for j in range(1, kk)] + [np.zeros_like(u)]
        for i, name in enumerate(names):
            expect = passed[i] - passed[i + 1]
            got = won.get(name, np.zeros_like(u))
            assert np.allclose(got, expect, atol=1e-9), (trial, name)


def test_chain_weights_errors():
    v = np.zeros((2, 3))
    with pytest.raises(ValueError, match="joints"):
        skin.chain_weights(v, ["a", "b"], [[0, 0, 0], [0, 0, 1]])
    with pytest.raises(ValueError, match="duplicate"):
        skin.chain_weights(v, ["a", "a"], [[0, 0, 0], [0, 0, 1], [0, 0, 2]])
    with pytest.raises(ValueError, match="blend"):
        skin.chain_weights(v, ["a", "b"], [[0, 0, 0], [0, 0, 1], [0, 0, 2]], blend=[0.1, 0.1])
    with pytest.raises(ValueError, match="blend"):
        skin.chain_weights(v, ["a", "b"], [[0, 0, 0], [0, 0, 1], [0, 0, 2]], blend=-1.0)
    with pytest.raises(ValueError, match="coincide"):
        skin.chain_weights(v, ["a", "b"], [[0, 0, 0], [0, 0, 0], [0, 0, 0]])
    with pytest.raises(ValueError):
        skin.chain_weights(v, [], [[0, 0, 0]])


# ---------------------------------------------------------------- envelope
def _segments():
    return {"a": ([0, 0, 0], [0, 0, 1]), "b": ([0.5, 0, 0], [0.5, 0, 1]), "c": ([10, 0, 0], [10, 0, 1])}


def test_envelope_prefers_the_nearest_bone():
    v = np.array([[0.1, 0, 0.5], [0.0, 0, 0.5], [0.25, 0, 0.5], [0.45, 0, 2.0]])
    w = skin.envelope_weights(v, _segments(), k=2)
    assert list(w) == ["a", "b"]                                       # 'c' is never among the two nearest
    assert np.allclose(_total(w), 1.0)
    assert w["a"][0] == pytest.approx((0.1 ** -4) / (0.1 ** -4 + 0.4 ** -4), rel=1e-5) and w["a"][0] > 0.99
    assert w["a"][1] > 0.999999 and w["b"][1] < 1e-6
    assert w["a"][2] == pytest.approx(0.5) and w["b"][2] == pytest.approx(0.5)
    assert w["b"][3] > w["a"][3] > 0                                   # nearest end cap wins above the segments
    near = skin.envelope_weights(v, _segments(), k=1)
    assert near["a"][[0, 1, 3]].tolist() == [1.0, 1.0, 0.0]            # v[2] is equidistant: either bone may win
    assert near["b"][3] == 1.0 and np.allclose(near["a"] + near["b"], 1.0)
    pw = skin.envelope_weights(v[:1], _segments(), power=1.0, k=2)
    assert pw["a"][0] == pytest.approx((1 / 0.1) / (1 / 0.1 + 1 / 0.4), rel=1e-6)
    every = skin.envelope_weights(v, _segments(), k=9)                 # k larger than the number of bones
    assert set(every) == {"a", "b", "c"} and np.allclose(_total(every), 1.0)


def test_envelope_radius_global_and_per_bone():
    v = np.array([[0.1, 0, 0.5], [3.0, 3.0, 3.0], [0.45, 0, 0.5]])
    g = skin.envelope_weights(v, _segments(), radius=0.3)
    assert list(g) == ["a", "b"]
    assert np.allclose(g["a"], [1, 0, 0]) and np.allclose(g["b"], [0, 0, 1])    # beyond the radius: nothing
    assert np.allclose(_total(g), [1, 0, 1])                                       # vertex 1 is outside every radius
    fixed = skin.normalise(g, 3, fallback="root")
    assert np.allclose(fixed["root"], [0, 1, 0])
    p = skin.envelope_weights(v, _segments(), radius={"a": 0.05, "b": 1.0}, k=1)  # 'c' is unlimited
    assert np.allclose(p["b"], [1, 0, 1])                                      # 'a' is out of range, so b takes its slot
    assert np.allclose(p["c"], [0, 1, 0]) and "a" not in p                     # (3, 3, 3): only the unlimited 'c' is left
    with pytest.raises(ValueError):
        skin.envelope_weights(v, {})
    with pytest.raises(ValueError):
        skin.envelope_weights(v, _segments(), radius=-1.0)
    with pytest.raises(ValueError):
        skin.envelope_weights(v, _segments(), k=0)
    pt = skin.envelope_weights(np.array([[0.0, 0, 0.0]]), {"p": ([0, 0, 0], [0, 0, 0]), "q": ([1, 0, 0], [1, 0, 0])})
    assert pt["p"][0] > 0.999999                                               # a point bone (head == tail) works


def test_envelope_large_input_is_chunked_and_consistent():
    rng = np.random.default_rng(2)
    segs = {f"s{i}": (rng.uniform(-1, 1, 3), rng.uniform(-1, 1, 3)) for i in range(40)}
    v = rng.uniform(-1, 1, (3000, 3))
    w = skin.envelope_weights(v, segs, k=4)
    st = skin.check(w, 3000)
    assert st["max_bones_per_vertex"] <= 4 and st["ok"]
    j = 17                                                                     # recompute one vertex by brute force
    d = {}
    for b, (a, e) in segs.items():
        a, e = np.asarray(a), np.asarray(e)
        t = np.clip((v[j] - a) @ (e - a) / ((e - a) @ (e - a)), 0, 1)
        d[b] = np.linalg.norm(v[j] - a - t * (e - a))
    near = sorted(d, key=d.get)[:4]
    raw = np.array([1.0 / (d[b] ** 4 + 1e-9) for b in near])
    for b, r in zip(near, raw / raw.sum()):
        assert w[b][j] == pytest.approx(r, rel=1e-9)


# ---------------------------------------------------------------- triangulation and closest points
def test_triangulate_fans_keep_winding_and_face_index():
    faces = [[0, 1, 2], [3, 4, 5, 6], [7, 8, 9, 10, 11], [2, 1, 0]]
    tris, face = skin.triangulate(faces)
    assert tris.dtype == np.int64 and face.dtype == np.int64
    assert tris.tolist() == [[0, 1, 2], [3, 4, 5], [3, 5, 6], [7, 8, 9], [7, 9, 10], [7, 10, 11], [2, 1, 0]]
    assert face.tolist() == [0, 1, 1, 2, 2, 2, 3]
    t2, f2 = skin.triangulate(np.array([[0, 1, 2, 3], [4, 5, 6, 7]]))
    assert t2.tolist() == [[0, 1, 2], [0, 2, 3], [4, 5, 6], [4, 6, 7]] and f2.tolist() == [0, 0, 1, 1]
    t3, f3 = skin.triangulate([])
    assert t3.shape == (0, 3) and f3.shape == (0,)
    with pytest.raises(ValueError):
        skin.triangulate([[0, 1]])


def _soup(rng):
    verts = rng.uniform(-1, 1, (80, 3))
    faces = [[int(x) for x in rng.choice(80, 3, replace=False)] for _ in range(10)]
    faces += [[int(x) for x in rng.choice(80, 4, replace=False)] for _ in range(7)]
    faces += [[int(x) for x in rng.choice(80, 5, replace=False)] for _ in range(3)]
    return verts, faces


def test_closest_on_mesh_matches_brute_force_on_random_triangles_quads_ngons():
    rng = np.random.default_rng(3)
    verts, faces = _soup(rng)
    tris, face_of = skin.triangulate(faces)
    pts = [rng.uniform(-1.5, 1.5, (60, 3))]
    for t in tris[:12]:                                               # points that project onto vertices and edges
        a, b, c = verts[t]
        n = np.cross(b - a, c - a)
        n /= np.linalg.norm(n)
        side = np.cross(b - a, n)
        side /= np.linalg.norm(side)
        pts.append([a + 0.1 * n, a - 0.2 * (b - a) - 0.05 * n, (a + b) / 2 - 0.05 * side + 0.02 * n,
                    (a + b) / 2 + 0.05 * side + 0.02 * n, c + 0.4 * (c - a) + 0.1 * n])
    pts = np.vstack([np.asarray(p).reshape(-1, 3) for p in pts])
    ref = _brute_distance(pts, verts, faces)
    for method in ("brute", "tree", "auto"):
        q, tri, bary, face = skin.closest_on_mesh(pts, verts, faces, method=method)
        assert q.shape == (len(pts), 3) and tri.shape == face.shape == (len(pts),) and bary.shape == (len(pts), 3)
        assert np.allclose(np.linalg.norm(pts - q, axis=1), ref, atol=1e-12), method
        assert np.allclose(bary.sum(axis=1), 1.0) and bary.min() >= 0.0
        rebuilt = np.einsum("ij,ijk->ik", bary, verts[tris[tri]])
        assert np.allclose(rebuilt, q, atol=1e-12)                    # q really lies on the reported triangle
        assert np.array_equal(face, face_of[tri])
        assert face.max() < len(faces)
    a0, b0, c0 = verts[tris[0]]
    on_surface = np.array([a0, (a0 + b0) / 2, (a0 + b0 + c0) / 3])
    q = skin.closest_on_mesh(on_surface, verts, faces)[0]
    assert np.allclose(q, on_surface, atol=1e-12)                       # points on the surface stay put
    with pytest.raises(ValueError):
        skin.closest_on_mesh(pts, verts, [[0, 1, 99]])
    with pytest.raises(ValueError):
        skin.closest_on_mesh(pts, verts, [])
    with pytest.raises(ValueError):
        skin.closest_on_mesh(pts, verts, faces, method="nope")
    empty = skin.closest_on_mesh(np.zeros((0, 3)), verts, faces)
    assert empty[0].shape == (0, 3) and empty[1].shape == (0,) and empty[2].shape == (0, 3) and empty[3].shape == (0,)


def test_closest_on_mesh_degenerate_triangles_and_single_point():
    verts = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [5, 5, 5], [5, 5, 5], [5, 5, 5], [0, 1, 0], [0, 2, 0], [1, 1, 3]], float)
    faces = [[0, 1, 2], [3, 4, 5]]                                    # a collinear sliver and a collapsed triangle
    pts = np.array([[0.5, 1.0, 0.0], [3.0, 0.0, 0.0], [-1.0, -1.0, 0.0], [5.0, 5.0, 6.0], [4, 4, 4]])
    for method in ("brute", "tree"):
        q, tri, bary, _ = skin.closest_on_mesh(pts, verts, faces, method=method)
        assert np.isfinite(q).all() and np.isfinite(bary).all()
        assert np.allclose(q[0], [0.5, 0, 0]) and np.allclose(q[1], [2, 0, 0]) and np.allclose(q[2], [0, 0, 0])
        assert np.allclose(q[3], [5, 5, 5]) and np.allclose(q[4], [5, 5, 5]) and tri[3] == 1 and tri[0] == 0
    one = skin.closest_on_mesh([0.0, 0.0, 1.0], verts, [[0, 6, 7], [0, 1, 8]])     # a single point as a 3-vector
    assert one[0].shape == (1, 3)


def test_closest_on_mesh_cannot_miss_large_triangles_among_tiny_ones():
    # a huge triangle whose centroid is far away, plus 100 tiny triangles that crowd every centroid-based candidate list
    rng = np.random.default_rng(1)
    centre = np.array([4.0, -4.0, 0.05])
    verts = [[-5, -5, 0], [5, -5, 0], [0, 5, 0]]
    faces = [[0, 1, 2]]
    for _ in range(100):
        base = len(verts)
        p0 = centre + rng.uniform(-0.03, 0.03, 3)
        verts += [p0, p0 + [0.004, 0, 0], p0 + [0, 0.004, 0]]
        faces.append([base, base + 1, base + 2])
    verts = np.array(verts)
    pts = np.array([[4.0, -4.5, 0.001], [3.9, -4.2, 0.002], [0.0, 0.0, 0.5], [4.0, -4.0, 0.06], [-4.0, 3.0, -0.2]])
    ref = _brute_distance(pts, verts, faces)
    for method in ("brute", "tree"):
        q, tri, _, face = skin.closest_on_mesh(pts, verts, faces, method=method)
        assert np.allclose(np.linalg.norm(pts - q, axis=1), ref, atol=1e-12)
    assert face[0] == 0 and face[1] == 0 and face[2] == 0 and face[4] == 0 and face[3] != 0
    # long thin triangles: a 'fence' of 2 m slivers next to a fine patch
    fence_v = [[0, y, 0] for y in np.linspace(0, 1, 9)] + [[2, y, 0] for y in np.linspace(0, 1, 9)]
    fence_f = [[i, i + 1, 9 + i + 1, 9 + i] for i in range(8)]
    pv, pf = _grid(10, 10)
    pv = pv * 0.1 + [1.0, 0.45, 0.01]
    allv = np.vstack([fence_v, pv])
    allf = fence_f + [[i + 18 for i in f] for f in pf]
    pts = np.random.default_rng(7).uniform([-0.5, -0.2, -0.3], [2.5, 1.2, 0.3], (300, 3))
    qt = skin.closest_on_mesh(pts, allv, allf, method="tree")[0]
    ref = _brute_distance(pts[:60], allv, allf)
    assert np.allclose(np.linalg.norm(pts[:60] - qt[:60], axis=1), ref, atol=1e-12)
    qb = skin.closest_on_mesh(pts, allv, allf, method="brute")[0]
    assert np.allclose(qt, qb, atol=1e-9)


def test_closest_on_mesh_tree_equals_brute_on_a_larger_surface():
    rng = np.random.default_rng(11)
    v, f = _grid(30, 30)
    v[:, 2] = 0.05 * np.sin(8 * v[:, 0]) * np.cos(6 * v[:, 1])
    pts = rng.uniform([-0.1, -0.1, -0.1], [1.1, 1.1, 0.4], (400, 3))
    qb, tb, bb, fb = skin.closest_on_mesh(pts, v, f, method="brute")
    qt, tt, bt, ft = skin.closest_on_mesh(pts, v, f, method="tree")
    assert np.allclose(np.linalg.norm(pts - qb, axis=1), np.linalg.norm(pts - qt, axis=1), atol=1e-12)
    assert np.allclose(qb, qt, atol=1e-9)


def test_closest_on_mesh_closed_surface_is_unsigned_and_winding_independent():
    cube_v = np.array([[x, y, z] for z in (0, 1) for y in (0, 1) for x in (0, 1)], float)
    cube_f = [[0, 2, 3, 1], [4, 5, 7, 6], [0, 1, 5, 4], [2, 6, 7, 3], [0, 4, 6, 2], [1, 3, 7, 5]]
    pts = np.array([[0.5, 0.5, 0.5], [0.2, 0.5, 0.5], [2.0, 0.5, 0.5], [2.0, 2.0, 2.0], [0.5, 1.5, 1.5]])
    q, tri, bary, face = skin.closest_on_mesh(pts, cube_v, cube_f)
    assert np.allclose(np.linalg.norm(pts - q, axis=1), [0.5, 0.2, 1.0, np.sqrt(3.0), np.hypot(0.5, 0.5)])
    assert np.allclose(q[3], [1, 1, 1]) and np.allclose(q[4], [0.5, 1, 1]) and np.allclose(q[1], [0, 0.5, 0.5])
    assert face[1] == 4 and face[2] == 5                                  # the x = 0 and x = 1 faces
    flipped = skin.closest_on_mesh(pts, cube_v, [f[::-1] for f in cube_f])[0]
    assert np.allclose(flipped, q)


def _same(a, b):
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(_same(a[k], b[k]) for k in a)
    if isinstance(a, (tuple, list)):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    a, b = np.asarray(a), np.asarray(b)
    return a.shape == b.shape and (np.array_equal(a, b) if a.dtype.kind in "iub" else np.allclose(a, b, atol=1e-12))


def test_chunked_paths_give_the_same_results(monkeypatch):
    rng = np.random.default_rng(21)
    verts, faces = _soup(rng)
    pts = rng.uniform(-1.5, 1.5, (90, 3))
    joints = rng.normal(size=(5, 3)) * 0.5
    segs = {f"s{i}": (rng.normal(size=3), rng.normal(size=3)) for i in range(7)}

    def run():
        return (skin.closest_on_mesh(pts, verts, faces, method="tree"), skin.closest_on_mesh(pts, verts, faces, method="brute"),
                skin.chain_param(pts, joints), skin.chain_weights(pts, list("abcd"), joints, blend=0.3),
                skin.envelope_weights(pts, segs, radius=1.2, k=3))

    ref = run()
    monkeypatch.setattr(skin, "_PAIR_BUDGET", 50)                          # a handful of points per chunk
    monkeypatch.setattr(skin, "_POINT_CHUNK", 7)
    assert _same(ref, run())


def test_closest_on_mesh_without_scipy_uses_the_numpy_path(monkeypatch):
    rng = np.random.default_rng(8)
    verts, faces = _soup(rng)
    pts = rng.uniform(-1.5, 1.5, (50, 3))
    ref = skin.closest_on_mesh(pts, verts, faces, method="brute")
    monkeypatch.setattr(skin, "_kdtree", lambda: None)
    monkeypatch.setattr(skin, "_BRUTE_PAIRS", 0)                           # would pick the tree search with scipy
    assert _same(ref, skin.closest_on_mesh(pts, verts, faces))
    with pytest.raises(ImportError):
        skin.closest_on_mesh(pts, verts, faces, method="tree")


def test_empty_inputs_and_tuple_faces():
    v, f = _grid(2, 2)
    assert skin.triangulate([tuple(x) for x in f])[0].shape == (8, 3)
    assert skin.transfer(v, f, {"a": np.ones(9)}, np.zeros((0, 3)), fallback="r") == {}
    assert skin.laplacian_smooth({}, [], 0) == {}
    assert skin.laplacian_smooth({"a": np.ones(3)}, [], 3)["a"].tolist() == [1.0, 1.0, 1.0]   # no edges: unchanged
    assert skin.normalise({"a": np.ones(0)}, 0) == {}
    assert skin.chain_param(np.zeros((0, 3)), [[0, 0, 0], [0, 0, 1]])[0].shape == (0,)
    assert skin.envelope_weights(np.zeros((0, 3)), {"a": ([0, 0, 0], [0, 0, 1])}) == {}
    assert skin.sphere_ramp(np.zeros((0, 3)), [0, 0, 0], 1, 2).shape == (0,)
    assert skin.closest_on_mesh([], v, f)[0].shape == (0, 3)


# ---------------------------------------------------------------- transfer
def _two_bone_tube(rings, segs, radius=0.1, phase=0.0):
    verts, faces = _tube(rings, segs, radius=radius, phase=phase)
    w = skin.chain_weights(verts, ["lo", "hi"], [[0, 0, 0], [0, 0, 0.5], [0, 0, 1]], blend=0.3)
    return verts, faces, w


def test_transfer_reproduces_source_weights_on_an_offset_copy():
    sv, sf, sw = _two_bone_tube(21, 16)
    dv, df, dw = _two_bone_tube(33, 24, radius=0.103, phase=0.07)           # other tessellation, 3 mm outside
    assert sw["lo"].min() == 0.0 and sw["hi"].max() == 1.0
    r = skin.transfer(sv, sf, sw, dv)
    assert set(r) == {"lo", "hi"} and all(x.shape == (len(dv),) for x in r.values())
    assert np.allclose(_total(r), 1.0)
    err = np.abs(r["lo"] - dw["lo"])
    assert err.max() < 0.03 and err.mean() < 0.005
    st = skin.check(r, len(dv), bones=["lo", "hi"])
    assert st["ok"] and st["max_bones_per_vertex"] <= 2 and not st["unknown_bones"]
    same = skin.transfer(sv, sf, sw, sv)                                    # onto its own vertices: exact (to the floor)
    assert np.abs(same["lo"] - sw["lo"]).max() < 2e-4
    flat = skin.transfer(sv, np.array(skin.triangulate(sf)[0]), sw, dv[:50])  # triangle array as faces
    assert np.allclose(flat["lo"], r["lo"][:50])


def test_transfer_max_dist_and_fallback_blend():
    sv, sf, sw = _two_bone_tube(21, 16)
    z = 0.25                                                               # a ring of the tube; lo = 1 up there
    ang = 2 * np.pi * np.arange(5) / 16                                    # radially above source vertices
    out = np.array([0.0, 0.03, 0.075, 0.09, 0.3])                          # distance from the tube surface
    dv = np.stack([(0.1 + out) * np.cos(ang), (0.1 + out) * np.sin(ang), np.full(5, z)], axis=1)
    r = skin.transfer(sv, sf, sw, dv, max_dist=0.05, fallback="root")
    assert np.allclose(r["lo"][:2], 1.0) and np.allclose(r["root"][:2], 0.0, atol=1e-12)    # within max_dist: untouched
    assert r["root"][2] == pytest.approx(0.5) and r["lo"][2] == pytest.approx(0.5)          # one and a half max_dist
    assert r["root"][3] == pytest.approx(float(skin.smoothstep(0.8)))                       # smooth in between
    assert r["root"][4] == 1.0 and r["lo"][4] == 0.0                                        # beyond 2 * max_dist
    assert np.allclose(_total(r), 1.0)
    plain = skin.transfer(sv, sf, sw, dv, max_dist=0.05)                   # no fallback: max_dist has no effect
    assert np.allclose(plain["lo"], 1.0) and list(plain) == ["lo"]
    plain2 = skin.transfer(sv, sf, sw, dv, fallback="root")
    assert np.allclose(plain2["lo"], 1.0)
    with pytest.raises(ValueError):
        skin.transfer(sv, sf, sw, dv, max_dist=0.0, fallback="root")
    with pytest.raises(ValueError, match="smooth_faces"):
        skin.transfer(sv, sf, sw, dv, smooth_iters=2)


def test_transfer_handles_unweighted_source_vertices_and_caps_bones():
    verts, faces = _grid(2, 1)                                             # 3 x 2 vertices, two quads
    w = {f"{kind}{i}": np.zeros(6) for i in range(6) for kind in "pr"}
    for i in range(6):
        w[f"p{i}"][i] = 0.6                                                # two own bones per vertex
        w[f"r{i}"][i] = 0.4
    del w["p5"], w["r5"]                                                   # vertex 5 (x=1, y=1) is unweighted
    dst = np.array([[0.275, 0.35, 0.1], [1.0, 1.0, 0.0], [0.9, 0.9, 0.0]])
    r = skin.transfer(verts, faces, w, dst, fallback="root")
    assert skin.check(r, 3)["ok"] and skin.check(r, 3)["max_bones_per_vertex"] <= 4
    # point 0 sits in triangle (0, 1, 4) with barycentrics (0.45, 0.2, 0.35): six bones, the four largest are kept
    assert sorted(b for b in r if r[b][0] > 0) == ["p0", "p4", "r0", "r4"]
    assert r["p0"][0] == pytest.approx(0.27 / 0.8) and r["r4"][0] == pytest.approx(0.14 / 0.8)
    # point 2 is next to the unweighted corner: that corner counts as absent and the others are renormalised
    assert sorted(b for b in r if r[b][2] > 0) == ["p1", "p4", "r1", "r4"]
    assert r["p1"][2] == pytest.approx(0.3) and r["r4"][2] == pytest.approx(0.2)
    assert np.allclose(r["root"], [0, 1, 0])                               # point 1 is on the corner itself
    with pytest.raises(ValueError, match="1 of 3"):
        skin.transfer(verts, faces, w, dst)


def test_transfer_smoothing_over_the_destination_mesh():
    sv, sf, sw = _two_bone_tube(21, 16)
    dv, df = _tube(17, 12, radius=0.102, phase=0.2)
    raw = skin.transfer(sv, sf, sw, dv)
    sm = skin.transfer(sv, sf, sw, dv, smooth_faces=df, smooth_iters=3, lam=0.5)
    assert np.allclose(_total(sm), 1.0)
    assert not np.allclose(raw["lo"], sm["lo"])
    assert skin.check(sm, len(dv))["ok"]
    # smoothing never leaves the 0..1 range and softens the steepest part of the cross-fade
    assert sm["lo"].max() <= 1.0 + 1e-12 and np.abs(np.diff(sm["lo"][::12])).max() <= np.abs(np.diff(raw["lo"][::12])).max() + 1e-9


def test_transfer_from_meshes_matches_transfer_on_the_merged_surface():
    sv, sf, sw = _two_bone_tube(21, 16)
    half = 10 * 16                                                          # faces of rings 0..10
    fa = sf[:half]
    fb = [[i - 10 * 16 for i in f] for f in sf[half:]]
    va, vb = sv[:11 * 16], sv[10 * 16:]
    fa = [list(f) for f in fa]
    a = Mesh("lower", va, fa, weights={"lo": sw["lo"][:11 * 16], "hi": sw["hi"][:11 * 16]})
    b = Mesh("upper", vb, fb, weights={"hi": sw["hi"][10 * 16:]})           # no 'lo' here: counts as 0
    dv, _ = _tube(13, 10, radius=0.104, phase=0.3)
    got = skin.transfer_from_meshes([a, b], dv)
    # reference: merged by hand (vertices of ring 10 appear in both meshes)
    mv = np.vstack([va, vb])
    mf = fa + [[i + len(va) for i in f] for f in fb]
    mw = {"lo": np.concatenate([sw["lo"][:11 * 16], np.zeros(len(vb))]),
          "hi": np.concatenate([sw["hi"][:11 * 16], sw["hi"][10 * 16:]])}
    want = skin.transfer(mv, mf, mw, dv)
    assert set(got) == set(want)
    assert all(np.allclose(got[k], want[k], atol=1e-12) for k in got)
    single = skin.transfer_from_meshes(a, dv[:20], fallback="root")          # a single Mesh is fine too
    assert np.allclose(_total(single), 1.0)
    with pytest.raises(ValueError):
        skin.transfer_from_meshes([], dv)


# ---------------------------------------------------------------- smoothing
def test_laplacian_smooth_keeps_sum_pins_vertices_and_counts_edges_once():
    verts, faces = _grid(6, 6)
    n = len(verts)
    a = (verts[:, 0] < 0.5).astype(float)
    w = {"a": a, "b": 1.0 - a}
    pinned = np.zeros(n, dtype=bool)
    pinned[::5] = True
    pinned[:7] = True
    s = skin.laplacian_smooth(w, faces, n, iterations=3, lam=0.5, pinned=pinned)
    assert np.allclose(_total(s), 1.0, atol=1e-12)
    assert np.array_equal(s["a"][pinned], a[pinned]) and np.array_equal(s["b"][pinned], 1.0 - a[pinned])
    assert 0.0 <= s["a"].min() and s["a"].max() <= 1.0
    mid = (np.abs(verts[:, 0] - 0.5) < 0.2) & ~pinned
    assert np.any((s["a"][mid] > 0) & (s["a"][mid] < 1))                  # the step got softened
    assert np.array_equal(w["a"], a)                                       # input untouched
    free = skin.laplacian_smooth(w, faces, n, iterations=0)
    assert np.array_equal(free["a"], a) and free["a"] is not w["a"]
    idx_pinned = skin.laplacian_smooth(w, faces, n, iterations=2, pinned=np.flatnonzero(pinned))
    assert np.array_equal(idx_pinned["a"][pinned], a[pinned])
    # two quads sharing an edge: vertex 4's neighbours are 3, 1, 5 (the shared edge 1-4 counts once)
    quad_faces = [[0, 1, 4, 3], [1, 2, 5, 4]]
    one = {"a": np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0]), "b": np.array([1.0, 0.0, 1.0, 1.0, 1.0, 1.0])}
    r = skin.laplacian_smooth(one, quad_faces, 6, iterations=1, lam=1.0)
    assert r["a"][4] == pytest.approx(1.0 / 3.0) and r["a"][0] == pytest.approx(0.5)   # vertex 0: neighbours 1 and 3
    iso = {"a": np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]), "b": np.array([1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0])}
    r = skin.laplacian_smooth(iso, quad_faces, 7, iterations=2, lam=1.0)     # vertex 6 has no edges
    assert r["a"][6] == 1.0 and r["b"][6] == 0.0
    with pytest.raises(ValueError):
        skin.laplacian_smooth(one, quad_faces, 6, lam=1.5)
    with pytest.raises(ValueError):
        skin.laplacian_smooth(one, [[0, 1, 9]], 6)
    with pytest.raises(ValueError):
        skin.laplacian_smooth(one, quad_faces, 6, pinned=np.zeros(3, dtype=bool))


def test_laplacian_smooth_spreads_new_bones_and_drops_empty_ones():
    verts, faces = _grid(4, 1)
    n = len(verts)
    w = {"a": np.zeros(n), "b": np.zeros(n)}
    w["a"][0] = 1.0
    w["b"][1:] = 1.0
    s = skin.laplacian_smooth(w, faces, n, iterations=1)
    assert set(s) == {"a", "b"} and s["a"][1] > 0 and np.allclose(_total(s), 1.0)
    z = skin.laplacian_smooth({"a": np.zeros(n), "b": np.ones(n)}, faces, n, iterations=2)
    assert list(z) == ["b"] and np.allclose(z["b"], 1.0)


# ---------------------------------------------------------------- check
def test_check_reports_everything():
    w = {"a": np.array([1.0, 0.5, 0.0, 0.2, 0.3]), "b": np.array([0.0, 0.5, 0.0, -0.1, 0.3]),
         "c": np.array([0.0, 0.0, 0.0, 0.0, 0.3]), "d": np.array([0.0, 0.0, 0.0, 0.0, 0.1]),
         "e": np.array([0.0, 0.0, 0.0, 0.0, 0.1])}
    st = skin.check(w, 5, bones=["a", "b", "c", "d"])
    assert st["max_bones_per_vertex"] == 5
    assert st["unweighted"] == 1                                       # vertex 2
    assert st["sum_min"] == 0.0 and st["sum_max"] == pytest.approx(1.1)
    assert st["unknown_bones"] == ["e"]
    assert st["negative"] == 1 and st["nonfinite"] == 0 and st["n_bones"] == 5
    assert st["ok"] is False
    good = skin.normalise({k: v for k, v in w.items()}, 5, fallback="a")
    st = skin.check(good, 5, bones=list(good) + ["x"], cap=4)
    assert st["ok"] and st["unweighted"] == 0 and st["negative"] == 0 and st["unknown_bones"] == []
    assert st["sum_min"] == pytest.approx(1.0) and st["sum_max"] == pytest.approx(1.0)
    assert skin.check(w, 5)["unknown_bones"] == []                      # no skeleton given: nothing unknown
    assert skin.check({"a": np.full(3, 0.5)}, 3)["ok"] is False        # sums to 0.5
    assert skin.check({"a": np.full(3, 0.9995)}, 3, tol=1e-3)["ok"] is True
    capped = skin.check({"a": np.full(2, 0.5), "b": np.full(2, 0.5)}, 2, cap=1)
    assert capped["max_bones_per_vertex"] == 2 and capped["ok"] is False
    bad = skin.check({"a": np.array([1.0, np.nan])}, 2)
    assert bad["nonfinite"] == 1 and bad["ok"] is False and bad["unweighted"] == 1
    empty = skin.check({}, 4)
    assert empty["unweighted"] == 4 and empty["max_bones_per_vertex"] == 0 and empty["ok"] is False
    zero = skin.check({}, 0)
    assert zero["unweighted"] == 0 and zero["ok"] is True and zero["max_bones_per_vertex"] == 0
    with pytest.raises(ValueError):
        skin.check({"a": np.ones(4)}, 3)
