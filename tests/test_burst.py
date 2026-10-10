"""A burst, bpy-free (mkmmd.core.burst): the pieces' layout, their driver expressions against the numpy reference, a piece's
life (hidden, popping in, flying out, gone) and the piece meshes. The Blender half (the drivers on the objects giving the
same places) is checked in tests/test_build_stages.py."""
import math
import re

import numpy as np
import pytest

from mkmmd.core import burst as BU


def _clamp(x, lo, hi):
    return max(lo, min(hi, x))


def _smoothstep(a, b, x):
    t = _clamp((x - a) / (b - a), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


BLENDER = {"min": min, "max": max, "clamp": _clamp, "smoothstep": _smoothstep, "sin": math.sin, "cos": math.cos,
           "abs": abs, "radians": math.radians, "sqrt": math.sqrt, "fmod": math.fmod}
F0, FPS = 31.0, 30.0


def _run(ex, P, t):
    """The expressions of one piece at clip second t, as Blender evaluates them (the helper u first)."""
    env = dict(BLENDER, **P, f0=F0, fps=FPS, frame=F0 + t * FPS)
    env["u"] = eval(ex["u"], {"__builtins__": {}}, env)                                    # noqa: S307
    return {k: eval(v, {"__builtins__": {}}, env) for k, v in ex.items() if k != "u"}      # noqa: S307


def test_the_kinds_follow_the_mix_and_the_layout_is_the_seed_s():
    L = BU.layout(24, 3, BU.MIX)
    assert {k: sum(p["kind"] == k for p in L) for k in BU.KINDS} == {"heart": 11, "star": 8, "puff": 5}
    again, other = BU.layout(24, 3, BU.MIX), BU.layout(24, 4, BU.MIX)
    assert all((a["dir"] == b["dir"]).all() and a["kind"] == b["kind"] for a, b in zip(L, again))
    assert any((a["dir"] != b["dir"]).any() for a, b in zip(L, other))
    assert {p["kind"] for p in BU.layout(5, 1, {"star": 1.0})} == {"star"}


def test_pieces_fly_across_the_picture_more_than_toward_the_camera():
    D = np.array([p["dir"] for p in BU.layout(60, 2, BU.MIX)])
    assert np.abs(D[:, 1]).max() <= BU.FLAT + 1e-9                       # toward or away from the lens, less far
    assert D[:, 0].min() < -0.8 and D[:, 0].max() > 0.8 and D[:, 2].min() < -0.9 and D[:, 2].max() > 0.9


@pytest.mark.parametrize("P", [dict(BU.options({})[3], start=2.0),
                               dict(BU.options({})[3], start=-0.5, reach=0.4, life=0.6, spin=-90.0, size=0.3, amount=0.5)])
def test_expressions_equal_the_numpy_reference(P):
    for piece in BU.layout(12, 7, BU.MIX):
        ex = BU.expressions(piece)
        for t in np.linspace(P["start"] - 0.5, P["start"] + P["life"] * 1.3, 23):
            got = _run(ex, P, float(t))
            pos, rx, ry, s = BU.pose(float(t), P, piece)
            assert [got["x"], got["y"], got["z"]] == pytest.approx(pos, abs=1e-6)
            assert (got["rx"], got["ry"], got["s"]) == pytest.approx((rx, ry, s), abs=1e-6)


def test_expressions_are_blender_simple_expressions():
    ok = set(BU.PARAMS) | set(BU.HELPERS) | BU.FUNCS | {"frame", "f0", "fps"}
    for piece in BU.layout(30, 1, BU.MIX):
        for text in BU.expressions(piece).values():
            assert len(text) <= 255 and not re.search(r"\*\*|//|%|\.[A-Za-z_]", text)
            assert set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", text)) <= ok


def test_a_piece_is_hidden_before_the_blast_pops_in_flies_out_and_is_gone_after_its_life():
    P = dict(BU.options({})[3], start=1.0)
    piece = BU.layout(8, 1, BU.MIX)[3]
    t0 = P["start"] + piece["delay"] * P["life"]
    before, _, _, s_before = BU.pose(t0 - 0.01, P, piece)
    assert np.linalg.norm(before) == 0 and s_before == BU.MIN_SCALE
    full = P["size"] * piece["size"] * P["amount"]
    assert BU.pose(t0 + BU.POP * P["life"], P, piece)[3] == pytest.approx(full)             # popped in, full size
    ts = np.linspace(t0, t0 + P["life"], 40)
    dist = np.linalg.norm(BU.pose(ts, P, piece)[0], axis=-1)
    assert (np.diff(dist) >= -1e-12).all()                                                    # only ever outward
    end = BU.pose(t0 + P["life"], P, piece)
    assert end[0] == pytest.approx(piece["dir"] * P["reach"] * piece["reach"]) and end[3] == BU.MIN_SCALE   # out, gone
    assert BU.pose(t0 + 0.9 * P["life"], P, piece)[3] < 0.5 * full                         # shrinking away by then


@pytest.mark.parametrize("slots, frag", [
    ({"count": 0}, "count is a whole number"),
    ({"count": 2.5}, "count is a whole number"),
    ({"mix": {"confetti": 1.0}}, "mix is"),
    ({"mix": {"heart": 0.0}}, "mix is"),
    ({"reach": 0}, "reach must be a positive number"),
    ({"life": -1}, "life must be a positive number"),
    ({"amount": -0.5}, "amount must be a number"),
    ({"glow": -1}, "glow is a number"),
])
def test_a_bad_burst_is_refused_saying_what_to_change(slots, frag):
    with pytest.raises(ValueError, match=frag):
        BU.options(slots)


@pytest.mark.parametrize("kind, tips", [("heart", None), ("star", 5), ("puff", 6)])
def test_every_piece_is_a_closed_pillow_of_unit_width_facing_out(kind, tips):
    m = BU.mesh(kind)
    edges = {}
    for f in list(m.Q) + list(m.T):
        for a, b in zip(f, np.roll(f, -1)):
            edges[tuple(sorted((int(a), int(b))))] = edges.get(tuple(sorted((int(a), int(b)))), 0) + 1
    assert set(edges.values()) == {2} and m.volume() > 0                                     # closed, outward
    assert np.ptp(m.V[:, 0]) == pytest.approx(1.0, abs=0.01)
    if tips:
        O = BU.star_outline() if kind == "star" else BU.puff_outline()
        r = np.linalg.norm(O, axis=1)
        assert int(((r > np.roll(r, 1)) & (r > np.roll(r, -1))).sum()) == tips              # its points, its lobes
