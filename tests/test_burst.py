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
    assert {k: sum(p["kind"] == k for p in L) for k in BU.KINDS} == {"heart": 11, "star": 8, "puff": 5, "shard": 0,
                                                                     "sherd": 0, "drop": 0}
    again, other = BU.layout(24, 3, BU.MIX), BU.layout(24, 4, BU.MIX)
    assert all((a["dir"] == b["dir"]).all() and a["kind"] == b["kind"] for a, b in zip(L, again))
    assert any((a["dir"] != b["dir"]).any() for a, b in zip(L, other))
    assert {p["kind"] for p in BU.layout(5, 1, {"star": 1.0})} == {"star"}
    assert {p["kind"] for p in BU.layout(6, 1, {"shard": 1.0})} == {"shard"}       # glass alone when the mix says so


def test_pieces_fly_across_the_picture_more_than_toward_the_camera():
    D = np.array([p["dir"] for p in BU.layout(60, 2, BU.MIX)])
    assert np.abs(D[:, 1]).max() <= BU.FLAT + 1e-9                       # toward or away from the lens, less far
    assert D[:, 0].min() < -0.8 and D[:, 0].max() > 0.8 and D[:, 2].min() < -0.9 and D[:, 2].max() > 0.9


FORMS = [(None, BU.FADE, None), (540.0, 1.0, [0.0, 0.0, 1.0])]          # (tumble, fade, aim): as built by default; a smash


@pytest.mark.parametrize("tumble, fade, aim", FORMS)
@pytest.mark.parametrize("P", [dict(BU.options({})[3], start=2.0),
                               dict(BU.options({})[3], start=-0.5, reach=0.4, life=0.6, spin=-90.0, size=0.3, amount=0.5),
                               dict(BU.options({})[3], start=1.0, gravity=9.8, ground=-0.05)])
def test_expressions_equal_the_numpy_reference(P, tumble, fade, aim):
    for piece in BU.layout(12, 7, BU.MIX, aim, 70.0):
        ex = BU.expressions(piece, tumble, fade)
        for t in np.linspace(P["start"] - 0.5, P["start"] + P["life"] * 1.3, 23):
            got = _run(ex, P, float(t))
            pos, rx, ry, s = BU.pose(float(t), P, piece, tumble, fade)
            assert [got["x"], got["y"], got["z"]] == pytest.approx(pos, abs=1e-6)
            assert (got["rx"], got["ry"], got["s"]) == pytest.approx((rx, ry, s), abs=1e-6)


def test_a_smash_flies_up_and_out_in_its_cone_falls_back_to_the_ground_and_lies_there():
    """A cup smashing on the floor: the pieces go out in a cone about the aim (up), come down under gravity and never
    under the ground they stay on, tumbling as they fly and keeping their size once they lie there (fade 1)."""
    count, seed, mix, P, _, _, form = BU.options({"mix": {"sherd": 0.8, "drop": 0.2}, "aim": [0, 0, 2], "cone": 70,
                                                  "tumble": 540, "fade": 1, "gravity": 9.8, "ground": 0.0,
                                                  "reach": 0.8, "life": 1.2, "start": 1.0})
    assert form["aim"] == [0.0, 0.0, 1.0] and form["tumble"] == 540.0 and form["fade"] == 1.0
    L = BU.layout(count, seed, mix, form["aim"], form["cone"])
    assert {p["kind"] for p in L} == {"sherd", "drop"}
    D = np.array([p["dir"] for p in L])
    assert np.linalg.norm(D, axis=1) == pytest.approx(1.0, abs=1e-4)
    assert D[:, 2].min() >= math.cos(math.radians(70)) - 1e-4               # all inside the cone ...
    assert np.hypot(D[:, 0], D[:, 1]).max() > 0.9 and D[:, 2].max() > 0.95  # ... and over all of it
    ts = np.linspace(1.0, 2.6, 81)
    rise = []
    for p in L:
        z = BU.pose(ts, P, p, form["tumble"], form["fade"])[0][:, 2]
        assert z.min() >= 0.0 and z[-1] == 0.0                               # never into the floor, and back down on it
        rise.append(z.max())
        end = BU.pose(ts[-1], P, p, form["tumble"], form["fade"])
        assert end[3] == pytest.approx(P["size"] * p["size"])                # lying there, whole
        assert abs(end[1]) == pytest.approx(math.radians(540.0))             # turned end over end on the way
    assert max(rise) > 0.05 and np.median(rise) > 0.005                      # up off the floor on the way


def test_expressions_are_blender_simple_expressions():
    ok = set(BU.PARAMS) | set(BU.HELPERS) | BU.FUNCS | {"frame", "f0", "fps"}
    for tumble, fade, aim in FORMS:
        for piece in BU.layout(30, 1, BU.MIX, aim):
            for text in BU.expressions(piece, tumble, fade).values():
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
    ({"gravity": -9.8}, "gravity must be a number"),
    ({"glow": -1}, "glow is a number"),
    ({"aim": [0, 0, 0]}, "aim is the way"),
    ({"aim": "up"}, "aim is the way"),
    ({"cone": 0}, "cone is the half-angle"),
    ({"tumble": -10}, "tumble is the degrees"),
    ({"fade": 1.5}, "fade is the share"),
])
def test_a_bad_burst_is_refused_saying_what_to_change(slots, frag):
    with pytest.raises(ValueError, match=frag):
        BU.options(slots)


@pytest.mark.parametrize("kind, tips", [("heart", None), ("star", 5), ("puff", 6), ("sherd", None), ("drop", None)])
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
