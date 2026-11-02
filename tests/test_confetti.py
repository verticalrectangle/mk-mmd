"""Confetti, bpy-free (mkmmd.core.confetti): the volleys' layout (kinds by mix, colours in turn, directions in the cone,
all from the seed), the driver expressions against the numpy reference, a piece's life (hidden, popping in, flying out,
falling, gone after its life or at the floor), the bounds, the refusals and the paper meshes. The Blender half (the
drivers on the objects, the keys stage dealing `start` round the volleys) is tests/test_confetti_build.py."""
import math
import re

import numpy as np
import pytest

from mkmmd.core import confetti as CF
from mkmmd.core import shell as S


def _clamp(x, lo, hi):
    return max(lo, min(hi, x))


def _smoothstep(a, b, x):
    t = _clamp((x - a) / (b - a), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


BLENDER = {"min": min, "max": max, "clamp": _clamp, "smoothstep": _smoothstep, "sin": math.sin, "cos": math.cos,
           "exp": math.exp, "abs": abs, "radians": math.radians, "sqrt": math.sqrt}
SIMPLE = {"exp", "log", "pow", "floor", "ceil", "round", "trunc", "fmod", "sin", "cos", "tan", "asin", "atan", "atan2",
          "sqrt", "abs", "min", "max", "clamp", "lerp", "smoothstep", "radians", "degrees"}    # verified simple in 4.2
F0, FPS = 31.0, 30.0
CHANNELS = ("x", "y", "z", "rx", "ry", "rz", "s", "sz")


def _run(ex, P, t):
    """The expressions of one piece at clip second t, as Blender evaluates them (the helpers a, g, z first)."""
    env = dict(BLENDER, **P, f0=F0, fps=FPS, frame=F0 + t * FPS)
    for h in CF.HELPERS:
        env[h] = eval(ex[h], {"__builtins__": {}}, env)                                    # noqa: S307
    out = {k: eval(v, {"__builtins__": {}}, env) for k, v in ex.items() if k not in CF.HELPERS}   # noqa: S307
    out.update({h: env[h] for h in CF.HELPERS})
    out.setdefault("sz", out["s"])
    return out


def _counts(pieces, key):
    out = {}
    for p in pieces:
        out[p[key]] = out.get(p[key], 0) + 1
    return out


def test_the_kinds_follow_the_mix_the_colours_take_turns_and_the_layout_is_the_seed_s():
    O = CF.options({"count": 20, "volleys": 2, "seed": 3})
    L = CF.layout(O)
    assert len(L) == 40 and [p["volley"] for p in L] == [0] * 20 + [1] * 20
    for k in range(2):
        vol = [p for p in L if p["volley"] == k]
        assert _counts(vol, "kind") == {"square": 11, "ribbon": 3, "heart": 3, "star": 3}       # 20 x 0.55 / 0.15 ...
        assert sorted(_counts(vol, "colour").values()) == [2] * 1 + [3] * 6                     # 20 over 7 colours
    again, other = CF.layout(O), CF.layout(CF.options({"count": 20, "volleys": 2, "seed": 4}))
    assert all((a["v"] == b["v"]).all() and a["kind"] == b["kind"] and a["delay"] == b["delay"] for a, b in zip(L, again))
    assert any((a["v"] != b["v"]).any() for a, b in zip(L, other))
    assert any((a["v"] != b["v"]).any() for a, b in zip(L[:20], L[20:]))                   # each volley its own
    assert {p["kind"] for p in CF.layout(CF.options({"count": 7, "mix": {"star": 1.0}}))} == {"star"}
    ribbons = CF.layout(CF.options({"count": 4, "volleys": 1, "mix": {"ribbon": 1.0}, "colors": ["gold"]}))
    assert {p["kind"] for p in ribbons} == {"ribbon"} and {p["colour"] for p in ribbons} == {0}


@pytest.mark.parametrize("aim, cone", [(CF.AIM, CF.CONE), ((1.0, 0.0, 0.0), 10.0), ((0.0, 0.0, 1.0), 60.0)])
def test_the_pieces_fly_out_in_the_cone_about_aim_and_a_streamer_leaves_along_its_flight(aim, cone):
    O = CF.options({"count": 80, "volleys": 1, "aim": list(aim), "cone": cone})
    a = np.array(aim) / np.linalg.norm(aim)
    ang = []
    for p in CF.layout(O):
        d = p["v"] / np.linalg.norm(p["v"])
        ang.append(math.degrees(math.acos(min(1.0, float(d @ a)))))
        if p["kind"] == "ribbon":                                       # its curl (local Z) along its flight at the mouth
            R = S.rot_matrix(*np.degrees(p["rot0"]))
            assert R @ [0.0, 0.0, 1.0] == pytest.approx(d, abs=1e-4)
    assert max(ang) <= cone + 0.01 and max(ang) > 0.85 * cone                  # inside the cone, out to its edge
    D = CF.cone(500, aim, cone)                                               # evenly: equal solid angle each
    c = np.degrees(np.arccos(np.clip(D @ a, -1.0, 1.0)))
    inner = np.degrees(np.arccos(1.0 - 0.5 * (1.0 - math.cos(math.radians(cone)))))   # the half-solid-angle cap
    assert abs(int((c < inner).sum()) - 250) <= 2                               # half the directions in half the cap


def test_euler_xyz_is_blender_s_xyz_order():
    for ang in ((10.0, -40.0, 75.0), (-170.0, 20.0, 5.0), (33.0, 80.0, -120.0)):
        M = S.rot_matrix(*ang)
        assert np.degrees(CF.euler_xyz(M)) == pytest.approx(ang, abs=1e-9)


def _keyed(P, starts):
    return dict(P, **{f"start{k}": v for k, v in enumerate(starts)})


@pytest.mark.parametrize("slots, starts", [
    ({"floor": -0.8}, (2.0, 3.5, CF.NEVER)),
    ({"reach": 1.5, "drag": 0.3, "fall": 1.2, "flutter": 0.3, "sway": 1.5, "spin": -2.0, "size": 0.08, "life": 3.0,
      "floor": -0.5, "amount": 0.6, "ribbon": 0.8}, (-0.5, 0.25, 1.0)),
])
def test_expressions_equal_the_numpy_reference(slots, starts):
    O = CF.options(dict(slots, count=8, volleys=3, seed=7))
    P = _keyed(CF.root_values(O), starts)
    for piece in CF.layout(O):
        ex = CF.expressions(piece)
        t0 = P[f"start{piece['volley']}"]
        if t0 >= CF.NEVER:
            t0 = 1.0                                                     # never: hidden at the mouth all along
        for t in np.linspace(t0 - 0.5, t0 + P["life"] * 1.2, 31):
            got = _run(ex, P, float(t))
            pos, rot, scale = CF.pose(float(t), P, piece)
            h = CF.motion(CF.age(float(t), P, piece), P, piece)[3]
            assert [got[c] for c in ("x", "y", "z")] == pytest.approx(pos, abs=1e-6)
            assert [got[c] for c in ("rx", "ry", "rz")] == pytest.approx(rot, abs=1e-6)
            assert [got["s"], got["s"], got["sz"]] == pytest.approx(scale, abs=1e-7)
            assert [got[c] for c in CF.HELPERS] == pytest.approx([float(h[c]) for c in CF.HELPERS], abs=1e-6)


def test_expressions_are_blender_simple_expressions():
    O = CF.options({"count": 40, "volleys": 4, "seed": 2})
    ok = set(CF.PARAMS) | {f"start{k}" for k in range(4)} | set(CF.HELPERS) | set(CF.FUNCS) | {"frame", "f0", "fps"}
    assert CF.FUNCS <= SIMPLE
    for piece in CF.layout(O):
        ex = CF.expressions(piece)
        assert list(ex)[:3] == list(CF.HELPERS) and set(ex) <= set(CF.HELPERS) | set(CHANNELS)
        assert ("sz" in ex) == (piece["kind"] == "ribbon")
        for text in ex.values():
            assert len(text) <= 255 and not re.search(r"\*\*|//|%|\.[A-Za-z_]|e[-+]?\d", text)
            assert set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", text)) <= ok


def test_a_piece_is_hidden_before_its_start_pops_in_flies_out_falls_and_is_gone_after_its_life():
    O = CF.options({"count": 12, "volleys": 2, "seed": 5})
    P = _keyed(CF.root_values(O), (1.0, CF.NEVER))
    piece = next(p for p in CF.layout(O) if p["volley"] == 0 and p["kind"] == "square")
    t0 = P["start0"] + piece["delay"]
    pos, _, sc = CF.pose(t0 - 0.01, P, piece)
    assert np.linalg.norm(pos) == 0 and (sc == CF.MIN_SCALE).all()                      # hidden at the mouth
    full = P["size"] * piece["share"] * P["amount"]
    assert CF.pose(t0 + 0.5 * CF.POP, P, piece)[2][0] == pytest.approx(0.5 * full)       # popping in
    assert CF.pose(t0 + CF.POP, P, piece)[2] == pytest.approx([full] * 3)                # full size at once
    ts = t0 + np.linspace(0.0, 0.4, 20)
    dist = np.linalg.norm(CF.pose(ts, P, piece)[0], axis=-1)
    assert (np.diff(dist) > 0).all() and dist[-1] > 0.5                                   # flying out
    z = CF.pose(t0 + np.linspace(2.5, 5.0, 20), P, piece)[0][:, 2]
    assert (np.diff(z) < 0).all()                                                         # then falling
    assert CF.pose(t0 + P["life"], P, piece)[2] == pytest.approx([CF.MIN_SCALE] * 3)       # gone after its life
    assert CF.pose(t0 + 0.95 * P["life"], P, piece)[2][0] < 0.5 * full                    # shrinking away by then
    rot = CF.pose(np.array([t0 + 1.0, t0 + 2.0]), P, piece)[1]
    assert rot[1] - rot[0] == pytest.approx(P["spin"] * piece["omega"])                   # a tumble at `spin`
    other = next(p for p in CF.layout(O) if p["volley"] == 1)
    assert CF.pose(np.linspace(0.0, 20.0, 50), P, other)[2] == pytest.approx(CF.MIN_SCALE)  # volley 1 never went off


def test_a_streamer_unrolls_along_its_curl_as_it_flies():
    O = CF.options({"count": 10, "volleys": 1, "mix": {"ribbon": 1.0}})
    P = CF.root_values(O)
    piece = CF.layout(O)[0]
    early, late = (CF.motion(a, P, piece) for a in (CF.POP, 2.0))
    assert early[2][2] / early[2][0] == pytest.approx(CF.UNROLL + (1 - CF.UNROLL) * early[3]["g"])
    assert early[2][2] / early[2][0] < 0.35 and late[2][2] / late[2][0] > 0.95


def test_a_piece_shrinks_away_at_the_floor_and_never_passes_it():
    O = CF.options({"count": 30, "volleys": 1, "seed": 9, "floor": -0.6, "life": 30.0})
    P = CF.root_values(O)
    a = np.linspace(0.0, 30.0, 3001)
    for piece in CF.layout(O):
        pos, _, sc, h = CF.motion(a, P, piece)
        r = CF.radius(piece["kind"], CF.length(O)) * sc.max(-1)
        shown = sc[:, 0] > 1.01 * CF.MIN_SCALE
        assert (pos[shown, 2] - r[shown] >= P["floor"] - 1e-9).all()                     # no part of it below the floor
        assert not shown[-1] and (sc[h["z"] <= P["floor"]] == CF.MIN_SCALE).all()          # gone once it gets there


def test_the_bounds_hold_the_whole_flight():
    for slots in ({}, {"floor": -0.7, "aim": [1.0, 0.0, 0.4], "cone": 40}):
        O = CF.options(dict(slots, count=25, volleys=2, seed=4))
        P, lo, hi = CF.root_values(O), *CF.bounds(O)
        a = np.linspace(0.0, P["life"], 400)
        for piece in CF.layout(O):
            pos, _, sc, _ = CF.motion(a, P, piece)
            shown = sc[:, 0] > 1.01 * CF.MIN_SCALE
            assert (pos[shown] >= lo - 1e-9).all() and (pos[shown] <= hi + 1e-9).all()
        if "floor" in slots:
            assert lo[2] == pytest.approx(-0.7)
    lo, hi = CF.bounds(CF.options({}))
    assert hi[2] > 2.5 and -lo[1] > 2.0 * hi[1] > 0                                       # up, and out toward -Y


def test_the_defaults_rise_two_to_three_metres_and_rain_down_over_about_five_seconds():
    O = CF.options({})
    P = CF.root_values(O)
    a = np.linspace(0.0, P["life"], 1301)
    peak, down = [], []
    for piece in CF.layout(O):
        z = CF.motion(a, P, piece)[0][:, 2]
        peak.append(z.max())
        below = np.nonzero(z < -1.0)[0]                                 # a mouth 1 m off the floor
        down.append(a[below[0]] - a[int(np.argmax(z))] if len(below) else np.inf)
    assert 1.9 < np.median(peak) and 2.5 < max(peak) < 3.0
    assert 4.0 < np.median(down) < 5.5


def test_root_values_and_deals():
    O = CF.options({"start": 2.5, "volleys": 3})
    P = CF.root_values(O)
    assert (P["start0"], P["start1"], P["start2"]) == (2.5, CF.NEVER, CF.NEVER) and "start3" not in P
    assert CF.deals(3) == [{"from": "start", "into": ["start0", "start1", "start2"]}]


@pytest.mark.parametrize("slots, frag", [
    ({"count": 0}, "count is a whole number 1 .. 200"),
    ({"count": 201}, "count is a whole number 1 .. 200"),
    ({"count": 2.5}, "count is a whole number"),
    ({"volleys": 5}, "volleys is a whole number 1 .. 4"),
    ({"seed": "a"}, "seed is a whole number"),
    ({"mix": {"confetti": 1.0}}, "mix is"),
    ({"mix": {"square": 0.0}}, "mix is"),
    ({"mix": {"square": -1.0, "star": 2.0}}, "mix is"),
    ({"colors": "love"}, "colors is a list"),
    ({"colors": []}, "colors is a list"),
    ({"aim": [0, 0, 0]}, "aim is the direction"),
    ({"aim": [0, 1]}, "aim is the direction"),
    ({"cone": 200}, "cone is a number 0 .. 180"),
    ({"reach": 0}, "reach is a number > 0"),
    ({"drag": -1}, "drag is a number > 0"),
    ({"life": 0}, "life is a number > 0"),
    ({"fall": -0.1}, "fall is a number >= 0"),
    ({"amount": True}, "amount is a number"),
    ({"floor": "low"}, "floor is a number"),
    ({"glow": 9}, "glow is a number 0 .. 5"),
    ({"ribbon": 0}, "ribbon is a number > 0"),
    ({"ribbon": 5.0, "size": 0.05}, "ribbon is at most 80 times size"),
])
def test_a_bad_confetti_is_refused_saying_what_to_change(slots, frag):
    with pytest.raises(ValueError, match=re.escape(frag)):
        CF.options(slots)


@pytest.mark.parametrize("kind", CF.KINDS)
def test_every_piece_is_a_closed_thin_paper_shape(kind):
    m = CF.mesh(kind, 10.0)
    assert m.is_closed() and m.volume() > 0                                                 # closed, outward
    size = m.size()
    if kind == "ribbon":                                       # a curl along Z: as long as asked, a helix round it
        assert size[2] == pytest.approx(10.0 + CF.RIB_W, abs=0.05)
        assert size[:2] == pytest.approx([2 * CF.RIB_R] * 2, abs=0.04)
        r = np.hypot(m.V[:, 0], m.V[:, 1])                    # on the curl's cylinder (the strip's edges a little off)
        assert r.min() > CF.RIB_R - CF.THICK and r.max() < CF.RIB_R + 0.03
    else:                                                      # a unit-wide die-cut facing -Y; a square gently curled
        assert size[0] == pytest.approx(1.0, abs=0.02)
        assert size[1] == pytest.approx(CF.THICK, abs=1e-6) if kind != "square" else 0.05 < size[1] < 0.1
        n = m.vertex_normals()
        face = n[np.abs(m.V[:, 0]) < 0.1]
        assert np.abs(face[:, 1]).max() > 0.99                                              # the faces look along Y
    assert CF.radius(kind, 10.0) == pytest.approx(np.linalg.norm(m.V, axis=1).max())
