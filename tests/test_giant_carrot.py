"""The giant carrot's layout (mkmmd.blender.library.props.giant_carrot_layout and _parts, no bpy): closed outward meshes in
the card's frame (the tip at the origin, +Z to the shoulder), the card's numbers and the grip's profile read off the mesh,
the bites eating the end from the tip up through non-additive shape keys, the chunks' flight, and every driver expression
a Blender simple expression that gives the numpy maths. The Blender half (the drivers in a real build) is
tests/test_giant_carrot_build.py."""
import math
import re

import numpy as np
import pytest

from mkmmd.blender.library.props import giant_carrot_layout as LAY
from mkmmd.blender.library.props import giant_carrot_parts as PT


def _clamp(x, lo, hi):
    return max(lo, min(hi, x))


def _smoothstep(a, b, x):
    t = _clamp((x - a) / (b - a), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


BLENDER = {"min": min, "max": max, "clamp": _clamp, "smoothstep": _smoothstep, "sin": math.sin, "cos": math.cos}
F0, FPS = 31.0, 30.0


def _env(t, bites, rx=0.0, ry=0.0):
    env = dict(BLENDER, f0=F0, fps=FPS, frame=F0 + t * FPS, rx=rx, ry=ry)
    env.update({f"bite{k}": b for k, b in enumerate(bites, 1)})
    return env


def _eval(text, env):
    return eval(text, {"__builtins__": {}}, env)                                      # noqa: S307


def _parts(m):
    """Vertex sets of the mesh's connected parts."""
    n = len(m.V)
    parent = np.arange(n)

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for f in list(m.Q) + list(m.T):
        r0 = find(int(f[0]))
        for v in f[1:]:
            r = find(int(v))
            if r != r0:
                parent[r] = r0
    roots = np.array([find(i) for i in range(n)])
    return [np.nonzero(roots == r)[0] for r in np.unique(roots)]


def _sub(m, ids):
    """The faces of `m` on the vertices `ids` as a mesh of their own."""
    keep = np.zeros(len(m.V), bool)
    keep[ids] = True
    remap = np.cumsum(keep) - 1
    Q, T = m.Q[keep[m.Q].all(1)], m.T[keep[m.T].all(1)]
    return PT.S.Mesh(m.V[keep], remap[Q], remap[T])


def test_the_body_stands_on_its_tip_and_is_a_closed_outward_lathe():
    m, Z0, _ = PT.body()
    assert m.is_closed() and m.volume() > 0
    lo, hi = m.bbox()
    assert lo[2] == pytest.approx(0.0, abs=1e-12) and hi[2] == pytest.approx(LAY.L + LAY.DOME_H, abs=1e-9)
    assert m.V[0].tolist() == [0.0, 0.0, 0.0]                                       # the tip, on the axis
    r = np.hypot(m.V[:, 0], m.V[:, 1])
    assert r.max() == pytest.approx(LAY.R_S, abs=1e-6)                             # 0.10 across the shoulder
    assert abs(m.V[np.argmax(r), 2] - LAY.L) < 0.01                                 # widest at the shoulder
    shaft = np.isfinite(Z0) & (Z0 > 0.0)
    order = np.argsort(Z0[shaft])
    assert np.all(np.diff(LAY.radius(Z0[shaft][order])) >= -1e-12)                 # a cone tapering to its tip


def test_the_rings_and_dimples_only_cut_in_and_stay_skin():
    m, Z0, _ = PT.body()
    root = np.isfinite(Z0) & (Z0 > 0.002)
    rel = np.hypot(m.V[root, 0], m.V[root, 1]) / LAY.radius(Z0[root])
    assert rel.max() <= 1.0 + 1e-12                                                 # nothing outside the profile
    assert rel.min() > LAY.SKIN_IN + 0.02                                           # the material still reads skin
    assert (rel < 0.995).sum() > 1000 and len(PT.grooves()) >= 10 and len(PT.dimples()) >= 6
    z = np.linspace(0.01, LAY.L - 0.01, 400)
    assert (PT.cut_in(z, 0.0) >= 0.0).all()


def test_the_grip_s_profile_is_the_mesh_s():
    m, Z0, _ = PT.body()
    g = LAY.hold_grip()
    assert g["type"] == "hold" and g["axis"] == [0.0, 0.0, -1.0]
    assert g["length"] == pytest.approx(LAY.HOLD_TO - LAY.HOLD_FROM) and g["length"] == pytest.approx(0.5, abs=0.02)
    assert g["center"] == pytest.approx([0.0, 0.0, LAY.HOLD_FROM + 0.5 * g["length"]], abs=1e-6)
    prof = np.array(g["profile"])
    assert len(prof) == 8 and prof[0, 0] == 0.0 and prof[-1, 0] == pytest.approx(g["length"], abs=1e-6)
    assert g["radius_tip"] == prof[0, 1] and g["radius"] == prof[-1, 1]
    for s, r in prof:
        ring = np.abs(Z0 - (LAY.HOLD_FROM + s)) < 1e-6                              # a ring of the mesh at exactly s
        assert ring.sum() == PT.N_AROUND, s
        assert np.hypot(m.V[ring, 0], m.V[ring, 1]).max() == pytest.approx(r, abs=2e-6), s
    assert np.all(np.diff(prof[:, 1]) > 0)                                          # widening toward the shoulder


def test_the_card_s_numbers_are_the_meshes():
    card = LAY.card("c", LAY.options({})["roles"], LAY.options({})["bites"], PT.size())
    V = np.concatenate([PT.body()[0].V, PT.greens_mesh().V])
    assert card["size"] == pytest.approx((V.max(0) - V.min(0)).tolist(), abs=1e-4)
    assert card["origin"] == "tip" and card["front"] == "-Y"
    assert card["params"] == {f"bite{k}": LAY.NEVER for k in range(1, LAY.BITES_MAX + 1)}
    looks = {u["name"]: u["point"] for u in card["use"]["look"]}
    assert looks["grip"] == card["use"]["grip"][0]["center"] and looks["tip"] == [0.0, 0.0, 0.0]
    for k, b in enumerate(card["use"]["bite"], 1):
        assert b["name"] == b["prop"] == f"bite{k}" and b["dir"] == [0.0, 0.0, -1.0]
        assert b["point"] == pytest.approx([0.0, 0.0, LAY.BITE_STEP * k])
        assert PT.stage(k)[0] == pytest.approx(b["point"], abs=1e-9)               # the tip's apex: the face's middle
    assert len(card["use"]["bite"]) == LAY.BITES_MAX == 4
    names = card["objects"]
    assert names[:2] == ["c_body", "c_greens"] and len(names) == 2 + sum(LAY.CHUNKS)
    assert {n for n in names[2:]} == {f"c_chunk{k}_{j}" for k in range(1, 5) for j in range(LAY.CHUNKS[k - 1])}


def test_the_bites_eat_the_end_from_the_tip_up_and_the_keys_are_not_additive():
    m, Z0, _ = PT.body()
    st = [PT.stage(k) for k in range(LAY.BITES_MAX + 1)]
    deltas = PT.key_deltas()
    for k in range(1, LAY.BITES_MAX + 1):
        assert np.array_equal(deltas[k - 1], st[k] - st[k - 1])                     # key k holds P_k - P_(k-1)
        values = [1.0] * k + [0.0] * (LAY.BITES_MAX - k)
        mixed = m.V + sum(v * d for v, d in zip(values, deltas))                    # Blender's relative keys
        assert np.allclose(mixed, st[k], atol=1e-12)                                # bites 1..k on: stage k
        lo, hi = LAY.rim_range(k)
        assert st[k][:, 2].min() >= lo - 1e-9                                       # nothing left under the face
        assert st[k][:, 2].min() <= LAY.bite_height(k) + 1e-9
        assert lo > LAY.rim_range(k - 1)[1] if k > 1 else lo > 0.0                  # each bite clears the last one
        moved = np.linalg.norm(deltas[k - 1], axis=1) > 0
        assert np.nanmax(np.where(moved, Z0, np.nan)) < hi                          # a key moves the end only
        mesh = PT.S.Mesh(st[k], m.Q, m.T)
        assert mesh.is_closed() and 0.0 < mesh.volume() < PT.S.Mesh(st[k - 1], m.Q, m.T).volume()
    assert LAY.bite_height(1) == pytest.approx(0.06) and LAY.bite_height(4) == pytest.approx(0.24)


def test_a_bite_s_rim_is_scalloped_by_tooth_notches_with_cusps_between():
    th = np.linspace(0.0, 2.0 * math.pi, 7201)[:-1]
    for k in range(1, LAY.BITES_MAX + 1):
        n, _, tilt, az = LAY.BITE_SHAPES[k - 1]
        sway = tilt * LAY.radius(LAY.bite_height(k))                                # the face's tilt, taken out
        z = LAY.rim(k, th) - sway * np.cos(th - math.radians(az))
        peaks = (z > np.roll(z, 1)) & (z >= np.roll(z, -1))
        assert peaks.sum() == n                                                     # n round notches
        assert z.max() - z.min() == pytest.approx(LAY.NOTCH * LAY.radius(LAY.bite_height(k)), rel=1e-3)
        d2 = np.roll(z, 1) - 2.0 * z + np.roll(z, -1)
        assert d2[np.argmin(z)] > 50.0 * abs(d2[np.argmax(z)])                      # sharp cusps, round notches
        relief = [np.ptp(LAY.face(k, u, th) - u * sway * np.cos(th - math.radians(az))) for u in (0.0, 0.5, 0.9)]
        assert relief[0] == 0.0 and relief[0] < relief[1] < relief[2]               # the tracks fade to the middle


def test_the_material_reads_the_profile_well_where_a_face_can_be():
    x = np.array([p for p, _ in LAY.profile_ramp()]) * LAY.L
    r = np.array([v for _, v in LAY.profile_ramp()])
    assert len(x) <= 32 and np.all(np.diff(x) > 0)                                  # a colour ramp holds 32 stops
    z = np.linspace(LAY.rim_range(1)[0], LAY.L, 3000)
    assert np.abs(np.interp(z, x, r) / LAY.radius(z) - 1.0).max() < 0.01


def test_the_key_expressions_are_simple_and_give_the_maths():
    bites = [0.2, 0.5, 0.8, 1.1]
    for k in range(1, LAY.BITES_MAX + 1):
        text = LAY.key_expression(k)
        assert len(text) <= 255 and not re.search(r"\*\*|//|%|\.[A-Za-z_]|\bsign\b", text)
        assert set(re.findall(r"[A-Za-z_]\w*", text)) <= set(LAY.ROOT_VARS) | LAY.FUNCS | {"frame"}
        for t in np.linspace(-0.5, 1.5, 81):
            assert _eval(text, _env(t, bites)) == pytest.approx(LAY.key_value(k, t, bites), abs=1e-9)
        t0 = bites[k - 1]
        assert LAY.key_value(k, t0 - 0.01, bites) == 0.0
        assert LAY.key_value(k, t0 + LAY.BITE_TIME, bites) == pytest.approx(1.0) == LAY.key_value(k, t0 + 1.0, bites)
        assert 0.0 < LAY.key_value(k, t0 + 0.5 * LAY.BITE_TIME, bites) < 1.0      # carves in over BITE_TIME
    assert all(LAY.key_value(k, 100.0, [LAY.NEVER] * 4) == 0.0 for k in range(1, 5))   # never by default


def test_the_chunk_expressions_are_simple_and_give_the_maths():
    bites = [0.2, 0.35, 0.8, 1.1]
    ok = set(LAY.ROOT_VARS) | set(LAY.TURN_VARS) | set(LAY.HELPERS) | LAY.FUNCS | {"frame"}
    for c in LAY.chunks():
        ex = LAY.chunk_expressions(c)
        for text in ex.values():
            assert len(text) <= 255 and not re.search(r"\*\*|//|%|\.[A-Za-z_]|\bsign\b|\de", text)
            assert set(re.findall(r"[A-Za-z_]\w*", text)) <= ok
        for t in np.linspace(-0.2, 2.0, 45):
            for rx, ry in ((0.0, 0.0), (-2.6, 0.4), (1.2, -0.9)):
                env = _env(t, bites, rx, ry)
                env["a"] = _eval(ex["a"], env)
                pos, rot, s = LAY.chunk_pose(c, t, bites, rx, ry)
                got = [_eval(ex[ch], env) for ch in ("x", "y", "z", "rx", "ry", "rz", "s")]
                assert got == pytest.approx(list(pos) + list(rot) + [s], abs=1e-9)


def test_a_chunk_pops_in_at_its_bite_flies_up_and_falls_in_the_world_and_is_gone_after_its_life():
    bites = [1.0, 1.1, LAY.NEVER, LAY.NEVER]
    ch = LAY.chunks()
    assert [sum(c["bite"] == k for c in ch) for k in range(1, 5)] == list(LAY.CHUNKS)
    assert all(2 <= n <= 4 for n in LAY.CHUNKS)
    rx, ry = math.radians(-150.0), math.radians(20.0)                               # a carrot held tip up, tilted
    R = PT.S.rot_matrix(-150.0, 20.0, 0.0)
    for c in (x for x in ch if x["bite"] == 1):
        before = LAY.chunk_pose(c, 0.9, bites, rx, ry)
        assert before[2] == LAY.MIN_SCALE and np.allclose(before[0], c["p0"])       # waits at the bite, unseen
        assert np.linalg.norm(np.array(c["p0"][:2])) < LAY.radius(LAY.bite_height(1))
        up = [(R @ LAY.chunk_pose(c, 1.0 + a, bites, rx, ry)[0])[2] for a in np.linspace(0.0, LAY.LIFE, 61)]
        top = int(np.argmax(up))
        assert 0 < top < 60 and up[-1] < up[0]                                      # rises, then falls below its start
        assert LAY.chunk_pose(c, 1.1, bites, rx, ry)[2] > 0.9                       # full size in flight
        assert LAY.chunk_pose(c, 1.0 + LAY.LIFE + 0.01, bites, rx, ry)[2] == LAY.MIN_SCALE
    both = [LAY.chunk_pose(c, 1.2, bites)[2] for c in ch if c["bite"] in (1, 2)]
    assert min(both) > 0.5                                                          # overlapping bites both show


def test_every_part_is_closed_and_faces_out():
    g = PT.greens_mesh()
    parts = _parts(g)
    stems = [p for p in parts if (g.Qm[np.isin(g.Q, p).all(1)] == PT.STEM).any()]
    assert len(stems) == PT.FRONDS and 5 <= PT.FRONDS <= 7
    assert len(parts) == PT.FRONDS * (2 * PT.PAIRS + 2)
    for p in parts:
        sub = _sub(g, p)
        assert sub.is_closed() and sub.volume() > 0
    top = g.V[:, 2].max() - LAY.L
    assert top == pytest.approx(LAY.GREENS_H, abs=0.03)                             # ~0.33 above the shoulder
    for c in LAY.chunks():
        m = PT.chunk_mesh(c["size"], c["seed"])
        assert m.is_closed() and m.volume() > 0
        mats = set(np.concatenate([m.Qm, m.Tm]).tolist())
        assert mats == {PT.CARROT, PT.CUT}                                          # skin on one side, broken flesh
        assert m.size().max() <= 1.2 * c["size"]


def test_bad_options_are_refused_saying_what_to_change():
    assert LAY.options({"bite2": 1.5, "base": "#000000", "wobble": 3})["bites"] == [LAY.NEVER, 1.5, LAY.NEVER, LAY.NEVER]
    assert LAY.options({"leaf": "pine"})["roles"]["leaf"] == "pine"
    for slots, frag in (({"bite1": "soon"}, "bite1"), ({"bite3": True}, "bite3"), ({"bite2": float("nan")}, "bite2"),
                        ({"skin": "orange"}, "skin"), ({"core": "#fff"}, "core")):
        with pytest.raises(ValueError, match=frag):
            LAY.options(slots)
