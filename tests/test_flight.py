"""Flights, bpy-free (mkmmd.core.flight): moves land on their time, the table refuses what cannot fly, the flown prop
keeps clear of its target, the follow lags without overshooting, and the wire never passes through the target."""
import math

import numpy as np
import pytest

from mkmmd.core import flight as FL

FPS = 60.0


def _ts(t0, t1):
    return np.arange(round(t0 * FPS), round(t1 * FPS) + 1) / FPS


def _flight(moves, **kw):
    return FL.normalize(dict({"prop": "p", "moves": moves}, **kw))


def test_moves_are_there_on_their_time():
    F = _flight([{"t": 1.0, "do": "in", "from": [1.0, -0.3, 0.2], "to": [0.4, -0.2, 0.0]},
                 {"t": 2.0, "do": "dart", "to": [-0.4, -0.2, 0.1]},
                 {"t": 3.5, "do": "glide", "to": [0.0, -0.4, 0.5], "dur": 0.8},
                 {"t": 5.0, "do": "perch", "on": [0.0, 0.0, 0.15]}])
    ts = _ts(0.0, 6.0)
    pr = FL.path(F, ts)
    O = pr["O"] - 0.0
    i = {t: int(round(t * FPS)) for t in (0.0, 1.0, 2.0, 3.5, 5.0, 5.9)}
    nohover = dict(F, moves=[dict(m) for m in F["moves"]] + [])
    calm = FL.path(dict(nohover, moves=nohover["moves"] + [dict(do="hover", t=0.0, darts=0.0, every=1.0, bob=0.0,
                                                                  index=9, t0=0.0, t1=0.0)]), ts)["O"]
    assert calm[i[0.0]] == pytest.approx([1.0, -0.3, 0.2])                       # waits where it flies in from
    assert calm[i[1.0]] == pytest.approx([0.4, -0.2, 0.0], abs=1e-6)             # in: there ON its time
    assert calm[i[2.0]] == pytest.approx([-0.4, -0.2, 0.1], abs=1e-6)            # dart: there ON its time...
    past = calm[i[2.0]:i[2.0] + 15, 0].min()
    assert past < -0.42                                                          # ...runs past it...
    assert calm[i[2.0] + 40] == pytest.approx([-0.4, -0.2, 0.1], abs=0.006)     # ...and settles back
    assert calm[i[3.5]] == pytest.approx([0.0, -0.4, 0.5], abs=1e-6)             # glide
    assert calm[i[5.0], :2] == pytest.approx([0.0, 0.0], abs=1e-6) and calm[i[5.0], 2] == pytest.approx(0.15, abs=1e-6)
    assert pr["perch"][i[5.9]] == 1.0 and pr["rest"][i[5.9]] == 1.0 and pr["perch"][i[3.5]] == 0.0
    assert np.isfinite(O).all()
    jitter = np.linalg.norm(pr["O"] - calm, axis=1)
    assert jitter[i[3.5] + 30:i[3.5] + 60].max() > 0.01                          # it hovers between moves...
    assert jitter[i[5.9]] == pytest.approx(0.0, abs=1e-9)                       # ...not perched


def test_the_table_refuses_what_cannot_fly():
    with pytest.raises(FL.FlightError, match=r"moves\[1\] \(dart on 1.1 s\) starts at 0.88 s, before moves\[0\]"):
        _flight([{"t": 1.0, "do": "glide", "to": [0.3, 0.0, 0.0]}, {"t": 1.1, "do": "dart", "to": [0.0, -0.3, 0.0]}])
    with pytest.raises(FL.FlightError, match="passes through the target"):
        _flight([{"t": 1.0, "do": "circle", "radius": 0.1}], clear=0.15)
    with pytest.raises(FL.FlightError, match="unknown keys"):
        _flight([{"t": 1.0, "do": "dart", "to": [0.3, 0.0, 0.0], "speed": 2}])


def test_follow_lags_and_never_runs_past():
    X = np.zeros((240, 1))
    X[60:] = 1.0
    Y = FL.follow(X, 1.0 / FPS, 0.3)
    assert Y[59, 0] == 0.0 and 0.1 < Y[90, 0] < 0.6                              # late...
    assert Y.max() <= 1.0 + 1e-9 and Y[-1, 0] > 0.97                             # ...and never past where it goes


def test_the_prop_and_its_wire_keep_out_of_the_target():
    """A circle round her head with the hand on the far side: the prop stays `clear` away, and so does every point of
    the wire, which has to bend round her (a straight wire would go through)."""
    F = _flight([{"t": 0.5, "do": "circle", "dur": 2.0, "radius": 0.35, "turns": 1.0}],
                start=[0.35, -0.1, 0.0], hand=[-1.4, 0.3, 1.5], clear=0.15)
    ts = _ts(0.0, 3.0)
    T = np.tile([0.0, 0.0, 1.45], (len(ts), 1)) + np.stack([0.05 * np.sin(ts * 3.0), 0 * ts, 0.03 * np.sin(ts * 7.0)], 1)
    pr = FL.path(F, ts)
    P = FL.place(F, ts, T, np.eye(3), pr)
    assert (np.linalg.norm(P - T, axis=1) >= 0.15 - 1e-9).all()
    M = FL.attitude(P, T, ts, np.eye(3), pr["perch"], F["show"])
    assert np.allclose(np.einsum("nij,nik->njk", M, M), np.eye(3), atol=1e-6)   # a rotation on every frame
    D = P + np.einsum("nij,j->ni", M, [0.0, 0.005, -0.027])
    H = FL.hand_track(F, ts, [0.0, 0.0, 0.0])
    W = FL.wire(H, D, -M[:, :, 2], ts, [(T, F["clear"])])
    assert np.allclose(W[:, 0], H) and np.allclose(W[:, -1], D)
    least, _at = FL.clearance(W, [(T, 0.15)])
    assert least >= -1e-9
    straight = H[:, None] + (D - H)[:, None] * np.linspace(0, 1, 32)[None, :, None]
    assert (np.linalg.norm(straight - T[:, None], axis=2) < 0.15).any()          # the test does make it bend round
