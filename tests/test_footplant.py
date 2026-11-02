import numpy as np

from mkmmd.core import footplant as FP

FLOOR = 0.1                                   # the ankle's height when the foot stands flat


def _step():
    """A foot that rests 20 frames drifting 3 mm a frame (a retargeted motion's slide), swings 0.4 m forward over 10
    frames lifted 8 cm, and rests again 20 frames, dipping 5 mm into the floor."""
    p = np.zeros((50, 3))
    p[:20, 0] = 0.003 * np.arange(20)
    u = np.linspace(0.0, 1.0, 12)[1:-1]
    p[20:30, 0] = p[19, 0] + (0.4 - p[19, 0]) * u
    p[20:30, 2] = 0.08 * np.sin(np.pi * u)
    p[30:, 0] = 0.4 + 0.003 * np.arange(20)
    p[30:, 2] = -0.005
    p[:, 2] += FLOOR
    return p


def test_a_resting_foot_is_held_where_it_landed_and_eased_back_onto_its_path_after_lift_off():
    p = _step()
    out, spans = FP.plant(p, FLOOR)
    assert len(spans) == 2 and spans[0][0] == 0 and spans[1][1] == 49
    for a, b in spans:
        assert np.allclose(out[a:b + 1, :2], p[a, :2])                 # still across the floor while it rests
    assert np.all(out[spans[1][0]:, 2] >= FLOOR - 1e-12)                # and never under the floor
    a1 = spans[1][0]
    assert np.all(np.linalg.norm(np.diff(out[:, :2], axis=0), axis=1) < 0.09)   # no pop: eased, not jumped
    lifted = slice(spans[0][1] + 1, a1)
    assert np.allclose(out[lifted, 2], p[lifted, 2])                    # it lifts as it did
    own = slice(spans[0][1] + FP.EASE + 1, a1)
    assert own.stop - own.start >= 3 and np.allclose(out[own], p[own])  # and is back on its own path before landing


def test_a_glide_and_a_touch_too_short_are_left_alone():
    p = np.zeros((40, 3))
    p[:, 0] = 0.02 * np.arange(40)                                       # gliding across the floor, 2 cm a frame
    p[:, 2] = FLOOR
    out, spans = FP.plant(p, FLOOR)
    assert spans == [] and np.allclose(out, p)
    q = np.zeros((20, 3))
    q[:, 2] = FLOOR + 0.1                                                # in the air but for three frames
    q[8:11, 2] = FLOOR
    assert FP.contacts(q, FLOOR) == []
    q[8:12, 2] = FLOOR                                                   # four frames are a rest
    assert FP.contacts(q, FLOOR) == [(8, 11)]
