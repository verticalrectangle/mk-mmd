import shutil

import numpy as np
import pytest

from mkmmd.core import lipsync as LS
from mkmmd.core import perform as PF
from mkmmd.core import timeline as TL


def test_gaze_weights_rise_hold_return_and_never_exceed_one():
    g = PF.GazeEvents([(1.0, 2.0, 2.5, 1, 0.2), (1.9, 3.0, 3.4, 2, 0.2)])
    ts = np.array([0.5, 1.3, 1.95, 2.4, 3.6])
    W = g.weights(ts)
    assert W[:, 0].sum() == 0 and W[:, -1].sum() == 0
    assert W[0, 1] == pytest.approx(1.0)
    assert W.sum(0).max() <= 1.0 + 1e-9                       # overlapping events share the gaze
    assert W[1, 3] > W[0, 3]                                  # the later event takes over


def test_glance_weights_are_the_micro_eye_lift_of_the_reference_video():
    ts = np.linspace(0.0, 4.0, 401)
    W = PF.glance_weights(ts, [(1.0, 0.75, 0.12, 0.25)])
    assert W.shape == (1, 401) and W[0, 0] == 0 and W[0, -1] == 0
    assert W[0, 150] == pytest.approx(1.0)                                # inside the hold (t = 1.5)
    t = 1.9                                                               # in the fall: the reference's own formula
    ref = PF.smooth((t - 1.0) / 0.12) * (1.0 - PF.smooth((t - 1.0 - 0.75) / 0.25))
    assert W[0, 190] == pytest.approx(ref)
    assert 0 < W[0, 190] < 1


def test_glance_weights_that_overlap_share_the_eyes():
    ts = np.linspace(0.0, 3.0, 301)
    W = PF.glance_weights(ts, [(0.5, 1.0, 0.1, 0.4), (1.2, 1.0, 0.1, 0.4)])
    assert W.sum(0).max() <= 1.0 + 1e-9
    assert PF.glance_weights(ts, []).shape == (0, 301)


def test_blink_schedule_spacing_extras_and_avoid():
    extra = [(2.0, 0.3)]
    b = PF.blink_schedule(0.0, 60.0, per_min=20, seed=1, extra=extra, avoid=[(10.0, 20.0)], min_gap=1.2)
    times = [t for t, _ in b]
    assert (2.0, 0.3) in b
    assert all(not (10.0 <= t <= 20.0) for t, d in b if (t, d) != (2.0, 0.3))
    assert min(np.diff(sorted(times))) >= 1.2 - 1e-9
    assert 12 <= len(b) <= 24                                 # ~20 per minute, minus the avoided 10 s
    assert PF.blink_schedule(0, 60, 20, seed=1) == PF.blink_schedule(0, 60, 20, seed=1)


def test_blink_curve_closes_fully_and_reopens():
    ts = np.linspace(0, 1, 301)
    c = PF.blink_curve(ts, [(0.2, 0.157)])
    assert c.max() == pytest.approx(1.0) and c[0] == 0 and c[-1] == 0


def test_beat_bob_peaks_just_after_beats_with_accent():
    ts = np.linspace(0, 2, 601)
    bob = PF.beat_bob(ts, [0.5, 1.5], deg=2.0, accent=[1.0, 2.0])
    k1, k2 = np.argmax(bob[:300]), 300 + np.argmax(bob[300:])
    assert 0.49 <= ts[k1] <= 0.55 and bob[k2] == pytest.approx(2 * bob[k1], rel=1e-6)


def test_timeline_reads_both_layouts():
    assert TL.beats({"beats": [1, 2], "downbeats": [1]}) == ([1, 2], [1])
    assert TL.beats({"tempo": {"beats": [0.5], "downbeats": [0.5]}}) == ([0.5], [0.5])


@pytest.mark.skipif(shutil.which("espeak-ng") is None, reason="espeak-ng not installed")
def test_lipsync_closes_for_m_and_between_distant_words():
    tl = {"fps": 30, "lines": [{"words": [{"text": "mama", "start": 1.0, "voiced_end": 1.6}]},
                               {"words": [{"text": "papa", "start": 3.0, "voiced_end": 3.5}]}]}
    k = LS.keyframes(tl, mouth=0.8)
    a = k["a"]
    assert max(v for _, v in a) > 0.4                         # open vowels
    gap = [v for t, v in a if 1.62 < t < 2.97]                # rest keys 0.06 s after / 0.05 s before the words
    assert len(gap) == 2 and max(gap) == 0.0
    k1 = LS.keyframes(tl, lines=[1, 1])
    assert max(t for t, _ in k1["a"]) < 2.0                    # only the first line


def test_head_turn_is_clamped_and_the_eyes_keep_the_rest():
    f, u = (0.0, -1.0, 0.0), (0.0, 0.0, 1.0)                  # a model facing -Y, Z up
    yaw, elev = PF.yaw_elevation(f, u, (1.0, 0.0, 0.0))        # straight to her left
    assert yaw == pytest.approx(np.pi / 2) and elev == pytest.approx(0.0)
    assert PF.head_angles(yaw, elev, 0.7) == pytest.approx((0.7 * np.pi / 2, 0.0))   # inside the range: share only
    yaw, elev = PF.yaw_elevation(f, u, (0.0, -0.2, 1.0))       # a sign almost overhead
    y, p = PF.head_angles(yaw, elev, 0.7)
    assert y == pytest.approx(0.0) and p == pytest.approx(np.radians(35.0))
    yaw, elev = PF.yaw_elevation(f, u, (-1.0, 1.0, 0.0))       # behind, over her right shoulder
    y, p = PF.head_angles(yaw, elev, 0.7, {"yaw": 60})
    assert yaw == pytest.approx(-0.75 * np.pi) and y == pytest.approx(np.radians(-60.0))


def test_yaw_elevation_does_not_depend_on_how_the_head_is_turned():
    rng = np.random.default_rng(3)
    a, b = rng.normal(size=3), rng.normal(size=3)
    q, _ = np.linalg.qr(np.stack([a, b, np.cross(a, b)], 1))
    q *= np.sign(np.linalg.det(q))                              # a rotation, not a reflection
    f, u, d = np.array([0.0, -1.0, 0.0]), np.array([0.0, 0.0, 1.0]), np.array([0.3, -0.8, 0.5])
    assert PF.yaw_elevation(q @ f, q @ u, q @ d) == pytest.approx(PF.yaw_elevation(f, u, d))
