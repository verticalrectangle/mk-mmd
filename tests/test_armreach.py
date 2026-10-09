"""Where an arm's elbow can be (mkmmd.core.armreach): the elbows keep the arm's lengths and stay out of the body and
below the shoulder; a wrist out of reach gets the straight arm; the straightest wrist the allowed elbows give."""
import numpy as np

from mkmmd.core import armreach as AR

S, UP, OUT = np.zeros(3), np.array([0.0, 0.0, 1.0]), np.array([1.0, 0.0, 0.0])
A, B = 0.26, 0.23
W = np.array([0.15, -0.30, -0.10])                          # in front, a little out and down: within reach


def test_every_elbow_keeps_the_arm_lengths_and_stays_out_of_the_body_and_below_the_shoulder():
    E = AR.elbows(S, W, A, B, UP, OUT)
    assert len(E) > 10
    assert np.allclose(np.linalg.norm(E - S, axis=1), A) and np.allclose(np.linalg.norm(W - E, axis=1), B)
    assert ((E - S) @ OUT >= 0.0).all() and ((E - S) @ UP <= -AR.ELBOW_DOWN).all()


def test_a_wrist_out_of_reach_gets_the_straight_arm_and_one_up_and_inward_gets_no_elbow():
    far = np.array([0.30, -0.40, -0.25])                     # 0.56 m away, the arm 0.49 m long
    E = AR.elbows(S, far, A, B, UP, OUT)
    assert len(E) == 1 and np.allclose(E[0], A * far / np.linalg.norm(far))
    assert len(AR.elbows(S, np.array([-0.30, 0.0, 0.25]), A, B, UP, OUT)) == 0
    assert AR.min_bend(S, np.array([-0.30, 0.0, 0.25]), A, B, UP, OUT, UP) == (None, None)


def test_the_straightest_wrist_is_the_one_that_continues_an_allowed_forearm():
    E = AR.elbows(S, W, A, B, UP, OUT)
    k = len(E) // 2
    hand = (W - E[k]) / np.linalg.norm(W - E[k])            # a hand continuing one allowed forearm
    bend, best = AR.min_bend(S, W, A, B, UP, OUT, hand)
    assert bend < 0.01 and np.allclose(best, E[k])
    assert AR.min_bend(S, W, A, B, UP, OUT, -hand)[0] > 120.0   # pointing back at the elbow
