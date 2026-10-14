"""Helper bones at a joint (mkmmd.model.helpers): the weights a joint's two sides share go to the helper. bpy-free."""
import numpy as np

from mkmmd.model import helpers as HP


def test_the_helper_takes_what_the_two_sides_of_the_joint_share():
    """A vertex split evenly between the bones above and below the joint goes to the helper as far as `peak` says, one
    split 3:1 half as far, one wholly on either side not at all; what it takes comes from both sides in proportion, the
    weights of other bones stay, and every vertex keeps its total (a mesh whose weights no longer sum to one shrinks
    towards the origin under skinning)."""
    W = {"up": np.array([0.5, 0.75, 1.0, 0.0, 0.3]), "up2": np.array([0.0, 0.0, 0.0, 0.0, 0.1]),
         "low": np.array([0.5, 0.25, 0.0, 1.0, 0.2]), "other": np.array([0.0, 0.0, 0.0, 0.0, 0.4])}
    out = HP.helper_weights(W, above=("up", "up2"), below=("low",), helper="h", peak=0.8)
    assert np.allclose(out["h"], [0.8, 0.4, 0.0, 0.0, 0.32])
    assert np.allclose(sum(out.values()), sum(W.values()))
    assert np.allclose(out["other"], W["other"])
    assert np.allclose(out["up"][:2] / out["low"][:2], W["up"][:2] / W["low"][:2])     # both sides give in proportion
    assert np.allclose(out["up"][2:4], W["up"][2:4]) and np.allclose(out["low"][2:4], W["low"][2:4])
    assert np.allclose(W["up"], [0.5, 0.75, 1.0, 0.0, 0.3])                         # the input is left as it was
