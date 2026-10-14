"""Helper bones at a joint (numpy only), for any part whose mesh bends there: a garment's sleeve at the elbow, its
armhole at the shoulder, a skirt at the hip.

Linear blend skinning bends a joint like a hinge drawn in rubber: a vertex shared by the two bones of the joint goes
to the mean of where each of them would take it, so when the joint folds far the mesh there pulls in towards the
joint, flattens on the outside of the bend and sinks into itself in the crook. A helper bone at the joint turns a
share (`ratio`, half by default) of the joint bone's rotation, by a PMX rotation grant (MMD and mmd_tools do it
themselves; the rig description lists it as the bone's `grant`), and the vertices the two bones share go to it as far
as they are shared: those half on each side all of the way, so they turn rigidly with it and keep their distance from
the joint, as a sleeve does round a bent elbow.

    b = helper_bone("左ひじ補助", joint="左ひじ", parent="左腕捩", head=elbow, tail=elbow + along * 0.03)
    W = helper_weights(W, above=("左腕", "左腕捩"), below=("左ひじ", "左手捩"), helper="左ひじ補助")

`helper_weights` takes `peak` x twice the smaller of the two sides' weights (all of a vertex split evenly, none of one
wholly on either side) from both sides pro rata; the rest of the weights, other bones' too, stay."""
import numpy as np

from .part import Bone


def helper_bone(name, joint, parent, head, tail, ratio=0.5, name_en=""):
    """A helper bone at `head` that turns `ratio` of `joint`'s rotation (its grant parent) on top of `parent`'s (the
    bone above the joint, so the helper rides the limb above it). Invisible, deforming, after the joint in the PMX's
    transform order (a grant needs its parent first: the assembler orders bones by parents and grant parents)."""
    return Bone(name=name, head=tuple(float(x) for x in head), tail=tuple(float(x) for x in tail), parent=parent,
                name_en=name_en, visible=False, grant={"parent": joint, "rotate": True, "move": False,
                                                       "ratio": float(ratio)})


def helper_weights(W, above, below, helper, peak=1.0):
    """Weights {bone: (n,)} with the share the two sides of a joint split given to `helper`: per vertex `peak` x 2 x the
    smaller of the summed weights on the bones `above` and `below` the joint, taken from both sides in proportion to
    their weights. Returns a new dict (the input is not changed); weights of other bones stay."""
    W = {b: np.asarray(w, float) for b, w in W.items()}
    n = len(next(iter(W.values()))) if W else 0
    zero = np.zeros(n)
    a = sum((W[b] for b in above if b in W), zero)
    b_ = sum((W[b] for b in below if b in W), zero)
    share = np.minimum(float(peak) * 2.0 * np.minimum(a, b_), a + b_)
    if not share.any():
        return W
    scale = np.where(a + b_ > 0.0, 1.0 - share / np.maximum(a + b_, 1e-12), 1.0)
    out = {k: (w * scale if k in above or k in below else w) for k, w in W.items()}
    out[helper] = out.get(helper, zero) + share
    return out
