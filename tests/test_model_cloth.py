"""Cloth copies (mkmmd.model.cloth): the coarse grid a piece knit in rings gives the simulation, the skin the cloth
collides with, and how rig.json keeps a copy (its arrays beside the PMX). numpy only."""
import json

import numpy as np

from mkmmd.core import rig as RG
from mkmmd.model import cloth as CL
from mkmmd.model.parts import outfit_fit as OF


def _rings(K, M, r=0.1):
    t = np.linspace(0.0, 2.0 * np.pi, M, endpoint=False)
    return np.stack([np.stack([r * np.cos(t), r * np.sin(t), np.full(M, 1.0 - 0.01 * k)], 1) for k in range(K)])


def test_a_ring_grid_keeps_both_edges_and_closes_every_row():
    """The grid off a tube of rings keeps its first and last ring (the edge under a band and the hem) whatever the step,
    and each row of quads closes round the tube (no open seam for the cloth to part at)."""
    R = _rings(8, 12)
    g = CL.ring_grid(R, 2)
    k = np.unique(g.verts[:, 2])
    assert np.allclose(np.sort(k)[[0, -1]], [R[-1, 0, 2], R[0, 0, 2]])     # the last ring (k = 7) though 7 is odd
    assert len(k) == 5 and len(g.verts) == 5 * 6
    assert np.allclose(g.verts, R.reshape(-1, 3)[g.index])
    E = np.sort(np.concatenate([g.quads[:, [0, 1]], g.quads[:, [1, 2]], g.quads[:, [2, 3]], g.quads[:, [3, 0]]]), 1)
    E, n = np.unique(E, axis=0, return_counts=True)
    boundary = E[n == 1]
    assert len(boundary) == 2 * 6                                          # only the top and the bottom ring are edges
    assert set(np.unique(g.verts[boundary.ravel(), 2])) == {R[0, 0, 2], R[-1, 0, 2]}


def test_the_collider_sits_inside_the_skin_by_its_inset_and_faces_out_whichever_way_the_skin_is_wound():
    """The cloth collides with the skin set in by `inset` (its rest shape must start clear of it), and the collider's
    faces point out of the body even when the skin is wound inward (else the cloth is pushed into her)."""
    R = _rings(9, 16, r=0.1)
    R[..., 2] = np.linspace(-0.08, 0.08, 9)[:, None]
    cap = np.array([[0, 0, -0.1], [0, 0, 0.1]])
    V = np.concatenate([R.reshape(-1, 3), cap])
    q = np.arange(9 * 16).reshape(9, 16)
    tris = []
    for k in range(8):
        for j in range(16):
            a, b, c, d = q[k, j], q[k + 1, j], q[k + 1, (j + 1) % 16], q[k, (j + 1) % 16]
            tris += [[a, d, c], [a, c, b]]
    for j in range(16):
        tris += [[144, q[0, (j + 1) % 16], q[0, j]], [145, q[8, j], q[8, (j + 1) % 16]]]
    for wound in (np.array(tris), np.array(tris)[:, ::-1]):
        skin = OF.Skin(V, wound, {"上半身": np.ones(len(V))})
        mask = V[wound, 2].mean(1) > -0.05                                  # the sides and the top
        cv, ct, cw = CL.skin_collider(skin, mask, 0.004)
        r0 = np.linalg.norm(V[:144, :2], axis=1).mean()
        side = np.abs(cv[:, 2]) < 0.05
        assert np.allclose(np.linalg.norm(cv[side, :2], axis=1), r0 - 0.004, atol=6e-4)
        fn = np.cross(cv[ct[:, 1]] - cv[ct[:, 0]], cv[ct[:, 2]] - cv[ct[:, 0]])
        out = cv[ct].mean(1)
        assert (np.einsum("ij,ij->i", fn, out) > 0).mean() > 0.95
        assert set(cw) == {"上半身"} and len(cv) < len(V)


def test_rig_json_keeps_the_cloth_copy_s_arrays_aside_by_name_and_its_settings_in_place():
    """rig.json is JSON: a cloth entry keeps its name, materials, bones and settings and names the file and the arrays
    that hold its points, faces, pins and weights; nothing else in the rig changes."""
    quad = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], float)
    entry = CL.copy("skirt", ["skirt"], [{"verts": quad, "quads": [[0, 1, 2, 3]], "pins": np.ones(4),
                                          "weights": {"下半身": np.ones(4)}}],
                    (quad[:3] * 0.5, np.array([[0, 1, 2]]), {"下半身": np.ones(3)}), {"hold": 0.2})
    rig = {"schema": 2, "bones": {"下半身": {}}, "cloth": [entry]}
    out, arrays = RG.cloth_split(rig, "mq.cloth.npz")
    e = out["cloth"][0]
    assert e["file"] == "mq.cloth.npz" and e["settings"] == {"hold": 0.2} and e["bones"] == ["下半身"]
    assert set(e["arrays"]) == set(arrays) == {f"skirt.{k}" for k in ("sim", "collide", *RG.CLOTH_ARRAYS)}
    json.dumps(out)                                                       # what rig.json is
    assert np.array_equal(arrays["skirt.collide"], quad[:3] * 0.5) and out["bones"] == rig["bones"]
    assert RG.cloth_split({"schema": 2}, "x.npz") == ({"schema": 2}, {})
