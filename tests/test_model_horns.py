"""bpy-free invariants of the horns part: an outward-facing horn with a texture seam that tapers to its point, a
mirrored pair on the head. The head and body are the stand-ins of hair_fixtures.py."""
import numpy as np
import pytest

import hair_fixtures as HF
from mkmmd.model import part as PT
from mkmmd.model import spec as SP
from mkmmd.model.parts import horns
from mkmmd.model.parts.hair_geo import face_normals


def test_a_horn_faces_out_tapers_to_its_point_and_curls():
    c = SP.merge(horns.DEFAULTS, {"rings": 0})
    root, heading = np.array([0.1, 0.0, 1.5]), np.array([1.0, 0.0, 0.0])
    V, faces, uv, N, tip = horns.horn(root, heading, np.array([0.0, 0.0, 1.0]), c)
    n, k = int(c["along"]), int(c["around"]) + 1
    rings = V[:-1].reshape(n, k, 3)
    assert np.allclose(rings[:, 0], rings[:, -1]) and np.allclose(uv[:-1].reshape(n, k, 2)[:, [0, -1], 0], [0.0, 1.0])
    assert np.allclose(V[-1], tip) and np.allclose(np.linalg.norm(N, axis=1), 1.0)
    fn = face_normals(V, faces)
    mid = np.array([V[f].mean(0) for f in faces])
    axis = np.array([rings[min(int(i), n - 1), :-1].mean(0) for i in np.arange(len(faces)) // (k - 1)])
    assert (np.einsum("ij,ij->i", fn, mid - axis) > 0.0).all()             # every face looks away from the middle line
    assert (np.einsum("ij,ij->i", N[:-1], (V[:-1] - np.repeat(rings[:, :-1].mean(1), k, 0))) > 0.0).all()
    width = np.linalg.norm(rings[:, :-1].max(1) - rings[:, :-1].min(1), axis=1)
    assert width[0] > 0.9 * c["base"][0] and (np.diff(width) < 1e-9).all() and width[-1] < 0.25 * width[0]
    assert tip[2] > root[2] + 0.05 and np.linalg.norm(tip - root) < c["length"]      # it turned upward on its way


def test_the_pair_is_mirrored_and_rides_the_head(tmp_path):
    ctx = HF.make_ctx(tmp_path, {"horns": {"at": [70, 40], "length": 0.2}, "colors": {"horns": {"base": "#d020d0"}}})
    ctx.part = "horns"
    part = horns.build(ctx)
    PT.check(part)
    mesh, land = part.meshes[0], part.info["landmarks"]
    half = len(mesh.verts) // 2
    left, right = mesh.verts[:half], mesh.verts[half:] * [-1.0, 1.0, 1.0]
    apart = np.linalg.norm(left[:, None] - right[None], axis=2)
    assert apart.min(0).max() < 1e-4 and apart.min(1).max() < 1e-4        # each is the other's mirror
    assert np.allclose(mesh.weights["頭"], 1.0) and list(mesh.weights) == ["頭"]
    assert land["horn.L"][0] > 0.0 > land["horn.R"][0] and land["horn_tip.L"][2] > land["horn.L"][2]

    def volume(faces, origin):                                            # signed: positive when the faces look outward
        tri = [(mesh.verts[f[0]] - origin, mesh.verts[f[i]] - origin, mesh.verts[f[i + 1]] - origin)
               for f in faces for i in range(1, len(f) - 1)]
        return sum(a @ np.cross(b, c) for a, b, c in tri) / 6.0
    vl = volume([f for f in mesh.faces if max(f) < half], land["horn.L"])
    vr = volume([f for f in mesh.faces if min(f) >= half], land["horn.R"])
    assert vl > 1e-5 and vr == pytest.approx(vl, rel=1e-2)                # the mirror faces out too
