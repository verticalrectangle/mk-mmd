"""Static drape of a bone chain (a skirt, a coat, a scarf): every bone points along a direction chosen by the author.

The pose stage (docs: mkmmd.blender.build.pose, `[[pose.<cast>.drape]]`) wants, for each bone of a chain from its root,
the rotation to key. Rotations are composed in the ARMATURE's axes: bone i ends up with the world delta
`D_i = D_(i-1) . q_i` (D_0 the delta of the chain root's parent), so keying `q_i = D_(i-1)^-1 . D_i` puts every bone's own
direction wherever D_i takes its rest direction, whatever the bones above did. D_i is the least rotation taking the bone's
rest direction onto the wanted one. numpy only, tested without Blender."""
import numpy as np


def unit(v):
    v = np.asarray(v, float)
    n = float(np.linalg.norm(v))
    if n < 1e-12:
        raise ValueError("a direction of zero length")
    return v / n


def min_rotation(a, b):
    """The 3x3 rotation of least angle taking direction a onto direction b (a half turn picks an axis perpendicular to a)."""
    a, b = unit(a), unit(b)
    c = float(a @ b)
    v = np.cross(a, b)
    s = float(np.linalg.norm(v))
    if s < 1e-12:
        if c > 0:
            return np.eye(3)
        axis = unit(np.cross(a, [1.0, 0.0, 0.0] if abs(a[0]) < 0.9 else [0.0, 1.0, 0.0]))
        return 2.0 * np.outer(axis, axis) - np.eye(3)
    K = np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])
    return np.eye(3) + K + K @ K * ((1.0 - c) / (s * s))


def chain_keys(rest_dirs, dirs, parent=None):
    """[3x3 rotation q_i in armature axes to key on each bone of a chain] so that bone i points along dirs[i] (least
    rotation from rest_dirs[i]) whatever the bones above it do; `parent` is the world delta of the first bone's parent
    (default: none). Also returns the composed deltas D_i."""
    if len(rest_dirs) != len(dirs):
        raise ValueError(f"{len(dirs)} directions for {len(rest_dirs)} bones")
    D_prev = np.eye(3) if parent is None else np.asarray(parent, float)
    keys, deltas = [], []
    for r, d in zip(rest_dirs, dirs):
        D = min_rotation(r, d)
        keys.append(D_prev.T @ D)
        deltas.append(D)
        D_prev = D
    return keys, deltas
