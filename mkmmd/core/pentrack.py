"""Pen tracks: the maths of a pen held in a solved grip while its nib follows a recorded path (docs/design.md: Grips).

A track is the project's reference data (docs/design.md: Project file): `{"frames": [Blender frames], "<channel>":
[[x, y, z], ...]}` in world metres, one position per frame, e.g. `tracks/nib.json` with the channel `target` = where
the nib has to be. The pose stage (mkmmd.blender.build.pose) solves ONE grip for the pen on the hand (the finger
rotations, `target_in_wrist` and the writing orientation of the pen), then for every frame puts the pen frame on the
track and the wrist where the grip says: `wrist = pen_frame @ inv(target_in_wrist)`, the arm IK goal acting on the bone's
tail. The pen object rides the wrist bone, so its origin (the nib) lands on the track within the IK's accuracy.

numpy only: shared by the pose stage (Blender's Python) and the tests (no Blender). Frames are 4x4 matrices whose columns
are x, y, z and the origin, as in mkmmd.core.gripframe."""
import json

import numpy as np

from . import perform as PF


def load(path, channel="target"):
    """(frames (n,) int, positions (n, 3)) of a track file. The frames must increase and every sample must be a point."""
    with open(path, encoding="utf-8") as fh:
        t = json.load(fh)
    if "frames" not in t or channel not in t:
        raise ValueError(f"{path}: needs `frames` and the channel {channel!r} (has {sorted(t)})")
    frames = np.asarray(t["frames"], float)
    vals = t[channel]
    if len(vals) != len(frames) or len(frames) == 0:
        raise ValueError(f"{path}: {len(frames)} frames but {len(vals)} values in {channel!r}")
    if any(v is None for v in vals):
        raise ValueError(f"{path}: {channel!r} has gaps (null): a pen track needs a position on every frame")
    try:
        pos = np.asarray(vals, float)
    except ValueError:
        pos = np.zeros(0)
    if pos.shape != (len(frames), 3) or not np.isfinite(pos).all():
        raise ValueError(f"{path}: {channel!r} must be one finite [x, y, z] per frame")
    if np.any(np.diff(frames) <= 0):
        raise ValueError(f"{path}: frames must be strictly increasing")
    return frames, pos


def positions_at(frames, pos, want):
    """(m, 3) positions at the frames `want`: linear between the samples, held before the first and after the last."""
    frames, pos, want = np.asarray(frames, float), np.asarray(pos, float), np.asarray(want, float)
    return np.stack([np.interp(want, frames, pos[:, k]) for k in range(3)], 1)


def writing_spot(pos, paper_z=None):
    """The point the writing posture is solved at: the track's mean x, y at paper height (`paper_z`, default the lowest
    z of the track, where the nib touches)."""
    pos = np.asarray(pos, float)
    z = float(pos[:, 2].min()) if paper_z is None else float(paper_z)
    return np.array([pos[:, 0].mean(), pos[:, 1].mean(), z])


def tilt_wobble(n, deg, tau_frames=18.0, seed=3):
    """(n, 2) slow random tilt of the pen, degrees about the world X and Y axes: low-passed noise (critically damped,
    `tau_frames`) times `deg`, deterministic per seed, already settled on the first frame (the filter runs 6 time
    constants ahead of it). A hand never holds a pen perfectly still."""
    if not deg:
        return np.zeros((n, 2))
    warm = int(6 * tau_frames)
    raw = np.random.default_rng(int(seed)).normal(0.0, 1.0, (n + warm, 2))
    return PF.lowpass(raw, float(tau_frames))[warm:] * float(deg)


def _axis(axis, deg):
    a = np.radians(np.asarray(deg, float))
    c, s = np.cos(a), np.sin(a)
    o, z = np.ones_like(a), np.zeros_like(a)
    rows = {"x": ((o, z, z), (z, c, -s), (z, s, c)), "y": ((c, z, s), (z, o, z), (-s, z, c))}[axis]
    return np.moveaxis(np.array(rows), (0, 1), (-2, -1))


def pen_rotations(R0, wobble=None, n=None):
    """(n, 3, 3) world rotations of the pen frame: the writing orientation `R0` (3x3), tilted by the wobble (degrees
    about world X, then world Y, applied after R0). Without a wobble every frame is R0 (give `n`)."""
    R0 = np.asarray(R0, float)
    if wobble is None:
        return np.tile(R0, (n, 1, 1))
    w = np.asarray(wobble, float)
    return _axis("x", w[:, 0]) @ _axis("y", w[:, 1]) @ R0


def quat_matrix(q):
    """3x3 rotation of a unit quaternion (w, x, y, z)."""
    w, x, y, z = (float(v) for v in np.asarray(q, float) / np.linalg.norm(q))
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def pen_frames(positions, rotations):
    """(n, 4, 4) pen frames: origin the nib position, axes the pen's rotation."""
    n = len(positions)
    G = np.tile(np.eye(4), (n, 1, 1))
    G[:, :3, :3] = rotations
    G[:, :3, 3] = positions
    return G


def goals(frames, target_in_wrist, bone_length):
    """(n, 4, 4) arm IK goals that put the grip frame of the pen on `frames` ((n, 4, 4) pen frames): the wrist head frame
    `G @ inv(target_in_wrist)` moved along the bone by its length (the IK acts on the tail)."""
    T = np.eye(4)
    T[1, 3] = float(bone_length)
    return np.asarray(frames, float) @ (np.linalg.inv(np.asarray(target_in_wrist, float)) @ T)[None]


def posture(user, positions, facing=None, shoulder=None, decimals=4):
    """The solver's `posture` table (mkmmd.solvers.grip _Posture) for a hand with a track: the user's keys win; `nib`
    (the track's writing spot at `paper_z`), `facing` (the character's horizontal heading) and `shoulder` (where it
    really is in the seated pose) are filled in when missing. Numbers are rounded to 0.1 mm so that a rebuild with the
    same pose hits the solver's cache. The user's table is not changed."""
    p = dict(user or {})
    if p.get("nib") is None:
        p["nib"] = writing_spot(positions, p.get("paper_z"))
    if p.get("paper_z") is None:
        p["paper_z"] = float(np.asarray(p["nib"], float).reshape(-1, 3)[:, 2].mean())
    if p.get("facing") is None and facing is not None:
        p["facing"] = [float(v) for v in facing]
    if p.get("shoulder") is None and shoulder is not None:
        p["shoulder"] = [float(v) for v in shoulder]
    for k in ("nib", "shoulder", "facing", "pole"):
        if p.get(k) is not None:
            p[k] = np.round(np.asarray(p[k], float), decimals).tolist()
    p["paper_z"] = round(float(p["paper_z"]), decimals)
    return p


def table_spec(plane, up=(0.0, 0.0, 1.0)):
    """The posture's `table` ({z, center [x, y], radius}) from a world rest plane: its centre point and radius (None:
    the plane is unbounded, every point over it counts)."""
    c = np.asarray(plane["center"], float)
    out = {"z": round(float(c @ np.asarray(up, float)), 4), "center": [round(float(c[0]), 4), round(float(c[1]), 4)]}
    if plane.get("radius") is not None:
        out["radius"] = round(float(plane["radius"]), 4)
    return out
