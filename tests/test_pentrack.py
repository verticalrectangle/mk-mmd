import json

import numpy as np
import pytest

from mkmmd.core import gripframe as GF
from mkmmd.core import pentrack as PT


def _rot(axis, deg):
    a = np.asarray(axis, float)
    a = a / np.linalg.norm(a)
    th = np.radians(deg)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K


def _track(tmp_path, frames, pos, name="nib.json", **extra):
    p = tmp_path / name
    p.write_text(json.dumps({"frames": list(frames), "target": pos, **extra}), encoding="utf-8")
    return p


# ---------------------------------------------------------------- the track file
def test_load_reads_frames_and_positions(tmp_path):
    p = _track(tmp_path, [181, 182, 184], [[0, 0, 0.74], [0.01, 0, 0.74], [0.03, 0.01, 0.75]], writing=[1, 1, 0])
    frames, pos = PT.load(p)
    assert frames.tolist() == [181, 182, 184] and pos.shape == (3, 3)
    assert pos[2].tolist() == [0.03, 0.01, 0.75]
    f2, p2 = PT.load(p, "target")
    assert (f2 == frames).all() and (p2 == pos).all()


@pytest.mark.parametrize("doc, fragment", [
    ({"frames": [1, 2]}, "needs `frames`"),
    ({"frames": [1, 2], "target": [[0, 0, 0]]}, "1 values"),
    ({"frames": [1, 2], "target": [[0, 0, 0], None]}, "gaps"),
    ({"frames": [1, 2], "target": [[0, 0, 0], [0, 0]]}, "[x, y, z]"),
    ({"frames": [2, 1], "target": [[0, 0, 0], [0, 0, 1]]}, "increasing"),
    ({"frames": [], "target": []}, "0 frames"),
])
def test_load_rejects_what_cannot_be_a_pen_path(tmp_path, doc, fragment):
    p = tmp_path / "t.json"
    p.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(ValueError, match=fragment.replace("[", r"\[").replace("]", r"\]")):
        PT.load(p)


def test_load_another_channel(tmp_path):
    p = tmp_path / "t.json"
    p.write_text(json.dumps({"frames": [1, 2], "target": [[0, 0, 0], [0, 0, 1]], "alt": [[1, 1, 1], [2, 2, 2]]}))
    assert PT.load(p, "alt")[1][1].tolist() == [2, 2, 2]
    with pytest.raises(ValueError, match="'nope'"):
        PT.load(p, "nope")


def test_positions_interpolate_hold_the_ends_and_keep_vectors_together():
    frames = np.array([10, 20, 40])
    pos = np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0], [1.0, 2.0, 5.0]])
    got = PT.positions_at(frames, pos, [0, 10, 15, 20, 30, 40, 99])
    assert got[0].tolist() == [0, 0, 0] and got[-1].tolist() == [1, 2, 5]           # held before and after
    assert got[2].tolist() == [0.5, 1.0, 1.5]
    assert got[4].tolist() == [1.0, 2.0, 4.0]
    assert got.shape == (7, 3)


def test_writing_spot_is_the_mean_at_paper_height():
    pos = np.array([[0.0, 0.0, 0.7411], [0.2, 0.1, 0.7411], [0.1, 0.5, 0.7631]])
    s = PT.writing_spot(pos)
    assert s.tolist() == pytest.approx([0.1, 0.2, 0.7411])
    assert PT.writing_spot(pos, paper_z=0.7408)[2] == 0.7408


# ---------------------------------------------------------------- orientation
def test_wobble_is_slow_deterministic_and_scaled():
    a, b = PT.tilt_wobble(500, 2.0, seed=3), PT.tilt_wobble(500, 2.0, seed=3)
    assert a.shape == (500, 2) and (a == b).all()
    assert not (a == PT.tilt_wobble(500, 2.0, seed=4)).all()
    assert np.abs(a).max() < 2.0                                                    # a fraction of `deg`
    assert np.abs(np.diff(a, axis=0)).max() < 0.2 * 2.0                              # no jitter from frame to frame
    assert PT.tilt_wobble(500, 4.0, seed=3) == pytest.approx(2.0 * a)
    assert not PT.tilt_wobble(10, 0.0).any()


def test_rotations_tilt_about_world_axes_after_the_writing_orientation():
    R0 = _rot([0.3, 1.0, 0.2], 70.0)
    still = PT.pen_rotations(R0, n=4)
    assert still.shape == (4, 3, 3) and all(np.allclose(r, R0) for r in still)
    w = np.array([[1.5, 0.0], [0.0, -2.0], [0.0, 0.0]])
    R = PT.pen_rotations(R0, w)
    assert np.allclose(R[0], _rot([1, 0, 0], 1.5) @ R0)
    assert np.allclose(R[1], _rot([0, 1, 0], -2.0) @ R0)
    assert np.allclose(R[2], R0)
    for r in R:
        assert np.allclose(r @ r.T, np.eye(3)) and np.linalg.det(r) == pytest.approx(1.0)


def test_quaternion_to_matrix_agrees_with_axis_angle():
    th = np.radians(37.0)
    ax = np.array([0.2, -0.5, 0.8])
    ax /= np.linalg.norm(ax)
    q = np.r_[np.cos(th / 2), np.sin(th / 2) * ax]
    assert np.allclose(PT.quat_matrix(q), _rot(ax, 37.0))
    assert np.allclose(PT.quat_matrix(2.0 * q), _rot(ax, 37.0))                      # not necessarily unit


# ---------------------------------------------------------------- wrist goals
def _grip_frame_in_wrist():
    T = np.eye(4)
    T[:3, :3] = _rot([0.2, -0.4, 1.0], 100.0)
    T[:3, 3] = (0.022, 0.104, 0.073)
    return T


def test_goals_put_the_grip_frame_on_every_pen_frame():
    T = _grip_frame_in_wrist()
    pos = np.array([[-0.02, -0.35, 0.7411], [-0.05, -0.37, 0.7431], [0.01, -0.40, 0.7411]])
    R = PT.pen_rotations(_rot([0.1, 0.9, 0.3], 55.0), PT.tilt_wobble(3, 3.0))
    G = PT.pen_frames(pos, R)
    L = 0.06
    goal = PT.goals(G, T, L)
    for g, want in zip(goal, G):
        head = g @ np.linalg.inv(np.array([[1, 0, 0, 0], [0, 1, 0, L], [0, 0, 1, 0], [0, 0, 0, 1.0]]))
        assert np.allclose(head @ T, want)                                          # wrist head frame @ grip = pen frame
        assert (g[:3, 3] - head[:3, 3]) == pytest.approx(L * head[:3, 1])            # the IK acts on the tail
    one = GF.wrist_goal(G[1], T, L)[1]                                              # the static-grip maths agrees
    assert np.allclose(goal[1], one)


def test_goals_follow_the_track_with_a_fixed_orientation():
    T = _grip_frame_in_wrist()
    R0 = _rot([0.1, 0.9, 0.3], 55.0)
    pos = np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.1, 0.2, 0.05]])
    goal = PT.goals(PT.pen_frames(pos, PT.pen_rotations(R0, n=3)), T, 0.05)
    assert np.allclose(goal[:, :3, :3], goal[0, :3, :3])                            # orientation never changes
    shift = goal[1, :3, 3] - goal[0, :3, 3]
    assert shift == pytest.approx([0.1, 0.0, 0.0])                                  # the wrist moves as the nib moves
    assert (goal[2, :3, 3] - goal[0, :3, 3]) == pytest.approx([0.1, 0.2, 0.05])


# ---------------------------------------------------------------- the posture of a hand with a track
def test_posture_fills_in_the_writing_spot_facing_and_shoulder():
    pos = np.array([[-0.1, -0.4, 0.7411], [0.0, -0.3, 0.7431]])
    p = PT.posture({"table": {"z": 0.74}}, pos, facing=(0, -1, 0), shoulder=(-0.0889561, -0.0147457, 0.9323825))
    assert p["nib"] == pytest.approx([-0.05, -0.35, 0.7411])
    assert p["paper_z"] == 0.7411
    assert p["facing"] == [0.0, -1.0, 0.0]
    assert p["shoulder"] == [-0.089, -0.0147, 0.9324]                                # rounded to 0.1 mm for the cache
    assert p["table"] == {"z": 0.74}


def test_posture_keeps_the_users_keys_and_leaves_their_table_alone():
    pos = np.array([[0.0, 0.0, 0.75], [0.1, 0.1, 0.75]])
    user = {"nib": [0.3, 0.3, 0.7408], "shoulder": [1.0, 2.0, 3.0], "facing": [1, 0, 0], "paper_z": 0.7408,
            "pole": [-0.55, 0.2, 0.8]}
    before = json.loads(json.dumps(user))
    p = PT.posture(user, pos, facing=(0, -1, 0), shoulder=(9, 9, 9))
    assert user == before
    assert p["nib"] == [0.3, 0.3, 0.7408] and p["shoulder"] == [1.0, 2.0, 3.0] and p["facing"] == [1.0, 0.0, 0.0]
    assert p["pole"] == [-0.55, 0.2, 0.8] and p["paper_z"] == 0.7408


def test_posture_paper_height_comes_from_the_nib_when_only_the_nib_is_given():
    p = PT.posture({"nib": [[0.0, 0.0, 0.7], [0.0, 0.0, 0.8]]}, np.zeros((1, 3)))
    assert p["paper_z"] == pytest.approx(0.75)
    assert p["nib"] == [[0.0, 0.0, 0.7], [0.0, 0.0, 0.8]]                            # a schedule stays a schedule
    assert "shoulder" not in p and "facing" not in p                                 # the solver's own defaults apply


def test_table_spec_from_a_rest_plane():
    assert PT.table_spec({"center": [0.0, -0.48, 0.74], "radius": 0.32}) == {"z": 0.74, "center": [0.0, -0.48],
                                                                              "radius": 0.32}
    assert PT.table_spec({"center": [1.0, 2.0, 0.5]}) == {"z": 0.5, "center": [1.0, 2.0]}
