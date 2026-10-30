"""Arm twist (docs/design.md: Posing, Moves, Checks): how far a forearm turns about its own length (pronation and
supination) and an upper arm about its own (humeral rotation), measured one way by the `joint_limits` check, held by the
pose stage's arm rig and kept by the moves library's elbow choice. Pure numpy (numpy 1.24), any frame.

Both are the swing-twist twist of a bone's posed rotation relative to its parent's, against the rest pose: D = R_parent
P_parent^-1 P_child R_child^-1 (posed P and rest R rotations, in one frame), its twist about the child's rest axis
pointing out along the arm, signed so that + is supination of the forearm and external rotation of the upper arm on
either side (SIGN). The forearm's child is the hand (the wrist bone) and its parent the elbow, so the twist bone's roll
and any the wrist adds both count; the upper arm's child is the arm's distal frame (its twist bone, else the arm bone)
and its parent the chest, so a rig or a motion that splits the turn between the arm and its twist bone reads the same.

MMD rests the arms out, palms down, and the pose stage's arm IK folds the elbow forward from there (PREBEND): the crease
of the elbow faces forward at rest, so the rest roll is the forearm's neutral (mid-prone, the thumb forward) and the
humerus's. The arm IK is a hinge (`hinge`): its elbow frame is the rest one turned so the rest fold's plane goes to the
plane of the shoulder, elbow and wrist, which predicts both twists of an arm the IK poses from where its joints and its
hand are (`twists`), within about a degree."""
import math

import numpy as np

SIGN = {"L": -1.0, "R": 1.0}       # + supination / external rotation, for an axis pointing out along the arm
FOREARM = (-90.0, 100.0)           # degrees from the rest roll: pronation, supination (a human forearm turns ~80-90 each
                                   # way from mid-prone; a hand fretting a guitar neck takes ~100)
UPPER = (-95.0, 95.0)              # degrees: internal, external rotation of the upper arm (~70-90 each way, depending on
                                   # how far the arm is raised; a hand behind the back takes ~80 in), the external bound
                                   # widening as the arm rises above the shoulder (`upper_limits`)
PREBEND = 20.0                     # degrees the arm IK folds the forearm toward the front before it solves (the pose stage)
FRONT = (0.0, -1.0, 0.0)           # the character's front in its own frame (x its left, y behind it, z up)
DOWN = (0.0, 0.0, -1.0)            # ... and its down
REST = {"upper": (0.866, 0.0, -0.5), "fore": (0.866, 0.0, -0.5), "dir": (0.866, 0.0, -0.5),
        "palm": (-0.5, 0.0, -0.866)}   # a typical MMD left arm at rest (30 degrees down, straight, palm down)


def _unit(v):
    v = np.asarray(v, float)
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.where(n > 1e-12, n, 1.0)


def elevation(upper_dir, down=DOWN):
    """Degrees an upper arm pointing along `upper_dir` (arm head -> elbow head) is raised from hanging along `down` (the
    chest's down: its rest -Z as it stands): 0 hanging, 90 level, 180 overhead."""
    return float(np.degrees(np.arccos(np.clip(_unit(upper_dir) @ _unit(down), -1.0, 1.0))))


def upper_limits(elev):
    """(lo, hi) degrees of the upper arm's twist at `elev` degrees of elevation (`elevation`): UPPER, the external bound
    widened by as much as the arm is raised past level (an arm raised overhead out to the side turns ~90 out on its way
    up, and swing-twist from the rest reads an arm raised forward overhead near 180)."""
    return UPPER[0], UPPER[1] + max(0.0, float(elev) - 90.0)


def quaternion(D):
    """(..., 4) unit quaternions (w, x, y, z), w >= 0, of rotation matrices D (..., 3, 3) (Shepperd's method: each from its
    largest diagonal term)."""
    D = np.asarray(D, float)
    m = lambda i, j: D[..., i, j]                                                        # noqa: E731
    tr = m(0, 0) + m(1, 1) + m(2, 2)
    k = np.argmax(np.stack([tr, m(0, 0), m(1, 1), m(2, 2)], -1), -1)
    s = [2.0 * np.sqrt(np.maximum(v, 1e-12)) for v in (1.0 + tr, 1.0 + m(0, 0) - m(1, 1) - m(2, 2),
                                                       1.0 - m(0, 0) + m(1, 1) - m(2, 2), 1.0 - m(0, 0) - m(1, 1) + m(2, 2))]
    wx, wy, wz = m(2, 1) - m(1, 2), m(0, 2) - m(2, 0), m(1, 0) - m(0, 1)
    xy, xz, yz = m(0, 1) + m(1, 0), m(0, 2) + m(2, 0), m(1, 2) + m(2, 1)
    q = np.choose(k[..., None], [np.stack([0.25 * s[0], wx / s[0], wy / s[0], wz / s[0]], -1),
                                 np.stack([wx / s[1], 0.25 * s[1], xy / s[1], xz / s[1]], -1),
                                 np.stack([wy / s[2], xy / s[2], 0.25 * s[2], yz / s[2]], -1),
                                 np.stack([wz / s[3], xz / s[3], yz / s[3], 0.25 * s[3]], -1)])
    q = q * np.where(q[..., :1] < 0.0, -1.0, 1.0)
    return q / np.linalg.norm(q, axis=-1, keepdims=True)


def twist(D, axis):
    """Degrees in (-180, 180] that rotations D (..., 3, 3) turn about the unit `axis` (..., 3), right-handed: the twist of
    their swing-twist decomposition (D = swing . twist, the swing about an axis square to `axis`)."""
    q = quaternion(D)
    t = 2.0 * np.degrees(np.arctan2(np.sum(q[..., 1:] * _unit(axis), -1), q[..., 0]))
    t = np.where(t <= -180.0, t + 360.0, np.where(t > 180.0, t - 360.0, t))
    return t if np.ndim(t) else float(t)


def relative_twist(P_parent, R_parent, P_child, R_child, axis, side):
    """Degrees a child bone turns about `axis` (its rest axis, out along the arm) relative to its parent, against the rest
    pose, + supination / external rotation (SIGN): P_* (..., 3, 3) posed rotations (unscaled), R_* (3, 3) rest ones,
    all in one frame."""
    P_parent, P_child = np.asarray(P_parent, float), np.asarray(P_child, float)
    D = np.asarray(R_parent, float) @ np.swapaxes(P_parent, -1, -2) @ P_child @ np.asarray(R_child, float).T
    return SIGN[side] * twist(D, axis)


def basis(x, y):
    """(3, 3) rotation whose columns are `x`, `y` made square to it, and their cross."""
    x = _unit(x)
    y = np.asarray(y, float) - (np.asarray(y, float) @ x) * x
    y = _unit(y)
    return np.stack([x, y, np.cross(x, y)], 1)


def rotation(axis, deg):
    """(3, 3) rotation by `deg` degrees about `axis` (right-handed)."""
    a = _unit(axis)
    K = np.array([[0.0, -a[2], a[1]], [a[2], 0.0, -a[0]], [-a[1], a[0], 0.0]])
    t = math.radians(deg)
    return np.eye(3) + math.sin(t) * K + (1.0 - math.cos(t)) * (K @ K)


def rest_hinge(rest):
    """The elbow's fold axis at rest as the arm IK starts from it (the left arm, the character's frame): the normal of the
    plane of the rest upper arm and the forearm folded PREBEND toward the front, u x f."""
    u, a = _unit(rest["upper"]), _unit(rest["fore"])
    fold = np.cross(a, FRONT)
    if np.linalg.norm(fold) < 1e-6:                    # a forearm pointing straight ahead at rest: it folds up
        fold = np.cross(a, (0.0, 0.0, 1.0))
    return _unit(np.cross(u, rotation(fold, PREBEND) @ a))


def hinge(S, E, W, P=None):
    """The fold axis of an arm from shoulder S over elbow E to wrist W, u x f; for an arm too straight to tell, from the
    side its pole P is on (None: the elbow's own side)."""
    S, E, W = (np.asarray(v, float) for v in (S, E, W))
    n = np.cross(E - S, W - E)
    if np.linalg.norm(n) > 1e-4 * np.linalg.norm(W - S) ** 2 or P is None:
        return _unit(n)
    sw = _unit(W - S)
    v = (np.asarray(P, float) - S) - ((np.asarray(P, float) - S) @ sw) * sw
    return _unit(np.cross(v, sw))


def twists(rest, side, S, E, W, d, p, P=None):
    """(forearm, upper arm) twist in degrees (+ supination, external rotation) of the arm `side` an arm IK poses with its
    shoulder at S, elbow at E, wrist at W and the hand pointing along `d`, its palm toward `p` (points and directions in
    the character's frame; `rest` the left arm's rest directions {upper, fore, dir, palm}, the right arm mirrored onto
    it; P the elbow's pole, for an arm too straight to show its fold). The elbow is a hinge: its frame is the rest one
    turned so the rest fold axis (`rest_hinge`) lies along `hinge`, the upper arm's likewise."""
    m = np.array([1.0 if side == "L" else -1.0, 1.0, 1.0])     # the right arm's points and directions onto the left's
    S, E, W, d, p = (np.asarray(v, float) * m for v in (S, E, W, d, p))
    P = None if P is None else np.asarray(P, float) * m
    u0, a0, h0 = _unit(rest["upper"]), _unit(rest["fore"]), rest_hinge(rest)
    h = hinge(S, E, W, P)
    Re = basis(W - E, h) @ basis(a0, h0).T                # the elbow's rotation from rest
    Ru = basis(E - S, h) @ basis(u0, h0).T                # the upper arm's
    Rh = basis(d, p) @ basis(rest["dir"], rest["palm"]).T  # the hand's
    return SIGN["L"] * twist(Re.T @ Rh, a0), SIGN["L"] * twist(Ru, u0)


def excess(value, limits, margin=0.0):
    """How far (degrees, >= 0) `value` lies outside `limits` (lo, hi) narrowed by `margin` on each side; 0 inside."""
    return max(limits[0] + margin - value, value - limits[1] + margin, 0.0)
