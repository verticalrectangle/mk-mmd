"""Where an arm's elbow can be for a wrist placed at W, and how straight the wrist can then be: the elbow swings round
the shoulder-wrist line (`circle`), kept out of the body (not inward of the shoulder) and below the shoulder (`elbows`).
The build's pose stage scores grips with it; the wheel grip solver keeps the hand where the forearm can meet it; the move
compiler predicts the elbow an arm IK gives (`pole_elbow`) to keep a wrist natural, and turns the elbow round the circle
to keep the arm's twist in range (mkmmd.core.moves). Pure numpy, any frame."""
import numpy as np

ELBOW_DOWN = 0.05                     # m: an allowed elbow is at least this far below its shoulder ...
ELBOW_STEP = 3.0                      # deg: ... and not inward of it; the elbow circle is sampled in these steps
POLE = (0.45, 0.25, 0.30)             # m: the pose stage's default elbow pole from the shoulder: outward, behind, below


def pole_elbow(S, W, a, b, P):
    """The elbow an arm IK with pole target P gives the arm (upper arm a, forearm b) from shoulder S to wrist W: in the
    plane of S, W and P, on P's side of the shoulder-wrist line; the straight arm's when W is out of reach."""
    S, W, P = (np.asarray(v, float) for v in (S, W, P))
    sw = W - S
    c = float(np.linalg.norm(sw))
    if c < 1e-9:
        return S.copy()
    u = sw / c
    if c >= a + b:
        return S + a * u
    v = (P - S) - ((P - S) @ u) * u
    n = float(np.linalg.norm(v))
    v = v / n if n > 1e-9 else np.cross(u, [0.0, 0.0, 1.0]) / max(np.linalg.norm(np.cross(u, [0.0, 0.0, 1.0])), 1e-9)
    al = np.arccos(np.clip((a * a + c * c - b * b) / (2.0 * a * c), -1.0, 1.0))
    return S + a * (np.cos(al) * u + np.sin(al) * v)


def circle(S, W, a, b, step=ELBOW_STEP, up=(0.0, 0.0, 1.0), out=(1.0, 0.0, 0.0)):
    """((k, 3) elbows, (k, 3) unit directions) of every elbow the arm (upper arm a, forearm b) can have for shoulder S and
    wrist W, `step` degrees apart round the shoulder-wrist line from the side square to it and `up`, and for each the
    direction from that line toward it (the side its pole is on); the straight arm's for all of them when W is out of
    reach (the IK then stretches the arm toward it); (0, 3) twice when W is on S."""
    S, W, up, out = (np.asarray(v, float) for v in (S, W, up, out))
    sw = W - S
    c = float(np.linalg.norm(sw))
    if c < 1e-6:
        return np.zeros((0, 3)), np.zeros((0, 3))
    u = sw / c
    p = np.cross(u, up)
    if np.linalg.norm(p) < 1e-6:
        p = np.cross(u, out)
    p /= np.linalg.norm(p)
    q = np.cross(u, p)
    f = np.radians(np.arange(0.0, 360.0, step))[:, None]
    V = np.cos(f) * p + np.sin(f) * q
    if c >= a + b:
        return np.repeat((S + a * u)[None], len(V), 0), V
    al = np.arccos(np.clip((a * a + c * c - b * b) / (2.0 * a * c), -1.0, 1.0))
    return S + a * (np.cos(al) * u + np.sin(al) * V), V


def elbows(S, W, a, b, up, out, step=ELBOW_STEP, down=ELBOW_DOWN):
    """(k, 3) every elbow the arm (upper arm a, forearm b) can have for shoulder S and wrist W, `step` degrees apart
    round the shoulder-wrist line (`circle`), that is not inward of the shoulder (along `out`) and at least `down` below
    it (along `up`); the straight arm's when W is out of reach (the IK then stretches the arm toward it); (0, 3) when
    none is."""
    S, W, up, out = (np.asarray(v, float) for v in (S, W, up, out))
    if float(np.linalg.norm(W - S)) < 1e-6:
        return np.zeros((0, 3))
    if float(np.linalg.norm(W - S)) >= a + b:
        E = (S + a * (W - S) / np.linalg.norm(W - S))[None]
    else:
        E = circle(S, W, a, b, step, up, out)[0]
    ok = ((E - S) @ out >= 0.0) & ((E - S) @ up <= -down)
    return E[ok]


def bend(E, W, hand_dir):
    """Degrees between the forearm from elbow E ((3,) or (k, 3)) to wrist W and the hand's direction."""
    d = np.asarray(W, float) - np.asarray(E, float)
    d = d / np.linalg.norm(d, axis=-1, keepdims=True)
    return np.degrees(np.arccos(np.clip(d @ np.asarray(hand_dir, float), -1.0, 1.0)))


def min_bend(S, W, a, b, up, out, hand_dir):
    """(degrees, elbow) of the straightest wrist the allowed elbows give; (None, None) when no elbow is allowed."""
    E = elbows(S, W, a, b, up, out)
    if not len(E):
        return None, None
    d = bend(E, W, hand_dir)
    i = int(np.argmin(d))
    return float(d[i]), E[i]
