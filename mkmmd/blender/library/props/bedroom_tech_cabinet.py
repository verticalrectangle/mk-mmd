"""Moulded cabinet of the radio-cassette player (pure numpy, no bpy: importable from tests, see
tests/test_bedroom_tech_cabinet.py). One closed, outward-facing quad mesh: a side section (y, z) with real fillets, a dished
underside and a leaning lower back is swept along X and closed at both ends by quarter-round rims and shallow domes.

Frame: the player's prop frame (x right, y deep, front = -Y, z up). `cabinet` returns welded arrays (V, Q, T): vertices,
quads and the few triangles of the dome centres. Every arc is sampled with an ease (short segments next to the flat faces) so
smooth shading does not tilt the big flat panels."""
import math

import numpy as np

from ....core import shell as S


def _graded(n):
    """n + 1 fractions 0..1 with short steps at both ends (a cosine ease)."""
    return 0.5 * (1.0 - np.cos(np.pi * np.arange(n + 1) / n))


def _arc(c, R, a0, a1, n):
    t = a0 + (a1 - a0) * _graded(n)
    return np.stack([c[0] + R * np.cos(t), c[1] + R * np.sin(t)], 1)


def section(yf, yb, z0, z1, r_ft, r_bt, r_fb, r_bb, dish=0.0, lean=0.0, knee=0.0, n=6, n_dish=14, step=0.02):
    """Closed side section (m, 2) of (y, z), counter-clockwise with y to the right and z up: the front face is the straight
    edge at y = yf, the top the straight edge at z = z1 between the two top fillets, the back the straight edge at y = yb
    above height `knee` and a slope to the underside below it (`lean` m shallower at the bottom: the lower back leans in).
    r_*: fillet radii (front/back, top/bottom), `dish`: how far the middle of the underside is lifted (a quartic dish with
    a horizontal tangent at both ends). Straight edges longer than `step` get extra points, so the domed ends, which are
    built from scaled copies of this outline, bend smoothly along them."""
    ybl = yb - lean
    pts = [_arc((yf + r_fb, z0 + r_fb), r_fb, math.pi, 1.5 * math.pi, n)]
    a, b = yf + r_fb, ybl - r_bb
    if dish > 0:
        u = np.linspace(-1.0, 1.0, n_dish + 2)[1:-1]
        pts.append(np.stack([0.5 * (a + b) + 0.5 * (b - a) * u, z0 + dish * (1.0 - u * u) ** 2], 1))
    pts.append(_arc((ybl - r_bb, z0 + r_bb), r_bb, 1.5 * math.pi, 2.0 * math.pi, n))
    if lean > 0:                                              # slope up to the knee, then the vertical back
        pts.append(np.array([[ybl, z0 + r_bb], [yb, z0 + knee]]))
    pts.append(_arc((yb - r_bt, z1 - r_bt), r_bt, 0.0, 0.5 * math.pi, n))
    pts.append(_arc((yf + r_ft, z1 - r_ft), r_ft, 0.5 * math.pi, math.pi, n))
    P = np.concatenate(pts)
    keep = np.r_[True, np.linalg.norm(np.diff(P, axis=0), axis=1) > 1e-9]
    return resample(P[keep], step)


def resample(P, step):
    """The closed polygon P with extra points on every edge longer than `step` (evenly spaced)."""
    out = []
    for i in range(len(P)):
        a, b = P[i], P[(i + 1) % len(P)]
        m = max(1, int(math.ceil(float(np.linalg.norm(b - a)) / step)))
        out.append(a + (b - a) * (np.arange(m) / m)[:, None])
    return np.concatenate(out)


def cabinet(L, P, r_end, k=5, dome=0.0, rings=6, tol=1e-7):
    """Sweep the section P (counter-clockwise in (y, z)) along X from -L to +L. Each end is a quarter-round rim of radius
    `r_end` (the section moves in by r_end (1 - sin a) along its mitre vectors while x runs in by r_end (1 - cos a)), then a
    cap of `rings` concentric rings that rises `dome` beyond x = +-L toward the middle (a parabola). r_end must not exceed
    the smallest fillet radius of P. Returns welded (V (n, 3), Q (m, 4), T (k, 3)), normals outward."""
    P = np.asarray(P, float)
    n = len(P)
    M = S.mitre_offsets(P)
    ang = (0.5 * math.pi) * _graded(k)                       # 0 = the cap boundary, 90 deg = the full section

    def ring_pts(x, s):
        return np.concatenate([np.full((n, 1), x), P + M * s], 1)

    # rim rings: the +X end first (x runs in from +L), then the -X end (mirror): one sequence of rings of n points
    seq = [ring_pts(L - r_end * (1.0 - math.cos(a)), r_end * (1.0 - math.sin(a))) for a in ang]
    seq += [ring_pts(-L + r_end * (1.0 - math.cos(a)), r_end * (1.0 - math.sin(a))) for a in ang[::-1]]
    V = [np.concatenate(seq)]
    idx = np.arange(len(seq) * n).reshape(len(seq), n)
    j, j2 = np.arange(n), (np.arange(n) + 1) % n
    Q = [np.stack([idx[t][j], idx[t + 1][j], idx[t + 1][j2], idx[t][j2]], 1) for t in range(len(seq) - 1)]
    T = []
    nv = len(V[0])
    for sign, first in ((1, 0), (-1, len(seq) - 1)):          # the caps: scaled copies of the boundary ring, lifted by the dome
        b = seq[first]
        c = b[:, 1:].mean(0)
        chain = [idx[first]]
        for lm in (1.0 - np.arange(1, rings) / rings):
            V.append(np.concatenate([np.full((n, 1), sign * (L + dome * (1.0 - lm * lm))), c + (b[:, 1:] - c) * lm], 1))
            chain.append(nv + np.arange(n))
            nv += n
        V.append(np.array([[sign * (L + dome), c[0], c[1]]]))
        ci = nv
        nv += 1
        for t in range(len(chain) - 1):
            a, bb = chain[t], chain[t + 1]
            Q.append(np.stack([a[j], bb[j], bb[j2], a[j2]], 1)[:, ::-1 if sign > 0 else 1])
        last = chain[-1]
        tri = np.stack([last[j], np.full(n, ci), last[j2]], 1)
        T.append(tri[:, ::-1] if sign > 0 else tri)
    m = S.Mesh(np.concatenate(V), np.concatenate(Q), np.concatenate(T)).weld(tol)
    V2, Q2, T2 = m.V, m.Q, m.T
    if volume(V2, Q2, T2) < 0:                                 # safety net: a wrong global orientation
        Q2, T2 = Q2[:, ::-1], T2[:, ::-1]
    return V2, np.ascontiguousarray(Q2), np.ascontiguousarray(T2)


def volume(V, Q, T):
    tri = np.concatenate([T, Q[:, [0, 1, 2]], Q[:, [0, 2, 3]]])
    a, b, c = V[tri[:, 0]], V[tri[:, 1]], V[tri[:, 2]]
    return float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6.0)


def cap_lift(P, r_end, dome):
    """f(y, z) -> how far the domed end surface stands beyond the end plane x = +-L at (y, z) (0 on and outside the cap
    boundary): dome (1 - lam^2), lam = distance from the cap centre over the distance of the boundary in that direction. A
    small detail built on the plane x = +-L is carried onto the dome by moving each vertex by this much along X."""
    B = np.asarray(P, float) + S.mitre_offsets(P) * r_end
    c = B.mean(0)
    ang = np.arctan2(B[:, 1] - c[1], B[:, 0] - c[0])
    rad = np.hypot(B[:, 0] - c[0], B[:, 1] - c[1])
    order = np.argsort(ang)
    ang, rad = ang[order], rad[order]
    keep = np.r_[True, np.diff(ang) > 1e-9]
    ang, rad = ang[keep], rad[keep]

    def f(y, z):
        a = math.atan2(z - c[1], y - c[0])
        rho = float(np.interp(a, ang, rad, period=2.0 * math.pi))
        lam = min(math.hypot(y - c[0], z - c[1]) / max(rho, 1e-9), 1.0)
        return dome * (1.0 - lam * lam)
    return f
