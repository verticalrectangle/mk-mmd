"""Paper craft shared by the wall papers (`paper_bunting`, `taped_papers`; pure Python + numpy, no bpy: their layouts and
tests import it): sheets of paper as open grids, a strip of masking tape laid over whatever lies on the wall, and the
paper colours of a list of colour specs with a little tone between the pieces.

Frame of a wall prop (as `cafe_poster`'s): the wall is the plane y = 0, the room at -Y, +Z up, +X to the right as seen
from the room. Paper is a single sheet (both sides drawn by its material); everything stands a hair off what it lies on,
so nothing z-fights."""
import colorsys
import math

import numpy as np

from ....core import shell as S
from . import showkit as SK

TAPE_W = 0.019                  # masking tape 3/4 inch wide
TAPE_LIFT = 0.0004              # how far a strip stands in front of what it lies on
TEAR = 0.0011                   # the zigzag of a torn end


def facing_front(m):
    """The mesh with its faces turned to look toward -Y (the room): a sheet's winding follows its grid."""
    nq, nt = m.face_vectors()
    ny = (nq[:, 1].sum() if len(nq) else 0.0) + (nt[:, 1].sum() if len(nt) else 0.0)
    return m.flip() if ny > 0 else m


def sheet(rows, mat=0, uv=None):
    """An open sheet through a grid of points `rows` (ns, m, 3): quads between neighbouring rows, its faces toward -Y;
    `uv` (ns, m, 2) per point."""
    G = np.asarray(rows, float)
    m = S.cage(G, closed=False, mat=mat)
    if uv is not None:
        m.UV = np.asarray(uv, float).reshape(-1, 2)
    return facing_front(m)


def radial_sheet(outline, rings=6, mat=0):
    """An open sheet filling a closed outline (n, 2: x, z) that is star-shaped about its centroid: copies of the outline
    scaled toward the centroid and a fan round it; in the plane y = 0, its faces toward -Y."""
    P = np.asarray(outline, float)
    c = P.mean(0)
    s = np.linspace(0.0, 1.0, rings + 1)[1:]
    R = c + (P[None] - c) * s[:, None, None]                              # (rings, n, 2)
    n = len(P)
    V = np.vstack([[c[0], 0.0, c[1]], np.column_stack([R[..., 0].ravel(), np.zeros(rings * n), R[..., 1].ravel()])])
    j = np.arange(n)
    j2 = (j + 1) % n
    T = np.stack([np.zeros(n, np.int64), 1 + j, 1 + j2], 1)
    Q = [np.stack([1 + k * n + j, 1 + (k + 1) * n + j, 1 + (k + 1) * n + j2, 1 + k * n + j2], 1) for k in range(rings - 1)]
    return facing_front(S.Mesh(V, np.concatenate(Q) if Q else None, T, mat=mat))


def tape_strip(centre, angle, length, ground, rng, width=TAPE_W, mat=0, nc=8, nr=28):
    """A strip of masking tape on the wall centred at `centre` (x, z), its length along `angle` degrees (anticlockwise
    from +X as seen from the room), both ends torn in a zigzag. `ground(x, z)` (vectorised) is the y of the nearest
    surface toward the room under each point (the wall's 0, a sheet's -0.002); the strip lies TAPE_LIFT in front of it
    and bridges a step by running down off its edge. UV: u across, v along, in metres. Faces toward -Y."""
    th = math.radians(angle)
    e = np.array([math.cos(th), math.sin(th)])                            # along
    d = np.array([-math.sin(th), math.cos(th)])                           # across
    a = np.linspace(-0.5 * width, 0.5 * width, nc + 1)
    odd = np.arange(nc + 1) % 2
    t0 = TEAR * rng.uniform(0.35, 1.0, nc + 1) * odd
    t1 = TEAR * rng.uniform(0.35, 1.0, nc + 1) * (1 - odd)
    s = np.linspace(0.0, 1.0, nr + 1)
    s0, s1 = -0.5 * length + t0[:, None], 0.5 * length - t1[:, None]
    along = s0 + (s1 - s0) * s[None, :]                                   # (nc + 1, nr + 1)
    X = centre[0] + a[:, None] * d[0] + along * e[0]
    Z = centre[1] + a[:, None] * d[1] + along * e[1]
    Y = np.asarray(ground(X.ravel(), Z.ravel()), float).reshape(X.shape)
    for _ in range(3):                                                    # run down off a step, never behind it
        Ys = Y.copy()
        Ys[:, 1:-1] = (Y[:, :-2] + Y[:, 1:-1] + Y[:, 2:]) / 3.0
        Y = np.minimum(Y, Ys)
    V = np.stack([X, Y - TAPE_LIFT, Z], -1)
    UV = np.stack([np.broadcast_to(a[:, None] + 0.5 * width, X.shape), along + 0.5 * length], -1)
    return sheet(V, mat=mat, uv=UV)


def tones(specs, n, slots, rng, amount, what):
    """Linear RGB (n, 3): item i takes the colour spec specs[i % len(specs)] (a palette slot, #hex or `a:b:t`), its
    lightness moved by up to +-`amount` and its hue by a hair (seeded), so no two pieces of one colour are quite alike."""
    base = [SK.rgb(sp, slots, f"{what}[{k}]") for k, sp in enumerate(specs)]
    out = np.zeros((n, 3))
    for i in range(n):
        r, g, b = (max(0.0, c) ** (1.0 / 2.2) for c in base[i % len(base)])
        h, l, s = colorsys.rgb_to_hls(r, g, b)
        l = min(0.97, max(0.03, l * (1.0 + amount * rng.uniform(-1.0, 1.0))))
        h = (h + 0.012 * amount / 0.07 * rng.uniform(-1.0, 1.0)) % 1.0
        out[i] = [c ** 2.2 for c in colorsys.hls_to_rgb(h, l, s)]
    return out


def smoothstep(e0, e1, x):
    t = np.clip((np.asarray(x, float) - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)
