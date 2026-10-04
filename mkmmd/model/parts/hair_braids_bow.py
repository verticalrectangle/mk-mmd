"""A ribbon bow tied round a braid (bpy-free, numpy only): a band wrapped round the plait, a knot on its face, two puffy
folded loops (wings) and two short swallow-tailed tails.

All pieces are closed thin shells (the outline is drawn from closed shells). Frame of a bow: `t` down the braid, `n` the
way the bow faces (the plait's face turned by the bow yaw about t), `w` across (the wings' span, away from the midline for
the outer wing), see `bow_frame`. Texture coordinates come from hair_braids_tex.ribbon_uv."""
import numpy as np

from . import hair_braids_tex as TX
from .hair_braids_geo import orient_outward, rot_axis
from .hair_braids_plait import Block, ellipsoid_shell
from .hair_geo import unit

# planform of one wing (x along the span 0 = knot .. 1 = tip, y across -1 .. 1): a fan that flares from the knot
WING_RIM = np.array([(0.00, -0.40), (0.22, -0.58), (0.50, -0.82), (0.78, -0.97), (0.95, -0.80), (1.00, -0.38),
                     (1.00, 0.38), (0.95, 0.80), (0.78, 0.97), (0.50, 0.82), (0.22, 0.58), (0.00, 0.40)])


def bow_frame(plan, sigma, yaw_deg, roll_deg=0.0):
    """(origin on the centreline, t, n, w): t down the braid, n = the plait's face turned about t by yaw (towards the
    outside for positive yaw), w = t x n (across, outer wing positive), all rotated about n by roll (wings tilt)."""
    p, T, B, N = plan.at(sigma)
    g = np.radians(yaw_deg)
    n = unit(np.cos(g) * N + np.sin(g) * B)
    w = unit(np.cross(T, n))
    r = np.radians(roll_deg)
    R = rot_axis(n, r)
    return p, T, n, unit(R @ w), unit(R @ T)


def ellipse_radius(a, b, psi):
    """Radius of the ellipse with semi-axes a (along B) and b (along N) in the direction at angle psi from N towards B."""
    return 1.0 / np.sqrt((np.cos(psi) / b) ** 2 + (np.sin(psi) / a) ** 2)


# ---------------------------------------------------------------- band and knot
def band_block(O, T, B, N, a, b, h, thick, n_around=14):
    """A flat ribbon ring (closed shell, lens section) round the braid: ellipse semi-axes a along B, b along N, height h along T,
    thickness `thick`."""
    ang = 2.0 * np.pi * np.arange(n_around) / n_around
    verts, uv = [], []
    for th in ang:
        c = O + a * np.sin(th) * B + b * np.cos(th) * N
        rad = unit(np.sin(th) / a * B + np.cos(th) / b * N)
        for dv, du, vv in ((0.5 * thick * rad, 0.5, 0.5), (0.5 * h * T, 0.5, 0.97), (-0.5 * thick * rad, 0.5, 0.5),
                           (-0.5 * h * T, 0.5, 0.03)):
            verts.append(c + dv)
            uv.append((0.045 + 0.02 * du, vv))
    faces = []
    for j in range(n_around):
        k = (j + 1) % n_around
        for i in range(4):
            i2 = (i + 1) % 4
            faces.append([j * 4 + i, k * 4 + i, k * 4 + i2, j * 4 + i2])
    verts = np.array(verts)
    return Block("band", verts, orient_outward(verts, faces), TX.ribbon_uv(*np.array(uv).T))


def knot_block(c, t, n, kt, kw, kd):
    """The knot: a rounded block centred on c, `kt` semi-height along t, `kw` across, `kd` along the facing direction n."""
    verts, faces, u, v = ellipsoid_shell(c, t, n, kt, kw, kd, n_around=8, n_rings=3, p=0.7)
    return Block("knot", verts, faces, TX.ribbon_uv(0.045 + 0.03 * v, 0.25 + 0.5 * u))


# ---------------------------------------------------------------- wings
def wing_block(O, w, t, n, L, H, th, curl=0.16, fold=0.42):
    """One loop: a closed puffy lens over the fan planform WING_RIM scaled to span L and half height H, root at O, span along
    w, height along t, puffed by `th` (full thickness) along n and curling away from the viewer by `curl` (fraction of L at
    the tip). A dimple (`fold` of the puff at the centre) gives the folded-ribbon read."""
    rim = WING_RIM
    cen = np.array([0.50, 0.0])
    inner = cen + (rim - cen) * 0.52
    pts2 = np.vstack([rim, inner, cen[None]])
    z_front = np.concatenate([np.zeros(len(rim)), np.full(len(inner), 0.5 * th * 0.78), [0.5 * th * fold]])
    curve = -curl * L * pts2[:, 0] ** 2
    nr, ni = len(rim), len(inner)
    P_rim = O + (pts2[:nr, 0, None] * L) * w + (pts2[:nr, 1, None] * H) * t + (curve[:nr, None]) * n
    P_in_f = O + (pts2[nr:, 0, None] * L) * w + (pts2[nr:, 1, None] * H) * t + (curve[nr:, None] + z_front[nr:, None]) * n
    P_in_b = O + (pts2[nr:, 0, None] * L) * w + (pts2[nr:, 1, None] * H) * t + (curve[nr:, None] - z_front[nr:, None]) * n
    verts = np.vstack([P_rim, P_in_f, P_in_b])
    uv_xy = np.vstack([pts2[:nr], pts2[nr:], pts2[nr:]])
    uvs = TX.ribbon_uv(uv_xy[:, 0], 0.5 + 0.5 * uv_xy[:, 1])
    ri = lambda i: i % nr
    fi = lambda i: nr + (i % ni)          # inner ring, front
    bi = lambda i: nr + ni + 1 + (i % ni)  # inner ring, back
    fc, bc = nr + ni, nr + 2 * ni + 1     # centre points
    faces = []
    for i in range(nr):
        faces.append([ri(i), ri(i + 1), fi(i + 1), fi(i)])
        faces.append([ri(i + 1), ri(i), bi(i), bi(i + 1)])
        faces.append([fi(i), fi(i + 1), fc])
        faces.append([bi(i + 1), bi(i), bc])
    return Block("wing", verts, orient_outward(verts, faces), uvs)


# ---------------------------------------------------------------- tails
def tail_block(path, face, width, thick, notch, w_end=0.9):
    """A short flat ribbon along `path` (m, 3) with a V-notched end: rows of a lens section (left edge, front, right edge,
    back); the end row's middle points are pulled back by `notch`. `face` is the direction the ribbon's front looks."""
    path = np.asarray(path, float)
    m = len(path)
    d = np.gradient(path, axis=0)
    d = unit(d)
    verts, uv = [], []
    for i in range(m):
        f = unit(face - np.dot(face, d[i]) * d[i])
        a = unit(np.cross(d[i], f))
        wi = width * (1.0 + (w_end - 1.0) * i / (m - 1))
        last = i == m - 1
        pull = -notch * d[i] if last else 0.0
        v0 = i / (m - 1)
        verts += [path[i] - 0.5 * wi * a, path[i] + 0.5 * thick * f + pull, path[i] + 0.5 * wi * a, path[i] - 0.5 * thick * f + pull]
        uv += [(0.2 + 0.78 * v0, 0.0), (0.2 + 0.78 * v0, 0.5), (0.2 + 0.78 * v0, 1.0), (0.2 + 0.78 * v0, 0.5)]
    faces = []
    for i in range(m - 1):
        for j in range(4):
            j2 = (j + 1) % 4
            faces.append([i * 4 + j, i * 4 + j2, (i + 1) * 4 + j2, (i + 1) * 4 + j])
    faces.append([3, 2, 1, 0])
    faces.append([(m - 1) * 4 + k for k in range(4)])
    verts = np.array(verts)
    uv = np.array(uv)
    return Block("tail", verts, orient_outward(verts, faces), TX.ribbon_uv(uv[:, 0], uv[:, 1]))
