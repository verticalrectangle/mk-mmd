"""Shared helpers of the cat ears and tails (bpy-free, numpy only; numpy 1.24 and 2.x).

  palette(ctx)            the ears / tail colours of the spec ([colors.ears], defaults = RinStudy's colors.toml)
  toon(ctx, name, shadow) a warm toon ramp PNG written through `ctx.save_png` -> file name for `Material.toon`
  rmf(P, n0)              rotation-minimising frames along a polyline (a tube that never twists)
  pchip(x, y)             monotone cubic interpolation (planform edges as functions of height)
  euler_matrix(e)         Blender XYZ Euler (rad) -> rotation matrix, the convention of `RigidBody.rotation`
  body_sdf(rb, P)         signed distance from points to a static rigid body (sphere, capsule or box)
  capsule_clearance(...)  smallest gap between a capsule and a list of static bodies
  mesh_distance(P, V, F)  distance from points to a triangle mesh (signed by the nearest face normal)
All lengths are metres in model space."""
import numpy as np

from . import hair_tex as TEX
from .hair_geo import tangents, unit

COLORS = {
    "outer": "#1b1517", "outer_sheen": "#3d3236", "inner": "#8e2236", "inner_deep": "#5a1224", "tuft": "#f2e1d8",
    "tuft_shadow": "#c9aea4", "tail": "#1e181a", "tail_tip_sheen": "#463a3e",
}
# the fur is black with a faint cool sheen: the warm colours of the table pushed towards blue-violet
COOL = np.array([0.20, 0.21, 0.31])


def palette(ctx):
    """name -> float rgb (sRGB values) from `[colors.ears]` over COLORS, plus derived `cool` (bluish sheen)."""
    tab = dict(COLORS)
    got = ctx.get("colors.ears", {}) or {}
    tab.update({k: v for k, v in got.items() if k in COLORS and isinstance(v, str)})
    pal = {k: TEX.srgb(v) for k, v in tab.items()}
    pal["cool"] = COOL.copy()
    return pal


def toon(ctx, name, shadow):
    """Write a toon ramp whose shaded side is `shadow` (per-channel multiplier) and return the PNG file name."""
    return ctx.save_png(name, TEX.toon_ramp(shadow=tuple(shadow)))


def cfg_merge(base, over):
    """Defaults over which the spec table `over` is laid (nested dicts merged, other values replaced)."""
    out = dict(base)
    for k, v in (over or {}).items():
        out[k] = cfg_merge(out[k], v) if isinstance(out.get(k), dict) and isinstance(v, dict) else v
    return out


def srgb_u8(c):
    return tuple(float(x) for x in np.clip(np.asarray(c, float), 0.0, 1.0))


def darker(c, k=0.55):
    """A darker version of the colour (edge / outline colour): value scaled by k."""
    return tuple(float(x) for x in np.clip(np.asarray(c, float) * k, 0.0, 1.0))


# ---------------------------------------------------------------- curves and frames
def rmf(P, n0):
    """Rotation-minimising frames (double reflection, Wang et al. 2008) along the polyline P (m, 3) from the normal
    hint n0 (made perpendicular to the first tangent): T, N, B = T x N, all unit (m, 3)."""
    P = np.asarray(P, float)
    T = tangents(P)
    N = np.empty_like(T)
    n = np.asarray(n0, float)
    n = n - (n @ T[0]) * T[0]
    if np.linalg.norm(n) < 1e-6:
        n = np.cross(T[0], [1.0, 0.0, 0.0] if abs(T[0][0]) < 0.9 else [0.0, 1.0, 0.0])
    N[0] = unit(n)
    for i in range(len(P) - 1):
        v1 = P[i + 1] - P[i]
        c1 = float(v1 @ v1)
        if c1 < 1e-18:
            N[i + 1] = N[i]
            continue
        rl = N[i] - (2.0 / c1) * (v1 @ N[i]) * v1
        tl = T[i] - (2.0 / c1) * (v1 @ T[i]) * v1
        v2 = T[i + 1] - tl
        c2 = float(v2 @ v2)
        N[i + 1] = rl - (2.0 / c2) * (v2 @ rl) * v2 if c2 > 1e-18 else rl
    N = unit(N)
    B = unit(np.cross(T, N))
    return T, N, B


def pchip(x, y):
    """Monotone cubic Hermite interpolant (Fritsch-Carlson) through (x, y), x strictly increasing -> callable f(xq);
    outside the range the end values are held."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    h = np.diff(x)
    d = np.diff(y) / h
    m = np.zeros(len(x))
    if len(x) == 2:
        m[:] = d[0]
    else:
        m[1:-1] = np.where(d[:-1] * d[1:] > 0, 2.0 * d[:-1] * d[1:] / np.where(d[:-1] + d[1:] == 0, 1.0, d[:-1] + d[1:]),
                           0.0)
        m[0] = d[0]
        m[-1] = d[-1]

    def f(xq):
        xq = np.clip(np.asarray(xq, float), x[0], x[-1])
        i = np.clip(np.searchsorted(x, xq, side="right") - 1, 0, len(x) - 2)
        t = (xq - x[i]) / h[i]
        t2, t3 = t * t, t * t * t
        return ((2 * t3 - 3 * t2 + 1) * y[i] + (t3 - 2 * t2 + t) * h[i] * m[i] + (-2 * t3 + 3 * t2) * y[i + 1] +
                (t3 - t2) * h[i] * m[i + 1])
    return f


def rotate(v, axis, ang):
    """Rodrigues rotation of v about the unit axis by ang (rad)."""
    v = np.asarray(v, float)
    k = unit(axis)
    c, s = np.cos(ang), np.sin(ang)
    return v * c + np.cross(k, v) * s + k * (k @ v) * (1.0 - c)


# ---------------------------------------------------------------- the model's static colliders
def euler_matrix(e):
    """Rotation matrix of an XYZ Euler (rad), Blender's convention R = Rz Ry Rx (RigidBody.rotation)."""
    rx, ry, rz = (float(a) for a in e)
    cx, sx, cy, sy, cz, sz = np.cos(rx), np.sin(rx), np.cos(ry), np.sin(ry), np.cos(rz), np.sin(rz)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def body_sdf(rb, P):
    """Signed distance (< 0 inside) from points P (n, 3) to the rigid body `rb`: sphere (r), capsule (r, straight
    height, axis = local Z) or box (half extents); rotation and location as in the part contract."""
    P = np.atleast_2d(np.asarray(P, float))
    q = (P - np.asarray(rb.location, float)) @ euler_matrix(rb.rotation)
    s = rb.size
    if rb.shape == "sphere":
        return np.linalg.norm(q, axis=1) - s[0]
    if rb.shape == "capsule":
        h = 0.5 * s[1]
        dz = q[:, 2] - np.clip(q[:, 2], -h, h)
        return np.sqrt(q[:, 0] ** 2 + q[:, 1] ** 2 + dz ** 2) - s[0]
    if rb.shape == "box":
        d = np.abs(q) - np.asarray(s[:3], float)
        return np.linalg.norm(np.maximum(d, 0.0), axis=1) + np.minimum(d.max(axis=1), 0.0)
    raise ValueError(f"unknown rigid body shape {rb.shape!r}")


def static_colliders(ctx):
    """The body part's `col_*` static bodies (what mk's strand solver collides chains with)."""
    out = []
    for p in ctx.parts.values():
        out += [rb for rb in p.bodies if rb.mode == "static" and rb.name.startswith("col_")]
    return out


def capsule_clearance(bodies, a, b, r, step=0.006):
    """(gap, body name): smallest distance (m, negative = overlap) between the capsule a-b with radius r and `bodies`."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    n = max(3, int(np.ceil(np.linalg.norm(b - a) / step)) + 1)
    pts = a[None] + np.linspace(0.0, 1.0, n)[:, None] * (b - a)[None]
    best, who = np.inf, ""
    for rb in bodies:
        g = float(body_sdf(rb, pts).min()) - r
        if g < best:
            best, who = g, rb.name
    return best, who


def chain_clearance(bodies, pts, radii):
    """Per bone (gap, body name) of a chain running through pts (n + 1, 3) with capsule radii (n,)."""
    return [capsule_clearance(bodies, pts[k], pts[k + 1], float(radii[k])) for k in range(len(pts) - 1)]


def mesh_distance(P, V, F, chunk=24):
    """Distance (m) from points P (n, 3) to the triangle mesh (V, F), negative when the nearest point lies behind its
    face (inside a closed mesh with outward faces). F: index lists, ngons are fanned."""
    V = np.asarray(V, float)
    tris = np.array([[f[0], f[k], f[k + 1]] for f in F for k in range(1, len(f) - 1)], int)
    A, B, C = V[tris[:, 0]], V[tris[:, 1]], V[tris[:, 2]]
    ab, ac = B - A, C - A
    nrm = np.cross(ab, ac)
    nl = np.linalg.norm(nrm, axis=1)
    ok = nl > 1e-14
    A, B, C, ab, ac, nrm, nl = A[ok], B[ok], C[ok], ab[ok], ac[ok], nrm[ok], nl[ok]
    nun = nrm / nl[:, None]
    d00 = (ab * ab).sum(1)
    d01 = (ab * ac).sum(1)
    d11 = (ac * ac).sum(1)
    den = d00 * d11 - d01 * d01
    P = np.atleast_2d(np.asarray(P, float))
    out = np.empty(len(P))

    def seg(p, a, b):
        e = b - a
        t = np.clip(((p - a) * e).sum(-1) / np.maximum((e * e).sum(-1), 1e-18), 0.0, 1.0)
        return a + t[..., None] * e

    for s in range(0, len(P), chunk):
        p = P[s:s + chunk, None, :]                                   # (c, 1, 3)
        ap = p - A[None]
        dist_plane = (ap * nun[None]).sum(-1)
        q = p - dist_plane[..., None] * nun[None]
        v2 = q - A[None]
        d20 = (v2 * ab[None]).sum(-1)
        d21 = (v2 * ac[None]).sum(-1)
        v = (d11[None] * d20 - d01[None] * d21) / den[None]
        w = (d00[None] * d21 - d01[None] * d20) / den[None]
        inside = (v >= 0) & (w >= 0) & (v + w <= 1)
        c1 = seg(p, A[None], B[None])
        c2 = seg(p, B[None], C[None])
        c3 = seg(p, C[None], A[None])
        best = q
        bd = np.where(inside, 0.0, np.inf)
        for c in (c1, c2, c3):
            dd = np.linalg.norm(p - c, axis=-1)
            take = (~inside) & (dd < bd)
            best = np.where(take[..., None], c, best)
            bd = np.where(take, dd, bd)
        d = np.linalg.norm(p - best, axis=-1)
        k = d.argmin(1)
        rows = np.arange(len(k))
        side = ((p[:, 0, :] - best[rows, k]) * nun[k]).sum(-1)
        out[s:s + chunk] = np.where(side < 0, -d[rows, k], d[rows, k])
    return out
