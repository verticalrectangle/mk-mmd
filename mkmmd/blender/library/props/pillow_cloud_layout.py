"""Layout of the pillow cloud (pure Python + numpy, no bpy: importable from tests): the pillow, the cloud it puffs into,
the colour roles and the card.

Frame (metres): origin at the pillow's centre, held upright as a person hugs it: its broad faces look along -Y (the
front, toward whoever faces the hugger) and +Y (the back, against the chest), X across (W), Z up (H). The pillow is two
fabric panels sewn together round their edge: each panel is a grid of (NU + 1) x (NV + 1) points over the plan, the seam
is the grid's boundary ring that the two panels share, and the stuffing gives each panel its height off the mid-plane,
T/2 x P(u) x P(v) (P: PROFILE, a plateau that falls steeply to the seam, so the panels meet there at an angle and the
seam reads as a pinched ridge), rippled by folds at the ears (WRINKLE). The sides pull in at their middles (BOW) and
the corners stand out as the ears. A cord is sewn into the seam (PIPING).

The cloud is the same points moved onto a cartoon cumulus (LOBES: balls of fluff smoothly united, a flat soft base at
CLOUD_BASE): each panel's grid is laid over a hemisphere (`directions`), and every point casts a ray from the cloud's
centre that way and lands on the cloud's outermost crossing. The two shapes have one topology, so the cloud is a shape
key of the pillow; the root property `puff` (0 the pillow, 1 the cloud, a little over 1 an overshoot) is its value. The
cord sinks into the cloud as a thread."""
import math

import numpy as np

from ....core import shell as S
from . import convertible_palette as CP

W, T, H = 0.56, 0.17, 0.38          # the pillow: width (x), thickness (y), height (z)
BOW = (0.07, 0.10)                  # the sides pull in at their middles by this share of the half width / half height
NU, NV = 60, 42                     # quads across (x) and up (z) on each panel: the cloud's creases need them
EDGE_DENSE = 0.45                   # 0: an even grid; 1: rows bunched hard at the seam (sin spacing)
PROFILE = ((0.0, 1.0), (0.5, 0.975), (0.75, 0.91), (0.88, 0.76), (0.96, 0.48), (1.0, 0.0))    # panel height / (T / 2)
WRINKLE = (0.35, 7.0, 0.42)         # the ears' fabric folds: depth (share of the panel's height there), folds, reach
PIPING = (0.0045, 0.02, 3)          # the cord in the seam: radius; how far in it sinks into the cloud; passes easing the ears

# the cloud: balls (x, y, z, r) united smoothly, a soft flat base; centre of the rays and the box the directions stretch to
LOBES = np.array([
    (0.00, 0.00, 0.08, 0.27), (-0.30, 0.00, -0.01, 0.20), (0.31, 0.00, 0.00, 0.21),                 # the big domes
    (-0.50, 0.01, -0.08, 0.125), (0.52, 0.01, -0.07, 0.13),                                          # the ends
    (-0.17, 0.01, 0.23, 0.13), (0.18, -0.01, 0.21, 0.14),                                            # the crown
    (-0.13, -0.13, -0.04, 0.17), (0.16, -0.14, -0.05, 0.16), (0.40, -0.08, -0.08, 0.12),             # the front puffs
    (0.10, 0.13, 0.00, 0.18), (-0.19, 0.12, -0.03, 0.16), (-0.40, 0.07, -0.07, 0.12),                # the back puffs
])
CLOUD_K = 0.011                     # smooth-union radius: the creases between the puffs
CLOUD_BASE, BASE_K = -0.17, 0.022   # the flat underside's height and how softly it rolls into the puffs
CLOUD_O = (0.0, 0.0, 0.0)           # the rays' origin, inside the cloud
CLOUD_HALF = (0.62, 0.30, 0.30)     # the box the pillow's directions are stretched to (half sizes)

ROLES = {
    "case": "foam",                 # the pillowcase
    "dots": "surface",              # its polka dots
    "piping": "rose",               # the cord sewn into its seam
    "cloud": "surface:4,base:1",    # the cloud's fluff
    "shade": "iris:1,surface:2",    # the cloud's shaded underside
}
PARAMS = {"puff": (0.0, (0.0, 1.25), "0 the pillow, 1 the cloud it puffs into (a little over 1 overshoots)")}


class Pal(CP.Pal):
    """The pillow's colour roles (`ROLES`) over the palette: each a slot, a hex or a 'slot:weight,...' blend."""

    def __init__(self, slots=None):
        super().__init__(slots)
        slots = dict(slots or {})
        self.specs = {r: str(slots.get(r, d)) for r, d in ROLES.items()}


def options(slots):
    """The starting `puff` from the prop's slots, checked."""
    v = dict(slots or {}).get("puff", PARAMS["puff"][0])
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
        raise ValueError(f"pillow_cloud: puff is a number (0 the pillow .. 1 the cloud), got {v!r}")
    if not -0.25 <= v <= 1.5:
        raise ValueError(f"pillow_cloud: puff is 0 (the pillow) .. 1 (the cloud), a little over 1 at most, got {v:g}")
    return {"puff": float(v)}


# ===================================================================================================================
# the pillow
# ===================================================================================================================
def spacing(n, dense=EDGE_DENSE):
    """n + 1 parameters from -1 to 1, closer together toward both ends (the seam and the ears)."""
    t = np.linspace(-1.0, 1.0, n + 1)
    return (1.0 - dense) * t + dense * np.sin(0.5 * math.pi * t)


def profile(s):
    """P(s): the stuffed panel's height over the mid-plane as a share of T / 2, at |s| (0 the middle, 1 the seam)."""
    p = np.asarray(PROFILE, float)
    return S.pchip(p[:, 0], p[:, 1:], np.clip(np.abs(np.asarray(s, float)), 0.0, 1.0).ravel())[:, 0].reshape(np.shape(s))


def plan(u, v):
    """(x, z) of the grid parameters (u, v) in [-1, 1]^2: the outline's sides pull in at their middles."""
    return 0.5 * W * u * (1.0 - BOW[0] * (1.0 - v * v)), 0.5 * H * v * (1.0 - BOW[1] * (1.0 - u * u))


def lift(u, v):
    """The panel's height over the mid-plane at (u, v): the stuffing's profile, rippled by the ears' folds (a share of the
    height, so the seam stays on the mid-plane)."""
    depth, n, reach = WRINKLE
    ripple = np.zeros(np.shape(u))
    for cu in (-1.0, 1.0):
        for cv in (-1.0, 1.0):
            r = np.hypot(cu - u, cv - v) / reach                        # from the corner, 1 at the folds' reach
            a = np.arctan2(np.abs(cv - v), np.abs(cu - u))              # 0 .. pi/2 round the corner
            ripple = ripple + depth * np.sin(math.pi * np.clip(r, 0.0, 1.0)) * np.sin(n * 2.0 * a)
    return 0.5 * T * profile(u) * profile(v) * (1.0 + ripple)


def topology():
    """(index grids of the front and back panels (NU + 1, NV + 1), number of vertices): the boundary ring is shared."""
    front = np.arange((NU + 1) * (NV + 1)).reshape(NU + 1, NV + 1)
    back = front.copy()
    inner = np.zeros_like(front, bool)
    inner[1:-1, 1:-1] = True
    back[inner] = front.size + np.arange(inner.sum())
    return front, back, front.size + int(inner.sum())


def _developed(P):
    """uv of a panel's grid of points (NU + 1, NV + 1, 3) as its fabric lies flat: the arc length along each grid line
    from the panel's middle lines, over the pillow's width and height (so the print keeps its size up the walls)."""
    su = np.concatenate([np.zeros((1, P.shape[1])), np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=2), 0)], 0)
    sv = np.concatenate([np.zeros((P.shape[0], 1)), np.cumsum(np.linalg.norm(np.diff(P, axis=1), axis=2), 1)], 1)
    return 0.5 + (su - su[NU // 2][None, :]) / W, 0.5 + (sv - sv[:, NV // 2][:, None]) / H


def pillow_mesh():
    """The pillow: both panels as quads (material 0), outward, with a uv per vertex (the fabric laid flat, the same print
    on both panels: see `_developed`)."""
    front, back, n = topology()
    u, v = np.meshgrid(spacing(NU), spacing(NV), indexing="ij")
    x, z = plan(u, v)
    h = lift(u, v)
    V = np.zeros((n, 3))
    UV = np.zeros((n, 2))
    for grid, side in ((front, -1.0), (back, 1.0)):
        P = np.stack([x, side * h, z], -1)
        V[grid.ravel()] = P.reshape(-1, 3)
        UV[grid.ravel()] = np.stack(_developed(P), -1).reshape(-1, 2)
    i, j = np.meshgrid(np.arange(NU), np.arange(NV), indexing="ij")
    qf = np.stack([front[i, j], front[i + 1, j], front[i + 1, j + 1], front[i, j + 1]], -1).reshape(-1, 4)  # -Y
    qb = np.stack([back[i, j], back[i, j + 1], back[i + 1, j + 1], back[i + 1, j]], -1).reshape(-1, 4)      # +Y
    return S.Mesh(V, np.concatenate([qf, qb]), UV=UV)


def directions():
    """A unit direction per vertex (n, 3) for the cloud's rays: each panel's grid squeezed onto a disc (the elliptical
    square-to-disc map), the disc laid over a hemisphere (its rim the seam on the equator, the front panel's middle at
    -Y, the back's at +Y), then stretched to the cloud's box. Smooth and one-to-one, so the cloud never folds; even in
    angle, so the rows next to the seam stay next to it on the cloud."""
    front, back, n = topology()
    u, v = np.meshgrid(spacing(NU), spacing(NV), indexing="ij")
    xd, zd = u * np.sqrt(1.0 - 0.5 * v * v), v * np.sqrt(1.0 - 0.5 * u * u)
    r = np.minimum(np.hypot(xd, zd), 1.0)
    theta, phi = 0.5 * math.pi * r, np.arctan2(zd, xd)
    D = np.zeros((n, 3))
    for grid, side in ((front, -1.0), (back, 1.0)):
        d = np.stack([np.sin(theta) * np.cos(phi), side * np.cos(theta), np.sin(theta) * np.sin(phi)], -1)
        D[grid.ravel()] = d.reshape(-1, 3)
    D = D * np.asarray(CLOUD_HALF)
    return D / np.linalg.norm(D, axis=1)[:, None]


def seam_ring():
    """The seam's vertex ids in order round the pillow (counter-clockwise seen from the front), each once."""
    front, _, _ = topology()
    return np.concatenate([front[:-1, 0], front[-1, :-1], front[:0:-1, -1], front[0, :0:-1]])


def _smoothed(P, rounds=PIPING[2]):
    """A closed polyline with its sharpest turns (the ears) eased: `rounds` passes of neighbour averaging."""
    P = np.asarray(P, float)
    for _ in range(int(rounds)):
        P = P + 0.5 * (0.5 * (np.roll(P, 1, 0) + np.roll(P, -1, 0)) - P)
    return P


def ring_tube(path, radius, sides=8, mat=1):
    """A closed round tube along a closed path (n, 3) lying in the x-z plane: n rings of `sides` vertices (ring i's
    vertex k is i * sides + k), each ring joined to the next and the last to the first, faces outward."""
    P = np.asarray(path, float)
    n = len(P)
    T = np.roll(P, -1, 0) - np.roll(P, 1, 0)
    T /= np.linalg.norm(T, axis=1)[:, None]
    B = np.broadcast_to(np.array([0.0, 1.0, 0.0]), T.shape)            # the path's plane normal: no twist round it
    U = np.cross(B, T)
    a = 2.0 * math.pi * np.arange(sides) / sides
    V = P[:, None, :] + radius * (np.cos(a)[None, :, None] * U[:, None, :] + np.sin(a)[None, :, None] * B[:, None, :])
    i, k = np.meshgrid(np.arange(n), np.arange(sides), indexing="ij")
    i1, k1 = (i + 1) % n, (k + 1) % sides
    Q = np.stack([i * sides + k, i * sides + k1, i1 * sides + k1, i1 * sides + k], -1).reshape(-1, 4)
    return S.Mesh(V.reshape(-1, 3), Q, mat=mat)


def piping_meshes(pillow_V, cloud_V):
    """(the cord sewn into the seam (material 1), its positions in the cloud): a tube of PIPING[0] along the seam ring;
    in the cloud it shrinks to a thread PIPING[1] under the cloud's surface (along its normal) at the ring's cloud
    positions, so it sinks out of sight as the pillow puffs."""
    ring = seam_ring()
    cord = ring_tube(_smoothed(pillow_V[ring]), PIPING[0])
    c = cloud_V[ring]
    inside = _smoothed(c - PIPING[1] * cloud_normals(c))
    inside[:, 1] = 0.0                                                   # the seam's plane (y = 0), as the cord's
    return cord, ring_tube(inside, 1e-4).V


# ===================================================================================================================
# the cloud
# ===================================================================================================================
def cloud_sdf(P):
    """Signed distance (approximate, m; negative inside) of points (n, 3) to the cloud: the balls united with a smooth
    minimum (CLOUD_K), then cut flat underneath at CLOUD_BASE with a soft roll (BASE_K)."""
    P = np.asarray(P, float).reshape(-1, 3)
    d = np.linalg.norm(P[:, None, :] - LOBES[None, :, :3], axis=2) - LOBES[None, :, 3]
    m = d.min(1)
    u = m - CLOUD_K * np.log(np.exp(-(d - m[:, None]) / CLOUD_K).sum(1))
    b = CLOUD_BASE - P[:, 2]                                             # > 0 below the base
    hi = np.maximum(u, b)
    return hi + BASE_K * np.log(np.exp((u - hi) / BASE_K) + np.exp((b - hi) / BASE_K))


def cloud_normals(P, e=1e-5):
    """Unit outward normals of the cloud at points (n, 3): the distance's gradient (central differences)."""
    P = np.asarray(P, float).reshape(-1, 3)
    g = np.stack([cloud_sdf(P + e * a) - cloud_sdf(P - e * a) for a in np.eye(3)], 1)
    return g / np.linalg.norm(g, axis=1)[:, None]


def cloud_rays(dirs, t_max=1.0, min_step=0.002, iters=18):
    """Distances along unit directions (n, 3) from CLOUD_O to the cloud's outermost crossing: traced in from t_max (each
    step 0.8 of the distance to the cloud, so no puff is jumped) to the first point inside, then bisected."""
    o = np.asarray(CLOUD_O, float)
    d = np.asarray(dirs, float).reshape(-1, 3)
    t = np.full(len(d), float(t_max))
    prev = t.copy()
    live = np.arange(len(d))
    for _ in range(600):
        if not len(live):
            break
        f = cloud_sdf(o + t[live, None] * d[live])
        out = live[f >= 0.0]
        prev[out] = t[out]
        t[out] -= np.maximum(0.8 * f[f >= 0.0], min_step)
        live = out
        if np.any(t[live] <= 0.0):
            raise ValueError("pillow_cloud: a ray from the cloud's centre never enters it")
    if len(live):
        raise ValueError("pillow_cloud: the cloud's rays did not converge")
    lo, hi = t, prev                                                     # inside at lo, outside at hi
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        inside = cloud_sdf(o + mid[:, None] * d) < 0.0
        lo, hi = np.where(inside, mid, lo), np.where(inside, hi, mid)
    return 0.5 * (lo + hi)


def cloud_points():
    """The cloud's position of every pillow vertex (n, 3): the cloud's outermost crossing along its `directions`."""
    d = directions()
    return np.asarray(CLOUD_O) + cloud_rays(d)[:, None] * d


def meshes():
    """(the pillow mesh, the cloud's vertex positions (n, 3) in the same order)."""
    return pillow_mesh(), cloud_points()


# ===================================================================================================================
# the card
# ===================================================================================================================
def _r(v, nd=4):
    return [round(float(x), nd) for x in v]


def ends():
    """x of the pillow's two ends at mid-height (the seam at the sides' middles)."""
    return 0.5 * W * (1.0 - BOW[0])


def card(name, roles=None, start=None, cloud=None):
    """The prop card (docs/design.md: Prop card): `hug` (the middle of its back, `dir` the way its front faces), the
    palms' planes on its two ends (`hold_L` on +X: the hugger's left hand when she faces -Y like the pillow), the centre
    to look at, the live `puff` and the cloud's own size and bounds (`cloud`: (lo, hi) of its points)."""
    xe = ends()
    rest = [{"name": nm, "type": "plane", "center": [s * round(xe, 4), 0.0, 0.0], "normal": [s, 0.0, 0.0],
             "size": [T, round(0.5 * H, 4)]} for nm, s in (("hold_L", 1.0), ("hold_R", -1.0))]
    out = {"size": [W, T, H], "origin": "center", "front": "-Y", "slots": dict(roles or {}),
           "use": {"anchor": [{"name": "hug", "point": [0.0, round(0.5 * T, 4), 0.0], "dir": [0.0, -1.0, 0.0]}],
                   "rest": rest, "look": [{"name": "centre", "point": [0.0, 0.0, 0.0]}]},
           "params": dict(start or {k: v[0] for k, v in PARAMS.items()}),
           "objects": [f"{name}_pillow", f"{name}_piping"]}
    if cloud is not None:
        lo, hi = np.min(cloud, 0), np.max(cloud, 0)
        out["cloud"] = {"size": _r(hi - lo, 3), "bounds": {"min": _r(lo), "max": _r(hi)}}
    return out
