"""Eyebrows: thin tapered strips on the forehead above each eye (decals), and their morphs as changes of the centre line.
Head-local metres; the left brow (+x) is built, the right is its mirror."""
import numpy as np

BROW = dict(
    x=(0.0256, 0.0768),               # inner and outer end (x of the left brow)
    z=0.0995,                         # height of the brow line over the head bone
    arch=0.0034,                      # how far the middle rises over the ends
    slope=-0.0016,                    # outer end lower than the inner end by this much
    thick=0.0040,                     # width at the inner end
    n=16,
    height=0.0004,                    # over the skin
)


def _sm(t):
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def centreline(cfg=None, dz=0.0, arch_scale=1.0, d_inner=0.0, d_outer=0.0, dx=0.0, x_scale=1.0, thick_scale=1.0):
    """Centre line (n, 2) in (x, z) of the left brow, and its width per point."""
    c = dict(BROW)
    c.update(cfg or {})
    s = np.linspace(0.0, 1.0, c["n"])
    x0, x1 = c["x"]
    xm = 0.5 * (x0 + x1)
    x = xm + (x0 + (x1 - x0) * s - xm) * x_scale + dx
    z = (c["z"] + c["arch"] * arch_scale * np.sin(np.pi * s) ** 0.85 + c["slope"] * (s - 0.5) + dz
         + d_inner * (1 - s) ** 1.3 + d_outer * s ** 1.3)
    w = c["thick"] * thick_scale * (0.35 + 0.65 * (1 - s) ** 0.7) * (1 - s ** 6)
    return np.stack([x, z], -1), w


def edges(P, w):
    """Lower and upper edge (n, 2) of the strip from a centre line and widths."""
    t = np.gradient(P, axis=0)
    t /= np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-12)
    n = np.stack([-t[:, 1], t[:, 0]], -1)                   # up
    return P - 0.5 * w[:, None] * n, P + 0.5 * w[:, None] * n


# morph name -> centreline parameters (see `centreline`); metres
MORPHS = {
    "brow_up": dict(dz=0.0050),
    "brow_down": dict(dz=-0.0042),
    "cheerful": dict(dz=0.0020, arch_scale=1.7, d_outer=-0.0030),
    "serious": dict(dz=-0.0018, arch_scale=0.35, d_inner=-0.0008),
    "troubled": dict(dz=0.0008, arch_scale=0.7, d_inner=0.0058, d_outer=-0.0024, dx=-0.0012),
    "angry": dict(dz=-0.0012, arch_scale=0.5, d_inner=-0.0062, d_outer=0.0030, x_scale=0.95),
    "sad": dict(dz=0.0010, arch_scale=0.8, d_inner=0.0052, d_outer=-0.0034, dx=0.0008),
    "surprised": dict(dz=0.0078, arch_scale=1.4),
    "jito": dict(dz=-0.0024, arch_scale=0.45, d_inner=-0.0014),
}
