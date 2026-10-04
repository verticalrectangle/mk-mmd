"""What the head publishes for the parts built after it (hair, outfit): the closed skin shell, the hairline, ear anchors,
eye and brow lines, the neck ring and the face region. All in MODEL space (the caller passes the frame)."""
import numpy as np

from . import head_skin as SK
from .head_shape import directions


def skin_shell(shape, ring_z, cfg=None):
    """A coarse closed copy of the head (cranium, face and neck, no eye or mouth holes), star-shaped about
    shape.centre: (verts, triangles) in head-local metres, outward winding."""
    c = dict(n_cols=48, face_deg=3.5, back_deg=9.0, ring_z=ring_z, spacing_face=0.0075, spacing_neck=0.008,
             spacing_top=0.010)
    c.update(cfg or {})
    g = SK.Grid(shape, c)
    F = []
    for r in range(len(g.rows) - 1):
        F += g.band_faces(r, r + 1)
    F += g.fan_faces()
    T = []
    for f in F:
        for k in range(1, len(f) - 1):
            T.append((int(f[0]), int(f[k]), int(f[k + 1])))
    return np.asarray(g.V, float), T


def surface_at_azimuth(shape, theta, z_from, z_to, n):
    """n points on the skull surface along the meridian at azimuth theta between two heights (head-local)."""
    c = shape.centre
    # elevation of the points on a rough cylinder of the skull radius, refined by the ray hit
    rho = 0.09
    ph = np.linspace(np.arctan2(z_from - c[2], rho), np.arctan2(z_to - c[2], rho), n)
    return shape.surface(directions(np.full(n, theta), ph))


def hairline(shape, centre_z, temple, n=41, side_n=14, jaw_z=-0.004, theta_sb=np.radians(71.0), margin=0.002):
    """(front (n, 3) left temple -> forehead -> right temple, side {"L","R"} (m, 3) temple -> jaw level in front of the
    ear, nape (k, 3) +x -> -x) in head-local metres. `temple` = (x, z) of the temple hairline."""
    from .head_skin import lift
    xt, zt = temple
    xt = min(xt, float(shape.prof.at(zt)[0]) - margin)
    t = np.linspace(1.0, -1.0, n)
    x = xt * t
    z = zt + (centre_z - zt) * (1.0 - np.abs(t) ** 2.2)
    front = lift(shape, x, z)
    side = {}
    for name, sg, k in (("L", 1.0, 0), ("R", -1.0, -1)):
        pts = surface_at_azimuth(shape, sg * theta_sb, zt, jaw_z, side_n)
        side[name] = np.vstack([front[k], pts])
    xn = np.linspace(0.062, -0.062, 17)
    zn = -0.006 + 0.020 * (xn / 0.062) ** 2
    c = shape.centre
    far = np.stack([xn, np.full_like(xn, 0.20), zn], -1)
    d = far - c
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    nape = shape.surface(d)
    return front, side, nape


def tangent_frame(shape, p, side_sign, tilt_out_deg=12.0):
    """The frame of a cat ear on the skull at the point nearest p (head-local): origin on the skin, unit normal, `up`
    along the surface towards the crown, `forward` along the surface towards the face (-y side), and `axis`, the suggested
    long axis of the ear: between the normal and straight up, leaning out by tilt_out_deg."""
    c = shape.centre
    d = np.asarray(p, float) - c
    d /= np.linalg.norm(d)
    o = shape.surface(d[None])[0]
    n = shape.normal(o[None])[0]
    up = np.array([0.0, 0.0, 1.0])
    up = up - n * (up @ n)
    up /= np.linalg.norm(up)
    fwd = np.cross(up, n) * side_sign
    fwd -= up * (fwd @ up)
    if fwd[1] > 0:
        fwd = -fwd
    fwd /= np.linalg.norm(fwd)
    axis = n + np.array([0.0, 0.0, 1.0])
    axis /= np.linalg.norm(axis)
    axis = axis + side_sign * np.tan(np.radians(tilt_out_deg)) * np.array([1.0, 0.0, 0.0])
    axis /= np.linalg.norm(axis)
    return o, n, up, fwd, axis


def jaw_line(shape, n=512, r=0.035):
    """Height of the jaw underside per azimuth (theta = ((i + 0.5) / n - 0.5) 2 pi, 0 = straight ahead): the lowest height at
    which the head (without the neck) reaches radius r from the neck axis. Head-local metres; the face texture paints the
    shadow under the jaw from it."""
    th = ((np.arange(n) + 0.5) / n - 0.5) * 2 * np.pi
    zs = np.linspace(-0.045, 0.09, 270)
    yc = float(shape.centre[1])
    X = np.broadcast_to((r * np.sin(th))[:, None], (n, len(zs)))
    Y = np.broadcast_to((yc - r * np.cos(th))[:, None], (n, len(zs)))
    Z = np.broadcast_to(zs[None, :], (n, len(zs)))
    inside = shape.phi(np.stack([X, Y, Z], -1), neck=False) < 0
    z = zs[np.argmax(inside, axis=1)].copy()
    z[~inside.any(axis=1)] = zs[-1]
    pad = np.concatenate([z[-4:], z, z[:4]])
    return np.convolve(pad, np.ones(9) / 9.0, "valid")
