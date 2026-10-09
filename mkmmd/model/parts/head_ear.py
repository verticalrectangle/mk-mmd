"""Human ears: a small standing shell with a raised rim and a dimple, planted on the side of the skull (mostly hidden by
the hair). Head-local metres; the left ear (+x) is built, the right is its mirror."""
import numpy as np

from .. import spec as SP

EAR = dict(
    centre=(0.0027, 0.0463),            # (y, z) of the ear centre in head-local metres
    size=(0.0112, 0.0182),              # half width (y) and half height (z) of the ear outline
    tilt_deg=14.0,                      # the top leans back by this much
    rim=0.0062, dimple=0.0030,          # rim crest height and the dimple height over the skin
    rings=(0.0, 0.28, 0.52, 0.74, 0.88, 1.0, 1.10), spokes=20,
)


def skin_x(shape, y, z, hi=0.16, iters=36):
    """x of the head surface on the +x side at (y, z) (bisection along x; the shape is star-shaped)."""
    y = np.asarray(y, float)
    z = np.asarray(z, float)
    lo = np.zeros_like(y)
    hi = np.full_like(y, hi)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        inside = shape.phi(np.stack([mid, y, z], -1)) < 0.0
        lo = np.where(inside, mid, lo)
        hi = np.where(inside, hi, mid)
    return 0.5 * (lo + hi)


def _height(r, rim, dimple, base=-0.0010):
    """Profile of the ear over the skin against the normalised radius r (0 centre, 1 rim crest, 1.1 the skin again)."""
    r = np.asarray(r, float)
    bowl = dimple + (rim - dimple) * np.clip(r / 0.88, 0, 1) ** 2.2          # dimple to the rim crest
    crest = np.where(r <= 1.0, bowl, rim * (1.0 - np.clip((r - 1.0) / 0.10, 0, 1) ** 1.2) + base * np.clip((r - 1.0) / 0.10, 0, 1))
    return crest


def ear(shape, cfg=None, n_rings=None):
    """(verts (n, 3), faces, info) of the left ear. Faces are CCW seen from outside (+x). `info` has the anchor points:
    top, lobe, front, back (on the rim), centre (the dimple), out (unit vector)."""
    c = SP.merge(EAR, cfg)
    rings = np.asarray(c["rings"], float)
    ns = int(c["spokes"])
    cy, cz = c["centre"]
    a, b = c["size"]
    tilt = np.radians(c["tilt_deg"])
    P = []
    ang = 2 * np.pi * np.arange(ns) / ns
    for r in rings:
        for t in ang:
            if r == 0.0 and t > 0:
                continue
            # outline: an egg, narrower at the bottom (the lobe) and rounder on top
            ey, ez = np.cos(t), np.sin(t)
            wy = a * (1.0 + 0.10 * ez)
            wz = b * (1.0 - 0.12 * (ez < 0) * (1.0 - ez * ez))
            y0, z0 = r * wy * ey, r * wz * ez
            y = cy + y0 * np.cos(tilt) + z0 * np.sin(tilt)
            z = cz - y0 * np.sin(tilt) + z0 * np.cos(tilt)
            x = float(skin_x(shape, np.array([y]), np.array([z]))[0]) + float(_height(r, c["rim"], c["dimple"]))
            P.append((x, y, z))
    P = np.array(P)
    # indices
    idx = {}
    k = 0
    for ri, r in enumerate(rings):
        for si in range(ns):
            if r == 0.0 and si > 0:
                idx[(ri, si)] = idx[(ri, 0)]
                continue
            idx[(ri, si)] = k
            k += 1
    F = []
    for ri in range(len(rings) - 1):
        for si in range(ns):
            s1 = (si + 1) % ns
            a0, a1, b0, b1 = idx[(ri, si)], idx[(ri, s1)], idx[(ri + 1, si)], idx[(ri + 1, s1)]
            F.append((a0, b0, b1) if a0 == a1 else (a0, a1, b1, b0))
    # make sure the winding points outward (+x): flip if the first face normal has a negative x component
    f0 = F[len(F) // 2]
    n = np.cross(P[f0[1]] - P[f0[0]], P[f0[2]] - P[f0[0]])
    if n[0] < 0:
        F = [tuple(reversed(f)) for f in F]
    ring_rim = [idx[(list(rings).index(1.0), si)] for si in range(ns)]
    rim = P[ring_rim]
    info = dict(top=rim[np.argmax(rim[:, 2])], lobe=rim[np.argmin(rim[:, 2])], front=rim[np.argmin(rim[:, 1])],
                back=rim[np.argmax(rim[:, 1])], centre=P[idx[(0, 0)]], out=np.array([1.0, 0.0, 0.0]))
    return P, F, info
