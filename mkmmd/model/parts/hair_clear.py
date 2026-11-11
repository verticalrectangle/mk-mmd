"""What falling hair keeps off (numpy and scipy's KD-tree): the body's skin under the neck seam and the garments built
before the hair, as a field that is cheap to ask many times while a lock's path is found.

`Surround.away(X)` says how far outside the nearest surface each point is and which way is out. The surfaces are
sampled once (their vertices and their triangles' middles, with outward normals) and a point is measured against the
plane of the sample nearest to it, which is exact to a fraction of a millimetre at the few millimetres hair lies off
skin sampled every centimetre or closer. The body's skin is closed and its winding says what is out; a garment is a thin
open sheet, so each of its samples faces away from the body's skin nearest to it (the inner layer of a folded band
too), and it only counts where the point is over it: within `REACH` of the sample across its plane, so hair beside or
below a garment's edge is not pushed by it. `Surround.keep_off` moves a lock's path until all of it is clear."""
import numpy as np
from scipy.spatial import cKDTree

from .hair_geo import arclen, frames, smooth_polyline, taper_factor, unit
from .outfit_fit import Skin

REACH = 0.02                    # metres across a garment sample's plane within which a point is over the garment
FAR = 1.0e3


def _samples(skin):
    """(points, outward unit normals) of a Skin: its vertices and its triangles' middles."""
    fn, _, corner = skin._pseudo_normals()
    mid = skin.verts[skin.tris].mean(1)
    return np.vstack([skin.verts, mid]), np.vstack([unit(corner), unit(fn)])


class Surround:
    """The body (`body_meshes`, open at the neck seam `neck_top`: {z, center, rx, ry}) and the garments (`wear_meshes`)
    round the falling hair."""

    def __init__(self, body_meshes, wear_meshes, neck_top):
        (self.cx, self.cy) = neck_top["center"]
        self.rx, self.ry, self.z_seam = float(neck_top["rx"]), float(neck_top["ry"]), float(neck_top["z"])
        self.bp, self.bn = _samples(Skin.from_meshes([m for m in body_meshes if len(m.verts)], "body"))
        self.body = cKDTree(self.bp)
        self.wear = None
        wear = [m for m in wear_meshes if len(m.verts)]
        if wear:
            self.wp, wn = _samples(Skin.from_meshes(wear, "wear"))
            nb = self.bn[self.body.query(self.wp)[1]]
            self.wn = wn * np.where(np.einsum("ij,ij->i", wn, nb) < 0.0, -1.0, 1.0)[:, None]
            self.wear = cKDTree(self.wp)

    def measured(self, X):
        """The points of X the body says something about: under the neck seam, or beside it (the seam's own ellipse is
        the neck's open top, the head's)."""
        over = (X[:, 2] >= self.z_seam) & (((X[:, 0] - self.cx) / self.rx) ** 2 + ((X[:, 1] - self.cy) / self.ry) ** 2 <= 1.0)
        return (X[:, 2] < self.z_seam + 0.004) & ~over

    def away(self, X):
        """(clearance (n,), out (n, 3)): how far outside the nearest surface each point of X is (negative inside the body
        or under a garment) and the unit way out there. Points the body says nothing about are FAR clear."""
        X = np.asarray(X, float)
        d, out = np.full(len(X), FAR), np.zeros_like(X)
        m = np.flatnonzero(self.measured(X))
        if not len(m):
            return d, out
        Q = X[m]
        i = self.body.query(Q)[1]
        db = np.einsum("ij,ij->i", Q - self.bp[i], self.bn[i])
        nb = self.bn[i]
        if self.wear is not None:
            j = self.wear.query(Q)[1]
            v = Q - self.wp[j]
            dw = np.einsum("ij,ij->i", v, self.wn[j])
            across = np.linalg.norm(v - dw[:, None] * self.wn[j], axis=1)
            on = (across < REACH) & (dw < db) & (dw > -0.05)
            db = np.where(on, dw, db)
            nb = np.where(on[:, None], self.wn[j], nb)
        d[m], out[m] = db, nb
        return d, out

    def keep_off(self, P, hint, width, thick, cup, tip_start, gap, first=0, rounds=24, step=0.005):
        """The path P (n, 3) of a lock moved until the lock is `gap` clear: its middle's inner side and both its edges
        (`width` across, tapering from `tip_start` of its length into its point, `thick` deep, its edges bent `cup`
        toward the body; `hint(P)` the way it faces) are measured, each point of the path moved by its worst, at most
        `step` a round, the way that is out at its own middle (the way out of a point deep inside says little), and the
        path smoothed between rounds. Points before index `first` (the hair on the head) stay."""
        P = np.array(P, float)
        if not self.measured(P).any():
            return P
        n = len(P)
        idx = np.arange(n)
        for it in range(rounds):
            _, B, N = frames(P, hint(P))
            w = width * taper_factor(arclen(P) / max(arclen(P)[-1], 1e-9), "point", tip_start, 1.0)
            dip = 0.125 * cup * w * w
            X = np.concatenate([P - N * (0.45 * thick), P + B * (0.5 * w)[:, None] - N * dip[:, None],
                                P - B * (0.5 * w)[:, None] - N * dip[:, None]])
            d, out = self.away(X)
            need = (gap - d).reshape(3, n)
            k = need.argmax(0)
            worst = need[k, idx]
            worst[:first] = 0.0
            if worst.max() < 1e-4:
                break
            out = out.reshape(3, n, 3)
            way = np.where((d[:n] < FAR)[:, None], out[0], out[k, idx])
            P = P + way * np.clip(worst, 0.0, step)[:, None]
            if it < rounds - 6 and n - first > 4:              # the last rounds only push: the clearance is exact
                P[first:] = smooth_polyline(P[max(first - 1, 0):], 1)[first - max(first - 1, 0):]
        return P
