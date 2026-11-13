"""What falling hair keeps off (numpy and scipy's KD-tree): the body's skin under the neck seam and the garments built
before the hair, as a field that is cheap to ask many times while a lock's path is found.

`Surround.away(X)` says how far outside the nearest surface each point is and which way is out. The surfaces are
sampled once (their vertices and their triangles' middles, with outward normals) and a point is measured against the
plane of the sample nearest to it, which is exact to a fraction of a millimetre at the few millimetres hair lies off
skin sampled every centimetre or closer. The body's skin is closed and its winding says what is out; a garment is a thin
open sheet, so each of its samples faces away from the body's skin nearest to it (the inner layer of a folded band
too), and it only counts where the point is over it: within `REACH` of the sample across its plane, so hair beside or
below a garment's edge is not pushed by it. Skin that no garment covers is `cloth` metres thicker: the room left for
clothes that are not built before the hair (COVER tells what a garment covers). `Surround.keep_off` moves a lock's
path until all of it is clear.

`Drape` is the same body and garments as a heavy mass of hair lies on them: one smooth surface round a vertical axis
through her neck that does not follow folds, frills, seams or the arms, and that hair moves straight out from, a
whole column at a time (`Surround.drape` makes it)."""
import numpy as np
from scipy.spatial import cKDTree

from .hair_geo import arclen, frames, smooth_polyline, taper_factor, unit
from .outfit_fit import Skin

REACH = 0.02                    # metres across a garment sample's plane within which a point is over the garment
COVER = 0.04                    # a skin sample with a garment's sample this near outside it is covered by the garment
FAR = 1.0e3
DRAPE = {"step": 0.01, "bins": 120, "round": 3, "down": 3, "slope": 0.25}   # a drape's rows (m) and angles round its
#                                 axis; the bins round and the rows down its reach is widened and averaged over; how
#                                 fast (m per m) it may recede going down


def _box(R, k, axis, wrap):
    """The mean of R over 2k + 1 neighbours along `axis` (round it when `wrap`, else the edges repeated)."""
    if k < 1:
        return R
    if wrap:
        return np.mean([np.roll(R, s, axis=axis) for s in range(-k, k + 1)], axis=0)
    pad = [(0, 0)] * R.ndim
    pad[axis] = (k, k)
    E = np.pad(R, pad, mode="edge")
    n = R.shape[axis]
    return np.mean([np.take(E, np.arange(s, s + n), axis=axis) for s in range(2 * k + 1)], axis=0)


def _spread(R, k, axis, wrap):
    """The max of R over 2k + 1 neighbours along `axis` (round it when `wrap`)."""
    if k < 1:
        return R
    if wrap:
        return np.max([np.roll(R, s, axis=axis) for s in range(-k, k + 1)], axis=0)
    pad = [(0, 0)] * R.ndim
    pad[axis] = (k, k)
    E = np.pad(R, pad, mode="edge")
    n = R.shape[axis]
    return np.max([np.take(E, np.arange(s, s + n), axis=axis) for s in range(2 * k + 1)], axis=0)


class Drape:
    """How far out the body and the garments reach round a vertical axis (`axis`, (x, y)), at every height up to
    `z_hi` (rows `step` apart) and angle (`bins` round), from their surface `points`: widened over `round` bins and
    `down` rows and averaged back over as many (a hanging sheet bridges folds and frills instead of following them),
    and receding no faster than `slope` going down (hair that hangs from a bulge bridges the hollow under it, as long
    hair hangs from the shoulder blades past the small of the back). `clear(X, gap)` moves points straight out from
    the axis until they lie `gap` over it; `normal(X)` is the way it faces there."""

    def __init__(self, points, axis, z_hi, step=DRAPE["step"], bins=DRAPE["bins"], round_=DRAPE["round"],
                 down=DRAPE["down"], slope=DRAPE["slope"]):
        self.ax, self.ay = float(axis[0]), float(axis[1])
        self.step, self.bins = float(step), int(bins)
        P = np.asarray(points, float)
        P = P[P[:, 2] < z_hi]
        self.z0 = float(P[:, 2].min())
        rows = int(np.ceil((float(z_hi) - self.z0) / self.step)) + 1
        i = np.clip(np.rint((P[:, 2] - self.z0) / self.step).astype(int), 0, rows - 1)
        j = np.rint(self._angle(P) / (2.0 * np.pi) * self.bins).astype(int) % self.bins
        R = np.full((rows, self.bins), -np.inf)
        np.maximum.at(R, (i, j), self._radius(P))
        full = np.isfinite(R)
        grid = np.arange(self.bins)
        for r in range(rows):                               # bins nothing fell in: from their neighbours round
            if full[r].any():
                R[r] = np.interp(grid, grid[full[r]], R[r, full[r]], period=self.bins)
        has = full.any(1)
        for r in np.flatnonzero(~has):                      # rows nothing fell in: the nearest row's
            R[r] = R[np.flatnonzero(has)[np.argmin(np.abs(np.flatnonzero(has) - r))]]
        R = _spread(_spread(R, int(round_), 1, True), int(down), 0, False)
        R = _box(_box(R, int(round_), 1, True), int(down), 0, False)
        for r in range(rows - 2, -1, -1):                   # going down it recedes no faster than `slope`
            R[r] = np.maximum(R[r], R[r + 1] - float(slope) * self.step)
        self.R, self.z_hi = R, float(z_hi)

    def _angle(self, X):
        return np.arctan2(X[:, 1] - self.ay, X[:, 0] - self.ax) % (2.0 * np.pi)

    def _radius(self, X):
        return np.hypot(X[:, 0] - self.ax, X[:, 1] - self.ay)

    def reach(self, X):
        """How far out from the axis the drape is at the heights and angles of the points X (n, 3); NaN above it."""
        X = np.asarray(X, float)
        fi = (X[:, 2] - self.z0) / self.step
        fj = self._angle(X) / (2.0 * np.pi) * self.bins
        i0 = np.clip(np.floor(fi).astype(int), 0, len(self.R) - 2)
        a = np.clip(fi - i0, 0.0, 1.0)
        j0 = np.floor(fj).astype(int) % self.bins
        j1 = (j0 + 1) % self.bins
        b = fj - np.floor(fj)
        R = self.R
        r = (1 - a) * ((1 - b) * R[i0, j0] + b * R[i0, j1]) + a * ((1 - b) * R[i0 + 1, j0] + b * R[i0 + 1, j1])
        return np.where(X[:, 2] <= self.z_hi, r, np.nan)

    def clear(self, X, gap):
        """X (n, 3) with every point that lies less than `gap` over the drape moved straight out from the axis to it."""
        X = np.array(X, float)
        need = self.reach(X) + gap
        r = self._radius(X)
        move = np.isfinite(need) & (r < need)
        if move.any():
            d = np.stack([X[move, 0] - self.ax, X[move, 1] - self.ay], -1) / np.maximum(r[move], 1e-9)[:, None]
            X[move, :2] = np.array([self.ax, self.ay]) + d * need[move][:, None]
        return X

    def over(self, X):
        """How far each point of X lies over the drape (FAR above it)."""
        r = self.reach(X)
        return np.where(np.isfinite(r), self._radius(np.asarray(X, float)) - np.nan_to_num(r), FAR)

    def normal(self, X):
        """The unit outward normal of the drape at the heights and angles of X (n, 3)."""
        X = np.asarray(X, float)
        a, h = self._angle(X), 0.5 * 2.0 * np.pi / self.bins
        da = 2.0 * np.pi / self.bins
        r = np.nan_to_num(self.reach(X))
        ra = (np.nan_to_num(self.reach(self._at(X, a + h))) - np.nan_to_num(self.reach(self._at(X, a - h)))) / da
        up, dn = X.copy(), X.copy()
        up[:, 2] += 0.5 * self.step
        dn[:, 2] -= 0.5 * self.step
        rz = (np.nan_to_num(self.reach(up)) - np.nan_to_num(self.reach(dn))) / self.step
        radial = np.stack([np.cos(a), np.sin(a), np.zeros_like(a)], -1)
        around = np.stack([-np.sin(a), np.cos(a), np.zeros_like(a)], -1)
        n = radial - (ra / np.maximum(r, 1e-6))[:, None] * around - rz[:, None] * np.array([0.0, 0.0, 1.0])
        return unit(n)

    def _at(self, X, a):
        """X's points turned round the axis to the angles `a`, at their own radius and height."""
        r = self._radius(X)
        out = np.array(X, float)
        out[:, 0], out[:, 1] = self.ax + r * np.cos(a), self.ay + r * np.sin(a)
        return out


def _samples(skin, share=None):
    """(points, outward unit normals) of a Skin: its vertices and its triangles' middles; with `share` (per vertex)
    also that per sample (a middle's is its corners' mean)."""
    fn, _, corner = skin._pseudo_normals()
    mid = skin.verts[skin.tris].mean(1)
    out = np.vstack([skin.verts, mid]), np.vstack([unit(corner), unit(fn)])
    return out if share is None else (*out, np.concatenate([share, share[skin.tris].mean(1)]))


def _share(skin, bones):
    """Per vertex of a Skin, the share of its weight on `bones`."""
    tot = np.zeros(len(skin.verts))
    for b in bones:
        if b in skin.weights:
            tot += skin.weights[b]
    return tot


class Surround:
    """The body (`body_meshes`, open at the neck seam `neck_top`: {z, center, rx, ry}) and the garments (`wear_meshes`)
    round the falling hair; `cloth` metres over the skin no garment covers; `arms` the bones of her arms (and of what
    hangs from them), which a drape leaves out."""

    def __init__(self, body_meshes, wear_meshes, neck_top, cloth=0.0, arms=()):
        (self.cx, self.cy) = neck_top["center"]
        self.rx, self.ry, self.z_seam = float(neck_top["rx"]), float(neck_top["ry"]), float(neck_top["z"])
        skin = Skin.from_meshes([m for m in body_meshes if len(m.verts)], "body")
        self.bp, self.bn, self.b_arm = _samples(skin, _share(skin, arms))
        self.body = cKDTree(self.bp)
        self.room = np.full(len(self.bp), float(cloth))     # metres each skin sample is taken to be thicker
        self.wear = None
        wear = [m for m in wear_meshes if len(m.verts)]
        if wear:
            cloth_skin = Skin.from_meshes(wear, "wear")
            self.wp, wn, self.w_arm = _samples(cloth_skin, _share(cloth_skin, arms))
            nb = self.bn[self.body.query(self.wp)[1]]
            self.wn = wn * np.where(np.einsum("ij,ij->i", wn, nb) < 0.0, -1.0, 1.0)[:, None]
            self.wear = cKDTree(self.wp)
            if cloth > 0.0:
                d, j = self.wear.query(self.bp)
                over = np.einsum("ij,ij->i", self.wp[j] - self.bp, self.bn) > -0.005
                self.room[(d < COVER) & over] = 0.0

    def drape(self):
        """The Drape of what lies under the falling hair (round the vertical axis through the neck seam's middle):
        the skin `room` thicker where it has it and the garments, under the neck seam, the arms left out."""
        pts = [(self.bp + self.bn * self.room[:, None])[self.b_arm < 0.5]]
        if self.wear is not None:
            pts.append(self.wp[self.w_arm < 0.5])
        return Drape(np.vstack(pts), (self.cx, self.cy), self.z_seam)

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
        db = np.einsum("ij,ij->i", Q - self.bp[i], self.bn[i]) - self.room[i]
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

    def keep_off(self, P, hint, width, thick, cup, tip_start, gap, first=0, rounds=24, step=0.005, tip="point"):
        """The path P (n, 3) of a lock moved until the lock is `gap` clear: its middle's inner side and both its edges
        (`width` across, tapering from `tip_start` of its length into its `tip` (hair_geo.taper_factor), `thick` deep,
        its edges bent `cup` toward the body; `hint(P)` the way it faces) are measured, each point of the path moved by
        its worst, at most `step` a round, the way that is out at its own middle (the way out of a point deep inside
        says little), and the path smoothed between rounds. Points before index `first` (the hair on the head) stay."""
        P = np.array(P, float)
        if not self.measured(P).any():
            return P
        n = len(P)
        idx = np.arange(n)
        for it in range(rounds):
            _, B, N = frames(P, hint(P))
            w = width * taper_factor(arclen(P) / max(arclen(P)[-1], 1e-9), tip, tip_start, 1.0)
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
