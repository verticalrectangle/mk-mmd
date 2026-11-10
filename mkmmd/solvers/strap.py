"""Strap: the band of a worn prop's strap through a clip. Where it lies on its wearer she carries it, at the buttons the
prop holds it, and between the two it hangs: a soft band, not a bar. numpy only, never imports bpy, deterministic.

Model (a position-based band)
  points     the band's line as it was laid on her at rest (mkmmd.core.wear.strap_path), each of a kind: 0 held by the
             prop (a button, a run across its pin), 1 carried by the wearer (from the band's first touch on her to its
             last), 2 free (the runs between her and a button).
  per frame  held and carried points go where the prop and her skin take them (the build samples them: a helper mesh
             skinned like her). A free run is a little longer than the straight way between its ends (`slack`: a strap
             that carries a prop is nearly taut, and a band drawn dead straight reads as a bar), so it hangs in a
             shallow curve, swings when she turns and settles again.
  per substep  verlet prediction with gravity, the points' motion relative to their run's ends dying at DRAG -> a weak
             pull of every free point toward the midpoint of its neighbours (BEND: a band bends in curves, not kinks)
             -> ITERS rounds of: segment lengths (even and odd pairs in turn; held and carried ends do not move), the
             band's length from each free point to either end of its run (never farther from an end than the band
             between them is long: rounds of neighbours alone let a long run stretch like rubber) and her surface
             under the run (the plane through the point of her skin that lay under each free point at rest, carried
             by her: a slack run lies on her chest or her hip, `standoff` off it).
  settle     the first frame is held for `settle_s` seconds, so the runs hang before the clip starts.
  faces      a held or carried point faces as the prop or her skin turns it; along a free run the facing goes from one
             end's to the other's and leans into the run's bow (a hanging band bends about its flat side: one that
             sags edge on rolls over).

Files (python -m mkmmd.solvers.strap IN.npz OUT.npz; mkmmd.blender.build.sim writes IN):
  IN   arrays rest (n, 3) the line at rest; kind (n,); held, carried (F, n, 3) each point as the prop and as her skin
       take it, world, on every frame; held_n, carried_n (F, n, 3) its facing likewise; under (F, n, 3) the point of
       her skin under it (NaN: nothing under it); spec (JSON text): fps, substeps, slack, standoff, settle_s.
  OUT  x (F, n, 3) and n (F, n, 3): the line and its facing, world; report (JSON text): points, free, runs,
       bow_mm_max (how far a free run hangs from the straight way between its ends), stretch_pct_max, seconds."""
import json
import sys
import time

import numpy as np

VERSION = 1
ITERS = 12            # constraint rounds per substep
DRAG = 9.0            # 1/s: a free point's motion relative to its run's ends dies at this rate (a band is lossy)
BEND = 0.08           # share of a free point's offset from its neighbours' midpoint removed per substep
BOW = 0.03            # m: a run bowed this far from the straight way faces half way into its bow
GRAVITY = np.array([0.0, 0.0, -9.81])


def _unit(v):
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12)


def runs_of(kind):
    """The free runs: [(first, last)] index pairs of consecutive points of kind 2 (the point before `first` and the one
    after `last` hold the run)."""
    free = np.asarray(kind) == 2
    if free[0] or free[-1]:
        raise ValueError("a strap's ends are held: its first and last points cannot be free")
    edge = np.flatnonzero(np.diff(free.astype(int)))
    return [(int(a) + 1, int(b)) for a, b in zip(edge[::2], edge[1::2])]


class Band:
    def __init__(self, rest, kind, slack, standoff):
        self.kind = np.asarray(kind, int)
        self.free = self.kind == 2
        self.n = len(self.kind)
        seg = np.linalg.norm(np.diff(np.asarray(rest, float), axis=0), axis=1)
        self.loose = self.free[:-1] | self.free[1:]
        self.L = seg * np.where(self.loose, 1.0 + float(slack), 1.0)
        self.standoff = float(standoff)
        self.runs = runs_of(self.kind)
        self.u = np.zeros(self.n)                                # how far along its run a free point is: 0 .. 1
        self.a = np.arange(self.n)                               # the points that hold its run
        self.b = np.arange(self.n)
        self.sa, self.sb = np.zeros(self.n), np.zeros(self.n)    # the band's length from a free point to each of them
        for i0, i1 in self.runs:
            d = np.cumsum(seg[i0 - 1:i1 + 1])
            self.u[i0:i1 + 1] = d[:-1] / d[-1]
            self.a[i0:i1 + 1], self.b[i0:i1 + 1] = i0 - 1, i1 + 1
            long = np.cumsum(self.L[i0 - 1:i1 + 1])
            self.sa[i0:i1 + 1], self.sb[i0:i1 + 1] = long[:-1], long[-1] - long[:-1]
        # a free point's place when nothing has moved it yet: between the prop's idea of it and her skin's, by which end
        ka, kb = self.kind[self.a], self.kind[self.b]
        self.to_her = np.where(self.free, np.where(ka == 1, 1.0 - self.u, 0.0) + np.where(kb == 1, self.u, 0.0), 0.0)
        self.pairs = [np.array([i for i in range(par, self.n - 1, 2) if self.loose[i]], int) for par in (0, 1)]
        self.x = self.xp = None

    def targets(self, held, carried):
        """Where the prop and her skin put the points (free ones: where they start from)."""
        w = np.where(self.kind == 1, 1.0, self.to_her)[:, None]
        return held * (1.0 - w) + carried * w

    def start(self, tgt):
        self.x, self.xp = tgt.copy(), tgt.copy()
        self.tgt = tgt.copy()

    def step(self, dt, tgt, under, up):
        """One substep: `tgt` the held and carried points now, `under` (n, 3) her skin under each point (NaN: none) and
        `up` (n, 3) the way out of her there."""
        f, x = self.free, self.x
        moved = tgt - self.tgt                                   # what the ends did this substep
        ref = moved[self.a] * (1.0 - self.u)[:, None] + moved[self.b] * self.u[:, None]
        v = ref + (x - self.xp - ref) * max(0.0, 1.0 - DRAG * dt)
        new = np.where(f[:, None], x + v + GRAVITY * dt * dt, tgt)
        has = f & np.isfinite(under).all(1)
        fi = np.flatnonzero(f[1:-1]) + 1
        if len(fi):
            new[fi] += BEND * (0.5 * (new[fi - 1] + new[fi + 1]) - new[fi])
        for _ in range(ITERS):
            for i in self.pairs:
                if len(i):
                    d = new[i + 1] - new[i]
                    dist = np.maximum(np.linalg.norm(d, axis=1), 1e-9)
                    wa, wb = f[i].astype(float), f[i + 1].astype(float)
                    c = ((dist - self.L[i]) / dist / (wa + wb))[:, None] * d
                    new[i] += c * wa[:, None]
                    new[i + 1] -= c * wb[:, None]
            for end, far in ((self.a, self.sa), (self.b, self.sb)):
                d = new[f] - new[end[f]]
                dist = np.maximum(np.linalg.norm(d, axis=1), 1e-9)
                new[f] -= d * (np.maximum(dist - far[f], 0.0) / dist)[:, None]
            if has.any():
                short = self.standoff - ((new[has] - under[has]) * up[has]).sum(1)
                new[has] += up[has] * np.maximum(short, 0.0)[:, None]
        self.xp, self.x, self.tgt = x, new, tgt.copy()

    def faces(self, held_n, carried_n):
        """The band's facing (n, 3): the prop's and her skin's at the points they hold, and along a free run from one
        end's to the other's, leaning into the run's bow."""
        N = np.where((self.kind == 0)[:, None], held_n, carried_n).astype(float)
        x = self.x
        for i0, i1 in self.runs:
            a, b = i0 - 1, i1 + 1
            u = self.u[i0:i1 + 1, None]
            n = N[a] * (1.0 - u) + N[b] * u
            chord = _unit(x[b] - x[a])
            off = x[i0:i1 + 1] - x[a]
            bow = off - (off @ chord)[:, None] * chord
            lean = np.where((n * bow).sum(1, keepdims=True) < 0.0, -1.0, 1.0)
            N[i0:i1 + 1] = _unit(n) + lean * bow / BOW
        T = _unit(np.gradient(x, axis=0))
        N = N - (N * T).sum(1, keepdims=True) * T
        return _unit(N)

    def measure(self):
        """(the largest bow of a free run from the straight way between its ends (m), the largest stretch of a segment
        of one over its length (a ratio))."""
        x, bow = self.x, 0.0
        for i0, i1 in self.runs:
            chord = _unit(x[i1 + 1] - x[i0 - 1])
            off = x[i0:i1 + 1] - x[i0 - 1]
            bow = max(bow, float(np.linalg.norm(off - (off @ chord)[:, None] * chord, axis=1).max()))
        seg = np.linalg.norm(np.diff(x, axis=0), axis=1)
        return bow, float((seg[self.loose] / self.L[self.loose]).max() - 1.0) if self.loose.any() else 0.0


def simulate(rest, kind, held, carried, held_n, carried_n, under, fps=30.0, substeps=8, slack=0.006, standoff=0.007,
             settle_s=1.0):
    """(x (F, n, 3), faces (F, n, 3), report dict): the band through every frame (module docstring)."""
    t0 = time.perf_counter()
    held, carried, under = np.asarray(held, float), np.asarray(carried, float), np.asarray(under, float)
    band = Band(rest, kind, slack, standoff)
    F = len(held)
    dt = 1.0 / (float(fps) * int(substeps))

    def frame(f):
        up = _unit(carried[f] - under[f])                        # her skin carries both: out of her, as she turns
        return band.targets(held[f], carried[f]), under[f], up

    tgt, und, up = frame(0)
    band.start(tgt)
    for _ in range(int(round(float(settle_s) * float(fps) * int(substeps)))):
        band.step(dt, tgt, und, up)
    x, faces = np.zeros((F, band.n, 3)), np.zeros((F, band.n, 3))
    worst = [0.0, 0.0]
    for f in range(F):
        a = frame(f - 1) if f else (tgt, und, up)
        b = frame(f)
        for s in range(int(substeps)):
            w = (s + 1) / int(substeps)
            band.step(dt, a[0] + (b[0] - a[0]) * w, a[1] + (b[1] - a[1]) * w, _unit(a[2] + (b[2] - a[2]) * w))
        x[f], faces[f] = band.x, band.faces(held_n[f], carried_n[f])
        worst = [max(o, m) for o, m in zip(worst, band.measure())]
    rep = {"version": VERSION, "points": int(band.n), "free": int(band.free.sum()), "runs": len(band.runs),
           "bow_mm_max": round(worst[0] * 1000, 1), "stretch_pct_max": round(worst[1] * 100, 2),
           "seconds": round(time.perf_counter() - t0, 2)}
    return x, faces, rep


# ---------------------------------------------------------------- command line
def main(argv=None):
    """python -m mkmmd.solvers.strap IN.npz OUT.npz (module docstring: the file formats). Returns the exit code."""
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) != 2:
        print("usage: python -m mkmmd.solvers.strap IN.npz OUT.npz", file=sys.stderr)
        return 2
    try:
        with np.load(argv[0], allow_pickle=False) as z:
            a = {k: z[k] for k in z.files}
        spec = json.loads(str(a["spec"]))
        x, faces, rep = simulate(a["rest"], a["kind"], a["held"], a["carried"], a["held_n"], a["carried_n"], a["under"],
                                 float(spec.get("fps", 30.0)), int(spec.get("substeps", 8)), float(spec.get("slack", 0.006)),
                                 float(spec.get("standoff", 0.007)), float(spec.get("settle_s", 1.0)))
    except (KeyError, ValueError, OSError, json.JSONDecodeError) as e:
        print(f"strap: bad input: {type(e).__name__}: {e}", file=sys.stderr)
        return 2
    np.savez(argv[1], x=x.astype(np.float32), n=faces.astype(np.float32), report=np.array(json.dumps(rep)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
