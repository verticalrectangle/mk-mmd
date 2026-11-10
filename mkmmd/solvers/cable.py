"""Cable: a cord hanging from a moving plug (a guitar lead from its jack), swung by the plug's motion: numpy only, never
imports bpy, deterministic.

Model (a position-based rope)
  particles  points SEG apart along the cord, resampled from the hung shape (mkmmd.core.cable.hang's control points,
             smoothed): the first is the plug, the second is held out of the plug along its direction (a cord leaves its
             socket straight), the last PIN_TAIL lie where they are on the floor (the cord runs on to an amp out of the
             picture) until the cord is pulled taut on them: then they give and are dragged after the wearer (held for
             good, the cord had to stretch instead: 30 times its length when its wearer walked a metre off).
  per substep  verlet prediction with gravity and air drag -> ITERS rounds of: segment lengths (even and odd pairs in
             turn, both ends move unless held), a weak pull of every inner particle toward the midpoint of its
             neighbours (BEND: a lead is springy, it bends in curves, not kinks), the floor and, every COLLIDE_EVERY-th
             round and after the last, the wearer's collision bodies (mkmmd.solvers.geom shapes riding on their bones)
             -> floor friction (Coulomb: a touching particle's slide loses MU g dt^2, and one under STICK of that is
             cancelled: a cord lying on the floor stays put until it is pulled, and then slides as a dragged weight
             does) -> the cord never stretches: from the
             plug to the tail, a particle further than a segment (and TAUT of one) from the one before it is brought
             back onto it along the line between them (what the rounds had not yet passed down a taut cord), then the
             floor and the bodies once more.
  settle     the first frame is held for `settle_s` seconds, so the cord drops from the designed shape into its own
             before the clip starts.
  enables    particles within ANCHOR_FREE metres of the plug along the cord do not collide with bodies (the jack sits
             against the guitar, in front of the hips).
  output     x (F, N, 3): world positions on every frame of `frames`.

Files (python -m mkmmd.solvers.cable IN.npz OUT.npz; mkmmd.blender.build.sim writes IN):
  IN   arrays shape (M, 3) the hung cord's control points (world, the first at the plug); src_pos (F, S, 3) and
       src_quat (F, S, 4) the sources' world frames on every frame (as geom); spec (JSON text): sources ([kind, owner,
       name], geom.Shapes order), plug (index of the plug's source), dir_src and dir (the cord's direction out of the
       plug, in that source's frame), bodies (rig.json bodies to collide with), armature, floor_z, radius, fps,
       substeps, settle_s.
  OUT  x (F, N, 3) and report (JSON text): points, length_m, segment_mm, stretch_pct_max, body_pen_mm_max,
       floor_pen_mm_max, seconds."""
import json
import math
import sys
import time

import numpy as np

from . import geom

VERSION = 2
SEG = 0.03            # m between particles
ITERS = 16            # constraint rounds per substep
COLLIDE_EVERY = 4     # body collisions every this many rounds, and after the last
DRAG = 2.5            # 1/s: the cord's velocity relaxes toward still air (a lead is lossy: a swing dies in a second)
BEND = 0.12           # share of an inner particle's offset from its neighbours' midpoint removed per round
MU = 0.5              # the cord's friction on the floor: a slide loses MU g dt^2 per substep, as a weight dragged does (the
                      # floor took half of any slide back: glue, and a pulled cord lifted taut instead of sliding) ...
STICK = 1.8           # ... and a slide under this many times that is held: lying still, a cord stays where it lies
PIN_TAIL = 2          # particles at the far end held where they lie, until the cord is pulled taut on them
TAUT = 0.02           # share of its length a hanging segment may be over before it is brought in behind the one above it
DRAG_AT = 0.15        # ... and a segment lying on the floor: that much pull drags it over the floor (less is the rounds')
ANCHOR_FREE = 0.12    # m along the cord from the plug without body collisions
MARGIN = 0.002        # m added to the cord's radius for collisions
GRAVITY = np.array([0.0, 0.0, -9.81])


def smooth(points, rounds=3):
    """Chaikin corner cutting of a polyline, its ends kept: control points -> close to the curve they make."""
    P = np.asarray(points, float)
    for _ in range(rounds):
        mid = np.empty((2 * (len(P) - 1), 3))
        mid[0::2] = 0.75 * P[:-1] + 0.25 * P[1:]
        mid[1::2] = 0.25 * P[:-1] + 0.75 * P[1:]
        P = np.vstack([P[:1], mid, P[-1:]])
    return P


def resample(points, seg=SEG):
    """Points `seg` metres apart (as near as a whole number of segments allows) along the smoothed polyline."""
    P = smooth(points)
    s = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
    n = max(int(round(s[-1] / seg)), 3)
    t = np.linspace(0.0, s[-1], n + 1)
    return np.stack([np.interp(t, s, P[:, k]) for k in range(3)], 1)


class Rope:
    """The cord's particles and one substep of their motion."""

    def __init__(self, X0, radius, floor_z, pin_tail=PIN_TAIL):
        n = len(X0)
        if n < pin_tail + 3:
            raise ValueError(f"a cord of {n} particles is too short to hang")
        self.x, self.prev = X0.copy(), X0.copy()
        self.L = float(np.linalg.norm(np.diff(X0, axis=0), axis=1).mean())
        self.w = np.ones(n)
        self.w[:2] = 0.0
        self.w[n - pin_tail:] = 0.0
        self.tail = X0[n - pin_tail:].copy()
        self.r = np.full(n, float(radius) + MARGIN)
        self.floor = None if floor_z is None else float(floor_z)
        self.free = np.arange(n) * self.L > ANCHOR_FREE
        self.arc_head = np.arange(n) * self.L                      # cord length to the plug ...
        self.arc_tail = (n - pin_tail - np.arange(n)) * self.L     # ... and to the first pinned particle of the tail

    def _tethers(self, x):
        """Long-range attachments: no particle further from a pinned end than its length of cord (the rope's weight
        cannot stretch it; Kim, Chentanez and Mueller 2012)."""
        for end, arc in ((x[0], self.arc_head), (x[len(x) - len(self.tail)], self.arc_tail)):
            d = x - end
            ln = np.linalg.norm(d, axis=1)
            over = (ln > arc) & (self.w > 0) & (ln > 1e-12)
            x[over] = end + d[over] * (arc[over] / ln[over])[:, None]

    def _taut(self, x):
        """The cord does not stretch. What hangs follows the plug: from the plug down to where the cord first lies on
        the floor, a particle more than a segment (and TAUT of one) from the one above it comes back onto it along the
        line between them, without the speed that would add (a cord swung round fast is carried, not thrown: the rounds
        alone left it metres long behind a twirl). What lies on the floor is not touched: the floor's friction holds it
        and the rounds pull its curves straight from the pulled end, as a cord on a floor goes (brought in particle by
        particle it was reeled in whole, like a snake on rails). And the far end gives only when it must: when the plug
        is further from it than the cord is long it is dragged over the floor toward the plug by what is missing, a
        segment a substep at most. Returns whether anything moved."""
        k = len(x) - len(self.tail)
        moved = False
        d = x[0] - x[k]
        short = float(np.linalg.norm(d)) - self.L * k
        if short > 0.0:
            if self.floor is not None:
                d[2] = 0.0
            ln = float(np.linalg.norm(d))
            if ln > 1e-9:
                self.tail = self.tail + d / ln * min(short, self.L)
                x[k:] = self.tail
                moved = True
        hang, drag = self.L * (1.0 + TAUT), self.L * (1.0 + DRAG_AT)
        lying = -math.inf if self.floor is None else self.floor + 0.004
        for i in range(2, len(x)):
            on_floor = x[i, 2] - self.r[i] <= lying             # what lies on the floor is only dragged by a real pull
            d = x[i] - x[i - 1]
            ln = math.sqrt(float(d @ d))
            if ln > (drag if on_floor else hang):
                to = x[i - 1] + d * ((self.L if not on_floor else drag) / ln)
                self.prev[i] += to - x[i]
                x[i] = to
                moved = True
        if moved:                                               # a far end the pull reached lies where it was dragged to
            tail = x[k:]
            if self.floor is not None:
                tail[:, 2] = self.floor + self.r[k:]
            self.tail = tail.copy()
        return moved

    def _lengths(self, x):
        w = self.w
        for par in (0, 1):
            i = np.arange(par, len(x) - 1, 2)
            d = x[i + 1] - x[i]
            ln = np.linalg.norm(d, axis=1)
            s = w[i] + w[i + 1]
            c = np.where((s > 0) & (ln > 1e-12), (ln - self.L) / np.maximum(ln * s, 1e-12), 0.0)
            x[i] += (w[i] * c)[:, None] * d
            x[i + 1] -= (w[i + 1] * c)[:, None] * d

    def _bend(self, x):
        m = 0.5 * (x[:-2] + x[2:]) - x[1:-1]
        dx = np.zeros_like(x)
        dx[1:-1] += (2.0 / 3.0) * m
        dx[:-2] -= m / 3.0
        dx[2:] -= m / 3.0
        x += BEND * self.w[:, None] * dx

    def _floor(self, x):
        if self.floor is None:
            return
        z = self.floor + self.r
        low = (x[:, 2] < z) & (self.w > 0)
        x[low, 2] = z[low]

    def _bodies(self, x, W):
        pen = geom.penetration(x, self.r, W, None)
        depth, push = np.zeros(len(x)), np.zeros_like(x)
        rows = np.arange(len(x))
        for p, nrm in pen.values():
            p = np.where(self.free[:, None], p, -1.0)
            k = np.argmax(p, axis=1)
            pk = p[rows, k]
            deeper = pk > depth
            depth[deeper] = pk[deeper]
            push[deeper] = nrm[rows, k][deeper]
        mv = (depth > 0) & (self.w > 0)
        x[mv] += push[mv] * depth[mv, None]

    def step(self, dt, plug, d, W):
        x = self.x + (self.x - self.prev) * math.exp(-DRAG * dt) + GRAVITY * dt * dt
        self.prev = self.x
        x[0], x[1] = plug, plug + d * self.L
        x[len(x) - len(self.tail):] = self.tail
        for k in range(ITERS):
            self._lengths(x)
            self._tethers(x)
            self._bend(x)
            self._floor(x)
            if W and (k % COLLIDE_EVERY == COLLIDE_EVERY - 1 or k == ITERS - 1):
                self._bodies(x, W)
        if self.floor is not None:                              # Coulomb friction: the floor takes back of a touching
            touch = (x[:, 2] <= self.floor + self.r + 1e-4) & (self.w > 0)     # particle's slide what its weight's grip
            slide = x[touch, :2] - self.prev[touch, :2]         # can in a substep (MU g dt^2), and holds it still under
            s = np.linalg.norm(slide, axis=1)                   # STICK of that
            lim = MU * 9.81 * dt * dt
            keep = np.where(s < STICK * lim, 0.0, np.maximum(s - lim, 0.0) / np.maximum(s, 1e-12))
            x[touch, :2] = self.prev[touch, :2] + slide * keep[:, None]
        if self._taut(x):
            self._floor(x)
            if W:
                self._bodies(x, W)
        self.x = x

    def measure(self, W):
        """(the longest segment's stretch as a share of the rest length: a cord gives where it is squeezed, it bends,
        but it does not stretch; deepest body penetration m; deepest floor penetration m)."""
        x = self.x
        ln = np.linalg.norm(np.diff(x, axis=0), axis=1)
        stretch = float(max(np.max(ln - self.L), 0.0) / self.L)
        body = 0.0
        if W:
            for p, _n in geom.penetration(x, self.r - MARGIN, W, None).values():
                body = max(body, float(np.max(np.where(self.free[:, None], p, 0.0))))
        floor = 0.0 if self.floor is None else float(np.max(self.floor + self.r - MARGIN - x[:, 2]))
        return stretch, body, max(floor, 0.0)


def _unit(v):
    v = np.asarray(v, float)
    return v / max(float(np.linalg.norm(v)), 1e-12)


def simulate(shape, plug_pos, plug_dir, P, src_pos, src_R, floor_z, radius, fps=30.0, substeps=8, settle_s=2.0):
    """(x (F, N, 3), report dict): the cord through every frame. plug_pos, plug_dir (F, 3) are the plug's world position
    and the cord's direction out of it; P geom.Shapes.pack() of the bodies riding on src_pos (F, S, 3) / src_R
    (F, S, 3, 3); floor_z the floor's height or None."""
    t0 = time.perf_counter()
    plug_pos, plug_dir = np.asarray(plug_pos, float), np.asarray(plug_dir, float)
    F = len(plug_pos)
    X0 = resample(shape)
    X0[0] = plug_pos[0]
    rope = Rope(X0, radius, floor_z)
    have = any(P[k]["n"] for k in geom.KINDS)
    world = (lambda f: geom.world(P, src_R[f], src_pos[f])) if have else (lambda f: None)
    dt = 1.0 / (float(fps) * int(substeps))
    W = world(0)
    d0 = _unit(plug_dir[0])
    for _ in range(int(round(float(settle_s) * float(fps) * int(substeps)))):
        rope.step(dt, plug_pos[0], d0, W)
    out = np.zeros((F, len(X0), 3))
    worst = [0.0, 0.0, 0.0]
    for f in range(F):
        a, da = (plug_pos[f - 1], _unit(plug_dir[f - 1])) if f else (plug_pos[0], d0)
        b, db = plug_pos[f], _unit(plug_dir[f])
        W = world(f)
        for s in range(int(substeps)):
            u = (s + 1) / int(substeps)
            rope.step(dt, a + (b - a) * u, _unit(da + (db - da) * u), W)
        out[f] = rope.x
        worst = [max(o, m) for o, m in zip(worst, rope.measure(W))]
    length = float(np.linalg.norm(np.diff(X0, axis=0), axis=1).sum())
    rep = {"version": VERSION, "points": int(len(X0)), "length_m": round(length, 3),
           "segment_mm": round(rope.L * 1000, 1), "stretch_pct_max": round(worst[0] * 100, 2),
           "body_pen_mm_max": round(worst[1] * 1000, 2), "floor_pen_mm_max": round(worst[2] * 1000, 2),
           "seconds": round(time.perf_counter() - t0, 2)}
    return out, rep


# ---------------------------------------------------------------- command line
def main(argv=None):
    """python -m mkmmd.solvers.cable IN.npz OUT.npz (module docstring: the file formats). Returns the exit code."""
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) != 2:
        print("usage: python -m mkmmd.solvers.cable IN.npz OUT.npz", file=sys.stderr)
        return 2
    try:
        with np.load(argv[0], allow_pickle=False) as z:
            a = {k: z[k] for k in z.files}
        spec = json.loads(str(a["spec"]))
        shapes = geom.Shapes()
        for s in spec["sources"]:
            shapes.src(*s)
        n_src = len(shapes.sources)
        geom.model_shapes(shapes, {"bodies": spec.get("bodies", [])}, spec.get("armature", ""))
        if len(shapes.sources) != n_src:
            raise ValueError(f"spec.sources lacks {shapes.sources[n_src:]}")
        src_pos = np.asarray(a["src_pos"], float)
        src_R = geom.quat_to_mat(np.asarray(a["src_quat"], float).reshape(-1, 4)).reshape(src_pos.shape[:2] + (3, 3))
        plug, dsrc = int(spec["plug"]), int(spec["dir_src"])
        plug_dir = np.einsum("fij,j->fi", src_R[:, dsrc], np.asarray(spec["dir"], float))
        floor_z = spec.get("floor_z")
        x, rep = simulate(np.asarray(a["shape"], float), src_pos[:, plug], plug_dir, shapes.pack(), src_pos, src_R,
                          None if floor_z is None else float(floor_z), float(spec["radius"]),
                          float(spec.get("fps", 30.0)), int(spec.get("substeps", 8)), float(spec.get("settle_s", 2.0)))
    except (KeyError, ValueError, OSError, json.JSONDecodeError) as e:
        print(f"cable: bad input: {type(e).__name__}: {e}", file=sys.stderr)
        return 2
    np.savez(argv[1], x=x.astype(np.float32), report=np.array(json.dumps(rep)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
