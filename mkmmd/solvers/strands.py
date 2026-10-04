"""Strands: deterministic secondary motion (hair, ears, tails, skirts, ribbons...) from sampled arrays: numpy, with
optional numba kernels. Never imports bpy.

Bullet running a model's authored rig is tuned for dance: in a calm close-up its frame-to-frame noise is larger than the
hair's real motion, locks jut out when the head tilts, and nothing in it knows about the hands, a pen, the furniture or
the wind. This solver reads the same authored data (rig.json chains and bodies) and simulates it deterministically, so a
resting lock stays still and the result is plain keys. It is the solver of a finished music video, ported to arrays.

Model (position-based strands)
  chains     geom.Chains: one particle per chain bone at the end of its segment (= the head of its simulated child; a
             leaf continues its chain); it hangs from the chain's anchor bone. The caller puts head / end in world
             space.
  colliders  geom.Shapes: the model's other bodies and the scene's shapes riding on their sources. A pair is skipped
             when the particle already overlaps the body at rest, near the root for the body it hangs from, and for the
             PMX collision masks (geom.enable_matrix).
  per substep  verlet prediction with gravity and air drag toward the air velocity -> a pull toward the rest shape
             relative to the parent segment (strength: the sag angle a horizontal segment settles at under gravity) ->
             ITERS rounds of SWEEPS segment-length sweeps (Gauss-Seidel along the strand, both ends move, so tips keep
             their inertia) and collisions of three points per segment against the smooth union of all shapes
             (log-sum-exp soft minimum, SMOOTH: the normal is continuous, so a lock in the crease between two bodies
             settles) -> an exact length projection (follow the leader: root to tip, each particle to its rest distance
             from its parent) -> velocities, damped relative to the rest shape (zeta), capped, and by friction while
             touching. The first frame is settled under gravity.
  inextensible  every substep ends with exact segment lengths, so the particles ARE the forward kinematics of the keyed
             bones (which keep their rest length) and the penetration reported is the penetration that renders. Under
             strong forcing (wind, a fast carrier) sweeps alone leave 3-20 % stretch, and rendered bones then end far
             from the particles. Pure follow-the-leader kills the velocity that would stretch a segment and whips the
             tips; the tension that shortened a child also pulls its parent (dynamic FTL, Mueller et al. 2012: parent
             velocity -= FTL_DAMP * the child's move / dt), which keeps the swing: a strand in a 22 m/s carrier's wind is
             half as jerky (median) and four times calmer at the 95th percentile than with FTL_DAMP = 0.
  output     world rotation deltas D (bone world = D @ rest), keyed bone-locally by `local_quats`.
  engines    numpy: whole chains at once, level by level (the reference; minutes for 160 bones x 900 frames x 10
             substeps). numba (pip install 'mk-mmd[fast]', used when present): one chain at a time in compiled loops,
             chains on threads (seconds). Same dynamics and the same results to rounding (~1e-14 m) until a contact
             amplifies it; chains never interact, so a result does not depend on which other chains are simulated.
  port notes the constants and the order of operations are those of the solver this was ported from, with deliberate
             differences: collision pushes are always applied (it skipped them in the iterations where no particle
             anywhere touched anything: pushes <= 1e-7 m); the numba kernels skip shapes that weigh < e^-25 in the soft
             union (REACH); and, since VERSION 2, strands are inextensible (SWEEPS and the exact projection; the
             original left 0.7 % mean and up to 9 % stretch in a calm seated shot, up to 20 % in wind). Short calm
             hair (ears, bangs) agrees with it to ~0.01 deg; long hair draped on furniture differs by degrees because
             it no longer stretches.
  per-family parameters (`FAMILY_DEFAULTS`, override with `params={family: {...}}`)
    sag         (root_deg, tip_deg)  angle a horizontal segment settles at under gravity, root -> tip (small = keeps its
                                     modelled shape, large = hangs)
    drag        1/s                  air drag: velocity relaxes toward the air velocity (still air: toward 0)
    zeta        damping ratio of the motion relative to the rest shape (1: a jolt dies without ringing)
    radius      x the PMX body radius, capped at radius_max (m): the collision radius of a particle
    friction    1/s                  velocity damping while touching something
    wind_drag   1/s (default drag)   how strongly the air pushes the strand: it reaches wind_drag / drag of the air
                                     speed
Wind (`Wind`)
  The air velocity at a point is: constant wind + carrier_vel * (1 - exposure) + gust + turbulence. `carrier_vel` is the
  world velocity of the vehicle the characters ride in; exposure is the share of its motion the air does NOT follow (0:
  the air rides along, a closed cabin; ~0.3 a convertible; 1: still air, the characters move through it). The air
  enters the dynamics as v <- v exp(-drag dt) + u (1 - exp(-wind_drag dt)) each substep (u: the air at the particle),
  the speed cap MAX_SPEED grows by the air speed relative to the carrier, and friction slows a lock against the
  carrier, so a carrier at constant velocity with exposure 0 changes nothing (Galilean invariance) while exposure > 0
  blows the hair back; the first frame is settled in the carrier's frame and the strands then move on with it. Gusts
  swell the speed along the wind direction; turbulence is a smooth divergence-free field of random Fourier modes of
  size `scale`, carried along with the mean air flow.

Command line (the build pipeline runs it as a subprocess):  python -m mkmmd.solvers.strands IN.npz OUT.npz
  IN.npz arrays   head, end (N,3) chain segment rest head / end in world space, in geom.Chains(rig, families) order
                  bone_rest_R (N,3,3) rest rotation of every chain bone in the same world placement
                  src_pos (F,S,3), src_quat (F,S,4 w x y z, unit, sign-continuous): posed world frames of the S sources
                  rest_R (S,3,3), rest_p (S,3): their rest frames in that placement (identity / zero for objects, world)
                  carrier_vel (F,3) optional world velocity of the vehicle the characters ride in
                  spec        JSON text: {
                    "rig": rig.json dict, "families": [chain families; omit = all chains], "armature": name,
                    "frames": [f0, f1] (F inclusive), "fps": 30, "substeps": 10, "settle_s": 1.5,
                    "sources": [[kind, owner, name], ...] in the order of the src_* / rest_* arrays; it must hold every
                        chain anchor ["bone", armature, anchor bone], model body bone and collider source,
                    "model_bodies": true (the rig's other bodies collide), "colliders": [resolved items of the `sample`
                        op's reply], "params": {family: {...}}, "wind": {...Wind spec...} | null,
                    "use_masks": false, "anchor_free": 0.25, "window": [a, b] frames the report covers (default all),
                    "engine": "auto" | "numpy" | "numba", "threads": null }
  OUT.npz         quats (F,N,4) bone-local rotations w x y z to key (sign-continuous), x (F,N,3) particle positions,
                  pen (F,3) per frame [deepest, median, bones deeper than 2 mm] penetration in metres, report (JSON
                  text: {"bones", "frames", "engine", "seconds", "report"}). Exit codes: 0 ok, 2 bad input, 3 the
                  simulation went non-finite."""
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from . import geom

try:                                                 # pip install mk-mmd[fast]
    from numba import njit
    HAVE_NUMBA = True
except Exception:                                    # not installed, or not built for this numpy
    HAVE_NUMBA = False

VERSION = 2                                          # bump when results change (cache keys)
GRAVITY = np.array([0.0, 0.0, -9.81])
SMOOTH = 0.01                                        # soft-union width (m): creases between shapes fill this much
ITERS = 4                                            # constraint + collision iterations per substep
SWEEPS = 4                                           # length-relaxation sweeps per iteration (cheap next to collisions)
FTL_DAMP = 1.0                                       # share of a length correction handed to the parent as velocity
PUSH_SPEED = 0.25                                    # most velocity (m/s) a collision push may give a particle
MAX_SPEED = 2.5                                      # most speed (m/s) a particle may have relative to its rest shape
PUSH_CAP = 0.03                                      # largest single collision push (m)
LEVER_CAP = 1.5                                      # push multiplier for points nearer the root than the tip, at most
CONTACT_EPS = 1e-7                                   # a push longer than this (m) counts as touching
REACH = 25.0 * SMOOTH                               # a shape whose surface is this much (m) farther than the nearest
                                                    # weighs < e^-25 in the soft union: skipped by the numba kernels

PARAM_KEYS = ("sag", "drag", "zeta", "radius", "radius_max", "friction", "wind_drag")
FAMILY_DEFAULTS = {
    # tuned on a seated close-up: calm back hair (jitter ratio 0.38), lively side locks (0.68), stiff bangs and ears
    "ears":       dict(sag=(0.6, 1.2), drag=5.0, zeta=1.0, radius=0.45, radius_max=0.020, friction=10.0),
    "bangs":      dict(sag=(3.0, 8.0), drag=5.0, zeta=1.0, radius=0.5, radius_max=0.012, friction=10.0),
    "side_hair":  dict(sag=(12.0, 60.0), drag=5.0, zeta=0.7, radius=0.35, radius_max=0.010, friction=20.0),
    "back_hair":  dict(sag=(22.0, 90.0), drag=8.0, zeta=0.8, radius=0.5, radius_max=0.022, friction=25.0),
    # not tuned on a finished shot: physically plausible starting points
    "twintail":   dict(sag=(18.0, 85.0), drag=7.0, zeta=0.75, radius=0.5, radius_max=0.022, friction=22.0),
    "braid":      dict(sag=(15.0, 60.0), drag=8.0, zeta=0.9, radius=0.5, radius_max=0.022, friction=25.0),
    "hair":       dict(sag=(15.0, 70.0), drag=6.0, zeta=0.8, radius=0.5, radius_max=0.020, friction=20.0),
    "tail":       dict(sag=(10.0, 45.0), drag=4.0, zeta=0.6, radius=0.5, radius_max=0.040, friction=15.0),
    "skirt":      dict(sag=(4.0, 14.0), drag=4.0, zeta=0.9, radius=0.5, radius_max=0.030, friction=12.0),
    "ribbon":     dict(sag=(20.0, 80.0), drag=3.5, zeta=0.5, radius=0.5, radius_max=0.010, friction=10.0),
    "breasts":    dict(sag=(1.0, 2.0), drag=8.0, zeta=1.0, radius=0.5, radius_max=0.050, friction=10.0),
    "sleeve":     dict(sag=(6.0, 25.0), drag=5.0, zeta=0.8, radius=0.5, radius_max=0.030, friction=15.0),
    "coat":       dict(sag=(8.0, 30.0), drag=5.0, zeta=0.9, radius=0.5, radius_max=0.040, friction=15.0),
    "accessory":  dict(sag=(15.0, 45.0), drag=5.0, zeta=0.8, radius=0.5, radius_max=0.020, friction=15.0),
    "other":      dict(sag=(10.0, 45.0), drag=5.0, zeta=0.8, radius=0.5, radius_max=0.020, friction=15.0),
}


def resolve_params(families, params=None):
    """{family: complete parameter dict} for the families present: FAMILY_DEFAULTS (unknown family: "other") overridden
    by params[family]. Unknown family or parameter names raise ValueError; wind_drag defaults to drag."""
    params = params or {}
    bad = sorted(set(params) - set(FAMILY_DEFAULTS))
    if bad:
        raise ValueError(f"strands: params for unknown families {bad} (have {sorted(FAMILY_DEFAULTS)})")
    out = {}
    for fam in dict.fromkeys(families):
        p = dict(FAMILY_DEFAULTS.get(fam, FAMILY_DEFAULTS["other"]))
        user = dict(params.get(fam) or {})
        bad = sorted(set(user) - set(PARAM_KEYS))
        if bad:
            raise ValueError(f"strands: unknown parameter(s) {bad} for {fam} (have {list(PARAM_KEYS)})")
        p.update(user)
        sag = np.broadcast_to(np.asarray(p["sag"], float), (2,))
        p["sag"] = (float(sag[0]), float(sag[1]))
        p["wind_drag"] = p["drag"] if p.get("wind_drag") is None else p["wind_drag"]
        for key in PARAM_KEYS[1:]:
            p[key] = float(p[key])
        if min(p["sag"]) < 0 or min(p["drag"], p["zeta"], p["friction"], p["wind_drag"]) < 0 or \
                min(p["radius"], p["radius_max"]) <= 0:
            raise ValueError(f"strands: {fam}: sag, drag, zeta, friction, wind_drag must be >= 0 and radius, "
                             f"radius_max > 0, got {p}")
        out[fam] = p
    return out


# ---------------------------------------------------------------- wind
class Wind:
    """Velocity of the air (world m/s) at a point and time. spec keys (all optional):
      direction [x,y,z]   where the air blows to (any length; default: against the carrier's motion, else +X)
      speed               constant wind speed along it (m/s)
      exposure            0..1, the share of the carrier's motion the air does not follow (default 0: it rides along)
      gust, gust_period   amplitude (m/s) and period (s, default 4) of a slow irregular swell of the speed along it
      turbulence, scale   rms speed (m/s) of a smooth divergence-free fluctuation and its eddy size (m, default 0.5)
      seed                random seed of gusts and turbulence (default 0)
    carrier_vel (F,3): world velocity of the vehicle the characters ride in, one row per frame at `fps`, linear in
    between, held after the last (None: standing still; a single (3,) vector is constant). Time t is seconds from the
    first simulated frame. Air velocity = constant + carrier_vel * (1 - exposure) + gust + turbulence; the turbulence
    pattern drifts with the mean air flow (Taylor's frozen-eddy picture), so hair on a moving carrier meets new eddies.
    """
    KEYS = ("direction", "speed", "exposure", "gust", "gust_period", "turbulence", "scale", "seed")
    MODES = 16
    _GUST_AMP = np.array([0.6, 0.25, 0.15])          # three partials, peak <= 1
    _GUST_PERIOD = np.array([1.0, 0.6, 1.7])

    def __init__(self, spec, carrier_vel=None, fps=30.0):
        spec = {} if spec is None else dict(spec)
        bad = sorted(set(spec) - set(self.KEYS))
        if bad:
            raise ValueError(f"wind: unknown key(s) {bad} (have {list(self.KEYS)})")
        self.fps = float(fps)
        if not self.fps > 0:
            raise ValueError("wind: fps must be positive")
        self.carrier = None
        if carrier_vel is not None:
            c = np.asarray(carrier_vel, float)
            c = c.reshape(1, 3) if c.shape == (3,) else c
            if c.ndim != 2 or c.shape[1] != 3 or not len(c) or not np.isfinite(c).all():
                raise ValueError("wind: carrier_vel must be finite world velocities of shape (F, 3)")
            self.carrier = c
        self.exposure = float(spec.get("exposure", 0.0))
        self.speed = float(spec.get("speed", 0.0))
        self.gust = float(spec.get("gust", 0.0))
        self.gust_period = float(spec.get("gust_period", 4.0))
        self.turbulence = float(spec.get("turbulence", 0.0))
        self.scale = float(spec.get("scale", 0.5))
        self.seed = int(spec.get("seed", 0))
        if not 0.0 <= self.exposure <= 1.0:
            raise ValueError(f"wind: exposure must be within 0..1, got {self.exposure}")
        if self.gust < 0 or self.turbulence < 0 or not self.gust_period > 0 or not self.scale > 0:
            raise ValueError("wind: gust, turbulence must be >= 0 and gust_period, scale > 0")
        d = spec.get("direction")
        if d is None:
            mean_c = self.carrier.mean(0) if self.carrier is not None else np.zeros(3)
            d = -mean_c if np.linalg.norm(mean_c) > 1e-6 else (1.0, 0.0, 0.0)
        d = np.asarray(d, float)
        if d.shape != (3,) or not np.isfinite(d).all() or np.linalg.norm(d) < 1e-9:
            raise ValueError("wind: direction needs three finite numbers, not all zero")
        self.direction = d / np.linalg.norm(d)
        self.constant = self.direction * self.speed
        rng = np.random.default_rng(self.seed)       # the same draws whatever the amplitudes: they only scale
        self._g_phase = 2.0 * np.pi * rng.random(3)
        m = self.MODES
        cz, az = 2.0 * rng.random(m) - 1.0, 2.0 * np.pi * rng.random(m)
        n = np.stack([np.sqrt(1.0 - cz * cz) * np.cos(az), np.sqrt(1.0 - cz * cz) * np.sin(az), cz], 1)
        lam = self.scale * np.exp(np.log(0.5) + np.log(4.0) * rng.random(m))          # eddy sizes scale/2 .. 2 scale
        e = 2.0 * rng.random((m, 3)) - 1.0
        e -= n * np.einsum("mj,mj->m", e, n)[:, None]
        e /= np.maximum(np.linalg.norm(e, axis=1, keepdims=True), 1e-9)
        kv = (2.0 * np.pi / lam)[:, None] * n
        ae = (self.turbulence * np.sqrt(2.0 / m)) * e                              # rms |u| = turbulence
        ph = 2.0 * np.pi * rng.random(m)
        om = np.where(rng.random(m) < 0.5, -1.0, 1.0) * 2.0 * np.pi * self.turbulence / lam * (0.5 + rng.random(m))
        self.modes = (kv, ae, ph, om)
        self._disp = None
        if self.carrier is not None:                  # drift of the eddy pattern: integral of the mean air velocity
            mv = self.constant + self.carrier * (1.0 - self.exposure)
            self._disp = np.zeros_like(mv)
            self._disp[1:] = np.cumsum(0.5 * (mv[1:] + mv[:-1]), axis=0) / self.fps
            self._mv = mv

    def carrier_at(self, ts):
        """(T,3) carrier velocity at times ts (s)."""
        ts = np.atleast_1d(np.asarray(ts, float))
        if self.carrier is None:
            return np.zeros((len(ts), 3))
        last = len(self.carrier) - 1
        s = np.clip(ts * self.fps, 0.0, float(last))
        i0 = np.minimum(np.floor(s).astype(int), last)
        i1 = np.minimum(i0 + 1, last)
        a = (s - i0)[:, None]
        return self.carrier[i0] * (1.0 - a) + self.carrier[i1] * a

    def mean_at(self, ts):
        """(T,3) air velocity at times ts without turbulence: constant + carrier * (1 - exposure) + gust."""
        ts = np.atleast_1d(np.asarray(ts, float))
        out = self.constant[None] + self.carrier_at(ts) * (1.0 - self.exposure)
        if self.gust:
            ph = 2.0 * np.pi * ts[:, None] / (self.gust_period * self._GUST_PERIOD)[None] + self._g_phase
            out = out + (self.gust * (np.cos(ph) @ self._GUST_AMP))[:, None] * self.direction[None]
        return out

    def displacement_at(self, ts):
        """(T,3) how far the mean air flow (constant + carrier part, no gust) has carried the turbulence by ts."""
        ts = np.atleast_1d(np.asarray(ts, float))
        if self._disp is None:
            return self.constant[None] * ts[:, None]
        last = len(self._disp) - 1
        i0 = np.clip(np.floor(ts * self.fps).astype(int), 0, last)
        i1 = np.minimum(i0 + 1, last)
        tau = (ts - i0 / self.fps)[:, None]
        return self._disp[i0] + self._mv[i0] * tau + 0.5 * self.fps * (self._mv[i1] - self._mv[i0]) * tau * tau

    def fluctuation(self, X, disp, t):
        """(M,3) turbulent velocity at world points X for pattern drift `disp` (3,) at time t."""
        if not self.turbulence:
            return np.zeros_like(X)
        kv, ae, ph, om = self.modes
        return np.cos((X - disp) @ kv.T + ph + om * t) @ ae

    def velocity(self, t, X):
        """Air velocity (M,3) at world points X (M,3) at time t seconds."""
        X = np.asarray(X, float)
        single = X.ndim == 1
        X = X.reshape(-1, 3)
        t = float(t)
        u = self.mean_at([t])[0] + self.fluctuation(X, self.displacement_at([t])[0], t)
        return u[0] if single else u


# ---------------------------------------------------------------- what simulate returns
class Result:
    """frames (f0, f1); per frame and chain bone (chains' order): x (F,N,3) particle = segment end, D (F,N,3,3) world
    rotation delta of the bone (bone world = D @ rest), anchor_pos (F,N,3) / anchor_rot (F,N,3,3) the posed anchor
    frame of each bone's chain and anchor_rest_rot (N,3,3) its rest rotation; pen (F,3) [deepest, median, bones deeper
    than 2 mm] penetration (m); worst [(depth m, bone, shape label)] per frame; labels of the shapes (floor last);
    engine ("numba" | "numpy") and timings {setup, settle, run} in seconds."""


# ---------------------------------------------------------------- the model: constants of one simulation
class _Out:
    """Frames recorded by an engine: xs (F,N,3) particles, Ds (F,N,3,3) bone frames, dep (F,N) deepest penetration of
    each bone's sample points (m; < 0: clear) and who (F,N) the id of that shape."""


class _Block:
    """Kinematics of R consecutive substeps: chain-root positions / frames, shape geometry, air."""


def _rows(a, R, n):
    return np.asarray(a).reshape((R, n) + np.shape(a)[1:])


def _world_rows(P, Rs, ps):
    """geom.world for R samples at once: Rs (R,S,3,3), ps (R,S,3) -> {kind: geom.world's tuple, leading R axis}."""
    R, S = ps.shape[:2]
    off = (np.arange(R) * S)[:, None]
    PT = {}
    for kind, d in P.items():
        n = d["n"]
        e = {"n": R * n}
        if n:
            e["src"] = (d["src"][None] + off).reshape(-1)
            for key in ("c", "a", "b", "M", "half", "R", "rnd", "hh"):
                if key in d:
                    e[key] = np.tile(d[key], (R,) + (1,) * (d[key].ndim - 1))
        PT[kind] = e
    W = geom.world(PT, Rs.reshape(R * S, 3, 3), ps.reshape(R * S, 3))
    out = {}
    for kind, w in W.items():
        n = P[kind]["n"]
        out[kind] = tuple(tuple(_rows(a, R, n) for a in x) if isinstance(x, tuple) else _rows(x, R, n) for x in w)
    return out


def _world_row(Wb, r):
    return {kind: tuple(tuple(a[r] for a in x) if isinstance(x, tuple) else x[r] for x in w) for kind, w in Wb.items()}


class _Model:
    def __init__(self, chains, shapes, src_pos, src_quat, rest_R, rest_p, anchor_src, params, frames, rig, fps,
                 substeps, wind, use_masks, anchor_free):
        f0, f1 = int(frames[0]), int(frames[1])
        self.f0, self.F = f0, f1 - f0 + 1
        N, S = len(chains), len(shapes.sources)
        F = self.F
        if N == 0:
            raise ValueError("strands: no chain bones to simulate")
        if F < 1 or substeps < 1 or not fps > 0:
            raise ValueError(f"strands: need f0 <= f1, substeps >= 1 and fps > 0 (got {frames}, {substeps}, {fps})")
        self.src_pos, self.src_quat = np.asarray(src_pos, float), np.asarray(src_quat, float)
        self.rest_R, self.rest_p = np.asarray(rest_R, float), np.asarray(rest_p, float)
        self.a_src = np.asarray(anchor_src, int)
        for name, shp in (("src_pos", (F, S, 3)), ("src_quat", (F, S, 4)), ("rest_R", (S, 3, 3)), ("rest_p", (S, 3)),
                          ("a_src", (N,))):
            a = getattr(self, name)
            if a.shape != shp:
                raise ValueError(f"strands: {name if name != 'a_src' else 'anchor_src'} has shape {a.shape}, expected "
                                 f"{shp} for {F} frames, {S} sources, {N} chain bones")
            if not np.isfinite(a).all():
                raise ValueError(f"strands: {name} has non-finite values")
        if (np.abs(np.linalg.norm(self.src_quat, axis=2) - 1.0) > 1e-3).any():
            raise ValueError("strands: src_quat must be unit quaternions (w x y z)")
        if F > 1 and (np.einsum("fsj,fsj->fs", self.src_quat[1:], self.src_quat[:-1]) < 0).any():
            raise ValueError("strands: src_quat must be sign-continuous (negate q where it flips from the last frame)")
        if wind is not None and not isinstance(wind, Wind):
            raise TypeError("strands: wind must be a strands.Wind or None")
        if wind is not None and abs(wind.fps - fps) > 1e-9 * fps:
            raise ValueError(f"strands: the wind was built for {wind.fps} fps but the simulation runs at {fps}")
        if self.a_src.min() < 0 or self.a_src.max() >= S:
            raise ValueError(f"strands: anchor_src must index the {S} sources")
        self.N, self.S, self.fps, self.substeps, self.wind = N, S, float(fps), int(substeps), wind
        self.dt = dt = 1.0 / (fps * substeps)
        self.bones, self.family = list(chains.bones), list(chains.family)
        # ---- per-bone constants
        par = self.par = np.asarray(chains.parent, int)
        if (par >= np.arange(N)).any():
            raise ValueError("strands: chain bones must come after their parents (geom.Chains order)")
        kids = np.bincount(par[par >= 0], minlength=N)
        if kids.max() > 1:
            raise ValueError(f"strands: branching chains are not supported (bone {self.bones[int(np.argmax(kids))]})")
        level = np.zeros(N, int)
        for i in range(N):
            level[i] = 0 if par[i] < 0 else level[par[i]] + 1
        self.levels = [np.where(level == lv)[0] for lv in range(level.max() + 1)]
        self.child = np.full(N, -1)
        self.child[par[par >= 0]] = np.where(par >= 0)[0]
        self.roots = self.levels[0]
        self.rpos = np.full(N, -1)
        self.rpos[self.roots] = np.arange(len(self.roots))
        self.has_par = par >= 0
        self.head, self.end = np.asarray(chains.head, float), np.asarray(chains.end, float)
        self.seg = seg = self.end - self.head
        self.L = L = np.linalg.norm(seg, axis=1)
        if not (np.isfinite(seg).all() and (L > 1e-9).all()):
            raise ValueError("strands: chain segments must be finite and longer than 1 nm (head != end)")
        self.d0 = seg / L[:, None]
        below = np.zeros(N)
        for i in range(N - 1, -1, -1):
            below[i] = 0 if self.child[i] < 0 else below[self.child[i]] + 1
        s_chain = level / np.maximum(level + below, 1)
        pr = resolve_params(self.family, params)
        fam = self.family

        def get(key):
            return np.array([pr[f][key] for f in fam], float)

        sag = np.radians([pr[f]["sag"][0] + (pr[f]["sag"][1] - pr[f]["sag"][0]) * s for f, s in zip(fam, s_chain)])
        self.rad = np.minimum(np.asarray(chains.radius, float) * get("radius"), get("radius_max"))
        with np.errstate(divide="ignore"):
            self.alpha = np.clip(9.81 * dt * dt / (L * sag), 0.0, 1.0)
        self.decay = np.exp(-get("drag") * dt)
        self.wgain = 1.0 - np.exp(-get("wind_drag") * dt)
        self.stick = np.exp(-get("friction") * dt)          # contact friction: velocity kept per substep while touching
        # structural damping: motion relative to the rest-shape target (i.e. relative to the head) is damped at ratio
        # zeta of the segment's own frequency sqrt(alpha)/dt, so stiff parts (ears, bangs) follow a jolt without ringing
        self.sdecay = np.exp(-2.0 * get("zeta") * np.sqrt(self.alpha))
        self.heavy = np.full(N, np.exp(-30.0 * dt))          # settling damping
        self.sweeps, self.ftl = int(SWEEPS), float(FTL_DAMP)
        # ---- anchors
        self.a_root = self.a_src[self.roots]
        self.root_local = np.einsum("kji,kj->ki", self.rest_R[self.a_root],
                                    self.head[self.roots] - self.rest_p[self.a_root])
        self.rest_root_T = np.transpose(self.rest_R[self.a_root], (0, 2, 1))
        # ---- shapes and collision enables
        self.P = P = shapes.pack()
        self.floor_z = shapes.floor_z
        self.K = sum(P[k]["n"] for k in geom.KINDS)
        self.samples = geom.SAMPLES
        self.own = np.repeat(np.arange(N), len(self.samples))
        self.frac = np.tile(self.samples, N)
        self.rr = self.rad[self.own]
        enable = geom.enable_matrix(chains, P, rig, geom.world(P, self.rest_R, self.rest_p), radius=self.rad,
                                    use_masks=use_masks, anchor_free=anchor_free)
        self.enable = enable
        self.enable_u = np.concatenate([enable[k] for k in geom.KINDS if k in enable], axis=1) if enable else \
            np.zeros((N, 0), bool)
        self.labels = []
        for kind in geom.KINDS:
            for k in range(P[kind]["n"]):
                m = P[kind]["model"][k]
                bone = rig["bodies"][m]["bone"] if m is not None else None
                self.labels.append(P[kind]["tag"][k] if m is None else f"{P[kind]['tag'][k]}:{bone}")
        self.labels.append("floor")
        self.anchor_pos = self.src_pos[:, self.a_src]
        self.anchor_rot = geom.quat_to_mat(self.src_quat[:, self.a_src].reshape(-1, 4)).reshape(F, N, 3, 3)

    def block(self, ks, taus, ts, settle=False):
        """Kinematics at substeps (frame ks[i] + taus[i], clock ts[i] s). settle: pose held, the air taken in the
        carrier's frame (the characters ride along: the carrier is at rest there)."""
        ks, taus, ts = np.asarray(ks, int), np.asarray(taus, float), np.asarray(ts, float)
        R, S = len(ks), self.S
        k1 = np.minimum(ks + 1, self.F - 1)
        w1 = taus[:, None, None]
        q = self.src_quat[ks] * (1 - w1) + self.src_quat[k1] * w1
        q = q / np.linalg.norm(q, axis=2, keepdims=True)
        Rs = geom.quat_to_mat(q.reshape(-1, 4)).reshape(R, S, 3, 3)
        ps = self.src_pos[ks] * (1 - w1) + self.src_pos[k1] * w1
        b = _Block()
        b.R = R
        Ra = Rs[:, self.a_root]
        b.H0 = ps[:, self.a_root] + np.einsum("rkij,kj->rki", Ra, self.root_local)
        b.Da = Ra @ self.rest_root_T[None]
        b.W = _world_rows(self.P, Rs, ps)
        b.wind = self.wind
        if self.wind is not None:
            b.wmean, b.wdisp, b.wt, b.wcar = (self.wind.mean_at(ts), self.wind.displacement_at(ts), ts,
                                              self.wind.carrier_at(ts))
            if settle:
                b.wmean, b.wcar = b.wmean - b.wcar, np.zeros_like(b.wcar)
        return b

    def initial_state(self):
        """The rest shape carried rigidly by each chain's anchor on the first frame: bone frames D and particles x."""
        b = self.block([0], [0.0], [0.0])
        H0, Da = b.H0[0], b.Da[0]
        D, x = np.zeros((self.N, 3, 3)), np.zeros((self.N, 3))
        for lv, I in enumerate(self.levels):
            D[I] = Da[self.rpos[I]] if lv == 0 else D[self.par[I]]
            x[I] = (H0[self.rpos[I]] if lv == 0 else x[self.par[I]]) + np.einsum("kij,kj->ki", D[I], self.seg[I])
        return D, x


# ---------------------------------------------------------------- numpy engine: whole chains at once, level by level
class _NumpyEngine:
    name = "numpy"

    def __init__(self, M, out):
        self.M, self.out = M, out
        self.xs, self.Ds, self.order = out.xs, out.Ds, np.arange(M.N)            # recorded frames, in this bone order
        self.D, self.x = M.initial_state()
        self.v = np.zeros((M.N, 3))
        self.tgt, self.tgt_prev = self.x.copy(), self.x.copy()

    def shape_pass(self, p, H0, Da, pull):
        """Root to tip: each segment's rest-shape target (its rest direction carried by the parent's current frame),
        optionally pulled toward (bending stiffness), and the bone's frame D from its current direction."""
        M, D, tgt = self.M, self.D, self.tgt
        for lv, I in enumerate(M.levels):
            head, Dp = (H0[M.rpos[I]], Da[M.rpos[I]]) if lv == 0 else (p[M.par[I]], D[M.par[I]])
            tgt[I] = head + M.L[I, None] * np.einsum("kij,kj->ki", Dp, M.d0[I])
            if pull:
                p[I] += M.alpha[I, None] * (tgt[I] - p[I])
            dv = p[I] - head
            u = dv / np.maximum(np.linalg.norm(dv, axis=1), 1e-12)[:, None]
            D[I] = Dp @ geom.min_rot(M.d0[I], np.einsum("kji,kj->ki", Dp, u))

    def relax(self, p, H0):
        """Segment lengths (position-based distance constraints, Gauss-Seidel root to tip; both ends move unless one is
        a root): corrections travel along the strand over the iterations, so a tip keeps its inertia instead of being
        yanked into place the moment the root moves."""
        M = self.M
        for lv, I in enumerate(M.levels):
            head = H0[M.rpos[I]] if lv == 0 else p[M.par[I]]
            dv = p[I] - head
            n = np.maximum(np.linalg.norm(dv, axis=1), 1e-12)
            c = dv * ((n - M.L[I]) / n)[:, None]
            if lv == 0:
                p[I] -= c
            else:
                p[I] -= 0.5 * c
                p[M.par[I]] += 0.5 * c

    def project(self, p, H0):
        """Exact segment lengths, follow the leader (root to tip, only the child moves): each particle goes to the
        right distance from its parent along the line to where it was. Returns the moves (N, 3)."""
        M = self.M
        corr = np.empty_like(p)
        for lv, I in enumerate(M.levels):
            head = H0[M.rpos[I]] if lv == 0 else p[M.par[I]]
            dv = p[I] - head
            new = head + dv * (M.L[I] / np.maximum(np.linalg.norm(dv, axis=1), 1e-12))[:, None]
            corr[I] = new - p[I]
            p[I] = new
        return corr

    def points(self, p, H0):
        M = self.M
        H = np.empty_like(p)
        H[M.roots] = H0
        H[M.has_par] = p[M.par[M.has_par]]
        return H[M.own] + M.frac[:, None] * (p[M.own] - H[M.own])

    def collide(self, p, H0, W):
        """Points along every segment against the union of all enabled shapes. Pushes use a smooth union (log-sum-exp
        soft minimum of the signed distances, k = SMOOTH): its normal is a continuous blend, so a lock lying in the
        crease between overlapping body capsules settles instead of being bounced from one capsule into the other."""
        M, N = self.M, self.M.N
        X = self.points(p, H0)
        Ss, Ns = [], []
        for kind, (pen, n) in geom.penetration(X, M.rr, W, M.floor_z).items():
            if kind in M.enable:
                pen = np.where(M.enable[kind][M.own], pen, -1.0)
            Ss.append(-pen)
            Ns.append(n)
        if not Ss:
            return np.zeros_like(p), np.zeros(N, bool)
        S, Nn = np.concatenate(Ss, 1), np.concatenate(Ns, 1)
        smin = S.min(1)
        e = np.exp(-(S - smin[:, None]) / SMOOTH)
        soft = smin - SMOOTH * np.log(e.sum(1))
        nrm = np.einsum("mk,mkj->mj", e, Nn)
        nrm /= np.maximum(np.linalg.norm(nrm, axis=1), 1e-12)[:, None]
        best = np.maximum(-soft, 0.0)
        lever = np.minimum(1.0 / M.frac, LEVER_CAP)                                 # points near the root: capped lever
        push = (nrm * (best * lever)[:, None]).reshape(N, len(M.samples), 3)
        mag = np.linalg.norm(push, axis=2)
        j = np.argmax(mag, axis=1)
        m = mag[np.arange(N), j]
        return push[np.arange(N), j] * (np.minimum(m, PUSH_CAP) / np.maximum(m, 1e-12))[:, None], m > CONTACT_EPS

    def substep(self, blk, r, decay):
        M, dt = self.M, self.M.dt
        H0, Da, W = blk.H0[r], blk.Da[r], _world_row(blk.W, r)
        if blk.wind is not None:
            u = blk.wmean[r] + blk.wind.fluctuation(self.x, blk.wdisp[r], blk.wt[r])
            v = self.v * decay[:, None] + u * M.wgain[:, None]
        else:
            v = self.v * decay[:, None]
        p = self.x + v * dt + GRAVITY * dt * dt
        self.shape_pass(p, H0, Da, True)
        contact = np.zeros(M.N, bool)
        pushed = np.zeros_like(p)
        for _ in range(ITERS):                          # lengths, then collisions
            for _ in range(M.sweeps):
                self.relax(p, H0)
            push, c = self.collide(p, H0, W)
            contact |= c
            p += push
            pushed += push
        corr = self.project(p, H0)                      # exact lengths: the bones keyed from these frames render this
        self.shape_pass(p, H0, Da, False)               # final frames and targets from the settled positions
        vn = (p - self.x) / dt
        if M.ftl:                                       # dynamic FTL: a child's move pulls its parent
            has = M.child >= 0
            vn[has] -= M.ftl * corr[M.child[has]] / dt
        vp = pushed / dt                                # depenetration moves a lock but may not launch it: a body or
        m = np.linalg.norm(vp, axis=1)                  # prop can push hair along at up to PUSH_SPEED, a squeeze
        vn -= vp * (1.0 - np.minimum(1.0, PUSH_SPEED / np.maximum(m, 1e-12)))[:, None]   # cannot fling it
        vt = (self.tgt - self.tgt_prev) / dt
        vn = vt + (vn - vt) * M.sdecay[:, None]
        rel = np.linalg.norm(vn - vt, axis=1)           # safety net for impossible configurations: never faster than
        cap = MAX_SPEED if blk.wind is None else MAX_SPEED + np.linalg.norm(u - blk.wcar[r], axis=1)
        vn = vt + (vn - vt) * np.minimum(1.0, cap / np.maximum(rel, 1e-12))[:, None]   # MAX_SPEED vs the head
        self.tgt_prev = self.tgt.copy()
        if blk.wind is None:
            vn[contact] *= M.stick[contact, None]
        else:                                           # friction slows a lock against the carrier, not the ground
            vn[contact] = blk.wcar[r] + (vn[contact] - blk.wcar[r]) * M.stick[contact, None]
        self.v, self.x = vn, p

    def settle(self, blk, steps):
        for _ in range(steps):
            self.substep(blk, 0, self.M.heavy)

    def start(self, v0):
        """Leave the settle (done in the carrier's frame): the strands move on with the carrier's velocity v0."""
        self.v += v0

    def record(self, f, blk, r):
        M, out = self.M, self.out
        X = self.points(self.x, blk.H0[r])
        pen, sid = geom.measure(X, M.rr, _world_row(blk.W, r), M.floor_z, M.P, M.enable, M.own)
        per, who = pen.reshape(M.N, -1), sid.reshape(M.N, -1)
        c = np.argmax(per, 1)
        out.dep[f], out.who[f] = per[np.arange(M.N), c], who[np.arange(M.N), c]
        out.xs[f], out.Ds[f] = self.x, self.D

    def run(self, blk, f_first):
        """The substeps of blk (M.substeps per frame); the last of each frame is recorded as f_first, f_first + 1..."""
        M = self.M
        for r in range(blk.R):
            self.substep(blk, r, M.decay)
            if (r + 1) % M.substeps == 0:
                self.record(f_first + r // M.substeps, blk, r)

    def finish(self):
        pass

    def close(self):
        pass


# ---------------------------------------------------------------- numba kernels: one chain at a time, all substeps
# Chains never touch each other (only the read-only shapes), so a kernel advances any set of chains independently and
# threads split them. Per pair the maths is geom.penetration's; shapes too far to matter (REACH) are skipped.
if HAVE_NUMBA:
    _jit = njit(cache=True, nogil=True, error_model="numpy")

    @_jit
    def _nb_dist(kind, k, x, y, z, r, sc, sb, sm, sk):
        """(S, nx, ny, nz): signed distance S = -penetration of a particle (centre x y z, radius r) to shape k of `kind`
        (0 sphere 1 capsule 2 box 3 cylinder) and the world direction to push it out (geom.penetration, one pair)."""
        if kind == 0:
            dx, dy, dz = x - sc[k, 0], y - sc[k, 1], z - sc[k, 2]
            dist = np.sqrt(dx * dx + dy * dy + dz * dz)
            den = max(dist, 1e-9)
            return -(sk[k, 0] + r - dist), dx / den, dy / den, dz / den
        if kind == 1:
            ax, ay, az = sc[k, 0], sc[k, 1], sc[k, 2]
            abx, aby, abz = sb[k, 0] - ax, sb[k, 1] - ay, sb[k, 2] - az
            t = ((x - ax) * abx + (y - ay) * aby + (z - az) * abz) / max(abx * abx + aby * aby + abz * abz, 1e-12)
            t = min(max(t, 0.0), 1.0)
            dx, dy, dz = x - (ax + t * abx), y - (ay + t * aby), z - (az + t * abz)
            dist = np.sqrt(dx * dx + dy * dy + dz * dz)
            den = max(dist, 1e-9)
            return -(sk[k, 0] + r - dist), dx / den, dy / den, dz / den
        dx, dy, dz = x - sc[k, 0], y - sc[k, 1], z - sc[k, 2]
        l0 = sm[k, 0, 0] * dx + sm[k, 1, 0] * dy + sm[k, 2, 0] * dz              # the point in the shape's axes
        l1 = sm[k, 0, 1] * dx + sm[k, 1, 1] * dy + sm[k, 2, 1] * dz
        l2 = sm[k, 0, 2] * dx + sm[k, 1, 2] * dy + sm[k, 2, 2] * dz
        rnd = sk[k, 4]
        clear = r + rnd
        if kind == 2:                                                              # box with rounded edges
            h0, h1, h2 = max(sk[k, 1] - rnd, 1e-6), max(sk[k, 2] - rnd, 1e-6), max(sk[k, 3] - rnd, 1e-6)
            e0, e1, e2 = l0 - min(max(l0, -h0), h0), l1 - min(max(l1, -h1), h1), l2 - min(max(l2, -h2), h2)
            dist = np.sqrt(e0 * e0 + e1 * e1 + e2 * e2)
            if dist > 1e-9:
                den = max(dist, 1e-9)
                pen = clear - dist
                n0, n1, n2 = e0 / den, e1 / den, e2 / den
            else:                                                                  # inside: leave by the nearest face
                g0, g1, g2 = h0 - abs(l0), h1 - abs(l1), h2 - abs(l2)
                ax, gm, lv = 0, g0, l0
                if g1 < gm:
                    ax, gm, lv = 1, g1, l1
                if g2 < gm:
                    ax, gm, lv = 2, g2, l2
                sg = -1.0 if lv < 0 else 1.0
                pen = gm + clear
                n0, n1, n2 = (sg if ax == 0 else 0.0), (sg if ax == 1 else 0.0), (sg if ax == 2 else 0.0)
        else:                                                                      # cylinder (axis z) with rounded rims
            rc, hh = max(sk[k, 0] - rnd, 1e-6), max(sk[k, 1] - rnd, 1e-6)
            rad = np.sqrt(l0 * l0 + l1 * l1)
            den = max(rad, 1e-9)
            rx, ry = l0 / den, l1 / den
            qr = min(rad, rc)
            e0, e1, e2 = l0 - rx * qr, l1 - ry * qr, l2 - min(max(l2, -hh), hh)
            dist = np.sqrt(e0 * e0 + e1 * e1 + e2 * e2)
            if dist > 1e-9:
                den = max(dist, 1e-9)
                pen = clear - dist
                n0, n1, n2 = e0 / den, e1 / den, e2 / den
            else:
                side_gap, top_gap = rc - rad, hh - abs(l2)
                pen = min(side_gap, top_gap) + clear
                if side_gap < top_gap:
                    n0, n1, n2 = rx, ry, 0.0
                else:
                    n0, n1, n2 = 0.0, 0.0, (1.0 if l2 >= 0 else -1.0)
        return (-pen, sm[k, 0, 0] * n0 + sm[k, 0, 1] * n1 + sm[k, 0, 2] * n2,
                sm[k, 1, 0] * n0 + sm[k, 1, 1] * n1 + sm[k, 1, 2] * n2,
                sm[k, 2, 0] * n0 + sm[k, 2, 1] * n1 + sm[k, 2, 2] * n2)

    @_jit
    def _nb_gather(hx, hy, hz, px, py, pz, r, en, bsc, bsr, cand, cut, tnear):
        """Enabled shapes whose bounding sphere comes within `cut` of the segment head -> p (grown by the radius r) go
        to cand; returns how many, and how many of those come within `tnear`."""
        mx, my, mz = 0.5 * (hx + px), 0.5 * (hy + py), 0.5 * (hz + pz)
        hl = 0.5 * np.sqrt((px - hx) ** 2 + (py - hy) ** 2 + (pz - hz) ** 2)
        n, near = 0, 0
        for k in range(en.shape[0]):
            if en[k]:
                dx, dy, dz = mx - bsc[k, 0], my - bsc[k, 1], mz - bsc[k, 2]
                lb = np.sqrt(dx * dx + dy * dy + dz * dz) - bsr[k] - hl - r
                if lb <= cut:
                    cand[n] = k
                    n += 1
                    if lb <= tnear:
                        near += 1
        return n, near

    @_jit
    def _nb_eval(x, y, z, r, cand, ncand, sc, sb, sm, skind, sk, floor_z, has_floor, Sb, Nb):
        """Signed distances and normals of one point to the candidates (+ the floor) into Sb Nb; returns how many, the
        smallest distance and the id of its shape (the floor is id K)."""
        n, smin, imin = 0, 1e300, -1
        for ci in range(ncand):
            k = cand[ci]
            s, nx, ny, nz = _nb_dist(skind[k], k, x, y, z, r, sc, sb, sm, sk)
            Sb[n], Nb[n, 0], Nb[n, 1], Nb[n, 2] = s, nx, ny, nz
            n += 1
            if s < smin:
                smin, imin = s, k
        if has_floor:
            s = -(floor_z + r - z)
            Sb[n], Nb[n, 0], Nb[n, 1], Nb[n, 2] = s, 0.0, 0.0, 1.0
            n += 1
            if s < smin:
                smin, imin = s, skind.shape[0]
        return n, smin, imin

    @_jit
    def _nb_push(hx, hy, hz, px, py, pz, r, cand, ncand, sc, sb, sm, skind, sk, floor_z, has_floor, samp, Sb, Nb):
        """The collision push of one segment (head -> p): of its sample points the one pushed hardest, by the smooth
        union of the shapes (the numpy engine's `collide` for one bone). Returns (dx, dy, dz, touching)."""
        best, bx, by, bz = -1.0, 0.0, 0.0, 0.0
        for s in range(samp.shape[0]):
            f = samp[s]
            x, y, z = hx + f * (px - hx), hy + f * (py - hy), hz + f * (pz - hz)
            n, smin, imin = _nb_eval(x, y, z, r, cand, ncand, sc, sb, sm, skind, sk, floor_z, has_floor, Sb, Nb)
            vx, vy, vz = 0.0, 0.0, 0.0
            if n > 0:
                se, nx, ny, nz = 0.0, 0.0, 0.0, 0.0
                for i in range(n):
                    d = Sb[i] - smin
                    if d < 36.0 * SMOOTH:                       # e^-36 vanishes next to the nearest shape's weight 1
                        e = np.exp(-d / SMOOTH)
                        se += e
                        nx, ny, nz = nx + e * Nb[i, 0], ny + e * Nb[i, 1], nz + e * Nb[i, 2]
                soft = smin - SMOOTH * np.log(se)
                nn = max(np.sqrt(nx * nx + ny * ny + nz * nz), 1e-12)
                lev = max(-soft, 0.0) * min(1.0 / f, LEVER_CAP)
                vx, vy, vz = nx / nn * lev, ny / nn * lev, nz / nn * lev
            mag = np.sqrt(vx * vx + vy * vy + vz * vz)
            if mag > best:
                best, bx, by, bz = mag, vx, vy, vz
        sc_ = min(best, PUSH_CAP) / max(best, 1e-12)
        return bx * sc_, by * sc_, bz * sc_, best > CONTACT_EPS

    @_jit
    def _nb_measure(hx, hy, hz, px, py, pz, r, cand, ncand, sc, sb, sm, skind, sk, floor_z, has_floor, samp, Sb, Nb):
        """Deepest penetration (m, > 0 inside; -1e300 if no shape is near) of a segment's points and its shape id."""
        best, who = -1e300, -1
        for s in range(samp.shape[0]):
            f = samp[s]
            x, y, z = hx + f * (px - hx), hy + f * (py - hy), hz + f * (pz - hz)
            n, smin, imin = _nb_eval(x, y, z, r, cand, ncand, sc, sb, sm, skind, sk, floor_z, has_floor, Sb, Nb)
            if n > 0 and -smin > best:
                best, who = -smin, imin
        return best, who

    @_jit
    def _nb_min_rot_mul(Dp, a0, a1, a2, b0, b1, b2, out):
        """out = Dp @ R where R is geom.min_rot's rotation taking unit vector a to unit vector b."""
        vx, vy, vz = a1 * b2 - a2 * b1, a2 * b0 - a0 * b2, a0 * b1 - a1 * b0
        c = a0 * b0 + a1 * b1 + a2 * b2
        R = np.empty((3, 3))
        if c < -0.99999:                                  # opposite: half turn about any axis perpendicular to a
            ex, ey = (1.0, 0.0) if abs(a0) < 0.9 else (0.0, 1.0)
            kx, ky, kz = a1 * 0.0 - a2 * ey, a2 * ex - a0 * 0.0, a0 * ey - a1 * ex
            kn = np.sqrt(kx * kx + ky * ky + kz * kz)
            kx, ky, kz = kx / kn, ky / kn, kz / kn
            ax = (kx, ky, kz)
            for i in range(3):
                for j in range(3):
                    R[i, j] = 2.0 * ax[i] * ax[j] - (1.0 if i == j else 0.0)
        else:
            K = np.zeros((3, 3))
            K[0, 1], K[0, 2], K[1, 0], K[1, 2], K[2, 0], K[2, 1] = -vz, vy, vz, -vx, -vy, vx
            s = 1.0 / max(1.0 + c, 1e-9)
            for i in range(3):
                for j in range(3):
                    R[i, j] = ((1.0 if i == j else 0.0) + K[i, j]) + (K[i, 0] * K[0, j] + K[i, 1] * K[1, j] +
                                                                       K[i, 2] * K[2, j]) * s
        for i in range(3):
            for j in range(3):
                out[i, j] = Dp[i, 0] * R[0, j] + Dp[i, 1] * R[1, j] + Dp[i, 2] * R[2, j]

    @_jit
    def _nb_shape_pass(n, H0r, Dar, p, D, tgt, Lc, d0, alpha, pull):
        """Root to tip: each segment's rest-shape target, optionally pulled toward, and the bone frame D (the numpy
        engine's shape_pass for one chain)."""
        for j in range(n):
            if j == 0:
                hx, hy, hz = H0r[0], H0r[1], H0r[2]
                Dp = Dar
            else:
                hx, hy, hz = p[j - 1, 0], p[j - 1, 1], p[j - 1, 2]
                Dp = D[j - 1]
            a0, a1, a2 = d0[j, 0], d0[j, 1], d0[j, 2]
            tx = hx + Lc[j] * (Dp[0, 0] * a0 + Dp[0, 1] * a1 + Dp[0, 2] * a2)
            ty = hy + Lc[j] * (Dp[1, 0] * a0 + Dp[1, 1] * a1 + Dp[1, 2] * a2)
            tz = hz + Lc[j] * (Dp[2, 0] * a0 + Dp[2, 1] * a1 + Dp[2, 2] * a2)
            tgt[j, 0], tgt[j, 1], tgt[j, 2] = tx, ty, tz
            if pull:
                p[j, 0] += alpha[j] * (tx - p[j, 0])
                p[j, 1] += alpha[j] * (ty - p[j, 1])
                p[j, 2] += alpha[j] * (tz - p[j, 2])
            dx, dy, dz = p[j, 0] - hx, p[j, 1] - hy, p[j, 2] - hz
            nn = max(np.sqrt(dx * dx + dy * dy + dz * dz), 1e-12)
            u0, u1, u2 = dx / nn, dy / nn, dz / nn
            _nb_min_rot_mul(Dp, a0, a1, a2, Dp[0, 0] * u0 + Dp[1, 0] * u1 + Dp[2, 0] * u2,
                            Dp[0, 1] * u0 + Dp[1, 1] * u1 + Dp[2, 1] * u2,
                            Dp[0, 2] * u0 + Dp[1, 2] * u1 + Dp[2, 2] * u2,
                            D[j])

    @_jit
    def _nb_relax(n, H0r, p, Lc):
        """Segment lengths, Gauss-Seidel root to tip (the numpy engine's relax for one chain)."""
        for j in range(n):
            if j == 0:
                hx, hy, hz = H0r[0], H0r[1], H0r[2]
            else:
                hx, hy, hz = p[j - 1, 0], p[j - 1, 1], p[j - 1, 2]
            dx, dy, dz = p[j, 0] - hx, p[j, 1] - hy, p[j, 2] - hz
            nn = max(np.sqrt(dx * dx + dy * dy + dz * dz), 1e-12)
            f = (nn - Lc[j]) / nn
            cx, cy, cz = dx * f, dy * f, dz * f
            if j == 0:
                p[j, 0] -= cx
                p[j, 1] -= cy
                p[j, 2] -= cz
            else:
                p[j, 0] -= 0.5 * cx
                p[j, 1] -= 0.5 * cy
                p[j, 2] -= 0.5 * cz
                p[j - 1, 0] += 0.5 * cx
                p[j - 1, 1] += 0.5 * cy
                p[j - 1, 2] += 0.5 * cz

    @_jit
    def _nb_project(n, H0r, p, Lc, corr):
        """Exact segment lengths, follow the leader: root to tip each particle goes to the right distance from its
        parent along the line to where it was; corr gets the moves (the numpy engine's project for one chain)."""
        for j in range(n):
            if j == 0:
                hx, hy, hz = H0r[0], H0r[1], H0r[2]
            else:
                hx, hy, hz = p[j - 1, 0], p[j - 1, 1], p[j - 1, 2]
            dx, dy, dz = p[j, 0] - hx, p[j, 1] - hy, p[j, 2] - hz
            f = Lc[j] / max(np.sqrt(dx * dx + dy * dy + dz * dz), 1e-12)
            nx, ny, nz = hx + dx * f, hy + dy * f, hz + dz * f
            corr[j, 0], corr[j, 1], corr[j, 2] = nx - p[j, 0], ny - p[j, 1], nz - p[j, 2]
            p[j, 0], p[j, 1], p[j, 2] = nx, ny, nz

    @_jit
    def _nb_air(x, y, z, r, wmean, wdisp, wt, kv, ae, ph, om, use_turb):
        """Air velocity at a point on substep row r (Wind.mean_at + Wind.fluctuation)."""
        ux, uy, uz = wmean[r, 0], wmean[r, 1], wmean[r, 2]
        if use_turb:
            for m in range(kv.shape[0]):
                a = (kv[m, 0] * (x - wdisp[r, 0]) + kv[m, 1] * (y - wdisp[r, 1]) + kv[m, 2] * (z - wdisp[r, 2]) +
                     ph[m] + om[m] * wt[r])
                cs = np.cos(a)
                ux, uy, uz = ux + ae[m, 0] * cs, uy + ae[m, 1] * cs, uz + ae[m, 2] * cs
        return ux, uy, uz

    @_jit
    def _nb_record(f, j0, n, H0r, x, Dst, rr, enable, sc, sb, sm, bsc, bsr, skind, sk, floor_z, has_floor, samp,
                   cand, Sb, Nb, xs, Ds, dep, who):
        """Store the state of a chain as frame f: particles, bone frames, deepest penetration and its shape."""
        for jj in range(n):
            j = j0 + jj
            if jj == 0:
                hx, hy, hz = H0r[0], H0r[1], H0r[2]
            else:
                hx, hy, hz = x[j - 1, 0], x[j - 1, 1], x[j - 1, 2]
            nc, _ = _nb_gather(hx, hy, hz, x[j, 0], x[j, 1], x[j, 2], rr[j], enable[j], bsc, bsr, cand, 0.0, 0.0)
            dep[f, j], who[f, j] = _nb_measure(hx, hy, hz, x[j, 0], x[j, 1], x[j, 2], rr[j], cand, nc, sc, sb, sm,
                                               skind, sk, floor_z, has_floor, samp, Sb, Nb)
            for i in range(3):
                xs[f, j, i] = x[j, i]
                for k in range(3):
                    Ds[f, j, i, k] = Dst[j, i, k]

    @_jit
    def _nb_advance(cids, cptr, nsteps, rstride, rec_every, rec_f0, rec_start,
                    Lc, d0, alpha, dec, wgain, stick, sdec, rr, enable,
                    x, v, tgt_prev, Dst, tgt,
                    H0, Da, sc, sb, sm, bsc, bsr, wmean, wdisp, wt, wcar,
                    skind, sk, floor_z, has_floor,
                    kv, ae, ph, om, use_wind, use_turb,
                    dt, samp, sweeps, ftl, xs, Ds, dep, who):
        """Advance chains cids by nsteps substeps (substep i uses block row i * rstride), with velocity decay `dec`;
        every rec_every steps store a frame (first rec_f0), and rec_start >= 0 stores the current state first as that
        frame. State arrays x v tgt_prev Dst tgt are per bone in chain-major order and updated in place."""
        K = skind.shape[0]
        cand = np.empty(K + 1, np.int64)
        Sb, Nb = np.empty(K + 1), np.empty((K + 1, 3))
        gdt2 = (-9.81 * dt) * dt
        tpush = SMOOTH * np.log(K + 1.0) + 1e-6                 # nearer than this (m) to a shape, a push is possible
        cut = tpush + REACH
        for ci in range(cids.shape[0]):
            c = cids[ci]
            j0, j1 = cptr[c], cptr[c + 1]
            n = j1 - j0
            p, pushed, push, ub = np.empty((n, 3)), np.empty((n, 3)), np.empty((n, 3)), np.zeros((n, 3))
            hit, cont = np.zeros(n, np.bool_), np.zeros(n, np.bool_)
            corr = np.empty((n, 3))
            Lg, dg, ag, Dg, tg = Lc[j0:j1], d0[j0:j1], alpha[j0:j1], Dst[j0:j1], tgt[j0:j1]
            if rec_start >= 0:
                _nb_record(rec_start, j0, n, H0[0, c], x, Dst, rr, enable, sc[0], sb[0], sm[0], bsc[0], bsr[0], skind,
                           sk, floor_z, has_floor, samp, cand, Sb, Nb, xs, Ds, dep, who)
            for step in range(nsteps):
                r = step * rstride
                H0r, Dar = H0[r, c], Da[r, c]
                for jj in range(n):                             # drag toward the air, then the verlet prediction
                    j = j0 + jj
                    if use_wind:
                        ux, uy, uz = _nb_air(x[j, 0], x[j, 1], x[j, 2], r, wmean, wdisp, wt, kv, ae, ph, om, use_turb)
                        ub[jj, 0], ub[jj, 1], ub[jj, 2] = ux, uy, uz
                        v[j, 0] = v[j, 0] * dec[j] + ux * wgain[j]
                        v[j, 1] = v[j, 1] * dec[j] + uy * wgain[j]
                        v[j, 2] = v[j, 2] * dec[j] + uz * wgain[j]
                    else:
                        v[j, 0] *= dec[j]
                        v[j, 1] *= dec[j]
                        v[j, 2] *= dec[j]
                    p[jj, 0] = x[j, 0] + v[j, 0] * dt
                    p[jj, 1] = x[j, 1] + v[j, 1] * dt
                    p[jj, 2] = (x[j, 2] + v[j, 2] * dt) + gdt2
                _nb_shape_pass(n, H0r, Dar, p, Dg, tg, Lg, dg, ag, True)
                for jj in range(n):
                    cont[jj] = False
                    pushed[jj, 0], pushed[jj, 1], pushed[jj, 2] = 0.0, 0.0, 0.0
                for it in range(ITERS):                         # lengths, then collisions
                    for sw in range(sweeps):
                        _nb_relax(n, H0r, p, Lg)
                    for jj in range(n):
                        j = j0 + jj
                        if jj == 0:
                            hx, hy, hz = H0r[0], H0r[1], H0r[2]
                        else:
                            hx, hy, hz = p[jj - 1, 0], p[jj - 1, 1], p[jj - 1, 2]
                        nc, near = _nb_gather(hx, hy, hz, p[jj, 0], p[jj, 1], p[jj, 2], rr[j], enable[j], bsc[r],
                                              bsr[r], cand, cut, tpush)
                        if has_floor and min(hz, p[jj, 2]) - floor_z - rr[j] <= tpush:
                            near += 1
                        if near > 0:
                            ux, uy, uz, hit[jj] = _nb_push(hx, hy, hz, p[jj, 0], p[jj, 1], p[jj, 2], rr[j], cand, nc,
                                                           sc[r], sb[r], sm[r], skind, sk, floor_z, has_floor, samp,
                                                           Sb, Nb)
                        else:
                            ux, uy, uz, hit[jj] = 0.0, 0.0, 0.0, False
                        push[jj, 0], push[jj, 1], push[jj, 2] = ux, uy, uz
                    for jj in range(n):
                        if hit[jj]:
                            cont[jj] = True
                        for i in range(3):
                            p[jj, i] += push[jj, i]
                            pushed[jj, i] += push[jj, i]
                _nb_project(n, H0r, p, Lg, corr)
                _nb_shape_pass(n, H0r, Dar, p, Dg, tg, Lg, dg, ag, False)
                for jj in range(n):                             # velocities
                    j = j0 + jj
                    vnx, vny, vnz = (p[jj, 0] - x[j, 0]) / dt, (p[jj, 1] - x[j, 1]) / dt, (p[jj, 2] - x[j, 2]) / dt
                    if jj + 1 < n:                              # dynamic FTL: the child's move pulls its parent
                        vnx, vny, vnz = (vnx - ftl * corr[jj + 1, 0] / dt, vny - ftl * corr[jj + 1, 1] / dt,
                                         vnz - ftl * corr[jj + 1, 2] / dt)
                    vpx, vpy, vpz = pushed[jj, 0] / dt, pushed[jj, 1] / dt, pushed[jj, 2] / dt
                    fp = 1.0 - min(1.0, PUSH_SPEED / max(np.sqrt(vpx * vpx + vpy * vpy + vpz * vpz), 1e-12))
                    vnx, vny, vnz = vnx - vpx * fp, vny - vpy * fp, vnz - vpz * fp
                    vtx, vty, vtz = (tg[jj, 0] - tgt_prev[j, 0]) / dt, (tg[jj, 1] - tgt_prev[j, 1]) / dt, \
                        (tg[jj, 2] - tgt_prev[j, 2]) / dt
                    vnx, vny, vnz = (vtx + (vnx - vtx) * sdec[j], vty + (vny - vty) * sdec[j],
                                     vtz + (vnz - vtz) * sdec[j])
                    rel = np.sqrt((vnx - vtx) ** 2 + (vny - vty) ** 2 + (vnz - vtz) ** 2)
                    cap = MAX_SPEED
                    if use_wind:
                        cap = MAX_SPEED + np.sqrt((ub[jj, 0] - wcar[r, 0]) ** 2 + (ub[jj, 1] - wcar[r, 1]) ** 2 +
                                                  (ub[jj, 2] - wcar[r, 2]) ** 2)
                    sp = min(1.0, cap / max(rel, 1e-12))
                    vnx, vny, vnz = vtx + (vnx - vtx) * sp, vty + (vny - vty) * sp, vtz + (vnz - vtz) * sp
                    for i in range(3):
                        tgt_prev[j, i] = tg[jj, i]
                    if cont[jj]:
                        cx, cy, cz = 0.0, 0.0, 0.0
                        if use_wind:
                            cx, cy, cz = wcar[r, 0], wcar[r, 1], wcar[r, 2]
                        vnx, vny, vnz = (cx + (vnx - cx) * stick[j], cy + (vny - cy) * stick[j],
                                         cz + (vnz - cz) * stick[j])
                    v[j, 0], v[j, 1], v[j, 2] = vnx, vny, vnz
                    x[j, 0], x[j, 1], x[j, 2] = p[jj, 0], p[jj, 1], p[jj, 2]
                if rec_every > 0 and (step + 1) % rec_every == 0:
                    _nb_record(rec_f0 + (step + 1) // rec_every - 1, j0, n, H0r, x, Dst, rr, enable, sc[r], sb[r],
                               sm[r], bsc[r], bsr[r], skind, sk, floor_z, has_floor, samp, cand, Sb, Nb, xs, Ds,
                               dep, who)


class _NumbaEngine:
    """The same dynamics as _NumpyEngine, chain by chain in compiled loops, chains spread over threads."""
    name = "numba"

    def __init__(self, M, out, threads):
        self.M, self.out = M, out
        chains = []
        for r in M.roots:
            b = [int(r)]
            while M.child[b[-1]] >= 0:
                b.append(int(M.child[b[-1]]))
            chains.append(b)
        o = self.order = np.concatenate(chains).astype(np.int64)
        self.cptr = np.concatenate([[0], np.cumsum([len(b) for b in chains])]).astype(np.int64)

        def take(a):
            return np.ascontiguousarray(np.asarray(a)[o])

        self.Lc, self.d0, self.alpha, self.rr = take(M.L), take(M.d0), take(M.alpha), take(M.rad)
        self.wgain, self.stick, self.sdec = take(M.wgain), take(M.stick), take(M.sdecay)
        self.decay, self.heavy, self.enable = take(M.decay), take(M.heavy), take(M.enable_u)
        D, x = M.initial_state()
        self.x, self.Dst, self.v = take(x), take(D), np.zeros((M.N, 3))
        self.tgt, self.tgt_prev = self.x.copy(), self.x.copy()
        P = M.P
        self.skind = np.concatenate([np.full(P[k]["n"], i) for i, k in enumerate(geom.KINDS)]).astype(np.int64)
        self.sk = np.zeros((M.K, 5))
        a = 0
        for kind in geom.KINDS:
            n, d = P[kind]["n"], P[kind]
            if kind in ("sphere", "capsule"):
                self.sk[a:a + n, 0] = d["R"]
            elif kind == "box":
                self.sk[a:a + n, 1:4], self.sk[a:a + n, 4] = d["half"], d["rnd"]
            else:
                self.sk[a:a + n, 0], self.sk[a:a + n, 1], self.sk[a:a + n, 4] = d["R"], d["hh"], d["rnd"]
            a += n
        F, N = M.F, M.N
        self.xs, self.Ds = np.zeros((F, N, 3)), np.zeros((F, N, 3, 3))
        self.dep, self.who = np.full((F, N), -1e300), np.full((F, N), -1, np.int64)
        want = threads if threads else (os.cpu_count() or 1)
        # longest chains first, each onto the lightest thread
        bins = [[] for _ in range(max(1, min(int(want), len(chains))))]
        load = [0] * len(bins)
        for c in sorted(range(len(chains)), key=lambda c: -len(chains[c])):
            t = load.index(min(load))
            bins[t].append(c)
            load[t] += len(chains[c])
        self.groups = [np.array(sorted(b), np.int64) for b in bins if b]
        self.pool = ThreadPoolExecutor(len(self.groups)) if len(self.groups) > 1 else None

    def _shapes(self, blk):
        """Per-row shape arrays for the kernels: centres / capsule ends, box and cylinder axes, bounding spheres."""
        if hasattr(blk, "nb"):
            return blk.nb
        P, R, K = self.M.P, blk.R, self.M.K
        sc, sb, bsc, bsr = np.zeros((R, K, 3)), np.zeros((R, K, 3)), np.zeros((R, K, 3)), np.zeros((R, K))
        sm = np.tile(np.eye(3), (R, K, 1, 1))
        a = 0
        for kind in geom.KINDS:
            n = P[kind]["n"]
            if not n:
                continue
            w, s = blk.W[kind], slice(a, a + n)
            if kind == "sphere":
                sc[:, s], bsc[:, s], bsr[:, s] = w[0], w[0], w[1]
            elif kind == "capsule":
                sc[:, s], sb[:, s] = w[0], w[1]
                bsc[:, s], bsr[:, s] = 0.5 * (w[0] + w[1]), w[2] + 0.5 * np.linalg.norm(w[1] - w[0], axis=2)
            elif kind == "box":
                sc[:, s], sm[:, s], bsc[:, s], bsr[:, s] = w[0], w[1], w[0], np.linalg.norm(w[2], axis=2)
            else:
                sc[:, s], sm[:, s], bsc[:, s] = w[0], w[1], w[0]
                bsr[:, s] = np.sqrt(w[2][0] ** 2 + w[2][1] ** 2)
            a += n
        blk.nb = (sc, sb, sm, bsc, bsr)
        return blk.nb

    def _call(self, blk, nsteps, rstride, rec_every, rec_f0, rec_start, dec):
        M = self.M
        sc, sb, sm, bsc, bsr = self._shapes(blk)
        if blk.wind is None:
            z3, e3, e1 = np.zeros((blk.R, 3)), np.zeros((0, 3)), np.zeros(0)
            wind = (z3, z3, np.zeros(blk.R), z3, e3, e3, e1, e1, 0, 0)
        else:
            kv, ae, ph, om = blk.wind.modes
            wind = (blk.wmean, blk.wdisp, blk.wt, blk.wcar, kv, ae, ph, om, 1, int(bool(blk.wind.turbulence)))
        floor = (0.0, 0) if M.floor_z is None else (float(M.floor_z), 1)

        def go(cids):
            _nb_advance(cids, self.cptr, nsteps, rstride, rec_every, rec_f0, rec_start, self.Lc, self.d0, self.alpha,
                        dec, self.wgain, self.stick, self.sdec, self.rr, self.enable, self.x, self.v, self.tgt_prev,
                        self.Dst, self.tgt, blk.H0, blk.Da, sc, sb, sm, bsc, bsr,
                        *[np.ascontiguousarray(a) for a in wind[:4]], self.skind, self.sk, floor[0], floor[1],
                        *wind[4:8], wind[8], wind[9], M.dt, M.samples, M.sweeps, M.ftl, self.xs, self.Ds, self.dep,
                        self.who)

        if self.pool is None:
            go(self.groups[0])
        else:
            for f in [self.pool.submit(go, g) for g in self.groups]:
                f.result()

    def settle(self, blk, steps):
        self._call(blk, steps, 0, 0, 0, -1, self.heavy)

    def start(self, v0):
        self.v += v0

    def record(self, f, blk, r):
        self._call(blk, 0, 0, 0, 0, f, self.decay)

    def run(self, blk, f_first):
        self._call(blk, blk.R, 1, self.M.substeps, f_first, -1, self.decay)

    def finish(self):
        o, out = self.order, self.out
        out.xs[:, o], out.Ds[:, o], out.dep[:, o], out.who[:, o] = self.xs, self.Ds, self.dep, self.who

    def close(self):
        if self.pool is not None:
            self.pool.shutdown()


# ---------------------------------------------------------------- simulate
def _nonfinite(M, eng, a, b):
    """Raise FloatingPointError for the first recorded frame in a..b-1 with a non-finite particle or bone frame."""
    xs, Ds = eng.xs, eng.Ds
    ok = np.isfinite(xs[a:b]).all(axis=(1, 2)) & np.isfinite(Ds[a:b]).all(axis=(1, 2, 3))
    if not ok.all():
        k = a + int(np.argmin(ok))
        bad = np.where(~(np.isfinite(xs[k]).all(1) & np.isfinite(Ds[k]).all((1, 2))))[0]
        names = [M.bones[eng.order[i]] for i in bad[:6]]
        raise FloatingPointError(f"strands: non-finite state at frame {M.f0 + k}: {names}")


def simulate(chains, shapes, src_pos, src_quat, rest_R, rest_p, anchor_src, params, frames, rig, fps=30.0, substeps=10,
             settle_s=1.5, wind=None, log=None, *, use_masks=False, anchor_free=geom.ANCHOR_FREE, engine="auto",
             threads=None):
    """Simulate the chains over frames (f0, f1) inclusive.

    chains: geom.Chains whose head / end are already in world space (the armature's world matrix at the first frame
    applied to the armature-space rest); shapes: geom.Shapes; src_pos (F,S,3) / src_quat (F,S,4, w x y z, unit,
    sign-continuous): posed world frames of shapes.sources over the frames; rest_R (S,3,3) / rest_p (S,3): the rest
    frames of the bone sources in the same world placement (identity / zero for objects and the world); anchor_src (N,):
    source index of every chain bone's anchor bone (a bone source of `shapes`); params {family: {...}} overrides
    FAMILY_DEFAULTS; rig: the rig.json dict (its bodies decide which pairs collide). wind: a Wind or None (still air).
    fps and substeps set the time step 1 / (fps * substeps); the first frame is settled for settle_s seconds.
    use_masks / anchor_free: geom.enable_matrix's options (use the same in the `penetration` check). engine: "auto"
    (numba when installed), "numpy" or "numba"; threads: numba threads (default: the CPU count, at most one per chain).
    Returns a Result."""
    t0 = time.perf_counter()
    if engine not in ("auto", "numpy", "numba"):
        raise ValueError(f"strands: engine must be auto, numpy or numba, not {engine!r}")
    if engine == "numba" and not HAVE_NUMBA:
        raise RuntimeError("strands: numba is not installed (pip install 'mk-mmd[fast]')")
    M = _Model(chains, shapes, src_pos, src_quat, rest_R, rest_p, anchor_src, params, frames, rig, fps, substeps, wind,
               use_masks, anchor_free)
    F, N = M.F, M.N
    out = _Out()
    out.xs, out.Ds = np.zeros((F, N, 3)), np.zeros((F, N, 3, 3))
    out.dep, out.who = np.full((F, N), -np.inf), np.full((F, N), -1, np.int64)
    use_numba = HAVE_NUMBA if engine == "auto" else engine == "numba"
    eng = _NumbaEngine(M, out, threads) if use_numba else _NumpyEngine(M, out)
    t1 = time.perf_counter()
    try:
        blk0 = M.block([0], [0.0], [0.0], settle=True)
        eng.settle(blk0, int(settle_s / M.dt))         # settle under gravity (and the air) on the first frame's pose
        eng.start(wind.carrier_at([0.0])[0] if wind is not None else 0.0)
        eng.record(0, blk0, 0)
        _nonfinite(M, eng, 0, 1)
        t2 = time.perf_counter()
        per_row = 8.0 * (M.K * 20 + len(M.roots) * 12 + 16) * 3               # block memory: about 48 MB at most
        nfb = max(1, int(48e6 // (M.substeps * per_row)))
        for kb in range(0, F - 1, nfb):
            ke = min(kb + nfb, F - 1)
            ks = np.repeat(np.arange(kb, ke), M.substeps)
            taus = np.tile(np.arange(1, M.substeps + 1) / M.substeps, ke - kb)
            eng.run(M.block(ks, taus, (ks + taus) / M.fps), kb + 1)
            _nonfinite(M, eng, kb + 1, ke + 1)
            if log:
                log(f"strands: simulated frame {M.f0 + ke}")
        eng.finish()
    finally:
        eng.close()
    t3 = time.perf_counter()
    res = Result()
    res.frames, res.x, res.D = (M.f0, M.f0 + F - 1), out.xs, out.Ds
    b = np.maximum(out.dep, 0.0)
    res.pen = np.stack([b.max(1), np.median(b, 1), (b > 0.002).sum(1)], 1).astype(float)
    jb = np.argmax(out.dep, axis=1)
    d = out.dep[np.arange(F), jb]
    res.worst = [(float(max(d[k], 0.0)), M.bones[jb[k]], M.labels[out.who[k, jb[k]]] if d[k] > 0 else "-")
                 for k in range(F)]
    res.labels = M.labels
    res.anchor_pos, res.anchor_rot, res.anchor_rest_rot = M.anchor_pos, M.anchor_rot, M.rest_R[M.a_src]
    res.engine = eng.name
    res.timings = {"setup": round(t1 - t0, 3), "settle": round(t2 - t1, 3), "run": round(t3 - t2, 3)}
    return res


# ---------------------------------------------------------------- bake + report
def _continuous(q):
    """Quaternions (F, ..., 4) with signs flipped so that neighbours along axis 0 have a non-negative dot product."""
    dots = np.einsum("f...j,f...j->f...", q[1:], q[:-1])
    flip = np.cumprod(np.concatenate([np.ones((1,) + dots.shape[1:]), np.where(dots + 1e-12 < 0, -1.0, 1.0)]), axis=0)
    return q * flip[..., None]


def quats_from_matrices(Rs):
    """(F,S,3,3) rotations -> (F,S,4) unit quaternions w x y z, sign-continuous over the frames: what `simulate`'s
    src_quat wants (matrices with scale removed first, e.g. geom.unscaled)."""
    Rs = np.asarray(Rs, float)
    return _continuous(geom.mat_to_quat(Rs.reshape(-1, 3, 3)).reshape(Rs.shape[:2] + (4,)))


def local_quats(chains, res, rest_R_bones):
    """(F,N,4) bone-local rotation quaternions (w x y z, sign-continuous in time) that put every chain bone where the
    simulation says: q = D_parent^T D (the bone's turn in world axes beyond its parent's), local = B^T q B with B the
    bone's rest rotation (rest_R_bones (N,3,3), in the same world placement as the simulation's rest frames). A chain
    root's parent is its anchor: D_anchor = anchor_rot @ anchor_rest_rot^T. Key these on every simulated frame with
    linear interpolation."""
    F, N = res.x.shape[:2]
    par = np.asarray(chains.parent, int)
    Da = res.anchor_rot @ np.transpose(res.anchor_rest_rot, (0, 2, 1))[None]
    Dp = np.where((par < 0)[None, :, None, None], Da, res.D[:, np.maximum(par, 0)])
    q = np.swapaxes(Dp, -1, -2) @ res.D
    B = np.asarray(rest_R_bones, float)[None]
    ql = geom.mat_to_quat((np.swapaxes(B, -1, -2) @ q @ B).reshape(-1, 3, 3)).reshape(F, N, 4)
    if not np.isfinite(ql).all():
        bad = np.where(~np.isfinite(ql).all((0, 2)))[0]
        k = np.where(~np.isfinite(ql[:, bad[0]]).all(1))[0]
        raise FloatingPointError(f"strands: non-finite rotation for {chains.bones[bad[0]]} at frames "
                                 f"{(res.frames[0] + k)[:6].tolist()}")
    return _continuous(ql)


def report(chains, res, window):
    """Per family over frames window=(a, b) inclusive, in the chain anchor's frame: median frame-to-frame change in
    velocity ('jerk') vs median speed (mm) and their ratio (Bullet hair while a character sits: ~1.4; calm solved hair
    ~0.4; dancing ~0.36), plus the residual penetration of segment points into any shape."""
    f0 = res.frames[0]
    a, b = window[0] - f0, window[1] - f0 + 1
    if a < 0 or b > len(res.x) or b - a < 3:
        raise ValueError(f"strands: report window {tuple(window)} needs >= 3 frames inside {res.frames}")
    x = np.einsum("fnji,fnj->fni", res.anchor_rot[a:b], res.x[a:b] - res.anchor_pos[a:b]) * 1000.0
    acc = np.linalg.norm(x[2:] - 2 * x[1:-1] + x[:-2], axis=2)
    vel = np.linalg.norm(x[1:] - x[:-1], axis=2)
    out = {}
    fam = np.array(chains.family)
    for f in sorted(set(chains.family)):
        sel = np.where(fam == f)[0]
        ma, mv = float(np.median(acc[:, sel])), float(np.median(vel[:, sel]))
        out[f] = dict(jerk_mm=round(ma, 3), speed_mm=round(mv, 3), ratio=round(ma / max(mv, 1e-6), 2),
                      jerk_p95_mm=round(float(np.percentile(acc[:, sel], 95)), 2))
    k = a + int(np.argmax(res.pen[a:b, 0]))
    deep = {}
    for d, bone, tag in res.worst[a:b]:
        if d > 0.002:
            deep[tag] = deep.get(tag, 0) + 1
    out["penetration_mm"] = dict(max=round(float(res.pen[k, 0]) * 1000, 2), at_frame=f0 + k, bone=res.worst[k][1],
                                 into=res.worst[k][2], median=round(float(np.median(res.pen[a:b, 1])) * 1000, 3),
                                 frames_over_2mm_by_shape=dict(sorted(deep.items(), key=lambda kv: -kv[1])[:5]))
    return out


# ---------------------------------------------------------------- command line
def main(argv=None):
    """python -m mkmmd.solvers.strands IN.npz OUT.npz (module docstring: the file formats). Returns the exit code."""
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) != 2:
        print("usage: python -m mkmmd.solvers.strands IN.npz OUT.npz", file=sys.stderr)
        return 2
    t0 = time.perf_counter()
    try:
        with np.load(argv[0], allow_pickle=False) as z:
            a = {k: z[k] for k in z.files}
        spec = json.loads(str(a["spec"]))
        rig = spec["rig"]
        fps = float(spec.get("fps", 30.0))
        chains = geom.Chains(rig, spec.get("families"))
        chains.head, chains.end = np.asarray(a["head"], float), np.asarray(a["end"], float)
        if chains.head.shape != (len(chains), 3) or chains.end.shape != (len(chains), 3):
            raise ValueError(f"head / end must be ({len(chains)}, 3) for the chains of families {spec.get('families')}")
        armature = spec.get("armature", "")
        shapes = geom.Shapes()
        for s in spec["sources"]:
            shapes.src(*s)
        n_src = len(shapes.sources)
        if spec.get("model_bodies", True):
            geom.model_shapes(shapes, rig, armature, skip_bones=chains.chain_bones)
        for it in spec.get("colliders", []):
            shapes.add_item(it)
        anchor_src = np.array([shapes.src("bone", armature, name) for name in chains.anchor])
        if len(shapes.sources) != n_src:
            raise ValueError(f"spec.sources lacks {shapes.sources[n_src:]}")
        wind = Wind(spec.get("wind"), a.get("carrier_vel"), fps) if spec.get("wind") is not None or "carrier_vel" in a \
            else None
        frames = tuple(int(f) for f in spec["frames"])
        window = tuple(int(f) for f in spec.get("window", frames))
        bone_rest_R = np.asarray(a["bone_rest_R"], float)
    except (KeyError, ValueError, OSError, json.JSONDecodeError) as e:
        print(f"strands: bad input: {type(e).__name__}: {e}", file=sys.stderr)
        return 2
    log = lambda m: print(m, file=sys.stderr, flush=True)                       # noqa: E731
    try:
        res = simulate(chains, shapes, a["src_pos"], a["src_quat"], a["rest_R"], a["rest_p"], anchor_src,
                       spec.get("params"), frames, rig, fps, int(spec.get("substeps", 10)),
                       float(spec.get("settle_s", 1.5)), wind, log, use_masks=bool(spec.get("use_masks", False)),
                       anchor_free=float(spec.get("anchor_free", geom.ANCHOR_FREE)), engine=spec.get("engine", "auto"),
                       threads=spec.get("threads"))
        quats = local_quats(chains, res, bone_rest_R)
        rep = report(chains, res, window)
    except (ValueError, RuntimeError) as e:                                    # bad input, or numba asked for but absent
        print(f"strands: bad input: {e}", file=sys.stderr)
        return 2
    except FloatingPointError as e:
        print(str(e), file=sys.stderr)
        return 3
    info = {"bones": chains.bones, "frames": list(res.frames), "engine": res.engine, "report": rep,
            "seconds": dict(res.timings, total=round(time.perf_counter() - t0, 3))}
    np.savez(argv[1], quats=quats, x=res.x, pen=res.pen, report=np.array(json.dumps(info, ensure_ascii=False)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
