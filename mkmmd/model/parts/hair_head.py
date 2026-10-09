"""The head hair of the hair part (bpy-free, numpy only): scalp cap, crown clumps, bangs, side locks and back hair.

Everything is built around `HeadFit.skull` and a `Volume` (how far the hair stands off the skin: ~3 cm at the crown,
~4 cm at the sides, ~2.7 cm at the back, tapering below the widest part). Layers are offsets from that hull: cap -14 mm,
crown clumps -7 mm, main clumps 0, front clumps +6 mm. Clumps ("locks") are lens-shaped shells swept along meridian paths
(attached to the hull, then hanging); the locks of bangs, side hair and back hair hang from dynamic chains (bones
前髪i_k, 横髪左i_k, 後髪i_k; a chain also carries the back-layer clumps next to it), the cap and crown clumps are rigid on
the head bone. Hanging locks keep BODY_GAP off the body part's skin below the neck seam (`off_body`): the hull goes on
under the head as a straight neck, a body's neck may widen sooner.

Vertex heights drive the texture (hair_tex.head_v), so the angel-ring band is one ring around the head across every clump."""
from dataclasses import dataclass, field, replace

import numpy as np

from . import hair_tex as TEX
from .hair_braids_path import relax
from .hair_fit import Volume, angles, dirs, unit
from .hair_geo import MeshAccum, Piece, arclen, resample, smooth_polyline, smoothstep, strip_normals, sweep
from .outfit_fit import Skin
from .. import spec as SP
from ..part import Material


def shade_proxy(P, center, z_hi, z_lo):
    """Unit normals of the smooth proxy at points P (n,3): from the nearest point of the vertical axis through `center`
    (between heights z_lo and z_hi) to the point."""
    P = np.asarray(P, float)
    axis = np.stack([np.full(len(P), center[0]), np.full(len(P), center[1]), np.clip(P[:, 2], z_lo, z_hi)], -1)
    return unit(P - axis)


def hex_rgb(h):
    h = h.lstrip("#")
    return np.array([int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4)])

DEFAULTS = {
    "volume": {"top": 0.036, "front": 0.020, "side": 0.034, "back": 0.027, "lift": 0.016, "lift_deg": 28.0,
               "lift_width_deg": 20.0, "lift_back": 0.5},
    "ring": {"enabled": True, "drop": 0.050, "width": 0.018},  # baked crown sheen: on/off, metres below the hair top, half thickness
    "cap": {"delta": -0.014, "cols": 72, "rows": 15},
    "crown": {"count": 9, "width": 0.070, "thick": 0.016, "delta": -0.004, "swirl": 0.45, "inner_delta": -0.011},
    "bangs": {"count": 11, "width": 0.028, "thick": 0.014, "delta": 0.004, "span_deg": 44.0, "above_eye": 0.027,
              "var": 0.005, "taper": [0.016, 0.026], "back_up": 0.010, "tip_delta": -0.008,
              "strands": [[20.0, -14.0, 0.011], [-3.0, 9.0, 0.010], [-24.0, 13.0, 0.012]], "strand_drop": 0.006,
              "shadow": {"enabled": True, "tint": [0.60, 0.36, 0.34], "tint2": [0.82, 0.62, 0.60], "tooth": 0.005,
                         "valley": 0.004, "round": 0.30, "step": 0.0035, "above": 0.016, "off": 0.0006}},
    "side": {"width": 0.042, "thick": 0.015, "delta": 0.002, "chin_gap": 0.0, "sway": 0.004},
    "back": {"count": 8, "width": 0.076, "thick": 0.018, "delta": 0.0, "end_above_chin": 0.0, "sway": 0.008},
    "k_outer": 7,                                       # vertices across a clump (rounder cross-section)
    "bulge": 0.5,                                       # lens exponent: 0.5 = elliptical (thick in the middle, clean edges)
    "toon": {"edge": 0.52, "soft": 0.20},
    "normal_mix": 0.88,                                 # share of the smooth head-shaped proxy in the normals of a clump
    "edge_mix": 0.78,                                   # ... at the edges of a clump (so a clump has no rim of its own)
    "shade": {"z_hi": 0.0, "z_lo": -0.035},             # the proxy: a vertical axis from head centre + z_lo to + z_hi
    "dynamic": True,
}


EN = {"前髪": "bangs", "横髪左": "side_L", "横髪右": "side_R", "後髪": "back"}
BODY_GAP = 0.004               # every vertex of a lock below the neck seam keeps this off the body's skin (m)


# ---------------------------------------------------------------- paths
def _hang(skull, P, tuck, flare, bend, step, margin, z_end, drop):
    """Free-hanging part after the attached points P: the heading relaxes towards straight down plus a tuck (inward) /
    flare (outward) component, kept `margin` above the skin, until z_end (or `drop` metres). Returns the full path."""
    c = skull.center
    pts = [P[-1]]
    h = unit(P[-1] - P[-2])
    for it in range(500):
        rad = pts[-1] - c
        rxy = unit(np.array([rad[0], rad[1], 0.0]))
        hang = unit(np.array([0.0, 0.0, -1.0]) - tuck * rxy + flare * rxy)
        h = unit(h + bend * (hang - h))
        p = skull.push_out(pts[-1] + step * h, margin)
        if z_end is not None and p[2] <= z_end:
            q = pts[-1]
            t = (q[2] - z_end) / max(q[2] - p[2], 1e-9)
            pts.append(q + t * (p - q))
            break
        pts.append(p)
        if z_end is None and it * step >= drop:
            break
    return np.vstack([P, np.array(pts[1:])])


def hermite(p0, m0, p1, m1, n):
    t = np.linspace(0.0, 1.0, n)[:, None]
    return ((2 * t ** 3 - 3 * t ** 2 + 1) * p0 + (t ** 3 - 2 * t ** 2 + t) * m0 + (-2 * t ** 3 + 3 * t ** 2) * p1 +
            (t ** 3 - t ** 2) * m1)


def meridian_path(skull, vol, th0, ph0, ph_leave, *, dth=0.0, delta0=-0.006, delta1=0.0, z_end=None, drop=0.0,
                  tuck=0.0, flare=0.0, bend=0.30, step=0.010, n_att=20, margin=0.010, x_end=None, end_dir=None, min_off=0.009):
    """Centreline of a lock. Attached: on the hull (skin + volume + delta) from (th0, ph0) down to polar angle ph_leave,
    azimuth drifting by dth, `delta` easing delta0 -> delta1. Then hanging: heading relaxes towards straight down with a
    `tuck` (inward) / `flare` (outward) component, kept `margin` above the skin, until z_end (or `drop` metres). With
    `x_end` the hanging part is a smooth Hermite curve to (x_end, y of the leave point, z_end) arriving along `end_dir`
    (default straight down)."""
    u = np.linspace(0.0, 1.0, n_att)
    ph = ph0 + (ph_leave - ph0) * u
    th = th0 + dth * u ** 1.6
    delta = delta0 + (delta1 - delta0) * smoothstep(u)
    P = skull.point(dirs(th, ph), np.maximum(vol(th, ph) + delta, min_off))
    if z_end is not None and (P[:, 2] <= z_end).any():
        k = int(np.argmax(P[:, 2] <= z_end))
        if k == 0:
            return P[:1].repeat(2, 0)
        t = (P[k - 1, 2] - z_end) / max(P[k - 1, 2] - P[k, 2], 1e-9)
        P = np.vstack([P[:k], P[k - 1] + t * (P[k] - P[k - 1])])
        return smooth_polyline(resample(P, max(int(arclen(P)[-1] / step) + 1, 5)), 1)
    if x_end is not None:
        h0 = unit(P[-1] - P[-2])
        E = np.array([x_end, P[-1, 1], z_end])
        e = unit(np.array([0.0, 0.0, -1.0]) if end_dir is None else np.asarray(end_dir, float))
        L = float(np.linalg.norm(E - P[-1]))
        Q = hermite(P[-1], h0 * L * 0.55, E, e * L * 0.55, max(int(L / step), 6))
        Q = np.array([skull.push_out(q, margin) for q in Q])
        P = np.vstack([P, Q[1:]])
        return smooth_polyline(resample(P, max(int(arclen(P)[-1] / step) + 1, 5)), 2)
    P = _hang(skull, P, tuck, flare, bend, step, margin, z_end, drop)
    P = resample(P, max(int(arclen(P)[-1] / step) + 1, 5))
    return smooth_polyline(P, 2)


def radial_hint(skull, P):
    return unit(P - skull.center)


@dataclass
class Lock:
    name: str
    path: np.ndarray
    width: float
    thick: float
    tip: str = "point"
    tip_start: float = 0.55
    tip_power: float = 1.5
    root_w: float = 0.7
    curv: float = 8.0
    tilt: float = 0.0
    end_taper: float = 0.0
    chain: str = ""                     # chain id (shared by several locks) or "" (rigid on the head)
    layer: int = 0
    theta: float = 0.0
    dark: bool = False                  # take the darker texture shade (inner layers)
    narrow: bool = False                # use only the middle of the tile (thin strands: no clump-edge shading)
    extra: dict = field(default_factory=dict)


class HeadHair:
    def __init__(self, ctx, fit, cfg, rig, pal):
        self.ctx, self.fit, self.rig, self.pal = ctx, fit, rig, pal
        self.cfg = SP.merge(DEFAULTS, cfg)
        self.rng = ctx.rng_for("hair.head")
        self.skull = fit.skull
        v = self.cfg["volume"]
        fit.vol = Volume(self.skull, v["top"], v["front"], v["side"], v["back"], taper_from=fit.vol.taper_from,
                         lift=v["lift"], lift_phi=np.radians(v["lift_deg"]), lift_width=np.radians(v["lift_width_deg"]),
                         lift_back=v["lift_back"])                          # shared with the other builders
        self.vol = fit.vol
        self.head_body = ctx.find_body("頭")
        body = getattr(ctx, "parts", {}).get("body")
        nt = (body.info or {}).get("neck_top") if body is not None else None
        self._body = (body, nt) if nt is not None else None
        self._skin = None
        self.chains = {}
        self.top_z = self.vol.max_z()
        self.z_top = self.top_z + 0.004
        self.z_bot = float(fit.chin[2]) - 0.05
        span = self.z_top - self.z_bot
        self.ring_f, self.ring_w = self.cfg["ring"]["drop"] / span, self.cfg["ring"]["width"] / span
        self.rng_tex = ctx.rng_for("hair.head.tex")
        self.mat = "髪"
        self.stats = {}

    # ---- helpers
    def height_uv(self, V, q, win, dz=0.0):
        v = TEX.head_v(V[:, 2] + dz, self.z_top, self.z_bot)
        u0, u1 = win
        return np.stack([u0 + (q + 1) * 0.5 * (u1 - u0), v], -1)

    def tint(self):
        return TEX.clump_u(self.rng)

    def phi_at_z(self, theta, z):
        """Polar angle at which the hull in direction theta reaches height z (None when it does not)."""
        ph = np.linspace(0.02, np.pi - 0.02, 400)
        P = self.skull.point(dirs(np.full(len(ph), theta), ph), self.vol(np.full(len(ph), theta), ph))
        k = np.argmax(P[:, 2] <= z)
        return float(ph[k]) if (P[:, 2] <= z).any() else None

    def s_at_phi(self, P, phi):
        v = P - self.skull.center
        ph = np.arccos(np.clip(v[:, 2] / np.linalg.norm(v, axis=1), -1, 1))
        s = arclen(P)
        k = int(np.argmax(ph >= phi)) if (ph >= phi).any() else len(P) - 1
        return float(s[k])

    def s_at_z(self, P, z):
        s = arclen(P)
        k = int(np.argmax(P[:, 2] <= z)) if (P[:, 2] <= z).any() else len(P) - 1
        return float(s[k])

    def proxy_normals(self, P):
        """Normals of the smooth head-shaped proxy the hair is shaded like (as the face is): from the nearest point of a vertical
        axis through the head (from `z_lo` below to `z_hi` above the head centre) to P. The crown and the fringe tilt up and
        are lit, the sides are horizontal, the lower hair tilts down: ONE smooth terminator across the whole mass instead of
        a step per clump."""
        return shade_proxy(P, self.fit.center, self.fit.center[2] + self.cfg["shade"]["z_hi"],
                           self.fit.center[2] + self.cfg["shade"]["z_lo"])

    def sway(self, P, amp, z_ref, z_end, turns=1.5):
        """A soft S-curve in a hanging lock: from `z_ref` down to `z_end` the path is displaced sideways (horizontally, along
        the head's surface) by amp x sin(turns x pi x u), so the tip swings out the other way; the attached part (above
        z_ref) stays on the hull. Points are kept clear of the skin."""
        P = np.array(P, float)
        span = max(z_ref - z_end, 1e-6)
        u = np.clip((z_ref - P[:, 2]) / span, 0.0, 1.0)
        rad = P - self.skull.center
        rad[:, 2] = 0.0
        lat = unit(np.cross(np.array([0.0, 0.0, 1.0]), rad))
        d = amp * np.sin(turns * np.pi * u) * smoothstep(u / 0.12)
        return smooth_polyline(self.skull.push_out(P + lat * d[:, None], 0.009), 1)

    # ---- the cap
    def cap(self, acc):
        c = self.cfg["cap"]
        sk, fit = self.skull, self.fit
        nth, rows = int(c["cols"]), int(c["rows"])
        th = -np.pi + np.arange(nth) * 2 * np.pi / nth
        phi_cap = fit.cap_phi(th) + np.radians(1.5)
        t = np.linspace(0.0, 1.0, rows + 1)[1:]
        PH = t[:, None] * phi_cap[None, :]
        TH = np.broadcast_to(th[None, :], PH.shape)
        D = dirs(TH, PH)
        dl = c["delta"]
        off = np.maximum(self.vol(TH, PH) + dl, 0.006)             # never closer than 6 mm to the skin (below the head's
        Po = sk.point(D, off)                                       # widest part the volume tapers off)
        Pi = sk.point(D, off - 0.003)
        top = np.array([0, 0, 1.0])
        off_top = max(float(self.vol(0.0, 0.0)) + dl, 0.006)
        pole_o, pole_i = sk.point(top, off_top), sk.point(top, off_top - 0.003)
        verts = [pole_o, pole_i, *Po.reshape(-1, 3), *Pi.reshape(-1, 3)]
        n_o = rows * nth
        io = lambda i, j: 2 + i * nth + (j % nth)
        ii = lambda i, j: 2 + n_o + i * nth + (j % nth)
        faces = []
        for j in range(nth):
            faces.append([0, io(0, j), io(0, j + 1)])                      # outer fan (CCW seen from above = outward)
            faces.append([1, ii(0, j + 1), ii(0, j)])                      # inner fan, reversed
        for i in range(rows - 1):
            for j in range(nth):
                faces.append([io(i, j), io(i + 1, j), io(i + 1, j + 1), io(i, j + 1)])
                faces.append([ii(i, j), ii(i, j + 1), ii(i + 1, j + 1), ii(i + 1, j)])
        for j in range(nth):                                                # the rim joins outer and inner
            faces.append([io(rows - 1, j), ii(rows - 1, j), ii(rows - 1, j + 1), io(rows - 1, j + 1)])
        V = np.array(verts)
        thv = np.concatenate([[0.0, 0.0], np.tile(th, rows), np.tile(th, rows)])
        uv = np.stack([TEX.cap_u(thv), TEX.head_v(V[:, 2], self.z_top, self.z_bot)], -1)
        nrm = self.proxy_normals(V)
        nrm[1 + n_o + 1:] *= -1.0                                    # the inner surface faces the skin
        nrm[1] *= -1.0
        acc.add(V, faces, uv, 0, weights={"頭": np.ones(len(V))}, normals=nrm)
        self.stats["cap_verts"] = len(V)

    # ---- clumps
    def clump(self, lk):
        """The swept clump (hair_geo.Strip) of a lock, as `emit` meshes it."""
        return sweep(lk.path, radial_hint(self.skull, lk.path), lk.width, lk.thick, tip=lk.tip, tip_start=lk.tip_start,
                     tip_power=lk.tip_power, root_w=lk.root_w, curv=lk.curv, tilt=lk.tilt, end_taper=lk.end_taper,
                     k_outer=self.cfg["k_outer"], bulge=self.cfg["bulge"])

    def emit(self, acc, lk, chain=None):
        win = TEX.clump_u(self.rng_tex, dark=lk.dark)
        if lk.narrow:
            win = (win[0] + 0.40 * (win[1] - win[0]), win[0] + 0.60 * (win[1] - win[0]))
        dz = float(self.rng_tex.uniform(-0.0035, 0.0035))          # moves this clump's highlight a little
        st = self.clump(lk)
        uv = self.height_uv(st.verts, st.q, win, dz)
        w = {"頭": np.ones(len(st.verts))} if chain is None else chain.weights(st.verts)
        nrm = strip_normals(st, self.proxy_normals(st.verts), self.cfg["normal_mix"], self.cfg["edge_mix"])
        acc.add(st.verts, st.faces, uv, 0, weights=w, normals=nrm)
        return st

    def off_body(self, lk):
        """`lk` with its path lifted where its clump comes closer than BODY_GAP to the body's skin. The skull hull the
        hanging locks are kept off goes on under the head as a straight neck, so a lock that hangs to the nape ends inside
        a body whose neck widens sooner. The body ends at the neck seam, open there: measured are the clump's vertices
        below the seam and up to the gap above it, but not the ones over the neck's open top (inside the seam's ellipse:
        the head's, where the body's skin says nothing). Each vertex short of the gap lifts the path point nearest along
        the lock, the shortest way off the skin, and hair_braids_path.relax blurs the lift along the path: the lock bends
        smoothly over the skin and its chain, made from the path, follows. A lock already clear comes back as it was."""
        if self._body is None:
            return lk
        body, nt = self._body
        (cx, cy), rx, ry, z_seam = nt["center"], float(nt["rx"]), float(nt["ry"]), float(nt["z"])

        def measured(X):
            over = (X[:, 2] >= z_seam) & (((X[:, 0] - cx) / rx) ** 2 + ((X[:, 1] - cy) / ry) ** 2 <= 1.0)
            return np.flatnonzero((X[:, 2] < z_seam + BODY_GAP) & ~over)

        X = self.clump(lk).verts
        low = measured(X)
        if not len(low):
            return lk
        if self._skin is None:
            self._skin = Skin.from_meshes([m for m in body.meshes if len(m.verts)], "body")
        T = self._skin.verts[self._skin.tris]
        lo, hi = X[low].min(0) - 0.05, X[low].max(0) + 0.05     # the skin within reach (a lift moves a lock ~1 cm)
        skin = self._skin.where(tri_mask=((T.max(1) >= lo) & (T.min(1) <= hi)).all(1))
        if not len(skin.tris):
            return lk

        def field(Q):
            st = self.clump(replace(lk, path=Q))
            low = measured(st.verts)
            sd, away = np.full(len(Q), np.inf), np.zeros_like(Q)
            if not len(low):
                return sd, away
            V = st.verts[low]
            d, q, tri, bary = skin.closest(V)
            n = V - q
            nl = np.linalg.norm(n, axis=1, keepdims=True)
            out = np.where(nl > 1e-6, np.sign(d)[:, None] * n / np.maximum(nl, 1e-12), skin.outward(tri, bary))
            k = np.abs(arclen(Q)[None, :] - st.arc[low][:, None]).argmin(1)
            order = np.lexsort((d, k))                          # per path point its vertex furthest under the gap
            first = order[np.concatenate([[True], k[order][1:] != k[order][:-1]])]
            sd[k[first]], away[k[first]] = d[first], out[first]
            return sd, away
        if not (field(lk.path)[0] < BODY_GAP).any():
            return lk
        return replace(lk, path=relax(lk.path, [(BODY_GAP, field)]))

    def make_chain(self, base, lock, kind, n_bones, s_root, radius):
        P = lock.path
        s = arclen(P)
        sh = np.linspace(s_root, s[-1], n_bones + 1)
        pts = np.stack([np.interp(sh, s, P[:, k]) for k in range(3)], -1)
        ab = self.head_body.name if self.head_body is not None else None
        en = base
        for jp, e in EN.items():
            en = en.replace(jp, e)
        return self.rig.chain(base, pts, "頭", kind=kind, radius=radius, anchor_body=ab, name_en=en,
                              dynamic=bool(self.cfg["dynamic"]))

    # ---- layouts
    def crown(self, acc):
        """Big rounded clumps swirling out of a soft whorl on the crown: an outer layer (light tiles) over an inner layer a
        half pitch aside (the darker tiles), whose darker red shows through the gaps between the outer clumps."""
        c = self.cfg["crown"]
        n = c["count"]
        for layer in (0, 1):
            for i in range(n):
                th = -np.pi + (i + 0.5 + 0.5 * layer + self.rng.uniform(-0.3, 0.3)) * 2 * np.pi / n
                ph_h = float(self.fit.cap_phi(th))
                ph0 = np.radians(self.rng.uniform(2, 5))
                ph_end = float(np.clip(self.rng.uniform(0.82, 1.0) * ph_h * (0.92 if layer else 1.0), np.radians(30),
                                       np.radians(90)))
                dl = c["inner_delta"] if layer else c["delta"]
                P = meridian_path(self.skull, self.vol, th, ph0, ph_end, dth=c["swirl"] + self.rng.uniform(-0.08, 0.08),
                                  delta0=dl - 0.002, delta1=dl + 0.002 + self.rng.uniform(-0.005, 0.006), n_att=16)
                w = c["width"] * self.rng.uniform(0.86, 1.18) * (1.1 if layer else 1.0)
                self.emit(acc, Lock(f"crown{layer}_{i}", P, w, c["thick"], tip="point", tip_start=0.55, tip_power=1.2,
                                    root_w=0.9, curv=9.0, layer=1, theta=th, dark=bool(layer)))
        self.stats["crown"] = 2 * n

    def bangs(self, acc):
        """The fringe: `count` separate pointed clumps across the forehead (each a tapered wedge ending in a sharp tip, tips
        within +-`var` of the cut line, which stays straight and follows the forehead a little), a darker back layer of
        shorter clumps half a pitch aside, and a few thin loose strands crossing in front. The clumps descend from over the
        crown to ~1 cm above the forehead, so a soft shadow strip (`forehead_shadow`) fits under them."""
        c, fit = self.cfg["bangs"], self.fit
        n = int(c["count"])
        z_cut = fit.eye_z() + c["above_eye"]
        span = np.radians(c["span_deg"])
        ths = np.linspace(span, -span, n)
        pitch = 2 * span / max(n - 1, 1)
        prim, tips = [], []
        for layer in (0, 1):
            th_l = ths[:-1] - 0.5 * pitch if layer == 0 else ths
            for i, th0 in enumerate(th_l):
                j = th0 + self.rng.uniform(-0.12, 0.12) * pitch
                var = self.rng.uniform(-1.0, 1.0) * c["var"]
                tip_z = z_cut + var + 0.004 * (abs(th0) / span) ** 2 + (c["back_up"] if layer == 0 else 0.0)
                ph_h = float(fit.cap_phi(j))
                ph_leave = self.phi_at_z(j, tip_z)
                dl = c["tip_delta"] - (0.004 if layer == 0 else 0.0) + 0.003 * (i % 2)
                P = meridian_path(self.skull, self.vol, j, 0.45 * ph_h,
                                  np.radians(125.0) if ph_leave is None else ph_leave + 0.02,
                                  dth=j * 0.06 + self.rng.uniform(-0.05, 0.05),
                                  delta0=c["delta"] - (0.004 if layer == 0 else 0.0), delta1=dl, z_end=tip_z, n_att=28)
                L = arclen(P)[-1]
                T = self.rng.uniform(*c["taper"])
                w = c["width"] * self.rng.uniform(0.9, 1.15) * (1.1 if layer == 0 else 1.0)
                prim.append(Lock(f"bangs{layer}_{i}", P, w, c["thick"], tip="point",
                                 tip_start=max(0.0, 1.0 - T / max(L, 1e-6)), tip_power=0.95, root_w=0.5, curv=11.0,
                                 layer=2 + layer, theta=j, dark=(layer == 0)))
                if layer == 1:
                    tips.append(tip_z)
        front = [l for l in prim if l.layer == 3]
        back = [l for l in prim if l.layer == 2]
        for i, lk in enumerate(front):
            s_root = self.s_at_z(lk.path, z_cut + 0.065)
            ch = self.make_chain(f"前髪{i + 1}_", lk, "bangs", 2, s_root, max(0.0035, c["width"] * 0.13))
            self.chains[f"bangs{i}"] = ch
            lk.chain = f"bangs{i}"
        for lk in back:
            lk.chain = f"bangs{int(np.argmin([abs(lk.theta - f.theta) for f in front]))}"
        loose = []
        for k, (th_d, drift_d, w) in enumerate(c["strands"]):                    # thin strands crossing in front of the fringe
            th0 = np.radians(th_d)
            z_l = z_cut - c["strand_drop"] * (0.5 + 0.5 * (k % 2)) + 0.002 * k
            ph_leave = self.phi_at_z(th0 + np.radians(drift_d), z_l) or np.radians(100.0)
            P = meridian_path(self.skull, self.vol, th0, 0.4 * float(fit.cap_phi(th0)), ph_leave + 0.02,
                              dth=np.radians(drift_d), delta0=c["delta"], delta1=c["tip_delta"] + 0.007, z_end=z_l, n_att=28)
            lk = Lock(f"bangs_loose{k}", P, w, c["thick"] * 0.55, tip="point", tip_start=0.35, tip_power=1.1, root_w=0.6,
                      curv=11.0, layer=4, theta=th0, narrow=True)
            lk.chain = f"bangs{int(np.argmin([abs(th0 - f.theta) for f in front]))}"
            loose.append(lk)
            tips.append(z_l)
        for lk in back + front + loose:
            self.emit(acc, lk, self.chains[lk.chain])
        self.stats["bangs"] = len(prim) + len(loose)
        self.stats["bangs_front"] = len(front)
        self.stats["bangs_back"] = len(back)
        self.stats["bangs_loose"] = len(loose)
        self.stats["bangs_bottom_z"] = float(min(tips))
        self.stats["bangs_cut_z"] = float(z_cut)
        self.stats["bangs_tip_z"] = [float(t) for t in tips[:len(front)]]
        self.tips = [(float(angles(l.path[-1] - self.skull.center)[0]), float(l.path[-1, 2])) for l in front]

    def skin_point_at_z(self, theta, z):
        """Points on the skin (radial surface) at azimuths theta (array) and heights z (array or scalar), plus the unit
        normals there."""
        theta = np.atleast_1d(np.asarray(theta, float))
        z = np.broadcast_to(np.asarray(z, float), theta.shape)
        ph = np.linspace(0.2, 2.4, 600)
        out = np.empty((len(theta), 3))
        for i, (t, zz) in enumerate(zip(theta, z)):
            P = self.skull.point(dirs(np.full(len(ph), t), ph), 0.0)
            k = max(int(np.argmax(P[:, 2] <= zz)), 1)
            a = (P[k - 1, 2] - zz) / max(P[k - 1, 2] - P[k, 2], 1e-9)
            out[i] = P[k - 1] + a * (P[k] - P[k - 1])
        d = out - self.skull.center
        return out, self.skull.normal(d / np.linalg.norm(d, axis=-1, keepdims=True))

    def forehead_shadow(self, ctx):
        """The shadow the fringe casts on the forehead, as two cel-shading steps: strips lying 0.3 mm above the skin, shaped
        like the fringe (the top hidden under the clumps, the lower edge a zig-zag with a tooth below every tip and a valley
        in every gap), drawn with the SKIN'S OWN material (same texture, toon and UVs, only darker and warmer), so they match
        the skin exactly and, being opaque, never dither. Returns (mesh, materials)."""
        c = self.cfg["bangs"]
        sh = c["shadow"]
        tips = sorted(self.tips, key=lambda t: -t[0])
        th_t = np.array([t[0] for t in tips])
        z_t = np.array([t[1] for t in tips])
        pitch = float(np.mean(np.abs(np.diff(th_t)))) if len(th_t) > 1 else 0.15
        ctrl_t, ctrl_z = [th_t[0] + pitch], [z_t[0] + 0.012]
        for i in range(len(th_t)):
            ctrl_t.append(th_t[i])
            ctrl_z.append(z_t[i] - sh["tooth"])
            if i + 1 < len(th_t):
                ctrl_t.append(0.5 * (th_t[i] + th_t[i + 1]))
                ctrl_z.append(0.5 * (z_t[i] + z_t[i + 1]) + sh["valley"])
        ctrl_t.append(th_t[-1] - pitch)
        ctrl_z.append(z_t[-1] + 0.012)
        ctrl_t, ctrl_z = np.array(ctrl_t)[::-1], np.array(ctrl_z)[::-1]                    # ascending azimuth for interp
        nu = 97
        ths = np.linspace(th_t[0] + 0.5 * pitch, th_t[-1] - 0.5 * pitch, nu)
        z_low = np.interp(ths, ctrl_t, ctrl_z)
        k = max(int(round(sh["round"] * pitch / abs(ths[1] - ths[0]))), 1)      # round the teeth: a soft undulating shadow edge,
        ker = np.exp(-0.5 * (np.arange(-3 * k, 3 * k + 1) / k) ** 2)           # not a second row of identical triangles
        z_low = np.convolve(np.pad(z_low, 3 * k, mode="edge"), ker / ker.sum(), mode="valid")
        z_top = np.full(nu, self.stats["bangs_cut_z"] + sh["above"])
        head_mats = {m.name: m for m in ctx.parts["head"].materials} if "head" in ctx.parts else {}
        verts_all, uv_all, faces_all, fm_all, hits, nrm_all = [], [], [], [], [], []
        skin_name = None
        for layer, (tint_key, drop) in enumerate((("tint2", sh["step"]), ("tint", 0.0))):          # the lighter step first
            nv = 4
            V = np.empty((nu, nv, 3))
            N = np.empty((nu, nv, 3))
            for j in range(nv):
                t = j / (nv - 1)
                zz = z_top + (z_low - drop - z_top) * t
                V[:, j], N[:, j] = self.skin_point_at_z(ths, zz)
            V = V.reshape(-1, 3)
            Nn = N.reshape(-1, 3)
            P, uv, mat, hit = self.fit.on_face(V)
            skin_name = skin_name or mat
            hits.append(hit)
            nrm_all.append(self.fit.face_normals(hit, Nn))
            P = P + Nn * (sh["off"] + 0.0001 * layer)
            off = sum(len(v) for v in verts_all)
            verts_all.append(P)
            uv_all.append(np.zeros((len(P), 2)) if uv is None else uv)
            faces_all += [[off + i * nv + j, off + (i + 1) * nv + j, off + (i + 1) * nv + j + 1, off + i * nv + j + 1]
                          for i in range(nu - 1) for j in range(nv - 1)]
            fm_all += [layer] * ((nu - 1) * (nv - 1))
        base = head_mats.get(skin_name or "")
        mats = []
        for layer, key in enumerate(("tint2", "tint")):
            k = np.asarray(sh[key], float)
            if base is not None:
                d, a = np.asarray(base.diffuse[:3], float), np.asarray(base.ambient, float)
                mats.append(replace(base, name=("髪影" if layer else "髪影淡"), name_en="fringe shadow" if layer else "fringe shadow (light)",
                                    diffuse=tuple(float(x) for x in np.clip(d * k, 0, 1)) + (1.0,),
                                    ambient=tuple(float(x) for x in np.clip(a * k, 0, 1)), edge=False, drop_shadow=False,
                                    self_shadow=False, self_shadow_map=False, alpha_blend=False,
                                    comment="soft shadow of the fringe on the forehead (the skin material, darker; generated)"))
            else:                                                    # no face mesh (tests): a flat warm shade of the skin colour
                sk = hex_rgb(ctx.get("colors.skin.base", "#f1e7d6"))
                c3 = np.clip(sk * k, 0, 1)
                mats.append(Material("髪影" if layer else "髪影淡", diffuse=(*(0.62 * c3), 1.0), ambient=tuple(0.38 * c3), toon="",
                                     edge=False, drop_shadow=False, self_shadow=False, self_shadow_map=False))
        acc = MeshAccum("hair_shadow", [m.name for m in mats])
        V = np.concatenate(verts_all, 0)
        acc.add(V, faces_all, np.concatenate(uv_all, 0), fm_all, weights={"頭": np.ones(len(V))}, normals=np.concatenate(nrm_all, 0))
        mesh = acc.to_mesh()
        if hits and hits[0] is not None:                          # ride the skin's morphs (brows up, troubled ...): no floating, no sinking
            hit = tuple(np.concatenate([h[k] for h in hits]) for k in range(4))
            mesh.morphs = self.fit.face_morphs(hit)
        return mesh, mats

    def side(self, acc):
        c, fit = self.cfg["side"], self.fit
        z_end = float(fit.chin[2]) + c["chin_gap"]
        z_leave = c.get("z_leave", float(fit.eye_z()) + 0.035)
        n = 0
        for sgn, tag in ((1.0, "左"), (-1.0, "右")):
            # a thin front lock, the main lock and a rear lock; lengths staggered, tips tucked in towards the jaw
            spec = [(56.0, -0.008, 0.034, 0.085, 0), (66.0, 0.008, 0.054, 0.097, 1), (77.0, 0.028, 0.048, 0.109, 2)]
            leaders = []
            for i, (th_d, extra, w, x_t, layer) in enumerate(spec):
                th0 = sgn * np.radians(th_d)
                ph_h = float(fit.cap_phi(th0))
                ph_leave = self.phi_at_z(th0, z_leave) or np.radians(95.0)
                P = meridian_path(self.skull, self.vol, th0, 0.5 * min(ph_h, np.radians(70)), ph_leave,
                                  dth=sgn * np.radians(1.5), delta0=-0.013, delta1=c["delta"] + 0.002 * i + self.rng.uniform(-0.003, 0.005),
                                  z_end=z_end + extra, x_end=sgn * (x_t + self.rng.uniform(-0.004, 0.004)), end_dir=[-sgn * 0.22, 0.0, -1.0], n_att=24)
                P = self.sway(P, c["sway"] * (1.0 if (i + (sgn > 0)) % 2 else -1.0) * self.rng.uniform(0.7, 1.1), z_leave, z_end + extra)
                leaders.append(self.off_body(Lock(f"side{tag}{i}", P, w, c["thick"], tip="point", tip_start=0.50,
                                                  tip_power=1.25, root_w=0.6, curv=9.0, layer=2 + layer, theta=th0,
                                                  dark=(i == 2))))
            main = leaders[1]
            ch = self.make_chain(f"横髪{tag}1_", main, "hair", 4, self.s_at_z(main.path, z_leave), 0.010)
            self.chains[f"side{tag}"] = ch
            for lk in leaders:
                lk.chain = f"side{tag}"
                self.emit(acc, lk, ch)
                n += 1
        self.stats["side"] = n
        self.stats["side_bottom_z"] = z_end

    def back(self, acc):
        """Two layers of big rounded clumps ending in sharp points of varied length: the inner layer (a shade darker)
        hangs lower, the outer layer overlaps it, so the nape edge is a layered, irregular row of pointed locks."""
        c, fit = self.cfg["back"], self.fit
        n = c["count"]
        z0 = float(fit.chin[2]) + c["end_above_chin"]
        locks = []
        for layer in (0, 1):
            m = n - layer
            ths = np.pi + np.linspace(-1, 1, m) * np.radians(66.0 if layer else 74.0)
            for i, th0 in enumerate(ths):
                z_end = z0 - (0.020 if layer == 0 else 0.0) + 0.007 * np.sin(2.3 * th0 + layer) + 0.004 * np.sin(5.1 * th0 + 2.0) \
                    + self.rng.uniform(-0.003, 0.003)                         # a smooth, layered nape line, not a saw
                P = meridian_path(self.skull, self.vol, th0, np.radians(30.0 if layer else 24.0), np.radians(125.0),
                                  dth=(th0 - np.pi) * 0.04, delta0=-0.013,
                                  delta1=c["delta"] + 0.006 * layer + self.rng.uniform(-0.003, 0.007), z_end=z_end,
                                  tuck=0.25, n_att=30)
                P = self.sway(P, c["sway"] * (1.0 if (i + layer) % 2 else -1.0) * self.rng.uniform(0.6, 1.1),
                              float(fit.center[2]) - 0.02, z_end)
                w = c["width"] * self.rng.uniform(0.9, 1.25)
                locks.append(self.off_body(Lock(f"back{layer}_{i}", P, w, c["thick"], tip="point",
                                                tip_start=self.rng.uniform(0.55, 0.68), tip_power=1.15, root_w=0.5,
                                                curv=7.0, layer=4 + layer, theta=th0, dark=(layer == 0))))
        outer = [l for l in locks if l.layer == 5]
        inner = [l for l in locks if l.layer == 4]
        ec = fit.ear_centre("L")
        z_root = (float(ec[2]) if ec is not None else fit.eye_z()) + c.get("root_above_ear", 0.03)
        for i, lk in enumerate(outer):
            ch = self.make_chain(f"後髪{i + 1}_", lk, "hair", 4, self.s_at_z(lk.path, z_root), 0.013)
            self.chains[f"back{i}"] = ch
            lk.chain = f"back{i}"
        for lk in inner:
            lk.chain = f"back{int(np.argmin([abs(lk.theta - o.theta) for o in outer]))}"
        for lk in inner + outer:
            self.emit(acc, lk, self.chains[lk.chain])
        self.stats["back"] = len(locks)
        self.stats["back_bottom_z"] = z0


def build_head_hair(ctx, fit, cfg, rig, pal, layers=("cap", "crown", "bangs", "side", "back")):
    """Head hair -> Piece (mesh `hair_head` with material 髪, the forehead shadow decal `hair_shadow` with 髪影) and
    bones/bodies/joints in `rig`."""
    hh = HeadHair(ctx, fit, cfg, rig, pal)
    acc = MeshAccum("hair_head", [hh.mat])
    for name in layers:
        getattr(hh, name)(acc)
    atlas = TEX.make_atlas(pal, ctx.rng_for("hair.atlas"), ring_f=hh.ring_f, ring_w=hh.ring_w,
                           ring=bool(hh.cfg["ring"]["enabled"]))
    tex = ctx.save_png("atlas", atlas)
    tn = hh.cfg["toon"]
    toon = ctx.save_png("toon_warm", TEX.toon_ramp(cfg.get("toon_shadow", TEX.toon_multiplier(ctx.get("colors.hair"))),
                                                   edge=tn["edge"], soft=tn["soft"]))
    mat = Material(hh.mat, name_en="hair", diffuse=(0.9, 0.9, 0.9, 1.0), specular=(0.0, 0.0, 0.0), shininess=0.0,
                   ambient=(0.5, 0.5, 0.5), texture=tex, toon=toon, edge=True,
                   edge_color=(0.30, 0.05, 0.07, 1.0), edge_size=0.8)
    meshes, mats = [acc.to_mesh()], [mat]
    if hh.cfg["bangs"]["shadow"]["enabled"] and "bangs" in layers:
        sm, smats = hh.forehead_shadow(ctx)
        meshes.append(sm)
        mats += smats
    info = dict(hh.stats)
    info["chains"] = {k: list(v.names) for k, v in hh.chains.items()}
    info["volume"] = {"top": hh.cfg["volume"]["top"], "side": hh.cfg["volume"]["side"], "lift": hh.cfg["volume"]["lift"]}
    info["top_z"] = hh.top_z
    info["ring_z"] = hh.z_top - hh.cfg["ring"]["drop"]
    return Piece(meshes=meshes, materials=mats, info=info)
