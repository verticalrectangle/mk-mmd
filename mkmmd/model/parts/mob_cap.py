"""A mob cap (part `mob_cap`, numpy only): a soft puffed crown of cloth gathered into a band round the head, a fine
pleated frill under the band, a crescent brooch pinned on the puff and a small satin bow on the band.

The band's lower edge is a tilted ring round her head: `edge` metres above the middle of her eyes (their eyeballs'
centres, which no lid edit moves) at the front and `drop` lower at the back (a plane through those two heights, tilted
about her left-right axis). The band is `band` high (along the plane's normal) and `thick` thick, hugs what is under it
`gap` off and is soft: its outer side bulges `bulge` between its edges, its lower edge is rolled under and its upper
edge turns in to where the crown is sewn on. At every height its section is the convex hull of the section of the
head's skin and of every part built before the cap but the body (the hair), and up the band it bridges hollows the same
way, as a taut ribbon spans the dips between locks. Under it hangs the frill (`[mob_cap.frill]`): `depth` below the
band, `pleats` soft pleats `amp` deep at its hem, the hem standing out `flare`, every pleat kept `margin` off the hair
(or skin) under it, measured as the nearest distance.

The crown (`[mob_cap.crown]`) is gathered into the top of the band and bulges out and up from it like a muffin top: its
widest line `rim` above the band's top, its highest point `top` above the top of the skull, its top flattened by `flat`
(2: an elliptic profile; more: flatter on top and rounder at the shoulder). It slouches toward `lean_toward` (degrees
round from the front toward her left: 135 is her back-left): it leans `lean` degrees that way, bending more toward its
top, and sags there: it bulges `overhang` past the band on the opposite side and `slump` more on that one, and its roll
hangs `hang` under the band's top edge on the opposite side and `droop` more on that one (in between they follow the
cosine of the angle), so the roll meets the band with no strip of the band's top showing over it. These are sized on
its plain shape meridian by meridian and measured again with its folds on. `pleats` soft pleats `pleat_depth` deep
gather it into the band and fade out `reach` of the way up (under the roll they fold in and out, not up and down, so
the roll's lower edge and the shadow it throws on the band stay smooth), and a seeded unevenness of at most `wobble`
metres keeps it from reading as a lathe shape. Where the hair under it reaches higher than the crown allows, the crown
is pushed out to stay `gap` off it.

The crescent (`[mob_cap.moon]`) is a gold plate `thick` thick with its edges rounded `round`: two circular arcs, the
outer `size` across, the inner cutting it so that its middle is `width` wide, both horns ending in rounded tips. It is
pinned on the outward face of the puff `at` degrees round from the front toward her left, its centre `lift` above the
lowest point of the puff's roll there, its horns pointing toward her left (+x: seen from her front it opens to the
right of the picture, its back toward the middle of the cap) and turned `tilt` degrees up. It is bent only as much as
the cloth under it curves (round the cap and up it, never twisted) and lies `proud` off it; the crown's folds and
unevenness are smoothed out under it. The bow (`[mob_cap.bow]`) is a satin bow on the band at `at` degrees (negative:
toward her right), under the crown's roll: `span` wide, two full rounded loops `loop` = [high, thick] (as long as the
span leaves beside the knot) with a soft fold (`fold`, a share of their thickness) down their middle, a knot `knot`
across, and two tails `tails` long and `tail_width` wide that lie on the band and over the frill, `splay` degrees
apart from straight down.

Everything rides `bone` (頭) with no physics of its own. Colours `[colors.mob_cap]`: cloth, cloth_shadow, cloth_light
(the crown), trim, trim_shadow (band and frill), line (the line round the band, `line` wide, its middle `line_at`
above the band's lower edge, so it shows under the crown's overhang; above it the band is painted in the shade the
crown throws), gold, gold_shadow, gold_light (the crescent), bow, bow_shadow; each material's toon shade is its shadow
colour over its base colour and its outline a darker shade of the same hue (`*_ink` sets one). Materials 帽子 (the
crown), 帽子バンド, 帽子フリル, 帽子月, 帽子リボン. `info` holds the cap's measures (`measure`): the band line, the
frill's depth, the crown's top, overhang, droop and lean, where the crescent and the bow sit and how far they stand off
the cloth, the smallest gaps to what is under the cap; landmarks cap_top, cap_moon, cap_bow."""
import colorsys

import numpy as np

from .. import spec as SP
from ..build import builder
from ..part import Part
from . import outfit_fit as OF
from . import outfit_geo as G
from . import outfit_wear as WR
from .hair_braids_bow import tail_block
from .hair_braids_plait import ellipsoid_shell
from .hair_fit import HeadFit
from .hair_geo import resample, smooth_polyline

DEFAULTS = {
    "edge": 0.065, "drop": 0.020, "band": 0.024, "gap": 0.002, "thick": 0.0025, "bulge": 0.0015, "line": 0.005,
    "line_at": 0.008,
    "frill": {"depth": 0.016, "pleats": 44, "amp": 0.002, "flare": 0.005, "margin": 0.002},
    "crown": {"overhang": 0.025, "slump": 0.008, "hang": 0.003, "droop": 0.006, "rim": 0.014, "top": 0.060, "flat": 2.4,
              "pleats": 17, "pleat_depth": 0.004, "reach": 1.0 / 3.0, "lean": 12.0, "lean_toward": 135.0,
              "wobble": 0.003},
    "moon": {"at": 25.0, "lift": 0.022, "size": 0.046, "width": 0.0135, "thick": 0.006, "round": 0.0015,
             "tilt": 20.0, "proud": 0.0015},
    "bow": {"at": -110.0, "span": 0.045, "loop": [0.014, 0.005], "fold": 0.3, "knot": 0.008,
            "tails": 0.030, "tail_width": 0.008, "splay": 40.0},
    "around": 144, "rings": 56, "bone": "頭",
}
COLORS = {"cloth": "#ead0ee", "cloth_shadow": "#b99ccb", "cloth_light": "#f7e6f8", "trim": "#fbf7fd",
          "trim_shadow": "#d6cde2", "line": "#a886d6", "gold": "#f2c84b", "gold_shadow": "#c9952a",
          "gold_light": "#fff0a8", "bow": "#f2c84b", "bow_shadow": "#c9952a",
          "cloth_ink": None, "trim_ink": None, "gold_ink": None, "bow_ink": None}
MATS = ("帽子", "帽子バンド", "帽子フリル", "帽子月", "帽子リボン")
CROWN, BAND, FRILL, MOON, BOW = range(5)
STEP = 0.0015                   # metres between the sections the band is fitted on
INNER = 0.006                   # how far up inside the band its rolled lip runs back
PLEAT_IN = 0.04                 # the share of a meridian over which the crown's pleats grow out of the seam
UP = np.array([0.0, 0.0, 1.0])


# ---------------------------------------------------------------- what the cap is fitted over
def under(parts, z_min):
    """The surface the cap is fitted over (outfit_fit.Skin): the head's closed skin shell and the meshes of every part
    built before the cap but the body and the head (the hair), cut to the triangles that reach above `z_min`."""
    skin = parts["head"].info["skin"]
    pieces = [(np.asarray(skin["verts"], float), np.asarray(skin["faces"], int).reshape(-1, 3))]
    for name, p in parts.items():
        if name in ("body", "head"):
            continue
        for m in p.meshes:
            if len(m.verts) and len(m.faces):
                pieces.append((np.asarray(m.verts, float), G.tri_arrays(list(m.faces))[0]))
    V, T, off = [], [], 0
    for v, t in pieces:
        V.append(v)
        T.append(t[v[t][:, :, 2].max(1) > z_min] + off)
        off += len(v)
    return OF.Skin(np.vstack(V), np.vstack(T), name="under")


def convex_hull(P):
    """Indices of the convex hull of 2-D points P (n, 2), counter-clockwise (Andrew's monotone chain)."""
    order = np.lexsort((P[:, 1], P[:, 0]))

    def half(idx):
        h = []
        for i in idx:
            while len(h) >= 2:
                a, b = P[h[-2]], P[h[-1]]
                if (b[0] - a[0]) * (P[i][1] - a[1]) - (b[1] - a[1]) * (P[i][0] - a[0]) > 0.0:
                    break
                h.pop()
            h.append(int(i))
        return h
    lo, hi = half(order), half(order[::-1])
    return np.array(lo[:-1] + hi[:-1])


def hull_radii(skin, centre, a, e0, phi, rmax=0.3):
    """Radii (len(phi),) of the convex hull of the section of `skin` by the plane through `centre` with unit normal `a`,
    along the directions cos(phi) e0 + sin(phi) (a x e0) from `centre`: a taut line round that section."""
    e1 = np.cross(a, e0)
    seg = skin.slice(centre, a).reshape(-1, 3) - centre
    P = np.stack([seg @ e0, seg @ e1], 1)
    P = P[np.hypot(P[:, 0], P[:, 1]) < rmax]
    if len(P) < 3:
        raise ValueError(f"mob_cap: nothing under the cap at {np.round(centre, 3).tolist()}")
    P = np.unique(np.round(P, 6), axis=0)
    H = P[convex_hull(P)]
    E = np.roll(H, -1, 0) - H
    D = np.stack([np.cos(phi), np.sin(phi)], 1)
    den = D[:, :1] * E[None, :, 1] - D[:, 1:] * E[None, :, 0]
    ok = np.abs(den) > 1e-14
    den = np.where(ok, den, 1.0)
    t = (H[None, :, 0] * E[None, :, 1] - H[None, :, 1] * E[None, :, 0]) / den
    s = (H[None, :, 0] * D[:, 1:] - H[None, :, 1] * D[:, :1]) / den
    r = np.where(ok & (s >= -1e-9) & (s <= 1.0 + 1e-9) & (t > 0.0), t, -np.inf).max(1)
    if not np.isfinite(r).all():
        raise ValueError("mob_cap: the head's axis lies outside what the cap is fitted over")
    return r


def concave_cover(h, r):
    """The least concave function over r(h) (h increasing) at the h samples: a taut line over the dips of r."""
    keep = []
    for i in range(len(h)):
        while len(keep) >= 2:
            i0, i1 = keep[-2], keep[-1]
            if (r[i1] - r[i0]) * (h[i] - h[i0]) >= (r[i] - r[i0]) * (h[i1] - h[i0]):
                break
            keep.pop()
        keep.append(i)
    return np.interp(h, h[keep], r[keep])


def periodic_interp(x, xp, fp):
    """fp (..., n) given at the n angles xp (evenly spaced, increasing over one turn) at the angles x."""
    n = len(xp)
    k = ((np.asarray(x, float) - xp[0]) % (2.0 * np.pi)) / (2.0 * np.pi / n)
    i0 = np.floor(k).astype(int) % n
    f = k - np.floor(k)
    return fp[..., i0] * (1.0 - f) + fp[..., (i0 + 1) % n] * f


# ---------------------------------------------------------------- the band
class Band:
    """The band's frame and the taut section of what is under it at every height: `centre` and `a` (the unit normal of
    the plane of its lower edge, leaning toward her back), `e0` (toward her front in that plane), `e1` (toward her
    left); `phi` the angles of its columns (from `e0` toward `e1`, starting at her back); `radius(h)` the hull's radius
    of every column at the height `h` over the plane, `ring(h, extra)` the ring that far out of it."""

    def __init__(self, skin, centre, z_front, drop, band, off, m, below):
        tau = np.arctan2(drop, 0.19)
        c = np.array([centre[0], centre[1], z_front - 0.5 * drop])
        for _ in range(5):                          # the plane through her front at z_front and her back drop lower
            a = np.array([0.0, np.sin(tau), np.cos(tau)])
            e0 = np.array([0.0, -np.cos(tau), np.sin(tau)])
            r = hull_radii(skin, c, a, e0, np.array([0.0, np.pi]))
            F, B = c + e0 * (r[0] + off), c - e0 * (r[1] + off)
            tau = np.arctan2(drop, B[1] - F[1])
            c = np.array([c[0], 0.5 * (F[1] + B[1]), z_front - 0.5 * drop])
        self.a = np.array([0.0, np.sin(tau), np.cos(tau)])
        self.e0 = np.array([0.0, -np.cos(tau), np.sin(tau)])
        self.e1 = np.cross(self.a, self.e0)
        self.centre = c
        self.phi = np.pi + 2.0 * np.pi * np.arange(m) / m
        self.levels = np.concatenate([np.arange(-below, 0.0, STEP), np.arange(0.0, band + 2.0 * STEP, STEP)])
        R = np.stack([hull_radii(skin, c + h * self.a, self.a, self.e0, self.phi) for h in self.levels])
        up = self.levels >= 0.0
        for j in range(m):                          # taut up the band: no dip between two heights
            R[up, j] = concave_cover(self.levels[up], R[up, j])
        self.R = R

    def dirs(self, phi=None):
        phi = self.phi if phi is None else np.asarray(phi, float)
        return np.cos(phi)[:, None] * self.e0 + np.sin(phi)[:, None] * self.e1

    def radius(self, h, phi=None):
        r = np.array([np.interp(h, self.levels, self.R[:, j]) for j in range(len(self.phi))])
        return r if phi is None else periodic_interp(phi, self.phi, r)

    def ring(self, h, extra, phi=None):
        return self.centre + h * self.a + self.dirs(phi) * (self.radius(h, phi) + extra)[:, None]

    def at(self, theta):
        """(column, share toward the next) where the azimuth `theta` (degrees round from her front toward her left)
        falls between the band's columns."""
        th = np.radians(theta)
        phi = np.arctan2(np.sin(th) * np.cos(np.arctan2(self.a[1], self.a[2])), np.cos(th))
        k = ((phi - self.phi[0]) % (2.0 * np.pi)) / (2.0 * np.pi / len(self.phi))
        return int(np.floor(k)) % len(self.phi), float(k - np.floor(k))


def ring_u(ring):
    """Texture u (len + 1,) round a closed ring by arc length: 0 at its first point, 1 back at it."""
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(np.vstack([ring, ring[:1]]), axis=0), axis=1))])
    return s / s[-1]


def band_rings(band, c):
    """The band as one strip of rings, from its inner side down round its rolled lip and up its soft outer side (bulging
    `bulge` between its edges) to the seam: (rings (K, M, 3), v (K,): 0 at the inner side's top, `v_lip` where the
    outer side starts, 1 at the seam; v_lip; the levels of the outer side's rings)."""
    gap, thick, h = c["gap"], c["thick"], c["band"]
    half = 0.5 * thick
    r_lip = np.maximum(band.radius(0.0), band.radius(half)) + gap + half
    inner = [band.ring(z, gap) for z in np.linspace(INNER, half, 4)]
    inner[-1] = band.centre + half * band.a + band.dirs() * (r_lip - half)[:, None]
    lip = [band.centre + half * (1.0 - np.sin(b)) * band.a + band.dirs() * (r_lip + half * np.cos(b))[:, None]
           for b in np.linspace(np.pi, 0.0, 7)[1:-1]]
    levels = np.linspace(half, h, 15)
    soft = c["bulge"] * np.sin(np.pi * (levels - half) / (h - half)) ** 0.75
    outer = [band.ring(z, gap + thick + s) for z, s in zip(levels, soft)]
    outer[0] = band.centre + half * band.a + band.dirs() * (r_lip + half)[:, None]
    rings = np.array(inner + lip + outer)
    v_lip = 0.12
    n_in = len(inner) + len(lip)
    v = np.concatenate([np.linspace(0.0, v_lip, n_in + 1)[:-1], v_lip + (1.0 - v_lip) * (levels - half) / (h - half)])
    return rings, v, v_lip, levels


# ---------------------------------------------------------------- the crown
def wave(u, n, seed, wobble=0.5):
    """Soft pleats round a ring: (sin of `n` turns, pleat depth factor) at the shares `u` of the way round, the pleats a
    little uneven in spacing and depth (seeded)."""
    t = 2.0 * np.pi * np.asarray(u, float)
    return np.sin(n * t + wobble * G.periodic_noise(t, seed, 4)), 1.0 + 0.25 * G.periodic_noise(t, seed + 1, 4)


UNDER = np.linspace(0.0, 0.5 * np.pi, 33)       # the parameter of a meridian's underside, seam to widest line


def sag(rim, droop):
    """How much (M,) of the bump sin(2a)^1.5 to take off each meridian's underside rim (1 - cos a) so that its lowest
    point hangs `droop` (M,) under the seam."""
    base, bump = rim * (1.0 - np.cos(UNDER)), np.sin(2.0 * UNDER) ** 1.5
    droop = np.maximum(np.asarray(droop, float), 0.0)
    lo, hi = np.zeros(len(droop)), np.full(len(droop), 10.0 * (rim + droop.max()) + 1e-3)
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        deep = (base[None] - mid[:, None] * bump[None]).min(1) < -droop
        lo, hi = np.where(deep, lo, mid), np.where(deep, mid, hi)
    return lo


def crown_grid(seam, centre, D, droop, z_top, cc, u, seeds, rings, detail=True, calm=None):
    """The crown's vertices (K, M, 3) from the seam ring (M, 3) up to its apex, and the share t (K, M) of the way up
    each meridian (0 at the seam, 1 at the apex). Each meridian goes out under its roll (hanging `droop` (M,) under the
    seam), round to its widest line `D` (M,) out from the seam and `rim` above it, and over the flattened top to the
    axis (vertical through `centre`) at `z_top`; with `detail` the pleats and the seeded unevenness are added, smoothed
    out by the share `calm` (K, M) (under the crescent, which holds the cloth flat); the whole leans toward
    `lean_toward`, bending more toward the top."""
    d = (seam - centre) * [1.0, 1.0, 0.0]
    R = np.linalg.norm(d, axis=1)
    o = d / R[:, None]
    zs, rim = seam[:, 2], cc["rim"]
    b = np.linspace(0.0, 0.5 * np.pi, 81)[1:]
    p = float(cc["flat"])
    rho = np.concatenate([R[:, None] + D[:, None] * np.sin(UNDER)[None],
                          (R + D)[:, None] * np.cos(b)[None] ** (2.0 / p)], 1)
    z = np.concatenate([zs[:, None] + rim * (1.0 - np.cos(UNDER))[None]
                        - sag(rim, droop)[:, None] * np.sin(2.0 * UNDER)[None] ** 1.5,
                        (zs + rim)[:, None] + (z_top - zs - rim)[:, None] * np.sin(b)[None]], 1)
    X = centre * [1.0, 1.0, 0.0] + o[:, None, :] * rho[..., None] + UP * z[..., None]          # (M, n, 3)
    K = int(rings)
    out, tt = np.empty((K, len(X), 3)), np.empty((K, len(X)))
    q = np.linspace(0.0, 1.0, K)
    for j in range(len(X)):
        s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(X[j], axis=0), axis=1))])
        out[:, j] = resample(X[j], K, q * s[-1])
        tt[:, j] = q
    if detail:
        N = grid_normals(out)
        pw, pdepth = wave(u, int(cc["pleats"]), seeds[0])
        reach = float(cc["reach"])
        env = G.smoothstep(tt / PLEAT_IN) * (1.0 - G.smoothstep((tt - PLEAT_IN) / max(reach - PLEAT_IN, 1e-6)))
        pleat = cc["pleat_depth"] * env * (pw * pdepth)[None]
        n1 = G.periodic_noise(2.0 * np.pi * u, seeds[1], 3)
        n2 = G.periodic_noise(2.0 * np.pi * u, seeds[2], 4)
        noise = 0.6 * n1[None] * np.sin(np.pi * tt) ** 0.8 + 0.4 * n2[None] * np.sin(2.0 * np.pi * tt)
        noise = cc["wobble"] * noise / max(float(np.abs(noise).max()), 1e-9)
        keep = 1.0 if calm is None else 1.0 - calm
        side = G.unit(out - centre * [1.0, 1.0, 0.0] - UP * out[..., 2:3]) * [1.0, 1.0, 0.0]
        under = (1.0 - G.smoothstep((tt - 0.10) / 0.10))[..., None]      # under the roll the pleats fold in and out,
        way = G.unit(N + (side - N) * under)                               # not up and down: its lower edge stays
        out = out + ((pleat + noise) * keep)[..., None] * way              # smooth over the band
    toward = G.hdir(np.radians(cc["lean_toward"]))
    return bend(out, centre, np.cross(UP, toward), np.radians(cc["lean"]) * G.smoothstep(tt)), tt


def grid_normals(P):
    """Unit normals (K, M, 3) of a grid of closed rings (K, M, 3) listed from the bottom up, facing out; the last ring
    (an apex) faces up."""
    dj = np.roll(P, -1, 1) - np.roll(P, 1, 1)
    dk = np.gradient(P, axis=0)
    N = G.unit(np.cross(dj, dk))
    N[-1] = UP
    return N


def bend(P, pivot, axis, angle):
    """Points P (..., 3) turned about the line through `pivot` along the unit `axis` by `angle` (...) radians each."""
    k = G.unit(axis)
    Q = P - pivot
    ca, sa = np.cos(angle)[..., None], np.sin(angle)[..., None]
    return pivot + Q * ca + np.cross(k, Q) * sa + k * (Q @ k)[..., None] * (1.0 - ca)


def crown_sizes(grid, seam, centre, reach):
    """Per meridian (M,): how far the crown bulges out past the band (horizontally, from the crown's axis) and how far
    its roll hangs under the seam."""
    o = G.unit((seam - centre) * [1.0, 1.0, 0.0])
    over = np.einsum("kmj,mj->km", (grid - centre) * [1.0, 1.0, 0.0], o).max(0) - reach
    return over, seam[:, 2] - grid[..., 2].min(0)


def push_clear(P, t, surf, gap):
    """The crown's grid (K, M, 3) moved out along its normals where it comes nearer than `gap` to `surf` (the hair
    under it), the move spread smoothly; returns (P, largest move)."""
    move = np.zeros(P.shape[:2])
    for _ in range(3):
        d = np.abs(surf.closest(P.reshape(-1, 3))[0]).reshape(P.shape[:2])
        need = np.maximum(gap - d, 0.0) * (t > 0.02)
        if need.max() < 1e-5:
            break
        for _ in range(3):                              # spread it: a dent of cloth, not a spike
            need = np.maximum(need, 0.5 * (np.roll(need, 1, 1) + np.roll(need, -1, 1)))
            need[1:-1] = np.maximum(need[1:-1], 0.5 * (need[:-2] + need[2:]))
        P = P + (need * 1.2)[..., None] * grid_normals(P)
        move += need * 1.2
    return P, float(move.max())


# ---------------------------------------------------------------- the crescent
def slab(a, b, r, nf=3, nc=4, ns=2):
    """Points (4 (nf + nc + ns), 2) round the section of a plate, half `a` wide and half `b` thick with its edges
    rounded `r`, as (across, up through it): counter-clockwise from the middle of its +across side, its front's middle
    a quarter of the way round and its back's middle three quarters."""
    r = min(r, a, b)
    q = [(a, (b - r) * k / ns) for k in range(ns)]
    q += [(a - r + r * np.cos(g), b - r + r * np.sin(g)) for g in 0.5 * np.pi * np.arange(nc) / nc]
    q += [((a - r) * (1.0 - k / nf), b) for k in range(nf)] + [(0.0, b)]
    q = np.array(q)
    back = q[::-1]
    return np.vstack([q[:-1], (back * [-1.0, 1.0])[:-1], (q * [-1.0, -1.0])[:-1], (back * [1.0, -1.0])[:-1]])


def crescent(c, n_along=40, n_tip=4):
    """The crescent in its own frame: (verts (n, 3) as (x toward its open side, y along it, h off its back), faces,
    uv (n, 2) per vertex, outline (k, 2) in (x, y)). Two circular arcs: the outer `size` across, the inner of 0.82 of
    its radius cutting it so that its middle is `width` wide; the horns are cut where they are 4 mm wide and end in a
    round; the section is a plate `thick` thick with its edges rounded `round`."""
    Ro, w0, T = 0.5 * c["size"], float(c["width"]), float(c["thick"])
    Ri = 0.82 * Ro
    dlt = Ri - Ro + w0                                  # the inner circle's centre, toward the open side

    def spoke(al):
        d = np.stack([-np.cos(al), np.sin(al)], -1)
        tin = dlt * d[..., 0] + np.sqrt(np.maximum(dlt ** 2 * d[..., 0] ** 2 - dlt ** 2 + Ri ** 2, 0.0))
        return d, tin
    xt = (Ro ** 2 - Ri ** 2 + dlt ** 2) / (2.0 * dlt)
    grid = np.linspace(0.0, np.pi - np.arctan2(np.sqrt(max(Ro ** 2 - xt ** 2, 0.0)), xt), 600)
    w_end = min(0.004, 0.5 * w0)
    al_end = float(np.interp(-w_end, -(Ro - spoke(grid)[1]), grid))
    al = al_end * np.sin(0.5 * np.pi * np.linspace(-1.0, 1.0, n_along))      # closer toward the tips
    d, tin = spoke(np.abs(al))
    d[:, 1] *= np.sign(al)
    mid = d * (0.5 * (Ro + tin))[:, None]
    wid = Ro - tin
    rows = []
    for i in range(n_along):
        sec = slab(0.5 * wid[i], 0.5 * T, c["round"])
        rows.append(np.column_stack([mid[i][None] + d[i][None] * sec[:, :1], 0.5 * T + sec[:, 1]]))
    n_round = len(rows[0])
    caps = []
    for i, j in ((0, 1), (n_along - 1, n_along - 2)):  # each horn closes in a round as wide as it is there
        tang = G.unit(mid[i] - mid[j])
        sec = slab(0.5 * wid[i], 0.5 * T, c["round"])
        ring = []
        for g in 0.5 * np.pi * np.arange(1, n_tip) / n_tip:
            P = mid[i][None] + tang[None] * 0.5 * wid[i] * np.sin(g) + d[i][None] * (sec[:, :1] * np.cos(g))
            ring.append(np.column_stack([P, 0.5 * T + sec[:, 1] * np.cos(g)]))
        caps.append((ring, np.concatenate([mid[i] + tang * 0.5 * wid[i], [0.5 * T]])))
    body = caps[0][0][::-1] + rows + caps[1][0]
    V = np.vstack(body + [caps[0][1][None], caps[1][1][None]])
    nr = len(body)
    a0, a1 = len(V) - 2, len(V) - 1
    faces = []
    for k in range(nr - 1):
        for j in range(n_round):
            j2 = (j + 1) % n_round
            faces.append([k * n_round + j, k * n_round + j2, (k + 1) * n_round + j2, (k + 1) * n_round + j])
    for j in range(n_round):
        j2 = (j + 1) % n_round
        faces.append([a0, j2, j])
        faces.append([a1, (nr - 1) * n_round + j, (nr - 1) * n_round + j2])
    faces = orient_closed(V, faces)
    u = np.concatenate([np.repeat(np.linspace(0.0, 1.0, nr), n_round), [0.0, 1.0]])
    v = np.concatenate([np.tile(np.arange(n_round) / n_round, nr), [0.5, 0.5]])
    g = np.linspace(0.0, np.pi, 9)[1:-1, None]           # its outline in order: out along the outer arc, round a tip,
    t0, t1 = G.unit(mid[0] - mid[1]), G.unit(mid[-1] - mid[-2])     # back along the inner one, round the other
    outline = np.vstack([mid + d * (0.5 * wid)[:, None],
                         mid[-1] + 0.5 * wid[-1] * (np.cos(g) * d[-1] + np.sin(g) * t1),
                         (mid - d * (0.5 * wid)[:, None])[::-1],
                         mid[0] + 0.5 * wid[0] * (-np.cos(g) * d[0] + np.sin(g) * t0)])
    return V, faces, np.stack([u, v], 1), outline


def orient_closed(V, faces):
    """Faces of a closed shell turned to face out (by its signed volume)."""
    V = np.asarray(V, float)
    vol = sum(float(V[f[0]] @ np.cross(V[f[i]], V[f[i + 1]])) for f in faces for i in range(1, len(f) - 1))
    return [list(f) for f in faces] if vol > 0.0 else [list(f)[::-1] for f in faces]


def first_hits(origins, d, verts, tris, chunk=128):
    """(distance, triangle) along the unit direction `d` from each of `origins` (n, 3) to the first of the triangles
    (verts, tris) it meets (inf and -1 where none)."""
    v0 = verts[tris[:, 0]]
    e1, e2 = verts[tris[:, 1]] - v0, verts[tris[:, 2]] - v0
    p = np.cross(d, e2)
    det = np.einsum("tj,tj->t", e1, p)
    ok = np.abs(det) > 1e-14
    inv = np.where(ok, 1.0 / np.where(ok, det, 1.0), 0.0)
    out, idx = np.full(len(origins), np.inf), np.full(len(origins), -1)
    for s in range(0, len(origins), chunk):
        rel = origins[s:s + chunk, None, :] - v0[None]
        u = np.einsum("ctj,tj->ct", rel, p) * inv
        q = np.cross(rel, e1[None])
        v = (q @ d) * inv
        t = np.einsum("ctj,tj->ct", q, e2) * inv
        t = np.where(ok & (u >= 0.0) & (v >= 0.0) & (u + v <= 1.0) & (t > 1e-9), t, np.inf)
        k = t.argmin(1)
        out[s:s + chunk] = t[np.arange(len(t)), k]
        idx[s:s + chunk] = np.where(np.isfinite(out[s:s + chunk]), k, -1)
    return out, idx


def moon_frame(band, grid, theta, lift):
    """Where the crescent sits on the crown's outward face at `theta` degrees round from her front: (centre, ex toward
    her left, ey up the cloth, f out of it), `lift` above the lowest point of the crown's roll there, and that point's
    height."""
    j0, fr = band.at(theta)
    j1 = (j0 + 1) % grid.shape[1]
    P = grid[:, j0] * (1.0 - fr) + grid[:, j1] * fr
    N = grid_normals(grid)
    N = G.unit(N[:, j0] * (1.0 - fr) + N[:, j1] * fr)
    kb = int(np.argmin(P[:, 2]))
    zc = P[kb, 2] + lift
    k = kb + int(np.argmax(P[kb:, 2] >= zc))
    s = (zc - P[k - 1, 2]) / (P[k, 2] - P[k - 1, 2])
    C, f = P[k - 1] + s * (P[k] - P[k - 1]), G.unit(N[k - 1] + s * (N[k] - N[k - 1]))
    ey = G.unit(UP - (UP @ f) * f)
    return C, np.cross(ey, f), ey, f, float(P[kb, 2])


def moon_shape(mc):
    """The crescent laid out in its frame's plane: (verts (n, 3) in its own frame, faces, uv per vertex, xy (n, 2) in
    the frame's (ex, ey), outline (k, 2) there), turned `tilt` degrees and its extent centred on the frame's centre."""
    V, F, uv, outline = crescent(mc)
    tl = np.radians(mc["tilt"])
    rot = np.array([[np.cos(tl), -np.sin(tl)], [np.sin(tl), np.cos(tl)]])
    xy, ol = V[:, :2] @ rot.T, outline @ rot.T
    shift = -0.5 * (ol.min(0) + ol.max(0))
    return V, F, uv, xy + shift, ol + shift


def outside(poly, x, y):
    """How far the points (x, y) lie outside the convex polygon `poly` (k, 2), counter-clockwise (0 inside)."""
    P = np.stack([np.asarray(x, float), np.asarray(y, float)], -1).reshape(-1, 2)
    A, E = poly, np.roll(poly, -1, 0) - poly
    t = np.clip(np.einsum("pkj,kj->pk", P[:, None] - A[None], E) / np.einsum("kj,kj->k", E, E), 0.0, 1.0)
    d = np.linalg.norm(P[:, None] - (A[None] + t[..., None] * E[None]), axis=2).min(1)
    inside = (((P[:, None, 0] - A[None, :, 0]) * E[None, :, 1] - (P[:, None, 1] - A[None, :, 1]) * E[None, :, 0])
              <= 0.0).all(1)
    return np.where(inside, 0.0, d).reshape(np.shape(x))


def calm_under(grid, frame, outline, ease):
    """The share (K, M) of the crown's pleats and unevenness the crescent smooths out: all of it under its outline,
    none `ease` beyond it, none on the far side of the cap."""
    C, ex, ey, f = frame
    Q = grid - C
    poly = outline[convex_hull(outline)]
    return (1.0 - G.smoothstep(outside(poly, Q @ ex, Q @ ey) / ease)) * (np.abs(Q @ f) < 0.02)


def moon_surface(frame, outline, verts, tris, n=41, margin=0.002):
    """The surface the crescent lies on, as its height along the frame's `f` over its (ex, ey) plane: the cloth (the
    triangles `tris` facing the crescent, hit from the front) under the outline, eased into a smooth sheet that touches
    the cloth's highest points and bridges its hollows (a plate bent only as much as the cloth under it curves), carried
    straight on where there is no cloth to lie on (under the roll). Returns a function (x, y) -> height."""
    C, ex, ey, f = frame
    lo, hi = outline.min(0) - margin, outline.max(0) + margin
    gx, gy = np.linspace(lo[0], hi[0], n), np.linspace(lo[1], hi[1], n)
    X, Y = np.meshgrid(gx, gy)
    t, idx = first_hits(C + X.reshape(-1, 1) * ex + Y.reshape(-1, 1) * ey + 0.06 * f, -f, verts, tris)
    tn = G.unit(np.cross(verts[tris[:, 1]] - verts[tris[:, 0]], verts[tris[:, 2]] - verts[tris[:, 0]]))
    ok = (np.isfinite(t) & ((tn[np.maximum(idx, 0)] @ f) > 0.25)).reshape(n, n)
    d = (0.06 - t).reshape(n, n)
    if ok.sum() < 8:
        raise ValueError("mob_cap: no cloth under the crescent")
    sheet = np.where(ok, d, np.nan)
    for j in range(n):                                  # straight on up and down from the cloth there is
        if ok[:, j].any():
            sheet[:, j] = np.interp(gy, gy[ok[:, j]], d[ok[:, j], j])
    cols = ok.any(0)
    for i in range(n):
        sheet[i] = np.interp(gx, gx[cols], sheet[i, cols])
    for _ in range(60):                                 # eased, never under the cloth
        pad = np.pad(sheet, 1, mode="edge")
        sheet = (pad[:-2, 1:-1] + pad[2:, 1:-1] + pad[1:-1, :-2] + pad[1:-1, 2:] + 4.0 * pad[1:-1, 1:-1]) / 8.0
        sheet = np.where(ok, np.maximum(sheet, d), sheet)
    sheet = sheet + max(0.0, float(np.max(np.where(ok, d - sheet, 0.0))))

    def height(x, y):
        fx = np.clip((np.asarray(x) - lo[0]) / (hi[0] - lo[0]) * (n - 1), 0.0, n - 1.000001)
        fy = np.clip((np.asarray(y) - lo[1]) / (hi[1] - lo[1]) * (n - 1), 0.0, n - 1.000001)
        i, j = fy.astype(int), fx.astype(int)
        a, b = fy - i, fx - j
        return ((1 - a) * (1 - b) * sheet[i, j] + (1 - a) * b * sheet[i, j + 1] + a * (1 - b) * sheet[i + 1, j]
                + a * b * sheet[i + 1, j + 1])
    return height


def lay_moon(mc, frame, shape, verts, tris):
    """The crescent's vertices (n, 3) laid `proud` off the cloth (verts, tris) at `frame` (moon_shape's `shape`)."""
    V, _, _, xy, ol = shape
    C, ex, ey, f = frame
    cen = verts[tris].mean(1) - C
    near = (np.abs(cen @ ex) < 0.06) & (np.abs(cen @ ey) < 0.06) & (np.abs(cen @ f) < 0.04)
    height = moon_surface(frame, ol, verts, tris[near])
    return C + xy[:, :1] * ex + xy[:, 1:] * ey + (height(xy[:, 0], xy[:, 1]) + mc["proud"] + V[:, 2])[:, None] * f


# ---------------------------------------------------------------- the bow
def loop_shell(O, w, up, n, L, H, T, fold, curl, nx=16, nb=20):
    """One loop of a satin bow, a closed shell: from `O` (inside the knot) `L` along `w`, `H` high along `up` and `T`
    thick along `n` at its fullest, pinched where it leaves the knot and rounded at its far end; a soft fold (`fold` of
    its thickness) runs down the middle of its front, and its middle line curls back `curl` x^2 toward the band.
    Returns (verts, faces, uv per vertex)."""
    h, t = 0.5 * H, 0.5 * T
    x = np.linspace(0.0, 1.0, nx)
    xr = max(1.0 - h / L, 0.3)                          # where its round end starts
    s = np.clip((x - xr) / (1.0 - xr), 0.0, 1.0)
    half = h * np.where(x < xr, 0.42 + 0.58 * np.sin(0.5 * np.pi * np.minimum(x / xr, 1.0)), np.sqrt(1.0 - s ** 2))
    thin = t * np.sqrt(half / h)
    beta = 2.0 * np.pi * np.arange(nb) / nb
    cb, sb = np.cos(beta), np.sin(beta)
    groove = 1.0 - fold * np.exp(-(cb / 0.35) ** 2) * (sb > 0.0)
    line = O + np.outer(x * L, w) - np.outer(curl * (x * L) ** 2, n)
    rings = [line[i] + np.outer(half[i] * cb, up) + np.outer(thin[i] * sb * groove, n) for i in range(nx - 1)]
    V = np.vstack(rings + [line[0][None], line[-1][None]])
    a0, a1 = len(V) - 2, len(V) - 1
    faces = []
    for k in range(nx - 2):
        for j in range(nb):
            j2 = (j + 1) % nb
            faces.append([k * nb + j, k * nb + j2, (k + 1) * nb + j2, (k + 1) * nb + j])
    for j in range(nb):
        j2 = (j + 1) % nb
        faces.append([a0, j2, j])
        faces.append([a1, (nx - 2) * nb + j, (nx - 2) * nb + j2])
    uv = np.vstack([np.column_stack([np.full(nb, x[i]), beta / (2.0 * np.pi)]) for i in range(nx - 1)] +
                   [[[0.0, 0.5], [1.0, 0.5]]])
    return V, orient_closed(V, faces), uv


def lay(support, start, heading, length, lift, step=0.0015):
    """A ribbon's middle line from `start` heading along the surface `support` (outfit_fit.Skin): every step goes on
    along it and is put back `lift` off it. Returns (n, 3) points about `length` long and the surface's mean normal."""
    def put(p):
        _, q, tri, bary = support.closest(p[None])
        n = support.outward(tri, bary)[0]
        return q[0] + n * lift, n
    p, n = put(np.asarray(start, float))
    pts, normals, h, s = [p], [n], G.unit(heading - (heading @ n) * n), 0.0
    while s < length:
        p, n = put(pts[-1] + step * h)
        go = G.unit(p - pts[-1])
        h = G.unit(go - (go @ n) * n)
        s += float(np.linalg.norm(p - pts[-1]))
        pts.append(p)
        normals.append(n)
    P = smooth_polyline(np.array(pts), 2)
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))])
    n = max(round(length / 0.003), 6)
    return resample(P, n, np.linspace(0.0, min(length, s[-1]), n)), G.unit(np.mean(normals, 0))


def bow_pieces(bc, frame, radius, cloth, support):
    """The bow on the band at `frame` = (point on the band, up it, out of it, round it): a knot, two loops spanning
    round the band and curled to its `radius`, and two tails laid on `support` (band, frill and what is under them), all
    off `cloth`. Returns ([(verts, faces, uv per vertex)], centre, tail lengths, the direction across it)."""
    P0, up, n, _ = frame
    w = G.unit(np.cross(-up, n))
    H, T = bc["loop"]
    k = 0.5 * bc["knot"]
    kd = 0.55 * T
    C = P0 + n * (kd + 0.0008)
    pieces = []
    V, F, u, v = ellipsoid_shell(C, up, n, k, k, kd, n_around=12, n_rings=7, p=0.6)
    pieces.append((V, orient_closed(V, F), np.column_stack([u, v])))
    root = max(k - 0.0015, 0.0)
    length = 0.5 * bc["span"] - root
    for sgn in (1.0, -1.0):
        O = C + sgn * w * root - n * (kd - 0.5 * T)
        pieces.append(loop_shell(O, sgn * w, up, n, length, H, T, bc["fold"], 0.5 / radius))
    for _ in range(4):                                  # lying on the band, never in it
        near = cloth.closest(np.vstack([p[0] for p in pieces]))[0].min()
        if near >= 0.00075:
            break
        pieces = [(p[0] + n * (0.0008 - near), p[1], p[2]) for p in pieces]
        C = C + n * (0.0008 - near)
    tails = []
    for sgn in (1.0, -1.0):
        start = C - n * kd - up * k * 0.4 + sgn * w * k * 0.3
        go = G.unit(-up * np.cos(np.radians(bc["splay"])) + sgn * w * np.sin(np.radians(bc["splay"])))
        P, face = lay(support, start, go, bc["tails"], 0.0011)
        tb = tail_block(P, face, bc["tail_width"], 0.0012, 0.004, w_end=1.12, edge=0.4)
        d, _, tri, bary = cloth.closest(tb.verts)       # a flat ribbon over a curving band: its edges kept off it
        pieces.append((tb.verts + cloth.outward(tri, bary) * np.maximum(0.0006 - d, 0.0)[:, None], tb.faces, tb.uv))
        tails.append(float(np.linalg.norm(np.diff(P, axis=0), axis=1).sum()))
    return pieces, C, tails, w


# ---------------------------------------------------------------- textures and materials
def _img(rgb):
    out = np.empty(rgb.shape[:2] + (4,), np.uint8)
    out[..., :3] = np.rint(np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8)
    out[..., 3] = 255
    return out


def _rows(h):
    """v (h, 1) of each row of a picture (row 0 at the top: v 1)."""
    return (1.0 - (np.arange(h) + 0.5) / h)[:, None]


def crown_picture(col, cc, seeds, w=1024, h=256):
    """The crown's picture: u round it from her back, v up it from the seam (v 0, the picture's bottom row) to the
    apex. The cloth colour, lighter toward the top, in shade under the bulge near the band and in the pleats' folds."""
    u = (np.arange(w) + 0.5) / w
    v = _rows(h)
    pw, _ = wave(u, int(cc["pleats"]), seeds[0])
    reach = float(cc["reach"])
    env = G.smoothstep(v / PLEAT_IN) * (1.0 - G.smoothstep((v - PLEAT_IN) / max(reach - PLEAT_IN, 1e-6)))
    fold = np.clip(-pw, 0.0, 1.0)[None] ** 1.5 * env
    shade = np.clip(0.55 * G.smoothstep(1.0 - v / 0.10) + 0.30 * fold, 0.0, 1.0)
    light = 0.45 * G.smoothstep((v - 0.45) / 0.5)
    rgb = col["cloth"] + (col["cloth_light"] - col["cloth"]) * light[..., None]
    rgb = rgb + (col["cloth_shadow"] - rgb) * shade[..., None]
    return _img(np.broadcast_to(rgb, (h, w, 3)))


def band_picture(col, v_lip, v_line, w=256, h=256):
    """The band's picture: u round it, v from the top of its inner side (0) round its lip (`v_lip`) up its outer side
    to the seam (1); white, the line round it (v_line: [low, high]), above the line in the shade the crown throws on it
    (the trim's shadow, going toward the crown's own toward the seam), and a little shade at its rolled lower edge."""
    v = _rows(h)
    px = 1.0 / h
    line = np.clip((v - v_line[0]) / px + 0.5, 0.0, 1.0) * np.clip((v_line[1] - v) / px + 0.5, 0.0, 1.0)
    rgb = col["trim"] + (col["line"] - col["trim"]) * line[..., None]
    above = v_line[1] + 0.02                            # in the crown's shade above the line, no lit strip there:
    shade = G.smoothstep((v - above) / 0.06)            # the trim's shadow, toward the crown's own up to the seam
    deep = (0.5 + 0.2 * G.smoothstep((v - above) / (1.0 - above))) * shade
    low = 0.25 * G.smoothstep(1.0 - (v - v_lip) / 0.06) * (v > v_lip - 0.08)
    rgb = rgb + (col["trim_shadow"] - rgb) * np.clip(shade + low, 0.0, 1.0)[..., None]
    rgb = rgb + (col["cloth_shadow"] - rgb) * deep[..., None]
    return _img(np.broadcast_to(rgb, (h, w, 3)))


def frill_picture(col, w=64, h=64):
    """The frill's picture: v from its hem (0) up to where it is sewn under the band (1): white, in shade at the top."""
    v = _rows(h)
    shade = 0.6 * G.smoothstep((v - 0.55) / 0.45)
    rgb = col["trim"] + (col["trim_shadow"] - col["trim"]) * shade[..., None]
    return _img(np.broadcast_to(rgb, (h, w, 3)))


def moon_picture(col, w=64, h=64):
    """The crescent's picture: u along it, v round its section (0.25 its front, 0.75 its back): lit gold on its front,
    in shade round the back."""
    v = _rows(h)[:, 0]
    front = np.clip(np.cos(2.0 * np.pi * (v - 0.25)), 0.0, 1.0)
    back = np.clip(np.cos(2.0 * np.pi * (v - 0.75)), 0.0, 1.0)
    rgb = col["gold"][None] + (col["gold_light"] - col["gold"])[None] * (0.7 * front ** 2)[:, None]
    rgb = rgb + (col["gold_shadow"] - rgb) * (0.8 * back)[:, None]
    return _img(np.broadcast_to(rgb[:, None, :], (h, w, 3)))


def ink(rgb):
    """An outline colour for a cloth in shade `rgb`: the same hue, a little more saturated and much darker."""
    h, s, v = colorsys.rgb_to_hsv(*(float(x) for x in rgb))
    return np.array(colorsys.hsv_to_rgb(h, min(1.0, s * 1.4 + 0.08), v * 0.52))


def material(ctx, name, name_en, col, key, picture, edge_size, **kw):
    """A cloth material of the cap: the toon shade is `<key>_shadow` over `<key>` (per channel), the outline `<key>_ink`
    or a darker shade of the shadow's hue."""
    base, shadow = col[key], col[f"{key}_shadow"]
    mult = tuple(float(x) for x in np.clip(shadow / np.maximum(base, 1e-3), 0.0, 1.0))
    mat = WR.material(ctx, name, name_en, base, picture, shadow=mult, edge_size=edge_size, **kw)
    line = col.get(f"{key}_ink")
    line = ink(shadow) if line is None else line
    mat.edge_color = (float(line[0]), float(line[1]), float(line[2]), 1.0)
    return mat


# ---------------------------------------------------------------- the builder
def _patch(V, faces, uv_v, mat, bone):
    """A Patch of vertices with per-vertex texture coordinates (made per face corner), all on `bone`."""
    uv = np.asarray(uv_v, float)[np.array([i for f in faces for i in f], int)]
    return G.Patch(np.asarray(V, float), [tuple(int(i) for i in f) for f in faces], uv, {bone: np.ones(len(V))}, mat)


def frame_on(points, along, k):
    """(point, up the cloth, out of it, round it) at row `k` of a column of the band's outer side."""
    up = G.unit(points[min(k + 1, len(points) - 1)] - points[max(k - 1, 0)])
    out = G.unit(np.cross(along[k], up))
    up = G.unit(up - (up @ out) * out)
    return points[k], up, out, np.cross(up, out)


def frill_rings(band, c, seeds, surf):
    """The frill's rings (rows from where it is sewn inside the band's lip down to its hem, 3 x the band's columns),
    their texture u (with the seam's 1) and v, and the rings of its crests (what a ribbon lying on it rests on). Every
    pleat stays `margin` off `surf` (the skin or the hair under it), measured as the nearest distance."""
    fc, gap, thick = c["frill"], c["gap"], c["thick"]
    M = 3 * len(band.phi)
    phi = np.pi + 2.0 * np.pi * np.arange(M) / M
    t = np.linspace(0.0, 1.0, 10)
    h = 0.5 * thick - t * (0.5 * thick + fc["depth"])
    dirs = band.dirs(phi)
    base = band.radius(0.5 * thick, phi) + gap + 0.5 * thick
    u = ring_u(band.centre + 0.5 * thick * band.a + dirs * base[:, None])
    pw, pdepth = wave(u[:-1], int(fc["pleats"]), seeds[3], 0.35)
    amp = fc["amp"] * t ** 1.2
    R = []
    for i in range(len(t)):
        need = band.radius(h[i], phi) + fc["margin"] + 1.25 * amp[i] * pdepth
        r = np.maximum(base + fc["flare"] * t[i] ** 1.5, G.dilate_circular(need, 3))
        R.append(np.maximum(G.smooth_circular(r, 2, 3), need))
    R = np.array(R)

    def rings_at(R, wave_share):
        return band.centre + h[:, None, None] * band.a + dirs[None] * (R + amp[:, None] * wave_share)[..., None]
    for _ in range(4):                                  # the nearest distance, not only round the band's axis
        rings = rings_at(R, pw * pdepth)
        short = np.maximum(fc["margin"] - np.abs(surf.closest(rings.reshape(-1, 3))[0]).reshape(R.shape), 0.0)
        short[0] = 0.0                                  # sewn inside the band's lip
        if short.max() < 2e-5:
            break
        short = np.array([G.smooth_circular(G.dilate_circular(s, 4), 1, 3) for s in short])
        R = R + 1.3 * np.maximum.accumulate(short, axis=0)      # a row pushed out takes the rows under it along
    return rings_at(R, pw * pdepth), u, 1.0 - t, rings_at(R + 0.0005, pdepth)


def slouch(cc, seam, centre):
    """Per meridian (M,), 0 on the side away from `lean_toward` and 1 on that side, following the cosine between."""
    o = (seam - centre) * [1.0, 1.0, 0.0]
    az = np.arctan2(o[:, 0], -o[:, 1])
    return 0.5 * (1.0 + np.cos(az - np.radians(cc["lean_toward"])))


def sector(seam, centre, theta, width=15.0):
    """The meridians (M,) bool within `width` degrees of the azimuth `theta` (round from her front toward her left)."""
    o = seam - centre
    az = np.degrees(np.arctan2(o[:, 0], -o[:, 1]))
    return np.abs((az - theta + 180.0) % 360.0 - 180.0) <= width


def shape_crown(band, seam, centre, reach, c, u, seeds, top_z):
    """The crown and where the crescent sits on it: (grid (K, M, 3), t (K, M), the crescent's frame (centre, ex, ey, f),
    its shape (moon_shape), the height of the roll's lowest point under it). Its plain shape is sized meridian by
    meridian (overhang, droop, the top's height); the crescent's place is found on it; the pleats and unevenness are
    added, smoothed out under the crescent, and the overhang and droop at her front-right and back-left measured on
    that and the plain shape sized again until they are what was asked."""
    cc, mc = c["crown"], c["moon"]
    side = slouch(cc, seam, centre)
    goal_over = cc["overhang"] + cc["slump"] * side
    goal_drop = cc["hang"] + cc["droop"] * side
    want_over, want_drop = goal_over.copy(), goal_drop.copy()
    fr, bl = sector(seam, centre, cc["lean_toward"] - 180.0), sector(seam, centre, cc["lean_toward"])
    R = np.linalg.norm((seam - centre)[:, :2], axis=1)
    D, droop, z_top = want_over + reach - R, want_drop.copy(), top_z
    shape = moon_shape(mc)
    for _ in range(3):
        for _ in range(4):
            plain, tt = crown_grid(seam, centre, D, droop, z_top, cc, u, seeds, c["rings"], detail=False)
            over, hang = crown_sizes(plain, seam, centre, reach)
            D = np.maximum(D + want_over - over, 0.5 * want_over)
            droop = np.maximum(droop + want_drop - hang, 0.0)
            z_top += top_z - float(plain[..., 2].max())
        C0, ex, ey, f, roll_z = moon_frame(band, plain, mc["at"], mc["lift"])
        calm = calm_under(plain, (C0, ex, ey, f), shape[4], 0.010)
        for _ in range(3):
            crown, tt = crown_grid(seam, centre, D, droop, z_top, cc, u, seeds, c["rings"], calm=calm)
            z_top += top_z - float(crown[..., 2].max())
        over, hang = crown_sizes(crown, seam, centre, reach)
        e_fr, e_bl = (goal_over[m].mean() - over[m].mean() for m in (fr, bl))
        d_fr, d_bl = (goal_drop[m].mean() - hang[m].mean() for m in (fr, bl))
        if max(abs(e_fr), abs(e_bl), abs(d_fr), abs(d_bl)) < 2e-4:
            break
        want_over = want_over + e_fr + (e_bl - e_fr) * side
        want_drop = np.maximum(want_drop + d_fr + (d_bl - d_fr) * side, 0.0)
    return crown, tt, (C0, ex, ey, f), shape, roll_z


@builder("mob_cap", needs=("body", "head"))
def build(ctx):
    c = SP.merge(DEFAULTS, ctx.cfg)
    fc, cc, mc, bc = c["frill"], c["crown"], c["moon"], c["bow"]
    head = ctx.need("head")
    fit = HeadFit(head.info, meshes=head.meshes)
    col = WR.colours(ctx, "mob_cap", COLORS)
    bone, gap, thick = c["bone"], c["gap"], c["thick"]
    seeds = [int(x) for x in ctx.rng.integers(0, 2 ** 31, 6)]
    z_front = float(np.mean([np.asarray(e["center"], float)[2] for e in fit.eyes.values()])) + c["edge"]
    skull_top = float(fit.top[2])
    below = fc["depth"] + fc["flare"] + 0.012
    surf = under(ctx.parts, z_front - c["drop"] - below - 0.02)
    band = Band(surf, fit.center, z_front, c["drop"], c["band"], gap + 0.5 * thick, int(c["around"]), below)

    rings, v_band, v_lip, levels = band_rings(band, c)          # inner side, rolled lip, soft outer side to the seam
    n_in = len(rings) - len(levels)
    seam = rings[-1]
    u = ring_u(seam)
    centre = seam.mean(0)
    o = G.unit((seam - centre) * [1.0, 1.0, 0.0])
    reach = np.einsum("kmj,mj->km", (rings[n_in - 3:] - centre) * [1.0, 1.0, 0.0], o).max(0)

    crown, tt, frame, shape, roll_z = shape_crown(band, seam, centre, reach, c, u[:-1], seeds, skull_top + cc["top"])
    crown, pushed = push_clear(crown, tt, surf, gap)
    if pushed > 0.003:
        ctx.log(f"WARNING the hair under the crown reaches higher than the crown allows: pushed out "
                f"{pushed * 1000:.1f} mm")

    band_patch = G.loft(rings, closed=True, u=u, v=v_band, flip=True, mat=BAND)
    crown_patch = G.loft(crown, closed=True, u=u, v=np.concatenate([tt, tt[:, :1]], 1), flip=True, mat=CROWN)
    frill, uf, vf, crest = frill_rings(band, c, seeds, surf)
    frill_patch = G.loft(frill, closed=True, u=uf, v=vf, mat=FRILL)
    crown_t = G.tri_arrays(crown_patch.f)[0]
    cloth = G.join([band_patch, crown_patch, frill_patch])
    cloth_t = G.tri_arrays(cloth.f)[0]
    cloth_skin = OF.Skin(cloth.v, cloth_t)

    C0, ex, ey, f = frame                                        # the crescent pinned on the puff
    Vm = lay_moon(mc, frame, shape, crown_patch.v, crown_t)
    Fm, uvm, ol, hm = shape[1], shape[2], shape[4], shape[0][:, 2]
    for _ in range(4):                                  # on the cloth, never in it
        near = float(cloth_skin.closest(Vm)[0].min())
        if near >= 0.00045:
            break
        Vm = Vm + f * (0.0005 - near)

    jb, fb = band.at(bc["at"])                                   # the bow on the middle of the band
    outer = rings[n_in:, jb] * (1.0 - fb) + rings[n_in:, (jb + 1) % len(band.phi)] * fb
    along = G.unit(rings[n_in:, (jb + 1) % len(band.phi)] - rings[n_in:, jb])
    roll = crown[:, jb] * (1.0 - fb) + crown[:, (jb + 1) % len(band.phi)] * fb
    roll = float((roll[np.argmin(roll[:, 2])] - band.centre) @ band.a)            # under the roll, on the band
    level = np.clip(roll - 0.5 * bc["loop"][0] - 0.002, 0.5 * bc["loop"][0] + 0.001, 0.5 * c["band"])
    kb = int(np.clip(np.argmin(np.abs(levels - level)), 1, len(levels) - 2))
    crest_t = G.tri_arrays(G.loft(crest, closed=True).f)[0]
    # the tails lie on the band, on the frill's crests and, only below the frill's hem, on the hair
    low = surf.tris[(((surf.verts - band.centre) @ band.a)[surf.tris] < -fc["depth"] - 0.001).all(1)]
    support = OF.Skin(np.vstack([cloth.v, crest.reshape(-1, 3), surf.verts]),
                      np.vstack([cloth_t, crest_t + len(cloth.v), low + len(cloth.v) + crest.size // 3]))
    bow, Cb, tails, wb = bow_pieces(bc, frame_on(outer, along, kb), float(np.linalg.norm((outer[kb] - centre)[:2])),
                                    cloth_skin, support)

    soup = G.Soup("mob_cap", list(MATS))
    for p in (band_patch, crown_patch, frill_patch):
        p.w = {bone: np.ones(len(p.v))}
        soup.add(p)
    soup.add(_patch(Vm, Fm, uvm, MOON, bone))
    for V, F, uv in bow:
        soup.add(_patch(V, F, uv, BOW, bone))
    mesh = soup.mesh()

    line = [v_lip + (1.0 - v_lip) * (c["line_at"] + 0.5 * s * c["line"] - 0.5 * thick) / (c["band"] - 0.5 * thick)
            for s in (-1.0, 1.0)]
    mats = [material(ctx, MATS[CROWN], "cap", col, "cloth", crown_picture(col, cc, seeds), 0.8, double_sided=False),
            material(ctx, MATS[BAND], "cap_band", col, "trim", band_picture(col, v_lip, line), 0.7,
                     double_sided=False),
            material(ctx, MATS[FRILL], "cap_frill", col, "trim", frill_picture(col), 0.45, double_sided=True),
            material(ctx, MATS[MOON], "cap_moon", col, "gold", moon_picture(col), 0.6, specular=(0.35, 0.30, 0.18),
                     shininess=24.0, double_sided=False),
            material(ctx, MATS[BOW], "cap_bow", col, "bow", None, 0.5, specular=(0.30, 0.26, 0.16), shininess=20.0,
                     double_sided=False)]

    info = measure(band, rings, n_in, crown, frill, centre, reach, surf, skull_top, c)
    loops = np.vstack([p[0] for p in bow[:3]])
    back = hm < 0.25 * mc["thick"]
    stand = cloth_skin.closest(Vm)[0]
    Q = Vm - C0
    info["crown"]["pushed"] = pushed
    jm, fm = band.at(mc["at"])
    seam_z = float(seam[jm, 2] * (1.0 - fm) + seam[(jm + 1) % len(seam), 2] * fm)
    info["moon"] = {"centre": C0, "theta": float(np.degrees(np.arctan2((C0 - centre)[0], -(C0 - centre)[1]))),
                    "above_roll": float(C0[2] - roll_z), "frame": [ex, ey, f], "outline": ol,
                    "height": float(np.ptp(Q @ ey)), "width": float(np.ptp(Q @ ex)), "thick": float(np.ptp(hm)),
                    "back": [float(stand[back].min()), float(stand[back].max())], "farthest": float(stand.max()),
                    "low_under_roll": float(roll_z - Vm[:, 2].min()), "low_under_seam": float(seam_z - Vm[:, 2].min())}
    info["bow"] = {"centre": Cb, "theta": float(np.degrees(np.arctan2((Cb - centre)[0], -(Cb - centre)[1]))),
                   "tails": tails, "width": float(np.ptp(loops @ wb)),
                   "high": float(np.ptp(loops @ G.unit(UP - (UP @ wb) * wb))),
                   "off": float(cloth_skin.closest(loops)[0].min())}
    info["landmarks"] = {"cap_top": crown.reshape(-1, 3)[int(np.argmax(crown[..., 2]))], "cap_moon": C0, "cap_bow": Cb}
    cr, mo = info["crown"], info["moon"]
    ctx.log(f"band edge z {info['band']['front']:.4f} front, {info['band']['back']:.4f} back; crown top "
            f"{cr['top']:.4f} ({cr['above_skull'] * 1000:.0f} mm over the skull), overhang front-right "
            f"{cr['overhang_front_right'] * 1000:.0f} mm, back-left {cr['overhang_back_left'] * 1000:.0f} mm, lean "
            f"{cr['lean']:.1f} deg toward {cr['lean_toward']:.0f}; crescent {mo['height'] * 1000:.0f} x "
            f"{mo['width'] * 1000:.0f} mm, its back {mo['back'][0] * 1000:.1f}..{mo['back'][1] * 1000:.1f} mm off the "
            f"cloth; gap {info['gap']['cap'] * 1000:.1f} mm")
    return Part("mob_cap", meshes=[mesh], materials=mats, info=info)


def measure(band, rings, n_in, crown, frill, centre, reach, surf, skull_top, c):
    """The cap's measures for `info`: the band's lower edge at her front and back, the frill's depth under it, the
    crown's top, overhang and droop (all round, and at her front-right and back-left: the meridians within 15 degrees,
    their mean), its lean, the smallest gaps to what is under the cap."""
    M = len(band.phi)
    edge = rings[n_in - 3]                                       # the lowest ring of the rolled lip
    over, hang = crown_sizes(crown, rings[-1], centre, reach)
    toward = float(c["crown"]["lean_toward"])
    fr, bl = sector(rings[-1], centre, toward - 180.0), sector(rings[-1], centre, toward)
    apex = crown[-1].mean(0)
    off = apex - centre
    hem = -((frill[-1] - band.centre) @ band.a)
    gaps = np.abs(surf.closest(np.vstack([rings.reshape(-1, 3), crown.reshape(-1, 3)]))[0])
    return {"band": {"front": float(edge[M // 2, 2]), "back": float(edge[0, 2]), "height": float(c["band"]),
                     "centre": band.centre, "normal": band.a, "lower_edge": edge, "seam": rings[-1]},
            "frill": {"depth": [float(hem.min()), float(hem.max())]},
            "crown": {"top": float(crown[..., 2].max()), "above_skull": float(crown[..., 2].max() - skull_top),
                      "overhang": [float(over.min()), float(over.mean()), float(over.max())],
                      "overhang_front_right": float(over[fr].mean()), "overhang_back_left": float(over[bl].mean()),
                      "droop_front_right": float(hang[fr].mean()), "droop_back_left": float(hang[bl].mean()),
                      "lean": float(np.degrees(np.arctan2(np.hypot(off[0], off[1]), off[2]))),
                      "lean_toward": float(np.degrees(np.arctan2(off[0], -off[1])))},
            "gap": {"cap": float(gaps.min()), "frill": float(np.abs(surf.closest(frill.reshape(-1, 3))[0]).min())}}
