"""Tools the truck's exterior hardware is built with (bpy-free, pure numpy on `core.shell`): tubes along bent paths with
rotation-minimising frames (round, open or capped ends), lathes turned onto any axis, sweeps of non-convex sections with
flat (ear-clipped) caps, coils, and `Body`: where TruckBody's smooth shells are, from the builder's ray-cast probes or,
without them (tests, a build before the shells exist), from a stand-in of rounded boxes laid out on the layout's numbers.
"""
import math

import numpy as np

from ....core import shell as S
from . import semi_truck_layout as L

M = L.M
UP = np.array([0.0, 0.0, 1.0])
EPS = 1e-9


def pos(m):
    """The mesh with its faces outward (positive volume)."""
    return m if m.volume() > 0 else m.flip()


def unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


def mirror_x(m):
    """A +X side part as the -X side's (the winding is kept outward)."""
    return m.mirrored_x()


def both_sides(m):
    """The +X part and its mirror image."""
    return [m, m.mirrored_x()]


# ===================================================================================================================
# paths
# ===================================================================================================================
def fillet3(points, radius, n=8):
    """Round the corners of a 3D polyline: every inner corner becomes a circular arc of `radius` (one value or one per
    point; 0 keeps it) in the plane of its two segments, n segments per arc. The tangent length is limited to half of
    each neighbouring segment (the whole first / last one), so close corners shrink instead of crossing."""
    P = np.asarray(points, float)
    N = len(P)
    rad = np.broadcast_to(np.asarray(radius, float), (N,))
    out = [P[:1]]
    for i in range(1, N - 1):
        a, b, c = P[i - 1], P[i], P[i + 1]
        d0, d1 = b - a, c - b
        l0, l1 = np.linalg.norm(d0), np.linalg.norm(d1)
        d0, d1 = d0 / l0, d1 / l1
        phi = math.acos(float(np.clip(d0 @ d1, -1.0, 1.0)))
        if rad[i] <= 0 or phi < 1e-4:
            out.append(b[None])
            continue
        t = min(rad[i] * math.tan(phi / 2), (l0 if i == 1 else 0.5 * l0), (l1 if i == N - 2 else 0.5 * l1))
        r = t / math.tan(phi / 2)
        p0 = b - d0 * t
        nrm = d1 - d0 * (d0 @ d1)
        nrm = nrm / np.linalg.norm(nrm)
        ctr = p0 + nrm * r
        u, v = p0 - ctr, d0 * r
        a_ = np.linspace(0.0, phi, n + 1)
        out.append(ctr + np.cos(a_)[:, None] * u + np.sin(a_)[:, None] * v)
    out.append(P[-1:])
    Q = np.concatenate(out)
    keep = np.concatenate([[True], np.linalg.norm(np.diff(Q, axis=0), axis=1) > 1e-7])
    return Q[keep]


def densify(path, step):
    """Extra points on the straight runs so no segment is longer than `step` (arcs keep their points)."""
    P = np.asarray(path, float)
    out = [P[:1]]
    for a, b in zip(P[:-1], P[1:]):
        k = max(1, int(math.ceil(np.linalg.norm(b - a) / step - 1e-9)))
        out.append(a + (b - a) * (np.arange(1, k + 1) / k)[:, None])
    return np.concatenate(out)


def arclength(P):
    return np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))])


def tangents(P):
    T = np.empty_like(P)
    T[1:-1] = P[2:] - P[:-2]
    T[0], T[-1] = P[1] - P[0], P[-1] - P[-2]
    return T / np.maximum(np.linalg.norm(T, axis=1), EPS)[:, None]


def rmf(P, up=None):
    """Rotation-minimising frames along a path (double reflection): tangents T, normals R, binormals B (n, 3) each.
    The first normal is `up` made perpendicular to the first tangent (default: world Z, or X for a vertical start)."""
    P = np.asarray(P, float)
    T = tangents(P)
    n = len(P)
    u = UP if up is None else np.asarray(up, float)
    r0 = u - T[0] * (u @ T[0])
    if np.linalg.norm(r0) < 1e-6:
        u = np.array([1.0, 0.0, 0.0]) if abs(T[0][0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        r0 = u - T[0] * (u @ T[0])
    R = np.empty_like(P)
    R[0] = r0 / np.linalg.norm(r0)
    for i in range(n - 1):
        v1 = P[i + 1] - P[i]
        c1 = v1 @ v1
        if c1 < 1e-18:
            R[i + 1] = R[i]
            continue
        rL = R[i] - (2.0 / c1) * (v1 @ R[i]) * v1
        tL = T[i] - (2.0 / c1) * (v1 @ T[i]) * v1
        v2 = T[i + 1] - tL
        c2 = v2 @ v2
        r = rL - (2.0 / c2) * (v2 @ rL) * v2 if c2 > 1e-18 else rL
        r = r - T[i + 1] * (r @ T[i + 1])
        R[i + 1] = r / np.linalg.norm(r)
    return T, R, np.cross(T, R)


# ===================================================================================================================
# tubes
# ===================================================================================================================
def _end_rings(p, t, r, b, rad, ang, kind, wall, depth):
    """Rings past an end of a tube at point p looking along t (outward): 'round' a quarter-round dome, 'flat' a rolled
    rim onto a flat face, 'open' a rolled lip, the inner wall `depth` back inside; returns (rings, cap bulge, inner)."""
    ca, sa = np.cos(ang), np.sin(ang)

    def ring(c, rr):
        return c + rr * (ca[:, None] * r + sa[:, None] * b)

    if kind == "round":
        phis = np.radians([25.0, 50.0, 70.0])
        return [ring(p + t * rad * math.sin(f), rad * math.cos(f)) for f in phis], rad * (1 - math.sin(phis[-1])), False
    if kind == "flat":
        e = min(0.25 * rad, 0.004)
        phis = np.radians([40.0, 75.0])
        return [ring(p + t * e * math.sin(f), rad - e + e * math.cos(f)) for f in phis], 0.0, False
    if kind == "open":
        w = wall
        rings = [ring(p + t * 0.5 * w, rad - 0.5 * w), ring(p, rad - w)]
        rings += [ring(p - t * min(0.02, 0.4 * depth), rad - w), ring(p - t * depth, rad - w)]
        return rings, 0.0, True
    if kind is None:
        return [], 0.0, False
    raise ValueError(f"tube end {kind!r}: round, flat, open or None")


def tube(path, radius, sides=16, mat=0, ends=("round", "round"), up=None, wall=0.003, depth=0.08, dark=None, phase=0.0):
    """A closed tube along `path` (n, 3) with rotation-minimising frames (no twist round bends). radius: one value or one
    per point. ends (start, end): 'round' (a dome), 'flat' (a rolled rim and a flat face), 'open' (a pipe's mouth: a
    rolled lip of `wall` and the inner wall `depth` deep, its faces in material `dark` past the first 15 mm) or None
    (an open ring: only for ends buried inside another closed part)."""
    P = np.asarray(path, float)
    n = len(P)
    rad = np.broadcast_to(np.asarray(radius, float), (n,)).astype(float)
    T, R, B = rmf(P, up)
    ang = phase + 2.0 * math.pi * np.arange(sides) / sides
    ca, sa = np.cos(ang), np.sin(ang)
    body = [P[i] + rad[i] * (ca[:, None] * R[i] + sa[:, None] * B[i]) for i in range(n)]
    pre, bulge0, in0 = _end_rings(P[0], -T[0], R[0], B[0], rad[0], ang, ends[0], wall, depth)
    post, bulge1, in1 = _end_rings(P[-1], T[-1], R[-1], B[-1], rad[-1], ang, ends[1], wall, depth)
    rings = np.stack(pre[::-1] + body + post)
    caps = ("fan" if ends[0] is not None else None, "fan" if ends[1] is not None else None)
    m = S.cage(rings, closed=True, caps=caps, cap_axes=(-T[0], T[-1]), cap_bulge=(bulge0, bulge1), cap_rings=0 if ends[0] == "round" or ends[1] == "round" else 1, mat=mat)
    m = pos(m) if all(e is not None for e in ends) else m
    if dark is not None:
        for flag, p, t in ((in0, P[0], -T[0]), (in1, P[-1], T[-1])):
            if flag:
                m.assign(dark, lambda c, nn, p=p, t=t: (c - p) @ t < -0.015)
    return m


def lathe_on(profile, origin, axis, up=None, seg=24, mat=0, phase=0.0):
    """S.lathe (about +Z, profile (r, z)) turned so its +Z runs along `axis` and moved to `origin`."""
    m = S.lathe(np.asarray(profile, float), seg=seg, mat=mat, phase=phase)
    a = unit(axis)
    u = up if up is not None else (UP if abs(a[2]) < 0.9 else np.array([0.0, 1.0, 0.0]))
    return pos(S.place(m, origin, a, u))


def disc_profile(r, h, e):
    """(r, z) profile of a short closed cylinder: radius r, height h along +z, edges rolled with radius e."""
    a = np.linspace(0.0, 0.5 * math.pi, 4)
    lo = [(r - e + e * math.sin(t), e - e * math.cos(t)) for t in a]
    hi = [(r - e + e * math.cos(t), h - e + e * math.sin(t)) for t in a]
    return [(0.0, 0.0)] + lo + hi + [(0.0, h)]


# ===================================================================================================================
# sweeps with flat caps
# ===================================================================================================================
def _in_tri(p, a, b, c):
    def s(p1, p2, p3):
        return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])
    d1, d2, d3 = s(p, a, b), s(p, b, c), s(p, c, a)
    return not ((d1 < -1e-14 or d2 < -1e-14 or d3 < -1e-14) and (d1 > 1e-14 or d2 > 1e-14 or d3 > 1e-14))


def ear_clip(P):
    """Triangles (k, 3) of a simple polygon (m, 2), counter-clockwise whatever the polygon's own orientation."""
    P = np.asarray(P, float)
    idx = list(range(len(P)))
    if S.polygon_area(P) < 0:
        idx = idx[::-1]
    tris = []
    guard = 0
    while len(idx) > 3:
        guard += 1
        if guard > 10 * len(P) ** 2:
            raise ValueError("ear_clip: not a simple polygon")
        m = len(idx)
        for k in range(m):
            i0, i1, i2 = idx[k - 1], idx[k], idx[(k + 1) % m]
            a, b, c = P[i0], P[i1], P[i2]
            if (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0]) <= 1e-14:
                continue
            if any(_in_tri(P[j], a, b, c) for j in idx if j not in (i0, i1, i2)):
                continue
            tris.append((i0, i1, i2))
            idx.pop(k)
            break
        else:
            raise ValueError("ear_clip: no ear (a self-crossing polygon?)")
    tris.append(tuple(idx))
    return np.array(tris, np.int64)


def sweep_solid(path, section, scale=None, up=UP, mat=0):
    """S.sweep of any simple section (non-convex too: a C channel, a bracket's outline) closed with flat caps from an
    ear-clipped triangulation of the section. Closed and outward."""
    sec = np.asarray(section, float)
    m = S.sweep(path, sec, scale=scale, up=up, caps=(False, False), mat=mat)
    k = len(sec)
    ns = len(m.V) // k
    tri = ear_clip(sec)
    if S.polygon_area(sec) < 0:
        tri = tri[:, ::-1]
    T = np.concatenate([tri[:, ::-1], tri + (ns - 1) * k])
    out = S.Mesh(m.V, m.Q, T, m.Qm, np.full(len(T), mat, np.int32))
    return pos(out)


def ring_sweep(path, section, up=UP, mat=0):
    """S.sweep of `section` round a closed `path` (a band, a strap): the seam welded shut, closed and outward."""
    return pos(S.sweep(np.asarray(path, float), np.asarray(section, float), up=up, closed=True, mat=mat).weld(1e-9))


def offset_polyline(P, d):
    """A 2D polyline moved d along its left normals (per vertex, from the neighbours' chord)."""
    P = np.asarray(P, float)
    t = np.empty_like(P)
    t[1:-1] = P[2:] - P[:-2]
    t[0], t[-1] = P[1] - P[0], P[-1] - P[-2]
    t /= np.linalg.norm(t, axis=1, keepdims=True)
    return P + d * np.stack([-t[:, 1], t[:, 0]], 1)


def sheet_section(outer, t):
    """The closed section of a formed sheet of thickness t whose outer face is the 2D polyline `outer`: the face, a
    half-round at the far end, the inner face back (moved t along the left normals) and a half-round at the start."""
    A = np.asarray(outer, float)
    Bi = offset_polyline(A, t)

    def tip(p, q, d):
        c, h = 0.5 * (p + q), 0.5 * (q - p)
        n = np.array([-h[1], h[0]])
        n = n / max(np.linalg.norm(n), EPS) * np.linalg.norm(h)
        s = 1.0 if (n @ d) > 0 else -1.0
        return [c + h * math.cos(a) + s * n * math.sin(a) for a in (2.0 * math.pi / 3.0, math.pi / 3.0)]

    d_end = A[-1] - A[-2]
    d_start = A[0] - A[1]
    end = tip(A[-1], Bi[-1], d_end)
    start = tip(Bi[0], A[0], d_start)
    return np.concatenate([A, np.array(end), Bi[::-1], np.array(start)])


# ===================================================================================================================
# coils
# ===================================================================================================================
def coil(axis_path, coil_r, pitch, per_turn=10, phase=0.0, up=None):
    """A helix wound round a curve: the centre line of a coiled hose or a spring. Returns (m, 3)."""
    A = densify(np.asarray(axis_path, float), 0.02)
    s = arclength(A)
    turns = s[-1] / pitch
    m = max(2, int(math.ceil(turns * per_turn)) + 1)
    t = np.linspace(0.0, s[-1], m)
    C = np.stack([np.interp(t, s, A[:, k]) for k in range(3)], 1)
    T, R, B = rmf(C, up)
    a = phase + 2.0 * math.pi * t / pitch
    return C + coil_r * (np.cos(a)[:, None] * R + np.sin(a)[:, None] * B)


# ===================================================================================================================
# where the body is
# ===================================================================================================================
FENDER_FACE_Y = -6.965      # the stand-in's fender front (TruckBody's fenders face forward there at x 0.60-1.10)


def standin_shells():
    """Rounded boxes on the layout's numbers in place of TruckBody's shells: {key: Mesh}: "hood" (the hood and the
    fenders), "cab" (the cab, the cowl in front of it and the sloping visor), "sleeper". Used when the builder hands no
    probe for a key (tests, or a build before the shells exist); the real shells replace them."""
    hood = S.merge([
        S.rounded_box((2 * 0.76, L.COWL_Y - (L.GRILLE_Y + 0.02), L.HOOD_Z[0] - 1.30),
                      (0.0, 0.5 * (L.COWL_Y + L.GRILLE_Y + 0.02), 0.5 * (L.HOOD_Z[0] + 1.30)), r=0.06),
        S.rounded_box((2 * L.FENDER_HALF_W, -5.70 - FENDER_FACE_Y, L.FENDER_TOP_Z - 0.92),
                      (0.0, 0.5 * (-5.70 + FENDER_FACE_Y), 0.5 * (L.FENDER_TOP_Z + 0.92)), r=0.06),
    ])
    slope = math.degrees(math.atan2(0.10, 0.32))
    cab = S.merge([
        S.rounded_box((2 * L.CAB_HALF_W, L.BOC_Y - L.CAB_FRONT_Y, L.CAB_ROOF_Z - L.CAB_BOTTOM_Z),
                      (0.0, 0.5 * (L.BOC_Y + L.CAB_FRONT_Y), 0.5 * (L.CAB_ROOF_Z + L.CAB_BOTTOM_Z)), r=0.08),
        S.rounded_box((2 * 0.70, L.CAB_FRONT_Y - L.COWL_Y + 0.02, 2.05 - 1.10), (0.0, 0.5 * (L.CAB_FRONT_Y + L.COWL_Y), 0.5 * (2.05 + 1.10)),
                      r=0.05),                                                                    # the cowl
        S.rounded_box((2 * 0.97, 0.335, 0.04), r=0.015).rot(slope).moved((0.0, -5.08, 2.78)),     # the visor
    ])
    sleeper = S.rounded_box((2 * L.SLEEPER_HALF_W, L.SLEEPER_Y[1] - L.SLEEPER_Y[0], L.SLEEPER_ROOF_Z - L.CAB_BOTTOM_Z),
                            (0.0, 0.5 * (L.SLEEPER_Y[0] + L.SLEEPER_Y[1]), 0.5 * (L.SLEEPER_ROOF_Z + L.CAB_BOTTOM_Z)), r=0.10)
    return {"hood": hood, "cab": cab, "sleeper": sleeper}


class Body:
    """Ray casts on TruckBody's shells ("hood" with the fenders, "cab" with the visor, "sleeper"): the builder's probes
    when it has the key, else a stand-in of rounded boxes (standin_shells). `hit` returns (point, unit normal facing
    the ray's origin)."""

    def __init__(self, probes=None):
        self.probes = probes
        self._standin = {}

    def real(self, key):
        return self.probes is not None and key in self.probes

    def _probe(self, key):
        if self.real(key):
            return self.probes[key]
        if key not in self._standin:
            self._standin[key] = S.Probe.from_mesh(standin_shells()[key])
        return self._standin[key]

    def hit(self, key, origin, direction):
        h = self._probe(key).ray(np.asarray(origin, float), np.asarray(direction, float))
        if h is None and self.real(key):
            h = self.probes[key].ray(np.asarray(origin, float), np.asarray(direction, float))
        if h is None:
            raise ValueError(f"semi_truck_exterior: no '{key}' surface from {tuple(np.round(origin, 3))} along "
                             f"{tuple(np.round(direction, 3))}")
        return np.asarray(h[1], float), unit(h[2])

    def try_hit(self, key, origin, direction):
        try:
            return self.hit(key, origin, direction)
        except ValueError:
            return None
