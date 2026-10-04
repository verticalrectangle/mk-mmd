"""Closing an imported face shell (bpy-free, numpy only): boundary loops, a skull cap that continues the shell over the back
and the top of the head, and fans that fill its holes (eyes, mouth) for the published skin shell.

A face imported from a PMX is a front shell: it ends in an open boundary where the original model's hair began (and in the
neck, which joins the imported body natively). The cap closes the back and top with a surface that starts exactly on the shell's
outer loop (the cap's first ring IS the loop's vertices, so there is no gap or ledge) and relaxes to the parametric skull of the
head part: every ring contracts the loop's directions (seen from the head centre) towards the pole, and the radial mismatch
between the loop and the skull prior fades with the ring. All positions are in one common frame (the caller's), directions are
measured from `centre`.
"""
from collections import defaultdict

import numpy as np


def weld(V, tol=2e-5):
    """Group coincident vertices: rep[i] = index of the first vertex at i's position (rounded to `tol`)."""
    key = np.round(np.asarray(V, float) / tol).astype(np.int64)
    _, first, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
    return first[np.asarray(inv).reshape(-1)]


def boundary_loops(V, faces):
    """Boundary loops of a CCW-outward surface after welding coincident vertices. Returns a list of (welded ids, vertex ids)
    per loop, in the order of the faces' own winding (the surface lies on the left of the travel direction seen from
    outside); `vertex ids` picks, for every position, the vertex of the face that owns the outgoing boundary edge."""
    rep = weld(V)
    count = defaultdict(int)
    directed = []
    for f in faces:
        n = len(f)
        for k in range(n):
            a, b = int(f[k]), int(f[(k + 1) % n])
            u, v = int(rep[a]), int(rep[b])
            if u == v:
                continue
            count[(min(u, v), max(u, v))] += 1
            directed.append((u, v, a))
    nxt = defaultdict(list)
    bnd = []
    for u, v, a in directed:
        if count[(min(u, v), max(u, v))] == 1:
            nxt[u].append((v, a))
            bnd.append((u, v, a))
    used, loops = set(), []
    for u0, v0, a0 in bnd:
        if (u0, v0) in used:
            continue
        w, o = [], []
        u, v, a = u0, v0, a0
        while (u, v) not in used:
            used.add((u, v))
            w.append(u)
            o.append(a)
            cands = [c for c in nxt[v] if (v, c[0]) not in used]
            if not cands:
                break
            u, (v, a) = v, cands[0]
        loops.append((np.array(w, int), np.array(o, int)))
    return loops


def azimuth(P, c):
    """Azimuth about the vertical axis through c: 0 straight ahead (-y), positive towards +x."""
    d = np.asarray(P, float) - c
    return np.arctan2(d[..., 0], -d[..., 1])


def _smooth(t):
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _monotone(th, eps=np.radians(0.05)):
    """Azimuths of a loop that goes once round the axis, made strictly monotone (the loop may wiggle back by a fraction of a
    degree where it runs along a meridian): a running maximum in the direction of travel."""
    u = np.unwrap(th)
    s = 1.0 if u[-1] >= u[0] else -1.0
    v = s * u
    for i in range(1, len(v)):
        v[i] = max(v[i], v[i - 1] + eps)
    return s * v


def build_cap(loop_pos, centre, prior, rings=9, fade=0.72, pole_gap=0.10):
    """Rings of new vertices that close the region inside `loop_pos` (ordered closed polyline, (n, 3)).

    The loop must be star-shaped about the pole as seen from `centre` (polar angle about +z as a function of azimuth: a
    single value per azimuth). Ring k (1..rings-1) has the loop's azimuths and `(1 - k / rings)` of its polar angles; its
    radius is `prior(dirs)` (the skull prior's radius along each direction) plus the loop's own mismatch
    `r_loop - prior(d_loop)` faded out over the first `fade` of the way to the pole. The last ring is closed by one apex
    vertex on the pole. `pole_gap`: the apex ring's polar-angle fraction stays this far from the pole before the fan.

    Returns (new_verts (m, 3), faces) with faces over local indices: 0..n-1 are the loop vertices, n.. the new ones; faces
    are quads and triangles in the winding of increasing ring index with the loop order (the caller orients them)."""
    L = np.asarray(loop_pos, float)
    n = len(L)
    c = np.asarray(centre, float)
    d0 = L - c
    r0 = np.linalg.norm(d0, axis=1)
    u0 = d0 / r0[:, None]
    th = _monotone(np.arctan2(u0[:, 0], -u0[:, 1]))
    psi = np.arccos(np.clip(u0[:, 2], -1.0, 1.0))
    delta = r0 - np.linalg.norm(prior(u0) - c, axis=1)
    new, faces = [], []
    ring_ids = [np.arange(n)]
    base = n
    ts = np.linspace(0.0, 1.0 - pole_gap, rings + 1)[1:]
    for t in ts:
        p = psi * (1.0 - t)
        d = np.stack([np.sin(p) * np.sin(th), -np.sin(p) * np.cos(th), np.cos(p)], -1)
        R = np.linalg.norm(prior(d) - c, axis=1) + delta * (1.0 - _smooth(t / fade))
        new.append(c + d * R[:, None])
        ring_ids.append(np.arange(base, base + n))
        base += n
    apex_dir = np.array([0.0, 0.0, 1.0])
    Ra = float(np.linalg.norm(prior(apex_dir[None])[0] - c))
    new.append((c + apex_dir * Ra)[None])
    apex = base
    for k in range(len(ring_ids) - 1):
        a, b = ring_ids[k], ring_ids[k + 1]
        for i in range(n):
            j = (i + 1) % n
            faces.append((int(a[i]), int(a[j]), int(b[j]), int(b[i])))
    last = ring_ids[-1]
    for i in range(n):
        faces.append((int(last[i]), int(last[(i + 1) % n]), apex))
    return np.concatenate(new, 0), faces


def orient_outward(V, faces, centre):
    """Flip faces whose normal points towards `centre` (for convex-ish closed surfaces seen from inside)."""
    V = np.asarray(V, float)
    out = []
    for f in faces:
        a, b, c_ = V[f[0]], V[f[1]], V[f[2]]
        nrm = np.cross(b - a, c_ - a)
        mid = V[list(f)].mean(0) - centre
        out.append(tuple(f) if nrm @ mid >= 0 else tuple(reversed(f)))
    return out


def hole_fan(loop_pos, centre):
    """A fan over a hole: one apex at the loop's mean direction with the mean radius, triangles to every loop edge. Returns
    (apex (3,), faces over local indices: the loop 0..n-1, the apex n)."""
    L = np.asarray(loop_pos, float)
    c = np.asarray(centre, float)
    d = L - c
    r = np.linalg.norm(d, axis=1)
    m = (d / r[:, None]).mean(0)
    m /= np.linalg.norm(m)
    apex = c + m * r.mean()
    n = len(L)
    return apex, [(i, (i + 1) % n, n) for i in range(n)]
