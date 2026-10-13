"""Ink lines on garments (numpy only): the drawn lines of a 2D design, the ribs of a knit band, the folds of a sleeve,
the creases behind a knee, as thin dark strips lying just above the cloth. A strip takes the weights of the cloth under
it, so it moves with it in any pose; it is drawn the same width whatever the texture's scale, with pointed ends like a
pen stroke.

    lines = isolines(sh, P, f, value)                      # where a field crosses a value: [(points, normals, W, fields)]
    lines = ribs(sh, P, u, 0.02, "z", lo, hi, rng)         # rib lines across a ribbed piece, every 2 cm round it
    lines = arcs(sh, P, f, value, keep, fields=("az",))     # the pieces of a level line where keep(fields) holds
    lines = dashes(lines, rng, (0.05, 0.15), (0.01, 0.03))  # broken by hand into pieces and gaps (metres)
    patch = strokes(sh.bones, lines, width=0.0014)          # an outfit_geo.Patch of pen strokes

`sh` is an outfit_shell.Shell and P its positions (the cloth as built); W are weights over `sh.bones`. A line's normals
are its triangles' (the shell is wound like the skin: outward), so the strokes lie on the outside of the cloth."""
import numpy as np

from . import outfit_geo as G


def _unit(v):
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12)


def _tri_normals(P, T):
    a, b, c = P[T[:, 0]], P[T[:, 1]], P[T[:, 2]]
    return _unit(np.cross(b - a, c - a))


def isolines(sh, P, f, value, fields=(), period=None, tris=None):
    """The level lines f == value on the shell at positions P, chained into polylines: a list of (points (k, 3),
    normals (k, 3), weights (k, B), {name: (k,)} the shell's `fields` named in `fields` along them); a closed line ends
    on its first point. Only the triangles in `tris` (a bool mask, default all) are crossed. A periodic f (an angle:
    `period` a turn) is taken as its signed offset from `value` (within half a turn), so it only wraps half a turn away
    from the line, where no triangle is crossed."""
    P = np.asarray(P, float)
    f = np.asarray(f, float) - value
    if period:
        f = (f + 0.5 * period) % period - 0.5 * period
    T = sh.tris
    nrm = _tri_normals(P, T)
    side = f[T] >= 0.0
    cut = side.any(1) & ~side.all(1)
    if period:
        cut &= np.ptp(f[T], axis=1) <= 0.5 * period
    if tris is not None:
        cut &= np.asarray(tris, bool)
    seg_of = {}                                      # edge key -> list of triangles crossing it
    tri_edges = {}
    for t in np.flatnonzero(cut):
        tri = T[t]
        keys = []
        for k in range(3):
            i, j = int(tri[k]), int(tri[(k + 1) % 3])
            if (f[i] >= 0.0) != (f[j] >= 0.0):
                keys.append((min(i, j), max(i, j)))
        if len(keys) == 2:
            tri_edges[t] = keys
            for key in keys:
                seg_of.setdefault(key, []).append(t)
    seen, out = set(), []
    for t0 in list(tri_edges):
        if t0 in seen:
            continue
        # walk both ways from t0 along shared edges
        chain_t, chain_e = [t0], list(tri_edges[t0])
        seen.add(t0)
        for direction in (1, 0):
            e = tri_edges[t0][direction]
            t = t0
            while True:
                nxt = [u for u in seg_of.get(e, []) if u != t and u not in seen]
                if not nxt:
                    break
                t = nxt[0]
                seen.add(t)
                e2 = [k for k in tri_edges[t] if k != e]
                if not e2:
                    break
                if direction == 1:
                    chain_t.append(t)
                    chain_e.append(e2[0])
                else:
                    chain_t.insert(0, t)
                    chain_e.insert(0, e2[0])
                e = e2[0]
        ii = np.array([k[0] for k in chain_e])
        jj = np.array([k[1] for k in chain_e])
        tt = (f[ii] / (f[ii] - f[jj]))[:, None]
        pts = P[ii] + tt * (P[jj] - P[ii])
        W = sh.W[ii] + tt * (sh.W[jj] - sh.W[ii])
        tn = nrm[np.array(chain_t)]                    # a triangle between each two crossings: the point takes their mean
        nn = _unit(np.vstack([tn[:1], 0.5 * (tn[1:] + tn[:-1]), tn[-1:]]))
        fl = {k: sh.fields[k][ii] + tt[:, 0] * (sh.fields[k][jj] - sh.fields[k][ii]) for k in fields}
        out.append((pts, nn, W, fl))
    return out


def clip_line(line, keep):
    """The runs of a line (points, normals, W, fields) where the bool mask `keep` (per point) holds, as lines."""
    pts, nn, W, fl = line
    out, start = [], None
    for k in range(len(pts) + 1):
        on = k < len(pts) and bool(keep[k])
        if on and start is None:
            start = k
        elif not on and start is not None:
            if k - start >= 2:
                sl = slice(start, k)
                out.append((pts[sl], nn[sl], W[sl], {n: v[sl] for n, v in fl.items()}))
            start = None
    return out


def arcs(sh, P, f, value, keep, fields=(), period=None, tris=None):
    """The pieces of the level line f == value where `keep` (a function of the line's {field: (k,)}) is true: a fold
    drawn over part of a ring, a crease only behind the knee."""
    out = []
    for ln in isolines(sh, P, f, value, fields=fields, period=period, tris=tris):
        out += clip_line(ln, keep(ln[3]))
    return out


def ribs(sh, P, u, spacing, across, lo, hi, rng, tris=None, inset=0.06, broken=0.3):
    """Rib lines across a ribbed piece, drawn by hand: along the piece's middle (the level line of the shell's field
    `across` halfway between `lo` and `hi`) a line every `spacing` metres (each moved by up to a fifth of it) follows
    the periodic coordinate `u` (a shell field, a turn = 1) across the piece, kept where `across` lies between lo and
    hi with each end in by up to `inset` of that depth; with chance `broken` a line has a short gap. Only the triangles
    in `tris` (default all) are drawn on, so `u` need only be a coordinate there."""
    out = []
    depth = hi - lo
    f, uf = sh.fields[across], sh.fields[u]
    on = np.ones(len(sh.tris), bool) if tris is None else np.asarray(tris, bool)
    cs = {"_c": np.cos(2 * np.pi * uf), "_s": np.sin(2 * np.pi * uf)}  # u carried round the turn without a wrap
    sh.fields.update(cs)
    mids = isolines(sh, P, f, 0.5 * (lo + hi), fields=tuple(cs), tris=on)
    for key in cs:
        sh.fields.pop(key)
    for pts, _, _, fl in mids:
        s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))])
        n = int(s[-1] / spacing)
        for k in range(n):
            sk = (k + 0.5 + rng.uniform(-0.2, 0.2)) * s[-1] / n
            at = np.array([np.interp(sk, s, pts[:, i]) for i in range(3)])
            val = float(np.arctan2(np.interp(sk, s, fl["_s"]), np.interp(sk, s, fl["_c"])) / (2 * np.pi)) % 1.0
            lines = isolines(sh, P, uf, val, fields=(across,), period=1.0, tris=on)
            if not lines:
                continue
            ln = min(lines, key=lambda l_: float(np.linalg.norm(l_[0] - at, axis=1).min()))
            a0, a1 = lo + depth * rng.uniform(0.0, inset), hi - depth * rng.uniform(0.0, inset)
            a = ln[3][across]
            keep = (a >= a0) & (a <= a1)
            if rng.uniform() < broken:
                keep &= np.abs(a - rng.uniform(a0, a1)) > 0.06 * depth
            out += clip_line(ln, keep)
    return out


def dashes(lines, rng, on, off):
    """The lines broken by hand: pieces `on` (lo, hi) metres long with gaps `off` (lo, hi) between them."""
    out = []
    for ln in lines:
        pts = ln[0]
        s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))])
        keep = np.zeros(len(pts), bool)
        x = -rng.uniform(0.0, on[1])
        while x < s[-1]:
            d = rng.uniform(*on)
            keep |= (s >= x) & (s <= x + d)
            x += d + rng.uniform(*off)
        out += clip_line(ln, keep)
    return out


def strokes(bones, lines, width, lift=0.0004, taper=0.5, mat=0, tag="ink"):
    """An outfit_geo.Patch of pen strokes along `lines` (points, normals, weights over `bones`, ...): ribbons `width`
    wide (scalar or one per line) lying `lift` above the cloth, narrowing to points at both ends (the width goes as
    sin(pi t) ** taper along each line). Faces look out along the normals. A line's repeated points are dropped, and a
    line shorter than its width is no stroke. None without a stroke."""
    widths = np.broadcast_to(np.asarray(width, float), (len(lines),))
    verts, faces, uvs, Ws = [], [], [], []
    nv = 0
    for (pts, nn, W, _), wd in zip(lines, widths):
        step = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        keep = np.concatenate([[True], step > 1e-7])
        pts, nn, W = pts[keep], nn[keep], W[keep]
        k = len(pts)
        s = np.concatenate([[0.0], np.cumsum(step[step > 1e-7])])
        if k < 2 or s[-1] < wd:
            continue
        tg = _unit(np.gradient(pts, axis=0))
        sd = _unit(np.cross(nn, tg))
        u = s / s[-1]
        half = 0.5 * wd * np.maximum(np.clip(np.sin(np.pi * u), 0.0, 1.0) ** taper, 0.08)
        base = pts + nn * lift
        verts += [base - sd * half[:, None], base + sd * half[:, None]]
        Ws += [W, W]
        for a in range(k - 1):
            faces.append((nv + a, nv + a + 1, nv + a + 1 + k, nv + a + k))
            uvs += [(u[a], 0.0), (u[a + 1], 0.0), (u[a + 1], 1.0), (u[a], 1.0)]
        nv += 2 * k
    if not faces:
        return None
    Wm = np.concatenate(Ws)
    w = {b: Wm[:, i] for i, b in enumerate(bones) if Wm[:, i].max() > 1e-6}
    return G.Patch(np.concatenate(verts), faces, np.array(uvs), w, mat, tag)
