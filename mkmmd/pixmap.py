"""Pixel maps (mk look --map, mk measure, mk mark): what is at each pixel of a rendered image, from the scene's own
geometry rasterised through the camera the image was rendered with.

The Blender side (mkmmd.blender.ops_look) hands over, per image, its camera and its frame's geometry: every mesh that
renders, as it renders (`posed`: world metres) and at rest (`rest`: its mesh's own coordinates before its armature, shape
keys and modifiers, the model's space; NaN where a modifier changes the vertex count), in triangles with their object,
material and polygon. `rasterise` finds the nearest triangle on each pixel's centre ray (both faces, every material,
transparent ones too) and a `Map` holds the result with the geometry and the camera; it is saved beside its image as
<image stem>.map.npz:

  tri (h, w) int32        the triangle at each pixel (-1: nothing)
  bary (h, w, 2) float32  its barycentric weights of the triangle's 2nd and 3rd corners, exact on the pixel's ray
  depth (h, w) float32    metres along the camera's view axis (inf: nothing)
  posed, rest (n, 3)      the vertices (float32)
  tris (t, 3)             the triangles, with tri_object, tri_material and tri_face (t,): indices into `objects` and
                          `materials` (names) and the polygon of the object's mesh each comes from
  camera                  JSON (`Camera`): matrix (4x4 world), type, lens (mm), sensor (w, h mm), fit, shift (x, y),
                          ortho_scale, clip_start, size (w, h px)

Pixels are (x, y) from the image's top left, x right, y down: pixel (x, y) covers [x, x + 1) x [y, y + 1) and is sampled
at its centre. The camera follows Blender: the sensor fit (AUTO: the sensor's width on the image's longer side), the lens
shift in units of the fitted side (the view window moves, so the picture moves the other way), perspective or
orthographic. Ranges of pixels (`span`) are inclusive."""
import json
from pathlib import Path

import numpy as np

CHUNK = 1 << 20                      # candidate (triangle, pixel) pairs tested at once


class MapError(ValueError):
    pass


class Camera:
    """A camera as an image saw it (`info`: the dict described in the module docstring)."""

    def __init__(self, info):
        self.info = dict(info)
        M = np.asarray(info["matrix"], float)
        self.R = M[:3, :3] / np.linalg.norm(M[:3, :3], axis=0, keepdims=True)    # columns: right, up, back
        self.t = M[:3, 3]
        self.w, self.h = (int(v) for v in info["size"])
        self.ortho = info.get("type", "PERSP") == "ORTHO"
        fit = info.get("fit", "AUTO")
        sw, sh = info.get("sensor", (36.0, 24.0))
        if fit == "VERTICAL":
            side, sensor = self.h, float(sh)
        elif fit == "HORIZONTAL":
            side, sensor = self.w, float(sw)
        else:
            side, sensor = max(self.w, self.h), float(sw)
        sx, sy = info.get("shift", (0.0, 0.0))
        self.cx = self.w / 2 - float(sx) * side          # where the view axis meets the picture (px from the top left)
        self.cy = self.h / 2 + float(sy) * side
        # perspective: pixels per metre at a metre's depth; orthographic: pixels per metre
        self.f = side / float(info["ortho_scale"]) if self.ortho else float(info["lens"]) / sensor * side
        self.near = float(info.get("clip_start", 1e-3))

    def to_camera(self, P):
        """(n, 3) points in the camera's axes: x right, y up, looking down -z."""
        return (np.asarray(P, float).reshape(-1, 3) - self.t) @ self.R

    def project(self, P):
        """(x, y, depth) of world points P (n, 3): pixel coordinates (from the top left) and metres along the view axis
        (not positive: behind the camera)."""
        q = self.to_camera(P)
        depth = -q[:, 2]
        if self.ortho:
            return self.cx + self.f * q[:, 0], self.cy - self.f * q[:, 1], depth
        with np.errstate(divide="ignore", invalid="ignore"):
            return self.cx + self.f * q[:, 0] / depth, self.cy - self.f * q[:, 1] / depth, depth

    def rays(self, px, py):
        """(origins, directions) (k, 3) in the camera's axes of the rays through the centres of pixels (px, py); a
        perspective ray's direction has z = -1, so its parameter at a hit is the depth there."""
        a = (np.asarray(px, float) + 0.5 - self.cx) / self.f
        b = (self.cy - np.asarray(py, float) - 0.5) / self.f
        if self.ortho:
            o = np.stack([a, b, np.zeros_like(a)], 1)
            d = np.zeros_like(o)
            d[:, 2] = -1.0
            return o, d
        d = np.stack([a, b, -np.ones_like(a)], 1)
        return np.zeros_like(d), d

    def m_per_px(self, depth):
        """Metres one pixel spans at `depth` (metres along the view axis)."""
        depth = np.asarray(depth, float)
        return np.full(depth.shape, 1.0 / self.f) if self.ortho else depth / self.f


def _boxes(cam, Q):
    """Per triangle (corners Q (t, 3, 3) in camera axes): the inclusive pixel box (x0, y0, x1, y1) whose centres it may
    cover, empty (x0 > x1) when it covers none or lies wholly behind the camera; a perspective triangle that reaches
    behind the camera gets the whole picture (its corners have no place on it)."""
    w, h = cam.w, cam.h
    depth = -Q[:, :, 2]
    if cam.ortho:
        sx, sy = cam.cx + cam.f * Q[:, :, 0], cam.cy - cam.f * Q[:, :, 1]
    else:
        with np.errstate(divide="ignore", invalid="ignore"):
            sx, sy = cam.cx + cam.f * Q[:, :, 0] / depth, cam.cy - cam.f * Q[:, :, 1] / depth
    x0 = np.ceil(np.nanmin(sx, 1) - 0.5)
    x1 = np.floor(np.nanmax(sx, 1) - 0.5)
    y0 = np.ceil(np.nanmin(sy, 1) - 0.5)
    y1 = np.floor(np.nanmax(sy, 1) - 0.5)
    front = depth > cam.near
    if not cam.ortho:
        part = front.any(1) & ~front.all(1)
        x0[part], y0[part], x1[part], y1[part] = 0, 0, w - 1, h - 1
    gone = ~front.any(1)
    x0[gone], y0[gone], x1[gone], y1[gone] = 0, 0, -1, -1
    x0, y0 = np.clip(np.nan_to_num(x0), 0, w), np.clip(np.nan_to_num(y0), 0, h)
    x1, y1 = np.clip(np.nan_to_num(x1, nan=-1.0), -1, w - 1), np.clip(np.nan_to_num(y1, nan=-1.0), -1, h - 1)
    return x0.astype(np.int64), y0.astype(np.int64), x1.astype(np.int64), y1.astype(np.int64)


def rasterise(P, T, cam, chunk=CHUNK):
    """(tri (h, w) int32, bary (h, w, 2) float32, depth (h, w) float32) of the triangles T (t, 3) over the world points
    P (n, 3) seen through Camera `cam`: at each pixel the nearest triangle its centre ray meets in front of the camera."""
    w, h = cam.w, cam.h
    T = np.asarray(T, np.int64).reshape(-1, 3)
    Q = cam.to_camera(P)[T]                                     # (t, 3 corners, 3)
    x0, y0, x1, y1 = _boxes(cam, Q)
    bw, bh = x1 - x0 + 1, y1 - y0 + 1
    keep = np.flatnonzero((bw > 0) & (bh > 0))
    cnt = bw[keep] * bh[keep]
    cum = np.cumsum(cnt)
    total = int(cum[-1]) if len(cum) else 0
    A = Q[keep, 0]
    e1, e2 = Q[keep, 1] - A, Q[keep, 2] - A
    zbuf = np.full(w * h, np.inf)
    tbuf = np.full(w * h, -1, np.int64)
    ubuf = np.zeros(w * h, np.float32)
    vbuf = np.zeros(w * h, np.float32)
    eps = 1e-9
    for g0 in range(0, total, chunk):
        g = np.arange(g0, min(g0 + chunk, total), dtype=np.int64)
        k = np.searchsorted(cum, g, side="right")                # which kept triangle
        off = g - (cum[k] - cnt[k])
        ti = keep[k]
        px = x0[ti] + off % bw[ti]
        py = y0[ti] + off // bw[ti]
        o, d = cam.rays(px, py)
        E1, E2 = e1[k], e2[k]
        pv = np.cross(d, E2)
        det = np.einsum("ij,ij->i", E1, pv)
        ok = np.abs(det) > 1e-15
        inv = np.where(ok, 1.0 / np.where(ok, det, 1.0), 0.0)
        tv = o - A[k]
        u = np.einsum("ij,ij->i", tv, pv) * inv
        qv = np.cross(tv, E1)
        v = np.einsum("ij,ij->i", d, qv) * inv
        t = np.einsum("ij,ij->i", E2, qv) * inv
        hit = ok & (u >= -eps) & (v >= -eps) & (u + v <= 1.0 + eps) & (t > cam.near)
        if not hit.any():
            continue
        pix = (py * w + px)[hit]
        t, u, v, ti = t[hit], u[hit], v[hit], ti[hit]
        order = np.lexsort((t, pix))                              # nearest first within each pixel
        ps = pix[order]
        first = np.ones(len(ps), bool)
        first[1:] = ps[1:] != ps[:-1]
        sel = order[first]
        up = t[sel] < zbuf[pix[sel]]
        sel = sel[up]
        p = pix[sel]
        zbuf[p], tbuf[p], ubuf[p], vbuf[p] = t[sel], ti[sel], u[sel], v[sel]
    bary = np.stack([ubuf, vbuf], 1).reshape(h, w, 2)
    return tbuf.reshape(h, w).astype(np.int32), bary, zbuf.reshape(h, w).astype(np.float32)


def parse_span(spec, limit, what):
    """Pixel indices of an inclusive range "A:B" or "A:B:STEP" (or one index "A"), kept inside 0 .. limit - 1."""
    try:
        parts = [int(float(p)) for p in str(spec).split(":")]
    except ValueError:
        raise MapError(f"{what} {spec!r}: want A, A:B or A:B:STEP (pixels)") from None
    if len(parts) == 1:
        parts = [parts[0], parts[0]]
    if len(parts) not in (2, 3) or (len(parts) == 3 and parts[2] <= 0):
        raise MapError(f"{what} {spec!r}: want A, A:B or A:B:STEP (pixels, STEP > 0)")
    a, b = sorted(parts[:2])
    out = np.arange(max(a, 0), min(b, limit - 1) + 1, parts[2] if len(parts) == 3 else 1)
    if not len(out):
        raise MapError(f"{what} {spec!r} lies outside the image (0 .. {limit - 1})")
    return out


def _runs(row):
    """Inclusive (start, end) of the runs of True in a 1-D bool array."""
    d = np.diff(np.concatenate([[0], row.astype(np.int8), [0]]))
    return list(zip(np.flatnonzero(d == 1).tolist(), (np.flatnonzero(d == -1) - 1).tolist()))


def _r(v, n=4):
    return None if v is None or not np.all(np.isfinite(v)) else np.round(np.asarray(v, float), n).tolist()


class Map:
    """A pixel map (module docstring): `make` it from geometry and a camera, `save` and `load` it, ask it questions."""

    def __init__(self, tri, bary, depth, geo, camera):
        self.tri, self.bary, self.depth = tri, bary, depth
        self.geo = geo
        self.cam = camera if isinstance(camera, Camera) else Camera(camera)
        self.h, self.w = tri.shape
        self.materials = [str(m) for m in geo["materials"]]
        self.objects = [str(o) for o in geo["objects"]]
        self._mat = None

    @classmethod
    def make(cls, geo, camera):
        cam = camera if isinstance(camera, Camera) else Camera(camera)
        tri, bary, depth = rasterise(geo["posed"], geo["tris"], cam)
        return cls(tri, bary, depth, geo, cam)

    @classmethod
    def load(cls, path):
        try:
            z = np.load(path, allow_pickle=False)
        except (OSError, ValueError) as e:
            raise MapError(f"{path}: not a pixel map ({e})") from None
        need = ("tri", "bary", "depth", "posed", "rest", "tris", "tri_object", "tri_material", "tri_face", "objects",
                "materials", "camera")
        miss = [k for k in need if k not in z.files]
        if miss:
            raise MapError(f"{path}: not a pixel map (missing {', '.join(miss)})")
        geo = {k: z[k] for k in need[3:11]}
        return cls(z["tri"], z["bary"], z["depth"], geo, json.loads(str(z["camera"])))

    def save(self, path):
        g = self.geo
        np.savez_compressed(path, tri=self.tri, bary=self.bary, depth=self.depth,
                            posed=np.asarray(g["posed"], np.float32), rest=np.asarray(g["rest"], np.float32),
                            tris=np.asarray(g["tris"], np.int32), tri_object=np.asarray(g["tri_object"], np.int32),
                            tri_material=np.asarray(g["tri_material"], np.int32),
                            tri_face=np.asarray(g["tri_face"], np.int32), objects=np.asarray(self.objects, str),
                            materials=np.asarray(self.materials, str), camera=json.dumps(self.cam.info))
        return str(path)

    # ---------------------------------------------------------------- what is where
    def material(self):
        """(h, w) material index at each pixel (-1: nothing)."""
        if self._mat is None:
            tm = np.asarray(self.geo["tri_material"])
            self._mat = np.where(self.tri >= 0, tm[np.maximum(self.tri, 0)], -1)
        return self._mat

    def object(self):
        """(h, w) object index at each pixel (-1: nothing)."""
        to = np.asarray(self.geo["tri_object"])
        return np.where(self.tri >= 0, to[np.maximum(self.tri, 0)], -1)

    def mask(self, names=None):
        """(h, w) bool: the pixels showing one of the materials `names` (a list or a comma string); None or "*": any
        surface. An unknown name is a MapError that lists the map's materials."""
        if names is None or names == "*" or names == ["*"]:
            return self.tri >= 0
        if isinstance(names, str):
            names = [n.strip() for n in names.split(",") if n.strip()]
        bad = [n for n in names if n not in self.materials]
        if bad:
            raise MapError(f"no material {', '.join(map(repr, bad))} in this map (it has: {', '.join(self.materials)})")
        return np.isin(self.material(), [self.materials.index(n) for n in names])

    def points(self, xs, ys, rest=False):
        """(k, 3) the surface point at pixels (xs, ys): where it renders (world), or with `rest` on the model at rest; NaN
        where the pixel shows nothing."""
        xs, ys = np.asarray(xs, int).reshape(-1), np.asarray(ys, int).reshape(-1)
        tri = self.tri[ys, xs]
        V = np.asarray(self.geo["rest" if rest else "posed"], float)
        T = np.asarray(self.geo["tris"])[np.maximum(tri, 0)]
        u, v = self.bary[ys, xs, 0].astype(float), self.bary[ys, xs, 1].astype(float)
        P = (1.0 - u - v)[:, None] * V[T[:, 0]] + u[:, None] * V[T[:, 1]] + v[:, None] * V[T[:, 2]]
        P[tri < 0] = np.nan
        return P

    def cm_per_px(self, xs, ys):
        d = self.depth[np.asarray(ys, int), np.asarray(xs, int)].astype(float)
        return np.where(np.isfinite(d), self.cam.m_per_px(np.where(np.isfinite(d), d, 1.0)) * 100.0, np.nan)

    def at(self, x, y):
        """What the pixel (x, y) shows: material, object, polygon, the point in the world and at rest, depth and the
        centimetres a pixel spans there."""
        self._inside(x, y)
        t = int(self.tri[y, x])
        if t < 0:
            return {"x": x, "y": y, "material": None}
        g = self.geo
        return {"x": x, "y": y, "material": self.materials[int(g["tri_material"][t])],
                "object": self.objects[int(g["tri_object"][t])], "face": int(g["tri_face"][t]),
                "world": _r(self.points([x], [y])[0]), "rest": _r(self.points([x], [y], rest=True)[0]),
                "depth": round(float(self.depth[y, x]), 4), "cm_per_px": round(float(self.cm_per_px([x], [y])[0]), 4)}

    def _inside(self, x, y):
        if not (0 <= x < self.w and 0 <= y < self.h):
            raise MapError(f"pixel ({x}, {y}) lies outside the image ({self.w} x {self.h})")

    def _between(self, a, b):
        """(cm in the world, cm at rest) between the surface points at pixels a and b (x, y)."""
        P = self.points([a[0], b[0]], [a[1], b[1]])
        R = self.points([a[0], b[0]], [a[1], b[1]], rest=True)
        return (round(float(np.linalg.norm(P[1] - P[0]) * 100), 2) if np.isfinite(P).all() else None,
                round(float(np.linalg.norm(R[1] - R[0]) * 100), 2) if np.isfinite(R).all() else None)

    def _seen(self, a, b, px):
        """Centimetres `px` pixels span as seen, at the mean depth of the pixels a and b (x, y)."""
        c = self.cm_per_px([a[0], b[0]], [a[1], b[1]])
        return round(float(px * np.nanmean(c)), 2) if np.isfinite(c).any() else None

    # ---------------------------------------------------------------- measurements
    def edges(self, names, xs, ys=None):
        """Per column x of `xs`: the first and last row (`top`, `bottom`) showing the materials `names` (within the rows
        `ys` when given), the rows between in pixels, the centimetres they span as seen (`seen_cm`: pixels times the
        centimetres a pixel spans at their depth) and between their surface points (`cm` in the world, `cm_rest` on the
        model at rest), and how many separate runs the column has (2 or more: something crosses the material there)."""
        m = self.mask(names)
        rows = np.arange(self.h) if ys is None else np.asarray(ys, int)
        out = []
        for x in np.asarray(xs, int).tolist():
            col = m[rows, x]
            if not col.any():
                out.append({"x": x, "top": None})
                continue
            hit = rows[col]
            top, bottom = int(hit[0]), int(hit[-1])
            cm, cm_rest = self._between((x, top), (x, bottom))
            out.append({"x": x, "top": top, "bottom": bottom, "px": bottom - top + 1,
                        "seen_cm": self._seen((x, top), (x, bottom), bottom - top + 1), "cm": cm, "cm_rest": cm_rest,
                        "runs": len(_runs(col)), "top_world": _r(self.points([x], [top])[0]),
                        "bottom_world": _r(self.points([x], [bottom])[0])})
        return out

    def extent(self, names, ys, xs=None):
        """Per row y of `ys`: the first and last column (`left`, `right`) showing the materials `names` (within the
        columns `xs` when given), the pixels between, the centimetres as seen and between their surface points (as
        `edges`) and the runs."""
        m = self.mask(names)
        cols = np.arange(self.w) if xs is None else np.asarray(xs, int)
        out = []
        for y in np.asarray(ys, int).tolist():
            row = m[y, cols]
            if not row.any():
                out.append({"y": y, "left": None})
                continue
            hit = cols[row]
            left, right = int(hit[0]), int(hit[-1])
            cm, cm_rest = self._between((left, y), (right, y))
            out.append({"y": y, "left": left, "right": right, "px": right - left + 1,
                        "seen_cm": self._seen((left, y), (right, y), right - left + 1), "cm": cm, "cm_rest": cm_rest,
                        "runs": len(_runs(row))})
        return out

    def gaps(self, ys, names=None, xs=None):
        """Per row y of `ys`: the gaps, runs of pixels that do not show the materials `names` (default: any surface, so
        a gap shows nothing) with such pixels on both sides (within the columns `xs` when given): [left, right] inclusive,
        their width in pixels, in centimetres as seen and between the surface points that bound them."""
        m = self.mask(names)
        cols = np.arange(self.w) if xs is None else np.asarray(xs, int)
        out = []
        for y in np.asarray(ys, int).tolist():
            row = m[y, cols]
            gs = []
            for a, b in _runs(~row):
                if a == 0 or b == len(cols) - 1:
                    continue                                       # open to the side: outside, not a gap
                l, r = int(cols[a]), int(cols[b])
                pa, pb = (int(cols[a - 1]), y), (int(cols[b + 1]), y)
                cm, _ = self._between(pa, pb)
                gs.append({"left": l, "right": r, "px": r - l + 1, "seen_cm": self._seen(pa, pb, r - l + 1), "cm": cm})
            out.append({"y": y, "gaps": gs})
        return out

    def distance(self, a, b):
        """Centimetres between the surface points at pixels a and b (x, y), in the world and at rest."""
        for p in (a, b):
            self._inside(*p)
        cm, cm_rest = self._between(a, b)
        return {"from": list(a), "to": list(b), "cm": cm, "cm_rest": cm_rest}

    def where(self, point):
        """Where a world point is in the image: pixel (x, y), depth, and whether it is hidden behind a surface there (or
        off the picture)."""
        x, y, d = (float(v[0]) for v in self.cam.project(np.asarray(point, float)[None]))
        ix, iy = int(np.floor(x)), int(np.floor(y))
        on = 0 <= ix < self.w and 0 <= iy < self.h and d > 0
        hidden = None
        if on:
            seen = float(self.depth[iy, ix])
            hidden = bool(np.isfinite(seen) and seen < d - 0.005)
        return {"point": [float(v) for v in point], "x": round(x, 2), "y": round(y, 2), "depth": round(d, 4),
                "on_picture": bool(on), "hidden": hidden}

    def summary(self):
        """The image size, the camera's depth at the picture's centre and every material seen: its pixels and box."""
        mat = self.material()
        mats = []
        for i, name in enumerate(self.materials):
            ys, xs = np.nonzero(mat == i)
            if len(xs):
                mats.append({"material": name, "pixels": len(xs),
                             "box": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]})
        mats.sort(key=lambda m: -m["pixels"])
        return {"size": [self.w, self.h], "camera": {k: self.cam.info.get(k) for k in ("type", "lens", "size")},
                "objects": self.objects, "materials": mats}


def map_path(image):
    """The map beside an image: <image stem>.map.npz."""
    p = Path(image)
    return p.with_name(p.stem + ".map.npz")
