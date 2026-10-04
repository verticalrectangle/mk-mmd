"""Anime eyelids: the face skin never moves for a blink. Each eye has two skin-coloured SHEETS (an upper and a lower lid)
tucked behind the lid margins when the eye is open; a lid morph unrolls them over the eyeball toward a closed-lid curve
(they clip the eye graphics like real lids, so a half blink shows the iris cut by the lid). The lash band, the lower lash
and the double-eyelid crease are strips that ride on the lids and reshape into the closed-eye drawing: an arc for a
blink, a "^" arch for the smile, ">" / "<" chevrons for hau, flat dashes for calm, a straight half-lid line for jito.

Eye-frame coordinates (u, v) are relative to the iris centre (u outward, v up), as in head_eye."""
import numpy as np

from . import head_eye as EY
from . import head_skin as SK

SHEET = dict(gap=0.0018, nu=26, rows=7, tuck=0.0010, sliver=0.00006)
DISPLAY_GAP = 0.0021                    # strips ride this far in front of the eyeball shell (just over the sheets)
CLOSED_WIDTH = 0.0054                   # lash band thickness when the eye is drawn closed


def smooth(t):
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def strip_keep(k, tip=False, ends_pointed=False):
    """Which of the 2k interleaved strip vertices exist (same convention as head.bind_strip)."""
    keep = np.ones(2 * k, bool)
    if tip:
        keep[2 * k - 2] = False
    if ends_pointed:
        keep[0] = False
        keep[2 * k - 2] = False
    return keep


def _tangent_normal(P):
    t = np.gradient(P, axis=0)
    t /= np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-12)
    return np.stack([-t[:, 1], t[:, 0]], -1)


class Lids:
    """Sheets and strip targets of one eye."""

    def __init__(self, shape, eye, nu=None):
        self.shape, self.e = shape, eye
        self.sg = eye["sign"]
        self.cx, self.cz = eye["centre"]
        self.bar, self.alm, self.dip = eye["barrel"], eye["alm"], eye["dip"]
        up, lo = EY.lid_curves(600)
        self.u_in, self.u_out = float(up[0, 0]), float(up[-1, 0])
        self.up, self.lo = self._fn(up), self._fn(lo)
        self.nu = nu or SHEET["nu"]
        self.s = np.linspace(0.0, 1.0, self.nu)
        self.u = self.u_in + self.s * (self.u_out - self.u_in)
        self.vU, self.vL = self.up(self.u), self.lo(self.u)
        self.chord = EY.CORNER_IN[1] + (self.u - self.u_in) * (EY.CORNER_OUT[1] - EY.CORNER_IN[1]) / (self.u_out - self.u_in)

    @staticmethod
    def _fn(c):
        o = np.argsort(c[:, 0])
        u, v = c[o, 0], c[o, 1]
        return lambda x: np.interp(x, u, v)

    # ---- positions on the eye
    def xz(self, u, v):
        return self.cx + self.sg * np.asarray(u, float), self.cz + np.asarray(v, float)

    def sheet_points(self, v, gap=None):
        x, z = self.xz(np.tile(self.u, len(v) // self.nu), v)
        return self.bar.point(x, z, SHEET["gap"] if gap is None else gap)

    def disp(self, u, v, h):
        """Points for strips: on the lid shell (just over the sheets) inside the opening, on the skin outside, blended
        over 4 mm; `h` over the surface."""
        u, v = np.asarray(u, float), np.asarray(v, float)
        x, z = self.xz(u, v)
        Pb = self.bar.point(x, z, DISPLAY_GAP + (h - 0.0004))
        Ps = SK.lift(self.shape, x, z)
        Ps[:, 1] += self.dip(x, z) - h
        ang = np.arctan2(v - self.alm.c[1], u - self.alm.c[0])
        out = np.maximum(np.hypot(u - self.alm.c[0], v - self.alm.c[1]) - np.interp(ang, self.alm.a, self.alm.r), 0.0)
        b = smooth(out / 0.004)[:, None]
        return Pb + b * (Ps - Pb)

    # ---- the sheets
    def sheet_rest_and_faces(self, which):
        """(rest vertices (rows+1)*nu, faces, v_top) of the upper or lower sheet: all rows lie on the margin (a hair
        apart) until a morph unrolls them."""
        K = SHEET["rows"]
        sign = -1.0 if which == "upper" else 1.0
        v_top = (self.vU + SHEET["tuck"]) if which == "upper" else (self.vL - SHEET["tuck"])
        rows = [v_top + sign * SHEET["sliver"] * i for i in range(K + 1)]
        P = self.sheet_points(np.concatenate(rows))
        F = [(i * self.nu + j, i * self.nu + j + 1, (i + 1) * self.nu + j + 1, (i + 1) * self.nu + j)
             for i in range(K) for j in range(self.nu - 1)]
        return P, F, v_top

    def sheet_target(self, which, v_top, edge):
        """Vertices when the sheet's free edge lies on the curve `edge(u)` (rows evenly between the tucked top and it)."""
        K = SHEET["rows"]
        rows = [v_top + (i / K) * (edge - v_top) for i in range(K + 1)]
        return self.sheet_points(np.concatenate(rows))

    # ---- closed-eye curves (edge of the sheets and the lash path)
    def arc(self, depth, power=0.9):
        """A shallow arc below the corner chord (a closed eye): depth > 0 sags, depth < 0 arches ("^")."""
        c = self.chord - depth * np.sin(np.pi * self.s) ** power
        return np.clip(c, self.vL + 0.0015, self.vU - 0.0015)

    def flat(self, v0):
        return np.clip(np.full_like(self.u, v0), self.vL, self.vU)

    # ---- the band path
    def band_path(self, kind, s, edge=None):
        """Centre line (n, 2) and half-width scale of the upper lash for cross-section parameters s (0..1 along the
        lid, > 1 along the wing): kind 'arc' (follows `edge`), 'chevron', 'dash'."""
        u0, u1 = self.u_in, self.u_out
        P = np.empty((len(s), 2))
        body = s <= 1.0
        if kind == "arc":
            u = u0 + s[body] * (u1 - u0)
            P[body] = np.stack([u, np.interp(u, self.u, edge)], -1)
        elif kind == "dash":
            u = u0 + 0.003 + s[body] * (u1 - u0 - 0.004)
            P[body] = np.stack([u, edge[0] + 0.0010 * (2 * s[body] - 1) ** 2], -1)
        elif kind == "chevron":                         # apex toward the nose, strokes opening outward
            lo_tip = np.array([u1 - 0.0030, -0.0128])
            apex = np.array([u0 + 0.0085, -0.0010])
            hi_tip = np.array([u1 - 0.0030, 0.0112])
            k = s[body]
            A = lo_tip + (apex - lo_tip) * (k / 0.5)[:, None]
            B = apex + (hi_tip - apex) * ((k - 0.5) / 0.5)[:, None]
            Pc = np.where((k < 0.5)[:, None], A, B)
            dense = np.stack([np.interp(np.linspace(0, 1, 200), k, Pc[:, c]) for c in (0, 1)], -1)
            for _ in range(10):                          # round the apex so the strip does not fold there
                dense[1:-1] = 0.25 * dense[:-2] + 0.5 * dense[1:-1] + 0.25 * dense[2:]
            P[body] = np.stack([np.interp(k, np.linspace(0, 1, 200), dense[:, c]) for c in (0, 1)], -1)
        w = ~body
        if w.any():
            e = P[body][-1]
            flick = np.array([0.0120, 0.0055]) if kind != "chevron" else np.array([0.0030, 0.0020])
            P[w] = e + ((s[w] - 1.0) / 0.30)[:, None] * flick
        return P

    def rest_width(self, s):
        """Thickness of the open eye's lash band at the lid parameters s."""
        up = EY.UPPER_LASH
        prof = np.interp(np.minimum(s, 1.0), [p for p, _ in up["profile"]], [v for _, v in up["profile"]])
        return up["width"] * prof

    def band_width(self, s, scale=1.0, rest=False):
        """Band thickness: the closed-eye drawing's, or with rest=True the open eye's (a lowered but open lid)."""
        prof = np.interp(np.minimum(s, 1.0), [0.0, 0.10, 0.30, 0.55, 0.85, 1.0], [0.30, 0.62, 0.92, 1.0, 0.95, 0.80])
        taper = np.where(s > 1.0, np.clip(1.0 - (s - 1.0) / 0.30, 0.0, 1.0), 1.0)
        if rest:
            return self.rest_width(s) * taper
        return CLOSED_WIDTH * scale * prof * taper


# ----------------------------------------------------------------------------------------------- the lid morph book
def morph_table(L):
    """name -> dict(up, lo: sheet edges (nu,), band: (kind, edge/params), crease, lower)."""
    blink = L.arc(0.0062)
    smile = L.arc(-0.0058)

    def lowered(inner, outer, mid=0.0):
        """Upper lid edge lowered by `inner` (at the nose) .. `outer` (at the temple), `mid` more in the middle."""
        d = inner + (outer - inner) * L.s + mid * np.sin(np.pi * L.s)
        return np.clip(L.vU - d, L.vL + 0.0016, L.vU)

    def raised_lower(k):
        return L.vL + k * np.sin(np.pi * L.s) ** 0.8

    def mood(up, lo=None):
        lo = L.vL if lo is None else np.minimum(lo, up - 0.0016)
        return dict(up=up, lo=lo, band=("hang", up), crease="follow", lower="lift")
    return {
        "blink": dict(up=blink, lo=blink, band=("arc", blink), crease="follow", lower="merge"),
        "smile_eyes": dict(up=smile, lo=smile, band=("arc", smile), crease="follow", lower="merge"),
        "hau": dict(up=blink, lo=blink, band=("chevron", None), crease="fade", lower="merge"),
        "calm": dict(up=L.flat(-0.0050), lo=L.flat(-0.0050), band=("dash", [-0.0050]), crease="fade", lower="merge"),
        "jito": dict(up=L.flat(0.0004), lo=L.vL, band=("arc", L.flat(0.0004)), crease="follow", lower="keep"),
        "surprised": dict(up=L.vU, lo=L.vL, band=("lift", 0.0030), crease="lift", lower="drop"),
        # moods that read through the fringe: the upper lid droops (troubled, sad), flattens (serious), glares (angry)
        "troubled": mood(lowered(0.0006, 0.0046), raised_lower(0.0008)),
        "sad": mood(lowered(0.0010, 0.0066), raised_lower(0.0014)),
        "serious": mood(lowered(0.0016, 0.0016, 0.0012)),
        "angry": mood(lowered(0.0062, 0.0), raised_lower(0.0006)),
        "cheerful": mood(lowered(0.0004, 0.0012), raised_lower(0.0030)),
    }


def _strip_points(lo, hi, tip=False, ends_pointed=False):
    pts = np.empty((2 * len(lo), 2))
    pts[0::2], pts[1::2] = lo, hi
    return pts[strip_keep(len(lo), tip, ends_pointed)]


def lid_morphs(M, H):
    """Fill the book with the lid morphs: sheet unrolling and strip reshaping per side. `H`: lids {side: Lids},
    sheets {side: dict(upper=(rows, v_top), lower=(rows, v_top))}, lids_mesh, lines, decals, strips {piece: dict}."""
    from .head_morph import NAMES, piece_rows
    book = M.book
    tables = {sd: morph_table(L) for sd, L in H.lids.items()}
    plans = [("blink", "blink", ("L", "R")), ("smile_eyes", "smile_eyes", ("L", "R")), ("hau", "hau", ("L", "R")),
             ("calm", "calm", ("L", "R")), ("jito", "jito", ("L", "R")), ("surprised", "surprised", ("L", "R")),
             ("wink_l", "blink", ("L",)), ("wink_r", "blink", ("R",)), ("wink2_l", "smile_eyes", ("L",)),
             ("wink2_r", "smile_eyes", ("R",)), ("cheerful", "cheerful", ("L", "R")), ("serious", "serious", ("L", "R")),
             ("troubled", "troubled", ("L", "R")), ("sad", "sad", ("L", "R")), ("angry", "angry", ("L", "R"))]
    for key, kind, sides in plans:
        name = NAMES[key][0]
        for sd in sides:
            L, spec = H.lids[sd], tables[sd][kind]
            # sheets
            for which in ("upper", "lower"):
                rows, v_top = H.sheets[sd][which]
                tgt = L.sheet_target(which, v_top, spec["up"] if which == "upper" else spec["lo"])
                delta = tgt - np.asarray(H.lids_mesh.V)[rows]
                if np.abs(delta).max() > 0:
                    book.add(name, "lids", delta, rows)
            # strips
            lash = H.strips[f"lash.{sd}"]
            s = lash["s"]
            bk, bp = spec["band"]
            if bk == "lift":
                P = (lash["lo"] + lash["hi"]) / 2
                lo_t = lash["lo"] + np.array([0.0, bp])
                hi_t = lash["hi"] + np.array([0.0, bp])
                half = None
            elif bk == "hang":                              # an open but lowered lid: the band hangs from the new margin
                path0 = L.band_path("arc", s, bp)
                w = L.band_width(s, rest=True)
                n = _tangent_normal(path0)
                lo_t, hi_t = path0, path0 + w[:, None] * n
                half = (path0 + 0.5 * w[:, None] * n, w, n)
            else:
                path = L.band_path(bk, s, bp)
                w = L.band_width(s)
                n = _tangent_normal(path)
                lo_t, hi_t = path - 0.5 * w[:, None] * n, path + 0.5 * w[:, None] * n
                half = (path, w, n)
            _move(book, name, H, f"lash.{sd}", L, lash, lo_t, hi_t, piece_rows)
            # lower lash
            ll = H.strips[f"lashlow.{sd}"]
            if spec["lower"] == "merge" and half is not None:
                path, w, n = half
                q = np.interp(ll["s"], s, np.arange(len(s)))
                pm = np.stack([np.interp(q, np.arange(len(s)), path[:, k]) for k in (0, 1)], -1)
                nm = np.stack([np.interp(q, np.arange(len(s)), n[:, k]) for k in (0, 1)], -1)
                wm = np.interp(q, np.arange(len(s)), w)
                base = pm - 0.5 * wm[:, None] * nm
                _move(book, name, H, f"lashlow.{sd}", L, ll, base, base, piece_rows)
            elif spec["lower"] == "lift":                   # the lower lash rides up with the lower lid
                dv = np.interp(ll["lo"][:, 0], L.u, spec["lo"] - L.vL)
                off = np.stack([np.zeros_like(dv), dv], -1)
                _move(book, name, H, f"lashlow.{sd}", L, ll, ll["lo"] + off, ll["hi"] + off, piece_rows)
            elif spec["lower"] == "drop":
                _move(book, name, H, f"lashlow.{sd}", L, ll, ll["lo"] + [0.0, -0.0006], ll["hi"] + [0.0, -0.0006], piece_rows)
            # crease
            cr = H.strips[f"crease.{sd}"]
            if spec["crease"] == "follow" and half is not None:
                path, w, n = half
                q = np.interp(cr["s"], s, np.arange(len(s)))
                pm = np.stack([np.interp(q, np.arange(len(s)), path[:, k]) for k in (0, 1)], -1)
                nm = np.stack([np.interp(q, np.arange(len(s)), n[:, k]) for k in (0, 1)], -1)
                wm = np.interp(q, np.arange(len(s)), w)
                lo_c = pm + nm * (0.5 * wm + 0.0024)[:, None]
                width = np.linalg.norm(cr["hi"] - cr["lo"], axis=1, keepdims=True)
                _move(book, name, H, f"crease.{sd}", L, cr, lo_c, lo_c + nm * width, piece_rows)
            elif spec["crease"] == "lift":
                _move(book, name, H, f"crease.{sd}", L, cr, cr["lo"] + [0.0, bp], cr["hi"] + [0.0, bp], piece_rows)
            else:                                           # fade: the strip collapses to its centre line
                mid = 0.5 * (cr["lo"] + cr["hi"])
                _move(book, name, H, f"crease.{sd}", L, cr, mid, mid, piece_rows)


def _move(book, name, H, piece, L, st, lo_t, hi_t, piece_rows):
    """Delta of a strip piece (rows of the lines mesh) from its rest positions to the target edges (u, v)."""
    dec = H.decals[piece]
    rest = dec.positions()
    pts = _strip_points(np.asarray(lo_t, float), np.asarray(hi_t, float), st["tip"], st["ends_pointed"])
    tgt = L.disp(pts[:, 0], pts[:, 1], st["h"])
    delta = tgt - rest
    book.add(name, "lines", delta, piece_rows(H.lines, piece))
