"""Anime eyelids: the face skin never moves for a blink. Each eye has two skin-coloured SHEETS (an upper and a lower lid)
tucked behind the lid margins when the eye is open; a lid morph unrolls them over the eyeball toward a closed-lid curve
(they clip the eye graphics like real lids, so a half blink shows the iris cut by the lid). The lines above the eye (the
lash band with its fork and wing, the double-eyelid crease) and the lower lash are strips that ride on the lids and
reshape into the closed-eye drawing: an arc for a blink, a "^" arch for the smile, ">" / "<" chevrons for hau, flat dashes
for calm, a straight half-lid line for jito; the fork and the wing fold into the drawn line. When the eye stays open (a
lowered lid in the moods, the surprised eye) everything above it moves with the band.

Eye-frame coordinates (u, v) are relative to the iris centre (u outward, v up), as in head_eye."""
import numpy as np

from . import head_decal as DC
from . import head_eye as EY
from . import head_skin as SK

SHEET = dict(gap=EY.SHEET_GAP, nu=30, rows=7, tuck=0.0010, sliver=0.00006)
DISPLAY_GAP = 0.0023                    # strips ride this far in front of the eyeball shell: clear of the sheets' faces
CLOSED_WIDTH = 0.0054                   # lash band thickness when the eye is drawn closed
CLOSED_FLICK = (0.0058, -0.0046)        # the drawn line ends in a flick down and out (the band's run down the outer side;
                                        # for UPPER_LASH's out_v: closed_flick scales it with a spec's) ...
CLOSED_TAIL = (-0.0050, 0.0005)         # ... and starts in a point this far from the inner corner (the band's tail)
ROUND = 200                             # smoothing passes that round the drawn line where it turns (Lids.band_path)
OUT_LIFT = 0.0007                       # morphed strips stand this much further off the skin outside the opening (Lids.disp)


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


def closed_flick(cfg):
    """The closed eye's flick for the [head] table `cfg`: CLOSED_FLICK scaled with how far the open band runs down the eye's
    outer side (its lash out_v against UPPER_LASH's), so a band that drops less closes into a shorter flick."""
    out_v = float(((cfg or {}).get("lash") or {}).get("out_v", EY.UPPER_LASH["out_v"]))
    k = out_v / EY.UPPER_LASH["out_v"]
    return (CLOSED_FLICK[0] * k, CLOSED_FLICK[1] * k)


class Lids:
    """Sheets and strip targets of one eye."""

    def __init__(self, shape, eye, nu=None, flick=CLOSED_FLICK):
        self.shape, self.e = shape, eye
        self.flick = tuple(flick)                     # the closed line's flick (closed_flick)
        self.sg = eye["sign"]
        self.cx, self.cz = eye["centre"]
        self.bar, self.alm, self.dip = eye["lid_shell"], eye["alm"], eye["dip"]
        up, lo = EY.lid_curves(600)
        self.u_in, self.u_out = float(up[0, 0]), float(up[-1, 0])
        self.up, self.lo = self._fn(up), self._fn(lo)
        self.nu = nu or SHEET["nu"]
        # sheet columns spaced along the lids (the longer of the two each step), not evenly in u: the upper lid's outer side
        # drops almost straight down, and the tucked sheet between two far columns would cut across the opening
        ug = np.linspace(self.u_in, self.u_out, 4001)
        step = np.maximum(np.hypot(np.diff(ug), np.diff(self.up(ug))), np.hypot(np.diff(ug), np.diff(self.lo(ug))))
        arc = np.concatenate([[0.0], np.cumsum(step)])
        self.u = np.interp(np.linspace(0.0, arc[-1], self.nu), arc, ug)
        self.s = EY.lid_s(self.u)                     # lid parameter of each column
        self.vU, self.vL = self.up(self.u), self.lo(self.u)
        self.chord = EY.CORNER_IN[1] + self.s * (EY.CORNER_OUT[1] - EY.CORNER_IN[1])

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
        over 4 mm; `h` over the surface. Outside the opening the blend never goes behind the skin (the shell lies deeper
        there: a strip raised just above the margin would sink into the face), and the strip stands OUT_LIFT further off
        it: a strip's faces span several mm and would dip into curved skin between their corners."""
        u, v = np.asarray(u, float), np.asarray(v, float)
        x, z = self.xz(u, v)
        Pb = self.bar.point(x, z, DISPLAY_GAP + (h - 0.0004))
        ang = np.arctan2(v - self.alm.c[1], u - self.alm.c[0])
        out = np.maximum(np.hypot(u - self.alm.c[0], v - self.alm.c[1]) - np.interp(ang, self.alm.a, self.alm.r), 0.0)
        b = smooth(out / 0.004)
        Ps = SK.lift(self.shape, x, z)
        Ps[:, 1] += self.dip(x, z) - h - OUT_LIFT * b
        P = Pb + b[:, None] * (Ps - Pb)
        P[:, 1] = np.where(out > 0.0, np.minimum(P[:, 1], Ps[:, 1]), P[:, 1])
        return P

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

    # ---- the closed-eye drawing: the band's centre line and thickness
    def band_path(self, kind, s, edge, ends):
        """Centre line (n, 2) of the drawn closed eye at the band's lid parameters s: kind 'arc' (along `edge`), 'chevron',
        'dash'. `ends` = (s_in, s_peel): the lid parameters of the open band's inner point and of where it meets the lid;
        in an arc its tail (s < s_peel) runs straight to a point CLOSED_TAIL from the inner corner. Past s = 1 (the band's run
        down the eye's outer side) the line goes on straight in a flick (closed_flick). The line is drawn on a fixed grid of
        lid parameters that runs on straight past both ends, rounded (ROUND passes: about 1 mm) where it turns, and sampled
        at s."""
        s_in, s_peel = ends
        u0, u1 = self.u_in, self.u_out
        q = np.linspace(-0.4, 1.6, 1001)
        lid = q <= 1.0

        def along(sb):
            if kind == "arc":
                u = u0 + np.maximum(sb, s_peel) * (u1 - u0)
                P = np.stack([u, np.interp(u, self.u, edge)], -1)
                tail = sb < s_peel
                tip = np.array([u0 + CLOSED_TAIL[0], self.up(u0) + CLOSED_TAIL[1]])
                P[tail] += ((s_peel - sb[tail]) / (s_peel - s_in))[:, None] * (tip - P[tail])
                return P
            if kind == "dash":
                u = u0 + 0.003 + sb * (u1 - u0 - 0.004)
                return np.stack([u, edge[0] + 0.0010 * (2 * sb - 1) ** 2], -1)
            # chevron: apex toward the nose, the strokes opening outward, along the band's own run k (0 .. 1)
            lo_tip, apex = np.array([u1 - 0.0030, -0.0128]), np.array([u0 + 0.0085, -0.0010])
            hi_tip = np.array([u1 - 0.0030, 0.0112])
            k = ((sb - s_in) / (1.0 - s_in))[:, None]
            return np.where(k < 0.5, lo_tip + (apex - lo_tip) * (k / 0.5), apex + (hi_tip - apex) * ((k - 0.5) / 0.5))
        P = np.empty((len(q), 2))
        P[lid] = along(q[lid])
        flick = np.array(self.flick if kind != "chevron" else (0.0030, 0.0020))
        P[~lid] = along(np.array([1.0]))[0] + ((q[~lid] - 1.0) / EY.OUTER_RUN)[:, None] * flick
        for _ in range(ROUND):
            P[1:-1] = 0.25 * P[:-2] + 0.5 * P[1:-1] + 0.25 * P[2:]
        s = np.atleast_1d(np.asarray(s, float))
        return np.stack([np.interp(s, q, P[:, c]) for c in (0, 1)], -1)

    def closed_line(self, kind, s, edge, ends):
        """(centre line, unit normals to its upper side) of the drawn closed eye at the lid parameters s; the normals come
        from the line's direction over +-0.002 of s, so any parameters (repeated, a few) get them."""
        s = np.atleast_1d(np.asarray(s, float))
        d = self.band_path(kind, s + 0.002, edge, ends) - self.band_path(kind, s - 0.002, edge, ends)
        t = d / np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-12)
        return self.band_path(kind, s, edge, ends), np.stack([-t[:, 1], t[:, 0]], -1)

    @staticmethod
    def band_width(s, s_in):
        """The drawn band's thickness at the lid parameters s: pointed at the inner point (s_in) and at the flick's end."""
        s = np.asarray(s, float)
        prof = np.interp(np.minimum(s, 1.0), [s_in, 0.10, 0.30, 0.55, 0.85, 1.0], [0.0, 0.62, 0.92, 1.0, 0.95, 0.80])
        return CLOSED_WIDTH * prof * np.clip(1.0 - (s - 1.0) / EY.OUTER_RUN, 0.0, 1.0)


# ----------------------------------------------------------------------------------------------- the lid morph book
def morph_table(L):
    """name -> dict(up, lo: sheet edges (nu,); band: (kind, edge / parameter); lower: how the lower lash moves; for the
    drawn closed eyes (band kinds arc, chevron, dash) also crease: 'follow' | 'fade')."""
    blink = L.arc(0.0062)
    smile = L.arc(-0.0058)

    # a lowered lid leaves the eye `keep` open, less towards the corners, where the lower lid climbs to meet the upper one
    keep = 0.0016 * np.clip(np.minimum(L.s, 1.0 - L.s) / 0.06, 0.0, 1.0)

    def lowered(inner, outer, mid=0.0):
        """Upper lid edge lowered by `inner` (at the nose) .. `outer` (at the temple), `mid` more in the middle."""
        d = inner + (outer - inner) * L.s + mid * np.sin(np.pi * L.s)
        return np.maximum(L.vU - d, L.vL + keep)

    def raised_lower(k):
        return L.vL + k * np.sin(np.pi * L.s) ** 0.8

    def mood(up, lo=None):
        lo = L.vL if lo is None else np.minimum(lo, up - keep)
        return dict(up=up, lo=lo, band=("hang", up), lower="lift")
    return {
        "blink": dict(up=blink, lo=blink, band=("arc", blink), crease="follow", lower="merge"),
        "smile_eyes": dict(up=smile, lo=smile, band=("arc", smile), crease="follow", lower="merge"),
        "hau": dict(up=blink, lo=blink, band=("chevron", None), crease="fade", lower="merge"),
        "calm": dict(up=L.flat(-0.0050), lo=L.flat(-0.0050), band=("dash", [-0.0050]), crease="follow", lower="merge"),
        "jito": dict(up=L.flat(0.0004), lo=L.vL, band=("arc", L.flat(0.0004)), crease="follow", lower="keep"),
        "surprised": dict(up=L.vU, lo=L.vL, band=("lift", 0.0030), lower="drop"),
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
            st = {p: H.strips[f"{p}.{sd}"] for p in ("lash", "lashfork", "lashwing", "crease", "lashlow")}

            def move(piece, lo_t, hi_t):
                _move(book, name, H, f"{piece}.{sd}", L, st[piece], lo_t, hi_t, piece_rows)
            lash = st["lash"]
            s = lash["s"]
            ends = (float(s[0]), lash["hold"][0])
            bk, bp = spec["band"]
            if bk in ("hang", "lift"):
                # the eye stays open and everything above it moves with the band; the band's run down the outer side stays
                # on the lower lid. A lowered lid (hang) puts the band's lower edge on the lowered lid and drops every other
                # point as far as the lid drops at its lid parameter, held at the band's `hold` (the tail and the fork beyond
                # where the band meets the lid, the elbow, the outer side and the wing beyond its turn move rigidly);
                # surprised (lift) raises them all.
                outer = np.clip(1.0 - (s - 1.0) / EY.OUTER_RUN, 0.0, 1.0)
                up = lambda P, d: P + np.stack([np.zeros_like(d), d], -1)
                if bk == "hang":
                    drop = lambda q: np.interp(L.u_in + q * (L.u_out - L.u_in), L.u, bp - L.vU)
                    held = lambda P: drop(np.clip(EY.lid_s(P[:, 0]), *lash["hold"]))
                    lo_t = up(lash["lo"], np.where(s < ends[1], drop(ends[1]), 0.0))
                    on_lid = (s >= ends[1]) & (s <= 1.0)
                    u = L.u_in + s[on_lid] * (L.u_out - L.u_in)
                    lo_t[on_lid] = np.stack([u, np.interp(u, L.u, bp)], -1)
                    hi_t = up(lash["hi"], held(lash["hi"]) * outer)
                else:
                    held = lambda P: np.full(len(P), bp)
                    lo_t, hi_t = up(lash["lo"], bp * outer), up(lash["hi"], bp * outer)
                move("lash", lo_t, hi_t)
                for p in ("lashfork", "lashwing", "crease"):
                    move(p, up(st[p]["lo"], held(st[p]["lo"])), up(st[p]["hi"], held(st[p]["hi"])))
            else:
                path, n = L.closed_line(bk, s, bp, ends)
                w = L.band_width(s, ends[0])
                move("lash", path - 0.5 * w[:, None] * n, path + 0.5 * w[:, None] * n)
                for p in ("lashfork", "lashwing"):            # the blades fold into the drawn line (its inner end, its flick)
                    F = L.band_path(bk, st[p]["s"], bp, ends)
                    move(p, F, F)
                cr = st["crease"]
                if spec["crease"] == "follow":                # the crease keeps its height over the drawn band
                    P, nc = L.closed_line(bk, cr["s"], bp, ends)
                    lo_c = P + nc * (0.5 * L.band_width(cr["s"], ends[0]) + cr["gap"])[:, None]
                    move("crease", lo_c, lo_c + nc * np.linalg.norm(cr["hi"] - cr["lo"], axis=1)[:, None])
                else:                                         # or fades: collapses to its centre line
                    mid = 0.5 * (cr["lo"] + cr["hi"])
                    move("crease", mid, mid)
            # lower lash
            ll = st["lashlow"]
            if spec["lower"] == "merge":                      # into the drawn band's lower edge
                P, n = L.closed_line(bk, ll["s"], bp, ends)
                base = P - 0.5 * L.band_width(ll["s"], ends[0])[:, None] * n
                move("lashlow", base, base)
            elif spec["lower"] == "lift":                     # the lower lash rides up with the lower lid
                dv = np.interp(ll["lo"][:, 0], L.u, spec["lo"] - L.vL)
                off = np.stack([np.zeros_like(dv), dv], -1)
                move("lashlow", ll["lo"] + off, ll["hi"] + off)
            elif spec["lower"] == "drop":
                move("lashlow", ll["lo"] + [0.0, -0.0006], ll["hi"] + [0.0, -0.0006])


def _move(book, name, H, piece, L, st, lo_t, hi_t, piece_rows):
    """Delta of a strip piece (rows of the lines mesh) from its rest positions to the target edges (u, v), raised where its
    faces would cut into the skin (a long face over the curved skin round the eye's outer corner would)."""
    pts = _strip_points(np.asarray(lo_t, float), np.asarray(hi_t, float), st["tip"], st["ends_pointed"])
    tgt = DC.clear_points(H.surf, L.disp(pts[:, 0], pts[:, 1], st["h"]), st["faces"])
    book.add(name, "lines", tgt - H.decals[piece].positions(), piece_rows(H.lines, piece))
