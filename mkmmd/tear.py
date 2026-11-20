"""The tear drawn (docs/design.md: Transitions and inserts), the post layer (numpy, OpenCV): the outgoing picture torn in
two over the incoming one. mkmmd.core.tear checks it and works out the line, the fringe of bare paper along it and where
the halves are on each window frame; this draws a window frame of it.

Each half is made where it lay (in the picture's own place): its print (the picture on its side of the torn line,
antialiased) and its paper (the strip of bare paper the half carries past the print where the fringe is on its side,
and the fibres: short pale hairs standing out of the paper's torn edge all along it, now and then longer). Then it is
moved as a whole: bent away from the line behind the rip's front, turned about its pivot and tilted toward the lens (an
exact inverse map, cv2.remap), a little darker as it turns from the lamp. The halves' papers lie under both prints, so
before they part the picture is whole; the soft shadow of both halves, offset and blurred by their height, falls on the
incoming picture."""
from functools import lru_cache

import cv2
import numpy as np

from . import matte as M
from .peel import shadow as _shadow
from .core import sheet3d as S3
from .core import tear as TE

FIBRE = (0.004, 0.011, 0.03)      # a fibre's length: short sides, the most of the common ones, the long ones
FIBRE_EVERY = 0.0018              # fibres this far apart along the edge (short sides)
FIBRE_ALPHA = 0.8
FLAP = (0.05, 0.004, 0.009)       # small flaps on the torn edge: how often (share of fibres), half their base, length
PAPER_SHADE = 0.94                # the bare paper a touch duller than the lightest colour at its torn edge
AMBIENT = 0.7                     # the light a half bowed or tilted from the lamp still gets
BOW_HEIGHT = 2.5                  # how high the torn edge lifts for each px the bow draws it back
CONTACT = (1.2, 1.5, 0.35)        # the thin shadow under the torn edges: offset and blur (px at 1080), darkness


def frame(tr, frame, main, incoming, fps):
    """The outgoing frame `main` (float RGB) torn at `frame` of the transition `tr`'s window over `incoming`."""
    h, w = main.shape[:2]
    size = (w, h)
    k = frame - tr["first"]
    st = TE.state(tr, k, fps, size)
    lay = _layout(tr["angle"], tr["from"], tr["jag"], tr["seed"], tuple(tr["center"]), size)
    main = np.ascontiguousarray(main, np.float32)
    halves, height, cover = {}, np.zeros((h, w), np.float32), np.zeros((h, w), np.float32)
    for half in ("a", "b"):
        box = _box(tr, size, half, st)
        if box is None:                                          # it has left the frame
            continue
        y0, y1, x0, x1 = box
        pr = lay["print_" + half]
        stack = np.concatenate([main * pr[..., None], pr[..., None], lay["paper_" + half][..., None]], axis=2)
        mx, my, z, lit = _inverse(tr, size, half, st, lay, box)
        got = cv2.remap(stack, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        lit = lit * np.float32(_light(tr, size, half, st))
        got[..., :3] *= lit[..., None]
        halves[half] = (box, got, lit)
        here = got[..., 3] + got[..., 4]
        np.copyto(height[y0:y1, x0:x1], z + np.float32(st["lift"]), where=here > 0.5)
        np.maximum(cover[y0:y1, x0:x1], np.clip(here, 0.0, 1.0), out=cover[y0:y1, x0:x1])
    out = np.ascontiguousarray(incoming, np.float32).copy()
    ink = np.asarray(tr["ink"], np.float32)
    if tr["shadow"] > 0.0 and halves:
        px = min(size) / 1080.0
        off = CONTACT[0] * px                                    # a thin contact shadow right under the edges
        moved = cv2.warpAffine(cover, np.float32([[1, 0, off], [0, 1, off]]), size, borderValue=0)
        contact = cv2.GaussianBlur(moved, (0, 0), CONTACT[1] * px) * np.float32(CONTACT[2])
        sh = 1.0 - (1.0 - _shadow(cover, height)) * (1.0 - contact)
        out += (ink - out) * (np.float32(tr["shadow"]) * sh)[..., None]
    paper = np.asarray(tr["back"], np.float32) * PAPER_SHADE
    for (y0, y1, x0, x1), got, lit in halves.values():          # the bare paper under both prints
        a = np.clip(got[..., 4], 0.0, 1.0)[..., None]
        o = out[y0:y1, x0:x1]
        o += (paper * lit[..., None] - o) * a
    for (y0, y1, x0, x1), got, _ in halves.values():
        a = np.clip(got[..., 3], 0.0, 1.0)[..., None]
        out[y0:y1, x0:x1] = out[y0:y1, x0:x1] * (1.0 - a) + got[..., :3]
    _crumbs(out, tr, k, fps, size, paper, ink)
    return np.clip(out, 0.0, 1.0)


def _crumbs(out, tr, k, fps, size, paper, ink):
    """The paper crumbs thrown from the rip (mkmmd.core.tear.crumbs) drawn on `out` in place: small turning bits of
    paper, lit as they turn, each over a faint shadow."""
    h, w = out.shape[:2]
    sub = 1 << 3
    for c in TE.crumbs(tr, fps, size):
        at = TE.crumb_at(tr, c, k, fps, size)
        if at is None:
            continue
        x, y, z, ang = at
        sx, sy, m = S3.project(x, y, z, size)
        r = c["r"] * m
        if not (-r < sx < w + r and -r < sy < h + r):
            continue
        squash = abs(np.cos(ang * 0.7))                           # it turns over as it flies: thinner, then wider
        pts = np.float64([[np.cos(ang + t) * r * (1.0 if i % 2 else 0.35 + 0.65 * squash),
                           np.sin(ang + t) * r * (1.0 if i % 2 else 0.35 + 0.65 * squash)]
                          for i, t in enumerate((0.0, 1.7, 3.1, 4.6))])
        run = (S3.SHADOW_RUN[0] * z * 0.3, S3.SHADOW_RUN[1] * z * 0.3)
        for (dx, dy), colour, alpha in ((run, ink, 0.25), ((0.0, 0.0), paper, 1.0)):
            q = np.round((pts + (sx + dx, sy + dy)) * sub).astype(np.int32)
            x0, y0 = max(int(q[:, 0].min() // sub) - 1, 0), max(int(q[:, 1].min() // sub) - 1, 0)
            x1, y1 = min(int(q[:, 0].max() // sub) + 2, w), min(int(q[:, 1].max() // sub) + 2, h)
            if x1 <= x0 or y1 <= y0:
                continue
            mask = np.zeros((y1 - y0, x1 - x0), np.uint8)
            cv2.fillConvexPoly(mask, q - np.int32([x0 * sub, y0 * sub]), 255, cv2.LINE_AA, 3)
            a = (mask.astype(np.float32) * np.float32(alpha / 255.0))[..., None]
            shade = np.float32(0.8 + 0.2 * squash) if alpha == 1.0 else np.float32(1.0)
            region = out[y0:y1, x0:x1]
            region += (colour * shade - region) * a


def _box(tr, size, half, st):
    """(y0, y1, x0, x1): the part of the frame the half covers in state `st` (its outline seen, the fibres' reach and
    a pixel to spare), or None once it has left the frame."""
    w, h = size
    pts = np.float64([TE.place(tr, size, half, x, y, st) for x, y in TE.outline(tr, size, half)])
    m = (TE.FIBRE_REACH + TE.BOW[0]) * min(size) + 2.0
    x0, y0 = max(int(np.floor(pts[:, 0].min() - m)), 0), max(int(np.floor(pts[:, 1].min() - m)), 0)
    x1, y1 = min(int(np.ceil(pts[:, 0].max() + m)) + 1, w), min(int(np.ceil(pts[:, 1].max() + m)) + 1, h)
    return (y0, y1, x0, x1) if x1 > x0 and y1 > y0 else None


def _light(tr, size, half, st):
    """How lit a half is, tilted as it is (1 flat; never brighter)."""
    _, u0, _ = TE.hinge(tr, size, half)
    a, b = st["turn"][half], st["tilt"]
    u = (u0[0] * np.cos(a) - u0[1] * np.sin(a), u0[0] * np.sin(a) + u0[1] * np.cos(a))
    n = (-np.sin(b) * u[0], -np.sin(b) * u[1], np.cos(b))
    L = S3.LIGHT
    return AMBIENT + (1.0 - AMBIENT) * min(max(n[0] * L[0] + n[1] * L[1] + n[2] * L[2], 0.0) / L[2], 1.0)


def _inverse(tr, size, half, st, lay, box):
    """(map_x, map_y, z, light): for every pixel of `box` = (y0, y1, x0, x1) the point of the picture the half shows
    there, the half's height there (px) and how lit it is across its bow. Each step of mkmmd.core.tear.place is undone
    exactly: the perspective of the tilted plane (a 2 x 2 solve per pixel), the turn about the pivot, then the bow and
    the bend away from the line, both across the line (a quadratic per pixel for the bow)."""
    y0, y1, x0, x1 = box
    xs, ys = lay["grid"][0][y0:y1, x0:x1], lay["grid"][1][y0:y1, x0:x1]
    (px, py), u0, out = TE.hinge(tr, size, half)
    tx, ty = px + out[0] * st["throw"][half], py + out[1] * st["throw"][half]   # the pivot where it has been thrown
    a, b = st["turn"][half], st["tilt"]
    ca, sa = np.cos(a), np.sin(a)
    u = (u0[0] * ca - u0[1] * sa, u0[0] * sa + u0[1] * ca)
    v = (-u[1], u[0])
    D = np.float32(S3.distance(size))
    cx, cy = S3.centre(size)
    qx, qy = xs - np.float32(cx), ys - np.float32(cy)
    rx, ry = qx - np.float32(tx - cx), qy - np.float32(ty - cy)
    cb, sb = np.float32(np.cos(b)), np.float32(np.sin(b))
    c1x, c1y = cb * np.float32(u[0]) + qx * sb / D, cb * np.float32(u[1]) + qy * sb / D
    vx, vy = np.float32(v[0]), np.float32(v[1])
    det = c1x * vy - c1y * vx
    along = (rx * vy - ry * vx) / det
    across = (c1x * ry - c1y * rx) / det
    dx = along * np.float32(u[0]) + across * vx                   # the turned half, flat, about the pivot
    dy = along * np.float32(u[1]) + across * vy
    x = np.float32(px) + dx * np.float32(ca) + dy * np.float32(sa)    # the turn undone
    y = np.float32(py) - dx * np.float32(sa) + dy * np.float32(ca)
    o, d, e = lay["axes"]
    sign = np.float32(1.0 if half == "a" else -1.0)
    l = (x - np.float32(o[0])) * np.float32(d[0]) + (y - np.float32(o[1])) * np.float32(d[1])
    q = (x - np.float32(o[0])) * np.float32(e[0]) + (y - np.float32(o[1])) * np.float32(e[1])
    l0, l1 = lay["span"]
    S = float(min(size))
    L = np.float32(l1 - l0)
    front = np.float32(st["front"])
    opened = np.float32(0.5 * TE.OPEN * S) * (np.clip(front - l, 0.0, None) / L) ** np.float32(0.8)
    J = np.interp(l, *lay["line"]).astype(np.float32)
    r2 = sign * (q - J) - opened                                  # how far in from the torn line, bowed
    W = np.float32(TE.BOW[1] * S)
    B = np.float32(TE.BOW[0] * S * (0.35 + 0.65 * ((st.get("k", 0) + 1) / tr["n"]) ** 0.8)) * \
        np.clip((front - l) / (np.float32(0.2) * L), 0.0, 1.0)
    A = np.maximum(B / (W * W), np.float32(1e-9))                 # r + B (1 - r / W)^2 = r2, for 0 < r < W
    bq = 1.0 - 2.0 * B / W
    quad = (-bq + np.sqrt(np.maximum(bq * bq - 4.0 * A * (B - r2), 0.0))) / (2.0 * A)
    quad = np.where(B > 1e-3, quad, r2)
    r = np.where(r2 <= B, r2 - B, np.where(r2 >= W, r2, quad))
    qs = J + sign * r
    xs_, ys_ = x + (qs - q) * np.float32(e[0]), y + (qs - q) * np.float32(e[1])
    t = np.clip(1.0 - np.maximum(r, 0.0) / W, 0.0, 1.0)
    Hb = np.float32(BOW_HEIGHT) * B
    zb = Hb * t * t
    slope = 2.0 * Hb / W * t                                      # how steeply it rises toward the edge
    gx, gy = -slope * sign * np.float32(e[0]), -slope * sign * np.float32(e[1])   # the height's gradient on screen
    Lt = S3.LIGHT
    lam = (-gx * np.float32(Lt[0]) - gy * np.float32(Lt[1]) + np.float32(Lt[2])) / np.sqrt(gx * gx + gy * gy + 1.0)
    light = np.float32(AMBIENT) + np.float32(1.0 - AMBIENT) * np.clip(lam / np.float32(Lt[2]), 0.0, 1.0)
    return xs_, ys_, np.maximum(along * sb, 0.0) + zb, light


@lru_cache(maxsize=4)
def _layout(angle, start, jag, seed, center, size):
    """What does not change over the window: each half's print and paper coverage in the picture's own place, the
    fibres, the line's axes and span and the pixel grid."""
    tr = {"angle": angle, "from": start, "jag": jag, "seed": seed, "center": list(center)}
    w, h = size
    o, d, e = TE.axes(tr, size)
    line = TE.rip(tr, size)
    xs, ys = M.grid(h, w)
    l = (xs - np.float32(o[0])) * np.float32(d[0]) + (ys - np.float32(o[1])) * np.float32(d[1])
    q = (xs - np.float32(o[0])) * np.float32(e[0]) + (ys - np.float32(o[1])) * np.float32(e[1])
    ls = np.float32(line["l"])
    J = np.interp(l, ls, np.float32(line["q"])).astype(np.float32)
    F = np.interp(l, ls, np.float32(line["fringe"])).astype(np.float32)
    pa = np.clip(q - J + 0.5, 0.0, 1.0)                          # half a's print: q > J
    pb = 1.0 - pa
    fa = np.clip(q - (J - np.maximum(F, 0.0)) + 0.5, 0.0, 1.0) * pb   # a's bare paper, past its print into b's side
    fb = np.clip((J + np.maximum(-F, 0.0)) - q + 0.5, 0.0, 1.0) * pa
    fib_a, fib_b = _fibres(line, o, d, e, size, seed)
    paper_a = np.maximum(fa, fib_a * pb)
    paper_b = np.maximum(fb, fib_b * pa)
    shade = np.float32(PAPER_SHADE)
    out = {"print_a": pa.astype(np.float32), "print_b": pb.astype(np.float32), "paper_a": paper_a.astype(np.float32),
           "paper_b": paper_b.astype(np.float32), "paper_shade": shade, "grid": (xs, ys), "axes": (o, d, e),
           "span": TE.span(tr, size), "line": (ls, np.float32(line["q"]))}
    for v in out.values():
        if isinstance(v, np.ndarray):
            v.setflags(write=False)
    return out


def _fibres(line, o, d, e, size, seed):
    """(a, b): the fibres standing out of each half's torn paper edge, coverage (h, w) 0..1: short pale hairs at random
    slants into the other half's side, a few long ones."""
    w, h = size
    S = float(min(size))
    sub = 1 << 3
    rng = np.random.default_rng(seed + 7919)
    out = []
    for sign in (1.0, -1.0):                                    # a's hairs point to -e (into b), b's to +e
        canvas = np.zeros((h, w), np.uint8)
        ls, qs, fs = (np.float64(line[k]) for k in ("l", "q", "fringe"))
        total = ls[-1] - ls[0]
        n = int(total / (FIBRE_EVERY * S))
        at = np.sort(rng.uniform(ls[0], ls[-1], n))
        q = np.interp(at, ls, qs)
        f = np.interp(at, ls, fs)
        edge = q - sign * np.maximum(sign * f, 0.0)              # the paper's torn edge on this half
        length = np.where(rng.random(n) < 0.1, rng.uniform(FIBRE[1], FIBRE[2], n), rng.uniform(FIBRE[0], FIBRE[1], n)) * S
        slant = rng.normal(0.0, 0.6, n)
        x0 = o[0] + at * d[0] + edge * e[0]
        y0 = o[1] + at * d[1] + edge * e[1]
        ux, uy = -sign * e[0] + slant * d[0], -sign * e[1] + slant * d[1]
        norm = np.hypot(ux, uy)
        x1, y1 = x0 + ux / norm * length, y0 + uy / norm * length
        width = max(1, int(round(S / 1080.0)))
        for i in range(n):
            if -20 < x0[i] < w + 20 and -20 < y0[i] < h + 20:
                cv2.line(canvas, (int(x0[i] * sub), int(y0[i] * sub)), (int(x1[i] * sub), int(y1[i] * sub)),
                         int(255 * FIBRE_ALPHA * rng.uniform(0.5, 1.0)), width, cv2.LINE_AA, 3)
        for i in np.flatnonzero(rng.random(n) < FLAP[0]):            # small flaps of paper torn half free
            if -20 < x0[i] < w + 20 and -20 < y0[i] < h + 20:
                base, tip = FLAP[1] * S * rng.uniform(0.6, 1.4), FLAP[2] * S * rng.uniform(0.5, 1.5)
                nx, ny = ux[i] / norm[i], uy[i] / norm[i]
                pts = np.float64([[x0[i] - ny * base, y0[i] + nx * base], [x0[i] + ny * base, y0[i] - nx * base],
                                  [x0[i] + nx * tip + ny * base * rng.uniform(-1, 1),
                                   y0[i] + ny * tip - nx * base * rng.uniform(-1, 1)]])
                cv2.fillConvexPoly(canvas, np.round(pts * sub).astype(np.int32), 235, cv2.LINE_AA, 3)
        out.append(canvas.astype(np.float32) / 255.0)
    return out[0], out[1]
