"""The iris insert drawn (docs/design.md: Transitions and inserts), the post layer (numpy, OpenCV): a round window over the
host frame onto the inner picture, seen as through a lens at the end of a tube. mkmmd.core.iris times it and says where
the circle is (its centre and radius, the inner picture's scale and place); mkmmd.cutfx hands each window frame here with
the inner picture (the inner shot's plate, or the video, at the frame's size) and the projected anchor. A `ring` (a
record, a plate, chrome rings: mkmmd.portal) is laid over the host round the circle first, the window over it.

Inside the circle the inner picture, scaled about its own centre (mkmmd.matte.warp_scaled); near the rim its colour
channels pulled apart along the radius (red outward, blue inward, `fringe` px at 1080 on the short side at the edge: a
lens's chromatic edge), then darkened toward the rim's colour over the outer `width` of the radius, by `dark` at the edge
(a smoothstep: it starts imperceptibly and settles into a dark lip at the rim, as inside a tube), then a glint: a thin
crescent of the palette's lightest colour in that band at the upper left, tapering at both ends, in a soft bloom, and a
fainter, shorter one opposite. Round the outside an optional `edge` ring. The edge is antialiased over one pixel. Only
the circle's box is worked on (the rim only where it runs: see _parts); inside the vignette's inner edge the picture is
left as it is, and outside the circle and its ring the host frame exactly as it was."""
import math

import cv2
import numpy as np

from . import matte as M
from . import portal as PORTAL
from .core import iris as IR

GLINTS = ((135.0, 72.0, 1.0), (-45.0, 38.0, 0.35))   # (middle, span) in degrees counter-clockwise from the right with y
#                                                      up, and strength: the upper left one and the fainter one opposite
GLINT_AT = 0.8                  # the glint's line across the rim's band: 0 at the vignette's inner edge, 1 at the rim
GLINT_THICK = 0.1               # its thickness at its middle, a share of the band (at least 1.2 px)
BLOOM = (3.0, 0.25)             # its soft glow: reaching this many times its thickness to either side (never past the
#                                 band's inner edge: the middle of the circle stays the picture), and its strength
SLABS = 16                      # the rim is worked on in slabs across the circle's box, each cut to the columns it runs in


def frame(ins, frame, host, inner, anchor, per_metre, output, t=0.0, sprites=(), tiles=None, eyes=None):
    """The host frame `host` (float RGB) with the iris of insert `ins` at `frame` over it: `inner` the picture inside at
    the frame's size, `anchor` the projected anchor in px (index coordinates), `per_metre` frame heights per metre at its
    depth (the point layer's `m`), `output` the output's name (its own rest size). Its ring, if any, is drawn `t` seconds
    into the window with a plate's `sprites`, their `tiles` and their `eyes` (mkmmd.portal.draw)."""
    h, w = host.shape[:2]
    c = IR.circle(ins, IR.state(ins, frame), output, (w, h), anchor, per_metre)
    px = min(w, h) / 1080.0
    if ins.get("ring") is not None:
        host = PORTAL.draw(host, inner, c, ins["ring"], t, px, sprites, tiles, eyes)
    return draw(host, inner, c, ins["rim"], ins["edge"], px)


def draw(host, inner, c, rim, edge, px):
    """`host` with the circle `c` (mkmmd.core.iris.circle) over it showing `inner`, its `rim` (None: none) and `edge`
    (None: none); `px` pixels per pixel at 1080 on the short side. `host` itself when the circle is under half a pixel;
    exactly `inner` when the circle covers the frame with its rim outside it."""
    r = float(c["radius"])
    if r < IR.MIN_RADIUS:
        return host
    h, w = host.shape[:2]
    cx, cy = c["centre"]
    ring = edge["width"] * px if edge else 0.0
    reach = r + ring + 2.0
    x0, y0 = max(0, int(math.floor(cx - reach))), max(0, int(math.floor(cy - reach)))
    x1, y1 = min(w, int(math.ceil(cx + reach)) + 1), min(h, int(math.ceil(cy + reach)) + 1)
    if x1 <= x0 or y1 <= y0:
        return host
    xs, ys = M.grid(h, w)
    dx = xs[y0:y1, x0:x1] - np.float32(cx)
    dy = ys[y0:y1, x0:x1] - np.float32(cy)
    rho = cv2.magnitude(dx, dy)
    if c["k"] == 1.0 and tuple(c["at"]) == ((w - 1) / 2.0, (h - 1) / 2.0):     # the picture as it is: the frame filled
        pic = inner[y0:y1, x0:x1]
    else:
        pic = M.warp_scaled(inner, c["at"], c["k"], (x0, y0, x1, y1))
    if rim:
        pic = _lens(pic, rim, c, (cx - x0, cy - y0), dx, dy, rho, px)
    cov = np.clip(np.float32(r + 0.5) - rho, 0.0, 1.0)                 # the circle, antialiased over a pixel
    if float(cov.min()) >= 1.0:                                         # the circle covers its box: the picture alone
        box = np.array(pic)
    else:
        box = host[y0:y1, x0:x1]
        if edge:                                                        # the ring, grown out from under the circle
            box = _paint(box, edge["color"], np.clip(np.float32(r + ring + 0.5) - rho, 0.0, 1.0))
        box = cv2.add(cv2.multiply(box, _three(1.0 - cov)), cv2.multiply(pic, _three(cov)))   # exact at 0 and 1
    if (x0, y0, x1, y1) == (0, 0, w, h):
        return box
    out = host.copy()
    out[y0:y1, x0:x1] = box
    return out


def _three(a):
    return cv2.merge([a, a, a])


def _paint(img, colour, w):
    """`colour` laid on the picture `img` (h, w, 3) with the coverage `w` (h, w): exactly `img` where w is 0 and the
    colour where it is 1."""
    return cv2.add(cv2.multiply(img, _three(1.0 - w)), cv2.multiply(_three(w), tuple(float(v) for v in colour) + (0.0,)))


def vignette(t):
    """The vignette's darkening at `t` (0 at its inner edge .. 1 at the rim), 0 .. 1: a smoothstep, flat where it starts
    and settling into the dark lip at the rim."""
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _lens(pic, rim, c, centre, dx, dy, rho, px):
    """The inner picture `pic` over the circle's box as the lens shows it: the fringe, the vignette and the glint. `centre`
    is the circle's centre in the box, `dx`, `dy`, `rho` each pixel's offset and distance from it. Inside the vignette's
    inner edge the picture is left as it is (and not worked on)."""
    r = float(c["radius"])
    inner = r * (1.0 - rim["width"])
    if float(rho.max()) <= inner:                                       # the rim is outside the frame
        return pic
    out = np.array(pic)
    scale = np.float32(1.0 / (r * rim["width"]))
    for y0, y1, x0, x1 in _parts((0, 0, pic.shape[1], pic.shape[0]), centre, inner):
        sl = (slice(y0, y1), slice(x0, x1))
        t = np.clip((rho[sl] - np.float32(inner)) * scale, 0.0, 1.0)
        if rim["fringe"] > 0.0:
            _fringe(pic, out[sl], centre, dx[sl], dy[sl], rho[sl], t, rim["fringe"] * px)
        if rim["dark"] > 0.0:
            out[sl] = _paint(out[sl], rim["color"], (vignette(t) * np.float32(rim["dark"])).astype(np.float32))
    if c["glint"] > 0.0:
        _glint(out, c, centre, dx, dy, rho, inner, rim["light"], c["glint"])
    return out


def _parts(box, centre, inner, outer=math.inf, slabs=SLABS):
    """[(y0, y1, x0, x1)]: parts of `box` = (x0, y0, x1, y1) that hold every pixel of it whose distance from `centre` lies
    between `inner` and `outer`: the box cut into `slabs` slabs across, each cut to the columns `outer` reaches on one of
    its rows, less the columns within `inner` on all its rows."""
    bx0, by0, bx1, by1 = box
    cx, cy = centre
    rows = np.linspace(by0, by1, min(slabs, by1 - by0) + 1).round().astype(int).tolist()
    out = []
    for y0, y1 in zip(rows[:-1], rows[1:]):
        near = 0.0 if y0 <= cy <= y1 - 1 else min(abs(y0 - cy), abs(y1 - 1 - cy))   # its rows nearest and farthest
        far = max(abs(y0 - cy), abs(y1 - 1 - cy))
        a0, a1 = bx0, bx1
        if math.isfinite(outer):
            if near >= outer:
                continue
            half = math.sqrt(outer * outer - near * near) + 1.0
            a0, a1 = max(bx0, int(math.floor(cx - half))), min(bx1, int(math.ceil(cx + half)) + 1)
        half = math.sqrt(inner * inner - far * far) - 1.0 if far < inner else -1.0
        cut = [(a0, a1)]
        if half > 0.0:                                               # columns i0 .. i1 - 1 lie within `inner` throughout
            i0, i1 = int(math.ceil(cx - half)), int(math.floor(cx + half)) + 1
            cut = [(a0, min(a1, i0)), (max(a0, i1), a1)]
        out += [(y0, y1, p0, p1) for p0, p1 in cut if p1 > p0]
    return out


def _fringe(pic, part, centre, dx, dy, rho, t, amount):
    """`part` (a part of `out`, the box picture `pic` being worked on; its offsets `dx`, `dy` and distances `rho` from the
    circle's centre `centre`) with red pulled outward and blue inward along the radius by `amount` * t² px (t: 0 at the
    vignette's inner edge, 1 at the rim), each read from `pic` that far in / out, in place; green as it is, and where
    t = 0 nothing moves."""
    s = cv2.divide(cv2.multiply(t, t, scale=float(amount)), cv2.max(rho, 1e-3))     # px moved per px from the centre
    sx, sy = cv2.multiply(dx, s), cv2.multiply(dy, s)
    bx, by = dx + np.float32(centre[0]), dy + np.float32(centre[1])                 # where each pixel is in the box
    red = cv2.remap(pic, cv2.subtract(bx, sx), cv2.subtract(by, sy), cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    blue = cv2.remap(pic, cv2.add(bx, sx), cv2.add(by, sy), cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    part[..., 0], part[..., 2] = red[..., 0], blue[..., 2]


def _glint(pic, c, centre, dx, dy, rho, inner, light, strength):
    """The glints painted into `pic`: thin crescents of `light` in the rim's band (from `inner` out to the rim), each
    tapering to nothing at its ends and thinning there, in a soft bloom; the upper left one and a fainter, shorter one
    opposite (GLINTS). Worked on where they run: the box of each arc, cut to the ring of its bloom."""
    r = float(c["radius"])
    line = inner + GLINT_AT * (r - inner)
    thick = max(GLINT_THICK * (r - inner), 1.2)
    glow, bloom = BLOOM
    reach = min(thick * glow, line - inner)
    if reach <= 0.0:
        return
    h, w = pic.shape[:2]
    for mid, span, share in GLINTS:
        a = np.radians(np.linspace(mid - span / 2.0, mid + span / 2.0, 25))
        px_ = np.concatenate([centre[0] + rr * np.cos(a) for rr in (line - reach, line + reach)])
        py_ = np.concatenate([centre[1] - rr * np.sin(a) for rr in (line - reach, line + reach)])
        box = (max(0, int(math.floor(px_.min() - 1.0))), max(0, int(math.floor(py_.min() - 1.0))),
               min(w, int(math.ceil(px_.max() + 1.0)) + 1), min(h, int(math.ceil(py_.max() + 1.0)) + 1))
        if box[2] <= box[0] or box[3] <= box[1]:
            continue
        # the taper along the arc from the cosine of the angle off its middle: (1 - u²)² for u = angle / half span
        mx, my = math.cos(math.radians(mid)), -math.sin(math.radians(mid))
        lo = math.cos(math.radians(span / 2.0))
        for y0, y1, x0, x1 in _parts(box, centre, line - reach, line + reach, slabs=8):
            ddx, ddy, rr = dx[y0:y1, x0:x1], dy[y0:y1, x0:x1], rho[y0:y1, x0:x1]
            cos = (ddx * np.float32(mx) + ddy * np.float32(my)) / np.maximum(rr, np.float32(1e-3))
            taper = np.clip((cos - np.float32(lo)) * np.float32(1.0 / (1.0 - lo)), 0.0, 1.0) ** 2
            off = rr - np.float32(line)
            half = np.minimum(np.maximum(np.float32(thick / 2.0) * np.sqrt(taper), np.float32(0.6)), np.float32(reach))
            core = np.clip(1.0 - (off / half) ** 2, 0.0, 1.0) ** 2
            halo = np.clip(1.0 - (off * np.float32(1.0 / reach)) ** 2, 0.0, 1.0) ** 2
            g = np.minimum(np.float32(strength * share) * taper * (core + np.float32(bloom) * halo), 1.0)
            pic[y0:y1, x0:x1] = _paint(pic[y0:y1, x0:x1], light, g.astype(np.float32))
