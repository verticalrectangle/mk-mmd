"""Ring portals drawn (docs/design.md: Transitions and inserts, ring portals), the post layer (numpy, OpenCV): the `ring`
round an iris's window, laid over the host frame before the window itself is (mkmmd.iris.frame). mkmmd.core.portal
checks a ring and lays it out in window radii; mkmmd.core.iris says where the window is and how big. Every part of a
ring is a function of the distance from the window's centre and the angle round it, and is laid in passes, each worked
out over the annulus it covers alone (mkmmd.iris._parts) and antialiased over a pixel at its edges: a ring grows and
shrinks with the window as smoothly as the window does, and off the frame it costs nothing.

    vinyl   a soft shadow to the lower right; the record: the dead wax, the grooves (their unevenness, smooth gaps between
            tracks) and the lead-in edge catching the light, under the sheen: two bright wedges toward and away from
            the light at the upper left that stand in the light, swing a little and shimmer as the grooves turn through
            them; then the label (its ground, an arch of the band colour over the top with a dot under it, two lines of
            small print, a ring at its rim), turning clockwise at the record's rpm with the grooves' unevenness.
    plate   a soft shadow; the plate: a glazed height profile (the slope up from the well, the rim gently domed, the lip
            rolling over) lit from the upper left: diffuse light, a crisp highlight and a soft sheen, a broad glaze
            highlight across the rim at the upper left (a fainter one opposite); the two lines under the glaze; then the
            sprites standing round the rim on soft shadows of their own, with googly eyes when the ring gives them some
            (mkmmd.eyes), swinging as they ride the rim and grow with the window.
    lens    a glow of the tint round and between the rings; then each ring, a glass tube: the window's picture (the inner
            shot, as the window places it) sampled across the tube flipped and magnified, swirled with the ring's turn,
            red and blue pulled apart toward the tube's edges; over it a chrome reflection of a two-tone studio (the
            palette's lightest above a bright horizon line, its darkest below) and a highlight; darker toward the tube's
            edges; broken into arcs where the ring has breaks."""
import math
from collections import OrderedDict
from functools import lru_cache

import cv2
import numpy as np

from . import matte as M
from .core import portal as PT

LIGHT = np.float32([-0.45, -0.55, 0.70]) / np.float32(math.sqrt(0.45 ** 2 + 0.55 ** 2 + 0.70 ** 2))  # x right, y down
HALF = (LIGHT + np.float32([0.0, 0.0, 1.0])) / np.float32(np.linalg.norm(LIGHT + np.float32([0.0, 0.0, 1.0])))
LIGHT_AT = math.degrees(math.atan2(0.55, -0.45))      # the light's way round the window (y up): 129 degrees, upper left
SHADOWS = {"vinyl": (0.03, 0.045, 0.04, 0.4), "plate": (0.025, 0.04, 0.035, 0.32)}   # (dx, dy, soft) ring radii, alpha
GROOVES = 160                   # the grooves' unevenness: samples across the grooves' band
SHEEN_WIDTH = 13.0              # degrees: the sheen's wedges (a Gaussian's sigma)
SHEEN_SWING = (6.0, 1.7)        # degrees and seconds: how far and how slowly the wedges swing
GLAZE = ((126.0, 24.0, 0.42), (-54.0, 18.0, 0.14))   # the plate's glaze highlights: (where, sigma) degrees, strength
GLOW = (0.1, 0.16, 0.5)         # the lens's glow: over its rings, its peak at their outer edge, and its reach (radii)
CHROME = 0.3                    # how much of a lens ring is chrome (the rest the picture through it)
SPRITE_STEP = 2.0 ** 0.25       # sprite tiles are drawn at sizes this far apart, then scaled
SPRITE_SHADOW = (0.05, 0.07, 0.06, 0.35)   # a sprite's shadow on the rim: (dx, dy, blur) its size, alpha
TILES = 48                      # sprite tiles kept from frame to frame


def draw(host, inner, c, ring, t, px, sprites=(), tiles=None, eyes=None):
    """`host` (float RGB) with `ring` (mkmmd.core.portal) round the iris circle `c` (mkmmd.core.iris.circle) at `t`
    seconds into the window: `inner` is the picture inside at the frame's size (a lens refracts it, placed as the window
    places it), `px` pixels per pixel at 1080 on the short side, `sprites` a plate's sprite files, `tiles` an
    OrderedDict that keeps their tiles from frame to frame and `eyes` what their googly eyes swing by ({t: clip seconds,
    hits: clip seconds, circle_at: callable(clip seconds) -> [circles]}). `host` itself when none of it is in the frame."""
    from . import iris as IRIS             # mkmmd.iris imports this module: it is all there by the time this runs
    r = float(c["radius"])
    if r < 0.5:
        return host
    h, w = host.shape[:2]
    centre = (float(c["centre"][0]), float(c["centre"][1]))
    xs, ys = M.grid(h, w)
    out = None
    for lo, hi, paint in _passes(ring, r, t, px, inner, c, centre):
        box = (max(0, int(math.floor(centre[0] - hi - 1.0))), max(0, int(math.floor(centre[1] - hi - 1.0))),
               min(w, int(math.ceil(centre[0] + hi + 1.0)) + 1), min(h, int(math.ceil(centre[1] + hi + 1.0)) + 1))
        if box[2] <= box[0] or box[3] <= box[1]:
            continue
        for y0, y1, x0, x1 in IRIS._parts(box, centre, max(lo, 0.0), hi):
            if out is None:
                out = np.array(host, np.float32, copy=True)
            dx = xs[y0:y1, x0:x1] - np.float32(centre[0])
            dy = ys[y0:y1, x0:x1] - np.float32(centre[1])
            region = out[y0:y1, x0:x1]
            region[:] = paint(region, dx, dy, cv2.magnitude(dx, dy))
    if ring["kind"] == "plate" and sprites:
        out = np.array(host, np.float32, copy=True) if out is None else out
        _seat(out, centre, r, ring, t, sprites, OrderedDict() if tiles is None else tiles, eyes)
    return host if out is None else out


def _passes(ring, r, t, px, inner, c, centre):
    """[(inner px, outer px, paint(img, dx, dy, rho) -> img)]: the ring's layers in the order they are laid, each with
    the annulus it covers."""
    big = r * ring["size"]
    out = []
    if ring["kind"] in SHADOWS:
        sx, sy, soft, _ = SHADOWS[ring["kind"]]
        out.append((big - 1.0, big * (1.0 + math.hypot(sx, sy) + 2.0 * soft) + 1.0,
                    lambda img, dx, dy, rho: _shadow(img, dx, dy, r, ring)))
    if ring["kind"] == "vinyl":
        l1 = PT.vinyl_layout(ring)["label"][1] * r
        out.append((l1 - 1.0, big + 1.0, lambda img, dx, dy, rho: _record(img, dx, dy, rho, r, ring, t)))
        out.append((r - 1.5, l1 + 1.0, lambda img, dx, dy, rho: _label(img, dx, dy, rho, r, ring, t)))
    elif ring["kind"] == "plate":
        out.append((r - 1.5, big + 1.0, lambda img, dx, dy, rho: _plate(img, dx, dy, rho, r, ring)))
    else:
        sample = _sampler(inner, c)
        out.append((r - 1.5, r * (ring["size"] + GLOW[2]) + 1.0, lambda img, dx, dy, rho: _glow(img, rho, r, ring)))
        for band in ring["bands"]:
            out.append((band["a"] * r - 1.0, band["b"] * r + 1.0,
                        lambda img, dx, dy, rho, band=band: _tube(img, dx, dy, rho, r, ring, band, t, px, sample, centre)))
    return out


def _reach(ring, r):
    """How far from the window's centre a ring draws anything, px: its size, and its shadow, its glow or its sprites."""
    if ring["kind"] in SHADOWS:
        sx, sy, soft, _ = SHADOWS[ring["kind"]]
        reach = r * ring["size"] * (1.0 + math.hypot(sx, sy) + 2.0 * soft) + 1.0
        if ring["kind"] == "plate" and ring["sprites"]:
            reach = max(reach, r * (PT.plate_layout(ring)["seat"] + 0.75 * ring["scale"] * (1.0 + SPRITE_SHADOW[2] * 3.0)))
        return reach
    return r * (ring["size"] + GLOW[2]) + 1.0


# ================================================================================================= shared
def smoothstep(e0, e1, x):
    u = np.clip((x - np.float32(e0)) / np.float32(e1 - e0), 0.0, 1.0)
    return u * u * (3.0 - 2.0 * u)


def _rgb(c):
    return np.asarray(c, np.float32)


def _over(img, colour, a):
    """`colour` (rgb, or a picture of img's shape) laid on `img` (h, w, 3) with the coverage `a` (h, w)."""
    return img + (np.asarray(colour, np.float32) - img) * a[..., None]


def _inside(rho, edge):
    """The coverage of a disc of radius `edge` px about the centre, antialiased over a pixel."""
    return np.clip(np.float32(edge + 0.5) - rho, 0.0, 1.0)


def _band(rho, lo, hi):
    """The coverage of the annulus `lo` .. `hi` px from the centre, antialiased over a pixel."""
    return np.clip(np.minimum(rho - np.float32(lo), np.float32(hi) - rho) + 0.5, 0.0, 1.0)


def _shadow(img, dx, dy, r, ring):
    """The soft shadow a record or a plate casts on what is under it, to the lower right (away from the light)."""
    sx, sy, soft, alpha = SHADOWS[ring["kind"]]
    big = r * ring["size"]
    d = cv2.magnitude(dx - np.float32(sx * big), dy - np.float32(sy * big))
    s = soft * big
    return _over(img, ring["shade"], np.float32(alpha) * (1.0 - smoothstep(big - 2.0 * s, big + 2.0 * s, d)))


def _angles(dx, dy):
    """Each pixel's angle round the centre, radians counter-clockwise from the right with y up."""
    return np.arctan2(-dy, dx)


def _wrap(a):
    """Radians into -pi .. pi."""
    return np.mod(a + np.float32(math.pi), np.float32(2.0 * math.pi)) - np.float32(math.pi)


def _arc(deg, lo, hi, rho):
    """The coverage of the arc from `lo` to `hi` degrees (counter-clockwise, hi - lo below 360) at the pixels' angles `deg`
    (0 .. 360), antialiased over a pixel at their radius `rho`."""
    x = np.mod(deg - np.float32(lo), np.float32(360.0))
    span = np.float32(hi - lo)
    d = np.where(x <= span, np.minimum(x, span - x), -np.minimum(x - span, np.float32(360.0) - x))
    return np.clip(np.radians(d) * rho + 0.5, 0.0, 1.0)


# ================================================================================================= vinyl
@lru_cache(maxsize=4)
def _unevenness(n, seed=5):
    """The grooves' unevenness: n smooth samples in -1 .. 1 across the grooves' band."""
    v = np.random.default_rng(seed).standard_normal(n + 8).astype(np.float32)
    v = np.convolve(v, np.float32([0.25, 0.5, 0.25]), mode="same")[4:-4]
    return v / np.float32(max(float(np.abs(v).max()), 1e-6))


@lru_cache(maxsize=4)
def _print_lines(seed=11, n=2048):
    """The label's small print: two lines of word-like dashes under its middle, as coverage along the record's own angle
    (n samples a turn, counter-clockwise from its right), and their running sums: ((line, sums), (line, sums))."""
    rng = np.random.default_rng(seed)
    out = []
    for lo, hi in ((208.0, 332.0), (236.0, 304.0)):
        line = np.zeros(n, np.float32)
        a = lo
        while a < hi:
            b = min(a + rng.uniform(4.0, 13.0), hi)
            line[int(a / 360.0 * n):int(b / 360.0 * n)] = 1.0
            a = b + rng.uniform(2.5, 4.5)
        out.append((line, np.concatenate([[0.0], np.cumsum(np.concatenate([line, line, line]))]).astype(np.float32)))
    return tuple(out)


def _along(table, deg, rho):
    """A line of print (`_print_lines`) read at `deg` degrees, box-filtered over a pixel at radius `rho` (so the print
    stays smooth when the label is small)."""
    line, csum = table
    n = len(line)
    span = np.clip(np.float32(n / (2.0 * math.pi)) / np.maximum(rho, 1.0), 1.0, n / 8.0)    # samples per pixel there
    x = np.mod(deg, np.float32(360.0)) * np.float32(n / 360.0) + np.float32(n)
    return (_interp(csum, x + span / 2.0) - _interp(csum, x - span / 2.0)) / span


def _interp(csum, x):
    i = np.clip(np.floor(x).astype(np.int32), 0, len(csum) - 2)
    f = x - i
    return csum[i] * (1.0 - f) + csum[i + 1] * f


def _sheen(phi, t):
    """The sheen's two wedges toward and away from the light at the pixels' angles `phi`, swinging a little with `t`."""
    at = math.radians(LIGHT_AT + SHEEN_SWING[0] * math.sin(2.0 * math.pi * t / SHEEN_SWING[1]))
    sig = np.float32(math.radians(SHEEN_WIDTH))
    return (np.exp(-(_wrap(phi - np.float32(at)) / sig) ** 2)
            + np.exp(-(_wrap(phi - np.float32(at + math.pi)) / sig) ** 2))


def _record(img, dx, dy, rho, r, ring, t):
    """The record from the label out: the dead wax, the grooves and the lead-in edge under the sheen."""
    lay = PT.vinyl_layout(ring)
    q = rho / np.float32(r)
    phi = _angles(dx, dy)
    o = phi + np.float32(math.radians(PT.spin(ring, t)))                      # the record's own angle at each pixel
    light, black = _rgb(ring["light"]), _rgb(ring["color"])
    wedge = _sheen(phi, t)
    shimmer = 1.0 + 0.16 * (np.sin(3.0 * o + 0.7) + 0.6 * np.sin(7.0 * o + 2.1) + 0.4 * np.sin(13.0 * o + 4.0))
    # the grooves: their unevenness across the band, smooth in the gaps between tracks, faded where finer than a pixel
    g0, g1 = lay["grooves"]
    band_px = (g1 - g0) * r
    amp = float(np.clip((band_px / GROOVES - 0.7) / 1.3, 0.0, 1.0))
    u = np.clip((q - np.float32(g0)) / np.float32(g1 - g0), 0.0, 1.0) * np.float32(GROOVES - 1)
    i = np.minimum(u.astype(np.int32), GROOVES - 2)
    f = u - i
    table = _unevenness(GROOVES)
    grooved = _band(rho, g0 * r, g1 * r)
    gaps = np.zeros_like(q)
    width = max(1.2, 0.004 * band_px) / r
    for tr in lay["tracks"]:
        gaps = np.maximum(gaps, np.exp(-((q - np.float32(tr)) / np.float32(width)) ** 2))
    rough = (table[i] * (1.0 - f) + table[i + 1] * f) * np.float32(amp) * grooved * (1.0 - gaps)
    sheen = np.float32(ring["sheen"]) * (0.05 + wedge * shimmer * (0.6 + 0.3 * rough - 0.22 * gaps * grooved))
    col = black[None, None, :] * (1.0 + 0.35 * rough)[..., None] + light * sheen[..., None]
    # the lead-in edge: a rounded rim, lit toward the light
    e0, e1 = lay["edge"]
    v = np.clip((q - np.float32(e0)) / np.float32(e1 - e0), 0.0, 1.0)
    lit = np.maximum(np.cos(phi - np.float32(math.radians(LIGHT_AT))), 0.0)
    col = _over(col, light, (0.28 * lit + 0.06) * np.sin(np.pi * v) * _band(rho, e0 * r, e1 * r))
    return _over(img, col, _inside(rho, ring["size"] * r))


def _label(img, dx, dy, rho, r, ring, t):
    """The label, an original design turning with the record: its ground, the arch over its top and the dot under it,
    two lines of print, the ring at its rim; paper, with a little of the sheen."""
    l0, l1 = PT.vinyl_layout(ring)["label"]
    span = (l1 - l0) * r
    at = lambda share: (l0 + share * (l1 - l0)) * r            # noqa: E731  px from the centre, a share across the label
    phi = _angles(dx, dy)
    turn = PT.spin(ring, t)
    own = np.mod(np.degrees(phi) + np.float32(turn), np.float32(360.0))       # the record's own angle at each pixel
    ground, arch, ink = (_rgb(x) for x in ring["label"])
    paper = np.broadcast_to(ground, img.shape).astype(np.float32)
    paper = _over(paper, arch, _band(rho, at(0.56), at(0.8)) * _arc(own, 32.0, 148.0, rho))
    dot = math.radians(270.0 - turn)                                         # its place on the screen, turning
    paper = _over(paper, arch, np.clip(np.float32(0.1 * span + 0.5) - cv2.magnitude(
        dx - np.float32(at(0.68) * math.cos(dot)), dy + np.float32(at(0.68) * math.sin(dot))), 0.0, 1.0))
    first, second = _print_lines()
    cover = np.maximum(_band(rho, at(0.2), at(0.31)) * _along(first, own, rho),
                       _band(rho, at(0.37), at(0.45)) * _along(second, own, rho))
    paper = _over(paper, ink, np.maximum(cover, _band(rho, at(0.88), at(0.93))))
    across = np.clip((rho - np.float32(l0 * r)) / np.float32(span), 0.0, 1.0)
    paper = (paper * (0.95 + 0.05 * (1.0 - across))[..., None]
             + _rgb(ring["light"]) * (0.05 * np.float32(ring["sheen"]) * _sheen(phi, t))[..., None])
    return _over(img, paper, _inside(rho, l1 * r))


# ================================================================================================= plate
def _slope(q, lay):
    """The plate's rise per window radius outward (its height's derivative), at `q` window radii: up the slope from the
    well, a gentle dome across the rim, a small rise and the roll down over the lip; continuous throughout."""
    (s0, s1), (f0, f1), (b0, b1) = lay["slope"], lay["flange"], lay["bead"]
    us = np.clip((q - np.float32(s0)) / np.float32(s1 - s0), 0.0, 1.0)
    uf = np.clip((q - np.float32(f0)) / np.float32(f1 - f0), 0.0, 1.0)
    ub = np.clip((q - np.float32(b0)) / np.float32(b1 - b0), 0.0, 1.0)
    return np.where(q < s1, 0.85 * np.sin(np.pi * us),
                    np.where(q < f1, 0.08 * np.sin(2.0 * np.pi * uf),
                             0.45 * np.sin(np.pi * 0.8 * ub) - 2.0 * ub ** 3)).astype(np.float32)


def _plate(img, dx, dy, rho, r, ring):
    lay = PT.plate_layout(ring)
    q = rho / np.float32(r)
    inv = 1.0 / np.maximum(rho, np.float32(1e-3))
    hp = _slope(q, lay)
    nx, ny = -hp * dx * inv, -hp * dy * inv                                  # a rise outward tilts it inward
    norm = 1.0 / np.sqrt(nx * nx + ny * ny + 1.0)
    nx, ny, nz = nx * norm, ny * norm, norm
    diffuse = np.maximum(nx * LIGHT[0] + ny * LIGHT[1] + nz * LIGHT[2], 0.0) / LIGHT[2]
    nh = np.maximum(nx * HALF[0] + ny * HALF[1] + nz * HALF[2], 0.0)
    light, glaze = _rgb(ring["light"]), _rgb(ring["color"])
    col = np.broadcast_to(glaze, img.shape).astype(np.float32)
    if ring["band"] is not None:
        for lo, hi in lay["bands"]:
            col = _over(col, ring["band"], _band(rho, lo * r, hi * r))
    col = col * (0.52 + 0.4 * diffuse)[..., None]                           # a flat glaze at 0.92: its highlights show
    phi = _angles(dx, dy)
    f0, f1 = lay["flange"]
    mid, sig = (f0 + f1) / 2.0, 0.24 * (f1 - f0)
    streak = np.zeros_like(q)
    for at, spread, k in GLAZE:
        streak += np.float32(k) * np.exp(-(_wrap(phi - np.float32(math.radians(at))) / np.float32(math.radians(spread))) ** 2)
    streak *= np.exp(-((q - np.float32(mid)) / np.float32(sig)) ** 2)
    spec = 0.85 * nh ** 140 + 0.1 * nh ** 12 + streak
    col = np.minimum(col + light * spec[..., None], 1.0)
    return _over(img, col, _inside(rho, ring["size"] * r))


def _seat(out, centre, r, ring, t, sprites, tiles, eyes):
    """The plate's sprites standing round its rim (on `out`, in place): each one's tile drawn at a size near its own (a
    step of SPRITE_STEP) and scaled, on a soft shadow, its googly eyes swinging when it has some."""
    from . import eyes as EYD
    from . import mandala as MDF
    seat, size = PT.plate_layout(ring)["seat"] * r, ring["scale"] * r
    if size < 2.0:
        return
    step = SPRITE_STEP ** round(math.log(size, SPRITE_STEP))
    dxs, dys, blur, alpha = SPRITE_SHADOW
    shadow = (tuple(ring["shade"]), (dxs * size, dys * size), blur * size, alpha)
    if ring["face"] is not None and eyes is not None and eyes.get("circle_at") is not None:
        eyes = dict(eyes, circle_at=_once(eyes["circle_at"]))     # every copy's eyes ask for the same clip seconds
    for angle, j in PT.seats(ring, t):
        path = str(sprites[PT.sprite_pick(ring, j, len(sprites))])
        key = (path, round(step, 4))
        tile = tiles.get(key)
        if tile is None:
            tile = tiles[key] = MDF.sprite_tile(path, step, 0.0, face=ring["face"])
            while len(tiles) > TILES:
                tiles.popitem(last=False)
        else:
            tiles.move_to_end(key)
        a = math.radians(angle)
        at = (centre[0] - seat * math.sin(a), centre[1] - seat * math.cos(a))
        M_ = MDF.placed(tile, at, angle if ring["orient"] == "radial" else 0.0, size / step)
        pupils = None
        if ring["face"] is not None and eyes is not None:
            pupils = EYD.pupils(ring["face"], tile, eyes["t"], hits=eyes["hits"], key=(j,),
                                maps=_maps(tile, ring, j, step, eyes),
                                look=EYD.look_at(ring["face"], out.shape[1::-1], centre))
        MDF.lay(out, tile, M_, None, shadow, pupils=pupils)


def _once(circle_at):
    """`circle_at` remembering its last answer (the copies of one frame all ask for the same seconds)."""
    held = {}

    def get(ts):
        ts = np.atleast_1d(np.asarray(ts, np.float64))
        key = (len(ts), float(ts[0]), float(ts[-1]))
        if held.get("key") != key:
            held["key"], held["got"] = key, circle_at(ts)
        return held["got"]
    return get


def _maps(tile, ring, j, step, eyes):
    """callable(clip seconds) -> the affine maps ((n, 2, 3), as mkmmd.mandala.placed makes them) that laid sprite j's
    tile then (how it rode the rim and grew with the window), for its eyes' swing; None when the caller cannot say
    where the window was."""
    circle_at = eyes.get("circle_at")
    if circle_at is None:
        return None
    seat_share = PT.plate_layout(ring)["seat"]
    ax, ay = tile.anchor

    def maps(ts):
        got = circle_at(ts)
        cx = np.array([g[0][0] for g in got], np.float64)
        cy = np.array([g[0][1] for g in got], np.float64)
        r = np.array([g[1] for g in got], np.float64)
        tw = np.array([g[2] for g in got], np.float64)
        a = np.radians((360.0 * j / ring["count"] + ring["turn"] * tw) % 360.0)
        x, y = cx - seat_share * r * np.sin(a), cy - seat_share * r * np.cos(a)
        k = np.maximum(ring["scale"] * r, 1e-3) / step
        turn = a if ring["orient"] == "radial" else np.zeros_like(a)
        c, s = np.cos(turn) * k, np.sin(turn) * k
        out = np.empty((len(r), 2, 3), np.float64)
        out[:, 0, 0], out[:, 0, 1], out[:, 1, 0], out[:, 1, 1] = c, s, -s, c
        out[:, 0, 2] = x - (c * ax + s * ay)
        out[:, 1, 2] = y - (-s * ax + c * ay)
        return out
    return maps


# ================================================================================================= lens
def _sampler(inner, c):
    """callable(xs, ys, channel) -> that channel of the window's picture at frame points (xs, ys): the inner picture as
    the window places it (scaled `k` about its centre, which lands at `at`; mkmmd.matte.warp_scaled), through a pyramid
    when it is shrunk below half size, its edges mirrored. The pyramid and the channels are made when first asked for."""
    held = {}

    def get(xs, ys, ch):
        if not held:
            src, kk = inner, float(c["k"])
            while kk < 0.5 and min(src.shape[:2]) > 2:
                src = cv2.pyrDown(src)
                kk *= 2.0
            held.update(kk=kk, chans=cv2.split(np.ascontiguousarray(src, np.float32)),
                        mid=((src.shape[1] - 1) / 2.0, (src.shape[0] - 1) / 2.0))
        kk, (ox, oy), at = held["kk"], held["mid"], c["at"]
        mx = ((xs - np.float32(at[0])) * np.float32(1.0 / kk) + np.float32(ox)).astype(np.float32)
        my = ((ys - np.float32(at[1])) * np.float32(1.0 / kk) + np.float32(oy)).astype(np.float32)
        return cv2.remap(held["chans"][ch], mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    return get


def _glow(img, rho, r, ring):
    """The tint's glow over and round the rings: faint across them, brightest at their outer edge, fading beyond."""
    q = rho / np.float32(r)
    size = ring["size"]
    glow = (GLOW[0] * (1.0 - smoothstep(size, size + GLOW[2], q))
            + GLOW[1] * np.exp(-((q - np.float32(size)) / np.float32(0.1)) ** 2))
    return _over(img, ring["tint"], glow.astype(np.float32))


def _tube(img, dx, dy, rho, r, ring, band, t, px, sample, centre):
    """One lens ring, a glass tube with a chrome sheen, broken where its breaks are."""
    a, b = np.float32(band["a"] * r), np.float32(band["b"] * r)
    cover = np.clip(np.minimum(rho - a, b - rho) + 0.5, 0.0, 1.0)
    phi = _angles(dx, dy)
    if band["breaks"]:                                       # the breaks turn with their ring
        own = np.mod(np.degrees(phi) - np.float32(band["phase"] + band["turn"] * t), np.float32(360.0))
        for start, span in band["breaks"]:
            cover = cover * (1.0 - _arc(own, start, start + span, rho))
    # the picture through the tube: across it flipped and magnified, swirled with the ring's turn, red and blue apart
    mid, half = (a + b) * np.float32(0.5), max((b - a) * np.float32(0.5), np.float32(0.5))
    u = np.clip((rho - mid) / half, -1.0, 1.0)
    rs = mid * np.float32(band["zoom"]) - u * half * np.float32(band["flip"])
    ps = phi + np.float32(math.radians(0.5 * band["turn"] * t))
    cs, sn = np.cos(ps), np.sin(ps)
    split = np.float32(ring["chroma"] * px) * u
    pic = np.empty(img.shape, np.float32)
    for ch, off in ((0, split), (1, 0.0), (2, -split)):
        rr = rs + off
        pic[..., ch] = sample(np.float32(centre[0]) + rr * cs, np.float32(centre[1]) - rr * sn, ch)
    # the chrome: a two-tone studio reflected in the tube (its normal: radial by u, out of the frame by sqrt(1 - u²))
    inv = 1.0 / np.maximum(rho, np.float32(1e-3))
    nz = np.sqrt(np.maximum(1.0 - u * u, 0.0))
    nx, ny = u * dx * inv, u * dy * inv
    up = -(2.0 * nz * ny)                                     # the reflected ray's way up the frame
    light, shade, tint = _rgb(ring["light"]), _rgb(ring["shade"]), _rgb(ring["tint"])
    env = _over(np.broadcast_to(shade, img.shape).astype(np.float32), light, smoothstep(-0.45, 0.35, up))
    env = (env + light * (0.45 * np.exp(-((up + 0.04) / 0.07) ** 2))[..., None]) * 0.8 + tint * 0.2
    nh = np.maximum(nx * HALF[0] + ny * HALF[1] + nz * HALF[2], 0.0)
    col = (pic + (env - pic) * np.float32(CHROME)) * (0.55 + 0.45 * nz)[..., None] + light * (0.75 * nh ** 60)[..., None]
    return _over(img, np.clip(col, 0.0, 1.0), cover)
