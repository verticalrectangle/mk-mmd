"""Glitch, numpy only (docs/design.md: Shots: Glitches): a figure breaking up and vanishing over a window of frames. `mk post`
and `mk look` (mkmmd.cutfx) composite it from the cut's frame and its `bare` layer, the same frame rendered with the glitch's
objects hidden: the figure is where the two differ, its outline and the shadows it casts included.

Each frame of the window cuts the figure into horizontal slices of random heights. Some slices jump sideways (by up to
`shift` of the frame width, further as the window goes on), two copies of the figure's shape in the ghost colours trail
`split` pixels to either side of it, a block or two of ghost colour breaks it up, and a growing share of the slices drops
out. On some frames the figure blinks out whole, early on it may hold whole for a frame (a stutter), and on the window's
last frame only the bare frame is left. Every frame draws from its own seeded stream, so a render is repeatable and a
frame does not depend on the others."""
import numpy as np

EDGE = (0.03, 0.12)          # colour difference (any channel, 0..1) between the frame and the bare one: figure from the
#                              first, all figure from the second (an antialiased edge between)
BANDS = (0.012, 0.07)        # slice heights, shares of the frame height
JUMPS = 0.45                 # the share of the slices that jump on a glitching frame
POWER = (0.35, 1.3)          # how far they jump: this share of `shift` on the first frame, rising by the second per window
DROP = 1.3                   # the share of the slices dropped out is u ** DROP (u: 0 on the first frame, 1 on the last)
BLINK = (0.12, 0.3)          # the chance a frame (not the first) shows no figure: the first, rising by the second
STUTTER = 0.15               # the chance the first frames show the figure whole and still, falling to 0 by the end
GHOST = 0.85                 # the ghosts' opacity
BLOCKS = 2                   # at most this many blocks of ghost colour across the figure per frame
BLOCK_H, BLOCK_W = (0.01, 0.03), (0.04, 0.14)    # their heights and widths, shares of the frame's


def coverage(main, bare):
    """(h, w) 0..1: how much of each pixel of `main` is the figure, measured against `bare`."""
    d = np.abs(np.asarray(main, np.float32) - np.asarray(bare, np.float32)).max(axis=2)
    return np.clip((d - EDGE[0]) / (EDGE[1] - EDGE[0]), 0.0, 1.0)


def _shift(img, dx):
    """`img` moved `dx` pixels to the right (to the left when negative); the side it uncovers is zero."""
    dx = int(dx)
    if dx == 0:
        return img
    out = np.zeros_like(img)
    if abs(dx) >= img.shape[1]:
        return out
    if dx > 0:
        out[:, dx:] = img[:, :-dx]
    else:
        out[:, :dx] = img[:, -dx:]
    return out


def _bands(rng, h):
    """[(y0, y1)]: slices covering the rows, top to bottom."""
    out, y = [], 0
    while y < h:
        y1 = min(h, y + max(1, int(round(rng.uniform(*BANDS) * h))))
        out.append((y, y1))
        y = y1
    return out


def frame(main, bare, k, n, seed=1, shift=0.06, split=6.0, colors=((0.92, 0.44, 0.57), (0.61, 0.81, 0.85))):
    """Frame `k` (0 .. n - 1) of an `n`-frame glitch window: `main` the cut's frame, `bare` the same frame without the
    figure, both (h, w, 3) float RGB. `shift` is the farthest a slice jumps, a share of the frame width; `split` the
    ghosts' offset in pixels; `colors` the two ghosts' colours. Returns a new (h, w, 3) float32 image."""
    main = np.asarray(main, np.float32)
    bare = np.asarray(bare, np.float32)
    if k >= n - 1:
        return bare.copy()
    a = coverage(main, bare)
    if not a.any():
        return main.copy()
    rng = np.random.default_rng([int(seed), int(k)])
    u = k / (n - 1)
    h, w = a.shape
    roll = rng.random()
    if k > 0 and roll < BLINK[0] + BLINK[1] * u:
        return bare.copy()                                         # a blink: gone for a frame
    if roll > 1.0 - STUTTER * (1.0 - u):
        return main.copy()                                         # a stutter: whole and still for a frame
    power = min(1.0, POWER[0] + POWER[1] * u)
    drop = u ** DROP
    cov = np.zeros_like(a)
    fig = np.zeros_like(main)                                      # the figure's slices, premultiplied by their coverage
    for y0, y1 in _bands(rng, h):
        dropped = rng.random() < drop
        jump = rng.random() < JUMPS
        dx = int(round(rng.uniform(-1.0, 1.0) * shift * w * power)) if jump else 0
        if dropped:
            continue
        cov[y0:y1] = _shift(a[y0:y1], dx)
        fig[y0:y1] = _shift(main[y0:y1] * a[y0:y1, :, None], dx)
    for _ in range(int(rng.integers(0, BLOCKS + 1))):
        ys, xs = np.nonzero(cov > 0.5)
        if not len(ys):
            break
        i = int(rng.integers(len(ys)))
        bh = max(1, int(round(rng.uniform(*BLOCK_H) * h)))
        bw = max(1, int(round(rng.uniform(*BLOCK_W) * w)))
        y0, x0 = max(0, int(ys[i]) - bh // 2), max(0, int(xs[i]) - bw // 2)
        col = np.asarray(colors[int(rng.integers(2))], np.float32)
        fig[y0:y0 + bh, x0:x0 + bw] = col * cov[y0:y0 + bh, x0:x0 + bw, None]
    out = bare.copy()
    s = int(round(split * (0.5 + power)))
    for col, d in ((colors[0], -s), (colors[1], s)):                # the ghosts, behind the figure
        g = (_shift(cov, d) * GHOST)[..., None]
        out = out * (1.0 - g) + np.asarray(col, np.float32) * g
    return out * (1.0 - cov[..., None]) + fig
