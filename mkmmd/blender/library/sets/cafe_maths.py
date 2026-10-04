"""Pure-numpy maths of the cafe window glass (no bpy, importable from tests): the opening and its UV map, the
panes, the condensation noise image and the deterministic rain-splat list.

Glass UV: one map for the whole window. u = (y + 1.75) / 2.4 and v = (z - 0.82) / 2.03 over the opening in the wall plane
x = WINDOW_X (the room frame of cafe_room), so u = 0 is the edge nearest the front cameras and v = 0 the bottom."""
import os
import struct
import zlib

import numpy as np

# ---------------------------------------------------------------- opening / UV mapping (metres, room frame)
OPEN_Y = (-1.75, 0.65)       # opening span along y (u direction)
OPEN_Z = (0.82, 2.85)        # opening span along z (v direction)
OPEN_W = OPEN_Y[1] - OPEN_Y[0]       # 2.40
OPEN_H = OPEN_Z[1] - OPEN_Z[0]       # 2.03

FOG_SIZE = (2048, 1732)      # (W, H): ~square texels over the 2.40 x 2.03 m opening
FOG_SEED = 7
FOG_SOFT = 0.11              # smoothstep half-width in noise units (the image is histogram-equalised: noise ~ U[0,1])
FOG_DIR = os.path.join(os.path.expanduser("~"), ".cache", "mk", "cafe")

FRAME_W = 0.09                              # painted frame member width (visible face)
BAR_W = 0.05                                # mullion bar width
BARS_Y = (-1.12, -0.45, 0.22)               # vertical bar centres (col 0 is nearest the front cameras)
BARS_Z = (1.52, 2.02, 2.46)                 # horizontal bar centres (row 0 is the bottom row)
SPLAT_SEED = 11


def uv_from_world(y, z):
    """World (y, z) on the glass plane -> glass (u, v)."""
    return (np.asarray(y) - OPEN_Y[0]) / OPEN_W, (np.asarray(z) - OPEN_Z[0]) / OPEN_H


# ---------------------------------------------------------------- fog noise image
def _blobs(rng, shape, sx, sy):
    """Smooth Gaussian random field (zero mean, unit std), correlation lengths sx, sy in pixels (FFT filtering)."""
    h, w = shape
    fy = np.fft.fftfreq(h)[:, None]
    fx = np.fft.fftfreq(w)[None, :]
    amp = np.exp(-2.0 * np.pi ** 2 * ((sx * fx) ** 2 + (sy * fy) ** 2))
    amp[0, 0] = 0.0
    f = np.fft.ifft2(np.fft.fft2(rng.standard_normal((h, w))) * amp).real
    f -= f.mean()
    return f / f.std()


def make_fog_noise(size=FOG_SIZE, seed=FOG_SEED):
    """Condensation field, float32 (H, W) in 0..1, uniformly distributed (rank-equalised) so that a threshold at
    1 - L covers a fraction L of the glass. Feathered round patches (taller than wide: condensation drips) over a
    finer bead grain; the bottom of the glass fogs first."""
    w, h = size
    rng = np.random.default_rng(seed)
    big = _blobs(rng, (h, w), 85.0, 140.0)             # ~10-16 cm patches
    mid = _blobs(rng, (h, w), 26.0, 52.0)              # ~3-6 cm
    fine = _blobs(rng, (h, w), 7.0, 9.0)               # ~1 cm grain
    field = 1.0 * big + 0.42 * mid + 0.14 * fine
    vv = 1.0 - (np.arange(h) + 0.5) / h                # v of each row (row 0 is the top)
    field = field + 0.9 * (1.0 - vv)[:, None] ** 1.3 * field.std() * 0.5
    ranks = np.empty(h * w, np.float64)
    ranks[np.argsort(field, axis=None)] = np.arange(h * w)
    return (ranks / (h * w - 1)).reshape(h, w).astype(np.float32)


def _chunk(tag, data):
    return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)


def write_png16(path, a):
    """16-bit grayscale PNG from a uint16 (H, W) array, 'Up' filtered (smooth fields compress well). No PIL needed."""
    h, w = a.shape
    rows = np.frombuffer(a.astype(">u2").tobytes(), np.uint8).reshape(h, w * 2)
    prev = np.vstack([np.zeros((1, w * 2), np.uint8), rows[:-1]])
    out = np.empty((h, w * 2 + 1), np.uint8)
    out[:, 0] = 2
    out[:, 1:] = (rows.astype(np.int16) - prev.astype(np.int16)).astype(np.uint8)
    png = (b"\x89PNG\r\n\x1a\n" + _chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 16, 0, 0, 0, 0))
           + _chunk(b"IDAT", zlib.compress(out.tobytes(), 9)) + _chunk(b"IEND", b""))
    with open(path, "wb") as f:
        f.write(png)


def ensure_fog_png(size=FOG_SIZE, seed=FOG_SEED):
    """Path of the fog image (generated on first use, then cached for every project)."""
    path = os.path.join(FOG_DIR, f"fog_noise_{seed}_{size[0]}x{size[1]}.png")
    if not os.path.exists(path):
        os.makedirs(FOG_DIR, exist_ok=True)
        tmp = f"{path}.{os.getpid()}.tmp"                 # concurrent builds must not share a temp file
        write_png16(tmp, np.clip(np.round(make_fog_noise(size, seed) * 65535.0), 0, 65535).astype(np.uint16))
        os.replace(tmp, path)
    return path


# ---------------------------------------------------------------- panes and rain splats
def pane_rects():
    """Visible glass rectangle of every pane: [{'row', 'col', 'y0', 'y1', 'z0', 'z1'}], row 0 = bottom, col 0 = front."""
    ys = [OPEN_Y[0] + FRAME_W]
    for yc in BARS_Y:
        ys += [yc - BAR_W / 2, yc + BAR_W / 2]
    ys.append(OPEN_Y[1] - FRAME_W)
    zs = [OPEN_Z[0] + FRAME_W]
    for zc in BARS_Z:
        zs += [zc - BAR_W / 2, zc + BAR_W / 2]
    zs.append(OPEN_Z[1] - FRAME_W)
    return [{"row": r, "col": c, "y0": ys[2 * c], "y1": ys[2 * c + 1], "z0": zs[2 * r], "z1": zs[2 * r + 1]}
            for r in range(4) for c in range(4)]


def default_ticks(duration):
    """Splat times (clip seconds) when the project gives none: every 1.5 s."""
    return [round(1.5 * k, 6) for k in range(1, int(duration / 1.5 + 1e-9) + 1) if 1.5 * k < duration]


def splat_drops(ticks):
    """Deterministic splat list [(t, u, v, radius_m), ...], one per tick. Each lands inside a pane (never on a bar);
    the panes are visited in a fixed shuffled order (never the same column twice in a row); radius 13..20 mm."""
    rng = np.random.default_rng(SPLAT_SEED)
    panes = pane_rects()
    order = []
    while len(order) < len(ticks):
        pool = [k for k in rng.permutation(len(panes)) if k not in order[-8:]]
        pick = next((k for k in pool if not order or panes[k]["col"] != panes[order[-1]]["col"]), pool[0])
        order.append(int(pick))
    out = []
    for i, t in enumerate(ticks):
        p = panes[order[i]]
        r = float(rng.uniform(0.013, 0.020))
        m = 0.06
        y = rng.uniform(p["y0"] + m, p["y1"] - m)
        z = rng.uniform(p["z0"] + m, p["z1"] - m)
        u, v = uv_from_world(y, z)
        out.append((float(t), float(u), float(v), r))
    return out
