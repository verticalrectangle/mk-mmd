"""Metrics on rendered frames: flicker (temporal noise) and palette (near-black share, distance to a palette)."""
import numpy as np
from PIL import Image

from . import CheckError, Metric, metric

PALETTES = {
    "rose-pine-moon": ["#232136", "#2a273f", "#393552", "#6e6a86", "#908caa", "#e0def4", "#eb6f92", "#f6c177",
                       "#ea9a97", "#3e8fb0", "#9ccfd8", "#c4a7e7", "#2a283e", "#44415a", "#56526e"],
    "rose-pine": ["#191724", "#1f1d2e", "#26233a", "#6e6a86", "#908caa", "#e0def4", "#eb6f92", "#f6c177",
                  "#ebbcba", "#31748f", "#9ccfd8", "#c4a7e7", "#21202e", "#403d52", "#524f67"],
    "rose-pine-dawn": ["#faf4ed", "#fffaf3", "#f2e9e1", "#9893a5", "#797593", "#575279", "#b4637a", "#ea9d34",
                       "#d7827e", "#286983", "#56949f", "#907aa9", "#f4ede8", "#dfdad9", "#cecacd"],
}


def _load(path, width):
    im = Image.open(path).convert("RGB")
    if width and im.width > width:
        im = im.resize((width, max(1, round(im.height * width / im.width))), Image.BILINEAR)
    return np.asarray(im, np.float32) / 255.0


def _linear(c):
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _lab(rgb):
    lin = _linear(rgb)
    M = np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]])
    xyz = lin @ M.T / np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], -1)


def _hex(h):
    h = h.lstrip("#")
    return np.array([int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4)])


def luma(rgb):
    return rgb @ np.array([0.2126, 0.7152, 0.0722], np.float32)


@metric
class Flicker(Metric):
    name = "flicker"
    doc = ("Temporal noise between consecutive rendered frames (shadow / sampling / texture flicker): per 8x8 block, "
           "|L(t) - (L(t-1) + L(t+1)) / 2| of luma; motion makes this large too, so compare shots with themselves. "
           "Value: the worst frame's 99th percentile (0..1).")
    args = {"images": "glob of consecutive frames (relative to the project)", "width": "analysis width (default 320)"}
    sampled = False
    uses_frames = False

    def compute(self, args, ctx, data, frames, st):
        files = ctx.images(args["images"])
        if len(files) < 3:
            raise CheckError("flicker: needs at least 3 consecutive frames")
        L = np.stack([luma(_load(f, int(args.get("width", 320)))) for f in files])
        h, w = (L.shape[1] // 8) * 8, (L.shape[2] // 8) * 8
        B = L[:, :h, :w].reshape(len(L), h // 8, 8, w // 8, 8).mean((2, 4))
        d = np.abs(B[1:-1] - 0.5 * (B[:-2] + B[2:]))
        per = np.percentile(d.reshape(len(d), -1), 99, axis=1)
        k = int(np.argmax(per))
        return float(per[k]), {"frames": len(files), "worst": files[k + 1].name,
                               "median_frame_p99": round(float(np.median(per)), 4)}


@metric
class Palette(Metric):
    name = "palette"
    doc = ("Colour discipline of rendered frames: the share of near-black pixels (luma below `black`), and the median "
           "CIE76 distance of pixels to the nearest palette colour. Value: near-black share (or the distance with "
           "measure = \"distance\"), worst image.")
    args = {"images": "glob of frames", "palette": "name (" + ", ".join(PALETTES) + ") or list of hex colours",
            "black": "luma threshold for near-black (default 0.035)", "measure": "black | distance",
            "width": "analysis width (default 240)"}
    sampled = False
    uses_frames = False

    def compute(self, args, ctx, data, frames, st):
        files = ctx.images(args["images"])
        pal = args.get("palette", "rose-pine-moon")
        cols = PALETTES[pal] if isinstance(pal, str) else pal
        pal_lab = _lab(np.array([_hex(c) for c in cols]))
        thr = float(args.get("black", 0.035))
        rows = []
        for f in files:
            rgb = _load(f, int(args.get("width", 240))).reshape(-1, 3)
            black = float((luma(rgb) < thr).mean())
            lab = _lab(rgb)
            d = np.linalg.norm(lab[:, None, :] - pal_lab[None], axis=2).min(1)
            rows.append((f.name, black, float(np.median(d)), float(np.percentile(d, 90))))
        key = 2 if args.get("measure", "black") == "distance" else 1
        k = int(np.argmax([r[key] for r in rows]))
        return rows[k][key], {"images": len(rows), "worst": rows[k][0],
                              "near_black_max": round(max(r[1] for r in rows), 4),
                              "near_black_median": round(float(np.median([r[1] for r in rows])), 4),
                              "distance_median": round(float(np.median([r[2] for r in rows])), 2),
                              "distance_p90_max": round(max(r[3] for r in rows), 2)}
