"""Marks drawn on an image (mk mark): a pixel grid to read positions, crops and zooms that keep the image's own
coordinates, a plan (lines and curves, fills, arrows, circles, boxes, text) from a TOML file, and a photo beside it.

A marks file is TOML, one array of tables per kind. A point is [x, y] in the image's pixels (from its top left) or
[x, y, z] in the world's metres, projected through the camera of the image's pixel map (mk look --map):

  [[line]]    points (2 or more), color, width (px, default 3), dashed (false), smooth (false: true runs a Catmull-Rom
              curve through the points), closed (false), name (for fills)
  [[fill]]    points (3 or more) or between = ["line a", "line b"] (along line a, then back along line b), color,
              opacity (0.35), smooth (false)
  [[arrow]]   from, to, color, width
  [[circle]]  at, radius (px), color, width
  [[box]]     box = [x, y, w, h], color, width
  [[text]]    at (its top left), text (new lines split it), color (white), size (px, 20), back (the box behind it,
              "#28203c"; "none" for none), opacity of the box (0.88)

Kinds are drawn in that order: fills under the lines, text on top. Colours are anything Pillow reads (#rrggbb, names).
Widths and sizes are pixels of the finished picture, so a zoomed crop keeps its lines crisp."""
import itertools
import tomllib

import numpy as np
from PIL import Image, ImageColor, ImageDraw

from . import fonts

KEYS = {"line": {"points", "color", "width", "dashed", "smooth", "closed", "name"},
        "fill": {"points", "between", "color", "opacity", "smooth"},
        "arrow": {"from", "to", "color", "width"},
        "circle": {"at", "radius", "color", "width"},
        "box": {"box", "color", "width"},
        "text": {"at", "text", "color", "size", "back", "opacity"}}
GRID = (40, 90, 230)
TEXT_BACK = "#28203c"


class MarkError(ValueError):
    pass


def load(path):
    """The marks of a TOML file, checked (see the module docstring): {kind: [table, ...]}."""
    try:
        with open(path, "rb") as fh:
            doc = tomllib.load(fh)
    except (OSError, tomllib.TOMLDecodeError) as e:
        raise MarkError(f"{path}: {e}") from None
    return check(doc, str(path))


def check(doc, where="marks"):
    out = {}
    for kind, items in doc.items():
        if kind not in KEYS:
            raise MarkError(f"{where}: unknown kind [[{kind}]] (kinds: {', '.join(KEYS)})")
        if not isinstance(items, list):
            raise MarkError(f"{where}: {kind} must be an array of tables, [[{kind}]]")
        for i, m in enumerate(items):
            bad = set(m) - KEYS[kind]
            if bad:
                raise MarkError(f"{where}: [[{kind}]] {i + 1}: unknown {', '.join(sorted(bad))} (keys: "
                                f"{', '.join(sorted(KEYS[kind]))})")
            need = {"line": ["points"], "arrow": ["from", "to"], "circle": ["at", "radius"], "box": ["box"],
                    "text": ["at", "text"], "fill": []}[kind]
            if kind == "fill" and ("points" in m) == ("between" in m):
                raise MarkError(f"{where}: [[fill]] {i + 1}: give points or between, one of them")
            miss = [k for k in need if k not in m]
            if miss:
                raise MarkError(f"{where}: [[{kind}]] {i + 1}: needs {', '.join(miss)}")
            for k in ("color", "back"):
                if k in m and not (k == "back" and m[k] == "none"):
                    try:
                        ImageColor.getrgb(str(m[k]))
                    except ValueError:
                        raise MarkError(f"{where}: [[{kind}]] {i + 1}: {k} {m[k]!r} is not a colour") from None
        out[kind] = list(items)
    names = [m.get("name") for m in out.get("line", []) if m.get("name")]
    for i, f in enumerate(out.get("fill", [])):
        for n in f.get("between", []):
            if n not in names:
                raise MarkError(f"{where}: [[fill]] {i + 1}: between names line {n!r}, which no [[line]] has")
    return out


def catmull(P, closed=False, n=16):
    """Points of a uniform Catmull-Rom curve through P (k, 2)."""
    P = np.asarray(P, float)
    if len(P) < 3:
        return P
    Q = np.concatenate([P[-1:], P, P[:2]]) if closed else np.concatenate([P[:1], P, P[-1:]])
    segs = len(P) if closed else len(P) - 1
    t = np.linspace(0.0, 1.0, n, endpoint=False)[:, None]
    out = []
    for i in range(segs):
        p0, p1, p2, p3 = Q[i], Q[i + 1], Q[i + 2], Q[i + 3]
        out.append(0.5 * (2 * p1 + (p2 - p0) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t ** 2
                          + (3 * p1 - p0 - 3 * p2 + p3) * t ** 3))
    out.append(Q[segs + 1][None] if not closed else Q[1][None])
    return np.concatenate(out)


class View:
    """Where image pixels land on the drawn picture: a crop's top left `origin` and a `zoom`; `project` takes world
    points (n, 3) to image pixels (x, y) (a pixel map's camera), or is None."""

    def __init__(self, origin=(0, 0), zoom=1.0, project=None):
        self.origin, self.zoom, self.project = np.asarray(origin, float), float(zoom), project

    def __call__(self, pts, where):
        P = np.asarray(pts, float)
        if P.ndim != 2 or P.shape[1] not in (2, 3):
            raise MarkError(f"{where}: points are [x, y] (pixels) or [x, y, z] (metres)")
        if P.shape[1] == 3:
            if self.project is None:
                raise MarkError(f"{where}: [x, y, z] points need the image's pixel map (--map)")
            x, y, d = self.project(P)
            if (np.asarray(d) <= 0).any():
                raise MarkError(f"{where}: a point lies behind the camera")
            P = np.stack([x, y], 1)
        return (P - self.origin) * self.zoom


def _dashed(draw, pts, fill, width, dash, gap):
    pts = [np.asarray(p, float) for p in pts]
    left, on = dash, True
    for a, b in itertools.pairwise(pts):
        seg, t = float(np.linalg.norm(b - a)), 0.0
        while t < seg:
            step = min(left, seg - t)
            if on:
                draw.line([tuple(a + (b - a) * t / seg), tuple(a + (b - a) * (t + step) / seg)], fill=fill, width=width)
            t += step
            left -= step
            if left <= 1e-9:
                on = not on
                left = dash if on else gap


def _line_points(m, view, where):
    P = view(m["points"], where)
    if len(P) < 2:
        raise MarkError(f"{where}: a line has at least two points")
    closed = bool(m.get("closed", False))
    if m.get("smooth"):
        P = catmull(P, closed)
    elif closed:
        P = np.concatenate([P, P[:1]])
    return P


def draw(im, marks, view=None):
    """`im` (PIL RGB) with `marks` (as `check` returns them) drawn over it, in `view`'s coordinates."""
    view = view or View()
    im = im.convert("RGBA")
    lines = {}
    for i, m in enumerate(marks.get("line", [])):
        P = _line_points(m, view, f"[[line]] {i + 1}")
        lines[m.get("name") or f"#{i}"] = P
    over = Image.new("RGBA", im.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(over)
    for i, m in enumerate(marks.get("fill", [])):
        where = f"[[fill]] {i + 1}"
        if "between" in m:
            a, b = (lines[n] for n in m["between"])
            P = np.concatenate([a, b[::-1]])
        else:
            P = view(m["points"], where)
            if len(P) < 3:
                raise MarkError(f"{where}: a fill has at least three points")
            P = catmull(P, closed=True) if m.get("smooth") else P
        rgb = ImageColor.getrgb(str(m.get("color", "#7846aa")))[:3]
        d.polygon([tuple(p) for p in P], fill=rgb + (round(255 * float(m.get("opacity", 0.35))),))
    im = Image.alpha_composite(im, over)
    d = ImageDraw.Draw(im)
    for i, m in enumerate(marks.get("line", [])):
        P = lines[m.get("name") or f"#{i}"]
        w = int(m.get("width", 3))
        col = ImageColor.getrgb(str(m.get("color", "#ffe128")))
        if m.get("dashed"):
            _dashed(d, P, col, w, 4.0 * w + 4, 2.5 * w + 3)
        else:
            d.line([tuple(p) for p in P], fill=col, width=w, joint="curve")
    for i, m in enumerate(marks.get("arrow", [])):
        a, b = view([m["from"], m["to"]], f"[[arrow]] {i + 1}")
        w = int(m.get("width", 3))
        col = ImageColor.getrgb(str(m.get("color", "#e61e28")))
        d.line([tuple(a), tuple(b)], fill=col, width=w)
        u = b - a
        if np.linalg.norm(u) > 0:
            u /= np.linalg.norm(u)
            size = 4 * w + 8
            for s in (1.0, -1.0):
                ang = s * np.radians(28)
                r = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]]) @ -u
                d.line([tuple(b), tuple(b + r * size)], fill=col, width=w)
    for i, m in enumerate(marks.get("circle", [])):
        c = view([m["at"]], f"[[circle]] {i + 1}")[0]
        r = float(m["radius"]) * view.zoom
        d.ellipse([c[0] - r, c[1] - r, c[0] + r, c[1] + r], outline=ImageColor.getrgb(str(m.get("color", "#e61e28"))),
                  width=int(m.get("width", 4)))
    for i, m in enumerate(marks.get("box", [])):
        x, y, bw, bh = (float(v) for v in m["box"])
        a, b = view([[x, y], [x + bw, y + bh]], f"[[box]] {i + 1}")
        d.rectangle([tuple(np.minimum(a, b)), tuple(np.maximum(a, b))],
                    outline=ImageColor.getrgb(str(m.get("color", "#e61e28"))), width=int(m.get("width", 3)))
    texts = Image.new("RGBA", im.size, (0, 0, 0, 0))
    td = ImageDraw.Draw(texts)
    for i, m in enumerate(marks.get("text", [])):
        x, y = view([m["at"]], f"[[text]] {i + 1}")[0]
        font = fonts.label_font(int(m.get("size", 20)))
        text = str(m["text"])
        back = m.get("back", TEXT_BACK)
        if back != "none":
            l, t, r, b = td.multiline_textbbox((x, y), text, font=font)
            td.rounded_rectangle((l - 8, t - 6, r + 8, b + 6), radius=8,
                                 fill=ImageColor.getrgb(str(back))[:3] + (round(255 * float(m.get("opacity", 0.88))),))
        td.multiline_text((x, y), text, font=font, fill=ImageColor.getrgb(str(m.get("color", "white"))))
    return Image.alpha_composite(im, texts).convert("RGB")


def grid(im, view, step=50):
    """`im` with a grid of the image's own pixels: a line every `step` pixels, labelled every other one (at the top and
    the left), placed through `view` (a crop and a zoom keep the image's coordinates)."""
    im = im.convert("RGBA")
    over = Image.new("RGBA", im.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(over)
    font = fonts.label_font(13)
    W, H = im.size
    x0, y0 = view.origin
    x1, y1 = x0 + W / view.zoom, y0 + H / view.zoom
    for v in np.arange(np.ceil(x0 / step) * step, x1, step):
        X = (v - x0) * view.zoom
        major = int(np.rint(v / step)) % 2 == 0
        d.line([(X, 0), (X, H)], fill=GRID + (150 if major else 80,), width=1)
        if major:
            d.text((X + 3, 2), str(int(v)), font=font, fill=GRID + (255,), stroke_width=2, stroke_fill=(255, 255, 255))
    for v in np.arange(np.ceil(y0 / step) * step, y1, step):
        Y = (v - y0) * view.zoom
        major = int(np.rint(v / step)) % 2 == 0
        d.line([(0, Y), (W, Y)], fill=GRID + (150 if major else 80,), width=1)
        if major:
            d.text((2, Y + 2), str(int(v)), font=font, fill=GRID + (255,), stroke_width=2, stroke_fill=(255, 255, 255))
    return Image.alpha_composite(im, over).convert("RGB")


def crop(im, box, zoom=1.0):
    """(the picture, its View): `box` = (x0, y0, x1, y1) of the image's pixels (x1, y1 excluded; None: all of it),
    scaled by `zoom`."""
    box = (0, 0, im.width, im.height) if box is None else tuple(int(v) for v in box)
    x0, y0, x1, y1 = box
    if not (0 <= x0 < x1 <= im.width and 0 <= y0 < y1 <= im.height):
        raise MarkError(f"crop {list(box)} lies outside the image ({im.width} x {im.height}) or is empty")
    out = im.crop(box)
    if zoom != 1.0:
        out = out.resize((max(1, round(out.width * zoom)), max(1, round(out.height * zoom))), Image.LANCZOS)
    return out, View((x0, y0), zoom)


def beside(photo, im, pad=8):
    """The photo scaled to the picture's height on its left, on white."""
    photo = photo.convert("RGB")
    photo = photo.resize((max(1, round(photo.width * im.height / photo.height)), im.height), Image.LANCZOS)
    out = Image.new("RGB", (photo.width + pad + im.width, im.height), (255, 255, 255))
    out.paste(photo, (0, 0))
    out.paste(im, (photo.width + pad, 0))
    return out
