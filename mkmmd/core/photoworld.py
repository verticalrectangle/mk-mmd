"""photo_world, bpy-free (docs/design.md: Sets): the scene's world made of ONE photo projected round it, so a camera of
any lens (a fisheye too) sees it bent as through that lens, it lights the cast and it shows in reflections.
mkmmd/blender/library/sets/photo_world.py builds it in world shader nodes that follow `photo_xy` and `uv` step by step.

The photo's frame (`frame`): f, the unit `toward` (the world direction of the photo's centre); u, the world's +Z made
perpendicular to f (+Y when f is vertical), the photo's up; r = f x u, its right; then r and u turned clockwise by `roll`
degrees about f, as a camera looking along f sees it (the convention of a shot's `roll`).

A world direction d makes the angle theta with f and the azimuth phi about it (from r toward u). The photo shows it at
the true angle alpha = theta * fov / spread, at

    (x, y) = tan(alpha) / tan(fov / 2) * (cos phi, sin phi)

in units of half the photo's long side from its centre (x right, y up). With spread = fov the photo is the picture a
pinhole camera took of the world; a larger spread stretches it angularly and uniformly about its centre, so a modest photo
fills a fisheye's 180 degrees and its straight lines still bend as through a lens. Where alpha reaches 90 degrees or
(x, y) falls outside the photo's rectangle the world is `fill`. Texture coordinates (s, t) are Blender's: (0, 0) the
photo's bottom-left corner, (1, 1) its top-right.

    parse(spec)                      the [[set]] table checked, defaults filled in (PhotoWorldError says what to change)
    is_file(ref), resolve(ref, root, assets)    `image`: a file (project-relative, absolute or ~) or a registry slug
    frame(toward, roll)              (f, r, u)
    photo_xy(d, cfg)                 (x, y, seen) of world directions d (n, 3)
    uv(d, cfg, size)                 (s, t, inside) on a photo of size (w, h) px
    direction(s, t, cfg, size)       the inverse: the world directions a photo's points show"""
import json
import math
import os
from pathlib import Path

import numpy as np

SPEC = {"image": None, "toward": [0.0, 1.0, 0.0], "roll": 0.0, "fov": None, "spread": None, "fill": "#080808",
        "strength": 1.0, "light": None, "saturation": 1.0}
BUILD_KEYS = {"name", "kind", "at", "yaw"}                 # keys of the build stage itself
IMAGE_EXT = (".jpg", ".jpeg", ".png", ".webp", ".exr", ".hdr")
EDGE = 1e-4                       # radians short of 90 degrees where tan(alpha) is cut (the fill shows from 90 on)


class PhotoWorldError(ValueError):
    """A bad key of a photo_world set, or an image that is missing or not an image: says what to change."""


def _number(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def parse(spec):
    """The spec with every default filled in: `spread` defaults to `fov`, `light` to `strength`, `toward` comes back
    as a unit vector. Unknown keys, missing `image` or `fov` and values out of range are errors."""
    bad = sorted(set(spec) - set(SPEC) - BUILD_KEYS)
    if bad:
        raise PhotoWorldError(f"photo_world: unknown keys {bad} (have {sorted(SPEC)})")
    cfg = {k: spec.get(k, v) for k, v in SPEC.items()}
    if cfg["image"] is None:
        raise PhotoWorldError("photo_world: give `image`, an image file or an asset slug")
    if not isinstance(cfg["image"], str) or not cfg["image"].strip():
        raise PhotoWorldError(f"photo_world: image is an image file or an asset slug, not {cfg['image']!r}")
    if cfg["fov"] is None:
        raise PhotoWorldError("photo_world: give `fov`, the degrees across the photo's long side (2 atan(18 / f) for a "
                              "35 mm-equivalent focal length f)")
    t = cfg["toward"]
    if not (isinstance(t, (list, tuple)) and len(t) == 3 and all(_number(v) for v in t)
            and math.hypot(*(float(v) for v in t)) > 1e-9):
        raise PhotoWorldError(f"photo_world: toward is a world direction [x, y, z], not {t!r}")
    cfg["toward"] = (np.asarray(t, float) / math.hypot(*(float(v) for v in t))).tolist()
    if not _number(cfg["roll"]):
        raise PhotoWorldError(f"photo_world: roll is degrees, not {cfg['roll']!r}")
    if not (_number(cfg["fov"]) and 0.0 < cfg["fov"] < 180.0):
        raise PhotoWorldError(f"photo_world: fov is the degrees across the photo's long side, 0 < fov < 180, not "
                              f"{cfg['fov']!r}")
    if cfg["spread"] is None:
        cfg["spread"] = cfg["fov"]
    if not (_number(cfg["spread"]) and 0.0 < cfg["spread"] <= 360.0):
        raise PhotoWorldError(f"photo_world: spread is the degrees the photo's long side spans in the world, "
                              f"0 < spread <= 360, not {cfg['spread']!r}")
    if not isinstance(cfg["fill"], str) or not cfg["fill"].strip():
        raise PhotoWorldError(f"photo_world: fill is a palette slot or '#hex', not {cfg['fill']!r}")
    if cfg["light"] is None:
        cfg["light"] = cfg["strength"]
    for k in ("strength", "light"):
        if not (_number(cfg[k]) and 0.0 <= cfg[k] <= 1000.0):
            raise PhotoWorldError(f"photo_world: {k} is 0 .. 1000, not {cfg[k]!r}")
    if not (_number(cfg["saturation"]) and 0.0 <= cfg["saturation"] <= 10.0):
        raise PhotoWorldError(f"photo_world: saturation is 0 .. 10 (1 the photo's own), not {cfg['saturation']!r}")
    return cfg


# ---------------------------------------------------------------------------------------------------- the image


def is_file(ref):
    """Whether `ref` names a file (a path separator, `~` or an image extension) rather than an asset-registry slug."""
    s = str(ref)
    return any(c in s for c in "/\\~") or Path(s).suffix.lower() in IMAGE_EXT


def resolve(ref, root=None, assets=None):
    """The path of the photo `ref`: a file (project-relative to `root`, absolute or ~/...) or the slug of an entry of the
    asset registry in `assets` (default ~/mk-assets) whose path is an image. A PhotoWorldError when there is none."""
    s = str(ref).strip()
    kinds = ", ".join(e.lstrip(".") for e in IMAGE_EXT)
    if is_file(s):
        p = Path(s).expanduser()
        if not p.is_absolute():
            p = Path(root or os.getcwd()) / p
        if p.suffix.lower() not in IMAGE_EXT:
            raise PhotoWorldError(f"photo_world: image {s!r} is not an image ({kinds})")
        if not p.is_file():
            raise PhotoWorldError(f"photo_world: image {s!r}: {p} does not exist")
        return p
    base = Path(assets or "~/mk-assets").expanduser()
    reg = base / "registry.json"
    entries = json.loads(reg.read_text(encoding="utf-8")) if reg.is_file() else []
    e = next((x for x in entries if x.get("slug") == s), None)
    if e is None:
        images = sorted(x["slug"] for x in entries if Path(x.get("path") or "").suffix.lower() in IMAGE_EXT)
        raise PhotoWorldError(f"photo_world: image {s!r} is not an image file and not an asset in {reg} (the images "
                              f"there: {', '.join(images) or 'none'}; `mk assets add FILE --kind texture` registers one)")
    p = Path(e.get("path") or "").expanduser()
    if not p.is_absolute():
        p = base / p
    if p.suffix.lower() not in IMAGE_EXT:
        raise PhotoWorldError(f"photo_world: image {s!r}: the asset's path {p} is not an image ({kinds})")
    if not p.is_file():
        raise PhotoWorldError(f"photo_world: image {s!r}: the asset's file {p} does not exist")
    return p


# ---------------------------------------------------------------------------------------------------- the mapping


def frame(toward, roll=0.0):
    """(f, r, u): the photo's axis, right and up as unit world vectors (see the module docstring)."""
    f = np.asarray(toward, float)
    f = f / np.linalg.norm(f)
    u = np.array([0.0, 0.0, 1.0]) - f[2] * f
    if np.linalg.norm(u) < 1e-9:                                  # looking straight up or down: +Y is the photo's up
        u = np.array([0.0, 1.0, 0.0]) - f[1] * f
    u = u / np.linalg.norm(u)
    r = np.cross(f, u)
    a = math.radians(float(roll))
    return f, r * math.cos(a) - u * math.sin(a), u * math.cos(a) + r * math.sin(a)


def photo_xy(d, cfg):
    """(x, y, seen) of world directions d (n, 3): where they fall on the photo, in units of half its long side from its
    centre (x right, y up), and whether their true angle alpha is under 90 degrees. Any length of d will do."""
    d = np.atleast_2d(np.asarray(d, float))
    f, r, u = frame(cfg["toward"], cfg["roll"])
    a, b, c = d @ r, d @ u, d @ f
    h = np.hypot(a, b)
    alpha = np.arctan2(h, c) * (float(cfg["fov"]) / float(cfg["spread"]))
    rho = np.tan(np.minimum(alpha, 0.5 * math.pi - EDGE)) / math.tan(math.radians(float(cfg["fov"])) / 2.0)
    g = rho / np.maximum(h, 1e-12)
    return a * g, b * g, alpha < 0.5 * math.pi


def uv(d, cfg, size):
    """(s, t, inside) of world directions d (n, 3) on a photo of `size` (w, h) px: Blender's texture coordinates, and
    whether the photo shows there (else `fill`)."""
    w, h = (float(v) for v in size)
    L = max(w, h)
    x, y, seen = photo_xy(d, cfg)
    s, t = 0.5 + 0.5 * x * L / w, 0.5 + 0.5 * y * L / h
    return s, t, seen & (np.abs(s - 0.5) < 0.5) & (np.abs(t - 0.5) < 0.5)


def direction(s, t, cfg, size):
    """The unit world directions (n, 3) where points (s, t) of a photo of `size` (w, h) px show: the inverse of `uv`."""
    w, h = (float(v) for v in size)
    L = max(w, h)
    x = (np.atleast_1d(np.asarray(s, float)) - 0.5) * 2.0 * w / L
    y = (np.atleast_1d(np.asarray(t, float)) - 0.5) * 2.0 * h / L
    alpha = np.arctan(np.hypot(x, y) * math.tan(math.radians(float(cfg["fov"])) / 2.0))
    theta = alpha * float(cfg["spread"]) / float(cfg["fov"])
    phi = np.arctan2(y, x)
    f, r, u = frame(cfg["toward"], cfg["roll"])
    side = np.cos(phi)[:, None] * r + np.sin(phi)[:, None] * u
    return np.cos(theta)[:, None] * f + np.sin(theta)[:, None] * side
