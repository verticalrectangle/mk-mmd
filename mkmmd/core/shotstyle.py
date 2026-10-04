"""Shot styles and lens shift, bpy-free (docs/design.md: Shots): the arithmetic and the spec handling behind the
`[[shot]]` keys `shift`, `style`, `colors`, `hide`, `keep`, `accent`, `tint`, `knockout` and `reflection`. The Blender
half lives in `mkmmd/blender/styles.py` (the render-time switching) and `mkmmd/blender/build/shots.py` (cameras).

Lens shift. Blender's `shift_x` / `shift_y` are fractions of the LARGER image side and move the picture the other way:
a positive `shift_y` lowers the optical axis in the frame by `shift_y * max(w, h)` pixels (probed in Blender 4.2:
shift_y = +0.25 puts the axis point at v = 0.25 of a square frame, v = 0 at the bottom). A square crop of a tall master
(`crop_shift`) therefore needs a negative shift when the crop band lies below the master's centre.

Styles. A shot spec (aspect overrides already merged in) is normalised by `normalize` into the plain dict stored in the
scene's shot table, with every colour resolved to display-space sRGB floats, so the render-time code needs neither the
project nor its palette."""
import fnmatch
from typing import NamedTuple

from . import palette as PAL


class StyleError(ValueError):
    """A bad style / reflection / shift key in a [[shot]]."""


STYLES = ("silhouette",)
REFLECTION_KEYS = {"object", "strength", "dim", "roughness", "tint", "probe", "hide", "only", "bend", "world"}
COLOR_ROLES = ("background", "subject", "accent")
DEFAULT_COLORS = {"background": "base", "subject": "text", "accent": "surface"}


# ================================================================================================= lens shift
def shift_pair(value, what="shift"):
    """[x, y] -> (float, float); anything else is an error (`what` names the key in the message)."""
    try:
        x, y = value
        return float(x), float(y)
    except (TypeError, ValueError):
        raise StyleError(f"{what} = {value!r}: expected [x, y] (fractions of the larger image side)") from None


def crop_shift(master, crop, top=0.0, left=None):
    """Lens shift (x, y) that turns a camera rendered at `crop` = (w, h) px into an exact crop of the camera rendered at
    `master` = (w, h) px, the crop's top-left corner lying `top` px below the master's top edge and `left` px right of
    its left edge (default: centred, so 0 for a full-width crop). Both cameras use sensor fit AUTO on a 36 mm sensor and
    the same orientation and position; `crop_camera` gives the lens and f-stop factors that go with it."""
    (mw, mh), (cw, ch) = master, crop
    left = (mw - cw) / 2.0 if left is None else float(left)
    big = float(max(cw, ch))
    axis_x = mw / 2.0 - left                 # the master's optical axis, in crop pixels (x right, y down)
    axis_y = mh / 2.0 - float(top)
    # the axis lies (axis - centre) px from the crop's centre and Blender moves it opposite to x, with y
    return (cw / 2.0 - axis_x) / big, (axis_y - ch / 2.0) / big


def crop_camera(master, crop, top=0.0, left=None):
    """What a crop of the master needs besides the pose: {lens_scale, fstop_scale, shift}. A pixel keeps its angular size
    when the lens scales by k = max(master) / max(crop) (the 36 mm sensor spans fewer pixels). A defocus blur of f^2 / N
    millimetres on the sensor is f^2 / N * max_side / 36 pixels; with f scaled by k and max_side by 1 / k that stays the
    master's blur in pixels only if the f-stop N scales by k as well."""
    k = max(master) / float(max(crop))
    return {"lens_scale": k, "fstop_scale": k, "shift": crop_shift(master, crop, top, left)}


def pinhole_px(point, lens, size, shift=(0.0, 0.0), sensor=36.0):
    """Pixel (x right, y down, from the top-left corner) where `point` = (x right, y up, z) in camera space (the camera
    looks down -z) lands for a Blender camera with sensor fit AUTO (the sensor spans the larger image side), `lens` mm
    and lens shift: a positive shift_x moves the picture left, a positive shift_y down (probed in Blender 4.2.3: the
    point on the optical axis lands at x = w / 2 - shift_x * max(w, h), y = h / 2 + shift_y * max(w, h))."""
    w, h = size
    big = float(max(w, h))
    f_px = lens / sensor * big
    x, y, z = point
    d = -z
    return (w / 2.0 + f_px * x / d - shift[0] * big, h / 2.0 - f_px * y / d + shift[1] * big)


# ================================================================================================= colours
def srgb_to_linear(c):
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def linear_to_srgb(c):
    return c * 12.92 if c <= 0.0031308 else 1.055 * c ** (1 / 2.4) - 0.055


def resolve_colour(spec, palette):
    """A palette slot (`foam`), a `#hex`, a mix of two such (`iris:surface:0.3`, the second weighs t) or three floats
    in 0..1 -> display-space sRGB (r, g, b) floats."""
    if isinstance(spec, (list, tuple)):
        if len(spec) != 3 or not all(isinstance(v, (int, float)) and 0.0 <= v <= 1.0 for v in spec):
            raise StyleError(f"colour {spec!r}: expected [r, g, b] in 0..1")
        return tuple(float(v) for v in spec)
    if not isinstance(spec, str):
        raise StyleError(f"colour {spec!r}: expected a palette slot, '#hex', 'a:b:t' or [r, g, b]")
    parts = spec.split(":")
    try:
        if len(parts) == 1:
            return PAL.srgb(PAL.resolve(parts[0], palette))
        if len(parts) == 3:
            t = float(parts[2])
            if not 0.0 <= t <= 1.0:
                raise StyleError(f"colour {spec!r}: the mix weight must be in 0..1")
            return PAL.srgb(PAL.mix(PAL.resolve(parts[0], palette), PAL.resolve(parts[1], palette), t))
    except KeyError as e:
        raise StyleError(f"colour {spec!r}: {e.args[0]}") from None
    except ValueError:
        raise StyleError(f"colour {spec!r}: bad mix weight") from None
    raise StyleError(f"colour {spec!r}: expected a slot, '#hex' or 'a:b:t'")


def _rgb(c):
    return [round(float(v), 6) for v in c]


# ================================================================================================= image helpers
def dilate(a, r):
    """Grey dilation of a 2D array by a (2r+1) square: the cord that reads like a bold wire."""
    if r <= 0:
        return a
    import numpy as np
    out = a.copy()
    for k in range(1, r + 1):
        out[:, k:] = np.maximum(out[:, k:], a[:, :-k])
        out[:, :-k] = np.maximum(out[:, :-k], a[:, k:])
    res = out.copy()
    for k in range(1, r + 1):
        res[k:] = np.maximum(res[k:], out[:-k])
        res[:-k] = np.maximum(res[:-k], out[k:])
    return res


def png_bytes(rgb8):
    """An 8-bit RGB PNG from an (h, w, 3) uint8 array (top row first), Sub-filtered: what Blender's image writer would need
    a whole image datablock for."""
    import struct
    import zlib
    import numpy as np
    h, w, _ = rgb8.shape
    line = np.ascontiguousarray(rgb8).reshape(h, w * 3)
    raw = np.empty((h, w * 3 + 1), np.uint8)
    raw[:, 0] = 1
    raw[:, 1:4] = line[:, :3]
    raw[:, 4:] = line[:, 3:] - line[:, :-3]

    def chunk(tag, data):
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw.tobytes(), 3)) + chunk(b"IEND", b""))


# ================================================================================================= selecting objects
class Obj(NamedTuple):
    """What a pattern can match: the object's name, the names of the collections it is in (ancestors included) and the
    names of its truthy custom properties."""
    name: str
    collections: tuple = ()
    props: frozenset = frozenset()


def matches(pattern, ob):
    """`name*` (fnmatch on the object name), `@collection*` (any collection the object is in), `prop:key` (the object
    has that custom property set)."""
    if pattern.startswith("@"):
        return any(fnmatch.fnmatchcase(c, pattern[1:]) for c in ob.collections)
    if pattern.startswith("prop:"):
        return pattern[5:] in ob.props
    return fnmatch.fnmatchcase(ob.name, pattern)


def select(objects, patterns):
    """Names of the objects that match any pattern, in the objects' order."""
    patterns = list(patterns or [])
    return [ob.name for ob in objects if any(matches(p, ob) for p in patterns)]


def _patterns(spec, key):
    v = spec.get(key, [])
    if isinstance(v, str):
        v = [v]
    if not isinstance(v, (list, tuple)) or not all(isinstance(p, str) for p in v):
        raise StyleError(f"{key} = {v!r}: expected a list of object name patterns ('name*', '@collection', 'prop:key')")
    return list(v)


# ================================================================================================= normalising specs
def _tint(layer, palette):
    """One background layer: the background moves toward `color` by min(1, gain * <custom property>), everywhere or as a
    soft elliptical glow (`glow = {at = [x, y], size = [sx, sy]}`, fractions of the frame's width and height from the
    top-left, gaussian exp(-(dx / sx)^2 - (dy / sy)^2))."""
    if not isinstance(layer, dict):
        raise StyleError(f"tint entry {layer!r}: expected {{object, prop, color, gain, glow}}")
    unknown = sorted(set(layer) - {"object", "prop", "color", "gain", "glow"})
    if unknown:
        raise StyleError(f"tint entry: unknown keys {unknown}")
    for k in ("object", "prop", "color"):
        if k not in layer:
            raise StyleError(f"tint entry {layer!r}: needs {k}")
    out = {"object": str(layer["object"]), "prop": str(layer["prop"]),
           "color": _rgb(resolve_colour(layer["color"], palette)), "gain": float(layer.get("gain", 1.0)), "glow": None}
    g = layer.get("glow")
    if g is not None:
        if not isinstance(g, dict) or set(g) - {"at", "size"}:
            raise StyleError(f"tint glow {g!r}: expected {{at = [x, y], size = [sx, sy]}}")
        at, size = shift_pair(g.get("at", [0.5, 0.5]), "glow.at"), shift_pair(g.get("size", [1.0, 1.0]), "glow.size")
        if min(size) <= 0:
            raise StyleError("tint glow size must be positive")
        out["glow"] = {"at": list(at), "size": list(size)}
    return out


def normalize_silhouette(spec, palette):
    """The silhouette look of a shot: flat colours instead of the lit scene (docs/design.md: Shots)."""
    colors = spec.get("colors") or {}
    if not isinstance(colors, dict) or set(colors) - set(COLOR_ROLES):
        raise StyleError(f"colors = {colors!r}: expected a table with {', '.join(COLOR_ROLES)}")
    cols = {role: _rgb(resolve_colour(colors.get(role, DEFAULT_COLORS[role]), palette)) for role in COLOR_ROLES}
    samples = int(spec.get("samples", 16))
    if samples < 1:
        raise StyleError("samples must be at least 1")
    ko = spec.get("knockout")
    if ko is not None:
        if not isinstance(ko, dict) or set(ko) - {"objects", "color"} or "objects" not in ko:
            raise StyleError(f"knockout = {ko!r}: expected {{objects = [patterns], color = slot}}")
        ko = {"objects": _patterns(ko, "objects"), "color": _rgb(resolve_colour(ko.get("color", "text"), palette))}
    tint = spec.get("tint", [])
    if isinstance(tint, dict):
        tint = [tint]
    grow = float(spec.get("grow", 1.0))
    if grow < 0:
        raise StyleError("grow must not be negative")
    return {"colors": cols, "hide": _patterns(spec, "hide"), "keep": _patterns(spec, "keep"),
            "accent": _patterns(spec, "accent"), "tint": [_tint(t, palette) for t in tint], "knockout": ko,
            "samples": samples, "grow": grow}


def normalize_reflection(spec, palette):
    """`reflection = {object = "<glass>", strength = 0.5, dim = 0.0, roughness = 0.0, hide = [patterns], only = [patterns],
    bend = false, world = false}`: a mirror layer on a glass object, fed by an EEVEE plane probe (docs/design.md: Shots).
    `hide` / `only` choose the objects the pane reflects (patterns as `hide` of a silhouette), `bend` lets the rain on the
    glass bend the reflection, `world` keeps the sky in it (by default the pane reflects objects on black, so the street
    behind the glass shows through)."""
    if not isinstance(spec, dict) or "object" not in spec:
        raise StyleError(f"reflection = {spec!r}: expected {{object = '<glass object>', strength, dim, roughness}}")
    unknown = sorted(set(spec) - REFLECTION_KEYS)
    if unknown:
        raise StyleError(f"reflection: unknown keys {unknown} (known: {sorted(REFLECTION_KEYS)})")
    out = {"object": str(spec["object"]), "strength": float(spec.get("strength", 0.5)), "dim": float(spec.get("dim", 0.0)),
           "roughness": float(spec.get("roughness", 0.0)), "tint": None, "probe": dict(spec.get("probe") or {}),
           "hide": _patterns(spec, "hide"), "only": _patterns(spec, "only"), "bend": bool(spec.get("bend", False)),
           "world": bool(spec.get("world", False))}
    for k in ("strength", "dim"):
        if not 0.0 <= out[k] <= 1.0:
            raise StyleError(f"reflection {k} = {out[k]}: expected 0..1")
    if not 0.0 <= out["roughness"] <= 1.0:
        raise StyleError(f"reflection roughness = {out['roughness']}: expected 0..1")
    if spec.get("tint") is not None:
        out["tint"] = _rgb(resolve_colour(spec["tint"], palette))
    return out


def normalize(spec, palette):
    """{'silhouette': {...}} and / or {'reflection': {...}} for one shot spec (an aspect's overrides merged in); {} when
    the shot has neither. `style = "none"` switches an inherited style off for one aspect."""
    style = spec.get("style")
    out = {}
    if style not in (None, "none"):
        if style not in STYLES:
            raise StyleError(f"shot {spec.get('name')!r}: style = {style!r} (known: {', '.join(STYLES)})")
        out["silhouette"] = normalize_silhouette(spec, palette)
    if spec.get("reflection") is not None:
        if out:
            raise StyleError(f"shot {spec.get('name')!r}: a {style} shot cannot have a reflection")
        out["reflection"] = normalize_reflection(spec["reflection"], palette)
    return out


# ================================================================================================= the cut
def shot_at(table, frame):
    """The shot-table entry in force at `frame`: the last one that starts at or before it (a cut belongs to the incoming
    shot, like the timeline markers), the first one before the first cut. None for an empty table. Plate shots (entries
    with `plate`: rendered only where a transition or insert needs them) are not in the cut."""
    table = [e for e in table or [] if not e.get("plate")]
    if not table:
        return None
    best = None
    for e in table:
        if e["from"] <= frame and (best is None or e["from"] >= best["from"]):
            best = e
    return best or min(table, key=lambda e: e["from"])


# ================================================================================================= the background
def glow_weight(x, y, at, size):
    """Gaussian glow at frame fractions (x, y) (from the top-left), centred on `at`, reaching `size`."""
    import numpy as np
    return np.exp(-(((np.asarray(x) - at[0]) / size[0]) ** 2 + ((np.asarray(y) - at[1]) / size[1]) ** 2))


def tint_weight(layer, value):
    """How far the background moves toward the layer's colour at the custom property's current value."""
    return min(1.0, max(0.0, layer["gain"] * float(value)))


def background_rgb(base, layers, values):
    """The flat part of the background: `base` moved toward each layer's colour (those without a glow) by its weight;
    `values` holds one custom property value per layer. Glow layers are per-pixel (`background_image`)."""
    c = [float(v) for v in base]
    for layer, v in zip(layers, values):
        if layer["glow"] is None:
            w = tint_weight(layer, v)
            c = [a + (b - a) * w for a, b in zip(c, layer["color"])]
    return c


def background_image(size, base, layers, values):
    """The whole background (h, w, 3) float32 in display space, glow layers included."""
    import numpy as np
    w, h = size
    img = np.empty((h, w, 3), np.float32)
    img[:] = background_rgb(base, layers, values)
    glows = [(l, v) for l, v in zip(layers, values) if l["glow"] is not None and tint_weight(l, v) > 0]
    if glows:
        xs = (np.arange(w, dtype=np.float32) + 0.5) / w
        ys = (np.arange(h, dtype=np.float32) + 0.5) / h
        for layer, v in glows:
            k = glow_weight(xs[None, :], ys[:, None], layer["glow"]["at"], layer["glow"]["size"]) * tint_weight(layer, v)
            img += (np.asarray(layer["color"], np.float32) - img) * k[..., None].astype(np.float32)
    return img


# ================================================================================================= composing the silhouette
ALPHA_FULL = 254.0 / 255.0               # Workbench's film alpha over a fully covered pixel (measured: 254 of 255)
SHARE_FLOOR = 0.04                       # below this share of the accent colour a flat pixel counts as plain subject


def compose_silhouette(p, spec, tint_values, hard=False, subject=True):
    """The finished silhouette frame, float32 RGB (h, w, 3) in display space, from the passes `p` (numpy arrays):
      flat        (h, w, 4) straight RGBA of the subject (and hard accents) in flat colours, coverage in alpha
      soft_alpha  (h, w) alpha of the soft accent objects (a bolt), hidden behind the subject; soft_aov (h, w) their weight
      type        (h, w, 4) straight RGBA of the type; knock (h, w, 4) coverage of the knock-out type in alpha
    `spec` is the normalised silhouette spec, `tint_values` one custom property value per background layer, `hard` says
    whether flat holds accent-coloured objects (cords) to grow. The order is the original's post: background, subject,
    the bolt's weight in the accent colour and what its alpha has beyond that as a glow in the subject colour
    (`sil = alpha - bolt`), the cords, then type, then knock-out type (ink on the background, background colour where
    it overlaps the silhouette). `subject = False` leaves the subject out: the frame as it would be without the figure
    (what a transition shows round a figure that has turned into a window, docs/design.md: Transitions and inserts)."""
    import numpy as np
    cols = spec["colors"]
    subj, acc = np.asarray(cols["subject"], np.float32), np.asarray(cols["accent"], np.float32)
    F = p["flat"]
    h, w = F.shape[:2]
    s = np.clip(F[..., 3] / ALPHA_FULL, 0.0, 1.0)
    sv = s if subject else np.zeros_like(s)                      # the coverage painted as the subject
    bg = background_image((w, h), cols["background"], spec["tint"], tint_values)
    img = bg * (1.0 - sv[..., None]) + F[..., :3] * sv[..., None]
    g = np.zeros((h, w), np.float32)
    if "soft_alpha" in p:
        g = np.minimum(np.clip(p["soft_alpha"] - p["soft_aov"], 0.0, 1.0), 1.0 - sv)
        img = img + (subj - bg) * g[..., None]
        img = img + (acc - img) * p["soft_aov"][..., None]
    if hard:
        d = acc - subj
        dd = float((d * d).sum())
        if dd > 1e-6:
            share = np.clip(((F[..., :3] - subj) @ d) / dd, 0.0, 1.0)
            share = np.clip((share - SHARE_FLOOR) / (1.0 - SHARE_FLOOR), 0.0, 1.0)     # Workbench's colour is off by 1/255
            a = dilate(s * share, int(round(spec["grow"] * w / 1080.0)))
            img = img + (acc - img) * a[..., None]
    if "type" in p:
        t = p["type"]
        a = np.clip(t[..., 3:4], 0.0, 1.0)
        img = img * (1.0 - a) + t[..., :3] * a
    if "knock" in p:
        k = np.clip(p["knock"][..., 3:4], 0.0, 1.0)
        ink = np.asarray(spec["knockout"]["color"], np.float32)
        sil = np.clip(sv + g, 0.0, 1.0)[..., None]
        flat_bg = np.asarray(cols["background"], np.float32)
        img = img * (1.0 - k) + (flat_bg * sil + ink * (1.0 - sil)) * k
    return img.astype(np.float32)
