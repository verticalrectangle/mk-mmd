"""Shot styles and lens shift, bpy-free (docs/design.md: Shots): the arithmetic and the spec handling behind the
`[[shot]]` keys `shift`, `style`, `colors`, `tones`, `hide`, `keep`, `accent`, `tint`, `knockout` and `reflection`, and
the project's `[vector]` table. The Blender half lives in `mkmmd/blender/styles.py` (the render-time switching) and
`mkmmd/blender/build/shots.py` (cameras).

Lens shift. Blender's `shift_x` / `shift_y` are fractions of the LARGER image side and move the picture the other way:
a positive `shift_y` lowers the optical axis in the frame by `shift_y * max(w, h)` pixels (probed in Blender 4.2:
shift_y = +0.25 puts the axis point at v = 0.25 of a square frame, v = 0 at the bottom). A square crop of a tall master
(`crop_shift`) therefore needs a negative shift when the crop band lies below the master's centre.

Styles. A shot spec (aspect overrides already merged in) is normalised by `normalize` into the plain dict stored in the
scene's shot table, with every colour resolved to display-space sRGB floats, so the render-time code needs neither the
project nor its palette."""
import fnmatch
import re
from typing import NamedTuple

from . import palette as PAL


class StyleError(ValueError):
    """A bad style / reflection / shift key in a [[shot]]."""


STYLES = ("silhouette", "vector")
REFLECTION_KEYS = {"object", "strength", "dim", "roughness", "tint", "probe", "hide", "only", "bend", "world"}
COLOR_ROLES = ("background", "subject", "accent")
DEFAULT_COLORS = {"background": "base", "subject": "text", "accent": "surface"}
SILHOUETTE_ONLY = ("accent", "tint", "knockout", "grow", "samples")     # shot keys no other style reads
VECTOR_KEYS = ("colors", "tones", "materials", "lines", "shadow", "light", "opposite")
VECTOR_ROLES = ("background", "line", "inner")
VECTOR_COLORS = {"background": "base", "line": "text"}                  # `inner` defaults to `line`
VECTOR_TONES = {"fill": {"lit": "text", "shade": "subtle"}}             # a project without [vector] tones
VECTOR_LINES = {"outline": 5.0, "inner": 2.0}                           # line widths, px at 1080 on the frame's short side
VECTOR_LIGHT = (0.45, -0.35, 0.82)                                      # Workbench's shadow direction (toward the light)
_SUFFIX = re.compile(r"\.\d{3}$")                                       # Blender's name-clash suffix: `face.001`


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


def _tone(name, t, palette):
    """One tone of the vector look: a colour (flat), `{lit, shade}` (lit and in shadow under the look's light; no `shade`:
    flat) or `{colors, at, grey}` (drawn: the texture's brightness picks one of the colours, so what the model draws on
    itself, eyes and mouth, carries over)."""
    col = lambda c: _rgb(resolve_colour(c, palette))
    if not isinstance(t, dict):
        return {"kind": "flat", "lit": col(t)}
    drawn = "colors" in t
    unknown = sorted(set(t) - ({"colors", "at", "grey"} if drawn else {"lit", "shade"}))
    if unknown:
        raise StyleError(f"vector tone {name!r}: unknown keys {unknown} (a tone is a colour, {{lit, shade}} or "
                         f"{{colors, at, grey}})")
    if drawn:
        cols, at = t["colors"], t.get("at", [])
        if not isinstance(cols, list) or len(cols) < 2 or not isinstance(at, list) or len(at) != len(cols) - 1:
            raise StyleError(f"vector tone {name!r}: colors = [n colours] and at = [n - 1 brightness steps]")
        at = [float(a) for a in at]
        if any(not 0.0 < a < 1.0 for a in at) or at != sorted(at):
            raise StyleError(f"vector tone {name!r}: at = {at}: increasing brightness steps inside 0..1")
        grey = float(t.get("grey", 1.0))
        if not 0.0 < grey <= 1.0:
            raise StyleError(f"vector tone {name!r}: grey = {grey}: a saturation in 0..1")
        return {"kind": "drawn", "colors": [col(c) for c in cols], "at": at, "grey": grey}
    if "lit" not in t:
        raise StyleError(f"vector tone {name!r}: needs lit (or colors)")
    if "shade" not in t:
        return {"kind": "flat", "lit": col(t["lit"])}
    return {"kind": "lit", "lit": col(t["lit"]), "shade": col(t["shade"])}


def _material_rule(i, rule, tones):
    what = f"[vector] materials[{i}]"
    if not isinstance(rule, dict) or set(rule) - {"match", "tone", "group"} or not {"match", "tone"} <= set(rule):
        raise StyleError(f"{what} = {rule!r}: expected {{match = [material name patterns], tone, group}}")
    match = [rule["match"]] if isinstance(rule["match"], str) else rule["match"]
    if not isinstance(match, list) or not match or not all(isinstance(p, str) for p in match):
        raise StyleError(f"{what}: match = {rule['match']!r}: expected material name patterns")
    if rule["tone"] not in tones:
        raise StyleError(f"{what}: tone {rule['tone']!r} is not one of the tones ({', '.join(tones)})")
    group = rule.get("group")
    return {"match": list(match), "tone": str(rule["tone"]), "group": None if group is None else str(group)}


def normalize_vector(spec, palette, project=None):
    """The flat-vector look of a shot: the project's `[vector]` table (`project`) with the shot's own `colors`, `tones`,
    `hide` and `keep` laid over it (docs/design.md: Looks: vector)."""
    project = {} if project is None else project
    if not isinstance(project, dict):
        raise StyleError("[vector] must be a table")
    unknown = sorted(set(project) - set(VECTOR_KEYS))
    if unknown:
        raise StyleError(f"[vector]: unknown keys {unknown} (known: {', '.join(VECTOR_KEYS)})")
    other = [k for k in SILHOUETTE_ONLY if k in spec]
    if other:
        raise StyleError(f"{', '.join(other)}: silhouette keys; a vector shot does not read them")
    colors = dict(VECTOR_COLORS)
    for where, c in (("[vector] colors", project.get("colors")), ("colors", spec.get("colors"))):
        if c is None:
            continue
        if not isinstance(c, dict) or set(c) - set(VECTOR_ROLES):
            raise StyleError(f"{where} = {c!r}: expected a table with {', '.join(VECTOR_ROLES)}")
        colors.update(c)
    colors.setdefault("inner", colors["line"])
    given_tones = (("[vector] tones", project.get("tones") or VECTOR_TONES), ("tones", spec.get("tones") or {}))
    for where, t in given_tones:
        if not isinstance(t, dict):
            raise StyleError(f"{where} = {t!r}: expected a table of tones")
    tones = {str(n): _tone(n, t, palette) for n, t in {**given_tones[0][1], **given_tones[1][1]}.items()}
    rules = project.get("materials") or []
    if not isinstance(rules, list):
        raise StyleError("[vector] materials: expected a list of {match, tone, group}")
    lines = dict(VECTOR_LINES)
    given = project.get("lines") or {}
    if not isinstance(given, dict) or set(given) - set(VECTOR_LINES):
        raise StyleError(f"[vector] lines = {given!r}: expected {{outline, inner}} (px at 1080)")
    for k, v in given.items():
        lines[k] = float(v)
        if lines[k] < 0:
            raise StyleError(f"[vector] lines.{k} must not be negative")
    shadow = float(project.get("shadow", 0.3))
    if not 0.0 <= shadow <= 1.0:
        raise StyleError(f"[vector] shadow = {shadow}: a share of the light in 0..1")
    light = project.get("light", list(VECTOR_LIGHT))
    try:
        light = [float(v) for v in light]
    except (TypeError, ValueError):
        light = []
    if len(light) != 3 or not any(light):
        raise StyleError(f"[vector] light = {project.get('light')!r}: expected [x, y, z] toward the light")
    cols = {k: _rgb(resolve_colour(colors[k], palette)) for k in VECTOR_ROLES}
    return {"colors": cols, "tones": tones, "opposite": _opposite(project.get("opposite"), colors, tones, palette),
            "materials": [_material_rule(i, r, tones) for i, r in enumerate(rules)],
            "default": "fill" if "fill" in tones else next(iter(tones)), "lines": lines, "shadow": shadow,
            "light": light, "hide": _patterns(spec, "hide"), "keep": _patterns(spec, "keep")}


def _opposite(opp, colors, tones, palette):
    """`[vector] opposite = {colors, tones}`: the palette inside a ring, laid over the look's own (`colors` as given,
    `tones` normalised); None without one. A tone keeps its kind: a drawn tone gives only its `colors`, the same count."""
    if opp is None:
        return None
    if not isinstance(opp, dict) or set(opp) - {"colors", "tones"}:
        raise StyleError(f"[vector] opposite = {opp!r}: expected {{colors, tones}}")
    oc = opp.get("colors") or {}
    if not isinstance(oc, dict) or set(oc) - set(VECTOR_ROLES):
        raise StyleError(f"[vector] opposite colors = {oc!r}: expected a table with {', '.join(VECTOR_ROLES)}")
    ot = opp.get("tones") or {}
    if not isinstance(ot, dict):
        raise StyleError(f"[vector] opposite tones = {ot!r}: expected a table of tones")
    out = dict(tones)
    for name, t in ot.items():
        if name not in tones:
            raise StyleError(f"[vector] opposite tone {name!r} is not one of the tones ({', '.join(tones)})")
        base = tones[name]
        kept = f"[vector] opposite tone {name!r}: a {base['kind']} tone keeps its kind (and its colours' count)"
        if base["kind"] == "drawn" and isinstance(t, dict) and set(t) == {"colors"}:
            if not isinstance(t["colors"], list) or len(t["colors"]) != len(base["colors"]):
                raise StyleError(kept)
            t = {"colors": t["colors"], "at": base["at"], "grey": base["grey"]}
        new = _tone(name, t, palette)
        if new["kind"] != base["kind"] or len(new.get("colors", ())) != len(base.get("colors", ())):
            raise StyleError(kept)
        out[name] = new
    return {"colors": {k: _rgb(resolve_colour({**colors, **oc}[k], palette)) for k in VECTOR_ROLES}, "tones": out}


def normalize(spec, palette, vector=None):
    """{'silhouette' | 'vector': {...}} and / or {'reflection': {...}} for one shot spec (an aspect's overrides merged
    in); {} when the shot has none. `vector` is the project's `[vector]` table. `style = "none"` and `reflection = false`
    switch an inherited look off for one aspect."""
    style = spec.get("style")
    reflection = spec.get("reflection")
    out = {}
    if style not in (None, "none"):
        if style not in STYLES:
            raise StyleError(f"shot {spec.get('name')!r}: style = {style!r} (known: {', '.join(STYLES)})")
        if style == "vector":
            out["vector"] = normalize_vector(spec, palette, vector)
        else:
            if "tones" in spec:
                raise StyleError(f"shot {spec.get('name')!r}: tones belong to the vector style")
            out["silhouette"] = normalize_silhouette(spec, palette)
    elif "tones" in spec:
        raise StyleError(f"shot {spec.get('name')!r}: tones belong to the vector style")
    if reflection is not None and reflection is not False:
        if out:
            raise StyleError(f"shot {spec.get('name')!r}: a {style} shot cannot have a reflection")
        out["reflection"] = normalize_reflection(reflection, palette)
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
    whether flat holds accent-coloured objects (cords) to grow. The layers go in this order: background, subject,
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


# ================================================================================================= composing the vector look
VECTOR_SCALE = 2          # the vector passes are rendered at this many times the frame's size and boxed down
LIGHT_FULL = 0.40         # Workbench's default studio light (Raw) on a white surface facing it: what `shadow` is a share of


def vector_name(name):
    """A material's name as the vector look's rules match it: Blender's `.001` clash suffix dropped."""
    return _SUFFIX.sub("", name)


def vector_table(materials, spec):
    """(tones, groups) per material id for `compose_vector`: id i + 1 is `materials[i]` (Blender material names, matched
    by `vector_name`), id 0 is the background (tone None, group 0). A material takes the first rule whose patterns match
    it, else the look's default tone; its line group is the rule's `group`, else its own name: lines are drawn where two
    groups meet."""
    tones, groups, ids = [None], [0], {}
    for name in materials:
        base = vector_name(name)
        rule = next((r for r in spec["materials"] if any(fnmatch.fnmatchcase(base, p) for p in r["match"])), None)
        tones.append(rule["tone"] if rule else spec["default"])
        group = rule["group"] if rule and rule["group"] else "material:" + base
        groups.append(ids.setdefault(group, len(ids) + 1))
    return tones, groups


def dilate_round(m, r):
    """Binary dilation of the mask `m` by a disc of radius `r` pixels, an octagon: a square of 0.41 r, then a diamond of
    the rest, so a line keeps its width in every direction."""
    import numpy as np
    if r < 0.5:
        return m.copy()
    a = int(round(0.414 * r))
    out = dilate(m.astype(bool), a)
    for _ in range(int(round(r)) - a):
        n = out.copy()
        n[1:] |= out[:-1]
        n[:-1] |= out[1:]
        n[:, 1:] |= out[:, :-1]
        n[:, :-1] |= out[:, 1:]
        out = n
    return np.asarray(out, bool)


def edges(labels):
    """Pixels whose right or lower neighbour has another label."""
    import numpy as np
    e = np.zeros(labels.shape, bool)
    e[:, :-1] |= labels[:, :-1] != labels[:, 1:]
    e[:-1] |= labels[:-1] != labels[1:]
    return e


def depth_breaks(z, cover, step=0.01, rel=0.004):
    """Where one surface passes in front of another: the depth's second difference along x or y beyond `step` metres plus
    `rel` of the distance, between three covered pixels. A surface turning away from the camera bends the depth smoothly
    and draws no line; a step does."""
    import numpy as np
    zz = np.where(cover, z, 0.0).astype(np.float64)
    e = np.zeros(z.shape, bool)
    ok = cover[:, :-2] & cover[:, 1:-1] & cover[:, 2:]
    d2 = np.abs(zz[:, :-2] - 2.0 * zz[:, 1:-1] + zz[:, 2:])
    e[:, 1:-1] |= ok & (d2 > step + rel * zz[:, 1:-1])
    ok = cover[:-2] & cover[1:-1] & cover[2:]
    d2 = np.abs(zz[:-2] - 2.0 * zz[1:-1] + zz[2:])
    e[1:-1] |= ok & (d2 > step + rel * zz[1:-1])
    return e


def _vector_slots(p, spec, table, scale, subject):
    """(slots, keys): every pass pixel's colour as an index into `keys`, the colour roles of the look: "background",
    "line", "inner" and (tone, "lit" | "shade") or (tone, "band", i). A palette (`_vector_palette`) turns them into colours,
    so the drawing is worked out once and painted in as many palettes as the frame needs."""
    import numpy as np
    tones_of, groups_of = table
    mid = p["id"]
    H, W = mid.shape
    keys = ["background", "line", "inner"]
    slots = np.zeros((H, W), np.int16)
    if not subject:
        return slots, keys
    cover = mid > 0
    names = list(spec["tones"])
    T = np.asarray([-1] + [names.index(t) for t in tones_of[1:]], np.int32)[mid]
    r = max(1, int(round(scale)))
    lit = p["shade"] >= spec["shadow"] * LIGHT_FULL
    lit = dilate_round(~dilate_round(~lit, r), r)                            # opened: no lit slivers
    lit = ~dilate_round(~dilate_round(lit, r), r)                            # closed: no shadow pinholes
    tex = p.get("tex")
    for k, name in enumerate(names):
        m = cover & (T == k)
        if not m.any():
            continue
        t = spec["tones"][name]
        if t["kind"] == "flat":
            slots[m] = len(keys)
            keys.append((name, "lit"))
        elif t["kind"] == "lit":
            slots[m & lit], slots[m & ~lit] = len(keys), len(keys) + 1
            keys += [(name, "lit"), (name, "shade")]
        else:
            if tex is None:
                raise ValueError(f"vector tone {name!r} is drawn from the textures: the passes need `tex`")
            c = tex[m][:, :3]
            band = np.searchsorted(np.asarray(t["at"]), c @ np.asarray([0.299, 0.587, 0.114]), side="right")
            hi, lo = c.max(1), c.min(1)
            last = len(t["colors"]) - 1
            band = np.where((band == last) & ((hi - lo) >= t["grey"] * np.maximum(hi, 1e-6)), last - 1, band)
            slots[m] = len(keys) + band
            keys += [(name, "band", i) for i in range(last + 1)]
    px = min(H, W) / 1080.0                                                # pass pixels per px at 1080 on the short side
    groups = np.asarray(groups_of, np.int32)[mid]
    inner = (edges(groups) | depth_breaks(p["depth"], cover)) & cover
    slots[dilate_round(inner, spec["lines"]["inner"] * px / 2.0) & dilate_round(cover, r)] = 2
    slots[dilate_round(edges(cover), spec["lines"]["outline"] * px / 2.0)] = 1
    return slots, keys


def _vector_palette(look, keys):
    """The colours of `keys` in one palette: `look` holds `colors` and `tones` (the look's own, or its `opposite`)."""
    import numpy as np
    out = []
    for k in keys:
        if isinstance(k, str):
            out.append(look["colors"][k])
        else:
            t = look["tones"][k[0]]
            out.append(t["colors"][k[2]] if k[1] == "band" else t["shade"] if k[1] == "shade" else t["lit"])
    return np.asarray(out, np.float32)


def compose_vector(p, spec, table, scale=VECTOR_SCALE, subject=True, flip=None, edges_on=()):
    """The finished vector frame, float32 RGB (h, w, 3) in display space, from the passes `p` (numpy arrays at `scale`
    times the frame's size):
      id      (H, W) int material ids (`vector_table`'s; 0 where nothing is)
      shade   (H, W) Workbench's studio light with cast shadows on plain white, linear (a share of LIGHT_FULL)
      depth   (H, W) metres from the camera (anything where nothing is)
      tex     (H, W, 3) the textures, flat, display space: only for drawn tones
    Every material is filled with its tone: flat, lit or in shadow (`shade` under `spec["shadow"] * LIGHT_FULL`, opened
    and closed so no sliver or pinhole is left), or drawn (the texture's luma picks the colour; the last colour takes only
    texels greyer than `grey`). Then the inner lines (where two line groups meet, where the depth breaks) and the outline
    round everything, `spec["lines"]` px wide at 1080 on the short side. `subject = False` leaves the figures out.
    `flip` (H, W) bool paints those pixels in the look's `opposite` palette (the inside of the rings); `edges_on` is
    [(mask (H, W) bool, rgb)] painted last (the rings' wavefronts)."""
    import numpy as np
    slots, keys = _vector_slots(p, spec, table, scale, subject)
    out = _vector_palette(spec, keys)[slots]
    if flip is not None and flip.any():
        if spec.get("opposite") is None:
            raise ValueError("rings flip the picture to the [vector] opposite palette, and the look has none")
        out[flip] = _vector_palette(spec["opposite"], keys)[slots[flip]]
    for mask, rgb in edges_on:
        out[mask] = rgb
    H, W = slots.shape
    s = int(scale)
    return out.reshape(H // s, s, W // s, s, 3).mean((1, 3)).astype(np.float32)
