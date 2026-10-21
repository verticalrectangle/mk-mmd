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
VECTOR_ONLY = ("tones", "materials", "lines", "sketch")                 # shot keys only the vector style reads
VECTOR_ROLES = ("background", "line", "inner")
VECTOR_COLORS = {"background": "base", "line": "text"}                  # `inner` defaults to `line`
VECTOR_TONES = {"fill": {"lit": "text", "shade": "subtle"}}             # a project without [vector] tones
VECTOR_LINES = {"outline": 5.0, "inner": 2.0}                           # line widths, px at 1080 on the frame's short side
VECTOR_LIGHT = (0.45, -0.35, 0.82)                                      # Workbench's shadow direction (toward the light)
TEX_KINDS = ("drawn", "cel")                                            # tones that read the texture pass
SKETCH = {"paper": "text", "ink": "base", "boil": 1.2, "fps": 12.0, "strokes": 2, "grain": 0.35, "hatch": 0.0,
          "seed": 1}                                                    # a vector look drawn as line art (`sketch`)
SKETCH_CELL = 48.0                                                      # px at 1080 between two bends of a sketch's wobble
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
    flat), `{colors, at, grey}` (drawn: the texture's brightness picks one of the colours, so what the model draws on
    itself, eyes and mouth, carries over) or `{texture = true, shade}` (cel: the model's own texture colours, multiplied by
    the `shade` colour where the look's light does not reach; no `shade`: the textures under flat light)."""
    col = lambda c: _rgb(resolve_colour(c, palette))
    if not isinstance(t, dict):
        return {"kind": "flat", "lit": col(t)}
    kind = "drawn" if "colors" in t else "cel" if "texture" in t else "lit"
    known = {"drawn": {"colors", "at", "grey"}, "cel": {"texture", "shade"}, "lit": {"lit", "shade"}}[kind]
    unknown = sorted(set(t) - known)
    if unknown:
        raise StyleError(f"vector tone {name!r}: unknown keys {unknown} (a tone is a colour, {{lit, shade}}, "
                         f"{{colors, at, grey}} or {{texture = true, shade}})")
    if kind == "drawn":
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
    if kind == "cel":
        if t["texture"] is not True:
            raise StyleError(f"vector tone {name!r}: texture = {t['texture']!r}: true (the model's own colours)")
        return {"kind": "cel", "lit": [1.0, 1.0, 1.0], "shade": col(t["shade"]) if "shade" in t else None}
    if "lit" not in t:
        raise StyleError(f"vector tone {name!r}: needs lit (or colors, or texture)")
    if "shade" not in t:
        return {"kind": "flat", "lit": col(t["lit"])}
    return {"kind": "lit", "lit": col(t["lit"]), "shade": col(t["shade"])}


def _material_rule(what, rule, tones):
    """One rule of a vector look's `materials` (`what` names it in messages and in the normalised rule's `where`)."""
    if not isinstance(rule, dict) or set(rule) - {"match", "tone", "group"} or not {"match", "tone"} <= set(rule):
        raise StyleError(f"{what} = {rule!r}: expected {{match = [material name patterns], tone, group}}")
    match = [rule["match"]] if isinstance(rule["match"], str) else rule["match"]
    if not isinstance(match, list) or not match or not all(isinstance(p, str) for p in match):
        raise StyleError(f"{what}: match = {rule['match']!r}: expected material name patterns")
    if rule["tone"] not in tones:
        raise StyleError(f"{what}: tone {rule['tone']!r} is not one of the tones ({', '.join(tones)})")
    group = rule.get("group")
    return {"match": list(match), "tone": str(rule["tone"]), "group": None if group is None else str(group),
            "where": what}


def _lines(given, where, lines):
    """`lines = {outline, inner}` (px at 1080 on the frame's short side) laid over `lines`."""
    if given is None:
        return lines
    if not isinstance(given, dict) or set(given) - set(VECTOR_LINES):
        raise StyleError(f"{where} = {given!r}: expected {{outline, inner}} (px at 1080)")
    out = dict(lines)
    for k, v in given.items():
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise StyleError(f"{where}.{k} = {v!r}: a width in px at 1080")
        out[k] = float(v)
        if out[k] < 0:
            raise StyleError(f"{where}.{k} must not be negative")
    return out


def _sketch_spec(s, palette):
    """`sketch = true | {paper, ink, boil, fps, strokes, grain, hatch, seed}`: the vector look drawn as line art
    (`_sketch`), with SKETCH's defaults; None without one."""
    if s is None or s is False:
        return None
    s = {} if s is True else s
    if not isinstance(s, dict):
        raise StyleError(f"sketch = {s!r}: expected true or a table {{{', '.join(SKETCH)}}}")
    unknown = sorted(set(s) - set(SKETCH))
    if unknown:
        raise StyleError(f"sketch: unknown keys {unknown} (known: {', '.join(SKETCH)})")
    o = {**SKETCH, **s}
    num = {}
    for k in ("boil", "fps", "grain", "hatch"):
        if isinstance(o[k], bool) or not isinstance(o[k], (int, float)):
            raise StyleError(f"sketch: {k} = {o[k]!r} is not a number")
        num[k] = float(o[k])
    if num["boil"] < 0 or num["hatch"] < 0:
        raise StyleError("sketch: boil and hatch are px at 1080, 0 or more")
    if num["fps"] <= 0:
        raise StyleError(f"sketch: fps = {num['fps']}: how many times a second the lines are drawn again, above 0")
    if not 0.0 <= num["grain"] <= 1.0:
        raise StyleError(f"sketch: grain = {num['grain']}: a share in 0..1")
    strokes, seed = o["strokes"], o["seed"]
    if isinstance(strokes, bool) or not isinstance(strokes, int) or not 1 <= strokes <= 4:
        raise StyleError(f"sketch: strokes = {strokes!r}: how many times each line is drawn, 1 .. 4")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise StyleError(f"sketch: seed = {seed!r}: a whole number, 0 or more")
    return {"paper": _rgb(resolve_colour(o["paper"], palette)), "ink": _rgb(resolve_colour(o["ink"], palette)), **num,
            "strokes": strokes, "seed": seed}


def normalize_vector(spec, palette, project=None):
    """The flat-vector look of a shot: the project's `[vector]` table (`project`) with the shot's own `colors`, `tones`,
    `lines`, `hide` and `keep` laid over it, its own `materials` rules checked before the project's, and its `sketch`
    (docs/design.md: Looks: vector)."""
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
    own = spec.get("materials") or []
    if not isinstance(own, list):
        raise StyleError("materials: expected a list of {match, tone, group}")
    lines = _lines(spec.get("lines"), "lines", _lines(project.get("lines"), "[vector] lines", dict(VECTOR_LINES)))
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
            "materials": [_material_rule(f"materials[{i}]", r, tones) for i, r in enumerate(own)]
                         + [_material_rule(f"[vector] materials[{i}]", r, tones) for i, r in enumerate(rules)],
            "default": "fill" if "fill" in tones else next(iter(tones)), "lines": lines, "shadow": shadow,
            "light": light, "sketch": _sketch_spec(spec.get("sketch"), palette), "hide": _patterns(spec, "hide"),
            "keep": _patterns(spec, "keep")}


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
        if new["kind"] != base["kind"] or len(new.get("colors", ())) != len(base.get("colors", ())) or \
                (new["kind"] == "cel" and (new["shade"] is None) != (base["shade"] is None)):
            raise StyleError(kept + ("; a cel tone keeps its shade or its lack of one" if new["kind"] == "cel" else ""))
        out[name] = new
    return {"colors": {k: _rgb(resolve_colour({**colors, **oc}[k], palette)) for k in VECTOR_ROLES}, "tones": out}


def normalize(spec, palette, vector=None):
    """{'silhouette' | 'vector': {...}} and / or {'reflection': {...}} for one shot spec (an aspect's overrides merged
    in); {} when the shot has none. `vector` is the project's `[vector]` table. `style = "none"` and `reflection = false`
    switch an inherited look off for one aspect, with the look's own keys it inherits."""
    style = spec.get("style")
    reflection = spec.get("reflection")
    out = {}
    if style not in (None, "none"):
        if style not in STYLES:
            raise StyleError(f"shot {spec.get('name')!r}: style = {style!r} (known: {', '.join(STYLES)})")
        if style == "vector":
            out["vector"] = normalize_vector(spec, palette, vector)
        else:
            _vector_only(spec)
            out["silhouette"] = normalize_silhouette(spec, palette)
    elif style is None:
        _vector_only(spec)
    if reflection is not None and reflection is not False:
        if out:
            raise StyleError(f"shot {spec.get('name')!r}: a {style} shot cannot have a reflection")
        out["reflection"] = normalize_reflection(reflection, palette)
    return out


def _vector_only(spec):
    """The vector style's own keys (`tones`, `materials`, `lines`, `sketch`) on a shot of another look are a mistake."""
    given = [k for k in VECTOR_ONLY if k in spec]
    if given:
        raise StyleError(f"shot {spec.get('name')!r}: {', '.join(given)} belong to the vector style")


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


def vector_sort(objects, spec, types=()):
    """The vector look's objects (Obj records of every object that may render, in the scene's order): {"hide": hidden by
    the shot's `hide` and not kept by `keep`, "type": the text stage's objects among the rest (`types`, by name), drawn
    over the picture, "subject": everything else, drawn}, each a list of names in the records' order."""
    hide = set(select(objects, spec["hide"])) - set(select(objects, spec["keep"]))
    out = {"hide": [], "subject": [], "type": []}
    for ob in objects:
        out["hide" if ob.name in hide else "type" if ob.name in types else "subject"].append(ob.name)
    return out


def vector_materials(lists):
    """The materials of a vector look's subject, every name once in the order first met: `lists` holds each subject
    object's material names (its slots, then the Set Material nodes of its geometry nodes). `vector_table` numbers them."""
    out, seen = [], set()
    for names in lists:
        for n in names:
            if n not in seen:
                seen.add(n)
                out.append(n)
    return out


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


def _lit(p, spec, scale):
    """Where the look's light reaches, (H, W) bool: `shade` at or over `spec["shadow"] * LIGHT_FULL`, opened and closed so
    no lit sliver or shadow pinhole is left."""
    r = max(1, int(round(scale)))
    lit = p["shade"] >= spec["shadow"] * LIGHT_FULL
    lit = dilate_round(~dilate_round(~lit, r), r)                            # opened: no lit slivers
    return ~dilate_round(~dilate_round(lit, r), r)                           # closed: no shadow pinholes


def _vector_slots(p, spec, table, scale, subject):
    """(slots, keys, lit): every pass pixel's colour as an index into `keys`, the colour roles of the look: "background",
    "line", "inner" and (tone, "lit" | "shade") or (tone, "band", i). A palette (`_vector_palette`) turns them into colours,
    so the drawing is worked out once and painted in as many palettes as the frame needs. `lit` is `_lit`'s mask (None
    when the figures are left out)."""
    import numpy as np
    tones_of, groups_of = table
    mid = p["id"]
    H, W = mid.shape
    keys = ["background", "line", "inner"]
    slots = np.zeros((H, W), np.int16)
    if not subject:
        return slots, keys, None
    cover = mid > 0
    names = list(spec["tones"])
    T = np.asarray([-1] + [names.index(t) for t in tones_of[1:]], np.int32)[mid]
    r = max(1, int(round(scale)))
    lit = _lit(p, spec, scale)
    tex = p.get("tex")
    for k, name in enumerate(names if not spec.get("sketch") else ()):          # a sketch draws its lines alone
        m = cover & (T == k)
        if not m.any():
            continue
        t = spec["tones"][name]
        if t["kind"] == "flat" or (t["kind"] == "cel" and t["shade"] is None):
            slots[m] = len(keys)
            keys.append((name, "lit"))
        elif t["kind"] in ("lit", "cel"):
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
    return slots, keys, lit


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


def compose_vector(p, spec, table, scale=VECTOR_SCALE, subject=True, flip=None, edges_on=(), t=None):
    """The finished vector frame, float32 RGB (h, w, 3) in display space, from the passes `p` (numpy arrays at `scale`
    times the frame's size):
      id      (H, W) int material ids (`vector_table`'s; 0 where nothing is)
      shade   (H, W) Workbench's studio light with cast shadows on plain white, linear (a share of LIGHT_FULL)
      depth   (H, W) metres from the camera (anything where nothing is)
      tex     (H, W, 3) the textures, flat, display space: only for drawn and cel tones
    Every material is filled with its tone: flat, lit or in shadow (`shade` under `spec["shadow"] * LIGHT_FULL`, opened
    and closed so no sliver or pinhole is left), drawn (the texture's luma picks the colour; the last colour takes only
    texels greyer than `grey`) or cel (the texture itself, times the tone's shade colour in shadow). Then the inner lines
    (where two line groups meet, where the depth breaks) and the outline round everything, `spec["lines"]` px wide at 1080
    on the short side. `subject = False` leaves the figures out. `flip` (H, W) bool paints those pixels in the look's
    `opposite` palette (the inside of the rings); `edges_on` is [(mask (H, W) bool, rgb)] painted last (the rings'
    wavefronts). A look with a `sketch` is line art instead (`_sketch`: the lines alone, in ink on paper, boiling with
    the clip second `t`; no flip)."""
    import numpy as np
    slots, keys, lit = _vector_slots(p, spec, table, scale, subject)
    if spec.get("sketch"):
        out = _sketch(slots, lit, p, spec, t)
    else:
        out = _vector_palette(spec, keys)[slots]
        if flip is not None and flip.any():
            if spec.get("opposite") is None:
                raise ValueError("rings flip the picture to the [vector] opposite palette, and the look has none")
            out[flip] = _vector_palette(spec["opposite"], keys)[slots[flip]]
        cel = [i for i, k in enumerate(keys) if not isinstance(k, str) and spec["tones"][k[0]]["kind"] == "cel"]
        if cel:
            if p.get("tex") is None:
                raise ValueError("a cel tone takes the model's own colours from the textures: the passes need `tex`")
            m = np.isin(slots, cel)
            out[m] *= p["tex"][..., :3][m]
    for mask, rgb in edges_on:
        out[mask] = rgb
    H, W = slots.shape
    s = int(scale)
    return out.reshape(H // s, s, W // s, s, 3).mean((1, 3)).astype(np.float32)


def _smooth_noise(rng, shape, cell):
    """(H, W) float32 noise in -1..1 that bends once every `cell` pixels: random values on a coarse grid blended with
    smoothstep weights along each axis (two matrix products: no full-size temporaries but the result)."""
    import numpy as np
    H, W = shape
    cell = max(float(cell), 1.0)
    grid = rng.uniform(-1.0, 1.0, (int(H / cell) + 2, int(W / cell) + 2)).astype(np.float32)

    def weights(n, m):
        u = np.arange(n, dtype=np.float64) / cell
        i = np.floor(u).astype(np.intp)
        f = u - i
        f = f * f * (3.0 - 2.0 * f)
        w = np.zeros((n, m), np.float32)
        w[np.arange(n), i], w[np.arange(n), i + 1] = 1.0 - f, f
        return w
    return weights(H, grid.shape[0]) @ grid @ weights(W, grid.shape[1]).T


def _warp(m, dy, dx):
    """`m` (H, W) as a wobbling hand draws it: every pixel takes the value `dy`, `dx` pixels away (nearest)."""
    import numpy as np
    H, W = m.shape
    yy = np.clip(np.rint(np.arange(H, dtype=np.float32)[:, None] + dy), 0, H - 1).astype(np.int32)
    xx = np.clip(np.rint(np.arange(W, dtype=np.float32)[None, :] + dx), 0, W - 1).astype(np.int32)
    return m[yy, xx]


def _sketch(slots, lit, p, spec, t):
    """The vector look drawn as line art, float32 (H, W, 3) at the passes' size: paper everywhere and the look's outline
    and inner lines in ink, drawn `strokes` times by a hand that wobbles up to `boil` px (at 1080) and draws the picture
    again `fps` times a second of the clip (`t`), so the lines boil; each pass of a pencil (two strokes or more) a little
    lighter than ink, its pressure drifting along the line and broken by the paper's `grain`. With `hatch` (px at 1080
    between strokes) what the look's light does not reach is shaded with diagonal strokes, crossed by the second pass."""
    import numpy as np
    sk = spec["sketch"]
    H, W = slots.shape
    px = min(H, W) / 1080.0
    step = 0 if t is None else int(np.floor(float(t) * sk["fps"] + 1e-6))
    lines = (slots == 1) | (slots == 2)
    shadow = (p["id"] > 0) & ~lit if sk["hatch"] > 0 and lit is not None else None
    alpha = np.float32(1.0 if sk["strokes"] == 1 else 0.72)
    ys, xs = np.arange(H, dtype=np.float32)[:, None], np.arange(W, dtype=np.float32)[None, :]
    ink = np.zeros((H, W), np.float32)
    for s in range(sk["strokes"]):
        rng = np.random.default_rng([sk["seed"], s, step % (1 << 32)])
        dy = _smooth_noise(rng, (H, W), SKETCH_CELL * px) * np.float32(sk["boil"] * px)
        dx = _smooth_noise(rng, (H, W), SKETCH_CELL * px) * np.float32(sk["boil"] * px)
        a = _warp(lines, dy, dx).astype(np.float32)
        if shadow is not None:
            per = sk["hatch"] * px
            u = ((xs + dx) + (ys + dy) if s % 2 == 0 else (xs + dx) - (ys + dy)) / np.float32(per * np.sqrt(2.0))
            hatch = (np.abs(u - np.rint(u)) * np.float32(per) < 0.5 * px) & _warp(shadow, dy, dx)
            a = np.maximum(a, hatch.astype(np.float32) * np.float32(0.6))
        a *= alpha * (np.float32(0.8) + np.float32(0.2) * _smooth_noise(rng, (H, W), 4.0 * SKETCH_CELL * px))
        if sk["grain"] > 0:
            a *= np.float32(1.0) - np.float32(sk["grain"]) * rng.random((H, W), dtype=np.float32)
        ink = np.float32(1.0) - (np.float32(1.0) - ink) * (np.float32(1.0) - a)
    paper, pen = np.asarray(sk["paper"], np.float32), np.asarray(sk["ink"], np.float32)
    return paper + (pen - paper) * ink[..., None]
