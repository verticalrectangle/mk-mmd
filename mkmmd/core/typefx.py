"""Type effects, the numbers (docs/design.md: Text, Lyrics): the bpy-free half of `kinetic`, `backing` and `outline` of the
build's `text` stage. Validation of the three tables, their defaults and the size of the strip behind a word; the node
trees, materials and keys are built by `mkmmd/blender/build/text.py`. Errors are `TextError`s without the text's name (the
stage adds it) and never carry the string a text shows."""
from .typeset import TextError

# ------------------------------------------------------------------------------------------------------ kinetic
# `kinetic = {...}` (times in clip seconds, [[t, v], ...] lists are LINEAR keys, one per frame where it matters):
#   show = [t_on, t_off]   the word is on screen from the frame at t_on until the frame at t_off (hidden after; optional)
#   scale, sx, sy          uniform scale and per-axis squash / stretch about the pivot (1 = rest)
#   dx, dy                 offset along the panel's right / up, in em (the String to Curves size)
#   dxp, dyp               the same, in shares of the panel's width / height
#   rot                    rotation about the pivot, degrees (counter-clockwise seen from the front)
#   pivot                  "center" (default) | "bottom" | "top" of the word's ink box
#   tracking               Character Spacing multiplier over the text's own `tracking` (letters spreading)
#   weight                 radius (em) the filled letters are grown by (a bolder weight: copies on a ring)
#   opacity                0..1 alpha of the whole word (alpha-hashed, so the material is `DITHERED`)
#   tint = {color, keys}   mix toward another palette colour, 0 = the text's own colour
#   drip = {t, g, stretch, life, seed}   the exit: from clip time t every letter stretches downward from its top and
#                          falls (`g` px/s^2 at a 150 px em, `stretch` extra height, gone after `life` s), the first
#                          letter on t, the rest within 0.1 s; opacity fades per letter
#   headroom               share the fit allows for the letters' spread (the lyrics stage sets it from the spread)
KINETIC_KEYS = {"show", "lyric", "scale", "sx", "sy", "dx", "dy", "dxp", "dyp", "rot", "tracking", "weight", "opacity",
                "tint", "drip", "pivot", "headroom"}
MOTION = ("scale", "sx", "sy", "dx", "dy", "dxp", "dyp", "rot")
KEYED = MOTION + ("tracking", "weight", "opacity")          # channels that are plain key lists
DRIP_KEYS = {"t", "g", "stretch", "life", "seed"}
LOOK_KEYS = ("value", "blink", "flicker", "fade", "ghost", "halo")      # emission tricks that need their own material
PIVOTS = ("center", "bottom", "top")


def key_list(v, what):
    """`[[t, v], ...]` as sorted (float, float) pairs."""
    try:
        keys = sorted((float(t), float(x)) for t, x in v)
    except (TypeError, ValueError):
        raise TextError(f"kinetic {what} is a list of [t, value] keys") from None
    if not keys:
        raise TextError(f"kinetic {what} has no keys")
    return keys


def kinetic_spec(spec):
    """The validated `kinetic` table of a text (keys as sorted (t, v) lists), or None when it has none. A kinetic text
    takes none of LOOK_KEYS: they need a material of their own."""
    k = spec.get("kinetic")
    if k is None:
        return None
    if not isinstance(k, dict):
        raise TextError("kinetic must be a table")
    unknown = sorted(set(k) - KINETIC_KEYS)
    if unknown:
        raise TextError(f"unknown kinetic keys {unknown} (known: {sorted(KINETIC_KEYS)})")
    for bad in LOOK_KEYS:
        if spec.get(bad) is not None:
            raise TextError(f"kinetic cannot be combined with `{bad}`")
    out = {}
    for key, v in k.items():
        if key in KEYED:
            out[key] = key_list(v, key)
        elif key == "tint":
            if not isinstance(v, dict) or "color" not in v or not v.get("keys"):
                raise TextError("kinetic tint is {color = slot, keys = [[t, v], ...]}")
            out[key] = {"color": v["color"], "keys": key_list(v["keys"], "tint")}
        elif key == "show":
            if not isinstance(v, (list, tuple)) or not 1 <= len(v) <= 2:
                raise TextError("kinetic show is [t_on] or [t_on, t_off]")
            out[key] = [float(x) for x in v]
        elif key == "pivot":
            if v not in PIVOTS:
                raise TextError(f"kinetic pivot is one of {list(PIVOTS)}")
            out[key] = v
        elif key == "drip":
            if not isinstance(v, dict) or set(v) - DRIP_KEYS or "t" not in v:
                raise TextError("kinetic drip is {t, g, stretch, life, seed} (t required)")
            out[key] = {"t": float(v["t"]), "g": float(v.get("g", 2600.0)), "stretch": float(v.get("stretch", 3.2)),
                        "life": float(v.get("life", 0.55)), "seed": int(v.get("seed", 1))}
        elif key == "headroom":
            out[key] = float(v)
        else:
            out[key] = v
    return out


# ------------------------------------------------------------------------------------------------------ backing
BACKING_KEYS = {"color", "pattern", "pattern_color", "pad", "height", "torn", "dz", "glow", "scale", "seed"}
PATTERNS = ("plain", "stripe", "dots", "check")
TAPE_SCALE = {"stripe": (0.012, 0.38), "dots": (0.0165, 0.20), "check": (0.0102, 0.5)}   # pitch (m), duty / dot radius


def backing_spec(spec):
    """The validated `backing` of a text: {color, pattern, pattern_color, pad [x, y] em, height em or None, torn (share of
    the height), dz m, glow, scale (pattern pitch factor), seed} or None."""
    b = spec.get("backing")
    if b is None or b is False:
        return None
    b = {} if b is True else b
    if not isinstance(b, dict):
        raise TextError("backing is a table (color, pattern, pad, ...)")
    unknown = sorted(set(b) - BACKING_KEYS)
    if unknown:
        raise TextError(f"unknown backing keys {unknown} (known: {sorted(BACKING_KEYS)})")
    if b.get("pattern", "plain") not in PATTERNS:
        raise TextError(f"backing pattern must be one of {list(PATTERNS)}")
    pad = b.get("pad", (0.55, 0.3))
    if not isinstance(pad, (list, tuple)) or len(pad) != 2:
        raise TextError("backing pad is [x, y] in em")
    out = {"color": b.get("color", "surface"), "pattern": b.get("pattern", "plain"),
           "pattern_color": b.get("pattern_color", "surface"), "pad": [float(pad[0]), float(pad[1])],
           "height": None if b.get("height") is None else float(b["height"]), "torn": float(b.get("torn", 0.1)),
           "dz": float(b.get("dz", 0.0006)), "glow": float(b.get("glow", 0.3)), "scale": float(b.get("scale", 1.0)),
           "seed": int(b.get("seed", 1))}
    if out["height"] is not None and not out["height"] > 0 or not out["scale"] > 0 or not 0.0 <= out["torn"] < 0.5:
        raise TextError("backing height and scale must be positive and torn in [0, 0.5)")
    return out


def tape_size(b, ink_w, ink_h, em):
    """(width, height) in metres of the strip behind an ink box (em units at size 1) set at `em`: the ink plus `pad` each
    side, or the fixed `height` (em)."""
    h = b["height"] * em if b["height"] is not None else (ink_h + 2.0 * b["pad"][1]) * em
    return (ink_w + 2.0 * b["pad"][0]) * em, h


# ------------------------------------------------------------------------------------------------------ outline
OUTLINE_KEYS = {"color", "width", "alpha", "dz"}


def outline_spec(spec):
    """The validated `outline` of a text: {color, width (em), alpha, dz (m)} or None."""
    o = spec.get("outline")
    if o is None or o is False:
        return None
    o = {} if o is True else o
    if not isinstance(o, dict):
        raise TextError("outline is a table (color, width, alpha)")
    unknown = sorted(set(o) - OUTLINE_KEYS)
    if unknown:
        raise TextError(f"unknown outline keys {unknown} (known: {sorted(OUTLINE_KEYS)})")
    out = {"color": o.get("color", "text"), "width": float(o.get("width", 0.05)), "alpha": float(o.get("alpha", 0.5)),
           "dz": float(o.get("dz", 0.0003))}
    if not out["width"] > 0 or not 0.0 < out["alpha"] <= 1.0:
        raise TextError("outline width must be positive and alpha in (0, 1]")
    return out
