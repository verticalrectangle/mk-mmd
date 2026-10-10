"""The per-member tables of mk.toml, bpy-free (docs/design.md: Cast, Posing, Perform, Sim): `[pose.<cast>]`,
`[perform.<cast>]` and `[sim.<cast>]` are named after a cast member and hold the keys their stage reads; anything else is
a typo the stage would otherwise skip without a word. `check_section` refuses a table that names nobody and a key that is
in no schema, naming what there is.

A schema maps each key a stage reads to what is below it: None (anything: numbers, strings, targets, lists of numbers), a
tuple (the keys of a table, or of every table in a list of tables, e.g. `[[pose.<cast>.drape]]` or the events of
`gaze`; values of other kinds pass) or a dict (the same, one schema per key). The stages read the keys; this is the list of
the ones they read, so a key added to a stage is added here too."""
from . import perform as PF
from . import strum

SIDES = ("L", "R")

HAND = {k: None for k in ("at", "dir", "palm", "pole", "rest", "along", "offset", "lift", "ride", "fingers", "grip", "clock",
                          "approach", "wrap", "edge", "face", "seeds", "skin_radius", "posture", "track", "channel", "fret",
                          "chord", "press", "move", "thumb", "tip")}
HAND["keys"] = ("t", "at", "dir", "palm", "fret", "chord", "move", "fingers")
HAND["wobble"] = ("deg", "tau", "seed")

POSE = {
    "sit": ("hip", "facing", "floor_z", "pelvis_deg", "back_deg"), "sit_offset": None, "feet": SIDES,
    "hips": ("shift", "roll", "yaw"), "toes": SIDES, "lean": None, "turn": None, "lean_share": None, "turn_share": None,
    "head": ("pitch", "yaw", "roll", "neck"), "hands": {s: HAND for s in SIDES}, "fingers": SIDES,
    "drape": ("chain", "dirs", "scale"),
}

PERFORM = {
    "look": None, "gaze": ("t", "at", "hold", "back", "rise", "blink"),
    "glance": ("t", "at", "dur", "pitch", "rise", "fall", "blink"), "head_share": None,
    "head_limits": tuple(PF.HEAD_LIMITS), "neck_share": None, "eye_max": None, "breath": ("per_min", "deg"),
    "sway": ("deg", "period"), "nod": ("deg", "period"), "bob": ("deg", "timeline", "beats", "downbeat_accent"),
    "startle": None, "lean": None, "turn": None, "tilt": None, "head_tilt": None, "rock": ("deg", "period", "phase", "axis"),
    "head_rock": ("deg", "period", "phase", "axis"), "blink": ("per_min", "seed", "extra"), "lids": None,
    "sing": ("timeline", "lines", "words", "mouth", "lead", "voice"), "expressions": ("morph", "keys"),
    "twitch": ("bones", "family", "t", "deg", "axis", "dur"),
    "bounce": ("depth", "timeline", "beats", "downbeat_accent", "from", "to", "attack", "decay"), "rise": None,
    "kick": ("foot", "height", "back", "hold", "timeline", "beats", "downbeat_accent", "from", "to", "attack", "decay"),
    "strum": ("hand", "prop", "grip", "timeline", "sigma", "rhythm", "from", "to", "accent", "kick", "windmill",
              "windmill_dur") + tuple(strum.DEFAULTS),
    "drum": ("hand", "fingers", "deg", "lift", "roll", "timeline", "beats", "downbeat_accent", "from", "to"),
}

SIM = {k: None for k in ("families", "params", "colliders", "props", "fingers", "floor", "wind", "use_masks", "anchor_free",
                         "substeps", "settle_s", "engine")}

SCHEMAS = {"pose": POSE, "perform": PERFORM, "sim": SIM}


class TableError(ValueError):
    """A per-member table that names no cast member, or holds a key its stage does not read."""


def check(value, schema, header, inner=""):
    """Raise TableError when the table (or each table of a list) `value` holds a key `schema` does not have, and likewise
    below it for a dict schema. `header` names the member's table (`[pose.rin]`) and `inner` the place inside it
    (`hands.L`, `gaze[1]`) in the message. Values that are no tables are not looked at."""
    if schema is None:
        return
    if isinstance(value, list):
        for i, item in enumerate(value):
            if isinstance(item, dict):
                check(item, schema, header, f"{inner}[{i}]")
        return
    if not isinstance(value, dict):
        return
    known = list(schema)
    unknown = sorted(set(value) - set(known))
    if unknown:
        where = f"{header} {inner}" if inner else header
        raise TableError(f"{where}: unknown key{'s' if len(unknown) > 1 else ''} {', '.join(repr(k) for k in unknown)} "
                         f"(known: {', '.join(sorted(known))})")
    if isinstance(schema, dict):
        for k, v in value.items():
            check(v, schema[k], header, f"{inner}.{k}" if inner else k)


def check_section(section, tables, cast, schema=None):
    """The tables of `[<section>.<name>]` (`tables`, the section as parsed) against the project's cast names: each is a
    cast member's, and holds only keys of the section's schema (default SCHEMAS[section])."""
    schema = SCHEMAS[section] if schema is None else schema
    if tables is None:
        return
    members = ", ".join(cast) if cast else "none: the project has no [[cast]]"
    if not isinstance(tables, dict):
        raise TableError(f"[{section}] holds tables named after cast members, [{section}.<cast>] (cast: {members})")
    for name, table in tables.items():
        header = f"[{section}.{name}]"
        if not isinstance(table, dict):
            raise TableError(f"[{section}] has `{name}` as a plain key: it holds tables named after cast members, "
                             f"[{section}.<cast>], and the keys go inside them (cast: {members})")
        if name not in cast:
            raise TableError(f"{header}: no cast member {name!r} (cast: {members})")
        check(table, schema, header)


def floor_z(value, where="[sim.<cast>] floor"):
    """The height in metres of the ground a `floor` key asks for: `true` is the ground at z 0 (what the key defaults to),
    `false` no floor (None), a number that height."""
    if value is True:
        return 0.0
    if value is False:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    raise TableError(f"{where} = {value!r}: expected true (the ground at z 0), false (no floor) or a height in metres")
