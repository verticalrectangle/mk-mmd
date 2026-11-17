"""Cast members ([[cast]] in mk.toml): naming rules, the `physics` word and the `shows` spans, shared by the build and the
checks."""

PHYSICS = ("mk", "none", "bullet")


def armature_name(member):
    """The armature object a cast member gets: `armature` when given, else "<Name>_arm"."""
    if member.get("armature"):
        return member["armature"]
    n = member["name"]
    return f"{n[:1].upper()}{n[1:]}_arm"


def root_name(member):
    n = member["name"]
    return f"{n[:1].upper()}{n[1:]}"


def physics(member):
    """`[[cast]] physics`: "mk" (the default: no Bullet, the chains come from rig.json and `[sim.<name>]` solves them), "none" (no
    secondary motion at all: no sim) or "bullet" (the author's rigid bodies). ValueError for any other word."""
    word = member.get("physics", "mk")
    if word not in PHYSICS:
        raise ValueError(f"physics = {word!r}: one of {', '.join(PHYSICS)}")
    return word


def shows(member):
    """`[[cast]] shows`: the clip seconds the member renders in, [(from, to), ...] in order (a shot's way: shown from the
    frame of `from` up to the frame of `to`), or None without the key (every frame). ValueError says what is wrong."""
    spans = member.get("shows")
    if spans is None:
        return None
    if not isinstance(spans, list) or not spans or not all(isinstance(s, list) and len(s) == 2 for s in spans):
        raise ValueError(f"shows = {spans!r}: a list of [from, to] clip seconds, e.g. shows = [[0.0, 9.5]]")
    out = []
    for a, b in spans:
        if not all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in (a, b)) or not a < b:
            raise ValueError(f"shows: [{a!r}, {b!r}] is not [from, to] in clip seconds with from before to")
        out.append((float(a), float(b)))
    return sorted(out)
