"""Cast members ([[cast]] in mk.toml): naming rules and the `physics` word, shared by the build and the checks."""

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
