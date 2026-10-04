"""Cast members ([[cast]] in mk.toml): naming rules shared by the build and the checks."""


def armature_name(member):
    """The armature object a cast member gets: `armature` when given, else "<Name>_arm"."""
    if member.get("armature"):
        return member["armature"]
    n = member["name"]
    return f"{n[:1].upper()}{n[1:]}_arm"


def root_name(member):
    n = member["name"]
    return f"{n[:1].upper()}{n[1:]}"
