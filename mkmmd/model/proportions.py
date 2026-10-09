"""Whole-figure changes of the [proportions] tables, made on the spec the builders see before any part reads it
(`BuildCtx` calls `apply`):

  leg_extra   metres the legs grow: every height from the hip joint (landmark leg.L) up rises by it, heights between the
              ankle and the hip stretch by one factor (the thigh and the shin alike), the ankle and the feet stay; the
              torso's sections rise whole, the cuts under the hip joint too (they are the torso's, not the legs')

Heights are the numbers the parts read as absolute z (HEIGHTS): the landmarks' z, [proportions.sections.torso] z and
[proportions.sections.neck] z, [proportions.head] crown_skull, chin_point and forehead_front, [proportions.face] eye_z,
brow_z, nose_tip and mouth_z, and the guide heights of the hair and the outfit. A part that reads a new absolute height
from [proportions] adds it to HEIGHTS, or leg_extra leaves it behind."""

# table under [proportions] -> {key: where the z is: None (the value), an index into a list, or "all" (a list of z)}
HEIGHTS = {
    ("sections", "torso"): {"z": "all"},
    ("sections", "neck"): {"z": None},
    ("head",): {"crown_skull": None, "chin_point": 1, "forehead_front": 1},
    ("face",): {"eye_z": None, "brow_z": None, "nose_tip": 1, "mouth_z": None},
    ("hair_guides",): {"hairline_centre_z": None, "hairline_temple": 1, "cat_ear_base": 2},
    ("outfit_guides",): {"collar_top_z": None, "sash_z": None},
}
WHOLE = {("sections", "torso")}               # rise by leg_extra wherever they are: the torso is lofted through them


def leg_stretch(hip, ankle, extra):
    """z -> z with the legs `extra` longer: above the hip joint z + extra, between the ankle and the hip stretched, below
    the ankle as it was."""
    k = (hip + extra - ankle) / (hip - ankle)

    def f(z):
        z = float(z)
        return z + extra if z >= hip else z if z <= ankle else ankle + (z - ankle) * k
    return f


def apply(spec):
    """Make `spec`'s [proportions] changes (leg_extra) in its tables, in place: new values replace the old ones, lists are
    not changed in place (the loaded spec shares them). Only `leg_extra` is read as a key: the heights are looked up
    without counting as read (`spec.unread` still reports the ones no part uses)."""
    prop = spec.get("proportions")
    extra = float((prop or {}).get("leg_extra") or 0.0)
    if not extra:
        return
    land = dict.get(prop, "landmarks") or {}
    if "leg.L" not in dict.keys(land) or "ankle.L" not in dict.keys(land):
        raise ValueError("[proportions] leg_extra needs the landmarks leg.L (hip joint) and ankle.L")
    f = leg_stretch(float(dict.get(land, "leg.L")[2]), float(dict.get(land, "ankle.L")[2]), extra)
    for k, v in list(dict.items(land)):
        dict.__setitem__(land, k, [v[0], v[1], f(v[2])])
    for path, keys in HEIGHTS.items():
        g = (lambda z: float(z) + extra) if path in WHOLE else f
        t = prop
        for name in path:
            t = dict.get(t, name) if isinstance(t, dict) else None
        if not isinstance(t, dict):
            continue
        for key, at in keys.items():
            v = dict.get(t, key)
            if v is None:
                continue
            if at is None:
                v = g(v)
            elif at == "all":
                v = [g(z) for z in v]
            else:
                v = list(v)
                v[at] = g(v[at])
            dict.__setitem__(t, key, v)
