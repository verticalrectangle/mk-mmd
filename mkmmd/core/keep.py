"""Colour kept through the grade (docs/design.md: Post, `keep`): the bpy-free half of `[post] keep`. A black-and-white
picture with only a cast member's eyes in colour: `mk render` draws a mask of where the named materials are seen in each
frame (`mkmmd/blender/styles.py` keep_mask, `keep/<frame>.png` beside the frame) and `mk post` saturates by it, the
grade's own `saturation` elsewhere. This module checks the table and names the mask's file."""
KEYS = {"cast", "materials", "saturation"}
LAYER_DIR = "keep"


def spec(post, cast):
    """`[post] keep = {cast, materials, saturation}` (`post` the [post] table, `cast` the project's [[cast]] entries) ->
    {"cast", "materials", "saturation"} or None without the key. ValueError names what is wrong: an unknown key, a cast
    member the project does not have, no materials, a saturation below 0."""
    k = (post or {}).get("keep")
    if k is None:
        return None
    where = "[post] keep"
    if not isinstance(k, dict):
        raise ValueError(f'{where} is a table: {{cast = "<name>", materials = ["<PMX material>", ...], '  # noqa: TRY004
                         f"saturation = 1.0}}")
    bad = sorted(set(k) - KEYS)
    if bad:
        raise ValueError(f"{where}: unknown key{'s' if len(bad) > 1 else ''} {', '.join(bad)} "
                         f"(have {', '.join(sorted(KEYS))})")
    names = [c.get("name") for c in cast or []]
    if k.get("cast") not in names:
        raise ValueError(f"{where}: cast = {k.get('cast')!r} is not a [[cast]] member "
                         f"(have {', '.join(map(str, names)) or 'none'})")
    mats = k.get("materials")
    if isinstance(mats, str):
        mats = [mats]
    if not mats or not all(isinstance(m, str) and m for m in mats):
        raise ValueError(f'{where}: materials = [...] the PMX names of the materials that keep their colour (the eyes: "瞳")')
    sat = k.get("saturation", 1.0)
    if isinstance(sat, bool) or not isinstance(sat, (int, float)) or sat < 0:
        raise ValueError(f"{where}: saturation = {sat!r}: a number >= 0 (1 keeps the frame's colour)")
    return {"cast": k["cast"], "materials": list(mats), "saturation": float(sat)}


def rel(frame):
    """Path of a frame's mask inside a frame folder: `keep/<frame>.png`, grey, the frame's size: 1 where a kept material is
    seen whole, 0 where none is, the antialiased share in between."""
    return f"{LAYER_DIR}/{int(frame):05d}.png"
