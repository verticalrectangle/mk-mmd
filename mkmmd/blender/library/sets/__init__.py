"""Procedural sets (docs/design.md: Sets). A set builder makes its objects under one root empty and returns its card.

    builder(name, coll, root, spec, palette) -> card
      name     the set's name (prefix every object with it)
      coll     the collection to link objects into
      root     the set's root empty (its frame; world placement is the build stage's job)
      spec     the [[set]] table from mk.toml (builder-specific keys, documented in the builder's docstring)
      palette  {slot: "#rrggbb"} the project's palette (docs/design.md: Palettes); builders colour by slot names
    card = {
      "kind": str,
      "paths": {name: {"points": [[x, y, z], ...] (centerline control points, root frame), "width": m,
                       "lanes": [{"name", "offset": m left of the centerline, "dir": +1 | -1}]}},
      "use": {"look": [{"name", "point"}], "surface": [{"name", "center", "normal", "up", "size": [w, h]}]},
      "colliders": [collider specs (docs/design.md: Colliders)],
      "lights": [object names],
    }
Surfaces are flat faces (sign boards, billboards, screens) where kinetic type or images go later."""

BUILDERS = {}


def register(name):
    def deco(fn):
        BUILDERS[name] = fn
        return fn
    return deco


def load_all():
    """Import every builder module in this package (they register themselves)."""
    import importlib
    import pkgutil
    for mod in pkgutil.iter_modules(__path__):
        importlib.import_module(f"{__name__}.{mod.name}")
    return BUILDERS
