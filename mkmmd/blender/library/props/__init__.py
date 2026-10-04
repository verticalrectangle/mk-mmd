"""Procedural props with their cards (docs/design.md: Prop card). A builder makes the objects under one root empty
(the prop's frame) and returns its card: use points (sit, feet, grip, rest, look, surface) in the prop's local frame
and collider specs on its hidden collider objects. Characters sitting in a prop face its -Y axis.

    builder(name, coll, root, slots) -> card      slots: {slot: "#hex"} colour overrides

Builders live in the modules of this package and register themselves with @register("name")."""

BUILDERS = {}


def register(name):
    def deco(fn):
        BUILDERS[name] = fn
        return fn
    return deco


def load_all():
    import importlib
    import pkgutil
    for mod in pkgutil.iter_modules(__path__):
        importlib.import_module(f"{__name__}.{mod.name}")
    return BUILDERS
