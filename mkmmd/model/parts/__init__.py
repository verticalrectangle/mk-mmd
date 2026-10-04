"""Part builders: one module per part (`mkmmd/model/parts/<name>.py`), each registering `@builder("<name>")`.

`KNOWN` lists the parts the framework knows about (used for messages); a spec may build any part whose module exists,
or name another module in `[model.builders]` (`tails = "mkmmd.model.parts.hair"`). Modules are imported lazily by
`mkmmd.model.build.get_builder`."""

KNOWN = ("mannequin", "body", "head", "hair", "outfit")


def module_for(name, overrides=None):
    """Dotted module path of the builder for part `name`."""
    if overrides and name in overrides:
        return str(overrides[name])
    return f"{__name__}.{name}"
