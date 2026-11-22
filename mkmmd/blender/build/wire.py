"""wire: the [[flight]] props baked where their targets ended up and their wires drawn, after everything that moves a
target (docs/design.md: Flights; mkmmd/blender/build/flight.py `finish` does it)."""
from . import flight as FLIGHT


def run(ctx):
    return FLIGHT.finish(ctx)
