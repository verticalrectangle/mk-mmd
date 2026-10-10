"""Freezes (hit-stops), bpy-free (docs/design.md: Shots: Freezes): `[[freeze]] {from, to}` in clip seconds holds the world
on the frame of `from` until the frame of `to` while the cameras go on moving, and the looks (rings, styles) go on with
the clock: the world stops on the hit and the camera swings round it. At `to` the world is where the timeline has got to,
so moves keyed on the beat stay on it.

The shots stage keeps the windows in scene["mk_freeze"] as Blender frames; `mk render` and `mk look` set the scene to
`source(windows, f)` and the camera to its own keys at `f` (mkmmd.blender.freeze)."""


class FreezeError(ValueError):
    """A bad `[[freeze]]`."""


def normalize(specs, fps, frame0):
    """`[[freeze]]` entries -> sorted [[f0, f1], ...] Blender frames: the world holds frame f0 on frames f0 .. f1 - 1. A
    window shorter than a frame, or one that overlaps another, is an error."""
    out = []
    for i, spec in enumerate(specs or []):
        what = f"freeze {i}"
        if not isinstance(spec, dict) or set(spec) - {"from", "to"} or not {"from", "to"} <= set(spec):
            raise FreezeError(f"{what} = {spec!r}: expected {{from, to}} (clip seconds)")
        f0, f1 = (int(round(frame0 + float(spec[k]) * fps)) for k in ("from", "to"))
        if f1 <= f0:
            raise FreezeError(f"{what}: `to` must come at least a frame after `from`")
        out.append([f0, f1])
    out.sort()
    for a, b in zip(out, out[1:]):
        if b[0] < a[1]:
            raise FreezeError(f"freezes at frames {a} and {b} overlap")
    return out


def source(windows, frame):
    """The frame the world is evaluated at on `frame`: the start of the freeze it falls in, else the frame itself."""
    for f0, f1 in windows:
        if f0 <= frame < f1:
            return f0
    return frame
