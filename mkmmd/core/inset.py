"""A picture set into a clip, bpy-free (docs/design.md: Frame effects, Insets): the `[[fx]]` kind `inset` checked and
its view worked out; mkmmd/inset.py draws it.

    inset   the picture as it reached the effect set into a part of a video's frame (`into`): the cut playing on a
            screen in a recording (a window on a recorded desktop, a phone, a monitor on a desk), the recording round
            it. Its `view` runs the frame between the recording as a whole (0) and the picture alone (1): a push into
            the screen, a pull back out of it.

The clip covers the frame as a video sticker's covers its box (`focus`). `into` is in fractions of the clip's own frame,
as the file has it, so it is the same place in every output. At a view between 0 and 1 the frame is a window on the
recording between its whole frame and `into`: its size on a log scale (a view keyed linearly is a steady push) and its
middle on the line that goes with it, so every point of the recording keeps its heading, as under a zoom lens. At 1 the
window is `into` itself and the frame is exactly the picture that reached the effect, whatever shape `into` has: a part
of another shape shows the picture squeezed to it, as a player in a window does, and the squeeze goes with the view.

The effects after an inset work on the whole frame, the recording with the picture in it. Those that read a figure's
layer or a scene point read them in the picture's own frame: put them before the inset, or give them a figure from a
`shot` of its own, set in by `at` and `height`."""

KINDS = ("inset",)
KEYS = {"inset": {"clip", "in", "rate", "loop", "focus", "mirror", "into", "view"}}
DEFAULTS = {"inset": {"in": 0.0, "rate": 1.0, "loop": False, "focus": [0.5, 0.5], "mirror": False, "view": 0.0}}


def _fx():
    from . import framefx as FX          # framefx imports this module: its helpers are there by the time this runs
    return FX


def normalize(spec, out, palette, what):
    """An inset's own keys checked and filled in, in place (`out` has the kind's defaults under `spec`)."""
    FX = _fx()
    FX._clip_keys(out, what)
    if out.get("into") is None:
        raise FX.FxError(f"{what}: `into` is required: [x0, y0, x1, y1], the part of the clip's frame the picture is "
                         f"set into (fractions of the clip, 0, 0 its top left)")
    out["into"] = FX._part(out["into"], f"{what}: into")
    out["view"] = FX._keys(out["view"], f"{what} view")
    if any(not 0.0 <= v <= 1.0 for _, v in out["view"]):
        raise FX.FxError(f"{what}: view runs from 0 (the clip's whole frame) to 1 (the picture alone)")


def view(fx, t):
    """The inset's view at clip second `t`: linear between its keys, held outside them."""
    keys = fx["view"]
    if t <= keys[0][0]:
        return keys[0][1]
    for (t0, v0), (t1, v1) in zip(keys, keys[1:]):
        if t <= t1:
            return v0 + (v1 - v0) * (t - t0) / (t1 - t0)
    return keys[-1][1]


def window(box, size, v):
    """(x0, y0, x1, y1): the part of the frame of `size` (w, h), as the clip covers it at view 0, that fills the frame at
    view `v`, between the whole frame (0) and `box` (1: the picture's place there, px). Each axis by itself: its half
    size from the frame's to the box's on a log scale, its middle moving with the size so the axis keeps one fixed
    point, the one a zoom lens would hold still."""
    out = []
    for lo, hi, full in ((box[0], box[2], float(size[0])), (box[1], box[3], float(size[1]))):
        s0, s1, c0, c1 = full / 2.0, (hi - lo) / 2.0, full / 2.0, (lo + hi) / 2.0
        s = s0 * (s1 / s0) ** v
        c = c1 if abs(s1 - s0) < 1e-9 else c0 + (c1 - c0) * (s - s0) / (s1 - s0)
        out.append((c - s, c + s))
    return out[0][0], out[1][0], out[0][1], out[1][1]


def place(box, win, size):
    """(x0, y0, x1, y1): where `box` (px of the frame at view 0) lands in the frame of `size` when the window `win`
    (window) fills it."""
    kx, ky = size[0] / (win[2] - win[0]), size[1] / (win[3] - win[1])
    return (box[0] - win[0]) * kx, (box[1] - win[1]) * ky, (box[2] - win[0]) * kx, (box[3] - win[1]) * ky
