"""Every mesh of the electric bass, by object suffix (pure numpy, no bpy: the Blender builder only turns them into objects).

`parts(stripes)` -> {suffix: core.shell.Mesh}: body, pickguard, control_plate, neck, fretboard (with its dots), frets, nut,
tuners, tree, pickups (with the guard's screws), bridge, knobs, plate_parts (its screws and the jack's nut), buttons,
back_plates (the neck plate and the string ferrules), string_1 .. string_4 (one tube per string, numbered as players number
them: 1 the G, 4 the low E), plug, pick, decals (graphic layers), and with `stripes` the
racing stripes. The meshes are in the prop's frame (the pick in its own pinch frame). `bounds()` is the box of the whole
prop with its cord.
"""
import numpy as np

from ....core import shell as S
from . import electric_bass_body as B
from . import electric_bass_hw as H
from . import electric_bass_neck as N

GRAPHIC = ("decals", "stripes")                # layers lifted a fraction of a millimetre off a surface (tagged exempt)

_CACHE = {}


def parts(stripes=False):
    key = ("parts", bool(stripes))
    if key not in _CACHE:
        out = {
            "body": B.body_mesh(), "pickguard": B.guard_mesh(), "control_plate": B.plate_mesh(),
            "neck": N.neck_mesh(), "fretboard": N.board_mesh(), "frets": N.frets_mesh(), "nut": N.nut_mesh(),
            "tuners": N.tuners_mesh(), "tree": N.tree_mesh(),
            **{f"string_{4 - i}": m for i, m in enumerate(N.string_meshes())},
            "pickups": S.merge([H.pickups_mesh(), H.guard_screws_mesh()]), "bridge": H.bridge_mesh(),
            "knobs": H.knobs_mesh(), "plate_parts": H.plate_parts_mesh(), "buttons": H.buttons_mesh(),
            "back_plates": H.back_mesh(), "plug": H.plug_mesh(), "pick": H.pick_mesh(), "decals": H.decals_mesh(),
        }
        if stripes:
            out["stripes"] = B.stripes_mesh()
        _CACHE[key] = out
    return _CACHE[key]


def bounds():
    """(lo, hi) of the visible prop in its frame: every part (the pick at rest) and the default cord."""
    pts = []
    loc, R = H.pick_rest()
    for key, m in parts().items():
        V = m.V if key != "pick" else m.V @ R.T + loc
        pts.append(V)
    pts.append(H.cable_path())
    P = np.concatenate(pts)
    return P.min(0), P.max(0)
