"""Every mesh of the electric guitar, by object suffix (pure numpy, no bpy: the Blender builder only turns them into objects).

`parts()` -> {suffix: core.shell.Mesh}: body, pickguard, neck, fretboard (with its dots), frets, nut, tuners, truss, pickups
(with the guard's screws), bridge, knobs, switch, jack_plate, buttons, back_plates, strings, plug, pick, decals (graphic layers). The meshes
are in the prop's frame (the pick in its own pinch frame). `bounds()` is the box of the whole prop with its cord.
"""
import numpy as np

from ....core import shell as S
from . import electric_guitar_body as B
from . import electric_guitar_hw as H
from . import electric_guitar_neck as N

GRAPHIC = ("decals",)                          # layers lifted a fraction of a millimetre off a surface (tagged exempt)

_CACHE = {}


def parts():
    if "parts" not in _CACHE:
        _CACHE["parts"] = {
            "body": B.body_mesh(), "pickguard": B.guard_mesh(),
            "neck": N.neck_mesh(), "fretboard": N.board_mesh(), "frets": N.frets_mesh(), "nut": N.nut_mesh(),
            "tuners": N.tuners_mesh(), "truss": N.truss_mesh(), "strings": N.strings_mesh(),
            "pickups": S.merge([H.pickups_mesh(), H.guard_screws_mesh()]), "bridge": H.bridge_mesh(),
            "knobs": H.knobs_mesh(), "switch": H.switch_mesh(), "jack_plate": H.jack_plate_mesh(),
            "buttons": H.buttons_mesh(), "back_plates": H.back_mesh(), "plug": H.plug_mesh(), "pick": H.pick_mesh(), "decals": H.decals_mesh(),
        }
    return _CACHE["parts"]


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
