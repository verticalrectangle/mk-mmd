"""Every mesh of the offset guitar, by object suffix (pure numpy, no bpy: the Blender builder only turns them into objects).

`parts()` -> {suffix: core.shell.Mesh}: body, pickguard, neck, fretboard (with its dots), frets, nut, tuners, truss (the
electric guitar's neck), pickups (with the guard's screws), bridge, tremolo (the plate with its string anchor, lock button
and collet), knobs, switch (the toggle, the slide switch and the thumb wheels), jack, buttons, back_plates,
string_1 .. string_6 (one tube per string, numbered as players number them: 1 the high e, 6 the low E), plug, pick, decals
(graphic layers). The meshes are in the prop's frame (the pick in its own pinch frame). `arm(arm)` is the tremolo arm in
its own frame (`layout.arm_geometry` stands it). `bounds(arm)` is the box of the whole prop with its cord.
"""
import numpy as np

from ....core import shell as S
from . import electric_guitar_hw as EH
from . import electric_guitar_neck as N
from . import offset_guitar_body as B
from . import offset_guitar_hw as H
from . import offset_guitar_layout as L

GRAPHIC = ("decals",)                          # layers lifted a fraction of a millimetre off a surface (tagged exempt)

_CACHE = {}


def parts():
    if "parts" not in _CACHE:
        _CACHE["parts"] = {
            "body": B.body_mesh(), "pickguard": B.guard_mesh(),
            "neck": N.neck_mesh(), "fretboard": N.board_mesh(), "frets": N.frets_mesh(), "nut": N.nut_mesh(),
            "tuners": N.tuners_mesh(), "truss": N.truss_mesh(),
            **{f"string_{6 - i}": m for i, m in enumerate(H.string_meshes())},
            "pickups": S.merge([H.pickups_mesh(), H.guard_screws_mesh()]), "bridge": H.bridge_mesh(),
            "tremolo": H.tremolo_mesh(), "knobs": H.knobs_mesh(), "switch": H.switch_mesh(), "jack": H.jack_mesh(),
            "buttons": H.buttons_mesh(), "back_plates": H.back_mesh(), "plug": H.plug_mesh(), "pick": EH.pick_mesh(L.PICK),
            "decals": H.decals_mesh(),
        }
    return _CACHE["parts"]


def arm(arm=None):
    """The tremolo arm's mesh in its own frame, for `layout.ARM`'s keys (default `ARM`)."""
    key = ("arm", tuple(sorted(dict(L.ARM, **(arm or {})).items())))
    if key not in _CACHE:
        _CACHE[key] = H.arm_mesh(arm)
    return _CACHE[key]


def bounds(arm_spec=None, has_arm=True):
    """(lo, hi) of the visible prop in its frame: every part (the pick and the arm at rest) and the default cord."""
    loc, R = H.pick_rest()
    pts = [m.V if key != "pick" else m.V @ R.T + loc for key, m in parts().items()]
    if has_arm:
        bend, _tip, Ra = L.arm_geometry(arm_spec)
        pts.append(arm(arm_spec).V @ Ra.T + bend)
    pts.append(H.cable_path())
    P = np.concatenate(pts)
    return P.min(0), P.max(0)
