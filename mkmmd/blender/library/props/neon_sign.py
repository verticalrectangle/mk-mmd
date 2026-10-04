"""neon_sign: a roadside motel / diner sign: a tall pole carrying a sign board ringed by a neon tube, and a smaller board
under it. Both faces are card `use.surface` entries, so the words are [[text]] on them (`on = "motel:face"`, glowing,
`flicker`, `blink` for the small one).

    [[prop]]
    name = "motel"
    card = "library:neon_sign"
    at = { path = "road:road", s = 360.0, offset = -17.0 }
    slots = { neon = "love", board = "base", height = 6.5 }

Options (the `slots`; colours are palette slots or hex):
  board    colour of the boards ("base")     neon   colour of the tubes ("love")     pole   colour of the pole ("muted")
  glow     emission of the tubes (7)          width / tall   the main board, m (3.2 / 1.5)
  height   of the main board's bottom above the ground (6.5)   sub   [width, height] of the small board ([2.0, 0.55];
           [] for none)                       sub_neon   colour of its tube ("foam")
The sign faces -Y (its front is the side a car coming along +Y sees: turn it with `yaw`); the origin is the pole's foot.

Card: size, bounds, front "-Y", use.surface [face, sub], form_max 0.7 (a sign board is a board)."""
import bpy

from ....core import shell as CS
from .. import shell as SH
from . import register
from .hearts import _colour
from .traffic_car import _material, slab


def _board(w, h, depth=0.28):
    """A sign board: a slab (w x h, `depth` thick) with rounded corners and edges, centred on the origin, facing -Y."""
    return slab(w, h, -depth / 2, depth / 2, corner=min(w, h) * 0.12, edge=min(0.07, depth * 0.3))


def _tube(w, h, inset=0.12, r=0.035):
    """The neon tube round a board's face, `inset` m in from its edge, just in front of it (y < 0)."""
    sec = CS.rrect(h - 2 * inset, w - 2 * inset, min(w, h) * 0.1, n=6)
    path = [(v, -0.16, u) for u, v in sec]
    return CS.tube(path + path[:1], r, sides=8, closed=True)


@register("neon_sign")
def neon_sign(name, coll, root, slots=None):
    s = dict(slots or {})
    w, tall, height = float(s.get("width", 3.2)), float(s.get("tall", 1.5)), float(s.get("height", 6.5))
    sub = list(s.get("sub", [2.0, 0.55]))
    col = {k: _colour(s.get(k, d), s) for k, d in (("board", "base"), ("neon", "love"), ("pole", "muted"),
                                                    ("sub_neon", "foam"))}
    glow = float(s.get("glow", 7.0))
    board_m = _material(f"{name}_board", col["board"], 0.6)
    pole_m = _material(f"{name}_pole", col["pole"], 0.45, metal=0.6)
    neon_m = _material(f"{name}_neon", col["neon"], 0.2, emit=glow)
    sub_m = _material(f"{name}_neon_sub", col["sub_neon"], 0.2, emit=glow * 0.8)
    zc = height + tall / 2
    pole = CS.lathe([(0.0, 0.0), (0.16, 0.0), (0.16, 0.05), (0.11, 0.12), (0.11, zc + tall / 2), (0.0, zc + tall / 2)], seg=16)
    SH.mesh_object(f"{name}_pole", pole, coll, root, lambda _r: pole_m, loc=(0.0, 0.18, 0.0))
    SH.mesh_object(f"{name}_board", _board(w, tall), coll, root, lambda _r: board_m, loc=(0.0, 0.0, zc))
    neon = SH.mesh_object(f"{name}_tube", _tube(w, tall), coll, root, lambda _r: neon_m, loc=(0.0, 0.0, zc))
    SH.exempt(neon, "a neon tube on the board's face (a light, not a form)")
    surf = [{"name": "face", "center": [0.0, -0.141, zc], "normal": [0.0, -1.0, 0.0], "up": [0.0, 0.0, 1.0],
             "size": [w - 0.5, tall - 0.4]}]
    top = zc + tall / 2
    if sub:
        sw, sh = float(sub[0]), float(sub[1])
        zs = height - 0.25 - sh / 2
        SH.mesh_object(f"{name}_sub", _board(sw, sh, 0.22), coll, root, lambda _r: board_m, loc=(0.0, 0.0, zs))
        t = SH.mesh_object(f"{name}_sub_tube", _tube(sw, sh, 0.07, 0.025), coll, root, lambda _r: sub_m, loc=(0.0, 0.03, zs))
        SH.exempt(t, "a neon tube on the board's face (a light, not a form)")
        surf.append({"name": "sub", "center": [0.0, -0.111, zs], "normal": [0.0, -1.0, 0.0], "up": [0.0, 0.0, 1.0],
                     "size": [sw - 0.3, sh - 0.16]})
    lo, hi = [-w / 2, -0.3, 0.0], [w / 2, 0.35, top]
    bpy.context.view_layer.update()
    return {"size": [round(b - a, 3) for a, b in zip(lo, hi)], "origin": "floor_center", "front": "-Y", "blocks": True,
            "bounds": {"min": lo, "max": hi}, "use": {"surface": surf}, "form_max": 0.7,
            "slots": {k: s.get(k, d) for k, d in (("board", "base"), ("neon", "love"))}}
