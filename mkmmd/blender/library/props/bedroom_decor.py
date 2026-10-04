"""Bedroom decor props: `poster_80s`, `alarm_clock`, `bedroom_nightstand`, `wall_shelf`.

Conventions. Metres, Z up, floor z = 0, front = local -Y (the side a person faces), everything named `<name>_<part>`,
colours only from palette slots (blended in linear light; `paper` / `ink` = the brightest / darkest neutral slot of the
palette), built deterministically (random choices come from the instance name or `seed`). Builder options are
non-colour keys of `[[prop]] slots`. Hidden helpers are empties or objects flagged `mk_collider` (not rendered, not part
of the card's `size`). Procedural materials only, no image files; for EEVEE.

poster_80s - a printed paper poster on a wall. Frame: origin = the centre of the sheet on the wall plane (the wall is
y = 0), the sheet faces -Y, +Z up, +X right as seen from the room ("wall_center", as cafe_poster). The sheet is slightly
bowed paper whose mean plane stands ~3 mm off the wall (1.2 mm at a fixing, up to ~5 mm between). Options (slots):
  style    sunset_grid (default) | memphis | trio | car | sunburst
             sunset_grid  synthwave: striped sun behind faceted mountains with a neon ridge line, stars, a perspective
                          neon grid floor (foam / iris), chrome logotype blocks, keyline
             memphis      pale field with terrazzo specks and confetti, flat stickers with drop shadows and pattern
                          fills (zigzags, hex dots, stripes, checker, triangle), squiggle, rainbow arcs
             trio         a circle, a triangle and a square mixing light over concentric rings, sparkles, a chrome
                          logotype and rows of small print (abstract blocks: no readable text anywhere)
             car          a low wedge coupe in side view (dark body, lit roof edge, tinted glass, wheels, head and tail
                          light) on a road against banded sunset stripes and a pale sun, chrome title
             sunburst     banded sun on the horizon, rays, glints on the sea, two palm silhouettes (overlay)
  width, height  sheet size in metres (default 0.50 x 0.70; 0.60 x 0.90 and 0.40 x 0.30 work: the layout adapts to the
                 aspect)
  margin   white paper margin around the print, metres (default 0)
  mount    pins (default: four push-pins at the corners, `<name>_pin1..4`) | tape (two strips across the top corners,
           `<name>_tape1..2`, the free end curls out) | none
  folds    cross (default: one vertical and one horizontal crease, faint) | thirds (two letter folds) | none
  torn     corners torn off: "TR", "TL,BR", ["BL"] ... (default none; pins / tape are left out at a torn corner)
  variant  colourway: 0 as drawn, 1 warm <-> cool slots swapped, 2 rotated (a re-assignment of the accent slots)
  seed     changes the random layout (mountain ridge, ...);   accent  slot name of the pins / tape (default by name)
The print is shader-node maths (`bedroom_decor_prints`, compiled by `bedroom_decor_nodes`): resolution independent,
anti-aliased, with a faint warm paper cast, fibre grain and bump, softly worn creases, and a printed-look sheen: the ink
is a little glossier than the bare paper (no emission). The designs are mid-value with bright accents (checked on
rose-pine-moon: nothing near-black, readable at 2-3 m under a weak light).
  Objects: `<name>_print` (material `<name>_paper`), `<name>_pin1..4` (materials `_pin`, `_needle`) or
  `<name>_tape1..2` (`_tape`), empties `<name>_TL / TR / BR / BL` (the sheet's corners on its front face).
  Card: origin "wall_center", front "-Y"; use.anchor TL TR BR BL; use.surface `print` (centre = the sheet's centre on its
  front face, normal [0, -1, 0], up [0, 0, 1], size [width, height]); use.look `print`; no colliders; extra keys `style`
  and `printed` (size of the printed area).

alarm_clock - a bedside digital alarm clock, 0.16 x 0.07 x 0.07 m. Frame: origin on the table under the centre of the
footprint ("floor_center"), the display faces -Y (the face leans back 15 degrees). Options: `time` "HH:MM" or "H:MM"
(default "02:47"; a one-digit hour leaves the first digit blank), `led` slot of the lit segments (default love; gold
works), `light` watts of the small point light that spills the display's glow on the table (default 0.2, 0 = none).
Objects: `_shell` (moulded pale case: the side profile with fillets of 3 / 9 / 11 / 5.5 mm, both ends rolled over by 10 mm,
one cage from `bedroom_decor_soft.clock_shell`), `_plinth` (rolled slab, plan corners 14 mm), `_stripe`, `_feet` (domes),
`_plate` and `_glass` (display plate and
smoked pane) on the empty `_face`, `_seg_on` / `_seg_off` (lit segments are emissive meshes; unlit ones glow faintly),
`_colon`, `_alarm_led`, `_btn1/2` (gold, iris), `_grille`, `_slot` / `_knob` (alarm slider), `_col_body` (collider), light
`_light`. Custom properties on the root, all keyable: `glow` (gain of the display's emission and of the light, 0..4,
default 1), `colon` (0/1, the colon's lit state: key it to blink), `alarm` (0/1: alarm LED and the slider knob, driven).
Card: use.look `display` (a point 3.6 mm in front of the plate), use.surface `display` (centre, normal
[0, -0.964, 0.266], up [0, 0.266, 0.964], size [0.132, 0.042]); one box collider (rnd 4 mm); extra keys `time` and
`properties`.

bedroom_nightstand - a small Memphis bedside cabinet, 0.40 x 0.35 x 0.55 m, origin on the floor under the centre
("floor_center"), the drawer faces -Y. Moulded, not boxy (`form` score 0.02): every slab is a `bedroom_decor_soft.soft_slab`,
a plan outline with large corner radii and both rims rolled over. Parts: `_top` (terrazzo laminate, 25 mm, plan corners
45 mm, bullnose teal edge), `_body` (drawer carcass z 0.405-0.525, plan corners 50 mm, rims rolled 26 / 22 mm), `_reveal`,
`_drawer` (a crowned pad, 4 mm crown, rolled rim; pink front with an iris triangle, gold zigzags and a foam dot patch: a
procedural
pattern in object metres), `_pull` (gold tube), `_legs` (four splayed teal tubes), `_feet` (metal caps), `_shelf` (iris
board at z 0.20), colliders `_col_top` (the slab) and `_col_body` (the drawer unit). The drawer is fixed (closed).
Card: use.rest `top` (plane, centre [0, 0, 0.55], normal +Z, size [0.40, 0.35]) and `shelf` (plane at z 0.209, size
[0.304, 0.253]); use.look `top`; 2 box colliders (rnd 4 mm). No options. The pull stands 7 mm proud of the slab's front
(visible size 0.40 x 0.357 x 0.55).

wall_shelf - a floating wall shelf with things on it. Frame: origin = the centre of the board's top-back edge on the wall
plane ("wall_center", the wall is y = 0): the board is x -0.40..0.40, y -0.20..0, z -0.03..0 (its top is z = 0), front
-Y, +Z up. Option `seed` (default 0) varies the books. Parts: `_board` (pale lacquer, teal front edge, no visible
brackets), `_books` (nine real books in one mesh at the LEFT end, x -0.37..-0.06, 15-27 cm tall: a round spine, boards
a little larger than the page block, a bowed fore-edge, bands and a label printed on the curved spine, a few pulled out
by up to 7 mm, the last one leaning on its neighbour), `_tapes` (four compact cassettes 100 x 63.5 x 12.5 mm lying flat
in a stack, rounded shells, coloured printed spines, the top one with label, window and hubs; x about -0.04..0.07, 5 cm
tall), `_ornament` (a Memphis totem: rounded cylinder, ball, soft-tipped cone, at x 0.345, 12 cm tall),
colliders `_col_board`, `_col_books`. Card: use.rest `top` (the whole board: centre [0, -0.10, 0], size [0.80, 0.20]) and
`free` (the stretch NOT taken by the books, tapes and totem, x 0.10..0.30: centre [0.20, -0.10, 0], size [0.20, 0.17]:
put the clock or another small prop there); use.look `top`; 2 box colliders; extra key `zones` (x ranges of books, tapes,
ornament, free). The books stand with their backs 12 mm in front of the wall."""
import math
import zlib

import bmesh

from . import bedroom_decor_clock as clock
from . import bedroom_decor_furniture as furniture
from . import register
from .bedroom_decor_field import Palette
from .bedroom_decor_layout import corner_points, pin_points, sheet_gap, sheet_grid, tape_specs, tape_strip
from .bedroom_decor_nodes import poster_material
from .bedroom_decor_prints import STYLES, build, finish
from .cafe_kit import Kit, bm_lathe, principled

ACCENTS = ("love", "gold", "foam", "iris", "rose", "pine")
MOUNTS = ("pins", "tape", "none")
FOLDS = ("none", "cross", "thirds")


def _num(slots, key, default):
    try:
        return float(slots.get(key, default))
    except (TypeError, ValueError):
        return float(default)


def _choice(slots, key, default, allowed):
    v = str(slots.get(key, default))
    if v not in allowed:
        raise ValueError(f"{key} {v!r}: one of {allowed}")
    return v


def _corners(v):
    """'TL,BR' | ['TL'] | None -> a tuple of corner tags."""
    if not v:
        return ()
    items = v.replace(",", " ").split() if isinstance(v, str) else list(v)
    return tuple(t.upper() for t in items if t.upper() in ("TL", "TR", "BL", "BR"))


def _pin_mesh():
    """A push-pin along +Z: the dome head (material 0) from z = 0 up, the needle (material 1) below z = 0."""
    bm = bmesh.new()
    head = [(0.0, 0.0), (0.0050, 0.0), (0.0053, 0.0007), (0.0048, 0.0021), (0.0037, 0.0033), (0.0021, 0.0040),
            (0.0, 0.0043)]
    bm_lathe(bm, head, segs=24, mat=0)
    bm_lathe(bm, [(0.0, -0.0035), (0.0005, -0.0030), (0.0005, 0.0), (0.0, 0.0)], segs=8, mat=1)
    return bm


def _tape_mat(K, colour):
    m, nt, out = K.new_mat("tape")
    principled(nt, out, **{"Base Color": colour, "Roughness": 0.55, "Specular IOR Level": 0.4, "Alpha": 0.82})
    return m


@register("poster_80s")
def poster_80s(name, coll, root, slots=None):
    """A printed paper poster on the wall in one of five designs (see the module docstring)."""
    K = Kit(name, coll, root, slots)
    opts = K.slots
    style = str(opts.get("style", "sunset_grid"))
    if style not in STYLES:
        raise KeyError(f"poster_80s {name}: no style {style!r} (have {sorted(STYLES)})")
    w, h = _num(opts, "width", 0.50), _num(opts, "height", 0.70)
    margin = min(max(_num(opts, "margin", 0.0), 0.0), 0.2 * min(w, h))
    mount = _choice(opts, "mount", "pins", MOUNTS)
    folds = _choice(opts, "folds", "cross", FOLDS)
    torn = _corners(opts.get("torn"))
    variant, seed = int(_num(opts, "variant", 0)), int(_num(opts, "seed", 0))
    crc = zlib.crc32(name.encode())
    accent = str(opts.get("accent") or ACCENTS[crc % len(ACCENTS)])
    ph = h - 2.0 * margin
    aspect = (w - 2.0 * margin) / ph
    pal = Palette(colors=K.colors)
    design = build(style, aspect, seed, variant)
    fin = finish(design, aspect, w / ph, h / ph, folds, torn)
    mat = poster_material(K, "paper", pal, fin, w, h, margin)

    # the sheet: a grid following the bowed paper, UV 0..1 over the whole sheet
    nx, ny = sheet_grid(w, h)
    bm = bmesh.new()
    uvl = bm.loops.layers.uv.verify()
    vs = [[bm.verts.new((-w / 2 + w * i / nx, -sheet_gap(-w / 2 + w * i / nx, -h / 2 + h * j / ny, w, h, mount),
                         -h / 2 + h * j / ny)) for i in range(nx + 1)] for j in range(ny + 1)]
    for j in range(ny):
        for i in range(nx):
            f = bm.faces.new((vs[j][i], vs[j][i + 1], vs[j + 1][i + 1], vs[j + 1][i]))
            f.smooth = True
            for lp, uv in zip(f.loops, ((i, j), (i + 1, j), (i + 1, j + 1), (i, j + 1))):
                lp[uvl].uv = (uv[0] / nx, uv[1] / ny)
    K.to_obj("print", bm, [mat])

    corners = corner_points(w, h, mount)
    for tag, p in corners.items():
        K.empty(tag, loc=p, size=0.012, hidden=True)

    if mount == "pins":
        pin_m = K.simple_mat("pin", K.slot(accent), rough=0.22, coat=0.6, spec=0.6)
        needle = K.simple_mat("needle", K.blend(subtle=.7, text=.3), rough=0.3, metal=1.0)
        for i, (x, z) in enumerate(pin_points(w, h, torn, mount), 1):
            K.to_obj(f"pin{i}", _pin_mesh(), [pin_m, needle], loc=(x, -sheet_gap(x, z, w, h, mount), z),
                     rot=(math.pi / 2, 0.0, 0.0))
    elif mount == "tape":
        tape_m = _tape_mat(K, pal.blend(paper=.55, **{accent: .45}))
        for i, (x0, ang) in enumerate(tape_specs(w, h, crc, torn), 1):
            verts, quads, uvs, org = tape_strip(x0, ang, w, h, mount)
            tb = bmesh.new()
            tuv = tb.loops.layers.uv.verify()
            tv = [tb.verts.new(v) for v in verts]
            for q in quads:
                f = tb.faces.new([tv[k] for k in q])
                f.smooth = True
                for lp, k in zip(f.loops, q):
                    lp[tuv].uv = uvs[k]
            K.to_obj(f"tape{i}", tb, [tape_m], loc=org).visible_shadow = False

    cy = -sheet_gap(0.0, 0.0, w, h, mount)
    use = {"anchor": [{"name": tag, "point": [round(v, 4) for v in p]} for tag, p in corners.items()],
           "surface": [{"name": "print", "center": [0.0, round(cy, 4), 0.0], "normal": [0.0, -1.0, 0.0],
                        "up": [0.0, 0.0, 1.0], "size": [w, h]}],
           "look": [{"name": "print", "point": [0.0, round(cy, 4), 0.0]}]}
    return K.card(use=use, origin="wall_center", front="-Y", style=style, printed=[round(aspect * ph, 4), round(ph, 4)])


@register("alarm_clock")
def alarm_clock(name, coll, root, slots=None):
    """A bedside digital alarm clock with a seven-segment display (see the module docstring)."""
    return clock.build(name, coll, root, slots)


@register("bedroom_nightstand")
def bedroom_nightstand(name, coll, root, slots=None):
    """A small Memphis bedside cabinet with a drawer and a shelf (see the module docstring)."""
    return furniture.build_nightstand(name, coll, root, slots)


@register("wall_shelf")
def wall_shelf(name, coll, root, slots=None):
    """A floating wall shelf with books, cassettes and a small ornament (see the module docstring)."""
    return furniture.build_wall_shelf(name, coll, root, slots)
