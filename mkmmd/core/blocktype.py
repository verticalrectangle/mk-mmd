"""Toy alphabet blocks (docs/design.md: Text, Block type): the bpy-free half of `mkmmd/blender/build/blocktype.py`.

A `[[text]]` with `lyrics = {style = "blocks", ...}` (the sung words) or with `text` and `blocks = {...}` (a label) is set
in toy blocks: one wooden cube per letter or digit, its edges rounded, its faces painted in palette colours, the letter
raised (or printed) on the front and other letters or little pictures on the other five faces. This module decides all
that is not Blender: which blocks there are and what each face shows, where they stand (rows that build up from the bottom,
as real blocks do), how big they are on the panel, and how each one moves, sampled on every frame:

  fall     dropped from above (just out of the picture, or `fall.height` edges up) so that it lands ON its word's onset,
           tilted as it falls and straight as it lands, then bouncing (restitution `bounce`) and squashing on each impact;
           the letters of a word land `stagger` s apart, the first on the onset
  hits     on the hits of a lane of the [audio] hits file: `jump`, a hop off the floor rippling along the rows, or `slam`,
           lifted a moment before the hit and slammed down on it
  next     when the next lyric line comes the earlier lines `stack` (they stay and the new rows build on top of them) or
           `tumble` away, knocked off from where the new line's first block comes down
  crash    at a clip second every block still up tumbles away

The motion is ballistics in units of the block's edge (`g` edges/s^2: by default a 5 cm wooden block under 9.81 m/s^2).
The geometry of a block (a rounded cube, raised or printed glyphs, a rim round each lettered face, the pictures) is made
here too, in the block's own frame (edge 1, centred); Blender hands over each glyph's filled outline (String to Curves and
Fill Curve, at em size 1) and builds meshes, materials and keys.

The letters live only in `Unit.letters`, `Block.mesh` and `MeshSpec.faces` (all kept out of every repr) and in the geometry
made from them: messages, object tags and the summary carry (line, word) numbers, letter places and counts."""
import math
from dataclasses import dataclass, field

import numpy as np

from . import lanes as LN
from . import tapefx as TF
from . import wordtype as LY
from .typeset import TextError, parse_align

ENTRY_KEYS = {"name", "on", "mount", "at", "facing", "up", "box", "back", "lift", "size", "fit", "align", "valign", "offset",
              "font", "glow", "lit", "screen", "aspect", "text", "lyrics", "blocks", "_aspect", "_multi"}
BLOCK_KEYS = {"paint", "ink", "wood", "ground", "letter", "relief", "round", "rim", "sides", "pictures", "layout", "rows",
              "gap", "space", "view", "fall", "hits", "next", "crash", "tumble", "land", "to"}
LYRIC_KEYS = {"timeline", "line", "lines", "words", "case", "clean", "style", "from", "to"}
PAINT = ("love", "gold", "foam", "iris", "rose", "pine")
INK, WOOD, GROUND = "surface", "gold:base:0.6", "surface"
LETTERS = ("raised", "printed")
SIDES = ("mixed", "letters", "pictures", "plain")
LAYOUTS = ("flow", "stack")
NEXTS = ("stack", "tumble")
MOVES = ("jump", "slam")
G_EDGE = 9.81 / 0.05                    # edges/s^2: a 5 cm block under the earth's gravity
FALL = {"height": "top", "g": G_EDGE, "bounce": 0.35, "stagger": 0.03, "spin": 25.0, "squash": 0.12}
JUMP = {"height": 0.6, "wave": 0.012, "turn": 0.0}
SLAM = {"height": 0.35, "wave": 0.0, "lead": 0.09, "squash": 0.28}
HIT_KEYS = {"lane", "min_db", "every", "from", "to", "move"}
TUMBLE = {"kick": 5.0, "pop": 7.0, "spin": 420.0, "gap": 0.02, "life": 1.2, "floor": None}
VIEW_SCREEN, VIEW_WORLD = (16.0, 12.0), (0.0, 0.0)
SQUASH_TAU, SQUASH_PERIOD = 0.05, 0.18  # s: the damped spring of a squash
REF_DROP = 4.0                          # edges: a fall from this high squashes by the whole `squash`
MIN_HOP = 0.012                         # edges: a bounce lower than this is the block coming to rest
REACH = 0.87                            # edges from a block's centre to its farthest corner (half the diagonal, rounded up)
SLAM_PEAK = 0.6                         # share of a slam's lead spent rising; the rest is the plunge
SLAM_BOUNCE = 0.5                       # a slammed block bounces back with this share of the fall's restitution
TUMBLE_DT = 1.0 / 240.0                 # s: the step of a tumble that lands on the floor
FLOOR_KEEP = 0.6                        # share of its speed along the floor and of its spin a block keeps at each bounce
SLIDE_TAU = 0.15                        # s: the time constant its slide and spin die away with while it lies on the floor
SETTLE = 0.15                           # s a nearly stopped block takes to lie flat on its nearest face
TOL = 1e-6                              # keys closer than this to the straight line between their neighbours are dropped
PRINT_LIFT = 0.0012                     # edges a printed glyph or picture sits above its face
SINK = 0.003                            # edges a raised glyph's walls reach into the face
RIM_INSET, RIM_WIDTH = 0.035, 0.045     # edges: the rim round a lettered face, in from the painted square's edge
ARC_STEPS = 3                           # grid steps over the 45 degrees of a rounded edge each face carries


class BlockError(ValueError):
    """A block entry that cannot be built. Messages carry numbers and key names, never lyric text."""


# ------------------------------------------------------------------------------------------------------ pictures
def _star(cx, cy, ro, ri, n, deg=90.0):
    """Points of a star with `n` points (outer radius `ro`, inner `ri`), the first point at `deg`."""
    out = []
    for i in range(2 * n):
        a = math.radians(deg) + math.pi * i / n
        r = ro if i % 2 == 0 else ri
        out.append((cx + r * math.cos(a), cy + r * math.sin(a)))
    return out


def _arch(cx, cy, r, n=14):
    """A half disc standing on its flat side (a mushroom's cap)."""
    return [(cx + r * math.cos(math.pi * i / n), cy + r * math.sin(math.pi * i / n)) for i in range(n + 1)]


# Little pictures for the faces that show no letter: parts in a [-0.5, 0.5] square (x right, y up) drawn in order, each a
# palette colour and a shape, ("disc", cx, cy, rx, ry, degrees) or ("poly", [(x, y), ...]) star-shaped about its centroid.
PICTURES = {
    "carrot": [("pine", "disc", -0.11, 0.30, 0.055, 0.15, 25.0), ("pine", "disc", 0.0, 0.33, 0.055, 0.17, 0.0),
               ("pine", "disc", 0.11, 0.30, 0.055, 0.15, -25.0),
               ("gold", "poly", [(-0.17, 0.20), (0.17, 0.20), (0.05, -0.36), (0.0, -0.44), (-0.05, -0.36)])],
    "tomato": [("love", "disc", 0.0, -0.06, 0.37, 0.32, 0.0), ("pine", "poly", _star(0.0, 0.20, 0.17, 0.06, 5)),
               ("pine", "disc", 0.0, 0.30, 0.025, 0.07, 0.0)],
    "peapod": [("foam", "disc", 0.0, 0.0, 0.44, 0.17, 18.0)]
              + [("pine", "disc", t * math.cos(math.radians(18.0)), t * math.sin(math.radians(18.0)), 0.085, 0.085, 0.0)
                 for t in (-0.22, 0.0, 0.22)],
    "eggplant": [("iris", "disc", 0.04, -0.08, 0.21, 0.33, 28.0), ("pine", "poly", _star(-0.115, 0.21, 0.13, 0.05, 4, 45.0)),
                 ("pine", "disc", -0.16, 0.31, 0.025, 0.08, 30.0)],
    "radish": [("pine", "disc", -0.08, 0.30, 0.05, 0.14, 25.0), ("pine", "disc", 0.08, 0.30, 0.05, 0.14, -25.0),
               ("love", "disc", 0.0, -0.02, 0.27, 0.26, 0.0), ("rose", "poly", [(-0.06, -0.24), (0.06, -0.24), (0.0, -0.46)])],
    "broccoli": [("foam", "poly", [(-0.08, -0.42), (0.08, -0.42), (0.12, -0.02), (-0.12, -0.02)]),
                 ("pine", "disc", -0.17, 0.08, 0.17, 0.16, 0.0), ("pine", "disc", 0.17, 0.08, 0.17, 0.16, 0.0),
                 ("pine", "disc", 0.0, 0.20, 0.20, 0.18, 0.0)],
    "corn": [("gold", "disc", 0.0, 0.04, 0.16, 0.38, 0.0), ("pine", "poly", [(-0.02, -0.42), (-0.30, -0.05), (-0.14, -0.18)]),
             ("pine", "poly", [(0.02, -0.42), (0.30, -0.05), (0.14, -0.18)])],
    "mushroom": [("subtle", "poly", [(-0.10, -0.40), (0.10, -0.40), (0.12, 0.02), (-0.12, 0.02)]),
                 ("love", "poly", _arch(0.0, -0.02, 0.36)), ("surface", "disc", -0.15, 0.13, 0.05, 0.05, 0.0),
                 ("surface", "disc", 0.08, 0.21, 0.05, 0.05, 0.0), ("surface", "disc", 0.19, 0.06, 0.04, 0.04, 0.0)],
    "apple": [("love", "disc", 0.0, -0.05, 0.34, 0.33, 0.0), ("pine", "disc", 0.12, 0.33, 0.12, 0.05, 30.0),
              ("subtle", "disc", 0.0, 0.30, 0.025, 0.09, -10.0)],
    "sun": [("gold", "poly", _star(0.0, 0.0, 0.45, 0.28, 10))],
    "star": [("gold", "poly", _star(0.0, -0.02, 0.45, 0.19, 5))],
    "heart": [("love", "disc", -0.13, 0.08, 0.17, 0.17, 0.0), ("love", "disc", 0.13, 0.08, 0.17, 0.17, 0.0),
              ("love", "poly", [(-0.29, 0.03), (0.29, 0.03), (0.0, -0.38)])],
}

# The faces of a block: normal, then u (right) and v (up) as the face reads from outside, upright when the block is
# turned so that face looks at the viewer by the shortest turn.
FACES = ("front", "back", "top", "bottom", "right", "left")
FRAMES = {"front": ((0, 0, 1), (1, 0, 0), (0, 1, 0)), "back": ((0, 0, -1), (-1, 0, 0), (0, 1, 0)),
          "top": ((0, 1, 0), (1, 0, 0), (0, 0, -1)), "bottom": ((0, -1, 0), (1, 0, 0), (0, 0, 1)),
          "right": ((1, 0, 0), (0, 0, -1), (0, 1, 0)), "left": ((-1, 0, 0), (0, 0, 1), (0, 1, 0))}
PAINT_STEP = {"front": 0, "right": 1, "back": 2, "bottom": 3, "top": 4, "left": 5}    # paint list offset of each face
SLOT_WOOD, SLOT_INK, SLOT_EXTRA = 0, 7, 8                                             # 1..6: the faces' paint, FACES order
WHAT = {"mixed": {"back": "case", "top": "picture", "bottom": "digit", "right": "next", "left": "picture"},
        "letters": {"back": "case", "top": "next", "bottom": "digit", "right": "next2", "left": "previous"},
        "pictures": {f: "picture" for f in FACES[1:]},
        "plain": {}}


# ------------------------------------------------------------------------------------------------------ options
def is_blocks(entry):
    """Whether a [[text]] entry is set in blocks: it has `blocks` (a table or true) or `lyrics.style = "blocks"`."""
    lyr = entry.get("lyrics")
    return entry.get("blocks") not in (None, False) or (isinstance(lyr, dict) and lyr.get("style") == "blocks")


def _number(v, what, lo=None, hi=None, lo_open=False):
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(float(v)):
        raise BlockError(f"{what} must be a number")
    v = float(v)
    if (lo is not None and (v < lo or (lo_open and v <= lo))) or (hi is not None and v > hi):
        rng = f"{'(' if lo_open else '['}{lo if lo is not None else '-inf'}, {hi if hi is not None else 'inf'}]"
        raise BlockError(f"{what} = {v:g} is outside {rng}")
    return v


def _choice(tab, key, choices, what):
    v = tab.get(key, choices[0])
    if v not in choices:
        raise BlockError(f"{what}{key} is one of {list(choices)}")
    return v


def _colour(v, what):
    if not isinstance(v, str) or not v:
        raise BlockError(f"{what} is a palette slot, a #hex colour or a mix 'slot:slot:t'")
    return v


def _table(tab, key, defaults, what):
    """tab[key] checked against `defaults` and filled with them (`true` or absent: all defaults)."""
    v = tab.get(key, {})
    if v is True:
        v = {}
    if not isinstance(v, dict):
        raise BlockError(f"{what}{key} is a table ({', '.join(sorted(defaults))})")
    unknown = sorted(set(v) - set(defaults))
    if unknown:
        raise BlockError(f"{what}{key}: unknown keys {unknown} (known: {sorted(defaults)})")
    return dict(defaults, **v)


def _hit_options(h, i, what):
    if not isinstance(h, dict):
        raise BlockError(f"{what}hits[{i}] is a table {{lane, move, ...}}")
    move = h.get("move", "jump")
    if move not in MOVES:
        raise BlockError(f"{what}hits[{i}]: move is one of {list(MOVES)}")
    own = JUMP if move == "jump" else SLAM
    unknown = sorted(set(h) - HIT_KEYS - set(own))
    if unknown:
        raise BlockError(f"{what}hits[{i}]: unknown keys {unknown} (a {move} takes {sorted(HIT_KEYS | set(own))})")
    try:
        c = LN.choice(h, f"{what}hits[{i}]")
    except LN.LaneError as e:
        raise BlockError(str(e)) from None
    o = dict(own, **{k: h[k] for k in own if k in h})
    w = f"{what}hits[{i}]."
    o["height"] = _number(o["height"], w + "height", 0.0, 20.0, lo_open=True)
    o["wave"] = _number(o["wave"], w + "wave", 0.0, 1.0)
    if move == "jump":
        o["turn"] = _number(o["turn"], w + "turn", -1080.0, 1080.0)
        if abs(o["turn"] / 360.0 - round(o["turn"] / 360.0)) > 1e-9:
            raise BlockError(f"{w}turn is whole turns (a multiple of 360 degrees): the block lands on its floor face again")
    else:
        o["lead"] = _number(o["lead"], w + "lead", 0.0, 1.0, lo_open=True)
        o["squash"] = _number(o["squash"], w + "squash", 0.0, 0.6)
    o.update(move=move, lane=c["lane"], min_db=c["min_db"], every=c["every"], t0=c["t0"], t1=c["t1"])
    return o


def options(entry, screen=False):
    """The checked block options of an entry (every default filled in). `screen`: the entry is type of the picture (its
    default `view` turns the blocks to show their tops and sides)."""
    name = entry.get("name")
    if not name:
        raise BlockError("a [[text]] set in blocks needs a name")
    where, what = f"text {name!r}", "blocks."
    try:
        unknown = sorted(set(entry) - ENTRY_KEYS)
        if unknown:
            raise BlockError(f"{unknown} do not apply to blocks (an entry set in blocks takes "
                             f"{sorted(k for k in ENTRY_KEYS if not k.startswith('_'))}; the colours are blocks.paint, "
                             "blocks.ink, blocks.wood and blocks.ground)")
        lyr = entry.get("lyrics")
        if lyr is not None:
            if not isinstance(lyr, dict):
                raise BlockError("lyrics must be a table")
            if "text" in entry:
                raise BlockError("blocks set the sung words (`lyrics`) or a label (`text`), not both")
            if lyr.get("style", "blocks") != "blocks":
                raise BlockError(f"`blocks` sets the words in toy blocks: lyrics style is 'blocks' (not {lyr['style']!r})")
            bad = sorted(set(lyr) - LYRIC_KEYS)
            if bad:
                raise BlockError(f"lyrics keys {bad} do not apply to style 'blocks' (it takes {sorted(LYRIC_KEYS)}; the "
                                 "look, the layout and the motion are in `blocks`)")
        elif "text" not in entry:
            raise BlockError("blocks set a label (`text`) or the sung words (`lyrics = {style = \"blocks\", ...}`)")
        b = entry.get("blocks", {})
        b = {} if b in (True, None) else b
        if not isinstance(b, dict):
            raise BlockError("blocks is true or a table")
        unknown = sorted(set(b) - BLOCK_KEYS)
        if unknown:
            raise BlockError(f"blocks: unknown keys {unknown} (known: {sorted(BLOCK_KEYS)})")
        o = {}
        paint = b.get("paint", list(PAINT))
        paint = [paint] if isinstance(paint, str) else paint
        if not isinstance(paint, (list, tuple)) or not paint:
            raise BlockError("blocks.paint is a colour or a list of them, cycled over the blocks")
        o["paint"] = [_colour(c, "blocks.paint") for c in paint]
        for key, d in (("ink", INK), ("wood", WOOD), ("ground", GROUND)):
            o[key] = _colour(b.get(key, d), f"blocks.{key}")
        o["letter"] = _choice(b, "letter", LETTERS, what)
        o["relief"] = _number(b.get("relief", 0.05), "blocks.relief", 0.0, 0.2, lo_open=True)
        o["round"] = _number(b.get("round", 0.12), "blocks.round", 0.02, 0.3)
        rim = b.get("rim", True)
        if not isinstance(rim, bool):
            raise BlockError("blocks.rim is true or false")
        o["rim"] = rim
        o["sides"] = _choice(b, "sides", SIDES, what)
        pics = b.get("pictures", list(PICTURES))
        pics = [pics] if isinstance(pics, str) else pics
        if not isinstance(pics, (list, tuple)) or not pics or any(p not in PICTURES for p in pics):
            raise BlockError(f"blocks.pictures is a list of {sorted(PICTURES)}")
        o["pictures"] = list(pics)
        o["layout"] = _choice(b, "layout", LAYOUTS, what)
        rows = b.get("rows", "auto")
        if rows != "auto" and (isinstance(rows, bool) or not isinstance(rows, int) or rows < 1):
            raise BlockError("blocks.rows is \"auto\" or a whole number of rows per line, at least 1")
        o["rows"] = rows
        o["gap"] = _number(b.get("gap", 0.06), "blocks.gap", 0.0, 2.0)
        o["space"] = _number(b.get("space", 0.6), "blocks.space", 0.0, 10.0)
        view = b.get("view", list(VIEW_SCREEN if screen else VIEW_WORLD))
        if not isinstance(view, (list, tuple)) or len(view) != 2:
            raise BlockError("blocks.view is [pitch, yaw] in degrees")
        o["view"] = [_number(view[0], "blocks.view pitch", -60.0, 60.0), _number(view[1], "blocks.view yaw", -60.0, 60.0)]
        f = _table(b, "fall", FALL, what)
        if f["height"] != "top":
            f["height"] = _number(f["height"], "blocks.fall.height", 0.0, 1000.0, lo_open=True)
        f["g"] = _number(f["g"], "blocks.fall.g", 0.0, 1e5, lo_open=True)
        f["bounce"] = _number(f["bounce"], "blocks.fall.bounce", 0.0, 0.95)
        f["stagger"] = _number(f["stagger"], "blocks.fall.stagger", 0.0, 2.0)
        f["spin"] = _number(f["spin"], "blocks.fall.spin", 0.0, 180.0)
        f["squash"] = _number(f["squash"], "blocks.fall.squash", 0.0, 0.6)
        o["fall"] = f
        hits = b.get("hits", [])
        hits = [hits] if isinstance(hits, dict) else hits
        if not isinstance(hits, (list, tuple)):
            raise BlockError("blocks.hits is a table or a list of tables {lane, move, ...}")
        o["hits"] = [_hit_options(h, i, what) for i, h in enumerate(hits)]
        o["next"] = _choice(b, "next", NEXTS, what)
        o["crash"] = None if b.get("crash") is None else _number(b["crash"], "blocks.crash")
        t = _table(b, "tumble", TUMBLE, what)
        for key, hi in (("kick", 100.0), ("pop", 100.0), ("spin", 7200.0), ("gap", 1.0)):
            t[key] = _number(t[key], f"blocks.tumble.{key}", 0.0, hi)
        t["life"] = _number(t["life"], "blocks.tumble.life", 0.0, 30.0, lo_open=True)
        t["floor"] = not screen if t["floor"] is None else t["floor"]
        if not isinstance(t["floor"], bool):
            raise BlockError("blocks.tumble.floor is true or false")
        o["tumble"] = t
        if lyr is not None:
            for key in ("land", "to"):
                if key in b:
                    raise BlockError(f"blocks.{key} is a label's; sung words land on their onsets and go at lyrics.to")
        o["land"] = _number(b.get("land", 0.0), "blocks.land")
        o["to"] = None if b.get("to") is None else _number(b["to"], "blocks.to")
        try:
            o["align"], o["valign"] = parse_align(entry.get("align"), entry.get("valign"))
        except TextError as e:
            raise BlockError(str(e)) from None
        off = entry.get("offset", (0.0, 0.0))
        if not isinstance(off, (list, tuple)) or len(off) != 2:
            raise BlockError("offset is [u, v]")
        o["offset"] = [_number(off[0], "offset u"), _number(off[1], "offset v")]
        o["size"] = None if entry.get("size") is None else _number(entry["size"], "size (the edge of a block)", 0.0,
                                                                     lo_open=True)
        o["fit"] = None if entry.get("fit") is None else _number(entry["fit"], "fit", 0.0, 1.0, lo_open=True)
    except BlockError as e:
        raise BlockError(f"{where}: {e}") from None
    return o


# ------------------------------------------------------------------------------------------------------ the words
@dataclass(frozen=True)
class Unit:
    """A word set in blocks: its lyric numbers (a label's: line 0, its place in the label), its group (the lyric line it
    belongs to; a label is one group), the label row it stands in, when its first block lands (clip frame) and its
    letters, the one field that holds lyric text, kept out of every repr."""
    line: int
    word: int
    k: int
    group: int
    row: int
    land: int
    letters: str = field(repr=False, compare=False, default="")

    def __repr__(self):
        return f"Unit({self.line},{self.word})"

    __str__ = __repr__


def letters_of(text):
    """The characters of a word that get a block: its letters and digits (an apostrophe or a hyphen gets none)."""
    return "".join(c for c in text if c.isalnum())


def _lyric_units(entry, tl, fps, frame0):
    lyr = entry["lyrics"]
    try:
        words = LY.select(tl, lyr.get("line"), lyr.get("lines"), lyr.get("words"), fps, lyr.get("case", "keep"),
                          lyr.get("clean", False))
        first = LY.cut_frame(_number(lyr["from"], "lyrics.from"), fps, frame0) if lyr.get("from") is not None else None
        t_to = _number(lyr["to"], "lyrics.to") if lyr.get("to") is not None else max(w.voiced_end for w in words)
    except LY.LyricsError as e:
        raise BlockError(str(e)) from None
    lines = []
    for w in words:
        if w.line not in lines:
            lines.append(w.line)
    units = [Unit(w.line, w.index, w.k, lines.index(w.line), -1, LY.landing_frame(w.start, fps, first), letters_of(w.text))
             for w in words]
    return units, first, LY.cut_frame(t_to, fps, frame0), lines


def _label_units(entry, o, fps):
    text = entry.get("text")
    if isinstance(text, (list, tuple)) and all(isinstance(s, str) for s in text):
        text = "\n".join(text)
    if not isinstance(text, str) or not text.strip():
        raise BlockError("text is the label to set in blocks (a string; `\\n` or a list of strings for rows)")
    rows = [r.split() for r in text.split("\n")]
    rows = [r for r in rows if r]
    units, k = [], 0
    for i, r in enumerate(rows):
        for j, w in enumerate(r):
            units.append(Unit(0, k + 1, k, 0, i, 0, letters_of(w)))
            k += 1
    return units, len(rows)


# ------------------------------------------------------------------------------------------------------ layout
def view_matrix(pitch, yaw):
    """3x3 turning the block world for a viewer in front of the panel (looking along -z): `pitch` degrees turns the tops
    toward the viewer, `yaw` degrees the right-hand sides."""
    p, y = math.radians(pitch), math.radians(-yaw)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, math.cos(p), -math.sin(p)], [0.0, math.sin(p), math.cos(p)]])
    ry = np.array([[math.cos(y), 0.0, math.sin(y)], [0.0, 1.0, 0.0], [-math.sin(y), 0.0, math.cos(y)]])
    return ry @ rx


def _word_width(n, gap):
    return n + gap * (n - 1)


def place_rows(rows, gap, space, align="center"):
    """Block centres (x, row) of rows of words: `rows` is a list (bottom row first) of lists of block counts; the blocks of
    a word stand `gap` edges apart, words `space` edges apart, every row set against the widest by `align`. Returns
    [[x of each block of each word] per row], the widest row's width."""
    widths = [sum(_word_width(n, gap) for n in r) + space * (len(r) - 1) for r in rows]
    big = max(widths) if widths else 0.0
    out = []
    for r, w in zip(rows, widths):
        x = {"left": -0.5 * big, "center": -0.5 * w, "right": 0.5 * big - w}[align]
        xs = []
        for n in r:
            xs.append([x + 0.5 + j * (1.0 + gap) for j in range(n)])
            x += _word_width(n, gap) + space
        out.append(xs)
    return out, big


CORNERS = np.array([[sx, sy, sz] for sx in (-0.5, 0.5) for sy in (-0.5, 0.5) for sz in (-0.5, 0.5)])


def fit_blocks(centres, R, panel, size, fit, align, valign, offset):
    """(edge, tx, ty, tz, fits): the block edge (m) and where the block world's origin goes in the panel frame for blocks
    standing at `centres` (edges) turned by `R`: the largest edge at which their outline fills `fit` of the panel (at most
    `size`; `size` alone does not shrink), the outline aligned in the panel's `fit` margin as text is, and `tz` lifting
    them so no corner pokes behind the panel."""
    q = (np.asarray(centres, float)[:, None, :] + CORNERS[None]).reshape(-1, 3) @ np.asarray(R).T
    x0, x1, y0, y1 = q[:, 0].min(), q[:, 0].max(), q[:, 1].min(), q[:, 1].max()
    pw, ph = (float(panel[0]), float(panel[1])) if panel else (0.0, 0.0)
    f = 0.9 if fit is None else float(fit)
    aw, ah = pw * f, ph * f
    e_fit = math.inf
    if aw > 0:
        e_fit = aw / (x1 - x0)
    if ah > 0:
        e_fit = min(e_fit, ah / (y1 - y0))
    if size is not None:
        e = float(size) if fit is None else min(float(size), e_fit)
    elif math.isinf(e_fit):
        raise BlockError("give `size` (the edge of a block, m) or a panel to fit the blocks in (`on`, `box` or `screen`)")
    else:
        e = e_fit
    tx = {"left": -0.5 * aw - e * x0, "center": -0.5 * e * (x0 + x1), "right": 0.5 * aw - e * x1}[align] + offset[0]
    ty = {"bottom": -0.5 * ah - e * y0, "middle": -0.5 * e * (y0 + y1), "top": 0.5 * ah - e * y1}[valign] + offset[1]
    tz = -e * float(q[:, 2].min())
    fits = (pw <= 0 or e * (x1 - x0) <= pw * (1 + 1e-9)) and (ph <= 0 or e * (y1 - y0) <= ph * (1 + 1e-9))
    return e, tx, ty, tz, bool(fits)


def _layout(units, o, n_label_rows, R_view, panel):
    """Rest centres (x, y, z in edges) of every block, in unit order then letter order, and the rows a flow took per line.
    Rows build up from the bottom: a line's first row is its lowest (the first words land first, on the floor), and
    with `next = "stack"` every later line stands on the one before; a label reads top to bottom."""
    gap, space = o["gap"], o["space"]
    by_group = {}
    for u in units:
        by_group.setdefault(u.group, []).append(u)
    groups = [by_group[g] for g in sorted(by_group)]

    def arrange(n_rows):
        rows_of = {}                                    # unit k -> physical row (0 = the floor)
        base = 0
        per_group = []
        for us in groups:
            if n_label_rows:                            # a label: its own rows, the last at the bottom
                rows = [n_label_rows - 1 - u.row for u in us]
            elif o["layout"] == "stack":
                rows = list(range(len(us)))
            else:
                widths = [_word_width(len(u.letters), gap) for u in us]
                rows = TF.wrap_rows(widths, min(n_rows, len(us)), gap=space)
            for u, r in zip(us, rows):
                rows_of[u.k] = base + r
            n = max(rows) + 1
            per_group.append(n)
            if o["next"] == "stack":
                base += n
        n_phys = max(rows_of.values()) + 1
        row_lists = [[u for u in units if rows_of[u.k] == r] for r in range(n_phys)]
        xs, _ = place_rows([[len(u.letters) for u in r] for r in row_lists], gap, space, o["align"])
        cen = {}
        for r, (us, xr) in enumerate(zip(row_lists, xs)):
            for u, xw in zip(us, xr):
                cen[u.k] = [(x, r + 0.5, 0.5) for x in xw]
        if o["next"] == "tumble" and not n_label_rows:  # each line in the same place: x per line, not per pile
            for us in groups:
                rows = sorted({rows_of[u.k] for u in us})
                rl = [[u for u in us if rows_of[u.k] == r] for r in rows]
                xs_g, _ = place_rows([[len(u.letters) for u in r] for r in rl], gap, space, o["align"])
                for r, (ur, xr) in zip(rows, zip(rl, xs_g)):
                    for u, xw in zip(ur, xr):
                        cen[u.k] = [(x, r + 0.5, 0.5) for x in xw]
        return cen, per_group

    if n_label_rows or o["layout"] == "stack" or o["rows"] != "auto":
        cen, per = arrange(1 if o["rows"] == "auto" else o["rows"])
        return cen, per
    best = None
    for n in range(1, max(len(us) for us in groups) + 1):     # auto: the rows that set the blocks biggest
        cen, per = arrange(n)
        pts = np.array([p for k in sorted(cen) for p in cen[k]])
        if panel:
            e = fit_blocks(pts, R_view, panel, o["size"], 1.0, "center", "middle", (0.0, 0.0))[0]
        else:
            e = 1.0 / n
        if best is None or e > best[0] * (1 + 1e-9):
            best = (e, cen, per)
    return best[1], best[2]


# ------------------------------------------------------------------------------------------------------ motion
def _quat(axis, angle):
    """Quaternions (w, x, y, z) of turns by `angle` (radians, an array) about the unit `axis`."""
    a = np.asarray(angle, float) * 0.5
    ax = np.asarray(axis, float)
    return np.stack([np.cos(a), ax[0] * np.sin(a), ax[1] * np.sin(a), ax[2] * np.sin(a)], axis=-1)


def _qmul(a, b):
    w1, x1, y1, z1 = np.moveaxis(a, -1, 0)
    w2, x2, y2, z2 = np.moveaxis(b, -1, 0)
    return np.stack([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2], axis=-1)


def _unit_vec(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


def _qmat(q):
    """3x3 rotation of a unit quaternion (w, x, y, z)."""
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def _matq(m):
    """Unit quaternion (w, x, y, z) of a 3x3 rotation."""
    t = m[0, 0] + m[1, 1] + m[2, 2]
    if t > 0:
        s = 2.0 * math.sqrt(t + 1.0)
        q = [0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s]
    elif m[0, 0] >= m[1, 1] and m[0, 0] >= m[2, 2]:
        s = 2.0 * math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
        q = [(m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s]
    elif m[1, 1] >= m[2, 2]:
        s = 2.0 * math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
        q = [(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s]
    else:
        s = 2.0 * math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
        q = [(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s]
    q = np.array(q)
    return q / np.linalg.norm(q)


def _cube_turns():
    """The 24 turns that take a cube onto itself (signed permutation matrices with determinant +1)."""
    out = []
    for perm in ((0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)):
        for signs in ((1, 1, 1), (1, 1, -1), (1, -1, 1), (1, -1, -1), (-1, 1, 1), (-1, 1, -1), (-1, -1, 1), (-1, -1, -1)):
            m = np.zeros((3, 3))
            for i, (j, s) in enumerate(zip(perm, signs)):
                m[i, j] = s
            if np.linalg.det(m) > 0:
                out.append(m)
    return np.array(out)


CUBE_TURNS = _cube_turns()


def _slerp(a, b, u):
    d = float(np.dot(a, b))
    if d < 0:
        b, d = -b, -d
    if d > 0.9995:
        q = a + u * (b - a)
        return q / np.linalg.norm(q)
    th = math.acos(min(1.0, d))
    return (math.sin((1.0 - u) * th) * a + math.sin(u * th) * b) / math.sin(th)


def tumble_path(p0, v0, q0, axis, omega, g, e, t0, ts, floor):
    """Centres (n, 3) and turns (n, 4) of a tumbling block at clip seconds `ts` (all >= t0): thrown from `p0` at `v0`
    (edges, edges/s), turning `omega` rad/s about `axis`, under gravity. Without `floor` it falls on for ever (exact
    ballistics). With it, it lands on the floor of the bottom row (y = 0: its lowest corner touches it), bouncing with
    restitution `e`, losing FLOOR_KEEP of its speed along the floor and of its spin at every bounce, sliding to a stop,
    and lies flat on its nearest face (SETTLE s), stepped TUMBLE_STEPS times a frame."""
    ts = np.asarray(ts, float)
    p0, v0, q0 = (np.asarray(x, float) for x in (p0, v0, q0))
    tau = ts - t0
    if not floor:
        pos = p0 + v0 * tau[:, None]
        pos[:, 1] -= 0.5 * g * tau * tau
        return pos, _qmul(_quat(axis, omega * tau), np.broadcast_to(q0, (len(ts), 4)))
    dt = TUMBLE_DT
    v_rest = math.sqrt(2.0 * g * MIN_HOP)
    p, v, q, w = p0.copy(), v0.copy(), q0.copy(), np.asarray(axis, float) * omega
    T, P, Q = [t0], [p.copy()], [q.copy()]
    settle = None
    t, t_end = t0, float(ts.max()) if len(ts) else t0
    while t < t_end:
        t += dt
        if settle is None:
            v[1] -= g * dt
            p = p + v * dt
            wn = float(np.linalg.norm(w))
            if wn > 1e-12:
                q = _qmul(_quat(w / wn, wn * dt), q)
            low = p[1] - 0.5 * float(np.abs(_qmat(q)[1]).sum())
            if low < 0:                                 # on the floor: pushed out, bouncing off it
                p[1] -= low
                if v[1] < 0:
                    v[1] = -e * v[1]
                    v[0] *= FLOOR_KEEP
                    v[2] *= FLOOR_KEEP
                    w *= FLOOR_KEEP
            if low < 1e-3:                              # lying on it: the slide and the spin die away
                k = math.exp(-dt / SLIDE_TAU)
                v[0] *= k
                v[2] *= k
                w *= k
                if abs(v[1]) < v_rest and math.hypot(v[0], v[2]) < 0.3 and np.linalg.norm(w) < 1.5 or t - t0 > 3.0:
                    m = _qmat(q)
                    snap = _matq(CUBE_TURNS[int(np.argmax((CUBE_TURNS * m[None]).sum(axis=(1, 2))))])
                    settle = (t, p.copy(), q.copy(), snap)
        else:
            t1, p1, q1, snap = settle
            u = LY.sstep(0.0, 1.0, (t - t1) / SETTLE)
            q = _slerp(q1, snap, u)
            p = p1.copy()
            p[1] = max(p1[1] + (0.5 - p1[1]) * u, 0.5 * float(np.abs(_qmat(q)[1]).sum()))   # rocking onto its face
        T.append(t)
        P.append(p.copy())
        Q.append(q.copy())
    T, P, Q = np.array(T), np.array(P), np.array(Q)
    for i in range(1, len(Q)):
        if float(np.dot(Q[i], Q[i - 1])) < 0:
            Q[i] = -Q[i]
    pos = np.stack([np.interp(ts, T, P[:, i]) for i in range(3)], axis=-1)
    quat = np.stack([np.interp(ts, T, Q[:, i]) for i in range(4)], axis=-1)
    return pos, quat / np.linalg.norm(quat, axis=1, keepdims=True)


class Motion:
    """One block's vertical life on its floor and what follows: the fall from `height` edges landing at `t_land`, its
    bounces, the hits it takes while it stands, and a tumble. Times in clip seconds, heights in edges above its floor."""

    def __init__(self, t_land, height, f):
        self.g, self.e = f["g"], f["bounce"]
        self.squash = f["squash"]
        self.v_ref = math.sqrt(2.0 * self.g * REF_DROP)
        self.v_min = math.sqrt(2.0 * self.g * MIN_HOP)
        self.t_land = t_land
        self.t_start = t_land - math.sqrt(2.0 * height / self.g)
        self.segs = [("arc", self.t_start, t_land, height, 0.0, 0.0)]     # (kind, t0, t1, y0, v0, turn degrees)
        self.impacts = []                                                  # (t, squash amount, a slam's: True)
        self.free = self._impact(t_land, self.g * (t_land - self.t_start), self.e, self._amp)
        self.moves = []
        self.tumble = None

    def _amp(self, v):
        return self.squash * min(1.0, v / self.v_ref)

    def _impact(self, t, v, e, amp, slam=False):
        """An impact at `t` at speed `v` (edges/s): a squash, then bounces until they are too low to see; returns when the
        block is at rest again."""
        self.impacts.append((t, amp(v), slam))
        while True:
            v *= e
            if v < self.v_min:
                return t
            dur = 2.0 * v / self.g
            self.segs.append(("arc", t, t + dur, 0.0, v, 0.0))
            self.impacts.append((t + dur, self._amp(v), False))
            t += dur

    def jump(self, t, height, turn):
        """A hop off the floor at `t` (when the block stands there), `turn` degrees about its x axis on the way."""
        if t < self.free - 1e-9:
            return False
        v = math.sqrt(2.0 * self.g * height)
        dur = 2.0 * v / self.g
        self.segs.append(("arc", t, t + dur, 0.0, v, turn))
        self.free = self._impact(t + dur, v, self.e, self._amp)
        self.moves.append(("jump", t))
        return True

    def slam(self, t, height, lead, squash):
        """Lifted `height` over the first part of `lead` before `t`, slammed down onto the floor on `t`."""
        if t - lead < self.free - 1e-9:
            return False
        self.segs.append(("slam", t - lead, t, height, lead, 0.0))
        v = 2.0 * height / ((1.0 - SLAM_PEAK) * lead)
        self.free = self._impact(t, v, self.e * SLAM_BOUNCE, lambda _v: squash, slam=True)
        self.moves.append(("slam", t))
        return True

    def height(self, t):
        """(height above the floor, upward speed, turn about x in degrees) at clip seconds `t` (an array)."""
        t = np.asarray(t, float)
        y, v, turn = np.zeros_like(t), np.zeros_like(t), np.zeros_like(t)
        for kind, t0, t1, a, b, c in self.segs:
            m = (t >= t0) & (t < t1)
            if not m.any():
                continue
            tau = t[m] - t0
            if kind == "arc":
                y[m] = a + b * tau - 0.5 * self.g * tau * tau
                v[m] = b - self.g * tau
                turn[m] = c * tau / (t1 - t0)
            else:                                    # slam: up over SLAM_PEAK of the lead (ease out), then the plunge
                u = tau / b
                up = u < SLAM_PEAK
                w = np.where(up, 0.0, (u - SLAM_PEAK) / (1.0 - SLAM_PEAK))
                y[m] = a * np.where(up, np.sin(0.5 * math.pi * u / SLAM_PEAK), 1.0 - w * w)
                v[m] = a * np.where(up, 0.5 * math.pi / (SLAM_PEAK * b) * np.cos(0.5 * math.pi * u / SLAM_PEAK),
                                    -2.0 * w / ((1.0 - SLAM_PEAK) * b))
        return y, v, turn

    def squash_at(self, t):
        """(own, slam): the squash of the block's own impacts, and of the slams, which press every block of a pile at once
        (the pile shortens with them: a block's floor sinks by its row times the slam's squash)."""
        t = np.asarray(t, float)
        own, slam = np.zeros_like(t), np.zeros_like(t)
        for ti, amp, is_slam in self.impacts:
            if amp <= 0:
                continue
            tau = t - ti
            m = tau >= 0
            s = own if not is_slam else slam
            s[m] += amp * np.exp(-tau[m] / SQUASH_TAU) * np.cos(2.0 * math.pi * tau[m] / SQUASH_PERIOD)
        return own, slam


def _hash(*ints):
    return LY.h01(*[int(i) for i in ints])


def sample(b, mo, fps, R=None, e=1.0, ty=0.0, bottom=None, spin=0.0, on_floor=False):
    """Frames and keys of block `b` (a Block with rest centre, landing, tumble set) from its first frame on screen to the
    last: (frames, loc (n, 3) edges, quat (n, 4) w x y z, scale (n, 3)) and the clip frame it is gone from. A tumbling
    block lands on the floor with `on_floor` (tumble_path); otherwise it is gone when its life is over or, with `bottom`
    (screen type), when it has fallen below the picture."""
    f_on, f_off = b.on, b.off
    frames = np.arange(f_on, f_off)
    t = frames / fps
    y, _v, turn = mo.height(t)
    own, slam = mo.squash_at(t)
    s = own + slam
    x0, y0, z0 = b.rest
    base = (y0 - 0.5) * (1.0 - slam)
    loc = np.stack([np.full_like(t, x0), base + y + 0.5 * (1.0 - s), np.full_like(t, z0)], axis=-1)

    def orient(ts, turns):
        """The block's turn at times `ts` before any tumble: tilted as it falls, turning at a steady rate to lie flat on
        the impact; a hop's turn about its x axis."""
        q = np.zeros((len(ts), 4))
        q[:, 0] = 1.0
        fall = ts < mo.t_land
        if fall.any() and spin > 0:
            r = (mo.t_land - ts[fall]) / max(mo.t_land - mo.t_start, 1e-9)
            q[fall] = _quat(b.fall_axis, math.radians(spin) * b.fall_share * r)
        tm = np.abs(turns) > 0
        if tm.any():
            q[tm] = _quat((1.0, 0.0, 0.0), np.radians(turns[tm]))
        return q

    q = orient(t, turn)
    scale = np.stack([1.0 + 0.5 * s, 1.0 - s, 1.0 + 0.5 * s], axis=-1)
    if mo.tumble is not None:
        t_tb, vel, axis, omega = mo.tumble
        m = t >= t_tb
        if m.any():
            y_tb, v_tb, turn_tb = mo.height(np.array([t_tb]))
            q0 = orient(np.array([t_tb]), turn_tb)[0]
            p0 = (x0, y0 + float(y_tb[0]), z0)
            v0 = (vel[0], float(v_tb[0]) + vel[1], vel[2])
            loc[m], q[m] = tumble_path(p0, v0, q0, axis, omega, mo.g, mo.e, t_tb, t[m], on_floor)
            scale[m] = 1.0
            if bottom is not None and R is not None:   # out of the picture: gone
                py = e * (loc[m] @ np.asarray(R)[1]) + ty + e * REACH
                below = np.nonzero(py < bottom)[0]
                if len(below):
                    cut = int(np.nonzero(m)[0][0] + below[0])
                    f_off = int(frames[cut])
                    frames, loc, q, scale = frames[:cut], loc[:cut], q[:cut], scale[:cut]
    for i in range(1, len(q)):                     # one hemisphere: the keys blend the short way
        if float(np.dot(q[i], q[i - 1])) < 0:
            q[i] = -q[i]
    return frames, loc, q, scale, f_off


def thin(frames, values, tol=TOL):
    """The indices of the keys to keep of a channel sampled on whole frames: the first, the last and every key off the
    straight line through its neighbours (dropping the others leaves linear keys exact on every frame)."""
    v = np.asarray(values, float)
    n = len(v)
    if n <= 2:
        return np.arange(n)
    d2 = np.abs(v[:-2] - 2.0 * v[1:-1] + v[2:])
    keep = np.concatenate([[True], d2 > tol, [True]])
    return np.nonzero(keep)[0]


# ------------------------------------------------------------------------------------------------------ the plan
@dataclass
class Block:
    """One block: its tag (`l<line>w<word>c<letter>`, a label's `c<k>`), lyric numbers (None for a label), place in its
    word (1-based), slot colours (wood, the six faces' paint in FACES order, ink, then its pictures'), mesh key (holds its
    letters: never printed), rest centre (edges), landing (clip frame), the lyric line it belongs to (0-based, in the
    selection), fall height, frames on screen [on, off) and keys."""
    tag: str
    lyric: tuple
    char: int
    k: int
    colors: list
    mesh: "MeshSpec" = field(repr=False)
    rest: tuple = (0.0, 0.0, 0.0)
    land: int = 0
    group: int = 0
    height: float = 0.0
    on: int = 0
    off: int = 0
    fall_axis: tuple = (0.0, 0.0, 1.0)
    fall_share: float = 1.0
    keys: dict = field(default=None, repr=False)
    moves: list = field(default_factory=list, repr=False)

    def __repr__(self):
        return f"Block({self.tag})"


@dataclass
class Plan:
    """The blocks of an entry (each with its MeshSpec in `mesh`: blocks with one letter share it), the block world's place
    in the panel frame (`matrix`, 4x4: edges -> panel metres) and the summary (numbers only)."""
    blocks: list
    edge: float
    matrix: np.ndarray
    summary: dict


@dataclass(frozen=True)
class MeshSpec:
    """What a block's six faces show: per face ("glyph", character) | ("picture", name) | None, in FACES order; the
    characters are lyric text, kept out of every repr."""
    faces: tuple = field(repr=False)
    letter: str = "raised"
    relief: float = 0.05
    round: float = 0.12
    rim: bool = True

    def __repr__(self):
        return f"MeshSpec({self.letter}, rim={self.rim})"


def _neighbour(c, step):
    if "a" <= c.lower() <= "z" and c.isascii():
        base = ord("A") if c.isupper() else ord("a")
        return chr(base + (ord(c) - base + step) % 26)
    if c.isdigit() and c.isascii():
        return str((int(c) + step) % 10)
    return c


def face_plan(ch, o, draws=None):
    """MeshSpec of a block whose front shows `ch` (`draws(c)`: whether the font draws `c`; a face whose glyph it cannot
    draw is left plain) and the colours of its picture slots (slot 8 on)."""
    what = WHAT[o["sides"]]
    pics = o["pictures"]
    code = sum(ord(c) for c in ch)
    faces, extra = [], []
    for i, f in enumerate(FACES):
        if f == "front":
            faces.append(("glyph", ch))
            continue
        kind = what.get(f)
        c = None
        if kind == "case":
            c = ch.swapcase() if ch.swapcase() != ch else ch
        elif kind in ("next", "next2", "previous"):
            c = _neighbour(ch, {"next": 1, "next2": 2, "previous": -1}[kind])
        elif kind == "digit":
            c = str(code % 9 + 1)
        if c is not None:
            faces.append(("glyph", c) if draws is None or draws(c) else None)
        elif kind == "picture":
            name = pics[(code * 7 + i * 5) % len(pics)]
            faces.append(("picture", name))
            for col, *_ in PICTURES[name]:
                if col not in extra:
                    extra.append(col)
        else:
            faces.append(None)
    spec = MeshSpec(tuple(faces), o["letter"], o["relief"], o["round"], o["rim"])
    return spec, extra


def _colors(k, spec, extra, o):
    p = o["paint"]
    face_cols = []
    for f, c in zip(FACES, spec.faces):
        face_cols.append(o["ground"] if c is not None and c[0] == "picture" else p[(k + PAINT_STEP[f]) % len(p)])
    return [o["wood"]] + face_cols + [o["ink"]] + list(extra)


def plan(entry, *, fps, frame0=0, tl=None, panel=None, top=None, bottom=None, hits=None, end=None, screen=False,
         draws=None):
    """The blocks of a `[[text]]` set in blocks, placed and keyed.

    entry    the [[text]] table (`name`, placement, `size` / `fit`, `align`, `offset`, `lyrics` or `text`, `blocks`)
    fps, frame0   the project's; `lyrics.from` / `to` and a label's `blocks.to` round to frames as the shots stage cuts
    tl       the loaded timeline (sung words)
    panel    (w, h) of the entry's panel (m; frame heights for screen type) or None (then `size` is the edge)
    top, bottom   the top and bottom of the picture in the panel frame (screen type): blocks drop in from above it and a
             tumbling block is gone below it; without `top` they start just above the panel (or 6 edges up)
    hits     the [audio] hits file ({lane: [...]}) for `blocks.hits`
    end      the clip frame a label's blocks are gone from without `blocks.to` (the end of the build)
    draws    callable(character) -> bool: whether the font draws it (a block needs its front letter; other faces fall back)
    Returns a Plan; nothing in it but `Block.mesh` and `MeshSpec.faces` holds a letter."""
    o = options(entry, screen)
    name = entry["name"]
    where = f"text {name!r}"
    f = o["fall"]
    label = entry.get("lyrics") is None
    try:
        if label:
            units, n_label_rows = _label_units(entry, o, fps)
            first, lines = None, []
            cut = LY.cut_frame(o["to"], fps, frame0) if o["to"] is not None else (int(end) if end is not None else 10 ** 9)
        else:
            if tl is None:
                raise BlockError("the sung words need the timeline")
            units, first, cut, lines = _lyric_units(entry, tl, fps, frame0)
            n_label_rows = 0
        empty = [u for u in units if not u.letters]
        units = [u for u in units if u.letters]
        if not units:
            raise BlockError("nothing to set in blocks: no letter or digit in the words")
        if draws is not None:
            for u in units:
                for j, c in enumerate(u.letters):
                    if not draws(c):
                        what = f"word ({u.line}, {u.word})" if not label else f"word {u.word} of the label"
                        raise BlockError(f"the font draws nothing for letter {j + 1} of {what}")
        R = view_matrix(*o["view"])
        cen, rows_per = _layout(units, o, n_label_rows, R, panel)
        pts = np.array([p for u in units for p in cen[u.k]])
        e, tx, ty, tz, fits = fit_blocks(pts, R, panel, o["size"], o["fit"], o["align"], o["valign"], o["offset"])
    except BlockError as err:
        raise BlockError(f"{where}: {err}") from None

    # every block: its landing, colours, faces and mesh
    blocks, mesh_of = [], {}
    order = sorted(units, key=lambda u: (min(p[1] for p in cen[u.k]), u.k)) if label else units
    i_land = 0
    stagger = f["stagger"]
    land0 = int(math.floor(o["land"] * fps + 0.4)) if label else None
    for u in order:
        for j, c in enumerate(u.letters):
            if label:
                land = land0 + int(round(i_land * stagger * fps))
            else:
                land = u.land + int(round(j * stagger * fps))
            i_land += 1
            if c not in mesh_of:
                spec, extra = face_plan(c, o, draws)
                mesh_of[c] = (spec, extra)
            spec, extra = mesh_of[c]
            tag = f"l{u.line}w{u.word}c{j + 1}" if not label else f"c{len(blocks) + 1}"
            k = len(blocks)
            blocks.append(Block(tag, None if label else (u.line, u.word), j + 1, k, _colors(k, spec, extra, o), spec,
                                rest=tuple(cen[u.k][j]), land=land, group=u.group))
    if label:                                         # a label's tags in reading order, whatever order its blocks land in
        reading = sorted(blocks, key=lambda b: (-b.rest[1], b.rest[0]))
        for i, b in enumerate(reading):
            b.tag = f"c{i + 1}"
        blocks.sort(key=lambda b: int(b.tag[1:]))
        for i, b in enumerate(blocks):
            b.k = i

    # fall heights: from just above the picture (screen type), the panel, or `fall.height` edges
    R1 = R[1]
    for b in blocks:
        if f["height"] != "top":
            b.height = float(f["height"])
            continue
        lim = top if top is not None else (0.5 * panel[1] if panel else None)
        if lim is None:
            b.height = 6.0
            continue
        rest_y = e * float(R1 @ np.asarray(b.rest)) + ty
        b.height = max(2.0, ((lim - rest_y) / e + REACH) / max(R1[1], 0.2))

    # motion: fall, bounces, hits, tumbles
    motions = {}
    for b in blocks:
        mo = Motion(b.land / fps, b.height, f)
        b.fall_axis = tuple(_unit_vec([0.45 * (_hash(b.k, 11) - 0.5), 0.4 * (_hash(b.k, 12) - 0.5),
                                       1.0 if _hash(b.k, 13) < 0.5 else -1.0]))
        b.fall_share = 0.6 + 0.4 * _hash(b.k, 14)
        motions[b.k] = mo
    pitch = 1.0 + o["gap"]
    xs = np.array([b.rest[0] for b in blocks])
    tumbles = []
    groups = sorted({u.group for u in units})
    if not label and o["next"] == "tumble":           # each line knocked off as the next one comes down
        for g, g_next in zip(groups, groups[1:]):
            mine = [b for b in blocks if b.group == g]
            nxt = [b for b in blocks if b.group == g_next]
            first_new = min(nxt, key=lambda b: (motions[b.k].t_start, b.k))
            t0 = max(motions[first_new.k].t_start, max(b.land for b in mine) / fps + 0.05)
            tumbles.append((t0, first_new.rest[0], mine))
    if o["crash"] is not None:
        tumbles.append((o["crash"], float(xs.mean()), list(blocks)))

    # hits: in time order, each block moves when it stands on its floor
    hit_counts = {}
    if o["hits"]:
        if hits is None:
            raise BlockError(f"{where}: blocks.hits reads the [audio] hits file and the project names none (`mk timeline "
                             f"hits` writes it)")
        events = []
        for h in o["hits"]:
            try:
                ts = LN.pick(hits, h["lane"], h["min_db"], h["every"], h["t0"], h["t1"])
            except LN.LaneError as err:
                raise BlockError(f"{where}: blocks.hits: {err}") from None
            events += [(t, h) for t in ts]
            hit_counts[h["move"]] = hit_counts.get(h["move"], 0) + len(ts)
        events.sort(key=lambda ev: ev[0])
        x_min = float(xs.min())
        tumble_at = {}
        for t0, x_imp, bs in tumbles:
            for b in bs:
                d = o["tumble"]["gap"] * abs(b.rest[0] - x_imp) / pitch
                tumble_at[b.k] = min(tumble_at.get(b.k, math.inf), t0 + d)
        for t_h, h in events:                         # on the hit's frame (the words' landing rule), rippling along x
            t_f = LY.landing_frame(t_h, fps) / fps
            for b in blocks:
                mo = motions[b.k]
                t = t_f + h["wave"] * (b.rest[0] - x_min) / pitch
                if t >= tumble_at.get(b.k, math.inf) or t >= cut / fps:
                    continue
                if h["move"] == "jump":
                    mo.jump(t, h["height"], h["turn"])
                else:
                    mo.slam(t, h["height"], h["lead"], h["squash"])
    tb = o["tumble"]
    tumble_summary = []
    for t0, x_imp, bs in tumbles:
        n = 0
        for b in bs:
            mo = motions[b.k]
            t = t0 + tb["gap"] * abs(b.rest[0] - x_imp) / pitch
            if mo.tumble is not None and mo.tumble[0] <= t:
                continue
            side = 1.0 if b.rest[0] > x_imp + 0.25 else -1.0 if b.rest[0] < x_imp - 0.25 else \
                (1.0 if _hash(b.k, 21) < 0.5 else -1.0)
            near = 1.0 / (1.0 + 0.15 * abs(b.rest[0] - x_imp))
            vel = (side * tb["kick"] * (0.6 + 0.4 * _hash(b.k, 22)) * (0.5 + 0.5 * near),
                   tb["pop"] * (0.7 + 0.3 * _hash(b.k, 23)),
                   tb["kick"] * 0.5 * (0.3 + 0.7 * _hash(b.k, 24)))
            axis = tuple(_unit_vec([0.8 * (_hash(b.k, 25) - 0.5), 0.6 * (_hash(b.k, 26) - 0.5), -side]))
            omega = math.radians(tb["spin"]) * (0.6 + 0.4 * _hash(b.k, 27))
            mo.tumble = (t, vel, axis, omega)
            n += 1
        tumble_summary.append({"t": round(t0, 4), "blocks": n})

    # frames on screen and keys
    kept = []
    for b in blocks:
        mo = motions[b.k]
        on = int(math.ceil(mo.t_start * fps - 1e-6))
        if first is not None:
            on = max(on, first)
        off = cut
        if mo.tumble is not None and not tb["floor"]:   # flown off: gone after its life (lying on the floor: to the cut)
            off = min(off, int(math.ceil((mo.tumble[0] + tb["life"]) * fps)))
        b.on, b.off = on, off
        if off <= on:
            continue
        frames, loc, q, scale, off = sample(b, mo, fps, R, e, ty, bottom if top is not None else None, f["spin"],
                                            tb["floor"])
        b.off = off
        if off <= on or not len(frames):
            continue
        b.keys = {"frames": frames, "loc": loc, "quat": q, "scale": scale}
        b.moves = list(mo.moves)
        kept.append(b)
    skipped = []
    if not label:                                     # words with no letter, and words none of whose blocks is ever up
        shown = {b.lyric for b in kept}
        skipped = [[u.line, u.word] for u in empty] + [list(s) for s in sorted({b.lyric for b in blocks} - shown)]
    M = np.eye(4)
    M[:3, :3] = R * e
    M[:3, 3] = (tx, ty, tz)
    moves = {}
    for b in kept:
        for kind, _t in b.moves:
            moves[kind] = moves.get(kind, 0) + 1
    summary = {"style": "blocks", "label": label, "blocks": len(kept), "words": len({b.lyric for b in kept}) if not label
               else len(units), "lines": lines, "layout": o["layout"], "rows": rows_per, "next": o["next"],
               "edge": float(round(e, 6)), "view": list(o["view"]), "fits": fits, "skipped": skipped,
               "first_frame": min((b.land for b in kept), default=None), "last_frame": max((b.land for b in kept), default=None),
               "cut_frame": cut if cut < 10 ** 9 else None, "hits": hit_counts, "moves": moves, "tumbles": tumble_summary,
               "crash": o["crash"], "meshes": len({b.mesh for b in kept})}
    return Plan(kept, e, M, summary)


# ------------------------------------------------------------------------------------------------------ geometry
@dataclass
class Mesh:
    """A block's mesh in its own frame (edge 1, centred): vertices, polygons as loops (`loops` vertex indices, `sizes` per
    polygon), a normal per loop and a material slot per polygon (0 wood, 1-6 the faces' paint in FACES order, 7 ink, 8 on
    the pictures' colours in `extra` order)."""
    V: np.ndarray
    loops: np.ndarray
    sizes: np.ndarray
    normals: np.ndarray
    mats: np.ndarray
    extra: list


def rounded_cube(r=0.12, steps=ARC_STEPS):
    """A unit cube (edge 1, centred) with its edges and corners rounded at radius `r`: (V, quads (q, 4), vertex normals,
    face of each quad: its index in FACES for the flat painted square, -1 for the wooden round)."""
    h, a = 0.5, 0.5 - r
    t = [a + r * math.tan(math.radians(45.0 * i / steps)) for i in range(steps + 1)]
    axis = np.array([-x for x in reversed(t)] + t)
    n = len(axis)
    mid = steps                                         # the interval from -a to a: the painted square
    pts, quads, which = [], [], []
    for fi, f in enumerate(FACES):
        nrm, u, v = (np.array(x, float) for x in FRAMES[f])
        uu, vv = np.meshgrid(axis, axis, indexing="ij")
        P = nrm * h + uu[..., None] * u + vv[..., None] * v
        base = fi * n * n
        pts.append(P.reshape(-1, 3))
        for i in range(n - 1):
            for j in range(n - 1):
                quads.append([base + i * n + j, base + (i + 1) * n + j, base + (i + 1) * n + j + 1, base + i * n + j + 1])
                which.append(fi if (i == mid and j == mid) else -1)
    P = np.concatenate(pts)
    Q = np.clip(P, -a, a)
    D = P - Q
    N = D / np.linalg.norm(D, axis=1, keepdims=True)
    V = Q + r * N
    key = np.round(V / 1e-7).astype(np.int64)
    _, first, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
    inv = inv.reshape(-1)
    return V[first], inv[np.array(quads)], N[first], np.array(which)


def _fan(points):
    """A star-shaped polygon (about its centroid) as CCW triangles: (V2, T)."""
    p = np.asarray(points, float)
    area = 0.5 * float(np.sum(p[:, 0] * np.roll(p[:, 1], -1) - np.roll(p[:, 0], -1) * p[:, 1]))
    if area < 0:
        p = p[::-1]
    c = p.mean(axis=0)
    V = np.vstack([c[None], p])
    m = len(p)
    T = np.array([[0, 1 + i, 1 + (i + 1) % m] for i in range(m)])
    return V, T


def _disc(cx, cy, rx, ry, deg, n=24):
    a = np.linspace(0.0, 2.0 * math.pi, n, endpoint=False)
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    x, y = rx * np.cos(a), ry * np.sin(a)
    return np.stack([cx + c * x - s * y, cy + s * x + c * y], axis=-1)


def picture_parts(name):
    """[(colour, V2, T)]: the parts of a picture as CCW triangles in its [-0.5, 0.5] square, in drawing order."""
    out = []
    for part in PICTURES[name]:
        col, kind = part[0], part[1]
        pts = _disc(*part[2:]) if kind == "disc" else part[2]
        V, T = _fan(pts)
        out.append((col, V, T))
    return out


def ring2d(outer, inner):
    """A square ring (half sizes `outer` > `inner`) as CCW triangles."""
    o = [(-outer, -outer), (outer, -outer), (outer, outer), (-outer, outer)]
    i = [(-inner, -inner), (inner, -inner), (inner, inner), (-inner, inner)]
    V = np.array(o + i, float)
    T = []
    for k in range(4):
        k1 = (k + 1) % 4
        T += [[k, k1, 4 + k1], [k, 4 + k1, 4 + k]]
    return V, np.array(T)


def _weld2d(V, T):
    key = np.round(np.asarray(V, float) / 1e-7).astype(np.int64)
    _, first, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
    return np.asarray(V, float)[first], inv.reshape(-1)[np.asarray(T)]


def slab(V2, T, relief, printed):
    """A flat shape (V2, CCW triangles T) on a face, in face coordinates (u, v, w: w out of the face, 0 on it): printed,
    a skin PRINT_LIFT above it; raised, a slab `relief` high standing on it, walls round its outline. Returns (V (n, 3),
    polygons [list of vertex index lists], a normal per polygon (face coordinates))."""
    V2, T = _weld2d(V2, T)
    T = [t for t in T.tolist() if len(set(t)) == 3]
    if printed:
        V = np.column_stack([V2, np.full(len(V2), PRINT_LIFT)])
        return V, T, [(0.0, 0.0, 1.0)] * len(T)
    cap = np.column_stack([V2, np.full(len(V2), relief)])
    edges = set()
    for a, b, c in T:
        edges.update(((a, b), (b, c), (c, a)))
    border = [(a, b) for a, b in edges if (b, a) not in edges]
    ring = sorted({v for e in border for v in e})
    low = {v: len(V2) + i for i, v in enumerate(ring)}
    base = np.column_stack([V2[ring], np.full(len(ring), -SINK)]) if ring else np.zeros((0, 3))
    V = np.vstack([cap, base])
    polys = list(T)
    normals = [(0.0, 0.0, 1.0)] * len(T)
    for a, b in sorted(border):
        d = V2[b] - V2[a]
        ln = float(np.hypot(*d))
        if ln < 1e-12:
            continue
        polys.append([low[a], low[b], b, a])
        normals.append((float(d[1]) / ln, float(-d[0]) / ln, 0.0))
    return V, polys, normals


def glyph_fit(V2, cap_ratio, cap, box):
    """A glyph's outline at em size 1 scaled so a capital is `cap` edges tall (shrunk to stay inside a `box` square) and
    centred on its own ink: the scaled vertices."""
    V2 = np.asarray(V2, float)
    lo, hi = V2.min(axis=0), V2.max(axis=0)
    s = cap / cap_ratio
    size = (hi - lo) * s
    if size.max() > box:
        s *= box / size.max()
    return (V2 - 0.5 * (lo + hi)) * s


def block_mesh(spec, glyph, cap_ratio):
    """The mesh of a block (MeshSpec `spec`; `glyph(c)` -> (V2, T) CCW triangles at em size 1, or None) for a font whose
    capital is `cap_ratio` em tall."""
    V, Q, N, which = rounded_cube(spec.round)
    flat = 1.0 - 2.0 * spec.round
    verts, polys, pnorm, vnorm_polys, mats = [V], [], [], [], []
    for q, w in zip(Q.tolist(), which.tolist()):
        polys.append(q)
        vnorm_polys.append(True)
        pnorm.append(None)
        mats.append(SLOT_WOOD if w < 0 else 1 + w)
    n_v = len(V)
    extra = []
    printed = spec.letter == "printed"
    for fi, (f, content) in enumerate(zip(FACES, spec.faces)):
        if content is None:
            continue
        nrm, u, v = (np.array(x, float) for x in FRAMES[f])
        parts = []                                   # (V2, T, slot, printed, w offset)
        if content[0] == "glyph":
            g = glyph(content[1])
            if g is None:
                continue
            outer = 0.5 * flat - RIM_INSET
            inner = outer - RIM_WIDTH
            box = 2.0 * inner - 0.06 if spec.rim else 0.84 * flat
            cap = (0.56 if spec.rim else 0.66) * flat
            parts.append((glyph_fit(g[0], cap_ratio, cap, box), np.asarray(g[1]), SLOT_INK, printed, 0.0))
            if spec.rim:
                rv, rt = ring2d(outer, inner)
                parts.append((rv, rt, SLOT_INK, printed, 0.0))
        else:
            for i, (col, pv, pt) in enumerate(picture_parts(content[1])):
                if col not in extra:
                    extra.append(col)
                parts.append((pv * (0.8 * flat), pt, SLOT_EXTRA + extra.index(col), True, 0.0005 * i))
        for V2, T, slot, prn, dw in parts:
            sv, sp, sn = slab(V2, T, spec.relief, prn)
            if not len(sp):
                continue
            P = 0.5 * nrm + sv[:, :1] * u + sv[:, 1:2] * v + (sv[:, 2:3] + dw) * nrm
            verts.append(P)
            for poly, pn in zip(sp, sn):
                polys.append([n_v + i for i in poly])
                vnorm_polys.append(False)
                pnorm.append(pn[0] * u + pn[1] * v + pn[2] * nrm)
                mats.append(slot)
            n_v += len(P)
    Vall = np.vstack(verts)
    sizes = np.array([len(p) for p in polys], np.int32)
    loops = np.array([i for p in polys for i in p], np.int32)
    normals = np.empty((len(loops), 3))
    at = 0
    for p, use_v, pn in zip(polys, vnorm_polys, pnorm):
        normals[at:at + len(p)] = N[p] if use_v else pn
        at += len(p)
    return Mesh(Vall, loops, sizes, normals, np.array(mats, np.int32), extra)
