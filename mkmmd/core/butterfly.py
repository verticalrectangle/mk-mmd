"""One butterfly, bpy-free (the library props `butterfly` and `butterflies`, docs/design.md: the library table): four wing
styles, their outlines, the wing and body meshes, the colour roles, and the flap maths both props share (with the driver
expressions of the single prop).

Frame: the origin is the middle of the thorax, the head toward -Y, the back +Z, its left +X (a character's frame). Sizes are
at unit span: 1.0 from wing tip to wing tip with the wings flat; a builder scales by its `size` (metres).

Styles, each modelled in proportion on photographs of the real butterfly spread flat (set like a specimen: the forewing's
hind edge square to the body, the hindwing tucked under it), unpatterned here (the patterns are core.butterfly_pattern):
  swallowtail   a tiger swallowtail: a long forewing with a near-straight outer edge, a scalloped hindwing with a tail
  monarch       a monarch: a long, round-tipped forewing, a round hindwing
  morpho        a blue morpho: broad wings, a round forewing tip over a gently waved outer edge
  blue          a small blue (a common blue): short, round wings with a fringe

Flap. A wing is lifted `wing_angle(phase)` above flat: mid + amp / 2 cos(2 pi phase) degrees (amp the full stroke, mid its
middle), phase = beats + phase offset (+ one more cycle over `hit_time` after a hit: `hit_flap`). The left wing (+X) turns
about its hinge (an axis along Y through `hinge_L`) by Euler Y = -angle, the right one by +angle. The body bobs along its
own +Z by `body_bob(phase)` x span: up at the end of the down stroke."""
import functools
import hashlib
import math

import numpy as np

from . import palette as PAL
from . import shell as S

STYLES = ("swallowtail", "monarch", "morpho", "blue")
# a style's span and wing beat against the others' (for a swarm of mixed styles; the single prop's `size` is its span):
# a morpho is the biggest and the slowest, floppy flier, a blue the smallest and quickest
SPAN_FACTOR = {"swallowtail": 1.05, "monarch": 1.0, "morpho": 1.15, "blue": 0.75}
RATE_FACTOR = {"swallowtail": 0.9, "monarch": 0.95, "morpho": 0.8, "blue": 1.35}

# ===================================================================================================================
# flap: the live root properties both props share (root property: (default, unit, doc)); `[[key]]` keys them
# ===================================================================================================================
FLAP_PARAMS = {
    "flap": (3.0, "beats/s", "wing beats a second (a rate: the keys stage integrates it into `beats`)"),
    "flap_amp": (100.0, "deg", "the full stroke: the wings sweep from mid + amp/2 down to mid - amp/2"),
    "flap_mid": (20.0, "deg", "the middle of the stroke, above flat"),
    "phase": (0.0, "cycles", "where in its beat the wing is at beat 0 (0 up, 0.25 mid going down, 0.5 down)"),
    "hit": (-1000.0, "s", "the clip second of the last hit: one more wing beat starts then (a [[key]] lane keys it)"),
    "hit_time": (0.3, "s", "how long the one beat of a hit takes"),
}
BOB = 0.03                          # the body's bob over a beat (a share of the span, half up and half down)


class NumpyOps:
    """The two functions the flap maths needs, on numpy arrays (a geometry-nodes backend passes its own: the maths below
    uses nothing else but `+ - * /` with Python floats)."""

    @staticmethod
    def cos(x):
        return np.cos(x)

    @staticmethod
    def smoothstep(a, b, x):
        t = np.clip((np.asarray(x, float) - a) / (b - a), 0.0, 1.0)
        return t * t * (3.0 - 2.0 * t)


NP = NumpyOps()


def wing_angle(phase, amp_deg, mid_deg, ops=None):
    """How far a wing is lifted above flat (radians) at `phase` (cycles): mid + amp / 2 cos(2 pi phase) degrees."""
    ops = ops or NP
    return (mid_deg + 0.5 * amp_deg * ops.cos(2.0 * math.pi * phase)) * (math.pi / 180.0)


def hit_flap(u, ops=None):
    """The phase a hit adds: 0 before it, one whole cycle once it is over (smoothstep over u = (t - hit) / hit_time)."""
    return (ops or NP).smoothstep(0.0, 1.0, u)


def body_bob(phase, ops=None):
    """The body's rise along its own +Z, a share of the span: lowest with the wings up, highest at the end of the down
    stroke (the lift of the down stroke carries it)."""
    return -0.5 * BOB * (ops or NP).cos(2.0 * math.pi * phase)


def flap_phase(beats, P, t, ops=None):
    """The flap phase of the single prop at clip second(s) t: beats + phase + hit_flap((t - hit) / hit_time)."""
    return beats + P["phase"] + hit_flap((t - P["hit"]) / max(P["hit_time"], 0.001), ops)


# ===================================================================================================================
# the single prop: options and the drivers' expressions (Blender simple expressions on the root's properties)
# ===================================================================================================================
DEFAULTS = {"style": "monarch", "size": 0.09, "way": 0, "tile": 1024}
GLOW = 0.1                          # the wings' own light: what ambient light would give an underside in shade
FUNCS = frozenset({"radians", "cos", "smoothstep", "max"})
NAMES = ("beats", "phase", "hit", "hit_time", "flap_mid", "flap_amp")      # the root properties the drivers read
TAU = 6.283185307


def options(slots):
    """(O, P) of the single prop's `slots`: O = {style, size, way, sheen, tile, glow}, P the starting values of
    FLAP_PARAMS. Refuses what it cannot build, saying what to change."""
    slots = dict(slots or {})
    style = str(slots.get("style", DEFAULTS["style"]))
    if style not in STYLES:
        raise ValueError(f"butterfly: style {style!r} is not one of {', '.join(STYLES)}")
    size = _number(slots, "size", DEFAULTS["size"])
    if size <= 0:
        raise ValueError(f"butterfly: size is the wingspan in metres (> 0), got {size:g}")
    way = slots.get("way", DEFAULTS["way"])
    if isinstance(way, bool) or not isinstance(way, int) or not 0 <= way < len(WAYS[style]):
        raise ValueError(f"butterfly: way picks a colourway of the {style}: 0..{len(WAYS[style]) - 1}, got {way!r}")
    sheen = _number(slots, "sheen", SHEEN[style])
    if sheen < 0:
        raise ValueError(f"butterfly: sheen is 0 (none) or more, got {sheen:g}")
    tile = slots.get("tile", DEFAULTS["tile"])
    if isinstance(tile, bool) or not isinstance(tile, int) or not 64 <= tile <= 4096:
        raise ValueError(f"butterfly: tile is the pattern's pixels per wing, 64..4096, got {tile!r}")
    P = {k: _number(slots, k, d) for k, (d, _u, _doc) in FLAP_PARAMS.items()}
    if P["flap"] < 0:
        raise ValueError(f"butterfly: flap is wing beats a second (>= 0), got {P['flap']:g}")
    if P["hit_time"] <= 0:
        raise ValueError(f"butterfly: hit_time is the seconds one beat of a hit takes (> 0), got {P['hit_time']:g}")
    return {"style": style, "size": size, "way": int(way), "sheen": sheen, "tile": int(tile),
            "glow": glow_of(slots)}, P


def glow_of(slots):
    """The `glow` option of a prop's slots (emission strength of the wings in their own colours; GLOW by default)."""
    glow = _number(dict(slots or {}), "glow", GLOW)
    if glow < 0:
        raise ValueError(f"butterfly: glow is 0 (lit by the scene alone) or more, got {glow:g}")
    return glow


def _number(slots, key, default):
    v = slots.get(key, default)
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise ValueError(f"butterfly: {key} is a number, got {v!r}")       # noqa: TRY004 (a bad slot)
    return float(v)


def expressions(f0, fps, size):
    """The drivers of the single prop: {"wing_L", "wing_R": Euler Y of the wings (radians), "bob": the body's z (m)}, read
    from the root's properties NAMES and the scene frame; `f0` is the frame of clip second 0."""
    phase = (f"(beats + phase + smoothstep(0, 1, ((frame - {f0:.6g}) / {fps:.6g} - hit) / max(hit_time, 0.001)))")
    angle = f"radians(flap_mid + 0.5 * flap_amp * cos({TAU} * {phase}))"
    return {"wing_L": f"-{angle}", "wing_R": angle, "bob": f"{-0.5 * BOB * size:.6g} * cos({TAU} * {phase})"}


def bounds(style, size, P):
    """{min, max} of the single prop in its frame: the body and the wings flat, and the wing tips through the stroke."""
    lo, hi = body_mesh(style).bbox()
    L = landmarks(style)
    pts = np.vstack([L["fore"], L["hind"]])
    r = float(np.max(np.hypot(pts[:, 0] - HINGE_X, 0.0)))
    up = math.radians(min(90.0, P["flap_mid"] + 0.5 * P["flap_amp"]))
    down = math.radians(max(-90.0, P["flap_mid"] - 0.5 * P["flap_amp"]))
    zs = [HINGE_Z + r * math.sin(up), HINGE_Z + r * math.sin(down), lo[2], hi[2]]
    bob = 0.5 * BOB
    out_lo = [-0.5, min(lo[1], float(pts[:, 1].min())), min(zs) - bob]
    out_hi = [0.5, max(hi[1], float(pts[:, 1].max())), max(zs) + bob]
    return {"min": [round(v * size, 5) for v in out_lo], "max": [round(v * size, 5) for v in out_hi]}


def use_points(style, size):
    """The card's use points (the prop's frame, metres): `look` head, body, tip_L, tip_R (the forewings' tips, flat) and
    `anchor` perch (between the feet, pointing down: where it stands on a finger or a flower)."""
    tip = landmarks(style)["fore"]
    k = int(np.argmax(tip[:, 0]))
    lo, _hi = body_mesh(style).bbox()
    s = size
    return {"look": [{"name": "head", "point": [0.0, round(-0.075 * s, 5), round(0.004 * s, 5)]},
                     {"name": "body", "point": [0.0, 0.0, 0.0]},
                     {"name": "tip_L", "point": [round(float(tip[k, 0]) * s, 5), round(float(tip[k, 1]) * s, 5),
                                                 round(HINGE_Z * s, 5)]},
                     {"name": "tip_R", "point": [round(-float(tip[k, 0]) * s, 5), round(float(tip[k, 1]) * s, 5),
                                                 round(HINGE_Z * s, 5)]}],
            "anchor": [{"name": "perch", "point": [0.0, round(0.01 * s, 5), round(float(lo[2]) * s, 5)],
                        "dir": [0.0, 0.0, -1.0]}]}


# ===================================================================================================================
# deterministic numbers
# ===================================================================================================================
def unit(seed, *key):
    """A deterministic number in [0, 1) from a seed and a key (the same on every platform)."""
    h = hashlib.blake2b(("/".join(str(k) for k in (seed,) + key)).encode(), digest_size=8).digest()
    return int.from_bytes(h, "little") / 2.0 ** 64


# ===================================================================================================================
# colours: roles, each a palette slot, a '#hex' or a blend 'slot:weight,slot:weight' (mixed in linear light), where two
# names of its own may stand for slots: `dark` (the palette's darkest slot at DARK of its light: the black of veins and
# borders, in a light palette as in a dark one) and `light` (its lightest slot: white spots and fringes). The defaults
# are fitted on Rose Pine Dawn and recolour with any palette.
# ===================================================================================================================
ROLE_DOCS = {
    "ground": "the upper side's ground colour",
    "under": "the underside's ground colour",
    "ink": "the dark pattern: veins, borders, stripes, eye rings",
    "spot": "the light pattern: edge dots, fringes, pale spots, pupils",
    "accent": "coloured spots: eyespot rings, lunules, red spots",
    "sheen": "the iridescent colour the sheen turns to at a glancing view (`sheen` option)",
    "body": "the body, legs and antennae",
    "eye": "the eyes",
}
ROLES = {
    "swallowtail": {"ground": "gold:1,light:1.1", "under": "gold:1,light:2.4", "ink": "dark", "spot": "gold:1,light:1.1",
                    "accent": "love:2,gold:1", "sheen": "foam:2,pine:1", "body": "dark", "eye": "dark"},
    "monarch": {"ground": "gold", "under": "gold:2,light:1", "ink": "dark", "spot": "light", "accent": "gold:1,light:1.2",
                "sheen": "gold:1,light:1", "body": "dark", "eye": "dark"},
    "morpho": {"ground": "pine:1,foam:1", "under": "gold:1,rose:1,dark:3", "ink": "dark", "spot": "light",
               "accent": "gold", "sheen": "foam:2,iris:1", "body": "dark:3,gold:1", "eye": "dark"},
    "blue": {"ground": "iris", "under": "light:3,subtle:1", "ink": "dark", "spot": "light", "accent": "gold:2,love:1",
             "sheen": "foam:1,iris:1", "body": "iris:1,dark:1", "eye": "dark"},
}
SHEEN = {"swallowtail": 0.0, "monarch": 0.0, "morpho": 1.0, "blue": 0.6}     # the iridescence of each style by default
# other colourways of a style (role overrides), for a swarm of many colours; way 0 is the style's own
WAYS = {
    "swallowtail": ({}, {"ground": "rose:1,light:1", "under": "rose:1,light:2.4", "spot": "rose:1,light:1"},
                    {"ground": "foam:1,light:1.4", "under": "foam:1,light:3", "spot": "foam:1,light:1.4",
                     "accent": "gold"}),
    "monarch": ({}, {"ground": "love", "under": "love:2,light:1", "accent": "rose:1,light:1"},
                {"ground": "rose", "under": "rose:2,light:1"}),
    "morpho": ({}, {"ground": "iris", "sheen": "foam:1,light:1"}, {"ground": "love", "sheen": "rose:1,light:1"}),
    "blue": ({}, {"ground": "foam", "sheen": "foam:1,light:1"}, {"ground": "rose", "sheen": "rose:1,light:1"}),
}
DAWN = PAL.PALETTES["rose-pine-dawn"]
DARK = 0.42                         # `dark`: the darkest slot at this share of its light


def roles(style, slots=None, way=0):
    """{role: colour spec} of a style in colourway `way`, with the prop's `slots` overriding any role."""
    if style not in ROLES:
        raise ValueError(f"butterfly: no style {style!r} (have {', '.join(STYLES)})")
    if not 0 <= int(way) < len(WAYS[style]):
        raise ValueError(f"butterfly: the {style} has colourways 0..{len(WAYS[style]) - 1}, not {way}")
    slots = dict(slots or {})
    out = dict(ROLES[style], **WAYS[style][int(way)])
    out.update({k: str(slots[k]) for k in ROLES[style] if k in slots})
    return out


def _luma(rgb):
    return 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]


def colour(spec, slots=None):
    """Linear RGB of a colour spec: a palette slot (the prop's `slots` palette, else Rose Pine Dawn's), `dark`, `light`,
    a '#hex' or a blend 'name:weight,name:weight' (weights normalised, mixed in linear light)."""
    palette = dict(DAWN, **{k: v for k, v in (slots or {}).items() if k in DAWN and isinstance(v, str)})
    lins = [PAL.linear(v) for v in palette.values()]
    special = {"dark": tuple(DARK * c for c in min(lins, key=_luma)), "light": max(lins, key=_luma)}
    out, tot = [0.0, 0.0, 0.0], 0.0
    for part in str(spec).split(","):
        name, sep, w = part.strip().rpartition(":")
        try:
            name, weight = (name, float(w)) if sep else (w, 1.0)
        except ValueError:
            raise ValueError(f"butterfly: colour {spec!r}: {part.strip()!r} is not 'name' or 'name:weight'") from None
        rgb = special[name] if name in special else PAL.linear(PAL.resolve(name, palette))
        out = [o + weight * c for o, c in zip(out, rgb)]
        tot += weight
    if tot <= 0:
        raise ValueError(f"butterfly: colour {spec!r} has no weight")
    return tuple(o / tot for o in out)


# ===================================================================================================================
# outlines: landmarks read off photographs of the real butterflies spread flat (pixels of the photograph: x toward the
# butterfly's left, y toward its tail, from the middle of its thorax; `half` is the half span in pixels), never traced.
# Both wings are then turned about the forewing's root until the forewing's hind edge (root to `tornus`, its hind
# corner) lies `sweep` degrees behind square to the body (the hindwing by `hind_turn` of that turn: a set specimen's
# hindwings come forward less than its forewings), and the whole is scaled to unit span. A wing's points run
# from its root's front corner out along the leading edge (the costa), round the tip, along the outer edge and back along
# the hind edge to its root's rear corner; the first and last point are corners. The hindwing's leading edge lies under
# the forewing (its first points are hidden).
# ===================================================================================================================
_MEASURED = {
    # a monarch: an engraving of 1907 in the public domain (a set specimen, top view)
    "monarch": {"half": 508.0, "tornus": 14, "sweep": 12.0, "hind_turn": 1.0, "emerge": 5,
                "fore": [(28, -17), (98, -82), (158, -162), (218, -222), (298, -267), (378, -297), (438, -310),
                         (488, -310), (508, -290), (498, -247), (478, -192), (448, -122), (418, -52), (392, -2),
                         (368, 18), (238, 18), (118, 13), (28, 8)],
                "hind": [(28, 6), (138, -22), (288, -14), (385, -12), (402, 40), (398, 68), (408, 118), (400, 158), (378, 200),
                         (348, 240), (298, 280), (238, 310), (178, 328), (128, 334), (98, 308), (68, 238), (43, 148),
                         (30, 70), (26, 34)]},
    # a blue morpho basking, top view (CC BY 3.0 photograph)
    "morpho": {"half": 522.0, "tornus": 13, "sweep": 16.0, "hind_turn": 0.6, "emerge": 5,
               "fore": [(19, -20), (62, -46), (125, -59), (200, -64), (269, -65), (375, -53), (450, -21), (500, 10),
                        (522, 54), (512, 97), (469, 129), (406, 166), (356, 204), (319, 247), (200, 168), (100, 92),
                        (12, 25)],
               "hind": [(10, 5), (150, 30), (265, 85), (312, 165), (318, 230), (312, 254), (294, 285), (269, 322), (231, 360),
                        (187, 404), (137, 429), (87, 435), (37, 422), (12, 385), (8, 250), (8, 120), (10, 40)]},
    # a tiger swallowtail basking, top view (GFDL photograph; the tail drawn to a male's length)
    "swallowtail": {"half": 596.0, "tornus": 11, "sweep": 16.0, "hind_turn": 0.35, "emerge": 5,
                    "fore": [(20, -35), (128, -50), (271, -50), (378, -21), (485, 29), (556, 71), (596, 129),
                             (585, 157), (485, 186), (385, 221), (285, 254), (271, 261), (150, 150), (60, 60),
                             (18, 20)],
                    "hind": [(15, 8), (110, 10), (215, 35), (272, 100), (284, 190), (276, 250), (268, 290), (250, 318),
                             (222, 338), (198, 352), (194, 400), (192, 440), (180, 456), (166, 446), (162, 400),
                             (156, 362), (120, 372), (82, 378), (45, 388), (22, 382), (12, 330), (8, 200), (10, 90),
                             (12, 40)]},
    # a male common blue basking, top view (CC BY-SA 3.0 photograph)
    "blue": {"half": 403.0, "tornus": 11, "sweep": 16.0, "hind_turn": 0.6, "emerge": 4, "tuck": False,
             "fore": [(20, -12), (90, -45), (170, -60), (250, -63), (330, -57), (380, -43), (403, -25), (400, 5),
                      (380, 45), (350, 75), (322, 98), (298, 111), (270, 119), (237, 128), (209, 131), (181, 122),
                      (148, 108), (114, 92), (81, 69), (55, 50), (18, 12)],
             "hind": [(15, 8), (100, 45), (180, 92), (212, 124), (214, 136), (228, 186), (231, 231), (220, 264),
                      (203, 286), (153, 303), (92, 306), (31, 297), (10, 275), (5, 175), (10, 80), (12, 25)]},
}
# waves of an outer edge: [(first landmark, last landmark, waves, depth as a share of the span), ...] per wing
_SCALLOPS = {
    "monarch": {"hind": [(5, 13, 7, 0.004)]},
    "morpho": {"fore": [(9, 13, 5, 0.003)], "hind": [(5, 12, 7, 0.006)]},
    "swallowtail": {"fore": [(7, 11, 6, 0.002)], "hind": [(5, 9, 3, 0.012), (15, 18, 2, 0.010)]},
    "blue": {},
}
OVERLAP = 0.035                     # the hindwing's hidden leading edge lies this far ahead of the forewing's hind edge
HINGE_X = 0.012                     # the hinge line (x of both wing roots' axis) and its height on the thorax
HINGE_Z = 0.006
FORE_LIFT = 0.006                   # the forewing lies this much over the hindwing where they overlap
THICK = 0.0035                      # the membrane's thickness (unit span: 0.3 mm at 9 cm)
CAMBER = 0.03                       # the wing curves up toward its tip: z = CAMBER x^2


def _cr_segment(P0, P1, P2, P3, t, alpha=0.5):
    """Centripetal Catmull-Rom between P1 and P2 at parameters t (m,) in 0..1 -> (m, 2) (Barry-Goldman)."""
    t0 = 0.0
    t1 = t0 + max(float(np.linalg.norm(P1 - P0)), 1e-12) ** alpha
    t2 = t1 + max(float(np.linalg.norm(P2 - P1)), 1e-12) ** alpha
    t3 = t2 + max(float(np.linalg.norm(P3 - P2)), 1e-12) ** alpha
    u = (t1 + (t2 - t1) * np.asarray(t, float))[:, None]
    A1 = (t1 - u) / (t1 - t0) * P0 + (u - t0) / (t1 - t0) * P1
    A2 = (t2 - u) / (t2 - t1) * P1 + (u - t1) / (t2 - t1) * P2
    A3 = (t3 - u) / (t3 - t2) * P2 + (u - t2) / (t3 - t2) * P3
    B1 = (t2 - u) / (t2 - t0) * A1 + (u - t0) / (t2 - t0) * A2
    B2 = (t3 - u) / (t3 - t1) * A2 + (u - t1) / (t3 - t1) * A3
    return (t2 - u) / (t2 - t1) * B1 + (u - t1) / (t2 - t1) * B2


def _open_curve(Q, per=24):
    """Dense points of the Catmull-Rom curve through Q (n, 2), the end points included (reflected end tangents)."""
    Q = np.asarray(Q, float)
    ext = np.vstack([2 * Q[0] - Q[1], Q, 2 * Q[-1] - Q[-2]])
    t = np.linspace(0.0, 1.0, per, endpoint=False)
    return np.concatenate([_cr_segment(ext[i], ext[i + 1], ext[i + 2], ext[i + 3], t) for i in range(len(Q) - 1)]
                          + [Q[-1:]])


def _resample(P, n):
    """n points evenly spaced by arc length along the open polyline P (both ends kept)."""
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))])
    q = np.linspace(0.0, s[-1], n)
    return np.stack([np.interp(q, s, P[:, 0]), np.interp(q, s, P[:, 1])], 1)


def _rot2(P, centre, deg):
    """Points (n, 2) turned `deg` forward (toward -y, the head) about `centre`, seen from above with x out."""
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    d = np.asarray(P, float) - centre
    return np.stack([c * d[:, 0] + s * d[:, 1], -s * d[:, 0] + c * d[:, 1]], 1) + centre


def landmarks(style):
    """The posed landmarks {"fore": (n, 2), "hind": (m, 2)} of a style at unit span (x out from the body's axis, y toward
    the tail): the photograph's points turned so the forewing's hind edge lies `sweep` degrees behind square, scaled so
    that the farthest point lies at x = 0.5."""
    if style not in _MEASURED:
        raise ValueError(f"butterfly: no style {style!r} (have {', '.join(STYLES)})")
    m = _MEASURED[style]
    F, H = (np.asarray(m[w], float) / (2.0 * m["half"]) for w in ("fore", "hind"))
    root = 0.5 * (F[0] + F[-1])
    edge = F[m["tornus"]] - root
    turn = math.degrees(math.atan2(edge[1], edge[0])) - m["sweep"]
    out = {"fore": _rot2(F, root, turn), "hind": _rot2(H, root, turn * m["hind_turn"])}
    # the hindwing's hidden leading edge (landmarks 1 .. emerge - 1) runs OVERLAP ahead of the forewing's hind edge, so
    # the wings overlap all the way to the forewing's hind corner, whatever the pose
    Fp, Hp, e = out["fore"], out["hind"].copy(), m["emerge"]
    edge = Fp[m["tornus"]:][::-1]                                   # the forewing's hind edge, from its root out
    s_edge = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(edge, axis=0), axis=1))])
    for k in range(1, e):
        q = s_edge[-1] * k / e
        Hp[k] = [np.interp(q, s_edge, edge[:, 0]), np.interp(q, s_edge, edge[:, 1]) - OVERLAP]
    if m.get("tuck", True):                                         # it comes out right behind the forewing's corner
        Hp[e] = Fp[m["tornus"]] + [0.004, 0.012]
    out["hind"] = Hp
    scale = 0.5 / max(float(_open_curve(v)[:, 0].max()) for v in out.values())    # the curve's tip, not a landmark's
    return {k: v * scale for k, v in out.items()}


def landmark_s(style, wing):
    """Where each landmark of a wing lies along its outline: arc length from the root's front corner as a share of the
    curve to the rear corner (the straight root edge is not counted; `outline`'s first n - 1 points are evenly spaced
    from 0 to 1 in it)."""
    L = landmarks(style)[wing]
    dense = _open_curve(L, 24)
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(dense, axis=0), axis=1))])
    return s[np.arange(len(L)) * 24] / s[-1]


def outline(style, wing, n=240):
    """The closed outline (n, 2) of a style's `wing` ("fore" or "hind") at unit span, counter-clockwise seen from above
    (x out, y toward the tail: the left wing seen from its back), the root's corners kept sharp and the outer edge waved
    (scallops: round lobes with notches between). Its first point is the root's front corner, its last but one the rear
    corner (the last is the middle of the straight root edge)."""
    L = landmarks(style)[wing]
    per = 24
    dense = _open_curve(L, per)
    pts = _resample(dense, n - 1)
    s_dense = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(dense, axis=0), axis=1))])
    s_p = np.linspace(0.0, s_dense[-1], n - 1)                         # where the even samples lie along the curve
    bite = np.zeros(n - 1)
    for a, b, waves, depth in _SCALLOPS[style].get(wing, ()):
        s_a, s_b = s_dense[a * per], s_dense[b * per]                 # landmark i is dense point i * per
        u = (s_p - s_a) / max(s_b - s_a, 1e-12)
        bite += np.where((u > 0) & (u < 1), depth * (1.0 - np.abs(np.sin(math.pi * waves * u))) ** 2, 0.0)
    pts = pts + bite[:, None] * _inward(pts)
    P = np.vstack([pts, 0.5 * (L[-1] + L[0])])
    if S.polygon_area(P) <= 0:
        raise ValueError(f"butterfly: the {style} {wing}wing's landmarks do not run counter-clockwise")
    return P


def _inward(P):
    """Unit normals of an open polyline, to its left walking along it: inward for a counter-clockwise outline."""
    t = np.gradient(P, axis=0)
    t /= np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-12)
    return np.stack([-t[:, 1], t[:, 0]], 1)


# ===================================================================================================================
# the wings as meshes: a membrane THICK thick with a rolled edge, its two faces one triangulation of the outline (a
# Delaunay triangulation of the edge's points and a lattice inside: a tail or a deep scallop is no harder than a round
# wing), curving up toward the tip; uv in the style's pattern atlas
# ===================================================================================================================
RIM = (0.7071, 0.0, -0.7071)        # the rolled edge between the faces: heights (shares of half the thickness) of its rings
LATTICE = 0.016                     # the faces' inner points lie this far apart (unit span)


def _circles(pts, T):
    a, b, c = pts[T[:, 0]], pts[T[:, 1]], pts[T[:, 2]]
    d = 2.0 * (a[:, 0] * (b[:, 1] - c[:, 1]) + b[:, 0] * (c[:, 1] - a[:, 1]) + c[:, 0] * (a[:, 1] - b[:, 1]))
    d = np.where(np.abs(d) < 1e-18, 1e-18, d)
    a2, b2, c2 = (a * a).sum(1), (b * b).sum(1), (c * c).sum(1)
    ux = (a2 * (b[:, 1] - c[:, 1]) + b2 * (c[:, 1] - a[:, 1]) + c2 * (a[:, 1] - b[:, 1])) / d
    uy = (a2 * (c[:, 0] - b[:, 0]) + b2 * (a[:, 0] - c[:, 0]) + c2 * (b[:, 0] - a[:, 0])) / d
    U = np.stack([ux, uy], 1)
    return U, ((a - U) ** 2).sum(1)


def delaunay(P):
    """Delaunay triangles (k, 3) of the points P (n, 2), counter-clockwise (Bowyer-Watson: each point cuts out the triangles
    whose circumcircle holds it and fans the hole to it)."""
    P = np.asarray(P, float)
    n = len(P)
    c = 0.5 * (P.min(0) + P.max(0))
    r = max(float(np.ptp(P, axis=0).max()), 1e-9) * 1000.0
    pts = np.vstack([P, [c[0] - r, c[1] - 0.7 * r], [c[0] + r, c[1] - 0.7 * r], [c[0], c[1] + r]])
    T = np.array([[n, n + 1, n + 2]], np.int64)
    CC, R2 = _circles(pts, T)
    for i in range(n):
        d = CC - pts[i]
        bad = (d * d).sum(1) < R2 * (1.0 - 1e-10)
        tb = T[bad]
        E = np.concatenate([tb[:, [0, 1]], tb[:, [1, 2]], tb[:, [2, 0]]])
        key = np.minimum(E[:, 0], E[:, 1]) * (n + 3) + np.maximum(E[:, 0], E[:, 1])
        _, first, count = np.unique(key, return_index=True, return_counts=True)
        edge = E[first[count == 1]]
        new = np.concatenate([edge, np.full((len(edge), 1), i, np.int64)], 1)
        nc, nr = _circles(pts, new)
        keep = ~bad
        T, CC, R2 = np.concatenate([T[keep], new]), np.concatenate([CC[keep], nc]), np.concatenate([R2[keep], nr])
    return T[(T < n).all(1)]


def _inside_poly(Q, P):
    """Even-odd: which points Q (q, 2) lie inside the closed polygon P (n, 2)."""
    Q = np.asarray(Q, float)
    a, b = P, np.roll(P, -1, axis=0)
    out = np.zeros(len(Q), bool)
    for (ax, ay), (bx, by) in zip(a, b):
        hit = (ay > Q[:, 1]) != (by > Q[:, 1])
        x = ax + (Q[:, 1] - ay) * (bx - ax) / np.where(by == ay, 1e-18, by - ay)
        out ^= hit & (Q[:, 0] < x)
    return out


def _edge_distance(Q, P):
    """Distance from points Q (q, 2) to the edges of the closed polygon P (n, 2)."""
    a, b = P, np.roll(P, -1, axis=0)
    ab = b - a
    t = np.clip(((Q[:, None, :] - a[None]) * ab[None]).sum(-1) / np.maximum((ab * ab).sum(-1), 1e-18)[None], 0.0, 1.0)
    return np.linalg.norm(Q[:, None, :] - a[None] - t[..., None] * ab[None], axis=-1).min(1)


def fill(ring, spacing=LATTICE, margin=0.7):
    """Triangles filling the closed CCW polygon `ring` (m, 2): (V (n, 2), T (k, 3)). The first m points are the ring itself
    (every ring edge is a triangle edge: keep the ring's spacing well under `spacing`), the others a triangular lattice
    inside it at least `margin * spacing` clear of it. Counter-clockwise."""
    ring = np.asarray(ring, float)
    lo, hi = ring.min(0), ring.max(0)
    h = spacing * math.sqrt(3.0) / 2.0
    rows = []
    for k, y in enumerate(np.arange(lo[1], hi[1] + h, h)):
        xs = np.arange(lo[0] + (0.5 * spacing if k % 2 else 0.0), hi[0] + spacing, spacing)
        rows.append(np.stack([xs, np.full(len(xs), y)], 1))
    Q = np.concatenate(rows)
    Q = Q + (np.random.default_rng(7).random(Q.shape) - 0.5) * spacing * 1e-3     # no four points on one circle
    Q = Q[_inside_poly(Q, ring)]
    Q = Q[_edge_distance(Q, ring) >= margin * spacing]
    V = np.vstack([ring, Q])
    T = delaunay(V)
    return V, T[_inside_poly(V[T].mean(1), ring)]


def camber(x):
    """The wing's height over its hinge plane at distance x out from the body's axis: it curves up toward the tip."""
    return CAMBER * x * x


def atlas(style):
    """The pattern atlas of a style: {"fore": (x0, y0, size), "hind": (x0, y0, size)}, each wing's square box in the
    body's frame (unit span; y toward the tail). The atlas image is 2 x 2 tiles: the forewing left, the hindwing right,
    the upper side (dorsal) on top and the underside (ventral) below; a tile's top row is the box's front edge."""
    out = {}
    for wing in ("fore", "hind"):
        P = outline(style, wing)
        lo, hi = P.min(0), P.max(0)
        size = float((hi - lo).max()) * 1.04
        mid = 0.5 * (lo + hi)
        out[wing] = (float(mid[0] - 0.5 * size), float(mid[1] - 0.5 * size), size)
    return out


def atlas_uv(style, wing, xy):
    """uv (k, 2) in the atlas of points xy (k, 2) of a wing (the dorsal tiles; the underside is v - 0.5)."""
    x0, y0, size = atlas(style)[wing]
    xy = np.asarray(xy, float)
    tile = 0.0 if wing == "fore" else 1.0
    u = (tile + np.clip((xy[:, 0] - x0) / size, 0.0, 1.0)) * 0.5
    v = 0.5 + 0.5 * np.clip(1.0 - (xy[:, 1] - y0) / size, 0.0, 1.0)
    return np.stack([u, v], 1)


def _wing_surface(style, wing, n=200):
    """(V (k, 3) in the body's frame, Q, T, uv (k, 2), dorsal (k,)) of one closed wing membrane: its upper face, the rolled
    edge round the outline, its underside."""
    P = outline(style, wing, n)
    h = 0.5 * THICK
    lift = FORE_LIFT if wing == "fore" else 0.0
    nrm = _inward(np.vstack([P[-1:], P, P[:1]]))[1:-1]          # the closed outline's inward normals
    F, Tf = fill(P + h * nrm)                                    # the faces' edge ring (n points) and their inside
    m = len(P)

    def lift_to(xy, dz):
        return np.column_stack([xy, HINGE_Z + lift + camber(xy[:, 0]) + dz])

    rings = [lift_to(F[:m], h)]                                  # the upper face's edge ring, then round the edge
    flags = [1.0]
    for r in RIM:
        phi = math.asin(r)
        rings.append(lift_to(P + h * (1.0 - math.cos(phi)) * nrm, h * r))
        flags.append(1.0 if r > 0 else (0.5 if r == 0 else 0.0))
    rings.append(lift_to(F[:m], -h))                             # the underside's edge ring
    flags.append(0.0)
    R = np.stack(rings)                                          # (K, m, 3)
    K = len(R)
    idx = np.arange(K * m).reshape(K, m)
    j, j2 = np.arange(m), (np.arange(m) + 1) % m
    Q = np.concatenate([np.stack([idx[k][j], idx[k + 1][j], idx[k + 1][j2], idx[k][j2]], 1) for k in range(K - 1)])
    inner = F[m:]
    base = K * m
    top_ids = np.concatenate([idx[0], base + np.arange(len(inner))])
    bot_ids = np.concatenate([idx[K - 1], base + len(inner) + np.arange(len(inner))])
    T = np.concatenate([top_ids[Tf], bot_ids[Tf][:, ::-1]])
    V = np.concatenate([R.reshape(-1, 3), lift_to(inner, h), lift_to(inner, -h)])
    dorsal = np.concatenate([np.repeat(flags, m), np.ones(len(inner)), np.zeros(len(inner))])
    uv = atlas_uv(style, wing, V[:, :2])
    return V, Q, T, uv, dorsal


def wing_mesh(style, side="L"):
    """One side's wings (fore over hind) as a closed `Mesh` in the wing's own frame: the hinge at the origin, its axis along
    Y, the wing out along +X for the left side ("L") and -X for the right ("R"); uv in the atlas. Returns (mesh, dorsal):
    dorsal (k,) per vertex, 1 on the upper side, 0 on the underside, 0.5 round the edge."""
    Vs, Qs, Ts, UVs, Ds, off = [], [], [], [], [], 0
    for wing in ("hind", "fore"):
        V, Q, T, uv, d = _wing_surface(style, wing)
        Vs.append(V)
        Qs.append(Q + off)
        Ts.append(T + off)
        UVs.append(uv)
        Ds.append(d)
        off += len(V)
    V = np.concatenate(Vs) - np.array([HINGE_X, 0.0, HINGE_Z])
    m = S.Mesh(V, np.concatenate(Qs), np.concatenate(Ts), UV=np.concatenate(UVs))
    dorsal = np.concatenate(Ds)
    if side == "R":
        m = m.mirrored_x()
    elif side != "L":
        raise ValueError(f"butterfly: side is 'L' or 'R', not {side!r}")
    return m, dorsal


def hinge(side="L"):
    """The hinge point of a side's wings in the butterfly's frame (unit span)."""
    return (HINGE_X if side == "L" else -HINGE_X, 0.0, HINGE_Z)


# ===================================================================================================================
# the body: head, thorax and a ringed abdomen turned on a lathe along Y, eyes, clubbed antennae, legs and the coiled
# proboscis (material 0 the body, 1 the eyes)
# ===================================================================================================================
# (radius, position along the body toward the head) from the abdomen's tip to the front of the head
_BODY = ((0.0, -0.265), (0.0035, -0.262), (0.0068, -0.252), (0.0098, -0.232), (0.0122, -0.2), (0.0138, -0.16),
         (0.0146, -0.12), (0.0146, -0.08), (0.0138, -0.055), (0.0118, -0.042), (0.0128, -0.036), (0.0172, -0.026),
         (0.0205, -0.01), (0.0214, 0.006), (0.0204, 0.024), (0.0172, 0.038), (0.0128, 0.047), (0.0118, 0.053),
         (0.0132, 0.061), (0.0136, 0.069), (0.0122, 0.078), (0.0086, 0.085), (0.0, 0.0885))
_SEGMENTS = (-0.255, -0.045, 7)       # the abdomen's rings: from, to, how many
LEGS = {"swallowtail": 3, "blue": 3, "monarch": 2, "morpho": 2}     # brush-footed butterflies walk on four legs
EYE_R = 0.0095


def _body_profile():
    P = np.asarray(_BODY, float)
    z = np.linspace(P[0, 1], P[-1, 1], 220)
    r = np.interp(z, P[:, 1], P[:, 0])
    a, b, k = _SEGMENTS
    u = np.clip((z - a) / (b - a), 0.0, 1.0)
    ring = np.where((z > a) & (z < b), 0.06 * np.cos(2.0 * math.pi * k * u) - 0.03, 0.0)   # a groove between rings
    r = r * (1.0 + ring)
    r[0] = r[-1] = 0.0
    return np.stack([r, z], 1)


def _along_y(mesh):
    """A mesh turned about X so its +Z (the lathe's axis) points along -Y (the head)."""
    return mesh.rot(rx=90.0)


def _antenna(sx):
    t = np.linspace(0.0, 1.0, 28)
    p0, p1, p2 = np.array([0.0045 * sx, -0.083, 0.009]), np.array([0.026 * sx, -0.16, 0.05]), \
        np.array([0.062 * sx, -0.235, 0.066])
    path = ((1 - t) ** 2)[:, None] * p0 + (2 * (1 - t) * t)[:, None] * p1 + (t ** 2)[:, None] * p2
    club = np.clip((t - 0.78) / 0.22, 0.0, 1.0)
    r = 0.0015 + 0.0032 * np.sin(math.pi * np.clip(club * 0.62, 0.0, 1.0)) ** 1.2
    r[-1] = 0.0012
    return S.tube(path, r, sides=8)


def _leg(sx, y0, reach):
    """A leg from under the thorax: out and down at the knee, back in a little to the foot (y: forward is negative)."""
    knots = np.array([[0.010 * sx, y0, -0.012], [0.042 * sx, y0 + 0.4 * reach, -0.030],
                      [0.050 * sx, y0 + 0.8 * reach, -0.062], [0.047 * sx, y0 + 1.1 * reach, -0.084]])
    path = _open_curve3(knots, 8)
    r = np.linspace(0.0021, 0.0011, len(path))
    return S.tube(path, r, sides=6)


def _open_curve3(K, per):
    """A Catmull-Rom curve through 3D knots (uniform; the ends included)."""
    K = np.asarray(K, float)
    ext = np.vstack([2 * K[0] - K[1], K, 2 * K[-1] - K[-2]])
    t = np.linspace(0.0, 1.0, per, endpoint=False)[:, None]
    out = []
    for i in range(len(K) - 1):
        p0, p1, p2, p3 = ext[i], ext[i + 1], ext[i + 2], ext[i + 3]
        out.append(0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t ** 2
                          + (-p0 + 3 * p1 - 3 * p2 + p3) * t ** 3))
    return np.concatenate(out + [K[-1:]])


def _proboscis():
    """The coiled tongue under the head: two turns of a flat spiral in the y-z plane."""
    a = np.linspace(0.0, 4.0 * math.pi, 60)
    r = 0.0085 * (1.0 - a / (4.6 * math.pi))
    path = np.stack([np.zeros_like(a), -0.079 - r * np.sin(a), -0.016 + r * np.cos(a)], 1)
    return S.tube(path, np.linspace(0.0012, 0.0008, len(a)), sides=6)


def body_mesh(style):
    """The body at unit span in the butterfly's frame (closed parts; material 0 the body, 1 the eyes)."""
    parts = [_along_y(S.lathe(_body_profile(), seg=24))]
    for sx in (-1.0, 1.0):
        eye = S.lathe(np.stack([EYE_R * np.sin(np.linspace(0, math.pi, 9)),
                                -EYE_R * np.cos(np.linspace(0, math.pi, 9))], 1), seg=16, mat=1)
        parts.append(eye.moved((0.0105 * sx, -0.071, 0.003)))
        parts.append(_antenna(sx))
        for k, (y0, reach) in enumerate(((-0.022, -0.035), (-0.004, 0.012), (0.014, 0.05))):
            if k == 0 and LEGS[style] == 2:
                continue                                        # the forelegs of a brush-foot are folded up small
            parts.append(_leg(sx, y0, reach))
    parts.append(_proboscis())
    return S.merge(parts)


@functools.cache
def parts(style):
    """Everything a builder needs of a style at unit span (cached: treat it as read-only): {"body": Mesh, "wing_L": Mesh, "wing_R": Mesh, "hinge_L",
    "hinge_R": (x, y, z), "attrs": {"wing_L": {"mk_dorsal": (k,)}, "wing_R": {...}}} (see `wing_mesh`, `body_mesh`)."""
    wl, dl = wing_mesh(style, "L")
    wr, dr = wing_mesh(style, "R")
    return {"body": body_mesh(style), "wing_L": wl, "wing_R": wr, "hinge_L": hinge("L"), "hinge_R": hinge("R"),
            "attrs": {"wing_L": {"mk_dorsal": dl}, "wing_R": {"mk_dorsal": dr}}}
