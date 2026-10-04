"""The parts of the single bed's frame: boxes and profiles (pure numpy through the shell toolkit, no bpy, importable from
tests), and the meshes made from them. Every board is a `mkmmd.core.shell.rounded_box` with real fillets (never a
hard-edged cuboid), every post a rounded-square section swept up a tapered leg with rounded ends.

Each part is a dict: name, lo, hi (corners, metres), group (post | rail | board | slat | cleat | pad), round (the fillet
radius of its edges, metres), seg (k of `rounded_box`: 2k segments per corner arc) and, for posts, taper = (k, h) (the
leg is k times as wide at the floor as from z = lo.z + h upward) and end (the round-over of its two ends). Boards run
`LAP` into the posts (hidden tenons) and flush faces are inset by `INSET`, so no two faces of different parts share a
plane.

Proportions of planed pine furniture: posts 55 mm square with soft corners and tapered legs, side rails 25 x 120 mm,
head and foot rails 30 mm thick and 65-80 mm high, slats 20 x 42 mm; every arris is a 9-14 mm round-over, the way a
bullnose plane leaves a rail."""
import math

import numpy as np

from ....core import shell as CS
from . import bedroom_soft_geo as G

LAP = 0.010                  # a board's depth into a post
INSET = 0.0005               # flush faces stand this far inside the neighbouring face
SLAT_W, SLAT_GAP, SLAT_R = 0.042, 0.050, 0.0092
POST_R = 0.016              # corner radius of a post's square section
LEG_TAPER = (0.78, 0.18)     # a post is 78 % as wide at the floor, straight above z = 0.18


def _box(name, lo, hi, group, rnd=0.0, seg=2, taper=None, end=0.0):
    part = {"name": name, "lo": tuple(float(v) for v in lo), "hi": tuple(float(v) for v in hi), "group": group,
            "round": rnd, "seg": seg}
    if taper:
        part["taper"] = tuple(taper)
    if end:
        part["end"] = end
    return part


def slat_xs(x0, x1, width, gap):
    """Centres of the slats that fill [x0, x1] evenly: as many of `width` as fit with about `gap` between them, the same
    gap at both ends."""
    n = max(1, int(round((x1 - x0 - gap) / (width + gap))))
    g = (x1 - x0 - n * width) / (n + 1)
    return [x0 + g + width / 2 + i * (width + g) for i in range(n)]


def frame_parts(headboard="slats"):
    """All the wooden parts of the frame, in the bed's frame. The head is at +Y. headboard = "padded" leaves out the
    slatted board's rails and slats (`padded_slab()` takes their place)."""
    P, W = [], G.POST_W
    hx, hy = G.FRAME_HX, G.FRAME_HY
    inner = hx - W                                               # between the posts
    ex = inner + LAP
    # posts: soft-cornered squares, legs tapering toward the floor, rounded ends; the head posts stand proud of the board
    for sx, xn in ((-1, "L"), (1, "R")):
        x0, x1 = sorted((sx * (hx - W), sx * hx))
        P.append(_box(f"post_foot_{xn}", (x0, -hy, 0.0), (x1, -hy + W, G.FOOT_TOP), "post", POST_R, 2, LEG_TAPER, 0.005))
        P.append(_box(f"post_head_{xn}", (x0, hy - W, 0.0), (x1, hy, G.HEAD_TOP), "post", POST_R, 2, LEG_TAPER, 0.005))
    # side rails: 25 x 120 mm planks on edge, the mattress rests on their top edge
    for sx, xn in ((-1, "L"), (1, "R")):
        x0, x1 = sorted((sx * (hx - G.RAIL_T), sx * (hx - INSET)))
        P.append(_box(f"rail_{xn}", (x0, -hy + W - LAP, G.RAIL_Z0), (x1, hy - W + LAP, G.RAIL_Z1), "rail", 0.0122, 2))
    # headboard: a top rail below the post tops, a mid and a base rail, slats between the top and the mid rail
    yh0, yh1 = hy - G.BOARD_T, hy - INSET
    if headboard != "padded":
        P.append(_box("head_rail_top", (-ex, yh0, 0.845), (ex, yh1, 0.925), "board", 0.0135, 2))
        P.append(_box("head_rail_mid", (-ex, yh0, 0.400), (ex, yh1, 0.465), "board", 0.0115, 2))
        for i, x in enumerate(slat_xs(-inner, inner, SLAT_W, SLAT_GAP)):
            P.append(_box(f"head_slat_{i}", (x - SLAT_W / 2, yh0 + 0.005, 0.465 - LAP),
                          (x + SLAT_W / 2, yh1 - 0.005, 0.845 + LAP), "slat", SLAT_R, 2))
    P.append(_box("head_rail_base", (-ex, yh0, 0.200), (ex, yh1, 0.265), "board", 0.0115, 2))
    # footboard: the top rail flush with the post tops, a base rail, slats between
    yf0, yf1 = -hy + INSET, -hy + G.BOARD_T
    P.append(_box("foot_rail_top", (-ex, yf0, 0.470), (ex, yf1, G.FOOT_TOP), "board", 0.0135, 2))
    P.append(_box("foot_rail_base", (-ex, yf0, 0.200), (ex, yf1, 0.265), "board", 0.0115, 2))
    for i, x in enumerate(slat_xs(-inner, inner, SLAT_W, SLAT_GAP)):
        P.append(_box(f"foot_slat_{i}", (x - SLAT_W / 2, yf0 + 0.005, 0.265 - LAP),
                      (x + SLAT_W / 2, yf1 - 0.005, 0.470 + LAP), "slat", SLAT_R, 2))
    # the base under the mattress: cleats on the rails and cross slats (only a low camera sees them)
    for sx, xn in ((-1, "L"), (1, "R")):
        x0, x1 = sorted((sx * (hx - G.RAIL_T - 0.030), sx * (hx - G.RAIL_T + LAP)))
        P.append(_box(f"cleat_{xn}", (x0, -hy + W - LAP, 0.240), (x1, hy - W + LAP, 0.270), "cleat", 0.0125, 2))
    n = 9
    for i in range(n):
        y = -0.84 + i * (1.68 / (n - 1))
        P.append(_box(f"base_slat_{i}", (-hx + G.RAIL_T - LAP, y - 0.035, 0.270),
                      (hx - G.RAIL_T + LAP, y + 0.035, G.MAT_Z0 - 0.001), "cleat", 0.0135, 2))
    return P


def finial_profile(scale=1.0):
    """Turned finial: (r, z) points of the lathe profile, z from the top of the post (0): a collar, a cove, a neck and a
    ball. 0.052 m tall at scale 1. The first point is on the axis, below the post top (it sits in the post)."""
    prof = [(0.0, -0.004), (0.0195, -0.004), (0.022, -0.0015), (0.022, 0.001), (0.0205, 0.0035), (0.0165, 0.0055),
            (0.0115, 0.0072), (0.0088, 0.0105), (0.0080, 0.0150), (0.0080, 0.0200)]
    cz, br = 0.035, 0.017                                         # the ball: centre height, radius
    for t in range(-54, 91, 18):
        a = math.radians(t)
        prof.append((br * math.cos(a) if t < 90 else 0.0, cz + br * math.sin(a)))
    return [(r * scale, z * scale) for r, z in prof]


def finials():
    """(name, base centre) of the turned finials on the head posts of the wooden frame."""
    return [(f"finial_{xn}", (sx * (G.FRAME_HX - G.POST_W / 2), G.FRAME_HY - G.POST_W / 2, G.HEAD_TOP))
            for sx, xn in ((-1, "L"), (1, "R"))]


def padded_slab():
    """The padded headboard: one upholstered slab between the head posts (bevel radius = its `round`)."""
    inner = G.FRAME_HX - G.POST_W
    return _box("head_pad", (-inner, G.FRAME_HY - 0.075, 0.38), (inner, G.FRAME_HY - 0.004, G.HEAD_TOP - 0.01), "pad",
                0.032, 4)


def tube_parts(headboard="slats"):
    """The tubular steel frame: [(name, path points, radius)] for the posts, the arched head rail, the low rails and the
    bars of the head and foot (the head bars only for headboard = "slats"); the side rails and the base are the wooden
    version's boxes (`tube_boxes`). The foot's tubes stay under FOOT_TOP."""
    import numpy as np
    r_post = 0.016
    cx, cy = G.FRAME_HX - r_post - 0.004, G.FRAME_HY - r_post - 0.004
    T = []
    for sx, xn in ((-1, "L"), (1, "R")):
        T.append((f"post_foot_{xn}", [(sx * cx, -cy, 0.0), (sx * cx, -cy, G.FOOT_TOP - 0.004)], r_post))
        T.append((f"post_head_{xn}", [(sx * cx, cy, 0.0), (sx * cx, cy, G.HEAD_TOP - 0.02)], r_post))
    z_arch = lambda x: 0.905 + 0.035 * (1 - (x / cx) ** 2)                       # noqa: E731
    if headboard != "padded":
        T.append(("head_rail_top", [(float(x), cy, z_arch(x)) for x in np.linspace(-cx, cx, 25)], 0.0125))
        T.append(("head_rail_low", [(-cx, cy, 0.44), (cx, cy, 0.44)], 0.011))
        for i, x in enumerate(np.linspace(-cx + 0.07, cx - 0.07, 13)):
            T.append((f"head_bar_{i}", [(float(x), cy, 0.44), (float(x), cy, z_arch(float(x)))], 0.006))
    T.append(("foot_rail_top", [(-cx, -cy, 0.520), (cx, -cy, 0.520)], 0.0125))
    T.append(("foot_rail_low", [(-cx, -cy, 0.25), (cx, -cy, 0.25)], 0.011))
    for i, x in enumerate(np.linspace(-cx + 0.07, cx - 0.07, 13)):
        T.append((f"foot_bar_{i}", [(float(x), -cy, 0.25), (float(x), -cy, 0.520)], 0.006))
    return T


def tube_boxes():
    """Boxes of the steel frame: the side rails (flat bars) and the base under the mattress, as in the wooden frame."""
    return [p for p in frame_parts() if p["group"] in ("rail", "cleat")]


TUBE_FINIAL_SCALE = 0.6


def tube_finials():
    """(name, base centre, scale of finial_profile) of the finials on the head tubes of the steel frame."""
    r_post = 0.016
    cx, cy = G.FRAME_HX - r_post - 0.004, G.FRAME_HY - r_post - 0.004
    return [(f"finial_{xn}", (sx * cx, cy, G.HEAD_TOP - 0.02), TUBE_FINIAL_SCALE) for sx, xn in ((-1, "L"), (1, "R"))]


def rounded_rect_path(hx, hy, r, z=0.0, n_arc=6):
    """Closed counter-clockwise path (list of (x, y, z)) round a rectangle of half sizes hx, hy with corner radius r."""
    pts = []
    corners = ((hx - r, hy - r, 0.0), (-hx + r, hy - r, 90.0), (-hx + r, -hy + r, 180.0), (hx - r, -hy + r, 270.0))
    for cx, cy, a0 in corners:
        for k in range(n_arc + 1):
            a = math.radians(a0 + 90.0 * k / n_arc)
            pts.append((cx + r * math.cos(a), cy + r * math.sin(a), z))
    return pts


# ===================================================================================================== meshes
def _wood_uv(V, long_axis, mix=(0, 1, 2)):
    """Per-vertex wood UVs: u = the coordinate along the board's long axis (metres), v = the sum of the two cross
    coordinates: continuous round every arris (no seams), grain runs along u."""
    a, b = [i for i in range(3) if i != long_axis]
    return np.stack([V[:, long_axis], V[:, a] + V[:, b]], axis=1)


def board_mesh(part):
    """A planed board: `core.shell.rounded_box` with the part's fillet radius, wood UVs along its longest side."""
    lo, hi = np.array(part["lo"]), np.array(part["hi"])
    m = CS.rounded_box(hi - lo, (lo + hi) / 2, r=part["round"], k=part["seg"])
    m.UV = _wood_uv(m.V, int(np.argmax(hi - lo)))
    return m


def post_mesh(part, n_end=4):
    """A post: a rounded-square section swept up the leg (z), `taper` = (k, h): k times as wide at the floor, full from
    z = lo.z + h; both ends rounded over by part["end"] in `n_end` steps (a circular arc)."""
    lo, hi = np.array(part["lo"]), np.array(part["hi"])
    c, w = (lo + hi) / 2, float(hi[0] - lo[0])
    k, h = part.get("taper", (1.0, 0.0))
    ro = part.get("end", 0.0)
    z0, z1 = float(lo[2]), float(hi[2])
    phi = np.linspace(0.0, math.pi / 2, n_end + 1)
    zs, ds = [], []
    for z, d in zip(z0 + ro * (1 - np.cos(phi)), ro * (1 - np.sin(phi))):               # bottom round-over
        zs.append(z)
        ds.append(d)
    if h > ro:
        zs.append(z0 + h)
        ds.append(0.0)
    for z, d in zip(z1 - ro * (1 - np.cos(phi[::-1])), ro * (1 - np.sin(phi[::-1]))):     # top round-over
        zs.append(z)
        ds.append(d)
    zs, ds = np.array(zs), np.array(ds)
    scale = (k + (1.0 - k) * np.clip((zs - z0) / max(h, 1e-9), 0.0, 1.0)) - 2.0 * ds / w
    path = np.stack([np.full(len(zs), c[0]), np.full(len(zs), c[1]), zs], axis=1)
    m = CS.sweep(path, CS.rrect(w, w, part["round"], n=part["seg"] + 1), scale=scale, up=(0.0, 1.0, 0.0))
    m.UV = _wood_uv(m.V, 2)
    return m


def frame_mesh(headboard="slats"):
    """The whole wooden frame as one `core.shell.Mesh` (material 0): posts, rails, boards, slats, cleats."""
    return CS.merge([post_mesh(p) if p["group"] == "post" else board_mesh(p) for p in frame_parts(headboard)])


def padded_mesh():
    """The padded headboard slab: a `rounded_box` with big soft edges, UV = (x, z) metres for the tufting."""
    part = padded_slab()
    lo, hi = np.array(part["lo"]), np.array(part["hi"])
    m = CS.rounded_box(hi - lo, (lo + hi) / 2, r=part["round"], k=part["seg"])
    m.UV = np.stack([m.V[:, 0], m.V[:, 2]], axis=1)
    return m


def tube_mesh(headboard="slats", sides=14):
    """The steel frame as one mesh: the tubes (open ends: they end inside other parts or on the floor) and the wooden
    version's rails and base boards."""
    meshes = []
    for _, pts, r in tube_parts(headboard):
        m = CS.tube(np.array(pts, float), r, sides=sides if r > 0.01 else 8, caps=(False, False))
        m.UV = np.zeros((len(m.V), 2))
        meshes.append(m)
    meshes += [board_mesh(p) for p in tube_boxes()]
    return CS.merge(meshes)
