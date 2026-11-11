"""Horns (part `horns`, numpy only): a pair of curved horns on the head, each a mirror of the other.

A horn is a tube swept along a curve that leaves the head and curls: it stands at `at` ([theta, phi] degrees on the
head: theta round it from the front toward her left, phi down from the crown; the other horn is its mirror), sunk
`sink` into the head so the hair closes round its root, heads `rise` degrees above level and `sweep` degrees toward
her back, and turns by `curl` degrees over its `length`, up and `inward` (0: straight up, 1: as much toward the middle
as up), faster toward its point (`curl_power`). Its section is an ellipse, `base` = [across the curl, thickness] metres
at the root, drawn to a point by `taper` (1 a cone; larger keeps it full longer); `rings` ridges `ring_depth` deep run
round its lower `ring_span`.

It is painted along its length (`[colors.horns]`): `base`, going to `tip` at its point, `shadow` on the inside of the
curl and in the grooves between the ridges, `light` in a stripe along the outside of the curl. It rides the head bone
(`bone`, 頭). Materials 角; landmarks horn.L / horn.R (the roots) and horn_tip.L / horn_tip.R."""
import numpy as np

from .. import spec as SP
from ..build import builder
from ..part import Part
from . import outfit_wear as WR
from .hair_fit import HeadFit, dirs, unit
from .hair_geo import MeshAccum

DEFAULTS = {"at": [62.0, 40.0], "sink": 0.015, "rise": 25.0, "sweep": 8.0, "curl": 105.0, "curl_power": 1.25,
            "inward": 0.45, "length": 0.20, "base": [0.058, 0.042], "taper": 1.7, "rings": 5, "ring_depth": 0.05,
            "ring_span": 0.5, "around": 20, "along": 44, "bone": "頭"}
COLORS = {"base": "#d42ad2", "shadow": "#77187a", "light": "#f96af4", "tip": "#e84ee6"}
MAT = "角"
TEX = (128, 512)                # the horn's texture: round it, along it


def spine(root, head, curl, toward, length, n, power):
    """(n + 1, 3) points of a horn's middle line from `root`, heading `head` (unit) and turning by `curl` radians toward
    `toward` over `length`, the turn made by arc share t going as t ** `power`; and its unit headings."""
    axis = unit(np.cross(head, toward))
    t = np.linspace(0.0, 1.0, n + 1)
    a = curl * t ** power
    K = np.array([[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]])
    D = np.array([head + np.sin(x) * (K @ head) + (1.0 - np.cos(x)) * (K @ (K @ head)) for x in a])
    mid = 0.5 * (D[:-1] + D[1:])
    P = np.vstack([root, root + np.cumsum(mid * (length / n), axis=0)])
    return P, unit(D), axis


def horn(root, head, toward, c):
    """One horn's mesh: (verts, faces (counter-clockwise from outside), uv (per vertex: u round it from the outside of
    the curl, v from the root to the point), unit normals, the tip). Each ring has its first point twice, at u 0 and
    u 1: the texture's seam, down the outside of the curl."""
    n, m = int(c["along"]), int(c["around"])
    P, D, axis = spine(root, head, np.radians(c["curl"]), toward, c["length"], n, c["curl_power"])
    t = np.linspace(0.0, 1.0, n + 1)
    scale = 1.0 - t ** c["taper"]
    ridge = 1.0 + c["ring_depth"] * 0.5 * (1.0 - np.cos(2.0 * np.pi * c["rings"] * t / max(c["ring_span"], 1e-6))) * \
        (t < c["ring_span"])
    out = unit(np.cross(D, axis))                       # the outside of the curl
    phi = 2.0 * np.pi * np.arange(m + 1) / m
    a, b = 0.5 * c["base"][0], 0.5 * c["base"][1]
    cs, sn = np.cos(phi)[None, :, None], np.sin(phi)[None, :, None]
    rings = P[:-1, None, :] + (scale[:-1] * ridge[:-1])[:, None, None] * (a * cs * out[:-1, None, :] +
                                                                         b * sn * axis[None, None, :])
    N = unit(cs / a * out[:-1, None, :] + sn / b * axis[None, None, :])
    V = np.vstack([rings.reshape(-1, 3), P[-1][None]])
    N = np.vstack([N.reshape(-1, 3), D[-1][None]])
    tip, k = len(V) - 1, m + 1
    faces = [[i * k + j, (i + 1) * k + j, (i + 1) * k + j + 1, i * k + j + 1] for i in range(n - 1) for j in range(m)]
    faces += [[(n - 1) * k + j, tip, (n - 1) * k + j + 1] for j in range(m)]
    q = V[faces[0]]
    if np.cross(q[1] - q[0], q[2] - q[1]) @ N[faces[0][0]] < 0.0:
        faces = [f[::-1] for f in faces]
    uv = np.vstack([np.stack([np.tile(np.arange(k) / m, n), np.repeat(t[:-1], k)], -1), [[0.5, 1.0]]])
    return V, faces, uv, N, P[-1]


def texture(col, c):
    """The horn's picture (TEX[1], TEX[0], 4) uint8: v down it from the root (row 0) to the point, u round it from the
    outside of the curl."""
    w, h = TEX
    u = (np.arange(w) + 0.5) / w
    t = (np.arange(h) + 0.5) / h
    face = np.cos(2.0 * np.pi * u)[None, :]             # 1 on the outside of the curl, -1 on the inside
    rgb = col["base"][None, None, :] + (col["tip"] - col["base"])[None, None, :] * (t ** 1.5)[:, None, None]
    inside = np.clip(0.15 - face, 0.0, 1.0) / 1.15
    groove = 0.5 * (1.0 + np.cos(2.0 * np.pi * c["rings"] * t / max(c["ring_span"], 1e-6))) * (t < c["ring_span"])
    dark = np.clip(0.85 * inside + 0.35 * groove[:, None] + 0.25 * np.clip(1.0 - t / 0.12, 0.0, 1.0)[:, None], 0.0, 1.0)
    rgb = rgb + (col["shadow"][None, None, :] - rgb) * dark[..., None]
    stripe = np.clip((face - 0.80) / 0.20, 0.0, 1.0) ** 2 * np.clip(1.2 - t, 0.0, 1.0)[:, None]
    rgb = rgb + (col["light"][None, None, :] - rgb) * (0.75 * stripe)[..., None]
    img = np.empty((h, w, 4), np.uint8)
    img[..., :3] = np.rint(np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8)
    img[..., 3] = 255
    return img


@builder("horns", needs=("body", "head"))
def build(ctx):
    c = SP.merge(DEFAULTS, ctx.cfg)
    head = ctx.need("head")
    fit = HeadFit(head.info, meshes=head.meshes)
    col = WR.colours(ctx, "horns", COLORS)
    acc = MeshAccum("horns", [MAT])
    land = {}
    for sx, tag in ((1.0, "L"), (-1.0, "R")):
        d = dirs(np.radians(sx * c["at"][0]), np.radians(c["at"][1]))
        root = fit.skull.point(d, -c["sink"])
        rise, sweep = np.radians(c["rise"]), np.radians(c["sweep"])
        heading = np.array([sx * np.cos(rise) * np.cos(sweep), np.cos(rise) * np.sin(sweep), np.sin(rise)])
        toward = unit(np.array([-sx * c["inward"], 0.0, 1.0]))
        V, faces, uv, N, tip = horn(root, heading, toward, c)
        uvc = np.stack([uv[:, 0], 1.0 - uv[:, 1]], -1)      # v up: the root at the top of the picture
        acc.add(V, faces, uvc, 0, weights={c["bone"]: np.ones(len(V))}, normals=N)
        land[f"horn.{tag}"], land[f"horn_tip.{tag}"] = root, tip
    mat = WR.material(ctx, MAT, "horn", col["base"], texture(col, c), shadow=(0.62, 0.50, 0.72), edge_size=0.8,
                      specular=(0.25, 0.2, 0.25), shininess=30.0, double_sided=False)
    ctx.log(f"two horns, {c['length'] * 1000:.0f} mm along, tips at z {land['horn_tip.L'][2]:.3f}")
    return Part("horns", meshes=[acc.to_mesh()], materials=[mat], info={"landmarks": land, "length": c["length"]})
