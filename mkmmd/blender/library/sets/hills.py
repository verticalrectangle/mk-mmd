"""hills: layered ridges on the horizon of a night scene, the far ones fading into the sky's horizon haze.

    [[set]]
    name = "hills"
    kind = "hills"
    center = [0.0, -600.0]
    layers = [{ distance = 1300, height = 70, arc = [-60, 240] }, { distance = 2100, height = 150, seed = 2 }]

Each layer is a curtain round `center` (x, y; z 0 is the ground) at `distance` m: a ridge line of rolling hills `height` m
high at most (its own `seed`, `rough` 0..1: how many small bumps ride the big ones), from a little under the ground up to the
ridge, facing the centre. `arc` = [from, to] degrees (azimuth from +X toward +Y, as night_sky's moon) leaves the rest of the
circle open (a city beyond). The colour is flat (emissive, so the hills read as silhouettes whatever lights the scene): the
palette's `base` pulled toward the horizon haze (the colour the highway fades to) by `haze` (0..1; default rising with the
layer's distance: 0.25 for the nearest, 0.6 for the farthest). Unknown keys raise ValueError.

Card: hills.layers (distance, height, arc, object) for the record; nothing to stand on."""
import math

import numpy as np

from . import nightgeo as G
from . import nightkit as K
from . import register

LAYER = {"distance": 1500.0, "height": 90.0, "arc": [0.0, 360.0], "seed": 1, "rough": 0.5, "haze": None, "segments": 720}
TOP = {"center", "layers", "color", "name", "kind", "at", "yaw"}


def ridge(n, seed, rough):
    """Heights 0..1 of a closed ridge line sampled at n angles: a few long swells and, by `rough`, smaller bumps."""
    rng = np.random.default_rng(int(seed))
    a = np.linspace(0.0, 2 * math.pi, n, endpoint=False)
    h = np.zeros(n)
    for k, amp in ((2, 1.0), (3, 0.7), (5, 0.5), (9, 0.35 * rough), (17, 0.22 * rough), (31, 0.12 * rough)):
        h += amp * np.sin(k * a + rng.uniform(0, 2 * math.pi))
    h = (h - h.min()) / max(np.ptp(h), 1e-9)
    return 0.18 + 0.82 * h


def arc_indices(n, arc):
    """Indices of the n azimuth samples (0, 360/n, ...) inside `arc` = [from, to] degrees, in order from `from` (an arc may
    cross 0); a full circle comes back to its first sample so the curtain closes."""
    az = np.linspace(0.0, 360.0, n, endpoint=False)
    a0, a1 = (float(v) for v in arc)
    span = (a1 - a0) % 360.0 or 360.0
    sel = np.flatnonzero(((az - a0) % 360.0) <= span)
    if len(sel) and not np.all(np.diff(sel) == 1):          # across 0: start after the gap
        sel = np.roll(sel, -(int(np.argmax(np.diff(sel) > 1)) + 1))
    if span >= 360.0 and len(sel):
        sel = np.append(sel, sel[0])
    return sel


def curtain(center, distance, heights, arc):
    """(V, Q): the strip from below the ground to the ridge over the azimuths of `arc`, its faces toward the centre."""
    sel = arc_indices(len(heights), arc)
    if len(sel) < 2:
        return np.zeros((0, 3)), np.zeros((0, 4), np.int64)
    ang = np.radians(np.linspace(0.0, 360.0, len(heights), endpoint=False)[sel])
    x, y = center[0] + distance * np.cos(ang), center[1] + distance * np.sin(ang)
    bottom = np.stack([x, y, np.full(len(sel), -25.0)], 1)
    top = np.stack([x, y, heights[sel]], 1)
    m = len(sel)
    i = np.arange(m - 1)
    return np.concatenate([bottom, top]), np.stack([i, i + m, i + m + 1, i + 1], 1)    # wound to face the centre


@register("hills")
def build(name, coll, root, spec, palette):
    unknown = sorted(set(spec) - TOP)
    if unknown:
        raise ValueError(f"hills {name!r}: unknown keys {unknown} (known: {sorted(TOP - {'name', 'kind', 'at', 'yaw'})})")
    center = [float(v) for v in spec.get("center", (0.0, 0.0))]
    layers = list(spec.get("layers") or [{}])
    order = sorted(range(len(layers)), key=lambda i: float(layers[i].get("distance", LAYER["distance"])))
    base = K.rgb(palette, spec.get("color", "base"))
    haze = K.rgb(palette, K.haze_hex(palette))
    out = []
    for rank, i in enumerate(order):
        lay = layers[i]
        bad = sorted(set(lay) - set(LAYER))
        if bad:
            raise ValueError(f"hills {name!r} layer {i}: unknown keys {bad} (known: {sorted(LAYER)})")
        L = {**LAYER, **lay}
        h = ridge(int(L["segments"]), L["seed"], float(L["rough"])) * float(L["height"])
        V, Q = curtain(center, float(L["distance"]), h, L["arc"])
        if not len(Q):
            continue
        k = L["haze"] if L["haze"] is not None else 0.25 + 0.35 * rank / max(len(layers) - 1, 1)
        rgb = tuple(b + (z - b) * float(k) for b, z in zip(base, haze))
        m, nb = K.new_material(f"{name}_layer{i}")
        nb.output(nb.emission(rgb, 1.0))
        mesh = G.Mesh()
        mesh.add(G.Geom(V, Q), 0)
        o = K.to_object(mesh, f"{name}_layer{i}", coll, root, [m], shadow=False)
        out.append({"distance": float(L["distance"]), "height": float(L["height"]), "arc": list(L["arc"]), "object": o.name})
    return {"kind": "hills", "hills": {"layers": out}}
