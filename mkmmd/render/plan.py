"""Which frames mk's own renderer draws, and with what (docs/design.md: Rendering), bpy-free, from the scene's bake
(mkmmd/blender/build/bake.py). A shot of an output is mk's when its look is vector and everything it shows over its
frames is in the bake; anything else (another look, type over the picture, an object the bake cannot hold, a drawn
tone whose texture it lacks, a ring whose centre failed) leaves the whole shot to Blender, with the reason.

The vector look sorts the objects and numbers the materials as Blender's does (mkmmd.core.shotstyle.vector_sort,
vector_materials, vector_table), and a ring's centre is projected through the bake's camera as world_to_camera_view
does (the page's mkmmd/site/web/js/scene/shotcam.js draws through the same camera)."""
import json
from pathlib import Path

import numpy as np

from ..core import rings as RG
from ..core import shotstyle as SS

BAKE_VERSION = 2            # the bake that holds what the vector look needs


class BakeError(ValueError):
    """No bake the renderer can draw from."""


def load(bake_dir):
    """(bake.json as a dict, bake.bin as float32, memory-mapped)."""
    d = Path(bake_dir)
    try:
        doc = json.loads((d / "bake.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise BakeError(f"no bake in {d} ({e}): run `mk build --bake`") from None
    if doc.get("version", 1) < BAKE_VERSION:
        raise BakeError(f"the bake in {d} is older than mk's renderer: run `mk build --bake`")
    return doc, np.memmap(d / "bake.bin", dtype="<f4", mode="r")


def render_size(size, percent):
    """The pixels Blender renders for an output `size` at `percent` (it rounds down)."""
    return [int(size[0]) * int(percent) // 100, int(size[1]) * int(percent) // 100]


def camera_matrices(blob, cam, k, size):
    """(view, projection) of a bake camera on bake frame `k` for a picture `size` (row-major 4x4, float64): the view takes
    glTF's world to Blender's camera axes, the projection is Blender's lens on the fitted sensor with its shift."""
    o = cam["offset"] + 15 * k
    c = np.asarray(blob[o:o + 15], np.float64)
    C = np.eye(4)
    C[:3, :] = c[:12].reshape(4, 3).T
    U = np.array([[1.0, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
    W, H = size
    fit = cam.get("fit", "AUTO")
    S = H if fit == "VERTICAL" else W if fit == "HORIZONTAL" else max(W, H)
    f = c[12] / (cam.get("sensor_height") or cam["sensor"] if fit == "VERTICAL" else cam["sensor"])
    near, far = cam.get("clip", (0.1, 100.0))
    P = np.array([[2 * S * f / W, 0, 2 * S * c[13] / W, 0], [0, 2 * S * f / H, 2 * S * c[14] / H, 0],
                  [0, 0, -(far + near) / (far - near), -2 * far * near / (far - near)], [0, 0, -1, 0]])
    return U @ np.linalg.inv(C), P


def project(blob, cam, k, size, point):
    """A world point (glTF's axes) through the camera on bake frame `k`, as frame fractions from the top left (Blender's
    world_to_camera_view, its y turned), or None behind the camera."""
    V, P = camera_matrices(blob, cam, k, size)
    q = V @ np.array([*point, 1.0])
    if -q[2] <= 0:
        return None
    clip = P @ q
    return (float(clip[0] / clip[3] + 1) / 2, float(1 - clip[1] / clip[3]) / 2)


def _hits(ranges, k0, k1):
    return any(a < k1 and b > k0 for a, b in ranges)


def shot_look(doc, entry, output, span):
    """The vector look of a shot for an output when mk can draw it over the bake frames [k0, k1) of `span`:
    (look, None), else (None, why). A look: {shot, output, spec, table, draw (the object names drawn), ids ({material:
    id}), light, tex (whether a drawn tone needs the texture pass)}."""
    look = (entry.get("styles") or {}).get(output)
    if not look:
        return None, "lit by the scene (no flat look)"
    kind = next(iter(look))
    if kind != "vector":
        return None, f"a {kind} look"
    spec = look[kind]
    cam = (entry.get("cameras") or {}).get(output)
    if not any(c["name"] == cam for c in doc["cameras"]):
        return None, f"its camera {cam!r} is not in the bake"
    k0, k1 = span
    objs = {o["name"]: o for o in doc["objects"]}
    sort = SS.vector_sort([SS.Obj(o["name"], tuple(o["collections"]), frozenset(o["props"])) for o in doc["objects"]],
                          spec, {o["name"] for o in doc["objects"] if o["text"]})
    for n in sort["subject"]:
        o = objs[n]
        if o["baked"] is None and _hits(o["shows"], k0, k1):
            return None, f"{n}: {o.get('why', 'not in the bake')}"
    for n in sort["type"]:
        if _hits(objs[n]["shows"], k0, k1):
            return None, f"{n}: type over the picture (Blender draws it)"
    for b in doc.get("blocks", []):
        if _hits(b["shows"], k0, k1):
            return None, f"{b['name']}: {b['why']}"
    mats = SS.vector_materials(objs[n]["materials"] for n in sort["subject"])
    table = SS.vector_table(mats, spec)
    by_frame = {m for n in sort["subject"] if objs[n]["baked"] == "by_frame" for m in objs[n]["materials"]}
    drawn = False
    for i, m in enumerate(mats):
        if spec["tones"][table[0][i + 1]]["kind"] != "drawn":
            continue
        drawn = True
        rec = doc["materials"].get(m) or {}
        if rec.get("texture") == "other":
            return None, f"material {m}: drawn from a texture the bake does not hold"
        if rec.get("texture") == "glb" and m in by_frame:
            return None, f"material {m}: drawn from a texture on a mesh baked without UVs"
    rings = (doc.get("rings") or {}).get("table") or {}
    errors = (doc.get("rings") or {}).get("errors") or {}
    if errors and rings:
        t0, t1 = k0 / doc["fps"], k1 / doc["fps"]          # clip seconds (the ring table's frame0 is the clip's)
        for r in rings["rings"]:
            if str(r["index"]) in errors and r["at"] < t1 and r["until"] > t0:
                return None, errors[str(r["index"])]
    return {"shot": entry["name"], "output": output, "spec": spec, "table": [list(table[0]), list(table[1])],
            "draw": [n for n in sort["subject"] if objs[n]["baked"]], "ids": {m: i + 1 for i, m in enumerate(mats)},
            "light": list(spec["light"]), "tex": drawn}, None


def plan(doc, blob, outputs, frames, percent):
    """What mk draws of `frames` (Blender frames) for `outputs` [(name, size)] at `percent`:
      looks    [look] (shot_look's)
      items    [{output, frame, k, camera, look (index), size (the passes' [W, H]), t (clip seconds), centres}] in order,
               `centres` the live rings' centres as frame fractions ({index: (x, y) or None})
      rings    the ring table's rings (the finish draws the live ones from `centres`)
      blender  {output: [frames Blender renders]}
      shots    {output: {shot: {"engine": "mk" | "blender", "frames": n, "why": ...}}}"""
    frame0, n = doc["frame0"], doc["frames"]
    cams = {c["name"]: c for c in doc["cameras"]}
    shots = [e for e in doc.get("shots") or [] if not e.get("plate")]
    rings = (doc.get("rings") or {}).get("table") or {}
    points = (doc.get("rings") or {}).get("points") or {}
    at = {f: SS.shot_at(shots, f) for f in range(frame0, frame0 + n)}
    spans = {}
    for f, e in at.items():
        if e is not None:
            k = f - frame0
            a, b = spans.get(e["name"], (k, k + 1))
            spans[e["name"]] = (min(a, k), max(b, k + 1))
    looks, items, blender, report, decided = [], [], {}, {}, {}
    for name, size in outputs:
        w, h = render_size(size, percent)
        blender[name], report[name] = [], {}
        for f in frames:
            k = f - frame0
            e = at.get(f)
            if e is None:
                blender[name].append(f)
                continue
            key = (e["name"], name)
            if key not in decided:
                look, why = shot_look(doc, e, name, spans[e["name"]])
                if look is not None:
                    looks.append(look)
                decided[key] = (len(looks) - 1 if look is not None else None, why)
            li, why = decided[key]
            rec = report[name].setdefault(e["name"], {"engine": "blender" if li is None else "mk", "frames": 0,
                                                       **({"why": why} if why else {})})
            rec["frames"] += 1
            if li is None:
                blender[name].append(f)
                continue
            cam = cams[e["cameras"][name]]
            t = (f - rings["frame0"]) / rings["fps"] if rings else k / doc["fps"]
            centres = {}
            for r in RG.live(rings.get("rings") or [], t) if rings else []:
                if r["center"]["mode"] == "frame":
                    centres[r["index"]] = tuple(r["center"]["at"])
                else:
                    o = points[str(r["index"])] + 3 * k
                    p = np.asarray(blob[o:o + 3], np.float64)
                    centres[r["index"]] = None if not np.isfinite(p).all() else project(blob, cam, k, (w, h), p)
            items.append({"output": name, "frame": f, "k": k, "camera": cam["name"], "look": li,
                          "size": [w * SS.VECTOR_SCALE, h * SS.VECTOR_SCALE], "t": t, "centres": centres})
    return {"looks": looks, "items": items, "rings": rings.get("rings") or [], "blender": blender, "shots": report}
