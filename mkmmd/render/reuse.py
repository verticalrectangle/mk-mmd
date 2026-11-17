"""Which of mkr's frames a rebuilt scene left as they were, so `mk render` keeps them instead of drawing them again.

mkr draws the same picture from the same input, bit for bit (but for a speck of the driver's colour compression in a
frame now and then: docs/design.md, mkr, `RADV_DEBUG=nodcc`), and a frame's input is the bake, the run's settings and
the frame's own layers. The bake says which of its values it holds a row a frame (bake.json `per_frame`: [float offset,
floats a row] into bake.bin, marked where the bake stage writes them: bones, matrices, visibility, shape keys, lights,
cameras, keyed node values), and whose rows they are where only some frames read them ([float offset, floats a row,
scope]: "camera:NAME" the frames seen through it, "armature:NAME" the frames that show a mesh it bends, "mesh:NAME" the
frames that show it); everything else in the bake counts as part of every frame, but for the looks of [[look.span]]s
(bake.json lit.view `spans`: a run of frames seen through another look's LUT), which are their frames' own. So

    signature(bake)   {"static": a digest of everything but those rows and spans, "frames": [a digest of frame k's
                       rows of no scope], "scoped": {scope: [a digest of its rows of frame k]}, "shows": {scope:
                       [whether frame k shows it]} for the armatures and meshes some frames do not, "view": [a digest
                       of the look frame k is seen through, None for the view's own]}
    keys(...)         {frame: key}: the static digest, the run's settings, the frame's layers, the frame's look, and
                      the rows of the frames a shutter can reach (RADIUS either side of it): all those of no scope, and
                      those of the cameras it is seen through and of the armatures and meshes it shows

and a frame on disk whose stored key (keys.json beside the frames: `load`, `save`) is its key now shows what a render
now would draw, however old the file is. The scopes are what makes a change stay where it is: a camera's clip range
follows what its shot shows and is written on every frame, and a character keeps moving (her hair, by the sim) while
a later scene hides her; a span's new exposure draws the span's frames again and no others. What the marks do not cover
costs frames, never a wrong one: a material, a mesh, a by-frame geometry, a word of screen type or the view's own look
that changes is part of every frame, and all of them are drawn again. Pure numpy, outside Blender."""
import hashlib
import json
import os
import re
from pathlib import Path

import numpy as np

FILE = "keys.json"
VERSION = 1
RADIUS = 2                                   # frames either side whose rows a picture can read: mkr's shutter is at
#                                              most 2 frames long, wherever the scene's motion blur puts it
PARTS = ("lit.bin", "view_lut.bin", "looks_lut.bin", "screen_lut.bin")   # the bake's other files mkr reads: whole
VOLATILE = re.compile(r"mmd_bind-?\d+")      # mmd_tools names these nodes after a hash: new on every build


class ReuseError(ValueError):
    pass


def _digest():
    return hashlib.blake2b(digest_size=16)


def _file(h, path):
    """Feed a file's bytes to `h` ("-" for one that is not there)."""
    try:
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 22), b""):
                h.update(chunk)
    except OSError:
        h.update(b"-")


def canonical(doc):
    """bake.json as a signature reads it: without what a build writes anew and no picture reads (the self-check's
    measurements, `lit.check`) or what only some frames read (the [[look.span]]s' looks, lit.view `spans`: `_views`),
    and with mmd_tools' bind nodes numbered in the order they come (their names differ from build to build)."""
    doc = dict(doc)
    if isinstance(doc.get("lit"), dict):
        lit = {k: v for k, v in doc["lit"].items() if k != "check"}
        if isinstance(lit.get("view"), dict):
            lit["view"] = {k: v for k, v in lit["view"].items() if k != "spans"}
        doc["lit"] = lit
    seen = {}
    return VOLATILE.sub(lambda m: seen.setdefault(m.group(0), f"mmd_bind#{len(seen)}"),
                        json.dumps(doc, ensure_ascii=False, separators=(",", ":")))


def _views(bake, doc, n):
    """[hex or None per frame]: the digest of the look frame k is seen through where a [[look.span]] holds it (bake.json
    lit.view `spans`: its record but for its frames and its LUT's file name, and the LUT's bytes), None where the
    view's own holds it (in the static digest)."""
    out = [None] * n
    frame0, luts = int(doc.get("frame0") or 0), {}
    for s in (((doc.get("lit") or {}).get("view") or {}).get("spans") or ()):
        rec = {k: v for k, v in s.items() if k != "frames"}
        rec["lut"] = {k: v for k, v in rec["lut"].items() if k != "file"}
        if s["lut"]["file"] not in luts:
            h = _digest()
            _file(h, bake / s["lut"]["file"])
            luts[s["lut"]["file"]] = h.hexdigest()
        h = _digest()
        h.update(json.dumps([rec, luts[s["lut"]["file"]]], sort_keys=True).encode("utf-8"))
        look = h.hexdigest()
        for k in range(max(int(s["frames"][0]) - frame0, 0), min(int(s["frames"][1]) - frame0 + 1, n)):
            out[k] = look
    return out


def _shows(doc, data, n):
    """{scope: (n,) bool} the frames mkr shows a mesh on ("mesh:NAME": its `shown` flags, a row a frame in bake.bin; a
    mesh without them is on every frame) or any mesh an armature bends ("armature:NAME"), from the lit section's
    meshes. A scope that is not in it is read by every frame."""
    out = {}
    floats = data[:len(data) // 4 * 4].view("<f4")
    for m in ((doc.get("lit") or {}).get("meshes") or []):
        at = m.get("shown")
        flags = np.ones(n, bool) if at is None else np.asarray(floats[at:at + n]) != 0.0
        for scope in ["mesh:" + m["name"]] + (["armature:" + m["armature"]] if m.get("armature") else []):
            out[scope] = out[scope] | flags if scope in out else flags
    return out


def signature(bake, doc=None):
    """{"static": hex, "parts": {name: hex} (what the static digest is made of: bake.json, bake.bin's values that are
    not a row a frame, each other file of the bake mkr reads, the textures), "frames": [hex per frame], "scoped":
    {scope: [hex per frame]}, "shows": {scope: [bool per frame]}, "view": [hex or None per frame] (`_views`)} of the
    bake folder `bake` (`doc`: its bake.json, read already); None when the bake does not say what it holds a row a
    frame (an older mk's: `mk build --bake` writes it). ReuseError for a `per_frame` table that does not fit bake.bin."""
    bake = Path(bake)
    if doc is None:
        doc = json.loads((bake / "bake.json").read_text(encoding="utf-8"))
    rows, n = doc.get("per_frame"), int(doc.get("frames") or 0)
    if rows is None or n < 1:
        return None
    path = bake / "bake.bin"
    data = np.memmap(path, np.uint8, "r") if path.exists() and path.stat().st_size else np.zeros(0, np.uint8)
    # what is held a row a frame: (byte offset, bytes a row, whose rows they are or None)
    blocks = sorted((4 * int(r[0]), 4 * int(r[1]), r[2] if len(r) > 2 else None) for r in rows)
    parts = {name: _digest() for name in ("bake.json", "bake.bin", *PARTS, "images")}
    parts["bake.json"].update(canonical(doc).encode("utf-8"))
    frames = [_digest() for _ in range(n)]
    scoped = {}
    at = 0
    for b0, per, scope in blocks:
        if b0 < at or b0 + per * n > len(data):
            raise ReuseError(f"{bake}: bake.json's per_frame rows overlap or run past bake.bin")
        parts["bake.bin"].update(data[at:b0])
        block = memoryview(np.asarray(data[b0:b0 + per * n]))
        into = frames if scope is None else scoped.setdefault(scope, [_digest() for _ in range(n)])
        for k in range(n):
            into[k].update(block[k * per:(k + 1) * per])
        at = b0 + per * n
    parts["bake.bin"].update(data[at:])
    for name in PARTS:
        _file(parts[name], bake / name)
    for name, rec in sorted(((doc.get("lit") or {}).get("images") or {}).items()):   # the textures, where they are
        parts["images"].update(name.encode("utf-8"))
        if rec.get("file"):
            _file(parts["images"], bake / rec["file"])     # an absolute path stays one
    parts = {name: h.hexdigest() for name, h in parts.items()}
    static = _digest()
    static.update(json.dumps(parts, sort_keys=True).encode())
    return {"static": static.hexdigest(), "parts": parts, "frames": [h.hexdigest() for h in frames],
            "scoped": {s: [h.hexdigest() for h in hs] for s, hs in scoped.items()},
            "shows": {s: flags.tolist() for s, flags in _shows(doc, data, n).items()
                      if s in scoped and not flags.all()},
            "view": _views(bake, doc, n)}


def keys(sig, frame0, frames, settings, layers, cameras=None):
    """{frame: key} of `frames` (Blender frames; one outside the clip gets none) under the signature `sig`:
    `settings` is what the run draws with and for (JSON: the renderer, its samples and tunables, the size, the
    shutter, the output), `layers(frame)` the frame's own layers (JSON), `cameras(frame)` the names of the cameras the
    frame and its layers are seen through (None, or no `cameras`: every camera's rows count)."""
    n = len(sig["frames"])
    views = sig.get("view")                                # per frame: the look of the [[look.span]] that holds it
    base = _digest()
    base.update(json.dumps([VERSION, sig["static"], settings], sort_keys=True, ensure_ascii=False).encode("utf-8"))
    reach = {}                                             # scope: the frames whose shutter can reach one that shows it
    for scope, flags in sig["shows"].items():
        flags = np.asarray(flags, bool)
        reach[scope] = [bool(flags[max(k - RADIUS, 0):k + RADIUS + 1].any()) for k in range(n)]
    out = {}
    for f in frames:
        k = int(f) - int(frame0)
        if not 0 <= k < n:
            continue
        h = base.copy()
        h.update(json.dumps(layers(f), sort_keys=True, ensure_ascii=False).encode("utf-8"))
        look = views[k] if views else None
        if look is not None:                               # a [[look.span]]'s: the view's own is in the static digest
            h.update(b"view:" + look.encode())
        seen = cameras(f) if cameras is not None else None
        seen = None if seen is None else {"camera:" + c for c in seen}
        window = range(k - RADIUS, k + RADIUS + 1)
        for j in window:
            h.update(sig["frames"][j].encode() if 0 <= j < n else b"-")
        for scope in sorted(set(sig["scoped"]) | (seen or set())):   # a camera the bake has no rows of: its name alone
            if scope.startswith("camera:") and seen is not None and scope not in seen:
                continue
            if scope in reach and not reach[scope][k]:
                continue
            h.update(scope.encode("utf-8"))
            for j in window:
                h.update(sig["scoped"][scope][j].encode() if scope in sig["scoped"] and 0 <= j < n else b"-")
        out[int(f)] = h.hexdigest()
    return out


def load(d, about=False):
    """{frame: key} stored beside the frames of the folder `d` ({} without a readable file); with `about`, what was
    stored with them instead (`save`'s: {} without)."""
    try:
        doc = json.loads((Path(d) / FILE).read_text(encoding="utf-8"))
        if doc.get("version") != VERSION:
            return {}
        return dict(doc.get("about") or {}) if about else {int(f): str(k) for f, k in doc["keys"].items()}
    except (OSError, ValueError, KeyError, AttributeError, TypeError):
        return {}


def save(d, stored, about=None):
    """Store {frame: key} beside the frames of `d`, whole or not at all, and `about` with them (JSON: what the keys were
    made under, a signature's parts and the run's settings, for `changed`; None keeps what is stored)."""
    path = Path(d) / FILE
    part = path.with_suffix(".json.part")
    about = load(d, about=True) if about is None else about
    part.write_text(json.dumps({"version": VERSION, "about": about,
                                "keys": {f"{f:05d}": k for f, k in sorted(stored.items())}}), encoding="utf-8")
    os.replace(part, path)


def changed(was, now):
    """The names of what differs between two `about` records ({"parts": {name: hex}, "settings": {...}}): the parts
    of the bake every frame reads, and "settings" (the renderer, its samples, the size...). What a render says when
    it has to draw every frame again."""
    parts = sorted(k for k in set(was.get("parts") or {}) | set(now.get("parts") or {})
                   if (was.get("parts") or {}).get(k) != (now.get("parts") or {}).get(k))
    return parts + (["settings"] if was.get("settings") != now.get("settings") else [])
