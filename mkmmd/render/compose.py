"""The finish of mk's own renderers (docs/design.md: Rendering): a frame's passes composed with
mkmmd.core.shotstyle's compose_vector / compose_silhouette and the rings' masks exactly as mkmmd.blender.styles composes
Blender's passes, and written as the frame's 8-bit PNG (whole or not at all). Runs in worker processes.

  finish       the vector look's passes as the Chromium page sent them (passes.js: id and shade, depth, the texture pass;
               bottom row first), for engine = "mk"
  finish_look  a look's passes as mkr wrote them (mkr/mkr/src/looks.rs: one file per frame or layer, top row first), for
               engine = "mkr": a silhouette's or a vector look's frame, a plate or a bare frame in its look, or a matte
               (the frame without its figure, back/, then the figure's coverage, matte/)"""
import json
import os

import numpy as np

from ..core import rings as RG
from ..core import shotstyle as SS


def passes(buf, size, tex):
    """The passes compose_vector reads, top row first, from the page's bytes for passes of `size` [W, H]."""
    W, H = size
    n = W * H * 4
    a = np.frombuffer(buf, np.uint8)
    if a.size != n * (3 if tex else 2):
        raise ValueError(f"{a.size} bytes of passes, expected {n * (3 if tex else 2)} for {W} x {H}")
    idsh = a[:n].reshape(H, W, 4)[::-1]
    p = {"id": idsh[..., 0].astype(np.int32) | (idsh[..., 1].astype(np.int32) << 8),
         "shade": idsh[..., 2].astype(np.float32) / 255.0,
         "depth": a[n:2 * n].view("<f4").reshape(H, W)[::-1]}
    if tex:
        p["tex"] = a[2 * n:].reshape(H, W, 4)[::-1, :, :3].astype(np.float32) / 255.0
    return p


# ================================================================================================= mkr's passes
LOOK_MAGIC = b"MKRLOOK1"
DTYPES = {"u8": np.uint8, "u16": "<u2", "f32": "<f4"}


def read_look(path):
    """A look's pass file as mkr writes it: MAGIC, u32 header length, the JSON header {job, passes: {name: {offset,
    shape, type}}}, then the arrays (offsets from the file's start; u8 / u16 / f32, little-endian, top row first).
    Returns (header, {name: array})."""
    with open(path, "rb") as fh:
        buf = fh.read()
    if buf[:8] != LOOK_MAGIC:
        raise ValueError(f"{path} is not a look's pass file")
    n = int.from_bytes(buf[8:12], "little")
    head = json.loads(buf[12:12 + n].decode("utf-8"))
    out = {}
    for name, rec in head["passes"].items():
        shape = [int(v) for v in rec["shape"]]
        out[name] = np.frombuffer(buf, DTYPES[rec["type"]], count=int(np.prod(shape)), offset=int(rec["offset"])).reshape(shape)
    return head, out


def look_passes(raw, kind):
    """The passes compose_silhouette / compose_vector read, as Blender's own passes give them (8-bit files read back
    as floats in 0..1, ids as ints, depth in metres), from mkr's arrays."""
    unit = lambda a: a.astype(np.float32) / 255.0        # noqa: E731 - an 8-bit pass as Blender reads its PNG
    p = {}
    if kind == "silhouette":
        p["flat"] = unit(raw["flat"])
        if "soft" in raw:
            p["soft_alpha"], p["soft_aov"] = unit(raw["soft"][..., 0]), unit(raw["soft"][..., 1])
        if "type" in raw:
            p["type"] = unit(raw["type"])
        knocks = [unit(raw[k]) for k in ("knock", "screen_knock") if k in raw]
        if knocks:
            p["knock"] = knocks[0] if len(knocks) == 1 else np.maximum(knocks[0], knocks[1])
        return p
    p["id"] = raw["id"].astype(np.int32)
    p["depth"] = raw["depth"]
    p["shade"] = unit(raw["shade"])
    if "tex" in raw:
        p["tex"] = unit(raw["tex"])
    if "type" in raw:
        p["type"] = unit(raw["type"])
    return p


def _write_png(path, img):
    """Float RGB (h, w, 3) in 0..1 as Blender's looks save it (styles._save): 8-bit, rounded half up; whole or not at
    all, its folder made."""
    img8 = (np.clip(img, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = f"{path}.{os.getpid()}.part"
    with open(tmp, "wb") as fh:
        fh.write(SS.png_bytes(img8))
    os.replace(tmp, path)


def compose_look(p, job, subject=True):
    """The finished look picture (float RGB, display space) of a job's passes, as styles.Looks.compose makes it: the
    silhouette from its passes, the background layers at the frame's tint values; the vector look with the rings live
    on its frame (`ring_t`: the ring table's clock), boiling on its clip second `t`, the type laid over it."""
    if job["look"] == "silhouette":
        return SS.compose_silhouette(p, job["spec"], job["tints"], hard=job["hard"], subject=subject)
    flip, edges = None, ()
    size = (p["id"].shape[1] / SS.VECTOR_SCALE, p["id"].shape[0] / SS.VECTOR_SCALE)
    if job["rings"] and RG.live(job["rings"], job["ring_t"]):
        live = {int(k): v for k, v in job["centres"].items()}
        flip, edges = RG.masks(RG.shapes(job["rings"], job["ring_t"], live, size), size, SS.VECTOR_SCALE)
    img = SS.compose_vector(p, job["spec"], (job["table"][0], job["table"][1]), subject=subject, flip=flip,
                            edges_on=edges, t=job["t"])
    if "type" in p:
        a = np.clip(p["type"][..., 3:4], 0.0, 1.0)
        img = img * (1.0 - a) + p["type"][..., :3] * a
    return img


def finish_look(path, job):
    """Compose a job's pass file (`path`, deleted afterwards) and write its files, the finishing one last: a frame, a
    plate or a bare frame is one picture; a matte is the frame without its figure (back/) and then the figure's
    coverage (matte/, grey: Workbench's alpha over a whole pixel's 254, mkmmd.core.shotstyle.ALPHA_FULL)."""
    head, raw = read_look(path)
    if int(head["job"]) != int(job["id"]):
        raise ValueError(f"{path} holds job {head['job']}, not {job['id']}")
    p = look_passes(raw, job["look"])
    if job["role"] == "matte":
        _write_png(job["files"][0], compose_look(p, job, subject=False))
        a = np.clip(raw["matte"].astype(np.float32) / 255.0 / SS.ALPHA_FULL, 0.0, 1.0)
        _write_png(job["files"][1], np.repeat(a[..., None], 3, axis=2))
    else:
        _write_png(job["files"][0], compose_look(p, job))
    os.unlink(path)


def finish(path, buf, size, tex, spec, table, rings, t, centres):
    """Compose one frame and write it to `path`: `rings` the ring table's rings, `t` the clip second, `centres` the live
    rings' centres as frame fractions."""
    p = passes(buf, size, tex)
    flip, edges = None, ()
    live = RG.live(rings, t)
    if live:
        fs = (size[0] / SS.VECTOR_SCALE, size[1] / SS.VECTOR_SCALE)
        flip, edges = RG.masks(RG.shapes(rings, t, centres, fs), fs, SS.VECTOR_SCALE)
    img = SS.compose_vector(p, spec, (table[0], table[1]), flip=flip, edges_on=edges, t=t)
    img8 = (np.clip(img, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    tmp = f"{path}.{os.getpid()}.part"
    with open(tmp, "wb") as fh:
        fh.write(SS.png_bytes(img8))
    os.replace(tmp, path)
