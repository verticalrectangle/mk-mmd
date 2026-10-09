"""A posed model as one binary glTF 2.0 file (.glb) with its textures inside: what Tern's 3D block, and any glTF viewer,
turns, pans and zooms. No Blender: the lab's Model (a PMX file or a spec built in-process) posed with its own weights.

  write(path, model, V, N, name=None, extras=None, keep=None) -> {"vertices", "triangles", "materials", "images", "bytes"}
        keep: the triangles to write (a region, say); only the vertices they use go in, so a viewer frames them

Model space (metres, Z up, -Y forward, +X her left) becomes glTF's (+Y up, +Z forward, +X her left): (x, y, z) ->
(x, z, -y), a rotation. PMX keeps its faces clockwise seen from the front and model space keeps them so; glTF draws
counter-clockwise faces, so each triangle is written reversed. UVs need nothing: PMX and glTF both run v down from the
top-left corner. Each drawn material is one primitive over a shared vertex buffer: its texture (embedded as PNG; other
formats converted), its diffuse alpha, BLEND when the texture or the alpha is see-through, and two-sided when the PMX
says so. Materials the lab hides (diffuse alpha ~ 0) are left out.
"""
import io
import json
import struct
from pathlib import Path

import numpy as np
from PIL import Image

AXES = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]])     # model -> glTF, det +1
SEE_THROUGH = 250                       # a texture with any texel's alpha below this (of 255) is blended
_FLOAT, _UINT = 5126, 5125              # glTF component types
_ARRAY, _ELEMENTS = 34962, 34963        # bufferView targets


def _image(path):
    """(PNG bytes, see-through) of the texture at `path`; formats other than PNG are converted."""
    p = Path(path)
    with Image.open(p) as im:
        im.load()
        if im.mode == "P" and "transparency" in im.info:
            im = im.convert("RGBA")
        alpha = im.getchannel("A") if im.mode in ("RGBA", "LA") else None
        see = alpha is not None and alpha.getextrema()[0] < SEE_THROUGH
        if im.format == "PNG" and p.suffix.lower() == ".png":
            return p.read_bytes(), see
        buf = io.BytesIO()
        im.convert("RGBA" if alpha is not None else "RGB").save(buf, "PNG")
        return buf.getvalue(), see


class _Bin:
    """The BIN chunk: 4-byte aligned views, as glTF asks."""

    def __init__(self):
        self.parts, self.size, self.views = [], 0, []

    def view(self, data, target=None):
        pad = (-self.size) % 4
        if pad:
            self.parts.append(b"\0" * pad)
            self.size += pad
        v = {"buffer": 0, "byteOffset": self.size, "byteLength": len(data)}
        if target is not None:
            v["target"] = target
        self.views.append(v)
        self.parts.append(data)
        self.size += len(data)
        return len(self.views) - 1

    def bytes(self):
        out = b"".join(self.parts)
        return out + b"\0" * ((-len(out)) % 4)


def write(path, model, V, N, name=None, extras=None, keep=None):
    """Write the model with positions V and normals N (model space, as `Model.deform` returns them) to `path`. `keep`
    (t,) bool picks the triangles to write (default all); only the vertices they use are written, so the bounds a
    viewer frames are theirs."""
    keep = np.ones(len(model.T), bool) if keep is None else np.asarray(keep, bool).copy()
    keep &= ~model.hidden[model.tri_mat]
    used = np.unique(model.T[keep])
    remap = np.full(len(V), -1, np.int64)
    remap[used] = np.arange(len(used))
    V = np.asarray(V, float)[used] @ AXES.T
    N = np.asarray(N, float)[used] @ AXES.T
    binary, accessors = _Bin(), []

    def accessor(arr, kind, target, ctype=_FLOAT, minmax=False):
        a = {"bufferView": binary.view(np.ascontiguousarray(arr).tobytes(), target), "componentType": ctype,
             "count": int(len(arr)), "type": kind}
        if minmax:
            a["min"], a["max"] = [float(x) for x in arr.min(axis=0)], [float(x) for x in arr.max(axis=0)]
        accessors.append(a)
        return len(accessors) - 1

    if not len(used):
        raise ValueError(f"{model.label}: no drawn triangles to write")
    pos = accessor(V.astype(np.float32), "VEC3", _ARRAY, minmax=True)
    nrm = accessor(N.astype(np.float32), "VEC3", _ARRAY)
    tex = accessor(np.asarray(model.uv, np.float32)[used], "VEC2", _ARRAY)
    images, textures, by_path, materials, prims = [], [], {}, [], []
    for m in range(len(model.mat_names)):
        tris = remap[model.T[(model.tri_mat == m) & keep]]
        if not len(tris):
            continue
        rgba = [float(x) for x in np.clip(model.rgba[m], 0.0, 1.0)]
        pbr = {"metallicFactor": 0.0, "roughnessFactor": 0.85}
        see = rgba[3] < 0.999
        path_m = model.textures[m]
        if path_m is not None and Path(path_m).is_file():
            if path_m not in by_path:
                data, img_see = _image(path_m)
                images.append({"bufferView": binary.view(data), "mimeType": "image/png", "name": Path(path_m).name})
                textures.append({"sampler": 0, "source": len(images) - 1})
                by_path[path_m] = (len(textures) - 1, img_see)
            ti, img_see = by_path[path_m]
            pbr["baseColorTexture"] = {"index": ti}
            rgb = np.asarray(rgba[:3])                              # MMD lights a texture by its diffuse: keep the
            tint = rgb / rgb.max() if rgb.max() > 1e-6 else np.ones(3)  # hue at full brightness, white stays white
            pbr["baseColorFactor"] = [*(float(x) for x in tint), rgba[3]]
            see = see or img_see
        else:
            pbr["baseColorFactor"] = rgba
        materials.append({"name": model.mat_names[m], "pbrMetallicRoughness": pbr,
                          "alphaMode": "BLEND" if see else "OPAQUE", "doubleSided": bool(model.double_sided[m])})
        idx = accessor(np.asarray(tris[:, [0, 2, 1]], np.uint32).reshape(-1), "SCALAR", _ELEMENTS, ctype=_UINT)
        prims.append({"attributes": {"POSITION": pos, "NORMAL": nrm, "TEXCOORD_0": tex}, "indices": idx,
                      "material": len(materials) - 1})
    title = name or model.names[0] or model.label
    doc = {"asset": {"version": "2.0", "generator": "mk-mmd model glb"}, "scene": 0,
           "scenes": [{"name": title, "nodes": [0]}],
           "nodes": [{"name": title, "mesh": 0, **({"extras": extras} if extras else {})}],
           "meshes": [{"name": title, "primitives": prims}], "materials": materials, "accessors": accessors}
    if images:
        doc.update(images=images, textures=textures,
                   samplers=[{"magFilter": 9729, "minFilter": 9987, "wrapS": 10497, "wrapT": 10497}])
    blob = binary.bytes()
    doc["buffers"] = [{"byteLength": len(blob)}]
    doc["bufferViews"] = binary.views
    js = json.dumps(doc, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    js += b" " * ((-len(js)) % 4)
    out = struct.pack("<4sII", b"glTF", 2, 12 + 8 + len(js) + 8 + len(blob))
    out += struct.pack("<I4s", len(js), b"JSON") + js + struct.pack("<I4s", len(blob), b"BIN\0") + blob
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_bytes(out)
    return {"vertices": int(len(V)), "triangles": int(sum(a["count"] for a in accessors[3:] if a["type"] == "SCALAR")) // 3,
            "materials": len(materials), "images": len(images), "bytes": len(out)}
