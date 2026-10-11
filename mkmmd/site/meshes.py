"""The meshes the site's server cuts: a .glb read back as world-space triangles in glTF's axes (+Y up), exactly as the
page's viewer draws it, so a section the server computes lies on what the person sees."""
import json
import struct
from pathlib import Path

import numpy as np

COMP = {5120: np.int8, 5121: np.uint8, 5122: np.int16, 5123: np.uint16, 5125: np.uint32, 5126: np.float32}
SIZE = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}


def _trs(n):
    if "matrix" in n:
        return np.asarray(n["matrix"], float).reshape(4, 4).T
    x, y, z, w = n.get("rotation", [0.0, 0.0, 0.0, 1.0])
    R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                  [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                  [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
    M = np.eye(4)
    M[:3, :3] = R * np.asarray(n.get("scale", [1.0, 1.0, 1.0]), float)
    M[:3, 3] = n.get("translation", [0.0, 0.0, 0.0])
    return M


def read_glb(path):
    """(V (n, 3), T (m, 3)): every triangle primitive of the .glb's scene, its node's transform applied."""
    data = Path(path).read_bytes()
    magic, _version, _length = struct.unpack_from("<III", data, 0)
    if magic != 0x46546C67:
        raise ValueError(f"{path}: not a .glb")
    n = struct.unpack_from("<I", data, 12)[0]
    doc = json.loads(data[20:20 + n])
    off = 20 + n
    binary = data[off + 8:off + 8 + struct.unpack_from("<I", data, off)[0]] if off + 8 <= len(data) else b""

    def accessor(i):
        a = doc["accessors"][i]
        bv = doc["bufferViews"][a["bufferView"]]
        dt, k = np.dtype(COMP[a["componentType"]]), SIZE[a["type"]]
        stride = bv.get("byteStride") or dt.itemsize * k
        arr = np.ndarray((a["count"], k), dtype=dt, buffer=binary, offset=bv.get("byteOffset", 0) + a.get("byteOffset", 0),
                         strides=(stride, dt.itemsize))
        out = arr.astype(np.float64)
        if a.get("normalized") and dt.kind in "iu":
            out /= np.iinfo(dt).max
        return out

    Vs, Ts, count = [], [], 0
    cache = {}

    def walk(ni, parent):
        node = doc["nodes"][ni]
        M = parent @ _trs(node)
        if "mesh" in node:
            for prim in doc["meshes"][node["mesh"]]["primitives"]:
                if prim.get("mode", 4) != 4:
                    continue
                key = (ni, prim["attributes"]["POSITION"])
                if key not in cache:
                    P = accessor(prim["attributes"]["POSITION"])
                    nonlocal count
                    cache[key] = count
                    Vs.append(P @ M[:3, :3].T + M[:3, 3])
                    count += len(P)
                base = cache[key]
                idx = (accessor(prim["indices"]).astype(np.int64).reshape(-1, 3) if "indices" in prim
                       else np.arange(doc["accessors"][prim["attributes"]["POSITION"]]["count"]).reshape(-1, 3))
                if np.linalg.det(M[:3, :3]) < 0:
                    idx = idx[:, [0, 2, 1]]
                Ts.append(idx + base)
        for c in node.get("children", []):
            walk(c, M)

    for ni in doc["scenes"][doc.get("scene", 0)]["nodes"]:
        walk(ni, np.eye(4))
    if not Ts:
        raise ValueError(f"{path}: no triangles")
    return np.concatenate(Vs), np.concatenate(Ts)
