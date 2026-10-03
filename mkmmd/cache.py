"""Result cache keyed by a hash of the inputs (see docs/design.md: Cache)."""
import hashlib
import json
from pathlib import Path

import numpy as np

_BIG = 64 * 1024 * 1024


def fingerprint(path):
    """Content hash of a file (head + tail + size for files over 64 MB)."""
    path = Path(path)
    st = path.stat()
    h = hashlib.sha1()
    with open(path, "rb") as fh:
        if st.st_size <= _BIG:
            for block in iter(lambda: fh.read(1 << 20), b""):
                h.update(block)
        else:
            h.update(fh.read(1 << 20))
            fh.seek(-(1 << 20), 2)
            h.update(fh.read(1 << 20))
            h.update(str(st.st_size).encode())
    return h.hexdigest()


def key(*parts):
    """SHA-256 over JSON-serialisable parts; Path objects are replaced by their content fingerprints."""
    def norm(x):
        if isinstance(x, Path):
            return {"file": x.name, "sha1": fingerprint(x)}
        if isinstance(x, dict):
            return {str(k): norm(v) for k, v in sorted(x.items())}
        if isinstance(x, (list, tuple)):
            return [norm(v) for v in x]
        if hasattr(x, "tolist"):
            return x.tolist()
        return x
    blob = json.dumps([norm(p) for p in parts], sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:32]


class Cache:
    def __init__(self, root):
        self.root = Path(root)

    def path(self, op, k, ext):
        d = self.root / op
        d.mkdir(parents=True, exist_ok=True)
        return d / f"{k}.{ext}"

    def load_json(self, op, k):
        p = self.path(op, k, "json")
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None

    def save_json(self, op, k, data):
        p = self.path(op, k, "json")
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        tmp.replace(p)
        return p

    def load_npz(self, op, k):
        p = self.path(op, k, "npz")
        if not p.exists():
            return None
        with np.load(p, allow_pickle=False) as z:
            return {name: z[name] for name in z.files}

    def save_npz(self, op, k, arrays):
        p = self.path(op, k, "npz")
        tmp = p.with_name(p.stem + ".tmp.npz")
        np.savez_compressed(tmp, **arrays)
        tmp.replace(p)
        return p
