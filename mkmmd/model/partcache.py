"""Built parts kept between builds: `mk model build` and `mk model lab` reuse a part whose inputs did not change
(`--no-cache` builds every part again).

A part's inputs are the code (every .py file of the mkmmd package and every file of the model bases, by size and time),
the model seed, the parts built before it (their inputs, chained: a part built again builds every part after it again)
and what it read from the spec while it was built (`spec.Reads`): the values of the keys it read (a value naming a file
counts with that file's size and time), the keys it looked up that were not set, the key sets of the tables it listed.
A hair change so builds the hair and what comes after it again, never the body or the head.

An entry holds the Part, the textures it wrote and its log lines (its warnings come back with it): up to KEEP per part and
chain, in <cache>/model_parts/<chain>/<reads digest>.{json,pkl}; chains unused for STALE_DAYS are removed, and the least
recently used builds once the cache passes MAX_BYTES."""
import hashlib
import json
import os
import pickle
import shutil
import time
from pathlib import Path

from .. import config as CFG
from . import spec as SP

KEEP = 6
STALE_DAYS = 14
MAX_BYTES = 2 * 1024 ** 3
_CODE = {}


def code_digest():
    """Size and time of every .py file of the mkmmd package and of every file of the model bases (once per process)."""
    if "d" not in _CODE:
        root = Path(__file__).resolve().parents[1]
        files = [p for p in root.rglob("*.py")] + [p for p in (root / "model" / "bases").rglob("*") if p.is_file()]
        h = hashlib.sha1()
        for p in sorted(f for f in files if "__pycache__" not in f.parts):
            st = p.stat()
            h.update(f"{p.relative_to(root)}:{st.st_size}:{st.st_mtime_ns}\n".encode())
        _CODE["d"] = h.hexdigest()
    return _CODE["d"]


def _norm(v, base, top=True):
    """A read value as JSON for the digest: a spec table as "<table>" (the keys read from it count on their own), a
    string that names an existing file with the file's size and time."""
    if isinstance(v, dict):
        return "<table>" if top else {str(k): _norm(x, base, False) for k, x in sorted(dict.items(v))}
    if isinstance(v, (list, tuple)):
        return [_norm(x, base, False) for x in v]
    if isinstance(v, str):
        try:
            f = Path(SP.expand(v, base))
            if f.is_file():
                st = f.stat()
                return [v, st.st_size, st.st_mtime_ns]
        except (OSError, ValueError, SP.SpecError):
            pass
    return v


def reads_digest(spec, seen, missed, listed):
    """Digest of what a part read (`spec.Reads.since`), looked up in `spec` now."""
    base = getattr(spec, "dir", Path("."))
    rows = []
    for p in sorted(seen):
        ok, v = SP.at(spec, p)
        rows.append(["seen", list(p), _norm(v, base) if ok else "<unset>"])
    for p in sorted(missed):
        rows.append(["missed", list(p), SP.at(spec, p)[0]])
    for p in sorted(listed):
        ok, v = SP.at(spec, p)
        rows.append(["listed", list(p), sorted(dict.keys(v)) if ok and isinstance(v, dict) else "<unset>"])
    return hashlib.sha1(json.dumps(rows, default=str).encode("utf-8")).hexdigest()[:24]


class PartCache:
    def __init__(self, root=None):
        self.root = Path(root) if root else CFG.cache_dir() / "model_parts"

    def chain(self, name, seed, prior):
        """The folder of part `name` built after the parts whose cache keys are `prior`, with this code and seed."""
        return self.root / hashlib.sha1(json.dumps([name, int(seed), code_digest(), list(prior)]).encode()).hexdigest()[:24]

    def find(self, spec, name, seed, prior):
        """(key, entry) of a build of `name` after `prior` whose reads look the same in `spec` now, else None. entry:
        part, textures {file name: bytes}, logs, reads (seen, missed, listed)."""
        d = self.chain(name, seed, prior)
        if not d.is_dir():
            return None
        for meta in sorted(d.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
            try:
                r = json.loads(meta.read_text(encoding="utf-8"))
                reads = tuple({tuple(p) for p in r[k]} for k in ("seen", "missed", "listed"))
                if reads_digest(spec, *reads) != meta.stem:
                    continue
                with open(meta.with_suffix(".pkl"), "rb") as fh:
                    entry = pickle.load(fh)
            except (OSError, ValueError, KeyError, EOFError, pickle.UnpicklingError, AttributeError, ImportError):
                continue
            os.utime(meta)                                   # the most recently used is tried first
            entry["reads"] = reads
            return f"{d.name}:{meta.stem}", entry
        return None

    def key(self, spec, name, seed, prior, reads):
        return f"{self.chain(name, seed, prior).name}:{reads_digest(spec, *reads)}"

    def store(self, spec, name, seed, prior, reads, part, textures, logs):
        """Keep a build of `name` (reads: `spec.Reads.since`); returns its key. A part that does not pickle is not kept
        (its key still chains the parts after it)."""
        key = self.key(spec, name, seed, prior, reads)
        d = self.chain(name, seed, prior)
        stem = key.split(":")[1]
        try:
            d.mkdir(parents=True, exist_ok=True)
            tmp = d / f"{stem}.pkl.tmp"
            with open(tmp, "wb") as fh:
                pickle.dump({"part": part, "textures": textures, "logs": list(logs)}, fh, protocol=pickle.HIGHEST_PROTOCOL)
            tmp.replace(d / f"{stem}.pkl")
            tmp = d / f"{stem}.json.tmp"
            tmp.write_text(json.dumps({k: sorted(map(list, s)) for k, s in zip(("seen", "missed", "listed"), reads)}),
                           encoding="utf-8")
            tmp.replace(d / f"{stem}.json")
        except (OSError, pickle.PicklingError, TypeError, AttributeError):
            return key
        for old in sorted(d.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)[KEEP:]:
            old.unlink(missing_ok=True)
            old.with_suffix(".pkl").unlink(missing_ok=True)
        self.prune()
        return key

    def prune(self):
        """Remove the chains nothing used for STALE_DAYS (old code, old specs), then the least recently used builds
        until the cache holds at most MAX_BYTES."""
        cut = time.time() - STALE_DAYS * 86400
        for d in self.root.iterdir():
            if d.is_dir() and all(p.stat().st_mtime < cut for p in d.glob("*.json")):
                shutil.rmtree(d, ignore_errors=True)
        builds = sorted(((m.stat().st_mtime, m) for m in self.root.glob("*/*.json")), key=lambda t: t[0])
        sizes = {m: sum(f.stat().st_size for f in (m, m.with_suffix(".pkl")) if f.exists()) for _, m in builds}
        total = sum(sizes.values())
        for _, m in builds:
            if total <= MAX_BYTES:
                break
            total -= sizes[m]
            m.unlink(missing_ok=True)
            m.with_suffix(".pkl").unlink(missing_ok=True)
