"""Built parts kept between builds: `mk model build` and `mk model lab` reuse a part whose inputs did not change
(`--no-cache` builds every part again).

A part's inputs are the code that can have made it (the modules of the mkmmd package loaded when it was built, by size
and time: a builder's module is loaded when its part is first built, so a change to the hair's code builds the hair
again and never the body or the head), its builder (module, name and file), the model seed, the parts built before it
(their inputs, chained: a part built again builds every part after it again)
and what it read from the spec while it was built (`spec.Reads`): the values of the keys it read (a value naming a file
counts with that file's size and time, a model base's meshes so), the keys it looked up that were not set, the key
sets of the tables it listed.

An entry holds the Part, the textures it wrote and its log lines (its warnings come back with it): up to KEEP per part and
chain, in <cache>/model_parts/<chain>/<reads digest>.{json,pkl}; chains unused for STALE_DAYS are removed, and the least
recently used builds once the cache passes MAX_BYTES."""
import hashlib
import inspect
import json
import os
import pickle
import shutil
import sys
import time
from pathlib import Path

from .. import config as CFG
from . import spec as SP

KEEP = 6
STALE_DAYS = 14
MAX_BYTES = 2 * 1024 ** 3
PACKAGE = Path(__file__).resolve().parents[1]


def stamp(path):
    """[size, time] of a file, None when it is not there."""
    try:
        st = Path(path).stat()
    except OSError:
        return None
    return [st.st_size, st.st_mtime_ns]


def loaded():
    """{file: [size, time]} of every module of the mkmmd package loaded now (files inside the package by their path in
    it): the code a part built by now can have run, and with it the data files of the model bases (their meshes, which
    a builder may read without the spec naming them; the bases' .toml files are the spec, which `Reads` follows). The
    command line's own modules (mkmmd.cli) are left out: they start a build and make no part."""
    out = {}
    for name, mod in list(sys.modules.items()):
        f = getattr(mod, "__file__", None)
        if f and (name == "mkmmd" or name.startswith("mkmmd.")) and not name.startswith("mkmmd.cli"):
            p = Path(f).resolve()
            s = stamp(p)
            if s is not None:
                out[str(p.relative_to(PACKAGE)) if p.is_relative_to(PACKAGE) else str(p)] = s
    for p in (PACKAGE / "model" / "bases").rglob("*"):
        if p.is_file() and p.suffix not in (".py", ".pyc", ".toml"):
            out[str(p.relative_to(PACKAGE))] = stamp(p)
    return out


def unchanged(code):
    """Whether every file of `code` (`loaded`'s) is as it was."""
    return all(stamp(PACKAGE / f) == s for f, s in code.items())


def builder_id(fn):
    """A builder in the cache key: its module and name, and the size and time of its file (a project's own builder,
    `[model.builders]`, is outside the package; a test's stand-in never passes for the real one)."""
    try:
        f = Path(inspect.getsourcefile(fn) or "")
        st = f.stat() if f.is_file() else None
    except (TypeError, OSError):
        st = None
    return [getattr(fn, "__module__", ""), getattr(fn, "__qualname__", repr(fn)), [st.st_size, st.st_mtime_ns] if st else None]


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

    def chain(self, name, seed, prior, fn=None):
        """The folder of part `name` built by `fn` after the parts whose cache keys are `prior`, with this seed."""
        ident = json.dumps([name, int(seed), builder_id(fn) if fn is not None else None, list(prior)])
        return self.root / hashlib.sha1(ident.encode()).hexdigest()[:24]

    def find(self, spec, chain):
        """(key, entry) of a build in `chain` whose reads look the same in `spec` now and whose code is unchanged, else
        None. entry: part, textures {file name: bytes}, logs, reads (seen, missed, listed)."""
        if not chain.is_dir():
            return None
        for meta in sorted(chain.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
            try:
                r = json.loads(meta.read_text(encoding="utf-8"))
                reads = tuple({tuple(p) for p in r[k]} for k in ("seen", "missed", "listed"))
                if reads_digest(spec, *reads) != r["reads"] or not unchanged(r["code"]):
                    continue
                with open(meta.with_suffix(".pkl"), "rb") as fh:
                    entry = pickle.load(fh)
            except (OSError, ValueError, KeyError, EOFError, pickle.UnpicklingError, AttributeError, ImportError):
                continue
            os.utime(meta)                                   # the most recently used is tried first
            entry["reads"] = reads
            return f"{chain.name}:{meta.stem}", entry
        return None

    def store(self, spec, chain, reads, part, textures, logs):
        """Keep a build in `chain` (reads: `spec.Reads.since`) with the code loaded now; returns its key (the reads and
        the code in it: the parts after it are built again when either changes). A part that does not pickle is not
        kept (its key still chains the parts after it)."""
        digest, code = reads_digest(spec, *reads), loaded()
        stem = hashlib.sha1(json.dumps([digest, sorted(code.items())]).encode()).hexdigest()[:24]
        key = f"{chain.name}:{stem}"
        try:
            chain.mkdir(parents=True, exist_ok=True)
            tmp = chain / f"{stem}.pkl.tmp"
            with open(tmp, "wb") as fh:
                pickle.dump({"part": part, "textures": textures, "logs": list(logs)}, fh, protocol=pickle.HIGHEST_PROTOCOL)
            tmp.replace(chain / f"{stem}.pkl")
            tmp = chain / f"{stem}.json.tmp"
            tmp.write_text(json.dumps({**{k: sorted(map(list, s)) for k, s in zip(("seen", "missed", "listed"), reads)},
                                       "reads": digest, "code": code}), encoding="utf-8")
            tmp.replace(chain / f"{stem}.json")
        except (OSError, pickle.PicklingError, TypeError, AttributeError):
            return key
        for old in sorted(chain.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)[KEEP:]:
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
